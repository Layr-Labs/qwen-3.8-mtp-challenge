// Port of mlx_lm.generate.GenerationBatch.
// https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
//
// MTP (Multi-Token Prediction) dispatch is added at the bottom of this file
// as a private extension. It intercepts init/next/filter/extend when the
// model is MTPCapable and batchSize == 1.
// Port of omlx commit 696d90a: patches/mlx_lm_mtp/batch_generator.py

import Foundation
import MLX
import MLXRandom
import MLXNN

/// Picks one token per row from a `[B, vocab]` logits tensor.
public typealias RowSampler = @Sendable (MLXArray) -> MLXArray

/// Deterministic greedy sampler.
@Sendable public func greedySampler(_ logprobs: MLXArray) -> MLXArray {
    argMax(logprobs, axis: -1)
}

/// Per-row response from a single decode step.
///
/// Marked `@unchecked Sendable` because `promptCache` carries non-Sendable
/// `KVCacheSimple` references for cross-request prefix caching; cross-actor
/// transfer is the caller's responsibility.
public struct GenerationBatchResponse: @unchecked Sendable {
    public let uid: Int
    public let token: Int

    /// `"length"`, `"stop"`, or nil if the row is still generating.
    public let finishReason: String?

    /// The matched stop sequence if a multi-token stop completed on this token.
    public let matchedSequence: [Int]?

    /// State machine state name after this token's transition (nil = terminated).
    public let currentState: String?

    /// All produced tokens for this row. Set only on the final response.
    public let allTokens: [Int]?

    /// Single-row prompt cache for prefix caching across requests.
    /// Set only on the final response.
    public let promptCache: [any KVCache]?
}

/// Decode-phase batch over a shared `[any BatchedCache]` (one per layer).
/// Each layer's cache is the appropriate batched type for that layer:
/// `BatchKVCache` for full attention, `ArraysCache`/`MambaCache` for SSM.
/// Construct after prefill has populated the caches; call `next()` to
/// drive generation one step at a time.
public final class GenerationBatch: @unchecked Sendable {

    public let model: any LanguageModel
    public private(set) var uids: [Int]
    public private(set) var promptCache: [any BatchedCache]
    public private(set) var tokens: [[Int]]
    public private(set) var maxTokens: [Int]

    public private(set) var samplers: [RowSampler?]
    public let fallbackSampler: RowSampler
    public private(set) var stateMachines: [SequenceStateMachine]

    /// When true, `step()` skips the per-step `logSumExp` normalization of the
    /// decode logits and hands the raw logits to the row samplers. This is a
    /// pure host/GPU-overhead optimization and is **only** safe when no active
    /// row needs the normalized logprobs: temperature scaling, top-k, min-p,
    /// categorical and argMax are all invariant to the per-row constant
    /// `logSumExp` shift, but top-p (nucleus) sampling's cumulative-probability
    /// threshold is not, and the repetition/presence/frequency penalty sampler
    /// applies its (non-shift-invariant) transform to the values it receives.
    ///
    /// The `Scheduler` sets this from the admitted rows' sampling params (it is
    /// the only place that knows them); it defaults to `false` so any batch that
    /// is never told otherwise keeps the exact previous behavior. `extend`
    /// combines with logical AND so a merged batch normalizes if *either* side
    /// needed it.
    public var skipLogprobNormalization: Bool = false

    /// When false, finished rows skip the `extractBatched` dequant-and-copy of
    /// their prompt cache (only a prefix cache consumes it). The scheduler sets
    /// this true only when a prefix cache is active, so runs without prefix
    /// caching — e.g. KV-quant models whose prefix tier isn't the in-GPU block
    /// cache — don't pay a per-finish fp16 copy of the KV history.
    public var capturePromptCacheOnFinish: Bool = false

    /// Tokens queued for the next model call. At construction this is the
    /// final prompt token for each row. After priming, and after every
    /// decode step, it holds the sampled token that should be returned on
    /// the next `next()` call. `[B]`.
    private var nextTokens: MLXArray
    private var numTokens: [Int]
    private var matcherStates: [SequenceStateMachineState]

    /// MTP state. Non-nil when the batch is running the MTP draft+verify cycle.
    /// Set by `postInitMTP` immediately after `init`; cleared on finish or fallback.
    /// Port of omlx commit 696d90a: batch_generator.py _omlx_mtp_state
    internal var _omlxMtpState: MTPState?

    /// Optional drafter-based MTP runtime (Gemma 4). When set and the batch is
    /// eligible (size ≤ `maxBatch`, all rows greedy), `next()` speculates one
    /// MTP round per call instead of a single decode step. Injected by the
    /// `Scheduler`; nil = plain decode (the default).
    internal let mtpRuntime: (any BatchedMTPRuntime)?
    /// Live drafter-MTP session over the current rows. Non-nil once entered;
    /// persists across `next()` calls; compacted as rows finish; cleared when
    /// the batch empties. While non-nil the `Scheduler` must not `extend()`
    /// this batch (the session carries ragged per-row pending tokens that the
    /// shared-index cache can't absorb mid-stream — see BatchedMTP.swift).
    private var mtpSession: (any BatchedMTPSession)?

    /// True while a drafter-MTP session is active. The scheduler gates
    /// admission on this (no mid-session `extend`).
    public var hasActiveMTPSession: Bool { mtpSession != nil }

    /// Compiled forward closure for B=1 standard decode. When non-nil,
    /// `step()` uses this + `_compiledCache` instead of the batched
    /// `promptCache` for the model forward pass. Built after the init's
    /// priming step for non-MTP B=1 batches when `DARKBLOOM_COMPILED_DECODE=1`.
    ///
    /// This collapses hundreds of per-layer FFI crossings into a single
    /// compiled call per decode step, significantly improving single-stream
    /// TPS on Apple Silicon.
    private var _compiledForward: (@Sendable ([MLXArray]) -> [MLXArray])?

    /// Single-stream compilable caches used by the compiled decode path.
    /// Contains `CompilableKVCache` / `CompilableRotatingKVCache` instances
    /// extracted from the batched `promptCache` at setup time. Non-nil iff
    /// `_compiledForward` is non-nil.
    private var _compiledCache: [any KVCache]?

    public init(
        model: any LanguageModel,
        uids: [Int],
        seedTokens: MLXArray,
        promptCache: [any BatchedCache],
        tokens: [[Int]],
        maxTokens: [Int],
        samplers: [RowSampler?]? = nil,
        fallbackSampler: @escaping RowSampler = greedySampler,
        stateMachines: [SequenceStateMachine]? = nil,
        mtpRuntime: (any BatchedMTPRuntime)? = nil
    ) {
        self.mtpRuntime = mtpRuntime
        precondition(uids.count == tokens.count, "uids/tokens count mismatch")
        precondition(uids.count == maxTokens.count, "uids/max_tokens count mismatch")
        self.model = model
        self.uids = uids
        self.promptCache = promptCache
        self.tokens = tokens
        self.maxTokens = maxTokens
        self.samplers = samplers ?? Array(repeating: nil, count: uids.count)
        self.fallbackSampler = fallbackSampler
        let machines = stateMachines ?? Array(repeating: SequenceStateMachine(), count: uids.count)
        self.stateMachines = machines
        self.matcherStates = machines.map { $0.makeState() }
        self.numTokens = Array(repeating: 0, count: uids.count)
        self.nextTokens = seedTokens

        // Match upstream mlx_lm.GenerationBatch: immediately run one
        // decode step in the constructor so the first call to `next()`
        // returns an already-computed token while scheduling the following
        // token. This double-buffer keeps the GPU queue ahead of the CPU
        // token extraction path.
        if !uids.isEmpty {
            _ = step()
            // Attempt MTP post-init for eligible single-sequence batches.
            // omlx: batch_generator.py patched_init → _post_init_mtp
            if batchSize == 1, let mtpModel = model as? (any MTPCapable), mtpModel.hasMTPHead {
                postInitMTP(model: mtpModel)
            }
            // Compiled decode for non-MTP B=1 standard decode. MTP paths
            // bypass step() and call the model directly with promptCache,
            // so compiled decode is incompatible with active MTP.
            if _omlxMtpState == nil && mtpRuntime == nil {
                setupCompiledDecodeIfEligible()
            }
        }
    }

    /// Run one decode step. Finished rows (length / stop) are filtered out
    /// of the active set after this call; their final responses appear with
    /// non-nil `finishReason`.
    public func next() -> [GenerationBatchResponse] {
        if uids.isEmpty { return [] }

        // MTP dispatch: for eligible single-sequence batches, emit from queue or
        // run the 2-token verify cycle.
        // omlx: batch_generator.py patched_next → _mtp_next
        if batchSize == 1,
           let state = _omlxMtpState,
           let mtpModel = model as? (any MTPCapable)
        {
            do {
                return try mtpNext(state: state, model: mtpModel)
            } catch {
                // Fall back to standard step; drop state to prevent half-built cycles.
                _omlxMtpState = nil
            }
        }

        // Drafter-based MTP (Gemma 4): one speculative round per call when a
        // runtime is attached and the batch is eligible (size ≤ maxBatch, all
        // greedy). A session, once entered, persists until the batch empties.
        if let runtime = mtpRuntime, mtpEligible(runtime) {
            if let responses = gemma4MtpNext(runtime: runtime) {
                return responses
            }
            // beginSession returned nil (model not MTP-capable): fall through
            // to plain decode and never retry MTP for this batch.
        }

        let stepTokens = step()

        var keep: [Int] = []
        var responses: [GenerationBatchResponse] = []
        responses.reserveCapacity(uids.count)

        for i in 0 ..< uids.count {
            numTokens[i] += 1

            var finishReason: String? = nil
            if numTokens[i] >= maxTokens[i] {
                finishReason = "length"
            }

            let machine = stateMachines[i]
            let (nextState, matchedSequence, currentState) =
                machine.match(matcherStates[i], stepTokens[i])
            matcherStates[i] = nextState
            if matchedSequence != nil, currentState == nil {
                finishReason = "stop"
            }

            if finishReason != nil {
                let extracted: [any KVCache]?
                if capturePromptCacheOnFinish {
                    if let compiledCache = _compiledCache {
                        // Compiled decode (B=1): the live KV state lives in
                        // _compiledCache; the batched promptCache is stale.
                        // Snapshot each compiled cache into its standard KVCache
                        // form so a prefix-cache consumer never reads a compiled
                        // cache's inherited view directly. CompilableRotatingKVCache
                        // (sliding window) deliberately leaves the inherited
                        // idx/offset/state stale during compiled decode; returning
                        // it raw exposed that stale view. See compiledCaptureSnapshot.
                        extracted = compiledCache.map { Self.compiledCaptureSnapshot($0) }
                    } else {
                        extracted = promptCache.map { $0.extractBatched(i) }
                    }
                } else {
                    extracted = nil
                }
                responses.append(
                    GenerationBatchResponse(
                        uid: uids[i],
                        token: stepTokens[i],
                        finishReason: finishReason,
                        matchedSequence: matchedSequence,
                        currentState: currentState,
                        allTokens: tokens[i],
                        promptCache: extracted
                    ))
            } else {
                keep.append(i)
                responses.append(
                    GenerationBatchResponse(
                        uid: uids[i],
                        token: stepTokens[i],
                        finishReason: nil,
                        matchedSequence: matchedSequence,
                        currentState: currentState,
                        allTokens: nil,
                        promptCache: nil
                    ))
            }
        }

        if keep.count < uids.count {
            filter(keep: keep)
        }

        return responses
    }

    /// In-place keep only the rows at the given indices.
    public func filter(keep: [Int]) {
        let keepArr = MLXArray(keep.map { Int32($0) })

        if keep.isEmpty {
            promptCache.removeAll()
        } else {
            for cache in promptCache {
                cache.filterBatched(batchIndices: keepArr)
            }
        }

        // Invalidate compiled decode whenever batch size changes. For B=1
        // this fires on empty (the only transition). For B>1 it also fires
        // when rows are evicted, because the compiled trace was captured for
        // the old batch shape. _compiledCache (B=1 single-stream caches)
        // is released to free stale GPU buffers. For B>1, the promoted
        // compilable caches live in promptCache and will be used uncompiled.
        if keep.count < uids.count {
            _compiledForward = nil
            _compiledCache = nil
        }

        // Drafter-MTP: keep the session carry aligned with the surviving rows.
        // `filter` is the single eviction choke point (round emission, scheduler
        // abort, external shrink all route through here), so maintaining the
        // session invariant here — rather than only in the round emitter —
        // keeps `mtpSession.activeCount == uids.count` and clears the session
        // when the batch empties (preventing a stale non-nil session that would
        // wedge admission). `keep` indexes the pre-filter rows, matching the
        // session's current row order. Done before the metadata remap below.
        if let session = mtpSession {
            if keep.isEmpty {
                mtpSession = nil
            } else if keep.count < session.activeCount {
                session.compactCarry(keepLocal: keep)
            }
        }

        uids = keep.map { uids[$0] }
        tokens = keep.map { tokens[$0] }
        samplers = keep.map { samplers[$0] }
        maxTokens = keep.map { maxTokens[$0] }
        stateMachines = keep.map { stateMachines[$0] }
        matcherStates = keep.map { matcherStates[$0] }
        numTokens = keep.map { numTokens[$0] }
        if !keep.isEmpty {
            nextTokens = take(nextTokens, keepArr, axis: 0)
        }
        // MTP: drop state when batch is emptied externally (e.g. scheduler abort).
        // omlx: batch_generator.py patched_filter
        if keep.isEmpty { _omlxMtpState = nil }
    }

    /// In-place: append `other`'s rows to this batch. Per-layer caches are
    /// concatenated via `BatchedCache.extendBatched`.
    public func extend(_ other: GenerationBatch) {
        precondition(
            promptCache.count == other.promptCache.count,
            "Cannot extend with a batch that has a different layer count"
        )
        // Compiled decode: sync compiled cache state back to promptCache
        // BEFORE extendBatched() so the merge works with up-to-date KV
        // state. Extending changes batch size, invalidating the compiled
        // trace (shape mismatch on the next step).
        if let compiledCache = _compiledCache {
            for (i, cc) in compiledCache.enumerated() where i < promptCache.count {
                // B>1: promoted batch caches live in promptCache; sync counters.
                if let syncable = cc as? CompilableBatchKVCache {
                    syncable.syncFromCompiled()
                } else if let syncable = cc as? CompilableBatchRotatingKVCache {
                    syncable.syncFromCompiled()
                }
                // B=1: compiled caches are separate single-stream instances.
                // Copy the valid KV data back into the batched promptCache
                // so extendBatched merges with current (not stale) state.
                else if let compiled = cc as? CompilableKVCache,
                        let batch = promptCache[i] as? BatchKVCache {
                    let validState = compiled.state
                    if validState.count == 2 {
                        let off = compiled.offset
                        batch.keys = validState[0]
                        batch.values = validState[1]
                        batch._idx = off
                        batch.batchOffset = MLXArray([Int32(off)])
                    }
                } else if let compiled = cc as? CompilableRotatingKVCache,
                          let batch = promptCache[i] as? BatchRotatingKVCache {
                    eval(compiled.idxArray, compiled.offsetArray)
                    let physIdx = Int(compiled.idxArray[0].item(Int32.self))
                    let totalOffset = Int(compiled.offsetArray[0].item(Int32.self))
                    let validLen = min(totalOffset, compiled.maxCacheSize)
                    if let k = compiled.keys, let v = compiled.values {
                        if totalOffset > compiled.maxCacheSize, physIdx != 0 {
                            // Unwind ring to temporal order.
                            batch.keys = concatenated([
                                k[.ellipsis, physIdx..., 0...],
                                k[.ellipsis, ..<physIdx, 0...],
                            ], axis: 2)
                            batch.values = concatenated([
                                v[.ellipsis, physIdx..., 0...],
                                v[.ellipsis, ..<physIdx, 0...],
                            ], axis: 2)
                        } else {
                            batch.keys = k[.ellipsis, ..<validLen, 0...]
                            batch.values = v[.ellipsis, ..<validLen, 0...]
                        }
                        batch._idx = validLen
                        batch._base = 0
                        batch.batchOffset = MLXArray([Int32(totalOffset)])
                    }
                }
            }
        }
        for (a, b) in zip(promptCache, other.promptCache) {
            a.extendBatched(b)
        }
        uids.append(contentsOf: other.uids)
        tokens.append(contentsOf: other.tokens)
        samplers.append(contentsOf: other.samplers)
        maxTokens.append(contentsOf: other.maxTokens)
        stateMachines.append(contentsOf: other.stateMachines)
        matcherStates.append(contentsOf: other.matcherStates)
        numTokens.append(contentsOf: other.numTokens)
        nextTokens = concatenated([nextTokens, other.nextTokens], axis: 0)
        // Only keep skipping normalization if BOTH batches were safe to skip;
        // a single incoming top-p / penalty row forces normalization again.
        skipLogprobNormalization = skipLogprobNormalization && other.skipLogprobNormalization
        // MTP: transfer state from donor batch when BatchGenerator merges a fresh
        // single-sequence batch into self via extend(). The MTP post-init fires on
        // the donor (whose __init__ ran with uids=[1]); without this transfer the
        // state would be lost when the donor is released.
        // omlx: batch_generator.py patched_extend
        if let donorState = other._omlxMtpState, _omlxMtpState == nil {
            _omlxMtpState = donorState
            other._omlxMtpState = nil
        }
        _compiledForward = nil
        _compiledCache = nil
    }

    /// Snapshot a compiled-decode cache into a standard `KVCache` for prefix
    /// capture on finish.
    ///
    /// - Full-attention `CompilableKVCache` overrides `state`/`offset` to read
    ///   the live `offsetArray`, so it already exposes a correct, temporally
    ///   ordered prefix. We copy that valid slice into a plain `KVCacheSimple`
    ///   so the captured cache is a normal, reusable, full-history prefix
    ///   (and is no longer a compiled object carrying decode-only buffers).
    /// - Sliding-window `CompilableRotatingKVCache` only retains the last
    ///   `windowSize` tokens and its inherited `idx`/`offset`/`state` are
    ///   intentionally NOT synced during compiled decode (syncing forces an
    ///   `eval` that `compile()` rejects). A windowed cache cannot serve as a
    ///   block-level prefix (which needs full-history KV), so it is returned
    ///   unchanged: it is not a `KVCacheSimple`, and `PrefixCache.storePrefix`
    ///   already requires every layer to cast to `KVCacheSimple`, so a response
    ///   containing any rotating layer is correctly excluded from prefix
    ///   storage. Coercing it to `KVCacheSimple` would be the actual bug —
    ///   it would present windowed KV as a full-history prefix.
    static func compiledCaptureSnapshot(_ cache: any KVCache) -> any KVCache {
        if let full = cache as? CompilableKVCache {
            let s = full.state
            guard s.count == 2 else { return cache }
            // Materialize so the snapshot owns its storage instead of lazily
            // aliasing the compiled cache's wide (maxLength) live buffer —
            // mirrors BatchKVCache.extract(). Defensive: capture happens at
            // finish (the row is then filtered out, nil-ing _compiledCache), so
            // the source is not mutated afterward today, but owning the buffer
            // keeps this correct if capture ever moves off the finish path and
            // releases the maxLength buffer once storePrefix slices its blocks.
            eval(s[0], s[1])
            let simple = KVCacheSimple()
            simple.state = s
            return simple
        }
        return cache
    }

    public var isEmpty: Bool { uids.isEmpty }
    public var batchSize: Int { uids.count }

    // MARK: - MTP post-init

    /// Called from `init` immediately after the standard `step()` for eligible
    /// single-sequence batches. Sets up `_omlxMtpState` and seeds the emit queue
    /// with two confirmed tokens so the first two `next()` calls bypass `step()`.
    /// omlx: batch_generator.py patched_init → _post_init_mtp
    internal func postInitMTP(model: any MTPCapable) {
        let sampler = samplers[0] ?? fallbackSampler

        // nextTokens = main_tok: sampled from prompt[-1]'s logits in init's step().
        let mainTok = nextTokens  // shape (1,) — asyncEval'd in step(); will force on use

        // 1. Backbone forward at main_tok → (logits [1,1,vocab], preNormHidden [1,1,H]).
        // omlx: _post_init_mtp — "1-token backbone forward at main_tok with hidden state"
        // Returns pre-norm hidden: the MTP head's pre_fc_norm_hidden applies its own norm.
        let mainInput = mainTok.reshaped(1, 1)
        let backboneCache = promptCache.map { $0 as any KVCache }
        let (logits, hidden) = model.callWithHidden(
            input: LMInput.Text(tokens: mainInput),
            cache: backboneCache,
            nConfirmed: 0
        )

        // 2. Sample next_main_tok from logits[:, -1, :]. Upcast to float32 to avoid
        // BF16 underflow on low-probability tokens (MTPLX: "fp32 p/q ratio" note).
        let nextMainLogits = logits[0..., -1, 0...].asType(.float32)  // [1, vocab]
        let nextMainLp = nextMainLogits - logSumExp(nextMainLogits, axis: -1, keepDims: true)
        let nextMainTok = sampler(nextMainLp)  // (1,)

        // 3. Run MTP head: (hidden_at_main [1,1,H], next_main_tok [1,1]) → draft [1,1,vocab].
        // omlx: _post_init_mtp — "MTP head sees (hidden_at_main, next_main_tok)"
        // Allocate the persistent MTP cache here so it's shared with all subsequent
        // stepMTP calls. The MTP head is auto-regressive (trained with growing KV context).
        // PR #990: `mtp_cache = model.make_mtp_cache()` — created once, reused every cycle.
        let mtpCache = model.makeMTPCache()
        let T = hidden.dim(1)
        let hiddenAtMain = hidden[0..., (T - 1) ..< T, 0...]  // [1, 1, H]
        let nextIds = nextMainTok.reshaped(1, 1)               // [1, 1]
        let mtpLogits = model.mtpForward(
            hidden: hiddenAtMain, nextTokenIds: nextIds, cache: mtpCache)
        let draftLogits2d = mtpLogits[0..., -1, 0...].asType(.float32)  // [1, vocab]
        let draftLp2d = draftLogits2d - logSumExp(draftLogits2d, axis: -1, keepDims: true)
        let draftTok = compactCategorical(draftLp2d, k: 512)  // (1,) — sample draft for losslessness

        // 4. Single eval for all sampled tokens. Cache draft_id as Int to avoid a
        //    GPU→CPU sync in the first verify cycle's accept/reject check.
        // omlx: _post_init_mtp — "mx.eval(main_tok, next_main_tok, draft_tok)"
        eval(mainTok, nextMainTok, draftTok)
        let mainId = Int(mainTok.asArray(UInt32.self)[0])
        let nextMainId = Int(nextMainTok.asArray(UInt32.self)[0])
        let draftId = Int(draftTok.asArray(UInt32.self)[0])

        let state = MTPState()
        state.mtpCache = mtpCache
        state.nextMain = nextMainTok
        state.draftTok = draftTok
        state.draftLp = draftLp2d[0]  // [vocab]
        state.draftId = draftId
        // Queue the two confirmed tokens. The first two next() calls emit from
        // this queue without any model forward — "two confirmed tokens" from PR 990.
        state.queue.append(MTPQueueItem(tokenId: mainId, source: "init"))
        state.queue.append(MTPQueueItem(tokenId: nextMainId, source: "init"))

        _omlxMtpState = state
    }

    // MARK: - Standard decode step (unchanged)

    /// One forward pass + per-row sample, double-buffered like upstream
    /// `mlx_lm.generate.GenerationBatch._step`.
    ///
    /// `nextTokens` is treated as the *current* token batch to return from
    /// this call. We immediately feed it back through the model, sample the
    /// following token batch, and `asyncEval` that future batch before
    /// synchronously materializing the current tokens for CPU-side stop
    /// detection / response dispatch.
    private func step() -> [Int] {
        let stepStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let currentTokens = nextTokens
        let inputs = currentTokens.reshaped(uids.count, 1)

        // Compiled decode: use the compiled forward closure instead of the
        // raw model call. The compiled graph fuses hundreds of per-layer FFI
        // crossings into a single call. For B=1 uses single-stream caches in
        // _compiledCache; for B>1 uses promoted batched caches in promptCache.
        // Inactive during MTP decode.
        let logits: MLXArray
        if let compiledForward = _compiledForward {
            logits = compiledForward([inputs])[0]
        } else {
            logits = model.callAsFunction(inputs, cache: promptCache.map { $0 as any KVCache })
        }
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "leg.forward.build", seconds: CFAbsoluteTimeGetCurrent() - stepStart)
        }

        // [B, 1, vocab] -> [B, vocab]
        let stepLogits = logits[0..., -1, 0...]

        let sampledTokens: MLXArray
        if samplers.contains(where: { $0 != nil }) {
            // The `logSumExp` reduction over the full vocabulary is only needed
            // when a row uses top-p (or a penalty sampler keyed on logprobs).
            // When no active row needs it (`skipLogprobNormalization`), pass the
            // raw logits through: every other filter/sampler is invariant to the
            // per-row constant shift, so results are unchanged. See the property
            // doc on `skipLogprobNormalization`.
            let logprobs =
                skipLogprobNormalization
                ? stepLogits
                : stepLogits - logSumExp(stepLogits, axis: -1, keepDims: true)
            var samples: [MLXArray] = []
            samples.reserveCapacity(uids.count)
            for i in 0 ..< uids.count {
                let rowLogprobs = logprobs[i ..< (i + 1), 0...]
                let sampler = samplers[i] ?? fallbackSampler
                samples.append(sampler(rowLogprobs))
            }
            sampledTokens = concatenated(samples, axis: 0)
        } else {
            // Greedy fast path. Avoid the full-vocabulary logSumExp when
            // all rows are greedy: argMax(logits) == argMax(logprobs),
            // and Swift does not currently expose logprobs downstream.
            // This removes one expensive reduction kernel per decode step.
            sampledTokens = argMax(stepLogits, axis: -1)
        }

        // Start computing the next token before forcing the current token
        // values back to the CPU. This overlaps GPU work with the CPU
        // extraction / response-building path.
        nextTokens = sampledTokens
        let evalStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        asyncEval(sampledTokens)

        let readbackStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record("leg.asyncEval.submit", seconds: readbackStart - evalStart)
        }
        eval(currentTokens)
        // For the common single-row batch, read the one scalar via `item`
        // instead of materializing a 1-element Swift array. Multi-row batches
        // keep the bulk `asArray` extraction unchanged.
        let stepTokens: [Int]
        if uids.count == 1 {
            stepTokens = [currentTokens.item(Int.self)]
        } else {
            stepTokens = currentTokens.asArray(UInt32.self).map { Int($0) }
        }
        if CBv2StepProfiler.enabled {
            let now = CFAbsoluteTimeGetCurrent()
            CBv2StepProfiler.record("leg.readback.wait", seconds: now - readbackStart)
            CBv2StepProfiler.record("leg.step.wall", seconds: now - stepStart)
        }

        for (i, t) in stepTokens.enumerated() {
            tokens[i].append(t)
        }

        return stepTokens
    }

    // MARK: - Compiled decode setup

    /// Set up compiled decode for standard decode (any batch size).
    ///
    /// **B=1 path**: Extracts single-row caches from the batched
    /// `promptCache`, promotes them to compilable types
    /// (`CompilableKVCache` / `CompilableRotatingKVCache`), and builds a
    /// compiled forward closure stored in `_compiledCache`.
    ///
    /// **B>1 path**: Promotes batched caches in-place in `promptCache` to
    /// `CompilableBatchKVCache` / `CompilableBatchRotatingKVCache` and
    /// builds a compiled forward closure. The promoted caches remain in
    /// `promptCache` so filter/extend operate on them directly. On
    /// filter/extend the compiled forward is invalidated (batch-size
    /// change) and the caches continue as regular batched caches.
    ///
    /// Skipped when:
    /// - `DARKBLOOM_COMPILED_DECODE` is disabled
    /// - hardware doesn't support compiled decode
    /// - cache layers include unsupported types (e.g. `ArraysCache`)
    private func setupCompiledDecodeIfEligible() {
        guard CompiledDecode.isEnabled else { return }

        if batchSize == 1 {
            // B=1: extract → promote single-stream caches → compile.
            var singleCaches: [KVCache] = promptCache.map { $0.extractBatched(0) }

            let currentOffset = singleCaches.map { $0.offset }.max() ?? 0
            let maxRemaining = maxTokens.first ?? 256
            let needed = currentOffset + maxRemaining + 64
            let maxCacheLength = max(4096, ((needed + 255) / 256) * 256)

            if let compiled = CompiledDecode.setupCompiledDecode(
                model: model, cache: &singleCaches, maxCacheLength: maxCacheLength
            ) {
                _compiledForward = compiled
                _compiledCache = singleCaches
            }
        } else {
            // B>1: promote batched caches in-place → compile.
            let currentOffset = promptCache.map { ($0 as any KVCache).offset }.max() ?? 0
            let maxRemaining = maxTokens.max() ?? 256
            let needed = currentOffset + maxRemaining + 64
            // Size tightly for B>1 — the overflow bin overhead (attention
            // over masked zero positions) scales with B, so minimizing the
            // buffer width is critical for throughput.
            let maxCacheLength = max(256, ((needed + 255) / 256) * 256)

            if let compiled = CompiledDecode.setupBatchCompiledDecode(
                model: model, cache: &promptCache, maxCacheLength: maxCacheLength
            ) {
                _compiledForward = compiled
                // _compiledCache stays nil — the caches live in promptCache.
            }
        }
    }

    // MARK: - Drafter-based MTP (Gemma 4) decode

    /// Whether the batch may speculate this call: size within the runtime's
    /// cap and every row greedy (the batched round loop is greedy in v1).
    private func mtpEligible(_ runtime: any BatchedMTPRuntime) -> Bool {
        guard !uids.isEmpty else { return false }
        guard uids.count <= runtime.maxBatch else { return false }
        guard !samplers.contains(where: { $0 != nil }) else { return false }
        return true
    }

    /// Run one drafter-MTP round (or enter a session) and emit its tokens.
    /// Returns per-row responses (possibly several per uid — the scheduler
    /// appends them in order), or nil if a session couldn't be started (the
    /// caller then falls back to plain decode for this batch).
    private func gemma4MtpNext(runtime: any BatchedMTPRuntime) -> [GenerationBatchResponse]? {
        let caches = promptCache.map { $0 as any KVCache }
        var blocks: [[Int]]

        if mtpSession == nil {
            // ENTER: feed the buffered token (seed) via one forward, capturing
            // the per-row carry + bonus. Emit [seed, bonus]: seed is the
            // model's confirmed next token; bonus is the round-1 draft seed and
            // is not re-emitted by the round, so it must be emitted here.
            eval(nextTokens)
            let seed = nextTokens.asArray(UInt32.self).map { Int($0) }
            guard let (session, bonus) = runtime.beginSession(
                seedTokens: nextTokens, caches: caches)
            else { return nil }
            mtpSession = session
            blocks = (0 ..< uids.count).map { [seed[$0], bonus[$0]] }
        } else {
            // STEADY: one round, capped per row by the remaining budget.
            let maxNewPerRow = (0 ..< uids.count).map {
                Swift.max(1, maxTokens[$0] - numTokens[$0])
            }
            blocks = mtpSession!.step(caches: caches, maxNewPerRow: maxNewPerRow)
        }

        return emitMTPBlocks(blocks)
    }

    /// Apply the standard per-token epilogue to each row's emitted block,
    /// truncating a row at its first finish, then evict finished rows
    /// (filter + session compaction). Mirrors the standard `next()` epilogue.
    private func emitMTPBlocks(_ blocks: [[Int]]) -> [GenerationBatchResponse] {
        let n = uids.count
        var responses: [GenerationBatchResponse] = []
        var keep: [Int] = []
        keep.reserveCapacity(n)

        for i in 0 ..< n {
            var finishedRow = false
            for t in blocks[i] {
                if finishedRow { break }
                numTokens[i] += 1
                tokens[i].append(t)

                var finishReason: String? = nil
                if numTokens[i] >= maxTokens[i] { finishReason = "length" }
                let machine = stateMachines[i]
                let (nextState, matchedSequence, currentState) =
                    machine.match(matcherStates[i], t)
                matcherStates[i] = nextState
                if matchedSequence != nil, currentState == nil { finishReason = "stop" }

                if let finishReason {
                    finishedRow = true
                    // Prefix-cache storage is skipped for MTP rows (promptCache
                    // nil): the live cache is the confirmed prefix and excludes
                    // the row's pending tail, so the extracted cache wouldn't
                    // match the full emitted sequence. Storing the complete
                    // prefix is a follow-up (feed pendings before extract).
                    responses.append(GenerationBatchResponse(
                        uid: uids[i], token: t, finishReason: finishReason,
                        matchedSequence: matchedSequence, currentState: currentState,
                        allTokens: tokens[i], promptCache: nil))
                } else {
                    responses.append(GenerationBatchResponse(
                        uid: uids[i], token: t, finishReason: nil,
                        matchedSequence: matchedSequence, currentState: currentState,
                        allTokens: nil, promptCache: nil))
                }
            }
            if !finishedRow { keep.append(i) }
        }

        if keep.count < n {
            // `filter` is the single choke point that compacts the session
            // carry (and clears it when the batch empties) — see filter(keep:).
            filter(keep: keep)
        }

        return responses
    }
}

// MARK: - MTP fallback error

/// Signals a clean fallback to the standard step() inside the MTP dispatch path.
/// omlx: batch_generator.py _MtpStepFallback
private enum MTPFallback: Error {
    case missingVerifyInputs
    case verifyProducedNoTokens
    case cacheRollbackFailed
}

// MARK: - MTP dispatch (private extension)
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/batch_generator.py
//   _mtp_next, _run_verify_cycle, _step_mtp, _residual_sample, _emit_response,
//   _restore_or_trim_caches, _clear_rollback, _bump_emit_stat
//
// This extension intercepts next() for eligible single-sequence batches.
// The queue emitted by postInitMTP (two confirmed tokens) is drained first;
// subsequent calls run the 2-token verify cycle.

private extension GenerationBatch {

    // MARK: next() dispatch

    /// Emit one token from the queue; run a verify cycle if the queue is empty.
    /// omlx: batch_generator.py _mtp_next
    func mtpNext(state: MTPState, model: any MTPCapable) throws -> [GenerationBatchResponse] {
        if !state.queue.isEmpty {
            let item = state.queue.removeFirst()
            bumpEmitStat(&state.stats, source: item.source)
            return emitMTPToken(item, state: state)
        }

        try runVerifyCycle(state: state, model: model)
        guard !state.queue.isEmpty else {
            // Verify cycle must always produce at least the rejected-verify token.
            // omlx: _mtp_next — "verify cycle produced no emit tokens"
            throw MTPFallback.verifyProducedNoTokens
        }

        let item = state.queue.removeFirst()
        bumpEmitStat(&state.stats, source: item.source)
        return emitMTPToken(item, state: state)
    }

    // MARK: Verify cycle

    /// 2-token backbone forward [next_main, draft] with nConfirmed=1.
    /// Populates state.queue with 1 (reject) or 2 (accept) tokens for the
    /// upcoming emit calls. Updates state.nextMain / draftTok / draftLp for
    /// the following cycle.
    /// omlx: batch_generator.py _run_verify_cycle
    func runVerifyCycle(state: MTPState, model: any MTPCapable) throws {
        guard let nextMain = state.nextMain, let draftTok = state.draftTok else {
            throw MTPFallback.missingVerifyInputs
        }

        let sampler = samplers[0] ?? fallbackSampler
        let isGreedy = samplers[0] == nil

        // Concatenate [next_main (1,), draft_tok (1,)] → (2,) → [1, 2] input.
        let inputs = concatenated([nextMain, draftTok], axis: 0).reshaped(1, 2)
        let backboneCache = promptCache.map { $0 as any KVCache }

        // --- backbone forward (2-token, nConfirmed=1) ---
        // nConfirmed=1: GatedDeltaNet runs two sequential M=1 processChunk calls, saving
        // a mid-point rollbackState snapshot for the reject path. Benchmarks show this costs
        // only ~2ms more than nConfirmed=0 (single M=2 processChunk) since the DeltaNet
        // layers are not the bottleneck — the 28 attention layers dominate KV cache bandwidth.
        // omlx: _run_verify_cycle — "logits, hidden = gen_batch.model(..., n_confirmed=1)"
        let t0 = CFAbsoluteTimeGetCurrent()
        let (logits, hidden) = model.callWithHidden(
            input: LMInput.Text(tokens: inputs),
            cache: backboneCache,
            nConfirmed: 1
        )
        state.stats.backboneMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000

        // verify position = [:, 0, :]; bonus position = [:, 1, :]
        let verifyLogits = logits[0..., 0, 0...]  // [1, vocab]
        let bonusLogits  = logits[0..., 1, 0...]  // [1, vocab]

        // Batched logprob: one logsumexp over (2, vocab) vs two over (1, vocab).
        // Upcast to float32 before log-softmax: BF16 underflows at low probabilities,
        // causing -inf logprobs that break the acceptance ratio (MTPLX "fp32 p/q ratio" note).
        // omlx: _run_verify_cycle — "combined_logits = mx.concatenate([verify_logits, bonus_logits])"
        let tSample = CFAbsoluteTimeGetCurrent()
        let combinedLogits = concatenated([verifyLogits, bonusLogits], axis: 0).asType(.float32)  // [2, vocab]
        let combinedLp = combinedLogits - logSumExp(combinedLogits, axis: -1, keepDims: true)
        let verifyLp2d = combinedLp[0 ..< 1, 0...]  // [1, vocab]
        let bonusLp2d  = combinedLp[1 ..< 2, 0...]  // [1, vocab]

        let draftId = state.draftId

        // Accept check WITHOUT full categorical sampling.
        // Greedy: argmax(verifyLp2d) == draft token (fast, O(vocab) max reduction).
        // Stochastic: log P_target(draft) / log P_draft(draft) ratio check (single index read).
        // Deferring sampling saves one categorical (~18ms at vocab=151K) per cycle:
        // on accept we sample bonusTok; on reject we sample from residual distribution.
        // Both paths perform exactly ONE categorical vs the previous unconditional TWO.
        let accept: Bool
        if isGreedy {
            let verifyArgmax = argMax(verifyLp2d, axis: -1)
            eval(verifyArgmax)
            accept = Int(verifyArgmax.asArray(UInt32.self)[0]) == draftId
        } else {
            let logTargetAtDraft = Double(verifyLp2d[0, draftId].asArray(Float.self)[0])
            let logDraftAtDraft  = Double(state.draftLp![draftId].asArray(Float.self)[0])
            let logAccept = logTargetAtDraft - logDraftAtDraft
            accept = logAccept >= 0 || Double.random(in: 0 ..< 1) < exp(logAccept)
        }
        state.stats.sampleMs += (CFAbsoluteTimeGetCurrent() - tSample) * 1000

        // Hidden at each verify-forward position.
        let hiddenAtConfirmed = hidden[0..., 0 ..< 1, 0...]  // [1, 1, H]
        let hiddenAtDraft     = hidden[0..., 1 ..< 2, 0...]  // [1, 1, H]

        state.stats.cycles += 1

        if accept {
            state.stats.accepts += 1

            // Accept path: clear rollback snapshots, sample bonus, run MTP head at draft position.
            // ONE categorical (bonusTok) — verify token not needed on accept path.
            // omlx: _run_verify_cycle accept branch
            let tCache = CFAbsoluteTimeGetCurrent()
            clearRollback()
            state.stats.cacheOpsMs += (CFAbsoluteTimeGetCurrent() - tCache) * 1000

            // Use compact sampling (argPartition top-K) to avoid categorical(151K) ≈ 18ms.
            // Greedy: argmax is already O(N) optimal. Stochastic: pre-filter to top-512
            // and sample from that compact distribution — semantically identical to full
            // vocab when topP nucleus fits within 512 tokens (true for topP≤0.9 on 151K vocab).
            let bonusTok = isGreedy
                ? argMax(bonusLp2d, axis: -1)
                : compactSample(bonusLp2d, sampler: sampler)
            eval(bonusTok)
            let bonusId = Int(bonusTok.asArray(UInt32.self)[0])

            let (newDraft, newDraftLp) = stepMTP(
                state: state, model: model,
                hiddenAtPosition: hiddenAtDraft,
                nextMainTok: bonusTok
            )
            // Queue: accepted draft uses MTP head's original draft distribution;
            // bonus uses the verify forward's bonus distribution.
            // omlx: _run_verify_cycle — "state.queue.append((draft_id, state.draft_lp, 'draft'))"
            state.queue.append(MTPQueueItem(tokenId: draftId, source: "draft"))
            state.queue.append(MTPQueueItem(tokenId: bonusId, source: "bonus"))
            state.nextMain = bonusTok
            state.draftTok = newDraft
            state.draftLp  = newDraftLp
            return
        }

        // Reject path: restore / trim caches, sample corrected token, run MTP at confirmed.
        // ONE categorical (residualSample) or zero (greedy fallback) — bonus not needed.
        // omlx: _run_verify_cycle reject branch
        state.stats.rejects += 1
        let tCache = CFAbsoluteTimeGetCurrent()
        guard restoreOrTrimCaches() else {
            throw MTPFallback.cacheRollbackFailed
        }
        state.stats.cacheOpsMs += (CFAbsoluteTimeGetCurrent() - tCache) * 1000

        let emitId: Int
        if isGreedy {
            // Greedy: emit argmax of verify distribution (already computed in accept check).
            let verifyArgmax = argMax(verifyLp2d, axis: -1)
            eval(verifyArgmax)
            emitId = Int(verifyArgmax.asArray(UInt32.self)[0])
        } else {
            // Stochastic: residual sample from max(P_target - P_draft, 0) / Z.
            // Fall back to argmax if residual mass is negligible (z ≤ 1e-8).
            let (rid, _) = residualSample(
                verifyLp2d: verifyLp2d, draftLp1d: state.draftLp!)
            if let rid {
                emitId = rid
            } else {
                let verifyArgmax = argMax(verifyLp2d, axis: -1)
                eval(verifyArgmax)
                emitId = Int(verifyArgmax.asArray(UInt32.self)[0])
            }
        }

        let emitTok = MLXArray([UInt32(emitId)])  // (1,) uint32
        let (newDraft, newDraftLp) = stepMTP(
            state: state, model: model,
            hiddenAtPosition: hiddenAtConfirmed,
            nextMainTok: emitTok
        )
        state.queue.append(MTPQueueItem(tokenId: emitId, source: "verify"))
        state.nextMain = emitTok
        state.draftTok = newDraft
        state.draftLp  = newDraftLp
    }

    // MARK: MTP head forward

    /// One MTP-head forward + sample. Returns (draft_tok [1], draft_lp [vocab]).
    /// Side effect: stores the host-side int in `state.draftId` to avoid a
    /// GPU→CPU sync on the next verify cycle's accept/reject check.
    /// omlx: batch_generator.py _step_mtp
    func stepMTP(
        state: MTPState,
        model: any MTPCapable,
        hiddenAtPosition: MLXArray,
        nextMainTok: MLXArray
    ) -> (MLXArray, MLXArray) {
        let t0 = CFAbsoluteTimeGetCurrent()

        let nextIds = nextMainTok.reshaped(1, 1)  // [1, 1]
        // Use the persistent MTP cache from MTPState. The MTP head is auto-regressive
        // (one full-attention transformer layer) and was trained with a growing KV context
        // accumulated across the generation. PR #990 creates the cache once at init.
        let mtpCache = state.mtpCache ?? model.makeMTPCache()
        let mtpLogits = model.mtpForward(
            hidden: hiddenAtPosition, nextTokenIds: nextIds, cache: mtpCache)
        let mtpLogits2d = mtpLogits[0..., -1, 0...].asType(.float32)  // [1, vocab]
        let newLp2d = mtpLogits2d - logSumExp(mtpLogits2d, axis: -1, keepDims: true)

        // Sample the MTP draft from compactCategorical(top-512) so q matches the
        // acceptance distribution exactly — fully lossless speculative decoding.
        // compactCategorical is ~0.1ms/cycle (vs ~18ms for full 150K-vocab categorical).
        // draft_lp (newLp2d[0]) is stored as the correct q distribution for accept check.
        let newTok = compactCategorical(newLp2d, k: 512)  // (1,) — compact-categorical draft

        // Force eval + cache int: avoids re-sync on next cycle's accept check.
        // omlx: _step_mtp — "draft_id_int = int(new_tok.tolist()[0])"
        eval(newTok)
        state.draftId = Int(newTok.asArray(UInt32.self)[0])
        state.stats.mtpHeadMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000

        return (newTok, newLp2d[0])  // ([1], [vocab])
    }

    // MARK: Compact sampling helpers

    /// Sample from log-probability distribution lp [1, vocab] using a compact top-K subset.
    ///
    /// Uses `argPartition` (O(N)) to select the K highest-logprob tokens, then calls the
    /// sampler closure on the compact [1, K] distribution, and maps the local index back
    /// to the full vocabulary. This reduces the categorical kernel from vocab=151K to K,
    /// eliminating the dominant ~18ms sampling bottleneck per MTP verify cycle.
    ///
    /// Correctness: the sampler's topP filter is applied to the K pre-filtered tokens.
    /// This is equivalent to the full-vocab filter when K >= topP nucleus size (typically
    /// 50–200 tokens for topP=0.9 on a 151K vocab), which K=512 satisfies conservatively.
    private func compactSample(_ lp: MLXArray, sampler: RowSampler, k: Int = 512) -> MLXArray {
        let vocab = lp.dim(-1)
        if vocab <= k { return sampler(lp) }

        // argPartition(-lp, kth: k-1): indices at positions [0, k-1] are the K largest
        // elements of lp (= K smallest of -lp), in arbitrary order. O(N) vs O(N log N) sort.
        let partIdx = argPartition(-lp, kth: k - 1, axis: -1)  // [1, vocab]
        let topKIdx = partIdx[0..., 0 ..< k]                   // [1, K]
        let topKLp  = takeAlong(lp, topKIdx, axis: -1)         // [1, K] compact log-probs

        // Sample from the compact distribution using the sampler (temp + topP + categorical).
        let localTok = sampler(topKLp)  // (1,) in [0, K-1]

        // Map local index back to the full vocabulary.
        return takeAlong(topKIdx, localTok.reshaped(1, 1), axis: -1).reshaped(-1)  // (1,)
    }

    /// Direct compact categorical over lp [1, vocab] without a sampler closure.
    /// Used for residual sampling where temperature has already been folded into the distribution.
    private func compactCategorical(_ lp: MLXArray, k: Int = 512) -> MLXArray {
        let vocab = lp.dim(-1)
        if vocab <= k { return MLXRandom.categorical(lp, axis: -1) }

        let partIdx   = argPartition(-lp, kth: k - 1, axis: -1)  // [1, vocab]
        let topKIdx   = partIdx[0..., 0 ..< k]                   // [1, K]
        let topKLp    = takeAlong(lp, topKIdx, axis: -1)         // [1, K]
        let localTok  = MLXRandom.categorical(topKLp, axis: -1)  // (1,)
        return takeAlong(topKIdx, localTok.reshaped(1, 1), axis: -1).reshaped(-1)  // (1,)
    }

    // MARK: Residual sampling

    /// Sample from max(P_target - P_draft, 0) / Z  (Leviathan et al. 2023).
    /// Returns (tokenId or nil, verify_lp_1d). Caller falls back to verify_id when nil.
    /// omlx: batch_generator.py _residual_sample
    func residualSample(
        verifyLp2d: MLXArray,
        draftLp1d: MLXArray
    ) -> (Int?, MLXArray) {
        let pTarget  = exp(verifyLp2d[0])   // [vocab]
        let pDraft   = exp(draftLp1d)        // [vocab]
        let residual = maximum(pTarget - pDraft, MLXArray(Float(0)))
        let zArr     = residual.sum()
        eval(zArr)
        let z = Double(zArr.asArray(Float.self)[0])
        if z <= 1e-8 {
            return (nil, verifyLp2d[0])
        }
        // Sample from the normalised residual distribution.
        // omlx: _residual_sample — "mx.random.categorical(mx.log(residual / z + 1e-10)...)"
        let logResidNorm = log(residual / Float(z) + Float(1e-10)).reshaped(1, -1)  // [1, vocab]
        // Compact categorical: argPartition top-512 → categorical(512) instead of categorical(151K).
        // The residual mass is concentrated on tokens where P_target > P_draft (i.e. high-prob
        // tokens), so top-512 safely contains the entire residual nucleus.
        let sample = compactCategorical(logResidNorm, k: 512)  // (1,)
        eval(sample)
        return (Int(sample.asArray(Int32.self)[0]), verifyLp2d[0])
    }

    // MARK: Response builder

    /// Build a single-element response, applying the standard per-token epilogue
    /// (token append, maxTokens / matcher checks). Calls filter([]) and clears
    /// MTP state on the final token.
    /// omlx: batch_generator.py _emit_response
    func emitMTPToken(_ item: MTPQueueItem, state: MTPState) -> [GenerationBatchResponse] {
        let tokenId = item.tokenId

        tokens[0].append(tokenId)
        numTokens[0] += 1

        var finishReason: String? = nil
        if numTokens[0] >= maxTokens[0] {
            finishReason = "length"
        }

        let machine = stateMachines[0]
        let (nextMatchState, matchedSequence, currentState) =
            machine.match(matcherStates[0], tokenId)
        matcherStates[0] = nextMatchState
        if matchedSequence != nil, currentState == nil {
            finishReason = "stop"
        }

        if let finishReason {
            let extracted: [any KVCache] = promptCache.map { $0.extractBatched(0) }
            let response = GenerationBatchResponse(
                uid: uids[0],
                token: tokenId,
                finishReason: finishReason,
                matchedSequence: matchedSequence,
                currentState: currentState,
                allTokens: tokens[0],
                promptCache: extracted
            )
            // Drop MTP state before filter([]) so patched_filter doesn't double-clear.
            // omlx: _emit_response finish path — "delattr(gen_batch, '_omlx_mtp_state')"
            if let s = _omlxMtpState {
                let st = s.stats
                let n = Double(max(1, st.cycles))
                print("[MTP] cycles=\(st.cycles) accepts=\(st.accepts) rejects=\(st.rejects) acceptRate=\(String(format:"%.2f",st.acceptRate)) emits=init:\(st.initEmits)+draft:\(st.draftEmits)+bonus:\(st.bonusEmits)+verify:\(st.verifyEmits)=\(st.totalEmits) backbone=\(String(format:"%.1f",st.backboneMs/n))ms/cycle mtp=\(String(format:"%.1f",st.mtpHeadMs/n))ms/cycle sample=\(String(format:"%.1f",st.sampleMs/n))ms/cycle cacheOps=\(String(format:"%.1f",st.cacheOpsMs/n))ms/cycle")
            }
            _omlxMtpState = nil
            filter(keep: [])
            return [response]
        }

        return [GenerationBatchResponse(
            uid: uids[0],
            token: tokenId,
            finishReason: nil,
            matchedSequence: matchedSequence,
            currentState: currentState,
            allTokens: nil,
            promptCache: nil
        )]
    }

    // MARK: Cache rollback helpers

    /// Roll back one token from each layer cache after a draft rejection.
    /// SSM/linear-attention layers restore their `rollbackState` snapshot;
    /// full-attention layers trim by 1. Returns false if any layer supports neither.
    /// omlx: batch_generator.py _restore_or_trim_caches
    @discardableResult
    func restoreOrTrimCaches() -> Bool {
        for cache in promptCache {
            if let ac = cache as? ArraysCache, let snap = ac.rollbackState {
                let (convSnap, ssmSnap) = snap
                ac[0] = convSnap
                ac[1] = ssmSnap
                ac.rollbackState = nil
                continue
            }
            if cache.isTrimmable {
                cache.trim(1)
                continue
            }
            return false
        }
        return true
    }

    /// Drop rollback snapshots after a draft is accepted.
    /// omlx: batch_generator.py _clear_rollback
    func clearRollback() {
        for cache in promptCache {
            if let ac = cache as? ArraysCache {
                ac.rollbackState = nil
            }
        }
    }

    // MARK: Stat helpers

    /// Increment the emit-source stat counter.
    /// omlx: batch_generator.py _bump_emit_stat
    func bumpEmitStat(_ stats: inout MTPStats, source: String) {
        switch source {
        case "init":   stats.initEmits += 1
        case "draft":  stats.draftEmits += 1
        case "bonus":  stats.bonusEmits += 1
        case "verify": stats.verifyEmits += 1
        default:       break
        }
    }
}
