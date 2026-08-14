// CBv2Fixtures.swift — WS-G shared fixtures for the ContinuousBatchingV2 harness.
//
// Everything here is built against the FROZEN contract in
// `Libraries/MLXLMCommon/ContinuousBatchingV2/CBv2Contracts.swift`. Types owned
// by other workstreams (WS-A sequence KV / layer caches, WS-B engine loop) are
// mocked with minimal-but-numerically-correct reference implementations so the
// cross-cutting suites (invariance, parity, invariant suite) are executable
// before integration. At integration the suites re-point at the real
// implementations; the mocks define the semantics those implementations must
// reproduce (token-exact vs the legacy B=1 engine).
//
// Contents:
//  - `TinyTestModel` — a real 2-layer transformer (hidden 64, 4 query heads,
//    2 KV heads, head dim 16, vocab 128, seeded random weights) with one
//    full-attention layer and one sliding-window(16) layer, optional GPT-OSS
//    style attention sinks. Implements BOTH the legacy path (`LanguageModel`
//    + `KVCache`, runs through the old Scheduler/EngineCore) and the v2 path
//    (`CBv2SteppableModel` + `CBv2AttendingLayerCache`).
//  - `HarnessFullSequenceKV` / `HarnessWindowedSequenceKV` — mock
//    `CBv2SequenceKV` (per-request storage, absolute position counters,
//    window eviction keyed to absolute positions, recent end kept).
//  - `HarnessLayerCache` — mock `CBv2AttendingLayerCache`: per-row SDPA,
//    no shared frontier, no left padding, decode is mask-free by construction.
//  - `HarnessKVBackend` — mock `CBv2KVBackend` with truthful byte accounting
//    and static sink-eligibility enforcement.
//  - `CBv2HarnessEngine` — a minimal deterministic greedy step-loop driver
//    (per-request chunked prefill, rectangular [B,1] decode, join/leave).
//
// NO model downloads: weights are seeded random (MLXRandom.seed).

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLMCommon

// NOTE (integration): WS-G's `CBv2HarnessSteppableModel` mirror protocol is
// retired — `TinyTestModel` now conforms to the real `CBv2SteppableModel`
// (EngineLoopV2.swift), which has the identical shape. See
// docs/engine-v2/CONTRACT-ISSUES-G-harness.md item 1.

// MARK: - Host-sync instrumentation

/// Counts host-side synchronization points, split by origin, so the harness
/// can assert the layer-cache/KV hot path performs ZERO host syncs during
/// decode (report 10 §4 invariant 7; WS-A spec "No-.item()-in-decode").
///
/// The mock layer caches / sequence KVs perform no `.item()`/`eval` at all —
/// any future sync added to a mock must increment `layerCacheSyncs`, which
/// the invariant-7 test pins to zero. The engine driver's per-step token
/// readback is the one allowed boundary sync and is counted separately
/// (`boundarySyncs`, exactly one per decode step).
enum CBv2HarnessSyncCounter {
    nonisolated(unsafe) static var layerCacheSyncs = 0
    nonisolated(unsafe) static var boundarySyncs = 0

    static func reset() {
        layerCacheSyncs = 0
        boundarySyncs = 0
    }
}

// MARK: - Mock per-sequence KV (WS-A owns the real ones)

/// Internal read access shared by the mocks so `attendBorrowing` can read a
/// source layer's retained K/V without updating it.
protocol HarnessReadableKV: AnyObject {
    /// Current retained (keys, values) in temporal order, or nil when empty.
    func currentViews() -> (MLXArray, MLXArray)?
}

/// Full-attention per-sequence KV: contiguous append, retains everything.
final class HarnessFullSequenceKV: CBv2SequenceKV, HarnessReadableKV {
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var absoluteOffset: Int = 0

    var retainedCount: Int { keys?.dim(2) ?? 0 }

    var byteCount: Int {
        ((keys?.nbytes ?? 0) + (values?.nbytes ?? 0))
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        precondition(newKeys.dim(0) == 1, "per-sequence KV is [1, kvHeads, n, headDim]")
        let n = newKeys.dim(2)
        if let k = keys, let v = values {
            keys = concatenated([k, newKeys], axis: 2)
            values = concatenated([v, newValues], axis: 2)
        } else {
            keys = newKeys
            values = newValues
        }
        absoluteOffset += n
        return (keys!, values!)
    }

    func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        guard let k = keys, let v = values else {
            fatalError("snapshot() on empty HarnessFullSequenceKV")
        }
        return (k, v, absoluteOffset)
    }

    func rollback(_ n: Int) {
        guard n > 0 else { return }
        precondition(n <= retainedCount, "rollback(\(n)) exceeds retained \(retainedCount)")
        let keep = retainedCount - n
        if keep == 0 {
            keys = nil
            values = nil
        } else {
            // Slice out the tail so rolled-back K/V can never be attended to.
            keys = keys![.ellipsis, ..<keep, 0...]
            values = values![.ellipsis, ..<keep, 0...]
        }
        absoluteOffset -= n
    }

    func currentViews() -> (MLXArray, MLXArray)? {
        guard let k = keys, let v = values else { return nil }
        return (k, v)
    }

    /// Test hook (rollback scrubbing): overwrite the tail `n` entries with a
    /// sentinel value, simulating stale speculative K/V that a later rollback
    /// must make unreachable.
    func poisonTail(_ n: Int, value: Float) {
        guard let k = keys, n > 0, n <= retainedCount else { return }
        let keep = retainedCount - n
        let sentinelShape = [1, k.dim(1), n, k.dim(3)]
        let sentinel = MLXArray.full(sentinelShape, values: MLXArray(value))
        if keep == 0 {
            keys = sentinel
            values = sentinel
        } else {
            keys = concatenated([k[.ellipsis, ..<keep, 0...], sentinel], axis: 2)
            values = concatenated([values![.ellipsis, ..<keep, 0...], sentinel], axis: 2)
        }
    }
}

/// Sliding-window per-sequence KV. Eviction is keyed to ABSOLUTE positions
/// and keeps the RECENT end. Semantics match the legacy rotating cache's
/// battle-tested convention (post-update retention):
///  - multi-token update (prefill chunk): old context is trimmed to
///    `window - 1` BEFORE appending, so every query in the chunk still sees
///    its full window (mask enforces per-query visibility);
///  - single-token update (decode): append then trim to exactly `window`.
final class HarnessWindowedSequenceKV: CBv2SequenceKV, HarnessReadableKV {
    let window: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var absoluteOffset: Int = 0

    init(window: Int) {
        precondition(window > 0)
        self.window = window
    }

    /// Contract `fastForward(to:)` — advance the counter without storage
    /// (prefix adoption places windowed rows at the uniform adopted offset).
    func fastForward(to offset: Int) {
        precondition(keys == nil && absoluteOffset == 0, "fastForward requires a fresh state")
        absoluteOffset = offset
    }

    var retainedCount: Int { keys?.dim(2) ?? 0 }

    var byteCount: Int {
        ((keys?.nbytes ?? 0) + (values?.nbytes ?? 0))
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        precondition(newKeys.dim(0) == 1, "per-sequence KV is [1, kvHeads, n, headDim]")
        let n = newKeys.dim(2)
        if var k = keys, var v = values {
            if n > 1 {
                // Prefill chunk: keep at most (window - 1) most-recent old
                // entries so the oldest query in the chunk still has its full
                // window available. (Same convention as RotatingKVCache /
                // BatchRotatingKVCache stepCount > 1.)
                let trimSize = k.dim(2) - (window - 1)
                if trimSize > 0 {
                    k = k[.ellipsis, trimSize..., 0...]
                    v = v[.ellipsis, trimSize..., 0...]
                }
                keys = concatenated([k, newKeys], axis: 2)
                values = concatenated([v, newValues], axis: 2)
            } else {
                // Decode: append, then evict down to exactly `window`,
                // keeping the RECENT end.
                var mergedK = concatenated([k, newKeys], axis: 2)
                var mergedV = concatenated([v, newValues], axis: 2)
                let trimSize = mergedK.dim(2) - window
                if trimSize > 0 {
                    mergedK = mergedK[.ellipsis, trimSize..., 0...]
                    mergedV = mergedV[.ellipsis, trimSize..., 0...]
                }
                keys = mergedK
                values = mergedV
            }
        } else {
            keys = newKeys
            values = newValues
        }
        absoluteOffset += n
        return (keys!, values!)
    }

    func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        guard let k = keys, let v = values else {
            fatalError("snapshot() on empty HarnessWindowedSequenceKV")
        }
        return (k, v, absoluteOffset)
    }

    func rollback(_ n: Int) {
        guard n > 0 else { return }
        precondition(
            n <= retainedCount,
            "windowed rollback(\(n)) exceeds retained \(retainedCount)")
        let keep = retainedCount - n
        if keep == 0 {
            keys = nil
            values = nil
        } else {
            keys = keys![.ellipsis, ..<keep, 0...]
            values = values![.ellipsis, ..<keep, 0...]
        }
        absoluteOffset -= n
    }

    func currentViews() -> (MLXArray, MLXArray)? {
        guard let k = keys, let v = values else { return nil }
        return (k, v)
    }
}

// MARK: - Mock attending layer cache (WS-A owns the real one)

/// Reference `CBv2AttendingLayerCache`: per-row attention against each row's
/// own KV. No left padding, no shared frontier, no batch-wide state.
///
/// Mask policy is PINNED — a pure function of (L, layer kind), never
/// data-dependent, never drifting across steps (report 10 §4 invariant 5):
///  - decode (L == 1): `.none` for every layer kind — each row attends exactly
///    its own retained KV, all of which is in-window by construction;
///  - prefill (L > 1, B == 1): boolean causal(∧window) mask built from
///    absolute positions relative to the row's retained storage — identical
///    content to the legacy `createCausalMask` path, which is what makes B=1
///    legacy-vs-v2 parity bit-exact.
final class HarnessLayerCache: CBv2AttendingLayerCache {
    let layerIndex: Int
    let kind: CBv2LayerKind
    var rows: [CBv2SequenceKV]

    /// Log of mask modes chosen, per updateAndAttend call ("none" | "array"),
    /// used by the pinned-attention-path invariant test.
    private(set) var maskModeLog: [String] = []

    init(layerIndex: Int, kind: CBv2LayerKind, rows: [CBv2SequenceKV] = []) {
        self.layerIndex = layerIndex
        self.kind = kind
        self.rows = rows
    }

    func setRows(_ rows: [CBv2SequenceKV]) {
        self.rows = rows
    }

    /// Per-row absolute RoPE offsets as a device array [B]. Built from the
    /// host-side per-row counters (host → device upload; NOT a GPU→CPU sync).
    /// Read BEFORE update() advances the counters — snapshot semantics
    /// (report 10 §4 invariant 1).
    var positionOffsets: MLXArray {
        MLXArray(rows.map { Int32($0.absoluteOffset) })
    }

    private var windowSize: Int? {
        if case .slidingWindow(let w) = kind.attention { return w }
        return nil
    }

    /// Boolean causal(∧window) mask for a [1, L] prefill chunk against
    /// `retained` keys (temporal order, `retained - L` of them older than the
    /// chunk). Content-identical to the legacy
    /// `createCausalMask(n:offset:windowSize:)`.
    private func prefillMask(chunkLength L: Int, retainedAfterUpdate retained: Int) -> MLXArray {
        createCausalMask(n: L, offset: retained - L, windowSize: windowSize)
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(B == rows.count, "row order mismatch: B=\(B) rows=\(rows.count)")
        precondition(
            B == 1 || L == 1,
            "decode must be rectangular [B, 1]; prefill must be [1, chunk] (B=\(B), L=\(L))")
        if kind.hasSinks {
            precondition(sinks != nil, "layer declares sinks but none were passed")
        }

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for (i, row) in rows.enumerated() {
            let rowQueries = B == 1 ? queries : queries[i ..< (i + 1)]
            let rowKeys = B == 1 ? keys : keys[i ..< (i + 1)]
            let rowValues = B == 1 ? values : values[i ..< (i + 1)]

            let (k, v) = row.update(keys: rowKeys, values: rowValues)

            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if L == 1 {
                mask = .none
                maskModeLog.append("none")
            } else {
                mask = .array(prefillMask(chunkLength: L, retainedAfterUpdate: k.dim(2)))
                maskModeLog.append("array")
            }

            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: rowQueries, keys: k, values: v,
                    scale: scale, mask: mask, sinks: sinks))
        }
        return B == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    func attendBorrowing(
        source: CBv2AttendingLayerCache,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(kind.sharesKVWithLayer != nil, "attendBorrowing on a non-shared layer")
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(B == source.rows.count, "row order mismatch vs source")
        precondition(B == 1 || L == 1)

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for (i, sourceRow) in source.rows.enumerated() {
            guard let readable = sourceRow as? HarnessReadableKV,
                let (k, v) = readable.currentViews()
            else {
                fatalError("attendBorrowing: source row \(i) has no retained KV")
            }
            let rowQueries = B == 1 ? queries : queries[i ..< (i + 1)]

            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if L == 1 {
                mask = .none
                maskModeLog.append("none")
            } else {
                mask = .array(prefillMask(chunkLength: L, retainedAfterUpdate: k.dim(2)))
                maskModeLog.append("array")
            }

            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: rowQueries, keys: k, values: v,
                    scale: scale, mask: mask, sinks: sinks))
        }
        return B == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }
}

// MARK: - Mock KV backend (WS-A/WS-C own the real ones)

/// Mock `CBv2KVBackend` producing the harness sequence KVs. Enforces static
/// sink eligibility (report 10 §4 invariant 9) and truthful byte accounting.
final class HarnessKVBackend: CBv2KVBackend {
    /// Whether this backend can honor attention sinks. When false, building
    /// state for a model whose layer kinds declare sinks MUST throw.
    let supportsSinks: Bool
    let bytesCapacity: Int

    private var allocations: [ObjectIdentifier: [CBv2SequenceKV?]] = [:]

    init(supportsSinks: Bool = true, bytesCapacity: Int = 1 << 30) {
        self.supportsSinks = supportsSinks
        self.bytesCapacity = bytesCapacity
    }

    var bytesInUse: Int {
        allocations.values.reduce(0) { total, state in
            total + state.reduce(0) { $0 + ($1?.byteCount ?? 0) }
        }
    }

    private func check(layerKinds: [CBv2LayerKind]) throws {
        if !supportsSinks, layerKinds.contains(where: { $0.hasSinks }) {
            throw CBv2KVError.backendIneligible(
                reason: "backend does not support attention sinks")
        }
    }

    private func track(_ state: [CBv2SequenceKV?]) {
        if let anchor = state.compactMap({ $0 }).first {
            allocations[ObjectIdentifier(anchor)] = state
        }
    }

    func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try check(layerKinds: layerKinds)
        let state: [CBv2SequenceKV?] = layerKinds.map { kind in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .full:
                return HarnessFullSequenceKV()
            case .slidingWindow(let w):
                return HarnessWindowedSequenceKV(window: w)
            }
        }
        track(state)
        return state
    }

    func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try check(layerKinds: layerKinds)
        precondition(prefix.count == layerKinds.count)
        // Uniform adopted offset (the engine slices the matched prefix by
        // cbv2RequiredRecompute before adopting, so all non-nil entries
        // agree).
        let matched = prefix.compactMap { $0?.offset }.first ?? 0
        let state: [CBv2SequenceKV?] = zip(layerKinds, prefix).map { kind, donated in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .full:
                let kv = HarnessFullSequenceKV()
                if let donated {
                    _ = kv.update(keys: donated.keys, values: donated.values)
                }
                return kv
            case .slidingWindow(let w):
                // Windowed layers never adopt full-history prefixes — the
                // engine replays the trailing tokens (contract:
                // CBv2PrefixCache.lookup returns nil for windowed layers).
                precondition(donated == nil, "windowed layer must not adopt a prefix")
                let kv = HarnessWindowedSequenceKV(window: w)
                if matched > 0 { kv.fastForward(to: matched) }
                return kv
            }
        }
        track(state)
        return state
    }

    func release(_ state: [CBv2SequenceKV?]) {
        if let anchor = state.compactMap({ $0 }).first {
            allocations.removeValue(forKey: ObjectIdentifier(anchor))
        }
    }
}

// MARK: - TinyTestModel

struct TinyTestModelConfig {
    var hiddenSize = 64
    var queryHeads = 4
    var kvHeads = 2
    var headDim = 16
    var vocabSize = 128
    var windowSize = 16
    var mlpSize = 128
    var withSinks = false
    /// Adds a THIRD block that KV-shares with the sliding-window layer 1
    /// (Gemma-4 style: projects Q only, borrows the source layer's K/V via
    /// `attendBorrowing`) plus a FOURTH full-attention block AFTER it, so
    /// the borrowing layer's prefill outputs feed persistent downstream KV
    /// (Gemma-4 interleaves shared layers between storage-owning ones — a
    /// borrowing bug must corrupt decode, not just discarded prefill
    /// logits). v2 path only — the legacy path traps.
    var withKVSharing = false
    /// Makes layer 1 FULL attention instead of sliding-window(16), so the
    /// model has no windowed layers at all (prefix adoption then requires
    /// zero trailing recompute; every layer's KV can be quantized/donated).
    /// Mutually exclusive with `withKVSharing` (which wires layer 1 as a
    /// sliding-window KV-share source).
    var fullAttentionOnly = false
    /// Wires TWO stacked sliding-window layers (1, 2) followed by a
    /// DOWNSTREAM full-attention layer (3): [full, sliding(W), sliding(W),
    /// full]. The two stacked windowed layers compound the prefix-adoption
    /// receptive field, and the trailing full layer CACHES their polluted
    /// early-replay outputs — so a single-window trailing recompute drifts
    /// the retained region (Codex P1). Mutually exclusive with
    /// `withKVSharing` / `fullAttentionOnly`.
    var stackedSlidingFull = false
    var ropeBase: Float = 10_000
    var rmsNormEps: Float = 1e-5

    var scale: Float { 1.0 / Float(headDim).squareRoot() }
}

/// One attention layer of the tiny fixture model. Shared verbatim between the
/// legacy path and the v2 path (same weights, same RoPE, same scale).
final class TinyAttention: Module {
    let config: TinyTestModelConfig
    let isSliding: Bool

    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear
    let rope: RoPE
    /// GPT-OSS style learned per-query-head sink logits (nil ⇒ no sinks).
    let sinks: MLXArray?

    init(config: TinyTestModelConfig, isSliding: Bool) {
        self.config = config
        self.isSliding = isSliding
        self.qProj = Linear(config.hiddenSize, config.queryHeads * config.headDim, bias: false)
        self.kProj = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        self.vProj = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        self.oProj = Linear(config.queryHeads * config.headDim, config.hiddenSize, bias: false)
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeBase)
        // Sink activity is a static model property — no `.item()` probe in
        // any hot path (report 10 §4 invariant 9, "avoid per-request host
        // readbacks like sinksActive .item()").
        self.sinks = config.withSinks ? MLXRandom.normal([config.queryHeads]) * 0.5 : nil
        super.init()
    }

    func project(_ x: MLXArray) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped(B, L, -1, config.headDim).swappedAxes(1, 2)
        let k = kProj(x).reshaped(B, L, -1, config.headDim).swappedAxes(1, 2)
        let v = vProj(x).reshaped(B, L, -1, config.headDim).swappedAxes(1, 2)
        return (q, k, v)
    }

    /// Legacy path: cache-managed attention (old engine, `KVCache` protocol).
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var (q, k, v) = project(x)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: config.scale, mask: mask, sinks: sinks)
        return oProj(out.swappedAxes(1, 2).reshaped(B, L, -1))
    }

    /// v2 path: attention owned by the layer cache object.
    func forwardV2(_ x: MLXArray, cache: CBv2AttendingLayerCache) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var (q, k, v) = project(x)

        // Per-row absolute RoPE offsets, captured BEFORE updateAndAttend
        // advances the per-row counters (snapshot semantics, invariant 1).
        let offsets = cache.positionOffsets
        q = rope(q, offset: offsets)
        k = rope(k, offset: offsets)

        let out = cache.updateAndAttend(
            queries: q, keys: k, values: v,
            scale: config.scale, sinks: sinks)
        return oProj(out.swappedAxes(1, 2).reshaped(B, L, -1))
    }

    /// Independent full-sequence reference attention (multimodal harness):
    /// process the WHOLE sequence at once (offset 0) against an explicit
    /// boolean mask, with no KV cache. Non-shared layers project + RoPE
    /// their own K/V; shared layers project Q only and borrow the passed
    /// (already-RoPE'd) source K/V. Returns (output, keys, values) so a
    /// downstream shared layer can borrow. Used ONLY by the vision reference
    /// — it never touches the engine's span-mask code.
    func referenceForward(
        _ x: MLXArray, maskArray: MLXArray, sharedKV: (keys: MLXArray, values: MLXArray)?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(B, L, -1, config.headDim).swappedAxes(1, 2)
        q = rope(q, offset: 0)
        let k: MLXArray
        let v: MLXArray
        if let sharedKV {
            k = sharedKV.keys
            v = sharedKV.values
        } else {
            var kk = kProj(x).reshaped(B, L, -1, config.headDim).swappedAxes(1, 2)
            kk = rope(kk, offset: 0)
            k = kk
            v = vProj(x).reshaped(B, L, -1, config.headDim).swappedAxes(1, 2)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: config.scale, mask: .array(maskArray), sinks: sinks)
        return (oProj(out.swappedAxes(1, 2).reshaped(B, L, -1)), k, v)
    }

    /// v2 path for a KV-SHARED layer (Gemma-4 style): project Q only, RoPE
    /// with the SOURCE layer's PRE-update offsets (a KV-shared cache owns no
    /// rows, so its own `positionOffsets` is empty — the
    /// `gemma4CapturePositionOffset` discipline), and borrow the source K/V.
    func forwardV2Borrowing(
        _ x: MLXArray, cache: CBv2AttendingLayerCache,
        source: CBv2AttendingLayerCache, sourceOffsets: MLXArray
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let q = rope(
            qProj(x).reshaped(B, L, -1, config.headDim).swappedAxes(1, 2),
            offset: sourceOffsets)
        let out = cache.attendBorrowing(
            source: source, queries: q, scale: config.scale, sinks: sinks)
        return oProj(out.swappedAxes(1, 2).reshaped(B, L, -1))
    }
}

final class TinyBlock: Module {
    let attention: TinyAttention
    let attnNorm: RMSNorm
    let mlpNorm: RMSNorm
    let mlpUp: Linear
    let mlpDown: Linear

    init(config: TinyTestModelConfig, isSliding: Bool) {
        self.attention = TinyAttention(config: config, isSliding: isSliding)
        self.attnNorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.mlpNorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.mlpUp = Linear(config.hiddenSize, config.mlpSize, bias: false)
        self.mlpDown = Linear(config.mlpSize, config.hiddenSize, bias: false)
        super.init()
    }

    private func mlp(_ x: MLXArray) -> MLXArray {
        mlpDown(silu(mlpUp(x)))
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var h = x + attention(attnNorm(x), mask: mask, cache: cache)
        h = h + mlp(mlpNorm(h))
        return h
    }

    func forwardV2(_ x: MLXArray, cache: CBv2AttendingLayerCache) -> MLXArray {
        var h = x + attention.forwardV2(attnNorm(x), cache: cache)
        h = h + mlp(mlpNorm(h))
        return h
    }

    func forwardV2Borrowing(
        _ x: MLXArray, cache: CBv2AttendingLayerCache,
        source: CBv2AttendingLayerCache, sourceOffsets: MLXArray
    ) -> MLXArray {
        var h =
            x
            + attention.forwardV2Borrowing(
                attnNorm(x), cache: cache, source: source, sourceOffsets: sourceOffsets)
        h = h + mlp(mlpNorm(h))
        return h
    }

    /// Full-sequence reference block (vision harness): residual + MLP around
    /// `TinyAttention.referenceForward`. Returns the block output and this
    /// layer's (keys, values) for a downstream shared layer to borrow.
    func referenceForward(
        _ x: MLXArray, maskArray: MLXArray, sharedKV: (keys: MLXArray, values: MLXArray)?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let (out, keys, values) = attention.referenceForward(
            attnNorm(x), maskArray: maskArray, sharedKV: sharedKV)
        var h = x + out
        h = h + mlp(mlpNorm(h))
        return (h, keys, values)
    }
}

/// The fixture model. Layer 0 = full attention, layer 1 = sliding window(16).
final class TinyTestModel: Module, LanguageModel, KVCacheDimensionProvider,
    CBv2SteppableModel
{
    let config: TinyTestModelConfig
    let embed: Embedding
    let blocks: [TinyBlock]
    let finalNorm: RMSNorm
    let lmHead: Linear

    var vocabularySize: Int { config.vocabSize }
    var kvHeads: [Int] { blocks.map { _ in config.kvHeads } }

    /// Model structure as data (report 10 §4 invariant 11).
    var layerKinds: [CBv2LayerKind] {
        var kinds = [
            CBv2LayerKind(
                attention: .full, hasSinks: config.withSinks,
                headDim: config.headDim, kvHeads: config.kvHeads,
                queryHeads: config.queryHeads),
            CBv2LayerKind(
                attention: config.fullAttentionOnly
                    ? .full : .slidingWindow(config.windowSize),
                hasSinks: config.withSinks,
                headDim: config.headDim, kvHeads: config.kvHeads,
                queryHeads: config.queryHeads),
        ]
        if config.withKVSharing {
            kinds.append(
                CBv2LayerKind(
                    attention: .slidingWindow(config.windowSize), sharesKVWithLayer: 1,
                    hasSinks: config.withSinks,
                    headDim: config.headDim, kvHeads: config.kvHeads,
                    queryHeads: config.queryHeads))
            kinds.append(
                CBv2LayerKind(
                    attention: .full, hasSinks: config.withSinks,
                    headDim: config.headDim, kvHeads: config.kvHeads,
                    queryHeads: config.queryHeads))
        }
        if config.stackedSlidingFull {
            // Second stacked sliding layer, then a downstream full layer.
            kinds.append(
                CBv2LayerKind(
                    attention: .slidingWindow(config.windowSize),
                    hasSinks: config.withSinks,
                    headDim: config.headDim, kvHeads: config.kvHeads,
                    queryHeads: config.queryHeads))
            kinds.append(
                CBv2LayerKind(
                    attention: .full, hasSinks: config.withSinks,
                    headDim: config.headDim, kvHeads: config.kvHeads,
                    queryHeads: config.queryHeads))
        }
        return kinds
    }

    /// Build with deterministic seeded weights. Same seed ⇒ same weights.
    /// `headDim` override: the paged Metal kernel supports {64, 128, 256,
    /// 512} only, so paged end-to-end runs use headDim 64.
    static func make(
        seed: UInt64 = 0xC0FFEE, withSinks: Bool = false, headDim: Int = 16,
        withKVSharing: Bool = false, fullAttentionOnly: Bool = false,
        stackedSlidingFull: Bool = false, windowSize: Int? = nil
    ) -> TinyTestModel {
        MLXRandom.seed(seed)
        var config = TinyTestModelConfig()
        config.withSinks = withSinks
        config.headDim = headDim
        config.withKVSharing = withKVSharing
        config.fullAttentionOnly = fullAttentionOnly
        config.stackedSlidingFull = stackedSlidingFull
        if let windowSize { config.windowSize = windowSize }
        let model = TinyTestModel(config: config)
        // Force lazy parameter materialization deterministically.
        eval(model)
        return model
    }

    init(config: TinyTestModelConfig) {
        precondition(
            !(config.fullAttentionOnly && config.withKVSharing),
            "fullAttentionOnly is mutually exclusive with withKVSharing")
        precondition(
            !(config.stackedSlidingFull && (config.withKVSharing || config.fullAttentionOnly)),
            "stackedSlidingFull is mutually exclusive with withKVSharing / fullAttentionOnly")
        self.config = config
        self.embed = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        var blocks = [
            TinyBlock(config: config, isSliding: false),
            TinyBlock(config: config, isSliding: !config.fullAttentionOnly),
        ]
        if config.withKVSharing {
            // KV-shared sliding block (borrows layer 1's K/V at attention),
            // then a storage-owning full block downstream of it.
            blocks.append(TinyBlock(config: config, isSliding: true))
            blocks.append(TinyBlock(config: config, isSliding: false))
        }
        if config.stackedSlidingFull {
            // A SECOND stacked sliding block, then a downstream full block:
            // [full, sliding, sliding, full].
            blocks.append(TinyBlock(config: config, isSliding: true))
            blocks.append(TinyBlock(config: config, isSliding: false))
        }
        self.blocks = blocks
        self.finalNorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.lmHead = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    // MARK: Legacy path (`LanguageModel`, old engine)

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult
    {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        precondition(
            !config.withKVSharing,
            "TinyTestModel: the KV-sharing variant is v2-only (legacy KVCache path unsupported)")
        var h = embed(inputs)
        let caches: [KVCache?] = cache ?? [KVCache?](repeating: nil, count: blocks.count)
        let n = inputs.dim(1)

        for (i, block) in blocks.enumerated() {
            let mask = makeAttentionMask(
                n: n,
                cache: caches[i],
                windowSize: block.attention.isSliding ? config.windowSize : nil
            )
            h = block(h, mask: mask, cache: caches[i])
        }
        return lmHead(finalNorm(h))
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        precondition(
            !config.withKVSharing,
            "TinyTestModel: the KV-sharing variant is v2-only (legacy KVCache path unsupported)")
        return [
            KVCacheSimple(),
            config.fullAttentionOnly
                ? KVCacheSimple()
                : RotatingKVCache(maxSize: config.windowSize, keep: 0),
        ]
    }

    // MARK: v2 path (`CBv2SteppableModel`)

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        forwardV2(hidden: embed(tokens), caches: caches)
    }

    /// Shared v2 trunk from a pre-layer-0 hidden state (the token path
    /// embeds; the multimodal path passes spliced embeddings). One code
    /// path ⇒ the text behavior cannot drift.
    private func forwardV2(hidden: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        precondition(caches.count == blocks.count)
        var h = hidden
        // KV-shared layers reuse the SOURCE layer's PRE-update position
        // offsets — capture before the source layer's updateAndAttend
        // advances them (the gemma4CapturePositionOffset discipline).
        let sharedSourceOffsets: [Int: MLXArray] = Dictionary(
            uniqueKeysWithValues: layerKinds.enumerated().compactMap { index, kind in
                kind.sharesKVWithLayer.map { source in (index, caches[source].positionOffsets) }
            })
        for (i, block) in blocks.enumerated() {
            if let source = layerKinds[i].sharesKVWithLayer {
                h = block.forwardV2Borrowing(
                    h, cache: caches[i], source: caches[source],
                    sourceOffsets: sharedSourceOffsets[i]!)
            } else {
                h = block.forwardV2(h, cache: caches[i])
            }
        }
        return lmHead(finalNorm(h))
    }
}

/// Vision-like fixture surface: TinyTestModel has no embedding scale, so the
/// spliceable pre-layer-0 hidden state is `embed(tokens)` directly; the
/// multimodal forward runs the SAME v2 trunk from the spliced embeddings.
extension TinyTestModel: CBv2MultimodalSteppableModel {
    func embedPromptTokens(_ tokens: MLXArray) -> MLXArray {
        embed(tokens)
    }

    func forward(
        tokens: MLXArray, inputEmbeddings: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        forwardV2(hidden: inputEmbeddings, caches: caches)
    }
}

/// Opt the fixture model into compiled [B, 1] decode: its v2 forward drives
/// the layer caches purely through the contract surface, so EngineV2 tests
/// exercise the compiled path with the default configuration (eager
/// fallback still covers mock backends whose rows are not contiguous).
extension TinyTestModel: CBv2CompiledSteppableModel {}

// MARK: - v2 harness engine (test-local greedy step loop)

/// Minimal deterministic v2 driver: per-request chunked prefill ([1, chunk]),
/// rectangular [B, 1] greedy decode, join/leave between steps. This is a TEST
/// harness, not WS-B's engine — it exists so the invariance/parity suites are
/// executable against the contract before integration.
final class CBv2HarnessEngine {
    struct RequestState {
        let id: CBv2RequestID
        let promptTokens: [Int]
        var generated: [Int] = []
        var kv: [CBv2SequenceKV?]
        var lastToken: Int
        var maxTokens: Int
        var finished: Bool { generated.count >= maxTokens }
    }

    let model: TinyTestModel
    let backend: HarnessKVBackend
    let prefillChunkSize: Int
    private(set) var layerCaches: [HarnessLayerCache]
    private(set) var active: [RequestState] = []
    private var nextID: UInt64 = 0

    /// Wall-clock decode step timings for benchmark-style assertions.
    private(set) var stepDurations: [TimeInterval] = []

    init(
        model: TinyTestModel,
        backend: HarnessKVBackend? = nil,
        prefillChunkSize: Int = 16
    ) {
        self.model = model
        self.backend = backend ?? HarnessKVBackend()
        self.prefillChunkSize = prefillChunkSize
        self.layerCaches = model.layerKinds.enumerated().map { i, kind in
            HarnessLayerCache(layerIndex: i, kind: kind)
        }
    }

    /// Membership change: rebuild the per-layer row lists in row order.
    private func rebuildRows() {
        for (i, cache) in layerCaches.enumerated() {
            cache.rows = active.map { request in
                guard let kv = request.kv[i] else {
                    fatalError("layer \(i) has no storage (KV-shared layers unsupported here)")
                }
                return kv
            }
        }
    }

    /// Admit a request: allocate per-request KV, run chunked prefill
    /// [1, chunk] over prompt[0 ..< n-1] against ITS OWN caches only
    /// (batchmates untouched), then join the decode batch with the LAST
    /// prompt token as the pending step token. The first generated token is
    /// sampled on the next decodeStep() — same step discipline as the legacy
    /// engine (Scheduler prefills n-1 tokens, the last prompt token goes
    /// through the decode step), which keeps every kernel invocation
    /// shape-identical between the two engines (bit-exact parity) AND puts
    /// the new row's first forward inside the batched step, exactly where
    /// the v1 bug class lived.
    @discardableResult
    func join(prompt: [Int], maxTokens: Int) throws -> CBv2RequestID {
        precondition(!prompt.isEmpty, "prompt must be non-empty")
        let id = CBv2RequestID(nextID)
        nextID += 1

        let kv = try backend.makeSequenceState(
            layerKinds: model.layerKinds,
            promptLength: prompt.count,
            maxLength: prompt.count + maxTokens)

        // Per-request prefill caches: rows contain ONLY this request.
        let prefillCaches = model.layerKinds.enumerated().map { i, kind in
            HarnessLayerCache(layerIndex: i, kind: kind, rows: [kv[i]!])
        }

        let prefillTokens = Array(prompt.dropLast())
        var index = 0
        while index < prefillTokens.count {
            let chunk = Array(
                prefillTokens[index ..< min(index + prefillChunkSize, prefillTokens.count)])
            let tokens = MLXArray(chunk.map { Int32($0) }).reshaped(1, chunk.count)
            _ = model.forward(tokens: tokens, caches: prefillCaches)
            index += chunk.count
        }

        let request = RequestState(
            id: id, promptTokens: prompt, kv: kv, lastToken: prompt.last!,
            maxTokens: maxTokens)
        active.append(request)
        rebuildRows()
        return id
    }

    /// Remove a request (finish or cancel); frees its KV, O(1) membership edit.
    func remove(id: CBv2RequestID) {
        guard let index = active.firstIndex(where: { $0.id == id }) else { return }
        backend.release(active[index].kv)
        active.remove(at: index)
        rebuildRows()
    }

    /// Requests that reached maxTokens (caller decides when to remove them).
    var finishedIDs: [CBv2RequestID] {
        active.filter(\.finished).map(\.id)
    }

    /// One rectangular [B, 1] greedy decode step for every active row.
    /// Returns the tokens appended this step, in row order.
    @discardableResult
    func decodeStep() -> [Int] {
        precondition(!active.isEmpty, "decodeStep with no active requests")
        let started = CFAbsoluteTimeGetCurrent()

        let tokens = MLXArray(active.map { Int32($0.lastToken) }).reshaped(active.count, 1)
        let logits = model.forward(tokens: tokens, caches: layerCaches)

        // One host readback per step (the sampled tokens) — the boundary sync.
        let sampled = argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self).map(Int.init)
        CBv2HarnessSyncCounter.boundarySyncs += 1

        for (i, token) in sampled.enumerated() {
            active[i].generated.append(token)
            active[i].lastToken = token
        }
        stepDurations.append(CFAbsoluteTimeGetCurrent() - started)
        return sampled
    }

    func generated(for id: CBv2RequestID) -> [Int]? {
        active.first(where: { $0.id == id })?.generated
    }

    /// Convenience: run a single request solo to completion and return its
    /// greedy output tokens.
    static func runSolo(
        model: TinyTestModel, prompt: [Int], maxTokens: Int, prefillChunkSize: Int = 16
    ) throws -> [Int] {
        let engine = CBv2HarnessEngine(model: model, prefillChunkSize: prefillChunkSize)
        let id = try engine.join(prompt: prompt, maxTokens: maxTokens)
        while !(engine.active.first(where: { $0.id == id })?.finished ?? true) {
            engine.decodeStep()
        }
        return engine.generated(for: id) ?? []
    }
}

// MARK: - Deterministic prompt generation

/// SplitMix64 — deterministic cross-platform PRNG for prompt fixtures and
/// churn scripts (never uses the system RNG, so failures reproduce exactly).
struct CBv2HarnessRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { self.state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

func makePromptTokens(length: Int, seed: UInt64, vocabSize: Int = 128) -> [Int] {
    var rng = CBv2HarnessRNG(seed: seed)
    return (0 ..< length).map { _ in Int.random(in: 0 ..< vocabSize, using: &rng) }
}
