// CompiledDecodeV2.swift
//
// Whole-step compiled [B, 1] decode for ContinuousBatchingV2 — one MLX
// `compile(inputs:outputs:)` graph per batch-size bucket, traced at engine
// build (pre-warm; NEVER on the request path) and replayed for every
// pure-decode step whose rows fit the shape discipline.
//
// ## How churn survival works (the legacy engine's fatal flaw, fixed)
// `compile(inputs:outputs:)` re-reads `innerState()` from the registered
// containers at EVERY call, so the hidden graph inputs are whatever arrays
// are bound at that moment. Batch membership change within a bucket is a
// lane REBIND (store different array references + upload two int scalars per
// lane) — no copy, no retrace, no `syncFromCompiled`: the request rows' own
// buffers ARE the compiled state, mutated in place via `_updateInternal`,
// so eager readers (snapshot/donation/rollback/eager fallback) always see
// current KV. Bucket change = switch to another pre-traced function. Any
// unsupported shape (quantized rows, paged rows, row past `kvCapacity`,
// B beyond the ladder) falls back to the eager v2 step — and compiled mode
// re-establishes automatically on the next eligible step (self-healing via
// host offset mirrors, never a permanent invalidation).
//
// ## Dummy lanes
// A bucket bigger than the live batch pads with SCRATCH lanes: private
// `CBv2FullSequenceKV`/`CBv2WindowedSequenceKV` rows owned here, never the
// real caches. Every lane has identical graph structure (the trace cannot
// know which lanes are real), so inertness is purely a matter of binding:
// scratch writes land in scratch storage and padded logits rows are sliced
// off before sampling. Scratch full-attention lanes are recycled before
// their offset can reach the buffer capacity.

import Foundation
import MLX
import os

private let log = Logger(subsystem: "darkbloom", category: "CBv2CompiledDecode")

/// Compiled-decode step executor. Engine-thread confined (built at engine
/// construction, warmed and stepped on the engine queue only).
public final class CBv2CompiledDecode {

    // MARK: - Types

    private enum LaneBinding {
        case unbound
        case scratch
        case row(ObjectIdentifier)
    }

    private final class Bucket {
        let size: Int
        let counters: CBv2CompiledLaneCounters
        let statics: CBv2CompiledLaneStatics
        let caches: [CBv2CompiledLayerCache]
        let fn: @Sendable ([MLXArray]) -> [MLXArray]
        var bound: [LaneBinding]
        /// Strong references to bound row states (real and scratch) so an
        /// ObjectIdentifier can never alias a deallocated row.
        var boundRows: [[CBv2SequenceKV?]?]

        init(
            size: Int, counters: CBv2CompiledLaneCounters, statics: CBv2CompiledLaneStatics,
            caches: [CBv2CompiledLayerCache], fn: @escaping @Sendable ([MLXArray]) -> [MLXArray]
        ) {
            self.size = size
            self.counters = counters
            self.statics = statics
            self.caches = caches
            self.fn = fn
            self.bound = Array(repeating: .unbound, count: size)
            self.boundRows = Array(repeating: nil, count: size)
        }
    }

    private enum State {
        case pending
        case ready
        case disabled(String)
    }

    // MARK: - Stored

    let config: CBv2CompiledDecodeConfig
    let layerKinds: [CBv2LayerKind]
    private let model: CBv2SteppableModel
    /// Distinct sliding windows, ascending (mask statics layout).
    private let windows: [Int]
    private var buckets: [Bucket] = []
    private var state: State = .pending
    /// Per-layer KV dtypes the traces are specialized on (probed at warmup
    /// from a real forward — binding rejects rows whose buffers differ, so
    /// a replay can never silently retrace on the request path). K and V are
    /// probed and specialized INDEPENDENTLY: a layer can cache K and V at
    /// different precisions, and the trace binds each accordingly (PR#62
    /// review). Layers CAN also disagree with each other: GPT-OSS
    /// full-attention layers cache float32 K/V (YarnRoPE computes in fp32)
    /// while its sliding layers cache bfloat16. nil for KV-shared layers
    /// (no storage).
    private var layerDTypes: [(keys: DType, values: DType)?] = []

    /// Telemetry. Mutated on the engine thread; read from arbitrary threads
    /// (provider heartbeats, tests) — lock-protected snapshot semantics.
    var stats: CBv2CompiledDecodeStats {
        statsLock.lock()
        defer { statsLock.unlock() }
        return _stats
    }

    private let statsLock = NSLock()
    private var _stats = CBv2CompiledDecodeStats()

    private func mutateStats(_ body: (inout CBv2CompiledDecodeStats) -> Void) {
        statsLock.lock()
        body(&_stats)
        statsLock.unlock()
    }

    // MARK: - Build

    /// Build the compiled-decode executor, or nil when statically
    /// ineligible. `nil` is normal — the engine simply stays eager.
    ///
    /// `kvBytesCapacity` is the backend's byte budget: the compiled path
    /// pads full-attention rows to `kvCapacity` slots (beyond what token
    /// admission budgets), so a worst-case padding reserve is computed here
    /// and — when the executor builds — the engine carves it out of the
    /// admission ledger (`admissionPaddingReserve`). If the reserve would
    /// eat half the budget, compiled decode refuses to build on this box.
    public static func build(
        model: CBv2SteppableModel,
        layerKinds: [CBv2LayerKind],
        config: CBv2CompiledDecodeConfig,
        maxConcurrentRequests: Int,
        kvBytesCapacity: Int = .max
    ) -> CBv2CompiledDecode? {
        guard config.enabled else { return nil }
        guard CBv2CompiledDecodeConfig.envEnabled else { return nil }
        guard MLXHardwareInfo.isCompiledDecodeSupported else {
            log.info("CBv2 compiled decode skipped: hardware not supported")
            return nil
        }
        guard model is CBv2CompiledSteppableModel else {
            log.info("CBv2 compiled decode skipped: model does not opt in")
            return nil
        }
        guard config.attentionSoftcap == nil else {
            log.info("CBv2 compiled decode skipped: attention softcap not supported")
            return nil
        }
        guard config.kvCapacity > 0 else { return nil }
        let buckets = Array(Set(config.buckets.filter { $0 >= 1 && $0 <= maxConcurrentRequests }))
            .sorted()
        guard !buckets.isEmpty else { return nil }
        guard !layerKinds.isEmpty else { return nil }
        let reserve = paddingReserveBytes(
            layerKinds: layerKinds, kvCapacity: config.kvCapacity,
            bucketSizes: buckets, maxConcurrentRequests: maxConcurrentRequests)
        guard reserve <= kvBytesCapacity / 2 else {
            log.info(
                "CBv2 compiled decode skipped: padding reserve \(reserve >> 20) MiB exceeds half of the \(kvBytesCapacity >> 20) MiB KV budget (lower kvCapacity or raise the budget)"
            )
            return nil
        }
        return CBv2CompiledDecode(
            model: model, layerKinds: layerKinds, config: config, bucketSizes: buckets,
            admissionPaddingReserve: reserve)
    }

    /// Worst-case EXTRA bytes the compiled path may hold beyond what token
    /// admission budgets: every concurrent request's full-attention buffers
    /// padded to `kvCapacity` slots, plus scratch lanes (full padding AND
    /// window rings — scratch rows are invisible to admission entirely).
    /// 4 bytes/element worst case: GPT-OSS full-attention layers cache fp32
    /// K/V. Deliberately conservative — the reserve is subtracted from the
    /// admission ledger so real usage can never silently exceed the
    /// backend's byte budget (review P1).
    static func paddingReserveBytes(
        layerKinds: [CBv2LayerKind], kvCapacity: Int,
        bucketSizes: [Int], maxConcurrentRequests: Int
    ) -> Int {
        let bytesPerElement = 4
        var fullPerRow = 0
        var windowedPerRow = 0
        for kind in layerKinds where kind.sharesKVWithLayer == nil {
            switch kind.attention {
            case .full:
                fullPerRow += 2 * kind.kvHeads * kvCapacity * kind.headDim * bytesPerElement
            case .slidingWindow(let window):
                windowedPerRow += 2 * kind.kvHeads * window * kind.headDim * bytesPerElement
            }
        }
        // Padded lanes a bucket can hold beyond the smallest batch it
        // serves (B in (prevBucket, size] ⇒ up to size - prev - 1 scratch).
        var scratchLanes = 0
        var previous = 0
        for size in bucketSizes {
            scratchLanes += max(0, size - previous - 1)
            previous = size
        }
        return maxConcurrentRequests * fullPerRow
            + scratchLanes * (fullPerRow + windowedPerRow)
    }

    /// Worst-case padding bytes the engine must carve out of the admission
    /// ledger while this executor exists (see `paddingReserveBytes`).
    public let admissionPaddingReserve: Int

    private init(
        model: CBv2SteppableModel, layerKinds: [CBv2LayerKind],
        config: CBv2CompiledDecodeConfig, bucketSizes: [Int],
        admissionPaddingReserve: Int
    ) {
        self.model = model
        self.layerKinds = layerKinds
        self.config = config
        self.admissionPaddingReserve = admissionPaddingReserve
        var windowSet = Set<Int>()
        for kind in layerKinds where kind.sharesKVWithLayer == nil {
            if case .slidingWindow(let window) = kind.attention { windowSet.insert(window) }
        }
        self.windows = windowSet.sorted()
        self.bucketSizes = bucketSizes
    }

    private let bucketSizes: [Int]

    // MARK: - Warmup (engine queue, before the first step)

    /// True once compiled decode is live. `decodeStep` returns nil (eager
    /// fallback) in every other state.
    public var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    /// Reason compiled decode is off, if it is (build report / telemetry).
    public var disabledReason: String? {
        if case .disabled(let reason) = state { return reason }
        return nil
    }

    /// Trace every bucket. Runs ONCE, on the engine queue, before the first
    /// step (model-load budget). Any MLX failure (a model structure that
    /// resists tracing) disables compiled decode with the exact reason —
    /// the engine keeps serving eagerly.
    func warmupIfNeeded() {
        guard case .pending = state else { return }
        do {
            let probed = try withError { try probeLayerDTypesAndPrime() }
            self.layerDTypes = probed
            for size in bucketSizes {
                let started = CFAbsoluteTimeGetCurrent()
                let bucket = try withError { try traceBucket(size: size) }
                let seconds = CFAbsoluteTimeGetCurrent() - started
                mutateStats { $0.warmup.append((bucket: size, seconds: seconds)) }
                // Drop this bucket's throwaway scratch before tracing the
                // next one: warmup's transient footprint stays bounded by
                // a single bucket's lanes.
                releaseAllLanes(bucket)
                buckets.append(bucket)
                log.info(
                    "CBv2 compiled decode: bucket \(size) traced in \(String(format: "%.2f", seconds))s"
                )
            }
            state = .ready
        } catch {
            let reason = "trace failed: \(error)"
            state = .disabled(reason)
            mutateStats { $0.disabledReason = reason }
            buckets = []
            log.error("CBv2 compiled decode disabled — \(reason)")
        }
    }

    /// One EAGER forward through the production layer-cache path with
    /// throwaway contiguous rows: primes any lazy model-side host probes
    /// (e.g. GPT-OSS `sinksActiveResolved`) OFF the traced region — compile
    /// rejects host syncs inside a trace — and reveals each layer's actual
    /// K AND V dtypes, which the traces must be specialized on independently
    /// (per LAYER, and per K/V within a layer: e.g. GPT-OSS caches fp32 on
    /// full-attention layers, bf16 on sliding ones, and a layer could cache
    /// K and V at different precisions).
    private func probeLayerDTypesAndPrime() throws -> [(keys: DType, values: DType)?] {
        let rows = makeScratchRow(fullCapacity: 8)
        let bank = CBv2LayerCacheBank(layerKinds: layerKinds)
        let caches = bank.layerCaches(rowStates: [rows])
        let logits = model.forward(
            tokens: MLXArray([Int32(0)]).reshaped(1, 1), caches: caches)
        eval(logits)
        guard rows.contains(where: { $0 != nil }) else {
            throw CBv2KVError.backendIneligible(reason: "model owns no KV storage")
        }
        // dtype is graph metadata — no readback. The forward wrote one
        // token into every storage-owning row, so snapshots are real.
        return rows.map { row in
            row.map { seq in
                let snap = seq.snapshot()
                return (keys: snap.keys.dtype, values: snap.values.dtype)
            }
        }
    }

    /// Build + trace one bucket: bind every lane to fresh scratch, compile
    /// the whole forward (+ in-graph offset advance), execute once, eval.
    private func traceBucket(size: Int) throws -> Bucket {
        let counters = CBv2CompiledLaneCounters(lanes: size)
        let statics = CBv2CompiledLaneStatics(lanes: size, windows: windows)
        let caches = layerKinds.enumerated().map { index, kind in
            CBv2CompiledLayerCache(
                layerIndex: index, kind: kind, kvCapacity: config.kvCapacity,
                counters: counters, statics: statics)
        }

        let model = self.model
        let attendingCaches = caches.map { $0 as CBv2AttendingLayerCache }
        let fn = compile(
            inputs: [counters, statics] + caches.map { $0 as any Updatable },
            outputs: [counters] + caches.map { $0 as any Updatable }
        ) { (args: [MLXArray]) -> [MLXArray] in
            let logits = model.forward(tokens: args[0], caches: attendingCaches)
            // Advance AFTER the forward: every in-forward read (RoPE, masks,
            // write positions) observed pre-update offsets.
            counters.advanceAllInGraph()
            return [logits[0..., -1, 0...]]
        }

        let bucket = Bucket(
            size: size, counters: counters, statics: statics, caches: caches, fn: fn)
        for lane in 0 ..< size {
            guard bindScratch(bucket, lane: lane) else {
                throw CBv2KVError.backendIneligible(reason: "scratch bind failed at warmup")
            }
        }
        let tokens = MLXArray.zeros([size, 1], dtype: .int32)
        let out = bucket.fn([tokens])
        guard out.count == 1 else {
            throw CBv2KVError.backendIneligible(reason: "compiled trace returned \(out.count) outputs")
        }
        eval(out[0])
        return bucket
    }

    // MARK: - The compiled step

    /// Execute one pure-decode step through the compiled graph, or return
    /// nil for eager fallback. `rowStates` in batch row order (== plan
    /// order), `tokens` is the `[B, 1] int32` input (lazy is fine — chained
    /// decode feeds the previous step's unevaluated samples).
    ///
    /// On success the returned logits are `[B, vocab]` (padded lanes sliced
    /// off) and all host-side row counters have been advanced.
    public func decodeStep(rowStates: [[CBv2SequenceKV?]], tokens: MLXArray) -> MLXArray? {
        guard case .ready = state else { return nil }
        let B = rowStates.count
        guard B > 0 else { return nil }
        guard let bucket = buckets.first(where: { $0.size >= B }) else {
            recordFallback("batch_\(B)_exceeds_ladder")
            return nil
        }
        precondition(
            tokens.dtype == .int32 && tokens.ndim == 2 && tokens.dim(0) == B && tokens.dim(1) == 1,
            "CBv2CompiledDecode: tokens must be [B, 1] int32")

        // Row eligibility (host-only type/offset checks).
        var infos: [LaneInfo] = []
        infos.reserveCapacity(B)
        for row in rowStates {
            guard let info = laneInfo(for: row) else {
                recordFallback("ineligible_row")
                return nil
            }
            infos.append(info)
        }

        // Bind real lanes (identity + host-offset self-heal; no-ops on the
        // stable chained fast path).
        for (lane, info) in infos.enumerated() {
            if case .row(let identity) = bucket.bound[lane],
                identity == info.identity,
                bucket.counters.hostOffsets[lane] == info.anchorOffset
            {
                continue
            }
            guard bindRow(bucket, lane: lane, info: info) else {
                // Drop the half-updated lane entirely: a partial bind mixes
                // the new row's and the PREVIOUS binding's buffers, and the
                // previous row's forgetRows can no longer match this lane.
                clearLane(bucket, lane: lane)
                recordFallback("bind_failed(\(lastBindFailure ?? "?"))")
                return nil
            }
        }

        // Scratch lanes for the padded remainder, recycled before a
        // full-attention scratch buffer can overflow.
        if B < bucket.size {
            for lane in B ..< bucket.size {
                let needsReset =
                    bucket.counters.hostOffsets[lane] >= config.kvCapacity - 1
                if case .scratch = bucket.bound[lane], !needsReset { continue }
                if case .scratch = bucket.bound[lane], needsReset {
                    mutateStats { $0.scratchResets += 1 }
                }
                guard bindScratch(bucket, lane: lane) else {
                    recordFallback("scratch_bind_failed")
                    return nil
                }
            }
        }

        // Pad tokens to the bucket shape ([bucket, 1]; ids 0 are inert —
        // padded logits rows are sliced off below).
        let stepTokens: MLXArray
        if B == bucket.size {
            stepTokens = tokens
        } else {
            stepTokens = concatenated(
                [tokens, MLXArray.zeros([bucket.size - B, 1], dtype: .int32)], axis: 0)
        }

        let out = bucket.fn([stepTokens])
        guard out.count == 1 else {
            recordFallback("compiled_call_failed")
            return nil
        }

        // Host bookkeeping: mirror the in-graph advances so eager consumers
        // and the next bind check see consistent counters.
        for info in infos {
            for sequence in info.row {
                switch sequence {
                case let full as CBv2FullSequenceKV: full.noteCompiledAdvance()
                case let windowed as CBv2WindowedSequenceKV: windowed.noteCompiledAdvance()
                default: break
                }
            }
        }
        bucket.counters.noteAdvance()
        mutateStats { $0.compiledSteps += 1 }

        let logits = out[0]
        return B == bucket.size ? logits : logits[0 ..< B]
    }

    // MARK: - Lane info & binding

    private struct LaneInfo {
        let row: [CBv2SequenceKV?]
        let identity: ObjectIdentifier
        let anchorOffset: Int
        /// Oldest-valid absolute position per distinct window (ascending
        /// window order, matching `CBv2CompiledLaneStatics.windows`).
        let oldestByWindow: [Int]
    }

    /// Host-only eligibility scan of one row. nil ⇒ this step must run
    /// eager (foreign row types, quantized/paged storage, row too long).
    private func laneInfo(for row: [CBv2SequenceKV?]) -> LaneInfo? {
        guard row.count == layerKinds.count else { return nil }
        var anchor: CBv2SequenceKV?
        var anchorOffset = -1
        var oldestByWindow = [Int: Int]()
        for (index, kind) in layerKinds.enumerated() {
            guard kind.sharesKVWithLayer == nil else { continue }
            guard let sequence = row[index] else { return nil }
            let offset: Int
            switch kind.attention {
            case .full:
                guard let full = sequence as? CBv2FullSequenceKV,
                    full.absoluteOffset < config.kvCapacity
                else { return nil }
                offset = full.absoluteOffset
            case .slidingWindow(let window):
                guard let windowed = sequence as? CBv2WindowedSequenceKV,
                    windowed.window == window
                else { return nil }
                offset = windowed.absoluteOffset
                if let existing = oldestByWindow[window] {
                    // Lockstep advance + uniform adoption make equal-window
                    // layers agree; if they ever do not, stay eager.
                    guard existing == windowed.compiledOldestValidPosition else { return nil }
                } else {
                    oldestByWindow[window] = windowed.compiledOldestValidPosition
                }
            }
            if anchor == nil {
                anchor = sequence
                anchorOffset = offset
            } else if offset != anchorOffset {
                // Layers of one row always advance in lockstep; a mismatch
                // means out-of-band mutation — stay eager, never guess.
                return nil
            }
        }
        guard let anchor else { return nil }
        return LaneInfo(
            row: row,
            identity: ObjectIdentifier(anchor),
            anchorOffset: anchorOffset,
            oldestByWindow: windows.map { oldestByWindow[$0] ?? anchorOffset })
    }

    /// Bind a real request row into a lane (membership change / self-heal).
    private func bindRow(_ bucket: Bucket, lane: Int, info: LaneInfo) -> Bool {
        for (index, kind) in layerKinds.enumerated() {
            guard kind.sharesKVWithLayer == nil, let dtypes = layerDTypes[index] else { continue }
            let storage: (keys: MLXArray, values: MLXArray)?
            switch kind.attention {
            case .full:
                storage = (info.row[index] as? CBv2FullSequenceKV)?
                    .compiledStorage(
                        capacity: config.kvCapacity,
                        keysDType: dtypes.keys, valuesDType: dtypes.values)
            case .slidingWindow:
                storage = (info.row[index] as? CBv2WindowedSequenceKV)?
                    .compiledStorage(keysDType: dtypes.keys, valuesDType: dtypes.values)
            }
            guard let storage else {
                lastBindFailure = bindFailureDiagnostic(
                    row: info.row[index], layer: index, kind: kind)
                return false
            }
            bucket.caches[index].bindLane(lane, keys: storage.keys, values: storage.values)
        }
        bucket.counters.bind(lane: lane, offset: info.anchorOffset)
        bucket.statics.bind(lane: lane, oldestByWindow: info.oldestByWindow)
        bucket.bound[lane] = .row(info.identity)
        bucket.boundRows[lane] = info.row
        mutateStats { $0.laneRebinds += 1 }
        return true
    }

    /// Bind a fresh scratch row into a lane (padding / warmup).
    private func bindScratch(_ bucket: Bucket, lane: Int) -> Bool {
        let row = makeScratchRow(fullCapacity: config.kvCapacity)
        guard let info = laneInfo(for: row) else { return false }
        guard bindRow(bucket, lane: lane, info: info) else { return false }
        bucket.bound[lane] = .scratch
        return true
    }

    /// Fresh per-layer scratch rows (zeros at offset 0). Never registered
    /// with any backend — inert by construction.
    private func makeScratchRow(fullCapacity: Int) -> [CBv2SequenceKV?] {
        layerKinds.map { kind -> CBv2SequenceKV? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .full:
                return CBv2FullSequenceKV(
                    promptLength: 0, maxLength: max(1, fullCapacity),
                    kvHeads: kind.kvHeads, headDim: kind.headDim)
            case .slidingWindow(let window):
                return CBv2WindowedSequenceKV(
                    window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
            }
        }
    }

    private func releaseAllLanes(_ bucket: Bucket) {
        for lane in 0 ..< bucket.size { clearLane(bucket, lane: lane) }
    }

    private func clearLane(_ bucket: Bucket, lane: Int) {
        for cache in bucket.caches { cache.clearLane(lane) }
        bucket.counters.bind(lane: lane, offset: 0)
        bucket.bound[lane] = .unbound
        bucket.boundRows[lane] = nil
    }

    /// Release any lane bindings referencing this retired row state so its
    /// (padded) buffers are not kept alive by an idle bucket. Called by the
    /// engine loop wherever KV state is released (finish, cancel,
    /// preemption). The in-flight step's graph holds its own references, so
    /// clearing immediately is safe even for fenced releases.
    public func forgetRows(_ state: [CBv2SequenceKV?]) {
        guard let anchor = state.compactMap({ $0 }).first else { return }
        let identity = ObjectIdentifier(anchor)
        for bucket in buckets {
            for lane in 0 ..< bucket.size {
                if case .row(let bound) = bucket.bound[lane], bound == identity {
                    clearLane(bucket, lane: lane)
                }
            }
        }
    }

    private func recordFallback(_ reason: String) {
        mutateStats { $0.fallbacks[reason, default: 0] += 1 }
    }

    /// Last bind refusal, in enough detail to diagnose from fleet stats.
    private var lastBindFailure: String?

    /// Stat-key diagnostic for a bind refusal. Deliberately excludes the
    /// row offset so the fallback dictionary's cardinality stays bounded
    /// (layer × kind × dtype); a permanently ineligible row must not mint
    /// a fresh key every step.
    private func bindFailureDiagnostic(
        row: CBv2SequenceKV?, layer: Int, kind: CBv2LayerKind
    ) -> String {
        guard let row else { return "layer\(layer)_missing" }
        let shape: String
        switch row {
        case let full as CBv2FullSequenceKV:
            let snap = full.snapshot()
            shape = "full k=\(snap.keys.dtype) v=\(snap.values.dtype)"
        case let windowed as CBv2WindowedSequenceKV:
            let snap = windowed.snapshot()
            shape = "win k=\(snap.keys.dtype) v=\(snap.values.dtype)"
        default:
            shape = "type=\(type(of: row))"
        }
        let expected =
            layerDTypes[safe: layer].flatMap { $0 }.map { "k=\($0.keys) v=\($0.values)" } ?? "?"
        return "layer\(layer) \(kind.attention) \(shape) expected=\(expected)"
    }
}

extension Array {
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
