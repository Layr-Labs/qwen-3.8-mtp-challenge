// CompiledLayerCacheV2.swift
//
// The per-layer batch-facing cache the model drives INSIDE the compiled
// [B, 1] decode graph. Same `CBv2AttendingLayerCache` surface as the eager
// `CBv2LayerCache`, but every operation is expressed so the MLX compile
// tracer can follow it (the legacy CompilableKVCache "overflow bin"
// pattern, adapted to v2's per-request storage):
//
//  - Each lane's K/V is a FIXED-SHAPE buffer (`kvCapacity` slots for full
//    attention, `window` slots for sliding window) owned by the REQUEST's
//    own `CBv2FullSequenceKV` / `CBv2WindowedSequenceKV` — there is still no
//    shared frontier and no cross-row storage. The buffers are bound into
//    lanes by object reference; `compile` re-reads them every call, so
//    membership churn is a pointer swap, not a copy and not a retrace.
//  - Writes go through `dynamicSliceUpdate` with the lane's offset as graph
//    DATA (`[1] int32` state), then `_updateInternal` preserves the array
//    object identity the compile wrapper tracks.
//  - `update()` returns the FULL buffer; a boolean mask keyed to the lane's
//    pre-update offset marks valid positions. Sliding-window masks compare
//    MODULAR TOKEN INDICES (slot j holds absolute position
//    `o - ((o + w - j) % w)`), keeping the RECENT end — the legacy P1 class
//    of ring-wrap bugs is handled in absolute-position space by
//    construction, and `fastForward`/rollback residue is excluded by the
//    per-lane `oldest-valid` static bound.
//  - Attention is PER LANE (B small SDPAs, not one batched call over shared
//    storage): batch-composition invariance holds inside the compiled graph
//    by construction, exactly like the eager `CBv2AttentionV1` decode path,
//    and a fully-masked lane cannot exist (the just-written token is always
//    valid).
//
// RoPE offsets: `positionOffsets` reads the lane counters, which advance
// in-graph only AFTER the whole forward (`advanceAllInGraph`), so every read
// during the forward observes pre-update absolute positions — models keep
// the exact `+ 0` snapshot discipline they use on the eager path, and
// KV-shared (Gemma-4) layers reusing the SOURCE layer's capture see
// identical values because both read the same per-lane counters.

import Foundation
import MLX

/// Compiled counterpart of `CBv2LayerCache` for decode steps. Engine-thread
/// confined; lanes are bound by `CBv2CompiledDecode`.
final class CBv2CompiledLayerCache: CBv2AttendingLayerCache {

    let layerIndex: Int
    let kind: CBv2LayerKind
    /// Fixed slot count of every lane buffer at this layer: `kvCapacity`
    /// for full attention, the model window for sliding window. 0 for
    /// KV-shared layers (no storage).
    let capacity: Int

    private let counters: CBv2CompiledLaneCounters
    private let statics: CBv2CompiledLaneStatics

    /// Per-lane K/V buffer objects (empty for KV-shared layers). These are
    /// the bound rows' OWN arrays — `innerState()` hands them to the
    /// compile wrapper as hidden inputs/outputs.
    private(set) var laneKeys: [MLXArray] = []
    private(set) var laneValues: [MLXArray] = []

    /// Column indices `[0, capacity)` — mask constant, traced as a graph
    /// constant (never state).
    private let cols: MLXArray

    init(
        layerIndex: Int, kind: CBv2LayerKind, kvCapacity: Int,
        counters: CBv2CompiledLaneCounters, statics: CBv2CompiledLaneStatics
    ) {
        self.layerIndex = layerIndex
        self.kind = kind
        self.counters = counters
        self.statics = statics
        if kind.sharesKVWithLayer != nil {
            self.capacity = 0
            self.cols = MLXArray([Int32]())
        } else {
            switch kind.attention {
            case .full:
                self.capacity = kvCapacity
            case .slidingWindow(let window):
                self.capacity = window
            }
            self.cols = MLXArray(Int32(0) ..< Int32(capacity))
            self.laneKeys = []
            self.laneValues = []
        }
    }

    // MARK: - Lane binding (CBv2CompiledDecode only)

    /// Bind one lane's buffers (the request row's own arrays). Lanes must be
    /// bound densely `0 ..< lanes` before the first step.
    func bindLane(_ lane: Int, keys: MLXArray, values: MLXArray) {
        precondition(kind.sharesKVWithLayer == nil, "KV-shared layers own no lanes")
        precondition(keys.dim(2) == capacity && values.dim(2) == capacity,
            "CBv2CompiledLayerCache: lane buffer shape mismatch (layer \(layerIndex))")
        if lane == laneKeys.count {
            laneKeys.append(keys)
            laneValues.append(values)
        } else {
            precondition(lane < laneKeys.count, "lanes must be bound densely")
            laneKeys[lane] = keys
            laneValues[lane] = values
        }
    }

    /// Drop a lane's buffer references (row finished/preempted) so a
    /// retired request's padded buffers are not kept alive by this bucket.
    /// The lane MUST be rebound before the next compiled call (the
    /// orchestrator marks it unbound).
    func clearLane(_ lane: Int) {
        guard kind.sharesKVWithLayer == nil, lane < laneKeys.count else { return }
        let placeholder = MLXArray(Int32(0))
        laneKeys[lane] = placeholder
        laneValues[lane] = placeholder
    }

    // MARK: - CBv2AttendingLayerCache

    /// Lane binding replaces row binding here; the engine's eager providers
    /// never hand these caches out, so nothing should call this.
    var rows: [CBv2SequenceKV] { [] }

    func setRows(_ rows: [CBv2SequenceKV]) {
        preconditionFailure(
            "CBv2CompiledLayerCache: lanes are bound by CBv2CompiledDecode, not setRows")
    }

    /// Pre-update per-lane absolute offsets `[lanes] int32` (device). The
    /// counters advance in-graph only after the forward, so this is the
    /// pre-update snapshot for every read inside the step.
    var positionOffsets: MLXArray { counters.positionOffsets }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(kind.sharesKVWithLayer == nil,
            "CBv2CompiledLayerCache: KV-shared layer \(layerIndex) must use attendBorrowing")
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(L == 1, "compiled path is decode-only ([B, 1])")
        precondition(B == laneKeys.count,
            "CBv2CompiledLayerCache: batch \(B) != bound lanes \(laneKeys.count)")
        let effectiveSinks = kind.hasSinks ? sinks : nil

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for lane in 0 ..< B {
            let prev = counters.offsets[lane]  // [1] int32, pre-update
            let start: MLXArray
            switch kind.attention {
            case .full:
                start = prev
            case .slidingWindow(let window):
                start = prev % Int32(window)
            }
            laneKeys[lane]._updateInternal(
                dynamicSliceUpdate(
                    laneKeys[lane], update: keys[lane ..< (lane + 1)], start: start, axes: [2]))
            laneValues[lane]._updateInternal(
                dynamicSliceUpdate(
                    laneValues[lane], update: values[lane ..< (lane + 1)], start: start, axes: [2])
            )
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: queries[lane ..< (lane + 1)],
                    keys: laneKeys[lane], values: laneValues[lane],
                    scale: scale,
                    mask: .array(mask(lane: lane, kind: kind)),
                    sinks: effectiveSinks))
        }
        return B == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    func attendBorrowing(
        source: CBv2AttendingLayerCache,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(kind.sharesKVWithLayer != nil,
            "CBv2CompiledLayerCache: attendBorrowing on a storage-owning layer")
        precondition(kind.sharesKVWithLayer == source.layerIndex,
            "CBv2CompiledLayerCache: layer \(layerIndex) shares KV with "
                + "\(kind.sharesKVWithLayer!), not \(source.layerIndex)")
        guard let src = source as? CBv2CompiledLayerCache else {
            preconditionFailure(
                "CBv2CompiledLayerCache: source layer \(source.layerIndex) is not compiled")
        }
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(L == 1, "compiled path is decode-only ([B, 1])")
        precondition(B == src.laneKeys.count,
            "CBv2CompiledLayerCache: batch \(B) != source lanes \(src.laneKeys.count)")
        let effectiveSinks = kind.hasSinks ? sinks : nil

        // Decode borrow attends the source's POST-update state (the source
        // layer already wrote this step's token earlier in the forward);
        // the source's validity mask is keyed to the same pre-update offset,
        // so the visible set == the eager `snapshot()` decode borrow.
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for lane in 0 ..< B {
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: queries[lane ..< (lane + 1)],
                    keys: src.laneKeys[lane], values: src.laneValues[lane],
                    scale: scale,
                    mask: .array(src.mask(lane: lane, kind: src.kind)),
                    sinks: effectiveSinks))
        }
        return B == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    // MARK: - Masks (post-update validity, pre-update offset as data)

    /// Boolean `[1, 1, 1, capacity]` mask over the full lane buffer AFTER
    /// this step's token was written at pre-update offset `o`:
    ///  - full attention: index j valid ⇔ `j <= o`;
    ///  - sliding window w: slot j holds absolute position
    ///    `a_j = o - ((o + w - j) % w)` (`a_j == o` for the just-written
    ///    slot); valid ⇔ `a_j >= max(oldestValid, o + 1 - w)`. Unwritten
    ///    slots (fresh or fast-forwarded lanes) and post-wrap rollback
    ///    residue fall below `oldestValid`; the just-written token always
    ///    satisfies both bounds, so no lane can be fully masked.
    private func mask(lane: Int, kind: CBv2LayerKind) -> MLXArray {
        let prev = counters.offsets[lane]  // [1] int32
        switch kind.attention {
        case .full:
            return (cols .<= prev).reshaped(1, 1, 1, capacity)
        case .slidingWindow(let window):
            let w = Int32(window)
            let slotPositions = prev - (((prev + w) - cols) % w)  // [w] int32
            let oldest = statics.entry(lane: lane, window: window)
            let validFrom = maximum(oldest, prev + Int32(1) - w)  // [1] int32
            return (slotPositions .>= validFrom).reshaped(1, 1, 1, capacity)
        }
    }
}

// MARK: - Legacy KVCache conformance (adapter plumbing only)

extension CBv2CompiledLayerCache: KVCache {
    /// Host-mirror max offset; never a device readback.
    var offset: Int { counters.hostOffsets.max() ?? 0 }

    var maxSize: Int? {
        switch kind.attention {
        case .full: return nil
        case .slidingWindow(let window): return window
        }
    }

    /// The compile wrapper's hidden state for this layer: every lane's K/V
    /// buffer, interleaved per lane. Deterministic order — the trace and
    /// every replay must flatten identically.
    func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        arrays.reserveCapacity(laneKeys.count * 2)
        for lane in 0 ..< laneKeys.count {
            arrays.append(laneKeys[lane])
            arrays.append(laneValues[lane])
        }
        return arrays
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "CBv2CompiledLayerCache.update is unsupported — v2 models call updateAndAttend (layer \(layerIndex))"
        )
    }

    var state: [MLXArray] {
        get { [] }
        set { fatalError("CBv2CompiledLayerCache has no serializable state (layer \(layerIndex))") }
    }

    var metaState: [String] {
        get { [] }
        set { fatalError("CBv2CompiledLayerCache has no metaState (layer \(layerIndex))") }
    }

    var isTrimmable: Bool { false }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        fatalError(
            "CBv2CompiledLayerCache.makeMask is unsupported — compiled attention owns its masks (layer \(layerIndex))"
        )
    }

    func copy() -> any KVCache {
        fatalError("CBv2CompiledLayerCache.copy is unsupported (layer \(layerIndex))")
    }
}
