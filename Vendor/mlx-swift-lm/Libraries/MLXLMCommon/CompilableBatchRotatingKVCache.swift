// CompilableBatchRotatingKVCache: compile-traceable batched rotating KV cache.
//
// Standard BatchRotatingKVCache breaks compile() for the same reasons as
// BatchKVCache (dynamic growth, Swift-Int idx, variable-shape returns).
//
// This subclass fixes all three by:
//   - Pre-allocating the buffer to [B, H, maxCacheSize, D] at promotion time.
//   - Tracking the write index as idxArray (MLXArray [1] int32) with wrap
//     arithmetic in MLXArray ops.
//   - Tracking total tokens as offsetArray (MLXArray [1] int32).
//   - Using dynamicSliceUpdate for compile-traceable writes.
//   - Returning the FULL buffer from update(). The attention mask restricts
//     valid positions.
//
// When the ring fills (offsetArray >= maxCacheSize), new tokens overwrite the
// oldest positions (idxArray wraps modulo maxCacheSize). leftPadding decreases
// as padding positions are overwritten.
//
// On fallback to uncompiled mode (filter/extend), syncFromCompiled() unwinds
// the ring layout back to temporal order so the parent's mutations work.

import Foundation
import MLX
import MLXNN

/// Compile-traceable specialisation of ``BatchRotatingKVCache``.
public final class CompilableBatchRotatingKVCache: BatchRotatingKVCache {

    /// Current write index within the ring buffer, as `MLXArray[1] int32`.
    /// Wraps modulo maxCacheSize.
    public var idxArray: MLXArray

    /// Total tokens seen, as `MLXArray[1] int32`. Monotonically increasing.
    /// Used by makeMask to distinguish the linear-fill phase from the
    /// ring-full phase.
    public var offsetArray: MLXArray

    /// Pre-computed column indices for mask creation.
    private lazy var maskRinds: MLXArray = MLXArray(Int32(0) ..< Int32(maxCacheSize))

    // MARK: - Init

    /// Promote an existing populated ``BatchRotatingKVCache`` to a
    /// compile-traceable variant.
    ///
    /// - Parameters:
    ///   - source: Populated cache from prefill.
    ///   - maxLength: Unused (kept for API symmetry with CompilableBatchKVCache).
    ///     The buffer is sized to `source.maxCacheSize`.
    public static func promote(
        from source: BatchRotatingKVCache, maxLength: Int
    ) -> CompilableBatchRotatingKVCache {
        return CompilableBatchRotatingKVCache(from: source)
    }

    init(from source: BatchRotatingKVCache) {
        // Normalize the source first so we get a clean temporal-order layout.
        source.normalizeToWindow()

        self.idxArray = MLXArray([Int32(source._idx)])
        self.offsetArray = MLXArray([Int32(source.offset)])

        super.init(maxSize: source.maxCacheSize, leftPadding: [0])

        // Copy metadata from source.
        self.batchOffset = source.batchOffset
        self.leftPadding = source.leftPadding
        self._idx = source._idx
        self._base = 0

        // Disable the parent's fast decode ring — the compiled ring-buffer
        // update replaces it.
        self.useFastDecodePath = false

        // Pre-allocate the unified buffer at full maxCacheSize.
        if let srcK = source.keys, let srcV = source.values {
            let B = srcK.dim(0)
            let H = srcK.dim(1)
            let kD = srcK.dim(3)
            let vD = srcV.dim(3)
            let srcLen = source._idx

            if srcLen < maxCacheSize {
                // Grow to full size.
                let padLen = maxCacheSize - srcLen
                let padK = MLXArray.zeros([B, H, padLen, kD], dtype: srcK.dtype)
                let padV = MLXArray.zeros([B, H, padLen, vD], dtype: srcV.dtype)
                self.keys = concatenated([srcK, padK], axis: 2)
                self.values = concatenated([srcV, padV], axis: 2)
            } else {
                self.keys = srcK
                self.values = srcV
            }
        }
    }

    // MARK: - Compiled update

    /// Compile-traceable append with ring-buffer wrap semantics.
    ///
    /// Writes new tokens at `idxArray` via `dynamicSliceUpdate`, advances
    /// counters with wrap arithmetic in MLXArray ops, and returns the FULL
    /// `[B, H, maxCacheSize, D]` buffer. `makeMask` restricts attention.
    public override func update(
        keys newKeys: MLXArray, values newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        let nTokens = newKeys.dim(2)

        // Lazy-allocate (defensive — shouldn't happen after promotion).
        if self.keys == nil {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let kD = newKeys.dim(3)
            let vD = newValues.dim(3)
            self.keys = MLXArray.zeros([B, H, maxCacheSize, kD], dtype: newKeys.dtype)
            self.values = MLXArray.zeros([B, H, maxCacheSize, vD], dtype: newValues.dtype)
        }

        // Check if left-padding should decrease (ring was full before this write).
        let maxSz = MLXArray([Int32(maxCacheSize)])
        let wasFull = offsetArray .>= maxSz

        // Write new tokens at idxArray position.
        self.keys!._updateInternal(
            dynamicSliceUpdate(self.keys!, update: newKeys, start: idxArray, axes: [2]))
        self.values!._updateInternal(
            dynamicSliceUpdate(self.values!, update: newValues, start: idxArray, axes: [2]))

        // Advance write index with wrap: newIdx = advancedIdx % maxCacheSize
        // when advancedIdx >= maxCacheSize, else advancedIdx.
        let advance = MLXArray([Int32(nTokens)])
        let advancedIdx = idxArray + advance
        let newIdx = MLX.`where`(advancedIdx .< maxSz, advancedIdx, advancedIdx % maxSz)
        idxArray._updateInternal(newIdx)

        // Advance total offset and per-row offset.
        offsetArray._updateInternal(offsetArray + advance)
        batchOffset._updateInternal(batchOffset + advance)

        // leftPadding decreases when the ring was full (old data overwritten).
        let lpDecrease = MLX.`where`(wasFull, advance, MLXArray([Int32(0)]))
        leftPadding._updateInternal(leftPadding - lpDecrease)

        // NOTE: no asyncEval here. The parent's DAR-325 collapse of per-step
        // chains is not needed (and not allowed) inside a compile trace.

        return (self.keys!, self.values!)
    }

    // MARK: - Mask

    /// Attention mask over the full `[B, H, maxCacheSize, D]` buffer.
    ///
    /// - **Linear phase** (offsetArray < maxCacheSize): valid positions are
    ///   `[leftPadding[b], offsetArray)`. Causal mask + leftPadding.
    /// - **Post-wrap** (offsetArray >= maxCacheSize): all positions are valid
    ///   (ring full). Only leftPadding exclusion applies (typically ≤0 by now).
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let linds: MLXArray
        if n == 1 {
            linds = offsetArray.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + offsetArray).reshaped(n, 1)
        }

        let rinds = maskRinds.reshaped(1, maxCacheSize)

        // Causal: attend to j where j <= query position.
        // `>=` so the current token's slot is included — models that build
        // the mask BEFORE cache.update() need it visible; models that build
        // after update have offsetArray already advanced past the current
        // token, so `>=` still excludes the next unwritten slot.
        let causal = linds .>= rinds

        // Post-wrap: all maxCacheSize positions are valid → all-true.
        let maxSzArr = MLXArray([Int32(maxCacheSize)]).reshaped(1, 1)
        let allTrue = MLX.broadcast(
            MLXArray([true]).reshaped(1, 1),
            to: [linds.dim(0), rinds.dim(1)]
        )
        var mask = MLX.`where`(linds .>= maxSzArr, allTrue, causal)

        // Per-row leftPadding exclusion.
        let lp = leftPadding[0..., .newAxis, .newAxis, .newAxis]
        mask = mask & (lp .<= rinds)

        // Sliding window: use modular token-index comparison (not physical
        // ring-column) so the mask works correctly after ring wrap, when the
        // valid window is split across the buffer end and beginning.
        // tokenInds maps each position to its distance from the write cursor:
        // idxArray → 0 (oldest/next-write), idxArray-1 → maxCacheSize-1
        // (most recent). Keep the RECENT end of the ring.
        if let windowSize {
            let tokenInds = (rinds - idxArray + MLXArray(Int32(maxCacheSize))) % Int32(maxCacheSize)
            let windowFilter = tokenInds .>= Int32(maxCacheSize - windowSize)
            mask = mask & windowFilter
        }

        return .array(mask)
    }

    // MARK: - innerState (compile tracking)

    public override func innerState() -> [MLXArray] {
        var state = [MLXArray]()
        if let k = keys { state.append(k) }
        if let v = values { state.append(v) }
        state.append(idxArray)
        state.append(offsetArray)
        state.append(batchOffset)
        state.append(leftPadding)
        return state
    }

    // MARK: - State sync for non-compiled operations

    /// Sync Swift-Int state from MLXArray state and unwind the ring layout
    /// to temporal order. Must be called before any inherited operation that
    /// reads `_idx` or assumes temporal ordering (filter, extend, extract).
    func syncFromCompiled() {
        eval(idxArray, offsetArray)
        if let k = keys, let v = values { eval(k, v) }

        let totalOffset = Int(offsetArray[0].item(Int32.self))
        let physIdx = Int(idxArray[0].item(Int32.self))
        let validLen = min(totalOffset, maxCacheSize)

        // Unwind ring layout to temporal order if the ring has wrapped and
        // the write cursor isn't at position 0 (which would already be in
        // order).
        if totalOffset > maxCacheSize, physIdx != 0,
            let k = keys, let v = values
        {
            keys = concatenated([
                k[.ellipsis, physIdx..., 0...],
                k[.ellipsis, ..<physIdx, 0...],
            ], axis: 2)
            values = concatenated([
                v[.ellipsis, physIdx..., 0...],
                v[.ellipsis, ..<physIdx, 0...],
            ], axis: 2)
        }

        _idx = validLen
        offset = totalOffset
        _base = 0
    }

    // MARK: - Override normalizeToWindow

    /// No-op during compiled mode. The buffer is always pre-allocated at
    /// maxCacheSize with _base=0. syncFromCompiled() handles the unwind
    /// before falling back to the parent's mutations.
    override func normalizeToWindow() {
        // After syncFromCompiled, the buffer is already in canonical layout.
        // Let the parent handle it.
        super.normalizeToWindow()
    }

    // MARK: - Batch operations with sync

    public override func filterBatched(batchIndices: MLXArray) {
        syncFromCompiled()
        super.filterBatched(batchIndices: batchIndices)
        // Re-sync tracked arrays from parent state.
        idxArray = MLXArray([Int32(_idx)])
        offsetArray = MLXArray([Int32(offset)])
    }

    public override func extendBatched(_ other: any BatchedCache) {
        syncFromCompiled()
        super.extendBatched(other)
        idxArray = MLXArray([Int32(_idx)])
        offsetArray = MLXArray([Int32(offset)])
    }

    public override func extractBatched(_ idx: Int) -> any KVCache {
        syncFromCompiled()
        return super.extractBatched(idx)
    }

    // MARK: - Trim

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        syncFromCompiled()
        let trimmed = super.trim(n)
        idxArray = MLXArray([Int32(_idx)])
        offsetArray = MLXArray([Int32(offset)])
        return trimmed
    }

    // MARK: - Copy

    public override func copy() -> any KVCache {
        syncFromCompiled()
        let src = BatchRotatingKVCache(maxSize: maxCacheSize, leftPadding: [0])
        src.keys = windowKeys
        src.values = windowValues
        src.batchOffset = batchOffset[0...]
        src.leftPadding = leftPadding[0...]
        src._idx = _idx
        src._base = 0
        return CompilableBatchRotatingKVCache(from: src)
    }
}
