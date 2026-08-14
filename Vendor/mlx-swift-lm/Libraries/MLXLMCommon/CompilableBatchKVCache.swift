// CompilableBatchKVCache: compile-traceable batched KV cache.
//
// Standard BatchKVCache breaks compile() because:
//   1. Buffer growth via concat rebinds keys/values — the trace captures the
//      original objects and rebinding loses them.
//   2. _idx is a Swift Int, invisible to the compile tracer.
//   3. update() returns a dynamic slice keys[..., ..<_idx, ...] whose shape
//      changes every step.
//
// This subclass fixes all three by:
//   - Pre-allocating the buffer to [B, H, maxLength, D] at promotion time.
//   - Tracking the write index as idxArray (MLXArray [1] int32).
//   - Using dynamicSliceUpdate for compile-traceable writes.
//   - Returning the FULL buffer from update(). The attention mask restricts
//     valid positions (overflow bin pattern).
//
// Initialise via the static promote(from:maxLength:) factory after prefill
// has populated the source BatchKVCache.

import Foundation
import MLX
import MLXNN

/// Compile-traceable specialisation of ``BatchKVCache``.
///
/// All batch-mutation operations (filter, extend, extract) sync the Swift-Int
/// `_idx` from `idxArray` before delegating to the parent, so the transition
/// from compiled to uncompiled mode is transparent.
public final class CompilableBatchKVCache: BatchKVCache {

    /// Current write index within the buffer, as `MLXArray[1] int32`.
    /// Tracked by `compile(inputs:outputs:)` through the computation graph.
    public var idxArray: MLXArray

    /// Maximum sequence length the buffer can hold.
    public let maxLength: Int

    /// Pre-computed column indices for mask creation `[0, 1, ..., maxLength-1]`.
    private lazy var maskRinds: MLXArray = MLXArray(Int32(0) ..< Int32(maxLength))

    // MARK: - Init

    /// Promote an existing populated ``BatchKVCache`` to a compile-traceable
    /// variant. Copies the state and allocates the fixed-size buffer at
    /// `[B, H, maxLength, D]`.
    ///
    /// - Parameters:
    ///   - source: Populated cache from prefill. Must have non-nil keys/values.
    ///   - maxLength: Maximum sequence length for the pre-allocated buffer.
    public static func promote(from source: BatchKVCache, maxLength: Int) -> CompilableBatchKVCache {
        return CompilableBatchKVCache(from: source, maxLength: maxLength)
    }

    init(from source: BatchKVCache, maxLength: Int) {
        self.maxLength = maxLength
        self.idxArray = MLXArray([Int32(source._idx)])

        // Dummy super.init — we overwrite all inherited state below.
        super.init(leftPadding: [0])

        // Copy metadata from source.
        self.batchOffset = source.batchOffset
        self.leftPadding = source.leftPadding
        self._idx = source._idx

        // Pre-allocate fixed-size buffers and copy existing data.
        if let srcK = source.keys, let srcV = source.values {
            let B = srcK.dim(0)
            let H = srcK.dim(1)
            let kD = srcK.dim(3)
            let vD = srcV.dim(3)
            let srcLen = source._idx

            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: srcK.dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: srcV.dtype)

            if srcLen > 0 {
                self.keys![.ellipsis, ..<srcLen, 0...] = srcK[.ellipsis, ..<srcLen, 0...]
                self.values![.ellipsis, ..<srcLen, 0...] = srcV[.ellipsis, ..<srcLen, 0...]
            }
        }
    }

    // MARK: - Compiled update

    /// Compile-traceable append. Writes new tokens at `idxArray` via
    /// `dynamicSliceUpdate`, advances counters in MLXArray ops, and returns
    /// the FULL `[B, H, maxLength, D]` buffer. `makeMask` restricts
    /// attention to valid positions.
    public override func update(
        keys newKeys: MLXArray, values newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        let nTokens = newKeys.dim(2)

        // Lazy-allocate on first call (shouldn't happen after promotion, but
        // defensive).
        if self.keys == nil {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let kD = newKeys.dim(3)
            let vD = newValues.dim(3)
            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: newKeys.dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: newValues.dtype)
        }

        // Write new tokens at idxArray position via dynamicSliceUpdate.
        // _updateInternal preserves object identity for the compile tracer.
        self.keys!._updateInternal(
            dynamicSliceUpdate(self.keys!, update: newKeys, start: idxArray, axes: [2]))
        self.values!._updateInternal(
            dynamicSliceUpdate(self.values!, update: newValues, start: idxArray, axes: [2]))

        // Advance counters in graph space.
        let advance = MLXArray([Int32(nTokens)])
        idxArray._updateInternal(idxArray + advance)
        batchOffset._updateInternal(batchOffset + advance)

        // NOTE: no asyncEval here. The parent's DAR-325 collapse of per-step
        // batchOffset chains is not needed (and not allowed) inside a compile
        // trace: the traced graph fuses the chain into a single update.

        // OVERFLOW BIN: return the full static-size buffer.
        return (self.keys!, self.values!)
    }

    // MARK: - Mask (Overflow Bin)

    /// Generate attention mask for the full-buffer return.
    ///
    /// Since update() returns the entire maxLength buffer, we ALWAYS need an
    /// array mask. The mask combines causal ordering with per-row leftPadding
    /// exclusion:
    ///   mask[b, j] = (j < idxArray) AND (j >= leftPadding[b])
    ///
    /// Uses `idxArray` and `leftPadding` (both MLXArray) so compile() can
    /// trace through the mask computation.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let currentIdx = idxArray  // [1] int32, post-update

        // Query positions
        let linds: MLXArray
        if n == 1 {
            linds = currentIdx.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + currentIdx).reshaped(n, 1)
        }

        // Key positions over the full buffer
        let rinds = maskRinds.reshaped(1, maxLength)

        // Causal + overflow bin: attend to positions j where j <= query.
        // `>=` so the slot at idxArray is included — models that build the
        // mask BEFORE cache.update() (e.g. Gemma 4) need the current
        // token's slot visible. For models that build the mask after
        // update, idxArray has already advanced past the current token and
        // `>=` still excludes the next unwritten slot correctly.
        var mask = linds .>= rinds

        // Per-row leftPadding exclusion: don't attend to j < leftPadding[b]
        // leftPadding [B] → [B, 1, 1, 1] for broadcast with [n, maxLength]
        let lp = leftPadding[0..., .newAxis, .newAxis, .newAxis]
        mask = mask & (lp .<= rinds)

        // Sliding window
        if let windowSize {
            mask = mask & (rinds .>= (linds - Int32(windowSize - 1)))
        }

        return .array(mask)
    }

    // MARK: - innerState (compile tracking)

    /// Return state arrays tracked by `compile(inputs:outputs:)`.
    /// Includes the full-size buffers and all MLXArray counters.
    public override func innerState() -> [MLXArray] {
        var state = [MLXArray]()
        if let k = keys { state.append(k) }
        if let v = values { state.append(v) }
        state.append(idxArray)
        state.append(batchOffset)
        state.append(leftPadding)
        return state
    }

    // MARK: - State sync for non-compiled operations

    /// Sync Swift-Int `_idx` from `idxArray`. Must be called before any
    /// inherited operation that reads `_idx` (filter, extend, extract).
    /// During compiled decode, `_idx` is NOT updated (the compile tracer
    /// doesn't follow Swift Int assignments), so this bridges the gap.
    func syncFromCompiled() {
        eval(idxArray)
        _idx = Int(idxArray[0].item(Int32.self))
    }

    public override func filterBatched(batchIndices: MLXArray) {
        syncFromCompiled()
        super.filterBatched(batchIndices: batchIndices)
        // Re-sync idxArray from _idx (parent's filter may adjust _idx via
        // left-padding shave).
        idxArray = MLXArray([Int32(_idx)])
    }

    public override func extendBatched(_ other: any BatchedCache) {
        syncFromCompiled()
        super.extendBatched(other)
        idxArray = MLXArray([Int32(_idx)])
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
        return trimmed
    }

    // MARK: - Copy

    public override func copy() -> any KVCache {
        syncFromCompiled()
        let src = BatchKVCache(leftPadding: [0])
        src.keys = keys
        src.values = values
        src.batchOffset = batchOffset[0...]
        src.leftPadding = leftPadding[0...]
        src._idx = _idx
        return CompilableBatchKVCache(from: src, maxLength: maxLength)
    }
}
