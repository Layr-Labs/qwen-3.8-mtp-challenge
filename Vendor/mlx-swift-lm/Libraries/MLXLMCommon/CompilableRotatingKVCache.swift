// CompilableRotatingKVCache: compile-traceable rotating KV cache.
//
// Ported from osaurus-ai/vmlx-swift-lm
// (Libraries/MLXLMCommon/BatchEngine/CompilableRotatingKVCache.swift).
//
// Standard RotatingKVCache breaks compile() in two ways:
//   1. Buffer growth via concat rebinds the keys property — the trace
//      captures the original object and rebinding loses it.
//   2. Ring-buffer wrap via Swift Int reset doesn't match the trace's
//      assumed linear layout.
//
// This subclass fixes both by:
//   - Pre-allocating the unified buffer to maxCacheSize at promotion time.
//     No concat-growth during decode.
//   - Tracking the write index as idxArray (MLXArray [1] int32). Wrap
//     arithmetic uses MLXArray ops (where_ + modulo) so the compile tracer
//     can follow.
//   - Returning the FULL [B, H, maxCacheSize, D] buffer from update().
//     The attention mask restricts valid positions.
//
// Initialise via init(from:) on an existing RotatingKVCache populated by
// prefill, or via the static promote(from:maxLength:) helper.

import Foundation
import MLX
import MLXNN

/// Compile-traceable specialisation of ``RotatingKVCache``.
///
/// Compared to the parent:
///
/// - `keys` and `values` are eagerly allocated at `[B, H, maxCacheSize, D]`
///   at promotion time. No growth-via-concat during decode.
/// - `idxArray: MLXArray[1] int32` replaces Swift-Int `idx`. All wrap
///   arithmetic happens in MLXArray ops so the compile tracer can follow.
/// - `offsetArray: MLXArray[1] int32` mirrors `offset`. Used by `makeMask`
///   for the causal upper bound; tracked as an MLXArray so the tracer
///   follows per-step advances.
/// - `update(keys:values:)` writes new tokens at `idxArray` via
///   `dynamicSliceUpdate`, then advances `idxArray` with wrap semantics
///   entirely in MLXArray space.
/// - `makeMask` always returns `.array(mask)` — the full-buffer return
///   means attention must be told which positions are valid.
public final class CompilableRotatingKVCache: RotatingKVCache, @unchecked Sendable {

    /// Current write index within the ring buffer, as `MLXArray[1] int32`.
    /// In the linear segment (before wrap), this equals `offsetArray`.
    /// After wrap, this rotates through `[keep, maxCacheSize)`.
    public var idxArray: MLXArray

    /// Total valid tokens seen, as `MLXArray[1] int32`. In the linear
    /// segment, equals `idxArray` and is a tight upper bound on valid
    /// positions. Post-wrap, `offsetArray >= maxCacheSize` and ALL
    /// `maxCacheSize` positions in the buffer are valid (ring full).
    public var offsetArray: MLXArray

    /// Pre-computed column indices `[0, 1, ..., maxCacheSize-1]` used by
    /// `makeMask` to build a causal mask over the full buffer.
    private lazy var maskRinds: MLXArray = MLXArray(Int32(0) ..< Int32(maxCacheSize))

    // MARK: - Init

    /// Direct constructor matching the parent. Primarily for testing.
    public override init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.idxArray = MLXArray([Int32(0)])
        self.offsetArray = MLXArray([Int32(0)])
        super.init(maxSize: maxSize, keep: keep, step: step)
    }

    /// Promote an existing populated ``RotatingKVCache`` to a compile-
    /// traceable variant. Copies the state references AND allocates the
    /// unified buffer at full `maxCacheSize` size if the parent's buffer
    /// is smaller (the parent grows lazily in `step`-sized chunks).
    ///
    /// - Parameter rotating: Source cache. Typically the result of a
    ///   prefill that has populated keys/values.
    public convenience init(from rotating: RotatingKVCache) {
        self.init(
            maxSize: rotating.maxCacheSize,
            keep: rotating.keep,
            step: rotating.step
        )

        // Copy state references from the source. Same-module subclass
        // access works because parent's state is `internal`.
        self.idx = rotating.idx
        self.offset = rotating.offset

        // Pre-allocate or extend the unified buffer to full maxCacheSize.
        // This prevents the compile-breaking concat-growth path from ever
        // firing during decode.
        if let srcK = rotating.keys, let srcV = rotating.values {
            let B = srcK.dim(0)
            let H = srcK.dim(1)
            let kD = srcK.dim(3)
            let vD = srcV.dim(3)
            let curLen = srcK.dim(2)

            if curLen < maxCacheSize {
                // Need to grow — but this is a ONE-TIME growth during
                // promotion, not inside a compile trace. Use concat to
                // extend to full size.
                let padLen = maxCacheSize - curLen
                let padK = MLXArray.zeros([B, H, padLen, kD], dtype: srcK.dtype)
                let padV = MLXArray.zeros([B, H, padLen, vD], dtype: srcV.dtype)
                self.keys = concatenated([srcK, padK], axis: 2)
                self.values = concatenated([srcV, padV], axis: 2)
            } else {
                self.keys = srcK
                self.values = srcV
            }
        }
        // else: keys/values remain nil; first `update` call allocates
        // them at full size.

        self.idxArray = MLXArray([Int32(self.idx)])
        self.offsetArray = MLXArray([Int32(self.offset)])
    }

    /// Static promote helper for symmetry with `CompilableKVCache.promote`.
    public static func promote(from cache: RotatingKVCache, maxLength: Int) -> CompilableRotatingKVCache {
        // maxLength is unused here because RotatingKVCache already has maxCacheSize,
        // but the parameter keeps the API symmetric with CompilableKVCache.promote.
        return CompilableRotatingKVCache(from: cache)
    }

    // MARK: - Overridden update

    /// Compile-traceable append. Writes new tokens at `idxArray` position
    /// via `dynamicSliceUpdate`, advances counters with wrap semantics in
    /// MLXArray ops.
    ///
    /// Returns the FULL `[B, H, maxCacheSize, D]` buffer. `makeMask`
    /// restricts attention to valid positions.
    public override func update(
        keys newKeys: MLXArray, values newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        let nTokens = newKeys.dim(2)

        // Lazy-allocate the unified buffer if empty (first-call init).
        if keys == nil {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let kD = newKeys.dim(3)
            let vD = newValues.dim(3)
            keys = MLXArray.zeros([B, H, maxCacheSize, kD], dtype: newKeys.dtype)
            values = MLXArray.zeros([B, H, maxCacheSize, vD], dtype: newValues.dtype)
        }

        // Write new tokens at idxArray position.
        keys!._updateInternal(
            dynamicSliceUpdate(keys!, update: newKeys, start: idxArray, axes: [2]))
        values!._updateInternal(
            dynamicSliceUpdate(values!, update: newValues, start: idxArray, axes: [2]))

        // Advance counters. Wrap arithmetic on idxArray:
        //   newIdx = advance < maxCacheSize ? advance : keep + (advance - keep) % (maxCacheSize - keep)
        // We use `where_` so both branches live in the MLXArray graph.
        let advance = MLXArray([Int32(nTokens)])
        let advancedIdx = idxArray + advance
        let maxSz = MLXArray([Int32(maxCacheSize)])
        let keepArr = MLXArray([Int32(keep)])
        let cycleLen = maxSz - keepArr  // number of rotating slots

        let rotatedIdx: MLXArray
        if keep > 0 {
            rotatedIdx = keepArr + ((advancedIdx - keepArr) % cycleLen)
        } else {
            rotatedIdx = advancedIdx % maxSz
        }
        // where_(cond, true_branch, false_branch)
        let newIdx = MLX.`where`(advancedIdx .< maxSz, advancedIdx, rotatedIdx)

        idxArray._updateInternal(newIdx)
        offsetArray._updateInternal(offsetArray + advance)

        // DELIBERATELY no Swift-Int mirror updates here:
        // `idx = Int(newIdx.item(Int32.self))` would force an `eval`
        // call, which MLX compile rejects. Consumers that need the Int
        // view should read the MLXArray counters and materialize
        // OUTSIDE the compiled trace.

        return (keys!, values!)
    }

    // MARK: - makeMask

    /// Build an attention mask over the full `[B, H, maxCacheSize, D]`
    /// buffer.
    ///
    /// Mask semantics:
    ///  - **Linear phase** (offsetArray < maxCacheSize): valid positions
    ///    are `[0, offsetArray)`. Causal mask is `linds >= rinds`.
    ///  - **Post-wrap phase** (offsetArray >= maxCacheSize): all
    ///    `maxCacheSize` positions are valid (the ring is full). The
    ///    ring layout means positions are NOT in logical order — but for
    ///    single-query decode with `n=1`, every position is attendable.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let linds: MLXArray
        if n == 1 {
            linds = offsetArray.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + offsetArray).reshaped(n, 1)
        }

        let rinds = maskRinds.reshaped(1, maxCacheSize)
        // Causal: attend to positions j <= query_position.
        let causal = linds .>= rinds

        // Post-wrap: if offsetArray >= maxCacheSize, all positions are
        // valid. For n=1 this is `linds >= maxCacheSize ? all-true : causal`.
        let maxSzArr = MLXArray([Int32(maxCacheSize)]).reshaped(1, 1)
        let allTrueMask = MLX.broadcast(
            MLXArray([true]).reshaped(1, 1),
            to: [linds.dim(0), rinds.dim(1)]
        )
        var mask = MLX.`where`(linds .>= maxSzArr, allTrueMask, causal)

        if let windowSize {
            // After ring wrap, the recent window may be split across buffer
            // end and beginning. Compare in modular token-index space rather
            // than physical ring-column space so both halves are included.
            // tokenInds maps each physical position to its distance from the
            // write cursor: idxArray → 0 (oldest/next-write), idxArray-1 →
            // maxCacheSize-1 (most recent). Keep the RECENT end of the ring.
            let tokenInds = (rinds - idxArray + MLXArray(Int32(maxCacheSize))) % Int32(maxCacheSize)
            let windowFilter = tokenInds .>= Int32(maxCacheSize - windowSize)
            mask = mask & windowFilter
        }

        return .array(mask)
    }

    // MARK: - innerState

    /// Return ONLY state that mutates during decode: the keys/values
    /// buffers and the two MLXArray counters.
    public override func innerState() -> [MLXArray] {
        var state = [MLXArray]()
        if let k = keys { state.append(k) }
        if let v = values { state.append(v) }
        state.append(idxArray)
        state.append(offsetArray)
        return state
    }
}
