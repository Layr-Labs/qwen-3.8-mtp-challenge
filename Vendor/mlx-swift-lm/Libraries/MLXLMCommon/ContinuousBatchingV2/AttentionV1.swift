// AttentionV1.swift
//
// The v1 attention dispatch used by `CBv2LayerCache.updateAndAttend`:
// per-row `MLXFast.scaledDotProductAttention` against each sequence's own
// contiguous KV. The paged backend (workstream C) replaces this behind the
// same `CBv2AttendingLayerCache` protocol.
//
// ## Why this path cannot have the left-padding bug class
// - Decode is rectangular [B, 1]: each row attends EXACTLY its own KV, with
//   no mask at all — a fully-masked row cannot exist by construction, so
//   NaN poisoning (ee2a921) is impossible.
// - Prefill is per-request [1, chunk]: masks are per-request, derived only
//   from that request's own lengths — batch composition cannot influence
//   them.
// - The mask mode is a PURE FUNCTION of (L, returned KV length, window):
//   never data-dependent, never drifting across steps for the same logical
//   computation (MLX #3384 / report 10 §4 invariant 5).

import Foundation
import MLX

/// Namespace for the v1 (per-row SDPA) attention dispatch.
enum CBv2AttentionV1 {

    /// Mask mode for a single-request attention call.
    ///
    /// - `L == 1` (decode): `.none`. The row's retained KV IS its window —
    ///   sliding-window eviction already dropped everything outside it.
    /// - `L > 1` (prefill chunk) against `kL` returned KV entries:
    ///   - `.causal` when no window is configured, or when `kL <= window`
    ///     (the window cannot bind: the oldest returned entry is inside
    ///     every query's window).
    ///   - causal ∧ window ARRAY mask when `kL > window` (a windowed layer's
    ///     multi-token update returned pre-eviction history so early chunk
    ///     tokens see their full window; later tokens must not over-attend).
    ///
    /// Pure in (L, kL, window): the same request produces the same mask mode
    /// at the same point in its lifetime regardless of batchmates.
    static func maskMode(L: Int, kL: Int, window: Int?)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        if L == 1 { return .none }
        if let window, kL > window {
            // Relative coordinates: keys span [0, kL), queries are the last
            // L positions. Window comparisons are translation-invariant, so
            // the absolute offset is irrelevant.
            return .array(createCausalMask(n: L, offset: kL - L, windowSize: window))
        }
        return .causal
    }

    /// Update each row with this step's K/V and attend.
    ///
    /// - queries/keys/values: `[B, heads, L, headDim]` with `L == 1` for
    ///   decode (B == rows.count) or `B == 1` for a prefill chunk.
    /// - softcap: construction-time attention-logit soft cap
    ///   (`cap * tanh(qk / cap)` before softmax, Gemma-2 style). When set,
    ///   BOTH phases take the composed reference path (SDPA cannot express
    ///   softcapping) — still one pinned path per phase.
    /// - spanContext: non-nil ONLY for a vision prefill chunk containing
    ///   image spans (a NEW pinned path — see `spanChunkMask`). Text-only
    ///   chunks and decode always pass nil and are untouched.
    /// - Returns `[B, queryHeads, L, headDim]`.
    static func updateAndAttend(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float? = nil,
        spanContext: CBv2SpanChunkContext? = nil
    ) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(!rows.isEmpty, "CBv2AttentionV1: no rows")
        precondition(
            B == rows.count,
            "CBv2AttentionV1: batch \(B) != rows \(rows.count) — prefill must run per-request [1, chunk]"
        )
        precondition(
            B == 1 || L == 1,
            "CBv2AttentionV1: ragged shapes are impossible in v2 — decode is [B, 1], prefill is [1, chunk]"
        )
        precondition(
            spanContext == nil || (B == 1 && L > 1),
            "CBv2AttentionV1: span contexts exist only for [1, chunk] prefill — decode never carries one"
        )
        let effectiveSinks = kind.hasSinks ? sinks : nil

        if B == 1 {
            let (cachedKeys, cachedValues) = rows[0].update(keys: keys, values: values)
            if let spanContext, L > 1 {
                return attendSpanChunk(
                    queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                    L: L, kL: cachedKeys.dim(2), window: window(of: kind),
                    context: spanContext, sinks: effectiveSinks, softcap: softcap)
            }
            return attend(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                L: L, kL: cachedKeys.dim(2), window: window(of: kind),
                sinks: effectiveSinks, softcap: softcap)
        }

        // Batched decode: split queries per row, per-row update + SDPA
        // against that row's own KV, then concatenate. No masks — each row
        // sees exactly its own KV, so batch-composition invariance holds by
        // construction and fully-masked rows cannot exist.
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for (index, row) in rows.enumerated() {
            let (cachedKeys, cachedValues) = row.update(
                keys: keys[index ..< (index + 1)],
                values: values[index ..< (index + 1)])
            outputs.append(
                attend(
                    queries: queries[index ..< (index + 1)],
                    keys: cachedKeys, values: cachedValues, scale: scale,
                    L: 1, kL: cachedKeys.dim(2), window: nil,
                    sinks: effectiveSinks, softcap: softcap))
        }
        return concatenated(outputs, axis: 0)
    }

    /// Attend against `sourceRows`' KV WITHOUT updating (Gemma-4 cross-layer
    /// KV sharing: shared layers project Q only and borrow the source
    /// layer's K/V — the source layer already appended this step's tokens
    /// earlier in the forward pass).
    ///
    /// View selection is a pure function of L (pinned-path discipline):
    ///  - CHUNK borrow (`L > 1`): `chunkBorrowViews(of:)`, NOT `snapshot()`.
    ///    After a windowed source's multi-token update, `snapshot()` is the
    ///    POST-eviction ring (≤ window entries), while the source layer's
    ///    own attention saw the PRE-eviction history + chunk
    ///    (`window - 1 + n` entries) so the chunk's earliest queries keep
    ///    their full window. Borrowing must attend the same view, with the
    ///    same `kL > window` array-mask path (the paged backend is already
    ///    exact here by construction — this mirrors its semantics).
    ///  - DECODE borrow (`L == 1`): `snapshot()`. The retained ring IS the
    ///    window (the source's same-step decode update already evicted),
    ///    so the mask-free decode path stays exact.
    static func attendBorrowing(
        sourceRows: [CBv2SequenceKV], sourceKind: CBv2LayerKind, kind: CBv2LayerKind,
        queries: MLXArray, scale: Float, sinks: MLXArray?, softcap: Float? = nil,
        spanContext: CBv2SpanChunkContext? = nil
    ) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(!sourceRows.isEmpty, "CBv2AttentionV1: no source rows to borrow from")
        precondition(
            B == sourceRows.count,
            "CBv2AttentionV1: batch \(B) != source rows \(sourceRows.count)")
        precondition(
            B == 1 || L == 1,
            "CBv2AttentionV1: ragged shapes are impossible in v2 — decode is [B, 1], prefill is [1, chunk]"
        )
        precondition(
            spanContext == nil || (B == 1 && L > 1),
            "CBv2AttentionV1: span contexts exist only for [1, chunk] prefill — decode never carries one"
        )
        let effectiveSinks = kind.hasSinks ? sinks : nil

        if B == 1, L > 1 {
            let (cachedKeys, cachedValues) = chunkBorrowViews(of: sourceRows[0])
            if let spanContext {
                // Same overlay as the storage-owning path: the MLXVLM
                // reference applies the bidirectional-span overlay to the
                // masks KV-shared layers reuse (they share their source's
                // layer type, hence its window).
                return attendSpanChunk(
                    queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                    L: L, kL: cachedKeys.dim(2), window: window(of: sourceKind),
                    context: spanContext, sinks: effectiveSinks, softcap: softcap)
            }
            return attend(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                L: L, kL: cachedKeys.dim(2), window: window(of: sourceKind),
                sinks: effectiveSinks, softcap: softcap)
        }

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for (index, row) in sourceRows.enumerated() {
            let (cachedKeys, cachedValues, _) = row.snapshot()
            outputs.append(
                attend(
                    queries: B == 1 ? queries : queries[index ..< (index + 1)],
                    keys: cachedKeys, values: cachedValues, scale: scale,
                    L: 1, kL: cachedKeys.dim(2), window: nil,
                    sinks: effectiveSinks, softcap: softcap))
        }
        return B == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    /// The views a borrowing layer must attend for the current PREFILL-CHUNK
    /// step: the windowed source's step-scoped pre-eviction views (see
    /// `CBv2WindowedSequenceKV.borrowableViews()`), else `snapshot()`
    /// (full-attention sources retain everything).
    private static func chunkBorrowViews(of row: CBv2SequenceKV) -> (MLXArray, MLXArray) {
        if let windowed = row as? CBv2WindowedSequenceKV {
            return windowed.borrowableViews()
        }
        let (keys, values, _) = row.snapshot()
        return (keys, values)
    }

    // MARK: - Private

    private static func window(of kind: CBv2LayerKind) -> Int? {
        switch kind.attention {
        case .full: return nil
        case .slidingWindow(let window): return window
        }
    }

    /// Single-request attention dispatch. Without a softcap this is MLXFast
    /// SDPA with `maskMode(L:kL:window:)`; with a softcap it is the composed
    /// fp32 reference (SDPA cannot express logit softcapping) with the
    /// EQUIVALENT boolean mask — both are pure functions of (L, kL, window),
    /// so each configuration keeps exactly one pinned path per phase.
    private static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        L: Int, kL: Int, window: Int?, sinks: MLXArray?, softcap: Float?
    ) -> MLXArray {
        guard let softcap else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale,
                mask: maskMode(L: L, kL: kL, window: window),
                sinks: sinks)
        }
        return PagedAttentionReference.composedAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            boolMask: boolMask(L: L, kL: kL, window: window),
            sinks: sinks, softcap: softcap)
    }

    /// Boolean causal(∧window) mask (true == attend) equivalent to
    /// `maskMode(L:kL:window:)`, for the composed softcap path. nil for
    /// decode (L == 1: the retained KV IS the window).
    static func boolMask(L: Int, kL: Int, window: Int?) -> MLXArray? {
        guard L > 1 else { return nil }
        // Relative coordinates: keys span [0, kL), queries are the last L.
        let qpos = MLXArray(Int32(kL - L) ..< Int32(kL)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(0) ..< Int32(kL)).expandedDimensions(axis: 0)
        var mask = kpos .<= qpos
        if let window, kL > window {
            mask = mask .&& (kpos .> (qpos - Int32(window)))
        }
        return mask
    }

    // MARK: - Vision span-chunk path (pinned; span-containing chunks only)

    /// Boolean mask for a span-containing prefill chunk: the causal(∧window)
    /// base OR bidirectional-within-block, matching MLXVLM Gemma4's
    /// `gemma4BidirectionalVisionMask` overlay exactly (`baseMask ∨
    /// sameBlock`, where sameBlock un-masks BOTH forward attention inside an
    /// image block and window-evicted backward attention inside it).
    ///
    /// Coordinates: queries are the last `L` absolute positions before
    /// `context.chunkEnd`; keys are the last `kL` (both backends' chunk
    /// views end at the chunk's last token). Blocks are absolute and fully
    /// inside the chunk, so every same-block (q, k) pair is present in the
    /// returned KV view by construction.
    static func spanChunkMask(
        L: Int, kL: Int, window: Int?, context: CBv2SpanChunkContext
    ) -> MLXArray {
        precondition(L > 1, "span chunks are multi-token by construction")
        var mask = boolMask(L: L, kL: kL, window: window)!
        let chunkEnd = context.chunkEnd
        let qAbs = MLXArray(Int32(chunkEnd - L) ..< Int32(chunkEnd)).expandedDimensions(axis: 1)
        let kAbs = MLXArray(Int32(chunkEnd - kL) ..< Int32(chunkEnd)).expandedDimensions(axis: 0)
        for block in context.blocks {
            let lo = Int32(block.tokenOffset)
            let hi = Int32(block.end)
            let qIn = (qAbs .>= lo) .&& (qAbs .< hi)  // [L, 1]
            let kIn = (kAbs .>= lo) .&& (kAbs .< hi)  // [1, kL]
            mask = mask .|| (qIn .&& kIn)
        }
        return mask
    }

    /// Span-chunk attention dispatch: always an explicit boolean array mask
    /// (the bidirectional overlay cannot ride the symbolic `.causal` mode).
    /// One pinned path per configuration — plain SDPA, or the composed fp32
    /// reference when a softcap is configured (same split as `attend`).
    private static func attendSpanChunk(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        L: Int, kL: Int, window: Int?, context: CBv2SpanChunkContext,
        sinks: MLXArray?, softcap: Float?
    ) -> MLXArray {
        let mask = spanChunkMask(L: L, kL: kL, window: window, context: context)
        guard let softcap else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale,
                mask: .array(mask), sinks: sinks)
        }
        return PagedAttentionReference.composedAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            boolMask: mask, sinks: sinks, softcap: softcap)
    }
}
