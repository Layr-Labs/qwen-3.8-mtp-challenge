// PagedAttentionReference.swift
//
// Composed (matmul + softmax, fp32) attention reference for the paged
// backend (WS-C). Three consumers:
//   - kernel parity tests (CBv2PagedKernelTests) — the correctness bar is
//     rel-err <= 1e-2 fp16 elementwise vs this reference;
//   - the prefill fallback for models with an attention-logit softcap
//     (SDPA cannot express softcapping);
//   - benchmark sanity checks.
//
// Semantics mirror the Metal kernel exactly: GQA broadcast, optional bool
// mask (true == attend), softcap BEFORE softmax, and attention sinks folded
// into the softmax DENOMINATOR only (never the value accumulation).

import Foundation
import MLX

public enum PagedAttentionReference {

    /// Composed attention.
    ///
    /// - Parameters:
    ///   - queries: `[B, queryHeads, L, headDim]`
    ///   - keys/values: `[B, kvHeads, T, headDim]`
    ///   - scale: query scale
    ///   - boolMask: optional `[L, T]`-broadcastable bool mask, true = attend
    ///   - sinks: optional `[queryHeads]` sink logits (denominator-only)
    ///   - softcap: optional logit soft cap: `cap * tanh(qk / cap)`
    /// - Returns: `[B, queryHeads, L, headDim]` in the query dtype.
    public static func composedAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, boolMask: MLXArray? = nil,
        sinks: MLXArray? = nil, softcap: Float? = nil
    ) -> MLXArray {
        let queryHeads = queries.dim(1)
        let kvHeads = keys.dim(1)
        precondition(queryHeads % kvHeads == 0)
        let gqa = queryHeads / kvHeads

        let q = queries.asType(.float32) * scale
        var k = keys.asType(.float32)
        var v = values.asType(.float32)
        if gqa > 1 {
            // Query head h attends kv head h / gqa — element-wise repeat.
            k = repeated(k, count: gqa, axis: 1)
            v = repeated(v, count: gqa, axis: 1)
        }

        var scores = matmul(q, k.transposed(0, 1, 3, 2))  // [B, QH, L, T]
        if let softcap {
            scores = tanh(scores / softcap) * softcap
        }
        if let boolMask {
            scores = which(boolMask, scores, MLXArray(Float(-1e30)))
        }

        if let sinks {
            // Denominator-only: softmax over [scores, sink], then drop the
            // sink column before the value matmul.
            let sinkColumn = broadcast(
                sinks.asType(.float32).reshaped([1, queryHeads, 1, 1]),
                to: [scores.dim(0), queryHeads, scores.dim(2), 1])
            let augmented = concatenated([scores, sinkColumn], axis: -1)
            let probs = softmax(augmented, axis: -1, precise: true)
            let attended = probs[0..., 0..., 0..., 0 ..< scores.dim(3)]
            return matmul(attended, v).asType(queries.dtype)
        }

        let probs = softmax(scores, axis: -1, precise: true)
        return matmul(probs, v).asType(queries.dtype)
    }
}
