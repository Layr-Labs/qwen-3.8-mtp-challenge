import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Focused numerical-parity tests for the quantized scaled-dot-product attention
/// kernel path after the decode performance changes:
///
///  * Lever A — the softmax is fused via `compile(shapeless:)`. These tests pin
///    that the fused result still matches an independent (plain-softmax,
///    dequantized-matmul) reference within fp tolerance, for both single-query
///    decode (`n == 1`) and multi-query prefill (`n > 1`), with grouped-query
///    attention (GQA).
///  * Lever B — `QuantizedBatchKVCacheBase.makeMask` returns `.none` for the
///    single-query decode no-op-mask case. These tests pin that skipping the
///    mask is bit-identical to applying the materialized all-`true` mask, and
///    that the fast path is taken only under the safe conditions.
@Suite("QuantizedSDPATests")
struct QuantizedSDPATests {

    // MARK: - Config / helpers

    /// `groupSize` must divide the head dim for the quantized cache.
    static let groupSize = 64
    static let bits = 8
    static let headDim = 64

    /// Deterministic, varied, non-degenerate data in roughly `[-1, 1]` so the
    /// test does not depend on `MLXRandom` (which the test target does not link).
    private func varied(_ shape: [Int], phase: Float) -> MLXArray {
        let total = shape.reduce(1, *)
        let idx = MLXArray(Int32(0) ..< Int32(total)).asType(.float32)
        return sin(idx * 0.13 + phase).reshaped(shape)
    }

    private func quantize(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray?) {
        let q = quantized(x, groupSize: Self.groupSize, bits: Self.bits)
        return (q.wq, q.scales, q.biases)
    }

    private func dequantize(_ t: (MLXArray, MLXArray, MLXArray?)) -> MLXArray {
        dequantized(
            t.0, scales: t.1, biases: t.2,
            groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
    }

    /// Repeat KV heads to match query heads, mirroring the GQA layout used by the
    /// kernel: `[B, nKVHeads, T, D] -> [B, nQHeads, T, D]`.
    private func repeatKVHeads(_ x: MLXArray, repeats: Int) -> MLXArray {
        if repeats == 1 { return x }
        let (b, h, t, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let expanded = expandedDimensions(x, axis: 2)  // [B, H, 1, T, D]
        let tiled = broadcast(expanded, to: [b, h, repeats, t, d])
        return tiled.reshaped([b, h * repeats, t, d])
    }

    /// Independent reference attention computed from the **dequantized** K/V with
    /// a plain `softmax`. `maskBool` is an optional `[n, T]` (or broadcastable)
    /// boolean mask: `true` = attend, `false` = masked out.
    private func referenceAttention(
        queries: MLXArray,
        keysDeq: MLXArray,
        valuesDeq: MLXArray,
        scale: Float,
        maskBool: MLXArray?
    ) -> MLXArray {
        let nQHeads = queries.dim(1)
        let nKVHeads = keysDeq.dim(1)
        let repeats = nQHeads / nKVHeads

        let k = repeatKVHeads(keysDeq, repeats: repeats)
        let v = repeatKVHeads(valuesDeq, repeats: repeats)

        var scores = matmul(queries * scale, k.swappedAxes(-1, -2))  // [B, nQHeads, n, T]
        if let maskBool {
            scores = MLX.where(maskBool, scores, MLXArray(-Float.greatestFiniteMagnitude))
        }
        let weights = softmax(scores, axis: -1)
        return matmul(weights, v)  // [B, nQHeads, n, D]
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    private static func isNone(_ m: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
        if case .none = m { return true }
        return false
    }

    private static func isArray(_ m: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
        if case .array = m { return true }
        return false
    }

    // MARK: - Lever A: fused softmax matches reference (no sinks)

    /// Single-query decode (`n == 1`), GQA, a few cached tokens, no mask.
    @Test func decodeMatchesReference() {
        let (b, nKVHeads, nQHeads, d) = (1, 2, 4, Self.headDim)
        let t = 6  // a few cached tokens
        let scale = 1.0 / Float(d).squareRoot()

        let q = varied([b, nQHeads, 1, d], phase: 0.0)
        let k = varied([b, nKVHeads, t, d], phase: 1.0)
        let v = varied([b, nKVHeads, t, d], phase: 2.0)

        let qK = quantize(k)
        let qV = quantize(v)

        let out = quantizedScaledDotProductAttention(
            queries: q, quantizedKeys: qK, quantizedValues: qV,
            scale: scale, mask: .none,
            groupSize: Self.groupSize, bits: Self.bits, mode: .affine)

        let ref = referenceAttention(
            queries: q, keysDeq: dequantize(qK), valuesDeq: dequantize(qV),
            scale: scale, maskBool: nil)

        #expect(out.shape == [b, nQHeads, 1, d])
        let diff = maxAbsDiff(out, ref)
        #expect(diff < 1e-3, "decode max|Δ| = \(diff)")
    }

    /// Multi-query prefill (`n > 1`), GQA, causal mask, batch > 1 (exercises the
    /// `[n, T]` mask broadcasting against `[B, nQHeads, n, T]` scores).
    @Test func prefillMatchesReference() {
        let (b, nKVHeads, nQHeads, d) = (2, 2, 4, Self.headDim)
        let n = 4  // a few cached tokens, all attended causally
        let scale = 1.0 / Float(d).squareRoot()

        let q = varied([b, nQHeads, n, d], phase: 0.5)
        let k = varied([b, nKVHeads, n, d], phase: 1.5)
        let v = varied([b, nKVHeads, n, d], phase: 2.5)

        let qK = quantize(k)
        let qV = quantize(v)

        // Production n>1 path uses an explicit causal array mask.
        let maskArray = createCausalMask(n: n, offset: 0)  // [n, n] lower-triangular

        let out = quantizedScaledDotProductAttention(
            queries: q, quantizedKeys: qK, quantizedValues: qV,
            scale: scale, mask: .array(maskArray),
            groupSize: Self.groupSize, bits: Self.bits, mode: .affine)

        let ref = referenceAttention(
            queries: q, keysDeq: dequantize(qK), valuesDeq: dequantize(qV),
            scale: scale, maskBool: maskArray)

        #expect(out.shape == [b, nQHeads, n, d])
        let diff = maxAbsDiff(out, ref)
        #expect(diff < 1e-3, "prefill max|Δ| = \(diff)")
    }

    // MARK: - Lever A: fused sink softmax matches reference

    /// Single-query decode with attention sinks (the compiled sink core).
    @Test func decodeWithSinksMatchesReference() {
        let (b, nKVHeads, nQHeads, d) = (1, 2, 4, Self.headDim)
        let t = 5
        let scale = 1.0 / Float(d).squareRoot()

        let q = varied([b, nQHeads, 1, d], phase: 0.2)
        let k = varied([b, nKVHeads, t, d], phase: 1.2)
        let v = varied([b, nKVHeads, t, d], phase: 2.2)
        let sinks = varied([nQHeads], phase: 3.0)  // per-query-head sink logits

        let qK = quantize(k)
        let qV = quantize(v)

        let out = quantizedScaledDotProductAttention(
            queries: q, quantizedKeys: qK, quantizedValues: qV,
            scale: scale, mask: .none,
            groupSize: Self.groupSize, bits: Self.bits, mode: .affine,
            sinks: sinks)

        // Reference: same scores, but softmax denominator includes the sink's
        // exp() term (the sink has no value vector, so it only removes mass).
        let kDeq = repeatKVHeads(dequantize(qK), repeats: nQHeads / nKVHeads)
        let vDeq = repeatKVHeads(dequantize(qV), repeats: nQHeads / nKVHeads)
        let scores = matmul(q * scale, kDeq.swappedAxes(-1, -2))  // [b, nQHeads, 1, t]
        let sinkLogits = sinks.reshaped([1, nQHeads, 1, 1])
        let m = maximum(scores.max(axis: -1, keepDims: true), sinkLogits)
        let expScores = exp(scores - m)
        let denom = expScores.sum(axis: -1, keepDims: true) + exp(sinkLogits - m)
        let ref = matmul(expScores / denom, vDeq)

        #expect(out.shape == [b, nQHeads, 1, d])
        let diff = maxAbsDiff(out, ref)
        #expect(diff < 1e-3, "decode+sinks max|Δ| = \(diff)")
    }

    // MARK: - Lever B: skipping the n==1 mask is numerically identical

    /// For `n == 1` with no left padding / no window, the causal mask is all
    /// `true`, so attending with `.none` must equal attending with the explicit
    /// all-`true` array mask (bit-identical: same scores feed the same softmax).
    @Test func maskSkipParityDecode() {
        let (b, nKVHeads, nQHeads, d) = (1, 2, 4, Self.headDim)
        let t = 7
        let scale = 1.0 / Float(d).squareRoot()

        let q = varied([b, nQHeads, 1, d], phase: 0.7)
        let k = varied([b, nKVHeads, t, d], phase: 1.7)
        let v = varied([b, nKVHeads, t, d], phase: 2.7)

        let qK = quantize(k)
        let qV = quantize(v)

        // The all-`true` mask the old always-masked path would have built for a
        // single decode query over `t` cached keys.
        let allTrueMask = createCausalMask(n: 1, offset: t - 1)  // [1, t], all true

        let outSkipped = quantizedScaledDotProductAttention(
            queries: q, quantizedKeys: qK, quantizedValues: qV,
            scale: scale, mask: .none,
            groupSize: Self.groupSize, bits: Self.bits, mode: .affine)

        let outMasked = quantizedScaledDotProductAttention(
            queries: q, quantizedKeys: qK, quantizedValues: qV,
            scale: scale, mask: .array(allTrueMask),
            groupSize: Self.groupSize, bits: Self.bits, mode: .affine)

        let diff = maxAbsDiff(outSkipped, outMasked)
        #expect(diff < 1e-5, ".none vs all-true mask max|Δ| = \(diff)")
    }

    // MARK: - Lever B: makeMask fast-path conditions

    /// `makeMask` returns `.none` only for the safe single-query decode case and
    /// otherwise materializes the array mask.
    @Test func makeMaskFastPathConditions() {
        let (nKVHeads, d) = (2, Self.headDim)
        let t = 5

        func freshCache(leftPadding: [Int]) -> QuantizedBatchKVCache {
            let cache = QuantizedBatchKVCache(
                leftPadding: leftPadding, groupSize: Self.groupSize, bits: Self.bits)
            let bSize = leftPadding.count
            let k = varied([bSize, nKVHeads, t, d], phase: 1.1)
            let v = varied([bSize, nKVHeads, t, d], phase: 2.1)
            _ = cache.updateQuantized(keys: k, values: v)
            return cache
        }

        // n == 1, no left padding, no window -> fast path (.none).
        let c1 = freshCache(leftPadding: [0])
        #expect(Self.isNone(c1.makeMask(n: 1, windowSize: nil, returnArray: false)))

        // n > 1 -> materialized mask (prefill/chunked).
        #expect(Self.isArray(c1.makeMask(n: 2, windowSize: nil, returnArray: false)))

        // n == 1 but a sliding window is set -> conservative materialized mask.
        #expect(Self.isArray(c1.makeMask(n: 1, windowSize: 4, returnArray: false)))

        // n == 1 but left padding present -> materialized mask.
        let c2 = freshCache(leftPadding: [3])
        #expect(Self.isArray(c2.makeMask(n: 1, windowSize: nil, returnArray: false)))
    }
}
