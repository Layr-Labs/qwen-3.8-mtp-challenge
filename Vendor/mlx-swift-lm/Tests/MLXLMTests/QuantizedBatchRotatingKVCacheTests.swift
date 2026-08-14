// Copyright © 2026 Apple Inc.
//
// Tests for QuantizedBatchRotatingKVCache: the sliding-window counterpart of
// QuantizedBatchKVCache. Verifies that quantizing the sliding-window layers
// (the majority on Gemma-4 / GPT-OSS, previously left fp16) preserves the
// BatchRotatingKVCache window semantics — front-trim, retained length, mask —
// within quantization tolerance.
//
// Group-safety note: quantization groups run along the HEAD-DIM axis
// (groupSize divides head_dim), never the time axis, so the window front-trim
// (dropping whole leading tokens) can never split a group. Tests use D=64,
// groupSize=64 to satisfy MLX's group constraint.

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("QuantizedBatchRotatingKVCache", .serialized)
struct QuantizedBatchRotatingKVCacheTests {

    private let H = 2
    private let D = 64
    private let groupSize = 64
    private let bits = 8

    /// Relative L2 of `a` vs `b`, the quantization-fidelity metric used by the
    /// PR #43 validation.
    private func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        let num = sqrt((a - b).square().sum()).item(Float.self)
        let den = sqrt(b.square().sum()).item(Float.self) + 1e-9
        return num / den
    }

    private func randn(_ B: Int, _ T: Int) -> MLXArray {
        MLXRandom.normal([B, H, T, D])
    }

    // MARK: - Window front-trim parity vs fp16 BatchRotatingKVCache

    @Test("quantized rotating cache mirrors fp16 rotating window length")
    func windowLengthParity() {
        MLXRandom.seed(0)
        let B = 4
        let maxSize = 8
        let fp16 = BatchRotatingKVCache(maxSize: maxSize, leftPadding: Array(repeating: 0, count: B))
        let quant = QuantizedBatchRotatingKVCache(
            maxSize: maxSize, leftPadding: Array(repeating: 0, count: B),
            groupSize: groupSize, bits: bits)

        // Prefill 5 tokens (< window), then decode 10 single tokens (window fills
        // and starts trimming at step 9). The retained time-axis length must
        // track the fp16 reference exactly at every step.
        let prefillT = 5
        let (kp, vp) = (randn(B, prefillT), randn(B, prefillT))
        let (fk, _) = fp16.update(keys: kp, values: vp)
        let (qk, _) = quant.update(keys: kp, values: vp)
        #expect(fk.dim(2) == qk.dim(2))
        #expect(quant.offset == fp16.offset)

        for _ in 0 ..< 10 {
            let (k1, v1) = (randn(B, 1), randn(B, 1))
            let (fkD, _) = fp16.update(keys: k1, values: v1)
            let (qkD, _) = quant.update(keys: k1, values: v1)
            // Retained length identical, and never exceeds the window.
            #expect(fkD.dim(2) == qkD.dim(2))
            #expect(qkD.dim(2) <= maxSize)
            #expect(quant.offset == fp16.offset)
        }
        // After 5 + 10 = 15 tokens with window 8, the cache holds exactly 8.
        #expect(quant.offset == maxSize)
    }

    // MARK: - Numerical fidelity of the retained window

    @Test("dequantized window matches fp16 within quant tolerance")
    func windowValueFidelity() {
        MLXRandom.seed(1)
        let B = 4
        let maxSize = 8
        let fp16 = BatchRotatingKVCache(maxSize: maxSize, leftPadding: Array(repeating: 0, count: B))
        let quant = QuantizedBatchRotatingKVCache(
            maxSize: maxSize, leftPadding: Array(repeating: 0, count: B),
            groupSize: groupSize, bits: bits)

        // Drive both well past the window so the front-trim has fired repeatedly.
        var fpK = MLXArray.zeros([B, H, 0, D])
        var qK = MLXArray.zeros([B, H, 0, D])
        for i in 0 ..< 14 {
            let stepT = i == 0 ? 3 : 1
            let (k, v) = (randn(B, stepT), randn(B, stepT))
            (fpK, _) = fp16.update(keys: k, values: v)
            (qK, _) = quant.update(keys: k, values: v)
        }
        eval(fpK, qK)
        // The quantized retained window dequantizes to the fp16 window within
        // int8 tolerance. (Same logical tokens retained — verified by shape — so
        // this is pure quantization error, not a windowing mismatch.)
        #expect(fpK.dim(2) == qK.dim(2))
        let err = relL2(qK, fpK)
        #expect(err < 0.02, "int8 rotating window relL2 \(err) exceeded 0.02")
    }

    // MARK: - extractBatched restores a windowed single-stream cache

    @Test("extractBatched returns a RotatingKVCache carrying the window")
    func extractRestoresRotating() {
        MLXRandom.seed(2)
        let B = 2
        let maxSize = 8
        let quant = QuantizedBatchRotatingKVCache(
            maxSize: maxSize, leftPadding: Array(repeating: 0, count: B),
            groupSize: groupSize, bits: bits)
        for i in 0 ..< 12 {
            let stepT = i == 0 ? 4 : 1
            _ = quant.update(keys: randn(B, stepT), values: randn(B, stepT))
        }
        let extracted = quant.extractBatched(0)
        // Must be a windowed single-stream cache, not a full-attention one.
        #expect(extracted is RotatingKVCache)
        #expect(extracted.maxSize == maxSize)
    }

    @Test("empty extractBatched returns an empty RotatingKVCache")
    func extractEmptyRotating() {
        let quant = QuantizedBatchRotatingKVCache(
            maxSize: 8, leftPadding: [0], groupSize: groupSize, bits: bits)
        let extracted = quant.extractBatched(0)
        #expect(extracted is RotatingKVCache)
    }

    // MARK: - maxSize / isTrimmable reflect the window

    @Test("maxSize and isTrimmable reflect the window state")
    func windowMetadata() {
        let maxSize = 4
        let quant = QuantizedBatchRotatingKVCache(
            maxSize: maxSize, leftPadding: [0], groupSize: groupSize, bits: bits)
        #expect(quant.maxSize == maxSize)
        #expect(quant.isTrimmable)  // empty < window
        _ = quant.update(keys: randn(1, maxSize), values: randn(1, maxSize))
        #expect(!quant.isTrimmable)  // full → not trimmable, like BatchRotatingKVCache
    }

    // MARK: - Mask fast-path parity (mlx#3384 avoidance on the dequant path)

    @Test("single-token in-window decode mask matches fp16 (.none fast path)")
    func maskFastPathParity() {
        let maxSize = 8
        let fp16 = BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0])
        let quant = QuantizedBatchRotatingKVCache(
            maxSize: maxSize, leftPadding: [0], groupSize: groupSize, bits: bits)

        // Empty cache, single-token decode, no left padding → both must take the
        // `.none` fast path (NOT an explicit array mask, which would route the
        // dequant path back through mlx#3384).
        func isNone(_ m: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
            if case .none = m { return true }
            return false
        }
        let fpMask = fp16.makeMask(n: 1, windowSize: maxSize, returnArray: false)
        let qMask = quant.makeMask(n: 1, windowSize: maxSize, returnArray: false)
        #expect(isNone(fpMask) == isNone(qMask))
        #expect(isNone(qMask), "single-token in-window decode should yield .none, not .array")

        // Multi-token prefill must NOT take the fast path (returns an array mask).
        let fpPrefill = fp16.makeMask(n: 4, windowSize: maxSize, returnArray: false)
        let qPrefill = quant.makeMask(n: 4, windowSize: maxSize, returnArray: false)
        #expect(isNone(fpPrefill) == isNone(qPrefill))
        #expect(!isNone(qPrefill))
    }

    // MARK: - The base (non-windowed) cache is unaffected

    @Test("non-windowed QuantizedBatchKVCache reports nil maxSize and grows")
    func nonWindowedUnbounded() {
        let cache = QuantizedBatchKVCache(
            leftPadding: [0], groupSize: groupSize, bits: bits)
        #expect(cache.maxSize == nil)
        for _ in 0 ..< 20 { _ = cache.update(keys: randn(1, 1), values: randn(1, 1)) }
        // No window → grows unbounded; retained length == all tokens.
        #expect(cache.offset == 20)
    }
}
