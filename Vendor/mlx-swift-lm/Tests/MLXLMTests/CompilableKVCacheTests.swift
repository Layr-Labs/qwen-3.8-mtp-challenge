// Tests for the compiled-decode foundation: CompilableKVCache (overflow-bin,
// compile-traceable KV cache), DynamicSlice, and CompiledDecode.compileForward.
//
// These pin two properties the future compiled-decode path depends on:
//   1. Numerical equivalence — the overflow-bin full buffer + array mask is
//      equivalent to KVCacheSimple's dynamically-sized [..offset] slice.
//   2. Compile-traceability — a decode step (model forward + cache write) can be
//      captured by `compile(inputs:outputs:)` and re-invoked, with the MLXArray
//      `offsetArray` advancing across compiled calls (proving DynamicSlice +
//      `_updateInternal` thread through the tracer).

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("CompilableKVCacheTests")
struct CompilableKVCacheTests {

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    private func maskArray(
        _ mode: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray? {
        if case .array(let m) = mode { return m }
        return nil
    }

    // MARK: - 1. Overflow-bin equivalence

    /// The valid region of CompilableKVCache's full buffer matches the slice
    /// KVCacheSimple returns, step-for-step, for a prefill + several decodes.
    @Test func overflowBinValidRegionMatchesSimpleCache() {
        let (B, H, D, maxLen) = (1, 2, 4, 16)
        let ck = CompilableKVCache(maxLength: maxLen)
        let simple = KVCacheSimple()

        var offset = 0
        // 3-token prefill, then three single-token decodes.
        for (idx, s) in [3, 1, 1, 1].enumerated() {
            // Distinct value per step so concatenation order is verifiable.
            let k = MLXArray.ones([B, H, s, D]) * Float(idx + 1)
            let v = MLXArray.ones([B, H, s, D]) * Float(-(idx + 1))

            let (ckK, ckV) = ck.update(keys: k, values: v)
            let (skK, skV) = simple.update(keys: k, values: v)
            offset += s

            // Full buffer keeps a constant shape across steps.
            #expect(ckK.shape == [B, H, maxLen, D])
            #expect(ckV.shape == [B, H, maxLen, D])

            let ckValidK = ckK[.ellipsis, ..<offset, 0...]
            let ckValidV = ckV[.ellipsis, ..<offset, 0...]
            #expect(maxAbsDiff(ckValidK, skK) < 1e-6, "keys mismatch at step \(idx)")
            #expect(maxAbsDiff(ckValidV, skV) < 1e-6, "values mismatch at step \(idx)")
            #expect(ck.offset == simple.offset)
        }
    }

    // MARK: - 2. Mask semantics

    /// makeMask marks exactly the valid (causal) positions. For the last query
    /// row, the number of attended keys equals the post-update offset (full
    /// attention) or `min(offset, window)` (sliding window).
    @Test func makeMaskCausalAndWindowedCounts() {
        let (B, H, D, maxLen) = (1, 1, 2, 32)

        // Full attention: drive to offset O, then a decode-step mask (n=1) at
        // pre-update offset O should attend to O+1 positions [0...O].
        let ck = CompilableKVCache(maxLength: maxLen)
        for o in 0 ..< 5 {
            let mode = ck.makeMask(n: 1, windowSize: nil, returnArray: true)
            let mask = try! #require(maskArray(mode))
            #expect(mask.shape == [1, maxLen])
            // pre-update offset == o, so trues = o + 1
            #expect(Int(mask.asType(.int32).sum().item(Int32.self)) == o + 1)
            // advance one token
            _ = ck.update(
                keys: MLXArray.ones([B, H, 1, D]), values: MLXArray.ones([B, H, 1, D]))
        }

        // Sliding window: window w caps attended keys at min(offset+1, w).
        let w = 3
        let cw = CompilableKVCache(maxLength: maxLen)
        for o in 0 ..< 6 {
            let mode = cw.makeMask(n: 1, windowSize: w, returnArray: true)
            let mask = try! #require(maskArray(mode))
            let expected = Swift.min(o + 1, w)
            #expect(
                Int(mask.asType(.int32).sum().item(Int32.self)) == expected,
                "windowed trues at offset \(o)")
            _ = cw.update(
                keys: MLXArray.ones([B, H, 1, D]), values: MLXArray.ones([B, H, 1, D]))
        }
    }

    // MARK: - 3. from: conversion

    /// CompilableKVCache(from: KVCacheSimple) preserves the prefilled KV data and
    /// offset, copying it into the fixed-size buffer at position 0.
    @Test func convertFromSimpleCachePreservesState() {
        let (B, H, D) = (1, 2, 4)
        let simple = KVCacheSimple()
        let k = MLXArray.ones([B, H, 5, D]) * 7
        let v = MLXArray.ones([B, H, 5, D]) * 9
        _ = simple.update(keys: k, values: v)

        let ck = CompilableKVCache(from: simple, maxLength: 16)
        #expect(ck.offset == 5)
        let validK = ck.keys![.ellipsis, ..<5, 0...]
        #expect(maxAbsDiff(validK, k) < 1e-6)
    }

    // MARK: - 4. Compile-traceability via CompiledDecode.compileForward

    /// A compile-traceable stub model: writes the token into the KV cache (pure
    /// MLX ops, no CPU readback) and returns broadcast logits. Unlike
    /// IncrTestLanguageModel it never calls `.asArray`/`.item`, so it can be
    /// captured by `compile()`.
    final class ProbeModel: Module, LanguageModel, KVCacheDimensionProvider {
        let vocabularySize = 8
        var kvHeads: [Int] { [1] }

        func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
            -> PrepareResult
        {
            .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
            let B = inputs.dim(0)
            let S = inputs.dim(1)
            if let cache {
                let f = inputs.reshaped(B, 1, S, 1).asType(.float32)
                for c in cache { _ = c.update(keys: f, values: f * 2) }
            }
            // logits [B, S, vocab] from a pure broadcast — no host readback.
            return MLXArray.zeros([B, S, vocabularySize])
                + inputs.reshaped(B, S, 1).asType(.float32)
        }
    }

    /// compileForward builds a compiled closure that runs the decode forward and
    /// advances the cache's MLXArray offset across compiled invocations, matching
    /// an uncompiled reference run.
    @Test func compileForwardAdvancesOffsetAndMatchesUncompiled() {
        let model = ProbeModel()
        let maxLen = 8

        func prefilled() -> CompilableKVCache {
            let c = CompilableKVCache(maxLength: maxLen)
            // Allocate buffers + set offset BEFORE compiling (so innerState()
            // captures keys/values, not just offsetArray).
            _ = model(
                LMInput.Text(tokens: MLXArray([Int32(1)]).reshaped(1, 1)),
                cache: [c], state: nil)
            eval(c.keys!, c.values!, c.offsetArray)
            return c
        }

        // Uncompiled reference.
        let refCache = prefilled()
        var refLogits: [Float] = []
        for t in [Int32(2), 3, 4] {
            let out = model(
                LMInput.Text(tokens: MLXArray([t]).reshaped(1, 1)),
                cache: [refCache], state: nil
            ).logits
            eval(out)
            refLogits.append(out.reshaped(-1)[0].item(Float.self))
        }

        // Compiled.
        let cache = prefilled()
        #expect(CompiledDecode.eligible([cache]))
        let forward = CompiledDecode.compileForward(model: model, cacheRef: [cache])
        var gotLogits: [Float] = []
        for t in [Int32(2), 3, 4] {
            let out = forward([MLXArray([t]).reshaped(1, 1)])[0]
            eval(out)
            gotLogits.append(out.reshaped(-1)[0].item(Float.self))
        }

        // Both did 1 prefill + 3 decode steps.
        #expect(refCache.offset == 4)
        #expect(cache.offset == 4)
        for (a, b) in zip(refLogits, gotLogits) {
            #expect(Swift.abs(a - b) < 1e-5)
        }
    }

    // MARK: - compiledCaptureSnapshot (prefix capture on finish)

    /// A full-attention `CompilableKVCache` snapshots into a plain
    /// `KVCacheSimple` holding exactly the valid, temporally ordered prefix,
    /// so a prefix-cache consumer reads correct state (never the compiled
    /// object's decode-only buffers).
    @Test("compiledCaptureSnapshot promotes full-attention cache to KVCacheSimple")
    func compiledCaptureSnapshotFullAttention() {
        let B = 1, H = 2, D = 4
        let cache = CompilableKVCache(maxLength: 16)
        _ = cache.update(keys: MLXArray.ones([B, H, 3, D]), values: MLXArray.ones([B, H, 3, D]) * 2)
        _ = cache.update(keys: MLXArray.ones([B, H, 1, D]) * 3, values: MLXArray.ones([B, H, 1, D]) * 4)
        eval(cache.offsetArray)

        let snap = GenerationBatch.compiledCaptureSnapshot(cache)

        // Must be a plain KVCacheSimple so PrefixCache.storePrefix accepts it.
        #expect(snap is KVCacheSimple)
        let s = snap.state
        #expect(s.count == 2)
        // Exactly the 4 valid tokens (not the 16-wide overflow buffer).
        #expect(s[0].dim(2) == 4)
        #expect(snap.offset == 4)
        // Temporally ordered: first 3 keys == 1, last key == 3.
        #expect(maxAbsDiff(s[0][.ellipsis, 0 ..< 3, 0...], MLXArray.ones([B, H, 3, D])) == 0)
        #expect(maxAbsDiff(s[0][.ellipsis, 3 ..< 4, 0...], MLXArray.ones([B, H, 1, D]) * 3) == 0)
    }

    /// A sliding-window `CompilableRotatingKVCache` only holds the last
    /// `windowSize` tokens, so it must NOT be coerced into a `KVCacheSimple`
    /// (that would present windowed KV as a full-history prefix). It is returned
    /// unchanged so `PrefixCache`'s `as? KVCacheSimple` guard excludes the
    /// whole response from block prefix storage.
    @Test("compiledCaptureSnapshot leaves rotating cache non-KVCacheSimple")
    func compiledCaptureSnapshotRotatingExcluded() {
        let B = 1, H = 2, D = 4
        let cache = CompilableRotatingKVCache(maxSize: 8)
        _ = cache.update(keys: MLXArray.ones([B, H, 2, D]), values: MLXArray.ones([B, H, 2, D]))

        let snap = GenerationBatch.compiledCaptureSnapshot(cache)
        #expect(!(snap is KVCacheSimple))
    }
}
