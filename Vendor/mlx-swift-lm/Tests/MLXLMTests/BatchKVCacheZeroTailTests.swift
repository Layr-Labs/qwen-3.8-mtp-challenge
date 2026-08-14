// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Testing

@Suite("BatchKVCache.zeroTailPerRow")
struct BatchKVCacheZeroTailTests {

    /// Build a BatchKVCache with `B` rows, each containing `T` timesteps of
    /// non-zero marker values so zeroing is observable.
    private func makeCache(B: Int, T: Int, H: Int = 2, D: Int = 4) -> BatchKVCache {
        let cache = BatchKVCache(leftPadding: Array(repeating: 0, count: B))
        let k = MLXArray.ones([B, H, T, D], dtype: .float32)
        let v = MLXArray.ones([B, H, T, D], dtype: .float32) * 2.0
        _ = cache.update(keys: k, values: v)
        return cache
    }

    @Test func fullKeepIsNoop() {
        let cache = makeCache(B: 3, T: 8)
        let keep = MLXArray([Int32(8), 8, 8])
        cache.zeroTailPerRow(keepLengths: keep)
        // All slots should remain 1.0 / 2.0.
        let state = cache.state
        let keysSum = state[0].sum().item(Float.self)
        let valuesSum = state[1].sum().item(Float.self)
        // 3 rows × 2 heads × 8 time × 4 dim = 192 slots at 1.0 → sum == 192
        #expect(keysSum == 192.0)
        #expect(valuesSum == 384.0)
    }

    @Test func zeroKeepErasesEverything() {
        let cache = makeCache(B: 3, T: 8)
        let keep = MLXArray([Int32(0), 0, 0])
        cache.zeroTailPerRow(keepLengths: keep)
        let state = cache.state
        #expect(state[0].sum().item(Float.self) == 0.0)
        #expect(state[1].sum().item(Float.self) == 0.0)
    }

    @Test func mixedPerRowKeep() {
        // B=3, T=8, keep=[2, 5, 8]. Row 0 keeps 2 slots, row 1 keeps 5,
        // row 2 keeps all 8.
        let cache = makeCache(B: 3, T: 8, H: 1, D: 1)
        let keep = MLXArray([Int32(2), 5, 8])
        cache.zeroTailPerRow(keepLengths: keep)
        let state = cache.state
        // Expected keys sum: row0=2, row1=5, row2=8 = 15 (value 1.0 each)
        #expect(state[0].sum().item(Float.self) == 15.0)
        // Values sum: 15 * 2.0 = 30
        #expect(state[1].sum().item(Float.self) == 30.0)
    }

    @Test func mixedKeepLeavesKeptRegionUntouched() {
        // Verify row 1 still has its first 5 values unchanged.
        let cache = makeCache(B: 3, T: 8, H: 1, D: 1)
        let keep = MLXArray([Int32(2), 5, 8])
        cache.zeroTailPerRow(keepLengths: keep)
        let keys = cache.state[0]  // [3, 1, 8, 1]
        // Row 1, first 5 slots, all ones.
        let row1Head = keys[1, 0, ..<5, 0]
        #expect(row1Head.sum().item(Float.self) == 5.0)
        // Row 1, slots 5..<8, all zeros.
        let row1Tail = keys[1, 0, 5..., 0]
        #expect(row1Tail.sum().item(Float.self) == 0.0)
    }

    @Test func emptyCacheIsNoop() {
        // No update() called; keys/values are nil.
        let cache = BatchKVCache(leftPadding: [0, 0])
        cache.zeroTailPerRow(keepLengths: MLXArray([Int32(0), 0]))
        // No crash; state should still report the metadata-only shape.
        #expect(cache.isEmpty())
    }
}
