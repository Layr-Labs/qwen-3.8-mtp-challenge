import Foundation
import MLX
import Testing

// @testable on both: the stub subclasses BaseKVCache, whose initializer and
// maxSize are internal/non-open, and the probe builds a LagunaModel from an
// in-memory LagunaConfiguration.
@testable import MLXLLM
@testable import MLXLMCommon

// Regression tests for the sliding-window WRAP SEAM in DFlash rollback.
//
// Block decode was broken at the seam and the ranked window is exactly where it
// bites: Laguna's sliding window is 512 and the scored window is a 512-token
// seed plus 128 decode steps, so the first block already crosses the ring
// boundary. It went unnoticed through bring-up because every measurement used a
// 26-68 token prompt.
//
// Mechanism. `RotatingKVCache.isTrimmable` is `offset < maxCacheSize`, which is
// CORRECT and not merely conservative: after the ring wraps, rolling the offset
// back would need the entries the wrap just overwrote, and those are the oldest
// rows still inside the window. A wrapped cache must therefore be rolled back by
// snapshot and replay. The bug was that the snapshot decision happens BEFORE the
// block is written while the trim happens AFTER, so the one round that starts
// trimmable and ends wrapped got neither a snapshot nor a usable trim.
//
// These tests exercise the decision function directly with stub caches, so they
// need no model and no GPU and can run in CI.
@Suite
struct DFlashRollbackSeamTests {
    /// Minimal cache stub: only the properties the snapshot decision reads
    /// (`offset`, `maxSize`, `isTrimmable`) plus the protocol's required members.
    private final class SeamStubCache: BaseKVCache {
        private let capacity: Int?
        private var trimmedRows = 0

        init(offset: Int, maxSize: Int?) {
            self.capacity = maxSize
            super.init()
            self.offset = offset
        }

        override var maxSize: Int? { capacity }

        // Mirrors RotatingKVCache: a ring stops being trimmable once it wraps.
        override var isTrimmable: Bool {
            guard let capacity else { return true }
            return offset < capacity
        }

        override func trim(_ n: Int) -> Int {
            let trimmed = min(offset, n)
            offset -= trimmed
            trimmedRows += trimmed
            return trimmed
        }

        override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
            (keys, values)
        }

        override func copy() -> any KVCache {
            SeamStubCache(offset: offset, maxSize: capacity)
        }
    }

    /// The decision under test is a `DFlashTargetModel` protocol extension, and
    /// `DFlashTargetModel: LLMModel`, so it needs a real conformer rather than a
    /// hand-rolled stub. A tiny `LagunaModel` built from an in-memory config is
    /// enough: `makeDefaultDFlashCacheRollbackState` only inspects the caches it
    /// is handed, so no weights, no tokenizer and no GPU work are involved.
    private static let tinyConfigJSON = """
        {
          "model_type": "laguna",
          "vocab_size": 32,
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 4,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "max_position_embeddings": 4096,
          "rms_norm_eps": 1e-6,
          "attention_bias": false,
          "qkv_bias": false,
          "tie_word_embeddings": false,
          "rope_theta": 500000.0,
          "sliding_window": 8,
          "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
          "gating": "per-head",
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 8,
          "shared_expert_intermediate_size": 8,
          "moe_routed_scaling_factor": 1.0,
          "norm_topk_prob": true,
          "decoder_sparse_step": 1,
          "moe_router_logit_softcapping": 0.0
        }
        """

    private func decisionProbe() throws -> LagunaModel {
        LagunaModel(
            try JSONDecoder.json5().decode(
                LagunaConfiguration.self, from: Data(Self.tinyConfigJSON.utf8)
            )
        )
    }

    // A round that stays comfortably inside the ring needs no snapshot: rolling
    // back is just moving the offset, and copying the cache every round would be
    // pure overhead on the hot path.
    @Test
    func roundWellInsideTheRingNeedsNoSnapshot() throws {
        let cache: [KVCache] = [SeamStubCache(offset: 100, maxSize: 512)]
        let state = try decisionProbe().makeDefaultDFlashCacheRollbackState(
            cache: cache, plannedWriteCount: 8
        )
        #expect(state == nil)
    }

    // THE REGRESSION. Trimmable at the start (500 < 512) but the block write
    // lands past the boundary (500 + 16 >= 512). Before the fix this returned
    // nil and the round then failed with untrimmableCache; it must now snapshot.
    @Test
    func roundThatCrossesTheRingBoundaryTakesASnapshot() throws {
        let cache: [KVCache] = [SeamStubCache(offset: 500, maxSize: 512)]
        #expect(cache[0].isTrimmable, "precondition: trimmable BEFORE the write")

        let state = try decisionProbe().makeDefaultDFlashCacheRollbackState(
            cache: cache, plannedWriteCount: 16
        )
        #expect(
            state is DFlashCopiedTargetRollbackState,
            "a round that ends wrapped must carry a snapshot; without one it can neither trim nor replay"
        )
    }

    // Exactly-at-the-boundary is the same case: offset + width == maxSize means
    // the last written row occupies the final slot and the next write wraps, so
    // the conservative side is the correct side.
    @Test
    func roundLandingExactlyOnTheBoundaryTakesASnapshot() throws {
        let cache: [KVCache] = [SeamStubCache(offset: 504, maxSize: 512)]
        let state = try decisionProbe().makeDefaultDFlashCacheRollbackState(
            cache: cache, plannedWriteCount: 8
        )
        #expect(state is DFlashCopiedTargetRollbackState)
    }

    // An already-wrapped cache was always handled (it reports untrimmable, so the
    // old code snapshotted too). Pinned so a future optimisation cannot regress
    // it back to trusting isTrimmable alone.
    @Test
    func alreadyWrappedCacheTakesASnapshot() throws {
        let cache: [KVCache] = [SeamStubCache(offset: 900, maxSize: 512)]
        #expect(!cache[0].isTrimmable)
        let state = try decisionProbe().makeDefaultDFlashCacheRollbackState(
            cache: cache, plannedWriteCount: 8
        )
        #expect(state is DFlashCopiedTargetRollbackState)
    }

    // Laguna interleaves sliding-window and full-attention layers. A full
    // attention cache (no maxSize) never wraps, so the decision must be driven by
    // the ring-bounded members: one crossing cache forces the snapshot for all.
    @Test
    func mixedFullAttentionAndSlidingCachesFollowTheRingBoundedMember() throws {
        let unbounded = SeamStubCache(offset: 5_000, maxSize: nil)
        #expect(unbounded.isTrimmable, "an unbounded cache trims exactly")

        let safe: [KVCache] = [unbounded, SeamStubCache(offset: 100, maxSize: 512)]
        #expect(
            try decisionProbe().makeDefaultDFlashCacheRollbackState(
                cache: safe, plannedWriteCount: 8
            ) == nil
        )

        let crossing: [KVCache] = [unbounded, SeamStubCache(offset: 508, maxSize: 512)]
        #expect(
            try decisionProbe().makeDefaultDFlashCacheRollbackState(
                cache: crossing, plannedWriteCount: 8
            ) is DFlashCopiedTargetRollbackState
        )
    }

    // Callers that cannot know the width keep the old behaviour, so this fix is
    // additive: only rounds that declare a planned write get the new protection.
    @Test
    func omittingThePlannedWriteCountPreservesTheOldDecision() throws {
        let cache: [KVCache] = [SeamStubCache(offset: 500, maxSize: 512)]
        #expect(
            try decisionProbe().makeDefaultDFlashCacheRollbackState(cache: cache) == nil
        )
    }

    // MARK: - L4 ring-index consistency (contract Amendment 8)

    /// Rows shaped the way a KV cache sees them, with distinguishable contents so
    /// a restore can be compared byte-for-byte rather than only by shape.
    private func kvRows(_ count: Int, base: Int) -> MLXArray {
        MLXArray((0 ..< count).map { Float(base + $0) }, [1, 1, count, 1])
    }

    /// The contract's L4 proposed `idx != offset % maxSize` as the structural
    /// tell for an elision that rewinds the logical offset but leaves the
    /// physical write index — and therefore the rejected bytes — in place.
    ///
    /// That check is INVALID on the path block decode actually takes, and this
    /// test is what establishes it. `RotatingKVCache.update` dispatches on width:
    /// only a single row goes through the in-place ring (`updateInPlace`), where
    /// `idx` really is a ring write index. Anything wider — the seed prefill, and
    /// every K>=2 verify — goes through `updateConcat`, which rebuilds the array
    /// in TEMPORAL order and sets `idx` to the physical row count. On that path
    /// `idx` is not a ring index at all, so comparing it against `offset % maxSize`
    /// would fail honest code on the first block.
    @Test
    func multiRowWritesLeaveTheRingInTemporalOrderNotAtOffsetModuloMaxSize() {
        let maxSize = 511
        let cache = RotatingKVCache(maxSize: maxSize, keep: 0)

        // Seed prefill, wider than the ring.
        _ = cache.update(keys: kvRows(600, base: 0), values: kvRows(600, base: 1000))
        // One K=3 verify block.
        _ = cache.update(keys: kvRows(3, base: 600), values: kvRows(3, base: 1600))

        #expect(cache.offset == 603)
        // Temporal order, not a ring: `idx` tracks the physical row count, which
        // `updateConcat` lets grow to maxSize + S - 1.
        #expect(cache.idx == cache.keys?.dim(2))
        #expect(cache.idx == 513)
        // The invariant L4 asked for would have demanded 603 % 511 == 92 here.
        #expect(cache.offset % maxSize == 92)
    }

    /// What L4 CAN carry: the snapshot the seam fix falls back on is exact.
    ///
    /// This is a self-consistency property, so unlike a candidate-vs-reference KV
    /// digest (Amendment 4) it holds regardless of build. It is also the property
    /// the whole wrap-seam fix rests on — if `copy()` were shallow, or lost `idx`
    /// or the physical row count, rollback past the seam would silently restore
    /// the wrong window.
    @Test
    func copyingARotatingCacheCapturesRingStateExactlyAcrossTheSeam() throws {
        let maxSize = 511
        let cache = RotatingKVCache(maxSize: maxSize, keep: 0)
        _ = cache.update(keys: kvRows(509, base: 0), values: kvRows(509, base: 1000))

        #expect(cache.isTrimmable)
        let snapshot = try #require(cache.copy() as? RotatingKVCache)
        let snapshotKeys = try #require(snapshot.keys)

        // Cross the seam: this write takes the ring from trimmable to wrapped, so
        // the offset can no longer be rewound and only the snapshot can restore it.
        _ = cache.update(keys: kvRows(3, base: 509), values: kvRows(3, base: 1509))
        #expect(cache.offset == 512)
        #expect(cache.isTrimmable == false)

        // Exact in every component the ring's behaviour depends on.
        #expect(snapshot.offset == 509)
        #expect(snapshot.idx == 509)
        #expect(snapshotKeys.dim(2) == 509)
        #expect(snapshot.maxSize == maxSize)

        // ...and in bytes: the post-snapshot write must not have reached through
        // into the snapshot's storage.
        let expected = kvRows(509, base: 0)
        #expect(MLX.all(snapshotKeys .== expected).item(Bool.self))
    }
}
