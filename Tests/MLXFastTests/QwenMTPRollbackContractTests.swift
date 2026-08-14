import Foundation
import MLX
import MLXFastCore
import MLXLMCommon
@testable import MLXFastModel
import Testing

// The rollback contract of the native-MTP round, tested on SYNTHETIC caches.
//
// Why this file exists at all: the rollback is the one part of the migrated
// driver that fails SILENTLY and LATE. Every other invariant either throws (the
// round-top cache-offset check) or shows up immediately as a wrong token. A
// broken rollback desyncs the 48 recurrent layers from the 16 attention layers
// only on a PARTIAL acceptance, and the damage surfaces many tokens later as a
// divergence with no obvious cause. So the two properties that make it correct
// are pinned here rather than left to a 512-token end-to-end run to notice.
//
// Constructing `MambaCache()` / `KVCacheSimple()` and poking `.offset` allocates
// no MLXArray and loads no model, so the structural half runs on any machine --
// the same precedent `syntheticQwenHybridCaches` sets in
// RuntimeWorkerSupportTests. The half that genuinely needs arrays is gated
// behind MLXFAST_RUN_MLX_RUNTIME_TESTS.

/// The Qwen 3.6 hybrid stack: `KVCacheSimple` on the 16 full-attention layers
/// (index % 4 == 3), `MambaCache` on the 48 gated-delta layers.
private func syntheticMTPCaches(
    attentionOffset: Int,
    layerCount: Int = MLXFastConstants.numHiddenLayers
) -> [any KVCache] {
    let interval = MLXFastConstants.fullAttentionInterval
    return (0 ..< layerCount).map { index in
        let isFullAttention = index % interval == interval - 1
        let cache: BaseKVCache = isFullAttention ? KVCacheSimple() : MambaCache()
        // The recurrent caches never count positions on this stack; only the
        // trimmable ones carry the sequence offset.
        cache.offset = isFullAttention ? attentionOffset : 0
        return cache
    }
}

@Suite
struct QwenMTPRollbackContractTests {

    /// The sequence position is read off a TRIMMABLE cache, never off a
    /// recurrent one.
    ///
    /// This is the shape of a bug this branch already shipped once on the serial
    /// path: `verifyQwenCachePosition` adopted recurrent layer 0's offset as the
    /// sequence position, and layer 0 of this stack is gated-delta and pinned at
    /// zero forever. Here the same mistake would make the round-top invariant
    /// compare `0` against `seed + emitted` and throw on round one -- loudly, but
    /// for entirely the wrong reason.
    @Test
    func theSequencePositionComesFromATrimmableCache() throws {
        let caches = syntheticMTPCaches(attentionOffset: 517)
        #expect(Qwen36MTPBlockSession.trimmableOffset(caches) == 517)
        // Layer 0 is recurrent and pinned at 0; reading it would give 0.
        #expect(caches[0].offset == 0)
        #expect(caches[0] is ArraysCache)
    }

    /// A stack with no trimmable cache reports -1 rather than a plausible 0. The
    /// round-top invariant then throws instead of silently accepting a position
    /// it never actually read.
    @Test
    func aStackWithNoTrimmableCacheReportsAnImpossibleOffset() throws {
        let recurrentOnly: [any KVCache] = (0 ..< 4).map { _ in MambaCache() }
        #expect(Qwen36MTPBlockSession.trimmableOffset(recurrentOnly) == -1)
    }

    /// The trimmable caches lose the WHOLE verify window, and the recurrent ones
    /// are never trimmed.
    ///
    /// `trim()` on an `ArraysCache` only decrements `offset` -- it leaves the
    /// conv/SSM state exactly where the verify forward left it. Trimming a
    /// recurrent cache and calling that a rollback is the specific failure the
    /// vendored `RecurrentRollbackCache` degenerates to when its innovation tape
    /// is nil, which it always is because nothing ever calls `recordTape`.
    @Test
    func rollbackTrimsTheWholeVerifyWindowAndNeverTrimsRecurrentCaches() throws {
        let base = 512
        let depth = 2
        let verified = 1 + depth        // [primary] + drafts
        let caches = syntheticMTPCaches(attentionOffset: base + verified)
        let snapshot = Qwen36MTPBlockSession.snapshotRecurrent(caches)
        // Every recurrent layer is in the snapshot; no trimmable one is.
        #expect(snapshot.count == 48)

        Qwen36MTPBlockSession.rollbackAfterVerify(
            caches, snapshot, verifiedTokens: verified, to: base)

        for (index, cache) in caches.enumerated() {
            if cache is ArraysCache {
                #expect(
                    cache.offset == 0,
                    """
                    recurrent cache \(index) had its offset moved by the \
                    rollback. Recurrent state is restored from the snapshot, \
                    never trimmed: trim() would decrement the offset and leave \
                    the SSM state from the discarded verify window in place.
                    """
                )
            } else {
                #expect(
                    cache.offset == base,
                    """
                    trimmable cache \(index) is at \(cache.offset), not \(base). \
                    A partial acceptance must undo the ENTIRE verify window; \
                    trimming only the rejected tail leaves the attention layers \
                    one frame ahead of the repair forward.
                    """
                )
            }
        }
    }

    /// A rollback is idempotent with respect to a cache already at the base: the
    /// full-acceptance path takes no rollback at all, and a stray call must not
    /// trim a cache backwards past the committed prefix.
    @Test
    func rollbackDoesNothingToACacheAlreadyAtTheBase() throws {
        let base = 512
        let caches = syntheticMTPCaches(attentionOffset: base)
        let snapshot = Qwen36MTPBlockSession.snapshotRecurrent(caches)
        Qwen36MTPBlockSession.rollbackAfterVerify(
            caches, snapshot, verifiedTokens: 3, to: base)
        for cache in caches where !(cache is ArraysCache) {
            #expect(cache.offset == base)
        }
    }

    /// The rollback clears any depth-1 `rollbackState` the GDN forward may have
    /// written. That snapshot describes a frame this rollback just discarded, so
    /// leaving it in place would arm the vendored restore path with a stale
    /// state for a LATER round.
    /// Gated with the snapshot test below: constructing ANY `MLXArray` loads the
    /// default metallib, which is not available on a machine that has not built
    /// it, so a bare `MLXArray([...])` here would fail the whole suite offline.
    @Test(.enabled(if: ProcessInfo.processInfo
        .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"))
    func rollbackClearsTheVendoredDepthOneRollbackState() throws {
        let caches = syntheticMTPCaches(attentionOffset: 515)
        let snapshot = Qwen36MTPBlockSession.snapshotRecurrent(caches)
        let marker = MLXArray([Float(1), 2, 3])
        for cache in caches {
            (cache as? ArraysCache)?.rollbackState = (marker, marker)
        }
        Qwen36MTPBlockSession.rollbackAfterVerify(
            caches, snapshot, verifiedTokens: 3, to: 512)
        for cache in caches {
            #expect((cache as? ArraysCache)?.rollbackState == nil)
        }
    }

    /// THE SNAPSHOT MUST NOT ALIAS THE LIVE CACHE.
    ///
    /// `MLXArray` is a reference type and subscript-assignment mutates it IN
    /// PLACE. `snapshotRecurrent` therefore captures a fresh slice EXPRESSION
    /// (`[.ellipsis]`), which references the array's value at capture time. A
    /// snapshot that stored the bare reference would be rewritten by an in-place
    /// write and the rollback would restore the post-verify state -- correct
    /// today only because the gated-delta forward happens to REBIND its cache
    /// slots, which nothing pins it to.
    ///
    /// This is the regression that would otherwise appear months after someone
    /// optimizes the GDN forward to write in place, as rare late divergence with
    /// no failing assertion anywhere.
    @Test(.enabled(if: ProcessInfo.processInfo
        .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"))
    func theSnapshotSurvivesAnInPlaceWriteToTheLiveCache() throws {
        let cache = MambaCache()
        cache[0] = MLXArray([Float(1), 2, 3, 4])
        cache[1] = MLXArray([Float(9), 9, 9, 9])
        let caches: [any KVCache] = [cache, KVCacheSimple()]
        let snapshot = Qwen36MTPBlockSession.snapshotRecurrent(caches)

        // Simulate a gated-delta forward that mutates its state IN PLACE rather
        // than rebinding the slot.
        cache[0]?[0..<4] = MLXArray([Float(7), 7, 7, 7])
        eval(cache[0]!)

        Qwen36MTPBlockSession.rollbackAfterVerify(
            caches, snapshot, verifiedTokens: 1, to: 0)
        let restored = try #require(cache[0])
        eval(restored)
        #expect(
            restored.asArray(Float.self) == [1, 2, 3, 4],
            """
            the recurrent snapshot did not survive an in-place write to the live \
            cache, so a partial acceptance would restore the POST-verify state \
            and desync the recurrent layers from the trimmed attention layers. \
            snapshotRecurrent must capture `a[i]?[.ellipsis]`, never `a[i]`.
            """
        )
    }
}
