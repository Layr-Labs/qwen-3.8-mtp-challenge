// Tests for ContinuousBatchingV2 workstream D — BlockHasher + PrefixCacheV2.
//
// Runs without model weights: KV state is mocked with tiny position-coded
// arrays (contract shape [1, kvHeads, offset, headDim]) so donated snapshots
// and adopted slices are verifiable by value.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - Helpers

private func hex(_ data: Data) -> String {
    data.map { String(format: "%02x", $0) }.joined()
}

private func fullLayer() -> CBv2LayerKind {
    CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
}

private func windowedLayer(_ window: Int) -> CBv2LayerKind {
    CBv2LayerKind(attention: .slidingWindow(window), headDim: 1, kvHeads: 1, queryHeads: 1)
}

private func sharedLayer(source: Int, window: Int? = nil) -> CBv2LayerKind {
    let attention: CBv2LayerKind.Attention
    if let window { attention = .slidingWindow(window) } else { attention = .full }
    return CBv2LayerKind(
        attention: attention,
        sharesKVWithLayer: source, headDim: 1, kvHeads: 1, queryHeads: 1)
}

/// Mock per-sequence KV (spec: do NOT depend on WS-A internals). Keys encode
/// the temporal position t at [0, 0, t, 0]; values encode 1000 + t.
private final class MockSequenceKV: CBv2SequenceKV {
    private(set) var absoluteOffset: Int
    var retainedCount: Int { absoluteOffset }
    private let keys: MLXArray
    private let values: MLXArray
    let byteCount: Int

    init(offset: Int) {
        self.absoluteOffset = offset
        self.keys = MLXArray((0 ..< offset).map { Float($0) }).reshaped([1, 1, offset, 1])
        self.values = MLXArray((0 ..< offset).map { Float(1000 + $0) })
            .reshaped([1, 1, offset, 1])
        self.byteCount = keys.nbytes + values.nbytes
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)  // unused in these tests
    }

    func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        (keys, values, absoluteOffset)
    }

    func rollback(_ n: Int) { absoluteOffset = max(0, absoluteOffset - n) }
}

/// Bytes one MockSequenceKV(offset:) pins: 2 arrays × offset × 4 (float32).
private func mockBytes(offset: Int) -> Int { 2 * offset * 4 }

private func makeCache(
    blockSize: Int = 4, salt: String = "", maxBytes: Int? = nil
) -> PrefixCacheV2 {
    PrefixCacheV2(
        config: CBv2PrefixCacheConfig(
            blockSize: blockSize, modelName: "test", cacheSalt: salt, maxBytes: maxBytes))
}

// MARK: - §1 Chain-hash vectors

final class CBv2PrefixCacheHasherTests: XCTestCase {

    // Fixed vectors computed independently (Python hashlib) against the
    // documented format:
    //   h_i = SHA256(model ‖ (parent ?? "omlx-root")
    //                ‖ [first block: "cbv2-salt-v1" ‖ u64le(len) ‖ salt]
    //                ‖ "(t0, t1, …)")

    func testFixedVectorNoModelNoSalt() {
        let hasher = CBv2BlockHasher(blockSize: 3)
        let h = hasher.blockHash(parent: nil, blockTokens: [1, 2, 3])
        XCTAssertEqual(
            hex(h), "bbae1d7b58b0342987ac2fc87c5c3a190f737d0357d739d9c3fb1f65b2df8d4f")
    }

    func testFixedVectorModelAndChain() {
        let hasher = CBv2BlockHasher(blockSize: 3, modelName: "m")
        let h1 = hasher.blockHash(parent: nil, blockTokens: [1, 2, 3])
        XCTAssertEqual(
            hex(h1), "87540c796cfe1ad7f636e1429ee7e96b9df90ac1ea8b3c79c844ecdc64cf1eef")
        let h2 = hasher.blockHash(parent: h1, blockTokens: [4, 5, 6])
        XCTAssertEqual(
            hex(h2), "d03e1c265f8b6b657b243049714a11629d4d6f18c270895077656ba8d56ac3da")
        // chainHashes must produce exactly [h1, h2] for two whole blocks.
        XCTAssertEqual(hasher.chainHashes(tokens: [1, 2, 3, 4, 5, 6]), [h1, h2])
        // Partial third block is not hashed.
        XCTAssertEqual(hasher.chainHashes(tokens: [1, 2, 3, 4, 5, 6, 7]), [h1, h2])
    }

    func testSaltSeparationVectors() {
        let unsalted = CBv2BlockHasher(blockSize: 3, modelName: "m")
        let tenantA = CBv2BlockHasher(blockSize: 3, modelName: "m", cacheSalt: "tenantA")
        let tenantB = CBv2BlockHasher(blockSize: 3, modelName: "m", cacheSalt: "tenantB")

        let hA = tenantA.blockHash(parent: nil, blockTokens: [1, 2, 3])
        let hB = tenantB.blockHash(parent: nil, blockTokens: [1, 2, 3])
        let h0 = unsalted.blockHash(parent: nil, blockTokens: [1, 2, 3])

        XCTAssertEqual(
            hex(hA), "3d85d37a09c31f646b3e3c4f7634c73eece1c87ae3911292fd8f8c6f932cee83")
        XCTAssertEqual(
            hex(hB), "a5e78ee5a02603c59c7bf70eabd87e7b0e42bda81567c86bc4673eabcf0a66a3")
        // Three distinct namespaces for identical tokens.
        XCTAssertNotEqual(hA, hB)
        XCTAssertNotEqual(hA, h0)
        XCTAssertNotEqual(hB, h0)
    }

    func testSaltAppliesToFirstBlockOnlyAndPropagatesViaChain() {
        let unsalted = CBv2BlockHasher(blockSize: 3, modelName: "m")
        let tenantA = CBv2BlockHasher(blockSize: 3, modelName: "m", cacheSalt: "tenantA")

        let h1 = tenantA.blockHash(parent: nil, blockTokens: [1, 2, 3])
        let h2 = tenantA.blockHash(parent: h1, blockTokens: [4, 5, 6])
        XCTAssertEqual(
            hex(h2), "15fa28e9dd45684845d5e2a178c586f6b00db04e5bce956f1a6515818b8fe939")
        // Non-first blocks mix no salt bytes: the salted and unsalted hashers
        // agree given the same parent (the tenant scope rides the chain).
        XCTAssertEqual(h2, unsalted.blockHash(parent: h1, blockTokens: [4, 5, 6]))
    }

    func testEmptySaltIsByteCompatibleWithLegacyCheckpointTier() {
        // The SSD checkpoint tier interop contract: with an empty salt the v2
        // chain is byte-identical to legacy computeBlockHash (PrefixCache.swift).
        let tokens = Array(0 ..< 12)
        let hasher = CBv2BlockHasher(blockSize: 4, modelName: "test_model")
        let v2Chain = hasher.chainHashes(tokens: tokens)

        var parent: Data?
        var legacyChain: [Data] = []
        for i in 0 ..< 3 {
            let block = Array(tokens[(i * 4) ..< (i * 4 + 4)])
            let h = computeBlockHash(
                parentHash: parent, tokenIds: block, modelName: "test_model")
            legacyChain.append(h)
            parent = h
        }
        XCTAssertEqual(v2Chain, legacyChain)
    }

    func testFullBlockCountAndLookupCap() {
        let hasher = CBv2BlockHasher(blockSize: 4)
        XCTAssertEqual(hasher.fullBlockCount(tokenCount: 0), 0)
        XCTAssertEqual(hasher.fullBlockCount(tokenCount: 3), 0)
        XCTAssertEqual(hasher.fullBlockCount(tokenCount: 4), 1)
        XCTAssertEqual(hasher.fullBlockCount(tokenCount: 11), 2)

        // vLLM last-token rule: at least one token stays uncached, so an
        // exact block multiple reserves its final block.
        XCTAssertEqual(hasher.maxLookupBlocks(tokenCount: 0), 0)
        XCTAssertEqual(hasher.maxLookupBlocks(tokenCount: 4), 0)
        XCTAssertEqual(hasher.maxLookupBlocks(tokenCount: 5), 1)
        XCTAssertEqual(hasher.maxLookupBlocks(tokenCount: 8), 1)
        XCTAssertEqual(hasher.maxLookupBlocks(tokenCount: 9), 2)
        XCTAssertEqual(hasher.maxLookupBlocks(tokenCount: 13), 3)
    }

    func testChainHashesMaxBlocksCap() {
        let hasher = CBv2BlockHasher(blockSize: 4, modelName: "m")
        let tokens = Array(0 ..< 12)
        let full = hasher.chainHashes(tokens: tokens)
        XCTAssertEqual(full.count, 3)
        XCTAssertEqual(hasher.chainHashes(tokens: tokens, maxBlocks: 2), Array(full.prefix(2)))
        XCTAssertEqual(hasher.chainHashes(tokens: tokens, maxBlocks: 0), [])
    }
}

// MARK: - §2 Donate → lookup roundtrip

final class CBv2PrefixCacheTests: XCTestCase {

    func testDonateLookupRoundtrip() {
        let cache = makeCache()
        // Gemma-style mix: full layer owns storage, windowed layer excluded,
        // KV-sharing layer owns nothing.
        let kinds = [fullLayer(), windowedLayer(4), sharedLayer(source: 0)]
        let tokens = Array(0 ..< 10)  // 2 whole blocks (8) + partial tail
        let state: [CBv2SequenceKV?] = [
            MockSequenceKV(offset: 10), MockSequenceKV(offset: 10), nil,
        ]

        cache.donate(tokens: tokens, state: state, layerKinds: kinds)
        // Only the full-attention storage-owning layer is retained.
        XCTAssertEqual(cache.bytesInUse, mockBytes(offset: 10))

        guard let hit = cache.lookup(tokens: tokens, layerKinds: kinds) else {
            return XCTFail("expected a prefix hit")
        }
        XCTAssertEqual(hit.matched, 8)  // whole blocks only; tail dropped
        XCTAssertEqual(hit.prefix.count, 3)
        XCTAssertNil(hit.prefix[1], "windowed layer must never be adopted")
        XCTAssertNil(hit.prefix[2], "KV-sharing layer owns no storage")

        guard let layer0 = hit.prefix[0] else { return XCTFail("full layer missing") }
        XCTAssertEqual(layer0.offset, 8)
        XCTAssertEqual(layer0.keys.shape, [1, 1, 8, 1])
        XCTAssertEqual(layer0.values.shape, [1, 1, 8, 1])
        // Value check: temporal positions 0..<8, tail (8, 9) sliced away.
        XCTAssertEqual(
            layer0.keys.reshaped([8]).asArray(Float.self), (0 ..< 8).map(Float.init))
        XCTAssertEqual(
            layer0.values.reshaped([8]).asArray(Float.self), (0 ..< 8).map { Float(1000 + $0) })

        let stats = cache.stats()
        XCTAssertEqual(stats.hits, 1)
        XCTAssertEqual(stats.tokensSaved, 8)
        XCTAssertEqual(stats.entryCount, 1)
    }

    func testLookupMissesOnEmptyCache() {
        let cache = makeCache()
        XCTAssertNil(cache.lookup(tokens: Array(0 ..< 10), layerKinds: [fullLayer()]))
        XCTAssertEqual(cache.stats().misses, 1)
    }

    func testLastTokenCapOnExactBlockMultiple() {
        let cache = makeCache()
        let kinds = [fullLayer()]
        let tokens = Array(0 ..< 8)  // exactly 2 blocks
        cache.donate(
            tokens: tokens, state: [MockSequenceKV(offset: 8)], layerKinds: kinds)

        // Same 8 tokens: the final token must be recomputed for logits, so
        // only 1 block (4 tokens) may match despite 2 being stored.
        guard let hit = cache.lookup(tokens: tokens, layerKinds: kinds) else {
            return XCTFail("expected a capped hit")
        }
        XCTAssertEqual(hit.matched, 4)
        XCTAssertEqual(hit.prefix[0]?.offset, 4)
        XCTAssertEqual(hit.prefix[0]?.keys.shape, [1, 1, 4, 1])
    }

    func testLongestMatchWinsAmongNestedPrefixes() {
        let cache = makeCache()
        let kinds = [fullLayer()]

        // Short donation first: 1 whole block.
        let short = Array(0 ..< 5)
        cache.donate(tokens: short, state: [MockSequenceKV(offset: 5)], layerKinds: kinds)
        // Longer donation sharing block 0: 3 whole blocks. The shared key
        // repoints to the longer entry and the shadowed short one is freed.
        let long = Array(0 ..< 13)
        cache.donate(tokens: long, state: [MockSequenceKV(offset: 13)], layerKinds: kinds)
        XCTAssertEqual(cache.bytesInUse, mockBytes(offset: 13))
        XCTAssertEqual(cache.stats().entryCount, 1)

        // Full lookup: longest match (3 blocks = 12 of 13 tokens).
        guard let hit = cache.lookup(tokens: long, layerKinds: kinds) else {
            return XCTFail("expected longest-prefix hit")
        }
        XCTAssertEqual(hit.matched, 12)

        // Diverging after block 0: only the nested 1-block prefix matches.
        var diverged = long
        diverged[5] = 999
        guard let partial = cache.lookup(tokens: diverged, layerKinds: kinds) else {
            return XCTFail("expected nested 1-block hit")
        }
        XCTAssertEqual(partial.matched, 4)
        XCTAssertEqual(
            partial.prefix[0]?.keys.reshaped([4]).asArray(Float.self),
            (0 ..< 4).map(Float.init))
    }

    func testMissOnDivergenceMidBlock() {
        let cache = makeCache()
        let kinds = [fullLayer()]
        cache.donate(
            tokens: Array(0 ..< 10), state: [MockSequenceKV(offset: 10)], layerKinds: kinds)

        var tokens = Array(0 ..< 10)
        tokens[2] = 999  // diverges inside block 0
        XCTAssertNil(cache.lookup(tokens: tokens, layerKinds: kinds))
    }

    func testDonateWithoutWholeBlockIsDropped() {
        let cache = makeCache()
        cache.donate(
            tokens: [1, 2, 3], state: [MockSequenceKV(offset: 3)], layerKinds: [fullLayer()])
        XCTAssertEqual(cache.bytesInUse, 0)
        XCTAssertEqual(cache.stats().entryCount, 0)
    }

    func testDonateDedupKeepsSingleEntry() {
        let cache = makeCache()
        let kinds = [fullLayer()]
        let tokens = Array(0 ..< 10)
        cache.donate(tokens: tokens, state: [MockSequenceKV(offset: 10)], layerKinds: kinds)
        cache.donate(tokens: tokens, state: [MockSequenceKV(offset: 10)], layerKinds: kinds)
        XCTAssertEqual(cache.stats().entryCount, 1)
        XCTAssertEqual(cache.bytesInUse, mockBytes(offset: 10))
    }

    func testDonateDroppedWhenFullLayerStateMissing() {
        let cache = makeCache()
        cache.donate(tokens: Array(0 ..< 10), state: [nil], layerKinds: [fullLayer()])
        XCTAssertEqual(cache.bytesInUse, 0)
    }

    func testDonateDroppedWhenSnapshotTooShort() {
        let cache = makeCache()
        // 10 tokens ⇒ 8-token whole-block prefix, but the snapshot only
        // covers 6 positions — it cannot seed the prefix.
        cache.donate(
            tokens: Array(0 ..< 10), state: [MockSequenceKV(offset: 6)],
            layerKinds: [fullLayer()])
        XCTAssertEqual(cache.bytesInUse, 0)
        XCTAssertNil(cache.lookup(tokens: Array(0 ..< 10), layerKinds: [fullLayer()]))
    }

    func testAllWindowedModelStoresNothing() {
        let cache = makeCache()
        let kinds = [windowedLayer(4), windowedLayer(4)]
        cache.donate(
            tokens: Array(0 ..< 10),
            state: [MockSequenceKV(offset: 10), MockSequenceKV(offset: 10)],
            layerKinds: kinds)
        XCTAssertEqual(cache.bytesInUse, 0)
        XCTAssertNil(cache.lookup(tokens: Array(0 ..< 10), layerKinds: kinds))
    }

    func testSaltedCacheRoundtrips() {
        let cache = makeCache(salt: "tenantA")
        let kinds = [fullLayer()]
        let tokens = Array(0 ..< 10)
        cache.donate(tokens: tokens, state: [MockSequenceKV(offset: 10)], layerKinds: kinds)
        XCTAssertEqual(cache.lookup(tokens: tokens, layerKinds: kinds)?.matched, 8)
    }

    func testLayerCountMismatchIsMiss() {
        let cache = makeCache()
        cache.donate(
            tokens: Array(0 ..< 10), state: [MockSequenceKV(offset: 10)],
            layerKinds: [fullLayer()])
        // A lookup describing a different layer layout must not adopt.
        XCTAssertNil(
            cache.lookup(tokens: Array(0 ..< 10), layerKinds: [fullLayer(), fullLayer()]))
    }
}

// MARK: - §3 LRU eviction, byte budget, in-use pins

final class CBv2PrefixCacheEvictionTests: XCTestCase {

    private let kinds = [fullLayer()]

    /// Donate a 1-block entry with a disjoint token namespace.
    private func donateEntry(_ cache: PrefixCacheV2, base: Int) -> [Int] {
        let tokens = Array(base ..< base + 5)
        cache.donate(tokens: tokens, state: [MockSequenceKV(offset: 5)], layerKinds: kinds)
        return tokens
    }

    func testLRUEvictionOrder() {
        let cache = makeCache()
        let t1 = donateEntry(cache, base: 10)
        let t2 = donateEntry(cache, base: 20)
        let t3 = donateEntry(cache, base: 30)
        XCTAssertEqual(cache.bytesInUse, 3 * mockBytes(offset: 5))

        // Touch entry 1 so it becomes most-recent; release its pin.
        XCTAssertNotNil(cache.lookup(tokens: t1, layerKinds: kinds))
        cache.endAdoption(tokens: t1, matched: 4)

        cache.evict(toFit: mockBytes(offset: 5))
        XCTAssertEqual(cache.bytesInUse, mockBytes(offset: 5))

        XCTAssertNotNil(cache.lookup(tokens: t1, layerKinds: kinds))
        cache.endAdoption(tokens: t1, matched: 4)
        XCTAssertNil(cache.lookup(tokens: t2, layerKinds: kinds))
        XCTAssertNil(cache.lookup(tokens: t3, layerKinds: kinds))
    }

    func testEvictionRespectsInUseEntries() {
        let cache = makeCache()
        let t1 = donateEntry(cache, base: 10)
        _ = donateEntry(cache, base: 20)

        // Pin entry 1 (adoption in flight), then demand a zero-byte fit.
        XCTAssertNotNil(cache.lookup(tokens: t1, layerKinds: kinds))
        cache.evict(toFit: 0)
        XCTAssertEqual(
            cache.bytesInUse, mockBytes(offset: 5),
            "pinned entry must survive eviction; unpinned one must not")
        XCTAssertNotNil(cache.lookup(tokens: t1, layerKinds: kinds))
        cache.endAdoption(tokens: t1, matched: 4)

        // Release both pins (two lookups above) — now it is evictable.
        cache.endAdoption(tokens: t1, matched: 4)
        cache.evict(toFit: 0)
        XCTAssertEqual(cache.bytesInUse, 0)
        XCTAssertEqual(cache.stats().entryCount, 0)
    }

    func testDonateAutoEvictsToConfiguredBudget() {
        // Budget fits two 1-block entries (40 bytes each).
        let cache = makeCache(maxBytes: 2 * mockBytes(offset: 5))
        let t1 = donateEntry(cache, base: 10)
        let t2 = donateEntry(cache, base: 20)
        let t3 = donateEntry(cache, base: 30)

        XCTAssertEqual(cache.bytesInUse, 2 * mockBytes(offset: 5))
        XCTAssertNil(cache.lookup(tokens: t1, layerKinds: kinds), "LRU entry auto-evicted")
        XCTAssertNotNil(cache.lookup(tokens: t2, layerKinds: kinds))
        XCTAssertNotNil(cache.lookup(tokens: t3, layerKinds: kinds))
    }

    func testRepointNeverStealsKeysFromPinnedEntry() {
        let cache = makeCache()
        let short = Array(0 ..< 5)
        cache.donate(tokens: short, state: [MockSequenceKV(offset: 5)], layerKinds: kinds)

        // Pin the short entry mid-adoption.
        XCTAssertEqual(cache.lookup(tokens: short, layerKinds: kinds)?.matched, 4)

        // A longer overlapping donation arrives while the pin is held: it
        // must NOT steal the shared block-0 key (endAdoption must still
        // resolve the pinned entry), but its own longer keys register.
        let long = Array(0 ..< 13)
        cache.donate(tokens: long, state: [MockSequenceKV(offset: 13)], layerKinds: kinds)
        XCTAssertEqual(cache.stats().entryCount, 2)
        XCTAssertEqual(
            cache.bytesInUse, mockBytes(offset: 5) + mockBytes(offset: 13))

        cache.endAdoption(tokens: short, matched: 4)

        guard let hit = cache.lookup(tokens: long, layerKinds: kinds) else {
            return XCTFail("expected long-entry hit")
        }
        XCTAssertEqual(hit.matched, 12)

        // The short entry is unpinned now — evictable as normal LRU.
        cache.endAdoption(tokens: long, matched: 12)
        cache.evict(toFit: mockBytes(offset: 13))
        XCTAssertEqual(cache.bytesInUse, mockBytes(offset: 13))
    }

    /// Codex P2 regression — a pinned shorter entry bequeaths its shared
    /// boundary keys to a longer resident entry when it is evicted.
    ///
    /// A longer donation that arrives while a shorter overlapping entry is
    /// PINNED cannot repoint the shared boundary hash (a pinned entry's
    /// in-flight adoption must keep resolving by that hash), so it registers
    /// only its own extra blocks. Previously, evicting the (now unpinned)
    /// shorter entry deleted that boundary outright, so lookups for the
    /// shorter prefix MISSED FOREVER even though the KV was still resident
    /// inside the longer entry. `removeLocked` now hands each boundary to the
    /// longest resident heir that covers it.
    func testEvictedPinnedShorterEntryBequeathsKeysToLongerEntry() {
        let cache = makeCache()
        let short = Array(0 ..< 5)
        let long = Array(0 ..< 13)

        // donate short → pin short (lookup).
        cache.donate(tokens: short, state: [MockSequenceKV(offset: 5)], layerKinds: kinds)
        XCTAssertEqual(cache.lookup(tokens: short, layerKinds: kinds)?.matched, 4)

        // donate long: shares block-0 with the pinned short entry, so it may
        // not steal block-0's key (endAdoption of short must still resolve).
        cache.donate(tokens: long, state: [MockSequenceKV(offset: 13)], layerKinds: kinds)
        XCTAssertEqual(cache.stats().entryCount, 2)

        // unpin → evict short. Its block-0 boundary is INHERITED by the
        // longer entry, not deleted.
        cache.endAdoption(tokens: short, matched: 4)
        cache.evict(toFit: mockBytes(offset: 13))
        XCTAssertEqual(cache.stats().entryCount, 1, "only the longer entry remains")
        XCTAssertEqual(cache.bytesInUse, mockBytes(offset: 13))

        // lookup short must HIT the long entry (sliced to the shared prefix).
        guard let hit = cache.lookup(tokens: Array(long[..<5]), layerKinds: kinds) else {
            return XCTFail("shorter prefix must still resolve via the surviving longer entry")
        }
        XCTAssertEqual(hit.matched, 4, "1-block prefix served as a slice of the longer entry")
        cache.endAdoption(tokens: Array(long[..<5]), matched: 4)

        // The longer entry itself still resolves at full length.
        XCTAssertEqual(cache.lookup(tokens: long, layerKinds: kinds)?.matched, 12)
        cache.endAdoption(tokens: long, matched: 12)
    }

    /// When NO resident entry covers an evicted entry's boundary (every
    /// overlapping entry is gone too), the key is dropped — and a later
    /// re-donation of the same prefix re-registers it (dedup path).
    func testEvictionDropsKeyWithNoHeirThenDedupReregisters() {
        let cache = makeCache()
        let long = Array(0 ..< 13)

        cache.donate(tokens: long, state: [MockSequenceKV(offset: 13)], layerKinds: kinds)
        // Evict the only entry: no heir exists, so block-0's key is dropped.
        cache.evict(toFit: 0)
        XCTAssertEqual(cache.stats().entryCount, 0)
        XCTAssertNil(cache.lookup(tokens: Array(long[..<5]), layerKinds: kinds))

        // Re-donating re-registers every boundary from scratch.
        cache.donate(tokens: long, state: [MockSequenceKV(offset: 13)], layerKinds: kinds)
        XCTAssertEqual(cache.stats().entryCount, 1)
        guard let hit = cache.lookup(tokens: Array(long[..<5]), layerKinds: kinds) else {
            return XCTFail("1-block prefix must hit after re-donation")
        }
        XCTAssertEqual(hit.matched, 4)
        cache.endAdoption(tokens: Array(long[..<5]), matched: 4)
    }

    func testEndAdoptionOnUnknownPrefixIsNoOp() {
        let cache = makeCache()
        cache.endAdoption(tokens: Array(0 ..< 8), matched: 4)  // nothing stored
        cache.endAdoption(tokens: Array(0 ..< 8), matched: 3)  // not block-aligned
        cache.endAdoption(tokens: Array(0 ..< 8), matched: 0)
        XCTAssertEqual(cache.bytesInUse, 0)
    }

    func testConcurrentDonateLookupEvictIsRaceFree() {
        let cache = makeCache()
        let iterations = 64
        // Mocks pre-built on this thread (offset == whole-block prefix, so
        // hits return the stored arrays without building new MLX graph nodes)
        // — the concurrent section exercises the CACHE's locking only.
        let mocks = (0 ..< iterations).map { _ in MockSequenceKV(offset: 8) }
        let prefixes = (0 ..< 8).map { Array(($0 * 100) ..< ($0 * 100 + 9)) }

        DispatchQueue.concurrentPerform(iterations: iterations) { i in
            let tokens = prefixes[i % prefixes.count]
            cache.donate(tokens: tokens, state: [mocks[i]], layerKinds: kinds)
            if let hit = cache.lookup(tokens: tokens, layerKinds: kinds) {
                cache.endAdoption(tokens: tokens, matched: hit.matched)
            }
            if i % 4 == 0 {
                cache.evict(toFit: 4 * mockBytes(offset: 8))
            }
        }

        // Accounting is consistent and every pin was released: a full evict
        // must be able to drain the cache to zero.
        let stats = cache.stats()
        XCTAssertEqual(stats.bytesInUse, cache.bytesInUse)
        XCTAssertLessThanOrEqual(stats.entryCount, prefixes.count)
        cache.evict(toFit: 0)
        XCTAssertEqual(cache.bytesInUse, 0)
        XCTAssertEqual(cache.stats().entryCount, 0)
    }
}

// MARK: - §4 Windowed-layer policy (requiredRecompute)

final class CBv2PrefixCacheWindowedPolicyTests: XCTestCase {

    /// Gemma-4-like: interleaved sliding(512)/full with KV-sharing layers.
    private var gemma4Like: [CBv2LayerKind] {
        [
            windowedLayer(512), windowedLayer(512), fullLayer(),
            sharedLayer(source: 0, window: 512), sharedLayer(source: 2),
            windowedLayer(512), fullLayer(),
        ]
    }

    /// GPT-OSS-like: alternating sliding(128)/full.
    private var gptossLike: [CBv2LayerKind] {
        [windowedLayer(128), fullLayer(), windowedLayer(128), fullLayer()]
    }

    func testGemma4LikeRecomputeScalesWithWindowedDepth() {
        // gemma4Like has 4 windowed layers (indices 0, 1, 5 own storage;
        // index 3 is a KV-shared windowed layer whose borrowed attention
        // output still pollutes downstream). 4 × 512 = 2048 > 1024, so the
        // recompute clamps to the whole matched prefix (near-full recompute
        // — the correct fallback for a deep hybrid).
        XCTAssertEqual(
            PrefixCacheV2.requiredRecompute(layerKinds: gemma4Like, matched: 1024), 1024)
    }

    func testGPTOSSLikeRecomputeScalesWithWindowedDepth() {
        // gptossLike has 2 windowed(128) layers: 2 × 128 = 256 trailing
        // tokens replayed (was 128 under the single-window bound).
        XCTAssertEqual(
            PrefixCacheV2.requiredRecompute(layerKinds: gptossLike, matched: 1024), 256)
    }

    func testAllFullModelRequiresNoRecompute() {
        let kinds = [fullLayer(), fullLayer(), sharedLayer(source: 0)]
        XCTAssertEqual(PrefixCacheV2.requiredRecompute(layerKinds: kinds, matched: 1024), 0)
    }

    func testSingleWindowedLastLayerRecomputesOneWindow() {
        // The TinyTestModel shape: full then a single windowed(16) layer.
        // One windowed layer ⇒ recompute == window (unchanged), because its
        // polluted early outputs feed nothing downstream.
        let kinds = [fullLayer(), windowedLayer(16)]
        XCTAssertEqual(PrefixCacheV2.requiredRecompute(layerKinds: kinds, matched: 1024), 16)
    }

    func testRecomputeClampedToMatchedPrefix() {
        // A 256-token match cannot re-prefill more than 256 tokens (the
        // gemma4Like product 4 × 512 already exceeds it).
        XCTAssertEqual(
            PrefixCacheV2.requiredRecompute(layerKinds: gemma4Like, matched: 256), 256)
        XCTAssertEqual(
            PrefixCacheV2.requiredRecompute(layerKinds: gemma4Like, matched: 0), 0)
    }

    func testInstanceConvenienceMatchesStatic() {
        let cache = makeCache()
        XCTAssertEqual(
            cache.requiredRecompute(layerKinds: gptossLike, matched: 512),
            PrefixCacheV2.requiredRecompute(layerKinds: gptossLike, matched: 512))
    }
}

// MARK: - §5 Per-request cache salt (TB-007)

final class CBv2PrefixCacheRequestSaltTests: XCTestCase {

    private let kinds = [fullLayer()]
    private let tokens = Array(0 ..< 10)  // 2 whole blocks at blockSize 4

    private func donate(_ cache: PrefixCacheV2, salt: String?) {
        cache.donate(
            tokens: tokens, state: [MockSequenceKV(offset: 10)], layerKinds: kinds,
            cacheSalt: salt)
    }

    func testDifferentRequestSaltsNeverCrossHit() {
        // Direction 1: donated under "A" — invisible to "B" and to nil.
        let cache = makeCache()
        donate(cache, salt: "A")
        XCTAssertNil(cache.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: "B"))
        XCTAssertNil(cache.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: nil))
        XCTAssertEqual(
            cache.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: "A")?.matched, 8)
        cache.endAdoption(tokens: tokens, matched: 8, cacheSalt: "A")

        // Direction 2: donated under "B" — invisible to "A" and to nil.
        let reversed = makeCache()
        donate(reversed, salt: "B")
        XCTAssertNil(reversed.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: "A"))
        XCTAssertNil(reversed.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: nil))
        XCTAssertEqual(
            reversed.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: "B")?.matched, 8)
        reversed.endAdoption(tokens: tokens, matched: 8, cacheSalt: "B")

        // And a nil-salt donation is invisible to any request salt.
        let unsalted = makeCache()
        donate(unsalted, salt: nil)
        XCTAssertNil(unsalted.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: "A"))
        XCTAssertEqual(
            unsalted.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: nil)?.matched, 8)
        unsalted.endAdoption(tokens: tokens, matched: 8, cacheSalt: nil)
    }

    func testNilRequestSaltIsByteIdenticalToUnsaltedBehavior() {
        // nil request salt resolves through the cache-level hasher — the
        // salted overloads are then interchangeable with the legacy ones,
        // so existing hash vectors keep applying verbatim.
        let cache = makeCache()
        XCTAssertEqual(cache.hasher(cacheSalt: nil), cache.hasher)

        cache.donate(
            tokens: tokens, state: [MockSequenceKV(offset: 10)], layerKinds: kinds)
        XCTAssertEqual(
            cache.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: nil)?.matched, 8)
        cache.endAdoption(tokens: tokens, matched: 8, cacheSalt: nil)
        XCTAssertEqual(cache.lookup(tokens: tokens, layerKinds: kinds)?.matched, 8)
        cache.endAdoption(tokens: tokens, matched: 8)
        cache.evict(toFit: 0)
        XCTAssertEqual(cache.bytesInUse, 0, "both pins released — nothing leaks")
    }

    func testRequestSaltReplacesCacheLevelSaltWithNilFallback() {
        // Cache-level salt "tenant": nil falls back to it, an explicit
        // request salt equal to it resolves the SAME namespace, and any
        // other request salt is isolated.
        let cache = makeCache(salt: "tenant")
        XCTAssertEqual(cache.hasher(cacheSalt: "tenant"), cache.hasher)
        XCTAssertNotEqual(cache.hasher(cacheSalt: "other"), cache.hasher)

        donate(cache, salt: nil)
        XCTAssertEqual(
            cache.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: "tenant")?.matched, 8)
        cache.endAdoption(tokens: tokens, matched: 8, cacheSalt: "tenant")
        XCTAssertNil(cache.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: "other"))
    }

    func testSaltedPinBalanceRequiresMatchingSalt() {
        // endAdoption in the WRONG salt scope must not release the pin.
        let cache = makeCache()
        donate(cache, salt: "A")
        XCTAssertEqual(
            cache.lookup(tokens: tokens, layerKinds: kinds, cacheSalt: "A")?.matched, 8)

        cache.endAdoption(tokens: tokens, matched: 8, cacheSalt: "B")  // wrong scope
        cache.evict(toFit: 0)
        XCTAssertEqual(
            cache.bytesInUse, mockBytes(offset: 10),
            "a mismatched-salt release must not unpin the entry")

        cache.endAdoption(tokens: tokens, matched: 8, cacheSalt: "A")  // matching scope
        cache.evict(toFit: 0)
        XCTAssertEqual(cache.bytesInUse, 0)
    }
}
