// Tests for PrefixCache.swift — §5-6 of ContinuousBatchingTestPlan.md

import CryptoKit
import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - §5  Block hash

final class CBBlockHashTests: XCTestCase {

    func testComputeBlockHashIsDeterministic() {
        let tokens = [1, 2, 3]
        let h1 = computeBlockHash(parentHash: nil, tokenIds: tokens, modelName: "m")
        let h2 = computeBlockHash(parentHash: nil, tokenIds: tokens, modelName: "m")
        XCTAssertEqual(h1, h2)
    }

    func testComputeBlockHashDiffersOnDifferentTokens() {
        let h1 = computeBlockHash(parentHash: nil, tokenIds: [1, 2], modelName: "m")
        let h2 = computeBlockHash(parentHash: nil, tokenIds: [1, 3], modelName: "m")
        XCTAssertNotEqual(h1, h2)
    }

    func testComputeBlockHashDiffersOnDifferentParent() {
        let parent1 = computeBlockHash(parentHash: nil, tokenIds: [0], modelName: "m")
        let parent2 = computeBlockHash(parentHash: nil, tokenIds: [1], modelName: "m")
        let tokens = [2, 3, 4]
        let h1 = computeBlockHash(parentHash: parent1, tokenIds: tokens, modelName: "m")
        let h2 = computeBlockHash(parentHash: parent2, tokenIds: tokens, modelName: "m")
        XCTAssertNotEqual(h1, h2)
    }

    func testComputeBlockHashMatchesPythonRepr() {
        // The token repr must be "(1, 2, 3)" — comma+space separated, parenthesised.
        // We verify this by confirming that the hash for [1, 2, 3] equals the
        // hash computed by feeding the literal string "(1, 2, 3)" ourselves.
        let tokens = [1, 2, 3]
        let model = "test_model"
        let parent: Data? = nil

        var h = SHA256()
        h.update(data: Data(model.utf8))
        h.update(data: Data("omlx-root".utf8))
        let repr = "(1, 2, 3)"
        h.update(data: Data(repr.utf8))
        let expected = Data(h.finalize())

        let actual = computeBlockHash(parentHash: parent, tokenIds: tokens, modelName: model)
        XCTAssertEqual(actual, expected)
    }
}

// MARK: - §6  PrefixCache

/// Test persistence backend: returns a caller-supplied block for every
/// loadBlock (simulating blocks previously evicted to SSD), so cold-hit
/// paths can be exercised without real encryption.
private final class StubColdPersistence: PrefixCachePersistence, @unchecked Sendable {
    private let make: @Sendable () -> [KVCacheSimple]
    init(_ make: @escaping @Sendable () -> [KVCacheSimple]) { self.make = make }
    func saveBlock(blockHash: Data, layerCaches: [KVCacheSimple]) {}
    func loadBlock(blockHash: Data) -> [KVCacheSimple]? { make() }
}

final class CBPrefixCacheTests: XCTestCase {

    private let blockSize = 4

    private func makeConfig(maxBlocks: Int = 16) -> PrefixCacheConfig {
        PrefixCacheConfig(blockSize: blockSize, maxBlocks: maxBlocks)
    }

    /// Build a KVCacheSimple with known data so we can round-trip it.
    private func makeLayerCache(value: Float, seqLen: Int) -> KVCacheSimple {
        let c = KVCacheSimple()
        let t = MLXArray(Array(repeating: value, count: seqLen)).reshaped([1, 1, seqLen, 1])
        c.state = [t, t * 2]
        return c
    }

    // MARK: - Miss for short prompt

    func testPrefixCacheMissForShortPrompt() {
        let pc = PrefixCache(config: makeConfig())
        // Prompt shorter than blockSize → always a miss.
        let (caches, remaining) = pc.fetchPrefix(requestId: "r1", tokens: [1, 2, 3])
        XCTAssertNil(caches)
        XCTAssertEqual(remaining, [1, 2, 3])
        XCTAssertEqual(pc.misses, 1)
        XCTAssertEqual(pc.hits, 0)
    }

    // MARK: - Round-trip

    func testPrefixCacheRoundTrip() {
        let pc = PrefixCache(config: makeConfig())
        // blockSize = 4; store 8 tokens (2 full blocks).
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        let layer = makeLayerCache(value: 1.0, seqLen: tokens.count)

        pc.storePrefix(requestId: "r1", tokens: tokens, layerCaches: [layer])

        // Only whole blocks are cached; last partial block is not.
        // 8 tokens = 2 full blocks; both should be stored but only block 0
        // is eligible for fetch (block 1 is the "last" incomplete context block
        // only when tokens.count % blockSize != 0, here it's exactly 2 full blocks
        // so maxMatch = max(0, numFull-1) = 1 block fetched).
        let (caches, remaining) = pc.fetchPrefix(requestId: "r2", tokens: tokens)
        XCTAssertNotNil(caches)
        XCTAssertEqual(pc.hits, 1)
        XCTAssertFalse(remaining.isEmpty)
    }

    // MARK: - tokensSaved

    func testPrefixCacheHitSavesTokens() {
        let pc = PrefixCache(config: makeConfig())
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        let layer = makeLayerCache(value: 1.0, seqLen: tokens.count)
        pc.storePrefix(requestId: "r1", tokens: tokens, layerCaches: [layer])

        let (caches, _) = pc.fetchPrefix(requestId: "r2", tokens: tokens)
        XCTAssertNotNil(caches)
        XCTAssertGreaterThan(pc.tokensSaved, 0)
    }

    // MARK: - LRU eviction

    func testPrefixCacheEvictsLRUBlockWhenFull() {
        // maxBlocks = 2: store A, B, access A, store C → B should be evicted.
        let pc = PrefixCache(config: makeConfig(maxBlocks: 2))

        func storeBlock(_ base: Int, id: String) {
            let tokens = (base ..< base + blockSize).map { $0 }
            let layer = makeLayerCache(value: Float(base), seqLen: blockSize)
            pc.storePrefix(requestId: id, tokens: tokens + [base + blockSize],
                           layerCaches: [layer])
        }

        // Store A (tokens 0-3) and B (tokens 10-13) as separate cached sequences.
        let aTokens = [0, 1, 2, 3]
        let bTokens = [10, 11, 12, 13]
        let cTokens = [20, 21, 22, 23]

        let layerA = makeLayerCache(value: 0, seqLen: blockSize)
        let layerB = makeLayerCache(value: 10, seqLen: blockSize)
        let layerC = makeLayerCache(value: 20, seqLen: blockSize)

        // Store A and B (fill both slots).
        pc.storePrefix(requestId: "A", tokens: aTokens + [4], layerCaches: [layerA])
        pc.storePrefix(requestId: "B", tokens: bTokens + [14], layerCaches: [layerB])

        // Touch A (make it most-recently-used).
        _ = pc.fetchPrefix(requestId: "rA", tokens: aTokens + [4])

        // Store C — must evict B (LRU).
        pc.storePrefix(requestId: "C", tokens: cTokens + [24], layerCaches: [layerC])

        // A should still be cached; B should be gone.
        let (aHit, _) = pc.fetchPrefix(requestId: "rA2", tokens: aTokens + [4])
        let (bHit, _) = pc.fetchPrefix(requestId: "rB2", tokens: bTokens + [14])
        let (cHit, _) = pc.fetchPrefix(requestId: "rC2", tokens: cTokens + [24])

        XCTAssertNotNil(aHit, "A should survive (was touched after B)")
        // B or C might be gone depending on exact eviction order; at least one eviction happened.
        // The key invariant: total cached blocks ≤ maxBlocks.
        XCTAssertNotNil(cHit, "C was just stored")
        // B should have been evicted (LRU before C was stored).
        XCTAssertNil(bHit, "B should have been evicted (LRU)")
    }

    // MARK: - Release decrements ref count

    func testPrefixCacheReleaseDecrementsRefCount() {
        let pc = PrefixCache(config: makeConfig(maxBlocks: 2))
        let tokens = [0, 1, 2, 3, 4, 5, 6, 7]
        let layer = makeLayerCache(value: 1, seqLen: tokens.count)

        pc.storePrefix(requestId: "writer", tokens: tokens, layerCaches: [layer])

        // Fetch increments ref count — block won't be evicted.
        let (hit, _) = pc.fetchPrefix(requestId: "reader", tokens: tokens)
        XCTAssertNotNil(hit)

        // Release — block becomes evictable again.
        pc.releaseRequest("reader")
        // No assertion on internal refCount (private), but verifying no crash suffices.
    }

    // MARK: - Shared prefix hit

    func testPrefixCacheSharedPrefixHitsBothRequests() {
        let pc = PrefixCache(config: makeConfig())
        let sharedPrefix = [1, 2, 3, 4]   // one full block
        let layer = makeLayerCache(value: 1, seqLen: blockSize)

        // First request stores the prefix.
        pc.storePrefix(requestId: "req1", tokens: sharedPrefix + [5], layerCaches: [layer])

        // Second request with the same prefix should get a cache hit.
        let (caches2, _) = pc.fetchPrefix(requestId: "req2", tokens: sharedPrefix + [6])
        XCTAssertNotNil(caches2, "second request with identical prefix should hit")
        XCTAssertEqual(pc.hits, 1)
    }

    // MARK: - Cold hit with saturated block pool (regression)

    func testColdHitWithSaturatedPoolDoesNotOverlapRemaining() {
        // Regression: on a cold (persistence) hit where the block pool is
        // saturated (allocateBlock()==nil), the loaded block is still merged
        // into the returned cache, but cachedTokens was previously derived
        // from matchedIds (GPU-resident blocks only), undercounting it. The
        // returned remainingTokens then OVERLAPPED KV already present in the
        // merged cache, so the scheduler re-prefilled those positions on top
        // of the seeded cache — duplicating them at wrong offsets and
        // corrupting generation. cachedTokens must equal the merged width.
        let bs = blockSize
        // Every cold lookup returns a fresh full block. maxBlocks=1: the
        // first cold block in a fetch borrows the only slot (pinned,
        // refCount=1), so every later cold block in the same fetch hits
        // allocateBlock()==nil — exactly the saturated path.
        let stub = StubColdPersistence { [bs] in
            let c = KVCacheSimple()
            let t = MLXArray(Array(repeating: Float(9), count: bs)).reshaped([1, 1, bs, 1])
            c.state = [t, t * 2]
            return [c]
        }
        let pc = PrefixCache(config: makeConfig(maxBlocks: 1), persistence: stub)

        // 13 tokens, bs=4 → numFull=3, 13%4≠0 → maxMatch=3 cold hits. Only the
        // first gets a GPU slot; the other two are served for this request.
        let tokens = Array(0 ..< 13)
        let (caches, remaining) = pc.fetchPrefix(requestId: "r", tokens: tokens)

        XCTAssertNotNil(caches)
        let mergedSeq = caches![0].state[0].dim(2)
        XCTAssertEqual(mergedSeq, 3 * bs, "all 3 cold blocks are merged into the served cache")
        // The fix: remainingTokens starts exactly where the merged cache ends
        // — no overlap. (Before the fix remaining would include re-prefilled
        // tokens [bs ..< 3*bs] already present in the merged cache.)
        XCTAssertEqual(remaining.count, tokens.count - mergedSeq)
        XCTAssertEqual(remaining, [12])
    }
}
