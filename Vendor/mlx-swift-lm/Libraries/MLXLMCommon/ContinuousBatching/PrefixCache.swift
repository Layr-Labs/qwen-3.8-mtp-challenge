// Copyright © 2026 Eigen Labs.
//
// Block-level prefix caching for continuous batching.
// Port of omlx/omlx/cache/paged_cache.py + prefix_cache.py
//
// Token sequences are divided into fixed-size blocks (default 256 tokens).
// Each block's KV state is content-addressed by a SHA-256 chain hash over
// its token IDs. Requests sharing a common prompt prefix skip re-prefilling
// the matching blocks.

import CryptoKit
import Foundation
import MLX

// MARK: - Chain hash

/// SHA-256 chain hash for a block: H(modelName || parentHash || tokenIds).
/// Matches omlx's `compute_block_hash` so SSD files are cross-compatible.
func computeBlockHash(parentHash: Data?, tokenIds: [Int], modelName: String = "") -> Data {
    var h = SHA256()
    if !modelName.isEmpty { h.update(data: Data(modelName.utf8)) }
    h.update(data: parentHash ?? Data("omlx-root".utf8))
    // Mirror Python: str(tuple(token_ids)) → "(1, 2, 3)"
    let repr = "(\(tokenIds.map(String.init).joined(separator: ", ")))"
    h.update(data: Data(repr.utf8))
    return Data(h.finalize())
}

// MARK: - CacheBlock

final class CacheBlock {
    let blockId: Int
    var refCount: Int = 0
    var blockHash: Data?
    var tokenCount: Int = 0
    var lastAccess: Date = .init()
    /// Per-layer KV state for this block. `nil` when the block is free or
    /// its data has been evicted to SSD.
    var cacheData: [KVCacheSimple]?

    init(blockId: Int) { self.blockId = blockId }
    var isEvictable: Bool { refCount == 0 }
    func touch() { lastAccess = .init() }
}

// MARK: - PrefixCachePersistence

/// Optional encrypted/cold-storage backend for the in-GPU prefix cache.
///
/// The `PrefixCache` keeps hot blocks in GPU/unified memory and evicts
/// the LRU block when it needs room. With a persistence backend wired
/// in, an evicted block is handed to `saveBlock` (so it survives the
/// eviction, and process restart) and a fetch miss for a known block
/// hash is retried against `loadBlock` (so an evicted/persisted block
/// can be reloaded instead of re-prefilled).
///
/// Both calls are SYNCHRONOUS — they run inside the engine's step loop.
/// Implementations must therefore do any key unwrap up front and keep
/// `saveBlock`/`loadBlock` non-async (e.g. hold an already-unwrapped
/// symmetric key and do synchronous crypto + file I/O). `saveBlock` is
/// best-effort: it must not throw and should fail silently (a lost
/// block just means a future cold prefill).
///
/// Block keying is the content-addressed `computeBlockHash` chain, so a
/// backend's on-disk layout is stable across runs and models.
public protocol PrefixCachePersistence: AnyObject, Sendable {
    /// Persist a block's per-layer KV under its content hash. Best-effort.
    func saveBlock(blockHash: Data, layerCaches: [KVCacheSimple])
    /// Load a previously-persisted block's per-layer KV, or nil if absent
    /// (or if the backend rejects it, e.g. a model-binding mismatch).
    func loadBlock(blockHash: Data) -> [KVCacheSimple]?
}

// MARK: - PrefixCacheConfig

public struct PrefixCacheConfig: Sendable, Equatable {
    /// Tokens per block. Must be consistent across all stored blocks.
    public var blockSize: Int
    /// Maximum number of in-GPU-memory blocks.
    public var maxBlocks: Int

    public init(blockSize: Int = 256, maxBlocks: Int = 4096) {
        self.blockSize = blockSize
        self.maxBlocks = maxBlocks
    }
}

// MARK: - PrefixCache

/// In-memory block-level prefix cache.
///
/// Divides token sequences into fixed-size blocks and reuses KV state across
/// requests that share a common prompt prefix.
///
/// ### Usage
/// ```swift
/// let prefixCache = PrefixCache(config: .init(), modelName: modelName)
///
/// // Before prefilling a new request:
/// let (cached, remaining) = prefixCache.fetchPrefix(requestId: rid, tokens: promptTokens)
/// // pass `cached` as existingCache to doExternalPrefill, `remaining` as tokens
///
/// // After a request finishes:
/// if let simpleCache = resp.promptCache?.compactMap({ $0 as? KVCacheSimple }),
///    simpleCache.count == resp.promptCache?.count {
///     prefixCache.storePrefix(requestId: rid, tokens: allTokens, layerCaches: simpleCache)
/// }
/// prefixCache.releaseRequest(rid)
/// ```
public final class PrefixCache: @unchecked Sendable {
    public let config: PrefixCacheConfig
    private let modelName: String
    /// Optional encrypted/cold-storage backend (see `PrefixCachePersistence`).
    /// nil = in-GPU only (default; evicted blocks are dropped).
    private let persistence: PrefixCachePersistence?

    // Blocks that are completely free (no data).
    private var freeBlocks: [CacheBlock]
    // Blocks that hold data (refCount may be 0 = evictable, or >0 = in use).
    private var hashIndex: [Data: Int] = [:]
    private var allocatedBlocks: [Int: CacheBlock] = [:]
    // Per-request borrowed block IDs (for ref-count bookkeeping).
    private var requestBlocks: [String: [Int]] = [:]

    public private(set) var hits = 0
    public private(set) var misses = 0
    public private(set) var tokensSaved = 0

    public init(
        config: PrefixCacheConfig = .init(),
        modelName: String = "",
        persistence: PrefixCachePersistence? = nil
    ) {
        self.config = config
        self.modelName = modelName
        self.persistence = persistence
        self.freeBlocks = (0 ..< config.maxBlocks).map { CacheBlock(blockId: $0) }
    }

    // MARK: - Fetch

    /// Find cached prefix blocks for `tokens`.
    ///
    /// Returns per-layer merged `KVCacheSimple` caches covering the matched
    /// prefix, and the remaining tokens that still need to be prefilled.
    /// At least 1 token is always left in `remainingTokens` so a valid seed
    /// token is available for `GenerationBatch`.
    public func fetchPrefix(
        requestId: String,
        tokens: [Int]
    ) -> (caches: [KVCacheSimple]?, remainingTokens: [Int]) {
        let bs = config.blockSize
        let numFull = tokens.count / bs
        // Reserve the last block so we always have at least bs remaining tokens
        // when the prompt is an exact multiple of blockSize.
        let maxMatch = tokens.count % bs == 0 ? max(0, numFull - 1) : numFull
        guard maxMatch > 0 else { misses += 1; return (nil, tokens) }

        var matchedPerBlock: [[KVCacheSimple]] = []
        var matchedIds: [Int] = []
        var parentHash: Data?

        for i in 0 ..< maxMatch {
            let start = i * bs
            let chunk = Array(tokens[start ..< start + bs])
            let hash = computeBlockHash(parentHash: parentHash, tokenIds: chunk, modelName: modelName)

            if let blockId = hashIndex[hash], let block = allocatedBlocks[blockId],
               let layerCaches = block.cacheData
            {
                block.refCount += 1
                block.touch()
                matchedPerBlock.append(layerCaches)
                matchedIds.append(blockId)
                parentHash = hash
            } else if let p = persistence, let layerCaches = p.loadBlock(blockHash: hash) {
                // Cold hit: the block was evicted (or persisted in a prior
                // run). Reload it into a GPU block so subsequent fetches hit
                // in-memory; if there's no room, use it for this request only.
                if let block = allocateBlock() {
                    block.blockHash = hash
                    block.tokenCount = bs
                    block.cacheData = layerCaches
                    block.refCount = 1
                    block.touch()
                    hashIndex[hash] = block.blockId
                    allocatedBlocks[block.blockId] = block
                    matchedIds.append(block.blockId)
                }
                matchedPerBlock.append(layerCaches)
                parentHash = hash
            } else {
                break
            }
        }

        guard !matchedPerBlock.isEmpty else { misses += 1; return (nil, tokens) }

        requestBlocks[requestId] = matchedIds

        // Merge blocks per layer by concatenating along the time axis (axis 2).
        let numLayers = matchedPerBlock[0].count
        let merged: [KVCacheSimple] = (0 ..< numLayers).map { layer in
            guard matchedPerBlock.count > 1 else { return matchedPerBlock[0][layer] }
            let ks = matchedPerBlock.compactMap { $0[layer].state.first }
            let vs = matchedPerBlock.compactMap { $0[layer].state.dropFirst().first }
            guard ks.count == matchedPerBlock.count else { return matchedPerBlock[0][layer] }
            let merged = KVCacheSimple()
            merged.state = [concatenated(ks, axis: 2), concatenated(vs, axis: 2)]
            return merged
        }

        // Count tokens from the blocks actually merged into `merged`, NOT
        // from matchedIds. On a cold (persistence) hit where the block pool
        // is saturated and allocateBlock() returns nil, the loaded block is
        // appended to matchedPerBlock (and served) but NOT to matchedIds
        // (which tracks only GPU-resident blocks for ref-count bookkeeping).
        // Using matchedIds.count here would undercount cachedTokens, so the
        // returned remainingTokens would overlap KV already present in
        // `merged` — the scheduler would then re-prefill those positions on
        // top of the seeded cache, duplicating them at wrong offsets and
        // corrupting generation. matchedPerBlock.count*bs is exactly the
        // width of `merged`.
        let cachedTokens = matchedPerBlock.count * bs
        hits += 1
        tokensSaved += cachedTokens
        return (merged, Array(tokens[cachedTokens...]))
    }

    // MARK: - Store

    /// Store completed KV state in cache blocks.
    ///
    /// Slices `layerCaches` into `blockSize`-token blocks and indexes any new
    /// complete blocks. Only stores `KVCacheSimple` layers; hybrid models with
    /// SSM layers are not supported yet.
    ///
    /// - Parameters:
    ///   - requestId: Request identifier (used for logging only).
    ///   - tokens: Full token sequence (prompt + output) for this request.
    ///   - layerCaches: Per-layer `KVCacheSimple` holding the full sequence KV state.
    public func storePrefix(requestId: String, tokens: [Int], layerCaches: [KVCacheSimple]) {
        let bs = config.blockSize
        let numFull = tokens.count / bs
        guard numFull > 0 else { return }
        guard layerCaches.first?.state.count ?? 0 >= 2 else { return }

        var parentHash: Data?
        for i in 0 ..< numFull {
            let start = i * bs
            let chunk = Array(tokens[start ..< start + bs])
            let hash = computeBlockHash(parentHash: parentHash, tokenIds: chunk, modelName: modelName)
            parentHash = hash

            if hashIndex[hash] != nil { continue }   // already cached
            guard let block = allocateBlock() else { return }

            let sliced: [KVCacheSimple] = layerCaches.map { layer in
                let c = KVCacheSimple()
                if layer.state.count >= 2 {
                    let k = layer.state[0][.ellipsis, start ..< (start + bs), 0...]
                    let v = layer.state[1][.ellipsis, start ..< (start + bs), 0...]
                    eval(k, v)
                    c.state = [k, v]
                }
                return c
            }

            block.blockHash = hash
            block.tokenCount = bs
            block.cacheData = sliced
            block.refCount = 0          // immediately evictable but findable
            block.touch()
            hashIndex[hash] = block.blockId
            allocatedBlocks[block.blockId] = block

        }
    }

    // MARK: - Release

    /// Decrement ref counts for all blocks borrowed by `requestId`.
    public func releaseRequest(_ requestId: String) {
        guard let ids = requestBlocks.removeValue(forKey: requestId) else { return }
        for id in ids {
            guard let block = allocatedBlocks[id] else { continue }
            block.refCount = max(0, block.refCount - 1)
            // Block stays in allocatedBlocks; evictable when refCount == 0.
        }
    }

    // MARK: - Stats

    public func getStats() -> [String: Any] {
        [
            "hits": hits,
            "misses": misses,
            "tokens_saved": tokensSaved,
            "cached_blocks": hashIndex.count,
            "free_blocks": freeBlocks.count,
        ]
    }

    // MARK: - Private

    private func allocateBlock() -> CacheBlock? {
        if !freeBlocks.isEmpty {
            return freeBlocks.removeLast()
        }
        // Evict the oldest evictable (refCount == 0) block to make room.
        guard let victim = allocatedBlocks.values
            .filter({ $0.isEvictable })
            .min(by: { $0.lastAccess < $1.lastAccess })
        else { return nil }

        // Persist to cold storage before dropping the in-GPU data, so an
        // evicted block can be reloaded (and survives restart) instead of
        // being re-prefilled. Best-effort; nil backend = drop as before.
        if let p = persistence, let hash = victim.blockHash, let data = victim.cacheData {
            p.saveBlock(blockHash: hash, layerCaches: data)
        }

        if let hash = victim.blockHash { hashIndex.removeValue(forKey: hash) }
        allocatedBlocks.removeValue(forKey: victim.blockId)
        victim.blockHash = nil
        victim.cacheData = nil
        victim.tokenCount = 0
        return victim
    }
}
