// PrefixCacheV2.swift
//
// Content-addressed, zero-copy prefix cache for the ContinuousBatchingV2
// engine (workstream D). Implements `CBv2PrefixCache`.
//
// Design (docs/engine-v2/specs/D-prefix-cache.md; report 08 §2, report 09 §2,
// report 10 invariant 6):
//
//  - Keys are SHA-256 chain hashes at fixed block granularity
//    (`CBv2BlockHasher`, byte-compatible with the legacy checkpoint tier).
//    Fixed blocks, deliberately NOT a radix tree — at provider-scale queue
//    depths hash-chained blocks capture ~all the win (plan §7 Phase 4).
//  - `donate` is a zero-copy move: the finished request's per-layer
//    `snapshot()` arrays are stored as-is (no eval, no reshuffle — safe to
//    call off the engine thread; nothing here ever forces a host sync).
//    Only the longest whole-block prefix is indexed; the partial tail rides
//    along inside the stored arrays and is sliced off lazily at lookup.
//  - `lookup` returns the LONGEST chain-hash match in whole blocks, capped
//    at `tokens.count - 1` so the final position is always recomputed for
//    logits (vLLM last-token rule). A hit bumps LRU recency and pins the
//    entry in-use until `endAdoption` releases it.
//  - Windowed-layer policy (report 10 invariant 6): sliding-window layers
//    NEVER enter full-history prefix reuse. Donation stores snapshots for
//    full-attention layers that own storage; windowed and KV-sharing layers
//    are nil. The engine re-prefills the trailing `requiredRecompute(...)`
//    tokens through ALL layers after adopting a prefix. That span is
//    (windowed-layer count × largest window), clamped to the match, because
//    each stacked windowed layer compounds the receptive field and any
//    downstream full layer caches its polluted early outputs — see the
//    derivation on `cbv2RequiredRecompute`.
//  - Eviction is LRU by last access; entries currently being adopted
//    (in-use refcount > 0) are never evicted, and key repointing skips
//    pinned entries so an in-flight adoption always stays resolvable.
//
// Thread safety: a single NSLock guards all mutable state. Every public
// method is safe to call from any thread; none performs MLX evaluation.

import Foundation
import MLX

// MARK: - Config

public struct CBv2PrefixCacheConfig: Sendable, Equatable {
    /// Tokens per hash block. Must stay consistent for the cache lifetime.
    public var blockSize: Int
    /// Model namespace folded into every block hash (cross-model isolation).
    public var modelName: String
    /// Tenant-scope namespace folded into the first block hash (TB-007).
    /// Empty ⇒ unscoped, byte-identical to the legacy chain scheme.
    public var cacheSalt: String
    /// Byte budget enforced automatically after each donation.
    /// nil ⇒ unbounded; the owner calls `evict(toFit:)` explicitly.
    public var maxBytes: Int?
    /// Materialize (device-eval) donated snapshot arrays before indexing
    /// them. REQUIRED for the paged backend: donated views reference the
    /// live slabs, whose pages are recycled once the donor state is
    /// released. Not a host readback — device materialization only, and
    /// donation already runs off the engine step thread. ON by default
    /// (safe for every backend; contiguous deployments whose snapshot views
    /// own their buffers via ARC may opt out to skip the ~free eval).
    public var materializeOnDonate: Bool

    public init(
        blockSize: Int = CBv2BlockHasher.defaultBlockSize,
        modelName: String = "",
        cacheSalt: String = "",
        maxBytes: Int? = nil,
        materializeOnDonate: Bool = true
    ) {
        self.blockSize = blockSize
        self.modelName = modelName
        self.cacheSalt = cacheSalt
        self.maxBytes = maxBytes
        self.materializeOnDonate = materializeOnDonate
    }
}

// MARK: - Stats

public struct CBv2PrefixCacheStats: Sendable, Equatable {
    public var hits: Int
    public var misses: Int
    public var tokensSaved: Int
    public var entryCount: Int
    public var bytesInUse: Int
}

// MARK: - PrefixCacheV2

public final class PrefixCacheV2: CBv2PrefixCache, @unchecked Sendable {

    /// One donated whole-block prefix. `snapshots` holds the donated arrays
    /// as-is (zero-copy; they may extend past `tokenCount` — the tail is
    /// sliced off lazily at lookup). Windowed / KV-sharing layers are nil.
    private final class Entry {
        let id: UInt64
        let blockCount: Int
        let tokenCount: Int
        /// h_1 … h_blockCount. Every whole-block boundary of this entry is
        /// indexed, so shorter prefixes of a long donation still hit.
        let chainHashes: [Data]
        let snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?]
        /// Bytes physically retained (the FULL donated arrays, including any
        /// partial tail — truthful accounting of what this entry pins).
        let bytes: Int
        var lastAccess: UInt64
        /// Adoptions in flight (lookup → endAdoption). Pinned entries are
        /// never evicted and never lose index keys to repointing.
        var refInUse: Int = 0
        /// Index keys currently resolving to this entry (≤ blockCount; keys
        /// can be lost to a longer donation while the entry is unpinned).
        var liveKeys: Int = 0

        init(
            id: UInt64, blockCount: Int, tokenCount: Int, chainHashes: [Data],
            snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?],
            bytes: Int, lastAccess: UInt64
        ) {
            self.id = id
            self.blockCount = blockCount
            self.tokenCount = tokenCount
            self.chainHashes = chainHashes
            self.snapshots = snapshots
            self.bytes = bytes
            self.lastAccess = lastAccess
        }
    }

    public let config: CBv2PrefixCacheConfig
    public let hasher: CBv2BlockHasher

    private let lock = NSLock()
    private var entries: [UInt64: Entry] = [:]
    /// chainHash → entry id. Multiple keys may resolve to one entry.
    private var index: [Data: UInt64] = [:]
    private var nextEntryID: UInt64 = 0
    /// Monotonic access clock (deterministic LRU; no wall-clock ties).
    private var tick: UInt64 = 0
    private var _bytesInUse = 0

    private var _hits = 0
    private var _misses = 0
    private var _tokensSaved = 0

    public init(config: CBv2PrefixCacheConfig = .init()) {
        self.config = config
        self.hasher = CBv2BlockHasher(
            blockSize: config.blockSize,
            modelName: config.modelName,
            cacheSalt: config.cacheSalt
        )
    }

    /// Hasher for a request-scoped salt (TB-007). A non-nil request salt
    /// REPLACES the cache-level salt in the first block hash — the chain
    /// then propagates the scope to every later block — so entries donated
    /// under different salts can never resolve each other. nil falls back
    /// to the configured cache-level hasher (byte-identical hash vectors to
    /// the pre-salt behavior).
    func hasher(cacheSalt: String?) -> CBv2BlockHasher {
        guard let cacheSalt else { return hasher }
        return CBv2BlockHasher(
            blockSize: config.blockSize, modelName: config.modelName, cacheSalt: cacheSalt)
    }

    // MARK: - CBv2PrefixCache: lookup

    public func lookup(
        tokens: [Int], layerKinds: [CBv2LayerKind]
    ) -> (matched: Int, prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?])? {
        lookup(tokens: tokens, layerKinds: layerKinds, cacheSalt: nil)
    }

    public func lookup(
        tokens: [Int], layerKinds: [CBv2LayerKind], cacheSalt: String?
    ) -> (matched: Int, prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?])? {
        let hasher = hasher(cacheSalt: cacheSalt)
        let maxBlocks = hasher.maxLookupBlocks(tokenCount: tokens.count)
        guard maxBlocks > 0 else { return nil }
        let hashes = hasher.chainHashes(tokens: tokens, maxBlocks: maxBlocks)

        lock.lock()
        defer { lock.unlock() }

        // Longest match wins. Scan longest → shortest: eviction can remove a
        // SHORT diverging entry that once held a shared early key, so the
        // indexed boundary set is not guaranteed downward-closed and a
        // left-to-right stop-at-first-miss scan could miss a longer hit.
        for k in stride(from: hashes.count, through: 1, by: -1) {
            guard let id = index[hashes[k - 1]], let entry = entries[id] else { continue }
            // Defensive: a stored entry must describe the same layer layout
            // the caller is adopting into.
            guard entry.snapshots.count == layerKinds.count else { continue }

            let matched = k * hasher.blockSize
            tick += 1
            entry.lastAccess = tick
            entry.refInUse += 1
            _hits += 1
            _tokensSaved += matched

            let prefix = slicedPrefix(entry: entry, matched: matched, layerKinds: layerKinds)
            return (matched, prefix)
        }

        _misses += 1
        return nil
    }

    /// Lazily slice the stored per-layer arrays down to `matched` tokens.
    /// Pure graph construction — no eval, no host sync.
    private func slicedPrefix(
        entry: Entry, matched: Int, layerKinds: [CBv2LayerKind]
    ) -> [(keys: MLXArray, values: MLXArray, offset: Int)?] {
        (0 ..< layerKinds.count).map { i in
            guard let snap = entry.snapshots[i] else { return nil }
            // Honor the CALLER's layer kinds too: never hand a snapshot to a
            // windowed or KV-sharing layer (report 10 invariant 6).
            guard Self.isCacheable(layerKinds[i]) else { return nil }
            if snap.offset == matched {
                return (keys: snap.keys, values: snap.values, offset: matched)
            }
            return (
                keys: snap.keys[.ellipsis, 0 ..< matched, 0...],
                values: snap.values[.ellipsis, 0 ..< matched, 0...],
                offset: matched
            )
        }
    }

    // MARK: - CBv2PrefixCache: donate

    public func donate(tokens: [Int], state: [CBv2SequenceKV?], layerKinds: [CBv2LayerKind]) {
        donate(tokens: tokens, state: state, layerKinds: layerKinds, cacheSalt: nil)
    }

    /// Salted state-based donation (concrete-type convenience; the engine
    /// path uses the pre-snapshotted protocol overload).
    public func donate(
        tokens: [Int], state: [CBv2SequenceKV?], layerKinds: [CBv2LayerKind],
        cacheSalt: String?
    ) {
        guard state.count == layerKinds.count else { return }
        // Snapshot the full-attention storage-owning layers. Windowed layers
        // must not enter full-history prefix reuse (report 10 invariant 6);
        // KV-sharing layers own no storage. Both stay nil.
        var snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?] = []
        snapshots.reserveCapacity(layerKinds.count)
        for (i, kind) in layerKinds.enumerated() {
            guard Self.isCacheable(kind), let seq = state[i] else {
                snapshots.append(nil)
                continue
            }
            // Lossy snapshots (quantized rows) never enter the cache — a
            // donate→adopt round trip would re-quantize onto a different
            // grid (see CBv2SequenceKV.snapshotIsLossless). The donation as
            // a whole is dropped: an entry missing a cacheable layer is
            // unusable anyway.
            guard seq.snapshotIsLossless else { return }
            snapshots.append(seq.snapshot())
        }
        donate(tokens: tokens, snapshots: snapshots, layerKinds: layerKinds, cacheSalt: cacheSalt)
    }

    /// Pre-snapshotted donation (the engine-integration path): the per-layer
    /// snapshots are built ON the engine thread (graph-only, so views over
    /// shared slabs are consistent with in-flight writes) and handed here
    /// from the donation queue. Windowed / KV-sharing layers must be nil.
    public func donate(
        tokens: [Int],
        snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind]
    ) {
        donate(tokens: tokens, snapshots: snapshots, layerKinds: layerKinds, cacheSalt: nil)
    }

    public func donate(
        tokens: [Int],
        snapshots rawSnapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind], cacheSalt: String?
    ) {
        guard rawSnapshots.count == layerKinds.count else { return }
        let hasher = hasher(cacheSalt: cacheSalt)
        let blockCount = hasher.fullBlockCount(tokenCount: tokens.count)
        guard blockCount > 0 else { return }  // nothing block-aligned to keep
        let prefixTokens = blockCount * hasher.blockSize

        var snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?] = []
        snapshots.reserveCapacity(layerKinds.count)
        var cacheableLayers = 0
        for (i, kind) in layerKinds.enumerated() {
            guard Self.isCacheable(kind) else {
                snapshots.append(nil)
                continue
            }
            // A cacheable layer with missing or too-short state cannot cover
            // the prefix — the donation as a whole is unusable.
            guard let snap = rawSnapshots[i],
                snap.offset >= prefixTokens,
                snap.keys.ndim == 4, snap.keys.dim(2) >= prefixTokens
            else { return }
            snapshots.append(snap)
            cacheableLayers += 1
        }
        // All-windowed/all-shared model: an entry would carry no KV at all.
        guard cacheableLayers > 0 else { return }

        if config.materializeOnDonate {
            var toEval: [MLXArray] = []
            for snap in snapshots {
                guard let snap else { continue }
                toEval.append(snap.keys)
                toEval.append(snap.values)
            }
            eval(toEval)
        }

        let hashes = hasher.chainHashes(tokens: tokens, maxBlocks: blockCount)
        let bytes = snapshots.reduce(0) { sum, snap in
            guard let snap else { return sum }
            return sum + snap.keys.nbytes + snap.values.nbytes
        }

        lock.lock()
        defer { lock.unlock() }

        // Dedup: this exact whole-block prefix is already represented (the
        // entry holding its terminal hash covers at least these blocks).
        // Re-register any missing intermediate block keys while we're here:
        // an intermediate key can be lost when a shorter entry that owned
        // it (a pinned entry survives repointing) is later evicted — after
        // which shorter prefixes of this donation stopped hitting.
        if let existingID = index[hashes[blockCount - 1]], let existing = entries[existingID] {
            tick += 1
            existing.lastAccess = tick
            for hash in hashes {
                guard index[hash].map({ entries[$0] == nil }) ?? true else { continue }
                index[hash] = existing.id
                existing.liveKeys += 1
            }
            return
        }

        tick += 1
        nextEntryID += 1
        let entry = Entry(
            id: nextEntryID, blockCount: blockCount, tokenCount: prefixTokens,
            chainHashes: hashes, snapshots: snapshots, bytes: bytes, lastAccess: tick
        )

        // Register every whole-block boundary. On a key contest, prefer the
        // LONGER entry (dropping a fully-shadowed shorter one frees its
        // duplicate storage) — but never steal keys from a pinned entry, so
        // in-flight adoptions stay resolvable by hash.
        for hash in hashes {
            if let oldID = index[hash], oldID != entry.id {
                guard let old = entries[oldID] else {
                    index[hash] = entry.id
                    entry.liveKeys += 1
                    continue
                }
                if old.blockCount >= blockCount || old.refInUse > 0 { continue }
                index[hash] = entry.id
                entry.liveKeys += 1
                old.liveKeys -= 1
                if old.liveKeys <= 0 { removeLocked(old) }
            } else {
                index[hash] = entry.id
                entry.liveKeys += 1
            }
        }

        entries[entry.id] = entry
        _bytesInUse += entry.bytes

        if let budget = config.maxBytes {
            evictLocked(toFit: budget)
        }
    }

    // MARK: - Adoption lifecycle

    /// Release the in-use pin taken by a successful `lookup`. `matched` is
    /// the matched token count that lookup returned for these `tokens`.
    ///
    /// NOTE: the frozen `CBv2PrefixCache` protocol has no adoption-release
    /// hook, so this lives on the concrete type (see
    /// docs/engine-v2/CONTRACT-ISSUES-D-prefix-cache.md). Pinned entries
    /// always keep their index keys, so resolving by hash here is exact.
    public func endAdoption(tokens: [Int], matched: Int) {
        endAdoption(tokens: tokens, matched: matched, cacheSalt: nil)
    }

    public func endAdoption(tokens: [Int], matched: Int, cacheSalt: String?) {
        let hasher = hasher(cacheSalt: cacheSalt)
        guard matched > 0, matched % hasher.blockSize == 0 else { return }
        let blocks = matched / hasher.blockSize
        guard blocks * hasher.blockSize <= tokens.count else { return }
        let hashes = hasher.chainHashes(tokens: tokens, maxBlocks: blocks)
        guard hashes.count == blocks else { return }

        lock.lock()
        defer { lock.unlock() }
        guard let id = index[hashes[blocks - 1]], let entry = entries[id] else { return }
        entry.refInUse = max(0, entry.refInUse - 1)
        // A pinned entry that lost relevance keeps its keys until released,
        // so no keyless-entry sweep is needed here; eviction handles LRU.
        if let budget = config.maxBytes, _bytesInUse > budget {
            evictLocked(toFit: budget)
        }
    }

    // MARK: - CBv2PrefixCache: evict

    public func evict(toFit byteBudget: Int) {
        lock.lock()
        defer { lock.unlock() }
        evictLocked(toFit: byteBudget)
    }

    private func evictLocked(toFit byteBudget: Int) {
        while _bytesInUse > byteBudget {
            let victim = entries.values
                .filter { $0.refInUse == 0 }
                .min { $0.lastAccess < $1.lastAccess }
            guard let victim else { return }  // everything left is in use
            removeLocked(victim)
        }
    }

    private func removeLocked(_ entry: Entry) {
        // Before dropping a key, hand it to a resident HEIR that shares this
        // exact chain-hash boundary (Codex P2): a longer donation that
        // arrived while `entry` was PINNED could not repoint `entry`'s
        // boundary hashes (a pinned entry's in-flight adoption must keep
        // resolving by hash), so it registered only its own extra blocks.
        // Without this, evicting the (now unpinned) shorter entry would
        // delete a boundary that the longer entry still physically covers,
        // and lookups for the shorter prefix would miss forever even though
        // the KV is resident. Repointing to the longest such heir preserves
        // the shorter prefix as a slice of the longer entry.
        for (blockIndex, hash) in entry.chainHashes.enumerated() where index[hash] == entry.id {
            if let heir = longestHeirLocked(forHash: hash, blockIndex: blockIndex, excluding: entry.id)
            {
                index[hash] = heir.id
                heir.liveKeys += 1
            } else {
                index.removeValue(forKey: hash)
            }
        }
        entry.liveKeys = 0
        if entries.removeValue(forKey: entry.id) != nil {
            _bytesInUse -= entry.bytes
        }
    }

    /// The longest resident entry (other than `excluding`) whose chain hash
    /// at `blockIndex` equals `hash` — i.e. one that shares this whole-block
    /// boundary and can serve the shorter prefix as a slice. Chain hashing
    /// makes a shared boundary hash imply an identical token prefix through
    /// that block, so any such entry is a valid owner of the key.
    private func longestHeirLocked(
        forHash hash: Data, blockIndex: Int, excluding excludedID: UInt64
    ) -> Entry? {
        var best: Entry?
        for candidate in entries.values where candidate.id != excludedID {
            guard candidate.chainHashes.count > blockIndex,
                candidate.chainHashes[blockIndex] == hash
            else { continue }
            if best == nil || candidate.blockCount > best!.blockCount { best = candidate }
        }
        return best
    }

    // MARK: - Accounting

    public var bytesInUse: Int {
        lock.lock()
        defer { lock.unlock() }
        return _bytesInUse
    }

    public func stats() -> CBv2PrefixCacheStats {
        lock.lock()
        defer { lock.unlock() }
        return CBv2PrefixCacheStats(
            hits: _hits, misses: _misses, tokensSaved: _tokensSaved,
            entryCount: entries.count, bytesInUse: _bytesInUse
        )
    }

    // MARK: - Windowed-layer policy

    /// Tokens the engine must re-prefill through ALL layers after adopting
    /// a `matched`-token prefix so windowed layers are exact for the
    /// retained tail. Delegates to the contract-level free function
    /// `cbv2RequiredRecompute` (pure model-shape logic, shared by all
    /// backends); see its derivation for why the span scales with the
    /// number of stacked windowed layers.
    public static func requiredRecompute(layerKinds: [CBv2LayerKind], matched: Int) -> Int {
        cbv2RequiredRecompute(layerKinds: layerKinds, matched: matched)
    }

    /// Instance convenience for callers holding the cache.
    public func requiredRecompute(layerKinds: [CBv2LayerKind], matched: Int) -> Int {
        cbv2RequiredRecompute(layerKinds: layerKinds, matched: matched)
    }

    /// A layer's KV enters the prefix cache only when it is full-attention
    /// AND owns its storage (KV-sharing layers borrow the source layer's).
    static func isCacheable(_ kind: CBv2LayerKind) -> Bool {
        guard kind.sharesKVWithLayer == nil else { return false }
        if case .slidingWindow = kind.attention { return false }
        return true
    }
}
