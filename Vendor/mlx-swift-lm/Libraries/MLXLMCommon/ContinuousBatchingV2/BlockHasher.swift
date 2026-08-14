// BlockHasher.swift
//
// SHA-256 chain hashing at fixed token-block granularity for the v2 prefix
// cache (workstream D).
//
// The chain scheme is byte-compatible with the legacy checkpoint tier
// (`ContinuousBatching/PrefixCache.swift` → `computeBlockHash`, itself a port
// of omlx's `compute_block_hash`), so SSD-persisted blocks written by the
// legacy tier can interop with v2 later:
//
//     h_i = SHA256( utf8(modelName)            // skipped when empty
//                 ‖ (h_{i-1} ?? "omlx-root")   // chain parent, or root marker
//                 ‖ salt-section               // FIRST block only, see below
//                 ‖ utf8("(t0, t1, …)") )      // Python tuple repr of the block
//
// Because each hash folds in its parent, h_i fingerprints the ENTIRE prefix
// up to and including block i — matches are downward-closed, so a
// longest-match scan can stop at the first miss (vLLM V1 rule; report 08 §2).
//
// Tenant scoping (TB-007): a non-empty `cacheSalt` is folded into the FIRST
// block hash only (vLLM folds `cache_salt` into block 0's extra keys — the
// chain propagates it to every later block). The salt section is
// domain-tagged and length-prefixed so it can never collide with token-repr
// bytes or with a different-length salt:
//
//     salt-section = "cbv2-salt-v1" ‖ u64le(len(saltUTF8)) ‖ saltUTF8
//
// BACK-COMPAT INVARIANT: an EMPTY salt mixes in NOTHING — the digest is then
// byte-identical to the legacy `computeBlockHash` chain (verified in
// CBv2PrefixCacheTests against the legacy function and fixed vectors).

import CryptoKit
import Foundation

/// Stateless block chain hasher. Configure once per (model, tenant scope)
/// and reuse; all methods are pure.
public struct CBv2BlockHasher: Sendable, Equatable {

    /// Tokens per block. 256 matches the legacy checkpoint tier and the
    /// coarsest common divisor of the fleet's checkpoint ladders.
    public static let defaultBlockSize = 256

    /// Root marker hashed in place of a parent for the first block.
    /// Byte-compatible with omlx / the legacy tier.
    static let rootMarker = Data("omlx-root".utf8)

    /// Domain-separation tag for the salt section (first block only).
    static let saltTag = Data("cbv2-salt-v1".utf8)

    public let blockSize: Int
    /// Model namespace folded into every block hash (legacy-compatible:
    /// empty ⇒ skipped entirely).
    public let modelName: String
    /// Tenant-scope namespace folded into the FIRST block hash (TB-007).
    /// Empty ⇒ skipped entirely (legacy byte-compatible).
    public let cacheSalt: String

    public init(
        blockSize: Int = CBv2BlockHasher.defaultBlockSize,
        modelName: String = "",
        cacheSalt: String = ""
    ) {
        precondition(blockSize > 0, "blockSize must be positive")
        self.blockSize = blockSize
        self.modelName = modelName
        self.cacheSalt = cacheSalt
    }

    // MARK: - Single block

    /// Hash one block. `parent == nil` marks the first block of the chain
    /// (the root marker is hashed instead, and the salt section applies).
    public func blockHash(parent: Data?, blockTokens: some Collection<Int>) -> Data {
        var h = SHA256()
        if !modelName.isEmpty { h.update(data: Data(modelName.utf8)) }
        h.update(data: parent ?? Self.rootMarker)
        if parent == nil, !cacheSalt.isEmpty {
            let saltBytes = Data(cacheSalt.utf8)
            h.update(data: Self.saltTag)
            var len = UInt64(saltBytes.count).littleEndian
            withUnsafeBytes(of: &len) { h.update(data: Data($0)) }
            h.update(data: saltBytes)
        }
        // Mirror the Python tuple repr used by omlx: "(1, 2, 3)".
        let repr = "(\(blockTokens.map(String.init).joined(separator: ", ")))"
        h.update(data: Data(repr.utf8))
        return Data(h.finalize())
    }

    // MARK: - Chains

    /// Number of whole blocks covered by `tokenCount` tokens.
    public func fullBlockCount(tokenCount: Int) -> Int {
        max(0, tokenCount) / blockSize
    }

    /// Largest number of whole blocks a LOOKUP over `tokenCount` tokens may
    /// match: at least one token must remain uncached so the engine always
    /// recomputes the final position for logits (vLLM last-token rule,
    /// report 08 §2). Equivalent to reserving the last block when the count
    /// is an exact block multiple.
    public func maxLookupBlocks(tokenCount: Int) -> Int {
        max(0, tokenCount - 1) / blockSize
    }

    /// Chain hashes `[h_1, …, h_k]` for the first `k` whole blocks of
    /// `tokens`, where `k = min(fullBlockCount, maxBlocks ?? ∞)`.
    /// `h_j` (index `j-1`) fingerprints the entire prefix `tokens[0 ..< j·bs]`.
    public func chainHashes(tokens: [Int], maxBlocks: Int? = nil) -> [Data] {
        var count = fullBlockCount(tokenCount: tokens.count)
        if let maxBlocks { count = min(count, max(0, maxBlocks)) }
        guard count > 0 else { return [] }

        var hashes: [Data] = []
        hashes.reserveCapacity(count)
        var parent: Data? = nil
        for i in 0 ..< count {
            let start = i * blockSize
            let hash = blockHash(parent: parent, blockTokens: tokens[start ..< start + blockSize])
            hashes.append(hash)
            parent = hash
        }
        return hashes
    }
}
