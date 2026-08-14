// Copyright © 2026 Eigen Labs.
//
// Engine-facing abstraction for batched Multi-Token Prediction (MTP)
// speculative decoding. Declared in MLXLMCommon so the continuous-batching
// engine (`GenerationBatch`) can drive MTP without importing MLXLLM (which
// would create a layering cycle). Concrete implementations live in the
// model layer (MLXLLM model families with MTP heads).

import Foundation
import MLX

/// An in-progress batched MTP session over a fixed set of active rows.
///
/// One `step` runs a full speculative round (draft → verify → accept-walk →
/// exact confirmed-prefix rollback) over the active rows and returns the
/// per-row emitted tokens (each ≥ 1). The session owns the MTP carry state
/// (per-row pending tokens, the target hidden seed, the shared-KV snapshot);
/// the caller owns the target KV caches, per-row budgets, stop detection, and
/// emission.
public protocol BatchedMTPSession: AnyObject {
    /// Number of currently active rows (aligns with `step` input/output order).
    var activeCount: Int { get }

    /// Run exactly one MTP round.
    /// - Parameter caches: the target per-layer KV caches (trimmed in place by
    ///   the round's rollback).
    /// - Parameter maxNewPerRow: per active-row cap on tokens emitted this
    ///   round (the row's remaining budget); every entry must be ≥ 1.
    /// - Returns: per active-row emitted token ids, each `1...maxNewPerRow[b]`.
    func step(caches: [any KVCache], maxNewPerRow: [Int]) -> [[Int]]

    /// Shrink the carry to keep only the active-row indices in `keepLocal`
    /// (after some rows finished). The caller separately filters the caches.
    func compactCarry(keepLocal: [Int])
}

/// A model-specific runtime that can begin batched MTP sessions. Held by the
/// engine (via `ContinuousBatchingConfig`) and consulted at decode time.
///
/// `@unchecked Sendable`: implementations hold non-Sendable model/`MLXArray`
/// state but are only ever touched from the engine's single serialized
/// executor (mirrors the rest of the continuous-batching design).
public protocol BatchedMTPRuntime: AnyObject, Sendable {
    /// Speculate only when the active batch size is ≤ this. Above it, the
    /// engine falls back to plain batched decode (on the MoE target the
    /// expert-union verify tax makes speculation break-even by ~B=4).
    var maxBatch: Int { get }

    /// Verify width / block size for each round (drafts = blockSize − 1).
    var blockSize: Int { get }

    /// Seed a session by advancing `caches` over `seedTokens` (`[B]` — each
    /// row's most recent sampled token, not yet in the cache) and capturing
    /// the per-row MTP carry. The seed forward also samples each row's next
    /// token (the "bonus"), returned alongside the session so the caller can
    /// emit `[seedToken, bonus]` per row before the first round (the bonus is
    /// the round-1 draft seed and is not re-emitted by the round).
    /// Returns nil when MTP cannot run for this model/state (e.g. the model
    /// isn't MTP-capable or the drafter is unbound) — caller stays on plain
    /// decode.
    func beginSession(
        seedTokens: MLXArray, caches: [any KVCache]
    ) -> (session: BatchedMTPSession, bonus: [Int])?
}
