// SamplerV2.swift
//
// Vectorized token selection for ContinuousBatchingV2 (workstream E).
// Consumes the transformed [B, vocab] logits produced by
// `LogitsPipelineV2.process` and returns one token per row [B].
//
//  - All-greedy fast path: a single `argMax` when every row is greedy
//    (temperature < 1e-5), matching the legacy vectorized greedy path
//    bit for bit.
//  - Mixed batches: Gumbel-max via the exponential-noise race
//    (`argmax(probs / e)`, e ~ Exp(1)) so there is no multinomial and no
//    host sync. Greedy and sampled picks are merged with a per-row
//    `where(greedy, argmax, sampled)`.
//  - Per-row keyed RNG: each row's noise is generated from its own key,
//    derived ONLY from (seed, requestID, per-request step index). A row's
//    random stream therefore never depends on its batchmates or on when it
//    joined the batch — the batch-composition-invariance requirement
//    (research report 12, item 5). Reproducibility is best-effort under
//    batching, per the contract: fixed (seed, requestID) reproduces the
//    same stream across runs given the same per-row logits.
//
// Threading: `sample` builds graph nodes only (no eval, no `.item()`).
// `setRows` runs at batch-membership change; `commit` is O(B) host counter
// bookkeeping (no device work).

import Foundation
import MLX
import MLXRandom

public final class SamplerV2 {

    /// Temperatures below this are treated as greedy (matches vLLM and
    /// `LogitsPipelineV2.greedyEpsilon`).
    public static let greedyEpsilon: Float = LogitsPipelineV2.greedyEpsilon

    /// Engine-level fallback used for rows that did not supply a seed.
    /// Fixed at init so nil-seed rows are still batch-invariant within a
    /// process; across runs they are intentionally non-deterministic.
    private let fallbackSeed: UInt64

    private struct RowState {
        var id: CBv2RequestID
        var seed: UInt64
        var greedy: Bool
        /// Per-request decode step index (number of tokens sampled so far
        /// for this request). Keying noise to the per-request index — not
        /// the global engine step — is what keeps a row's stream
        /// independent of when it joined the batch.
        var step: UInt64
    }

    private var rows: [RowState] = []
    private var allGreedy = true
    /// [B] bool, built once per membership change.
    private var greedyMask: MLXArray?

    public init(fallbackSeed: UInt64? = nil) {
        self.fallbackSeed = fallbackSeed ?? UInt64.random(in: .min ... .max)
    }

    // MARK: Membership change

    /// Rebuild per-row sampling state. Row order must match the logits row
    /// order passed to `sample`. Mid-flight rows resume their step index
    /// from `outputTokens.count`, so re-vectorization after a membership
    /// change does not perturb their random streams.
    public func setRows(_ rows: [CBv2SamplerRow]) {
        self.rows = rows.map { row in
            RowState(
                id: row.id,
                seed: row.params.seed ?? fallbackSeed,
                greedy: row.params.temperature < Self.greedyEpsilon,
                step: UInt64(row.outputTokens.count)
            )
        }
        allGreedy = self.rows.allSatisfy(\.greedy)
        greedyMask = self.rows.isEmpty ? nil : MLXArray(self.rows.map(\.greedy))
    }

    // MARK: Step path

    /// Select one token per row from transformed logits [B, vocab].
    /// Returns [B] int32. Pure graph construction — call `commit()` once
    /// per step afterwards to advance the per-row step indices.
    public func sample(from logits: MLXArray) -> MLXArray {
        precondition(
            logits.dim(0) == rows.count,
            "logits rows (\(logits.dim(0))) != configured rows (\(rows.count)) — call setRows")

        // Fast path: every row greedy ⇒ one argMax, bit-identical to the
        // legacy vectorized greedy decode.
        let greedyTokens = argMax(logits, axis: -1).asType(.int32)
        if allGreedy {
            return greedyTokens
        }

        // Mixed batch: exponential-race Gumbel-max. probs/e keeps masked
        // tokens (-inf logits → probability 0) unreachable, and argmax of
        // the ratio is an exact categorical draw.
        let vocab = logits.dim(-1)
        let probs = softmax(logits, axis: -1)
        let noise = exponentialNoise(vocab: vocab)
        let sampledTokens = argMax(probs / noise, axis: -1).asType(.int32)

        guard let greedyMask else { return sampledTokens }
        return which(greedyMask, greedyTokens, sampledTokens)
    }

    /// Advance every row's per-request step index. Call exactly once per
    /// sampled step, after `sample` (alongside `LogitsPipelineV2.commit`).
    public func commit() {
        for i in rows.indices {
            rows[i].step &+= 1
        }
    }

    // MARK: - Per-row keyed noise

    /// Exp(1) noise [B, vocab]; row r is generated from key
    /// mix(seed_r, id_r, step_r) and stacked, so each row's draw is a pure
    /// function of that row's identity — never of batchmates. Greedy rows
    /// receive a constant placeholder (their sampled pick is discarded by
    /// the `where` merge).
    private func exponentialNoise(vocab: Int) -> MLXArray {
        var perRow = [MLXArray]()
        perRow.reserveCapacity(rows.count)
        for row in rows {
            if row.greedy {
                perRow.append(MLXArray.full([1, vocab], values: MLXArray(Float(1))))
                continue
            }
            let key = MLXRandom.key(Self.mix(seed: row.seed, id: row.id.raw, step: row.step))
            let u = MLXRandom.uniform(
                low: Float(0), high: Float(1), [1, vocab], type: Float.self, key: key)
            // e = -log(1 - u) ~ Exp(1); u ∈ [0, 1) keeps the log argument in
            // (0, 1]. Clamp away e == 0 (u == 0) so probs/e never divides
            // by zero.
            let e = maximum(-log(1 - u), MLXArray(Float(1e-20)))
            perRow.append(e)
        }
        return concatenated(perRow, axis: 0)
    }

    /// SplitMix64-style mix of (seed, requestID, step) into one RNG key
    /// seed. Deterministic across processes (no `Hasher`), and a pure
    /// function of exactly the three inputs the contract allows.
    static func mix(seed: UInt64, id: UInt64, step: UInt64) -> UInt64 {
        func splitmix(_ value: UInt64) -> UInt64 {
            var z = value &+ 0x9E37_79B9_7F4A_7C15
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        return splitmix(splitmix(splitmix(seed) ^ id) ^ step)
    }
}
