// Tests for Sampling.swift and repetition-penalty helpers — §3-4 of ContinuousBatchingTestPlan.md

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - §3  Sampling

final class CBSamplingTests: XCTestCase {

    // Helper: [1, vocab] logprob row.
    private func logprobs(_ values: [Float]) -> MLXArray {
        MLXArray(values)[.newAxis, .ellipsis]
    }

    func testApplyTopPFiltersLowProbTokens() {
        // One dominant token at index 2; others ~0.
        let lp = logprobs([-10, -10, 0, -10, -10])
        let filtered = applyTopP(lp, topP: 0.9)
        // All tokens other than token 2 should be masked to -inf.
        let flat = filtered.reshaped(-1).asArray(Float.self)
        let finite = flat.filter { $0.isFinite }
        XCTAssertEqual(finite.count, 1)
        XCTAssertTrue(flat[2].isFinite)
    }

    func testApplyMinPFiltersTokensBelowThreshold() {
        // token 0: log(0.8), token 1: log(0.1), token 2: log(0.1)
        // minP = 0.5 → threshold = 0.5 * 0.8 = 0.4 → keep only token 0
        let lp = logprobs([log(0.8), log(0.1), log(0.1)])
        let filtered = applyMinP(lp, minP: 0.5)
        let flat = filtered.reshaped(-1).asArray(Float.self)
        let finite = flat.filter { $0.isFinite }
        XCTAssertEqual(finite.count, 1, "only the dominant token survives minP filter")
        XCTAssertTrue(flat[0].isFinite)
    }

    func testTemperatureZeroIsGreedy() {
        let sampler = makeRowSampler(temperature: 0)
        // [1, 5] logit row; token 3 has highest logit.
        let logits = MLXArray([0.1, 0.2, 0.1, 5.0, 0.1] as [Float])[.newAxis, .ellipsis]
        for _ in 0 ..< 5 {
            XCTAssertEqual(sampler(logits).item(Int.self), 3)
        }
    }

    func testTopKOneAlwaysSelectsBestToken() {
        let sampler = makeRowSampler(temperature: 1.0, topK: 1)
        let logits = MLXArray([0.1, 5.0, 0.2, 0.1] as [Float])[.newAxis, .ellipsis]
        for _ in 0 ..< 5 {
            XCTAssertEqual(sampler(logits).item(Int.self), 1)
        }
    }
}

// MARK: - §4  Repetition penalty

final class CBRepetitionPenaltyTests: XCTestCase {

    private func baseGreedy() -> RowSampler { { argMax($0, axis: -1) } }

    // Helper: run sampler on a [1, vocab] logit tensor.
    private func sample(_ sampler: RowSampler, _ logits: [Float]) -> Int {
        sampler(MLXArray(logits)[.newAxis, .ellipsis]).item(Int.self)
    }

    func testRepetitionSamplerPassesThroughWithNoHistory() {
        let history = TokenHistoryHolder(tokens: [])
        let sampler = makeRepetitionSampler(
            base: baseGreedy(), history: history,
            repetitionPenalty: 2.0, presencePenalty: 0.5, frequencyPenalty: 0.5
        )
        // With no history the output must equal the base (greedy) sampler.
        let logits: [Float] = [0.1, 0.2, 3.0, 0.0]
        XCTAssertEqual(sample(sampler, logits), 2)
    }

    func testRepetitionSamplerReducesPositiveLogit() {
        // Token 0 has positive logit 2.0 and is in history → should be halved to 1.0.
        // Token 1 has logit 1.5 and is NOT in history → stays 1.5.
        // After penalty, token 1 should be chosen (1.5 > 1.0).
        let history = TokenHistoryHolder(tokens: [0])
        let sampler = makeRepetitionSampler(
            base: baseGreedy(), history: history,
            repetitionPenalty: 2.0, presencePenalty: 0.0, frequencyPenalty: 0.0
        )
        // token 0 → 2.0/2 = 1.0; token 1 → 1.5 (unchanged)
        XCTAssertEqual(sample(sampler, [2.0, 1.5, 0.0, 0.0]), 1)
    }

    func testRepetitionSamplerAmplifiesNegativeLogit() {
        // Token 0 has negative logit -1.0 and is in history → multiplied by 2 = -2.0.
        // Token 1 has logit -0.5 → stays -0.5.
        // After penalty, token 1 should win (-0.5 > -2.0).
        let history = TokenHistoryHolder(tokens: [0])
        let sampler = makeRepetitionSampler(
            base: baseGreedy(), history: history,
            repetitionPenalty: 2.0, presencePenalty: 0.0, frequencyPenalty: 0.0
        )
        XCTAssertEqual(sample(sampler, [-1.0, -0.5, -2.0, -3.0]), 1)
    }

    func testPresencePenaltyAppliesFlatDecrease() {
        // Token 0 at logit 2.0; presencePenalty = 1.0 → 2.0 - 1.0 = 1.0.
        // Token 1 at logit 1.5 (no history) → 1.5; token 1 wins.
        let history = TokenHistoryHolder(tokens: [0])
        let sampler = makeRepetitionSampler(
            base: baseGreedy(), history: history,
            repetitionPenalty: 1.0, presencePenalty: 1.0, frequencyPenalty: 0.0
        )
        XCTAssertEqual(sample(sampler, [2.0, 1.5, 0.0, 0.0]), 1)
    }

    func testFrequencyPenaltyScalesWithCount() {
        // Token 0 appears 3 times in history with frequencyPenalty = 0.5.
        // Penalty = 0.5 * 3 = 1.5.  logit[0] = 3.0 - 1.5 = 1.5.
        // Token 1 at logit 1.6 (no history) → wins.
        let history = TokenHistoryHolder(tokens: [0, 0, 0])
        let sampler = makeRepetitionSampler(
            base: baseGreedy(), history: history,
            repetitionPenalty: 1.0, presencePenalty: 0.0, frequencyPenalty: 0.5
        )
        XCTAssertEqual(sample(sampler, [3.0, 1.6, 0.0, 0.0]), 1)
    }
}

// MARK: - §4b  On-device penalty sampler (gather/scatter) numerics

/// Box used to capture the penalized logits the repetition sampler hands to its
/// base sampler, so the test can compare them against the reference CPU formula.
private final class CapturedLogitsBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: MLXArray?
    func set(_ v: MLXArray) {
        lock.lock(); value = v; lock.unlock()
    }
    func get() -> MLXArray? {
        lock.lock(); defer { lock.unlock() }; return value
    }
}

/// Verifies the on-device gather/scatter repetition sampler is numerically
/// equivalent to the original CPU formula it replaced, and that the incremental
/// counts maintained on `TokenHistoryHolder.append` match a full rescan.
final class CBRepetitionPenaltyDeviceTests: XCTestCase {

    /// The exact original CPU formula, computed host-side for reference.
    private func cpuReference(
        logits: MLXArray, history: [Int],
        repetitionPenalty rp: Float, presencePenalty pp: Float, frequencyPenalty fp: Float
    ) -> [Float] {
        let vocab = logits.dim(-1)
        var flat = logits.reshaped(-1).asArray(Float.self)
        var counts: [Int: Int] = [:]
        for t in history where t >= 0 && t < vocab { counts[t, default: 0] += 1 }
        for (tokenId, count) in counts {
            var v = flat[tokenId]
            if rp != 1.0 { v = v > 0 ? v / rp : v * rp }
            v -= pp
            v -= fp * Float(count)
            flat[tokenId] = v
        }
        return flat
    }

    /// Capture the logits passed to `base` after on-device penalization.
    private func devicePenalized(
        logits: MLXArray, history: TokenHistoryHolder,
        repetitionPenalty rp: Float, presencePenalty pp: Float, frequencyPenalty fp: Float
    ) -> [Float] {
        let captured = CapturedLogitsBox()
        let sampler = makeRepetitionSampler(
            base: { @Sendable l in captured.set(l); return argMax(l, axis: -1) },
            history: history,
            repetitionPenalty: rp, presencePenalty: pp, frequencyPenalty: fp
        )
        _ = sampler(logits)
        return captured.get()!.reshaped(-1).asArray(Float.self)
    }

    func testOnDevicePenaltyMatchesCPUFormula() {
        let vocab = 16
        // Repeats across positive and negative logits exercise both the
        // repetition-scaling branches and the per-count frequency penalty.
        let history = [1, 1, 2, 5, 5, 5, 7, 0, 3, 3, 9, 9, 2]
        let rp: Float = 1.3
        let pp: Float = 0.4
        let fp: Float = 0.25

        // Mixed-sign deterministic row [1, vocab]: -7.5 ... +7.5 (no exact zeros).
        let raw: [Float] = (0 ..< vocab).map { Float($0) - 7.5 }
        let logits = MLXArray(raw)[.newAxis, .ellipsis]

        let expected = cpuReference(
            logits: logits, history: history,
            repetitionPenalty: rp, presencePenalty: pp, frequencyPenalty: fp)
        let got = devicePenalized(
            logits: logits, history: TokenHistoryHolder(tokens: history),
            repetitionPenalty: rp, presencePenalty: pp, frequencyPenalty: fp)

        XCTAssertEqual(got.count, vocab)
        for i in 0 ..< vocab {
            XCTAssertEqual(got[i], expected[i], accuracy: 1e-4, "mismatch at token \(i)")
        }
    }

    /// presence/frequency-only (repetitionPenalty == 1) must also match — the
    /// repetition-scaling branch is skipped in that case.
    func testOnDevicePresenceFrequencyOnlyMatchesCPUFormula() {
        let vocab = 16
        let history = [2, 2, 2, 8, 0, 0, 14]
        let rp: Float = 1.0
        let pp: Float = 0.6
        let fp: Float = 0.5

        let raw: [Float] = (0 ..< vocab).map { Float($0) * 0.5 - 4.0 }
        let logits = MLXArray(raw)[.newAxis, .ellipsis]

        let expected = cpuReference(
            logits: logits, history: history,
            repetitionPenalty: rp, presencePenalty: pp, frequencyPenalty: fp)
        let got = devicePenalized(
            logits: logits, history: TokenHistoryHolder(tokens: history),
            repetitionPenalty: rp, presencePenalty: pp, frequencyPenalty: fp)

        for i in 0 ..< vocab {
            XCTAssertEqual(got[i], expected[i], accuracy: 1e-4, "mismatch at token \(i)")
        }
    }

    /// The incremental counts maintained on `append` must produce identical
    /// penalized logits to a holder constructed from the full token list.
    func testIncrementalCountsMatchFullRescan() {
        let vocab = 16
        let prompt = [3, 3, 1, 0]
        let appended = [5, 5, 3, 9, 5, 1]
        let rp: Float = 1.5
        let pp: Float = 0.3
        let fp: Float = 0.5

        let raw: [Float] = (0 ..< vocab).map { Float($0) - 8.0 }
        let logits = MLXArray(raw)[.newAxis, .ellipsis]

        let full = TokenHistoryHolder(tokens: prompt + appended)
        let incremental = TokenHistoryHolder(tokens: prompt)
        for t in appended { incremental.append(t) }

        XCTAssertEqual(full.counts, incremental.counts, "incremental counts diverged from rescan")

        let a = devicePenalized(
            logits: logits, history: full,
            repetitionPenalty: rp, presencePenalty: pp, frequencyPenalty: fp)
        let b = devicePenalized(
            logits: logits, history: incremental,
            repetitionPenalty: rp, presencePenalty: pp, frequencyPenalty: fp)

        for i in 0 ..< vocab {
            XCTAssertEqual(a[i], b[i], accuracy: 1e-6, "incremental vs rescan mismatch at token \(i)")
        }
    }

    /// Empty history and out-of-range-only history must pass through to base
    /// untouched (the two original guards).
    func testGuardsPassThroughUnchanged() {
        let vocab = 8
        let raw: [Float] = (0 ..< vocab).map { Float($0) - 3.0 }
        let logits = MLXArray(raw)[.newAxis, .ellipsis]

        // Empty history → identical to input.
        let empty = devicePenalized(
            logits: logits, history: TokenHistoryHolder(tokens: []),
            repetitionPenalty: 2.0, presencePenalty: 1.0, frequencyPenalty: 1.0)
        for i in 0 ..< vocab {
            XCTAssertEqual(empty[i], raw[i], accuracy: 1e-6)
        }

        // Only out-of-range ids (>= vocab) → no in-range counts → unchanged.
        let oob = devicePenalized(
            logits: logits, history: TokenHistoryHolder(tokens: [99, 100, 100]),
            repetitionPenalty: 2.0, presencePenalty: 1.0, frequencyPenalty: 1.0)
        for i in 0 ..< vocab {
            XCTAssertEqual(oob[i], raw[i], accuracy: 1e-6)
        }
    }
}
