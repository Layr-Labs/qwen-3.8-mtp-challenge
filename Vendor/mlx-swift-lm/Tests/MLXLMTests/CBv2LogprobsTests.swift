// CBv2LogprobsTests.swift — end-to-end logprobs emission through EngineV2.
//
// Drives the REAL engine loop (chained decode included) with a scripted
// deterministic-logits model and the production CBv2DefaultSampler, and
// asserts the CBv2Event.delta logprobs surface:
//
//  (i)   the chosen token's logprob matches a scalar-reference log_softmax
//        of the RAW logits — including under a logit bias that steers
//        sampling away from the raw argmax (raw-logprobs-first rule);
//  (ii)  topLogprobs = 3 reports exactly 3 alternatives, ordered descending,
//        matching the reference ranking;
//  (iii) requests with topLogprobs == 0 pay ZERO extra tensor work: no
//        logprob graph nodes are ever built (pipeline + sampler counters);
//  (iv)  batch invariance: a logprobs row batched next to a non-logprobs
//        row — both produce their solo tokens, the logprobs row reports its
//        solo logprob values, the other row reports nil.
//
// No model weights: logits come from a fixed distinct-valued table.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - Scripted table model

/// Deterministic `CBv2SteppableModel`: for input token t the logits are
/// `table[(t + 1) % vocab]`, a row of DISTINCT values whose argmax sits at
/// index (t + 1) % vocab. Greedy decode therefore produces the scripted
/// +1 token chain while every row still has an unambiguous top-k ranking
/// for reference logprob checks.
private final class CBv2LogprobTableModel: CBv2SteppableModel, @unchecked Sendable {
    let vocab: Int
    /// Host copy for scalar references.
    let table: [[Float]]
    private let deviceTable: MLXArray

    init(vocab: Int = 24) {
        self.vocab = vocab
        var table = [[Float]]()
        table.reserveCapacity(vocab)
        for r in 0 ..< vocab {
            // (r·31 + j·17) mod 89 is injective in j for vocab ≤ 89, and the
            // resulting values lie in [-4, 4] on a 1/11 grid — distinct per
            // row, far from fp tie boundaries.
            var row = (0 ..< vocab).map { j in
                Float((r * 31 + j * 17) % 89) / 11.0 - 4.0
            }
            row[r] = 6.0  // unique maximum ⇒ greedy next-token = row index
            table.append(row)
        }
        self.table = table
        self.deviceTable = MLXArray(table.flatMap { $0 }, [vocab, vocab])
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        let next = (tokens + 1) % vocab
        return take(deviceTable, next, axis: 0)  // [B, L, vocab]
    }
}

// MARK: - Helpers

private func logSoftmax(_ logits: [Float]) -> [Float] {
    let maxV = logits.max() ?? 0
    let sum = logits.reduce(Float(0)) { $0 + exp($1 - maxV) }
    let lse = maxV + log(sum)
    return logits.map { $0 - lse }
}

private struct CollectedWithLogprobs {
    var tokens: [Int] = []
    /// One entry per emitted token, aligned with `tokens` (nil = no
    /// logprobs reported in that delta).
    var logprobs: [CBv2TokenLogprob?] = []
    var finishReason: CBv2FinishReason?
}

private func collectWithLogprobs(
    _ stream: AsyncStream<CBv2Event>, timeoutSeconds: TimeInterval = 20
) async -> CollectedWithLogprobs {
    let task = Task { () -> CollectedWithLogprobs in
        var collected = CollectedWithLogprobs()
        for await event in stream {
            switch event {
            case .delta(_, let tokens, let logprobs):
                for (i, token) in tokens.enumerated() {
                    collected.tokens.append(token)
                    collected.logprobs.append(
                        (logprobs?.indices.contains(i) ?? false) ? logprobs![i] : nil)
                }
            case .finished(let reason, _):
                collected.finishReason = reason
                return collected
            }
        }
        return collected
    }
    let watchdog = Task {
        try? await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000))
        task.cancel()
    }
    let result = await task.value
    watchdog.cancel()
    return result
}

// MARK: - Suite

final class CBv2LogprobsTests: XCTestCase {

    private let vocab = 24

    private struct Harness {
        let engine: EngineV2
        let model: CBv2LogprobTableModel
        let sampler: CBv2DefaultSampler
    }

    private func makeHarness() -> Harness {
        let model = CBv2LogprobTableModel(vocab: vocab)
        let sampler = CBv2DefaultSampler(fallbackSeed: 1)
        let layerKinds = CBv2SchedFixtures.tinyLayerKinds()
        let engine = EngineV2(
            model: model,
            layerKinds: layerKinds,
            backend: CBv2SchedMockBackend(),
            cacheProvider: CBv2SchedMockCacheProvider(layerKinds: layerKinds),
            sampler: sampler,
            admissionConfig: .init(watermarkFraction: 0))
        return Harness(engine: engine, model: model, sampler: sampler)
    }

    private func request(
        id: UInt64, prompt: [Int], maxTokens: Int, topLogprobs: Int,
        logitBias: [Int: Float] = [:]
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(
                temperature: 0, logitBias: logitBias, topLogprobs: topLogprobs),
            maxTokens: maxTokens)
    }

    /// Reference: the raw logits row that produced a step's sample is
    /// `table[(input + 1) % vocab]` for that step's input token.
    private func referenceLogprobs(model: CBv2LogprobTableModel, input: Int) -> [Float] {
        logSoftmax(model.table[(input + 1) % model.vocab])
    }

    // MARK: (i) + (ii) chosen logprob & ordered top-k vs reference

    func testChosenAndTopKLogprobsMatchReferenceLogSoftmax() async throws {
        let harness = makeHarness()
        let prompt = [3, 5]
        let budget = 6
        let collected = await collectWithLogprobs(
            try harness.engine.submit(
                request(id: 1, prompt: prompt, maxTokens: budget, topLogprobs: 3)))
        await harness.engine.shutdown()

        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertEqual(collected.tokens.count, budget)
        // Greedy scripted chain: last prompt token 5 ⇒ 6, 7, 8, …
        XCTAssertEqual(collected.tokens, (0 ..< budget).map { (5 + 1 + $0) % vocab })

        var input = prompt.last!
        for (i, token) in collected.tokens.enumerated() {
            let expected = referenceLogprobs(model: harness.model, input: input)
            let lp = try XCTUnwrap(collected.logprobs[i], "delta \(i) must carry logprobs")
            XCTAssertEqual(lp.token, token)
            XCTAssertEqual(
                lp.logprob, expected[token], accuracy: 2e-4,
                "chosen logprob must equal reference log_softmax (step \(i))")

            // topLogprobs = 3 ⇒ exactly 3 ordered alternatives matching the
            // reference ranking (greedy: rank 0 is the chosen token itself).
            XCTAssertEqual(lp.topLogprobs.count, 3)
            let ranked = (0 ..< vocab).sorted { expected[$0] > expected[$1] }
            for j in 0 ..< 3 {
                XCTAssertEqual(lp.topLogprobs[j].token, ranked[j], "step \(i) rank \(j)")
                XCTAssertEqual(
                    lp.topLogprobs[j].logprob, expected[ranked[j]], accuracy: 2e-4)
                if j > 0 {
                    XCTAssertGreaterThanOrEqual(
                        lp.topLogprobs[j - 1].logprob, lp.topLogprobs[j].logprob,
                        "alternatives must be ordered descending")
                }
            }
            XCTAssertEqual(lp.topLogprobs[0].token, token, "greedy pick is the raw argmax")
            input = token
        }
    }

    // MARK: (i-b) raw-first rule under a sampling-steering logit bias

    func testLogprobsReflectRawLogitsDespiteLogitBias() async throws {
        let harness = makeHarness()
        let prompt = [2]
        let budget = 4
        let biased = 0  // +100 bias ⇒ every greedy pick is token 0
        let collected = await collectWithLogprobs(
            try harness.engine.submit(
                request(
                    id: 1, prompt: prompt, maxTokens: budget, topLogprobs: 2,
                    logitBias: [biased: 100])))
        await harness.engine.shutdown()

        XCTAssertEqual(collected.tokens, Array(repeating: biased, count: budget))
        var input = prompt.last!
        for i in 0 ..< budget {
            let expected = referenceLogprobs(model: harness.model, input: input)
            let rawArgmax = (input + 1) % vocab
            let lp = try XCTUnwrap(collected.logprobs[i])
            XCTAssertEqual(lp.token, biased)
            XCTAssertEqual(
                lp.logprob, expected[biased], accuracy: 2e-4,
                "chosen logprob must be the RAW (pre-bias) log_softmax value")
            XCTAssertEqual(
                lp.topLogprobs[0].token, rawArgmax,
                "top alternative ranks by RAW logits, not by the biased ones")
            XCTAssertNotEqual(rawArgmax, biased, "test premise: bias steers off the raw argmax")
            input = biased
        }
    }

    // MARK: (iii) zero tensor work when no row asks

    func testNoLogprobGraphNodesWhenNotRequested() async throws {
        let harness = makeHarness()
        let collected = await collectWithLogprobs(
            try harness.engine.submit(
                request(id: 1, prompt: [4, 9], maxTokens: 8, topLogprobs: 0)))
        await harness.engine.shutdown()

        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertEqual(collected.tokens.count, 8)
        XCTAssertTrue(
            collected.logprobs.allSatisfy { $0 == nil },
            "topLogprobs == 0 must never report logprobs")
        XCTAssertEqual(
            harness.sampler.pipelineLogprobBuildCount, 0,
            "no raw-logprob capture may be built for logprobs-free batches")
        XCTAssertEqual(
            harness.sampler.logprobGatherCount, 0,
            "no gather nodes may be built for logprobs-free batches")
    }

    // MARK: (iv) batch invariance: logprobs row next to a non-logprobs row

    func testLogprobsRowBatchedWithPlainRowBothCorrect() async throws {
        let promptA = [2]
        let promptB = [9]
        let budget = 6

        // Solo baselines on fresh engines.
        let soloHarnessA = makeHarness()
        let soloA = await collectWithLogprobs(
            try soloHarnessA.engine.submit(
                request(id: 900, prompt: promptA, maxTokens: budget, topLogprobs: 3)))
        await soloHarnessA.engine.shutdown()

        let soloHarnessB = makeHarness()
        let soloB = await collectWithLogprobs(
            try soloHarnessB.engine.submit(
                request(id: 901, prompt: promptB, maxTokens: budget, topLogprobs: 0)))
        await soloHarnessB.engine.shutdown()

        // Batched run: both submitted before consumption so they decode
        // side by side.
        let harness = makeHarness()
        let streamA = try harness.engine.submit(
            request(id: 1, prompt: promptA, maxTokens: budget, topLogprobs: 3))
        let streamB = try harness.engine.submit(
            request(id: 2, prompt: promptB, maxTokens: budget, topLogprobs: 0))
        let batchedA = await collectWithLogprobs(streamA)
        let batchedB = await collectWithLogprobs(streamB)
        await harness.engine.shutdown()

        XCTAssertEqual(batchedA.tokens, soloA.tokens, "A's tokens must be batch-invariant")
        XCTAssertEqual(batchedB.tokens, soloB.tokens, "B's tokens must be batch-invariant")
        XCTAssertTrue(
            batchedB.logprobs.allSatisfy { $0 == nil },
            "the non-logprobs batchmate must keep reporting nil")

        for i in 0 ..< budget {
            let solo = try XCTUnwrap(soloA.logprobs[i])
            let batched = try XCTUnwrap(
                batchedA.logprobs[i], "A must keep its logprobs under batching (step \(i))")
            XCTAssertEqual(batched.token, solo.token)
            XCTAssertEqual(batched.logprob, solo.logprob, accuracy: 1e-5)
            XCTAssertEqual(batched.topLogprobs.count, solo.topLogprobs.count)
            for j in 0 ..< solo.topLogprobs.count {
                XCTAssertEqual(batched.topLogprobs[j].token, solo.topLogprobs[j].token)
                XCTAssertEqual(
                    batched.topLogprobs[j].logprob, solo.topLogprobs[j].logprob,
                    accuracy: 1e-5)
            }
        }
    }
}
