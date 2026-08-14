// CBv2SamplingTests.swift
//
// Workstream E unit tests: LogitsPipelineV2, SamplerV2, DetokenizerV2,
// StopHoldback. No model weights required — synthetic logits and a
// hand-rolled byte-level tokenizer stub.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - Shared helpers

private let vocab = 16

private func row(
    id: UInt64,
    temperature: Float = 0,
    topP: Float = 1,
    topK: Int = 0,
    minP: Float = 0,
    repetitionPenalty: Float = 1,
    repetitionContextSize: Int = 64,
    frequencyPenalty: Float = 0,
    presencePenalty: Float = 0,
    seed: UInt64? = nil,
    logitBias: [Int: Float] = [:],
    topLogprobs: Int = 0,
    prompt: [Int] = [],
    output: [Int] = []
) -> CBv2SamplerRow {
    CBv2SamplerRow(
        id: CBv2RequestID(id),
        params: CBv2SamplingParams(
            temperature: temperature, topP: topP, topK: topK, minP: minP,
            repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize,
            frequencyPenalty: frequencyPenalty, presencePenalty: presencePenalty,
            seed: seed, logitBias: logitBias, topLogprobs: topLogprobs),
        promptTokens: prompt, outputTokens: output)
}

/// Deterministic pseudo-random floats (SplitMix64-based) so tests are
/// reproducible across runs without touching MLX's global RNG.
private struct TestRNG {
    var state: UInt64
    init(_ seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [-scale, scale), quantized to a 0.25 grid to keep
    /// reference comparisons away from floating-point tie boundaries.
    mutating func logit(scale: Float = 4) -> Float {
        let u = Float(next() >> 40) / Float(1 << 24)  // [0, 1)
        return ((u * 2 - 1) * scale * 4).rounded() / 4
    }
    /// Distinct values per row so sort order (and thus top-k/top-p keep
    /// sets) is unambiguous between MLX and the scalar reference.
    mutating func logits(_ count: Int) -> [Float] {
        var seen = Set<Float>()
        var out = [Float]()
        while out.count < count {
            let value = logit()
            if seen.insert(value).inserted { out.append(value) }
        }
        return out
    }
    mutating func token(_ vocabSize: Int) -> Int {
        Int(next() % UInt64(vocabSize))
    }
}

/// Scalar reference for stages (1)-(3): bias, repetition (windowed over
/// prompt ∪ output), frequency/presence (output only), temperature.
private func referenceTransform(
    logits: [Float], row: CBv2SamplerRow
) -> [Float] {
    var x = logits
    let p = row.params

    for (token, bias) in p.logitBias where token >= 0 && token < x.count {
        x[token] += bias
    }

    if p.repetitionPenalty != 1, p.repetitionContextSize > 0 {
        let history = row.promptTokens + row.outputTokens
        let window = Set(history.suffix(p.repetitionContextSize))
        for token in window where token >= 0 && token < x.count {
            x[token] = x[token] > 0 ? x[token] / p.repetitionPenalty : x[token] * p.repetitionPenalty
        }
    }

    if p.frequencyPenalty != 0 || p.presencePenalty != 0 {
        var counts = [Float](repeating: 0, count: x.count)
        for token in row.outputTokens where token >= 0 && token < x.count {
            counts[token] += 1
        }
        for i in 0 ..< x.count {
            x[i] -= p.frequencyPenalty * counts[i]
            x[i] -= p.presencePenalty * (counts[i] > 0 ? 1 : 0)
        }
    }

    if p.temperature >= LogitsPipelineV2.greedyEpsilon, p.temperature != 1 {
        for i in 0 ..< x.count { x[i] /= p.temperature }
    }
    return x
}

private func logSoftmax(_ logits: [Float]) -> [Float] {
    let maxV = logits.max() ?? 0
    let sum = logits.reduce(Float(0)) { $0 + exp($1 - maxV) }
    let lse = maxV + log(sum)
    return logits.map { $0 - lse }
}

private func batchArray(_ rows: [[Float]]) -> MLXArray {
    MLXArray(rows.flatMap { $0 }, [rows.count, rows[0].count])
}

// MARK: - LogitsPipelineV2

final class CBv2SamplingPipelineTests: XCTestCase {

    func testRawLogprobsReflectRawLogitsDespiteBiasAndPenalties() throws {
        // Row 0: heavy bias + all penalties + logprobs on. Row 1: plain.
        let rows = [
            row(
                id: 1, temperature: 0.7, repetitionPenalty: 2, repetitionContextSize: 8,
                frequencyPenalty: 0.5, presencePenalty: 0.5,
                logitBias: [3: 25, 7: -25], topLogprobs: 3,
                prompt: [1, 2, 3], output: [3, 4]),
            row(id: 2, temperature: 0.9, prompt: [5]),
        ]
        let pipeline = LogitsPipelineV2(vocabSize: vocab)
        pipeline.setRows(rows)

        var rng = TestRNG(7)
        let raw = [rng.logits(vocab), rng.logits(vocab)]
        let out = pipeline.process(batchArray(raw))

        let rawLogprobs = try XCTUnwrap(out.rawLogprobs)
        let got = rawLogprobs.asArray(Float.self)
        for r in 0 ..< 2 {
            let expected = logSoftmax(raw[r])
            for i in 0 ..< vocab {
                XCTAssertEqual(
                    got[r * vocab + i], expected[i], accuracy: 1e-4,
                    "raw logprobs must ignore bias/penalties (row \(r), token \(i))")
            }
        }

        // The sampling logits, by contrast, must reflect the bias.
        let sampling = out.sampling.asArray(Float.self)
        XCTAssertGreaterThan(
            sampling[3] - raw[0][3], 10,
            "bias of +25 must move the sampling logits")
    }

    func testNoLogprobsWhenNoRowAsks() {
        let pipeline = LogitsPipelineV2(vocabSize: vocab)
        pipeline.setRows([row(id: 1), row(id: 2)])
        var rng = TestRNG(9)
        let out = pipeline.process(batchArray([rng.logits(vocab), rng.logits(vocab)]))
        XCTAssertNil(out.rawLogprobs)
    }

    func testPenaltiesParityVsScalarReferenceOnRandomHistories() {
        var rng = TestRNG(11)
        let rows = (0 ..< 4).map { i -> CBv2SamplerRow in
            row(
                id: UInt64(i + 1),
                temperature: [0, 0.7, 1.0, 1.3][i],
                repetitionPenalty: [1.5, 1.0, 2.0, 1.2][i],
                repetitionContextSize: [4, 64, 3, 8][i],
                frequencyPenalty: [0.3, 0, 0.7, 0][i],
                presencePenalty: [0.4, 0, 0, 1.1][i],
                logitBias: i == 1 ? [2: 5.5, 9: -3] : [:],
                prompt: (0 ..< 10).map { _ in rng.token(vocab) },
                output: (0 ..< 6).map { _ in rng.token(vocab) })
        }
        let pipeline = LogitsPipelineV2(vocabSize: vocab)
        pipeline.setRows(rows)

        let raw = rows.map { _ in rng.logits(vocab) }
        let got = pipeline.process(batchArray(raw)).sampling.asArray(Float.self)

        for (r, rowSpec) in rows.enumerated() {
            let expected = referenceTransform(logits: raw[r], row: rowSpec)
            for i in 0 ..< vocab {
                XCTAssertEqual(
                    got[r * vocab + i], expected[i], accuracy: 1e-4,
                    "row \(r) token \(i)")
            }
        }
    }

    func testIncrementalCommitMatchesReferenceIncludingWindowEviction() {
        // Tiny repetition window (2) so eviction is exercised, plus
        // frequency/presence so outputCounts are exercised.
        let spec = row(
            id: 42, temperature: 1.0, repetitionPenalty: 2.0, repetitionContextSize: 2,
            frequencyPenalty: 0.25, presencePenalty: 0.5, prompt: [0, 1])
        let pipeline = LogitsPipelineV2(vocabSize: vocab)
        pipeline.setRows([spec])

        // Feed sampled tokens 5, 6, 5, 7 one step at a time.
        let sampled = [5, 6, 5, 7]
        for token in sampled {
            pipeline.commit(sampledTokens: MLXArray([Int32(token)]))
        }

        var rng = TestRNG(13)
        let raw = rng.logits(vocab)
        let got = pipeline.process(batchArray([raw])).sampling.asArray(Float.self)

        // Reference: same row, with the committed tokens as output history.
        var refRow = spec
        refRow.outputTokens = sampled
        let expected = referenceTransform(logits: raw, row: refRow)
        for i in 0 ..< vocab {
            XCTAssertEqual(got[i], expected[i], accuracy: 1e-4, "token \(i)")
        }

        // Window of 2 with history [0,1,5,6,5,7]: only {5,7} may carry the
        // repetition penalty. Sanity-check tokens 0, 1, 6 are unpenalized.
        for clean in [0, 1, 6] {
            var noRep = refRow
            noRep.params.repetitionPenalty = 1
            let unpenalized = referenceTransform(logits: raw, row: noRep)
            XCTAssertEqual(got[clean], unpenalized[clean], accuracy: 1e-4)
        }
    }

    func testMidFlightRebuildMatchesIncrementalState() {
        // Committing tokens step by step must land in the same state as
        // rebuilding with those tokens as outputTokens (membership churn).
        let base = row(
            id: 9, temperature: 1.0, repetitionPenalty: 1.7, repetitionContextSize: 3,
            frequencyPenalty: 0.6, presencePenalty: 0.2, prompt: [1, 2, 3, 4])
        let sampled = [8, 9, 8]

        let incremental = LogitsPipelineV2(vocabSize: vocab)
        incremental.setRows([base])
        for token in sampled {
            incremental.commit(sampledTokens: MLXArray([Int32(token)]))
        }

        var resumed = base
        resumed.outputTokens = sampled
        let rebuilt = LogitsPipelineV2(vocabSize: vocab)
        rebuilt.setRows([resumed])

        var rng = TestRNG(17)
        let raw = rng.logits(vocab)
        let a = incremental.process(batchArray([raw])).sampling.asArray(Float.self)
        let b = rebuilt.process(batchArray([raw])).sampling.asArray(Float.self)
        for i in 0 ..< vocab {
            XCTAssertEqual(a[i], b[i], accuracy: 1e-5, "token \(i)")
        }
    }

    func testTopKTopPMinPVectorizedKeepSets() {
        // Row 0: top-k=3. Row 1: top-p=0.7. Row 2: min-p=0.25. Row 3: off.
        let rows = [
            row(id: 1, temperature: 1, topK: 3),
            row(id: 2, temperature: 1, topP: 0.7),
            row(id: 3, temperature: 1, minP: 0.25),
            row(id: 4, temperature: 1),
        ]
        let pipeline = LogitsPipelineV2(vocabSize: vocab)
        pipeline.setRows(rows)

        var rng = TestRNG(19)
        let raw = rows.map { _ in rng.logits(vocab) }
        let got = pipeline.process(batchArray(raw)).sampling.asArray(Float.self)

        for (r, spec) in rows.enumerated() {
            // Scalar reference keep-set with the pipeline's semantics.
            let x = raw[r]
            let order = (0 ..< vocab).sorted { x[$0] > x[$1] }
            let maxV = x.max()!
            let expSum = x.reduce(Float(0)) { $0 + exp($1 - maxV) }
            let probs = order.map { exp(x[$0] - maxV) / expSum }
            var keep = Set<Int>()
            var cumulative: Float = 0
            for (rank, token) in order.enumerated() {
                let p = probs[rank]
                var kept = true
                if spec.params.topK > 0, rank >= spec.params.topK { kept = false }
                if spec.params.topP < 1, cumulative > spec.params.topP { kept = false }
                if spec.params.minP > 0, p < spec.params.minP * probs[0] { kept = false }
                cumulative += p
                if kept { keep.insert(token) }
            }
            XCTAssertTrue(keep.contains(order[0]), "top token always kept (row \(r))")
            for i in 0 ..< vocab {
                let value = got[r * vocab + i]
                if keep.contains(i) {
                    XCTAssertEqual(value, x[i], accuracy: 1e-4, "row \(r) token \(i) kept")
                } else {
                    XCTAssertEqual(
                        value, -Float.infinity, "row \(r) token \(i) must be masked")
                }
            }
        }
    }

    func testGreedyRowsPassThroughUntouched() {
        // Default rows (greedy, no penalties/bias): process must be the
        // identity (modulo exact f16→f32 style cast; inputs are f32 here).
        let pipeline = LogitsPipelineV2(vocabSize: vocab)
        pipeline.setRows([row(id: 1), row(id: 2)])
        var rng = TestRNG(23)
        let raw = [rng.logits(vocab), rng.logits(vocab)]
        let out = pipeline.process(batchArray(raw)).sampling.asArray(Float.self)
        for r in 0 ..< 2 {
            for i in 0 ..< vocab {
                XCTAssertEqual(out[r * vocab + i], raw[r][i], "must be bit-identical")
            }
        }
        XCTAssertTrue(pipeline.allGreedy)
    }

    func testLogprobGatherAndAssemble() throws {
        let rows = [row(id: 1, topLogprobs: 3), row(id: 2, topLogprobs: 0)]
        let pipeline = LogitsPipelineV2(vocabSize: vocab)
        pipeline.setRows(rows)
        var rng = TestRNG(29)
        let raw = [rng.logits(vocab), rng.logits(vocab)]
        let out = pipeline.process(batchArray(raw))
        let rawLogprobs = try XCTUnwrap(out.rawLogprobs)

        let sampledTokens = [4, 11]
        let gathered = CBv2Logprobs.gather(
            rawLogprobs: rawLogprobs, sampledTokens: MLXArray(sampledTokens.map(Int32.init)),
            k: 3)
        let assembled = CBv2Logprobs.assemble(
            gathered, sampledTokens: sampledTokens, topLogprobsPerRow: [3, 0])

        XCTAssertNil(assembled[1], "row with topLogprobs == 0 reports nil")
        let lp = try XCTUnwrap(assembled[0])
        XCTAssertEqual(lp.token, 4)
        let expected = logSoftmax(raw[0])
        XCTAssertEqual(lp.logprob, expected[4], accuracy: 1e-4)
        XCTAssertEqual(lp.topLogprobs.count, 3)
        let ranked = (0 ..< vocab).sorted { expected[$0] > expected[$1] }
        for (j, item) in lp.topLogprobs.enumerated() {
            XCTAssertEqual(item.token, ranked[j])
            XCTAssertEqual(item.logprob, expected[ranked[j]], accuracy: 1e-4)
        }
    }
}

// MARK: - SamplerV2

final class CBv2SamplingSamplerTests: XCTestCase {

    /// Open-loop decode: feed fixed per-step logits (independent of the
    /// sampled tokens) and collect the sampled token per step.
    private func run(
        rows: [CBv2SamplerRow], steps: Int,
        logitsForStep: (Int) -> [[Float]]
    ) -> [[Int]] {
        let pipeline = LogitsPipelineV2(vocabSize: vocab)
        let sampler = SamplerV2(fallbackSeed: 0)
        pipeline.setRows(rows)
        sampler.setRows(rows)
        var tokens = [[Int]](repeating: [], count: rows.count)
        for step in 0 ..< steps {
            let logits = batchArray(logitsForStep(step))
            let out = pipeline.process(logits)
            let sampled = sampler.sample(from: out.sampling)
            pipeline.commit(sampledTokens: sampled)
            sampler.commit()
            let host = sampled.asArray(Int32.self)
            for r in 0 ..< rows.count { tokens[r].append(Int(host[r])) }
        }
        return tokens
    }

    private func rowXLogits(step: Int) -> [Float] {
        var rng = TestRNG(1000 &+ UInt64(step))
        return rng.logits(vocab)
    }

    func testGreedyFastPathBitIdenticalToArgMax() {
        let rows = [row(id: 1), row(id: 2), row(id: 3)]
        let sampler = SamplerV2(fallbackSeed: 0)
        sampler.setRows(rows)
        var rng = TestRNG(31)
        let raw = rows.map { _ in rng.logits(vocab) }
        let logits = batchArray(raw)
        let sampled = sampler.sample(from: logits).asArray(Int32.self)
        let reference = argMax(logits, axis: -1).asType(.int32).asArray(Int32.self)
        XCTAssertEqual(sampled, reference)
    }

    func testSameSeedRequestIDIdenticalAtB1AndInsideB4AcrossRuns() {
        let steps = 12
        let x = row(id: 77, temperature: 0.8, topP: 0.9, seed: 42)

        // B=1 run.
        let alone = run(rows: [x], steps: steps) { step in [rowXLogits(step: step)] }[0]

        // B=4 run: greedy neighbors, X at row 2, twice.
        let neighbors = [
            row(id: 1), row(id: 2), x, row(id: 3),
        ]
        func batchLogits(step: Int) -> [[Float]] {
            var rng = TestRNG(9000 &+ UInt64(step))
            return (0 ..< 4).map { r in r == 2 ? rowXLogits(step: step) : rng.logits(vocab) }
        }
        let inBatchA = run(rows: neighbors, steps: steps, logitsForStep: batchLogits)[2]
        let inBatchB = run(rows: neighbors, steps: steps, logitsForStep: batchLogits)[2]

        XCTAssertEqual(alone, inBatchA, "batch composition must not change X's tokens")
        XCTAssertEqual(inBatchA, inBatchB, "two identical runs must produce identical tokens")
    }

    func testJoinTimeInvariance() {
        // X resuming mid-flight (outputTokens non-empty) must continue the
        // exact stream it would have produced without the membership change.
        let steps = 10
        let x = row(id: 5, temperature: 1.0, seed: 7)
        let full = run(rows: [x], steps: steps) { step in [rowXLogits(step: step)] }[0]

        // Same request, re-vectorized after 4 steps (join of a batchmate).
        let prefix = run(rows: [x], steps: 4) { step in [rowXLogits(step: step)] }[0]
        var resumed = x
        resumed.outputTokens = prefix
        let tail = run(rows: [resumed, row(id: 6)], steps: steps - 4) { step in
            [rowXLogits(step: step + 4), rowXLogits(step: step)]
        }[0]

        XCTAssertEqual(prefix + tail, full, "step keying must be per-request, not global")
    }

    func testMixedBatchGreedyRowsUnaffectedBySampledNeighbors() {
        let rows = [row(id: 1), row(id: 2, temperature: 1.2, seed: 3)]
        let pipeline = LogitsPipelineV2(vocabSize: vocab)
        let sampler = SamplerV2(fallbackSeed: 0)
        pipeline.setRows(rows)
        sampler.setRows(rows)
        var rng = TestRNG(37)
        let raw = [rng.logits(vocab), rng.logits(vocab)]
        let out = pipeline.process(batchArray(raw))
        let sampled = sampler.sample(from: out.sampling).asArray(Int32.self)

        let greedyExpected = raw[0].firstIndex(of: raw[0].max()!)!
        XCTAssertEqual(Int(sampled[0]), greedyExpected, "greedy row must be pure argmax")
        XCTAssertTrue((0 ..< vocab).contains(Int(sampled[1])))
    }

    func testSampledTokensRespectTopKMask() {
        // With top-k = 1 every sampled token must be the argmax, even at
        // high temperature — proves masked (-inf) tokens are unreachable.
        let x = row(id: 8, temperature: 2.0, topK: 1, seed: 99)
        let tokens = run(rows: [x], steps: 8) { step in [rowXLogits(step: step)] }[0]
        for (step, token) in tokens.enumerated() {
            let logits = rowXLogits(step: step)
            XCTAssertEqual(token, logits.firstIndex(of: logits.max()!)!)
        }
    }

    func testMixDependsOnAllThreeInputs() {
        let base = SamplerV2.mix(seed: 1, id: 2, step: 3)
        XCTAssertNotEqual(base, SamplerV2.mix(seed: 9, id: 2, step: 3))
        XCTAssertNotEqual(base, SamplerV2.mix(seed: 1, id: 9, step: 3))
        XCTAssertNotEqual(base, SamplerV2.mix(seed: 1, id: 2, step: 9))
        XCTAssertEqual(base, SamplerV2.mix(seed: 1, id: 2, step: 3))
    }
}

// MARK: - CBv2DefaultSampler request-id lifecycle (PR#62 review)

final class CBv2SamplingDefaultSamplerLifecycleTests: XCTestCase {

    /// A finished request's id may legally be reused by a FUTURE request.
    /// When the reused row set matches the retired configuration exactly
    /// (same single id), the fingerprint check would skip `setRows` and the
    /// new request would inherit the retired request's RNG step index and
    /// penalty state. `requestDidFinish` must invalidate the fingerprint so
    /// the reused id starts fresh (RNG step 0, empty penalties).
    func testRequestIDReuseAfterFinishGetsFreshSamplerState() {
        let requestID = CBv2RequestID(9001)
        let samplingRow = row(id: 9001, temperature: 1.5, seed: 4242, prompt: [1])
        var rng = TestRNG(2026)
        let logits = batchArray([rng.logits(vocab)])
        eval(logits)

        func draw(_ sampler: CBv2DefaultSampler, steps: Int) -> [Int32] {
            (0 ..< steps).map { _ in
                sampler.sample(
                    logits: logits, params: [samplingRow.params], requestIDs: [requestID],
                    stepIndex: 0, pendingSampledTokens: nil,
                    rowContext: { [samplingRow] }
                ).item(Int32.self)
            }
        }

        let sampler = CBv2DefaultSampler(fallbackSeed: 3)
        let firstRun = draw(sampler, steps: 3)

        // Sanity: the keyed RNG stream must actually advance step to step,
        // otherwise stale reuse would be undetectable by this test.
        let continued = draw(sampler, steps: 3)
        XCTAssertNotEqual(
            continued, firstRun,
            "sampler draws must differ across RNG steps for this regression to bite")

        // The request finishes; a fresh request reuses the id solo — the
        // exact fingerprint-collision scenario.
        sampler.requestDidFinish(requestID)
        let reused = draw(sampler, steps: 3)
        XCTAssertEqual(
            reused, firstRun,
            "a reused id must restart from RNG step 0 with fresh state, not continue the retired request's stream"
        )

        // Cross-check against a brand-new sampler (defines "fresh").
        let fresh = CBv2DefaultSampler(fallbackSeed: 3)
        XCTAssertEqual(draw(fresh, steps: 3), firstRun)
    }

    /// Retiring an id that is NOT part of the configured fingerprint must
    /// not disturb the current batch's incremental state.
    func testUnrelatedFinishDoesNotResetConfiguredRows() {
        let requestID = CBv2RequestID(7)
        let samplingRow = row(id: 7, temperature: 1.2, seed: 11, prompt: [2])
        var rng = TestRNG(99)
        let logits = batchArray([rng.logits(vocab)])
        eval(logits)

        func draw(_ sampler: CBv2DefaultSampler, steps: Int) -> [Int32] {
            (0 ..< steps).map { _ in
                sampler.sample(
                    logits: logits, params: [samplingRow.params], requestIDs: [requestID],
                    stepIndex: 0, pendingSampledTokens: nil,
                    rowContext: { [samplingRow] }
                ).item(Int32.self)
            }
        }

        // Reference: six uninterrupted draws.
        let reference = draw(CBv2DefaultSampler(fallbackSeed: 5), steps: 6)

        // Same draws with an UNRELATED id retiring mid-stream.
        let sampler = CBv2DefaultSampler(fallbackSeed: 5)
        var interrupted = draw(sampler, steps: 3)
        sampler.requestDidFinish(CBv2RequestID(999))
        interrupted.append(contentsOf: draw(sampler, steps: 3))
        XCTAssertEqual(interrupted, reference)
    }
}

// MARK: - DetokenizerV2

/// Byte-table tokenizer stub: each token id maps to raw UTF-8 bytes, so
/// multi-byte characters can be split across tokens exactly like a
/// byte-fallback BPE.
private struct ByteStubTokenizer: Tokenizer {
    let table: [Int: [UInt8]]

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        var bytes = [UInt8]()
        for id in tokenIds { bytes.append(contentsOf: table[id] ?? []) }
        return String(decoding: bytes, as: UTF8.self)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
}

final class CBv2SamplingDetokenizerTests: XCTestCase {

    func testEmojiSplitAcrossTwoTokensNeverEmitsReplacementChar() {
        // "🦄" = F0 9F A6 84, split 2+2.
        let tokenizer = ByteStubTokenizer(table: [
            0: Array("Hi".utf8), 1: [0xF0, 0x9F], 2: [0xA6, 0x84], 3: Array("!".utf8),
        ])
        let detok = DetokenizerV2(tokenizer: tokenizer)
        var chunks = [String]()
        for token in [0, 1, 2, 3] { chunks.append(detok.append(token)) }
        chunks.append(detok.flush())

        for chunk in chunks {
            XCTAssertFalse(chunk.contains("\u{fffd}"), "mid-stream replacement char in \(chunks)")
        }
        XCTAssertEqual(chunks.joined(), "Hi🦄!")
        XCTAssertEqual(chunks[1], "", "incomplete emoji bytes must be held back")
        XCTAssertEqual(chunks[2], "🦄")
    }

    func testCJKSplitAcrossTokens() {
        // "中" = E4 B8 AD split 1+2; "文" = E6 96 87 split 2+1.
        let tokenizer = ByteStubTokenizer(table: [
            0: [0xE4], 1: [0xB8, 0xAD], 2: [0xE6, 0x96], 3: [0x87],
        ])
        let detok = DetokenizerV2(tokenizer: tokenizer)
        var text = ""
        for token in [0, 1, 2, 3] {
            let chunk = detok.append(token)
            XCTAssertFalse(chunk.contains("\u{fffd}"))
            text += chunk
        }
        text += detok.flush()
        XCTAssertEqual(text, "中文")
    }

    func testZWJEmojiSequenceByteExact() {
        // "👩‍👧" = 👩 (F0 9F 91 A9) + ZWJ (E2 80 8D) + 👧 (F0 9F 91 A7),
        // split at hostile boundaries.
        let tokenizer = ByteStubTokenizer(table: [
            0: [0xF0, 0x9F, 0x91, 0xA9, 0xE2],
            1: [0x80, 0x8D, 0xF0],
            2: [0x9F, 0x91, 0xA7],
        ])
        let detok = DetokenizerV2(tokenizer: tokenizer)
        var text = ""
        for token in [0, 1, 2] {
            let chunk = detok.append(token)
            XCTAssertFalse(chunk.contains("\u{fffd}"))
            text += chunk
        }
        text += detok.flush()
        XCTAssertEqual(Array(text.utf8), Array("👩‍👧".utf8), "byte-exact final text")
    }

    func testSegmentRestartOnNewlinePreservesText() {
        let tokenizer = ByteStubTokenizer(table: [
            0: Array("line one\n".utf8), 1: Array("line ".utf8), 2: Array("two\n".utf8),
            3: Array("end".utf8),
        ])
        let detok = DetokenizerV2(tokenizer: tokenizer)
        var text = ""
        for token in [0, 1, 2, 3, 1, 0] { text += detok.append(token) }
        text += detok.flush()
        XCTAssertEqual(text, "line one\nline two\nendline line one\n")
    }

    func testLongGenerationBoundedSegmentsByteExact() {
        // Force many segment restarts via maxSegmentTokens and verify the
        // concatenation is still byte-exact.
        let tokenizer = ByteStubTokenizer(table: [
            0: Array("ab".utf8), 1: [0xE4], 2: [0xB8, 0xAD], 3: Array(" ".utf8),
        ])
        let detok = DetokenizerV2(tokenizer: tokenizer, maxSegmentTokens: 4)
        let stream = Array(repeating: [0, 1, 2, 3], count: 20).flatMap { $0 }
        var text = ""
        for token in stream {
            let chunk = detok.append(token)
            XCTAssertFalse(chunk.contains("\u{fffd}"))
            text += chunk
        }
        text += detok.flush()
        XCTAssertEqual(text, String(repeating: "ab中 ", count: 20))
    }

    /// PR#62 review: reaching `maxSegmentTokens` on a token that leaves an
    /// incomplete UTF-8 sequence held back must NOT restart the segment —
    /// a restart there would record the last token's U+FFFD decode as
    /// already emitted and the completing token's real bytes would never be
    /// released. The cap is soft: the restart defers until the holdback
    /// clears.
    func testMultiByteCharStraddlingSegmentLimitEmitsNoReplacementChar() {
        // "€" = E2 82 AC split 2+1; the incomplete [E2 82] token arrives
        // exactly at maxSegmentTokens (2, the floor).
        let tokenizer = ByteStubTokenizer(table: [
            0: Array("a".utf8), 1: [0xE2, 0x82], 2: [0xAC], 3: Array("b".utf8),
        ])
        let detok = DetokenizerV2(tokenizer: tokenizer, maxSegmentTokens: 2)
        var chunks = [String]()
        for token in [0, 1, 2, 3] { chunks.append(detok.append(token)) }
        chunks.append(detok.flush())

        for chunk in chunks {
            XCTAssertFalse(
                chunk.contains("\u{fffd}"),
                "segment-cap restart leaked a replacement char: \(chunks)")
        }
        XCTAssertEqual(chunks[1], "", "incomplete € bytes held back across the segment cap")
        XCTAssertEqual(chunks[2], "€", "the completing token must release the character")
        XCTAssertEqual(chunks.joined(), "a€b", "byte-exact despite the straddling restart")
    }

    func testGenuineReplacementCharSurvivesToFlush() {
        // A token whose bytes ARE a replacement char (EF BF BD) is held
        // (indistinguishable from an incomplete sequence) and released once
        // followed by more text or at flush.
        let tokenizer = ByteStubTokenizer(table: [
            0: Array("x".utf8), 1: [0xEF, 0xBF, 0xBD],
        ])
        let detok = DetokenizerV2(tokenizer: tokenizer)
        XCTAssertEqual(detok.append(0), "x")
        XCTAssertEqual(detok.append(1), "", "trailing U+FFFD is ambiguous — held")
        XCTAssertEqual(detok.flush(), "\u{fffd}", "released at end of stream")
    }

    func testIncompleteTrailingBytesFlushAsReplacement() {
        let tokenizer = ByteStubTokenizer(table: [0: Array("ok ".utf8), 1: [0xF0, 0x9F]])
        let detok = DetokenizerV2(tokenizer: tokenizer)
        var text = detok.append(0)
        text += detok.append(1)
        XCTAssertEqual(text, "ok ", "incomplete bytes never emitted mid-stream")
        XCTAssertEqual(detok.flush(), "\u{fffd}")
    }
}

// MARK: - StopHoldback

final class CBv2SamplingStopHoldbackTests: XCTestCase {

    func testStopStringSplitAcrossThreeChunks() {
        let stop = StopHoldback(stopStrings: ["</s>"])
        let first = stop.ingest("abc<")
        XCTAssertEqual(first, CBv2StopScan(text: "abc", stopped: false))
        let second = stop.ingest("/s")
        XCTAssertEqual(second, CBv2StopScan(text: "", stopped: false))
        let third = stop.ingest(">xyz")
        XCTAssertEqual(third, CBv2StopScan(text: "", stopped: true))
        // Nothing after stop is ever emitted.
        XCTAssertEqual(stop.ingest("more"), CBv2StopScan(text: "", stopped: true))
        XCTAssertEqual(stop.flush(), "")
    }

    func testOverlappingStopCandidates() {
        let stop = StopHoldback(stopStrings: ["</s>", "</section>"])
        XCTAssertEqual(stop.ingest("body"), CBv2StopScan(text: "body", stopped: false))
        XCTAssertEqual(stop.ingest("</se"), CBv2StopScan(text: "", stopped: false))
        XCTAssertEqual(stop.ingest("ction"), CBv2StopScan(text: "", stopped: false))
        XCTAssertEqual(stop.ingest(">tail"), CBv2StopScan(text: "", stopped: true))
    }

    func testAmbiguousPrefixReleasedWhenDisambiguated() {
        let stop = StopHoldback(stopStrings: ["END"])
        XCTAssertEqual(stop.ingest("THE EN"), CBv2StopScan(text: "THE ", stopped: false))
        // "GINE" disambiguates "EN" (ENGINE ≠ END); trailing "E" is again a
        // possible prefix of "END" so it stays held.
        XCTAssertEqual(stop.ingest("GINE"), CBv2StopScan(text: "ENGIN", stopped: false))
        XCTAssertEqual(stop.flush(), "E")
    }

    func testStopInMiddleOfSingleChunkTruncatesAtMatchStart() {
        let stop = StopHoldback(stopStrings: ["</s>"])
        XCTAssertEqual(stop.ingest("hello</s>world"), CBv2StopScan(text: "hello", stopped: true))
    }

    func testEarliestOfMultipleStopsWins() {
        let stop = StopHoldback(stopStrings: ["ZZ", "world"])
        XCTAssertEqual(
            stop.ingest("say world then ZZ"),
            CBv2StopScan(text: "say ", stopped: true))
    }

    func testFlushReleasesHeldTailOnNaturalFinish() {
        let stop = StopHoldback(stopStrings: ["<|eot|>"])
        XCTAssertEqual(stop.ingest("answer <|eo"), CBv2StopScan(text: "answer ", stopped: false))
        XCTAssertEqual(stop.flush(), "<|eo", "held text released when no stop can complete")
    }

    func testPassthroughWithoutStopStrings() {
        let stop = StopHoldback(stopStrings: [])
        XCTAssertTrue(stop.isPassthrough)
        XCTAssertEqual(stop.ingest("anything"), CBv2StopScan(text: "anything", stopped: false))
    }

    func testStopSpanningDetokenizerChunks() {
        // End-to-end with DetokenizerV2: stop "STOP" arrives split across
        // token boundaries; user-visible text must end before the stop.
        let tokenizer = ByteStubTokenizer(table: [
            0: Array("result: ".utf8), 1: Array("42 ST".utf8), 2: Array("OP tail".utf8),
        ])
        let detok = DetokenizerV2(tokenizer: tokenizer)
        let stop = StopHoldback(stopStrings: ["STOP"])
        var text = ""
        var stopped = false
        for token in [0, 1, 2] {
            let scan = stop.ingest(detok.append(token))
            text += scan.text
            if scan.stopped {
                stopped = true
                break
            }
        }
        XCTAssertTrue(stopped)
        XCTAssertEqual(text, "result: 42 ")
    }
}
