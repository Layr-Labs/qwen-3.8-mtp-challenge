// CBv2InvarianceTests.swift — WS-G: THE core batch-composition invariance
// suite (report 12 item 5, plan §6 "Correctness harness").
//
// Contract under test: a request's greedy output is TOKEN-EXACT identical no
// matter what batchmates it runs with — alone, alongside 1/2/3 neighbors,
// joining mid-stream, or through a churn storm of joins/leaves. This is the
// property the v1 left-padded engine could not guarantee (scalar RoPE
// offsets, shared `_idx`, batch-wide trims) and the property that makes the
// v2 per-request-KV design correct by construction.
//
// All tests run on `TinyTestModel` (seeded random weights, no downloads)
// through the v2 path (per-request KV + per-row attention). At integration
// the same suites re-point at the real WS-A/WS-B implementations.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2InvarianceTests: XCTestCase {

    /// ≥64 decode steps per the spec.
    private static let soloSteps = 64

    // Shared fixture: one model per class (weights deterministic via seed).
    // XCTest runs the class's tests serially; `nonisolated(unsafe)` matches
    // that execution model.
    private nonisolated(unsafe) static var model: TinyTestModel!
    private nonisolated(unsafe) static var subjectPrompt: [Int] = []
    private nonisolated(unsafe) static var soloOutput: [Int] = []

    override class func setUp() {
        super.setUp()
        model = TinyTestModel.make(seed: 0xC0FFEE)
        // Subject prompt is longer than the sliding window (16) so windowed
        // layers are exercised from the first decode step, and 64 decode
        // steps cross the window repeatedly.
        subjectPrompt = makePromptTokens(length: 24, seed: 1)
        soloOutput = try! CBv2HarnessEngine.runSolo(
            model: model, prompt: subjectPrompt, maxTokens: soloSteps)
    }

    override class func tearDown() {
        model = nil
        super.tearDown()
    }

    private var model: TinyTestModel { Self.model }

    // MARK: - Solo == batched with N neighbors

    /// Decode the subject alongside `neighborCount` different requests; the
    /// subject's output must equal its solo run token-for-token.
    private func assertSubjectUnaffected(byNeighbors neighborCount: Int) throws {
        let engine = CBv2HarnessEngine(model: model)
        let subject = try engine.join(prompt: Self.subjectPrompt, maxTokens: Self.soloSteps)

        for n in 0 ..< neighborCount {
            // Different lengths (some > window) and different content.
            let length = 5 + n * 11
            try engine.join(
                prompt: makePromptTokens(length: length, seed: 100 + UInt64(n)),
                maxTokens: Self.soloSteps)
        }

        while !(engine.active.first(where: { $0.id == subject })?.finished ?? true) {
            engine.decodeStep()
        }

        XCTAssertEqual(
            engine.generated(for: subject), Self.soloOutput,
            "subject output changed with \(neighborCount) neighbor(s) in the batch")
    }

    func testSoloEqualsBatchWithOneNeighbor() throws {
        try assertSubjectUnaffected(byNeighbors: 1)
    }

    func testSoloEqualsBatchWithTwoNeighbors() throws {
        try assertSubjectUnaffected(byNeighbors: 2)
    }

    func testSoloEqualsBatchWithThreeNeighbors() throws {
        try assertSubjectUnaffected(byNeighbors: 3)
    }

    // MARK: - Mid-stream join, neighbors finishing around the subject

    /// The subject joins while two neighbors are mid-decode; one neighbor
    /// finishes early (leaves), another joins later, and the first-joined
    /// neighbors all finish before the subject. Subject must be unaffected.
    func testJoinMidStreamWithNeighborsFinishingAround() throws {
        let engine = CBv2HarnessEngine(model: model)

        let n0 = try engine.join(
            prompt: makePromptTokens(length: 7, seed: 200), maxTokens: 12)
        let n1 = try engine.join(
            prompt: makePromptTokens(length: 30, seed: 201), maxTokens: 40)

        // Neighbors decode for a while before the subject arrives.
        for _ in 0 ..< 8 { engine.decodeStep() }

        let subject = try engine.join(prompt: Self.subjectPrompt, maxTokens: Self.soloSteps)

        // n0 finishes after 4 more steps (12 total) and leaves; n2 joins.
        for _ in 0 ..< 4 { engine.decodeStep() }
        XCTAssertTrue(engine.finishedIDs.contains(n0), "n0 should have hit maxTokens")
        engine.remove(id: n0)

        let n2 = try engine.join(
            prompt: makePromptTokens(length: 18, seed: 202), maxTokens: 20)

        // Run until every neighbor has finished and left, subject still going.
        while !(engine.active.first(where: { $0.id == n1 })?.finished ?? true) {
            engine.decodeStep()
        }
        engine.remove(id: n1)
        while !(engine.active.first(where: { $0.id == n2 })?.finished ?? true) {
            engine.decodeStep()
        }
        engine.remove(id: n2)

        // Subject finishes (possibly alone again).
        while !(engine.active.first(where: { $0.id == subject })?.finished ?? true) {
            engine.decodeStep()
        }

        XCTAssertEqual(
            engine.generated(for: subject), Self.soloOutput,
            "mid-stream join with churn around the subject changed its output")
    }

    // MARK: - Sliding-window crossing during the run

    /// Same invariance with prompts and runs specifically sized so windowed
    /// layers cross their eviction boundary mid-decode for every request.
    func testWindowCrossingUnderBatching() throws {
        // Short prompt (< window): the window boundary is crossed DURING
        // decode, not during prefill.
        let prompt = makePromptTokens(length: 6, seed: 300)
        let steps = 3 * TinyTestModelConfig().windowSize  // cross 3× over

        let solo = try CBv2HarnessEngine.runSolo(model: model, prompt: prompt, maxTokens: steps)

        let engine = CBv2HarnessEngine(model: model)
        let subject = try engine.join(prompt: prompt, maxTokens: steps)
        // Neighbors on both sides of their own window boundaries.
        try engine.join(prompt: makePromptTokens(length: 40, seed: 301), maxTokens: steps)
        try engine.join(prompt: makePromptTokens(length: 15, seed: 302), maxTokens: steps)

        while !(engine.active.first(where: { $0.id == subject })?.finished ?? true) {
            engine.decodeStep()
        }

        XCTAssertEqual(
            engine.generated(for: subject), solo,
            "window crossing under batching diverged from the solo run")
    }

    // MARK: - Sinks (GPT-OSS shape) under batching

    /// Same core invariance on the sink-enabled fixture: per-head learned
    /// sinks are folded into every row's softmax identically regardless of
    /// batch composition.
    func testInvarianceWithAttentionSinks() throws {
        let sinkModel = TinyTestModel.make(seed: 0xC0FFEE, withSinks: true)
        let prompt = makePromptTokens(length: 24, seed: 400)

        let solo = try CBv2HarnessEngine.runSolo(model: sinkModel, prompt: prompt, maxTokens: 48)

        let engine = CBv2HarnessEngine(model: sinkModel)
        let subject = try engine.join(prompt: prompt, maxTokens: 48)
        try engine.join(prompt: makePromptTokens(length: 9, seed: 401), maxTokens: 48)
        try engine.join(prompt: makePromptTokens(length: 33, seed: 402), maxTokens: 48)

        while !(engine.active.first(where: { $0.id == subject })?.finished ?? true) {
            engine.decodeStep()
        }

        XCTAssertEqual(
            engine.generated(for: subject), solo,
            "sink model output changed under batching")
    }

    // MARK: - Stochastic sampling with per-request seeds (WS-E)

    /// The contract keys the RNG on (seed, requestID, per-request step), so
    /// a SEEDED stochastic request must reproduce token-exactly under
    /// batching. Runs the REAL EngineV2 with the production
    /// `CBv2DefaultSampler` (LogitsPipelineV2 + SamplerV2): once solo, once
    /// with two stochastic neighbors — the subject keeps the same request
    /// id and seed, so its keyed noise stream is identical in both runs.
    func testPerRequestSeedInvarianceUnderBatching() async throws {
        let prompt = makePromptTokens(length: 12, seed: 500)
        let sampling = CBv2SamplingParams(temperature: 0.8, topP: 0.9, seed: 42)
        let maxTokens = 24

        func run(withNeighbors: Bool) async throws -> [Int] {
            let engine = EngineV2(
                model: model,
                layerKinds: model.layerKinds,
                backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
                cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
                sampler: CBv2DefaultSampler(fallbackSeed: 7),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: 4, prefillChunkSize: 8))
            let subject = CBv2Request(
                id: CBv2RequestID(1), promptTokens: prompt, sampling: sampling,
                maxTokens: maxTokens)
            var neighborStreams: [AsyncStream<CBv2Event>] = []
            let subjectStream = try engine.submit(subject)
            if withNeighbors {
                for n: UInt64 in 2 ... 3 {
                    let neighbor = CBv2Request(
                        id: CBv2RequestID(n),
                        promptTokens: makePromptTokens(length: 6 + Int(n) * 9, seed: 500 + n),
                        sampling: CBv2SamplingParams(
                            temperature: 0.7, topK: 20, seed: 1000 + n),
                        maxTokens: maxTokens)
                    neighborStreams.append(try engine.submit(neighbor))
                }
            }
            let collected = await cbv2SchedCollect(subjectStream)
            for stream in neighborStreams {
                _ = await cbv2SchedCollect(stream)
            }
            await engine.shutdown()
            XCTAssertEqual(collected.finishReason, .length)
            XCTAssertEqual(collected.tokens.count, maxTokens)
            return collected.tokens
        }

        let solo = try await run(withNeighbors: false)
        let batched = try await run(withNeighbors: true)
        XCTAssertEqual(
            solo, batched,
            "seeded stochastic request diverged under batching — keyed RNG must be a pure function of (seed, requestID, per-request step)"
        )
    }

    // MARK: - Churn storm

    /// 50 seeded random join/leave events; every request that ever ran must
    /// produce output token-exact vs its own solo run (compared over however
    /// many tokens it actually generated before finish/cancel).
    func testChurnStormEveryRequestMatchesSolo() throws {
        var rng = CBv2HarnessRNG(seed: 0xD00D)
        let engine = CBv2HarnessEngine(model: model)
        let maxConcurrent = 4

        struct Record {
            let prompt: [Int]
            let maxTokens: Int
            var output: [Int] = []
        }
        var records: [UInt64: Record] = [:]
        var liveIDs: [CBv2RequestID] = []

        func joinRandom() throws {
            let prompt = makePromptTokens(
                length: Int.random(in: 3 ... 28, using: &rng),
                seed: rng.next())
            let maxTokens = Int.random(in: 4 ... 24, using: &rng)
            let id = try engine.join(prompt: prompt, maxTokens: maxTokens)
            records[id.raw] = Record(prompt: prompt, maxTokens: maxTokens)
            liveIDs.append(id)
        }

        func leave(_ id: CBv2RequestID) {
            // Capture what the request generated (finish OR early cancel).
            records[id.raw]?.output = engine.generated(for: id) ?? []
            engine.remove(id: id)
            liveIDs.removeAll { $0 == id }
        }

        var events = 0
        while events < 50 {
            let wantJoin =
                liveIDs.isEmpty
                || (liveIDs.count < maxConcurrent && Bool.random(using: &rng))
            if wantJoin {
                try joinRandom()
            } else {
                // Random leave: prefer finished rows, otherwise cancel one.
                let id = engine.finishedIDs.first ?? liveIDs.randomElement(using: &rng)!
                leave(id)
            }
            events += 1

            // A few decode steps between events; harvest natural finishes so
            // finished rows don't keep decoding forever.
            if !liveIDs.isEmpty {
                for _ in 0 ..< Int.random(in: 1 ... 4, using: &rng) {
                    guard !engine.active.isEmpty else { break }
                    engine.decodeStep()
                    for id in engine.finishedIDs { leave(id) }
                }
            }
        }

        // Drain the stragglers.
        while !engine.active.isEmpty {
            engine.decodeStep()
            for id in engine.finishedIDs { leave(id) }
        }

        XCTAssertGreaterThanOrEqual(records.count, 20, "storm should admit many requests")

        // Every request must match the prefix of its solo run.
        for (raw, record) in records.sorted(by: { $0.key < $1.key }) {
            guard !record.output.isEmpty else { continue }
            let solo = try CBv2HarnessEngine.runSolo(
                model: model, prompt: record.prompt, maxTokens: record.maxTokens)
            XCTAssertEqual(
                record.output, Array(solo.prefix(record.output.count)),
                "request cbv2-\(raw) diverged from its solo run during the churn storm")
        }
    }
}
