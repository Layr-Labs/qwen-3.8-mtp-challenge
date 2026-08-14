// CBv2EndToEndTests.swift — integration: the REAL engine assembly, end to end.
//
// Constructs the production EngineV2 from the real subsystems — TinyTestModel
// (a real 2-layer transformer conforming to CBv2SteppableModel; layer 0 full
// attention, layer 1 sliding-window(16)), a real KV backend
// (CBv2ContiguousKVBackend or PagedKVBackend), CBv2LayerCacheBank,
// CBv2DefaultSampler (LogitsPipelineV2 + SamplerV2), CBv2TextDetokenizerFactory
// (DetokenizerV2 + StopHoldback), and PrefixCacheV2 — and asserts:
//
//  (i)   3 concurrent streaming requests with mixed prompt lengths finish
//        with correct text and TOKEN-EXACT batch invariance vs solo runs;
//  (ii)  a stop string completed mid-stream truncates the text exactly at
//        the match start (holdback: no text at/past the match ever emitted);
//  (iii) cancel mid-decode frees the slot promptly and batchmates are
//        unaffected (still token-exact vs solo);
//  (iv)  a second submit of the same prompt hits the prefix cache
//        (usage.prefixCacheHitTokens > 0) with IDENTICAL greedy output.
//        The adopted full-attention KV is bit-identical (donated arrays),
//        and the windowed layer is last, so the trailing-window replay
//        recomputes the same values — but the replay's SDPA calls run at
//        different shapes than the cold run's chunks, and cross-shape SDPA
//        determinism is an empirical property of the kernels on this
//        hardware, not a guarantee. Greedy argmax over these logit gaps
//        makes the token-exact assertion stable in practice;
//  (v)   the same suite runs green with the paged backend (fp16 pages,
//        custom Metal decode kernel) substituted for the contiguous one.
//
// No model downloads; weights are seeded random.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - Test tokenizer

/// Deterministic id → text mapping ("<id>") so expected text is a pure
/// function of the token stream; distinct ids can never be substrings of
/// each other's rendering ("<12>" vs "<123>").
private struct CBv2E2ETokenizer: Tokenizer {
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "<\($0)>" }.joined()
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

private func renderedText(_ tokens: [Int]) -> String {
    tokens.map { "<\($0)>" }.joined()
}

// MARK: - Suite

final class CBv2EndToEndTests: XCTestCase {

    private enum BackendKind {
        case contiguous
        case paged
    }

    private struct Stack {
        let engine: EngineV2
        let model: TinyTestModel
        let prefixCache: PrefixCacheV2
    }

    /// Window of the fixture model (both head-dim variants).
    private let window = TinyTestModelConfig().windowSize  // 16
    /// Small hash blocks so short prompts produce multi-block hits.
    private let blockSize = 8

    /// One model per backend kind per test (seeded ⇒ identical weights for
    /// every stack built from it, so solo and batched runs share weights).
    private func makeModel(_ kind: BackendKind) -> TinyTestModel {
        switch kind {
        case .contiguous: return TinyTestModel.make(seed: 0xC0FFEE)
        // The paged Metal kernel supports headDim ∈ {64,128,256,512}.
        case .paged: return TinyTestModel.make(seed: 0xC0FFEE, headDim: 64)
        }
    }

    private func makeStack(
        _ kind: BackendKind, model: TinyTestModel, enablePrefixCache: Bool = true
    ) throws -> Stack {
        let backend: CBv2KVBackend
        let bank: CBv2LayerCacheBank
        var materializeOnDonate = false
        switch kind {
        case .contiguous:
            backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28))
            bank = CBv2LayerCacheBank(layerKinds: model.layerKinds)
        case .paged:
            let paged: PagedKVBackend
            do {
                paged = try PagedKVBackend(
                    layerKinds: model.layerKinds,
                    config: PagedKVPoolConfig(
                        capacityBytes: 64 << 20,
                        maxPrefillChunk: 64,
                        nominalMaxSequenceLength: 512))
            } catch let error as CBv2KVError {
                throw XCTSkip("paged backend unavailable on this hardware: \(error)")
            }
            backend = paged
            bank = CBv2LayerCacheBank(caches: paged.makeLayerCaches())
            // Donated views reference the live fp16 slabs; pages are
            // recycled after release, so donations must materialize.
            materializeOnDonate = true
        }
        let prefixCache = PrefixCacheV2(
            config: .init(
                blockSize: blockSize, modelName: "cbv2-e2e",
                materializeOnDonate: materializeOnDonate))
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: backend,
            cacheProvider: bank,
            sampler: CBv2DefaultSampler(fallbackSeed: 11),
            detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: CBv2E2ETokenizer()),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16,
                enablePrefixCache: enablePrefixCache),
            prefixCache: prefixCache)
        return Stack(engine: engine, model: model, prefixCache: prefixCache)
    }

    private func greedyRequest(
        id: UInt64, prompt: [Int], maxTokens: Int, stopStrings: [String] = []
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0),
            maxTokens: maxTokens, stopStrings: stopStrings)
    }

    /// Run one greedy request to completion on a FRESH engine (the solo
    /// baseline) and return what it produced.
    private func soloRun(
        _ kind: BackendKind, model: TinyTestModel, prompt: [Int], maxTokens: Int
    ) async throws -> CBv2SchedCollected {
        let stack = try makeStack(kind, model: model, enablePrefixCache: false)
        let collected = await cbv2SchedCollect(
            try stack.engine.submit(
                greedyRequest(id: 900, prompt: prompt, maxTokens: maxTokens)))
        await stack.engine.shutdown()
        return collected
    }

    // MARK: (i) Concurrent streaming — text correctness + batch invariance

    private func runConcurrentStreamingInvariance(_ kind: BackendKind) async throws {
        let model = makeModel(kind)
        // Mixed prompt lengths: shorter than the window, spanning it, and
        // several prefill chunks long.
        let prompts = [
            makePromptTokens(length: 5, seed: 11),
            makePromptTokens(length: 24, seed: 22),
            makePromptTokens(length: 41, seed: 33),
        ]
        let maxTokens = [32, 24, 40]

        var solos: [CBv2SchedCollected] = []
        for (prompt, budget) in zip(prompts, maxTokens) {
            solos.append(
                try await soloRun(kind, model: model, prompt: prompt, maxTokens: budget))
        }

        let stack = try makeStack(kind, model: model, enablePrefixCache: false)
        var streams: [AsyncStream<CBv2Event>] = []
        for (i, prompt) in prompts.enumerated() {
            streams.append(
                try stack.engine.submit(
                    greedyRequest(id: UInt64(i + 1), prompt: prompt, maxTokens: maxTokens[i])))
        }
        var batched: [CBv2SchedCollected] = []
        for stream in streams {
            batched.append(await cbv2SchedCollect(stream))
        }
        await stack.engine.shutdown()

        for i in prompts.indices {
            XCTAssertEqual(batched[i].finishReason, .length, "request \(i)")
            XCTAssertEqual(batched[i].tokens.count, maxTokens[i], "request \(i)")
            XCTAssertEqual(
                batched[i].tokens, solos[i].tokens,
                "request \(i) diverged from its solo run under batching")
            XCTAssertEqual(
                batched[i].text, renderedText(batched[i].tokens),
                "request \(i): streamed text must be the exact rendering of its tokens")
            XCTAssertEqual(batched[i].usage?.promptTokens, prompts[i].count, "request \(i)")
            XCTAssertEqual(batched[i].usage?.completionTokens, maxTokens[i], "request \(i)")
        }
        XCTAssertGreaterThan(
            stack.engine.capacity().stepsExecuted, 0, "step counter must be published")
    }

    func testConcurrentStreamingInvariance_Contiguous() async throws {
        try await runConcurrentStreamingInvariance(.contiguous)
    }

    func testConcurrentStreamingInvariance_Paged() async throws {
        try await runConcurrentStreamingInvariance(.paged)
    }

    // MARK: (ii) Stop string mid-stream

    private func runStopStringTruncation(_ kind: BackendKind) async throws {
        let model = makeModel(kind)
        let prompt = makePromptTokens(length: 12, seed: 44)
        let budget = 48

        // Baseline (no stop) discovers the greedy continuation, from which
        // a mid-stream stop string is derived.
        let baseline = try await soloRun(kind, model: model, prompt: prompt, maxTokens: budget)
        XCTAssertEqual(baseline.tokens.count, budget)
        let stopToken = baseline.tokens[budget / 2]
        let stopString = renderedText([stopToken])
        let firstHit = baseline.tokens.firstIndex(of: stopToken)!
        let expectedText = renderedText(Array(baseline.tokens[..<firstHit]))

        let stack = try makeStack(kind, model: model, enablePrefixCache: false)
        let collected = await cbv2SchedCollect(
            try stack.engine.submit(
                greedyRequest(
                    id: 1, prompt: prompt, maxTokens: budget, stopStrings: [stopString])))
        await stack.engine.shutdown()

        XCTAssertEqual(collected.finishReason, .stop)
        XCTAssertEqual(
            collected.text, expectedText,
            "stop-string truncation must cut EXACTLY at the match start")
        XCTAssertFalse(
            collected.text.contains(stopString),
            "no text at or past the stop match may ever be emitted")
    }

    func testStopStringTruncation_Contiguous() async throws {
        try await runStopStringTruncation(.contiguous)
    }

    func testStopStringTruncation_Paged() async throws {
        try await runStopStringTruncation(.paged)
    }

    // MARK: (ii-b) Stop TOKEN text suppression

    /// Regression: the stop token's rendering used to be emitted in the
    /// final delta before the `.stop` finish. OpenAI behavior excludes it:
    /// `text` must end exactly before the stop token, while the raw delta
    /// `tokens` still carry its id and usage still counts it (see the
    /// CBv2Event.delta contract doc).
    private func runStopTokenTextSuppression(_ kind: BackendKind) async throws {
        let model = makeModel(kind)
        let prompt = makePromptTokens(length: 12, seed: 88)
        let budget = 48

        // Baseline greedy run discovers a token to stop on mid-stream.
        let baseline = try await soloRun(kind, model: model, prompt: prompt, maxTokens: budget)
        XCTAssertEqual(baseline.tokens.count, budget)
        let stopToken = baseline.tokens[budget / 2]
        let firstHit = baseline.tokens.firstIndex(of: stopToken)!

        let stack = try makeStack(kind, model: model, enablePrefixCache: false)
        var request = greedyRequest(id: 1, prompt: prompt, maxTokens: budget)
        request.stopTokens = [stopToken]
        let collected = await cbv2SchedCollect(try stack.engine.submit(request))
        await stack.engine.shutdown()

        XCTAssertEqual(collected.finishReason, .stop)
        // Raw ids include the stop token; text must NOT.
        XCTAssertEqual(collected.tokens, Array(baseline.tokens[...firstHit]))
        XCTAssertEqual(
            collected.text, renderedText(Array(baseline.tokens[..<firstHit])),
            "the stop token's rendering must never be emitted")
        XCTAssertFalse(collected.text.contains(renderedText([stopToken])))
        // Usage still counts the sampled stop token.
        XCTAssertEqual(collected.usage?.completionTokens, firstHit + 1)
    }

    func testStopTokenTextSuppression_Contiguous() async throws {
        try await runStopTokenTextSuppression(.contiguous)
    }

    func testStopTokenTextSuppression_Paged() async throws {
        try await runStopTokenTextSuppression(.paged)
    }

    // MARK: (iii) Cancel mid-decode

    private func runCancelMidDecode(_ kind: BackendKind) async throws {
        let model = makeModel(kind)
        let victimPrompt = makePromptTokens(length: 9, seed: 55)
        let survivorPrompt = makePromptTokens(length: 21, seed: 66)
        let survivorBudget = 40

        let survivorSolo = try await soloRun(
            kind, model: model, prompt: survivorPrompt, maxTokens: survivorBudget)

        let stack = try makeStack(kind, model: model, enablePrefixCache: false)
        let victimID = CBv2RequestID(1)
        let victimStream = try stack.engine.submit(
            greedyRequest(id: 1, prompt: victimPrompt, maxTokens: 400))
        let survivorStream = try stack.engine.submit(
            greedyRequest(id: 2, prompt: survivorPrompt, maxTokens: survivorBudget))

        // Consume a few victim deltas mid-decode, then cancel.
        var victimTokens: [Int] = []
        var victimReason: CBv2FinishReason?
        for await event in victimStream {
            switch event {
            case .delta(_, let tokens, _):
                victimTokens.append(contentsOf: tokens)
                if victimTokens.count == 5 {
                    stack.engine.cancel(victimID)
                }
            case .finished(let reason, _):
                victimReason = reason
            }
        }
        XCTAssertEqual(victimReason, .cancelled)
        XCTAssertLessThan(victimTokens.count, 400, "cancel must stop generation promptly")

        // The batchmate is unaffected: finishes with its solo output.
        let survivor = await cbv2SchedCollect(survivorStream)
        XCTAssertEqual(survivor.finishReason, .length)
        XCTAssertEqual(
            survivor.tokens, survivorSolo.tokens,
            "cancelling a batchmate must not perturb other requests")

        // The victim's slot is freed: capacity drains to zero active.
        let drained = await cbv2SchedWait {
            stack.engine.capacity().activeRequests == 0
        }
        XCTAssertTrue(drained, "cancelled + finished requests must free their slots")
        await stack.engine.shutdown()
    }

    func testCancelMidDecode_Contiguous() async throws {
        try await runCancelMidDecode(.contiguous)
    }

    func testCancelMidDecode_Paged() async throws {
        try await runCancelMidDecode(.paged)
    }

    // MARK: (iv) Prefix-cache hit round trip

    private func runPrefixCacheRoundTrip(_ kind: BackendKind) async throws {
        let model = makeModel(kind)
        // 41-token prompt, blockSize 8: lookup is capped at prompt-1 = 40
        // tokens = 5 whole blocks; window 16 ⇒ recompute 16 ⇒ 24 adopted.
        let prompt = makePromptTokens(length: 41, seed: 77)
        let budget = 8
        let expectedHit = 5 * blockSize - window  // 24

        let stack = try makeStack(kind, model: model, enablePrefixCache: true)

        let first = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 1, prompt: prompt, maxTokens: budget)))
        XCTAssertEqual(first.finishReason, .length)
        XCTAssertEqual(first.usage?.prefixCacheHitTokens, 0, "first run is a miss")

        // Donation is asynchronous (off the engine queue) — wait for it.
        let donated = await cbv2SchedWait {
            stack.prefixCache.stats().entryCount >= 1
        }
        XCTAssertTrue(donated, "finished request must donate its prefix")

        let second = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 2, prompt: prompt, maxTokens: budget)))
        await stack.engine.shutdown()

        XCTAssertEqual(second.finishReason, .length)
        let hit = second.usage?.prefixCacheHitTokens ?? 0
        XCTAssertGreaterThan(hit, 0, "second submit of the same prompt must hit")
        XCTAssertEqual(hit, expectedHit, "hit = whole-block match minus windowed recompute")
        XCTAssertEqual(
            second.tokens, first.tokens,
            "prefix-cache adoption must be token-exact vs the cold run")
        XCTAssertEqual(second.text, first.text)
        XCTAssertGreaterThan(stack.prefixCache.stats().hits, 0)
    }

    func testPrefixCacheRoundTrip_Contiguous() async throws {
        try await runPrefixCacheRoundTrip(.contiguous)
    }

    func testPrefixCacheRoundTrip_Paged() async throws {
        try await runPrefixCacheRoundTrip(.paged)
    }

    // MARK: (iv-c) Stacked-sliding + downstream-full drift (Codex P1)

    /// Regression for the under-replayed sliding-window adoption bound.
    ///
    /// Model shape [full, sliding(16), sliding(16), full]: two STACKED
    /// windowed layers whose first `window-1` replayed outputs are polluted
    /// (their true window reaches below `effective`, where no windowed KV was
    /// replayed), and a DOWNSTREAM full layer that CACHES those polluted
    /// outputs as K/V inside the retained region. The old single-window bound
    /// (`recompute = window`) adopts `effective = matched - 16`, so the full
    /// layer decodes against corrupted KV and drifts. The revised bound
    /// (`windowedCount × window = 32 ≥ matched = 24`) clamps to the whole
    /// match, degrading adoption to full recompute — which is exact. Under
    /// the OLD bound this assertion FAILS; under the fix it holds.
    func testStackedSlidingRecomputeIsTokenExact() async throws {
        // blockSize 8, window 16, prompt 25 ⇒ matched = 3×8 = 24 (capped at
        // prompt-1). windowedCount 2 × window 16 = 32 ≥ 24 ⇒ full recompute.
        let model = TinyTestModel.make(seed: 0xD00D_F00D, stackedSlidingFull: true)
        XCTAssertEqual(model.layerKinds.count, 4, "shape must be [full, sliding, sliding, full]")
        let prompt = makePromptTokens(length: 25, seed: 0x5EED)
        let budget = 16

        let stack = try makeStack(.contiguous, model: model, enablePrefixCache: true)

        let first = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 1, prompt: prompt, maxTokens: budget)))
        XCTAssertEqual(first.finishReason, .length)
        XCTAssertEqual(first.usage?.prefixCacheHitTokens, 0, "first run is a miss")

        // Donation is asynchronous (off the engine queue) — wait for it.
        let donated = await cbv2SchedWait { stack.prefixCache.stats().entryCount >= 1 }
        XCTAssertTrue(donated, "finished request must donate its prefix")

        let second = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 2, prompt: prompt, maxTokens: budget)))
        await stack.engine.shutdown()

        XCTAssertEqual(second.finishReason, .length)
        XCTAssertEqual(
            second.tokens, first.tokens,
            "stacked-sliding adoption must be token-exact vs the cold run "
                + "(the windowed-layer-count recompute bound guarantees it)")
        XCTAssertEqual(second.text, first.text)
    }

    // MARK: (iv-b) Per-request cache salt isolation (TB-007)

    /// Requests carrying different `cacheSalt`s must never share cached KV
    /// (in either direction, nil included), while the same salt round-trips
    /// with a hit — end to end through the real engine, donation queue, and
    /// PrefixCacheV2.
    func testPrefixCacheSaltIsolation_Contiguous() async throws {
        let model = makeModel(.contiguous)
        let prompt = makePromptTokens(length: 41, seed: 99)
        let budget = 8
        let expectedHit = 5 * blockSize - window  // 24, as in the round trip

        let stack = try makeStack(.contiguous, model: model, enablePrefixCache: true)
        func salted(_ id: UInt64, _ salt: String?) -> CBv2Request {
            var request = greedyRequest(id: id, prompt: prompt, maxTokens: budget)
            request.cacheSalt = salt
            return request
        }

        let first = await cbv2SchedCollect(try stack.engine.submit(salted(1, "tenantA")))
        XCTAssertEqual(first.finishReason, .length)
        XCTAssertEqual(first.usage?.prefixCacheHitTokens, 0, "first run is a miss")
        let donated = await cbv2SchedWait { stack.prefixCache.stats().entryCount >= 1 }
        XCTAssertTrue(donated, "finished request must donate under its salt")

        // Different salt and nil salt: no cross-hit, greedy output unchanged.
        let otherTenant = await cbv2SchedCollect(try stack.engine.submit(salted(2, "tenantB")))
        XCTAssertEqual(
            otherTenant.usage?.prefixCacheHitTokens, 0,
            "a different salt must never adopt tenantA's KV")
        XCTAssertEqual(otherTenant.tokens, first.tokens)

        let unsalted = await cbv2SchedCollect(try stack.engine.submit(salted(3, nil)))
        XCTAssertEqual(
            unsalted.usage?.prefixCacheHitTokens, 0,
            "a nil-salt request must never adopt salted KV")
        XCTAssertEqual(unsalted.tokens, first.tokens)

        // Same salt: hits, token-exact.
        let second = await cbv2SchedCollect(try stack.engine.submit(salted(4, "tenantA")))
        await stack.engine.shutdown()
        XCTAssertEqual(second.usage?.prefixCacheHitTokens, expectedHit, "same salt must hit")
        XCTAssertEqual(second.tokens, first.tokens)
    }

    // MARK: (v) Backend / prefix-cache pairing guard

    /// `EngineV2.init` preconditions on `prefixCachePairingViolation`: a
    /// backend whose donated snapshots reference recyclable storage (paged
    /// slabs) must never feed a prefix cache with `materializeOnDonate:
    /// false`. Asserted through the internal validator (a precondition is
    /// not catchable in-process); construction with a safe pairing is also
    /// exercised for real.
    func testPagedBackendRejectsNonMaterializingPrefixCache() throws {
        let model = makeModel(.paged)
        let paged: PagedKVBackend
        do {
            paged = try PagedKVBackend(
                layerKinds: model.layerKinds,
                config: PagedKVPoolConfig(
                    capacityBytes: 64 << 20, maxPrefillChunk: 64,
                    nominalMaxSequenceLength: 512))
        } catch let error as CBv2KVError {
            throw XCTSkip("paged backend unavailable on this hardware: \(error)")
        }
        let contiguous = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 24))
        XCTAssertTrue(paged.requiresMaterializedSnapshots)
        XCTAssertFalse(contiguous.requiresMaterializedSnapshots)

        let unsafeCache = PrefixCacheV2(
            config: .init(blockSize: blockSize, materializeOnDonate: false))
        let safeCache = PrefixCacheV2(config: .init(blockSize: blockSize))  // default: true

        XCTAssertNotNil(
            EngineV2.prefixCachePairingViolation(backend: paged, prefixCache: unsafeCache),
            "paged + materializeOnDonate:false must be rejected at construction")
        XCTAssertNil(
            EngineV2.prefixCachePairingViolation(backend: paged, prefixCache: safeCache))
        XCTAssertNil(EngineV2.prefixCachePairingViolation(backend: paged, prefixCache: nil))
        XCTAssertNil(
            EngineV2.prefixCachePairingViolation(backend: contiguous, prefixCache: unsafeCache),
            "contiguous snapshot views own their buffers via ARC — opt-out stays legal")

        // A safe pairing constructs and shuts down cleanly.
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: paged.makeLayerCaches()),
            sampler: CBv2DefaultSampler(fallbackSeed: 11),
            detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: CBv2E2ETokenizer()),
            schedulerConfig: CBv2SchedulerConfig(enablePrefixCache: true),
            prefixCache: safeCache)
        let drained = expectation(description: "shutdown")
        Task {
            await engine.shutdown()
            drained.fulfill()
        }
        wait(for: [drained], timeout: 10)
    }
}
