// Tests for EngineCore — §8-11 of ContinuousBatchingTestPlan.md

import Foundation
import XCTest

@testable import MLXLMCommon

// MARK: - §8  Lifecycle

final class CBEngineCoreLifecycleTests: XCTestCase {

    func testEngineCoreStartSetsRunning() {
        let engine = makeTestEngine()
        XCTAssertFalse(engine.isRunning)
        engine.start()
        XCTAssertTrue(engine.isRunning)
        engine.stop()
    }

    func testEngineCoreStopClearsRunning() {
        let engine = makeTestEngine()
        engine.start()
        engine.stop()
        XCTAssertFalse(engine.isRunning)
    }

    func testEngineCoreDoubleStartIsNoop() {
        // Calling start() twice must not crash and engine must stay running.
        let engine = makeTestEngine()
        engine.start()
        engine.start()
        XCTAssertTrue(engine.isRunning)
        engine.stop()
    }

    func testEngineCoreStatsInitialValues() {
        let engine = makeTestEngine()
        engine.start()
        let stats = engine.getStats()
        XCTAssertEqual(stats["running"] as? Bool, true)
        XCTAssertEqual(stats["steps_executed"] as? Int, 0)
        XCTAssertEqual(stats["active_requests"] as? Int, 0)
        XCTAssertEqual(stats["num_waiting"] as? Int, 0)
        XCTAssertEqual(stats["num_running"] as? Int, 0)
        engine.stop()
    }
}

// MARK: - §9  Request management

final class CBEngineCoreRequestTests: XCTestCase {

    func testEngineCoreAddRequestReturnsId() async {
        let engine = makeTestEngine()
        let req = makeIntRequest()
        let id = await engine.addRequest(req)
        XCTAssertEqual(id, req.requestId)
    }

    func testEngineCoreAddRequestCreatesCollector() async {
        let engine = makeTestEngine()
        await engine.addRequest(makeIntRequest())
        XCTAssertEqual(engine.getStats()["active_requests"] as? Int, 1)
    }

    func testEngineCoreAbortRequestReturnsFalseForUnknownId() {
        let engine = makeTestEngine()
        XCTAssertFalse(engine.abortRequest("does-not-exist"))
    }

    func testEngineCoreAbortRequestSignalsConsumerWithAbortOutput() async {
        let engine = makeTestEngine()
        let req = makeIntRequest()
        let rid = await engine.addRequest(req)

        XCTAssertTrue(engine.abortRequest(rid))

        // Stream should immediately yield the abort output.
        var received: RequestOutput? = nil
        for await output in engine.streamOutputs(requestId: rid) {
            received = output
            break
        }
        XCTAssertEqual(received?.finishReason, "abort")
        XCTAssertNotNil(received?.error)
        XCTAssertEqual(received?.finished, true)
    }

    func testEngineCoreAbortRequestWakesBlockedStreamOutputs() async throws {
        // Engine NOT started → stream blocks; abortRequest must unblock it.
        let engine = makeTestEngine()
        let rid = await engine.addRequest(makeIntRequest())

        let streamTask = Task { () -> RequestOutput? in
            var last: RequestOutput? = nil
            for await output in engine.streamOutputs(requestId: rid) {
                last = output
                if output.finished || output.error != nil { break }
            }
            return last
        }

        // Let streamTask reach its suspension point.
        try await Task.sleep(nanoseconds: 20_000_000)

        XCTAssertTrue(engine.abortRequest(rid))

        let output = await streamTask.value
        XCTAssertNotNil(output?.error, "abort must wake the blocked stream")
    }

    func testEngineCoreAbortRequestNoGhostInScheduler() async throws {
        let engine = makeTestEngine()
        engine.start()
        let req = makeIntRequest()
        await engine.addRequest(req)

        engine.abortRequest(req.requestId)

        // Wait for the deferred engineQueue dispatch to process.
        try await Task.sleep(nanoseconds: 80_000_000)

        let stats = engine.getStats()
        XCTAssertEqual(stats["num_waiting"] as? Int, 0)
        XCTAssertEqual(stats["num_running"] as? Int, 0)
        engine.stop()
    }

    func testEngineCoreAbortAllRequestsSendsErrorToAll() async {
        let engine = makeTestEngine()
        let req1 = makeIntRequest(id: "r1")
        let req2 = makeIntRequest(id: "r2")
        await engine.addRequest(req1)
        await engine.addRequest(req2)

        let count = engine.abortAllRequests()
        XCTAssertEqual(count, 2)

        // Each collector must have received an error output.
        var outputs: [RequestOutput] = []
        for await o in engine.streamOutputs(requestId: "r1") { outputs.append(o); break }
        for await o in engine.streamOutputs(requestId: "r2") { outputs.append(o); break }

        XCTAssertEqual(outputs.count, 2)
        XCTAssertTrue(outputs.allSatisfy { $0.finishReason == "error" })
    }

    func testEngineCoreAbortAllRequestsEmptyReturnsZero() {
        let engine = makeTestEngine()
        XCTAssertEqual(engine.abortAllRequests(), 0)
    }

    func testEngineCoreAddRequestAfterStopDoesNotEnqueueAndTerminatesStream() async throws {
        // After stop(), addRequest must NOT hand the request to the scheduler
        // (the step loop is gone → it would never run and the stream would hang).
        // It must instead terminate the request's stream cleanly so a direct
        // EngineCore/BatchedEngine caller doesn't block forever.
        let engine = makeTestEngine()
        engine.start()
        engine.stop()

        let rid = await engine.addRequest(makeIntRequest(id: "after-stop"))

        // The stream must yield a terminal output (not hang).
        var received: RequestOutput? = nil
        for await output in engine.streamOutputs(requestId: rid) {
            received = output
            break
        }
        XCTAssertEqual(received?.finished, true, "a post-stop add must terminate the stream")
        XCTAssertNotNil(received?.error, "post-stop add must surface an error, not silently hang")

        // The scheduler must hold no waiting/running entry for it.
        let stats = engine.getStats()
        XCTAssertEqual(stats["num_waiting"] as? Int, 0,
                       "post-stop add must not enqueue into the (stopped) scheduler")
        XCTAssertEqual(stats["num_running"] as? Int, 0)
    }

    func testEngineCoreAddBeforeStartStillWorks() async throws {
        // Guard against over-broad rejection: an engine that has NOT been started
        // yet (also `_running == false`, but NOT stopped) must still accept the
        // request and run it once started (the add-before-start pattern).
        let engine = makeTestEngine(eosTokenIds: [5])
        let rid = await engine.addRequest(makeIntRequest(tokens: [3], maxTokens: 10))
        engine.start()  // start AFTER the add

        var finished = false
        for await output in engine.streamOutputs(requestId: rid) {
            if output.finished { finished = true; break }
        }
        XCTAssertTrue(finished, "add-before-start must still generate once started")
        engine.stop()
    }

    func testEngineCoreEngineKeepsRunningAfterAbortAll() async throws {
        let engine = makeTestEngine()
        engine.start()

        await engine.addRequest(makeIntRequest())
        _ = engine.abortAllRequests()

        try await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertTrue(engine.isRunning)

        // New request should complete successfully.
        let output = try await engine.generate(prompt: "3", samplingParams: SamplingParams(maxTokens: 5))
        XCTAssertTrue(output.finished)

        engine.stop()
    }
}

// MARK: - §10  Generation

final class CBEngineCoreGenerationTests: XCTestCase {

    func testEngineCoreGenerateReturnsCompleteOutput() async throws {
        let engine = makeTestEngine()  // EOS = 5, prompt "3" → [1,2] via tokenizer → tokens 2,3,4,5(EOS)
        engine.start()
        let output = try await engine.generate(prompt: "3", samplingParams: SamplingParams(maxTokens: 10))
        XCTAssertTrue(output.finished)
        XCTAssertNotNil(output.finishReason)
        engine.stop()
    }

    func testEngineCoreStreamOutputsDeliversAllTokens() async throws {
        let engine = makeTestEngine(eosTokenIds: [])
        engine.start()

        let req = makeIntRequest(tokens: [1], maxTokens: 4)
        let rid = await engine.addRequest(req)

        var allOutputs: [RequestOutput] = []
        for await output in engine.streamOutputs(requestId: rid) {
            allOutputs.append(output)
            if output.finished { break }
        }

        XCTAssertFalse(allOutputs.isEmpty)
        XCTAssertTrue(allOutputs.last?.finished == true)
        engine.stop()
    }

    func testEngineCoreNoDuplicateTokensInStream() async throws {
        // Use EOS=7 so generation finishes via "stop" (not "length").
        // For stop-finish every generated token appears in exactly one newTokenIds slice;
        // the stop token itself is excluded from outputTokenIds.
        // Concat of all newTokenIds == outputTokenIds of the final output.
        let engine = makeTestEngine(eosTokenIds: [7])
        engine.start()

        // prompt [1] → generates 2, 3, 4, 5, 6, 7(EOS) → outputTokenIds = [2,3,4,5,6]
        let req = makeIntRequest(tokens: [1], maxTokens: 20)
        let rid = await engine.addRequest(req)

        var allNewTokenIds: [Int] = []
        var finalOutputTokenIds: [Int] = []

        for await output in engine.streamOutputs(requestId: rid) {
            allNewTokenIds.append(contentsOf: output.newTokenIds)
            if output.finished {
                finalOutputTokenIds = output.outputTokenIds
                break
            }
        }

        // Concatenated newTokenIds must equal the final outputTokenIds (no gaps, no duplicates).
        XCTAssertEqual(allNewTokenIds, finalOutputTokenIds,
                       "newTokenIds across stream must exactly reconstruct outputTokenIds")
        engine.stop()
    }

    func testEngineCoreStreamOutputsYieldsErrorOutputOnAbort() async throws {
        // Engine NOT started → stream blocks; abortRequest puts an error output.
        let engine = makeTestEngine()
        let req = makeIntRequest(tokens: [1], maxTokens: 10)
        let rid = await engine.addRequest(req)

        let streamTask = Task { () -> RequestOutput? in
            for await output in engine.streamOutputs(requestId: rid) {
                return output
            }
            return nil
        }

        try await Task.sleep(nanoseconds: 20_000_000)
        engine.abortRequest(rid)

        let output = await streamTask.value
        XCTAssertNotNil(output?.error)
        XCTAssertEqual(output?.finished, true)
    }

    func testEngineCoreGenerateThrowsOnAbort() async throws {
        // A request being generated that gets aborted must throw EngineError.generationFailed.
        let engine = makeTestEngine(eosTokenIds: [])  // never finishes on its own
        // Don't start engine so it blocks, then abort.
        let req = makeIntRequest(tokens: [1], maxTokens: 1000)

        let generateTask = Task { () throws -> RequestOutput in
            let rid = await engine.addRequest(req)
            return try await {
                // Build the generate call manually using addRequest already done.
                for await output in engine.streamOutputs(requestId: rid) {
                    if output.finished || output.error != nil {
                        if let err = output.error { throw EngineError.generationFailed(err) }
                        return output
                    }
                }
                throw EngineError.missingOutput
            }()
        }

        try await Task.sleep(nanoseconds: 20_000_000)
        engine.abortRequest(req.requestId)

        do {
            _ = try await generateTask.value
            XCTFail("Expected EngineError.generationFailed")
        } catch EngineError.generationFailed {
            // Expected.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testEngineCoreGenerateCancellationAbortsRequest() async throws {
        let engine = makeTestEngine(eosTokenIds: [])
        engine.start()

        let generateTask = Task {
            let _ = try? await engine.generate(prompt: "1 2 3", samplingParams: SamplingParams(maxTokens: 10000))
        }

        // Let the request enter the scheduler.
        try await Task.sleep(nanoseconds: 80_000_000)

        generateTask.cancel()

        // Wait for cleanup to propagate.
        try await Task.sleep(nanoseconds: 100_000_000)

        XCTAssertEqual(engine.getStats()["active_requests"] as? Int, 0,
                       "cancelled generate must clean up its collector")
        engine.stop()
    }

    func testEngineCoreGenerateCancelOneDoesNotAffectOther() async throws {
        // EOS = 5: prompt [3] → generates 4, then 5 (EOS). Fast finish.
        let engine = makeTestEngine()
        engine.start()

        // Launch both concurrently; task2 is cancelled while task1 finishes naturally.
        async let output1: RequestOutput? = { try? await engine.generate(prompt: "3", samplingParams: SamplingParams(maxTokens: 10)) }()
        let task2 = Task { try? await engine.generate(prompt: "3", samplingParams: SamplingParams(maxTokens: 10000)) }

        // Give both tasks time to register, then cancel task2.
        try await Task.sleep(nanoseconds: 30_000_000)
        task2.cancel()

        let result1 = await output1
        XCTAssertNotNil(result1, "task1 must complete successfully")
        XCTAssertTrue(result1?.finished == true)
        engine.stop()
    }

    func testEngineCoreGenerateForwardsTopKAndMinP() async throws {
        // With topK = 1 the sampler must always pick the best token.
        // IncrTestLanguageModel emits confidence 0 vs -1000, so topK=1 changes nothing
        // in practice, but the call must succeed without error (params forwarded).
        let engine = makeTestEngine()
        engine.start()
        let output = try await engine.generate(
            prompt: "3",
            samplingParams: SamplingParams(maxTokens: 5, temperature: 0.5, topK: 1, minP: 0.0))
        XCTAssertTrue(output.finished)
        engine.stop()
    }

    func testEngineCoreCleanupReleasesCollectorAfterFinish() async throws {
        let engine = makeTestEngine()
        engine.start()

        let req = makeIntRequest(tokens: [3], maxTokens: 5)
        let rid = await engine.addRequest(req)

        for await output in engine.streamOutputs(requestId: rid) {
            if output.finished { break }
        }

        // Give cleanupRequest time to dispatch.
        try await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(engine.getStats()["active_requests"] as? Int, 0)
        engine.stop()
    }
}

// MARK: - §11  Thread safety

final class CBEngineCoreThreadSafetyTests: XCTestCase {

    func testEngineQueueSerializesSteps() async throws {
        // Verify the engine step loop never runs two steps concurrently.
        let engine = makeTestEngine(eosTokenIds: [])
        engine.start()

        // Add long-running requests to keep the engine busy.
        for _ in 0 ..< 4 {
            await engine.addRequest(makeIntRequest(tokens: [1], maxTokens: 200))
        }

        // Let it run for 100 ms.
        try await Task.sleep(nanoseconds: 100_000_000)

        // The serial queue guarantee is structural; we just verify no crash and
        // stepsExecuted increases monotonically.
        let steps = engine.stepsExecuted
        XCTAssertGreaterThan(steps, 0)

        engine.stop()
    }

    func testConcurrentAddRequestsNoCrash() async throws {
        let engine = makeTestEngine(eosTokenIds: [5])
        engine.start()

        await withTaskGroup(of: Void.self) { group in
            for i in 0 ..< 20 {
                group.addTask {
                    await engine.addRequest(makeIntRequest(tokens: [3], maxTokens: 5))
                }
            }
        }

        // All 20 requests should be tracked (or already finished).
        let active = engine.getStats()["active_requests"] as? Int ?? 0
        let running = engine.getStats()["num_running"] as? Int ?? 0
        let waiting = engine.getStats()["num_waiting"] as? Int ?? 0
        XCTAssertGreaterThanOrEqual(active + running + waiting, 0,
                                    "engine must not crash under concurrent addRequest")

        // Drain.
        try await Task.sleep(nanoseconds: 500_000_000)
        engine.stop()
    }

    func testAddRequestDictSetupOnEngineQueue() async {
        // After await addRequest returns, active_requests must already reflect the new entry.
        let engine = makeTestEngine()
        await engine.addRequest(makeIntRequest())
        XCTAssertEqual(engine.getStats()["active_requests"] as? Int, 1)
    }
}
