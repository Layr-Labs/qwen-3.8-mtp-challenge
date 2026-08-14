// Copyright © 2026 Eigen Labs.
//
// WS-B: engine-loop tests with a scripted fake model (NO model weights).
// The scripted model emits next-token = (input + 1) % vocab per row, so
// expected outputs are exact and batch-composition invariant.
//
// Spec coverage: chained decode emits correct tokens one step late; stop
// token honored with ≤1 wasted step (KV tail rolled back); backpressure
// pauses exactly the slow stream; watchdog fires; deadlines error-finish;
// preemption end-to-end; cancel; shutdown drain.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2SchedulerLoopTests: XCTestCase {

    /// PR#62 review (paged admission alignment): when several same-step
    /// admissions race for the last capacity, the loser of the backend's
    /// atomic charge must be REQUEUED to waiting — an accepted request waits
    /// for room; it never error-finishes on capacity.
    func testCapacityLoserIsRequeuedNotErrored() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 8))
        harness.backend.maxLiveStates = 1
        async let a = cbv2SchedCollect(
            try harness.engine.submit(
                CBv2SchedFixtures.request(prompt: Array(0 ..< 6), maxTokens: 4)))
        async let b = cbv2SchedCollect(
            try harness.engine.submit(
                CBv2SchedFixtures.request(prompt: Array(6 ..< 12), maxTokens: 4)))
        let (ra, rb) = try await (a, b)
        await harness.engine.shutdown()
        XCTAssertEqual(ra.finishReason, .length, "first request must complete")
        XCTAssertEqual(
            rb.finishReason, .length,
            "capacity loser must complete after room frees, not error")
        XCTAssertGreaterThanOrEqual(
            harness.engine.loopForTesting.capacityRequeueCount, 1,
            "the loser must have gone through requeue-on-capacity")
    }

    /// PR#62 review: request ids are legally reusable after finish, so the
    /// per-id capacity requeue tally must be cleared on EVERY finish path —
    /// including the error-finish that exhausted it. Phase 1 exhausts
    /// `maxCapacityRequeues` for an id (a hog pins the only KV slot for
    /// more steps than the attempt budget; the victim burns one attempt per
    /// step). Phase 2 resubmits the SAME id behind a SHORT hog: it must get
    /// a full requeue window and complete, not inherit the stale tally and
    /// error-finish on its first transient trip.
    ///
    /// Determinism: within one engine, submission order = waiting FCFS
    /// order = allocation order, so a hog submitted first ALWAYS wins the
    /// only KV slot and the second request always trips capacityExhausted
    /// while the hog runs. No sleeps or polling on the contention path.
    func testReusedIDGetsFreshCapacityRequeueBudget() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 8))
        harness.backend.maxLiveStates = 1
        let reusedID = CBv2RequestID(9002)
        func request(_ id: CBv2RequestID, prompt: [Int], maxTokens: Int) -> CBv2Request {
            CBv2Request(id: id, promptTokens: prompt, maxTokens: maxTokens)
        }

        // Phase 1 — exhaust the attempt budget: the hog decodes for well
        // over maxCapacityRequeues steps, so the victim error-finishes
        // while the hog still holds the only slot.
        let hogStream = try harness.engine.submit(
            request(
                CBv2RequestID(9001), prompt: [3],
                maxTokens: EngineLoopV2.maxCapacityRequeues * 2 + 50))
        let victim = await cbv2SchedCollect(
            try harness.engine.submit(request(reusedID, prompt: [9], maxTokens: 2)))
        guard case .error(let message)? = victim.finishReason else {
            XCTFail(
                "victim must exhaust its requeue budget, got \(String(describing: victim.finishReason))"
            )
            return
        }
        XCTAssertTrue(message.contains("KV allocation failed"), "unexpected error: \(message)")
        let hog = await cbv2SchedCollect(hogStream)
        XCTAssertEqual(hog.finishReason, .length)
        let drained = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(drained, "phase 1 KV must be released")
        let requeuesAfterPhase1 = harness.engine.loopForTesting.capacityRequeueCount
        XCTAssertEqual(
            requeuesAfterPhase1, EngineLoopV2.maxCapacityRequeues,
            "phase 1 must have burned exactly the victim's attempt budget")

        // Phase 2 — reuse the id behind a short hog (well under the attempt
        // budget). The reused id's first capacityExhausted must REQUEUE
        // (fresh budget) and the request must complete once the hog frees
        // the slot — not error-finish off the stale phase-1 tally.
        let secondHogStream = try harness.engine.submit(
            request(CBv2RequestID(9003), prompt: [5], maxTokens: 24))
        let reused = await cbv2SchedCollect(
            try harness.engine.submit(request(reusedID, prompt: [7], maxTokens: 2)))
        let secondHog = await cbv2SchedCollect(secondHogStream)
        await harness.engine.shutdown()
        XCTAssertEqual(secondHog.finishReason, .length)
        XCTAssertEqual(
            reused.finishReason, .length,
            "reused id must complete with a fresh requeue budget, not inherit the exhausted tally"
        )
        XCTAssertGreaterThan(
            harness.engine.loopForTesting.capacityRequeueCount, requeuesAfterPhase1,
            "the reused id must actually have tripped capacity and been requeued")
    }

    /// PR#62 review (provider classification parity): the engine-thread
    /// maxWaiting backstop (authoritative; `EngineV2.submit`'s gauge check
    /// is only the fast path) must reject with the SAME canonical
    /// queue-full marker the submit path produces — submit throws
    /// `capacityExhausted(needed: 1, available: 0)`, which the provider
    /// renders as "token_budget_exhausted: request queue full" and
    /// classifies as a retryable 429 via a "queue full" substring match.
    /// The backstop's previous wording ("waiting queue is full") missed
    /// that match and turned the same condition into a 500.
    func testWaitingQueueBackstopEmitsCanonicalQueueFullMarker() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 1))
        // Hog pins the single RUNNING slot for many steps… (wait on the
        // PUBLISHED gauge, not the backend, so the waiter's submit-side
        // fast path is guaranteed to see the waiting queue as empty).
        let hogStream = try harness.engine.submit(
            CBv2Request(id: CBv2RequestID(9101), promptTokens: [3], maxTokens: 60))
        let hogRunning = await cbv2SchedWait {
            let snapshot = harness.engine.capacity()
            return snapshot.activeRequests == 1 && snapshot.waitingRequests == 0
        }
        XCTAssertTrue(hogRunning, "hog must be running before filling the waiting queue")
        // …and the waiter fills the single WAITING slot (it cannot be
        // admitted while the hog runs).
        let waiterStream = try harness.engine.submit(
            CBv2Request(id: CBv2RequestID(9102), promptTokens: [5], maxTokens: 2))

        // Drive the loop's enqueue path directly, the way EngineV2.submit
        // does AFTER its gauge fast path — this is exactly the backstop
        // that fires when the fast path raced a stale gauge snapshot.
        let loop = harness.engine.loopForTesting
        let overflow = CBv2OutputStream(id: CBv2RequestID(9103))
        XCTAssertTrue(loop.register(stream: overflow))
        loop.enqueue(CBv2Request(id: CBv2RequestID(9103), promptTokens: [7], maxTokens: 2))
        let rejected = await cbv2SchedCollect(overflow.makeStream())
        guard case .error(let message)? = rejected.finishReason else {
            XCTFail(
                "backstop must reject the overflow request, got \(String(describing: rejected.finishReason))"
            )
            return
        }
        XCTAssertEqual(
            message, "token_budget_exhausted: request queue full",
            "backstop rejection must carry the provider's canonical queue-full marker")

        let hog = await cbv2SchedCollect(hogStream)
        let waiter = await cbv2SchedCollect(waiterStream)
        await harness.engine.shutdown()
        XCTAssertEqual(hog.finishReason, .length)
        XCTAssertEqual(waiter.finishReason, .length, "the queued waiter still completes")
    }

    /// Expected scripted-model generation: prompt's last token + 1, +2, ...
    private func expectedTokens(prompt: [Int], count: Int, vocab: Int = 64) -> [Int] {
        var current = prompt.last!
        return (0 ..< count).map { _ in
            current = (current + 1) % vocab
            return current
        }
    }

    // MARK: Basic generation

    func testSingleRequestGeneratesExactSequence() async throws {
        let harness = CBv2SchedHarness()
        let request = CBv2SchedFixtures.request(prompt: [3], maxTokens: 5)
        let events = try harness.engine.submit(request)
        let collected = await cbv2SchedCollect(events)

        XCTAssertEqual(collected.tokens, [4, 5, 6, 7, 8])
        XCTAssertEqual(collected.text, "<4><5><6><7><8>")
        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertEqual(collected.usage?.promptTokens, 1)
        XCTAssertEqual(collected.usage?.completionTokens, 5)

        // KV cleaned up after finish.
        let released = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(released)
    }

    func testMultiChunkPrefillThenDecode() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64, prefillChunkSize: 16,
                maxWaiting: 8))
        let prompt = Array(0 ..< 40)  // 3 chunks: 16 + 16 + 8
        let request = CBv2SchedFixtures.request(prompt: prompt, maxTokens: 3)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        XCTAssertEqual(collected.tokens, [40, 41, 42])
        XCTAssertEqual(collected.finishReason, .length)
        // Prefill ran as [1, chunk] forwards: 16, 16, then the sampling 8.
        let shapes = harness.model.forwardShapes
        XCTAssertTrue(shapes.contains([1, 16]), "chunked prefill must run [1, chunk]: \(shapes)")
        XCTAssertTrue(shapes.contains([1, 8]), "final partial chunk: \(shapes)")
    }

    // MARK: Chained decode

    func testChainedDecodeProducesCorrectTokensOneStepLate() async throws {
        let harness = CBv2SchedHarness()
        let request = CBv2SchedFixtures.request(prompt: [10], maxTokens: 12)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        XCTAssertEqual(collected.tokens, expectedTokens(prompt: [10], count: 12))
        XCTAssertEqual(collected.finishReason, .length)
        // A single decoding request must chain nearly every decode step
        // (tokens are inspected one step late while N+1 runs).
        XCTAssertGreaterThanOrEqual(
            harness.engine.chainedStepCount, 8,
            "chained overlap must engage for steady decode")
    }

    func testStopTokenHonoredWithAtMostOneWastedStepAndKVTailRollback() async throws {
        let harness = CBv2SchedHarness()
        // Scripted tokens: 4, 5, 6, ... — stop at 6.
        let request = CBv2SchedFixtures.request(prompt: [3], maxTokens: 100, stopTokens: [6])
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        XCTAssertEqual(collected.tokens, [4, 5, 6], "nothing past the stop token")
        XCTAssertEqual(collected.finishReason, .stop)
        XCTAssertEqual(collected.usage?.completionTokens, 3)

        // The chained step launched before the stop was inspected computed
        // one extra token; its KV tail must have been rolled back exactly 1.
        let released = await cbv2SchedWait { harness.backend.releasedStates.count == 1 }
        XCTAssertTrue(released)
        let state = harness.backend.releasedStates[0]
        let sequence = state[0] as! CBv2SchedMockSequenceKV
        XCTAssertEqual(
            sequence.rollbackCalls, [1],
            "wasted chained-step KV write must be rolled back exactly once")
    }

    func testStopStringViaDetokenizerHoldback() async throws {
        // The scripted detokenizer flags a stop-string match when it sees
        // token 20 (stands in for WS-E's StopHoldback).
        let harness = CBv2SchedHarness(stopTrigger: 20)
        let request = CBv2SchedFixtures.request(
            prompt: [17], maxTokens: 100, stopStrings: ["<stop>"])
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        XCTAssertEqual(collected.tokens, [18, 19, 20])
        XCTAssertEqual(collected.finishReason, .stop)
    }

    // MARK: Batch composition

    func testBatchmatesDoNotChangeEachOthersOutput() async throws {
        // Solo run.
        let solo = CBv2SchedHarness()
        let soloOut = await cbv2SchedCollect(
            try solo.engine.submit(CBv2SchedFixtures.request(prompt: [3], maxTokens: 6)))

        // Same request logic with two batchmates of different lengths.
        let batched = CBv2SchedHarness()
        let a = CBv2SchedFixtures.request(prompt: [3], maxTokens: 6)
        let b = CBv2SchedFixtures.request(prompt: [30, 31], maxTokens: 9)
        let c = CBv2SchedFixtures.request(prompt: [50], maxTokens: 2)
        let streamA = try batched.engine.submit(a)
        let streamB = try batched.engine.submit(b)
        let streamC = try batched.engine.submit(c)
        async let outA = cbv2SchedCollect(streamA)
        async let outB = cbv2SchedCollect(streamB)
        async let outC = cbv2SchedCollect(streamC)
        let (collectedA, collectedB, collectedC) = await (outA, outB, outC)

        XCTAssertEqual(collectedA.tokens, soloOut.tokens, "batch-composition invariance")
        XCTAssertEqual(collectedA.tokens, [4, 5, 6, 7, 8, 9])
        XCTAssertEqual(collectedB.tokens, expectedTokens(prompt: [30, 31], count: 9))
        XCTAssertEqual(collectedC.tokens, [51, 52])
        XCTAssertEqual(collectedA.finishReason, .length)
        XCTAssertEqual(collectedB.finishReason, .length)
        XCTAssertEqual(collectedC.finishReason, .length)
    }

    // MARK: Cancellation

    func testCancelRunningRequestFinishesPromptlyAndFreesKV() async throws {
        let harness = CBv2SchedHarness()
        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 1_000_000)
        let stream = try harness.engine.submit(request)

        let collector = Task { await cbv2SchedCollect(stream) }
        // Let it decode a bit, then cancel.
        _ = await cbv2SchedWait { harness.engine.stepCount > 5 }
        harness.engine.cancel(request.id)
        let collected = await collector.value

        XCTAssertEqual(collected.finishReason, .cancelled)
        // Output up to the cancel is the correct prefix.
        XCTAssertEqual(
            collected.tokens, expectedTokens(prompt: [1], count: collected.tokens.count))
        let released = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(released, "cancelled request's KV must be freed")
    }

    func testCancelImmediatelyAfterSubmitIsHonored() async throws {
        // Race: a cancel arrives after `submit` registered the stream but
        // BEFORE the engine's enqueue block runs (the scheduler has no record
        // yet). Artificially delay enqueue to make the window deterministic;
        // the request must never start (PR#62 review). Without the fix the
        // cancel is dropped and the request generates its full budget.
        let harness = CBv2SchedHarness()
        harness.engine.loopForTesting.enqueueStartDelayForTesting = 0.2

        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 50)
        let stream = try harness.engine.submit(request)
        // Cancel while the enqueue block is still sleeping (pre-scheduler).
        harness.engine.cancel(request.id)

        let collected = await cbv2SchedCollect(stream)
        XCTAssertEqual(
            collected.finishReason, .cancelled,
            "an early cancel must abort the request before it starts")
        XCTAssertTrue(
            collected.tokens.isEmpty, "a request cancelled before start must emit no tokens")

        // Nothing was ever admitted / allocated.
        XCTAssertEqual(harness.backend.makeCalls, 0, "KV must never be allocated")
        let idle = await cbv2SchedWait { harness.engine.capacity().activeRequests == 0 }
        XCTAssertTrue(idle)
    }

    // MARK: Backpressure

    func testBackpressurePausesExactlyTheSlowStream() async throws {
        let harness = CBv2SchedHarness(
            loopConfig: CBv2EngineLoopConfig(eventBufferCapacity: 4))
        let slow = CBv2SchedFixtures.request(prompt: [1], maxTokens: 30)
        let fast = CBv2SchedFixtures.request(prompt: [20], maxTokens: 30)
        let slowStream = try harness.engine.submit(slow)
        let fastStream = try harness.engine.submit(fast)

        // Consume fast eagerly; leave slow unread.
        let fastOut = await cbv2SchedCollect(fastStream)
        XCTAssertEqual(fastOut.tokens, expectedTokens(prompt: [20], count: 30))
        XCTAssertEqual(fastOut.finishReason, .length)

        // Exactly the slow request is paused, slot retained.
        let paused = await cbv2SchedWait {
            harness.engine.loopForTesting.pausedIDsSnapshot() == [slow.id]
        }
        XCTAssertTrue(paused, "slow stream must be paused while fast one finished")
        XCTAssertEqual(harness.engine.capacity().activeRequests, 1)

        // Draining the slow stream resumes it; the full sequence arrives in
        // order with nothing lost.
        let slowOut = await cbv2SchedCollect(slowStream)
        XCTAssertEqual(slowOut.tokens, expectedTokens(prompt: [1], count: 30))
        XCTAssertEqual(slowOut.finishReason, .length)
    }

    // MARK: Preemption end-to-end

    func testPreemptionRequeuesVictimWhichStillCompletesCorrectly() async throws {
        // 16 B/token (1 layer, kv1, hd4, fp16). Capacity 200 B ⇒ 12 tokens
        // of soft ledger. Two requests of prompt 4 + 6 generated = 10 tokens
        // each can never coexist ⇒ the younger is preempted mid-decode and
        // must resume (re-prefill with KEPT generated tokens) and finish
        // with a seamless token stream.
        let harness = CBv2SchedHarness(backendCapacity: 200)
        let older = CBv2SchedFixtures.request(prompt: [1, 2, 3, 4], maxTokens: 6)
        let younger = CBv2SchedFixtures.request(prompt: [40, 41, 42, 43], maxTokens: 6)
        let olderStream = try harness.engine.submit(older)
        let youngerStream = try harness.engine.submit(younger)
        async let olderOut = cbv2SchedCollect(olderStream)
        async let youngerOut = cbv2SchedCollect(youngerStream)
        let (olderCollected, youngerCollected) = await (olderOut, youngerOut)

        XCTAssertEqual(olderCollected.tokens, [5, 6, 7, 8, 9, 10])
        XCTAssertEqual(olderCollected.finishReason, .length)
        XCTAssertEqual(
            youngerCollected.tokens, [44, 45, 46, 47, 48, 49],
            "preempted request must keep generated tokens and continue seamlessly")
        XCTAssertEqual(youngerCollected.finishReason, .length)
        XCTAssertGreaterThan(harness.engine.preemptionCount, 0, "capacity forces preemption")
    }

    /// Regression: `prefixHitTokens` was not cleared when an ADOPTED request
    /// was preempted. Preemption discards the adopted KV and recomputes the
    /// full prompt from scratch, so a stale entry over-credited
    /// usage.prefixCacheHitTokens at finish. Both preemption paths now drop
    /// the entry; a non-preempted adopted batchmate keeps its true credit.
    func testPreemptedAdoptedRequestReportsZeroPrefixHitTokens() async throws {
        // Same capacity shape as the preemption test above (12-token soft
        // ledger), plus a scripted prefix cache that claims a 2-token hit
        // for every prompt. Victim = youngest ⇒ the second request is
        // preempted mid-decode and recomputes everything.
        let matched = 2
        let prefixCache = CBv2SchedScriptedPrefixCache(matched: matched)
        let harness = CBv2SchedHarness(
            backendCapacity: 200,
            schedulerConfig: CBv2SchedulerConfig(enablePrefixCache: true),
            prefixCache: prefixCache)
        let older = CBv2SchedFixtures.request(prompt: [1, 2, 3, 4], maxTokens: 6)
        let younger = CBv2SchedFixtures.request(prompt: [40, 41, 42, 43], maxTokens: 6)
        let olderStream = try harness.engine.submit(older)
        let youngerStream = try harness.engine.submit(younger)
        async let olderOut = cbv2SchedCollect(olderStream)
        async let youngerOut = cbv2SchedCollect(youngerStream)
        let (olderCollected, youngerCollected) = await (olderOut, youngerOut)

        XCTAssertGreaterThan(harness.engine.preemptionCount, 0, "capacity forces preemption")
        // Token streams stay seamless either way.
        XCTAssertEqual(olderCollected.tokens, [5, 6, 7, 8, 9, 10])
        XCTAssertEqual(youngerCollected.tokens, [44, 45, 46, 47, 48, 49])

        // Control: the non-preempted adopted request keeps its true credit.
        XCTAssertEqual(
            olderCollected.usage?.prefixCacheHitTokens, matched,
            "adopted, never preempted: usage reports the skipped tokens")
        // Regression: preemption + full recompute means NOTHING was skipped.
        XCTAssertEqual(
            youngerCollected.usage?.prefixCacheHitTokens, 0,
            "preempted-adopted request recomputed everything; stale credit must be dropped")

        // Every lookup pin was balanced by exactly one endAdoption.
        XCTAssertEqual(prefixCache.lookups, 2)
        XCTAssertEqual(prefixCache.endAdoptions, prefixCache.lookups)
    }

    // MARK: Deadlines & watchdog

    func testRequestDeadlineErrorFinishes() async throws {
        let harness = CBv2SchedHarness(
            loopConfig: CBv2EngineLoopConfig(requestTimeout: 0.05))
        harness.model.forwardDelay = 0.02  // a few slow steps blow the deadline
        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 1_000_000)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        guard case .error(let message)? = collected.finishReason else {
            return XCTFail("expected deadline error, got \(String(describing: collected.finishReason))")
        }
        XCTAssertTrue(message.contains("deadline"), message)
        let released = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(released)
    }

    func testWatchdogFiresOnWedgedStepAndErrorsStreams() async throws {
        final class WedgeFlag: @unchecked Sendable {
            private let lock = NSLock()
            private var _count = 0
            func fire() {
                lock.lock()
                _count += 1
                lock.unlock()
            }
            var count: Int {
                lock.lock()
                defer { lock.unlock() }
                return _count
            }
        }
        let flag = WedgeFlag()
        let harness = CBv2SchedHarness(
            loopConfig: CBv2EngineLoopConfig(
                stepTimeout: 0.05, watchdogInterval: 0.01))
        harness.engine.onStepWedge = { _ in flag.fire() }
        harness.model.forwardDelay = 0.5  // wedge the first step

        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 100)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        guard case .error(let message)? = collected.finishReason else {
            return XCTFail("expected watchdog error, got \(String(describing: collected.finishReason))")
        }
        XCTAssertTrue(message.contains("watchdog"), message)
        XCTAssertGreaterThanOrEqual(flag.count, 1, "wedge signal must fire")
        // The loop resumes after the wedge and cleans up the row.
        let cleaned = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(cleaned)
        XCTAssertTrue(harness.engine.isHealthy, "health recovers once the step completes")
    }

    // MARK: Degenerate submissions

    func testDegenerateSubmissionsFinishImmediately() async throws {
        let harness = CBv2SchedHarness()
        let zeroBudget = await cbv2SchedCollect(
            try harness.engine.submit(CBv2SchedFixtures.request(prompt: [1], maxTokens: 0)))
        XCTAssertEqual(zeroBudget.finishReason, .length)
        XCTAssertTrue(zeroBudget.tokens.isEmpty)

        let empty = await cbv2SchedCollect(
            try harness.engine.submit(CBv2SchedFixtures.request(prompt: [], maxTokens: 5)))
        guard case .error? = empty.finishReason else {
            return XCTFail("empty prompt must error-finish")
        }
    }

    func testOversizedRequestThrowsCapacityExhausted() throws {
        // 16 B/token, capacity 160 B ⇒ 10 tokens worst case.
        let harness = CBv2SchedHarness(backendCapacity: 160)
        XCTAssertThrowsError(
            try harness.engine.submit(
                CBv2SchedFixtures.request(prompt: Array(0 ..< 8), maxTokens: 8))
        ) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
        }
    }

    /// Regression (livelock): a request in `(capacity - watermark, capacity]`
    /// used to pass `canEverFit` (full capacity) but could never complete a
    /// reservation (`reserve` enforces capacity - watermark) — it hit the
    /// wall, self-preempted, restarted, and looped until its deadline. It
    /// must be rejected at submit with capacityExhausted instead.
    func testRequestInsideWatermarkBandRejectedAtSubmit() throws {
        // 16 B/token, capacity 1600 B, watermark 5 % (80 B) ⇒ 95 tokens is
        // the true ceiling; 96..100 tokens sit in the livelock band.
        let harness = CBv2SchedHarness(
            backendCapacity: 1600,
            admissionConfig: .init(watermarkFraction: 0.05))
        XCTAssertThrowsError(
            try harness.engine.submit(
                CBv2SchedFixtures.request(prompt: Array(0 ..< 48), maxTokens: 50))
        ) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
        }
        // Just under the watermark-adjusted ceiling still admits.
        XCTAssertNoThrow(
            try harness.engine.submit(
                CBv2SchedFixtures.request(prompt: Array(0 ..< 45), maxTokens: 50)))
    }

    // MARK: Shutdown

    func testShutdownDrainsRunningCancelsWaitingAndRejectsNew() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 2048, prefillChunkSize: 512,
                maxWaiting: 8))
        let running = CBv2SchedFixtures.request(prompt: [1], maxTokens: 40)
        let queued = CBv2SchedFixtures.request(prompt: [2], maxTokens: 5)
        let runningStream = try harness.engine.submit(running)
        // Ensure `running` occupies the single slot before `queued` arrives.
        _ = await cbv2SchedWait { harness.engine.capacity().activeRequests == 1 }
        let queuedStream = try harness.engine.submit(queued)
        _ = await cbv2SchedWait { harness.engine.capacity().waitingRequests == 1 }

        async let runningOut = cbv2SchedCollect(runningStream)
        async let queuedOut = cbv2SchedCollect(queuedStream)
        await harness.engine.shutdown()
        let (runningCollected, queuedCollected) = await (runningOut, queuedOut)

        XCTAssertEqual(
            runningCollected.tokens, expectedTokens(prompt: [1], count: 40),
            "running request finishes naturally during drain")
        XCTAssertEqual(runningCollected.finishReason, .length)
        XCTAssertEqual(queuedCollected.finishReason, .cancelled, "waiting request cancelled")

        XCTAssertThrowsError(
            try harness.engine.submit(CBv2SchedFixtures.request(prompt: [9], maxTokens: 1))
        ) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected capacityExhausted after shutdown, got \(error)")
            }
        }
    }

    /// Regression: `shutdown()` waits for the drain on the ENGINE queue, so
    /// a wedged queue (a step blocked inside eval) hung it forever. It is
    /// now bounded by `shutdownTimeout`: live streams are force-finished
    /// with `.error` and shutdown returns while the wedged step is still
    /// blocked.
    func testShutdownTimesOutWhenEngineQueueIsWedged() async throws {
        let harness = CBv2SchedHarness(
            loopConfig: CBv2EngineLoopConfig(
                requestTimeout: 60,
                stepTimeout: 60,  // keep the step watchdog out of the way
                shutdownTimeout: 0.3))
        harness.model.forwardDelay = 3.0  // wedge the engine queue

        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 100)
        let stream = try harness.engine.submit(request)
        async let collectedOut = cbv2SchedCollect(stream)

        // Give the engine a moment to enter the wedged step.
        try await Task.sleep(nanoseconds: 200_000_000)

        let started = Date()
        await harness.engine.shutdown()
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertLessThan(
            elapsed, 2.5,
            "shutdown must return at the timeout, not wait out the wedged step")

        let collected = await collectedOut
        guard case .error(let message)? = collected.finishReason else {
            return XCTFail(
                "expected shutdown-timeout error, got \(String(describing: collected.finishReason))"
            )
        }
        XCTAssertTrue(message.contains("shutdown"), message)
    }

    // MARK: Duplicate request ids at the engine surface (PR#62 review)

    /// A second `submit` reusing a still-live id must throw WITHOUT touching
    /// the first request's stream. Regression: registration used to
    /// overwrite `streams[id]` before the scheduler rejected the duplicate,
    /// and the rejection path then tore down the replacement — orphaning the
    /// FIRST request's deltas and terminal event.
    func testDuplicateSubmitThrowsAndFirstStreamStillDelivers() async throws {
        let harness = CBv2SchedHarness(
            loopConfig: CBv2EngineLoopConfig(eventBufferCapacity: 4))
        let first = CBv2SchedFixtures.request(prompt: [1], maxTokens: 30)
        let firstStream = try harness.engine.submit(first)
        // Hold the first request live deterministically: the unread stream
        // hits backpressure and the request pauses with its slot retained.
        let held = await cbv2SchedWait {
            harness.engine.loopForTesting.pausedIDsSnapshot() == [first.id]
        }
        XCTAssertTrue(held, "first request must be live (paused) before the duplicate arrives")

        let dup = CBv2Request(id: first.id, promptTokens: [9], maxTokens: 5)
        XCTAssertThrowsError(try harness.engine.submit(dup)) { error in
            XCTAssertEqual(
                error as? CBv2SchedulerError, .duplicateRequestID(first.id),
                "duplicate id must be rejected before any stream registration")
        }

        // The FIRST request's stream is untouched: draining it resumes
        // decoding and the full scripted sequence + terminal arrive.
        let out = await cbv2SchedCollect(firstStream)
        XCTAssertEqual(out.tokens, expectedTokens(prompt: [1], count: 30))
        XCTAssertEqual(out.finishReason, .length)

        // After the finish the id is legally reusable.
        let reuse = CBv2Request(id: first.id, promptTokens: [5], maxTokens: 3)
        let reuseOut = await cbv2SchedCollect(try harness.engine.submit(reuse))
        XCTAssertEqual(reuseOut.tokens, expectedTokens(prompt: [5], count: 3))
        XCTAssertEqual(reuseOut.finishReason, .length)
        await harness.engine.shutdown()
    }

    // MARK: Sampler id retirement (PR#62 review)

    /// Every finish must retire the id from the sampler, so a FUTURE request
    /// reusing it cannot inherit stale per-request sampler state.
    func testFinishNotifiesSamplerOfRetiredID() async throws {
        final class SpySampler: CBv2StepSampler, @unchecked Sendable {
            private let inner = CBv2GreedySampler()
            private let lock = NSLock()
            private var _retired: [CBv2RequestID] = []
            var retired: [CBv2RequestID] {
                lock.lock()
                defer { lock.unlock() }
                return _retired
            }
            func sample(
                logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
                stepIndex: Int, pendingSampledTokens: MLXArray?,
                rowContext: () -> [CBv2SamplerRow]
            ) -> MLXArray {
                inner.sample(
                    logits: logits, params: params, requestIDs: requestIDs,
                    stepIndex: stepIndex, pendingSampledTokens: pendingSampledTokens,
                    rowContext: rowContext)
            }
            func requestDidFinish(_ id: CBv2RequestID) {
                lock.lock()
                _retired.append(id)
                lock.unlock()
            }
        }

        let layerKinds = CBv2SchedFixtures.tinyLayerKinds()
        let sampler = SpySampler()
        let engine = EngineV2(
            model: CBv2SchedScriptedModel(),
            layerKinds: layerKinds,
            backend: CBv2SchedMockBackend(),
            cacheProvider: CBv2SchedMockCacheProvider(layerKinds: layerKinds),
            sampler: sampler,
            admissionConfig: .init(watermarkFraction: 0))

        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 4)
        let collected = await cbv2SchedCollect(try engine.submit(request))
        XCTAssertEqual(collected.finishReason, .length)
        let told = await cbv2SchedWait { sampler.retired.contains(request.id) }
        XCTAssertTrue(told, "finishRequest must retire the id from the sampler")
        await engine.shutdown()
    }

    // MARK: Idle KV retention (PR#62 review)

    /// After a request finishes and the engine goes idle, NO sequence-KV row
    /// may remain strongly retained: the eager layer-cache bank must drop
    /// its row bindings at idle instead of pinning the last batch's retired
    /// rows until some future rebind.
    func testIdleEngineDropsEagerBankRowBindings() async throws {
        let layerKinds = CBv2SchedFixtures.tinyLayerKinds()
        let backend = CBv2SchedWeakTrackingBackend()
        let engine = EngineV2(
            model: CBv2SchedScriptedModel(),
            layerKinds: layerKinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: layerKinds),
            sampler: CBv2GreedySampler(),
            admissionConfig: .init(watermarkFraction: 0))

        let request = CBv2SchedFixtures.request(prompt: [1, 2], maxTokens: 5)
        let collected = await cbv2SchedCollect(try engine.submit(request))
        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertGreaterThan(backend.trackedRowCount, 0, "sanity: rows were minted")

        let released = await cbv2SchedWait { backend.liveTrackedRowCount == 0 }
        XCTAssertTrue(
            released,
            "idle engine still retains \(backend.liveTrackedRowCount) retired KV row(s) — the eager bank must drop its bindings at idle"
        )
        await engine.shutdown()
    }

    /// Bank-level pin: `releaseBoundRows` drops the strong row bindings and
    /// a later bind starts a fresh composition.
    func testBankReleaseBoundRowsDropsRowRetention() {
        let layerKinds = CBv2SchedFixtures.tinyLayerKinds()
        let bank = CBv2LayerCacheBank(layerKinds: layerKinds)

        func bindDisposableRow() -> () -> Bool {
            weak var weakRow: CBv2SchedMockSequenceKV?
            let row = CBv2SchedMockSequenceKV()
            weakRow = row
            _ = bank.layerCaches(rowStates: [[row]])
            return { weakRow == nil }
        }

        let rowIsGone = bindDisposableRow()
        XCTAssertFalse(rowIsGone(), "bank retains bound rows until released")
        bank.releaseBoundRows()
        XCTAssertTrue(rowIsGone(), "releaseBoundRows must drop the strong binding")

        // Rebinding afterwards works (fresh composition), and releasing an
        // already-empty bank is a no-op.
        let row = CBv2SchedMockSequenceKV()
        let caches = bank.layerCaches(rowStates: [[row]])
        XCTAssertEqual((caches[0] as? CBv2LayerCache)?.rows.count, 1)
        bank.releaseBoundRows()
        bank.releaseBoundRows()
    }
}

// MARK: - Weak-tracking backend (idle-retention tests)

/// Mints mock rows and tracks each one WEAKLY, retaining nothing itself
/// (unlike `CBv2SchedMockBackend`, whose `releasedStates` log keeps rows
/// alive) — so a live weak reference after finish + idle can only mean the
/// engine side still pins the row.
private final class CBv2SchedWeakTrackingBackend: CBv2KVBackend, @unchecked Sendable {
    private final class WeakRow {
        weak var value: AnyObject?
        init(_ value: AnyObject) { self.value = value }
    }

    private let lock = NSLock()
    private var tracked: [WeakRow] = []

    var trackedRowCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return tracked.count
    }

    var liveTrackedRowCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return tracked.filter { $0.value != nil }.count
    }

    private func mint(layerKinds: [CBv2LayerKind]) -> [CBv2SequenceKV?] {
        let state = layerKinds.map { kind -> CBv2SequenceKV? in
            kind.sharesKVWithLayer == nil ? CBv2SchedMockSequenceKV() : nil
        }
        lock.lock()
        for row in state {
            if let row { tracked.append(WeakRow(row)) }
        }
        lock.unlock()
        return state
    }

    func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        mint(layerKinds: layerKinds)
    }

    func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        mint(layerKinds: layerKinds)
    }

    func release(_ state: [CBv2SequenceKV?]) {}
    var bytesInUse: Int { 0 }
    var bytesCapacity: Int { 1 << 30 }
}
