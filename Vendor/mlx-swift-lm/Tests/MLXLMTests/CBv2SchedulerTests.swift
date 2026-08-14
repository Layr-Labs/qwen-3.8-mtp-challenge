// Copyright © 2026 Eigen Labs.
//
// WS-B: pure scheduler tests (no MLX arrays, no weights, no engine thread).
// Spec: docs/engine-v2/specs/B-scheduler.md — budget invariants, chunked
// prefill, preemption victim + requeue-front, optimistic-advance rollback,
// priority ordering, cancel in every state, Poisson simulation.

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBv2SchedulerTests: XCTestCase {

    private func makeScheduler(
        maxConcurrent: Int = 4, budget: Int = 2048, chunk: Int = 512, maxWaiting: Int = 64,
        capacity: CBv2StepCapacity? = nil
    ) -> SchedulerV2 {
        SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent, maxBatchedTokensPerStep: budget,
                prefillChunkSize: chunk, maxWaiting: maxWaiting),
            capacity: capacity)
    }

    private func assertInvariants(
        _ scheduler: SchedulerV2, plan: CBv2StepPlan, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let total = plan.assignments.reduce(0) { $0 + $1.numTokens }
        XCTAssertLessThanOrEqual(
            total, scheduler.config.maxBatchedTokensPerStep, "budget exceeded", file: file,
            line: line)
        XCTAssertLessThanOrEqual(
            scheduler.runningCount, scheduler.config.maxConcurrentRequests,
            "running exceeds maxConcurrent", file: file, line: line)
        for (id, n) in plan.assignments {
            XCTAssertGreaterThan(n, 0, file: file, line: line)
            XCTAssertTrue(
                scheduler.running.contains { $0.id == id },
                "assignment to a non-running request", file: file, line: line)
        }
    }

    // MARK: Chunked prefill & decode

    func testChunkedPrefillProgressAndDecodeTransition() throws {
        let scheduler = makeScheduler(budget: 2048, chunk: 512)
        let request = CBv2SchedFixtures.request(prompt: Array(0 ..< 1000), maxTokens: 3)
        try scheduler.enqueue(request)

        // Chunk 1: admitted with a partial token count.
        var plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.count, 1)
        XCTAssertEqual(plan.assignments[0].numTokens, 512)
        XCTAssertEqual(scheduler.record(for: request.id)?.numComputedTokens, 512)
        XCTAssertTrue(CBv2SchedSim.confirm(scheduler, plan: plan).isEmpty)

        // Chunk 2: remainder, completes the prompt and samples.
        plan = scheduler.plan()
        XCTAssertEqual(plan.assignments[0].numTokens, 488)
        XCTAssertEqual(CBv2SchedSim.confirm(scheduler, plan: plan), [request.id])

        // Decode: exactly one token per step.
        for _ in 0 ..< 2 {
            plan = scheduler.plan()
            XCTAssertEqual(plan.assignments.count, 1)
            XCTAssertEqual(plan.assignments[0].numTokens, 1)
            CBv2SchedSim.confirm(scheduler, plan: plan)
        }
        XCTAssertEqual(scheduler.record(for: request.id)?.generatedTokenCount, 3)
    }

    func testBudgetSplitsAcrossConcurrentPartialPrefills() throws {
        let scheduler = makeScheduler(budget: 300, chunk: 256)
        let first = CBv2SchedFixtures.request(prompt: Array(repeating: 1, count: 400), maxTokens: 1)
        let second = CBv2SchedFixtures.request(prompt: Array(repeating: 2, count: 400), maxTokens: 1)
        try scheduler.enqueue(first)
        try scheduler.enqueue(second)

        // Both admitted in one step: 256 + 44 = 300 (vLLM multiple partial
        // prefills under one budget — no phases).
        let plan = scheduler.plan()
        assertInvariants(scheduler, plan: plan)
        XCTAssertEqual(plan.assignments.map(\.numTokens), [256, 44])
        XCTAssertEqual(scheduler.runningCount, 2)
    }

    func testRunningScheduledBeforeWaiting() throws {
        let scheduler = makeScheduler(budget: 100, chunk: 100)
        let running = CBv2SchedFixtures.request(prompt: Array(repeating: 1, count: 90), maxTokens: 5)
        try scheduler.enqueue(running)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // prefill done

        let waiting = CBv2SchedFixtures.request(prompt: Array(repeating: 2, count: 200), maxTokens: 5)
        try scheduler.enqueue(waiting)

        // Decode row gets its token FIRST; the waiting prompt gets leftovers.
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments[0].id, running.id)
        XCTAssertEqual(plan.assignments[0].numTokens, 1)
        XCTAssertEqual(plan.assignments[1].id, waiting.id)
        XCTAssertEqual(plan.assignments[1].numTokens, 99)
    }

    // MARK: Priority & FCFS

    func testPriorityOrderingWithFCFSWithinClass() throws {
        let scheduler = makeScheduler(maxConcurrent: 1)
        let lowFirst = CBv2SchedFixtures.request(prompt: [1], maxTokens: 1, priority: 0)
        let lowSecond = CBv2SchedFixtures.request(prompt: [2], maxTokens: 1, priority: 0)
        let high = CBv2SchedFixtures.request(prompt: [3], maxTokens: 1, priority: 5)
        try scheduler.enqueue(lowFirst)
        try scheduler.enqueue(lowSecond)
        try scheduler.enqueue(high)

        // Higher priority admitted first even though it arrived last.
        XCTAssertEqual(scheduler.waiting.map(\.id), [high.id, lowFirst.id, lowSecond.id])
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.first?.id, high.id)
    }

    // MARK: maxWaiting

    func testMaxWaitingRejectsWithCapacityError() throws {
        let scheduler = makeScheduler(maxWaiting: 2)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: [1], maxTokens: 1))
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: [2], maxTokens: 1))
        XCTAssertThrowsError(
            try scheduler.enqueue(CBv2SchedFixtures.request(prompt: [3], maxTokens: 1))
        ) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
        }
    }

    // MARK: Duplicate request ids (PR#62 review)

    func testDuplicateRequestIDWhileWaitingIsRejected() throws {
        let scheduler = makeScheduler()
        let first = CBv2SchedFixtures.request(prompt: [1, 2, 3], maxTokens: 5)
        try scheduler.enqueue(first)

        // Same id, original still WAITING (no plan yet).
        let dup = CBv2Request(id: first.id, promptTokens: [9], maxTokens: 5)
        XCTAssertThrowsError(try scheduler.enqueue(dup)) { error in
            XCTAssertEqual(
                error as? CBv2SchedulerError, .duplicateRequestID(first.id),
                "duplicate must be rejected, not silently clobber byID")
        }
        // The original record is untouched and no phantom queue entry exists.
        XCTAssertEqual(scheduler.record(for: first.id)?.request.promptTokens, [1, 2, 3])
        XCTAssertEqual(scheduler.waitingCount, 1)
        XCTAssertEqual(scheduler.runningCount, 0)
    }

    func testDuplicateRequestIDWhileRunningIsRejected() throws {
        let scheduler = makeScheduler()
        let first = CBv2SchedFixtures.request(prompt: [1, 2, 3], maxTokens: 5)
        try scheduler.enqueue(first)
        _ = scheduler.plan()  // promotes to RUNNING
        XCTAssertEqual(scheduler.runningCount, 1)

        let dup = CBv2Request(id: first.id, promptTokens: [9], maxTokens: 5)
        XCTAssertThrowsError(try scheduler.enqueue(dup)) { error in
            XCTAssertEqual(error as? CBv2SchedulerError, .duplicateRequestID(first.id))
        }
        // Running record intact; no stale waiting entry orphaned in a queue.
        XCTAssertEqual(scheduler.runningCount, 1)
        XCTAssertEqual(scheduler.waitingCount, 0)
        XCTAssertEqual(scheduler.record(for: first.id)?.request.promptTokens, [1, 2, 3])
    }

    func testRequestIDReusableAfterFinish() throws {
        let scheduler = makeScheduler()
        let first = CBv2SchedFixtures.request(prompt: [1], maxTokens: 1)
        try scheduler.enqueue(first)
        scheduler.finish(id: first.id, reason: .cancelled)
        // Once fully finished, the id is free to reuse.
        let reuse = CBv2Request(id: first.id, promptTokens: [7], maxTokens: 1)
        XCTAssertNoThrow(try scheduler.enqueue(reuse))
        XCTAssertEqual(scheduler.record(for: first.id)?.request.promptTokens, [7])
    }

    // MARK: Preemption

    func testPreemptionPicksLowestPriorityYoungestAndRequeuesFront() throws {
        let capacity = CBv2SchedMockCapacity(tokenLimit: 32)
        let scheduler = makeScheduler(capacity: capacity)
        // Arrival order: older(pri 1), middle(pri 0), youngest(pri 0).
        let older = CBv2SchedFixtures.request(prompt: Array(repeating: 1, count: 10), maxTokens: 8, priority: 1)
        let middle = CBv2SchedFixtures.request(prompt: Array(repeating: 2, count: 10), maxTokens: 8, priority: 0)
        let youngest = CBv2SchedFixtures.request(prompt: Array(repeating: 3, count: 10), maxTokens: 8, priority: 0)
        try scheduler.enqueue(older)
        try scheduler.enqueue(middle)
        try scheduler.enqueue(youngest)

        // Prefill all three: 30 of 32 tokens reserved.
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
        XCTAssertEqual(scheduler.runningCount, 3)
        XCTAssertEqual(capacity.totalReserved, 30)

        // Decode step: older(+1)=31, middle(+1)=32, youngest throws → victim
        // is the lowest-priority youngest — youngest itself (self-preempt ⇒
        // scheduling stops).
        let plan = scheduler.plan()
        XCTAssertEqual(plan.preemptions, [youngest.id])
        XCTAssertEqual(plan.assignments.map(\.id), [older.id, middle.id])

        let victim = scheduler.record(for: youngest.id)!
        XCTAssertEqual(victim.status, .preempted)
        XCTAssertEqual(victim.numComputedTokens, 0, "preemption is a full restart")
        XCTAssertEqual(victim.tokens.count, 11, "generated tokens are KEPT")
        XCTAssertEqual(scheduler.waiting.first?.id, youngest.id, "requeued at the front")
        XCTAssertNil(capacity.reserved[youngest.id], "reservations released")
        XCTAssertTrue(capacity.releaseAllCalls.contains(youngest.id))
    }

    /// The refund path (vLLM scheduler.py:550-567): the victim already got an
    /// assignment EARLIER in the same plan; preempting it must return that
    /// budget and drop the zeroed assignment.
    ///
    /// Setup puts the low-priority row FIRST in running order (admitted a
    /// step earlier), so it reserves successfully before the high-priority
    /// row's reserve throws.
    func testVictimWithSamePlanAssignmentIsRefunded() throws {
        let capacity = CBv2SchedMockCapacity(tokenLimit: 33)
        let scheduler = makeScheduler(capacity: capacity)
        let low = CBv2SchedFixtures.request(
            prompt: Array(repeating: 1, count: 10), maxTokens: 8, priority: 0)
        try scheduler.enqueue(low)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // low running, 10 reserved

        let high = CBv2SchedFixtures.request(
            prompt: Array(repeating: 2, count: 19), maxTokens: 8, priority: 5)
        try scheduler.enqueue(high)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // low +1, high 19 → 30
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // both decode → 32

        // Plan: low (running[0]) reserves 33/33 ✓ and is ASSIGNED; high
        // throws at 34 → victim = low (lowest priority) ≠ requester → low's
        // fresh assignment is refunded/zeroed, its tokens released, and
        // high's reserve succeeds on retry.
        let plan = scheduler.plan()
        XCTAssertEqual(plan.preemptions, [low.id])
        XCTAssertEqual(plan.assignments.map(\.id), [high.id])
        XCTAssertEqual(scheduler.record(for: low.id)?.status, .preempted)
        XCTAssertNil(capacity.reserved[low.id])
        XCTAssertEqual(scheduler.waiting.first?.id, low.id)
        assertInvariants(scheduler, plan: plan)
    }

    /// vLLM scheduler.py:573-575: when the victim IS the requester,
    /// scheduling stops for the step.
    func testSelfPreemptionStopsScheduling() throws {
        let capacity = CBv2SchedMockCapacity(tokenLimit: 32)
        let scheduler = makeScheduler(capacity: capacity)
        let low = CBv2SchedFixtures.request(
            prompt: Array(repeating: 1, count: 10), maxTokens: 8, priority: 0)
        try scheduler.enqueue(low)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // 10 reserved

        let high = CBv2SchedFixtures.request(
            prompt: Array(repeating: 2, count: 19), maxTokens: 8, priority: 5)
        try scheduler.enqueue(high)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // low +1, high 19 → 30
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // both decode → 32

        // low (running[0], lowest priority) throws on its own reserve at 33
        // → it is its own victim → self-preempt and STOP: high gets nothing
        // this step even though low's release made room.
        let plan = scheduler.plan()
        XCTAssertEqual(plan.preemptions, [low.id])
        XCTAssertTrue(plan.assignments.isEmpty)

        // Next plan: high decodes with the freed capacity; the preempted
        // victim is admitted again only when capacity allows.
        let next = scheduler.plan()
        XCTAssertEqual(next.assignments.first?.id, high.id)
        assertInvariants(scheduler, plan: next)
    }

    func testNoWaitingAdmissionInPreemptionStep() throws {
        let capacity = CBv2SchedMockCapacity(tokenLimit: 20)
        let scheduler = makeScheduler(capacity: capacity)
        let a = CBv2SchedFixtures.request(prompt: Array(repeating: 1, count: 10), maxTokens: 8)
        let b = CBv2SchedFixtures.request(prompt: Array(repeating: 2, count: 9), maxTokens: 8)
        try scheduler.enqueue(a)
        try scheduler.enqueue(b)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // 19/20

        let waiting = CBv2SchedFixtures.request(prompt: [9], maxTokens: 1)
        try scheduler.enqueue(waiting)

        // a decodes to 20/20; b throws → b self-preempts (youngest, same
        // priority). The waiting request must NOT be admitted this step.
        let plan = scheduler.plan()
        XCTAssertEqual(plan.preemptions, [b.id])
        XCTAssertFalse(plan.assignments.contains { $0.id == waiting.id })
    }

    // MARK: Optimistic advance rollback

    func testOptimisticAdvanceRollbackRestoresStateAndReservations() throws {
        let capacity = CBv2SchedMockCapacity()
        let scheduler = makeScheduler(capacity: capacity)
        let request = CBv2SchedFixtures.request(prompt: Array(0 ..< 100), maxTokens: 4)
        try scheduler.enqueue(request)

        let plan = scheduler.plan()
        XCTAssertEqual(scheduler.record(for: request.id)?.numComputedTokens, 100)
        XCTAssertEqual(capacity.reserved[request.id], 100)

        scheduler.rollback(plan)
        XCTAssertEqual(scheduler.record(for: request.id)?.numComputedTokens, 0)
        XCTAssertNil(capacity.reserved[request.id])

        // Re-planning reproduces the same assignment.
        let replan = scheduler.plan()
        XCTAssertEqual(replan.assignments.map(\.id), plan.assignments.map(\.id))
        XCTAssertEqual(replan.assignments.map(\.numTokens), plan.assignments.map(\.numTokens))
    }

    // MARK: Cancellation

    func testCancelMarkedRowsAreNeverScheduled() throws {
        let scheduler = makeScheduler()
        let decoding = CBv2SchedFixtures.request(prompt: [1, 2], maxTokens: 10)
        let prefilling = CBv2SchedFixtures.request(prompt: Array(0 ..< 600), maxTokens: 10)
        let queued = CBv2SchedFixtures.request(prompt: [5], maxTokens: 10)
        try scheduler.enqueue(decoding)
        try scheduler.enqueue(prefilling)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // decoding done, prefilling mid-chunk
        try scheduler.enqueue(queued)

        // Cancel one in each state: decoding (running), prefilling
        // (running, mid-prompt), queued (waiting).
        scheduler.requestCancel(decoding.id)
        scheduler.requestCancel(prefilling.id)
        scheduler.requestCancel(queued.id)

        let plan = scheduler.plan()
        XCTAssertTrue(plan.assignments.isEmpty, "cancel-pending rows must not be scheduled")

        // Engine-boundary drop: finish removes them in any state.
        for id in [decoding.id, prefilling.id, queued.id] {
            XCTAssertNotNil(scheduler.finish(id: id, reason: .cancelled))
            XCTAssertNil(scheduler.record(for: id))
        }
        XCTAssertEqual(scheduler.runningCount, 0)
        XCTAssertEqual(scheduler.waitingCount, 0)
    }

    // MARK: Chain eligibility

    func testChainCandidateIDsPureDecodeMembership() throws {
        let scheduler = makeScheduler(maxConcurrent: 2)
        let a = CBv2SchedFixtures.request(prompt: [1], maxTokens: 10)
        let b = CBv2SchedFixtures.request(prompt: [2], maxTokens: 10)
        try scheduler.enqueue(a)
        try scheduler.enqueue(b)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())

        // Both decode-ready, no waiting → chainable, in running order.
        XCTAssertEqual(scheduler.chainCandidateIDs(), [a.id, b.id])

        // A join possibility breaks the chain… only when a slot is free.
        let c = CBv2SchedFixtures.request(prompt: [3], maxTokens: 10)
        try scheduler.enqueue(c)
        XCTAssertEqual(
            scheduler.chainCandidateIDs(), [a.id, b.id],
            "no free slot ⇒ waiting request cannot join ⇒ chain holds")
        scheduler.finish(id: b.id, reason: .length)
        XCTAssertNil(scheduler.chainCandidateIDs(), "free slot + waiting ⇒ join possible ⇒ break")
    }

    func testChainCandidateIDsBreaksOnPrefillCancelAndSkipsPaused() throws {
        let scheduler = makeScheduler()
        let decode = CBv2SchedFixtures.request(prompt: [1], maxTokens: 10)
        let prefill = CBv2SchedFixtures.request(prompt: Array(0 ..< 600), maxTokens: 10)
        try scheduler.enqueue(decode)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
        XCTAssertEqual(scheduler.chainCandidateIDs(), [decode.id])

        try scheduler.enqueue(prefill)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // prefill mid-chunk now
        XCTAssertNil(scheduler.chainCandidateIDs(), "mid-prefill row ⇒ not pure decode")

        // Pause the mid-prefill row: the remaining decode row can chain.
        scheduler.pause(prefill.id)
        XCTAssertEqual(scheduler.chainCandidateIDs(), [decode.id])

        scheduler.requestCancel(decode.id)
        XCTAssertNil(scheduler.chainCandidateIDs(), "pending cancel ⇒ membership will change")
    }

    // MARK: Pause / resume

    func testPausedRowKeepsSlotButIsNotScheduled() throws {
        let scheduler = makeScheduler(maxConcurrent: 1)
        let paused = CBv2SchedFixtures.request(prompt: [1], maxTokens: 10)
        let queued = CBv2SchedFixtures.request(prompt: [2], maxTokens: 10)
        try scheduler.enqueue(paused)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
        try scheduler.enqueue(queued)

        scheduler.pause(paused.id)
        let plan = scheduler.plan()
        // Slot retained: queued cannot be admitted, paused not scheduled.
        XCTAssertTrue(plan.assignments.isEmpty)
        XCTAssertEqual(scheduler.runningCount, 1)

        scheduler.resume(paused.id)
        XCTAssertEqual(scheduler.plan().assignments.map(\.id), [paused.id])
    }

    /// PR#62 review: backpressure must survive preemption. A request paused
    /// for stream backpressure that is demoted back to WAITING keeps
    /// `isPaused`, and waiting admission must skip it — not re-admit it and
    /// decode into an overwhelmed consumer — until `resume` clears the
    /// flag. Skipping is per-row: unpaused rows behind it still admit, and
    /// the skipped row costs no budget and no capacity reservation.
    func testPausedPreemptedRequestNotReadmittedUntilResumed() throws {
        let capacity = CBv2SchedMockCapacity()
        let scheduler = makeScheduler(maxConcurrent: 2, capacity: capacity)
        let slow = CBv2SchedFixtures.request(prompt: [1], maxTokens: 4)
        try scheduler.enqueue(slow)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())  // prefill + sample

        // Consumer over the high watermark → paused; then demoted back to
        // waiting (capacity requeue / preemption preserves the flag).
        scheduler.pause(slow.id)
        XCTAssertTrue(scheduler.requeueOnCapacity(slow.id))
        XCTAssertEqual(scheduler.record(for: slow.id)?.isPaused, true)
        XCTAssertEqual(scheduler.waiting.first?.id, slow.id)

        // A later, unpaused arrival behind the paused head must admit; the
        // paused row stays waiting with nothing reserved and no progress.
        let eager = CBv2SchedFixtures.request(prompt: [2], maxTokens: 4)
        try scheduler.enqueue(eager)
        let plan = scheduler.plan()
        assertInvariants(scheduler, plan: plan)
        XCTAssertEqual(plan.assignments.map(\.id), [eager.id])
        XCTAssertTrue(
            scheduler.waiting.contains { $0.id == slow.id },
            "paused row must remain in waiting")
        XCTAssertNil(capacity.reserved[slow.id], "skipped row must reserve nothing")
        XCTAssertEqual(scheduler.record(for: slow.id)?.numComputedTokens, 0)
        CBv2SchedSim.confirm(scheduler, plan: plan)

        // Still paused on subsequent plans.
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
        XCTAssertTrue(scheduler.waiting.contains { $0.id == slow.id })

        // Resume → re-admitted and runs to completion.
        scheduler.resume(slow.id)
        var resumedPlan = scheduler.plan()
        XCTAssertTrue(
            resumedPlan.assignments.contains { $0.id == slow.id },
            "resumed request must be re-admitted")
        var guardStep = 0
        while scheduler.record(for: slow.id)!.generatedTokenCount < 4 {
            CBv2SchedSim.confirm(scheduler, plan: resumedPlan)
            resumedPlan = scheduler.plan()
            guardStep += 1
            XCTAssertLessThan(guardStep, 32, "resumed request must make progress")
        }
        XCTAssertEqual(scheduler.record(for: slow.id)?.generatedTokenCount, 4)
    }

    // MARK: Poisson simulation

    func testPoissonSimulationHoldsInvariantsAndCompletesEverything() throws {
        let preemptions = try runPoissonSimulation(tokenLimit: 2048, seed: 0xB5C0_FFEE)
        XCTAssertEqual(preemptions, 0, "roomy capacity should never preempt")
    }

    func testPoissonSimulationTightCapacityPreemptsAndStillCompletes() throws {
        // Worst-case single request = 300 prompt + 24 generated = 324 ≤ 400,
        // so everything can finish — but 4 concurrent cannot coexist, so the
        // preemption backstop must fire and the sim must still drain.
        let preemptions = try runPoissonSimulation(tokenLimit: 400, seed: 0xB5C0_FFEE)
        XCTAssertGreaterThan(preemptions, 0, "tight capacity must exercise preemption")
    }

    /// Returns the number of preemptions observed.
    private func runPoissonSimulation(tokenLimit: Int, seed: UInt64) throws -> Int {
        var rng = CBv2SchedSplitMix64(seed: seed)
        let capacity = CBv2SchedMockCapacity(tokenLimit: tokenLimit)
        let scheduler = makeScheduler(
            maxConcurrent: 4, budget: 256, chunk: 64, maxWaiting: 128, capacity: capacity)

        var submitted = 0
        var finished = 0
        var preemptions = 0
        let totalToSubmit = 60
        var steps = 0

        while steps < 50000, submitted < totalToSubmit || scheduler.hasWork {
            steps += 1
            // Poisson-ish arrivals: p = 0.15 per step, bursty prompt sizes.
            if submitted < totalToSubmit, rng.next() % 100 < 15 {
                let promptLen = 1 + Int(rng.next() % 300)
                let maxTokens = 1 + Int(rng.next() % 24)
                let priority = Int(rng.next() % 3)
                try scheduler.enqueue(
                    CBv2SchedFixtures.request(
                        prompt: Array(repeating: 7, count: promptLen), maxTokens: maxTokens,
                        priority: priority))
                submitted += 1
            }

            let plan = scheduler.plan()
            assertInvariants(scheduler, plan: plan)
            preemptions += plan.preemptions.count
            for id in plan.preemptions {
                XCTAssertNil(capacity.reserved[id], "victim reservations must be released")
                XCTAssertEqual(scheduler.record(for: id)?.numComputedTokens, 0)
            }

            // Execute: confirm samples, finish at maxTokens (mirrors loop).
            for (id, _) in plan.assignments {
                guard let rec = scheduler.record(for: id) else { continue }
                if rec.numComputedTokens == rec.effectiveTokenCount {
                    scheduler.markPendingSamples(ids: [id])
                    scheduler.recordSampled(id: id, token: 1)
                    if rec.generatedTokenCount >= rec.request.maxTokens {
                        scheduler.finish(id: id, reason: .length)
                        capacity.releaseAll(id: id)
                        finished += 1
                    }
                }
            }
        }

        XCTAssertEqual(submitted, totalToSubmit)
        XCTAssertEqual(finished, totalToSubmit, "every request must complete")
        XCTAssertEqual(scheduler.runningCount, 0)
        XCTAssertEqual(scheduler.waitingCount, 0)
        XCTAssertEqual(capacity.totalReserved, 0, "all reservations returned")
        return preemptions
    }
}
