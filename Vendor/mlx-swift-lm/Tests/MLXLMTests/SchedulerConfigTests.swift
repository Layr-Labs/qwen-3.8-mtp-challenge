// Tests for Scheduler.setMaxNumSeqs(_:) — the runtime concurrency cap
// added to support adaptive batching policies that live above the engine.
//
// These tests intentionally bypass EngineCore and exercise Scheduler
// directly so they don't need a live model loop. The serialisation
// contract (engine queue) is documented on the API; here we verify the
// state-level behaviour.

import Foundation
import XCTest

@testable import MLXLMCommon

final class SchedulerConfigTests: XCTestCase {

    func testSetMaxNumSeqsClampsBelowOne() {
        // The init path does NOT clamp (preserves the `maxNumSeqs = 0`
        // sentinel that blocks all admission). Only the runtime setter
        // clamps to prevent permanent admission deadlock.
        let s = makeTestScheduler()
        s.setMaxNumSeqs(0)
        XCTAssertEqual(
            s.getStats()["max_num_seqs"] as? Int, 1, "0 must clamp to 1")
        s.setMaxNumSeqs(-5)
        XCTAssertEqual(
            s.getStats()["max_num_seqs"] as? Int, 1, "negative must clamp to 1")
    }

    func testSetMaxNumSeqsUpdatesValue() {
        let s = makeTestScheduler()
        XCTAssertEqual(
            s.getStats()["max_num_seqs"] as? Int, 4,
            "default from test fixture is 4")
        s.setMaxNumSeqs(2)
        if let v = s.getStats()["max_num_seqs"] as? Int {
            XCTAssertEqual(v, 2, "getStats() must surface the runtime cap")
        } else {
            XCTFail("getStats() missing max_num_seqs key")
        }
    }

    /// After lowering the cap to 2, admitting 3 requests must leave 2
    /// running and 1 waiting. The third request is held until a slot
    /// frees up.
    func testRuntimeCapGatesAdmission() {
        let s = makeTestScheduler(eosTokenIds: [])
        s.setMaxNumSeqs(2)

        s.addRequest(makeIntRequest(id: "r1", tokens: [1], maxTokens: 20))
        s.addRequest(makeIntRequest(id: "r2", tokens: [6], maxTokens: 20))
        s.addRequest(makeIntRequest(id: "r3", tokens: [9], maxTokens: 20))

        // First step admits and starts chunked prefill for the first 2
        // requests; the third stays in the waiting queue.
        _ = s.step()
        // Second step promotes the cold pending-prefill batch into the
        // generation batch.
        _ = s.step()

        XCTAssertEqual(s.getNumRunning(), 2, "cap=2 limits running to 2")
        XCTAssertEqual(s.getNumWaiting(), 1, "third request must remain waiting")
    }

    /// Raising the cap from 1 to 3 must allow a previously-blocked request
    /// to be admitted on a subsequent step.
    func testRuntimeCapRaisingAllowsAdmission() {
        let s = makeTestScheduler(eosTokenIds: [])
        s.setMaxNumSeqs(1)

        s.addRequest(makeIntRequest(id: "r1", tokens: [1], maxTokens: 20))
        s.addRequest(makeIntRequest(id: "r2", tokens: [6], maxTokens: 20))

        _ = s.step()  // admit r1 (chunked prefill kicks off)
        _ = s.step()  // promote r1 to genBatch
        XCTAssertEqual(s.getNumRunning(), 1)
        XCTAssertEqual(s.getNumWaiting(), 1)

        s.setMaxNumSeqs(3)
        _ = s.step()  // pendingPrefill should now admit r2
        _ = s.step()  // promote r2 to genBatch
        XCTAssertEqual(s.getNumRunning(), 2, "raising the cap must let r2 in")
        XCTAssertEqual(s.getNumWaiting(), 0)
    }
}
