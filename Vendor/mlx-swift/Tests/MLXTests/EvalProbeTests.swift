// Copyright © 2026 Eigen Labs.

import Foundation
import XCTest

@testable import MLX

/// Regression tests for `EvalProbe` (measurement-only instrumentation of the
/// blocking eval path used by the Darkbloom provider's first-token-wedge probe).
class EvalProbeTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// Fix 1: a lazy realization via `asArray()` blocks in `MLXArray.eval()`
    /// (wrapping `mlx_array_eval` under `evalLock`), so it MUST advance the probe.
    /// Before the fix only the free `eval(...)` functions were bracketed, so this
    /// read-path eval was invisible to `eval_in_flight_ms` / `evals_completed`.
    func testAsArrayRealizationAdvancesProbe() {
        let before = EvalProbe.evalsCompleted
        let a = MLXArray([1, 2, 3, 4]) + MLXArray([10, 20, 30, 40])
        XCTAssertEqual(a.asArray(Int32.self), [11, 22, 33, 44])
        XCTAssertGreaterThan(EvalProbe.evalsCompleted, before)
        // The realization returned, so nothing is in flight afterward.
        XCTAssertFalse(EvalProbe.inFlight)
        XCTAssertEqual(EvalProbe.currentEvalElapsedMs, 0)
    }

    /// Fix 1: `item()` also realizes through `MLXArray.eval()`.
    func testItemRealizationAdvancesProbe() {
        let before = EvalProbe.evalsCompleted
        let v: Int = (MLXArray([7]) * MLXArray([6])).item()
        XCTAssertEqual(v, 42)
        XCTAssertGreaterThan(EvalProbe.evalsCompleted, before)
    }

    /// Fix 2: the probe state is behind a lock, so a writer (evals on one queue)
    /// and many concurrent reads (heartbeat on another) are race-free. Run under
    /// the Thread Sanitizer this fails on the previous `nonisolated(unsafe)`
    /// scalars and passes with the `OSAllocatedUnfairLock` box.
    func testConcurrentReadsAreRaceFree() {
        let evalsDone = expectation(description: "evals done")
        let readsDone = expectation(description: "reads done")

        DispatchQueue.global().async {
            for _ in 0..<200 {
                let a = MLXArray([1, 2, 3, 4]) * MLXArray([2, 2, 2, 2])
                _ = a.asArray(Int32.self)
            }
            evalsDone.fulfill()
        }
        DispatchQueue.global().async {
            for _ in 0..<10000 {
                _ = EvalProbe.currentEvalElapsedMs
                _ = EvalProbe.inFlight
                _ = EvalProbe.evalsCompleted
                _ = EvalProbe.longestEvalMs
            }
            readsDone.fulfill()
        }

        wait(for: [evalsDone, readsDone], timeout: 60)
        // Counters never go backwards / negative.
        XCTAssertGreaterThanOrEqual(EvalProbe.evalsCompleted, 200)
        XCTAssertGreaterThanOrEqual(EvalProbe.longestEvalMs, 0)
        XCTAssertGreaterThanOrEqual(EvalProbe.currentEvalElapsedMs, 0)
    }
}
