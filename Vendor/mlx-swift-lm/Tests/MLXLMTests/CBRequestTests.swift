// Tests for Request.swift — §1 of ContinuousBatchingTestPlan.md

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBRequestStatusTests: XCTestCase {

    func testRequestStatusFinishedPredicateAndFinishReasons() {
        XCTAssertFalse(RequestStatus.waiting.isFinished)
        XCTAssertFalse(RequestStatus.running.isFinished)
        XCTAssertFalse(RequestStatus.preempted.isFinished)

        XCTAssertTrue(RequestStatus.finishedStopped.isFinished)
        XCTAssertEqual(RequestStatus.finishedStopped.finishReason, "stop")

        XCTAssertTrue(RequestStatus.finishedLengthCapped.isFinished)
        XCTAssertEqual(RequestStatus.finishedLengthCapped.finishReason, "length")

        XCTAssertTrue(RequestStatus.finishedAborted.isFinished)
        XCTAssertEqual(RequestStatus.finishedAborted.finishReason, "abort")

        XCTAssertNil(RequestStatus.waiting.finishReason)
        XCTAssertNil(RequestStatus.running.finishReason)
    }

    func testRequestStatusOrdering() {
        XCTAssertTrue(RequestStatus.waiting < RequestStatus.finishedStopped)
        XCTAssertTrue(RequestStatus.waiting < RequestStatus.running)
        XCTAssertFalse(RequestStatus.finishedStopped < RequestStatus.waiting)
    }

    func testSamplingParamsDefaults() {
        let p = SamplingParams()
        XCTAssertEqual(p.maxTokens, 256)
        XCTAssertEqual(p.temperature, 0.7, accuracy: 1e-6)
        XCTAssertEqual(p.topP, 0.9, accuracy: 1e-6)
        XCTAssertEqual(p.topK, 0)
        XCTAssertEqual(p.minP, 0.0, accuracy: 1e-6)
        XCTAssertEqual(p.repetitionPenalty, 1.0, accuracy: 1e-6)
        XCTAssertEqual(p.presencePenalty, 0.0, accuracy: 1e-6)
        XCTAssertEqual(p.frequencyPenalty, 0.0, accuracy: 1e-6)
        XCTAssertTrue(p.stop.isEmpty)
        XCTAssertTrue(p.stopTokenIds.isEmpty)
    }

    func testRequestAppendOutputToken() {
        let req = Request(requestId: "r1", prompt: "hello" as AnyHashable)
        XCTAssertEqual(req.outputTokenIds.count, 0)
        XCTAssertEqual(req.numComputedTokens, 0)

        req.appendOutputToken(42)
        XCTAssertEqual(req.outputTokenIds, [42])
        XCTAssertEqual(req.numComputedTokens, 1)

        req.appendOutputToken(7)
        XCTAssertEqual(req.outputTokenIds, [42, 7])
        XCTAssertEqual(req.numComputedTokens, 2)
    }

    func testRequestSetFinished() {
        let req = Request(requestId: "r1", prompt: "hello" as AnyHashable)
        req.setFinished(.finishedStopped)
        XCTAssertEqual(req.status, .finishedStopped)
        XCTAssertEqual(req.finishReason, "stop")
        XCTAssertTrue(req.isFinished)
    }

    func testRequestSetFinishedWithCustomReason() {
        let req = Request(requestId: "r2", prompt: "hello" as AnyHashable)
        req.setFinished(.finishedAborted, reason: "user_cancelled")
        XCTAssertEqual(req.status, .finishedAborted)
        XCTAssertEqual(req.finishReason, "user_cancelled")
    }
}
