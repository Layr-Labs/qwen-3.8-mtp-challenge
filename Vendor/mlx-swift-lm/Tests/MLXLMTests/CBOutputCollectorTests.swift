// Tests for OutputCollector.swift — §2 of ContinuousBatchingTestPlan.md

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBOutputCollectorTests: XCTestCase {

    private func makeOutput(id: String = "r1", tokens: [Int] = [1], finished: Bool = false) -> RequestOutput {
        RequestOutput(
            requestId: id,
            newTokenIds: tokens,
            newText: tokens.map { String($0) }.joined(),
            finished: finished
        )
    }

    // MARK: - getNowait

    func testCollectorGetNowaitReturnsPutOutput() {
        let col = RequestOutputCollector(aggregate: false)
        let out = makeOutput(tokens: [42])
        col.put(out)

        let got = col.getNowait()
        XCTAssertEqual(got?.newTokenIds, [42])
        XCTAssertNil(col.getNowait(), "second getNowait must return nil")
    }

    // MARK: - Aggregation

    func testCollectorAggregationMergesTokenLists() {
        let col = RequestOutputCollector(aggregate: true)
        let a = makeOutput(tokens: [1, 2])
        let b = makeOutput(tokens: [3, 4])
        col.put(a)
        col.put(b)

        let merged = col.getNowait()
        XCTAssertEqual(merged?.newTokenIds, [1, 2, 3, 4])
        XCTAssertNil(col.getNowait())
    }

    func testCollectorNoAggregationOverwritesOutput() {
        let col = RequestOutputCollector(aggregate: false)
        col.put(makeOutput(tokens: [1]))
        col.put(makeOutput(tokens: [2]))

        XCTAssertEqual(col.getNowait()?.newTokenIds, [2])
    }

    // MARK: - Async get

    func testCollectorGetBlocksUntilPut() async {
        let col = RequestOutputCollector(aggregate: false)
        let expected = makeOutput(tokens: [99])

        let getTask = Task { await col.get() }

        // Brief yield so getTask reaches its suspension point.
        await Task.yield()

        col.put(expected)

        let result = await getTask.value
        XCTAssertEqual(result.newTokenIds, [99])
    }

    func testCollectorNoDuplicateDeliveryAfterContinuationResume() async {
        // Regression: put() must NOT buffer AND resume continuation.
        let col = RequestOutputCollector(aggregate: false)
        let out = makeOutput(tokens: [7])

        let getTask = Task { await col.get() }
        await Task.yield()

        col.put(out)

        _ = await getTask.value

        // After direct delivery, buffer must be empty.
        XCTAssertNil(col.getNowait())
    }

    // MARK: - clear

    func testCollectorClearDiscardsPendingOutput() {
        let col = RequestOutputCollector(aggregate: false)
        col.put(makeOutput(tokens: [5]))
        col.clear()
        XCTAssertNil(col.getNowait())
    }

    func testCollectorGetAfterClearResumesClearedContinuation() async {
        // put then clear — a subsequent get should block until a new put arrives.
        let col = RequestOutputCollector(aggregate: false)
        col.put(makeOutput(tokens: [1]))
        col.clear()

        let getTask = Task { await col.get() }
        await Task.yield()

        let fresh = makeOutput(tokens: [2])
        col.put(fresh)

        let result = await getTask.value
        XCTAssertEqual(result.newTokenIds, [2])
    }
}
