// Tests for GenerationBatch tensor shapes — §14 of ContinuousBatchingTestPlan.md

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class CBGenerationBatchShapeTests: XCTestCase {

    private func makeBatchCache(batchSize: Int, numLayers: Int = 1) -> [any BatchedCache] {
        (0 ..< numLayers).map { _ in BatchKVCache(leftPadding: Array(repeating: 0, count: batchSize)) }
    }

    // MARK: - Single-row decode input shape

    func testGenerationBatchDecodeInputShape() {
        let model = IncrTestLanguageModel()
        let cache = makeBatchCache(batchSize: 1)

        let batch = GenerationBatch(
            model: model,
            uids: [0],
            seedTokens: MLXArray([UInt32(1)]),
            promptCache: cache,
            tokens: [[1]],
            maxTokens: [5]
        )

        // next() must return exactly one response without shape errors.
        let responses = batch.next()
        XCTAssertEqual(responses.count, 1)
        let tok = responses[0].token
        XCTAssertGreaterThanOrEqual(tok, 0)
        XCTAssertLessThan(tok, model.vocabularySize)
    }

    // MARK: - Multi-row logit extraction

    func testGenerationBatchLogitExtractionPerRow() {
        // Two rows. IncrTestLanguageModel deterministically outputs (input+1)%16.
        // Row 0: seed=2, expect token=3.
        // Row 1: seed=7, expect token=8.
        let model = IncrTestLanguageModel()
        let cache = makeBatchCache(batchSize: 2)

        let batch = GenerationBatch(
            model: model,
            uids: [0, 1],
            seedTokens: MLXArray([UInt32(2), UInt32(7)]),
            promptCache: cache,
            tokens: [[2], [7]],
            maxTokens: [10, 10]
        )

        let responses = batch.next()
        XCTAssertEqual(responses.count, 2)

        let tokenByUID = Dictionary(uniqueKeysWithValues: responses.map { ($0.uid, $0.token) })
        XCTAssertEqual(tokenByUID[0], 3, "row 0 (seed 2) → token 3")
        XCTAssertEqual(tokenByUID[1], 8, "row 1 (seed 7) → token 8")
    }
}
