// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite("LagunaModel forwardForDFlash")
struct LagunaDFlashForwardTests {

    private func tinyModel() throws -> LagunaModel {
        let config = try JSONDecoder.json5().decode(
            LagunaConfiguration.self, from: Data(LagunaModelTests.tinyConfigJSON.utf8))
        return LagunaModel(config)
    }

    @Test func exposesDFlashSurface() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            #expect(model.dFlashVocabularySize == 32)
            #expect(model.dFlashHiddenSize == 8)
            #expect(model.dFlashLayerCount == 4)
        }
    }

    @Test func logitsMatchPlainForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let plain = model(tokens, cache: model.newCache(parameters: nil))
            let dflash = try model.forwardForDFlash(
                tokens, cache: model.newCache(parameters: nil), targetLayerIds: [1])
            eval(plain, dflash.logits)
            #expect(allClose(plain, dflash.logits, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        }
    }

    @Test func capturesRequestedHiddenStatesInRequestedOrder() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let forward = try model.forwardForDFlash(
                tokens, cache: model.newCache(parameters: nil), targetLayerIds: [3, 1])
            #expect(forward.hiddenStates.count == 2)
            #expect(forward.hiddenStates.allSatisfy { $0.shape == [1, 3, 8] })
            #expect(forward.targetHidden.shape == [1, 3, 16])
            // Captured states are the post-layer residual stream: running the
            // same tokens with ids [1] must reproduce the second slot of [3, 1].
            let single = try model.forwardForDFlash(
                tokens, cache: model.newCache(parameters: nil), targetLayerIds: [1])
            eval(forward.hiddenStates[1], single.hiddenStates[0])
            #expect(
                allClose(forward.hiddenStates[1], single.hiddenStates[0],
                    rtol: 1e-4, atol: 1e-4).item(Bool.self))
        }
    }

    @Test func rejectsInvalidLayerIds() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            let tokens = MLXArray([Int32(1)])[.newAxis, .ellipsis]
            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    tokens, cache: nil, targetLayerIds: [7])
            }
            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    tokens, cache: nil, targetLayerIds: [])
            }
        }
    }
}
