import Foundation
import MLX
import MLXFastModel
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

// These tests construct tiny CPU-only models through the real factory. They
// exercise loading/sanitization, not generation or a downloaded checkpoint.
@Suite(.serialized)
struct MTPWrappedLoaderTests {
    @Test("Raw qwen3_5 factory retains the declared draft head and precision islands", arguments: [false, true])
    func wrappedHeadLoads(alreadyWrapped: Bool) async throws {
        try await Device.withDefaultDevice(.cpu) {
            let previous = _qwen35MTPEnabled
            _qwen35MTPEnabled = true
            defer { _qwen35MTPEnabled = previous }
            let factoryModel = try await LLMModelFactory.shared.typeRegistry.createModel(
                configuration: Self.configuration(wrapped: true), modelType: "qwen3_5")
            let model = try #require(factoryModel as? MLXLLM.Qwen35Model)
            let target = try #require(factoryModel as? any Qwen36MTPTarget)
            #expect(target.hasMTPHead)

            let base = MLXArray.full([16], values: MLXArray(Float(7)))
            let fc = MLXArray.zeros([16, 32])
            let draftW = MLXArray.zeros([8, 2], dtype: .uint32)
            let draftS = MLXArray.ones([8, 1])
            let draftZ = MLXArray.zeros([8, 1])
            let prefix = alreadyWrapped ? "language_model." : ""
            let weights: [String: MLXArray] = [
                "language_model.model.norm.weight": base,
                prefix + "mtp.fc.weight": fc,
                prefix + "mtp.draft_lm_head.weight": draftW,
                prefix + "mtp.draft_lm_head.scales": draftS,
                prefix + "mtp.draft_lm_head.biases": draftZ,
                prefix + "mtp.precision_islands.q.weight": MLXArray.full([1, 16], values: MLXArray(Float(2))),
                prefix + "mtp.precision_islands.q.indices": MLXArray([Int32(0)]),
                prefix + "mtp.precision_islands.k.weight": MLXArray.full([1, 16], values: MLXArray(Float(3))),
                prefix + "mtp.precision_islands.k.indices": MLXArray([Int32(0)]),
                prefix + "mtp.precision_islands.v.weight": MLXArray.full([1, 16], values: MLXArray(Float(4))),
                prefix + "mtp.precision_islands.v.indices": MLXArray([Int32(0)]),
            ]
            let sanitized = model.sanitize(weights: weights)
            #expect(Set(sanitized.keys) == ["language_model.model.norm.weight", "language_model.mtp.fc.weight"])
            #expect(sanitized["language_model.model.norm.weight"] === base)
            #expect(sanitized["language_model.mtp.fc.weight"] === fc)

            // Strict update must accept the real parameter namespace. Retaining
            // the non-parameter side-channel keys reproduces the GUI failure.
            try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.noUnusedKeys])
            let text = try #require(model.namedModules().first { $0.0 == "language_model" }?.1)
            #expect(Mirror(reflecting: text).descendant("_draftHeadW") as? MLXArray === draftW)
            #expect(Mirror(reflecting: text).descendant("_draftHeadS") as? MLXArray === draftS)
            #expect(Mirror(reflecting: text).descendant("_draftHeadZ") as? MLXArray === draftZ)
            let attention = try #require(model.namedModules().first { $0.0 == "language_model.mtp.layers.0.self_attn" }?.1)
            let islandWeight = try #require(Mirror(reflecting: attention).descendant("_exactQKVWeight") as? MLXArray)
            let islandIndices = try #require(Mirror(reflecting: attention).descendant("_exactQKVIndices") as? MLXArray)
            #expect(islandWeight.asArray(Float.self) == Array(repeating: 2, count: 16) + Array(repeating: 3, count: 16) + Array(repeating: 4, count: 16))
            #expect(islandIndices.asArray(Int32.self) == [0, 12_288, 13_312])
        }
    }

    @Test("The head-disabled wrapper keeps base tensors and its existing key normalization")
    func headDisabledControl() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let previous = _qwen35MTPEnabled
            _qwen35MTPEnabled = false
            defer { _qwen35MTPEnabled = previous }
            let factoryModel = try await LLMModelFactory.shared.typeRegistry.createModel(
                configuration: Self.configuration(wrapped: true), modelType: "qwen3_5")
            let model = try #require(factoryModel as? MLXLLM.Qwen35Model)
            #expect(!model.hasMTPHead)
            let norm = MLXArray.ones([16])
            let embedding = MLXArray.zeros([32, 16])
            let head = MLXArray.zeros([32, 16])
            let weights = model.sanitize(weights: [
                "model.language_model.norm.weight": norm,
                "language_model.model.embed_tokens.weight": embedding,
                "lm_head.weight": head,
                "mtp.draft_lm_head.weight": MLXArray.zeros([8, 2], dtype: .uint32),
                "vision_tower.ignored": norm,
            ])
            #expect(Set(weights.keys) == ["language_model.model.norm.weight", "language_model.model.embed_tokens.weight", "language_model.lm_head.weight"])
            #expect(weights["language_model.model.norm.weight"] === norm)
            #expect(weights["language_model.model.embed_tokens.weight"] === embedding)
            #expect(weights["language_model.lm_head.weight"] === head)
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        }
    }

    @Test("MTP wrapper preserves each base key/value and refuses unknown head parameters")
    func headEnabledBaseControl() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let previous = _qwen35MTPEnabled
            _qwen35MTPEnabled = true
            defer { _qwen35MTPEnabled = previous }
            let factoryModel = try await LLMModelFactory.shared.typeRegistry.createModel(
                configuration: Self.configuration(wrapped: true), modelType: "qwen3_5")
            let model = try #require(factoryModel as? MLXLLM.Qwen35Model)
            let norm = MLXArray.ones([16])
            let embedding = MLXArray.zeros([32, 16])
            let head = MLXArray.zeros([32, 16])
            let fc = MLXArray.zeros([16, 32])
            let weights = model.sanitize(weights: [
                "model.language_model.norm.weight": norm,
                "language_model.model.embed_tokens.weight": embedding,
                "lm_head.weight": head,
                "mtp.fc.weight": fc,
                "vision_tower.ignored": norm,
            ])
            #expect(Set(weights.keys) == ["language_model.model.norm.weight", "language_model.model.embed_tokens.weight", "language_model.lm_head.weight", "language_model.mtp.fc.weight"])
            #expect(weights["language_model.model.norm.weight"] === norm)
            #expect(weights["language_model.model.embed_tokens.weight"] === embedding)
            #expect(weights["language_model.lm_head.weight"] === head)
            #expect(weights["language_model.mtp.fc.weight"] === fc)
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
            let unknown = model.sanitize(weights: ["mtp.unsupported.weight": norm])
            #expect(unknown["language_model.mtp.unsupported.weight"] === norm)
            #expect(throws: (any Error).self) {
                try model.update(parameters: ModuleParameters.unflattened(unknown), verify: [.noUnusedKeys])
            }
        }
    }

    @Test("The winning bare qwen3_5_text loader retains its draft side channels")
    func bareHeadControl() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let previous = _qwen35MTPEnabled
            _qwen35MTPEnabled = true
            defer { _qwen35MTPEnabled = previous }
            let factoryModel = try await LLMModelFactory.shared.typeRegistry.createModel(
                configuration: Self.configuration(wrapped: false), modelType: "qwen3_5_text")
            let model = try #require(factoryModel as? Qwen35TextModel)
            let target = try #require(factoryModel as? any Qwen36MTPTarget)
            #expect(target.hasMTPHead)
            let draft = MLXArray.zeros([8, 2], dtype: .uint32)
            let fc = MLXArray.zeros([16, 32])
            let weights = model.sanitize(weights: ["mtp.draft_lm_head.weight": draft, "mtp.fc.weight": fc])
            #expect(Set(weights.keys) == ["mtp.fc.weight"])
            #expect(weights["mtp.fc.weight"] === fc)
            #expect(Mirror(reflecting: model).descendant("_draftHeadW") as? MLXArray === draft)
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        }
    }

    @Test("Ambiguous raw and wrapped names are refused before any tensor is selected", arguments: [
        ["mtp.fc.weight", "language_model.mtp.fc.weight"],
        ["model.language_model.norm.weight", "language_model.model.norm.weight"],
    ])
    func ambiguousNamesAreRefused(names: [String]) throws {
        try Device.withDefaultDevice(.cpu) {
            let first = MLXArray(Float(1))
            let second = MLXArray(Float(2))
            #expect(throws: (any Error).self) {
                try MLXLLM.Qwen35Model.mtpTextWeights(from: [names[0]: first, names[1]: second])
            }
        }
    }

    private static func configuration(wrapped: Bool) -> Data {
        let text = """
        {"model_type":"qwen3_5_text","hidden_size":16,"num_hidden_layers":1,
         "intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,
         "head_dim":8,"linear_num_value_heads":1,"linear_num_key_heads":1,
         "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,
         "full_attention_interval":1,"vocab_size":32,"tie_word_embeddings":false,
         "mtp_num_hidden_layers":1}
        """
        return Data((wrapped ? "{\"model_type\":\"qwen3_5\",\"text_config\":" + text + "}" : text).utf8)
    }
}
