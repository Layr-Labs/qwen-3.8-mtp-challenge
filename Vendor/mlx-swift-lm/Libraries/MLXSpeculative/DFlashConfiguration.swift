// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

public enum DFlashLayerType: String, Codable, Sendable, Equatable {
    case fullAttention = "full_attention"
    case slidingAttention = "sliding_attention"
}

public enum DFlashDecoderLayerType: String, Codable, Sendable, Equatable {
    case qwen3
    case lagunaXS = "laguna_xs"
}

public struct DFlashConfiguration: Codable, Sendable, Equatable {
    public struct Metadata: Codable, Sendable, Equatable {
        public var targetLayerIds: [Int]
        public var maskTokenId: Int
        public var ignoredConfigKeys: [String]

        enum CodingKeys: String, CodingKey, CaseIterable {
            case targetLayerIds = "target_layer_ids"
            case maskTokenId = "mask_token_id"
            case ignoredConfigKeys = "_mlx_ignored_config_keys"
        }

        public init(targetLayerIds: [Int], maskTokenId: Int) {
            self.targetLayerIds = targetLayerIds
            self.maskTokenId = maskTokenId
            self.ignoredConfigKeys = []
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.targetLayerIds = try container.decode([Int].self, forKey: .targetLayerIds)
            self.maskTokenId = try container.decodeIfPresent(Int.self, forKey: .maskTokenId) ?? 0
            self.ignoredConfigKeys = try Self.unknownKeys(from: decoder)
        }

        private static func unknownKeys(from decoder: Decoder) throws -> [String] {
            let container = try decoder.container(keyedBy: DFlashJSONKey.self)
            let known = Set(CodingKeys.allCases.map(\.stringValue))
            return container.allKeys.map(\.stringValue).filter { !known.contains($0) }.sorted()
        }
    }

    public var architectures: [String]
    public var modelType: String
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var vocabularySize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int
    public var blockSize: Int
    public var numTargetLayers: Int
    public var dflashConfig: Metadata
    public var ropeScaling: [String: StringOrNumber]?
    public var layerTypes: [DFlashLayerType]
    public var slidingWindow: Int?
    public var finalLogitSoftcapping: Float?
    public var tieWordEmbeddings: Bool
    public var decoderLayerType: DFlashDecoderLayerType
    public var gating: String?
    public var ignoredConfigKeys: [String]

    public var targetLayerIds: [Int] { dflashConfig.targetLayerIds }
    public var maskTokenId: Int { dflashConfig.maskTokenId }
    public var targetHiddenSize: Int { targetLayerIds.count * hiddenSize }
    public var recommendedBlockSize: Int {
        guard layerTypes.contains(.slidingAttention), let slidingWindow else {
            return blockSize
        }
        return Swift.min(blockSize, Swift.max(2, slidingWindow))
    }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case architectures
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case blockSize = "block_size"
        case numTargetLayers = "num_target_layers"
        case dflashConfig = "dflash_config"
        case ropeScaling = "rope_scaling"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case decoderLayerType = "decoder_layer_type"
        case gating
        case ignoredConfigKeys = "_mlx_ignored_config_keys"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures)
            ?? ["DFlashDraftModel"]
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.blockSize = try container.decode(Int.self, forKey: .blockSize)
        guard blockSize >= 2 else {
            throw DecodingError.dataCorruptedError(
                forKey: .blockSize,
                in: container,
                debugDescription: "DFlash block_size must be at least 2.")
        }
        self.numTargetLayers =
            try container.decodeIfPresent(Int.self, forKey: .numTargetLayers)
            ?? hiddenLayers
        self.dflashConfig = try container.decode(Metadata.self, forKey: .dflashConfig)
        guard dflashConfig.maskTokenId >= 0 && dflashConfig.maskTokenId < vocabularySize else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflashConfig,
                in: container,
                debugDescription:
                    "DFlash mask_token_id \(dflashConfig.maskTokenId) is outside 0..<\(vocabularySize).")
        }
        let targetLayerIds = dflashConfig.targetLayerIds
        guard !targetLayerIds.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflashConfig,
                in: container,
                debugDescription: "DFlash target_layer_ids must not be empty.")
        }
        guard Set(targetLayerIds).count == targetLayerIds.count else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflashConfig,
                in: container,
                debugDescription: "DFlash target_layer_ids must be unique.")
        }
        var invalidLayer: Int?
        for layerId in targetLayerIds where layerId < 0 || layerId >= numTargetLayers {
            invalidLayer = layerId
            break
        }
        if let invalidLayer {
            throw DecodingError.dataCorruptedError(
                forKey: .dflashConfig,
                in: container,
                debugDescription:
                    "DFlash target layer id \(invalidLayer) is outside 0..<\(numTargetLayers).")
        }
        self.ropeScaling =
            try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.decoderLayerType =
            try container.decodeIfPresent(DFlashDecoderLayerType.self, forKey: .decoderLayerType)
            ?? .qwen3
        self.gating = try container.decodeIfPresent(String.self, forKey: .gating)
        if decoderLayerType == .lagunaXS, let gating, gating != "per-head" {
            throw DecodingError.dataCorruptedError(
                forKey: .gating, in: container,
                debugDescription:
                    "DFlash laguna_xs drafters require per-head gating; got \(gating).")
        }
        self.ignoredConfigKeys = try Self.unknownKeys(from: decoder)

        let decodedLayerTypes =
            try container.decodeIfPresent([DFlashLayerType].self, forKey: .layerTypes)
            ?? Array(repeating: .fullAttention, count: hiddenLayers)
        guard decodedLayerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes,
                in: container,
                debugDescription:
                    "DFlash layer_types count \(decodedLayerTypes.count) must match num_hidden_layers \(hiddenLayers).")
        }
        if decodedLayerTypes.contains(.slidingAttention), slidingWindow == nil {
            throw DecodingError.dataCorruptedError(
                forKey: .slidingWindow,
                in: container,
                debugDescription:
                    "DFlash configs with sliding_attention layers must define sliding_window.")
        }
        self.layerTypes = decodedLayerTypes
    }

    private static func unknownKeys(from decoder: Decoder) throws -> [String] {
        let container = try decoder.container(keyedBy: DFlashJSONKey.self)
        let known = Set(CodingKeys.allCases.map(\.stringValue))
        return container.allKeys.map(\.stringValue).filter { !known.contains($0) }.sorted()
    }
}

private struct DFlashJSONKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil

    init?(stringValue: String) {
        self.stringValue = stringValue
    }

    init?(intValue: Int) {
        return nil
    }
}
