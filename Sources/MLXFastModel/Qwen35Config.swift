import CoreFoundation
import Foundation
import MLXFastCore

public enum Qwen35LayerType: String, Equatable, Sendable {
    case linear = "linear_attention"
    case full = "full_attention"
}

public struct Qwen35RopeSpec: Equatable, Sendable {
    public let theta: Double
    public let type: String
    public let partialRotaryFactor: Double
    public let mropeInterleaved: Bool
    public let mropeSection: [Int]

    public init(
        theta: Double,
        type: String,
        partialRotaryFactor: Double,
        mropeInterleaved: Bool,
        mropeSection: [Int]
    ) {
        self.theta = theta
        self.type = type
        self.partialRotaryFactor = partialRotaryFactor
        self.mropeInterleaved = mropeInterleaved
        self.mropeSection = mropeSection
    }
}

/// Frozen text-tower contract for
/// `mlx-community/Qwen3.6-27B-4bit@c000ac2c2057d94be3fa931000c31723aac53282`.
///
/// The artifact is named Qwen3.6; its immutable internal architecture name is
/// `qwen3_5_text`.
public struct Qwen35Config: Equatable, Sendable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let linearNumValueHeads: Int
    public let linearNumKeyHeads: Int
    public let linearValueHeadDim: Int
    public let linearKeyHeadDim: Int
    public let linearConvKernelDim: Int
    public let fullAttentionInterval: Int
    public let layerTypes: [Qwen35LayerType]
    public let rmsNormEps: Double
    public let hiddenActivation: String
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    public let attentionDropout: Double
    public let attentionOutputGate: Bool
    public let outputGateType: String
    public let tieWordEmbeddings: Bool
    public let mambaSSMDType: String
    public let dtype: String
    public let useCache: Bool
    public let rope: Qwen35RopeSpec
    public let quantizationGroupSize: Int
    public let quantizationBits: Int
    public let quantizationMode: String
    /// Metadata only: the pinned checkpoint has no `mtp.*` tensors.
    public let mtpNumHiddenLayers: Int
    public let mtpUseDedicatedEmbeddings: Bool

    public static let expectedLayerTypes: [Qwen35LayerType] =
        (0..<MLXFastConstants.numHiddenLayers).map {
            $0 % 4 == 3 ? .full : .linear
        }

    public init(
        modelType: String,
        vocabSize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        linearNumValueHeads: Int,
        linearNumKeyHeads: Int,
        linearValueHeadDim: Int,
        linearKeyHeadDim: Int,
        linearConvKernelDim: Int,
        fullAttentionInterval: Int,
        layerTypes: [Qwen35LayerType],
        rmsNormEps: Double,
        hiddenActivation: String,
        maxPositionEmbeddings: Int,
        attentionBias: Bool,
        attentionDropout: Double,
        attentionOutputGate: Bool,
        outputGateType: String,
        tieWordEmbeddings: Bool,
        mambaSSMDType: String,
        dtype: String,
        useCache: Bool,
        rope: Qwen35RopeSpec,
        quantizationGroupSize: Int,
        quantizationBits: Int,
        quantizationMode: String,
        mtpNumHiddenLayers: Int,
        mtpUseDedicatedEmbeddings: Bool
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.linearNumValueHeads = linearNumValueHeads
        self.linearNumKeyHeads = linearNumKeyHeads
        self.linearValueHeadDim = linearValueHeadDim
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearConvKernelDim = linearConvKernelDim
        self.fullAttentionInterval = fullAttentionInterval
        self.layerTypes = layerTypes
        self.rmsNormEps = rmsNormEps
        self.hiddenActivation = hiddenActivation
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.attentionOutputGate = attentionOutputGate
        self.outputGateType = outputGateType
        self.tieWordEmbeddings = tieWordEmbeddings
        self.mambaSSMDType = mambaSSMDType
        self.dtype = dtype
        self.useCache = useCache
        self.rope = rope
        self.quantizationGroupSize = quantizationGroupSize
        self.quantizationBits = quantizationBits
        self.quantizationMode = quantizationMode
        self.mtpNumHiddenLayers = mtpNumHiddenLayers
        self.mtpUseDedicatedEmbeddings = mtpUseDedicatedEmbeddings
    }

    public static func load(from weightsPath: String) throws -> Qwen35Config {
        let path = URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json")
        try requireFile(path.path, description: "transformed weights config")

        let data = try Data(contentsOf: path)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("config.json must be a JSON object")
        }

        let ropeObject = try qwenObject("rope_parameters", in: root)
        let quantization = try qwenQuantization(in: root)

        // Reject unsafe counts before parsing the layer array.
        let numHiddenLayers = try qwenInt("num_hidden_layers", in: root)
        guard numHiddenLayers == MLXFastConstants.numHiddenLayers else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 config invariant check failed: num_hidden_layers="
                    + "\(numHiddenLayers) expected \(MLXFastConstants.numHiddenLayers)"
            )
        }

        let ropePartial = try qwenDouble("partial_rotary_factor", in: ropeObject)
        let topLevelPartial = try qwenDouble("partial_rotary_factor", in: root)
        guard topLevelPartial == ropePartial else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 config invariant check failed: top-level "
                    + "partial_rotary_factor=\(topLevelPartial) does not match "
                    + "rope_parameters partial_rotary_factor=\(ropePartial)"
            )
        }

        let config = Qwen35Config(
            modelType: try qwenString("model_type", in: root),
            vocabSize: try qwenInt("vocab_size", in: root),
            hiddenSize: try qwenInt("hidden_size", in: root),
            intermediateSize: try qwenInt("intermediate_size", in: root),
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: try qwenInt("num_attention_heads", in: root),
            numKeyValueHeads: try qwenInt("num_key_value_heads", in: root),
            headDim: try qwenInt("head_dim", in: root),
            linearNumValueHeads: try qwenInt("linear_num_value_heads", in: root),
            linearNumKeyHeads: try qwenInt("linear_num_key_heads", in: root),
            linearValueHeadDim: try qwenInt("linear_value_head_dim", in: root),
            linearKeyHeadDim: try qwenInt("linear_key_head_dim", in: root),
            linearConvKernelDim: try qwenInt("linear_conv_kernel_dim", in: root),
            fullAttentionInterval: try qwenInt("full_attention_interval", in: root),
            layerTypes: try qwenLayerTypes("layer_types", in: root),
            rmsNormEps: try qwenDouble("rms_norm_eps", in: root),
            hiddenActivation: try qwenString("hidden_act", in: root),
            maxPositionEmbeddings: try qwenInt("max_position_embeddings", in: root),
            attentionBias: try qwenBool("attention_bias", in: root),
            attentionDropout: try qwenDouble("attention_dropout", in: root),
            attentionOutputGate: try qwenBool("attn_output_gate", in: root),
            outputGateType: try qwenString("output_gate_type", in: root),
            tieWordEmbeddings: try qwenBool("tie_word_embeddings", in: root),
            mambaSSMDType: try qwenString("mamba_ssm_dtype", in: root),
            dtype: try qwenString("dtype", in: root),
            useCache: try qwenBool("use_cache", in: root),
            rope: Qwen35RopeSpec(
                theta: try qwenDouble("rope_theta", in: ropeObject),
                type: try qwenString("rope_type", in: ropeObject),
                partialRotaryFactor: ropePartial,
                mropeInterleaved: try qwenBool("mrope_interleaved", in: ropeObject),
                mropeSection: try qwenIntArray("mrope_section", in: ropeObject)
            ),
            quantizationGroupSize: quantization.groupSize,
            quantizationBits: quantization.bits,
            quantizationMode: quantization.mode,
            mtpNumHiddenLayers: try qwenInt("mtp_num_hidden_layers", in: root),
            mtpUseDedicatedEmbeddings: try qwenBool(
                "mtp_use_dedicated_embeddings",
                in: root
            )
        )
        try config.validateFrozenInvariants()
        try config.validateStructuralValues()
        return config
    }

    public func validateFrozenInvariants() throws {
        var errors: [String] = []
        func expect<T: Equatable>(_ name: String, _ actual: T, _ expected: T) {
            if actual != expected {
                errors.append(
                    "\(name)=\(String(describing: actual)) "
                        + "expected \(String(describing: expected))"
                )
            }
        }

        expect("model_type", modelType, "qwen3_5_text")
        expect("vocab_size", vocabSize, 248_320)
        expect("hidden_size", hiddenSize, 5_120)
        expect("intermediate_size", intermediateSize, 17_408)
        expect("num_hidden_layers", numHiddenLayers, 64)
        expect("num_attention_heads", numAttentionHeads, 24)
        expect("num_key_value_heads", numKeyValueHeads, 4)
        expect("head_dim", headDim, 256)
        expect("linear_num_value_heads", linearNumValueHeads, 48)
        expect("linear_num_key_heads", linearNumKeyHeads, 16)
        expect("linear_value_head_dim", linearValueHeadDim, 128)
        expect("linear_key_head_dim", linearKeyHeadDim, 128)
        expect("linear_conv_kernel_dim", linearConvKernelDim, 4)
        expect("full_attention_interval", fullAttentionInterval, 4)
        expect("rms_norm_eps", rmsNormEps, 1e-6)
        expect("hidden_act", hiddenActivation, "silu")
        expect("max_position_embeddings", maxPositionEmbeddings, 262_144)
        expect("attention_bias", attentionBias, false)
        expect("attention_dropout", attentionDropout, 0)
        expect("attn_output_gate", attentionOutputGate, true)
        expect("output_gate_type", outputGateType, "swish")
        expect("tie_word_embeddings", tieWordEmbeddings, false)
        expect("mamba_ssm_dtype", mambaSSMDType, "float32")
        expect("dtype", dtype, "bfloat16")
        expect("use_cache", useCache, true)
        expect("rope_parameters.rope_theta", rope.theta, 10_000_000)
        expect("rope_parameters.rope_type", rope.type, "default")
        expect("rope_parameters.partial_rotary_factor", rope.partialRotaryFactor, 0.25)
        expect("rope_parameters.mrope_interleaved", rope.mropeInterleaved, true)
        expect("rope_parameters.mrope_section", rope.mropeSection, [11, 11, 10])
        expect("quantization.group_size", quantizationGroupSize, 64)
        expect("quantization.bits", quantizationBits, 4)
        expect("quantization.mode", quantizationMode, "affine")
        expect("mtp_num_hidden_layers", mtpNumHiddenLayers, 1)
        expect("mtp_use_dedicated_embeddings", mtpUseDedicatedEmbeddings, false)

        if layerTypes != Self.expectedLayerTypes {
            if layerTypes.count != Self.expectedLayerTypes.count {
                errors.append(
                    "layer_types count=\(layerTypes.count) "
                        + "expected \(Self.expectedLayerTypes.count)"
                )
            } else if let mismatch = zip(layerTypes, Self.expectedLayerTypes)
                .enumerated()
                .first(where: { $0.element.0 != $0.element.1 })
            {
                errors.append(
                    "layer_types[\(mismatch.offset)]=\(mismatch.element.0.rawValue) "
                        + "expected \(mismatch.element.1.rawValue)"
                )
            }
        }

        guard errors.isEmpty else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 config invariant check failed: \(errors.joined(separator: ", "))"
            )
        }
    }

    public func validateStructuralValues() throws {
        guard vocabSize > 0,
              hiddenSize > 0,
              intermediateSize > 0,
              numHiddenLayers > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 vocabulary and model dimensions must be positive"
            )
        }
        guard numAttentionHeads > 0,
              numKeyValueHeads > 0,
              numAttentionHeads.isMultiple(of: numKeyValueHeads),
              headDim > 0,
              headDim.isMultiple(of: 2)
        else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 attention heads and head_dim are structurally invalid"
            )
        }
        guard linearNumValueHeads > 0,
              linearNumKeyHeads > 0,
              linearNumValueHeads.isMultiple(of: linearNumKeyHeads),
              linearValueHeadDim > 0,
              linearKeyHeadDim > 0,
              linearConvKernelDim > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 Gated DeltaNet dimensions are structurally invalid"
            )
        }
        guard qwenProduct([numAttentionHeads, headDim, 2]) != nil,
              qwenProduct([numKeyValueHeads, headDim]) != nil,
              qwenProduct([linearNumValueHeads, linearValueHeadDim]) != nil,
              qwenProduct([linearNumKeyHeads, linearKeyHeadDim]) != nil
        else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 attention projection dimensions overflow Int"
            )
        }
        guard fullAttentionInterval > 0,
              layerTypes.count == numHiddenLayers
        else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 layer_types must cover every decoder layer"
            )
        }
        guard maxPositionEmbeddings > 0,
              rmsNormEps.isFinite,
              rmsNormEps > 0,
              attentionDropout.isFinite,
              attentionDropout >= 0,
              attentionDropout < 1
        else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 position, normalization, or dropout values are invalid"
            )
        }
        guard rope.theta.isFinite,
              rope.theta > 0,
              rope.partialRotaryFactor.isFinite,
              rope.partialRotaryFactor > 0,
              rope.partialRotaryFactor <= 1,
              let rotaryDimensions = Int(
                  exactly: Double(headDim) * rope.partialRotaryFactor
              ),
              rotaryDimensions >= 2,
              rotaryDimensions.isMultiple(of: 2),
              !rope.mropeSection.isEmpty,
              rope.mropeSection.allSatisfy({ $0 > 0 }),
              qwenSum(rope.mropeSection) == rotaryDimensions / 2
        else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 partial MRoPE dimensions are structurally invalid"
            )
        }
        guard quantizationGroupSize > 0,
              hiddenSize.isMultiple(of: quantizationGroupSize),
              intermediateSize.isMultiple(of: quantizationGroupSize),
              [2, 4, 8].contains(quantizationBits)
        else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 quantization dimensions or bit width are invalid"
            )
        }
        guard mtpNumHiddenLayers >= 0 else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 mtp_num_hidden_layers must not be negative"
            )
        }
    }
}

private struct Qwen35QuantizationSpec: Equatable {
    let groupSize: Int
    let bits: Int
    let mode: String
}

private func qwenRequiredValue(_ key: String, in root: [String: Any]) throws -> Any {
    guard let value = root[key] else {
        throw MLXFastError.invalidInput("config field \(key) is required")
    }
    guard !(value is NSNull) else {
        throw MLXFastError.invalidInput("config field \(key) must not be null")
    }
    return value
}

private func qwenObject(
    _ key: String,
    in root: [String: Any]
) throws -> [String: Any] {
    let value = try qwenRequiredValue(key, in: root)
    guard let result = value as? [String: Any] else {
        throw MLXFastError.invalidInput("config field \(key) must be a JSON object")
    }
    return result
}

private func qwenQuantization(in root: [String: Any]) throws -> Qwen35QuantizationSpec {
    let hasQuantization = root.keys.contains("quantization")
    let hasQuantizationConfig = root.keys.contains("quantization_config")
    guard hasQuantization || hasQuantizationConfig else {
        throw MLXFastError.invalidInput(
            "config requires quantization or quantization_config"
        )
    }

    func parse(_ key: String) throws -> Qwen35QuantizationSpec {
        let object = try qwenObject(key, in: root)
        return Qwen35QuantizationSpec(
            groupSize: try qwenInt("group_size", in: object),
            bits: try qwenInt("bits", in: object),
            mode: try qwenString("mode", in: object)
        )
    }

    let quantization = try hasQuantization ? parse("quantization") : nil
    let quantizationConfig = try hasQuantizationConfig
        ? parse("quantization_config")
        : nil
    if let quantization, let quantizationConfig,
       quantization != quantizationConfig
    {
        throw MLXFastError.invalidInput(
            "config quantization and quantization_config must agree"
        )
    }
    if let quantization {
        return quantization
    }
    guard let quantizationConfig else {
        throw MLXFastError.invalidInput(
            "config requires quantization or quantization_config"
        )
    }
    return quantizationConfig
}

private func qwenInt(_ key: String, in root: [String: Any]) throws -> Int {
    try qwenParseInt(
        qwenRequiredValue(key, in: root),
        field: key
    )
}

private func qwenDouble(
    _ key: String,
    in root: [String: Any]
) throws -> Double {
    let value = try qwenRequiredValue(key, in: root)
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID()
    else {
        throw MLXFastError.invalidInput("config field \(key) must be a finite number")
    }
    let result = number.doubleValue
    guard result.isFinite else {
        throw MLXFastError.invalidInput("config field \(key) must be a finite number")
    }
    return result
}

private func qwenBool(
    _ key: String,
    in root: [String: Any]
) throws -> Bool {
    let value = try qwenRequiredValue(key, in: root)
    guard let number = value as? NSNumber,
          CFGetTypeID(number) == CFBooleanGetTypeID()
    else {
        throw MLXFastError.invalidInput("config field \(key) must be a boolean")
    }
    return number.boolValue
}

private func qwenString(
    _ key: String,
    in root: [String: Any]
) throws -> String {
    let value = try qwenRequiredValue(key, in: root)
    guard let result = value as? String else {
        throw MLXFastError.invalidInput("config field \(key) must be a string")
    }
    return result
}

private func qwenIntArray(
    _ key: String,
    in root: [String: Any]
) throws -> [Int] {
    let value = try qwenRequiredValue(key, in: root)
    guard let values = value as? [Any] else {
        throw MLXFastError.invalidInput("config field \(key) must be an integer array")
    }
    return try values.enumerated().map {
        try qwenParseInt($0.element, field: "\(key)[\($0.offset)]")
    }
}

private func qwenLayerTypes(
    _ key: String,
    in root: [String: Any]
) throws -> [Qwen35LayerType] {
    let value = try qwenRequiredValue(key, in: root)
    guard let values = value as? [String] else {
        throw MLXFastError.invalidInput("config field \(key) must be a string array")
    }
    return try values.map {
        guard let value = Qwen35LayerType(rawValue: $0) else {
            throw MLXFastError.invalidInput(
                "config field \(key) contains unsupported layer type \($0)"
            )
        }
        return value
    }
}

private func qwenParseInt(_ value: Any, field: String) throws -> Int {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID(),
          !CFNumberIsFloatType(number),
          let integer = Int(number.stringValue)
    else {
        throw MLXFastError.invalidInput(
            "config field \(field) must be a finite integer in Int range"
        )
    }
    return integer
}

private func qwenProduct(_ values: [Int]) -> Int? {
    var product = 1
    for value in values {
        let result = product.multipliedReportingOverflow(by: value)
        guard !result.overflow else {
            return nil
        }
        product = result.partialValue
    }
    return product
}

private func qwenSum(_ values: [Int]) -> Int? {
    var sum = 0
    for value in values {
        let result = sum.addingReportingOverflow(value)
        guard !result.overflow else {
            return nil
        }
        sum = result.partialValue
    }
    return sum
}
