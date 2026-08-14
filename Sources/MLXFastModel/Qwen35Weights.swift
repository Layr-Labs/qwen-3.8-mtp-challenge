import Foundation
import MLX
import MLXFastCore

/// Tensor names for the text-only Qwen3.6/Qwen35 checkpoint.
public enum Qwen35WeightNames {
    private static let modelPrefix = "language_model.model"

    public static let embedTokens = "\(modelPrefix).embed_tokens.weight"
    public static let finalNorm = "\(modelPrefix).norm.weight"
    public static let lmHead = "language_model.lm_head.weight"

    public static func layer(_ layerIndex: Int, _ suffix: String) -> String {
        "\(modelPrefix).layers.\(layerIndex).\(suffix)"
    }

    public static func linearAttention(_ layerIndex: Int, _ suffix: String) -> String {
        layer(layerIndex, "linear_attn.\(suffix)")
    }

    public static func fullAttention(_ layerIndex: Int, _ suffix: String) -> String {
        layer(layerIndex, "self_attn.\(suffix)")
    }

    public static func mlp(_ layerIndex: Int, _ suffix: String) -> String {
        layer(layerIndex, "mlp.\(suffix)")
    }
}

/// Dense tensor access and startup metadata validation for Qwen35.
///
/// The scored loader uses `denseStore` to load complete safetensor shards into
/// MLX. Primitive accessors remain useful for tiny metadata/bridge tests.
public struct Qwen35WeightLoader {
    public static let requiredTopLevelTensorCount = 7
    public static let requiredLinearLayerTensorCount = 30
    public static let requiredFullAttentionLayerTensorCount = 25
    public static let requiredTensorCount = 1_847

    public let denseStore: DenseTensorStore
    private let bridge: MLXArrayTensorBridge

    public init(
        weightsPath: String,
        bridge: MLXArrayTensorBridge = MLXArrayTensorBridge()
    ) throws {
        self.denseStore = try DenseTensorStore(weightsPath: weightsPath)
        self.bridge = bridge
    }

    public init(
        denseStore: DenseTensorStore,
        bridge: MLXArrayTensorBridge = MLXArrayTensorBridge()
    ) {
        self.denseStore = denseStore
        self.bridge = bridge
    }

    public func materializedTensor(
        named name: String,
        expectedShape: [Int]? = nil
    ) throws -> MaterializedTensor {
        let tensor = try denseStore.materializedTensor(named: name)
        try validateShape(
            tensor.shape,
            expectedShape: expectedShape,
            tensorName: name
        )
        return tensor
    }

    public func denseArray(
        named name: String,
        expectedShape: [Int]? = nil
    ) throws -> MLXArray {
        try bridge.makeArray(
            from: materializedTensor(named: name, expectedShape: expectedShape)
        )
    }

    public func optionalDenseArray(
        named name: String,
        expectedShape: [Int]? = nil
    ) throws -> MLXArray? {
        guard denseStore.record(named: name) != nil else {
            return nil
        }
        return try denseArray(named: name, expectedShape: expectedShape)
    }

    /// Compatibility primitive retained for the existing tiny bridge tests.
    /// The scored Qwen runtime loads and updates the library model directly.
    public func linearWeight(
        named baseName: String,
        outFeatures: Int,
        inFeatures: Int,
        expectedGroupSize: Int? = nil,
        expectedBits: Int? = nil,
        requireQuantized: Bool = false
    ) throws -> Qwen35LinearWeight {
        let quantization = try validateLinearMetadata(
            named: baseName,
            expectedRows: outFeatures,
            expectedInputFeatures: inFeatures,
            expectedGroupSize: expectedGroupSize,
            expectedBits: expectedBits
        )
        let tensor = try materializedTensor(named: baseName)
        guard let quantization else {
            guard !requireQuantized else {
                throw MLXFastError.invalidInput(
                    "Qwen35 checkpoint tensor \(baseName) must be quantized"
                )
            }
            return Qwen35LinearWeight(try bridge.makeArray(from: tensor))
        }

        let scalesName = Self.companionName(for: baseName, suffix: "scales")
        let biasesName = Self.companionName(for: baseName, suffix: "biases")
        return try Qwen35LinearWeight(
            weight: try bridge.makeArray(from: tensor),
            scales: try denseArray(named: scalesName),
            biases: try denseArray(named: biasesName),
            logicalShape: [outFeatures, inFeatures],
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
    }

    /// Validate the complete frozen Qwen35 tensor contract without
    /// materializing the ~15 GiB checkpoint.
    public func validateRequiredMetadata(config: Qwen35Config) throws {
        try config.validateFrozenInvariants()
        try config.validateStructuralValues()

        let actualNames = Set(denseStore.tensorNames)
        guard !actualNames.contains(where: {
            $0.split(separator: ".").contains("mtp")
        }) else {
            throw MLXFastError.invalidInput(
                "Qwen35 MTP weights are unsupported in the eager baseline"
            )
        }

        let expectedNames = try Self.requiredTensorNames(config: config)
        guard actualNames == expectedNames else {
            let missing = expectedNames.subtracting(actualNames).sorted()
            let unexpected = actualNames.subtracting(expectedNames).sorted()
            throw MLXFastError.invalidInput(
                "Qwen35 tensor inventory mismatch: expected "
                    + "\(Self.requiredTensorCount), found \(actualNames.count); "
                    + "missing=\(Self.summarize(missing)); "
                    + "unexpected=\(Self.summarize(unexpected))"
            )
        }

        try validateQuantizedLinear(
            named: Qwen35WeightNames.embedTokens,
            outFeatures: config.vocabSize,
            inFeatures: config.hiddenSize,
            config: config
        )
        try validateDenseTensorMetadata(
            named: Qwen35WeightNames.finalNorm,
            expectedShape: [config.hiddenSize]
        )
        try validateQuantizedLinear(
            named: Qwen35WeightNames.lmHead,
            outFeatures: config.vocabSize,
            inFeatures: config.hiddenSize,
            config: config
        )

        let linearKeySize = try checkedProduct(
            [config.linearNumKeyHeads, config.linearKeyHeadDim],
            label: "linear key size"
        )
        let linearValueSize = try checkedProduct(
            [config.linearNumValueHeads, config.linearValueHeadDim],
            label: "linear value size"
        )
        let linearConvSize = try checkedSum(
            [linearKeySize, linearKeySize, linearValueSize],
            label: "linear convolution size"
        )
        let fullQuerySize = try checkedProduct(
            [config.numAttentionHeads, config.headDim, 2],
            label: "full-attention query/gate size"
        )
        let fullKVSize = try checkedProduct(
            [config.numKeyValueHeads, config.headDim],
            label: "full-attention KV size"
        )
        let fullOutputSize = try checkedProduct(
            [config.numAttentionHeads, config.headDim],
            label: "full-attention output size"
        )

        for layerIndex in 0..<config.numHiddenLayers {
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ] {
                try validateDenseTensorMetadata(
                    named: Qwen35WeightNames.layer(layerIndex, suffix),
                    expectedShape: [config.hiddenSize]
                )
            }

            for (suffix, output, input) in [
                ("gate_proj.weight", config.intermediateSize, config.hiddenSize),
                ("up_proj.weight", config.intermediateSize, config.hiddenSize),
                ("down_proj.weight", config.hiddenSize, config.intermediateSize),
            ] {
                try validateQuantizedLinear(
                    named: Qwen35WeightNames.mlp(layerIndex, suffix),
                    outFeatures: output,
                    inFeatures: input,
                    config: config
                )
            }

            switch config.layerTypes[layerIndex] {
            case .linear:
                try validateDenseTensorMetadata(
                    named: Qwen35WeightNames.linearAttention(
                        layerIndex,
                        "conv1d.weight"
                    ),
                    expectedShape: [
                        linearConvSize,
                        config.linearConvKernelDim,
                        1,
                    ]
                )
                for suffix in ["A_log", "dt_bias"] {
                    try validateDenseTensorMetadata(
                        named: Qwen35WeightNames.linearAttention(
                            layerIndex,
                            suffix
                        ),
                        expectedShape: [config.linearNumValueHeads]
                    )
                }
                try validateDenseTensorMetadata(
                    named: Qwen35WeightNames.linearAttention(
                        layerIndex,
                        "norm.weight"
                    ),
                    expectedShape: [config.linearValueHeadDim]
                )
                for (suffix, output, input) in [
                    ("in_proj_qkv.weight", linearConvSize, config.hiddenSize),
                    ("in_proj_z.weight", linearValueSize, config.hiddenSize),
                    ("in_proj_b.weight", config.linearNumValueHeads, config.hiddenSize),
                    ("in_proj_a.weight", config.linearNumValueHeads, config.hiddenSize),
                    ("out_proj.weight", config.hiddenSize, linearValueSize),
                ] {
                    try validateQuantizedLinear(
                        named: Qwen35WeightNames.linearAttention(
                            layerIndex,
                            suffix
                        ),
                        outFeatures: output,
                        inFeatures: input,
                        config: config
                    )
                }

            case .full:
                for suffix in ["q_norm.weight", "k_norm.weight"] {
                    try validateDenseTensorMetadata(
                        named: Qwen35WeightNames.fullAttention(
                            layerIndex,
                            suffix
                        ),
                        expectedShape: [config.headDim]
                    )
                }
                for (suffix, output, input) in [
                    ("q_proj.weight", fullQuerySize, config.hiddenSize),
                    ("k_proj.weight", fullKVSize, config.hiddenSize),
                    ("v_proj.weight", fullKVSize, config.hiddenSize),
                    ("o_proj.weight", config.hiddenSize, fullOutputSize),
                ] {
                    try validateQuantizedLinear(
                        named: Qwen35WeightNames.fullAttention(
                            layerIndex,
                            suffix
                        ),
                        outFeatures: output,
                        inFeatures: input,
                        config: config
                    )
                }
            }
        }
    }

    @discardableResult
    func validateLinearMetadata(
        named name: String,
        expectedRows: Int,
        expectedInputFeatures: Int,
        expectedGroupSize: Int? = nil,
        expectedBits: Int? = nil
    ) throws -> (groupSize: Int, bits: Int)? {
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        if record.dtype != "U32" {
            try validateDenseTensorMetadata(
                named: name,
                expectedShape: [expectedRows, expectedInputFeatures]
            )
            return nil
        }
        guard expectedInputFeatures > 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) logical input must be positive"
            )
        }
        guard record.shape.count == 2,
              record.shape[0] == expectedRows,
              record.shape[1] > 0
        else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) shape \(record.shape) must be "
                    + "[\(expectedRows), packedInput]"
            )
        }

        let (packedBits, overflow) = record.shape[1].multipliedReportingOverflow(
            by: 32
        )
        guard !overflow, packedBits % expectedInputFeatures == 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) packed input \(record.shape[1]) "
                    + "is incompatible with logical input \(expectedInputFeatures)"
            )
        }
        let bits = packedBits / expectedInputFeatures
        guard [2, 4, 8].contains(bits) else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) has unsupported bit width \(bits)"
            )
        }

        let scalesName = Self.companionName(for: name, suffix: "scales")
        let biasesName = Self.companionName(for: name, suffix: "biases")
        guard let scales = denseStore.record(named: scalesName) else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) is missing companion \(scalesName)"
            )
        }
        guard let biases = denseStore.record(named: biasesName) else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) is missing companion \(biasesName)"
            )
        }
        guard scales.dtype == "BF16", biases.dtype == "BF16" else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) scales and biases must use BF16"
            )
        }
        guard scales.shape.count == 2,
              scales.shape[0] == expectedRows,
              scales.shape[1] > 0,
              expectedInputFeatures % scales.shape[1] == 0
        else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) scales shape \(scales.shape) is "
                    + "incompatible with logical input \(expectedInputFeatures)"
            )
        }
        guard biases.shape == scales.shape else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) biases shape \(biases.shape) "
                    + "does not match scales shape \(scales.shape)"
            )
        }

        let groupSize = expectedInputFeatures / scales.shape[1]
        if let expectedGroupSize, groupSize != expectedGroupSize {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) group size \(groupSize) "
                    + "does not match configured group size \(expectedGroupSize)"
            )
        }
        if let expectedBits, bits != expectedBits {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) bit width \(bits) "
                    + "does not match configured bit width \(expectedBits)"
            )
        }
        return (groupSize, bits)
    }

    private func validateQuantizedLinear(
        named name: String,
        outFeatures: Int,
        inFeatures: Int,
        config: Qwen35Config
    ) throws {
        let metadata = try validateLinearMetadata(
            named: name,
            expectedRows: outFeatures,
            expectedInputFeatures: inFeatures,
            expectedGroupSize: config.quantizationGroupSize,
            expectedBits: config.quantizationBits
        )
        guard metadata != nil else {
            throw MLXFastError.invalidInput(
                "Qwen35 checkpoint tensor \(name) must use affine quantization"
            )
        }
    }

    private func validateDenseTensorMetadata(
        named name: String,
        expectedShape: [Int],
        expectedDType: String = "BF16"
    ) throws {
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        guard record.dtype == expectedDType else {
            throw MLXFastError.invalidInput(
                "tensor \(name) dtype \(record.dtype) does not match "
                    + "expected dtype \(expectedDType)"
            )
        }
        guard record.shape == expectedShape else {
            throw MLXFastError.invalidInput(
                "tensor \(name) shape \(record.shape) does not match "
                    + "expected shape \(expectedShape)"
            )
        }
    }

    private func validateShape(
        _ actualShape: [Int],
        expectedShape: [Int]?,
        tensorName: String
    ) throws {
        guard let expectedShape else {
            return
        }
        guard actualShape == expectedShape else {
            throw MLXFastError.invalidInput(
                "tensor \(tensorName) shape \(actualShape) does not match "
                    + "expected shape \(expectedShape)"
            )
        }
    }

    private static func requiredTensorNames(
        config: Qwen35Config
    ) throws -> Set<String> {
        var names: Set<String> = []

        func addDense(_ name: String) {
            names.insert(name)
        }

        func addQuantized(_ name: String) {
            names.insert(name)
            names.insert(Self.companionName(for: name, suffix: "scales"))
            names.insert(Self.companionName(for: name, suffix: "biases"))
        }

        addQuantized(Qwen35WeightNames.embedTokens)
        addDense(Qwen35WeightNames.finalNorm)
        addQuantized(Qwen35WeightNames.lmHead)
        guard names.count == Self.requiredTopLevelTensorCount else {
            throw MLXFastError.invalidInput(
                "Qwen35 internal top-level tensor contract is inconsistent"
            )
        }

        for layerIndex in 0..<config.numHiddenLayers {
            let countBeforeLayer = names.count
            addDense(Qwen35WeightNames.layer(layerIndex, "input_layernorm.weight"))
            addDense(Qwen35WeightNames.layer(layerIndex, "post_attention_layernorm.weight"))
            for suffix in [
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            ] {
                addQuantized(Qwen35WeightNames.mlp(layerIndex, suffix))
            }

            let expectedLayerTensorCount: Int
            switch config.layerTypes[layerIndex] {
            case .linear:
                for suffix in [
                    "conv1d.weight",
                    "A_log",
                    "dt_bias",
                    "norm.weight",
                ] {
                    addDense(Qwen35WeightNames.linearAttention(layerIndex, suffix))
                }
                for suffix in [
                    "in_proj_qkv.weight",
                    "in_proj_z.weight",
                    "in_proj_b.weight",
                    "in_proj_a.weight",
                    "out_proj.weight",
                ] {
                    addQuantized(Qwen35WeightNames.linearAttention(layerIndex, suffix))
                }
                expectedLayerTensorCount = Self.requiredLinearLayerTensorCount

            case .full:
                for suffix in ["q_norm.weight", "k_norm.weight"] {
                    addDense(Qwen35WeightNames.fullAttention(layerIndex, suffix))
                }
                for suffix in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                ] {
                    addQuantized(Qwen35WeightNames.fullAttention(layerIndex, suffix))
                }
                expectedLayerTensorCount =
                    Self.requiredFullAttentionLayerTensorCount
            }

            guard names.count - countBeforeLayer == expectedLayerTensorCount else {
                throw MLXFastError.invalidInput(
                    "Qwen35 internal layer \(layerIndex) tensor contract is inconsistent"
                )
            }
        }

        guard names.count == Self.requiredTensorCount else {
            throw MLXFastError.invalidInput(
                "Qwen35 internal tensor contract has \(names.count) names, "
                    + "expected \(Self.requiredTensorCount)"
            )
        }
        return names
    }

    private static func summarize(_ names: [String], limit: Int = 8) -> String {
        guard !names.isEmpty else {
            return "[]"
        }
        let prefix = names.prefix(limit).joined(separator: ", ")
        if names.count <= limit {
            return "[\(prefix)]"
        }
        return "[\(prefix), ... +\(names.count - limit)]"
    }

    private static func companionName(
        for baseName: String,
        suffix: String
    ) -> String {
        guard baseName.hasSuffix(".weight") else {
            return "\(baseName).\(suffix)"
        }
        return String(baseName.dropLast(".weight".count)) + ".\(suffix)"
    }
}

private func checkedProduct(_ values: [Int], label: String) throws -> Int {
    var result = 1
    for value in values {
        let next = result.multipliedReportingOverflow(by: value)
        guard !next.overflow else {
            throw MLXFastError.invalidInput("\(label) overflows Int")
        }
        result = next.partialValue
    }
    return result
}

private func checkedSum(_ values: [Int], label: String) throws -> Int {
    var result = 0
    for value in values {
        let next = result.addingReportingOverflow(value)
        guard !next.overflow else {
            throw MLXFastError.invalidInput("\(label) overflows Int")
        }
        result = next.partialValue
    }
    return result
}
