import MLX
import MLXFastCore
import MLXLMCommon
import MLXNN

public struct Qwen35AttentionSpec {
    public let hiddenSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDimension: Int
    public let rmsNormEps: Double
    public let rope: Qwen35RoPE
    private let computedQuerySize: Int
    private let computedQueryAndGateProjectionSize: Int
    private let computedKeyValueSize: Int

    public init(
        hiddenSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDimension: Int,
        rmsNormEps: Double,
        ropeSpec: Qwen35RopeSpec
    ) throws {
        guard hiddenSize > 0,
              numAttentionHeads > 0,
              numKeyValueHeads > 0,
              numAttentionHeads.isMultiple(of: numKeyValueHeads),
              headDimension > 0,
              rmsNormEps.isFinite,
              rmsNormEps > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 full-attention dimensions are invalid"
            )
        }
        let querySize = try qwen35CheckedProduct(
            [numAttentionHeads, headDimension],
            label: "Qwen35 attention query size"
        )
        let queryAndGateProjectionSize = try qwen35CheckedProduct(
            [querySize, 2],
            label: "Qwen35 attention query/gate size"
        )
        let keyValueSize = try qwen35CheckedProduct(
            [numKeyValueHeads, headDimension],
            label: "Qwen35 attention key/value size"
        )
        self.hiddenSize = hiddenSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDimension = headDimension
        self.rmsNormEps = rmsNormEps
        self.rope = try Qwen35RoPE(
            headDimensions: headDimension,
            spec: ropeSpec
        )
        self.computedQuerySize = querySize
        self.computedQueryAndGateProjectionSize =
            queryAndGateProjectionSize
        self.computedKeyValueSize = keyValueSize
    }

    public init(config: Qwen35Config) throws {
        try self.init(
            hiddenSize: config.hiddenSize,
            numAttentionHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads,
            headDimension: config.headDim,
            rmsNormEps: config.rmsNormEps,
            ropeSpec: config.rope
        )
    }

    public var querySize: Int {
        computedQuerySize
    }

    public var queryAndGateProjectionSize: Int {
        computedQueryAndGateProjectionSize
    }

    public var keyValueSize: Int {
        computedKeyValueSize
    }

    public var scale: Float {
        1 / Float(headDimension).squareRoot()
    }
}

public struct Qwen35AttentionWeights {
    public let queryProjection: Qwen35LinearWeight
    public let keyProjection: Qwen35LinearWeight
    public let valueProjection: Qwen35LinearWeight
    public let outputProjection: Qwen35LinearWeight
    public let queryNorm: MLXArray
    public let keyNorm: MLXArray

    public init(
        queryProjection: Qwen35LinearWeight,
        keyProjection: Qwen35LinearWeight,
        valueProjection: Qwen35LinearWeight,
        outputProjection: Qwen35LinearWeight,
        queryNorm: MLXArray,
        keyNorm: MLXArray
    ) {
        self.queryProjection = queryProjection
        self.keyProjection = keyProjection
        self.valueProjection = valueProjection
        self.outputProjection = outputProjection
        self.queryNorm = queryNorm
        self.keyNorm = keyNorm
    }
}

/// Qwen35 gated global GQA copied from pinned `Qwen35Attention`.
///
/// For the frozen checkpoint q_proj is logically `5120 -> 12288`; reshaping
/// to `[B, L, 24, 512]` and splitting the final axis yields 24x256 queries
/// plus an equivalent 6144-element gate. K/V are 4x256, Q/K use learned
/// RMSNorm, attention scales by `1/sqrt(256)`, and the sigmoid output gate is
/// applied before the untied `6144 -> 5120` output projection.
public enum Qwen35FullAttention {
    public static func forward(
        _ input: MLXArray,
        weights: Qwen35AttentionWeights,
        spec: Qwen35AttentionSpec,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: (any KVCache)? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        try validate(
            input,
            weights: weights,
            spec: spec,
            cache: cache,
            positionOffset: positionOffset
        )

        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)
        let queryProjection = Qwen35Ops.linear(
            input,
            weights.queryProjection
        )
        let querySplit = queryProjection
            .reshaped(
                batchSize,
                sequenceLength,
                spec.numAttentionHeads,
                -1
            )
            .split(parts: 2, axis: -1)

        var queries = querySplit[0]
        let gate = querySplit[1].reshaped(
            batchSize,
            sequenceLength,
            -1
        )
        var keys = Qwen35Ops.linear(input, weights.keyProjection)
        var values = Qwen35Ops.linear(input, weights.valueProjection)

        queries = Qwen35Ops.rmsNorm(
            queries,
            weight: weights.queryNorm,
            eps: spec.rmsNormEps
        ).transposed(0, 2, 1, 3)
        keys = Qwen35Ops.rmsNorm(
            keys.reshaped(
                batchSize,
                sequenceLength,
                spec.numKeyValueHeads,
                spec.headDimension
            ),
            weight: weights.keyNorm,
            eps: spec.rmsNormEps
        ).transposed(0, 2, 1, 3)
        values = values
            .reshaped(
                batchSize,
                sequenceLength,
                spec.numKeyValueHeads,
                spec.headDimension
            )
            .transposed(0, 2, 1, 3)

        queries = spec.rope.applied(
            to: queries,
            cache: cache,
            fallbackOffset: positionOffset
        )
        keys = spec.rope.applied(
            to: keys,
            cache: cache,
            fallbackOffset: positionOffset
        )

        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: spec.scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, sequenceLength, -1)

        return Qwen35Ops.linear(
            attended * sigmoid(gate),
            weights.outputProjection
        )
    }

    private static func validate(
        _ input: MLXArray,
        weights: Qwen35AttentionWeights,
        spec: Qwen35AttentionSpec,
        cache: (any KVCache)?,
        positionOffset: Int
    ) throws {
        guard input.ndim == 3,
              input.dim(2) == spec.hiddenSize,
              positionOffset >= 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 full-attention input or position is invalid"
            )
        }
        if let cache, cache.offset != positionOffset {
            throw MLXFastError.invalidInput(
                "Qwen35 full-attention cache offset \(cache.offset) does not "
                    + "match position \(positionOffset)"
            )
        }
        try validateContract(weights: weights, spec: spec)
    }

    static func validateContract(
        weights: Qwen35AttentionWeights,
        spec: Qwen35AttentionSpec
    ) throws {
        try weights.queryProjection.validate()
        try weights.keyProjection.validate()
        try weights.valueProjection.validate()
        try weights.outputProjection.validate()
        guard weights.queryProjection.shape
                == [spec.queryAndGateProjectionSize, spec.hiddenSize],
              weights.keyProjection.shape
                == [spec.keyValueSize, spec.hiddenSize],
              weights.valueProjection.shape
                == [spec.keyValueSize, spec.hiddenSize],
              weights.outputProjection.shape
                == [spec.hiddenSize, spec.querySize],
              weights.queryNorm.shape == [spec.headDimension],
              weights.keyNorm.shape == [spec.headDimension]
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 full-attention weight shapes do not match the spec"
            )
        }
    }
}
