import MLX
import MLXFastCore
import MLXLMCommon

public enum Qwen35BlockMixer {
    case linear(
        weights: Qwen35GatedDeltaWeights,
        spec: Qwen35GatedDeltaSpec
    )
    case full(
        weights: Qwen35AttentionWeights,
        spec: Qwen35AttentionSpec
    )

    public var layerType: Qwen35LayerType {
        switch self {
        case .linear:
            .linear
        case .full:
            .full
        }
    }
}

public struct Qwen35BlockWeights {
    public let inputLayerNorm: MLXArray
    public let postAttentionLayerNorm: MLXArray
    public let mixer: Qwen35BlockMixer
    public let mlp: Qwen35MLPWeights

    public init(
        inputLayerNorm: MLXArray,
        postAttentionLayerNorm: MLXArray,
        mixer: Qwen35BlockMixer,
        mlp: Qwen35MLPWeights
    ) {
        self.inputLayerNorm = inputLayerNorm
        self.postAttentionLayerNorm = postAttentionLayerNorm
        self.mixer = mixer
        self.mlp = mlp
    }
}

/// Per-layer cache for the inactive custom trunk. Exactly one member is used,
/// selected by the block's linear/full layer type.
public final class Qwen35BlockCache {
    public var fullAttention: (any KVCache)?
    public var gatedDelta: Qwen35GatedDeltaState?

    public init(
        fullAttention: (any KVCache)? = nil,
        gatedDelta: Qwen35GatedDeltaState? = nil
    ) {
        self.fullAttention = fullAttention
        self.gatedDelta = gatedDelta
    }

    func snapshot() -> Qwen35BlockCacheSnapshot {
        Qwen35BlockCacheSnapshot(
            fullAttention: fullAttention?.copy(),
            gatedDelta: gatedDelta?.copy()
        )
    }

    func restore(_ snapshot: Qwen35BlockCacheSnapshot) {
        fullAttention = snapshot.fullAttention
        gatedDelta = snapshot.gatedDelta
    }
}

struct Qwen35BlockCacheSnapshot {
    let fullAttention: (any KVCache)?
    let gatedDelta: Qwen35GatedDeltaState?
}

/// Two-pre-norm Qwen35 decoder block copied from pinned
/// `Qwen35DecoderLayer`: mixer(norm(x)) + x, then MLP(norm(h)) + h.
public enum Qwen35Block {
    public static func forward(
        _ input: MLXArray,
        weights: Qwen35BlockWeights,
        rmsNormEps: Double,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        ssmMask: MLXArray? = nil,
        cache: Qwen35BlockCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        guard input.ndim == 3,
              weights.inputLayerNorm.shape == [input.dim(2)],
              weights.postAttentionLayerNorm.shape == [input.dim(2)]
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 block input or norm shapes are invalid"
            )
        }
        #if DEBUG
        try validateContract(
            weights: weights,
            hiddenSize: input.dim(2)
        )
        #endif

        let normalizedInput = Qwen35Ops.rmsNorm(
            input,
            weight: weights.inputLayerNorm,
            eps: rmsNormEps
        )
        let mixed: MLXArray
        switch weights.mixer {
        case let .linear(linearWeights, spec):
            guard cache?.fullAttention == nil else {
                throw MLXFastError.invalidInput(
                    "Qwen35 linear block received a full-attention cache"
                )
            }
            if let cache, cache.gatedDelta == nil {
                cache.gatedDelta = try Qwen35GatedDeltaState(
                    batchSize: input.dim(0),
                    spec: spec,
                    activationDType: input.dtype
                )
            }
            mixed = try Qwen35GatedDeltaNet.forward(
                normalizedInput,
                weights: linearWeights,
                spec: spec,
                mask: ssmMask,
                state: cache?.gatedDelta
            )

        case let .full(attentionWeights, spec):
            guard let cache,
                  let fullAttentionCache = cache.fullAttention,
                  cache.gatedDelta == nil
            else {
                throw MLXFastError.invalidInput(
                    "Qwen35 full block requires exactly one full-attention cache"
                )
            }
            mixed = try Qwen35FullAttention.forward(
                normalizedInput,
                weights: attentionWeights,
                spec: spec,
                mask: attentionMask,
                cache: fullAttentionCache,
                positionOffset: positionOffset
            )
        }

        let residual = input + mixed
        let normalizedResidual = Qwen35Ops.rmsNorm(
            residual,
            weight: weights.postAttentionLayerNorm,
            eps: rmsNormEps
        )
        return residual + Qwen35MLP.forward(
            normalizedResidual,
            weights: weights.mlp
        )
    }

    static func validateContract(
        weights: Qwen35BlockWeights,
        hiddenSize: Int
    ) throws {
        guard hiddenSize > 0,
              weights.inputLayerNorm.shape == [hiddenSize],
              weights.postAttentionLayerNorm.shape == [hiddenSize]
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 block norm shapes are invalid"
            )
        }
        switch weights.mixer {
        case let .linear(linearWeights, spec):
            guard spec.hiddenSize == hiddenSize else {
                throw MLXFastError.invalidInput(
                    "Qwen35 linear block spec hidden size is invalid"
                )
            }
            try Qwen35GatedDeltaNet.validateContract(
                weights: linearWeights,
                spec: spec
            )
        case let .full(attentionWeights, spec):
            guard spec.hiddenSize == hiddenSize else {
                throw MLXFastError.invalidInput(
                    "Qwen35 full block spec hidden size is invalid"
                )
            }
            try Qwen35FullAttention.validateContract(
                weights: attentionWeights,
                spec: spec
            )
        }
        try Qwen35MLP.validateContract(
            weights: weights.mlp,
            hiddenSize: hiddenSize
        )
    }
}
