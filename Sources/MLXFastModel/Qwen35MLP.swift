import MLX
import MLXFastCore
import MLXNN

public struct Qwen35MLPWeights {
    public let gateProjection: Qwen35LinearWeight
    public let upProjection: Qwen35LinearWeight
    public let downProjection: Qwen35LinearWeight

    public init(
        gateProjection: Qwen35LinearWeight,
        upProjection: Qwen35LinearWeight,
        downProjection: Qwen35LinearWeight
    ) {
        self.gateProjection = gateProjection
        self.upProjection = upProjection
        self.downProjection = downProjection
    }
}

/// Dense Qwen35 SwiGLU copied from pinned `Qwen3NextMLP`:
/// `down_proj(silu(gate_proj(x)) * up_proj(x))`.
public enum Qwen35MLP {
    public static func forward(
        _ input: MLXArray,
        weights: Qwen35MLPWeights
    ) -> MLXArray {
        let gate = silu(Qwen35Ops.linear(input, weights.gateProjection))
        let up = Qwen35Ops.linear(input, weights.upProjection)
        return Qwen35Ops.linear(gate * up, weights.downProjection)
    }

    static func validateContract(
        weights: Qwen35MLPWeights,
        hiddenSize: Int
    ) throws {
        try weights.gateProjection.validate()
        try weights.upProjection.validate()
        try weights.downProjection.validate()
        guard weights.gateProjection.shape.count == 2,
              weights.upProjection.shape == weights.gateProjection.shape,
              weights.gateProjection.shape[1] == hiddenSize,
              weights.downProjection.shape
                == [hiddenSize, weights.gateProjection.shape[0]]
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 MLP weight shapes are invalid"
            )
        }
    }
}
