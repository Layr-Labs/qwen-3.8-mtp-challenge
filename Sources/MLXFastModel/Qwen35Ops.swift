import MLX
import MLXNN

/// Architecture-neutral tensor primitives used by the custom Qwen35 path.
public enum Qwen35Ops {
    /// Avoid an otherwise redundant graph node when the dtype already matches.
    @inline(__always)
    public static func cast(_ input: MLXArray, to dtype: DType) -> MLXArray {
        input.dtype == dtype ? input : input.asType(dtype)
    }

    public static func embedding(
        inputIDs: MLXArray,
        weight: Qwen35LinearWeight
    ) -> MLXArray {
        guard let scales = weight.scales else {
            return weight.weight[inputIDs]
        }
        return dequantized(
            weight.weight[inputIDs],
            scales: scales[inputIDs],
            biases: weight.biases.map { $0[inputIDs] },
            groupSize: weight.groupSize,
            bits: weight.bits,
            mode: .affine
        )
    }

    public static func linear(
        _ input: MLXArray,
        _ weight: Qwen35LinearWeight
    ) -> MLXArray {
        guard let scales = weight.scales else {
            return matmul(input, weight.weight.T)
        }
        return quantizedMM(
            input,
            weight.weight,
            scales: scales,
            biases: weight.biases,
            transpose: true,
            groupSize: weight.groupSize,
            bits: weight.bits,
            mode: .affine
        )
    }

    /// Qwen35's learned RMSNorm uses the checkpoint weight directly.
    @inline(__always)
    public static func rmsNorm(
        _ input: MLXArray,
        weight: MLXArray,
        eps: Double
    ) -> MLXArray {
        MLXFast.rmsNorm(input, weight: weight, eps: Float(eps))
    }

    /// Weightless RMSNorm used for Gated DeltaNet Q/K normalization.
    @inline(__always)
    public static func rmsNorm(
        _ input: MLXArray,
        eps: Double
    ) -> MLXArray {
        MLXFast.rmsNorm(
            input,
            weight: MLXArray.mlxNone,
            eps: Float(eps)
        )
    }

    /// Precise gated RMSNorm from pinned `Qwen3NextRMSNormGated`:
    /// normalize first, then compute SiLU(gate) and the product in float32.
    public static func preciseGatedRMSNorm(
        _ input: MLXArray,
        gate: MLXArray,
        weight: MLXArray,
        eps: Double
    ) -> MLXArray {
        let normalized = rmsNorm(input, weight: weight, eps: eps)
        let preciseGate = silu(gate.asType(.float32))
        return (preciseGate * normalized.asType(.float32))
            .asType(input.dtype)
    }
}
