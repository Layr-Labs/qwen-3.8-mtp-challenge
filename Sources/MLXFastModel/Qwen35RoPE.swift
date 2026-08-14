import MLX
import MLXFastCore
import MLXLMCommon

/// Qwen35 partial rotary embedding for the text-only tower.
///
/// The pinned model rotates the first `0.25 * 256 == 64` dimensions with
/// theta 10,000,000 and leaves the remaining 192 dimensions unchanged.
/// Qwen's interleaved MRoPE position channels are identical for text tokens,
/// so the pinned text-only formula reduces exactly to ordinary partial RoPE.
/// This matches `Qwen35Attention` in the pinned MLXLLM `Qwen35.swift`;
/// the explicit reduction is also shown by MLXVLM's Qwen35 MRoPE helper.
public final class Qwen35RoPE {
    public let headDimensions: Int
    public let rotaryDimensions: Int
    public let theta: Float
    public let mropeSection: [Int]

    public init(
        headDimensions: Int,
        spec: Qwen35RopeSpec
    ) throws {
        guard headDimensions > 0, headDimensions.isMultiple(of: 2) else {
            throw MLXFastError.invalidInput(
                "Qwen35 RoPE head dimension must be positive and even"
            )
        }
        guard spec.type == "default", spec.theta.isFinite, spec.theta > 0 else {
            throw MLXFastError.invalidInput(
                "Qwen35 text RoPE requires a positive default theta"
            )
        }
        guard spec.mropeInterleaved else {
            throw MLXFastError.invalidInput(
                "Qwen35 text RoPE requires interleaved MRoPE metadata"
            )
        }

        // Pinned Qwen35.swift computes this through Float before truncating.
        let scaledRotaryDimensions =
            Float(headDimensions) * Float(spec.partialRotaryFactor)
        guard scaledRotaryDimensions.isFinite,
              scaledRotaryDimensions > 0,
              scaledRotaryDimensions < Float(Int.max)
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 partial RoPE dimension is outside Int range"
            )
        }
        let rotaryDimensions = Int(scaledRotaryDimensions)
        guard spec.mropeSection.count == 3,
              spec.mropeSection.allSatisfy({ $0 > 0 })
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 partial MRoPE sections must contain three positive values"
            )
        }
        let mropeSectionTotal = try qwen35CheckedSum(
            spec.mropeSection,
            label: "Qwen35 partial MRoPE section sum"
        )
        guard rotaryDimensions > 0,
              rotaryDimensions <= headDimensions,
              rotaryDimensions.isMultiple(of: 2),
              mropeSectionTotal == rotaryDimensions / 2
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 partial MRoPE dimensions are invalid"
            )
        }

        self.headDimensions = headDimensions
        self.rotaryDimensions = rotaryDimensions
        self.theta = Float(spec.theta)
        self.mropeSection = spec.mropeSection
    }

    public func applied(to input: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(
            input,
            dimensions: rotaryDimensions,
            traditional: false,
            base: theta,
            scale: 1,
            offset: offset
        )
    }

    public func applied(to input: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            input,
            dimensions: rotaryDimensions,
            traditional: false,
            base: theta,
            scale: 1,
            offset: offset
        )
    }

    /// Uses graph-visible offsets for compilable caches and scalar offsets for
    /// ordinary KV caches, matching pinned `applyRotaryPosition`.
    public func applied(
        to input: MLXArray,
        cache: (any KVCache)?,
        fallbackOffset: Int
    ) -> MLXArray {
        if let offset = graphOffsetArray(for: cache) {
            return applied(to: input, offset: offset)
        }
        return applied(to: input, offset: cache?.offset ?? fallbackOffset)
    }
}
