// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/utils/sampling.py — mx.compile-free samplers.
// https://github.com/jundot/omlx/blob/main/omlx/utils/sampling.py
//
// The mlx-lm Python version decorates sampling functions with
// @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state),
// which fails to advance RNG state after the first call in long-running
// server environments, making identical prompts produce identical output
// even at temperature > 1. Ours does not have that issue because we never
// applied mx.compile, but we mirror omlx's structure for compatibility.

import Foundation
import MLX
import MLXRandom

/// Apply top-p (nucleus) filtering — keep the smallest set of tokens whose
/// cumulative probability mass is at least `topP`.
public func applyTopP(_ logprobs: MLXArray, topP: Float) -> MLXArray {
    let probs = exp(logprobs)
    let sortedIndices = argSort(logprobs, axis: -1)
    let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)
    let cumulativeProbs = cumsum(sortedProbs, axis: -1)

    // Scatter cumulative probabilities back to original token order
    let vocab = sortedIndices.dim(-1)
    let arange = MLXArray(0 ..< Int32(vocab))
    let inverseIndices = putAlong(
        MLXArray.zeros(like: sortedIndices),
        sortedIndices,
        values: arange,
        axis: -1
    )
    let cumProbsScattered = takeAlong(cumulativeProbs, inverseIndices, axis: -1)

    return MLX.where(
        cumProbsScattered .> (1 - topP),
        logprobs,
        MLXArray(-Float.infinity)
    )
}

/// Apply min-p filtering — drop tokens with probability below
/// `max(p) * minP`, while always keeping at least `minTokensToKeep` tokens.
public func applyMinP(
    _ logprobs: MLXArray,
    minP: Float,
    minTokensToKeep: Int = 1
) -> MLXArray {
    let maxLogprob = logprobs.max(axis: -1, keepDims: true)
    let threshold = maxLogprob + log(Float(minP))
    var tokensToRemove = logprobs .< threshold

    if minTokensToKeep > 1 {
        let topIndices = argPartition(logprobs, kth: -minTokensToKeep, axis: -1)
        let keep = topIndices[0..., (-minTokensToKeep)...]
        tokensToRemove = putAlong(tokensToRemove, keep, values: MLXArray(false), axis: -1)
    }

    return MLX.where(tokensToRemove, MLXArray(-Float.infinity), logprobs)
}

/// Apply top-k filtering — keep only the `topK` highest-probability tokens.
public func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
    let vocab = logprobs.dim(-1)
    guard topK > 0, topK < vocab else { return logprobs }

    let maskIdx = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
    return putAlong(
        logprobs,
        maskIdx,
        values: MLXArray(-Float.infinity),
        axis: -1
    )
}

/// Sample a token ID using categorical sampling from `logits / temp`.
public func categoricalSampling(logits: MLXArray, temp: Float) -> MLXArray {
    MLXRandom.categorical(logits * (1 / temp))
}

/// Build a row-sampling function from generation parameters.
///
/// Returns `argMax` when `temperature == 0`; otherwise composes optional
/// top-p / min-p / top-k filters and finishes with categorical sampling.
///
/// This mirrors `omlx.utils.sampling.make_sampler` and
/// `mlx_lm.sample_utils.make_sampler` behavior.
public func makeRowSampler(
    temperature: Float = 0.0,
    topP: Float = 0.0,
    minP: Float = 0.0,
    minTokensToKeep: Int = 1,
    topK: Int = 0
) -> @Sendable (MLXArray) -> MLXArray {
    if temperature == 0 {
        return { logits in argMax(logits, axis: -1) }
    }

    return { @Sendable logits in
        var lp = logits * (1.0 / temperature)

        if topP > 0, topP < 1 {
            lp = applyTopP(lp, topP: topP)
        }
        if minP != 0.0 {
            lp = applyMinP(lp, minP: minP, minTokensToKeep: minTokensToKeep)
        }
        if topK > 0 {
            lp = applyTopK(lp, topK: topK)
        }

        return MLXRandom.categorical(lp, axis: -1)
    }
}
