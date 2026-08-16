// Copyright © 2026 Eigen Labs.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/qwen35_model.py  (MTPDecoderLayer, MTPModule)
//   patches/mlx_lm_mtp/__init__.py        (is_mtp_active / set_mtp_active)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Module-level MTP flag

/// Controls whether Qwen3.5/3.6 model inits attach the MTP head.
/// Set to `true` before calling `MLXLLM.load(...)` when MTP should be active.
/// Mirrors omlx `is_mtp_active()` / `set_mtp_active()` from
/// patches/mlx_lm_mtp/__init__.py.
public nonisolated(unsafe) var _qwen35MTPEnabled: Bool = false

// MARK: - MTPDecoderLayer

/// Full-attention transformer layer used inside the Qwen3.5/3.6 MTP head.
/// Unlike `Qwen35DecoderLayer`, this always uses full attention (never SSM/linear).
/// MoE config is honoured when `num_experts > 0`.
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPDecoderLayer
final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            // Same fused gate/up MLP as the backbone layers; here the linears
            // stay bf16 and the fuse takes the plain-weight path. Head side —
            // proposal-only, no exactness constraint.
            _mlp.wrappedValue = Qwen35FusedMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        // omlx: MTPDecoderLayer.__call__
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    /// Populate this layer's K/V history without computing a dead layer
    /// output. Only valid when no later MTP layer consumes that output.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        selfAttn.appendHistoryKV(inputLayerNorm(x), cache: cache)
    }
}

// MARK: - Compiled fusion FC (512-row seed + later multi-row flushes)

/// Compile the still-eager MTP fusion host graph: quantized embed gather +
/// two pre-FC RMSNorms + concat. `fc` stays a module call so a DFlash
/// wrapper, if present, keeps its kernel. Same 1e-6 RMS contract as the
/// eager `RMSNorm` on this model. Serial depth-0 never enters MTP.
private let qwen35CompiledFusionEmbedNorms:
    @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
{
    let body:
        @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) ->
        MLXArray =
    { ids, hidden, embW, embS, embB, eW, hW in
        let embeds = dequantized(
            embW[ids], scales: embS[ids], biases: embB[ids],
            groupSize: 64, bits: 4, mode: .affine)
        let e = MLXFast.rmsNorm(embeds, weight: eW, eps: 1e-6)
        let h = MLXFast.rmsNorm(hidden, weight: hW, eps: 1e-6)
        return concatenated([e, h], axis: -1)
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Fallback when the backbone embed is not 4-bit/g64 affine: compile only
/// the two pre-FC norms + concat after the eager embed lookup.
private let qwen35CompiledFusionNormsConcat:
    @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
{
    let body: @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
    { embeds, hidden, eW, hW in
        let e = MLXFast.rmsNorm(embeds, weight: eW, eps: 1e-6)
        let h = MLXFast.rmsNorm(hidden, weight: hW, eps: 1e-6)
        return concatenated([e, h], axis: -1)
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

private func qwen35FuseHiddenAndNextToken(
    hidden: MLXArray,
    nextTokenIds: MLXArray,
    embedTokens: Embedding,
    preFcNormEmbedding: RMSNorm,
    preFcNormHidden: RMSNorm,
    fc: Linear
) -> MLXArray {
    let concat: MLXArray
    if let qe = embedTokens as? QuantizedEmbedding,
       qe.groupSize == 64, qe.bits == 4, qe.mode == .affine,
       let embB = qe.biases
    {
        concat = qwen35CompiledFusionEmbedNorms(
            nextTokenIds, hidden,
            qe.weight, qe.scales, embB,
            preFcNormEmbedding.weight, preFcNormHidden.weight)
    } else {
        let embeds = embedTokens(nextTokenIds)
        concat = qwen35CompiledFusionNormsConcat(
            embeds, hidden,
            preFcNormEmbedding.weight, preFcNormHidden.weight)
    }
    return fc(concat)
}

// MARK: - MTPModule

/// Multi-Token Prediction head for Qwen3.5/3.6.
///
/// Fuses the backbone's pre-norm hidden state at position t with the embedding of
/// the sampled main token (t+1) to predict the draft token at (t+2).
///
/// Architecture (port of PR #990):
/// ```
/// pre_fc_norm_hidden:    RMSNorm(hidden_size)
/// pre_fc_norm_embedding: RMSNorm(hidden_size)
/// fc:                    Linear(hidden_size * 2 → hidden_size, bias: false)
/// layers:                [MTPDecoderLayer]  × mtp_num_hidden_layers
/// norm:                  RMSNorm(hidden_size)
/// ```
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPModule
final class Qwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    // `layers` uses the default ModuleInfo key derived from the property name.
    let layers: [Qwen35MTPDecoderLayer]
    let norm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _preFcNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFcNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self.layers = (0 ..< args.mtpNumHiddenLayers).map { _ in
            Qwen35MTPDecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        var fused = qwen35FuseHiddenAndNextToken(
            hidden: hidden,
            nextTokenIds: nextTokenIds,
            embedTokens: embedTokens,
            preFcNormEmbedding: preFcNormEmbedding,
            preFcNormHidden: preFcNormHidden,
            fc: fc)

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run each MTPDecoderLayer.
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            fused = layer(fused, mask: mask, cache: c)
        }

        // 4. Return pre-lm_head hidden (norm applied; lm_head is in TextModel).
        return norm(fused)
    }

    /// Run one proposal flush while omitting leading-row outputs that have no
    /// consumer. Every supplied row still participates in the fusion stage and
    /// contributes K/V state; only the final row needs a full decoder output.
    /// Multi-layer heads fail closed before mutating cache state.
    func lastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray? {
        guard layers.count == 1, cache.count == 1,
              hidden.dim(1) > 1,
              nextTokenIds.dim(1) == hidden.dim(1)
        else { return nil }

        let fused = qwen35FuseHiddenAndNextToken(
            hidden: hidden,
            nextTokenIds: nextTokenIds,
            embedTokens: embedTokens,
            preFcNormEmbedding: preFcNormEmbedding,
            preFcNormHidden: preFcNormHidden,
            fc: fc)
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }

}
