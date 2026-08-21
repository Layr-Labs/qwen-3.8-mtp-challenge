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
        // omlx: MTPDecoderLayer.__call__ — attention runs exactly once, inside
        // `forwardTail`; this wrapper only materialises the final residual add
        // for callers that do not fuse it with a following norm.
        let (h, mlpOut) = forwardTail(x, mask: mask, cache: cache)
        return h + mlpOut
    }

    /// The layer up to — but not including — its final residual add: returns
    /// `(h, mlpOut)` where the layer output is `h + mlpOut`. The module fuses
    /// that add with its own final RMSNorm (one launch, same kernel and same
    /// bf16-round-before-square arithmetic as the post-attention boundary
    /// below), so the add is never materialised on its own.
    func forwardTail(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> (MLXArray, MLXArray) {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        // The backbone's decoder layer has fused this residual+norm boundary
        // since `qwen35FusedResidualRMSNorm` landed; the head layer was left on
        // the eager pair. Same kernel, same bf16/5120 guard, same
        // bf16-round-before-square argument, so the values are bit-identical to
        // `h = x + r; postAttentionLayerNorm(h)` — one launch and one host graph
        // node instead of two, paid once per PROPOSED token (draftCount times a
        // round) rather than once per layer.
        if x.dtype == .bfloat16, r.dtype == .bfloat16, x.dim(-1) == 5120 {
            let (h, postAttnNorm) = qwen35FusedResidualRMSNorm(
                x: x, r: r,
                weight: postAttentionLayerNorm.weight,
                eps: postAttentionLayerNorm.eps)
            return (h, (mlp as! UnaryLayer)(postAttnNorm))
        }
        let h = x + r
        return (h, (mlp as! UnaryLayer)(postAttentionLayerNorm(h)))
    }

    /// Populate this layer's K/V history without computing a dead layer
    /// output. Only valid when no later MTP layer consumes that output.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        selfAttn.appendHistoryKV(inputLayerNorm(x), cache: cache)
    }
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

    /// `fc`'s input `[normE(embed(ids)) | normH(hidden)]` in ONE launch when the
    /// embedding is the 4-bit/group-64 affine table this checkpoint ships: the
    /// kernel reads each token's packed row and dequantises in-register with
    /// MLX's own `affine_dequantize` expression, so the three gathers and the
    /// dequantize launch (four per proposed token) disappear along with the
    /// two norms and the concat. Falls back to the eager embedding lookup plus
    /// `preFcConcat` when the table or shapes do not qualify. Proposal-only.
    private func preFcInput(
        nextTokenIds: MLXArray, embedTokens: Embedding, hidden: MLXArray
    ) -> MLXArray {
        if let quantized = embedTokens as? QuantizedEmbedding,
           hidden.dtype == .bfloat16, hidden.dim(-1) == 5120,
           preFcNormEmbedding.eps == preFcNormHidden.eps,
           let fused = qwen35QEmbedDualRMSNormConcat(
               tokenIds: nextTokenIds,
               embedding: quantized,
               hidden: hidden,
               eWeight: preFcNormEmbedding.weight,
               hWeight: preFcNormHidden.weight,
               eps: preFcNormEmbedding.eps)
        {
            return fused
        }
        return preFcConcat(embeds: embedTokens(nextTokenIds), hidden: hidden)
    }

    /// `norm(h + mlpOut)` as one fused residual+RMSNorm launch (the backbone's
    /// boundary kernel, bit-identical arithmetic) when the shapes qualify; the
    /// eager pair otherwise. Proposal-only.
    private func finalNormed(h: MLXArray, mlpOut: MLXArray) -> MLXArray {
        if h.dtype == .bfloat16, mlpOut.dtype == .bfloat16, h.dim(-1) == 5120 {
            return qwen35FusedResidualRMSNorm(
                x: h, r: mlpOut, weight: norm.weight, eps: norm.eps).normed
        }
        return norm(h + mlpOut)
    }

    /// Dual RMSNorm written straight into the `[e | h]` layout `fc` consumes.
    /// Same arithmetic as `qwen35DualRMSNorm` + `concatenated([e, h], -1)`;
    /// the extra concat launch is gone. Proposal-only.
    private func preFcConcat(embeds: MLXArray, hidden: MLXArray) -> MLXArray {
        if embeds.dtype == .bfloat16, hidden.dtype == .bfloat16,
           embeds.dim(-1) == 5120, hidden.dim(-1) == 5120,
           embeds.shape == hidden.shape,
           preFcNormEmbedding.eps == preFcNormHidden.eps
        {
            return qwen35DualRMSNormConcat(
                a: embeds, b: hidden,
                aWeight: preFcNormEmbedding.weight,
                bWeight: preFcNormHidden.weight,
                eps: preFcNormEmbedding.eps)
        }
        return concatenated(
            [preFcNormEmbedding(embeds), preFcNormHidden(hidden)], axis: -1)
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        var fused = fc(preFcInput(
            nextTokenIds: nextTokenIds, embedTokens: embedTokens, hidden: hidden))

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run each MTPDecoderLayer; the LAST layer's residual add is fused
        //    into the module norm below.
        guard let lastLayer = layers.last else { return norm(fused) }
        for (i, layer) in layers.dropLast().enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            fused = layer(fused, mask: mask, cache: c)
        }
        let lastIndex = layers.count - 1
        let lastCache: (any KVCache)? =
            lastIndex < cache.count ? cache[lastIndex] : nil
        let (h, mlpOut) = lastLayer.forwardTail(fused, mask: mask, cache: lastCache)

        // 4. Return pre-lm_head hidden (norm applied; lm_head is in TextModel).
        return finalNormed(h: h, mlpOut: mlpOut)
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

        let fused = fc(preFcInput(
            nextTokenIds: nextTokenIds, embedTokens: embedTokens, hidden: hidden))
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        let (h, mlpOut) = layers[0].forwardTail(current, mask: mask, cache: cache[0])
        return finalNormed(h: h, mlpOut: mlpOut)
    }

}
