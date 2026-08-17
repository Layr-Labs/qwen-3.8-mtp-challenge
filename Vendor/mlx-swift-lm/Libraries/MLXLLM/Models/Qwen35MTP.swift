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

    // Proposal-only derived cache. Compact draft IDs are confined to the
    // 98,304-token prefix plus Qwen's 26 control tokens, so the embedding half
    // of the bias-free fusion FC is input-independent for each possible draft.
    // Materialise that half once during untimed warmup and stream only the
    // hidden half of FC on chained proposal steps.
    private var compactEmbeddingFC: MLXArray?
    private var compactHiddenFC: QuantizedLinear?
    private static let compactPrefixCount = 98_304
    private static let compactControlStart = 248_044
    private static let compactControlEnd = 248_070

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
        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        var fused = fc(concatenated([e, h], axis: -1))

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

    /// Build the input-independent embedding-half FC table used by chained
    /// compact-vocabulary proposals. Returns false without changing execution
    /// when the declared head does not have the promoted affine/g64 geometry.
    func prepareCompactFusionLookup(embedTokens: Embedding) -> Bool {
        if compactEmbeddingFC != nil { return true }
        guard let quantized = fc as? QuantizedLinear,
              quantized.bias == nil,
              quantized.groupSize == 64,
              quantized.bits == 4,
              quantized.mode == .affine,
              quantized.biases != nil,
              quantized.shape.1 == 10_240,
              quantized.shape.0 == 5_120
        else { return false }

        let half = quantized.shape.1 / 2
        let packedHalf = half * quantized.bits / 32
        let groupsHalf = half / quantized.groupSize
        let embeddingFC = QuantizedLinear(
            weight: quantized.weight[0..., 0 ..< packedHalf],
            scales: quantized.scales[0..., 0 ..< groupsHalf],
            biases: quantized.biases.map { $0[0..., 0 ..< groupsHalf] },
            groupSize: quantized.groupSize,
            bits: quantized.bits,
            mode: quantized.mode)
        compactHiddenFC = QuantizedLinear(
            weight: quantized.weight[0..., packedHalf...],
            scales: quantized.scales[0..., groupsHalf...],
            biases: quantized.biases.map { $0[0..., groupsHalf...] },
            groupSize: quantized.groupSize,
            bits: quantized.bits,
            mode: quantized.mode)

        let ids = Array(Int32(0) ..< Int32(Self.compactPrefixCount))
            + Array(Int32(Self.compactControlStart) ..< Int32(Self.compactControlEnd))
        let chunkSize = 2_048
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((ids.count + chunkSize - 1) / chunkSize)
        for start in stride(from: 0, to: ids.count, by: chunkSize) {
            let end = min(start + chunkSize, ids.count)
            let tokenIDs = MLXArray(Array(ids[start ..< end])).reshaped([1, end - start])
            let normalized = preFcNormEmbedding(embedTokens(tokenIDs))
            let contribution = embeddingFC(normalized)[0]
            eval(contribution)
            chunks.append(contribution)
        }
        let table = concatenated(chunks, axis: 0)
        eval(table)
        compactEmbeddingFC = table
        return true
    }

    /// Chained proposal step whose token ID came from the compact draft
    /// selector. Falls back to the ordinary head before cache mutation if the
    /// untimed lookup preparation was unavailable.
    func compactTokenForward(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        guard let compactEmbeddingFC, let compactHiddenFC else {
            return callAsFunction(
                hidden: hidden, nextTokenIds: nextTokenIds,
                embedTokens: embedTokens, cache: cache)
        }
        let compactIDs = which(
            nextTokenIds .< Self.compactPrefixCount,
            nextTokenIds,
            nextTokenIds
                - (Self.compactControlStart - Self.compactPrefixCount))
        let embeddingContribution = take(
            compactEmbeddingFC, compactIDs, axis: 0)
        let h = preFcNormHidden(hidden)
        var fused = embeddingContribution + compactHiddenFC(h)
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            fused = layer(fused, mask: mask, cache: c)
        }
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

        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        let fused = fc(concatenated([e, h], axis: -1))
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }

}
