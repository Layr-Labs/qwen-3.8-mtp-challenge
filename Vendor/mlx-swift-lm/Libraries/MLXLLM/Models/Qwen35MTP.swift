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

    // MARK: compiled draft step (proposal side, L == 1)

    /// Compiled residual/MLP tail: attention projection -> residual ->
    /// post-norm -> fused gate/up GEMM -> silu*mul -> down GEMM -> residual,
    /// folded into one dispatch. Lazily built; nil until the MLP proves the
    /// declared quantized head's affine-4/g64 geometry.
    private var _tailCompiled:
        (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    /// One full decoder-layer draft step with the elementwise/reshape glue
    /// compiled away. Head-only; returns nil (caller falls back) unless the
    /// quantized geometry guards hold.
    func compiledDraftStep(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray? {
        guard let fusedMLP = mlp as? Qwen35FusedMLP else { return nil }
        guard let attnProj = selfAttn.headCompiledForward(
            x,
            preNormWeight: inputLayerNorm.weight,
            preNormEps: inputLayerNorm.eps,
            mask: mask,
            cache: cache)
        else { return nil }
        if _tailCompiled == nil {
            guard let parts = fusedMLP.headQuantizedParts() else { return nil }
            let postW = postAttentionLayerNorm.weight
            let postEps = postAttentionLayerNorm.eps
            let guW = parts.guW
            let guS = parts.guS
            let guZ = parts.guZ
            let gateOut = parts.gateOut
            let dW = parts.dW
            let dS = parts.dS
            let dZ = parts.dZ
            _tailCompiled = compile(shapeless: false) {
                (xIn: MLXArray, attn: MLXArray) -> MLXArray in
                let h = xIn + attn
                let n = MLXFast.rmsNorm(h, weight: postW, eps: postEps)
                let y = quantizedMM(
                    n, guW, scales: guS, biases: guZ, transpose: true,
                    groupSize: 64, bits: 4, mode: .affine)
                let g = y[.ellipsis, ..<gateOut]
                let u = y[.ellipsis, gateOut...]
                let d = quantizedMM(
                    silu(g) * u, dW, scales: dS, biases: dZ, transpose: true,
                    groupSize: 64, bits: 4, mode: .affine)
                return h + d
            }
        }
        guard let tail = _tailCompiled else { return nil }
        return tail(x, attnProj)
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

    /// Compiled fuse front: embedding/hidden norms -> concat -> fc GEMM in
    /// one dispatch. The embedding gather stays outside (it may be a
    /// quantized embedding whose lookup is its own primitive).
    private var _frontCompiled:
        (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    /// Single-draft-step fast path with the chain's elementwise glue
    /// compiled away. Proposal side only; nil (fall back) unless the head
    /// is the declared affine-4/g64 quantized geometry with one layer.
    private func compiledSingleStep(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray? {
        guard layers.count == 1, cache.count == 1,
              hidden.dim(0) == 1, hidden.dim(1) == 1,
              nextTokenIds.dim(1) == 1,
              let fcQ = fc as? QuantizedLinear,
              fcQ.groupSize == 64, fcQ.bits == 4, fcQ.mode == .affine,
              let fcZ = fcQ.biases
        else { return nil }
        if _frontCompiled == nil {
            let eW = preFcNormEmbedding.weight
            let eEps = preFcNormEmbedding.eps
            let hW = preFcNormHidden.weight
            let hEps = preFcNormHidden.eps
            let fcW = fcQ.weight
            let fcS = fcQ.scales
            _frontCompiled = compile(shapeless: false) {
                (embeds: MLXArray, hiddenIn: MLXArray) -> MLXArray in
                let e = MLXFast.rmsNorm(embeds, weight: eW, eps: eEps)
                let h = MLXFast.rmsNorm(hiddenIn, weight: hW, eps: hEps)
                return quantizedMM(
                    concatenated([e, h], axis: -1), fcW, scales: fcS,
                    biases: fcZ, transpose: true, groupSize: 64, bits: 4,
                    mode: .affine)
            }
        }
        guard let front = _frontCompiled else { return nil }
        let fused = front(embedTokens(nextTokenIds), hidden)
        let mask = createAttentionMask(h: fused, cache: cache[0])
        guard let out = layers[0].compiledDraftStep(
            fused, mask: mask, cache: cache[0])
        else { return nil }
        return norm(out)
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // Proposal-side compiled fast path for the chained single-token
        // draft step (the hot shape: every drafting round runs this once
        // per draft). Falls back to the ordinary chain whenever any guard
        // fails, including the pinned bf16 head.
        if let compiled = compiledSingleStep(
            hidden: hidden, nextTokenIds: nextTokenIds,
            embedTokens: embedTokens, cache: cache)
        {
            return compiled
        }
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
