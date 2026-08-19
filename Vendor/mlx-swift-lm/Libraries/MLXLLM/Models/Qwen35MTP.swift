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

// MARK: - MTP conv guard

/// DFlash2 local-convolution guard. Proposal-only, zero-training.
/// `MLXFAST_MTP_CONV=0` disables fixed 0.9/0.1 mixing; default ON for ranked
/// where suffix-decay matters. Set MLXFAST_MTP_CONV=0 for hot-path microbench.
private var _mtpConvEnabled: Bool {
    ProcessInfo.processInfo.environment["MLXFAST_MTP_CONV"] == "1"
}

// MARK: - Two-tap Dynamic Depthwise Conv (per-layer, per-site)

/// Lightweight local 2-tap depthwise conv for the MTP head (DFlash2).
/// `Conv_k(x)_t = k0 ⊙ x_t + k1 ⊙ x_{t-1}` with per-channel base weights
/// and optional per-group dynamic correction (shared across 16 channels).
/// Identity init (`baseK0=ones, baseK1=zeros`) guarantees no regression
/// when guard is disabled or when `useDynamic=false` with identity fallback.
/// Fixed zero-training variant uses constants `k0=0.9, k1=0.1` (or `1/0`
/// identity) without new learned weights.
/// First position's `x_{t-1}` reads the caller-supplied `prev` (last verified
/// token's hidden, `[B,1,H]`); when `prev==nil` the tap sees zeros.
final class MTPDepthwiseConv2Tap: Module {
    let hiddenSize: Int
    let groups: Int // hiddenSize / 16, shared across 16 channels
    let useDynamic: Bool
    @ModuleInfo(key: "base_k0") var baseK0: MLXArray // [H] ones (identity)
    @ModuleInfo(key: "base_k1") var baseK1: MLXArray // [H] zeros (identity)
    // Dynamic path (optional, only when useDynamic == true)
    @ModuleInfo(key: "dyn_down") var dynDown: Linear?
    @ModuleInfo(key: "dyn_up") var dynUp: Linear?

    init(hiddenSize: Int, useDynamic: Bool = false, bottleneck: Int = 32) {
        self.hiddenSize = hiddenSize
        self.groups = hiddenSize / 16
        self.useDynamic = useDynamic
        _baseK0.wrappedValue = MLXArray.ones([hiddenSize])
        _baseK1.wrappedValue = MLXArray.zeros([hiddenSize])
        if useDynamic {
            _dynDown.wrappedValue = Linear(hiddenSize, bottleneck, bias: false)
            _dynUp.wrappedValue = Linear(bottleneck, groups * 2, bias: false)
            // zero-init up projection so delta==0 at step 0 (identity start)
            // caller may additionally zero weights after init if needed
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, prev: MLXArray? = nil) -> MLXArray {
        // Proposal-only fast path: no new dispatch when disabled.
        guard _mtpConvEnabled else { return x }
        // Fixed lightweight path (zero-training) — 0.9/0.1 mean filter.
        // baseK0/baseK1 remain identity-initialized for future learned variant.
        if !useDynamic {
            let T = x.dim(1)
            let prevRow: MLXArray
            if let p = prev {
                if p.ndim == 3 && p.dim(1) == 1 {
                    prevRow = p
                } else if p.ndim == 3 {
                    // multi-row prev: take its last row (caller may pass [B,T,H])
                    prevRow = p[0..., (p.dim(1) - 1) ..< p.dim(1), 0...]
                } else {
                    prevRow = MLXArray.zeros([x.dim(0), 1, hiddenSize], dtype: x.dtype)
                }
            } else {
                prevRow = MLXArray.zeros([x.dim(0), 1, hiddenSize], dtype: x.dtype)
            }
            let xPrev: MLXArray
            if T == 1 {
                xPrev = prevRow
            } else {
                let prefix = x[0..., 0 ..< (T - 1), 0...]
                xPrev = concatenated([prevRow, prefix], axis: 1)
            }
            // Fixed 2-tap coefficients: 0.9 local + 0.1 history.
            // Alternative identity is k0=1.0/k1=0.0; 0.9/0.1 gives mild smoothing.
            let k0: MLXArray = MLXArray(0.9, dtype: x.dtype)
            let k1: MLXArray = MLXArray(0.1, dtype: x.dtype)
            // Depthwise: elementwise, no channel mixing, bf16 deterministic.
            return k0 * x + k1 * xPrev
        }
        // Dynamic lightweight variant (requires head retrain, declared via manifest).
        // base + delta per group (groups = H/16, bottleneck=32).
        // Kept for structure; identity when delta==0.
        let T = x.dim(1)
        // For dynamic path we still need shift-concat handling.
        let prevRow: MLXArray
        if let p = prev {
            if p.ndim == 3 && p.dim(1) == 1 {
                prevRow = p
            } else if p.ndim == 3 {
                prevRow = p[0..., (p.dim(1) - 1) ..< p.dim(1), 0...]
            } else {
                prevRow = MLXArray.zeros([x.dim(0), 1, hiddenSize], dtype: x.dtype)
            }
        } else {
            prevRow = MLXArray.zeros([x.dim(0), 1, hiddenSize], dtype: x.dtype)
        }
        let xPrev: MLXArray
        if T == 1 {
            xPrev = prevRow
        } else {
            let prefix = x[0..., 0 ..< (T - 1), 0...]
            xPrev = concatenated([prevRow, prefix], axis: 1)
        }
        // Dynamic delta generation would be:
        //   delta = tanh(up(silu(down(x)))) * 0.1  // [B,T,2*G]
        //   d0,d1 split and broadcast to 16 channels
        //   k0 = baseK0 + bcast(d0), k1 = baseK1 + bcast(d1)
        // Kept as structure with zero delta for proposal-only zero-training phase.
        // Use base weights broadcast to [1,1,H] for elementwise.
        let k0b = baseK0.reshaped([1, 1, hiddenSize])
        let k1b = baseK1.reshaped([1, 1, hiddenSize])
        return k0b * x + k1b * xPrev
    }
}

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
    // DFlash2 local conv: four 2-tap depthwise per layer (pre/post attention, pre/post MLP)
    @ModuleInfo(key: "conv_pre_attn") var convPreAttn: MTPDepthwiseConv2Tap
    @ModuleInfo(key: "conv_post_attn") var convPostAttn: MTPDepthwiseConv2Tap
    @ModuleInfo(key: "conv_pre_mlp") var convPreMlp: MTPDepthwiseConv2Tap
    @ModuleInfo(key: "conv_post_mlp") var convPostMlp: MTPDepthwiseConv2Tap

    init(_ args: Qwen35TextConfiguration, useDynamicConv: Bool = false) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            // Same fused gate/up MLP as the backbone layers; here the linears
            // stay bf16 and the fuse takes the plain-weight path. Head side —
            // proposal-only, no exactness constraint. Flag for M=1 unfuse (9b4550d).
            let headMLP = Qwen35FusedMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
            headMLP.isHeadMLP = true
            _mlp.wrappedValue = headMLP
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        let h = args.hiddenSize
        _convPreAttn.wrappedValue = MTPDepthwiseConv2Tap(hiddenSize: h, useDynamic: useDynamicConv)
        _convPostAttn.wrappedValue = MTPDepthwiseConv2Tap(hiddenSize: h, useDynamic: useDynamicConv)
        _convPreMlp.wrappedValue = MTPDepthwiseConv2Tap(hiddenSize: h, useDynamic: useDynamicConv)
        _convPostMlp.wrappedValue = MTPDepthwiseConv2Tap(hiddenSize: h, useDynamic: useDynamicConv)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        // Backward-compat entry without prev: identity prev (zeros)
        return callAsFunction(x, mask: mask, cache: cache, prev: nil)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?,
        prev: MLXArray? = nil
    ) -> MLXArray {
        // omlx: MTPDecoderLayer.__call__ with DFlash2 conv taps
        // Proposal-only: guard ensures no effect when MLXFAST_MTP_CONV=0
        if !_mtpConvEnabled {
            let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        }
        // Conv-pre-attention: local recurrence before attention frees attention for long context
        let x1 = convPreAttn(x, prev: prev)
        let r = selfAttn(inputLayerNorm(x1), mask: mask, cache: cache)
        let h = x + r
        // Conv-post-attention (pre-MLP boundary)
        let h1 = convPostAttn(h, prev: nil)
        let h2 = convPreMlp(h1, prev: nil)
        let h3 = h1 + (mlp as! UnaryLayer)(postAttentionLayerNorm(h2))
        return convPostMlp(h3, prev: nil)
    }

    /// Populate this layer's K/V history without computing a dead layer
    /// output. Only valid when no later MTP layer consumes that output.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        appendHistoryKV(x, cache: cache, prev: nil)
    }

    func appendHistoryKV(_ x: MLXArray, cache: any KVCache, prev: MLXArray? = nil) {
        if _mtpConvEnabled {
            let x1 = convPreAttn(x, prev: prev)
            selfAttn.appendHistoryKV(inputLayerNorm(x1), cache: cache)
        } else {
            selfAttn.appendHistoryKV(inputLayerNorm(x), cache: cache)
        }
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

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        return callAsFunction(hidden: hidden, nextTokenIds: nextTokenIds, embedTokens: embedTokens, cache: cache, prev: nil)
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache],
        prev: MLXArray? = nil
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

        // 3. Run each MTPDecoderLayer. Thread prev only to first layer's first tap
        //    for block-boundary handling (x_{-1} = last verified hidden's last row
        //    when T>1). Interior layers use nil (their own shifted history).
        var layerPrev: MLXArray? = prev
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            let effectivePrev: MLXArray? = (i == 0) ? layerPrev : nil
            fused = layer(fused, mask: mask, cache: c, prev: effectivePrev)
            layerPrev = nil
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
        return lastHiddenWithKVOnlyHistory(hidden: hidden, nextTokenIds: nextTokenIds, embedTokens: embedTokens, cache: cache, prev: nil)
    }

    func lastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache],
        prev: MLXArray? = nil
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

        let history = fused[0..., 0 ..< historyCount, 0...]
        layers[0].appendHistoryKV(history, cache: cache[0], prev: prev)

        let current = fused[0..., historyCount..., 0...]
        // Prev for current's 2-tap when T==1 is the last history row
        let currentPrev: MLXArray = history[0..., (historyCount - 1) ..< historyCount, 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0], prev: currentPrev))
    }

}
