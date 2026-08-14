// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
//
// Gemma 4 VLM — vision-language model with:
//   - Linear patch embedding with 2D position embeddings
//   - 2D multidimensional RoPE for vision encoder
//   - VisionPooler for downsampling patches
//   - MultimodalEmbedder projecting vision features into text space
//   - Full Gemma4 text decoder (MoE 26B or Dense 31B)
//
// Python reference: mlx_vlm/models/gemma4/

import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Darkbloom note: wholesale port of vMLX Libraries/MLXVLM/Models/Gemma4.swift
// (the proven 79.6 TPS Gemma4-26B-A4B decode path). Adapted to our integration
// base: HardwareInfo -> MLXHardwareInfo, RuntimeMoETopKOverride dropped (plain
// decode), graphOffsetArray stubbed (host-int offset fallback), audio guard and
// cacheScopeSalt removed (not in our LMInput), QwenVL.intExtent replaced by a
// local helper. MTP/capture hooks intentionally dropped (provider only references
// Gemma4 in comments; the decode benchmark uses only the LanguageModel API).

// Compiled logit softcap — fuses divide + tanh + multiply into one Metal dispatch.
private let compiledLogitSoftcap: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { (x: MLXArray, cap: MLXArray) -> MLXArray in
        tanh(x / cap) * cap
    }
    return MLXHardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

// Graph-traceable / per-row RoPE offset for KV-sharing layers. Mirrors
// `applyRotaryPosition` (used by the non-shared path) and the proven vMLX
// behavior: when the source cache carries per-sequence offsets
// (`BatchPositionedKVCache` — i.e. BatchKVCache / BatchRotatingKVCache /
// QuantizedBatchKVCache) we thread its per-row `batchOffset`, so mixed-length
// continuous batches RoPE each row at its own position instead of a single
// scalar (max-across-rows) offset that corrupts the shorter rows. A
// `CompilableKVCache` exposes a graph-traceable `offsetArray`, so KV-sharing
// layers keep the shared-KV offset inside the compiled graph (no host readback
// of `cache.offset`). The `+ 0` snapshots the value before the source layer's
// `cache.update` advances it (`CompilableKVCache` mutates `offsetArray` in
// place via `_updateInternal`; `+ 0` references the captured array, not the
// repointed wrapper). Plain B=1 / non-compiled decode uses RotatingKVCache /
// KVCacheSimple, which match neither branch, so this returns nil and the caller
// keeps the scalar host-int offset — the fast text decode hot path is unchanged.
private func gemma4VLMGraphOffsetArray(for cache: KVCache?) -> MLXArray? {
    if let batchCache = cache as? BatchPositionedKVCache {
        return batchCache.batchOffset + 0
    }
    if let compilable = cache as? CompilableKVCache {
        return compilable.offsetArray + 0
    }
    return nil
}

// Local replacement for vMLX's `QwenVL.intExtent` (absent in our tree). Rejects
// zero-area / non-finite extents so the scale-factor math below cannot divide by
// zero or trap inside `Int(floor(.nan))` / `Int(.infinity)`.
private func gemma4IntExtent(_ size: CGSize) throws -> (Int, Int) {
    let w = size.width
    let h = size.height
    guard w.isFinite, h.isFinite, w > 0, h > 0 else {
        throw VLMError.imageProcessingFailure(
            "Gemma4: image has a zero-area or non-finite extent (w=\(w), h=\(h)).")
    }
    return (Int(h), Int(w))
}


// MARK: - Shared Norm Utilities

/// Standard Gemma4 RMSNorm — weight used directly, NO +1 offset
private class G4RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// Vision RMSNorm — full float32 computation for precision
private class VisionRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let v = (xf * xf).mean(axis: -1, keepDims: true)
        return ((xf * rsqrt(v + eps)) * weight.asType(.float32)).asType(x.dtype)
    }
}

/// Parameterless RMS normalization
func rmsNormNoScale(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
}

private func visionRmsNormNoScale(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let xf = x.asType(.float32)
    let v = (xf * xf).mean(axis: -1, keepDims: true)
    return (xf * rsqrt(v + eps)).asType(x.dtype)
}

// MARK: - Configurations

public struct Gemma4VisionConfig: Codable, Sendable {
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let patchSize: Int
    let positionEmbeddingSize: Int
    let defaultOutputLength: Int
    let poolingKernelSize: Int
    let standardize: Bool
    let useClippedLinears: Bool
    let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case patchSize = "patch_size"
        case positionEmbeddingSize = "position_embedding_size"
        case defaultOutputLength = "default_output_length"
        case poolingKernelSize = "pooling_kernel_size"
        case standardize
        case useClippedLinears = "use_clipped_linears"
    }

    enum TopKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 16
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 12
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        positionEmbeddingSize = try c.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10240
        defaultOutputLength = try c.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        standardize = try c.decodeIfPresent(Bool.self, forKey: .standardize) ?? false
        useClippedLinears = try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? false

        if let rc = try? decoder.container(keyedBy: TopKeys.self),
           let rp = try? rc.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters),
           let t = rp["rope_theta"]?.asFloat()
        {
            ropeTheta = t
        } else {
            ropeTheta = 100.0
        }
    }
}

/// Inline text config for VLM — mirrors MLXLLM's Gemma4TextConfiguration
struct G4TextConfig: Codable, Sendable {
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numGlobalKeyValueHeads: Int?
    let headDim: Int
    let globalHeadDim: Int
    let intermediateSize: Int
    let vocabSize: Int
    let rmsNormEps: Float
    let slidingWindow: Int
    let layerTypes: [String]
    let finalLogitSoftcapping: Float?
    let tieWordEmbeddings: Bool
    let attentionBias: Bool
    let attentionKEqV: Bool
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int
    let numKvSharedLayers: Int
    let useDoubleWideMlp: Bool
    let enableMoeBlock: Bool
    let moeIntermediateSize: Int
    let numExperts: Int
    let topKExperts: Int
    let ropeTraditional: Bool
    let ropeParameters: [String: [String: StringOrNumber]]
    // "vision" enables Gemma4 blockwise bidirectional attention within
    // image/video soft-token spans during prefill. nil/other => ordinary causal.
    let useBidirectionalAttention: String?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case attentionKEqV = "attention_k_eq_v"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKvSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case enableMoeBlock = "enable_moe_block"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case ropeTraditional = "rope_traditional"
        case ropeParameters = "rope_parameters"
        case useBidirectionalAttention = "use_bidirectional_attention"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2816
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 30
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        numGlobalKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2112
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        finalLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionKEqV = try c.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        let decodedHiddenSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        var decodedVocabSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 0
        // PLE coherence: paired E2B/E4B fields. See Gemma4Text.swift / deep-trace §7.6.
        // `hidden_size_per_layer_input == 0` is the authoritative PLE-off signal
        // for full Gemma4 rows. Some shipped configs still carry the ordinary
        // vocab size in `vocab_size_per_layer_input`; normalize that to PLE off.
        // The opposite shape remains invalid because a positive PLE hidden size
        // requires a positive PLE vocab.
        if decodedHiddenSizePerLayerInput == 0 {
            decodedVocabSizePerLayerInput = 0
        } else if decodedVocabSizePerLayerInput == 0 {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenSizePerLayerInput, in: c,
                debugDescription:
                    "Gemma4 PLE config incoherent: hidden_size_per_layer_input=\(decodedHiddenSizePerLayerInput) "
                    + "and vocab_size_per_layer_input=\(decodedVocabSizePerLayerInput); vocab must be positive when PLE hidden size is positive.")
        }
        hiddenSizePerLayerInput = decodedHiddenSizePerLayerInput
        vocabSizePerLayerInput = decodedVocabSizePerLayerInput
        numKvSharedLayers = try c.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        useDoubleWideMlp = try c.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        enableMoeBlock = try c.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        topKExperts = try c.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 0
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeParameters = try c.decodeIfPresent([String: [String: StringOrNumber]].self, forKey: .ropeParameters) ?? [:]
        useBidirectionalAttention = try c.decodeIfPresent(String.self, forKey: .useBidirectionalAttention)
    }
}

public struct Gemma4Configuration: Codable, Sendable {
    let textConfig: G4TextConfig
    let visionConfig: Gemma4VisionConfig
    let modelType: String
    let imageTokenId: Int
    let videoTokenId: Int?
    let visionSoftTokensPerImage: Int
    let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(G4TextConfig.self, forKey: .textConfig)
        visionConfig = try c.decode(Gemma4VisionConfig.self, forKey: .visionConfig)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4"
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258_880
        // Default to the Gemma4 video token id so the model stays in sync with the
        // processor (which emits video placeholders) even when config.json omits it.
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 258_884
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage)
            ?? visionConfig.defaultOutputLength
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantization)
    }
}

// MARK: - Vision Components

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

private func applyMultidimensionalRope(_ inputs: MLXArray, positions: MLXArray, base: Float) -> MLXArray {
    let headDim = inputs.dim(-1)
    let ndim = positions.dim(-1)
    let chPerDim = 2 * (headDim / (2 * ndim))
    let halfPerDim = chPerDim / 2

    var parts: [MLXArray] = []
    for d in 0 ..< ndim {
        let xPart = inputs[.ellipsis, (d * chPerDim) ..< ((d + 1) * chPerDim)]
        let freqExp = (2.0 / Float(chPerDim)) * MLXArray(0 ..< halfPerDim).asType(.float32)
        let timescale = pow(base, freqExp)
        let sinInp = positions[.ellipsis, d ..< (d + 1)].asType(.float32) / timescale
        var cosD = cos(sinInp)
        var sinD = sin(sinInp)
        cosD = concatenated([cosD, cosD], axis: -1).asType(inputs.dtype)
        sinD = concatenated([sinD, sinD], axis: -1).asType(inputs.dtype)
        cosD = expandedDimensions(cosD, axis: 2)
        sinD = expandedDimensions(sinD, axis: 2)
        parts.append(xPart * cosD + rotateHalf(xPart) * sinD)
    }
    return concatenated(parts, axis: -1)
}

private func oneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    (expandedDimensions(indices, axis: -1) .== MLXArray(0 ..< Int32(numClasses))).asType(.float32)
}

// MARK: - Bidirectional vision attention overlay

/// Per-token block id for vision spans: each contiguous run of vision tokens
/// shares an id, non-vision tokens get -1. Mirrors Python
/// `_block_sequence_ids_for_mask`.
private func gemma4VisionBlockIds(_ isVision: MLXArray) -> MLXArray {
    let length = isVision.dim(1)
    let leading = MLXArray.zeros([isVision.dim(0), 1], dtype: .bool)
    let prev = concatenated([leading, isVision[0..., ..<(length - 1)]], axis: 1)
    let starts = logicalAnd(isVision, logicalNot(prev))
    let groupIds = cumsum(starts.asType(.int32), axis: 1) - 1
    return MLX.where(isVision, groupIds, MLXArray(Int32(-1)))
}

/// Overlay blockwise bidirectional attention for vision-token spans onto a
/// boolean causal mask (True = attend). Tokens in the same image/video block
/// attend to each other in both directions. Mirrors Python
/// `_apply_blockwise_bidirectional_overlay`.
private func gemma4BidirectionalVisionMask(_ baseMask: MLXArray, isVision: MLXArray) -> MLXArray {
    let blockIds = gemma4VisionBlockIds(isVision)
    let qBlocks = expandedDimensions(blockIds, axis: -1)  // [B, L, 1]
    let kBlocks = expandedDimensions(blockIds, axis: -2)  // [B, 1, L]
    let sameBlock = logicalAnd(qBlocks .!= MLXArray(Int32(-1)), qBlocks .== kBlocks)  // [B, L, L]
    return logicalOr(baseMask, expandedDimensions(sameBlock, axis: 1))  // -> [B, 1, L, L]
}

/// If `mode` carries a boolean array mask, overlay the vision bidirectional
/// attention; pass other modes (`.causal`, `.none`) through unchanged.
private func gemma4OverlayBidirectionalVision(
    _ mode: MLXFast.ScaledDotProductAttentionMaskMode, isVision: MLXArray
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mode {
    case .array(let maskArray):
        return .array(gemma4BidirectionalVisionMask(maskArray, isVision: isVision))
    default:
        return mode
    }
}

/// Symmetrize a boolean causal / windowed-causal mask (True = attend) so every
/// token attends bidirectionally — fully for global layers, within its sliding
/// window for sliding layers. Mirrors Gemma4's `use_bidirectional_attention ==
/// "all"` (HF sets `is_causal = use_bidirectional_attention != "all"`): ORing the
/// causal mask with its transpose yields the symmetric version of whatever
/// lower-triangular pattern `createAttentionMask` produced. Non-array modes (the
/// single-token decode hot path stays on symbolic `.causal`) pass through.
private func gemma4SymmetrizeMask(
    _ mode: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mode {
    case .array(let maskArray):
        return .array(logicalOr(maskArray, maskArray.swappedAxes(-1, -2)))
    default:
        return mode
    }
}

// Vision Attention
private class VisionAttn: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeBase: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: VisionRMSNorm

    init(_ cfg: Gemma4VisionConfig) {
        numHeads = cfg.numAttentionHeads
        numKVHeads = cfg.numKeyValueHeads
        headDim = cfg.headDim
        ropeBase = cfg.ropeTheta
        _qProj.wrappedValue = Linear(cfg.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(cfg.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(cfg.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, cfg.hiddenSize, bias: false)
        _qNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        _kNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim)
        q = qNorm(q); k = kNorm(k); v = visionRmsNormNoScale(v)
        q = applyMultidimensionalRope(q, positions: positions, base: ropeBase)
        k = applyMultidimensionalRope(k, positions: positions, base: ropeBase)
        q = q.transposed(0, 2, 1, 3); k = k.transposed(0, 2, 1, 3); v = v.transposed(0, 2, 1, 3)
        // vmlx #52: Gemma 4 vision tower weights are float16 and attention
        // scores can exceed ±65504, producing -inf → NaN propagation through
        // embed_vision → model emits only <pad> tokens. Promote Q/K/V to
        // float32 for the SDPA, then cast back. Mirrors the Python
        // v1.3.29 patch.
        let origDType = q.dtype
        if origDType == .float16 {
            q = q.asType(.float32)
            k = k.asType(.float32)
            v = v.asType(.float32)
        }
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0,
            mask: mask != nil ? .array(mask!) : .none)
        if origDType == .float16 {
            out = out.asType(.float16)
        }
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

private class VisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    init(_ cfg: Gemma4VisionConfig) {
        _gateProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(safeGeluApproximate(gateProj(x)) * upProj(x)) }
}

private class VisionBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VisionAttn
    @ModuleInfo var mlp: VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLN: VisionRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnLN: VisionRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFFLN: VisionRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFFLN: VisionRMSNorm

    init(_ cfg: Gemma4VisionConfig) {
        _selfAttn.wrappedValue = VisionAttn(cfg)
        self.mlp = VisionMLP(cfg)
        _inputLN.wrappedValue = VisionRMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _postAttnLN.wrappedValue = VisionRMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _preFFLN.wrappedValue = VisionRMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _postFFLN.wrappedValue = VisionRMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + postAttnLN(selfAttn(inputLN(x), positions: positions, mask: mask))
        h = h + postFFLN(mlp(preFFLN(h)))
        return h
    }
}

private class VisionPatchEmbedder: Module {
    let patchSize: Int
    let posEmbSize: Int
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "position_embedding_table") var posTable: MLXArray

    init(_ cfg: Gemma4VisionConfig) {
        patchSize = cfg.patchSize
        posEmbSize = cfg.positionEmbeddingSize
        _inputProj.wrappedValue = Linear(3 * cfg.patchSize * cfg.patchSize, cfg.hiddenSize, bias: false)
        _posTable.wrappedValue = MLXArray.ones([2, cfg.positionEmbeddingSize, cfg.hiddenSize])
        super.init()
    }

    func callAsFunction(pixels: MLXArray, patchPos: MLXArray, padPos: MLXArray) -> MLXArray {
        let (B, C, H, W) = (pixels.dim(0), pixels.dim(1), pixels.dim(2), pixels.dim(3))
        let p = patchSize
        let patches = pixels.reshaped(B, C, H / p, p, W / p, p)
            .transposed(0, 2, 4, 3, 5, 1).reshaped(B, (H / p) * (W / p), C * p * p)
        let normalized = 2 * (patches - 0.5)
        let embedded = inputProj(normalized.asType(inputProj.weight.dtype))

        let oh = oneHot(patchPos, numClasses: posEmbSize)
            .transposed(0, 2, 1, 3).asType(posTable.dtype)
        var posEmb = matmul(oh, posTable).sum(axis: 1)
        posEmb = MLX.where(expandedDimensions(padPos, axis: -1), MLXArray(Float(0), dtype: posEmb.dtype), posEmb)
        return embedded + posEmb
    }
}

private class VisionPooler: Module {
    let defaultLen: Int
    let rootH: Float
    init(_ cfg: Gemma4VisionConfig) {
        defaultLen = cfg.defaultOutputLength
        rootH = sqrt(Float(cfg.hiddenSize))
        super.init()
    }
    /// Pool `h` down to `outputLen` soft tokens. Images use the default (image)
    /// budget; video frames pass a smaller per-frame budget (e.g. 70 vs 280) so a
    /// frame resized to the video patch budget pools to its own token count.
    func callAsFunction(
        _ h: MLXArray, patchPos: MLXArray, padPos: MLXArray, outputLen: Int? = nil
    ) -> (MLXArray, MLXArray) {
        let outLen = outputLen ?? defaultLen
        let L = h.dim(1)
        if L == outLen { return (h * rootH, logicalNot(padPos)) }
        let k = Int(sqrt(Float(L / outLen)))
        let kSq = Float(k * k)
        let clamped = maximum(patchPos, MLXArray(Int32(0)))
        let maxX = clamped[.ellipsis, 0].max(axis: -1, keepDims: true) + 1
        let ki = floor(clamped.asType(.float32) / Float(k)).asType(.int32)
        let linearIdx = ki[.ellipsis, 0] + (maxX / MLXArray(Int32(k))) * ki[.ellipsis, 1]
        let w = oneHot(linearIdx, numClasses: outLen) / kSq
        let out = matmul(w.transposed(0, 2, 1), h)
        let mask = logicalNot(all(w .== Float(0), axis: 1))
        return (out.asType(h.dtype) * rootH, mask)
    }
}

private class VisionEncoder: Module {
    @ModuleInfo var layers: [VisionBlock]
    init(_ cfg: Gemma4VisionConfig) {
        _layers.wrappedValue = (0 ..< cfg.numHiddenLayers).map { _ in VisionBlock(cfg) }
        super.init()
    }
    func callAsFunction(_ x: MLXArray, pos: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x; for l in layers { h = l(h, positions: pos, mask: mask) }; return h
    }
}

private class VisionTower: Module {
    let cfg: Gemma4VisionConfig
    @ModuleInfo(key: "patch_embedder") var patchEmb: VisionPatchEmbedder
    @ModuleInfo var encoder: VisionEncoder
    @ModuleInfo var pooler: VisionPooler
    @ModuleInfo(key: "std_bias") var stdBias: MLXArray?
    @ModuleInfo(key: "std_scale") var stdScale: MLXArray?

    init(_ cfg: Gemma4VisionConfig) {
        self.cfg = cfg
        _patchEmb.wrappedValue = VisionPatchEmbedder(cfg)
        self.encoder = VisionEncoder(cfg)
        self.pooler = VisionPooler(cfg)
        if cfg.standardize { _stdBias.wrappedValue = MLXArray.zeros([cfg.hiddenSize]); _stdScale.wrappedValue = MLXArray.ones([cfg.hiddenSize]) }
        super.init()
    }

    /// Encode one image / video frame to `outputLength` soft tokens. `outputLength`
    /// defaults to the image budget (`defaultOutputLength`); video frames pass a
    /// smaller per-frame budget so the local patch budget (`outputLength * pool^2`)
    /// and the pooler output both shrink to the trained video-frame representation.
    func callAsFunction(_ pixels: MLXArray, outputLength: Int? = nil) -> MLXArray {
        let (B, _, H, W) = (pixels.dim(0), pixels.dim(1), pixels.dim(2), pixels.dim(3))
        let outLen = outputLength ?? cfg.defaultOutputLength
        let localMaxPatches = max(1, outLen) * cfg.poolingKernelSize * cfg.poolingKernelSize
        let p = cfg.patchSize
        var pH = H / p
        let pW = W / p

        // Truncate oversized grids CONSISTENTLY: drop whole trailing rows so the
        // patch embedding (built from the same cropped pixels), the position grid,
        // and the pooler all agree on the patch count. Previously `nReal` was
        // clamped to `maxPatches` but `posFlat` was still built for every `pH * pW`
        // patch, so the `reshaped(1, nReal, 2)` below trapped before the clamp could
        // help — and the pixels stayed untruncated, mismatching the clamped
        // positions. Cropping rows keeps `pW` (and the per-row x positions) intact.
        var croppedPixels = pixels
        if pH * pW > localMaxPatches {
            pH = max(1, localMaxPatches / max(1, pW))
            croppedPixels = pixels[0..., 0..., ..<(pH * p), 0...]
        }
        let nReal = pH * pW
        let nPad = localMaxPatches - nReal

        // Build position grid [nReal, 2] then expand to [B, nReal, 2]
        var posFlat = [Int32]()
        posFlat.reserveCapacity(nReal * 2)
        for y in 0 ..< pH { for x in 0 ..< pW { posFlat.append(Int32(x)); posFlat.append(Int32(y)) } }
        var patchPos = MLXArray(posFlat).reshaped(1, nReal, 2)
        patchPos = repeated(patchPos, count: B, axis: 0)
        var padPos = MLXArray.zeros([B, localMaxPatches]).asType(.bool)

        if nPad > 0 {
            let padFlat = [Int32](repeating: -1, count: nPad * 2)
            let pp = MLXArray(padFlat).reshaped(1, nPad, 2)
            patchPos = concatenated([patchPos, repeated(pp, count: B, axis: 0)], axis: 1)
            padPos = concatenated([MLXArray.zeros([B, nReal]).asType(.bool), MLXArray.ones([B, nPad]).asType(.bool)], axis: 1)
        }

        var emb = patchEmb(pixels: croppedPixels, patchPos: patchPos[0..., ..<nReal], padPos: padPos[0..., ..<nReal])
        if nPad > 0 { emb = concatenated([emb, MLXArray.zeros([B, nPad, cfg.hiddenSize]).asType(emb.dtype)], axis: 1) }

        let valid = logicalNot(padPos).asType(.float32)
        var mask = expandedDimensions(valid, axis: 1) * expandedDimensions(valid, axis: 2)
        let zeroVal = MLXArray(Float(0), dtype: emb.dtype)
        let negInfVal = MLXArray(Float(-1e9), dtype: emb.dtype)
        mask = MLX.where(mask .> MLXArray(Float(0), dtype: mask.dtype), zeroVal, negInfVal)
        mask = expandedDimensions(mask, axis: 1)

        var h = encoder(emb, pos: patchPos, mask: mask)
        let (pooled, _) = pooler(h, patchPos: patchPos, padPos: padPos, outputLen: outLen)
        // Return all `outLen` features — the processor inserts exactly that many
        // image/video soft tokens, so maskedScatter needs them all to match.
        h = pooled
        if cfg.standardize, let sb = stdBias, let ss = stdScale { h = (h - sb) * ss }
        return h
    }
}

// MARK: - ScaledLinear (for per-layer model projection)

private class G4ScaledLinear: Module {
    let weight: MLXArray; let scalar: Float
    init(inputDims: Int, outputDims: Int, scalar: Float) {
        self.weight = MLXArray.zeros([outputDims, inputDims]); self.scalar = scalar; super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { matmul(x, weight.T) * scalar }
}

// MARK: - Text Model Components (inline for VLM — MLXVLM can't import MLXLLM)

// Text Attention, MLP, Router, Experts, DecoderLayer, Model — same as Gemma4Text.swift
// but scoped privately within this file.

private class TextAttn: Module {
    let nH: Int; let nKV: Int; let hD: Int; let scale: Float; let isSliding: Bool; let useKEqV: Bool; let eps: Float
    @ModuleInfo(key: "q_proj") var qP: Linear
    @ModuleInfo(key: "k_proj") var kP: Linear
    @ModuleInfo(key: "v_proj") var vP: Linear?
    @ModuleInfo(key: "o_proj") var oP: Linear
    @ModuleInfo(key: "q_norm") var qN: G4RMSNorm
    @ModuleInfo(key: "k_norm") var kN: G4RMSNorm
    @ModuleInfo var rope: RoPELayer

    init(_ cfg: G4TextConfig, layerIndex: Int) {
        let lt = layerIndex < cfg.layerTypes.count ? cfg.layerTypes[layerIndex] : "sliding_attention"
        isSliding = lt == "sliding_attention"
        useKEqV = cfg.attentionKEqV && !isSliding
        eps = cfg.rmsNormEps
        if isSliding { nH = cfg.numAttentionHeads; nKV = cfg.numKeyValueHeads; hD = cfg.headDim }
        else { nH = cfg.numAttentionHeads; nKV = cfg.numGlobalKeyValueHeads ?? cfg.numKeyValueHeads; hD = cfg.globalHeadDim }
        scale = 1.0
        _qP.wrappedValue = Linear(cfg.hiddenSize, nH * hD, bias: cfg.attentionBias)
        _kP.wrappedValue = Linear(cfg.hiddenSize, nKV * hD, bias: cfg.attentionBias)
        if !useKEqV { _vP.wrappedValue = Linear(cfg.hiddenSize, nKV * hD, bias: cfg.attentionBias) }
        _oP.wrappedValue = Linear(nH * hD, cfg.hiddenSize, bias: cfg.attentionBias)
        _qN.wrappedValue = G4RMSNorm(dimensions: hD, eps: cfg.rmsNormEps)
        _kN.wrappedValue = G4RMSNorm(dimensions: hD, eps: cfg.rmsNormEps)
        let lk = isSliding ? "sliding_attention" : "full_attention"
        let rp = cfg.ropeParameters[lk] ?? [:]
        let rt = rp["rope_theta"]?.asFloat() ?? (isSliding ? 10000.0 : 1_000_000.0)
        let prf = rp["partial_rotary_factor"]?.asFloat() ?? (isSliding ? 1.0 : 0.25)
        self.rope = initializeRope(dims: max(1, Int(Float(hD) * prf)), base: rt, traditional: cfg.ropeTraditional, scalingConfig: nil, maxPositionEmbeddings: nil)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        sharedKV: (keys: MLXArray, values: MLXArray)? = nil, sharedOffset: Int? = nil,
        sharedOffsetArray: MLXArray? = nil
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray, offset: Int) {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qP(x).reshaped(B, L, nH, hD); q = qN(q); q = q.transposed(0, 2, 1, 3)
        let cK: MLXArray; let cV: MLXArray; let off: Int
        if let sharedKV {
            off = sharedOffset ?? 0
            if let sharedOffsetArray { q = rope(q, offset: sharedOffsetArray) }
            else { q = rope(q, offset: off) }
            cK = sharedKV.keys; cV = sharedKV.values
        } else {
            off = cache?.offset ?? 0
            var k = kP(x).reshaped(B, L, nKV, hD)
            let v: MLXArray
            if useKEqV { v = rmsNormNoScale(k, eps: eps) } else if let vP { v = rmsNormNoScale(vP(x).reshaped(B, L, nKV, hD), eps: eps) } else { v = rmsNormNoScale(k, eps: eps) }
            k = kN(k)
            let vT = v.transposed(0, 2, 1, 3); var kT = k.transposed(0, 2, 1, 3)
            kT = applyRotaryPosition(rope, to: kT, cache: cache)
            q = applyRotaryPosition(rope, to: q, cache: cache)
            if let cache { (cK, cV) = cache.update(keys: kT, values: vT) } else { (cK, cV) = (kT, vT) }
        }
        // vmlx #52 text-path: Gemma 4 text attention scores can exceed
        // fp16 max (±65504) on long contexts, especially in combination
        // with the final-logit softcap amplifying tails. Mirror the
        // vision-tower fp32 upcast when the activation dtype is fp16.
        // Critical for sliding-window layers since the windowed key set
        // concentrates softmax mass on fewer entries.
        let origDType = q.dtype
        var qF = q, kF = cK, vF = cV
        if origDType == .float16 {
            qF = qF.asType(.float32)
            kF = kF.asType(.float32)
            vF = vF.asType(.float32)
        }
        var sdpa = MLXFast.scaledDotProductAttention(queries: qF, keys: kF, values: vF, scale: scale, mask: mask)
        if origDType == .float16 { sdpa = sdpa.asType(.float16) }
        let out = sdpa.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return (oP(out), cK, cV, off)
    }
}

private class TextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gP: Linear; @ModuleInfo(key: "up_proj") var uP: Linear; @ModuleInfo(key: "down_proj") var dP: Linear
    init(_ cfg: G4TextConfig, intermediateSize: Int? = nil) {
        let iS = intermediateSize ?? cfg.intermediateSize
        _gP.wrappedValue = Linear(cfg.hiddenSize, iS, bias: false); _uP.wrappedValue = Linear(cfg.hiddenSize, iS, bias: false); _dP.wrappedValue = Linear(iS, cfg.hiddenSize, bias: false); super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = safeGeluApproximate(gP(x))
        let u = uP(x)
        let product: MLXArray
        product = g * u
        return dP(product)
    }
}

private class TextRouter: Module {
    @ModuleInfo(key: "proj") var proj: Linear; @ModuleInfo(key: "scale") var sc: MLXArray; @ModuleInfo(key: "per_expert_scale") var pes: MLXArray
    let nE: Int; let topK: Int; let rs: Float; let eps: Float
    init(_ cfg: G4TextConfig) {
        nE = cfg.numExperts; topK = cfg.topKExperts; rs = pow(Float(cfg.hiddenSize), -0.5); eps = cfg.rmsNormEps
        _proj.wrappedValue = Linear(cfg.hiddenSize, cfg.numExperts, bias: false)
        _sc.wrappedValue = MLXArray.ones([cfg.hiddenSize]); _pes.wrappedValue = MLXArray.ones([cfg.numExperts])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var h = rmsNormNoScale(x, eps: eps) * rs * sc
        let s = proj(h); let p = softmax(s, axis: -1, precise: true)
        let ti = argPartition(-s, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var tw = takeAlong(p, ti, axis: -1); tw = tw / tw.sum(axis: -1, keepDims: true); tw = tw * pes[ti]
        return (ti, tw)
    }
}

private class TextExperts: Module {
    @ModuleInfo(key: "switch_glu") var sg: SwitchGLU
    init(_ cfg: G4TextConfig) {
        _sg.wrappedValue = SwitchGLU(inputDims: cfg.hiddenSize, hiddenDims: cfg.moeIntermediateSize, numExperts: cfg.numExperts, activation: { safeGeluApproximate($0) }, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, idx: MLXArray, wts: MLXArray) -> MLXArray {
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2)); let K = idx.dim(-1)
        let o = sg(x.reshaped(B * S, H), idx.reshaped(B * S, K))
        return (o * expandedDimensions(wts.reshaped(B * S, K), axis: -1)).sum(axis: -2).reshaped(B, S, H)
    }
}

private class TextLayer: Module {
    let hasMoE: Bool
    @ModuleInfo(key: "self_attn") var attn: TextAttn; @ModuleInfo var mlp: TextMLP
    @ModuleInfo var router: TextRouter?; @ModuleInfo var experts: TextExperts?
    @ModuleInfo(key: "input_layernorm") var iLN: G4RMSNorm; @ModuleInfo(key: "post_attention_layernorm") var paLN: G4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var pfLN: G4RMSNorm; @ModuleInfo(key: "post_feedforward_layernorm") var pffLN: G4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var pfLN2: G4RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var pffLN1: G4RMSNorm?; @ModuleInfo(key: "post_feedforward_layernorm_2") var pffLN2: G4RMSNorm?
    @ModuleInfo(key: "per_layer_input_gate") var pliGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var pliProj: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var pliNorm: G4RMSNorm?
    @ModuleInfo(key: "layer_scalar") var ls: MLXArray

    init(_ cfg: G4TextConfig, i: Int) {
        hasMoE = cfg.enableMoeBlock && cfg.numExperts > 0
        _attn.wrappedValue = TextAttn(cfg, layerIndex: i)
        let fks = cfg.numHiddenLayers - cfg.numKvSharedLayers
        let isShared = cfg.numKvSharedLayers > 0 && i >= fks
        let iSize = (cfg.useDoubleWideMlp && isShared) ? cfg.intermediateSize * 2 : cfg.intermediateSize
        self.mlp = TextMLP(cfg, intermediateSize: iSize)
        if hasMoE {
            self.router = TextRouter(cfg); self.experts = TextExperts(cfg)
            _pfLN2.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
            _pffLN1.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
            _pffLN2.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        }
        if cfg.hiddenSizePerLayerInput > 0 {
            _pliGate.wrappedValue = Linear(cfg.hiddenSize, cfg.hiddenSizePerLayerInput, bias: false)
            _pliProj.wrappedValue = Linear(cfg.hiddenSizePerLayerInput, cfg.hiddenSize, bias: false)
            _pliNorm.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        }
        _iLN.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _paLN.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _pfLN.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _pffLN.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _ls.wrappedValue = MLXArray([Float(1.0)])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        perLayerInput: MLXArray? = nil,
        sharedKV: (keys: MLXArray, values: MLXArray)? = nil, sharedOffset: Int? = nil,
        sharedOffsetArray: MLXArray? = nil
    ) -> (h: MLXArray, keys: MLXArray, values: MLXArray, offset: Int) {
        var r = x
        let (aOut, aK, aV, aOff) = attn(iLN(x), mask: mask, cache: cache, sharedKV: sharedKV, sharedOffset: sharedOffset, sharedOffsetArray: sharedOffsetArray)
        var h = paLN(aOut); h = r + h; r = h
        if hasMoE, let router, let experts, let pfLN2, let pffLN1, let pffLN2 {
            var h1 = mlp(pfLN(h)); h1 = pffLN1(h1)
            let (ti, tw) = router(h); var h2 = experts(pfLN2(h), idx: ti, wts: tw); h2 = pffLN2(h2)
            h = h1 + h2
        } else { h = mlp(pfLN(h)) }
        h = pffLN(h); h = r + h
        if let pliGate, let pliProj, let pliNorm, let perLayerInput {
            r = h; var g = safeGeluApproximate(pliGate(h)); g = g * perLayerInput
            g = pliProj(g); g = pliNorm(g); h = r + g
        }
        h = h * ls
        return (h, aK, aV, aOff)
    }
}

private class TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var emb: Embedding; @ModuleInfo var layers: [TextLayer]; @ModuleInfo var norm: G4RMSNorm
    @ModuleInfo(key: "embed_tokens_per_layer") var embPL: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var plProj: G4ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var plNorm: G4RMSNorm?
    let cfg: G4TextConfig
    let previousKVs: [Int]

    init(_ cfg: G4TextConfig) {
        self.cfg = cfg; _emb.wrappedValue = Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize)
        _layers.wrappedValue = (0 ..< cfg.numHiddenLayers).map { TextLayer(cfg, i: $0) }
        self.norm = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        if cfg.hiddenSizePerLayerInput > 0 {
            _embPL.wrappedValue = Embedding(embeddingCount: cfg.vocabSizePerLayerInput,
                dimensions: cfg.numHiddenLayers * cfg.hiddenSizePerLayerInput)
            _plProj.wrappedValue = G4ScaledLinear(inputDims: cfg.hiddenSize,
                outputDims: cfg.numHiddenLayers * cfg.hiddenSizePerLayerInput,
                scalar: pow(Float(cfg.hiddenSize), -0.5))
            _plNorm.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSizePerLayerInput, eps: cfg.rmsNormEps)
        }
        let lt = cfg.layerTypes.isEmpty ? Array(repeating: "sliding_attention", count: cfg.numHiddenLayers) : cfg.layerTypes
        var pkvs = Array(0 ..< cfg.numHiddenLayers)
        if cfg.numKvSharedLayers > 0 {
            let fks = cfg.numHiddenLayers - cfg.numKvSharedLayers
            var byType: [String: Int] = [:]; for i in 0 ..< fks { byType[lt[i]] = i }
            for j in fks ..< cfg.numHiddenLayers { if let s = byType[lt[j]] { pkvs[j] = s } }
        }
        self.previousKVs = pkvs
        super.init()
    }

    private func getPerLayerInputs(_ ids: MLXArray) -> MLXArray? {
        guard let embPL else { return nil }
        let r = embPL(ids) * pow(Float(cfg.hiddenSizePerLayerInput), 0.5)
        return r.reshaped(Array(ids.shape) + [cfg.numHiddenLayers, cfg.hiddenSizePerLayerInput])
    }

    private func projectPerLayerInputs(_ h: MLXArray, pli: MLXArray?) -> MLXArray? {
        guard let plProj, let plNorm else { return nil }
        var p = plProj(h).reshaped(Array(h.shape.dropLast()) + [cfg.numHiddenLayers, cfg.hiddenSizePerLayerInput])
        p = plNorm(p)
        guard let pli else { return p }
        return (p + pli) * pow(Float(2.0), Float(-0.5))
    }

    func callAsFunction(_ inputs: MLXArray?, inputEmbedding: MLXArray? = nil, cache: [KVCache?]? = nil, imageTokenMask: MLXArray? = nil) -> MLXArray {
        // Ensure batch dimension — callers may pass 1D tokens [N] on cache-reuse turns
        let inputs = inputs.map { $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0 }
        var h: MLXArray
        if let ie = inputEmbedding {
            h = ie.ndim == 2 ? ie.expandedDimensions(axis: 0) : ie
        } else {
            h = emb(inputs!) * MLXArray(sqrt(Float(cfg.hiddenSize)), dtype: emb.weight.dtype)
        }

        var pliList: [MLXArray?]
        if cfg.hiddenSizePerLayerInput > 0 {
            let raw = inputs.flatMap { getPerLayerInputs($0) }
            if let final = projectPerLayerInputs(h, pli: raw) {
                pliList = (0 ..< layers.count).map { final[0..., 0..., $0, 0...] }
            } else { pliList = Array(repeating: nil, count: layers.count) }
        } else { pliList = Array(repeating: nil, count: layers.count) }

        let lc = cache ?? Array(repeating: nil as KVCache?, count: layers.count)
        let lt = cfg.layerTypes.isEmpty ? Array(repeating: "sliding_attention", count: cfg.numHiddenLayers) : cfg.layerTypes
        let gIdx = lt.firstIndex(of: "full_attention") ?? (cfg.numHiddenLayers - 1)
        let sIdx = lt.firstIndex(of: "sliding_attention") ?? 0
        let gc: KVCache? = cache.flatMap { gIdx < $0.count ? $0[gIdx] : nil }
        let sc: KVCache? = cache.flatMap { sIdx < $0.count ? $0[sIdx] : nil }
        // Gemma4 applies blockwise bidirectional attention within image/video
        // soft-token spans during prefill. The overlay needs a materialized
        // boolean mask, so force `returnArray` only when it is active. The
        // text-only / single-token decode hot path keeps `imageTokenMask == nil`,
        // so it stays on the symbolic `.causal` mask (no array materialized).
        let useBidirectionalVision =
            imageTokenMask != nil && cfg.useBidirectionalAttention == "vision" && h.dim(1) > 1
        // `use_bidirectional_attention == "all"` makes the WHOLE prefill non-causal
        // (HF: `is_causal = use_bidirectional_attention != "all"`), independent of any
        // vision span. Only relevant during multi-token prefill — single-token decode
        // (`h.dim(1) == 1`) attends to all past keys either way, so the fast text path
        // stays on the symbolic causal mask.
        let useBidirectionalAll = cfg.useBidirectionalAttention == "all" && h.dim(1) > 1
        let forceArrayMask = useBidirectionalVision || useBidirectionalAll
        var gm = createAttentionMask(h: h, cache: gc, returnArray: forceArrayMask)
        var sm = createAttentionMask(h: h, cache: sc, windowSize: cfg.slidingWindow, returnArray: forceArrayMask)
        if useBidirectionalVision, let imageTokenMask {
            gm = gemma4OverlayBidirectionalVision(gm, isVision: imageTokenMask)
            sm = gemma4OverlayBidirectionalVision(sm, isVision: imageTokenMask)
        } else if useBidirectionalAll {
            gm = gemma4SymmetrizeMask(gm)
            sm = gemma4SymmetrizeMask(sm)
        }

        var intermediates: [(keys: MLXArray, values: MLXArray, offset: Int, offsetArray: MLXArray?)?] = Array(repeating: nil, count: layers.count)
        for (i, l) in layers.enumerated() {
            let isGlobal = (i < lt.count ? lt[i] : "sliding_attention") == "full_attention"
            let prevIdx = previousKVs[i]
            let skv: (keys: MLXArray, values: MLXArray)?; let soff: Int?; let soffArr: MLXArray?
            if prevIdx != i, let prev = intermediates[prevIdx] { skv = (prev.keys, prev.values); soff = prev.offset; soffArr = prev.offsetArray }
            else { skv = nil; soff = nil; soffArr = nil }
            let ce = prevIdx == i ? (i < lc.count ? lc[i] : nil) : nil
            // Capture the per-row / graph offset BEFORE the source layer's
            // attention advances the cache, so the snapshot matches the
            // pre-update scalar `res.offset` that KV-sharing consumer layers
            // also use to RoPE their queries (mirrors `gemma4CapturePositionOffset`
            // in Libraries/MLXLLM/Models/Gemma4Text.swift). Covers BatchKVCache /
            // BatchRotatingKVCache / QuantizedBatchKVCache (per-row `batchOffset`)
            // and CompilableKVCache (graph-traceable `offsetArray`); nil for the
            // scalar B=1 / non-compiled decode hot path.
            let layerOffArr = gemma4VLMGraphOffsetArray(for: ce)
            let res = l(h, mask: isGlobal ? gm : sm, cache: ce, perLayerInput: pliList[i], sharedKV: skv, sharedOffset: soff, sharedOffsetArray: soffArr)
            h = res.h; intermediates[i] = (res.keys, res.values, res.offset, layerOffArr)
        }
        return norm(h)
    }
}

private class G4LanguageModel: Module {
    @ModuleInfo var model: TextModel; @ModuleInfo(key: "lm_head") var lmHead: Linear?
    let cfg: G4TextConfig
    init(_ cfg: G4TextConfig) {
        self.cfg = cfg; self.model = TextModel(cfg)
        if !cfg.tieWordEmbeddings { _lmHead.wrappedValue = Linear(cfg.hiddenSize, cfg.vocabSize, bias: false) }
        super.init()
    }
    func callAsFunction(_ inputs: MLXArray?, inputEmbedding: MLXArray? = nil, cache: [KVCache?]? = nil, imageTokenMask: MLXArray? = nil) -> MLXArray {
        var o = model(inputs, inputEmbedding: inputEmbedding, cache: cache, imageTokenMask: imageTokenMask)
        if let lh = lmHead { o = lh(o) } else { o = model.emb.asLinear(o) }
        if let cap = cfg.finalLogitSoftcapping, cap > 0 { o = compiledLogitSoftcap(o, MLXArray(cap)) }
        return o
    }
    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let fks = cfg.numKvSharedLayers > 0 ? cfg.numHiddenLayers - cfg.numKvSharedLayers : cfg.numHiddenLayers
        return (0 ..< fks).map { i in
            let lt = i < cfg.layerTypes.count ? cfg.layerTypes[i] : "sliding_attention"
            if lt == "full_attention" { return parameters?.maxKVSize.map { RotatingKVCache(maxSize: $0, keep: 4) } ?? KVCacheSimple() }
            else { return RotatingKVCache(maxSize: cfg.slidingWindow, keep: 0) }
        }
    }
}

// MARK: - Multimodal Embedder

private class MultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var proj: Linear
    init(embDim: Int, textDim: Int) { _proj.wrappedValue = Linear(embDim, textDim, bias: false); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { rmsNormNoScale(proj(x)) }
}

private func maskedScatter(input: MLXArray, mask: MLXArray, source: MLXArray) throws -> MLXArray {
    let inputShape = input.shape
    let inputFlat = input.flattened()
    let maskFlat = mask.flattened()
    let sourceFlat = source.flattened()

    let maskValues = maskFlat.asArray(Bool.self)
    let positions = maskValues.enumerated().compactMap { i, v in v ? UInt32(i) : nil }

    guard !positions.isEmpty else { return input }

    let posArray = MLXArray(positions)
    // Surface the bundle/processor-config mismatch as a recoverable
    // VLMError instead of an abort. Per `docs/GEMMA4-DEEP-TRACE-2026-05-10.md`
    // §7.3, a `fatalError` here was never reachable cleanly — a
    // mis-stamped `imageSeqLength` would crash the whole process on
    // first image. Throw so the caller (osaurus, JANG Studio, etc.)
    // can surface the diagnostic without process abort.
    guard sourceFlat.shape[0] == posArray.shape[0] else {
        throw VLMError.processing(
            """
            Gemma4 maskedScatter: size mismatch between vision features and image token positions. \
            Vision features: \(sourceFlat.shape[0]), image positions: \(posArray.shape[0]). \
            Check that imageSeqLength in preprocessor_config matches vision tower output (defaultOutputLength).
            """)
    }
    inputFlat[posArray] = sourceFlat
    return inputFlat.reshaped(inputShape)
}

// MARK: - Gemma4 VLM

public class Gemma4: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: VisionTower
    @ModuleInfo(key: "language_model") private var languageModel: G4LanguageModel
    @ModuleInfo(key: "embed_vision") private var embedVision: MultimodalEmbedder

    public let config: Gemma4Configuration
    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var kvHeads: [Int] {
        let tc = config.textConfig
        return (0 ..< tc.numHiddenLayers).map { i in
            let lt = i < tc.layerTypes.count ? tc.layerTypes[i] : "sliding_attention"
            return lt == "full_attention" ? (tc.numGlobalKeyValueHeads ?? tc.numKeyValueHeads) : tc.numKeyValueHeads
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] { languageModel.newCache(parameters: parameters) }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        _visionTower.wrappedValue = VisionTower(config.visionConfig)
        _languageModel.wrappedValue = G4LanguageModel(config.textConfig)
        _embedVision.wrappedValue = MultimodalEmbedder(embDim: config.visionConfig.hiddenSize, textDim: config.textConfig.hiddenSize)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        // Gemma4 VLM does not implement audio. Our `LMInput` carries no audio
        // field, and the `sanitize` path drops `audio_tower.*` / `embed_audio.*`
        // weights, so there is nothing to guard here.
        var emb = languageModel.model.emb(input.text.tokens)
        emb = emb * MLXArray(sqrt(Float(config.textConfig.hiddenSize)), dtype: emb.dtype)

        // Accumulate the image+video soft-token positions so the text tower can
        // apply Gemma4's blockwise bidirectional attention over those spans. nil
        // keeps the text-only hot path on the symbolic causal mask.
        var visualTokenMask: MLXArray? = nil

        if let pixels = input.image?.pixels {
            let imgFeatures = encodeVisionFeatures(
                pixels: pixels, frames: input.image?.frames, dtype: emb.dtype)
            let imgMask = MLX.equal(input.text.tokens, MLXArray(Int32(config.imageTokenId)))
            let imgMaskExp = MLX.broadcast(expandedDimensions(imgMask, axis: -1), to: emb.shape)
            emb = try maskedScatter(input: emb, mask: imgMaskExp, source: imgFeatures)
            visualTokenMask = imgMask
        }

        // Video frames run through the same vision tower as images; only the
        // placeholder token they scatter into differs (image_token vs video_token).
        if let videoPixels = input.video?.pixels, let videoTokenId = config.videoTokenId {
            let vidFeatures = encodeVisionFeatures(
                pixels: videoPixels, frames: input.video?.frames, dtype: emb.dtype, isVideo: true)
            let vidMask = MLX.equal(input.text.tokens, MLXArray(Int32(videoTokenId)))
            let vidMaskExp = MLX.broadcast(expandedDimensions(vidMask, axis: -1), to: emb.shape)
            emb = try maskedScatter(input: emb, mask: vidMaskExp, source: vidFeatures)
            visualTokenMask = visualTokenMask.map { logicalOr($0, vidMask) } ?? vidMask
        }

        let paddedCache = padCache(cache)
        let out = languageModel(
            input.text.tokens, inputEmbedding: emb, cache: paddedCache,
            imageTokenMask: visualTokenMask)
        return .logits(.init(logits: out))
    }

    /// Encode one batch of images / video frames through the vision tower and
    /// project into the text embedding space. Each frame is sliced to its real
    /// (un-padded) dimensions stored in `frames`. Images yield `defaultOutputLength`
    /// soft tokens; video frames (`isVideo`) yield a per-frame, data-driven count
    /// (`(h/patch)*(w/patch)/pool^2`) from their video-budget-resized grid, matching
    /// the processor's per-frame placeholder expansion (HF/mlx-vlm Gemma4 use a
    /// smaller per-frame video soft-token budget, ~70, vs 280 for images).
    private func encodeVisionFeatures(
        pixels: MLXArray, frames: [THW]?, dtype: DType, isVideo: Bool = false
    ) -> MLXArray {
        let featuresList = visionFeatureList(pixels: pixels, frames: frames, isVideo: isVideo)
        return (featuresList.count == 1 ? featuresList[0] : concatenated(featuresList))
            .asType(dtype)
    }

    /// Shared tower + projector loop: one `[1, softTokens, textHidden]` array
    /// per image / video frame, in the projector's native dtype (callers cast).
    /// Split out of `encodeVisionFeatures` so the external per-image seam
    /// (`perImageVisionFeatures`) reuses exactly the arrays `prepare` scatters.
    private func visionFeatureList(
        pixels: MLXArray, frames: [THW]?, isVideo: Bool
    ) -> [MLXArray] {
        let B = pixels.dim(0)
        var featuresList = [MLXArray]()
        featuresList.reserveCapacity(B)
        let p = config.visionConfig.patchSize
        let poolSq = config.visionConfig.poolingKernelSize * config.visionConfig.poolingKernelSize
        // All frames of one video share the same resized size, so this per-frame
        // count is uniform within a video and matches the processor's placeholder
        // expansion (both derive it from the same video-budget-resized grid).
        func videoOutputLength(h: Int, w: Int) -> Int {
            max(1, ((h / p) * (w / p)) / max(1, poolSq))
        }
        for i in 0 ..< B {
            // Extract each frame at its original dimensions (stored in frames)
            // to avoid processing zero-padded regions through the vision tower.
            if let frames, i < frames.count {
                let h = frames[i].h
                let w = frames[i].w
                let single = pixels[i, 0..., ..<h, ..<w].expandedDimensions(axis: 0)
                let outLen = isVideo ? videoOutputLength(h: h, w: w) : nil
                featuresList.append(embedVision(visionTower(single, outputLength: outLen)))
            } else {
                let single = pixels[i].expandedDimensions(axis: 0)
                let outLen = isVideo ? videoOutputLength(h: pixels.dim(2), w: pixels.dim(3)) : nil
                featuresList.append(embedVision(visionTower(single, outputLength: outLen)))
            }
        }
        return featuresList
    }

    private func padCache(_ cache: [any KVCache]?) -> [KVCache?]? {
        cache.map { c in
            c.map { $0 as KVCache? } + Array(repeating: nil as KVCache?,
                count: max(0, config.textConfig.numHiddenLayers - c.count))
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: padCache(cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var p = [String: MLXArray]()
        for (k, v) in weights {
            var nk = k
            if nk.hasPrefix("model.") { nk = String(nk.dropFirst("model.".count)) }
            // Skip audio — Gemma4 VLM doesn't implement audio, these weights have no module
            if nk.hasPrefix("audio_tower.") || nk.hasPrefix("embed_audio.") { continue }
            // Skip clipped linear params — training artifacts, not used in inference (we use plain Linear)
            if nk.contains("input_min") || nk.contains("input_max") || nk.contains("output_min") || nk.contains("output_max") { continue }
            if nk.contains("rotary_emb") { continue }
            // Remap language_model keys to include model. prefix — EXCEPT
            // `language_model.lm_head.*`. The untied LM head is the
            // `G4LanguageModel.lmHead` module (key "lm_head") that lives directly
            // under `language_model`, NOT under the `model` (TextModel) submodule.
            // Blanket-remapping it to `language_model.model.lm_head.*` points at a
            // module that does not exist, so the head silently fails to load for
            // untied checkpoints (tie_word_embeddings == false). The vocab-trim
            // loop below already expects it at `language_model.lm_head.*`.
            if nk.hasPrefix("language_model.") && !nk.hasPrefix("language_model.model.")
                && !nk.hasPrefix("language_model.lm_head")
            {
                nk = "language_model.model." + String(nk.dropFirst("language_model.".count))
            }
            if nk.contains(".switch_mlp.") { nk = nk.replacingOccurrences(of: ".switch_mlp.", with: ".experts.switch_glu.") }
            // Vision tower uses ClippableLinear wrappers — checkpoint has .linear. segment
            // that doesn't exist in our module tree (we use plain Linear)
            if nk.hasPrefix("vision_tower.") && nk.contains(".linear.") {
                nk = nk.replacingOccurrences(of: ".linear.", with: ".")
            }
            p[nk] = v
        }
        let ev = config.textConfig.vocabSize
        for k in ["language_model.model.embed_tokens.weight", "language_model.model.embed_tokens.scales", "language_model.model.embed_tokens.biases", "language_model.lm_head.weight", "language_model.lm_head.scales", "language_model.lm_head.biases"] {
            if let w = p[k], w.dim(0) != ev { p[k] = w[0 ..< ev] }
        }
        return p
    }
}

extension Gemma4: LoRAModel { public var loraLayers: [Module] { languageModel.model.layers } }

// MARK: - External vision-embedding seam (CBv2 multimodal prefill)

extension Gemma4 {
    /// The token id every image soft-token position carries in the tokenized
    /// prompt (`image_token_id`; the processor writes `imageSeqLength` of
    /// these per image between the ordinary `boi`/`eoi` delimiter tokens).
    /// External engines locate the per-image placeholder runs with it.
    public var imagePlaceholderTokenId: Int { config.imageTokenId }

    /// Per-image soft-token embeddings — the EXACT arrays `prepare` scatters
    /// over the image placeholder positions (vision tower + multimodal
    /// projector, `embedVision(visionTower(frame))`): one
    /// `[1, softTokens, textHidden]` array per image, in the text embedding
    /// space and the language model's token-embedding dtype (matching
    /// `prepare`'s `emb.dtype`).
    ///
    /// This is the seam external continuous-batching engines (CBv2
    /// multimodal prefill) use to precompute the embeddings they splice at
    /// the placeholder spans; `prepare` itself is unchanged and shares the
    /// same private loop, so the two paths cannot drift.
    public func perImageVisionFeatures(pixels: MLXArray, frames: [THW]?) -> [MLXArray] {
        // Same dtype resolution as `prepare`: the token-embedding output
        // dtype (dtype/shape are lazy graph metadata — nothing evaluates).
        let dtype = languageModel.model.emb(MLXArray([Int32(0)]).reshaped([1, 1])).dtype
        return visionFeatureList(pixels: pixels, frames: frames, isVideo: false)
            .map { $0.asType(dtype) }
    }
}

// MARK: - Processor

/// Nested `video_processor` sub-config from `preprocessor_config.json`. Gemma4
/// ships a separate, lower per-frame soft-token budget for video than for images
/// (`video_processor.max_soft_tokens` ~70 vs the image `max_soft_tokens` 280).
private struct Gemma4VideoProcessorConfiguration: Codable {
    let maxSoftTokens: Int?
    enum CodingKeys: String, CodingKey {
        case maxSoftTokens = "max_soft_tokens"
    }
}

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let patchSize: Int
    public let maxSoftTokens: Int
    public let videoMaxSoftTokens: Int
    public let poolingKernelSize: Int
    public let imageSeqLength: Int
    public let audioSeqLength: Int
    // Gemma4 wraps every image / video-frame soft-token block with begin-of-image
    // (boi) and end-of-image (eoi) delimiter tokens, matching the Python
    // processor. These are ordinary vocab tokens (embedded normally, attended
    // causally) — they are NOT image/video soft tokens, so they do not affect the
    // maskedScatter count or the bidirectional visual-span mask.
    public let boiTokenId: Int
    public let eoiTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case patchSize = "patch_size"
        case maxSoftTokens = "max_soft_tokens"
        case poolingKernelSize = "pooling_kernel_size"
        case imageSeqLength = "image_seq_length"
        case audioSeqLength = "audio_seq_length"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
    }

    // `video_processor` is a nested block, so it is read via its own keyed
    // container below rather than the main `CodingKeys`. `videoMaxSoftTokens` is
    // therefore decoded separately and intentionally omitted from `CodingKeys`
    // (the synthesized Encodable simply skips it; this config is decode-only in
    // practice). Mirrors the `TopKeys`/`rope_parameters` pattern in
    // `Gemma4VisionConfig`.
    enum VideoProcessorTopKeys: String, CodingKey {
        case videoProcessor = "video_processor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "Gemma4Processor"
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        maxSoftTokens = try c.decodeIfPresent(Int.self, forKey: .maxSoftTokens) ?? 280
        // Video frames use the nested `video_processor.max_soft_tokens` budget (~70),
        // not the image `max_soft_tokens` (280). Tolerate the key being absent or a
        // non-dict so text/image-only configs keep loading.
        var decodedVideoSoftTokens: Int? = nil
        if let vc = try? decoder.container(keyedBy: VideoProcessorTopKeys.self),
            let videoSub = try? vc.decodeIfPresent(
                Gemma4VideoProcessorConfiguration.self, forKey: .videoProcessor)
        {
            decodedVideoSoftTokens = videoSub.maxSoftTokens
        }
        videoMaxSoftTokens = decodedVideoSoftTokens ?? 70
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        imageSeqLength = try c.decodeIfPresent(Int.self, forKey: .imageSeqLength) ?? 280
        audioSeqLength = try c.decodeIfPresent(Int.self, forKey: .audioSeqLength) ?? 750
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258_882
    }
}

/// Gemma4-specific chat message generator. Unlike `Qwen2VLMessageGenerator`,
/// Gemma4's chat template expects SYSTEM messages as a plain string `content`
/// (the template folds the system text into the first user turn); only the
/// non-system turns carry typed multimodal content arrays. Reusing Qwen2VL's
/// generator emits a typed array for the system role too, which renders malformed
/// Gemma4 prompts. Modality order is images, then videos, then text — matching the
/// HF Gemma4 chat template / processor. (PR #56's vMLX decode port dropped this;
/// `UserInputTests.testGemma4Conversion*` still pin the plain-string-system
/// contract, so they reference this type.)
public struct Gemma4MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        if message.role == .system {
            return [
                "role": message.role.rawValue,
                "content": message.content,
            ]
        }
        return [
            "role": message.role.rawValue,
            "content": message.images.map { _ in ["type": "image"] }
                + message.videos.map { _ in ["type": "video"] }
                + [["type": "text", "text": message.content]],
        ]
    }
}

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config; self.tokenizer = tokenizer
    }

    /// Resize a single image/frame to Gemma4's soft-token budget and convert it
    /// to the sRGB tone curve the vision tower was trained on (PIL/Python
    /// default). Shared by the image and video paths so frames go through the
    /// exact same pipeline.
    private func resizedSRGB(from ci: CIImage, maxSoftTokens: Int? = nil) throws -> CIImage {
        let ps = config.patchSize
        let budget = maxSoftTokens ?? config.maxSoftTokens
        let maxP = budget * config.poolingKernelSize * config.poolingKernelSize
        // Reject zero-area, infinite, and NaN extents explicitly. The scale-factor
        // math below divides by `w * h`; a CIImage with a zero extent produces an
        // infinite `f` and a NaN trap inside `Int(floor(.nan))`. A non-finite
        // extent (e.g. `CIImage(color:)` returns `(.infinity, .infinity)`) traps
        // even earlier inside `Int(.infinity)`. Both surface as
        // VLMError.imageProcessingFailure now.
        let (h, w) = try gemma4IntExtent(ci.extent.size)
        let f = sqrt(Float(maxP * ps * ps) / Float(w * h))
        let sm = config.poolingKernelSize * ps
        var tH = Int(floor(f * Float(h) / Float(sm))) * sm
        var tW = Int(floor(f * Float(w) / Float(sm))) * sm
        if tH == 0 { tH = sm }
        if tW == 0 { tW = sm }
        let resized = MediaProcessing.resampleBicubic(ci, to: CGSize(width: tW, height: tH))
        return MediaProcessing.inSRGBToneCurveSpace(resized)
    }

    /// Resize + convert one image/frame to a `[1, C, H, W]` (NCHW) pixel tensor
    /// with float values in `[0, 1]`.
    private func resizedPixels(from ci: CIImage) throws -> MLXArray {
        MediaProcessing.asMLXArray(try resizedSRGB(from: ci))
    }

    /// Pack per-image/per-frame `[1, C, H, W]` tensors into one flat batch and
    /// record each one's real (un-padded) size in `frames`, so the vision tower
    /// processes each at its original size. Shorter tensors are zero-padded up to
    /// the batch max so they can share a single storage tensor.
    private func packFrames(_ arrays: [MLXArray]) -> (pixels: MLXArray, frames: [THW]) {
        let sizes = arrays.map { THW(1, $0.dim(2), $0.dim(3)) }
        if arrays.count == 1 { return (arrays[0], sizes) }
        let maxH = arrays.map { $0.dim(2) }.max()!
        let maxW = arrays.map { $0.dim(3) }.max()!
        let stored = arrays.map { arr -> MLXArray in
            let h = arr.dim(2)
            let w = arr.dim(3)
            if h == maxH && w == maxW { return arr }
            return MLX.padded(arr, widths: [[0, 0], [0, 0], [0, maxH - h], [0, maxW - w]])
        }
        return (concatenated(stored), sizes)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(from: input)
        var tokens = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        // ── Images ── each image is resized independently (aspect-ratio
        // preserving) and stored as a flat `[N, C, H, W]` batch indexed by
        // `frames`; the vision tower processes each at its real, un-padded size.
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let arrays = try input.images.map { try resizedPixels(from: $0.asCIImage()) }
            let packed = packFrames(arrays)
            processedImage = LMInput.ProcessedImage(pixels: packed.pixels, frames: packed.frames)
        }

        // ── Videos ── frames are sampled (~32 spread uniformly across the whole
        // clip, matching the Python Gemma4VideoProcessor) and resized with the
        // lower per-frame VIDEO soft-token budget (`video_processor.max_soft_tokens`,
        // ~70) rather than the image budget (280), then stored the same flat way.
        // Per-frame timestamps (seconds) are captured for the `mm:ss` prompt prefix
        // (HF `Gemma4Processor.replace_video_token`); the per-video soft-token count
        // is data-driven from the resized frame grid so it matches the vision tower's
        // pooled output exactly.
        var processedVideo: LMInput.ProcessedVideo?
        var videoTimestamps: [[Double]] = []
        var videoSoftTokenCounts: [Int] = []
        if !input.videos.isEmpty {
            var frames: [MLXArray] = []
            let poolSq = config.poolingKernelSize * config.poolingKernelSize
            for video in input.videos {
                let sequence = try await MediaProcessing.asProcessedSequence(
                    video, targetFPS: { duration in 32.0 / max(duration.seconds, 1.0) },
                    maxFrames: 32
                ) { frame in
                    VideoFrame(
                        frame: try resizedSRGB(
                            from: frame.frame, maxSoftTokens: config.videoMaxSoftTokens),
                        timeStamp: frame.timeStamp)
                }
                videoTimestamps.append(
                    sequence.timestamps.map { $0.seconds.isFinite ? $0.seconds : 0 })
                // All frames of one video share the same resized size, so the
                // per-frame soft-token count is uniform; derive it from the first
                // frame's grid (`(h/patch)*(w/patch)/pool^2`), matching the model's
                // `videoOutputLength` and the placeholder expansion below.
                if let first = sequence.frames.first {
                    let h = first.dim(2)
                    let w = first.dim(3)
                    videoSoftTokenCounts.append(
                        max(1, ((h / config.patchSize) * (w / config.patchSize)) / max(1, poolSq)))
                } else {
                    videoSoftTokenCounts.append(config.videoMaxSoftTokens)
                }
                frames.append(contentsOf: sequence.frames)
            }
            if !frames.isEmpty {
                let packed = packFrames(frames)
                processedVideo = LMInput.ProcessedVideo(pixels: packed.pixels, frames: packed.frames)
            }
        }

        // ── Expand placeholders into delimited soft-token blocks ──
        // Gemma4 represents each image / video frame as `boi + soft_token*count +
        // eoi`. For images `count` is `imageSeqLength` (== the model's
        // `defaultOutputLength`); for video frames it is the per-frame video budget
        // (`video_processor.max_soft_tokens`, derived from the resized grid), and
        // each block is additionally prefixed with the frame's `mm:ss` timestamp.
        // Either way the expanded soft-token count matches the vision features
        // `maskedScatter` writes. The boi/eoi delimiters are ordinary tokens (not
        // soft tokens) so they do not change that count or the visual-span mask.
        //
        // `convertTokenToId` is used rather than `encode("<|image|>").last` so the
        // lookup goes straight through the tokenizer's special-token map and never
        // picks up an appended BOS/EOS — `encode(text:)` defaults to
        // `addSpecialTokens: true`, which on some tokenizers prepends BOS. The
        // numeric fallbacks cover tokenizers that don't expose the placeholders as
        // addable special tokens.
        if processedImage != nil || processedVideo != nil {
            let imgId = tokenizer.convertTokenToId("<|image|>") ?? 258_880
            let vidId = tokenizer.convertTokenToId("<|video|>") ?? 258_884
            func appendBlock(_ acc: inout [Int], _ tokenId: Int, _ count: Int) {
                acc.append(config.boiTokenId)
                acc.append(contentsOf: Array(repeating: tokenId, count: count))
                if let eoi = config.eoiTokenId { acc.append(eoi) }
            }
            var expanded = [Int]()
            expanded.reserveCapacity(tokens.count)
            var videoIndex = 0
            for t in tokens {
                if t == imgId, processedImage != nil {
                    appendBlock(&expanded, imgId, config.imageSeqLength)
                } else if t == vidId, processedVideo != nil {
                    // One `mm:ss <boi> <|video|>*count <eoi>` block per sampled frame,
                    // joined by a space — mirrors HF Gemma4Processor.replace_video_token
                    // `" ".join("{ts} {boi}{video*n}{eoi}")`. The leading space on
                    // frames after the first reproduces that join separator. The mm:ss
                    // prefix is encoded as ordinary text tokens (addSpecialTokens:false
                    // so no BOS leaks in). One block per frame keeps the total video
                    // soft-token count equal to the vision features (frames == timestamps
                    // count); `count` is the per-frame video budget.
                    let timestamps =
                        videoIndex < videoTimestamps.count ? videoTimestamps[videoIndex] : []
                    let count =
                        videoIndex < videoSoftTokenCounts.count
                        ? videoSoftTokenCounts[videoIndex] : config.videoMaxSoftTokens
                    for (frameIdx, seconds) in timestamps.enumerated() {
                        let totalSeconds = max(0, Int(seconds.rounded(.towardZero)))
                        let tsString = String(
                            format: "%02d:%02d", totalSeconds / 60, totalSeconds % 60)
                        let prefix = (frameIdx == 0 ? "" : " ") + tsString + " "
                        expanded.append(
                            contentsOf: tokenizer.encode(text: prefix, addSpecialTokens: false))
                        appendBlock(&expanded, vidId, count)
                    }
                    videoIndex += 1
                } else {
                    expanded.append(t)
                }
            }
            tokens = expanded
        }

        let pa = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: pa, mask: ones(like: pa).asType(.int8)),
            image: processedImage,
            video: processedVideo)
    }
}
