//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Shared Gemma 4 namespace

/// Public namespace for Gemma 4 types that need cross-module visibility.
public enum Gemma4 {

    /// Position offset for RoPE, either a single scalar (standard decode)
    /// or a per-row `MLXArray` (continuous-batching paths).
    ///
    /// `@unchecked Sendable` because `MLXArray` is not `Sendable`; callers must
    /// treat these values as immutable snapshots (they are only ever read, never
    /// mutated in place).
    public enum PositionOffset: @unchecked Sendable {
        case scalar(Int)
        case batch(MLXArray)
        /// Graph-tracked offset from `CompilableKVCache.offsetArray`.
        /// Must stay as an `MLXArray` so `compile()` can trace through
        /// the RoPE computation without forcing a host readback.
        case graphArray(MLXArray)
    }
}

// MARK: - vMLX decode hot-path helpers (ported from osaurus/main Gemma4Text)
//
// File-private, self-contained compiled fusions. They do NOT depend on the
// SwitchLayers / HardwareInfo lane so this file builds stand-alone; when that
// lane lands public `safeGeluApproximate` / `MLXHardwareInfo` these can collapse
// to the shared symbols (identical math + same env knob) with no behavior change.
// `compile(shapeless: true)` is gated by `MLX_COMPILED_DECODE` (default on),
// mirroring `MLXHardwareInfo.isCompiledDecodeSupported`, so M1/M2 + macOS Tahoe
// (MLX #3329) can opt out without a code change. Matches the ungated
// `compiledSiluProduct` / `weightedExpertSum` convention already in this tree.
private let gemma4CompiledDecodeSupported: Bool = {
    if let raw = ProcessInfo.processInfo.environment["MLX_COMPILED_DECODE"] {
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }
    return true
}()

/// Approximate (tanh) GELU written with `x * x * x` instead of the Power
/// primitive (`x ** 3`) so it is safe under `compile(shapeless: true)` — the
/// Power primitive returns zero results on the Tahoe Metal JIT (MLX #3329).
/// Numerically identical to `MLXNN.geluApproximate` (vMLX `safeGeluApproximate`).
private let gemma4SafeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    return gemma4CompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Final-logit softcap (`tanh(x / cap) * cap`) fused into one Metal dispatch
/// (vMLX `compiledLogitSoftcap`). The untyped (float32) cap keeps the softcap
/// math — and the logits handed to the sampler — full precision.
private let gemma4CompiledLogitSoftcap: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (x: MLXArray, cap: MLXArray) -> MLXArray in
        tanh(x / cap) * cap
    }
    return gemma4CompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

// MARK: - Configuration

struct Gemma4WeightQuantizationMetadata: Codable, Sendable {
    var bits: Int?
    var groupSize: Int?

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }
}

public struct Gemma4TextConfiguration: Codable, Sendable {
    public internal(set) var modelType: String = "gemma4_text"
    public internal(set) var hiddenSize: Int = 1536
    public internal(set) var numHiddenLayers: Int = 35
    public internal(set) var intermediateSize: Int = 6144
    public internal(set) var numAttentionHeads: Int = 8
    public internal(set) var headDim: Int = 256
    public internal(set) var globalHeadDim: Int = 512
    public internal(set) var globalPartialRotaryFactor: Float = 0.25
    public internal(set) var rmsNormEps: Float = 1e-6
    public internal(set) var vocabSize: Int = 262144
    public internal(set) var vocabSizePerLayerInput: Int = 262144
    public internal(set) var numKeyValueHeads: Int = 1
    public internal(set) var numGlobalKeyValueHeads: Int?
    public var numKvSharedLayers: Int = 20
    public internal(set) var hiddenSizePerLayerInput: Int = 256
    public internal(set) var slidingWindow: Int = 512
    public internal(set) var slidingWindowPattern: Int = 5
    public internal(set) var maxPositionEmbeddings: Int = 131072
    public internal(set) var attentionKeqV: Bool = false
    public internal(set) var finalLogitSoftcapping: Float = 30.0
    public internal(set) var useDoubleWideMlp: Bool = true
    public internal(set) var layerTypes: [String] = []
    public internal(set) var tieWordEmbeddings: Bool = true
    public internal(set) var quantizationBits: Int?
    public internal(set) var quantizationGroupSize: Int?

    // MoE (only set on the 26B-A4B variant; 2B/4B/31B are dense)
    public internal(set) var enableMoeBlock: Bool = false
    public internal(set) var numExperts: Int?
    public internal(set) var topKExperts: Int?
    public internal(set) var moeIntermediateSize: Int?

    // RoPE parameters (nested dict with full_attention/sliding_attention sub-configs)
    public internal(set) var ropeParameters: [String: [String: StringOrNumber]]?

    // "vision" enables Gemma4 blockwise bidirectional attention within
    // image/video soft-token spans during prefill (mirrors the VLM twin's
    // G4TextConfig field). nil/other => ordinary causal. Only consulted when
    // a caller passes an `imageTokenMask`; pure-text configs omit it.
    public internal(set) var useBidirectionalAttention: String?

    // Derived properties
    public internal(set) var slidingRopeTheta: Float = 10000.0
    public internal(set) var fullRopeTheta: Float = 1_000_000.0
    public internal(set) var fullPartialRotaryFactor: Float = 1.0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case globalPartialRotaryFactor = "global_partial_rotary_factor"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKeqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case layerTypes = "layer_types"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case useBidirectionalAttention = "use_bidirectional_attention"
    }

    enum QuantizationCodingKeys: String, CodingKey {
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let quantizationContainer = try decoder.container(keyedBy: QuantizationCodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 35
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        self.globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        self.globalPartialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .globalPartialRotaryFactor) ?? 0.25
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        self.vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 262144
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        self.numGlobalKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        self.numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 20
        self.hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 256
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        self.slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.attentionKeqV =
            try container.decodeIfPresent(Bool.self, forKey: .attentionKeqV) ?? false
        self.finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        self.useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? true
        if let decoded = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            self.layerTypes = decoded
        } else {
            // Derive layer types from sliding window pattern
            var pattern = [String]()
            for i in 0 ..< slidingWindowPattern {
                pattern.append(
                    i == slidingWindowPattern - 1 ? "full_attention" : "sliding_attention")
            }
            var types = [String]()
            while types.count < numHiddenLayers {
                types.append(contentsOf: pattern)
            }
            self.layerTypes = Array(types.prefix(numHiddenLayers))
        }
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        let quantization =
            try quantizationContainer.decodeIfPresent(
                Gemma4WeightQuantizationMetadata.self, forKey: .quantization)
            ?? quantizationContainer.decodeIfPresent(
                Gemma4WeightQuantizationMetadata.self, forKey: .quantizationConfig)
        self.quantizationBits = quantization?.bits
        self.quantizationGroupSize = quantization?.groupSize
        self.ropeParameters =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)

        // MoE (Gemma 4 26B-A4B)
        self.enableMoeBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        self.topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        self.useBidirectionalAttention =
            try container.decodeIfPresent(String.self, forKey: .useBidirectionalAttention)

        // Extract RoPE parameters from nested config
        if let ropeParams = ropeParameters {
            if let sliding = ropeParams["sliding_attention"] {
                self.slidingRopeTheta = sliding["rope_theta"]?.asFloat() ?? 10000.0
            }
            if let full = ropeParams["full_attention"] {
                self.fullRopeTheta = full["rope_theta"]?.asFloat() ?? 1_000_000.0
                self.fullPartialRotaryFactor =
                    full["partial_rotary_factor"]?.asFloat() ?? 1.0
            }
        }
    }
}

extension Gemma4TextConfiguration {

    /// Predicate for whether a layer uses shared K/V (consuming it from an
    /// earlier layer rather than projecting its own).
    ///
    /// A layer is shared when either:
    /// - `forceSharedKV` is true (drafter / assistant models where every layer
    ///   borrows K/V from the target), or
    /// - the config declares `numKvSharedLayers > 0` AND this layer's index
    ///   falls within the trailing shared block.
    public func layerUsesSharedKV(layerIdx: Int, forceSharedKV: Bool = false) -> Bool {
        if forceSharedKV { return true }
        guard numKvSharedLayers > 0 else { return false }
        let firstShared = numHiddenLayers - numKvSharedLayers
        return layerIdx >= firstShared
    }
}

// MARK: - Helper Modules

private class RMSNormNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

private class ScaledLinear: Module {
    let weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.weight = MLXArray.zeros([outFeatures, inFeatures])
        self.scalar = scalar
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T) * scalar
    }
}

@inline(__always)
internal func gemma4CapturePositionOffset(from cache: KVCache?) -> Gemma4.PositionOffset {
    if let compilableRot = cache as? CompilableRotatingKVCache {
        // Snapshot: `+ 0` creates a graph-safe copy so cache.update()
        // advancing offsetArray doesn't shift the query RoPE position.
        .graphArray(compilableRot.offsetArray + 0)
    } else if let compilable = cache as? CompilableKVCache {
        // Snapshot: `+ 0` creates a graph-safe copy so cache.update()
        // advancing offsetArray doesn't shift the query RoPE position.
        .graphArray(compilable.offsetArray + 0)
    } else if let batchCache = cache as? BatchPositionedKVCache {
        // Snapshot the per-sequence offsets before cache.update(...) advances them.
        .batch(batchCache.batchOffset + 0)
    } else {
        .scalar(cache?.offset ?? 0)
    }
}

@inline(__always)
internal func gemma4ApplyRotaryPosition<R: RoPELayer>(
    _ rope: R,
    to x: MLXArray,
    offset: Gemma4.PositionOffset
) -> MLXArray {
    switch offset {
    case .scalar(let value):
        rope(x, offset: value)
    case .batch(let values):
        rope(x, offset: values)
    case .graphArray(let offsetArray):
        rope(x, offset: offsetArray)
    }
}

private func gemma4AttentionFallback(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let (B, nQHeads, L, D) = (
        queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)
    )
    let nKVHeads = keys.dim(1)
    let repeats = nQHeads / nKVHeads

    var q = queries * scale
    var k = keys
    var v = values
    if repeats > 1 {
        q = q.reshaped([B, nKVHeads, repeats, L, D])
        k = expandedDimensions(k, axis: 2)
        v = expandedDimensions(v, axis: 2)
    }

    var scores = matmul(q, k.swappedAxes(-1, -2))

    func applyMask(_ maskArray: MLXArray) {
        var mask = maskArray
        if scores.ndim == 5 && mask.ndim == 4 && mask.dim(0) == scores.dim(0) {
            mask = expandedDimensions(mask, axis: 2)
        }
        if mask.dtype == .bool {
            scores = MLX.where(
                mask, scores, MLXArray(-Float.infinity, dtype: scores.dtype))
        } else {
            scores = scores + mask
        }
    }

    switch mask {
    case .none:
        break
    case .causal:
        let qL = scores.dim(-2)
        let kL = scores.dim(-1)
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1),
            expandedDimensions(kIndices, axis: -2))
        applyMask(causalMask)
    case .array(let maskArray):
        applyMask(maskArray)
    case .arrays(let maskArrays):
        if let maskArray = maskArrays.first {
            applyMask(maskArray)
        }
    }

    var probs = softmax(scores.asType(.float32), axis: -1, precise: true)
    // A fully-masked query row (every key masked -> all -inf) softmaxes to NaN.
    // For left-padded batches these are the padding query positions, whose
    // outputs are discarded — but `0 * NaN = NaN` in the value matmul below
    // would propagate NaN into the hidden state, and a later layer's real
    // queries (which mask padding keys to weight 0) then hit `0 * NaN` again
    // and corrupt EVERY row of the batch. Map NaN -> 0 so a fully-masked query
    // contributes nothing. This matches `MLXFast.scaledDotProductAttention`,
    // which this manual fallback replaces for the batched (ragged) path.
    probs = MLX.where(probs .!= probs, MLXArray(Float(0)), probs)
    scores = probs.asType(scores.dtype)
    var output = matmul(scores, v)
    if repeats > 1 {
        output = output.reshaped([B, nQHeads, L, values.dim(3)])
    }
    return output
}

// MARK: - Attention

private class Gemma4Attention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let effectiveHeadDim: Int
    let nHeads: Int
    let nKvHeads: Int
    let useKeqV: Bool
    let usesSharedKV: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    @ModuleInfo(key: "v_norm") var vNorm: RMSNormNoScale?

    @ModuleInfo var rope: RoPELayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int, forceSharedKV: Bool = false) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.usesSharedKV = config.layerUsesSharedKV(
            layerIdx: layerIdx, forceSharedKV: forceSharedKV)

        // Full attention uses globalHeadDim, sliding uses headDim
        self.effectiveHeadDim =
            isSliding ? config.headDim : config.globalHeadDim

        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads

        // K-eq-V for full attention layers
        self.useKeqV = config.attentionKeqV && !isSliding
        if useKeqV, let globalKvHeads = config.numGlobalKeyValueHeads {
            self.nKvHeads = globalKvHeads
        } else {
            self.nKvHeads = config.numKeyValueHeads
        }

        self.scale = 1.0

        self._qProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        if !usesSharedKV {
            self._kProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
            if !useKeqV {
                self._vProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
            }
            self._kNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
            self._vNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)
        }
        self._oProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)

        // RoPE: sliding uses default, full uses proportional with partial rotation
        if isSliding {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.slidingRopeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.fullRopeTheta, traditional: false,
                scalingConfig: [
                    "type": .string("proportional"),
                    "partial_rotary_factor": .float(config.fullPartialRotaryFactor),
                ],
                maxPositionEmbeddings: nil)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        positionOffset: Gemma4.PositionOffset? = nil,
        v2SharedSource: (any CBv2AttendingLayerCache)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4.PositionOffset) {
        // ContinuousBatchingV2: the layer cache owns both the KV update and
        // the attention computation (no masks, no padding — see
        // CBv2Contracts.swift). Entirely separate branch; the legacy paths
        // below are untouched.
        if let layerCacheV2 = cache as? (any CBv2AttendingLayerCache) {
            return forwardV2(
                x, layerCache: layerCacheV2, source: v2SharedSource,
                sharedKV: sharedKV, positionOffset: positionOffset)
        }

        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
        queries = qNorm(queries)

        let keys: MLXArray
        let values: MLXArray
        let activePositionOffset = positionOffset ?? gemma4CapturePositionOffset(from: cache)

        if let (sharedK, sharedV) = sharedKV {
            // KV-shared layers use pre-computed KV from an earlier layer
            keys = sharedK
            values = sharedV
        } else {
            guard let kProj, let kNorm, let vNorm else {
                preconditionFailure("Gemma4 shared-KV layers require sharedKV input")
            }

            let kRaw = kProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
            var k = kNorm(kRaw)
            k = k.transposed(0, 2, 1, 3)
            k = gemma4ApplyRotaryPosition(rope, to: k, offset: activePositionOffset)

            // K-eq-V (`attention_k_eq_v: true` on Gemma 4 26B/31B):
            // values reuses the raw key projection (pre-norm), then goes
            // through its own `vNorm` and transpose to land in the same
            // `[B, n_kv_heads, L, D]` layout as keys.
            var v: MLXArray
            if let vProj {
                v = vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
            } else {
                v = kRaw
            }
            v = vNorm(v)
            v = v.transposed(0, 2, 1, 3)

            if let cache {
                let (updatedK, updatedV) = cache.update(keys: k, values: v)
                keys = updatedK
                values = updatedV
            } else {
                keys = k
                values = v
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: activePositionOffset)

        // Adjust mask if cache size differs from mask size
        var adjustedMask = mask
        if case .array(let maskArray) = mask {
            let keysSeqLen = keys.dim(2)
            if maskArray.dim(-1) != keysSeqLen {
                adjustedMask = .array(maskArray[.ellipsis, 0 ..< keysSeqLen])
            }
        }

        let hasCachedPrefix: Bool
        switch activePositionOffset {
        case .scalar(let offset):
            hasCachedPrefix = offset > 0
        case .batch:
            hasCachedPrefix = true
        case .graphArray:
            // CompilableKVCache: can't read Int offset without readback.
            // During compiled decode L==1, so L>1 && hasCachedPrefix is
            // false anyway. Setting true is safe for the prefill path.
            hasCachedPrefix = true
        }

        let attention: MLXArray
        if L > 1 && hasCachedPrefix {
            attention = gemma4AttentionFallback(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: adjustedMask ?? .none)
        } else {
            attention = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: adjustedMask ?? .none
            )
        }

        let output = attention
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return (oProj(output), (keys, values), activePositionOffset)
    }

    /// ContinuousBatchingV2 attention path. The `CBv2AttendingLayerCache`
    /// owns the KV update AND the attention computation, so this method only
    /// projects/normalizes/ropes Q (and K/V for non-shared layers) and
    /// dispatches. The model never builds masks and never pads — decode is
    /// rectangular `[B, 1]`, prefill is per-request `[1, chunk]`.
    ///
    /// Invariant 1 (report 10 §4): RoPE offsets are per-row absolutes,
    /// snapshotted BEFORE `updateAndAttend` advances the rows, and KV-shared
    /// layers reuse the SOURCE layer's captured snapshot byte-identically.
    private func forwardV2(
        _ x: MLXArray,
        layerCache: any CBv2AttendingLayerCache,
        source: (any CBv2AttendingLayerCache)?,
        sharedKV: (MLXArray, MLXArray)?,
        positionOffset: Gemma4.PositionOffset?
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4.PositionOffset) {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)

        if usesSharedKV {
            // KV-shared layer: projects queries only and borrows (K, V) from
            // the source layer's cache at attention time. The RoPE offsets
            // MUST be the source layer's pre-update snapshot (threaded by the
            // trunk) — reading `source.positionOffsets` here would observe
            // positions already advanced by the source's update this step.
            guard let source, let positionOffset, let sharedKV else {
                preconditionFailure(
                    """
                    Gemma4 CBv2 shared-KV layer \(layerIdx) requires the source \
                    layer cache, its captured position offsets, and its per-step \
                    K/V (threaded by Gemma4TextModelInner)
                    """)
            }
            queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: positionOffset)
            let attention = layerCache.attendBorrowing(
                source: source, queries: queries, scale: scale, sinks: nil)
            let output = attention.transposed(0, 2, 1, 3).reshaped(B, L, -1)
            return (oProj(output), sharedKV, positionOffset)
        }

        guard let kProj, let kNorm, let vNorm else {
            preconditionFailure("Gemma4 non-shared layers require K/V projection modules")
        }

        // Snapshot the per-row absolute offsets BEFORE updateAndAttend
        // advances the rows (`+ 0` = graph-safe copy, same convention as
        // gemma4CapturePositionOffset). KV-shared consumers of this layer
        // reuse this exact snapshot via the returned PositionOffset.
        let capturedOffsets = layerCache.positionOffsets + 0
        let captured = Gemma4.PositionOffset.batch(capturedOffsets)

        queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: captured)

        let kRaw = kProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
        var k = kNorm(kRaw)
        k = k.transposed(0, 2, 1, 3)
        k = gemma4ApplyRotaryPosition(rope, to: k, offset: captured)

        // K-eq-V (`attention_k_eq_v: true` on Gemma 4 26B/31B): values reuse
        // the raw key projection (pre-norm) through their own vNorm — same as
        // the legacy path.
        var v: MLXArray
        if let vProj {
            v = vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
        } else {
            v = kRaw
        }
        v = vNorm(v)
        v = v.transposed(0, 2, 1, 3)

        let attention = layerCache.updateAndAttend(
            queries: queries, keys: k, values: v, scale: scale, sinks: nil)

        let output = attention.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return (oProj(output), (k, v), captured)
    }
}

// MARK: - MoE (26B-A4B)

/// Expert router. Norms `x` with a learnable scale, projects to expert
/// scores, and returns top-K (indices, weights) where weights are
/// softmax-normalized and scaled by a per-expert scalar.
private class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    let topK: Int
    let eps: Float
    let rootSize: Float

    init(_ config: Gemma4TextConfiguration) {
        precondition(
            config.numExperts != nil && config.topKExperts != nil,
            "Gemma4Router requires num_experts and top_k_experts in the config"
        )
        let numExperts = config.numExperts ?? 0
        self.topK = config.topKExperts ?? 0
        self.eps = config.rmsNormEps
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (topKIndices: MLXArray, topKWeights: MLXArray) {
        let normed = MLXFast.rmsNorm(x, weight: scale * rootSize, eps: eps)
        let expertScores = proj(normed)

        let kth = expertScores.dim(-1) - topK
        var topKIndices = MLX.argPartition(expertScores, kth: kth, axis: -1)
        topKIndices = topKIndices[.ellipsis, kth...]

        var topKWeights = MLX.takeAlong(expertScores, topKIndices, axis: -1)
        topKWeights = MLX.softmax(topKWeights, axis: -1, precise: true)
        topKWeights = topKWeights * perExpertScale[topKIndices]

        return (topKIndices, topKWeights)
    }
}

/// Sparse MoE feed-forward block. Wraps `SwitchGLU` with GeGLU activation.
private class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(_ config: Gemma4TextConfiguration) {
        let numExperts = config.numExperts ?? 1
        let moeIntermediate = config.moeIntermediateSize ?? config.intermediateSize

        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeIntermediate,
            numExperts: numExperts,
            activation: { gemma4SafeGeluApproximate($0) },
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        // Flatten [B, S, H] -> [B*S, H] (and indices/weights to [B*S, K]) so
        // SwitchGLU runs its optimized 2D gather/sort path, then fuse the
        // per-expert scale + reduce via the shared compiled `weightedExpertSum`
        // and reshape back. Numerically identical to the prior
        // `(expandDims(weights, -1) * switchGLU(x, idx)).sum(-2)`; matches vMLX.
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = topKIndices.dim(-1)
        let y = switchGLU(x.reshaped(B * S, H), topKIndices.reshaped(B * S, K))
        return weightedExpertSum(y, topKWeights.reshaped(B * S, K)).reshaped(B, S, H)
    }
}

// MARK: - MLP

private class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let isKvSharedLayer = config.layerUsesSharedKV(layerIdx: layerIdx)
        let useDoubleWide = config.useDoubleWideMlp && isKvSharedLayer
        let intermediateSize = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(gemma4SafeGeluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

/// Gemma 4 decoder layer. Combines `Gemma4Attention` with an MLP (or MoE)
/// block, the per-layer-input (PLE) path, and residual / layer-scalar
/// plumbing. Consumed by `Gemma4TextModelInner`; not intended as a
/// user-facing composable layer.
public class Gemma4DecoderLayer: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "self_attn") fileprivate var selfAttn: Gemma4Attention
    @ModuleInfo fileprivate var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    // MoE-only modules (26B-A4B); nil on dense variants.
    @ModuleInfo(key: "router") fileprivate var router: Gemma4Router?
    @ModuleInfo(key: "experts") fileprivate var experts: Gemma4Experts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm?

    // Per-layer input (PLE) gating
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    // Per-layer scalar
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    let isMoE: Bool

    public init(_ config: Gemma4TextConfiguration, layerIdx: Int, forceSharedKV: Bool = false) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        self.isMoE = config.enableMoeBlock

        self._selfAttn.wrappedValue = Gemma4Attention(
            config, layerIdx: layerIdx, forceSharedKV: forceSharedKV)
        self._mlp.wrappedValue = Gemma4MLP(config, layerIdx: layerIdx)

        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.enableMoeBlock {
            self._router.wrappedValue = Gemma4Router(config)
            self._experts.wrappedValue = Gemma4Experts(config)
            self._postFeedforwardLayernorm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        if hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1], dtype: .float16)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        positionOffset: Gemma4.PositionOffset? = nil,
        v2SharedSource: (any CBv2AttendingLayerCache)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4.PositionOffset) {
        let residual = x

        let h = inputLayernorm(x)
        let (attnOut, kvPair, attnPositionOffset) = selfAttn(
            h, mask: mask, cache: cache, sharedKV: sharedKV, positionOffset: positionOffset,
            v2SharedSource: v2SharedSource)
        let postAttn = postAttentionLayernorm(attnOut)
        var out = residual + postAttn

        let residual2 = out

        if isMoE,
            let router,
            let experts,
            let postFeedforwardLayernorm1,
            let preFeedforwardLayernorm2,
            let postFeedforwardLayernorm2
        {
            // Dense + sparse branches in parallel, summed into one residual.
            var h1 = preFeedforwardLayernorm(out)
            h1 = mlp(h1)
            h1 = postFeedforwardLayernorm1(h1)

            let (topKIndices, topKWeights) = router(out)
            var h2 = preFeedforwardLayernorm2(out)
            h2 = experts(h2, topKIndices: topKIndices, topKWeights: topKWeights)
            h2 = postFeedforwardLayernorm2(h2)

            out = h1 + h2
        } else {
            out = preFeedforwardLayernorm(out)
            out = mlp(out)
        }

        out = postFeedforwardLayernorm(out)
        out = residual2 + out

        // PLE gating
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let perLayerInput
        {
            let residual3 = out
            var g = gate(out)
            g = gemma4SafeGeluApproximate(g)
            g = g * perLayerInput
            g = proj(g)
            g = norm(g)
            out = residual3 + g
        }

        out = out * layerScalar

        return (out, kvPair, attnPositionOffset)
    }
}

// MARK: - Text Model

/// Inner Gemma 4 trunk: embeddings + per-layer-input (PLE) + 35 decoder
/// layers + final norm. Not intended as a user-facing model — use
/// `Gemma4TextModel` for standalone inference.
public class Gemma4TextModelInner: Module {
    let config: Gemma4TextConfiguration
    let embedScale: Float
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [Gemma4DecoderLayer]
    @ModuleInfo public var norm: RMSNorm

    // Per-layer embeddings (PLE)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") fileprivate var perLayerModelProjection: ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    // KV sharing mapping: for each layer, which earlier layer provides KVs
    let previousKvs: [Int]
    let firstKvSharedLayerIdx: Int

    /// Index of the last non-shared full-attention layer (-1 if none).
    /// Used by the shared-KV capture hook.
    let lastFullAttentionNonSharedIdx: Int
    let lastSlidingAttentionNonSharedIdx: Int

    public init(_ config: Gemma4TextConfiguration, forceSharedKV: Bool = false) {
        self.config = config
        self.embedScale = Float(config.hiddenSize).squareRoot()
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4DecoderLayer(config, layerIdx: $0, forceSharedKV: forceSharedKV)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // PLE
        if config.hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * config.hiddenSizePerLayerInput)
            self._perLayerModelProjection.wrappedValue = ScaledLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.numHiddenLayers * config.hiddenSizePerLayerInput,
                scalar: pow(Float(config.hiddenSize), -0.5))
            self._perLayerProjectionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        // Build KV-sharing map
        self.firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        var kvMap = Array(0 ..< config.numHiddenLayers)
        if config.numKvSharedLayers > 0 {
            // Find the last non-shared layer of each type
            var lastByType = [String: Int]()
            for i in 0 ..< firstKvSharedLayerIdx {
                lastByType[config.layerTypes[i]] = i
            }
            // Shared layers reference the last non-shared layer of the same type
            for j in firstKvSharedLayerIdx ..< config.numHiddenLayers {
                if let prev = lastByType[config.layerTypes[j]] {
                    kvMap[j] = prev
                }
            }
        }
        self.previousKvs = kvMap

        // Capture indices: the last layer of each type that
        // still has its own K/V (not shared from an earlier layer).
        let firstShared = self.firstKvSharedLayerIdx
        var lastFull = -1
        var lastSliding = -1
        for i in 0 ..< firstShared {
            if config.layerTypes[i] == "full_attention" { lastFull = i }
            if config.layerTypes[i] == "sliding_attention" { lastSliding = i }
        }
        self.lastFullAttentionNonSharedIdx = lastFull
        self.lastSlidingAttentionNonSharedIdx = lastSliding

        super.init()
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        captureHook: ((Int, (MLXArray, MLXArray)) -> Void)? = nil,
        inputEmbedding: MLXArray? = nil,
        imageTokenMask: MLXArray? = nil
    ) -> MLXArray {
        forwardTrunk(
            inputs, cache: cache, captureHook: captureHook, capturePreNorm: false,
            inputEmbedding: inputEmbedding, imageTokenMask: imageTokenMask
        ).postNorm
    }

    /// Variant that ALSO returns the pre-norm last-layer hidden state
    /// (HF captures `hidden_states` at the decoder-layer boundary,
    /// BEFORE `model.norm`); the LM head consumes the post-norm hidden.
    /// The standard path goes through `callAsFunction`.
    public func callCapturingPreNorm(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        captureHook: ((Int, (MLXArray, MLXArray)) -> Void)? = nil
    ) -> (postNorm: MLXArray, preNorm: MLXArray) {
        let r = forwardTrunk(
            inputs, cache: cache, captureHook: captureHook, capturePreNorm: true)
        return (r.postNorm, r.preNorm!)
    }

    private func forwardTrunk(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureHook: ((Int, (MLXArray, MLXArray)) -> Void)?,
        capturePreNorm: Bool,
        inputEmbedding: MLXArray? = nil,
        imageTokenMask: MLXArray? = nil
    ) -> (postNorm: MLXArray, preNorm: MLXArray?) {
        // Vision prefill (mirrors the inline VLM twin `TextModel.callAsFunction`):
        // `inputEmbedding` — the scaled text embeddings with image soft-token
        // embeddings spliced at placeholder positions — replaces the trunk's
        // own lookup; token ids still feed the per-layer embeddings (PLE)
        // below. nil keeps the text path byte-identical.
        var h: MLXArray
        if let inputEmbedding {
            h = inputEmbedding.ndim == 2 ? inputEmbedding.expandedDimensions(axis: 0) : inputEmbedding
        } else {
            h = embedTokens(inputs) * embedScale
        }

        // Compute per-layer inputs (PLE)
        var perLayerInputs: [MLXArray?]
        if hiddenSizePerLayerInput > 0,
            let embedPerLayer = embedTokensPerLayer,
            let modelProj = perLayerModelProjection,
            let projNorm = perLayerProjectionNorm
        {
            // Token-based PLE
            let tokenPLE =
                embedPerLayer(inputs)
                * Float(config.hiddenSizePerLayerInput).squareRoot()

            // [B, L, numLayers * hiddenSizePerLayerInput] -> [B, L, numLayers, hiddenSizePerLayerInput]
            let reshapedTokenPLE = tokenPLE.reshaped(
                tokenPLE.dim(0), tokenPLE.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)

            // Model projection PLE
            let modelPLE = modelProj(h).reshaped(
                h.dim(0), h.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)
            let normedModelPLE = projNorm(modelPLE)

            // Combine: (model_proj + token_embed) * 2^{-0.5}
            let perLayerInputScale = pow(Float(2.0), -0.5)
            let combined = (normedModelPLE + reshapedTokenPLE) * perLayerInputScale

            perLayerInputs = (0 ..< config.numHiddenLayers).map { i in
                combined[.ellipsis, i, 0...]
            }
        } else {
            perLayerInputs = Array(repeating: nil, count: config.numHiddenLayers)
        }

        // Extend cache array for shared layers (which get nil caches)
        var fullCache: [KVCache?]
        if let cache {
            fullCache = cache.map { Optional($0) }
            while fullCache.count < config.numHiddenLayers {
                fullCache.append(nil)
            }
        } else {
            fullCache = Array(repeating: nil, count: config.numHiddenLayers)
        }

        // ContinuousBatchingV2 detection: v2 layer caches own attention AND
        // masking, so the trunk builds no masks at all on that path (there is
        // no padding and no shared frontier to mask). In v2 mode every layer
        // (including KV-shared ones) has a cache object.
        let isCBv2 = fullCache.contains { ($0 as? (any CBv2AttendingLayerCache)) != nil }

        // Build masks: one per attention type (legacy path only).
        //
        // Vision prefill (mirrors the inline VLM twin): when the config
        // enables `use_bidirectional_attention == "vision"` and the caller
        // passes an `imageTokenMask` ([B, L] bool, true at image soft-token
        // positions), overlay blockwise bidirectional attention within the
        // image spans onto BOTH mask types. The overlay needs a materialized
        // boolean mask, so `returnArray` is forced only when active; the
        // text-only / single-token decode hot path keeps `imageTokenMask ==
        // nil` and stays on the symbolic `.causal` mask.
        var maskByType = [String: MLXFast.ScaledDotProductAttentionMaskMode]()
        if !isCBv2 {
            let useBidirectionalVision =
                imageTokenMask != nil && config.useBidirectionalAttention == "vision"
                && h.dim(1) > 1
            for (i, layer) in layers.enumerated() {
                let lt = layer.layerType
                if maskByType[lt] == nil {
                    var mask: MLXFast.ScaledDotProductAttentionMaskMode
                    if lt == "sliding_attention" {
                        mask = createAttentionMask(
                            h: h, cache: fullCache[i], windowSize: config.slidingWindow,
                            returnArray: useBidirectionalVision)
                    } else {
                        mask = createAttentionMask(
                            h: h, cache: fullCache[i], windowSize: nil,
                            returnArray: useBidirectionalVision)
                    }
                    if useBidirectionalVision, let imageTokenMask {
                        mask = gemma4TextOverlayBidirectionalVision(mask, isVision: imageTokenMask)
                    }
                    maskByType[lt] = mask
                }
            }
        }

        // Forward through layers, tracking intermediate KV pairs for sharing
        var intermediates = [(kv: (MLXArray, MLXArray)?, positionOffset: Gemma4.PositionOffset?)](
            repeating: (nil, nil), count: config.numHiddenLayers)

        for (idx, layer) in layers.enumerated() {
            let prevIdx = previousKvs[idx]
            let sharedKV = intermediates[prevIdx].kv
            let sharedPositionOffset = intermediates[prevIdx].positionOffset

            // CBv2: KV-shared layers attend by borrowing the SOURCE layer's
            // cache object (attendBorrowing) instead of consuming raw K/V
            // tensors. Thread the source cache alongside the source's
            // captured (pre-update) position offsets.
            let v2SharedSource: (any CBv2AttendingLayerCache)? =
                isCBv2 && prevIdx != idx
                ? fullCache[prevIdx] as? (any CBv2AttendingLayerCache) : nil

            let mask = maskByType[layer.layerType]
            let (out, kvPair, positionOffset) = layer(
                h,
                mask: mask,
                cache: fullCache[idx],
                perLayerInput: perLayerInputs[idx],
                sharedKV: sharedKV,
                positionOffset: sharedPositionOffset,
                v2SharedSource: v2SharedSource
            )
            h = out
            intermediates[idx] = (kvPair, positionOffset)
            captureHook?(idx, kvPair)
        }

        let postNorm = norm(h)
        return (postNorm, capturePreNorm ? h : nil)
    }
}

// MARK: - Bidirectional vision attention overlay (mirror of the VLM twin)

/// Per-token block id for vision spans: each contiguous run of vision tokens
/// shares an id, non-vision tokens get -1. Exact mirror of
/// `gemma4VisionBlockIds` in Libraries/MLXVLM/Models/Gemma4.swift (Python
/// `_block_sequence_ids_for_mask`).
private func gemma4TextVisionBlockIds(_ isVision: MLXArray) -> MLXArray {
    let length = isVision.dim(1)
    let leading = MLXArray.zeros([isVision.dim(0), 1], dtype: .bool)
    let prev = concatenated([leading, isVision[0..., ..<(length - 1)]], axis: 1)
    let starts = logicalAnd(isVision, logicalNot(prev))
    let groupIds = cumsum(starts.asType(.int32), axis: 1) - 1
    return MLX.where(isVision, groupIds, MLXArray(Int32(-1)))
}

/// Overlay blockwise bidirectional attention for vision-token spans onto a
/// boolean causal mask (true = attend): tokens in the same image block
/// attend each other in BOTH directions. Exact mirror of
/// `gemma4BidirectionalVisionMask` (Python
/// `_apply_blockwise_bidirectional_overlay`).
private func gemma4TextBidirectionalVisionMask(
    _ baseMask: MLXArray, isVision: MLXArray
) -> MLXArray {
    let blockIds = gemma4TextVisionBlockIds(isVision)
    let qBlocks = expandedDimensions(blockIds, axis: -1)  // [B, L, 1]
    let kBlocks = expandedDimensions(blockIds, axis: -2)  // [B, 1, L]
    var sameBlock = logicalAnd(qBlocks .!= MLXArray(Int32(-1)), qBlocks .== kBlocks)  // [B, L, L]
    // Cached (chunked) prefill: `baseMask` covers ALL key columns
    // (`offset + L`) while `sameBlock` only describes the current window's
    // L columns. Left-pad with `false` so the overlay lands on the LAST L
    // key columns — cached keys stay causal. Callers must never split an
    // image block across the cache boundary (the CBv2 scheduler snaps
    // chunks to block edges; whole-prompt prefill has offset 0), or the
    // overlay could not see the cached half of the block (PR#63 review).
    let L = isVision.dim(1)
    let keyColumns = baseMask.dim(-1)
    if keyColumns > L {
        let pad = MLXArray.zeros([sameBlock.dim(0), L, keyColumns - L], dtype: .bool)
        sameBlock = concatenated([pad, sameBlock], axis: -1)  // [B, L, offset+L]
    }
    return logicalOr(baseMask, expandedDimensions(sameBlock, axis: 1))  // -> [B, 1, L, offset+L]
}

/// If `mode` carries a boolean array mask, overlay the vision bidirectional
/// attention; pass other modes (`.causal`, `.none`) through unchanged.
private func gemma4TextOverlayBidirectionalVision(
    _ mode: MLXFast.ScaledDotProductAttentionMaskMode, isVision: MLXArray
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mode {
    case .array(let maskArray):
        return .array(gemma4TextBidirectionalVisionMask(maskArray, isVision: isVision))
    default:
        return mode
    }
}

// MARK: - Public Model

public class Gemma4TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let config: Gemma4TextConfiguration
    let model: Gemma4TextModelInner

    /// Read-only accessor for the underlying text configuration.
    public var configuration: Gemma4TextConfiguration { config }

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self.model = Gemma4TextModelInner(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return applyLMHead(hidden)
    }

    /// Vision forward (mirror of the VLM wrapper's
    /// `languageModel(tokens, inputEmbedding:cache:imageTokenMask:)` call):
    /// `inputEmbedding` replaces the trunk's own embedding lookup (spliced
    /// image soft tokens; token ids still feed the PLE side inputs), and
    /// `imageTokenMask` ([B, L] bool) enables the blockwise bidirectional
    /// overlay on the LEGACY mask path (v2 layer caches own their masks and
    /// ignore it). Both nil ⇒ byte-identical to `callAsFunction(_:cache:)`.
    public func callAsFunction(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        imageTokenMask: MLXArray? = nil
    ) -> MLXArray {
        applyLMHead(
            model(
                inputs, cache: cache, inputEmbedding: inputEmbedding,
                imageTokenMask: imageTokenMask))
    }

    /// Apply the LM head (tied embedding or explicit `lm_head`) plus the
    /// configured final-logit softcap. Pure function of the post-norm hidden.
    func applyLMHead(_ hidden: MLXArray) -> MLXArray {
        var out: MLXArray
        if let lmHead {
            out = lmHead(hidden)
        } else {
            out = model.embedTokens.asLinear(hidden)
        }
        // Fused tanh-softcap (vMLX `compiledLogitSoftcap`). Untyped (float32) cap
        // keeps the softcap math + sampler logits full precision; when the compile
        // gate is off this is equivalent to `tanh(out / cap) * cap`.
        out = gemma4CompiledLogitSoftcap(out, MLXArray(config.finalLogitSoftcapping))
        return out
    }

    /// Parse the layer index out of a weight key like
    /// `"model.layers.15.self_attn.k_proj.weight"`. Returns nil if the key
    /// doesn't match the expected `...layers.<N>...` pattern.
    private func extractLayerIdx(from key: String) -> Int? {
        guard let layersRange = key.range(of: "layers.") else { return nil }
        let after = key[layersRange.upperBound...]
        let end = after.firstIndex(of: ".") ?? after.endIndex
        return Int(after[..<end])
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (k, v) in weights {
            // Skip vision/audio/rotary/quantization-range weights.
            if k.contains("self_attn.rotary_emb")
                || k.contains("input_max")
                || k.contains("input_min")
                || k.contains("output_max")
                || k.contains("output_min")
            {
                continue
            }

            // Skip k_proj/v_proj/k_norm/v_norm weights for layers that
            // borrow K/V from an earlier non-shared layer (num_kv_shared_layers
            // tail). Our `Gemma4Attention.init` doesn't allocate these modules
            // for shared-KV layers, so the checkpoint's copies would fail the
            // strict `update(parameters:verify:.all)` check.
            if let layerIdx = extractLayerIdx(from: k),
                config.layerUsesSharedKV(layerIdx: layerIdx),
                k.contains(".self_attn.k_proj.")
                    || k.contains(".self_attn.v_proj.")
                    || k.contains(".self_attn.k_norm.")
                    || k.contains(".self_attn.v_norm.")
            {
                continue
            }

            // 26B-A4B checkpoints ship the experts as a fused
            // `gate_up_proj` (concatenated along axis -2) plus a separate
            // `down_proj`. SwitchGLU expects three separate
            // `switch_glu.{gate,up,down}_proj.weight` tensors.
            if k.hasSuffix(".experts.gate_up_proj") {
                let base = String(k.dropLast(".gate_up_proj".count))
                let parts = MLX.split(v, parts: 2, axis: -2)
                sanitized["\(base).switch_glu.gate_proj.weight"] = parts[0]
                sanitized["\(base).switch_glu.up_proj.weight"] = parts[1]
                continue
            }

            if k.hasSuffix(".experts.down_proj") {
                let base = String(k.dropLast(".down_proj".count))
                sanitized["\(base).switch_glu.down_proj.weight"] = v
                continue
            }

            sanitized[k] = v
        }
        return sanitized
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let firstKvShared = config.numHiddenLayers - config.numKvSharedLayers

        var caches = [any KVCache]()
        for i in 0 ..< firstKvShared {
            if config.layerTypes[i] == "full_attention" {
                caches.append(StandardKVCache())
            } else {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }
}

// MARK: - LoRA

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.map { $0.selfAttn }
    }
}

// MARK: - ContinuousBatchingV2

extension Gemma4TextConfiguration {
    /// Per-layer attention structure for the CBv2 engine, derived purely
    /// from this configuration (invariant 11: model structure is data).
    /// Matches `Gemma4Attention.init` / `Gemma4TextModelInner.previousKvs`
    /// layer for layer.
    public var cbv2LayerKinds: [CBv2LayerKind] {
        CBv2LayerKindDerivation.gemma4LayerKinds(
            layerTypes: layerTypes,
            slidingWindow: slidingWindow,
            numKvSharedLayers: numKvSharedLayers,
            headDim: headDim,
            globalHeadDim: globalHeadDim,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            numGlobalKeyValueHeads: numGlobalKeyValueHeads,
            attentionKeqV: attentionKeqV
        )
    }
}

extension Gemma4TextModel {
    /// Per-layer CBv2 attention structure for this model (one entry per
    /// hidden layer, including the trailing KV-shared block).
    public var cbv2LayerKinds: [CBv2LayerKind] {
        config.cbv2LayerKinds
    }

    /// Build the per-layer CBv2 attending caches for this model: one
    /// `CBv2AttendingLayerCache` per hidden layer (KV-shared layers get a
    /// cache object too — it owns no storage and serves `attendBorrowing`).
    ///
    /// The concrete layer-cache classes are owned by the CBv2 core runtime;
    /// `makeLayerCache` is the injection point (typically wrapping a
    /// `CBv2KVBackend`). This model file codes purely against the contract.
    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { index, kind in
            try makeLayerCache(index, kind)
        }
    }
}

// MARK: - ContinuousBatchingV2 multimodal (vision prefill)

/// The CBv2 engine's embedding-spliced prefill surface
/// (`CBv2SteppableLanguageModelAdapter` forwards through this). The v2
/// attention branch is reached exactly as for token forwards — the layer
/// caches detected in `cache` own attention AND masking (the engine binds
/// the span-mask context on them) — only the embedding source differs.
/// Positions, KV sharing, and dual RoPE are untouched.
extension Gemma4TextModel: CBv2EmbeddingForwardable {

    /// Only configs whose weights were trained with the bidirectional
    /// image-span attention may serve CBv2 vision spans — the same gate the
    /// legacy `imageTokenMask` path applies. Text-only Gemma4 configs
    /// (nil / non-`"vision"`) reject multimodal requests at submit instead
    /// of silently serving logits under masks the weights never saw
    /// (PR#63 review).
    public var supportsVisionSpanPrefill: Bool {
        config.useBidirectionalAttention == "vision"
    }

    /// `embed(tokens) * embedScale` — exactly the trunk's pre-layer-0 hidden
    /// state, the tensor the engine splices image embeddings into (the
    /// VLM wrapper's `prepare` computes the same product before
    /// `maskedScatter`).
    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        model.embedTokens(inputs) * model.embedScale
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        applyLMHead(model(inputs, cache: cache, inputEmbedding: inputEmbedding))
    }
}
