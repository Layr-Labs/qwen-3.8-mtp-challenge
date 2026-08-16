//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    // MTP — number of Multi-Token Prediction head layers.
    // Port of omlx commit 696d90a: patches/mlx_lm_mtp/qwen35_model.py
    // `_patch_text_model_args` attaches this from config.json at runtime.
    var mtpNumHiddenLayers: Int = 0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.mtpNumHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }
}

// MARK: - GatedDelta verify helpers

/// Fuse the ordinary elementwise primitives that form the recurrence's fp32
/// `g` and `beta` inputs. The third input is the per-layer memoized
/// `-exp(A_log)`, so the promoted compiled prologue and the input-independent
/// gate memo stack instead of replacing one another. Shapeless compilation
/// shares one trace across verify widths.
private let qwen35CompiledGatedDeltaGBeta:
    @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray) =
{
    let body: @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray
    ) = { a, b, negExpALog, dtBias in
        let g = exp(negExpALog * softplus(a + dtBias))
        let beta = sigmoid(b).asType(.float32)
        return (g, beta)
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Run the existing recurrence kernel from already-computed fp32 `g`/`beta`.
/// The official M5 path uses this after the compiled helper; callers retain the
/// original `gatedDeltaUpdate` fallback when compiled decode is disabled.
private func qwen35GatedDeltaPrepared(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray?,
    mask: MLXArray?
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    var preparedState = state
        ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if preparedState.dtype != .float32 {
        preparedState = preparedState.asType(.float32)
    }
    return gatedDeltaKernel(
        q: q, k: k, v: v, g: g, beta: beta,
        state: preparedState, mask: mask)
}

/// Fuse the precise fp32 SiLU gate and product after the existing RMS norm.
/// The explicit casts mirror `Qwen3NextRMSNormGated` and keep the RMS reduction
/// itself on the unchanged MLXFast path.
private let qwen35CompiledGatedDeltaPostNorm:
    @Sendable (MLXArray, MLXArray) -> MLXArray =
{
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { x, gate in
        let gate32 = gate.asType(.float32)
        let activated = gate32 * sigmoid(gate32)
        return (activated * x.asType(.float32)).asType(x.dtype)
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Fuse SiLU activation and elementwise product in SwiGLU: `silu(g) * u`.
private let qwen35CompiledSwiGLU:
    @Sendable (MLXArray, MLXArray) -> MLXArray =
{
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { g, u in
        silu(g) * u
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Fuse Sigmoid gate and product in Attention output: `x * sigmoid(gate)`.
private let qwen35CompiledSigmoidMultiply:
    @Sendable (MLXArray, MLXArray) -> MLXArray =
{
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { x, gate in
        x * sigmoid(gate)
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

// MARK: - GatedDelta kernel with mid-state checkpoint

/// Clone of the vendored `gated_delta_step` kernel (GatedDelta.swift) with a
/// THIRD output: the recurrent state after timestep 0. For the width-2 MTP
/// verify this makes the post-primary rollback checkpoint a free by-product
/// of the single recurrence launch, instead of paying a second launch plus a
/// full fp32 state round-trip through device memory (3.15 MB/layer/launch,
/// x48 layers). Bit-exact vs two T=1 launches: the state lives in fp32
/// registers across the in-kernel t-loop and the eliminated round-trip was a
/// lossless f32 store/reload. Body is textually the vendored kernel plus the
/// `m_state` pointer and the `t == 0` store; no new template parameters and
/// no unused constexpr (M5 gen-17 JIT builds are strict about that).
private let qwen35GatedDeltaMidKernel: MLXFast.MLXFastKernel? = {
    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]; state_mid: [B, T-1, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (true) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_[hv_idx];
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              } else {
                y[dv_idx] = static_cast<InT>(0);
              }
              if (t < T - 1) {
                auto m_state = state_mid
                    + (((b_idx * (T - 1) + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  m_state[s_idx] = static_cast<StT>(state[i]);
                }
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """
    return MLXFast.metalKernel(
        name: "qwen35_gated_delta_step_mid",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out", "state_mid"],
        source: source
    )
}()

// MARK: - GatedDeltaNet

final class Qwen35GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    // Decode-width fused input projections: ONE affine-4 matmul over the
    // concatenated [qkv | z | b | a] output rows (N = 10240+6144+48+48 =
    // 16480) instead of four separate launches per layer per forward. The
    // two 48-wide launches are pure overhead at M<=2. Bit-exact per output
    // row for the same reason as the FA QKV fuse: groups run along K, every
    // N here and the fused N keep qmv_fast eligibility (N%8==0, K%512==0),
    // and a row's reduction order does not depend on the launch it rides in.
    // Prefill (S > 2) keeps the four separate calls so qmm/split-k reduction
    // orders match the pinned baseline byte-for-byte. Underscore storage so
    // none of this is a Module parameter.
    private var _inW: MLXArray?
    private var _inS: MLXArray?
    private var _inZ: MLXArray?
    private var _inGS = 64
    private var _inBits = 4
    private var _inMode = QuantizationMode.affine

    // Input-independent per-layer memo of `-exp(A_log)` in fp32 — the only
    // input-independent factor of the decay gate `g = exp(-exp(A_log) *
    // softplus(a + dt_bias))`. `computeGatedDeltaG` rebuilt it (astype, exp,
    // negate — three launches per layer per recurrence call) every round;
    // the multiply and outer exp that consume it are unchanged, and
    // `-exp(x) * s` was already `negative(exp(x)) * s`, so g is
    // arithmetically identical. Pure weight-derived cache, the allowed
    // input-independent kind.
    private var _negExpALog: MLXArray?
    private var negExpALog: MLXArray {
        if let cached = _negExpALog { return cached }
        let value = -exp(aLog.asType(.float32))
        _negExpALog = value
        return value
    }

    /// `gatedDeltaUpdate` with the g-gate's input-independent factor served
    /// from `negExpALog`. The prologue replicates the vendored function's
    /// arithmetic exactly (fp32 beta and g, fp32 state coercion) and the
    /// dispatch reuses the same internal kernel/ops pair. Guarded on the
    /// same kernel-availability condition class as the S == 2 mid-kernel:
    /// when custom kernels are unavailable we fall back to the untouched
    /// vendored `gatedDeltaUpdate`, never to a mismatched dispatch.
    ///
    /// Apply the weight-derived memo uniformly at every sequence width,
    /// including the depth-0 serial-control path. The optimization is
    /// input-independent and arithmetically identical for both benchmark
    /// legs; the kernel-availability guard below is the only dispatch gate.
    private func gatedDeltaUpdateMemoG(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        a: MLXArray,
        b: MLXArray,
        state: MLXArray?,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        guard qwen35GatedDeltaMidKernel != nil else {
            return gatedDeltaUpdate(
                q: q, k: k, v: v, a: a, b: b,
                aLog: aLog, dtBias: dtBias, state: state, mask: mask)
        }
        let beta = sigmoid(b).asType(.float32)
        let g = exp(negExpALog * softplus(a + dtBias))
        let B = q.dim(0)
        let Dk = q.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
        if state.dtype != .float32 {
            state = state.asType(.float32)
        }
        return gatedDeltaKernel(
            q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    init(_ args: Qwen35TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    /// Lazily build and apply the fused [qkv|z|b|a] projection. Returns nil
    /// until every input projection is a matching affine `QuantizedLinear`
    /// (bf16 trees fall back to the four separate calls).
    private func fusedInProjections(
        _ x: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray)? {
        if let w = _inW, let s = _inS, let zp = _inZ {
            let y = quantizedMM(
                x, w, scales: s, biases: zp, transpose: true,
                groupSize: _inGS, bits: _inBits, mode: _inMode)
            let qkvEnd = keyDim * 2 + valueDim
            let zEnd = qkvEnd + valueDim
            let bEnd = zEnd + numVHeads
            return (
                y[.ellipsis, ..<qkvEnd],
                y[.ellipsis, qkvEnd ..< zEnd],
                y[.ellipsis, zEnd ..< bEnd],
                y[.ellipsis, bEnd...]
            )
        }
        guard let q = inProjQKV as? QuantizedLinear,
              let z = inProjZ as? QuantizedLinear,
              let b = inProjB as? QuantizedLinear,
              let a = inProjA as? QuantizedLinear,
              q.groupSize == z.groupSize, z.groupSize == b.groupSize,
              b.groupSize == a.groupSize,
              q.bits == z.bits, z.bits == b.bits, b.bits == a.bits,
              q.mode == z.mode, z.mode == b.mode, b.mode == a.mode,
              q.mode == .affine,
              let qz = q.biases, let zz = z.biases, let bz = b.biases,
              let az = a.biases
        else { return nil }
        _inW = concatenated([q.weight, z.weight, b.weight, a.weight], axis: 0)
            .contiguous()
        _inS = concatenated([q.scales, z.scales, b.scales, a.scales], axis: 0)
            .contiguous()
        _inZ = concatenated([qz, zz, bz, az], axis: 0).contiguous()
        _inGS = q.groupSize
        _inBits = q.bits
        _inMode = q.mode
        return fusedInProjections(x)
    }

    // MARK: - _processChunk (MTP helper)

    /// Process one time-chunk of the linear-attention layer.
    ///
    /// Extracted from `callAsFunction` so the MTP verify cycle can run the prefix
    /// (n_confirmed tokens) and draft suffix separately, snapshotting the SSM/conv
    /// state in between for rollback on draft rejection.
    ///
    /// Port of omlx commit 696d90a:
    ///   patches/mlx_lm_mtp/qwen35_model.py `GatedDeltaNet._process_chunk`
    ///
    /// - Parameters:
    ///   - qkv: Already-masked QKV for this chunk [B, S_chunk, conv_dim]
    ///   - a, b: Input projections for this chunk [B, S_chunk, ...]
    ///   - convState: Initial conv state [B, conv_kernel_size-1, conv_dim]
    ///   - ssmState: Initial SSM state (nil on first token)
    ///   - mask: SSM mask for `gatedDeltaUpdate` (optional)
    /// - Returns: `(out, newConvState, newSsmState)`
    private func processChunk(
        qkv: MLXArray,
        a: MLXArray,
        b: MLXArray,
        convState: MLXArray,
        ssmState: MLXArray?,
        mask: MLXArray?
    ) -> (out: MLXArray, newConvState: MLXArray, newSsmState: MLXArray) {
        let B = qkv.dim(0)
        let S = qkv.dim(1)

        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = convInput[0..., (convInput.dim(1) - nKeep)...]
        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, newSsmState) = gatedDeltaUpdateMemoG(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            state: ssmState,
            mask: mask
        )
        return (out, newConvState, newSsmState)
    }

    /// Single-chunk verify twin that retains only the ingredients needed to
    /// reconstruct a committed recurrent prefix after the target acceptance
    /// walk. The target output is the ordinary `gatedDeltaUpdate` output; no
    /// midpoint state tensor is produced on the hot path.
    private func processChunkStashingPrefix(
        qkv: MLXArray,
        a: MLXArray,
        b: MLXArray,
        convState: MLXArray,
        ssmState: MLXArray?,
        mask: MLXArray?
    ) -> (
        out: MLXArray, newConvState: MLXArray, newSsmState: MLXArray,
        tape: ArraysCache.PrefixReplayTape
    ) {
        let B = qkv.dim(0)
        let S = qkv.dim(1)
        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = convInput[0..., (convInput.dim(1) - nKeep)...]
        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(
            convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        // Keep the recurrence and conv prologue wide. The promoted compiled
        // g/beta launch reduction feeds the same single recurrence, while the
        // SDPA helper alone bridges the width-sensitive kernel boundary.
        let (g, beta) = qwen35CompiledGatedDeltaGBeta(
            a, b, negExpALog, dtBias)
        let recurrence: (MLXArray, MLXArray)
        if MLXHardwareInfo.isCompiledDecodeSupported {
            recurrence = qwen35GatedDeltaPrepared(
                q: qNormed, k: kNormed, v: v,
                g: g, beta: beta, state: ssmState, mask: mask)
        } else {
            recurrence = gatedDeltaUpdate(
                q: qNormed, k: kNormed, v: v, a: a, b: b,
                aLog: aLog, dtBias: dtBias, state: ssmState, mask: mask)
        }
        let out = recurrence.0
        let newSsmState = recurrence.1
        let tape = ArraysCache.PrefixReplayTape(
            convInput: convInput,
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            g: g,
            beta: beta,
            ssmPre: ssmState.map { $0[.ellipsis] },
            mask: mask.map { $0[.ellipsis] },
            rowCount: S,
            convStateRows: nKeep)
        return (out, newConvState, newSsmState, tape)
    }

    fileprivate func canReplayPrefix(
        cache: MambaCache, committedRows: Int
    ) -> Bool {
        guard let tape = cache.prefixReplayTape,
              committedRows > 0,
              committedRows < tape.rowCount,
              tape.convStateRows == convKernelSize - 1,
              tape.convInput.dim(1)
                  >= committedRows + tape.convStateRows,
              tape.q.dim(1) == tape.rowCount,
              tape.k.dim(1) == tape.rowCount,
              tape.v.dim(1) == tape.rowCount,
              tape.a.dim(1) == tape.rowCount,
              tape.b.dim(1) == tape.rowCount,
              tape.g.dim(1) == tape.rowCount,
              tape.beta.dim(1) == tape.rowCount
        else { return false }
        return true
    }

    /// Reconstruct the fp32 recurrent state after `committedRows` verify rows
    /// from the exact pre-verify state and transformed recurrence inputs.
    /// Call only after every GDN layer has passed `canReplayPrefix`.
    fileprivate func replayPrefix(
        cache: MambaCache, committedRows: Int
    ) -> Bool {
        guard canReplayPrefix(cache: cache, committedRows: committedRows),
              let tape = cache.prefixReplayTape
        else { return false }
        let rows = 0 ..< committedRows
        let recurrence: (MLXArray, MLXArray)
        if MLXHardwareInfo.isCompiledDecodeSupported {
            recurrence = qwen35GatedDeltaPrepared(
                q: tape.q[0..., rows, 0...],
                k: tape.k[0..., rows, 0...],
                v: tape.v[0..., rows, 0...],
                g: tape.g[0..., rows],
                beta: tape.beta[0..., rows],
                state: tape.ssmPre,
                mask: tape.mask.map { $0[0..., rows] })
        } else {
            recurrence = gatedDeltaUpdate(
                q: tape.q[0..., rows, 0...],
                k: tape.k[0..., rows, 0...],
                v: tape.v[0..., rows, 0...],
                a: tape.a[0..., rows, 0...],
                b: tape.b[0..., rows, 0...],
                aLog: aLog,
                dtBias: dtBias,
                state: tape.ssmPre,
                mask: tape.mask.map { $0[0..., rows] })
        }
        let boundarySsm = recurrence.1
        cache[0] = tape.convInput[
            0...,
            committedRows ..< (committedRows + tape.convStateRows),
            0...]
        cache[1] = boundarySsm
        cache.prefixReplayTape = nil
        cache.rollbackState = nil
        cache.rollbackCheckpoints = []
        return true
    }

    // MARK: - callAsFunction

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil,
        nConfirmed: Int = 0
    ) -> MLXArray {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py GatedDeltaNet.__call__
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        var qkv: MLXArray
        let z: MLXArray
        let b: MLXArray
        let a: MLXArray
        if S <= 9, let fused = fusedInProjections(inputs) {
            qkv = fused.0
            z = fused.1.reshaped(B, S, numVHeads, headVDim)
            b = fused.2
            a = fused.3
        } else {
            qkv = inProjQKV(inputs)
            z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
            b = inProjB(inputs)
            a = inProjA(inputs)
        }

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        // Apply mask to full qkv before any chunking.
        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let ssmState = cache?[1]
        let out: MLXArray
        let finalConvState: MLXArray
        let finalSsmState: MLXArray
        var pendingPrefixTape: ArraysCache.PrefixReplayTape?

        if nConfirmed == 1 && S >= 3 && mask == nil {
            // K>=2 verify: keep the ordinary single-chunk recurrence and a
            // compact replay tape. This avoids writing one 3 MiB fp32 state per
            // boundary per GDN layer on every round. K=1 deliberately remains
            // on the already-promoted eager-checkpoint kernel below.
            let (o, c, s, tape) = processChunkStashingPrefix(
                qkv: qkv, a: a, b: b,
                convState: convState, ssmState: ssmState, mask: mask)
            out = o
            finalConvState = c
            finalSsmState = s
            pendingPrefixTape = tape
        } else if nConfirmed == 1 && S == 2 && mask == nil,
           let midKernel = qwen35GatedDeltaMidKernel
        {
            // Width-2 MTP verify, single-launch form. The old split path ran
            // EVERY satellite op twice (conv, silu, split, reshapes, q/k norms,
            // sigmoid, g) and paid two recurrence launches with a full fp32
            // state round-trip between them, solely to observe the
            // post-primary state. Here the prework runs once over both rows —
            // all of it position-local, so per-row bit-identical to the split
            // form — and the cloned kernel emits the timestep-0 state as a
            // third output, so the rollback checkpoint is free.
            let convInput = concatenated([convState, qkv], axis: 1)
            let nKeep = convKernelSize - 1
            let newConvState = convInput[0..., (convInput.dim(1) - nKeep)...]
            let convOut = silu(conv1d(convInput))

            let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
            let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
            let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

            let dtype = q.dtype
            let invScale = pow(Float(headKDim), -0.5)
            let qNormed =
                MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
            let kNormed =
                MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

            // Replicates gatedDeltaUpdate's fp32 prologue, fusing beta/g while
            // serving the gate's input-independent factor from the layer memo.
            let (g, beta) = qwen35CompiledGatedDeltaGBeta(
                a, b, negExpALog, dtBias)
            var state = ssmState
                ?? MLXArray.zeros(
                    [B, numVHeads, headVDim, headKDim], dtype: .float32)
            if state.dtype != .float32 { state = state.asType(.float32) }

            let outputs = midKernel(
                [qNormed, kNormed, v, g, beta, state, MLXArray(S)],
                template: [
                    ("InT", dtype),
                    ("StT", DType.float32),
                    ("Dk", headKDim),
                    ("Dv", headVDim),
                    ("Hk", numKHeads),
                    ("Hv", numVHeads),
                ],
                grid: (32, headVDim, B * numVHeads),
                threadGroup: (32, 4, 1),
                outputShapes: [
                    [B, S, numVHeads, headVDim],
                    state.shape,
                    [B, S - 1, numVHeads, headVDim, headKDim],
                ],
                outputDTypes: [dtype, .float32, .float32]
            )
            // Per-boundary checkpoints: after row t, the conv state is rows
            // (t+1)..(t+nKeep) of [convState | x0 .. x_{S-1}] and the SSM
            // state is the kernel's mid output slice t. Checkpoint 0 doubles
            // as the legacy single-slot `rollbackState` for the K=1 path.
            var checkpoints: [(MLXArray, MLXArray)] = []
            checkpoints.reserveCapacity(S - 1)
            for t in 0 ..< (S - 1) {
                checkpoints.append((
                    convInput[0..., (t + 1) ..< (t + 1 + nKeep)],
                    outputs[2][0..., t]
                ))
            }
            cache?.rollbackState = checkpoints.first
            cache?.rollbackCheckpoints = checkpoints
            out = outputs[0]
            finalConvState = newConvState
            finalSsmState = outputs[1]
        } else if nConfirmed > 0 && nConfirmed < S {
            // Split at nConfirmed boundary for the MTP 2-token verify forward.
            // Run confirmed prefix first, snapshot rollback state, then run draft.
            // omlx: GatedDeltaNet.__call__ nConfirmed > 0 branch
            let maskC = mask.map { $0[0..., 0..<nConfirmed] }
            let maskD = mask.map { $0[0..., nConfirmed...] }

            let (outC, convC, ssmC) = processChunk(
                qkv: qkv[0..., 0..<nConfirmed, 0...],
                a: a[0..., 0..<nConfirmed, 0...],
                b: b[0..., 0..<nConfirmed, 0...],
                convState: convState,
                ssmState: ssmState,
                mask: maskC
            )
            // Snapshot (conv_state, ssm_state) after confirmed prefix for rollback.
            // omlx: cache.rollback_state = (conv_c, ssm_c)
            cache?.rollbackState = (convC, ssmC)

            let (outD, convF, ssmF) = processChunk(
                qkv: qkv[0..., nConfirmed..., 0...],
                a: a[0..., nConfirmed..., 0...],
                b: b[0..., nConfirmed..., 0...],
                convState: convC,
                ssmState: ssmC,
                mask: maskD
            )
            out = concatenated([outC, outD], axis: 1)
            finalConvState = convF
            finalSsmState = ssmF
        } else {
            // Standard single-chunk path (nConfirmed == 0 or S == 1).
            let (o, c, s) = processChunk(
                qkv: qkv, a: a, b: b,
                convState: convState,
                ssmState: ssmState,
                mask: mask
            )
            out = o
            finalConvState = c
            finalSsmState = s
        }

        if let cache {
            cache[0] = finalConvState
            cache[1] = finalSsmState
            // A forward that did not request replay must erase any prior tape;
            // otherwise a later partial miss could restore a stale frame.
            cache.prefixReplayTape = pendingPrefixTape
            if pendingPrefixTape != nil {
                cache.rollbackState = nil
                cache.rollbackCheckpoints = []
            }
        }

        let normedOut: MLXArray
        if nConfirmed == 1 && S >= 2 {
            let rmsOut = MLXFast.rmsNorm(out, weight: norm.weight, eps: norm.eps)
            normedOut = qwen35CompiledGatedDeltaPostNorm(rmsOut, z)
        } else {
            normedOut = norm(out, gate: z)
        }
        return outProj(normedOut.reshaped(B, S, -1))
    }
}

// MARK: - Fused MLP

/// Decode-width fused gate/up MLP. Same math as the vendored `Qwen3NextMLP` —
/// `down(silu(gate(x)) * up(x))` — but at decode/verify widths (S <= 9) the
/// gate and up projections run as ONE matmul over concatenated output rows
/// (N = 34816), halving the biggest per-layer launch count on the hot path.
///
/// Row-concat on N is bit-exact per output element: affine groups run along
/// K, every N here (17408 and 34816) keeps the fast-kernel eligibility
/// (N%8==0 at K%512==0), and a row's dot-product arithmetic in `qmv_fast`
/// does not depend on which launch it rides in — the in-tree precedent is
/// the promoted FA QKV fuse below. The guard covers exactly the widths the
/// host still serves through the per-row-exact QMV dispatch (crossrow for
/// M <= 5, per-row qmv_fast above it; the qmv batch limit is 10+ for these
/// shapes on this generation — the same coverage the S <= 9 GDN in-proj
/// fuse gate relies on); prefill (S > 9) keeps the two separate calls so
/// the qmm/split-k reduction order of the pinned baseline is preserved
/// byte-for-byte. The bf16 `Linear` variant (the MTP head's MLP)
/// fuses the plain weights the same way; the head only proposes, so that
/// side carries no exactness constraint at all.
final class Qwen35FusedMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    private var _fqW: MLXArray?
    private var _fqS: MLXArray?
    private var _fqZ: MLXArray?
    private var _fqGS = 64
    private var _fqBits = 4
    private var _fqMode = QuantizationMode.affine
    private var _fbfW: MLXArray?
    private var _gateOut = 0

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    private func fusedGateUp(_ x: MLXArray) -> (MLXArray, MLXArray)? {
        if let w = _fqW, let s = _fqS, let z = _fqZ {
            let y = quantizedMM(
                x, w, scales: s, biases: z, transpose: true,
                groupSize: _fqGS, bits: _fqBits, mode: _fqMode)
            return (y[.ellipsis, ..<_gateOut], y[.ellipsis, _gateOut...])
        }
        if let w = _fbfW {
            let y = matmul(x, w.T)
            return (y[.ellipsis, ..<_gateOut], y[.ellipsis, _gateOut...])
        }
        if let g = gateProj as? QuantizedLinear,
           let u = upProj as? QuantizedLinear,
           g.groupSize == u.groupSize, g.bits == u.bits,
           g.mode == u.mode, g.mode == .affine,
           let gz = g.biases, let uz = u.biases
        {
            _fqW = concatenated([g.weight, u.weight], axis: 0).contiguous()
            _fqS = concatenated([g.scales, u.scales], axis: 0).contiguous()
            _fqZ = concatenated([gz, uz], axis: 0).contiguous()
            _fqGS = g.groupSize
            _fqBits = g.bits
            _fqMode = g.mode
            _gateOut = g.shape.0
            return fusedGateUp(x)
        }
        if !(gateProj is QuantizedLinear), !(upProj is QuantizedLinear) {
            _fbfW = concatenated([gateProj.weight, upProj.weight], axis: 0)
                .contiguous()
            _gateOut = gateProj.weight.dim(0)
            return fusedGateUp(x)
        }
        return nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if x.dim(-2) <= 9, let (g, u) = fusedGateUp(x) {
            return downProj(qwen35CompiledSwiGLU(g, u))
        }
        return downProj(qwen35CompiledSwiGLU(gateProj(x), upProj(x)))
    }

}

// MARK: - Attention

final class Qwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    // Packed Q/K/V concat on N. Underscore so it is not a Module parameter.
    // Built once from already-quantized q/k/v; never attached as a child.
    private var _qkvW: MLXArray?
    private var _qkvS: MLXArray?
    private var _qkvZ: MLXArray?
    private var _qkvGS = 64
    private var _qkvBits = 4
    private var _qkvMode = QuantizationMode.affine
    private var _qOut = 0
    private var _kOut = 0
    // Dense (bf16) Q/K/V pack for the MTP head layers, same concat-on-N
    // argument as the quantized pack above: rows are independent, so one
    // GEMM over concatenated weights is bit-exact with three separate
    // launches — and this is the proposal side regardless. Cuts two
    // launches and two host ops from every head chain step.
    private var _qkvDenseW: MLXArray?

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    /// One affine-4 GEMM for Q+gate, K, and V. Rows are independent, so
    /// concatenating already-packed weights on N is bit-exact with three
    /// separate qmv_fast launches. Unquantized (MTP bf16) falls back.
    private func qkv(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        if let w = _qkvW, let s = _qkvS, let z = _qkvZ {
            let y = quantizedMM(
                x, w, scales: s, biases: z, transpose: true,
                groupSize: _qkvGS, bits: _qkvBits, mode: _qkvMode)
            let qEnd = _qOut
            let kEnd = _qOut + _kOut
            return (y[.ellipsis, ..<qEnd], y[.ellipsis, qEnd ..< kEnd], y[.ellipsis, kEnd...])
        }
        if let w = _qkvDenseW {
            let y = matmul(x, w.transposed(1, 0))
            let qEnd = _qOut
            let kEnd = _qOut + _kOut
            return (y[.ellipsis, ..<qEnd], y[.ellipsis, qEnd ..< kEnd], y[.ellipsis, kEnd...])
        }
        if let q = qProj as? QuantizedLinear,
           let k = kProj as? QuantizedLinear,
           let v = vProj as? QuantizedLinear,
           q.groupSize == k.groupSize, k.groupSize == v.groupSize,
           q.bits == k.bits, k.bits == v.bits,
           q.mode == k.mode, q.mode == .affine,
           let qz = q.biases, let kz = k.biases, let vz = v.biases
        {
            _qkvW = concatenated([q.weight, k.weight, v.weight], axis: 0).contiguous()
            _qkvS = concatenated([q.scales, k.scales, v.scales], axis: 0).contiguous()
            _qkvZ = concatenated([qz, kz, vz], axis: 0).contiguous()
            _qkvGS = q.groupSize
            _qkvBits = q.bits
            _qkvMode = q.mode
            _qOut = q.shape.0
            _kOut = k.shape.0
            return qkv(x)
        }
        if !(qProj is QuantizedLinear), !(kProj is QuantizedLinear),
           !(vProj is QuantizedLinear),
           qProj.bias == nil, kProj.bias == nil, vProj.bias == nil
        {
            _qkvDenseW = concatenated(
                [qProj.weight, kProj.weight, vProj.weight], axis: 0
            ).contiguous()
            _qOut = qProj.weight.dim(0)
            _kOut = kProj.weight.dim(0)
            return qkv(x)
        }
        return (qProj(x), kProj(x), vProj(x))
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let (qProjOutput, keysIn, valuesIn) = qkv(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = keysIn
        var values = valuesIn

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(qwen35CompiledSigmoidMultiply(output, gate))
    }
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined + sharedY
    }
}

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen35FusedMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
        nConfirmed: Int = 0
    ) -> MLXArray {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py DecoderLayer.__call__
        // Passes nConfirmed through to the linear-attention sublayer.
        let r: MLXArray
        if isLinear {
            r = linearAttn!(
                inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                nConfirmed: nConfirmed)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

// MARK: - Text Model

public class Qwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    /// Returns the pre-norm hidden state from the final layer.
    ///
    /// The caller (`Qwen35TextModel`) applies `norm` and the LM head on top.
    /// This split lets `callWithHidden` return both pre-norm hidden (for the MTP head)
    /// and the normalised logits in one forward pass.
    ///
    /// Port of omlx commit 696d90a:
    ///   patches/mlx_lm_mtp/qwen35_model.py `_patch_qwen3_5_text_model`
    ///   (returns hidden_states before self.model.norm so TextModel can apply it)
    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache?]? = nil,
        nConfirmed: Int = 0
    ) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        // Decode-width asyncEval ladder: at S <= 2 (serial step / width-2 MTP
        // verify) the host builds a ~64-layer graph before anything reaches
        // the GPU. Firing asyncEval at a few layer boundaries lets the GPU
        // start on the early layers while the host is still building the
        // rest. Pure enqueue-timing change — no op is added, no reduction
        // order moves, so the emitted stream is bit-identical (Laguna receipt
        // for the same schedule shape: off 10.37 ms vs ladder 9.45 ms/step;
        // schedule scaled from 40 to 64 layers, front rungs kept).
        let ladderActive = inputs.dim(1) <= 9
        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask,
                cache: cacheArray?[i], nConfirmed: nConfirmed)
            if ladderActive {
                switch i {
                case 0, 1, 9, 19, 29, 39, 49, 57:
                    asyncEval(hiddenStates)
                default:
                    break
                }
            }
        }

        // Return pre-norm hidden states. Norm is applied by Qwen35TextModel.
        return hiddenStates
    }

    /// Atomically rebuild every linear-attention layer at the same committed
    /// verify prefix. The first pass is read-only; a failure leaves the entire
    /// mixed recurrent/attention cache available to the session's generic
    /// snapshot-and-repair fallback.
    func replayRecurrentPrefix(
        cache: [KVCache?], committedRows: Int
    ) -> Bool {
        guard cache.count == layers.count, committedRows > 0 else {
            return false
        }
        for (i, layer) in layers.enumerated() where layer.isLinear {
            guard let mamba = cache[i] as? MambaCache,
                  let linear = layer.linearAttn,
                  linear.canReplayPrefix(
                    cache: mamba, committedRows: committedRows)
            else { return false }
        }
        for (i, layer) in layers.enumerated() where layer.isLinear {
            guard let mamba = cache[i] as? MambaCache,
                  let linear = layer.linearAttn,
                  linear.replayPrefix(
                    cache: mamba, committedRows: committedRows)
            else { return false }
        }
        return true
    }
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    // Declared DRAFT-ONLY vocabulary projection, carried by the declared MTP
    // head tree as `draft_lm_head.{weight,scales,biases}` (merged under the
    // `mtp.` prefix and intercepted in `sanitize` below). A coarser affine
    // copy of the exact lm_head used exclusively to argmax DRAFT proposals —
    // the head side only proposes, so its numerics are competitive surface;
    // every ledger/verify value still comes from the exact `lmHead`. Plain
    // stored arrays, deliberately not Module parameters.
    private var _draftHeadW: MLXArray?
    private var _draftHeadS: MLXArray?
    private var _draftHeadZ: MLXArray?

    // Input-independent compact copy of the loaded exact lm_head, used only
    // for draft proposals when no declared draft_lm_head is present. It is not
    // ModuleInfo because it is derived during warmup, not checkpoint state.
    private var _compactDraftHead: Linear?
    // Prefix 98_304, the promoted trim. A 49_152 halving was measured on the
    // public longcopy gate and REGRESSED: three of its committed argmax ids
    // live in [49_152, 248_044), the head could no longer propose them, and
    // the forced rejects cost more round-bases than the halved compact-head
    // read saved (accept 1.00 -> 0.877, 21.1 -> 22.8 ms/token). The read is
    // ~315 MB of affine-4 rows per draft step (~0.6 ms), so the ceiling of
    // any further trim is small and the acceptance downside is not.
    private static let compactDraftPrefixCount = 98_304
    private static let compactDraftControlStart = 248_044
    private static let compactDraftControlEnd = 248_070
    private static let compactDraftRealCount =
        compactDraftPrefixCount + compactDraftControlEnd - compactDraftControlStart
    private static let compactDraftPaddedCount = 98_336

    /// MTP head. Non-nil only when `_qwen35MTPEnabled == true` at init time
    /// AND `args.mtpNumHiddenLayers > 0`.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.__init__ (MTPModule attachment)
    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPModule?

    public init(_ args: Qwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }

        // Attach MTP head only when enabled and config declares MTP layers.
        // omlx: `if n_mtp > 0 and is_mtp_active(): self.mtp = q35.MTPModule(args)`
        if args.mtpNumHiddenLayers > 0 && _qwen35MTPEnabled {
            _mtp.wrappedValue = Qwen35MTPModule(args)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Inner model now returns pre-norm hidden; apply norm + lm_head here.
        // omlx: TextModel.__call__ (normed = self.model.norm(hidden); out = lm_head(normed))
        let hidden = model(inputs, cache: cache)
        var out = model.norm(hidden)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py TextModel.sanitize
        //
        // Key differences from stock mlx-lm:
        //  1. Gate norm shift on unsanitized conv1d shape ONLY (not on MTP key presence).
        //     Stock code uses `hasMTPWeights || hasUnsanitizedConv1d`, which double-shifts
        //     already-converted MLX checkpoints that have mtp.* keys.
        //  2. Keep mtp.* keys when the MTP head is attached; strip them otherwise.
        //  3. Extend norm-shift key set with MTP-specific norm names.

        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasUnsanitizedConv1d  // NOT hasMTPWeights

        var weights = weights

        // Keep mtp.* keys if the head is attached; strip them otherwise.
        // omlx: `if not hasattr(self, "mtp"): weights = {k:v if "mtp." not in k}`
        if mtp == nil {
            weights = weights.filter { !$0.key.contains("mtp.") }
        } else if let draftW = weights.removeValue(
            forKey: "mtp.draft_lm_head.weight")
        {
            // Declared draft-only lm_head: side-channel the triple out of the
            // parameter update (there is no Module parameter for it) into the
            // draft projection storage above.
            _draftHeadW = draftW
            _draftHeadS = weights.removeValue(forKey: "mtp.draft_lm_head.scales")
            _draftHeadZ = weights.removeValue(forKey: "mtp.draft_lm_head.biases")
        }
        if mtp != nil, !weights.keys.contains(where: { $0.contains("mtp.") }) {
            // MTP enabled but no mtp.* keys in checkpoint → needs re-conversion.
            // omlx: raises ValueError with "weights are missing the mtp.* tensors"
            print(
                "[WARNING] Qwen35TextModel.sanitize: MTP head is enabled but no mtp.* "
                + "weights found. Load will likely fail or produce garbage. "
                + "Re-convert the checkpoint with a converter that preserves MTP weights.")
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // Extended norm key set includes MTP-specific names.
        // omlx: norm_keys tuple with ".pre_fc_norm_hidden.weight" etc.
        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }
}

// MARK: - Qwen35TextModel + MTPCapable

extension Qwen35TextModel: MTPCapable {
    public var hasMTPHead: Bool { mtp != nil }

    /// Run a backbone forward that also returns pre-norm hidden states.
    ///
    /// Returns `(logits, preNormHidden)` where `preNormHidden` is the raw backbone output
    /// BEFORE `model.norm`. The MTP head applies its own `pre_fc_norm_hidden` normalization,
    /// so it expects un-normalized input. Passing post-norm would double-normalize.
    ///
    /// PR #990: `return out, hidden  # pre-norm hidden for MTP head`
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.__call__ with return_hidden=True
    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        let cacheOpt: [KVCache?] = cache.map { Optional($0) }
        let hidden = model(input.tokens, cache: cacheOpt, nConfirmed: nConfirmed)
        let normed = model.norm(hidden)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(normed)
        } else {
            logits = model.embedTokens.asLinear(normed)
        }
        // Return pre-norm hidden, not post-norm. The MTP module's pre_fc_norm_hidden
        // is the normalization step — it expects the raw backbone output as input.
        return (logits, hidden)
    }

    /// Rebuild the target's recurrent cache after an accepted verify prefix.
    public func replayRecurrentPrefix(
        cache: [any KVCache], committedRows: Int
    ) -> Bool {
        model.replayRecurrentPrefix(
            cache: cache.map { Optional($0) },
            committedRows: committedRows)
    }

    /// Apply the backbone's final `model.norm` to a hidden state.
    ///
    /// `callWithHidden` returns the PRE-norm hidden by design. MTPLX -- the exactness
    /// reference this track's accept/verify loop was validated against -- defaults to
    /// `base_hidden_variant == "post_norm"` (mtplx/mtp_patch.py:50, and
    /// `hidden = pre_norm if variant == "pre_norm" else post_norm` in
    /// `_MTPLXTextModel.__call__`), i.e. the hidden fed to the MTP head is the backbone
    /// output AFTER `model.norm`, even though the head then applies its own
    /// `pre_fc_norm_hidden` on top. This accessor lets a caller produce that variant
    /// without changing `callWithHidden`'s existing pre-norm contract.
    ///
    /// The variant is NOT a correctness knob and a wrong choice fails SILENTLY: a
    /// pre-norm draft chain still verifies exact (the target decides every emitted
    /// token), it just stops predicting -- acceptance collapses toward zero and the
    /// speculation buys nothing. Any validation of this path must therefore read the
    /// ACCEPTANCE RATE alongside the exactness verdict.
    ///
    /// `Qwen35TextModelInner.norm` is not visible outside this module, which is the
    /// only reason this accessor exists.
    public func applyFinalNorm(_ x: MLXArray) -> MLXArray {
        model.norm(x)
    }

    /// Run the MTP head forward, returning `(logits, headHidden)`.
    ///
    /// `headHidden` is the MTP head's own post-`mtp.norm` output, which is what MTPLX
    /// chains into the next draft level when `mtp_hidden_variant == "post_norm"` (its
    /// default): `h = hidden_level[:, -1:, :]` in `_make_device_draft_core.chain_fn`
    /// (mtplx/generation.py) with `post_norm = self.mtp.norm(x)` in `_mtp_core`
    /// (mtplx/mtp_patch.py). Required for multi-step (depth > 1) drafting: re-feeding
    /// the TRUNK hidden to every sub-step would draft every level from the same state.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.mtp_forward(return_hidden=True)
    public func mtpForwardWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> (MLXArray, MLXArray) {
        guard let mtp else {
            fatalError("mtpForwardWithHidden called but MTP head is not attached. "
                + "Set _qwen35MTPEnabled = true before loading the model.")
        }
        let mtpOut = mtp(
            hidden: hidden,
            nextTokenIds: nextTokenIds,
            embedTokens: model.embedTokens,
            cache: cache)
        let logits: MLXArray
        if configuration.tieWordEmbeddings {
            logits = model.embedTokens.asLinear(mtpOut)
        } else {
            logits = lmHead!(mtpOut)
        }
        return (logits, mtpOut)
    }

    /// Run the MTP head forward.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.mtp_forward
    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        mtpForwardWithHidden(hidden: hidden, nextTokenIds: nextTokenIds, cache: cache).0
    }

    /// MTP head module forward WITHOUT the lm_head projection.
    ///
    /// Appends the fused `(hidden, nextToken)` positions to `cache` and returns
    /// the head's post-`mtp.norm` hidden rows `[1, M, H]`. This is the
    /// committed-history maintenance primitive (MTPLX `update_mtp_cache`,
    /// runtime.py:364): history rows only need the KV side effect, so skipping
    /// the 248320-wide vocabulary projection on them is pure savings. The
    /// caller applies `applyLMHead` to whichever rows it actually samples.
    public func mtpHeadHiddenForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        guard let mtp else {
            fatalError("mtpHeadHiddenForward called but MTP head is not attached. "
                + "Set _qwen35MTPEnabled = true before loading the model.")
        }
        return mtp(
            hidden: hidden,
            nextTokenIds: nextTokenIds,
            embedTokens: model.embedTokens,
            cache: cache)
    }

    /// The backbone's lm_head (or tied-embedding projection) applied to hidden
    /// rows. Companion to `mtpHeadHiddenForward` for the rows that need logits.
    public func applyLMHead(_ x: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(x)
        }
        return model.embedTokens.asLinear(x)
    }

    /// Draft-only vocabulary projection: the declared head's coarser lm_head
    /// copy when it ships one (`draft_lm_head.*` in the declared head tree),
    /// the exact lm_head otherwise. ONLY used to choose draft proposals —
    /// never for ledger or verify values.
    public func applyDraftLMHead(_ x: MLXArray) -> MLXArray {
        if let w = _draftHeadW, let s = _draftHeadS, let z = _draftHeadZ {
            let k = s.dim(1) * 64
            let bits = w.dim(1) * 32 / k
            return quantizedMM(
                x, w, scales: s, biases: z, transpose: true,
                groupSize: 64, bits: bits, mode: .affine)
        }
        guard usesCompactDraftVocabulary else { return applyLMHead(x) }
        if _compactDraftHead == nil {
            _compactDraftHead = makeCompactDraftHead()
        }
        let padded = _compactDraftHead!(x)
        // Padding exists only to retain qmv_fast's N % 8 shape. Removing it
        // before argmax makes the duplicate rows semantically unreachable.
        return padded[0..., 0..., 0 ..< Self.compactDraftRealCount]
    }

    /// Map compact draft IDs back to the tokenizer's full ID space without a
    /// host readback. The low `compactDraftPrefixCount` rows retain their
    /// IDs; the appended rows are Qwen's official text/control tokens
    /// 248,044 ... 248,069.
    public func mapDraftTokenIds(_ ids: MLXArray) -> MLXArray {
        guard usesCompactDraftVocabulary else { return ids }
        return which(
            ids .< Self.compactDraftPrefixCount,
            ids,
            ids + (Self.compactDraftControlStart - Self.compactDraftPrefixCount))
    }

    private var usesCompactDraftVocabulary: Bool {
        configuration.vocabularySize == 248_320
            && lmHead != nil && _draftHeadW == nil
    }

    private func makeCompactDraftHead() -> Linear {
        guard let full = lmHead else {
            fatalError("compact draft vocabulary requires an untied lm_head")
        }

        func compactRows(_ array: MLXArray) -> MLXArray {
            let prefix = array[0 ..< Self.compactDraftPrefixCount]
            let controls = array[
                Self.compactDraftControlStart ..< Self.compactDraftControlEnd]
            let paddingCount =
                Self.compactDraftPaddedCount - Self.compactDraftRealCount
            let padding = array[0 ..< paddingCount]
            return concatenated([prefix, controls, padding], axis: 0)
        }

        if let quantized = full as? QuantizedLinear {
            return QuantizedLinear(
                weight: compactRows(quantized.weight),
                bias: quantized.bias.map(compactRows),
                scales: compactRows(quantized.scales),
                biases: quantized.biases.map(compactRows),
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode)
        }
        return Linear(
            weight: compactRows(full.weight),
            bias: full.bias.map(compactRows))
    }


    /// Allocate a fresh KV cache for the MTP head layers.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.make_mtp_cache
    public func makeMTPCache() -> [any KVCache] {
        guard let mtp else { return [] }
        return mtp.layers.map { _ in KVCacheSimple() as any KVCache }
    }
}

extension Qwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level Model

public class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    public init(_ args: Qwen35Configuration) {
        let textModel = Qwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - Qwen35Model + MTPCapable

/// VLM-outer-wrapper pass-through for MTPCapable.
/// Forwards all MTP calls to the inner `languageModel` (a Qwen35TextModel).
/// omlx: patches/mlx_lm_mtp/qwen35_model.py `_patch_outer_model`
extension Qwen35Model: MTPCapable {
    public var hasMTPHead: Bool { languageModel.hasMTPHead }

    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        languageModel.callWithHidden(input: input, cache: cache, nConfirmed: nConfirmed)
    }

    /// See `Qwen35TextModel.replayRecurrentPrefix`.
    public func replayRecurrentPrefix(
        cache: [any KVCache], committedRows: Int
    ) -> Bool {
        languageModel.replayRecurrentPrefix(
            cache: cache, committedRows: committedRows)
    }

    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        languageModel.mtpForward(hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    /// See `Qwen35TextModel.mtpForwardWithHidden`.
    public func mtpForwardWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> (MLXArray, MLXArray) {
        languageModel.mtpForwardWithHidden(
            hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    /// See `Qwen35TextModel.mtpHeadHiddenForward`.
    public func mtpHeadHiddenForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        languageModel.mtpHeadHiddenForward(
            hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    /// See `Qwen35TextModel.applyLMHead`.
    public func applyLMHead(_ x: MLXArray) -> MLXArray {
        languageModel.applyLMHead(x)
    }

    /// See `Qwen35TextModel.applyDraftLMHead`.
    public func applyDraftLMHead(_ x: MLXArray) -> MLXArray {
        languageModel.applyDraftLMHead(x)
    }

    /// See `Qwen35TextModel.mapDraftTokenIds`.
    public func mapDraftTokenIds(_ ids: MLXArray) -> MLXArray {
        languageModel.mapDraftTokenIds(ids)
    }


    /// See `Qwen35TextModel.applyFinalNorm`.
    public func applyFinalNorm(_ x: MLXArray) -> MLXArray {
        languageModel.applyFinalNorm(x)
    }

    public func makeMTPCache() -> [any KVCache] {
        languageModel.makeMTPCache()
    }
}
