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

/// Fuse the full-attention output gate `x * sigmoid(gate)` into one compiled
/// elementwise pass, replacing the separate sigmoid and multiply launches (and
/// their intermediate materialization) in every full-attention layer call.
/// Same primitive arithmetic as `sigmoidMultiply` — sigmoid first, then
/// multiply — so the values are bit-identical to the two-launch path; only
/// the launches and the intermediate buffer disappear. Shapeless compilation
/// shares one trace across prefill and verify widths.
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


// MARK: - packed GDN prework mixer (verify widths 3...9)
//
// ONE launch replacing the wide verify's GDN prework chain — conv1d + SiLU +
// split + Q/K rmsNorm-and-scale + the g/beta producer — for S in 3...9 on
// the one-wide-call path. Six outputs: normed/scaled Q and K, activated V,
// the next 3-row conv state, and fp32 `g` and `beta`. An exhaustive sweep of
// all 65,280 finite bf16 inputs found the in-kernel sigmoid diverges from MLX's
// graph sigmoid by 1 ulp on exactly one input (0xC0DB = -6.84375). The helper
// below maps that input directly to MLX's bf16 output word (0x3A8B) before the
// lossless fp32 expansion; every other input retains the inherited expression.
// This removes the final [1,S,48] elementwise launch without changing the
// recurrence's beta bytes. The original five outputs measured bit-exact at
// S=3...9 over 5 seeds x 4 compile modes (12,983,040 element comparisons,
// zero mismatches)
// on the vendored MLX version, with a +1-row conv-window negative control
// failing exactly the three outputs that read the window. S=2 breaks the
// conv-state copy (a state row would come from the OLD conv state, which the
// copy loop does not read), hence the hard S >= 3 gate. The fused in-proj
// carrier's live row stride (16480, not 10240) is consumed via the provided
// stride arrays — ensureRowContiguous stays FALSE; forcing contiguity here
// would silently insert a full-carrier copy and give back the launch saving.
private let qwen35PackedGDNPreworkKernel: MLXFast.MLXFastKernel = {
    let header = """
        typedef bfloat16_t InT;

        inline InT qwen35_prework_sigmoid(InT x) {
          auto y = 1 / (1 + metal::exp(metal::abs(x)));
          return (x < 0) ? y : 1 - y;
        }

        inline InT qwen35_prework_beta(InT x) {
          const uint16_t bits = as_type<uint16_t>(x);
          if (bits == uint16_t(0xC0DB)) {
            return as_type<InT>(uint16_t(0x3A8B));
          }
          return qwen35_prework_sigmoid(x);
        }

        inline InT qwen35_prework_logaddexp(InT x, InT y) {
          if (metal::isnan(x) || metal::isnan(y)) {
            return metal::numeric_limits<InT>::quiet_NaN();
          }
          constexpr InT inf = metal::numeric_limits<InT>::infinity();
          InT maxval = metal::max(x, y);
          InT minval = metal::min(x, y);
          return (minval == -inf || maxval == inf)
              ? maxval
              : (maxval + log1p(metal::exp(minval - maxval)));
        }
        """
    let source = """
        const uint lane = thread_position_in_threadgroup.x;
        const uint row = threadgroup_position_in_grid.y;
        const uint logical_head = threadgroup_position_in_grid.z;

        constexpr uint q_heads = Hk;
        constexpr uint k_head_base = Hk;
        constexpr uint v_head_base = 2 * Hk;
        const bool is_q = logical_head < q_heads;
        const bool is_k = logical_head >= k_head_base
                       && logical_head < v_head_base;
        const uint head = is_q ? logical_head
                         : (is_k ? logical_head - k_head_base
                                 : logical_head - v_head_base);
        const uint channel_base = is_q ? head * Dk
                                  : (is_k ? Hk * Dk + head * Dk
                                          : 2 * Hk * Dk + head * Dv);

        InT activated[4];
        float sumsq = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
          const uint channel = channel_base + lane * 4 + i;
          float acc = 0.0f;
          #pragma clang loop unroll(full)
          for (uint tap = 0; tap < 4; ++tap) {
            const uint input_row = row + tap;
            const ulong input_offset = input_row < NKeep
                ? ulong(input_row) * ulong(conv_state_strides[1])
                    + ulong(channel) * ulong(conv_state_strides[2])
                : ulong(input_row - NKeep) * ulong(qkv_strides[1])
                    + ulong(channel) * ulong(qkv_strides[2]);
            const InT xv = input_row < NKeep
                ? conv_state[input_offset]
                : qkv[input_offset];
            const ulong weight_offset =
                ulong(channel) * ulong(conv_weight_strides[0])
                + ulong(tap) * ulong(conv_weight_strides[1]);
            acc += static_cast<float>(xv) * conv_weight[weight_offset];
          }
          const InT conv = static_cast<InT>(acc);
          const InT act = conv * qwen35_prework_sigmoid(conv);
          activated[i] = act;
          const float value = static_cast<float>(act);
          sumsq += value * value;
        }

        if (is_q || is_k) {
          threadgroup float local_inv_mean[1];
          threadgroup float local_sums[32];
          sumsq = simd_sum(sumsq);
          local_sums[lane] = 0.0f;
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (lane == 0) {
            local_sums[0] = sumsq;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          sumsq = simd_sum(local_sums[lane]);
          if (lane == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(sumsq / Dk + 1e-6f);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          const InT scale = is_q ? q_scale : k_scale;
          const uint output_base = (row * Hk + head) * Dk + lane * 4;
          #pragma clang loop unroll(full)
          for (uint i = 0; i < 4; ++i) {
            const InT rms = InT(1) * static_cast<InT>(
                static_cast<float>(activated[i]) * local_inv_mean[0]);
            const InT value = scale * rms;
            if (is_q) {
              q_out[output_base + i] = value;
            } else {
              k_out[output_base + i] = value;
            }
          }
        } else {
          const uint output_base = (row * Hv + head) * Dv + lane * 4;
          #pragma clang loop unroll(full)
          for (uint i = 0; i < 4; ++i) {
            v_out[output_base + i] = activated[i];
          }

          if (lane == 0) {
            const ulong a_offset = ulong(row) * ulong(a_strides[1])
                + ulong(head) * ulong(a_strides[2]);
            const ulong b_offset = ulong(row) * ulong(b_strides[1])
                + ulong(head) * ulong(b_strides[2]);
            const InT shifted = a[a_offset] + dt_bias[head];
            const InT softplus = qwen35_prework_logaddexp(shifted, InT(0));
            const float exp_a = metal::precise::exp(
                static_cast<float>(a_log[head]));
            const float neg_exp_a = -exp_a;
            const float product = neg_exp_a * static_cast<float>(softplus);
            const uint scalar_output = row * Hv + head;
            g_out[scalar_output] = metal::precise::exp(product);
            beta_out[scalar_output] = static_cast<float>(
                qwen35_prework_beta(b[b_offset]));
          }
        }

        if (row + NKeep >= uint(T)) {
          const uint state_row = row + NKeep - T;
          const ulong raw_base = ulong(row) * ulong(qkv_strides[1])
              + ulong(channel_base + lane * 4) * ulong(qkv_strides[2]);
          const uint state_base = state_row * C + channel_base + lane * 4;
          #pragma clang loop unroll(full)
          for (uint i = 0; i < 4; ++i) {
            conv_out[state_base + i] =
                qkv[raw_base + ulong(i) * ulong(qkv_strides[2])];
          }
        }
        """
    return MLXFast.metalKernel(
        name: "qwen35_packed_gdn_prework",
        inputNames: [
            "qkv", "a", "b", "conv_state", "conv_weight", "a_log",
            "dt_bias", "q_scale", "k_scale",
        ],
        outputNames: [
            "q_out", "k_out", "v_out", "conv_out", "g_out", "beta_out",
        ],
        source: source,
        header: header,
        ensureRowContiguous: false)
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
        // Packed-prework mixer gate: fail closed onto the stock chain for any
        // shape, geometry, or dtype outside the byte-receipt envelope. The
        // S >= 3 lower bound is hard (the kernel's conv-state copy reads only
        // qkv rows, which is wrong at S < nKeep); above 9 no verify exists.
        let mixerHit = MLXHardwareInfo.isCompiledDecodeSupported
            && B == 1 && S >= 3 && S <= 9 && nKeep == 3
            && numKHeads == 16 && numVHeads == 48
            && headKDim == 128 && headVDim == 128
            && qkv.dim(2) == 16 * 128 * 2 + 48 * 128
            && qkv.dtype == .bfloat16 && convState.dtype == .bfloat16
            && a.dtype == .bfloat16 && b.dtype == .bfloat16
        let qNormed: MLXArray
        let kNormed: MLXArray
        let v: MLXArray
        let g: MLXArray
        let beta: MLXArray
        let newConvState: MLXArray
        if mixerHit {
            let invScale = pow(Float(headKDim), -0.5)
            let outs = qwen35PackedGDNPreworkKernel(
                [qkv, a, b, convState, conv1d.weight, aLog, dtBias,
                 MLXArray(pow(invScale, 2)).asType(.bfloat16),
                 MLXArray(invScale).asType(.bfloat16)],
                template: [
                    ("Hk", numKHeads), ("Dk", headKDim),
                    ("Hv", numVHeads), ("Dv", headVDim),
                    ("NKeep", nKeep), ("C", qkv.dim(2)), ("T", S),
                ],
                grid: (32, S, 2 * numKHeads + numVHeads),
                threadGroup: (32, 1, 1),
                outputShapes: [
                    [B, S, numKHeads, headKDim],
                    [B, S, numKHeads, headKDim],
                    [B, S, numVHeads, headVDim],
                    [B, nKeep, qkv.dim(2)],
                    [B, S, numVHeads],
                    [B, S, numVHeads],
                ],
                outputDTypes: [
                    .bfloat16, .bfloat16, .bfloat16, .bfloat16, .float32,
                    .float32,
                ]
            )
            qNormed = outs[0]
            kNormed = outs[1]
            v = outs[2]
            newConvState = outs[3]
            g = outs[4]
            beta = outs[5]
        } else {
            newConvState = convInput[0..., (convInput.dim(1) - nKeep)...]
            let convOut = silu(conv1d(convInput))

            let convSplit = MLX.split(
                convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
            let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
            v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

            let dtype = q.dtype
            let invScale = pow(Float(headKDim), -0.5)
            qNormed =
                MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
            kNormed =
                MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

            // Keep the recurrence and conv prologue wide. The promoted
            // compiled g/beta launch reduction feeds the same single
            // recurrence, while the SDPA helper alone bridges the
            // width-sensitive kernel boundary.
            let gBeta = qwen35CompiledGatedDeltaGBeta(
                a, b, negExpALog, dtBias)
            g = gBeta.0
            beta = gBeta.1
        }
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
        if S >= 2 {
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
///
/// Fuse the SiLU gate and product after the fused gate-up GEMM into one
/// Metal pass: `silu(y[..., :half]) * y[..., half...]` reads each output
/// element once and writes the activation once, replacing the two-kernel
/// slice+silu+mul path (one silu launch, one multiply launch, one
/// intermediate materialization) with a single launch. The arithmetic is
/// elementwise and unchanged — silu first, then multiply, same rounding —
/// so the values are bit-identical to the two-kernel path; only the
/// intermediate buffer disappears. Shapeless compilation shares one trace
/// across verify widths (the fused GEMM's N is constant, so the half-split
/// offsets are width-invariant).
private let qwen35CompiledFusedSwiGLU:
    @Sendable (MLXArray) -> MLXArray =
{
    let body: @Sendable (MLXArray) -> MLXArray = { y in
        let half = y.dim(-1) / 2
        return silu(y[.ellipsis, ..<half]) * y[.ellipsis, half...]
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(body)
    }
    return body
}()

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

    private func fusedGateUp(_ x: MLXArray) -> MLXArray? {
        if let w = _fqW, let s = _fqS, let z = _fqZ {
            return quantizedMM(
                x, w, scales: s, biases: z, transpose: true,
                groupSize: _fqGS, bits: _fqBits, mode: _fqMode)
        }
        if let w = _fbfW {
            return matmul(x, w.T)
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
        // The fused path is only taken when the gate/up split is provably
        // equal halves (`_gateOut * 2 == N`); a mismatched pair falls back
        // to the exact two-projection expression, preserving the original
        // slicing semantics in every case.
        if x.dim(-2) <= 16, let y = fusedGateUp(x), _gateOut * 2 == y.dim(-1) {
            return downProj(qwen35CompiledFusedSwiGLU(y))
        }
        return downProj(silu(gateProj(x)) * upProj(x))
    }

}

// MARK: - Full-attention Q/K preparation

/// Prepare BF16 Q and K rows with the same RMSNorm and partial-RoPE arithmetic
/// as the stock primitives, but read the projection views directly and write
/// their final head-major layout in one dispatch.  Sequence and batch extents
/// are entirely input-derived; the caller gates only on the frozen Qwen model
/// semantics that determine the numerical contract.
private let qwen35AttentionQKRMSRoPEKernel = MLXFast.metalKernel(
    name: "qwen35_attention_qk_rms_rope_bf16_v1",
    inputNames: ["q", "k", "q_weight", "k_weight", "eps", "offset", "log2_base"],
    outputNames: ["q_out", "k_out"],
    source: """
        constexpr uint n_reads = 4;
        constexpr uint simd_size = 32;
        constexpr uint rotary_dimensions = 64;
        constexpr uint rotary_pairs = rotary_dimensions / 2;

        uint row = threadgroup_position_in_grid.x;
        uint thread_id = thread_position_in_threadgroup.x;
        uint simd_thread = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;

        uint batch_size = uint(q_shape[0]);
        uint sequence_length = uint(q_shape[1]);
        uint query_heads = uint(q_shape[2]);
        uint key_heads = uint(k_shape[2]);
        uint axis_size = uint(q_shape[3]);
        uint query_rows = batch_size * query_heads * sequence_length;
        bool is_query = row < query_rows;
        uint local_row = is_query ? row : row - query_rows;
        uint head_count = is_query ? query_heads : key_heads;
        uint batch = local_row / (head_count * sequence_length);
        uint head_sequence = local_row % (head_count * sequence_length);
        uint head = head_sequence / sequence_length;
        uint sequence = head_sequence % sequence_length;

        ulong input_base;
        ulong input_axis_stride;
        ulong weight_stride;
        ulong output_base = ulong(local_row) * ulong(axis_size);
        if (is_query) {
            input_base = ulong(batch) * ulong(q_strides[0])
                + ulong(sequence) * ulong(q_strides[1])
                + ulong(head) * ulong(q_strides[2]);
            input_axis_stride = ulong(q_strides[3]);
            weight_stride = ulong(q_weight_strides[0]);
        } else {
            input_base = ulong(batch) * ulong(k_strides[0])
                + ulong(sequence) * ulong(k_strides[1])
                + ulong(head) * ulong(k_strides[2]);
            input_axis_stride = ulong(k_strides[3]);
            weight_stride = ulong(k_weight_strides[0]);
        }

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[simd_size];
        threadgroup bfloat normalized[256];

        float acc = 0.0f;
        uint first = thread_id * n_reads;
        for (uint i = 0; i < n_reads; ++i) {
            uint element = first + i;
            if (element < axis_size) {
                ulong index = input_base + ulong(element) * input_axis_stride;
                float value = is_query ? float(q[index]) : float(k[index]);
                acc += value * value;
            }
        }

        acc = simd_sum(acc);
        if (simd_group == 0) {
            local_sums[simd_thread] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_thread == 0) {
            local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            acc = simd_sum(local_sums[simd_thread]);
            if (simd_thread == 0) {
                local_inv_mean[0] = metal::precise::rsqrt(
                    acc / axis_size + eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float inv_mean = local_inv_mean[0];
        for (uint i = 0; i < n_reads; ++i) {
            uint element = first + i;
            if (element < axis_size) {
                ulong index = input_base + ulong(element) * input_axis_stride;
                bfloat input_value = is_query ? q[index] : k[index];
                bfloat rms_value = bfloat(float(input_value) * inv_mean);
                bfloat weight = is_query
                    ? q_weight[ulong(element) * weight_stride]
                    : k_weight[ulong(element) * weight_stride];
                normalized[element] = weight * rms_value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The stock RoPE primitive copies dimensions 64...255 unchanged before
        // rotating nontraditional pairs (i, i + 32).  Here the final output is
        // new storage, so only the pass-through tail needs an explicit copy.
        for (uint i = 0; i < n_reads; ++i) {
            uint element = first + i;
            if (element >= rotary_dimensions && element < axis_size) {
                if (is_query) {
                    q_out[output_base + ulong(element)] = normalized[element];
                } else {
                    k_out[output_base + ulong(element)] = normalized[element];
                }
            }
        }

        if (thread_id < rotary_pairs / n_reads) {
            for (uint i = 0; i < n_reads; ++i) {
                uint pair = first + i;
                float d = float(pair) / float(rotary_pairs);
                float inv_freq = metal::exp2(-d * float(log2_base));
                float position = float(int(sequence) + int(offset));
                float theta = position * inv_freq;
                float costheta = metal::fast::cos(theta);
                float sintheta = metal::fast::sin(theta);
                float x1 = float(normalized[pair]);
                float x2 = float(normalized[pair + rotary_pairs]);
                bfloat rx1 = bfloat(x1 * costheta - x2 * sintheta);
                bfloat rx2 = bfloat(x1 * sintheta + x2 * costheta);
                if (is_query) {
                    q_out[output_base + ulong(pair)] = rx1;
                    q_out[output_base + ulong(pair + rotary_pairs)] = rx2;
                } else {
                    k_out[output_base + ulong(pair)] = rx1;
                    k_out[output_base + ulong(pair + rotary_pairs)] = rx2;
                }
            }
        }
    """,
    ensureRowContiguous: false
)

/// Internal for exact Metal parity tests. Inputs are `[B,L,H,D]`; outputs are
/// row-contiguous `[B,H,L,D]` tensors ready for the unchanged attention/cache
/// path.
func qwen35AttentionQKRMSRoPE(
    queries: MLXArray,
    keys: MLXArray,
    qWeight: MLXArray,
    kWeight: MLXArray,
    eps: Float,
    offset: Int,
    log2Base: Float
) -> (queries: MLXArray, keys: MLXArray) {
    let B = queries.dim(0)
    let L = queries.dim(1)
    let queryHeads = queries.dim(2)
    let keyHeads = keys.dim(2)
    let D = queries.dim(3)
    let totalRows = B * L * (queryHeads + keyHeads)
    let outputs = qwen35AttentionQKRMSRoPEKernel(
        [queries, keys, qWeight, kWeight, eps, offset, log2Base],
        grid: (totalRows * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[B, queryHeads, L, D], [B, keyHeads, L, D]],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

// MARK: - Attention

final class Qwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float
    let headDim: Int
    let usesFusedQKPreparation: Bool
    let ropeLog2Base: Float

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

    // Packed K/V concat for committed MTP-head history rows whose layer
    // outputs are dead. Kept separate from the full Q/K/V pack so those rows
    // never stream or compute the unused query+gate projection.
    private var _kvW: MLXArray?
    private var _kvS: MLXArray?
    private var _kvZ: MLXArray?
    private var _kvGS = 64
    private var _kvBits = 4
    private var _kvMode = QuantizationMode.affine
    private var _kvOut = 0
    private var _kvDenseW: MLXArray?

    // Proposal-head-only precision islands.  The declared artifact preserves
    // the promoted affine-4 head and additionally carries selected exact BF16
    // output rows. Target-model attention never installs these arrays.
    private var _exactQKVWeight: MLXArray?
    private var _exactQKVIndices: MLXArray?
    private var _exactKVIndices: MLXArray?
    private var _exactQRowCount = 0

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)
        self.headDim = headDim

        let ropeType: String = {
            if let config = args.ropeScaling,
               let typeValue = config["type"] ?? config["rope_type"],
               case .string(let value) = typeValue
            {
                return value
            }
            return "default"
        }()
        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.usesFusedQKPreparation =
            args.attentionHeads == 24
            && args.kvHeads == 4
            && headDim == 256
            && ropeDims == 64
            && args.ropeTheta == 10_000_000
            && ropeType == "default"
        self.ropeLog2Base = Foundation.log2(args.ropeTheta)

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
            var y = quantizedMM(
                x, w, scales: s, biases: z, transpose: true,
                groupSize: _qkvGS, bits: _qkvBits, mode: _qkvMode)
            y = replaceExactRows(y, input: x, kvOnly: false)
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

    /// One projection for K and V when no query output is observable. The
    /// pack is model-general and is built lazily from the attached linears.
    private func kv(_ x: MLXArray) -> (MLXArray, MLXArray) {
        if let w = _kvW, let s = _kvS, let z = _kvZ {
            var y = quantizedMM(
                x, w, scales: s, biases: z, transpose: true,
                groupSize: _kvGS, bits: _kvBits, mode: _kvMode)
            y = replaceExactRows(y, input: x, kvOnly: true)
            return (y[.ellipsis, ..<_kvOut], y[.ellipsis, _kvOut...])
        }
        if let w = _kvDenseW {
            let y = matmul(x, w.transposed(1, 0))
            return (y[.ellipsis, ..<_kvOut], y[.ellipsis, _kvOut...])
        }
        if let k = kProj as? QuantizedLinear,
           let v = vProj as? QuantizedLinear,
           k.groupSize == v.groupSize,
           k.bits == v.bits,
           k.mode == v.mode, k.mode == .affine,
           let kz = k.biases, let vz = v.biases
        {
            _kvW = concatenated([k.weight, v.weight], axis: 0).contiguous()
            _kvS = concatenated([k.scales, v.scales], axis: 0).contiguous()
            _kvZ = concatenated([kz, vz], axis: 0).contiguous()
            _kvGS = k.groupSize
            _kvBits = k.bits
            _kvMode = k.mode
            _kvOut = k.shape.0
            return kv(x)
        }
        if !(kProj is QuantizedLinear), !(vProj is QuantizedLinear),
           kProj.bias == nil, vProj.bias == nil
        {
            _kvDenseW = concatenated(
                [kProj.weight, vProj.weight], axis: 0
            ).contiguous()
            _kvOut = kProj.weight.dim(0)
            return kv(x)
        }
        return (kProj(x), vProj(x))
    }

    private func replaceExactRows(
        _ base: MLXArray, input: MLXArray, kvOnly: Bool
    ) -> MLXArray {
        guard let exactWeight = _exactQKVWeight else { return base }
        let weight: MLXArray
        let indices: MLXArray?
        if kvOnly {
            weight = exactWeight[_exactQRowCount...]
            indices = _exactKVIndices
        } else {
            weight = exactWeight
            indices = _exactQKVIndices
        }
        guard let indices else { return base }
        let exact = matmul(input, weight.transposed(1, 0))
        let indexShape = Array(repeating: 1, count: max(0, base.ndim - 1)) + [-1]
        return putAlong(
            base, indices.reshaped(indexShape), values: exact, axis: -1)
    }

    func installExactQKVRows(
        qWeight: MLXArray, qIndices: MLXArray, qOutputCount: Int,
        kWeight: MLXArray, kIndices: MLXArray, kOutputCount: Int,
        vWeight: MLXArray, vIndices: MLXArray
    ) {
        precondition(
            qWeight.dim(0) == qIndices.dim(0)
                && kWeight.dim(0) == kIndices.dim(0)
                && vWeight.dim(0) == vIndices.dim(0),
            "Qwen MTP precision-island weights and indices must have equal row counts")
        let weight = concatenated([qWeight, kWeight, vWeight], axis: 0).contiguous()
        let qkvIndices = concatenated(
            [qIndices, kIndices + qOutputCount,
             vIndices + qOutputCount + kOutputCount], axis: 0)
            .asType(.int32).contiguous()
        let kvIndices = concatenated(
            [kIndices, vIndices + kOutputCount], axis: 0)
            .asType(.int32).contiguous()
        eval(weight, qkvIndices, kvIndices)

        _exactQKVWeight = weight
        _exactQKVIndices = qkvIndices
        _exactKVIndices = kvIndices
        _exactQRowCount = qWeight.dim(0)
    }

    /// Append rows to an attention cache without producing query outputs.
    /// The target model never uses this proposal-head maintenance primitive.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        let B = x.dim(0)
        let L = x.dim(1)
        var (keys, values) = kv(x)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)
        _ = cache.update(keys: keys, values: values)
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

        keys = keys.reshaped(B, L, kvHeads, -1)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let hasArrayOffset = cache is CompilableRotatingKVCache
            || cache is CompilableKVCache
            || cache is BatchPositionedKVCache
        if usesFusedQKPreparation,
           L <= 32,
           !hasArrayOffset,
           queries.dtype == .bfloat16,
           keys.dtype == .bfloat16,
           qNorm.weight.dtype == .bfloat16,
           kNorm.weight.dtype == .bfloat16,
           queries.shape == [B, L, attentionHeads, headDim],
           keys.shape == [B, L, kvHeads, headDim],
           qNorm.weight.shape == [headDim],
           kNorm.weight.shape == [headDim],
           qNorm.eps == kNorm.eps
        {
            let prepared = qwen35AttentionQKRMSRoPE(
                queries: queries,
                keys: keys,
                qWeight: qNorm.weight,
                kWeight: kNorm.weight,
                eps: qNorm.eps,
                offset: cache?.offset ?? 0,
                log2Base: ropeLog2Base
            )
            queries = prepared.queries
            keys = prepared.keys
        } else {
            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

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
        let prefillLadder = inputs.dim(1) >= 512
        let ladderActive = inputs.dim(1) <= 9 || prefillLadder
        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask,
                cache: cacheArray?[i], nConfirmed: nConfirmed)
            if ladderActive {
                if prefillLadder {
                    if i == 0 || i % 3 == 2 {
                        asyncEval(hiddenStates)
                    }
                } else {
                    switch i {
                    case 0, 1, 9, 19, 29, 39, 49, 57:
                        asyncEval(hiddenStates)
                    default:
                        break
                    }
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

// MARK: - fused compact-draft selection
//
// PROPOSAL SIDE ONLY. The draft argmax picks which token the MTP head
// PROPOSES; the pinned target re-derives every emitted token from the exact
// `lmHead` and the trusted parent replays the whole stream afterwards, so
// nothing downstream of this kernel can reach an emitted token or a ledger
// value. See `applyDraftLMHead`'s doc comment for the same argument applied
// to the compact row set (promoted 7b33621).
//
// It replaces SIX MLX primitives that existed only to turn a 98,336-wide row
// into one integer:
//     padded[0..., 0..., 0 ..< 98_330]     // slice off the fast-shape padding
//     argMax(axis: -1)                     // uint32
//     .asType(.int32)
//     ids .< 98_304                        // mapDraftTokenIds
//     ids + 149_740
//     which(...)
// Ordering is identical to `argMax`: strictly-greater value wins, an exact tie
// goes to the LOWER id, and a NaN never beats a non-NaN. Bounding at
// `REAL_COUNT` in the kernel is exactly what the pre-argmax slice did, so the
// six duplicated padding rows stay unreachable even on a tie.
private let qwen35DraftSelectKernel = MLXFast.metalKernel(
    name: "qwen_mtp_draft_select",
    inputNames: ["logits"],
    outputNames: ["token_id"],
    source: """
        uint thread_id = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        float best_value = 0.0f;
        uint  best_id    = 0;
        bool  have       = false;

        for (uint index = thread_id; index < REAL_COUNT; index += TG_SIZE) {
            float value = float(logits[index]);
            bool take = !have || qwen_draft_better(
                value, index, best_value, best_id);
            if (take) { best_value = value; best_id = index; have = true; }
        }

        if (!have) {
            best_value = NAN;
            best_id = 0xFFFFFFFFu;
        }

        // First level: each 32-thread SIMD group reduces its private winners
        // with shuffle operations, without touching threadgroup memory.
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float other_value = simd_shuffle_down(best_value, offset);
            uint other_id = simd_shuffle_down(best_id, offset);
            if (simd_lane < offset && qwen_draft_better(
                    other_value, other_id, best_value, best_id)) {
                best_value = other_value;
                best_id = other_id;
            }
        }

        // Second level: publish only 32 SIMD winners, synchronize once, and
        // let the first SIMD group reduce them with the identical total order.
        threadgroup float scratch_value[TG_SIZE / 32];
        threadgroup uint  scratch_id[TG_SIZE / 32];
        if (simd_lane == 0) {
            scratch_value[simd_group] = best_value;
            scratch_id[simd_group] = best_id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            best_value = scratch_value[simd_lane];
            best_id = scratch_id[simd_lane];
            for (uint offset = 16; offset > 0; offset >>= 1) {
                float other_value = simd_shuffle_down(best_value, offset);
                uint other_id = simd_shuffle_down(best_id, offset);
                if (simd_lane < offset && qwen_draft_better(
                        other_value, other_id, best_value, best_id)) {
                    best_value = other_value;
                    best_id = other_id;
                }
            }
            if (simd_lane == 0) {
                token_id[0] = int(
                    best_id < PREFIX_COUNT
                        ? best_id
                        : best_id + CONTROL_OFFSET);
            }
        }
    """,
    header: """
        inline bool qwen_draft_better(
            float candidate_value,
            uint candidate_id,
            float current_value,
            uint current_id
        ) {
            if (candidate_id == 0xFFFFFFFFu) { return false; }
            if (current_id == 0xFFFFFFFFu) { return true; }
            bool candidate_nan = isnan(candidate_value);
            bool current_nan = isnan(current_value);
            if (candidate_nan != current_nan) { return !candidate_nan; }
            if (candidate_value > current_value) { return true; }
            if (candidate_value < current_value) { return false; }
            return candidate_id < current_id;
        }
    """,
    ensureRowContiguous: false
)

// PROPOSAL SIDE ONLY. A coarse affine-2 compact readout chooses 32 rows; the
// incumbent affine-4 compact readout evaluates those rows, and this single
// SIMDgroup applies the incumbent value/id total order to select the proposal.
// The target lm_head, verify values, cache state, and row ledger are untouched.
private let qwen35DraftRerankKernel = MLXFast.metalKernel(
    name: "qwen_mtp_draft_rerank",
    inputNames: ["logits", "candidate_ids"],
    outputNames: ["token_id"],
    source: """
        uint lane = thread_index_in_simdgroup;
        float best_value = float(logits[lane]);
        uint best_id = uint(candidate_ids[lane]);

        for (uint offset = 16; offset > 0; offset >>= 1) {
            float other_value = simd_shuffle_down(best_value, offset);
            uint other_id = simd_shuffle_down(best_id, offset);
            if (lane < offset && qwen_draft_rerank_better(
                    other_value, other_id, best_value, best_id)) {
                best_value = other_value;
                best_id = other_id;
            }
        }

        if (lane == 0) {
            token_id[0] = int(
                best_id < PREFIX_COUNT
                    ? best_id
                    : best_id + CONTROL_OFFSET);
        }
    """,
    header: """
        inline bool qwen_draft_rerank_better(
            float candidate_value,
            uint candidate_id,
            float current_value,
            uint current_id
        ) {
            bool candidate_nan = isnan(candidate_value);
            bool current_nan = isnan(current_value);
            if (candidate_nan != current_nan) { return !candidate_nan; }
            if (candidate_value > current_value) { return true; }
            if (candidate_value < current_value) { return false; }
            return candidate_id < current_id;
        }
    """,
    ensureRowContiguous: false
)

// ---------------------------------------------------------------------------
// PROPOSAL-SIDE TOP-32 SHORTLIST
//
// Replaces `MLX.argPartition(coarse, kth: 98_298, axis: -1)[98_298...]`.
//
// `ArgPartition::eval_gpu` in this vendored MLX is a stub -- its own comment
// reads "We direct arg partition to sort for now" -- so it calls
// `gpu_merge_sort(..., argsort=true)`. For a 98,330-wide bf16 row that
// resolves to `multi_block_sort` (bn=512, tn=4, n_blocks=49): one block sort
// + six merge levels x two kernels + one output copy = 14 dependent
// dispatches and five device temporaries, to FULLY ARGSORT 98,330 elements
// from which exactly 32 are ever read. This pair answers the same question
// in two dispatches.
//
// EXACTNESS. MLX's merge sort is STABLE ASCENDING under `LessThan<T>`:
// ThreadSort swaps only on strict less, merge_step takes from B only on
// strict less, merge_partition advances into A on ties, and block_sort seeds
// ascending global indices with an `idx < size_sorted_axis` write guard so
// the padding slots never reach the output. Its tail-32 is therefore the
// unique 32-element set maximal under (value asc, index asc) -- ties broken
// toward the HIGHER index, NaN ranking ABOVE every number because LessThan
// returns (!an) & bn. `qwen_top32_ordinal` is a monotone map from float into
// uint32 inducing exactly that order, including both corners: -0.0 folds
// into +0.0 so the pair ties and breaks by index, and every NaN payload
// collapses to one ordinal so all NaNs tie and break by index.
//
// Bounding the scan at REAL_COUNT inside the kernel is exactly what the
// removed `[0 ..< 98_330]` pre-slice did, so the six duplicated padding rows
// stay unreachable even on a tie.
//
// Downstream, `qwen35DraftRerankKernel` reduces the 32 candidates under a
// strict total order on (value, id), which is order-independent -- so set
// identity would suffice. Element-wise identity is a strictly stronger
// property and makes the offline gate a plain array equality.
private let qwen35Top32RealCount    = 98_330
private let qwen35Top32K            = 32
private let qwen35Top32TG           = 256
private let qwen35Top32Tiles        = 64
private let qwen35Top32Stride       = qwen35Top32Tiles * qwen35Top32TG
private let qwen35Top32PerThread    =
    (qwen35Top32RealCount + qwen35Top32Stride - 1) / qwen35Top32Stride
private let qwen35Top32Cands        = qwen35Top32Tiles * qwen35Top32K
private let qwen35Top32FinPerThread = qwen35Top32Cands / qwen35Top32TG

private let qwen35Top32Header = """
    inline uint qwen_top32_ordinal(float v) {
        if (isnan(v))  { return 0xFFFFFFFFu; }
        if (v == 0.0f) { return 0x80000000u; }
        uint u = as_type<uint>(v);
        return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
    }
    """

// Stage 1: 64 threadgroups partition [0, REAL_COUNT); each emits its top 32
// as (ordinal, index) pairs. 64 * 32 = 2,048 candidates.
private let qwen35DraftTop32PartialKernel = MLXFast.metalKernel(
    name: "qwen_mtp_draft_top32_partial",
    inputNames: ["logits"],
    outputNames: ["cand_ord", "cand_idx"],
    source: """
        constexpr uint REAL_COUNT = \(qwen35Top32RealCount);
        constexpr uint TG_SIZE    = \(qwen35Top32TG);
        constexpr uint STRIDE     = \(qwen35Top32Stride);
        constexpr uint PER_THREAD = \(qwen35Top32PerThread);
        constexpr uint TOPK       = \(qwen35Top32K);
        constexpr uint SIMD_SIZE  = 32;
        constexpr uint NSIMD      = TG_SIZE / SIMD_SIZE;
        constexpr uint PB         = (NSIMD * TOPK) / SIMD_SIZE;
        // `taken` / `tk2` are 32-bit bitmasks indexed by slot, so a slot count
        // above 32 would shift out of range and silently corrupt the result.
        static_assert(PER_THREAD <= 32, "PER_THREAD exceeds taken-bitmask width");
        static_assert(PB <= 32, "PB exceeds tk2-bitmask width");

        uint tile = threadgroup_position_in_grid.x;
        uint tid  = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg   = simdgroup_index_in_threadgroup;

        uint ord[PER_THREAD];
        uint idx[PER_THREAD];
        for (uint t = 0; t < PER_THREAD; ++t) { ord[t] = 0u; idx[t] = 0u; }
        uint n = 0;
        for (uint i = tile * TG_SIZE + tid; i < REAL_COUNT; i += STRIDE) {
            ord[n] = qwen_top32_ordinal(float(logits[i]));
            idx[n] = i;
            n++;
        }

        threadgroup uint sc_ord[NSIMD * TOPK];
        threadgroup uint sc_idx[NSIMD * TOPK];

        uint taken = 0u;
        for (uint r = 0; r < TOPK; ++r) {
            uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
            for (uint t = 0; t < PER_THREAD; ++t) {
                if ((taken & (1u << t)) != 0u) { continue; }
                if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {
                    bo = ord[t]; bi = idx[t]; bs = t;
                }
            }
            uint mo = simd_max(bo);
            uint mi = simd_max((bo == mo) ? bi : 0u);
            if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
                taken |= (1u << bs);
            }
            if (lane == 0) {
                sc_ord[sg * TOPK + r] = mo;
                sc_idx[sg * TOPK + r] = mi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            uint o2[PB];
            uint i2[PB];
            for (uint t = 0; t < PB; ++t) {
                uint p = t * SIMD_SIZE + lane;
                o2[t] = sc_ord[p];
                i2[t] = sc_idx[p];
            }
            uint tk2 = 0u;
            for (uint r = 0; r < TOPK; ++r) {
                uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
                for (uint t = 0; t < PB; ++t) {
                    if ((tk2 & (1u << t)) != 0u) { continue; }
                    if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {
                        bo = o2[t]; bi = i2[t]; bs = t;
                    }
                }
                uint mo = simd_max(bo);
                uint mi = simd_max((bo == mo) ? bi : 0u);
                if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
                    tk2 |= (1u << bs);
                }
                if (lane == 0) {
                    cand_ord[tile * TOPK + r] = mo;
                    cand_idx[tile * TOPK + r] = mi;
                }
            }
        }
        """,
    header: qwen35Top32Header,
    ensureRowContiguous: false
)

// Stage 2: one threadgroup reduces the 2,048 candidates to the final 32,
// written ASCENDING so the result is element-wise identical to
// `argPartition(...)[kth...]`.
private let qwen35DraftTop32FinalizeKernel = MLXFast.metalKernel(
    name: "qwen_mtp_draft_top32_finalize",
    inputNames: ["cand_ord", "cand_idx"],
    outputNames: ["token_ids"],
    source: """
        constexpr uint TG_SIZE    = \(qwen35Top32TG);
        constexpr uint PER_THREAD = \(qwen35Top32FinPerThread);
        constexpr uint TOPK       = \(qwen35Top32K);
        constexpr uint SIMD_SIZE  = 32;
        constexpr uint NSIMD      = TG_SIZE / SIMD_SIZE;
        constexpr uint PB         = (NSIMD * TOPK) / SIMD_SIZE;
        static_assert(PER_THREAD <= 32, "PER_THREAD exceeds taken-bitmask width");
        static_assert(PB <= 32, "PB exceeds tk2-bitmask width");

        uint tid  = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg   = simdgroup_index_in_threadgroup;

        uint ord[PER_THREAD];
        uint idx[PER_THREAD];
        for (uint t = 0; t < PER_THREAD; ++t) {
            uint p = t * TG_SIZE + tid;
            ord[t] = cand_ord[p];
            idx[t] = cand_idx[p];
        }

        threadgroup uint sc_ord[NSIMD * TOPK];
        threadgroup uint sc_idx[NSIMD * TOPK];

        uint taken = 0u;
        for (uint r = 0; r < TOPK; ++r) {
            uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
            for (uint t = 0; t < PER_THREAD; ++t) {
                if ((taken & (1u << t)) != 0u) { continue; }
                if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {
                    bo = ord[t]; bi = idx[t]; bs = t;
                }
            }
            uint mo = simd_max(bo);
            uint mi = simd_max((bo == mo) ? bi : 0u);
            if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
                taken |= (1u << bs);
            }
            if (lane == 0) {
                sc_ord[sg * TOPK + r] = mo;
                sc_idx[sg * TOPK + r] = mi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            uint o2[PB];
            uint i2[PB];
            for (uint t = 0; t < PB; ++t) {
                uint p = t * SIMD_SIZE + lane;
                o2[t] = sc_ord[p];
                i2[t] = sc_idx[p];
            }
            uint tk2 = 0u;
            for (uint r = 0; r < TOPK; ++r) {
                uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
                for (uint t = 0; t < PB; ++t) {
                    if ((tk2 & (1u << t)) != 0u) { continue; }
                    if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {
                        bo = o2[t]; bi = i2[t]; bs = t;
                    }
                }
                uint mo = simd_max(bo);
                uint mi = simd_max((bo == mo) ? bi : 0u);
                if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
                    tk2 |= (1u << bs);
                }
                if (lane == 0) { token_ids[TOPK - 1u - r] = mi; }
            }
        }
        """,
    header: "",
    ensureRowContiguous: false
)

// `MLXFAST_QWEN_MTP_TOP32=0` restores the argPartition path bit-for-bit.
private let qwen35Top32Enabled: Bool =
    ProcessInfo.processInfo.environment["MLXFAST_QWEN_MTP_TOP32"] != "0"

/// Exact top-32 of `row` (shape [REAL_COUNT], bf16) as ascending uint32 ids.
private func qwen35DraftTop32(_ row: MLXArray) -> MLXArray {
    // Mirrors the kernel static_asserts; see the bitmask note there.
    precondition(qwen35Top32PerThread <= 32 && qwen35Top32FinPerThread <= 32,
                 "top-32 slot count exceeds the 32-bit selection bitmask")
    let partial = qwen35DraftTop32PartialKernel(
        [row],
        grid: (qwen35Top32Tiles * qwen35Top32TG, 1, 1),
        threadGroup: (qwen35Top32TG, 1, 1),
        outputShapes: [[qwen35Top32Cands], [qwen35Top32Cands]],
        outputDTypes: [.uint32, .uint32]
    )
    return qwen35DraftTop32FinalizeKernel(
        [partial[0], partial[1]],
        grid: (qwen35Top32TG, 1, 1),
        threadGroup: (qwen35Top32TG, 1, 1),
        outputShapes: [[qwen35Top32K]],
        outputDTypes: [.uint32]
    )[0]
}

/// Offline equivalence gate. Needs no checkpoint and no MTP head: it exercises
/// the two selection kernels against `MLX.argPartition` on synthetic bf16 rows
/// of the live width. Returns (checked, mismatches, firstBadTrial).
/// Never called on a scored path.
/// Isolated micro-benchmark: times the two-dispatch top-32 against
/// `MLX.argPartition` on a real-width bf16 row. Never called on a scored path.
public func qwen35BenchDraftTop32(iters: Int = 200) -> (Double, Double, Int, Int) {
    MLXRandom.seed(3)
    let row = MLXRandom.normal([qwen35Top32RealCount]).asType(.bfloat16)
    let kth = qwen35Top32RealCount - qwen35Top32K
    // warm both paths (JIT + allocator)
    for _ in 0 ..< 10 {
        eval(qwen35DraftTop32(row))
        eval(MLX.argPartition(row, kth: kth, axis: -1)[(kth)...])
    }
    var t0 = Date()
    for _ in 0 ..< iters { eval(MLX.argPartition(row, kth: kth, axis: -1)[(kth)...]) }
    let baseUs = Date().timeIntervalSince(t0) / Double(iters) * 1e6
    t0 = Date()
    for _ in 0 ..< iters { eval(qwen35DraftTop32(row)) }
    let mineUs = Date().timeIntervalSince(t0) / Double(iters) * 1e6
    return (baseUs, mineUs, qwen35Top32Tiles, qwen35Top32PerThread)
}

public func qwen35VerifyDraftTop32(trials: Int = 64, seed: UInt64 = 1) -> (Int, Int, Int) {
    MLXRandom.seed(seed)
    var bad = 0
    var firstBad = -1
    let kth = qwen35Top32RealCount - qwen35Top32K
    for trial in 0 ..< trials {
        var row = MLXRandom.normal([qwen35Top32RealCount]).asType(.bfloat16)
        if trial % 4 == 1 {
            // force heavy ties: quantise hard so many values collide
            row = (MLXRandom.normal([qwen35Top32RealCount]) * 4).round().asType(.bfloat16)
        }
        let mine = qwen35DraftTop32(row)
        let theirs = MLX.argPartition(row, kth: kth, axis: -1)[(kth)...]
            .asType(.uint32)
        eval(mine, theirs)
        let same = MLX.all(MLX.equal(mine, theirs)).item(Bool.self)
        if !same {
            bad += 1
            if firstBad < 0 { firstBad = trial }
        }
    }
    return (trials, bad, firstBad)
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
    private static let draftRerankCandidateCount = 32

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

        // A declared packed head may carry a compact BF16 correction set for
        // its proposal attention.  Side-channel these non-parameter tensors
        // out of the strict Module update, then install them only on the MTP
        // layer.  The target model never sees or consumes this artifact.
        let islandPrefix = "mtp.precision_islands."
        let islandKeys = weights.keys.filter { $0.hasPrefix(islandPrefix) }
        if !islandKeys.isEmpty {
            let qWeight = weights.removeValue(forKey: islandPrefix + "q.weight")
            let qIndices = weights.removeValue(forKey: islandPrefix + "q.indices")
            let kWeight = weights.removeValue(forKey: islandPrefix + "k.weight")
            let kIndices = weights.removeValue(forKey: islandPrefix + "k.indices")
            let vWeight = weights.removeValue(forKey: islandPrefix + "v.weight")
            let vIndices = weights.removeValue(forKey: islandPrefix + "v.indices")
            guard let qWeight, let qIndices, let kWeight, let kIndices,
                  let vWeight, let vIndices, let layer = mtp?.layers.first
            else {
                fatalError(
                    "Qwen MTP precision-island artifact is incomplete; expected "
                        + "Q/K/V weight+indices tensors")
            }
            if ProcessInfo.processInfo.environment[
                "MLXFAST_QWEN_MTP_EXACT_QKV_ROWS"] != "0"
            {
                layer.selfAttn.installExactQKVRows(
                    qWeight: qWeight, qIndices: qIndices, qOutputCount: 12_288,
                    kWeight: kWeight, kIndices: kIndices, kOutputCount: 1_024,
                    vWeight: vWeight, vIndices: vIndices)
            }
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

    /// Publish the post-norm block this same forward already needs for its
    /// vocabulary projection. The verify session can reuse its rows on the
    /// proposal side instead of launching identical single-row RMSNorms.
    public func callWithHiddenAndNormed(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray, MLXArray?) {
        let cacheOpt: [KVCache?] = cache.map { Optional($0) }
        let hidden = model(input.tokens, cache: cacheOpt, nConfirmed: nConfirmed)
        let normed = model.norm(hidden)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(normed)
        } else {
            logits = model.embedTokens.asLinear(normed)
        }
        return (logits, hidden, normed)
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

    /// Return the final proposal hidden row while populating preceding history
    /// through a K/V-only path. Returns nil before mutation when unavailable.
    public func mtpHeadLastHiddenWithKVOnlyHistory(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray? {
        guard let mtp else { return nil }
        return mtp.lastHiddenWithKVOnlyHistory(
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
            let logits = quantizedMM(
                x, w, scales: s, biases: z, transpose: true,
                groupSize: 64, bits: bits, mode: .affine)
            if w.dim(0) == Self.compactDraftPaddedCount {
                return logits[0..., 0..., 0 ..< Self.compactDraftRealCount]
            }
            return logits
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

    /// One draft proposal: the compact projection's argmax, already mapped back
    /// to the tokenizer's ID space, as a device-resident `[1, 1]` int32.
    ///
    /// Same value as `mapDraftTokenIds(argMax(applyDraftLMHead(x), axis: -1))`,
    /// produced in ONE dispatch instead of six. `applyDraftLMHead` and
    /// `mapDraftTokenIds` are unchanged and still serve the declared-head path
    /// and the untimed warm.
    public func draftTokenID(_ x: MLXArray) -> MLXArray {
        if _draftHeadW != nil {
            if let reranked = draftTokenIDWithDeclaredRerank(x) {
                return reranked
            }
            return mapDraftTokenIds(
                argMax(applyDraftLMHead(x), axis: -1).asType(.int32))
        }
        guard usesCompactDraftVocabulary else {
            return argMax(applyDraftLMHead(x), axis: -1).asType(.int32)
        }
        if _compactDraftHead == nil {
            _compactDraftHead = makeCompactDraftHead()
        }
        let padded = _compactDraftHead!(x)
        let tgSize = 1024
        let outputs = qwen35DraftSelectKernel(
            [padded.reshaped([Self.compactDraftPaddedCount])],
            template: [
                ("REAL_COUNT", Self.compactDraftRealCount),
                ("PREFIX_COUNT", Self.compactDraftPrefixCount),
                ("CONTROL_OFFSET",
                 Self.compactDraftControlStart - Self.compactDraftPrefixCount),
                ("TG_SIZE", tgSize),
            ],
            grid: (tgSize, 1, 1),
            threadGroup: (tgSize, 1, 1),
            outputShapes: [[1, 1]],
            outputDTypes: [.int32]
        )
        return outputs[0]
    }

    private func draftTokenIDWithDeclaredRerank(_ x: MLXArray) -> MLXArray? {
        guard let coarseWeight = _draftHeadW,
              let coarseScales = _draftHeadS,
              let coarseBiases = _draftHeadZ,
              coarseWeight.dim(0) == Self.compactDraftPaddedCount,
              coarseWeight.dim(1) == 320,
              coarseScales.shape == [Self.compactDraftPaddedCount, 80],
              coarseBiases.shape == [Self.compactDraftPaddedCount, 80],
              x.shape == [1, 1, configuration.hiddenSize]
        else { return nil }

        if _compactDraftHead == nil {
            _compactDraftHead = makeCompactDraftHead()
        }
        guard let exact = _compactDraftHead as? QuantizedLinear,
              exact.groupSize == 64,
              exact.bits == 4,
              exact.weight.shape == [Self.compactDraftPaddedCount, 640],
              exact.scales.shape == [Self.compactDraftPaddedCount, 80],
              let exactBiases = exact.biases,
              exactBiases.shape == [Self.compactDraftPaddedCount, 80]
        else { return nil }

        let coarse = quantizedMM(
            x, coarseWeight, scales: coarseScales, biases: coarseBiases,
            transpose: true, groupSize: 64, bits: 2, mode: .affine
        )
        let candidateCount = Self.draftRerankCandidateCount
        // Drift guard: the kernels bake these shapes in as constexpr.
        guard qwen35Top32RealCount == Self.compactDraftRealCount,
              qwen35Top32K == candidateCount
        else { return nil }
        let candidateIDs: MLXArray
        if qwen35Top32Enabled {
            candidateIDs = qwen35DraftTop32(
                coarse[0..., 0..., 0 ..< Self.compactDraftRealCount]
                    .reshaped([Self.compactDraftRealCount]))
        } else {
            let kth = Self.compactDraftRealCount - candidateCount
            candidateIDs = MLX.argPartition(
                coarse[0..., 0..., 0 ..< Self.compactDraftRealCount],
                kth: kth, axis: -1
            )[.ellipsis, (kth)...].reshaped([candidateCount])
        }

        let exactWeight = MLX.take(exact.weight, candidateIDs, axis: 0)
        let exactScales = MLX.take(exact.scales, candidateIDs, axis: 0)
        let exactZeroPoints = MLX.take(exactBiases, candidateIDs, axis: 0)
        let exactLogits = quantizedMM(
            x, exactWeight, scales: exactScales, biases: exactZeroPoints,
            transpose: true, groupSize: 64, bits: 4, mode: .affine)

        return qwen35DraftRerankKernel(
            [exactLogits.reshaped([candidateCount]), candidateIDs],
            template: [
                ("PREFIX_COUNT", Self.compactDraftPrefixCount),
                ("CONTROL_OFFSET",
                 Self.compactDraftControlStart - Self.compactDraftPrefixCount),
            ],
            grid: (candidateCount, 1, 1),
            threadGroup: (candidateCount, 1, 1),
            outputShapes: [[1, 1]],
            outputDTypes: [.int32]
        )[0]
    }

    /// Map compact draft IDs back to the tokenizer's full ID space without a
    /// host readback. The low `compactDraftPrefixCount` rows retain their
    /// IDs; the appended rows are Qwen's official text/control tokens
    /// 248,044 ... 248,069.
    public func mapDraftTokenIds(_ ids: MLXArray) -> MLXArray {
        let declaredCompact =
            _draftHeadW.map { $0.dim(0) == Self.compactDraftPaddedCount }
            ?? false
        guard usesCompactDraftVocabulary || declaredCompact else { return ids }
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

    /// See `Qwen35TextModel.callWithHiddenAndNormed`.
    public func callWithHiddenAndNormed(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray, MLXArray?) {
        languageModel.callWithHiddenAndNormed(
            input: input, cache: cache, nConfirmed: nConfirmed)
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

    /// See `Qwen35TextModel.mtpHeadLastHiddenWithKVOnlyHistory`.
    public func mtpHeadLastHiddenWithKVOnlyHistory(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray? {
        languageModel.mtpHeadLastHiddenWithKVOnlyHistory(
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

    /// See `Qwen35TextModel.draftTokenID`.
    public func draftTokenID(_ x: MLXArray) -> MLXArray {
        languageModel.draftTokenID(x)
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
