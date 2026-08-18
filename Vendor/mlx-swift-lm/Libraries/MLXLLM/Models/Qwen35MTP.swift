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

/// Head-only fused single-row draft step. Set per round by the session.
/// PROPOSAL SIDE ONLY: every kernel below mirrors the stock op chain's
/// arithmetic bit for bit (same reduction orders, same bf16 rounding points,
/// same qdot/load_vector expressions as the running JIT twins), so the drafts
/// are unchanged; only the dispatch count changes. The target tower never
/// reads this flag.
public nonisolated(unsafe) var _qwen35MTPHeadFuseEnabled: Bool = false

// MARK: - Fused proposal-step kernels
//
// The steady single-row draft step on the declared q4/g64 head dispatches 35
// kernels (measured, MLX_DISPATCH_LOG). Three custom kernels remove 11 of
// them without touching the tuned stock QMVs where they are bandwidth-bound:
//
//   prefc_prepare  embed gather x3 + dequant + 2 pre-fc RMSNorms + concat x2
//                  -> 1 launch (the fc QMV itself stays stock)
//   o_fused        sigmoid-gate multiply + o_proj QMV + residual add -> 1
//   down_fused     SwiGLU + down_proj QMV + residual add -> 1
//
// The two fused QMVs are verbatim copies of the running `qmv_fast_impl`
// (b4/g64 specialization, mlx-generated/quantized.cpp) with the elementwise
// producer folded into `load_vector` and the residual folded into the write.
// Per-element arithmetic, K accumulation order and `simd_sum` placement are
// identical, so each output row is bit-identical to the stock sequence.

/// Embed-row dequant (mirrors `affine_dequantize` b4) + both pre-fc RMSNorms
/// (mirrors `rms_looped` at lsize 1024, N_READS 4) + concat, in one launch.
private let qwenMTPPreFCPrepareKernel = MLXFast.metalKernel(
    name: "qwen_mtp_prefc_prepare_v1",
    inputNames: ["ids", "emb_w", "emb_s", "emb_b", "hid", "w_e", "w_h", "eps"],
    outputNames: ["fused"],
    source: """
        constexpr uint AX = 5120;
        constexpr uint TG = 1024;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;

        threadgroup float local_sums[32];
        threadgroup float inv_shared[2];
        threadgroup bfloat16_t e_row[5120];

        uint token = uint(ids[0]);
        const device uint8_t* wrow =
            ((const device uint8_t*)emb_w) + ulong(token) * ulong(AX / 2);
        const device bfloat16_t* srow = emb_s + ulong(token) * ulong(AX / 64);
        const device bfloat16_t* brow = emb_b + ulong(token) * ulong(AX / 64);
        for (uint kb = lid; kb < AX / 2; kb += TG) {
            uint8_t v = wrow[kb];
            uint g = (2 * kb) / 64;
            bfloat16_t s = srow[g];
            bfloat16_t b = brow[g];
            uint8_t d0 = v & 0x0f;
            uint8_t d1 = (v >> 4) & 0x0f;
            e_row[2 * kb] = d0 * s + b;
            e_row[2 * kb + 1] = d1 * s + b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint part = 0; part < 2; part++) {
            float acc = 0;
            for (uint r = 0; r < AX; r += TG * 4) {
                if (r + lid * 4 + 4 <= AX) {
                    for (uint i = 0; i < 4; i++) {
                        uint idx = r + lid * 4 + i;
                        float xi = (part == 0) ? float(e_row[idx]) : float(hid[idx]);
                        acc += xi * xi;
                    }
                } else {
                    for (uint i = 0; i < 4; i++) {
                        uint idx = r + lid * 4 + i;
                        if (idx < AX) {
                            float xi =
                                (part == 0) ? float(e_row[idx]) : float(hid[idx]);
                            acc += xi * xi;
                        }
                    }
                }
            }
            acc = simd_sum(acc);
            if (simd_group == 0) {
                local_sums[simd_lane] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                local_sums[simd_group] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                acc = simd_sum(local_sums[simd_lane]);
                if (simd_lane == 0) {
                    inv_shared[part] = metal::precise::rsqrt(acc / AX + eps);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float inv_e = inv_shared[0];
        float inv_h = inv_shared[1];
        for (uint k = lid; k < AX; k += TG) {
            fused[k] = w_e[k] * static_cast<bfloat16_t>(float(e_row[k]) * inv_e);
            fused[AX + k] =
                w_h[k] * static_cast<bfloat16_t>(float(hid[k]) * inv_h);
        }
        """)

/// One launch for the whole pre-fc block. Returns nil (caller uses the stock
/// chain) unless the head is the q4/g64 affine configuration.
func qwenMTPFusedPreFC(
    hidden: MLXArray, tokenIds: MLXArray, embed: Embedding,
    normE: RMSNorm, normH: RMSNorm
) -> MLXArray? {
    guard let quantEmbed = embed as? QuantizedEmbedding,
          quantEmbed.groupSize == 64, quantEmbed.bits == 4,
          quantEmbed.mode == .affine,
          let embBiases = quantEmbed.biases,
          hidden.dim(0) == 1, hidden.dim(1) == 1, hidden.dim(2) == 5120,
          hidden.dtype == .bfloat16,
          tokenIds.dtype == .int32,
          tokenIds.dim(0) == 1, tokenIds.dim(1) == 1,
          quantEmbed.weight.dim(1) == 5120 * 4 / 32,
          normE.weight.dtype == .bfloat16, normH.weight.dtype == .bfloat16,
          normE.weight.dim(0) == 5120, normH.weight.dim(0) == 5120,
          normE.eps == normH.eps
    else { return nil }
    let outputs = qwenMTPPreFCPrepareKernel(
        [tokenIds, quantEmbed.weight, quantEmbed.scales, embBiases,
         hidden, normE.weight, normH.weight, normE.eps],
        grid: (1024, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, 1, 10240]],
        outputDTypes: [.bfloat16])
    return outputs[0]
}

/// Single-row affine-4/g64 QMV with the residual add folded into the row
/// write: verbatim `qmv_fast_impl` (b4/g64 specialization of the running JIT
/// twin: same load_vector bf16 sum chain, same masked-nibble qdot, same
/// simd_sum placement), then `h_out[row] = res[row] + bf16(result)` exactly
/// where the stock kernel writes `y[row] = bf16(result)` and a separate
/// vv_Add would re-read it. The elementwise PRODUCER of x (sigmoid gate,
/// SwiGLU) deliberately stays a stock kernel: folding it in here recomputes
/// the transcendental once per weight tile, ~640x redundantly, and measured
/// +6% per step. Materialize once, read many.
private let qwenMTPQmvResidKernel = MLXFast.metalKernel(
    name: "qwen_mtp_qmv_resid_v2",
    inputNames: ["w", "scales", "biases", "x", "res"],
    outputNames: ["h_out"],
    source: """
        constexpr int packs_per_thread = 2;
        constexpr int num_simdgroups = 2;
        constexpr int results_per_simdgroup = 4;
        constexpr int pack_factor = 8;
        constexpr int bytes_per_pack = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = 512;
        constexpr int scale_step_per_thread = 4;
        constexpr int in_vec_size = IN_VEC;
        constexpr int in_vec_size_w = IN_VEC / 2;
        constexpr int in_vec_size_g = IN_VEC / 64;

        uint3 tid = threadgroup_position_in_grid;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        const device uint8_t* ws = (const device uint8_t*)w;
        typedef float U;
        thread U x_thread[values_per_thread];
        thread U result[results_per_simdgroup] = {0};

        const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
            simd_gid * results_per_simdgroup;

        ws += out_row * in_vec_size_w +
            simd_lid * packs_per_thread * bytes_per_pack;
        const device bfloat16_t* sl0 =
            scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        const device bfloat16_t* bl0 =
            biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        const device bfloat16_t* xp = x + simd_lid * values_per_thread;

        for (int k = 0; k < in_vec_size; k += block_size) {
            // load_vector mirror: bf16 sum chain, then float accumulate.
            U sum = 0;
            for (int i = 0; i < values_per_thread; i += 4) {
                sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3];
                x_thread[i] = xp[i];
                x_thread[i + 1] = xp[i + 1] / 16.0f;
                x_thread[i + 2] = xp[i + 2] / 256.0f;
                x_thread[i + 3] = xp[i + 3] / 4096.0f;
            }

            for (int row = 0; row < results_per_simdgroup; row++) {
                const device uint16_t* ws16 =
                    (const device uint16_t*)(ws + row * in_vec_size_w);
                const device bfloat16_t* sl = sl0 + row * in_vec_size_g;
                const device bfloat16_t* bl = bl0 + row * in_vec_size_g;
                U s = sl[0];
                U b = bl[0];
                U accum = 0;
                for (int i = 0; i < (values_per_thread / 4); i++) {
                    accum +=
                        (x_thread[4 * i] * (ws16[i] & 0x000f) +
                         x_thread[4 * i + 1] * (ws16[i] & 0x00f0) +
                         x_thread[4 * i + 2] * (ws16[i] & 0x0f00) +
                         x_thread[4 * i + 3] * (ws16[i] & 0xf000));
                }
                result[row] += s * accum + sum * b;
            }

            ws += block_size * bytes_per_pack / pack_factor;
            sl0 += block_size / 64;
            bl0 += block_size / 64;
            xp += block_size;
        }

        for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) {
                bfloat16_t val = static_cast<bfloat16_t>(result[row]);
                h_out[out_row + row] = res[out_row + row] + val;
            }
        }
        """)

func qwenMTPQmvResid(
    x: MLXArray, projection: QuantizedLinear, biases: MLXArray,
    residual: MLXArray, inVec: Int
) -> MLXArray {
    let outputs = qwenMTPQmvResidKernel(
        [projection.weight, projection.scales, biases, x, residual],
        template: [("IN_VEC", inVec)],
        grid: (32, 2 * (5120 / 8), 1),
        threadGroup: (32, 2, 1),
        outputShapes: [[1, 1, 5120]],
        outputDTypes: [.bfloat16])
    return outputs[0]
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
        if _qwen35MTPHeadFuseEnabled, x.dim(0) == 1, x.dim(1) == 1,
           let fused = fusedProposalForward(x, mask: mask, cache: cache)
        {
            return fused
        }
        // omlx: MTPDecoderLayer.__call__
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    /// Single-row fused step: sigmoid-gate + o_proj + residual in one launch,
    /// SwiGLU + down_proj + residual in one launch. Bit-identical to the
    /// stock chain (see the kernel headers); every guard fires BEFORE any
    /// cache mutation, so a nil return has no side effects.
    private func fusedProposalForward(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray? {
        guard let mlpFused = mlp as? Qwen35FusedMLP else { return nil }
        guard let h = selfAttn.forwardFusedProposalTail(
            inputLayerNorm(x), residual: x, mask: mask, cache: cache)
        else { return nil }
        let normed = postAttentionLayerNorm(h)
        if let out = mlpFused.fusedDownProposal(normed, residual: h) {
            return out
        }
        return h + mlpFused(normed)
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

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        //    Single-row fused form when enabled: one launch replaces the
        //    embed gather chain, both pre-fc norms and the concat copies;
        //    the fc QMV itself stays stock. Bit-identical values.
        var fused: MLXArray
        if _qwen35MTPHeadFuseEnabled,
           let pre = qwenMTPFusedPreFC(
               hidden: hidden, tokenIds: nextTokenIds, embed: embedTokens,
               normE: preFcNormEmbedding, normH: preFcNormHidden)
        {
            fused = fc(pre)
        } else {
            let embeds = embedTokens(nextTokenIds)
            let e = preFcNormEmbedding(embeds)
            let h = preFcNormHidden(hidden)
            fused = fc(concatenated([e, h], axis: -1))
        }

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
