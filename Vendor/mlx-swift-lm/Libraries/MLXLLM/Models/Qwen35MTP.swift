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

// MARK: - Head-tail fused kernels (proposal path ONLY — never on the verify path)
//
// Three custom kernels collapse the head step's small-op tail: the dual
// pre-fc RMSNorm + concat, the residual-add + RMSNorm pairs, and the SwiGLU
// epilogue. All three reproduce the stock op chains BYTE-FOR-BYTE:
// - The RMS reduction transcribes `rms_looped<bfloat16_t, 4>`
//   (rms_norm.metal:83-157) at the exact dispatch normalization.cpp selects
//   for axis 5120 (> RMS_LOOPED_LIMIT 4096): threadgroup 1024, N_READS 4,
//   per-thread fp32 accumulation at stride lsize*N_READS, simd_sum, one
//   cross-simd simd_sum, precise rsqrt of the correctly-rounded mean.
// - The SwiGLU sigmoid is the unary Sigmoid struct at T=bfloat with per-op
//   bf16 rounding and metal::precise::exp / precise::divide — MLX op kernels
//   compile WITHOUT fast math while custom kernels default to fast math, so
//   the precise:: variants are required; verified byte-exact against
//   mx.sigmoid over ALL 65,536 possible bf16 inputs.
// Validation: 45/45 byte-identity checks (S in {1,2,5,8,511} x magnitude
// scales {1e-3, 1, 30}) plus a bf16-accumulator negative control detected
// 8/8. Every call site below fails closed onto the stock chain.

private let qwen35MTPRMSBody = """
        constexpr uint N_READS = 4;
        constexpr uint SIMD_SIZE = 32;
        uint lid = thread_position_in_threadgroup.x;
        uint lsize = threads_per_threadgroup.x;
        uint simd_lane_id = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        float acc = 0.0f;
        for (uint r = 0; r < axis_size; r += lsize * N_READS) {
          if (r + lid * N_READS + N_READS <= axis_size) {
            for (uint i = 0; i < N_READS; i++) {
              float xi = float(xrow[r + lid * N_READS + i]);
              acc += xi * xi;
            }
          } else {
            for (uint i = 0; i < N_READS; i++) {
              if ((r + lid * N_READS + i) < axis_size) {
                float xi = float(xrow[r + lid * N_READS + i]);
                acc += xi * xi;
              }
            }
          }
        }
        acc = simd_sum(acc);
        if (simd_group_id == 0) {
          local_sums[simd_lane_id] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane_id == 0) {
          local_sums[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
          acc = simd_sum(local_sums[simd_lane_id]);
          if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(
                metal::precise::divide(acc, float(axis_size)) + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_mean = local_inv_mean[0];
        for (uint r = 0; r < axis_size; r += lsize * N_READS) {
          for (uint i = 0; i < N_READS; i++) {
            uint e = r + lid * N_READS + i;
            if (e < axis_size) {
              orow[e] = wvec[e] * bfloat(float(xrow[e]) * inv_mean);
            }
          }
        }
    """

private let qwen35MTPDualRMSConcatKernel = MLXFast.metalKernel(
    name: "qwen35_mtp_prologue_dualrms_v1",
    inputNames: ["e_arr", "h_arr", "w_e", "w_h", "eps_in"],
    outputNames: ["out"],
    source: """
        uint row = threadgroup_position_in_grid.y;
        uint half_index = threadgroup_position_in_grid.z;
        uint axis_size = uint(e_arr_shape[2]);
        float eps = eps_in[0];
        const device bfloat* xrow = (half_index == 0)
            ? (e_arr + ulong(row) * ulong(axis_size))
            : (h_arr + ulong(row) * ulong(axis_size));
        const device bfloat* wvec = (half_index == 0) ? w_e : w_h;
        device bfloat* orow = out
            + ulong(row) * ulong(2 * axis_size)
            + ulong(half_index) * ulong(axis_size);
    """ + qwen35MTPRMSBody
)

private let qwen35MTPAddRMSNormKernel = MLXFast.metalKernel(
    name: "qwen35_mtp_add_rmsnorm_v1",
    inputNames: ["x_arr", "r_arr", "w_n", "eps_in"],
    outputNames: ["s_out", "n_out"],
    source: """
        uint row = threadgroup_position_in_grid.y;
        uint axis_size = uint(x_arr_shape[2]);
        float eps = eps_in[0];
        uint lid0 = thread_position_in_threadgroup.x;
        uint lsize0 = threads_per_threadgroup.x;
        const device bfloat* xr = x_arr + ulong(row) * ulong(axis_size);
        const device bfloat* rr = r_arr + ulong(row) * ulong(axis_size);
        device bfloat* srow = s_out + ulong(row) * ulong(axis_size);
        for (uint e = lid0; e < axis_size; e += lsize0) {
          srow[e] = xr[e] + rr[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device bfloat* xrow = srow;
        const device bfloat* wvec = w_n;
        device bfloat* orow = n_out + ulong(row) * ulong(axis_size);
    """ + qwen35MTPRMSBody
)

let qwen35MTPSwiGLUKernel = MLXFast.metalKernel(
    name: "qwen35_mtp_swiglu_v1",
    inputNames: ["y_arr", "gate_out"],
    outputNames: ["out"],
    source: """
        uint idx = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint hdim = uint(gate_out[0]);
        uint total = uint(y_arr_shape[2]);
        if (idx >= hdim) { return; }
        ulong base = ulong(row) * ulong(total);
        bfloat g = y_arr[base + idx];
        bfloat u = y_arr[base + ulong(hdim) + idx];
        bfloat ex = bfloat(metal::precise::exp(
            float(bfloat(metal::abs(float(g))))));
        bfloat den = bfloat(1.0f + float(ex));
        bfloat yv = bfloat(metal::precise::divide(1.0f, float(den)));
        bfloat sig = (float(g) < 0.0f) ? yv : bfloat(1.0f - float(yv));
        bfloat si = g * sig;
        out[ulong(row) * ulong(hdim) + idx] = si * u;
    """
)

/// The looped-RMS transcription is exact only at the geometry the stock
/// dispatch uses threadgroup 1024 for: bf16 rows wider than
/// RMS_LOOPED_LIMIT (4096). The head's hidden size is the one shape the
/// proposal path ever offers; everything else fails closed.
private func qwen35MTPRMSFusable(_ x: MLXArray, weight: MLXArray) -> Bool {
    x.dtype == .bfloat16 && weight.dtype == .bfloat16
        && x.ndim == 3 && x.dim(0) == 1 && x.dim(2) == 5120
        && weight.ndim == 1 && weight.dim(0) == 5120
}

/// K1: `concat(rmsnorm(e, w_e), rmsnorm(h, w_h))` in one dispatch.
private func qwen35MTPDualRMSConcat(
    e: MLXArray, h: MLXArray, normE: RMSNorm, normH: RMSNorm
) -> MLXArray? {
    guard qwen35MTPRMSFusable(e, weight: normE.weight),
          qwen35MTPRMSFusable(h, weight: normH.weight),
          e.dim(1) == h.dim(1), normE.eps == normH.eps
    else { return nil }
    let S = e.dim(1)
    return qwen35MTPDualRMSConcatKernel(
        [e, h, normE.weight, normH.weight, MLXArray([normE.eps])],
        grid: (1024, S, 2),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, S, 2 * 5120]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// K2: `s = x + r; n = rmsnorm(s, w)` in one dispatch, returning both.
func qwen35MTPAddRMSNorm(
    x: MLXArray, r: MLXArray, norm: RMSNorm
) -> (sum: MLXArray, normed: MLXArray)? {
    guard qwen35MTPRMSFusable(x, weight: norm.weight),
          r.dtype == .bfloat16, r.shape == x.shape
    else { return nil }
    let S = x.dim(1)
    let outs = qwen35MTPAddRMSNormKernel(
        [x, r, norm.weight, MLXArray([norm.eps])],
        grid: (1024, S, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, S, 5120], [1, S, 5120]],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outs[0], outs[1])
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
        // omlx: MTPDecoderLayer.__call__
        let parts = callResidualParts(x, mask: mask, cache: cache)
        return parts.h + parts.m
    }

    /// Same forward as `callAsFunction` but returns the final residual pair
    /// unsummed, so a caller that immediately RMS-norms the layer output can
    /// fuse the trailing add into its norm dispatch. Attention runs exactly
    /// once whatever the inner fusion gates decide; every fused stage falls
    /// closed onto the stock expression with identical bytes (the fused
    /// kernels are byte-exact transcriptions — see the kernel block above).
    func callResidualParts(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> (h: MLXArray, m: MLXArray) {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h: MLXArray
        let ln2h: MLXArray
        if let fused = qwen35MTPAddRMSNorm(
            x: x, r: r, norm: postAttentionLayerNorm)
        {
            h = fused.sum
            ln2h = fused.normed
        } else {
            h = x + r
            ln2h = postAttentionLayerNorm(h)
        }
        let m: MLXArray
        if let fusedMLP = mlp as? Qwen35FusedMLP,
           let fm = fusedMLP.headFusedForward(ln2h)
        {
            m = fm
        } else {
            m = (mlp as! UnaryLayer)(ln2h)
        }
        return (h, m)
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
        // 1. Embed next-token ids and fuse with normed hidden state. The two
        //    pre-fc norms + concat collapse into one dispatch when eligible.
        let embeds = embedTokens(nextTokenIds)
        var fused: MLXArray
        if let packed = qwen35MTPDualRMSConcat(
            e: embeds, h: hidden,
            normE: preFcNormEmbedding, normH: preFcNormHidden)
        {
            fused = fc(packed)
        } else {
            let e = preFcNormEmbedding(embeds)
            let h = preFcNormHidden(hidden)
            fused = fc(concatenated([e, h], axis: -1))
        }

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run each MTPDecoderLayer. The last layer's trailing residual add
        //    fuses into the final norm's dispatch when eligible.
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            if i == layers.count - 1 {
                let parts = layer.callResidualParts(fused, mask: mask, cache: c)
                if let fusedTail = qwen35MTPAddRMSNorm(
                    x: parts.h, r: parts.m, norm: norm)
                {
                    return fusedTail.normed
                }
                return norm(parts.h + parts.m)
            }
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
        let fused: MLXArray
        if let packed = qwen35MTPDualRMSConcat(
            e: embeds, h: hidden,
            normE: preFcNormEmbedding, normH: preFcNormHidden)
        {
            fused = fc(packed)
        } else {
            let e = preFcNormEmbedding(embeds)
            let h = preFcNormHidden(hidden)
            fused = fc(concatenated([e, h], axis: -1))
        }
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        let parts = layers[0].callResidualParts(
            current, mask: mask, cache: cache[0])
        if let fusedTail = qwen35MTPAddRMSNorm(
            x: parts.h, r: parts.m, norm: norm)
        {
            return fusedTail.normed
        }
        return norm(parts.h + parts.m)
    }

}
