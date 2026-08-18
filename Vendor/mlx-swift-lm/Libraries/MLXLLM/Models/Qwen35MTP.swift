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
        let (residual, mlpOut) = forwardParts(x, mask: mask, cache: cache)
        return residual + mlpOut
    }

    /// Attention/MLP halves with the residual boundary left UNMERGED, so the
    /// caller can finish the layer through the boundary-fused add+RMSNorm
    /// junction (one launch instead of a standalone add plus a standalone
    /// norm). Bit-exact against the eager sequence: the fused kernel rounds
    /// the sum to bf16 before squaring and mirrors `rms_norm.metal`'s
    /// reduction tree — the same contract the backbone's promoted chain uses.
    /// Head side — proposal-only — but the kernel is the backbone's
    /// ranked-green one, so the schedule fingerprint is unchanged anyway.
    /// Falls back to the eager expression for any non-bf16 input.
    func forwardParts(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> (residual: MLXArray, mlpOut: MLXArray) {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        if x.dtype == .bfloat16 && r.dtype == .bfloat16 {
            let junction = qwen35FusedResidualRMSNorm(
                x: x, r: r,
                weight: postAttentionLayerNorm.weight,
                eps: postAttentionLayerNorm.eps)
            return (junction.residual, (mlp as! UnaryLayer)(junction.normed))
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

// MARK: - fused pre-fc gather + dual RMSNorm + concat

/// Fused `concat([RMSNorm_e(embed(ids)), RMSNorm_h(hidden)])` in ONE launch,
/// replacing the gather, two RMSNorm, and concat dispatches that precede `fc`
/// on every head forward (four launches to one). Head side — proposal-only —
/// but the arithmetic mirrors `rms_norm.metal` exactly (same 1024-thread
/// looped read, same simd_sum → per-simd-group local_sums → simd_sum tree,
/// same precise::rsqrt, same `w * bf16(x * inv_mean)` write rounding), so the
/// fused output is bit-identical to the eager chain and the draft schedule is
/// unchanged. The gather itself is a pure row copy, exact by construction.
private let qwen35MTPPreFcFusedKernel = MLXFast.metalKernel(
    name: "qwen35_mtp_prefc_fused",
    inputNames: ["ids", "table", "hidden", "e_weight", "h_weight", "eps_e", "eps_h"],
    outputNames: ["fused"],
    source: """
        constexpr uint n_reads = 4;
        constexpr uint simd_size = 32;
        constexpr uint lsize = 1024;

        uint row = threadgroup_position_in_grid.x;
        uint thread_id = thread_position_in_threadgroup.x;
        uint simd_thread = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;

        uint axis_size = uint(hidden_shape[hidden_ndim - 1]);

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[simd_size];

        ulong tok = ulong(ids[row]);
        ulong e_offset = tok * ulong(axis_size);
        ulong h_offset = ulong(row) * ulong(axis_size);
        ulong out_row = ulong(row) * ulong(2 * axis_size);

        // ---- phase 1: RMSNorm of the gathered embedding row (left half) ----
        float acc = 0.0f;
        for (uint r_start = 0; r_start < axis_size; r_start += lsize * n_reads) {
            uint elem = r_start + thread_id * n_reads;
            if (elem + n_reads <= axis_size) {
                for (uint i = 0; i < n_reads; ++i) {
                    float xi = float(table[e_offset + elem + i]);
                    acc += xi * xi;
                }
            } else {
                for (uint i = 0; i < n_reads; ++i) {
                    if (elem + i < axis_size) {
                        float xi = float(table[e_offset + elem + i]);
                        acc += xi * xi;
                    }
                }
            }
        }
        acc = simd_sum(acc);
        if (simd_group == 0) { local_sums[simd_thread] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_thread == 0) { local_sums[simd_group] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            acc = simd_sum(local_sums[simd_thread]);
            if (simd_thread == 0) {
                local_inv_mean[0] = metal::precise::rsqrt(
                    acc / float(axis_size) + eps_e);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_mean_e = local_inv_mean[0];
        for (uint r_start = 0; r_start < axis_size; r_start += lsize * n_reads) {
            uint elem = r_start + thread_id * n_reads;
            if (elem + n_reads <= axis_size) {
                for (uint i = 0; i < n_reads; ++i) {
                    float xi = float(table[e_offset + elem + i]);
                    bfloat wi = e_weight[elem + i];
                    fused[out_row + elem + i] = wi * bfloat(xi * inv_mean_e);
                }
            } else {
                for (uint i = 0; i < n_reads; ++i) {
                    if (elem + i < axis_size) {
                        float xi = float(table[e_offset + elem + i]);
                        bfloat wi = e_weight[elem + i];
                        fused[out_row + elem + i] = wi * bfloat(xi * inv_mean_e);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- phase 2: RMSNorm of the hidden row (right half) ----
        acc = 0.0f;
        for (uint r_start = 0; r_start < axis_size; r_start += lsize * n_reads) {
            uint elem = r_start + thread_id * n_reads;
            if (elem + n_reads <= axis_size) {
                for (uint i = 0; i < n_reads; ++i) {
                    float xi = float(hidden[h_offset + elem + i]);
                    acc += xi * xi;
                }
            } else {
                for (uint i = 0; i < n_reads; ++i) {
                    if (elem + i < axis_size) {
                        float xi = float(hidden[h_offset + elem + i]);
                        acc += xi * xi;
                    }
                }
            }
        }
        acc = simd_sum(acc);
        if (simd_group == 0) { local_sums[simd_thread] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_thread == 0) { local_sums[simd_group] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            acc = simd_sum(local_sums[simd_thread]);
            if (simd_thread == 0) {
                local_inv_mean[0] = metal::precise::rsqrt(
                    acc / float(axis_size) + eps_h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_mean_h = local_inv_mean[0];
        for (uint r_start = 0; r_start < axis_size; r_start += lsize * n_reads) {
            uint elem = r_start + thread_id * n_reads;
            if (elem + n_reads <= axis_size) {
                for (uint i = 0; i < n_reads; ++i) {
                    float xi = float(hidden[h_offset + elem + i]);
                    bfloat wi = h_weight[elem + i];
                    fused[out_row + axis_size + elem + i] =
                        wi * bfloat(xi * inv_mean_h);
                }
            } else {
                for (uint i = 0; i < n_reads; ++i) {
                    if (elem + i < axis_size) {
                        float xi = float(hidden[h_offset + elem + i]);
                        bfloat wi = h_weight[elem + i];
                        fused[out_row + axis_size + elem + i] =
                            wi * bfloat(xi * inv_mean_h);
                    }
                }
            }
        }
    """,
    ensureRowContiguous: false
)

/// Fused pre-fc chain. `ids` is `[1, L]` int32, `table` the bf16 embedding
/// `[V, H]`, `hidden` `[1, L, H]` bf16; returns `[1, L, 2H]` bf16 laid out
/// exactly as `concatenated([normE(embed(ids)), normH(hidden)], axis: -1)`.
/// Returns nil for any shape/dtype outside the fused envelope so the caller
/// can fall back to the eager chain.
func qwen35MTPPreFcFused(
    ids: MLXArray,
    table: MLXArray,
    hidden: MLXArray,
    eWeight: MLXArray,
    hWeight: MLXArray,
    epsE: Float,
    epsH: Float
) -> MLXArray? {
    let H = hidden.dim(-1)
    guard hidden.dtype == .bfloat16,
          table.dtype == .bfloat16,
          (ids.dtype == .int32 || ids.dtype == .uint32 || ids.dtype == .int64),
          hidden.ndim == 3,
          hidden.dim(0) == 1,
          ids.dim(0) == 1,
          ids.size == hidden.dim(1),
          table.dim(-1) == H,
          eWeight.shape == [H],
          hWeight.shape == [H],
          eWeight.dtype == .bfloat16,
          hWeight.dtype == .bfloat16
    else { return nil }
    let L = hidden.dim(1)
    let outputs = qwen35MTPPreFcFusedKernel(
        [ids, table, hidden, eWeight, hWeight, MLXArray(epsE), MLXArray(epsH)],
        grid: (L * 1024, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, L, 2 * H]],
        outputDTypes: [.bfloat16]
    )
    return outputs[0]
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
        // 1. Embed next-token ids and fuse with normed hidden state. The
        //    gather + dual RMSNorm + concat rides one fused kernel when the
        //    shapes allow (bit-identical to the eager chain); any surprise
        //    falls back to the eager expression.
        let fusedCat = qwen35MTPPreFcFused(
            ids: nextTokenIds, table: embedTokens.weight, hidden: hidden,
            eWeight: preFcNormEmbedding.weight, hWeight: preFcNormHidden.weight,
            epsE: preFcNormEmbedding.eps, epsH: preFcNormHidden.eps)
        var fused: MLXArray
        if let fusedCat {
            fused = fc(fusedCat)
        } else {
            let embeds = embedTokens(nextTokenIds)
            let e = preFcNormEmbedding(embeds)
            let h = preFcNormHidden(hidden)
            fused = fc(concatenated([e, h], axis: -1))
        }

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run each MTPDecoderLayer. The LAST layer's exit junction rides
        //    the boundary-fused add+RMSNorm kernel (one launch instead of a
        //    standalone add plus the module norm), bit-exact against the
        //    eager `norm(h + mlpOut)` — the backbone chain's ranked-green
        //    kernel and contract. Non-bf16 or multi-layer heads keep the
        //    eager boundary.
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            if i == layers.count - 1, fused.dtype == .bfloat16 {
                let parts = layer.forwardParts(fused, mask: mask, cache: c)
                if parts.mlpOut.dtype == .bfloat16 {
                    return qwen35FusedResidualRMSNorm(
                        x: parts.residual, r: parts.mlpOut,
                        weight: norm.weight, eps: norm.eps
                    ).normed
                }
                return norm(parts.residual + parts.mlpOut)
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

        let fusedCat = qwen35MTPPreFcFused(
            ids: nextTokenIds, table: embedTokens.weight, hidden: hidden,
            eWeight: preFcNormEmbedding.weight, hWeight: preFcNormHidden.weight,
            epsE: preFcNormEmbedding.eps, epsH: preFcNormHidden.eps)
        let fused: MLXArray
        if let fusedCat {
            fused = fc(fusedCat)
        } else {
            let embeds = embedTokens(nextTokenIds)
            let e = preFcNormEmbedding(embeds)
            let h = preFcNormHidden(hidden)
            fused = fc(concatenated([e, h], axis: -1))
        }
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        let parts = layers[0].forwardParts(current, mask: mask, cache: cache[0])
        if current.dtype == .bfloat16, parts.mlpOut.dtype == .bfloat16 {
            return qwen35FusedResidualRMSNorm(
                x: parts.residual, r: parts.mlpOut,
                weight: norm.weight, eps: norm.eps
            ).normed
        }
        return norm(parts.residual + parts.mlpOut)
    }

}
