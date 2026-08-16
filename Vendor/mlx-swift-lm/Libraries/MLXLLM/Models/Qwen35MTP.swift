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

// The native MTP head always begins by gathering the next-token embedding,
// RMS-normalizing that row and the paired trunk hidden independently, then
// concatenating the two normalized results. On the shipped BF16 head those
// are four materialization boundaries before the fusion projection can even
// start. This kernel writes the final `[embeddingNorm, hiddenNorm]` carrier
// directly. Its two threadgroups per logical row use the same 1024-thread,
// four-read loop and SIMD reduction tree as MLX's `rms_looped_bf16` kernel for
// Qwen's 5120-wide rows. The embedding half reads the table row directly, so
// the standalone gather and both normalized intermediates disappear as well
// as the concatenation copy.
private let qwen35MTPFusedInputKernel = MLXFast.metalKernel(
    name: "qwen35_mtp_fused_input_bf16_v1",
    inputNames: [
        "token_ids", "embedding_table", "hidden",
        "embedding_norm_weight", "hidden_norm_weight",
        "embedding_eps", "hidden_eps",
    ],
    outputNames: ["fused_input"],
    source: """
        constexpr uint n_reads = 4;
        constexpr uint simd_size = 32;
        constexpr uint threads = 1024;

        const uint pair_group = threadgroup_position_in_grid.x;
        const uint row = pair_group >> 1;
        const bool is_embedding = (pair_group & 1) == 0;
        const uint lid = thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;

        const uint sequence = uint(hidden_shape[1]);
        const uint batch = row / sequence;
        const uint position = row - batch * sequence;

        ulong input_base;
        ulong input_stride;
        if (is_embedding) {
            const ulong token_offset = ulong(batch) * ulong(token_ids_strides[0])
                + ulong(position) * ulong(token_ids_strides[1]);
            const int token = int(token_ids[token_offset]);
            input_base = ulong(token) * ulong(embedding_table_strides[0]);
            input_stride = ulong(embedding_table_strides[1]);
        } else {
            input_base = ulong(batch) * ulong(hidden_strides[0])
                + ulong(position) * ulong(hidden_strides[1]);
            input_stride = ulong(hidden_strides[2]);
        }

        const ulong weight_stride = is_embedding
            ? ulong(embedding_norm_weight_strides[0])
            : ulong(hidden_norm_weight_strides[0]);

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[simd_size];

        float acc = 0.0f;
        for (uint r = 0; r < H; r += threads * n_reads) {
            for (uint i = 0; i < n_reads; ++i) {
                const uint element = r + lid * n_reads + i;
                if (element < H) {
                    const ulong input_index =
                        input_base + ulong(element) * input_stride;
                    const float value = is_embedding
                        ? float(embedding_table[input_index])
                        : float(hidden[input_index]);
                    acc += value * value;
                }
            }
        }

        acc = simd_sum(acc);
        if (simd_group == 0) {
            local_sums[simd_lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            acc = simd_sum(local_sums[simd_lane]);
            if (simd_lane == 0) {
                const float eps = is_embedding ? embedding_eps : hidden_eps;
                local_inv_mean[0] = metal::precise::rsqrt(acc / H + eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ulong output_base = ulong(row) * ulong(2 * H)
            + (is_embedding ? ulong(0) : ulong(H));
        const float inv_mean = local_inv_mean[0];
        for (uint r = 0; r < H; r += threads * n_reads) {
            for (uint i = 0; i < n_reads; ++i) {
                const uint element = r + lid * n_reads + i;
                if (element < H) {
                    const ulong input_index =
                        input_base + ulong(element) * input_stride;
                    const bfloat input_value = is_embedding
                        ? embedding_table[input_index]
                        : hidden[input_index];
                    const bfloat rms_value = static_cast<bfloat>(
                        float(input_value) * inv_mean);
                    const bfloat weight = is_embedding
                        ? embedding_norm_weight[ulong(element) * weight_stride]
                        : hidden_norm_weight[ulong(element) * weight_stride];
                    fused_input[output_base + ulong(element)] =
                        weight * rms_value;
                }
            }
        }
    """,
    ensureRowContiguous: false
)

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

    /// Produce the exact carrier consumed by `fc`. The optimized route is
    /// deliberately narrow: the ranked BF16 head, `[B, S, H]` hidden rows,
    /// aligned int32 token IDs, and the same H for both norm weights and the
    /// embedding table. Any future dtype or layout contract keeps the original
    /// gather + two RMSNorms + concatenation expression.
    private func fusionInput(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding
    ) -> MLXArray {
        let hiddenSize = hidden.dim(-1)
        if hidden.ndim == 3,
           nextTokenIds.ndim == 2,
           nextTokenIds.dim(0) == hidden.dim(0),
           nextTokenIds.dim(1) == hidden.dim(1),
           nextTokenIds.dtype == .int32,
           hidden.dtype == .bfloat16,
           embedTokens.weight.ndim == 2,
           embedTokens.weight.dim(1) == hiddenSize,
           embedTokens.weight.dtype == .bfloat16,
           preFcNormEmbedding.weight.dim(0) == hiddenSize,
           preFcNormEmbedding.weight.dtype == .bfloat16,
           preFcNormHidden.weight.dim(0) == hiddenSize,
           preFcNormHidden.weight.dtype == .bfloat16
        {
            let rows = hidden.dim(0) * hidden.dim(1)
            let threads = 1024
            let outputs = qwen35MTPFusedInputKernel(
                [
                    nextTokenIds, embedTokens.weight, hidden,
                    preFcNormEmbedding.weight, preFcNormHidden.weight,
                    preFcNormEmbedding.eps, preFcNormHidden.eps,
                ],
                template: [("H", hiddenSize)],
                grid: (rows * 2 * threads, 1, 1),
                threadGroup: (threads, 1, 1),
                outputShapes: [[hidden.dim(0), hidden.dim(1), hiddenSize * 2]],
                outputDTypes: [.bfloat16]
            )
            return outputs[0]
        }

        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        return concatenated([e, h], axis: -1)
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        var fused = fc(fusionInput(
            hidden: hidden, nextTokenIds: nextTokenIds,
            embedTokens: embedTokens))

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

        let fused = fc(fusionInput(
            hidden: hidden, nextTokenIds: nextTokenIds,
            embedTokens: embedTokens))
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }

}
