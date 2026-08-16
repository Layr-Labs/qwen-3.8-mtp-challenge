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

/// Test seam: when true, all fused head dispatches (`qwen35_mtp_pre_fuse`,
/// `qwen35_mtp_swiglu`, `qwen35_mtp_attention_gate` and both
/// `qwen35_mtp_residual_norm` sites) decline every call so
/// `Qwen35MTPModule`/`Qwen35MTPDecoderLayer` run the composed op chain the
/// fusions replace. The exactness suite flips this to compare the fused and
/// composed end-to-end outputs on the same weights, then restores it.
public nonisolated(unsafe) var _qwen35MTPFusionsDisabled: Bool = false

/// Test seam companion: counts accepted fused dispatches on the model path
/// (reset by the test), so the end-to-end exactness test can assert the
/// fused kernels actually fired, not merely that both paths agree.
public nonisolated(unsafe) var _qwen35MTPFusionStats:
    (preFuse: Int, swiglu: Int, attnGate: Int, residNorm: Int) = (0, 0, 0, 0)

// MARK: - Fused head pre-stage (embed + pre_fc norms + packed concat)

/// One dispatch for the MTP head's pre-fusion stage at decode width ([1, 1, •]):
///
///     embeds = embedTokens(nextTokenIds)          // 4-bit affine g64 gather+dequant
///     e      = preFcNormEmbedding(embeds)          // rms_loopedbfloat16, axis 5120
///     h      = preFcNormHidden(hidden)             // rms_loopedbfloat16, axis 5120
///     packed = concatenated([e, h], axis: -1)      // [1, 1, 10240]
///
/// That is SEVEN dispatches in the composed graph (three gathers, one
/// dequantize, two RMSNorms, one concat) collapsing into one. Bit-identity is
/// the contract, so every numerical sub-expression is copied from the stock
/// kernels it replaces:
///
///   * The embedding row is unpacked exactly like `affine_dequantize`
///     (quantized.metal): little-endian nibbles of the uint32 row, one byte
///     per two elements, `scale * digit + bias` in bfloat arithmetic per
///     group of 64 — the same bfloat rounding the eager gather+dequantize
///     chain produces.
///   * Each RMSNorm reproduces `rms_looped<bfloat16_t, 4>` from
///     rms_norm.metal with the dispatched thread count the host uses for
///     axis 5120 (> RMS_LOOPED_LIMIT 4096 → looped, threadgroup =
///     maxTotalThreadsPerThreadgroup = 1024 on this target): each lane sums
///     `x*x` in fp32 over 4-element strided chunks, `simd_sum`, one
///     cross-simdgroup partial, `metal::precise::rsqrt(acc / axis + eps)`,
///     and the write pass `w * bfloat(x * inv_mean)` — the same reduce
///     structure the promoted `qwen35_attention_qk_rms_rope_bf16_v1`
///     replicates for its 256-dim rows.
///
/// Threadgroup 0 computes the embedding row (dequantized values are staged
/// in threadgroup memory so the write pass reads the SAME rounded bfloat
/// values the eager composition would have materialized), threadgroup 1 the
/// hidden row; both write into the packed output in concatenation order.
private let qwen35MTPPreFuseKernel = MLXFast.metalKernel(
    name: "qwen35_mtp_pre_fuse",
    inputNames: [
        "ids", "hidden", "emb_weight", "emb_scales", "emb_biases",
        "e_weight", "h_weight", "eps_e", "eps_h",
    ],
    outputNames: ["packed"],
    source: """
        constexpr uint n_reads = 4;
        constexpr uint simd_size = 32;
        constexpr uint axis_size = 5120;
        constexpr uint thread_count = 1024;

        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_thread = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[simd_size];
        threadgroup bfloat embed_values[axis_size];

        uint id = uint(ids[0]);
        uint vocab_rows = uint(emb_weight_shape[0]);
        if (id >= vocab_rows) {
            id = vocab_rows - 1;
        }
        ulong w_row = ulong(id) * ulong(axis_size / 8);
        ulong s_row = ulong(id) * ulong(axis_size / 64);

        float acc = 0.0f;
        for (uint r = 0; r < axis_size; r += thread_count * n_reads) {
            uint base = r + lid * n_reads;
            if (base + n_reads <= axis_size) {
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = base + i;
                    float xv;
                    if (row == 0) {
                        uint word = emb_weight[w_row + (element >> 3)];
                        uint digit = (word >> (4 * (element & 7))) & 0xF;
                        bfloat scale = emb_scales[s_row + (element >> 6)];
                        bfloat bias = emb_biases[s_row + (element >> 6)];
                        bfloat value = scale * digit + bias;
                        embed_values[element] = value;
                        xv = float(value);
                    } else {
                        xv = float(hidden[element]);
                    }
                    acc += xv * xv;
                }
            } else {
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = base + i;
                    if (element < axis_size) {
                        float xv;
                        if (row == 0) {
                            uint word = emb_weight[w_row + (element >> 3)];
                            uint digit = (word >> (4 * (element & 7))) & 0xF;
                            bfloat scale = emb_scales[s_row + (element >> 6)];
                            bfloat bias = emb_biases[s_row + (element >> 6)];
                            bfloat value = scale * digit + bias;
                            embed_values[element] = value;
                            xv = float(value);
                        } else {
                            xv = float(hidden[element]);
                        }
                        acc += xv * xv;
                    }
                }
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
                float eps = row == 0 ? eps_e : eps_h;
                local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float inv_mean = local_inv_mean[0];
        for (uint r = 0; r < axis_size; r += thread_count * n_reads) {
            uint base = r + lid * n_reads;
            if (base + n_reads <= axis_size) {
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = base + i;
                    bfloat xv = row == 0 ? embed_values[element] : hidden[element];
                    bfloat w = row == 0 ? e_weight[element] : h_weight[element];
                    packed[row * axis_size + element] = w * bfloat(xv * inv_mean);
                }
            } else {
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = base + i;
                    if (element < axis_size) {
                        bfloat xv = row == 0 ? embed_values[element] : hidden[element];
                        bfloat w = row == 0 ? e_weight[element] : h_weight[element];
                        packed[row * axis_size + element] = w * bfloat(xv * inv_mean);
                    }
                }
            }
        }
    """
)

/// Decode-width fused pre-stage: `concatenated([preFcNormEmbedding
/// (embedTokens(nextTokenIds)), preFcNormHidden(hidden)])` in ONE dispatch.
/// Returns nil — caller keeps the composed path — whenever any contract of
/// the fused kernel is not met (non-decode width, non-bf16 hidden, a plain
/// or differently quantized embedding table, unexpected table geometry).
/// Every accepted call is bit-identical to the composed path by
/// construction (see the kernel note above).
func qwen35MTPPreFused(
    nextTokenIds: MLXArray,
    hidden: MLXArray,
    embedTokens: Embedding,
    eNorm: RMSNorm,
    hNorm: RMSNorm
) -> MLXArray? {
    guard !_qwen35MTPFusionsDisabled,
        let qEmb = embedTokens as? QuantizedEmbedding,
        let embBiases = qEmb.biases
    else { return nil }
    let out = qwen35MTPPreFuseDispatch(
        ids: nextTokenIds,
        hidden: hidden,
        embWeight: qEmb.weight,
        embScales: qEmb.scales,
        embBiases: embBiases,
        eWeight: eNorm.weight,
        hWeight: hNorm.weight,
        epsE: eNorm.eps,
        epsH: hNorm.eps)
    if out != nil { _qwen35MTPFusionStats.preFuse += 1 }
    return out
}

/// Raw gated dispatch behind `qwen35MTPPreFused`, also reachable from the
/// test bridge. Geometry is pinned to the head contract: 4-bit affine
/// group-64 embedding rows of 5120 elements (packed uint32 `[V, 640]`,
/// bf16 scales/biases `[V, 80]`).
func qwen35MTPPreFuseDispatch(
    ids: MLXArray,
    hidden: MLXArray,
    embWeight: MLXArray,
    embScales: MLXArray,
    embBiases: MLXArray,
    eWeight: MLXArray,
    hWeight: MLXArray,
    epsE: Float,
    epsH: Float
) -> MLXArray? {
    let axis = 5120
    guard ids.ndim <= 2, ids.size == 1,
        ids.dtype == .int32 || ids.dtype == .uint32,
        hidden.dtype == .bfloat16,
        hidden.shape == [1, 1, axis],
        embWeight.dtype == .uint32, embWeight.ndim == 2,
        embWeight.dim(1) * 8 == axis,
        embScales.dtype == .bfloat16, embScales.ndim == 2,
        embScales.dim(0) == embWeight.dim(0), embScales.dim(1) * 64 == axis,
        embBiases.dtype == .bfloat16, embBiases.shape == embScales.shape,
        eWeight.dtype == .bfloat16, eWeight.shape == [axis],
        hWeight.dtype == .bfloat16, hWeight.shape == [axis]
    else { return nil }
    let outputs = qwen35MTPPreFuseKernel(
        [
            ids, hidden, embWeight, embScales, embBiases,
            eWeight, hWeight, epsE, epsH,
        ],
        grid: (2 * 1024, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, 1, 2 * axis]],
        outputDTypes: [.bfloat16],
        stream: .default
    )
    return outputs[0]
}

// MARK: - Fused SwiGLU tail (silu(gate) * up in one dispatch)

/// One dispatch for `silu(g) * u` over the fused gate/up buffer, replacing
/// the compiled-SiLU launch plus the binary-mul launch. Dtype-generic: the
/// kernel is templated on the element type so a bf16 gemv buffer (the
/// ranked head) and an fp32 buffer each get their stock rounding.
///
/// Bit-identity contract: every sub-expression is the stock one it replaces,
/// in the stock order. MLXNN's `silu` is `x * sigmoid(x)` and the MLX
/// `Sigmoid` op (unary_ops.h) is
/// `y = 1 / (1 + metal::exp(metal::abs(x))); x < 0 ? y : 1 - y` evaluated at
/// the element dtype; the two multiplies are the stock elementwise
/// `Multiply`, with the silu result rounded to the element type before the
/// final multiply exactly as the two-dispatch composition materializes it.
private let qwen35MTPSwiGLUKernel = MLXFast.metalKernel(
    name: "qwen35_mtp_swiglu",
    inputNames: ["gu"],
    outputNames: ["out"],
    source: """
        uint index = thread_position_in_grid.x;
        uint row = index / uint(GATE_OUT);
        uint column = index - row * uint(GATE_OUT);
        uint base = row * 2 * uint(GATE_OUT) + column;
        T g = gu[base];
        T u = gu[base + uint(GATE_OUT)];
        T y = 1 / (1 + metal::exp(metal::abs(g)));
        T s = (g < 0) ? y : 1 - y;
        out[index] = (g * s) * u;
    """
)

/// Gated fused `silu(gate) * up`: `gateUp` is the `[B, L, 2 * gateOut]`
/// gate/up gemv buffer (gate rows first, matching the fusedGateUp concat
/// order). Returns nil for non-bf16/fp32 or mis-shaped inputs so the caller
/// keeps the composed path.
func qwen35MTPSwiGLU(_ gateUp: MLXArray, gateOut: Int) -> MLXArray? {
    guard gateOut > 0,
        gateUp.ndim == 3,
        gateUp.dim(-1) == 2 * gateOut
    else { return nil }
    let dtype = gateUp.dtype
    guard dtype == .bfloat16 || dtype == .float32 else { return nil }
    let batch = gateUp.dim(0)
    let rows = gateUp.dim(1)
    let total = batch * rows * gateOut
    let outputs = qwen35MTPSwiGLUKernel(
        [gateUp],
        template: [("T", dtype), ("GATE_OUT", gateOut)],
        grid: (total, 1, 1),
        threadGroup: (min(total, 512), 1, 1),
        outputShapes: [[batch, rows, gateOut]],
        outputDTypes: [dtype],
        stream: .default
    )
    return outputs[0]
}

// MARK: - Fused attention output gate (x * sigmoid(gate) in one dispatch)

/// One dispatch for the attention output gate `sigmoidMultiply(output,
/// gate) = output * sigmoid(gate)`, replacing the unary-Sigmoid launch plus
/// the binary-mul launch in the gated full-attention tail (Qwen3-next-style
/// gated attention, `Qwen35Attention.callAsFunction`). Dtype-generic like
/// `qwen35_mtp_swiglu`: the ranked head is bf16 ([1, 1, 6144] = 24 heads x
/// 256), fp32 assets get the same formula at fp32.
///
/// Bit-identity contract: the MLX `Sigmoid` op (unary_ops.h) is
/// `y = 1 / (1 + metal::exp(metal::abs(x))); x < 0 ? y : 1 - y` evaluated at
/// the ELEMENT dtype (bf16_math.h routes abs/exp through the bfloat
/// overloads, so every intermediate is bfloat for bf16 input), and the stock
/// elementwise `Multiply` multiplies the rounded sigmoid result by `x` at
/// the element dtype. The kernel reproduces both sub-expressions in the
/// stock order with the stock rounding points.
private let qwen35MTPAttentionGateKernel = MLXFast.metalKernel(
    name: "qwen35_mtp_attention_gate",
    inputNames: ["x", "gate"],
    outputNames: ["out"],
    source: """
        uint index = thread_position_in_grid.x;
        T xv = x[index];
        T gv = gate[index];
        T y = 1 / (1 + metal::exp(metal::abs(gv)));
        T s = (gv < 0) ? y : 1 - y;
        out[index] = xv * s;
    """
)

/// Gated fused `x * sigmoid(gate)`: nil for mismatched shapes/dtypes or a
/// dtype outside the bf16/fp32 contract, so the caller keeps the composed
/// `sigmoidMultiply` pair.
func qwen35MTPAttentionGate(_ x: MLXArray, _ gate: MLXArray) -> MLXArray? {
    guard !x.shape.isEmpty, x.shape == gate.shape else { return nil }
    let dtype = x.dtype
    guard dtype == .bfloat16 || dtype == .float32,
        gate.dtype == dtype
    else { return nil }
    let total = x.size
    guard total > 0 else { return nil }
    let outputs = qwen35MTPAttentionGateKernel(
        [x, gate],
        template: [("T", dtype)],
        grid: (total, 1, 1),
        threadGroup: (min(total, 512), 1, 1),
        outputShapes: [x.shape],
        outputDTypes: [dtype],
        stream: .default
    )
    return outputs[0]
}

// MARK: - Fused residual+norm (add then RMSNorm in one dispatch)

/// One dispatch for the head layer's residual+norm pairs at decode width
/// ([1, 1, 5120]): `h = x + r` followed by `rmsNorm(h, weight, eps)`,
/// replacing the binary-add launch plus the RMSNorm launch. Both outputs are
/// materialized: the residual (consumed by the next residual site) and the
/// normalized row.
///
/// Bit-identity contract: the stock add is the elementwise `Add` at the
/// element dtype, and the stock RMS reduce is `rms_looped<T, 4>` from
/// rms_norm.metal (axis 5120 > RMS_LOOPED_LIMIT 4096 -> looped, threadgroup
/// = maxTotalThreadsPerThreadgroup = 1024 on this target): each lane sums
/// `x*x` in fp32 over 4-element strided chunks off the ROUNDED residual
/// values, `simd_sum`, one cross-simdgroup partial,
/// `metal::precise::rsqrt(acc / axis + eps)`, and the write pass
/// `w * T(float(h) * inv_mean)`. The fused kernel stages the rounded
/// residual in threadgroup memory so BOTH passes (squares and the write)
/// read the same values the eager composition materializes — identical to
/// the staging argument the pre-fuse kernel makes for its embedding row.
private let qwen35MTPResidualNormKernel = MLXFast.metalKernel(
    name: "qwen35_mtp_residual_norm",
    inputNames: ["x", "r", "w", "eps"],
    outputNames: ["h", "normed"],
    source: """
        constexpr uint n_reads = 4;
        constexpr uint simd_size = 32;
        constexpr uint axis_size = 5120;
        constexpr uint thread_count = 1024;

        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_thread = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[simd_size];
        threadgroup T staged[axis_size];

        float acc = 0.0f;
        for (uint base_r = 0; base_r < axis_size; base_r += thread_count * n_reads) {
            uint base = base_r + lid * n_reads;
            if (base + n_reads <= axis_size) {
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = base + i;
                    ulong index = ulong(row) * ulong(axis_size) + ulong(element);
                    T hv = x[index] + r[index];
                    staged[element] = hv;
                    float xv = float(hv);
                    acc += xv * xv;
                }
            } else {
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = base + i;
                    if (element < axis_size) {
                        ulong index = ulong(row) * ulong(axis_size) + ulong(element);
                        T hv = x[index] + r[index];
                        staged[element] = hv;
                        float xv = float(hv);
                        acc += xv * xv;
                    }
                }
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
                local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float inv_mean = local_inv_mean[0];
        for (uint base_r = 0; base_r < axis_size; base_r += thread_count * n_reads) {
            uint base = base_r + lid * n_reads;
            if (base + n_reads <= axis_size) {
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = base + i;
                    ulong index = ulong(row) * ulong(axis_size) + ulong(element);
                    T hv = staged[element];
                    h[index] = hv;
                    normed[index] = w[element] * T(float(hv) * inv_mean);
                }
            } else {
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = base + i;
                    if (element < axis_size) {
                        ulong index = ulong(row) * ulong(axis_size) + ulong(element);
                        T hv = staged[element];
                        h[index] = hv;
                        normed[index] = w[element] * T(float(hv) * inv_mean);
                    }
                }
            }
        }
    """
)

/// Raw gated dispatch behind the layer's residual+norm sites, also reachable
/// from the test bridge. Geometry pinned to the head contract: flat rows of
/// 5120 elements (axis > RMS_LOOPED_LIMIT selects the looped stock kernel
/// the body replicates), one `[B, L, 5120]` leading-dim grid. Returns both
/// the residual and the normalized row; nil outside the bf16/fp32 contract.
func qwen35MTPResidualNormDispatch(
    x: MLXArray,
    r: MLXArray,
    weight: MLXArray,
    eps: Float
) -> (h: MLXArray, normed: MLXArray)? {
    let axis = 5120
    guard x.ndim == 3, x.dim(0) == 1, x.dim(-1) == axis,
        r.shape == x.shape,
        weight.shape == [axis]
    else { return nil }
    let dtype = x.dtype
    guard dtype == .bfloat16 || dtype == .float32,
        r.dtype == dtype, weight.dtype == dtype
    else { return nil }
    let rows = x.size / axis
    let outputs = qwen35MTPResidualNormKernel(
        [x, r, weight, eps],
        template: [("T", dtype)],
        grid: (rows * 1024, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [x.shape, x.shape],
        outputDTypes: [dtype, dtype],
        stream: .default
    )
    return (outputs[0], outputs[1])
}

/// `h = x + r; (h, rmsNorm(h, weight, eps))` from an `RMSNorm` module in one
/// dispatch — nil outside the fused contract.
func qwen35MTPResidualNorm(
    _ x: MLXArray, _ r: MLXArray, norm: RMSNorm
) -> (h: MLXArray, normed: MLXArray)? {
    qwen35MTPResidualNormDispatch(x: x, r: r, weight: norm.weight, eps: norm.eps)
}

// MARK: - Test bridge

/// Public surface for the exactness suite (Tests/MLXFastTests/
/// Qwen35MTPFusedTests.swift): exposes the GATED head fusions so the tests
/// exercise the identical dispatch the model path wires in. Both return nil
/// outside the fused contract, mirroring the production fallback.
public enum Qwen35MTPFusionBridge {
    /// `concatenated([rmsNorm(embeddingGather(ids)), rmsNorm(hidden)])` in
    /// one dispatch — nil unless the same gates the model applies hold.
    public static func preFused(
        ids: MLXArray,
        hidden: MLXArray,
        embWeight: MLXArray,
        embScales: MLXArray,
        embBiases: MLXArray,
        eWeight: MLXArray,
        hWeight: MLXArray,
        epsE: Float,
        epsH: Float
    ) -> MLXArray? {
        qwen35MTPPreFuseDispatch(
            ids: ids,
            hidden: hidden,
            embWeight: embWeight,
            embScales: embScales,
            embBiases: embBiases,
            eWeight: eWeight,
            hWeight: hWeight,
            epsE: epsE,
            epsH: epsH)
    }

    /// `silu(gate) * up` over the fused gate/up buffer in one dispatch —
    /// nil outside the bf16/fp32 contract.
    public static func swiglu(_ gateUp: MLXArray, gateOut: Int) -> MLXArray? {
        qwen35MTPSwiGLU(gateUp, gateOut: gateOut)
    }

    /// `x * sigmoid(gate)` (the attention output gate) in one dispatch —
    /// nil outside the bf16/fp32 contract.
    public static func attentionGate(_ x: MLXArray, _ gate: MLXArray) -> MLXArray? {
        qwen35MTPAttentionGate(x, gate)
    }

    /// `h = x + r; (h, rmsNorm(h, weight, eps))` — residual and normalized
    /// row in one dispatch, nil outside the fused contract.
    public static func residualNorm(
        _ x: MLXArray,
        _ r: MLXArray,
        weight: MLXArray,
        eps: Float
    ) -> (h: MLXArray, normed: MLXArray)? {
        qwen35MTPResidualNormDispatch(x: x, r: r, weight: weight, eps: eps)
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

    // MTP-head-only fused gate/up pack for the SwiGLU tail. Mirrors
    // `Qwen35FusedMLP.fusedGateUp` construction verbatim (same concat-on-N
    // contiguous pack feeding ONE gemv) so the gate/up buffer is bit-equal
    // to what the composed path produces; `Qwen35FusedMLP.fusedGateUp`
    // stays untouched (it is shared with the backbone) and this pack is
    // built lazily on first fused decode call.
    private var _mtpGuW: MLXArray?
    private var _mtpGuS: MLXArray?
    private var _mtpGuZ: MLXArray?
    private var _mtpGuGS = 64
    private var _mtpGuBits = 4
    private var _mtpGuMode = QuantizationMode.affine
    private var _mtpBfGuW: MLXArray?
    private var _mtpGateOut = 0

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

    // MTP-head-only packed Q/K/V concat ON N for the fused attention
    // pass-through, built lazily from this layer's OWN attention module
    // exactly as `Qwen35Attention.qkv` builds its private pack (same
    // concat-on-N argument: rows are independent, so the packed gemv is
    // bit-equal to three separate calls — and to the backbone's pack).
    private var _mtpaQkvW: MLXArray?
    private var _mtpaQkvS: MLXArray?
    private var _mtpaQkvZ: MLXArray?
    private var _mtpaQkvGS = 64
    private var _mtpaQkvBits = 4
    private var _mtpaQkvMode = QuantizationMode.affine
    private var _mtpaQkvQOut = 0
    private var _mtpaQkvKOut = 0
    private var _mtpaQkvDenseW: MLXArray?

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        // omlx: MTPDecoderLayer.__call__
        let stage = forward(x, mask: mask, cache: cache)
        return stage.h + stage.mlpOut
    }

    /// Variant for the LAST head layer: the MLP residual and the module's
    /// final `mtp.norm` collapse into ONE `qwen35_mtp_residual_norm`
    /// dispatch. Returns nil only BEFORE any cache-mutating work (fusion
    /// disabled, or x outside the [1, 1, 5120] bf16 decode contract) — after
    /// the forward runs, a declined fused helper falls back to the composed
    /// add+norm over the SAME outputs, never to a second forward (cache
    /// state would double-append).
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?,
        finalNorm: RMSNorm
    ) -> MLXArray? {
        guard !_qwen35MTPFusionsDisabled,
            x.dtype == .bfloat16, x.shape == [1, 1, 5120]
        else { return nil }
        let stage = forward(x, mask: mask, cache: cache)
        if let pair = qwen35MTPResidualNorm(stage.h, stage.mlpOut, norm: finalNorm) {
            _qwen35MTPFusionStats.residNorm += 1
            return pair.normed
        }
        return finalNorm(stage.h + stage.mlpOut)
    }

    /// The layer forward split at the MLP residual so both call shapes can
    /// fuse the trailing add+norm. `h` is the post-attention residual,
    /// `mlpOut` the MLP output over the post-attention norm.
    private func forward(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> (h: MLXArray, mlpOut: MLXArray) {
        let attnIn = inputLayerNorm(x)
        let r = mtpAttention(attnIn, mask: mask, cache: cache)
            ?? selfAttn(attnIn, mask: mask, cache: cache)
        let h: MLXArray
        let normed: MLXArray
        // Post-attention residual + norm in ONE dispatch (site A). Gated to
        // the decode contract the kernel pins: [1, 1, 5120] bf16.
        if !_qwen35MTPFusionsDisabled,
            x.dtype == .bfloat16, r.dtype == .bfloat16,
            x.shape == [1, 1, 5120], r.shape == x.shape,
            let pair = qwen35MTPResidualNorm(x, r, norm: postAttentionLayerNorm)
        {
            h = pair.h
            normed = pair.normed
            _qwen35MTPFusionStats.residNorm += 1
        } else {
            h = x + r
            normed = postAttentionLayerNorm(h)
        }
        if let fusedTail = mtpFusedMLPTail(normed) {
            return (h, fusedTail)
        }
        return (h, (mlp as! UnaryLayer)(normed))
    }

    /// The packed Q/K/V gemv for the fused attention pass-through, built
    /// bit-equal to `Qwen35Attention.qkv`'s private pack. Returns nil where
    /// that helper itself would fall through to per-projection calls —
    /// callers then keep the untouched module path.
    private func mtpQKV(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray)? {
        let attn = selfAttn
        if let w = _mtpaQkvW, let s = _mtpaQkvS, let z = _mtpaQkvZ {
            let y = quantizedMM(
                x, w, scales: s, biases: z, transpose: true,
                groupSize: _mtpaQkvGS, bits: _mtpaQkvBits, mode: _mtpaQkvMode)
            let qEnd = _mtpaQkvQOut
            let kEnd = _mtpaQkvQOut + _mtpaQkvKOut
            return (y[.ellipsis, ..<qEnd], y[.ellipsis, qEnd ..< kEnd], y[.ellipsis, kEnd...])
        }
        if let w = _mtpaQkvDenseW {
            let y = matmul(x, w.transposed(1, 0))
            let qEnd = _mtpaQkvQOut
            let kEnd = _mtpaQkvQOut + _mtpaQkvKOut
            return (y[.ellipsis, ..<qEnd], y[.ellipsis, qEnd ..< kEnd], y[.ellipsis, kEnd...])
        }
        if let q = attn.qProj as? QuantizedLinear,
            let k = attn.kProj as? QuantizedLinear,
            let v = attn.vProj as? QuantizedLinear,
            q.groupSize == k.groupSize, k.groupSize == v.groupSize,
            q.bits == k.bits, k.bits == v.bits,
            q.mode == k.mode, q.mode == .affine,
            let qz = q.biases, let kz = k.biases, let vz = v.biases
        {
            _mtpaQkvW = concatenated([q.weight, k.weight, v.weight], axis: 0).contiguous()
            _mtpaQkvS = concatenated([q.scales, k.scales, v.scales], axis: 0).contiguous()
            _mtpaQkvZ = concatenated([qz, kz, vz], axis: 0).contiguous()
            _mtpaQkvGS = q.groupSize
            _mtpaQkvBits = q.bits
            _mtpaQkvMode = q.mode
            _mtpaQkvQOut = q.shape.0
            _mtpaQkvKOut = k.shape.0
            return mtpQKV(x)
        }
        if !(attn.qProj is QuantizedLinear), !(attn.kProj is QuantizedLinear),
            !(attn.vProj is QuantizedLinear),
            attn.qProj.bias == nil, attn.kProj.bias == nil, attn.vProj.bias == nil
        {
            _mtpaQkvDenseW = concatenated(
                [attn.qProj.weight, attn.kProj.weight, attn.vProj.weight], axis: 0
            ).contiguous()
            _mtpaQkvQOut = attn.qProj.weight.dim(0)
            _mtpaQkvKOut = attn.kProj.weight.dim(0)
            return mtpQKV(x)
        }
        return nil
    }

    /// MTP-owned attention pass-through: `Qwen35Attention.callAsFunction`
    /// verbatim (own packed Q/K/V gemv, same fused Q/K norm+RoPE kernel or
    /// its composed fallback with the same gate expression, same SDPA-with-
    /// cache-update), with the output gate `output * sigmoid(gate)` fused
    /// into ONE `qwen35_mtp_attention_gate` dispatch. Returns nil only
    /// before any cache mutation; a declined gate kernel after the SDPA
    /// falls back to the composed `sigmoidMultiply` over the same tensors
    /// rather than re-running attention.
    private func mtpAttention(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray? {
        guard !_qwen35MTPFusionsDisabled else { return nil }
        let attn = selfAttn
        let B = x.dim(0)
        let L = x.dim(1)
        // Decode-width contract: the fused gate kernel contract is bf16 rows
        // of 24 heads x 256 (the output/gate row is [1, 1, 6144]). Wider
        // flush/prefill rows keep the untouched module path.
        guard B == 1, L == 1, x.dtype == .bfloat16 else { return nil }
        guard let (qProjOutput, keysIn, valuesIn) = mtpQKV(x) else { return nil }
        let qSplit = qProjOutput.reshaped(B, L, attn.attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = keysIn.reshaped(B, L, attn.kvHeads, -1)
        let values = valuesIn.reshaped(B, L, attn.kvHeads, -1).transposed(0, 2, 1, 3)

        let hasArrayOffset = cache is CompilableRotatingKVCache
            || cache is CompilableKVCache
            || cache is BatchPositionedKVCache
        if attn.usesFusedQKPreparation,
            !hasArrayOffset,
            queries.dtype == .bfloat16,
            keys.dtype == .bfloat16,
            attn.qNorm.weight.dtype == .bfloat16,
            attn.kNorm.weight.dtype == .bfloat16,
            queries.shape == [B, L, attn.attentionHeads, attn.headDim],
            keys.shape == [B, L, attn.kvHeads, attn.headDim],
            attn.qNorm.weight.shape == [attn.headDim],
            attn.kNorm.weight.shape == [attn.headDim],
            attn.qNorm.eps == attn.kNorm.eps
        {
            let prepared = qwen35AttentionQKRMSRoPE(
                queries: queries,
                keys: keys,
                qWeight: attn.qNorm.weight,
                kWeight: attn.kNorm.weight,
                eps: attn.qNorm.eps,
                offset: cache?.offset ?? 0,
                log2Base: attn.ropeLog2Base
            )
            queries = prepared.queries
            keys = prepared.keys
        } else {
            queries = attn.qNorm(queries).transposed(0, 2, 1, 3)
            keys = attn.kNorm(keys).transposed(0, 2, 1, 3)
            queries = applyRotaryPosition(attn.rope, to: queries, cache: cache)
            keys = applyRotaryPosition(attn.rope, to: keys, cache: cache)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: attn.scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        if let gated = qwen35MTPAttentionGate(output, gate) {
            _qwen35MTPFusionStats.attnGate += 1
            return attn.oProj(gated)
        }
        return attn.oProj(sigmoidMultiply(output, gate))
    }

    /// The gate/up gemv buffer for the fused SwiGLU tail, bit-equal to what
    /// `Qwen35FusedMLP.callAsFunction` computes internally (same
    /// concat-on-N pack, same quantized or dense gemv). Returns nil for the
    /// mixed quantized/dense combination `Qwen35FusedMLP` itself declines.
    private func mtpGateUpBuffer(_ f: Qwen35FusedMLP, _ x: MLXArray) -> MLXArray? {
        if let w = _mtpGuW, let s = _mtpGuS, let z = _mtpGuZ {
            return quantizedMM(
                x, w, scales: s, biases: z, transpose: true,
                groupSize: _mtpGuGS, bits: _mtpGuBits, mode: _mtpGuMode)
        }
        if let w = _mtpBfGuW {
            return matmul(x, w.T)
        }
        if let g = f.gateProj as? QuantizedLinear,
           let u = f.upProj as? QuantizedLinear,
           g.groupSize == u.groupSize, g.bits == u.bits,
           g.mode == u.mode, g.mode == .affine,
           let gz = g.biases, let uz = u.biases
        {
            _mtpGuW = concatenated([g.weight, u.weight], axis: 0).contiguous()
            _mtpGuS = concatenated([g.scales, u.scales], axis: 0).contiguous()
            _mtpGuZ = concatenated([gz, uz], axis: 0).contiguous()
            _mtpGuGS = g.groupSize
            _mtpGuBits = g.bits
            _mtpGuMode = g.mode
            _mtpGateOut = g.shape.0
            return mtpGateUpBuffer(f, x)
        }
        if !(f.gateProj is QuantizedLinear), !(f.upProj is QuantizedLinear) {
            _mtpBfGuW = concatenated([f.gateProj.weight, f.upProj.weight], axis: 0)
                .contiguous()
            _mtpGateOut = f.gateProj.weight.dim(0)
            return mtpGateUpBuffer(f, x)
        }
        return nil
    }

    /// MTP-head-only fused MLP forward at decode width: one gate/up gemv,
    /// one `qwen35_mtp_swiglu` dispatch, one down projection — replacing the
    /// composed `downProj(silu(g) * u)` whose silu and multiply ride two
    /// further launches on top of the same gemv. The gemv and the down
    /// projection keep their exact stock dispatches; only the silu/mul pair
    /// is fused, and that fusion is bit-identical to the two-op composition
    /// per dtype (see the kernel note). Gate mirrors
    /// `Qwen35FusedMLP.callAsFunction` (`x.dim(-2) <= 9` plus a usable
    /// gate/up pack); anything else keeps the untouched module path,
    /// including prefill widths and MoE heads.
    private func mtpFusedMLPTail(_ x: MLXArray) -> MLXArray? {
        guard !_qwen35MTPFusionsDisabled,
            let fused = mlp as? Qwen35FusedMLP, x.dim(-2) <= 9
        else { return nil }
        guard let gateUp = mtpGateUpBuffer(fused, x), _mtpGateOut > 0,
            gateUp.dtype == .bfloat16 || gateUp.dtype == .float32
        else { return nil }
        guard let activated = qwen35MTPSwiGLU(gateUp, gateOut: _mtpGateOut)
        else { return nil }
        _qwen35MTPFusionStats.swiglu += 1
        return fused.downProj(activated)
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
        // 1. Embed next-token ids and fuse with normed hidden state. At
        //    decode width ([1, 1, •]) the gather, both pre_fc RMSNorms and
        //    the concat ride ONE bit-identical dispatch; wider flushes keep
        //    the composed path.
        let packed: MLXArray
        if let preFused = qwen35MTPPreFused(
            nextTokenIds: nextTokenIds, hidden: hidden,
            embedTokens: embedTokens,
            eNorm: preFcNormEmbedding, hNorm: preFcNormHidden)
        {
            packed = preFused
        } else {
            let embeds = embedTokens(nextTokenIds)
            let e = preFcNormEmbedding(embeds)
            let h = preFcNormHidden(hidden)
            packed = concatenated([e, h], axis: -1)
        }
        var fused = fc(packed)

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run each MTPDecoderLayer. The LAST layer also owns the MLP
        //    residual + `mtp.norm` pair: at decode width both collapse into
        //    ONE qwen35_mtp_residual_norm dispatch (site B). Off-contract
        //    shapes keep the composed `norm(layer(x))` chain.
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            if i == layers.count - 1,
                let normedFinal = layer(fused, mask: mask, cache: c, finalNorm: norm)
            {
                return normedFinal
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
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        let fused = fc(concatenated([e, h], axis: -1))
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        // Last layer at decode width: residual + `mtp.norm` in ONE dispatch.
        if let normedFinal = layers[0](
            current, mask: mask, cache: cache[0], finalNorm: norm)
        {
            return normedFinal
        }
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }

}
