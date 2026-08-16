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
    guard let qEmb = embedTokens as? QuantizedEmbedding,
        let embBiases = qEmb.biases
    else { return nil }
    return qwen35MTPPreFuseDispatch(
        ids: nextTokenIds,
        hidden: hidden,
        embWeight: qEmb.weight,
        embScales: qEmb.scales,
        embBiases: embBiases,
        eWeight: eNorm.weight,
        hWeight: hNorm.weight,
        epsE: eNorm.eps,
        epsH: hNorm.eps)
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

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        // omlx: MTPDecoderLayer.__call__
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let normed = postAttentionLayerNorm(h)
        if let fusedTail = mtpFusedMLPTail(normed) {
            return h + fusedTail
        }
        return h + (mlp as! UnaryLayer)(normed)
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
        guard let fused = mlp as? Qwen35FusedMLP, x.dim(-2) <= 9 else { return nil }
        guard let gateUp = mtpGateUpBuffer(fused, x), _mtpGateOut > 0,
            gateUp.dtype == .bfloat16 || gateUp.dtype == .float32
        else { return nil }
        guard let activated = qwen35MTPSwiGLU(gateUp, gateOut: _mtpGateOut)
        else { return nil }
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
