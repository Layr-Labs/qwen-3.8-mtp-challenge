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

// MARK: - fused MTP draft tail (K1/K2/K3)
//
// PROPOSAL SIDE ONLY (the fused compact-draft-selection argument in
// Qwen35.swift): nothing below can reach an emitted token or a ledger value.
// Three launches replace the head decoder layer's post-attention tail at
// B == 1, L == 1 -- K1 h1 = resid + o_proj(gated attention); K2 act =
// silu(g) * u with [g; u] = gate_up(rmsnorm(h1)) over the _fqW pack; K3
// h2 = h1 + down_proj(act) -- cutting ~9 graph nodes per draft step.
// EXACTNESS: expected BYTE-EQUAL to the stock chain. The vendored MLX
// JIT-compiles the stock kernels from source with the same compile options,
// so cloning their expressions term for term gives identical bits (QMV
// header below; K2's rms_looped replay; the mixer-swept in-kernel silu).
// The attention gate's sigmoidMultiply is NOT folded (1-ulp knife-edge);
// K1 consumes the graph-side output unchanged. Off-envelope falls back to
// the stock chain verbatim, and warmAllDepths dispatches the exact live
// expression on whichever route the gate selects (7b33621).

/// Same-binary kill-switch: `MLXFAST_QWEN_MTP_FUSED_TAIL=0` restores the
/// stock decoder-layer chain for A/B parity and regression bisection.
private let qwen35MTPFusedTailEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXFAST_QWEN_MTP_FUSED_TAIL"] != "0"

/// K2 gate rows per threadgroup; the A/B sweep candidates {8 (stock-geometry
/// twin), 64, 128} all divide the 17408-row gate half.
private let qwen35MTPFusedTailK2Rows = 64

// Exact-order clone of the stock `qmv_fast_impl` lane arithmetic
// (affine-4/g64, mlx quantized.h): 32 lanes own 16 consecutive K elements,
// pre-scaled 1, /16, /256, /4096 with the raw-value `sum` kept for the bias
// term, ONE `scale * accum + sum * bias` term per 512-element block,
// ascending k, one caller-side `simd_sum` per output element. The x pointer
// of `qwen_mtp_qmv4` is templated only so K2 can feed its
// threadgroup-resident normalized row through identical arithmetic.
private let qwen35MTPQMVHeader = """
    inline float qwen_mtp_qdot(
        const device uint8_t* w, const thread float* x_thread,
        float scale, float bias, float sum) {
      float accum = 0;
      const device uint16_t* ws = (const device uint16_t*)w;
      for (int i = 0; i < 4; i++) {
        accum +=
            (x_thread[4 * i] * (ws[i] & 0x000f) +
             x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
             x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
             x_thread[4 * i + 3] * (ws[i] & 0xf000));
      }
      return scale * accum + sum * bias;
    }

    template <typename XPtr>
    inline float qwen_mtp_load16(XPtr x, thread float* x_thread) {
      float sum = 0;
      for (int i = 0; i < 16; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      return sum;
    }

    template <typename XPtr>
    inline void qwen_mtp_qmv4(
        const device uint32_t* w, const device bfloat16_t* s,
        const device bfloat16_t* z, XPtr x, int in_k,
        uint out_row, uint simd_lid, thread float* result) {
      const int in_vec_size_w = in_k / 2;
      const int in_vec_size_g = in_k / 64;
      const device uint8_t* ws = (const device uint8_t*)w
          + out_row * in_vec_size_w + simd_lid * 8;
      const device bfloat16_t* sl = s + out_row * in_vec_size_g + simd_lid / 4;
      const device bfloat16_t* bl = z + out_row * in_vec_size_g + simd_lid / 4;
      XPtr xp = x + simd_lid * 16;
      float x_thread[16];
      for (int k = 0; k < in_k; k += 512) {
        float sum = qwen_mtp_load16(xp, x_thread);
        for (int row = 0; row < 4; row++) {
          result[row] += qwen_mtp_qdot(
              ws + row * in_vec_size_w, x_thread,
              sl[row * in_vec_size_g], bl[row * in_vec_size_g], sum);
        }
        ws += 256;
        sl += 8;
        bl += 8;
        xp += 512;
      }
    }

    inline bfloat16_t qwen_mtp_sigmoid(bfloat16_t x) {
      auto y = 1 / (1 + metal::exp(metal::abs(x)));
      return (x < 0) ? y : 1 - y;
    }
    """

// K2: rmsnorm + fused gate/up QMV + silu(g) * u in one dispatch. The RMS
// prologue is redundant per threadgroup; the normalized row lives in 10.25 KB
// of threadgroup memory and feeds the header's stock lane arithmetic.
private let qwen35MTPGateUpActSource = """
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;

    // rms_looped<bfloat16_t, 4> replay: the stock pipeline runs axis 5120 >
    // RMS_LOOPED_LIMIT at its 1024-thread maximum, so partials are
    // re-derived on that VIRTUAL lane partition -- vt owns x[4vt..4vt+3],
    // second visit at +4096, the stock bounds guards collapsing to the one
    // below for AXIS in (4096, 8192] with AXIS % 4 == 0 (no partial tail) --
    // with identical simd_sum lane placement, then the same scratch fold and
    // metal::precise::rsqrt.
    threadgroup float rms_partials[32];
    threadgroup float rms_inv[1];
    threadgroup bfloat16_t x_norm[AXIS];

    for (uint g = simd_gid; g < 32u; g += 2u) {
      const uint vbase = (g * 32u + simd_lid) * 4u;
      float acc = 0.0f;
      for (uint i = 0; i < 4u; ++i) {
        float xi = h1[vbase + i];
        acc += xi * xi;
      }
      if (4096u + vbase + 4u <= uint(AXIS)) {
        for (uint i = 0; i < 4u; ++i) {
          float xi = h1[4096u + vbase + i];
          acc += xi * xi;
        }
      }
      acc = simd_sum(acc);
      if (simd_lid == 0) {
        rms_partials[g] = acc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
      float acc = simd_sum(rms_partials[simd_lid]);
      if (simd_lid == 0) {
        rms_inv[0] = metal::precise::rsqrt(acc / AXIS + eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_mean = rms_inv[0];
    for (uint e = thread_position_in_threadgroup.x; e < uint(AXIS); e += 64u) {
      x_norm[e] = w_post[e] * static_cast<bfloat16_t>(h1[e] * inv_mean);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gate rows [tile*ROWS, (tile+1)*ROWS) and their paired up rows
    // N_GATE + j in stock four-row chunks per simdgroup, so lane 0 finishes
    // the graph's bf16 silu(g) * u with no further barrier.
    const uint tile = threadgroup_position_in_grid.x;
    const threadgroup bfloat16_t* xn = x_norm;
    for (uint chunk = simd_gid * 4u; chunk < uint(ROWS); chunk += 8u) {
      const uint gate_row = tile * uint(ROWS) + chunk;
      float result_g[4] = {0};
      float result_u[4] = {0};
      qwen_mtp_qmv4(fq_w, fq_s, fq_z, xn, AXIS, gate_row, simd_lid, result_g);
      qwen_mtp_qmv4(fq_w, fq_s, fq_z, xn, AXIS,
          uint(N_GATE) + gate_row, simd_lid, result_u);
      for (int row = 0; row < 4; row++) {
        result_g[row] = simd_sum(result_g[row]);
        result_u[row] = simd_sum(result_u[row]);
        if (simd_lid == 0) {
          const bfloat16_t gv = static_cast<bfloat16_t>(result_g[row]);
          const bfloat16_t uv = static_cast<bfloat16_t>(result_u[row]);
          act[gate_row + row] = (gv * qwen_mtp_sigmoid(gv)) * uv;
        }
      }
    }
    """

/// K1 and K3 verbatim: stock qmv_fast launch geometry (64-thread groups, two
/// simdgroups, four consecutive rows each) with the residual add folded into
/// the store lane 0 already performs.
private func qwen35MTPResidualQMVKernel(_ name: String) -> MLXFast.MLXFastKernel {
    MLXFast.metalKernel(
        name: name,
        inputNames: ["x", "w", "s", "z", "resid"],
        outputNames: ["out"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint out_row =
                threadgroup_position_in_grid.x * 8u + simd_gid * 4u;
            float result[4] = {0};
            qwen_mtp_qmv4(w, s, z, x, IN_K, out_row, simd_lid, result);
            for (int row = 0; row < 4; row++) {
              result[row] = simd_sum(result[row]);
              if (simd_lid == 0) {
                out[out_row + row] = resid[out_row + row]
                    + static_cast<bfloat16_t>(result[row]);
              }
            }
            """,
        header: qwen35MTPQMVHeader)
}

private let qwen35MTPOProjResKernel = qwen35MTPResidualQMVKernel("qwen_mtp_oproj_res")
private let qwen35MTPDownResKernel = qwen35MTPResidualQMVKernel("qwen_mtp_down_res")
private let qwen35MTPGateUpActKernel = MLXFast.metalKernel(
    name: "qwen_mtp_postnorm_gateup_act",
    inputNames: ["h1", "w_post", "eps", "fq_w", "fq_s", "fq_z"],
    outputNames: ["act"],
    source: qwen35MTPGateUpActSource,
    header: qwen35MTPQMVHeader)

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

    // Armed once by Qwen35MTPModule.init on head-owned single-layer
    // instances: a per-INSTANCE gate, never a shape inference, so no
    // backbone layer can drift onto the fused tail.
    var fusedDraftTail = false

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
        if let fused = fusedTailForward(x, mask: mask, cache: cache) {
            return fused
        }
        // omlx: MTPDecoderLayer.__call__
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    /// The three-kernel draft tail, or nil when any envelope condition
    /// misses (the caller then runs the stock chain verbatim). The envelope
    /// pins the frozen head geometry the kernels were receipted at -- hidden
    /// 5120, gated attention 6144, intermediate 17408, every projection
    /// affine 4-bit/g64 with bf16 scales -- plus the per-instance flag,
    /// B == L == 1, the kill-switch, and the hardware gate the other custom
    /// kernels honour; the pinned bf16 head fails the QuantizedLinear casts.
    /// The multi-row flush's FINAL decoder row (L == 1 by construction)
    /// deliberately routes here too: same shapes, one warm family.
    private func fusedTailForward(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray? {
        func affine4g64(_ l: Linear) -> QuantizedLinear? {
            guard let q = l as? QuantizedLinear, q.bits == 4,
                  q.groupSize == 64, q.mode == .affine, q.bias == nil,
                  q.biases != nil
            else { return nil }
            return q
        }
        guard fusedDraftTail, qwen35MTPFusedTailEnabled,
              MLXHardwareInfo.isCompiledDecodeSupported,
              x.dim(0) == 1, x.dim(1) == 1, x.dim(2) == 5120,
              x.dtype == .bfloat16,
              let fusedMLP = mlp as? Qwen35FusedMLP,
              let oProj = affine4g64(selfAttn.oProj), let oZ = oProj.biases,
              oProj.shape == (5120, 6144),
              let down = affine4g64(fusedMLP.downProj),
              let downZ = down.biases, down.shape == (5120, 17408),
              let pack = fusedMLP.quantizedGateUpPack(),
              pack.bits == 4, pack.groupSize == 64, pack.gateOut == 17408,
              postAttentionLayerNorm.weight.shape == [5120],
              [oProj.scales, oZ, down.scales, downZ, pack.s, pack.z,
               postAttentionLayerNorm.weight]
                .allSatisfy({ $0.dtype == .bfloat16 })
        else { return nil }
        let gAtt = selfAttn.gatedAttentionOutput(
            inputLayerNorm(x), mask: mask, cache: cache)
        let h1 = qwen35MTPOProjResKernel(
            [gAtt, oProj.weight, oProj.scales, oZ, x],
            template: [("IN_K", 6144)],
            grid: (5120 / 8 * 64, 1, 1), threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, 5120]], outputDTypes: [.bfloat16])[0]
        let act = qwen35MTPGateUpActKernel(
            [h1, postAttentionLayerNorm.weight, postAttentionLayerNorm.eps,
             pack.w, pack.s, pack.z],
            template: [
                ("AXIS", 5120), ("N_GATE", 17408),
                ("ROWS", qwen35MTPFusedTailK2Rows),
            ],
            grid: (17408 / qwen35MTPFusedTailK2Rows * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, 17408]], outputDTypes: [.bfloat16])[0]
        return qwen35MTPDownResKernel(
            [act, down.weight, down.scales, downZ, h1],
            template: [("IN_K", 17408)],
            grid: (5120 / 8 * 64, 1, 1), threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, 5120]], outputDTypes: [.bfloat16])[0]
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
        // Arm the fused tail only on the single-layer head configuration the
        // gate and warm receipts cover; the layer still fails closed per call.
        if layers.count == 1 {
            layers[0].fusedDraftTail = true
        }
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        var fused = fc(concatenated([e, h], axis: -1))

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
