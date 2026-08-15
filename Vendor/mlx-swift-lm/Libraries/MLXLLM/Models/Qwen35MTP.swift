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

// MARK: - Fused MTP-head projections

/// `MLXFAST_QWEN_MTP_FUSED_HEAD_PROJECTIONS` (DEFAULT ON; set `"0"` to restore
/// the original per-projection path exactly).
///
/// WHAT IT GATES. Two same-input projection groups inside the MTP head, and
/// nothing else:
///
///   1. the head layer's `q_proj` / `k_proj` / `v_proj`, via the BF16 branch
///      added to `Qwen35Attention.qkv(_:)` in `Qwen35.swift` — the branch that
///      submission `088f763` left as an explicit `return (qProj(x), kProj(x),
///      vProj(x))` fallback with the comment "Unquantized (MTP bf16) falls
///      back". The backbone's 16 full-attention layers are affine-4-bit and
///      therefore never reach it; the head is the only unquantized attention in
///      the tree, so that branch is head-only by construction.
///   2. the head layer's `gate_proj` / `up_proj`, via
///      `Qwen35MTPFusedGateUp` below, reached only from
///      `Qwen35MTPDecoderLayer.callAsFunction`.
///
/// `o_proj` and `down_proj` cannot join either group: they consume the
/// attention output and the gated MLP product respectively, not the layer
/// input, so there is no shared input vector to share a dispatch over.
///
/// Read ONCE, as a module-level `let`, so the guarded and unguarded paths
/// differ by a branch on an already-resolved `Bool` and no environment lookup
/// ever happens inside the timed window. Mirrors the
/// `MLXFAST_QWEN_MTP_SEED_TAIL_PROJECTION` convention on the session side.
///
/// WHAT THIS DELIBERATELY IS NOT: a requantization. 0xkydo's `5205c88`
/// measured all-linear 4-bit on this head dropping the accept rate from 0.641
/// to 0.600 — head precision is load-bearing for acceptance. Every array here
/// is a `concatenated` copy of BF16 weights, which preserves dtype exactly, and
/// every guard below refuses to fuse anything that is not a plain, bias-free,
/// same-dtype, same-input-width `Linear`.
let qwen35MTPFusedHeadProjectionsEnabled =
    ProcessInfo.processInfo.environment["MLXFAST_QWEN_MTP_FUSED_HEAD_PROJECTIONS"]
        != "0"

/// Concatenated `[gate_proj; up_proj]` weight for one MTP-head layer, built
/// once, lazily, on that layer's first forward.
///
/// WHY THE FUSED GEMV IS BIT-EXACT, not merely equivalent. For a `[1, 1, K]`
/// input the vendored `matmul` routes to `gemv` (`metal/matmul.cpp`: "Route to
/// gemv if needed", `std::min(M, N) == 1`), with `mat` = the weight in its
/// natural `[N, K]` row-major layout and `transpose_mat == false`. In that
/// branch the reduction geometry over K is fixed by `(BN, SN, TN, K)`: each
/// lane owns `TN = 4` contiguous K elements at `bn = (simdN + thrN) * TN` and
/// strides by `blockN = BN * SN * TN = 128`, accumulating into a `float`, and
/// then a `simd_shuffle_down` tree of width `SN = 32` reduces the lanes. `BN`
/// is 1 and `SM`/`SN`/`TM`/`TN` are constants in this branch, so with
/// `K = 5120` fixed by the head's hidden size, every output element sums the
/// SAME 40 chunks in the SAME order whatever `N` is. The one `N`-dependent
/// dispatch parameter is `BM` (`out_vector_len >= 4096 ? 8 : 4`), and `BM`
/// appears only in `out_row = tid.x * blockM + bm`, i.e. it selects WHICH
/// OUTPUT ROWS a threadgroup owns — never the K loop, never the reduction
/// (`needs_tgp_reduction` is `BN > 1`, false on both sides). Every fused output
/// length used here is an exact multiple of `blockM`, so not even the
/// tail-shift edge case engages, and `gemv.h` documents the `gemv_al`
/// aligned-load twin as loading "identical values in identical order
/// (bit-exact)". Concatenation therefore changes the dispatch count and
/// nothing else about the arithmetic.
///
/// WHY IT IS RESTRICTED TO THE SINGLE-ROW SHAPE. The paragraph above is a GEMV
/// argument. A wider call would be a Steel GEMM whose tile selection does
/// depend on `N`, and the claim would not carry. This track's head is only ever
/// called with one row (`Qwen36MTPBlockSession.generateRound` drafts one token
/// at a time, `nextTokenIds` shaped `[1, 1]`), so restricting to
/// `B == 1 && L == 1` costs nothing and keeps the claim airtight. Any other
/// shape takes the original path.
///
/// AND WHY EVEN A DIFFERENCE WOULD BE HARMLESS. The MTP head only PROPOSES.
/// Every emitted token is an argmax of the ORGANIZER-PINNED TARGET's own verify
/// forward, and the trusted parent re-checks the whole stream against a hidden
/// serial trajectory after the clock stops. `mtp-head.manifest.json` states the
/// invariant outright: a head "moves the accept rate, which is the game, and
/// cannot move the output". So the worst case this could reach — a last-ulp
/// difference flipping an exact near-tie draft PROPOSAL — is a change in
/// acceptance, never in fidelity.
///
/// NOT A `Module`, AND NOT AN `MLXArray` PROPERTY ON ONE. `MLXNN.Module`
/// reflects every stored `MLXArray` property into `parameters()`
/// (`ModuleValue.build`: `case let v as MLXArray: return
/// .value(.parameters(v))`), so a fused copy stored directly on a `Module`
/// would be visible to the parameter walk, `update(parameters:)` and the
/// quantization walk as a tensor no checkpoint names. It escapes today only
/// because `Module` caches `items()` on first use, while the array is still
/// nil. Holding it in a plain `final class` removes the dependence on that
/// timing: a non-`MLXArray`, non-`Module` property is classified
/// `.value(.other)` and is invisible to all three walks, exactly like the
/// existing `let scale: Float` on `Qwen35Attention`.
final class Qwen35MTPFusedGateUp {
    /// Set on the first build attempt whether or not it succeeded, so a layer
    /// whose shape refuses fusion re-checks nothing per round.
    private var attempted = false
    /// `[2 * hidden, inputWidth]`, or nil when the pair did not fuse.
    private var weight: MLXArray?
    private var hidden = 0

    /// `down_proj(silu(gate) * up)` with ONE GEMV instead of two, or `nil` when
    /// this layer/shape is not eligible and the caller should take the original
    /// path.
    ///
    /// Identical to `Qwen3NextMLP.callAsFunction` with `gateProj(x)` and
    /// `upProj(x)` replaced by two slices of one concatenated projection. The
    /// slices are the leading and trailing halves of a single output row, so
    /// each is contiguous, and `down_proj` still consumes the same freshly
    /// materialised `silu(gate) * up` product it consumes today.
    func callAsFunction(_ mlp: Qwen3NextMLP, _ x: MLXArray) -> MLXArray? {
        guard x.ndim == 3, x.dim(0) == 1, x.dim(1) == 1 else { return nil }
        buildIfNeeded(mlp)
        guard let weight else { return nil }
        let projected = matmul(x, weight.T)
        let gate = projected[.ellipsis, ..<hidden]
        let up = projected[.ellipsis, hidden...]
        return mlp.downProj(silu(gate) * up)
    }

    /// FAIL-OPEN BY DESIGN: any precondition that is not met leaves `weight`
    /// nil and the original two-launch path in service. The preconditions are
    /// the ones the bit-exactness argument actually rests on:
    ///
    ///  * plain `Linear`, never `QuantizedLinear` — which is a SUBCLASS of
    ///    `Linear`, so an `is` test is required rather than a type annotation.
    ///    A quantized layer's `weight` is packed `uint32` alongside `scales` /
    ///    `biases`; concatenating it as a dense matrix would be silently wrong.
    ///    The pinned head is BF16 and carries no `.scales`, so the
    ///    `quantize(model:)` walk leaves it alone — this guard is what makes
    ///    that a checked fact rather than an assumption.
    ///  * no bias on either leg, so the fused result needs no split bias.
    ///  * identical dtype, so `concatenated` cannot promote and change the
    ///    arithmetic.
    ///  * identical input width, which is what makes one shared input vector
    ///    legitimate at all.
    private func buildIfNeeded(_ mlp: Qwen3NextMLP) {
        guard !attempted else { return }
        attempted = true
        let legs = [mlp.gateProj, mlp.upProj]
        for leg in legs {
            if leg is QuantizedLinear { return }
            if leg.bias != nil { return }
            if leg.weight.ndim != 2 { return }
            if leg.weight.dtype != mlp.gateProj.weight.dtype { return }
            if leg.weight.dim(1) != mlp.gateProj.weight.dim(1) { return }
        }
        hidden = mlp.gateProj.weight.dim(0)
        let w = concatenated(legs.map { $0.weight }, axis: 0).contiguous()
        // Materialise once, HERE, so the concatenation is paid on the first
        // head forward rather than re-entering the graph every round. The
        // first head forward is `Qwen36MTPBlockSession.warmAllDepths`'s draft
        // step, which the worker runs OUTSIDE every scored window.
        eval(w)
        weight = w
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

    /// Lazily-built concatenated `[gate_proj; up_proj]` for THIS layer. Not a
    /// `Module` and not an `MLXArray`, so `Module`'s reflection classifies it
    /// `.other` and the parameter / quantization walks never see it. See
    /// `Qwen35MTPFusedGateUp`.
    let fusedGateUp = Qwen35MTPFusedGateUp()

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
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
        //
        // The attention call is UNCHANGED: `Qwen35Attention.qkv(_:)` now packs
        // q/k/v for this layer too (its BF16 branch), so the fusion arrives
        // without a second call shape here.
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let postNormed = postAttentionLayerNorm(h)
        // FUSED GATE/UP SEAM (`MLXFAST_QWEN_MTP_FUSED_HEAD_PROJECTIONS`,
        // default on; `"0"` restores the exact original expression below). The
        // two branches converge immediately -- same input, same returned shape
        // -- so the flag is a true ablation of the launch-count reduction
        // rather than two divergent head forwards. Anything not eligible
        // returns nil and falls through.
        if qwen35MTPFusedHeadProjectionsEnabled,
            let dense = mlp as? Qwen3NextMLP,
            let fused = fusedGateUp(dense, postNormed)
        {
            return h + fused
        }
        return h + (mlp as! UnaryLayer)(postNormed)
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
}
