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

    // MARK: fc embedding-half fold (input-independent, proposal-only)
    //
    // `fc` is bias-free, so fc([e; h]) = W_e @ e + W_h @ h where W_e / W_h are
    // the first / second K-half column blocks. The e-term
    // `W_e @ preFcNormEmbedding(embed(token))` depends only on the token id and
    // fixed weights, so it is precomputable for the whole vocabulary once at
    // first head use (inside the untimed warm window). Each draft step then
    // pays one row gather + a K=hidden qmv instead of embed-gather + RMSNorm +
    // a [e;h] concat + a K=2*hidden qmv. The fold is proposal-side only: the
    // head proposes, the exact target verifies, so the (tiny, bf16-rounding)
    // numeric difference vs the fused kernel can only move accept rate, never
    // an emitted token. Fail-closed: any unexpected layout leaves the fused
    // path in place. Kill switch: MLXFAST_QWEN_MTP_FC_FOLD=0.
    private var _fcFoldAttempted = false
    private var _fcEmbedTable: MLXArray?  // [vocab, H] = preFcNormEmbedding(E) @ W_e^T
    private var _fcHiddenW: MLXArray?  // packed h-half (quantized fc)
    private var _fcHiddenS: MLXArray?
    private var _fcHiddenZ: MLXArray?
    private var _fcHiddenGroupSize = 64
    private var _fcHiddenBits = 4
    private var _fcHiddenMode: QuantizationMode = .affine
    private var _fcHiddenDenseWT: MLXArray?  // [H, H] transposed h-half (bf16 fc)

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

    /// Build (or refuse, once) the fc embedding-half fold. Runs on the first
    /// head forward, which the session always issues inside the untimed warm
    /// window; the serial (never-drafting) leg never calls the head and never
    /// pays for or holds the table.
    private func prepareFcFoldIfNeeded(embedTokens: Embedding) {
        guard !_fcFoldAttempted else { return }
        _fcFoldAttempted = true
        if ProcessInfo.processInfo.environment["MLXFAST_QWEN_MTP_FC_FOLD"] == "0" {
            return
        }

        let hiddenSize = preFcNormHidden.weight.dim(0)
        let vocabRows = embedTokens.weight.dim(0)

        // Split fc into K-halves. Both halves must land exactly on packed-word
        // and quantization-group boundaries; otherwise leave the fused path.
        var eHalf: ((MLXArray) -> MLXArray)?
        if let qfc = fc as? QuantizedLinear {
            let packedCols = qfc.weight.dim(1)
            let scaleCols = qfc.scales.dim(1)
            guard qfc.bias == nil,
                  qfc.weight.dim(0) == hiddenSize,
                  packedCols % 2 == 0, scaleCols % 2 == 0,
                  hiddenSize % qfc.groupSize == 0,
                  qfc.biases != nil
            else { return }
            let pHalf = packedCols / 2
            let gHalf = scaleCols / 2
            let wE = qfc.weight[0..., 0 ..< pHalf]
            let sE = qfc.scales[0..., 0 ..< gHalf]
            let zE = qfc.biases![0..., 0 ..< gHalf]
            _fcHiddenW = qfc.weight[0..., pHalf...].contiguous()
            _fcHiddenS = qfc.scales[0..., gHalf...].contiguous()
            _fcHiddenZ = qfc.biases![0..., gHalf...].contiguous()
            _fcHiddenGroupSize = qfc.groupSize
            _fcHiddenBits = qfc.bits
            _fcHiddenMode = qfc.mode
            let wEc = wE.contiguous()
            let sEc = sE.contiguous()
            let zEc = zE.contiguous()
            eHalf = { x in
                quantizedMM(
                    x, wEc, scales: sEc, biases: zEc, transpose: true,
                    groupSize: qfc.groupSize, bits: qfc.bits, mode: qfc.mode)
            }
        } else {
            let w = fc.weight
            guard fc.bias == nil,
                  w.dim(0) == hiddenSize, w.dim(1) == 2 * hiddenSize
            else { return }
            let wE = w[0..., 0 ..< hiddenSize].contiguous()
            _fcHiddenDenseWT = w[0..., hiddenSize...].transposed(1, 0).contiguous()
            eHalf = { x in matmul(x, wE.transposed(1, 0)) }
        }
        guard let eHalf else { return }

        // Precompute T[token] = W_e @ preFcNormEmbedding(embed(token)) for the
        // whole vocabulary, chunked to bound the transient dequant footprint.
        let chunk = 65536
        var parts: [MLXArray] = []
        var row = 0
        while row < vocabRows {
            let hi = min(row + chunk, vocabRows)
            let rows: MLXArray
            if let qe = embedTokens as? QuantizedEmbedding {
                rows = dequantized(
                    qe.weight[row ..< hi], scales: qe.scales[row ..< hi],
                    biases: qe.biases == nil ? nil : qe.biases![row ..< hi],
                    groupSize: qe.groupSize, bits: qe.bits, mode: qe.mode)
            } else {
                rows = embedTokens.weight[row ..< hi]
            }
            let part = eHalf(preFcNormEmbedding(rows))
            eval(part)
            parts.append(part)
            row = hi
        }
        let table = parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
        eval(table)
        _fcEmbedTable = table
        FileHandle.standardError.write(Data(
            ("Qwen35MTPModule: fc embedding-half fold active "
             + "(table \(table.dim(0))x\(table.dim(1)) \(table.dtype))\n").utf8))
    }

    /// Fusion-stage rows: `fc([preFcNormEmbedding(embed(ids)); preFcNormHidden(hidden)])`,
    /// via the fold when available, else the reference fused path.
    private func fusedInput(
        hidden: MLXArray, nextTokenIds: MLXArray, embedTokens: Embedding
    ) -> MLXArray {
        prepareFcFoldIfNeeded(embedTokens: embedTokens)
        if let table = _fcEmbedTable {
            let shape = nextTokenIds.shape
            let eTerm = table[nextTokenIds.flattened()].reshaped(shape + [-1])
            let h = preFcNormHidden(hidden)
            let hTerm: MLXArray
            if let w = _fcHiddenW, let s = _fcHiddenS, let z = _fcHiddenZ {
                hTerm = quantizedMM(
                    h, w, scales: s, biases: z, transpose: true,
                    groupSize: _fcHiddenGroupSize, bits: _fcHiddenBits,
                    mode: _fcHiddenMode)
            } else if let wt = _fcHiddenDenseWT {
                hTerm = matmul(h, wt)
            } else {
                // Unreachable when the table exists; keep the reference result.
                let e = preFcNormEmbedding(embedTokens(nextTokenIds))
                return fc(concatenated([e, h], axis: -1))
            }
            return eTerm + hTerm
        }
        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        return fc(concatenated([e, h], axis: -1))
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        var fused = fusedInput(
            hidden: hidden, nextTokenIds: nextTokenIds, embedTokens: embedTokens)

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

        let fused = fusedInput(
            hidden: hidden, nextTokenIds: nextTokenIds, embedTokens: embedTokens)
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }

}
