// Multi-Token Prediction (MTP) head for DeepSeek-V4.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/deepseek_v4_model.py  (MTPBlock, mtp_forward, make_mtp_cache)
// Reference: Blaizzy/mlx-lm#15

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Module-level MTP flag

/// Controls whether DeepSeek-V4 model inits attach the MTP head.
/// Set to `true` before calling `MLXLLM.load(...)` when MTP should be active.
/// Mirrors omlx `is_mtp_active()` / `set_mtp_active()` from
/// patches/mlx_lm_mtp/__init__.py.
public nonisolated(unsafe) var _deepseekV4MTPEnabled: Bool = false

// MARK: - DeepseekV4MTPBlock

/// One MTP layer in DeepSeek-V4's prediction stack.
///
/// Fuses the backbone's 4D pre-hcHead hidden state `h` `[B, S, hc, D]` with the
/// embedding of the next-position token, then runs a full `DeepseekV4Block` over
/// the result. The final block's `hc_head` collapses the HyperConnection dimension
/// before `norm` and the shared `lm_head` produce logits.
///
/// omlx: patches/mlx_lm_mtp/deepseek_v4_model.py  MTPBlock
final class DeepseekV4MTPBlock: Module {
    let config: DeepseekV4Configuration

    var block: DeepseekV4Block

    var e_proj: Linear
    var h_proj: Linear
    var enorm: RMSNorm
    var hnorm: RMSNorm
    var norm: RMSNorm

    /// HyperHead parameters for collapsing the hc dimension after this block.
    var hc_head: HCParams

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self.block = DeepseekV4Block(config: config, layerIdx: layerIdx)

        let dim = config.hiddenSize
        self.e_proj = Linear(dim, dim, bias: false)
        self.h_proj = Linear(dim, dim, bias: false)
        self.enorm = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        self.hnorm = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        self.norm = RMSNorm(dimensions: dim, eps: config.rmsNormEps)

        let hc = config.hcMult
        self.hc_head = HCParams(
            fn: zeros([hc, hc * dim]),
            base: zeros([hc]),
            scale: ones([1]))
    }

    /// Forward one MTP block.
    ///
    /// - Parameters:
    ///   - h: 4D hidden `[B, S, hc, D]` from backbone or previous MTP block
    ///   - embedTokens: shared embedding table from the backbone model
    ///   - inputIds: next-token ids `[B, S]`
    ///   - mask: attention mask (pre-computed for the full MTP chain)
    ///   - cache: KV cache for this MTP block
    /// - Returns: updated 4D hidden `[B, S, hc, D]`
    ///
    /// omlx: MTPBlock.__call__
    func callAsFunction(
        _ h: MLXArray,
        embedTokens: Embedding,
        inputIds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        // 1. Norm and project the embedding of the next token.
        let e = enorm(embedTokens(inputIds))              // [B, S, D]

        // 2. Norm the 4D hidden state (RMSNorm operates on last dim).
        let hNorm = hnorm(h)                              // [B, S, hc, D]

        // 3. Fuse: broadcast projected embedding to [B, S, 1, D] then add.
        //    h_proj maps [B, S, hc, D] → [B, S, hc, D] (Linear on last dim).
        let x = e_proj(e).expandedDimensions(axis: 2) + h_proj(hNorm)  // [B, S, hc, D]

        // 4. Run full transformer block (returns 4D hidden).
        return block(x, mask: mask, cache: cache, inputIds: inputIds)
    }
}

// MARK: - MTPCapable conformance

extension DeepseekV4Model: MTPCapable {
    public var hasMTPHead: Bool { mtp != nil }

    /// Backbone forward that also returns the raw 4D pre-hcHead hidden state.
    ///
    /// The raw hidden `[B, S, hc, D]` is passed to `mtpForward` to produce draft logits.
    /// `nConfirmed` is unused in this model (no prefix caching in the V4 port).
    ///
    /// omlx: Model.__call__ with return_hidden=True
    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        let cacheArr: [KVCache] = cache.isEmpty ? [] : cache
        let (normed, rawHidden) = model.forward(
            input.tokens,
            cache: cacheArr.isEmpty ? nil : cacheArr,
            returnRawHidden: true)
        let logits = lmHead(normed)
        // rawHidden is non-nil because returnRawHidden=true.
        return (logits, rawHidden!)
    }

    /// Run the chained MTP blocks and produce draft logits.
    ///
    /// Each MTP block fuses `hidden` (4D, `[B, S, hc, D]`) with embedded
    /// `nextTokenIds` through its `e_proj`/`h_proj` projections, then passes the
    /// result through a `DeepseekV4Block`. The final block's `hc_head` collapses
    /// the HyperConnection dimension before `norm` and the shared `lm_head`.
    ///
    /// omlx: Model.mtp_forward
    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        guard let mtpBlocks = mtp else {
            fatalError(
                "mtpForward called but MTP head is not attached. "
                    + "Set _deepseekV4MTPEnabled = true before loading the model.")
        }

        var h = hidden  // [B, S, hc, D]

        // Build attention mask from the collapsed [B, S, hc*D] representation.
        let B = h.dim(0), S = h.dim(1), hc = h.dim(2), D = h.dim(3)
        let maskInput = h.reshaped([B, S, hc * D])
        let firstCache: (any KVCache)? = cache.isEmpty ? nil : cache[0]
        let mask = createAttentionMask(
            h: maskInput, cache: firstCache,
            windowSize: args.slidingWindow, returnArray: true)

        var lastBlock: DeepseekV4MTPBlock?
        for (i, mtpBlock) in mtpBlocks.enumerated() {
            let c: KVCache? = i < cache.count ? cache[i] : nil
            h = mtpBlock(
                h,
                embedTokens: model.embedTokens,
                inputIds: nextTokenIds,
                mask: mask,
                cache: c)
            lastBlock = mtpBlock
        }

        guard let lb = lastBlock else { fatalError("MTP block list is empty") }

        // Collapse hc dimension with the last block's hc_head, then norm + lm_head.
        let out = hcHeadReduce(
            x: h, hcFn: lb.hc_head.fn, hcScale: lb.hc_head.scale,
            hcBase: lb.hc_head.base, eps: args.hcEps)
        return lmHead(lb.norm(out))
    }

    /// Allocate fresh KV caches for the MTP head layers.
    ///
    /// MTP layers default to `compress_ratio=0` (no PoolingCache), so each block
    /// gets a plain `KVCacheSimple`.
    ///
    /// omlx: Model.make_mtp_cache
    public func makeMTPCache() -> [any KVCache] {
        guard let mtpBlocks = mtp else { return [] }
        return mtpBlocks.map { _ in KVCacheSimple() as any KVCache }
    }
}
