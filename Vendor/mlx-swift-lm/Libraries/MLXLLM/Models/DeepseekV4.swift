// DeepSeek-V4 model port.
//
// Faithful port of ml-explore/mlx-lm#1192 (base model by Blaizzy, 0xClandestine,
// angeloskath, pcuenca, eauchs) and Blaizzy/mlx-lm#15 (MTP).
// Reference: omlx/patches/deepseek_v4/deepseek_v4_model.py
//            omlx/patches/deepseek_v4/hyper_connection.py
//            omlx/patches/deepseek_v4/cache_extras.py (PoolingCache)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct DeepseekV4Configuration: Codable, Sendable {
    var vocabSize: Int = 129280
    var hiddenSize: Int = 4096
    var moeIntermediateSize: Int = 2048
    var numHiddenLayers: Int = 43
    var numAttentionHeads: Int = 64
    var headDim: Int = 512
    var qLoraRank: Int = 1024
    var qkRopeHeadDim: Int = 64
    var rmsNormEps: Float = 1e-6
    var oGroups: Int = 8
    var oLoraRank: Int = 1024
    var slidingWindow: Int = 128
    var compressRatios: [Int] = []
    var compressRopeTheta: Float = 160000.0
    var nRoutedExperts: Int = 256
    var nSharedExperts: Int = 1
    var numExpertsPerTok: Int = 6
    var scoringFunc: String = "sqrtsoftplus"
    var routedScalingFactor: Float = 1.5
    var swiguLimit: Float = 10.0
    var numHashLayers: Int = 3
    var numNextnPredictLayers: Int = 1
    var normTopkProb: Bool = true
    var hcMult: Int = 4
    var hcSinkhornIters: Int = 20
    var hcEps: Float = 1e-6
    var ropeTheta: Float = 10000.0
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int = 1_048_576
    var indexNHeads: Int = 64
    var indexHeadDim: Int = 128
    var indexTopk: Int = 512

    var nopeHeadDim: Int { headDim - qkRopeHeadDim }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case compressRopeTheta = "compress_rope_theta"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case scoringFunc = "scoring_func"
        case routedScalingFactor = "routed_scaling_factor"
        case swiguLimit = "swiglu_limit"
        case numHashLayers = "num_hash_layers"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case normTopkProb = "norm_topk_prob"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
    }

    /// Populate default compress_ratios if empty (mirrors Python __post_init__).
    mutating func fillDefaultCompressRatios() {
        guard compressRatios.isEmpty else { return }
        let n = numHiddenLayers
        compressRatios =
            [0]
            + (0..<max(n - 2, 0)).map { i in i % 2 == 0 ? 128 : 4 }
            + (n >= 2 ? [0] : [])
        compressRatios = Array(compressRatios.prefix(numHiddenLayers))
    }
}

// MARK: - PoolingCache

/// Accumulates tokens into windows of size `ratio`, compresses them, and stores pooled KVs.
/// Mirrors Python PoolingCache from mlx-lm#1192 cache.py.
final class PoolingCache {
    let ratio: Int
    var bufKV: MLXArray?    // [B, remainder, out_dim]
    var bufGate: MLXArray?  // [B, remainder, out_dim]
    var pooled: MLXArray?   // [B, n_pooled, head_dim]

    var pooledCount: Int { pooled?.dim(1) ?? 0 }

    init(ratio: Int) {
        self.ratio = ratio
    }

    /// Split `kv`/`gate` into complete windows of `ratio` tokens.
    /// Returns (readyKV, readyGate, poolBase) where readyKV has shape [B, n_windows*ratio, D].
    func accumulateWindows(kv: MLXArray, gate: MLXArray, offset: Int)
        -> (MLXArray, MLXArray, Int)
    {
        let B = kv.dim(0), L = kv.dim(1), D1 = kv.dim(2), D2 = gate.dim(2)

        if L > 1 {
            // Prompt mode
            let remainder = bufKV?.dim(1) ?? 0
            let total = L + remainder
            let usable = (total / ratio) * ratio
            let newRemainder = total % ratio

            let readyKV: MLXArray
            let readyGate: MLXArray
            let poolBase: Int

            if usable > 0 {
                let prevKV = bufKV ?? zeros([B, 0, D1], dtype: kv.dtype)
                let prevGate = bufGate ?? zeros([B, 0, D2], dtype: gate.dtype)
                let combinedKV = concatenated([prevKV, kv], axis: 1)
                let combinedGate = concatenated([prevGate, gate], axis: 1)
                readyKV = combinedKV[0..., ..<usable, 0...]
                readyGate = combinedGate[0..., ..<usable, 0...]
                poolBase = offset - remainder
            } else {
                readyKV = zeros([B, 0, D1], dtype: kv.dtype)
                readyGate = zeros([B, 0, D2], dtype: gate.dtype)
                poolBase = 0
            }

            if newRemainder > 0 {
                bufKV = kv[0..., (L - newRemainder)..., 0...]
                bufGate = gate[0..., (L - newRemainder)..., 0...]
            } else {
                bufKV = nil
                bufGate = nil
            }

            return (readyKV, readyGate, poolBase)
        } else {
            // Decode mode: accumulate one token at a time
            let newBufKV: MLXArray
            let newBufGate: MLXArray
            if let cur = bufKV, let curG = bufGate {
                newBufKV = concatenated([cur, kv], axis: 1)
                newBufGate = concatenated([curG, gate], axis: 1)
            } else {
                newBufKV = kv
                newBufGate = gate
            }

            if newBufKV.dim(1) == ratio {
                // Full window ready
                let readyKV = newBufKV
                let readyGate = newBufGate
                let poolBase = offset - ratio + 1
                bufKV = nil
                bufGate = nil
                return (readyKV, readyGate, poolBase)
            } else {
                bufKV = newBufKV
                bufGate = newBufGate
                return (zeros([B, 0, D1], dtype: kv.dtype), zeros([B, 0, D2], dtype: gate.dtype), 0)
            }
        }
    }

    /// Append newly compressed tokens to the pool and return full pooled array.
    func updateAndFetch(_ px: MLXArray) -> MLXArray {
        if px.dim(1) == 0 {
            return pooled ?? zeros([px.dim(0), 0, px.dim(2)], dtype: px.dtype)
        }
        if let p = pooled {
            pooled = concatenated([p, px], axis: 1)
        } else {
            pooled = px
        }
        return pooled!
    }

    /// Build a causal validity mask for pooled positions.
    /// Returns `[L, P]` bool mask (nil for decode or empty pool).
    func makeMask(L: Int, offset: Int) -> MLXArray? {
        guard let p = pooled, L > 1 else { return nil }
        let P = p.dim(1)
        let poolIdx = MLXArray(Int32(0)..<Int32(P))           // [P]
        let queryPos = MLXArray(Int32(offset + 1)..<Int32(offset + L + 1))  // [L]
        return poolIdx .< (queryPos[0..., .newAxis] / Int32(ratio))  // [L, P]
    }
}

// MARK: - DeepseekV4LayerCache

/// KVCache-conformant wrapper for compressed attention layers.
/// Holds a RotatingKVCache (local window) + 1–2 PoolingCaches.
///
/// Implemented as a direct KVCache conformance (not subclassing BaseKVCache) because
/// BaseKVCache.init() and its `offset`/`maxSize` properties are not open for external override.
final class DeepseekV4LayerCache: KVCache {
    let rotating: RotatingKVCache
    var pooling: [PoolingCache]

    init(rotating: RotatingKVCache, pooling: [PoolingCache]) {
        self.rotating = rotating
        self.pooling = pooling
    }

    // MARK: KVCache / Evaluatable / Updatable

    func innerState() -> [MLXArray] { rotating.innerState() }

    var offset: Int {
        get { rotating.offset }
        set { rotating.offset = newValue }
    }

    var maxSize: Int? { rotating.maxSize }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        rotating.update(keys: keys, values: values)
    }

    var state: [MLXArray] {
        get { rotating.state }
        set { rotating.state = newValue }
    }

    var metaState: [String] {
        get { rotating.metaState }
        set { rotating.metaState = newValue }
    }

    var isTrimmable: Bool { rotating.isTrimmable }

    @discardableResult
    func trim(_ n: Int) -> Int { rotating.trim(n) }

    func copy() -> any KVCache {
        DeepseekV4LayerCache(rotating: rotating.copy() as! RotatingKVCache, pooling: pooling)
    }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        rotating.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }
}

// MARK: - DeepseekV4RoPE

/// Custom RoPE that prepends `nope_pairs` infinite-frequency pairs before the real RoPE dims.
/// Supports `inverse=true` for un-rotating attention outputs.
/// Mirrors Python DeepseekV4RoPE from mlx-lm#1192.
final class DeepseekV4RoPE {
    let dims: Int
    let freqScale: Int
    private let baseFreqs: MLXArray  // [dims/2] – base periods
    private var cache: [String: MLXArray] = [:]

    init(
        dims: Int,
        base: Float,
        scalingConfig: [String: StringOrNumber]?,
        maxPositionEmbeddings: Int,
        freqScale: Int = 1
    ) {
        self.dims = dims
        self.freqScale = freqScale

        // inv_freq[i] = base^(2i/dims)
        let exponents =
            MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32) / Float(dims)
        var invFreq = MLX.pow(base, exponents)  // [dims/2]

        // Yarn / DeepSeek-Yarn scaling
        let ropeType: String?
        if let cfg = scalingConfig {
            if case .string(let s) = cfg["type"] { ropeType = s }
            else if case .string(let s) = cfg["rope_type"] { ropeType = s }
            else { ropeType = nil }
        } else {
            ropeType = nil
        }

        if ropeType == "yarn" || ropeType == "deepseek_yarn" {
            let factor = scalingConfig?["factor"]?.asFloat() ?? 1.0
            let origMaxPos =
                (scalingConfig?["original_max_position_embeddings"]?.asInts()?.first)
                ?? maxPositionEmbeddings
            let betaFast = scalingConfig?["beta_fast"]?.asFloat() ?? 32.0
            let betaSlow = scalingConfig?["beta_slow"]?.asFloat() ?? 1.0

            func correctionDim(_ numRot: Float) -> Float {
                Float(dims) * log(Float(origMaxPos) / (numRot * 2 * .pi)) / (2 * log(base))
            }

            let low = max(Int(floor(correctionDim(betaFast))), 0)
            var high = min(Int(ceil(correctionDim(betaSlow))), dims - 1)
            if low == high { high += 1 }

            let ramp =
                (MLXArray(stride(from: 0, to: dims / 2, by: 1)).asType(.float32) - Float(low))
                / Float(high - low)
            let smooth = 1.0 - MLX.clip(ramp, min: MLXArray(Float(0)), max: MLXArray(Float(1)))
            invFreq = invFreq / factor * (1.0 - smooth) + invFreq * smooth
        }

        // baseFreqs = 1 / inv_freq = periods
        self.baseFreqs = MLXArray(1.0) / invFreq
    }

    private func freqs(headDim: Int, inverse: Bool) -> MLXArray {
        let key = "\(headDim)_\(inverse)"
        if let cached = cache[key] { return cached }

        var f = baseFreqs
        if freqScale != 1 {
            f = f / Float(freqScale)
        }
        if inverse {
            f = -f
        }
        let nopePairs = (headDim - dims) / 2
        if nopePairs > 0 {
            // Infinite frequency → angle = pos / inf = 0 → identity rotation (NOPE)
            let infArr = MLXArray(Array(repeating: Float.infinity, count: nopePairs))
            f = concatenated([infArr, f])
        }
        cache[key] = f
        return f
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0, inverse: Bool = false) -> MLXArray {
        let headDim = x.dim(-1)
        let f = freqs(headDim: headDim, inverse: inverse)
        let actualOffset = freqScale != 1 ? offset / freqScale : offset
        return MLXFast.RoPE(
            x,
            dimensions: headDim,
            traditional: true,
            base: nil,
            scale: 1.0,
            offset: actualOffset,
            freqs: f)
    }
}

// MARK: - MultiLinear

/// Grouped linear: o_groups independent projections applied in batch.
/// Weight shape: [groups, out_features, in_features].
/// Used for wo_a (output gate projection before wo_b).
final class DeepseekV4MultiLinear: Module {
    var weight: MLXArray  // [groups, out_features, in_features]

    init(inFeatures: Int, outFeatures: Int, groups: Int) {
        self.weight = zeros([groups, outFeatures, inFeatures])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, groups, L, in_features]
        // weight.T (axis 1,2): [groups, in_features, out_features]
        return matmul(x, weight.transposed(0, 2, 1))
    }
}

// MARK: - HCParams

/// Holds the three learnable tensors for one HyperConnection block.
class HCParams: Module {
    var fn: MLXArray
    var base: MLXArray
    var scale: MLXArray

    init(fn: MLXArray, base: MLXArray, scale: MLXArray) {
        self.fn = fn
        self.base = base
        self.scale = scale
        super.init()
    }
}

// MARK: - HC Sinkhorn Collapse Metal Kernel

/// Fused Sinkhorn-normalisation + HC-collapse Metal kernel.
///
/// Mirrors `_make_hc_sinkhorn_collapse_kernel` from
/// omlx/patches/deepseek_v4/hyper_connection.py.
///
/// Layout: one threadgroup per token (B×S rows), 256 threads per group.
///   - SIMD-group 0 (lanes 0–3 active): compute pre / post / comb (Phase 1).
///   - All 256 threads: collapse [HC, D] → [D] with pre weights (Phase 2).
///
/// Template parameters (all constexpr):
///   T        input dtype, U output dtype (both x.dtype)
///   HC       hc_mult (4 for DeepSeek-V4)
///   ITERS    hc_sinkhorn_iters
///   D        hidden_size
///   EPS_INT  eps expressed as integer × 1e-9 (avoids float template args)
private let _hcSinkhornCollapseKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "hc_sinkhorn_collapse",
    inputNames: ["x_in", "mixes", "scale", "base"],
    outputNames: ["collapsed", "post", "comb"],
    source: """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int   MIX      = (2 + HC) * HC;
        constexpr int   BASE_OFF = 2 * HC;
        constexpr float EPS      = EPS_INT * 1e-9f;

        const device float* mix      = (const device float*)mixes + row * MIX;
              device float* post_out = (device float*)post     + row * HC;
              device float* comb_out = (device float*)comb     + row * HC * HC;

        threadgroup float pre_shared[HC];

        // ── Phase 1: branchless Sinkhorn on SIMD-group 0 ──────────────────────
        // Lanes >= HC multiply by active=0 so they never corrupt sums —
        // no divergent branches inside the Sinkhorn loop.
        if (sg == 0) {
            const float pre_scale  = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            float pre_z  = mix[llane]      * pre_scale  + base[llane];
            float post_z = mix[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post_out[lane]   = post_v;
            }

            // comb row: float4 load (HC=4), softmax, Sinkhorn normalisation
            float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC) * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC)) * active;

            float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;

            // Initial column normalisation via simd_sum (free SIMD shuffle)
            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
            r *= col_inv;

            // Sinkhorn iterations: row-norm then col-norm, zero branches
            for (int iter = 1; iter < ITERS; ++iter) {
                r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
                r *= col_inv;
            }

            if (lane < (uint)HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 2: vectorised weighted collapse [HC, D] → [D] ──────────────
        const float p0 = pre_shared[0];
        const float p1 = pre_shared[1];
        const float p2 = pre_shared[2];
        const float p3 = pre_shared[3];

        const device T* x_row   = (const device T*)x_in + row * (HC * D);
              device U* out_row = (device U*)collapsed   + row * D;

        using T4 = vec<T, 4>;
        using U4 = vec<U, 4>;
        const device T4* x_row0 = (const device T4*)(x_row + 0 * D);
        const device T4* x_row1 = (const device T4*)(x_row + 1 * D);
        const device T4* x_row2 = (const device T4*)(x_row + 2 * D);
        const device T4* x_row3 = (const device T4*)(x_row + 3 * D);
              device U4* out4   = (device U4*)out_row;

        constexpr uint D4 = (uint)D / 4;
        for (uint d4 = tid; d4 < D4; d4 += 256) {
            float4 x0 = float4(x_row0[d4]);
            float4 x1 = float4(x_row1[d4]);
            float4 x2 = float4(x_row2[d4]);
            float4 x3 = float4(x_row3[d4]);
            out4[d4] = U4(fma(float4(p0), x0,
                         fma(float4(p1), x1,
                         fma(float4(p2), x2, float4(p3) * x3))));
        }

        // Scalar tail for D not divisible by 4
        #if (D % 4) != 0
        for (uint d = D4 * 4 + tid; d < (uint)D; d += 256) {
            float val = p0 * (float)x_row[0*D + d] + p1 * (float)x_row[1*D + d]
                      + p2 * (float)x_row[2*D + d] + p3 * (float)x_row[3*D + d];
            out_row[d] = (U)val;
        }
        #endif
    """,
    ensureRowContiguous: true
)

/// Dispatch the fused Sinkhorn-collapse kernel.
/// - Parameters:
///   - x:     4D hidden `[B, S, hc, D]` (model dtype)
///   - mixes: pre-computed projections `[B*S, (2+hc)*hc]` (float32)
///   - scale: `[3]` float32 — pre / post / comb scale factors
///   - base:  `[(2+hc)*hc]` float32 — biases
/// - Returns: `(collapsed [B, S, D], post [B, S, HC], comb [B, S, HC, HC])`
private func hcKernel(
    x: MLXArray,
    mixes: MLXArray,
    scale: MLXArray,
    base: MLXArray,
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let B = x.dim(0), S = x.dim(1), hc = x.dim(2), D = x.dim(3)
    let BL = B * S
    let epsInt = max(1, Int((eps / 1e-9).rounded()))
    let dtype = x.dtype

    let outputs = _hcSinkhornCollapseKernel(
        [x, mixes, scale, base],
        template: [
            ("T", dtype),
            ("U", dtype),
            ("HC", hcMult),
            ("ITERS", sinkhornIters),
            ("D", D),
            ("EPS_INT", epsInt),
        ],
        grid: (BL * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[BL, D], [BL, hc], [BL, hc, hc]],
        outputDTypes: [dtype, .float32, .float32]
    )

    return (
        outputs[0].reshaped([B, S, D]),
        outputs[1].reshaped([B, S, hc]),
        outputs[2].reshaped([B, S, hc, hc])
    )
}

// MARK: - HyperConnection helpers

/// Compute the three gating matrices (pre, post, comb) from the 4D hidden state.
/// Corrected from Python hyper_connection.py:
///   post = 2 * sigmoid(...)  (no eps, factor of 2)
///   comb = softmax(...) + eps  (not sigmoid)
private func hcSplitSinkhorn(
    _ mixes: MLXArray,
    hcScale: MLXArray,
    hcBase: MLXArray,
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let hc = hcMult
    let B = mixes.dim(0), S = mixes.dim(1)

    let preMix = mixes[.ellipsis, ..<hc]
    let postMix = mixes[.ellipsis, hc ..< 2 * hc]
    let combMix = mixes[.ellipsis, (2 * hc)...]

    let preBase = hcBase[..<hc]
    let postBase = hcBase[hc ..< 2 * hc]
    let combBase = hcBase[(2 * hc)...]

    // pre: sigmoid + eps, then row-normalize
    var pre = sigmoid(preMix * hcScale[0] + preBase) + eps
    pre = pre / pre.sum(axis: -1, keepDims: true)

    // post: 2 * sigmoid (no eps)
    let post = 2 * sigmoid(postMix * hcScale[1] + postBase)

    // comb: softmax init + eps, then initial col norm, then Sinkhorn
    let combScaled = combMix.reshaped([B, S, hc, hc]) * hcScale[2]
        + combBase.reshaped([hc, hc])
    var comb = softmax(combScaled, axis: -1) + eps
    comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)  // initial col norm
    for _ in 0 ..< max(sinkhornIters - 1, 0) {
        comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)  // row norm
        comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)  // col norm
    }

    return (pre, post, comb)
}

/// HC pre-sublayer: collapse [B, S, hc, D] → [B, S, D].
/// Returns (collapsed, post, comb) for use in hcPost.
///
/// Uses the fused Metal kernel on GPU (eliminates 38+ extra dispatches per call).
/// Falls back to pure-ops on CPU.
func hcPre(
    x: MLXArray,
    hcFn: MLXArray,
    hcScale: MLXArray,
    hcBase: MLXArray,
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let dtype = x.dtype
    let B = x.dim(0), S = x.dim(1), hc = x.dim(2), D = x.dim(3)

    let xFlat = x.reshaped([B, S, hc * D]).asType(.float32)
    let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)
    let mixes = matmul(xFlat, hcFn.T) * normScale  // [B, S, (2+hc)*hc]

    // Fused Metal kernel: Sinkhorn + collapse in one dispatch
    if Device.defaultDevice().deviceType == .gpu {
        return hcKernel(
            x: x,
            mixes: mixes.reshaped([B * S, mixes.dim(2)]),
            scale: hcScale.asType(.float32),
            base: hcBase.asType(.float32),
            hcMult: hcMult, sinkhornIters: sinkhornIters, eps: eps)
    }

    // Pure-ops fallback (CPU / non-Metal)
    let (pre, post, comb) = hcSplitSinkhorn(
        mixes, hcScale: hcScale, hcBase: hcBase,
        hcMult: hcMult, sinkhornIters: sinkhornIters, eps: eps)
    let y = (pre.expandedDimensions(axis: -1).asType(dtype) * x).sum(axis: -2)
    return (y, post, comb)
}

/// HC post-sublayer: expand [B, S, D] back to [B, S, hc, D].
/// term1 = post[..., None] * x[:, :, None, :]
/// term2 = comb.swapaxes(-1,-2) @ residual  (equiv to expand-multiply-sum trick)
func hcPost(
    x: MLXArray,
    residual: MLXArray,
    post: MLXArray,
    comb: MLXArray
) -> MLXArray {
    let term1 = post.expandedDimensions(axis: -1) * x.expandedDimensions(axis: -2)
    // term2: comb.swapaxes(-1,-2) @ residual
    // comb: [B, S, hc, hc], residual: [B, S, hc, D]
    // Using expand trick: combExp [B,S,hc,hc,1] * residualExp [B,S,hc,1,D] → sum axis 2
    let combExp = comb.expandedDimensions(axis: -1)
    let residualExp = residual.expandedDimensions(axis: -2)
    let term2 = (combExp * residualExp).sum(axis: 2)
    return (term1.asType(.float32) + term2.asType(.float32)).asType(x.dtype)
}

/// HC head: collapse 4D [B, S, hc, D] → [B, S, D] using HyperHead weights.
func hcHeadReduce(
    x: MLXArray,
    hcFn: MLXArray,
    hcScale: MLXArray,
    hcBase: MLXArray,
    eps: Float
) -> MLXArray {
    let dtype = x.dtype
    let B = x.dim(0), S = x.dim(1), hc = x.dim(2), D = x.dim(3)

    let xFlat = x.reshaped([B, S, hc * D]).asType(.float32)
    let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)
    let mixes = matmul(xFlat, hcFn.T) * normScale
    let pre = sigmoid(mixes * hcScale + hcBase) + eps

    return (pre.expandedDimensions(axis: -1).asType(dtype) * x).sum(axis: -2).asType(dtype)
}

// MARK: - Helpers

private func sqrtSoftplus(_ x: MLXArray) -> MLXArray {
    let sp = MLX.maximum(x, MLXArray(Float(0))) + MLX.log1p(MLX.exp(-MLX.abs(x)))
    return MLX.sqrt(sp)
}

private func headRmsNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
}

/// Extend an attention mask to cover pooled KV positions.
/// If mask is .none (decode mode), returns .none unchanged.
private func extendMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    poolMask: MLXArray?,
    L: Int,
    P: Int
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    guard case .array(let localArr) = mask, P > 0 else { return mask }
    let poolCols: MLXArray
    if let pm = poolMask {
        poolCols = pm.asType(.bool)  // [L, P]
    } else {
        poolCols = MLXArray.ones([L, P], dtype: .bool)
    }
    return .array(concatenated([localArr, poolCols], axis: -1))
}

// MARK: - Compressor

/// Compresses windows of `ratio` tokens into a single pooled KV.
/// Used by CompressedAttention (ratio=128) and SparseCompressedAttention/Indexer (ratio=4).
final class Compressor: Module {
    let compressRatio: Int
    let headDim: Int
    let overlap: Bool  // true when ratio=4
    let outDim: Int

    var wkv: Linear
    var wgate: Linear
    var ape: MLXArray   // [ratio, out_dim] – positional encoding for windows
    var norm: RMSNorm
    let rope: DeepseekV4RoPE  // non-Module: computed from hyperparams

    init(config: DeepseekV4Configuration, compressRatio: Int, headDim: Int) {
        self.compressRatio = compressRatio
        self.headDim = headDim
        self.overlap = compressRatio == 4
        self.outDim = headDim * (overlap ? 2 : 1)

        self.wkv = Linear(config.hiddenSize, outDim, bias: false)
        self.wgate = Linear(config.hiddenSize, outDim, bias: false)
        self.ape = zeros([compressRatio, outDim])
        self.norm = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.rope = DeepseekV4RoPE(
            dims: config.qkRopeHeadDim,
            base: config.compressRopeTheta,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            freqScale: compressRatio)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, poolCache: PoolingCache?, offset: Int) -> MLXArray {
        let B = x.dim(0), D = x.dim(2)
        let kv = wkv(x)    // [B, S, out_dim]
        let gate = wgate(x) // [B, S, out_dim]

        let readyKV: MLXArray
        let readyGate: MLXArray
        let poolBase: Int

        if let pc = poolCache {
            (readyKV, readyGate, poolBase) = pc.accumulateWindows(kv: kv, gate: gate, offset: offset)
        } else {
            let usable = (kv.dim(1) / compressRatio) * compressRatio
            readyKV = kv[0..., ..<usable, 0...]
            readyGate = gate[0..., ..<usable, 0...]
            poolBase = offset
        }

        let newPooled: MLXArray
        if readyKV.dim(1) == 0 {
            newPooled = zeros([B, 0, headDim], dtype: x.dtype)
        } else {
            let nWindows = readyKV.dim(1) / compressRatio
            let kvW = readyKV.reshaped([B, nWindows, compressRatio, outDim])
            let gateW = readyGate.reshaped([B, nWindows, compressRatio, outDim])
            let compressed: MLXArray
            if overlap {
                compressed = overlapCompressKV(kvW, gateW)
            } else {
                compressed = simpleCompressKV(kvW, gateW)
            }
            let normed = norm(compressed)  // [B, nWindows, headDim]
            // Apply RoPE: add head dim → [B, 1, nWindows, headDim], rope, squeeze
            let roped = rope.callAsFunction(normed.expandedDimensions(axis: 1), offset: poolBase)
            newPooled = roped.squeezed(axis: 1)
        }

        if let pc = poolCache {
            return pc.updateAndFetch(newPooled)
        }
        return newPooled
    }

    private func simpleCompressKV(_ kv: MLXArray, _ gate: MLXArray) -> MLXArray {
        // kv: [B, nWindows, ratio, out_dim=headDim]
        // weights over ratio dim
        let weights = softmax(gate.asType(.float32) + ape, axis: -2)
        return (kv * weights.asType(kv.dtype)).sum(axis: -2)  // [B, nWindows, headDim]
    }

    private func overlapCompressKV(_ kv: MLXArray, _ gate: MLXArray) -> MLXArray {
        // kv: [B, nWindows, ratio, out_dim=headDim*2]
        let B = kv.dim(0), L = kv.dim(1), R = kv.dim(2)
        let D = kv.dim(3)
        let g = gate + ape.asType(gate.dtype)

        // Split along last dim: first half = prev-window tokens, second half = current
        let kvParts = split(kv, indices: [D / 2], axis: -1)     // [B,L,R,D/2], [B,L,R,D/2]
        var kvA = kvParts[0]
        let kvB = kvParts[1]
        let gParts = split(g, indices: [D / 2], axis: -1)
        var gA = gParts[0]
        let gB = gParts[1]

        // Shift kvA by 1 window (prev window's first half)
        let kv0 = zeros([B, 1, R, D / 2], dtype: kv.dtype)
        kvA = concatenated([kv0, kvA[0..., ..<(L - 1), 0..., 0...]], axis: 1)
        let g0 = full([B, 1, R, D / 2], values: MLXArray(-Float.infinity), dtype: gate.dtype)
        gA = concatenated([g0, gA[0..., ..<(L - 1), 0..., 0...]], axis: 1)

        // Combine: [B, L, 2R, D/2]
        let kvCombined = concatenated([kvA, kvB], axis: 2)
        let gCombined = concatenated([gA, gB], axis: 2)

        let weights = softmax(gCombined.asType(.float32), axis: -2)
        return (kvCombined * weights.asType(kv.dtype)).sum(axis: -2)  // [B, L, D/2] = [B, L, headDim]
    }
}

// MARK: - Indexer

/// Builds a sparse top-k index over pooled KV positions for SparseCompressedAttention.
final class Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let indexTopk: Int
    let scale: Float

    var wq_b: Linear
    var weights_proj: Linear
    var compressor: Compressor

    init(config: DeepseekV4Configuration, compressRatio: Int) {
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.indexTopk = config.indexTopk
        self.scale = pow(Float(config.indexHeadDim), -0.5)

        self.wq_b = Linear(config.qLoraRank, nHeads * headDim, bias: false)
        self.weights_proj = Linear(config.hiddenSize, nHeads, bias: false)
        self.compressor = Compressor(config: config, compressRatio: compressRatio, headDim: headDim)

        super.init()
    }

    /// Returns top-k indices [B, L, k] into the pooled KV, or nil if pool is empty.
    func callAsFunction(
        _ x: MLXArray,
        qResidual: MLXArray,
        rope: DeepseekV4RoPE,
        poolCache: PoolingCache?,
        offset: Int
    ) -> MLXArray? {
        let B = x.dim(0), L = x.dim(1)
        let pooled = compressor(x, poolCache: poolCache, offset: offset)
        guard pooled.dim(1) > 0 else { return nil }
        let P = pooled.dim(1)

        var q = wq_b(qResidual).reshaped([B, L, nHeads, headDim])
        q = q.transposed(0, 2, 1, 3)  // [B, nHeads, L, headDim]
        q = rope.callAsFunction(q, offset: offset)

        // scores: [B, nHeads, L, P]
        let scores = q.asType(.float32)
            .matmul(pooled[0..., .newAxis, 0..., 0...].transposed(0, 1, 3, 2).asType(.float32))
        let posScores = MLX.maximum(scores, MLXArray(Float(0))) * scale

        // Head weights: [B, L, nHeads]
        let w = weights_proj(x).asType(.float32) * pow(Float(nHeads), -0.5)
        // Combine heads: [B, L, P]
        let combined = (posScores * w.transposed(0, 2, 1).expandedDimensions(axis: -1))
            .sum(axis: 1)

        // Apply causal pool mask if in prefill
        var maskedScores = combined
        if let pc = poolCache, let pm = pc.makeMask(L: L, offset: offset) {
            maskedScores = MLX.where(pm[0..., 0...].expandedDimensions(axis: 0), combined,
                MLXArray(Float(-Float.infinity)))
        }

        let k = min(indexTopk, P)
        return argPartition(-maskedScores, kth: k - 1, axis: -1)[0..., 0..., ..<k]
    }
}

// MARK: - MoEGate

class DeepseekV4Gate: Module {
    let topK: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let scoringFunc: String
    let isHash: Bool

    var weight: MLXArray
    var e_score_correction_bias: MLXArray?
    var tid2eid: MLXArray?  // [vocab_size, topK] int32 – only for hash layers

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.topK = config.numExpertsPerTok
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self.scoringFunc = config.scoringFunc
        self.isHash = layerIdx < config.numHashLayers
        self.weight = zeros([config.nRoutedExperts, config.hiddenSize])
        if isHash {
            self.tid2eid = zeros([config.vocabSize, config.numExpertsPerTok], dtype: .int32)
        } else {
            self.e_score_correction_bias = zeros([config.nRoutedExperts])
        }
        super.init()
    }

    override func updateMissing(
        parameter: String, verify: VerifyUpdate, path: [String], modulePath: [String]
    ) throws {
        // e_score_correction_bias absent on hash layers; tid2eid absent on normal layers
        if parameter == "e_score_correction_bias" || parameter == "tid2eid" { return }
        try super.updateMissing(
            parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }

    func callAsFunction(_ x: MLXArray, inputIds: MLXArray?) -> (MLXArray, MLXArray) {
        let logits = x.matmul(weight.T).asType(.float32)

        let scores: MLXArray
        switch scoringFunc {
        case "softmax": scores = softmax(logits, axis: -1)
        case "sigmoid": scores = sigmoid(logits)
        default: scores = sqrtSoftplus(logits)
        }

        let inds: MLXArray
        let selectedScores: MLXArray

        if isHash, let ids = inputIds, let t2e = tid2eid {
            // Hash routing: expert IDs determined by token ID
            inds = t2e[ids]  // [B, S, topK]
            selectedScores = takeAlong(scores, inds, axis: -1)
        } else {
            let bias = e_score_correction_bias ?? zeros([weight.dim(0)])
            let scoresForChoice = scores + bias
            inds = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            selectedScores = takeAlong(scores, inds, axis: -1)
        }

        var ws = selectedScores
        if scoringFunc != "softmax" && normTopkProb {
            ws = ws / (ws.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (inds, ws * routedScalingFactor)
    }
}

// MARK: - DeepseekV4MLP (for shared experts)

class DeepseekV4MLP: Module, UnaryLayer {
    var gate_proj: Linear
    var up_proj: Linear
    var down_proj: Linear
    let swiguLimit: Float

    init(hiddenSize: Int, intermediateSize: Int, swiguLimit: Float) {
        self.gate_proj = Linear(hiddenSize, intermediateSize, bias: false)
        self.up_proj = Linear(hiddenSize, intermediateSize, bias: false)
        self.down_proj = Linear(intermediateSize, hiddenSize, bias: false)
        self.swiguLimit = swiguLimit
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var g = gate_proj(x)
        var u = up_proj(x)
        if swiguLimit > 0 {
            g = MLX.minimum(g, MLXArray(swiguLimit))
            u = clip(u, min: MLXArray(-swiguLimit), max: MLXArray(swiguLimit))
        }
        return down_proj(silu(g) * u)
    }
}

// MARK: - DeepseekV4MoE

class DeepseekV4MoE: Module {
    let numExpertsPerTok: Int
    let swiguLimit: Float

    var gate: DeepseekV4Gate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.numExpertsPerTok = config.numExpertsPerTok
        self.swiguLimit = config.swiguLimit
        self.gate = DeepseekV4Gate(config: config, layerIdx: layerIdx)
        let limit = config.swiguLimit
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: { x in limit > 0 ? silu(clip(x, min: -limit, max: limit)) : silu(x) })
        self._sharedExperts.wrappedValue = DeepseekV4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swiguLimit: config.swiguLimit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, inputIds: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x, inputIds: inputIds)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis].asType(y.dtype)).sum(axis: -2)
        return y + sharedExperts(x)
    }
}

// MARK: - Attention output projection helper

private func applyOutputProjection(
    _ out: MLXArray,
    woA: DeepseekV4MultiLinear,
    woB: Linear,
    oGroups: Int,
    oLoraRank: Int,
    numHeads: Int,
    headDim: Int
) -> MLXArray {
    // out: [B, numHeads, L, headDim]
    let B = out.dim(0), L = out.dim(2)
    let nHeadsPerGroup = numHeads / oGroups
    let groupFeat = nHeadsPerGroup * headDim

    let r = out
        .reshaped([B, oGroups, nHeadsPerGroup, L, headDim])  // [B, oG, nHpG, L, hD]
        .transposed(0, 1, 3, 2, 4)                            // [B, oG, L, nHpG, hD]
        .reshaped([B, oGroups, L, groupFeat])                  // [B, oG, L, groupFeat]
    let lora = woA(r)                                          // [B, oG, L, oLoraRank]
    return woB(
        lora
            .transposed(0, 2, 1, 3)                           // [B, L, oG, oLoraRank]
            .reshaped([B, L, oGroups * oLoraRank]))            // [B, L, oG*oLoraRank]
}

// MARK: - LocalAttention

/// Attention with no KV compression (compress_ratio=0).
/// Uses ropeTheta=10000.
final class LocalAttention: Module {
    let numHeads: Int
    let headDim: Int
    let oGroups: Int
    let oLoraRank: Int
    let scale: Float
    let eps: Float

    let rope: DeepseekV4RoPE  // non-Module

    var wq_a: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    var wq_b: Linear
    var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    var wo_a: DeepseekV4MultiLinear
    var wo_b: Linear
    var attn_sink: MLXArray

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.scale = pow(Float(config.headDim), -0.5)
        self.eps = config.rmsNormEps

        self.wq_a = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self.wq_b = Linear(config.qLoraRank, config.numAttentionHeads * config.headDim, bias: false)
        self.wkv = Linear(config.hiddenSize, config.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self.wo_a = DeepseekV4MultiLinear(
            inFeatures: config.numAttentionHeads * config.headDim / config.oGroups,
            outFeatures: config.oLoraRank,
            groups: config.oGroups)
        self.wo_b = Linear(config.oGroups * config.oLoraRank, config.hiddenSize, bias: false)
        self.attn_sink = zeros([config.numAttentionHeads])

        // Local attention uses base rope_theta (not compress_rope_theta)
        self.rope = DeepseekV4RoPE(
            dims: config.qkRopeHeadDim,
            base: config.ropeTheta,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        let offset = cache?.offset ?? 0

        var q = wq_b(qNorm(wq_a(x)))
        q = q.reshaped([B, L, numHeads, headDim])
        q = headRmsNorm(q, eps: eps)
        q = q.transposed(0, 2, 1, 3)       // [B, numHeads, L, headDim]
        q = rope.callAsFunction(q, offset: offset)

        var kv = kvNorm(wkv(x)).reshaped([B, 1, L, headDim])
        kv = rope.callAsFunction(kv, offset: offset)
        if let c = cache {
            let (cached, _) = c.update(keys: kv, values: kv)
            kv = cached
        }

        let sinks: MLXArray? = attn_sink.sum().item(Float.self) != 0
            ? attn_sink.asType(q.dtype) : nil
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: kv, values: kv,
            scale: scale, mask: mask, sinks: sinks)  // [B, numHeads, L, headDim]

        out = rope.callAsFunction(out, offset: offset, inverse: true)

        return applyOutputProjection(out, woA: wo_a, woB: wo_b,
            oGroups: oGroups, oLoraRank: oLoraRank, numHeads: numHeads, headDim: headDim)
    }
}

// MARK: - CompressedAttention

/// Attention with pooled KV compression (compress_ratio=128).
/// Uses compressRopeTheta=160000 with Yarn scaling.
final class CompressedAttention: Module {
    let numHeads: Int
    let headDim: Int
    let oGroups: Int
    let oLoraRank: Int
    let scale: Float
    let eps: Float

    let rope: DeepseekV4RoPE
    var compressor: Compressor

    var wq_a: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    var wq_b: Linear
    var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    var wo_a: DeepseekV4MultiLinear
    var wo_b: Linear
    var attn_sink: MLXArray

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.scale = pow(Float(config.headDim), -0.5)
        self.eps = config.rmsNormEps

        self.wq_a = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self.wq_b = Linear(config.qLoraRank, config.numAttentionHeads * config.headDim, bias: false)
        self.wkv = Linear(config.hiddenSize, config.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self.wo_a = DeepseekV4MultiLinear(
            inFeatures: config.numAttentionHeads * config.headDim / config.oGroups,
            outFeatures: config.oLoraRank,
            groups: config.oGroups)
        self.wo_b = Linear(config.oGroups * config.oLoraRank, config.hiddenSize, bias: false)
        self.attn_sink = zeros([config.numAttentionHeads])

        self.rope = DeepseekV4RoPE(
            dims: config.qkRopeHeadDim,
            base: config.compressRopeTheta,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        self.compressor = Compressor(
            config: config,
            compressRatio: config.compressRatios[layerIdx],
            headDim: config.headDim)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        let layerCache = cache as? DeepseekV4LayerCache
        let localCache = layerCache?.rotating
        let poolCache = layerCache?.pooling.first
        let offset = localCache?.offset ?? 0

        var q = wq_b(qNorm(wq_a(x)))
        q = q.reshaped([B, L, numHeads, headDim])
        q = headRmsNorm(q, eps: eps)
        q = q.transposed(0, 2, 1, 3)
        q = rope.callAsFunction(q, offset: offset)

        var kv = kvNorm(wkv(x)).reshaped([B, 1, L, headDim])
        kv = rope.callAsFunction(kv, offset: offset)
        if let lc = localCache {
            let (cached, _) = lc.update(keys: kv, values: kv)
            kv = cached
        }

        // Pooled KV from Compressor
        let pooled = compressor(x, poolCache: poolCache, offset: offset)  // [B, P, headDim]
        var effectiveMask = mask
        let P = pooled.dim(1)
        if P > 0 {
            let poolMask = poolCache?.makeMask(L: L, offset: offset)
            let fullKV = concatenated([kv, pooled.expandedDimensions(axis: 1)], axis: 2)
            effectiveMask = extendMask(mask, poolMask: poolMask, L: L, P: P)

            let sinks: MLXArray? = attn_sink.sum().item(Float.self) != 0
                ? attn_sink.asType(q.dtype) : nil
            var out = MLXFast.scaledDotProductAttention(
                queries: q, keys: fullKV, values: fullKV,
                scale: scale, mask: effectiveMask, sinks: sinks)
            out = rope.callAsFunction(out, offset: offset, inverse: true)
            return applyOutputProjection(out, woA: wo_a, woB: wo_b,
                oGroups: oGroups, oLoraRank: oLoraRank, numHeads: numHeads, headDim: headDim)
        }

        let sinks: MLXArray? = attn_sink.sum().item(Float.self) != 0
            ? attn_sink.asType(q.dtype) : nil
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: kv, values: kv,
            scale: scale, mask: mask, sinks: sinks)
        out = rope.callAsFunction(out, offset: offset, inverse: true)
        return applyOutputProjection(out, woA: wo_a, woB: wo_b,
            oGroups: oGroups, oLoraRank: oLoraRank, numHeads: numHeads, headDim: headDim)
    }
}

// MARK: - SparseCompressedAttention

/// Attention with sparse indexed pooled KV (compress_ratio=4).
final class SparseCompressedAttention: Module {
    let numHeads: Int
    let headDim: Int
    let oGroups: Int
    let oLoraRank: Int
    let scale: Float
    let eps: Float

    let rope: DeepseekV4RoPE
    var compressor: Compressor
    var indexer: Indexer

    var wq_a: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    var wq_b: Linear
    var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    var wo_a: DeepseekV4MultiLinear
    var wo_b: Linear
    var attn_sink: MLXArray

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.scale = pow(Float(config.headDim), -0.5)
        self.eps = config.rmsNormEps

        self.wq_a = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self.wq_b = Linear(config.qLoraRank, config.numAttentionHeads * config.headDim, bias: false)
        self.wkv = Linear(config.hiddenSize, config.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self.wo_a = DeepseekV4MultiLinear(
            inFeatures: config.numAttentionHeads * config.headDim / config.oGroups,
            outFeatures: config.oLoraRank,
            groups: config.oGroups)
        self.wo_b = Linear(config.oGroups * config.oLoraRank, config.hiddenSize, bias: false)
        self.attn_sink = zeros([config.numAttentionHeads])

        self.rope = DeepseekV4RoPE(
            dims: config.qkRopeHeadDim,
            base: config.compressRopeTheta,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        let compressRatio = config.compressRatios[layerIdx]
        self.compressor = Compressor(config: config, compressRatio: compressRatio, headDim: config.headDim)
        self.indexer = Indexer(config: config, compressRatio: compressRatio)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        let layerCache = cache as? DeepseekV4LayerCache
        let localCache = layerCache?.rotating
        let compCache = layerCache?.pooling.first
        let idxCache = layerCache?.pooling.count ?? 0 > 1 ? layerCache?.pooling[1] : nil
        let offset = localCache?.offset ?? 0

        let qResidual = qNorm(wq_a(x))
        var q = wq_b(qResidual).reshaped([B, L, numHeads, headDim])
        q = headRmsNorm(q, eps: eps)
        q = q.transposed(0, 2, 1, 3)
        q = rope.callAsFunction(q, offset: offset)

        var kv = kvNorm(wkv(x)).reshaped([B, 1, L, headDim])
        kv = rope.callAsFunction(kv, offset: offset)
        if let lc = localCache {
            let (cached, _) = lc.update(keys: kv, values: kv)
            kv = cached
        }

        let pooled = compressor(x, poolCache: compCache, offset: offset)
        let pmask = compCache?.makeMask(L: L, offset: offset)
        let topk = indexer(x, qResidual: qResidual, rope: rope, poolCache: idxCache, offset: offset)
        let sinks: MLXArray? = attn_sink.sum().item(Float.self) != 0
            ? attn_sink.asType(q.dtype) : nil
        let P = pooled.dim(1)
        let indexTopk = indexer.indexTopk

        let out: MLXArray
        if P == 0 {
            // No pooled KV yet – pure local attention
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: kv, values: kv,
                scale: scale, mask: mask, sinks: sinks)
        } else if P <= indexTopk {
            // Pool fits in topk – full attention with all pooled
            let fullKV = concatenated([kv, pooled.expandedDimensions(axis: 1)], axis: 2)
            let extMask = extendMask(mask, poolMask: pmask, L: L, P: P)
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: fullKV, values: fullKV,
                scale: scale, mask: extMask, sinks: sinks)
        } else if let tk = topk {
            // Sparse: gather top-k pooled vectors and attend
            out = sparsePooledAttention(
                q: q, localKV: kv, pooled: pooled, topk: tk,
                localMask: mask, pooledMask: pmask, sinks: sinks)
        } else {
            // Indexer returned nil (shouldn't happen if P > 0 and indexer returns)
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: kv, values: kv,
                scale: scale, mask: mask, sinks: sinks)
        }

        let result = rope.callAsFunction(out, offset: offset, inverse: true)
        return applyOutputProjection(result, woA: wo_a, woB: wo_b,
            oGroups: oGroups, oLoraRank: oLoraRank, numHeads: numHeads, headDim: headDim)
    }

    /// Sparse pooled attention: gather top-k pooled KVs, combine with local in log-space.
    private func sparsePooledAttention(
        q: MLXArray,
        localKV: MLXArray,
        pooled: MLXArray,
        topk: MLXArray,     // [B, L, k]
        localMask: MLXFast.ScaledDotProductAttentionMaskMode,
        pooledMask: MLXArray?,
        sinks: MLXArray?
    ) -> MLXArray {
        let B = q.dim(0), L = q.dim(2), D = q.dim(3)
        let k = topk.dim(2)

        // Gather top-k pooled vectors: [B, L, k, D]
        let flatTopk = topk.reshaped([B, L * k])            // [B, L*k]
        let expandedIdx = flatTopk.expandedDimensions(axis: -1)  // [B, L*k, 1]
        let tiledIdx = expandedIdx + zeros([B, L * k, D], dtype: .int32)  // [B, L*k, D]
        let gathered = takeAlong(pooled, tiledIdx, axis: 1)  // [B, L*k, D]
        let pooledTopk = gathered.reshaped([B, L, k, D])    // [B, L, k, D]

        let qScaled = q * scale

        // Local scores: [B, H, L, localLen]
        let localScores: MLXArray
        switch localMask {
        case .none, .causal:
            localScores = qScaled.matmul(localKV.transposed(0, 1, 3, 2))
        case .array(let arr):
            let raw = qScaled.matmul(localKV.transposed(0, 1, 3, 2))
            let arrExpanded = arr.expandedDimensions(axis: 0).expandedDimensions(axis: 1)
            localScores = MLX.where(arrExpanded, raw, MLXArray(Float(-Float.infinity)))
        @unknown default:
            localScores = qScaled.matmul(localKV.transposed(0, 1, 3, 2))
        }

        let maxLocal = localScores.max(axis: -1, keepDims: true)
        var logNorm = maxLocal + MLX.log(
            MLX.exp(localScores - maxLocal).sum(axis: -1, keepDims: true) + 1e-20)

        // Pooled scores: [B, H, L, k]
        let qBL = qScaled.transposed(0, 2, 1, 3)  // [B, L, H, D]
        var pooledScores = qBL.matmul(pooledTopk.transposed(0, 1, 3, 2))  // [B, L, H, k]
        pooledScores = pooledScores.transposed(0, 2, 1, 3)  // [B, H, L, k]

        // Apply sparse pool mask
        if let pm = pooledMask {
            // topk: [B, L, k], pm: [L, P] or [B, L, P]
            let pmExpanded = pm.ndim == 2 ? pm.expandedDimensions(axis: 0) : pm
            let sparsePM = takeAlong(pmExpanded, topk, axis: -1)  // [B, L, k]
            pooledScores = MLX.where(
                sparsePM[0..., .newAxis, 0..., 0...],
                pooledScores,
                MLXArray(Float(-Float.infinity)))
        }

        let maxPooled = pooledScores.max(axis: -1, keepDims: true)
        let logNormPooled = maxPooled + MLX.log(
            MLX.exp(pooledScores - maxPooled).sum(axis: -1, keepDims: true) + 1e-20)
        logNorm = logNorm + MLX.log1p(MLX.exp(logNormPooled - logNorm))

        if let s = sinks {
            logNorm = logNorm + MLX.log1p(MLX.exp(s[.newAxis, 0..., .newAxis, .newAxis] - logNorm))
        }

        let localWeights = MLX.exp(localScores - logNorm)
        let pooledWeights = MLX.exp(pooledScores - logNorm)

        let outLocal = localWeights.matmul(localKV)  // [B, H, L, D]
        let pwBL = pooledWeights.transposed(0, 2, 1, 3)  // [B, L, H, k]
        let outPooled = pwBL.matmul(pooledTopk).transposed(0, 2, 1, 3)  // [B, H, L, D]
        return (outLocal + outPooled).asType(q.dtype)
    }
}

// MARK: - Attention factory

private func v4AttentionFactory(config: DeepseekV4Configuration, layerIdx: Int) -> Module {
    let ratio = layerIdx < config.compressRatios.count ? config.compressRatios[layerIdx] : 0
    if ratio == 0 { return LocalAttention(config: config, layerIdx: layerIdx) }
    if ratio == 128 { return CompressedAttention(config: config, layerIdx: layerIdx) }
    return SparseCompressedAttention(config: config, layerIdx: layerIdx)
}

// MARK: - Decoder Block

class DeepseekV4Block: Module {
    let config: DeepseekV4Configuration

    @ModuleInfo(key: "attn") var attn: Module
    var ffn: DeepseekV4MoE
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    var attn_hc: HCParams  // key matches Python attn_hc.fn/base/scale
    var ffn_hc: HCParams

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self._attn.wrappedValue = v4AttentionFactory(config: config, layerIdx: layerIdx)
        self.ffn = DeepseekV4MoE(config: config, layerIdx: layerIdx)
        self._attnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        let hc = config.hcMult
        let mixHc = (2 + hc) * hc
        let hcDim = hc * config.hiddenSize
        self.attn_hc = HCParams(fn: zeros([mixHc, hcDim]), base: zeros([mixHc]), scale: ones([3]))
        self.ffn_hc = HCParams(fn: zeros([mixHc, hcDim]), base: zeros([mixHc]), scale: ones([3]))

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?,
        inputIds: MLXArray
    ) -> MLXArray {
        // Attention sub-block
        var residual = x
        let (xAttn, postAttn, combAttn) = hcPre(
            x: residual, hcFn: attn_hc.fn, hcScale: attn_hc.scale, hcBase: attn_hc.base,
            hcMult: config.hcMult, sinkhornIters: config.hcSinkhornIters, eps: config.hcEps)

        let attnOut: MLXArray
        let normedAttn = attnNorm(xAttn)
        if let la = attn as? LocalAttention {
            attnOut = la(normedAttn, mask: mask, cache: cache)
        } else if let ca = attn as? CompressedAttention {
            attnOut = ca(normedAttn, mask: mask, cache: cache)
        } else if let sca = attn as? SparseCompressedAttention {
            attnOut = sca(normedAttn, mask: mask, cache: cache)
        } else {
            fatalError("Unknown attention type in DeepseekV4Block")
        }

        var h = hcPost(x: attnOut, residual: residual, post: postAttn, comb: combAttn)

        // FFN sub-block
        residual = h
        let (xFfn, postFfn, combFfn) = hcPre(
            x: residual, hcFn: ffn_hc.fn, hcScale: ffn_hc.scale, hcBase: ffn_hc.base,
            hcMult: config.hcMult, sinkhornIters: config.hcSinkhornIters, eps: config.hcEps)
        let ffnOut = ffn(ffnNorm(xFfn), inputIds: inputIds)
        return hcPost(x: ffnOut, residual: residual, post: postFfn, comb: combFfn)
    }
}

// MARK: - Inner Model

public class DeepseekV4ModelInner: Module {
    var config: DeepseekV4Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    var hc_head: HCParams

    init(config: DeepseekV4Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekV4Block(config: config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        let hc = config.hcMult
        self.hc_head = HCParams(
            fn: zeros([hc, hc * config.hiddenSize]),
            base: zeros([hc]),
            scale: ones([1]))
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let (h, _) = forward(inputIds, cache: cache, returnRawHidden: false)
        return h
    }

    /// Forward with optional return of the raw 4D hidden state (for MTP).
    func forward(
        _ inputIds: MLXArray,
        cache: [any KVCache]?,
        returnRawHidden: Bool
    ) -> (MLXArray, MLXArray?) {
        let B = inputIds.dim(0), S = inputIds.dim(1)
        let hc = config.hcMult

        let emb = embedTokens(inputIds)
        var h = repeated(emb.expandedDimensions(axis: 2), count: hc, axis: 2)  // [B, S, hc, D]

        // Compute attention mask from first rotating cache
        let firstRotatingCache: (any KVCache)? = {
            guard let c = cache?.first else { return nil }
            return (c as? DeepseekV4LayerCache)?.rotating ?? c
        }()
        let hForMask = h.reshaped([B, S, hc * config.hiddenSize])
        let maskMode = createAttentionMask(
            h: hForMask,
            cache: firstRotatingCache,
            windowSize: config.slidingWindow,
            returnArray: true)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: maskMode, cache: cache?[i], inputIds: inputIds)
        }

        let rawHidden = returnRawHidden ? h : nil
        h = hcHeadReduce(
            x: h, hcFn: hc_head.fn, hcScale: hc_head.scale, hcBase: hc_head.base,
            eps: config.hcEps)
        return (norm(h), rawHidden)
    }
}

// MARK: - Top-level Model

public class DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int]

    var args: DeepseekV4Configuration
    public var model: DeepseekV4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    /// MTP prediction blocks. Non-nil only when `_deepseekV4MTPEnabled == true` at init time.
    var mtp: [DeepseekV4MTPBlock]?

    init(_ args: DeepseekV4Configuration) {
        var config = args
        config.fillDefaultCompressRatios()
        self.args = config
        self.kvHeads = Array(repeating: 1, count: config.numHiddenLayers)
        self.model = DeepseekV4ModelInner(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)

        if config.numNextnPredictLayers > 0 && _deepseekV4MTPEnabled {
            let n = config.numHiddenLayers
            self.mtp = (0 ..< config.numNextnPredictLayers).map {
                DeepseekV4MTPBlock(config: config, layerIdx: n + $0)
            }
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]? = nil) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public func makeCache(parameters: GenerateParameters) -> [any KVCache] {
        args.compressRatios.map { ratio in
            if ratio == 0 {
                return RotatingKVCache(maxSize: args.slidingWindow) as any KVCache
            } else if ratio == 4 {
                return DeepseekV4LayerCache(
                    rotating: RotatingKVCache(maxSize: args.slidingWindow),
                    pooling: [PoolingCache(ratio: ratio), PoolingCache(ratio: ratio)]
                ) as any KVCache
            } else {
                // ratio == 128
                return DeepseekV4LayerCache(
                    rotating: RotatingKVCache(maxSize: args.slidingWindow),
                    pooling: [PoolingCache(ratio: ratio)]
                ) as any KVCache
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights
        let hasMTP = mtp != nil
        let hasMTPWeights = weights.keys.contains { $0.hasPrefix("mtp.") }

        if hasMTP && !hasMTPWeights {
            self.mtp = nil
        }

        // FP8 dequantization
        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs
            var p = MLX.padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            p = p.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = p * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }
        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let wt = weights[weightKey] {
                    w[weightKey] = dequant(weight: wt, scaleInv: value)
                }
            } else if w[key] == nil {
                w[key] = value
            }
        }

        // Remap keys: hc_attn_fn → attn_hc.fn, w1 → gate_proj, etc.
        // Also: ffn.gate.bias → ffn.gate.e_score_correction_bias
        // Also: model.embed → model.embed_tokens, etc.
        let topRemap: [String: String] = [
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        ]
        let wRemap: [String: String] = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]

        var remapped: [String: MLXArray] = [:]
        for (key, value) in w {
            var nk = key.hasPrefix("layers.") ? "model." + key : key
            // Remap flat hc keys: .hc_attn_fn → .attn_hc.fn
            for sub in ["attn", "ffn"] {
                for param in ["fn", "base", "scale"] {
                    nk = nk.replacingOccurrences(
                        of: ".hc_\(sub)_\(param)", with: ".\(sub)_hc.\(param)")
                }
            }
            // Gate bias rename
            nk = nk.replacingOccurrences(
                of: ".ffn.gate.bias", with: ".ffn.gate.e_score_correction_bias")
            // Shared expert weight name remap
            for (old, new) in wRemap {
                nk = nk.replacingOccurrences(
                    of: ".shared_experts.\(old).", with: ".shared_experts.\(new).")
            }
            // Top-level remap
            if let mapped = topRemap[nk] { remapped[mapped] = value }
            else { remapped[nk] = value }
        }
        w = remapped

        // Stack per-expert routed weights → switch_mlp format
        for l in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for (src, dst) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for suffix in ["weight", "scales"] {
                    let key0 = "\(prefix).ffn.experts.0.\(src).\(suffix)"
                    if w[key0] != nil {
                        let stacked = (0 ..< args.nRoutedExperts).compactMap {
                            w.removeValue(forKey: "\(prefix).ffn.experts.\($0).\(src).\(suffix)")
                        }
                        if stacked.count == args.nRoutedExperts {
                            w["\(prefix).ffn.switch_mlp.\(dst).\(suffix)"] = MLX.stacked(stacked)
                        }
                    }
                }
            }
        }

        // Stack MTP expert weights
        if hasMTP && hasMTPWeights {
            for i in 0 ..< args.numNextnPredictLayers {
                let prefix = "mtp.\(i).block.ffn.experts"
                for (src, dst) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                    for suffix in ["weight", "scales"] {
                        let key0 = "\(prefix).0.\(src).\(suffix)"
                        if w[key0] != nil {
                            let stacked = (0 ..< args.nRoutedExperts).compactMap {
                                w.removeValue(forKey: "\(prefix).\($0).\(src).\(suffix)")
                            }
                            if stacked.count == args.nRoutedExperts {
                                w["mtp.\(i).block.ffn.switch_mlp.\(dst).\(suffix)"] = MLX.stacked(stacked)
                            }
                        }
                    }
                }
            }
        }

        // Reshape wo_a weights from 2D → 3D [o_groups, o_lora_rank, -1]
        for l in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(l).attn.wo_a"
            for key in ["\(prefix).weight", "\(prefix).scales", "\(prefix).biases"] {
                if let v = w[key], v.ndim == 2 {
                    w[key] = v.reshaped([args.oGroups, args.oLoraRank, -1])
                }
            }
        }

        return w.filter { key, _ in
            if key.hasPrefix("mtp.") && !hasMTP { return false }
            if key.contains("rotary_emb.inv_freq") { return false }
            return true
        }
    }

    public var loraLayers: [Module] { model.layers }
}
