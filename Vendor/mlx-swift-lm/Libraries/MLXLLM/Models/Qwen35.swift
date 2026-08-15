//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    // MTP — number of Multi-Token Prediction head layers.
    // Port of omlx commit 696d90a: patches/mlx_lm_mtp/qwen35_model.py
    // `_patch_text_model_args` attaches this from config.json at runtime.
    var mtpNumHiddenLayers: Int = 0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.mtpNumHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }
}

// MARK: - GatedDeltaNet

final class Qwen35GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    // MARK: - _processChunk (MTP helper)

    /// Process one time-chunk of the linear-attention layer.
    ///
    /// Extracted from `callAsFunction` so the MTP verify cycle can run the prefix
    /// (n_confirmed tokens) and draft suffix separately, snapshotting the SSM/conv
    /// state in between for rollback on draft rejection.
    ///
    /// Port of omlx commit 696d90a:
    ///   patches/mlx_lm_mtp/qwen35_model.py `GatedDeltaNet._process_chunk`
    ///
    /// - Parameters:
    ///   - qkv: Already-masked QKV for this chunk [B, S_chunk, conv_dim]
    ///   - a, b: Input projections for this chunk [B, S_chunk, ...]
    ///   - convState: Initial conv state [B, conv_kernel_size-1, conv_dim]
    ///   - ssmState: Initial SSM state (nil on first token)
    ///   - mask: SSM mask for `gatedDeltaUpdate` (optional)
    /// - Returns: `(out, newConvState, newSsmState)`
    private func processChunk(
        qkv: MLXArray,
        a: MLXArray,
        b: MLXArray,
        convState: MLXArray,
        ssmState: MLXArray?,
        mask: MLXArray?
    ) -> (out: MLXArray, newConvState: MLXArray, newSsmState: MLXArray) {
        let B = qkv.dim(0)
        let S = qkv.dim(1)

        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = convInput[0..., (convInput.dim(1) - nKeep)...]
        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, newSsmState) = gatedDeltaUpdate(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: ssmState,
            mask: mask
        )
        return (out, newConvState, newSsmState)
    }

    // MARK: - fused-verify boundary stash (MTP accept-path helper)

    /// STASHING TWIN of ``processChunk(qkv:a:b:convState:ssmState:mask:)``.
    ///
    /// Identical arithmetic, identical order, identical results -- it just also
    /// hands back everything needed to rebuild the post-row-0 recurrent boundary
    /// later, WITHOUT re-running conv1d / silu / rmsNorm:
    ///
    /// - the post-row-0 convolution state is `convInput[1 ..< 1 + nKeep]`,
    ///   i.e. a different window of the very array this chunk convolved. The
    ///   split path's chunk-C computes `concat(convState, row0)[-nKeep:]`, which
    ///   is the same three rows. Bit-exact by slicing, not by recomputation.
    /// - the recurrence inputs are the row-0 slices of the tensors this chunk
    ///   has ALREADY produced (`qNormed`, `kNormed`, `v`) plus the row-0 gate
    ///   projections and the incoming ssm state.
    ///
    /// Kept as a separate function rather than a flag on `processChunk` so the
    /// shipped split path stays byte-for-byte the code that earned the current
    /// frontier. KEEP THE TWO IN SYNC if `processChunk` ever changes.
    private func processChunkStashingRow0(
        qkv: MLXArray,
        a: MLXArray,
        b: MLXArray,
        convState: MLXArray,
        ssmState: MLXArray?,
        mask: MLXArray?
    ) -> (
        out: MLXArray, newConvState: MLXArray, newSsmState: MLXArray,
        stash: ArraysCache.FusedPrimaryBoundary
    ) {
        let B = qkv.dim(0)
        let S = qkv.dim(1)

        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = convInput[0..., (convInput.dim(1) - nKeep)...]
        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, newSsmState) = gatedDeltaUpdate(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: ssmState,
            mask: mask
        )

        // Row-0 boundary ingredients. Every leaf is a slice expression, so it
        // references the value at capture time and cannot be rewritten by a
        // later in-place cache write (same discipline as `snapshotRecurrent`).
        let stash = ArraysCache.FusedPrimaryBoundary(
            conv: convInput[0..., 1 ..< (1 + nKeep)],
            q: qNormed[0..., 0 ..< 1, 0...],
            k: kNormed[0..., 0 ..< 1, 0...],
            v: v[0..., 0 ..< 1, 0...],
            a: a[0..., 0 ..< 1, 0...],
            b: b[0..., 0 ..< 1, 0...],
            ssmPre: ssmState.map { $0[.ellipsis] },
            mask: mask.map { $0[0..., 0 ..< 1] }
        )
        return (out, newConvState, newSsmState, stash)
    }

    /// Rebuild the post-row-0 recurrent boundary from a stash recorded by
    /// ``processChunkStashingRow0(qkv:a:b:convState:ssmState:mask:)`` and write
    /// it into `cache`. Returns `false` (leaving the cache untouched) when no
    /// stash is present, so the caller can fall back to a generic repair.
    ///
    /// BIT-EXACT. `gatedDeltaUpdate` at `T == 1` runs the same Metal kernel the
    /// fused call ran; the kernel's per-`t` body is a fixed sequence of fp32
    /// operations on a register-resident state, so executing iteration `t == 0`
    /// alone from `ssmPre` reproduces exactly the state the fused call held
    /// after its own `t == 0` (`StT` is fp32, so the extra store/load round trip
    /// is lossless). The convolution half is not recomputed at all -- it is the
    /// slice taken at stash time.
    func recomputeFusedPrimaryBoundary(cache: MambaCache) -> Bool {
        guard let stash = cache.fusedPrimaryBoundary else { return false }
        let (_, boundarySsm) = gatedDeltaUpdate(
            q: stash.q,
            k: stash.k,
            v: stash.v,
            a: stash.a,
            b: stash.b,
            aLog: aLog,
            dtBias: dtBias,
            state: stash.ssmPre,
            mask: stash.mask
        )
        cache[0] = stash.conv
        cache[1] = boundarySsm
        cache.fusedPrimaryBoundary = nil
        cache.rollbackState = nil
        return true
    }

    // MARK: - callAsFunction

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil,
        nConfirmed: Int = 0,
        stashPrimaryBoundary: Bool = false
    ) -> MLXArray {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py GatedDeltaNet.__call__
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        // Apply mask to full qkv before any chunking.
        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let ssmState = cache?[1]
        let out: MLXArray
        let finalConvState: MLXArray
        let finalSsmState: MLXArray
        var pendingStash: ArraysCache.FusedPrimaryBoundary?

        if stashPrimaryBoundary && nConfirmed == 0 && S > 1 {
            // FUSED VERIFY WITH A LAZY BOUNDARY (MTP accept path).
            //
            // The `nConfirmed > 0` branch below buys a free reject by paying a
            // SPLIT on every verify: two chunks means two `conv1d`, two `silu`,
            // two 3-way splits, four `rmsNorm`s, two recurrence launches and an
            // output `concatenated` per gated-delta layer -- on all 48 of them,
            // including the ~2/3 of rounds that fully accept and never look at
            // the checkpoint.
            //
            // This branch runs the same rows as ONE chunk (exactly the geometry
            // every pre-checkpoint submission verified with) and instead records
            // the ingredients for rebuilding the post-row-0 boundary with a
            // single recurrence step, consumed only if the draft is rejected.
            let (o, c, s, stash) = processChunkStashingRow0(
                qkv: qkv, a: a, b: b,
                convState: convState,
                ssmState: ssmState,
                mask: mask
            )
            out = o
            finalConvState = c
            finalSsmState = s
            pendingStash = stash
        } else if nConfirmed > 0 && nConfirmed < S {
            // Split at nConfirmed boundary for the MTP 2-token verify forward.
            // Run confirmed prefix first, snapshot rollback state, then run draft.
            // omlx: GatedDeltaNet.__call__ nConfirmed > 0 branch
            let maskC = mask.map { $0[0..., 0..<nConfirmed] }
            let maskD = mask.map { $0[0..., nConfirmed...] }

            let (outC, convC, ssmC) = processChunk(
                qkv: qkv[0..., 0..<nConfirmed, 0...],
                a: a[0..., 0..<nConfirmed, 0...],
                b: b[0..., 0..<nConfirmed, 0...],
                convState: convState,
                ssmState: ssmState,
                mask: maskC
            )
            // Snapshot (conv_state, ssm_state) after confirmed prefix for rollback.
            // omlx: cache.rollback_state = (conv_c, ssm_c)
            cache?.rollbackState = (convC, ssmC)

            let (outD, convF, ssmF) = processChunk(
                qkv: qkv[0..., nConfirmed..., 0...],
                a: a[0..., nConfirmed..., 0...],
                b: b[0..., nConfirmed..., 0...],
                convState: convC,
                ssmState: ssmC,
                mask: maskD
            )
            out = concatenated([outC, outD], axis: 1)
            finalConvState = convF
            finalSsmState = ssmF
        } else {
            // Standard single-chunk path (nConfirmed == 0 or S == 1).
            let (o, c, s) = processChunk(
                qkv: qkv, a: a, b: b,
                convState: convState,
                ssmState: ssmState,
                mask: mask
            )
            out = o
            finalConvState = c
            finalSsmState = s
        }

        if let cache {
            cache[0] = finalConvState
            cache[1] = finalSsmState
            // Fail-closed: a forward that did not record a boundary must not
            // leave a previous round's stash behind, or a stale boundary could
            // be restored as though it described this frame.
            cache.fusedPrimaryBoundary = pendingStash
        }

        let normedOut = norm(out, gate: z)
        return outProj(normedOut.reshaped(B, S, -1))
    }
}

// MARK: - Attention

final class Qwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    // Packed Q/K/V concat on N. Underscore so it is not a Module parameter.
    // Built once from already-quantized q/k/v; never attached as a child.
    private var _qkvW: MLXArray?
    private var _qkvS: MLXArray?
    private var _qkvZ: MLXArray?
    private var _qkvGS = 64
    private var _qkvBits = 4
    private var _qkvMode = QuantizationMode.affine
    private var _qOut = 0
    private var _kOut = 0

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    /// One affine-4 GEMM for Q+gate, K, and V. Rows are independent, so
    /// concatenating already-packed weights on N is bit-exact with three
    /// separate qmv_fast launches. Unquantized (MTP bf16) falls back.
    private func qkv(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        if let w = _qkvW, let s = _qkvS, let z = _qkvZ {
            let y = quantizedMM(
                x, w, scales: s, biases: z, transpose: true,
                groupSize: _qkvGS, bits: _qkvBits, mode: _qkvMode)
            let qEnd = _qOut
            let kEnd = _qOut + _kOut
            return (y[.ellipsis, ..<qEnd], y[.ellipsis, qEnd ..< kEnd], y[.ellipsis, kEnd...])
        }
        if let q = qProj as? QuantizedLinear,
           let k = kProj as? QuantizedLinear,
           let v = vProj as? QuantizedLinear,
           q.groupSize == k.groupSize, k.groupSize == v.groupSize,
           q.bits == k.bits, k.bits == v.bits,
           q.mode == k.mode, q.mode == .affine,
           let qz = q.biases, let kz = k.biases, let vz = v.biases
        {
            _qkvW = concatenated([q.weight, k.weight, v.weight], axis: 0).contiguous()
            _qkvS = concatenated([q.scales, k.scales, v.scales], axis: 0).contiguous()
            _qkvZ = concatenated([qz, kz, vz], axis: 0).contiguous()
            _qkvGS = q.groupSize
            _qkvBits = q.bits
            _qkvMode = q.mode
            _qOut = q.shape.0
            _kOut = k.shape.0
            return qkv(x)
        }
        return (qProj(x), kProj(x), vProj(x))
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let (qProjOutput, keysIn, valuesIn) = qkv(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = keysIn
        var values = valuesIn

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(sigmoidMultiply(output, gate))
    }
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined + sharedY
    }
}

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
        nConfirmed: Int = 0,
        stashPrimaryBoundary: Bool = false
    ) -> MLXArray {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py DecoderLayer.__call__
        // Passes nConfirmed through to the linear-attention sublayer.
        let r: MLXArray
        if isLinear {
            r = linearAttn!(
                inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                nConfirmed: nConfirmed,
                stashPrimaryBoundary: stashPrimaryBoundary)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

// MARK: - Text Model

public class Qwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    /// Returns the pre-norm hidden state from the final layer.
    ///
    /// The caller (`Qwen35TextModel`) applies `norm` and the LM head on top.
    /// This split lets `callWithHidden` return both pre-norm hidden (for the MTP head)
    /// and the normalised logits in one forward pass.
    ///
    /// Port of omlx commit 696d90a:
    ///   patches/mlx_lm_mtp/qwen35_model.py `_patch_qwen3_5_text_model`
    ///   (returns hidden_states before self.model.norm so TextModel can apply it)
    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache?]? = nil,
        nConfirmed: Int = 0,
        stashPrimaryBoundary: Bool = false
    ) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask,
                cache: cacheArray?[i], nConfirmed: nConfirmed,
                stashPrimaryBoundary: stashPrimaryBoundary)
        }

        // Return pre-norm hidden states. Norm is applied by Qwen35TextModel.
        return hiddenStates
    }

    /// Rebuild every gated-delta layer's post-row-0 recurrent boundary from the
    /// stash a `stashPrimaryBoundary` forward left behind.
    ///
    /// Preflight every layer before mutating any of them -- returning `false`
    /// leaves the whole cache untouched so the caller can fall back to the
    /// generic snapshot-and-repair path safely. This mirrors the fail-closed
    /// contract of the checkpoint restore it replaces.
    func recomputeFusedPrimaryBoundary(cache: [KVCache?]) -> Bool {
        guard cache.count == layers.count else { return false }
        for (i, layer) in layers.enumerated() where layer.isLinear {
            guard let mamba = cache[i] as? MambaCache,
                  mamba.fusedPrimaryBoundary != nil,
                  layer.linearAttn != nil
            else { return false }
        }
        for (i, layer) in layers.enumerated() where layer.isLinear {
            guard let mamba = cache[i] as? MambaCache,
                  layer.linearAttn?.recomputeFusedPrimaryBoundary(cache: mamba) == true
            else { return false }
        }
        return true
    }
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// MTP head. Non-nil only when `_qwen35MTPEnabled == true` at init time
    /// AND `args.mtpNumHiddenLayers > 0`.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.__init__ (MTPModule attachment)
    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPModule?

    public init(_ args: Qwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }

        // Attach MTP head only when enabled and config declares MTP layers.
        // omlx: `if n_mtp > 0 and is_mtp_active(): self.mtp = q35.MTPModule(args)`
        if args.mtpNumHiddenLayers > 0 && _qwen35MTPEnabled {
            _mtp.wrappedValue = Qwen35MTPModule(args)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Inner model now returns pre-norm hidden; apply norm + lm_head here.
        // omlx: TextModel.__call__ (normed = self.model.norm(hidden); out = lm_head(normed))
        let hidden = model(inputs, cache: cache)
        var out = model.norm(hidden)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py TextModel.sanitize
        //
        // Key differences from stock mlx-lm:
        //  1. Gate norm shift on unsanitized conv1d shape ONLY (not on MTP key presence).
        //     Stock code uses `hasMTPWeights || hasUnsanitizedConv1d`, which double-shifts
        //     already-converted MLX checkpoints that have mtp.* keys.
        //  2. Keep mtp.* keys when the MTP head is attached; strip them otherwise.
        //  3. Extend norm-shift key set with MTP-specific norm names.

        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasUnsanitizedConv1d  // NOT hasMTPWeights

        var weights = weights

        // Keep mtp.* keys if the head is attached; strip them otherwise.
        // omlx: `if not hasattr(self, "mtp"): weights = {k:v if "mtp." not in k}`
        if mtp == nil {
            weights = weights.filter { !$0.key.contains("mtp.") }
        } else if !weights.keys.contains(where: { $0.contains("mtp.") }) {
            // MTP enabled but no mtp.* keys in checkpoint → needs re-conversion.
            // omlx: raises ValueError with "weights are missing the mtp.* tensors"
            print(
                "[WARNING] Qwen35TextModel.sanitize: MTP head is enabled but no mtp.* "
                + "weights found. Load will likely fail or produce garbage. "
                + "Re-convert the checkpoint with a converter that preserves MTP weights.")
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // Extended norm key set includes MTP-specific names.
        // omlx: norm_keys tuple with ".pre_fc_norm_hidden.weight" etc.
        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }
}

// MARK: - Qwen35TextModel + MTPCapable

extension Qwen35TextModel: MTPCapable {
    public var hasMTPHead: Bool { mtp != nil }

    /// Run a backbone forward that also returns pre-norm hidden states.
    ///
    /// Returns `(logits, preNormHidden)` where `preNormHidden` is the raw backbone output
    /// BEFORE `model.norm`. The MTP head applies its own `pre_fc_norm_hidden` normalization,
    /// so it expects un-normalized input. Passing post-norm would double-normalize.
    ///
    /// PR #990: `return out, hidden  # pre-norm hidden for MTP head`
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.__call__ with return_hidden=True
    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        let cacheOpt: [KVCache?] = cache.map { Optional($0) }
        let hidden = model(input.tokens, cache: cacheOpt, nConfirmed: nConfirmed)
        let normed = model.norm(hidden)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(normed)
        } else {
            logits = model.embedTokens.asLinear(normed)
        }
        // Return pre-norm hidden, not post-norm. The MTP module's pre_fc_norm_hidden
        // is the normalization step — it expects the raw backbone output as input.
        return (logits, hidden)
    }

    /// FUSED-VERIFY VARIANT of `callWithHidden` for the K=1 speculative round.
    ///
    /// Identical epilogue -- the final `model.norm` and the vocabulary
    /// projection still run over EVERY row in one call, so the logits this
    /// returns are the same tensor `callWithHidden` would return. (Deliberate:
    /// narrowing the head to row 0 and projecting the bonus row separately
    /// afterwards turns one ~715 MB weight stream into two on the accept path,
    /// which is a regression, not a saving.)
    ///
    /// What differs is the gated-delta geometry: `nConfirmed` is 0, so the
    /// stack runs the verify rows as one fused chunk instead of splitting at
    /// the confirmed prefix, and each layer records a lazy post-row-0 boundary
    /// stash for ``recomputeFusedPrimaryBoundary(cache:)`` to consume if the
    /// draft is rejected. This is the same forward geometry every submission
    /// before the checkpoint split used; it is the reject path, not the rows,
    /// that this reconstructs differently.
    ///
    /// Additive: `callWithHidden` above is untouched and remains the path the
    /// trusted reference and parity tests take.
    public func callWithHiddenStashingPrimaryBoundary(
        input: LMInput.Text, cache: [any KVCache]
    ) -> (MLXArray, MLXArray) {
        let cacheOpt: [KVCache?] = cache.map { Optional($0) }
        let hidden = model(
            input.tokens, cache: cacheOpt, nConfirmed: 0,
            stashPrimaryBoundary: true)
        let normed = model.norm(hidden)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(normed)
        } else {
            logits = model.embedTokens.asLinear(normed)
        }
        return (logits, hidden)
    }

    /// See ``Qwen35TextModelInner/recomputeFusedPrimaryBoundary(cache:)``.
    public func recomputeFusedPrimaryBoundary(cache: [any KVCache]) -> Bool {
        let cacheOpt: [KVCache?] = cache.map { Optional($0) }
        return model.recomputeFusedPrimaryBoundary(cache: cacheOpt)
    }

    /// SEED-PREFILL VARIANT of `callWithHidden`: identical backbone forward, but
    /// the final norm and the vocabulary projection are applied to the LAST ROW
    /// ONLY.
    ///
    /// WHY THIS EXISTS. `Qwen36MTPBlockSession.begin` bulk-forwards the whole
    /// 512-token seed and then keeps exactly two things: the last logit row
    /// (whose argmax is the first pending primary) and the last pre-norm hidden
    /// row (the context the first draft is chained from). `callWithHidden`
    /// nevertheless projects ALL 512 hidden rows through the untied 248,320-way
    /// head, so 511 of the 512 `[1, V]` logit rows are computed and immediately
    /// discarded. MLX will not elide that: slicing the last row of a lazily
    /// built matmul still forces the whole matmul, because a slice does not push
    /// down through a `quantized_matmul` node.
    ///
    /// WHAT IS AND IS NOT SHARED WITH `callWithHidden`. The backbone call is
    /// byte-identical -- same token tensor, same cache array, same `nConfirmed`
    /// -- so all 64 layers still run over every seed position and every
    /// recurrent/attention cache ends in the same state at the same offset. Only
    /// the epilogue narrows. `RMSNorm` reduces over the last axis and the LM head
    /// is a per-row projection, so both are row-independent and the surviving row
    /// is mathematically the same vector `callWithHidden` would have put at index
    /// `S - 1`.
    ///
    /// THE ONE REAL RISK, STATED PLAINLY: `[1, 1, H]` and `[1, S, H]` select
    /// different quantized-matmul geometries (GEMV vs GEMM), which can differ in
    /// the last ulp. That cannot change the mathematics but could in principle
    /// flip an exact near-tie in the seed argmax. `Qwen36MTPReferenceSession` is
    /// deliberately NOT moved onto this seam, so a local exactness run compares
    /// the narrowed candidate decision against the original full-projection
    /// frame instead of letting both sides adopt the same change.
    ///
    /// Additive: `callWithHidden` above is untouched and remains the path every
    /// other caller (verify, serial rounds, repair forwards, the reference
    /// session, the parity tests) uses.
    public func callWithLastTokenHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        let cacheOpt: [KVCache?] = cache.map { Optional($0) }
        let hidden = model(input.tokens, cache: cacheOpt, nConfirmed: nConfirmed)
        let last = hidden.dim(1) - 1
        let tail = hidden[0..., last ..< (last + 1), 0...]
        let normed = model.norm(tail)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(normed)
        } else {
            logits = model.embedTokens.asLinear(normed)
        }
        // Pre-norm hidden, exactly as `callWithHidden` returns -- one row.
        return (logits, tail)
    }

    /// Apply the backbone's final `model.norm` to a hidden state.
    ///
    /// `callWithHidden` returns the PRE-norm hidden by design. MTPLX -- the exactness
    /// reference this track's accept/verify loop was validated against -- defaults to
    /// `base_hidden_variant == "post_norm"` (mtplx/mtp_patch.py:50, and
    /// `hidden = pre_norm if variant == "pre_norm" else post_norm` in
    /// `_MTPLXTextModel.__call__`), i.e. the hidden fed to the MTP head is the backbone
    /// output AFTER `model.norm`, even though the head then applies its own
    /// `pre_fc_norm_hidden` on top. This accessor lets a caller produce that variant
    /// without changing `callWithHidden`'s existing pre-norm contract.
    ///
    /// The variant is NOT a correctness knob and a wrong choice fails SILENTLY: a
    /// pre-norm draft chain still verifies exact (the target decides every emitted
    /// token), it just stops predicting -- acceptance collapses toward zero and the
    /// speculation buys nothing. Any validation of this path must therefore read the
    /// ACCEPTANCE RATE alongside the exactness verdict.
    ///
    /// `Qwen35TextModelInner.norm` is not visible outside this module, which is the
    /// only reason this accessor exists.
    public func applyFinalNorm(_ x: MLXArray) -> MLXArray {
        model.norm(x)
    }

    /// Run the MTP head forward, returning `(logits, headHidden)`.
    ///
    /// `headHidden` is the MTP head's own post-`mtp.norm` output, which is what MTPLX
    /// chains into the next draft level when `mtp_hidden_variant == "post_norm"` (its
    /// default): `h = hidden_level[:, -1:, :]` in `_make_device_draft_core.chain_fn`
    /// (mtplx/generation.py) with `post_norm = self.mtp.norm(x)` in `_mtp_core`
    /// (mtplx/mtp_patch.py). Required for multi-step (depth > 1) drafting: re-feeding
    /// the TRUNK hidden to every sub-step would draft every level from the same state.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.mtp_forward(return_hidden=True)
    public func mtpForwardWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> (MLXArray, MLXArray) {
        guard let mtp else {
            fatalError("mtpForwardWithHidden called but MTP head is not attached. "
                + "Set _qwen35MTPEnabled = true before loading the model.")
        }
        let mtpOut = mtp(
            hidden: hidden,
            nextTokenIds: nextTokenIds,
            embedTokens: model.embedTokens,
            cache: cache)
        let logits: MLXArray
        if configuration.tieWordEmbeddings {
            logits = model.embedTokens.asLinear(mtpOut)
        } else {
            logits = lmHead!(mtpOut)
        }
        return (logits, mtpOut)
    }

    /// Vocabulary-narrowed MTP forward: same fuse + layer + cache write as
    /// `mtpForwardWithHidden`, but `lm_head` sees only the last row.
    public func mtpForwardLastTokenWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> (MLXArray, MLXArray) {
        guard let mtp else {
            fatalError("mtpForwardLastTokenWithHidden called but MTP head is not attached. "
                + "Set _qwen35MTPEnabled = true before loading the model.")
        }
        let mtpOut = mtp(
            hidden: hidden,
            nextTokenIds: nextTokenIds,
            embedTokens: model.embedTokens,
            cache: cache)
        let last = mtpOut.dim(1) - 1
        let tail = mtpOut[0..., last ..< (last + 1), 0...]
        let logits: MLXArray
        if configuration.tieWordEmbeddings {
            logits = model.embedTokens.asLinear(tail)
        } else {
            logits = lmHead!(tail)
        }
        return (logits, tail)
    }

    /// History-only MTP step: write the fused rows into `cache`, no logits.
    public func mtpUpdateCache(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) {
        guard let mtp else {
            fatalError("mtpUpdateCache called but MTP head is not attached. "
                + "Set _qwen35MTPEnabled = true before loading the model.")
        }
        _ = mtp(
            hidden: hidden,
            nextTokenIds: nextTokenIds,
            embedTokens: model.embedTokens,
            cache: cache)
    }

    /// Seed prefill: last-row logits plus the full pre-norm hidden sequence.
    public func callWithLastTokenLogitsAndFullHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        let cacheOpt: [KVCache?] = cache.map { Optional($0) }
        let hidden = model(input.tokens, cache: cacheOpt, nConfirmed: nConfirmed)
        let last = hidden.dim(1) - 1
        let tail = hidden[0..., last ..< (last + 1), 0...]
        let normed = model.norm(tail)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(normed)
        } else {
            logits = model.embedTokens.asLinear(normed)
        }
        return (logits, hidden)
    }

    /// Run the MTP head forward.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.mtp_forward
    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        mtpForwardWithHidden(hidden: hidden, nextTokenIds: nextTokenIds, cache: cache).0
    }

    /// Allocate a fresh KV cache for the MTP head layers.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.make_mtp_cache
    public func makeMTPCache() -> [any KVCache] {
        guard let mtp else { return [] }
        return mtp.layers.map { _ in KVCacheSimple() as any KVCache }
    }
}

extension Qwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level Model

public class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    public init(_ args: Qwen35Configuration) {
        let textModel = Qwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - Qwen35Model + MTPCapable

/// VLM-outer-wrapper pass-through for MTPCapable.
/// Forwards all MTP calls to the inner `languageModel` (a Qwen35TextModel).
/// omlx: patches/mlx_lm_mtp/qwen35_model.py `_patch_outer_model`
extension Qwen35Model: MTPCapable {
    public var hasMTPHead: Bool { languageModel.hasMTPHead }

    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        languageModel.callWithHidden(input: input, cache: cache, nConfirmed: nConfirmed)
    }

    /// See `Qwen35TextModel.callWithLastTokenHidden`. Pure pass-through, exactly
    /// like `callWithHidden` above -- no wrapper-level arithmetic.
    public func callWithLastTokenHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        languageModel.callWithLastTokenHidden(
            input: input, cache: cache, nConfirmed: nConfirmed)
    }

    /// See `Qwen35TextModel.callWithHiddenStashingPrimaryBoundary`. Pure
    /// pass-through -- no wrapper-level arithmetic.
    public func callWithHiddenStashingPrimaryBoundary(
        input: LMInput.Text, cache: [any KVCache]
    ) -> (MLXArray, MLXArray) {
        languageModel.callWithHiddenStashingPrimaryBoundary(
            input: input, cache: cache)
    }

    /// See `Qwen35TextModel.recomputeFusedPrimaryBoundary`.
    public func recomputeFusedPrimaryBoundary(cache: [any KVCache]) -> Bool {
        languageModel.recomputeFusedPrimaryBoundary(cache: cache)
    }

    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        languageModel.mtpForward(hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    /// See `Qwen35TextModel.mtpForwardWithHidden`.
    public func mtpForwardWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> (MLXArray, MLXArray) {
        languageModel.mtpForwardWithHidden(
            hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    /// See `Qwen35TextModel.mtpForwardLastTokenWithHidden`.
    public func mtpForwardLastTokenWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> (MLXArray, MLXArray) {
        languageModel.mtpForwardLastTokenWithHidden(
            hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    /// See `Qwen35TextModel.mtpUpdateCache`.
    public func mtpUpdateCache(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) {
        languageModel.mtpUpdateCache(
            hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    /// See `Qwen35TextModel.callWithLastTokenLogitsAndFullHidden`.
    public func callWithLastTokenLogitsAndFullHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        languageModel.callWithLastTokenLogitsAndFullHidden(
            input: input, cache: cache, nConfirmed: nConfirmed)
    }

    /// See `Qwen35TextModel.applyFinalNorm`.
    public func applyFinalNorm(_ x: MLXArray) -> MLXArray {
        languageModel.applyFinalNorm(x)
    }

    public func makeMTPCache() -> [any KVCache] {
        languageModel.makeMTPCache()
    }
}
