import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Correctness-first Laguna runtime (Poolside Laguna XS 2.1, 256-expert MoE).
//
// This module tree closely follows the vendored reference implementation at
// `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Laguna.swift` (`LagunaModel` /
// `LagunaModelInner`), which is the behavior oracle for this port. It is a
// reimplementation rather than a wrapper for two load-bearing reasons:
//
// 1. The Poolside checkpoint stores the MoE router as a raw BF16
//    `mlp.gate.weight` matrix next to the F32
//    `mlp.gate.e_score_correction_bias`, while only expert projections are
//    NVFP4. The runtime mirrors those parameter paths exactly.
// 2. The vendored `LagunaModelInner`/`LagunaDecoderLayer` types are
//    fileprivate and `LagunaConfiguration`'s stored properties are internal
//    to MLXLLM, so the runtime layers (cache geometry, future fast-engine
//    and exact-verification waves) could not reach the internals through a
//    plain wrapper.
//
// All math is expressed with standard MLX ops and the vendored shared
// primitives (`attentionWithCacheUpdate`, `initializeRope`,
// `applyRotaryPosition`, `SwitchGLU`, `weightedExpertSum`, `RMSNorm`,
// `createAttentionMask`). No custom Metal kernels in this increment; the
// fused fast-engine and exact-pair/exact-four style optimizations are a
// later layer on top of this reference target.

func lagunaLastTokenRange(sequenceLength: Int) -> Range<Int>? {
    sequenceLength > 1 ? (sequenceLength - 1)..<sequenceLength : nil
}

func lagunaLastTokenHidden(_ hidden: MLXArray) -> MLXArray {
    guard let range = lagunaLastTokenRange(sequenceLength: hidden.dim(1)) else {
        return hidden
    }
    return hidden[0..., range, 0...]
}

/// Builds the `initializeRope` scaling dictionary for a per-type Laguna RoPE
/// spec. For `default` RoPE only the type is consulted; for YaRN the factory
/// reads factor / original context / betas. The XS config also serializes
/// `attention_factor: 1.0`, but both vendored MLX Laguna implementations
/// intentionally ignore that Hugging Face field. Do not forward it here:
/// leaving MLX's mscale/mscale_all_dim defaults at 1.0/0.0 yields the upstream
/// attention scaling of `0.1 * ln(32) + 1` (~1.34657).
func lagunaRopeScalingConfig(_ spec: LagunaRopeSpec) -> [String: StringOrNumber] {
    var scalingConfig: [String: StringOrNumber] = ["rope_type": .string(spec.type)]
    if spec.type == "yarn" {
        scalingConfig["factor"] = .float(Float(spec.factor))
        scalingConfig["original_max_position_embeddings"] = .int(
            spec.originalMaxPositionEmbeddings)
        scalingConfig["beta_fast"] = .float(Float(spec.betaFast))
        scalingConfig["beta_slow"] = .float(Float(spec.betaSlow))
    }
    return scalingConfig
}

// MARK: - Runtime fusion feature flags

// Each fusion below concatenates the OUTPUT ROWS of same-dtype projections
// that consume the same input. Per-row gemv/qmv/gather-qmv arithmetic is
// independent of which rows share a dispatch (every output row keeps its own
// K-loop and scale application in the original order), so the fused dispatch
// is bit-exact against the separate dispatches it replaces. The per-head
// g_proj (N=64) uses a different split-K gemv variant and is never fused.

/// `DARKBLOOM_FUSED_QKV` (default OFF; set "1" to enable): after checkpoint
/// load, retain one row-concatenated `[Wq; Wk; Wv]` BF16 weight per attention
/// layer and serve Q/K/V from a single projection dispatch. Ablation on the
/// paired local benchmark showed a mild prefill cost with no decode gain, so
/// this ships opt-in.
let lagunaFusedQKVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_QKV"] == "1"

/// `DARKBLOOM_FUSED_SHARED_GATE_UP` (default OFF; set "1" to enable): after
/// checkpoint load, retain one row-concatenated NVFP4 `[gate; up]` bank per
/// shared expert and serve both projections from a single quantized matmul.
/// Unproven in ablation, so this ships opt-in.
let lagunaFusedSharedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SHARED_GATE_UP"] == "1"

/// `DARKBLOOM_FUSED_ROUTED_GATE_UP` (default on; set "0" to disable): after
/// checkpoint load, retain one row-concatenated NVFP4 `[gate; up]` bank per
/// sparse layer's routed experts and serve single-token decode's gate/up from
/// one gather-QMM dispatch. DECODE-ONLY: the module tree, checkpoint keys,
/// and every multi-token (prefill) forward stay fully stock -- ablation
/// showed the fused bank helps decode (~+1.9%) but badly hurts the M=512
/// sorted gather-GEMM prefill path, so prefill always dispatches the stock
/// separate banks.
let lagunaFusedRoutedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_GATE_UP"] != "0"

// MARK: - Attention

private let lagunaCompiledSoftplusGate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { gate in
        softplus(gate.asType(.float32)).asType(gate.dtype)
    }
    return MLXHardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Laguna attention: GQA with per-head QK-norm, per-layer-type RoPE (YaRN on
/// full-attention layers over the first half of the head, plain RoPE on
/// sliding layers over the whole head), and per-head softplus output gating.
/// Mirrors the vendored `LagunaAttention` forward exactly.
final class LagunaRuntimeAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let gatingEnabled: Bool
    let gatePerHead: Bool
    let isSliding: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    /// Retained fused `[Wq; Wk; Wv]` weight (output rows concatenated, query
    /// rows first), built once after checkpoint load when
    /// `DARKBLOOM_FUSED_QKV` is enabled. Plain stored property with a leading
    /// underscore so Module reflection never treats this derived layout as a
    /// checkpoint parameter; the q/k/v `Linear` modules keep the original
    /// arrays for parameter integrity.
    var _fusedQKVWeight: MLXArray?

    /// Builds and retains the fused QKV weight from the loaded q/k/v
    /// projection weights. Called once after weights are installed and
    /// evaluated (before warmup); returns the new array so the caller can
    /// batch a single eval. Fuses only the exact stock configuration: three
    /// plain bias-free `Linear` projections of one dtype over the same input
    /// width, so the fused matmul is `matmul(x, w.T)` with every original
    /// output row unchanged.
    func prepareFusedQKVWeight() -> MLXArray? {
        guard _fusedQKVWeight == nil,
            type(of: wq) == Linear.self,
            type(of: wk) == Linear.self,
            type(of: wv) == Linear.self,
            wq.bias == nil, wk.bias == nil, wv.bias == nil,
            wq.weight.ndim == 2, wk.weight.ndim == 2, wv.weight.ndim == 2,
            wq.weight.dtype == wk.weight.dtype,
            wk.weight.dtype == wv.weight.dtype,
            wq.weight.dim(1) == wk.weight.dim(1),
            wk.weight.dim(1) == wv.weight.dim(1),
            wq.weight.dim(0) == nHeads * headDim,
            wk.weight.dim(0) == nKVHeads * headDim,
            wv.weight.dim(0) == nKVHeads * headDim
        else {
            return nil
        }
        let fused = concatenated([wq.weight, wk.weight, wv.weight], axis: 0)
        _fusedQKVWeight = fused
        return fused
    }

    init(_ config: LagunaConfig, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.heads(forLayer: layerIdx)
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)
        let layerGating = config.gatingMode(forLayer: layerIdx)
        self.gatingEnabled = layerGating.enabled
        self.gatePerHead = layerGating.isPerHead

        let layerType = config.layerType(forLayer: layerIdx)
        self.isSliding = layerType == .sliding

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: config.qkvBias)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.qkvBias)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.qkvBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: config.attentionBias)

        if gatingEnabled {
            let gateDim = gatePerHead ? nHeads : nHeads * headDim
            self._gProj.wrappedValue = Linear(dim, gateDim, bias: false)
        }

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

        let ropeSpec = config.rope(for: layerType)
        let ropeDims = Int(Float(headDim) * Float(ropeSpec.partialRotaryFactor))
        self.rope = initializeRope(
            dims: ropeDims,
            base: Float(ropeSpec.theta),
            traditional: false,
            scalingConfig: lagunaRopeScalingConfig(ropeSpec),
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray
        if let fusedQKVWeight = _fusedQKVWeight {
            // One dispatch over the row-concatenated [Wq; Wk; Wv] weight,
            // identical math to the three bias-free `Linear` calls
            // (`matmul(x, w.T)`). Each output row's K-loop is independent of
            // which rows share the dispatch, so every Q/K/V element is
            // bit-exact; the slices are views and the reshapes below may
            // copy, which does not change values.
            let qkv = matmul(x, fusedQKVWeight.T)
            let queryDim = nHeads * headDim
            let kvDim = nKVHeads * headDim
            queries = qkv[.ellipsis, 0 ..< queryDim]
            keys = qkv[.ellipsis, queryDim ..< (queryDim + kvDim)]
            values = qkv[.ellipsis, (queryDim + kvDim) ..< (queryDim + 2 * kvDim)]
        } else {
            queries = wq(x)
            keys = wk(x)
            values = wv(x)
        }

        queries = qNorm(queries.reshaped(B, L, nHeads, headDim)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        if gatingEnabled, let gProj {
            // Per-head softplus gate computed in float32, then broadcast
            // across the head dimension (or applied elementwise for a
            // per-element gate).
            let projectedGate = gProj(x)
            let gate = gatePerHead && projectedGate.dtype == output.dtype
                ? lagunaCompiledSoftplusGate(projectedGate)
                : softplus(projectedGate.asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output =
                    (output.reshaped(B, L, nHeads, headDim) * gate[.ellipsis, .newAxis])
                    .reshaped(B, L, -1)
            } else {
                output = output * gate
            }
        }

        return wo(output)
    }
}

// MARK: - Dense MLP (also used as the shared expert)

final class LagunaRuntimeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    /// Retained fused NVFP4 `[gate; up]` layout (gate output rows first),
    /// built once after checkpoint load for the shared expert when
    /// `DARKBLOOM_FUSED_SHARED_GATE_UP` is enabled. Plain stored properties
    /// with a leading underscore so Module reflection never treats the
    /// derived layout as checkpoint parameters; the quantized gate/up
    /// modules keep the original arrays for parameter integrity. Never set
    /// on the dense (BF16) layer-0 MLP.
    var _fusedGateUpWeight: MLXArray?
    var _fusedGateUpScales: MLXArray?
    var _fusedGateUpSplit: Int = 0

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    /// Builds and retains the fused gate/up NVFP4 bank from the loaded
    /// shared-expert projections. Called once after weights are installed
    /// and evaluated (before warmup); returns the new arrays so the caller
    /// can batch a single eval. Fuses only the exact stock shared-expert
    /// configuration: two bias-free NVFP4 group-16 4-bit `QuantizedLinear`
    /// projections with identical packed shapes and no affine biases.
    func prepareFusedSharedGateUp() -> [MLXArray] {
        guard _fusedGateUpWeight == nil, _fusedGateUpScales == nil,
            let gate = gateProj as? QuantizedLinear,
            let up = upProj as? QuantizedLinear,
            type(of: gate) == QuantizedLinear.self,
            type(of: up) == QuantizedLinear.self,
            gate.mode == .nvfp4, up.mode == .nvfp4,
            gate.groupSize == 16, up.groupSize == 16,
            gate.bits == 4, up.bits == 4,
            gate.bias == nil, up.bias == nil,
            gate.biases == nil, up.biases == nil,
            gate.weight.ndim == 2, up.weight.ndim == 2,
            gate.weight.dtype == .uint32, up.weight.dtype == .uint32,
            gate.scales.ndim == 2, up.scales.ndim == 2,
            gate.scales.dtype == .uint8, up.scales.dtype == .uint8,
            gate.weight.shape == up.weight.shape,
            gate.scales.shape == up.scales.shape,
            gate.scales.dim(0) == gate.weight.dim(0),
            gate.weight.dim(1) * 8 == gate.scales.dim(1) * 16
        else {
            return []
        }
        let fusedWeight = concatenated([gate.weight, up.weight], axis: 0)
        let fusedScales = concatenated([gate.scales, up.scales], axis: 0)
        _fusedGateUpWeight = fusedWeight
        _fusedGateUpScales = fusedScales
        _fusedGateUpSplit = gate.weight.dim(0)
        return [fusedWeight, fusedScales]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let fusedWeight = _fusedGateUpWeight, let fusedScales = _fusedGateUpScales {
            // One NVFP4 dispatch over the row-concatenated [gate; up] bank,
            // mirroring `QuantizedLinear.callAsFunction` exactly (transpose,
            // group 16, 4-bit, .nvfp4, no affine biases, no bias add; the
            // guards in `prepareFusedSharedGateUp` pin those literals). Each
            // quantized output row is computed independently, so the split
            // halves are bit-exact vs. the separate gate/up dispatches.
            let gateUp = MLX.quantizedMM(
                x,
                fusedWeight,
                scales: fusedScales,
                biases: nil,
                transpose: true,
                groupSize: 16,
                bits: 4,
                mode: .nvfp4
            )
            let gate = gateUp[.ellipsis, 0 ..< _fusedGateUpSplit]
            let up = gateUp[.ellipsis, _fusedGateUpSplit...]
            return downProj(compiledSiluProduct(gate, up))
        }
        return downProj(compiledSiluProduct(gateProj(x), upProj(x)))
    }
}

// MARK: - MoE

/// Compiled fused router tail for the softcap-free Laguna XS contract:
/// `[logits_bf16, bias_f32] -> [scores, -(scores + bias)]`, folding the
/// float32 cast, sigmoid, correction-bias add, and the argPartition-operand
/// negate -- four elementwise launches per router call, 39 router calls per
/// token -- into one compiled shapeless kernel. The body is the identical
/// expression tree the eager code builds (same `sigmoid`/`asType`/`+`/`-`
/// functors applied in the same order, no reassociation; compile fuses
/// elementwise ops without changing per-element arithmetic), so both
/// outputs are bit-exact against the uncompiled tail. Gated like the
/// sibling compiled fusions: when compiled decode is unsupported the
/// identical uncompiled body runs instead.
private let lagunaCompiledRouterTail: @Sendable ([MLXArray]) -> [MLXArray] = {
    let body: @Sendable ([MLXArray]) -> [MLXArray] = { inputs in
        let scores = sigmoid(inputs[0].asType(.float32))
        let scoresForChoice = scores + inputs[1].asType(scores.dtype)
        return [scores, -scoresForChoice]
    }
    return MLXHardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Compiled top-k mixture-weight renormalization
/// (`weights / weights.sum(axis: -1, keepDims: true)`), the router's
/// `normTopkProb` tail. Identical expression tree to the eager code: the
/// sum still dispatches the stock reduce kernel (compile does not fuse
/// reductions) and the divide is elementwise, so the result is bit-exact.
private let lagunaCompiledTopKNormalize: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { weights in
        weights / weights.sum(axis: -1, keepDims: true)
    }
    return MLXHardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Sigmoid top-k router. The routing math mirrors the vendored
/// `LagunaMoEGate` exactly (sigmoid scores, correction bias added only for
/// expert CHOICE, mixture weights taken from the pre-bias scores, optional
/// top-k renormalization).
final class LagunaRuntimeMoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routerLogitSoftcapping: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: LagunaConfig) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.routerLogitSoftcapping = Float(config.moeRouterLogitSoftcapping)
        self._weight.wrappedValue = zeros([config.numExperts, config.hiddenSize])
        self._eScoreCorrectionBias.wrappedValue = zeros([config.numExperts])
    }

    func callAsFunction(
        _ x: MLXArray,
        normalizeWeights: Bool = true
    ) -> (MLXArray, MLXArray) {
        if routerLogitSoftcapping <= 0 {
            // Pinned Laguna XS contract (config load rejects a nonzero
            // router softcap): the raw BF16 matmul feeds the compiled fused
            // tail, which returns the pre-bias sigmoid scores and the exact
            // `-(scores + bias)` operand the eager code hands argPartition.
            // argPartition/takeAlong themselves are untouched.
            let tail = lagunaCompiledRouterTail([x.matmul(weight.T), eScoreCorrectionBias])
            let scores = tail[0]
            let inds = argPartition(tail[1], kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            var weights = takeAlong(scores, inds, axis: -1)
            if normTopkProb && normalizeWeights {
                weights = lagunaCompiledTopKNormalize(weights)
            }
            return (inds, weights)
        }

        // Softcapped router variant (not the Laguna XS contract): the
        // original eager tail, unchanged.
        var logits = x.matmul(weight.T).asType(.float32)
        logits = tanh(logits / routerLogitSoftcapping) * routerLogitSoftcapping

        let scores = sigmoid(logits)
        let scoresForChoice = scores + eScoreCorrectionBias.asType(scores.dtype)

        let inds = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if normTopkProb && normalizeWeights {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (inds, weights)
    }
}

/// Compile the routed expert weighting, Laguna's pinned routed scale, and the
/// shared-expert residual as one graph. This preserves the eager expression
/// order (weighted reduction -> scale -> add) while avoiding a separately
/// materialized routed subtotal between the final sparse-MoE operations.
private let lagunaCompiledExpertCombine: @Sendable (
    MLXArray, MLXArray, MLXArray
) -> MLXArray = compile(shapeless: true) { outputs, weights, shared in
    let routed =
        (outputs * MLX.expandedDimensions(weights, axis: -1))
        .sum(axis: -2)
    return routed * Float(LagunaConstants.moeRoutedScalingFactor) + shared
}

/// Generic sparse-layer variant that also retains the decoder residual add
/// in the compiled combine. Grouping matches the eager layer exactly:
/// residual + ((weighted expert reduction * 2.5) + shared expert).
private let lagunaCompiledExpertCombineWithResidual: @Sendable (
    MLXArray, MLXArray, MLXArray, MLXArray
) -> MLXArray = compile(shapeless: true) { outputs, weights, shared, residual in
    let routed =
        (outputs * MLX.expandedDimensions(weights, axis: -1))
        .sum(axis: -2)
    let moe = routed * Float(LagunaConstants.moeRoutedScalingFactor) + shared
    return residual + moe
}

/// Pinned Laguna XS tail with top-k normalization kept in the same compiled
/// graph as the promoted expert combine. Arithmetic order remains:
/// FP32 normalize -> output-dtype cast -> weighted reduction -> * 2.5 -> add.
private let lagunaCompiledNormalizedExpertCombine: @Sendable (
    MLXArray, MLXArray, MLXArray
) -> MLXArray = compile(shapeless: true) { outputs, weights, shared in
    let normalized = weights / weights.sum(axis: -1, keepDims: true)
    let typedWeights = normalized.asType(outputs.dtype)
    let routed =
        (outputs * MLX.expandedDimensions(typedWeights, axis: -1))
        .sum(axis: -2)
    return routed * Float(LagunaConstants.moeRoutedScalingFactor) + shared
}

/// Pinned normalized tail plus the outer sparse decoder residual. Arithmetic
/// stays residual + ((normalize -> cast -> reduction -> * 2.5) + shared).
private let lagunaCompiledNormalizedExpertCombineWithResidual: @Sendable (
    MLXArray, MLXArray, MLXArray, MLXArray
) -> MLXArray = compile(shapeless: true) { outputs, weights, shared, residual in
    let normalized = weights / weights.sum(axis: -1, keepDims: true)
    let typedWeights = normalized.asType(outputs.dtype)
    let routed =
        (outputs * MLX.expandedDimensions(typedWeights, axis: -1))
        .sum(axis: -2)
    let moe = routed * Float(LagunaConstants.moeRoutedScalingFactor) + shared
    return residual + moe
}

final class LagunaRuntimeSparseMoEBlock: Module, UnaryLayer {
    let usesDeferredTopKNormalization: Bool

    @ModuleInfo(key: "gate") var gate: LagunaRuntimeMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaRuntimeMLP

    /// Retained fused NVFP4 `[gate; up]` routed-expert banks (per-expert
    /// output rows concatenated, gate rows first), built once after
    /// checkpoint load when `DARKBLOOM_FUSED_ROUTED_GATE_UP` is enabled, plus
    /// a reference to the stock `switch_mlp.down_proj` module for the fused
    /// decode path. Plain stored properties with a leading underscore so
    /// Module reflection never treats the derived layout as checkpoint
    /// parameters or a second child module; `switchMLP` keeps the original
    /// separate banks (they still serve every multi-token forward and
    /// parameter integrity).
    var _fusedRoutedGateUpWeight: MLXArray?
    var _fusedRoutedGateUpScales: MLXArray?
    var _fusedRoutedGateUpSplit: Int = 0
    var _routedDownProj: SwitchLinear?

    /// Builds and retains the fused routed gate/up NVFP4 banks from the
    /// loaded stock `SwitchGLU` submodules (reached through the public
    /// `children()`/`parameters()` Module APIs). Called once after weights
    /// are installed and evaluated (before warmup); returns the new arrays
    /// so the caller can batch a single eval. Fuses only the exact stock
    /// configuration: two bias-free NVFP4 group-16 4-bit
    /// `QuantizedSwitchLinear` banks with identical packed shapes.
    func prepareFusedRoutedGateUp() -> [MLXArray] {
        guard _fusedRoutedGateUpWeight == nil, _fusedRoutedGateUpScales == nil else {
            return []
        }
        let children = Dictionary(uniqueKeysWithValues: switchMLP.children().flattened())
        guard let gateModule = children["gate_proj"] as? QuantizedSwitchLinear,
            let upModule = children["up_proj"] as? QuantizedSwitchLinear,
            let downModule = children["down_proj"] as? SwitchLinear,
            type(of: gateModule) == QuantizedSwitchLinear.self,
            type(of: upModule) == QuantizedSwitchLinear.self,
            gateModule.mode == .nvfp4, upModule.mode == .nvfp4,
            gateModule.groupSize == 16, upModule.groupSize == 16,
            gateModule.bits == 4, upModule.bits == 4
        else {
            return []
        }
        let gateParams = Dictionary(uniqueKeysWithValues: gateModule.parameters().flattened())
        let upParams = Dictionary(uniqueKeysWithValues: upModule.parameters().flattened())
        guard let gateWeight = gateParams["weight"], let gateScales = gateParams["scales"],
            let upWeight = upParams["weight"], let upScales = upParams["scales"],
            gateParams["bias"] == nil, gateParams["biases"] == nil,
            upParams["bias"] == nil, upParams["biases"] == nil,
            gateWeight.ndim == 3, upWeight.ndim == 3,
            gateScales.ndim == 3, upScales.ndim == 3,
            gateWeight.dtype == .uint32, upWeight.dtype == .uint32,
            gateScales.dtype == .uint8, upScales.dtype == .uint8,
            gateWeight.shape == upWeight.shape,
            gateScales.shape == upScales.shape,
            gateScales.dim(0) == gateWeight.dim(0),
            gateScales.dim(1) == gateWeight.dim(1),
            gateWeight.dim(2) * 8 == gateScales.dim(2) * 16
        else {
            return []
        }
        let fusedWeight = concatenated([gateWeight, upWeight], axis: 1)
        let fusedScales = concatenated([gateScales, upScales], axis: 1)
        _fusedRoutedGateUpWeight = fusedWeight
        _fusedRoutedGateUpScales = fusedScales
        _fusedRoutedGateUpSplit = gateWeight.dim(1)
        _routedDownProj = downModule
        return [fusedWeight, fusedScales]
    }

    init(_ config: LagunaConfig) {
        self.usesDeferredTopKNormalization =
            config.normTopkProb
            && config.moeRouterLogitSoftcapping <= 0
            && config.moeRoutedScalingFactor == LagunaConstants.moeRoutedScalingFactor
        self._gate.wrappedValue = LagunaRuntimeMoEGate(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        self._sharedExpert.wrappedValue = LagunaRuntimeMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, residual: nil)
    }

    func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray {
        callAsFunction(x, residual: Optional(residual))
    }

    private func callAsFunction(_ x: MLXArray, residual: MLXArray?) -> MLXArray {
        let (inds, weights) = gate(
            x,
            normalizeWeights: !usesDeferredTopKNormalization
        )
        var y: MLXArray
        if let fusedWeight = _fusedRoutedGateUpWeight,
            let fusedScales = _fusedRoutedGateUpScales,
            let downProj = _routedDownProj,
            x.dim(1) == 1, inds.size < 64
        {
            // DECODE-ONLY fused gate/up: replicate exactly SwitchGLU's
            // unsorted small-batch path (`indices.size < 64`, so no
            // gatherSort/scatterUnsort) with one gather-QMM over the
            // row-concatenated [gate; up] bank instead of two. The gather
            // call mirrors `QuantizedSwitchLinear.callAsFunction` (biases
            // nil, rhsIndices, transpose, group 16, 4-bit, .nvfp4,
            // sortedIndices false; the prepare guards pin those literals).
            // Each gathered output row is computed independently, so the
            // split halves (gate rows first) are bit-exact vs. the separate
            // banks; down_proj is the stock module invoked exactly as
            // SwitchGLU does. Multi-token forwards (prefill) below keep the
            // fully stock sorted gather-GEMM path and never see the fused
            // bank.
            let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
            let gateUp = MLX.gatherQuantizedMM(
                expanded,
                fusedWeight,
                scales: fusedScales,
                biases: nil,
                rhsIndices: inds,
                transpose: true,
                groupSize: 16,
                bits: 4,
                mode: .nvfp4,
                sortedIndices: false
            )
            let xGate = gateUp[.ellipsis, 0 ..< _fusedRoutedGateUpSplit]
            let xUp = gateUp[.ellipsis, _fusedRoutedGateUpSplit...]
            let activated = compiledSiluProduct(xGate, xUp)
            y = MLX.squeezed(downProj(activated, inds, sortedIndices: false), axis: -2)
        } else {
            y = switchMLP(x, inds)
        }
        let shared = sharedExpert(x)
        if usesDeferredTopKNormalization {
            if let residual {
                return lagunaCompiledNormalizedExpertCombineWithResidual(
                    y,
                    weights,
                    shared,
                    residual
                )
            }
            return lagunaCompiledNormalizedExpertCombine(
                y,
                weights,
                shared
            )
        }
        let typedWeights = weights.asType(y.dtype)
        if let residual {
            return lagunaCompiledExpertCombineWithResidual(
                y,
                typedWeights,
                shared,
                residual
            )
        }
        return lagunaCompiledExpertCombine(
            y,
            typedWeights,
            shared
        )
    }
}

// MARK: - Decoder Layer

final class LagunaRuntimeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: LagunaRuntimeAttention
    let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let attentionType: LagunaLayerType

    init(_ config: LagunaConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = LagunaRuntimeAttention(config, layerIdx: layerIdx)

        if config.isSparse(layer: layerIdx) {
            self.mlp = LagunaRuntimeSparseMoEBlock(config)
        } else {
            self.mlp = LagunaRuntimeMLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))

        self.attentionType = config.layerType(forLayer: layerIdx)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let mlpInput = postAttentionLayerNorm(h)
        if let sparseMLP = mlp as? LagunaRuntimeSparseMoEBlock {
            return sparseMLP(mlpInput, residual: h)
        }
        return h + mlp(mlpInput)
    }
}

// MARK: - Model

/// The Laguna text tower: unscaled embedding, 40 decoder layers, final
/// RMSNorm.
/// Returns post-norm hidden states for every input position.
final class LagunaRuntimeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LagunaRuntimeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerTypes: [LagunaLayerType]
    let slidingWindow: Int
    let fullAttentionIdx: Int
    let slidingAttentionIdx: Int

    init(_ config: LagunaConfig) {
        precondition(config.vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            LagunaRuntimeDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))

        self.layerTypes = config.layerTypes
        self.slidingWindow = config.slidingWindow
        self.fullAttentionIdx = config.layerTypes.firstIndex(of: .full) ?? 0
        self.slidingAttentionIdx = config.layerTypes.firstIndex(of: .sliding) ?? 0
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // One mask per attention family, derived from a representative
        // layer's cache offset: all full-attention caches advance in
        // lockstep, as do all sliding caches (vendored `LagunaModelInner`
        // convention).
        let fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIdx])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[slidingAttentionIdx], windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = layerTypes[i] == .full ? fullMask : slidingMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

/// Scored Laguna runtime model: last-token vocabulary head over the
/// reimplemented Laguna text tower.
///
/// `callAsFunction(_:cache:)` serves both prompt prefill
/// (`[1, L]`) and single-token decode steps (`[1, 1]`) and returns
/// `[1, 1, vocab]` last-token logits; `newCache(parameters:)` creates the
/// per-layer cache stack (unbounded `StandardKVCache` for full-attention
/// layers, `RotatingKVCache(512)` for sliding layers). Laguna applies NO
/// final logit softcap and NO embedding scaling.
public final class LagunaRuntimeModel: Module, LanguageModel {
    @ModuleInfo(key: "model") var model: LagunaRuntimeModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let configuration: LagunaConfig

    public init(_ config: LagunaConfig) {
        self.configuration = config
        self._model.wrappedValue = LagunaRuntimeModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()

        // Match the vendored Poolside Laguna model exactly: only the routed
        // experts and shared expert are NVFP4. Quantizing one sparse decoder
        // layer at a time avoids asking Module.update to descend through the
        // dense layer 0, which has no quantized child.
        for layer in model.layers where layer.mlp is LagunaRuntimeSparseMoEBlock {
            quantize(model: layer) { path, _ in
                if path.contains("switch_mlp") || path.contains("shared_expert") {
                    return (
                        groupSize: config.quantization.groupSize,
                        bits: config.quantization.bits,
                        mode: .nvfp4
                    )
                }
                return nil
            }
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let fullHidden = model(inputs, cache: cache)
        // Every consumer of multi-token logits reads only the LAST
        // position's row; slicing before the head removes a
        // [length-1, vocab]-sized slab of dead work from every prefill.
        let hidden = lagunaLastTokenHidden(fullHidden)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func newCache(parameters _: GenerateParameters?) -> [KVCache] {
        (0..<configuration.numHiddenLayers).map { layerIndex in
            if configuration.layerTypes[layerIndex] == .full {
                StandardKVCache()
            } else {
                RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
            }
        }
    }

    /// Builds the retained fused runtime weight layouts (fused QKV, fused
    /// shared-expert gate/up, fused routed gate/up decode banks) once the
    /// checkpoint parameters are installed and evaluated. Called by the
    /// weight cache after `update` + `eval`, before constructor-time warmup,
    /// so the concatenations read materialized weights and the fused arrays
    /// are resident before the first forward. The module tree and its
    /// checkpoint parameters are never restructured; every fused layout is a
    /// derived side copy.
    func prepareFusedRuntimeWeights() {
        var fusedArrays: [MLXArray] = []
        for layer in model.layers {
            if lagunaFusedQKVEnabled, let fused = layer.selfAttn.prepareFusedQKVWeight() {
                fusedArrays.append(fused)
            }
            if let sparse = layer.mlp as? LagunaRuntimeSparseMoEBlock {
                if lagunaFusedSharedGateUpEnabled {
                    fusedArrays.append(contentsOf: sparse.sharedExpert.prepareFusedSharedGateUp())
                }
                if lagunaFusedRoutedGateUpEnabled {
                    fusedArrays.append(contentsOf: sparse.prepareFusedRoutedGateUp())
                }
            }
        }
        if !fusedArrays.isEmpty {
            eval(fusedArrays)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        // Drop precomputed rotary tables if a checkpoint ships them.
        return weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}
