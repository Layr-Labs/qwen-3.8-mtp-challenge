import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Faithful Swift/MLX port of z-lab DFlash2 for Qwen3.8-27B.
//
// The drafter is proposal-only. It shares the pinned target's embedding and
// lm_head through closures, while every proposed row is still verified by the
// trusted Qwen target and replayed by the parent. The declared artifact carries
// only DFlash2's learned tensors.

private enum QwenDFlash2Geometry {
    static let hidden = 5_120
    static let layers = 5
    static let heads = 32
    static let kvHeads = 8
    static let headDim = 128
    static let intermediate = 17_408
    static let vocabulary = 248_320
    static let targetFeatureWidth = 25_600
    static let maskToken = 248_070
    static let window = 2_048
    static let convKernel = 2
    static let convGroup = 16
    static let selectorRank = 256
    static let selectorTopK = 16
    static let rmsEpsilon: Float = 1e-6
    static let ropeTheta: Float = 10_000_000
    static let maxPositions = 262_144
}

private let qwenDFlash2SiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { gate, up in silu(gate) * up }

private final class QwenDFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    // Pack the already-quantized gate/up rows on their output dimension.  A
    // single affine GEMM is bit-identical to the two independent projections,
    // while avoiding one weight-stream launch in every draft layer.
    private var fusedWeight: MLXArray?
    private var fusedScales: MLXArray?
    private var fusedBiases: MLXArray?
    private var fusedGroupSize = 64
    private var fusedBits = 4
    private var fusedMode = QuantizationMode.affine
    private var gateWidth = 0

    override init() {
        _gate.wrappedValue = Linear(
            QwenDFlash2Geometry.hidden,
            QwenDFlash2Geometry.intermediate,
            bias: false)
        _up.wrappedValue = Linear(
            QwenDFlash2Geometry.hidden,
            QwenDFlash2Geometry.intermediate,
            bias: false)
        _down.wrappedValue = Linear(
            QwenDFlash2Geometry.intermediate,
            QwenDFlash2Geometry.hidden,
            bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let weight = fusedWeight,
           let scales = fusedScales,
           let biases = fusedBiases
        {
            let projected = quantizedMM(
                x, weight, scales: scales, biases: biases, transpose: true,
                groupSize: fusedGroupSize, bits: fusedBits, mode: fusedMode)
            return down(qwenDFlash2SiluProduct(
                projected[.ellipsis, ..<gateWidth],
                projected[.ellipsis, gateWidth...]))
        }
        if let gate = gate as? QuantizedLinear,
           let up = up as? QuantizedLinear,
           gate.groupSize == up.groupSize,
           gate.bits == up.bits,
           gate.mode == up.mode,
           gate.mode == .affine,
           let gateBiases = gate.biases,
           let upBiases = up.biases
        {
            fusedWeight = concatenated([gate.weight, up.weight], axis: 0)
                .contiguous()
            fusedScales = concatenated([gate.scales, up.scales], axis: 0)
                .contiguous()
            fusedBiases = concatenated([gateBiases, upBiases], axis: 0)
                .contiguous()
            fusedGroupSize = gate.groupSize
            fusedBits = gate.bits
            fusedMode = gate.mode
            gateWidth = gate.shape.0
            return callAsFunction(x)
        }
        return down(qwenDFlash2SiluProduct(gate(x), up(x)))
    }
}

private final class QwenDFlash2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let rope: RoPELayer
    private let scale = pow(Float(QwenDFlash2Geometry.headDim), -0.5)

    // Proposal rows need Q/K/V; committed context rows need only K/V.  Keep a
    // packed form for each so neither path computes dead query rows, while both
    // replace three/two affine launches with one exact concatenated projection.
    private var qkvWeight: MLXArray?
    private var qkvScales: MLXArray?
    private var qkvBiases: MLXArray?
    private var qkvGroupSize = 64
    private var qkvBits = 4
    private var qkvMode = QuantizationMode.affine
    private var qWidth = 0
    private var kWidth = 0
    private var kvWeight: MLXArray?
    private var kvScales: MLXArray?
    private var kvBiases: MLXArray?
    private var kvGroupSize = 64
    private var kvBits = 4
    private var kvMode = QuantizationMode.affine
    private var kvKWidth = 0

    override init() {
        let g = QwenDFlash2Geometry.self
        _qProj.wrappedValue = Linear(g.hidden, g.heads * g.headDim, bias: false)
        _kProj.wrappedValue = Linear(g.hidden, g.kvHeads * g.headDim, bias: false)
        _vProj.wrappedValue = Linear(g.hidden, g.kvHeads * g.headDim, bias: false)
        _oProj.wrappedValue = Linear(g.heads * g.headDim, g.hidden, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: g.headDim, eps: g.rmsEpsilon)
        _kNorm.wrappedValue = RMSNorm(dimensions: g.headDim, eps: g.rmsEpsilon)
        self.rope = initializeRope(
            dims: g.headDim,
            base: g.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: g.maxPositions)
        super.init()
    }

    private func qkv(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        if let weight = qkvWeight,
           let scales = qkvScales,
           let biases = qkvBiases
        {
            let projected = quantizedMM(
                x, weight, scales: scales, biases: biases, transpose: true,
                groupSize: qkvGroupSize, bits: qkvBits, mode: qkvMode)
            let kEnd = qWidth + kWidth
            return (
                projected[.ellipsis, ..<qWidth],
                projected[.ellipsis, qWidth ..< kEnd],
                projected[.ellipsis, kEnd...])
        }
        if let q = qProj as? QuantizedLinear,
           let k = kProj as? QuantizedLinear,
           let v = vProj as? QuantizedLinear,
           q.groupSize == k.groupSize,
           k.groupSize == v.groupSize,
           q.bits == k.bits,
           k.bits == v.bits,
           q.mode == k.mode,
           k.mode == v.mode,
           q.mode == .affine,
           let qBiases = q.biases,
           let kBiases = k.biases,
           let vBiases = v.biases
        {
            qkvWeight = concatenated([q.weight, k.weight, v.weight], axis: 0)
                .contiguous()
            qkvScales = concatenated([q.scales, k.scales, v.scales], axis: 0)
                .contiguous()
            qkvBiases = concatenated([qBiases, kBiases, vBiases], axis: 0)
                .contiguous()
            qkvGroupSize = q.groupSize
            qkvBits = q.bits
            qkvMode = q.mode
            qWidth = q.shape.0
            kWidth = k.shape.0
            return qkv(x)
        }
        return (qProj(x), kProj(x), vProj(x))
    }

    private func kv(_ x: MLXArray) -> (MLXArray, MLXArray) {
        if let weight = kvWeight,
           let scales = kvScales,
           let biases = kvBiases
        {
            let projected = quantizedMM(
                x, weight, scales: scales, biases: biases, transpose: true,
                groupSize: kvGroupSize, bits: kvBits, mode: kvMode)
            return (
                projected[.ellipsis, ..<kvKWidth],
                projected[.ellipsis, kvKWidth...])
        }
        if let k = kProj as? QuantizedLinear,
           let v = vProj as? QuantizedLinear,
           k.groupSize == v.groupSize,
           k.bits == v.bits,
           k.mode == v.mode,
           k.mode == .affine,
           let kBiases = k.biases,
           let vBiases = v.biases
        {
            kvWeight = concatenated([k.weight, v.weight], axis: 0).contiguous()
            kvScales = concatenated([k.scales, v.scales], axis: 0).contiguous()
            kvBiases = concatenated([kBiases, vBiases], axis: 0).contiguous()
            kvGroupSize = k.groupSize
            kvBits = k.bits
            kvMode = k.mode
            kvKWidth = k.shape.0
            return kv(x)
        }
        return (kProj(x), vProj(x))
    }

    func callAsFunction(
        _ x: MLXArray,
        context initialContext: MLXArray,
        cache: any KVCache
    ) -> MLXArray {
        let g = QwenDFlash2Geometry.self
        let B = x.dim(0)
        let L = x.dim(1)
        var context = initialContext
        var S = context.dim(1)

        // The official DFlash2 layers are all sliding attention. The ranked
        // 512+512 window stays below this seam, but retaining the upstream
        // truncation makes longer local validation position-correct too.
        let keepContext = g.window - 1
        if S > keepContext {
            let skip = S - keepContext
            context = context[0..., skip..., 0...]
            S = context.dim(1)
            if let base = cache as? BaseKVCache {
                base.offset += skip
            }
        }

        var (queries, proposalKeys, proposalValues) = qkv(x)
        var (contextKeys, contextValues) = kv(context)

        queries = qNorm(queries.reshaped(B, L, g.heads, g.headDim))
            .transposed(0, 2, 1, 3)
        contextKeys = qwenDFlash2NormalizeKeys(
            contextKeys, norm: kNorm, B: B, L: S)
        contextValues = contextValues.reshaped(B, S, g.kvHeads, g.headDim)
            .transposed(0, 2, 1, 3)
        proposalKeys = qwenDFlash2NormalizeKeys(
            proposalKeys, norm: kNorm, B: B, L: L)
        proposalValues = proposalValues.reshaped(B, L, g.kvHeads, g.headDim)
            .transposed(0, 2, 1, 3)

        let baseOffset = cache.offset
        queries = rope(queries, offset: baseOffset + S)
        contextKeys = rope(contextKeys, offset: baseOffset)
        proposalKeys = rope(proposalKeys, offset: baseOffset + S)

        let (cachedKeys, cachedValues) = cache.update(
            keys: contextKeys, values: contextValues)
        let keys = concatenated([cachedKeys, proposalKeys], axis: 2)
        let values = concatenated([cachedValues, proposalValues], axis: 2)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .causal)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, g.heads * g.headDim)
        return oProj(output)
    }
}

private func qwenDFlash2NormalizeKeys(
    _ keys: MLXArray, norm: RMSNorm, B: Int, L: Int
) -> MLXArray {
    norm(keys.reshaped(
        B, L, QwenDFlash2Geometry.kvHeads, QwenDFlash2Geometry.headDim))
        .transposed(0, 2, 1, 3)
}

private final class QwenDFlash2GroupedCausalConv: Module {
    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    override init() {
        let g = QwenDFlash2Geometry.self
        _baseKernel.wrappedValue = zeros(
            [2, g.convKernel, g.hidden], dtype: .bfloat16)
        _kernelProjection.wrappedValue = Linear(
            g.hidden,
            2 * g.convKernel * (g.hidden / g.convGroup),
            bias: false)
        super.init()
    }

    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let g = QwenDFlash2Geometry.self
        let dynamic = kernelProjection(hidden).reshaped(
            hidden.dim(0), hidden.dim(1), 2, g.convKernel,
            g.hidden / g.convGroup)
        let halves = dynamic.split(parts: 2, axis: 2)
        return (
            convolve(hidden, dynamic: halves[0].squeezed(axis: 2), stage: 0),
            halves[1].squeezed(axis: 2)
        )
    }

    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        convolve(hidden, dynamic: dynamic, stage: 1)
    }

    private func convolve(
        _ hidden: MLXArray,
        dynamic: MLXArray,
        stage: Int
    ) -> MLXArray {
        let g = QwenDFlash2Geometry.self
        let B = hidden.dim(0)
        let L = hidden.dim(1)
        let groups = g.hidden / g.convGroup
        let blocks = hidden.reshaped(B, L, groups, g.convGroup)
        var output = zeros(blocks.shape, dtype: blocks.dtype)
        for offset in 0 ..< g.convKernel {
            let values: MLXArray
            if offset == 0 {
                values = blocks
            } else {
                let pad = zeros(
                    [B, offset, groups, g.convGroup], dtype: hidden.dtype)
                let prefix = blocks[0..., 0 ..< (L - offset), 0..., 0...]
                values = concatenated([pad, prefix], axis: 1)
            }
            let kernel = baseKernel[stage, offset, 0...]
                .reshaped(1, 1, groups, g.convGroup)
                .asType(hidden.dtype)
            let dynamicOffset = dynamic[0..., 0..., offset, 0...]
                .expandedDimensions(axis: -1)
            output = output + kernel * values + dynamicOffset * values
        }
        return output.reshaped(hidden.shape)
    }
}

private final class QwenDFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: QwenDFlash2Attention
    @ModuleInfo var mlp: QwenDFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: QwenDFlash2GroupedCausalConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: QwenDFlash2GroupedCausalConv

    override init() {
        let g = QwenDFlash2Geometry.self
        _attention.wrappedValue = QwenDFlash2Attention()
        _mlp.wrappedValue = QwenDFlash2MLP()
        _inputNorm.wrappedValue = RMSNorm(dimensions: g.hidden, eps: g.rmsEpsilon)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: g.hidden, eps: g.rmsEpsilon)
        _attentionConv.wrappedValue = QwenDFlash2GroupedCausalConv()
        _mlpConv.wrappedValue = QwenDFlash2GroupedCausalConv()
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        cache: any KVCache
    ) -> MLXArray {
        let residual = x
        let (preparedAttention, attentionKernel) = attentionConv.prepare(inputNorm(x))
        let afterAttention = residual + attentionConv.finish(
            attention(preparedAttention, context: context, cache: cache),
            dynamic: attentionKernel)
        let (preparedMLP, mlpKernel) = mlpConv.prepare(
            postAttentionNorm(afterAttention))
        return afterAttention + mlpConv.finish(
            mlp(preparedMLP), dynamic: mlpKernel)
    }
}

private final class QwenDFlash2CandidateSelector: Module {
    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    override init() {
        let g = QwenDFlash2Geometry.self
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: g.vocabulary, dimensions: g.selectorRank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: g.vocabulary, dimensions: g.selectorRank)
        _hiddenProjection.wrappedValue = Linear(
            g.hidden, g.selectorRank, bias: false)
        super.init()
    }

    func select(
        hidden: MLXArray,
        logits: MLXArray,
        anchor: MLXArray
    ) -> MLXArray {
        let topK = QwenDFlash2Geometry.selectorTopK
        let kth = logits.dim(-1) - topK
        let candidates = argPartition(logits, kth: kth, axis: -1)[
            .ellipsis, kth...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)
        var predecessor = anchor
        var path: [MLXArray] = []
        path.reserveCapacity(hidden.dim(1))
        for position in 0 ..< hidden.dim(1) {
            let candidateRow = candidates[0..., position, 0...]
            let pred = predecessorCodebook(predecessor)
                .expandedDimensions(axis: 1)
            let feature = projected[0..., position, 0...]
                .expandedDimensions(axis: 1)
            let successor = successorCodebook(candidateRow)
            let edge = (pred * feature * successor).sum(axis: -1)
            let scores = unary[0..., position, 0...] + edge
            let selected = argMax(scores, axis: -1)
            predecessor = takeAlong(
                candidateRow,
                selected[.ellipsis, .newAxis],
                axis: -1).squeezed(axis: -1)
            path.append(predecessor)
        }
        return stacked(path, axis: 1).asType(.int32)
    }
}

final class QwenDFlash2DraftModel: Module {
    @ModuleInfo(key: "fc") var contextProjection: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") fileprivate var layers: [QwenDFlash2DecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") fileprivate var candidateSelector:
        QwenDFlash2CandidateSelector

    private let embed: (MLXArray) -> MLXArray
    private let lmHead: (MLXArray) -> MLXArray

    init(
        embed: @escaping (MLXArray) -> MLXArray,
        lmHead: @escaping (MLXArray) -> MLXArray
    ) {
        let g = QwenDFlash2Geometry.self
        _contextProjection.wrappedValue = Linear(
            g.targetFeatureWidth, g.hidden, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: g.hidden, eps: g.rmsEpsilon)
        _layers.wrappedValue = (0 ..< g.layers).map { _ in
            QwenDFlash2DecoderLayer()
        }
        _norm.wrappedValue = RMSNorm(dimensions: g.hidden, eps: g.rmsEpsilon)
        _candidateSelector.wrappedValue = QwenDFlash2CandidateSelector()
        self.embed = embed
        self.lmHead = lmHead
        super.init()
    }

    static func load(
        from directory: URL,
        target: any Qwen36MTPTarget
    ) throws -> QwenDFlash2DraftModel {
        let url = directory.appendingPathComponent("model.safetensors")
        var weights = try loadArrays(url: url)
        // z-lab's BF16 file stores the codebooks as bare parameter names;
        // Embedding modules use the canonical `.weight` spelling.
        for name in ["predecessor_codebook", "successor_codebook"] {
            let bare = "candidate_selector.\(name)"
            let canonical = bare + ".weight"
            if weights[canonical] == nil, let value = weights.removeValue(forKey: bare) {
                weights[canonical] = value
            }
        }

        let model = QwenDFlash2DraftModel(
            embed: { [target] tokens in target.embedTokensForDFlash2(tokens) },
            lmHead: { [target] hidden in target.applyLMHead(hidden) })
        quantize(model: model) { path, _ in
            guard let packed = weights["\(path).weight"],
                  let scales = weights["\(path).scales"]
            else { return nil }
            let inputWidth = scales.dim(1) * 64
            let bits = packed.dim(1) * 32 / inputWidth
            return (groupSize: 64, bits: bits, mode: .affine)
        }
        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.all])
        eval(model)
        return model
    }

    func makeCache() -> [any KVCache] {
        (0 ..< QwenDFlash2Geometry.layers).map { _ in
            KVCacheSimple() as any KVCache
        }
    }

    func propose(
        anchorToken: Int,
        targetFeatures: MLXArray,
        cache: [any KVCache],
        draftCount: Int
    ) -> MLXArray {
        precondition(cache.count == layers.count)
        precondition(draftCount >= 1 && draftCount <= 8)
        let tokens = MLXArray(
            [Int32(anchorToken)]
                + Array(repeating: Int32(QwenDFlash2Geometry.maskToken),
                        count: draftCount))
            .reshaped([1, draftCount + 1])
        var hidden = embed(tokens)
        let context = hiddenNorm(contextProjection(targetFeatures))
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, context: context, cache: cache[index])
        }
        hidden = norm(hidden)[0..., 1..., 0...]
        let logits = lmHead(hidden)
        return candidateSelector.select(
            hidden: hidden,
            logits: logits,
            anchor: MLXArray([Int32(anchorToken)]))
    }
}
