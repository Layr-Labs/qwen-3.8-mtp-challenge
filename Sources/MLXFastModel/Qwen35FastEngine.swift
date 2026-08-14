import MLX
import MLXFastCore
import MLXLMCommon

/// Editable custom text trunk. It deliberately stops at normalized hidden
/// states so vocabulary projection policy is an independent seam.
public struct Qwen35FastTrunk {
    public let embedding: Qwen35LinearWeight
    public let blocks: [Qwen35BlockWeights]
    public let finalNorm: MLXArray
    public let hiddenSize: Int
    public let rmsNormEps: Double

    public init(
        embedding: Qwen35LinearWeight,
        blocks: [Qwen35BlockWeights],
        finalNorm: MLXArray,
        hiddenSize: Int,
        rmsNormEps: Double
    ) throws {
        guard hiddenSize > 0,
              !blocks.isEmpty,
              embedding.shape.count == 2,
              embedding.shape[1] == hiddenSize,
              finalNorm.shape == [hiddenSize]
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 fast trunk dimensions are invalid"
            )
        }
        self.embedding = embedding
        self.blocks = blocks
        self.finalNorm = finalNorm
        self.hiddenSize = hiddenSize
        self.rmsNormEps = rmsNormEps
        try validateContract()
    }

    public var layerTypes: [Qwen35LayerType] {
        blocks.map(\.mixer.layerType)
    }

    public func callAsFunction(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache? = nil,
        positionOffset: Int = 0,
        ssmMask: MLXArray? = nil
    ) throws -> MLXArray {
        try forward(
            inputIDs,
            cache: cache,
            positionOffset: positionOffset,
            ssmMask: ssmMask,
            afterLayer: nil
        )
    }

    func forward(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache?,
        positionOffset: Int,
        ssmMask: MLXArray?,
        afterLayer: ((Int) throws -> Void)?
    ) throws -> MLXArray {
        guard inputIDs.ndim == 2,
              inputIDs.dim(0) > 0,
              inputIDs.dim(1) > 0,
              positionOffset >= 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 fast trunk input IDs or position are invalid"
            )
        }
        if let ssmMask {
            guard ssmMask.shape == [inputIDs.dim(0), inputIDs.dim(1)],
                  ssmMask.dtype == .bool
            else {
                throw MLXFastError.invalidInput(
                    "Qwen35 fast trunk SSM mask must be boolean [batch, length]"
                )
            }
        }
        if cache == nil, positionOffset != 0 {
            throw MLXFastError.invalidInput(
                "Qwen35 fast trunk requires cache state for nonzero positions"
            )
        }

        let nextPosition = try cache?.validateAdvance(
            positionOffset: positionOffset,
            inputLength: inputIDs.dim(1),
            expectedLayerCount: blocks.count
        )
        var hidden = Qwen35Ops.embedding(
            inputIDs: inputIDs,
            weight: embedding
        )

        for (index, block) in blocks.enumerated() {
            let layerCache = cache?.layers[index]
            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            switch block.mixer {
            case .linear:
                attentionMask = .none
            case .full:
                attentionMask = makeAttentionMask(
                    n: inputIDs.dim(1),
                    cache: layerCache?.fullAttention
                )
            }
            hidden = try Qwen35Block.forward(
                hidden,
                weights: block,
                rmsNormEps: rmsNormEps,
                attentionMask: attentionMask,
                ssmMask: ssmMask,
                cache: layerCache,
                positionOffset: positionOffset
            )
            try afterLayer?(index)
        }

        let normalized = Qwen35Ops.rmsNorm(
            hidden,
            weight: finalNorm,
            eps: rmsNormEps
        )
        if let nextPosition {
            cache?.commit(positionOffset: nextPosition)
        }
        return normalized
    }

    func validateContract() throws {
        try embedding.validate()
        guard hiddenSize > 0,
              !blocks.isEmpty,
              embedding.shape.count == 2,
              embedding.shape[1] == hiddenSize,
              finalNorm.shape == [hiddenSize],
              rmsNormEps.isFinite,
              rmsNormEps > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 fast trunk contract is invalid"
            )
        }
        for block in blocks {
            try Qwen35Block.validateContract(
                weights: block,
                hiddenSize: hiddenSize
            )
        }
    }
}

/// Hybrid cache for the custom trunk: ordinary public KV caches for full
/// attention and package-owned states for Gated DeltaNet.
public final class Qwen35FastCache {
    public let layers: [Qwen35BlockCache]
    public private(set) var expectedPositionOffset = 0

    public init(trunk: Qwen35FastTrunk) {
        self.layers = trunk.blocks.map { block in
            switch block.mixer {
            case .linear:
                Qwen35BlockCache()
            case .full:
                Qwen35BlockCache(fullAttention: KVCacheSimple())
            }
        }
    }

    func validateAdvance(
        positionOffset: Int,
        inputLength: Int,
        expectedLayerCount: Int
    ) throws -> Int {
        guard layers.count == expectedLayerCount,
              positionOffset == expectedPositionOffset,
              inputLength > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 fast cache position or topology is invalid"
            )
        }
        let next = expectedPositionOffset.addingReportingOverflow(
            inputLength
        )
        guard !next.overflow else {
            throw MLXFastError.invalidInput(
                "Qwen35 fast cache position overflows Int"
            )
        }
        for layer in layers {
            if let fullAttention = layer.fullAttention,
               fullAttention.offset != positionOffset
            {
                throw MLXFastError.invalidInput(
                    "Qwen35 fast full-attention cache offset "
                        + "\(fullAttention.offset) does not match "
                        + "\(positionOffset)"
                )
            }
        }
        return next.partialValue
    }

    func commit(positionOffset: Int) {
        expectedPositionOffset = positionOffset
    }

    func withRollback<Result>(
        _ body: () throws -> Result
    ) throws -> Result {
        let snapshot = Qwen35FastCacheSnapshot(
            expectedPositionOffset: expectedPositionOffset,
            layers: layers.map { $0.snapshot() }
        )
        do {
            return try body()
        } catch {
            expectedPositionOffset = snapshot.expectedPositionOffset
            for (layer, saved) in zip(layers, snapshot.layers) {
                layer.restore(saved)
            }
            throw error
        }
    }

    public func materialize() {
        var arrays: [MLXArray] = []
        for layer in layers {
            if let fullAttention = layer.fullAttention {
                arrays.append(contentsOf: fullAttention.state)
            }
            if let gatedDelta = layer.gatedDelta {
                arrays.append(gatedDelta.convolution)
                arrays.append(gatedDelta.recurrent)
            }
        }
        eval(arrays)
    }
}

private struct Qwen35FastCacheSnapshot {
    let expectedPositionOffset: Int
    let layers: [Qwen35BlockCacheSnapshot]
}

/// Explicit untied output head. `lastTokenLogits` slices hidden states before
/// the 248,320-way projection, allowing a future scored prefill path to avoid
/// materializing all-position logits without changing the trunk.
public struct Qwen35LMHead {
    public let weight: Qwen35LinearWeight

    public init(weight: Qwen35LinearWeight) {
        self.weight = weight
    }

    public func logits(_ normalizedHidden: MLXArray) throws -> MLXArray {
        try validate(normalizedHidden)
        return Qwen35Ops.linear(normalizedHidden, weight)
    }

    public func lastTokenLogits(
        _ normalizedHidden: MLXArray
    ) throws -> MLXArray {
        try validate(normalizedHidden)
        let last = normalizedHidden[
            0...,
            (normalizedHidden.dim(1) - 1)..<normalizedHidden.dim(1),
            0...
        ]
        return Qwen35Ops.linear(last, weight)
    }

    private func validate(_ normalizedHidden: MLXArray) throws {
        guard normalizedHidden.ndim == 3,
              normalizedHidden.dim(1) > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 LM-head input or weight shape is invalid"
            )
        }
        try validateContract(hiddenSize: normalizedHidden.dim(2))
    }

    func validateContract(hiddenSize: Int) throws {
        try weight.validate()
        guard weight.shape.count == 2,
              weight.shape[0] > 0,
              weight.shape[1] == hiddenSize
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 LM-head weight shape is invalid"
            )
        }
    }
}

enum Qwen35FastFailurePoint: Equatable {
    case afterLayer(Int)
    case afterHead
}

private enum Qwen35FastInjectedFailure: Error {
    case requested
}

private enum Qwen35HeadProjection {
    case allPositions
    case lastToken
}

public struct Qwen35FastEngine {
    public let trunk: Qwen35FastTrunk
    public let lmHead: Qwen35LMHead

    public init(
        trunk: Qwen35FastTrunk,
        lmHead: Qwen35LMHead
    ) throws {
        self.trunk = trunk
        self.lmHead = lmHead
        try validateContract()
    }

    public func newCache() -> Qwen35FastCache {
        Qwen35FastCache(trunk: trunk)
    }

    public func allPositionLogits(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache? = nil,
        positionOffset: Int = 0,
        ssmMask: MLXArray? = nil
    ) throws -> MLXArray {
        try forward(
            inputIDs,
            cache: cache,
            positionOffset: positionOffset,
            ssmMask: ssmMask,
            projection: .allPositions,
            failurePoint: nil
        )
    }

    public func lastTokenLogits(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache? = nil,
        positionOffset: Int = 0,
        ssmMask: MLXArray? = nil
    ) throws -> MLXArray {
        try forward(
            inputIDs,
            cache: cache,
            positionOffset: positionOffset,
            ssmMask: ssmMask,
            projection: .lastToken,
            failurePoint: nil
        )
    }

    func allPositionLogits(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache,
        positionOffset: Int,
        ssmMask: MLXArray? = nil,
        failurePoint: Qwen35FastFailurePoint
    ) throws -> MLXArray {
        try forward(
            inputIDs,
            cache: cache,
            positionOffset: positionOffset,
            ssmMask: ssmMask,
            projection: .allPositions,
            failurePoint: failurePoint
        )
    }

    private func forward(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache?,
        positionOffset: Int,
        ssmMask: MLXArray?,
        projection: Qwen35HeadProjection,
        failurePoint: Qwen35FastFailurePoint?
    ) throws -> MLXArray {
        // Validate the complete immutable graph before any layer can mutate a
        // full-attention or recurrent cache.
        try validateContract()
        let activeCache = cache ?? newCache()
        return try activeCache.withRollback {
            let hidden = try trunk.forward(
                inputIDs,
                cache: activeCache,
                positionOffset: positionOffset,
                ssmMask: ssmMask,
                afterLayer: { layerIndex in
                    if failurePoint == .afterLayer(layerIndex) {
                        throw Qwen35FastInjectedFailure.requested
                    }
                }
            )
            let logits: MLXArray
            switch projection {
            case .allPositions:
                logits = try lmHead.logits(hidden)
            case .lastToken:
                logits = try lmHead.lastTokenLogits(hidden)
            }
            if failurePoint == .afterHead {
                throw Qwen35FastInjectedFailure.requested
            }
            return logits
        }
    }

    private func validateContract() throws {
        try trunk.validateContract()
        try lmHead.validateContract(hiddenSize: trunk.hiddenSize)
    }

    /// Materializes a complete custom engine from the transformed checkpoint.
    /// Production does not call this while the readiness gate is closed.
    public static func load(
        loader: Qwen35WeightLoader,
        config: Qwen35Config
    ) throws -> Qwen35FastEngine {
        try config.validateFrozenInvariants()
        try config.validateStructuralValues()
        try loader.denseStore.validateReadableByteRanges()
        try loader.validateRequiredMetadata(config: config)

        func linear(
            _ name: String,
            output: Int,
            input: Int
        ) throws -> Qwen35LinearWeight {
            try loader.linearWeight(
                named: name,
                outFeatures: output,
                inFeatures: input,
                expectedGroupSize: config.quantizationGroupSize,
                expectedBits: config.quantizationBits,
                requireQuantized: true
            )
        }

        let embedding = try linear(
            Qwen35WeightNames.embedTokens,
            output: config.vocabSize,
            input: config.hiddenSize
        )
        let attentionSpec = try Qwen35AttentionSpec(config: config)
        let gatedDeltaSpec = try Qwen35GatedDeltaSpec(config: config)
        var blocks: [Qwen35BlockWeights] = []
        blocks.reserveCapacity(config.numHiddenLayers)

        for layerIndex in 0..<config.numHiddenLayers {
            let mlp = Qwen35MLPWeights(
                gateProjection: try linear(
                    Qwen35WeightNames.mlp(
                        layerIndex,
                        "gate_proj.weight"
                    ),
                    output: config.intermediateSize,
                    input: config.hiddenSize
                ),
                upProjection: try linear(
                    Qwen35WeightNames.mlp(
                        layerIndex,
                        "up_proj.weight"
                    ),
                    output: config.intermediateSize,
                    input: config.hiddenSize
                ),
                downProjection: try linear(
                    Qwen35WeightNames.mlp(
                        layerIndex,
                        "down_proj.weight"
                    ),
                    output: config.hiddenSize,
                    input: config.intermediateSize
                )
            )

            let mixer: Qwen35BlockMixer
            switch config.layerTypes[layerIndex] {
            case .linear:
                let prefix: (String) -> String = {
                    Qwen35WeightNames.linearAttention(layerIndex, $0)
                }
                mixer = .linear(
                    weights: Qwen35GatedDeltaWeights(
                        inputQKVProjection: try linear(
                            prefix("in_proj_qkv.weight"),
                            output: gatedDeltaSpec.convolutionDimension,
                            input: config.hiddenSize
                        ),
                        inputZProjection: try linear(
                            prefix("in_proj_z.weight"),
                            output: gatedDeltaSpec.valueSize,
                            input: config.hiddenSize
                        ),
                        inputBProjection: try linear(
                            prefix("in_proj_b.weight"),
                            output: gatedDeltaSpec.numValueHeads,
                            input: config.hiddenSize
                        ),
                        inputAProjection: try linear(
                            prefix("in_proj_a.weight"),
                            output: gatedDeltaSpec.numValueHeads,
                            input: config.hiddenSize
                        ),
                        convolution: try loader.denseArray(
                            named: prefix("conv1d.weight"),
                            expectedShape: [
                                gatedDeltaSpec.convolutionDimension,
                                gatedDeltaSpec.convolutionKernelSize,
                                1,
                            ]
                        ),
                        timeStepBias: try loader.denseArray(
                            named: prefix("dt_bias"),
                            expectedShape: [
                                gatedDeltaSpec.numValueHeads
                            ]
                        ),
                        aLog: try loader.denseArray(
                            named: prefix("A_log"),
                            expectedShape: [
                                gatedDeltaSpec.numValueHeads
                            ]
                        ),
                        outputNorm: try loader.denseArray(
                            named: prefix("norm.weight"),
                            expectedShape: [
                                gatedDeltaSpec.valueHeadDimension
                            ]
                        ),
                        outputProjection: try linear(
                            prefix("out_proj.weight"),
                            output: config.hiddenSize,
                            input: gatedDeltaSpec.valueSize
                        )
                    ),
                    spec: gatedDeltaSpec
                )

            case .full:
                let prefix: (String) -> String = {
                    Qwen35WeightNames.fullAttention(layerIndex, $0)
                }
                mixer = .full(
                    weights: Qwen35AttentionWeights(
                        queryProjection: try linear(
                            prefix("q_proj.weight"),
                            output: attentionSpec
                                .queryAndGateProjectionSize,
                            input: config.hiddenSize
                        ),
                        keyProjection: try linear(
                            prefix("k_proj.weight"),
                            output: attentionSpec.keyValueSize,
                            input: config.hiddenSize
                        ),
                        valueProjection: try linear(
                            prefix("v_proj.weight"),
                            output: attentionSpec.keyValueSize,
                            input: config.hiddenSize
                        ),
                        outputProjection: try linear(
                            prefix("o_proj.weight"),
                            output: config.hiddenSize,
                            input: attentionSpec.querySize
                        ),
                        queryNorm: try loader.denseArray(
                            named: prefix("q_norm.weight"),
                            expectedShape: [config.headDim]
                        ),
                        keyNorm: try loader.denseArray(
                            named: prefix("k_norm.weight"),
                            expectedShape: [config.headDim]
                        )
                    ),
                    spec: attentionSpec
                )
            }

            blocks.append(
                Qwen35BlockWeights(
                    inputLayerNorm: try loader.denseArray(
                        named: Qwen35WeightNames.layer(
                            layerIndex,
                            "input_layernorm.weight"
                        ),
                        expectedShape: [config.hiddenSize]
                    ),
                    postAttentionLayerNorm: try loader.denseArray(
                        named: Qwen35WeightNames.layer(
                            layerIndex,
                            "post_attention_layernorm.weight"
                        ),
                        expectedShape: [config.hiddenSize]
                    ),
                    mixer: mixer,
                    mlp: mlp
                )
            )
        }

        let trunk = try Qwen35FastTrunk(
            embedding: embedding,
            blocks: blocks,
            finalNorm: try loader.denseArray(
                named: Qwen35WeightNames.finalNorm,
                expectedShape: [config.hiddenSize]
            ),
            hiddenSize: config.hiddenSize,
            rmsNormEps: config.rmsNormEps
        )
        return try Qwen35FastEngine(
            trunk: trunk,
            lmHead: Qwen35LMHead(
                weight: try linear(
                    Qwen35WeightNames.lmHead,
                    output: config.vocabSize,
                    input: config.hiddenSize
                )
            )
        )
    }
}
