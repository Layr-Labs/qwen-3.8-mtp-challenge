import MLX
import MLXFastCore
import MLXLLM
import MLXNN

public struct Qwen35GatedDeltaSpec {
    public let hiddenSize: Int
    public let numValueHeads: Int
    public let numKeyHeads: Int
    public let keyHeadDimension: Int
    public let valueHeadDimension: Int
    public let convolutionKernelSize: Int
    public let rmsNormEps: Double
    private let computedKeySize: Int
    private let computedValueSize: Int
    private let computedConvolutionDimension: Int

    public init(
        hiddenSize: Int,
        numValueHeads: Int,
        numKeyHeads: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int,
        convolutionKernelSize: Int,
        rmsNormEps: Double
    ) throws {
        guard hiddenSize > 0,
              numValueHeads > 0,
              numKeyHeads > 0,
              numValueHeads.isMultiple(of: numKeyHeads),
              keyHeadDimension > 0,
              valueHeadDimension > 0,
              convolutionKernelSize > 1,
              rmsNormEps.isFinite,
              rmsNormEps > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 Gated DeltaNet dimensions are invalid"
            )
        }
        let keySize = try qwen35CheckedProduct(
            [numKeyHeads, keyHeadDimension],
            label: "Qwen35 Gated DeltaNet key size"
        )
        let valueSize = try qwen35CheckedProduct(
            [numValueHeads, valueHeadDimension],
            label: "Qwen35 Gated DeltaNet value size"
        )
        let doubledKeySize = try qwen35CheckedProduct(
            [keySize, 2],
            label: "Qwen35 Gated DeltaNet doubled key size"
        )
        let convolutionDimension = try qwen35CheckedSum(
            [doubledKeySize, valueSize],
            label: "Qwen35 Gated DeltaNet convolution dimension"
        )
        _ = try qwen35CheckedProduct(
            [numValueHeads, valueHeadDimension, keyHeadDimension],
            label: "Qwen35 Gated DeltaNet recurrent state size"
        )
        _ = try qwen35CheckedProduct(
            [convolutionKernelSize - 1, convolutionDimension],
            label: "Qwen35 Gated DeltaNet convolution state size"
        )
        self.hiddenSize = hiddenSize
        self.numValueHeads = numValueHeads
        self.numKeyHeads = numKeyHeads
        self.keyHeadDimension = keyHeadDimension
        self.valueHeadDimension = valueHeadDimension
        self.convolutionKernelSize = convolutionKernelSize
        self.rmsNormEps = rmsNormEps
        self.computedKeySize = keySize
        self.computedValueSize = valueSize
        self.computedConvolutionDimension = convolutionDimension
    }

    public init(config: Qwen35Config) throws {
        try self.init(
            hiddenSize: config.hiddenSize,
            numValueHeads: config.linearNumValueHeads,
            numKeyHeads: config.linearNumKeyHeads,
            keyHeadDimension: config.linearKeyHeadDim,
            valueHeadDimension: config.linearValueHeadDim,
            convolutionKernelSize: config.linearConvKernelDim,
            rmsNormEps: config.rmsNormEps
        )
    }

    public var keySize: Int {
        computedKeySize
    }

    public var valueSize: Int {
        computedValueSize
    }

    public var convolutionDimension: Int {
        computedConvolutionDimension
    }

    public func convolutionStateShape(batchSize: Int) -> [Int] {
        [
            batchSize,
            convolutionKernelSize - 1,
            convolutionDimension,
        ]
    }

    public func recurrentStateShape(batchSize: Int) -> [Int] {
        [
            batchSize,
            numValueHeads,
            valueHeadDimension,
            keyHeadDimension,
        ]
    }
}

public struct Qwen35GatedDeltaWeights {
    public let inputQKVProjection: Qwen35LinearWeight
    public let inputZProjection: Qwen35LinearWeight
    public let inputBProjection: Qwen35LinearWeight
    public let inputAProjection: Qwen35LinearWeight
    public let convolution: MLXArray
    public let timeStepBias: MLXArray
    public let aLog: MLXArray
    public let outputNorm: MLXArray
    public let outputProjection: Qwen35LinearWeight

    public init(
        inputQKVProjection: Qwen35LinearWeight,
        inputZProjection: Qwen35LinearWeight,
        inputBProjection: Qwen35LinearWeight,
        inputAProjection: Qwen35LinearWeight,
        convolution: MLXArray,
        timeStepBias: MLXArray,
        aLog: MLXArray,
        outputNorm: MLXArray,
        outputProjection: Qwen35LinearWeight
    ) {
        self.inputQKVProjection = inputQKVProjection
        self.inputZProjection = inputZProjection
        self.inputBProjection = inputBProjection
        self.inputAProjection = inputAProjection
        self.convolution = convolution
        self.timeStepBias = timeStepBias
        self.aLog = aLog
        self.outputNorm = outputNorm
        self.outputProjection = outputProjection
    }
}

/// Cache owned by the custom Qwen35 linear-attention path.
///
/// At pinned dimensions these are `[B, 3, 10240]` and
/// `[B, 48, 128, 128]`; the recurrent state is always float32.
public final class Qwen35GatedDeltaState {
    public private(set) var convolution: MLXArray
    public private(set) var recurrent: MLXArray

    public init(
        batchSize: Int,
        spec: Qwen35GatedDeltaSpec,
        activationDType: DType
    ) throws {
        guard batchSize > 0 else {
            throw MLXFastError.invalidInput(
                "Qwen35 Gated DeltaNet batch size must be positive"
            )
        }
        _ = try qwen35CheckedProduct(
            spec.convolutionStateShape(batchSize: batchSize),
            label: "Qwen35 Gated DeltaNet batched convolution state"
        )
        _ = try qwen35CheckedProduct(
            spec.recurrentStateShape(batchSize: batchSize),
            label: "Qwen35 Gated DeltaNet batched recurrent state"
        )
        self.convolution = MLXArray.zeros(
            spec.convolutionStateShape(batchSize: batchSize),
            dtype: activationDType
        )
        self.recurrent = MLXArray.zeros(
            spec.recurrentStateShape(batchSize: batchSize),
            dtype: .float32
        )
    }

    func replace(
        convolution: MLXArray,
        recurrent: MLXArray
    ) {
        self.convolution = convolution
        self.recurrent = recurrent.dtype == .float32
            ? recurrent
            : recurrent.asType(.float32)
    }

    func copy() -> Qwen35GatedDeltaState {
        Qwen35GatedDeltaState(
            convolution: convolution,
            recurrent: recurrent
        )
    }

    private init(
        convolution: MLXArray,
        recurrent: MLXArray
    ) {
        self.convolution = convolution
        self.recurrent = recurrent
    }
}

/// Qwen35 Gated DeltaNet scaffold copied from pinned `Qwen35GatedDeltaNet`.
///
/// The public `MLXLLM.gatedDeltaUpdate(q:k:v:a:b:aLog:dtBias:state:mask:)`
/// call below is intentionally the recurrence source of truth. That API
/// computes beta and decay `g` in float32 and preserves float32 recurrent
/// state, as documented in pinned `GatedDelta.swift`.
public enum Qwen35GatedDeltaNet {
    public static func forward(
        _ input: MLXArray,
        weights: Qwen35GatedDeltaWeights,
        spec: Qwen35GatedDeltaSpec,
        mask: MLXArray? = nil,
        state: Qwen35GatedDeltaState? = nil
    ) throws -> MLXArray {
        #if DEBUG
        try validate(
            input,
            weights: weights,
            spec: spec,
            mask: mask,
            state: state
        )
        #endif

        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)
        var mixedQKV = Qwen35Ops.linear(
            input,
            weights.inputQKVProjection
        )
        let z = Qwen35Ops.linear(
            input,
            weights.inputZProjection
        ).reshaped(
            batchSize,
            sequenceLength,
            spec.numValueHeads,
            spec.valueHeadDimension
        )
        let b = Qwen35Ops.linear(input, weights.inputBProjection)
        let a = Qwen35Ops.linear(input, weights.inputAProjection)

        if let mask {
            mixedQKV = MLX.where(
                mask[.ellipsis, .newAxis],
                mixedQKV,
                0
            )
        }

        let convolutionState = state?.convolution
            ?? MLXArray.zeros(
                spec.convolutionStateShape(batchSize: batchSize),
                dtype: input.dtype
            )
        let convolutionInput = concatenated(
            [convolutionState, mixedQKV],
            axis: 1
        )
        let keepCount = spec.convolutionKernelSize - 1
        let nextConvolutionState = convolutionInput[
            0...,
            (convolutionInput.dim(1) - keepCount)...,
            0...
        ]

        // Pinned Qwen35 Conv1d is depthwise, causal via the prepended state,
        // with weight shape [convDimension, kernel=4, 1].
        let convolutionOutput = silu(
            conv1d(
                convolutionInput,
                weights.convolution,
                groups: spec.convolutionDimension
            )
        )
        let split = MLX.split(
            convolutionOutput,
            indices: [spec.keySize, 2 * spec.keySize],
            axis: -1
        )
        let query = split[0].reshaped(
            batchSize,
            sequenceLength,
            spec.numKeyHeads,
            spec.keyHeadDimension
        )
        let key = split[1].reshaped(
            batchSize,
            sequenceLength,
            spec.numKeyHeads,
            spec.keyHeadDimension
        )
        let value = split[2].reshaped(
            batchSize,
            sequenceLength,
            spec.numValueHeads,
            spec.valueHeadDimension
        )

        // Exact pinned normalization: q gets invScale², k gets invScale.
        let inputDType = query.dtype
        let inverseScale = 1 / Float(spec.keyHeadDimension).squareRoot()
        let normalizedQuery =
            MLXArray(inverseScale * inverseScale).asType(inputDType)
            * Qwen35Ops.rmsNorm(query, eps: 1e-6)
        let normalizedKey =
            MLXArray(inverseScale).asType(inputDType)
            * Qwen35Ops.rmsNorm(key, eps: 1e-6)

        let (updated, nextRecurrentState) = gatedDeltaUpdate(
            q: normalizedQuery,
            k: normalizedKey,
            v: value,
            a: a,
            b: b,
            aLog: weights.aLog,
            dtBias: weights.timeStepBias,
            state: state?.recurrent,
            mask: mask
        )
        state?.replace(
            convolution: nextConvolutionState,
            recurrent: nextRecurrentState
        )

        let gated = Qwen35Ops.preciseGatedRMSNorm(
            updated,
            gate: z,
            weight: weights.outputNorm,
            eps: spec.rmsNormEps
        )
        return Qwen35Ops.linear(
            gated.reshaped(batchSize, sequenceLength, -1),
            weights.outputProjection
        )
    }

    private static func validate(
        _ input: MLXArray,
        weights: Qwen35GatedDeltaWeights,
        spec: Qwen35GatedDeltaSpec,
        mask: MLXArray?,
        state: Qwen35GatedDeltaState?
    ) throws {
        guard input.ndim == 3,
              input.dim(2) == spec.hiddenSize
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 Gated DeltaNet input shape is invalid"
            )
        }
        if let mask {
            guard mask.shape == [input.dim(0), input.dim(1)],
                  mask.dtype == .bool
            else {
                throw MLXFastError.invalidInput(
                    "Qwen35 Gated DeltaNet mask must be boolean [batch, length]"
                )
            }
        }
        if let state, state.convolution.dtype != input.dtype {
            throw MLXFastError.invalidInput(
                "Qwen35 Gated DeltaNet convolution state dtype does not match input"
            )
        }
        try validateContract(weights: weights, spec: spec)
        if let state {
            guard state.convolution.shape
                    == spec.convolutionStateShape(batchSize: input.dim(0)),
                  state.recurrent.shape
                    == spec.recurrentStateShape(batchSize: input.dim(0)),
                  state.recurrent.dtype == .float32
            else {
                throw MLXFastError.invalidInput(
                    "Qwen35 Gated DeltaNet cache state is invalid"
                )
            }
        }
    }

    static func validateContract(
        weights: Qwen35GatedDeltaWeights,
        spec: Qwen35GatedDeltaSpec
    ) throws {
        try weights.inputQKVProjection.validate()
        try weights.inputZProjection.validate()
        try weights.inputBProjection.validate()
        try weights.inputAProjection.validate()
        try weights.outputProjection.validate()
        guard weights.inputQKVProjection.shape
                == [spec.convolutionDimension, spec.hiddenSize],
              weights.inputZProjection.shape
                == [spec.valueSize, spec.hiddenSize],
              weights.inputBProjection.shape
                == [spec.numValueHeads, spec.hiddenSize],
              weights.inputAProjection.shape
                == [spec.numValueHeads, spec.hiddenSize],
              weights.convolution.shape
                == [
                    spec.convolutionDimension,
                    spec.convolutionKernelSize,
                    1,
                ],
              weights.timeStepBias.shape == [spec.numValueHeads],
              weights.aLog.shape == [spec.numValueHeads],
              weights.outputNorm.shape == [spec.valueHeadDimension],
              weights.outputProjection.shape
                == [spec.hiddenSize, spec.valueSize]
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 Gated DeltaNet input or weight shapes are invalid"
            )
        }
    }
}
