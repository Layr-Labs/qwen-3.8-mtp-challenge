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

// MARK: - Local recurrence kernel (bit-exact row-interleaved GDN scan)

/// Clone of the vendored `gated_delta_step` kernel (MLXLLM `GatedDelta.swift`,
/// unmasked instantiation) carrying one measured change: each simdgroup owns
/// `R` consecutive (hv, dv) rows instead of one, interleaving them to hide the
/// two dependent `simd_sum` latencies per step and to amortize the q/k row
/// loads shared by every dv row of a head.
///
/// Bit-exactness by construction: per (dv, dk) element the arithmetic sequence
/// is unchanged — lane L still holds dk {4L..4L+3}, partials still accumulate
/// in ascending i, and each row's two reductions are `simd_sum` over the same
/// 32 lane values through the same hardware butterfly tree. Rows are
/// independent, so interleaving them reorders no element's operations.
/// Verified bitwise (byte-identical fp16/bf16 y and fp32 state_out) against
/// the shipped kernel at the pinned shapes for T in {1, 2, 4, 8, 32, 128,
/// 512}; ~43% faster per layer at T=512 prefill (0.48 ms vs 0.83 ms), with no
/// decode regression at T=1/2.
private let qwen35GatedDeltaRowsKernel: MLXFast.MLXFastKernel? = {
    let source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        constexpr int rows_per_sg = R;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_base = thread_position_in_grid.y * rows_per_sg;

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        float state[rows_per_sg][n_per_t];
        for (int r = 0; r < rows_per_sg; ++r) {
          auto i_state = state_in + (n * Dv + dv_base + r) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[r][i] = static_cast<float>(i_state[s_idx]);
          }
        }

        for (int t = 0; t < T; ++t) {
          float kv_mem[rows_per_sg];
          for (int r = 0; r < rows_per_sg; ++r) {
            float acc = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[r][i] = state[r][i] * g_[hv_idx];
              acc += state[r][i] * k_[s_idx];
            }
            kv_mem[r] = acc;
          }
          for (int r = 0; r < rows_per_sg; ++r) {
            kv_mem[r] = simd_sum(kv_mem[r]);
          }
          float delta[rows_per_sg];
          for (int r = 0; r < rows_per_sg; ++r) {
            delta[r] = (v_[dv_base + r] - kv_mem[r]) * beta_[hv_idx];
          }
          float out_acc[rows_per_sg];
          for (int r = 0; r < rows_per_sg; ++r) {
            float o = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[r][i] = state[r][i] + k_[s_idx] * delta[r];
              o += state[r][i] * q_[s_idx];
            }
            out_acc[r] = o;
          }
          for (int r = 0; r < rows_per_sg; ++r) {
            out_acc[r] = simd_sum(out_acc[r]);
          }
          if (thread_index_in_simdgroup == 0) {
            for (int r = 0; r < rows_per_sg; ++r) {
              y[dv_base + r] = static_cast<InT>(out_acc[r]);
            }
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int r = 0; r < rows_per_sg; ++r) {
          auto o_state = state_out + (n * Dv + dv_base + r) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[r][i]);
          }
        }
    """
    return MLXFast.metalKernel(
        name: "qwen35_gated_delta_step_rows",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: source
    )
}()

/// Largest simdgroup row count dividing `Dv`, preferring the measured optimum
/// (8). Head dims that no candidate divides fall back to the shipped mapping.
private func qwen35GatedDeltaRowsPerSimdgroup(Dv: Int) -> Int {
    [8, 4, 2, 1].first { Dv % $0 == 0 } ?? 1
}

/// Unmasked GDN recurrence over the local row-interleaved kernel. Returns nil
/// when the shape is outside the clone's support so the caller can fall back
/// to the shared `gatedDeltaUpdate`. The g/beta math replicates MLXLLM's
/// `gatedDeltaUpdate` verbatim: fp32 sigmoid beta, fp32 decay via
/// `exp(-exp(aLog) * softplus(a + dtBias))`, fp32 recurrent state.
func qwen35GatedDeltaRecurrence(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray?
) -> (MLXArray, MLXArray)? {
    guard let kernel = qwen35GatedDeltaRowsKernel else { return nil }

    let B = q.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    guard Dk % 32 == 0 else { return nil }
    let rows = qwen35GatedDeltaRowsPerSimdgroup(Dv: Dv)
    guard Dv % rows == 0 else { return nil }

    let beta = sigmoid(b).asType(.float32)
    let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    var recurrenceState = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if recurrenceState.dtype != .float32 {
        recurrenceState = recurrenceState.asType(.float32)
    }

    let outputs = kernel(
        [q, k, v, decay, beta, recurrenceState, MLXArray(T)],
        template: [
            ("InT", q.dtype),
            ("StT", recurrenceState.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("R", rows),
        ] as [(String, any KernelTemplateArg)],
        grid: (32, Dv / rows, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], recurrenceState.shape],
        outputDTypes: [q.dtype, recurrenceState.dtype]
    )
    return (outputs[0], outputs[1])
}

/// Recurrence entry for the fast engine: the local row-interleaved kernel for
/// unmasked runs, the shared `MLXLLM.gatedDeltaUpdate` for padded batches and
/// for shapes the local clone does not support. Both forms are bit-identical.
private func recurrence(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray?,
    mask: MLXArray?
) -> (MLXArray, MLXArray) {
    if mask == nil, let local = qwen35GatedDeltaRecurrence(
        q: q, k: k, v: v, a: a, b: b,
        aLog: aLog, dtBias: dtBias, state: state
    ) {
        return local
    }
    return gatedDeltaUpdate(
        q: q, k: k, v: v, a: a, b: b,
        aLog: aLog, dtBias: dtBias, state: state, mask: mask
    )
}

/// Qwen35 Gated DeltaNet scaffold copied from pinned `Qwen35GatedDeltaNet`.
///
/// The recurrence has two bit-identical forms. Unmasked runs (prefill and
/// single-sequence decode) take `qwen35GatedDeltaRecurrence` above — a local
/// row-interleaved clone of the vendored kernel, ~43% faster per layer at
/// T=512 prefill. Masked runs (padded batches) keep the shared
/// `MLXLLM.gatedDeltaUpdate`, whose masked variant is untouched. Both compute
/// beta and decay `g` in float32 and preserve float32 recurrent state, as
/// documented in pinned `GatedDelta.swift`; the local clone's g/beta math is
/// copied from there verbatim.
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

        let (updated, nextRecurrentState) = recurrence(
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
