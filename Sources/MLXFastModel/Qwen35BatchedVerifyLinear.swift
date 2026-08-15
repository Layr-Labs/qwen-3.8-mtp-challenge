// Copyright © 2026 EigenLabs. All rights reserved.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXFast

/// Batched multi-row verify GEMM for 4-bit affine quantized weights.
///
/// The Qwen 3.8 MTP verify pass pushes T = draftCount + 1 rows (2...9)
/// through every quantized projection of the target tower. On the ranked M5
/// box the stock dispatch runs the `qmv_fast` kernel with `grid.x = T`, so
/// each row's threadgroup independently re-reads the same packed weight
/// stream: weight DRAM traffic scales with T while a single read would serve
/// every row. This port keeps the exact per-row arithmetic of `qmv_fast`
/// (same lane-to-k mapping, same per-lane FMA order, same `simd_sum`
/// reduction) but loops all M rows inside one threadgroup, so each packed
/// weight word is fetched once and reused from cache for every row.
/// Per-row results are therefore bit-identical to the stock M=1 kernel,
/// which is what the exact-token fidelity gate requires: the serial control
/// leg computes its logits with the same per-row math on the same build.
///
/// The kernel only engages for the speculative verify shape (2...9 rows);
/// serial decode (M=1), drafting, and the 512-row seed prefill fall back to
/// the stock path untouched.
public final class Qwen35BatchedVerifyLinear: QuantizedLinear {

    public static let defaultMaxRows = 9

    /// `MLXFAST_QWEN_MTP_BATCHED_VERIFY_MAX_ROWS` (default 0 = inert; the
    /// batched path measured neutral-to-negative locally and carries M5
    /// fast-math contraction risk against the exact-token gate). Kept in-tree
    /// for diagnosis; set 2...9 to engage.
    public static let maxRows: Int = {
        if let raw = ProcessInfo.processInfo.environment[
            "MLXFAST_QWEN_MTP_BATCHED_VERIFY_MAX_ROWS"],
            let parsed = Int(raw), parsed >= 0 {
            return parsed
        }
        return 0
    }()

    private nonisolated(unsafe) static var installOnceDone = false
    private static let installLock = NSLock()

    /// Idempotent install for the in-worker load path: swaps eligible
    /// `QuantizedLinear` leaves and registers the packed-QKV bridge hook.
    /// Safe to call from every `Qwen36MTPBlockSession.init`; the worker builds
    /// a warmup session and a real one over the same loaded module.
    public static func installOnce(on module: Module) {
        installLock.lock()
        defer { installLock.unlock() }
        guard !installOnceDone else { return }
        installOnceDone = true
        let swapped = install(on: module)
        if swapped == 0 {
            fputs(
                "mlxfast-worker: batched verify install found no eligible "
                    + "linears; staying on the stock quantizedMM path\n",
                stderr)
        }
        Qwen35BatchedVerifyBridge.quantizedMMHook = {
            (x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
                groupSize: Int, bits: Int) -> MLXArray? in
            batchedQuantizedMM(
                x,
                weight: weight,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
        }
        fputs(
            "mlxfast-worker: batched verify kernel installed on \(swapped) "
                + "eligible linears (max_rows=\(maxRows))\n",
            stderr)
    }

    private let maxRowsEffective: Int
    private lazy var kernelsByDType: [DType: MLXFast.MLXFastKernel] = [
        .bfloat16: Self.makeKernel(groupSize: groupSize, bits: bits),
        .float16: Self.makeKernel(groupSize: groupSize, bits: bits),
    ]

    public init(_ other: QuantizedLinear, maxRows: Int? = nil) {
        self.maxRowsEffective = maxRows ?? Self.maxRows
        super.init(
            weight: other.weight,
            bias: other.bias,
            scales: other.scales,
            biases: other.biases,
            groupSize: other.groupSize,
            bits: other.bits,
            mode: other.mode
        )
        self.freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y =
            Self.batchedVerifyQuantizedMM(
                x,
                weight: weight,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                maxRows: maxRowsEffective,
                kernelsByDType: kernelsByDType
            ) ?? fallback(x)
        if let bias {
            y = y + bias
        }
        return y
    }

    private func fallback(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    // MARK: - Eligibility

    public static func isEligible(_ linear: QuantizedLinear) -> Bool {
        guard !(linear is Qwen35BatchedVerifyLinear) else { return false }
        guard maxRows >= 2 else { return false }
        guard linear.bits == 4 else { return false }
        guard linear.mode == .affine else { return false }
        guard linear.biases != nil else { return false }
        guard [32, 64, 128].contains(linear.groupSize) else { return false }

        let outputDimensions = linear.weight.shape[0]
        let packedK = linear.weight.shape[1]
        let inputDimensions = packedK * (32 / linear.bits)
        guard outputDimensions % 8 == 0 else { return false }
        guard inputDimensions % 512 == 0 else { return false }
        guard inputDimensions % linear.groupSize == 0 else { return false }
        return true
    }

    // MARK: - Install

    /// Replace compatible 4-bit affine `QuantizedLinear` leaves with the
    /// batched verify variant. Shape guarded: every other shape falls back
    /// to the stock `quantizedMM` path.
    @discardableResult
    public static func install(on model: Module) -> Int {
        guard maxRows >= 2 else { return 0 }
        let updates =
            model
            .leafModules()
            .flattened()
            .compactMap { path, module -> (String, Module)? in
                guard let linear = module as? QuantizedLinear else { return nil }
                guard isEligible(linear) else { return nil }
                return (path, Qwen35BatchedVerifyLinear(linear))
            }

        guard !updates.isEmpty else { return 0 }
        model.update(modules: ModuleChildren.unflattened(updates))
        return updates.count
    }

    // MARK: - Batched quantized MM

    /// Runs the batched verify kernel when the shape qualifies; returns nil
    /// otherwise so callers fall back to the stock path.
    public static func batchedQuantizedMM(
        _ x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) -> MLXArray? {
        batchedVerifyQuantizedMM(
            x,
            weight: weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            maxRows: maxRows,
            kernelsByDType: nil
        )
    }

    private static func batchedVerifyQuantizedMM(
        _ x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        maxRows: Int,
        kernelsByDType: [DType: MLXFast.MLXFastKernel]?
    ) -> MLXArray? {
        guard maxRows >= 2, bits == 4, mode == .affine else { return nil }
        guard let biases else { return nil }
        guard [32, 64, 128].contains(groupSize) else { return nil }
        guard x.dtype == .bfloat16 || x.dtype == .float16 else { return nil }
        guard let inputDimensions = x.shape.last else { return nil }

        let rowCount = x.shape.dropLast().reduce(1, *)
        guard rowCount >= 2, rowCount <= maxRows else { return nil }

        let outputDimensions = weight.shape[0]
        guard outputDimensions % 8 == 0 else { return nil }
        guard inputDimensions == weight.shape[1] * (32 / bits) else { return nil }
        guard inputDimensions % 512 == 0 else { return nil }
        guard inputDimensions % groupSize == 0 else { return nil }

        guard let kernel = kernelsByDType?[x.dtype]
            ?? lazyKernel(groupSize: groupSize, bits: bits, dtype: x.dtype)
        else { return nil }

        let x2 = x.reshaped([rowCount, inputDimensions]).contiguous()
        let y = kernel(
            [
                x2, weight, scales, biases,
                MLXArray(inputDimensions), MLXArray(outputDimensions),
            ],
            template: [("InT", x.dtype), ("GS", groupSize), ("M", rowCount)],
            grid: (64, outputDimensions / 8, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[rowCount, outputDimensions]],
            outputDTypes: [x.dtype]
        )[0]
        return y.reshaped(Array(x.shape.dropLast()) + [outputDimensions])
    }

    nonisolated(unsafe) private static var kernelCache: [String: MLXFast.MLXFastKernel] = [:]
    private static let kernelCacheLock = NSLock()

    private static func lazyKernel(
        groupSize: Int, bits: Int, dtype: DType
    ) -> MLXFast.MLXFastKernel? {
        let key = "\(groupSize)_\(bits)_\(dtype)"
        kernelCacheLock.lock()
        defer { kernelCacheLock.unlock() }
        if let cached = kernelCache[key] { return cached }
        let kernel = makeKernel(groupSize: groupSize, bits: bits)
        kernelCache[key] = kernel
        return kernel
    }

    private static func makeKernel(groupSize: Int, bits: Int) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "qwen35_batched_verify_qmv_fast_\(bits)bit_gs\(groupSize)",
            inputNames: ["x", "w_q", "scales", "biases", "K_size", "N_size"],
            outputNames: ["y"],
            source: kernelSource(),
            header: kernelHeader()
        )
    }

    // MARK: - Kernel source

    /// Helpers copied verbatim (4-bit branches only) from the vendored MLX
    /// `qmv_fast` infrastructure (`mlx-generated/quantized.cpp`) so the
    /// per-row arithmetic matches the stock kernel statement for statement.
    private static func kernelHeader() -> String {
        """
        using namespace metal;

        template <typename T, typename U, int values_per_thread>
        METAL_FUNC U load_vector_4(
            const device T* x,
            thread U* x_thread) {
          U sum = 0;
          for (int i = 0; i < values_per_thread; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        template <typename U, int values_per_thread>
        METAL_FUNC U qdot_4(
            const device uint8_t* w,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum) {
          U accum = 0;
          const device uint16_t* ws = (const device uint16_t*)w;
          for (int i = 0; i < (values_per_thread / 4); i++) {
            accum +=
                (x_thread[4 * i] * (ws[i] & 0x000f) +
                 x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                 x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                 x_thread[4 * i + 3] * (ws[i] & 0xf000));
          }
          return scale * accum + sum * bias;
        }
        """
    }

    /// Batched form of `qmv_fast_impl` (4-bit affine): identical thread
    /// mapping (64 threads = 2 simdgroups x 4 output rows per threadgroup,
    /// lane-to-k layout `simd_lid * 16` per 512-value block) with an M-row
    /// loop sharing each packed weight fetch.
    private static func kernelSource() -> String {
        """
        uint tid = thread_position_in_threadgroup.x;
        uint simd_gid = tid / 32;
        uint simd_lid = tid % 32;
        uint tg_y = threadgroup_position_in_grid.y;

        constexpr int packs_per_thread = 2;
        constexpr int num_simdgroups = 2;
        constexpr int results_per_simdgroup = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int scale_step_per_thread = GS / values_per_thread;
        constexpr int bytes_per_pack = 4;

        const int in_vec_size = int(K_size);
        const int out_vec_size = int(N_size);
        const int in_vec_size_w = in_vec_size * bytes_per_pack / 8;
        const int in_vec_size_g = in_vec_size / GS;

        const int out_row = int(tg_y) * (num_simdgroups * results_per_simdgroup)
            + int(simd_gid) * results_per_simdgroup;

        const device uint8_t* ws = (const device uint8_t*)w_q;
        ws += out_row * in_vec_size_w + int(simd_lid) * packs_per_thread * bytes_per_pack;
        const device InT* scales_p =
            scales + out_row * in_vec_size_g + int(simd_lid) / scale_step_per_thread;
        const device InT* biases_p =
            biases + out_row * in_vec_size_g + int(simd_lid) / scale_step_per_thread;

        typedef float U;

        thread U x_thread[values_per_thread];
        U result[M][results_per_simdgroup];
        for (int m = 0; m < M; ++m) {
            for (int r = 0; r < results_per_simdgroup; ++r) {
                result[m][r] = 0;
            }
        }

        for (int k = 0; k < in_vec_size; k += block_size) {
            const device InT* x_p = x + int(simd_lid) * values_per_thread + k;
            for (int m = 0; m < M; ++m) {
                U sum = load_vector_4<InT, U, values_per_thread>(x_p, x_thread);
                for (int r = 0; r < results_per_simdgroup; ++r) {
                    const device uint8_t* wl = ws + r * in_vec_size_w;
                    U s = (scales_p + r * in_vec_size_g)[0];
                    U b = (biases_p + r * in_vec_size_g)[0];
                    result[m][r] += qdot_4<U, values_per_thread>(wl, x_thread, s, b, sum);
                }
                x_p += in_vec_size;
            }
            ws += block_size * bytes_per_pack / 8;
            scales_p += block_size / GS;
            biases_p += block_size / GS;
        }

        for (int m = 0; m < M; ++m) {
            for (int r = 0; r < results_per_simdgroup; ++r) {
                U value = simd_sum(result[m][r]);
                if (simd_lid == 0) {
                    y[(m * out_vec_size) + out_row + r] = static_cast<InT>(value);
                }
            }
        }
        """
    }
}
