// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Quantized linear wrapper with an M=16 verify-time matmul fast path.
///
/// DFlash verifies one speculative block with the target model. For K=16
/// verification the target's linear layers see exactly 16 flattened rows, where
/// the generic quantized matmul path leaves measurable throughput on the table.
/// This wrapper keeps normal `QuantizedLinear` behavior for every other shape.
public final class DFlashVerifyQuantizedLinear: QuantizedLinear {
    private let enableQMM: Bool
    private let qmmWeight: MLXArray
    private let qmmScales: MLXArray
    private let qmmBiases: MLXArray?

    private lazy var bigBF16Kernel = Self.makeBigKernel(
        groupSize: groupSize, bits: bits, dtypeTag: "bf16")
    private lazy var bigFP16Kernel = Self.makeBigKernel(
        groupSize: groupSize, bits: bits, dtypeTag: "fp16")
    private lazy var pipeBF16Kernel = Self.makePipeKernel(
        groupSize: groupSize, bits: bits, dtypeTag: "bf16")
    private lazy var pipeFP16Kernel = Self.makePipeKernel(
        groupSize: groupSize, bits: bits, dtypeTag: "fp16")

    public init(_ other: QuantizedLinear, enableQMM: Bool = true) {
        self.enableQMM = enableQMM
        self.qmmWeight = other.weight.contiguous()
        self.qmmScales = other.scales.contiguous()
        self.qmmBiases = other.biases?.contiguous()
        super.init(
            weight: other.weight,
            bias: other.bias,
            scales: other.scales,
            biases: other.biases,
            groupSize: other.groupSize,
            bits: other.bits,
            mode: other.mode
        )
        if let qmmBiases {
            eval(qmmWeight, qmmScales, qmmBiases)
        } else {
            eval(qmmWeight, qmmScales)
        }
        self.freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y =
            enableQMM
            ? verifyQMM(x) ?? fallback(x)
            : fallback(x)
        if let bias {
            y = y + bias
        }
        return y
    }

    public static func isEligible(_ linear: QuantizedLinear, maxOutputDimensions: Int = 100_000)
        -> Bool
    {
        guard !(linear is DFlashVerifyQuantizedLinear) else { return false }
        guard [4, 8].contains(linear.bits) else { return false }
        guard [32, 64, 128].contains(linear.groupSize) else { return false }
        guard linear.mode == .affine else { return false }
        guard linear.biases != nil else { return false }

        let outputDimensions = linear.weight.shape[0]
        let inputDimensions = linear.weight.shape[1] * (32 / linear.bits)
        guard outputDimensions < maxOutputDimensions else { return false }
        guard outputDimensions % 32 == 0 else { return false }
        guard inputDimensions % 32 == 0 else { return false }
        guard inputDimensions >= 256 else { return false }
        if Self.resolvedQMMVariant(
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions
        ) == .pipe {
            return inputDimensions % (32 * Self.pipeKParts) == 0
        }
        return true
    }

    public static func pathTag(_ path: String) -> String {
        if path.hasSuffix(".mlp.gate_proj") { return "mlp_gate" }
        if path.hasSuffix(".mlp.up_proj") { return "mlp_up" }
        if path.hasSuffix(".mlp.down_proj") { return "mlp_down" }
        if path.hasSuffix(".self_attn.q_proj") { return "attn_q" }
        if path.hasSuffix(".self_attn.k_proj") { return "attn_k" }
        if path.hasSuffix(".self_attn.v_proj") { return "attn_v" }
        if path.hasSuffix(".self_attn.o_proj") { return "attn_o" }
        if path.hasSuffix(".router.proj") { return "router" }
        return "other"
    }

    public static func includeAllows(path: String, include: String) -> Bool {
        let rawTags = include
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty }
        guard !rawTags.isEmpty else { return true }
        if rawTags.contains("all") { return true }

        var tags = Set(rawTags)
        if tags.contains("mlp") {
            tags.formUnion(["mlp_gate", "mlp_up", "mlp_down"])
        }
        if tags.contains("attn") {
            tags.formUnion(["attn_q", "attn_k", "attn_v", "attn_o"])
        }
        return tags.contains(pathTag(path))
    }

    public static func enablesCustomQMMByDefault(path: String) -> Bool {
        if pathTag(path) == "attn_o" {
            return attentionOutputQMMEnabled
        }
        return true
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

    private func verifyQMM(_ x: MLXArray) -> MLXArray? {
        guard [4, 8].contains(bits), mode == .affine else { return nil }
        guard qmmBiases != nil else { return nil }
        guard x.dtype == .bfloat16 || x.dtype == .float16 else { return nil }
        guard let inputDimensions = x.shape.last else { return nil }

        let rowCount = x.shape.dropLast().reduce(1, *)
        guard rowCount == Self.blockRows else { return nil }

        let outputDimensions = qmmWeight.shape[0]
        guard outputDimensions % 32 == 0 else { return nil }
        guard inputDimensions == qmmWeight.shape[1] * (32 / bits) else { return nil }

        let x2 = x.reshaped([Self.blockRows, inputDimensions]).contiguous()
        guard let qmmBiases else { return nil }
        let variant = Self.resolvedQMMVariant(
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions)
        if variant == .pipe {
            guard inputDimensions % (32 * Self.pipeKParts) == 0 else { return nil }
            let kernel = x.dtype == .bfloat16 ? pipeBF16Kernel : pipeFP16Kernel
            let partials = kernel(
                [
                    x2, qmmWeight, qmmScales, qmmBiases, Self.blockRows, inputDimensions,
                    outputDimensions, Self.pipeKParts,
                ],
                template: [("T", x.dtype)],
                grid: (64, outputDimensions / 32, Self.pipeKParts),
                threadGroup: (64, 1, 1),
                outputShapes: [[Self.pipeKParts, Self.blockRows, outputDimensions]],
                outputDTypes: [.float32]
            )[0]
            let y = partials.sum(axis: 0).asType(x.dtype)
            return y.reshaped(Array(x.shape.dropLast()) + [outputDimensions])
        }

        guard inputDimensions % 32 == 0 else { return nil }
        let kernel = x.dtype == .bfloat16 ? bigBF16Kernel : bigFP16Kernel
        let y = kernel(
            [x2, qmmWeight, qmmScales, qmmBiases, Self.blockRows, inputDimensions, outputDimensions],
            template: [("T", x.dtype)],
            grid: (64, outputDimensions / 32, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[Self.blockRows, outputDimensions]],
            outputDTypes: [x.dtype]
        )[0]
        return y.reshaped(Array(x.shape.dropLast()) + [outputDimensions])
    }

    private enum QMMVariant: String {
        case auto
        case pipe
        case mma2big
    }

    private static let qmmVariant: QMMVariant = {
        let raw = ProcessInfo.processInfo.environment["MLX_DFLASH_VERIFY_QMM_VARIANT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        switch raw {
        case "pipe", "mma2big_pipe":
            return .pipe
        case "big", "mma2big":
            return .mma2big
        default:
            return .auto
        }
    }()

    private static let attentionOutputQMMEnabled: Bool = {
        switch ProcessInfo.processInfo.environment["MLX_DFLASH_VERIFY_QMM_ATTN_O"]?
            .lowercased()
        {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }()

    private static let pipeKParts: Int = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_DFLASH_VERIFY_QMM_KPARTS"],
            let parsed = Int(raw),
            parsed > 0
        else {
            return 4
        }
        return parsed
    }()

    private static func resolvedQMMVariant(
        inputDimensions: Int,
        outputDimensions: Int
    ) -> QMMVariant {
        switch qmmVariant {
        case .auto:
            return inputDimensions >= 8192 || outputDimensions <= 8192 ? .pipe : .mma2big
        case .pipe:
            return .pipe
        case .mma2big:
            return .mma2big
        }
    }

    private static let blockRows = 16

    private static func makeBigKernel(
        groupSize: Int, bits: Int, dtypeTag: String
    ) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "dflash_verify_mma2big_\(bits)bit_gs\(groupSize)_\(dtypeTag)",
            inputNames: ["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
            outputNames: ["y"],
            source: bigKernelSource(groupSize: groupSize, bits: bits)
        )
    }

    private static func bigKernelSource(groupSize: Int, bits: Int) -> String {
        if bits == 8 {
            return bigKernel8BitSource(groupSize: groupSize)
        }

        return """
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = \(groupSize);

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        threadgroup T B_tile[BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        for (int k0 = 0; k0 < K; k0 += BK) {
            {
                int n_global = n0 + dq_n_a;
                int k_base = k0 + dq_k_a * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float(nib) * s + b);
                }
            }
            {
                int n_global = n0 + dq_n_b;
                int k_base = k0 + dq_k_b * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float(nib) * s + b);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {
                simdgroup_load(a_top, x + k0 + ks * BK_SUB, K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB, K);
                simdgroup_load(b_L, B_tile + ks * BK_SUB * BN + sg_n_off, BN);
                simdgroup_load(b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8, BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_matrix<T, 8, 8> c_tL_T, c_tR_T, c_bL_T, c_bR_T;
        c_tL_T.thread_elements()[0] = T(c_tL.thread_elements()[0]);
        c_tL_T.thread_elements()[1] = T(c_tL.thread_elements()[1]);
        c_tR_T.thread_elements()[0] = T(c_tR.thread_elements()[0]);
        c_tR_T.thread_elements()[1] = T(c_tR.thread_elements()[1]);
        c_bL_T.thread_elements()[0] = T(c_bL.thread_elements()[0]);
        c_bL_T.thread_elements()[1] = T(c_bL.thread_elements()[1]);
        c_bR_T.thread_elements()[0] = T(c_bR.thread_elements()[0]);
        c_bR_T.thread_elements()[1] = T(c_bR.thread_elements()[1]);
        simdgroup_store(c_tL_T, y + n0 + sg_n_off, N);
        simdgroup_store(c_tR_T, y + n0 + sg_n_off + 8, N);
        simdgroup_store(c_bL_T, y + 8 * N + n0 + sg_n_off, N);
        simdgroup_store(c_bR_T, y + 8 * N + n0 + sg_n_off + 8, N);
        """
    }

    private static func bigKernel8BitSource(groupSize: Int) -> String {
        return """
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = \(groupSize);

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_4  = K / 4;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        threadgroup T B_tile[BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        for (int k0 = 0; k0 < K; k0 += BK) {
            {
                int n_global = n0 + dq_n_a;
                int k_base = k0 + dq_k_a * 8;
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];
                for (int ki = 0; ki < 4; ++ki) {
                    B_tile[(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);
                }
                for (int ki = 0; ki < 4; ++ki) {
                    B_tile[(dq_k_a * 8 + 4 + ki) * BN + dq_n_a] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);
                }
            }
            {
                int n_global = n0 + dq_n_b;
                int k_base = k0 + dq_k_b * 8;
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];
                for (int ki = 0; ki < 4; ++ki) {
                    B_tile[(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);
                }
                for (int ki = 0; ki < 4; ++ki) {
                    B_tile[(dq_k_b * 8 + 4 + ki) * BN + dq_n_b] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {
                simdgroup_load(a_top, x + k0 + ks * BK_SUB, K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB, K);
                simdgroup_load(b_L, B_tile + ks * BK_SUB * BN + sg_n_off, BN);
                simdgroup_load(b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8, BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_matrix<T, 8, 8> c_tL_T, c_tR_T, c_bL_T, c_bR_T;
        c_tL_T.thread_elements()[0] = T(c_tL.thread_elements()[0]);
        c_tL_T.thread_elements()[1] = T(c_tL.thread_elements()[1]);
        c_tR_T.thread_elements()[0] = T(c_tR.thread_elements()[0]);
        c_tR_T.thread_elements()[1] = T(c_tR.thread_elements()[1]);
        c_bL_T.thread_elements()[0] = T(c_bL.thread_elements()[0]);
        c_bL_T.thread_elements()[1] = T(c_bL.thread_elements()[1]);
        c_bR_T.thread_elements()[0] = T(c_bR.thread_elements()[0]);
        c_bR_T.thread_elements()[1] = T(c_bR.thread_elements()[1]);
        simdgroup_store(c_tL_T, y + n0 + sg_n_off, N);
        simdgroup_store(c_tR_T, y + n0 + sg_n_off + 8, N);
        simdgroup_store(c_bL_T, y + 8 * N + n0 + sg_n_off, N);
        simdgroup_store(c_bR_T, y + 8 * N + n0 + sg_n_off + 8, N);
        """
    }

    private static func makePipeKernel(
        groupSize: Int, bits: Int, dtypeTag: String
    ) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "dflash_verify_mma2big_pipe_\(bits)bit_gs\(groupSize)_\(dtypeTag)",
            inputNames: ["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size", "K_parts"],
            outputNames: ["partials"],
            source: pipeKernelSource(groupSize: groupSize, bits: bits)
        )
    }

    private static func pipeKernelSource(groupSize: Int, bits: Int) -> String {
        if bits == 8 {
            return pipeKernel8BitSource(groupSize: groupSize)
        }

        return """
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = \(groupSize);

        uint tid       = thread_position_in_threadgroup.x;
        uint sg_id     = tid / 32;
        uint tg_n      = threadgroup_position_in_grid.y;
        uint tg_k_part = threadgroup_position_in_grid.z;

        int K = int(K_size);
        int N = int(N_size);
        int KP = int(K_parts);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int k_slice = K / KP;
        int k_begin = k_slice * int(tg_k_part);
        int k_end   = k_begin + k_slice;

        threadgroup T B_tile[2][BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        #define STAGE_B(slot, k0_stage) {{                                              \\
            {{                                                                          \\
                int n_global = n0 + dq_n_a;                                             \\
                int k_base = (k0_stage) + dq_k_a * 8;                                   \\
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];               \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);            \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);            \\
                _Pragma("unroll")                                                       \\
                for (int ki = 0; ki < 8; ++ki) {{                                       \\
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;                         \\
                    B_tile[slot][(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float(nib) * s + b); \\
                }}                                                                      \\
            }}                                                                          \\
            {{                                                                          \\
                int n_global = n0 + dq_n_b;                                             \\
                int k_base = (k0_stage) + dq_k_b * 8;                                   \\
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];               \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);            \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);            \\
                _Pragma("unroll")                                                       \\
                for (int ki = 0; ki < 8; ++ki) {{                                       \\
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;                         \\
                    B_tile[slot][(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float(nib) * s + b); \\
                }}                                                                      \\
            }}                                                                          \\
        }}

        STAGE_B(0, k_begin);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int read_slot = 0;
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            int write_slot = 1 - read_slot;
            int k0_next = k0 + BK;

            if (k0_next < k_end) {{
                STAGE_B(write_slot, k0_next);
            }}

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            read_slot = write_slot;
        }}

        int part_off = int(tg_k_part) * BM * N;
        simdgroup_store(c_tL, partials + part_off + n0 + sg_n_off,                     N);
        simdgroup_store(c_tR, partials + part_off + n0 + sg_n_off + 8,                 N);
        simdgroup_store(c_bL, partials + part_off + 8 * N + n0 + sg_n_off,             N);
        simdgroup_store(c_bR, partials + part_off + 8 * N + n0 + sg_n_off + 8,         N);

        #undef STAGE_B
        """
    }

    private static func pipeKernel8BitSource(groupSize: Int) -> String {
        return """
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = \(groupSize);

        uint tid       = thread_position_in_threadgroup.x;
        uint sg_id     = tid / 32;
        uint tg_n      = threadgroup_position_in_grid.y;
        uint tg_k_part = threadgroup_position_in_grid.z;

        int K = int(K_size);
        int N = int(N_size);
        int KP = int(K_parts);
        int K_by_4  = K / 4;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int k_slice = K / KP;
        int k_begin = k_slice * int(tg_k_part);
        int k_end   = k_begin + k_slice;

        threadgroup T B_tile[2][BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        #define STAGE_B(slot, k0_stage) {{                                                                                  \\
            {{                                                                                                              \\
                int n_global = n0 + dq_n_a;                                                                                 \\
                int k_base   = (k0_stage) + dq_k_a * 8;                                                                     \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);                                                \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);                                                \\
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];                                                       \\
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];                                                   \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_a * 8 + ki)     * BN + dq_n_a] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);         \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_a * 8 + 4 + ki) * BN + dq_n_a] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);         \\
            }}                                                                                                              \\
            {{                                                                                                              \\
                int n_global = n0 + dq_n_b;                                                                                 \\
                int k_base   = (k0_stage) + dq_k_b * 8;                                                                     \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);                                                \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);                                                \\
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];                                                       \\
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];                                                   \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_b * 8 + ki)     * BN + dq_n_b] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);         \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_b * 8 + 4 + ki) * BN + dq_n_b] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);         \\
            }}                                                                                                              \\
        }}

        STAGE_B(0, k_begin);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int read_slot = 0;
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            int write_slot = 1 - read_slot;
            int k0_next = k0 + BK;

            if (k0_next < k_end) {{
                STAGE_B(write_slot, k0_next);
            }}

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            read_slot = write_slot;
        }}

        int part_off = int(tg_k_part) * BM * N;
        simdgroup_store(c_tL, partials + part_off + n0 + sg_n_off,                     N);
        simdgroup_store(c_tR, partials + part_off + n0 + sg_n_off + 8,                 N);
        simdgroup_store(c_bL, partials + part_off + 8 * N + n0 + sg_n_off,             N);
        simdgroup_store(c_bR, partials + part_off + 8 * N + n0 + sg_n_off + 8,         N);

        #undef STAGE_B
        """
    }
}

public enum DFlashVerifyLinear {
    /// Replace compatible 4/8-bit `QuantizedLinear` leaves with
    /// `DFlashVerifyQuantizedLinear`.
    ///
    /// The replacement is shape guarded and falls back to `quantizedMM` except
    /// for the DFlash verify shape `M == 16`.
    @discardableResult
    public static func install(
        on model: Module,
        enableQMM: Bool = true,
        include: String = "all",
        maxOutputDimensions: Int = 100_000
    ) -> Int {
        let updates =
            model
            .leafModules()
            .flattened()
            .compactMap { path, module -> (String, Module)? in
            guard let linear = module as? QuantizedLinear else { return nil }
            guard DFlashVerifyQuantizedLinear.includeAllows(path: path, include: include) else {
                return nil
            }
            guard DFlashVerifyQuantizedLinear.isEligible(
                linear, maxOutputDimensions: maxOutputDimensions)
            else {
                return nil
            }
            let pathEnableQMM = enableQMM
                && DFlashVerifyQuantizedLinear.enablesCustomQMMByDefault(path: path)
            return (path, DFlashVerifyQuantizedLinear(linear, enableQMM: pathEnableQMM))
        }

        guard !updates.isEmpty else { return 0 }
        model.update(modules: ModuleChildren.unflattened(updates))
        return updates.count
    }
}
