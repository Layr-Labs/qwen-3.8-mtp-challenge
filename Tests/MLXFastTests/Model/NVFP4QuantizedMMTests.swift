import Foundation
import MLX
@testable import MLXFastModel
import Testing

@Suite(.serialized)
struct NVFP4QuantizedMMTests {
    @Test
    func nvfp4DatatypeHelpersStayInAOTAndJITSources() throws {
        let kernelRoot =
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels"
        let generatedRoot =
            "Vendor/mlx-swift/Source/Cmlx/mlx-generated"
        let fp4 = try String(contentsOfFile: "\(kernelRoot)/fp4.h", encoding: .utf8)
        let fp8 = try String(contentsOfFile: "\(kernelRoot)/fp8.h", encoding: .utf8)
        let fp4Body = fp4.components(separatedBy: "\n").dropFirst(2)
            .joined(separator: "\n")
        let fp8Body = fp8.components(separatedBy: "\n").dropFirst(2)
            .joined(separator: "\n")

        #expect(fp4Body.contains("struct fp4_e2m1"))
        #expect(fp8Body.contains("struct fp8_e4m3"))
        #expect(fp8Body.contains("struct fp8_e8m0"))

        for path in [
            "\(kernelRoot)/fp_quantized.h",
            "\(kernelRoot)/fp_quantized_nax.h",
        ] {
            let aot = try String(contentsOfFile: path, encoding: .utf8)
            #expect(aot.contains(#"#include "mlx/backend/metal/kernels/fp4.h""#))
            #expect(aot.contains(#"#include "mlx/backend/metal/kernels/fp8.h""#))
        }

        for path in [
            "\(generatedRoot)/fp_quantized.cpp",
            "\(generatedRoot)/fp_quantized_nax.cpp",
        ] {
            let jit = try String(contentsOfFile: path, encoding: .utf8)
            #expect(jit.contains(#"Contents from "mlx/backend/metal/kernels/fp4.h""#))
            #expect(jit.contains(#"Contents from "mlx/backend/metal/kernels/fp8.h""#))
            #expect(jit.contains(fp4Body))
            #expect(jit.contains(fp8Body))
        }

        let unaryAOT = try String(
            contentsOfFile: "\(kernelRoot)/unary_ops.h",
            encoding: .utf8
        )
        let unaryJIT = try String(
            contentsOfFile: "\(generatedRoot)/unary_ops.cpp",
            encoding: .utf8
        )
        #expect(unaryAOT.contains(#"#include "mlx/backend/metal/kernels/fp8.h""#))
        #expect(unaryJIT.contains(#"Contents from "mlx/backend/metal/kernels/fp8.h""#))
        #expect(unaryJIT.contains(fp8Body))
    }

    @Test
    func nvfp4Group16SplitKUsesKernelTileAlignment() throws {
        let source = try String(
            contentsOfFile:
                "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp",
            encoding: .utf8
        )
        #expect(source.contains("int k_align = std::max(group_size, 32);"))
        #expect(!source.contains("int k_align = group_size;"))
    }

    @Test
    func nvfp4NAXGatherFactoryUsesDeclaredTemplateNames() throws {
        let kernelRoot =
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels"
        let generatedRoot =
            "Vendor/mlx-swift/Source/Cmlx/mlx-generated"
        let declaredFunctions = [
            "fp_gather_qmm_t_nax",
            "fp_gather_qmm_n_nax",
        ]
        for path in [
            "\(kernelRoot)/fp_quantized_nax.h",
            "\(generatedRoot)/fp_quantized_nax.cpp",
        ] {
            let source = try String(contentsOfFile: path, encoding: .utf8)
            for function in declaredFunctions {
                #expect(source.contains("void \(function)("))
            }
        }

        let factory = try String(
            contentsOfFile:
                "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp",
            encoding: .utf8
        )
        #expect(factory.contains(#""gather_qmm_t_nax","#))
        #expect(factory.contains(#""gather_qmm_n_nax","#))
        #expect(!factory.contains(#""gather_qmm_t_nax_","#))
        #expect(!factory.contains(#""gather_qmm_n_nax_","#))
    }

    @Test
    func nvfp4PackedNibblesDecodeLowNibbleFirstWhenRuntimeTestsAreEnabled() {
        guard nvfp4RuntimeTestsEnabled else { return }

        let packed = MLXArray(
            packNVFP4Nibbles(Array(0...15).map(UInt8.init)),
            [1, 2]
        )
        let scales = MLXArray([UInt8(0x38)], [1, 1]) // E4M3 1.0
        let decoded = dequantized(
            packed,
            scales: scales,
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            dtype: .float32
        )

        #expect(
            decoded.asArray(Float.self) == [
                0, 0.5, 1, 1.5, 2, 3, 4, 6,
                0, -0.5, -1, -1.5, -2, -3, -4, -6,
            ]
        )
    }

    @Test
    func nvfp4U8E4M3ScalesDecodePerGroupWhenRuntimeTestsAreEnabled() {
        guard nvfp4RuntimeTestsEnabled else { return }

        // FP4 code 0x2 is exactly 1.0. E4M3 scale bytes 0x30 and 0x40
        // are 0.5 and 2.0, respectively.
        let packed = MLXArray(
            Array(repeating: UInt32(0x2222_2222), count: 4),
            [1, 4]
        )
        let scales = MLXArray([UInt8(0x30), UInt8(0x40)], [1, 2])
        let decoded = dequantized(
            packed,
            scales: scales,
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            dtype: .float32
        )

        #expect(
            decoded.asArray(Float.self)
                == Array(repeating: Float(0.5), count: 16)
                + Array(repeating: Float(2), count: 16)
        )
    }

    @Test
    func nvfp4Group16SplitKMatchesNonSplitReferenceWhenRuntimeTestsAreEnabled() {
        guard nvfp4RuntimeTestsEnabled else { return }

        let m = 32
        let n = 128
        let k = 64
        let weight = deterministicNVFP4Source(shape: [n, k], salt: 3)
        let x = deterministicNVFP4Source(shape: [m, k], salt: 11)
        let (packedWeight, scales, biases) = quantized(
            weight,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        )
        #expect(packedWeight.shape == [n, k / 8])
        #expect(scales.shape == [n, k / 16])
        #expect(biases == nil)

        let splitK = quantizedMM(
            x,
            packedWeight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        )
        let nonSplit = nvfp4NonSplitReference(
            x: x,
            packedWeight: packedWeight,
            scales: scales
        )

        expectNVFP4Close(
            splitK,
            nonSplit,
            rtol: 1e-3,
            atol: 1e-3,
            context:
                "NVFP4 M=32 N=128 K=64 group-16 split-K versus batched non-split qmm"
        )
    }

    @Test
    func affineGroup32SplitKRemainsUnchangedWhenRuntimeTestsAreEnabled() throws {
        guard nvfp4RuntimeTestsEnabled else { return }

        let m = 32
        let n = 128
        let k = 64
        let weight = deterministicNVFP4Source(shape: [n, k], salt: 7)
        let x = deterministicNVFP4Source(shape: [m, k], salt: 19)
        let (packedWeight, scales, biases) = quantized(
            weight,
            groupSize: 32,
            bits: 4,
            mode: .affine
        )
        let affineBiases = try #require(biases)

        let splitK = quantizedMM(
            x,
            packedWeight,
            scales: scales,
            biases: affineBiases,
            transpose: true,
            groupSize: 32,
            bits: 4,
            mode: .affine
        )
        let nonSplit = quantizedMM(
            stacked([x, x]),
            stacked([packedWeight, packedWeight]),
            scales: stacked([scales, scales]),
            biases: stacked([affineBiases, affineBiases]),
            transpose: true,
            groupSize: 32,
            bits: 4,
            mode: .affine
        )[0]

        expectNVFP4Close(
            splitK,
            nonSplit,
            rtol: 1e-3,
            atol: 1e-3,
            context: "affine M=32 N=128 K=64 group-32 split-K"
        )
    }

    @Test
    func nvfp4ActualSharedProjectionShapesMatchNonSplitReferenceWhenRuntimeTestsAreEnabled() {
        guard nvfp4RuntimeTestsEnabled else { return }
        defer { Memory.clearCache() }

        let projectionShapes = [
            (
                label: "shared gate/up",
                outputFeatures: LagunaConstants.sharedExpertIntermediateSize,
                inputFeatures: LagunaConstants.hiddenSize
            ),
            (
                label: "shared down",
                outputFeatures: LagunaConstants.hiddenSize,
                inputFeatures: LagunaConstants.sharedExpertIntermediateSize
            ),
        ]

        for (index, projection) in projectionShapes.enumerated() {
            let x = deterministicNVFP4Source(
                shape: [32, projection.inputFeatures],
                salt: 23 + index
            )
            let weight = deterministicNVFP4Source(
                shape: [projection.outputFeatures, projection.inputFeatures],
                salt: 41 + index
            )
            let (packedWeight, scales, biases) = quantized(
                weight,
                groupSize: LagunaConstants.quantizationGroupSize,
                bits: LagunaConstants.quantizationBits,
                mode: .nvfp4
            )
            #expect(biases == nil)

            let splitK = quantizedMM(
                x,
                packedWeight,
                scales: scales,
                biases: nil,
                transpose: true,
                groupSize: LagunaConstants.quantizationGroupSize,
                bits: LagunaConstants.quantizationBits,
                mode: .nvfp4
            )
            let nonSplit = nvfp4NonSplitReference(
                x: x,
                packedWeight: packedWeight,
                scales: scales
            )

            expectNVFP4Close(
                splitK,
                nonSplit,
                rtol: 2e-3,
                atol: 2e-3,
                context:
                    "\(projection.label) M=32 N=\(projection.outputFeatures) K=\(projection.inputFeatures)"
            )
        }
    }

    @Test
    func nvfp4GatherRoutedDecodeAndPrefillShapesMatchNonGatherReferenceWhenRuntimeTestsAreEnabled() {
        guard nvfp4RuntimeTestsEnabled else { return }
        defer { Memory.clearCache() }

        expectNVFP4GatherMatchesReference(
            label: "routed gate/up decode",
            expertCount: 4,
            outputFeatures: LagunaConstants.moeIntermediateSize,
            inputFeatures: LagunaConstants.hiddenSize,
            rowsPerRoute: 1,
            expertIndices: [3, 0, 2, 1, 3, 1, 0, 2],
            sortedIndices: false,
            salt: 59
        )

        expectNVFP4GatherMatchesReference(
            label: "routed gate/up multi-row prefill",
            expertCount: 4,
            outputFeatures: LagunaConstants.moeIntermediateSize,
            inputFeatures: LagunaConstants.hiddenSize,
            rowsPerRoute: 32,
            expertIndices: [0, 3, 1, 2],
            sortedIndices: false,
            salt: 67
        )

        // A representative sorted prefill subset. Sixteen one-row routes over
        // four experts satisfies the same B/E >= 4 dispatch condition as the
        // full 512-token, top-8 Laguna prefill without allocating all 256
        // experts in this regression test.
        expectNVFP4GatherMatchesReference(
            label: "routed down sorted prefill",
            expertCount: 4,
            outputFeatures: LagunaConstants.hiddenSize,
            inputFeatures: LagunaConstants.moeIntermediateSize,
            rowsPerRoute: 1,
            expertIndices: [
                0, 0, 0, 0,
                1, 1, 1, 1,
                2, 2, 2, 2,
                3, 3, 3, 3,
            ],
            sortedIndices: true,
            salt: 71
        )
    }
}

private let nvfp4RuntimeTestsEnabled =
    ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"

private func packNVFP4Nibbles(_ nibbles: [UInt8]) -> [UInt32] {
    precondition(nibbles.count.isMultiple(of: 8))
    return stride(from: 0, to: nibbles.count, by: 8).map { start in
        var word = UInt32(0)
        for offset in 0..<8 {
            word |= UInt32(nibbles[start + offset] & 0x0f) << (offset * 4)
        }
        return word
    }
}

private func deterministicNVFP4Source(shape: [Int], salt: Int) -> MLXArray {
    let count = shape.reduce(1, *)
    let values = (0..<count).map { index -> Float in
        let centered = ((index * 17 + salt) % 31) - 15
        return Float(centered) / 16
    }
    return MLXArray(values, shape)
}

private func nvfp4NonSplitReference(
    x: MLXArray,
    packedWeight: MLXArray,
    scales: MLXArray
) -> MLXArray {
    quantizedMM(
        stacked([x, x]),
        stacked([packedWeight, packedWeight]),
        scales: stacked([scales, scales]),
        biases: nil,
        transpose: true,
        groupSize: 16,
        bits: 4,
        mode: .nvfp4
    )[0]
}

private func expectNVFP4GatherMatchesReference(
    label: String,
    expertCount: Int,
    outputFeatures: Int,
    inputFeatures: Int,
    rowsPerRoute: Int,
    expertIndices: [Int],
    sortedIndices: Bool,
    salt: Int
) {
    let weight = deterministicNVFP4Source(
        shape: [expertCount, outputFeatures, inputFeatures],
        salt: salt
    )
    let (packedWeight, scales, biases) = quantized(
        weight,
        groupSize: 16,
        bits: 4,
        mode: .nvfp4
    )
    precondition(biases == nil)

    let x = deterministicNVFP4Source(
        shape: [expertIndices.count, rowsPerRoute, inputFeatures],
        salt: salt + 13
    )
    let indices = MLXArray(expertIndices.map(UInt32.init), [expertIndices.count])
    let gathered = gatherQuantizedMM(
        x,
        packedWeight,
        scales: scales,
        biases: nil,
        rhsIndices: indices,
        transpose: true,
        groupSize: 16,
        bits: 4,
        mode: .nvfp4,
        sortedIndices: sortedIndices
    )

    let selectedWeights = stacked(expertIndices.map { packedWeight[$0] })
    let selectedScales = stacked(expertIndices.map { scales[$0] })
    let nonGather = quantizedMM(
        x,
        selectedWeights,
        scales: selectedScales,
        biases: nil,
        transpose: true,
        groupSize: 16,
        bits: 4,
        mode: .nvfp4
    )

    expectNVFP4Close(
        gathered,
        nonGather,
        rtol: 2e-3,
        atol: 2e-3,
        context:
            "\(label) B=\(expertIndices.count) M=\(rowsPerRoute) N=\(outputFeatures) K=\(inputFeatures)"
    )
}

private func expectNVFP4Close(
    _ actual: MLXArray,
    _ expected: MLXArray,
    rtol: Double,
    atol: Double,
    context: String
) {
    #expect(actual.shape == expected.shape, Comment(rawValue: context))
    let finite = isFinite(actual).all()
    let close = actual.allClose(expected, rtol: rtol, atol: atol)
    let maxError = abs(actual.asType(.float32) - expected.asType(.float32)).max()
    eval(finite, close, maxError)

    let maxErrorValue = maxError.item(Float.self)
    #expect(
        finite.item(Bool.self),
        Comment(rawValue: "\(context): output contains non-finite values")
    )
    #expect(
        close.item(Bool.self),
        Comment(rawValue: "\(context): max absolute error \(maxErrorValue)")
    )
}
