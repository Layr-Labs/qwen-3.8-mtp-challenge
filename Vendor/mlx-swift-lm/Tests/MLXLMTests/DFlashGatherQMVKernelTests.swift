// Copyright © 2026 Apple Inc.

import Darwin
import MLX
import MLXRandom
import Testing

@Suite("DFlash gather qmv kernel")
struct DFlashGatherQMVKernelTests {
    @Test func quantizedGatherQMVFastTailMatchesDequantizedGatherMM() {
        MLXRandom.seed(45)

        let expertCount = 4
        let rowsPerExpert = 16
        let rowCount = expertCount * rowsPerExpert
        let inputDimensions = 2816
        let outputDimensions = 64
        let groupSize = 64
        let bits = 4

        let x = MLXRandom.normal([rowCount, 1, inputDimensions]).asType(.bfloat16)
        let w = MLXRandom.normal([expertCount, outputDimensions, inputDimensions]).asType(.bfloat16)
        let (wq, scales, biases) = MLX.quantized(w, groupSize: groupSize, bits: bits)
        let indices = MLXArray(
            (0..<expertCount).flatMap { expert in
                [Int32](repeating: Int32(expert), count: rowsPerExpert)
            })

        let dequantizedWeight = MLX.dequantized(
            wq,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            dtype: .bfloat16)
        let expected = MLX.gatherMM(
            x,
            dequantizedWeight.swappedAxes(-1, -2),
            rhsIndices: indices,
            sortedIndices: true)
        let actual = MLX.gatherQuantizedMM(
            x,
            wq,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            sortedIndices: true)

        eval(expected, actual)

        #expect(allClose(actual, expected, rtol: 5e-2, atol: 5e-2).item(Bool.self))
    }

    @Test func quantizedGatherQMVFastTailMatchesGenericGatherQMV() {
        MLXRandom.seed(46)

        let expertCount = 4
        let rowsPerExpert = 16
        let rowCount = expertCount * rowsPerExpert
        let inputDimensions = 2816
        let outputDimensions = 64
        let groupSize = 64
        let bits = 4

        let x = MLXRandom.normal([rowCount, 1, inputDimensions]).asType(.bfloat16)
        let w = MLXRandom.normal([expertCount, outputDimensions, inputDimensions]).asType(.bfloat16)
        let (wq, scales, biases) = MLX.quantized(w, groupSize: groupSize, bits: bits)
        let indices = MLXArray(
            (0..<expertCount).flatMap { expert in
                [Int32](repeating: Int32(expert), count: rowsPerExpert)
            })

        let expected = withEnvironment("MLX_GATHER_QMV_FAST_TAIL_256", value: "0") {
            let result = gatherQuantized(
                x: x,
                wq: wq,
                scales: scales,
                biases: biases,
                indices: indices,
                groupSize: groupSize,
                bits: bits)
            eval(result)
            return result
        }
        let actual = withEnvironment("MLX_GATHER_QMV_FAST_TAIL_256", value: "1") {
            let result = gatherQuantized(
                x: x,
                wq: wq,
                scales: scales,
                biases: biases,
                indices: indices,
                groupSize: groupSize,
                bits: bits)
            eval(result)
            return result
        }

        #expect(allClose(actual, expected, rtol: 5e-2, atol: 5e-2).item(Bool.self))
    }

    private func gatherQuantized(
        x: MLXArray,
        wq: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        indices: MLXArray,
        groupSize: Int,
        bits: Int
    ) -> MLXArray {
        MLX.gatherQuantizedMM(
            x,
            wq,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            sortedIndices: true)
    }
}

private func withEnvironment<T>(_ name: String, value: String?, _ body: () -> T) -> T {
    let previous = getenv(name).map { String(cString: $0) }
    if let value {
        setenv(name, value, 1)
    } else {
        unsetenv(name)
    }

    let result = body()

    if let previous {
        setenv(name, previous, 1)
    } else {
        unsetenv(name)
    }
    return result
}
