import Foundation
import MLX
import MLXFastCore
@testable import MLXFastModel
import Testing

@Test
func mlxTensorBridgeMapsSafetensorsDTypes() {
    #expect(MLXArrayTensorBridge.mlxDType(for: .bool) == .bool)
    #expect(MLXArrayTensorBridge.mlxDType(for: .u8) == .uint8)
    #expect(MLXArrayTensorBridge.mlxDType(for: .u16) == .uint16)
    #expect(MLXArrayTensorBridge.mlxDType(for: .u32) == .uint32)
    #expect(MLXArrayTensorBridge.mlxDType(for: .u64) == .uint64)
    #expect(MLXArrayTensorBridge.mlxDType(for: .i8) == .int8)
    #expect(MLXArrayTensorBridge.mlxDType(for: .i16) == .int16)
    #expect(MLXArrayTensorBridge.mlxDType(for: .i32) == .int32)
    #expect(MLXArrayTensorBridge.mlxDType(for: .i64) == .int64)
    #expect(MLXArrayTensorBridge.mlxDType(for: .f16) == .float16)
    #expect(MLXArrayTensorBridge.mlxDType(for: .bf16) == .bfloat16)
    #expect(MLXArrayTensorBridge.mlxDType(for: .f32) == .float32)
    #expect(MLXArrayTensorBridge.mlxDType(for: .f64) == .float64)
}

@Test
func mlxTensorBridgeCreatesArrayWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let tensor = try MaterializedTensor(
        name: "dense.weight",
        dtype: .f32,
        shape: [2, 2],
        bytes: float32Bytes([1, 2, 3, 4])
    )

    let array = try MLXArrayTensorBridge().makeArray(from: tensor)

    #expect(array.shape == [2, 2])
    #expect(array.dtype == .float32)
}

private func float32Bytes(_ values: [Float]) -> Data {
    var data = Data()
    for value in values {
        var bits = value.bitPattern.littleEndian
        data.append(Data(bytes: &bits, count: 4))
    }
    return data
}
