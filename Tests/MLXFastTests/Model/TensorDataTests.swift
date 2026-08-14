import Foundation
import Testing
@testable import MLXFastCore

@Test
func materializedTensorDecodesSupportedValues() throws {
    let uint8 = try MaterializedTensor(
        name: "u8",
        dtype: .u8,
        shape: [3],
        bytes: Data([1, 2, 255])
    )
    #expect(try uint8.uint8Values() == [1, 2, 255])

    let ints = try MaterializedTensor(
        name: "i32",
        dtype: .i32,
        shape: [2],
        bytes: int32Bytes([7, -3])
    )
    #expect(try ints.int32Values() == [7, -3])

    let uints = try MaterializedTensor(
        name: "u32",
        dtype: .u32,
        shape: [2],
        bytes: uint32Bytes([7, UInt32.max - 2])
    )
    #expect(try uints.uint32Values() == [7, UInt32.max - 2])

    let floats = try MaterializedTensor(
        name: "f32",
        dtype: .f32,
        shape: [2],
        bytes: float32Bytes([1.25, -2.5])
    )
    #expect(try floats.float32Values() == [1.25, -2.5])
}

@Test
func materializedTensorRejectsMismatchedShapeAndBytes() throws {
    #expect(throws: MLXFastError.self) {
        _ = try MaterializedTensor(
            name: "bad",
            dtype: .i32,
            shape: [2],
            bytes: Data([1, 2, 3, 4])
        )
    }
}

private func int32Bytes(_ values: [Int32]) -> Data {
    var data = Data()
    for value in values {
        var littleEndian = value.littleEndian
        data.append(Data(bytes: &littleEndian, count: 4))
    }
    return data
}

private func uint32Bytes(_ values: [UInt32]) -> Data {
    var data = Data()
    for value in values {
        var littleEndian = value.littleEndian
        data.append(Data(bytes: &littleEndian, count: 4))
    }
    return data
}

private func float32Bytes(_ values: [Float]) -> Data {
    var data = Data()
    for value in values {
        var bits = value.bitPattern.littleEndian
        data.append(Data(bytes: &bits, count: 4))
    }
    return data
}
