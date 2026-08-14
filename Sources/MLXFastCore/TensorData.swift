import Foundation

public enum TensorDType: String, Codable, Equatable {
    case bool = "BOOL"
    case u8 = "U8"
    case i8 = "I8"
    case i16 = "I16"
    case u16 = "U16"
    case i32 = "I32"
    case u32 = "U32"
    case i64 = "I64"
    case u64 = "U64"
    case f16 = "F16"
    case bf16 = "BF16"
    case f32 = "F32"
    case f64 = "F64"

    public var byteWidth: Int {
        switch self {
        case .bool, .u8, .i8:
            return 1
        case .i16, .u16, .f16, .bf16:
            return 2
        case .i32, .u32, .f32:
            return 4
        case .i64, .u64, .f64:
            return 8
        }
    }

    public static func parse(_ raw: String) throws -> TensorDType {
        guard let dtype = TensorDType(rawValue: raw) else {
            throw MLXFastError.invalidInput("unsupported safetensors dtype \(raw)")
        }
        return dtype
    }
}

public struct MaterializedTensor: Equatable {
    public let name: String
    public let dtype: TensorDType
    public let shape: [Int]
    public let bytes: Data

    public init(name: String, dtype: TensorDType, shape: [Int], bytes: Data) throws {
        let expectedByteCount = try expectedTensorByteCount(name: name, dtype: dtype, shape: shape)
        guard expectedByteCount == bytes.count else {
            throw MLXFastError.invalidInput(
                "tensor \(name) byte count \(bytes.count) does not match dtype \(dtype.rawValue) and shape \(shape) expected \(expectedByteCount)"
            )
        }
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.bytes = bytes
    }

    public var elementCount: Int {
        shape.reduce(1, *)
    }

    public func uint8Values() throws -> [UInt8] {
        guard dtype == .u8 || dtype == .bool else {
            throw MLXFastError.invalidInput("tensor \(name) has dtype \(dtype.rawValue), not U8/BOOL")
        }
        return Array(bytes)
    }

    public func uint32Values() throws -> [UInt32] {
        guard dtype == .u32 else {
            throw MLXFastError.invalidInput("tensor \(name) has dtype \(dtype.rawValue), not U32")
        }
        return bytes.withUnsafeBytes { raw in
            (0..<elementCount).map { index in
                raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self).littleEndian
            }
        }
    }

    public func int32Values() throws -> [Int32] {
        guard dtype == .i32 else {
            throw MLXFastError.invalidInput("tensor \(name) has dtype \(dtype.rawValue), not I32")
        }
        return bytes.withUnsafeBytes { raw in
            (0..<elementCount).map { index in
                raw.loadUnaligned(fromByteOffset: index * 4, as: Int32.self).littleEndian
            }
        }
    }

    public func float32Values() throws -> [Float] {
        guard dtype == .f32 else {
            throw MLXFastError.invalidInput("tensor \(name) has dtype \(dtype.rawValue), not F32")
        }
        return bytes.withUnsafeBytes { raw in
            (0..<elementCount).map { index in
                let bits = raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self).littleEndian
                return Float(bitPattern: bits)
            }
        }
    }
}

public func expectedTensorByteCount(name: String, dtype: TensorDType, shape: [Int]) throws -> Int {
    guard shape.allSatisfy({ $0 >= 0 }) else {
        throw MLXFastError.invalidInput("tensor \(name) has negative dimension in shape \(shape)")
    }
    var elements = 1
    for dimension in shape {
        let result = elements.multipliedReportingOverflow(by: dimension)
        guard !result.overflow else {
            throw MLXFastError.invalidInput("tensor \(name) shape \(shape) element count overflows Int")
        }
        elements = result.partialValue
    }
    let bytes = elements.multipliedReportingOverflow(by: dtype.byteWidth)
    guard !bytes.overflow else {
        throw MLXFastError.invalidInput("tensor \(name) byte count overflows Int for dtype \(dtype.rawValue) and shape \(shape)")
    }
    return bytes.partialValue
}

public func materializeTensor(
    name: String,
    dtype rawDType: String,
    shape: [Int],
    bytes: Data
) throws -> MaterializedTensor {
    try MaterializedTensor(
        name: name,
        dtype: TensorDType.parse(rawDType),
        shape: shape,
        bytes: bytes
    )
}
