import Foundation
import MLX
import MLXFastCore

public protocol MLXTensorBridge {
    associatedtype Array

    func makeArray(from tensor: MaterializedTensor) throws -> Array
}

public struct MLXArrayTensorBridge: MLXTensorBridge {
    public typealias Array = MLXArray

    public init() {}

    public func makeArray(from tensor: MaterializedTensor) throws -> MLXArray {
        MLXArray(tensor.bytes, tensor.shape, dtype: Self.mlxDType(for: tensor.dtype))
    }

    public static func mlxDType(for dtype: TensorDType) -> DType {
        dtype.mlxDType
    }
}

private extension TensorDType {
    var mlxDType: DType {
        switch self {
        case .bool:
            return .bool
        case .u8:
            return .uint8
        case .i8:
            return .int8
        case .i16:
            return .int16
        case .u16:
            return .uint16
        case .i32:
            return .int32
        case .u32:
            return .uint32
        case .i64:
            return .int64
        case .u64:
            return .uint64
        case .f16:
            return .float16
        case .bf16:
            return .bfloat16
        case .f32:
            return .float32
        case .f64:
            return .float64
        }
    }
}
