import MLX
import MLXFastCore

/// Dense or affine-quantized matrix storage shared by the custom Qwen35
/// embedding, projection, MLP, attention, and LM-head scaffolds.
///
/// Quantized weights retain their logical `[outFeatures, inFeatures]` shape
/// separately from MLX's packed U32 storage shape.
public struct Qwen35LinearWeight {
    public let weight: MLXArray
    public let scales: MLXArray?
    public let biases: MLXArray?
    public let logicalShape: [Int]
    public let groupSize: Int
    public let bits: Int

    public init(_ weight: MLXArray) {
        self.weight = weight
        self.scales = nil
        self.biases = nil
        self.logicalShape = weight.shape
        self.groupSize = 0
        self.bits = 0
    }

    public init(
        weight: MLXArray,
        scales: MLXArray?,
        biases: MLXArray?,
        logicalShape: [Int],
        groupSize: Int,
        bits: Int
    ) throws {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.logicalShape = logicalShape
        self.groupSize = groupSize
        self.bits = bits
        try validate()
    }

    public var isQuantized: Bool {
        scales != nil
    }

    public var shape: [Int] {
        logicalShape
    }

    func validate() throws {
        guard logicalShape.count == 2,
              logicalShape[0] > 0,
              logicalShape[1] > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 linear logical shape must be positive rank-2"
            )
        }

        guard let scales else {
            guard biases == nil,
                  groupSize == 0,
                  bits == 0,
                  weight.shape == logicalShape
            else {
                throw MLXFastError.invalidInput(
                    "Qwen35 dense linear storage does not match its logical shape"
                )
            }
            return
        }

        let outputFeatures = logicalShape[0]
        let inputFeatures = logicalShape[1]
        guard weight.dtype == .uint32,
              groupSize > 0,
              inputFeatures.isMultiple(of: groupSize),
              [2, 4, 8].contains(bits),
              let biases
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 quantized linear metadata is invalid"
            )
        }
        let packedBits = try qwen35CheckedProduct(
            [inputFeatures, bits],
            label: "Qwen35 packed linear width"
        )
        guard packedBits.isMultiple(of: 32) else {
            throw MLXFastError.invalidInput(
                "Qwen35 quantized linear width is not U32-packable"
            )
        }
        let packedWidth = packedBits / 32
        let groupCount = inputFeatures / groupSize
        let companionShape = [outputFeatures, groupCount]
        guard weight.shape == [outputFeatures, packedWidth],
              scales.shape == companionShape,
              biases.shape == companionShape
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 packed weight/scales/biases shapes are inconsistent"
            )
        }
    }
}

func qwen35CheckedProduct(
    _ values: [Int],
    label: String
) throws -> Int {
    var result = 1
    for value in values {
        guard value >= 0 else {
            throw MLXFastError.invalidInput(
                "\(label) contains a negative dimension"
            )
        }
        let next = result.multipliedReportingOverflow(by: value)
        guard !next.overflow else {
            throw MLXFastError.invalidInput("\(label) overflows Int")
        }
        result = next.partialValue
    }
    return result
}

func qwen35CheckedSum(
    _ values: [Int],
    label: String
) throws -> Int {
    var result = 0
    for value in values {
        guard value >= 0 else {
            throw MLXFastError.invalidInput(
                "\(label) contains a negative dimension"
            )
        }
        let next = result.addingReportingOverflow(value)
        guard !next.overflow else {
            throw MLXFastError.invalidInput("\(label) overflows Int")
        }
        result = next.partialValue
    }
    return result
}
