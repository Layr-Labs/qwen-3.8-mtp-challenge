import Foundation
import MLXFastCore

/// Gemma 4 tied-embedding packed13 sidecar generator. Invoked only on the
/// `.gemma4` transform family path -- Laguna's `lm_head` is a separate
/// quantized tensor (untied), so there is no tied vocabulary head to pack,
/// and the Laguna weight contract forbids metadata sidecars entirely. The
/// production-shape gate below ([262144, 672] packed embedding) is
/// Gemma-4-specific on purpose.
enum TiedHeadMetadataCoding {
    static let shardName = "mlxfast-tied-head-metadata.safetensors"
    static let weightName = "language_model.model.embed_tokens.weight"
    static let scalesName = "language_model.model.embed_tokens.scales"
    static let biasesName = "language_model.model.embed_tokens.biases"
    static let packedIndicesName =
        "language_model.model.embed_tokens.tied_head_packed13_indices"
    static let lutName =
        "language_model.model.embed_tokens.tied_head_packed13_lut"

    static let rowCount = 262_144
    static let groupsPerRow = 84
    static let bitsPerIndex = 13
    static let wordsPerRow = 35
    static let maximumLUTCount = 1 << bitsPerIndex

    static func writeSidecar(
        sourceDirectory: URL,
        index: CheckpointIndex,
        sourceHeaders: [String: SafetensorsHeader],
        selectedKeys: Set<String>,
        destinationDirectory: URL
    ) throws -> GeneratedAffineMetadataReport {
        guard selectedKeys.contains(weightName) else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }
        let weightInfo = try tensorInfo(
            named: weightName,
            index: index,
            sourceHeaders: sourceHeaders
        )
        // Tiny synthetic fixtures and unquantized checkpoints are not eligible
        // for this production-shape optimization.
        guard weightInfo.dtype == TensorDType.u32.rawValue,
              weightInfo.shape == [rowCount, 672]
        else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }
        guard !Set(index.weightMap.values).contains(shardName) else {
            throw MLXFastError.invalidInput(
                "checkpoint already contains reserved generated shard \(shardName)"
            )
        }
        guard selectedKeys.contains(scalesName), selectedKeys.contains(biasesName) else {
            throw MLXFastError.invalidInput(
                "quantized tied embedding is missing BF16 scales or biases"
            )
        }
        let scalesInfo = try tensorInfo(
            named: scalesName,
            index: index,
            sourceHeaders: sourceHeaders
        )
        let biasesInfo = try tensorInfo(
            named: biasesName,
            index: index,
            sourceHeaders: sourceHeaders
        )
        guard scalesInfo.dtype == TensorDType.bf16.rawValue,
              biasesInfo.dtype == TensorDType.bf16.rawValue,
              scalesInfo.shape == [rowCount, groupsPerRow],
              biasesInfo.shape == scalesInfo.shape
        else {
            throw MLXFastError.invalidInput(
                "quantized tied embedding has incompatible scale or bias metadata"
            )
        }

        let scales = try tensorBytes(
            named: scalesName,
            sourceDirectory: sourceDirectory,
            index: index,
            sourceHeaders: sourceHeaders
        )
        let biases = try tensorBytes(
            named: biasesName,
            sourceDirectory: sourceDirectory,
            index: index,
            sourceHeaders: sourceHeaders
        )
        let lut = try makeOrderedLUT(
            scales: scales,
            biases: biases,
            name: "model.embed_tokens"
        )
        guard lut.count <= maximumLUTCount else {
            throw MLXFastError.invalidInput(
                "tied embedding affine metadata LUT requires more than 13 bits"
            )
        }
        let packedIndices = try makePacked13Indices(
            scales: scales,
            biases: biases,
            lut: lut,
            rows: rowCount,
            groupsPerRow: groupsPerRow,
            name: "model.embed_tokens"
        )

        let destination = destinationDirectory.appendingPathComponent(shardName)
        try writeSidecar(
            destination,
            packedIndices: packedIndices,
            lut: lut
        )
        let lutByteCount = lut.count * TensorDType.u32.byteWidth
        return GeneratedAffineMetadataReport(
            weightMap: [
                packedIndicesName: shardName,
                lutName: shardName,
            ],
            tensorByteCount: packedIndices.count + lutByteCount
        )
    }

    static func makeOrderedLUT(
        scales: Data,
        biases: Data,
        name: String
    ) throws -> [UInt32] {
        guard scales.count == biases.count, scales.count.isMultiple(of: 2) else {
            throw MLXFastError.invalidInput(
                "scale and bias payloads differ for packed metadata projection \(name)"
            )
        }
        var pairToIndex: [UInt32: UInt16] = [:]
        pairToIndex.reserveCapacity(maximumLUTCount)
        var lut: [UInt32] = []
        lut.reserveCapacity(maximumLUTCount)
        try scales.withUnsafeBytes { scaleBytes in
            try biases.withUnsafeBytes { biasBytes in
                for offset in stride(from: 0, to: scales.count, by: 2) {
                    let scale = scaleBytes.loadUnaligned(
                        fromByteOffset: offset,
                        as: UInt16.self
                    ).littleEndian
                    let bias = biasBytes.loadUnaligned(
                        fromByteOffset: offset,
                        as: UInt16.self
                    ).littleEndian
                    let pair = UInt32(scale) | (UInt32(bias) << 16)
                    if pairToIndex[pair] == nil {
                        guard lut.count < maximumLUTCount else {
                            throw MLXFastError.invalidInput(
                                "affine metadata LUT exceeds 13-bit capacity for \(name)"
                            )
                        }
                        pairToIndex[pair] = UInt16(lut.count)
                        lut.append(pair)
                    }
                }
            }
        }
        guard !lut.isEmpty else {
            throw MLXFastError.invalidInput("affine metadata LUT is empty for \(name)")
        }
        return lut
    }

    static func makePacked13Indices(
        scales: Data,
        biases: Data,
        lut: [UInt32],
        rows: Int,
        groupsPerRow: Int,
        name: String
    ) throws -> Data {
        let (elementCount, elementOverflow) = rows.multipliedReportingOverflow(
            by: groupsPerRow
        )
        let (payloadByteCount, payloadOverflow) = elementCount.multipliedReportingOverflow(
            by: 2
        )
        guard !elementOverflow, !payloadOverflow,
              scales.count == payloadByteCount,
              biases.count == payloadByteCount,
              groupsPerRow > 0
        else {
            throw MLXFastError.invalidInput(
                "scale and bias payload shape is invalid for packed metadata projection \(name)"
            )
        }
        let wordsPerRow = (groupsPerRow * bitsPerIndex + 31) / 32
        let (wordCount, wordOverflow) = rows.multipliedReportingOverflow(by: wordsPerRow)
        let (outputByteCount, byteOverflow) = wordCount.multipliedReportingOverflow(by: 4)
        guard !wordOverflow, !byteOverflow else {
            throw MLXFastError.invalidInput(
                "packed metadata shape overflows for \(name)"
            )
        }
        let pairToIndex = Dictionary(
            uniqueKeysWithValues: lut.enumerated().map {
                ($0.element, UInt32($0.offset))
            }
        )
        var output = Data(count: outputByteCount)
        try output.withUnsafeMutableBytes { outputBytes in
            try scales.withUnsafeBytes { scaleBytes in
                try biases.withUnsafeBytes { biasBytes in
                    for row in 0..<rows {
                        for column in 0..<groupsPerRow {
                            let element = row * groupsPerRow + column
                            let sourceOffset = element * 2
                            let scale = scaleBytes.loadUnaligned(
                                fromByteOffset: sourceOffset,
                                as: UInt16.self
                            ).littleEndian
                            let bias = biasBytes.loadUnaligned(
                                fromByteOffset: sourceOffset,
                                as: UInt16.self
                            ).littleEndian
                            let pair = UInt32(scale) | (UInt32(bias) << 16)
                            guard let index = pairToIndex[pair], index < maximumLUTCount else {
                                throw MLXFastError.invalidInput(
                                    "affine metadata pair missing from packed LUT for \(name)"
                                )
                            }

                            let bitOffset = column * bitsPerIndex
                            let wordInRow = bitOffset / 32
                            let shift = bitOffset % 32
                            let outputWord = row * wordsPerRow + wordInRow
                            try orLittleEndianWord(
                                index << UInt32(shift),
                                at: outputWord,
                                in: outputBytes
                            )
                            if shift > 32 - bitsPerIndex {
                                try orLittleEndianWord(
                                    index >> UInt32(32 - shift),
                                    at: outputWord + 1,
                                    in: outputBytes
                                )
                            }
                        }
                    }
                }
            }
        }
        return output
    }

    static func unpackPacked13Row(
        _ packed: Data,
        row: Int,
        groupsPerRow: Int
    ) -> [UInt16] {
        let wordsPerRow = (groupsPerRow * bitsPerIndex + 31) / 32
        return packed.withUnsafeBytes { bytes in
            (0..<groupsPerRow).map { column in
                let bitOffset = column * bitsPerIndex
                let wordInRow = bitOffset / 32
                let shift = bitOffset % 32
                let wordOffset = (row * wordsPerRow + wordInRow) * 4
                let low = bytes.loadUnaligned(
                    fromByteOffset: wordOffset,
                    as: UInt32.self
                ).littleEndian
                var value = low >> UInt32(shift)
                if shift > 32 - bitsPerIndex {
                    let high = bytes.loadUnaligned(
                        fromByteOffset: wordOffset + 4,
                        as: UInt32.self
                    ).littleEndian
                    value |= high << UInt32(32 - shift)
                }
                return UInt16(value & 0x1fff)
            }
        }
    }

    private static func orLittleEndianWord(
        _ value: UInt32,
        at index: Int,
        in bytes: UnsafeMutableRawBufferPointer
    ) throws {
        let offset = index * 4
        guard offset >= 0, offset + 4 <= bytes.count else {
            throw MLXFastError.invalidInput("packed metadata word offset is out of bounds")
        }
        let existing = bytes.loadUnaligned(
            fromByteOffset: offset,
            as: UInt32.self
        ).littleEndian
        bytes.storeBytes(
            of: (existing | value).littleEndian,
            toByteOffset: offset,
            as: UInt32.self
        )
    }

    private static func tensorInfo(
        named name: String,
        index: CheckpointIndex,
        sourceHeaders: [String: SafetensorsHeader]
    ) throws -> SafetensorInfo {
        guard let shardName = index.weightMap[name],
              let info = sourceHeaders[shardName]?.tensors[name]
        else {
            throw MLXFastError.invalidInput("missing validated tensor metadata for \(name)")
        }
        return info
    }

    private static func tensorBytes(
        named name: String,
        sourceDirectory: URL,
        index: CheckpointIndex,
        sourceHeaders: [String: SafetensorsHeader]
    ) throws -> Data {
        guard let sourceShard = index.weightMap[name],
              let header = sourceHeaders[sourceShard],
              let info = header.tensors[name]
        else {
            throw MLXFastError.invalidInput("missing validated tensor metadata for \(name)")
        }
        let source = sourceDirectory.appendingPathComponent(sourceShard)
        let handle = try FileHandle(forReadingFrom: source)
        defer { try? handle.close() }
        let (offset, overflow) = header.dataBaseOffset.addingReportingOverflow(
            UInt64(info.dataStart)
        )
        guard !overflow else {
            throw MLXFastError.invalidInput("tensor byte offset overflows for \(name)")
        }
        try handle.seek(toOffset: offset)
        let bytes = handle.readData(ofLength: info.byteCount)
        guard bytes.count == info.byteCount else {
            throw MLXFastError.invalidInput(
                "short read while generating tied-head metadata for \(name)"
            )
        }
        return bytes
    }

    private static func writeSidecar(
        _ destination: URL,
        packedIndices: Data,
        lut: [UInt32]
    ) throws {
        let lutData = littleEndianData(lut)
        let packedEnd = packedIndices.count
        let (lutEnd, overflow) = packedEnd.addingReportingOverflow(lutData.count)
        guard !overflow else {
            throw MLXFastError.invalidInput("tied-head metadata shard size overflows Int")
        }
        let headerObject: [String: Any] = [
            "__metadata__": ["format": "mlxfast-tied-head-packed13-v1"],
            packedIndicesName: [
                "dtype": TensorDType.u32.rawValue,
                "shape": [rowCount, wordsPerRow],
                "data_offsets": [0, packedEnd],
            ],
            lutName: [
                "dtype": TensorDType.u32.rawValue,
                "shape": [lut.count],
                "data_offsets": [packedEnd, lutEnd],
            ],
        ]
        var header = try JSONSerialization.data(
            withJSONObject: headerObject,
            options: [.sortedKeys]
        )
        while !header.count.isMultiple(of: 8) {
            header.append(0x20)
        }

        try Data().write(to: destination, options: [.withoutOverwriting])
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        var headerLength = UInt64(header.count).littleEndian
        try output.write(contentsOf: Data(bytes: &headerLength, count: 8))
        try output.write(contentsOf: header)
        try output.write(contentsOf: packedIndices)
        try output.write(contentsOf: lutData)
        try output.synchronize()
    }

    private static func littleEndianData(_ values: [UInt32]) -> Data {
        var output = Data(count: values.count * 4)
        output.withUnsafeMutableBytes { bytes in
            for (index, value) in values.enumerated() {
                bytes.storeBytes(
                    of: value.littleEndian,
                    toByteOffset: index * 4,
                    as: UInt32.self
                )
            }
        }
        return output
    }
}
