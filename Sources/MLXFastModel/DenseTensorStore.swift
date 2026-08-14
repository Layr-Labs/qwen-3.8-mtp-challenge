import Darwin
import Foundation
import MLXFastCore

public struct DenseTensorRecord: Equatable {
    public let name: String
    public let shard: String
    public let dtype: String
    public let shape: [Int]
    public let byteOffset: Int
    public let byteLength: Int
}

public final class DenseTensorStore {
    public let weightsPath: String
    private let recordsByName: [String: DenseTensorRecord]

    public init(weightsPath: String) throws {
        self.weightsPath = weightsPath
        self.recordsByName = try DenseTensorStore.loadRecords(weightsPath: weightsPath)
    }

    public var tensorNames: [String] {
        recordsByName.keys.sorted()
    }

    var shardNames: [String] {
        Set(recordsByName.values.map(\.shard)).sorted()
    }

    func tensorNames(inShard shard: String) -> Set<String> {
        Set(recordsByName.values.lazy.filter { $0.shard == shard }.map(\.name))
    }

    /// Visits a shard in file order while keeping each tensor's source `Data`
    /// inside its own autorelease pool. Callers that copy the tensor into its
    /// final runtime representation during `body` never retain source bytes
    /// beyond one tensor.
    func forEachMaterializedTensor(
        inShard shard: String,
        _ body: (DenseTensorRecord, MaterializedTensor) throws -> Void
    ) throws {
        let records = recordsByName.values
            .filter { $0.shard == shard }
            .sorted {
                if $0.byteOffset == $1.byteOffset {
                    return $0.name < $1.name
                }
                return $0.byteOffset < $1.byteOffset
            }
        guard !records.isEmpty else {
            throw MLXFastError.invalidInput(
                "dense safetensors index references no tensors in \(shard)"
            )
        }

        let handle = try uncachedReadHandle(forShard: shard)
        defer {
            try? handle.close()
        }
        for record in records {
            try autoreleasepool {
                let tensor = try materializeTensor(
                    name: record.name,
                    dtype: record.dtype,
                    shape: record.shape,
                    bytes: readBytes(for: record, from: handle)
                )
                try body(record, tensor)
            }
        }
    }

    public func record(named name: String) -> DenseTensorRecord? {
        recordsByName[name]
    }

    public func tensorBytes(named name: String) throws -> Data {
        guard let record = recordsByName[name] else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }

        let handle = try uncachedReadHandle(forShard: record.shard)
        defer {
            try? handle.close()
        }
        return try readBytes(for: record, from: handle)
    }

    private func uncachedReadHandle(forShard shard: String) throws -> FileHandle {
        let shardURL = URL(fileURLWithPath: weightsPath).appendingPathComponent(shard)
        let handle = try FileHandle(forReadingFrom: shardURL)
        // These descriptors feed short-lived staging buffers that are
        // immediately copied into long-lived MLX allocations. Retaining the
        // same bytes in the unified buffer cache can otherwise double the
        // model-load footprint on memory-constrained Apple Silicon.
        _ = Darwin.fcntl(handle.fileDescriptor, F_NOCACHE, 1)
        _ = Darwin.fcntl(handle.fileDescriptor, F_RDAHEAD, 0)
        return handle
    }

    private func readBytes(
        for record: DenseTensorRecord,
        from handle: FileHandle
    ) throws -> Data {
        guard let byteOffset = UInt64(exactly: record.byteOffset) else {
            throw MLXFastError.invalidInput(
                "negative byte offset for dense tensor \(record.name)"
            )
        }
        try handle.seek(toOffset: byteOffset)
        let data = handle.readData(ofLength: record.byteLength)
        guard data.count == record.byteLength else {
            throw MLXFastError.invalidInput(
                "short read for dense tensor \(record.name): \(data.count)/\(record.byteLength)"
            )
        }
        return data
    }

    public func materializedTensor(named name: String) throws -> MaterializedTensor {
        guard let record = recordsByName[name] else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        return try materializeTensor(
            name: record.name,
            dtype: record.dtype,
            shape: record.shape,
            bytes: tensorBytes(named: name)
        )
    }

    public func validateReadableByteRanges(fileManager: FileManager = .default) throws {
        let recordsByShard = Dictionary(grouping: recordsByName.values) { $0.shard }
        for shard in recordsByShard.keys.sorted() {
            let shardPath = URL(fileURLWithPath: weightsPath).appendingPathComponent(shard).path
            let attributes = try fileManager.attributesOfItem(atPath: shardPath)
            let byteCount = try fileSizeByteCount(from: attributes, path: shardPath)
            for record in recordsByShard[shard, default: []] {
                let dtype = try TensorDType.parse(record.dtype)
                let expectedByteLength = try expectedTensorByteCount(
                    name: record.name,
                    dtype: dtype,
                    shape: record.shape
                )
                guard record.byteLength == expectedByteLength else {
                    throw MLXFastError.invalidInput(
                        "dense tensor \(record.name) byte length \(record.byteLength) does not match dtype \(record.dtype) and shape \(record.shape) expected \(expectedByteLength)"
                    )
                }
                let (end, overflow) = record.byteOffset.addingReportingOverflow(record.byteLength)
                guard
                    !overflow,
                    record.byteOffset >= 0,
                    record.byteLength > 0,
                    end <= byteCount
                else {
                    throw MLXFastError.invalidInput(
                        "dense tensor \(record.name) byte range \(record.byteOffset)..<\(end) exceeds shard size \(byteCount)"
                    )
                }
            }
        }
    }

    private static func loadRecords(weightsPath: String) throws -> [String: DenseTensorRecord] {
        let weightsURL = URL(fileURLWithPath: weightsPath)
        try requireFile(
            weightsURL.appendingPathComponent("model.safetensors.index.json").path,
            description: "dense safetensors index"
        )

        let weightMap = try loadWeightMap(
            weightsURL.appendingPathComponent("model.safetensors.index.json")
        )
        for shard in Set(weightMap.values).sorted() {
            try validateSafetensorsShardName(shard, context: "dense safetensors index")
        }
        let keysByShard = Dictionary(grouping: weightMap.keys) { key in
            weightMap[key] ?? ""
        }

        var records: [String: DenseTensorRecord] = [:]
        for shard in keysByShard.keys.sorted() {
            let shardURL = weightsURL.appendingPathComponent(shard)
            let header = try Safetensors.readHeader(shardURL)
            for key in keysByShard[shard, default: []] {
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput(
                        "tensor \(key) is listed in dense index but missing from \(shard)"
                    )
                }
                guard let baseOffset = Int(exactly: header.dataBaseOffset) else {
                    throw MLXFastError.invalidInput("safetensors header offset exceeds Int range in \(shard)")
                }
                let (byteOffset, overflow) = baseOffset.addingReportingOverflow(info.dataStart)
                guard !overflow else {
                    throw MLXFastError.invalidInput("dense tensor byte offset overflows Int for \(key)")
                }
                records[key] = DenseTensorRecord(
                    name: key,
                    shard: shard,
                    dtype: info.dtype,
                    shape: info.shape,
                    byteOffset: byteOffset,
                    byteLength: info.byteCount
                )
            }
        }

        guard !records.isEmpty else {
            throw MLXFastError.invalidInput("dense tensor store contains no safetensors tensors")
        }
        return records
    }

    private static func loadWeightMap(_ path: URL) throws -> [String: String] {
        let data = try Data(contentsOf: path)
        let object = try JSONSerialization.jsonObject(with: data)
        guard
            let root = object as? [String: Any],
            let weightMap = root["weight_map"] as? [String: String]
        else {
            throw MLXFastError.invalidInput("dense safetensors index missing weight_map")
        }
        return weightMap
    }
}
