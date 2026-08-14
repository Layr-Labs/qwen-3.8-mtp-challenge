import Foundation
import MLXFastCore

struct CheckpointIndex {
    var raw: [String: Any]
    var weightMap: [String: String]

    static func load(from path: URL) throws -> CheckpointIndex {
        let data = try Data(contentsOf: path)
        let object = try JSONSerialization.jsonObject(with: data)
        guard var raw = object as? [String: Any] else {
            throw MLXFastError.invalidInput("checkpoint index must be a JSON object: \(path.path)")
        }
        guard let weightMap = raw["weight_map"] as? [String: String] else {
            throw MLXFastError.invalidInput("checkpoint index missing weight_map: \(path.path)")
        }
        raw["weight_map"] = weightMap
        return CheckpointIndex(raw: raw, weightMap: weightMap)
    }

    func canonicalData() throws -> Data {
        try JSONSerialization.data(
            withJSONObject: raw,
            options: [.sortedKeys, .withoutEscapingSlashes]
        )
    }

    func writeStripped(
        to path: URL,
        keeping keys: Set<String>,
        totalTensorByteCount: Int,
        additionalWeightMap: [String: String] = [:]
    ) throws {
        var output = raw
        var outputWeightMap = weightMap.filter { keys.contains($0.key) }
        for key in additionalWeightMap.keys.sorted() {
            guard outputWeightMap[key] == nil else {
                throw MLXFastError.invalidInput(
                    "generated tensor name collides with checkpoint tensor \(key)"
                )
            }
            outputWeightMap[key] = additionalWeightMap[key]
        }
        output["weight_map"] = outputWeightMap
        var metadata = output["metadata"] as? [String: Any] ?? [:]
        metadata["total_size"] = totalTensorByteCount
        output["metadata"] = metadata
        let data = try JSONSerialization.data(
            withJSONObject: output,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try FileManager.default.createDirectory(
            at: path.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: path)
    }
}

extension CheckpointIndex {
    static func buildFromSafetensors(in directory: URL) throws -> CheckpointIndex {
        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        let shards = files
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var weightMap: [String: String] = [:]
        for shard in shards {
            let header = try Safetensors.readHeader(shard)
            for key in header.tensors.keys.sorted() {
                if let existingShard = weightMap[key] {
                    throw MLXFastError.invalidInput(
                        "duplicate tensor \(key) appears in safetensors shards \(existingShard) and \(shard.lastPathComponent)"
                    )
                }
                weightMap[key] = shard.lastPathComponent
            }
        }
        return CheckpointIndex(raw: ["weight_map": weightMap], weightMap: weightMap)
    }
}

public enum CheckpointIndexTools {
    public static func safetensorShardNames(from indexPath: String) throws -> [String] {
        let index = try CheckpointIndex.load(from: URL(fileURLWithPath: indexPath))
        let shards = Set(index.weightMap.values)
        guard !shards.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no shard names: \(indexPath)")
        }
        for shard in shards.sorted() {
            try validateSafetensorsShardName(shard, context: "checkpoint index")
        }
        return shards.sorted()
    }
}
