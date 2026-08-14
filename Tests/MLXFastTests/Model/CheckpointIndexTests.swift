import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastTransform

@Test
func checkpointIndexToolsReturnsUniqueSortedSafetensorShards() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let index = root.appendingPathComponent("model.safetensors.index.json")
    try writeCheckpointIndex(
        index,
        weightMap: [
            "b": "model-00002.safetensors",
            "a": "model-00001.safetensors",
            "c": "model-00002.safetensors",
        ]
    )

    let shards = try CheckpointIndexTools.safetensorShardNames(from: index.path)

    #expect(shards == ["model-00001.safetensors", "model-00002.safetensors"])
}

@Test
func checkpointIndexToolsRejectsUnsupportedShard() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let index = root.appendingPathComponent("model.safetensors.index.json")
    try writeCheckpointIndex(
        index,
        weightMap: [
            "a": "pytorch_model.bin",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try CheckpointIndexTools.safetensorShardNames(from: index.path)
    }
}

@Test
func checkpointIndexToolsRejectsUnsafeShardPath() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let index = root.appendingPathComponent("model.safetensors.index.json")
    try writeCheckpointIndex(
        index,
        weightMap: [
            "a": "../model-00001.safetensors",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try CheckpointIndexTools.safetensorShardNames(from: index.path)
    }
}

private func writeCheckpointIndex(_ path: URL, weightMap: [String: String]) throws {
    let data = try JSONSerialization.data(
        withJSONObject: ["weight_map": weightMap],
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )
    try data.write(to: path)
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
