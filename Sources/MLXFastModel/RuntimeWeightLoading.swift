import Foundation
import MLX
import MLXFastCore

// Model-agnostic transformed-weight loading helpers shared by the runtime
// weight caches. Everything here is keyed only on the
// safetensors index/shard structure and the `language_model.` text-tower
// prefix convention -- nothing is specific to one architecture, so the
// model-specific loaders can come and go without touching this file.

func loadRuntimeWeightArrays(
    denseStore: DenseTensorStore
) throws -> [String: MLXArray] {
    let bridge = MLXArrayTensorBridge()
    return try loadRuntimeWeightValues(denseStore: denseStore) { tensor in
        try bridge.makeArray(from: tensor)
    }
}

/// Materializes runtime weights one tensor at a time. `DenseTensorStore`
/// drains the source buffer's autorelease pool before visiting the next
/// tensor; only the detached value returned by `makeValue` remains resident.
/// The production value is an `MLXArray`, whose Data initializer copies the
/// bytes into MLX-owned storage.
func loadRuntimeWeightValues<Value>(
    denseStore: DenseTensorStore,
    makeValue: (MaterializedTensor) throws -> Value
) throws -> [String: Value] {
    let directory = URL(fileURLWithPath: denseStore.weightsPath)
    let entries = try FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil
    )
    let discoveredShards = entries
        .filter { $0.pathExtension == "safetensors" }
        .map(\.lastPathComponent)
    let shardNames = try validateRuntimeShardInventory(
        referencedShards: denseStore.shardNames,
        discoveredShards: discoveredShards
    )

    var loadedWeights: [String: Value] = [:]
    loadedWeights.reserveCapacity(denseStore.tensorNames.count)
    var expectedLoadedNames: Set<String> = []
    var nameTracker = RuntimeWeightNameTracker()
    for shardName in shardNames {
        let expectedNames = denseStore.tensorNames(inShard: shardName)
        expectedLoadedNames.formUnion(expectedNames)
        let shard = directory.appendingPathComponent(shardName)
        let discoveredNames = Set(try Safetensors.readHeader(shard).tensors.keys)
        try validateRuntimeTensorInventory(
            shardName: shardName,
            expectedNames: expectedNames,
            discoveredNames: discoveredNames
        )
        try denseStore.forEachMaterializedTensor(inShard: shardName) { record, tensor in
            let renamed = try nameTracker.register(
                originalName: record.name,
                shardName: shardName,
                expectedNames: expectedNames
            )
            loadedWeights[renamed] = try makeValue(tensor)
        }
    }
    try nameTracker.validateComplete(expectedNames: expectedLoadedNames)
    return loadedWeights
}

func validateRuntimeTensorInventory(
    shardName: String,
    expectedNames: Set<String>,
    discoveredNames: Set<String>
) throws {
    let missing = expectedNames.subtracting(discoveredNames).sorted()
    guard missing.isEmpty else {
        throw MLXFastError.invalidInput(
            "safetensors shard \(shardName) is missing indexed tensors: "
                + missing.joined(separator: ", ")
        )
    }
    let unindexed = discoveredNames.subtracting(expectedNames).sorted()
    guard unindexed.isEmpty else {
        throw MLXFastError.invalidInput(
            "safetensors shard \(shardName) contains unindexed or misplaced tensors: "
                + unindexed.joined(separator: ", ")
        )
    }
}

func validateRuntimeShardInventory(
    referencedShards: [String],
    discoveredShards: [String]
) throws -> [String] {
    let referenced = Set(referencedShards)
    let discovered = Set(discoveredShards)
    guard !referenced.isEmpty else {
        throw MLXFastError.missingFile("dense safetensors index references no shards")
    }
    let missing = referenced.subtracting(discovered).sorted()
    guard missing.isEmpty else {
        throw MLXFastError.missingFile(
            "dense safetensors index references missing shards: \(missing.joined(separator: ", "))"
        )
    }
    let unindexed = discovered.subtracting(referenced).sorted()
    guard unindexed.isEmpty else {
        throw MLXFastError.invalidInput(
            "weights directory contains unindexed safetensors shards: \(unindexed.joined(separator: ", "))"
        )
    }
    return referenced.sorted()
}

struct RuntimeWeightNameTracker {
    private static let languageModelPrefix = "language_model."
    private var originalNames: Set<String> = []
    private var runtimeNames: Set<String> = []

    mutating func register(
        originalName: String,
        shardName: String,
        expectedNames: Set<String>
    ) throws -> String {
        guard expectedNames.contains(originalName) else {
            throw MLXFastError.invalidInput(
                "safetensors shard \(shardName) contains unindexed or misplaced tensor \(originalName)"
            )
        }
        guard originalNames.insert(originalName).inserted else {
            throw MLXFastError.invalidInput("duplicate safetensors tensor \(originalName)")
        }
        let runtimeName = originalName.hasPrefix(Self.languageModelPrefix)
            ? String(originalName.dropFirst(Self.languageModelPrefix.count))
            : originalName
        guard runtimeNames.insert(runtimeName).inserted else {
            throw MLXFastError.invalidInput(
                "safetensors tensor names collide after runtime rename: \(runtimeName)"
            )
        }
        return runtimeName
    }

    func validateComplete(expectedNames: Set<String>) throws {
        let missing = expectedNames.subtracting(originalNames).sorted()
        guard missing.isEmpty else {
            throw MLXFastError.invalidInput(
                "indexed safetensors tensors were not loaded: \(missing.joined(separator: ", "))"
            )
        }
    }
}
