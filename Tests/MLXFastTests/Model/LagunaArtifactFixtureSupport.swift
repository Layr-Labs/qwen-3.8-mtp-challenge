import CryptoKit
import Foundation
import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastTransform

let poolsideLagunaRepository = "poolside/Laguna-XS-2.1-NVFP4-mlx"
let poolsideLagunaRevision = "841778bda563a36104dd521e37d99218e46f4f25"

// Tests/MLXFastTests/Model/<this file> -> repository root is four levels up.
private let lagunaArtifactRepositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

let poolsideLagunaConfigFixtureURL = lagunaArtifactRepositoryRoot
    .appendingPathComponent("fixtures/poolside_laguna_xs_2_1_nvfp4_config.json")

let poolsideLagunaInventoryFixtureURL = lagunaArtifactRepositoryRoot
    .appendingPathComponent(
        "fixtures/poolside_laguna_xs_2_1_nvfp4_tensor_inventory.json"
    )

struct PoolsideLagunaTensorRecord: Decodable, Equatable {
    var name: String
    var dtype: String
    var shape: [Int]
    var shard: Int

    init(name: String, dtype: String, shape: [Int], shard: Int) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.shard = shard
    }

    init(from decoder: Decoder) throws {
        var values = try decoder.unkeyedContainer()
        name = try values.decode(String.self)
        dtype = try values.decode(String.self)
        shape = try values.decode([Int].self)
        shard = try values.decode(Int.self)
        guard values.isAtEnd else {
            throw DecodingError.dataCorruptedError(
                in: values,
                debugDescription: "Laguna tensor record must contain exactly four values"
            )
        }
    }
}

struct PoolsideLagunaInventoryFixture: Decodable {
    struct Source: Decodable {
        let repository: String
        let revision: String
        let configSHA256: String
        let indexSHA256: String
        let indexTotalSize: Int
        let indexTotalParameters: Int

        enum CodingKeys: String, CodingKey {
            case repository
            case revision
            case configSHA256 = "config_sha256"
            case indexSHA256 = "index_sha256"
            case indexTotalSize = "index_total_size"
            case indexTotalParameters = "index_total_parameters"
        }
    }

    struct Summary: Decodable {
        let tensorCount: Int
        let dtypeCounts: [String: Int]
        let canonicalSHA256: String

        enum CodingKeys: String, CodingKey {
            case tensorCount = "tensor_count"
            case dtypeCounts = "dtype_counts"
            case canonicalSHA256 = "canonical_sha256"
        }
    }

    struct Shard: Decodable {
        let name: String
        let headerLength: Int
        let headerSHA256: String
        let tensorCount: Int
        let dtypeCounts: [String: Int]
        let canonicalSHA256: String

        enum CodingKeys: String, CodingKey {
            case name
            case headerLength = "header_length"
            case headerSHA256 = "header_sha256"
            case tensorCount = "tensor_count"
            case dtypeCounts = "dtype_counts"
            case canonicalSHA256 = "canonical_sha256"
        }
    }

    let schemaVersion: Int
    let source: Source
    let canonicalization: String
    let summary: Summary
    let shards: [Shard]
    let representative: [PoolsideLagunaTensorRecord]
    let tensors: [PoolsideLagunaTensorRecord]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case source
        case canonicalization
        case summary
        case shards
        case representative
        case tensors
    }
}

func poolsideLagunaConfigData() throws -> Data {
    try Data(contentsOf: poolsideLagunaConfigFixtureURL)
}

func poolsideLagunaConfigObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: poolsideLagunaConfigData()
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput("Poolside Laguna config fixture must be a JSON object")
    }
    return object
}

func poolsideLagunaInventoryFixture() throws -> PoolsideLagunaInventoryFixture {
    try JSONDecoder().decode(
        PoolsideLagunaInventoryFixture.self,
        from: Data(contentsOf: poolsideLagunaInventoryFixtureURL)
    )
}

func loadPoolsideLagunaConfig(_ object: [String: Any]) throws -> LagunaConfig {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mlxfast-laguna-config-\(UUID().uuidString)",
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let data = try JSONSerialization.data(
        withJSONObject: object,
        options: [.sortedKeys, .withoutEscapingSlashes]
    )
    try data.write(to: root.appendingPathComponent("config.json"))
    return try LagunaConfig.load(from: root.path)
}

func poolsideLagunaCanonicalInventoryData(
    _ records: [PoolsideLagunaTensorRecord],
    shards: [PoolsideLagunaInventoryFixture.Shard]
) -> Data {
    let lines = records.sorted { $0.name < $1.name }.map { record -> String in
        precondition((1...shards.count).contains(record.shard))
        let shape = record.shape.map(String.init).joined(separator: ",")
        let shardName = shards[record.shard - 1].name
        return #"["\#(record.name)","\#(record.dtype)",[\#(shape)],"\#(shardName)"]"#
    }
    return Data((lines.joined(separator: "\n") + "\n").utf8)
}

func poolsideLagunaSHA256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

struct PoolsideLagunaValidationMetadata {
    let selectedKeys: Set<String>
    let index: CheckpointIndex
    let headers: [String: SafetensorsHeader]
}

func poolsideLagunaValidationMetadata(
    records: [PoolsideLagunaTensorRecord],
    shards: [PoolsideLagunaInventoryFixture.Shard]
) -> PoolsideLagunaValidationMetadata {
    var weightMap: [String: String] = [:]
    var tensorsByShard: [String: [String: SafetensorInfo]] = [:]

    for record in records {
        precondition((1...shards.count).contains(record.shard))
        let shardName = shards[record.shard - 1].name
        precondition(weightMap.updateValue(shardName, forKey: record.name) == nil)
        tensorsByShard[shardName, default: [:]][record.name] = SafetensorInfo(
            name: record.name,
            dtype: record.dtype,
            shape: record.shape,
            dataStart: 0,
            dataEnd: 1
        )
    }

    let headers = Dictionary(uniqueKeysWithValues: shards.map { shard in
        (
            shard.name,
            SafetensorsHeader(
                headerLength: shard.headerLength,
                metadata: ["format": "pt"],
                tensors: tensorsByShard[shard.name, default: [:]]
            )
        )
    })
    return PoolsideLagunaValidationMetadata(
        selectedKeys: Set(records.map(\.name)),
        index: CheckpointIndex(
            raw: ["weight_map": weightMap],
            weightMap: weightMap
        ),
        headers: headers
    )
}

func exactPoolsideLagunaQuantization() throws -> LagunaTransformQuantizationSpec {
    try LagunaCheckpointValidation.quantizationSpec(fromConfigRoot: poolsideLagunaConfigObject())
}
