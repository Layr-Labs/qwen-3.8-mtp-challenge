import CryptoKit
import Foundation
import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastTransform

/// The identity of the checkpoint the Qwen3627B4bit CAPTURE fixtures were taken
/// from. Since the 2026-08-14 backbone repin this is NO LONGER
/// `MLXFastConstants.referenceModelRepository` / `_Revision`, which name the
/// pinned Qwen 3.8 target (`EigenLabs/Qwen3.8-27B-4bit`, revision pending
/// publish). The divergence is deliberate: these fixtures record what was
/// actually captured, and renaming them to match the new pin would assert a
/// capture nobody made. The tower GEOMETRY they encode was verified identical
/// across the two.
let qwen36Repository = "mlx-community/Qwen3.6-27B-4bit"
let qwen36Revision = "c000ac2c2057d94be3fa931000c31723aac53282"

/// SHA256 of the checkpoint's own `config.json` bytes as published, NOT of the
/// normalized `fixtures/qwen3_6_27b_config.json` re-render. The cross-check
/// against `fixtures/reference_qwen3_8_27b_4bit.sha256` in
/// `qwen36ConfigContractDigestMatchesTheReferenceManifest` is HALF-SUSPENDED:
/// that manifest was repointed at the 3.8 backbone and its per-file body is a
/// stub, so it carries no `config.json` record to compare against yet.
let qwen36ConfigSHA256 =
    "ede24666ac51e6d5ab948a8a1e6c72fc6effd941ba3aabb6dd942eb517c78043"

// Tests/MLXFastTests/Model/<this file> -> repository root is four levels up.
private let qwen36ArtifactRepositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

let qwen36ConfigFixtureURL = qwen36ArtifactRepositoryRoot
    .appendingPathComponent("fixtures/qwen3_6_27b_config.json")

let qwen36InventoryFixtureURL = qwen36ArtifactRepositoryRoot
    .appendingPathComponent("fixtures/qwen3_6_27b_tensor_inventory.json")

let qwen36ReferenceManifestURL = qwen36ArtifactRepositoryRoot
    .appendingPathComponent("fixtures/reference_qwen3_8_27b_4bit.sha256")

let qwen36ConfigContractURL = qwen36ArtifactRepositoryRoot
    .appendingPathComponent("Tests/Fixtures/Qwen3627B4bit/config-contract.json")

let qwen36HeaderInventoryContractURL = qwen36ArtifactRepositoryRoot
    .appendingPathComponent(
        "Tests/Fixtures/Qwen3627B4bit/header-inventory-contract.json"
    )

struct Qwen36TensorRecord: Decodable, Equatable {
    var name: String
    var dtype: String
    var shape: [Int]
    var shard: Int

    init(from decoder: Decoder) throws {
        var values = try decoder.unkeyedContainer()
        name = try values.decode(String.self)
        dtype = try values.decode(String.self)
        shape = try values.decode([Int].self)
        shard = try values.decode(Int.self)
        guard values.isAtEnd else {
            throw DecodingError.dataCorruptedError(
                in: values,
                debugDescription: "Qwen3.6 tensor record must contain exactly four values"
            )
        }
    }
}

struct Qwen36InventoryFixture: Decodable {
    struct Source: Decodable {
        let repository: String
        let revision: String
        let configSHA256: String
        let indexSHA256: String
        let indexTotalSize: Int

        enum CodingKeys: String, CodingKey {
            case repository
            case revision
            case configSHA256 = "config_sha256"
            case indexSHA256 = "index_sha256"
            case indexTotalSize = "index_total_size"
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
    let representative: [Qwen36TensorRecord]
    let tensors: [Qwen36TensorRecord]

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

/// `Tests/Fixtures/Qwen3627B4bit/header-inventory-contract.json`.
///
/// Its `canonical_inventory_sha256` is deliberately a DIFFERENT digest from
/// the `fixtures/` one: this canonicalization omits the shard name, so the
/// contract stays valid across a re-sharded republish of the same tensors.
struct Qwen36HeaderInventoryContract: Decodable {
    struct Source: Decodable {
        let repository: String
        let revision: String
    }

    struct RepresentativeTensor: Decodable {
        let name: String
        let dtype: String
        let shape: [Int]
    }

    let schemaVersion: Int
    let source: Source
    let tensorCount: Int
    let dtypeCounts: [String: Int]
    let canonicalRecordFormat: String
    let canonicalInventorySHA256: String
    let representativeTensors: [RepresentativeTensor]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case source
        case tensorCount = "tensor_count"
        case dtypeCounts = "dtype_counts"
        case canonicalRecordFormat = "canonical_record_format"
        case canonicalInventorySHA256 = "canonical_inventory_sha256"
        case representativeTensors = "representative_tensors"
    }
}

func qwen36ConfigData() throws -> Data {
    try Data(contentsOf: qwen36ConfigFixtureURL)
}

func qwen36ConfigObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: qwen36ConfigData()
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput("Qwen3.6 config fixture must be a JSON object")
    }
    return object
}

func qwen36InventoryFixture() throws -> Qwen36InventoryFixture {
    try JSONDecoder().decode(
        Qwen36InventoryFixture.self,
        from: Data(contentsOf: qwen36InventoryFixtureURL)
    )
}

func qwen36HeaderInventoryContract() throws -> Qwen36HeaderInventoryContract {
    try JSONDecoder().decode(
        Qwen36HeaderInventoryContract.self,
        from: Data(contentsOf: qwen36HeaderInventoryContractURL)
    )
}

func qwen36SHA256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

/// `fixtures/` canonicalization: compact JSON `[name,dtype,shape,shard_name]`
/// per line, sorted by name. The shard NAME is part of the record.
func qwen36CanonicalInventoryData(
    _ records: [Qwen36TensorRecord],
    shards: [Qwen36InventoryFixture.Shard]
) -> Data {
    let lines = records.sorted { $0.name < $1.name }.map { record -> String in
        precondition((1...shards.count).contains(record.shard))
        let shape = record.shape.map(String.init).joined(separator: ",")
        let shardName = shards[record.shard - 1].name
        return #"["\#(record.name)","\#(record.dtype)",[\#(shape)],"\#(shardName)"]"#
    }
    return Data((lines.joined(separator: "\n") + "\n").utf8)
}

/// `Tests/Fixtures/` canonicalization: `name<TAB>dtype<TAB>shape<LF>`, sorted
/// by name, shard-independent.
func qwen36CanonicalHeaderInventoryData(_ records: [Qwen36TensorRecord]) -> Data {
    let lines = records.sorted { $0.name < $1.name }.map { record in
        "\(record.name)\t\(record.dtype)\t\(record.shape.map(String.init).joined(separator: ","))"
    }
    return Data((lines.joined(separator: "\n") + "\n").utf8)
}

struct Qwen36ValidationMetadata {
    let selectedKeys: Set<String>
    let index: CheckpointIndex
    let headers: [String: SafetensorsHeader]
}

/// Rebuild the checkpoint index and per-shard headers from the public
/// inventory fixture, so transform-side validation can be exercised without
/// the 15 GB checkpoint. `selectedKeys` mirrors what
/// `SwiftTransform.isSelectedTextTowerKey` picks for `.qwen35`.
func qwen36ValidationMetadata(
    records: [Qwen36TensorRecord],
    shards: [Qwen36InventoryFixture.Shard]
) -> Qwen36ValidationMetadata {
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
    return Qwen36ValidationMetadata(
        selectedKeys: Set(
            records.map(\.name).filter {
                SwiftTransform.isSelectedTextTowerKey($0, family: .qwen35)
            }
        ),
        index: CheckpointIndex(
            raw: ["weight_map": weightMap],
            weightMap: weightMap
        ),
        headers: headers
    )
}

func exactQwen36Quantization() throws -> Qwen35TransformQuantizationSpec {
    try Qwen35CheckpointValidation.quantizationSpec(fromConfigRoot: qwen36ConfigObject())
}
