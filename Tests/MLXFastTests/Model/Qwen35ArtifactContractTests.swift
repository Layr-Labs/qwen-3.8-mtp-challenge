import CryptoKit
import Foundation
import MLXFastCore
import Testing
@testable import MLXFastModel
@testable import MLXFastRuntimeWorkerSupport
@testable import MLXFastTransform

/// The public `config.json` of `mlx-community/Qwen3.6-27B-4bit`, checked in at
/// `fixtures/qwen3_6_27b_config.json` normalized only for JSON formatting.
///
/// Note the artifact-snake prefix `qwen3_6_27b` deliberately omits a vendor
/// segment where the Laguna fixtures carry `poolside_`: mlx-community is the
/// distributor of an Alibaba model, not its author, so a `mlx_community_`
/// prefix would name the wrong party. See docs/qwen3.6-weight-contract.md.
@Test
func qwen36PublicConfigFixturePinsExactArtifactSemantics() throws {
    let root = try qwen36ConfigObject()
    #expect(try qwen36ConfigData().count < 10_000)
    #expect(Set(root.keys) == [
        "architectures",
        "eos_token_id",
        "image_token_id",
        "language_model_only",
        "model_type",
        "quantization",
        "quantization_config",
        "text_config",
        "tie_word_embeddings",
        "transformers_version",
        "video_token_id",
        "vision_config",
        "vision_end_token_id",
        "vision_start_token_id",
    ])
    #expect(root["model_type"] as? String == "qwen3_5")
    #expect(root["architectures"] as? [String] == ["Qwen3_5ForConditionalGeneration"])
    #expect(root["tie_word_embeddings"] as? Bool == false)

    // The multimodal wrapper is present but out of scope: the transform
    // selects `language_model.*` only, and `vision_config` never reaches the
    // runtime config.
    let vision = try #require(root["vision_config"] as? [String: Any])
    #expect(vision["depth"] as? Int == 27)
    #expect(vision["out_hidden_size"] as? Int == MLXFastConstants.hiddenSize)

    let text = try #require(root["text_config"] as? [String: Any])
    #expect(text["model_type"] as? String == "qwen3_5_text")
    #expect(text["vocab_size"] as? Int == MLXFastConstants.vocabSize)
    #expect(text["hidden_size"] as? Int == MLXFastConstants.hiddenSize)
    #expect(text["intermediate_size"] as? Int == MLXFastConstants.intermediateSize)
    #expect(text["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
    #expect(text["num_attention_heads"] as? Int == MLXFastConstants.attentionHeads)
    #expect(text["num_key_value_heads"] as? Int == 4)
    #expect(text["head_dim"] as? Int == 256)
    #expect(text["full_attention_interval"] as? Int == 4)
    #expect(text["tie_word_embeddings"] as? Bool == false)
    #expect(text["mtp_num_hidden_layers"] as? Int == 1)
    #expect(text["mtp_use_dedicated_embeddings"] as? Bool == false)
    #expect(text["pad_token_id"] is NSNull)

    let expectedLayerTypes = (0..<MLXFastConstants.numHiddenLayers).map {
        $0 % 4 == 3 ? "full_attention" : "linear_attention"
    }
    #expect(text["layer_types"] as? [String] == expectedLayerTypes)

    // Partial RoPE: 0.25 of head_dim 256 = 64 rotary dimensions, theta 1e7.
    let rope = try #require(text["rope_parameters"] as? [String: Any])
    #expect(rope["rope_type"] as? String == "default")
    #expect(rope["rope_theta"] as? Double == 10_000_000)
    #expect(rope["partial_rotary_factor"] as? Double == 0.25)
    #expect(rope["mrope_interleaved"] as? Bool == true)
    #expect(rope["mrope_section"] as? [Int] == [11, 11, 10])
    #expect(text["partial_rotary_factor"] as? Double == 0.25)

    // The same affine spec is published twice; both must agree.
    let spec = try Qwen35CheckpointValidation.quantizationSpec(fromConfigRoot: root)
    #expect(spec.groupSize == 64)
    #expect(spec.bits == 4)
    #expect(spec.mode == "affine")
}

/// End-to-end pin: the runtime config this checkpoint transforms into is
/// exactly what the runtime worker's pinned-configuration gate accepts.
///
/// This is the interface the weight contract documents, exercised in one step
/// -- transform emitter on the left, worker gate on the right -- so a change
/// to either side that breaks the other fails here rather than on box 3.
@Test
func qwen36TransformOutputSatisfiesTheRuntimeWorkerPinnedConfigGate() throws {
    let root = try qwen36ConfigObject()
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: root) == .qwen35)

    let runtimeConfig = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: root,
        family: .qwen35
    )
    try validateRuntimeWorkerPinnedConfigurationData(runtimeConfig)

    // The emitted config is the source `text_config` plus one `quantization`
    // block -- no vision config, no duplicated `quantization_config`.
    let emitted = try #require(
        try JSONSerialization.jsonObject(with: runtimeConfig) as? [String: Any]
    )
    #expect(emitted["vision_config"] == nil)
    #expect(emitted["quantization_config"] == nil)
    #expect(emitted["text_config"] == nil)
    let quantization = try #require(emitted["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == 64)
    #expect(quantization["bits"] as? Int == 4)
    #expect(quantization["mode"] as? String == "affine")

    // And it loads as the model target's own config type.
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mlxfast-qwen36-config-\(UUID().uuidString)",
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try runtimeConfig.write(to: directory.appendingPathComponent("config.json"))
    let parsed = try Qwen35Config.load(from: directory.path)
    #expect(parsed.numHiddenLayers == MLXFastConstants.numHiddenLayers)
    #expect(parsed.vocabSize == MLXFastConstants.vocabSize)
    #expect(parsed.quantizationGroupSize == 64)
    #expect(parsed.quantizationBits == 4)
    #expect(parsed.quantizationMode == "affine")
}

@Test
func qwen36TensorInventoryFixturePinsAllPublicHeaders() throws {
    let fixtureData = try Data(contentsOf: qwen36InventoryFixtureURL)
    let fixture = try qwen36InventoryFixture()

    #expect(fixtureData.count < 250_000)
    #expect(fixture.schemaVersion == 1)
    #expect(fixture.source.repository == qwen36Repository)
    #expect(fixture.source.revision == qwen36Revision)
    #expect(fixture.source.configSHA256 == qwen36ConfigSHA256)
    #expect(
        fixture.source.indexSHA256
            == "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8"
    )
    #expect(fixture.source.indexTotalSize == 16_054_262_240)
    #expect(fixture.canonicalization.contains("[name,dtype,shape,shard_name]"))

    // The published artifact carries the text tower AND the vision tower; the
    // transform selects only the former.
    #expect(fixture.tensors.count == 2_180)
    #expect(fixture.summary.tensorCount == fixture.tensors.count)
    #expect(fixture.summary.dtypeCounts == ["BF16": 1_682, "U32": 498])

    let textTower = fixture.tensors.filter {
        SwiftTransform.isSelectedTextTowerKey($0.name, family: .qwen35)
    }
    #expect(textTower.count == Qwen35CheckpointValidation.expectedTensorCount)
    #expect(textTower.count == 1_847)
    #expect(
        fixture.tensors.count - textTower.count
            == fixture.tensors.filter { $0.name.hasPrefix("vision_tower.") }.count
    )

    // Every name is unique and every shard index is in range.
    #expect(Set(fixture.tensors.map(\.name)).count == fixture.tensors.count)
    #expect(fixture.shards.count == 3)
    for record in fixture.tensors {
        #expect((1...fixture.shards.count).contains(record.shard))
        #expect(!record.shape.isEmpty)
        #expect(record.shape.allSatisfy { $0 > 0 })
    }

    var perShardCounts = [Int](repeating: 0, count: fixture.shards.count)
    for record in fixture.tensors {
        perShardCounts[record.shard - 1] += 1
    }
    for (offset, shard) in fixture.shards.enumerated() {
        #expect(shard.tensorCount == perShardCounts[offset], "\(shard.name)")
        #expect(shard.headerLength > 0)
        #expect(shard.headerSHA256.count == 64)
        #expect(
            shard.dtypeCounts.values.reduce(0, +) == shard.tensorCount,
            "\(shard.name)"
        )
    }

    let canonical = qwen36CanonicalInventoryData(fixture.tensors, shards: fixture.shards)
    let digest = qwen36SHA256(canonical)
    #expect(
        digest == fixture.summary.canonicalSHA256,
        Comment(rawValue: "actual canonical digest: \(digest)")
    )
    #expect(
        fixture.summary.canonicalSHA256
            == "f17f15bb2d498ab22478bea86b8e1a3e7fd7d939c65104338b04367cd11e3f54"
    )

    let byName = Dictionary(uniqueKeysWithValues: fixture.tensors.map { ($0.name, $0) })
    for representative in fixture.representative {
        #expect(byName[representative.name] == representative)
    }
}

/// The two checked-in canonical digests differ BY DESIGN: the `fixtures/`
/// record embeds the shard name, the `Tests/Fixtures/` record does not.
@Test
func qwen36HeaderInventoryContractPinsTheShardIndependentDigest() throws {
    let fixture = try qwen36InventoryFixture()
    let contract = try qwen36HeaderInventoryContract()

    #expect(contract.schemaVersion == 1)
    #expect(contract.source.repository == qwen36Repository)
    #expect(contract.source.revision == qwen36Revision)
    #expect(contract.canonicalRecordFormat == "UTF-8 name<TAB>dtype<TAB>comma-separated-shape<LF>, sorted by name")
    #expect(contract.tensorCount == fixture.tensors.count)
    #expect(contract.dtypeCounts == fixture.summary.dtypeCounts)

    let digest = qwen36SHA256(qwen36CanonicalHeaderInventoryData(fixture.tensors))
    #expect(
        digest == contract.canonicalInventorySHA256,
        Comment(rawValue: "actual header-inventory digest: \(digest)")
    )
    #expect(contract.canonicalInventorySHA256 != fixture.summary.canonicalSHA256)

    let byName = Dictionary(uniqueKeysWithValues: fixture.tensors.map { ($0.name, $0) })
    #expect(!contract.representativeTensors.isEmpty)
    for representative in contract.representativeTensors {
        let record = try #require(
            byName[representative.name],
            Comment(rawValue: representative.name)
        )
        #expect(record.dtype == representative.dtype)
        #expect(record.shape == representative.shape)
    }
}

@Test
func qwen36ConfigContractCarriesThePublicConfigAndItsPublishedDigest() throws {
    let data = try Data(contentsOf: qwen36ConfigContractURL)
    let contract = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    let source = try #require(contract["source"] as? [String: Any])
    #expect(source["repository"] as? String == qwen36Repository)
    #expect(source["revision"] as? String == qwen36Revision)
    #expect(source["config_sha256"] as? String == qwen36ConfigSHA256)

    // The contract's embedded config must be the same object the public
    // fixture carries; only the surrounding envelope differs.
    let embedded = try #require(contract["config"] as? [String: Any])
    let canonicalEmbedded = try JSONSerialization.data(
        withJSONObject: embedded,
        options: [.sortedKeys, .withoutEscapingSlashes]
    )
    let canonicalFixture = try JSONSerialization.data(
        withJSONObject: try qwen36ConfigObject(),
        options: [.sortedKeys, .withoutEscapingSlashes]
    )
    #expect(canonicalEmbedded == canonicalFixture)
}

/// The `config_sha256` both fixtures pin is the digest of the checkpoint's own
/// `config.json` bytes, so it must equal the line the checked-in reference
/// manifest carries for that file.
///
/// HALF-SUSPENDED SINCE 2026-08-14. The reference manifest was repointed at the
/// adopted Qwen 3.8 backbone and its per-file body is a stub pending
/// generation, so it carries no `config.json` record to compare against. The
/// two THREE-SIX fixtures (fixtures/qwen3_6_27b_config.json and
/// fixtures/qwen3_6_27b_tensor_inventory.json) keep their 3.6 names and their
/// 3.6 content on purpose -- they describe the checkpoint they were captured
/// from -- so the half of this check that compares them to each other still
/// binds and is kept. Only the manifest leg waits.
///
/// The stub is asserted rather than tolerated: if the 3.8 body lands, this test
/// must be restored to a real three-way comparison against the 3.8
/// `config.json` digest, and the two 3.6 fixtures re-captured or explicitly
/// re-scoped in the same commit.
@Test
func qwen36ConfigContractDigestMatchesTheReferenceManifest() throws {
    #expect(try qwen36InventoryFixture().source.configSHA256 == qwen36ConfigSHA256)

    let manifest = try String(contentsOf: qwen36ReferenceManifestURL, encoding: .utf8)
    let entry = manifest
        .split(separator: "\n")
        .map(String.init)
        .first { $0.hasSuffix(" config.json") && !$0.hasPrefix("#") }
    #expect(
        entry == nil,
        """
        the reference manifest has grown a config.json record. The 3.8 manifest \
        body has landed: restore the digest comparison here against the 3.8 \
        config.json, and re-scope the 3.6 config/inventory fixtures in the same \
        commit.
        """
    )
    #expect(manifest.contains("QWEN38-PENDING-RELEASE"))
}

/// The transform's hardcoded inventory is only worth having if it is the real
/// checkpoint's inventory. Compare it, tensor for tensor, against the public
/// header fixture.
@Test
func qwen35CheckpointValidationInventoryMatchesThePublicHeaders() throws {
    let fixture = try qwen36InventoryFixture()
    let expected = Qwen35CheckpointValidation.expectedTensorInventory()
    #expect(expected.count == 1_847)

    let actual = Dictionary(
        uniqueKeysWithValues: fixture.tensors
            .filter { SwiftTransform.isSelectedTextTowerKey($0.name, family: .qwen35) }
            .map { ($0.name, $0) }
    )
    #expect(Set(expected.keys) == Set(actual.keys))
    for (name, metadata) in expected.sorted(by: { $0.key < $1.key }) {
        let record = try #require(actual[name], Comment(rawValue: name))
        #expect(record.dtype == metadata.dtype, Comment(rawValue: name))
        #expect(record.shape == metadata.shape, Comment(rawValue: name))
    }

    // Per-layer counts the runtime loader independently pins.
    #expect(
        expected.keys.filter { !$0.contains(".layers.") }.count
            == Qwen35WeightLoader.requiredTopLevelTensorCount
    )
    for layerIndex in 0..<MLXFastConstants.numHiddenLayers {
        let prefix = "language_model.model.layers.\(layerIndex)."
        let count = expected.keys.filter { $0.hasPrefix(prefix) }.count
        let wanted = layerIndex % 4 == 3
            ? Qwen35WeightLoader.requiredFullAttentionLayerTensorCount
            : Qwen35WeightLoader.requiredLinearLayerTensorCount
        #expect(count == wanted, "layer \(layerIndex)")
    }
}

@Test
func qwen35TransformAcceptsTheExactPublicTextTower() throws {
    let fixture = try qwen36InventoryFixture()
    let metadata = qwen36ValidationMetadata(
        records: fixture.tensors,
        shards: fixture.shards
    )
    #expect(metadata.selectedKeys.count == 1_847)
    try Qwen35CheckpointValidation.validateSelectedTensors(
        selectedKeys: metadata.selectedKeys,
        index: metadata.index,
        headers: metadata.headers,
        quantization: try exactQwen36Quantization()
    )
}

@Test
func qwen35TransformRejectsInventoryAndPackingDrift() throws {
    let fixture = try qwen36InventoryFixture()
    let quantization = try exactQwen36Quantization()

    func expectRejected(
        _ name: String,
        _ mutate: (inout [Qwen36TensorRecord]) -> Void
    ) throws {
        var records = fixture.tensors
        mutate(&records)
        let metadata = qwen36ValidationMetadata(records: records, shards: fixture.shards)
        #expect(throws: MLXFastError.self, "case \(name)") {
            try Qwen35CheckpointValidation.validateSelectedTensors(
                selectedKeys: metadata.selectedKeys,
                index: metadata.index,
                headers: metadata.headers,
                quantization: quantization
            )
        }
    }

    let template = try #require(
        fixture.tensors.first { $0.name == "language_model.model.norm.weight" }
    )
    func record(
        _ name: String,
        dtype: String? = nil,
        shape: [Int]? = nil
    ) -> Qwen36TensorRecord {
        var copy = template
        copy.name = name
        if let dtype { copy.dtype = dtype }
        if let shape { copy.shape = shape }
        return copy
    }

    try expectRejected("missing-tensor") { records in
        records.removeAll { $0.name == "language_model.model.norm.weight" }
    }
    try expectRejected("extra-text-tensor") { records in
        records.append(record("language_model.model.unexpected.weight"))
    }
    try expectRejected("wrong-shape") { records in
        for index in records.indices
        where records[index].name == "language_model.model.norm.weight" {
            records[index].shape = [MLXFastConstants.hiddenSize + 1]
        }
    }
    try expectRejected("wrong-dtype") { records in
        for index in records.indices
        where records[index].name == "language_model.model.layers.3.self_attn.q_proj.weight" {
            records[index].dtype = "BF16"
        }
    }
    try expectRejected("missing-affine-biases") { records in
        records.removeAll { $0.name == "language_model.lm_head.biases" }
    }
    try expectRejected("mtp-head-smuggled-in") { records in
        records.append(record("language_model.model.mtp.layers.0.norm.weight"))
    }
    try expectRejected("compressed-tensors-alias") { records in
        records.append(
            record("language_model.model.layers.3.self_attn.q_proj.weight_packed")
        )
    }
    try expectRejected("fp8-kv-scale") { records in
        records.append(record("language_model.model.layers.3.self_attn.k_scale"))
    }
    try expectRejected("repacked-at-a-different-group-size") { records in
        // group_size 32 would double the scale columns; the packed U32 width
        // no longer matches group_count * group_size * bits / 32.
        for index in records.indices
        where records[index].name.hasPrefix("language_model.lm_head.")
            && records[index].name.hasSuffix("s")
        {
            records[index].shape = [MLXFastConstants.vocabSize, 160]
        }
    }
}

@Test
func qwen35TransformRejectsAlternateQuantizationSpecs() throws {
    var root = try qwen36ConfigObject()

    for (name, block) in [
        ("bits", ["group_size": 64, "bits": 8, "mode": "affine"] as [String: Any]),
        ("group", ["group_size": 32, "bits": 4, "mode": "affine"]),
        ("mode", ["group_size": 64, "bits": 4, "mode": "nvfp4"]),
        ("extra-field", ["group_size": 64, "bits": 4, "mode": "affine", "skip_modules": []]),
        ("missing-mode", ["group_size": 64, "bits": 4]),
    ] {
        var mutated = root
        mutated["quantization"] = block
        mutated["quantization_config"] = block
        #expect(throws: MLXFastError.self, "case \(name)") {
            _ = try Qwen35CheckpointValidation.quantizationSpec(fromConfigRoot: mutated)
        }
    }

    // Present but disagreeing blocks are rejected rather than silently
    // preferring one of them.
    root["quantization_config"] = ["group_size": 32, "bits": 4, "mode": "affine"]
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35CheckpointValidation.quantizationSpec(fromConfigRoot: root)
    }

    var neither = try qwen36ConfigObject()
    neither.removeValue(forKey: "quantization")
    neither.removeValue(forKey: "quantization_config")
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35CheckpointValidation.quantizationSpec(fromConfigRoot: neither)
    }
}
