import CryptoKit
import Foundation
import MLXFastCore
import Testing
@testable import MLXFastModel
@testable import MLXFastRuntimeWorkerSupport
@testable import MLXFastTransform

@Test
func poolsideLagunaPublicConfigFixturePinsExactArtifactSemantics() throws {
    let data = try poolsideLagunaConfigData()
    let root = try poolsideLagunaConfigObject()
    let expectedKeys: Set<String> = [
        "architectures",
        "attention_bias",
        "attention_dropout",
        "auto_map",
        "bos_token_id",
        "decoder_sparse_step",
        "eos_token_id",
        "gating",
        "gating_types",
        "head_dim",
        "hidden_size",
        "intermediate_size",
        "layer_types",
        "max_position_embeddings",
        "mlp_layer_types",
        "mlp_only_layers",
        "model_type",
        "moe_apply_router_weight_on_input",
        "moe_intermediate_size",
        "moe_routed_scaling_factor",
        "norm_topk_prob",
        "num_attention_heads",
        "num_attention_heads_per_layer",
        "num_experts",
        "num_experts_per_tok",
        "num_hidden_layers",
        "num_key_value_heads",
        "pad_token_id",
        "quantization",
        "quantization_config",
        "rms_norm_eps",
        "rope_parameters",
        "router_aux_loss_coef",
        "shared_expert_intermediate_size",
        "sliding_window",
        "tie_word_embeddings",
        "torch_dtype",
        "use_cache",
        "vocab_size",
    ]

    #expect(data.count < 10_000)
    #expect(Set(root.keys) == expectedKeys)
    #expect(root["model_type"] as? String == "laguna")
    #expect(root["max_position_embeddings"] as? Int == 262_144)
    #expect(root["rms_norm_eps"] as? Double == 1e-6)
    #expect(root["num_experts_per_tok"] as? Int == 8)
    #expect(root["norm_topk_prob"] as? Bool == true)
    #expect(root["moe_routed_scaling_factor"] as? Double == 2.5)
    #expect(root["moe_apply_router_weight_on_input"] as? Bool == false)
    #expect(root["use_cache"] as? Bool == true)

    let expectedAttentionSchedule = (0..<40).map {
        $0.isMultiple(of: 4) ? "full_attention" : "sliding_attention"
    }
    let expectedHeadSchedule = (0..<40).map { $0.isMultiple(of: 4) ? 48 : 64 }
    let expectedMLPSchedule = (0..<40).map { $0 == 0 ? "dense" : "sparse" }
    #expect(root["layer_types"] as? [String] == expectedAttentionSchedule)
    #expect(root["num_attention_heads_per_layer"] as? [Int] == expectedHeadSchedule)
    #expect(root["mlp_layer_types"] as? [String] == expectedMLPSchedule)
    #expect(root["gating_types"] as? [String] == Array(repeating: "per_head", count: 40))

    let rope = try #require(root["rope_parameters"] as? [String: Any])
    let full = try #require(rope["full_attention"] as? [String: Any])
    let sliding = try #require(rope["sliding_attention"] as? [String: Any])
    #expect(full["rope_type"] as? String == "yarn")
    #expect(full["rope_theta"] as? Double == 500_000)
    #expect(full["factor"] as? Double == 32)
    #expect(full["original_max_position_embeddings"] as? Int == 8_192)
    #expect(full["beta_fast"] as? Double == 64)
    #expect(full["beta_slow"] as? Double == 1)
    #expect(full["attention_factor"] as? Double == 1)
    #expect(full["partial_rotary_factor"] as? Double == 0.5)
    #expect(sliding["rope_type"] as? String == "default")
    #expect(sliding["rope_theta"] as? Double == 10_000)
    #expect(sliding["partial_rotary_factor"] as? Double == 1)

    // The public artifact explicitly carries these zero values, while qkv
    // bias and router softcapping are absent. JSON null is neither state.
    #expect(root.keys.contains("attention_dropout"))
    #expect(root["attention_dropout"] as? Double == 0)
    #expect(root.keys.contains("router_aux_loss_coef"))
    #expect(root["router_aux_loss_coef"] as? Double == 0)
    #expect(!root.keys.contains("qkv_bias"))
    #expect(!root.keys.contains("moe_router_logit_softcapping"))

    let canonical = try canonicalConfigData(root)
    var changed = root
    changed["attention_dropout"] = NSNull()
    #expect(try canonicalConfigData(changed) != canonical)
    changed = root
    changed["qkv_bias"] = NSNull()
    #expect(try canonicalConfigData(changed) != canonical)
    changed = root
    changed["moe_router_logit_softcapping"] = 0.0
    #expect(try canonicalConfigData(changed) != canonical)

    let parsed = try loadPoolsideLagunaConfig(root)
    #expect(parsed.gatingTypes == Array(repeating: .perHead, count: 40))
    #expect(parsed.mlpOnlyLayers == [0])
    #expect(parsed.decoderSparseStep == 1)
    #expect(!parsed.moeApplyRouterWeightOnInput)
    #expect(parsed.routerAuxLossCoef == 0)
    #expect(parsed.useCache)
    #expect(parsed.fullRope.attentionFactor == 1)

    // The runtime worker's pinned-config gate no longer accepts this artifact:
    // it guards the Qwen 3.6 tower the harness now drives. Assert the negative
    // rather than dropping the coverage -- a gate that accepted BOTH towers
    // would be a gate that pins neither.
    #expect(throws: MLXFastError.self) {
        try validateRuntimeWorkerPinnedConfigurationData(data)
    }
}

/// Behaviour-bearing mutations of the Poolside artifact must not parse as the
/// pinned Laguna config.
///
/// This used to drive the runtime worker's pinned-config gate. That gate now
/// guards the Qwen 3.6 artifact and rejects every Laguna config -- mutated or
/// not -- so it can no longer tell these mutations apart from the unmutated
/// baseline. `LagunaConfig`'s own frozen-invariant check is the instrument that
/// still discriminates, so the loop drives that instead.
@Test
func poolsideLagunaConfigRejectsEveryBehaviorBearingMutation() throws {
    typealias Mutation = (String, (inout [String: Any]) -> Void)
    let mutations: [Mutation] = [
        ("model type", { $0["model_type"] = "gemma4_text" }),
        ("vocabulary", { $0["vocab_size"] = 262_144 }),
        ("hidden size", { $0["hidden_size"] = 4_096 }),
        ("dense intermediate", { $0["intermediate_size"] = 16_384 }),
        ("layer count", { $0["num_hidden_layers"] = 48 }),
        ("fallback attention heads", { $0["num_attention_heads"] = 64 }),
        ("head schedule", {
            var values = $0["num_attention_heads_per_layer"] as! [Int]
            values[1] = 48
            $0["num_attention_heads_per_layer"] = values
        }),
        ("KV heads", { $0["num_key_value_heads"] = 16 }),
        ("head dimension", { $0["head_dim"] = 256 }),
        ("RMS epsilon", { $0["rms_norm_eps"] = 1e-5 }),
        ("maximum position", { $0["max_position_embeddings"] = 1_048_576 }),
        ("attention bias", { $0["attention_bias"] = true }),
        ("QKV bias", { $0["qkv_bias"] = true }),
        ("attention dropout", { $0["attention_dropout"] = 0.1 }),
        ("sliding window", { $0["sliding_window"] = 1_024 }),
        ("attention schedule", {
            var values = $0["layer_types"] as! [String]
            values.swapAt(0, 1)
            $0["layer_types"] = values
        }),
        ("MLP schedule", {
            var values = $0["mlp_layer_types"] as! [String]
            values[1] = "dense"
            $0["mlp_layer_types"] = values
        }),
        ("MLP-only layers", { $0["mlp_only_layers"] = [0, 1] }),
        ("sparse step", { $0["decoder_sparse_step"] = 2 }),
        ("global gating", { $0["gating"] = "per-element" }),
        ("per-layer gating", {
            var values = $0["gating_types"] as! [String]
            values[7] = "per_element"
            $0["gating_types"] = values
        }),
        ("tied embeddings", { $0["tie_word_embeddings"] = true }),
        ("expert count", { $0["num_experts"] = 128 }),
        ("router top-k", { $0["num_experts_per_tok"] = 10 }),
        ("expert intermediate", { $0["moe_intermediate_size"] = 1_024 }),
        ("shared intermediate", { $0["shared_expert_intermediate_size"] = 1_024 }),
        ("routed factor", { $0["moe_routed_scaling_factor"] = 1.0 }),
        ("top-k normalization", { $0["norm_topk_prob"] = false }),
        ("router weight placement", { $0["moe_apply_router_weight_on_input"] = true }),
        ("router softcap", { $0["moe_router_logit_softcapping"] = 30.0 }),
        ("router auxiliary coefficient", { $0["router_aux_loss_coef"] = 0.01 }),
        ("cache behavior", { $0["use_cache"] = false }),
        ropeMutation("sliding theta") { $0["rope_theta"] = 20_000.0 },
        ropeMutation("sliding type") { $0["rope_type"] = "yarn" },
        ropeMutation("sliding rotary fraction") { $0["partial_rotary_factor"] = 0.5 },
        ropeMutation("full theta", attention: "full_attention") {
            $0["rope_theta"] = 10_000.0
        },
        ropeMutation("full type", attention: "full_attention") {
            $0["rope_type"] = "default"
        },
        ropeMutation("YaRN factor", attention: "full_attention") { $0["factor"] = 8.0 },
        ropeMutation("YaRN original context", attention: "full_attention") {
            $0["original_max_position_embeddings"] = 4_096
        },
        ropeMutation("YaRN beta fast", attention: "full_attention") {
            $0["beta_fast"] = 32.0
        },
        ropeMutation("YaRN beta slow", attention: "full_attention") {
            $0["beta_slow"] = 2.0
        },
        ropeMutation("YaRN attention factor", attention: "full_attention") {
            $0["attention_factor"] = 0.5
        },
        ropeMutation("YaRN explicit mscale", attention: "full_attention") {
            $0["mscale"] = 1.0
        },
        ropeMutation("YaRN explicit all-dimension mscale", attention: "full_attention") {
            $0["mscale_all_dim"] = 1.0
        },
        ropeMutation("full rotary fraction", attention: "full_attention") {
            $0["partial_rotary_factor"] = 1.0
        },
        quantizationMutation("quantization bits", block: "quantization") {
            $0["bits"] = 8
        },
        quantizationMutation("quantization group", block: "quantization_config") {
            $0["group_size"] = 32
        },
        quantizationMutation("quantization mode", block: "quantization") {
            $0["mode"] = "affine"
        },
    ]

    for (name, mutate) in mutations {
        var object = try poolsideLagunaConfigObject()
        mutate(&object)
        #expect(throws: MLXFastError.self, "mutation: \(name)") {
            _ = try loadPoolsideLagunaConfig(object)
        }
    }
}

@Test
func runtimeWorkerRejectsPublicContractFieldsPreviouslyIgnoredByDecoder() throws {
    typealias Mutation = (String, (inout [String: Any]) -> Void)
    let mutations: [Mutation] = [
        ("per-layer gating", {
            var values = $0["gating_types"] as! [String]
            values[0] = "per_element"
            $0["gating_types"] = values
        }),
        ("router weight placement", { $0["moe_apply_router_weight_on_input"] = true }),
        ("router auxiliary coefficient", { $0["router_aux_loss_coef"] = 0.1 }),
        ("cache behavior", { $0["use_cache"] = false }),
        ropeMutation("YaRN attention factor", attention: "full_attention") {
            $0["attention_factor"] = 0.5
        },
        ropeMutation("YaRN explicit mscale", attention: "full_attention") {
            $0["mscale"] = 1.0
        },
        ("accidental Laguna S geometry", {
            $0["num_hidden_layers"] = 48
            $0["num_experts_per_tok"] = 10
            $0["max_position_embeddings"] = 1_048_576
        }),
    ]

    for (name, mutate) in mutations {
        var object = try poolsideLagunaConfigObject()
        mutate(&object)
        #expect(throws: MLXFastError.self, "mutation: \(name)") {
            _ = try loadPoolsideLagunaConfig(object)
        }
    }
}

@Test
func poolsideLagunaTensorInventoryFixturePinsAllPublicHeaders() throws {
    let fixtureData = try Data(contentsOf: poolsideLagunaInventoryFixtureURL)
    let fixture = try poolsideLagunaInventoryFixture()
    let recordsByName = Dictionary(uniqueKeysWithValues: fixture.tensors.map {
        ($0.name, $0)
    })

    #expect(fixtureData.count < 100_000)
    #expect(fixture.schemaVersion == 1)
    #expect(fixture.source.repository == poolsideLagunaRepository)
    #expect(fixture.source.revision == poolsideLagunaRevision)
    #expect(
        fixture.source.configSHA256
            == "ef1eaf6709b38ab58b47480fdf8b2732163b34427d4a0ec334f77af27e1ce3b9"
    )
    #expect(
        fixture.source.indexSHA256
            == "dd32fc37f3d8638b6a375c1be38fafb4cd4a8f1c4cbdcf70bd0a117fda4eb13d"
    )
    #expect(fixture.source.indexTotalSize == 21_561_408_512)
    #expect(fixture.source.indexTotalParameters == 33_442_617_088)
    #expect(fixture.summary.tensorCount == 912)
    #expect(fixture.tensors.count == 912)
    #expect(recordsByName.count == 912)
    #expect(fixture.tensors.map(\.name) == fixture.tensors.map(\.name).sorted())

    let dtypeCounts = Dictionary(grouping: fixture.tensors, by: \.dtype)
        .mapValues(\.count)
    let expectedCounts = ["BF16": 405, "F32": 39, "U32": 234, "U8": 234]
    #expect(dtypeCounts == expectedCounts)
    #expect(fixture.summary.dtypeCounts == expectedCounts)

    let canonical = poolsideLagunaCanonicalInventoryData(
        fixture.tensors,
        shards: fixture.shards
    )
    #expect(
        poolsideLagunaSHA256(canonical)
            == "6da902713b7348d11f3b9664b2ad64703c2544eae6d028133f3d2c45be926848"
    )
    #expect(poolsideLagunaSHA256(canonical) == fixture.summary.canonicalSHA256)

    #expect(fixture.shards.map(\.tensorCount) == [223, 236, 230, 222, 1])
    for (index, shard) in fixture.shards.enumerated() {
        let shardRecords = fixture.tensors.filter { $0.shard == index + 1 }
        #expect(shardRecords.count == shard.tensorCount)
        #expect(
            Dictionary(grouping: shardRecords, by: \.dtype).mapValues(\.count)
                == shard.dtypeCounts
        )
        let shardCanonical = poolsideLagunaCanonicalInventoryData(
            shardRecords,
            shards: fixture.shards
        )
        #expect(poolsideLagunaSHA256(shardCanonical) == shard.canonicalSHA256)
        #expect(shard.headerSHA256.count == 64)
    }

    for representative in fixture.representative {
        #expect(recordsByName[representative.name] == representative)
    }
    #expect(
        recordsByName["model.layers.1.mlp.switch_mlp.gate_proj.weight"]
            == PoolsideLagunaTensorRecord(
                name: "model.layers.1.mlp.switch_mlp.gate_proj.weight",
                dtype: "U32",
                shape: [256, 512, 256],
                shard: 1
            )
    )
    #expect(
        recordsByName["model.layers.39.mlp.shared_expert.down_proj.scales"]?.shape
            == [2_048, 32]
    )
    #expect(!recordsByName.keys.contains { $0.contains(".experts.") })
    #expect(!recordsByName.keys.contains { $0.hasSuffix(".biases") })
    #expect(!recordsByName.keys.contains { $0.contains("weight_packed") })
    #expect(!recordsByName.keys.contains { $0.contains("global_scale") })
    #expect(!recordsByName.keys.contains { $0.contains("kv_scale") })
}

@Test
func checkpointValidatorAcceptsExactPublic912TensorInventory() throws {
    let fixture = try poolsideLagunaInventoryFixture()
    let metadata = poolsideLagunaValidationMetadata(
        records: fixture.tensors,
        shards: fixture.shards
    )
    let quantization = try exactPoolsideLagunaQuantization()

    try LagunaCheckpointValidation.validateSelectedTensors(
        selectedKeys: metadata.selectedKeys,
        index: metadata.index,
        headers: metadata.headers,
        quantization: quantization
    )

    let expected = LagunaCheckpointValidation.expectedTensorInventory()
    #expect(expected.count == 912)
    for record in fixture.tensors {
        #expect(expected[record.name]?.dtype == record.dtype)
        #expect(expected[record.name]?.shape == record.shape)
    }
}

@Test
func checkpointValidatorRejectsForeignSchemasAndEveryInventoryMutation() throws {
    let fixture = try poolsideLagunaInventoryFixture()
    let quantization = try exactPoolsideLagunaQuantization()

    func expectRejected(
        _ name: String,
        mutate: (inout [PoolsideLagunaTensorRecord]) -> Void
    ) {
        var records = fixture.tensors
        mutate(&records)
        let metadata = poolsideLagunaValidationMetadata(
            records: records,
            shards: fixture.shards
        )
        #expect(throws: MLXFastError.self, "mutation: \(name)") {
            try LagunaCheckpointValidation.validateSelectedTensors(
                selectedKeys: metadata.selectedKeys,
                index: metadata.index,
                headers: metadata.headers,
                quantization: quantization
            )
        }
    }

    expectRejected("affine biases") {
        $0.append(
            PoolsideLagunaTensorRecord(
                name: "model.layers.1.mlp.switch_mlp.gate_proj.biases",
                dtype: "U8",
                shape: [256, 512, 128],
                shard: 1
            )
        )
    }
    expectRejected("compressed-tensors weight_packed") {
        $0.append(
            PoolsideLagunaTensorRecord(
                name: "model.layers.1.mlp.switch_mlp.gate_proj.weight_packed",
                dtype: "U32",
                shape: [256, 512, 256],
                shard: 1
            )
        )
    }
    expectRejected("compressed-tensors global scale") {
        $0.append(
            PoolsideLagunaTensorRecord(
                name: "model.layers.1.mlp.switch_mlp.gate_proj.global_scale",
                dtype: "F32",
                shape: [1],
                shard: 1
            )
        )
    }
    expectRejected("compressed-tensors KV scale") {
        $0.append(
            PoolsideLagunaTensorRecord(
                name: "model.layers.1.self_attn.k_proj.kv_scale",
                dtype: "F32",
                shape: [1],
                shard: 1
            )
        )
    }
    expectRejected("per-expert unstacked namespace") {
        $0.append(
            PoolsideLagunaTensorRecord(
                name: "model.layers.1.mlp.switch_mlp.experts.0.gate_proj.weight",
                dtype: "U32",
                shape: [512, 256],
                shard: 1
            )
        )
    }
    expectRejected("missing tensor") {
        $0.removeAll { $0.name == "lm_head.weight" }
    }
    expectRejected("extra tensor") {
        $0.append(
            PoolsideLagunaTensorRecord(
                name: "model.layers.40.input_layernorm.weight",
                dtype: "BF16",
                shape: [2_048],
                shard: 5
            )
        )
    }
    expectRejected("wrong dtype") {
        let index = $0.firstIndex { $0.name == "model.layers.7.mlp.gate.weight" }!
        $0[index].dtype = "F16"
    }
    expectRejected("wrong shape") {
        let index = $0.firstIndex { $0.name == "model.layers.4.self_attn.q_proj.weight" }!
        $0[index].shape = [8_192, 2_048]
    }
}

@Test
func compressedTensorQuantizationSchemasAndWrongGroupSizeAreRejected() throws {
    for foreignKey in ["weight_packed", "global_scale", "kv_scale"] {
        var root = try poolsideLagunaConfigObject()
        var block = root["quantization_config"] as! [String: Any]
        block[foreignKey] = true
        root["quantization_config"] = block
        #expect(throws: MLXFastError.self, "foreign key: \(foreignKey)") {
            _ = try LagunaCheckpointValidation.quantizationSpec(fromConfigRoot: root)
        }
    }

    var wrongGroup = try poolsideLagunaConfigObject()
    for name in ["quantization", "quantization_config"] {
        var block = wrongGroup[name] as! [String: Any]
        block["group_size"] = 32
        wrongGroup[name] = block
    }
    #expect(throws: MLXFastError.self) {
        _ = try LagunaCheckpointValidation.quantizationSpec(fromConfigRoot: wrongGroup)
    }

    var compressedTensors = try poolsideLagunaConfigObject()
    compressedTensors["quantization_config"] = [
        "format": "pack-quantized",
        "config_groups": [
            "group_0": [
                "weights": [
                    "num_bits": 4,
                    "strategy": "group",
                    "group_size": 16,
                ]
            ]
        ],
        "kv_cache_scheme": ["num_bits": 8],
    ]
    #expect(throws: MLXFastError.self) {
        _ = try LagunaCheckpointValidation.quantizationSpec(fromConfigRoot: compressedTensors)
    }
}

@Test
func mlxSwiftLMYaRNAuthorityKeepsDerivedMScale() throws {
    let config = try loadPoolsideLagunaConfig(try poolsideLagunaConfigObject())
    let serializedAttentionFactor = try #require(config.fullRope.attentionFactor)
    #expect(serializedAttentionFactor == 1)
    let runtimeScaling = lagunaRopeScalingConfig(config.fullRope)
    #expect(runtimeScaling["factor"] != nil)
    #expect(runtimeScaling["attention_factor"] == nil)
    #expect(runtimeScaling["mscale"] == nil)
    #expect(runtimeScaling["mscale_all_dim"] == nil)

    // Pin the vendored implementation without constructing an MLX array (the
    // default unit-test build intentionally has no mlx.metallib). Laguna
    // passes the full scaling dictionary to initializeRope; mlx-swift-lm reads
    // `mscale`, not Hugging Face's `attention_factor`, then applies this
    // formula. That implementation is the runtime authority.
    let repositoryRoot = poolsideLagunaConfigFixtureURL
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let ropeSource = try String(
        contentsOf: repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/RoPEUtils.swift"
        ),
        encoding: .utf8
    )
    #expect(
        ropeSource.contains(
            #"let mscale = scalingConfig?["mscale"]?.asFloat() ?? 1.0"#
        )
    )
    #expect(
        ropeSource.contains(
            "return 0.1 * mscale * log(scale) + 1.0"
        )
    )
    #expect(!ropeSource.contains(#""attention_factor""#))

    let derivedMScale = 1 + 0.1 * log(Float(config.fullRope.factor))
    #expect(abs(derivedMScale - 1.3465736) < 1e-6)
    #expect(derivedMScale != Float(serializedAttentionFactor))
}

private func canonicalConfigData(_ object: [String: Any]) throws -> Data {
    try JSONSerialization.data(
        withJSONObject: object,
        options: [.sortedKeys, .withoutEscapingSlashes]
    )
}

private func ropeMutation(
    _ name: String,
    attention: String = "sliding_attention",
    _ mutate: @escaping (inout [String: Any]) -> Void
) -> (String, (inout [String: Any]) -> Void) {
    (
        name,
        { root in
            var rope = root["rope_parameters"] as! [String: Any]
            var block = rope[attention] as! [String: Any]
            mutate(&block)
            rope[attention] = block
            root["rope_parameters"] = rope
        }
    )
}

private func quantizationMutation(
    _ name: String,
    block: String,
    _ mutate: @escaping (inout [String: Any]) -> Void
) -> (String, (inout [String: Any]) -> Void) {
    (
        name,
        { root in
            var values = root[block] as! [String: Any]
            mutate(&values)
            root[block] = values
        }
    )
}
