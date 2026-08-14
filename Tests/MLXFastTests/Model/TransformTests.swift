import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastHarness
@testable import MLXFastTransform

@Test
func transformSelectsTextTowerTensorsAndDropsVisionTensors() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)

    try gemmaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try #"{"tokenizer":"fixture"}"#.write(
        to: reference.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.layers.0.self_attn.q_proj.weight"
    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let embedVisionName = "embed_vision.embedding_projection.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [4], data: Data([1, 2, 3, 4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [3], data: Data([9, 8, 7])),
            TensorFixture(name: embedVisionName, dtype: "U8", shape: [2], data: Data([5, 6])),
        ]
    )
    try """
    {
      "metadata": {"total_size": 9},
      "weight_map": {
        "\(textName)": "\(shardName)",
        "\(visionName)": "\(shardName)",
        "\(embedVisionName)": "\(shardName)"
      }
    }
    """.write(
        to: reference.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )

    let report = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    #expect(report.denseTensorCount == 1)
    #expect(report.denseShardCount == 1)
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("config.json").path))
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("tokenizer.json").path))
    #expect(!FileManager.default.fileExists(atPath: output.appendingPathComponent("experts").path))

    let outputShard = output.appendingPathComponent(shardName)
    let outputHeader = try Safetensors.readHeader(outputShard)
    #expect(outputHeader.tensors.keys.sorted() == [textName])
    #expect(try tensorBytes(outputShard, header: outputHeader, name: textName) == Data([1, 2, 3, 4]))

    let strippedIndexData = try Data(
        contentsOf: output.appendingPathComponent("model.safetensors.index.json")
    )
    let strippedIndex = try JSONSerialization.jsonObject(with: strippedIndexData) as? [String: Any]
    let weightMap = try #require(strippedIndex?["weight_map"] as? [String: String])
    #expect(weightMap == [textName: shardName])
    let metadata = try #require(strippedIndex?["metadata"] as? [String: Any])
    #expect(metadata["total_size"] as? Int == 4)
}

@Test
func transformWritesFlattenedRuntimeConfigFromTextConfig() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try gemmaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.embed_tokens.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [TensorFixture(name: textName, dtype: "U8", shape: [1], data: Data([1]))]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [textName: shardName]
    )

    _ = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    let configData = try Data(contentsOf: output.appendingPathComponent("config.json"))
    let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
    #expect(config?["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
    #expect(config?["vocab_size"] as? Int == MLXFastConstants.vocabSize)
    #expect(config?["text_config"] == nil)
    let quantization = try #require(config?["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == 64)
    #expect(quantization["bits"] as? Int == 4)
}

@Test
func transformDetectsModelFamilyFromSourceConfig() throws {
    let laguna = try #require(
        try JSONSerialization.jsonObject(
            with: Data(lagunaReferenceConfigJSON().utf8)
        ) as? [String: Any]
    )
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: laguna) == .laguna)

    let gemma = try #require(
        try JSONSerialization.jsonObject(
            with: Data(gemmaReferenceConfigJSON().utf8)
        ) as? [String: Any]
    )
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: gemma) == .gemma4)

    // Qwen 3.6 carries a `text_config` exactly like the legacy Gemma 4
    // multimodal layout, so detection must discriminate on the `qwen3_5`
    // model-type prefix BEFORE the text_config-means-Gemma fallthrough --
    // otherwise the real checkpoint routes to the Gemma flattening.
    let qwen = try #require(
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: qwen36ConfigFixtureURL)
        ) as? [String: Any]
    )
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: qwen) == .qwen35)
    #expect(qwen["text_config"] != nil)

    // Either level declaring the prefix is sufficient: the top level alone,
    #expect(
        try SwiftTransform.detectModelFamily(
            sourceConfigRoot: ["model_type": "qwen3_5"]
        ) == .qwen35
    )
    // and the nested text tower alone (a language-model-only republish that
    // drops the multimodal wrapper's own model_type).
    #expect(
        try SwiftTransform.detectModelFamily(
            sourceConfigRoot: ["text_config": ["model_type": "qwen3_5_text"]]
        ) == .qwen35
    )
    // A point revision of the same family still routes to .qwen35 rather than
    // falling through to Gemma: the prefix is matched, not the exact string.
    #expect(
        try SwiftTransform.detectModelFamily(
            sourceConfigRoot: [
                "model_type": "qwen3_5_omni",
                "text_config": ["model_type": "qwen3_5_text"],
            ]
        ) == .qwen35
    )
    // But a text_config that is NOT qwen3_5 keeps the legacy Gemma route.
    #expect(
        try SwiftTransform.detectModelFamily(
            sourceConfigRoot: ["text_config": ["model_type": "gemma4_text"]]
        ) == .gemma4
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.detectModelFamily(sourceConfigRoot: ["model_type": "qwen3"])
    }
}

@Test
func transformKeySelectionDropsRotaryInvFreqOnlyForLaguna() {
    let invFreq = "model.layers.3.self_attn.rotary_emb.inv_freq"
    #expect(!SwiftTransform.isSelectedTextTowerKey(invFreq, family: .laguna))
    #expect(
        SwiftTransform.isSelectedTextTowerKey(
            "language_model.model.layers.3.self_attn.rotary_emb.inv_freq",
            family: .gemma4
        )
    )
    #expect(
        SwiftTransform.isSelectedTextTowerKey(
            "model.layers.1.mlp.switch_mlp.gate_proj.weight",
            family: .laguna
        )
    )
    #expect(
        !SwiftTransform.isSelectedTextTowerKey(
            "vision_tower.encoder.layers.0.self_attn.q_proj.weight",
            family: .laguna
        )
    )
}

@Test
func transformBuildsPoolsideLagunaNVFP4RuntimeConfig() throws {
    let configData = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigPath: poolsideLagunaConfigFixtureURL
    )
    let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
    #expect(config?["model_type"] as? String == LagunaConstants.modelType)
    #expect(config?["vocab_size"] as? Int == LagunaConstants.vocabSize)
    #expect(config?["hidden_size"] as? Int == LagunaConstants.hiddenSize)
    #expect(config?["intermediate_size"] as? Int == LagunaConstants.denseIntermediateSize)
    #expect(config?["num_hidden_layers"] as? Int == LagunaConstants.numHiddenLayers)
    #expect(config?["num_experts"] as? Int == LagunaConstants.numExperts)
    #expect(config?["num_experts_per_tok"] as? Int == LagunaConstants.numExpertsPerTok)
    #expect(config?["tie_word_embeddings"] as? Bool == false)
    #expect(config?["gating"] as? String == "per-head")
    #expect(config?["text_config"] == nil)
    #expect(config?["vision_config"] == nil)
    let layerTypes = try #require(config?["layer_types"] as? [String])
    #expect(layerTypes.count == LagunaConstants.numHiddenLayers)
    #expect(layerTypes[0] == "full_attention")
    #expect(layerTypes[1] == "sliding_attention")
    let mlpLayerTypes = try #require(config?["mlp_layer_types"] as? [String])
    #expect(mlpLayerTypes.count == LagunaConstants.numHiddenLayers)
    #expect(mlpLayerTypes.first == "dense")
    #expect(mlpLayerTypes.dropFirst().allSatisfy({ $0 == "sparse" }))

    let quantization = try #require(config?["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == LagunaConstants.quantizationGroupSize)
    #expect(quantization["bits"] as? Int == LagunaConstants.quantizationBits)
    #expect(quantization["mode"] as? String == LagunaConstants.quantizationMode)
    let overrideKeys = quantization.compactMap { key, value in
        value is [String: Any] ? key : nil
    }
    #expect(overrideKeys.isEmpty)
}

@Test
func transformRejectsPartialLagunaInventoryBeforePublishing() throws {
    let fixture = try writeLagunaCheckpointFixture(
        tensors: [
            TensorFixture(
                name: "model.embed_tokens.weight",
                dtype: "U8",
                shape: [1],
                data: Data([1])
            )
        ]
    )
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    var rejection: MLXFastError?
    do {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: fixture.output.path
            )
        )
    } catch let error as MLXFastError {
        rejection = error
    }

    #expect(rejection?.description.contains("exact public 912-tensor contract") == true)
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

@Test
func transformRejectsQuantizedLagunaExpertStackMissingScales() throws {
    let stem = "model.layers.1.mlp.switch_mlp.gate_proj"
    var tensors = nvfp4TensorPair(
        stem: stem,
        leadingShape: [4, 6],
        inputFeatures: 128
    )
    tensors.removeAll { $0.name == "\(stem).scales" }
    let fixture = try writeLagunaCheckpointFixture(tensors: tensors)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    var rejection: MLXFastError?
    do {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: fixture.output.path
            )
        )
    } catch let error as MLXFastError {
        rejection = error
    } catch {
        throw error
    }

    #expect(rejection?.description.contains("U8 scales") == true)
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

@Test
func transformRejectsPoolsideQuantizationOverrides() throws {
    var config = try #require(
        try JSONSerialization.jsonObject(
            with: Data(lagunaReferenceConfigJSON().utf8)
        ) as? [String: Any]
    )
    var quantization = try #require(config["quantization"] as? [String: Any])
    quantization["model.layers.1.mlp.switch_mlp.gate_proj"] = [
        "group_size": 32,
        "bits": 4,
    ]
    config["quantization"] = quantization
    let configData = try JSONSerialization.data(
        withJSONObject: config,
        options: [.sortedKeys]
    )
    let fixture = try writeLagunaCheckpointFixture(
        tensors: [
            bf16Tensor(name: "model.embed_tokens.weight", shape: [1, 1])
        ],
        configJSON: String(decoding: configData, as: UTF8.self)
    )
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    var rejection: MLXFastError?
    do {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: fixture.output.path
            )
        )
    } catch let error as MLXFastError {
        rejection = error
    } catch {
        throw error
    }

    #expect(rejection?.description.contains("unsupported fields") == true)
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

@Test
func transformRejectsNonMLXCompressedAndGlobalScaleSchemas() throws {
    for forbiddenName in [
        "model.layers.1.mlp.switch_mlp.gate_proj.weight_packed",
        "model.layers.1.mlp.switch_mlp.gate_proj.input_global_scale",
        "model.layers.1.mlp.switch_mlp.gate_proj.weight_global_scale",
        "model.layers.1.self_attn.k_scale",
        "model.layers.1.self_attn.v_scale",
        "model.layers.1.mlp.switch_mlp.gate_proj.biases",
    ] {
        let fixture = try writeLagunaCheckpointFixture(
            tensors: [
                TensorFixture(
                    name: forbiddenName,
                    dtype: "U8",
                    shape: [1],
                    data: Data([0])
                )
            ]
        )
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        var rejection: MLXFastError?
        do {
            _ = try SwiftTransform.run(
                TransformOptions(
                    referencePath: fixture.reference.path,
                    outputPath: fixture.output.path
                )
            )
        } catch let error as MLXFastError {
            rejection = error
        }
        #expect(
            rejection?.description.contains("rejects compressed-tensors") == true,
            Comment(rawValue: forbiddenName)
        )
        #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
    }
}

@Test
func transformRequiresMatchingExplicitPoolsideQuantizationBlocks() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let configPath = root.appendingPathComponent("config.json")

    func expectRejected(_ config: [String: Any]) throws {
        let data = try JSONSerialization.data(
            withJSONObject: config,
            options: [.sortedKeys]
        )
        try data.write(to: configPath)
        #expect(throws: MLXFastError.self) {
            _ = try SwiftTransform.makeRuntimeConfigData(
                sourceConfigPath: configPath
            )
        }
    }

    var config = try #require(
        try JSONSerialization.jsonObject(
            with: Data(lagunaReferenceConfigJSON().utf8)
        ) as? [String: Any]
    )
    config.removeValue(forKey: "quantization_config")
    try expectRejected(config)

    config = try #require(
        try JSONSerialization.jsonObject(
            with: Data(lagunaReferenceConfigJSON().utf8)
        ) as? [String: Any]
    )
    var quantizationConfig = try #require(
        config["quantization_config"] as? [String: Any]
    )
    quantizationConfig["bits"] = 8
    config["quantization_config"] = quantizationConfig
    try expectRejected(config)

    config = try #require(
        try JSONSerialization.jsonObject(
            with: Data(lagunaReferenceConfigJSON().utf8)
        ) as? [String: Any]
    )
    var quantization = try #require(config["quantization"] as? [String: Any])
    quantization.removeValue(forKey: "mode")
    config["quantization"] = quantization
    try expectRejected(config)
}

@Test
func transformRejectsPoolsideLagunaRouterMissingCorrectionBias() throws {
    let tensors = [
        bf16Tensor(name: "model.layers.1.mlp.gate.weight", shape: [4, 128])
    ]
    let fixture = try writeLagunaCheckpointFixture(tensors: tensors)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    var rejection: MLXFastError?
    do {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: fixture.output.path
            )
        )
    } catch let error as MLXFastError {
        rejection = error
    } catch {
        throw error
    }

    #expect(rejection?.description.contains("missing its e_score_correction_bias") == true)
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

@Test
func transformRejectsLagunaRouterAndStackedExpertCountMismatch() throws {
    var tensors = [
        bf16Tensor(name: "model.layers.1.mlp.gate.weight", shape: [4, 128])
    ]
    tensors.append(
        f32Tensor(
            name: "model.layers.1.mlp.gate.e_score_correction_bias",
            shape: [4]
        )
    )
    tensors += nvfp4TensorPair(
        stem: "model.layers.1.mlp.switch_mlp.gate_proj",
        leadingShape: [3, 6],
        inputFeatures: 128
    )
    let fixture = try writeLagunaCheckpointFixture(tensors: tensors)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    var rejection: MLXFastError?
    do {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: fixture.output.path
            )
        )
    } catch let error as MLXFastError {
        rejection = error
    } catch {
        throw error
    }

    #expect(rejection?.description.contains("do not match the stacked expert count") == true)
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

@Test
func transformRuntimeConfigCaptureDoesNotRereadChangedSource() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let configPath = root.appendingPathComponent("config.json")
    try lagunaReferenceConfigJSON().write(
        to: configPath,
        atomically: true,
        encoding: .utf8
    )

    let captured = try SwiftTransform.makeRuntimeConfigData(sourceConfigPath: configPath)
    try #"{"text_config":{"num_hidden_layers":1}}"#.write(
        to: configPath,
        atomically: true,
        encoding: .utf8
    )

    let object = try JSONSerialization.jsonObject(with: captured) as? [String: Any]
    #expect(object?["num_hidden_layers"] as? Int == LagunaConstants.numHiddenLayers)
}

@Test
func transformRejectsSourceMetadataMutationsBeforePublishingOutput() throws {
    enum Mutation: CaseIterable {
        case config
        case index
        case tokenizer
    }

    for mutation in Mutation.allCases {
        let expectedError: String
        switch mutation {
        case .config:
            expectedError = "reference config changed while transform was running"
        case .index:
            expectedError = "checkpoint index changed while transform was running"
        case .tokenizer:
            expectedError = "reference tokenizer metadata changed while transform was running"
        }
        let fixture = try writeTransformFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try FileManager.default.createDirectory(
            at: fixture.output,
            withIntermediateDirectories: true
        )
        let sentinel = fixture.output.appendingPathComponent("sentinel.txt")
        try "preserve \(mutation)".write(to: sentinel, atomically: true, encoding: .utf8)

        var rejection: MLXFastError?
        do {
            _ = try SwiftTransform.run(
                TransformOptions(
                    referencePath: fixture.reference.path,
                    outputPath: fixture.output.path
                ),
                beforeSourceRevalidation: {
                    switch mutation {
                    case .config:
                        try #"{"text_config":{"num_hidden_layers":1}}"#.write(
                            to: fixture.reference.appendingPathComponent("config.json"),
                            atomically: true,
                            encoding: .utf8
                        )
                    case .index:
                        try writeCheckpointIndex(
                            fixture.reference.appendingPathComponent(
                                "model.safetensors.index.json"
                            ),
                            weightMap: [
                                "model.layers.0.self_attn.q_proj.weight":
                                    "model-00001-of-00001.safetensors",
                            ],
                            metadata: ["generation": 2]
                        )
                    case .tokenizer:
                        try #"{"tokenizer":"changed"}"#.write(
                            to: fixture.reference.appendingPathComponent("tokenizer.json"),
                            atomically: true,
                            encoding: .utf8
                        )
                    }
                }
            )
        } catch let error as MLXFastError {
            rejection = error
        } catch {
            throw error
        }
        #expect(rejection?.description == expectedError, "mutation: \(mutation)")

        #expect(
            try String(contentsOf: sentinel, encoding: .utf8) == "preserve \(mutation)",
            "mutation: \(mutation)"
        )
        #expect(
            try transformStagingDirectories(nextTo: fixture.output).isEmpty,
            "mutation: \(mutation)"
        )
    }
}

@Test
func transformRejectsSymlinkedTokenizerMetadataAndPreservesOutput() throws {
    let fixture = try writeTransformFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let tokenizer = fixture.reference.appendingPathComponent("tokenizer.json")
    let tokenizerTarget = fixture.root.appendingPathComponent("tokenizer-target.json")
    try #"{"tokenizer":"external"}"#.write(
        to: tokenizerTarget,
        atomically: true,
        encoding: .utf8
    )
    try FileManager.default.removeItem(at: tokenizer)
    try FileManager.default.createSymbolicLink(at: tokenizer, withDestinationURL: tokenizerTarget)

    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let sentinel = fixture.output.appendingPathComponent("sentinel.txt")
    try "preserve".write(to: sentinel, atomically: true, encoding: .utf8)

    var rejection: MLXFastError?
    do {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
        )
    } catch let error as MLXFastError {
        rejection = error
    } catch {
        throw error
    }

    #expect(rejection?.description.contains("reference metadata is not a regular file") == true)
    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "preserve")
    #expect(try transformStagingDirectories(nextTo: fixture.output).isEmpty)
}

@Test
func transformRejectsReferenceAsOutputWithoutMutatingCheckpoint() throws {
    let fixture = try writeTransformFixture()
    let originalConfig = try Data(
        contentsOf: fixture.reference.appendingPathComponent("config.json")
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: fixture.reference.path
            )
        )
    }

    #expect(
        try Data(contentsOf: fixture.reference.appendingPathComponent("config.json"))
            == originalConfig
    )
    #expect(
        FileManager.default.fileExists(
            atPath: fixture.reference.appendingPathComponent("model-00001-of-00001.safetensors").path
        )
    )
}

@Test
func transformRejectsSymlinkAliasOfReferenceAsOutput() throws {
    let fixture = try writeTransformFixture()
    let outputAlias = fixture.root.appendingPathComponent("reference-alias", isDirectory: true)
    try FileManager.default.createSymbolicLink(
        at: outputAlias,
        withDestinationURL: fixture.reference
    )
    let originalConfig = try Data(
        contentsOf: fixture.reference.appendingPathComponent("config.json")
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: outputAlias.path
            )
        )
    }

    #expect(
        try Data(contentsOf: fixture.reference.appendingPathComponent("config.json"))
            == originalConfig
    )
}

@Test
func transformRejectsOutputThatContainsWorkingDirectory() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference")
    let workingDirectory = root.appendingPathComponent("workspace/project")
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)

    #expect(throws: MLXFastError.self) {
        try SwiftTransform.validateDistinctDirectories(
            referenceDirectory: reference,
            outputDirectory: root,
            workingDirectory: workingDirectory
        )
    }
}

@Test
func transformRejectsOutputInsideReferenceDirectory() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference")
    let output = reference.appendingPathComponent("weights")
    let workingDirectory = root.appendingPathComponent("workspace")
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)

    #expect(throws: MLXFastError.self) {
        try SwiftTransform.validateDistinctDirectories(
            referenceDirectory: reference,
            outputDirectory: output,
            workingDirectory: workingDirectory
        )
    }
}

@Test
func transformRejectsExistingRegularFileOutputWithoutReplacingIt() throws {
    let fixture = try writeTransformFixture()
    let outputFile = fixture.root.appendingPathComponent("existing-output")
    try "preserve me".write(to: outputFile, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: outputFile.path
            )
        )
    }

    #expect(try String(contentsOf: outputFile, encoding: .utf8) == "preserve me")
}

@Test
func transformAtomicallyReplacesExistingOutputAndRemovesStaleFiles() throws {
    let fixture = try writeTransformFixture()
    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let staleFile = fixture.output.appendingPathComponent("stale.txt")
    try "stale".write(to: staleFile, atomically: true, encoding: .utf8)

    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )

    #expect(!FileManager.default.fileExists(atPath: staleFile.path))
    #expect(
        FileManager.default.fileExists(
            atPath: fixture.output.appendingPathComponent("model.safetensors.index.json").path
        )
    )
}

@Test
func transformFailurePreservesExistingOutputAndRemovesStagingDirectory() throws {
    let fixture = try writeTransformFixture()
    try #"{"vision_config":{}}"#.write(
        to: fixture.reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let sentinel = fixture.output.appendingPathComponent("sentinel.txt")
    try "original".write(to: sentinel, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
        )
    }

    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "original")
    let siblings = try FileManager.default.contentsOfDirectory(
        at: fixture.output.deletingLastPathComponent(),
        includingPropertiesForKeys: nil
    )
    #expect(
        !siblings.contains {
            $0.lastPathComponent.hasPrefix(".weights.mlxfast-transform-")
        }
    )
}

@Test
func checkpointIndexBuildRejectsDuplicateTensorNamesAcrossShards() throws {
    let root = try temporaryDirectory()
    let tensorName = "language_model.model.embed_tokens.weight"
    try writeSafetensors(
        root.appendingPathComponent("model-00001-of-00002.safetensors"),
        tensors: [TensorFixture(name: tensorName, dtype: "U8", shape: [1], data: Data([1]))]
    )
    try writeSafetensors(
        root.appendingPathComponent("model-00002-of-00002.safetensors"),
        tensors: [TensorFixture(name: tensorName, dtype: "U8", shape: [1], data: Data([2]))]
    )

    #expect(throws: MLXFastError.self) {
        _ = try CheckpointIndex.buildFromSafetensors(in: root)
    }
}

@Test
func transformRejectsCheckpointWithoutTextTowerTensorsBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try lagunaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [TensorFixture(name: visionName, dtype: "U8", shape: [2], data: Data([1, 2]))]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [visionName: shardName]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformVerifierAcceptsFreshSubmittedTransformOutputAndIgnoresLocalCacheMarkers() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "cache\n".write(
        to: fixture.output.appendingPathComponent(".benchmark-source.sha256"),
        atomically: true,
        encoding: .utf8
    )
    FileManager.default.createFile(
        atPath: fixture.output.appendingPathComponent(".gitkeep").path,
        contents: Data()
    )

    let report = try TransformVerifier.verify(
        TransformVerificationOptions(
            referencePath: fixture.reference.path,
            weightsPath: fixture.output.path,
            temporaryParentPath: fixture.root.path
        )
    )

    #expect(report.referencePath == fixture.reference.path)
    #expect(report.weightsPath == fixture.output.path)
    #expect(report.fileCount > 0)
    #expect(report.byteCount > 0)
    #expect(report.maxByteCount == MLXFastConstants.defaultMaxTransformedWeightsBytes)
    #expect(report.sha256.count == 64)
    #expect(report.deterministic)
}

@Test
func transformVerifierRejectsOutputThatDiffersFromFreshTransformRun() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "changed".write(
        to: fixture.output.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path
            )
        )
    }
}

@Test
func transformVerifierRejectsStaleExtraGeneratedFile() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "extra".write(
        to: fixture.output.appendingPathComponent("extra.txt"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path
            )
        )
    }
}

@Test
func transformVerifierRejectsOutputAboveConfiguredByteLimit() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path,
                maxByteCount: 1
            )
        )
    }
}

@Test
func transformRejectsUnsupportedIndexShardBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try lagunaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "model.layers.0.self_attn.q_proj.weight": "pytorch_model.bin",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformRejectsUnsafeIndexShardBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try lagunaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "model.layers.0.self_attn.q_proj.weight": "../model-00001.safetensors",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformRejectsIndexTensorMissingFromShardHeaderBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try lagunaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: "model.layers.0.self_attn.k_proj.weight", dtype: "U8", shape: [2], data: Data([1, 2])),
        ]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "model.layers.0.self_attn.q_proj.weight": shardName,
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformAcceptsSparseShardLargerThanInt32() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try gemmaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.layers.0.self_attn.q_proj.weight"
    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let shardName = "model-00001-of-00001.safetensors"
    let shard = reference.appendingPathComponent(shardName)
    try writeSafetensors(
        shard,
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [1], data: Data([4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [1], data: Data([8])),
        ]
    )
    try truncateFile(shard, toByteCount: Int64(Int32.max) + 1024)
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            textName: shardName,
            visionName: shardName,
        ]
    )

    let report = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    #expect(report.denseTensorCount == 1)
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private struct TransformFixturePaths {
    let root: URL
    let reference: URL
    let output: URL
}

private func writeTransformFixture() throws -> TransformFixturePaths {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try gemmaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try #"{"tokenizer":"fixture"}"#.write(
        to: reference.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.layers.0.self_attn.q_proj.weight"
    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [4], data: Data([1, 2, 3, 4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [3], data: Data([9, 8, 7])),
        ]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            textName: shardName,
            visionName: shardName,
        ]
    )

    return TransformFixturePaths(root: root, reference: reference, output: output)
}

/// Legacy Gemma 4 multimodal source-config layout (nested `text_config`),
/// kept to cover the transform's `.gemma4` family path.
private func gemmaReferenceConfigJSON() -> String {
    """
    {
      "text_config": {
        "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
        "vocab_size": \(MLXFastConstants.vocabSize),
        "hidden_size": \(MLXFastConstants.hiddenSize)
      },
      "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
    }
    """
}

/// Flat Poolside Laguna XS 2.1 NVFP4 source config: Laguna geometry and the
/// exact global NVFP4 4-bit group-16 block, loaded from the immutable artifact
/// contract fixture rather than reconstructed synthetically here.
private func lagunaReferenceConfigJSON() throws -> String {
    let data = try JSONSerialization.data(
        withJSONObject: pinnedLagunaConfigObject(),
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )
    return String(decoding: data, as: UTF8.self)
}

/// Poolside NVFP4 pair: U32 packed codes plus U8 E4M3 group-16 scales.
private func nvfp4TensorPair(
    stem: String,
    leadingShape: [Int],
    inputFeatures: Int
) -> [TensorFixture] {
    let packedWidth = inputFeatures * LagunaConstants.quantizationBits / 32
    let groupWidth = inputFeatures / LagunaConstants.quantizationGroupSize
    let weightShape = leadingShape + [packedWidth]
    let companionShape = leadingShape + [groupWidth]
    return [
        TensorFixture(
            name: "\(stem).weight",
            dtype: "U32",
            shape: weightShape,
            data: Data(count: weightShape.reduce(1, *) * 4)
        ),
        TensorFixture(
            name: "\(stem).scales",
            dtype: "U8",
            shape: companionShape,
            data: Data(count: companionShape.reduce(1, *))
        ),
    ]
}

private func bf16Tensor(name: String, shape: [Int]) -> TensorFixture {
    TensorFixture(
        name: name,
        dtype: "BF16",
        shape: shape,
        data: Data(count: shape.reduce(1, *) * 2)
    )
}

private func f32Tensor(name: String, shape: [Int]) -> TensorFixture {
    TensorFixture(
        name: name,
        dtype: "F32",
        shape: shape,
        data: Data(count: shape.reduce(1, *) * 4)
    )
}

private struct LagunaFixturePaths {
    let root: URL
    let reference: URL
    let output: URL
    let shardName: String
}

private func writeLagunaCheckpointFixture(
    tensors: [TensorFixture],
    configJSON: String? = nil
) throws -> LagunaFixturePaths {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    let effectiveConfigJSON = try configJSON ?? lagunaReferenceConfigJSON()
    try effectiveConfigJSON.write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(reference.appendingPathComponent(shardName), tensors: tensors)
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, shardName) })
    )
    return LagunaFixturePaths(
        root: root,
        reference: reference,
        output: output,
        shardName: shardName
    )
}

private func writeCheckpointIndex(
    _ path: URL,
    weightMap: [String: String],
    metadata: [String: Any]? = nil
) throws {
    var object: [String: Any] = ["weight_map": weightMap]
    object["metadata"] = metadata
    let data = try JSONSerialization.data(
        withJSONObject: object,
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )
    try data.write(to: path)
}

private func transformStagingDirectories(nextTo output: URL) throws -> [URL] {
    try FileManager.default.contentsOfDirectory(
        at: output.deletingLastPathComponent(),
        includingPropertiesForKeys: nil
    ).filter {
        $0.lastPathComponent.hasPrefix(".\(output.lastPathComponent).mlxfast-transform-")
    }
}

private func writeSafetensors(_ path: URL, tensors: [TensorFixture]) throws {
    var object: [String: Any] = [:]
    var cursor = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        object[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + tensor.data.count],
        ]
        cursor += tensor.data.count
    }

    var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }

    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        output.append(tensor.data)
    }
    try output.write(to: path)
}

private func truncateFile(_ path: URL, toByteCount byteCount: Int64) throws {
    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(byteCount))
}

private func tensorBytes(_ path: URL, header: SafetensorsHeader, name: String) throws -> Data {
    let info = try #require(header.tensors[name])
    let data = try Data(contentsOf: path)
    let start = Int(header.dataBaseOffset) + info.dataStart
    return data.subdata(in: start..<(start + info.byteCount))
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
