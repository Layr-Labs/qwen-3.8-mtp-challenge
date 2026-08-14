import CoreFoundation
import Foundation
import MLXFastCore

/// Quantization expectations parsed from the pinned Qwen 3.6 checkpoint's
/// `quantization` / `quantization_config` blocks. The intended checkpoint is
/// exactly MLX affine 4-bit group-64, with no per-tensor overrides.
struct Qwen35TransformQuantizationSpec: Equatable {
    let groupSize: Int
    let bits: Int
    let mode: String
}

/// Transform-side structural validation of the Qwen 3.6 text-tower tensor set
/// against `docs/qwen3.6-weight-contract.md`.
///
/// The source checkpoint (`mlx-community/Qwen3.6-27B-4bit`) is already MLX
/// affine-quantized, so the transform passes tensors through unchanged. This
/// pass fails fast -- before the ~15 GB copy -- when the set it would copy
/// cannot satisfy the runtime loader (`Qwen35WeightLoader`):
///
/// - the selected `language_model.*` namespace must be EXACTLY the public
///   1,847-tensor inventory, tensor for tensor, at the exact dtype and shape;
/// - every quantized projection stored as packed U32 codes must ship BF16
///   `.scales` AND BF16 `.biases` (affine needs both; the Laguna NVFP4
///   contract forbids `.biases`, which is why the two validators cannot share
///   one rule) with matching leading dimensions;
/// - each packed width must match the group size and bit width the emitted
///   config.json declares (affine 4-bit group-64): `in / 8` U32 columns and
///   `in / 64` scale/bias columns;
/// - compressed-tensors aliases, global-scale tensors, and FP8 KV scales are
///   rejected outright;
/// - no `mtp.*` tensor may appear. The MTP head is a separately pinned
///   artifact (`mlx-community/Qwen3.6-27B-MTP-4bit`) and the pinned backbone
///   revision contains none; a checkpoint that smuggles one in would produce
///   a `weights/` tree the eager baseline cannot load.
///
/// Unquantized tensors (the norms and the gated-delta `A_log`/`dt_bias`/
/// `conv1d` state) are covered by the exact-inventory pass rather than by a
/// packing rule; the runtime re-validates their full geometry against
/// `Qwen35Config` before the first forward.
///
/// This is deliberately independent of shard placement: the public fixture
/// `fixtures/qwen3_6_27b_tensor_inventory.json` pins placement and header
/// digests, while this pass pins the complete tensor namespace, dtype, and
/// shape.
enum Qwen35CheckpointValidation {
    struct ExpectedTensorMetadata: Equatable {
        let dtype: String
        let shape: [Int]
    }

    /// Frozen geometry of the pinned `qwen3_5_text` tower. Kept as literals
    /// rather than read from the config under validation: a validator that
    /// derives its expectations from the artifact it is checking cannot
    /// detect a changed artifact.
    enum PinnedGeometry {
        static let vocabSize = 248_320
        static let hiddenSize = 5_120
        static let intermediateSize = 17_408
        static let layerCount = 64
        /// Every 4th layer (index % 4 == 3) is full attention; the other
        /// three are gated-delta linear attention.
        static let fullAttentionInterval = 4
        static let attentionHeads = 24
        static let keyValueHeads = 4
        static let headDim = 256
        static let linearValueHeads = 48
        static let linearKeyHeads = 16
        static let linearValueHeadDim = 128
        static let linearKeyHeadDim = 128
        static let linearConvKernelDim = 4
        static let quantizationGroupSize = 64
        static let quantizationBits = 4
        static let quantizationMode = "affine"
    }

    /// Total tensors in the transformed text-tower artifact: 7 top level plus
    /// 48 linear-attention layers of 30 and 16 full-attention layers of 25.
    static let expectedTensorCount = 1_847

    static let textTowerPrefix = "language_model."

    private static let layerPrefix = "language_model.model.layers."

    /// Parses the checkpoint's quantization block(s).
    ///
    /// The pinned artifact publishes the SAME spec twice, as `quantization`
    /// and `quantization_config`; `Transform.makeRuntimeConfigData` accepts
    /// either form and requires them to agree when both are present, so this
    /// validator applies exactly that policy rather than a stricter one -- a
    /// transform that emits a config must not reject the checkpoint that
    /// config came from.
    static func quantizationSpec(
        fromConfigRoot root: [String: Any]
    ) throws -> Qwen35TransformQuantizationSpec {
        func parseBlock(_ key: String) throws -> Qwen35TransformQuantizationSpec? {
            guard let value = root[key], !(value is NSNull) else {
                return nil
            }
            guard let block = value as? [String: Any] else {
                throw MLXFastError.invalidInput("Qwen3.6 config \(key) must be an object")
            }
            let allowedKeys: Set<String> = ["group_size", "bits", "mode"]
            let unexpectedKeys = Set(block.keys).subtracting(allowedKeys)
            guard unexpectedKeys.isEmpty else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 config \(key) contains unsupported fields: "
                        + unexpectedKeys.sorted().joined(separator: ", ")
                )
            }
            guard block["group_size"] != nil, block["bits"] != nil, block["mode"] != nil else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 config \(key) must explicitly define group_size, bits, and mode"
                )
            }
            let groupSize = try intField("group_size", in: block)
            let bits = try intField("bits", in: block)
            let mode = try stringField("mode", in: block)
            guard mode == PinnedGeometry.quantizationMode,
                  groupSize == PinnedGeometry.quantizationGroupSize,
                  bits == PinnedGeometry.quantizationBits
            else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 quantization must be affine 4-bit group_size 64"
                )
            }
            return Qwen35TransformQuantizationSpec(
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
        }

        let quantization = try parseBlock("quantization")
        let quantizationConfig = try parseBlock("quantization_config")
        if let quantization, let quantizationConfig {
            guard quantization == quantizationConfig else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 config quantization and quantization_config must match exactly"
                )
            }
            return quantization
        }
        guard let spec = quantization ?? quantizationConfig else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 config is missing both quantization and quantization_config"
            )
        }
        return spec
    }

    /// Exact metadata contract of the transformed text tower, extracted from
    /// the public three-shard `mlx-community/Qwen3.6-27B-4bit` artifact at
    /// revision `c000ac2c2057d94be3fa931000c31723aac53282`.
    static func expectedTensorInventory() -> [String: ExpectedTensorMetadata] {
        let hidden = PinnedGeometry.hiddenSize
        let intermediate = PinnedGeometry.intermediateSize
        let vocab = PinnedGeometry.vocabSize
        let bits = PinnedGeometry.quantizationBits
        let groupSize = PinnedGeometry.quantizationGroupSize

        var inventory: [String: ExpectedTensorMetadata] = [:]
        func add(_ name: String, _ dtype: TensorDType, _ shape: [Int]) {
            precondition(inventory[name] == nil, "duplicate expected Qwen3.6 tensor \(name)")
            inventory[name] = ExpectedTensorMetadata(dtype: dtype.rawValue, shape: shape)
        }
        /// One affine-quantized 2D projection: packed U32 codes plus the BF16
        /// scale and bias companions the affine scheme requires.
        func addAffine(_ stem: String, outFeatures: Int, inFeatures: Int) {
            add("\(stem).weight", .u32, [outFeatures, inFeatures * bits / 32])
            add("\(stem).scales", .bf16, [outFeatures, inFeatures / groupSize])
            add("\(stem).biases", .bf16, [outFeatures, inFeatures / groupSize])
        }

        addAffine(
            "language_model.model.embed_tokens",
            outFeatures: vocab,
            inFeatures: hidden
        )
        add("language_model.model.norm.weight", .bf16, [hidden])
        addAffine("language_model.lm_head", outFeatures: vocab, inFeatures: hidden)

        let linearKeySize = PinnedGeometry.linearKeyHeads * PinnedGeometry.linearKeyHeadDim
        let linearValueSize = PinnedGeometry.linearValueHeads * PinnedGeometry.linearValueHeadDim
        let linearConvSize = linearKeySize * 2 + linearValueSize
        // The full-attention query projection carries the per-head output
        // gate in the same matrix, hence the factor of two.
        let fullQuerySize = PinnedGeometry.attentionHeads * PinnedGeometry.headDim * 2
        let fullKeyValueSize = PinnedGeometry.keyValueHeads * PinnedGeometry.headDim
        let fullOutputSize = PinnedGeometry.attentionHeads * PinnedGeometry.headDim

        for layerIndex in 0..<PinnedGeometry.layerCount {
            let prefix = "\(layerPrefix)\(layerIndex)"
            add("\(prefix).input_layernorm.weight", .bf16, [hidden])
            add("\(prefix).post_attention_layernorm.weight", .bf16, [hidden])

            addAffine("\(prefix).mlp.gate_proj", outFeatures: intermediate, inFeatures: hidden)
            addAffine("\(prefix).mlp.up_proj", outFeatures: intermediate, inFeatures: hidden)
            addAffine("\(prefix).mlp.down_proj", outFeatures: hidden, inFeatures: intermediate)

            if layerIndex % PinnedGeometry.fullAttentionInterval
                == PinnedGeometry.fullAttentionInterval - 1
            {
                add("\(prefix).self_attn.q_norm.weight", .bf16, [PinnedGeometry.headDim])
                add("\(prefix).self_attn.k_norm.weight", .bf16, [PinnedGeometry.headDim])
                addAffine(
                    "\(prefix).self_attn.q_proj",
                    outFeatures: fullQuerySize,
                    inFeatures: hidden
                )
                addAffine(
                    "\(prefix).self_attn.k_proj",
                    outFeatures: fullKeyValueSize,
                    inFeatures: hidden
                )
                addAffine(
                    "\(prefix).self_attn.v_proj",
                    outFeatures: fullKeyValueSize,
                    inFeatures: hidden
                )
                addAffine(
                    "\(prefix).self_attn.o_proj",
                    outFeatures: hidden,
                    inFeatures: fullOutputSize
                )
                continue
            }

            add(
                "\(prefix).linear_attn.conv1d.weight",
                .bf16,
                [linearConvSize, PinnedGeometry.linearConvKernelDim, 1]
            )
            add("\(prefix).linear_attn.A_log", .bf16, [PinnedGeometry.linearValueHeads])
            add("\(prefix).linear_attn.dt_bias", .bf16, [PinnedGeometry.linearValueHeads])
            add("\(prefix).linear_attn.norm.weight", .bf16, [PinnedGeometry.linearValueHeadDim])
            addAffine(
                "\(prefix).linear_attn.in_proj_qkv",
                outFeatures: linearConvSize,
                inFeatures: hidden
            )
            addAffine(
                "\(prefix).linear_attn.in_proj_z",
                outFeatures: linearValueSize,
                inFeatures: hidden
            )
            addAffine(
                "\(prefix).linear_attn.in_proj_b",
                outFeatures: PinnedGeometry.linearValueHeads,
                inFeatures: hidden
            )
            addAffine(
                "\(prefix).linear_attn.in_proj_a",
                outFeatures: PinnedGeometry.linearValueHeads,
                inFeatures: hidden
            )
            addAffine(
                "\(prefix).linear_attn.out_proj",
                outFeatures: hidden,
                inFeatures: linearValueSize
            )
        }

        precondition(
            inventory.count == expectedTensorCount,
            "Qwen3.6 inventory must contain \(expectedTensorCount) tensors"
        )
        return inventory
    }

    static func validateSelectedTensors(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        quantization: Qwen35TransformQuantizationSpec
    ) throws {
        // Affine quantization legitimately ships `.biases`, so unlike the
        // Laguna NVFP4 contract that suffix is NOT forbidden here.
        let forbiddenSuffixes = [
            ".weight_packed",
            ".input_global_scale",
            ".weight_global_scale",
            ".k_scale",
            ".v_scale",
        ]
        if let forbiddenName = selectedKeys.sorted().first(where: { name in
            forbiddenSuffixes.contains { suffix in name.hasSuffix(suffix) }
        }) {
            throw MLXFastError.invalidInput(
                "Qwen3.6 MLX transform rejects compressed-tensors/global-scale and "
                    + "FP8 KV-scale tensor \(forbiddenName)"
            )
        }
        if let mtpName = selectedKeys.sorted().first(where: { name in
            name.split(separator: ".").contains("mtp")
        }) {
            throw MLXFastError.invalidInput(
                "Qwen3.6 backbone transform must not select MTP tensor \(mtpName); the MTP "
                    + "head is a separately pinned artifact"
            )
        }

        for name in selectedKeys.sorted() where name.hasSuffix(".weight") {
            let stem = String(name.dropLast(".weight".count))
            let scalesName = "\(stem).scales"
            let biasesName = "\(stem).biases"
            guard selectedKeys.contains(scalesName) || selectedKeys.contains(biasesName) else {
                continue
            }
            guard selectedKeys.contains(scalesName), selectedKeys.contains(biasesName) else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 affine projection \(stem) must ship both .scales and .biases"
                )
            }
            let weightInfo = try tensorInfo(named: name, index: index, headers: headers)
            let scalesInfo = try tensorInfo(named: scalesName, index: index, headers: headers)
            let biasesInfo = try tensorInfo(named: biasesName, index: index, headers: headers)
            guard weightInfo.dtype == TensorDType.u32.rawValue,
                  scalesInfo.dtype == TensorDType.bf16.rawValue,
                  biasesInfo.dtype == TensorDType.bf16.rawValue,
                  weightInfo.shape.count == 2,
                  scalesInfo.shape == biasesInfo.shape,
                  scalesInfo.shape.count == 2,
                  weightInfo.shape[0] == scalesInfo.shape[0],
                  weightInfo.shape.allSatisfy({ $0 > 0 }),
                  scalesInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 affine projection \(stem) has incompatible weight, scale, or "
                        + "bias metadata"
                )
            }

            let packedWidth = weightInfo.shape[1]
            let groupCount = scalesInfo.shape[1]
            let (inputFeatures, inputOverflow) = groupCount.multipliedReportingOverflow(
                by: quantization.groupSize
            )
            let (packedBits, packedOverflow) = packedWidth.multipliedReportingOverflow(by: 32)
            guard !inputOverflow, !packedOverflow else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 projection \(stem) packed width overflows Int"
                )
            }
            let (expectedPackedBits, expectedOverflow) = inputFeatures.multipliedReportingOverflow(
                by: quantization.bits
            )
            guard !expectedOverflow, packedBits == expectedPackedBits else {
                throw MLXFastError.invalidInput(
                    "quantized Qwen3.6 projection \(stem) stored width \(packedWidth) does not "
                        + "match config quantization group_size \(quantization.groupSize) "
                        + "bits \(quantization.bits) for input dimension \(inputFeatures)"
                )
            }
        }

        try validateExactPublicInventory(
            selectedKeys: selectedKeys,
            index: index,
            headers: headers
        )
    }

    static func validateExactPublicInventory(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws {
        let expected = expectedTensorInventory()
        let expectedNames = Set(expected.keys)
        // The public checkpoint also carries the `vision_tower.*` namespace,
        // which the transform never selects, so -- unlike the Laguna
        // validator -- the index and header name sets are supersets rather
        // than equal. Constrain them where it matters: every selected name
        // must be indexed exactly once and present in exactly one header.
        let headerNameList = headers.values.flatMap { $0.tensors.keys }
        let headerNames = Set(headerNameList)

        guard selectedKeys == expectedNames,
              expectedNames.isSubset(of: Set(index.weightMap.keys)),
              expectedNames.isSubset(of: headerNames),
              headerNameList.count == headerNames.count
        else {
            let missing = expectedNames.subtracting(selectedKeys).sorted()
            let extra = selectedKeys.subtracting(expectedNames).sorted()
            let unindexed = expectedNames.subtracting(Set(index.weightMap.keys)).sorted()
            throw MLXFastError.invalidInput(
                "Qwen3.6 checkpoint tensor inventory must match the exact public "
                    + "\(expectedTensorCount)-tensor contract "
                    + "(missing: \(missing.prefix(8).joined(separator: ", ")); "
                    + "extra: \(extra.prefix(8).joined(separator: ", ")); "
                    + "unindexed/duplicate header tensors: "
                    + "\(unindexed.prefix(8).joined(separator: ", ")))"
            )
        }

        for name in expected.keys.sorted() {
            guard let expectedMetadata = expected[name] else {
                preconditionFailure("missing expected Qwen3.6 metadata for \(name)")
            }
            let actual = try tensorInfo(named: name, index: index, headers: headers)
            guard actual.dtype == expectedMetadata.dtype,
                  actual.shape == expectedMetadata.shape
            else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 tensor \(name) dtype/shape \(actual.dtype) \(actual.shape) "
                        + "does not match exact public metadata "
                        + "\(expectedMetadata.dtype) \(expectedMetadata.shape)"
                )
            }
        }
    }

    private static func tensorInfo(
        named name: String,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws -> SafetensorInfo {
        guard let shardName = index.weightMap[name],
              let info = headers[shardName]?.tensors[name]
        else {
            throw MLXFastError.invalidInput("missing validated tensor metadata for \(name)")
        }
        return info
    }

    private static func intField(_ key: String, in object: [String: Any]) throws -> Int {
        guard let number = object[key] as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number),
              let integer = Int(number.stringValue)
        else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 quantization field \(key) must be a finite integer in Int range"
            )
        }
        return integer
    }

    private static func stringField(_ key: String, in object: [String: Any]) throws -> String {
        guard let string = object[key] as? String else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 quantization field \(key) must be a string"
            )
        }
        return string
    }
}
