import Foundation
import MLXFastCore
#if canImport(Darwin)
import Darwin
#endif

public struct TransformOptions: Equatable {
    public let referencePath: String
    public let outputPath: String

    public init(referencePath: String, outputPath: String) {
        self.referencePath = referencePath
        self.outputPath = outputPath
    }
}

public struct TransformReport: Equatable {
    public let referencePath: String
    public let outputPath: String
    public let denseTensorCount: Int
    public let denseShardCount: Int
    public let configPath: String
    public let indexPath: String

    public init(
        referencePath: String,
        outputPath: String,
        denseTensorCount: Int,
        denseShardCount: Int,
        configPath: String,
        indexPath: String
    ) {
        self.referencePath = referencePath
        self.outputPath = outputPath
        self.denseTensorCount = denseTensorCount
        self.denseShardCount = denseShardCount
        self.configPath = configPath
        self.indexPath = indexPath
    }
}

/// Model family of the source reference checkpoint, detected from its
/// config.json. The transform supports the pinned Poolside Laguna XS 2.1
/// MoE target (flat config with `model_type` "laguna", untied head) and the
/// legacy Gemma 4 multimodal layout (nested `text_config`, tied head).
///
/// The legacy `.gemma4` family no longer has a runtime consumer (the Gemma 4
/// runtime and its MTP track were removed); it is retained here deliberately
/// as the only source-config family whose full transform pipeline (staging,
/// atomic install, revalidation, verifier) can be exercised end to end with
/// small synthetic fixtures -- the Laguna family requires the exact pinned
/// 912-tensor inventory with real shapes.
/// The `.qwen35` family is the Qwen 3.6 native-MTP track target
/// (`mlx-community/Qwen3.6-27B-4bit`, internal architecture name
/// `qwen3_5_text`). It shares the legacy Gemma multimodal layout -- a nested
/// `text_config` and `language_model.*` tensor names -- so it is distinguished
/// from `.gemma4` by the `qwen3_5` model-type prefix inside `text_config`.
enum TransformModelFamily: Equatable {
    case gemma4
    case laguna
    case qwen35
}

/// Offline transform for the pinned reference checkpoint: selects ONLY the
/// text-tower tensors (`model.*` / `lm_head.*` for Poolside Laguna;
/// `language_model.*` for legacy Gemma), drops every vision/audio/
/// multimodal-projector tensor, and rewrites the
/// selected tensors into dense safetensors shard(s) plus a
/// `model.safetensors.index.json` and a runtime-authored `config.json`.
///
/// Two source checkpoint families are supported, detected from the source
/// config.json:
///
/// - Poolside Laguna XS 2.1 NVFP4 (flat config, `model_type` "laguna"): the
///   ranked serial-track target. The source is already MLX NVFP4-quantized,
///   so the transform validates and passes through -- byte-for-byte, source
///   tensor names unchanged -- the BF16/NVFP4 tensor set described by
///   `docs/laguna-weight-contract.md`: attention
///   q/k/v/o projections plus the per-head `g_proj` gates and q/k norms,
///   the layer-0 dense MLP, the SwitchGLU-STACKED `mlp.switch_mlp.*` NVFP4
///   expert tensors (leading experts axis; never split per expert), the raw
///   BF16 `mlp.gate.weight` routers with their F32 correction vectors, the
///   NVFP4 shared experts, and the untied BF16 `lm_head`. The
///   contract forbids derived metadata sidecars (the Gemma projection and
///   tied-head packed13 sidecars are never emitted for Laguna) and requires
///   `rotary_emb.inv_freq` tables to be left out. The runtime config.json is
///   the flat source config minus the empty `vision_config`, carrying the
///   checkpoint's matching NVFP4 4-bit group-16 `quantization` and
///   `quantization_config` blocks.
/// - Qwen 3.6 27B 4-bit (nested `text_config` whose `model_type` starts with
///   `qwen3_5`): the Qwen native-MTP track target. The source index holds only
///   `language_model.*` and `vision_tower.blocks.*`, so the text-tower prefix
///   selects the tower and drops the vision blocks; the runtime config is the
///   flattened `text_config` plus the checkpoint's affine `quantization` block,
///   which must agree with its duplicate `quantization_config` when both are
///   present. No metadata sidecars are emitted (untied `lm_head`).
/// - Legacy Gemma 4 31B 4-bit (nested `text_config`): the archived dense
///   path, unchanged: flattened `text_config` runtime config plus the
///   projection/tied-head metadata sidecars.
///
/// There is no expert streaming manifest -- the whole selected tree is one
/// flat set of dense tensors, matching how the model (including every routed
/// Laguna expert) is loaded fully into RAM at runtime init.
public enum SwiftTransform {
    /// Tensor name prefix that marks a checkpoint tensor as part of the text
    /// tower. Every other prefix (`vision_tower.`, `embed_vision.`,
    /// `audio_tower.`, `multi_modal_projector.`, ...) is vision/audio/
    /// multimodal-glue and is out of scope for this text-only challenge.
    static let textTowerPrefix = "language_model."

    public static func run(_ options: TransformOptions) throws -> TransformReport {
        try run(
            options,
            beforeSidecarGeneration: nil,
            beforeSourceRevalidation: nil
        )
    }

    static func run(
        _ options: TransformOptions,
        beforeSidecarGeneration: (() throws -> Void)? = nil,
        beforeSourceRevalidation: (() throws -> Void)?
    ) throws -> TransformReport {
        let referenceDirectory = canonicalURL(
            try findReferenceDirectory(URL(fileURLWithPath: options.referencePath))
        )
        let outputDirectory = canonicalURL(URL(fileURLWithPath: options.outputPath))
        try validateDistinctDirectories(
            referenceDirectory: referenceDirectory,
            outputDirectory: outputDirectory
        )
        var outputIsDirectory = ObjCBool(false)
        if FileManager.default.fileExists(
            atPath: outputDirectory.path,
            isDirectory: &outputIsDirectory
        ), !outputIsDirectory.boolValue {
            throw MLXFastError.invalidInput(
                "transform output exists and is not a directory: \(outputDirectory.path)"
            )
        }

        let referenceConfigPath = referenceDirectory.appendingPathComponent("config.json")
        try requireFile(
            referenceConfigPath.path,
            description: "reference checkpoint config"
        )
        let sourceConfigRoot = try loadReferenceConfigRoot(referenceConfigPath)
        let modelFamily = try detectModelFamily(sourceConfigRoot: sourceConfigRoot)
        let runtimeConfigData = try makeRuntimeConfigData(
            sourceConfigRoot: sourceConfigRoot,
            family: modelFamily
        )
        let metadataSnapshot = try captureMetadataFiles(from: referenceDirectory)

        let index = try loadIndex(referenceDirectory)
        let indexSnapshot = try index.canonicalData()
        let validatedHeaders = try validateCheckpointIndex(
            index,
            referenceDirectory: referenceDirectory
        )
        let textKeys = Set(
            index.weightMap.keys.filter { isSelectedTextTowerKey($0, family: modelFamily) }
        )
        guard !textKeys.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no text-tower tensors")
        }

        let textKeysByShard = Dictionary(grouping: textKeys) { key in
            index.weightMap[key] ?? ""
        }
        var totalTensorByteCount = 0
        for key in textKeys.sorted() {
            guard let shardName = index.weightMap[key],
                  let info = validatedHeaders[shardName]?.tensors[key]
            else {
                throw MLXFastError.invalidInput(
                    "missing validated tensor metadata for \(key)"
                )
            }
            let (nextTotal, overflow) = totalTensorByteCount.addingReportingOverflow(
                info.byteCount
            )
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "transformed tensor byte count overflows Int"
                )
            }
            totalTensorByteCount = nextTotal
        }

        // Fail before the multi-GB copy if the selected tensor set is
        // structurally inconsistent with the quantization spec the emitted
        // config.json declares (the runtime re-validates the full geometry
        // against its own config type at load). Gemma 4 is the legacy
        // synthetic-fixture family and has no hardcoded public inventory.
        switch modelFamily {
        case .laguna:
            try LagunaCheckpointValidation.validateSelectedTensors(
                selectedKeys: textKeys,
                index: index,
                headers: validatedHeaders,
                quantization: LagunaCheckpointValidation.quantizationSpec(
                    fromConfigRoot: sourceConfigRoot
                )
            )
        case .qwen35:
            try Qwen35CheckpointValidation.validateSelectedTensors(
                selectedKeys: textKeys,
                index: index,
                headers: validatedHeaders,
                quantization: Qwen35CheckpointValidation.quantizationSpec(
                    fromConfigRoot: sourceConfigRoot
                )
            )
        case .gemma4:
            break
        }

        let fileManager = FileManager.default
        let stagingDirectory = outputDirectory.deletingLastPathComponent().appendingPathComponent(
            ".\(outputDirectory.lastPathComponent).mlxfast-transform-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: stagingDirectory.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: false)
        var installed = false
        defer {
            if !installed {
                try? fileManager.removeItem(at: stagingDirectory)
            }
        }

        var copiedTensors = 0
        var stagedHeaders: [String: SafetensorsHeader] = [:]
        for shardName in textKeysByShard.keys.sorted() {
            let source = referenceDirectory.appendingPathComponent(shardName)
            let destination = stagingDirectory.appendingPathComponent(shardName)
            guard let header = validatedHeaders[shardName] else {
                throw MLXFastError.invalidInput("missing validated header for checkpoint shard \(shardName)")
            }
            let selectedNames = textKeysByShard[shardName, default: []].sorted()
            if Set(selectedNames) == Set(header.tensors.keys) {
                // Poolside's reference is already text-only. Preserve the
                // byte-identical shard and use an APFS copy-on-write clone
                // when source/output share a volume; fall back to a normal
                // independent file copy elsewhere. Never publish a symlink.
                try cloneOrCopyShard(from: source, to: destination)
                copiedTensors += selectedNames.count
            } else {
                copiedTensors += try Safetensors.copySubset(
                    from: source,
                    to: destination,
                    tensorNames: selectedNames,
                    validatedHeader: header
                )
            }
            stagedHeaders[shardName] = try Safetensors.readHeader(destination)
        }

        try beforeSidecarGeneration?()
        let generatedProjectionMetadata: GeneratedAffineMetadataReport
        let generatedTiedHeadMetadata: GeneratedAffineMetadataReport
        switch modelFamily {
        case .gemma4:
            generatedProjectionMetadata = try AffineMetadataCoding.writeProjectionSidecar(
                sourceDirectory: stagingDirectory,
                index: index,
                sourceHeaders: stagedHeaders,
                selectedKeys: textKeys,
                destinationDirectory: stagingDirectory
            )
            generatedTiedHeadMetadata = try TiedHeadMetadataCoding.writeSidecar(
                sourceDirectory: stagingDirectory,
                index: index,
                sourceHeaders: stagedHeaders,
                selectedKeys: textKeys,
                destinationDirectory: stagingDirectory
            )
        case .qwen35, .laguna:
            // Qwen 3.6 has an untied `lm_head` and the runtime reads the
            // checkpoint's own affine-quantized tensors directly, so neither
            // the Gemma projection sidecar nor the tied-head packed13 sidecar
            // means anything on this family -- emit nothing beyond the
            // pass-through tensor set. For Laguna,
            // docs/laguna-weight-contract.md forbids derived layouts and
            // metadata sidecars in the Poolside v2 contract, and the runtime loads exactly the
            // indexed checkpoint tensors (its untied lm_head makes the
            // Gemma tied-head packed13 sidecar meaningless anyway). Emit
            // nothing beyond the pass-through tensor set.
            generatedProjectionMetadata = GeneratedAffineMetadataReport(
                weightMap: [:],
                tensorByteCount: 0
            )
            generatedTiedHeadMetadata = GeneratedAffineMetadataReport(
                weightMap: [:],
                tensorByteCount: 0
            )
        }
        let (projectionOutputByteCount, projectionSizeOverflow) =
            totalTensorByteCount.addingReportingOverflow(
                generatedProjectionMetadata.tensorByteCount
            )
        let (outputTensorByteCount, tiedHeadSizeOverflow) =
            projectionOutputByteCount.addingReportingOverflow(
                generatedTiedHeadMetadata.tensorByteCount
            )
        guard !projectionSizeOverflow, !tiedHeadSizeOverflow else {
            throw MLXFastError.invalidInput("transformed tensor byte count overflows Int")
        }
        let generatedWeightMap = generatedProjectionMetadata.weightMap.merging(
            generatedTiedHeadMetadata.weightMap
        ) { _, _ in
            preconditionFailure("generated metadata tensor names collide")
        }

        try writeMetadataFiles(metadataSnapshot, to: stagingDirectory)
        try index.writeStripped(
            to: stagingDirectory.appendingPathComponent("model.safetensors.index.json"),
            keeping: textKeys,
            totalTensorByteCount: outputTensorByteCount,
            additionalWeightMap: generatedWeightMap
        )

        try runtimeConfigData.write(
            to: stagingDirectory.appendingPathComponent("config.json")
        )
        try beforeSourceRevalidation?()
        try validateConfigAndIndexSnapshot(
            referenceDirectory: referenceDirectory,
            referenceConfigPath: referenceConfigPath,
            runtimeConfigData: runtimeConfigData,
            indexSnapshot: indexSnapshot,
            metadataSnapshot: metadataSnapshot
        )
        for shardName in validatedHeaders.keys.sorted() {
            guard let header = validatedHeaders[shardName] else {
                throw MLXFastError.invalidInput(
                    "missing validated header for checkpoint shard \(shardName)"
                )
            }
            try Safetensors.validateSourceIdentity(
                referenceDirectory.appendingPathComponent(shardName),
                against: header
            )
        }
        try validateConfigAndIndexSnapshot(
            referenceDirectory: referenceDirectory,
            referenceConfigPath: referenceConfigPath,
            runtimeConfigData: runtimeConfigData,
            indexSnapshot: indexSnapshot,
            metadataSnapshot: metadataSnapshot
        )
        try installTransformedDirectory(
            stagingDirectory,
            at: outputDirectory,
            fileManager: fileManager
        )
        installed = true

        let indexPath = outputDirectory.appendingPathComponent("model.safetensors.index.json")
        let configPath = outputDirectory.appendingPathComponent("config.json")

        return TransformReport(
            referencePath: referenceDirectory.path,
            outputPath: outputDirectory.path,
            denseTensorCount: copiedTensors
                + generatedProjectionMetadata.tensorCount
                + generatedTiedHeadMetadata.tensorCount,
            denseShardCount: textKeysByShard.count
                + generatedProjectionMetadata.shardCount
                + generatedTiedHeadMetadata.shardCount,
            configPath: configPath.path,
            indexPath: indexPath.path
        )
    }

    private static func loadIndex(_ referenceDirectory: URL) throws -> CheckpointIndex {
        let indexPath = referenceDirectory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexPath.path) {
            return try CheckpointIndex.load(from: indexPath)
        }
        return try CheckpointIndex.buildFromSafetensors(in: referenceDirectory)
    }

    private static func validateCheckpointIndex(
        _ index: CheckpointIndex,
        referenceDirectory: URL
    ) throws -> [String: SafetensorsHeader] {
        guard !index.weightMap.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no tensors")
        }

        let keysByShard = Dictionary(grouping: index.weightMap.keys.sorted()) { key in
            index.weightMap[key] ?? ""
        }
        var headersByShard: [String: SafetensorsHeader] = [:]
        for shardName in keysByShard.keys.sorted() {
            try validateSafetensorsShardName(shardName, context: "checkpoint index")

            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            try requireFile(shardURL.path, description: "checkpoint shard \(shardName)")
            let header = try Safetensors.readHeader(shardURL)
            headersByShard[shardName] = header

            for key in keysByShard[shardName, default: []].sorted() {
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput(
                        "checkpoint index lists tensor \(key) in \(shardName), but the shard header does not contain it"
                    )
                }
                let dtype = try TensorDType.parse(info.dtype)
                let expectedByteLength = try expectedTensorByteCount(
                    name: key,
                    dtype: dtype,
                    shape: info.shape
                )
                guard info.byteCount == expectedByteLength else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) byte length \(info.byteCount) does not match dtype \(info.dtype) and shape \(info.shape) expected \(expectedByteLength)"
                    )
                }
                // readHeader validated every data range against the opened
                // target descriptor. Do not recheck via the shard pathname:
                // FileManager reports a symlink's own size, not its target's.
                guard info.dataStart >= 0, info.byteCount > 0 else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) has an empty or invalid byte range"
                    )
                }
            }
        }
        return headersByShard
    }

    private static func canonicalURL(_ url: URL) -> URL {
        url.standardizedFileURL.resolvingSymlinksInPath()
    }

    static func validateDistinctDirectories(
        referenceDirectory: URL,
        outputDirectory: URL,
        workingDirectory: URL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        )
    ) throws {
        guard referenceDirectory.path != outputDirectory.path else {
            throw MLXFastError.invalidInput(
                "transform reference and output directories must be different: \(referenceDirectory.path)"
            )
        }
        let outputPrefix = outputDirectory.path == "/" ? "/" : outputDirectory.path + "/"
        guard !referenceDirectory.path.hasPrefix(outputPrefix) else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot contain the reference directory: \(outputDirectory.path)"
            )
        }
        let referencePrefix = referenceDirectory.path == "/"
            ? "/"
            : referenceDirectory.path + "/"
        guard !outputDirectory.path.hasPrefix(referencePrefix) else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot be inside the reference directory: \(outputDirectory.path)"
            )
        }

        let canonicalWorkingDirectory = canonicalURL(workingDirectory)
        guard canonicalWorkingDirectory.path != outputDirectory.path,
              !canonicalWorkingDirectory.path.hasPrefix(outputPrefix)
        else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot contain the current working directory: \(outputDirectory.path)"
            )
        }
    }

    private static func installTransformedDirectory(
        _ stagedDirectory: URL,
        at outputDirectory: URL,
        fileManager: FileManager
    ) throws {
        if fileManager.fileExists(atPath: outputDirectory.path) {
            _ = try fileManager.replaceItemAt(outputDirectory, withItemAt: stagedDirectory)
        } else {
            try fileManager.moveItem(at: stagedDirectory, to: outputDirectory)
        }
    }

    private static func findReferenceDirectory(_ base: URL) throws -> URL {
        if FileManager.default.fileExists(
            atPath: base.appendingPathComponent("config.json").path
        ) {
            return base
        }

        guard let enumerator = FileManager.default.enumerator(
            at: base,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw MLXFastError.missingFile("reference path not found at \(base.path)")
        }

        for case let url as URL in enumerator {
            if url.lastPathComponent == "config.json" {
                return url.deletingLastPathComponent()
            }
        }

        throw MLXFastError.missingFile(
            "no config.json found under \(base.path); place the pinned reference checkpoint there"
        )
    }

    static func isTextTowerKey(_ key: String) -> Bool {
        key.hasPrefix(textTowerPrefix)
    }

    /// Poolside Laguna is already text-only and uses runtime-native
    /// `model.*` / `lm_head.*` names. Legacy Gemma retains the
    /// `language_model.*` selection. Precomputed rotary tables are omitted.
    static func isSelectedTextTowerKey(_ key: String, family: TransformModelFamily) -> Bool {
        switch family {
        case .gemma4:
            return isTextTowerKey(key)
        case .qwen35:
            // The pinned Qwen 3.6 index contains only `language_model.*` and
            // `vision_tower.blocks.*`; the text-tower prefix selects the former
            // and drops the latter. The MTP head (`*.mtp.*`) is not in the
            // pinned backbone revision and is never selected here -- phase 2
            // installs it as a separately pinned artifact.
            return isTextTowerKey(key)
        case .laguna:
            guard key.hasPrefix("model.") || key.hasPrefix("lm_head.") else {
                return false
            }
            return !key.contains("rotary_emb.inv_freq")
        }
    }

    private static func captureMetadataFiles(from source: URL) throws -> [String: Data] {
        let files = try FileManager.default.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )
        var snapshot: [String: Data] = [:]
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if file.lastPathComponent == "model.safetensors.index.json" || file.lastPathComponent == "config.json" {
                continue
            }
            if shouldCopyMetadataFile(file) {
                let values = try file.resourceValues(
                    forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
                )
                guard values.isRegularFile == true, values.isSymbolicLink != true else {
                    throw MLXFastError.invalidInput(
                        "reference metadata is not a regular file: \(file.path)"
                    )
                }
                snapshot[file.lastPathComponent] = try Data(contentsOf: file)
            }
        }
        return snapshot
    }

    private static func writeMetadataFiles(
        _ snapshot: [String: Data],
        to destination: URL
    ) throws {
        for name in snapshot.keys.sorted() {
            try snapshot[name]?.write(to: destination.appendingPathComponent(name))
        }
    }

    private static func shouldCopyMetadataFile(_ url: URL) -> Bool {
        let name = url.lastPathComponent
        if name.hasSuffix(".safetensors") {
            return false
        }
        switch url.pathExtension {
        // `jinja`: the pinned Qwen 3.6 checkpoint ships its chat template as
        // `chat_template.jinja` rather than inside `tokenizer_config.json`.
        case "jinja", "json", "model", "tiktoken", "txt":
            return true
        default:
            return name == "tokenizer" || name == "vocab"
        }
    }

    private static func cloneOrCopyShard(from source: URL, to destination: URL) throws {
        #if canImport(Darwin)
        let cloned = source.path.withCString { sourcePath in
            destination.path.withCString { destinationPath in
                clonefile(sourcePath, destinationPath, 0) == 0
            }
        }
        if !cloned {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.copyItem(at: source, to: destination)
        }
        #else
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: source, to: destination)
        #endif
        let values = try destination.resourceValues(
            forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )
        guard values.isRegularFile == true, values.isSymbolicLink != true else {
            throw MLXFastError.invalidInput(
                "transformed shard copy is not an independent regular file: \(destination.path)"
            )
        }
    }

    private static func loadReferenceConfigRoot(_ sourceConfigPath: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: sourceConfigPath)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("reference config.json must be a JSON object")
        }
        return root
    }

    /// Model-type prefix of the Qwen 3.6 target's text tower. The pinned
    /// checkpoint declares `qwen3_5_text`; the prefix is matched (rather than
    /// the exact string) so a point revision of the same architecture family
    /// is still routed to the Qwen path instead of silently falling through to
    /// the legacy Gemma flattening.
    static let qwen35TextModelTypePrefix = "qwen3_5"

    static func detectModelFamily(
        sourceConfigRoot root: [String: Any]
    ) throws -> TransformModelFamily {
        // The pinned Qwen checkpoint declares `qwen3_5` at the top level and
        // `qwen3_5_text` inside `text_config`; either is sufficient, and both
        // are checked before the `text_config`-means-Gemma fallthrough below.
        if let modelType = root["model_type"] as? String,
           modelType.hasPrefix(qwen35TextModelTypePrefix)
        {
            return .qwen35
        }
        if let textConfig = root["text_config"] as? [String: Any] {
            if let modelType = textConfig["model_type"] as? String,
               modelType.hasPrefix(qwen35TextModelTypePrefix)
            {
                return .qwen35
            }
            return .gemma4
        }
        if let modelType = root["model_type"] as? String, modelType == "laguna" {
            return .laguna
        }
        throw MLXFastError.invalidInput(
            "reference config.json is missing text_config and does not declare model_type laguna"
        )
    }

    static func makeRuntimeConfigData(sourceConfigPath: URL) throws -> Data {
        let root = try loadReferenceConfigRoot(sourceConfigPath)
        return try makeRuntimeConfigData(
            sourceConfigRoot: root,
            family: detectModelFamily(sourceConfigRoot: root)
        )
    }

    /// Writes the runtime's `config.json`.
    ///
    /// Gemma 4 (legacy): the source checkpoint's `text_config` fields
    /// flattened to the top level (the schema the archived Gemma 4 runtime
    /// read), plus the
    /// checkpoint-wide `quantization` block -- no vision or audio config, no
    /// architecture/tokenizer metadata duplicated from
    /// `tokenizer_config.json`.
    ///
    /// Laguna: the source config is already the flat schema
    /// `LagunaConfig.load` parses (per docs/laguna-weight-contract.md the
    /// transform may copy the source fields directly), so it is passed
    /// through minus the empty multimodal `vision_config` stub. Its matching
    /// NVFP4 4-bit group-16 `quantization` and `quantization_config` blocks are
    /// both required and preserved.
    static func makeRuntimeConfigData(
        sourceConfigRoot root: [String: Any],
        family: TransformModelFamily
    ) throws -> Data {
        var runtimeConfig: [String: Any]
        switch family {
        case .gemma4:
            guard let textConfig = root["text_config"] as? [String: Any] else {
                throw MLXFastError.invalidInput("reference config.json is missing text_config")
            }
            runtimeConfig = textConfig
            if let quantization = root["quantization"] {
                runtimeConfig["quantization"] = quantization
            } else if let quantizationConfig = root["quantization_config"] {
                runtimeConfig["quantization"] = quantizationConfig
            }
        case .qwen35:
            guard let textConfig = root["text_config"] as? [String: Any] else {
                throw MLXFastError.invalidInput("reference config.json is missing text_config")
            }
            runtimeConfig = textConfig
            // The pinned Qwen checkpoint publishes the SAME affine spec twice,
            // as `quantization` and `quantization_config`. Emitting one of two
            // conflicting specs would silently pick a quantization the shards
            // were not written with, so the parse below requires them to agree
            // when both are present rather than preferring either -- and pins
            // the values to affine 4-bit group-64 while it is there.
            let spec = try Qwen35CheckpointValidation.quantizationSpec(
                fromConfigRoot: root
            )
            runtimeConfig.removeValue(forKey: "quantization_config")
            runtimeConfig["quantization"] = [
                "group_size": spec.groupSize,
                "bits": spec.bits,
                "mode": spec.mode,
            ]
        case .laguna:
            _ = try LagunaCheckpointValidation.quantizationSpec(fromConfigRoot: root)
            runtimeConfig = root
            runtimeConfig["vision_config"] = nil
        }

        return try JSONSerialization.data(
            withJSONObject: runtimeConfig,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
    }

    private static func validateConfigAndIndexSnapshot(
        referenceDirectory: URL,
        referenceConfigPath: URL,
        runtimeConfigData: Data,
        indexSnapshot: Data,
        metadataSnapshot: [String: Data]
    ) throws {
        guard try makeRuntimeConfigData(sourceConfigPath: referenceConfigPath)
            == runtimeConfigData
        else {
            throw MLXFastError.invalidInput(
                "reference config changed while transform was running"
            )
        }
        guard try loadIndex(referenceDirectory).canonicalData() == indexSnapshot else {
            throw MLXFastError.invalidInput(
                "checkpoint index changed while transform was running"
            )
        }
        guard try captureMetadataFiles(from: referenceDirectory) == metadataSnapshot else {
            throw MLXFastError.invalidInput(
                "reference tokenizer metadata changed while transform was running"
            )
        }
    }
}
