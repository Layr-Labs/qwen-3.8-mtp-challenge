import Foundation
import MLX
import MLXFastCore
import MLXLMCommon
@testable import MLXFastModel
import Testing

// THE STRUCTURAL BLOCKER THIS FILE EXISTS FOR.
//
// The first migration hard-coded `Qwen35TextModel` through the session and the
// worker, and the worker was then unloadable in BOTH directions:
//
//   * the transformed `weights/` tree — the primary, the one the ranked flow
//     provides — declares `qwen3_5_text`, so the factory builds a bare
//     `Qwen35TextModel`; but its 1,847 tensor names still carry the
//     `language_model.` text-tower prefix the transform selects on, and a bare
//     text model addresses `model.*` / `lm_head.weight`. The load died on
//     `keyNotFound ["lm_head","weight"]`;
//   * the raw pinned reference declares `qwen3_5`, so the factory builds
//     `Qwen35Model` — which the concrete-type guard then rejected outright.
//
// So the class that could load was refused and the class that was accepted
// could not load, and zero tokens decoded. Both halves are pinned below, and
// they are cheap text/arithmetic checks: this class of failure does not need a
// model to catch, which is the point.

@Suite
struct QwenMTPBackboneLayoutTests {
    private typealias A = Qwen36MTPHeadAttachment

    private static func config(_ object: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: object)
    }

    /// The TRANSFORMED tree: flat text config. Factory builds the bare text
    /// model, so the wrapper prefix must be stripped.
    @Test
    func theTransformedTreeIsATextModelAndItsPrefixIsStripped() throws {
        let layout = try A.backboneLayout(
            configData: try Self.config([
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "mtp_num_hidden_layers": 1,
            ]))
        #expect(layout == .textModel)
        #expect(layout.primaryKeyPrefixStrip == "language_model.")
    }

    /// The RAW pinned reference: nested `text_config`. Factory builds the
    /// wrapper, whose own `sanitize` ADDS `language_model.` so the parameters
    /// address its `language_model` child — stripping first would leave every
    /// tensor addressed to nothing.
    @Test
    func theRawReferenceIsAWrappedModelAndItsPrefixIsKept() throws {
        let layout = try A.backboneLayout(
            configData: try Self.config([
                "model_type": "qwen3_5",
                "text_config": [
                    "model_type": "qwen3_5_text",
                    "num_hidden_layers": 64,
                    "mtp_num_hidden_layers": 1,
                ],
            ]))
        #expect(layout == .wrappedTextModel)
        #expect(
            layout.primaryKeyPrefixStrip == nil,
            """
            the wrapper layout wants a prefix strip. Qwen35Model.sanitize ADDS \
            `language_model.`; stripping it first addresses every tensor to \
            nothing and the load dies with keyNotFound on the first parameter.
            """
        )
    }

    /// A nested `text_config` decides even when the top-level `model_type` is
    /// the 3.6 spelling, because the STRUCTURE is what the factory keys on.
    @Test
    func theNestedStructureDecidesRegardlessOfTheModelTypeSpelling() throws {
        for modelType in ["qwen3_5", "qwen3_6", "qwen3_6_mtp"] {
            let layout = try A.backboneLayout(
                configData: try Self.config([
                    "model_type": modelType,
                    "text_config": ["model_type": "\(modelType)_text"],
                ]))
            #expect(layout == .wrappedTextModel, "\(modelType) misread")
        }
    }

    /// Pointing this track at some other checkpoint has to be a legible abort,
    /// not a keyNotFound from three layers down inside the factory.
    @Test
    func aNonQwenBackboneIsRefusedLoudly() throws {
        #expect(throws: (any Error).self) {
            _ = try A.backboneLayout(
                configData: try Self.config([
                    "model_type": "laguna", "num_hidden_layers": 32,
                ]))
        }
        #expect(throws: (any Error).self) {
            _ = try A.backboneLayout(
                configData: try Self.config(["num_hidden_layers": 32]))
        }
        #expect(throws: (any Error).self) {
            _ = try A.backboneLayout(configData: Data("not json".utf8))
        }
    }

    /// A Qwen config that is neither nested nor recognisably flat is refused
    /// rather than guessed at: a wrong guess picks the wrong prefix handling and
    /// the failure lands far from the cause.
    @Test
    func anUndeterminableQwenLayoutIsRefusedRatherThanGuessed() throws {
        #expect(throws: (any Error).self) {
            _ = try A.backboneLayout(
                configData: try Self.config(["model_type": "qwen3_5"]))
        }
    }

    /// The prefix this module strips must be the prefix the transform SELECTS
    /// on. They live in different targets (`SwiftTransform.textTowerPrefix` is
    /// internal to MLXFastTransform, which MLXFastModel does not depend on), so
    /// the only thing keeping them equal is this test.
    @Test
    func theStrippedPrefixIsTheTransformsOwnTextTowerPrefix() throws {
        let transform = try DFlashGateTextSupport.text(
            "Sources/MLXFastTransform/Transform.swift")
        #expect(
            transform.contains(
                "static let textTowerPrefix = \"\(A.SwiftTransformTextTowerPrefix)\""),
            """
            SwiftTransform.textTowerPrefix and \
            Qwen36MTPHeadAttachment.SwiftTransformTextTowerPrefix have drifted. \
            The transform names the tensors and this module renames them; if the \
            two disagree, the MTP worker loads a tree whose names it cannot \
            address.
            """
        )
    }

    /// It must be the SAME rule the repository's own eager loader applies to the
    /// same tree — that loader is what the serial track's natively byte-identical
    /// golden regeneration validated on box 3.
    @Test
    func theRenameMatchesTheEagerLoadersOwnRule() throws {
        let eager = try DFlashGateTextSupport.text(
            "Sources/MLXFastModel/RuntimeWeightLoading.swift")
        #expect(
            eager.contains(
                "languageModelPrefix = \"\(A.SwiftTransformTextTowerPrefix)\""))
        #expect(eager.contains("dropFirst(Self.languageModelPrefix.count)"))
    }

    /// The vendored strip must run BEFORE `sanitize` and BEFORE the quantize
    /// walk. That walk decides which submodules become quantized layers by
    /// asking whether `weights["<path>.scales"]` exists, so a rename applied
    /// after it would address quantized tensors to unquantized layers.
    @Test
    func theStripRunsBeforeSanitizeAndQuantize() throws {
        let load = try DFlashGateTextSupport.text(
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift")
        let strip = try #require(
            load.range(of: "strippingWeightKeyPrefix(strip, from: weights)"))
        let merge = try #require(
            load.range(
                of: "for source in _additionalWeightSources {",
                range: strip.upperBound ..< load.endIndex))
        let sanitize = try #require(
            load.range(
                of: "model.sanitize(weights: weights",
                range: merge.upperBound ..< load.endIndex))
        _ = try #require(
            load.range(
                of: "quantize(model: model)",
                range: sanitize.upperBound ..< load.endIndex),
            """
            the primary-tree strip, the head merge, sanitize and the quantize \
            walk are no longer in that order in loadWeights. The order is the \
            whole contract: the quantize walk keys on `<path>.scales` being \
            present, so anything merged or renamed after it arrives as quantized \
            tensors addressed to unquantized layers.
            """
        )
    }

    /// BOTH Qwen 3.6 classes are accepted, and the vendored one is spelled with
    /// its module qualifier.
    ///
    /// `MLXFastModel` declares its own `public enum Qwen35Model` (the scored
    /// serial forward's namespace), which SHADOWS the vendored class inside this
    /// module. Unqualified, the conformance silently binds to the local enum and
    /// fails with "non-class type 'Qwen35Model' cannot conform" — a diagnostic
    /// that points at the protocol rather than at the shadowing.
    @Test
    func bothQwenClassesConformAndTheVendoredOneIsQualified() throws {
        let source = try DFlashGateTextSupport.text(
            "Sources/MLXFastModel/Qwen36MTPTarget.swift")
        #expect(source.contains("extension Qwen35TextModel: Qwen36MTPTarget {}"))
        #expect(
            source.contains("extension MLXLLM.Qwen35Model: Qwen36MTPTarget {}"),
            """
            the vendored Qwen35Model conformance lost its MLXLLM. qualifier. \
            MLXFastModel declares its own `enum Qwen35Model`, which shadows the \
            vendored class inside this module.
            """
        )
        // The shadowing type is real and still there, so the qualifier is not
        // decoration.
        let shadow = try DFlashGateTextSupport.text(
            "Sources/MLXFastModel/Qwen35Model.swift")
        #expect(shadow.contains("public enum Qwen35Model {"))
    }

    /// The worker must accept either class through the protocol, never through a
    /// concrete-type cast.
    @Test
    func theWorkerAcceptsTheProtocolNotAConcreteClass() throws {
        let worker = try DFlashGateTextSupport.text(
            "Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift")
        #expect(worker.contains("as? any Qwen36MTPTarget"))
        #expect(
            !worker.contains("as? Qwen35TextModel"),
            """
            the worker is back to a concrete-type guard. That is what made it \
            reject the raw pinned reference (which builds Qwen35Model) while the \
            transformed tree it did accept could not load — zero tokens decoded, \
            in both directions.
            """
        )
        // And the session plumbing is protocol-typed all the way down.
        for path in [
            "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
            "Sources/MLXFastModel/Qwen36MTPReferenceSession.swift",
        ] {
            let source = try DFlashGateTextSupport.text(path)
            #expect(
                source.contains("any Qwen36MTPTarget"),
                "\(path) does not hold the target through the protocol")
            #expect(!source.contains(": Qwen35TextModel"))
        }
    }

    /// The staged head tree DOES carry `.gitattributes`, and the manifest pins it.
    ///
    /// The ranked workflow's `verify_cache` does a strict FLAT INVENTORY check in
    /// both directions — a record with no file is an error, a file with no record
    /// is an error — so the manifest's record set and the staged tree must agree
    /// file-for-file. A stock `snapshot_download` brings `.gitattributes` along
    /// because it is a genuine file of the pinned revision, so the manifest pins
    /// it rather than staging deleting it. Box 3 hit precisely this, from the
    /// other side: the earlier "drop it" posture is what failed the run.
    @Test
    func theHeadStagingInclusionIsDocumentedWhereItIsActionable() throws {
        let attachment = try DFlashGateTextSupport.text(
            "Sources/MLXFastModel/Qwen36MTPHeadAttachment.swift")
        #expect(
            attachment.contains(".gitattributes"),
            """
            the head-attachment docs no longer mention .gitattributes staging. \
            A stock HF snapshot carries that file, the manifest pins it, and \
            the workflow's flat inventory check rejects the whole head cache if \
            the staged tree and the record set disagree either way.
            """
        )
        #expect(
            attachment.contains("STAGING MUST INCLUDE `.gitattributes`"),
            """
            the head-attachment docs must state the CURRENT staging posture. \
            The retired posture told operators to `rm -f` the file, which is \
            exactly what broke run 31665285024.
            """
        )
        // The retired instruction must not survive anywhere in the doc block.
        #expect(!attachment.contains("rm -f <head>/.gitattributes"))
    }
}

// MARK: - the real rename + merge, over synthetic trees

/// Exercises the ACTUAL vendored code path with real safetensors on disk and no
/// model: the primary tree carries `language_model.`-prefixed names including
/// `lm_head.weight`, the head tree carries bare names, and the result must be
/// exactly the namespace a bare `Qwen35TextModel` addresses.
///
/// MLX-gated because constructing any `MLXArray` loads the default metallib.
/// `QwenMTPMetallibAvailabilityTests` below makes that gate fail loudly instead
/// of silently skipping when the metallib is absent.
@Suite
struct QwenMTPWeightRenameTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    @Test(.enabled(if: QwenMTPWeightRenameTests.enabled))
    func thePrimaryStripAndHeadMergeProduceTheTextModelNamespace() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwen-mtp-rename-\(UUID().uuidString)")
        let backbone = root.appendingPathComponent("weights")
        let head = root.appendingPathComponent("mtp-head")
        try FileManager.default.createDirectory(
            at: backbone, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: head, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // The shapes that actually failed on box 3: the transform's
        // `language_model.` prefix on every tensor, lm_head included.
        let scalar = MLXArray([Float(1)])
        try MLX.save(
            arrays: [
                "language_model.model.embed_tokens.weight": scalar,
                "language_model.model.layers.0.input_layernorm.weight": scalar,
                "language_model.model.norm.weight": scalar,
                "language_model.lm_head.weight": scalar,
            ],
            url: backbone.appendingPathComponent("model.safetensors"))
        try MLX.save(
            arrays: ["fc.weight": scalar, "norm.weight": scalar],
            url: head.appendingPathComponent("model.safetensors"))

        var merged = try strippingWeightKeyPrefix(
            Qwen36MTPHeadAttachment.SwiftTransformTextTowerPrefix,
            from: try loadArrays(directory: backbone))
        for (key, value) in try loadArrays(directory: head) {
            merged[Qwen36MTPHeadAttachment.headKeyPrefix + key] = value
        }

        #expect(
            Set(merged.keys) == [
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                "model.norm.weight",
                "lm_head.weight",
                "mtp.fc.weight",
                "mtp.norm.weight",
            ],
            "merged namespace is \(Set(merged.keys).sorted())"
        )
        // The exact key whose absence produced the box-3 abort.
        #expect(merged["lm_head.weight"] != nil)
        #expect(!merged.keys.contains { $0.hasPrefix("language_model.") })
    }

    /// A tree carrying BOTH spellings of a tensor must be refused: one of them
    /// would silently win and nothing downstream could tell which.
    @Test(.enabled(if: QwenMTPWeightRenameTests.enabled))
    func aTreeCarryingBothSpellingsIsRefused() throws {
        let scalar = MLXArray([Float(1)])
        #expect(throws: (any Error).self) {
            _ = try strippingWeightKeyPrefix(
                "language_model.",
                from: [
                    "language_model.lm_head.weight": scalar,
                    "lm_head.weight": scalar,
                ])
        }
    }
}

// MARK: - the metallib trap

@Suite
struct QwenMTPMetallibAvailabilityTests {
    /// MLX-GATED TESTS MUST NOT PASS BY NOT RUNNING.
    ///
    /// `tools/build-mlx-metallib.sh` writes `mlx.metallib` next to the WORKER
    /// binary only (`.build-worker/<config>/`). The xctest bundle is a third
    /// location, and Cmlx searches next to the running executable — so with
    /// `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` set but no metallib beside the test
    /// bundle, every MLX-gated test in this repository fails at the first
    /// `MLXArray` with "Failed to load the default metallib".
    ///
    /// This test converts that into ONE legible failure that names the fix,
    /// instead of a scatter of identical MLX errors across unrelated suites —
    /// and, critically, it means a box-3 run that sets the env and sees green
    /// really did exercise the gated paths.
    @Test(.enabled(if: ProcessInfo.processInfo
        .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"))
    func theMetallibIsBesideTheTestBundleWhenGatedTestsAreRequested() throws {
        // ANCHORED ON A TYPE IN THIS MODULE, not on argv[0] and not on
        // `Bundle.allBundles`. Two false-failure modes had to be ruled out, and
        // box 3 hit BOTH while the gated MLX tests were demonstrably running
        // (9s+ of real MLXArray work) and this check reported "missing":
        //
        //   * argv[0] under SwiftPM is the toolchain's `xctest` helper, which
        //     lives in the toolchain and never carries a metallib;
        //   * `Bundle.allBundles` does not list the .xctest bundle under
        //     swift-testing, so the loop contributed nothing -- and because
        //     `Bundle.main` still appended two toolchain paths, `candidates` was
        //     never empty and the "cannot identify a bundle" guard never fired.
        //
        // `Bundle(for:)` on a type defined in THIS module returns the bundle
        // that actually contains the loaded test code, which is what
        // `--all-build-roots` publishes into. That is the question being asked,
        // so ask it directly.
        let testBundle = Bundle(for: MetallibProbe.self)
        var candidates: [URL] = [
            testBundle.bundleURL.appendingPathComponent("Contents/MacOS"),
            testBundle.bundleURL.appendingPathComponent("Contents/Resources"),
            testBundle.bundleURL.deletingLastPathComponent(),
        ]
        for bundle in Bundle.allBundles
        where bundle.bundleURL.pathExtension == "xctest" {
            candidates.append(bundle.bundleURL
                .appendingPathComponent("Contents/MacOS"))
            candidates.append(bundle.bundleURL.deletingLastPathComponent())
        }
        let found = candidates.contains {
            FileManager.default.fileExists(
                atPath: $0.appendingPathComponent("mlx.metallib").path)
        }
        #expect(
            found,
            """
            MLXFAST_RUN_MLX_RUNTIME_TESTS=1 is set but no mlx.metallib was found \
            beside the loaded test bundle. Searched: \
            \(candidates.map(\.path).joined(separator: ", ")). \
            tools/build-mlx-metallib.sh publishes next to the WORKER binary only \
            and its test-bundle publish is gated on a debug configuration it does \
            not default to. Run `tools/build-mlx-metallib.sh --all-build-roots`. \
            Without it every MLX-gated test fails at its first MLXArray, and a \
            run that SKIPS them exits 0 having proven nothing.
            """
        )
    }
}

/// Anchor for `Bundle(for:)` in `QwenMTPMetallibAvailabilityTests`.
///
/// A class, and in this module, for the only reason that matters: `Bundle(for:)`
/// takes an AnyClass and resolves to the bundle containing that class's binary.
/// swift-testing suites are structs, so without this there is no type here to
/// hand it, which is how the check ended up asking `Bundle.allBundles` a
/// question it does not answer.
private final class MetallibProbe {}
