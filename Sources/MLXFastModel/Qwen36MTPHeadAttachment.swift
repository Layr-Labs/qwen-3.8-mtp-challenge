import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon

// Merge-at-load for the SEPARATELY PINNED Qwen 3.6 MTP head.
//
// THE DELTA FROM THE VALIDATED DRIVER, STATED PLAINLY. The MTP session's driver
// loaded a MERGED checkpoint: a directory produced by `merge-mtp-head.sh`, which
// hardlinks the backbone shards, writes the head's tensors into a new shard under
// an `mtp.` prefix, registers them in the index, and rewrites `config.json` to
// declare `model_type: qwen3_6_mtp` with `mtp_num_hidden_layers: 1`. The ranked
// track does NOT do that: the operator's Q8 decision is SEPARATE TREES, both
// upstream-revision-pinned and both byte-verified against their own manifests
// before anything loads. So the merge happens at load time, in memory, here.
//
// What has to be equivalent, and how each half is checked:
//
//  1. TENSOR NAMES. The published head (mlx-community/Qwen3.6-27B-MTP-4bit) uses
//     BARE names — `fc.*`, `norm.weight`, `pre_fc_norm_embedding.weight`,
//     `pre_fc_norm_hidden.weight`, `layers.0.*`. The merge script prefixes every
//     one with `mtp.`; `_additionalWeightSources` does exactly the same thing,
//     with the same prefix, before `sanitize` and before the quantization walk.
//     `expectedHeadTensorCount` pins the count the pinned revision carries, so a
//     head tree that lost or gained a tensor is a load-time failure rather than a
//     model that silently drafts from a partly-uninitialized head.
//
//  2. QUANTIZATION WIRING. `quantize(model:)` decides which submodules become
//     quantized layers by asking whether `weights["<path>.scales"]` exists. The
//     merge therefore has to land BEFORE that walk, which is why the vendored
//     hook sits where it does and not in a post-load `update(parameters:)`.
//
//  3. CONFIG SEMANTICS. The merge script sets `model_type: qwen3_6_mtp` and
//     `mtp_num_hidden_layers: 1`. Neither rewrite is needed here and NEITHER IS
//     SKIPPED SILENTLY: the pinned backbone's own `text_config` already carries
//     `mtp_num_hidden_layers: 1` (it is the same architecture family; the
//     backbone revision simply ships no `mtp.*` TENSORS), and the transform
//     copies `text_config` verbatim into the runtime `config.json` — so the
//     loaded configuration's `mtpNumHiddenLayers` is 1 either way, and
//     `Qwen35TextModel.init` attaches the head on `mtpNumHiddenLayers > 0 &&
//     _qwen35MTPEnabled`. The `qwen3_6_mtp` model_type alias exists only to route
//     the MERGED directory through the same factory entry the un-merged one
//     already reaches as `qwen3_5_text`. `verifyHeadConfiguration` asserts the
//     two config semantics the loader actually keys on, so a head tree whose
//     config drifted away from the backbone's family fails loudly.
//
//  4. PYTHON HAZARD, CARRIED FORWARD. mlx-lm's `qwen3_5` `sanitize()` treats any
//     `mtp.*` key as a signal to +1-shift every trunk norm weight, which corrupts
//     the model. That hazard belongs to Python consumers of a MERGED directory
//     (QWEN36-BOX3-MTP-RESPONSE.md §3.2); the Swift loader is unaffected, and this
//     path never writes a merged directory for anyone else to read. The note is
//     kept here because this is now the place the recipe lives.

/// Loads and validates the separately pinned Qwen 3.6 MTP head alongside a
/// backbone weights tree.
public enum Qwen36MTPHeadAttachment {
    /// Tensors the pinned head revision (83795d54) carries. Single source of
    /// truth is `MLXFastConstants`, because the trusted CLI reports the same
    /// number in the evidence payload and links no model code.
    public static let expectedHeadTensorCount =
        MLXFastConstants.qwenMTPHeadTensorCount

    /// The key prefix the head's bare tensor names are merged under.
    public static let headKeyPrefix = "mtp."

    /// Which class the factory will build for a backbone tree, and therefore
    /// which tensor-name namespace that tree's weights have to land in.
    ///
    /// The two trees this track can be pointed at disagree, and the disagreement
    /// is the structural blocker the first migration hit:
    ///
    ///   * the TRANSFORMED tree (`weights/`, the primary — it is what the ranked
    ///     flow provides and what `qwen-mtp-weights.sha256` pins) carries a
    ///     `config.json` that is the reference's `text_config` verbatim, so it
    ///     declares `qwen3_5_text` and the factory builds a bare
    ///     `Qwen35TextModel`. But its 1,847 tensor names still carry the
    ///     `language_model.` text-tower prefix the transform selects on
    ///     (`SwiftTransform.textTowerPrefix`), because that name set is a pinned
    ///     contract. A bare text model addresses `model.*` / `lm_head.weight`,
    ///     so the prefix must be stripped at load — which is precisely what this
    ///     repository's own eager loader (`RuntimeWeightNameTracker`) already
    ///     does for the same tree, by the same rule, and which the serial track's
    ///     natively byte-identical golden regeneration validated on box 3.
    ///
    ///   * the RAW pinned reference declares `qwen3_5` with a nested
    ///     `text_config`, so the factory builds the multimodal `Qwen35Model`
    ///     wrapper. That class's own `sanitize` ADDS `language_model.` so the
    ///     parameters address its `language_model` child, so stripping the
    ///     prefix first would leave every tensor addressed to nothing.
    ///
    /// Same prefix, opposite handling, and the ONLY place the answer is written
    /// down is the config's `model_type`. Hence this enum rather than a guess.
    public enum BackboneLayout: String, Sendable {
        /// Flat text config (`qwen3_5_text`): factory builds `Qwen35TextModel`.
        case textModel
        /// Nested config (`qwen3_5` / `qwen3_6`): factory builds `Qwen35Model`.
        case wrappedTextModel

        /// The prefix to strip from the primary tree's tensor names, if any.
        public var primaryKeyPrefixStrip: String? {
            switch self {
            case .textModel: return SwiftTransformTextTowerPrefix
            case .wrappedTextModel: return nil
            }
        }
    }

    /// The text-tower prefix the transform selects on. Duplicated from
    /// `SwiftTransform.textTowerPrefix` (which is `internal` to MLXFastTransform,
    /// a target MLXFastModel does not depend on) and pinned equal to it by
    /// `QwenMTPBackboneLayoutTests`.
    public static let SwiftTransformTextTowerPrefix = "language_model."

    /// Read the backbone's `config.json` and decide its layout.
    ///
    /// Refuses anything that is not a Qwen 3.5/3.6 family config LOUDLY: pointing
    /// this track at some other checkpoint has to be a legible abort, not a
    /// keyNotFound from three layers down inside the factory.
    public static func backboneLayout(configData: Data) throws -> BackboneLayout {
        guard let root = try? JSONSerialization.jsonObject(with: configData)
            as? [String: Any],
            let modelType = root["model_type"] as? String
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP backbone config.json declares no model_type")
        }
        guard modelType.hasPrefix("qwen3_5") || modelType.hasPrefix("qwen3_6")
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP backbone declares model_type \(modelType), which is "
                    + "not a Qwen 3.5/3.6 family checkpoint")
        }
        // A nested `text_config` is what makes the factory build the wrapper. The
        // model_type suffix agrees with it on every checkpoint this track pins,
        // and where they could disagree the STRUCTURE is what the factory keys
        // on, so the structure decides.
        if root["text_config"] is [String: Any] {
            return .wrappedTextModel
        }
        guard modelType.hasSuffix("_text") || root["num_hidden_layers"] != nil
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP backbone declares \(modelType) with neither a "
                    + "text_config nor flat text-model fields; its layout cannot "
                    + "be determined")
        }
        return .textModel
    }

    public static func backboneLayout(directory: URL) throws -> BackboneLayout {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL) else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP backbone tree carries no config.json: "
                    + directory.path)
        }
        return try backboneLayout(configData: data)
    }

    /// Run `body` with the head tree registered as an additional weight source,
    /// the primary tree's key rewrite selected from its own config, and
    /// `_qwen35MTPEnabled` set — restoring all three afterwards.
    ///
    /// ALL THREE globals are restored on EVERY exit path. They are process-global
    /// and read by the next load of any model; leaking one would silently change
    /// how an unrelated later load behaves, which on a worker that serves a
    /// reference replay after the candidate is exactly the kind of cross-phase
    /// coupling this track cannot have.
    @discardableResult
    public static func withHeadAttached<T>(
        backboneDirectory: URL,
        headDirectory: URL,
        _ body: (BackboneLayout) throws -> T
    ) throws -> T {
        let layout = try backboneLayout(directory: backboneDirectory)
        try verifyHeadTree(headDirectory)
        let previousSources = _additionalWeightSources
        let previousStrip = _primaryWeightKeyPrefixStrip
        let previousEnabled = _qwen35MTPEnabled
        _additionalWeightSources = [
            AdditionalWeightSource(
                directory: headDirectory, keyPrefix: headKeyPrefix)
        ]
        _primaryWeightKeyPrefixStrip = layout.primaryKeyPrefixStrip
        _qwen35MTPEnabled = true
        defer {
            _additionalWeightSources = previousSources
            _primaryWeightKeyPrefixStrip = previousStrip
            _qwen35MTPEnabled = previousEnabled
        }
        let result = try body(layout)
        // Untimed. Completes the later-window 2-pass SDPA JIT the live crown
        // (fkiene `59b321ee`, session `warmTargetLaterWindowSDPA`) started:
        // that warm compiles qL=1/5/4 at kL>=1024. AttentionUtils splits
        // verify widths 6–7 into 5+(qL-5), so chunk B is qL=1 (width 6, already
        // compiled) or qL=2 (width 7, the central-pair height, NOT compiled).
        // Width 8's chunk B is qL=3. 2-pass threadgroup is (32, gqa=6, qL), so
        // those qL values are distinct pipeline hashes. Dummy zeros, discarded.
        warmMissingLaterWindowSDPA()
        return result
    }

    /// kL=1024 / bf16 / causal / 4 KV heads × 6 GQA. On arch `'s'` (M5 Max)
    /// `N==1024` keeps `blocks=64`, matching the second half of the ranked
    /// 512+512 window. Does not touch live caches or the session file.
    private static func warmMissingLaterWindowSDPA() {
        let kvHeads = 4
        let qHeads = 24
        let headDim = 256
        let kL = 1024
        let scale = 1 / Float(headDim).squareRoot()
        let keys = MLXArray.zeros([1, kvHeads, kL, headDim], dtype: .bfloat16)
        let values = MLXArray.zeros([1, kvHeads, kL, headDim], dtype: .bfloat16)
        var outs: [MLXArray] = []
        for qL in [2, 3] {
            let queries = MLXArray.zeros(
                [1, qHeads, qL, headDim], dtype: .bfloat16)
            outs.append(
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: .causal
                )
            )
        }
        eval(outs)
    }

    /// Structural checks that do not need MLX and are therefore unit-testable.
    ///
    /// Deliberately NOT a hash check: the byte identity of the head tree is the
    /// ranked workflow's job (`verify_cache` against
    /// `fixtures/qwen3_8_27b_mtp_head.sha256`, every run, before anything loads).
    /// Repeating it here would be a second, weaker copy of a stronger gate; what
    /// this adds is the shape the LOADER depends on.
    ///
    /// STAGING MUST INCLUDE `.gitattributes`. The pinned head repo carries 8
    /// files and the manifest pins all 8, `.gitattributes` among them, exactly as
    /// the backbone manifest now pins all 16 of its revision's files.
    /// `verify_cache` runs a strict FLAT INVENTORY check in BOTH directions — a
    /// manifest record with no file on disk is an error, and a file on disk the
    /// manifest does not name is an error — so the two sets have to agree
    /// exactly. The earlier posture here was the opposite (pin 7, delete the
    /// file from the cache), and it was wrong in the direction that fails
    /// closed: `.gitattributes` is a genuine file of the pinned upstream
    /// revision, a stock `snapshot_download` brings it along, and the inventory
    /// half then rejected the whole head cache. Box 3 hit exactly this. Pinning
    /// the file is the fix; staging must NOT drop it.
    public static func verifyHeadTree(_ headDirectory: URL) throws {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(
            atPath: headDirectory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head directory does not exist: \(headDirectory.path)")
        }
        let indexURL = headDirectory.appendingPathComponent(
            "model.safetensors.index.json")
        if let indexData = try? Data(contentsOf: indexURL) {
            try verifyHeadIndex(indexData)
            let configURL = headDirectory.appendingPathComponent("config.json")
            guard let configData = try? Data(contentsOf: configURL) else {
                throw MLXFastError.invalidInput(
                    "the Qwen MTP head tree carries no config.json")
            }
            try verifyHeadConfiguration(configData)
            return
        }
        // DECLARED-HEAD STAGING. The ranked runner resolves a `remote` head
        // declaration by fetching exactly `model.safetensors` — no config, no
        // index — and digest-verifies it against the manifest before the
        // sandbox opens. The byte identity is therefore already enforced
        // upstream of this check; what the loader still needs is the same
        // STRUCTURAL shape it asserts on the pinned tree, read from the
        // safetensors header itself: bare (un-prefixed) names and the tensors
        // the merge cannot do without.
        let safetensorsURL = headDirectory.appendingPathComponent(
            "model.safetensors")
        guard fileManager.fileExists(atPath: safetensorsURL.path) else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head tree carries neither "
                    + "model.safetensors.index.json nor model.safetensors")
        }
        let names = try safetensorsTensorNames(safetensorsURL)
        if let prefixed = names.first(where: { $0.hasPrefix(headKeyPrefix) }) {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head tree already carries prefixed tensor names "
                    + "(e.g. \(prefixed)); this loader merges a BARE head tree and "
                    + "would double-prefix a pre-merged one")
        }
        for required in ["fc.weight", "norm.weight", "pre_fc_norm_hidden.weight"] {
            guard names.contains(required) else {
                throw MLXFastError.invalidInput(
                    "the Qwen MTP head safetensors is missing \(required)")
            }
        }
    }

    /// Tensor names from a safetensors file header (8-byte little-endian
    /// header length, then a JSON object whose keys are the tensor names plus
    /// an optional `__metadata__`).
    static func safetensorsTensorNames(_ url: URL) throws -> Set<String> {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: 8),
              lengthData.count == 8
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head safetensors is too short to carry a header")
        }
        var headerLength: UInt64 = 0
        for (index, byte) in lengthData.enumerated() {
            headerLength |= UInt64(byte) << (8 * UInt64(index))
        }
        guard headerLength > 0, headerLength < 100_000_000 else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head safetensors declares an implausible header "
                    + "length \(headerLength)")
        }
        guard let headerData = try handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength),
              let header = try? JSONSerialization.jsonObject(with: headerData)
                  as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head safetensors header is not readable JSON")
        }
        return Set(header.keys.filter { $0 != "__metadata__" })
    }

    /// The head index must name exactly the pinned tensor set, under BARE names.
    ///
    /// A head tree that already carries `mtp.`-prefixed names is REFUSED rather
    /// than accommodated: it would be a pre-merged artifact, the prefix would be
    /// applied twice, and the resulting `mtp.mtp.*` keys would be dropped by
    /// `update(parameters:)` leaving the head at its initialization values — a
    /// model that runs, drafts nonsense, accepts ~nothing, and still reports
    /// exact. Failing the load is the only outcome that is not silent.
    public static func verifyHeadIndex(_ indexData: Data) throws {
        guard let root = try? JSONSerialization.jsonObject(with: indexData)
            as? [String: Any],
            let weightMap = root["weight_map"] as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head index is not a JSON object with a weight_map")
        }
        // The organizer-pinned head carries exactly `expectedHeadTensorCount`
        // tensors; a DECLARED head (2026-08-14 contract: the head weights are
        // competitive surface, digest-pinned by mtp-head.manifest.json) may
        // carry a different count — e.g. a quantized head's weight/scales/
        // biases triples. The stale-pinned-tree hazard the exact-count check
        // guarded against is enforced upstream by the ranked verify_cache and
        // setup's digest check, so here the structural requirements are the
        // bare namespace and the required tensors below.
        guard weightMap.count >= 3 else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head index names only \(weightMap.count) tensors")
        }
        if let prefixed = weightMap.keys.first(where: {
            $0.hasPrefix(headKeyPrefix)
        }) {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head tree already carries prefixed tensor names "
                    + "(e.g. \(prefixed)); this loader merges a BARE head tree and "
                    + "would double-prefix a pre-merged one")
        }
        for required in ["fc.weight", "norm.weight", "pre_fc_norm_hidden.weight"] {
            guard weightMap[required] != nil else {
                throw MLXFastError.invalidInput(
                    "the Qwen MTP head index is missing \(required)")
            }
        }
    }

    /// The two config semantics the loader keys on, asserted on the HEAD tree.
    ///
    /// The head's own `config.json` is not what the model is built from — the
    /// backbone's is — so this is a compatibility assertion, not a load input: it
    /// proves the head being merged belongs to the same architecture family and
    /// declares the single MTP layer the backbone's `mtp_num_hidden_layers: 1`
    /// promises.
    public static func verifyHeadConfiguration(_ configData: Data) throws {
        guard let root = try? JSONSerialization.jsonObject(with: configData)
            as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head config.json is not a JSON object")
        }
        guard let modelType = root["model_type"] as? String,
              modelType.hasPrefix("qwen3_5") || modelType.hasPrefix("qwen3_6")
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head declares model_type "
                    + "\(root["model_type"] as? String ?? "<missing>"), which is not "
                    + "a Qwen 3.5/3.6 MTP head")
        }
        let textConfig = (root["text_config"] as? [String: Any]) ?? root
        guard let mtpLayers = textConfig["mtp_num_hidden_layers"] as? Int,
              mtpLayers == 1
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head declares mtp_num_hidden_layers "
                    + "\(textConfig["mtp_num_hidden_layers"] as? Int ?? -1); the "
                    + "pinned head is a single-layer head and the backbone's own "
                    + "text_config promises exactly one")
        }
    }
}
