// CBv2ModelTests.swift
//
// ContinuousBatchingV2 — WS-F (model adaptation) unit tests.
//
//  - CBv2ModelLayerKindTests: exact [CBv2LayerKind] derivation for Gemma 4
//    (35 layers, shared map congruent with previousKvs, K-eq-V, asymmetric
//    head dims) and GPT-OSS (24 layers alternating, sinks).
//  - CBv2ModelGemma4ForwardTests / CBv2ModelGPTOSSForwardTests: tiny-config
//    random-weight forward smoke tests driving the v2 attention branch with
//    a mock CBv2AttendingLayerCache — no model weights downloaded.
//
// The mock lives in this file per the integration protocol: workstreams code
// against CBv2Contracts.swift and mock the concrete classes other
// workstreams own.

import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

// MARK: - Mocks

/// Minimal contiguous per-sequence KV store (concat-grow, no eviction).
private final class CBv2ModelMockSequenceKV: CBv2SequenceKV {
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private(set) var absoluteOffset: Int = 0

    var retainedCount: Int { keys?.dim(2) ?? 0 }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            keys = concatenated([existingKeys, newKeys], axis: 2)
            values = concatenated([existingValues, newValues], axis: 2)
        } else {
            keys = newKeys
            values = newValues
        }
        absoluteOffset += newKeys.dim(2)
        return (keys!, values!)
    }

    func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        guard let keys, let values else {
            fatalError("snapshot of empty mock sequence KV")
        }
        return (keys, values, absoluteOffset)
    }

    func rollback(_ n: Int) {
        guard let existingKeys = keys, let existingValues = values,
            n <= existingKeys.dim(2)
        else {
            fatalError("rollback(\(n)) exceeds retained KV")
        }
        let keep = existingKeys.dim(2) - n
        keys = existingKeys[0..., 0..., 0 ..< keep, 0...]
        values = existingValues[0..., 0..., 0 ..< keep, 0...]
        absoluteOffset -= n
    }

    var byteCount: Int { (keys?.nbytes ?? 0) + (values?.nbytes ?? 0) }
}

/// Mock CBv2AttendingLayerCache: per-row storage + per-row SDPA, with call
/// recording so the tests can assert the model's dispatch behavior
/// (offsets snapshotted before update, shared layers borrowing from the
/// right source, no mask construction, no legacy update calls).
private final class CBv2ModelMockLayerCache: CBv2AttendingLayerCache, KVCache {

    enum Event: Equatable {
        case offsetsRead
        case updateAndAttend
        case attendBorrowing
    }

    struct UpdateCall {
        var offsetsBefore: [Int]
        var queryShape: [Int]
        var keyShape: [Int]
        var valueShape: [Int]
        var sinksProvided: Bool
    }

    struct BorrowCall {
        var sourceLayerIndex: Int
        var queryShape: [Int]
        var sinksProvided: Bool
    }

    let layerIndex: Int
    let kind: CBv2LayerKind
    var mockRows: [CBv2ModelMockSequenceKV]

    private(set) var events: [Event] = []
    private(set) var updateCalls: [UpdateCall] = []
    private(set) var borrowCalls: [BorrowCall] = []
    private(set) var positionOffsetsReads = 0
    private(set) var makeMaskCalls = 0

    init(layerIndex: Int, kind: CBv2LayerKind, rowCount: Int) {
        self.layerIndex = layerIndex
        self.kind = kind
        // Shared layers own no storage (contract: entries with
        // sharesKVWithLayer != nil receive no sequence state).
        self.mockRows =
            kind.sharesKVWithLayer == nil
            ? (0 ..< rowCount).map { _ in CBv2ModelMockSequenceKV() } : []
    }

    // MARK: CBv2AttendingLayerCache

    var rows: [CBv2SequenceKV] { mockRows }

    func setRows(_ rows: [CBv2SequenceKV]) {
        mockRows = rows.map { row in
            guard let mock = row as? CBv2ModelMockSequenceKV else {
                fatalError("mock layer cache only binds mock rows")
            }
            return mock
        }
    }

    var positionOffsets: MLXArray {
        positionOffsetsReads += 1
        events.append(.offsetsRead)
        return MLXArray(mockRows.map { Int32($0.absoluteOffset) })
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(
            mockRows.count == queries.dim(0),
            "row count \(mockRows.count) != batch dim \(queries.dim(0))")
        events.append(.updateAndAttend)
        updateCalls.append(
            UpdateCall(
                offsetsBefore: mockRows.map { $0.absoluteOffset },
                queryShape: queries.shape, keyShape: keys.shape,
                valueShape: values.shape, sinksProvided: sinks != nil))

        let L = queries.dim(2)
        var outputs = [MLXArray]()
        for (i, row) in mockRows.enumerated() {
            let (rowKeys, rowValues) = row.update(
                keys: keys[i ..< (i + 1)], values: values[i ..< (i + 1)])
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: queries[i ..< (i + 1)], keys: rowKeys, values: rowValues,
                    scale: scale, mask: L == 1 ? .none : .causal))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    func attendBorrowing(
        source: CBv2AttendingLayerCache,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(kind.sharesKVWithLayer != nil, "attendBorrowing on a non-shared layer")
        events.append(.attendBorrowing)
        borrowCalls.append(
            BorrowCall(
                sourceLayerIndex: source.layerIndex,
                queryShape: queries.shape, sinksProvided: sinks != nil))

        guard let mockSource = source as? CBv2ModelMockLayerCache else {
            fatalError("mock attendBorrowing requires a mock source")
        }
        precondition(
            mockSource.mockRows.count == queries.dim(0),
            "source row count \(mockSource.mockRows.count) != batch dim \(queries.dim(0))")

        let L = queries.dim(2)
        var outputs = [MLXArray]()
        for (i, row) in mockSource.mockRows.enumerated() {
            guard let rowKeys = row.keys, let rowValues = row.values else {
                fatalError("attendBorrowing before the source layer's update")
            }
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: queries[i ..< (i + 1)], keys: rowKeys, values: rowValues,
                    scale: scale, mask: L == 1 ? .none : .causal))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    // MARK: KVCache conformance (models on the v2 branch must never use these)

    var offset: Int { mockRows.map { $0.absoluteOffset }.max() ?? 0 }
    var maxSize: Int? { nil }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("v2 models must call updateAndAttend, not KVCache.update")
    }

    var state: [MLXArray] {
        get { [] }
        set { }
    }
    var metaState: [String] {
        get { [] }
        set { }
    }
    var isTrimmable: Bool { false }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        makeMaskCalls += 1
        return .none
    }

    func copy() -> any KVCache {
        fatalError("copy() unsupported in mock")
    }

    func innerState() -> [MLXArray] { [] }
}

// MARK: - Layer kind derivation

@Suite("CBv2 layer kind derivation")
struct CBv2ModelLayerKindTests {

    /// Real 4B-shaped topology (pattern 5, 35 layers, trailing 20 shared)
    /// with production attention geometry.
    private func gemma4PatternConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 1536,
                "num_hidden_layers": 35,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 20,
                "sliding_window": 512,
                "sliding_window_pattern": 5,
                "tie_word_embeddings": true,
                "vocab_size": 1024,
                "vocab_size_per_layer_input": 1024,
                "rms_norm_eps": 1e-6
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// 26B-shaped K-eq-V variant: attention_k_eq_v with dedicated global KV
    /// head count on full-attention layers.
    private func gemma4KeqVConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 1536,
                "num_hidden_layers": 35,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 2,
                "attention_k_eq_v": true,
                "num_kv_shared_layers": 20,
                "sliding_window": 512,
                "sliding_window_pattern": 5,
                "tie_word_embeddings": true,
                "vocab_size": 1024,
                "vocab_size_per_layer_input": 1024,
                "rms_norm_eps": 1e-6
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    @Test func patternLayerTypesDerivation() {
        let types = CBv2LayerKindDerivation.layerTypes(slidingWindowPattern: 5, numLayers: 35)
        #expect(types.count == 35)
        for (i, t) in types.enumerated() {
            #expect(t == (i % 5 == 4 ? "full_attention" : "sliding_attention"))
        }
    }

    @Test func gemma4ExactKinds() throws {
        let config = try gemma4PatternConfig()
        let kinds = config.cbv2LayerKinds
        #expect(kinds.count == 35)

        for (i, kind) in kinds.enumerated() {
            let isFull = i % 5 == 4
            if isFull {
                #expect(kind.attention == .full, "layer \(i)")
                #expect(kind.headDim == 512, "layer \(i)")
            } else {
                #expect(kind.attention == .slidingWindow(512), "layer \(i)")
                #expect(kind.headDim == 256, "layer \(i)")
            }
            #expect(kind.hasSinks == false, "layer \(i)")
            #expect(kind.kvHeads == 1, "layer \(i)")
            #expect(kind.queryHeads == 8, "layer \(i)")

            // Trailing 20 layers share KV with the last non-shared layer of
            // the same type: sliding → 13, full → 14.
            if i < 15 {
                #expect(kind.sharesKVWithLayer == nil, "layer \(i)")
            } else {
                #expect(kind.sharesKVWithLayer == (isFull ? 14 : 13), "layer \(i)")
            }
        }
    }

    @Test func gemma4SharedMapMatchesModelPreviousKvs() throws {
        // Tiny dims, real topology: derivation must be congruent with the
        // trunk's previousKvs map (identity == "no source").
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 35,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 20,
                "sliding_window": 16,
                "sliding_window_pattern": 5,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "vocab_size": 32,
                "vocab_size_per_layer_input": 32,
                "rms_norm_eps": 1e-6
            }
            """
        let config = try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
        let model = Gemma4TextModel(config)

        let sources = CBv2LayerKindDerivation.gemma4SharedKVSources(
            layerTypes: config.layerTypes, numKvSharedLayers: config.numKvSharedLayers)
        #expect(sources.count == model.model.previousKvs.count)
        for (j, prev) in model.model.previousKvs.enumerated() {
            #expect((sources[j] ?? j) == prev, "layer \(j)")
        }
        // And the kinds carry the same map.
        for (j, kind) in config.cbv2LayerKinds.enumerated() {
            #expect(kind.sharesKVWithLayer == sources[j], "layer \(j)")
        }
    }

    @Test func gemma4KeqVKvHeadAsymmetry() throws {
        let config = try gemma4KeqVConfig()
        let kinds = config.cbv2LayerKinds
        for (i, kind) in kinds.enumerated() {
            if i % 5 == 4 {
                // K-eq-V applies to full-attention layers only.
                #expect(kind.kvHeads == 2, "layer \(i)")
                #expect(kind.headDim == 512, "layer \(i)")
            } else {
                #expect(kind.kvHeads == 1, "layer \(i)")
                #expect(kind.headDim == 256, "layer \(i)")
            }
        }
    }

    @Test func gemma4KeqVWithoutGlobalHeadsFallsBack() {
        let kinds = CBv2LayerKindDerivation.gemma4LayerKinds(
            layerTypes: ["sliding_attention", "full_attention"],
            slidingWindow: 512,
            numKvSharedLayers: 0,
            headDim: 256,
            globalHeadDim: 512,
            numAttentionHeads: 8,
            numKeyValueHeads: 4,
            numGlobalKeyValueHeads: nil,
            attentionKeqV: true
        )
        #expect(kinds[0].kvHeads == 4)
        #expect(kinds[1].kvHeads == 4)
    }

    @Test func gemma4NoSharedLayers() {
        let sources = CBv2LayerKindDerivation.gemma4SharedKVSources(
            layerTypes: CBv2LayerKindDerivation.layerTypes(
                slidingWindowPattern: 5, numLayers: 10),
            numKvSharedLayers: 0)
        #expect(sources.allSatisfy { $0 == nil })
    }

    @Test func gptossExactKinds() throws {
        // Real 20B shape: 24 layers alternating sliding(128)/full, headDim
        // 64, 64 query heads / 8 KV heads, sinks on every layer.
        let json = """
            {
                "model_type": "gpt_oss",
                "num_hidden_layers": 24,
                "num_local_experts": 32,
                "num_experts_per_tok": 4,
                "vocab_size": 201088,
                "rms_norm_eps": 1e-5,
                "hidden_size": 2880,
                "intermediate_size": 2880,
                "head_dim": 64,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "sliding_window": 128
            }
            """
        let config = try JSONDecoder.json5().decode(
            GPTOSSConfiguration.self, from: Data(json.utf8))
        let kinds = config.cbv2LayerKinds
        #expect(kinds.count == 24)
        for (i, kind) in kinds.enumerated() {
            if i % 2 == 0 {
                #expect(kind.attention == .slidingWindow(128), "layer \(i)")
            } else {
                #expect(kind.attention == .full, "layer \(i)")
            }
            #expect(kind.sharesKVWithLayer == nil, "layer \(i)")
            #expect(kind.hasSinks == true, "layer \(i)")
            #expect(kind.headDim == 64, "layer \(i)")
            #expect(kind.kvHeads == 8, "layer \(i)")
            #expect(kind.queryHeads == 64, "layer \(i)")
        }
    }

    @Test func gptossExplicitLayerTypesRespected() {
        let kinds = CBv2LayerKindDerivation.gptossLayerKinds(
            layerTypes: ["full_attention", "full_attention", "sliding_attention"],
            numHiddenLayers: 3,
            slidingWindow: 128,
            headDim: 64,
            numAttentionHeads: 64,
            numKeyValueHeads: 8
        )
        #expect(kinds[0].attention == .full)
        #expect(kinds[1].attention == .full)
        #expect(kinds[2].attention == .slidingWindow(128))
    }
}

// MARK: - Gemma 4 forward smoke tests (v2 branch)

@Suite("CBv2 Gemma4 v2 attention branch")
struct CBv2ModelGemma4ForwardTests {

    /// 4-layer tiny config: [sliding, full, sliding, full], last 2 KV-shared
    /// (2 → 0, 3 → 1), asymmetric head dims 8/16, PLE and MoE off.
    private func tinyConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": 8,
                "global_head_dim": 16,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": 16,
                "final_logit_softcapping": 30.0,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func makeModelAndCaches(rowCount: Int = 1) throws -> (
        Gemma4TextModel, [CBv2ModelMockLayerCache]
    ) {
        let config = try tinyConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        var mocks = [CBv2ModelMockLayerCache]()
        let caches = model.newCacheV2 { index, kind in
            let mock = CBv2ModelMockLayerCache(layerIndex: index, kind: kind, rowCount: rowCount)
            mocks.append(mock)
            return mock
        }
        #expect(caches.count == 4)
        return (model, mocks)
    }

    @Test func factoryReceivesDerivedKinds() throws {
        let config = try tinyConfig()
        let (_, mocks) = try makeModelAndCaches()
        let kinds = config.cbv2LayerKinds
        #expect(mocks.map { $0.layerIndex } == [0, 1, 2, 3])
        #expect(mocks.map { $0.kind } == kinds)
        #expect(kinds.map { $0.sharesKVWithLayer } == [nil, nil, 0, 1])
        #expect(kinds.map { $0.headDim } == [8, 16, 8, 16])
    }

    @Test func prefillAndDecodeThroughV2Branch() throws {
        let (model, mocks) = try makeModelAndCaches()
        let caches: [KVCache] = mocks

        // 5-token prompt + 3 decode steps.
        let prompt = MLXArray([Int32(1), 2, 3, 4, 5])[.newAxis, .ellipsis]
        var logits = model(prompt, cache: caches)
        eval(logits)
        #expect(logits.shape == [1, 5, 64])

        for step in 0 ..< 3 {
            let token = MLXArray([Int32(7 + step)])[.newAxis, .ellipsis]
            logits = model(token, cache: caches)
            eval(logits)
            #expect(logits.shape == [1, 1, 64])
        }

        // Non-shared layers: 4 updateAndAttend calls with per-step offsets
        // snapshotted BEFORE the update (0, 5, 6, 7), correct shapes.
        for idx in [0, 1] {
            let mock = mocks[idx]
            #expect(mock.updateCalls.count == 4, "layer \(idx)")
            #expect(mock.borrowCalls.isEmpty, "layer \(idx)")
            #expect(
                mock.updateCalls.map { $0.offsetsBefore } == [[0], [5], [6], [7]],
                "layer \(idx)")
            let headDim = idx == 0 ? 8 : 16
            #expect(mock.updateCalls[0].queryShape == [1, 2, 5, headDim], "layer \(idx)")
            #expect(mock.updateCalls[0].keyShape == [1, 1, 5, headDim], "layer \(idx)")
            #expect(mock.updateCalls[1].queryShape == [1, 2, 1, headDim], "layer \(idx)")
            #expect(mock.updateCalls[1].keyShape == [1, 1, 1, headDim], "layer \(idx)")
            #expect(mock.mockRows[0].absoluteOffset == 8, "layer \(idx)")
            // Gemma has no sinks.
            #expect(mock.updateCalls.allSatisfy { !$0.sinksProvided }, "layer \(idx)")
            // Offsets are read (snapshot) exactly once per forward, BEFORE
            // the update — strict alternation.
            #expect(mock.positionOffsetsReads == 4, "layer \(idx)")
            #expect(
                mock.events == [
                    .offsetsRead, .updateAndAttend,
                    .offsetsRead, .updateAndAttend,
                    .offsetsRead, .updateAndAttend,
                    .offsetsRead, .updateAndAttend,
                ], "layer \(idx)")
        }

        // Shared layers: borrow-only, from the right source, and they never
        // read positionOffsets — they reuse the SOURCE layer's captured
        // (pre-update) snapshot threaded by the trunk (invariant 1).
        for (idx, expectedSource) in [(2, 0), (3, 1)] {
            let mock = mocks[idx]
            #expect(mock.updateCalls.isEmpty, "layer \(idx)")
            #expect(mock.borrowCalls.count == 4, "layer \(idx)")
            #expect(
                mock.borrowCalls.allSatisfy { $0.sourceLayerIndex == expectedSource },
                "layer \(idx)")
            let headDim = idx == 2 ? 8 : 16
            #expect(mock.borrowCalls[0].queryShape == [1, 2, 5, headDim], "layer \(idx)")
            #expect(mock.borrowCalls[1].queryShape == [1, 2, 1, headDim], "layer \(idx)")
            #expect(mock.positionOffsetsReads == 0, "layer \(idx)")
        }

        // The model must never build masks on the v2 path.
        #expect(mocks.allSatisfy { $0.makeMaskCalls == 0 })
    }

    @Test func multiRowDecodeUsesPerRowOffsets() throws {
        // Two rows at different absolute positions {5, 2} — the exact state
        // a mid-stream join produces. Per-row offsets must flow (invariant 1:
        // a row's positions cannot depend on batchmates).
        let (model, mocks) = try makeModelAndCaches(rowCount: 2)

        for mock in mocks where mock.kind.sharesKVWithLayer == nil {
            let headDim = mock.kind.headDim
            let kvHeads = mock.kind.kvHeads
            for (row, tokens) in zip(mock.mockRows, [5, 2]) {
                _ = row.update(
                    keys: MLXArray.zeros([1, kvHeads, tokens, headDim]),
                    values: MLXArray.zeros([1, kvHeads, tokens, headDim]))
            }
        }

        let tokens = MLXArray([Int32(7), Int32(9)]).reshaped(2, 1)
        let logits = model(tokens, cache: mocks as [KVCache])
        eval(logits)
        #expect(logits.shape == [2, 1, 64])

        for idx in [0, 1] {
            let mock = mocks[idx]
            #expect(mock.updateCalls.count == 1, "layer \(idx)")
            #expect(mock.updateCalls[0].offsetsBefore == [5, 2], "layer \(idx)")
            #expect(mock.mockRows.map { $0.absoluteOffset } == [6, 3], "layer \(idx)")
            let headDim = idx == 0 ? 8 : 16
            #expect(mock.updateCalls[0].queryShape == [2, 2, 1, headDim], "layer \(idx)")
            #expect(mock.updateCalls[0].keyShape == [2, 1, 1, headDim], "layer \(idx)")
        }
        for idx in [2, 3] {
            #expect(mocks[idx].borrowCalls.count == 1, "layer \(idx)")
            #expect(mocks[idx].positionOffsetsReads == 0, "layer \(idx)")
        }
        #expect(mocks.allSatisfy { $0.makeMaskCalls == 0 })
    }

    @Test func legacyCachePathUnchanged() throws {
        // Regression: the legacy newCache() path still works and produces
        // caches only for the non-shared prefix (2 of 4 layers here).
        let config = try tinyConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let caches = model.newCache(parameters: nil)
        #expect(caches.count == 2)
        let prompt = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
        let logits = model(prompt, cache: caches)
        eval(logits)
        #expect(logits.shape == [1, 3, 64])
    }

    /// Multimodal plumbing (Gemma4-specific): splicing the model's OWN scaled
    /// embeddings back in must reproduce the token forward bit-for-bit — this
    /// exercises the real embedScale + per-layer-input (token-PLE) handling
    /// on the CBv2 embedding-forward path, which TinyTestModel (no scale, no
    /// PLE) cannot cover. `use_bidirectional_attention` off ⇒ no overlay.
    @Test func embeddingForwardMatchesTokenForwardOnV2Branch() throws {
        let config = try tinyConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        #expect(model is CBv2EmbeddingForwardable)

        func freshCaches() -> [CBv2ModelMockLayerCache] {
            var mocks = [CBv2ModelMockLayerCache]()
            _ = model.newCacheV2 { index, kind in
                let mock = CBv2ModelMockLayerCache(layerIndex: index, kind: kind, rowCount: 1)
                mocks.append(mock)
                return mock
            }
            return mocks
        }

        let prompt = MLXArray([Int32(1), 2, 3, 4, 5])[.newAxis, .ellipsis]

        let tokenLogits = model(prompt, cache: freshCaches() as [KVCache])
        let spliced = (model as CBv2EmbeddingForwardable).scaledInputEmbeddings(prompt)
        let embeddingLogits = (model as CBv2EmbeddingForwardable).embeddingForward(
            prompt, inputEmbedding: spliced, cache: freshCaches() as [KVCache])
        eval(tokenLogits, embeddingLogits)

        #expect(tokenLogits.shape == embeddingLogits.shape)
        #expect(
            allClose(tokenLogits, embeddingLogits, atol: 1e-5).item(Bool.self),
            "splicing the model's own embeddings must reproduce the token forward")
    }

    /// PR#63 review: structural `CBv2EmbeddingForwardable` conformance must
    /// NOT grant multimodal capability by itself — a Gemma4TextModel from a
    /// TEXT-ONLY config (`use_bidirectional_attention` nil/non-"vision") was
    /// never trained for the bidirectional span masks CBv2 applies, and the
    /// adapter must reject vision requests against it at submit.
    @Test func adapterGatesMultimodalOnVisionConfig() throws {
        let textModel = Gemma4TextModel(try tinyConfig())
        #expect((textModel as CBv2EmbeddingForwardable).supportsVisionSpanPrefill == false)
        let textAdapter = CBv2SteppableLanguageModelAdapter(textModel)
        #expect(
            textAdapter.supportsMultimodalPrefill == false,
            "text-only Gemma4 config must not advertise multimodal prefill")

        var visionConfig = try tinyConfig()
        visionConfig.useBidirectionalAttention = "vision"
        let visionModel = Gemma4TextModel(visionConfig)
        #expect((visionModel as CBv2EmbeddingForwardable).supportsVisionSpanPrefill)
        let visionAdapter = CBv2SteppableLanguageModelAdapter(visionModel)
        #expect(visionAdapter.supportsMultimodalPrefill)
    }

    /// PR#63 review: the legacy `imageTokenMask` overlay with a NON-EMPTY KV
    /// cache — `baseMask` covers `offset + L` key columns while the
    /// same-block overlay describes only the current window's L columns. The
    /// overlay must land on the LAST L key columns (cached keys stay
    /// causal); pre-fix this either failed to broadcast or misaligned the
    /// bidirectional block. Chunked cached prefill must match the
    /// whole-prompt forward exactly.
    @Test func legacyImageTokenMaskAlignsWithCachedKeys() throws {
        var config = try tinyConfig()
        config.useBidirectionalAttention = "vision"
        let model = Gemma4TextModel(config)
        eval(model)

        let T = 10
        let split = 5
        let tokens = MLXArray((0 ..< T).map { Int32($0 % 8 + 1) }).reshaped(1, T)
        // Image block at absolute positions [6, 9).
        let fullVision = MLXArray((0 ..< T).map { $0 >= 6 && $0 < 9 }).reshaped(1, T)

        // Whole-prompt reference (offset 0 — the previously working shape).
        let reference = model(
            tokens, inputEmbedding: nil, cache: model.newCache(parameters: nil),
            imageTokenMask: fullVision)

        // Chunked: text chunk [0,5) fills the cache, then the vision chunk
        // [5,10) runs with offset 5 — baseMask [5, 10] vs overlay [1, 5, 5].
        let caches = model.newCache(parameters: nil)
        _ = model(tokens[0..., 0 ..< split], cache: caches)
        let chunkVision = MLXArray((split ..< T).map { $0 >= 6 && $0 < 9 }).reshaped(
            1, T - split)
        let chunked = model(
            tokens[0..., split ..< T], inputEmbedding: nil, cache: caches,
            imageTokenMask: chunkVision)

        eval(reference, chunked)
        #expect(chunked.shape == [1, T - split, 64])
        #expect(
            allClose(
                chunked[0..., -1, 0...], reference[0..., -1, 0...], atol: 1e-5
            ).item(Bool.self),
            "cached vision overlay must align with the last L key columns")
    }
}

// MARK: - GPT-OSS forward smoke tests (v2 branch)

@Suite("CBv2 GPT-OSS v2 attention branch")
struct CBv2ModelGPTOSSForwardTests {

    /// 4-layer tiny config: alternating [sliding, full], 4 experts top-2.
    private func tinyConfig() throws -> GPTOSSConfiguration {
        let json = """
            {
                "model_type": "gpt_oss",
                "num_hidden_layers": 4,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "vocab_size": 64,
                "rms_norm_eps": 1e-5,
                "hidden_size": 32,
                "intermediate_size": 32,
                "head_dim": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "sliding_window": 8
            }
            """
        return try JSONDecoder.json5().decode(
            GPTOSSConfiguration.self, from: Data(json.utf8))
    }

    private func makeModelAndCaches(
        sinksValue: Float? = nil, rowCount: Int = 1
    ) throws -> (GPTOSSModel, [CBv2ModelMockLayerCache]) {
        let model = GPTOSSModel(try tinyConfig())
        eval(model)
        if let sinksValue {
            for layer in model.loraLayers {
                guard let block = layer as? GPTOSSTransformerBlock else {
                    fatalError("unexpected layer type")
                }
                block.selfAttn.update(
                    parameters: ModuleParameters.unflattened([
                        "sinks": MLXArray(Array(repeating: sinksValue, count: 4))
                    ]))
            }
        }
        var mocks = [CBv2ModelMockLayerCache]()
        let caches = model.newCacheV2 { index, kind in
            let mock = CBv2ModelMockLayerCache(layerIndex: index, kind: kind, rowCount: rowCount)
            mocks.append(mock)
            return mock
        }
        #expect(caches.count == 4)
        return (model, mocks)
    }

    @Test func factoryReceivesDerivedKinds() throws {
        let config = try tinyConfig()
        let (_, mocks) = try makeModelAndCaches()
        #expect(mocks.map { $0.layerIndex } == [0, 1, 2, 3])
        #expect(mocks.map { $0.kind } == config.cbv2LayerKinds)
        #expect(
            mocks.map { $0.kind.attention } == [
                .slidingWindow(8), .full, .slidingWindow(8), .full,
            ])
        #expect(mocks.allSatisfy { $0.kind.hasSinks })
    }

    @Test func prefillAndDecodeThroughV2Branch() throws {
        let (model, mocks) = try makeModelAndCaches()
        let caches: [KVCache] = mocks

        let prompt = MLXArray([Int32(1), 2, 3, 4, 5])[.newAxis, .ellipsis]
        var logits = model(prompt, cache: caches)
        eval(logits)
        #expect(logits.shape == [1, 5, 64])

        for step in 0 ..< 3 {
            let token = MLXArray([Int32(7 + step)])[.newAxis, .ellipsis]
            logits = model(token, cache: caches)
            eval(logits)
            #expect(logits.shape == [1, 1, 64])
        }

        for (idx, mock) in mocks.enumerated() {
            #expect(mock.updateCalls.count == 4, "layer \(idx)")
            #expect(mock.borrowCalls.isEmpty, "layer \(idx)")
            #expect(
                mock.updateCalls.map { $0.offsetsBefore } == [[0], [5], [6], [7]],
                "layer \(idx)")
            #expect(mock.updateCalls[0].queryShape == [1, 4, 5, 8], "layer \(idx)")
            #expect(mock.updateCalls[0].keyShape == [1, 2, 5, 8], "layer \(idx)")
            #expect(mock.updateCalls[1].queryShape == [1, 4, 1, 8], "layer \(idx)")
            #expect(mock.mockRows[0].absoluteOffset == 8, "layer \(idx)")
            // Zero sinks (unloaded) ⇒ the model must pass nil, never an
            // all-zero sinks array (zero sinks are NOT a no-op in softmax).
            #expect(mock.updateCalls.allSatisfy { !$0.sinksProvided }, "layer \(idx)")
            #expect(mock.positionOffsetsReads == 4, "layer \(idx)")
            #expect(
                mock.events == [
                    .offsetsRead, .updateAndAttend,
                    .offsetsRead, .updateAndAttend,
                    .offsetsRead, .updateAndAttend,
                    .offsetsRead, .updateAndAttend,
                ], "layer \(idx)")
        }
        #expect(mocks.allSatisfy { $0.makeMaskCalls == 0 })
    }

    @Test func activeSinksArePassedThrough() throws {
        // Non-zero learned sinks ⇒ every updateAndAttend receives them.
        // newCacheV2 primes the activation Bool at load, so the v2 step path
        // never runs the `.item()` probe.
        let (model, mocks) = try makeModelAndCaches(sinksValue: 0.5)

        for layer in model.loraLayers {
            let block = layer as! GPTOSSTransformerBlock
            #expect(block.selfAttn.sinksActiveResolved() == true)
        }

        let caches: [KVCache] = mocks
        let prompt = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
        var logits = model(prompt, cache: caches)
        eval(logits)
        let token = MLXArray([Int32(7)])[.newAxis, .ellipsis]
        logits = model(token, cache: caches)
        eval(logits)
        #expect(logits.shape == [1, 1, 64])

        for (idx, mock) in mocks.enumerated() {
            #expect(mock.updateCalls.count == 2, "layer \(idx)")
            #expect(mock.updateCalls.allSatisfy { $0.sinksProvided }, "layer \(idx)")
        }
    }

    @Test func zeroSinksArePrimedInactiveAtLoad() throws {
        let (model, _) = try makeModelAndCaches()
        // newCacheV2 already ran: the Bool is primed (false for zero sinks)
        // without any further forward pass.
        for layer in model.loraLayers {
            let block = layer as! GPTOSSTransformerBlock
            #expect(block.selfAttn.sinksActiveResolved() == false)
        }
    }

    @Test func multiRowDecodeUsesPerRowOffsets() throws {
        let (model, mocks) = try makeModelAndCaches(rowCount: 2)

        for mock in mocks {
            for (row, tokens) in zip(mock.mockRows, [6, 2]) {
                _ = row.update(
                    keys: MLXArray.zeros([1, 2, tokens, 8]),
                    values: MLXArray.zeros([1, 2, tokens, 8]))
            }
        }

        let tokens = MLXArray([Int32(7), Int32(9)]).reshaped(2, 1)
        let logits = model(tokens, cache: mocks as [KVCache])
        eval(logits)
        #expect(logits.shape == [2, 1, 64])

        for (idx, mock) in mocks.enumerated() {
            #expect(mock.updateCalls.count == 1, "layer \(idx)")
            #expect(mock.updateCalls[0].offsetsBefore == [6, 2], "layer \(idx)")
            #expect(mock.mockRows.map { $0.absoluteOffset } == [7, 3], "layer \(idx)")
            #expect(mock.updateCalls[0].queryShape == [2, 4, 1, 8], "layer \(idx)")
        }
        #expect(mocks.allSatisfy { $0.makeMaskCalls == 0 })
    }

    @Test func legacyCachePathUnchanged() throws {
        // Regression: legacy newCache() still yields one cache per layer
        // (Standard for full, Rotating for sliding) and the forward works.
        let model = GPTOSSModel(try tinyConfig())
        eval(model)
        let caches = model.newCache(parameters: nil)
        #expect(caches.count == 4)
        #expect(caches[0] is RotatingKVCache)
        #expect(caches[1] is StandardKVCache)
        let prompt = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
        let logits = model(prompt, cache: caches)
        eval(logits)
        #expect(logits.shape == [1, 3, 64])
    }
}
