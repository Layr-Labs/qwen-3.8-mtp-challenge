// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXSpeculative

@Suite("Laguna DFlash draft model")
struct LagunaDFlashDraftModelTests {

    private func lagunaConfigJSON(decoderLayerType: String = "laguna_xs") -> String {
        """
        {
          "architectures": ["DFlashDraftModel"],
          "model_type": "laguna",
          "decoder_layer_type": "\(decoderLayerType)",
          "gating": "per-head",
          "hidden_size": 8,
          "num_hidden_layers": 2,
          "intermediate_size": 16,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "vocab_size": 32,
          "rms_norm_eps": 1e-6,
          "rope_theta": 500000.0,
          "max_position_embeddings": 4096,
          "block_size": 4,
          "num_target_layers": 4,
          "sliding_window": 8,
          "layer_types": ["sliding_attention", "sliding_attention"],
          "dflash_config": {"target_layer_ids": [0, 2], "mask_token_id": 3}
        }
        """
    }

    private func makeConfig(decoderLayerType: String = "laguna_xs") throws -> DFlashConfiguration {
        try JSONDecoder.json5().decode(
            DFlashConfiguration.self,
            from: Data(lagunaConfigJSON(decoderLayerType: decoderLayerType).utf8))
    }

    @Test func lagunaXSBuildsGatingAndAuxNormParameters() throws {
        try Device.withDefaultDevice(.cpu) {
            let draft = DFlashDraftModel(config: try makeConfig())
            let params = Dictionary(
                uniqueKeysWithValues: draft.parameters().flattened())
            #expect(params["aux_hidden_norms.0.weight"]?.shape == [8])
            #expect(params["aux_hidden_norms.1.weight"]?.shape == [8])
            #expect(params["layers.0.self_attn.g_proj.weight"]?.shape == [2, 8])
            #expect(params["layers.1.self_attn.g_proj.weight"]?.shape == [2, 8])
        }
    }

    @Test func qwen3ConfigBuildsNoLagunaParameters() throws {
        try Device.withDefaultDevice(.cpu) {
            let draft = DFlashDraftModel(config: try makeConfig(decoderLayerType: "qwen3"))
            let keys = Set(draft.parameters().flattened().map(\.0))
            #expect(!keys.contains { $0.hasPrefix("aux_hidden_norms") })
            #expect(!keys.contains { $0.contains("g_proj") })
        }
    }

    @Test func lagunaXSForwardDiffersFromQwen3WithSharedWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(7)
            let laguna = DFlashDraftModel(config: try makeConfig())
            MLXRandom.seed(7)
            let qwen = DFlashDraftModel(config: try makeConfig(decoderLayerType: "qwen3"))
            // Copy the shared subset of weights laguna->qwen so the only
            // differences are the laguna_xs behaviors themselves.
            let lagunaParams = Dictionary(
                uniqueKeysWithValues: laguna.parameters().flattened())
            let qwenKeys = Set(qwen.parameters().flattened().map(\.0))
            let shared = lagunaParams.filter { qwenKeys.contains($0.key) }
            try qwen.update(
                parameters: ModuleParameters.unflattened(shared), verify: [.noUnusedKeys])

            let target = LagunaDraftStubTarget(hiddenSize: 8, vocabularySize: 32, layerCount: 4)
            try laguna.bind(target: target)
            try qwen.bind(target: target)
            eval(laguna, qwen)

            let hidden = MLXRandom.normal([1, 3, 16]).asType(.bfloat16)
            let block = MLXArray([Int32(5), 3, 3, 3])[.newAxis, .ellipsis]
            let a = try laguna(
                block, targetHidden: hidden, cache: try laguna.makeCache(), logitsStart: 1)
            let b = try qwen(
                block, targetHidden: hidden, cache: try qwen.makeCache(), logitsStart: 1)
            eval(a, b)
            #expect(!allClose(a, b, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        }
    }

    @Test func lagunaXSForwardMatchesReferenceImplementation() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(11)
            let config = try makeConfig()
            let draft = DFlashDraftModel(config: config)
            let target = LagunaDraftStubTarget(hiddenSize: 8, vocabularySize: 32, layerCount: 4)
            try draft.bind(target: target)
            eval(draft)

            // MLXNN's RMSNorm initializes weights to all-ones. On freshly
            // constructed parameters that leaves two blind spots: (a) after
            // `hidden_norm` (ones) `ctx` is already unit-RMS, so each layer's
            // `input_layernorm(context)` is numerically ~identity and the test
            // cannot detect the per-layer context norm being removed or
            // miswired; (b) every `aux_hidden_norms` weight is an identical
            // all-ones vector, so the test cannot detect the two aux norms
            // being permuted or norm[0] being applied to every slice.
            // Overwrite every RMSNorm weight with distinct, non-unit, seeded
            // values before extracting parameters for the reference below, so
            // both bugs would actually move the numbers.
            MLXRandom.seed(31)
            var normOverrides = [String: MLXArray]()
            for (key, value) in draft.parameters().flattened()
            where key.hasSuffix("norm.weight") || key.contains("aux_hidden_norms") {
                normOverrides[key] = MLXRandom.uniform(
                    low: 0.5, high: 1.5, [value.dim(0)]
                ).asType(value.dtype)
            }
            try draft.update(parameters: ModuleParameters.unflattened(normOverrides), verify: [])
            eval(draft)

            let p = Dictionary(uniqueKeysWithValues: draft.parameters().flattened())
            func w(_ key: String) -> MLXArray { p[key]!.asType(.float32) }
            let eps: Float = 1e-6
            func rms(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
                MLXFast.rmsNorm(x, weight: weight, eps: eps)
            }

            let contextLength = 3
            MLXRandom.seed(21)
            let targetHidden = MLXRandom.normal([1, contextLength, 16]).asType(.float32)
            let block: [Int32] = [5, 3, 3, 3]
            let blockArray = MLXArray(block)[.newAxis, .ellipsis]

            // --- Reference forward (float32, mirrors vLLM DFlashLagunaForCausalLM) ---
            let slices = (0 ..< 2).map { j in
                rms(targetHidden[.ellipsis, (j * 8) ..< ((j + 1) * 8)],
                    w("aux_hidden_norms.\(j).weight"))
            }
            var ctx = concatenated(slices, axis: -1)
            ctx = matmul(ctx, w("fc.weight").T)
            ctx = rms(ctx, w("hidden_norm.weight"))

            var h = target.embedTokensForDFlash(blockArray).asType(.float32)
            let rope = initializeRope(
                dims: 4, base: 500000.0, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: 4096)

            for layer in 0 ..< 2 {
                let prefix = "layers.\(layer)"
                let xin = rms(h, w("\(prefix).input_layernorm.weight"))
                let ctxin = rms(ctx, w("\(prefix).input_layernorm.weight"))

                func heads(_ x: MLXArray, _ n: Int) -> MLXArray {
                    x.reshaped(1, x.dim(1), n, 4).transposed(0, 2, 1, 3)
                }
                var q = heads(matmul(xin, w("\(prefix).self_attn.q_proj.weight").T), 2)
                q = rms(q, w("\(prefix).self_attn.q_norm.weight"))
                var pk = heads(matmul(xin, w("\(prefix).self_attn.k_proj.weight").T), 1)
                pk = rms(pk, w("\(prefix).self_attn.k_norm.weight"))
                let pv = heads(matmul(xin, w("\(prefix).self_attn.v_proj.weight").T), 1)
                var ck = heads(matmul(ctxin, w("\(prefix).self_attn.k_proj.weight").T), 1)
                ck = rms(ck, w("\(prefix).self_attn.k_norm.weight"))
                let cv = heads(matmul(ctxin, w("\(prefix).self_attn.v_proj.weight").T), 1)

                q = rope(q, offset: contextLength)
                pk = rope(pk, offset: contextLength)
                ck = rope(ck, offset: 0)

                let keys = concatenated([ck, pk], axis: 2)
                let values = concatenated([cv, pv], axis: 2)
                let mask = createCausalMask(
                    n: 4, offset: contextLength, windowSize: 8)
                let attn = MLXFast.scaledDotProductAttention(
                    queries: q, keys: keys, values: values,
                    scale: pow(4.0, -0.5), mask: .array(mask))
                var out = attn.transposed(0, 2, 1, 3).reshaped(1, 4, 8)

                let gate = softplus(
                    matmul(xin, w("\(prefix).self_attn.g_proj.weight").T))
                out = (out.reshaped(1, 4, 2, 4) * gate[.ellipsis, .newAxis])
                    .reshaped(1, 4, 8)
                let attnOut = matmul(out, w("\(prefix).self_attn.o_proj.weight").T)
                h = h + attnOut

                let hin = rms(h, w("\(prefix).post_attention_layernorm.weight"))
                let mlpOut = matmul(
                    silu(matmul(hin, w("\(prefix).mlp.gate_proj.weight").T))
                        * matmul(hin, w("\(prefix).mlp.up_proj.weight").T),
                    w("\(prefix).mlp.down_proj.weight").T)
                h = h + mlpOut
            }
            let expected = target.logitsForDFlashHidden(
                rms(h, w("norm.weight")).asType(.bfloat16))[0..., 1..., 0...]

            // --- Model forward ---
            let actual = try draft(
                blockArray,
                targetHidden: targetHidden.asType(.bfloat16),
                cache: try draft.makeCache(),
                logitsStart: 1)
            eval(expected, actual)
            #expect(
                allClose(actual.asType(.float32), expected.asType(.float32),
                    rtol: 2e-2, atol: 2e-2).item(Bool.self),
                "laguna_xs draft forward diverged from reference")
        }
    }

    @Test func draftBlockRunsAgainstTinyLagunaTarget() throws {
        try Device.withDefaultDevice(.cpu) {
            // Draft with num_target_layers = 4 matching the tiny target's 4 layers.
            let draft = DFlashDraftModel(config: try makeConfig())
            let target = LagunaModel(
                try JSONDecoder.json5().decode(
                    LagunaConfiguration.self,
                    from: Data(LagunaModelTests.tinyConfigJSON.utf8)))
            try draft.bind(target: target)
            eval(draft, target)

            let prompt = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let targetOut = try target.forwardForDFlash(
                prompt,
                cache: target.newCache(parameters: nil),
                targetLayerIds: draft.config.targetLayerIds)
            let drafted = try draft.draftBlock(
                bonus: 4,
                targetHidden: targetOut.targetHidden,
                cache: try draft.makeCache(),
                blockSize: draft.config.recommendedBlockSize)
            eval(drafted)
            #expect(drafted.dim(-1) == draft.config.recommendedBlockSize - 1)
            let tokens = drafted.asArray(Int32.self)
            #expect(tokens.allSatisfy { $0 >= 0 && $0 < 32 })
        }
    }
}

/// Minimal deterministic target: embedding table + linear head, seeded.
final class LagunaDraftStubTarget: Module, DFlashTargetModel {
    let vocabularySize: Int
    let kvHeads: [Int] = []
    let dFlashVocabularySize: Int
    let dFlashHiddenSize: Int
    let dFlashLayerCount: Int
    let embedWeight: MLXArray
    let headWeight: MLXArray
    var loraLayers: [Module] { [] }

    init(hiddenSize: Int, vocabularySize: Int, layerCount: Int) {
        self.vocabularySize = vocabularySize
        self.dFlashVocabularySize = vocabularySize
        self.dFlashHiddenSize = hiddenSize
        self.dFlashLayerCount = layerCount
        MLXRandom.seed(99)
        self.embedWeight = MLXRandom.normal([vocabularySize, hiddenSize]).asType(.bfloat16)
        self.headWeight = MLXRandom.normal([vocabularySize, hiddenSize]).asType(.bfloat16)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logitsForDFlashHidden(embedTokensForDFlash(inputs))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

    func forwardForDFlash(
        _ inputs: MLXArray, cache: [KVCache]?, targetLayerIds: [Int]
    ) throws -> DFlashTargetForward {
        let h = embedTokensForDFlash(inputs)
        return DFlashTargetForward(
            logits: logitsForDFlashHidden(h),
            hiddenStates: targetLayerIds.map { _ in h })
    }

    func embedTokensForDFlash(_ tokens: MLXArray) -> MLXArray {
        embedWeight[tokens]
    }

    func logitsForDFlashHidden(_ hidden: MLXArray) -> MLXArray {
        matmul(hidden, headWeight.T)
    }
}

/// Structural load check against the real converted Laguna DFlash artifact.
/// Env-gated: set `MLX_SWIFT_LM_LAGUNA_DFLASH_DIR` to the converted artifact
/// directory (config.json + model.safetensors) to enable.
@Suite("Laguna DFlash converted artifact", .serialized)
struct LagunaDFlashConvertedArtifactTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment[
        "MLX_SWIFT_LM_LAGUNA_DFLASH_DIR"] != nil))
    func convertedArtifactLoadsWithFullVerification() async throws {
        let dir = URL(fileURLWithPath: ProcessInfo.processInfo.environment[
            "MLX_SWIFT_LM_LAGUNA_DFLASH_DIR"]!)
        let draft = try await DFlashDraftModel.load(from: dir)
        #expect(draft.config.decoderLayerType == .lagunaXS)
        #expect(draft.config.blockSize == 16)
        #expect(draft.config.targetLayerIds == [1, 13, 25, 33, 39])
        let params = Dictionary(
            uniqueKeysWithValues: draft.parameters().flattened())
        #expect(params.count == 68)
        #expect(params["layers.4.self_attn.g_proj.weight"]?.shape == [64, 2048])
        #expect(params["aux_hidden_norms.4.weight"]?.shape == [2048])
        #expect(params["fc.weight"]?.shape == [2048, 10240])
        #expect(params["layers.0.self_attn.q_proj.weight"]?.dtype == .bfloat16)
    }
}
