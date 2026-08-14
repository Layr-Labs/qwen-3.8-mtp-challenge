import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("LagunaModel")
struct LagunaModelTests {

    static let tinyConfigJSON = """
        {
          "model_type": "laguna",
          "vocab_size": 32,
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 4,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "max_position_embeddings": 4096,
          "rms_norm_eps": 1e-6,
          "attention_bias": false,
          "qkv_bias": false,
          "tie_word_embeddings": false,
          "rope_theta": 500000.0,
          "sliding_window": 8,
          "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
          "gating": "per-head",
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 8,
          "shared_expert_intermediate_size": 8,
          "moe_routed_scaling_factor": 1.0,
          "norm_topk_prob": true,
          "decoder_sparse_step": 1,
          "moe_router_logit_softcapping": 0.0
        }
        """

    private func tinyConfig() throws -> LagunaConfiguration {
        try JSONDecoder.json5().decode(
            LagunaConfiguration.self, from: Data(Self.tinyConfigJSON.utf8))
    }

    /// Same shape as `tinyConfigJSON`, but with dimensions that are
    /// multiples of 32 -- MLX's `quantize` op only supports group sizes of
    /// 32, 64, or 128, so `tinyConfigJSON`'s hidden_size: 8 / moe_intermediate:
    /// 8 cannot be quantized at all. This config exists solely to exercise
    /// the loader-style quantize pass below.
    static let tinyQuantizableConfigJSON = """
        {
          "model_type": "laguna",
          "vocab_size": 32,
          "hidden_size": 32,
          "intermediate_size": 64,
          "num_hidden_layers": 4,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 16,
          "max_position_embeddings": 4096,
          "rms_norm_eps": 1e-6,
          "attention_bias": false,
          "qkv_bias": false,
          "tie_word_embeddings": false,
          "rope_theta": 500000.0,
          "sliding_window": 8,
          "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
          "gating": "per-head",
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 32,
          "shared_expert_intermediate_size": 32,
          "moe_routed_scaling_factor": 1.0,
          "norm_topk_prob": true,
          "decoder_sparse_step": 1,
          "moe_router_logit_softcapping": 0.0
        }
        """

    private func tinyQuantizableConfig() throws -> LagunaConfiguration {
        try JSONDecoder.json5().decode(
            LagunaConfiguration.self, from: Data(Self.tinyQuantizableConfigJSON.utf8))
    }

    @Test func decodesTinyConfig() throws {
        let config = try tinyConfig()
        #expect(config.numHiddenLayers == 4)
        #expect(config.gatingEnabled)
        #expect(config.gatePerHead)
        // mlp_only_layers defaults to [0]: layer 0 dense, the rest sparse.
        #expect(!config.isSparse(layer: 0))
        #expect(config.isSparse(layer: 1))
    }

    @Test func forwardProducesLogitsAndFillsCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = LagunaModel(try tinyConfig())
            eval(model)
            let cache = model.newCache(parameters: nil)
            #expect(cache.count == 4)
            #expect(cache[0] is RotatingKVCache)
            #expect(cache[1] is KVCacheSimple)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let logits = model(tokens, cache: cache)
            eval(logits)
            #expect(logits.shape == [1, 3, 32])
            #expect(cache[0].offset == 3)
        }
    }

    @Test func factoryRegistersLaguna() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(Self.tinyConfigJSON.utf8), modelType: "laguna")
        #expect(model is LagunaModel)
    }

    @Test func sanitizeDropsRotaryTables() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = LagunaModel(try tinyConfig())
            let cleaned = model.sanitize(weights: [
                "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2])
            ])
            #expect(cleaned.isEmpty)
        }
    }

    /// Regression test for the loader-side whole-model quantize pass
    /// (mirrors `loadWeights` in Libraries/MLXLMCommon/Load.swift:40-52).
    ///
    /// That pass calls `quantize(model:filter:)`, which replaces every
    /// `SwitchLinear`/`Linear` submodule matched by the filter via
    /// `Module.update(modules:)`. That replacement mechanism requires the
    /// module hierarchy from the model root down to each replaced leaf to be
    /// reachable through mutable (`@ModuleInfo`) module-holding properties, or
    /// through plain array/module properties the update path can recurse
    /// into and rebuild. A checkpoint the size of the real 20 GB Laguna
    /// target always ships a `quantization` block, so this path always
    /// fires -- but the tiny test config here has no `quantization` block,
    /// so `loadWeights` itself never exercises the crash. This test drives
    /// the same `quantize(model:filter:)` API directly, without touching
    /// `loadWeights` or requiring any real weights, so it fails purely on
    /// module structure.
    @Test func loaderStyleQuantizePassSucceeds() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = LagunaModel(try tinyQuantizableConfig())
            // Mirror Load.swift's whole-model pass: quantize every
            // switch_mlp/shared_expert projection (the router/gate weight
            // stays BF16 and is intentionally left unmatched, matching the
            // real Laguna NVFP4 checkpoint's quantization scope).
            quantize(model: model) { path, module in
                if path.contains("switch_mlp") || path.contains("shared_expert"),
                    module is SwitchLinear || module is Linear
                {
                    // Quantizable config dims are all multiples of 32
                    // (hidden 32, intermediate 64, moe_intermediate 32,
                    // shared_expert_intermediate 32); MLX's quantize op only
                    // supports group sizes of 32, 64, or 128.
                    return (groupSize: 32, bits: 4, mode: .affine)
                }
                return nil
            }
            eval(model)

            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let logits = model(tokens, cache: model.newCache(parameters: nil))
            eval(logits)
            #expect(logits.shape == [1, 3, 32])
        }
    }
}
