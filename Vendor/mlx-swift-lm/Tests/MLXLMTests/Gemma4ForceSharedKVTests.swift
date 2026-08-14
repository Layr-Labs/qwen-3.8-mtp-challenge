// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Gemma4 shared-KV layer selection")
struct Gemma4ForceSharedKVTests {

    /// Build a 4-layer Gemma4TextConfiguration with every layer shared —
    /// the drafter shape.
    private func drafterLikeConfig() throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "intermediate_size": 2048,
            "num_attention_heads": 4,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_key_value_heads": 2,
            "num_kv_shared_layers": 4,
            "layer_types": ["sliding_attention", "sliding_attention",
                            "sliding_attention", "full_attention"],
            "sliding_window": 512,
            "final_logit_softcapping": null,
            "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": false,
            "tie_word_embeddings": true,
            "vocab_size": 1024,
            "vocab_size_per_layer_input": 1024,
            "rms_norm_eps": 1e-6
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: data)
    }

    /// Build a 35-layer Gemma4TextConfiguration with the last 20 shared —
    /// the 4B target shape.
    private func targetLikeConfig() throws -> Gemma4TextConfiguration {
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
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 1024,
            "vocab_size_per_layer_input": 1024,
            "rms_norm_eps": 1e-6
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: data)
    }

    @Test func drafterConfigMarksAllLayersShared() throws {
        let config = try drafterLikeConfig()
        // For a drafter: numKvSharedLayers == numHiddenLayers, firstShared == 0.
        // Every layer must be marked usesSharedKV == true.
        for i in 0 ..< config.numHiddenLayers {
            #expect(config.layerUsesSharedKV(layerIdx: i))
        }
    }

    @Test func targetConfigKeepsExistingSplit() throws {
        let config = try targetLikeConfig()
        // First 15 layers: not shared.
        for i in 0 ..< 15 {
            #expect(!config.layerUsesSharedKV(layerIdx: i))
        }
        // Last 20 layers: shared.
        for i in 15 ..< 35 {
            #expect(config.layerUsesSharedKV(layerIdx: i))
        }
    }

    @Test func forceSharedKVOverridesIndex() throws {
        let config = try targetLikeConfig()
        // Even layer 0 should be treated as shared when forceSharedKV is true.
        #expect(config.layerUsesSharedKV(layerIdx: 0, forceSharedKV: true))
    }
}
