// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

/// Regression tests for LFM2-MoE sigmoid routing (upstream e3cb1e1).
///
/// NB-3 fast-follow: the routing rewrite (softmax -> sigmoid, bias-selection-only,
/// `routed_scaling_factor`, eps 1e-20 -> 1e-6) shipped with no dedicated test.
/// These lock the three behaviours that the rewrite changed:
///   1. routing weights come from a *sigmoid* of the gate logits,
///   2. `expert_bias` steers top-k SELECTION only (not the weights),
///   3. the (normalized) weights are scaled by `routed_scaling_factor`.
final class LFM2MoERoutingTests: XCTestCase {

    /// 4-expert config; `hidden_size == num_experts` so a gate weight of the
    /// identity makes `gate(x) == x` and the input row *is* the logits.
    private func makeConfig(useExpertBias: Bool, scaling: Float) throws -> LFM2MoEConfiguration {
        let json = """
            {
                "model_type": "lfm2_moe",
                "vocab_size": 32,
                "hidden_size": 4,
                "intermediate_size": 8,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "norm_topk_prob": true,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "use_expert_bias": \(useExpertBias),
                "num_dense_layers": 0,
                "norm_eps": 1e-6,
                "conv_bias": false,
                "conv_L_cache": 3,
                "routed_scaling_factor": \(scaling)
            }
            """.data(using: .utf8)!
        return try JSONDecoder().decode(LFM2MoEConfiguration.self, from: json)
    }

    private func eye(_ n: Int) -> MLXArray {
        var data = [Float](repeating: 0, count: n * n)
        for i in 0 ..< n { data[i * n + i] = 1 }
        return MLXArray(data, [n, n])
    }

    private func sigmoidD(_ v: Double) -> Double { 1.0 / (1.0 + exp(-v)) }

    func testSigmoidRoutingBiasSelectsButWeightsStayUnbiased() throws {
        let scaling: Float = 2.5
        let config = try makeConfig(useExpertBias: true, scaling: scaling)
        let block = Lfm2MoeSparseMoeBlock(config)

        // gate(x) = x (identity). Bias lifts expert 2 (low logit) into the top-2
        // over expert 1, without changing the combination weights.
        block.update(
            parameters: ModuleParameters.unflattened([
                "gate.weight": eye(4),
                "expert_bias": MLXArray([Float(0), 0, 1, 0]),
            ]))

        let logits: [Float] = [2.0, 1.0, 0.5, 0.0]
        let x = MLXArray(logits, [1, 4])
        let (indices, weights) = block.route(x)
        eval(indices, weights)

        let idx = indices.asArray(Int32.self).map { Int($0) }
        let w = weights.asArray(Float.self)

        XCTAssertEqual(idx.count, 2)
        XCTAssertEqual(
            Set(idx), Set([0, 2]),
            "expert_bias must steer top-k SELECTION: expert 2 (low logit, high bias) "
                + "replaces expert 1; it must not do so by changing the weights.")

        // Weights = UNbiased sigmoid at the selected experts, normalized
        // (eps 1e-6) and scaled by routed_scaling_factor.
        let s0 = sigmoidD(2.0)
        let s2 = sigmoidD(0.5)
        let denom = s0 + s2 + 1e-6
        let expected: [Int: Double] = [
            0: s0 / denom * Double(scaling),
            2: s2 / denom * Double(scaling),
        ]
        for (j, expert) in idx.enumerated() {
            let want = try XCTUnwrap(expected[expert])
            XCTAssertEqual(
                Double(w[j]), want, accuracy: 1e-3,
                "weight for expert \(expert) must be the unbiased sigmoid, normalized, scaled.")
        }
    }

    func testSigmoidRoutingWithoutBiasSelectsUnbiasedTopK() throws {
        let scaling: Float = 1.5
        let config = try makeConfig(useExpertBias: false, scaling: scaling)
        let block = Lfm2MoeSparseMoeBlock(config)

        block.update(
            parameters: ModuleParameters.unflattened([
                "gate.weight": eye(4)
            ]))

        let logits: [Float] = [2.0, 1.0, 0.5, 0.0]
        let x = MLXArray(logits, [1, 4])
        let (indices, weights) = block.route(x)
        eval(indices, weights)

        let idx = indices.asArray(Int32.self).map { Int($0) }
        let w = weights.asArray(Float.self)

        XCTAssertEqual(
            Set(idx), Set([0, 1]),
            "without expert_bias the selection is the unbiased sigmoid top-k.")

        let s0 = sigmoidD(2.0)
        let s1 = sigmoidD(1.0)
        let denom = s0 + s1 + 1e-6
        let expected: [Int: Double] = [
            0: s0 / denom * Double(scaling),
            1: s1 / denom * Double(scaling),
        ]
        for (j, expert) in idx.enumerated() {
            let want = try XCTUnwrap(expected[expert])
            XCTAssertEqual(Double(w[j]), want, accuracy: 1e-3)
        }
    }
}
