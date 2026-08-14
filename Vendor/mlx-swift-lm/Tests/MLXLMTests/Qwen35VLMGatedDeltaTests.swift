// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

/// Regression tests for the Qwen3.5-VL gated-delta (GDN) recurrence.
///
/// NB-1 fast-follow: the VLM Qwen3.5 path used to carry a *private duplicate*
/// of the gated-delta recurrence that defaulted the recurrent SSM `state` to
/// `q.dtype` (bf16) and downcast `g`/`beta` to bf16. The LLM twin keeps the
/// state in fp32 across the T-step recurrence (upstream c566c95 + 93cf322). The
/// duplicate was deleted in favour of the shared `MLXLLM.gatedDeltaUpdate`;
/// this test pins that the VLM GDN now persists fp32 state into the cache.
///
/// Pre-fix the stored state arrives bf16, so `cache[1].dtype` is `.bfloat16`
/// and this fails; post-fix the shared fp32 implementation writes `.float32`.
final class Qwen35VLMGatedDeltaTests: XCTestCase {

    /// Tiny Qwen3.5 text config (GDN dims only). `TextConfiguration` only has a
    /// `Decodable` init, so build it from JSON. Dims mirror the LLM
    /// `GatedDeltaTests` (Hk=2, Dk=32, Hv=4, Dv=16) so the kernel dispatches.
    private func makeTextConfig() throws -> Qwen35Configuration.TextConfiguration {
        let json = """
            {
                "hidden_size": 64,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 1e-6
            }
            """.data(using: .utf8)!
        return try JSONDecoder().decode(
            Qwen35Configuration.TextConfiguration.self, from: json)
    }

    /// Cast every parameter of a module to bf16, so the GDN inputs are bf16 and
    /// a pre-fix bf16 state default would be observable in the cache.
    private func castToBFloat16(_ module: Module) {
        let bf16 = Dictionary(
            uniqueKeysWithValues: module.parameters().flattened().map {
                ($0.0, $0.1.asType(.bfloat16))
            })
        module.update(parameters: ModuleParameters.unflattened(bf16))
    }

    func testGatedDeltaNetPersistsFloat32State() throws {
        MLXRandom.seed(0)
        let config = try makeTextConfig()
        let net = Qwen35Language.GatedDeltaNet(config)
        castToBFloat16(net)

        let inputs = (MLXRandom.normal([1, 8, config.hiddenSize]) * MLXArray(Float(0.1)))
            .asType(.bfloat16)
        let cache = MambaCache()
        let out = net(inputs, mask: nil, cache: cache)
        eval(out)

        let ssmState = try XCTUnwrap(cache[1], "GDN must write the SSM state to cache[1]")
        eval(ssmState)
        XCTAssertEqual(
            ssmState.dtype, .float32,
            "Qwen3.5-VL GDN recurrent state must be fp32; a bf16 state default "
                + "loses precision across the T-step recurrence (NB-1)."
        )
    }
}
