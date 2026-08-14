// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import MLXNN
import MLXRandom
import Testing

@Suite("DFlash verify linear")
struct DFlashVerifyLinearTests {
    @Test func optimizedQuantizedLinearMatchesFallbackForSixteenRows() {
        MLXRandom.seed(42)
        let linear = Linear(256, 64, bias: true)
        let quantized = QuantizedLinear(linear, groupSize: 64, bits: 4)
        let verify = DFlashVerifyQuantizedLinear(quantized)
        let x = MLXRandom.normal([1, 16, 256]).asType(.bfloat16)

        let expected = quantized(x)
        let actual = verify(x)
        eval(expected, actual)

        #expect(allClose(actual, expected, rtol: 1e-2, atol: 1e-2).item(Bool.self))
    }

    @Test func optimizedEightBitQuantizedLinearMatchesFallbackForSixteenRows() {
        MLXRandom.seed(43)
        let linear = Linear(256, 64, bias: true)
        let quantized = QuantizedLinear(linear, groupSize: 64, bits: 8)
        let verify = DFlashVerifyQuantizedLinear(quantized)
        let x = MLXRandom.normal([1, 16, 256]).asType(.bfloat16)

        let expected = quantized(x)
        let actual = verify(x)
        eval(expected, actual)

        #expect(allClose(actual, expected, rtol: 1e-2, atol: 1e-2).item(Bool.self))
    }

    @Test func installReplacesOnlyEligibleQuantizedLinears() {
        let model = LinearPair()
        let replaced = DFlashVerifyLinear.install(on: model)

        #expect(replaced == 1)
        #expect(model.good is DFlashVerifyQuantizedLinear)
        #expect(model.bad is QuantizedLinear)
        #expect(!(model.bad is DFlashVerifyQuantizedLinear))
    }

    @Test func installReturnsZeroWhenNoLinearsAreEligible() {
        let model = NoEligibleLinears()
        let replaced = DFlashVerifyLinear.install(on: model)

        #expect(replaced == 0)
        #expect(model.smallQuantized is QuantizedLinear)
        #expect(!(model.smallQuantized is DFlashVerifyQuantizedLinear))
    }

    @Test func installReplacesEligibleEightBitQuantizedLinears() {
        let model = EightBitLinearPair()
        let replaced = DFlashVerifyLinear.install(on: model)

        #expect(replaced == 1)
        #expect(model.good is DFlashVerifyQuantizedLinear)
    }

    @Test func installDoesNotReplaceAlreadyWrappedLinears() {
        let model = LinearPair()
        let firstPass = DFlashVerifyLinear.install(on: model)
        let secondPass = DFlashVerifyLinear.install(on: model)

        #expect(firstPass == 1)
        #expect(secondPass == 0)
        #expect(model.good is DFlashVerifyQuantizedLinear)
    }

    @Test func includeFilterMatchesProjectionFamilies() {
        #expect(
            DFlashVerifyQuantizedLinear.includeAllows(
                path: "model.layers.0.self_attn.q_proj", include: "attn"))
        #expect(
            DFlashVerifyQuantizedLinear.includeAllows(
                path: "model.layers.0.self_attn.o_proj", include: "attn_o"))
        #expect(
            DFlashVerifyQuantizedLinear.includeAllows(
                path: "model.layers.0.mlp.down_proj", include: "mlp"))
        #expect(
            DFlashVerifyQuantizedLinear.includeAllows(
                path: "model.layers.0.router.proj", include: "router"))
        #expect(
            !DFlashVerifyQuantizedLinear.includeAllows(
                path: "model.layers.0.router.proj", include: "attn,mlp"))
    }

    @Test func attentionOutputProjectionUsesFallbackQMMByDefault() {
        #expect(
            DFlashVerifyQuantizedLinear.enablesCustomQMMByDefault(
                path: "model.layers.0.self_attn.q_proj"))
        #expect(
            !DFlashVerifyQuantizedLinear.enablesCustomQMMByDefault(
                path: "model.layers.0.self_attn.o_proj"))
    }
}

private final class LinearPair: Module {
    @ModuleInfo var good: Linear
    @ModuleInfo var bad: Linear

    override init() {
        self._good.wrappedValue = QuantizedLinear(Linear(256, 64), groupSize: 64, bits: 4)
        self._bad.wrappedValue = QuantizedLinear(Linear(128, 64), groupSize: 64, bits: 4)
        super.init()
    }
}

private final class EightBitLinearPair: Module {
    @ModuleInfo var good: Linear

    override init() {
        self._good.wrappedValue = QuantizedLinear(Linear(256, 64), groupSize: 64, bits: 8)
        super.init()
    }
}

private final class NoEligibleLinears: Module {
    @ModuleInfo var floatLinear: Linear
    @ModuleInfo var smallQuantized: Linear

    override init() {
        self._floatLinear.wrappedValue = Linear(256, 64)
        self._smallQuantized.wrappedValue = QuantizedLinear(Linear(128, 64), groupSize: 64, bits: 4)
        super.init()
    }
}
