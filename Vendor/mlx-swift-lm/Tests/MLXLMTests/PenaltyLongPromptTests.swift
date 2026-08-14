// Copyright © 2025 Apple Inc.

import MLX
import MLXLMCommon
import XCTest

/// Regression tests for the penalty-processor crash on VLM (image) requests.
///
/// The VLM generate path hands the penalty `LogitProcessor` a 2-D `[1, seq]`
/// prompt token array. `TokenRing.loadPrompt` previously read the token count
/// from `prompt.dim(0)` (the batch dim, 1) instead of the sequence length, which
/// mis-sized the ring buffer; the next `append` then broadcast `[capacity]`
/// against `[seq + padding]` → a fatal MLX broadcast error that aborted the
/// provider (surfacing to consumers as 502 "provider disconnected") on any image
/// request carrying `repetition_penalty`/`presence_penalty`/`frequency_penalty`.
///
/// Each test feeds a 2-D prompt longer than the context window and then appends a
/// sampled token — the exact sequence that crashed. Before the fix these abort
/// the test runner; after, they pass with correct penalization.
public class PenaltyLongPromptTests: XCTestCase {

    private let vocab = 100
    private func flatLogits(_ value: Float = 4.0) -> MLXArray {
        MLXArray.ones([1, 100], type: Float32.self) * value
    }
    // VLM-shaped prompt: 2-D [1, 50], longer than the 20-token window.
    private func twoDPrompt() -> MLXArray {
        MLXArray((0 ..< 50).map { Int32($0) }).reshaped(1, 50)
    }

    func testRepetitionPenalty2DPromptThenSample() {
        var ctx = RepetitionContext(repetitionPenalty: 2.0, repetitionContextSize: 20)
        ctx.prompt(twoDPrompt())
        var out = ctx.process(logits: flatLogits())
        // token 49 is within the last-20 window → divided by penalty (4/2=2).
        XCTAssertEqual(out[0, 49].item(Float.self), 2.0, accuracy: 1e-4)
        // token 0 is outside the window → unchanged.
        XCTAssertEqual(out[0, 0].item(Float.self), 4.0, accuracy: 1e-4)
        // the step that broadcast-crashed before the fix:
        ctx.didSample(token: MLXArray([Int32(7)]))
        out = ctx.process(logits: flatLogits())
        XCTAssertEqual(out[0, 7].item(Float.self), 2.0, accuracy: 1e-4)
    }

    func testPresencePenalty2DPromptThenSample() {
        var ctx = PresencePenaltyContext(presencePenalty: 1.0, presenceContextSize: 20)
        ctx.prompt(twoDPrompt())
        var out = ctx.process(logits: flatLogits())
        XCTAssertEqual(out[0, 49].item(Float.self), 3.0, accuracy: 1e-4)
        XCTAssertEqual(out[0, 0].item(Float.self), 4.0, accuracy: 1e-4)
        ctx.didSample(token: MLXArray([Int32(7)]))
        out = ctx.process(logits: flatLogits())
        XCTAssertEqual(out[0, 7].item(Float.self), 3.0, accuracy: 1e-4)
    }

    func testFrequencyPenalty2DPromptThenSample() {
        var ctx = FrequencyPenaltyContext(frequencyPenalty: 1.0, frequencyContextSize: 20)
        ctx.prompt(twoDPrompt())
        var out = ctx.process(logits: flatLogits())
        XCTAssertEqual(out[0, 49].item(Float.self), 3.0, accuracy: 1e-4)
        XCTAssertEqual(out[0, 0].item(Float.self), 4.0, accuracy: 1e-4)
        ctx.didSample(token: MLXArray([Int32(7)]))
        out = ctx.process(logits: flatLogits())
        XCTAssertEqual(out[0, 7].item(Float.self), 3.0, accuracy: 1e-4)
    }

    /// `repetition_penalty: 1` is the identity and must not build a processor —
    /// many OpenAI-compatible clients send it by default.
    func testRepetitionPenaltyIdentityProducesNoProcessor() {
        XCTAssertNil(GenerateParameters(repetitionPenalty: 1.0).processor())
        XCTAssertNil(GenerateParameters(repetitionPenalty: 0.0).processor())
        XCTAssertNotNil(GenerateParameters(repetitionPenalty: 1.3).processor())
    }
}
