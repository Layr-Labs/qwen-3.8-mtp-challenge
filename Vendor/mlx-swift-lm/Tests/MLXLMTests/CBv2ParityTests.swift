// CBv2ParityTests.swift — WS-G: v2 vs LEGACY engine parity at B=1
// (report 10 §4 invariant 10: "B=1 / batched / compiled semantic parity as a
// test contract").
//
// All paths run TinyTestModel with the SAME weights:
//  - legacy EAGER reference: the exact cache substitution the legacy
//    Scheduler performs (fp16 cold path: BatchKVCache /
//    BatchRotatingKVCache(maxSize:) at B=1, model-built masks) driven with
//    the engine's chunking discipline (prefill prompt[0..<n-1] in
//    prefillStepSize chunks, then 1-token decode steps). This is the
//    numerically pinned ground truth.
//  - legacy ENGINE: the old continuous-batching engine (Scheduler →
//    EngineCore) at maxNumSeqs=1, temperature=0.
//  - v2: the harness engine (per-request KV, per-row attention), B=1.
// All are unpadded single-request paths. v2 vs eager-reference is asserted
// token-exact UNCONDITIONALLY — any divergence is a v2 bug. v2 vs the full
// engine is asserted token-exact whenever the engine is on its eager path;
// see the KNOWN LEGACY DISCREPANCY note below.
//
// KNOWN LEGACY DISCREPANCY (found by this suite; not a v2 bug): the legacy
// engine's compiled B=1 decode path (`CompiledDecode`, ON by default via
// DARKBLOOM_COMPILED_DECODE) diverges from the legacy engine's OWN eager
// path on prompts that are already past a sliding-window boundary when
// compiled decode is set up (prompt > windowSize). With
// DARKBLOOM_COMPILED_DECODE=0 the engine is token-exact vs both the eager
// reference and v2 on every case in this suite. Details:
// docs/engine-v2/CONTRACT-ISSUES-G-harness.md item 2. When the compiled
// path is active AND diverges from eager, the engine-level assertion is
// skipped (with the doc pointer); the eager-reference assertion still gates.
// We deliberately do NOT flip DARKBLOOM_COMPILED_DECODE from inside the
// test process: `CompiledDecode.isEnabled` is a lazy process-global, and
// mutating the env here would silently disable the legacy engine's compiled
// coverage for every suite that runs after this one.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class CBv2ParityTests: XCTestCase {

    /// Legacy prefill chunk size; the v2 harness uses the same value so both
    /// paths issue identical per-chunk shapes (bit-exact kernel dispatch).
    private static let prefillStepSize = 16

    // MARK: - Legacy EAGER reference (numerically pinned ground truth)

    /// Drive TinyTestModel exactly as the legacy engine's eager path does at
    /// B=1: the Scheduler's fp16 cold cache substitution (KVCacheSimple →
    /// BatchKVCache, RotatingKVCache → BatchRotatingKVCache — see
    /// `Scheduler.makeCacheFactory`), model-built masks, prefill over
    /// prompt[0..<n-1] in `prefillStepSize` chunks, then greedy 1-token
    /// decode steps seeded with the last prompt token.
    private func runLegacyEagerReference(
        model: TinyTestModel, prompt: [Int], maxTokens: Int
    ) -> [Int] {
        let caches: [KVCache] = [
            BatchKVCache(leftPadding: [0]),
            BatchRotatingKVCache(maxSize: model.config.windowSize, leftPadding: [0]),
        ]

        let prefillTokens = Array(prompt.dropLast())
        var index = 0
        while index < prefillTokens.count {
            let chunk = Array(
                prefillTokens[index ..< min(index + Self.prefillStepSize, prefillTokens.count)])
            _ = model(
                MLXArray(chunk.map { Int32($0) }).reshaped(1, chunk.count), cache: caches)
            index += chunk.count
        }

        var output: [Int] = []
        var lastToken = prompt.last!
        for _ in 0 ..< maxTokens {
            let logits = model(
                MLXArray([Int32(lastToken)]).reshaped(1, 1), cache: caches)
            lastToken = argMax(logits[0..., -1, 0...], axis: -1).item(Int.self)
            output.append(lastToken)
        }
        return output
    }

    // MARK: - Legacy B=1 ENGINE driver

    /// Run one greedy request through the OLD engine (Scheduler/EngineCore,
    /// which substitutes BatchKVCache / BatchRotatingKVCache) and return its
    /// generated token ids.
    private func runLegacyGreedy(
        model: TinyTestModel, prompt: [Int], maxTokens: Int
    ) async throws -> [Int] {
        let scheduler = Scheduler(
            model: model,
            tokenizer: IdentityTokenizer(),
            config: SchedulerConfig(
                maxNumSeqs: 1,
                maxNumBatchedTokens: 512,
                prefillStepSize: Self.prefillStepSize,
                streamInterval: 1
            ),
            eosTokenIds: []  // no EOS — finish on maxTokens only
        )
        let engine = EngineCore(
            scheduler: scheduler,
            config: ContinuousBatchingConfig(stepInterval: 0.0005, yieldInterval: 1)
        )
        engine.start()
        defer { engine.stop() }

        let request = Request(
            requestId: UUID().uuidString,
            prompt: prompt as AnyHashable,
            samplingParams: SamplingParams(maxTokens: maxTokens, temperature: 0)
        )
        let requestId = await engine.addRequest(request)

        var output: [Int] = []
        for await chunk in engine.streamOutputs(requestId: requestId) {
            output = chunk.outputTokenIds
            if chunk.finished {
                XCTAssertNil(chunk.error, "legacy engine errored: \(chunk.error ?? "")")
                break
            }
        }
        return output
    }

    private func assertParity(
        prompt: [Int], maxTokens: Int, withSinks: Bool = false,
        file: StaticString = #filePath, line: UInt = #line
    ) async throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE, withSinks: withSinks)

        // 1. v2 vs the eager reference — UNCONDITIONAL. Any divergence here
        //    is a v2 bug; fail before any engine-path consideration.
        let eager = runLegacyEagerReference(
            model: model, prompt: prompt, maxTokens: maxTokens)
        let v2 = try CBv2HarnessEngine.runSolo(
            model: model, prompt: prompt, maxTokens: maxTokens,
            prefillChunkSize: Self.prefillStepSize)
        XCTAssertEqual(
            v2, eager,
            "v2 B=1 diverged from the legacy EAGER reference "
                + "(prompt \(prompt.count) tokens, sinks=\(withSinks))",
            file: file, line: line)

        // 2. v2 vs the full legacy engine. Token-exact whenever the engine is
        //    on its eager path; the one known legacy compiled-decode
        //    discrepancy downgrades to a documented skip (header comment).
        let legacy = try await runLegacyGreedy(
            model: model, prompt: prompt, maxTokens: maxTokens)
        XCTAssertEqual(
            legacy.count, maxTokens, "legacy run did not complete", file: file, line: line)
        if CompiledDecode.isEnabled && legacy != eager {
            throw XCTSkip(
                "legacy engine took its compiled B=1 decode path and diverged from its OWN "
                    + "eager path on this fixture (pre-existing legacy compiled-decode "
                    + "discrepancy for prompts already past the sliding window at compile "
                    + "time; v2 matches eager token-exactly — see "
                    + "docs/engine-v2/CONTRACT-ISSUES-G-harness.md item 2). "
                    + "Run with DARKBLOOM_COMPILED_DECODE=0 for strict engine-level parity.")
        }
        XCTAssertEqual(
            v2, legacy,
            "v2 B=1 diverged from legacy engine B=1 "
                + "(prompt \(prompt.count) tokens, sinks=\(withSinks))",
            file: file, line: line)
    }

    // MARK: - Greedy token parity

    /// Prompt shorter than the sliding window: window is crossed mid-decode.
    func testGreedyParityShortPrompt() async throws {
        try await assertParity(
            prompt: makePromptTokens(length: 8, seed: 10), maxTokens: 48)
    }

    /// Prompt longer than the sliding window: windowed prefill straddles the
    /// eviction boundary (multi-chunk), then decode crosses it again.
    func testGreedyParityPromptLongerThanWindow() async throws {
        try await assertParity(
            prompt: makePromptTokens(length: 24, seed: 11), maxTokens: 48)
    }

    /// Prompt spanning several prefill chunks and several windows.
    func testGreedyParityMultiChunkPrompt() async throws {
        try await assertParity(
            prompt: makePromptTokens(length: 50, seed: 12), maxTokens: 32)
    }

    /// Sink-enabled fixture (GPT-OSS shape): sinks must be folded into the
    /// softmax identically on both paths.
    func testGreedyParityWithSinks() async throws {
        try await assertParity(
            prompt: makePromptTokens(length: 24, seed: 13), maxTokens: 32, withSinks: true)
    }

    // MARK: - Offset / RoPE position parity

    /// After prefill + k decode steps, the v2 per-sequence absolute offsets
    /// must equal the position a solo legacy run would use: prompt + k
    /// tokens processed, for EVERY layer — full and windowed alike (the
    /// windowed layer keeps counting past its retention).
    func testAbsoluteOffsetParityAfterDecode() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let prompt = makePromptTokens(length: 24, seed: 14)
        let decodeSteps = 20

        let engine = CBv2HarnessEngine(
            model: model, prefillChunkSize: Self.prefillStepSize)
        let id = try engine.join(prompt: prompt, maxTokens: decodeSteps)
        while !(engine.active.first(where: { $0.id == id })?.finished ?? true) {
            engine.decodeStep()
        }

        let state = engine.active.first(where: { $0.id == id })!
        // Tokens written to KV: (prompt - 1) prefilled + 1 last prompt token
        // + (decodeSteps - 1) fed-back generations — the final sampled token
        // has not been fed back yet.
        let expected = prompt.count + decodeSteps - 1

        for (layer, kv) in state.kv.enumerated() {
            let kv = try XCTUnwrap(kv)
            XCTAssertEqual(
                kv.absoluteOffset, expected,
                "layer \(layer) absolute offset drifted")
        }

        // Full layer retains everything; windowed layer retains exactly its
        // window at decode steady state.
        let full = try XCTUnwrap(state.kv[0])
        let windowed = try XCTUnwrap(state.kv[1] as? HarnessWindowedSequenceKV)
        XCTAssertEqual(full.retainedCount, expected)
        XCTAssertEqual(windowed.retainedCount, windowed.window)
    }

    /// The device-side positionOffsets array [B] must mirror each row's
    /// host-side absolute counter, in row order, under join/leave churn.
    func testPositionOffsetsMirrorRowCounters() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let engine = CBv2HarnessEngine(model: model)

        let a = try engine.join(prompt: makePromptTokens(length: 5, seed: 15), maxTokens: 30)
        _ = try engine.join(prompt: makePromptTokens(length: 21, seed: 16), maxTokens: 30)
        for _ in 0 ..< 6 { engine.decodeStep() }
        engine.remove(id: a)
        _ = try engine.join(prompt: makePromptTokens(length: 9, seed: 17), maxTokens: 30)
        for _ in 0 ..< 3 { engine.decodeStep() }

        for cache in engine.layerCaches {
            let device = cache.positionOffsets.asArray(Int32.self).map(Int.init)
            let host = cache.rows.map { $0.absoluteOffset }
            XCTAssertEqual(device, host, "layer \(cache.layerIndex) positionOffsets drifted")
        }
    }

    /// RoPE numerical parity: applying RoPE with a scalar Int offset (legacy
    /// B=1) and with an MLXArray offset vector (v2 batched path) must agree —
    /// this pins the two overloads the two engines use onto each other.
    func testRoPEScalarVsArrayOffsetParity() {
        let config = TinyTestModelConfig()
        MLXRandom.seed(99)
        let x = MLXRandom.normal([1, config.queryHeads, 1, config.headDim])
        let rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeBase)

        for offset in [0, 1, 15, 16, 17, 100, 4096] {
            let scalar = rope(x, offset: offset)
            let array = rope(x, offset: MLXArray([Int32(offset)]))
            XCTAssertTrue(
                allClose(scalar, array, rtol: 0, atol: 0).item(Bool.self),
                "RoPE scalar-vs-array offset mismatch at offset \(offset)")
        }
    }
}
