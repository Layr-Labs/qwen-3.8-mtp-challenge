// Tests for the checkpoint KV-cache capture hook in Scheduler
// (advancePendingPrefill). Verifies (a) ISOLATION — with no hook installed
// the prefill path is unchanged and no capture occurs; (b) the hook fires
// exactly at checkpoint boundaries with the correct prefix tokens; (c) the
// captured caches are the per-row single-stream form (mixed simple +
// rotating) ready to restore. Uses a deterministic synthetic hybrid model;
// no weights required.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

// Hybrid synthetic model: layer 0 full-attention (KVCacheSimple), layer 1
// sliding-window (RotatingKVCache) — the Gemma-4 / GPT-OSS shape in
// miniature. Deterministic next = (tok+1) % 16.
private final class HybridIncrModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize = 16
    let window: Int
    var kvHeads: [Int] { [1, 1] }

    init(window: Int) { self.window = window }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        [KVCacheSimple(), RotatingKVCache(maxSize: window, keep: 0)]
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)
        if let cache {
            let keys = MLXArray.ones([B, 1, S, 1], dtype: .float32)
            let vals = keys * 2
            for c in cache { _ = c.update(keys: keys, values: vals) }
        }
        let tok = inputs.asArray(UInt32.self).map { Int($0) }
        var logits = Array(repeating: Float(-1_000), count: B * S * vocabularySize)
        for b in 0 ..< B {
            for t in 0 ..< S {
                let next = (tok[b * S + t] + 1) % vocabularySize
                logits[(b * S + t) * vocabularySize + next] = 0
            }
        }
        return MLXArray(logits).reshaped([B, S, vocabularySize])
    }
}

private func makeHybridScheduler(window: Int) -> Scheduler {
    Scheduler(
        model: HybridIncrModel(window: window),
        tokenizer: IdentityTokenizer(),
        config: SchedulerConfig(
            maxNumSeqs: 4, maxNumBatchedTokens: 128, prefillStepSize: 16, streamInterval: 1),
        eosTokenIds: []
    )
}

/// Drive a single request to completion through a synchronous step loop.
private func runToCompletion(_ sched: Scheduler, _ req: Request, maxSteps: Int = 200) {
    sched.addRequest(req)
    var steps = 0
    while !sched.finishedReqIds.contains(req.requestId), steps < maxSteps {
        _ = sched.step()
        steps += 1
    }
}

final class CBCheckpointCaptureTests: XCTestCase {

    // ISOLATION: no hook installed ⇒ no capture, and the request still
    // completes normally (the default prefill path is unchanged).
    func testNoHookNoCaptureAndNormalCompletion() {
        let sched = makeHybridScheduler(window: 8)
        // 20-token prompt, boundaries would be [4,8] if set — but no hook.
        let prompt = Array(0..<20).map { $0 % 16 }
        let req = makeIntRequest(tokens: prompt, maxTokens: 3)
        // onCheckpointCapture stays nil; checkpointBoundaries stays empty.
        runToCompletion(sched, req)
        XCTAssertTrue(sched.finishedReqIds.contains(req.requestId),
                      "request must complete normally with no hook")
    }

    // CAPTURE: hook fires at each boundary ≤ prompt length, with the exact
    // prefix tokens and one cache per layer (simple + rotating).
    func testCaptureFiresAtBoundariesWithCorrectTokens() {
        let window = 8
        let sched = makeHybridScheduler(window: window)
        sched.checkpointBoundaries = [4, 8]  // both ≤ window

        // Capture log: (checkpointLength, prefixTokens, layerClassNames).
        final class Log: @unchecked Sendable {
            var events: [(Int, [Int], [String])] = []
            let lock = NSLock()
        }
        let log = Log()
        sched.onCheckpointCapture = { tokens, length, caches in
            log.lock.lock()
            let names = caches.map { String(describing: type(of: $0)) }
            log.events.append((length, tokens, names))
            log.lock.unlock()
        }

        // 12-token prompt → boundaries 4 and 8 are both reached during prefill.
        let prompt = Array(0..<12).map { $0 % 16 }
        let req = makeIntRequest(tokens: prompt, maxTokens: 2)
        runToCompletion(sched, req)

        log.lock.lock(); let events = log.events; log.lock.unlock()
        let lengths = events.map { $0.0 }.sorted()
        XCTAssertEqual(lengths, [4, 8], "capture must fire exactly at boundaries 4 and 8, got \(lengths)")

        for (length, tokens, names) in events {
            XCTAssertEqual(tokens, Array(prompt.prefix(length)),
                           "captured prefix tokens must equal the first \(length) prompt tokens")
            XCTAssertEqual(tokens.count, length, "prefix length must equal the boundary")
            XCTAssertEqual(names.count, 2, "one cache per layer")
            XCTAssertTrue(names[0].contains("KVCacheSimple"), "layer 0 is the full-attention cache")
            XCTAssertTrue(names[1].contains("RotatingKVCache"), "layer 1 is the sliding cache")
        }
    }

    // The captured cache is restorable: its rotating layer round-trips and
    // resumes (delegates to fromSingleRow, proven elsewhere) — here we just
    // assert the captured rotating layer has the expected windowed length.
    func testCapturedRotatingLayerIsWindowed() {
        let window = 4
        let sched = makeHybridScheduler(window: window)
        sched.checkpointBoundaries = [4]

        final class Box: @unchecked Sendable { var caches: [any KVCache]? }
        let box = Box()
        sched.onCheckpointCapture = { _, _, caches in box.caches = caches }

        let prompt = Array(0..<6).map { $0 % 16 }
        runToCompletion(sched, makeIntRequest(tokens: prompt, maxTokens: 2))

        guard let caches = box.caches else { return XCTFail("no capture occurred") }
        XCTAssertEqual(caches.count, 2)
        // Rotating layer (window 4) captured at boundary 4 holds ≤4 tokens.
        if let rot = caches[1] as? RotatingKVCache, let k = rot.state.first {
            XCTAssertLessThanOrEqual(k.dim(2), window,
                "rotating layer must not exceed its window at capture")
            XCTAssertGreaterThan(k.dim(2), 0, "rotating layer must hold the windowed prefix")
        } else {
            XCTFail("layer 1 should be a non-empty RotatingKVCache")
        }
    }

    // Capture must NOT fire when batch size > 1 (B==1 restriction) — drive
    // two cold requests together and assert no capture.
    func testNoCaptureForMultiRowBatch() {
        let sched = makeHybridScheduler(window: 8)
        sched.checkpointBoundaries = [4, 8]
        final class Counter: @unchecked Sendable { var n = 0; let lock = NSLock() }
        let counter = Counter()
        sched.onCheckpointCapture = { _, _, _ in
            counter.lock.lock(); counter.n += 1; counter.lock.unlock()
        }

        // Submit two requests before stepping so they batch together cold.
        let p = Array(0..<12).map { $0 % 16 }
        sched.addRequest(makeIntRequest(id: "a", tokens: p, maxTokens: 2))
        sched.addRequest(makeIntRequest(id: "b", tokens: p, maxTokens: 2))
        var steps = 0
        while sched.finishedReqIds.count < 2, steps < 400 { _ = sched.step(); steps += 1 }

        counter.lock.lock(); let n = counter.n; counter.lock.unlock()
        XCTAssertEqual(n, 0, "capture must be suppressed for multi-row (B>1) batches")
    }
}
