// Numeric-equivalence gate for the checkpoint RESTORE path (hybrid
// sliding-window models). The non-negotiable correctness property:
//
//   decode(restore@L ; prefill suffix[L..])  ==  decode(cold prefill[0..N])
//
// i.e. seeding the scheduler with a restored mixed (simple + rotating)
// checkpoint cache and decoding the suffix yields the SAME output tokens as
// a cold full-prompt run. Driven end-to-end through the real Scheduler with
// the deterministic synthetic hybrid model (next = (tok+1) % 16); no weights.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

// Mirrors the hybrid model in CBCheckpointCaptureTests: layer 0 full
// (KVCacheSimple), layer 1 sliding (RotatingKVCache).
//
// POSITION-SENSITIVE on purpose: the chosen next-token folds in the rotating
// layer's `offset` (its absolute position) read BEFORE the update. So a
// restore bug that gets the rotating layer's absolute offset wrong (e.g.
// fromSingleRow dropping metaState[3]) or places the suffix at the wrong
// position WOULD change the output — making the equivalence test actually
// gate positional correctness, not just token-id arithmetic.
private final class RestoreHybridModel: Module, LanguageModel, KVCacheDimensionProvider {
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
        let B = inputs.dim(0), S = inputs.dim(1)
        // Read the SLIDING layer's absolute RoPE position exactly as the real
        // Gemma-4 model does — from BatchPositionedKVCache.batchOffset
        // (Gemma4Text.gemma4CapturePositionOffset) — snapshotted BEFORE the
        // update. This is the value fromSingleRow must reproduce on restore;
        // reading it makes the equivalence test actually gate the restored
        // offset. It's the absolute position (uncapped), so basePos+t is
        // chunk-invariant.
        var basePos = 0
        if let cache, cache.count > 1, let pos = cache[1] as? BatchPositionedKVCache {
            basePos = Int(pos.batchOffset[0].item(Int32.self))
        }
        if let cache {
            let keys = MLXArray.ones([B, 1, S, 1], dtype: .float32)
            for c in cache { _ = c.update(keys: keys, values: keys * 2) }
        }
        let tok = inputs.asArray(UInt32.self).map { Int($0) }
        var logits = Array(repeating: Float(-1_000), count: B * S * vocabularySize)
        for b in 0 ..< B {
            for t in 0 ..< S {
                // Fold token id AND absolute position so a wrong restored
                // offset produces wrong tokens.
                let next = (tok[b * S + t] + 1 + basePos + t) % vocabularySize
                logits[(b * S + t) * vocabularySize + next] = 0
            }
        }
        return MLXArray(logits).reshaped([B, S, vocabularySize])
    }
}

private func makeRestoreScheduler(window: Int) -> Scheduler {
    Scheduler(
        model: RestoreHybridModel(window: window),
        tokenizer: IdentityTokenizer(),
        config: SchedulerConfig(
            maxNumSeqs: 4, maxNumBatchedTokens: 128, prefillStepSize: 16, streamInterval: 1),
        eosTokenIds: [])
}

// head_dim = 64 variant so the full-attention layer (layer 0) is KV-quant-
// eligible (groupSize must divide head_dim). Same position-sensitive logit
// rule as RestoreHybridModel; the model ignores K/V VALUES, so quantization
// noise cannot change outputs — these tests gate the restore PATH (quantized
// cache assembly + positional correctness + extendBatched type-compat), not
// quant value fidelity (covered by QuantizedSDPATests). Layer 1 stays sliding
// (fp16 RotatingKVCache), mirroring GPT-OSS's hybrid full+sliding layout.
private final class RestoreQuantHybridModel: Module, LanguageModel, KVCacheDimensionProvider {
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
        let B = inputs.dim(0), S = inputs.dim(1)
        // PER-ROW absolute position (unlike the B==1 RestoreHybridModel which
        // reads row 0) so a multi-row batch — cold + restored together — is
        // positionally correct, which the concurrent test requires.
        var basePos = [Int](repeating: 0, count: B)
        if let cache, cache.count > 1, let pos = cache[1] as? BatchPositionedKVCache {
            let off = pos.batchOffset.asArray(Int32.self)
            for b in 0 ..< B { basePos[b] = b < off.count ? Int(off[b]) : 0 }
        }
        if let cache {
            let keys = MLXArray.ones([B, 1, S, 64], dtype: .float32)
            for c in cache { _ = c.update(keys: keys, values: keys * 2) }
        }
        let tok = inputs.asArray(UInt32.self).map { Int($0) }
        var logits = Array(repeating: Float(-1_000), count: B * S * vocabularySize)
        for b in 0 ..< B {
            for t in 0 ..< S {
                let next = (tok[b * S + t] + 1 + basePos[b] + t) % vocabularySize
                logits[(b * S + t) * vocabularySize + next] = 0
            }
        }
        return MLXArray(logits).reshaped([B, S, vocabularySize])
    }
}

private func makeRestoreQuantScheduler(window: Int) -> Scheduler {
    Scheduler(
        model: RestoreQuantHybridModel(window: window),
        tokenizer: IdentityTokenizer(),
        config: SchedulerConfig(
            maxNumSeqs: 4, maxNumBatchedTokens: 128, prefillStepSize: 16, streamInterval: 1,
            kvQuantization: KVQuantizationConfig(
                groupSize: 64, bits: 8, mode: .affine, cacheKind: .dequant)),
        eosTokenIds: [])
}

/// Run one request to completion, returning its output token ids.
private func runOutputs(_ sched: Scheduler, _ req: Request, maxSteps: Int = 400) -> [Int] {
    sched.addRequest(req)
    var steps = 0
    while !sched.finishedReqIds.contains(req.requestId), steps < maxSteps {
        _ = sched.step(); steps += 1
    }
    return req.outputTokenIds
}

final class CBCheckpointRestoreTests: XCTestCase {

    // The equivalence gate: capture a checkpoint at L on a cold run, then a
    // second request with the SAME prompt restored from that checkpoint must
    // produce identical output tokens.
    func testRestoreMatchesColdOutput() {
        let window = 8
        let prompt = Array(0..<12).map { $0 % 16 }   // 12-token prompt
        let maxTokens = 6

        // --- Cold reference run (also captures a checkpoint at 8). ---
        let coldSched = makeRestoreScheduler(window: window)
        coldSched.checkpointBoundaries = [4, 8]
        final class Cap: @unchecked Sendable { var atL: [Int: [any KVCache]] = [:] }
        let cap = Cap()
        coldSched.onCheckpointCapture = { _, length, caches in cap.atL[length] = caches }
        let coldOut = runOutputs(coldSched, makeIntRequest(tokens: prompt, maxTokens: maxTokens))

        // We need the checkpoint at L=8 to have been captured.
        guard let restoredAt8 = cap.atL[8] else {
            return XCTFail("expected a checkpoint capture at L=8")
        }
        XCTAssertEqual(coldOut.count, maxTokens, "cold run should emit \(maxTokens) tokens")

        // --- Restore run: same prompt, seeded from the L=8 checkpoint. ---
        let warmSched = makeRestoreScheduler(window: window)
        let req = makeIntRequest(tokens: prompt, maxTokens: maxTokens)
        // Deep-copy the captured caches so the cold scheduler's state can't
        // alias the restore (mirrors the provider boxing a fresh copy).
        req.restoredCheckpoint = (caches: restoredAt8.map { $0.copy() }, tokenCount: 8)
        let warmOut = runOutputs(warmSched, req)

        XCTAssertEqual(warmOut, coldOut,
            "restored-checkpoint output must match cold full-prompt output exactly")
        XCTAssertEqual(req.cachedTokens, 8, "restore should report 8 cached prompt tokens")
    }

    // Restore at the smaller boundary (4) must also match cold output.
    func testRestoreAtSmallerBoundaryMatchesCold() {
        let window = 8
        let prompt = Array(3..<17).map { $0 % 16 }   // 14 tokens
        let maxTokens = 5

        let coldSched = makeRestoreScheduler(window: window)
        coldSched.checkpointBoundaries = [4, 8]
        final class Cap: @unchecked Sendable { var atL: [Int: [any KVCache]] = [:] }
        let cap = Cap()
        coldSched.onCheckpointCapture = { _, length, caches in cap.atL[length] = caches }
        let coldOut = runOutputs(coldSched, makeIntRequest(tokens: prompt, maxTokens: maxTokens))

        guard let at4 = cap.atL[4] else { return XCTFail("expected capture at L=4") }
        let warmSched = makeRestoreScheduler(window: window)
        let req = makeIntRequest(tokens: prompt, maxTokens: maxTokens)
        req.restoredCheckpoint = (caches: at4.map { $0.copy() }, tokenCount: 4)
        let warmOut = runOutputs(warmSched, req)

        XCTAssertEqual(warmOut, coldOut, "restore@4 must match cold output")
    }

    // ISOLATION: a request with no restoredCheckpoint behaves exactly as
    // before (cold prefill), even when the scheduler COULD restore.
    func testNoRestoredCheckpointIsCold() {
        let sched = makeRestoreScheduler(window: 8)
        let prompt = Array(0..<10).map { $0 % 16 }
        let out = runOutputs(sched, makeIntRequest(tokens: prompt, maxTokens: 4))
        XCTAssertEqual(out.count, 4, "plain request must complete as a normal cold run")
    }

    // GUARD: a degenerate restoredCheckpoint (tokenCount >= prompt length,
    // or wrong layer count) must be ignored / fall back, never crash and
    // never produce wrong output.
    func testDegenerateRestoreFallsBackToCold() {
        let window = 8
        let prompt = Array(0..<10).map { $0 % 16 }
        let maxTokens = 4

        // Reference cold output.
        let coldOut = runOutputs(makeRestoreScheduler(window: window),
                                 makeIntRequest(tokens: prompt, maxTokens: maxTokens))

        // tokenCount == prompt length (no suffix) → must be ignored → cold.
        let s1 = makeRestoreScheduler(window: window)
        let r1 = makeIntRequest(tokens: prompt, maxTokens: maxTokens)
        r1.restoredCheckpoint = (caches: s1.model.newCache(parameters: nil), tokenCount: prompt.count)
        XCTAssertEqual(runOutputs(s1, r1), coldOut, "tokenCount==len must fall back to cold")

        // Wrong layer count → must fall back to cold.
        let s2 = makeRestoreScheduler(window: window)
        let r2 = makeIntRequest(tokens: prompt, maxTokens: maxTokens)
        r2.restoredCheckpoint = (caches: [KVCacheSimple()], tokenCount: 4)  // 1 layer, model has 2
        XCTAssertEqual(runOutputs(s2, r2), coldOut, "wrong layer count must fall back to cold")

        // Right COUNT but wrong per-position TYPE ORDER (model is
        // [KVCacheSimple, RotatingKVCache]; supply the reverse) → the
        // per-layer type check must reject and fall back, NOT silently
        // rebuild the wrong attention class.
        let s3 = makeRestoreScheduler(window: window)
        let r3 = makeIntRequest(tokens: prompt, maxTokens: maxTokens)
        r3.restoredCheckpoint = (
            caches: [RotatingKVCache(maxSize: window, keep: 0), KVCacheSimple()],
            tokenCount: 4)
        XCTAssertEqual(runOutputs(s3, r3), coldOut, "wrong per-position type order must fall back to cold")
    }

    // DAR-319 v2: restore must match cold output even when KV-quant is ON (the
    // full-attention layer is a QuantizedBatchKVCache). The restore path
    // re-quantizes the captured fp16 prefix into a quantized batched cache via
    // the cold cache factory; this gates that it runs, preserves positions, and
    // matches a cold full-prompt run under the same quantization.
    func testRestoreMatchesColdOutputUnderKVQuant() {
        let window = 8
        let prompt = Array(0..<12).map { $0 % 16 }
        let maxTokens = 6

        let coldSched = makeRestoreQuantScheduler(window: window)
        coldSched.checkpointBoundaries = [4, 8]
        final class Cap: @unchecked Sendable { var atL: [Int: [any KVCache]] = [:] }
        let cap = Cap()
        coldSched.onCheckpointCapture = { _, length, caches in cap.atL[length] = caches }
        let coldOut = runOutputs(coldSched, makeIntRequest(tokens: prompt, maxTokens: maxTokens))

        guard let restoredAt8 = cap.atL[8] else {
            return XCTFail("expected a checkpoint capture at L=8 under KV-quant")
        }
        XCTAssertEqual(coldOut.count, maxTokens)

        let warmSched = makeRestoreQuantScheduler(window: window)
        let req = makeIntRequest(tokens: prompt, maxTokens: maxTokens)
        req.restoredCheckpoint = (caches: restoredAt8.map { $0.copy() }, tokenCount: 8)
        let warmOut = runOutputs(warmSched, req)

        XCTAssertEqual(warmOut, coldOut,
            "restored-checkpoint output must match cold output under KV-quant")
        XCTAssertEqual(req.cachedTokens, 8, "restore should report 8 cached prompt tokens")
    }

    // The real DAR-319 v1 crash this fix removes: a restored row rebuilt as
    // fp16 (old behavior) trips `extendBatched`'s concrete-class precondition
    // when merged into a genBatch holding a QUANTIZED cold row. With the
    // re-quantizing restore both rows are QuantizedBatchKVCache, so a concurrent
    // cold + restored batch coexists and BOTH outputs stay correct.
    func testConcurrentRestoreAndColdUnderKVQuant() {
        let window = 8
        let promptB = Array(0..<12).map { $0 % 16 }
        let promptA = Array(2..<13).map { $0 % 16 }
        let maxTokens = 6

        // Capture a checkpoint for promptB and a cold reference for it.
        let capSched = makeRestoreQuantScheduler(window: window)
        capSched.checkpointBoundaries = [4, 8]
        final class Cap: @unchecked Sendable { var atL: [Int: [any KVCache]] = [:] }
        let cap = Cap()
        capSched.onCheckpointCapture = { _, length, caches in cap.atL[length] = caches }
        let coldB = runOutputs(capSched, makeIntRequest(tokens: promptB, maxTokens: maxTokens))
        guard let restoredAt8 = cap.atL[8] else {
            return XCTFail("expected a checkpoint capture at L=8")
        }
        let coldA = runOutputs(makeRestoreQuantScheduler(window: window),
                               makeIntRequest(tokens: promptA, maxTokens: maxTokens))

        // Concurrent: cold A + restored B share one scheduler → one genBatch.
        // The restored B merges first, then A's cold prefill completes and
        // extendBatched-es into B's quantized genBatch. Pre-fix this crashed.
        let sched = makeRestoreQuantScheduler(window: window)
        let reqA = makeIntRequest(tokens: promptA, maxTokens: maxTokens)
        let reqB = makeIntRequest(tokens: promptB, maxTokens: maxTokens)
        reqB.restoredCheckpoint = (caches: restoredAt8.map { $0.copy() }, tokenCount: 8)
        sched.addRequest(reqA)
        sched.addRequest(reqB)
        var steps = 0
        while !(sched.finishedReqIds.contains(reqA.requestId)
                && sched.finishedReqIds.contains(reqB.requestId)), steps < 800 {
            _ = sched.step(); steps += 1
        }
        XCTAssertEqual(reqB.outputTokenIds, coldB,
            "restored B must match its cold output when batched with a quantized cold row")
        XCTAssertEqual(reqA.outputTokenIds, coldA,
            "cold A must match its cold output when batched with a restored quantized row")
    }
}
