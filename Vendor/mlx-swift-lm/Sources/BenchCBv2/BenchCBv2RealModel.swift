// BenchCBv2RealModel — real-weights validation driver for the
// ContinuousBatchingV2 engine (EngineV2) against the legacy engine
// (Scheduler → EngineCore), per docs/engine-v2/specs/G-harness.md
// "integration re-baselines on real weights".
//
// Loads a LOCAL model directory explicitly through `LLMModelFactory.shared`
// (the registry tries the VLM factory first; text-only Gemma-4 checkpoints
// carry a vision_config + vision tower that would otherwise load the MLXVLM
// Gemma4, which has no CBv2 hooks). One loaded model instance drives every
// engine, so comparisons are same-weights by construction.
//
// Modes:
//   correctness (v2 contiguous, greedy):
//     1. b1-sanity ... chat prompt → coherent text, EOS stop, no repetition.
//     2. invariance .. same prompt solo / B=3 burst (short+long neighbors) /
//                      joining mid-stream — token-exact identical.
//     3. chunked ..... ~1500-token prompt, prefillChunkSize 512 vs
//                      no-chunking — token-exact identical.
//   perf: B ∈ {1,2,4} with mixed prompt lengths (~100/~500/~1500),
//     maxTokens 128, engines legacy / v2 (contiguous) / v2-paged.
//     Compiled decode for the legacy engine is a process-wide switch
//     (DARKBLOOM_COMPILED_DECODE, static let) — run separate processes to
//     compare compiled vs plain legacy.
//
// Usage:
//   swift run -c release BenchCBv2 --model <dir> [--mode all|correctness|perf]
//       [--engines legacy,v2,v2-paged] [--batches 1,2,4] [--steps 128]
//       [--label tag] [--out report.md]

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - Deterministic PRNG (same SplitMix64 as benchmarks/CBv2Benchmark.swift)

struct BenchRNG: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Synthetic prompt drawn from a "safe" id range (avoids BOS/EOS/control
/// ids at the top and bottom of the vocab) — identical across engines for a
/// given (length, seed).
func syntheticPrompt(length: Int, seed: UInt64, vocabSize: Int) -> [Int] {
    var rng = BenchRNG(seed: seed)
    let upper = min(vocabSize - 1, 50_000)
    return (0 ..< length).map { _ in Int.random(in: 100 ..< upper, using: &rng) }
}

func percentile(_ sorted: [Double], _ q: Double) -> Double {
    guard !sorted.isEmpty else { return 0 }
    let rank = q * Double(sorted.count - 1)
    let lo = Int(rank.rounded(.down)), hi = Int(rank.rounded(.up))
    if lo == hi { return sorted[lo] }
    let w = rank - Double(lo)
    return sorted[lo] * (1 - w) + sorted[hi] * w
}

// MARK: - V2 stack construction

/// CBv2 structural hooks for the loaded model (layer kinds + cache builder,
/// which for GPT-OSS also primes the sinks-activation probe at build time).
struct V2ModelHooks {
    let layerKinds: [CBv2LayerKind]
    let buildCaches:
        (@escaping (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache) throws ->
            [any CBv2AttendingLayerCache]
}

func v2Hooks(for model: any LanguageModel) -> V2ModelHooks? {
    if let m = model as? Gemma4Model {
        return V2ModelHooks(layerKinds: m.textModel.cbv2LayerKinds) { mk in
            try m.textModel.newCacheV2(makeLayerCache: mk)
        }
    }
    if let m = model as? Gemma4TextModel {
        return V2ModelHooks(layerKinds: m.cbv2LayerKinds) { mk in
            try m.newCacheV2(makeLayerCache: mk)
        }
    }
    if let m = model as? GPTOSSModel {
        return V2ModelHooks(layerKinds: m.cbv2LayerKinds) { mk in
            try m.newCacheV2(makeLayerCache: mk)
        }
    }
    return nil
}

/// Compiled-decode KV capacity override (--kv-capacity), 0 = default 4096.
nonisolated(unsafe) var benchCompiledKVCapacity = 4096

enum V2Backend: String {
    case contiguous = "v2"
    case paged = "v2-paged"
}

/// Build a fresh EngineV2 over the (already loaded) model. Throws
/// CBv2KVError.backendIneligible when the paged kernel cannot serve this
/// model/hardware — callers skip gracefully.
///
/// `compiledDecode` is EXPLICIT here (the engine's own default is on for
/// eligible models): the bench compares eager v2 against compiled v2, so
/// plain "v2" cells must stay eager.
func makeV2Engine(
    context: ModelContext, hooks: V2ModelHooks, backend: V2Backend,
    schedulerConfig: CBv2SchedulerConfig, kvBytes: Int, compiledDecode: Bool = false
) throws -> EngineV2 {
    let kvBackend: CBv2KVBackend
    let caches: [any CBv2AttendingLayerCache]
    switch backend {
    case .contiguous:
        kvBackend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: kvBytes))
        caches = try hooks.buildCaches { index, kind in
            CBv2LayerCache(layerIndex: index, kind: kind)
        }
    case .paged:
        // Static kernel eligibility (head dims + the part kernel's
        // threadgroup-memory budget) is validated inside
        // `PagedKVBackend.init` — it throws `backendIneligible` before any
        // dispatch could trap. Gemma-4 global layers (headDim 512 / GQA 8)
        // are eligible via the kernel's head split.
        let paged = try PagedKVBackend(
            layerKinds: hooks.layerKinds,
            config: PagedKVPoolConfig(
                capacityBytes: kvBytes,
                maxPrefillChunk: schedulerConfig.prefillChunkSize,
                nominalMaxSequenceLength: 4096))
        let pagedCaches = paged.makeLayerCaches()
        // Route through newCacheV2 so GPT-OSS primes its sinks probe.
        caches = try hooks.buildCaches { index, _ in pagedCaches[index] }
        kvBackend = paged
    }
    return EngineV2(
        model: CBv2SteppableLanguageModelAdapter(context.model),
        layerKinds: hooks.layerKinds,
        backend: kvBackend,
        cacheProvider: CBv2LayerCacheBank(caches: caches),
        sampler: CBv2DefaultSampler(),
        detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: context.tokenizer),
        schedulerConfig: schedulerConfig,
        compiledDecodeConfig: CBv2CompiledDecodeConfig(
            enabled: compiledDecode, kvCapacity: benchCompiledKVCapacity))
}

// MARK: - Request runners

struct RunResult: Sendable {
    var tokens: [Int] = []
    var logprobs: [CBv2TokenLogprob] = []
    var text = ""
    var finish: String = "?"
    var promptTokens = 0
    var completionTokens = 0
    var submittedAt: Double = 0
    var firstTokenAt: Double = 0
    var lastTokenAt: Double = 0
    var finishedAt: Double = 0
    var itlsMs: [Double] = []

    /// 0 when no token ever arrived (errored request).
    var ttftMs: Double { firstTokenAt == 0 ? 0 : (firstTokenAt - submittedAt) * 1e3 }
    /// Steady-state decode TPS (first token excluded — it carries prefill).
    var decodeTPS: Double {
        guard tokens.count > 1, lastTokenAt > firstTokenAt else { return 0 }
        return Double(tokens.count - 1) / (lastTokenAt - firstTokenAt)
    }
}

actor TokenCounter {
    private(set) var count = 0
    func add(_ n: Int) { count += n }
}

func collectV2(
    _ stream: AsyncStream<CBv2Event>, submittedAt: Double, counter: TokenCounter? = nil
) async -> RunResult {
    var r = RunResult()
    r.submittedAt = submittedAt
    var last: Double?
    for await event in stream {
        let now = CFAbsoluteTimeGetCurrent()
        switch event {
        case .delta(let text, let tokens, let logprobs):
            if !tokens.isEmpty {
                if last == nil {
                    r.firstTokenAt = now
                } else {
                    r.itlsMs.append((now - last!) * 1e3)
                }
                last = now
                r.lastTokenAt = now
                r.tokens.append(contentsOf: tokens)
                if let logprobs { r.logprobs.append(contentsOf: logprobs) }
                if let counter { await counter.add(tokens.count) }
            }
            r.text += text
        case .finished(let reason, let usage):
            r.finish = "\(reason)"
            r.promptTokens = usage.promptTokens
            r.completionTokens = usage.completionTokens
            r.finishedAt = now
        }
    }
    if r.finishedAt == 0 { r.finishedAt = CFAbsoluteTimeGetCurrent() }
    return r
}

func runV2Request(
    engine: EngineV2, id: UInt64, promptTokens: [Int], maxTokens: Int, stopTokens: Set<Int>,
    topLogprobs: Int = 0
) async throws -> RunResult {
    let submittedAt = CFAbsoluteTimeGetCurrent()
    let stream = try engine.submit(
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: promptTokens,
            sampling: CBv2SamplingParams(temperature: 0, topLogprobs: topLogprobs),
            maxTokens: maxTokens, stopTokens: stopTokens))
    return await collectV2(stream, submittedAt: submittedAt)
}

/// Human-readable dump of one side's logprob record at the divergence index:
/// the chosen token's raw logprob plus the top alternatives and the gap to
/// the runner-up (a near-zero gap = argmax tie flip from batched-kernel
/// numerics, not an engine logic bug).
func logprobDetail(_ r: RunResult, at index: Int, label: String) -> String {
    guard let lp = r.logprobs[safe: index] else { return "\(label): no logprobs captured" }
    let alts = lp.topLogprobs.prefix(4)
        .map { String(format: "%d:%.4f", $0.token, $0.logprob) }
        .joined(separator: " ")
    var gap = ""
    if lp.topLogprobs.count >= 2 {
        gap = String(format: " gap(top1-top2)=%.5f", lp.topLogprobs[0].logprob - lp.topLogprobs[1].logprob)
    }
    return String(format: "%@: chose %d (logprob %.4f) top=[%@]%@", label, lp.token, lp.logprob, alts, gap)
}

func runLegacyRequest(
    engine: EngineCore, id: String, promptTokens: [Int], maxTokens: Int
) async -> RunResult {
    var r = RunResult()
    r.submittedAt = CFAbsoluteTimeGetCurrent()
    let request = Request(
        requestId: id,
        prompt: promptTokens as AnyHashable,
        samplingParams: SamplingParams(maxTokens: maxTokens, temperature: 0))
    let requestId = await engine.addRequest(request)
    var last: Double?
    var seen = 0
    for await chunk in engine.streamOutputs(requestId: requestId) {
        let now = CFAbsoluteTimeGetCurrent()
        if chunk.outputTokenIds.count > seen {
            if last == nil {
                r.firstTokenAt = now
            } else {
                r.itlsMs.append((now - last!) * 1e3)
            }
            last = now
            r.lastTokenAt = now
            seen = chunk.outputTokenIds.count
            r.tokens = chunk.outputTokenIds
        }
        if chunk.finished {
            r.finish = chunk.finishReason ?? "finished"
            r.finishedAt = now
            break
        }
    }
    if r.finishedAt == 0 { r.finishedAt = CFAbsoluteTimeGetCurrent() }
    r.completionTokens = r.tokens.count
    r.promptTokens = promptTokens.count
    return r
}

// MARK: - Perf cells

struct CellResult: Sendable {
    var engine: String
    var batch: Int
    var promptMix: [Int]
    var decodeTPSPerRequest: Double
    var aggregateTPS: Double
    var ttftP50Ms: Double
    var itlP50Ms: Double
    var perRequest: [String]

    var markdownRow: String {
        String(
            format: "| %@ | %d | %@ | %.1f | %.1f | %.0f | %.1f |",
            engine, batch, promptMix.map(String.init).joined(separator: "/"),
            decodeTPSPerRequest, aggregateTPS, ttftP50Ms, itlP50Ms)
    }

    static let markdownHeader = """
        | engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
        |---|---|---|---|---|---|---|
        """
}

func summarize(engine: String, batch: Int, promptMix: [Int], results: [RunResult]) -> CellResult {
    let totalTokens = results.reduce(0) { $0 + $1.tokens.count }
    let start = results.map(\.submittedAt).min() ?? 0
    let end = results.map(\.finishedAt).max() ?? 1
    let ttfts = results.map(\.ttftMs).sorted()
    let itls = results.flatMap(\.itlsMs).sorted()
    let decodeTPS = results.map(\.decodeTPS)
    let perRequest = results.enumerated().map { i, r in
        String(
            format: "    req %d: prompt=%d tokens=%d ttft=%.0fms decodeTPS=%.1f finish=%@",
            i, r.promptTokens, r.tokens.count, r.ttftMs, r.decodeTPS, r.finish)
    }
    return CellResult(
        engine: engine, batch: batch, promptMix: promptMix,
        decodeTPSPerRequest: decodeTPS.reduce(0, +) / Double(max(1, decodeTPS.count)),
        aggregateTPS: Double(totalTokens) / max(end - start, 1e-9),
        ttftP50Ms: percentile(ttfts, 0.5),
        itlP50Ms: percentile(itls, 0.5),
        perRequest: perRequest)
}

func promptMix(batch: Int) -> [Int] {
    switch batch {
    case 1: return [500]
    case 2: return [100, 1500]
    case 4: return [100, 500, 1500, 500]
    default: return Array(repeating: 500, count: batch)
    }
}

func runV2Cell(
    context: ModelContext, hooks: V2ModelHooks, backend: V2Backend,
    batch: Int, promptLengths: [Int], steps: Int, vocabSize: Int, kvBytes: Int,
    compiledDecode: Bool = false, emit: ((String) -> Void)? = nil
) async throws -> CellResult {
    let engine = try makeV2Engine(
        context: context, hooks: hooks, backend: backend,
        schedulerConfig: CBv2SchedulerConfig(
            maxConcurrentRequests: batch, maxBatchedTokensPerStep: 2048,
            prefillChunkSize: 512, maxWaiting: 16),
        kvBytes: kvBytes, compiledDecode: compiledDecode)
    if compiledDecode {
        // Absorb this fresh engine's per-bucket traces (the compile cache is
        // per engine instance) so measured TTFTs reflect steady state; the
        // production provider pays this once at model load.
        let warmStart = CFAbsoluteTimeGetCurrent()
        _ = try? await runV2Request(
            engine: engine, id: 999_999,
            promptTokens: syntheticPrompt(length: 8, seed: 0xBEEF, vocabSize: vocabSize),
            maxTokens: 2, stopTokens: [])
        if let emit, let stats = engine.compiledDecodeStats {
            let perBucket = stats.warmup
                .map { String(format: "b%d=%.2fs", $0.bucket, $0.seconds) }
                .joined(separator: " ")
            emit(String(
                format: "    [v2-compiled B=%d] warmup %.2fs (%@)%@",
                batch, CFAbsoluteTimeGetCurrent() - warmStart, perBucket,
                stats.disabledReason.map { " DISABLED: \($0)" } ?? ""))
        }
    }
    // Measurement boundary: reset the process-wide peak AFTER engine
    // construction and the compiled-decode warm-up so the caller's per-row
    // gpuPeak covers only the measured request group — not trace-building
    // or an earlier cell's high-water mark.
    MLX.Memory.peakMemory = 0
    let results = await withTaskGroup(of: (Int, RunResult).self) { group in
        for (i, length) in promptLengths.enumerated() {
            let prompt = syntheticPrompt(
                length: length, seed: 0xFACE &+ UInt64(i), vocabSize: vocabSize)
            group.addTask { @Sendable in
                let r = try? await runV2Request(
                    engine: engine, id: UInt64(1000 + i), promptTokens: prompt,
                    maxTokens: steps, stopTokens: [])
                return (i, r ?? RunResult())
            }
        }
        var all: [(Int, RunResult)] = []
        for await entry in group { all.append(entry) }
        return all.sorted { $0.0 < $1.0 }.map(\.1)
    }
    await engine.shutdown()
    if compiledDecode, let emit, let stats = engine.compiledDecodeStats {
        emit(
            "    [v2-compiled B=\(batch)] compiledSteps=\(stats.compiledSteps) "
                + "rebinds=\(stats.laneRebinds) scratchResets=\(stats.scratchResets) "
                + "fallbacks=\(stats.fallbacks)")
    }
    return summarize(
        engine: compiledDecode ? "v2-compiled" : backend.rawValue,
        batch: batch, promptMix: promptLengths, results: results)
}

func runLegacyCell(
    context: ModelContext, batch: Int, promptLengths: [Int], steps: Int, vocabSize: Int,
    label: String
) async -> CellResult {
    let scheduler = Scheduler(
        model: context.model,
        tokenizer: context.tokenizer,
        config: SchedulerConfig(
            maxNumSeqs: batch, maxNumBatchedTokens: 4096, prefillStepSize: 512,
            streamInterval: 1),
        eosTokenIds: [])
    let engine = EngineCore(
        scheduler: scheduler,
        config: ContinuousBatchingConfig(stepInterval: 0.0005, yieldInterval: 1))
    engine.start()
    defer { engine.stop() }
    // Same measurement boundary as runV2Cell: per-row gpuPeak covers only
    // this cell's measured requests.
    MLX.Memory.peakMemory = 0
    let results = await withTaskGroup(of: (Int, RunResult).self) { group in
        for (i, length) in promptLengths.enumerated() {
            let prompt = syntheticPrompt(
                length: length, seed: 0xFACE &+ UInt64(i), vocabSize: vocabSize)
            group.addTask { @Sendable in
                let r = await runLegacyRequest(
                    engine: engine, id: "legacy-\(batch)-\(i)", promptTokens: prompt,
                    maxTokens: steps)
                return (i, r)
            }
        }
        var all: [(Int, RunResult)] = []
        for await entry in group { all.append(entry) }
        return all.sorted { $0.0 < $1.0 }.map(\.1)
    }
    return summarize(engine: label, batch: batch, promptMix: promptLengths, results: results)
}

// MARK: - Correctness

struct CorrectnessOutcome: Sendable {
    var name: String
    var pass: Bool
    var detail: String
}

func firstDivergence(_ a: [Int], _ b: [Int]) -> Int? {
    for i in 0 ..< min(a.count, b.count) where a[i] != b[i] { return i }
    return a.count == b.count ? nil : min(a.count, b.count)
}

/// Repetition-loop heuristic: does any 8-gram appear ≥ 4 times in the tail?
func hasRepetitionLoop(_ tokens: [Int]) -> Bool {
    guard tokens.count >= 32 else { return false }
    let tail = Array(tokens.suffix(96))
    var counts: [ArraySlice<Int>: Int] = [:]
    for i in 0 ... (tail.count - 8) {
        counts[tail[i ..< (i + 8)], default: 0] += 1
    }
    return counts.values.max() ?? 0 >= 4
}

func runCorrectness(
    context: ModelContext, hooks: V2ModelHooks, kvBytes: Int,
    chatTokens: (String) -> [Int], log: (String) -> Void
) async -> [CorrectnessOutcome] {
    var outcomes: [CorrectnessOutcome] = []
    let eos = context.configuration.eosTokenIds
    let defaultConfig = CBv2SchedulerConfig(
        maxConcurrentRequests: 4, maxBatchedTokensPerStep: 2048, prefillChunkSize: 512,
        maxWaiting: 16)

    func freshEngine(_ config: CBv2SchedulerConfig? = nil) throws -> EngineV2 {
        try makeV2Engine(
            context: context, hooks: hooks, backend: .contiguous,
            schedulerConfig: config ?? defaultConfig, kvBytes: kvBytes)
    }

    let target = chatTokens("Explain why the sky is blue in two sentences.")
    let shortNeighbor = chatTokens("List three prime numbers.")
    let longText = String(
        repeating: "The quick brown fox jumps over the lazy dog while the seasoned cartographer "
            + "annotates ancient maps with meticulous margin notes about tides, trade winds, "
            + "and the slow drift of continents across geological epochs. ",
        count: 40)
    let longNeighbor = chatTokens(
        "Summarize the following passage in one sentence: " + longText)
    log(
        "prompt tokens: target=\(target.count) short=\(shortNeighbor.count) "
            + "long=\(longNeighbor.count) eos=\(eos.sorted())")

    // 1a. B=1 sanity ------------------------------------------------------
    do {
        let engine = try freshEngine()
        let r = try await runV2Request(
            engine: engine, id: 1, promptTokens: target, maxTokens: 200, stopTokens: eos)
        await engine.shutdown()
        let stopped = r.finish == "stop"
        let looped = hasRepetitionLoop(r.tokens)
        let pass = stopped && !looped && !r.text.isEmpty
        log("[b1-sanity] finish=\(r.finish) tokens=\(r.tokens.count) loop=\(looped)")
        log("[b1-sanity] text: \(r.text)")
        outcomes.append(
            CorrectnessOutcome(
                name: "b1-sanity", pass: pass,
                detail:
                    "finish=\(r.finish) completion=\(r.tokens.count) repetitionLoop=\(looped)"))
    } catch {
        outcomes.append(
            CorrectnessOutcome(name: "b1-sanity", pass: false, detail: "error: \(error)"))
    }

    // 1b. Batch-composition invariance ------------------------------------
    var soloTokens: [Int] = []
    do {
        // (i) solo — run twice on fresh engines: same-composition
        // determinism is the precondition for any invariance claim.
        let engine = try freshEngine()
        let solo = try await runV2Request(
            engine: engine, id: 10, promptTokens: target, maxTokens: 64, stopTokens: eos,
            topLogprobs: 4)
        await engine.shutdown()
        let engine2 = try freshEngine()
        let solo2 = try await runV2Request(
            engine: engine2, id: 11, promptTokens: target, maxTokens: 64, stopTokens: eos,
            topLogprobs: 4)
        await engine2.shutdown()
        soloTokens = solo.tokens
        let soloRepeatDiv = firstDivergence(solo.tokens, solo2.tokens)
        log("[invariance] solo tokens=\(solo.tokens.count) finish=\(solo.finish)")
        log("[invariance] solo-repeat divergence=\(String(describing: soloRepeatDiv))")
        log("[invariance] solo text: \(solo.text)")

        // (ii) B=3 burst with short + long neighbors
        let burstEngine = try freshEngine()
        let burst = try await withThrowingTaskGroup(of: (Int, RunResult).self) { group in
            let submissions: [(Int, [Int], Int)] = [
                (0, shortNeighbor, 96), (1, longNeighbor, 96), (2, target, 64),
            ]
            for (i, prompt, budget) in submissions {
                group.addTask { @Sendable in
                    (
                        i,
                        try await runV2Request(
                            engine: burstEngine, id: UInt64(20 + i), promptTokens: prompt,
                            maxTokens: budget, stopTokens: eos, topLogprobs: 4)
                    )
                }
            }
            var all: [(Int, RunResult)] = []
            for try await entry in group { all.append(entry) }
            return all.sorted { $0.0 < $1.0 }.map(\.1)
        }
        await burstEngine.shutdown()
        let burstTarget = burst[2]
        let burstDiv = firstDivergence(soloTokens, burstTarget.tokens)
        log("[invariance] burst target tokens=\(burstTarget.tokens.count) divergence=\(String(describing: burstDiv))")
        if let d = burstDiv {
            log("[invariance] " + logprobDetail(solo, at: d, label: "solo @\(d)"))
            log("[invariance] " + logprobDetail(burstTarget, at: d, label: "burst @\(d)"))
        }

        // (iii) join mid-stream: neighbors decoding first, then the target.
        let joinEngine = try freshEngine()
        let c1 = TokenCounter()
        let c2 = TokenCounter()
        let n1Submitted = CFAbsoluteTimeGetCurrent()
        let n1Stream = try joinEngine.submit(
            CBv2Request(
                id: CBv2RequestID(30), promptTokens: shortNeighbor,
                sampling: CBv2SamplingParams(temperature: 0), maxTokens: 192, stopTokens: []))
        let n2Stream = try joinEngine.submit(
            CBv2Request(
                id: CBv2RequestID(31), promptTokens: longNeighbor,
                sampling: CBv2SamplingParams(temperature: 0), maxTokens: 192, stopTokens: []))
        let n1Task = Task { await collectV2(n1Stream, submittedAt: n1Submitted, counter: c1) }
        let n2Task = Task { await collectV2(n2Stream, submittedAt: n1Submitted, counter: c2) }
        let waitStart = CFAbsoluteTimeGetCurrent()
        while true {
            let (n1, n2) = await (c1.count, c2.count)
            if n1 >= 5 && n2 >= 5 { break }
            if CFAbsoluteTimeGetCurrent() - waitStart > 120 { break }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        let midStarted = await (c1.count, c2.count)
        let join = try await runV2Request(
            engine: joinEngine, id: 32, promptTokens: target, maxTokens: 64, stopTokens: eos,
            topLogprobs: 4)
        joinEngine.cancel(CBv2RequestID(30))
        joinEngine.cancel(CBv2RequestID(31))
        _ = await n1Task.value
        _ = await n2Task.value
        await joinEngine.shutdown()
        let joinDiv = firstDivergence(soloTokens, join.tokens)
        log(
            "[invariance] mid-join: neighbors had \(midStarted) tokens at join; "
                + "target tokens=\(join.tokens.count) divergence=\(String(describing: joinDiv))")
        if let d = joinDiv {
            log("[invariance] " + logprobDetail(solo, at: d, label: "solo @\(d)"))
            log("[invariance] " + logprobDetail(join, at: d, label: "join @\(d)"))
        }

        // Neighbor-content invariance at fixed B: burst and mid-join both
        // decode the target as one row of a 3-row batch but with DIFFERENT
        // neighbor progress/content. Token-equality here is the engine's
        // cross-request isolation guarantee; solo-vs-batched inequality is
        // batch-size GEMM numerics (kernel config keyed on B), not bleed.
        let neighborDiv = firstDivergence(burstTarget.tokens, join.tokens)
        log(
            "[invariance] neighbor-invariance (burst vs mid-join at B=3): "
                + "divergence=\(String(describing: neighborDiv))")

        let pass =
            soloRepeatDiv == nil && burstDiv == nil && joinDiv == nil && !soloTokens.isEmpty
        var detail = "solo=\(soloTokens.count) tok"
        detail +=
            soloRepeatDiv == nil
            ? "; solo-repeat deterministic"
            : "; SOLO-REPEAT NONDETERMINISTIC at index \(soloRepeatDiv!)"
        detail +=
            neighborDiv == nil
            ? "; neighbor-invariant at B=3 (burst == mid-join)"
            : "; NEIGHBOR-DEPENDENT at B=3, index \(neighborDiv!)"
        if let d = burstDiv {
            detail +=
                "; BURST DIVERGES from solo at index \(d) (solo=\(soloTokens[safe: d] ?? -1) "
                + "burst=\(burstTarget.tokens[safe: d] ?? -1))"
        } else {
            detail += "; burst identical to solo"
        }
        if let d = joinDiv {
            detail +=
                "; MID-JOIN DIVERGES from solo at index \(d) (solo=\(soloTokens[safe: d] ?? -1) "
                + "join=\(join.tokens[safe: d] ?? -1))"
        } else {
            detail += "; mid-join identical to solo"
        }
        outcomes.append(CorrectnessOutcome(name: "invariance", pass: pass, detail: detail))
    } catch {
        outcomes.append(
            CorrectnessOutcome(name: "invariance", pass: false, detail: "error: \(error)"))
    }

    // 1c. Chunked-prefill invariance ---------------------------------------
    do {
        let longPrompt = longNeighbor
        let chunkedEngine = try freshEngine()  // prefillChunkSize 512
        let chunked = try await runV2Request(
            engine: chunkedEngine, id: 40, promptTokens: longPrompt, maxTokens: 64,
            stopTokens: eos, topLogprobs: 4)
        await chunkedEngine.shutdown()

        let unchunkedEngine = try freshEngine(
            CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 8192,
                prefillChunkSize: 8192, maxWaiting: 16))
        let unchunked = try await runV2Request(
            engine: unchunkedEngine, id: 41, promptTokens: longPrompt, maxTokens: 64,
            stopTokens: eos, topLogprobs: 4)
        await unchunkedEngine.shutdown()

        let div = firstDivergence(chunked.tokens, unchunked.tokens)
        log(
            "[chunked-prefill] chunked=\(chunked.tokens.count) tok "
                + "unchunked=\(unchunked.tokens.count) tok divergence=\(String(describing: div))")
        if let d = div {
            log("[chunked-prefill] " + logprobDetail(chunked, at: d, label: "chunked @\(d)"))
            log("[chunked-prefill] " + logprobDetail(unchunked, at: d, label: "unchunked @\(d)"))
        }
        log("[chunked-prefill] chunked text: \(chunked.text)")
        let detail: String
        if let d = div {
            detail =
                "DIVERGES at index \(d) (chunked=\(chunked.tokens[safe: d] ?? -1) "
                + "unchunked=\(unchunked.tokens[safe: d] ?? -1)); "
                + "prompt=\(longPrompt.count) tok"
        } else {
            detail = "token-exact over \(chunked.tokens.count) tokens (prompt=\(longPrompt.count))"
        }
        outcomes.append(
            CorrectnessOutcome(name: "chunked-prefill", pass: div == nil, detail: detail))
    } catch {
        outcomes.append(
            CorrectnessOutcome(name: "chunked-prefill", pass: false, detail: "error: \(error)"))
    }

    // 1d. Compiled-decode parity + invariance ------------------------------
    // The compiled [B,1] path must reproduce eager v2 greedy tokens on real
    // weights. Full-buffer+mask SDPA vs sliced+maskless SDPA can reorder
    // reductions, so a divergence AT A NEAR-TIE (top1−top2 gap ≈ 0) is
    // kernel numerics, not an engine bug — reported distinctly.
    do {
        func compiledEngine() throws -> EngineV2 {
            try makeV2Engine(
                context: context, hooks: hooks, backend: .contiguous,
                schedulerConfig: defaultConfig, kvBytes: kvBytes, compiledDecode: true)
        }

        let eagerEngine = try freshEngine()
        let eagerSolo = try await runV2Request(
            engine: eagerEngine, id: 50, promptTokens: target, maxTokens: 64,
            stopTokens: eos, topLogprobs: 4)
        await eagerEngine.shutdown()

        let engine = try compiledEngine()
        let compiledSolo = try await runV2Request(
            engine: engine, id: 51, promptTokens: target, maxTokens: 64,
            stopTokens: eos, topLogprobs: 4)
        let stats = engine.compiledDecodeStats
        await engine.shutdown()

        let soloDiv = firstDivergence(eagerSolo.tokens, compiledSolo.tokens)
        let compiledSteps = stats?.compiledSteps ?? 0
        let warmup = (stats?.warmup ?? [])
            .map { String(format: "b%d=%.2fs", $0.bucket, $0.seconds) }
            .joined(separator: " ")
        log(
            "[compiled] solo divergence=\(String(describing: soloDiv)) "
                + "compiledSteps=\(compiledSteps) fallbacks=\(stats?.fallbacks ?? [:]) "
                + "warmup: \(warmup)"
                + (stats?.disabledReason.map { " DISABLED: \($0)" } ?? ""))
        if let d = soloDiv {
            log("[compiled] " + logprobDetail(eagerSolo, at: d, label: "eager @\(d)"))
            log("[compiled] " + logprobDetail(compiledSolo, at: d, label: "compiled @\(d)"))
        }
        log("[compiled] text: \(compiledSolo.text)")

        // Near-tie tolerance for the eager-vs-compiled cross-path check.
        var nearTie = false
        if let d = soloDiv, let lp = eagerSolo.logprobs[safe: d], lp.topLogprobs.count >= 2 {
            nearTie = abs(lp.topLogprobs[0].logprob - lp.topLogprobs[1].logprob) < 0.005
        }
        let soloPass = compiledSteps > 0 && (soloDiv == nil || nearTie)
        var detail =
            "compiledSteps=\(compiledSteps)"
            + (stats?.disabledReason.map { "; DISABLED: \($0)" } ?? "")
        if let d = soloDiv {
            detail += nearTie
                ? "; diverges from eager at index \(d) at a NEAR-TIE (kernel numerics)"
                : "; DIVERGES from eager at index \(d)"
        } else {
            detail += "; token-exact vs eager over \(compiledSolo.tokens.count) tokens"
        }
        outcomes.append(
            CorrectnessOutcome(name: "compiled-parity", pass: soloPass, detail: detail))

        // Same-mode invariance: compiled solo vs compiled B=3 burst target
        // (per-lane attention ⇒ neighbors must not change the target row).
        let burstEngine = try compiledEngine()
        let burst = try await withThrowingTaskGroup(of: (Int, RunResult).self) { group in
            let submissions: [(Int, [Int], Int)] = [
                (0, shortNeighbor, 96), (1, longNeighbor, 96), (2, target, 64),
            ]
            for (i, prompt, budget) in submissions {
                group.addTask { @Sendable in
                    (
                        i,
                        try await runV2Request(
                            engine: burstEngine, id: UInt64(60 + i), promptTokens: prompt,
                            maxTokens: budget, stopTokens: eos, topLogprobs: 4)
                    )
                }
            }
            var all: [(Int, RunResult)] = []
            for try await entry in group { all.append(entry) }
            return all.sorted { $0.0 < $1.0 }.map(\.1)
        }
        let burstStats = burstEngine.compiledDecodeStats
        await burstEngine.shutdown()
        let burstDiv = firstDivergence(compiledSolo.tokens, burst[2].tokens)
        log(
            "[compiled] burst-vs-solo divergence=\(String(describing: burstDiv)) "
                + "compiledSteps=\(burstStats?.compiledSteps ?? 0) "
                + "fallbacks=\(burstStats?.fallbacks ?? [:])")
        if let d = burstDiv {
            log("[compiled] " + logprobDetail(compiledSolo, at: d, label: "solo @\(d)"))
            log("[compiled] " + logprobDetail(burst[2], at: d, label: "burst @\(d)"))
        }
        outcomes.append(
            CorrectnessOutcome(
                name: "compiled-invariance",
                pass: burstDiv == nil && (burstStats?.compiledSteps ?? 0) > 0,
                detail: burstDiv == nil
                    ? "compiled burst identical to compiled solo "
                        + "(compiledSteps=\(burstStats?.compiledSteps ?? 0))"
                    : "COMPILED BURST DIVERGES from compiled solo at index \(burstDiv!)"))
    } catch {
        outcomes.append(
            CorrectnessOutcome(name: "compiled-parity", pass: false, detail: "error: \(error)"))
    }

    return outcomes
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

// MARK: - Entry point

func sysctlString(_ name: String) -> String {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    guard size > 0 else { return "unknown" }
    var buffer = [UInt8](repeating: 0, count: size)
    sysctlbyname(name, &buffer, &size, nil, 0)
    return String(decoding: buffer.prefix(while: { $0 != 0 }), as: UTF8.self)
}

/// 1-minute load average — stamped into the report header so a benchmark
/// taken on a contended host is visibly suspect. A busy host (parallel
/// swift build, a serving provider) halves eager decode TPS and skews
/// engine comparisons; that is exactly how the first GPT-OSS report came
/// to show v2 26 TPS vs legacy 35 when a clean host shows 107 vs 79.
func loadAverage1m() -> Double {
    var loads = [Double](repeating: 0, count: 3)
    guard getloadavg(&loads, 3) >= 1 else { return -1 }
    return loads[0]
}

/// Contention sentinel for the report header: 1m load average plus whether
/// any darkbloom provider process is alive (the usual GPU/CPU competitor on
/// dev machines).
func hostContentionSummary() -> (line: String, contended: Bool) {
    let load = loadAverage1m()
    let cores = ProcessInfo.processInfo.processorCount
    let task = Process()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
    // -x (exact process name): -f would false-positive on any shell whose
    // command line merely contains "DARKBLOOM_COMPILED_DECODE=…" — including
    // the one that launched this benchmark.
    task.arguments = ["-x", "darkbloom"]
    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = Pipe()
    var darkbloom = ""
    if (try? task.run()) != nil {
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()
        darkbloom = String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    let contended = load > Double(cores) / 2 || !darkbloom.isEmpty
    var line = String(format: "load avg (1m) %.1f / %d cores", load, cores)
    line += darkbloom.isEmpty ? "; no darkbloom process" : "; DARKBLOOM RUNNING"
    if contended { line += " — **HOST CONTENDED, numbers suspect**" }
    return (line, contended)
}

@main
struct BenchCBv2RealModel {
    static func main() async {
        // Unbuffered stdout: a Metal trap must not eat buffered results.
        setvbuf(stdout, nil, _IONBF, 0)
        var modelPath: String?
        var mode = "all"
        var engines = ["legacy", "v2", "v2-compiled", "v2-paged"]
        var batches = [1, 2, 4]
        var steps = 128
        var label = ""
        var outPath: String?
        var kvBytes = 16 << 30

        var args = Array(CommandLine.arguments.dropFirst())
        while !args.isEmpty {
            let arg = args.removeFirst()
            switch arg {
            case "--model": modelPath = args.isEmpty ? nil : args.removeFirst()
            case "--mode": mode = args.isEmpty ? mode : args.removeFirst()
            case "--engines":
                engines = args.isEmpty
                    ? engines : args.removeFirst().split(separator: ",").map(String.init)
            case "--batches":
                batches = args.isEmpty
                    ? batches : args.removeFirst().split(separator: ",").compactMap { Int($0) }
            case "--steps": steps = args.isEmpty ? steps : Int(args.removeFirst()) ?? steps
            case "--label": label = args.isEmpty ? label : args.removeFirst()
            case "--kv-gb":
                kvBytes = (args.isEmpty ? 16 : Int(args.removeFirst()) ?? 16) << 30
            case "--kv-capacity":
                benchCompiledKVCapacity =
                    args.isEmpty ? benchCompiledKVCapacity
                    : Int(args.removeFirst()) ?? benchCompiledKVCapacity
            case "--out": outPath = args.isEmpty ? nil : args.removeFirst()
            default:
                print("usage: BenchCBv2 --model <dir> [--mode all|correctness|perf]")
                print("       [--engines legacy,v2,v2-compiled,v2-paged] [--batches 1,2,4]")
                print("       [--steps N] [--label tag] [--kv-gb N] [--out report.md]")
                exit(64)
            }
        }
        guard let modelPath else {
            print("error: --model is required")
            exit(64)
        }
        let directory = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            print("error: model directory does not exist: \(directory.path)")
            exit(66)
        }

        let compiled = CompiledDecode.isEnabled
        let legacyLabel = label.isEmpty ? (compiled ? "legacy-compiled" : "legacy") : label
        let contention = hostContentionSummary()
        print("== BenchCBv2RealModel ==")
        print("model: \(modelPath)")
        print("mode: \(mode)  engines: \(engines)  batches: \(batches)  steps: \(steps)")
        print("DARKBLOOM_COMPILED_DECODE resolved: \(compiled) (affects legacy engine only)")
        print("host: \(contention.line)")
        if contention.contended {
            print("WARNING: host is contended — eager decode is CPU-bound, results will be skewed")
        }

        do {
            let loadStart = CFAbsoluteTimeGetCurrent()
            // Explicitly the LLM factory: text-only benchmarking of checkpoints
            // that may carry a vision tower (see file header).
            let container = try await LLMModelFactory.shared.loadContainer(
                from: directory, using: #huggingFaceTokenizerLoader())
            print(String(
                format: "model loaded in %.1fs", CFAbsoluteTimeGetCurrent() - loadStart))

            let (modeCopy, enginesCopy, batchesCopy, stepsCopy, kvBytesCopy, legacyLabelCopy) =
                (mode, engines, batches, steps, kvBytes, legacyLabel)
            let report: String = try await container.perform {
                (context: ModelContext) async throws -> String in
                var out = ""
                func emit(_ line: String) {
                    print(line)
                    out += line + "\n"
                }

                guard let hooks = v2Hooks(for: context.model) else {
                    throw CBv2KVError.backendIneligible(
                        reason: "model type \(type(of: context.model)) has no CBv2 hooks")
                }
                emit("model class: \(type(of: context.model)); layers: \(hooks.layerKinds.count)")

                // Vocab probe ([1,1] cache-less forward) for synthetic prompts.
                let probe = context.model(MLXArray([Int32(0)]).reshaped(1, 1), cache: nil)
                let vocabSize = probe.dim(-1)
                emit("vocabSize=\(vocabSize)")

                // Chat prompt builder: BatchedEngine.buildPrompt runs the full
                // production template cascade (tokenizer template → external
                // chat_template.jinja → per-family fallback).
                let promptBuilder = BatchedEngine(context: context)
                let chatTokens: (String) -> [Int] = { question in
                    let prompt = promptBuilder.buildPrompt(
                        messages: [["role": "user", "content": question]])
                    return context.tokenizer.encode(text: prompt, addSpecialTokens: false)
                }

                // Correctness ------------------------------------------------
                if modeCopy == "all" || modeCopy == "correctness" {
                    emit("\n## Correctness (v2 contiguous, greedy)\n")
                    let outcomes = await runCorrectness(
                        context: context, hooks: hooks, kvBytes: kvBytesCopy,
                        chatTokens: chatTokens, log: { emit($0) })
                    emit("")
                    for o in outcomes {
                        emit("- **\(o.name)**: \(o.pass ? "PASS" : "FAIL") — \(o.detail)")
                    }
                }

                // Profile ----------------------------------------------------
                // Phase-timer decomposition of the B=1 decode step for the
                // legacy engine (eager) and v2 (contiguous). One warmup cell
                // per engine absorbs kernel-compile costs; the measured cell
                // then runs with CBv2StepProfiler enabled.
                if modeCopy == "profile" {
                    emit("\n## Decode-step profile (B=1, maxTokens \(stepsCopy))\n")
                    for engineName in enginesCopy where engineName != "v2-paged" {
                        switch engineName {
                        case "legacy":
                            _ = await runLegacyCell(
                                context: context, batch: 1, promptLengths: [100],
                                steps: 8, vocabSize: vocabSize, label: "warmup")
                            CBv2StepProfiler.reset()
                            CBv2StepProfiler.enabled = true
                            let cell = await runLegacyCell(
                                context: context, batch: 1, promptLengths: [500],
                                steps: stepsCopy, vocabSize: vocabSize,
                                label: legacyLabelCopy)
                            CBv2StepProfiler.enabled = false
                            emit("### legacy (\(legacyLabelCopy)) B=1 — decodeTPS \(String(format: "%.1f", cell.decodeTPSPerRequest))\n")
                            emit(CBv2StepProfiler.summaryTable())
                        case "v2":
                            _ = try? await runV2Cell(
                                context: context, hooks: hooks, backend: .contiguous,
                                batch: 1, promptLengths: [100], steps: 8,
                                vocabSize: vocabSize, kvBytes: kvBytesCopy)
                            CBv2StepProfiler.reset()
                            CBv2StepProfiler.enabled = true
                            let cell = try await runV2Cell(
                                context: context, hooks: hooks, backend: .contiguous,
                                batch: 1, promptLengths: [500], steps: stepsCopy,
                                vocabSize: vocabSize, kvBytes: kvBytesCopy)
                            CBv2StepProfiler.enabled = false
                            emit("### v2 (contiguous) B=1 — decodeTPS \(String(format: "%.1f", cell.decodeTPSPerRequest))\n")
                            emit(CBv2StepProfiler.summaryTable())
                        default:
                            continue
                        }
                    }
                }

                // Perf -------------------------------------------------------
                if modeCopy == "all" || modeCopy == "perf" {
                    emit("\n## Performance (maxTokens \(stepsCopy))\n")
                    emit(CellResult.markdownHeader)
                    var details: [String] = []
                    for engineName in enginesCopy {
                        // Warmup: absorb kernel compile / cache-shape costs.
                        let warmPrompt = [100]
                        switch engineName {
                        case "legacy":
                            _ = await runLegacyCell(
                                context: context, batch: 1, promptLengths: warmPrompt,
                                steps: 8, vocabSize: vocabSize, label: "warmup")
                        case "v2":
                            _ = try? await runV2Cell(
                                context: context, hooks: hooks, backend: .contiguous,
                                batch: 1, promptLengths: warmPrompt, steps: 8,
                                vocabSize: vocabSize, kvBytes: kvBytesCopy)
                        case "v2-compiled":
                            _ = try? await runV2Cell(
                                context: context, hooks: hooks, backend: .contiguous,
                                batch: 1, promptLengths: warmPrompt, steps: 8,
                                vocabSize: vocabSize, kvBytes: kvBytesCopy,
                                compiledDecode: true, emit: { emit($0) })
                        case "v2-paged":
                            do {
                                _ = try await runV2Cell(
                                    context: context, hooks: hooks, backend: .paged,
                                    batch: 1, promptLengths: warmPrompt, steps: 8,
                                    vocabSize: vocabSize, kvBytes: kvBytesCopy)
                            } catch {
                                emit("| v2-paged | - | - | skipped: \(error) | | | |")
                                continue
                            }
                        default:
                            emit("| \(engineName) | - | - | unknown engine | | | |")
                            continue
                        }

                        for batch in batchesCopy {
                            let mix = promptMix(batch: batch)
                            let cell: CellResult
                            switch engineName {
                            case "legacy":
                                cell = await runLegacyCell(
                                    context: context, batch: batch, promptLengths: mix,
                                    steps: stepsCopy, vocabSize: vocabSize,
                                    label: legacyLabelCopy)
                            case "v2":
                                cell = try await runV2Cell(
                                    context: context, hooks: hooks, backend: .contiguous,
                                    batch: batch, promptLengths: mix, steps: stepsCopy,
                                    vocabSize: vocabSize, kvBytes: kvBytesCopy)
                            case "v2-compiled":
                                cell = try await runV2Cell(
                                    context: context, hooks: hooks, backend: .contiguous,
                                    batch: batch, promptLengths: mix, steps: stepsCopy,
                                    vocabSize: vocabSize, kvBytes: kvBytesCopy,
                                    compiledDecode: true,
                                    emit: { details.append($0) })
                            default:
                                cell = try await runV2Cell(
                                    context: context, hooks: hooks, backend: .paged,
                                    batch: batch, promptLengths: mix, steps: stepsCopy,
                                    vocabSize: vocabSize, kvBytes: kvBytesCopy)
                            }
                            emit(cell.markdownRow)
                            emit(String(
                                format: "    [mem after %@ B=%d] gpuActive=%.2f GiB gpuPeak=%.2f GiB",
                                cell.engine, batch,
                                Double(MLX.GPU.activeMemory) / Double(1 << 30),
                                Double(MLX.GPU.peakMemory) / Double(1 << 30)))
                            details.append(
                                "  \(cell.engine) B=\(batch):\n"
                                    + cell.perRequest.joined(separator: "\n"))
                        }
                    }
                    emit("\nPer-request detail:\n" + details.joined(separator: "\n"))
                }
                return out
            }

            let ram = ProcessInfo.processInfo.physicalMemory / (1 << 30)
            let os = ProcessInfo.processInfo.operatingSystemVersionString
            let header = """
                # BenchCBv2RealModel report

                | | |
                |---|---|
                | Model | \(modelPath) |
                | Chip | \(sysctlString("machdep.cpu.brand_string")) |
                | RAM | \(ram) GB |
                | OS | \(os) |
                | Compiled decode (legacy) | \(compiled) |
                | Host at start | \(contention.line) |
                | Date | \(ISO8601DateFormatter().string(from: Date())) |


                """
            if let outPath {
                try (header + report).write(
                    to: URL(fileURLWithPath: outPath), atomically: true, encoding: .utf8)
                print("report written to \(outPath)")
            }
            print("\nDONE")
        } catch {
            print("error: \(error)")
            exit(70)
        }
    }
}
