// CBv2Benchmark — WS-G benchmark harness for the ContinuousBatchingV2 engine
// (spec: docs/engine-v2/specs/G-harness.md deliverable 5; report 12 item 18
// "the engine repo has no benchmark harness").
//
// Two modes:
//
//  1. Tiny mode (default, CI-runnable, NO downloads): a self-contained
//     2-layer transformer (one full-attention + one sliding-window(16) layer,
//     seeded random weights — the same architecture as the test fixture
//     `TinyTestModel` in Tests/MLXLMTests/CBv2Fixtures.swift) driven through
//     a v2-style step loop: per-request KV, per-row SDPA, chunked [1, chunk]
//     prefill, rectangular [B, 1] greedy decode. Measures the v2 execution
//     SHAPE end to end; integration re-baselines these numbers on the real
//     EngineV2.
//
//  2. Real-model mode (`--model <path>`, manual runs — Gemma-4 / GPT-OSS
//     etc.): loads local weights and measures the LEGACY continuous-batching
//     engine (Scheduler → EngineCore) as the baseline the v2 engine must
//     beat. Integration adds an `--engine v2` switch that binds the same
//     scenarios to `CBv2Engine` once WS-B's engine merges.
//
// Matrix: B ∈ {1, 2, 4} × prompt ∈ {short 64, long 2048} tokens.
// Reported per cell (markdown table):
//   TPS/request, aggregate TPS, TTFT p50, ITL p50/p99, and — tiny mode
//   only — decode-step CPU μs vs GPU μs, split at the asyncEval boundary
//   (CPU = graph build + asyncEval enqueue; GPU = wait for the sampled-token
//   readback).
//
// Usage:
//   swift run -c release CBv2Benchmark [--steps N] [--out path.md]
//   swift run -c release CBv2Benchmark --model /path/to/model \
//       [--batch 1,2,4] [--prompts 64,2048] [--steps N] [--out path.md]

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXRandom
import MLXVLM

// MARK: - Deterministic PRNG (mirrors the test fixture's SplitMix64)

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

func benchPrompt(length: Int, seed: UInt64, vocabSize: Int) -> [Int] {
    var rng = BenchRNG(seed: seed)
    return (0 ..< length).map { _ in Int.random(in: 0 ..< vocabSize, using: &rng) }
}

// MARK: - Tiny model (same architecture as Tests/MLXLMTests/CBv2Fixtures.swift)

struct TinyBenchConfig {
    var hiddenSize = 64
    var queryHeads = 4
    var kvHeads = 2
    var headDim = 16
    var vocabSize = 128
    var windowSize = 16
    var mlpSize = 128
    var ropeBase: Float = 10_000
    var rmsNormEps: Float = 1e-5
    var scale: Float { 1.0 / Float(headDim).squareRoot() }
}

/// Per-request per-layer KV state: owns its storage and its absolute position
/// counter. Windowed eviction keeps the RECENT end (v2 semantics — no left
/// padding, no shared frontier, no masks over shared buffers).
final class BenchSequenceKV {
    /// nil ⇒ full attention (retain everything).
    let window: Int?
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var absoluteOffset = 0

    init(window: Int?) { self.window = window }

    var retainedCount: Int { keys?.dim(2) ?? 0 }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        if var k = keys, var v = values {
            if let window, n > 1 {
                // Prefill chunk: retain window-1 old entries pre-append so
                // every chunk query still sees its full window (mask enforces
                // per-query visibility).
                let trim = k.dim(2) - (window - 1)
                if trim > 0 {
                    k = k[.ellipsis, trim..., 0...]
                    v = v[.ellipsis, trim..., 0...]
                }
                keys = concatenated([k, newKeys], axis: 2)
                values = concatenated([v, newValues], axis: 2)
            } else {
                var mk = concatenated([k, newKeys], axis: 2)
                var mv = concatenated([v, newValues], axis: 2)
                if let window {
                    let trim = mk.dim(2) - window
                    if trim > 0 {
                        mk = mk[.ellipsis, trim..., 0...]
                        mv = mv[.ellipsis, trim..., 0...]
                    }
                }
                keys = mk
                values = mv
            }
        } else {
            keys = newKeys
            values = newValues
        }
        absoluteOffset += n
        return (keys!, values!)
    }

    /// The lazily-built K/V chain, for asyncEval scheduling after prefill.
    var state: [MLXArray] { [keys, values].compactMap { $0 } }
}

final class TinyBenchAttention: Module {
    let config: TinyBenchConfig
    let window: Int?
    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear
    let rope: RoPE

    init(config: TinyBenchConfig, window: Int?) {
        self.config = config
        self.window = window
        self.qProj = Linear(config.hiddenSize, config.queryHeads * config.headDim, bias: false)
        self.kProj = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        self.vProj = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        self.oProj = Linear(config.queryHeads * config.headDim, config.hiddenSize, bias: false)
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeBase)
        super.init()
    }

    /// v2-style attention: per-row absolute RoPE offsets (captured BEFORE the
    /// KV update advances them), per-row KV update + SDPA. Decode ([B, 1]) is
    /// mask-free; prefill ([1, chunk]) uses a causal(∧window) boolean mask
    /// against the row's own retained storage.
    func forward(_ x: MLXArray, rows: [BenchSequenceKV]) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        precondition(B == rows.count && (B == 1 || L == 1))

        var q = qProj(x).reshaped(B, L, -1, config.headDim).swappedAxes(1, 2)
        var k = kProj(x).reshaped(B, L, -1, config.headDim).swappedAxes(1, 2)
        let v = vProj(x).reshaped(B, L, -1, config.headDim).swappedAxes(1, 2)

        let offsets = MLXArray(rows.map { Int32($0.absoluteOffset) })
        q = rope(q, offset: offsets)
        k = rope(k, offset: offsets)

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for (i, row) in rows.enumerated() {
            let rq = B == 1 ? q : q[i ..< (i + 1)]
            let rk = B == 1 ? k : k[i ..< (i + 1)]
            let rv = B == 1 ? v : v[i ..< (i + 1)]
            let (keys, values) = row.update(keys: rk, values: rv)
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                L == 1
                ? .none
                : .array(createCausalMask(n: L, offset: keys.dim(2) - L, windowSize: window))
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: rq, keys: keys, values: values,
                    scale: config.scale, mask: mask))
        }
        let out = B == 1 ? outputs[0] : concatenated(outputs, axis: 0)
        return oProj(out.swappedAxes(1, 2).reshaped(B, L, -1))
    }
}

final class TinyBenchBlock: Module {
    let attention: TinyBenchAttention
    let attnNorm: RMSNorm
    let mlpNorm: RMSNorm
    let mlpUp: Linear
    let mlpDown: Linear

    init(config: TinyBenchConfig, window: Int?) {
        self.attention = TinyBenchAttention(config: config, window: window)
        self.attnNorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.mlpNorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.mlpUp = Linear(config.hiddenSize, config.mlpSize, bias: false)
        self.mlpDown = Linear(config.mlpSize, config.hiddenSize, bias: false)
        super.init()
    }

    func forward(_ x: MLXArray, rows: [BenchSequenceKV]) -> MLXArray {
        var h = x + attention.forward(attnNorm(x), rows: rows)
        h = h + mlpDown(silu(mlpUp(mlpNorm(h))))
        return h
    }
}

/// Layer 0 = full attention, layer 1 = sliding window — mirrors the fixture.
final class TinyBenchModel: Module {
    let config: TinyBenchConfig
    let embed: Embedding
    let blocks: [TinyBenchBlock]
    let finalNorm: RMSNorm
    let lmHead: Linear

    var layerWindows: [Int?] { [nil, config.windowSize] }

    static func make(seed: UInt64 = 0xC0FFEE) -> TinyBenchModel {
        MLXRandom.seed(seed)
        let model = TinyBenchModel(config: TinyBenchConfig())
        eval(model)
        return model
    }

    init(config: TinyBenchConfig) {
        self.config = config
        self.embed = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.blocks = [
            TinyBenchBlock(config: config, window: nil),
            TinyBenchBlock(config: config, window: config.windowSize),
        ]
        self.finalNorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.lmHead = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    /// kv[layer][row] — row order fixed for the scenario.
    func forward(tokens: MLXArray, kv: [[BenchSequenceKV]]) -> MLXArray {
        var h = embed(tokens)
        for (i, block) in blocks.enumerated() {
            h = block.forward(h, rows: kv[i])
        }
        return lmHead(finalNorm(h))
    }
}

// MARK: - Measurement plumbing

func percentile(_ sorted: [Double], _ q: Double) -> Double {
    guard !sorted.isEmpty else { return 0 }
    let rank = q * Double(sorted.count - 1)
    let lo = Int(rank.rounded(.down)), hi = Int(rank.rounded(.up))
    if lo == hi { return sorted[lo] }
    let w = rank - Double(lo)
    return sorted[lo] * (1 - w) + sorted[hi] * w
}

struct CellResult: Sendable {
    var mode: String
    var batch: Int
    var promptLength: Int
    var decodeSteps: Int
    var tpsPerRequest: Double
    var aggregateTPS: Double
    var ttftP50Ms: Double
    var itlP50Ms: Double
    var itlP99Ms: Double
    /// asyncEval-boundary split; nil for the legacy engine (opaque loop).
    var stepCPUMicros: Double?
    var stepGPUMicros: Double?

    var markdownRow: String {
        let cpu = stepCPUMicros.map { String(format: "%.0f", $0) } ?? "n/a"
        let gpu = stepGPUMicros.map { String(format: "%.0f", $0) } ?? "n/a"
        return String(
            format: "| %@ | %d | %d | %.1f | %.1f | %.1f | %.2f | %.2f | %@ | %@ |",
            mode, batch, promptLength, tpsPerRequest, aggregateTPS,
            ttftP50Ms, itlP50Ms, itlP99Ms, cpu, gpu)
    }

    static let markdownHeader = """
        | mode | B | prompt | TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) | ITL p99 (ms) | step CPU (μs) | step GPU (μs) |
        |---|---|---|---|---|---|---|---|---|---|
        """
}

// MARK: - Tiny-mode scenario (v2-style step loop, instrumented)

/// Burst arrival: B requests submitted at t=0, prefilled per request in
/// [1, chunk] chunks, then decoded together in rectangular [B, 1] greedy
/// steps. TTFT includes the queue effect of the other rows' prefills —
/// that is the batched-arrival latency a provider actually serves.
func runTinyCell(
    model: TinyBenchModel, batch: Int, promptLength: Int, decodeSteps: Int,
    prefillChunk: Int = 512
) -> CellResult {
    let config = model.config
    let prompts = (0 ..< batch).map {
        benchPrompt(length: promptLength, seed: 0xBEEF &+ UInt64($0), vocabSize: config.vocabSize)
    }

    // Per-request, per-layer KV (row-major bookkeeping kept layer-major for
    // the forward call).
    var kv: [[BenchSequenceKV]] = model.layerWindows.map { _ in [] }
    var lastTokens = [Int32](repeating: 0, count: batch)

    let started = CFAbsoluteTimeGetCurrent()

    // Prefill each request against ITS OWN caches ([1, chunk] — batchmates
    // untouched; batch-composition invariance by construction).
    for (r, prompt) in prompts.enumerated() {
        let rowKV = model.layerWindows.map { BenchSequenceKV(window: $0) }
        for (layer, state) in rowKV.enumerated() { kv[layer].append(state) }
        let prefillTokens = prompt.dropLast()
        var index = prefillTokens.startIndex
        while index < prefillTokens.endIndex {
            let end = min(index + prefillChunk, prefillTokens.endIndex)
            let chunk = Array(prefillTokens[index ..< end])
            let tokens = MLXArray(chunk.map { Int32($0) }).reshaped(1, chunk.count)
            let soloRow = rowKV.map { [$0] }
            _ = model.forward(tokens: tokens, kv: soloRow)
            index = end
        }
        // Enqueue the prefill graph now; the first decode step's readback
        // completes it (counted in TTFT, not in steady-state ITL).
        asyncEval(rowKV.flatMap(\.state))
        lastTokens[r] = Int32(prompt.last!)
    }

    // Rectangular [B, 1] greedy decode, instrumented at the asyncEval
    // boundary: CPU = graph build + enqueue, GPU = sampled-token readback.
    var stepCPU: [Double] = []
    var stepGPU: [Double] = []
    var stepDone: [Double] = []  // absolute completion time per step
    stepCPU.reserveCapacity(decodeSteps)
    stepGPU.reserveCapacity(decodeSteps)
    stepDone.reserveCapacity(decodeSteps)

    for _ in 0 ..< decodeSteps {
        let t0 = CFAbsoluteTimeGetCurrent()
        let tokens = MLXArray(lastTokens).reshaped(batch, 1)
        let logits = model.forward(tokens: tokens, kv: kv)
        let sampled = argMax(logits[0..., -1, 0...], axis: -1).asType(.int32)
        asyncEval(sampled)
        let t1 = CFAbsoluteTimeGetCurrent()
        lastTokens = sampled.asArray(Int32.self)  // boundary sync (one per step)
        let t2 = CFAbsoluteTimeGetCurrent()
        stepCPU.append((t1 - t0) * 1e6)
        stepGPU.append((t2 - t1) * 1e6)
        stepDone.append(t2)
    }

    let firstToken = stepDone[0]
    // Steady-state ITL: successive step completions after the first token
    // (step 0 carries the prefill drain and is TTFT, not ITL).
    let itls = zip(stepDone.dropFirst(), stepDone).map { ($0 - $1) * 1e3 }.sorted()
    let steadyTPS = Double(decodeSteps - 1) / (stepDone.last! - firstToken)

    return CellResult(
        mode: "tiny-v2", batch: batch, promptLength: promptLength, decodeSteps: decodeSteps,
        tpsPerRequest: steadyTPS,
        aggregateTPS: steadyTPS * Double(batch),
        ttftP50Ms: (firstToken - started) * 1e3,
        itlP50Ms: percentile(itls, 0.50),
        itlP99Ms: percentile(itls, 0.99),
        stepCPUMicros: percentile(stepCPU.sorted(), 0.50),
        stepGPUMicros: percentile(stepGPU.sorted(), 0.50)
    )
}

// MARK: - Real-model mode (LEGACY engine baseline; v2 binding at integration)

/// Minimal tokenizer for the Scheduler (prompts are pre-tokenized [Int]s;
/// only stop-string encoding would use this, and the benchmark sets none).
struct BenchPassthroughTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.split(separator: " ").compactMap { Int($0) }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }
    func convertTokenToId(_ token: String) -> Int? { Int(token) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

struct RequestTimings: Sendable {
    var ttftMs: Double
    var itlsMs: [Double]
    var tokens: Int
    var wallSeconds: Double
}

func runLegacyEngineCell(
    model: any LanguageModel, vocabSize: Int,
    batch: Int, promptLength: Int, decodeSteps: Int
) async -> CellResult {
    let scheduler = Scheduler(
        model: model,
        tokenizer: BenchPassthroughTokenizer(),
        config: SchedulerConfig(
            maxNumSeqs: batch,
            maxNumBatchedTokens: 4096,
            prefillStepSize: 512,
            streamInterval: 1
        ),
        eosTokenIds: []  // maxTokens governs — comparable across models
    )
    let engine = EngineCore(
        scheduler: scheduler,
        config: ContinuousBatchingConfig(stepInterval: 0.0005, yieldInterval: 1)
    )
    engine.start()
    defer { engine.stop() }

    let timings = await withTaskGroup(of: RequestTimings.self) { group in
        for r in 0 ..< batch {
            let prompt = benchPrompt(
                length: promptLength, seed: 0xFACE &+ UInt64(r), vocabSize: vocabSize)
            group.addTask { @Sendable in
                let submitted = CFAbsoluteTimeGetCurrent()
                let request = Request(
                    requestId: "bench-\(batch)x\(promptLength)-\(r)",
                    prompt: prompt as AnyHashable,
                    samplingParams: SamplingParams(maxTokens: decodeSteps, temperature: 0)
                )
                let requestId = await engine.addRequest(request)

                var ttftMs = 0.0
                var itlsMs: [Double] = []
                var lastArrival: Double?
                var tokenCount = 0
                for await chunk in engine.streamOutputs(requestId: requestId) {
                    let now = CFAbsoluteTimeGetCurrent()
                    if chunk.outputTokenIds.count > tokenCount {
                        if lastArrival == nil {
                            ttftMs = (now - submitted) * 1e3
                        } else {
                            itlsMs.append((now - lastArrival!) * 1e3)
                        }
                        lastArrival = now
                        tokenCount = chunk.outputTokenIds.count
                    }
                    if chunk.finished { break }
                }
                return RequestTimings(
                    ttftMs: ttftMs, itlsMs: itlsMs, tokens: tokenCount,
                    wallSeconds: CFAbsoluteTimeGetCurrent() - submitted)
            }
        }
        var all: [RequestTimings] = []
        for await t in group { all.append(t) }
        return all
    }

    let totalTokens = timings.reduce(0) { $0 + $1.tokens }
    let wall = timings.map(\.wallSeconds).max() ?? 1
    let itls = timings.flatMap(\.itlsMs).sorted()
    let ttfts = timings.map(\.ttftMs).sorted()
    let perRequestTPS = timings.map { $0.wallSeconds > 0 ? Double($0.tokens) / $0.wallSeconds : 0 }

    return CellResult(
        mode: "legacy", batch: batch, promptLength: promptLength, decodeSteps: decodeSteps,
        tpsPerRequest: perRequestTPS.reduce(0, +) / Double(max(1, perRequestTPS.count)),
        aggregateTPS: Double(totalTokens) / wall,
        ttftP50Ms: percentile(ttfts, 0.50),
        itlP50Ms: percentile(itls, 0.50),
        itlP99Ms: percentile(itls, 0.99),
        stepCPUMicros: nil,
        stepGPUMicros: nil
    )
}

// MARK: - Report

func sysctlString(_ name: String) -> String {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    guard size > 0 else { return "unknown" }
    var buffer = [UInt8](repeating: 0, count: size)
    sysctlbyname(name, &buffer, &size, nil, 0)
    return String(decoding: buffer.prefix(while: { $0 != 0 }), as: UTF8.self)
}

func environmentHeader(modelLabel: String, steps: Int) -> String {
    let formatter = ISO8601DateFormatter()
    let ram = ProcessInfo.processInfo.physicalMemory / (1 << 30)
    let os = ProcessInfo.processInfo.operatingSystemVersionString
    return """
        # CBv2 Benchmark

        | | |
        |---|---|
        | **Date** | \(formatter.string(from: Date())) |
        | **Chip** | \(sysctlString("machdep.cpu.brand_string")) |
        | **RAM** | \(ram) GB |
        | **OS** | \(os) |
        | **Model** | \(modelLabel) |
        | **Decode steps** | \(steps) |

        ## Results

        """
}

// MARK: - Entry point

@main
struct CBv2Benchmark {
    static func main() async {
        var modelPath: String?
        var batches = [1, 2, 4]
        var promptLengths = [64, 2048]
        var steps = 128
        var outPath: String?

        var args = Array(CommandLine.arguments.dropFirst())
        func intList(_ raw: String) -> [Int] {
            raw.split(separator: ",").compactMap { Int($0) }
        }
        while !args.isEmpty {
            let arg = args.removeFirst()
            switch arg {
            case "--model": modelPath = args.isEmpty ? nil : args.removeFirst()
            case "--batch": batches = args.isEmpty ? batches : intList(args.removeFirst())
            case "--prompts":
                promptLengths = args.isEmpty ? promptLengths : intList(args.removeFirst())
            case "--steps": steps = args.isEmpty ? steps : Int(args.removeFirst()) ?? steps
            case "--out": outPath = args.isEmpty ? nil : args.removeFirst()
            default:
                print("usage: CBv2Benchmark [--model <path>] [--batch 1,2,4]")
                print("       [--prompts 64,2048] [--steps N] [--out report.md]")
                exit(64)
            }
        }

        var results: [CellResult] = []
        var modelLabel = "TinyBenchModel (2-layer, full+window16, seeded — no downloads)"

        if let modelPath {
            modelLabel = modelPath
            let directory = URL(fileURLWithPath: modelPath)
            guard FileManager.default.fileExists(atPath: directory.path) else {
                print("error: model directory does not exist: \(directory.path)")
                exit(66)
            }
            do {
                let container = try await loadModelContainer(
                    from: directory, using: BenchTokenizerLoader())
                // Immutable copies for the @Sendable perform closure.
                let (promptLengths, batches, steps) = (promptLengths, batches, steps)
                results = try await container.perform {
                    (context: ModelContext) async throws -> [CellResult] in
                    // Vocab probe: one [1, 1] cache-less forward.
                    let probe = context.model(MLXArray([Int32(0)]).reshaped(1, 1), cache: nil)
                    let vocabSize = probe.dim(-1)
                    var cells: [CellResult] = []
                    // Warmup: absorb load/compile cost before measuring.
                    _ = await runLegacyEngineCell(
                        model: context.model, vocabSize: vocabSize,
                        batch: 1, promptLength: 32, decodeSteps: 8)
                    for promptLength in promptLengths {
                        for batch in batches {
                            let cell = await runLegacyEngineCell(
                                model: context.model, vocabSize: vocabSize,
                                batch: batch, promptLength: promptLength, decodeSteps: steps)
                            print("done: \(cell.markdownRow)")
                            cells.append(cell)
                        }
                    }
                    return cells
                }
            } catch {
                print("error: \(error)")
                exit(70)
            }
        } else {
            let model = TinyBenchModel.make()
            for promptLength in promptLengths {
                for batch in batches {
                    // Per-cell warmup: kernel compilation is shape-keyed, so
                    // warm the exact (B, prompt) shapes before measuring —
                    // otherwise the first cell's TTFT absorbs compile time.
                    _ = runTinyCell(
                        model: model, batch: batch, promptLength: promptLength,
                        decodeSteps: 8)
                    let cell = runTinyCell(
                        model: model, batch: batch, promptLength: promptLength,
                        decodeSteps: steps)
                    print("done: \(cell.markdownRow)")
                    results.append(cell)
                }
            }
        }

        var report = environmentHeader(modelLabel: modelLabel, steps: steps)
        report += CellResult.markdownHeader + "\n"
        for cell in results { report += cell.markdownRow + "\n" }
        report += """

            Notes:
            - tiny-v2 = v2-style step loop (per-request KV, per-row SDPA, [B,1] \
            greedy decode) on the seeded tiny model; CI-runnable, no downloads.
            - legacy = old continuous-batching engine (Scheduler → EngineCore), \
            real weights; the baseline the v2 engine must beat.
            - step CPU/GPU split is measured at the asyncEval boundary and only \
            defined for the tiny-v2 driver.
            - TTFT is burst-arrival: all B requests submitted at t=0.

            """

        print("\n" + report)
        if let outPath {
            do {
                try report.write(
                    to: URL(fileURLWithPath: outPath), atomically: true, encoding: .utf8)
                print("report written to \(outPath)")
            } catch {
                print("error writing report: \(error)")
                exit(74)
            }
        }
    }
}

/// Tokenizer loader for real-model mode: the benchmark never tokenizes text
/// (prompts are synthetic token ids), so any tokenizer satisfying the loader
/// suffices. Mirrors BenchmarkHelpers.NoOpTokenizerLoader without importing
/// the helper target.
struct BenchTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        BenchPassthroughTokenizer()
    }
}
