// PagedBackendBenchmark.swift
//
// Decode-attention microbenchmark for the paged backend (WS-C): paged
// Metal kernel vs v1-style per-row SDPA dispatch vs legacy dense-batch
// SDPA. Prints a markdown table with ms/step, achieved KV GB/s and
// aggregate decode tokens/s per attention layer.
//
// THE GATE (spec C-paged-backend §5): paged decode must beat v1 per-row
// dispatch at B >= 2 and stay within 15% of single-request SDPA at B = 1.
//
// Run via the env-gated test:
//   DARKBLOOM_CBV2_PAGED_BENCH=1 swift test --filter CBv2PagedBenchmarkTests
// Results are recorded in benchmarks/cbv2-paged-attention.md.

import Foundation
import MLX
import MLXFast
import MLXRandom

public struct PagedBackendBenchmark {

    public struct Scenario: Sendable {
        public var batch: Int
        public var context: Int
        public var queryHeads: Int
        public var kvHeads: Int
        public var headDim: Int

        public init(batch: Int, context: Int, queryHeads: Int, kvHeads: Int, headDim: Int) {
            self.batch = batch
            self.context = context
            self.queryHeads = queryHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
        }
    }

    public struct Measurement: Sendable {
        public var engine: String
        public var scenario: Scenario
        public var msPerStep: Double
        public var kvGBPerSec: Double
        public var tokensPerSec: Double
    }

    public var warmupSteps: Int
    public var timedSteps: Int

    public init(warmupSteps: Int = 10, timedSteps: Int = 40) {
        self.warmupSteps = warmupSteps
        self.timedSteps = timedSteps
    }

    /// GPT-OSS-20B-shaped attention (64 Q / 8 KV heads, headDim 64) across
    /// the spec's batch/context matrix.
    public static let defaultScenarios: [Scenario] = {
        var scenarios: [Scenario] = []
        for context in [512, 4096, 16384] {
            for batch in [1, 2, 4, 8] {
                scenarios.append(
                    Scenario(
                        batch: batch, context: context,
                        queryHeads: 64, kvHeads: 8, headDim: 64))
            }
        }
        return scenarios
    }()

    // MARK: - Engines

    /// Paged backend: per-row page-table writes + one kernel dispatch.
    func measurePaged(_ s: Scenario) throws -> Measurement {
        let kinds = [
            CBv2LayerKind(
                attention: .full, headDim: s.headDim, kvHeads: s.kvHeads,
                queryHeads: s.queryHeads)
        ]
        let maxLength = s.context + warmupSteps + timedSteps + 8
        let perSeqBytes = 2 * s.kvHeads * s.headDim * 2 * maxLength
        let config = PagedKVPoolConfig(
            capacityBytes: perSeqBytes * (s.batch + 1),
            nominalMaxSequenceLength: maxLength)
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let cache = backend.makeLayerCaches()[0]

        var states: [[CBv2SequenceKV?]] = []
        for _ in 0 ..< s.batch {
            states.append(
                try backend.makeSequenceState(
                    layerKinds: kinds, promptLength: s.context, maxLength: maxLength))
        }
        defer { states.forEach { backend.release($0) } }
        let rows = states.map { $0[0]! as! PagedSequenceKV }

        // Prefill the context in chunks.
        for row in rows {
            var remaining = s.context
            while remaining > 0 {
                let n = min(512, remaining)
                row.write(
                    keys: MLXRandom.normal([s.kvHeads, n, s.headDim], dtype: .float16),
                    values: MLXRandom.normal([s.kvHeads, n, s.headDim], dtype: .float16))
                remaining -= n
            }
        }
        cache.setRows(rows)
        // Force the prefill write chain (in-place writes ride the fence).
        eval([backend.pool.group(rows[0].groupKey).writeFence])

        let scale = Float(1.0 / Double(s.headDim).squareRoot())
        let step: () -> MLXArray = {
            let q = MLXRandom.normal([s.batch, s.queryHeads, 1, s.headDim], dtype: .float16)
            let k = MLXRandom.normal([s.batch, s.kvHeads, 1, s.headDim], dtype: .float16)
            let v = MLXRandom.normal([s.batch, s.kvHeads, 1, s.headDim], dtype: .float16)
            return cache.updateAndAttend(queries: q, keys: k, values: v, scale: scale, sinks: nil)
        }
        return time(engine: "paged-kernel", scenario: s, step: step)
    }

    /// v1-style backend: per-row contiguous caches, B separate SDPA
    /// dispatches per step.
    func measurePerRowSDPA(_ s: Scenario) -> Measurement {
        let capacity = s.context + warmupSteps + timedSteps + 8
        var ks: [MLXArray] = []
        var vs: [MLXArray] = []
        var lengths = [Int](repeating: s.context, count: s.batch)
        for _ in 0 ..< s.batch {
            let k = MLXArray.zeros([1, s.kvHeads, capacity, s.headDim], dtype: .float16)
            let v = MLXArray.zeros([1, s.kvHeads, capacity, s.headDim], dtype: .float16)
            k[0..., 0..., 0 ..< s.context, 0...] =
                MLXRandom.normal([1, s.kvHeads, s.context, s.headDim], dtype: .float16)
            v[0..., 0..., 0 ..< s.context, 0...] =
                MLXRandom.normal([1, s.kvHeads, s.context, s.headDim], dtype: .float16)
            ks.append(k)
            vs.append(v)
        }
        eval(ks + vs)

        let scale = Float(1.0 / Double(s.headDim).squareRoot())
        let step: () -> MLXArray = {
            var outs: [MLXArray] = []
            for i in 0 ..< s.batch {
                let t = lengths[i]
                let q = MLXRandom.normal([1, s.queryHeads, 1, s.headDim], dtype: .float16)
                let newK = MLXRandom.normal([1, s.kvHeads, 1, s.headDim], dtype: .float16)
                let newV = MLXRandom.normal([1, s.kvHeads, 1, s.headDim], dtype: .float16)
                ks[i][0..., 0..., t ..< (t + 1), 0...] = newK
                vs[i][0..., 0..., t ..< (t + 1), 0...] = newV
                lengths[i] = t + 1
                let out = MLXFast.scaledDotProductAttention(
                    queries: q,
                    keys: ks[i][0..., 0..., 0 ..< (t + 1), 0...],
                    values: vs[i][0..., 0..., 0 ..< (t + 1), 0...],
                    scale: scale, mask: .none)
                outs.append(out)
            }
            return concatenated(outs, axis: 0)
        }
        return time(engine: "v1-per-row-sdpa", scenario: s, step: step)
    }

    /// Legacy dense-batch shape: one rectangular [B, H, T, D] cache, one
    /// SDPA dispatch (uniform lengths — the dense engine's best case).
    func measureDenseBatch(_ s: Scenario) -> Measurement {
        let capacity = s.context + warmupSteps + timedSteps + 8
        let k = MLXArray.zeros([s.batch, s.kvHeads, capacity, s.headDim], dtype: .float16)
        let v = MLXArray.zeros([s.batch, s.kvHeads, capacity, s.headDim], dtype: .float16)
        k[0..., 0..., 0 ..< s.context, 0...] =
            MLXRandom.normal([s.batch, s.kvHeads, s.context, s.headDim], dtype: .float16)
        v[0..., 0..., 0 ..< s.context, 0...] =
            MLXRandom.normal([s.batch, s.kvHeads, s.context, s.headDim], dtype: .float16)
        eval([k, v])

        var t = s.context
        let scale = Float(1.0 / Double(s.headDim).squareRoot())
        let step: () -> MLXArray = {
            let q = MLXRandom.normal([s.batch, s.queryHeads, 1, s.headDim], dtype: .float16)
            let newK = MLXRandom.normal([s.batch, s.kvHeads, 1, s.headDim], dtype: .float16)
            let newV = MLXRandom.normal([s.batch, s.kvHeads, 1, s.headDim], dtype: .float16)
            k[0..., 0..., t ..< (t + 1), 0...] = newK
            v[0..., 0..., t ..< (t + 1), 0...] = newV
            t += 1
            return MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k[0..., 0..., 0 ..< t, 0...],
                values: v[0..., 0..., 0 ..< t, 0...],
                scale: scale, mask: .none)
        }
        return time(engine: "legacy-dense-batch", scenario: s, step: step)
    }

    private func time(
        engine: String, scenario s: Scenario, step: () -> MLXArray
    ) -> Measurement {
        for _ in 0 ..< warmupSteps {
            eval(step())
        }
        let start = DispatchTime.now()
        for _ in 0 ..< timedSteps {
            eval(step())
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
        let msPerStep = elapsed * 1000 / Double(timedSteps)
        // KV bytes touched per step: K+V for the (average) context, fp16.
        let avgContext = Double(s.context) + Double(warmupSteps) + Double(timedSteps) / 2
        let kvBytes = Double(s.batch) * 2 * Double(s.kvHeads) * avgContext * Double(s.headDim) * 2
        let gbPerSec = kvBytes / (elapsed / Double(timedSteps)) / 1e9
        let tokensPerSec = Double(s.batch) / (elapsed / Double(timedSteps))
        return Measurement(
            engine: engine, scenario: s, msPerStep: msPerStep,
            kvGBPerSec: gbPerSec, tokensPerSec: tokensPerSec)
    }

    // MARK: - Runner

    /// Run all engines over `scenarios`, returning a markdown report plus
    /// the raw measurements (for gate assertions).
    public func run(
        scenarios: [Scenario] = PagedBackendBenchmark.defaultScenarios
    ) throws -> (markdown: String, measurements: [Measurement]) {
        var all: [Measurement] = []
        var lines: [String] = []
        lines.append("| B | context | engine | ms/step | KV GB/s | tokens/s |")
        lines.append("|---|---------|--------|---------|---------|----------|")
        for scenario in scenarios {
            let paged = try measurePaged(scenario)
            let perRow = measurePerRowSDPA(scenario)
            let dense = measureDenseBatch(scenario)
            for m in [paged, perRow, dense] {
                all.append(m)
                lines.append(String(
                    format: "| %d | %d | %@ | %.3f | %.1f | %.0f |",
                    m.scenario.batch, m.scenario.context, m.engine,
                    m.msPerStep, m.kvGBPerSec, m.tokensPerSec))
            }
            Memory.clearCache()
        }
        return (lines.joined(separator: "\n"), all)
    }
}
