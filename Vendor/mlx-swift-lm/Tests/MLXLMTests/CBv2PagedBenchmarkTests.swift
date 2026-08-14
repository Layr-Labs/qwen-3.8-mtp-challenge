// CBv2PagedBenchmarkTests.swift
//
// Env-gated microbenchmark runner for the paged backend (WS-C spec §5).
// Run with:
//   DARKBLOOM_CBV2_PAGED_BENCH=1 swift test --filter CBv2PagedBenchmarkTests
//
// The gate: paged decode beats v1 per-row SDPA dispatch at B >= 2 and is
// within 15% of single-request SDPA at B = 1. Results are recorded in
// benchmarks/cbv2-paged-attention.md.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedBenchmark", .serialized)
struct CBv2PagedBenchmarkTests {

    @Test(.enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_PAGED_BENCH"] == "1"))
    func decodeMicrobenchmark() throws {
        let bench = PagedBackendBenchmark()
        let (markdown, measurements) = try bench.run()
        print("\n## CBv2 paged-attention decode microbenchmark\n")
        print(markdown)
        print("")

        // Gate assertions (per scenario).
        for scenario in Set(measurements.map { "\($0.scenario.batch)x\($0.scenario.context)" }) {
            let group = measurements.filter {
                "\($0.scenario.batch)x\($0.scenario.context)" == scenario
            }
            guard
                let paged = group.first(where: { $0.engine == "paged-kernel" }),
                let perRow = group.first(where: { $0.engine == "v1-per-row-sdpa" })
            else { continue }
            if paged.scenario.batch >= 2 {
                #expect(
                    paged.msPerStep < perRow.msPerStep,
                    """
                    GATE: paged (\(paged.msPerStep)ms) must beat per-row SDPA \
                    (\(perRow.msPerStep)ms) at B=\(paged.scenario.batch) \
                    ctx=\(paged.scenario.context)
                    """)
            } else {
                #expect(
                    paged.msPerStep <= perRow.msPerStep * 1.15,
                    """
                    GATE: paged (\(paged.msPerStep)ms) must be within 15% of \
                    single-request SDPA (\(perRow.msPerStep)ms) at B=1 \
                    ctx=\(paged.scenario.context)
                    """)
            }
        }
    }

    /// Always-on smoke: a tiny scenario exercises the full benchmark
    /// plumbing (pool setup, prefill writes, timed loop) in CI without
    /// meaningful wall time.
    @Test func benchmarkSmoke() throws {
        let bench = PagedBackendBenchmark(warmupSteps: 2, timedSteps: 3)
        let (markdown, measurements) = try bench.run(scenarios: [
            .init(batch: 2, context: 128, queryHeads: 8, kvHeads: 2, headDim: 64)
        ])
        #expect(measurements.count == 3)
        #expect(markdown.contains("paged-kernel"))
        #expect(measurements.allSatisfy { $0.msPerStep > 0 })
    }
}
