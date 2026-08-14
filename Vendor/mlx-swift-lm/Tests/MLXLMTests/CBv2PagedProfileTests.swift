// CBv2PagedProfileTests.swift
//
// Env-gated runner for the paged-decode root-cause profiler (kernel-opt
// track). Run with:
//   DARKBLOOM_CBV2_PAGED_PROFILE=1 swift test --filter CBv2PagedProfileTests
//
// Prints markdown ablation tables (slab-size sweep, write/dispatch split,
// 24-layer GPT-OSS emulation, dispatch-overhead probe). Numbers land in
// docs/engine-v2/kernel-research.md.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedProfile", .serialized)
struct CBv2PagedProfileTests {

    static let enabled =
        ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_PAGED_PROFILE"] == "1"

    @Test(.enabled(if: enabled))
    func slabScaling() throws {
        let profiler = PagedDecodeProfiler()
        let results = try profiler.slabScalingSuite()
        print("\n## Paged decode profile: slab-size sweep (1 layer, GPT-OSS shape)\n")
        print(PagedDecodeProfiler.markdownHeader)
        for r in results { print(r.markdownRow) }
        print("")
    }

    @Test(.enabled(if: enabled))
    func layerScaling() throws {
        let profiler = PagedDecodeProfiler()
        let results = try profiler.layerScalingSuite()
        print("\n## Paged decode profile: layer-count sweep (2 GiB pool)\n")
        print(PagedDecodeProfiler.markdownHeader)
        for r in results { print(r.markdownRow) }
        print("")
    }

    @Test(.enabled(if: enabled))
    func gptossEmulation() throws {
        let profiler = PagedDecodeProfiler()
        let results = try profiler.gptossEmulationSuite()
        print("\n## Paged decode profile: 24-layer GPT-OSS emulation (16 GiB pool)\n")
        print(PagedDecodeProfiler.markdownHeader)
        for r in results { print(r.markdownRow) }
        print("")
    }

    @Test(.enabled(if: enabled))
    func dispatchOverhead() throws {
        let profiler = PagedDecodeProfiler()
        let results = try profiler.dispatchOverheadSuite()
        print("\n## Paged decode profile: dispatch-overhead probe (ctx=16)\n")
        print(PagedDecodeProfiler.markdownHeader)
        for r in results { print(r.markdownRow) }
        print("")
    }

    /// Always-on smoke: one tiny scenario per mode exercises the profiler
    /// plumbing in CI without meaningful wall time.
    @Test func profilerSmoke() throws {
        let profiler = PagedDecodeProfiler(warmupSteps: 1, timedSteps: 2)
        for mode in PagedDecodeProfiler.StepMode.allCases {
            let r = try profiler.measurePaged(
                label: "smoke", layerCount: 2, capacityBytes: 8 << 20, batch: 2,
                context: 32, mode: mode)
            #expect(r.msPerStep > 0)
        }
        let sdpa = profiler.measureContiguousSDPA(
            label: "smoke", layerCount: 2, batch: 2, context: 32)
        #expect(sdpa.msPerStep > 0)
    }
}
