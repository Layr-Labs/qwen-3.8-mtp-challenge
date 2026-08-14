// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

class MemoryTests: XCTestCase {

    func testWiredMemory() {
        Memory.withWiredLimit(1024 * 1024 * 256) {
            let x = MLXArray(10)
            print(x * x)
        }
    }

    /// The Metal resource-COUNT getters are wired through (C++ -> mlx-c -> Swift).
    func testResourceCountGetters() {
        // resourceLimit is the iogpu.rsrc_limit sysctl (~499000) or 0 on
        // non-Metal backends; numResources is non-negative and <= the limit.
        let limit = Memory.resourceLimit
        XCTAssertGreaterThanOrEqual(limit, 0)
        let count = Memory.numResources
        XCTAssertGreaterThanOrEqual(count, 0)
        if limit > 0 {
            XCTAssertLessThanOrEqual(count, limit)
        }
    }

    /// Regression for the `[metal::malloc] Resource limit (...) exceeded` crash:
    /// under churn with many DISTINCT buffer sizes (the size-keyed cache fills
    /// with entries that are never reused at that exact size) and the byte cache
    /// limit set huge (so the byte-driven trim never fires), the resource COUNT
    /// must stay below resource_limit_ — the count-aware high-water trim in
    /// MetalAllocator::malloc clears the cache at 90% so the limit is never hit.
    ///
    /// Run with MLX_RESOURCE_LIMIT set low (e.g. 50000) to exercise the trim
    /// quickly; otherwise it validates against the OS default (~499000).
    func testResourceCountStaysUnderLimitUnderChurn() throws {
        let limit = Memory.resourceLimit
        try XCTSkipIf(limit == 0, "no Metal backend / resource limit")

        // Disable the byte-driven trim so only the count-aware trim can bound us.
        let savedCacheLimit = Memory.cacheLimit
        Memory.cacheLimit = 64 * 1024 * 1024 * 1024
        defer { Memory.cacheLimit = savedCacheLimit }

        // Push distinct-sized allocations (each a new size-keyed cache entry) in
        // enough waves to drive the count well past the 90% high-water mark were
        // there no trim. A fixed wave budget keeps the test bounded: to reach the
        // ~499000 default we'd need too many allocs, so size the waves to clearly
        // exceed the limit only when MLX_RESOURCE_LIMIT pins it low (CI/local
        // exercise of the trim); at the OS default we still assert we never throw.
        let perWave = 4000
        // ~1.5x the count needed to cross the high-water if nothing trimmed.
        let waves = max(20, Int((Double(limit) * 1.5) / Double(perWave)) + 5)
        let cappedWaves = min(waves, 200)  // hard cap so the default-limit run ends
        var cursor = 8
        var peak = 0
        for _ in 0..<cappedWaves {
            autoreleasepool {
                var keep: [MLXArray] = []
                keep.reserveCapacity(perWave)
                for _ in 0..<perWave {
                    cursor += 1
                    let a = MLXArray(Array(repeating: Float(1), count: cursor))
                    keep.append((a + a).sum())
                }
                eval(keep)
            }
            peak = max(peak, Memory.numResources)
            // The invariant, checked every wave: the count-aware trim keeps us
            // strictly below the hard limit. Pre-fix, allocation would THROW
            // (crashing the test process) when num_resources_ reached the limit.
            XCTAssertLessThan(
                Memory.numResources, limit,
                "resource count reached the hard limit — count-aware trim did not fire")
        }
        print("[MemoryTests] resource-count churn: peak=\(peak) limit=\(limit) "
            + "waves=\(cappedWaves) final=\(Memory.numResources)")
        Memory.clearCache()
    }
}
