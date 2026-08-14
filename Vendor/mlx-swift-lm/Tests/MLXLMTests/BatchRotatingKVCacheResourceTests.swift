// Copyright © 2026 Apple Inc.
//
// Regression test for DAR-325: a Metal live-resource COUNT leak on the
// single-token decode path of the continuous-batching KV caches.
//
// Root cause: each decode step rebuilds the metadata MLXArrays `batchOffset`
// (+=1, both caches) and `leftPadding` (-=trim, BatchRotatingKVCache) into lazy
// add-chains that are never collapsed; each pins a tiny scalar buffer. gemma-4
// consumes each from only ONE representative cache (shared RoPE perRowOffset
// from the first cache, Gemma4.swift:1264-1266/:1300; sliding mask built once,
// :1244/:1283, makeMask -> leftPadding.max().item()), so every other cache
// leaks ~1-2 buffers/step — flat bytes, climbing COUNT, ~53/step in prod (25
// sliding + 5 full), until numResources hits the iogpu ceiling and aborts.
// keys/values do NOT leak (attention consumes the returned (k,v) each step).
// Fix: `asyncEval(batchOffset[, leftPadding])` on the decode path.
//
// Faithfulness (violating any hides the leak): (1) MULTI-cache — one cache
// can't expose it; (2) consume (k,v) from every cache but metadata from only
// one representative (eval a probe, never the cache); (3) measure numResources
// AFTER clearCache(), min over samples; (4) raise cache/memoryLimit so the byte
// trim can't mask the COUNT.

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("BatchKVCache decode resource-count leak (DAR-325)", .serialized)
struct BatchRotatingKVCacheResourceTests {

    private let steps = 4000
    private let warmup = 500
    private let H = 2
    private let D = 4

    /// LIVE Metal resource count: trim reclaimable buffers first, then take the
    /// min over a few rapid samples so a concurrently-running suite's transient
    /// live buffers do not inflate the reading. The DAR-325 leak is graph-pinned
    /// state that `clearCache()` cannot reclaim, so it survives this.
    private func liveResourceCount(samples: Int = 4) -> Int {
        var m = Int.max
        for _ in 0 ..< samples {
            MLX.Memory.clearCache()
            m = min(m, MLX.Memory.numResources)
        }
        return m
    }

    private func token() -> MLXArray { MLXArray.ones([1, H, 1, D], dtype: .float32) }

    /// Per-step average live-resource growth between warmup (steady state) and
    /// the end of the run — the leak rate. `representativeProbe` lets the caller
    /// model the shared single-consumer (RoPE offset / sliding mask) that only
    /// touches cache[0].
    private func decodeSlope(
        caches: [any KVCache],
        representativeProbe: (any KVCache) -> Void
    ) -> (growth: Int, slope: Double) {
        var warmupCount = 0
        for step in 0 ..< steps {
            autoreleasepool {
                var probes: [MLXArray] = []
                probes.reserveCapacity(caches.count * 2)
                for c in caches {
                    let (k, v) = c.update(keys: token(), values: token())
                    // Per-layer attention consumes every cache's (k, v).
                    probes.append(k.sum())
                    probes.append(v.sum())
                }
                // Shared single-consumer touches only the representative cache.
                representativeProbe(caches[0])
                eval(probes)
            }
            if step == warmup { warmupCount = liveResourceCount() }
        }
        let growth = liveResourceCount() - warmupCount
        return (growth, Double(growth) / Double(steps - warmup))
    }

    private func skipIfNoMetal(_ label: String) -> Bool {
        if MLX.Memory.resourceLimit <= 0 {
            print("[DAR-325] \(label): no Metal backend; skipping resource-count assertion")
            return true
        }
        return false
    }

    private func withRaisedLimits(_ body: () -> Void) {
        let savedCache = MLX.Memory.cacheLimit
        let savedMemory = MLX.Memory.memoryLimit
        defer {
            MLX.Memory.cacheLimit = savedCache
            MLX.Memory.memoryLimit = savedMemory
        }
        MLX.Memory.memoryLimit = 80 << 30
        MLX.Memory.cacheLimit = 80 << 30
        body()
    }

    /// A genuine leak grows ~linearly (tens per step → tens of thousands total);
    /// the fix and the control stay flat. The threshold (< 0.5/step) sits two
    /// orders of magnitude under the pre-fix slope (~48/step sliding, ~24/step
    /// full) and well above any clearCache-suppressed cross-suite noise.
    private let maxSlope = 0.5

    // MARK: - Subject: BatchRotatingKVCache (gemma sliding layers)

    @Test("sliding (BatchRotatingKVCache): multi-cache decode is resource-count stable")
    func batchRotatingMultiCacheDecodeIsStable() {
        if skipIfNoMetal("BatchRotatingKVCache") { return }
        withRaisedLimits {
            let N = 25  // gemma-4 sliding-layer count
            let maxSize = 64
            let caches: [any KVCache] = (0 ..< N).map { _ in
                BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0])
            }
            let (growth, slope) = decodeSlope(caches: caches) { rep in
                guard let rep = rep as? BatchRotatingKVCache else { return }
                // Shared RoPE: one cache's batchOffset. Shared sliding mask:
                // makeMask's `leftPadding.max().item()` on the same cache.
                eval(rep.batchOffset.sum())
                _ = rep.leftPadding.max().item(Int32.self)
            }
            print(
                "[DAR-325] BatchRotatingKVCache x\(N): growth=\(growth) over "
                    + "\(steps - warmup) steps, slope=\(slope)/step")
            #expect(
                slope < maxSlope,
                """
                BatchRotatingKVCache leaks Metal resources during decode: \
                \(growth) live buffers over \(steps - warmup) single-token steps \
                (slope \(slope)/step). Both batchOffset and leftPadding chains must \
                be collapsed on the decode path. This is the DAR-325 \
                `[metal::malloc] Resource limit exceeded` crash.
                """)
        }
    }

    // MARK: - Sibling: BatchKVCache (gemma full-attention layers)
    //
    // gemma-4 shares ONE perRowOffset across full AND sliding layers, so the
    // full-attention caches leak batchOffset too (they never mutate leftPadding,
    // so there is no leftPadding leak there — and gpt-oss, which consumes each
    // cache's own batchOffset via the generic applyRotaryPosition, is unaffected
    // either way).

    @Test("full (BatchKVCache): multi-cache decode is resource-count stable")
    func batchFullMultiCacheDecodeIsStable() {
        if skipIfNoMetal("BatchKVCache") { return }
        withRaisedLimits {
            let N = 8
            let caches: [any KVCache] = (0 ..< N).map { _ in BatchKVCache(leftPadding: [0]) }
            let (growth, slope) = decodeSlope(caches: caches) { rep in
                guard let rep = rep as? BatchKVCache else { return }
                eval(rep.batchOffset.sum())  // shared RoPE offset (one cache)
            }
            print(
                "[DAR-325] BatchKVCache x\(N): growth=\(growth) over "
                    + "\(steps - warmup) steps, slope=\(slope)/step")
            #expect(
                slope < maxSlope,
                """
                BatchKVCache leaks Metal resources during decode: \(growth) live \
                buffers over \(steps - warmup) single-token steps (slope \
                \(slope)/step). batchOffset must be collapsed on the decode path.
                """)
        }
    }

    // MARK: - Control: RotatingKVCache (genuinely leak-free)
    //
    // Single-stream RotatingKVCache writes in place into a pre-allocated ring
    // (updateInPlace, KVCache.swift) and tracks offset as a plain Int — no
    // per-step MLXArray metadata chain. It is flat BEFORE and AFTER the fix, so
    // it pins the measurement methodology: the metric itself does not
    // manufacture a leak.

    @Test("control (RotatingKVCache): single-stream decode is resource-count stable")
    func rotatingControlIsStable() {
        if skipIfNoMetal("RotatingKVCache") { return }
        withRaisedLimits {
            let caches: [any KVCache] = (0 ..< 4).map { _ in
                RotatingKVCache(maxSize: 64, keep: 0)
            }
            let (growth, slope) = decodeSlope(caches: caches) { _ in }
            print(
                "[DAR-325] RotatingKVCache control: growth=\(growth) over "
                    + "\(steps - warmup) steps, slope=\(slope)/step")
            #expect(
                slope < maxSlope,
                """
                RotatingKVCache control unexpectedly grew (\(growth)); the \
                measurement methodology is manufacturing a leak.
                """)
        }
    }
}
