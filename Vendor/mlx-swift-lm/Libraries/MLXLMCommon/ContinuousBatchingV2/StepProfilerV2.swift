// StepProfilerV2.swift
//
// Opt-in phase timers for the decode hot loops (v2 EngineLoopV2 and the
// legacy GenerationBatch/Scheduler path). Used by BenchCBv2's profile mode
// to decompose per-step wall time into graph build / asyncEval submit /
// token readback / detokenization / bookkeeping.
//
// DISABLED by default: every instrumentation point is a single static Bool
// read (`CBv2StepProfiler.enabled`), so the production step path pays one
// predictable branch and nothing else. Enable programmatically (bench
// driver) or via CBV2_STEP_PROFILE=1.

import Foundation

public enum CBv2StepProfiler {

    /// Master switch. Read on hot paths; set it BEFORE the run under
    /// measurement and do not toggle mid-run (plain non-atomic Bool).
    nonisolated(unsafe) public static var enabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_STEP_PROFILE"].map {
            ["1", "true", "yes", "on"].contains($0.lowercased())
        } ?? false

    private static let lock = NSLock()
    nonisolated(unsafe) private static var samples: [String: [Double]] = [:]

    /// Record one duration (seconds) for a phase. No-op when disabled.
    @inline(__always)
    public static func record(_ phase: StaticString, seconds: Double) {
        guard enabled else { return }
        let key = "\(phase)"
        lock.lock()
        samples[key, default: []].append(seconds)
        lock.unlock()
    }

    /// Time a closure and record it under `phase`. No-op overhead when
    /// disabled beyond the closure call itself.
    @inline(__always)
    public static func time<T>(_ phase: StaticString, _ body: () -> T) -> T {
        guard enabled else { return body() }
        let start = CFAbsoluteTimeGetCurrent()
        let result = body()
        record(phase, seconds: CFAbsoluteTimeGetCurrent() - start)
        return result
    }

    public static func reset() {
        lock.lock()
        samples.removeAll()
        lock.unlock()
    }

    /// Snapshot of all recorded phases (name → sorted durations, seconds).
    public static func snapshot() -> [String: [Double]] {
        lock.lock()
        defer { lock.unlock() }
        return samples.mapValues { $0.sorted() }
    }

    /// Markdown decomposition table: per phase count, total, mean, p50,
    /// p95, max (milliseconds), sorted by total descending.
    public static func summaryTable() -> String {
        let snap = snapshot()
        func pct(_ sorted: [Double], _ q: Double) -> Double {
            guard !sorted.isEmpty else { return 0 }
            let rank = q * Double(sorted.count - 1)
            let lo = Int(rank.rounded(.down)), hi = Int(rank.rounded(.up))
            if lo == hi { return sorted[lo] }
            let w = rank - Double(lo)
            return sorted[lo] * (1 - w) + sorted[hi] * w
        }
        var out = "| phase | n | total ms | mean ms | p50 ms | p95 ms | max ms |\n"
        out += "|---|---|---|---|---|---|---|\n"
        let rows = snap.map { (name: $0.key, sorted: $0.value) }
            .sorted { $0.sorted.reduce(0, +) > $1.sorted.reduce(0, +) }
        for row in rows {
            let s = row.sorted
            let total = s.reduce(0, +)
            out += String(
                format: "| %@ | %d | %.1f | %.3f | %.3f | %.3f | %.3f |\n",
                row.name, s.count, total * 1e3, total / Double(s.count) * 1e3,
                pct(s, 0.5) * 1e3, pct(s, 0.95) * 1e3, (s.last ?? 0) * 1e3)
        }
        return out
    }
}
