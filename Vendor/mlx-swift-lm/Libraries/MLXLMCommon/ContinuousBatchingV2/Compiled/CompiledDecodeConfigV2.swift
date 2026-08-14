// CompiledDecodeConfigV2.swift
//
// ContinuousBatchingV2 — compiled [B, 1] decode (Phase 1 of the v2 plan).
//
// Configuration for the whole-step compiled decode path: one MLX
// `compile(inputs:outputs:)` graph per batch-size bucket, traced once at
// engine build (model-load time — compile must NEVER happen on the request
// path; the v0.6.30 lesson) and replayed for every pure-decode step whose
// row set fits the shape discipline. Any step that does not fit falls back
// to the eager v2 path — never wrong numerics, never a mid-request stall.

import Foundation
import MLX

/// Marker protocol: a `CBv2SteppableModel` whose forward pass is known to
/// drive `CBv2AttendingLayerCache` objects generically (the v2 contract
/// surface) and is therefore compile-traceable. The engine only activates
/// compiled decode for conforming models — scripted test mocks that bypass
/// the cache contract stay on the eager path without probing.
///
/// `CBv2SteppableLanguageModelAdapter` (the production bridge for Gemma 4 /
/// GPT-OSS v2 branches) conforms; test fixtures opt in explicitly.
public protocol CBv2CompiledSteppableModel: CBv2SteppableModel {}

/// Configuration for the v2 compiled decode path.
public struct CBv2CompiledDecodeConfig: Sendable {

    /// Master switch. Defaults to the `DARKBLOOM_CBV2_COMPILED` environment
    /// kill switch (absent ⇒ enabled). Even when enabled, compiled decode
    /// only activates for eligible models (`CBv2CompiledSteppableModel`) on
    /// supported hardware, with eager fallback everywhere else.
    public var enabled: Bool

    /// Batch-size bucket ladder. A decode step with B running rows uses the
    /// smallest bucket ≥ B, padding with inert scratch lanes; B larger than
    /// the biggest bucket runs eager. Sanitized at build (sorted, unique,
    /// clamped to the scheduler's `maxConcurrentRequests`).
    public var buckets: [Int]

    /// Fixed KV capacity (tokens) of every full-attention lane buffer in the
    /// compiled graph (the overflow-bin shape). Rows whose token count
    /// reaches this fall back to eager decode (which grows unbounded);
    /// sliding-window lanes use their model-defined window instead.
    public var kvCapacity: Int

    /// Construction-time attention-logit softcap of the ENGINE's layer
    /// caches (Gemma-2 family). The compiled attention path does not
    /// implement softcapping, so a non-nil value disables compiled decode
    /// entirely (matching numerics beats speed). Must mirror whatever the
    /// eager `CBv2LayerCache(attentionSoftcap:)` was built with.
    public var attentionSoftcap: Float?

    public init(
        enabled: Bool = CBv2CompiledDecodeConfig.envEnabled,
        buckets: [Int] = [1, 2, 4],
        kvCapacity: Int = 4096,
        attentionSoftcap: Float? = nil
    ) {
        self.enabled = enabled
        self.buckets = buckets
        self.kvCapacity = kvCapacity
        self.attentionSoftcap = attentionSoftcap
    }

    /// `DARKBLOOM_CBV2_COMPILED=0` (or false/no/off) kills the compiled
    /// path fleet-wide; any other (or absent) value keeps it on.
    public static let envEnabled: Bool = {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_COMPILED"] {
            return !["0", "false", "no", "off"].contains(raw.lowercased())
        }
        return true
    }()
}

/// Telemetry counters for the compiled decode path (engine-thread owned;
/// read via `EngineV2.compiledDecodeStats` snapshots).
public struct CBv2CompiledDecodeStats: Sendable {
    /// Steps that executed through a compiled bucket.
    public var compiledSteps = 0
    /// Pure-decode steps that fell back to eager, by reason.
    public var fallbacks: [String: Int] = [:]
    /// Lane (re)binds — must track membership changes, never steady-state.
    public var laneRebinds = 0
    /// Scratch-lane recycles (padding lanes hitting the capacity threshold).
    public var scratchResets = 0
    /// Per-bucket warmup trace cost, seconds (model-load budget signal).
    public var warmup: [(bucket: Int, seconds: Double)] = []
    /// Non-nil when compiled decode disabled itself (trace failure, dtype
    /// probe failure, config) — the exact reason, for eligibility reports.
    public var disabledReason: String?

    public init() {}
}
