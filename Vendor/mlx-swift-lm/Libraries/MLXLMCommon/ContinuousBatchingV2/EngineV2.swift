// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: the public engine (`CBv2Engine`).
//
// Thin wiring layer: scheduler + admission + loop + sampler + model adapter.
// Tokenization happens on the caller's task (requests already carry token
// ids per the contract); the engine thread only ever does graph-build +
// asyncEval. Cancellation drops the row at the next step boundary, O(1).

import Foundation
import MLX
import os

private let log = Logger(subsystem: "darkbloom", category: "CBv2Engine")

// MARK: - Shared gauges (submit-side admission ⇄ engine-thread truth)

/// Lock-protected counters bridging the caller-thread `submit`/`capacity`
/// surface and the engine thread. The loop publishes a full snapshot after
/// every step; `pendingSubmits` covers requests accepted on the caller
/// thread but not yet picked up by the engine queue (so the `maxWaiting`
/// bound holds even while a step blocks the queue).
final class CBv2EngineGauges: @unchecked Sendable {
    private let lock = NSLock()
    private var snapshot: CBv2CapacitySnapshot
    private var pendingSubmits = 0

    init(kvBytesCapacity: Int) {
        self.snapshot = CBv2CapacitySnapshot(
            activeRequests: 0, waitingRequests: 0, kvBytesInUse: 0,
            kvBytesCapacity: kvBytesCapacity, activeTokens: 0)
    }

    func update(_ newValue: CBv2CapacitySnapshot) {
        lock.lock()
        snapshot = newValue
        lock.unlock()
    }

    func read() -> CBv2CapacitySnapshot {
        lock.lock()
        defer { lock.unlock() }
        var value = snapshot
        value.waitingRequests += pendingSubmits
        return value
    }

    /// Fast-path `maxWaiting` bound; the scheduler's own check (on the
    /// engine thread) is authoritative.
    func beginSubmit(maxWaiting: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard snapshot.waitingRequests + pendingSubmits < maxWaiting else { return false }
        pendingSubmits += 1
        return true
    }

    func endSubmit() {
        lock.lock()
        pendingSubmits = max(0, pendingSubmits - 1)
        lock.unlock()
    }
}

// MARK: - EngineV2

/// Production continuous-batching engine v2 (WS-B).
///
/// ```swift
/// let engine = EngineV2(
///     model: adapter, layerKinds: kinds, backend: kvBackend,
///     cacheProvider: layerCacheFactory)
/// let events = try engine.submit(request)
/// for await event in events { ... }
/// ```
/// `@unchecked Sendable` justification (contract `CBv2Engine: Sendable`):
/// all mutable state is either lock-protected (`stateLock`,
/// `CBv2EngineGauges`) or confined to the engine's serial dispatch queue
/// inside `EngineLoopV2`; the only cross-thread surfaces are `submit`
/// (lock + queue hop), `cancel` (lock), `capacity()` (lock), and
/// `shutdown()` (queue-synchronized drain).
public final class EngineV2: CBv2Engine, @unchecked Sendable {
    private let loop: EngineLoopV2
    private let admission: AdmissionV2
    private let schedulerConfig: CBv2SchedulerConfig
    private let loopConfig: CBv2EngineLoopConfig
    private let gauges: CBv2EngineGauges
    private let layerKinds: [CBv2LayerKind]
    /// Non-nil only when active (instance supplied AND
    /// `schedulerConfig.enablePrefixCache`). Lookup + prefix slicing run on
    /// the submit thread (hashing is host work — never on the engine step
    /// thread); adoption and donation are handled by the loop.
    private let prefixCache: CBv2PrefixCache?

    private let stateLock = NSLock()
    private var rejectingSubmissions = false

    /// Engine health (step watchdog): false while a step exceeds the
    /// configured step timeout. Providers surface this in heartbeats.
    public var isHealthy: Bool { loop.isHealthy }
    /// Fired (once per wedge, from the watchdog thread) when a step exceeds
    /// the step timeout.
    public var onStepWedge: (@Sendable (TimeInterval) -> Void)? {
        get { loop.onStepWedge }
        set { loop.onStepWedge = newValue }
    }
    /// Telemetry/test hooks.
    public var stepCount: Int { loop.stepCount }
    public var chainedStepCount: Int { loop.chainedStepCount }
    public var preemptionCount: Int { loop.preemptionCount }
    /// Steps that evaluated the eager caches' offset/KV inner state (DAR-325
    /// guard). Test hook; engine-thread owned, read at quiescent points.
    public var offsetChainEvalSteps: Int { loop.offsetChainEvalSteps }
    /// Compiled-decode telemetry, or nil when the compiled path was never
    /// built (config off, ineligible model, unsupported hardware).
    /// NOTE: counters are engine-thread owned; read at quiescent points
    /// (after streams finish) for exact values.
    public var compiledDecodeStats: CBv2CompiledDecodeStats? { loop.compiledDecode?.stats }
    /// Internal test hook (engine-queue synchronized).
    var loopForTesting: EngineLoopV2 { loop }
    /// Internal test hook: the admission ledger, for asserting the compiled
    /// padding reserve is charged (or not) at construction.
    var admissionForTesting: AdmissionV2 { admission }

    public init(
        model: CBv2SteppableModel,
        layerKinds: [CBv2LayerKind],
        backend: CBv2KVBackend,
        cacheProvider: CBv2LayerCacheProvider,
        sampler: CBv2StepSampler = CBv2DefaultSampler(),
        detokenizerFactory: CBv2DetokenizerFactory = CBv2NullDetokenizerFactory(),
        schedulerConfig: CBv2SchedulerConfig = CBv2SchedulerConfig(),
        loopConfig: CBv2EngineLoopConfig = CBv2EngineLoopConfig(),
        admissionConfig: AdmissionV2.Config = AdmissionV2.Config(),
        prefixCache: CBv2PrefixCache? = nil,
        compiledDecodeConfig: CBv2CompiledDecodeConfig = CBv2CompiledDecodeConfig()
    ) {
        self.schedulerConfig = schedulerConfig
        self.loopConfig = loopConfig
        self.layerKinds = layerKinds
        let activePrefixCache = schedulerConfig.enablePrefixCache ? prefixCache : nil
        self.prefixCache = activePrefixCache
        if let violation = Self.prefixCachePairingViolation(
            backend: backend, prefixCache: activePrefixCache)
        {
            preconditionFailure(violation)
        }
        // Compiled [B, 1] decode (Phase 1): traced per bucket at warmup on
        // the engine queue, replayed for eligible pure-decode steps, eager
        // fallback everywhere else. nil when statically ineligible —
        // including when the eager layer caches carry an attention softcap
        // the compiled path cannot reproduce (numerics guard), when the
        // padding reserve would eat too much of the KV byte budget, or when
        // the backend mints rows the compiled path cannot bind (quantized
        // KV, paged slabs — see `producesCompiledDecodeEligibleRows`).
        var effectiveCompiledConfig = compiledDecodeConfig
        var softcapVeto = false
        if effectiveCompiledConfig.attentionSoftcap == nil {
            if let claim = cacheProvider.uniformAttentionSoftcap {
                // `.some(x)`: propagate so build() refuses (compiled SDPA
                // has no softcap path — never silently drift from eager
                // numerics). `.some(nil)`: uniformly uncapped, safe.
                effectiveCompiledConfig.attentionSoftcap = claim
            } else {
                // No claim (mixed caches, paged-only bank, or a custom
                // provider that cannot vouch): fail-safe — compiled decode
                // is vetoed rather than trusted to match eager numerics.
                softcapVeto = true
            }
        }
        // Backend row-type veto (PR#62 review): the compiled path can only
        // bind CBv2FullSequenceKV/CBv2WindowedSequenceKV rows. A backend
        // that mints anything else (contiguous with KV quantization, paged
        // slabs) would still warm the compiled graphs against fp16 scratch
        // and charge the admission padding reserve — then laneInfo rejects
        // every LIVE row, every step falls back eager, and the reserve
        // permanently tightens admission for zero benefit. Skip the build
        // entirely: nil executor, no reserve, stay eager.
        let backendVeto = !backend.producesCompiledDecodeEligibleRows
        if backendVeto, effectiveCompiledConfig.enabled, CBv2CompiledDecodeConfig.envEnabled {
            log.info(
                "CBv2 compiled decode skipped: \(type(of: backend), privacy: .public) mints rows the compiled path cannot bind (e.g. quantized KV) — staying eager, no padding reserve"
            )
        }
        let compiledDecode: CBv2CompiledDecode? =
            (softcapVeto || backendVeto)
            ? nil
            : CBv2CompiledDecode.build(
                model: model, layerKinds: layerKinds, config: effectiveCompiledConfig,
                maxConcurrentRequests: schedulerConfig.maxConcurrentRequests,
                kvBytesCapacity: backend.bytesCapacity)
        // Truthful admission under compiled padding: the compiled path pads
        // full-attention rows to its bucket capacity (more bytes than token
        // admission would budget), so its worst-case reserve is carved out
        // of the ledger as a REFUNDABLE external reserve — if warmup tracing
        // later disables compiled decode, the loop refunds it so admission
        // does not stay permanently tighter than the hardware truth (PR#62
        // review). Heartbeat capacity stays the hardware truth.
        let admission = AdmissionV2(
            layerKinds: layerKinds, bytesCapacity: backend.bytesCapacity,
            config: admissionConfig,
            externalReserveBytes: compiledDecode?.admissionPaddingReserve ?? 0)
        self.admission = admission
        let gauges = CBv2EngineGauges(kvBytesCapacity: backend.bytesCapacity)
        self.gauges = gauges
        self.loop = EngineLoopV2(
            model: model,
            layerKinds: layerKinds,
            backend: backend,
            cacheProvider: cacheProvider,
            sampler: sampler,
            detokenizerFactory: detokenizerFactory,
            scheduler: SchedulerV2(config: schedulerConfig, capacity: admission),
            capacity: admission,
            prefixCache: activePrefixCache,
            compiledDecode: compiledDecode,
            config: loopConfig,
            gauges: gauges)
        loop.start()
    }

    /// Structural safety check for the (backend, prefix-cache) pairing,
    /// enforced by `init` as a precondition: a backend whose donated
    /// snapshot views reference RECYCLABLE storage
    /// (`CBv2KVBackend.requiresMaterializedSnapshots`, e.g. the paged
    /// slabs) must never feed a `PrefixCacheV2` configured with
    /// `materializeOnDonate: false` — its entries would silently decay
    /// into other requests' bytes once the donor's pages are recycled.
    /// Returns the violation description, or nil when the pairing is safe.
    /// Custom `CBv2PrefixCache` implementations cannot be inspected
    /// structurally; they own this obligation themselves (see the
    /// `requiresMaterializedSnapshots` contract doc). Internal so tests
    /// can assert the rule without tripping the precondition.
    static func prefixCachePairingViolation(
        backend: CBv2KVBackend, prefixCache: CBv2PrefixCache?
    ) -> String? {
        guard backend.requiresMaterializedSnapshots,
            let concrete = prefixCache as? PrefixCacheV2,
            !concrete.config.materializeOnDonate
        else { return nil }
        return """
            EngineV2: \(type(of: backend)) donates snapshot views over recyclable \
            storage (requiresMaterializedSnapshots), but the prefix cache is \
            configured with materializeOnDonate: false — cached entries would \
            reference recycled pages. Use materializeOnDonate: true (the default).
            """
    }

    // MARK: CBv2Engine

    /// Submit a request; events stream until `.finished`. Throws
    /// `CBv2KVError.capacityExhausted` when truthful admission fails (worst
    /// case could never fit), the waiting queue is full, or the engine is
    /// shutting down — the provider maps this to 429/503 exactly as today.
    /// Throws `CBv2SchedulerError.duplicateRequestID` when the id is still
    /// live: the duplicate is rejected BEFORE any stream registration, so
    /// the original request's stream is never touched (PR#62 review).
    public func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> {
        stateLock.lock()
        let rejecting = rejectingSubmissions
        stateLock.unlock()
        guard !rejecting else {
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }

        // Degenerate requests: uniform event surface, no engine round-trip.
        if request.maxTokens <= 0 {
            return Self.immediateStream(
                reason: .length,
                usage: CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
        }
        if request.promptTokens.isEmpty {
            return Self.immediateStream(
                reason: .error("empty prompt"),
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
        }

        // Vision requests, CHEAP half only: span structure, model/backend
        // capability, block-vs-budget — no provider call, no MLX graphs.
        // The heavy embedding materialization waits until EVERY cheap
        // admission gate (canEverFit, maxWaiting, duplicate id) has passed,
        // so a rejected request never runs the vision tower and heavy work
        // is always counted in `pendingSubmits` (PR#63 review). All
        // multimodal failures remain submit-time throws
        // (`CBv2MultimodalError`); nothing reaches the scheduler.
        var multimodalBlocks: [CBv2ImageSpan] = []
        if let input = request.multimodal {
            multimodalBlocks = try CBv2MultimodalPlan.validate(
                input,
                promptTokenCount: request.promptTokens.count,
                model: loop.model,
                cacheProvider: loop.cacheProvider,
                maxBatchedTokensPerStep: schedulerConfig.maxBatchedTokensPerStep)
        }

        // Truthful admission: reject what could NEVER fit; everything else
        // is admitted optimistically (preemption is the backstop).
        guard
            admission.canEverFit(
                promptTokens: request.promptTokens.count, maxTokens: request.maxTokens)
        else {
            throw CBv2KVError.capacityExhausted(
                needed: admission.estimatedBytes(
                    forTokens: request.promptTokens.count + request.maxTokens),
                available: admission.admissibleBytesCapacity)
        }
        guard gauges.beginSubmit(maxWaiting: schedulerConfig.maxWaiting) else {
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }

        let loop = self.loop
        let stream = CBv2OutputStream(
            id: request.id,
            capacity: loopConfig.eventBufferCapacity,
            onBackpressure: { id, paused in loop.setPaused(id, paused) },
            onAbandoned: { id in loop.requestCancel(id) })
        // Registration doubles as the duplicate-id gate: it refuses to
        // replace a live stream, so the FIRST request keeps delivering and
        // the duplicate fails here — before the scheduler ever sees it.
        // (The scheduler's own `duplicateRequestID` rejection remains the
        // engine-thread backstop.)
        guard loop.register(stream: stream) else {
            gauges.endSubmit()
            throw CBv2SchedulerError.duplicateRequestID(request.id)
        }

        // Vision requests, HEAVY half: materialize the image embeddings ONCE,
        // here on the submit thread (the one provider call per request —
        // never on the engine step thread), now that every cheap gate has
        // passed. A failure must unwind the registration + pending-submit
        // count taken above (the stream was never handed to the caller).
        var multimodal: CBv2ResolvedMultimodal? = nil
        if let input = request.multimodal {
            do {
                multimodal = try CBv2MultimodalPlan.materialize(
                    input, blocks: multimodalBlocks, model: loop.model)
            } catch {
                loop.unregister(request.id)
                gauges.endSubmit()
                throw error
            }
        }

        loop.enqueue(request, adoption: makeAdoption(for: request), multimodal: multimodal)
        return stream.makeStream()
    }

    /// Prefix-cache lookup on the SUBMIT thread (SHA-256 hashing is host
    /// work; slicing is graph-only). Returns a prepared adoption sliced to
    /// `effective = matched - cbv2RequiredRecompute(...)` — the uniform
    /// offset every row adopts before the engine replays the trailing
    /// tokens. The lookup pin travels with the adoption; the loop balances
    /// it in every outcome.
    private func makeAdoption(for request: CBv2Request) -> CBv2PrefixAdoption? {
        guard let prefixCache else { return nil }
        // Vision requests NEVER look up the prefix cache (v1 policy — the
        // donation side is excluded symmetrically in
        // `EngineLoopV2.donationIntent`): token-id hashes cannot see image
        // content, so an adopted "hit" over another request's image span
        // would be silently wrong KV.
        guard request.multimodal == nil else { return nil }
        guard
            let hit = prefixCache.lookup(
                tokens: request.promptTokens, layerKinds: layerKinds,
                cacheSalt: request.cacheSalt)
        else { return nil }
        let recompute = cbv2RequiredRecompute(layerKinds: layerKinds, matched: hit.matched)
        let effective = hit.matched - recompute
        guard effective > 0 else {
            prefixCache.endAdoption(
                tokens: request.promptTokens, matched: hit.matched,
                cacheSalt: request.cacheSalt)
            return nil
        }
        let prefix = hit.prefix.map { entry -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            guard let entry else { return nil }
            guard entry.offset != effective else { return entry }
            return (
                keys: entry.keys[.ellipsis, 0 ..< effective, 0...],
                values: entry.values[.ellipsis, 0 ..< effective, 0...],
                offset: effective
            )
        }
        return CBv2PrefixAdoption(
            tokens: request.promptTokens, matched: hit.matched,
            effective: effective, prefix: prefix, cacheSalt: request.cacheSalt)
    }

    /// Cancel promptly: the in-flight step completes, the row is dropped
    /// O(1) at the next step boundary.
    public func cancel(_ id: CBv2RequestID) {
        loop.requestCancel(id)
    }

    /// Truthful capacity snapshot (actual bytes/tokens, not worst case),
    /// published by the engine thread after every step.
    public func capacity() -> CBv2CapacitySnapshot {
        gauges.read()
    }

    /// Graceful drain: new submissions are rejected, waiting requests are
    /// cancelled, running requests finish naturally, then the loop stops.
    /// Bounded by `CBv2EngineLoopConfig.shutdownTimeout` (default 10 s): if
    /// the engine queue is wedged, live streams are force-finished with
    /// `.error` and this returns instead of hanging forever.
    public func shutdown() async {
        beginRejectingSubmissions()
        await loop.drain()
    }

    /// Synchronous helper: `NSLock` is not async-safe to hold across
    /// suspension points, so the flag flip lives outside the async context.
    private func beginRejectingSubmissions() {
        stateLock.lock()
        rejectingSubmissions = true
        stateLock.unlock()
    }

    // MARK: Helpers

    private static func immediateStream(
        reason: CBv2FinishReason, usage: CBv2Usage
    ) -> AsyncStream<CBv2Event> {
        AsyncStream { continuation in
            continuation.yield(.finished(reason: reason, usage: usage))
            continuation.finish()
        }
    }
}
