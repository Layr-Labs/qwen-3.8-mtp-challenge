// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: the execution loop.
//
// Single engine thread (serial GCD queue, the EngineCore idiom): the loop
// does graph-build + `asyncEval` ONLY. Tokenization/detokenization state
// machines are pluggable (WS-E), prefix-cache donation and SSD I/O live
// elsewhere. Decode is rectangular [B, 1]; prefill runs per-request
// [1, chunk] under the shared token budget. There is no left padding, no
// shared frontier, and no batch-wide trim anywhere in this file.
//
// Chained async decode (SGLang-MLX pattern, report 09 §7): step N+1's [B, 1]
// forward is built ON TOP of step N's still-lazy sampled-token array and
// `asyncEval`ed BEFORE the loop blocks on step N's tokens for stop detection.
// Tokens are therefore inspected one step late; a finished request wastes at
// most one slot-step, and its extra token + KV tail are rolled back
// (`CBv2SequenceKV.rollback(1)`). The chain breaks on ANY membership change
// (prefill completion into a different set, finish, cancel, join, pause).

import Foundation
import MLX

// MARK: - Model interface (WS-F adapters / WS-G fixtures conform)

/// Minimal steppable-model surface the loop drives. `tokens` is [B, L] int32
/// ([B, 1] decode, [1, chunk] prefill); `caches` has one entry per model
/// layer with rows matching batch rows. Returns logits [B, L, vocab].
public protocol CBv2SteppableModel: AnyObject {
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray
}

/// Builds per-layer batch-facing cache views for a set of rows
/// (`rowStates[b][layer]`, row order == batch row order). WS-A's
/// `LayerCacheV2` conforms; see CONTRACT-ISSUES-B-scheduler.md §1.
public protocol CBv2LayerCacheProvider: AnyObject {
    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache]
    /// Attention-softcap claim for the eager caches this provider vends.
    /// `.some(nil)` = uniformly no softcap (compiled decode may build);
    /// `.some(x)` = uniform softcap x (propagated into the compiled config
    /// so `CBv2CompiledDecode.build` refuses — it has no softcap path);
    /// `nil` = no claim / mixed / unknown, which FAIL-SAFE VETOES compiled
    /// decode entirely. Conformers must answer truthfully: a provider that
    /// claims `.some(nil)` while vending softcapped caches produces silent
    /// numeric drift between eager and compiled steps.
    var uniformAttentionSoftcap: Float?? { get }
    /// True when EVERY cache this provider vends honors span-mask contexts
    /// (`CBv2SpanMaskBinding`), so vision prefill chunks can carry their
    /// causal-plus-bidirectional-within-span masks. Fail-safe default is
    /// false: providers that cannot vouch (paged backend, custom caches)
    /// reject multimodal requests at submit rather than silently serving
    /// them with plain causal masks.
    var supportsMultimodalSpans: Bool { get }
}

extension CBv2LayerCacheProvider {
    public var supportsMultimodalSpans: Bool { false }
}

// MARK: - Sampler interface (WS-E's CBv2DefaultSampler is the production impl)

/// Samples next tokens from last-position logits [B, vocab] → lazy token
/// array [B] (int32). MUST NOT host-sync; see
/// CONTRACT-ISSUES-B-scheduler.md §2 and CONTRACT-DECISIONS.md.
///
/// Stateful samplers (penalties, keyed RNG) reconfigure on membership
/// change using `rowContext` and track per-request progress:
///  - `params`/`requestIDs`: per-row, row order == logits row order.
///  - `stepIndex`: global engine step (telemetry only — per-request RNG must
///    key on per-request progress, never on this).
///  - `pendingSampledTokens`: lazy [B] tokens sampled for exactly these rows
///    by the still-in-flight previous step (chained decode), row-aligned;
///    nil when every row's history is fully confirmed. A sampler that
///    reconfigures from `rowContext` (confirmed history only) must fold
///    these in on-device to stay exact.
///  - `rowContext`: materializes full per-row context (id, params, prompt,
///    CONFIRMED output tokens). Copies token arrays — only call it on
///    membership change, never on the chained fast path.
public protocol CBv2StepSampler: AnyObject {
    func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray

    /// Consume the LAZY logprob gather built by the immediately preceding
    /// `sample` call (raw pre-transform logprobs; contract rule), or nil when
    /// no row of that call requested `topLogprobs > 0`. Graph-only handles:
    /// the loop adds them to the step's `asyncEval` set and materializes them
    /// at the finalize boundary alongside the sampled tokens — never an extra
    /// host sync on the chained decode path. Called at most once per `sample`
    /// (take semantics). Default: nil (samplers without logprob support).
    func takeStepLogprobs() -> CBv2StepLogprobs?

    /// The request left the engine for good (finish, cancel, error). Its id
    /// may legally be REUSED by a FUTURE request, so stateful samplers must
    /// drop any per-request configured state: without this, a later request
    /// whose row-id fingerprint matches the retired one exactly (e.g. the
    /// same id resubmitted solo) would skip reconfiguration and inherit the
    /// finished request's penalty counts and RNG step index (PR#62 review).
    /// NOT called on preemption — a preempted request is the SAME request
    /// and its sampler state remains a pure function of its history.
    /// Default: no-op (stateless samplers).
    func requestDidFinish(_ id: CBv2RequestID)
}

extension CBv2StepSampler {
    public func takeStepLogprobs() -> CBv2StepLogprobs? { nil }
    public func requestDidFinish(_ id: CBv2RequestID) {}
}

/// Greedy stub — vectorized argmax, batch-composition invariant by
/// construction (per-row reduction, no cross-row ops). Kept as the
/// deterministic fallback for scheduler/loop tests; production uses
/// `CBv2DefaultSampler`.
public final class CBv2GreedySampler: CBv2StepSampler {
    public init() {}
    public func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray {
        argMax(logits, axis: -1).asType(.int32)
    }
}

// MARK: - Detokenizer interface (WS-E conforms; null stub until then)

/// Incremental per-request detokenizer with UTF-8 + stop-string holdback.
/// See CONTRACT-ISSUES-B-scheduler.md §3.
public protocol CBv2IncrementalDetokenizer: AnyObject {
    /// Append confirmed tokens; returns text now safe to emit.
    func push(_ tokens: [Int]) -> String
    /// True once a stop string has matched (engine finishes with `.stop`).
    var matchedStopString: Bool { get }
    /// Held-back text still emittable at finish (excludes matched stop text).
    func flush() -> String
}

public protocol CBv2DetokenizerFactory: AnyObject {
    func makeDetokenizer(stopStrings: [String]) -> CBv2IncrementalDetokenizer
}

/// Default until WS-E lands: deltas carry token ids with empty text, stop
/// strings never match (stop tokens / maxTokens / deadlines still work).
public final class CBv2NullDetokenizerFactory: CBv2DetokenizerFactory {
    final class NullDetokenizer: CBv2IncrementalDetokenizer {
        let matchedStopString = false
        func push(_ tokens: [Int]) -> String { "" }
        func flush() -> String { "" }
    }
    public init() {}
    public func makeDetokenizer(stopStrings: [String]) -> CBv2IncrementalDetokenizer {
        NullDetokenizer()
    }
}

// MARK: - Loop configuration

public struct CBv2EngineLoopConfig: Sendable {
    /// Per-request wall-clock deadline; overdue requests are error-finished
    /// at the next step boundary.
    public var requestTimeout: TimeInterval
    /// Single-step watchdog: a step (graph build + blocking eval) exceeding
    /// this marks the engine unhealthy, error-finishes all live streams, and
    /// fires `onStepWedge`.
    public var stepTimeout: TimeInterval
    /// Watchdog polling interval.
    public var watchdogInterval: TimeInterval
    /// Idle re-check interval when there is no work.
    public var idleRecheckInterval: TimeInterval
    /// Per-request event buffer before backpressure pauses scheduling.
    public var eventBufferCapacity: Int
    /// Upper bound on a graceful drain. `shutdown()` waits for running
    /// requests to finish naturally; if the engine queue is wedged (a step
    /// blocked inside an eval), the drain would otherwise never complete —
    /// after this timeout every live stream is force-finished with
    /// `.error` and `shutdown()` returns. The wedged step may still be
    /// executing in the background; the process-level owner decides
    /// whether to exit.
    public var shutdownTimeout: TimeInterval

    public init(
        requestTimeout: TimeInterval = 120, stepTimeout: TimeInterval = 30,
        watchdogInterval: TimeInterval = 0.25, idleRecheckInterval: TimeInterval = 0.001,
        eventBufferCapacity: Int = 256, shutdownTimeout: TimeInterval = 10
    ) {
        self.requestTimeout = requestTimeout
        self.stepTimeout = stepTimeout
        self.watchdogInterval = watchdogInterval
        self.idleRecheckInterval = idleRecheckInterval
        self.eventBufferCapacity = eventBufferCapacity
        self.shutdownTimeout = shutdownTimeout
    }
}

// MARK: - In-flight step

/// One launched-but-not-finalized step. Its sampled tokens are still lazy;
/// finalization materializes them (the ONE host sync per step, overlapped
/// with the next step's GPU work when chained).
final class CBv2InFlightStep {
    /// Every request that computed anything this step (KV release for any of
    /// these must be deferred until finalization — see CONTRACT-ISSUES §4).
    let participants: Set<CBv2RequestID>
    /// Rows that sampled a token, in plan order (== row order of
    /// `sampledTokens`).
    let sampledRows: [CBv2RequestID]
    /// Lazy [K] int32, or nil when no row sampled (all mid-prefill chunks).
    let sampledTokens: MLXArray?
    /// Cheap handles that force evaluation of non-sampling prefill chunks.
    let evalTargets: [MLXArray]
    /// Rows finished/cancelled AFTER launch: their sampled token is
    /// discarded at finalization (the ≤1 wasted slot-step).
    var discard: Set<CBv2RequestID> = []
    /// Lazy per-step logprob gathers (rows that requested topLogprobs > 0
    /// exist in the batch). Graph-only until finalization, where they are
    /// materialized at the SAME boundary as the sampled tokens.
    var logprobSegments: [CBv2StepLogprobs] = []
    /// KV states whose release is fenced behind this step's completion.
    /// `rollbackOne` scrubs the wasted-token KV tail before release;
    /// `donation` (non-nil for natural finishes with prefix caching on)
    /// routes the retired state through the donation queue.
    var deferredReleases:
        [(
            id: CBv2RequestID, state: [CBv2SequenceKV?], rollbackOne: Bool,
            donation: CBv2DonationIntent?
        )] = []

    init(
        participants: Set<CBv2RequestID>, sampledRows: [CBv2RequestID],
        sampledTokens: MLXArray?, evalTargets: [MLXArray]
    ) {
        self.participants = participants
        self.sampledRows = sampledRows
        self.sampledTokens = sampledTokens
        self.evalTargets = evalTargets
    }
}

// MARK: - Prefix adoption handoff

/// A prefix-cache hit prepared on the submit thread (lookup + graph-only
/// slicing) and applied on the engine thread at enqueue. `prefix` is already
/// sliced to `effective = matched - cbv2RequiredRecompute(...)`, the uniform
/// offset every storage-owning row adopts; the engine replays
/// `[effective, prompt)` through all layers. The lookup's in-use pin is
/// balanced by exactly one `endAdoption(tokens:matched:)` when the adoption
/// is consumed or abandoned.
/// `@unchecked Sendable`: immutable value handed off exactly once, submit
/// thread → engine queue; the MLXArray views are graph-only until adopted.
struct CBv2PrefixAdoption: @unchecked Sendable {
    let tokens: [Int]
    let matched: Int
    let effective: Int
    let prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?]
    /// Request-scoped salt the lookup used (TB-007); the pin release and any
    /// fallback paths must carry the SAME salt or the pin leaks.
    let cacheSalt: String?
}

/// A finished request's donation intent: which tokens to donate and under
/// which salt scope (TB-007 — the entry must be indexed with the same salt
/// its donor request carried, or a differently-salted request could hit it).
struct CBv2DonationIntent {
    let tokens: [Int]
    let cacheSalt: String?
}

/// Single-consumer cross-queue handoff of engine-owned values (KV state,
/// donation snapshots). Safe by ownership transfer: the sender never
/// touches the value after enqueueing.
private struct CBv2Handoff<Value>: @unchecked Sendable {
    let value: Value
}

/// Resume-exactly-once wrapper for a drain continuation: the engine queue
/// (natural drain completion) and the shutdown-timeout timer race to
/// resume it; whichever wins consumes the continuation, the loser no-ops.
private final class CBv2DrainWaiter: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?

    init(_ continuation: CheckedContinuation<Void, Never>) {
        self.continuation = continuation
    }

    /// Resume if not already resumed; returns true when this call won.
    @discardableResult
    func resume() -> Bool {
        lock.lock()
        let c = continuation
        continuation = nil
        lock.unlock()
        c?.resume()
        return c != nil
    }
}

// MARK: - EngineLoopV2

/// The engine thread. All scheduler and MLX mutations happen on
/// `engineQueue` (serial, self-rescheduling — the EngineCore idiom that
/// keeps the GPU pipeline full with only GCD dispatch overhead between
/// steps). Cross-thread surface: `requestCancel`, `setPaused`, stream
/// registration, and the watchdog — all lock-protected.
public final class EngineLoopV2: @unchecked Sendable {
    let scheduler: SchedulerV2
    let capacity: CBv2StepCapacity?
    let backend: CBv2KVBackend
    let cacheProvider: CBv2LayerCacheProvider
    let model: CBv2SteppableModel
    let sampler: CBv2StepSampler
    let detokenizerFactory: CBv2DetokenizerFactory
    let layerKinds: [CBv2LayerKind]
    /// Non-nil only when prefix caching is active (instance supplied AND
    /// `CBv2SchedulerConfig.enablePrefixCache`).
    let prefixCache: CBv2PrefixCache?
    /// Compiled [B, 1] decode executor, or nil (eager only). Warmed on the
    /// engine queue before the first step; every pure-decode step tries it
    /// first and falls back to the eager path when it declines.
    let compiledDecode: CBv2CompiledDecode?
    let config: CBv2EngineLoopConfig
    let gauges: CBv2EngineGauges

    private let engineQueue = DispatchQueue(
        label: "com.eigen.cbv2.engine", qos: .userInitiated)
    private let watchdogQueue = DispatchQueue(
        label: "com.eigen.cbv2.watchdog", qos: .utility)
    /// Prefix-cache donation runs here (hashing + indexing + optional device
    /// materialization) — never on the engine step thread (invariant 6).
    /// Safe to eval CONCURRENTLY with engine-step evals over the same paged
    /// slabs: the snapshot views were graph-built on the engine thread, so
    /// MLX array versioning pins them to the slab contents as of that step
    /// (later slab writes produce new versions, never mutating pinned
    /// ones), and the release-after-donate ordering in `retire` keeps the
    /// donor's pages out of the pool until the donation has materialized.
    private let donationQueue = DispatchQueue(
        label: "com.eigen.cbv2.donation", qos: .utility)
    /// Detokenization (Tokenizer.decode + UTF-8 holdback) + stream emission
    /// for PASSTHROUGH requests (no stop strings) runs here, OFF the serial
    /// step thread — that host string work is bounded per token but at B>1
    /// it otherwise sits on the critical path between GPU steps for every
    /// row (PR#62 review). A single serial queue preserves global FIFO
    /// order, so each request's deltas + trailing flush + terminal event
    /// stay ordered (per-request order is a subsequence of global order).
    /// Requests WITH stop strings keep the fully synchronous engine-thread
    /// path: their holdback scan gates the deterministic one-step-late
    /// stop-string finish, which cannot be deferred without generating text
    /// past the stop (contract) or breaking the parity suites.
    private let detokQueue = DispatchQueue(
        label: "com.eigen.cbv2.detok", qos: .userInitiated)

    // Cross-thread state (stateLock).
    private let stateLock = NSLock()
    private var streams: [CBv2RequestID: CBv2OutputStream] = [:]
    private var pendingCancels: Set<CBv2RequestID> = []
    private var stepStartedNanos: UInt64 = 0
    private var wedgeReported = false
    private var _healthy = true

    // Engine-thread-confined state.
    private var detokenizers: [CBv2RequestID: CBv2IncrementalDetokenizer] = [:]
    private var kvStates: [CBv2RequestID: [CBv2SequenceKV?]] = [:]
    /// Tokens skipped via prefix-cache adoption, reported in usage.
    private var prefixHitTokens: [CBv2RequestID: Int] = [:]
    /// Resolved vision inputs (validated spans + materialized embeddings),
    /// keyed by request. Kept across PREEMPTION (a full re-prefill replays
    /// the span chunks and needs the embeddings again); dropped at finish.
    private var multimodalByID: [CBv2RequestID: CBv2ResolvedMultimodal] = [:]
    private var inFlight: CBv2InFlightStep?
    private var running = false
    private var draining = false
    private var drainWaiters: [CBv2DrainWaiter] = []
    /// True after a compiled decode step advanced rows OUTSIDE the eager
    /// provider's caches: the next eager bind must be forced to rebuild
    /// `positionOffsets` from host truth (see `eagerCaches`).
    private var eagerCompositionStale = false

    /// Telemetry / test hooks.
    public private(set) var stepCount = 0
    public private(set) var chainedStepCount = 0
    public private(set) var preemptionCount = 0
    /// Requests demoted back to waiting after a capacityExhausted at first
    /// allocation (test/telemetry hook), and the per-request attempt cap.
    private(set) var capacityRequeueCount = 0
    private var capacityRequeues: [CBv2RequestID: Int] = [:]
    static let maxCapacityRequeues = 64
    /// Steps that submitted eager layer-cache inner state (the offset chain +
    /// KV buffers) into the step's `asyncEval` set. The DAR-325 guard:
    /// evaluating the offset chain every eager step keeps its lazy `+ L`
    /// advance from accumulating O(steps) of graph. Zero for the compiled
    /// path (its counters advance in-graph) and for caches that vend no
    /// inner state (mocks). Test hook.
    public private(set) var offsetChainEvalSteps = 0
    /// Fired (from the watchdog thread) when a step exceeds `stepTimeout`.
    public var onStepWedge: (@Sendable (TimeInterval) -> Void)?
    /// Test hook: artificial delay at the START of the enqueue engine block,
    /// before the request reaches the scheduler. Widens the early-cancel race
    /// window (a cancel arriving after `submit` registered the stream but
    /// before enqueue ran) so the regression is deterministic. Zero in
    /// production. Set before submitting.
    var enqueueStartDelayForTesting: TimeInterval = 0

    public var isHealthy: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return _healthy
    }

    init(
        model: CBv2SteppableModel,
        layerKinds: [CBv2LayerKind],
        backend: CBv2KVBackend,
        cacheProvider: CBv2LayerCacheProvider,
        sampler: CBv2StepSampler,
        detokenizerFactory: CBv2DetokenizerFactory,
        scheduler: SchedulerV2,
        capacity: CBv2StepCapacity?,
        prefixCache: CBv2PrefixCache? = nil,
        compiledDecode: CBv2CompiledDecode? = nil,
        config: CBv2EngineLoopConfig,
        gauges: CBv2EngineGauges
    ) {
        self.model = model
        self.layerKinds = layerKinds
        self.backend = backend
        self.cacheProvider = cacheProvider
        self.sampler = sampler
        self.detokenizerFactory = detokenizerFactory
        self.scheduler = scheduler
        self.capacity = capacity
        self.prefixCache = prefixCache
        self.compiledDecode = compiledDecode
        self.config = config
        self.gauges = gauges
    }

    // MARK: Lifecycle

    func start() {
        engineQueue.async { [self] in
            guard !running else { return }
            running = true
            // Pre-warm compiled decode BEFORE the first step: compile must
            // never happen on the request path (the v0.6.30 lesson).
            // Requests submitted meanwhile queue behind this task.
            compiledDecode?.warmupIfNeeded()
            refundCompiledReserveIfDisabled()
            startWatchdog()
            engineQueue.async { [weak self] in self?.engineStep() }
        }
    }

    /// The compiled padding reserve was carved out of the admission ledger
    /// at engine build, but warmup tracing can DISABLE compiled decode (a
    /// model structure that resists tracing). The engine then serves eagerly
    /// forever and the padded buffers can never materialize — so the reserve
    /// must be refunded, or admission stays permanently tighter than the
    /// hardware truth (PR#62 review). Warmup is the only pending→disabled
    /// transition, so this runs exactly once, right after it, on the engine
    /// queue. (`AdmissionV2` is the only capacity oracle carrying the
    /// reserve; scripted test oracles never charge one.)
    private func refundCompiledReserveIfDisabled() {
        guard let compiledDecode, compiledDecode.disabledReason != nil else { return }
        (capacity as? AdmissionV2)?.refundExternalReserve()
    }

    /// Graceful drain: waiting requests are cancelled, running requests
    /// finish naturally, then the loop stops. Idempotent.
    ///
    /// Bounded by `config.shutdownTimeout`: the drain waits on the engine
    /// queue, so a wedged queue (a step blocked inside an eval) would hang
    /// it forever. The timeout runs on the watchdog queue; when it wins,
    /// every live stream is force-finished with `.error`, live requests
    /// are marked for cancellation (cleaned up if the loop ever resumes),
    /// and `drain()` returns — the wedged step may still be executing.
    func drain() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            let waiter = CBv2DrainWaiter(c)
            watchdogQueue.asyncAfter(deadline: .now() + config.shutdownTimeout) { [weak self] in
                guard let self else {
                    waiter.resume()
                    return
                }
                if waiter.resume() {
                    self.forceFinishStreamsOnShutdownTimeout()
                }
            }
            engineQueue.async { [self] in
                guard running else {
                    waiter.resume()
                    return
                }
                draining = true
                for rec in scheduler.waiting {
                    finishRequest(rec.id, reason: .cancelled)
                }
                publishGauges()
                if !scheduler.hasWork, inFlight == nil {
                    completeStop()
                    waiter.resume()
                } else {
                    drainWaiters.append(waiter)
                }
            }
        }
    }

    /// Shutdown-timeout path (watchdog queue): the engine queue is not
    /// making progress, so touch ONLY lock-protected state and the
    /// (thread-safe) streams — the same discipline as `watchdogTick()`.
    /// Stream `finish` is idempotent, so a later natural finish (if the
    /// loop ever resumes) is a harmless no-op.
    private func forceFinishStreamsOnShutdownTimeout() {
        stateLock.lock()
        let liveStreams = streams
        pendingCancels.formUnion(liveStreams.keys)
        stateLock.unlock()
        for (_, stream) in liveStreams {
            stream.finish(
                reason: .error(
                    "engine shutdown timed out after \(Int(config.shutdownTimeout))s"),
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
        }
    }

    private func completeStop() {
        running = false
        draining = false
        stopWatchdog()
    }

    // MARK: Submission (from EngineV2)

    /// Register the stream for a new request. Fails (returns false, state
    /// untouched) when a stream with the same id is still live: a duplicate
    /// `submit` must never REPLACE the original request's stream — the
    /// scheduler would later reject the duplicate id, and the rejection
    /// path's `takeStream` would tear down the replacement, leaving the
    /// FIRST request's deltas and terminal event with nowhere to go
    /// (PR#62 review). Ids only become reusable after `takeStream` removes
    /// the previous stream (request fully finished).
    func register(stream: CBv2OutputStream) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard streams[stream.id] == nil else { return false }
        streams[stream.id] = stream
        return true
    }

    /// Submit-side unwind: remove a stream registered by `submit` whose
    /// request never reached `enqueue` (multimodal materialization failed
    /// after registration — PR#63 review). Only legal BEFORE `enqueue`; a
    /// live request's stream is removed via `takeStream` at finish.
    func unregister(_ id: CBv2RequestID) {
        stateLock.lock()
        streams.removeValue(forKey: id)
        stateLock.unlock()
    }

    /// Runs on the engine queue. The stream must already be registered.
    /// `multimodal` is the submit-thread resolution of the request's vision
    /// input (nil for text requests) — mutually exclusive with `adoption`
    /// (vision requests never do prefix-cache lookup).
    func enqueue(
        _ request: CBv2Request, adoption: CBv2PrefixAdoption? = nil,
        multimodal: CBv2ResolvedMultimodal? = nil
    ) {
        engineQueue.async { [self] in
            defer {
                gauges.endSubmit()
                publishGauges()
            }
            if enqueueStartDelayForTesting > 0 {
                Thread.sleep(forTimeInterval: enqueueStartDelayForTesting)
            }
            guard running, !draining else {
                releaseAbandonedAdoption(adoption)
                takeStream(request.id)?.finish(
                    reason: .error("engine is shutting down"),
                    usage: CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
                return
            }
            // Early-cancel race: a cancel can arrive after `submit` registered
            // the stream but BEFORE this block runs — at which point the
            // scheduler has no record for it, so `processCancellations` cannot
            // act. A pending cancel is remembered until this point and
            // consumed here so the request is never started (PR#62 review).
            if consumeEarlyCancel(request.id) {
                releaseAbandonedAdoption(adoption)
                takeStream(request.id)?.finish(
                    reason: .cancelled,
                    usage: CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
                return
            }
            do {
                try scheduler.enqueue(
                    request,
                    deadline: Date().addingTimeInterval(config.requestTimeout))
                detokenizers[request.id] =
                    detokenizerFactory.makeDetokenizer(stopStrings: request.stopStrings)
                if let multimodal {
                    multimodalByID[request.id] = multimodal
                }
                if let adoption {
                    applyAdoption(adoption, requestID: request.id)
                }
            } catch let error as CBv2SchedulerError {
                releaseAbandonedAdoption(adoption)
                // Contract violation (duplicate live request id) — surfaced
                // distinctly from capacity backpressure.
                takeStream(request.id)?.finish(
                    reason: .error("scheduler rejected request: \(error)"),
                    usage: CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
            } catch {
                releaseAbandonedAdoption(adoption)
                // Precise maxWaiting enforcement (the submit-side gauge check
                // is the fast path; this is the authoritative one). The
                // MESSAGE IS A CONTRACT: it must classify exactly like
                // `EngineV2.submit`'s thrown queue-full sentinel
                // (`capacityExhausted(needed: 1, available: 0)`), which the
                // provider renders as this canonical string and maps to a
                // retryable 429 (`MultiModelBatchSchedulerEngineError
                // .fromSchedulerMessage` matches "queue full"). A divergent
                // wording here classified the SAME condition as a 500
                // (PR#62 review).
                takeStream(request.id)?.finish(
                    reason: .error("token_budget_exhausted: request queue full"),
                    usage: CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
            }
        }
    }

    // MARK: Prefix-cache adoption (engine thread)

    /// Adopt a prepared prefix hit into fresh KV state and fast-forward the
    /// scheduler record. Best-effort: any failure (capacity, backend) falls
    /// back to a full prefill — the request itself never fails here. Always
    /// balances the lookup pin.
    private func applyAdoption(_ adoption: CBv2PrefixAdoption, requestID: CBv2RequestID) {
        defer {
            prefixCache?.endAdoption(
                tokens: adoption.tokens, matched: adoption.matched,
                cacheSalt: adoption.cacheSalt)
        }
        guard let rec = scheduler.record(for: requestID), kvStates[requestID] == nil else {
            return
        }
        if let capacity {
            do {
                try capacity.reserve(id: requestID, additionalTokens: adoption.effective)
            } catch {
                return  // capacity tight — full prefill with the usual backstops
            }
        }
        do {
            let maxLength = rec.request.promptTokens.count + max(rec.request.maxTokens, 1)
            let state = try backend.makeSequenceState(
                adopting: adoption.prefix, layerKinds: layerKinds, maxLength: maxLength)
            kvStates[requestID] = state
            rec.numComputedTokens = adoption.effective
            prefixHitTokens[requestID] = adoption.effective
        } catch {
            capacity?.unreserve(id: requestID, tokens: adoption.effective)
        }
    }

    /// Balance a lookup pin for an adoption that never reached
    /// `applyAdoption` (shutdown, queue-full rejection).
    private func releaseAbandonedAdoption(_ adoption: CBv2PrefixAdoption?) {
        guard let adoption else { return }
        prefixCache?.endAdoption(
            tokens: adoption.tokens, matched: adoption.matched, cacheSalt: adoption.cacheSalt)
    }

    // MARK: Cancellation & backpressure (any thread)

    /// O(1): marks the request; the row is dropped at the next step boundary.
    /// Lock-based (not a queue hop) so cancellation works even while the
    /// engine thread is blocked inside a long step.
    func requestCancel(_ id: CBv2RequestID) {
        stateLock.lock()
        pendingCancels.insert(id)
        stateLock.unlock()
    }

    func setPaused(_ id: CBv2RequestID, _ paused: Bool) {
        engineQueue.async { [self] in
            if paused {
                scheduler.pause(id)
            } else {
                scheduler.resume(id)
            }
        }
    }

    // MARK: The step loop

    private func engineStep() {
        guard running else { return }
        let stepStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        markStepStarted()
        defer {
            markStepEnded()
            if CBv2StepProfiler.enabled {
                CBv2StepProfiler.record(
                    "v2.step.wall", seconds: CFAbsoluteTimeGetCurrent() - stepStart)
            }
        }

        processCancellations()
        processDeadlines()

        // Chained decode fast path: build step N+1 on step N's lazy tokens.
        if let previous = inFlight,
            previous.sampledTokens != nil,
            let ids = scheduler.chainCandidateIDs(),
            ids == previous.sampledRows,
            ids.allSatisfy({ kvStates[$0] != nil }),
            capacity?.hasHeadroom(additionalTokens: ids.count) ?? true
        {
            let plan = scheduler.plan()
            if isPureDecodePlan(plan, matching: ids) {
                if CBv2StepProfiler.enabled {
                    CBv2StepProfiler.record(
                        "v2.boundary", seconds: CFAbsoluteTimeGetCurrent() - stepStart)
                }
                let next = launchChainedDecode(plan, feeding: previous.sampledTokens!)
                inFlight = next
                chainedStepCount += 1
                stepCount += 1
                finalize(previous)
                publishGauges()
                scheduleNextStep()
                return
            }
            // Defensive: chainCandidateIDs and plan() disagree (only
            // possible when the capacity oracle's `hasHeadroom` was
            // optimistic and `reserve` preempted mid-plan). Roll the
            // optimistic advance back and fall through to the general path.
            // Preemptions are NOT rolled back (the scheduler already
            // requeued the victims), so their KV must be released here —
            // fenced behind the still-in-flight step that references it.
            // The victims are NOT added to `discard`: their in-flight
            // sample must be recorded at finalization (preemption keeps
            // generated tokens, and an unconfirmed `pendingSamples` would
            // block their re-admission forever).
            scheduler.rollback(plan)
            for id in plan.preemptions {
                preemptionCount += 1
                // A preempted request recomputes from scratch — any adopted
                // prefix credit no longer describes work that was skipped.
                prefixHitTokens.removeValue(forKey: id)
                guard let state = kvStates.removeValue(forKey: id) else { continue }
                compiledDecode?.forgetRows(state)
                if previous.participants.contains(id) {
                    previous.deferredReleases.append(
                        (id: id, state: state, rollbackOne: false, donation: nil))
                } else {
                    backend.release(state)
                }
            }
        }

        // Chain broken (or nothing chained): finalize before re-planning so
        // the plan sees confirmed tokens and post-stop membership.
        if let previous = inFlight {
            inFlight = nil
            finalize(previous)
        }

        guard scheduler.hasWork else {
            // Idle: no future step will rebind the eager caches, so drop
            // their row bindings now — otherwise the last batch's (retired)
            // rows stay strongly retained by the provider until some future
            // rebind, pinning dead KV on an idle engine (PR#62 review).
            // No-op after the first call while idle.
            (cacheProvider as? CBv2CompositionInvalidating)?.releaseBoundRows()
            publishGauges()
            if draining {
                completeStop()
                let waiters = drainWaiters
                drainWaiters = []
                for waiter in waiters { waiter.resume() }
                return
            }
            scheduleIdleRecheck()
            return
        }

        let plan = scheduler.plan()
        handlePreemptions(plan.preemptions)
        guard !plan.assignments.isEmpty else {
            publishGauges()
            scheduleIdleRecheck()
            return
        }
        inFlight = executeMixed(plan)
        stepCount += 1
        publishGauges()
        scheduleNextStep()
    }

    private func scheduleNextStep() {
        engineQueue.async { [weak self] in self?.engineStep() }
    }

    private func scheduleIdleRecheck() {
        engineQueue.asyncAfter(deadline: .now() + config.idleRecheckInterval) { [weak self] in
            self?.engineStep()
        }
    }

    // MARK: Step execution

    private func isPureDecodePlan(_ plan: CBv2StepPlan, matching ids: [CBv2RequestID]) -> Bool {
        guard plan.preemptions.isEmpty, plan.assignments.count == ids.count else { return false }
        for (i, assignment) in plan.assignments.enumerated() {
            guard assignment.numTokens == 1, assignment.id == ids[i] else { return false }
        }
        return true
    }

    /// Eager layer caches, with the provider's composition fingerprint
    /// force-invalidated when compiled steps advanced rows behind its back.
    private func eagerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        if eagerCompositionStale {
            (cacheProvider as? CBv2CompositionInvalidating)?.invalidateBoundComposition()
            eagerCompositionStale = false
        }
        return cacheProvider.layerCaches(rowStates: rowStates)
    }

    /// Inner-state arrays (lazily-mutated KV buffers + the on-device
    /// `positionOffsets` chain) of the eager layer caches, collected so the
    /// step's `asyncEval` collapses those lazy chains every step instead of
    /// letting `updateAndAttend`'s `+ L` offset advance accumulate O(steps)
    /// of unevaluated graph — the DAR-325 bug class (legacy `BatchKVCache`
    /// had exactly this). Empty for the compiled path (its counters advance
    /// in-graph) and for caches that vend no inner state (mocks).
    private func eagerCacheInnerState(_ caches: [CBv2AttendingLayerCache]) -> [MLXArray] {
        caches.flatMap { ($0 as? KVCache)?.innerState() ?? [] }
    }

    /// Last-position logits [B, vocab] for a rectangular [B, 1] decode
    /// batch: the compiled step when eligible, else the eager forward. The
    /// second tuple element is the eager caches' inner state (offset chain +
    /// KV buffers) that must ride the step's `asyncEval` (DAR-325); it is
    /// empty on the compiled path.
    /// Numerics note: both paths are pinned; they may only alternate at
    /// step boundaries, and the parity suites hold them token-exact.
    private func decodeLogits(
        rowStates: [[CBv2SequenceKV?]], tokens: MLXArray
    ) -> (logits: MLXArray, cacheInnerState: [MLXArray]) {
        if let compiledDecode,
            let compiled = compiledDecode.decodeStep(rowStates: rowStates, tokens: tokens)
        {
            // First compiled step after an eager bind: release the eager
            // caches' row bindings alongside marking them stale. Compiled
            // steps never rebind the eager caches, so a compiled-only
            // stretch would otherwise pin the last eager batch's rows —
            // retired ones included — indefinitely (PR#62 review). Guarded
            // by the stale flag so the chained compiled hot path pays this
            // exactly once per eager→compiled transition.
            if !eagerCompositionStale {
                eagerCompositionStale = true
                (cacheProvider as? CBv2CompositionInvalidating)?.releaseBoundRows()
            }
            return (compiled, [])
        }
        let caches = eagerCaches(rowStates: rowStates)
        let logits = model.forward(tokens: tokens, caches: caches)
        return (logits[0..., -1, 0...], eagerCacheInnerState(caches))
    }

    /// Pure-decode step fed by the previous step's still-lazy tokens.
    private func launchChainedDecode(
        _ plan: CBv2StepPlan, feeding lazyTokens: MLXArray
    ) -> CBv2InFlightStep {
        let buildStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let ids = plan.assignments.map(\.id)
        let rowStates = ids.map { kvStates[$0]! }  // presence pre-checked
        var params: [CBv2SamplingParams] = []
        params.reserveCapacity(ids.count)
        for id in ids { params.append(scheduler.record(for: id)!.request.sampling) }

        let inputs = lazyTokens.reshaped([ids.count, 1])
        let forwardStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let (last, cacheInnerState) = decodeLogits(rowStates: rowStates, tokens: inputs)  // [B, vocab]
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.forward.build", seconds: CFAbsoluteTimeGetCurrent() - forwardStart)
        }
        // `pendingSampledTokens` = the fed lazy tokens: each row has exactly
        // one launched-but-unconfirmed sample here (the chain invariant).
        let samplerStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let sampled = sampler.sample(
            logits: last, params: params, requestIDs: ids, stepIndex: stepCount,
            pendingSampledTokens: lazyTokens,
            rowContext: { [scheduler] in
                ids.map { Self.samplerRow(scheduler.record(for: $0)!) }
            })
        let stepLogprobs = sampler.takeStepLogprobs()
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.sampler.build", seconds: CFAbsoluteTimeGetCurrent() - samplerStart)
        }
        scheduler.markPendingSamples(ids: ids)
        var toEval = [sampled]
        if let stepLogprobs { toEval.append(contentsOf: stepLogprobs.evalTargets) }
        // Collapse the eager caches' lazy offset/KV chains this step (DAR-325).
        if !cacheInnerState.isEmpty {
            toEval.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }
        let evalStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        asyncEval(toEval)
        if CBv2StepProfiler.enabled {
            let now = CFAbsoluteTimeGetCurrent()
            CBv2StepProfiler.record("v2.asyncEval.submit", seconds: now - evalStart)
            CBv2StepProfiler.record("v2.launch.total", seconds: now - buildStart)
        }
        let step = CBv2InFlightStep(
            participants: Set(ids), sampledRows: ids, sampledTokens: sampled, evalTargets: [])
        if let stepLogprobs { step.logprobSegments = [stepLogprobs] }
        return step
    }

    /// General step: decode batch [B, 1] + per-request [1, chunk] prefills,
    /// interleaved per the plan, ONE `asyncEval` for the whole step.
    private func executeMixed(_ plan: CBv2StepPlan) -> CBv2InFlightStep? {
        struct RowWork {
            let rec: CBv2ScheduledRequest
            let start: Int
            let count: Int
            let samples: Bool
            let isDecode: Bool
        }

        var work: [RowWork] = []
        work.reserveCapacity(plan.assignments.count)
        for (id, n) in plan.assignments {
            guard let rec = scheduler.record(for: id) else { continue }
            guard ensureKVState(rec) != nil else { continue }  // error-finished
            let start = rec.numComputedTokens - n  // pre-optimistic-advance position
            // The step samples iff it computes through the last known token.
            // (`pendingSamples == 0` here: finalize always precedes
            // executeMixed, so every planned token value is host-visible.)
            let samples = rec.numComputedTokens == rec.effectiveTokenCount
            // A length-1 image span on the FINAL prompt token produces an
            // assignment shaped exactly like a decode row (n == 1, samples,
            // last known token) — but its placeholder token must take the
            // embedding-splice prefill path below, not the decode batch's
            // plain token-embedding path (PR#63 review). Genuine decode rows
            // never trip this: their positions are past every span.
            let finalTokenIsImageSpan =
                multimodalByID[id]?.containsSpan(at: rec.tokens.count - 1) ?? false
            let isDecode =
                n == 1 && samples && start == rec.tokens.count - 1 && !finalTokenIsImageSpan
            work.append(
                RowWork(rec: rec, start: start, count: n, samples: samples, isDecode: isDecode))
        }
        guard !work.isEmpty else { return nil }

        // Lazy offset/KV chains of every eager cache touched this step; ride
        // the step's asyncEval so the `+ L` offset advance can't accumulate
        // O(steps) of unevaluated graph (DAR-325).
        var cacheInnerState: [MLXArray] = []

        // Rectangular decode batch, in plan order.
        let decodeRows = work.filter(\.isDecode)
        var decodeSampled: MLXArray?
        var logprobSegments: [CBv2StepLogprobs] = []
        if !decodeRows.isEmpty {
            let inputs = MLXArray(decodeRows.map { Int32($0.rec.tokens[$0.start]) })
                .reshaped([decodeRows.count, 1])
            let (last, decodeInnerState) = decodeLogits(
                rowStates: decodeRows.map { kvStates[$0.rec.id]! }, tokens: inputs)
            cacheInnerState.append(contentsOf: decodeInnerState)
            decodeSampled = sampler.sample(
                logits: last,
                params: decodeRows.map(\.rec.request.sampling),
                requestIDs: decodeRows.map(\.rec.id),
                stepIndex: stepCount,
                pendingSampledTokens: nil,  // finalize preceded: all confirmed
                rowContext: { decodeRows.map { Self.samplerRow($0.rec) } })
            if let stepLogprobs = sampler.takeStepLogprobs() {
                logprobSegments.append(stepLogprobs)
            }
        }

        // Per-request prefill chunks [1, chunk].
        var prefillSampled: [CBv2RequestID: MLXArray] = [:]
        var evalTargets: [MLXArray] = []
        for row in work where !row.isDecode {
            let rec = row.rec
            let slice = rec.tokens[row.start ..< row.start + row.count]
            let inputs = MLXArray(slice.map(Int32.init)).reshaped([1, row.count])
            let caches = eagerCaches(rowStates: [kvStates[rec.id]!])
            let logits: MLXArray
            if let multimodal = multimodalByID[rec.id],
                let spanContext = multimodal.chunkContext(start: row.start, count: row.count)
            {
                // Vision chunk (contains image spans): the NEW pinned path —
                // spliced input embeddings + span attention masks. Chunks of
                // the SAME request without spans fall through to the
                // untouched text path (pure function of has-spans).
                logits = multimodalChunkForward(
                    tokens: inputs, start: row.start, count: row.count,
                    multimodal: multimodal, spanContext: spanContext, caches: caches)
            } else {
                logits = model.forward(tokens: inputs, caches: caches)
            }
            cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))
            if row.samples {
                prefillSampled[rec.id] = sampler.sample(
                    logits: logits[0..., -1, 0...],
                    params: [rec.request.sampling],
                    requestIDs: [rec.id],
                    stepIndex: stepCount,
                    pendingSampledTokens: nil,
                    rowContext: { [Self.samplerRow(rec)] })
                if let stepLogprobs = sampler.takeStepLogprobs() {
                    logprobSegments.append(stepLogprobs)
                }
            } else {
                // Cheap handle that forces this chunk's graph (incl. KV
                // writes) without materializing full logits on the host.
                evalTargets.append(logits[0, row.count - 1, 0 ..< 1])
            }
        }

        // Assemble sampled tokens in plan order so a following pure-decode
        // plan (same membership, same order) can chain off this array.
        var pieces: [MLXArray] = []
        var sampledRows: [CBv2RequestID] = []
        var decodeIdx = 0
        for row in work {
            if row.isDecode {
                pieces.append(decodeSampled![decodeIdx ..< decodeIdx + 1])
                decodeIdx += 1
                sampledRows.append(row.rec.id)
            } else if let s = prefillSampled[row.rec.id] {
                pieces.append(s)
                sampledRows.append(row.rec.id)
            }
        }
        let sampledTokens: MLXArray? =
            pieces.isEmpty ? nil : (pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0))

        scheduler.markPendingSamples(ids: sampledRows)
        var toEval = evalTargets
        if let sampledTokens { toEval.append(sampledTokens) }
        for segment in logprobSegments { toEval.append(contentsOf: segment.evalTargets) }
        if !cacheInnerState.isEmpty {
            toEval.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }
        asyncEval(toEval)

        let step = CBv2InFlightStep(
            participants: Set(work.map(\.rec.id)),
            sampledRows: sampledRows,
            sampledTokens: sampledTokens,
            evalTargets: evalTargets)
        step.logprobSegments = logprobSegments
        return step
    }

    // MARK: Vision prefill (span-containing chunks only)

    /// Forward one span-containing prefill chunk: splice the request's image
    /// embeddings over the scaled text embeddings, bind the chunk's span
    /// mask context to every layer cache for the duration of the graph
    /// build, and run the model's embedding forward. The context is unbound
    /// before returning, so every other computation (decode, text chunks,
    /// batchmates) sees nil — the mask can only ever apply to this one
    /// request's [1, chunk] rows.
    ///
    /// A 1-TOKEN chunk (`count == 1`) still splices its image embedding, but
    /// is NOT span-mask-bound: a chunk that small can only contain a
    /// length-1 block (blocks never split across chunks), and bidirectional
    /// attention within a single-token block is a no-op — the token attends
    /// only itself, identical under causal. Binding would trip the
    /// attention path's `L > 1` span precondition, so it is skipped
    /// (PR review: reachable via a length-1 span landing in a trailing
    /// 1-token prefill chunk).
    private func multimodalChunkForward(
        tokens: MLXArray, start: Int, count: Int,
        multimodal: CBv2ResolvedMultimodal, spanContext: CBv2SpanChunkContext,
        caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        guard let mmModel = model as? CBv2MultimodalSteppableModel else {
            // Unreachable: EngineV2.submit gates multimodal requests on this
            // capability before they ever reach the scheduler.
            preconditionFailure(
                "CBv2 multimodal chunk reached a model without embedding-forward support")
        }
        let textEmbeddings = mmModel.embedPromptTokens(tokens)
        let spliced = CBv2MultimodalPlan.spliceEmbeddings(
            textEmbeddings: textEmbeddings,
            chunkStart: start,
            spans: multimodal.spansInChunk(start: start, count: count))
        let bindables = count > 1 ? caches.compactMap { $0 as? CBv2SpanMaskBinding } : []
        for bindable in bindables { bindable.bindSpanContext(spanContext) }
        defer { for bindable in bindables { bindable.bindSpanContext(nil) } }
        return mmModel.forward(tokens: tokens, inputEmbeddings: spliced, caches: caches)
    }

    // MARK: Finalization (deferred stop detection)

    private func finalize(_ step: CBv2InFlightStep) {
        // THE host sync — overlapped with the successor step's GPU work when
        // chained. All-prefill steps block on their eval targets instead so
        // graph pipelining stays bounded at two steps.
        let readbackStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        var host: [Int32] = []
        if let tokens = step.sampledTokens {
            host = tokens.asArray(Int32.self)
        } else if !step.evalTargets.isEmpty {
            eval(step.evalTargets)
        }
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.readback.wait", seconds: CFAbsoluteTimeGetCurrent() - readbackStart)
        }

        // Materialize any lazy logprob gathers at this same boundary (they
        // rode the step's asyncEval, so this reads back finished results —
        // no extra GPU work, and only when a row asked for logprobs).
        var logprobsByID: [CBv2RequestID: CBv2TokenLogprob] = [:]
        if !step.logprobSegments.isEmpty {
            var tokenByID: [CBv2RequestID: Int] = [:]
            for (i, id) in step.sampledRows.enumerated() { tokenByID[id] = Int(host[i]) }
            for segment in step.logprobSegments {
                let sampledTokens = segment.rows.map { tokenByID[$0] ?? -1 }
                let assembled = CBv2Logprobs.assemble(
                    segment.gathered, sampledTokens: sampledTokens,
                    topLogprobsPerRow: segment.topLogprobsPerRow)
                for (row, logprob) in zip(segment.rows, assembled) {
                    if let logprob { logprobsByID[row] = logprob }
                }
            }
        }

        for (i, id) in step.sampledRows.enumerated() {
            if step.discard.contains(id) { continue }
            guard let rec = scheduler.record(for: id) else { continue }
            let token = Int(host[i])
            scheduler.recordSampled(id: id, token: token)

            // A stop TOKEN's text is never emitted (OpenAI behavior: the
            // stop token terminates the stream and its rendering is
            // excluded from content). It still counts toward
            // usage.completionTokens (recordSampled above) and still rides
            // in the delta's raw `tokens` — see the CBv2Event.delta doc.
            // Skipping the detokenizer push keeps its held-back text
            // intact for the finish-time flush.
            let isStopToken = rec.request.stopTokens.contains(token)
            let detokenizer = detokenizers[id]
            let logprobs = logprobsByID[id].map { [$0] }

            // Stop-string detection needs the holdback scan synchronously to
            // gate the deterministic one-step-late `.stop` finish; passthrough
            // requests can't stop on a string, so their decode + emit move
            // OFF the step thread (finding 2 — detokQueue). The buffer slot
            // is charged NOW, on the engine thread (`reserveEmission`), so
            // the backpressure pause still gates this request's scheduling
            // even though the emit itself is deferred — otherwise a slow
            // detokenizer/consumer could not fill the bounded buffer until
            // the queued blocks ran, and generation would run unbounded
            // ahead of it (PR#62 review).
            var matchedStopString = false
            if rec.request.stopStrings.isEmpty {
                let stream = stream(for: id)
                stream?.reserveEmission()
                detokQueue.async {
                    let text = isStopToken ? "" : (detokenizer?.push([token]) ?? "")
                    stream?.emit(
                        .delta(text: text, tokens: [token], logprobs: logprobs),
                        consumingReservation: true)
                }
            } else {
                let detokStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
                let text = isStopToken ? "" : (detokenizer?.push([token]) ?? "")
                if CBv2StepProfiler.enabled {
                    CBv2StepProfiler.record(
                        "v2.detok.push", seconds: CFAbsoluteTimeGetCurrent() - detokStart)
                }
                let emitStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
                stream(for: id)?.emit(.delta(text: text, tokens: [token], logprobs: logprobs))
                if CBv2StepProfiler.enabled {
                    CBv2StepProfiler.record(
                        "v2.stream.emit", seconds: CFAbsoluteTimeGetCurrent() - emitStart)
                }
                matchedStopString = detokenizer?.matchedStopString ?? false
            }

            // Stop detection — one step late by construction.
            if isStopToken {
                finishRequest(id, reason: .stop)
            } else if matchedStopString {
                finishRequest(id, reason: .stop)
            } else if rec.generatedTokenCount >= rec.request.maxTokens {
                finishRequest(id, reason: .length)
            } else if let deadline = rec.deadline, Date() > deadline {
                finishRequest(
                    id, reason: .error("request exceeded \(Int(config.requestTimeout))s deadline"))
            }
        }

        // Fenced frees: rows finished/cancelled while this step was in
        // flight. Scrub the wasted-token KV tail, then retire (donate to
        // the prefix cache when eligible, else release).
        for (_, state, rollbackOne, donation) in step.deferredReleases {
            if rollbackOne {
                for sequence in state { sequence?.rollback(1) }
            }
            retire(state: state, donating: donation)
        }
    }

    // MARK: Request completion

    private func finishRequest(_ id: CBv2RequestID, reason: CBv2FinishReason) {
        // Ids are legally reusable after finish: drop the per-id capacity
        // requeue count on EVERY finish path (including the error-finish
        // that exhausted it), or a reused id inherits the previous
        // attempt tally and its first transient KV-capacity trip
        // error-finishes immediately instead of requeueing (PR#62 review).
        capacityRequeues.removeValue(forKey: id)
        multimodalByID.removeValue(forKey: id)
        guard let rec = scheduler.finish(id: id, reason: reason) else {
            // Unknown to the scheduler (already finished) — make sure no
            // stream leaks regardless.
            prefixHitTokens.removeValue(forKey: id)
            takeStream(id)?.finish(
                reason: reason, usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
            return
        }
        capacity?.releaseAll(id: id)
        // Retire the id from the sampler's configured fingerprint: the id is
        // legally reusable after this finish, and a reused id with an
        // identical row set must reconfigure, not inherit stale penalties /
        // RNG progress (PR#62 review).
        sampler.requestDidFinish(id)

        if let state = kvStates.removeValue(forKey: id) {
            compiledDecode?.forgetRows(state)
            let donation = donationIntent(for: rec, reason: reason, state: state)
            if let inFlight, inFlight.participants.contains(id) {
                // The in-flight step still references this state — fence the
                // free behind its completion; roll back the wasted token iff
                // that step sampled for this row.
                inFlight.discard.insert(id)
                inFlight.deferredReleases.append(
                    (
                        id: id, state: state,
                        rollbackOne: inFlight.sampledRows.contains(id),
                        donation: donation
                    ))
            } else {
                retire(state: state, donating: donation)
            }
        }

        let detokenizer = detokenizers.removeValue(forKey: id)
        let stream = takeStream(id)
        let usage = CBv2Usage(
            promptTokens: rec.request.promptTokens.count,
            completionTokens: rec.generatedTokenCount,
            prefixCacheHitTokens: prefixHitTokens.removeValue(forKey: id) ?? 0)

        // Passthrough requests emit deltas on the detok queue; the trailing
        // flush + terminal MUST ride the same queue so they land AFTER those
        // deltas (FIFO ordering). Stop-string requests stay synchronous.
        if rec.request.stopStrings.isEmpty {
            detokQueue.async {
                let trailing = detokenizer?.flush() ?? ""
                if !trailing.isEmpty {
                    stream?.emit(.delta(text: trailing, tokens: [], logprobs: nil))
                }
                stream?.finish(reason: reason, usage: usage)
            }
        } else {
            let trailing = detokenizer?.flush() ?? ""
            if !trailing.isEmpty {
                stream?.emit(.delta(text: trailing, tokens: [], logprobs: nil))
            }
            stream?.finish(reason: reason, usage: usage)
        }
    }

    // MARK: Prefix-cache donation (engine thread → donation queue)

    /// Donation intent for a finished request, or nil when donation does
    /// not apply. Only NATURAL completions donate: their KV is a complete,
    /// confirmed prefix. The last confirmed token was sampled but never fed
    /// through the model, so it is dropped at donation time. The intent
    /// carries the request's cache salt so the entry lands in the donor's
    /// salt scope (TB-007).
    private func donationIntent(
        for rec: CBv2ScheduledRequest, reason: CBv2FinishReason, state: [CBv2SequenceKV?]
    ) -> CBv2DonationIntent? {
        guard prefixCache != nil else { return nil }
        // Vision requests NEVER donate (v1 policy, enforced in BOTH
        // directions — lookup is skipped in `EngineV2.makeAdoption`): the
        // prefix cache keys on token-id chain hashes, and an image span's
        // placeholder ids are identical for every image — a donated vision
        // prefix would be silently served to a request with different image
        // content. Image-digest extra keys in the block hash (vLLM-V1
        // `extra_keys`) are the documented follow-up.
        guard rec.request.multimodal == nil else { return nil }
        switch reason {
        case .stop, .length: break
        case .cancelled, .error: return nil
        }
        // Donation requires the full prompt to have been processed (at
        // least one sampled token) — mid-prefill finishes carry partial KV.
        guard rec.generatedTokenCount >= 1, rec.tokens.count > 1 else { return nil }
        // Lossy-snapshot rows (quantized KV) never donate: their snapshot
        // dequantizes, and MLX affine re-quantization is not idempotent —
        // a donate→adopt round trip would drift the adopted KV off the
        // cold-run values, compounding with each generation. Explicit,
        // documented skip (see CBv2SequenceKV.snapshotIsLossless); adoption
        // of full-precision donations INTO quantized backends stays legal.
        for (i, kind) in layerKinds.enumerated() {
            var cacheable = kind.sharesKVWithLayer == nil
            if case .slidingWindow = kind.attention { cacheable = false }
            guard cacheable, let sequence = state[i] else { continue }
            guard sequence.snapshotIsLossless else { return nil }
        }
        return CBv2DonationIntent(tokens: rec.tokens, cacheSalt: rec.request.cacheSalt)
    }

    /// Retire a finished request's KV state: donate to the prefix cache
    /// (snapshots graph-built HERE on the engine thread, so views over
    /// shared paged slabs are consistent with in-flight writes; hashing +
    /// indexing + optional materialization run on the donation queue), then
    /// release the storage back on the engine queue (the paged pool is
    /// engine-thread-affine).
    private func retire(state: [CBv2SequenceKV?], donating donation: CBv2DonationIntent?) {
        guard let prefixCache, let donation else {
            backend.release(state)
            return
        }
        let donated = Array(donation.tokens.dropLast())
        let cacheSalt = donation.cacheSalt
        var built: [(keys: MLXArray, values: MLXArray, offset: Int)?] = []
        built.reserveCapacity(layerKinds.count)
        for (i, kind) in layerKinds.enumerated() {
            var cacheable = kind.sharesKVWithLayer == nil
            if case .slidingWindow = kind.attention { cacheable = false }
            guard cacheable, let seq = state[i] else {
                built.append(nil)
                continue
            }
            built.append(seq.snapshot())
        }
        let layerKinds = self.layerKinds
        let handoff = CBv2Handoff(value: (state: state, snapshots: built))
        // Strong self on purpose: the deferred release is a pending
        // obligation of this loop — it must survive until the donation
        // lands, or the retired state leaks its pages (the paged pool is
        // engine-thread-affine, so the free hops back to the engine queue).
        // No cycle: the block releases its captures once it runs.
        donationQueue.async {
            prefixCache.donate(
                tokens: donated, snapshots: handoff.value.snapshots, layerKinds: layerKinds,
                cacheSalt: cacheSalt)
            self.releaseOnEngineQueue(handoff.value.state)
        }
    }

    private func releaseOnEngineQueue(_ state: [CBv2SequenceKV?]) {
        let handoff = CBv2Handoff(value: state)
        engineQueue.async { [self] in backend.release(handoff.value) }
    }

    // MARK: Sampler row context

    /// Full per-row sampler context (confirmed history only; in-flight
    /// chained samples travel separately as `pendingSampledTokens`).
    private static func samplerRow(_ rec: CBv2ScheduledRequest) -> CBv2SamplerRow {
        CBv2SamplerRow(
            id: rec.id,
            params: rec.request.sampling,
            promptTokens: rec.request.promptTokens,
            outputTokens: Array(rec.tokens.dropFirst(rec.request.promptTokens.count)))
    }

    // MARK: Boundary housekeeping

    private func processCancellations() {
        stateLock.lock()
        let cancels = pendingCancels
        stateLock.unlock()

        // Which cancels are fully resolved this boundary (drop from the set).
        // A cancel with no scheduler record AND a still-registered stream is
        // an EARLY cancel racing `enqueue` — keep it so the enqueue block
        // refuses to start the request (PR#62 review). A cancel with neither
        // is stale (request already finished) — drop it.
        var resolved: Set<CBv2RequestID> = []
        for id in cancels {
            if scheduler.record(for: id) != nil {
                finishRequest(id, reason: .cancelled)
                resolved.insert(id)
            } else if !hasRegisteredStream(id) {
                resolved.insert(id)
            }
        }
        guard !resolved.isEmpty else { return }
        stateLock.lock()
        pendingCancels.subtract(resolved)
        stateLock.unlock()
    }

    /// True when a stream is still registered for `id` (used to tell a
    /// racing pre-enqueue cancel from a stale post-finish one).
    private func hasRegisteredStream(_ id: CBv2RequestID) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return streams[id] != nil
    }

    /// Consume a remembered early cancel for `id` at enqueue time. Returns
    /// true when the request was cancelled before it started (the caller
    /// must finish its stream and never enqueue it).
    private func consumeEarlyCancel(_ id: CBv2RequestID) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return pendingCancels.remove(id) != nil
    }

    private func processDeadlines() {
        let now = Date()
        var overdue: [CBv2RequestID] = []
        for rec in scheduler.running where rec.deadline.map({ now > $0 }) == true {
            overdue.append(rec.id)
        }
        for rec in scheduler.waiting where rec.deadline.map({ now > $0 }) == true {
            overdue.append(rec.id)
        }
        for id in overdue {
            finishRequest(
                id, reason: .error("request exceeded \(Int(config.requestTimeout))s deadline"))
        }
    }

    private func handlePreemptions(_ ids: [CBv2RequestID]) {
        // Preemption only happens on the non-chained path, where the
        // previous step was finalized first — no in-flight step can
        // reference these states, so the release is immediate. The
        // scheduler already released the capacity reservations and requeued
        // the victims (generated tokens kept).
        assert(inFlight == nil, "preemption with a step in flight")
        for id in ids {
            preemptionCount += 1
            // A preempted request recomputes from scratch — any adopted
            // prefix credit no longer describes work that was skipped, so
            // usage.prefixCacheHitTokens must not over-credit at finish.
            prefixHitTokens.removeValue(forKey: id)
            if let state = kvStates.removeValue(forKey: id) {
                compiledDecode?.forgetRows(state)
                backend.release(state)
            }
        }
    }

    /// Create per-layer KV state on first execution. On `capacityExhausted`
    /// (several same-step admissions racing for the last bytes/pages — the
    /// backend's charge is atomic, so losers surface here) the request is
    /// requeued to waiting instead of error-finished: accepted requests wait
    /// for room. Bounded by `maxCapacityRequeues` (then error-finish) and by
    /// the request deadline. Other failures error-finish as before.
    private func ensureKVState(_ rec: CBv2ScheduledRequest) -> [CBv2SequenceKV?]? {
        if let state = kvStates[rec.id] { return state }
        do {
            let maxLength = rec.request.promptTokens.count + max(rec.request.maxTokens, 1)
            let state = try backend.makeSequenceState(
                layerKinds: layerKinds, promptLength: rec.tokens.count, maxLength: maxLength)
            kvStates[rec.id] = state
            capacityRequeues.removeValue(forKey: rec.id)
            return state
        } catch let kvError as CBv2KVError {
            if case .capacityExhausted = kvError {
                let attempts = capacityRequeues[rec.id, default: 0]
                if attempts < Self.maxCapacityRequeues, scheduler.requeueOnCapacity(rec.id) {
                    capacityRequeues[rec.id] = attempts + 1
                    capacityRequeueCount += 1
                    return nil
                }
            }
            finishRequest(rec.id, reason: .error("KV allocation failed: \(kvError)"))
            return nil
        } catch {
            finishRequest(rec.id, reason: .error("KV allocation failed: \(error)"))
            return nil
        }
    }

    // MARK: Streams

    private func stream(for id: CBv2RequestID) -> CBv2OutputStream? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return streams[id]
    }

    private func takeStream(_ id: CBv2RequestID) -> CBv2OutputStream? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return streams.removeValue(forKey: id)
    }

    // MARK: Test/telemetry introspection

    /// Engine-queue-synchronized snapshot of paused (backpressured) rows.
    /// Blocks until the current step boundary; test/telemetry use only.
    func pausedIDsSnapshot() -> Set<CBv2RequestID> {
        engineQueue.sync { Set(scheduler.running.filter(\.isPaused).map(\.id)) }
    }

    // MARK: Gauges

    private func publishGauges() {
        gauges.update(
            CBv2CapacitySnapshot(
                activeRequests: scheduler.runningCount,
                waitingRequests: scheduler.waitingCount,
                kvBytesInUse: backend.bytesInUse,
                kvBytesCapacity: backend.bytesCapacity,
                activeTokens: scheduler.activeTokens,
                stepsExecuted: stepCount))
    }

    // MARK: Watchdog (engine health signal)

    private var watchdogTimer: DispatchSourceTimer?

    private func markStepStarted() {
        stateLock.lock()
        stepStartedNanos = DispatchTime.now().uptimeNanoseconds
        stateLock.unlock()
    }

    private func markStepEnded() {
        stateLock.lock()
        stepStartedNanos = 0
        wedgeReported = false
        _healthy = true
        stateLock.unlock()
    }

    deinit {
        watchdogTimer?.cancel()
    }

    private func startWatchdog() {
        let timer = DispatchSource.makeTimerSource(queue: watchdogQueue)
        timer.schedule(
            deadline: .now() + config.watchdogInterval, repeating: config.watchdogInterval)
        timer.setEventHandler { [weak self] in self?.watchdogTick() }
        timer.resume()
        watchdogTimer = timer
    }

    private func stopWatchdog() {
        watchdogTimer?.cancel()
        watchdogTimer = nil
    }

    /// Runs on the watchdog queue. The engine thread may be blocked inside a
    /// wedged eval, so this touches ONLY lock-protected state and the
    /// (thread-safe) streams: consumers get a timely error, every live
    /// request is marked for cancellation (cleaned up if the loop ever
    /// resumes), and the health signal flips for provider heartbeats.
    private func watchdogTick() {
        stateLock.lock()
        let started = stepStartedNanos
        guard started != 0 else {
            stateLock.unlock()
            return
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds &- started) / 1_000_000_000
        guard elapsed > config.stepTimeout, !wedgeReported else {
            stateLock.unlock()
            return
        }
        wedgeReported = true
        _healthy = false
        let liveStreams = streams
        pendingCancels.formUnion(liveStreams.keys)
        stateLock.unlock()

        // Signal BEFORE erroring the streams: a consumer woken by the
        // terminal event must be able to observe the wedge side effects
        // (health metric, telemetry) immediately.
        onStepWedge?(elapsed)
        for (_, stream) in liveStreams {
            stream.finish(
                reason: .error(
                    "engine step exceeded \(Int(config.stepTimeout))s watchdog"),
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
        }
    }
}
