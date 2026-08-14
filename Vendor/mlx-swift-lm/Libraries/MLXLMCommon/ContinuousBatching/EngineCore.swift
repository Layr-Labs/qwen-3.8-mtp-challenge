// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/engine_core.py — Async engine loop for continuous batching.
// https://github.com/jundot/omlx/blob/main/omlx/engine_core.py
//
// The EngineCore runs the scheduler step loop on a background dispatch queue,
// while the high-level interface (BatchedEngine) provides the async API.
//
// Thread-safety model (mirrors omlx's single-executor design):
//   omlx routes ALL MLX + scheduler work through a single-worker ThreadPoolExecutor,
//   making dict access effectively single-threaded.  Here we replicate that by
//   running all dict mutations (add, cleanup) on `engineQueue` and protecting
//   reads from external callers with `_lock`.  The engine loop output distribution
//   also runs inside the engineQueue block so it is serialised with addRequest
//   and cleanupRequest.

import Foundation
import MLX
import os

// MARK: - EngineConfig

/// Configuration for the continuous-batching engine.
public struct ContinuousBatchingConfig: Sendable {
    public var schedulerConfig: SchedulerConfig
    /// Interval (in seconds) between engine loop iterations when idle.
    public var stepInterval: TimeInterval
    /// How often (in steps) to yield to the event loop.
    public var yieldInterval: Int
    /// Optional in-memory prefix cache configuration.
    /// Set to non-nil to enable block-level KV reuse across requests.
    public var prefixCacheConfig: PrefixCacheConfig?
    /// Enable Multi-Token Prediction (MTP) speculative decoding for compatible models.
    /// When true, the model loader should attach the MTP head (e.g. set `_qwen35MTPEnabled`
    /// before calling `MLXLLM.load`). Only affects single-sequence batches.
    /// Port of omlx commit 696d90a: patches/mlx_lm_mtp/qwen35_model.py is_mtp_active()
    public var mtpEnabled: Bool
    public init(
        schedulerConfig: SchedulerConfig = SchedulerConfig(),
        stepInterval: TimeInterval = 0.001,
        yieldInterval: Int = 5,
        prefixCacheConfig: PrefixCacheConfig? = nil,
        mtpEnabled: Bool = false
    ) {
        self.schedulerConfig = schedulerConfig
        self.stepInterval = stepInterval
        self.yieldInterval = yieldInterval
        self.prefixCacheConfig = prefixCacheConfig
        self.mtpEnabled = mtpEnabled
    }
}

// MARK: - EngineCore

/// Core engine for continuous batching.
///
/// This engine runs the generation loop and manages request lifecycle.
/// It provides both sync and async interfaces for request handling.
///
/// Design mirrors omlx's EngineCore:
/// - Scheduler manages request lifecycle and BatchGenerator interaction
/// - EngineCore owns the async loop, output collectors, and streaming
/// - BatchedEngine provides the high-level generate/stream API
public final class EngineCore: @unchecked Sendable {
    public let scheduler: Scheduler
    public let config: ContinuousBatchingConfig

    private let engineQueue = DispatchQueue(
        label: "com.eigen.engine",
        qos: .userInitiated
    )

    // Output collectors for streaming (vLLM pattern).
    // Mutations happen ONLY on engineQueue; reads from external contexts
    // must hold _lock.  The engine queue operations also hold _lock so that
    // external reads never race with a concurrent engine-queue write.
    private var outputCollectors: [String: RequestOutputCollector] = [:]
    private var streamStates: [String: RequestStreamState] = [:]
    private let _lock = NSLock()

    // Engine state
    private var _running = false
    // Set true by stop(); never reset. Distinct from `!_running`, which is also
    // true for a freshly-constructed engine that hasn't started yet (the
    // add-before-start pattern some callers/tests rely on). `addRequest` rejects
    // enqueues only when the engine was actually STOPPED, not merely not-started.
    private var _stopped = false
    // Engine loop runs purely on engineQueue (GCD self-rescheduling).
    // No Task, no cooperative pool involvement — eliminates the per-step
    // cooperative-pool → engineQueue → cooperative-pool round-trip that
    // starved the engine under concurrent consumer pipeline load.
    private var _startTime: Date?
    public private(set) var stepsExecuted: Int = 0

    // Idle-clear instrumentation (MEASUREMENT ONLY) — the `Stream().synchronize()`
    // + `Memory.clearCache()` hand-off is the M4 kernel-panic race flagged below
    // (candidate #1 for the first-token wedge: it runs right after load, and a
    // lost race vs IOKit's async `completeMemory()` can leave the Metal queue
    // stuck so the next forced `eval` wedges). These mark when a clear is in
    // flight so a clear that never exits pins the race as the trigger. Written
    // only on the engine loop / engineQueue (like `stepsExecuted`); read
    // lock-free from the heartbeat (benign diagnostic race).
    public private(set) var idleClearInFlight: Bool = false
    public private(set) var idleClearsCompleted: Int = 0
    private var idleClearStartedNanos: UInt64 = 0
    /// Milliseconds the current idle clear has been running, or 0 if none. A
    /// value in the seconds range with no matching exit is the clearCache wedge.
    public var idleClearElapsedMs: Int64 {
        guard idleClearInFlight, idleClearStartedNanos != 0 else { return 0 }
        let now = DispatchTime.now().uptimeNanoseconds
        guard now > idleClearStartedNanos else { return 0 }
        return Int64((now &- idleClearStartedNanos) / 1_000_000)
    }

    /// Run the GPU drain + buffer-cache flush with the idle-clear marker set, so
    /// a wedge inside `synchronize`/`clearCache` is visible (flag stays true,
    /// `idleClearElapsedMs` climbs). The two call sites (idle window + post-abort
    /// reclaim) share this so neither can wedge unobserved.
    private func synchronizeAndClearCache() {
        idleClearStartedNanos = DispatchTime.now().uptimeNanoseconds
        idleClearInFlight = true
        Stream().synchronize()
        Memory.clearCache()
        idleClearInFlight = false
        idleClearsCompleted += 1
    }

    // Steps spent idle after the last active request completed.
    // After `deferredClearDelay` idle steps we flush the Metal buffer cache
    // to reclaim GPU memory. The delay prevents races with IOKit's async
    // completeMemory() callbacks that cause kernel panics on M4 hardware.
    private var _idleSteps = 0
    private static let deferredClearDelay = 8
    // Set by the abort paths, consumed by the step loop (both on engineQueue):
    // requests a pool reclaim once the aborted rows have been filtered out.
    private var reclaimAfterStep = false
    // Coalesce post-abort reclaims so a burst of cancellations can't force a full
    // GPU sync (Stream().synchronize) on every step. The flag persists across
    // throttled steps, so no reclaim is dropped — only delayed by a few steps.
    private var lastReclaimStep = 0
    private static let reclaimMinStepGap = 4

    // Diagnostic: log the live Metal resource (buffer) COUNT vs the cache/active
    // BYTES on the batched-decode path. This distinguishes a cached-buffer count
    // climb (cache bytes track count -> a count-aware cache trim fixes it) from a
    // live-buffer climb (count climbs while cache stays small -> needs admission
    // control) — the latter is the `[metal::malloc] Resource limit (499000)` crash.
    //
    // DEFAULT OFF (opt-in): the `Memory.*` resource/byte reads in the step loop
    // (see `engineLoop`) sit on the per-step decode hot path, so they are gated
    // behind this flag and not evaluated at all unless an operator opts in.
    // Enable with DARKBLOOM_MLX_RESOURCE_DEBUG=1 (or true/yes/on) to collect the
    // `[rsrc]` trajectory; `darkbloom report` then captures the os_log line via
    // `log show --predicate 'subsystem == "dev.darkbloom.provider"'`. The
    // capability is unchanged — it simply costs nothing when disabled, which is
    // the default.
    private static let resourceDebug: Bool = {
        switch ProcessInfo.processInfo.environment["DARKBLOOM_MLX_RESOURCE_DEBUG"]?
            .lowercased()
        {
        case "1", "true", "yes", "on": return true
        default: return false
        }
    }()
    // Console cadence (stdout). Frequent, so the console line is TTY-gated (see
    // `stdoutIsTTY`) to keep it out of the unbounded daemon log.
    private static let resourceDebugEverySteps = 50
    // stdout is a real terminal only for interactive/foreground runs. Under
    // launchd the provider's stdout/stderr are redirected to
    // ~/.darkbloom/provider.log (which is NOT rotated), so printing the `[rsrc]`
    // line every ~50 steps on an always-on daemon would grow that log without
    // bound — and the operator can't disable it if the opt-out env var isn't
    // forwarded into the service. Gate the console print on a TTY: interactive
    // debugging still sees it, the daemon never does. The os_log emit below is
    // always on regardless and is what `darkbloom report` actually reads.
    private static let stdoutIsTTY = isatty(STDOUT_FILENO) != 0
    // Report cadence (os_log). Coarser so a 24h report stays well under the 10 MB
    // upload cap (~every 1000 busy steps ≈ tens of seconds), but we ALSO emit
    // whenever pressure is high (>=70% of the limit) to always capture the
    // dangerous ramp toward 499000.
    private static let resourceLogEverySteps = 1000
    private static let resourcePressurePct = 70.0
    // Fine cadence at which the high-pressure check is evaluated. Must divide both
    // coarse cadences above so their sampling is unchanged; finer than the coarse
    // sample so a fast ramp from >=70% to the limit is caught between coarse
    // samples (the window this telemetry exists to diagnose).
    private static let resourcePressureEverySteps = 10
    /// os_log channel for the `[rsrc]` line, under the subsystem that
    /// `darkbloom report` captures.
    private static let resourceLogger = Logger(
        subsystem: "dev.darkbloom.provider", category: "rsrc")

    public init(
        scheduler: Scheduler,
        config: ContinuousBatchingConfig = ContinuousBatchingConfig()
    ) {
        self.scheduler = scheduler
        self.config = config
    }

    // MARK: - Lifecycle

    /// Start the engine loop.
    ///
    /// The loop runs entirely on `engineQueue` via GCD self-rescheduling.
    /// Each step dispatches the next step directly with `engineQueue.async`,
    /// keeping the GPU pipeline full without ever touching the cooperative
    /// thread pool. This eliminates the ~3-8ms per-step scheduling gap that
    /// halved effective decode TPS under the old `Task`-based loop.
    public func start() {
        guard !_running else { return }
        _running = true
        _startTime = Date()
        engineQueue.async { [weak self] in self?.engineStep() }
    }

    /// Stop the engine loop.
    public func stop() {
        _running = false
        _stopped = true
        // No Task to cancel — the next engineStep() invocation sees
        // _running == false and returns without rescheduling.
    }

    /// Stop the loop and wait for it to fully exit, so the caller can safely
    /// release the engine and reclaim its MLX buffers. Unlike `stop()`, this
    /// drains the engine queue: once it returns, the in-flight step has finished
    /// and no more steps will be enqueued. A final queue drain then flushes any
    /// abort/cleanup blocks the loop left behind. This is what makes a subsequent
    /// `Memory.clearCache()` both safe (no step's MLX work runs concurrently —
    /// the IOKit completeMemory race seen on M4) and effective (the engine chain,
    /// and thus the batch KV, can actually be released).
    public func stopAndWait() async {
        _running = false
        _stopped = true
        // Drain the engine queue — the in-flight engineStep() will see
        // _running == false and return without rescheduling. This block
        // runs after that step completes (serial queue).
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            engineQueue.async { continuation.resume() }
        }
    }

    public var isRunning: Bool { _running }

    // MARK: - Request Management

    /// Add a request for processing.
    ///
    /// Dict setup and scheduler submission both happen on `engineQueue` so that
    /// state mutations are serialised with the step loop — mirroring omlx's
    /// `loop.run_in_executor(self._mlx_executor, self.scheduler.add_request, request)`.
    @discardableResult
    public func addRequest(
        _ request: Request
    ) async -> String {
        let rid = request.requestId
        await withCheckedContinuation { continuation in
            engineQueue.async { [weak self] in
                guard let self else { continuation.resume(); return }
                _lock.lock()
                // Reject enqueues after stop(): the step loop is gone, so a
                // request added now would never run and its stream would hang.
                // Gate on `_stopped` (set only by stop()), NOT `!_running` — a
                // never-started engine also has `_running == false` but legitimately
                // accepts add-before-start (the request waits for start()/abort).
                // Register a collector and put a terminal output so any
                // `streamOutputs(rid)` consumer unblocks cleanly, but do NOT call
                // `scheduler.addRequest` (nothing will step it).
                guard !_stopped else {
                    let collector = RequestOutputCollector(aggregate: true)
                    outputCollectors[rid] = collector
                    _lock.unlock()
                    collector.put(RequestOutput(
                        requestId: rid,
                        finished: true,
                        finishReason: "abort",
                        error: "Engine stopped"
                    ))
                    continuation.resume()
                    return
                }
                outputCollectors[rid] = RequestOutputCollector(aggregate: true)
                streamStates[rid] = RequestStreamState(
                    streamInterval: config.schedulerConfig.streamInterval
                )
                _lock.unlock()
                scheduler.addRequest(request)
                continuation.resume()
            }
        }
        return rid
    }

    /// Abort a request.
    ///
    /// Signals the consumer immediately (the collector is internally locked),
    /// then defers the scheduler abort to `engineQueue` to avoid racing
    /// scheduler state — matching omlx's deferred-abort comment.
    public func abortRequest(_ requestId: String) -> Bool {
        _lock.lock()
        let collector = outputCollectors[requestId]
        _lock.unlock()
        guard let collector else { return false }

        collector.put(RequestOutput(
            requestId: requestId,
            finished: true,
            finishReason: "abort",
            error: "Request aborted"
        ))

        engineQueue.async { [weak self] in
            guard let self else { return }
            _ = scheduler.abortRequest(requestId)
            reclaimAfterStep = true
        }
        return true
    }

    /// Update the scheduler's runtime concurrency cap on the engine queue.
    ///
    /// Hops onto `engineQueue` so the mutation is serialised with the step
    /// loop (mirrors `addRequest` / `abortRequest`). The new value takes
    /// effect on the next `admitWaiting()` call.
    public func setMaxNumSeqs(_ value: Int) {
        engineQueue.async { [weak self] in
            self?.scheduler.setMaxNumSeqs(value)
        }
    }

    /// Abort all active requests.
    @discardableResult
    public func abortAllRequests() -> Int {
        _lock.lock()
        let rids = Array(outputCollectors.keys)
        _lock.unlock()

        for rid in rids {
            engineQueue.async { [weak self] in
                guard let self else { return }
                _ = scheduler.abortRequest(rid)
                reclaimAfterStep = true
            }
            _lock.lock()
            let collector = outputCollectors[rid]
            _lock.unlock()
            collector?.put(RequestOutput(
                requestId: rid,
                finished: true,
                finishReason: "error",
                error: "All requests aborted"
            ))
        }
        return rids.count
    }

    // MARK: - Output Streaming

    /// Stream outputs for a request using non-blocking pattern.
    public func streamOutputs(
        requestId: String,
        timeout: TimeInterval? = nil
    ) -> AsyncStream<RequestOutput> {
        _lock.lock()
        let collector = outputCollectors[requestId]
        _lock.unlock()

        return AsyncStream { continuation in
            Task {
                guard let collector else {
                    continuation.finish()
                    return
                }

                while true {
                    if let output = collector.getNowait() {
                        continuation.yield(output)
                        if output.finished || output.error != nil {
                            break
                        }
                    } else {
                        let output = await collector.get()
                        continuation.yield(output)
                        if output.finished || output.error != nil {
                            break
                        }
                    }
                }

                cleanupRequest(requestId)
                continuation.finish()
            }
        }
    }

    /// Generate a complete response (non-streaming).
    ///
    /// Reuses `streamOutputs` rather than a blocking semaphore wait on
    /// `engineQueue`, eliminating the thread-stall / potential deadlock from
    /// the original `DispatchSemaphore.wait()` design.
    public func generate(
        prompt: String,
        samplingParams: SamplingParams
    ) async throws -> RequestOutput {
        let request = Request(
            requestId: UUID().uuidString,
            prompt: prompt,
            samplingParams: samplingParams
        )

        let rid = await addRequest(request)

        return try await withTaskCancellationHandler {
            for await output in streamOutputs(requestId: rid) {
                if output.finished || output.error != nil {
                    if let error = output.error {
                        throw EngineError.generationFailed(error)
                    }
                    return output
                }
            }
            throw EngineError.missingOutput
        } onCancel: { [weak self] in
            _ = self?.abortRequest(rid)
        }
    }

    // MARK: - Engine Loop

    /// Main engine loop — runs scheduler steps on the engine queue.
    ///
    /// Output distribution runs INSIDE the `engineQueue.async` block so that
    /// all `outputCollectors` access is serialised with `addRequest` and
    /// `cleanupRequest`, matching omlx's single-executor model.
    /// Single decode step, self-rescheduling on `engineQueue`.
    ///
    /// Runs entirely on `engineQueue` — no cooperative pool involvement.
    /// Each invocation either executes one `scheduler.step()` and dispatches
    /// the next step immediately via `engineQueue.async`, or (when idle)
    /// schedules a delayed re-check via `asyncAfter`. The GPU pipeline stays
    /// full with only GCD dispatch overhead (~1-5μs) between steps instead
    /// of the ~3-8ms cooperative pool scheduling gap.
    private func engineStep() {
        guard _running else { return }

        if scheduler.hasRequests() {
            _idleSteps = 0
            let outerStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let output = scheduler.step()
            if CBv2StepProfiler.enabled {
                CBv2StepProfiler.record(
                    "leg.schedstep.wall", seconds: CFAbsoluteTimeGetCurrent() - outerStart)
            }
            stepsExecuted += 1

            // Diagnostic (default-on): observe resource COUNT vs BYTES
            // on the real batched-decode path to classify the
            // [metal::malloc] Resource limit crash as cached vs live.
            // Sample on the fine pressure cadence (which divides the
            // coarse one) so the high-pressure check is NOT gated by the
            // coarse window: resources can cross 70% and hit the limit
            // between coarse samples, and that ramp is exactly what this
            // telemetry must capture.
            if Self.resourceDebug, stepsExecuted % Self.resourcePressureEverySteps == 0 {
                let res = Memory.numResources
                let lim = Memory.resourceLimit
                let pct = lim > 0 ? Double(res) / Double(lim) * 100 : 0
                let highPressure = pct >= Self.resourcePressurePct
                let coarseSample = stepsExecuted % Self.resourceDebugEverySteps == 0
                // Emit the line on the coarse cadence, or on every fine
                // sample while under pressure (dense ramp, still bounded).
                if coarseSample || highPressure {
                    let batch = output.outputs.count
                    let line = String(
                        format: "[rsrc] step=%d batch=%d resources=%d/%d (%.1f%%) cache=%.0fMB active=%.0fMB",
                        stepsExecuted, batch, res, lim, pct,
                        Double(Memory.cacheMemory) / 1_048_576,
                        Double(Memory.activeMemory) / 1_048_576)
                    if Self.stdoutIsTTY {
                        print(line)
                        fflush(stdout)
                    }
                    // os_log (captured by `darkbloom report`) on the coarse
                    // report cadence, plus every fine sample while pressure
                    // is high so the climb toward 499000 is never missed.
                    if stepsExecuted % Self.resourceLogEverySteps == 0 || highPressure {
                        Self.resourceLogger.log("\(line, privacy: .public)")
                    }
                }
            }

            // Distribute outputs to collectors inside the queue block.
            if !output.outputs.isEmpty {
                for reqOutput in output.outputs {
                    _lock.lock()
                    let collector = outputCollectors[reqOutput.requestId]
                    _lock.unlock()
                    collector?.put(reqOutput)
                }
            }

            if reclaimAfterStep,
                stepsExecuted - lastReclaimStep >= Self.reclaimMinStepGap {
                reclaimAfterStep = false
                lastReclaimStep = stepsExecuted
                // scheduler.step() filtered the aborted rows at its start,
                // so their KV is now in the reclaimable pool. Return it to
                // the OS here rather than waiting for the idle clear, which
                // never fires under sustained load. The step gate above
                // coalesces bursts so cancel churn can't sync every step.
                synchronizeAndClearCache()
            }

            // Next step immediately — zero scheduling gap.
            engineQueue.async { [weak self] in self?.engineStep() }
        } else {
            _idleSteps += 1
            // Flush the Metal buffer cache after a brief idle window to
            // reclaim GPU memory. The delay avoids races with IOKit's
            // async completeMemory() callbacks (M4 kernel-panic fix).
            if _idleSteps == Self.deferredClearDelay {
                synchronizeAndClearCache()
            }
            // Idle: re-check after stepInterval.
            let intervalNs = config.stepInterval
            engineQueue.asyncAfter(
                deadline: .now() + intervalNs
            ) { [weak self] in
                self?.engineStep()
            }
        }
    }

    // MARK: - Cleanup

    /// Dispatch cleanup to `engineQueue` so dict mutations are serialised
    /// with the step loop — the caller (streamOutputs Task) may run on any thread.
    private func cleanupRequest(_ requestId: String) {
        engineQueue.async { [weak self] in
            guard let self else { return }
            scheduler.removeFinishedRequest(requestId)
            _lock.lock()
            outputCollectors.removeValue(forKey: requestId)
            streamStates.removeValue(forKey: requestId)
            _lock.unlock()
        }
    }

    // MARK: - Stats

    public func getStats() -> [String: Any] {
        _lock.lock()
        let activeRequests = outputCollectors.count
        _lock.unlock()
        var stats: [String: Any] = [
            "running": _running,
            "steps_executed": stepsExecuted,
            "active_requests": activeRequests,
        ]
        stats.merge(scheduler.getStats()) { $1 }
        return stats
    }
}

// MARK: - Errors

public enum EngineError: Error {
    case missingOutput
    case generationFailed(String)
}
