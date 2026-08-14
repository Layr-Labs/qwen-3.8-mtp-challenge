// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: unified token-budget scheduler (vLLM-V1 style).
//
// There is no "prefill phase" and no "decode phase". Each request carries a
// token list (prompt + confirmed generated tokens) and `numComputedTokens`
// (tokens fed through the model). At every step `plan()` assigns each request
// `min(remaining, budget)` new tokens so `numComputedTokens` catches up to
// the number of known tokens; decode rows naturally request exactly 1.
// (Reference: vLLM v1 scheduler.py:395-404 — see research report 08 §1.)
//
// This file is PURE, SYNCHRONOUS bookkeeping: no MLX imports, no arrays, no
// I/O. It is fully unit-testable without model weights. All methods must be
// called from a single thread (the engine thread); the engine serializes.

import Foundation

// MARK: - Scheduler errors

/// Submit-side scheduler contract violations (distinct from
/// `CBv2KVError.capacityExhausted`, which is transient back-off).
public enum CBv2SchedulerError: Error, Equatable {
    /// A request with this id is already live (waiting or running). Request
    /// ids are engine-scoped: reusing one is only legal after the previous
    /// request carrying it has fully finished. Without this rejection a
    /// duplicate would silently clobber the live record in `byID` while the
    /// stale record kept its queue slot — orphaning the first request's
    /// bookkeeping. The provider already guards against duplicate ids; the
    /// engine now enforces it (PR#62 review).
    case duplicateRequestID(CBv2RequestID)
}

// MARK: - Per-request scheduling record

/// Book-keeping for one request inside the v2 scheduler.
///
/// Token accounting (the vLLM optimistic-advance model):
/// - `tokens` — prompt + CONFIRMED generated token values (host-visible).
/// - `pendingSamples` — samples launched but not yet confirmed (deferred stop
///   detection inspects tokens one step late; chained decode keeps them lazy).
/// - `numComputedTokens` — tokens fed through the model, advanced
///   OPTIMISTICALLY at plan time and rolled back on failure/rejection.
public final class CBv2ScheduledRequest {
    public let request: CBv2Request
    /// Monotonic admission sequence — FCFS tie-break within a priority class.
    public let arrivalSeq: UInt64
    public let submittedAt: Date
    /// Absolute wall-clock deadline; the engine error-finishes past this.
    public let deadline: Date?

    /// Prompt + confirmed generated tokens.
    public internal(set) var tokens: [Int]
    /// Tokens fed through the model (optimistically advanced at plan time).
    public internal(set) var numComputedTokens: Int = 0
    /// Samples launched but not yet confirmed on the host.
    public internal(set) var pendingSamples: Int = 0
    public internal(set) var status: CBv2RequestStatus = .waiting
    /// Backpressure: slot retained, scheduling skipped until the consumer
    /// drains its event stream.
    public internal(set) var isPaused: Bool = false
    /// Set by the engine's cancel path; the row is dropped at the next step
    /// boundary. `plan()` never assigns work to a cancel-pending row.
    public internal(set) var cancelRequested: Bool = false
    /// Times this request was preempted (telemetry).
    public internal(set) var preemptionCount: Int = 0

    public var id: CBv2RequestID { request.id }
    public var numTokens: Int { tokens.count }
    /// Known + in-flight tokens: what `numComputedTokens` catches up to.
    public var effectiveTokenCount: Int { tokens.count + pendingSamples }
    public var generatedTokenCount: Int { tokens.count - request.promptTokens.count }
    /// A decode row: exactly one un-computed token remains (its next input).
    public var isDecodeReady: Bool { effectiveTokenCount - numComputedTokens == 1 }
    var remainingTokens: Int { effectiveTokenCount - numComputedTokens }

    /// Coalesced multimodal mask blocks (absolute prompt positions; adjacent
    /// image spans merge — they attend bidirectionally as one block). Empty
    /// for text requests. Chunk planning must NEVER split one of these
    /// across prefill chunks (`snappedChunkTokens`); the whole block's
    /// bidirectional attention needs all of its keys in one forward.
    public let multimodalBlocks: [CBv2ImageSpan]

    init(request: CBv2Request, arrivalSeq: UInt64, submittedAt: Date, deadline: Date?) {
        self.request = request
        self.arrivalSeq = arrivalSeq
        self.submittedAt = submittedAt
        self.deadline = deadline
        self.tokens = request.promptTokens
        self.multimodalBlocks = CBv2MultimodalPlan.coalescedBlocks(
            spans: request.multimodal?.spans ?? [])
    }

    /// Snap a proposed prefill chunk `[start, start + proposed)` so no
    /// multimodal block is split across chunk boundaries:
    ///  - a proposed end STRICTLY inside a block SHRINKS to the block's
    ///    start (the block rides a later chunk whole), unless the chunk
    ///    begins exactly at that block — then it EXTENDS to the block's end
    ///    (may exceed `prefillChunkSize`; that is the intended "snap over"),
    ///    bounded by the step's remaining token `budget`;
    ///  - returns 0 when the block cannot fit this step's remaining budget
    ///    (the row simply waits for a step with more headroom — submit-time
    ///    validation guarantees every block fits a FULL step budget).
    /// Text requests (no blocks) return `proposed` unchanged.
    func snappedChunkTokens(start: Int, proposed: Int, budget: Int) -> Int {
        guard proposed > 0, !multimodalBlocks.isEmpty else { return proposed }
        // Chunks never split blocks, so a chunk can never START strictly
        // inside one (preemption restarts at 0; adoption is excluded for
        // multimodal requests).
        assert(
            !multimodalBlocks.contains { start > $0.tokenOffset && start < $0.end },
            "CBv2 multimodal: chunk start \(start) lies inside a block — a previous chunk split it"
        )
        let end = start + proposed
        guard let block = multimodalBlocks.first(where: { $0.tokenOffset < end && end < $0.end })
        else { return proposed }
        if block.tokenOffset > start {
            return block.tokenOffset - start
        }
        // The chunk begins at the block's start: the whole block must ride
        // this chunk (blocks are maximal contiguous runs, so the extended
        // end lands on a block edge, never inside another block).
        let needed = block.end - start
        return needed <= budget ? needed : 0
    }
}

// MARK: - SchedulerV2

/// vLLM-V1-style scheduler: single token budget per step, RUNNING first in
/// order, WAITING admitted while budget and slots allow (chunked prefill =
/// partial token counts), optimistic advance with rollback, and preemption
/// (lowest priority / youngest victim, requeued front) as the capacity
/// backstop.
public final class SchedulerV2 {
    public let config: CBv2SchedulerConfig
    /// Soft KV capacity oracle (AdmissionV2). Optional so pure simulations
    /// can run without any capacity model.
    let capacity: CBv2StepCapacity?

    /// RUNNING requests, in admission order (plan preserves this order,
    /// with ONE exception: a row starved by the block-chunk guard is moved
    /// to the front — see `deferredBlockRequestID`).
    public private(set) var running: [CBv2ScheduledRequest] = []
    /// WAITING requests, sorted by (priority desc, arrivalSeq asc); preempted
    /// requests are re-inserted at the FRONT of their priority class.
    public private(set) var waiting: [CBv2ScheduledRequest] = []

    private var byID: [CBv2RequestID: CBv2ScheduledRequest] = [:]
    private var nextArrivalSeq: UInt64 = 0

    /// Starvation guard for block-sized vision chunks (PR#63 review): the id
    /// of a row whose next multimodal block fits a FULL step budget (submit
    /// validates that) but not the budget REMAINING after earlier rows were
    /// assigned this step. The NEXT `plan()` gives that row first claim on
    /// the fresh budget — a running row is moved to the front of `running`,
    /// a waiting row is admitted ahead of the running pass — because
    /// otherwise persistent earlier rows (e.g. long decodes at 1 token/step)
    /// can pin the remaining budget below the block size for the row's whole
    /// deadline and it never progresses.
    private var deferredBlockRequestID: CBv2RequestID?

    public init(config: CBv2SchedulerConfig, capacity: CBv2StepCapacity? = nil) {
        self.config = config
        self.capacity = capacity
    }

    // MARK: Queries

    public var waitingCount: Int { waiting.count }
    public var runningCount: Int { running.count }
    public var hasWork: Bool { !running.isEmpty || !waiting.isEmpty }
    /// Tokens known + in flight across running requests (capacity snapshot).
    public var activeTokens: Int { running.reduce(0) { $0 + $1.effectiveTokenCount } }

    public func record(for id: CBv2RequestID) -> CBv2ScheduledRequest? { byID[id] }

    // MARK: Submission

    /// Enqueue a new request. Throws `CBv2SchedulerError.duplicateRequestID`
    /// when a request with the same id is still live (waiting or running),
    /// and `capacityExhausted` when the waiting queue is full (`maxWaiting`).
    ///
    /// ORDER IS LOAD-BEARING: the duplicate check runs BEFORE any state
    /// mutation, so a rejected duplicate leaves the live record's `byID`
    /// entry and queue slot untouched (`enqueue` is the only `byID` writer).
    /// The engine mirrors this discipline one layer up: `EngineV2.submit`
    /// refuses to register a stream for a live id, so the duplicate can
    /// never orphan the original request's stream either (PR#62 review).
    @discardableResult
    public func enqueue(
        _ request: CBv2Request, now: Date = Date(), deadline: Date? = nil
    ) throws -> CBv2ScheduledRequest {
        guard byID[request.id] == nil else {
            throw CBv2SchedulerError.duplicateRequestID(request.id)
        }
        guard waiting.count < config.maxWaiting else {
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }
        let record = CBv2ScheduledRequest(
            request: request, arrivalSeq: nextArrivalSeq, submittedAt: now, deadline: deadline)
        nextArrivalSeq += 1
        byID[request.id] = record
        insertWaiting(record, preemptedRequeue: false)
        return record
    }

    // MARK: Plan (the vLLM-V1 core)

    /// Produce one step's work assignment under a single token budget.
    ///
    /// - RUNNING first, in order: each gets `min(remaining, chunk, budget)`;
    ///   decode rows request exactly 1.
    /// - On `capacityExhausted` from the capacity oracle, preempt the lowest
    ///   priority / youngest running request (free KV via the returned
    ///   `preemptions`, keep generated tokens, requeue front,
    ///   `numComputedTokens = 0`). If the victim is the requester itself,
    ///   scheduling stops (vLLM scheduler.py:573-575).
    /// - WAITING admitted only if nothing was preempted this step
    ///   (vLLM scheduler.py:634), while budget and `maxConcurrentRequests`
    ///   allow; chunked prefill admits with a partial token count.
    /// - `numComputedTokens` advances optimistically at plan time; use
    ///   `rollback(_:)` if the planned step is never executed.
    public func plan() -> CBv2StepPlan {
        var budget = config.maxBatchedTokensPerStep
        var assignments: [(id: CBv2RequestID, numTokens: Int)] = []
        var assignmentIndex: [CBv2RequestID: Int] = [:]
        var preemptions: [CBv2RequestID] = []
        var stopScheduling = false

        // 0. Starved block-sized chunk from the previous step: first claim on
        // this step's full budget (see `deferredBlockRequestID`). One-shot —
        // re-armed below if the row starves again.
        var deferredAdmittedID: CBv2RequestID? = nil
        if let deferredID = deferredBlockRequestID {
            deferredBlockRequestID = nil
            if let dIdx = running.firstIndex(where: { $0.id == deferredID }) {
                // Move to the FRONT of the running order (persistently, so a
                // chunk of leading text this step still leaves it in front
                // for the block itself next step).
                if dIdx > 0 { running.insert(running.remove(at: dIdx), at: 0) }
            } else if let wIdx = waiting.firstIndex(where: { $0.id == deferredID }) {
                deferredAdmittedID = admitDeferredBlockRow(
                    at: wIdx, budget: &budget,
                    assignments: &assignments, assignmentIndex: &assignmentIndex)
            }
        }

        // 1. RUNNING first, in order.
        var idx = 0
        while idx < running.count, budget > 0, !stopScheduling {
            let rec = running[idx]
            if rec.isPaused || rec.cancelRequested || rec.remainingTokens <= 0
                || rec.id == deferredAdmittedID
            {
                idx += 1
                continue
            }
            var n = rec.remainingTokens
            if n > 1 { n = min(n, config.prefillChunkSize) }  // chunk prefill only
            n = min(n, budget)
            // Vision requests: never split a multimodal block across chunks
            // (snap to block edges; extend over prefillChunkSize when the
            // chunk starts at a block, bounded by the step budget). 0 ⇒ the
            // block cannot fit this step's remaining budget — skip the row
            // and arm the starvation guard so the NEXT step schedules it
            // first (earlier rows would otherwise starve it indefinitely).
            n = rec.snappedChunkTokens(start: rec.numComputedTokens, proposed: n, budget: budget)
            if n <= 0 {
                if deferredBlockRequestID == nil { deferredBlockRequestID = rec.id }
                idx += 1
                continue
            }

            // Reserve KV headroom; preemption is the backstop.
            var reserved = capacity == nil
            while !reserved {
                do {
                    try capacity?.reserve(id: rec.id, additionalTokens: n)
                    reserved = true
                } catch {
                    guard let victim = preemptionVictim() else {
                        stopScheduling = true
                        break
                    }
                    if victim === rec {
                        preempt(
                            victim, assignments: &assignments,
                            assignmentIndex: &assignmentIndex, budget: &budget)
                        preemptions.append(victim.id)
                        stopScheduling = true  // victim == requester ⇒ stop
                        break
                    }
                    if let vIdx = running.firstIndex(where: { $0 === victim }), vIdx < idx {
                        idx -= 1
                    }
                    preempt(
                        victim, assignments: &assignments,
                        assignmentIndex: &assignmentIndex, budget: &budget)
                    preemptions.append(victim.id)
                }
            }
            if !reserved { break }

            rec.numComputedTokens += n  // optimistic advance
            budget -= n
            assignmentIndex[rec.id] = assignments.count
            assignments.append((id: rec.id, numTokens: n))
            idx += 1
        }

        // 2. WAITING admission — skipped entirely if anything was preempted.
        if preemptions.isEmpty, !stopScheduling {
            var wIdx = 0
            while budget > 0, running.count < config.maxConcurrentRequests,
                wIdx < waiting.count
            {
                let rec = waiting[wIdx]
                // Backpressure survives preemption: a paused row demoted
                // back to waiting must NOT be re-admitted while its consumer
                // is still over the high watermark — `resume` clears the
                // flag. Skip it (slot NOT consumed, nothing reserved, no
                // optimistic advance) exactly like the running path skips
                // paused rows; later waiting rows may still admit (PR#62
                // review).
                if rec.isPaused {
                    wIdx += 1
                    continue
                }
                if rec.cancelRequested { break }  // engine cleans at the boundary
                // A preempted request whose in-flight sample is unconfirmed
                // cannot re-prefill yet (its token values are not host-visible).
                guard rec.pendingSamples == 0 else { break }
                var chunk = min(rec.remainingTokens, config.prefillChunkSize, budget)
                // Same block snapping as the running path. 0 ⇒ this step's
                // remaining budget cannot cover the request's first block —
                // stop admitting (FCFS: younger waiters must not jump a
                // block-bearing elder).
                chunk = rec.snappedChunkTokens(
                    start: rec.numComputedTokens, proposed: chunk, budget: budget)
                guard chunk > 0 else {
                    // Same starvation guard as the running path: a head-of-
                    // queue block that fits a full budget but not what the
                    // running rows left over gets first claim next step.
                    if deferredBlockRequestID == nil { deferredBlockRequestID = rec.id }
                    break
                }
                if let capacity {
                    do { try capacity.reserve(id: rec.id, additionalTokens: chunk) } catch {
                        break  // no preemption on behalf of WAITING requests
                    }
                }
                waiting.remove(at: wIdx)  // wIdx now points at the next record
                rec.status = .running
                rec.numComputedTokens += chunk
                budget -= chunk
                assignmentIndex[rec.id] = assignments.count
                assignments.append((id: rec.id, numTokens: chunk))
                running.append(rec)
            }
        }

        return CBv2StepPlan(
            assignments: assignments.filter { $0.numTokens > 0 },
            preemptions: preemptions)
    }

    /// One-off admission of a starved block-bearing WAITING row ahead of the
    /// running pass — the ONLY departure from RUNNING-first, taken at most
    /// once per step for the single deferred row: with the full step budget
    /// available its block is guaranteed to fit (submit validates every
    /// block against a full budget). Same eligibility rules as the regular
    /// admission path; no preemption on its behalf. Returns the admitted id
    /// (so the running pass skips it — one assignment per row per plan), or
    /// nil when the row is not currently admissible.
    private func admitDeferredBlockRow(
        at wIdx: Int, budget: inout Int,
        assignments: inout [(id: CBv2RequestID, numTokens: Int)],
        assignmentIndex: inout [CBv2RequestID: Int]
    ) -> CBv2RequestID? {
        let rec = waiting[wIdx]
        guard !rec.isPaused, !rec.cancelRequested, rec.pendingSamples == 0,
            running.count < config.maxConcurrentRequests
        else { return nil }
        var chunk = min(rec.remainingTokens, config.prefillChunkSize, budget)
        chunk = rec.snappedChunkTokens(
            start: rec.numComputedTokens, proposed: chunk, budget: budget)
        guard chunk > 0 else { return nil }
        if let capacity {
            do { try capacity.reserve(id: rec.id, additionalTokens: chunk) } catch {
                return nil  // no preemption on behalf of WAITING requests
            }
        }
        waiting.remove(at: wIdx)
        rec.status = .running
        rec.numComputedTokens += chunk
        budget -= chunk
        assignmentIndex[rec.id] = assignments.count
        assignments.append((id: rec.id, numTokens: chunk))
        running.append(rec)
        return rec.id
    }

    /// Undo the optimistic advance of an UNEXECUTED plan (failure/rejection
    /// path). Requests admitted from waiting by this plan stay in `running`
    /// with zero progress — the next `plan()` reassigns them (vLLM never
    /// un-admits except via preemption). Preemptions are NOT undone.
    public func rollback(_ plan: CBv2StepPlan) {
        for (id, n) in plan.assignments {
            guard let rec = byID[id] else { continue }
            rec.numComputedTokens = max(0, rec.numComputedTokens - n)
            capacity?.unreserve(id: id, tokens: n)
        }
    }

    // MARK: Post-execution accounting (engine → scheduler)

    /// The engine launched a step that will sample one token for each of
    /// `ids` (deferred confirmation — the values are still lazy).
    public func markPendingSamples(ids: [CBv2RequestID]) {
        for id in ids { byID[id]?.pendingSamples += 1 }
    }

    /// Confirm one sampled token (called at step finalization, one step
    /// late). Valid for running AND preempted records — a preempted request
    /// keeps its generated tokens.
    public func recordSampled(id: CBv2RequestID, token: Int) {
        guard let rec = byID[id] else { return }
        rec.tokens.append(token)
        rec.pendingSamples = max(0, rec.pendingSamples - 1)
    }

    /// Remove a request in any state. Returns the record for usage reporting.
    @discardableResult
    public func finish(id: CBv2RequestID, reason: CBv2FinishReason) -> CBv2ScheduledRequest? {
        guard let rec = byID.removeValue(forKey: id) else { return nil }
        running.removeAll { $0 === rec }
        waiting.removeAll { $0 === rec }
        rec.status = .finished(reason)
        return rec
    }

    // MARK: Cancellation & backpressure

    /// Mark a request for cancellation. The engine drops the row at the next
    /// step boundary (O(1)); `plan()` never assigns to a marked row.
    public func requestCancel(_ id: CBv2RequestID) {
        byID[id]?.cancelRequested = true
    }

    /// Backpressure: retain the slot, skip scheduling until resumed.
    public func pause(_ id: CBv2RequestID) { byID[id]?.isPaused = true }
    public func resume(_ id: CBv2RequestID) { byID[id]?.isPaused = false }

    // MARK: Chained-decode eligibility

    /// The exact row set the next `plan()` would schedule IF it is a pure
    /// rectangular-decode step with unchanged membership — or nil when the
    /// next step cannot chain (mid-prefill rows, joins possible, cancels
    /// pending, everything paused, or budget too small).
    ///
    /// The engine compares this against the in-flight step's sampled rows to
    /// decide whether to build step N+1 on top of step N's lazy tokens
    /// (SGLang-MLX chained overlap; chain breaks on ANY membership change).
    public func chainCandidateIDs() -> [CBv2RequestID]? {
        guard !running.isEmpty else { return nil }
        var ids: [CBv2RequestID] = []
        ids.reserveCapacity(running.count)
        for rec in running {
            if rec.cancelRequested { return nil }
            if rec.isPaused { continue }
            guard rec.isDecodeReady else { return nil }
            ids.append(rec.id)
        }
        guard !ids.isEmpty, ids.count <= config.maxBatchedTokensPerStep else { return nil }
        // A join would change membership: if any waiting request could be
        // admitted, the chain must break so the mixed step can run.
        if !waiting.isEmpty, running.count < config.maxConcurrentRequests { return nil }
        return ids
    }

    // MARK: Preemption internals

    /// Victim = LOWEST priority; tie → YOUNGEST (largest arrivalSeq).
    private func preemptionVictim() -> CBv2ScheduledRequest? {
        running.min { a, b in
            if a.request.priority != b.request.priority {
                return a.request.priority < b.request.priority
            }
            return a.arrivalSeq > b.arrivalSeq
        }
    }

    private func preempt(
        _ victim: CBv2ScheduledRequest,
        assignments: inout [(id: CBv2RequestID, numTokens: Int)],
        assignmentIndex: inout [CBv2RequestID: Int],
        budget: inout Int
    ) {
        // Refund an assignment the victim received earlier in this same plan
        // (vLLM scheduler.py:550-567).
        if let aIdx = assignmentIndex.removeValue(forKey: victim.id) {
            budget += assignments[aIdx].numTokens
            assignments[aIdx].numTokens = 0  // filtered out on return
        }
        capacity?.releaseAll(id: victim.id)
        running.removeAll { $0 === victim }
        // Full restart: keep generated tokens, recompute everything (the
        // prefix cache makes re-prefill cheap once WS-D lands).
        victim.numComputedTokens = 0
        victim.status = .preempted
        victim.preemptionCount += 1
        insertWaiting(victim, preemptedRequeue: true)
    }

    /// Loop-side capacity backstop: a running request whose first KV
    /// allocation threw `capacityExhausted` is demoted back to waiting
    /// (preempted-style full restart — generated tokens kept, capacity
    /// released, front-of-class requeue) instead of error-finishing, so an
    /// ACCEPTED request waits for room rather than failing when several
    /// same-step admissions race for the last bytes (PR#62 review, paged
    /// admission alignment). Returns false when the id is not running.
    public func requeueOnCapacity(_ id: CBv2RequestID) -> Bool {
        guard let rec = running.first(where: { $0.id == id }) else { return false }
        capacity?.releaseAll(id: rec.id)
        running.removeAll { $0 === rec }
        rec.numComputedTokens = 0
        rec.status = .preempted
        rec.preemptionCount += 1
        insertWaiting(rec, preemptedRequeue: true)
        return true
    }

    /// New arrivals: before the first STRICTLY lower priority (FCFS within a
    /// class). Preempted requeues: before the first SAME-or-lower priority
    /// (front of their class — vLLM prepends preempted requests).
    private func insertWaiting(_ rec: CBv2ScheduledRequest, preemptedRequeue: Bool) {
        let priority = rec.request.priority
        let idx: Int
        if preemptedRequeue {
            idx = waiting.firstIndex { $0.request.priority <= priority } ?? waiting.count
        } else {
            idx = waiting.firstIndex { $0.request.priority < priority } ?? waiting.count
        }
        waiting.insert(rec, at: idx)
    }
}
