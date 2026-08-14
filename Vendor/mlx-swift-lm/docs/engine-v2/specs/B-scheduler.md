# WS-B: Scheduler + engine loop (vLLM-V1 style, no phases)

## Deliverables (`Libraries/MLXLMCommon/ContinuousBatchingV2/`)

1. `SchedulerV2.swift` — pure, synchronous, fully unit-testable (NO MLX
   arrays in this file). State per request: `numComputedTokens`,
   `numTokens` (prompt+generated), status, priority, arrival. `plan()`
   produces `CBv2StepPlan` each step:
   - RUNNING first, in order: each gets `min(numTokens - numComputed,
     remaining budget)`; decode rows request 1.
   - WAITING admitted while budget and `maxConcurrentRequests` allow;
     chunked prefill = admit with a partial token count (chunk =
     min(remaining prompt, prefillChunkSize, leftover budget)).
   - Optimistic advance: `numComputedTokens += assigned` at plan time;
     rollback on failure/rejection (vLLM's key trick — enables planning
     step N+1 while N executes).
   - Preemption: when the KV backend throws `capacityExhausted` mid-plan,
     preempt the LOWEST priority / youngest running request: free its KV,
     keep its generated tokens, requeue front (`numComputedTokens = 0`;
     re-prefill will run through the prefix cache when D lands).
   - Priority: higher first among WAITING; FCFS within a class.
2. `EngineLoopV2.swift` — the execution loop (single engine thread/actor,
   graph-build + `asyncEval` ONLY):
   - Chained async decode (SGLang-MLX pattern): build step N+1's `[B,1]`
     forward on top of step N's still-lazy sampled-token array; `asyncEval`
     it; then block on step N's tokens for stop detection. Chain breaks on
     any membership change (prefill completion, finish, cancel, join).
   - Deferred stop detection: tokens are inspected one step late; a
     finished request wastes at most one slot-step; its extra token and KV
     tail are rolled back (`CBv2SequenceKV.rollback(1)`).
   - Prefill execution: per-request `[1, chunk]` forwards against that
     request's caches, interleaved per the plan (budget already bounds ITL
     impact). Use `eval` batching sparingly: one `asyncEval` per step.
   - Per-request deadlines and a step watchdog (configurable, default 120 s
     request / 30 s single step → error-finish + engine health signal).
3. `AdmissionV2.swift` — truthful admission: estimated KV bytes for
   (promptLen + maxTokens worst case) vs `bytesCapacity`, BUT reservation
   is soft (vLLM-style optimism): admit while `bytesInUse + nextStepNeed <
   capacity - watermark`; preemption is the backstop. Expose
   `CBv2CapacitySnapshot`.
4. `EngineV2.swift` — implements `CBv2Engine`: submit/cancel/capacity/
   shutdown; wires scheduler + loop + sampler (protocol from contract; use a
   greedy stub until E merges) + model adapter. Tokenization happens on the
   caller's task, never the engine thread. Cancellation drops the row at the
   next step boundary, O(1).
5. `OutputStreamV2.swift` — per-request `AsyncStream<CBv2Event>` with
   bounded buffering (default 256 events): on overflow, apply backpressure
   by pausing that request's scheduling (slot retained, decode skipped)
   until drained; never unbounded memory, never blocks the engine thread.

## Model interface
Code against a minimal internal protocol (define it in `EngineLoopV2.swift`):
`CBv2SteppableModel { func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray }`
G's tiny fixture and F's adapters satisfy it. Do not import MLXLLM.

## Tests (`Tests/MLXLMTests/CBv2SchedulerTests.swift`)
- Pure scheduler simulations: Poisson arrivals, mixed prompt lengths,
  budget invariants (sum ≤ maxBatchedTokens; running ≤ maxConcurrent),
  chunked prefill progress, preemption picks the right victim and requeues
  front, optimistic-advance rollback, priority ordering, cancel-while-
  waiting/running/prefilling.
- Loop tests with a scripted fake model (no MLX weights): chained decode
  emits correct tokens one step late; stop token honored with ≤1 wasted
  step; backpressure pauses exactly the slow stream; watchdog fires.

## References
Report 08 §1 (vLLM scheduler mechanics, optimistic advance), report 09 §7
(SGLang MLX chained overlap), plan §7 Phase 3. Corpus root:
`/Users/gaj/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/`.
