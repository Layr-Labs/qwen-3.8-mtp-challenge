# Contract issues — WS-B (scheduler + engine loop)

Places where `CBv2Contracts.swift` was insufficient for the WS-B deliverables,
and the closest conforming shape chosen. All new protocols live in WS-B-owned
files and are trivially replaceable at integration time.

## 1. No factory for `CBv2AttendingLayerCache`

The contract defines the per-layer batch-facing cache protocol but no way to
*construct* one for a given set of rows. The engine loop must build fresh
batch views every step (decode batch = current running rows; prefill = one
row), so it needs a factory.

**Shape chosen:** `CBv2LayerCacheProvider` in `EngineLoopV2.swift`:
`func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache]`
(`rowStates[b][layer]`, row order == batch row order). WS-A's `LayerCacheV2`
should conform (or integration adds a tiny adapter).

## 2. Spec references a sampler "protocol from contract" that does not exist

`B-scheduler.md` §4 says to wire "sampler (protocol from contract; use a
greedy stub until E merges)", but the contract only defines
`CBv2SamplingParams`, no sampler protocol.

**Shape chosen:** `CBv2StepSampler` in `EngineLoopV2.swift`:
`func sample(logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID], stepIndex: Int) -> MLXArray`
— logits `[B, vocab]` (last position per row) → lazy token array `[B]`, no
host syncs. `CBv2GreedySampler` is the stub. WS-E's `SamplerV2` should
conform.

## 3. No detokenizer surface, but `CBv2Event.delta` requires detokenized text

The contract's `.delta` carries text "already detokenized incrementally with
UTF-8 and stop-string holdback applied", and requests carry `stopStrings`,
but detokenization is WS-E's (`DetokenizerV2.swift`, `StopHoldback.swift`)
and no protocol exists for the engine to call.

**Shape chosen:** `CBv2IncrementalDetokenizer` / `CBv2DetokenizerFactory` in
`EngineLoopV2.swift` (`push(tokens) -> String`, `matchedStopString`,
`flush()`). Default is a null implementation: deltas stream token ids with
empty text and stop strings never match until WS-E lands. Token-level stop
detection (stop tokens, maxTokens, deadlines) is fully functional either way.

## 4. `CBv2KVBackend.release` timing vs in-flight lazy graphs is unspecified

With chained async decode, step N+1 is launched before step N's tokens are
inspected. A request that finishes (or is cancelled/preempted) while a step
that references its `CBv2SequenceKV` state is still in flight cannot have its
storage `release`d immediately: for the contiguous v1 backend MLX refcounting
makes this benign, but for the paged backend an O(1) metadata free could
reallocate pages to another sequence while the in-flight kernel still reads/
writes them (vLLM fences frees behind `processed_step_seq` for exactly this).

**Shape chosen:** the engine loop defers `release` of any state that
participated in the in-flight step until that step is finalized (its sampled
tokens have been materialized, which in MLX's dependency model implies the
whole step graph — including KV writes — has executed). WS-C should NOT need
its own fence, but this invariant should be promoted into the contract text
at integration.

## 5. `CBv2Engine.submit` has only `capacityExhausted` for rejections

Non-capacity rejections exist: engine shut down / draining, empty prompt,
`maxTokens <= 0`. The contract only documents throwing
`CBv2KVError.capacityExhausted`.

**Shape chosen:** shutdown/draining rejections throw
`capacityExhausted(needed: …, available: 0)` (the provider maps it to 503,
which is semantically right for a draining node). Degenerate requests (empty
prompt, `maxTokens <= 0`) do not throw; they return a stream that immediately
emits `.finished` (`.length` for `maxTokens <= 0`, `.error` for an empty
prompt), so callers get a uniform event surface.

## 6. Scheduler ↔ admission capacity hook is not in the contract

The spec requires the scheduler to react to `capacityExhausted` "mid-plan"
(preemption backstop), but `plan()` is pure and the contract offers no
capacity oracle the scheduler could consult.

**Shape chosen:** `CBv2StepCapacity` (WS-B-internal, in `AdmissionV2.swift`):
`reserve(id:additionalTokens:) throws`, `unreserve(id:tokens:)`,
`releaseAll(id:)`, `hasHeadroom(additionalTokens:)`. `AdmissionV2` implements
it with a soft byte ledger over `CBv2LayerKind`-derived per-token KV bytes.
This stays internal to WS-B; other workstreams never see it.
