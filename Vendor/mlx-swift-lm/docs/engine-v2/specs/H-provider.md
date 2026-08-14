# WS-H: Provider bridge (d-inference repo)

Bridge the Swift provider to the v2 engine behind a flag. You work in the
d-inference provider worktree (NOT this repo); the engine API you bind to
is `CBv2Engine` in `libs/mlx-swift-lm` `Libraries/MLXLMCommon/
ContinuousBatchingV2/CBv2Contracts.swift` (read it first).

## Deliverables (`provider-swift/Sources/ProviderCore/Inference/`)

1. `EngineV2Bridge.swift` — adapts `CBv2Engine` to the existing
   `BatchScheduler` engine surface (study `BatchScheduler+EngineFactory.
   swift` and `MultiModelBatchSchedulerEngine+Translation.swift` for the
   current contract): request translation (ChatRequest → CBv2Request:
   tokenization via the existing tokenizer path, sampling params incl.
   logitBias/seed/logprobs passthrough, stop resolution — reuse
   `buildStopTokenIds()` semantics so B=1/batched stay identical),
   event translation (CBv2Event → SSE chunk stream incl. usage), error
   mapping (capacityExhausted → the existing 429/503 classification via
   `InferenceErrorClassifier`).
2. `EngineV2Config.swift` — selection: env `DARKBLOOM_ENGINE_V2=1` or
   provider config key `engine_v2`; per-model allowlist (default: enabled
   for gemma-4* and gpt-oss* model ids when flag set); safe fallback to
   the legacy engine on init failure WITH a telemetry event.
3. Capacity: map `CBv2CapacitySnapshot` into the existing heartbeat
   capacity fields (`ProviderLoop+Capacity.swift`) — SAME protocol fields,
   truthful numbers (bytes-derived active token estimates). NO protocol
   changes in this workstream (coordinator compatibility preserved).
4. Cancellation: wire `ProviderLoop+Cancellation.swift` request-id
   cancellation to `CBv2Engine.cancel`.
5. Telemetry: emit engine_v2 tag on existing inference metrics; add
   `engine_v2.step_wedge` signal from the v2 watchdog to the existing
   WedgeMonitor plumbing.

## Constraints
- Do NOT modify coordinator Go code, the wire protocol, or the legacy
  engine path. Additive only; the flag off ⇒ byte-identical behavior.
- The submodule in your worktree may not yet contain merged v2 impls; code
  against the contract and keep the bridge compiling with
  `swift build` using protocol stubs in tests. Integration bumps the
  submodule and runs the full build.

## Tests (`provider-swift/Tests/ProviderCoreTests/EngineV2BridgeTests.swift`)
- Translation: OpenAI-style request → CBv2Request field-by-field (params,
  stops, maxTokens defaulting, logit_bias id parsing).
- Event → SSE: delta/finish/usage framing matches the legacy engine's
  framing (fixture-compare against a recorded legacy stream shape).
- Error mapping: capacityExhausted → retryable capacity error class.
- Config gating: flag off → legacy factory chosen; flag on + non-allowlisted
  model → legacy; init failure → fallback + telemetry event (use a failing
  stub engine).
- Live-isolated style: use a scripted in-process `CBv2Engine` stub emitting
  a canned stream; no network, no models, no prod anything.

## References
Contract file (path above), plan §7 Phase 2(e)/H notes, report 03
(provider scheduler layer map). Corpus:
`/Users/gaj/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/`.
