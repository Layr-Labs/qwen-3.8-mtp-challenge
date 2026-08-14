# ContinuousBatchingV2 — Architecture & Integration Contract

**Goal:** production-grade continuous batching at B=3–4 (max 8) on all Apple M generations, correct-by-construction for Gemma 4 and GPT-OSS 20B. Replaces the left-padded dense-batch engine (`ContinuousBatching/`) which stays untouched as fallback.

**Background:** full research corpus at `/Users/gaj/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/` (reports 01–12) and the plan at `/Users/gaj/Documents/Builds/d-inference/docs/batching-engine-v2-plan.md`. Read report 10 (Gemma4/GPT-OSS invariants) before writing any KV/attention code.

## Core design (why v2 cannot have the old bug class)

1. **Per-request KV state** (`CBv2SequenceKV`): each sequence owns its storage and its absolute position counter. There is no shared `_idx`, no `leftPadding`, no batch-wide trim. Join = add an object; leave = drop an object. (This is the SGLang-MLX-backend design: ragged scheduling + per-request contiguous caches.)
2. **Decode is rectangular `[B,1]`** — one token per running request, so decode raggedness is impossible; no masks needed (each row attends only its own KV). Prefill runs per-request `[1, chunk]` under a shared token budget (vLLM-style plan, no phases).
3. **Sliding windows are storage eviction keyed to absolute positions** (ring keeps the recent end, modular token indices). No window masks over shared buffers.
4. **Attention is owned by the layer cache object** (`CBv2AttendingLayerCache.updateAndAttend`). v1 backend = per-row `MLXFast.scaledDotProductAttention` (with sinks/GQA); v2 backend = paged pool + custom Metal kernel behind the same protocol. Models and scheduler never see the difference.
5. **Numerically pinned paths**: one attention path per (model, phase); never switch mask representation across steps (MLX #3384). Fully-masked rows cannot exist by construction (no padding).
6. **Engine thread discipline**: the engine loop does graph-build + `asyncEval` only. Tokenization, detokenization, prefix-cache donation, SSD I/O all run elsewhere. Stop detection is deferred one step (chained async decode), with detok holdback so users never see past-stop text.

## Module map & file ownership (conflict avoidance — do not touch files you don't own)

| WS | Owner scope (all under `Libraries/MLXLMCommon/ContinuousBatchingV2/` unless noted) |
|----|----|
| CONTRACT (frozen) | `CBv2Contracts.swift`, this doc |
| A core-runtime | `SequenceKV/` (FullSequenceKV, WindowedSequenceKV, QuantizedSequenceKV, ContiguousKVBackend), `LayerCacheV2.swift`, `AttentionV1.swift`; EXCLUSIVE: `Libraries/MLXLMCommon/AttentionUtils.swift` hook |
| B scheduler | `SchedulerV2.swift`, `EngineLoopV2.swift`, `AdmissionV2.swift`, `EngineV2.swift`, `OutputStreamV2.swift` |
| C paged backend | `Paged/` (PagedKVPool, PagedSequenceKV, `pagedattention.metal` MSL via MLXFast.metalKernel, PagedAttentionBackend, microbench) |
| D prefix cache | `PrefixCacheV2.swift`, `BlockHasher.swift` (reuse existing SHA-256 chain scheme) |
| E sampling | `SamplerV2.swift`, `LogitsPipelineV2.swift`, `DetokenizerV2.swift`, `StopHoldback.swift` |
| F models | EXCLUSIVE: `Libraries/MLXLLM/Models/Gemma4Text.swift`, `Libraries/MLXLLM/Models/GPTOSS.swift`; new `LayerKindDerivation.swift` |
| G harness | `Tests/MLXLMTests/CBv2*` (invariance, parity, scheduler sim, fixtures with tiny random-weight models), `benchmarks/` additions |
| H provider | d-inference repo: `provider-swift/.../Inference/EngineV2Bridge*` behind `DARKBLOOM_ENGINE_V2=1` |

Each workstream also owns its own unit tests (`Tests/MLXLMTests/CBv2<WS>Tests.swift` naming). G owns cross-cutting suites only.

## Integration protocol

- Branch from `feat/engine-v2` (contracts committed). Commit early and often to your worktree branch; final state = compiling code + passing `swift test --filter CBv2`.
- Need a type another workstream implements? Code against the protocol in `CBv2Contracts.swift` and write a minimal mock in your tests. Do NOT invent cross-workstream types outside the contract; if the contract is insufficient, note it in `docs/engine-v2/CONTRACT-ISSUES-<ws>.md` and continue with the closest conforming shape — integration resolves it.
- Builds: use `swift build -j 6` / `swift test -j 4 --filter <yours>` to be a good neighbor (several worktrees build concurrently on one machine).
- No changes to the legacy `ContinuousBatching/` engine, `BatchKVCache.swift`, `GenerationBatch.swift`, etc. v2 is additive.

## Acceptance (every workstream)

1. `swift build` succeeds in your worktree.
2. Your unit tests pass (`swift test --filter CBv2<yours>`); tests must run without model weights (tiny random-weight fixtures).
3. Batch-composition invariance where applicable: same request ⇒ identical output regardless of batchmates.
4. Commit with clear messages; leave `git status` clean. No `Co-Authored-By` trailers.
