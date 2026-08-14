# WS-F: Model adaptation — Gemma 4 + GPT-OSS 20B on the v2 attention path

You own the two production models' integration. EXCLUSIVE ownership of
`Libraries/MLXLLM/Models/Gemma4Text.swift` and `Libraries/MLXLLM/Models/GPTOSS.swift`.

## Deliverables

1. `Libraries/MLXLMCommon/ContinuousBatchingV2/LayerKindDerivation.swift` —
   `[CBv2LayerKind]` derivation from model configs:
   - Gemma 4: `layer_types` / slidingWindowPattern 5 (4×sliding(512) + 1×
     full), trailing `num_kv_shared_layers` (20 of 35) with
     `sharesKVWithLayer` = last non-shared layer OF THE SAME TYPE (mirror
     the existing `previousKvs` map, Gemma4Text.swift:877-893), asymmetric
     head dims (sliding 256 / global 512), K-eq-V variants, hasSinks=false.
   - GPT-OSS: alternating sliding(128)/full, hasSinks=true, headDim 64.
   Pure functions + unit tests against both configs (construct configs in
   code; no weight downloads).
2. Gemma4Text.swift: add a v2 attention branch — when the layer's cache is
   a `CBv2AttendingLayerCache`, route through `updateAndAttend` /
   `attendBorrowing` (KV-shared layers) instead of the manual mask/SDPA
   forks. Requirements:
   - Per-row RoPE via `cache.positionOffsets` (array-offset RoPE overload);
     KV-shared layers reuse the SOURCE layer's captured offsets (invariant
     1, report 10). Snapshot offsets BEFORE update.
   - Dual RoPE thetas and QK-norm flow unchanged.
   - The legacy (B=1 / old engine) paths must remain byte-identical — v2 is
     a new branch, not a rewrite. `gemma4AttentionFallback` and mask logic
     stay for legacy callers.
3. GPTOSS.swift: same v2 branch; sinks passed into `updateAndAttend`
   (contract guarantees denominator-only handling); kill the per-request
   `sinksActive` `.item()` host readback on the v2 path (compute once at
   load, cache the Bool).
4. `newCacheV2(backend:)` (or equivalent factory) on both models producing
   `[CBv2AttendingLayerCache]` via the contract types + LayerKindDerivation
   (WS-A implements the concrete classes; code against the protocols and a
   trivial in-file mock for tests — integration wires the real ones).

## Tests (`Tests/MLXLMTests/CBv2ModelTests.swift`)
- LayerKindDerivation: exact expected [CBv2LayerKind] for Gemma-4 (35
  layers, shared map matches previousKvs semantics) and GPT-OSS (24
  layers alternating), including K-eq-V and head-dim asymmetry.
- Tiny-config forward smoke test: construct Gemma4Text and GPTOSS with
  2–4 layer random-weight tiny configs; run a 5-token prompt + 3 decode
  steps through the v2 branch with a mock AttendingLayerCache; assert
  shapes, offset progression, and that the shared-layer branch calls
  `attendBorrowing` with the right source layer.
- Legacy-path regression: existing model tests still pass untouched.

## References
Report 10 §1 (exact model structures, file:line), §4 invariants 1, 2, 5,
9, 11. Corpus: `/Users/gaj/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/`.
