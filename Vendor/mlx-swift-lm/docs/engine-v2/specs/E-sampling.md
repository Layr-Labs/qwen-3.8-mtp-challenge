# WS-E: Vectorized sampling, logprobs, incremental detok + stop holdback

## Deliverables (`Libraries/MLXLMCommon/ContinuousBatchingV2/`)

1. `LogitsPipelineV2.swift` — ordered transform pipeline over `[B, vocab]`
   logits (vLLM order, report 08 §6):
   (0) capture RAW logprobs first when any row requests them (log_softmax
   on the raw row; gather top-k lazily); (1) logit_bias scatter-add;
   (2) penalties — repetition (multiply/divide on presence over prompt ∪
   output window), frequency (α·count), presence (β·mask), computed from
   per-row token-count tensors maintained incrementally (scatter-add per
   step, NEVER re-binned from scratch); (3) temperature divide; (4) top-k /
   top-p / min-p via vocab sort + cumsum threshold. ALL per-row parameters
   are tensors `[B]` or `[B,1]` built once at batch-membership change —
   zero per-row Swift loops in the step path.
2. `SamplerV2.swift` —
   - All-greedy fast path: single `argMax` when every row is greedy
     (temperature < 1e-5) — matches legacy vectorized greedy.
   - Mixed batch: Gumbel-max via exponential noise
     (`argmax(probs / -log(u))` pattern) so there is no multinomial sync;
     per-row keyed RNG: `MLXRandom.key(seed ⊕ requestID ⊕ step)` per row,
     stacked — a row's stream depends only on (seed, requestID, step),
     NEVER on batchmates (batch-invariance requirement).
   - `where(temp < eps, greedy, sampled)` merge for mixed batches.
3. `DetokenizerV2.swift` — per-request incremental detokenization: hold
   back bytes until valid UTF-8 boundary (multi-byte chars split across
   tokens must never emit replacement chars mid-stream).
4. `StopHoldback.swift` — stop-string streaming matcher: emit only text
   that provably cannot be a prefix of any stop string (longest-suffix
   holdback); on match, truncate at match start and signal stop. Handles
   stop strings spanning token boundaries and overlapping candidates.
   This is REQUIRED for B's one-step-late stop detection to be invisible.

## Tests (`Tests/MLXLMTests/CBv2SamplingTests.swift`)
- Pipeline order: logprobs reflect raw logits even with penalties/bias on.
- Penalties parity vs a scalar reference implementation on random histories.
- Determinism: same (seed, requestID) → same tokens at B=1 and inside a
  B=4 mixed batch (greedy neighbors), across two runs.
- Greedy fast path bit-identical to argMax.
- Detok: multi-byte UTF-8 (emoji, CJK) split across tokens; no replacement
  chars; byte-exact final text.
- StopHoldback: stop string split across 3 tokens; overlapping stops
  ("</s>", "</section>"); prefix-ambiguity holdback released when
  disambiguated; nothing after stop ever emitted.
- Use a real tokenizer fixture if a tiny one is available in the repo's
  test resources; otherwise a hand-rolled byte-pair stub is fine.

## References
Report 08 §6 (vLLM sampler order, Gumbel trick, BatchUpdate churn safety),
critique items on seed/batch-invariance and detok (report 12 items 5, 6).
Corpus: `/Users/gaj/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/`.
