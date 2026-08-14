# Contract issues — WS-E (sampling)

Gaps found while implementing `LogitsPipelineV2` / `SamplerV2` /
`DetokenizerV2` / `StopHoldback` against the frozen `CBv2Contracts.swift`.
None blocked the work; the closest conforming shape was chosen in each case.

1. **`topLogprobs == 0` cannot express "chosen-token logprob only".**
   OpenAI's API allows `logprobs: true` with `top_logprobs: 0` (report the
   sampled token's logprob, no alternatives). The contract only has
   `CBv2SamplingParams.topLogprobs: Int` with "0 = none". Chosen shape:
   `topLogprobs > 0` is the sole logprobs trigger; the chosen-token logprob
   rides along for free. If the provider needs the OpenAI shape, the
   contract needs a separate `wantsLogprobs: Bool`.

2. **No contract type for the sampling-stage membership update.** The
   engine must hand the sampler (per row): sampling params, prompt tokens,
   and output-so-far (for mid-flight re-vectorization). Defined a
   module-local `CBv2SamplerRow` (LogitsPipelineV2.swift) as that carrier;
   workstream B should construct it at batch-membership change. Fine as a
   non-contract type, but integration should bless it or fold it into the
   contract.

3. **Nil-seed semantics unspecified.** The contract documents the RNG key
   as `(seed, requestID, stepIndex)` but not what `seed == nil` means.
   Chosen shape: an engine-level fallback seed fixed at `SamplerV2` init
   (batch-invariant within a process; intentionally non-reproducible across
   runs, matching "best-effort" reproducibility).

4. **Repetition-window population unspecified.** `repetitionContextSize`
   does not say whether the window covers prompt ∪ output or output only.
   Per spec E (report 08 §6), implemented as the most recent
   `repetitionContextSize` tokens of prompt + output (sliding, evicting
   oldest); frequency/presence penalties count output tokens only,
   unwindowed (vLLM semantics).
