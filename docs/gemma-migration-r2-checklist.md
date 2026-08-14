# Gemma Migration: R2 / Private Artifact Checklist

> **Historical record.** This checklist covers the DeepSeek-to-Gemma
> migration and is complete. The 2026-07 re-pin to Poolside Laguna XS 2.1
> (new tokenizer with vocab 100352) subsequently repeated that regeneration
> pass — hidden goldens, GPQA references, public fixtures, and pins — for the
> v2 serial track (`laguna-xs-2.1-serial-v2`). The current content-addressed
> object keys and pins are recorded in `docs/private-benchmark-security.md`;
> the Gemma-era details below remain unchanged for provenance.
> Workflow-name note: the serial ranked pipeline referenced below as
> `serial-benchmark.yml` now runs under the canonical name `benchmark.yml`;
> the step names and env pins are unchanged. Mentions below are kept as
> written for provenance.

The DeepSeek V4 Flash to Gemma 4 31B 4-bit migration changed the model,
tokenizer, and layer count, so every private artifact that embeds prompt
tokens, expected tokens, or model-derived calibration was model-mismatched
and had to be regenerated. This file is the consolidated operator checklist;
all items are now **DONE**: the hidden goldens and GPQA references were
regenerated on the ranked M5 hardware from the `mlx-swift-lm` rebase
reference and uploaded to R2, and the checked-in public fixtures were
regenerated the same way.

Nothing here changed the R2 secret/env plumbing (`R2_ACCESS_KEY_ID`,
`R2_BUCKET_ENDPOINT`, `R2_SECRET_ACCESS_KEY` on the
`benchmark-private-prompts` environment) — that structure is intact and
correct. Only the *contents* of the private objects (and the pins that
verify them) were regenerated.

## 1. Hidden correctness golden (R2) — DONE (uploaded 2026-07-09)

- **Object:** `correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256-gemma.json`
- **Contents:** 512-token base prompt retokenized with the Gemma tokenizer;
  256 teacher-forced expected continuation tokens captured on the M5 from
  the trusted Gemma 4 31B 4-bit rebase reference; a
  `correctness_gates.free_run` gate covering the full timed decode offset
  range (`attach-free-run-gate` defaults, 128 steps); and the benchmark
  section required by the common harness schema. That section is not the
  ranked timed target: the ranked measurement wrapper self-generates its own
  oracle from the separately provisioned private timed prompt. Per-prompt
  `baseline_*_seconds_per_token` calibration is intentionally omitted: the
  ranked pipeline measures the pinned reference baseline live on the same box,
  and local modes fall back to the calibrated constants.
- **Consumed by:** `.github/workflows/serial-benchmark.yml` — the single ranked
  job's "Prepare hidden correctness golden" step downloads and pin-verifies
  the raw object; "Attach GPQA gates and verify augmented golden" then
  augments it with the GPQA behavior gates.
- **Pins:** `MLXFAST_RAW_CORRECTNESS_GOLDEN_SHA256`
  (`56c282dcaac433543ef0eecb625cd99bc20f1ae1f7b9415efe32a71e6eb4eae9`) and
  `MLXFAST_RAW_CORRECTNESS_GOLDEN_BYTES` (`38162`) in `serial-benchmark.yml` match
  the uploaded object. The augmented golden's hash/bytes
  (`MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_*`) are self-anchored at run time
  right after augmentation, as before.

## 2. Hidden GPQA reference cases (R2) — DONE (uploaded 2026-07-09)

- **Object:** `correctness_prompts/gpqa_reference_cases-gemma.json`
- **Contents:** the 9 GPQA multiple-choice prompt cases (prompt text, answer
  keys, domains unchanged) with each case's `accepted_token_sequences`
  regenerated from the Gemma 4 31B 4-bit rebase reference on the M5 (Gemma
  tokenizer, greedy first-answer-token capture), preserving the previous
  artifact's one-token-sequence shape; the semantic judge's reference
  answers continue to derive from the answer keys.
- **Consumed by:** `.github/workflows/serial-benchmark.yml` — "Attach GPQA gates
  and verify augmented golden" (`attach-gpqa-gates`), which drives the
  hidden GPQA behavior gates, the TTFT guardrail, and the semantic-GPQA
  answer capture judged by `run-semantic-gpqa-gate.sh`. The object carries
  9 cases, but each ranked run attaches and judges only the first 5
  token-budget-valid ones: `MLXFAST_GPQA_CASE_COUNT`,
  `MLXFAST_SEMANTIC_GPQA_CASE_COUNT`, and `MLXFAST_GPQA_TTFT_CASE_COUNT`
  are all `5` in `serial-benchmark.yml`, so per-run "N/5" gate results are
  consistent with the 9-case object.
- **No hash pin:** the augmented golden's hash/bytes are computed at run
  time, so no workflow constant needs updating for this object itself.
- **Semantic threshold — recalibrated (2026-07-09):** the semantic-GPQA
  pass-count threshold is calibrated against the unmodified baseline's
  judged answer quality, and a regeneration of the hidden
  prompts/references requires a fresh judged baseline on the ranked runner.
  After this upload, five judged official-runner baseline runs of
  unmodified main (29040771374, 29048752714, 29051462434, 29052276465,
  29053091705) all scored 2/5 with identical per-case verdicts, so
  `MLXFAST_SEMANTIC_GPQA_MIN_PASS` moved from 0 (aggregate-recording) to
  min(observed) - 1 = 1. See `MLXFastConstants.semanticGPQAMinPassCount`
  for the full calibration provenance.

## 3. Private prompt manifest (organizer-side, not workflow-consumed)

- The manifest of hidden prompt sources used to regenerate goldens offline
  (see `docs/private-benchmark-security.md`). It is never downloaded by the
  workflows; the organizer's offline regeneration pipeline switched to the
  Gemma tokenizer/reference before producing items 1 and 2.

## 4. Public fixtures (checked in) — DONE (regenerated on M5, 2026-07-09)

- `correctness_prompts/public_longcopy_gate_english_512_256.json` and
  `correctness_prompts/public_longcopy_gate_english_512_1024.json` were
  regenerated on the ranked M5 hardware against the Gemma 4 31B 4-bit rebase
  reference with `mlxfast-swift generate-golden` (Gemma-tokenized 512-token
  prompt, greedy reference continuations; the 256 fixture is a greedy prefix
  of the 1024 one). Because the expected tokens are M5 argmaxes, the local
  public gate can fail on other Apple Silicon generations at near-tie
  positions; that is a hardware divergence, not a fixture bug.
- Pins moved with the fixtures:
  - `MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_SHA256`
    (`182a7f98d24cc8f26e8b08505fe7a8b6d825702f99d0b78a83f49dd42f1b2aea`) /
    `..._BYTES` (`11140`) in `.github/workflows/serial-benchmark.yml`.
  - The fixture digests pinned in `Tests/MLXFastTests/GoldenTests.swift`
    (the 1024 fixture's digest is
    `2de5474bbe707bcb2e8b71d7d771ffd9be70c252d3ecce7f1511aa2a50933b4d`).

## 4b. Private timed decode target (R2) — STAGED, not yet uploaded

- **Object:** `correctness_prompts/timed_decode_lowsim_prose_v1.txt` (the
  independent timed prefill/decode target introduced by the timed-decode
  fairness change; see `docs/timed-decode-evaluation.md`).
- **Pins:** `MLXFAST_TIMED_DECODE_PROMPT_SHA256` / `_BYTES` variables on the
  `benchmark-private-prompts` environment. Set at upload time from the staged
  artifact; validated by the workflow before any bench phase runs.
- **Gate:** the target activates only after its M5-generated greedy
  continuation passes `analyze-ngram-similarity` at
  `benchmarkMaxPromptLookupHitRate` (3%), and after the coordinated
  measure-job wrapper (explicit `--prompt/--prompt-sha256/--target-id`
  interface, per-target oracle-cache identity) is deployed and the box
  baseline calibration is refreshed on the new target per the RUNBOOK.

## 5. Ranked baseline — DONE (superseded by the on-box pinned baseline)

- The ranked score no longer prices against a repo-pinned baseline ref or
  the calibrated constants: the single-machine pipeline measures the
  **pinned reference baseline tree provisioned on the M5 box**
  (`/opt/bench-runner/baseline/laguna-xs-2.1-serial-v2/current`,
  sanity-banded against the versioned
  `/opt/bench-runner/state/laguna-xs-2.1-serial-v2/baseline-calibration.json`)
  in the same session
  as the candidate. Regenerating that tree or its calibration is an
  operator procedure (RUNBOOK) and a ranking-contract change.
- The `officialBaselinePrefillSecondsPerToken` /
  `officialBaselineDecodeSecondsPerToken` constants in
  `Sources/MLXFastCore/Constants.swift` remain as local-mode estimates and
  the gates pass's skip-timed placeholder timing only; see
  `docs/benchmark-window-freeze.md`.
