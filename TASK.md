# mlxfast — Poolside Laguna XS 2.1 DFlash Swift Challenge

Optimize target-verified block speculative decode (DFlash) for Poolside Laguna
XS 2.1 (MoE text tower, Poolside NVFP4) on Apple Silicon while preserving the
model's exact greedy output. An organizer-provided DFlash draft model proposes
tokens, the Laguna target verifies K rows per forward, the longest correct
prefix is accepted, and the KV rows behind rejected draft tokens are rolled
back.

## Default ranked contract

`benchmark.json` and `.github/workflows/dflash-benchmark.yml` define the
default — and only — Yukon track, `laguna-xs-2.1-dflash-v1` (manifest name
`mlxfast-challenge-dev-dflash`). The former serial track
`laguna-xs-2.1-serial-v2` is retired: its ranked workflow
`.github/workflows/benchmark.yml` was deleted, and `benchmark.json` is now
the DFlash manifest. The track is live: `official_scoring_enabled`,
`reference_baseline.publication_allowed`, and
`token_fidelity_gate_status=implemented` are all set.

A ranked run on the self-hosted M5 box (runner label `m5-laguna-dflash`, the
dedicated DFlash label — not the serial track's `m5-bench`; the label in
`.github/` is the source of truth):

1. Verifies the pre-provisioned reference checkpoint against the pinned
   manifest and provisions the pinned DFlash drafter, builds the trusted CLI
   and the sandboxed participant worker, and transforms the reference
   checkpoint into `weights/`.
2. Runs the shared correctness stack (public drift tripwire, then the hidden
   512-token-prompt teacher-forced base case plus anchor/free-run/behavior/
   GPQA gates and the semantic GPQA judge) reused from the serial track, plus
   the DFlash token-fidelity gate (trusted sequential re-verification of every
   emitted token against the candidate's own emitted prefix, with a bounded
   near-tie budget; the rejected tail is priced).
3. Scrubs every hidden byte from the bench workspace, then runs the timed
   paired measurement LAST behind the fixed 40C thermal gate: the trusted
   serial K=1 target baseline and the candidate's DFlash decode are measured
   back to back on the same silicon by the box-owned `measure-dflash-job.sh`
   (telemetry-validated). The hidden timed prompt is sampled uniformly at
   random once per ranked run from an 8-entry, domain-varied
   `timed_prompt_pool` (anti-lottery).

Timing measures a decode-only window: 512 decode tokens with the seed prefill
charged inside the window (block size K=2 at the ranked dispatch, a runtime
choice bounded by the drafter's trained block size of 16; see the contract
fixture `fixtures/laguna_xs_2_1_dflash_track.json` and
`docs/dflash-track-correctness-contract.md`). The published score is the
decode-only paired speedup:

```text
raw   = mean(serial K=1 s/token) / mean(dflash s/token)   # ratio of means
score = dflash_decode_speedup = raw / noop_reference[sampled prompt]
```

The raw ratio-of-means over the accepted, thermal-gated pairs (minimum 3,
target 4) is normalised by the sampled prompt's own pinned no-op reference, so
every prompt's no-op maps to 1.0 and the score does not depend on which of the
8 hidden prompts was drawn. There is a single hard component floor on the
normalised speedup and no prefill component:

```text
dflash_decode_speedup >= 0.95      # within 5% of this prompt's own no-op
```

A run below the 0.95 floor, or with any token-fidelity failure, or that trips
the stall guardrail (max block latency > 4x p50 rejects the run, with one
gated retry), publishes no score. There is no two-sided acceptance band on the
DFlash track — the 0.95 normalised decode floor, the token-fidelity gate, and
the stall guardrail are the ranked gates.

Run `./setup.sh && ./setup-dflash.sh` (the second provisions the pinned DFlash
drafter), then `./benchmark-dflash.sh --local-iterate` or `--local-submit`
locally; local modes write an estimated local `score.json` only — the official
paired score comes exclusively from the ranked M5 run.

## Model Artifacts

By default, `setup.sh` stores the frozen reference checkpoint in a shared
Hugging Face-style cache under your home directory (so parallel clones reuse
one checkpoint):

```text
~/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4-mlx/snapshots/841778bda563a36104dd521e37d99218e46f4f25/
```

It also creates this compatibility symlink unless the path already exists:

```text
reference_weights/laguna-xs-2.1-nvfp4-mlx/
```

By default `setup.sh` downloads
`poolside/Laguna-XS-2.1-NVFP4-mlx@841778bda563a36104dd521e37d99218e46f4f25`
from the public organizer R2 mirror, with the exact Hugging Face revision as
fallback. It checks cached files
against the pinned SHA256 manifest and redownloads only missing, truncated, or
hash-mismatched files. The complete manifest covers 13 files totaling
21,568,905,520 bytes, including 5 safetensors shards; `setup.sh` requires
40 GiB free by default before starting. After a
full verification, setup writes `.mlxfast-reference-cache.lock`; later setup
runs use cheap size/mtime checks from that lock and skip the full checkpoint
hash pass when the cache is unchanged. Set
`MLXFAST_REFERENCE_CACHE_DIR` or `MLXFAST_REFERENCE_DIR` to a different local or
mounted volume when needed, or set
`MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1` when the checkpoint is provisioned externally.

The Swift transform writes benchmark-ready weights here:

```text
weights/
  config.json
  model.safetensors.index.json
  model-0000N-of-0000M.safetensors
  tokenizer.json
  tokenizer_config.json
```

The generated `weights/` tree is a runtime artifact set, not a second physical
copy on APFS: the source is already text-only (`model.*` / `lm_head.*`), so
the transform may clone complete shards copy-on-write and writes a runtime-authored
`config.json` (the Laguna geometry fields the runtime needs,
plus the checkpoint's quantization metadata). There is no expert
streaming manifest -- the whole model, including every routed expert, is
loaded fully into RAM at
init; there is no weight streaming of any kind. Submissions may adjust this
overlay by changing both `Sources/MLXFastTransform/` and
`Sources/MLXFastModel/`; correctness and benchmark results are the authority,
not byte equality with the baseline layout.

The DFlash track adds one more provisioned asset: the organizer-provided
DFlash draft model (an EAGLE-style speculator with its own fixed weights
(~924 MB BF16), block size
16, conditioned on target hidden states and borrowing the target embedding and
lm_head). It is provisioned by `./setup-dflash.sh` on top of `./setup.sh`.
Both the target checkpoint and the drafter weights are fixed — they are not a
submission surface; only the drafting/verification runtime around them is.

The public correctness-only prompt and Laguna-tokenized golden are committed
under `correctness_prompts/` so participants can run a local correctness smoke
test. The official correctness golden
is supplied by the benchmark operator and is intentionally not committed to
the public repo:

```text
correctness_golden.json
```

Use `MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json` when the
file is provisioned outside the repository root.
Benchmark CI consumes the checked-in public golden for correctness-only runs and
downloads the private precomputed correctness golden from protected storage for
full benchmark runs. The timed phase separately downloads a pinned private
evaluation prompt after the correctness scrub. The trusted box-side measurement
wrapper generates and caches a benchmark token oracle for that prompt per binary
and validates all charged outputs against it. Private prompt manifests, the
timed prompt, and hidden correctness goldens are not committed to the public
repository. Organizers regenerate correctness fixtures and rotate the timed
target through the controlled operator process.

## Editable Surface

The active editable surface is Swift-only and is defined by `benchmark.json`:

| Path | Scope |
|---|---|
| `Sources/MLXFastModel/` | Laguna XS 2.1 NVFP4 model implementation: attention (sliding-window + full, GQA, YaRN partial-rotary RoPE on full-attention layers), MoE MLP (256 routed experts + shared expert, per-token top-k routing; the per-head gate belongs to attention), RMSNorm, KV caches, weight loading, and prefill/decode execution. |
| `Sources/MLXFastTransform/` | Offline safetensors transform (text-tensor selection, config/tokenizer emission). |
| `Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlash*.swift` (the eight files listed in `benchmark.json` — `DFlashBenchmark.swift` is NOT editable), `Vendor/mlx-swift-lm/Libraries/MLXLLM/DFlashTarget.swift`, `DFlashVerifyLinear.swift` | DFlash drafting/verification runtime: block dispatch, multi-row target verification, longest-correct-prefix acceptance, and KV rollback of the rows behind rejected draft tokens. |
| Vendored Laguna model + `MLXLMCommon` plumbing + MLX Metal kernels (per `benchmark.json` `editablePaths`) | `Laguna.swift`, the KV-cache/RoPE/decode helpers, and the SDPA / quantized-matmul (NVFP4 `fp_quantized*`, `_nax`) / MoE gather-GEMM / `rope` / `rms_norm` / `softmax` / etc. kernel families the forward pass dispatches. |

`Sources/MLXFastCore/`, `Sources/MLXFastHarness/`,
`Sources/MLXFastCLI/`, scripts, tests, `benchmark.json`, generated
`weights/`, reference checkpoints, golden fixtures, and local scores are
harness/operator files, not submission surface. Correctness, scoring, timing,
golden generation, benchmark-oracle validation, and provenance checks live in
that trusted harness layer.

Account and submission management — login, clone, submit, and listing
submissions — are handled by the **Yukon CLI (`mlxfast`)**, not by
`mlxfast-swift`. The Swift binary now runs the benchmark domain only (transform,
correctness, benchmark, preflight, verify-transform, and the DFlash-track
`dflash-benchmark`/`dflash-probe`/`dflash-reference`); it no longer logs in or
uploads. The CLI installer defaults to `~/.local/bin`, so expose that directory
in the current shell before using it. Submit with:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
mlxfast login <api-key> --api <url>
mlxfast clone <benchmark-id-or-name>     # fresh checkout; an existing repo auto-links by its git remote
mlxfast submit --model "<model name>" --note "..."
mlxfast submissions
```

`mlxfast submit` reads `benchmark.json` and uploads only `editablePaths` as a
gzip tar archive with bearer-token auth; the backend applies it to the frozen
benchmark checkout and re-enforces the editable surface server-side before
running hidden validation. `--model` is required and is recorded for the
leaderboard; pass `--note-file PATH` or `--claimed-score N` as needed.
The benchmark contract also declares a local `preSubmitCommand`:
`./benchmark-dflash.sh --local-submit`. `mlxfast submit` does not run it — the upload
goes directly to official validation, and no local run blocks it. Running that
command yourself before submitting is the recommended local correctness and
timing check, without running the official hidden golden.

`mlxfast-swift verify-transform` is an organizer/debug check for deterministic
transform output. It re-runs the submitted transform and compares the generated
`weights/` tree against that fresh run. It is not a baseline-layout requirement.
The normal preflight/benchmark path also rejects generated `weights/` above the
default 25 GiB transformed-output cap before correctness or timing runs (the
text tower is about 21.6 GB, under that cap).
Override it with `MLXFAST_MAX_WEIGHTS_BYTES`; `verify-transform` additionally
accepts `--max-bytes`.

There is no Python harness path.

## Correctness Gates

Correctness is a hard gate. Each base golden case contains exactly 512 prompt
token IDs and at least 64 expected continuation token IDs. The harness checks
the first 64 continuation positions teacher-forced with temperature-zero
behavior: after each accepted step it feeds the golden previous token back into
the model. The first mismatch records only the case, step, expected token, and
actual token in the failed report.

The gate is intended as a first-stage filter: an implementation that fails it is
not eligible for the longer benchmark.

Private golden fixtures may add hidden `correctness_gates` on top of the base
teacher-forced cases:

- `anchors`: one-token checks at selected hidden contexts. These can require an
  exact expected token, explicit accepted tokens, or a bounded top-logit rank
  and delta for near-tie hardware cases.
- `free_run`: short greedy continuations whose exact prefix must match. These
  catch bugs that only appear when the model consumes its own generated tokens.
- `behavior`: GPQA-style or instruction-following prompts whose answer is
  checked exactly against precomputed accepted answer token sequences. Each
  accepted answer sequence must have at most `max_new_tokens` tokens; shorter
  sequences are matched as exact prefixes of the generated answer.

Full benchmark CI adds one more private layer after the correctness and gates
pass (and before the timed measurement, which runs last on the ranked
pipeline): it captures short answers for hidden GPQA cases and asks a Claude
judge whether each candidate is semantically equivalent to the private
reference answer. That semantic gate is pass/fail only and does not affect
the timing score; its pass-count threshold is baseline-calibrated (see
`MLXFastConstants.semanticGPQAMinPassCount`). The uploaded score records only
aggregate semantic counts and the judge model name.

The same hidden GPQA cases are also used for a TTFT guardrail: during the
hidden behavior correctness pass, the workflow times prompt prefill through
the first greedy answer token and verifies that the first token is accepted for
that case. The uploaded score records only
aggregate TTFT pass counts and timing statistics; first-token values and
accepted token sets are not logged or artifacted.

These layers keep the official gate mostly deterministic and token-based while
adding a small semantic backstop against implementations that pass the exact
prefix but damage answer meaning. The benchmark operator should keep private
prompts, accepted answer sequences, reference answers, and judge transcripts
outside the public repository.

The gate intentionally does not port a hidden-state comparison layer. The
benchmark contract cares about the externally observable text-to-text Laguna
output path, and hidden-state tensors are easier to make ambiguous around
normalization than token-level or logit-anchor checks.

VLM/image and audio inputs remain out of scope. Only the Laguna text tower
executes.

### DFlash token-fidelity gate

On top of the shared gates above, the DFlash track adds a token-fidelity gate
that keeps speculative decode honest. In the trusted parent, every emitted
token of every round is re-verified against a reference teacher-forced on the
candidate's own emitted prefix (a divergent token re-anchors every later row
to the candidate's actual chain), declared rows are re-checked in full, and
the rejected tail is priced rather than being free. No report publishes
without that reference pass — the gate fails closed. An emitted token is
admissible only if it matches the reference's own argmax (exact or at the
declared block width) or falls inside a bounded, per-thousand-token near-tie /
residual budget measured against the reference's own logits; anything else
rejects the run. `token_fidelity_gate_status` is `implemented`.

### DFlash block-decode rules (default track)

DFlash is the declared, trusted block-decode track, so target-verified block
speculative decode is the required ranked path, not an excluded one. Each
target forward may verify K draft rows proposed by the organizer-provided
drafter; the runtime accepts the longest correct prefix and rolls back the KV
rows behind rejected draft tokens, advancing logical and physical KV position
by exactly the number of accepted tokens. The trusted block protocol
(`trusted_dflash_block_v1`) and the token-fidelity gate above define what a
valid round is; the drafter and target weights are fixed.

What stays excluded is drafting that is not the provisioned DFlash drafter:
prompt-lookup decoding; n-gram, suffix, or other token-history drafting; and
any mechanism that manufactures or hardcodes an unsupplied future token
outside the trusted draft/verify protocol. Warming the pipeline before the
worker protocol hello or during model initialization does not exempt it from
the gates.

Ordinary within-request KV reuse remains allowed, as do input-independent
caches for weights, dequantized tensors, kernels, masks, or RoPE tables.
Multi-row kernels are the normal case here: every row is backed either by a
token supplied in that invocation (prefill) or by a drafted row the target
then verifies under the protocol.

The experimental trained-assistant MTP block-decode track
(`laguna-xs-2.1-mtp-v1`) that once required a separately declared trusted
block protocol was retired without going live; DFlash is that declared track,
and the rules above govern every ranked submission.

### Excluded on every track, and what the static-review gate cites

Two rule sets bind every submission and are what the trusted submission
static-review reads:

- No caches or memos keyed on a request's input tokens whose only possible hit
  is the benchmark harness repeating an identical computation — bit-identical
  output does not make that legitimate; the benchmark measures single-pass
  inference. Input-independent caches (weights, dequantized tensors, RoPE/mask
  tables keyed on shapes and offsets) and within-request KV reuse stay fine.
- The retired serial non-speculative track rules remain the reference the
  static review cites for UNsanctioned speculation: prompt-lookup decoding,
  n-gram / suffix / token-history drafting, same-target lookahead, cross-request
  future-logit/KV buffers, deferred cache rows, and commit / rollback / recommit
  markers are all excluded. Under that serial track a one-token decode request
  advances exactly one position and leaves no pending future token; DFlash's
  sanctioned organizer-drafter + target-verify block decode is the deliberate
  exception (above), but every unsanctioned future-token path stays excluded.

## Score

```text
raw   = mean(serial K=1 s/token) / mean(dflash s/token)   # ratio of means
score = dflash_decode_speedup = raw / noop_reference[sampled prompt]
```

Higher is better. This is a decode-only paired speedup: the trusted serial
K=1 target baseline's mean seconds/token divided by the candidate's DFlash
mean seconds/token, aggregated as a ratio-of-means over the accepted pairs,
with both sides measured on the same M5 in the same session behind the same
fixed 40C thermal / telemetry acceptance (the paired baseline cancels host
drift). That raw ratio is then normalised by the sampled prompt's own pinned
no-op reference (fixtures/laguna_xs_2_1_dflash_track.json
timed_prompt_pool[].noop_decode_speedup), so every prompt's no-op maps to 1.0
and the score does not depend on which hidden prompt was drawn. There is a
single hard component floor on the normalised speedup and no prefill component:

```text
dflash_decode_speedup >= 0.95      # within 5% of this prompt's own no-op
```

A run below the 0.95 floor, or with any token-fidelity failure, or that trips
the stall guardrail (max block latency > 4x p50, one gated retry), is
ineligible; the score is null when any gate fails. There is no two-sided
acceptance band on the DFlash track. `score.json` also carries the raw ratio,
the no-op reference used, the normalised paired speedup, decode
seconds/token, the floor verdict, gate results, and transformed-weight
identity.

## Useful Commands

```bash
swift test --force-resolved-versions
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
swift build -c release --force-resolved-versions
./setup.sh && ./setup-dflash.sh      # setup-dflash.sh provisions the pinned DFlash drafter
./benchmark-dflash.sh --local-iterate
./benchmark-dflash.sh --local-submit

# Submitting is done with the Yukon CLI (mlxfast), not mlxfast-swift:
mlxfast clone <benchmark-id-or-name>
mlxfast submit --model "<model name>" --note "..."
mlxfast submissions
```

Always pass `--force-resolved-versions` to direct `swift build` / `swift
test` runs: the dependency graph is frozen, and a bare invocation can
silently rewrite `Package.resolved`, after which `./setup.sh` and the
benchmark scripts (`./benchmark-dflash.sh`, which delegates to `./benchmark.sh`)
refuse to run until you restore it with `git checkout -- Package.resolved`.
