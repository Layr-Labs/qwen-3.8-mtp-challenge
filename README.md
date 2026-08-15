# mlxfast — Qwen 3.8 27B native MTP

A benchmark arena for compute-optimal LLM inference on Apple Silicon.

> **NOT LIVE — 3.6 → 3.8 CUTOVER IN PROGRESS.** This repository is being
> renamed `Layr-Labs/qwen-3.6-mtp-challenge-dev` →
> `Layr-Labs/qwen-3.8-mtp-challenge-dev` and the ranked track re-pinned from
> Qwen 3.6 27B onto **Qwen 3.8 27B**. Nothing here can score today: the
> ranked workflow trusts the post-rename slug, its calibration interlock is
> `"0"`, the contract fixture's two enablement flags are `false`, and every
> hidden-artifact pin reads the literal `QWEN38-PENDING-RELEASE`.
>
> **Backbone identity 2026-08-14.** The reference checkpoint is **our own**
> MLX 4-bit affine / group-64 conversion, produced under a pinned mlx 0.32.0
> toolchain, of the official bf16 base `Qwen/Qwen3.8-27B` @
> `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, publishing as
> `EigenLabs/Qwen3.8-27B-4bit`. It replaces a third-party conversion adopted
> earlier the same day and **terminated** by the operator's validation
> kill-switch — the deterministic reconversion cross-check found 994 of 1,847
> tensors numerically different from our own conversion. Geometry is unaffected
> and stays **identical to 3.6** — 64 layers on the same 4-layer hybrid repeat,
> vocab 248320, hidden 5120, 24 attention heads / 4 KV heads, head_dim 256 — so
> the geometry hedges stay discharged.
>
> **Both halves are published** as of 2026-08-14: the backbone at
> `eda45ab47f465d08d6558f0353a2346e2eb9d5b3` with a generated 10-record byte
> manifest, and the **MTP head** — `EigenLabs/Qwen3.8-27B-MTP-bf16`, 15 bf16
> tensors extracted from the same bf16 base — at
> `26a328e070875b0314d652a039b6b59902690f03` with a 4-record one (the
> republish that added the `config.json` and `model.safetensors.index.json`
> the head loader refuses to run without). Both
> repositories are **private**, so provisioning either needs a Hugging Face
> token with read access. What is still unpublished is every hidden and public
> artifact (`QWEN38-PENDING-RELEASE`), which is what keeps the track inert.
>
> Values still carrying a **3.6 measurement** behind
> that closed gate are marked `QWEN38-VERIFY-AT-RELEASE` and must be
> re-derived, not re-read. See [`docs/CUTOVER.md`](docs/CUTOVER.md);
> [`docs/qwen-mtp-go-live-runbook.md`](docs/qwen-mtp-go-live-runbook.md) is
> the record of the **3.6** go-live and its ids and numbers are 3.6-era
> throughout.

The default — and only — ranked track in **this** repository is
`qwen3.8-27b-mtp-v1`: the Qwen 3.8 27B target drafts a block with its own
native MTP head, verifies the block against itself, and decode advances by
the longest correct prefix. `benchmark.json` IS the Qwen-MTP manifest
(benchmark name `mlxfast-challenge-dev-qwen38-mtp`), the ranked pipeline is
[`.github/workflows/qwen-mtp-ranked-benchmark.yml`](.github/workflows/qwen-mtp-ranked-benchmark.yml),
and the track contract is `fixtures/qwen3_8_27b_mtp_track.json`. See
[`docs/qwen-mtp-go-live-runbook.md`](docs/qwen-mtp-go-live-runbook.md) for
the go-live procedure and [`docs/CUTOVER.md`](docs/CUTOVER.md) for the
operator cutover state.

> **CONTRACT CHANGE — 2026-08-14 (operator-ratified).** Two things moved
> together and they are one change. **(1) The editable surface is now the UNION**
> of MLX runner/kernel optimisation — what this challenge family has always been
> — and the *whole speculative apparatus*: the drafting code, the per-round draft
> depth and schedule (0…8 drafts per round, adaptive, non-drafting rounds legal),
> and the **MTP head weights themselves**, declared in
> [`mtp-head.manifest.json`](mtp-head.manifest.json) and fetched by the runner.
> **(2) Scoring is re-anchored at serial = 1.0**: the published score is the
> median of the eight per-prompt *raw* serial-relative speedups, floored at 0.90
> and ceilinged at 3.0, with no normalisation step. Serial decode is 1.0 by
> construction, and an unmodified tree need not be: on Qwen 3.6 it measured
> ~0.935, a real regression rather than the zero point. `QWEN38-VERIFY-AT-RELEASE`
> — that figure, the floor and the ceiling are all 3.6-era and are re-derived
> during the 3.8 bring-up. The definitive editable-vs-trusted
> table is [`docs/qwen-mtp-editable-surface.md`](docs/qwen-mtp-editable-surface.md);
> the scoring derivation is in the go-live runbook.

> **REPO SPLIT — 2026-08-13.** This repository was split from
> `Layr-Labs/mlxfast-challenge-dev@ba5f9703`, and much of the prose below is
> inherited from it and still describes the Poolside Laguna XS 2.1 DFlash
> track `laguna-xs-2.1-dflash-v1` — an organizer-provided DFlash draft model
> proposes a block of tokens and the target verifies them per forward. That
> track stays ranked in `mlxfast-challenge-dev` and is **not** ranked here;
> its sources, scripts and docs
> ([`docs/dflash-track-correctness-contract.md`](docs/dflash-track-correctness-contract.md),
> [`docs/dflash-go-live-runbook.md`](docs/dflash-go-live-runbook.md)) are
> retained in-tree for reference only. Where this README and the manifest
> disagree, `benchmark.json` and the Qwen-MTP sources named above win.

The earlier serial tracks `laguna-xs-2.1-serial-v1` and
`laguna-xs-2.1-serial-v2` are retired — the serial ranked workflow was
deleted. Each track scores on its own leaderboard namespace; submissions and
scores are never compared with or migrated across namespaces.

## Quickstart

```bash
# Build the Swift/Metal runtime, download the reference checkpoint, then
# stage the organizer-provided DFlash draft model.
./setup.sh && ./setup-dflash.sh

# Fast local edit-loop signal (correctness smoke + local timing estimate).
./benchmark-dflash.sh --local-iterate

# Longer local pre-submit signal.
./benchmark-dflash.sh --local-submit
```

Full model setup needs a moderate local SSD. The reference checkpoint is
`poolside/Laguna-XS-2.1-NVFP4-mlx` at revision
`841778bda563a36104dd521e37d99218e46f4f25`, with 5 safetensors shards
and 21,568,905,520 bytes across all 13 files. `setup.sh` downloads it from
the public organizer R2 mirror
(`https://ds4.darkbloom.ai/laguna-xs-2.1-nvfp4-mlx`) by default, with the
exact pinned Hugging Face revision as fallback and up to 3 shard
downloads in parallel (`MLXFAST_REFERENCE_DOWNLOAD_JOBS`), into a shared
Hugging Face-style cache under
`~/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4-mlx/snapshots/841778bda563a36104dd521e37d99218e46f4f25/`
(in `$HOME` by default so parallel clones reuse one checkpoint).
It verifies cached files against
`fixtures/reference_laguna_xs_2_1_nvfp4_mlx.sha256`
and redownloads only files that are missing, truncated, or hash-mismatched.
A compatibility symlink is created at
`reference_weights/laguna-xs-2.1-nvfp4-mlx`
for older commands, but current setup and CI pass the canonical cache directory
to transform explicitly. The downloader uses resumable `curl` requests, prints
numbered shard progress with elapsed time, and checks for at least 40 GiB free
by default. After a full SHA-256 verification, setup writes
`.mlxfast-reference-cache.lock` next to the checkpoint; later setup runs use
cheap size/mtime checks against that lock and skip the full hash pass
when the cache is unchanged. Use
`MLXFAST_REFERENCE_CACHE_DIR=/Volumes/ssd/hf-cache/.../snapshots/<revision>` or
`MLXFAST_REFERENCE_DIR=/Volumes/ssd/laguna-xs-2.1-nvfp4-mlx` to point at a larger
volume, or `MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh` when the checkpoint will
be supplied separately. If you use a custom cache path, either copy the exact
transform command printed by `setup.sh` or set `MLXFAST_REFERENCE_DIR` before
running `transform` or `benchmark.sh`. The Swift CLI also honors
`MLXFAST_REFERENCE_DIR`, `MLXFAST_WEIGHTS_PATH`,
`MLXFAST_CORRECTNESS_GOLDEN_PATH`, and `MLXFAST_SCORE_PATH` as defaults;
explicit CLI flags take precedence. For `benchmark.sh`, use those `MLXFAST_*`
environment variables for path overrides; pass `--weights`, `--golden`, and
`--score-path` only to `.build/release/mlxfast-swift benchmark` directly. Set
`MLXFAST_REFERENCE_BASE_URL` to use another HTTP checkpoint prefix serving
the same manifest-pinned Poolside files,
and `MLXFAST_REFERENCE_AUTH_HEADER` to pass an auth
header to a private checkpoint endpoint. Run `./setup.sh --help`
for the full local setup knobs. After `setup.sh`, run `./setup-dflash.sh` to
stage the organizer-provided DFlash draft model that the block-decode track
requires.

> **Correctness fixtures are M5-generated.** The checked-in goldens can hit
> near-tie argmax differences on other Apple Silicon generations; the ranked
> M5 result is authoritative. If unmodified `main` diverges at the same token
> position on your machine, rerun the local mode with
> `MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1` to keep the timing estimate — the
> score then records `passed_correctness: false` and the diverging tokens,
> so the divergence is never hidden.

### Ranked workflow

Yukon dispatches `.github/workflows/dflash-benchmark.yml`, the DFlash ranked
pipeline (the retired serial `benchmark.yml` is deleted). Unlike local
`setup.sh`, ranked M5 jobs never download a checkpoint: they verify the
pre-provisioned Poolside target cache and the DFlash draft-model cache against
their pinned manifests, build and transform submitted code in the sandbox, run
the public drift tripwire, then the hidden teacher-forced base case plus the
anchor/free-run/behavior/GPQA gates, the semantic GPQA judge, and the DFlash
token-fidelity gate. (The workflow drives those shared gates by invoking
`benchmark.sh --official` internally; that is a harness detail, not the
participant entrypoint.)

Timing runs last, behind the fixed 40C thermal gate. One prompt is sampled
uniformly at random from an 8-prompt hidden pool per run, and the trusted
on-box measure-job times the DFlash candidate and the trusted serial K=1
target oracle back to back on the same silicon over that prompt; the paired
ratio cancels host drift. The published score is decode-only:

```text
raw   = mean(serial K=1 s/token) / mean(dflash s/token)   # ratio of means
score = dflash_decode_speedup = raw / this prompt's pinned no-op reference
```

The raw paired ratio of means over the accepted thermal-gated pairs is
normalised by the sampled prompt's own pinned no-op speedup, so every prompt's
no-op maps to 1.0 and the score does not depend on which hidden prompt was
drawn. The one hard component floor is `0.95` (within 5% of the prompt's own
no-op); a token-fidelity failure, throttled sample, or invalid telemetry fails
the run. `score.json` publishes the raw ratio, the reference used, the
normalised speedup, and the floor verdict. See
[`docs/private-benchmark-security.md`](docs/private-benchmark-security.md)
for isolation details and
[`docs/dflash-track-correctness-contract.md`](docs/dflash-track-correctness-contract.md)
for the frozen timed window and fidelity contract.

## Why this challenge exists

Poolside Laguna XS 2.1 is a fine-grained MoE text model (256 routed experts
plus one shared expert per sparse layer, 8 experts per token, per-head
gating). The pinned Poolside export is already text-only (`model.*` / `lm_head.*`
tensors). Its sparse routed/shared expert projections are NVFP4 4-bit
group-16; attention, embeddings, the untied lm_head, layer-0 dense MLP, and
routers remain BF16. The 13-file checkpoint is exactly 21,568,905,520 bytes,
small enough to load
entirely into unified memory once at process startup on the official runner
(a self-hosted Apple M5 Max with 128 GB of unified memory, runner label
`m5-laguna-dflash`). There is no weight streaming: the model is RAM-resident
before scored decode.

At process startup, machines with less than 64 GiB select a low-memory
profile automatically. The profile is pure memory management: the MLX
allocator cache is capped at 6 GiB, command buffers are shortened, and free
warmup buffers are released before the worker begins serving requests. It
does not disable any code-path or output-affecting feature — the
compiled-decode fusions run everywhere, so a local run exercises the same
code path as the ranked box all the way down to the ~36 GiB practical local
minimum. A machine too small for the model plus the decode working set
fails loudly with an out-of-memory error rather than silently diverging
from ranked behavior. The profile announces itself on stderr and can be
forced either way with `DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low|auto`.
This changes local speed only; the 128 GB ranked runner keeps the full
profile.

That does not mean there is nothing left to optimize. Attention alternates
three sliding-window layers (512-token window, 64 heads) with one
full-attention layer per block of four (48 heads, YaRN rotary with a 0.5
partial-rotary factor; sliding layers use plain RoPE at theta 10000). All
layers are GQA with 8 KV heads at head_dim 128. Layer 0 has a dense MLP
(intermediate 8192); every other layer routes tokens through 8 of 256
experts plus a shared expert (per-head gating, MoE intermediate 512).
Only routed/shared expert projections are NVFP4-quantized; the whole forward
pass runs through MLX's kernel scheduler on every decode step. Kernel selection,
quantized matmul
dispatch, MoE expert gathering, KV-cache handling, attention masking, and MLX
graph/scheduling
overhead are all optimisation targets — and so are the vendored MLX Metal
kernels themselves, which are part of the editable surface (see "The
modifiable surface" below). On this track those same kernels are dispatched by
both the target's multi-row block verification and the DFlash draft model, and
the block-decode runtime itself — draft dispatch, multi-row verification, and
KV rollback of rejected rows — is editable and on the hot path. The generated
`weights/` tree is
expected to stay small: it is a runtime artifact overlay on top of the frozen
reference checkpoint (a straight text-tensor subset plus a runtime-authored
`config.json`), not a second full model copy. Submissions may change the
Swift transform, the Swift runtime, and the vendored Laguna model and
kernel sources, as long as the generated runnable artifacts pass the hidden
correctness and benchmark checks.

## The modifiable surface

Unlike typical inference benchmarks, the entire model execution pipeline is
in scope — including the vendored Laguna model code and the MLX Metal
kernels it runs on. The authoritative list is `editablePaths` in
`benchmark.json` (currently 89 entries), in five groups — the first of which is
new on 2026-08-14 and is the speculative apparatus itself:

| Path | What it controls |
|---|---|
| `mtp-head.manifest.json`, `mtp-head/` | **The MTP head itself.** Declare a head (`source`, `sha256`, `bytes`) and the runner fetches and digest-verifies it before the sandbox opens; ship a small one in `mtp-head/` for direct-dispatch testing. Absent declaration = the organizer-pinned head, which is the normal case. A head only *proposes* — the pinned target still decides every emitted token — which is why this can be yours. |
| `Sources/MLXFastModel/Qwen36MTP*.swift` | **The drafting code and the draft schedule.** `Qwen36MTPBlockSession.draftPolicy` is the shipped per-round schedule (a constant 2) and is the first thing worth changing: any count from 0 to the trusted maximum of 8 is legal, per round, adaptively. |
| `Sources/MLXFastModel/` | Qwen 3.6 runtime: weight loading, attention, MoE MLP, KV caches, prefill/decode execution. **Primary target.** |
| `Sources/MLXFastTransform/` | Offline reference-checkpoint transform into benchmark-ready `weights/`. |
| `Vendor/mlx-swift-lm/Libraries/` (listed files) | The vendored Laguna model (`MLXLLM/Models/Laguna.swift`), the DFlash target adapters (`MLXLLM/DFlashTarget.swift`, `MLXLLM/DFlashVerifyLinear.swift`), the DFlash block-decode runtime this track adds (the eight `MLXSpeculative/DFlash*.swift` files listed in `benchmark.json` — draft dispatch, batched multi-row verification, greedy rounds, KV rollback of rejected rows, draft-model configuration; `DFlashBenchmark.swift` matches the glob but is not editable), and the `MLXLMCommon` plumbing they use directly (MoE/attention dispatch helpers, KV caches, RoPE utilities/application, compiled decode, evaluation). |
| `Vendor/mlx-swift/Source/Cmlx/` (listed files) | The MLX Metal kernels Laguna dispatches — SDPA (`steel/attn`, `sdpa_vector`), NVFP4 `fp_quantized` matmul plus shared quantized dispatch (incl. `_nax`), MoE gather GEMM (`steel_gemm_gather*`), `steel/gemm`, `gemv`, `rope`, `rms_norm`, `softmax`, `sort`, `reduce`, `copy`, elementwise, `arg_reduce`, gather indexing — as AOT `.metal`/`.h` sources and their JIT `mlx-generated/*.cpp` twins. |

Two build forms matter for kernel edits, because the vendored MLX package
builds in JIT mode. Families with an `mlx-generated/*.cpp` twin (quantized
incl. `fp_quantized`, steel/gemm incl. the gather GEMM, steel/attn, gemv,
softmax, sort, reduce, copy, elementwise, gather) are compiled at
runtime from the C++ source strings embedded in those files — the twin is
the runtime-effective source, so edit it (and keep the readable
`.metal`/`.h` pair in sync). RoPE, RMSNorm, the SDPA vector kernel, and
`arg_reduce` load ahead-of-time from `mlx.metallib`, which
`tools/build-mlx-metallib.sh` (run by `./setup.sh`) compiles from the
vendored `.metal` sources — rerun it after editing those. `_nax` names are
the M5-generation kernel variants the ranked runner selects. After a kernel
edit: rerun the metallib build for AOT edits, then
`./benchmark.sh --local-iterate` (which rebuilds both binaries whenever a
build input is newer than them and refreshes the cached `weights/`), followed
by `./benchmark-dflash.sh --local-iterate` (which reuses those binaries and
never rebuilds them). A bare `swift build -c release`
is not enough on its own — without `--scratch-path .build-worker` it writes
`.build/release`, while the scored binary is
`.build-worker/release/mlxfast-runtime-worker`. Prioritize kernels reached
by the seed prefill and the timed DFlash decode phase (draft, verify, and
rollback).

Participant model and kernel code — `MLXFastModel` plus the vendored forks
— builds into the sandboxed `mlxfast-runtime-worker` binary. The trusted
`mlxfast-swift` binary owns correctness, scoring, timing, and provenance,
links no MLX, model, or kernel code, and drives the worker over a JSON
protocol. `Package.swift`/`Package.resolved` and the dependency graph are
frozen, and the rest of the vendored forks (other model families, shared
factory/tokenizer plumbing, and kernels Laguna does not dispatch) stay
non-editable. Kernel changes are bound by the same
hidden correctness gates as model changes: keep them prompt-independent and
model-general, and be conservative with numeric reassociation, which can
flip near-tie greedy argmaxes on the M5.

The repository is Swift-only (no Python): setup, transform, correctness,
and benchmark all run through the Swift package, plus the
`tools/build-mlx-metallib.sh` step for the vendored AOT Metal sources.

Submissions are made with the **Yukon CLI (`yukon`)**, a separate tool that
manages your account and uploads across all Yukon benchmarks. Always use
`yukon` for account and submission commands. The
`mlxfast-swift` binary runs the benchmark domain only (transform, correctness,
benchmark, preflight, verify-transform) and no longer logs in or uploads.

The Yukon CLI is installed by the external Yukon installer from your
challenge onboarding instructions, not by this repository or `./setup.sh`. If
`yukon` is not found after installing it, the installer's bin directory
(typically `~/.local/bin`) is not on your PATH; activate it with:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

`./setup.sh` checks for `yukon` at the end of setup and prints this same
remediation (with the detected directory) when the CLI is installed but not
activated on PATH. For the current shell only, the first line below exposes
the CLI's default install directory without editing your shell rc:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
yukon login <api-key> --api https://yukon-api.fly.dev
yukon clone <benchmark-id-or-name>     # fresh checkout; an existing repo auto-links by its git remote
yukon submit --model "Claude Opus 4.8" --note-file submission-note.md
yukon submissions
```

`yukon submit` reads `benchmark.json` and uploads only the paths listed in
`editablePaths` as `submission.tar.gz`, POSTed to Yukon with
`Authorization: Bearer <api-key>` and an idempotency key. Generated `weights/`,
reference checkpoints, golden files, and local scores live outside
`editablePaths` and are never uploaded; the backend re-enforces the editable
surface server-side after upload. `--model` is required and is recorded for the
leaderboard. `YUKON_API_URL` / `YUKON_API_TOKEN` configure the endpoint and
token for scripted runs.
`yukon submit` uploads directly: it does not run the contract
`preSubmitCommand` (`./benchmark-dflash.sh --local-submit`), and no local run
blocks the upload — the official M5 run is the gate. Run
`./benchmark-dflash.sh --local-submit` yourself before submitting: it runs the
public correctness fixture and a longer local DFlash decode timing pass, writes
`score.json`, and catches obvious correctness or speed regressions before
they spend official runner time.

## Local Commands

Use these modes for local development:

| Command | Purpose | What it checks | Output |
|---|---|---|---|
| `./benchmark-dflash.sh --local-iterate` | Fast directional edit loop. | Public-fixture correctness (teacher-forced) plus a short local DFlash decode timing pass. | `score.json` with a local estimated score. |
| `./benchmark-dflash.sh --local-submit` | Longer pre-submit signal. | Same correctness over a longer local DFlash decode timing pass. | `score.json` with a local estimated score. |

Both modes require the transformed `weights/` tree that
`./benchmark.sh --local-iterate` produces and caches, and the drafter staged
by `./setup-dflash.sh` (each mode aborts with instructions if either is
missing), then run the checked-in public correctness fixture and time DFlash
block decode locally. Local scores are estimates for direction only; the
official paired decode speedup exists only on the ranked M5 runner, measured
against the trusted serial K=1 target oracle.

## Scoring

> **Qwen-MTP track (`qwen3.8-27b-mtp-v1`), operator-ratified 2026-08-14.** The
> prose in the rest of this section is inherited DFlash prose; for the ranked
> track of this repository the rule is:
>
> ```text
> per prompt p:  raw_p = mean(serial depth-0 s/token) / mean(candidate s/token)
> published:     score = median(raw_1 .. raw_8)      # 8 hidden pool prompts
> ```
>
> **Serial decode is 1.0.** Both means come from the same thermally-gated
> session for that prompt, in alternating order, so the serial leg is the
> normaliser and it is measured rather than pinned. There is no normalisation
> step and no prompt lottery: a ranked run times all eight prompts and the
> median (mean of the two central order statistics — 8 is even) is the score.
>
> An **unmodified tree scores ~0.935**, not 1.0. The shipped depth-2
> configuration is a measured ~6.5% regression against true serial decode at
> the 512-token window; the leaderboard says so rather than defining it away.
> A candidate that **stops drafting entirely** is the serial control and scores
> exactly 1.0 — a legal, unremarkable submission.
>
> Floor `0.90` on the published median — "do not regress serial by more than
> 10%" — and ceiling `3.0` (a plausibility bound). The floor is a regression
> bound, not a quality bar: you may publish below 1.0 and the board will say so.
> If your change cannot beat serial, drafting nothing scores exactly 1.0 and is
> a legal submission. Token fidelity is unchanged and absolute: every emitted
> token must equal the serial trajectory.
>
> The pool's pinned `noop_decode_speedup` values are **retained and
> informational** — they are the shipped configuration's own per-prompt
> regression profile, published next to your raw ratio. Nothing divides by them.


```text
raw   = mean(serial K=1 s/token) / mean(dflash s/token)   # ratio of means
score = dflash_decode_speedup = raw / noop_reference[sampled prompt]
```

Higher is better. The score is decode-only: the DFlash candidate and the
trusted serial K=1 target oracle are timed over the same hidden prompt on the
same M5 in the same session, behind the same fixed 40C thermal gate and
telemetry acceptance, and the ratio of their mean seconds/token cancels host
drift. There is no separate prefill component — the seed prefill is charged
inside the decode window. The ranked decode window is 512 parent-counted
tokens at block size K=2. The raw paired ratio of means over the accepted
thermal-gated pairs (at least 3 accepted, target 4) is then normalised by the
sampled prompt's own pinned no-op reference, so every prompt's no-op maps to
1.0 and your score does not depend on which of the 8 hidden prompts was drawn.
There is a single hard component floor on the normalised speedup:

```text
dflash_decode_speedup >= 0.95      # within 5% of this prompt's own no-op
```

A run below the floor, a throttled or telemetry-invalid sample, or a block
whose latency exceeds 4x the median (the stall guardrail) fails the run, with
one gated retry. There is no two-sided acceptance band on this track.

To defeat submit-until-green tuning, the timed prompt is drawn uniformly at
random from an 8-prompt hidden pool of distinct domains once per ranked run;
the selected index is recorded only in the private audit directory.

Correctness is a hard gate on top of scoring: the teacher-forced base case,
the hidden anchor/free-run/behavior/GPQA gates, GPQA TTFT, the semantic GPQA
judge, the public drift tripwire, and the DFlash token-fidelity gate must all
pass, or the score is null. The token-fidelity gate re-verifies every emitted
token in the trusted parent against a reference teacher-forced on the
candidate's own emitted prefix, with a bounded near-tie budget; declared rows
are re-checked in full and the rejected tail is priced. The whole model is
RAM-resident with no weight streaming (`bandwidth_gb_per_token=0`). RAM and
phase-timing metrics are still reported for operator review and future
guardrails; they are not primary score factors. See
[`docs/dflash-track-correctness-contract.md`](docs/dflash-track-correctness-contract.md)
for the full correctness and fidelity specification.

## Architecture

```
Sources/
  MLXFastCLI/                trusted CLI entrypoint (mlxfast-swift)
  MLXFastCore/               score.json, golden cases, shared contracts
  MLXFastTransform/          editable Swift offline weight transform
  MLXFastModel/              editable Poolside Laguna XS 2.1 NVFP4 Swift runtime
  MLXFastTrustedHarness/     trusted correctness, golden, and benchmark runner
  MLXFastHarness/            worker-side runtime support (builds into the worker)
  MLXFastRuntimeWorkerCLI/   sandboxed participant worker (mlxfast-runtime-worker)
Vendor/
  mlx-swift/                 pinned MLX fork; the listed kernel sources are editable
  mlx-swift-lm/              pinned mlx-swift-lm fork; the Laguna model, the DFlash runtime (the listed MLXSpeculative/DFlash* files, DFlashTarget/DFlashVerifyLinear), and the listed MLXLMCommon files are editable
weights/                     transformed weights (harness loads from here)
  config.json                 runtime-authored text-tower config
  model.safetensors.index.json
~/.cache/huggingface/hub/... canonical frozen Poolside NVFP4 reference cache
reference_weights/...        compatibility symlink to the reference cache
correctness_prompts/         Laguna-tokenized public correctness prompt and checked-in golden
correctness_golden.json      hidden benchmark correctness cases
score.json                   written after each benchmark run
```

The runtime loads every text-tower tensor from `weights/` into unified memory
once at process init and keeps them resident for the process lifetime; there
is no streaming path and no dependency on the frozen reference checkpoint at
runtime (only the offline transform reads the reference checkpoint).

The standard preflight/benchmark path enforces a default 25 GiB cap on the
generated `weights/` tree before correctness or timing runs (the text tower is
about 21.6 GB, inside that cap). Change it with
`MLXFAST_MAX_WEIGHTS_BYTES`; use `0`, `none`, or `unlimited` only for organizer
debugging. For stricter organizer-side provenance, set
`MLXFAST_VERIFY_TRANSFORM=1` when running `benchmark.sh`. That re-runs the
submitted Swift transform into a clean temporary directory and fails unless
`weights/` is byte-equal to that fresh run. This checks determinism and stale
files; it does not require the baseline `weights/` layout. `verify-transform`
uses the same default cap and can also be changed with
`mlxfast-swift verify-transform --max-bytes N`.

### Correctness fixtures

The public correctness prompt and golden live in `correctness_prompts/`.
These fixtures are generated on the ranked M5 hardware against the Poolside
Laguna NVFP4 reference: the prompt text is tokenized with the Laguna tokenizer
(512 prompt tokens) and the expected tokens are greedy reference
continuations captured with `mlxfast-swift generate-golden`. Private prompt
manifests and hidden benchmark golden files are not committed or generated by
the benchmark workflow. In private benchmark CI, the normal path downloads
precomputed, content-addressed objects below
`correctness_prompts/laguna-xs-2.1-dflash/`: hidden correctness and GPQA
reference goldens are each pinned by SHA-256 and byte count, and the timed
decode target is drawn uniformly at random per run from the 8-entry
`timed_prompt_pool` (one `pool-<domain>.json` object per entry, spanning
science, history, biography, philosophy, political philosophy, drama, and
economics), every entry independently pinned by SHA-256 and byte count and
re-verified byte-for-byte after download. The workflow merges the GPQA
reference into the local golden as 9 hidden behavior checks. Generate final
hidden benchmark goldens outside the public repository and upload the
resulting files to those protected private R2 paths. `dflash-benchmark.yml`
keeps raw hidden material in a runner-only private directory, not the
repository workspace, scrubs every hidden byte out of the bench workspace
before the timed measurement, and uploads only hash and byte-count sidecars.
The semantic GPQA answer and judge result files are also kept under the
private runner directory and are not uploaded.

The older participant-facing Swift `make-golden` generator has been removed
from the public harness; the last commit on this branch containing it is
`bcc9438fabf95a9b371d5749dd64f2f5ccc60fd5`. Golden generation is operator work
(the `generate-golden` capture tool described above): benchmark CI consumes
precomputed, pin-verified correctness fixtures and never regenerates them,
while the ranked timed run's emitted tokens are checked by the trusted
parent's DFlash token-fidelity re-verification rather than any
participant-visible golden (see
[`docs/private-benchmark-security.md`](docs/private-benchmark-security.md)).

Each base correctness prompt must contain exactly 512 token IDs. The benchmark
prompt must contain at least 512 token IDs. The precomputed golden file stores
exact expected tokens for each 512-token correctness prompt continuation, the
512-token decode seed, and at least 512 tokens for the timed DFlash decode
window. During correctness, the harness checks the first 64
public continuation positions by default, plus hidden
behavior gates in official benchmark runs. It checks those continuation
positions teacher-forced: after each accepted step it feeds the
golden previous token back into the model. This keeps the gate stable across
Apple GPU/software differences by preventing one earlier mismatch from
cascading into unrelated later-token failures. A token is accepted only when it
matches the expected token, except for a true top-logit tie within the tiny
`1e-6` logit tolerance used by the harness.

Private fixtures can also include a `correctness_gates` object with hidden
anchor logits, short free-run prefixes, and answer-token behavior checks.
Those gates are additive: public local correctness still works with the
checked-in fixture, while official benchmark fixtures can cover more adversarial
behavior without exposing prompt or answer data. Behavior checks compare
accepted answer prefixes against up to `max_new_tokens` generated tokens, which
lets hidden GPQA questions require only a one-letter answer while tolerating
tokenizer whitespace variants.

## License and attribution

This repository's harness code is licensed per [LICENSE](LICENSE). The
Poolside Laguna model the harness downloads and benchmarks
(Laguna XS 2.1 NVFP4, © Poolside) is licensed OpenMDW-1.1 with terms at
<https://huggingface.co/poolside/Laguna-XS-2.1>; no model weights are
distributed in this repository. Full third-party attribution — models,
linked Swift packages, and the Apache-2.0 text — is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Requirements

- Apple Silicon Mac with enough unified memory for the ~21.6 GB RAM-resident
  model plus KV cache and buffers (roughly 36 GiB practical minimum; the
  ranked runner is a single self-hosted Apple M5 Max with 128 GB, so local
  timings — and, on non-M5 machines, near-tie greedy tokens — are
  directional only)
- macOS Sequoia or later
- Swift 6 through Xcode or Xcode Command Line Tools
- Xcode Metal Toolchain for `mlx.metallib`; `./setup.sh` tries
  `xcodebuild -downloadComponent MetalToolchain`, but users with only Command
  Line Tools may need full Xcode installed, opened once, and licensed with
  `sudo xcodebuild -license accept`
- CMake, installed by `./setup.sh` via Homebrew when missing and used by `tools/build-mlx-metallib.sh` to build `mlx.metallib`
