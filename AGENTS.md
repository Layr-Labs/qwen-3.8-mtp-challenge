# MLXFast Challenge Agent Guide

This repository is the Swift-only Poolside Laguna XS 2.1 NVFP4 MoE inference
optimization challenge.
Use this file as the working contract for coding agents and participants.

## Goal

Optimize Poolside Laguna XS 2.1 NVFP4 (text tower only) inference on Apple
Silicon without
changing the observable model behavior required by the correctness gates.

> **CONTRACT CHANGE — 2026-08-14 (operator-ratified).** For the ranked track of
> this repository, two things moved together. **(1) The editable surface is the
> UNION** of MLX runner/kernel optimisation and the *whole speculative
> apparatus*: drafting code, per-round draft depth and schedule (0…8, adaptive,
> non-drafting rounds legal), and the MTP head weights themselves via
> `mtp-head.manifest.json`. **(2) Scoring is anchored at serial = 1.0** — the
> median of eight per-prompt RAW serial-relative speedups, floor 0.90, ceiling
> 3.0, no normalisation. An unmodified tree scores ~0.935 and holds the board. Read
> `docs/qwen-mtp-editable-surface.md` for the definitive editable-vs-trusted
> table and the "Qwen 3.6 MTP track" section below for the working contract.
> **Every scoring and surface statement in the inherited DFlash prose below is
> about a different track and does not apply here.**

> **REPO SPLIT — 2026-08-13.** The default — and only — ranked track in
> **this** repository is `qwen3.8-27b-mtp-v1` (benchmark name
> `mlxfast-challenge-dev-qwen38-mtp`), the Qwen 3.8 27B native-MTP
> speculative-decode track described under
> "[Qwen 3.8 MTP track](#qwen-38-mtp-track-ranked-track-of-this-repository)"
> below. `benchmark.json` IS the Qwen-MTP manifest and
> `.github/workflows/qwen-mtp-ranked-benchmark.yml` is the ranked pipeline.
> This repository was split from `Layr-Labs/mlxfast-challenge-dev@ba5f9703`,
> so the Laguna/DFlash prose in the sections that follow is inherited: the
> DFlash track `laguna-xs-2.1-dflash-v1` stays ranked in
> `mlxfast-challenge-dev` and is **not** ranked here. Its sources, scripts,
> fixtures and docs are retained in-tree for reference only. Where this file
> and `benchmark.json` disagree, the manifest wins.

The DFlash target-verified block speculative-decode track
`laguna-xs-2.1-dflash-v1`, described in this section and the sections that
follow, rewards faster decode against a paired on-box serial baseline
measured in the same session:

```text
raw   = mean(serial K=1 seconds/token) / mean(dflash seconds/token)   # ratio of means
score = dflash_decode_speedup = raw / noop_reference[sampled prompt]
```

Higher is better. The score is a decode-only paired speedup: the trusted
serial K=1 target's seconds/token over the candidate's DFlash seconds/token,
aggregated as a ratio of means across the accepted thermally-gated pairs, then
normalised by the sampled prompt's own pinned no-op reference so the score is
independent of which hidden prompt was drawn (every prompt's no-op maps to
1.0). Both sides are measured on the same machine behind the same thermal gate.
There is no prefill component — the seed prefill is charged inside the decode
window. The single normalised decode-speedup floor is `0.95` (within 5% of the
prompt's own no-op), hard, block size is `K=2`, and every emitted token must
clear the DFlash token-fidelity gate. In `mlxfast-challenge-dev` that track's
manifest is `benchmark.json` (name `mlxfast-challenge-dev-dflash`) and
`.github/workflows/dflash-benchmark.yml` is its ranked pipeline; **in this
repository `benchmark.json` is the Qwen-MTP manifest instead** and the
retained `dflash-benchmark.yml` is not dispatched here. The former serial
track `laguna-xs-2.1-serial-v2` is retired: its ranked workflow
`.github/workflows/benchmark.yml` was deleted.

## Official Hardware

Ranked benchmark runs execute through GitHub Actions on a single self-hosted
Apple M5 Max machine with 128 GB of unified memory. The runner label
configured in `.github/` is the source of truth; today that is:

```text
m5-laguna-dflash
```

The box is operator-supervised: each ranked job runs on a fresh ephemeral
runner registration, every invocation of submitted code (build, transform,
correctness, benchmark) executes sandboxed, and the machine's protected
surface is integrity-audited between jobs — drift quarantines the box instead
of publishing a score. The pipeline verifies the pre-provisioned reference
checkpoint against the pinned manifest, builds and transforms, runs the
public drift tripwire and the hidden correctness/gates pass, then runs the
timed paired measurement LAST. Every timed phase starts only once the GPU has
cooled below a fixed 40C gate and rejects throttled or telemetry-invalid
measurements. See `.github/workflows/dflash-benchmark.yml` for the exact step order.

Because the candidate and the pinned on-box baseline are measured back to
back on the same silicon behind the same thermal gate, the paired speedup
ratio cancels host drift; the score is that ratio, not a comparison against a
stored constant. Poolside Laguna XS 2.1 NVFP4 is a fine-grained MoE model (256 routed
experts plus one shared expert per sparse layer, 8 experts per token routed
per-token top-k; attention carries a per-head output gate; layer 0 is a dense
MLP): the text tower is about
21.6 GB in Poolside NVFP4, fully RAM-resident on the ranked box — the runtime loads
every text-tower tensor, including all experts, once during untimed
initialization and keeps it
resident for the whole process lifetime. There is no weight streaming of any
kind, no expert cache, and no disk I/O on the scored prefill/decode path.
Optimization effort should go into compute — attention kernels
(sliding-window vs. full-attention dispatch, GQA, YaRN partial-rotary RoPE on
full-attention layers), quantized matmul and MoE gather-GEMM dispatch,
KV-cache handling, memory layout, and MLX
scheduling — not disk I/O.

Local work needs enough unified memory for the ~21.6 GB model plus KV state
and buffers; roughly 36 GiB is the practical local minimum.
Machines below 64 GiB automatically use a low-memory startup profile: the
MLX allocator cache is capped at 6 GiB, command buffers are shortened, and
free warmup buffers are cleared before the worker protocol starts. The
profile is pure memory management — compiled decode and every other ranked
code path stay enabled, so local runs execute the same code paths as the
ranked box all the way down to the documented 36 GiB local minimum. It
prints a stderr notice when it engages; set
`DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low|auto` to override the automatic
selection. A machine too small for the model plus the decode working set
fails loudly with an out-of-memory error rather than silently skipping
ranked code paths — if that happens, use a machine with more unified
memory or rely on the ranked run. The 128 GB ranked runner keeps the full profile.
The ranked box has more headroom than that, but memory-hungry strategies
tuned against a different machine still have to survive the paired
measurement on the M5, and a kernel or layout strategy that helps on one
Apple Silicon generation can move differently there — always rely on the
official benchmark for ranking.

## What You May Optimize

The submitted editable surface is defined by `editablePaths` in
`benchmark.json` — that list (currently 89 entries) is the source of truth.
For the ranked Qwen-MTP track it covers five groups, and the first two are the
2026-08-14 additions: `mtp-head.manifest.json` + `mtp-head/` (bring your own
MTP head, declared and digest-verified) and the draft schedule inside
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`. The remaining three are the
long-standing runtime/kernel groups:

```text
Sources/MLXFastModel/ Laguna runtime glue, custom kernels, decode path
Sources/MLXFastTransform/ offline weight transform
Vendor/mlx-swift-lm/ the Laguna model + MLXLMCommon plumbing + the DFlash draft/verify runtime
Vendor/mlx-swift/ the MLX Metal kernels Laguna dispatches
```

The vendored model surface is `Libraries/MLXLLM/Models/Laguna.swift` plus
the `MLXLMCommon` files it uses directly (KV caches, the `SwitchLayers.swift`
MoE expert gather-GEMM dispatch, the `AttentionUtils.swift` attention dispatch
and masking, RoPE utilities and application, compiled decode, evaluation
plumbing) and the DFlash
draft/verify runtime added for this track:
the eight `Libraries/MLXSpeculative/DFlash*.swift` runtime files listed in
`benchmark.json` (block dispatch, drafter, batched engine, KV rollback —
note `DFlashBenchmark.swift` matches that glob but is NOT editable) plus
`Libraries/MLXLLM/DFlashTarget.swift` and
`DFlashVerifyLinear.swift` (multi-row target verification). The exact file
list is in `benchmark.json`.

Know which of those vendored files the scored path actually executes. On the
DFlash track the scored target is the vendored `LagunaModel` (`Laguna.swift`)
reached through `LLMModelFactory`, driven by the DFlash block-decode runtime
under `Libraries/MLXSpeculative/` — not through
`Sources/MLXFastModel/LagunaRuntimeModel.swift`, which was the retired serial
track's scored forward pass. The DFlash draft model proposes a block, the
target verifies its rows through this reference forward, the longest correct
prefix is accepted, and rejected KV rows are rolled back. The vendored
`MLXLMCommon` helpers ARE executed on this path — `KVCache.swift` (standard
and rotating caches plus the rollback the verifier drives),
`RoPEUtils.swift` / `RoPEApplication.swift`, and `CompiledDecode.swift` with
its compilable cache variants — as are the vendored `Vendor/mlx-swift`
kernels described below.

The vendored kernel surface is the kernel families the Laguna forward pass
actually dispatches — SDPA (`scaled_dot_product_attention.metal`,
`sdpa_vector.h`, and `steel/attn/` with its `steel_attention*.cpp` twins:
Laguna's head dim is 128, so the fused steel attention kernels are
dispatchable at prefill), quantized matmul (the NVFP4 `fp_quantized*` kernels
and the shared `quantized*` infrastructure, including the `_nax`
variants), the MoE gather GEMM (`steel_gemm_gather*.cpp`), `steel/gemm/`,
`gemv`, `rope`, `rms_norm`, `softmax`, `sort`, `reduce`, `copy`, elementwise
(`unary`/`binary`/`ternary`), `arg_reduce`, and gather indexing
— in both forms the build uses: the AOT `.metal`/`.h` sources under
`Vendor/mlx-swift/.../backend/metal/kernels/` and their JIT twins under
`Vendor/mlx-swift/Source/Cmlx/mlx-generated/*.cpp`.

Know how a kernel edit becomes the running kernel. The vendored MLX Swift
package builds in JIT mode: families with an `mlx-generated/*.cpp` twin
(quantized incl. fp_quantized, steel/gemm incl. the gather GEMM, steel/attn,
gemv, softmax, sort, reduce, copy, elementwise, gather) are
compiled at runtime from the C++ source strings embedded in those files, so
for them the twin is the runtime-effective source — editing only the
`.h`/`.metal` form does not change what runs; edit the pair together.
Families without a twin (RoPE, RMSNorm, the SDPA vector kernel,
`arg_reduce`) are served ahead-of-time from `mlx.metallib`, built from the
vendored `.metal` sources by `tools/build-mlx-metallib.sh` (invoked by
`./setup.sh`; rerun either after editing an AOT `.metal`/`.h` file). `_nax`
names are the M5-generation kernel variants; the ranked M5 box selects
them, so tune the `_nax` twin as well as the plain one. Then test with
`./benchmark-dflash.sh --local-iterate`, which (through the `benchmark.sh` it
invokes internally) rebuilds both binaries for you whenever a build input is
newer than them. A bare `swift build -c release`
is **not** enough on its own: without `--scratch-path .build-worker` it
writes `.build/release`, while the scored binary is
`.build-worker/release/mlxfast-runtime-worker`.

All participant model and kernel code — `MLXFastModel` plus the vendored
forks — compiles into the sandboxed `mlxfast-runtime-worker` binary. The
trusted `mlxfast-swift` binary (timing, gates, scoring) links no MLX,
model, or kernel code and drives the worker over a JSON protocol; your
hot-path code runs only inside the worker.

Focus on:

- Reducing scored DFlash decode seconds per token (the seed prefill is
 charged inside that window, so prefill work still counts toward the score).
- Making block speculation itself cheaper: multi-row target verification,
 KV-cache handling and rollback, drafter dispatch, and scheduling around the
 block boundary.
- Optimizing the vendored Metal kernels on the prefill and decode paths.
- Optimizing kernels and hot-path MLX operations used by attention (both the
 sliding-window and full-attention layer types), the MoE MLP (routing,
 expert gather GEMM, shared expert), KV-cache
 handling, and weight materialization.
- Reducing model execution work on the hot path: MLX ops, synchronization,
 materialization, copies, and cache misses.
- Improving how RAM-resident weight bytes become MLXArrays (quantized
 linear construction, fewer copies, lazier Data-to-Metal conversions).
- Making the offline transform produce better runtime metadata or compact
 transformed artifacts.
- Improving prefill and single-token decode inside the Swift/MLX model path.

The target is Poolside Laguna XS 2.1, MoE, NVFP4 (untied embeddings, vocab
100352), text tower only (the empty `vision_config` is out of scope and never
loaded). The frozen reference checkpoint is about
21.6 GB across 5 safetensors shards, verified against the pinned manifest.
The transformed `weights/` tree holds the source's text-only `model.*` /
`lm_head.*` tensors plus a
runtime-authored `config.json`; it is an overlay/runtime artifact, not a
second physical copy of the model on APFS. Aim to keep generated transformed
weights under 22 GB (the default cap is 25 GiB).

## What Not To Change

Do not spend time modifying files outside `editablePaths` for a submission.
They are trusted harness/operator code and are not packaged by submit:

- `Sources/MLXFastCore/`, `Sources/MLXFastCLI/`,
 `Sources/MLXFastTrustedHarness/`, `Sources/MLXFastHarness/`, and
 `Sources/MLXFastRuntimeWorkerCLI/`
- `Package.swift` and `Package.resolved` — the dependency graph is frozen
 (the vendored forks are consumed as pinned local path dependencies)
- Everything in `Vendor/` not listed in `editablePaths`: other model
 families, shared model-factory/tokenizer plumbing, and kernels Laguna
 does not dispatch
- `.github/`, `fixtures/` (the DFlash track contract
 `laguna_xs_2_1_dflash_track.json` and the pinned reference/drafter `.sha256`
 manifests — trusted, outside `editablePaths`; the scoring step reads the
 contract from the trusted checkout precisely because `fixtures/` is not
 editable), scripts, tests, docs, and `benchmark.json`
- `weights/`, reference checkpoints, scores, golden files, local caches

Do not try to hardcode hidden prompts, hidden token IDs, GPQA answers, timing
shortcuts, protocol injection, network access, or filesystem exfiltration. The
official runner uses private artifacts, sandboxed runtime workers, artifact
validation, trusted workflow code, and static review gates. Hidden prompts and
goldens are not part of the public repo or submission payload.

Python is not part of the challenge runtime. Setup, transform, correctness, and
benchmark run through the Swift package. Account login, clone, and submission
use the Yukon CLI (`yukon`). Always use `yukon` for every participant CLI
operation.

## Correctness Gates

Correctness is a hard gate. The ranked M5 runner is the authority.

The official correctness stack reuses the serial gate stack plus the DFlash
token-fidelity gate:

- The public drift tripwire (one 64-step teacher-forced check against the
 checked-in public fixture, run before any hidden material enters the
 workspace).
- The hidden full-length teacher-forced base case (512-token prompts) plus
 hidden anchor, free-run, and GPQA behavior gates.
- The GPQA TTFT guardrail and the semantic GPQA judge (Anthropic).
- The DFlash token-fidelity gate: the trusted parent sequentially
 re-verifies every emitted token against a reference teacher-forced on the
 candidate's own emitted prefix, admits the reference argmax in the frame the
 candidate declared, prices the rejected tail, and rejects the run if a
 divergence falls outside a bounded near-tie budget. Exact-token equality
 against a purely sequential golden is unsatisfiable here — the target's own
 block-shaped forward diverges from its sequential forward at near-tie
 argmaxes with no drafter involved — so the gate binds emitted tokens to
 target compute actually performed rather than to sequential-token identity.
 `docs/dflash-track-correctness-contract.md` is the specification.
- Anti-lottery timed prompt: the timed target is sampled uniformly at random,
 once per ranked run, from a hidden pool of 8 distinct prompts spanning
 varied domains, so a failed ranked run cannot be cheaply retried into a
 lucky prompt.

Note the public checked-in fixtures are M5-generated; a near-tie argmax can
diverge on other Apple Silicon generations even for correct code.

## Timing And Score Measurement

The official benchmark measures DFlash decode seconds/token for the candidate
and the trusted serial K=1 target's decode seconds/token in the same session,
then publishes the decode-only paired speedup: the raw ratio of means
`raw = mean(serial K=1 seconds/token) / mean(dflash seconds/token)` over at
least three accepted thermally-gated pairs (target four) run in alternating
order, normalised by the sampled prompt's own pinned no-op reference
(`dflash_decode_speedup = raw / noop_reference`) so every prompt's no-op maps
to 1.0 and the score is independent of which hidden prompt was drawn. The
single normalised decode-speedup floor is `0.95`, hard. There is no prefill component — the seed prefill is charged inside the
decode window — and there is no separate acceptance band. The denominator is
the trusted serial K=1 target decode (the `dflash-probe` path) over the same
golden, same session, same thermal gate, same box, so the paired ratio
cancels host drift.

There is no two-sided acceptance band on this track; the retired serial
track's `[0.980, 1.053]` / `[0.952, 1.053]` window and its
`acceptance_band_failed` failure category do not apply. The ranked timing
gates are: the hard `0.95` floor on the normalised aggregate decode speedup
(the ratio-of-means aggregate divided by the sampled prompt's pinned no-op
reference, not applied per pair); the DFlash token-fidelity gate
(see "Correctness Gates"); and a stall guardrail that rejects a run whose
maximum block latency exceeds 4x its p50 block latency as
measurement-invalid, with one gated retry. A run that misses the floor,
fails token fidelity, or trips the stall guardrail publishes no score. The
floor is a uniform 0.952 margin against the sampled prompt's own pinned
no-op reference — a correct no-op normalises to exactly 1.0 on every prompt
(the raw no-ops cluster at 0.8726-0.8939) — so an unoptimized block-decode
build already clears it with ~5% margin.

The timed measurement runs last in the ranked job, after all correctness and
gate work and after every hidden byte is scrubbed from the bench workspace,
behind a fixed 40C GPU thermal gate with telemetry-validated acceptance
(throttled samples reject the measurement, with one gated retry). The timed
window is decode-only: a 512-token seed followed by a 512-token
parent-counted decode pass at block size `K=2`, with no separately scored
prefill phase. The timed target is not a single frozen prompt — it is sampled
uniformly at random, once per ranked run, from the 8-prompt hidden pool. See
`docs/dflash-track-correctness-contract.md` for the exact window and scoring
contract.

Diagnostic fields such as memory and read timings are recorded for audit and
future guardrails, but are not the primary score unless the benchmark contract
changes. There is no expert/weight-streaming bandwidth to report — every
routed expert is RAM-resident: `bandwidth_gb_per_token` is always `0` with
`bandwidth_source=ram_resident_model`. Do not optimize for that diagnostic
field as a standalone target; optimize changes that reduce the measured
DFlash decode time.

## Local Workflow

Before optimizing, sync to the latest challenge tip and record a same-machine
local baseline. Do not compare your changes against a stale branch or an old
local run:

```bash
git fetch origin main
git switch main
git pull --ff-only
./setup.sh && ./setup-dflash.sh
./benchmark-dflash.sh --local-iterate
cp score.json score.baseline.json
```

Create your working branch from that synced commit, or rebase/merge your
existing branch onto `origin/main` before trusting local timings. Every
`./benchmark-dflash.sh --local-iterate` result should be interpreted as
performance on top of the latest synced base commit measured on the same local machine,
with the same toolchain, model cache, power state, and thermal conditions. If
the base commit changes, rerun the local baseline before deciding whether an
optimization is faster.

Start with:

```bash
./setup.sh && ./setup-dflash.sh
```

`./setup.sh` checks the local Swift/Xcode toolchain, builds the Swift harness
and MLX Metal library, then downloads and verifies the pinned Laguna XS 2.1
reference checkpoint (use `MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1` while the
checked-in weight manifests are still entry-less placeholders, or when the
checkpoint is provisioned externally). `./setup-dflash.sh` then provisions the
organizer-pinned DFlash draft model this track drives.

Common commands:

```bash
swift test --force-resolved-versions
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
swift build -c release --force-resolved-versions
tools/build-mlx-metallib.sh
./benchmark-dflash.sh --local-iterate
./benchmark-dflash.sh --local-submit
```

Pass `--force-resolved-versions` on every direct `swift build` / `swift
test`: the dependency graph is frozen, and a bare invocation can silently
rewrite `Package.resolved` (SwiftPM re-resolves the ranged transitive pins
on toolchain drift), after which `./setup.sh` and `./benchmark.sh` refuse
to run until you restore it with `git checkout -- Package.resolved`. The
flag makes SwiftPM fail closed instead.

`./benchmark-dflash.sh --local-iterate` is the fast local edit-loop signal.
Use it to compare the current working tree against the latest-tip baseline you
recorded above, not against a result from an older branch.
`./benchmark-dflash.sh --local-submit` is the recommended manual pre-submit
check (`yukon submit` does not run it for you) and is intended to be longer
and closer to the official path; like `--local-iterate` it publishes only a
local estimated score (never the official ranked score). The DFlash local
wrapper drives both the serial K=1 control and the block-decode pass; it
invokes `./benchmark.sh` internally to build binaries and cache the
transformed `weights/` (that shared `--official` path is workflow-internal,
not the participant entrypoint). The hidden M5 goldens remain the fidelity
authority.

## Notes For Autonomous Agents

Operational contract for coding agents iterating in this repo. These
behaviors are expected, not bugs:

- **Cool-down gate.** `./benchmark-dflash.sh
 --local-iterate` and `--local-submit` wait for the GPU to cool below 40C
 before starting the timed run (read via `macmon`), printing a progress
 line roughly every 10 seconds while waiting. A benchmark invocation that
 pauses on "waiting for GPU to cool down" is working, not hung — do not
 kill it or treat the wait as a failure. If the GPU stays hot and is not
 trending down, the gate aborts with a non-zero exit after about 3
 minutes; that abort means "something else is loading the GPU — free it
 up and retry," not "the code change is wrong." (A hard 900-second
 ceiling applies even while the GPU is still slowly cooling.) If `macmon`
 is not installed the gate warns and skips; `./setup.sh` installs it (or
 `brew install macmon`). The gate mirrors the ranked runner's fixed
 40C / 1500 MHz / 900 s thermal contract (the serial-side frequency floor
 was set to 1500 by operator decision 2026-07-31 — at 1600 no ranked run
 could complete at the 512-token window), which is operator-owned and
 non-overridable.
- **Optional fan boost for a stalled local cool-down.** If the local gate
 sits hot for ~60 seconds with no cooling progress, it offers — once per
 run, interactive terminal only — to force the Mac's fans to a hard-coded
 70% of their maximum speed via `tools/fan-control.sh`. Fan targets are
 SMC keys that macOS only lets root write, so accepting the offer
 triggers sudo's own password prompt; the scripts never read, store, or
 log the password, and the cached credential is dropped (`sudo -k`)
 right after the write. `./benchmark.sh --fan-speed-normal` removes the
 70% override and returns the fans to macOS's automatic curve (no pinned
 RPM). Manual control uses the same helper:
 `tools/fan-control.sh boost|normal|status` (needs an `smc` CLI, e.g.
 from smcFanControl; fanless Macs are refused cleanly).
- **Measurement discipline.** Trust timing numbers only from a cool,
 quiescent machine. Back-to-back runs heat the GPU and throttle it; a
 2-3 minute cool-down between local runs is normal. The wrapper
 enforces this automatically. Do not fight the gate to iterate faster:
 `MLXFAST_LOCAL_COOL_GATE=0` is for debugging only and produces
 hot-start timings that are not comparable to gated ones. The ranked
 score is a paired speedup versus the on-box pinned baseline measured
 in the same session. Treat local scores as directional.
- **A local gate failure on non-M5 hardware may not be your bug.** The
 public goldens are M5-generated greedy continuations of the
 mlx-swift-lm reference; near-tie argmaxes can diverge on other Apple
 Silicon generations even for correct code. Before treating a local
 public-gate failure as a regression, check whether unmodified `main`
 fails at the same token position on your machine; the ranked M5 runner
 is the source of truth. If it does, rerun with
 `MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1` so the local mode still publishes
 its timing estimate instead of a `score: null` the Yukon CLI rejects.
 The override is local-only and does not hide the divergence: the score
 keeps `passed_correctness: false`, records the diverging tokens, and
 explains itself in `metrics.error`. Never use it to paper over a real
 regression -- if unmodified `main` passes on your machine, the mismatch
 is yours.
- **Know the runnable surface.** Only the `benchmark.json` `editablePaths`
 entries ship in a submission: `Sources/MLXFastModel/`,
 `Sources/MLXFastTransform/`, the vendored Laguna model and `MLXLMCommon`
 files, and the listed vendored kernel sources (both AOT `.metal`/`.h`
 and JIT `mlx-generated/*.cpp` forms); changes anywhere else will not
 upload even if they help locally. Official ranking requires hidden
 organizer goldens and is not runnable locally — use
 `./benchmark-dflash.sh --local-iterate` for the edit loop and
 `--local-submit` as the recommended pre-submit check.
- **One local run at a time; the memory guard is protecting your RAM.** The
 ~21.6 GB RAM-resident text tower means two simultaneous model residencies
 (an overlapping second local run, or a new run started while an orphaned
 model-holding worker from an aborted run lingers) can out-of-memory a
 local machine. A single worker is separately protected by the automatic
 low-memory startup profile described above. Local modes take a per-user
 run lock and refuse to start while a model-holding mlxfast process is
 still alive, printing the offending pid/rss/command list. Read that list
 before reacting: a
 ppid of 1 is usually an orphan from an aborted run (verify, then
 `kill <pid>`); a live ppid is usually a legitimately concurrent run
 (wait for it). The guard warns and aborts -- it never kills anything
 itself. Aborted local runs now reap their own worker on INT/TERM/EXIT
 and the worker exits if its parent dies, so the guard should fire
 rarely. Know its scope: the lock lives in `benchmark-dflash.sh` (and the
 `benchmark.sh` it invokes internally), so direct `mlxfast-swift` model
 commands (`correctness`, `correctness-trace`, `generate-golden`,
 `generate-gpqa-answers`, and the DFlash-track `dflash-benchmark`,
 `dflash-probe`, and `dflash-reference` — all model-holding: the ~21.6 GB
 target plus the drafter) take no lock
 and do not check for other runs -- run one model-holding command at a
 time, never concurrently with a local benchmark or with each other.
 (`swift test` never loads the real model and is safe alongside.)
 `MLXFAST_LOCAL_RUN_GUARD=0` disables the guard for harness debugging
 only -- never set it to resolve contention; wait for the other run
 instead. The ranked --official path is unaffected.
- **One ranked machine, one queue.** Ranked runs execute serially on the
 single M5 runner: one job at a time by construction, and duplicate
 dispatches queue behind the in-flight run instead of cancelling it.
 Expect queueing delays behind other submissions, and do not dispatch
 multiple ranked runs in parallel expecting concurrent results.

## Swift Tooling

Use the Swift toolchain that `./setup.sh` validates. `sourcekit-lsp` is the
standard Swift language server and is usually installed with Xcode or the Swift
toolchain. Point your editor at the repository root so SourceKit-LSP can read
`Package.swift` and resolve the SwiftPM targets.

Useful local tooling commands:

```bash
swift build -c release --force-resolved-versions
swift test --force-resolved-versions
sourcekit-lsp
xcode-select -p
xcrun --find sourcekit-lsp
```

Avoid bare `swift package resolve` / `swift package update`: they can
rewrite the frozen `Package.resolved` (there is no fail-closed flag for
`resolve`), and `./setup.sh` plus the flagged builds already resolve
fail-closed. If `Package.resolved` ever shows as modified, restore it with
`git checkout -- Package.resolved`.

For editor agents, prefer SourceKit-LSP symbol navigation and diagnostics over
string-only edits when changing Swift model code. Use `swift test` for cheap
contract checks, and use `MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test` when a
change touches MLX runtime behavior and the machine can run those tests.

## Submission Workflow

Use the Yukon CLI for all account and submission commands:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
yukon login <api-key> --api <url>
yukon clone <benchmark-id-or-name>
yukon submit --model "<exact model name>" --note-file submission-note.md
yukon submissions
```

Submit packages only `editablePaths`. It rejects generated artifacts, symlinks,
local scores, reference checkpoints, and source changes outside the editable
surface. `yukon submit` uploads the editable-path archive directly for
official validation; it does not run a local benchmark first, and no local run
blocks the upload. Run `./benchmark-dflash.sh --local-submit` yourself before
submitting — the official M5 run is the gate that ranks the submission.

## Practical Optimization Ideas

Good submissions are likely to improve one or more of:

- Kernel-level optimization inside the vendored Metal sources. Prioritize
 kernels reached by the timed decode window (the seed prefill is charged
 inside it). For the JIT families, edit the `mlx-generated/*.cpp` twin — that
 string is what compiles at runtime.
- DFlash block-speculation path: cheaper multi-row target verification, KV
 rollback of rejected rows, drafter dispatch, and scheduling across the block
 boundary — the work that most directly lifts the decode speedup.
- Attention kernel dispatch: sliding-window vs. full-attention masking, GQA
 head-group broadcasting (8 KV heads at head_dim 128), and the
 full-attention layers' YaRN partial-rotary (0.5) RoPE.
- NVFP4 matmul and MoE dispatch for the group-16 routed/shared expert
 projections: fewer dequantize/copy steps, better expert gather-GEMM
 batching across routed experts, the BF16 per-layer router gates, the
 shared expert, and reuse of derived weight views.
- KV cache handling: the sliding-window cache only ever needs the last 512
 positions; a tighter ring-buffer implementation can reduce both memory and
 copy overhead relative to the straightforward baseline.
- Weight loading and reuse: eager preparation at init, warm kernels
 before the first scored forward, and avoiding redundant Data-to-Metal
 conversions.
- MLX operation scheduling and synchronization.
- Transform metadata that lets runtime skip work safely.

Be careful with optimizations that only help a single public prompt or a single
machine. The hidden correctness and benchmark prompts are different from the
public local fixtures, and official scoring happens on the single self-hosted
M5 runner. Kernel edits are bound by the same correctness gates as model
edits: keep them prompt-independent and model-general for Laguna, and be
conservative with numeric reassociation — a changed accumulation order can
flip near-tie greedy argmaxes on the M5 and fail the exact-token gates.

## Avoid These Wrong Strategies

Do not assume the benchmark machine has the same memory budget as your local
Mac. The ranked box is one Apple M5 Max with 128 GB of unified memory; the
~21.6 GB text tower is comfortably RAM-resident there, but do not treat that
headroom as an invitation for memory-hungry strategies tuned on a different
machine — KV cache, buffers, and caches still compete, and what is fast on
your Apple Silicon generation can move differently on the M5. Although Laguna
is an MoE checkpoint, every expert is RAM-resident: there is no
"streaming fallback" regime here to mistune against.

Do not specialize for the public correctness prompt. Optimizations should be
prompt-independent and model-general for Laguna. Hidden correctness, GPQA,
and benchmark prompts are different from the public fixtures.

Do not treat local-only environment overrides as proof of a valid improvement.
Examples include disabling the sandbox, skipping transform without verifying
the produced `weights/`, pointing at a user-specific reference path, or tuning
with settings that are not part of the official benchmark contract. Those can
be useful for debugging one machine, but they do not establish a rankable
optimization.

Do not draw conclusions from a tiny local iterate run alone. Short local modes
are smoke tests for speed and correctness direction. They are not substitutes
for the official hidden benchmark, and they are especially weak for testing
sequence-length-dependent optimizations (e.g. attention kernel changes) since
they may not exercise the same sequence lengths or memory pressure as the
ranked run.

## Before Submitting

Run at least:

```bash
swift test --force-resolved-versions
./setup.sh && ./setup-dflash.sh
./benchmark-dflash.sh --local-submit
```

If local correctness fails, check the non-M5 near-tie caveat above before
assuming a regression; if performance improves but correctness is fragile,
prefer a more conservative optimization. The official benchmark will not rank
a submission that fails the hidden gates.

Do not add caches or memos keyed on a request's input tokens whose only
possible hit is the benchmark harness repeating an identical computation — for
example, memoizing a whole-prompt forward's logits or KV state so a repeated
identical forward can skip the work. Bit-identical output does not make this
legitimate. The benchmark measures single-pass inference: optimizations must
save work that recurs in single-pass production inference (one prefill, then
decode, per prompt), not work that only exists in the measurement protocol.
The harness never legitimately issues the same whole-prompt forward twice to
one worker process; any such repetition is a harness bug, never a contract to
rely on. Input-independent caching (weights, dequantized tensors, RoPE/mask
tables keyed on shapes and offsets) and within-request KV reuse remain fine.
Submissions in this category fail the static review as bypass behavior.

### Retired Serial Non-Speculative Track Rules (historical)

These rules governed the retired Laguna XS 2.1 serial track
`laguna-xs-2.1-serial-v2`. They no longer govern any ranked track — the
DFlash rules in the next section govern `laguna-xs-2.1-dflash-v1` — and
`benchmark.json` no longer registers the serial track. They are kept for provenance and because the
DFlash track reuses the same anti-cheat framing for input-derived drafts.
Under the serial track each model invocation could compute logits and KV rows
only for tokens supplied in that invocation, and had to advance logical and
physical KV position by exactly the supplied input length. The rule, stated as
the static-review gate still cites it: a one-token decode request advances
exactly one position and leaves no pending future token, logits, or KV state
for a later request. (DFlash's sanctioned block decode is the deliberate
exception — it verifies K organizer-drafted rows per forward and rolls back the
rejected tail — but every UNsanctioned way of manufacturing future tokens below
stays excluded on both tracks.)

Prompt-lookup decoding; n-gram, suffix, or token-history drafting;
same-target lookahead; and any other selection or evaluation of an unsupplied
future token are excluded. So are two-, three-, or more-row target-model paths
used to verify a draft from a one-token request, cross-request future-logit/KV
buffers, deferred cache rows, and commit, rollback, recommit, or discard
markers for those rows. Generic, bit-exact, or production-useful
implementations are still excluded under this track. Pre-hello or
initialization warmup of an excluded speculative pipeline is also excluded.

The model quantization is frozen to an accepted envelope, and every
submission may use all of it. The envelope is exactly two things: (1) the
reference NVFP4 weights as shipped — group size 16, 4 bits, mode `nvfp4`; and
(2) one established re-quantization, in which the attention Q/K/V, output, and
per-head gate (`g_proj`) projection weights may be re-represented as group-32
affine INT8 derived at
init from the loaded NVFP4 weights. That attention re-quant is accepted and
available to all submissions. Note the attention per-head gate `g_proj` is a
distinct parameter from the MoE router gate: `g_proj` is an attention
projection and is inside the envelope, whereas the MoE router gate is not and
stays as shipped. Nothing beyond this envelope is permitted: do
not re-quantize any other weight (the MoE routed or shared experts, MoE router
gate, embeddings, `lm_head`, and every other parameter must remain NVFP4
group-16 4-bit); do not use any bit width other than 4 or 8, any group size
other than 16 or 32, or any mode other than `nvfp4` or affine; and do not make
the attention re-quant (including `g_proj`) lossier than group-32 affine INT8
(a larger group or fewer bits). This holds even when a further re-quantization passes the
correctness gates, because going beyond the envelope substitutes a
further-degraded numerical representation of the model rather than optimizing
the accepted one. Pure memory relayout or co-tiling that preserves quantized
values, and input-independent dequantized caches, remain allowed.

Ordinary within-request KV reuse, current-token-only decode, and
input-independent weight, dequantization, kernel, mask, or RoPE caches were
allowed. Multi-row kernels were allowed when every row was backed by a token
supplied in that same invocation, such as prefill. Under the serial track,
organizer-provided speculative decoding required a separate explicit track
with a trusted variable-length block protocol, correctness contract, and
score. That track now exists and is enabled: it is the DFlash track described
in the next section, it is the default ranked path, and target-verified block
speculation is the ranked mechanism there — not a prohibited one.

### DFlash Speculative-Decode Track Rules (default, ranked)

`benchmark.json` registers `laguna-xs-2.1-dflash-v1` as the default and only
ranked track. Go-live is done: the contract fixture
(`fixtures/laguna_xs_2_1_dflash_track.json`) sets
`official_scoring_enabled: true`, `reference_baseline.publication_allowed:
true`, and `token_fidelity_gate_status: implemented`, and the workflow
(`.github/workflows/dflash-benchmark.yml`) ranks submissions on the
`m5-laguna-dflash` runner. This is the track to target. Its rules:

- **Speculation is the point.** A separate organizer-provisioned ~924 MB DFlash
  draft model (an EAGLE-style speculator with its own weights, trained block
  size 16, conditioned on target hidden states and borrowing the target
  embedding and lm_head) proposes tokens, the target verifies the block in one
  forward, the longest correct draft prefix is accepted, and rejected KV rows
  are rolled back. The retired serial track's speculative-decode prohibition is
  inverted here: block speculation is the ranked mechanism.
- **Participants never supply weights.** Both the target (the same pinned NVFP4
  group-16 reference checkpoint the serial track measures) and the drafter are
  organizer-provisioned and hash-pinned. Substituting, re-deriving, or
  re-quantizing the drafter is a fail (it stays BF16). The retired serial
  rules' frozen TARGET-quantization envelope carries over unchanged to this
  track's target handling: the same accepted representations, nothing beyond
  that envelope — the DFlash track changed the decode protocol, not what may
  be done to the target's weights.
- **Drafts must come from the pinned drafter, not from the input.** Bypassing the
  drafter is a distinct violation from substituting its weights, and it is
  excluded just as firmly. Every proposed token in a block must be produced by a
  forward pass of the pinned draft model on that round's bonus token and target
  hidden context. Prompt-lookup drafting, n-gram, suffix or token-history
  drafting, copying from the seed or from previously emitted tokens, and any
  other input-derived proposal source are excluded — the same techniques the
  serial track excludes, for the same reason, and the exclusion holds even where
  the implementation is generic, production-useful, or bit-exact. Hybrids that
  fall back to an input-derived proposal when the drafter is slow or its
  confidence is low are excluded too. You may make the drafter's *dispatch*
  cheaper (fusion, layout, scheduling); you may not replace what it computes.
  This is enforced the way the serial track enforces its own speculation ban —
  by rule and static review — see the DFlash correctness contract's L5 section
  for why a runtime numerical check is a weaker instrument than it looks.
- **What is measured is the reference forward.** The DFlash target is the
  vendored `LagunaModel` reached through `LLMModelFactory`, not the serial
  track's scored `Sources/MLXFastModel/LagunaRuntimeModel.swift`. DFlash
  speedups are therefore relative to the reference implementation and are
  neither additive with serial-track optimizations nor comparable to serial
  scores as absolute tokens/second.
- **Scoring** is decode-only paired speedup: mean serial K=1 seconds/token over
  mean DFlash seconds/token, ratio-of-means across at least three accepted
  thermally-gated pairs (target four) in alternating order, with a hard floor
  of `0.95` on the per-prompt-normalised aggregate ratio-of-means (not per
  pair; the raw ratio divided by the sampled prompt's pinned no-op reference)
  and a stall
  guardrail (a run whose maximum block latency exceeds 4x its p50 is rejected
  as measurement-invalid, one gated retry). Block size is `K=2` and the ranked
  decode window is 512 parent-counted tokens; there is no prefill component.
- **Which side runs whose build — read this before planning any work.** The
  denominator is a `dflash-probe` (serial K=1) run from an APFS copy-on-write
  clone of the **pinned baseline tree**; the numerator is `dflash-benchmark`
  (block decode) from **your** workspace. This mirrors the serial track: you are
  measured against a fixed on-box baseline, not against yourself. So general
  forward improvements — a better quantized matmul, a faster attention kernel —
  DO move your score, because they speed the numerator while the pinned
  denominator stays put. Serial-track techniques transfer.
  (An earlier revision of this file claimed both sides run the submitter's build
  and that generic wins therefore cancel. That was wrong; the box wrapper's own
  contract header is the authority.)
- **Block decode is not free, but the floor sits below a no-op.** A verify row
  costs only about 10% less than a standalone serial step, so raw block
  speculation reaches serial parity (raw ratio 1.0) only when draft acceptance
  is above ~92%. On the moderate-difficulty prose in the timed pool acceptance
  is ~75%, and an unoptimized (no-op) block-decode build measured at the ranked
  configuration (512-token seed, 512 parent-counted decode tokens, `K=2`) lands
  at a RAW 0.8726-0.8939 across the eight pool prompts — below serial parity.
  The score you are ranked on is that raw ratio NORMALISED by the sampled
  prompt's own pinned no-op reference, so a correct no-op normalises to 1.0 on
  every prompt and the `0.95` floor (within 5% of the prompt's own no-op)
  clears with margin; the job is to push the normalised ratio up from there.
  (The retired raw floor was `0.83` = worst no-op Russell 0.8726 x 0.952, which
  was correct only for the hardest prompt.) The highest-leverage work makes
  speculation itself cheaper — multi-row verify batching, KV handling and
  rollback cost, drafter dispatch, scheduling around the block boundary. General
  forward speedups (a better quantized matmul, a faster attention kernel) count
  too, since they speed your numerator while the pinned serial denominator stays
  put.
- **Do not tune block size on a self-generated prompt.** A greedy
  self-continuation of the model is degenerate — measured at 122 distinct tokens
  in 512, with no row whose top-2 logits are within 1.8 of each other — so a
  drafter predicts it almost perfectly and every K looks good. Earlier sweeps on
  such material reported K=3 at 1.09x and, on a 51-token prompt, K=8 at 1.86x.
  Neither survives on varied text. Measure on real prose, at the ranked window,
  with matched token counts on both sides.
- **Correctness is work-honesty, not token identity.** Exact-token equality
  against a sequential golden is *unsatisfiable* on this model: the target's own
  block-shaped forward diverges from its sequential forward at near-tie
  argmaxes, with no drafter involved (measured: 14/14 divergences target-only,
  max sequential-logit gap 0.625, under 1% of positions). The replacement
  contract admits the reference argmax in the frame the candidate declared, and
  separately binds emitted tokens to target compute actually performed for each
  row — per-row top-2 logit values, row accounting, KV vacancy checks, and a
  reference replay run by the pinned baseline build on organizer weights after
  the timed window. `docs/dflash-track-correctness-contract.md` is the
  specification; read it before assuming any behaviour of this track.

The editable surface is defined by `benchmark.json` `editablePaths` and
centres on the DFlash runtime under
`Vendor/mlx-swift-lm/Libraries/MLXSpeculative/` plus `DFlashTarget.swift` and
`DFlashVerifyLinear.swift`, on top of the same vendored MLX kernel families
the forward pass dispatches. Local scripts are `setup-dflash.sh` and
`benchmark-dflash.sh`; the retired Gemma-era MTP surface stays retired under
its own names and must not be revived.

## Qwen 3.8 MTP track (ranked track of this repository)

**Status: NOT LIVE — 3.6 → 3.8 cutover in progress.** `main` of this
repository IS the `qwen3.8-27b-mtp-v1` track — `benchmark.json` is its
manifest, `fixtures/qwen3_8_27b_mtp_track.json` its contract, and
`.github/workflows/qwen-mtp-ranked-benchmark.yml` its ranked pipeline — but
the track is INERT AND FAIL-CLOSED pending the Qwen 3.8 bring-up: the
calibration interlock is `"0"`, both contract enablement flags are `false`,
and every hidden-artifact pin reads `QWEN38-PENDING-RELEASE`.

**Backbone identity 2026-08-14.** The reference checkpoint is **our own** MLX
4-bit affine / group-64 conversion, produced under a pinned mlx 0.32.0
toolchain, of the official bf16 base `Qwen/Qwen3.8-27B` @
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (1,847 tensors, all
`language_model.`, no `mtp.`), publishing as `EigenLabs/Qwen3.8-27B-4bit`. It
replaces a third-party personal-account conversion adopted earlier the same day
and **terminated** by the operator's validation kill-switch: the deterministic
reconversion cross-check that adoption had reserved was performed and found 994
of 1,847 tensors numerically different from our own conversion. The geometry
reading is unaffected — it is a property of the bf16 base and the terminating
cross-check was numerical, not structural — so the geometry hedges stay
discharged. Nothing is published yet, under three markers: the backbone's
revision (`QWEN38-PENDING-BACKBONE-PUBLISH`), its per-file byte manifest, whose
fixture is a header-only stub (`QWEN38-PENDING-RELEASE`), and the MTP head —
`EigenLabs/Qwen3.8-27B-MTP-bf16`, 15 bf16 tensors extracted from the same bf16
base — whose revision and byte manifest read (`QWEN38-PENDING-HEAD-UPLOAD`).

Anything still carrying a 3.6 measurement is marked
`QWEN38-VERIFY-AT-RELEASE`. `docs/CUTOVER.md` is the operator procedure;
`docs/qwen-mtp-go-live-runbook.md` records the **3.6** go-live and is history. The
`laguna-xs-2.1-dflash-v1` pipeline described above is **not** ranked here; it
stays ranked in `Layr-Labs/mlxfast-challenge-dev`. The former staging branch
`qwen36-mtp-track` of that repository is superseded by this repository's
`main`. See `QWEN36-MTP-CHALLENGE-PLAN.md` (operator working copy, outside
this repo) for the phase plan.

### The working contract (as of 2026-08-14)

**What you may change.** The union of two surfaces, and the second half is new:

- *The MLX runner and kernel path* — the Qwen 3.6 text tower under
  `Sources/MLXFastModel/`, the offline transform, and the vendored MLX/Metal
  kernel families the forward pass dispatches. This is the primary axis and it
  is unchanged.
- *The whole speculative apparatus* — `Qwen36MTPBlockSession` and its siblings
  (drafting, verify-block assembly, accept walk, KV snapshot/rollback/repair);
  the **per-round draft schedule**, via the editable
  `Qwen36MTPBlockSession.draftPolicy`, which ships as a constant 2 and may
  return anything from 0 to the trusted maximum of 8, adaptively, per round —
  including 0, which is an ordinary adaptive skip; and the **MTP head weights**,
  declared in `mtp-head.manifest.json` (`source` / `sha256` / `bytes`, 2 GiB
  cap) and fetched + digest-verified by the runner before the sandbox opens.
  Absent declaration = the organizer-pinned head, which is the normal case;
  a *present but broken* declaration is a refusal, never a silent fall back.

**Why the head can be yours.** A head only *proposes* tokens. The
organizer-pinned target decides every emitted token, and the trusted parent
re-checks the whole stream against a hidden serial trajectory after the clock
stops. A substituted head moves the accept rate — which is the game — and
cannot move the output. The same argument is why the schedule is yours.

**What you may not change** is anything that verifies, measures or ledgers: the
target weights and transform contract, the tokenizer, goldens, the trusted
driver's audit replay and row accounting, gates, workflow, `.github/scripts`,
timing and telemetry. `docs/qwen-mtp-editable-surface.md` is the definitive
table.

**Scoring — serial is 1.0.**

```text
per prompt p:  raw_p = mean(serial depth-0 s/token) / mean(candidate s/token)
published:     score = median(raw_1 .. raw_8)      # all 8 hidden pool prompts
```

Both means come from the same thermally-gated session for that prompt, so the
serial leg is the normaliser and it is measured, not pinned. There is no
normalisation step. Floor `0.90` on the published median — "do not regress
serial by more than 10%" — and ceiling `3.0` (a plausibility bound). The floor
is a regression bound, not a quality bar: publishing below 1.0 is allowed and
the board will show it. The median rule is the mean of the two central order
statistics — 8 is even, so it matters.

**An unmodified tree scores ~0.935, not 1.0.** The shipped depth-2
configuration is a measured ~6.5% regression against true serial decode at the
512-token window; the score says so instead of defining it away. Two direct
consequences for how you should think about the work:

- *You are not starting at parity.* Roughly 6.5% of the gap to 1.0 is the cost
  of the shipped speculative machinery itself — verify-row cost, rollback, the
  head forward — and closing it is legitimate, on-target work.
- *Turning drafting off is legal and scores exactly 1.0.* A candidate whose
  policy returns 0 every round IS the serial control. That is an honest
  submission and a real (if unambitious) baseline; it is also why non-drafting
  rounds could be legalised without opening an exploit. Note the ordering this
  creates: 1.0 (draft nothing) beats 0.935 (the stock schedule), so the very
  first thing worth checking is whether your change is actually paying for the
  speculation it does.

**Token fidelity is unchanged and absolute.** Every emitted token must equal the
serial trajectory, the row ledger must close over the drafts *actually*
proposed, and the trusted parent reference-checks every declared row after the
window. The candidate declares nothing about its depth: effective per-round
counts are read out of the parent's own journal and sealed
(`effective_mean_draft_len`, `non_drafting_round_count`).

The section below describes the phase-1 target-runtime port and is retained
for provenance; it predates go-live and is not the ranked contract.

Pinned Qwen target identity, mirrored by `Sources/MLXFastCore/Constants.swift`
(`referenceModelRepository` / `referenceModelRevision`), by `setup.sh`
(`REFERENCE_MODEL_REPO` / `REFERENCE_REVISION` / `REFERENCE_MANIFEST_PATH`),
and by the checked-in manifest `fixtures/reference_qwen3_8_27b_4bit.sha256`:

```text
repository  EigenLabs/Qwen3.8-27B-4bit
revision    QWEN38-PENDING-BACKBONE-PUBLISH
manifest    fixtures/reference_qwen3_8_27b_4bit.sha256   (header-only stub)
upstream    Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

Pinned 2026-08-14, our own conversion. The manifest fixture pins only its
repository header line; its revision header reads the marker and its per-file
body is generated from the published snapshot in a follow-up commit, alongside
the workflow's `MLXFAST_QWEN_MTP_TARGET_MANIFEST_RECORDS` / `_BYTES`
(`QWEN38-PENDING-RELEASE`).

The MTP head is a separately pinned artifact and is NOT part of the backbone
port: the transform never selects `mtp.*` tensors, and the backbone contains
none (1,847 tensors, all `language_model.`). Its 3.8 repository is
`EigenLabs/Qwen3.8-27B-MTP-bf16` — 15 bf16 tensors, 849,398,784 tensor bytes in
a single 849,400,347-byte `model.safetensors`, extracted from the bf16 base
above — but it is not uploaded, so its revision and byte manifest read
`QWEN38-PENDING-HEAD-UPLOAD`. (The head's tensor count was `31` until
2026-08-14: that was the 3.6 head's 4-bit weight/scales/biases triples, not a
different architecture.) The last *pinned* head was
`mlx-community/Qwen3.6-27B-MTP-4bit` @ `83795d546e9d328160e593fb0bf10b2bf2fe637e`;
since 2026-08-14 the pinned head is a *default*, not the only option — it is
what the baseline leg always runs and what an absent `mtp-head.manifest.json`
selects for the candidate leg.

What phase 1 landed on this branch:

- `Sources/MLXFastModel/Qwen35*.swift` — the Qwen 3.6 text tower. The
  artifact is named Qwen3.6; its immutable internal architecture name is
  `qwen3_5_text`, which is why the sources carry a `Qwen35` prefix. 64
  layers on a 4-layer repeat (every 4th layer full attention, the other
  three gated-delta linear attention), vocab 248320, hidden 5120, untied
  `lm_head`, affine 4-bit quantization.
- `Sources/MLXFastTransform/Transform.swift` — a third
  `TransformModelFamily` case, `.qwen35`, selected by the `qwen3_5`
  model-type prefix inside the source `text_config`. The `laguna` and
  `gemma4` families are untouched.
- `Tests/MLXFastTests/Model/Qwen35ReferenceParityTests.swift` — the re-aimed
  streaming-schedule parity gate. It is opt-in and loads the real
  checkpoint, so it is skipped unless
  `MLXFAST_RUN_QWEN_REFERENCE_PARITY=1` and
  `MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH=<transformed-weights>` are both set.

The Laguna/DFlash surface (`Laguna*.swift`, `benchmark-dflash.sh`,
`setup-dflash.sh`, the DFlash track fixture and contract) is deliberately
left in place and unmodified; the two targets coexist on this branch until
the track identity is chosen. The retired Gemma-era MTP names stay retired —
a Qwen track id must not substring-collide with them.
