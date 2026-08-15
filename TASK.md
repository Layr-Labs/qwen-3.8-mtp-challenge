# mlxfast — Qwen 3.8 27B native MTP challenge

Make Qwen 3.8 27B decode faster on Apple Silicon with its own native
multi-token-prediction (MTP) head, without changing a single emitted token.

The target drafts a block with the MTP head, verifies that block against
itself in one forward pass, and advances decode by the longest correct
prefix. Every emitted token must equal the token serial decode would have
produced. The ranked track is `qwen3.8-27b-mtp-v1`. It is live and scoring.

## The pinned artifacts

Both checkpoints are public. They download anonymously, and no Hugging Face
token is required.

```text
backbone  EigenLabs/Qwen3.8-27B-4bit      @ eda45ab47f465d08d6558f0353a2346e2eb9d5b3
          pinned by fixtures/reference_qwen3_8_27b_4bit.sha256   (10 records)
head      EigenLabs/Qwen3.8-27B-MTP-bf16  @ 26a328e070875b0314d652a039b6b59902690f03
          pinned by fixtures/qwen3_8_27b_mtp_head.sha256         (4 records)
upstream  Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

The backbone is our own MLX 4-bit affine / group-64 conversion of that
upstream bf16 base, produced under a pinned mlx 0.32.0 toolchain. The head
ships separately: the backbone carries 1,847 tensors, all under
`language_model.`, and none under `mtp.`.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 89 entries in
five groups:

| Path | What it controls |
|---|---|
| `mtp-head.manifest.json`, `mtp-head/` | The MTP head weights. Declare a head (`source`, `sha256`, `bytes`) and the runner fetches and digest-verifies it before the sandbox opens. No declaration means the organizer-pinned head, which is the normal case. |
| `Sources/MLXFastModel/Qwen36MTP*.swift` | The drafting code and the draft schedule. `Qwen36MTPBlockSession.draftPolicy` ships as a constant 2. |
| `Sources/MLXFastModel/` | The Qwen 3.8 runtime: weight loading, attention, MoE MLP, KV caches, prefill and decode execution. |
| `Sources/MLXFastTransform/` | The offline transform of the reference checkpoint into `weights/`. |
| `Vendor/mlx-swift-lm/`, `Vendor/mlx-swift/` (listed files) | The vendored model plumbing and the MLX Metal kernels the forward pass dispatches. |

Draft depth is yours. Your policy may return any count from 0 to the trusted
maximum of 8, per round, adaptively. A round that drafts nothing is legal.

A head only *proposes* tokens. The organizer-pinned target decides every
emitted token, and the trusted parent re-checks the whole stream against a
hidden serial trajectory after the clock stops. That is why the head and the
schedule can be yours.

You may not change anything that verifies, measures or ledgers: the target
weights and the transform contract, the tokenizer, the goldens, the trusted
driver's audit replay and row accounting, the gates, the workflow,
`.github/scripts/`, and the timing and telemetry code.
`docs/qwen-mtp-editable-surface.md` is the definitive editable-vs-trusted
table.

## How to run

```bash
./setup.sh && ./setup-qwen-mtp.sh
./benchmark-qwen-mtp.sh --local-iterate
./benchmark-qwen-mtp.sh --local-submit
```

`./setup.sh` checks the toolchain, builds the Swift harness and the MLX Metal
library, and downloads and verifies the pinned backbone. `./setup-qwen-mtp.sh`
stages the MTP head on top of it. These are the manifest's `setupCommand`,
`benchmarkCommand` and `preSubmitCommand`.

`--local-iterate` is the fast edit loop. `--local-submit` is the longer
pre-submit check. Both write a local estimated `score.json`. The official
score exists only on the ranked runner (`m5-qwen38-27b-mtp`), so treat local
numbers as directional.

## Scoring

The score is a decode-only paired speedup, anchored at serial = 1.0. A ranked
run times all 8 hidden pool prompts over a 512-token decode window. For each
prompt, the trusted parent measures your candidate and a true serial control
(MTP depth 0) back to back in the same thermally-gated session, and takes the
raw ratio of their mean seconds per token. The published score is the median
of those 8 raw ratios; 8 is even, so the median is the mean of the two
central order statistics. There is no normalisation step and no prompt
lottery. The floor is `0.90` on the published median and the ceiling is
`3.0`. An unmodified tree medians ~0.994, so the shipped depth-2
configuration starts ~0.6% below serial. A candidate that stops drafting is
the serial control and scores exactly 1.0, which is a legal submission.
`benchmark.json` `scoring` and `fixtures/qwen3_8_27b_mtp_track.json` carry
the exact wording.

## Correctness

Correctness is a hard gate and the score is null when any gate fails. A
ranked run checks the public drift tripwire, the hidden teacher-forced base
case, the hidden anchor, free-run, behavior and GPQA gates, the GPQA TTFT
guardrail, the semantic GPQA judge, and the token-fidelity gate.

Token fidelity is absolute on this track. Every emitted token must equal the
serial trajectory, the row ledger must close over the drafts actually
proposed, and the trusted parent reference-checks every declared row after
the timed window. Your code declares nothing about its depth: the effective
per-round counts are read out of the parent's own journal and sealed into the
report.

The checked-in public fixtures are M5-generated. A near-tie argmax can
diverge on another Apple Silicon generation even for correct code, so check
whether unmodified `main` fails at the same token position before treating a
local failure as a regression.

## Excluded on every track

Two rule sets bind every submission and are what the trusted submission
static review reads.

- No caches or memos keyed on a request's input tokens whose only possible
  hit is the benchmark harness repeating an identical computation.
  Bit-identical output does not make that legitimate; the benchmark measures
  single-pass inference. Input-independent caches (weights, dequantized
  tensors, RoPE and mask tables keyed on shapes and offsets) and
  within-request KV reuse stay fine.
- The retired serial non-speculative track rules remain the reference the
  static review cites for unsanctioned speculation. That rule text comes from
  the retired Laguna XS 2.1 serial track and still binds here:
  prompt-lookup decoding, n-gram / suffix / token-history drafting,
  same-target lookahead, cross-request future-logit and KV buffers,
  deferred cache rows, and commit / rollback / recommit markers are all
  excluded. Under that serial track a one-token decode request advances
  exactly one position and leaves no pending future token. This track's
  sanctioned native-MTP draft-and-verify decode is the deliberate exception;
  every unsanctioned future-token path stays excluded.

## Submitting

Account and submission management run through the Yukon CLI (`mlxfast`), not
through `mlxfast-swift`. The installer defaults to `~/.local/bin`, so expose
that directory first:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
mlxfast login <api-key> --api <url>
mlxfast clone <benchmark-id-or-name>
mlxfast submit --model "<model name>" --note "..."
mlxfast submissions
```

`mlxfast submit` reads `benchmark.json` and uploads only `editablePaths` as a
gzip tar archive. The backend applies it to the frozen benchmark checkout and
re-enforces the editable surface server-side before running hidden
validation. Submission has replace semantics over `editablePaths`, and
`mtp-head.manifest.json` and `mtp-head/` are optional: an archive that omits
them keeps the organizer-pinned head. `mlxfast submit` does not run
`preSubmitCommand` for you.

## Useful commands

```bash
swift test --force-resolved-versions
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
swift build -c release --force-resolved-versions
tools/build-mlx-metallib.sh
./setup.sh && ./setup-qwen-mtp.sh
./benchmark-qwen-mtp.sh --local-iterate
./benchmark-qwen-mtp.sh --local-submit
```

Always pass `--force-resolved-versions` to direct `swift build` and `swift
test` runs. The dependency graph is frozen, and a bare invocation can
silently rewrite `Package.resolved`, after which `./setup.sh` and the
benchmark scripts refuse to run until you restore it with
`git checkout -- Package.resolved`.

## Authorities

| Question | File |
|---|---|
| What is editable, what the commands are, how the score aggregates | `benchmark.json` |
| Track contract: pins, timed pool, scoring semantics, calibration | `fixtures/qwen3_8_27b_mtp_track.json` |
| What a ranked run actually executes | `.github/workflows/qwen-mtp-ranked-benchmark.yml` |
| Editable-vs-trusted, path by path | `docs/qwen-mtp-editable-surface.md` |

Where this document and `benchmark.json` disagree, the manifest wins.
