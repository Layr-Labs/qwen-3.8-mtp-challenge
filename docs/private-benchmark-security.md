# Private Benchmark Security

The private correctness prompts and private golden data must be treated as
trusted harness material. They are not part of the contestant modifiable
surface. This document describes the security architecture of the ranked
single-machine pipeline: how private material is handled, what confines
submitted code while it runs, which output channels are closed, and which
gaps are known and stated rather than papered over.

The DFlash track (`laguna-xs-2.1-dflash-v1`) is the default — and only — ranked
track; its ranked pipeline runs under `.github/workflows/dflash-benchmark.yml`
(the retired serial track's `benchmark.yml` was deleted). The DFlash pipeline
REUSES the serial-era security architecture described below — the same sandboxing,
GPQA and teacher-forced base-model gates, and closed output channels — driven from
the trusted DFlash harness. This document was written for the serial track;
where it says "serial track", `benchmark.yml`, or runner `m5-bench`, read the live
DFlash track, `dflash-benchmark.yml`, and runner `m5-laguna-dflash` unless the text
is explicitly describing retired/historical behavior.

## Single-machine ranked topology

Ranked runs execute through `.github/workflows/dflash-benchmark.yml` as one
job on a single operator-supervised, self-hosted Apple M5 Max
machine (runner label `m5-laguna-dflash`). Build, reference-checkpoint transform,
hidden correctness/gates, and the paired baseline/candidate timing all run
in order on that box. The
previous multi-VM topology — parallel correctness slices, a separate
paired-baseline timing VM, and a combine job — is retired; the guards it
duplicated across privileged jobs now run once inside the single job, in
the order described below.

The runner registration is ephemeral: a root supervisor on the box mints a
single-use just-in-time (JIT) registration per job, authenticated through a
GitHub App installation (no long-lived PAT). Each registration is consumed
by exactly one job and a fresh one is minted for the next, so one job at a
time is structural and a compromised job cannot accept a second job. The
workflow uses non-cancelling per-run concurrency groups; the single active
`m5-bench` runner's label queue serializes duplicate dispatches behind an
in-flight measurement.

The workflow is `workflow_dispatch`-only with no PR or push triggers (fork
code must never reach a self-hosted runner), and it verifies at runtime
(see `enforce-trusted-benchmark-workflow.sh`) that it runs in this
repository via a `workflow_dispatch` event, that the dispatched ref is in
an explicitly permitted branch namespace — `main`, `submissions/*`
(orchestrator-created submission branches), `baseline/*` (operator
verification/measurement dispatches), or `yukon/baseline/*`
(platform-authored baseline refs) — and that the executing workflow file
is the dispatched ref's own. It benchmarks the permitted ref it is
dispatched on.

Because the workflow runs the dispatched ref's own workflow file, the real
boundary is the combination of:

- the benchmark orchestrator (Yukon eigenbot) being the only creator of
  `submissions/*` branches, built from remotely validated `editablePaths` so
  their non-`editablePaths` files match `main`;
- the `Enforce modifiable surface` step re-verifying at runtime that a
  `submissions/*` branch changes only `editablePaths` relative to `main`; and
- restricting who can push `submissions/*` — and the operator/platform
  `baseline/*` and `yukon/baseline/*` — branches and dispatch the workflow.
  Non-submission refs run their exact dispatched SHA (including their own
  workflow file), so creating them is a write-access-only, trusted-role
  operation.

## Privilege rings and the bench-exec bridge

The box separates three privilege rings. The host layer is provisioned by
the box operator and audited between jobs by a signed integrity manifest
(next section); the workflow only ever consumes it, never installs or
mutates it.

- **Ring 0 — admin/root:** provisioning and the runner supervisor. Never
  runs job code.
- **Ring 1 — the `runner` account:** runs the GitHub Actions runner and all
  trusted workflow steps — guard checks, private downloads and golden
  verification, the semantic judge call, score sealing, artifact
  validation. It holds the private secrets for the duration of those steps
  and never executes submitted code.
- **Ring 2 — the `bench` account:** a locked (no password, no ssh, no
  login), ephemeral uid that is deleted and recreated between jobs. It is
  the only context that ever executes submitted or branch-controlled code:
  build, transform, GPQA-gate attachment, the public gate, the
  correctness-plus-gates pass, and the timed benchmark.

Submitted code reaches ring 2 through exactly one fixed bridge. A sudoers
grant lets `runner` invoke two root-owned entry points and nothing else:
`/opt/bench/bench-exec.sh` (the bridge) and `/opt/bench/janitor.sh` (the
between-job reset). Every invocation of submitted code in the workflow goes
through the bridge, which drops privilege to the bench uid, confines writes
to the per-job workspace under a fixed job root (any other workspace path
is refused), applies a per-uid process cap, starts from a clean environment
(sudo `env_reset` plus the bridge's own secret denylist), and injects a
hardened, merged Seatbelt worker profile into the harness via
`MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE` / `MLXFAST_USE_RUNTIME_WORKER=1`.

The runtime worker — the one child process that executes submitted model
code — runs under that Seatbelt profile: no network, no fork/exec beyond
its own binary, writes confined to the workspace, and the hidden golden
denied by literal path (`BENCH_GOLDEN_PATH`). A sandbox probe step
re-verifies on every run — before any private material is fetched — that
the host's `sandbox-exec` still denies golden reads, private-directory
reads, non-workspace writes, network, and fork/exec under a worker-shaped
profile. A fourth designed layer, a uid-scoped PF network-egress block on
the bench account, is staged but currently deferred; see "Current posture
and known gaps".

## Between jobs: janitor, signed audit, quarantine

After every job — pass or fail — the janitor resets and audits the box. It
runs as the workflow's final always-on step, and the supervisor runs it
again between runner registrations:

- **Reset:** kill every bench-owned process, delete and recreate the bench
  account from a root-owned template, and purge bench-owned scratch and the
  per-job rendered sandbox profiles.
- **Audit:** verify the cryptographic signature over a baseline manifest of
  the protected surface (the sudoers bridge, the `/opt/bench` payload, PF
  wiring, launch daemons, the runner install), then diff a fresh scan of
  that surface against the manifest.

Any drift writes a quarantine flag, fails the run loudly, and parks the
supervisor with an operator alert. The workflow's first step refuses to
start while the flag exists, so a quarantined box takes no further ranked
jobs until an operator explains the drift or re-images the machine. A
compromised host must never silently publish a score.

## Required GitHub setup

Store private prompt/golden download credentials only as secrets on the
`benchmark-private-prompts-v2` GitHub Environment. Do not store them as
repository-wide or organization-wide secrets.

Configure the `benchmark-private-prompts-v2` Environment with:

- Deployment branches limited to the same namespaces the runtime guard
  (`enforce-trusted-benchmark-workflow.sh`) admits: `main`, `submissions/*`
  (orchestrator-created submission branches), `baseline/*` (operator
  verification/measurement dispatches), and `yukon/baseline/*`
  (platform-authored baseline refs). Keep this policy and the guard script
  in sync. Do not grant fork access. Pushing any of these refs and
  dispatching the workflow both require write access to this repository —
  contestants have neither, and submissions enter only through the
  orchestrator — and required reviewers still gate each run's access to
  the secrets.
- Required reviewers for private benchmark runs.
- R2 private-object credentials:
  - `R2_ACCESS_KEY_ID`
  - `R2_BUCKET_ENDPOINT`
  - `R2_SECRET_ACCESS_KEY`
- Non-secret timed-prompt operator mirrors:
  - `MLXFAST_TIMED_DECODE_PROMPT_SHA256` =
    `0b67162cbea948f380e693398b19ba797892b5100cd9e0e415a87e900ac79e03`
  - `MLXFAST_TIMED_DECODE_PROMPT_BYTES` = `2750`

The workflow keeps the reviewed timed-prompt pins in trusted source rather
than consuming these mutable variables. The Environment values are retained
for operator tooling and must match the source pins.

The single ranked job declares `environment: benchmark-private-prompts-v2`, so
the deployment-branch policy and required reviewers gate every run that can
read those secrets. The semantic GPQA judge additionally requires the
`ORG_ANTHROPIC_API_KEY` secret. All of these are consumed only by trusted
runner steps: they are never exported across the bench-exec bridge (sudo
resets the environment at the bridge, and the bridge's denylist
double-guards), so the bench uid — and therefore submitted code — never
sees them.

## Private golden and GPQA handling

Full benchmark dispatches (`run_benchmark=true`) download three objects from
the private R2 bucket in trusted steps. All live under
`correctness_prompts/laguna-xs-2.1-serial-v2/` and use these immutable,
content-addressed names:

- `hidden-correctness-golden-94239d59b435eb8f370c82bcf8c86822d1bbc1094e3650aeff3abc5558137023.json`
- `gpqa-reference-cases-4a6d847c6535561e8d4094e2bb764be96c2cd8f4ca310614120058c3c6a7d26f.json`
- `timed-decode-prompt-0b67162cbea948f380e693398b19ba797892b5100cd9e0e415a87e900ac79e03.txt`

The first is the hidden teacher-forced base case, the second is merged into
the golden as 5 behavior gates, and the third is the independent timed
prefill/decode target.
Correctness-only dispatches (`run_benchmark=false`) fetch no private
material at all. The 2026-07 Poolside Laguna XS 2.1 re-pin rotated these
object keys from their former `-gemma` names: the correctness objects are
regenerated from the Laguna reference — new tokenizer, vocab 100352 —
through the same organizer-controlled offline process that
`docs/gemma-migration-r2-checklist.md` records for the previous Gemma
migration (that checklist is the template for this pass).

Raw private bytes land only in a runner-only `0700` per-run directory; all
three objects are independently verified against SHA-256 and byte-count pins
before use. The final baseline-commit and calibration-ready interlocks remain
separate fail-closed checks. Protected M5 provisioning completed on
2026-07-23 for baseline commit
`15852ee52858def42ddd4f32bca7e59d275e020e`, so the workflow now pins
`MLXFAST_POOLSIDE_V2_CALIBRATION_READY=1`; a missing or mismatched on-box
baseline/calibration still fails host preflight.
The GPQA augmentation step (`attach-gpqa-gates`) executes code from the
submitted build (it loads the tokenizer), so it runs through the bench-exec
bridge like every other untrusted invocation: the raw inputs are copied
into the workspace as runner-owned files and removed again immediately
after, and the augmented golden is hash-anchored in the trusted shell the
moment it is produced (the workspace copy must remain byte-identical to the
trusted copy).

Each GPQA case must carry accepted reference-model output tokens or
responses; the GPQA answer key alone is never used as an exact-token
correctness oracle. The workflow treats the R2 objects as immutable trusted
inputs and never regenerates or uploads them; update them only through the
organizer-controlled offline process. The private prompt manifest is only
an organizer input for regenerating goldens outside the benchmark workflow.
It is never written into the repository workspace, uploaded, or cached.

The public behavior gate (the drift tripwire against the checked-in public
fixture) deliberately runs before any hidden material enters the bench
workspace. During the correctness-plus-gates pass the only hidden file in
the workspace is the augmented golden: the harness parent process (bench
uid) must read it to drive teacher forcing, but the runtime worker that
executes submitted model code is denied it by literal path. The hidden GPQA
gate checks one generated token per case (the stable prefix of any longer
precomputed reference sequence) and measures hidden GPQA TTFT from prompt
prefill through the first greedy answer token; only aggregate TTFT fields
are written to `score.json` — no prompt text, expected or generated token
IDs, accepted token sets, or per-case prompt lengths.

The semantic GPQA judge runs as a trusted step after the gates pass has
written its private answer capture. The Anthropic key never crosses the
bridge, and only aggregate semantic pass counts and the judge model name
are patched into `score.json`; prompts, references, candidate answers, and
judge transcripts stay in private runner paths covered by the artifact
deny-list. One documented residual: the answer capture (which embeds hidden
reference answers) is written into the bench workspace by the harness
parent at the very end of the gates pass, and — unlike on the VM pipeline,
whose harness profile could subpath-deny it — it is briefly readable by the
worker in the few seconds between that write and worker teardown. The
window gains a submission nothing (every behavior verdict is already final
by then), and a trusted step scrubs the capture — together with the
augmented golden and all other hidden material — before any later phase
spawns another worker.

## Timed measurement and score sealing

The timed prefill/decode measurement runs last: after all compute-heavy
correctness and gate work, after the correctness-hidden-material scrub, after
the separately pinned timed prompt is downloaded to the runner-only private
directory, and after a quiescence wait (the job fails rather than start timing
on a machine with residual load or GPU utilization). It is executed by a trusted,
runner-owned measurement wrapper whose thermal-stability contract is fixed
in the script itself — readonly, not environment-overridable: every timed
phase (pinned baseline and candidate, plus any oracle generation) starts
only once the GPU has cooled below 40C (waiting up to 900 seconds), runs
under 100 ms macmon GPU telemetry, and is rejected if any loaded sample
shows GPU clocks below the per-side frequency floor (1500 MHz since the
2026-07-31 operator decision — throttling), if telemetry is missing, or if the
sealed score shows token mismatches; one gated retry is allowed.

The measurement is paired: the pinned reference baseline tree provisioned
on the box is measured back to back with the candidate behind the same
thermal gate, and the ranked speedups are that paired ratio. This is the
single-machine replacement for the old pipeline's fresh timing VM: that VM
bought thermal isolation but introduced a per-job hardware lottery the
paired baseline then had to cancel, whereas here both sides run on the same
silicon in the same session, so the ratio cancels common-mode host drift
directly and the fixed cool-down gate keeps the earlier correctness pass's
heat out of the timed window. A calibration sanity band plus a pinned
baseline binary hash guard against a swapped or pathological baseline
silently rescaling every ratio. A trusted overlay
step (`overlay-paired-timing.sh`) then merges the measured paired timing
into the sealed gates score and applies the decode/prefill speedup floors
(kept in sync with the harness constants). The workflow passes the private
timed prompt, its SHA-256, and the stable evaluation-target ID explicitly to
the wrapper. The benchmark oracle used by the timed run is self-generated per
binary. Its cache identity includes the binary hash, prompt hash, and target
ID — input-independent implementation keying, not request-keyed memoization.
The hidden teacher-forced and GPQA fixtures are not inputs to this oracle.

Scores are sealed from process stdout: the score payload is what the
benchmark process wrote to stdout, captured in the trusted shell and
validated as exactly one JSON object shaped like a score, and the
harness-authored integrity record's score hash is cross-checked against the
sealed bytes. On-disk files inside the bench-writable workspace are
untrusted once submitted code has run; later tampering with a workspace
`score.json` is discarded.

## Submission flow

The benchmark orchestrator (Yukon eigenbot) creates a `submissions/*`
branch that differs from `main` only in `benchmark.json` `editablePaths`,
then dispatches `benchmark.yml` on that branch. The workflow benchmarks the
checked-out branch directly.

On `submissions/*` branches the workflow additionally:

- enforces the modifiable surface as a content rule against the current
  trusted `main` tip (`enforce-modifiable-surface.sh` diffs the candidate
  tree directly against `main`'s tree): every path where the candidate
  differs from current trusted `main` must be inside `editablePaths`, and
  each offending path is named on failure. The candidate commit is NOT
  required to be an ancestor-descendant of the current tip — trusted `main`
  moving within `editablePaths` after a candidate was created (a mid-flight
  promotion) does not invalidate the queued candidate, matching Yukon's own
  promotion-time rebase rule. Ancestry would constrain no executed byte
  anyway: the bench workspace is always trusted `main` content plus the
  validated editable-path overlay, and the workflow file is non-editable, so
  the content rule already forces the dispatched `benchmark.yml` to match
  current trusted `main` whenever a run proceeds,
- runs the static cheat review over the editable files that differ versus
  the current trusted `main` tip — the same base as the surface check, so
  the reviewed diff is exactly the effective delta the overlay executes
  (`run-submission-static-review.sh`; editable files byte-identical to
  trusted `main` content are not sent to the judge),
- suppresses submitted correctness/benchmark process logs, and
- uploads artifacts only after validation succeeds.

Maintainers can also dispatch the workflow on `main`, an operator
`baseline/*` ref, or a platform-authored `yukon/baseline/*` ref; the
trusted-context guard rejects every other namespace. Those runs execute
their exact dispatched SHA and skip the submission-only guards.

### Serial-track speculative boundary

The published serial track does not permit prompt-lookup or other
cross-request speculative decoding. The controlling rule is that each model
invocation may compute logits and KV rows only for tokens supplied in that
invocation and must advance logical and physical KV position by exactly that
input length. A one-token request may not select or evaluate an unsupplied
future token, run a two-/three-/more-row target verification path, or retain
future logits/KV and deferred-row commit/rollback state for the next request.
Pre-hello warmup does not make such a pipeline eligible. Ordinary
within-request KV reuse, current-token-only execution, input-independent
caches, and multi-row prefill kernels backed entirely by supplied tokens remain
allowed.

`run-submission-static-review.sh` quotes this controlling rule and names each
excluded mechanism in both the system prompt and the structured user policy.
That review is a detection control. The trusted worker structurally issues
serial one-token requests and owns the request counter, but submitted model and
cache code is editable; it can perform hidden extra target work and can lie
through any pending-state accessor added to that editable surface. The current
process boundary therefore cannot enforce a truthful in-process
"no speculative state" attestation. Maintainer frontier audit remains required.

A future organizer MTP track should use a separate trusted variable-length
block protocol that authorizes draft rows and validates every accepted token
and position transition. It must define its own correctness and score contract;
it cannot silently reuse this serial track.

## Output policy

For `submissions/*` runs:

- Correctness and benchmark process logs are redirected to private runner
  temp files and are not uploaded.
- The workflow prints only fixed heartbeat lines while submitted code is
  running.
- `score.json`, `benchmark-integrity.json`, and golden hash/byte sidecars
  are uploaded only after strict schema and hash validation succeeds, and a
  deny-list check rejects prompt, golden, GPQA, model, symlink, and
  oversized artifact paths before every upload.
- A failing `score.json` is never uploaded raw (its error and token fields
  can carry hidden-golden values or submitted-code-controlled content); a
  redacted failure category is derived and uploaded instead so operators
  can tell why a run failed.
- Audit artifacts (paired timing results with telemetry aggregates, the
  transformed-weights hash, the public tripwire report) carry only
  workspace paths, commit ids, timings, and telemetry statistics, and pass
  the same deny-list check.

This prevents submitted code from using GitHub logs or uploaded artifacts
as a direct prompt-exfiltration channel.

## Current posture and known gaps

Stated plainly so this document does not overclaim:

- **The bench-uid PF network-egress block is designed and staged but
  currently deferred; it is not enabled on the box today.** Until it is
  turned on, bench-uid processes outside the runtime worker (the build, the
  transform, the harness parent) have network egress. The controls that are
  active: the runtime worker that executes submitted model code is denied
  network by its Seatbelt profile; SwiftPM dependencies are resolved by a
  trusted step before submitted code runs, so nothing on the bench side
  needs the network; secrets never enter the bench environment; and the
  sealed-stdout score path, artifact deny-list, and failure redaction close
  the upload channels. Enabling the PF block closes the remaining
  parent-process egress channel; it is operator work on the box, not a
  workflow change.
- **The pinned baseline, its calibration, the reference checkpoint cache,
  and the hidden goldens are provisioned state, not repository content.**
  The baseline tree and its calibration live on the box and are regenerated
  through the operator runbook (changing them is a ranking-contract
  change). The reference checkpoint cache is hash-verified against the
  pinned manifest on every run, so a corrupted or swapped cache can never
  feed the transform. The hidden goldens live in private R2 and are
  pin-verified on every download. After a re-image, all of this is
  re-provisioned by the operator before the box returns to ranked service.

## Residual channel

Submitted code still participates in inference on hidden prompt tokens. Any
public feedback from that run, including pass/fail, score, timing, or
repeated submission attempts, is a possible low-bandwidth covert channel.
The hardening above blocks direct extraction paths, but competition policy
should still limit repeated private benchmark attempts and avoid exposing
per-case failure details for hidden cases.

The published ranking timings (`decode`/`prefill_seconds_per_token` and the
speedups/score) are currently full-precision, which widens this channel:
submitted code can drive data-dependent timing and read it back at high
resolution. Coarsening them is a scoring-backend publishing decision (the
harness must keep full precision internally to rank and to check the speedup
floors), so it is written up as a recommendation rather than fixed in-repo —
see
[`docs/ranked-timing-covert-channel-recommendation.md`](ranked-timing-covert-channel-recommendation.md).

No prompt manifest or generated correctness golden should be committed to
the public repository.
