# Repo cutover runbook — serving `qwen3.8-27b-mtp-v1` from this repository

> **REPO RENAME — 3.6 → 3.8.** This repository is being renamed
> `Layr-Labs/qwen-3.6-mtp-challenge-dev` → `Layr-Labs/qwen-3.8-mtp-challenge-dev`
> and repurposed for the Qwen 3.8 27B track. Every slug below that an operator
> would *run* against this repository (`gh -R …`, `GH_REPO=`, the Yukon
> `sourceUrl`) now reads the NEW name. Slugs that record a PAST event — the
> 2026-08-13 split out of `Layr-Labs/mlxfast-challenge-dev@ba5f9703`, and
> `mlxfast-challenge-dev` itself, which is a *different, still-live* repository
> serving the DFlash/serial track — are provenance and are deliberately left
> spelled as they happened. Nothing in this file goes live until the operator
> completes the 3.8 bring-up; see the `QWEN38-PENDING-RELEASE` markers in the
> ranked workflow.

> **BACKBONE IDENTITY — 2026-08-14, second decision of the day.** The reference
> checkpoint is **our own** MLX 4-bit affine / group_size 64 conversion,
> produced under a **pinned mlx 0.32.0 toolchain**, of the official bf16 base
> `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (55.6 GB).
> It publishes as `EigenLabs/Qwen3.8-27B-4bit`. 1,847 tensors — all
> `language_model.`, none `mtp.`. Geometry read and confirmed **identical to
> 3.6**, so every geometry hedge in the tree stays discharged; the quantization
> contract is carried unchanged because the `quantization` /
> `quantization_config` blocks are exactly
> `{"group_size": 64, "bits": 4, "mode": "affine"}` and equal.
>
> *What this replaced.* Earlier the same day the track adopted a third-party
> **personal-account** conversion of this same bf16 base, on inspection of its
> quantization blocks and file inventory, with a deterministic reconversion
> cross-check **reserved, not performed**. That adoption was **TERMINATED** by
> the operator's validation kill-switch: the reserved cross-check was run and
> **994 of 1,847 tensors differ numerically** from our own mlx-0.32.0
> conversion. A reference checkpoint whose bytes we cannot reproduce is not a
> reference. The upstream bf16 base did not move.
>
> **Box 3 must stage our snapshot** at
> `/opt/bench-runner/cache/huggingface/hub/models--EigenLabs--Qwen3.8-27B-4bit/snapshots/<REVISION>`,
> which is the path the ranked workflow and the goldens-provisioning workflow
> both pin (and which the host preflight asserts is spelled identically in
> `MLXFAST_QWEN_MTP_TARGET_DIR` and `MLXFAST_REFERENCE_DIR`). All three
> spellings read `eda45ab47f465d08d6558f0353a2346e2eb9d5b3` where `<REVISION>`
> goes, as of the 2026-08-14 publish, so the path resolves once box 3 stages the
> snapshot. **The repository is private** — staging needs a Hugging Face token
> with read access, and so does the head's.
>
> Still pending, now under **two** markers rather than four:
> `QWEN38-PENDING-RELEASE` (every hidden and public artifact, which needs the
> 3.8 tower to regenerate) and the retained `QWEN-MTP-PENDING-ORGANIZER`. The
> two upload markers are **retired**: the backbone published at the revision
> above with a generated 10-record byte manifest, and the MTP head
> `EigenLabs/Qwen3.8-27B-MTP-bf16` — 15 bf16 tensors extracted from the bf16
> base — published at `26a328e070875b0314d652a039b6b59902690f03` with a
> 4-record one (republished the same day to add the `config.json` and
> `model.safetensors.index.json` the head loader requires; 2 records /
> 849401866 bytes at the first publish, 4 / 849406438 now). Both markers were
> deleted from the workflow, the gate and the
> fixtures in the same commit that resolved them, because a marker no value can
> carry is documentation rather than a tripwire. The calibration interlock stays
> `"0"` and both contract enablement flags stay `false`, so every dispatch is
> still refused.

**Scope.** Everything that has to happen *outside this repository* for
`Layr-Labs/qwen-3.8-mtp-challenge` to serve the ranked Qwen 3.8 27B
native-MTP track. The in-repo half is already done and is not repeated here;
see [`qwen-mtp-go-live-runbook.md`](qwen-mtp-go-live-runbook.md) for the track
itself (hidden pool, goldens, calibration, the Step-D contract flip, rollback)
— that runbook records the **3.6** go-live and is history; the 3.8 bring-up
re-runs it against re-generated artifacts.

**Status.** `main` here IS the track. It was split from
`Layr-Labs/mlxfast-challenge-dev@ba5f9703` with the repo-identity repoints
applied, the TEMPORARY `qwen36-mtp-track` ref allowlist arm removed from the
ranked workflow and both `enforce-trusted-qwen-mtp-*` guard scripts, and
`benchmark.json` carrying the track manifest natively. A signed-commits ruleset
is active. Every step below is operator- or org-admin-gated and is tracked on
issue #1.

Content-addressed identities were deliberately NOT renamed at the 2026-08-13
**split** — they were chosen to survive a change of repository, and at that
point they still read `mlxfast-challenge-dev-qwen-mtp` (manifest `name`) and
`qwen3.6-27b-mtp-v1` (track id, leaderboard namespace, R2 object key prefix).

They ARE renamed at the 3.6 → 3.8 **model cutover**, which is a different
event: a new model means a new leaderboard namespace and a new R2 object key
prefix, because mixing 3.8 scores into the 3.6 namespace would silently merge
two incomparable populations. They now read `mlxfast-challenge-dev-qwen38-mtp`
and `qwen3.8-27b-mtp-v1`, with the timed-decode target id
`lowsim-prose-qwen38-v1`. The `mlxfast-challenge-dev-` prefix on the track name
is kept on purpose: it is the benchmark-family namespace, not a repository
slug, and Yukon joins on it.

Order matters: **1 → 2 → 3 → 4** must all be green before the verification
dispatches in **5**, and Yukon (**6**) is re-pointed only after a ranked
dispatch has actually produced a score.

---

## 1. Box 3 runner

Box 3 (`m5-max-128gb-3`, Apple M5 Max) is the only machine that serves this
track, under runner label **`m5-qwen38-27b-mtp`** — the label
`.github/workflows/qwen-mtp-ranked-benchmark.yml` pins in `runs-on:
[self-hosted, m5-qwen38-27b-mtp]`. The box currently serves
`Layr-Labs/mlxfast-challenge-dev`; the cutover repoints it here.

> **QWEN38-VERIFY-AT-RELEASE — the runner label was NOT renamed.** The label is
> a *physical property of box 3*, set by the operator's supervisor config, not
> something this repository can assert into existence. It is deliberately left
> at `m5-qwen38-27b-mtp` in the workflow and in `QwenMTPTrackNamingTests`
> rather than guessed at, and it is fail-closed either way: whichever of the
> two spellings the box does not carry, the job simply never gets picked up.
> Decide the label with the box, then move the workflow's `runs-on:`, the
> `("runner label", …)` entry in `newSurfaceNames` and its
> `theEnumeratedNamesAreTheOnesTheFilesUse` pin together in one commit.

Deploy detail lives in the operator repo at
`m5-machine-scripts/deploy/qwen36-mtp/DEPLOY-RUNBOOK.md` (§1.3 for box 3) and
`m5-machine-scripts/README.md` for the supervisor/manifest machinery. This
section is the *what changes at the cutover*, not a substitute for those.

**1a. Coordinate first.** Box 3 is a single-ranked-track box by operator
decision (2026-08-12), and there are active `ktest-*` baseline experiments
dispatching against `mlxfast-challenge-dev`. Repointing `GH_REPO` breaks their
dispatches the moment the supervisor mints its next registration. Agree a
window with whoever owns those runs before touching anything. The box lock is
per track, so two ranked tracks would *not* serialize against each other while
sharing `/Users/Shared/bench-jobs/.baseline-clones` and the bench uid that
`phase_quiesce` reaps wholesale — that is why the box serves one at a time.

**1b. Repoint the JIT supervisor.** There is no long-lived registered runner to
re-register: the root LaunchDaemon `com.bench.supervisor` mints a **single-use
JIT registration** per job via `/opt/bench/gh-app-mint.sh`, which sources
`/opt/bench/gh-app.env` for the App id, installation id, target repo and runner
group. The cutover is one line in that file:

```text
GH_REPO=Layr-Labs/mlxfast-challenge-dev   ->   GH_REPO=Layr-Labs/qwen-3.8-mtp-challenge
```

The App installation behind `GH_APP_ID` must already have access to this
repository or the mint fails closed — see step 3.

> **App identity, resolved 2026-08-14** (JWT-authenticated `GET /app` from the
> box's own key): the JIT-minting app is **`mlxfast-custom-runner`** — App ID
> `4252022`, org installation `145329821`, owner Layr-Labs, created 2026-07-09,
> permissions `administration:write` + `metadata:read`,
> https://github.com/apps/mlxfast-custom-runner. The name predates the Qwen era
> (it was created for the original mlxfast bench fleet), so do not expect
> "bench" or "qwen" in the installations list — this surprised the first
> operator to look. It is a DIFFERENT app from `yukon-eigen-dev` (step 3); both
> need repository access here.

**1c. Re-sign the signed manifest. [QUARANTINE RISK]** `/opt/bench/gh-app.env`
is content-pinned by the signed baseline manifest. Editing it without a re-sign
parks the box at the next between-job integrity audit. After the edit:

```bash
sudo /opt/bench/gen-manifest.sh                 # re-sign so installed == pinned
sudo launchctl kickstart -k system/com.bench.supervisor
```

Follow the re-sign procedure in the `m5-machine-scripts` runbook exactly
(park → apply → re-sign → audit) rather than improvising; the audit is what
proves the box is not drifting.

**1d. Confirm DFlash is no longer dispatched here.** The DFlash artifacts may
remain on disk — a retired track's tree is not deleted while it might still be
needed — but check the DFlash workflow's runner targeting before the window
opens. The installer's preflight only *warns* on an installed DFlash baseline
or an active DFlash box lock; treat that warning as a prompt to check dispatch,
never as a reason to delete anything.

**1e. Verify.** `/opt/bench/gh-app.env` reads the new `GH_REPO`, the manifest
audit is clean, `/opt/bench/quarantine.flag` is absent, and a job dispatched
from this repository is picked up by a `Runner.Worker` on box 3.

## 2. Repository secrets

Four secrets are referenced by the qwen-mtp workflows. The workflow computes
presence booleans (`MLXFAST_PRIVATE_PROMPTS_R2_PRESENT`,
`MLXFAST_ANTHROPIC_PRESENT`) and the hidden-fixture download and the semantic
GPQA judge fail closed without them.

| secret | where it lives today | action |
|---|---|---|
| `R2_ACCESS_KEY_ID` | repo-level secret on `mlxfast-challenge-dev` | copy to this repo |
| `R2_SECRET_ACCESS_KEY` | repo-level secret on `mlxfast-challenge-dev` | copy to this repo |
| `R2_BUCKET_ENDPOINT` | repo-level secret on `mlxfast-challenge-dev` | copy to this repo |
| `ORG_ANTHROPIC_API_KEY` | **org** secret, selected-repository visibility | add this repo to its visibility list |

The three R2 values are repo-level and their values are **not readable** from
the API — copying them is not a scripted migration. Whoever holds the values
runs, once per secret:

```bash
gh secret set R2_ACCESS_KEY_ID     -R Layr-Labs/qwen-3.8-mtp-challenge
gh secret set R2_SECRET_ACCESS_KEY -R Layr-Labs/qwen-3.8-mtp-challenge
gh secret set R2_BUCKET_ENDPOINT   -R Layr-Labs/qwen-3.8-mtp-challenge
```

`ORG_ANTHROPIC_API_KEY` is **not** copied — it stays a single org secret. An
org admin adds this repository under **Org settings → Secrets and variables →
Actions → `ORG_ANTHROPIC_API_KEY` → Repository access**. Creating a repo-level
copy would fork the key and defeat rotation.

Verify with `gh secret list -R Layr-Labs/qwen-3.8-mtp-challenge` (the org
secret shows once visibility is granted) — and then really by the gates-only
dispatch in step 5, which is the only check that exercises the values.

## 3. GitHub App installation (`yukon-eigen-dev`)

Yukon imports the manifest and dispatches ranked runs through the
`yukon-eigen-dev` GitHub App. Installation management refuses non-admin tokens,
so an org admin adds this repository under **Org settings → GitHub Apps →
`yukon-eigen-dev` → Configure → Repository access**.

Separately confirm the App id in `/opt/bench/gh-app.env` (the one
`gh-app-mint.sh` uses for JIT runner registration) also has access to this
repository. If the two are the same installation, step 3 covers step 1b; if
they are not, both need granting. Verify rather than assume — a missing grant
shows up as a supervisor that mints nothing and a queue that never drains.

## 4. Visibility: private → internal

The repository was created private; the creation-time flip to internal was
refused for a non-admin token. An org admin sets **Settings → General →
Danger Zone → Change repository visibility → Internal**.

Do this *before* announcing the track: participants and Yukon's importer both
need read access, and internal is the visibility the sibling challenge repos
use.

## 5. Verification dispatches

Gates first, ranked second:

```bash
gh workflow run qwen-mtp-ranked-benchmark.yml --ref main -f run_benchmark=false
# ... wait for it to finish and go green ...
gh workflow run qwen-mtp-ranked-benchmark.yml --ref main -f run_benchmark=true
```

- `run_benchmark=false` is a gates-only dry run: it exercises the trusted-repo
  and workflow-ref guards, the hidden-fixture download, correctness and the
  judge, and publishes **no** score. It is the cheap proof that steps 1–4
  landed.
- `run_benchmark=true` is the ranked run and is what Yukon will later
  reproduce.

**Serialize.** Runs serialise on the single `m5-qwen38-27b-mtp` runner
regardless of the workflow's concurrency group, so dispatch **sequentially**
rather than stacking the queue. **Between runs, confirm the quarantine flag is
absent** (`/opt/bench/quarantine.flag`); the workflow fails fast on it, but a
stacked queue turns one quarantine into a run of red jobs.

If the box did quarantine, read `/opt/bench/quarantine.flag.drift` for the full
`< baseline / > live` diff, fix the cause, then
`sudo /opt/bench/janitor.sh --clear-quarantine`. **Never reboot a box as a
troubleshooting step** — an OS update once reset `/etc/pf.conf`'s anchor lines
and the box served with open bench egress. Park, inspect, fix, re-sign.

Note what actually gates a ranked run, and budget effort accordingly: the
in-harness acceptance band does not. The real guardrails are the speedup floors
in `overlay-paired-timing.sh`, `baseline_band_check` against the on-box
calibration, `plausibility_check`, and `MAX_PLAUSIBLE_SPEEDUP`.

## 6. Yukon re-point

The dev benchmark **`6eea7522-1e3b-4741-89bb-6094f853d6f5`** was imported from
`mlxfast-challenge-dev`'s shim branch `qwen36-mtp-yukon-import` — a branch that
existed only to give Yukon a manifest to import before this repository existed.
It points at the wrong repository and must not survive the cutover.

There is no import button in the UI: import is a gated API call. All of the
calls below need `Authorization: Bearer <yukon API key>` where the key belongs
to an **importer-allowlisted account** — the allowlist is the API's
`YUKON_BENCHMARK_IMPORTER_EMAILS` Fly config (dev: `yukon/fly.dev.toml`), and
it matches the **account email**, not the GitHub username (`import-benchmark.ts`'s
header comment says username; it is wrong). Archive/open are additionally
**owner-gated**: only the account that imported a benchmark can mutate it, so
`6eea7522-…` must be archived by whoever imported it. `api-dev.ecdsa.fail` is
the `yukon-api-dev` Fly app.

In order:

1. **Archive** benchmark `6eea7522-1e3b-4741-89bb-6094f853d6f5`:

   ```bash
   curl -sX POST https://api-dev.ecdsa.fail/api/benchmarks/6eea7522-1e3b-4741-89bb-6094f853d6f5/archive \
     -H "Authorization: Bearer $YUKON_API_KEY"
   ```

   No request body. Archive is a soft delete (`deleted_at`), so its run history
   survives as the pre-cutover record. Archive is only allowed from `draft`,
   `ready`, `failed` or `closed` — **if it is `open`, `POST .../close` it
   first**, otherwise the archive 409s.

2. **Re-import** from this repository's `main`:

   ```bash
   curl -sX POST https://api-dev.ecdsa.fail/api/benchmarks \
     -H "Authorization: Bearer $YUKON_API_KEY" -H 'content-type: application/json' \
     -d '{"sourceUrl":"https://github.com/Layr-Labs/qwen-3.8-mtp-challenge","sourceBranch":"main"}'
   ```

   There is no `ref`/commit field — `sourceBranch` is the pin, and
   `manifestFilename` defaults to `benchmark.json`, which is exactly the
   Qwen-MTP manifest here, so neither it nor `rootDir` needs setting. The
   equivalent wrapper is
   `YUKON_API_KEY=… bun run import-benchmark --api https://api-dev.ecdsa.fail <url>`
   from the `yukon` repo.

   Import must run *after* step 4 (the importer needs read access to clone) and
   against a `main` that already produced a green ranked run in step 5. It
   returns **202** with `{ benchmark, job }` and queues a baseline job rather
   than blocking on it — poll `GET /api/benchmarks/:id` (public) until `status`
   is `ready` or `failed`. **The import mints a NEW benchmark UUID** — record
   it.

3. **Open** the new benchmark so it accepts submissions:

   ```bash
   curl -sX POST https://api-dev.ecdsa.fail/api/benchmarks/<NEW-UUID>/open \
     -H "Authorization: Bearer $YUKON_API_KEY" -H 'content-type: application/json' -d '{}'
   ```

   Omitting `closesAt` opens indefinitely; if you do set it, it must be at
   least 24h in the future or the call 400s with `invalid_close_time`.
   Submissions 409 unless status is exactly `open`, so skipping this is the
   failure mode where the board renders and every submission bounces.

4. **Re-point the board.** Set `NEXT_PUBLIC_QWEN38_BENCHMARK_REF` to the new
   UUID in **both**:

   > **Name collision, read this before grepping.** `NEXT_PUBLIC_QWEN38_…` is a
   > *pre-existing* Yukon UI variable name that has always referred to this
   > board, back when the track was Qwen **3.6**. It is not part of the 3.6 →
   > 3.8 rename and must NOT be "corrected" in either direction: it lives in
   > `yukon/apps/challenges-ui`, outside this repository, and renaming it here
   > would only desynchronise the two. It is the one place in this tree where
   > the string `qwen38` does not mean the 3.8 track.

   - `yukon/apps/challenges-ui/.env.local` (local dev), and
   - the Vercel project environment for `challenges-ui`.

   Then **redeploy**: `NEXT_PUBLIC_*` values are inlined at build time, so
   changing them in the Vercel dashboard alone does nothing. Use the UUID
   rather than `owner/name` — Vercel normalizes encoded slashes through the
   same-origin proxy. This variable is the `qwen38` board's own ref, not a
   shared one; nothing else needs changing.

5. **Delete the shim branch** `qwen36-mtp-yukon-import` from
   `mlxfast-challenge-dev`. It has no remaining consumer once the new benchmark
   is open, and leaving it invites a future re-import from the wrong repo.

Verify: the board loads against the new UUID, shows the ranked run from step 5,
and a test submission dispatches to box 3.

---

## Boundaries

**Never merge this `main` back into `mlxfast-challenge-dev`.** It is not a
policy preference, it fails by construction: this history removes the DFlash
manifest guards that the `benchmark.json` manifest swap invalidates, and
`benchmark.json` here is the Qwen-MTP manifest. Merging would take the DFlash
track's manifest out from under it and trip its guards. The two repositories
are permanently divergent by design; the split *was* the merge the go-live
runbook's removal contract referred to. Port fixes across as cherry-picks of
individual files, in the direction that makes sense, never as a branch merge.

**Signed commits are enforced.** A ruleset requires signed commits on `main`.
Verify locally with `git log --show-signature` or `git log --format='%h %G?'`
— every commit must show `G`. Unsigned commits are rejected at push, which on
an operator box mid-window is an unpleasant surprise.

**Organizer contract fixtures are report-only.** `fixtures/` carries organizer
contract material — most importantly `fixtures/qwen3_8_27b_mtp_track.json`,
whose `official_scoring_enabled_note` still references the removed TEMPORARY
`qwen36-mtp-track` allowlist arm. That is a known, tracked staleness (issue
#1). Do not edit it as drive-by cleanup: it is organizer-owned, it is what the
enablement guard reads, and a Step-D style change to it belongs in a
deliberate single commit alongside the workflow and test pins as the go-live
runbook describes. Report defects; do not fix them here.

**Hidden material stays hidden.** The timed-prompt pool, the hidden goldens and
the GPQA reference cases live in R2 under content-addressed, track-scoped keys
and are referenced by path only. Nothing in this cutover moves or re-uploads
them — the keys were deliberately chosen to survive the repo move.
