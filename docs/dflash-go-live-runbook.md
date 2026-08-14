# DFlash go-live runbook (`laguna-xs-2.1-dflash-v1`)

`.github/workflows/dflash-benchmark.yml` points operators here from four places,
including "step B", and until now the document did not exist. This is it.

Everything below is the work that CANNOT be done by an agent: it needs hidden
material, a judgement about whether the anti-cheat is sound enough to rank on, or a
change to an operator-owned measurement contract.

> **GO-LIVE COMPLETE — 2026-07-31 (Amendment 31); scoring updated 2026-08-01
> (Amendment 32).** Steps A–E below were EXECUTED; this runbook is retained as the
> record and as the procedure for taking the track back offline. Current live
> state: `official_scoring_enabled: true`, `reference_baseline.publication_allowed:
> true`, `tokenFidelityGateStatus: "implemented"`, `confirm_track_enabled` default
> `true`, and the ranked score is PER-PROMPT NORMALISED with a 0.95 normalised
> floor (Amendment 32). The pre-go-live "pending / false / inert" phrasing in the
> steps below is HISTORY. The box `MIN_ACCEPTED_SPEEDUP` decouple (decode-floor
> row, Amendment 32) was APPLIED 2026-08-01 (raw 0.83 → 0.50 loose sanity floor,
> manifest re-signed, audit clean); nothing in this runbook remains pending.

## 0. What is already done and verified (do not redo)

Measured on M5-C, 2026-07-30. Contract detail in
`docs/dflash-track-correctness-contract.md` (Amendments 1-32).

| thing | state |
|---|---|
| harness end-to-end | **Full `ACCEPT` at the shipping configuration** (K=2, 512 decode tokens, band-enforced, 0.83 floor): a band-enforced run on the WORST pool entry (Russell, no-op 0.8726) clears the floor; the gates path is green end-to-end in CI (run 30664724953). Each of the 8 pool entries measured no-op at the ranked window with `PARITY_OK`, aggregates 0.8726–0.8939 — see Amendment 30. Historical: 0.5892 single-lowsim-prompt (Amendment 28 era); 0.8705 on a different golden (Amendment 25 era) |
| pinned baseline | `/opt/bench-runner/baseline/laguna-xs-2.1-dflash-v1/<sha>` + `current` symlink, 23 GB, weights APFS-cloned from the serial baseline |
| baseline calibration | `/opt/bench-runner/state/laguna-xs-2.1-dflash-v1/baseline-calibration.json`, authored by `/Users/gaj/author-dflash-calibration.sh` (the wrapper does NOT write it). **The band is only valid at the decode token count it was measured at** — the seed prefill is charged inside the decode window, so the same baseline reads 0.0201 s/token at 128 tokens and ~0.0148 at 512. It must record `decode_tokens`; `measure-dflash-job.sh` fails closed if that field is absent or disagrees with the run. Re-author whenever the ranked window OR the pinned baseline binary changes — see Amendment 27. **Authored 2026-07-31 at the ranked window**: `decode_tokens: 512`, mean 0.014873 s/token from 4 accepted pairs, integrated path exercised live (`CALIBRATION_OK`). |
| runner | `m5-laguna-dflash-3-*` online on **mlxfast-challenge-dev**, labels `[self-hosted, m5-laguna-dflash]` only |
| dispatch chain | verified: job scheduled -> host preflight -> trusted context -> enablement guard fails CLOSED on the contract |
| decode floor | **0.83** (derived 2026-07-31 from the WORST no-op across the full 8-prompt pool at the ranked window: 0.8726 × 0.952, floored; supersedes 0.55/0.52/0.80, each measured off the ranked conditions), agreeing in all five sites: `benchmark.json` manifest, contract fixture, workflow comments, workflow env `MLXFAST_DFLASH_DECODE_SPEEDUP_FLOOR`, and `MIN_ACCEPTED_SPEEDUP` in the box wrapper. Rationale is Amendment 30. **SUPERSEDED by Amendment 32 (2026-08-01):** the ranked score is now per-prompt NORMALISED (raw ratio-of-means ÷ the sampled prompt's pinned no-op reference), the floor is **0.95** on the normalised aggregate (pinned equal in `benchmark.json`, the fixture, and the workflow env), and the box `MIN_ACCEPTED_SPEEDUP` is decoupled to a loose raw measurement-sanity floor of 0.50 — no longer equal to the ranked floor — APPLIED 2026-08-01 (backup `.pre-decouple050`, manifest re-signed, janitor audit clean). Pool no-ops cluster 0.8726–0.8939 (2.4% spread) so the lottery risk is small. **PROVISIONAL on character**: moderate-difficulty prose (~75% acceptance); a harder lowsim pool would lower the floor. Re-derive floor + K if the pool changes. |
| janitor audit | clean after re-signing; re-verified on all three boxes 2026-07-31 |
| hidden goldens | Correctness pair provisioned + pinned. TIMED POOL: **8 distinct entries pinned** (2026-07-31, one per domain — Amendment 30), goldens generated on M5-C and staged locally for operator R2 upload to the pinned `pool-*.json` keys; each sha re-verified after download. R2 prefix `correctness_prompts/laguna-xs-2.1-dflash/`, NO `gautham-experiments` segment (that is the BUCKET, carried by `R2_BUCKET_ENDPOINT`). Probe with `.github/workflows/dflash-probe-r2-keys.yml` before changing. |

## Step A — hidden timed-prompt pool (DONE 2026-07-31; operator R2 upload pending)

`fixtures/laguna_xs_2_1_dflash_track.json` -> `timed_prompt_pool` now holds
**8 distinct entries**, one per domain (Amendment 30), pinned by sha256/bytes.
The goldens were generated on M5-C and are staged locally at
`~/Downloads/dflash-pool-seeds/out/` with `pool-manifest.json`. **The only
remaining Step-A action is the operator uploading those 8 files to their pinned
`pool-*.json` R2 keys** (the workflow re-verifies each sha after download). The
"Select hidden DFlash timed target from the pool" step will then sample from all
8; until the flags are flipped (Step D) a ranked dispatch is still refused by
the enablement guard, so the pinned pool is inert and safe on main.

Historical requirements this satisfied (from the fixture `timed_prompt_pool_note`):

Requirements, from the fixture's own `timed_prompt_pool_note`:

- **At least 8** distinct hidden timed targets.
- **Varied prompt length and domain**, so acceptance-rate tuning cannot generalise
  across the pool.
- Each entry is `{r2_path, sha256, bytes}`, verified byte-for-byte after download.
- **Hashes are only ever pinned from the uploaded objects**, never pre-filled
  off-box.

Two measured reasons the variety requirement is load-bearing, not boilerplate:

1. A greedy self-continuation of the model is degenerate — 122 distinct tokens in
   512, no top-2 logit gap below 1.8 — and every gate on this track passed on such
   material while failing honest work on real prose (Amendment 10).
2. Draft acceptance is ~100% on repetitive text and **69%** on varied prose, which
   moves the score from 1.117x to 0.840x (Amendment 11). A pool of easy prompts
   would advertise a speedup that does not exist.

### What a pool entry is made of: a pre-tokenized prose SEED

A `timed_prompt_pool` entry's `r2_path` points at a frozen **golden**, and step B
builds that golden from a **seed**. Both are operator material and both live in
R2.

The trusted binary links no tokenizer, so a seed is **token ids, not text**: an
R2 object holding a seed-only `DFlashEmittedPlan`,

```json
{ "seed_tokens": [ 1234, 5678, ... ], "emitted": [] }
```

where `seed_tokens` is the tokenization of **real prose**. The public fixture
`correctness_prompts/public_longcopy_gate_english_512_256.json` →
`cases[0].prompt_tokens` is a checked-in example of the shape and the character
(512 tokens, 276 distinct English-prose tokens); the hidden seeds are yours and
must never be derived from the model's own greedy continuation — see step B, and
Amendment 10.

Provision one such seed object per intended pool entry (at least 8, varied in
length and domain), plus one more for the untimed correctness golden. Then run
step B once per pair. If step B's job reports

> `no operator seed named ... Provision them per docs/dflash-go-live-runbook.md step A`

it is this list that is missing.

## Step B — freeze and pin the IT-target goldens (BLOCKING)

The workflow refuses to run with an empty hidden-golden pin:

> `hidden DFlash golden pin <name> is empty; freeze the IT-target goldens (go-live
> runbook step B) and pin them here before enabling the track`

**Do not run `dflash-reference` by hand for this.** Dispatch
`.github/workflows/dflash-provision-goldens.yml`, which does the whole of step B
inside a job bound to the `benchmark-private-prompts-v2` environment — the only
place the R2 credentials exist. (Generation and upload cannot be split: the
credentials are GitHub environment secrets injected per run, and are on no dev
machine and on no runner's disk. That is why run 30604267251 hit
`404 NoSuchKey` with every other gate green — nothing had ever uploaded the
objects.)

```
gh workflow run dflash-provision-goldens.yml --repo Layr-Labs/mlxfast-challenge-dev \
  --ref main \
  -f confirm_provision_goldens=true \
  -f correctness_seed_r2_path=<R2 key of a prose seed plan> \
  -f correctness_object_path=<R2 key to write the correctness golden to> \
  -f bench_seed_r2_path=<R2 key of a DIFFERENT prose seed plan> \
  -f bench_object_path=<R2 key to write this timed golden to>
```

It generates both goldens from the **pinned baseline tree**, verifies them,
uploads them, **re-downloads each object and computes `sha256` + `bytes` from the
downloaded bytes**, and prints the four pins. It does not edit
`dflash-benchmark.yml`: apply the pins by hand, in a reviewed commit. Re-dispatch
once per `timed_prompt_pool` entry (step A wants at least 8, each from a
different prose seed).

**Already done once** (2026-07-31): the correctness golden and the first timed
golden are provisioned, uploaded and pinned. Do not redo that pair; dispatch this
for entries 2-8.

**Object paths are keys, not bucket-qualified paths.** Pass
`correctness_prompts/laguna-xs-2.1-dflash/<name>.json`, with **no**
`gautham-experiments/` segment -- `gautham-experiments` is the bucket and is
carried by `R2_BUCKET_ENDPOINT`. This cost three ranked dispatches to learn, in
both directions; probe run 30613434387 settled it. If a key is ever in doubt, run
`.github/workflows/dflash-probe-r2-keys.yml` (about a minute) instead of a
dispatch (30-40 minutes).

### The seed must be PRE-TOKENIZED REAL PROSE. Never `--seed-generate`.

This step used to read, in full: "`dflash-reference` builds them;
`--seed-generate N` extends a seed and `--generate N` produces the emitted
chain." **That instruction was wrong and it contradicted this track's own
correctness contract.**

`--seed-generate N` extends the seed by N **reference-generated** tokens — greedy
self-continuation. `docs/dflash-track-correctness-contract.md` **Amendment 10**
measured exactly that material and condemned it:

| golden | seed len | distinct seed tokens | rows w/ top-2 gap < 0.25 | min top-2 gap |
|---|---|---|---|---|
| `seam-512-golden.json` | 512 | 122 | **0** | 1.875 |
| `seam-b-golden.json` | 600 | 122 | **0** | 2.625 |
| `seam-a-golden.json` | 509 | 122 | **0** | 1.875 |
| `varied-512-golden.json` (prose) | 512 | 317 | 3 | **0.0000** |

Greedy self-continuation degenerates into repetition; the near-tie regime
Criterion E exists to handle is never entered; draft acceptance measures ~100%
against 69% on varied prose; and **Amendment 11** prices the difference at
**1.117x versus 0.840x**. A ranked golden built that way would advertise a
speedup that does not exist. The hidden ranked prompts are prose.

So: **the seed is operator-supplied, pre-tokenized real prose.** The trusted
binary links no tokenizer, so a seed arrives as **token ids**, not text — an R2
object holding `{"seed_tokens": [...], "emitted": []}`. The shape of a
legitimate one is visible in the public fixture
`correctness_prompts/public_longcopy_gate_english_512_256.json` →
`cases[0].prompt_tokens` (512 tokens, 276 distinct: the tokenization of real
English prose). Hidden prose seeds come from the operator's private store; the
provisioning job never invents one, and **fails closed** with no seed named
rather than falling back to `--seed-generate`.

This is enforced, not merely written down. `.github/scripts/check-dflash-golden-degeneracy.sh`
screens the operator's seed **before** generation and both goldens **after** it,
against thresholds derived from the table above (seed variety ≥ 0.40 distinct
fraction; at least one row with a top-2 gap < 0.25; minimum top-2 gap ≤ 1.0). It
prints all three statistics on every run, pass or fail. The provisioning job also
asserts its own generation argv contains no `--seed-generate`.

### What the provisioning job verifies before it uploads

Each golden must carry `reference_self_consistent: true` **and**
`emitted_tokens[i] == rows[i].sequential_argmax` for every row — the
self-consistency flag alone did not check that until Amendment 10, and three
goldens shipped with the contradiction. It must also carry at least as many rows
as the gate will request (512), and it must pass the degeneracy screen. Any of
those failing aborts before anything is uploaded.

## Step C — the serial frequency floor (RESOLVED 2026-07-31: operator set 1500)

**APPLIED by operator decision 2026-07-31** (`MIN_FREQ_SERIAL=1500`, matching the
dflash side; manifest re-signed, audit clean) **and validated the same day at the
ranked window**: across a progressively warm box (peak 50.1°C on dflash phases),
serial steady clocks ran 1585-1588 MHz — every phase would have died under 1600,
and every phase held >85 MHz above the new floor. Dflash side flat at 1602-1605.
The derivation comment travels with the constant in the wrapper itself. History
follows.

**Escalated 2026-07-31 from "intermittent" to BLOCKING.** Previously this was seen
only as the box warmed across pairs. Measured at the RANKED window (512 decode
tokens, the ranked window), it rejects **deterministically on the
first pair and on the gated retry**:

```
pair1-serial.a1   loaded=91  steady_n=84  min=1589 MHz  max=1620 MHz  -> REJECT
pair1-serial.a2   loaded=96  steady_n=87  min=1592 MHz  max=1620 MHz  -> REJECT
                                                        FATAL(code=5) pair 1
```

The floor sits **inside the machine's normal loaded range** (1589-1620 MHz) once a
phase runs the ranked length: a 512-token serial phase takes ~52 s against ~13 s at
128 tokens, so it settles into a 2% lower steady clock. At 128 tokens the same
baseline passed at 1603 MHz — 3 MHz of margin, which is why this read as
intermittent. Sample counts are healthy (91-96 loaded), so this is purely the
frequency arm, not the sample-count arm that bit the serial production wrapper.

**Consequence: the calibration band cannot be re-authored at the ranked window
until this is resolved**, because authoring needs four accepted pairs. That makes
this blocking for go-live, and it blocks Amendment 27's fix from being completed.

Original measurement, as the box warmed at 128 tokens: 1613 -> 1604 -> **1598** MHz,
crossing the floor on the third pair; the gated retry returned 1605. Genuine
sustained throttle on this silicon is **1447-1455 MHz** (operator record
2026-07-12), so every one of these figures — including 1589 — is well over 130 MHz
clear of real throttle.

Both sides run at ~1606 MHz, but dflash is judged against 1500 (106 MHz margin) and
serial against 1600 (4-13 MHz). The 1600 value was inherited from the serial
track's workload; this track's denominator is `dflash-probe`, a width-1 forward
through the DFlash path with the drafter resident.

**Recommended:** `MIN_FREQ_SERIAL=1500`, matching the dflash side, same throttle
record, still ~45 MHz above measured throttle. One line, then
`/opt/bench/gen-manifest.sh` and `/opt/bench/janitor.sh --audit-only`, on every box
serving the track.

Not applied by an agent: the thermal/telemetry stability contract is declared
`readonly` with "do not env-override" and is operator-owned. That has not changed
with the escalation — a throttle floor is exactly the kind of gate an agent must
not relax on its own authority, even with the measurement in hand. Left at 1600 the
track does not merely carry an intermittent false-reject: **no ranked run can
complete pair 1 at the 512-token window**, and the calibration band cannot be
authored, so go-live is blocked on this one line.

## Step D — flip the two trusted-contract fields

The guard requires BOTH, on `main`:

- `fixtures/laguna_xs_2_1_dflash_track.json` -> `official_scoring_enabled: true`
- the same fixture's `reference_baseline.publication_allowed: true`

`benchmark.json` -> `scoring.tokenFidelityGateStatus` was
`proposed-awaiting-operator-signoff` when this step was written and is now
`implemented` (flipped at go-live 2026-07-31, Amendment 31): the full spec is WRITTEN (fixture
`token_fidelity_gate`, rationale and evidence in contract Amendment 29 — every
number traces to a measurement, and the prose is pinned to the enforcing
constants by test). The pinned test
`trackCannotBeEnabledWhileTheFidelityGateIsUnspecified` fails the build if
official scoring is enabled while the status is not `implemented`. Flipping it
to `implemented` is the sign-off — read Amendment 29's "what adopting this
accepts" first, then set it in the same change as the contract fields.

### Evidence for that decision, assembled 2026-07-31 (spec since written — Amendment 29)

The declared gate is
`trusted-sequential-reverification-with-bounded-near-tie-budget`. Each of its
three clauses has an implementation and a live measurement behind it. The spec
text was written the same day (fixture `token_fidelity_gate` + Amendment 29)
and the status advanced to `proposed-awaiting-operator-signoff`; flipping it to
`implemented` remains the go-live judgement and belongs to whoever throws the
switch. The evidence is recorded here so that call does not have to be
re-derived.

| clause | implementation | observed in run `30613617340` |
|---|---|---|
| **trusted** | verifier lives in `Sources/MLXFastTrustedHarness/QwenRuntimeDFlash*.swift`; participant code cannot reach it (it links no MLX/model code) | gate ran inside the trusted parent, candidate confined to the bench sandbox |
| **sequential reverification** | `DFlashReferenceRow.sequentialArgmax` — "reference argmax in the K=1 sequential frame" — plus post-run replay | `reference_checked_row_total: 858`, `verify_block_replayed_round_count: 335` |
| **bounded near-tie budget** | `nearTieBudget` / `residualBudget`, bound `experimentalDFlashNearTieAdmissionBudgetPerThousand = 40`, enforced by the `residualBudgetExhausted` violation | `admissible_near_tie_count: 6`, `residual_divergence_count: 0`, i.e. the budget was exercised and not exhausted |

Supporting: `all_tokens_matched: true` with 512/512 rows admissible (503 exact +
6 near-tie + 3 declared-frame), `work_binding_comparison_count: 1716`,
`rejected_rows_reference_checked: 346`, `max_top2_logit_delta` 1.875 against the
4.875 tolerance.

**What this evidence does NOT settle.** Two questions remain judgement, not
measurement:

1. The near-tie budget of 40/1000 has never been probed near its limit, so the
   bound's *value* is untested even though its *enforcement* is demonstrated.
   Highest observed utilisation: **9 of 21 slots (43%) on the ranked timed
   golden** — the same 9 in three consecutive runs at the shipping
   configuration (K=2, 512 tokens, 2026-07-31), i.e. deterministic frame
   divergence that scales with how flat the material is (6 per 512 on the
   2026-07-30 golden, 9 per 512 on the ranked one). Pool entries must each be
   checked for baseline utilisation headroom — see Amendment 29.
2. Everything above was measured with the candidate being unmodified reference
   code. A ranked run's candidate is adversarially motivated; the gate has never
   faced one. Contract Amendments 18-21 record that this track's gates have
   repeatedly looked sound until first attacked.

Also flip `confirm_track_enabled`'s default to `true` in the workflow so Yukon
dispatches with defaults (DONE at go-live 2026-07-31 — the default is now `true`;
it was `false` while the track was inert, which made Yukon's automated dispatches
unarmable — see Amendment 31).

## Step E — verification dispatch

```
gh workflow run dflash-benchmark.yml --repo Layr-Labs/mlxfast-challenge-dev \
  --ref main -f confirm_track_enabled=true -f run_benchmark=false
```

Correct behaviour BEFORE step D: fails at "Enforce DFlash track enablement" citing
`official_scoring_enabled=false, publication_allowed=false`, everything after
skipped. That is the fail-closed path, verified 2026-07-30.

AFTER step D, the same dispatch should proceed past the guard through the untimed
gates. Only then run with `run_benchmark: true` for a timed measurement.

## Rollback

Set either contract field back to `false`. The guard fails closed on the next
dispatch; nothing needs unwinding on the box. The runner can be parked with
`launchctl unload -w /Library/LaunchDaemons/com.bench.supervisor.plist`.

## Known limits to accept, or fix first

These are documented, measured, and unresolved. Enabling scoring accepts them.

1. **The pool is moderate-difficulty prose, not lowsim** (Amendment 30). The 8
   shipped entries have no-op speedups 0.8726–0.8939 (~75% draft acceptance), so
   a no-op is only ~12% slower than serial K=1 and the raw floor was **0.83**
   (now **0.95** on the per-prompt-normalised score, Amendment 32). This is
   a materially EASIER benchmark than the original `lowsim-prose-v1` intent (that
   single prompt scored 0.5882 no-op at ~34% acceptance, floor would have been
   0.55). The easier pool means the entry bar to rank above a no-op is lower and
   the spread between submissions is compressed. **This is the character being
   shipped, and the operator chose it** when delegating prompt selection; a harder
   benchmark needs curated low-similarity prompts and would lower the floor. The
   tight 2.4% no-op spread does, however, nearly eliminate the cross-prompt
   lottery that a mixed-difficulty pool would create. Per-prompt no-op
   normalisation remains the durable fix if a wider-difficulty pool is ever used.
2. **L2's cross-build tail term is unmeasured.** Every rejected-tail number is
   same-build; cross-build was the larger term on the emitted rows (Amendment 21 §6).
3. **The L2 drift band cannot be closed by tightening** — it is occupied by honest
   cross-build drift, measured over 12,800 comparisons (Amendment 20).
4. **Drafter provenance is enforced by rule plus static review**, not at runtime;
   the exact draft-provenance detector is specified but unbuilt (Amendment 9).
5. **Two DFlash sources sit inside the SERIAL track's editable surface** via
   `benchmark.json`'s `Sources/MLXFastModel` directory entry, so serial submissions
   package them. Moving them changes a live competition's contract and was
   deliberately left alone.
