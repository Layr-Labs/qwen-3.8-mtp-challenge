# Qwen-MTP go-live runbook (`qwen3.6-27b-mtp-v1`)

> **REPO RENAME — 3.6 → 3.8.** This repository is being renamed
> `Layr-Labs/qwen-3.6-mtp-challenge-dev` → `Layr-Labs/qwen-3.8-mtp-challenge-dev`.
> The slugs an operator *dispatches against* below read the NEW name; the
> `qwen-3.6-mtp-challenge-dev` spelling survives only where this file records
> the 2026-08-13 split as a past event. `mlxfast-challenge-dev` is a different,
> still-live repository (DFlash/serial) and is never renamed here. This runbook
> as a whole documents the 3.6 go-live and is HISTORY for the 3.8 track: the
> 3.8 bring-up must re-run it end to end against re-generated artifacts.

> **REPO SPLIT — 2026-08-13.** This track now lives in its own repository:
> `main` of `Layr-Labs/qwen-3.8-mtp-challenge` IS the Qwen-MTP track. The
> source-of-truth history was taken from `Layr-Labs/mlxfast-challenge-dev` at
> `ba5f9703` (branch `qwen36-mtp-track`), and the split is the merge this
> runbook's removal contract referred to:
>
> - the TEMPORARY `qwen36-mtp-track` ref allowlist arm was REMOVED here, in the
>   ranked workflow and in both `enforce-trusted-qwen-mtp-*` guard scripts;
>   those guards are `refs/heads/main`-only again, and the ranked allowlist is
>   `main`, `submissions/*`, `baseline/*`, `yukon/baseline/*`;
> - `MLXFAST_QWEN_MTP_TRUSTED_REPOSITORY` and both guard scripts' trusted
>   repository now name `Layr-Labs/qwen-3.8-mtp-challenge`;
> - `benchmark.json` IS the Qwen-MTP manifest (byte-identical to
>   `benchmark.qwen-mtp.json`, which is kept because the workflow's
>   `MLXFAST_QWEN_MTP_EDITABLE_SURFACE_CONTRACT` and `QwenMTPTrackNamingTests`
>   pin that filename).
>
> Content-addressed identities were deliberately NOT renamed and still read
> `mlxfast-challenge-dev-qwen-mtp` (manifest `name`) / `qwen3.6-27b-mtp-v1`
> (track id, leaderboard namespace): they were chosen to survive this move.
> Anything below that speaks of a deferred merge to `main` is HISTORY.

`.github/workflows/qwen-mtp-provision-goldens.yml` points operators here from
four places, including "step A" and "step B" cited in error messages an operator
only ever sees when a dispatch fails closed. Until go-live the reference dangled
at nothing. This is it.

Everything below is work that cannot be done from the repo alone: it needs hidden
material, physical access to the serving box, or a judgement about whether the
anti-cheat is sound enough to rank on.

> **GO-LIVE COMPLETE — 2026-08-13.** Steps A–E below were EXECUTED. This runbook
> is retained as the record and as the procedure for taking the track back
> offline. **Scoring was re-anchored at serial = 1.0 on 2026-08-14** and the
> editable surface was opened to the whole speculative apparatus; the two
> changes are one contract change and are described in "Scoring semantics"
> below and in `docs/qwen-mtp-editable-surface.md`.
> Current live state: `official_scoring_enabled: true`,
> `reference_baseline.publication_allowed: true`,
> `tokenFidelityGateStatus: "implemented"`,
> `MLXFAST_QWEN_MTP_CALIBRATION_READY: "1"`, raw decode floor `0.90`, published
> ceiling `3.0`.
>
> **The merge to `main` is DONE, as a repo split** (see the note at the top of
> this file). Everything landed on the `qwen36-mtp-track` ref of
> `mlxfast-challenge-dev` and became `main` of `qwen-3.6-mtp-challenge-dev` at
> `ba5f9703`; the ranked workflow's TEMPORARY `qwen36-mtp-track` allowlist arm
> and the matching entries in the two `enforce-trusted-qwen-mtp-*` guard
> scripts were removed there. Pre-go-live "pending / false / inert" phrasing
> elsewhere in the tree is HISTORY.

## 0. What is already done and verified (do not redo)

Measured on `m5-max-128gb-3` ("box 3", Apple M5 Max), runner label
`m5-qwen38-27b-mtp`. Track contract in
`fixtures/qwen3_6_27b_mtp_track.json`; deploy detail in the operator repo's
`m5-machine-scripts/deploy/qwen36-mtp/DEPLOY-RUNBOOK.md`.

| thing | state |
|---|---|
| pinned baseline | `/opt/bench-runner/baseline/qwen3.6-27b-mtp-v1/current`, installed and calibrated from `e86655528c2abe534aa2e61a454565237801aa2d`. Pinned by PATH and by the signed §9d manifest — **not** by any `MLXFAST_BASELINE_COMMIT` workflow variable. That serial-era mechanism has no reader on this track and was deliberately not introduced. |
| baseline calibration | `/opt/bench-runner/state/qwen3.6-27b-mtp-v1/baseline-calibration.json`, paired schema, target `lowsim-prose-qwen-v1`. Bands the **serial denominator** of the pair (depth 0, same binary, same worker), not the baseline's absolute speed. `decode_tokens: 512` is mandatory and does not inherit: seconds/token moves ~2.1x between a 16- and a 1024-token window on this model, so a band without its window is a band against nothing. |
| MTP head | separately pinned tree, `mlx-community/Qwen3.6-27B-MTP-4bit @ 83795d54`, 8 files / 258490810 B, resident on **both** legs of the pair so its residency cost is charged to the denominator too. Only the drafting differs. |
| depth | candidate depth **2**; serial control depth **0** (MTP off — not 1; depth 1 still drafts and accepts ~70%, which understates the gain). |
| oracle cache | pinned **as empty**. Any file appearing in `state/qwen3.6-27b-mtp-v1/oracle-cache/` is drift. Correctness is the hidden golden, re-audited into `parity_all_ok`. |
| janitor audit | clean; box signed and serving. |

## Step A — the hidden timed-prompt pool (DONE)

`fixtures/qwen3_6_27b_mtp_track.json` → `timed_prompt_pool` holds **8 distinct
entries**, one per domain (beagle, botany, drama, essays, medicine, plutarch,
republic, travel), each pinned by `sha256` and `bytes`, each uploaded to R2 under
`correctness_prompts/qwen3.6-27b-mtp-v1/` and re-verified by digest after
download.

Each entry **is** a timed MTP reference-rows golden — the object the ranked job
hands to `mtp-timed --golden` — not the prose prompt it was seeded from. Each was
generated on box 3 with `mtp-verify --emitted <plan> --generate 513` from a
different 512-token prose seed, and each returned
`reference_self_consistent=true` with 513 rows.

A seed plan must be **pre-tokenized real prose**. Never `--seed-generate`: that
produces the degenerate greedy self-continuation the correctness contract
condemns, and the provisioning job refuses argv containing the flag.

### Why the pool exists, and why it must be varied

Acceptance is strongly prompt-dependent (measured 0.344–0.585 across the pool),
so the paired ratio spans **0.7623** (drama) to **1.0845** (medicine) — a 1.42x
span for *identical code*. Sampling one prompt per run would hand a candidate
drawn on `medicine` a 1.42x advantage over one drawn on `drama`. A ranked run
therefore times **all eight** and publishes the median, which cancels the draw
by construction.

Those eight numbers are still pinned in the fixture as `noop_decode_speedup`,
and reading what they now mean is worth thirty seconds: each is the *shipped
depth-2 configuration's own measured performance against true serial decode on
that prompt*. They used to be divisors. They are now the reference tree's
regression profile, published beside each submission's own ratio.

### Scoring semantics: serial = 1.0 (operator-ratified 2026-08-14)

A ranked run **times the whole pool**, not one sampled prompt. Formally it draws
one prompt per *collection*; each of the 8 collections is a singleton today, so
all 8 run.

```
per prompt p:  raw_p = mean(serial depth-0 s/tok over p's pairs)
                     / mean(candidate    s/tok over p's pairs)
published:     score = median(raw_1 .. raw_8)
```

That is the whole formula. **There is no normalisation step.** Both means come
from the same thermally-gated session for that prompt, in alternating order, on
the same box — so the serial leg *is* the normaliser, and it is measured rather
than pinned.

**Why the anchor moved.** The retired rule divided each prompt by its pinned
`noop_decode_speedup`, which made an *unmodified candidate* score exactly 1.0 by
construction. Since those references are the shipped depth-2 configuration's own
measurements, the rule declared the reference tree to be the zero point *by
definition rather than by measurement* — and the reference tree is not the zero
point. At the 512-token window it is a **~6.5% regression** against true serial
decode (raw median 0.935). Anchoring at serial makes 1.0 mean serial decode,
above 1.0 a real speedup, and the shipped configuration a starting position with
work in front of it.

It is also what makes the **opened speculative surface** safe. Head weights,
drafting code and the per-round draft schedule are competitive surface now
(`docs/qwen-mtp-editable-surface.md`). A candidate that stops drafting
degenerates to the serial control and scores 1.0 — so "turn speculation off" is
an honest, unremarkable submission rather than an exploit against a normalising
reference the candidate no longer resembles. That is the same reason the trusted
driver could stop refusing non-drafting rounds.

**The floor moved to `0.90`, and it had to.** The family convention is 0.95 and
this track cannot use it. Yukon's import dispatches a **baseline run of the
source tree** and requires a *published* score to seed `currentBestScore`; with
no score the benchmark never opens and the track accepts no submissions at all.
The stock tree medians ~0.935 raw — a measured ~6.5% regression against true
serial decode, which is precisely the fact serial anchoring exists to stop
hiding — so a 0.95 floor rejects the one run the whole track bootstraps from.
The 0.95 convention was inherited from a *normalised* score where an unmodified
tree sat at 1.0 by construction; it does not survive the change of anchor.

Operator decision, 2026-08-14: **`0.90` — "do not regress serial by more than
10%".** It is a regression bound, not a quality bar. A submission may be slower
than serial and the board will say so; it may not be much slower. A candidate
that genuinely cannot beat serial should stop drafting and take 1.0, which is
legal, costs nothing and is strictly better than publishing 0.92.

Three consequences, stated so nobody rediscovers them:

- **The stock tree publishes 0.935 and holds the board at −6.5%.** The board's
  opening number is a regression, on purpose.
- **Calibration dispatches publish normally.** 0.935 clears 0.90 with 0.035 of
  margin, so a calibration run is read off a green run — there is no
  read-the-median-out-of-a-sealed-failure workaround any more.
- **k3-class regressions (~0.777) still floor-fail.** That is the intended
  terminal state for a submission that is simply slower.

**`3.0` is new** — until 2026-08-14 the wrapper bounded per-*pair* ratios at 5.0
and nothing at all bounded the aggregate that reached the leaderboard, so an
implausible published median could only be caught by a human reading it. Both
bounds are enforced twice, independently, by the workflow and by the wrapper.

**The median rule is load-bearing because 8 is even:** the score is the mean of
the two central order statistics, *not* the lower-median rule used by the
per-pair `mtp_decode_speedup_median` diagnostic and by the CLI's own p50 fields.
`results.json` and `score.json` both name the rule they used.

**Pair budget is per prompt.** The ranked default is k=1 — 8 prompts × 1 pair =
8 pairs / 16 timed phases, roughly 45 min of timed work. The old run-level 3/4
budget bought pair-averaging on a single prompt; the median over 8 prompts does
that job and also cancels the lottery. At the measured 0.77% pair noise, k=1
puts the score's dispersion near 0.34%. Raise it with the wrapper's
`--pairs-per-prompt` if calibration data asks.

**The serial denominator band is unchanged and still load-bearing, and it is
pooled across prompts.** Re-anchoring the score did not touch it, and should
not: the band guards the *denominator*, the calibration expectation guards the
*score*, and they are different questions. A band on the denominator is what
stops a slow box from inflating every ratio at once.
 The serial control is
depth 0 — 512 plain forwards — and is prompt-invariant by construction, since
the measured pool spread lives entirely in the acceptance rate, which only the
numerator sees. The band therefore checks the mean of *all* serial pairs of
*all* prompts against the one top-level calibration: at 8+ pooled pairs that is
a strictly stronger test than the 4-pair band it replaces, and no re-authoring
of the installed calibration is required. Per-prompt serial means stay in the
sealed breakdown, so a denominator that began to move with the prompt would be
visible.

### Why each prompt-window still pays its own process

Batching all 8 prompts into one model residency per leg would be faster. It is
**not available**, and the reasons are worth recording because they also define
what taking that path would cost:

- the worker builds exactly one `Qwen36MTPBlockSession` before the protocol
  hello and guards `mtp_decode_begin` with `!state.began`; there is no reset
  request kind anywhere in the tree (`QwenRuntimeMTPWorker.swift`);
- the parent driver takes one golden per call and spawns a **second** worker
  afterwards for the post-window reference replay
  (`QwenRuntimeMTPDriver.swift`), so a batched leg would have to batch the audit
  replay too;
- the CLI's option parser rejects a repeated `--golden` outright ("duplicate
  option") — the batched argv shape is a hard error, not last-wins;
- **decisively**, the serial leg executes the *pinned baseline tree's* own
  prebuilt `mlxfast-swift` and its sibling worker, so no repo-side protocol
  change reaches it. Batching needs a new baseline build, a new signed §9d
  manifest and a full re-calibration;
- and it would change the measured quantity anyway: only the first
  prompt-window of a batched leg carries the cold working set, while the
  installed band was authored against per-window process start + allocator
  clear + warm + a seed prefill charged *inside* the clock.

**One consequence for the stall guard**, recorded because it constrains that
future path: under batching the post-prefill warmup belongs to the *leg*, not to
each window, so `STALL_EXCLUDE_FIRST_BLOCK` would have to become leg-aware —
exclude `block[0]` of window 1 only. Excluding `block[0]` of every window opens
8 blind spots at exactly the window boundaries where a stall is most likely;
including `block[0]` of window 1 reinstates the 5.72–5.90x false rejection the
calibration removed. Either way the CLI report would have to declare its
position within the leg. On the per-invocation shape kept here the question does
not arise: every window *is* a leg, and the guard is unchanged.

Batched legs remain a documented future path. Taking it is a measurement-
architecture change, not a scoring change.

## Step B — freeze and pin the hidden goldens (DONE)

Three classes of hidden object, all content-addressed, all pinned by digest AND
byte count in the ranked workflow's job env, with the R2 object key embedding the
same digest so key and digest cannot drift apart:

- **The hidden MTP correctness golden** — `MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_*`.
  Generated on box 3 against the Qwen tower with the pinned head, from the hidden
  correctness prompt's own 512 seed tokens; re-verified by a
  `mtp-verify --tokens 512 --mtp-depth 2` pass returning
  `all_tokens_matched=true`, `parity_all_ok=true` and a CLOSED row ledger.
- **The reused serial correctness golden** — `MLXFAST_RAW_CORRECTNESS_GOLDEN_*`,
  pinned at `faf1a679`. It carries the derived `.benchmark` oracle the ranked
  gates phase requires; a golden without one fails closed at "Correctness and
  gates" with `benchmark golden file must contain a benchmark oracle`.
- **The GPQA reference cases** — `MLXFAST_GPQA_REFERENCE_*`, pinned at
  `c8bce79c`, with `accepted_responses` filled from the reference model's own
  captures. See "Known limits" below; this one has a real caveat.

Upload happens **after** the digest is pinned, never before. A dispatch that
reaches the download step with an unresolvable key is the intended fail-closed
ordering.

## Step C — calibration, the floor and the ceiling (DONE; re-anchored 2026-08-14)

The serial denominator was banded from gated bootstrap sessions on the parked
box, at the ranked window (`--tokens 512`, `--mtp-depth 2`), in the thermal/fan
regime the track serves in. `GPU_LOADED_UTIL` is deliberately left at the
wrapper's own `0.70` default: the serial leg's GPU-util median is 0.7935 and
never sustains 0.9, so the inherited 0.95 starves `MIN_LOADED_SAMPLES` and
false-rejects honest runs. `TELEM_INTERVAL_MS=100` is likewise not optional on
box 3.

The decode floor is **0.90** and the published-median ceiling is **3.0**, each
pinned equal in the ranked workflow env
(`MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR` / `..._CEILING`, the enforcing site),
in `benchmark.qwen-mtp.json` → `scoring.decodeSpeedupFloor` /
`scoring.decodeSpeedupCeiling` (documentation), and in the box wrapper
(`MAX_PLAUSIBLE_PUBLISHED_SPEEDUP`, the second independent enforcing site).
`QwenMTPTrackNamingTests` pins them together.

**Prefill is measured but UNSCORED on this track.** `mtp_decode_speedup` is a
decode-only ratio-of-means; the seed prefill is charged *inside* the decode
window, identically on both legs; and the measurement wrapper seals
`prefill_component: "none"` in its `results.json`. A prefill figure is recorded
for historical tracking only. A reader who finds the prefill constant and assumes
it participates in scoring will mis-tune the track.

**The floor applies to the RAW MEDIAN** (see "Scoring semantics" under step A),
and both the number and its meaning moved on 2026-08-14. Under the retired
normalised rule an unmodified candidate medianed to 1.0 by construction and 0.95
was a 5% margin below parity that nothing honest could ever hit. Under serial
anchoring an unmodified candidate medians to ~0.935, so 0.95 would have rejected
the source-tree baseline run Yukon needs to open the benchmark; the floor is
**0.90**, a "do not regress serial by more than 10%" bound. A ceiling of **3.0**
was added at the same time, applied to the same published number.

### The constants landed at go-live

Both are **track-scoped** (`qwenMTP…`-prefixed, in the Qwen section of
`Sources/MLXFastCore/Constants.swift`). The unprefixed `officialBaseline*`
constants are the Poolside/Laguna serial calibration that the **live DFlash
track** compiles against; the decode figure here is 2.74x larger, so overwriting
the shared constant would silently retune another live track's acceptance-band
reference. They are quoted here character-for-character:

- `qwenMTPOfficialBaselineDecodeSecondsPerToken = 0.037994794617407023`
- `qwenMTPOfficialBaselinePrefillSecondsPerToken = 0.00115714`
- `qwenMTPSemanticGPQAMinPassCount = 8`

**Decode is SCORED** — it is the serial denominator of the paired ratio, and the
same value the on-box band checks against. **Prefill is UNSCORED**: this track's
score has no prefill component at all. `mtp_decode_speedup` is a decode-only
ratio-of-means, the seed prefill is charged *inside* the decode window
identically on both legs, and the wrapper seals `prefill_component: "none"` in
the `results.json` it signs. The prefill figure is retained for historical
tracking only.

Prefill is also deliberately **not** wired into the local-mode estimate. The
Qwen-MTP local path consumes no pinned baseline constant: it times both legs in
the same session and reports that ratio. There is no seam to redirect, and
adding one would replace a self-normalising measurement with a
hardware-absolute one — changing what the local number means rather than
improving it.

The GPQA floor is `min(observed) − 1` over the four go-live calibration
dispatches, each of which judged 9/9. Its honest limitation is recorded under
"Known limits" below and in the constant's own doc comment: it cannot reject a
constant-"A" answerer, and raising it to 9 would not fix that.

### The CALIBRATION acceptance window (derived 2026-08-13, not inherited)

This is the operator's own accept/reject band for **calibration dispatches** —
runs of a known baseline-equivalent candidate. It is **not** a submission-facing
gate: the 0.95 floor, the 3.0 ceiling, the `[0.95,1.05]` serial band and
`MAX_PLAUSIBLE_SPEEDUP` are untouched by anything in this section.

> **Read this section as a DERIVATION OF DISPERSION, not of a centre.** Every
> number below — 0.142%, 0.120%, 0.135%, the ±0.8% window — is about how much a
> repeated measurement of the same tree *scatters*, and scatter is invariant to
> the anchor: dividing every observation by a per-prompt constant shifts the
> centre and leaves the spread alone. So the whole derivation survives the
> 2026-08-14 re-anchoring intact. What does NOT survive is the *centre*: these
> runs were checked against parity because the normalised rule put an unmodified
> tree at 1.0 by construction. Under serial anchoring the centre is a MEASURED
> quantity, and it is ~0.935. See "Re-deriving the calibration expectation".

The inherited serial-era ±1% was never derived for this track and is wrong for
it in both directions. The window below is derived from measured variance.

**Inputs.** Three prompts have two *independent sessions* each (a ranked dispatch
and a re-measurement session): beagle +0.228%, drama −0.261%, medicine +0.023%.
The sd of a single session estimate is
`sqrt(mean(d²)/2)` = **0.142%**.

**Decomposition.** Typical within-run per-pair ratio sd is 0.150%, so a 4-pair
mean carries 0.075% from pair noise alone. The residual
`sqrt(0.142² − 0.075²)` = **0.120%** is *session-level* — thermal and frequency
state that every prompt in a run shares. It does not average down with more
pairs, which is why `pairs_per_prompt` beyond about 2 buys very little.

**Propagating to the median of 8.** The session term is **common-mode**: a ranked
run times all 8 prompts in one thermal session, so it shifts every prompt
together and passes straight through the median. Only the independent per-prompt
term is reduced (≈0.443× for the even-n mean-of-two-central rule at n=8). At the
ranked `pairs_per_prompt: 1`:

```
sd(median) = sqrt( 0.120²  +  (0.443 × 0.150)² )  =  0.135%
```

Monte Carlo over 200k trials with the same inputs agrees: **0.135%**. Raising k
to 4 only reaches 0.124% — confirming the common-mode term dominates.

**The window.** ±3σ on the point estimate is [0.9959, 1.0041]. That estimate
rests on only three paired comparisons, so σ is inflated by 1.8× (roughly the
upper 95% χ² bound at ~3 df) before rounding outward:

> **Calibration acceptance window: 0.992 – 1.008** (parity ±0.8%).

A calibration run landing outside it is a stop-and-investigate, not a retry.
**Re-derive it** once more paired sessions exist — the 1.8× inflation is
compensating for a thin sample and should shrink, not persist.

For contrast, the same inputs give a *single-draw* window of ±0.58%, and that is
before per-prompt reference error, which is what the old single-shot references
contributed and what made a 1.1% miss look normal. Median-of-8 is both tighter
and robust to one bad reference.

**CONFIRMED against the four go-live calibration dispatches** (31712368539,
31715555814, 31718615518, 31721547429), which published medians of
1.0012383415613857, 1.0004179605801127, 0.9990292566514924 and
1.001237773084561:

| quantity | value |
|---|---|
| mean of the four medians | 1.00048 |
| observed `sd(median)` | **0.104%** |
| worst single deviation from parity | 0.124% |
| declared window in observed sigmas | ±7.7σ |

The a-priori model predicted 0.135–0.188%; the measured 0.104% is slightly
*tighter*, so the derivation was sound and conservative rather than optimistic.
An empirical ±3σ would be [0.99687, 1.00313], or [0.99437, 1.00563] keeping the
1.8× inflation.

**Recommendation, not applied here:** on this evidence the window could tighten
to roughly **0.994 – 1.006** and still sit at ~5.8σ. It is deliberately left at
0.992–1.008 for now — four medians is three degrees of freedom, and the value of
a calibration band is that it does not move every time new data arrives. Revisit
after the next four ranked runs.

### Re-deriving the calibration expectation under serial anchoring

**Status: the procedure is written; the runs are not done.** The runner for this
repository is not registered, so no dispatch has executed under the 2026-08-14
semantics and no number in this section was measured under them. What is in the
fixture today is *seeded* and says so:

```
fixtures/qwen3_6_27b_mtp_track.json → calibration
  status                 "seeded_from_k_matrix_pending_post_cutover_remeasurement"
  expected_raw_median    0.9350424303
  band_pct               2.0
  expected_raw_median_provenance
                         the eight per-prompt raw ratios of the k2 dispatch,
                         the median rule applied to them, and the normalised
                         median that same run published (0.9998793467)
```

**Where the seed comes from.** k2 was the reference variant of the k-matrix: a
comment-only diff inside the editable surface over the installed baseline tree
`e86655528c2abe534aa2e61a454565237801aa2d`, i.e. an unmodified tree in every way
that matters. Its sealed `per_prompt` breakdown already carried each prompt's
RAW ratio-of-means, because the raw figure was reported alongside the normalised
one even when only the normalised one scored. Re-medianing those eight raw
numbers under the even-n rule gives 0.9350424303. No new measurement was
invented, and none could be. `theSeededCalibrationExpectationMatchesItsRecorded
Provenance` recomputes the median from the recorded inputs on every test run, so
the value cannot drift away from the derivation that justifies it.

**Why 2.0% and not the 0.8% window above.** The ±0.8% window bands a *repeated
measurement of a known centre*; this bands a *seeded* centre that has not yet
been confirmed on the box under these semantics. 2% is ~6σ on the measured
0.135% dispersion — loose enough not to false-reject on a seed that turns out to
be 0.5% off, tight enough to catch a regime shift like the +6.58%
thermal/frequency delta that invalidated the 6a3bee9-era references. Tighten it
to the dispersion-derived width once the runs below exist.

**The procedure** (run post-cutover, after the runner registers; mirrors the
§7 sessions that produced the 0.992–1.008 window):

1. **Confirm the tree is baseline-equivalent.** Dispatch on a `baseline/*` ref
   whose only delta from the installed baseline tree is inert (a comment). The
   point of a calibration run is that the *candidate* is not a variable.
2. **Four dispatches, not one**, spaced so each gets its own thermal session.
   Four is what the 2026-08-13 window rested on and it is the minimum that gives
   a usable dispersion estimate; fewer means the σ you compute is mostly noise.
3. **Record the raw median and all eight per-prompt raw ratios from each run.**
   The per-prompt numbers are what let you tell a *uniform* shift (a regime
   change, which passes through the median untouched) from a *per-prompt* one
   (an acceptance-profile change, which does not). The 2026-08-13
   re-measurement found the shift was NOT coherent — 6 entries up, 2 down — and
   that non-uniformity is the whole reason per-entry data is required.
4. **Expect green runs.** A baseline-equivalent tree medians ~0.935, which
   clears the 0.90 floor with 0.035 of margin, so each dispatch publishes its
   score normally and the median is read straight off `score.json`. (This step
   previously said the opposite: at the originally-proposed 0.95 floor these
   runs would all have been refused and the median had to be dug out of a
   sealed failure artifact. Moving the floor to 0.90 removed that workaround
   along with the launch blocker behind it.)
5. **Set `expected_raw_median` to the mean of the four medians**, and
   `band_pct` to ±3σ inflated by the same 1.8× thin-sample factor the
   0.992–1.008 derivation used, rounded outward. Update
   `MLXFAST_QWEN_MTP_CALIBRATION_EXPECTED_RAW_MEDIAN` and
   `MLXFAST_QWEN_MTP_CALIBRATION_BAND_PCT` in the ranked workflow to match, and
   flip `calibration.status` to a measured value naming the four run ids.
6. **Do not touch the top-level serial band while doing this.** It guards the
   denominator and it is confirmed at ratios 1.001674 / 1.001795 / 1.000355 /
   1.001264. If the serial band starts failing during calibration, the box moved
   and the calibration is invalid — stop rather than re-authoring the band to
   fit.
7. **Re-confirm the GPQA floor** in the same four runs, exactly as §7.4 did.

One caveat worth carrying: the four dispatches ran back-to-back inside a single
afternoon on one box. They therefore sample *within-day* thermal variation well
and *across-day* drift not at all. The window should be re-checked, not assumed,
after the first ranked runs on a different day.

## Step D — flip the two trusted-contract fields

The switch itself. In **one** commit, because the tests and docs are wired to
fail otherwise, and that friction is deliberate in both directions:

1. `fixtures/qwen3_6_27b_mtp_track.json`: `official_scoring_enabled` → `true`
   and `reference_baseline.publication_allowed` → `true`. The enablement guard
   refuses a RANKED dispatch while **either** is false, and a gates-only dry run
   publishes no score.
2. `.github/workflows/qwen-mtp-ranked-benchmark.yml`:
   `MLXFAST_QWEN_MTP_CALIBRATION_READY` → `"1"`. A hard-pinned literal, never an
   expression and never a dispatch input. It gates only TIMED runs, so
   `run_benchmark=false` correctness dispatches keep working while it is closed —
   that is the migration shape.
3. `benchmark.qwen-mtp.json`: `tokenFidelityGateStatus` → `"implemented"` and
   `scoring.decodeSpeedupFloor` → the enforced floor.
4. `Tests/MLXFastTests/QwenMTPTrackNamingTests.swift`: the armed-posture
   assertions.

Taking the track back OFFLINE means editing all four again in one commit.

## Step E — verification dispatch

`gh workflow run qwen-mtp-ranked-benchmark.yml --ref qwen36-mtp-track -f run_benchmark=true`

Runs serialise on the single `m5-qwen38-27b-mtp` runner regardless of
concurrency group, so dispatch **sequentially** rather than stacking the queue.
Between runs, confirm the quarantine flag is absent.

What actually gates a ranked run — budget effort accordingly. The in-harness
acceptance band does **not**: the gates pass with `MLXFAST_BENCHMARK_SKIP_TIMED=1`
(ratio exactly 1.0), and the measure wrapper explicitly tolerates
`acceptance band failed*` / `performance floor failed*`. The real guardrails are
the speedup floors in `overlay-paired-timing.sh`, `baseline_band_check` against
the on-box calibration, `plausibility_check`, and `MAX_PLAUSIBLE_SPEEDUP`.

## Rollback

Revert the Step-D commit; the old pins, fixtures and `editablePaths` come back
intact. Box-side rollback is
`sudo ./install-qwen-mtp-track.sh --execute --rollback` in the operator repo, and
deliberately does **not** remove the R2 uploads (immutable, content-addressed),
the `~gaj` staging copies, or the physical baseline tree.

If the box quarantined, read `/opt/bench/quarantine.flag.drift` for the full
`< baseline / > live` diff, fix the cause, then
`sudo /opt/bench/janitor.sh --clear-quarantine`. **Never reboot a box as a
troubleshooting step** — an OS update once reset `/etc/pf.conf`'s anchor lines
and the box served with open bench egress. Park, inspect, fix, re-sign.

## Known limits to accept, or fix first

**The semantic GPQA gate cannot reject a constant-"A" answerer.** Every
`answer_key` in the hidden GPQA fixture is `"A"` while each prompt presents four
options, and the reference model is measurably position-biased toward option A —
it tracks the correct option only ~2/9 when option order is rotated. Since
`accepted_responses` was filled from the reference model's own captures, the gate
now measures **fidelity to the reference implementation**, not question-answering
accuracy. 8 of the 9 reference captures are `"A"`, so a constant-"A" answerer
scores exactly 8 of 9 — which means a min-pass floor of 8 does **not** reject it.
Raising the floor to 9 is not the fix either: it would delete the error budget
that absorbs judge nondeterminism. **The real fix is shuffling option order,
which is organizer material and is deliberately not done in this repo.** Tracked
operator-side.

**The stall guardrail's first block is warmup, not a stall.** As measured,
`max_block_request_seconds` is the first block after the seed prefill and is flat
to within 1% across 8/32/128/512-round windows (serial ~0.19 s, depth-2
~0.267 s). At the 512 window the serial leg therefore trips the 4x rule
(measured 5.72–5.90x) while the depth-2 leg does not (3.30–3.36x). The guardrail
should exclude the first block.

**The pool's character is provisional.** Acceptance 0.344–0.585 on moderate prose;
a harder low-similarity pool would move the numbers. Re-derive the depth and the
floor across the WHOLE pool if the pool changes — a depth tuned to one prompt is
the same trap as a floor tuned to one prompt.

**The track identity is not formally ratified.** `qwen3.6-27b-mtp-v1` /
`mlxfast-challenge-dev-qwen-mtp` / `lowsim-prose-qwen-v1` are in use and embedded
in R2 object keys and goldens, and those uploads are content-addressed and
irreversible.

---

## The 2026-08-14 proof matrix (built, NOT dispatched)

Five branches exist on this repository to demonstrate the contract change
mechanically. **None has been dispatched**: the runner for this repository is
not registered, so every number below is a *prediction recomputed from sealed
per-prompt data of the pre-cutover dispatches*, not a measurement under the new
semantics.

### Predicted scores

Every prediction is the even-n median of that arm's OWN eight sealed per-prompt
**raw** ratios. Those raws were always reported alongside the normalised ones,
which is why no measurement had to be invented.

| arm | change | predicted raw median | vs serial | published under the retired rule | verdict |
|---|---|---|---|---|---|
| **k0** | `draftPolicy → 0` (draft nothing) | ~1.0000 | ±0% | *no score* — refused at gate | **PASS**, and it is the floor of honest play |
| **k1** | `draftPolicy → 1` | **1.0541081649** | +5.4% | 1.1387 (normalised) | **PASS** — best in matrix |
| **k2** | comment only (reference tree) | **0.9350424303** | −6.5% | 0.9999 (normalised) | **PASS** at the 0.90 floor — and it is what seeds the board |
| **k3** | `draftPolicy → 3` | **0.7768609425** | −22.3% | 0.8339 (normalised) | **REJECTED**, as before |
| **mixed-surface-test** | loop-hoist + `draftPolicy → 1` | ≥ k1 | ≥ +5.4% | n/a | expected PASS |

Four things in that table are worth reading twice.

**k0 inverted.** It was the negative control — the arm that existed to prove the
drafts-empty guard fired. It is now the lower-bound control: a non-drafting
candidate is the serial control and takes 1.0. Nothing honest should ever score
below it.

**k2, the reference tree, now publishes a regression instead of a 1.000.** That
is the re-anchoring stated as a consequence rather than as a principle: the
shipped depth-2 configuration is 6.5% slower than serial decode, the old rule
scored it 1.000 by dividing it by itself, and the new rule prices it. It also
seeds the board — Yukon's import dispatches exactly this shape — which is why
the floor is 0.90 and not the family's 0.95.

**k1's honest number is +5.4%, not +8%.** The +8% figure predates re-medianing
the per-prompt raws and is superseded. The retired rule's 1.1387 was larger
still because it credited k1 with beating the slow reference tree *as well as*
with beating serial.

**k3 changed category.** Depth 3 used to be unreachable from the editable
surface, so measuring it required editing trusted workflow env — an
operator-shaped branch. It is now a one-line policy change that passes surface
enforcement unmodified. Its verdict did not change; what changed is that
−22.3% now says *slower than serial decode* rather than *slower than an
already-slow reference*.

**Independent corroboration of the k2 seed.** Median the eight pinned
`noop_decode_speedup` values directly — they are the reference configuration's
own per-prompt measurements, taken in separate sessions from k2 — and you get
**0.93515** against k2's **0.93504**. Two independent measurement campaigns of
the same tree agreeing to 0.01% is the strongest evidence available that the
seeded calibration centre is right.

### Local verification already done (no box required)

- **Surface enforcement** admits all five branches against `benchmark.json`
  read from the base commit. k3 passing is the notable one: it is
  submission-shaped now.
- **Overlay** on `submissions/mixed-surface-test` reproduces exactly two
  modified files and leaves `Sources/MLXFastTrustedHarness`, `Sources/MLXFastCLI`,
  `Sources/MLXFastCore`, `.github/` and the manifests byte-identical.
- **Static review (native-MTP arm)** returns `passed=true severity=none` on the
  mixed-surface branch, with the qwen-mtp policy selected, both hunks present in
  the reviewed diff, `mtp-head` correctly excluded from the payload as
  byte-budget exempt, and surface growth 1,383 bytes against a 262,144 limit.

### Dispatch command list — POST-REGISTRATION ONLY

Do not run any of these until the `m5-qwen38-27b-mtp` runner is registered
against this repository **and** the wrapper has been re-signed under a §9.4
window (its sha256 changed; see the report). Ranked dispatch also requires the
workflow file to exist on the default branch, which it now does.

```sh
R=Layr-Labs/qwen-3.8-mtp-challenge
W=qwen-mtp-ranked-benchmark.yml

# 0. Gates-only smoke on the new tip FIRST. Proves the head-resolution step,
#    the provenance gate and the depth-provenance conjuncts before any gated
#    thermal time is spent.
gh workflow run "$W" -R "$R" --ref main -f run_benchmark=false

# 1. Reference / calibration arm. Expect ~0.935 and a PASS (0.90 floor). This
#    is the shape Yukon's import uses to seed currentBestScore, so it is also
#    the run that proves the benchmark can open at all.
gh workflow run "$W" -R "$R" --ref baseline/ktest-k2-4c02beb

# 2. The lower-bound control. Expect ~1.0 and a PASS.
gh workflow run "$W" -R "$R" --ref baseline/ktest-k0-4c02beb

# 3. Best-in-matrix. Expect ~1.054 and a PASS.
gh workflow run "$W" -R "$R" --ref baseline/ktest-k1-4c02beb

# 4. Depth 3. Expect ~0.777 and a floor rejection.
gh workflow run "$W" -R "$R" --ref baseline/ktest-k3-4c02beb

# 5. The union proof: both halves of the opened surface in one submission.
#    This one is submissions/*, so it exercises enforcement, overlay and the
#    static-review arm on the real box.
gh workflow run "$W" -R "$R" --ref submissions/mixed-surface-test

# 6. Calibration re-derivation: four spaced dispatches of the k2 arm, per
#    "Re-deriving the calibration expectation" above. Space them so each gets
#    its own thermal session.
```

Run them **in that order**. 0 before anything gated; k2 before the rest, because
if the reference arm does not land near 0.935 the calibration seed is wrong and
every other prediction is being read against the wrong centre.
