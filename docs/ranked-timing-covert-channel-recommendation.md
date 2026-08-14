# Recommendation: Quantize the Published Ranking Timings (Backend)

**Status:** recommendation only — not implemented in this repository. It is a
scoring-**backend** publishing-policy change (what precision the platform
exposes to participants), not something the in-repo harness can fully fix.

**Audience:** the Yukon/Darkbloom scoring backend that ingests each ranked
run's `score.json` and audit artifacts and publishes a result to the
participant.

## The channel

Submitted model code participates in inference on hidden prompt tokens and can
make its own execution time **data-dependent** (branch on activations, vary
work per token, etc.). Whatever timing the platform then publishes back to the
participant is a readout the model can drive — a covert channel. Its bandwidth
is set by how many *distinguishable* timing values the participant can observe
per submission, i.e. by the **precision** of the published numbers.

Today the ranked result publishes the following at **full IEEE-754 double
precision**:

- `metrics.decode_seconds_per_token`
- `metrics.prefill_seconds_per_token`
- `metrics.decode_speedup`
- `metrics.prefill_speedup`
- the top-level `score` (`decode_speedup^0.75 * prefill_speedup^0.25`)

This is deliberate in the harness: these are the ranking- and floor-critical
fields, so `ScoreMetrics.withCoarsenedPublicDiagnostics()` in
`Sources/MLXFastCore/Score.swift` explicitly **passes them through unrounded**
while it rounds every *diagnostic* analog field (peak RAM, wall/timed seconds,
TTFT seconds, resident memory, expert read seconds, …) to
`MLXFastConstants.publicDiagnosticSignificantFigures` (2 significant figures).
The carve-out is correct for the harness — scoring, the 0.95 speedup floors,
and `overlay-paired-timing.sh` must operate on full precision — but it means the
in-repo diagnostic coarsening does **not** cover the ranking fields, and a
full-precision seconds-per-token value carries ~15–17 significant decimal
digits (tens of bits) of model-controllable signal per run. Across repeated
submissions that is a wide channel, and it compounds the residual already
acknowledged in
[`docs/private-benchmark-security.md`](private-benchmark-security.md) ("Residual
channel": pass/fail, score, timing, and repeated attempts are a low-bandwidth
covert channel).

The same full-precision copies also travel in the **audit artifact**
`paired-results.json` (staged by the workflow's "Stage audit artifacts" step
from measure-job's `results.json`): `candidate.decode_seconds_per_token`,
`candidate.prefill_seconds_per_token`, the matching `baseline.*`, and the
`paired.*` speedups. Coarsening only `score.json` while `paired-results.json`
keeps full precision would leave the channel open through the audit path.

## Why this is a backend decision

The harness must keep full precision **internally** to compute the score and
evaluate the floors; it cannot both rank accurately and publish coarse numbers
from the same values. The only place that can compute the ranking from full
precision and then expose a coarsened result to the participant is the backend
that owns the published result and the audit-artifact retention policy. The
repository boundary is exactly the ranking-field carve-out in
`withCoarsenedPublicDiagnostics()` — everything downstream of the sealed
`score.json` / `paired-results.json` is backend territory.

## Recommendation

Compute the score, the speedups, and the floor verdicts from **full precision
internally**, and **publish only coarsely-quantized** values to participants:

1. **Bucket the published seconds.** Publish `decode_seconds_per_token` and
   `prefill_seconds_per_token` on a fixed coarse grid — e.g. a fixed relative
   grid (2–3 significant figures, matching the existing diagnostic
   `publicDiagnosticSignificantFigures` policy) or a fixed absolute
   quantum — rather than the raw double.
2. **Round the published speedups and score.** Publish `decode_speedup`,
   `prefill_speedup`, and `score` rounded to a small fixed number of decimals
   (e.g. 2–3), so the participant sees a rank-usable number without a
   high-resolution timing readout. Rank and break ties on the full-precision
   values server-side; publish only the rounded values.
3. **Apply the same quantization to `paired-results.json`** before it is
   exposed to (or downloadable by) the participant, or restrict that audit
   artifact to operators. It currently carries the same full-precision
   candidate/baseline seconds and paired speedups.
4. **Keep the coarsening monotone** so it cannot invert an ordering the scoring
   relies on (the harness already uses `roundedToSignificantFigures`, which is
   monotone and never rounds a positive value to zero — a good model to reuse).
5. **Prefer bucketing over added noise.** Deterministic quantization removes
   low-order model-controllable bits without perturbing the ranking; additive
   noise degrades ranking fidelity and can be averaged out across repeated
   submissions.

Choosing the exact grid is a bandwidth-vs-fairness trade-off (coarser =
narrower channel, but timing improvements below the quantum stop being
distinguishable in the published result). A grid comparable to the 2
significant figures already applied to the diagnostic fields is a reasonable
starting point; the backend can tune it against the observed spread of ranked
scores.

## Not a substitute for

This narrows the published-timing channel; it does not remove the residual that
any pass/fail or ranked feedback is itself a low-bandwidth channel. Competition
policy should still limit repeated private-benchmark attempts and avoid exposing
per-case failure detail for hidden cases, as
[`docs/private-benchmark-security.md`](private-benchmark-security.md) already
notes.
