# Timing Variance in Timed Benchmarks (tenki M4 Pro) — Investigation & Recommendation

> **HISTORICAL RUNNER NOTE (2026-07-11).** This investigation analyzed the
> retired Blacksmith/tenki M4 Pro *VM* runner class. Ranked runs have since
> moved to a single self-hosted Apple M5 Max (`m5-bench`), where the timed
> windows run last, pinned baseline then candidate back to back, each behind a
> fixed 40C thermal gate with telemetry-validated acceptance (see
> `.github/workflows/benchmark.yml` — the serial ranked pipeline, formerly
> named `serial-benchmark.yml` — and `benchmark-window-freeze.md`). The
> analysis below is kept as the measurement record behind the per-axis
> acceptance bands (`AcceptanceBand`, the `*BandTolerance` constants), which
> remain in force; the runner-class specifics below no longer describe the
> ranked box.

> **FINAL root cause (fresh-VM control, run 28893815980): the dominant ~2–3×
> decode variance was a CONTINUOUS-RUNNING ARTIFACT of the back-to-back probing,
> NOT the real ranked path.** On the real cadence — one full `./benchmark.sh` per
> fresh throwaway VM — **decode is rock-solid: 6/6 shots at 0.1332–0.1343, CV 0.3%,
> ZERO spikes, decode_speedup ~0.98 (near blacksmith parity, always clears the
> floor).** The back-to-back runs (floor 0.140 + ~43% spikes to 0.24–0.49) induced
> the spikes by keeping the VM under sustained load / reusing it. The one genuine
> residual is **prefill**: 5/6 tight at ~0.0106 but sitting right at the 0.95 floor,
> with 1/6 spiking to 0.0153 (speedup 0.66, hard fail) — the single-cold-forward
> fragility, fixable with a median of 2–3 prefill forwards.
>
> Caveat still open: the ranked job runs `./benchmark.sh` TWICE per VM (baseline →
> candidate), so the candidate is a 2nd run; the fresh-VM shots here are all 1st
> (cold) runs. A 2-runs-per-fresh-VM paired test is the last piece to clear the
> candidate. Earlier "host-GPU-scheduling dominates / decode needs median-of-N"
> conclusions below are SUPERSEDED for the real single-shot cadence — they
> described the back-to-back artifact.

Status: investigation complete (2026-07-07). Operator-facing analysis of
benchmark timing variance on the Blacksmith/tenki M4 Pro runner class, why it
leaks past the paired baseline, and the recommended harness fix. This is not a
submission and touches no editable surface; it is a record for whoever owns the
benchmark window / ranked workflow.

## TL;DR

- **The dominant variance in the scored 128-step decode is intermittent ~2×
  throttle SPIKES, not smooth thermal drift.** The un-throttled floor is stable
  and *dominant* (~0.140 s/tok, ~1–3 % CV, most windows); individual measurements
  spike to 2–3× that floor, roughly *periodic* (~17–18 min). Between-measurement
  ratio CV is ~40–50 % and is **flat across gaps of 5.5–33 min** — gap-independent,
  therefore transient.
- **Cause (resolved by bare-metal repro): the tenki ~2× dip is dominated by
  host/hypervisor GPU scheduling of the VM, NOT SoC thermal throttling.** Evidence:
  - In-VM telemetry is virtualized away (probes 28844180556 / 28846461724):
    `/arm-io/pmgr` absent (GPU-freq/CPU-limit unreadable); macOS thermal pressure
    pinned `Nominal` for all 254 samples even while prefill swung 2.4×.
  - **Bare-metal M4 Max repro (sensors work):** the same sustained MLX inference
    drove thermal pressure to **Moderate (144) / Heavy (92) — never Nominal**, GPU
    freq dipped 1578→~1318–1500 MHz, throughput dropped **only ~20 % (22.5→18.8
    tok/s)**. So (a) this workload *does* thermally throttle Apple Silicon AND
    macOS *does* register it → the tenki VM's permanent-`Nominal` is a **hidden
    sensor**, not "no throttle"; and (b) real SoC thermal here is only ~20 % on a
    *laptop* (worse-cooled than a datacenter Studio/mini host), so it cannot
    explain tenki's ~2× (≈100 %) dip. The excess is **host-imposed** (GPU
    time-slicing / overcommit of the VM). tenki is isolated from *other customers*,
    which does not preclude host-level GPU scheduling of this VM.
  - Host form factor: tenki hosts are **Mac minis** (operator-confirmed) — actively
    cooled desktops, so if anything they throttle *less* than the open-desk M4 Max
    laptop tested (~20%), which makes a ~2× SoC-thermal dip even less plausible.
  - Caveats: (1) different chip+model each side (M4 Max+Qwen-27B vs M4-Pro-VM+
    Gemma-31B), so 20%-vs-2× is indicative not exact; (2) the laptop was open-desk,
    whereas datacenter minis are often densely **racked** (constrained airflow,
    high ambient), which can raise thermal above the ~20% seen here — so racked-mini
    ambient thermal is a possible *secondary* contributor. But a clean 2×, the
    ~17-min periodicity, and the guest-invisibility still favor host scheduling as
    the dominant cause. The qualitative split (bare-metal thermal registers pressure
    + modest dip vs VM no-signal + huge dip) is the decisive part. Cleanest
    remaining confirmation: run the same `powermetrics`-under-load test on an actual
    Mac mini of that class (host-level, not in a VM).
  - Implication: the structural fix is a **provider question** (host GPU
    scheduling/overcommit on this VM class), or moving to **M3 Ultra / dedicated
    hardware**. Cooldown correctly does nothing (not the guest's heat); median-of-N
    still mitigates (rejects capped windows).
- **Primary fix: median-of-N on the timed window.** ~0.140 is the dominant state,
  so the median is both stable AND production-representative; N repeats reject the
  transient spike windows. (Prefer median over min-of-N: if the chip genuinely
  sustains a throttle in long production generations, min-of-N's boost-phase value
  would overstate real throughput.) Ranked scoring uses N=1 per axis today.
- Smooth thermal drift also exists but is **secondary**: prefill throttles ~2×
  cool→hot (≈0.010→0.020 s/tok); decode drifts ~0.6–0.8 %/min (up to +20 % over
  ~25 min). This drift is *directional in time* (candidate measured later/hotter
  than baseline), so the paired baseline — which cancels common-mode host
  differences — does not cancel it.
- **Adjacency reorder is a useful SECONDARY lever** (measure the two timed windows
  back-to-back so the smooth-drift component cancels), but it does **not** fix the
  spikes and so is not sufficient alone. The full-run ratio-vs-gap data shows the
  residual does not shrink at small gaps, confirming spikes dominate over drift.
- **Excluding prefill is not sufficient** (decode is the spiky axis too). **A
  2-minute idle settle does not work** (run 28832851608) — it cannot prevent a
  throttle event that occurs *during* a measurement. **Timing-only windows are
  ~5 min, model-load-bound** (run 28836241400), so no separate-process probe can
  measure a sub-~5-min gap.
- Net recommendation: **min/median-of-N on the timed window (primary)**, plus the
  **adjacency reorder (secondary)** for the residual drift. No idle wait.

## Context

(Historical: this investigation predates the Laguna XS 2.1 re-pin and
measured the then-current Gemma pin; the thermal conclusions are
model-independent.)

- Model: Gemma 4 31B 4-bit, dense, text tower only (~17 GB, fully RAM-resident).
- Score: `decode_speedup^0.75 * prefill_speedup^0.25`, per-axis 0.95 floors.
- Scoring uses a **paired baseline**: the pinned reference is built and timed on
  the same runner, same session, minutes before the candidate; speedups are the
  ratio. This cancels common-mode host/hour variance (see
  `benchmark-window-freeze.md`). It does **not** cancel a time-ordered drift
  between the two measurements — which is what thermal throttling is.
- Runner class: `blacksmith-12vcpu-macos-26` and `tenki-macos-latest-xlarge` are
  both `VirtualMac2,1` = Apple M4 Pro (Virtual) VMs, 12 vCPU. Measurements here
  are on tenki-xlarge; the thermal behavior is a property of the shared physical
  host, so treat magnitudes as indicative, not exact, for the ranked runner.

## What was measured

All probes ran the unmodified Gemma `main` reference on `tenki-macos-latest-xlarge`
via ad-hoc workflows on branch `gemma-tenki-decode` (baked ~/.cache Gemma
checkpoint, no re-download). Prefill is measured identically in every mode: one
cold 512-token forward.

| probe | run | what it showed |
|---|---|---|
| steady decode (`--local-submit`, 1023-step) | 28818728137 | decode CV **3.5 %** / max-min 1.10× (each machine once); prefill CV **33.8 %** / 2.23× |
| decode consistency (repeat 1023-step) | 28824046255 | within-host decode CV **14.4 %**, cross-host **2.2 %**; m1 drift **+20 %** (r=0.76), m3 +9.3 %, m2 a transient spike (0.17) |
| prefill variance (`--local-iterate`, timestamped) | live m5/m6 | cool prefill **0.0104–0.0113**, hot **0.018–0.024** (~2×); onset ~pass 2–3 (~5–10 min) |
| settle A/B (2-min idle before even windows) | 28832851608 | no-wait mean 0.0194 (CV 32 %), settle mean **0.0205** (CV 20.8 %) → settle **not cooler**; only the fresh first window is cool |
| adjacency oracle pilot (timing-only via self-built oracle) | 28836241400 | oracle path works; timing captured on floor-fail; **windows ~5 min = load-bound** |
| adjacency oracle full (128-step ratio-vs-gap, 2×8) | 28838838057 | ratio CV **~40–50 % flat across gaps 5.5–33 min** (gap-independent); decode floor ~0.140 (CV ~1–3 %) with intermittent 2–3× spikes; spikes ~17–18 min periodic |

Full-run ratio CV vs window gap (pooled, both machines):

| gap (min) | decode ratio CV | prefill ratio CV |
|---|---|---|
| 5.5 | 43 % | 49 % |
| 11 | 49 % | 52 % |
| 16.7 | 40 % | 21 % |
| 22.4 | 37 % | 54 % |
| 28 | 44 % | 64 % |
| 33 | 39 % | 26 % |

No monotonic growth with gap ⇒ the residual is dominated by a gap-independent
(transient) component, not by smooth drift. Example raw 128-step decode
(machine 2): `0.162, 0.282, 0.140, 0.138, 0.280, 0.141, 0.141, 0.140` — a stable
~0.140 floor with 2× spikes on windows 2 and 5 (~17 min apart).

Calibrated constants (blacksmith M4 Pro, `Constants.swift`): baseline decode
0.131727461265625, baseline prefill 0.01010573933984375 s/tok.

## Findings

### 1. Prefill variance is thermal throttling
Timestamped back-to-back windows show the *same machine* stepping from cool
(~0.010 s/tok, first window) to a hot plateau (~0.020 s/tok) within 1–3 windows
(~5–10 min), and staying there. Cross-machine spread (2.2×) is the same effect
sampled at different thermal states, not different hardware.

### 2. Decode also drifts (not thermally immune)
Repeated 1023-step steady decode on one host rises monotonically with time on
machine (m1: +20 %, r=0.76). The long window is "always hot" *within one
measurement*, which is why a single measurement per machine looked stable (3.5 %);
across repeated measurements the accumulating heat shows up as ~10–20 % within-host
CV. Decode tolerates a given gap better than prefill (two 128-step decodes 5 min
apart were ~identical in the pilot), but it is not flat.

### 3. The heater is sustained inference, not setup
The first timed window on a fresh machine is cool even though build + transform
already ran — so build/transform do not throttle the GPU. The ~5-min correctness
pass (256+ teacher-forced steps) is what heats it. The timed window itself
(~30–50 s of compute) heats little.

### 4. Windows are model-load-bound (~5 min), not correctness-bound
Skipping correctness entirely (timing-only via a self-built benchmark oracle)
still gave ~5-min windows (308 s process). The cost is dominated by 17 GB model
load + preflight + worker spawn. Consequence: **no separate-process probe can
measure a sub-~5-min gap**, so the true adjacent gap (~1–2 min) is only reachable
by extrapolation — or by an in-harness change that measures twice from one
resident model (not possible across two different code checkouts).

### 5. Why it leaks past the paired baseline
Score is `baseline/candidate`, both on the same host. Common-mode host speed
cancels. But the two measurements are **~6–7 min apart** today (baseline
correctness sits between the baseline timed window and the candidate timed
window), and the host is monotonically heating over that window — so the
candidate is measured hotter than the baseline. That difference does **not**
cancel; it biases the ratio.

## Recommended fix

### Primary: min/median-of-N on the timed window
The full ratio-vs-gap run shows the scored 128-step decode is dominated by
intermittent 2–3× throttle spikes over a stable ~0.140 floor, gap-independent.
The direct fix is to measure the timed window **N times and take the min (or
median)** rather than a single shot. Throttling only makes a window slower, so
min-of-N recovers the un-throttled floor; median-of-N is the more conventional,
slightly conservative choice. Either collapses the ~40 % single-shot ratio CV
toward the floor's ~1–3 %.

Cost: N× the timed window (~50 s each) per axis per side — modest next to the
~5-min model load that already dominates each measurement. This is a
`benchmark-window-freeze.md` **ranking-contract change** (it changes how a fixed
baseline maps to a score): update `benchmarkPrefillTimedRuns` / add a decode
repeat count, the doc, and `BenchmarkWindowFreezeTests.swift` together. It does
not change the *charged work per timed run*, so it is not a re-baseline, but note
that reporting min-of-N vs a single run shifts the baseline's own measured number
too — measure the paired baseline with the same N.

Caveat: if the spikes are a periodic *external* event (the ~17-min periodicity
hints at this), N repeats spread over a few minutes reduce but may not eliminate
the chance that the spike lands on every repeat. N should span more than one spike
period, or use a trimmed statistic robust to one bad window in N.

### Secondary: adjacency reorder (for the residual smooth drift)
Reorder the ranked pipeline so the two timed windows run back-to-back with both
correctness passes moved after:

1. Build baseline, build candidate, transform both.
2. Run the **two timed windows back-to-back** (baseline, then candidate).
3. Run **both correctness passes after** both timed windows.

This cuts the baseline↔candidate timed gap from ~6–7 min to ~1–2 min (bounded by
the candidate's model load), cancelling the smooth-drift component in the ratio.
It does **not** fix the spikes (they are gap-independent), so on its own it is
insufficient — the ratio-vs-gap data shows the residual does not shrink at small
gaps. Do it in addition to N-of-N, not instead. It requires separating "timed
window" from "correctness" in `benchmark.sh`; it does not change the charged work.
Confirm the frozen-window invariants still hold (one validated seed prefill + N
validated decode steps; no identical repeated charged forward).

### Root-cause follow-up
Done: in-guest telemetry (runs 28844180556 / 28846461724) + a bare-metal M4 Max
repro (`powermetrics` under sustained MLX load) together show the tenki ~2× is
host/hypervisor GPU scheduling, not SoC thermal (bare-metal thermal is only ~20%
and registers Moderate/Heavy pressure, which the VM hides). Remaining actions:
1. **Provider question (primary):** ask tenki/Blacksmith whether this VM class has
   host-level GPU scheduling / overcommit / periodic capping — an isolated-from-
   other-customers VM can still have its GPU access time-sliced by the host. This
   is the structural fix.
2. **Runner class:** re-measure on the **M3 Ultra** contract hardware (or any
   dedicated box with a reserved GPU). If the ~2× dip is absent there, the ranked
   hardware fixes it structurally — no scoring change needed.
3. **In-harness mitigation (works regardless of cause):** median-of-N on the timed
   window. Cooldown is confirmed useless (not the guest's heat).

## Rejected alternatives

- **Exclude prefill (weight 0).** Simple (`Constants.swift` weights + freeze
  doc/test, no re-baseline), but decode also drifts, so it does not reach the
  target; and it removes a real optimization axis and lets submissions regress
  prefill for free. At most, *down-weight*.
- **2-minute blanket idle settle.** Empirically insufficient (run 28832851608):
  settle windows were not cooler than no-wait. Cooldown constant > 2 min; a longer
  settle would work only at large, per-measurement time cost, applied to both
  sides — expensive and unproven.
- **Warm-up-then-measure-second (same prompt).** Rejected on soundness: an
  identical untimed warmup lets editable code memoize one forward and serve the
  timed one for free (the "reclaimable warmup" the freeze doc already removed);
  also measures a warm, non-production prefill. See `benchmark-window-freeze.md`.

## Reproduce / validate

Ad-hoc workflows on `gemma-tenki-decode` (dispatch-only unless noted):
`gemma-decode-consistency.yml`, `gemma-prefill-variance.yml`,
`gemma-settle-test.yml`, `gemma-adjacency-oracle.yml`. The last builds a
benchmark oracle on the runner (`generate-golden` → assemble `BenchmarkGolden`)
so the timing-only path runs without the private golden. To get the real
128-step decode + prefill residual-vs-gap curve (gaps ≥ 5 min, then extrapolate):
`gh workflow run gemma-adjacency-oracle.yml --ref gemma-tenki-decode -f machines="[1,2]" -f passes="8"`.

The definitive validation of the fix is to implement the reorder and compare the
scored `prefill_speedup` / `decode_speedup` CV across repeated ranked runs
against today's bundled order.
