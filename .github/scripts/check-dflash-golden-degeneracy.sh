#!/usr/bin/env bash
# Refuse DFlash reference material whose CHARACTER matches the degenerate
# greedy self-continuations that Amendment 10 of
# docs/dflash-track-correctness-contract.md condemned.
#
# WHY THIS EXISTS AT ALL. The go-live runbook used to tell the operator to build
# ranked goldens with `dflash-reference --seed-generate N`, which extends the
# seed by REFERENCE-GENERATED tokens -- greedy self-continuation. Amendment 10
# measured exactly that material and found it degenerates into repetition, never
# enters the near-tie regime Criterion E exists to handle, and produces a draft
# acceptance rate (~100%) that prose does not (69%). Amendment 11 put the score
# consequence at 1.117x on degenerate material versus 0.840x on prose. A golden
# frozen from such a seed would advertise a speedup that does not exist. Prose is
# therefore a REQUIREMENT of the material, and a requirement that lives only in
# prose in a runbook is a requirement nobody enforces -- so it is enforced here,
# mechanically, on the bytes.
#
# WHAT IT IS NOT. This is a screen on material CHARACTER, not a proof of
# provenance. It cannot tell a hand-written prose-shaped token array from a real
# tokenization of real prose, and it does not try to; a determined operator can
# satisfy every threshold below with synthetic tokens. What it does do is make
# the specific, already-made mistake -- pointing --seed-generate at the ranked
# goldens because the runbook said to -- fail loudly at provisioning time instead
# of silently at ranking time. See "Residual exposure" at the bottom.
#
# USAGE
#   check-dflash-golden-degeneracy.sh GOLDEN_PATH LABEL
#       Full screen of a generated golden (seed + rows).
#   check-dflash-golden-degeneracy.sh --seed-only PLAN_PATH LABEL
#       Seed-only pre-screen of an operator-supplied seed plan, before ~40
#       minutes of reference generation is spent on it. Same seed threshold,
#       same constant, so the two screens cannot drift apart.
#
# Exit 0 = accepted. Exit 1 = rejected (degenerate or unmeasurable). Exit 2 =
# usage/parse error. The three statistics are printed on EVERY run, pass or
# fail: an operator needs to see the material's character, not only a verdict.
set -euo pipefail

# ---------------------------------------------------------------------------
# THRESHOLDS. Each is derived below from Amendment 10's measured table:
#
#   | golden                    | seed len | distinct seed | rows gap < 0.25 | min gap |
#   | seam-512-golden.json      | 512      | 122           | 0               | 1.875   |
#   | seam-b-golden.json        | 600      | 122           | 0               | 2.625   |
#   | seam-a-golden.json        | 509      | 122           | 0               | 1.875   |
#   | varied-512b-golden.json   | 512      | 317           | 3               | 0.0000  |
#   | varied-512-golden.json    | 512      | 317           | 3               | 0.0000  |
#
# They are spelled here, `readonly`, and deliberately NOT env-overridable. A
# threshold a dispatch input can loosen is a threshold that gets loosened by the
# operator who is in a hurry, which is the failure this whole file exists to
# prevent. Change them by changing this file, in a commit, with a reason.
# ---------------------------------------------------------------------------

# (1) SEED VARIETY -- distinct tokens as a fraction of seed length.
#
# MEASURED: degenerate seeds are 122 distinct tokens, at seed lengths 509-600,
# i.e. fractions 0.2033 (122/600) .. 0.2397 (122/509). Amendment 10's prose
# seeds are 317/512 = 0.6191. The public English prose fixture checked into this
# repo (correctness_prompts/public_longcopy_gate_english_512_256.json,
# cases[0].prompt_tokens) is 276 distinct in 512 = 0.5391 -- a second,
# independent prose sample, and the lower of the two.
#
# JUDGEMENT: 0.40. The measured evidence pins an empty interval boundary
# somewhere in (0.2397, 0.5391]; 0.40 is roughly the midpoint of it, leaving
# ~1.67x headroom above the most varied degenerate sample and ~1.35x below the
# least varied prose sample. Nothing measured says 0.40 specifically. A fraction
# rather than an absolute count because seed lengths vary (509, 512, 600 in the
# table alone) and 122 distinct tokens means something different in each.
readonly MIN_DISTINCT_SEED_FRACTION="0.40"

# (2) NEAR-TIE ROW COUNT -- rows whose top-2 logit gap is below 0.25.
#
# MEASURED, and this is the sharpest separation in the table: degenerate
# material has EXACTLY ZERO such rows in every sample; prose has 3 (per 128
# positions, per Amendment 10 Defect 2). Zero is not a small number here, it is
# a structural statement -- the model is so confident on repetitive text that the
# near-tie regime is never entered, so every gate calibrated on it was calibrated
# on a regime it never saw.
#
# JUDGEMENT: >= 1, i.e. the smallest value that is not the measured degenerate
# value. Extrapolating prose's measured DENSITY (3 per 128 -> ~12 per 512) into a
# hard threshold would reject honest prose that happens to run more confidently,
# and one sample is not enough to set a density floor. The claim being made is
# only "this material enters the near-tie regime at all", which is exactly the
# claim the measurement supports.
readonly MIN_NEAR_TIE_ROWS="1"

# The near-tie predicate itself is Amendment 10's own column, not a new
# constant: gap < 0.25. MEASURED in the sense that it is the threshold the
# contract already reports material against.
readonly NEAR_TIE_GAP="0.25"

# (3) MINIMUM TOP-2 GAP -- the most confident row in the whole artifact.
#
# MEASURED: degenerate minima are 1.875, 1.875 and 2.625. Prose minima are
# exactly 0.0000 (true ties, where the argmax is decided by tie-break order).
#
# JUDGEMENT: 1.0, i.e. reject if the minimum gap EXCEEDS 1.0. Roughly the
# midpoint of (0, 1.875), so it clears the lowest measured degenerate minimum by
# 0.875 and the highest prose minimum by 1.0.
#
# HONEST SCOPE, stated plainly because a redundant guard that reads as
# independent is worse than no guard: at these values check (3) is IMPLIED by
# check (2). `near_tie_rows >= 1` means some row has gap < 0.25, which forces
# min_gap < 0.25 < 1.0, so (3) can never be the sole reason for a rejection. It
# is retained for two reasons that are not "extra safety": it is the statistic an
# operator actually reads to judge HOW confident the material is (a min gap of
# 0.24 and one of 0.0000 pass check (2) identically and are not the same
# artifact), and it becomes load-bearing the moment check (2) is retuned to a
# density or scoped to a row window. Do not describe it as a second independent
# barrier, because it is not one.
readonly MAX_MIN_TOP2_GAP="1.0"

usage() {
  echo "usage: check-dflash-golden-degeneracy.sh [--seed-only] PATH LABEL" >&2
  exit 2
}

seed_only=0
if [[ "${1:-}" == "--seed-only" ]]; then
  seed_only=1
  shift
fi
if [[ "$#" -ne 2 ]]; then usage; fi
path="$1"
label="$2"
if [[ -z "${path}" || -z "${label}" ]]; then usage; fi
if [[ ! -f "${path}" ]]; then
  echo "::error::check-dflash-golden-degeneracy: ${label}: no such file: ${path}" >&2
  exit 2
fi

# One jq pass produces every statistic AND the failure list, so the numbers that
# are printed are byte-for-byte the numbers that were judged. A shell that
# recomputed them for the report could report a different artifact than it
# rejected.
#
# `.rows[].top2_logits` is optional in the golden schema. A row without a
# 2-element readout cannot be screened, and material that cannot be screened is
# refused rather than waved through: "unmeasurable" is the state this whole file
# exists to stop being treated as "fine".
report="$(
  jq -c \
    --argjson seed_only "${seed_only}" \
    --argjson min_fraction "${MIN_DISTINCT_SEED_FRACTION}" \
    --argjson min_near_tie "${MIN_NEAR_TIE_ROWS}" \
    --argjson near_tie_gap "${NEAR_TIE_GAP}" \
    --argjson max_min_gap "${MAX_MIN_TOP2_GAP}" \
    '
    def seedTokens: (.seed_tokens // []);
    def gaps:
      [ (.rows // [])[]
        | (.top2_logits // [])
        | select(length >= 2)
        | (.[0] - .[1])
      ];

    (seedTokens | length) as $seed_len
    | (seedTokens | unique | length) as $seed_distinct
    | (if $seed_len > 0 then ($seed_distinct / $seed_len) else 0 end) as $fraction
    | ((.rows // []) | length) as $rows_total
    | gaps as $gaps
    | ($gaps | length) as $rows_measurable
    | ([ $gaps[] | select(. < $near_tie_gap) ] | length) as $near_tie_rows
    | (if $rows_measurable > 0 then ($gaps | min) else null end) as $min_gap
    | ((.emitted_tokens // []) | length) as $emitted_len
    | ((.emitted_tokens // []) | unique | length) as $emitted_distinct
    | [
        (if $seed_len == 0
         then "the seed is empty, so its variety cannot be measured"
         else empty end),
        (if $seed_len > 0 and $fraction < $min_fraction
         then "seed variety \($fraction) is below the \($min_fraction) floor "
              + "(\($seed_distinct) distinct tokens in \($seed_len)); "
              + "Amendment 10 measured 122 distinct in 512-600 for greedy "
              + "self-continuation and 276-317 in 512 for prose"
         else empty end),
        (if $seed_only == 0 and $rows_total == 0
         then "the artifact carries no rows to screen"
         else empty end),
        (if $seed_only == 0 and $rows_total > 0 and $rows_measurable < $rows_total
         then "\($rows_total - $rows_measurable) of \($rows_total) rows carry no "
              + "2-element top2_logits readout, so their near-tie character "
              + "cannot be measured; regenerate the golden with a reference "
              + "build that records top-2 logits"
         else empty end),
        (if $seed_only == 0 and $rows_measurable > 0 and $near_tie_rows < $min_near_tie
         then "\($near_tie_rows) row(s) have a top-2 gap below \($near_tie_gap), "
              + "below the required \($min_near_tie); Amendment 10 measured "
              + "EXACTLY 0 on every degenerate fixture and 3 per 128 positions "
              + "on prose -- material that never enters the near-tie regime is "
              + "the material every green result on this track was wrongly "
              + "measured against"
         else empty end),
        (if $seed_only == 0 and $min_gap != null and $min_gap > $max_min_gap
         then "the minimum top-2 gap \($min_gap) exceeds \($max_min_gap); "
              + "Amendment 10 measured 1.875-2.625 for degenerate material and "
              + "0.0000 for prose"
         else empty end)
      ] as $failures
    | {
        label: "",
        seed_len: $seed_len,
        seed_distinct: $seed_distinct,
        seed_distinct_fraction: $fraction,
        rows_total: $rows_total,
        rows_measurable: $rows_measurable,
        near_tie_rows: $near_tie_rows,
        min_top2_gap: $min_gap,
        emitted_len: $emitted_len,
        emitted_distinct: $emitted_distinct,
        failures: $failures
      }
    ' "${path}"
)" || {
  echo "::error::check-dflash-golden-degeneracy: ${label}: ${path} is not readable JSON" >&2
  exit 2
}

field() { jq -r --arg k "$1" '.[$k] // "n/a" | tostring' <<<"${report}"; }

# ALWAYS printed, pass or fail. A verdict-only guard teaches an operator nothing
# about the material they are about to freeze into a ranked artifact.
echo "dflash-degeneracy [${label}] seed_len=$(field seed_len) distinct_seed_tokens=$(field seed_distinct) distinct_fraction=$(field seed_distinct_fraction) (floor ${MIN_DISTINCT_SEED_FRACTION})"
if [[ "${seed_only}" == "0" ]]; then
  echo "dflash-degeneracy [${label}] rows=$(field rows_total) measurable=$(field rows_measurable) rows_with_top2_gap_lt_${NEAR_TIE_GAP}=$(field near_tie_rows) (floor ${MIN_NEAR_TIE_ROWS})"
  echo "dflash-degeneracy [${label}] min_top2_gap=$(field min_top2_gap) (ceiling ${MAX_MIN_TOP2_GAP})"
  # NOT A GATE, and labelled so. Amendment 10's finding is about the SEED, and
  # checks (2) and (3) cover the generated rows' near-tie character. But a
  # 512-token greedy chain grown from an honest prose seed can still drift into
  # repetition toward its tail, and no measurement in the contract sizes that
  # drift -- so it is reported for the operator to look at and not thresholded
  # against a number nobody has measured. See "Residual exposure" below.
  echo "dflash-degeneracy [${label}] emitted_chain_len=$(field emitted_len) emitted_chain_distinct=$(field emitted_distinct) (reported, NOT gated -- no measurement sizes chain-tail drift)"
else
  echo "dflash-degeneracy [${label}] seed-only pre-screen; row statistics are unavailable until the golden is generated"
fi

failure_count="$(jq -r '.failures | length' <<<"${report}")"
if [[ "${failure_count}" != "0" ]]; then
  while IFS= read -r reason; do
    echo "::error::degenerate DFlash material rejected [${label}]: ${reason}" >&2
  done < <(jq -r '.failures[]' <<<"${report}")
  echo "::error::[${label}] this is the material profile docs/dflash-track-correctness-contract.md Amendment 10 condemned. Ranked goldens must be generated from a PRE-TOKENIZED PROSE seed, never from 'dflash-reference --seed-generate' (greedy self-continuation). See docs/dflash-go-live-runbook.md steps A and B." >&2
  exit 1
fi

echo "dflash-degeneracy [${label}] ACCEPTED"

# ---------------------------------------------------------------------------
# Residual exposure, recorded rather than hidden:
#
# 1. Character, not provenance. Synthetic tokens shaped to pass are not
#    detected. The seed is operator-supplied hidden material; its provenance is
#    an operator responsibility, and this file does not pretend otherwise.
#    MEASURED by attacking this script rather than reasoning about it: a
#    degenerate golden whose 122-distinct seed is PADDED to 384/512 with unique
#    ids clears check (1) at 0.75 and is still rejected -- by checks (2) and (3),
#    on its rows. The seed check is the cheap pre-screen; the ROW checks are the
#    defense. Never treat a passing pre-screen as a verdict.
# 2. ACCEPTED BY DESIGN, and the sharpest edge here: ONE near-tie row is enough.
#    A prose-variety seed with exactly one row at gap 0.10 and 127 rows at 8.0
#    passes (attacked, exit 0). That is the direct consequence of setting check
#    (2) to the measured-minimal value instead of a density -- a density floor
#    would reject honest prose that happens to run confidently, and one prose
#    sample cannot set one. What makes it tolerable is not this check: a real
#    greedy self-continuation must also get its SEED past check (1), and
#    122/512 = 0.238 does not. Do not describe check (2) as sufficient alone.
# 3. Chain-tail drift is reported, not gated (see above).
# 4. The prose side of the seed-variety threshold rests on two samples
#    (317/512 from Amendment 10, 276/512 from the public English fixture). A
#    third prose sample below 0.40 would mean 0.40 is too high, not that the
#    sample is degenerate. Re-derive the constant if that happens; do not
#    override it at the call site.
# ---------------------------------------------------------------------------
