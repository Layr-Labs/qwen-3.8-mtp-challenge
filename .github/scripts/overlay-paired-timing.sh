#!/usr/bin/env bash
# Overlay the measure-job paired timing into the sealed gates score (gates +
# paired timing merge and harness-side floor check).
#
# Inputs (all runner-owned, produced earlier in the same job):
#   MLXFAST_SCORE_PATH             sealed gates+semantic score.json (base case
#                                  + gates verdicts; timing fields are the
#                                  skip-timed baseline placeholders)
#   MLXFAST_MEASURE_RESULTS_PATH   measure-job results.json (paired mode:
#                                  candidate + pinned-baseline seconds/token,
#                                  speedups, paired_score, telemetry stats)
#   MLXFAST_CANDIDATE_SCORE_PATH   measure-job's sealed candidate score.json
#                                  (harness-authored timing diagnostics:
#                                  timed_benchmark_seconds, wall seconds,
#                                  bandwidth fields -- already coarsened by the
#                                  harness where applicable)
#   MLXFAST_INTEGRITY_PATH         harness integrity record; its score_sha256
#                                  is re-anchored to the merged score
#   MLXFAST_DECODE_SPEEDUP_FLOOR / MLXFAST_PREFILL_SPEEDUP_FLOOR
#                                  keep in sync with MLXFastConstants.{score
#                                  DecodeSpeedupFloor,scorePrefillSpeedupFloor}
#
# Verdict semantics (identical to the old pipeline):
#   .passed = gates_score.passed AND both speedup floors
#   .score  = decode_speedup^0.75 * prefill_speedup^0.25 when passed, null
#             otherwise (with metrics.error set to the harness's own
#             "performance floor failed" prefix so redact-benchmark-failure.sh
#             categorizes it identically).
# Exits nonzero when the merged verdict is a failure, so the workflow job
# fails exactly where the old timing machine's benchmark.sh did.
set -euo pipefail

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
RESULTS_PATH="${MLXFAST_MEASURE_RESULTS_PATH:?MLXFAST_MEASURE_RESULTS_PATH is required}"
CANDIDATE_SCORE_PATH="${MLXFAST_CANDIDATE_SCORE_PATH:?MLXFAST_CANDIDATE_SCORE_PATH is required}"
INTEGRITY_PATH="${MLXFAST_INTEGRITY_PATH:-benchmark-integrity.json}"
DECODE_FLOOR="${MLXFAST_DECODE_SPEEDUP_FLOOR:?MLXFAST_DECODE_SPEEDUP_FLOOR is required}"
PREFILL_FLOOR="${MLXFAST_PREFILL_SPEEDUP_FLOOR:?MLXFAST_PREFILL_SPEEDUP_FLOOR is required}"
EXPECTED_COMMIT="${MLXFAST_EXPECTED_COMMIT:?MLXFAST_EXPECTED_COMMIT is required}"
if [[ ! "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::MLXFAST_EXPECTED_COMMIT must be a full lowercase commit SHA" >&2
  exit 1
fi

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "::error file=$1::required overlay input is missing or empty" >&2
    exit 1
  fi
}
require_file "${SCORE_PATH}"
require_file "${RESULTS_PATH}"
require_file "${CANDIDATE_SCORE_PATH}"
require_file "${INTEGRITY_PATH}"

# The paired results must be from a real paired run that measure-job ACCEPTed
# on both sides (its exit code already guaranteed that; assert the shape).
jq -e '
  .mode == "paired"
  and (.paired.decode_speedup | type == "number") and (.paired.decode_speedup > 0)
  and (.paired.prefill_speedup | type == "number") and (.paired.prefill_speedup > 0)
  and (.candidate.decode_seconds_per_token > 0)
  and (.candidate.prefill_seconds_per_token > 0)
  and (.baseline.decode_seconds_per_token > 0)
  and (.baseline.prefill_seconds_per_token > 0)
  and (.candidate.verdict == "ACCEPT")
  and (.baseline.verdict == "ACCEPT")
' "${RESULTS_PATH}" >/dev/null || {
  echo "::error file=${RESULTS_PATH}::paired results are not a clean ACCEPTed paired measurement" >&2
  exit 1
}

# Defense-in-depth on the candidate's sealed timing score: leak-field and
# expert-zero checks (the dense RAM-resident runtime must report structural
# zeros; a nonzero value is a schema regression that must fail loudly, not
# merge silently).
jq -e --arg expected_commit "${EXPECTED_COMMIT}" \
  --slurpfile gates "${SCORE_PATH}" '
  (.metrics.commit as $commit
  | ($commit | test("^[0-9a-f]{7,40}$"))
  and ($expected_commit | startswith($commit)))
  and (.metrics.first_failing_case == null)
  and (.metrics.first_failing_step == null)
  and (.metrics.expected_token == null)
  and (.metrics.actual_token == null)
  and (.metrics.expert_bytes_read == 0)
  and (.metrics.expert_cache_hits == 0)
  and (.metrics.expert_cache_misses == 0)
  and (.metrics.expert_cache_evictions == 0)
  and (.metrics.expert_read_seconds == 0)
  and (.metrics.expert_peak_cached_tensors == 0)
  and (.metrics.bandwidth_source == "ram_resident_model")
  and (.metrics.timed_benchmark_seconds > 0)
  and (.metrics.harness_hash | test("^[0-9a-f]{64}$"))
  and (.metrics.harness_hash == $gates[0].metrics.harness_hash)
  and (.metrics.weights_hash | test("^[0-9a-f]{64}$"))
  and (.metrics.weights_file_count | type == "number")
  and (.metrics.weights_file_count > 0)
  and (.metrics.weights_byte_count | type == "number")
  and (.metrics.weights_byte_count > 0)
  and (.metrics.weights_hash == $gates[0].metrics.weights_hash)
  and (.metrics.weights_file_count == $gates[0].metrics.weights_file_count)
  and (.metrics.weights_byte_count == $gates[0].metrics.weights_byte_count)
' "${CANDIDATE_SCORE_PATH}" >/dev/null || {
  echo "::error file=${CANDIDATE_SCORE_PATH}::candidate timing score failed the pre-merge checks" >&2
  exit 1
}

# Base the merged score on the gates score ($g) and overlay only the
# timing-derived fields:
#   - decode/prefill seconds per token: the candidate's measured timing.
#   - baseline_*: the pinned baseline's measured same-session timing.
#   - speedups/score: the paired ratio (ranking fields, full precision).
#   - floors: applied here in the trusted shell with the same constants.
#   - timed_benchmark_seconds / bandwidth_*: candidate harness diagnostics
#     (already coarsened by the harness).
#   - benchmark_wall_seconds, peak_ram_gb, process_resident_memory_gb: max()
#     across the gates pass and the timed pass (single serial process
#     semantics; diagnostic, not scored).
#   - expert_* counters: summed (structural zeros, asserted above).
merged_tmp="$(mktemp)"
trap 'rm -f "${merged_tmp}"' EXIT
jq -n \
  --slurpfile gates "${SCORE_PATH}" \
  --slurpfile timing "${CANDIDATE_SCORE_PATH}" \
  --slurpfile results "${RESULTS_PATH}" \
  --argjson decode_floor "${DECODE_FLOOR}" \
  --argjson prefill_floor "${PREFILL_FLOOR}" \
  '
  ($gates[0]) as $g | ($timing[0]) as $t | ($results[0]) as $r
  | ($r.paired.decode_speedup) as $ds
  | ($r.paired.prefill_speedup) as $ps
  | ($ds >= $decode_floor) as $decode_ok
  | ($ps >= $prefill_floor) as $prefill_ok
  | ($g.passed and $decode_ok and $prefill_ok) as $merged_passed
  | $g
  | .metrics.decode_seconds_per_token = $r.candidate.decode_seconds_per_token
  | .metrics.prefill_seconds_per_token = $r.candidate.prefill_seconds_per_token
  | .metrics.baseline_decode_seconds_per_token = $r.baseline.decode_seconds_per_token
  | .metrics.baseline_prefill_seconds_per_token = $r.baseline.prefill_seconds_per_token
  | .metrics.decode_speedup = $ds
  | .metrics.prefill_speedup = $ps
  | .metrics.decode_speedup_floor = $decode_floor
  | .metrics.prefill_speedup_floor = $prefill_floor
  | .metrics.passed_decode_speedup_floor = $decode_ok
  | .metrics.passed_prefill_speedup_floor = $prefill_ok
  | .metrics.timed_benchmark_seconds = $t.metrics.timed_benchmark_seconds
  | .metrics.bandwidth_gb_per_token = $t.metrics.bandwidth_gb_per_token
  | .metrics.bandwidth_source = $t.metrics.bandwidth_source
  | .metrics.benchmark_wall_seconds = ([.metrics.benchmark_wall_seconds, $t.metrics.benchmark_wall_seconds] | max)
  | .metrics.expert_bytes_read = (.metrics.expert_bytes_read + $t.metrics.expert_bytes_read)
  | .metrics.expert_cache_hits = (.metrics.expert_cache_hits + $t.metrics.expert_cache_hits)
  | .metrics.expert_cache_misses = (.metrics.expert_cache_misses + $t.metrics.expert_cache_misses)
  | .metrics.expert_cache_evictions = (.metrics.expert_cache_evictions + $t.metrics.expert_cache_evictions)
  | .metrics.expert_read_seconds = (.metrics.expert_read_seconds + $t.metrics.expert_read_seconds)
  | .metrics.expert_peak_cached_tensors = ([.metrics.expert_peak_cached_tensors, $t.metrics.expert_peak_cached_tensors] | max)
  | .metrics.peak_ram_gb = ([.metrics.peak_ram_gb, $t.metrics.peak_ram_gb] | max)
  | .metrics.process_resident_memory_gb = ([.metrics.process_resident_memory_gb, $t.metrics.process_resident_memory_gb] | max)
  | .metrics.partial_result = false
  | .passed = $merged_passed
  | if $merged_passed then
      .score = (pow($ds; 0.75) * pow($ps; 0.25))
    else
      .score = null
      | (if ($decode_ok and $prefill_ok) then . else
          .metrics.error = "performance floor failed (paired decode_speedup=\($ds) prefill_speedup=\($ps))"
        end)
    end
  ' > "${merged_tmp}"

mv "${merged_tmp}" "${SCORE_PATH}"
trap - EXIT

# Re-anchor the published-integrity pair: the sidecar must name the artifact
# filename (score.json) so `shasum -c` works on the downloaded artifact, and
# the harness integrity record must reference the merged bytes.
score_hash="$(shasum -a 256 "${SCORE_PATH}" | awk '{print $1}')"
printf '%s  score.json\n' "${score_hash}" > "${SCORE_PATH}.sha256"
integrity_tmp="$(mktemp)"
jq --arg hash "${score_hash}" '.score_sha256 = $hash' "${INTEGRITY_PATH}" > "${integrity_tmp}"
mv "${integrity_tmp}" "${INTEGRITY_PATH}"

merged_passed="$(jq -r '.passed' "${SCORE_PATH}")"
merged_score="$(jq -r '.score' "${SCORE_PATH}")"
echo "overlay-paired-timing: passed=${merged_passed} score=${merged_score} decode_speedup=$(jq -r '.metrics.decode_speedup' "${SCORE_PATH}") prefill_speedup=$(jq -r '.metrics.prefill_speedup' "${SCORE_PATH}")"

if [[ "${merged_passed}" != "true" ]]; then
  echo "::error::merged benchmark verdict is a failure (speedup floor or gates)" >&2
  exit 1
fi
