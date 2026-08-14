#!/usr/bin/env bash
# Emit a REDACTED failure record for a failed benchmark job, safe for public
# artifact upload.
#
# A failing score.json must never be uploaded raw: its free-text error and
# first_failing/expected/actual token fields can carry hidden-golden token
# values, and errors thrown inside the sandboxed worker can carry arbitrary
# submitted-code-controlled text (an exfiltration channel for anything the
# model code observed). Until now the failing score.json simply died with the
# ephemeral runner, so a floor failure, a decode-oracle token mismatch, and a
# harness crash all surfaced identically as "benchmark_failed" with no way to
# tell them apart afterwards.
#
# This trusted script closes that observability gap by deriving only:
#   - a failure CATEGORY matched against harness-authored error prefixes,
#   - harness-authored booleans (passed flags, partial_result),
#   - already-coarsened wall seconds,
#   - the first failing decode step floored to a coarse bucket of 32.
# The raw error string is never printed, uploaded, or partially quoted.
#
# The public behavior gate runs BEFORE any score.json exists, so every
# failure there used to collapse into the opaque "no_score" category --
# indistinguishable from any other pre-benchmark failure, which invited
# consumers to guess the failing gate from the surrounding (skipped) step
# names instead. When no score-derived category is available, this script
# now also consults the public gate report: its verdict booleans are
# authored by the pin-verified trusted CLI and reference only the
# checked-in public fixture, while its error string can embed sandboxed
# worker stderr and is therefore only prefix-matched against fixed
# harness-authored prefixes, never emitted.
set -euo pipefail

out="${1:?usage: redact-benchmark-failure.sh OUTPUT_PATH [SCORE_PATH] [PUBLIC_GATE_REPORT_PATH]}"
score_path="${2:-score.json}"
public_gate_report_path="${3:-public-gate-report.json}"

category="no_score"
step_bucket="null"
passed="null"
passed_correctness="null"
passed_decode_floor="null"
passed_prefill_floor="null"
partial_result="null"
benchmark_wall_seconds="null"
timed_benchmark_seconds="null"

if [[ -s "${score_path}" ]]; then
  if ! jq -e 'type == "object" and (.metrics | type == "object")' "${score_path}" >/dev/null 2>&1; then
    category="invalid_score"
  else
    # Plain field access (no `//` fallback): jq's alternative operator treats
    # false as empty, which would silently rewrite `false` booleans to the
    # fallback. Missing keys already come back as literal `null`.
    #
    # Every field below is additionally coerced to its expected scalar type
    # (booleans -> boolean-or-null, numbers -> number-or-null) before it is
    # embedded in the redacted record. score.json here is the trusted sealed
    # score, so these fields are already well-typed on a legitimate run and the
    # coercion is a no-op; the guard is defense in depth so that a
    # non-scalar/attacker-shaped value (object, array, or string) in any of
    # these fields can never be passed through --argjson verbatim into the
    # uploaded artifact as a covert exfiltration channel. The raw error string
    # is still never emitted; it is only matched against fixed prefixes below.
    error_text="$(jq -r 'if (.metrics.error | type) == "string" then .metrics.error else "" end' "${score_path}")"
    passed="$(jq 'if (.passed | type) == "boolean" then .passed else null end' "${score_path}")"
    passed_correctness="$(jq 'if (.metrics.passed_correctness | type) == "boolean" then .metrics.passed_correctness else null end' "${score_path}")"
    passed_decode_floor="$(jq 'if (.metrics.passed_decode_speedup_floor | type) == "boolean" then .metrics.passed_decode_speedup_floor else null end' "${score_path}")"
    passed_prefill_floor="$(jq 'if (.metrics.passed_prefill_speedup_floor | type) == "boolean" then .metrics.passed_prefill_speedup_floor else null end' "${score_path}")"
    partial_result="$(jq 'if (.metrics.partial_result | type) == "boolean" then .metrics.partial_result else null end' "${score_path}")"
    benchmark_wall_seconds="$(jq 'if (.metrics.benchmark_wall_seconds | type) == "number" then .metrics.benchmark_wall_seconds else null end' "${score_path}")"
    timed_benchmark_seconds="$(jq 'if (.metrics.timed_benchmark_seconds | type) == "number" then .metrics.timed_benchmark_seconds else null end' "${score_path}")"

    # Categories are matched ONLY against fixed prefixes that the trusted
    # harness itself authors (QwenRuntimeBenchmark). Anything else --
    # including error text that originated inside the sandboxed worker -- is
    # deliberately collapsed to an opaque category.
    if [[ "${error_text}" == "performance floor failed"* ]]; then
      category="floor_failed"
    # The acceptance band is evaluated BEFORE correctness runs, so the band
    # path leaves passed_correctness false (correctnessReport is still nil)
    # and would otherwise fall through to "correctness_failed" below -- the
    # single most misleading verdict we can hand a participant whose tokens
    # were never checked. Match the harness's own prefix
    # (TimedRunScoreEvaluation.firstFailureReason) ahead of that fallback.
    elif [[ "${error_text}" == "acceptance band failed"* ]]; then
      category="acceptance_band_failed"
    elif [[ "${error_text}" == "benchmark prefill token mismatch"* ]]; then
      category="prefill_token_mismatch"
    elif [[ "${error_text}" == "benchmark decode seed token mismatch"* ]]; then
      category="decode_seed_token_mismatch"
    # DFlash track (laguna-xs-2.1-dflash-v1). The trusted parent authors these
    # as "DFlash contract violation: <kind>[ at step N]" and, by construction,
    # they carry NO reference token ids and NO reference logit values -- only a
    # kind, a step, and counts (see DFlashContractViolation). That property is
    # what stops the validator becoming a query oracle for the hidden prompt
    # (contract layer L6), so surface the KIND but never the detail text, and
    # allowlist the kinds exactly like every other branch here.
    elif [[ "${error_text}" == "DFlash contract violation: "* ]]; then
      dflash_kind="${error_text#DFlash contract violation: }"
      dflash_kind="${dflash_kind%% *}"
      case "${dflash_kind}" in
        emptyBlock) category="dflash_empty_block" ;;
        oversizedBlock) category="dflash_oversized_block" ;;
        outOfVocabularyToken) category="dflash_out_of_vocabulary_token" ;;
        tokenNotAdmissible) category="dflash_token_not_admissible" ;;
        residualBudgetExhausted) category="dflash_residual_budget_exhausted" ;;
        rowAccountingMismatch) category="dflash_row_accounting_mismatch" ;;
        declaredRowsMissing) category="dflash_declared_rows_missing" ;;
        workBindingMissing) category="dflash_work_binding_missing" ;;
        workBindingLogitMismatch) category="dflash_work_binding_logit_mismatch" ;;
        cacheOffsetDiverged) category="dflash_cache_offset_diverged" ;;
        incompleteRun) category="dflash_incomplete_run" ;;
        # The rejected-tail kinds (correctness contract Amendment 21). These
        # are the three detections aimed at a verifier that fabricates rows it
        # never computed, so they are the ones an audit most needs named: a run
        # that trips one of these is reporting a probable cheat, not a bug.
        # Without these arms they collapsed into the generic bucket below --
        # never a leak, but indistinguishable from an ordinary shape error.
        fabricatedRejection) category="dflash_fabricated_rejection" ;;
        rejectedRowReadoutMismatch) category="dflash_rejected_row_readout_mismatch" ;;
        draftTokenBindingMismatch) category="dflash_draft_token_binding_mismatch" ;;
        # Fail SAFE, not silent: an unrecognised kind still redacts (it never
        # reaches the raw detail text), but DFlashRedactorKindCoverageTests
        # asserts this arm is unreachable, so adding a kind to
        # DFlashContractViolation.Kind without an arm above fails the suite
        # rather than quietly degrading the category.
        *) category="dflash_contract_violation" ;;
      esac
      first_failing_step="$(jq -r '.metrics.first_failing_step // ""' "${score_path}")"
      if [[ "${first_failing_step}" =~ ^[0-9]+$ ]]; then
        step_bucket="$((first_failing_step / 32 * 32))"
      fi
    elif [[ "${error_text}" == "benchmark decode token mismatch"* ]]; then
      category="decode_token_mismatch"
      first_failing_step="$(jq -r '.metrics.first_failing_step // ""' "${score_path}")"
      if [[ "${first_failing_step}" =~ ^[0-9]+$ ]]; then
        step_bucket="$((first_failing_step / 32 * 32))"
      fi
    # The expert-streaming byte-read plausibility category
    # ("seed_read_implausible") was removed with the dense-to-MoE migration:
    # the harness no longer authors that error prefix because there is no
    # streaming path to meter. Its threat (hiding timed work outside the
    # measured window) is covered by the single-seed timed decode protocol
    # and the submission static review.
    #
    # Semantic GPQA gate outcomes. Both prefixes are authored by the trusted
    # gate script (run-semantic-gpqa-gate.sh). The infra category means the
    # judge API itself was unreachable (curl transport errors / HTTP 429 /
    # 5xx through the bounded retry budget -- run 30169401200's 529 burst):
    # no verdict exists, the submission was NOT rejected, and the run is
    # re-dispatchable. It must never be conflated with semantic_gpqa_failed,
    # which is a real judged rejection below min_pass.
    elif [[ "${error_text}" == "semantic GPQA gate infra failure"* ]]; then
      category="semantic_gpqa_infra_failed"
    elif [[ "${error_text}" == "semantic GPQA gate failed"* ]]; then
      category="semantic_gpqa_failed"
    elif [[ "${passed_correctness}" == "false" ]]; then
      category="correctness_failed"
    elif [[ -n "${error_text}" ]]; then
      category="runtime_error"
    else
      # score.json says passed with no error, yet the job failed: a
      # validation/staging step tripped, not the benchmark itself.
      category="score_present_validation_failed"
    fi
  fi
fi

# Public gate verdict. `passed` is coerced to boolean-or-null exactly like
# the score fields above so a non-scalar value can never ride --argjson into
# the uploaded artifact. It is surfaced unconditionally: together with the
# category it distinguishes "failed at the public gate" from "passed the
# public gate, failed later" without consulting hidden material.
public_gate_passed="null"
if [[ -s "${public_gate_report_path}" ]] \
  && jq -e 'type == "object"' "${public_gate_report_path}" >/dev/null 2>&1; then
  public_gate_passed="$(jq 'if (.passed | type) == "boolean" then .passed else null end' "${public_gate_report_path}")"
  if [[ "${category}" == "no_score" && "${public_gate_passed}" == "false" ]]; then
    # The gate report's error string can carry sandboxed-worker stderr (an
    # exfiltration channel), so like the score error above it is matched
    # ONLY against fixed prefixes the trusted harness itself authors and is
    # never emitted. Anything unrecognized collapses to an opaque
    # public-gate category.
    gate_error_text="$(jq -r 'if (.error | type) == "string" then .error else "" end' "${public_gate_report_path}")"
    if [[ "${gate_error_text}" == "runtime worker closed stdout before returning a response"* ]]; then
      category="public_gate_worker_crash"
    elif [[ "${gate_error_text}" == "teacher-forced token mismatch"* ]]; then
      category="public_gate_token_mismatch"
    else
      category="public_gate_failed"
    fi
  fi
fi

jq -n \
  --arg category "${category}" \
  --arg mode "${MLXFAST_BENCHMARK_MODE:-unknown}" \
  --argjson step_bucket "${step_bucket}" \
  --argjson public_gate_passed "${public_gate_passed}" \
  --argjson passed "${passed}" \
  --argjson passed_correctness "${passed_correctness}" \
  --argjson passed_decode_speedup_floor "${passed_decode_floor}" \
  --argjson passed_prefill_speedup_floor "${passed_prefill_floor}" \
  --argjson partial_result "${partial_result}" \
  --argjson benchmark_wall_seconds "${benchmark_wall_seconds}" \
  --argjson timed_benchmark_seconds "${timed_benchmark_seconds}" \
  '{
    failure_category: $category,
    mode: $mode,
    first_failing_step_bucket: $step_bucket,
    public_gate_passed: $public_gate_passed,
    passed: $passed,
    passed_correctness: $passed_correctness,
    passed_decode_speedup_floor: $passed_decode_speedup_floor,
    passed_prefill_speedup_floor: $passed_prefill_speedup_floor,
    partial_result: $partial_result,
    benchmark_wall_seconds: $benchmark_wall_seconds,
    timed_benchmark_seconds: $timed_benchmark_seconds
  }' > "${out}"

echo "benchmark-failure: category=${category} mode=${MLXFAST_BENCHMARK_MODE:-unknown} wall_seconds=${benchmark_wall_seconds}"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "### Benchmark failure (redacted)"
    echo ""
    echo '```json'
    cat "${out}"
    echo '```'
  } >> "${GITHUB_STEP_SUMMARY}"
fi
