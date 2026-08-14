#!/usr/bin/env bash
# Verify the exact correctness golden used by the benchmark workflow.
set -euo pipefail

GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-correctness_golden.json}"
: "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256:?MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256 is required}"
: "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_BYTES:?MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_BYTES is required}"
: "${MLXFAST_EXPECTED_CORRECTNESS_STEPS:?MLXFAST_EXPECTED_CORRECTNESS_STEPS is required}"

if [[ ! -s "${GOLDEN_PATH}" ]]; then
  echo "::error::correctness golden missing or empty at ${GOLDEN_PATH}" >&2
  exit 1
fi

actual_hash="$(shasum -a 256 "${GOLDEN_PATH}" | awk '{print $1}')"
actual_bytes="$(wc -c < "${GOLDEN_PATH}" | tr -d ' ')"

if [[ "${actual_hash}" != "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256}" ]]; then
  echo "::error file=${GOLDEN_PATH}::correctness golden sha256 mismatch" >&2
  echo "expected=${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256}" >&2
  echo "actual=${actual_hash}" >&2
  exit 1
fi
if [[ "${actual_bytes}" != "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_BYTES}" ]]; then
  echo "::error file=${GOLDEN_PATH}::correctness golden byte count mismatch" >&2
  echo "expected=${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_BYTES}" >&2
  echo "actual=${actual_bytes}" >&2
  exit 1
fi

printf '%s  correctness_golden.json\n' "${actual_hash}" > "${GOLDEN_PATH}.sha256"
printf '%s  correctness_golden.json\n' "${actual_bytes}" > "${GOLDEN_PATH}.bytes"

coverage_json="$(jq -e --argjson base_steps "${MLXFAST_EXPECTED_CORRECTNESS_STEPS}" '
  (.correctness_gates // {}) as $g
  | ($g.anchors // []) as $anchors
  | ($g.free_run // []) as $free_run
  | ($g.behavior // []) as $behavior
  | {
      case_count: (
        (.cases | length)
        + ($anchors | length)
        + ($free_run | length)
        + ($behavior | length)
      ),
      checked_steps: (
        ((.cases | length) * $base_steps)
        + ($anchors | length)
        + ([$free_run[] | (.exact_prefix_tokens // (.expected_tokens | length))] | add // 0)
        + ([$behavior[] | .max_new_tokens] | add // 0)
      )
    }
' "${GOLDEN_PATH}")"
case_count="$(jq -r '.case_count' <<<"${coverage_json}")"
checked_steps="$(jq -r '.checked_steps' <<<"${coverage_json}")"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "MLXFAST_EXPECTED_CORRECTNESS_CASES=${case_count}"
    echo "MLXFAST_EXPECTED_CORRECTNESS_CHECKED_STEPS=${checked_steps}"
  } >> "${GITHUB_ENV}"
fi

echo "benchmark: verified correctness golden ${actual_hash} bytes=${actual_bytes} cases=${case_count} checked_steps=${checked_steps}"
