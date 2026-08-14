#!/usr/bin/env bash
# Ensure private benchmark material is only used by this repository's benchmark
# workflow on an explicitly permitted branch namespace.
set -euo pipefail

readonly TRUSTED_REPOSITORY="Layr-Labs/mlxfast-challenge-dev"
readonly WORKFLOW_PATH=".github/workflows/dflash-benchmark.yml"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKFLOW_REF:?GITHUB_WORKFLOW_REF is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"

if [[ "${GITHUB_REPOSITORY}" != "${TRUSTED_REPOSITORY}" ]]; then
  echo "::error::private benchmark workflow must run in ${TRUSTED_REPOSITORY}, not ${GITHUB_REPOSITORY}" >&2
  exit 1
fi

if [[ "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  echo "::error::private benchmark workflow only supports workflow_dispatch" >&2
  exit 1
fi

case "${GITHUB_REF}" in
  refs/heads/main|refs/heads/submissions/*|refs/heads/baseline/*|refs/heads/yukon/baseline/*)
    ;;
  *)
    echo "::error::private benchmark workflow ref is not allowed: ${GITHUB_REF}" >&2
    echo "::error::allowed branches are main, submissions/*, baseline/*, and yukon/baseline/*" >&2
    exit 1
    ;;
esac

expected_workflow_ref="${TRUSTED_REPOSITORY}/${WORKFLOW_PATH}@${GITHUB_REF}"
if [[ "${GITHUB_WORKFLOW_REF}" != "${expected_workflow_ref}" ]]; then
  echo "::error::unexpected workflow ref ${GITHUB_WORKFLOW_REF}" >&2
  echo "::error::expected ${expected_workflow_ref}" >&2
  exit 1
fi

echo "dflash-benchmark: trusted workflow verified ${GITHUB_WORKFLOW_REF}"
