#!/usr/bin/env bash
# Trusted-context guard for the DFlash golden PROVISIONING workflow.
#
# A near-twin of enforce-trusted-dflash-benchmark-workflow.sh, with one
# deliberate difference: the allowed branch set is `main` ONLY.
#
# The ranked job admits submissions/*, baseline/* and yukon/baseline/* because a
# submission has to be able to run against trusted main's harness. This job
# holds R2 WRITE credentials and its whole output is an object that later
# becomes ranked hidden material -- so a dispatch from any branch a participant
# can create must never reach it. Copying the ranked allowlist would have handed
# submissions/* the ability to overwrite a hidden golden.
set -euo pipefail

readonly TRUSTED_REPOSITORY="Layr-Labs/mlxfast-challenge-dev"
readonly WORKFLOW_PATH=".github/workflows/dflash-provision-goldens.yml"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKFLOW_REF:?GITHUB_WORKFLOW_REF is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"

if [[ "${GITHUB_REPOSITORY}" != "${TRUSTED_REPOSITORY}" ]]; then
  echo "::error::DFlash golden provisioning must run in ${TRUSTED_REPOSITORY}, not ${GITHUB_REPOSITORY}" >&2
  exit 1
fi

if [[ "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  echo "::error::DFlash golden provisioning only supports workflow_dispatch (got ${GITHUB_EVENT_NAME}); it must never run on push" >&2
  exit 1
fi

if [[ "${GITHUB_REF}" != "refs/heads/main" ]]; then
  echo "::error::DFlash golden provisioning ref is not allowed: ${GITHUB_REF}" >&2
  echo "::error::this job holds R2 WRITE credentials and freezes ranked hidden material; only refs/heads/main may dispatch it" >&2
  exit 1
fi

expected_workflow_ref="${TRUSTED_REPOSITORY}/${WORKFLOW_PATH}@${GITHUB_REF}"
if [[ "${GITHUB_WORKFLOW_REF}" != "${expected_workflow_ref}" ]]; then
  echo "::error::unexpected workflow ref ${GITHUB_WORKFLOW_REF}" >&2
  echo "::error::expected ${expected_workflow_ref}" >&2
  exit 1
fi

echo "dflash-provision-goldens: trusted workflow verified ${GITHUB_WORKFLOW_REF}"
