#!/usr/bin/env bash
# Trusted-context guard for the Qwen-MTP golden PROVISIONING workflow.
#
# The Qwen-MTP twin of enforce-trusted-dflash-provision-workflow.sh, checked
# against qwen-mtp-provision-goldens.yml instead of the DFlash provisioning
# workflow. Same four checks, same order, same fail-closed posture: this
# repository only, workflow_dispatch only, an allowlisted ref only, and a
# GITHUB_WORKFLOW_REF that names THIS workflow file at THAT ref.
#
# WHY THE BRANCH SET IS NARROW. The ranked job admits submissions/*, baseline/*
# and yukon/baseline/* because a submission has to be able to run against
# trusted main's harness. This job holds R2 WRITE credentials and its whole
# output is an object that later becomes ranked hidden material -- so a dispatch
# from any branch a participant can create must never reach it. Copying the
# ranked allowlist would hand submissions/* the ability to overwrite a hidden
# golden.
#
# WHAT THIS GUARD DOES NOT DO. `workflow_dispatch` runs the workflow file, and
# these scripts, from the ref the dispatcher selected, so this check is code the
# dispatched ref supplies about itself. It is decisive against a dispatch from a
# ref outside the allowlist, a fork or mirror of this repository, a non-dispatch
# trigger, and a job borrowing this guard under a different workflow file. It is
# NOT, by itself, a defence against an actor who can both push an allowlisted
# branch and dispatch it, because that actor can edit this file on that branch.
# The control that binds such an actor is the environment's deployment-branch
# policy on the credentialled environment, plus the branch protection on each
# allowlisted ref; this guard is the in-repo half of that pair, not a
# replacement for it.
set -euo pipefail

readonly TRUSTED_REPOSITORY="Layr-Labs/qwen-3.8-mtp-challenge"
readonly WORKFLOW_PATH=".github/workflows/qwen-mtp-provision-goldens.yml"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKFLOW_REF:?GITHUB_WORKFLOW_REF is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"

if [[ "${GITHUB_REPOSITORY}" != "${TRUSTED_REPOSITORY}" ]]; then
  echo "::error::Qwen-MTP golden provisioning must run in ${TRUSTED_REPOSITORY}, not ${GITHUB_REPOSITORY}" >&2
  exit 1
fi

if [[ "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  echo "::error::Qwen-MTP golden provisioning only supports workflow_dispatch (got ${GITHUB_EVENT_NAME}); it must never run on push" >&2
  exit 1
fi

if [[ "${GITHUB_REF}" != "refs/heads/main" ]]; then
  echo "::error::Qwen-MTP golden provisioning ref is not allowed: ${GITHUB_REF}" >&2
  echo "::error::this job holds R2 WRITE credentials and freezes ranked hidden material; only refs/heads/main may dispatch it" >&2
  exit 1
fi

expected_workflow_ref="${TRUSTED_REPOSITORY}/${WORKFLOW_PATH}@${GITHUB_REF}"
if [[ "${GITHUB_WORKFLOW_REF}" != "${expected_workflow_ref}" ]]; then
  echo "::error::unexpected workflow ref ${GITHUB_WORKFLOW_REF}" >&2
  echo "::error::expected ${expected_workflow_ref}" >&2
  exit 1
fi

echo "qwen-mtp-provision-goldens: trusted workflow verified ${GITHUB_WORKFLOW_REF}"
