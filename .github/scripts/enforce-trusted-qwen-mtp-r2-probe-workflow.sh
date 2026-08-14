#!/usr/bin/env bash
# Trusted-context guard for the Qwen-MTP R2 key probe workflow.
#
# The Qwen-MTP twin of enforce-trusted-dflash-probe-workflow.sh, checked against
# qwen-mtp-r2-key-probe.yml instead of the DFlash probe workflow. Same four
# checks, same order, same fail-closed posture: this repository only,
# workflow_dispatch only, an allowlisted ref only, and a GITHUB_WORKFLOW_REF
# that names THIS workflow file at THAT ref.
#
# WHY A READ-ONLY DIAGNOSTIC NEEDS THIS AT ALL. qwen-mtp-r2-key-probe.yml binds
# the real R2 credentials and executes `.github/scripts/download-r2-object.sh`
# FROM THE DISPATCHED REF against hidden competition material. "It only prints
# an HTTP status, a byte count and a sha256" is a property of the trusted copy
# of that script, not of the workflow: a dispatch from a branch whose
# download-r2-object.sh had been edited to print the object body would have
# copied hidden golden bytes into an Actions log.
#
# WHY THE BRANCH SET IS NARROW, and not the ranked allowlist. The ranked job
# admits submissions/*, baseline/* and yukon/baseline/* because a submission has
# to be able to run against trusted main's harness; those namespaces carry
# participant-authored content by design. This job has no such need -- it
# resolves object keys and prints a status, a byte count and a sha256 -- and it
# reads raw hidden material while doing it. Copying the ranked allowlist here
# would hand every branch namespace a participant can influence the ability to
# choose the code that touches a hidden golden, which is the same reasoning that
# makes enforce-trusted-qwen-mtp-provision-workflow.sh narrow.
#
# WHAT THIS GUARD DOES NOT DO. `workflow_dispatch` runs the workflow file, and
# these scripts, from the ref the dispatcher selected, so this check is code the
# dispatched ref supplies about itself. It is decisive against a dispatch from a
# ref outside the allowlist, a fork or mirror of this repository, a non-dispatch
# trigger, and a job borrowing this guard under a different workflow file -- and
# it makes any of those a loud refusal in the log. It is NOT, by itself, a
# defence against an actor who can both push an allowlisted branch and dispatch
# it, because that actor can edit this file on that branch. The control that
# binds such an actor is the environment's deployment-branch policy on the
# credentialled environment, plus the branch protection on each allowlisted ref;
# this guard is the in-repo half of that pair, not a replacement for it.
set -euo pipefail

readonly TRUSTED_REPOSITORY="Layr-Labs/qwen-3.8-mtp-challenge"
readonly WORKFLOW_PATH=".github/workflows/qwen-mtp-r2-key-probe.yml"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKFLOW_REF:?GITHUB_WORKFLOW_REF is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"

if [[ "${GITHUB_REPOSITORY}" != "${TRUSTED_REPOSITORY}" ]]; then
  echo "::error::Qwen-MTP R2 key probe must run in ${TRUSTED_REPOSITORY}, not ${GITHUB_REPOSITORY}" >&2
  exit 1
fi

if [[ "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  echo "::error::Qwen-MTP R2 key probe only supports workflow_dispatch (got ${GITHUB_EVENT_NAME}); it must never run on push" >&2
  exit 1
fi

if [[ "${GITHUB_REF}" != "refs/heads/main" ]]; then
  echo "::error::Qwen-MTP R2 key probe ref is not allowed: ${GITHUB_REF}" >&2
  echo "::error::this job holds R2 credentials and downloads hidden competition material with scripts taken from the dispatched ref; only refs/heads/main may dispatch it" >&2
  exit 1
fi

expected_workflow_ref="${TRUSTED_REPOSITORY}/${WORKFLOW_PATH}@${GITHUB_REF}"
if [[ "${GITHUB_WORKFLOW_REF}" != "${expected_workflow_ref}" ]]; then
  echo "::error::unexpected workflow ref ${GITHUB_WORKFLOW_REF}" >&2
  echo "::error::expected ${expected_workflow_ref}" >&2
  exit 1
fi

echo "qwen-mtp-r2-key-probe: trusted workflow verified ${GITHUB_WORKFLOW_REF}"
