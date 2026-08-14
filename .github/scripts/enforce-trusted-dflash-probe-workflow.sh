#!/usr/bin/env bash
# Trusted-context guard for the DFlash R2 key PROBE workflow.
#
# WHY THIS EXISTS AT ALL. dflash-probe-r2-keys.yml binds
# `benchmark-private-prompts-v2`, so it runs with the real R2 credentials, and
# it executes `.github/scripts/download-r2-object.sh` FROM THE DISPATCHED REF
# against hidden competition material. It shipped with no repository check, no
# ref allowlist and no GITHUB_WORKFLOW_REF check -- the only R2-credentialled
# DFlash workflow without one. A dispatch from a branch whose
# download-r2-object.sh had been edited to print the object body would have
# copied hidden golden bytes into an Actions log.
#
# WHY MAIN ONLY, and not the ranked allowlist. The ranked job admits
# submissions/*, baseline/* and yukon/baseline/* because a submission has to be
# able to run against trusted main's harness; those namespaces carry
# participant-authored content by design. This job has no such need -- it
# resolves object keys and prints a status, a byte count and a sha256 -- and it
# reads raw hidden material while doing it. Copying the ranked allowlist here
# would hand every branch namespace a participant can influence the ability to
# choose the code that touches a hidden golden, which is the same reasoning
# that made enforce-trusted-dflash-provision-workflow.sh main-only.
#
# WHAT THIS GUARD DOES NOT DO. `workflow_dispatch` runs the workflow file, and
# these scripts, from the ref the dispatcher selected, so this check is code the
# dispatched ref supplies about itself. It is decisive against a dispatch from
# the wrong ref, a fork or mirror of this repository, a non-dispatch trigger,
# and a job borrowing this guard under a different workflow file -- and it makes
# any of those a loud refusal in the log. It is NOT, by itself, a defence
# against an actor who can both push a branch and dispatch it, because that
# actor can edit this file on that branch. The control that binds such an actor
# is the environment's deployment-branch policy on
# `benchmark-private-prompts-v2`, which is where the credentials come from; this
# guard is the in-repo half of that pair, not a replacement for it.
set -euo pipefail

readonly TRUSTED_REPOSITORY="Layr-Labs/mlxfast-challenge-dev"
readonly WORKFLOW_PATH=".github/workflows/dflash-probe-r2-keys.yml"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKFLOW_REF:?GITHUB_WORKFLOW_REF is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"

if [[ "${GITHUB_REPOSITORY}" != "${TRUSTED_REPOSITORY}" ]]; then
  echo "::error::DFlash R2 key probe must run in ${TRUSTED_REPOSITORY}, not ${GITHUB_REPOSITORY}" >&2
  exit 1
fi

if [[ "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  echo "::error::DFlash R2 key probe only supports workflow_dispatch (got ${GITHUB_EVENT_NAME}); it must never run on push" >&2
  exit 1
fi

if [[ "${GITHUB_REF}" != "refs/heads/main" ]]; then
  echo "::error::DFlash R2 key probe ref is not allowed: ${GITHUB_REF}" >&2
  echo "::error::this job holds R2 credentials and downloads hidden competition material with scripts taken from the dispatched ref; only refs/heads/main may dispatch it" >&2
  exit 1
fi

expected_workflow_ref="${TRUSTED_REPOSITORY}/${WORKFLOW_PATH}@${GITHUB_REF}"
if [[ "${GITHUB_WORKFLOW_REF}" != "${expected_workflow_ref}" ]]; then
  echo "::error::unexpected workflow ref ${GITHUB_WORKFLOW_REF}" >&2
  echo "::error::expected ${expected_workflow_ref}" >&2
  exit 1
fi

echo "dflash-probe-r2-keys: trusted workflow verified ${GITHUB_WORKFLOW_REF}"
