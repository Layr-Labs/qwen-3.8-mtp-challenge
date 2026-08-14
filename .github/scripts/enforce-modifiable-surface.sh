#!/usr/bin/env bash
# Rejects diffs that touch files outside the benchmark contract's editablePaths.
# The allowlist is read from the BASE commit so a PR cannot grant itself access.
# Usage: BASE_SHA=<sha> HEAD_SHA=<sha> [CONTRACT_PATH=<contract>] enforce-modifiable-surface.sh
set -euo pipefail

: "${BASE_SHA:?BASE_SHA is required}"
: "${HEAD_SHA:?HEAD_SHA is required}"

# Which contract's editablePaths are in force. Deliberately the SAME variable
# name overlay-editable-paths.sh (:7) and run-submission-static-review.sh (:16)
# already read, so the three gates that consume an editable surface cannot
# disagree about which one a submission is judged against: a track sets
# CONTRACT_PATH once for the job and every gate follows. The serial track is
# the default and passes it unset.
CONTRACT_PATH="${CONTRACT_PATH:-benchmark.json}"

# benchmark.yml runs this script with the UNTRUSTED submission checkout as
# the working directory, where repo-local .git/config settings
# (core.fsmonitor, core.hooksPath, core.pager, filters) are
# attacker-influenced on submission branches. Per the doctrine in
# benchmark.yml's "Verify submitted commit and modifiable surface" step,
# every git read here goes through hardened-git.sh -- resolved next to THIS
# script, i.e. the trusted checkout's copy, never one inside the submission
# worktree.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
HARDENED_GIT="${SCRIPT_DIR}/hardened-git.sh"

if ! command -v jq >/dev/null 2>&1; then
  echo "::error::jq is required to read editablePaths from ${CONTRACT_PATH}"
  exit 1
fi

# Read the contract from the BASE commit, never from the submission work tree:
# a candidate that could point this gate at its own copy of a contract could
# widen its own surface just by editing one.
#
# Fail closed on a contract that cannot be read or carries no editablePaths.
# Both are configuration errors, and neither may degrade into a pass or into a
# rejection that names the wrong contract. There is deliberately no fallback to
# benchmark.json: silently judging a DFlash submission against the serial
# surface would reject it with a message pointing at a contract it was never
# submitted under.
if ! contract_source="$("${HARDENED_GIT}" show "${BASE_SHA}:${CONTRACT_PATH}")"; then
  echo "::error::cannot read ${CONTRACT_PATH} from base commit ${BASE_SHA}"
  exit 1
fi
# A contract with no editablePaths key (or a non-list value) makes jq fail; a
# contract with an empty list makes jq succeed with no output. Both mean the
# same thing here -- this gate has no allowlist -- and both must be loud. An
# empty allowlist left to run would reject every changed file while naming the
# wrong reason, and would pass silently whenever the diff is also empty.
if ! allowed="$(jq -r '.editablePaths[]' <<<"${contract_source}")" \
  || [[ -z "${allowed//[[:space:]]/}" ]]; then
  echo "::error::${CONTRACT_PATH} at base commit ${BASE_SHA} lists no usable editablePaths"
  exit 1
fi
changed="$("${HARDENED_GIT}" diff --name-only "${BASE_SHA}" "${HEAD_SHA}")"

bad=0
while IFS= read -r f; do
  [[ -z "${f}" ]] && continue
  ok=0
  while IFS= read -r allowed_path; do
    [[ -z "${allowed_path}" ]] && continue
    # Exact match OR file is inside an allowed directory prefix.
    if [[ "${f}" == "${allowed_path}" || "${f}" == "${allowed_path}/"* ]]; then
      ok=1
      break
    fi
  done <<<"${allowed}"
  if [[ "${ok}" == "0" ]]; then
    echo "::error file=${f}::${f} is outside the modifiable surface (see editablePaths in ${CONTRACT_PATH})"
    bad=1
  fi
done <<<"${changed}"
exit "${bad}"
