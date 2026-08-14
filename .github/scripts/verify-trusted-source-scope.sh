#!/usr/bin/env bash
# Pin the trusted-harness source scope independently of participant-controlled
# package and source manifests.
#
# The trusted binary (mlxfast-swift: timer, gates, score) is built inside the
# bench workspace from Package.swift target definitions plus the source trees
# those targets point at. The editable-surface enforcement and the overlay
# script are supposed to keep submissions out of that scope, but this check
# must not DEPEND on them: immediately before the trusted build, re-verify in
# the trusted shell that every source input feeding the trusted products in
# the bench workspace is byte-identical to the trusted git content of the
# checkout HEAD. A submission (or an overlay/enforcement bug) that expanded,
# redirected, or edited the trusted-harness source scope fails the run closed
# here, before any of it is compiled.
#
# Scope:
#   files  Package.swift, Package.resolved (the manifests that define which
#          sources feed which product; frozen by challenge policy)
#   dirs   Sources/MLXFastCLI, Sources/MLXFastTrustedHarness,
#          Sources/MLXFastCore (the timer/gates/score sources)
#
# Sources/MLXFastTransform is deliberately OUT of scope: it is a
# participant-editable path by contract, and its use inside the trusted
# binary is confined at run time by the parent-tool Seatbelt sandbox.
#
# The check also asserts, from the TRUSTED ref's benchmark.json, that no
# editablePaths entry overlaps the trusted scope -- a standing tripwire
# against the contract ever drifting to expose trusted sources.
#
# All git reads go through hardened-git.sh so a planted config/hook in a
# workspace .git can never execute as the invoking uid.
set -euo pipefail

TRUSTED_REPO="${1:?usage: verify-trusted-source-scope.sh TRUSTED_REPO BENCH_WORKSPACE}"
BENCH_WORKSPACE="${2:?bench workspace path is required}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
HARDENED_GIT="${SCRIPT_DIR}/hardened-git.sh"

TRUSTED_SCOPE_FILES=(
  "Package.swift"
  "Package.resolved"
)
TRUSTED_SCOPE_DIRS=(
  "Sources/MLXFastCLI"
  "Sources/MLXFastTrustedHarness"
  "Sources/MLXFastCore"
)

trusted_git() {
  "${HARDENED_GIT}" -C "${TRUSTED_REPO}" "$@"
}

failures=0
fail() {
  echo "::error::trusted source scope: $*" >&2
  failures=$((failures + 1))
}

require_regular_file() {
  local path="$1"
  if [[ -L "${path}" || ! -f "${path}" ]]; then
    fail "workspace file missing or not a regular file: ${path#"${BENCH_WORKSPACE}/"}"
    return 1
  fi
}

# Byte-compare one repo-relative path between the trusted ref and the bench
# workspace. `git cat-file blob` returns the exact committed bytes with no
# filter/attribute processing.
verify_scope_file() {
  local rel="$1"
  local ws_path="${BENCH_WORKSPACE}/${rel}"
  require_regular_file "${ws_path}" || return 0
  if ! trusted_git cat-file -e "HEAD:${rel}" 2>/dev/null; then
    fail "trusted ref is missing ${rel}"
    return 0
  fi
  if ! trusted_git cat-file blob "HEAD:${rel}" | cmp -s - "${ws_path}"; then
    fail "${rel} in the bench workspace differs from the trusted ref"
  fi
}

verify_scope_dir() {
  local dir="$1"
  local trusted_inventory workspace_inventory rel

  trusted_inventory="$(trusted_git ls-tree -r --name-only "HEAD" -- "${dir}" | LC_ALL=C sort)"
  if [[ -z "${trusted_inventory}" ]]; then
    fail "trusted ref has no files under ${dir}"
    return 0
  fi
  if [[ ! -d "${BENCH_WORKSPACE}/${dir}" || -L "${BENCH_WORKSPACE}/${dir}" ]]; then
    fail "workspace is missing directory ${dir} (or it is a symlink)"
    return 0
  fi
  # Every entry must be a plain file or directory: a symlink, socket, or
  # fifo inside the trusted scope is an expansion/redirection attempt.
  local irregular
  irregular="$(find "${BENCH_WORKSPACE}/${dir}" -mindepth 1 ! -type d ! -type f -print -quit)"
  if [[ -n "${irregular}" ]]; then
    fail "workspace contains a non-regular entry inside ${dir}: ${irregular#"${BENCH_WORKSPACE}/"}"
    return 0
  fi
  workspace_inventory="$(cd "${BENCH_WORKSPACE}" && find "${dir}" -type f | LC_ALL=C sort)"
  if [[ "${workspace_inventory}" != "${trusted_inventory}" ]]; then
    fail "file inventory under ${dir} differs from the trusted ref (added, removed, or renamed sources)"
    diff <(printf '%s\n' "${trusted_inventory}") \
      <(printf '%s\n' "${workspace_inventory}") | sed 's/^/::error::  /' >&2 || true
    return 0
  fi
  while IFS= read -r rel; do
    verify_scope_file "${rel}"
  done <<< "${trusted_inventory}"
}

paths_overlap() {
  local a="$1"
  local b="$2"
  [[ "${a}" == "${b}" || "${a}" == "${b}"/* || "${b}" == "${a}"/* ]]
}

# The contract itself comes from the TRUSTED ref, never from the (overlaid)
# work tree, so a submitted benchmark.json can not steer this check.
verify_contract_does_not_expose_trusted_scope() {
  local contract editable scope
  if ! contract="$(trusted_git cat-file blob "HEAD:benchmark.json" 2>/dev/null)"; then
    fail "trusted ref is missing benchmark.json"
    return 0
  fi
  while IFS= read -r editable; do
    [[ -n "${editable}" ]] || continue
    for scope in "${TRUSTED_SCOPE_FILES[@]}" "${TRUSTED_SCOPE_DIRS[@]}"; do
      if paths_overlap "${editable}" "${scope}"; then
        fail "editablePaths entry '${editable}' overlaps trusted scope path '${scope}'"
      fi
    done
  done < <(jq -r '.editablePaths[]' <<< "${contract}")
}

for scope_file in "${TRUSTED_SCOPE_FILES[@]}"; do
  verify_scope_file "${scope_file}"
done
for scope_dir in "${TRUSTED_SCOPE_DIRS[@]}"; do
  verify_scope_dir "${scope_dir}"
done
verify_contract_does_not_expose_trusted_scope

if [[ "${failures}" -ne 0 ]]; then
  echo "::error::trusted-harness source scope verification failed (${failures} finding(s)); refusing to build the trusted CLI from this workspace" >&2
  exit 1
fi
echo "benchmark: trusted-harness source scope verified against the trusted ref"
