#!/usr/bin/env bash
# Run a command under the macOS no-network Seatbelt profile.
set -euo pipefail

curl_profile=""
command_profile=""
command_pid=""
pending_signal=""
# shellcheck disable=SC2329  # invoked indirectly via `trap cleanup_profiles EXIT` below
cleanup_profiles() {
  [[ -z "${curl_profile}" ]] || rm -f "${curl_profile}" || true
  [[ -z "${command_profile}" ]] || rm -f "${command_profile}" || true
}
trap cleanup_profiles EXIT

forward_signal() {
  local signal="$1"
  pending_signal="${signal}"
  if [[ -n "${command_pid}" ]] && kill -0 "${command_pid}" 2>/dev/null; then
    kill -s "${signal}" "${command_pid}" 2>/dev/null || true
  fi
}
trap 'forward_signal HUP' HUP
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM

if [[ "$#" -eq 0 ]]; then
  echo "usage: run-offline.sh COMMAND [ARG...]" >&2
  exit 2
fi

if ! command -v sandbox-exec >/dev/null 2>&1; then
  echo "run-offline.sh: sandbox-exec not found; offline command execution requires macOS" >&2
  exit 1
fi

sandbox_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

absolute_path() {
  local path="$1"
  local dir
  local base
  dir="$(dirname "${path}")"
  base="$(basename "${path}")"
  if [[ "${dir}" = "." ]]; then
    printf '%s/%s\n' "$(pwd -P)" "${base}"
  else
    (cd -P "${dir}" 2>/dev/null && printf '%s/%s\n' "$(pwd -P)" "${base}") || printf '%s\n' "${path}"
  fi
}

absolute_executable() {
  local executable="$1"
  local dir
  local base
  if [[ "${executable}" == */* ]]; then
    dir="$(dirname "${executable}")"
    base="$(basename "${executable}")"
    (cd -P "${dir}" 2>/dev/null && printf '%s/%s\n' "$(pwd -P)" "${base}") || return 1
  else
    command -v "${executable}"
  fi
}

write_allowed_writes() {
  local raw="${MLXFAST_OFFLINE_WRITABLE_PATHS:-}"
  local old_ifs
  local path
  if [[ -z "${raw}" ]]; then
    return 0
  fi

  old_ifs="${IFS}"
  IFS=:
  for path in ${raw}; do
    if [[ -n "${path}" ]]; then
      printf '(allow file-write* (subpath "%s"))\n' "$(sandbox_escape "$(absolute_path "${path}")")"
    fi
  done
  IFS="${old_ifs}"
}

write_strict_profile() {
  local executable="$1"
  local executable_dir
  local profile
  local workspace_root
  executable_dir="$(dirname "${executable}")"
  workspace_root="$(pwd -P)"
  profile="$(mktemp "${TMPDIR:-/tmp}/mlxfast-offline.XXXXXX")"
  {
    cat <<EOF
(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny process-exec*)
(allow process-exec (literal "$(sandbox_escape "${executable}")"))
EOF
    if [[ "${executable}" == "${workspace_root}/"* ]]; then
      printf '(allow process-exec (subpath "%s"))\n' "$(sandbox_escape "${executable_dir}")"
    fi
    cat <<EOF
(deny file-write*)
EOF
    write_allowed_writes
  } > "${profile}"
  printf '%s\n' "${profile}"
}

curl_executable="$(absolute_executable curl)"
curl_profile="$(write_strict_profile "${curl_executable}")"
if sandbox-exec -f "${curl_profile}" \
    "${curl_executable}" -fsS --max-time 10 https://example.com -o /dev/null 2>/dev/null; then
  echo "run-offline.sh: sandbox profile did not block network access; refusing to run" >&2
  exit 1
fi
rm -f "${curl_profile}"
curl_profile=""

command_executable="$(absolute_executable "$1")"
shift
command_profile="$(write_strict_profile "${command_executable}")"

echo "run-offline.sh: network egress and child process execution are blocked; running: ${command_executable} $*"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9
if [[ -n "${pending_signal}" ]]; then
  # A fatal signal arrived before the command launched. Signaling the child
  # immediately after `&` races its fork-to-exec window: the forked shell
  # still has this script's trap handler installed, so it can consume the
  # signal and then discard it at exec, leaving the command running. Never
  # start already-cancelled work: exit like a process killed by that signal
  # and let the EXIT trap remove the sandbox profiles.
  echo "run-offline.sh: received SIG${pending_signal} before command launch; aborting" >&2
  exit "$((128 + $(kill -l "${pending_signal}")))"
fi
status=0
sandbox-exec -f "${command_profile}" "${command_executable}" "$@" &
command_pid="$!"
if [[ -n "${pending_signal}" ]]; then
  forward_signal "${pending_signal}"
fi
while true; do
  if wait "${command_pid}"; then
    status=0
    break
  else
    status="$?"
  fi
  # A trapped signal interrupts wait before the forwarded signal has
  # necessarily reaped the child. Keep waiting until it actually exits.
  if ! kill -0 "${command_pid}" 2>/dev/null; then
    break
  fi
  # The interrupting signal may have raced the child's pre-exec window and
  # been swallowed there; re-forward it before waiting again.
  if [[ -n "${pending_signal}" ]]; then
    forward_signal "${pending_signal}"
  fi
done
command_pid=""
exit "${status}"
