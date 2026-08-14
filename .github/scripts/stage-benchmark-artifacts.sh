#!/usr/bin/env bash
# Copy already-validated benchmark outputs into a single upload directory.
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: stage-benchmark-artifacts.sh DEST NAME=PATH..." >&2
  exit 2
fi

dest="$1"
shift
artifact_base=""
artifact_run_root=""

validate_artifact_destination() {
  local candidate="$1"
  local prefix="$2"
  local remainder
  local run_component
  local leaf

  [[ "${candidate}" == "${prefix}"* ]] || return 1
  remainder="${candidate#"${prefix}"}"
  [[ "${remainder}" == */* ]] || return 1
  run_component="${remainder%%/*}"
  leaf="${remainder#*/}"
  [[ -n "${run_component}" && -n "${leaf}" ]] || return 1
  [[ "${run_component}" != "." && "${run_component}" != ".." ]] || return 1
  [[ "${leaf}" != "." && "${leaf}" != ".." && "${leaf}" != */* ]] || return 1
  [[ "${candidate}" != *$'\n'* && "${candidate}" != *$'\r'* ]] || return 1
  artifact_base="${prefix%/mlxfast-artifacts-}"
  artifact_run_root="${prefix}${run_component}"
}

if ! validate_artifact_destination "${dest}" "/tmp/mlxfast-artifacts-" \
    && ! validate_artifact_destination "${dest}" "${RUNNER_TEMP:-/tmp}/mlxfast-artifacts-"; then
  echo "artifact destination must be one direct child of /tmp/mlxfast-artifacts-RUN or RUNNER_TEMP/mlxfast-artifacts-RUN: ${dest}" >&2
  exit 2
fi

canonical_artifact_base="$(cd -P "${artifact_base}" 2>/dev/null && pwd -P)" || {
  echo "artifact base directory does not exist: ${artifact_base}" >&2
  exit 2
}
if [[ -L "${artifact_run_root}" ]]; then
  echo "artifact run root must not be a symlink: ${artifact_run_root}" >&2
  exit 2
fi
if [[ -e "${artifact_run_root}" ]]; then
  if [[ ! -d "${artifact_run_root}" ]]; then
    echo "artifact run root is not a directory: ${artifact_run_root}" >&2
    exit 2
  fi
else
  umask 077
  mkdir "${artifact_run_root}" || {
    echo "could not create artifact run root: ${artifact_run_root}" >&2
    exit 1
  }
fi
canonical_artifact_run_root="$(cd -P "${artifact_run_root}" 2>/dev/null && pwd -P)" || {
  echo "could not resolve artifact run root: ${artifact_run_root}" >&2
  exit 2
}
if [[ "$(dirname "${canonical_artifact_run_root}")" != "${canonical_artifact_base}" \
    || "$(basename "${canonical_artifact_run_root}")" != "$(basename "${artifact_run_root}")" ]]; then
  echo "artifact run root escaped its allowed base: ${artifact_run_root}" >&2
  exit 2
fi
if [[ ! -O "${artifact_run_root}" ]]; then
  echo "artifact run root is not owned by the current user: ${artifact_run_root}" >&2
  exit 2
fi
chmod 0700 "${artifact_run_root}"

rm -rf -- "${dest}"
mkdir "${dest}"

for mapping in "$@"; do
  case "${mapping}" in
    *=*) ;;
    *)
      echo "artifact mapping must be NAME=PATH, got ${mapping}" >&2
      exit 2
      ;;
  esac

  name="${mapping%%=*}"
  source_path="${mapping#*=}"

  if [[ "${name}" == */* || "${name}" == "." || "${name}" == ".." || -z "${name}" ]]; then
    echo "artifact name must be a simple filename, got ${name}" >&2
    exit 2
  fi
  if [[ ! -f "${source_path}" || -L "${source_path}" ]]; then
    echo "artifact source must be a regular non-symlink file: ${source_path}" >&2
    exit 1
  fi

  # `--` terminates option parsing so a source path that begins with '-'
  # cannot be reinterpreted as a cp flag. The mappings here are trusted
  # workflow constants today; this keeps the copy safe if that ever changes.
  cp -- "${source_path}" "${dest}/${name}"
done

.github/scripts/deny-private-artifacts.sh "${dest}"
echo "benchmark: staged artifacts in ${dest}"
