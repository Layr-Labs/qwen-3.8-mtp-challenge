#!/usr/bin/env bash
# Refuse artifact/cache uploads that could leak private prompts, golden tokens,
# model shards, symlinks, or unexpectedly large files.
set -euo pipefail

MAX_BYTES="${MLXFAST_MAX_ARTIFACT_BYTES:-1048576}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: deny-private-artifacts.sh PATH..." >&2
  exit 2
fi

fail() {
  local path="$1"
  local message="$2"
  if [[ "${MLXFAST_GITHUB_ANNOTATIONS:-1}" == "0" ]]; then
    echo "${path}: ${message}" >&2
  else
    echo "::error file=${path}::${message}" >&2
  fi
  exit 1
}

check_file() {
  local path="$1"
  local base
  local size

  [[ -e "${path}" ]] || return 0

  if [[ -L "${path}" ]]; then
    fail "${path}" "artifact candidate must not be a symlink"
  fi
  if [[ -d "${path}" ]]; then
    while IFS= read -r -d '' child; do
      check_file "${child}"
    done < <(find "${path}" -mindepth 1 -print0)
    return 0
  fi
  if [[ ! -f "${path}" ]]; then
    fail "${path}" "artifact candidate must be a regular file"
  fi

  base="$(basename "${path}")"
  case "${base}" in
    *correctness_golden*.json|*golden_prompt*.json|*golden_prompt*.txt|*private_prompts*.json|*gpqa_reference_cases*.json|*gpqa_ttft_results*.json|*.safetensors|*.bin|*.gguf)
      fail "${path}" "private prompt, golden, or model files must not be uploaded or cached"
      ;;
  esac

  size="$(wc -c < "${path}" | tr -d ' ')"
  if [[ ! "${size}" =~ ^[0-9]+$ ]]; then
    fail "${path}" "could not determine artifact candidate size"
  fi
  if (( size > MAX_BYTES )); then
    fail "${path}" "artifact candidate is ${size} bytes, above MLXFAST_MAX_ARTIFACT_BYTES=${MAX_BYTES}"
  fi
}

for path in "$@"; do
  check_file "${path}"
done
