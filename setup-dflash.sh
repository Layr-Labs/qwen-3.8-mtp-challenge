#!/usr/bin/env bash
# Provision the organizer-pinned Laguna XS 2.1 DFlash DRAFTER (assistant slot).
#
# This script provisions the drafter and NOTHING else. The DFlash track's target
# IS the serial track's NVFP4 reference checkpoint (same repo, same revision,
# same fixtures/reference_laguna_xs_2_1_nvfp4_mlx.sha256 manifest), which
# ./setup.sh already downloads and verifies -- so there is no second target
# download here and no way for this script to switch the base challenge onto a
# different checkpoint.
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ASSISTANT_MODEL_ID="poolside/Laguna-XS-2.1-DFlash"
ASSISTANT_REVISION="5c36361aab23c8ed3afbd079c10c426b677bc607"
ASSISTANT_MANIFEST="${ROOT_DIR}/fixtures/dflash_laguna_xs_2_1_drafter.sha256"

DEFAULT_CACHE_ROOT="${HOME}/.cache/mlxfast/laguna-xs-2.1-dflash-v1"
CACHE_ROOT="${MLXFAST_DFLASH_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
# MLXFAST_DFLASH_DRAFTER_DIR is the name the Swift CLI actually reads
# (`dflash-benchmark`/`dflash-probe`/`dflash-reference` --drafter default);
# MLXFAST_DFLASH_ASSISTANT_DIR is the spelling the ranked workflow uses for the
# same directory and is accepted as an alias. The leaf stays `assistant` so the
# local default mirrors the runner layout
# (/opt/bench-runner/cache/dflash/laguna-xs-2.1-dflash-v1/assistant).
ASSISTANT_DIR="${MLXFAST_DFLASH_DRAFTER_DIR:-${MLXFAST_DFLASH_ASSISTANT_DIR:-${CACHE_ROOT}/assistant}}"

# The assistant slot is the BF16 DFlash draft. The pinned manifest does NOT
# describe the upstream Hugging Face blob: it describes the MLX repack produced
# by Vendor/mlx-swift-lm/scripts/convert_laguna_dflash.py (fused qkv split,
# BF16 preserved, 924137156 bytes vs upstream's 924135848). No conversion runs
# here -- the runtime feeds this directory straight to --drafter -- so the
# primary URL has to serve the REPACKED artifact for verification to pass.
# The fallback slot stays empty; an explicitly overridden primary has no
# implicit fallback.
# TODO(operator): the Hugging Face upstream below cannot satisfy the pinned
# hashes (different safetensors header). Add an organizer mirror (e.g. Darkbloom
# R2) of the repacked artifact and make it the primary; until then this script
# only succeeds with MLXFAST_DFLASH_ASSISTANT_BASE_URL pointed at that mirror,
# or against an already-provisioned verified cache.
DEFAULT_ASSISTANT_BASE_URL="https://huggingface.co/${ASSISTANT_MODEL_ID}/resolve/${ASSISTANT_REVISION}"
DEFAULT_ASSISTANT_FALLBACK_BASE_URL=""
ASSISTANT_BASE_URL="${MLXFAST_DFLASH_ASSISTANT_BASE_URL:-${DEFAULT_ASSISTANT_BASE_URL}}"
if [[ -n "${MLXFAST_DFLASH_ASSISTANT_FALLBACK_BASE_URL+x}" ]]; then
  ASSISTANT_FALLBACK_BASE_URL="${MLXFAST_DFLASH_ASSISTANT_FALLBACK_BASE_URL}"
elif [[ "${ASSISTANT_BASE_URL}" == "${DEFAULT_ASSISTANT_BASE_URL}" ]]; then
  ASSISTANT_FALLBACK_BASE_URL="${DEFAULT_ASSISTANT_FALLBACK_BASE_URL}"
else
  ASSISTANT_FALLBACK_BASE_URL=""
fi

DFLASH_APPEND_DOWNLOAD_QUERY="${MLXFAST_DFLASH_APPEND_DOWNLOAD_QUERY:-auto}"
DFLASH_DOWNLOAD_STALL_SECONDS="${MLXFAST_DFLASH_DOWNLOAD_STALL_SECONDS:-120}"
DFLASH_DOWNLOAD_MIN_BYTES_PER_SECOND="${MLXFAST_DFLASH_DOWNLOAD_MIN_BYTES_PER_SECOND:-1048576}"

VERIFY_ONLY=0
PRINT_PATHS=0

usage() {
  cat <<EOF
Usage: ./setup-dflash.sh [--verify-only] [--print-paths]

Provision the organizer-pinned Laguna XS 2.1 DFlash drafter. Downloads are
resumable and every byte is checked against the checked-in SHA256/size manifest.

The DFlash TARGET is not provisioned here: it is the same NVFP4 reference
checkpoint the serial track uses, which ./setup.sh downloads and verifies
against fixtures/reference_laguna_xs_2_1_nvfp4_mlx.sha256. Run ./setup.sh first
(without MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1), then this script.

Cache path:
  drafter: ${ASSISTANT_DIR}

Overrides:
  MLXFAST_DFLASH_CACHE_ROOT
  MLXFAST_DFLASH_DRAFTER_DIR   (alias: MLXFAST_DFLASH_ASSISTANT_DIR)
  MLXFAST_DFLASH_ASSISTANT_BASE_URL
  MLXFAST_DFLASH_ASSISTANT_FALLBACK_BASE_URL
  MLXFAST_DFLASH_APPEND_DOWNLOAD_QUERY
  MLXFAST_DFLASH_DOWNLOAD_STALL_SECONDS
  MLXFAST_DFLASH_DOWNLOAD_MIN_BYTES_PER_SECOND

This command does not alter or replace the normal base-track reference cache.
EOF
}

while (( "$#" > 0 )); do
  case "$1" in
    --verify-only)
      VERIFY_ONLY=1
      ;;
    --assistant-only|--drafter-only)
      # Accepted no-op: the drafter is now the only thing this script
      # provisions, so "only the drafter" is the unconditional behavior.
      ;;
    --print-paths)
      PRINT_PATHS=1
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "setup-dflash.sh: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "${PRINT_PATHS}" == "1" ]]; then
  # MLXFAST_DFLASH_DRAFTER_DIR is what the Swift CLI reads for --drafter, so
  # `eval "$(./setup-dflash.sh --print-paths)"` is enough to run the DFlash
  # subcommands without repeating the path.
  printf 'MLXFAST_DFLASH_DRAFTER_DIR=%q\n' "${ASSISTANT_DIR}"
  printf 'export MLXFAST_DFLASH_DRAFTER_DIR\n'
  exit 0
fi

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "setup-dflash.sh: required tool not found: $1" >&2
    exit 1
  fi
}

require_tool curl
require_tool shasum
require_tool stat

single_link_count() {
  stat -f '%l' "$1" 2>/dev/null || stat -c '%h' "$1"
}

validate_directory() {
  local path="$1"
  local label="$2"
  if [[ -L "${path}" ]]; then
    echo "setup-dflash.sh: ${label} must not be a symlink: ${path}" >&2
    return 1
  fi
  if [[ -e "${path}" && ! -d "${path}" ]]; then
    echo "setup-dflash.sh: ${label} is not a directory: ${path}" >&2
    return 1
  fi
  mkdir -p "${path}"
  if [[ -L "${path}" || ! -d "${path}" ]]; then
    echo "setup-dflash.sh: failed to create a non-symlink ${label}: ${path}" >&2
    return 1
  fi
}

validate_manifest() {
  local manifest="$1"
  local label="$2"
  local line
  local hash
  local size
  local relative
  local extra
  local entries=0

  if [[ ! -f "${manifest}" || -L "${manifest}" ]]; then
    echo "setup-dflash.sh: ${label} manifest is missing or unsafe: ${manifest}" >&2
    return 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r hash size relative extra <<< "${line}"
    if [[ -n "${extra:-}" || ! "${hash:-}" =~ ^[0-9a-f]{64}$ \
        || ! "${size:-}" =~ ^[1-9][0-9]*$ || -z "${relative:-}" ]]; then
      echo "setup-dflash.sh: malformed ${label} manifest line: ${line}" >&2
      return 1
    fi
    if [[ "${relative}" == /* || "${relative}" == *"/"* || "${relative}" == *"\\"* \
        || "${relative}" == "." || "${relative}" == ".." ]]; then
      echo "setup-dflash.sh: unsafe ${label} manifest path: ${relative}" >&2
      return 1
    fi
    entries=$((entries + 1))
  done < "${manifest}"
  if (( entries == 0 )); then
    echo "setup-dflash.sh: ${label} manifest contains no files" >&2
    return 1
  fi
}

verify_file() {
  local path="$1"
  local expected_hash="$2"
  local expected_size="$3"
  local label="$4"
  local actual_size
  local actual_hash
  local before_signature
  local after_signature

  if [[ ! -f "${path}" || -L "${path}" ]]; then
    return 1
  fi
  if [[ "$(single_link_count "${path}")" != "1" ]]; then
    echo "setup-dflash.sh: ${label} must not be hardlinked: ${path}" >&2
    return 1
  fi
  actual_size="$(wc -c < "${path}" | tr -d ' ')"
  [[ "${actual_size}" == "${expected_size}" ]] || return 1
  before_signature="$(stat -f '%d:%i:%z:%m:%c' "${path}" 2>/dev/null \
    || stat -c '%d:%i:%s:%Y:%Z' "${path}")"
  actual_hash="$(shasum -a 256 "${path}" | awk '{print $1}')"
  after_signature="$(stat -f '%d:%i:%z:%m:%c' "${path}" 2>/dev/null \
    || stat -c '%d:%i:%s:%Y:%Z' "${path}")"
  if [[ "${before_signature}" != "${after_signature}" ]]; then
    echo "setup-dflash.sh: ${label} changed while it was being verified" >&2
    return 1
  fi
  [[ "${actual_hash}" == "${expected_hash}" ]]
}

validate_download_settings() {
  local name
  local value

  for name in \
    DFLASH_DOWNLOAD_STALL_SECONDS \
    DFLASH_DOWNLOAD_MIN_BYTES_PER_SECOND; do
    value="${!name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
      echo "setup-dflash.sh: ${name} must be a positive integer" >&2
      return 1
    fi
  done
}

download_url_for_file() {
  local url="$1"
  local append_query=0
  local separator="?"

  case "${DFLASH_APPEND_DOWNLOAD_QUERY}" in
    1|true|TRUE|yes|YES)
      append_query=1
      ;;
    0|false|FALSE|no|NO)
      append_query=0
      ;;
    auto|"")
      if [[ "${url}" == https://huggingface.co/* || "${url}" == http://huggingface.co/* ]]; then
        append_query=1
      fi
      ;;
    *)
      echo "setup-dflash.sh: MLXFAST_DFLASH_APPEND_DOWNLOAD_QUERY must be auto, true, or false" >&2
      return 1
      ;;
  esac

  if [[ "${append_query}" == "1" ]]; then
    if [[ "${url}" == *\?* ]]; then
      separator="&"
    fi
    url="${url}${separator}download=true"
  fi

  printf '%s\n' "${url}"
}

download_file() {
  local base_url="$1"
  local fallback_base_url="$2"
  local relative="$3"
  local output="$4"
  local expected_hash="$5"
  local expected_size="$6"
  local label="$7"
  local partial="${output}.partial"
  local source_base_url
  local url
  local attempt
  local curl_status
  local source_index=0
  local source_count
  local base_urls=("${base_url}")

  if [[ -n "${fallback_base_url}" && "${fallback_base_url}" != "${base_url}" ]]; then
    base_urls+=("${fallback_base_url}")
  fi
  source_count="${#base_urls[@]}"

  if verify_file "${output}" "${expected_hash}" "${expected_size}" "${label}"; then
    echo "setup-dflash.sh: using verified ${label}"
    return 0
  fi
  if [[ "${VERIFY_ONLY}" == "1" ]]; then
    echo "setup-dflash.sh: ${label} is missing or does not match its pinned hash" >&2
    return 1
  fi
  if [[ -L "${partial}" ]]; then
    echo "setup-dflash.sh: refusing symlink partial download: ${partial}" >&2
    return 1
  fi
  if [[ -e "${partial}" && "$(single_link_count "${partial}")" != "1" ]]; then
    echo "setup-dflash.sh: refusing hardlinked partial download: ${partial}" >&2
    return 1
  fi

  for source_base_url in "${base_urls[@]}"; do
    source_index=$((source_index + 1))
    url="${source_base_url%/}/${relative}"
    if ! url="$(download_url_for_file "${url}")"; then
      return 1
    fi

    if [[ "${source_index}" -gt 1 ]]; then
      echo "setup-dflash.sh: trying fallback source for ${label}: ${source_base_url}"
    fi

    attempt=1
    while [[ "${attempt}" -le 2 ]]; do
      if [[ "${attempt}" == "1" ]]; then
        echo "setup-dflash.sh: downloading ${label}"
      else
        echo "setup-dflash.sh: redownloading ${label} from scratch after hash verification failed"
        rm -f "${partial}"
      fi

      curl_status=0
      curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --retry-delay 2 \
        --continue-at - \
        --speed-limit "${DFLASH_DOWNLOAD_MIN_BYTES_PER_SECOND}" \
        --speed-time "${DFLASH_DOWNLOAD_STALL_SECONDS}" \
        --output "${partial}" \
        "${url}" || curl_status=$?
      if [[ "${curl_status}" != "0" ]]; then
        echo "setup-dflash.sh: ${label} source failed or stalled (status=${curl_status}, source=${source_base_url})" >&2
        break
      fi

      if verify_file "${partial}" "${expected_hash}" "${expected_size}" "${label}"; then
        mv -f "${partial}" "${output}"
        verify_file "${output}" "${expected_hash}" "${expected_size}" "${label}"
        return
      fi

      attempt=$((attempt + 1))
    done

    if [[ "${source_index}" -lt "${source_count}" && "${curl_status}" == "0" ]]; then
      rm -f "${partial}"
    fi
  done

  echo "setup-dflash.sh: failed to download verified ${label}" >&2
  echo "setup-dflash.sh: the manifest pins the MLX-repacked drafter, not the upstream Hugging Face blob," >&2
  echo "setup-dflash.sh: so the Hugging Face default cannot satisfy it; point" >&2
  echo "setup-dflash.sh: MLXFAST_DFLASH_ASSISTANT_BASE_URL at the organizer mirror of the repacked" >&2
  echo "setup-dflash.sh: artifact, or provision the verified cache out of band." >&2
  return 1
}

provision_manifest() {
  local manifest="$1"
  local destination="$2"
  local base_url="$3"
  local fallback_base_url="$4"
  local label="$5"
  local line
  local hash
  local size
  local relative
  local extra
  local expected_files=()

  validate_manifest "${manifest}" "${label}"
  validate_directory "${destination}" "${label} cache"

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r hash size relative extra <<< "${line}"
    expected_files+=("${relative}")
    download_file \
      "${base_url}" "${fallback_base_url}" "${relative}" "${destination}/${relative}" \
      "${hash}" "${size}" "${label} ${relative}"
  done < "${manifest}"

  while IFS= read -r existing; do
    local name
    name="$(basename "${existing}")"
    if [[ "${name}" == *.partial ]]; then
      continue
    fi
    local found=0
    local expected
    for expected in "${expected_files[@]}"; do
      if [[ "${name}" == "${expected}" ]]; then
        found=1
        break
      fi
    done
    if [[ "${found}" != "1" ]]; then
      echo "setup-dflash.sh: unexpected file in ${label} cache: ${existing}" >&2
      return 1
    fi
  done < <(find "${destination}" -mindepth 1 -maxdepth 1 -print)

  echo "setup-dflash.sh: verified ${label} at ${destination}"
}

validate_download_settings

provision_manifest \
  "${ASSISTANT_MANIFEST}" "${ASSISTANT_DIR}" \
  "${ASSISTANT_BASE_URL}" "${ASSISTANT_FALLBACK_BASE_URL}" \
  "Laguna XS 2.1 DFlash drafter"

# Read the payload size back out of the manifest instead of restating it, so the
# summary can never disagree with the bytes that were actually verified.
DRAFTER_PAYLOAD_BYTES="$(
  awk '!/^#/ && $3 == "model.safetensors" { print $2; exit }' "${ASSISTANT_MANIFEST}"
)"

cat <<EOF
setup-dflash.sh: DFlash drafter ready
  drafter: ${ASSISTANT_DIR}
  drafter bytes: ${DRAFTER_PAYLOAD_BYTES:-unknown} (model.safetensors, BF16 MLX repack; no
    conversion runs at setup -- the runtime passes this directory to --drafter as is)
  target: not provisioned here; it is the serial track's NVFP4 reference
    checkpoint provisioned by ./setup.sh
  official score: disabled (fixtures/laguna_xs_2_1_dflash_track.json has
    official_scoring_enabled=false); local runs are directional only
EOF
