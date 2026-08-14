#!/usr/bin/env bash
# Download a small, manifest-verified subset of the Poolside Laguna XS 2.1 NVFP4
# reference checkpoint for cache experiments. The full checkpoint path
# intentionally stays in setup.sh so full downloads keep the same
# lock/verification behavior.
set -euo pipefail

SCOPE="${1:-}"
REFERENCE_DIR="${MLXFAST_REFERENCE_DIR:-.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4-mlx/snapshots/841778bda563a36104dd521e37d99218e46f4f25}"
DEFAULT_REFERENCE_BASE_URL="https://ds4.darkbloom.ai/laguna-xs-2.1-nvfp4-mlx"
DEFAULT_REFERENCE_FALLBACK_BASE_URL="https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4-mlx/resolve/841778bda563a36104dd521e37d99218e46f4f25"
REFERENCE_BASE_URL="${MLXFAST_REFERENCE_BASE_URL:-${DEFAULT_REFERENCE_BASE_URL}}"
if [[ -n "${MLXFAST_REFERENCE_FALLBACK_BASE_URL+x}" ]]; then
  REFERENCE_FALLBACK_BASE_URL="${MLXFAST_REFERENCE_FALLBACK_BASE_URL}"
elif [[ "${REFERENCE_BASE_URL}" == "${DEFAULT_REFERENCE_BASE_URL}" ]]; then
  REFERENCE_FALLBACK_BASE_URL="${DEFAULT_REFERENCE_FALLBACK_BASE_URL}"
else
  REFERENCE_FALLBACK_BASE_URL=""
fi
REFERENCE_AUTH_HEADER="${MLXFAST_REFERENCE_AUTH_HEADER:-}"
REFERENCE_APPEND_DOWNLOAD_QUERY="${MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY:-auto}"
REFERENCE_MANIFEST_PATH="${MLXFAST_REFERENCE_MANIFEST_PATH:-fixtures/reference_laguna_xs_2_1_nvfp4_mlx.sha256}"

usage() {
  cat >&2 <<EOF
usage: download-reference-cache-scope.sh metadata|first-shard|two-shards

Downloads and verifies only the selected reference checkpoint files.
Use setup.sh for full-checkpoint downloads.
EOF
}

case "${SCOPE}" in
  metadata|first-shard|two-shards) ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
  echo "cache-probe: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
  exit 1
fi

download_url_for_file() {
  local url="$1"
  local append_query=0
  local separator="?"

  case "${REFERENCE_APPEND_DOWNLOAD_QUERY}" in
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
      echo "cache-probe: MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY must be auto, true, or false" >&2
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

selected_files() {
  local shard_limit=0
  case "${SCOPE}" in
    metadata) shard_limit=0 ;;
    first-shard) shard_limit=1 ;;
    two-shards) shard_limit=2 ;;
  esac

  awk -v shard_limit="${shard_limit}" '
    NF >= 3 && $1 !~ /^#/ {
      if ($3 ~ /\.safetensors$/) {
        if (shards < shard_limit) {
          print $3
          shards++
        }
      } else {
        print $3
      }
    }
  ' "${REFERENCE_MANIFEST_PATH}"
}

manifest_entry() {
  local relative_path="$1"
  awk -v path="${relative_path}" '
    NF >= 3 && $1 !~ /^#/ && $3 == path {
      print $1 " " $2
      found=1
      exit
    }
    END {
      if (!found) {
        exit 1
      }
    }
  ' "${REFERENCE_MANIFEST_PATH}"
}

file_is_current() {
  local relative_path="$1"
  local output_path="$2"
  local expected_hash
  local expected_size
  local actual_hash
  local actual_size

  read -r expected_hash expected_size <<< "$(manifest_entry "${relative_path}")"
  [[ -f "${output_path}" ]] || return 1
  actual_size="$(wc -c < "${output_path}" | tr -d ' ')"
  [[ "${actual_size}" == "${expected_size}" ]] || return 1
  actual_hash="$(shasum -a 256 "${output_path}" | awk '{print $1}')"
  [[ "${actual_hash}" == "${expected_hash}" ]]
}

download_file() {
  local relative_path="$1"
  local output_path="${REFERENCE_DIR}/${relative_path}"
  local base_url
  local url
  local attempt
  local curl_status
  local source_index=0
  local source_count
  local base_urls=("${REFERENCE_BASE_URL}")

  if file_is_current "${relative_path}" "${output_path}"; then
    echo "cache-probe: using cached ${relative_path}"
    return 0
  fi

  if [[ -n "${REFERENCE_FALLBACK_BASE_URL}" && "${REFERENCE_FALLBACK_BASE_URL}" != "${REFERENCE_BASE_URL}" ]]; then
    base_urls+=("${REFERENCE_FALLBACK_BASE_URL}")
  fi
  source_count="${#base_urls[@]}"
  mkdir -p "$(dirname "${output_path}")"

  for base_url in "${base_urls[@]}"; do
    source_index=$((source_index + 1))
    url="$(download_url_for_file "${base_url%/}/${relative_path}")"
    if [[ "${source_index}" -gt 1 ]]; then
      echo "cache-probe: trying fallback source for ${relative_path}: ${base_url}"
    fi

    attempt=1
    while [[ "${attempt}" -le 2 ]]; do
      if [[ "${attempt}" == "1" ]]; then
        echo "cache-probe: downloading ${relative_path}"
      else
        echo "cache-probe: redownloading ${relative_path} after verification failed"
        rm -f "${output_path}"
      fi

      curl_status=0
      if [[ "${source_index}" == "1" && -n "${REFERENCE_AUTH_HEADER}" ]]; then
        curl \
          --fail \
          --location \
          --retry 5 \
          --retry-all-errors \
          --retry-delay 2 \
          --continue-at - \
          --silent \
          --show-error \
          -H "${REFERENCE_AUTH_HEADER}" \
          --output "${output_path}" \
          "${url}" || curl_status=$?
      else
        curl \
          --fail \
          --location \
          --retry 5 \
          --retry-all-errors \
          --retry-delay 2 \
          --continue-at - \
          --silent \
          --show-error \
          --output "${output_path}" \
          "${url}" || curl_status=$?
      fi

      if [[ "${curl_status}" != "0" ]]; then
        echo "cache-probe: source failed (status=${curl_status}, source=${base_url})" >&2
        break
      fi
      if file_is_current "${relative_path}" "${output_path}"; then
        echo "cache-probe: verified ${relative_path}"
        return 0
      fi

      attempt=$((attempt + 1))
    done

    if [[ "${source_index}" -lt "${source_count}" && "${curl_status}" == "0" ]]; then
      rm -f "${output_path}"
    fi
  done

  echo "cache-probe: failed to download verified ${relative_path}" >&2
  return 1
}

downloaded=0
while IFS= read -r file; do
  [[ -n "${file}" ]] || continue
  download_file "${file}"
  downloaded=$((downloaded + 1))
done < <(selected_files)

echo "cache-probe: downloaded ${downloaded} file(s) for scope ${SCOPE}"
