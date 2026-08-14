#!/usr/bin/env bash
# Write a reproducible content hash of a transformed weights/ tree to stdout (or a file
# if given a second argument). The ranked benchmark.yml hashes the transform output
# with this immediately after the (untrusted) transform runs, and the gates run's
# reported weights hash is cross-checked against it before any score is trusted.
set -euo pipefail

WEIGHTS_PATH="${1:?usage: hash-weights-directory.sh WEIGHTS_PATH [OUTPUT_PATH]}"
OUTPUT_PATH="${2:-}"

if [[ ! -d "${WEIGHTS_PATH}" || -L "${WEIGHTS_PATH}" ]]; then
  echo "hash-weights-directory: ${WEIGHTS_PATH} is not a non-symlink directory" >&2
  exit 1
fi
WEIGHTS_PATH="$(cd -P "${WEIGHTS_PATH}" && pwd -P)"

XXD_BIN="$(command -v xxd 2>/dev/null || true)"
if [[ -z "${XXD_BIN}" ]]; then
  echo "hash-weights-directory: xxd is required to encode raw SHA256 digest bytes" >&2
  exit 1
fi

# Match QwenRuntime.directoryDigest exactly: for every sorted relative path,
# hash path bytes + NUL + the raw 32-byte file SHA256 + NUL.
invalid_entry="$(find "${WEIGHTS_PATH}" -mindepth 1 ! -type d ! -type f -print -quit)"
if [[ -n "${invalid_entry}" ]]; then
  echo "hash-weights-directory: weights tree contains a symlink or non-regular entry: ${invalid_entry}" >&2
  exit 1
fi
# Reject hardlinked files (link count != 1): a second name aliasing these
# bytes -- e.g. planted by the sandboxed bench uid -- must not slip through the
# transform-output gate. Mirrors overlay-editable-paths.sh and
# QwenRuntime.directoryDigest / requireSingleHardLink.
hardlinked_entry="$(find "${WEIGHTS_PATH}" -type f -links +1 -print -quit)"
if [[ -n "${hardlinked_entry}" ]]; then
  echo "hash-weights-directory: weights tree contains a hardlinked file: ${hardlinked_entry}" >&2
  exit 1
fi

hash="$(
  find "${WEIGHTS_PATH}" -type f -print0 \
    | LC_ALL=C sort -z \
    | while IFS= read -r -d '' path; do
        relative_path="${path#"${WEIGHTS_PATH}"/}"
        if [[ "${relative_path}" == '.benchmark-source.sha256' \
            || "${relative_path}" == '.gitkeep' ]]; then
          continue
        fi
        file_hash="$(shasum -a 256 "${path}" | awk '{print $1}')"
        printf '%s\0' "${relative_path}"
        printf '%s' "${file_hash}" | "${XXD_BIN}" -r -p
        printf '\0'
      done \
    | shasum -a 256 \
    | awk '{print $1}'
)"

if [[ -n "${OUTPUT_PATH}" ]]; then
  printf '%s\n' "${hash}" > "${OUTPUT_PATH}"
else
  printf '%s\n' "${hash}"
fi
