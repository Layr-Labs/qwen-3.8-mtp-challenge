#!/usr/bin/env bash
# Provision the organizer-pinned Qwen 3.8 27B MTP HEAD for the native-MTP
# speculative-decode track (qwen3.8-27b-mtp-v1).
#
# This script provisions the MTP HEAD and NOTHING else. The track's TARGET is
# the same Qwen 3.8 27B 4-bit checkpoint ./setup.sh already pins, downloads and
# verifies (fixtures/reference_qwen3_8_27b_4bit.sha256) -- so there is no second
# target download here and no way for this script to switch the base challenge
# onto a different checkpoint.
#
# WHY THIS IS A WRAPPER AND NOT A DOWNLOADER. setup-dflash.sh reimplements
# ~200 lines of resumable, hash-verified, stall-detecting download logic that
# already exists in setup.sh, and the two copies have to be kept in step by
# hand. setup.sh's downloader is fully parameterised by environment
# (MLXFAST_REFERENCE_MODEL_REPO / _REVISION / _MANIFEST_PATH / _BASE_URL /
# _CACHE_DIR / _DIR / _COMPAT_LINK), so this script points those at the MTP head
# and delegates. One downloader, one verification path, one set of stall and
# resume semantics.
#
# The delegated run does NOT rebuild the Swift binaries: it passes
# MLXFAST_SKIP_SWIFT_BUILD=1, so an existing pair of products is reused. That
# rebuild was never a no-op -- SwiftPM still relinks both products, ~25 seconds
# on the ordinary `./setup.sh && ./setup-qwen-mtp.sh` path -- and nothing about
# them can have changed between the two commands. The knob fails open: on a
# fresh clone where ./setup.sh has not run, setup.sh builds anyway rather than
# leaving this script without a harness. The tool-installing legs (Homebrew,
# cmake, the Metal toolchain, macmon) and the metallib build are skipped for the
# same reason in the other direction: ./setup.sh owns those and this script must
# not mutate global state a second time.
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${ROOT_DIR}"

TRACK_ID="qwen3.8-27b-mtp-v1"

# The MTP head is a SEPARATELY pinned artifact. It is deliberately NOT part of
# the backbone manifest: the pinned backbone revision contains no mtp.* tensors
# and the transform never selects them, so target verification and head
# verification stay independent.
#
# REPINNED 2026-08-14 ONTO THE PUBLISHED 3.8 HEAD, in the same commit as
# fixtures/qwen3_8_27b_mtp_head.sha256 and the workflow env, exactly as the
# pending note that stood here required. These defaults were the Qwen 3.6 head
# for as long as the 3.8 upload was outstanding, because a placeholder here
# would have broken `./setup-qwen-mtp.sh` for everyone without gating anything
# ranked; that reason is spent now that the head resolves. Leaving them at 3.6
# would be worse than a placeholder ever was: the manifest this script verifies
# against pins the 3.8 tree, so a 3.6 download would now fail on a digest
# mismatch that reads as a corrupt fetch.
#
# The local pair is CONSISTENT again as a result. It was mismatched while these
# defaults lagged -- merging the 3.6 4-bit head (31 tensors) onto the 3.8
# backbone failed on MLXFastConstants.qwenMTPHeadTensorCount, which is 15 --
# and both halves of the checkpoint are published now, so ./setup.sh and this
# script resolve against the same 3.8 pair.
#
# THE HEAD REPOSITORY IS PUBLIC. An unauthenticated fetch of
# EigenLabs/Qwen3.8-27B-MTP-bf16 resolves, so no Hugging Face token and no
# login are needed for the normal path. Credentials remain OPTIONAL, through
# setup.sh's MLXFAST_REFERENCE_AUTH_HEADER, for a private mirror or a
# rate-limited anonymous fetch; alternatively stage the four files into
# MLXFAST_QWEN_MTP_HEAD_DIR and run --verify-only. Both variables stay
# env-overridable either way.
MTP_HEAD_MODEL_ID="${MLXFAST_QWEN_MTP_HEAD_REPO:-EigenLabs/Qwen3.8-27B-MTP-bf16}"
MTP_HEAD_REVISION="${MLXFAST_QWEN_MTP_HEAD_REVISION:-26a328e070875b0314d652a039b6b59902690f03}"
MTP_HEAD_MANIFEST="${MLXFAST_QWEN_MTP_HEAD_MANIFEST_PATH:-${ROOT_DIR}/fixtures/qwen3_8_27b_mtp_head.sha256}"

DEFAULT_CACHE_ROOT="${HOME}/.cache/mlxfast/${TRACK_ID}"
CACHE_ROOT="${MLXFAST_QWEN_MTP_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
# MLXFAST_QWEN_MTP_HEAD_DIR is the name both the Swift CLI (--mtp-head default)
# and the ranked workflow read for this directory. The leaf stays `mtp-head` so
# the local default mirrors the runner layout
# (/opt/bench-runner/cache/qwen38/qwen3.8-27b-mtp-v1/mtp-head).
MTP_HEAD_DIR="${MLXFAST_QWEN_MTP_HEAD_DIR:-${CACHE_ROOT}/mtp-head}"

DEFAULT_MTP_HEAD_BASE_URL="https://huggingface.co/${MTP_HEAD_MODEL_ID}/resolve/${MTP_HEAD_REVISION}"
MTP_HEAD_BASE_URL="${MLXFAST_QWEN_MTP_HEAD_BASE_URL:-${DEFAULT_MTP_HEAD_BASE_URL}}"
# An explicitly overridden primary has no implicit fallback (same rule setup.sh
# and setup-dflash.sh follow): a mirror the operator chose must not silently
# fall back to a source they did not.
if [[ -n "${MLXFAST_QWEN_MTP_HEAD_FALLBACK_BASE_URL+x}" ]]; then
  MTP_HEAD_FALLBACK_BASE_URL="${MLXFAST_QWEN_MTP_HEAD_FALLBACK_BASE_URL}"
else
  MTP_HEAD_FALLBACK_BASE_URL=""
fi

VERIFY_ONLY=0
PRINT_PATHS=0

usage() {
  cat <<EOF
Usage: ./setup-qwen-mtp.sh [--verify-only] [--print-paths]

Provision the organizer-pinned Qwen 3.8 27B MTP head. The download is anonymous
(the pinned head repository is public), resumable, and every byte is checked
against the checked-in SHA256/size manifest, because the download itself is
./setup.sh's downloader driven at this artifact.

The TARGET is not provisioned here: it is the Qwen 3.8 27B reference checkpoint
./setup.sh downloads and verifies against
fixtures/reference_qwen3_8_27b_4bit.sha256. Run ./setup.sh first (without
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1), then this script.

Pinned artifact:
  repository ${MTP_HEAD_MODEL_ID}
  revision   ${MTP_HEAD_REVISION}
  manifest   ${MTP_HEAD_MANIFEST}

Cache path:
  mtp head:  ${MTP_HEAD_DIR}

Overrides:
  MLXFAST_QWEN_MTP_CACHE_ROOT
  MLXFAST_QWEN_MTP_HEAD_DIR
  MLXFAST_QWEN_MTP_HEAD_REPO
  MLXFAST_QWEN_MTP_HEAD_REVISION
  MLXFAST_QWEN_MTP_HEAD_MANIFEST_PATH
  MLXFAST_QWEN_MTP_HEAD_BASE_URL
  MLXFAST_QWEN_MTP_HEAD_FALLBACK_BASE_URL

This command does not alter or replace the normal base-track reference cache.
EOF
}

while (( "$#" > 0 )); do
  case "$1" in
    --verify-only)
      VERIFY_ONLY=1
      ;;
    --head-only|--mtp-head-only)
      # Accepted no-op: the head is the only thing this script provisions, so
      # "only the head" is the unconditional behavior.
      ;;
    --print-paths)
      PRINT_PATHS=1
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "setup-qwen-mtp.sh: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "${PRINT_PATHS}" == "1" ]]; then
  # MLXFAST_QWEN_MTP_HEAD_DIR is what the Swift CLI reads for --mtp-head, so
  # `eval "$(./setup-qwen-mtp.sh --print-paths)"` is enough to run the MTP
  # subcommands without repeating the path. Printed BEFORE any manifest or
  # network check so the local benchmark runner can learn the path even while
  # the artifact is still being provisioned.
  printf 'MLXFAST_QWEN_MTP_HEAD_DIR=%q\n' "${MTP_HEAD_DIR}"
  printf 'export MLXFAST_QWEN_MTP_HEAD_DIR\n'
  exit 0
fi

if [[ ! -x "${ROOT_DIR}/setup.sh" ]]; then
  echo "setup-qwen-mtp.sh: ${ROOT_DIR}/setup.sh is missing or not executable;" >&2
  echo "setup-qwen-mtp.sh: this script delegates its downloader rather than reimplementing it" >&2
  exit 1
fi

# FAIL CLOSED on the pending manifest. QWEN-MTP-PENDING-ORGANIZER: the head is
# byte-verified on the staging box but its manifest is not checked in yet (phase
# 2 of QWEN36-MTP-CHALLENGE-PLAN.md). Downloading an unpinned speculative head
# would be worse than not downloading one: the whole point of the pin is that a
# participant cannot substitute, re-derive or re-quantize it.
if [[ ! -s "${MTP_HEAD_MANIFEST}" ]]; then
  cat >&2 <<EOF
setup-qwen-mtp.sh: QWEN-MTP-PENDING-ORGANIZER

The pinned byte manifest for the MTP head does not exist:

  ${MTP_HEAD_MANIFEST}

The head (${MTP_HEAD_MODEL_ID} @ ${MTP_HEAD_REVISION}) is byte-verified on the
staging box, but until its manifest is checked in there is nothing to verify a
download against -- and an UNPINNED speculative head is exactly what this
track's rules exclude. Land the manifest first; this script will then provision
and verify the head like any other pinned artifact.
EOF
  exit 1
fi

# --- delegate ---------------------------------------------------------------
# setup.sh is driven at the head instead of the backbone. Note
# MLXFAST_REFERENCE_COMPAT_LINK: without it the delegated run would repoint the
# shared reference_weights/ compatibility symlink at the head and every later
# transform would read the wrong checkpoint.
delegated_env=(
  MLXFAST_REFERENCE_MODEL_REPO="${MTP_HEAD_MODEL_ID}"
  MLXFAST_REFERENCE_REVISION="${MTP_HEAD_REVISION}"
  MLXFAST_REFERENCE_MANIFEST_PATH="${MTP_HEAD_MANIFEST}"
  MLXFAST_REFERENCE_BASE_URL="${MTP_HEAD_BASE_URL}"
  MLXFAST_REFERENCE_FALLBACK_BASE_URL="${MTP_HEAD_FALLBACK_BASE_URL}"
  MLXFAST_REFERENCE_DIR="${MTP_HEAD_DIR}"
  MLXFAST_REFERENCE_COMPAT_LINK="${ROOT_DIR}/reference_weights/Qwen3.6-27B-MTP-4bit"
  # The head is ~247 MB, not ~16 GB: the backbone's free-space floor would
  # refuse on machines that can comfortably hold it.
  MLXFAST_REFERENCE_MIN_FREE_GIB="${MLXFAST_QWEN_MTP_HEAD_MIN_FREE_GIB:-2}"
  # The closing summary must not tell this reader to transform REFERENCE_DIR
  # into weights/: REFERENCE_DIR is the MTP head for the delegated run, and
  # transforming a 15-tensor head over the target weights is exactly the wrong
  # next step. Prose only -- nothing about the download or verification changes.
  MLXFAST_SETUP_SUMMARY_ROLE=mtp-head
  # Reuse the Swift products ./setup.sh already built instead of relinking both
  # of them for a download. Fails open in setup.sh: if a product is missing (a
  # standalone run on a fresh clone) the build happens anyway.
  MLXFAST_SKIP_SWIFT_BUILD=1
  # ./setup.sh owns global tool state and the metallib; do not mutate either a
  # second time from here.
  MLXFAST_SKIP_HOMEBREW_INSTALL=1
  MLXFAST_SKIP_CMAKE_INSTALL=1
  MLXFAST_SKIP_METAL_TOOLCHAIN_INSTALL=1
  MLXFAST_SKIP_MACMON_INSTALL=1
  MLXFAST_SKIP_MLX_METALLIB=1
)

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  if [[ ! -d "${MTP_HEAD_DIR}" ]]; then
    echo "setup-qwen-mtp.sh: no MTP head cache at ${MTP_HEAD_DIR}; run ./setup-qwen-mtp.sh first" >&2
    exit 1
  fi
  delegated_env+=(MLXFAST_SKIP_WEIGHTS_DOWNLOAD=0)
  echo "setup-qwen-mtp.sh: verifying the pinned MTP head at ${MTP_HEAD_DIR}" >&2
else
  echo "setup-qwen-mtp.sh: provisioning the pinned MTP head into ${MTP_HEAD_DIR}" >&2
fi

env "${delegated_env[@]}" "${ROOT_DIR}/setup.sh"

if [[ ! -s "${MTP_HEAD_DIR}/config.json" ]]; then
  echo "setup-qwen-mtp.sh: the delegated run left no ${MTP_HEAD_DIR}/config.json; the MTP head is not usable" >&2
  exit 1
fi

echo "setup-qwen-mtp.sh: MTP head ready at ${MTP_HEAD_DIR} (${MTP_HEAD_MODEL_ID} @ ${MTP_HEAD_REVISION})"
