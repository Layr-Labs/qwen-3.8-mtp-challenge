#!/usr/bin/env bash
# Bootstrap system tools and build the Swift-only Qwen3.8 harness.
set -euo pipefail

# REPINNED 2026-08-14 onto the Qwen 3.8 backbone, in lockstep with
# MLXFastConstants.referenceModelRepository / _Revision and the ranked
# workflow's MLXFAST_QWEN_MTP_CHECKPOINT_REPO / _REVISION. This is OUR OWN MLX
# 4-bit affine / group_size 64 conversion, produced under a pinned mlx 0.32.0
# toolchain, of the official bf16 base Qwen/Qwen3.8-27B @
# 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. It replaces a third-party
# personal-account conversion adopted earlier the same day and terminated by
# the operator's validation kill-switch (994 of 1,847 tensors differed
# numerically from our reconversion); the termination is recorded in
# fixtures/qwen3_8_27b_mtp_track.json's target.upstream_source_note.
#
# THE DOWNLOAD IS LIVE AGAIN as of the 2026-08-14 publish. The revision below
# is the published commit sha, and fixtures/reference_qwen3_8_27b_4bit.sha256
# now carries a generated per-file body, so the fetch resolves and is
# hash-verified against real records. Both were blocked together and both
# resolve together: while the revision was a pending marker the URL could not
# resolve, and while the manifest was a header-only stub it verified nothing.
# The repository is PRIVATE, so a fetch needs credentials --
# MLXFAST_REFERENCE_AUTH_HEADER carries them. Both variables stay
# env-overridable so a locally staged copy can still be used without editing
# this file.
REFERENCE_MODEL_REPO="${MLXFAST_REFERENCE_MODEL_REPO:-EigenLabs/Qwen3.8-27B-4bit}"
REFERENCE_REVISION="${MLXFAST_REFERENCE_REVISION:-eda45ab47f465d08d6558f0353a2346e2eb9d5b3}"
REFERENCE_CACHE_REPO_DIR="models--${REFERENCE_MODEL_REPO//\//--}"
REFERENCE_CACHE_REVISION_DIR="${REFERENCE_REVISION//\//--}"
# No organizer-hosted mirror exists for this checkpoint yet (the Darkbloom R2
# mirror serves the retired Laguna target only). Resolve the immutable Hugging
# Face revision directly and leave the fallback empty by default.
# download_url_for_file appends "?download=true" only for huggingface.co URLs,
# so that resolves to the raw LFS/Xet bytes.
DEFAULT_REFERENCE_BASE_URL="https://huggingface.co/EigenLabs/Qwen3.8-27B-4bit/resolve/${REFERENCE_REVISION}"
DEFAULT_REFERENCE_FALLBACK_BASE_URL=""
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
REFERENCE_MANIFEST_PATH="${MLXFAST_REFERENCE_MANIFEST_PATH:-fixtures/reference_qwen3_8_27b_4bit.sha256}"
REFERENCE_HASH_VERIFY="${MLXFAST_REFERENCE_HASH_VERIFY:-1}"
REFERENCE_POST_DOWNLOAD_FULL_VERIFY="${MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY:-1}"
REFERENCE_MIN_FREE_GIB="${MLXFAST_REFERENCE_MIN_FREE_GIB:-40}"
# The pinned checkpoint has three safetensors shards; three jobs gives each
# shard one download stream. Override with MLXFAST_REFERENCE_DOWNLOAD_JOBS.
REFERENCE_DOWNLOAD_JOBS="${MLXFAST_REFERENCE_DOWNLOAD_JOBS:-3}"
REFERENCE_DOWNLOAD_PROGRESS_SECONDS="${MLXFAST_REFERENCE_DOWNLOAD_PROGRESS_SECONDS:-15}"
REFERENCE_DOWNLOAD_STALL_SECONDS="${MLXFAST_REFERENCE_DOWNLOAD_STALL_SECONDS:-120}"
REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND="${MLXFAST_REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND:-1048576}"
# macmon is optional (it only powers benchmark.sh's local GPU cool-down gate).
# When missing it is installed as a single pinned, hash-verified release
# binary dropped in ~/bin -- the same location the ranked boxes use and one
# benchmark.sh's gate lookup already searches -- never through Homebrew, so a
# normal setup run cannot mutate global Homebrew state (taps, dependencies,
# upgrades) for an optional tool. MLXFAST_SKIP_MACMON_INSTALL=1 skips it.
MACMON_VERSION="0.7.2"
MACMON_RELEASE_URL="https://github.com/vladkens/macmon/releases/download/v${MACMON_VERSION}/macmon-v${MACMON_VERSION}.tar.gz"
MACMON_RELEASE_SHA256="c79fdc7ab02b456b897dcc3ea041678420d7a1d5bd669aaac36fda3885572ad6"
MACMON_INSTALL_DIR="${HOME}/bin"
SETUP_PARALLEL_METALLIB="${MLXFAST_SETUP_PARALLEL_METALLIB:-${MLXFAST_SETUP_PARALLEL_BUILD:-1}}"
SWIFT_BIN="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
# The participant runtime worker builds under its own SwiftPM scratch root
# (.build-worker) so a participant-code build never writes into the trusted
# CLI's build tree (.build). mlx.metallib is a participant artifact and lives
# next to the worker binary, where Cmlx searches first.
RUNTIME_WORKER_BIN="${MLXFAST_RUNTIME_WORKER_EXECUTABLE:-.build-worker/release/mlxfast-runtime-worker}"
MLX_METALLIB="${MLXFAST_MLX_METALLIB:-$(dirname "${RUNTIME_WORKER_BIN}")/mlx.metallib}"
DEFAULT_REFERENCE_DIR="reference_weights/Qwen3.8-27B-4bit"
DEFAULT_HF_HOME="${MLXFAST_HF_HOME:-${HF_HOME:-${HOME:-${PWD}}/.cache/huggingface}}"
DEFAULT_HF_HUB_CACHE="${MLXFAST_HF_HUB_CACHE:-${HF_HUB_CACHE:-${DEFAULT_HF_HOME}/hub}}"
REFERENCE_CACHE_DIR="${MLXFAST_REFERENCE_CACHE_DIR:-${DEFAULT_HF_HUB_CACHE}/${REFERENCE_CACHE_REPO_DIR}/snapshots/${REFERENCE_CACHE_REVISION_DIR}}"
if [[ -n "${MLXFAST_REFERENCE_DIR:-}" ]]; then
  REFERENCE_DIR="${MLXFAST_REFERENCE_DIR}"
elif [[ -e "${DEFAULT_REFERENCE_DIR}" && ! -L "${DEFAULT_REFERENCE_DIR}" ]]; then
  REFERENCE_DIR="${DEFAULT_REFERENCE_DIR}"
else
  REFERENCE_DIR="${REFERENCE_CACHE_DIR}"
fi
REFERENCE_COMPAT_LINK="${MLXFAST_REFERENCE_COMPAT_LINK:-${DEFAULT_REFERENCE_DIR}}"
# This is a verification stamp, retained under its original environment-variable
# name for compatibility. It is not the inter-process mutation lock below.
REFERENCE_CACHE_LOCK_PATH="${MLXFAST_REFERENCE_CACHE_LOCK_PATH:-${REFERENCE_DIR}/.mlxfast-reference-cache.lock}"
if [[ -d "${REFERENCE_DIR}" ]]; then
  CANONICAL_REFERENCE_LOCK_BASE="$(cd -P "${REFERENCE_DIR}" && pwd -P)"
else
  REFERENCE_LOCK_PARENT="$(dirname "${REFERENCE_DIR}")"
  REFERENCE_LOCK_NAME="$(basename "${REFERENCE_DIR}")"
  CANONICAL_REFERENCE_LOCK_PARENT="$(cd -P "${REFERENCE_LOCK_PARENT}" 2>/dev/null && pwd -P)" \
    || CANONICAL_REFERENCE_LOCK_PARENT="${REFERENCE_LOCK_PARENT}"
  CANONICAL_REFERENCE_LOCK_BASE="${CANONICAL_REFERENCE_LOCK_PARENT}/${REFERENCE_LOCK_NAME}"
fi
REFERENCE_CACHE_MUTATION_LOCK_DIR="${MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR:-${CANONICAL_REFERENCE_LOCK_BASE}.mlxfast-setup.lock}"
REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS="${MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS:-1800}"
REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS="${MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS:-60}"
SETUP_STARTED_SECONDS="${SECONDS}"
METALLIB_BUILD_PID=""
METALLIB_BUILD_PROCESS_GROUP=""
METALLIB_BUILD_STATE="not_started"
REFERENCE_CACHE_MUTATION_LOCK_HELD=0
REFERENCE_DOWNLOAD_HEARTBEAT_PID=""
REFERENCE_CACHE_MUTATION_LOCK_TOKEN=""
REFERENCE_CACHE_MUTATION_OWNER_TEMP=""
REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR=""
REFERENCE_CACHE_MUTATION_LOCK_INITIALIZING_DIR=""
CURRENT_SHELL_PID=""
REFERENCE_VERIFIED_SIGNATURES_PATH=""
REFERENCE_VERIFIED_MANIFEST_PATH=""
REFERENCE_VERIFIED_MANIFEST_HASH=""
REFERENCE_VERIFIED_CONTENT_IDENTITY=""
REFERENCE_VERIFIED_EXPECTED_HASH=""
REFERENCE_VERIFIED_EXPECTED_SIZE=""
REFERENCE_VERIFIED_FILE_MANIFEST_HASH=""
REFERENCE_REQUIRED_METADATA_FILES=(
  ".gitattributes"
  "LICENSE.md"
  "README.md"
  "chat_template.jinja"
  "config.json"
  "model.safetensors.index.json"
  "tokenizer.json"
  "tokenizer_config.json"
)

print_help() {
  cat <<EOF
Usage: ./setup.sh

Checks the local macOS/Apple Silicon toolchain, builds the Swift harness,
builds mlx.metallib, and downloads the Poolside Laguna XS 2.1 NVFP4 reference
checkpoint when it is not already present.

Important environment variables:
  MLXFAST_REFERENCE_DIR              Reference checkpoint directory.
                                     Default: ${REFERENCE_DIR}
  MLXFAST_REFERENCE_CACHE_DIR        Shared Hugging Face-style cache path (under
                                     $HOME by default, so parallel clones reuse
                                     one checkpoint) used for new downloads when
                                     MLXFAST_REFERENCE_DIR is not set.
                                     Default: ${REFERENCE_CACHE_DIR}
  MLXFAST_REFERENCE_BASE_URL         HTTP prefix for checkpoint files.
                                     Default: ${DEFAULT_REFERENCE_BASE_URL}
  MLXFAST_REFERENCE_FALLBACK_BASE_URL
                                     Fallback HTTP prefix after a failed or
                                     stalled primary download. Set empty to
                                     disable. Default with the built-in mirror:
                                     ${DEFAULT_REFERENCE_FALLBACK_BASE_URL}
  MLXFAST_REFERENCE_MANIFEST_PATH    SHA256 manifest for the reference files.
                                     Default: ${REFERENCE_MANIFEST_PATH}
  MLXFAST_REFERENCE_CACHE_LOCK_PATH  Local stamp proving the checkpoint was
                                     fully verified by this manifest.
                                     Default: ${REFERENCE_CACHE_LOCK_PATH}
  MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR
                                     Inter-process lock directory serializing
                                     shared-cache downloads and repairs.
                                     Default: ${REFERENCE_CACHE_MUTATION_LOCK_DIR}
  MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS
                                     Maximum time to wait for another setup.
                                     Default: ${REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS}
  MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS
                                     Age for reclaiming ownerless lock dirs.
                                     Default: ${REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS}
  MLXFAST_REFERENCE_DOWNLOAD_JOBS    Parallel safetensors downloads.
                                     Default: ${REFERENCE_DOWNLOAD_JOBS}
  MLXFAST_REFERENCE_DOWNLOAD_PROGRESS_SECONDS
                                     Download progress heartbeat interval.
                                     Default: ${REFERENCE_DOWNLOAD_PROGRESS_SECONDS}
  MLXFAST_REFERENCE_DOWNLOAD_STALL_SECONDS
                                     Abort a transfer that remains below the
                                     minimum rate for this long, then fall back.
                                     Default: ${REFERENCE_DOWNLOAD_STALL_SECONDS}
  MLXFAST_REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND
                                     Minimum sustained per-transfer byte rate.
                                     Default: ${REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND}
  MLXFAST_REFERENCE_MIN_FREE_GIB     Required free space before download.
                                     Default: ${REFERENCE_MIN_FREE_GIB}
  MLXFAST_REFERENCE_HASH_VERIFY=0    Skip reference SHA256 verification.
  MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY=0
                                     Skip the second full-checkpoint SHA256 pass
                                     after all downloaded files were already
                                     verified by size and hash. CI-only speedup.
  MLXFAST_SETUP_PARALLEL_METALLIB=0  Disable overlapping the Metal library build
                                     with reference checkpoint download.
                                     MLXFAST_SETUP_PARALLEL_BUILD is accepted
                                     as a deprecated alias.
  MLXFAST_SWIFT_BIN                  Trusted user-facing Swift CLI.
                                     Default: ${SWIFT_BIN}
  MLXFAST_RUNTIME_WORKER_EXECUTABLE  Participant worker executable.
                                     Default: ${RUNTIME_WORKER_BIN}
  MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1    Build tools only; do not download weights.
  MLXFAST_SKIP_MLX_METALLIB=1        Skip mlx.metallib build.
  MLXFAST_SKIP_MACMON_INSTALL=1      Do not install macmon (the GPU temperature
                                     reader benchmark.sh's local cool-down gate
                                     uses; the gate is skipped when absent).
                                     When not skipped, macmon is installed as a
                                     pinned, hash-verified release binary in
                                     ~/bin; setup.sh never installs it through
                                     Homebrew.

After setup:
  MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" .github/scripts/run-offline.sh .build/release/mlxfast-swift transform --output weights
  ./benchmark.sh
EOF
}

if [[ "$#" -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      print_help
      exit 0
      ;;
    *)
      echo "setup.sh: unknown argument '$1'" >&2
      echo "Run ./setup.sh --help for usage." >&2
      exit 2
      ;;
  esac
fi

format_duration() {
  local total_seconds="${1:-0}"
  printf '%02d:%02d:%02d' \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

path_size_gib() {
  local path="$1"
  local size_kib

  if [[ ! -e "${path}" ]]; then
    printf '0.0'
    return 0
  fi

  size_kib="$(du -sk "${path}" 2>/dev/null | awk '{print $1}')"
  if [[ -z "${size_kib}" ]]; then
    printf 'unknown'
    return 0
  fi

  awk -v kib="${size_kib}" 'BEGIN { printf "%.1f", kib / 1024 / 1024 }'
}

print_setup_summary() {
  local reference_status="${1:-ready}"
  local elapsed="$((SECONDS - SETUP_STARTED_SECONDS))"
  local reference_line
  local metallib_line

  if [[ "${reference_status}" == "skipped" ]]; then
    reference_line="skipped (${REFERENCE_DIR})"
  elif [[ -f "${REFERENCE_DIR}/config.json" ]]; then
    reference_line="${REFERENCE_DIR} ($(path_size_gib "${REFERENCE_DIR}") GiB)"
  else
    reference_line="missing (${REFERENCE_DIR})"
  fi

  if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
    metallib_line="skipped (${MLX_METALLIB})"
  else
    metallib_line="${MLX_METALLIB}"
  fi

  cat <<EOF
setup.sh: setup complete elapsed=$(format_duration "${elapsed}")
setup.sh: summary
  trusted CLI: ${SWIFT_BIN}
  participant worker: ${RUNTIME_WORKER_BIN}
  mlx.metallib: ${metallib_line}
  reference checkpoint: ${reference_line}
  next:
    MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" .github/scripts/run-offline.sh ${SWIFT_BIN} transform --reference "${REFERENCE_DIR}" --output weights
    ${SWIFT_BIN} correctness --weights weights
    ./benchmark.sh  # defaults to --local-iterate against the public fixtures (--official needs the organizer-provisioned oracle)
EOF
}

load_homebrew_shellenv() {
  local candidate
  local candidates=()

  if [[ -n "${HOMEBREW_PREFIX:-}" ]]; then
    candidates+=("${HOMEBREW_PREFIX}/bin/brew")
  fi
  candidates+=(
    "/opt/homebrew/bin/brew"
    "/usr/local/bin/brew"
    "${HOME}/.linuxbrew/bin/brew"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      eval "$("${candidate}" shellenv)"
      return 0
    fi
  done

  return 1
}

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi

  if load_homebrew_shellenv && command -v brew >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${MLXFAST_SKIP_HOMEBREW_INSTALL:-0}" == "1" ]]; then
    echo "setup.sh: Homebrew is not installed and MLXFAST_SKIP_HOMEBREW_INSTALL=1" >&2
    return 1
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: automatic Homebrew installation is only supported on macOS" >&2
    return 1
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "setup.sh: curl is required to install Homebrew" >&2
    return 1
  fi

  echo "setup.sh: Homebrew not found; installing Homebrew"
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if ! load_homebrew_shellenv || ! command -v brew >/dev/null 2>&1; then
    echo "setup.sh: Homebrew installation finished, but brew is still not on PATH" >&2
    echo "setup.sh: open a new shell or run Homebrew's shellenv command, then retry" >&2
    return 1
  fi
}

find_cmake() {
  local candidate
  if [[ -n "${MLXFAST_CMAKE_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_CMAKE_BIN}" ]]; then
      printf '%s\n' "${MLXFAST_CMAKE_BIN}"
      return 0
    fi
    echo "setup.sh: MLXFAST_CMAKE_BIN is set but not executable: ${MLXFAST_CMAKE_BIN}" >&2
    return 1
  fi

  if candidate="$(command -v cmake 2>/dev/null)"; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  for candidate in /opt/homebrew/bin/cmake /usr/local/bin/cmake; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

ensure_cmake() {
  if find_cmake >/dev/null; then
    return 0
  fi

  if [[ "${MLXFAST_SKIP_CMAKE_INSTALL:-0}" == "1" ]]; then
    echo "setup.sh: cmake is not installed and MLXFAST_SKIP_CMAKE_INSTALL=1" >&2
    return 1
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: automatic cmake installation is only supported on macOS" >&2
    return 1
  fi

  ensure_homebrew
  echo "setup.sh: installing cmake with Homebrew"
  brew install cmake

  if ! find_cmake >/dev/null; then
    echo "setup.sh: cmake installation finished, but cmake was not found" >&2
    return 1
  fi
}

find_macmon() {
  # Same lookup order as benchmark.sh's local cool-down gate: explicit
  # override, PATH, then the usual install locations.
  local candidate
  if [[ -n "${MLXFAST_MACMON_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_MACMON_BIN}" ]]; then
      printf '%s\n' "${MLXFAST_MACMON_BIN}"
      return 0
    fi
    echo "setup.sh: MLXFAST_MACMON_BIN is set but not executable: ${MLXFAST_MACMON_BIN}" >&2
    return 1
  fi
  if candidate="$(command -v macmon 2>/dev/null)"; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  for candidate in /opt/homebrew/bin/macmon /usr/local/bin/macmon "${HOME}/bin/macmon"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

install_macmon_release() {
  # Self-contained install: fetch the pinned macmon release tarball, verify
  # its SHA256 against the pin above, and drop the single arm64 binary in
  # ~/bin -- a location both find_macmon here and benchmark.sh's cool-gate
  # lookup already search, and the same drop convention the ranked boxes use.
  # Deliberately no Homebrew: installing an optional tool must not tap,
  # upgrade, or otherwise mutate the user's global package state.
  local staging_dir
  local tarball
  local actual_hash

  command -v curl >/dev/null 2>&1 || return 1
  staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/mlxfast-macmon.XXXXXX")" || return 1
  tarball="${staging_dir}/macmon.tar.gz"

  if ! curl \
      --fail \
      --location \
      --silent \
      --show-error \
      --retry 3 \
      --retry-delay 2 \
      --output "${tarball}" \
      "${MACMON_RELEASE_URL}"; then
    rm -rf "${staging_dir}"
    return 1
  fi

  actual_hash="$(shasum -a 256 "${tarball}" | awk '{print $1}')"
  if [[ "${actual_hash}" != "${MACMON_RELEASE_SHA256}" ]]; then
    echo "setup.sh: macmon release sha256 mismatch (expected ${MACMON_RELEASE_SHA256}, got ${actual_hash}); not installing" >&2
    rm -rf "${staging_dir}"
    return 1
  fi

  if ! tar -xzf "${tarball}" -C "${staging_dir}" macmon \
      || [[ ! -f "${staging_dir}/macmon" ]]; then
    rm -rf "${staging_dir}"
    return 1
  fi

  if ! mkdir -p "${MACMON_INSTALL_DIR}" \
      || ! chmod 0755 "${staging_dir}/macmon" \
      || ! mv -f "${staging_dir}/macmon" "${MACMON_INSTALL_DIR}/macmon"; then
    rm -rf "${staging_dir}"
    return 1
  fi

  rm -rf "${staging_dir}"
}

ensure_macmon() {
  # macmon (https://github.com/vladkens/macmon) is the unprivileged GPU
  # temperature reader benchmark.sh's local cool-down gate uses to mirror the
  # ranked runner's 40C thermal gate. It is NOT required to run the benchmark:
  # if it stays missing, benchmark.sh warns and skips the gate, so a failed
  # install must never fail setup. The install stays on by default because
  # benchmark.sh directs participants to rerun ./setup.sh to get macmon, but
  # it is a pinned ~/bin binary drop, never a Homebrew mutation.
  local macmon_path
  if macmon_path="$(find_macmon)"; then
    echo "setup.sh: macmon found at ${macmon_path} (GPU cool-down gate enabled for local benchmark modes)"
    return 0
  fi

  if [[ "${MLXFAST_SKIP_MACMON_INSTALL:-0}" != "1" && "$(uname -s)" == "Darwin" ]]; then
    echo "setup.sh: installing macmon v${MACMON_VERSION} to ${MACMON_INSTALL_DIR}/macmon (GPU temperature reader for the local benchmark cool-down gate)"
    install_macmon_release || true
    if macmon_path="$(find_macmon)"; then
      echo "setup.sh: macmon installed at ${macmon_path}"
      return 0
    fi
  fi

  echo "setup.sh: macmon not found; benchmark.sh's local GPU cool-down thermal gate will be skipped (install macmon on PATH or in ~/bin, or set MLXFAST_MACMON_BIN, to enable it)" >&2
  return 0
}

ensure_swift_toolchain() {
  local developer_dir
  local xcodebuild_output

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: this Swift harness targets macOS on Apple Silicon" >&2
    exit 1
  fi

  if [[ "$(uname -m)" != "arm64" ]]; then
    echo "setup.sh: this Swift harness requires Apple Silicon (arm64)" >&2
    exit 1
  fi

  if ! command -v swift >/dev/null 2>&1; then
    echo "setup.sh: swift was not found; install Xcode command line tools with xcode-select --install" >&2
    exit 1
  fi

  if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "setup.sh: xcodebuild was not found; install Xcode" >&2
    exit 1
  fi

  # Diagnose the actual xcodebuild failure instead of mapping everything to
  # an unaccepted license. On a machine with only the Command Line Tools,
  # xcodebuild exists as a shim but fails with "requires Xcode, but active
  # developer directory ... is a command line tools instance" -- that is a
  # missing/unselected full Xcode, not a license problem.
  if ! xcodebuild_output="$(xcodebuild -version 2>&1)"; then
    developer_dir="$(xcode-select -p 2>/dev/null || true)"
    case "${xcodebuild_output}" in
      *"command line tools instance"*|*"requires Xcode"*)
        cat >&2 <<EOF
setup.sh: full Xcode is required, but the active developer directory is the
Command Line Tools (${developer_dir:-unknown}), so xcodebuild cannot run.

The Command Line Tools alone cannot provide the Metal toolchain that
mlx.metallib needs. Install full Xcode from the App Store or
https://developer.apple.com/xcode/, then select it and retry:

  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  sudo xcodebuild -runFirstLaunch

EOF
        ;;
      *[Ll]icense*)
        cat >&2 <<EOF
setup.sh: the Xcode license has not been accepted for the selected Xcode
(${developer_dir:-unknown}).

Accept it and retry:

  sudo xcodebuild -license accept

EOF
        ;;
      *)
        cat >&2 <<EOF
setup.sh: xcodebuild is installed but not usable (developer directory: ${developer_dir:-unknown}).

xcodebuild -version failed with:

  ${xcodebuild_output}

If full Xcode is installed, open it once, select its command line tools, and
accept the license (sudo xcodebuild -license accept), then retry.

EOF
        ;;
    esac
    exit 1
  fi
}

metal_toolchain_identifier() {
  xcodebuild -showComponent MetalToolchain 2>/dev/null \
    | awk -F': ' '/Toolchain Identifier/ { print $2; exit }' \
    || true
}

# POLICY (TOOLCHAINS override) — APPROVED: on macOS 26+ the Metal compiler
# ships as a separately installed "Metal Toolchain" component that xcrun only
# finds when TOOLCHAINS selects it. This helper exports TOOLCHAINS
# process-wide so every later xcrun/metal invocation in this script resolves
# the same pinned compiler, which keeps metallib builds deterministic. Two
# consequences to be aware of: (1) any TOOLCHAINS value the caller already
# exported (e.g. a custom Swift toolchain selection) is silently replaced for
# the rest of this script, and (2) MLXFAST_METAL_TOOLCHAIN_IDENTIFIER lets
# the environment choose which toolchain identifier gets exported. On
# participant machines that override is deliberate and supported. The ranked
# M5 runner does not honor ad-hoc values: .github/workflows/benchmark.yml
# pins MLXFAST_METAL_TOOLCHAIN_IDENTIFIER to the operator-chosen toolchain in
# the trusted build step, and the bench-exec bridge's env -i child
# environment means submitted code cannot substitute its own.
configure_metal_toolchain_environment() {
  local identifier
  identifier="${MLXFAST_METAL_TOOLCHAIN_IDENTIFIER:-$(metal_toolchain_identifier)}"
  if [[ -n "${identifier}" ]]; then
    export TOOLCHAINS="${identifier}"
  fi
}

metal_compiler_is_available() {
  configure_metal_toolchain_environment
  xcrun -sdk macosx metal -v >/dev/null 2>&1
}

ensure_metal_toolchain() {
  if metal_compiler_is_available; then
    return 0
  fi

  if [[ "${MLXFAST_SKIP_METAL_TOOLCHAIN_INSTALL:-0}" == "1" ]]; then
    cat >&2 <<EOF
setup.sh: Xcode's Metal Toolchain is not installed and MLXFAST_SKIP_METAL_TOOLCHAIN_INSTALL=1.

Install full Xcode from the App Store or Apple Developer, open it once, select
its command line tools, accept the license, then retry:

  sudo xcodebuild -license accept
  xcodebuild -downloadComponent MetalToolchain

If you only installed the Command Line Tools and this still fails, install full
Xcode; the MLX Metal runtime needs Apple's Metal compiler toolchain.

EOF
    return 1
  fi

  echo "setup.sh: installing Xcode Metal Toolchain"
  if ! xcodebuild -downloadComponent MetalToolchain; then
    cat >&2 <<EOF
setup.sh: failed to install Xcode's Metal Toolchain.

Install full Xcode from the App Store or Apple Developer, open it once, select
its command line tools, accept the license, then retry:

  sudo xcodebuild -license accept
  xcodebuild -downloadComponent MetalToolchain

If you only installed the Command Line Tools and this still fails, install full
Xcode; the MLX Metal runtime needs Apple's Metal compiler toolchain.

EOF
    return 1
  fi

  if ! metal_compiler_is_available; then
    echo "setup.sh: Metal Toolchain installation finished, but xcrun still cannot execute metal" >&2
    return 1
  fi
}

ensure_reference_space() {
  local directory="$1"
  local available_kib
  local required_kib

  if ! [[ "${REFERENCE_MIN_FREE_GIB}" =~ ^[0-9]+$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_MIN_FREE_GIB must be an integer" >&2
    return 1
  fi

  available_kib="$(df -Pk "${directory}" | awk 'NR == 2 {print $4}')"
  required_kib=$((REFERENCE_MIN_FREE_GIB * 1024 * 1024))
  if [[ -z "${available_kib}" || "${available_kib}" -lt "${required_kib}" ]]; then
    cat >&2 <<EOF
setup.sh: not enough free disk space for ${REFERENCE_MODEL_REPO}.

Need at least ${REFERENCE_MIN_FREE_GIB} GiB free under ${directory}; available is $((available_kib / 1024 / 1024)) GiB.
Set MLXFAST_REFERENCE_DIR to a larger SSD, or set MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1
and place/mount the checkpoint manually.

EOF
    return 1
  fi
}

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
      echo "setup.sh: MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY must be auto, true, or false" >&2
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

reference_hash_verification_enabled() {
  case "${REFERENCE_HASH_VERIFY}" in
    0|false|FALSE|no|NO)
      return 1
      ;;
    1|true|TRUE|yes|YES)
      return 0
      ;;
    *)
      echo "setup.sh: MLXFAST_REFERENCE_HASH_VERIFY must be 0 or 1" >&2
      return 2
      ;;
  esac
}

reference_post_download_full_verify_enabled() {
  case "${REFERENCE_POST_DOWNLOAD_FULL_VERIFY}" in
    0|false|FALSE|no|NO)
      return 1
      ;;
    1|true|TRUE|yes|YES)
      return 0
      ;;
    *)
      echo "setup.sh: MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY must be 0 or 1" >&2
      return 2
      ;;
  esac
}

reference_manifest_entry() {
  local relative_path="$1"
  local source_manifest="${2:-${REFERENCE_MANIFEST_PATH}}"
  local line
  local expected_hash
  local expected_size
  local manifest_path
  local extra

  [[ -f "${source_manifest}" ]] || return 1
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r expected_hash expected_size manifest_path extra <<< "${line}"
    if [[ "${manifest_path}" == "${relative_path}" ]]; then
      printf '%s %s\n' "${expected_hash}" "${expected_size}"
      return 0
    fi
  done < "${source_manifest}"

  return 1
}

reference_file_is_current() {
  local relative_path="$1"
  local output_path="$2"
  local label="${3:-${relative_path}}"
  local manifest_entry
  local expected_hash
  local expected_size
  local actual_size
  local actual_hash
  local before_signature
  local after_signature
  local before_content_identity
  local after_content_identity
  local manifest_snapshot
  local verified_manifest_hash
  local hash_status

  REFERENCE_VERIFIED_CONTENT_IDENTITY=""
  REFERENCE_VERIFIED_EXPECTED_HASH=""
  REFERENCE_VERIFIED_EXPECTED_SIZE=""
  REFERENCE_VERIFIED_FILE_MANIFEST_HASH=""

  if reference_hash_verification_enabled; then
    hash_status=0
  else
    hash_status="$?"
  fi
  if [[ "${hash_status}" == "1" ]]; then
    [[ -s "${output_path}" ]] || return 1
    REFERENCE_VERIFIED_CONTENT_IDENTITY="$(file_content_identity_signature "${output_path}")" || return 1
    return 0
  elif [[ "${hash_status}" != "0" ]]; then
    return 1
  fi

  if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
    echo "setup.sh: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi
  manifest_snapshot="$(mktemp "${TMPDIR:-/tmp}/mlxfast-reference-manifest.XXXXXX")" || return 1
  if ! cp "${REFERENCE_MANIFEST_PATH}" "${manifest_snapshot}"; then
    rm -f "${manifest_snapshot}"
    return 1
  fi
  verified_manifest_hash="$(shasum -a 256 "${manifest_snapshot}" | awk '{print $1}')" || {
    rm -f "${manifest_snapshot}"
    return 1
  }
  if ! manifest_entry="$(reference_manifest_entry "${relative_path}" "${manifest_snapshot}")"; then
    rm -f "${manifest_snapshot}"
    echo "setup.sh: reference manifest has no entry for ${relative_path}" >&2
    return 1
  fi
  read -r expected_hash expected_size <<< "${manifest_entry}"

  if [[ ! -f "${output_path}" ]]; then
    rm -f "${manifest_snapshot}"
    return 1
  fi

  actual_size="$(wc -c < "${output_path}" | tr -d ' ')"
  if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "setup.sh: cached ${label} size mismatch: expected ${expected_size}, got ${actual_size}"
    rm -f "${manifest_snapshot}"
    return 1
  fi

  before_signature="$(file_metadata_signature "${output_path}")" || {
    rm -f "${manifest_snapshot}"
    return 1
  }
  before_content_identity="$(file_content_identity_signature "${output_path}")" || {
    rm -f "${manifest_snapshot}"
    return 1
  }
  actual_hash="$(shasum -a 256 "${output_path}" | awk '{print $1}')" || {
    rm -f "${manifest_snapshot}"
    return 1
  }
  after_signature="$(file_metadata_signature "${output_path}")" || {
    rm -f "${manifest_snapshot}"
    return 1
  }
  after_content_identity="$(file_content_identity_signature "${output_path}")" || {
    rm -f "${manifest_snapshot}"
    return 1
  }
  if [[ "${before_signature}" != "${after_signature}" \
      || "${before_content_identity}" != "${after_content_identity}" ]]; then
    echo "setup.sh: cached ${label} changed while SHA256 was being verified" >&2
    rm -f "${manifest_snapshot}"
    return 1
  fi
  if [[ "${actual_hash}" != "${expected_hash}" ]]; then
    echo "setup.sh: cached ${label} sha256 mismatch"
    echo "setup.sh: expected ${expected_hash}"
    echo "setup.sh: actual   ${actual_hash}"
    rm -f "${manifest_snapshot}"
    return 1
  fi
  if [[ "$(reference_manifest_hash 2>/dev/null || true)" != "${verified_manifest_hash}" ]]; then
    echo "setup.sh: reference manifest changed while ${label} was being verified" >&2
    rm -f "${manifest_snapshot}"
    return 1
  fi

  rm -f "${manifest_snapshot}"
  REFERENCE_VERIFIED_CONTENT_IDENTITY="${after_content_identity}"
  REFERENCE_VERIFIED_EXPECTED_HASH="${expected_hash}"
  REFERENCE_VERIFIED_EXPECTED_SIZE="${expected_size}"
  REFERENCE_VERIFIED_FILE_MANIFEST_HASH="${verified_manifest_hash}"
  return 0
}

reference_manifest_hash() {
  if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
    echo "setup.sh: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi
  shasum -a 256 "${REFERENCE_MANIFEST_PATH}" | awk '{print $1}'
}

reference_manifest_totals() {
  local source_manifest="${1:-${REFERENCE_MANIFEST_PATH}}"
  local line
  local expected_hash
  local expected_size
  local relative_path
  local extra
  local file_count=0
  local byte_count=0

  if [[ ! -f "${source_manifest}" ]]; then
    echo "setup.sh: reference manifest missing at ${source_manifest}" >&2
    return 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r expected_hash expected_size relative_path extra <<< "${line}"
    if [[ -n "${extra:-}" || -z "${expected_hash:-}" || -z "${expected_size:-}" || -z "${relative_path:-}" ]]; then
      return 1
    fi
    if [[ ! "${expected_hash}" =~ ^[0-9a-f]{64}$ || ! "${expected_size}" =~ ^[0-9]+$ ]]; then
      return 1
    fi
    if [[ "${relative_path}" == /* || "${relative_path}" == *\\* ]]; then
      return 1
    fi
    case "/${relative_path}/" in
      *"/../"*|*"/./"*) return 1 ;;
    esac

    file_count=$((file_count + 1))
    byte_count=$((byte_count + expected_size))
  done < "${source_manifest}"

  [[ "${file_count}" -gt 0 ]] || return 1
  printf '%s %s\n' "${file_count}" "${byte_count}"
}

stat_value() {
  local bsd_format="$1"
  local gnu_format="$2"
  local path="$3"
  local value
  # GNU stat can print data for the valid path before its BSD-style probe fails.
  if value="$(stat -f "${bsd_format}" "${path}" 2>/dev/null)"; then
    printf '%s\n' "${value}"
    return 0
  fi
  stat -c "${gnu_format}" "${path}"
}

file_mtime_seconds() {
  local file_path="$1"
  stat_value '%m' '%Y' "${file_path}"
}

file_metadata_signature() {
  local file_path="$1"
  stat_value '%d:%i:%z:%Fm:%Fc:%v' '%d:%i:%s:%Y:%Z:0' "${file_path}"
}

file_content_identity_signature() {
  local file_path="$1"
  stat_value '%d:%i:%z:%Fm:%v' '%d:%i:%s:%Y:0' "${file_path}"
}

discard_reference_verified_signatures() {
  if [[ -n "${REFERENCE_VERIFIED_SIGNATURES_PATH}" ]]; then
    rm -f "${REFERENCE_VERIFIED_SIGNATURES_PATH}" || true
    REFERENCE_VERIFIED_SIGNATURES_PATH=""
  fi
  if [[ -n "${REFERENCE_VERIFIED_MANIFEST_PATH}" ]]; then
    rm -f "${REFERENCE_VERIFIED_MANIFEST_PATH}" || true
    REFERENCE_VERIFIED_MANIFEST_PATH=""
  fi
  REFERENCE_VERIFIED_MANIFEST_HASH=""
}

create_reference_verified_manifest_snapshot() {
  local parent
  parent="$(dirname "${REFERENCE_CACHE_LOCK_PATH}")"
  mkdir -p "${parent}" || return 1
  REFERENCE_VERIFIED_MANIFEST_PATH="$(
    mktemp "${REFERENCE_CACHE_LOCK_PATH}.manifest.XXXXXX"
  )" || {
    REFERENCE_VERIFIED_MANIFEST_PATH=""
    return 1
  }
  if ! cp "${REFERENCE_MANIFEST_PATH}" "${REFERENCE_VERIFIED_MANIFEST_PATH}"; then
    rm -f "${REFERENCE_VERIFIED_MANIFEST_PATH}"
    REFERENCE_VERIFIED_MANIFEST_PATH=""
    return 1
  fi
  REFERENCE_VERIFIED_MANIFEST_HASH="$(
    shasum -a 256 "${REFERENCE_VERIFIED_MANIFEST_PATH}" | awk '{print $1}'
  )" || {
    rm -f "${REFERENCE_VERIFIED_MANIFEST_PATH}"
    REFERENCE_VERIFIED_MANIFEST_PATH=""
    REFERENCE_VERIFIED_MANIFEST_HASH=""
    return 1
  }
}

create_reference_verified_signatures_path() {
  local parent
  parent="$(dirname "${REFERENCE_CACHE_LOCK_PATH}")"
  mkdir -p "${parent}" || return 1
  REFERENCE_VERIFIED_SIGNATURES_PATH="$(
    mktemp "${REFERENCE_CACHE_LOCK_PATH}.verified.XXXXXX"
  )" || {
    REFERENCE_VERIFIED_SIGNATURES_PATH=""
    return 1
  }
  : > "${REFERENCE_VERIFIED_SIGNATURES_PATH}"
}

write_verified_reference_marker() {
  local relative_path="$1"
  local file_path="$2"
  local marker_path="$3"
  local expected_content_identity="$4"
  local expected_hash="$5"
  local expected_size="$6"
  local expected_manifest_hash="$7"
  local current_content_identity
  local current_signature
  local marker_temp

  if [[ -z "${expected_hash}" || -z "${expected_size}" \
      || -z "${expected_manifest_hash}" ]]; then
    return 1
  fi
  if [[ "$(reference_manifest_hash 2>/dev/null || true)" != "${expected_manifest_hash}" ]]; then
    echo "setup.sh: reference manifest changed before ${relative_path} marker publication" >&2
    return 1
  fi
  current_content_identity="$(file_content_identity_signature "${file_path}")" || return 1
  if [[ -z "${expected_content_identity}" \
      || "${current_content_identity}" != "${expected_content_identity}" ]]; then
    echo "setup.sh: ${relative_path} changed before its verification marker was written" >&2
    return 1
  fi
  current_signature="$(file_metadata_signature "${file_path}")" || return 1
  marker_temp="$(mktemp "${marker_path}.tmp.XXXXXX")" || return 1
  if ! printf 'version=2\t%s\t%s\t%s\t%s\n' \
      "${expected_manifest_hash}" "${expected_hash}" "${expected_size}" "${current_signature}" > "${marker_temp}" \
      || ! mv "${marker_temp}" "${marker_path}"; then
    rm -f "${marker_temp}"
    return 1
  fi
}

mark_reference_file_complete() {
  local relative_path="$1"
  local file_path="$2"
  local marker_path="$3"
  local expected_content_identity="$4"
  local expected_hash="$5"
  local expected_size="$6"
  local expected_manifest_hash="$7"
  local hash_status

  if reference_hash_verification_enabled; then
    hash_status=0
  else
    hash_status="$?"
  fi
  if [[ "${hash_status}" == "0" ]]; then
    write_verified_reference_marker \
      "${relative_path}" "${file_path}" "${marker_path}" "${expected_content_identity}" \
      "${expected_hash}" "${expected_size}" "${expected_manifest_hash}"
  elif [[ "${hash_status}" == "1" ]]; then
    touch "${marker_path}"
  else
    return 1
  fi
}

directory_identity() {
  local directory="$1"
  stat_value '%d:%i' '%d:%i' "${directory}"
}

current_shell_pid() {
  local pid_path
  if ! pid_path="$(mktemp "${TMPDIR:-/tmp}/mlxfast-shell-pid.XXXXXX")"; then
    return 1
  fi
  if ! /bin/sh -c 'printf "%s\n" "$PPID" > "$1"' _ "${pid_path}" \
      || ! IFS= read -r CURRENT_SHELL_PID < "${pid_path}"; then
    rm -f "${pid_path}"
    return 1
  fi
  rm -f "${pid_path}"
  [[ "${CURRENT_SHELL_PID}" =~ ^[1-9][0-9]*$ ]]
}

reference_cache_lock_matches_manifest_snapshot() {
  local reference_dir="$1"
  local lock_path="$2"
  local manifest_snapshot="$3"
  local expected_manifest_hash="$4"
  local manifest_file_count="$5"
  local manifest_byte_count="$6"
  local line
  local key
  local value
  local in_files=0
  local version=""
  local model_repo=""
  local revision=""
  local manifest_hash=""
  local lock_file_count=""
  local lock_byte_count=""
  local actual_file_count=0
  local actual_byte_count=0
  local relative_path
  local expected_size
  local expected_signature
  local extra
  local file_path
  local actual_size
  local actual_signature
  local manifest_entry
  local manifest_expected_hash
  local manifest_expected_size

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    if [[ "${line}" == "--files--" ]]; then
      in_files=1
      continue
    fi

    if [[ "${in_files}" == "0" ]]; then
      key="${line%%=*}"
      value="${line#*=}"
      case "${key}" in
        version) version="${value}" ;;
        model_repo) model_repo="${value}" ;;
        revision) revision="${value}" ;;
        manifest_sha256) manifest_hash="${value}" ;;
        file_count) lock_file_count="${value}" ;;
        byte_count) lock_byte_count="${value}" ;;
      esac
      continue
    fi

    IFS=$'\t' read -r relative_path expected_size expected_signature extra <<< "${line}"
    if [[ -n "${extra:-}" || -z "${relative_path:-}" || -z "${expected_size:-}" || -z "${expected_signature:-}" ]]; then
      return 1
    fi
    if [[ "${relative_path}" == /* || "${relative_path}" == *\\* ]]; then
      return 1
    fi
    [[ "${expected_size}" =~ ^[0-9]+$ ]] || return 1
    [[ "${expected_signature}" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+([.][0-9]+)?:[0-9]+([.][0-9]+)?:[0-9]+$ ]] || return 1
    case "/${relative_path}/" in
      *"/../"*|*"/./"*) return 1 ;;
    esac
    if ! manifest_entry="$(reference_manifest_entry "${relative_path}" "${manifest_snapshot}")"; then
      return 1
    fi
    read -r manifest_expected_hash manifest_expected_size <<< "${manifest_entry}"
    [[ -n "${manifest_expected_hash}" ]] || return 1
    [[ "${expected_size}" == "${manifest_expected_size}" ]] || return 1

    file_path="${reference_dir}/${relative_path}"
    [[ -f "${file_path}" ]] || return 1
    actual_size="$(wc -c < "${file_path}" | tr -d ' ')"
    [[ "${actual_size}" == "${expected_size}" ]] || return 1
    if ! actual_signature="$(file_metadata_signature "${file_path}")"; then
      return 1
    fi
    [[ "${actual_signature}" == "${expected_signature}" ]] || return 1

    actual_file_count=$((actual_file_count + 1))
    actual_byte_count=$((actual_byte_count + actual_size))
  done < "${lock_path}"

  [[ "${version}" == "3" ]] || return 1
  [[ "${model_repo}" == "${REFERENCE_MODEL_REPO}" ]] || return 1
  [[ "${revision}" == "${REFERENCE_REVISION}" ]] || return 1
  [[ "${manifest_hash}" == "${expected_manifest_hash}" ]] || return 1
  [[ "${lock_file_count}" =~ ^[0-9]+$ ]] || return 1
  [[ "${lock_byte_count}" =~ ^[0-9]+$ ]] || return 1
  [[ "${lock_file_count}" == "${manifest_file_count}" ]] || return 1
  [[ "${lock_byte_count}" == "${manifest_byte_count}" ]] || return 1
  [[ "${actual_file_count}" == "${lock_file_count}" ]] || return 1
  [[ "${actual_byte_count}" == "${lock_byte_count}" ]] || return 1

  return 0
}

reference_cache_lock_is_current() {
  local reference_dir="$1"
  local lock_path="${REFERENCE_CACHE_LOCK_PATH}"
  local hash_status
  local manifest_snapshot
  local expected_manifest_hash
  local current_manifest_hash
  local expected_manifest_totals
  local manifest_file_count
  local manifest_byte_count
  local validation_status

  if reference_hash_verification_enabled; then
    hash_status=0
  else
    hash_status="$?"
  fi
  if [[ "${hash_status}" != "0" ]]; then
    return 1
  fi
  [[ -f "${lock_path}" ]] || return 1
  [[ -f "${REFERENCE_MANIFEST_PATH}" ]] || return 1

  manifest_snapshot="$(mktemp "${TMPDIR:-/tmp}/mlxfast-cache-lock-manifest.XXXXXX")" \
    || return 1
  if ! cp "${REFERENCE_MANIFEST_PATH}" "${manifest_snapshot}"; then
    rm -f "${manifest_snapshot}"
    return 1
  fi
  if ! expected_manifest_hash="$(shasum -a 256 "${manifest_snapshot}" | awk '{print $1}')"; then
    rm -f "${manifest_snapshot}"
    return 1
  fi
  if ! expected_manifest_totals="$(reference_manifest_totals "${manifest_snapshot}")"; then
    rm -f "${manifest_snapshot}"
    return 1
  fi
  read -r manifest_file_count manifest_byte_count <<< "${expected_manifest_totals}"

  validation_status=0
  reference_cache_lock_matches_manifest_snapshot \
    "${reference_dir}" \
    "${lock_path}" \
    "${manifest_snapshot}" \
    "${expected_manifest_hash}" \
    "${manifest_file_count}" \
    "${manifest_byte_count}" || validation_status=$?
  rm -f "${manifest_snapshot}"
  if [[ "${validation_status}" != "0" ]]; then
    return "${validation_status}"
  fi

  # The snapshot binds every entry used above. Re-read the live manifest only
  # after validation so a concurrent checkout/update cannot make an old stamp
  # authorize a different same-shape manifest for this invocation.
  current_manifest_hash="$(reference_manifest_hash 2>/dev/null || true)"
  if [[ "${current_manifest_hash}" != "${expected_manifest_hash}" ]]; then
    echo "setup.sh: reference manifest changed while the cache lock was being validated" >&2
    return 1
  fi

  echo "setup.sh: trusted reference cache lock at ${lock_path}; skipping full SHA256 verification"
  return 0
}

write_reference_cache_lock() {
  local reference_dir="$1"
  local lock_path="${REFERENCE_CACHE_LOCK_PATH}"
  local temp_path
  local manifest_hash
  local expected_size
  local relative_path
  local extra
  local file_path
  local actual_size
  local actual_signature
  local manifest_entry
  local manifest_expected_hash
  local manifest_expected_size
  local manifest_totals
  local manifest_file_count
  local manifest_byte_count
  local file_count=0
  local byte_count=0
  local files_path="${REFERENCE_VERIFIED_SIGNATURES_PATH}"
  local capture_epoch

  if [[ -z "${files_path}" || ! -f "${files_path}" \
      || -z "${REFERENCE_VERIFIED_MANIFEST_HASH}" ]]; then
    echo "setup.sh: refusing to stamp reference files without bound SHA256 verification" >&2
    return 1
  fi
  if ! manifest_hash="$(reference_manifest_hash)"; then
    discard_reference_verified_signatures
    return 1
  fi
  if [[ "${manifest_hash}" != "${REFERENCE_VERIFIED_MANIFEST_HASH}" ]]; then
    discard_reference_verified_signatures
    echo "setup.sh: reference manifest changed after SHA256 verification" >&2
    return 1
  fi
  if ! manifest_totals="$(reference_manifest_totals)"; then
    discard_reference_verified_signatures
    return 1
  fi
  read -r manifest_file_count manifest_byte_count <<< "${manifest_totals}"

  capture_epoch="$(date +%s)"

  mkdir -p "$(dirname "${lock_path}")"
  if ! temp_path="$(mktemp "${lock_path}.tmp.XXXXXX")"; then
    discard_reference_verified_signatures
    echo "setup.sh: failed to create temporary reference cache stamp" >&2
    return 1
  fi
  while IFS=$'\t' read -r relative_path expected_size actual_signature extra \
      || [[ -n "${relative_path:-}" ]]; do
    if [[ -n "${extra:-}" || -z "${relative_path:-}" \
        || -z "${expected_size:-}" || -z "${actual_signature:-}" ]]; then
      rm -f "${temp_path}"
      discard_reference_verified_signatures
      return 1
    fi
    if ! manifest_entry="$(reference_manifest_entry "${relative_path}")"; then
      rm -f "${temp_path}"
      discard_reference_verified_signatures
      return 1
    fi
    read -r manifest_expected_hash manifest_expected_size <<< "${manifest_entry}"
    if [[ -z "${manifest_expected_hash}" || "${expected_size}" != "${manifest_expected_size}" ]]; then
      rm -f "${temp_path}"
      discard_reference_verified_signatures
      return 1
    fi
    file_path="${reference_dir}/${relative_path}"
    [[ -f "${file_path}" ]] || {
      rm -f "${temp_path}"
      discard_reference_verified_signatures
      return 1
    }
    actual_size="$(wc -c < "${file_path}" | tr -d ' ')"
    if [[ "${actual_size}" != "${expected_size}" ]]; then
      rm -f "${temp_path}"
      discard_reference_verified_signatures
      return 1
    fi
    if [[ "$(file_metadata_signature "${file_path}" 2>/dev/null || true)" != "${actual_signature}" ]]; then
      rm -f "${temp_path}"
      discard_reference_verified_signatures
      return 1
    fi
    file_count=$((file_count + 1))
    byte_count=$((byte_count + actual_size))
  done < "${files_path}"

  if [[ "${file_count}" -eq 0 \
      || "${file_count}" != "${manifest_file_count}" \
      || "${byte_count}" != "${manifest_byte_count}" ]]; then
    rm -f "${temp_path}"
    discard_reference_verified_signatures
    return 1
  fi

  # GNU stat falls back to second-resolution timestamps. Cross a clock
  # boundary, then prove the SHA-bound signatures are still unchanged before
  # publishing. Darwin signatures additionally carry nanosecond timestamps.
  while (( $(date +%s) <= capture_epoch )); do
    sleep 0.1
  done
  while IFS=$'\t' read -r relative_path expected_size actual_signature extra \
      || [[ -n "${relative_path:-}" ]]; do
    if [[ -n "${extra:-}" || -z "${relative_path:-}" \
        || -z "${expected_size:-}" || -z "${actual_signature:-}" ]]; then
      rm -f "${temp_path}"
      discard_reference_verified_signatures
      return 1
    fi
    file_path="${reference_dir}/${relative_path}"
    if [[ ! -f "${file_path}" \
        || "$(wc -c < "${file_path}" | tr -d ' ')" != "${expected_size}" \
        || "$(file_metadata_signature "${file_path}" 2>/dev/null || true)" != "${actual_signature}" ]]; then
      rm -f "${temp_path}"
      discard_reference_verified_signatures
      return 1
    fi
  done < "${files_path}"

  if [[ "$(reference_manifest_hash 2>/dev/null || true)" \
      != "${REFERENCE_VERIFIED_MANIFEST_HASH}" ]]; then
    rm -f "${temp_path}"
    discard_reference_verified_signatures
    echo "setup.sh: reference manifest changed before cache stamp publication" >&2
    return 1
  fi

  if ! {
    printf 'version=3\n'
    printf 'model_repo=%s\n' "${REFERENCE_MODEL_REPO}"
    printf 'revision=%s\n' "${REFERENCE_REVISION}"
    printf 'manifest_sha256=%s\n' "${manifest_hash}"
    printf 'file_count=%s\n' "${file_count}"
    printf 'byte_count=%s\n' "${byte_count}"
    printf 'verified_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '%s\n' '--files--'
    cat "${files_path}"
  } > "${temp_path}"; then
    rm -f "${temp_path}"
    discard_reference_verified_signatures
    return 1
  fi
  discard_reference_verified_signatures
  if ! mv "${temp_path}" "${lock_path}"; then
    rm -f "${temp_path}"
    return 1
  fi
  echo "setup.sh: wrote reference cache lock ${lock_path}"
}

validate_reference_download_settings() {
  local name
  local value

  for name in \
    REFERENCE_DOWNLOAD_PROGRESS_SECONDS \
    REFERENCE_DOWNLOAD_STALL_SECONDS \
    REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND; do
    value="${!name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
      echo "setup.sh: ${name} must be a positive integer" >&2
      return 1
    fi
  done
}

download_reference_file() {
  local file="$1"
  local output_path="$2"
  local label="${3:-${file}}"
  local marker_path="${output_path}.complete"
  local partial_path="${output_path}.partial"
  local base_url
  local url
  local started_seconds
  local attempt
  local curl_status
  local source_index=0
  local source_count
  local base_urls=("${REFERENCE_BASE_URL}")
  local verified_content_identity
  local verified_expected_hash
  local verified_expected_size
  local verified_manifest_hash

  validate_reference_download_settings || return 1
  if [[ -n "${REFERENCE_FALLBACK_BASE_URL}" && "${REFERENCE_FALLBACK_BASE_URL}" != "${REFERENCE_BASE_URL}" ]]; then
    base_urls+=("${REFERENCE_FALLBACK_BASE_URL}")
  fi
  source_count="${#base_urls[@]}"

  if reference_file_is_current "${file}" "${output_path}" "${label}"; then
    echo "setup.sh: using cached ${label}"
    mark_reference_file_complete \
      "${file}" "${output_path}" "${marker_path}" \
      "${REFERENCE_VERIFIED_CONTENT_IDENTITY}" \
      "${REFERENCE_VERIFIED_EXPECTED_HASH}" \
      "${REFERENCE_VERIFIED_EXPECTED_SIZE}" \
      "${REFERENCE_VERIFIED_FILE_MANIFEST_HASH}" || return 1
    return 0
  fi
  rm -f "${marker_path}" || return 1

  mkdir -p "$(dirname "${output_path}")" || return 1
  started_seconds="${SECONDS}"
  for base_url in "${base_urls[@]}"; do
    source_index=$((source_index + 1))
    url="${base_url%/}/${file}"
    if ! url="$(download_url_for_file "${url}")"; then
      return 1
    fi

    if [[ "${source_index}" -gt 1 ]]; then
      echo "setup.sh: trying fallback source for ${label}: ${base_url}"
    fi

    attempt=1
    while [[ "${attempt}" -le 2 ]]; do
      if [[ "${attempt}" == "1" ]]; then
        echo "setup.sh: downloading ${label}"
      else
        echo "setup.sh: redownloading ${label} from scratch after hash verification failed"
        rm -f "${partial_path}" "${marker_path}" || return 1
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
          --speed-limit "${REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND}" \
          --speed-time "${REFERENCE_DOWNLOAD_STALL_SECONDS}" \
          -H "${REFERENCE_AUTH_HEADER}" \
          --output "${partial_path}" \
          "${url}" || curl_status=$?
      else
        curl \
          --fail \
          --location \
          --retry 5 \
          --retry-all-errors \
          --retry-delay 2 \
          --continue-at - \
          --speed-limit "${REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND}" \
          --speed-time "${REFERENCE_DOWNLOAD_STALL_SECONDS}" \
          --output "${partial_path}" \
          "${url}" || curl_status=$?
      fi
      if [[ "${curl_status}" != "0" ]]; then
        echo "setup.sh: ${label} source failed or stalled (status=${curl_status}, source=${base_url})" >&2
        break
      fi

      if reference_file_is_current "${file}" "${partial_path}" "${label}"; then
        verified_content_identity="${REFERENCE_VERIFIED_CONTENT_IDENTITY}"
        verified_expected_hash="${REFERENCE_VERIFIED_EXPECTED_HASH}"
        verified_expected_size="${REFERENCE_VERIFIED_EXPECTED_SIZE}"
        verified_manifest_hash="${REFERENCE_VERIFIED_FILE_MANIFEST_HASH}"
        mv "${partial_path}" "${output_path}" || return 1
        mark_reference_file_complete \
          "${file}" "${output_path}" "${marker_path}" \
          "${verified_content_identity}" \
          "${verified_expected_hash}" \
          "${verified_expected_size}" \
          "${verified_manifest_hash}" || return 1
        echo "setup.sh: downloaded ${label} elapsed=$(format_duration "$((SECONDS - started_seconds))")"
        return 0
      fi

      attempt=$((attempt + 1))
    done

    if [[ "${source_index}" -lt "${source_count}" && "${curl_status}" == "0" ]]; then
      rm -f "${partial_path}" "${marker_path}" || return 1
    fi
  done

  echo "setup.sh: failed to download verified ${label}" >&2
  return 1
}

reference_download_on_disk_bytes() {
  local output_dir="$1"
  shift
  local total_bytes=0
  local file_path
  local shard_file

  for shard_file in "$@"; do
    file_path="${output_dir}/${shard_file}"
    if [[ -f "${file_path}" ]]; then
      total_bytes=$((total_bytes + $(wc -c < "${file_path}" | tr -d ' ')))
    elif [[ -f "${file_path}.partial" ]]; then
      # In-flight curl transfers write to <shard>.partial and only rename on
      # verified completion; counting those bytes keeps the heartbeat moving
      # from the first transferred byte instead of reporting 0% until a whole
      # shard completes. elif, not a second if: a redownload can briefly leave
      # a stale final file next to a fresh .partial, and double-counting would
      # overstate progress.
      total_bytes=$((total_bytes + $(wc -c < "${file_path}.partial" | tr -d ' ')))
    fi
  done
  printf '%s\n' "${total_bytes}"
}

start_reference_download_heartbeat() {
  # A cold checkpoint download is ~20 GiB and the parallel per-shard curls run
  # silenced, which used to mean 10+ minutes with no output at all. This
  # heartbeat is the single progress reporter for the parallel shard phase:
  # every REFERENCE_DOWNLOAD_PROGRESS_SECONDS it sums the on-disk shard bytes
  # and prints one aggregate line with percentage, transfer rate, and ETA.
  # The baseline is sampled before the transfers start, so bytes resumed from
  # an earlier interrupted run do not inflate the reported rate. The
  # per-shard curls themselves stay --silent; stall detection is curl's
  # --speed-limit/--speed-time, not this reporter.
  local output_dir="$1"
  local expected_total_bytes="$2"
  shift 2
  local heartbeat_started_seconds="${SECONDS}"
  local initial_bytes

  initial_bytes="$(reference_download_on_disk_bytes "${output_dir}" "$@")"

  (
    local downloaded_bytes
    local elapsed
    local progress
    while :; do
      sleep "${REFERENCE_DOWNLOAD_PROGRESS_SECONDS}"
      kill -0 "$$" 2>/dev/null || exit 0
      downloaded_bytes="$(reference_download_on_disk_bytes "${output_dir}" "$@")"
      elapsed=$((SECONDS - heartbeat_started_seconds))
      if [[ "${expected_total_bytes}" -gt 0 ]]; then
        progress="$(awk \
          -v done="${downloaded_bytes}" \
          -v total="${expected_total_bytes}" \
          -v initial="${initial_bytes}" \
          -v elapsed="${elapsed}" \
          'BEGIN {
            added = done - initial
            if (added < 0) added = 0
            rate = elapsed > 0 ? added / elapsed : 0
            eta = (total > done && rate > 0) ? (total - done) / rate : 0
            printf "%.1f of %.1f GiB (%d%%) rate=%.1f MiB/s eta=%ds", \
              done / 1073741824, total / 1073741824, (done * 100) / total, \
              rate / 1048576, eta
          }')"
      else
        progress="$(awk -v done="${downloaded_bytes}" 'BEGIN { printf "%.1f GiB so far", done / 1073741824 }')"
      fi
      echo "setup.sh: still downloading safetensors shard(s): ${progress} elapsed=$(format_duration "${elapsed}")"
    done
  ) &
  REFERENCE_DOWNLOAD_HEARTBEAT_PID="$!"
}

stop_reference_download_heartbeat() {
  if [[ -n "${REFERENCE_DOWNLOAD_HEARTBEAT_PID}" ]]; then
    kill "${REFERENCE_DOWNLOAD_HEARTBEAT_PID}" >/dev/null 2>&1 || true
    wait "${REFERENCE_DOWNLOAD_HEARTBEAT_PID}" >/dev/null 2>&1 || true
    REFERENCE_DOWNLOAD_HEARTBEAT_PID=""
  fi
}

download_reference_shards() {
  local output_dir="$1"
  shift
  local jobs="${REFERENCE_DOWNLOAD_JOBS}"
  local total="$#"
  local started_seconds="${SECONDS}"

  if ! [[ "${jobs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_DOWNLOAD_JOBS must be a positive integer" >&2
    return 1
  fi
  validate_reference_download_settings || return 1

  if [[ "${jobs}" == "1" || "$#" -le 1 ]]; then
    local file
    local ordinal=0
    echo "setup.sh: downloading ${total} safetensors shard(s) with 1 parallel job"
    for file in "$@"; do
      ordinal=$((ordinal + 1))
      download_reference_file "${file}" "${output_dir}/${file}" "shard ${ordinal}/${total}: ${file}" || return 1
    done
    echo "setup.sh: downloaded ${total}/${total} safetensors shard(s) elapsed=$(format_duration "$((SECONDS - started_seconds))")"
    return 0
  fi

  local ordinal=0
  local manifest_entry
  local expected_hash
  local expected_size
  local expected_total_bytes=0
  local expected_total_known=1
  local download_status=0
  local download_manifest_snapshot
  local download_manifest_hash
  download_manifest_snapshot="$(mktemp "${TMPDIR:-/tmp}/mlxfast-download-manifest.XXXXXX")" || return 1
  if ! cp "${REFERENCE_MANIFEST_PATH}" "${download_manifest_snapshot}"; then
    rm -f "${download_manifest_snapshot}"
    return 1
  fi
  download_manifest_hash="$(
    shasum -a 256 "${download_manifest_snapshot}" | awk '{print $1}'
  )" || {
    rm -f "${download_manifest_snapshot}"
    return 1
  }

  # Pre-compute the aggregate expected byte count from the pinned manifest so
  # both the banner line and the progress heartbeat can report percentages.
  for file in "$@"; do
    expected_size=""
    if manifest_entry="$(reference_manifest_entry "${file}" "${download_manifest_snapshot}")"; then
      read -r expected_hash expected_size <<< "${manifest_entry}"
    fi
    if [[ "${expected_size}" =~ ^[0-9]+$ ]]; then
      expected_total_bytes=$((expected_total_bytes + expected_size))
    else
      expected_total_known=0
    fi
  done
  if [[ "${expected_total_known}" != "1" ]]; then
    expected_total_bytes=0
  fi

  if [[ "${expected_total_bytes}" -gt 0 ]]; then
    echo "setup.sh: downloading ${total} safetensors shard(s) with ${jobs} parallel job(s) ($(awk -v total="${expected_total_bytes}" 'BEGIN { printf "%.1f", total / 1073741824 }') GiB total; progress prints every ${REFERENCE_DOWNLOAD_PROGRESS_SECONDS}s)"
  else
    echo "setup.sh: downloading ${total} safetensors shard(s) with ${jobs} parallel job(s)"
  fi
  export REFERENCE_BASE_URL
  export REFERENCE_FALLBACK_BASE_URL
  export REFERENCE_AUTH_HEADER
  export REFERENCE_APPEND_DOWNLOAD_QUERY
  export REFERENCE_HASH_VERIFY
  export REFERENCE_DOWNLOAD_STALL_SECONDS
  export REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND
  export REFERENCE_MANIFEST_PATH
  export REFERENCE_DOWNLOAD_MANIFEST_HASH="${download_manifest_hash}"
  start_reference_download_heartbeat "${output_dir}" "${expected_total_bytes}" "$@"
  # The embedded program expands these variables in each child Bash process.
  # shellcheck disable=SC2016
  for file in "$@"; do
    ordinal=$((ordinal + 1))
    expected_hash=""
    expected_size=""
    if manifest_entry="$(reference_manifest_entry "${file}" "${download_manifest_snapshot}")"; then
      read -r expected_hash expected_size <<< "${manifest_entry}"
    fi
    printf "%s|%s|%s|%s\0" "${ordinal}" "${file}" "${expected_hash}" "${expected_size}"
  done | xargs -0 -I{} -P "${jobs}" bash -c '
    set -euo pipefail
    record="$1"
    output_dir="$2"
    total="$3"
    ordinal="${record%%|*}"
    remainder="${record#*|}"
    file="${remainder%%|*}"
    remainder="${remainder#*|}"
    expected_hash="${remainder%%|*}"
    expected_size="${remainder#*|}"
    output_path="${output_dir}/${file}"
    marker_path="${output_path}.complete"
    partial_path="${output_path}.partial"
    started_seconds="${SECONDS}"
    source_index=0
    base_urls=("${REFERENCE_BASE_URL}")
    if [[ -n "${REFERENCE_FALLBACK_BASE_URL:-}" && "${REFERENCE_FALLBACK_BASE_URL}" != "${REFERENCE_BASE_URL}" ]]; then
      base_urls+=("${REFERENCE_FALLBACK_BASE_URL}")
    fi
    source_count="${#base_urls[@]}"
    verified_content_identity=""

    stat_value() {
      local bsd_format="$1"
      local gnu_format="$2"
      local path="$3"
      local value
      # Discard output from a failed BSD-style probe before trying GNU stat.
      if value="$(stat -f "${bsd_format}" "${path}" 2>/dev/null)"; then
        printf "%s\n" "${value}"
        return 0
      fi
      stat -c "${gnu_format}" "${path}"
    }

    file_metadata_signature() {
      stat_value "%d:%i:%z:%Fm:%Fc:%v" "%d:%i:%s:%Y:%Z:0" "$1"
    }

    file_content_identity_signature() {
      stat_value "%d:%i:%z:%Fm:%v" "%d:%i:%s:%Y:0" "$1"
    }

    mark_complete() {
      local candidate_path="$1"
      local expected_content_identity="$2"
      local current_content_identity
      local current_signature
      local marker_temp
      case "${REFERENCE_HASH_VERIFY:-1}" in
        0|false|FALSE|no|NO)
          touch "${marker_path}"
          return
          ;;
      esac
      if [[ "$(shasum -a 256 "${REFERENCE_MANIFEST_PATH}" | awk "{print \$1}")" \
          != "${REFERENCE_DOWNLOAD_MANIFEST_HASH}" ]]; then
        echo "setup.sh: reference manifest changed before ${file} marker publication" >&2
        return 1
      fi
      current_content_identity="$(file_content_identity_signature "${candidate_path}")"
      if [[ -z "${expected_content_identity}" \
          || "${current_content_identity}" != "${expected_content_identity}" ]]; then
        echo "setup.sh: ${file} changed before its verification marker was written" >&2
        return 1
      fi
      current_signature="$(file_metadata_signature "${candidate_path}")"
      marker_temp="$(mktemp "${marker_path}.tmp.XXXXXX")"
      if ! printf "version=2\t%s\t%s\t%s\t%s\n" \
          "${REFERENCE_DOWNLOAD_MANIFEST_HASH}" "${expected_hash}" \
          "${expected_size}" "${current_signature}" > "${marker_temp}" \
          || ! mv "${marker_temp}" "${marker_path}"; then
        rm -f "${marker_temp}"
        return 1
      fi
    }

    download_url_for_file() {
      local url="$1"
      local append_query=0
      local separator="?"

      case "${REFERENCE_APPEND_DOWNLOAD_QUERY:-auto}" in
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
          echo "setup.sh: MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY must be auto, true, or false" >&2
          return 1
          ;;
      esac

      if [[ "${append_query}" == "1" ]]; then
        if [[ "${url}" == *\?* ]]; then
          separator="&"
        fi
        url="${url}${separator}download=true"
      fi

      printf "%s\n" "${url}"
    }

    # Fetch one source attempt for this shard. Stall abort is delegated to
    # curl (--speed-limit/--speed-time); the transfer itself stays --silent
    # because the aggregate heartbeat in the parent process is the single
    # progress reporter for the parallel shard phase. The auth header is only
    # ever sent to the primary (private mirror) source; the public fallback
    # must not receive those credentials.
    fetch_shard_from_source() {
      local download_url="$1"
      local source_label="$2"
      local curl_args=(
        --fail
        --location
        --retry 5
        --retry-all-errors
        --retry-delay 2
        --continue-at -
        --speed-limit "${REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND}"
        --speed-time "${REFERENCE_DOWNLOAD_STALL_SECONDS}"
        --silent
        --show-error
        --output "${partial_path}"
      )

      if [[ "${source_label}" == "primary" && -n "${REFERENCE_AUTH_HEADER:-}" ]]; then
        curl_args+=(-H "${REFERENCE_AUTH_HEADER}")
      fi

      curl "${curl_args[@]}" "${download_url}"
    }

    reference_file_is_current() {
      local candidate_path="$1"
      local actual_size
      local actual_hash
      local before_signature
      local after_signature
      local before_content_identity
      local after_content_identity

      verified_content_identity=""

      case "${REFERENCE_HASH_VERIFY:-1}" in
        0|false|FALSE|no|NO)
          [[ -s "${candidate_path}" ]] || return 1
          verified_content_identity="$(file_content_identity_signature "${candidate_path}")"
          return 0
          ;;
        1|true|TRUE|yes|YES)
          ;;
        *)
          echo "setup.sh: MLXFAST_REFERENCE_HASH_VERIFY must be 0 or 1" >&2
          return 1
          ;;
      esac

      if [[ -z "${expected_hash}" || -z "${expected_size}" ]]; then
        echo "setup.sh: reference manifest has no entry for ${file}" >&2
        return 1
      fi
      if [[ ! -f "${candidate_path}" ]]; then
        return 1
      fi

      actual_size="$(wc -c < "${candidate_path}" | tr -d " ")"
      if [[ "${actual_size}" != "${expected_size}" ]]; then
        echo "setup.sh: cached shard ${ordinal}/${total}: ${file} size mismatch: expected ${expected_size}, got ${actual_size}"
        return 1
      fi
      before_signature="$(file_metadata_signature "${candidate_path}")"
      before_content_identity="$(file_content_identity_signature "${candidate_path}")"
      actual_hash="$(shasum -a 256 "${candidate_path}" | awk "{print \$1}")"
      after_signature="$(file_metadata_signature "${candidate_path}")"
      after_content_identity="$(file_content_identity_signature "${candidate_path}")"
      if [[ "${before_signature}" != "${after_signature}" \
          || "${before_content_identity}" != "${after_content_identity}" ]]; then
        echo "setup.sh: cached shard ${ordinal}/${total}: ${file} changed while SHA256 was being verified" >&2
        return 1
      fi
      if [[ "${actual_hash}" != "${expected_hash}" ]]; then
        echo "setup.sh: cached shard ${ordinal}/${total}: ${file} sha256 mismatch"
        echo "setup.sh: expected ${expected_hash}"
        echo "setup.sh: actual   ${actual_hash}"
        return 1
      fi
      if [[ "$(shasum -a 256 "${REFERENCE_MANIFEST_PATH}" | awk "{print \$1}")" \
          != "${REFERENCE_DOWNLOAD_MANIFEST_HASH}" ]]; then
        echo "setup.sh: reference manifest changed while shard ${ordinal}/${total}: ${file} was verified" >&2
        return 1
      fi

      verified_content_identity="${after_content_identity}"
      return 0
    }

    if reference_file_is_current "${output_path}"; then
      echo "setup.sh: using cached shard ${ordinal}/${total}: ${file}"
      mark_complete "${output_path}" "${verified_content_identity}"
      exit 0
    fi
    rm -f "${marker_path}"

    mkdir -p "$(dirname "${output_path}")"
    for base_url in "${base_urls[@]}"; do
      source_index=$((source_index + 1))
      source_label="primary"
      if [[ "${source_index}" -gt 1 ]]; then
        source_label="fallback"
        echo "setup.sh: trying fallback source for shard ${ordinal}/${total}: ${file}: ${base_url}"
      fi
      url="${base_url%/}/${file}"
      url="$(download_url_for_file "${url}")"

      attempt=1
      while [[ "${attempt}" -le 2 ]]; do
        if [[ "${attempt}" == "1" ]]; then
          echo "setup.sh: downloading shard ${ordinal}/${total}: ${file} source=${source_label}"
        else
          echo "setup.sh: redownloading shard ${ordinal}/${total}: ${file} from scratch after hash verification failed"
          rm -f "${partial_path}" "${marker_path}"
        fi

        curl_status=0
        fetch_shard_from_source "${url}" "${source_label}" || curl_status=$?
        if [[ "${curl_status}" != "0" ]]; then
          echo "setup.sh: shard ${ordinal}/${total} source failed or stalled (status=${curl_status}, source=${base_url})" >&2
          if [[ "${source_label}" == "fallback" && "${attempt}" == "1" ]]; then
            rm -f "${partial_path}" "${marker_path}"
            attempt=$((attempt + 1))
            continue
          fi
          break
        fi

        if reference_file_is_current "${partial_path}"; then
          verified_partial_identity="${verified_content_identity}"
          mv "${partial_path}" "${output_path}"
          mark_complete "${output_path}" "${verified_partial_identity}"
          echo "setup.sh: downloaded shard ${ordinal}/${total}: ${file} elapsed=$((SECONDS - started_seconds))s source=${source_label}"
          exit 0
        fi

        attempt=$((attempt + 1))
      done

      if [[ "${source_index}" -lt "${source_count}" && "${curl_status}" == "0" ]]; then
        rm -f "${partial_path}" "${marker_path}"
      fi
    done

    echo "setup.sh: failed to download verified shard ${ordinal}/${total}: ${file}" >&2
    exit 1
  ' _ {} "${output_dir}" "${total}" || download_status=$?
  stop_reference_download_heartbeat
  if [[ "${download_status}" != "0" ]]; then
    rm -f "${download_manifest_snapshot}"
    return "${download_status}"
  fi
  if [[ "$(reference_manifest_hash 2>/dev/null || true)" != "${download_manifest_hash}" ]]; then
    rm -f "${download_manifest_snapshot}"
    echo "setup.sh: reference manifest changed during parallel shard verification" >&2
    return 1
  fi
  rm -f "${download_manifest_snapshot}"
  echo "setup.sh: downloaded ${total}/${total} safetensors shard(s) elapsed=$(format_duration "$((SECONDS - started_seconds))")"
}

list_reference_shards() {
  local index_path="$1"

  ensure_swift_harness_ready || return 1

  "${SWIFT_BIN}" checkpoint-shards --index "${index_path}"
}

verify_reference_weights() {
  local reference_dir="$1"
  local index_path="${reference_dir}/model.safetensors.index.json"
  local shard_list
  local file
  local shard_files=()
  local missing=0

  if [[ ! -f "${reference_dir}/config.json" ]]; then
    echo "setup.sh: reference checkpoint is missing config.json at ${reference_dir}" >&2
    return 1
  fi
  if [[ ! -f "${index_path}" ]]; then
    echo "setup.sh: reference checkpoint is missing model.safetensors.index.json at ${reference_dir}" >&2
    return 1
  fi

  if ! shard_list="$(list_reference_shards "${index_path}")"; then
    return 1
  fi
  start_mlx_metallib_build || return 1
  while IFS= read -r file; do
    if [[ -n "${file}" ]]; then
      shard_files+=("${file}")
    fi
  done <<< "${shard_list}"
  if [[ "${#shard_files[@]}" -eq 0 ]]; then
    echo "setup.sh: checkpoint index did not list any safetensors shards" >&2
    return 1
  fi

  for file in "${shard_files[@]}"; do
    if [[ ! -s "${reference_dir}/${file}" ]]; then
      echo "setup.sh: reference checkpoint is missing shard ${file} at ${reference_dir}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" != "0" ]]; then
    return 1
  fi

  if reference_cache_lock_is_current "${reference_dir}"; then
    echo "setup.sh: verified reference checkpoint at ${reference_dir} (${#shard_files[@]} safetensors shard(s))"
    return 0
  fi

  if ! verify_reference_manifest "${reference_dir}"; then
    rm -f "${REFERENCE_CACHE_LOCK_PATH}"
    return 1
  fi
  if [[ -n "${REFERENCE_VERIFIED_SIGNATURES_PATH}" ]]; then
    if ! write_reference_cache_lock "${reference_dir}"; then
      return 1
    fi
  else
    rm -f "${REFERENCE_CACHE_LOCK_PATH}"
  fi
  echo "setup.sh: verified reference checkpoint at ${reference_dir} (${#shard_files[@]} safetensors shard(s))"
}

verify_reference_weights_after_verified_download() {
  local reference_dir="$1"
  local index_path="${reference_dir}/model.safetensors.index.json"
  local hash_status
  local shard_list
  local file
  local shard_files=()
  local missing=0

  if reference_hash_verification_enabled; then
    hash_status=0
  else
    hash_status="$?"
  fi
  if [[ "${hash_status}" != "0" ]]; then
    echo "setup.sh: cannot skip post-download full verification unless MLXFAST_REFERENCE_HASH_VERIFY=1" >&2
    return 1
  fi

  if [[ ! -f "${reference_dir}/config.json" ]]; then
    echo "setup.sh: reference checkpoint is missing config.json at ${reference_dir}" >&2
    return 1
  fi
  if [[ ! -f "${index_path}" ]]; then
    echo "setup.sh: reference checkpoint is missing model.safetensors.index.json at ${reference_dir}" >&2
    return 1
  fi

  if ! shard_list="$(list_reference_shards "${index_path}")"; then
    return 1
  fi
  start_mlx_metallib_build || return 1
  while IFS= read -r file; do
    if [[ -n "${file}" ]]; then
      shard_files+=("${file}")
    fi
  done <<< "${shard_list}"
  if [[ "${#shard_files[@]}" -eq 0 ]]; then
    echo "setup.sh: checkpoint index did not list any safetensors shards" >&2
    return 1
  fi

  for file in "${shard_files[@]}"; do
    if [[ ! -s "${reference_dir}/${file}" ]]; then
      echo "setup.sh: reference checkpoint is missing shard ${file} at ${reference_dir}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" != "0" ]]; then
    return 1
  fi

  if ! verify_reference_manifest_sizes "${reference_dir}"; then
    rm -f "${REFERENCE_CACHE_LOCK_PATH}"
    return 1
  fi
  if ! write_reference_cache_lock "${reference_dir}"; then
    return 1
  fi
  echo "setup.sh: verified reference checkpoint at ${reference_dir} (${#shard_files[@]} safetensors shard(s)); skipped second SHA256 pass after verified downloads"
}

verify_reference_manifest_sizes() {
  local reference_dir="$1"
  local line
  local expected_hash
  local expected_size
  local relative_path
  local extra
  local file_path
  local marker_path
  local marker_line
  local marker_version
  local marker_manifest_hash
  local marker_hash
  local marker_size
  local marker_signature
  local marker_extra
  local actual_size
  local actual_signature
  local verified_manifest_hash
  local checked=0

  discard_reference_verified_signatures

  if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
    echo "setup.sh: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi
  if ! create_reference_verified_manifest_snapshot; then
    echo "setup.sh: failed to snapshot reference manifest" >&2
    discard_reference_verified_signatures
    return 1
  fi
  verified_manifest_hash="${REFERENCE_VERIFIED_MANIFEST_HASH}"
  if ! create_reference_verified_signatures_path; then
    echo "setup.sh: failed to create verified download signature record" >&2
    discard_reference_verified_signatures
    return 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r expected_hash expected_size relative_path extra <<< "${line}"
    if [[ -n "${extra:-}" || -z "${expected_hash:-}" || -z "${expected_size:-}" || -z "${relative_path:-}" ]]; then
      echo "setup.sh: malformed reference manifest line: ${line}" >&2
      discard_reference_verified_signatures
      return 1
    fi
    if [[ ! "${expected_hash}" =~ ^[0-9a-f]{64}$ || ! "${expected_size}" =~ ^[0-9]+$ ]]; then
      echo "setup.sh: malformed reference manifest line: ${line}" >&2
      discard_reference_verified_signatures
      return 1
    fi
    if [[ "${relative_path}" == /* || "${relative_path}" == *\\* ]]; then
      echo "setup.sh: unsafe reference manifest path: ${relative_path}" >&2
      discard_reference_verified_signatures
      return 1
    fi
    case "/${relative_path}/" in
      *"/../"*|*"/./"*)
        echo "setup.sh: unsafe reference manifest path: ${relative_path}" >&2
        discard_reference_verified_signatures
        return 1
        ;;
    esac

    file_path="${reference_dir}/${relative_path}"
    if [[ ! -f "${file_path}" ]]; then
      echo "setup.sh: reference checkpoint is missing manifest file ${relative_path}" >&2
      discard_reference_verified_signatures
      return 1
    fi
    actual_size="$(wc -c < "${file_path}" | tr -d ' ')"
    if [[ "${actual_size}" != "${expected_size}" ]]; then
      echo "setup.sh: reference file ${relative_path} size mismatch: expected ${expected_size}, got ${actual_size}" >&2
      discard_reference_verified_signatures
      return 1
    fi
    marker_path="${file_path}.complete"
    if [[ ! -f "${marker_path}" ]]; then
      echo "setup.sh: verified download marker missing for ${relative_path}" >&2
      discard_reference_verified_signatures
      return 1
    fi
    IFS= read -r marker_line < "${marker_path}" || {
      discard_reference_verified_signatures
      return 1
    }
    IFS=$'\t' read -r marker_version marker_manifest_hash marker_hash marker_size marker_signature marker_extra \
      <<< "${marker_line}"
    if [[ "${marker_version}" != "version=2" \
        || "${marker_manifest_hash}" != "${verified_manifest_hash}" \
        || "${marker_hash}" != "${expected_hash}" \
        || "${marker_size}" != "${expected_size}" \
        || -z "${marker_signature}" || -n "${marker_extra:-}" ]]; then
      echo "setup.sh: invalid verified download marker for ${relative_path}" >&2
      discard_reference_verified_signatures
      return 1
    fi
    actual_signature="$(file_metadata_signature "${file_path}")" || {
      discard_reference_verified_signatures
      return 1
    }
    if [[ "${actual_signature}" != "${marker_signature}" ]]; then
      echo "setup.sh: reference file ${relative_path} changed after download verification" >&2
      discard_reference_verified_signatures
      return 1
    fi
    printf '%s\t%s\t%s\n' \
      "${relative_path}" "${expected_size}" "${marker_signature}" \
      >> "${REFERENCE_VERIFIED_SIGNATURES_PATH}"
    checked=$((checked + 1))
  done < "${REFERENCE_VERIFIED_MANIFEST_PATH}"

  if [[ "${checked}" -eq 0 ]]; then
    echo "setup.sh: reference manifest contained no files: ${REFERENCE_MANIFEST_PATH}" >&2
    discard_reference_verified_signatures
    return 1
  fi
  if [[ "$(reference_manifest_hash 2>/dev/null || true)" != "${verified_manifest_hash}" ]]; then
    echo "setup.sh: reference manifest changed while download markers were aggregated" >&2
    discard_reference_verified_signatures
    return 1
  fi
  echo "setup.sh: verified ${checked} downloaded reference file signature(s)"
}

setup_parallel_metallib_enabled() {
  case "${SETUP_PARALLEL_METALLIB}" in
    0|false|FALSE|no|NO)
      return 1
      ;;
    1|true|TRUE|yes|YES)
      return 0
      ;;
    *)
      echo "setup.sh: MLXFAST_SETUP_PARALLEL_METALLIB must be 0 or 1" >&2
      return 2
      ;;
  esac
}

publish_lock_symlink() {
  local target="$1"
  local lock_path="$2"
  # GNU ln rejects BSD -h; retry with -T only if no competing lock appeared.
  if ln -s -h -- "${target}" "${lock_path}" 2>/dev/null; then
    return 0
  fi
  [[ ! -e "${lock_path}" && ! -L "${lock_path}" ]] || return 1
  ln -s -T -- "${target}" "${lock_path}" 2>/dev/null
}

reference_cache_lock_generation_path() {
  local lock_dir="${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
  local raw_target
  local candidate
  local canonical_lock_parent
  local canonical_candidate
  local expected_prefix
  local lock_uid

  [[ -L "${lock_dir}" ]] || return 1
  raw_target="$(readlink "${lock_dir}")" || return 1
  [[ "${raw_target}" == /* ]] || return 1
  candidate="${raw_target}"
  canonical_lock_parent="$(cd -P "$(dirname "${lock_dir}")" && pwd -P)" || return 1
  [[ -d "$(dirname "${candidate}")" ]] || return 1
  canonical_candidate="$(cd -P "$(dirname "${candidate}")" && pwd -P)/$(basename "${candidate}")"
  expected_prefix="$(basename "${lock_dir}").generation."
  [[ "$(dirname "${canonical_candidate}")" == "${canonical_lock_parent}" ]] || return 1
  [[ "$(basename "${canonical_candidate}")" == "${expected_prefix}"* ]] || return 1
  lock_uid="$(stat_value '%u' '%u' "${lock_dir}")" || return 1
  [[ "${lock_uid}" == "$(id -u)" ]] || return 1
  if [[ -e "${canonical_candidate}" || -L "${canonical_candidate}" ]]; then
    [[ -d "${canonical_candidate}" && ! -L "${canonical_candidate}" \
        && -O "${canonical_candidate}" ]] || return 1
  fi
  printf '%s\n' "${canonical_candidate}"
}

remove_reference_cache_lock_generation() {
  local generation_dir="$1"
  local published_target=""

  if [[ -L "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]; then
    published_target="$(readlink "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" 2>/dev/null || true)"
  fi
  if [[ -n "${generation_dir}" && "${published_target}" == "${generation_dir}" ]]; then
    rm -f "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" || return 1
  fi
  if [[ -n "${generation_dir}" ]]; then
    rm -rf "${generation_dir}" || return 1
  fi
}

reclaim_orphaned_reference_cache_lock_generations() {
  local lock_dir="${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
  local lock_parent
  local lock_name
  local local_host_hash
  local initializing_prefix
  local initializing_dir
  local initializing_name
  local initializing_remainder
  local initializing_pid
  local published_target=""
  local generation_dir
  local canonical_generation
  local owner_path
  local owner_line
  local owner_pid
  local owner_host
  local owner_token
  local field
  local claim_dir

  lock_parent="$(cd -P "$(dirname "${lock_dir}")" && pwd -P)" || return 1
  lock_name="$(basename "${lock_dir}")"
  local_host_hash="$(printf '%s' "$(hostname)" | shasum -a 256 | awk '{print substr($1, 1, 16)}')" \
    || return 1
  initializing_prefix="${lock_name}.initializing.${local_host_hash}."
  if [[ -L "${lock_dir}" ]]; then
    published_target="$(readlink "${lock_dir}" 2>/dev/null || true)"
  fi

  # Initializing directories encode their creator's host and PID before they
  # exist. A scan can therefore reclaim an empty SIGKILL orphan without ever
  # touching a live creator paused before its owner file is moved into place.
  while IFS= read -r -d '' initializing_dir; do
    [[ ! -L "${initializing_dir}" && -d "${initializing_dir}" \
        && -O "${initializing_dir}" ]] || continue
    initializing_name="$(basename "${initializing_dir}")"
    initializing_remainder="${initializing_name#"${initializing_prefix}"}"
    initializing_pid="${initializing_remainder%%.*}"
    [[ "${initializing_pid}" =~ ^[1-9][0-9]*$ ]] || continue
    kill -0 "${initializing_pid}" >/dev/null 2>&1 && continue
    [[ "$(cd -P "${initializing_dir}" && pwd -P)" == "${lock_parent}/${initializing_name}" ]] \
      || continue
    rm -rf "${initializing_dir}" || return 1
  done < <(find "${lock_parent}" -mindepth 1 -maxdepth 1 -type d \
    -name "${initializing_prefix}*" -print0)

  while IFS= read -r -d '' generation_dir; do
    [[ ! -L "${generation_dir}" && -d "${generation_dir}" && -O "${generation_dir}" ]] \
      || continue
    canonical_generation="$(cd -P "${generation_dir}" && pwd -P)" || continue
    [[ "$(dirname "${canonical_generation}")" == "${lock_parent}" ]] || continue
    [[ "${canonical_generation}" != "${published_target}" ]] || continue

    owner_path="${canonical_generation}/owner"
    [[ -f "${owner_path}" && ! -L "${owner_path}" ]] || continue
    owner_line="$(cat "${owner_path}" 2>/dev/null || true)"
    owner_pid=""
    owner_host=""
    owner_token=""
    for field in ${owner_line}; do
      case "${field}" in
        pid=*) owner_pid="${field#pid=}" ;;
        host=*) owner_host="${field#host=}" ;;
        token=*) owner_token="${field#token=}" ;;
      esac
    done
    [[ -n "${owner_token}" && "${owner_pid}" =~ ^[1-9][0-9]*$ \
        && "${owner_host}" == "$(hostname)" ]] || continue
    kill -0 "${owner_pid}" >/dev/null 2>&1 && continue

    claim_dir="${canonical_generation}/.orphan-cleanup-claim"
    mkdir "${claim_dir}" 2>/dev/null || continue
    if [[ -L "${lock_dir}" ]]; then
      published_target="$(readlink "${lock_dir}" 2>/dev/null || true)"
    else
      published_target=""
    fi
    if [[ "${canonical_generation}" == "${published_target}" \
        || "$(cat "${owner_path}" 2>/dev/null || true)" != "${owner_line}" ]]; then
      rmdir "${claim_dir}" 2>/dev/null || true
      continue
    fi
    rm -rf "${canonical_generation}" || return 1
  done < <(find "${lock_parent}" -mindepth 1 -maxdepth 1 -type d \
    -name "${lock_name}.generation.*" -print0)
}

recover_stale_reference_cache_mutation_lock() {
  local lock_dir="${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
  local owner_path="${lock_dir}/owner"
  local owner_line=""
  local owner_pid=""
  local owner_host=""
  local owner_token=""
  local observed_identity
  local claimed_identity
  local claim_dir
  local quarantine_dir
  local field
  local stale=0
  local owner_is_well_formed=0
  local owner_existed=0
  local metadata_path="${lock_dir}"
  local generation_dir=""
  local generation_missing=0

  observed_identity="$(directory_identity "${lock_dir}" 2>/dev/null)" || return 1
  if [[ -L "${lock_dir}" ]]; then
    generation_dir="$(reference_cache_lock_generation_path)" || return 1
    if [[ ! -e "${generation_dir}" && ! -L "${generation_dir}" ]]; then
      generation_missing=1
    fi
  fi
  if [[ "${generation_missing}" == "1" ]]; then
    claim_dir="${lock_dir}.recovery-claim"
  else
    claim_dir="${lock_dir}/.recovery-claim"
  fi

  if [[ -f "${owner_path}" ]]; then
    owner_existed=1
    owner_line="$(cat "${owner_path}" 2>/dev/null || true)"
    for field in ${owner_line}; do
      case "${field}" in
        pid=*) owner_pid="${field#pid=}" ;;
        host=*) owner_host="${field#host=}" ;;
        token=*) owner_token="${field#token=}" ;;
      esac
    done
    if [[ -n "${owner_token}" && -n "${owner_host}" \
        && "${owner_pid}" =~ ^[1-9][0-9]*$ ]]; then
      owner_is_well_formed=1
      if [[ "${owner_host}" == "$(hostname)" ]] \
          && ! kill -0 "${owner_pid}" >/dev/null 2>&1; then
        stale=1
      fi
    else
      metadata_path="${owner_path}"
    fi
  fi

  if [[ "${generation_missing}" == "1" ]]; then
    stale=1
  elif [[ "${stale}" != "1" && "${owner_is_well_formed}" != "1" ]]; then
    local metadata_mtime
    local now
    if metadata_mtime="$(file_mtime_seconds "${metadata_path}" 2>/dev/null)"; then
      now="$(date +%s)"
      if (( now - metadata_mtime >= REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS )); then
        stale=1
      fi
    fi
  fi

  if [[ "${stale}" != "1" ]]; then
    return 1
  fi
  # Pin this exact directory instance before deleting anything. If another
  # waiter already replaced the stale directory, the inode comparison fails;
  # once the claim exists, the directory cannot be removed out from under us.
  if ! mkdir "${claim_dir}" 2>/dev/null; then
    return 1
  fi
  claimed_identity="$(directory_identity "${lock_dir}" 2>/dev/null || true)"
  if [[ "${claimed_identity}" != "${observed_identity}" ]]; then
    rmdir "${claim_dir}" 2>/dev/null || true
    return 1
  fi
  if [[ "${owner_existed}" == "1" ]]; then
    if [[ "$(cat "${owner_path}" 2>/dev/null || true)" != "${owner_line}" ]]; then
      rmdir "${claim_dir}" 2>/dev/null || true
      return 1
    fi
  elif [[ -e "${owner_path}" ]]; then
    rmdir "${claim_dir}" 2>/dev/null || true
    return 1
  fi

  quarantine_dir="${lock_dir}.stale-$$-${RANDOM}"
  if ! mv "${lock_dir}" "${quarantine_dir}"; then
    rmdir "${claim_dir}" 2>/dev/null || true
    return 1
  fi
  if [[ "${generation_missing}" == "1" ]]; then
    rmdir "${claim_dir}" 2>/dev/null || true
  fi
  if [[ -n "${generation_dir}" ]]; then
    rm -f "${quarantine_dir}" || return 1
    rm -rf "${generation_dir}" || return 1
  else
    rm -rf "${quarantine_dir}" || return 1
  fi
  echo "setup.sh: recovered stale reference cache mutation lock ${lock_dir}"
  return 0
}

acquire_reference_cache_mutation_lock() {
  local lock_dir="${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
  local started_seconds="${SECONDS}"
  local announced_wait=0
  local owner_temp
  local owner_pid
  local token
  local generation_dir
  local generation_dir_raw
  local initializing_dir
  local initializing_dir_raw
  local local_host_hash
  local nested_publication

  if ! [[ "${REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS must be a non-negative integer" >&2
    return 1
  fi
  if ! [[ "${REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS must be a positive integer" >&2
    return 1
  fi
  mkdir -p "$(dirname "${lock_dir}")" || return 1
  reclaim_orphaned_reference_cache_lock_generations || return 1
  if ! current_shell_pid; then
    return 1
  fi
  owner_pid="${CURRENT_SHELL_PID}"
  token="${owner_pid}-$(date +%s)-${RANDOM}-${RANDOM}"
  local_host_hash="$(printf '%s' "$(hostname)" | shasum -a 256 | awk '{print substr($1, 1, 16)}')" \
    || return 1

  while true; do
    owner_temp="$(mktemp "${lock_dir}.owner.XXXXXX")" || {
      echo "setup.sh: failed to create reference cache mutation owner record" >&2
      return 1
    }
    REFERENCE_CACHE_MUTATION_OWNER_TEMP="${owner_temp}"
    if ! printf 'token=%s pid=%s host=%s started_at=%s\n' \
        "${token}" "${owner_pid}" "$(hostname)" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "${owner_temp}"; then
      rm -f "${owner_temp}"
      REFERENCE_CACHE_MUTATION_OWNER_TEMP=""
      echo "setup.sh: failed to initialize reference cache mutation lock owner" >&2
      return 1
    fi
    initializing_dir_raw="$(mktemp -d \
      "${lock_dir}.initializing.${local_host_hash}.${owner_pid}.XXXXXX")" || {
      rm -f "${owner_temp}"
      REFERENCE_CACHE_MUTATION_OWNER_TEMP=""
      echo "setup.sh: failed to create reference cache mutation lock generation" >&2
      return 1
    }
    initializing_dir="$(cd -P "${initializing_dir_raw}" && pwd -P)" || {
      rm -f "${owner_temp}"
      rm -rf "${initializing_dir_raw}"
      REFERENCE_CACHE_MUTATION_OWNER_TEMP=""
      return 1
    }
    REFERENCE_CACHE_MUTATION_LOCK_INITIALIZING_DIR="${initializing_dir}"
    if ! mv "${owner_temp}" "${initializing_dir}/owner"; then
      rm -f "${owner_temp}"
      rm -rf "${initializing_dir}"
      REFERENCE_CACHE_MUTATION_OWNER_TEMP=""
      REFERENCE_CACHE_MUTATION_LOCK_INITIALIZING_DIR=""
      return 1
    fi
    REFERENCE_CACHE_MUTATION_OWNER_TEMP=""
    generation_dir_raw="${lock_dir}.generation.${token}-${RANDOM}"
    if [[ -e "${generation_dir_raw}" || -L "${generation_dir_raw}" ]] \
        || ! mv "${initializing_dir}" "${generation_dir_raw}"; then
      rm -rf "${initializing_dir}"
      REFERENCE_CACHE_MUTATION_LOCK_INITIALIZING_DIR=""
      echo "setup.sh: failed to publish initialized reference cache lock generation" >&2
      return 1
    fi
    REFERENCE_CACHE_MUTATION_LOCK_INITIALIZING_DIR=""
    REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR="${generation_dir_raw}"
    generation_dir="$(cd -P "${generation_dir_raw}" && pwd -P)" || return 1
    REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR="${generation_dir}"

    # The public lock appears in one symlink(2) operation only after its unique
    # generation already contains a complete owner record. A stalled creator
    # therefore has no ownerless public directory that recovery can replace.
    if [[ ! -e "${lock_dir}" && ! -L "${lock_dir}" ]] \
        && publish_lock_symlink "${generation_dir}" "${lock_dir}"; then
      if [[ -L "${lock_dir}" \
          && "$(readlink "${lock_dir}" 2>/dev/null || true)" == "${generation_dir}" ]]; then
        REFERENCE_CACHE_MUTATION_LOCK_TOKEN="${token}"
        REFERENCE_CACHE_MUTATION_LOCK_HELD=1
        echo "setup.sh: acquired reference cache mutation lock ${lock_dir}"
        return 0
      fi
      # A legacy creator can race the precheck by mkdir'ing the fixed path.
      # BSD ln then places our symlink inside that directory; remove only that
      # exact self-target and continue through legacy stale-lock recovery.
      nested_publication="${lock_dir}/$(basename "${generation_dir}")"
      if [[ -L "${nested_publication}" \
          && "$(readlink "${nested_publication}" 2>/dev/null || true)" == "${generation_dir}" ]]; then
        rm -f "${nested_publication}" || return 1
      fi
    fi

    rm -rf "${generation_dir}" || return 1
    REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR=""
    # Fatal only for a genuine obstruction (an existing non-directory,
    # non-symlink entry such as a plain file). An absent path here is the
    # normal transient state after a concurrent holder released the lock
    # between our failed publish attempt and this check; retry instead.
    if [[ -e "${lock_dir}" && ! -d "${lock_dir}" && ! -L "${lock_dir}" ]]; then
      echo "setup.sh: reference cache mutation lock path exists and is not a lock directory: ${lock_dir}" >&2
      return 1
    fi
    if recover_stale_reference_cache_mutation_lock; then
      continue
    fi
    if [[ "${announced_wait}" == "0" ]]; then
      echo "setup.sh: waiting for reference cache mutation lock ${lock_dir}"
      announced_wait=1
    fi
    if (( SECONDS - started_seconds >= REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS )); then
      echo "setup.sh: timed out waiting for reference cache mutation lock ${lock_dir}" >&2
      if [[ -f "${lock_dir}/owner" ]]; then
        echo "setup.sh: lock owner: $(cat "${lock_dir}/owner")" >&2
      fi
      return 1
    fi
    sleep 1
  done
}

release_reference_cache_mutation_lock() {
  local lock_dir="${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
  local owner_path="${lock_dir}/owner"
  local owner_line
  local owner_token=""
  local published_target=""
  local field
  if [[ "${REFERENCE_CACHE_MUTATION_LOCK_HELD}" != "1" ]]; then
    return 0
  fi

  owner_line="$(cat "${owner_path}" 2>/dev/null || true)"
  for field in ${owner_line}; do
    case "${field}" in
      token=*) owner_token="${field#token=}" ;;
    esac
  done
  if [[ -z "${REFERENCE_CACHE_MUTATION_LOCK_TOKEN}" \
      || "${owner_token}" != "${REFERENCE_CACHE_MUTATION_LOCK_TOKEN}" ]]; then
    echo "setup.sh: refusing to release reference cache mutation lock owned by another process: ${lock_dir}" >&2
    return 1
  fi

  if [[ -L "${lock_dir}" ]]; then
    published_target="$(readlink "${lock_dir}" 2>/dev/null || true)"
    if [[ -z "${REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR}" \
        || "${published_target}" != "${REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR}" ]]; then
      echo "setup.sh: refusing to release a replaced reference cache mutation lock: ${lock_dir}" >&2
      return 1
    fi
    rm -f "${lock_dir}" || return 1
    REFERENCE_CACHE_MUTATION_LOCK_HELD=0
    REFERENCE_CACHE_MUTATION_LOCK_TOKEN=""
    if ! rm -rf "${REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR}"; then
      echo "setup.sh: failed to remove released reference cache lock generation" >&2
      return 1
    fi
    REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR=""
  else
    rm -f "${owner_path}" || return 1
    if ! rmdir "${lock_dir}"; then
      echo "setup.sh: failed to release reference cache mutation lock ${lock_dir}" >&2
      return 1
    fi
    REFERENCE_CACHE_MUTATION_LOCK_HELD=0
    REFERENCE_CACHE_MUTATION_LOCK_TOKEN=""
  fi
  echo "setup.sh: released reference cache mutation lock ${lock_dir}"
}

terminate_metallib_process_group() {
  local process_group="$1"
  local attempt
  kill -TERM -- "-${process_group}" >/dev/null 2>&1 || return 0
  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if ! kill -0 -- "-${process_group}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  kill -KILL -- "-${process_group}" >/dev/null 2>&1 || true
}

cleanup_background_builds() {
  local status="$?"
  stop_reference_download_heartbeat
  discard_reference_verified_signatures
  if [[ -n "${REFERENCE_CACHE_MUTATION_OWNER_TEMP}" ]]; then
    rm -f "${REFERENCE_CACHE_MUTATION_OWNER_TEMP}" || true
    REFERENCE_CACHE_MUTATION_OWNER_TEMP=""
  fi
  if [[ -n "${REFERENCE_CACHE_MUTATION_LOCK_INITIALIZING_DIR}" ]]; then
    rm -rf "${REFERENCE_CACHE_MUTATION_LOCK_INITIALIZING_DIR}" || true
    REFERENCE_CACHE_MUTATION_LOCK_INITIALIZING_DIR=""
  fi
  if [[ "${REFERENCE_CACHE_MUTATION_LOCK_HELD}" != "1" \
      && -n "${REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR}" ]]; then
    remove_reference_cache_lock_generation \
      "${REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR}" || true
    REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR=""
  fi
  if [[ "${status}" != "0" ]]; then
    if [[ -n "${METALLIB_BUILD_PID}" ]] && kill -0 "${METALLIB_BUILD_PID}" >/dev/null 2>&1; then
      if [[ -n "${METALLIB_BUILD_PROCESS_GROUP}" ]]; then
        terminate_metallib_process_group "${METALLIB_BUILD_PROCESS_GROUP}"
      else
        kill "${METALLIB_BUILD_PID}" >/dev/null 2>&1 || true
      fi
    fi
  fi
  if [[ -n "${METALLIB_BUILD_PID}" ]]; then
    wait "${METALLIB_BUILD_PID}" >/dev/null 2>&1 || true
  fi
  if ! release_reference_cache_mutation_lock; then
    if [[ "${status}" == "0" ]]; then
      status=1
    fi
  fi
  return "${status}"
}

# The dependency graph is frozen by challenge policy: before either build
# begins, Package.swift and Package.resolved must match the committed state
# (SwiftPM re-resolution, a submission, or local edits all show up as a
# work-tree diff). Skips quietly where git is unavailable, and
# --force-resolved-versions on the builds makes SwiftPM itself fail closed
# instead of silently re-resolving an out-of-date graph.
assert_frozen_dependency_graph() {
  command -v git >/dev/null 2>&1 || return 0
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0
  local manifest
  for manifest in Package.swift Package.resolved; do
    git cat-file -e "HEAD:${manifest}" 2>/dev/null || return 0
    if ! git diff --quiet HEAD -- "${manifest}" 2>/dev/null; then
      echo "setup.sh: ${manifest} differs from the committed state; the dependency graph is frozen by challenge policy" >&2
      echo "setup.sh: restore it (git checkout -- ${manifest}) and rerun" >&2
      return 1
    fi
  done
}

build_swift_harness() {
  echo "setup.sh: building trusted Swift harness and participant runtime worker"
  assert_frozen_dependency_graph || return 1
  # Independent SwiftPM build/cache roots: the trusted CLI builds in .build
  # and the participant worker (which compiles the vendored MLX forks) in
  # .build-worker, each with its own clang module cache, so a
  # participant-code build can never write into the trusted product tree.
  # An explicitly exported CLANG_MODULE_CACHE_PATH wins for both builds.
  mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
  CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${PWD}/.build/clang-module-cache}" \
    swift build -c release --force-resolved-versions --product mlxfast-swift
  CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${PWD}/.build-worker/clang-module-cache}" \
    swift build -c release --force-resolved-versions --scratch-path .build-worker --product mlxfast-runtime-worker
  if [[ ! -x "${SWIFT_BIN}" ]]; then
    echo "setup.sh: trusted Swift CLI missing at ${SWIFT_BIN}; build failed or MLXFAST_SWIFT_BIN is wrong" >&2
    return 1
  fi
  if [[ ! -x "${RUNTIME_WORKER_BIN}" ]]; then
    echo "setup.sh: participant runtime worker missing at ${RUNTIME_WORKER_BIN}; build failed or MLXFAST_RUNTIME_WORKER_EXECUTABLE is wrong" >&2
    return 1
  fi
}

ensure_swift_harness_ready() {
  if [[ ! -x "${SWIFT_BIN}" ]]; then
    echo "setup.sh: trusted Swift CLI missing at ${SWIFT_BIN}; build failed or MLXFAST_SWIFT_BIN is wrong" >&2
    return 1
  fi
  if [[ ! -x "${RUNTIME_WORKER_BIN}" ]]; then
    echo "setup.sh: participant runtime worker missing at ${RUNTIME_WORKER_BIN}; build failed or MLXFAST_RUNTIME_WORKER_EXECUTABLE is wrong" >&2
    return 1
  fi
}

build_mlx_metallib() {
  if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
    echo "setup.sh: skipping mlx.metallib build"
    return 0
  fi
  ensure_cmake || return 1
  ensure_metal_toolchain || return 1
  echo "setup.sh: building mlx.metallib for MLX Swift runtime"
  tools/build-mlx-metallib.sh || return 1
}

start_mlx_metallib_build() {
  local parallel_status
  if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
    METALLIB_BUILD_STATE="completed"
    return 0
  fi
  case "${METALLIB_BUILD_STATE}" in
    running|completed)
      return 0
      ;;
    failed)
      echo "setup.sh: mlx.metallib build already failed" >&2
      return 1
      ;;
    not_started)
      ;;
    *)
      echo "setup.sh: invalid mlx.metallib build state: ${METALLIB_BUILD_STATE}" >&2
      return 1
      ;;
  esac
  if setup_parallel_metallib_enabled; then
    METALLIB_BUILD_STATE="running"
    # Monitor mode gives the background build its own process group, allowing
    # failure cleanup to terminate CMake and compiler descendants as a unit.
    set -m
    build_mlx_metallib &
    METALLIB_BUILD_PID="$!"
    METALLIB_BUILD_PROCESS_GROUP="${METALLIB_BUILD_PID}"
    set +m
    echo "setup.sh: mlx.metallib build running in background pid=${METALLIB_BUILD_PID}"
  else
    parallel_status="$?"
    if [[ "${parallel_status}" != "1" ]]; then
      return "${parallel_status}"
    fi
    METALLIB_BUILD_STATE="running"
    if ! build_mlx_metallib; then
      METALLIB_BUILD_STATE="failed"
      return 1
    fi
    METALLIB_BUILD_STATE="completed"
  fi
}

wait_for_mlx_metallib_build() {
  if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
    return 0
  fi
  case "${METALLIB_BUILD_STATE}" in
    running)
      if [[ -z "${METALLIB_BUILD_PID}" ]]; then
        echo "setup.sh: mlx.metallib build is running without a background pid" >&2
        METALLIB_BUILD_STATE="failed"
        return 1
      fi
      echo "setup.sh: waiting for mlx.metallib build"
      if ! wait "${METALLIB_BUILD_PID}"; then
        METALLIB_BUILD_PID=""
        METALLIB_BUILD_PROCESS_GROUP=""
        METALLIB_BUILD_STATE="failed"
        echo "setup.sh: mlx.metallib build failed" >&2
        return 1
      fi
      METALLIB_BUILD_PID=""
      METALLIB_BUILD_PROCESS_GROUP=""
      METALLIB_BUILD_STATE="completed"
      ;;
    completed)
      ;;
    failed)
      echo "setup.sh: mlx.metallib build failed" >&2
      return 1
      ;;
    not_started)
      echo "setup.sh: mlx.metallib build was not started" >&2
      return 1
      ;;
    *)
      echo "setup.sh: invalid mlx.metallib build state: ${METALLIB_BUILD_STATE}" >&2
      return 1
      ;;
  esac
  if [[ ! -f "${MLX_METALLIB}" ]]; then
    echo "setup.sh: mlx.metallib missing at ${MLX_METALLIB}" >&2
    return 1
  fi
}

verify_reference_manifest() {
  local reference_dir="$1"
  local line
  local expected_hash
  local expected_size
  local relative_path
  local extra
  local file_path
  local actual_size
  local actual_hash
  local before_signature
  local after_signature
  local before_content_identity
  local after_content_identity
  local verified_manifest_hash
  local checked=0

  discard_reference_verified_signatures

  case "${REFERENCE_HASH_VERIFY}" in
    0|false|FALSE|no|NO)
      echo "setup.sh: skipping reference SHA256 verification"
      return 0
      ;;
    1|true|TRUE|yes|YES)
      ;;
    *)
      echo "setup.sh: MLXFAST_REFERENCE_HASH_VERIFY must be 0 or 1" >&2
      return 1
      ;;
  esac

  if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
    echo "setup.sh: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi
  if ! create_reference_verified_manifest_snapshot; then
    echo "setup.sh: failed to snapshot reference manifest" >&2
    discard_reference_verified_signatures
    return 1
  fi
  verified_manifest_hash="${REFERENCE_VERIFIED_MANIFEST_HASH}"
  if ! create_reference_verified_signatures_path; then
    echo "setup.sh: failed to create bound reference verification record" >&2
    discard_reference_verified_signatures
    return 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r expected_hash expected_size relative_path extra <<< "${line}"
    if [[ -n "${extra:-}" || -z "${expected_hash:-}" || -z "${expected_size:-}" || -z "${relative_path:-}" ]]; then
      echo "setup.sh: malformed reference manifest line: ${line}" >&2
      return 1
    fi
    if [[ ! "${expected_hash}" =~ ^[0-9a-f]{64}$ || ! "${expected_size}" =~ ^[0-9]+$ ]]; then
      echo "setup.sh: malformed reference manifest line: ${line}" >&2
      return 1
    fi
    if [[ "${relative_path}" == /* || "${relative_path}" == *\\* ]]; then
      echo "setup.sh: unsafe reference manifest path: ${relative_path}" >&2
      return 1
    fi
    case "/${relative_path}/" in
      *"/../"*|*"/./"*)
        echo "setup.sh: unsafe reference manifest path: ${relative_path}" >&2
        return 1
        ;;
    esac

    file_path="${reference_dir}/${relative_path}"
    if [[ ! -f "${file_path}" ]]; then
      echo "setup.sh: reference checkpoint is missing manifest file ${relative_path}" >&2
      discard_reference_verified_signatures
      return 1
    fi
    actual_size="$(wc -c < "${file_path}" | tr -d ' ')"
    if [[ "${actual_size}" != "${expected_size}" ]]; then
      echo "setup.sh: reference file ${relative_path} size mismatch: expected ${expected_size}, got ${actual_size}" >&2
      discard_reference_verified_signatures
      return 1
    fi
    before_signature="$(file_metadata_signature "${file_path}")" || {
      discard_reference_verified_signatures
      return 1
    }
    before_content_identity="$(file_content_identity_signature "${file_path}")" || {
      discard_reference_verified_signatures
      return 1
    }
    if ! actual_hash="$(shasum -a 256 "${file_path}" | awk '{print $1}')"; then
      discard_reference_verified_signatures
      return 1
    fi
    after_signature="$(file_metadata_signature "${file_path}")" || {
      discard_reference_verified_signatures
      return 1
    }
    after_content_identity="$(file_content_identity_signature "${file_path}")" || {
      discard_reference_verified_signatures
      return 1
    }
    if [[ "${before_signature}" != "${after_signature}" \
        || "${before_content_identity}" != "${after_content_identity}" ]]; then
      echo "setup.sh: reference file ${relative_path} changed while SHA256 was being verified" >&2
      discard_reference_verified_signatures
      return 1
    fi
    if [[ "${actual_hash}" != "${expected_hash}" ]]; then
      echo "setup.sh: reference file ${relative_path} sha256 mismatch" >&2
      echo "setup.sh: expected ${expected_hash}" >&2
      echo "setup.sh: actual   ${actual_hash}" >&2
      discard_reference_verified_signatures
      return 1
    fi
    printf '%s\t%s\t%s\n' \
      "${relative_path}" "${expected_size}" "${after_signature}" \
      >> "${REFERENCE_VERIFIED_SIGNATURES_PATH}"
    checked=$((checked + 1))
  done < "${REFERENCE_VERIFIED_MANIFEST_PATH}"

  if [[ "${checked}" -eq 0 ]]; then
    echo "setup.sh: reference manifest contained no files: ${REFERENCE_MANIFEST_PATH}" >&2
    discard_reference_verified_signatures
    return 1
  fi
  if [[ "$(reference_manifest_hash 2>/dev/null || true)" != "${verified_manifest_hash}" ]]; then
    echo "setup.sh: reference manifest changed while SHA256 verification was running" >&2
    discard_reference_verified_signatures
    return 1
  fi
  echo "setup.sh: verified ${checked} reference file hash(es)"
}

ensure_reference_compat_link() {
  local reference_dir="$1"
  local link_path="${REFERENCE_COMPAT_LINK}"
  local link_absolute
  local link_parent
  local link_name
  local link_target

  reference_dir="$(cd -P "${reference_dir}" 2>/dev/null && pwd -P)" || return 1
  if [[ -d "${link_path}" ]]; then
    link_absolute="$(cd -P "${link_path}" 2>/dev/null && pwd -P)" || return 1
  else
    link_parent="$(dirname "${link_path}")"
    link_name="$(basename "${link_path}")"
    mkdir -p "${link_parent}" || return 1
    link_parent="$(cd -P "${link_parent}" 2>/dev/null && pwd -P)" || return 1
    link_absolute="${link_parent}/${link_name}"
  fi

  if [[ "${reference_dir}" == "${link_absolute}" ]]; then
    return 0
  fi

  if [[ -L "${link_path}" ]]; then
    link_target="$(readlink "${link_path}")" || return 1
    if [[ "${link_target}" == "${reference_dir}" ]]; then
      return 0
    fi
    rm -f "${link_path}" || return 1
  elif [[ -e "${link_path}" ]]; then
    cat >&2 <<EOF
setup.sh: compatibility reference path exists and is not a symlink: ${link_path}

Move it aside, set MLXFAST_REFERENCE_DIR to that checkpoint directly, or set
MLXFAST_REFERENCE_COMPAT_LINK to another path. Leaving a stale compatibility
path in place would make later transform commands read the wrong checkpoint.

EOF
    return 1
  fi

  link_parent="$(dirname "${link_path}")"
  mkdir -p "${link_parent}" || return 1
  ln -s "${reference_dir}" "${link_path}" || return 1
  echo "setup.sh: linked ${link_path} -> ${reference_dir}"
}

download_reference_weights_locked() {
  local reference_dir="$1"
  local parent_dir
  local file
  local index_path
  local post_verify_status
  local shard_list
  local shard_files=()

  parent_dir="$(dirname "${reference_dir}")"
  ensure_reference_space "${parent_dir}" || return 1
  mkdir -p "${reference_dir}" || return 1

  echo "setup.sh: downloading ${REFERENCE_MODEL_REPO}@${REFERENCE_REVISION}"
  echo "setup.sh: primary reference source ${REFERENCE_BASE_URL}"
  if [[ -n "${REFERENCE_FALLBACK_BASE_URL}" ]]; then
    echo "setup.sh: fallback reference source ${REFERENCE_FALLBACK_BASE_URL}"
  fi
  echo "setup.sh: reference cache path ${reference_dir}"
  for file in "${REFERENCE_REQUIRED_METADATA_FILES[@]}"; do
    download_reference_file "${file}" "${reference_dir}/${file}" || return 1
  done

  if [[ ! -f "${reference_dir}/config.json" ]]; then
    echo "setup.sh: downloaded checkpoint is missing config.json" >&2
    return 1
  fi
  index_path="${reference_dir}/model.safetensors.index.json"
  if [[ ! -f "${index_path}" ]]; then
    echo "setup.sh: downloaded checkpoint is missing model.safetensors.index.json" >&2
    return 1
  fi

  if ! shard_list="$(list_reference_shards "${index_path}")"; then
    return 1
  fi
  start_mlx_metallib_build || return 1
  while IFS= read -r file; do
    if [[ -n "${file}" ]]; then
      shard_files+=("${file}")
    fi
  done <<< "${shard_list}"
  if [[ "${#shard_files[@]}" -eq 0 ]]; then
    echo "setup.sh: checkpoint index did not list any safetensors shards" >&2
    return 1
  fi

  echo "setup.sh: checkpoint index lists ${#shard_files[@]} safetensors shard(s)"
  download_reference_shards "${reference_dir}" "${shard_files[@]}" || return 1
  if reference_post_download_full_verify_enabled; then
    if ! verify_reference_weights "${reference_dir}"; then
      return 1
    fi
  else
    post_verify_status="$?"
    if [[ "${post_verify_status}" == "1" ]]; then
      if ! verify_reference_weights_after_verified_download "${reference_dir}"; then
        return 1
      fi
    else
      return 1
    fi
  fi

  find "${reference_dir}" -name "*.complete" -type f -delete || return 1
  ensure_reference_compat_link "${reference_dir}" || return 1
  echo "setup.sh: reference weights ready at ${reference_dir}"
}

download_reference_weights() {
  local reference_dir="$1"
  local parent_dir
  local operation_status=0
  local release_status=0

  if [[ -e "${reference_dir}" && ! -d "${reference_dir}" ]]; then
    cat >&2 <<EOF
setup.sh: ${reference_dir} exists but is not a directory.

Move it aside or set MLXFAST_REFERENCE_DIR to a checkpoint directory.

EOF
    return 1
  fi

  parent_dir="$(dirname "${reference_dir}")"
  mkdir -p "${parent_dir}" || return 1
  acquire_reference_cache_mutation_lock || return 1

  # Every cache read is checked under the same mutex as repair so a setup using
  # another manifest or path alias cannot mutate files during validation.
  if [[ -f "${reference_dir}/config.json" ]] && verify_reference_weights "${reference_dir}"; then
    echo "setup.sh: reference weights became ready while waiting at ${reference_dir}"
    ensure_reference_compat_link "${reference_dir}" || operation_status=$?
  else
    if [[ -f "${reference_dir}/config.json" ]]; then
      echo "setup.sh: reference cache at ${reference_dir} is incomplete or stale; repairing changed files"
    fi
    download_reference_weights_locked "${reference_dir}" || operation_status=$?
  fi

  release_reference_cache_mutation_lock || release_status=$?
  if [[ "${operation_status}" == "0" && "${release_status}" != "0" ]]; then
    operation_status="${release_status}"
  fi
  return "${operation_status}"
}

check_yukon_cli() {
  # Submissions use the Yukon CLI (`yukon`) for login/clone/submit (see
  # README.md "Submitting"). That CLI is distributed by the external Yukon
  # installer, not by this repository, so setup.sh cannot install it. This
  # check only surfaces the common onboarding gap where the CLI was installed
  # but its directory never made it onto PATH, which otherwise shows up much
  # later as "command not found: yukon" at submit time. Informational only;
  # never fails setup.
  local candidate
  local install_dir

  if command -v yukon >/dev/null 2>&1; then
    echo "setup.sh: Yukon CLI found at $(command -v yukon)"
    return 0
  fi

  for candidate in "${HOME}/.local/bin/yukon" "${HOME}/bin/yukon"; do
    if [[ -x "${candidate}" ]]; then
      install_dir="$(dirname "${candidate}")"
      cat >&2 <<EOF
setup.sh: warning: the Yukon CLI is installed at
${candidate} but ${install_dir} is not on PATH, so
'yukon clone/submit' will not work in this shell. Activate it with:

  echo 'export PATH="${install_dir}:\$PATH"' >> ~/.zshrc

then open a new shell (or 'source ~/.zshrc').

EOF
      return 0
    fi
  done

  echo "setup.sh: note: use the Yukon CLI ('yukon') for account and submission commands. It is not on PATH and is installed by the external Yukon installer, not by setup.sh. Build, transform, correctness, and local benchmarks work without it, but install it (and put its bin directory on PATH) before 'yukon submit' -- see README.md 'Submitting'."
  return 0
}

ensure_swift_toolchain
ensure_macmon
trap cleanup_background_builds EXIT

if [[ "${MLXFAST_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  build_swift_harness
  start_mlx_metallib_build
  wait_for_mlx_metallib_build
  echo "setup.sh: skipping reference weight download"
  check_yukon_cli
  print_setup_summary "skipped"
  exit 0
fi

build_swift_harness
start_mlx_metallib_build
download_reference_weights "${REFERENCE_DIR}"
wait_for_mlx_metallib_build
check_yukon_cli
print_setup_summary "ready"
