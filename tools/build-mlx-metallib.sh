#!/usr/bin/env bash
# Build mlx.metallib for the SwiftPM-linked MLX runtime and place it next to
# the participant worker, where Cmlx searches first. The metallib is compiled
# from participant-editable vendored Metal sources, so every default below
# lives under the participant worker's build root (.build-worker), never
# under the trusted CLI's .build tree.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

repository_path() {
  local candidate="$1"
  if [[ "${candidate}" == /* ]]; then
    printf '%s\n' "${candidate}"
  else
    printf '%s/%s\n' "${ROOT_DIR}" "${candidate#./}"
  fi
}

# --all-build-roots: also publish next to the trusted CLI and into EVERY xctest
# bundle under both build roots, in either configuration.
#
# WHY THIS FLAG EXISTS. Cmlx searches for `mlx.metallib` next to the RUNNING
# executable. The default publish target is the participant worker's build root,
# which is correct for a benchmark run and useless for `swift test`: the test
# bundle is a third location, `swift test` builds debug while this script
# defaults to release, and the pre-existing test-bundle publish below is gated on
# the configuration being debug. So the documented sequence
# "tools/build-mlx-metallib.sh && MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test"
# left every MLX-gated test failing at its first MLXArray with "Failed to load
# the default metallib" -- and a run WITHOUT the env var skips those tests and
# exits 0, so the gap reads as green. `QwenMTPMetallibAvailabilityTests` turns
# the missing file into one legible failure; this flag is the fix it names.
# Parsed by the argument handler further down, alongside --print-fingerprint,
# so that flag keeps working: it must be recognised BEFORE the fingerprint
# early-exit's own "unknown argument" branch, not by a second loop above it.
PUBLISH_ALL_BUILD_ROOTS=0

BUILD_CONFIGURATION="${MLXFAST_SWIFT_CONFIGURATION:-release}"
RUNTIME_WORKER_BIN="$(repository_path \
  "${MLXFAST_RUNTIME_WORKER_EXECUTABLE:-.build-worker/${BUILD_CONFIGURATION}/mlxfast-runtime-worker}")"
OUTPUT_PATH="$(repository_path \
  "${MLXFAST_MLX_METALLIB:-$(dirname "${RUNTIME_WORKER_BIN}")/mlx.metallib}")"

MLX_SWIFT_VENDOR="$(repository_path "${MLXFAST_MLX_SWIFT_VENDOR:-Vendor/mlx-swift}")"
MLX_SOURCE="${MLX_SWIFT_VENDOR}/Source/Cmlx/mlx"
METAL_CPP_SOURCE="${MLX_SWIFT_VENDOR}/Source/Cmlx/metal-cpp"
JSON_SOURCE="${MLX_SWIFT_VENDOR}/Source/Cmlx/json"
FMT_SOURCE="${MLX_SWIFT_VENDOR}/Source/Cmlx/fmt"

# Content fingerprint over every vendored source that can influence the AOT
# metallib (the whole vendored mlx tree) plus the runtime-effective JIT
# kernel strings (mlx-generated). Deliberately an over-approximation: a
# changed file that does not feed the metallib merely forces a rebuild,
# while any under-approximation would let a cached mlx.metallib mask a
# participant kernel edit. The recipe (per-file sha256 lines over
# byte-sorted relative paths, hashed again) is mirrored bit-for-bit by
# VendoredMetalFingerprint in Sources/MLXFastTrustedHarness, which the
# trusted CLI uses to re-verify the published fingerprint sidecar before
# spawning the participant worker. Keep the two implementations in sync.
FINGERPRINT_RECORD_PREFIX="mlxfast-metallib-fingerprint-v1"
compute_vendored_metal_fingerprint() {
  local cmlx_root="${MLX_SWIFT_VENDOR}/Source/Cmlx"
  local subtree
  local file_count
  for subtree in mlx mlx-generated; do
    if [[ ! -d "${cmlx_root}/${subtree}" ]]; then
      echo "build-mlx-metallib.sh: vendored source tree missing for fingerprint: ${cmlx_root}/${subtree}" >&2
      return 1
    fi
  done
  # An empty tree can only mean a broken checkout; never fingerprint it.
  file_count="$(cd "${cmlx_root}" && find mlx mlx-generated -type f | wc -l | tr -d ' ')"
  if [[ "${file_count}" -eq 0 ]]; then
    echo "build-mlx-metallib.sh: vendored source trees contain no files to fingerprint under ${cmlx_root}" >&2
    return 1
  fi
  (
    cd "${cmlx_root}"
    find mlx mlx-generated -type f -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 shasum -a 256 \
      | shasum -a 256 \
      | awk '{print $1}'
  )
}

# --print-fingerprint prints the vendored-source fingerprint and exits
# without building anything (no lock, no toolchain requirements). The ranked
# workflows use it for the metallib cache key and cache verification.
if [[ "${1:-}" == "--print-fingerprint" ]]; then
  compute_vendored_metal_fingerprint
  exit 0
fi
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --all-build-roots)
      PUBLISH_ALL_BUILD_ROOTS=1
      shift
      ;;
    *)
      echo "build-mlx-metallib.sh: unknown argument '$1' (supported: --print-fingerprint, --all-build-roots)" >&2
      exit 2
      ;;
  esac
done
CMAKE_BUILD_DIR="$(repository_path "${MLXFAST_MLX_METAL_BUILD_DIR:-.build-worker/mlx-metal}")"
BUILD_LOCK_DIR="$(repository_path \
  "${MLXFAST_MLX_METAL_BUILD_LOCK_DIR:-${CMAKE_BUILD_DIR}.mlxfast-build.lock}")"
BUILD_LOCK_TIMEOUT_SECONDS="${MLXFAST_MLX_METAL_BUILD_LOCK_TIMEOUT_SECONDS:-1800}"
BUILD_LOCK_STALE_SECONDS="${MLXFAST_MLX_METAL_BUILD_LOCK_STALE_SECONDS:-60}"
BUILD_LOCK_HELD=0
BUILD_LOCK_TOKEN=""
BUILD_LOCK_OWNER_TEMP=""
BUILD_LOCK_GENERATION_DIR=""
BUILD_LOCK_INITIALIZING_DIR=""
PUBLISH_TEMP_PATH=""
export CLANG_MODULE_CACHE_PATH
CLANG_MODULE_CACHE_PATH="$(repository_path \
  "${CLANG_MODULE_CACHE_PATH:-.build-worker/clang-module-cache}")"
mkdir -p "${CLANG_MODULE_CACHE_PATH}"
METAL_COMPILER_HOME="$(repository_path \
  "${MLXFAST_METAL_COMPILER_HOME:-${CMAKE_BUILD_DIR}/.mlxfast-home}")"
mkdir -p "${METAL_COMPILER_HOME}"

find_cmake() {
  local candidate
  if [[ -n "${MLXFAST_CMAKE_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_CMAKE_BIN}" ]]; then
      printf '%s\n' "${MLXFAST_CMAKE_BIN}"
      return 0
    fi
    echo "build-mlx-metallib.sh: MLXFAST_CMAKE_BIN is set but not executable: ${MLXFAST_CMAKE_BIN}" >&2
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

lock_directory_identity() {
  local directory="$1"
  stat_value '%d:%i' '%d:%i' "${directory}"
}

lock_mtime_seconds() {
  local path="$1"
  stat_value '%m' '%Y' "${path}"
}

build_lock_generation_path() {
  local raw_target
  local candidate
  local canonical_lock_parent
  local canonical_candidate
  local expected_prefix
  local lock_uid

  [[ -L "${BUILD_LOCK_DIR}" ]] || return 1
  raw_target="$(readlink "${BUILD_LOCK_DIR}")" || return 1
  [[ "${raw_target}" == /* ]] || return 1
  candidate="${raw_target}"
  canonical_lock_parent="$(cd -P "$(dirname "${BUILD_LOCK_DIR}")" && pwd -P)" || return 1
  [[ -d "$(dirname "${candidate}")" ]] || return 1
  canonical_candidate="$(cd -P "$(dirname "${candidate}")" && pwd -P)/$(basename "${candidate}")"
  expected_prefix="$(basename "${BUILD_LOCK_DIR}").generation."
  [[ "$(dirname "${canonical_candidate}")" == "${canonical_lock_parent}" ]] || return 1
  [[ "$(basename "${canonical_candidate}")" == "${expected_prefix}"* ]] || return 1
  lock_uid="$(stat_value '%u' '%u' "${BUILD_LOCK_DIR}")" || return 1
  [[ "${lock_uid}" == "$(id -u)" ]] || return 1
  if [[ -e "${canonical_candidate}" || -L "${canonical_candidate}" ]]; then
    [[ -d "${canonical_candidate}" && ! -L "${canonical_candidate}" \
        && -O "${canonical_candidate}" ]] || return 1
  fi
  printf '%s\n' "${canonical_candidate}"
}

remove_build_lock_generation() {
  local generation_dir="$1"
  local published_target=""

  if [[ -L "${BUILD_LOCK_DIR}" ]]; then
    published_target="$(readlink "${BUILD_LOCK_DIR}" 2>/dev/null || true)"
  fi
  if [[ -n "${generation_dir}" && "${published_target}" == "${generation_dir}" ]]; then
    rm -f "${BUILD_LOCK_DIR}" || return 1
  fi
  if [[ -n "${generation_dir}" ]]; then
    rm -rf "${generation_dir}" || return 1
  fi
}

reclaim_orphaned_build_lock_generations() {
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

  lock_parent="$(cd -P "$(dirname "${BUILD_LOCK_DIR}")" && pwd -P)" || return 1
  lock_name="$(basename "${BUILD_LOCK_DIR}")"
  local_host_hash="$(printf '%s' "$(hostname)" | shasum -a 256 | awk '{print substr($1, 1, 16)}')" \
    || return 1
  initializing_prefix="${lock_name}.initializing.${local_host_hash}."
  if [[ -L "${BUILD_LOCK_DIR}" ]]; then
    published_target="$(readlink "${BUILD_LOCK_DIR}" 2>/dev/null || true)"
  fi

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
    if [[ -L "${BUILD_LOCK_DIR}" ]]; then
      published_target="$(readlink "${BUILD_LOCK_DIR}" 2>/dev/null || true)"
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

recover_stale_build_lock() {
  local lock_dir="${BUILD_LOCK_DIR}"
  local owner_path="${lock_dir}/owner"
  local owner_line=""
  local owner_pid=""
  local owner_host=""
  local owner_token=""
  local owner_existed=0
  local owner_is_well_formed=0
  local stale=0
  local metadata_path="${lock_dir}"
  local observed_identity
  local claimed_identity
  local claim_dir
  local quarantine_dir
  local metadata_mtime
  local now
  local field
  local generation_dir=""
  local generation_missing=0

  observed_identity="$(lock_directory_identity "${lock_dir}" 2>/dev/null)" || return 1
  if [[ -L "${lock_dir}" ]]; then
    generation_dir="$(build_lock_generation_path)" || return 1
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
    metadata_path="${owner_path}"
    owner_line="$(cat "${owner_path}" 2>/dev/null || true)"
    for field in ${owner_line}; do
      case "${field}" in
        token=*) owner_token="${field#token=}" ;;
        pid=*) owner_pid="${field#pid=}" ;;
        host=*) owner_host="${field#host=}" ;;
      esac
    done
    if [[ -n "${owner_token}" && -n "${owner_host}" \
        && "${owner_pid}" =~ ^[1-9][0-9]*$ ]]; then
      owner_is_well_formed=1
      if [[ "${owner_host}" == "$(hostname)" ]] \
          && ! kill -0 "${owner_pid}" >/dev/null 2>&1; then
        stale=1
      fi
    fi
  fi
  if [[ "${generation_missing}" == "1" ]]; then
    stale=1
  elif [[ "${stale}" != "1" && "${owner_is_well_formed}" != "1" ]]; then
    if metadata_mtime="$(lock_mtime_seconds "${metadata_path}" 2>/dev/null)"; then
      now="$(date +%s)"
      if (( now - metadata_mtime >= BUILD_LOCK_STALE_SECONDS )); then
        stale=1
      fi
    fi
  fi
  [[ "${stale}" == "1" ]] || return 1

  mkdir "${claim_dir}" 2>/dev/null || return 1
  claimed_identity="$(lock_directory_identity "${lock_dir}" 2>/dev/null || true)"
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
}

acquire_build_lock() {
  local started_seconds="${SECONDS}"
  local announced_wait=0
  local owner_temp
  local token
  local generation_dir
  local generation_dir_raw
  local initializing_dir
  local initializing_dir_raw
  local local_host_hash
  local nested_publication

  [[ "${BUILD_LOCK_TIMEOUT_SECONDS}" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "build-mlx-metallib.sh: build lock timeout must be a non-negative integer" >&2
    return 1
  }
  [[ "${BUILD_LOCK_STALE_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "build-mlx-metallib.sh: build lock stale threshold must be a positive integer" >&2
    return 1
  }
  mkdir -p "$(dirname "${BUILD_LOCK_DIR}")" || return 1
  reclaim_orphaned_build_lock_generations || return 1
  token="$$-$(date +%s)-${RANDOM}-${RANDOM}"
  local_host_hash="$(printf '%s' "$(hostname)" | shasum -a 256 | awk '{print substr($1, 1, 16)}')" \
    || return 1

  while true; do
    owner_temp="$(mktemp "${BUILD_LOCK_DIR}.owner.XXXXXX")" || return 1
    BUILD_LOCK_OWNER_TEMP="${owner_temp}"
    if ! printf 'token=%s pid=%s host=%s started_at=%s\n' \
        "${token}" "$$" "$(hostname)" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "${owner_temp}"; then
      rm -f "${owner_temp}"
      BUILD_LOCK_OWNER_TEMP=""
      return 1
    fi
    initializing_dir_raw="$(mktemp -d \
      "${BUILD_LOCK_DIR}.initializing.${local_host_hash}.$$.XXXXXX")" || {
      rm -f "${owner_temp}"
      BUILD_LOCK_OWNER_TEMP=""
      return 1
    }
    initializing_dir="$(cd -P "${initializing_dir_raw}" && pwd -P)" || {
      rm -f "${owner_temp}"
      rm -rf "${initializing_dir_raw}"
      BUILD_LOCK_OWNER_TEMP=""
      return 1
    }
    BUILD_LOCK_INITIALIZING_DIR="${initializing_dir}"
    if ! mv "${owner_temp}" "${initializing_dir}/owner"; then
      rm -f "${owner_temp}"
      rm -rf "${initializing_dir}"
      BUILD_LOCK_OWNER_TEMP=""
      BUILD_LOCK_INITIALIZING_DIR=""
      return 1
    fi
    BUILD_LOCK_OWNER_TEMP=""
    generation_dir_raw="${BUILD_LOCK_DIR}.generation.${token}-${RANDOM}"
    if [[ -e "${generation_dir_raw}" || -L "${generation_dir_raw}" ]] \
        || ! mv "${initializing_dir}" "${generation_dir_raw}"; then
      rm -rf "${initializing_dir}"
      BUILD_LOCK_INITIALIZING_DIR=""
      return 1
    fi
    BUILD_LOCK_INITIALIZING_DIR=""
    BUILD_LOCK_GENERATION_DIR="${generation_dir_raw}"
    generation_dir="$(cd -P "${generation_dir_raw}" && pwd -P)" || return 1
    BUILD_LOCK_GENERATION_DIR="${generation_dir}"

    if [[ ! -e "${BUILD_LOCK_DIR}" && ! -L "${BUILD_LOCK_DIR}" ]] \
        && publish_lock_symlink "${generation_dir}" "${BUILD_LOCK_DIR}"; then
      if [[ -L "${BUILD_LOCK_DIR}" \
          && "$(readlink "${BUILD_LOCK_DIR}" 2>/dev/null || true)" == "${generation_dir}" ]]; then
        BUILD_LOCK_TOKEN="${token}"
        BUILD_LOCK_HELD=1
        return 0
      fi
      nested_publication="${BUILD_LOCK_DIR}/$(basename "${generation_dir}")"
      if [[ -L "${nested_publication}" \
          && "$(readlink "${nested_publication}" 2>/dev/null || true)" == "${generation_dir}" ]]; then
        rm -f "${nested_publication}" || return 1
      fi
    fi

    rm -rf "${generation_dir}" || return 1
    BUILD_LOCK_GENERATION_DIR=""
    # Fatal only for a genuine obstruction (an existing non-directory,
    # non-symlink entry such as a plain file). An absent path here is the
    # normal transient state after a concurrent holder released the lock
    # between our failed publish attempt and this check; retry instead.
    if [[ -e "${BUILD_LOCK_DIR}" && ! -d "${BUILD_LOCK_DIR}" && ! -L "${BUILD_LOCK_DIR}" ]]; then
      echo "build-mlx-metallib.sh: build lock path is not a lock directory: ${BUILD_LOCK_DIR}" >&2
      return 1
    fi
    if recover_stale_build_lock; then
      continue
    fi
    if [[ "${announced_wait}" == "0" ]]; then
      echo "build-mlx-metallib.sh: waiting for build lock ${BUILD_LOCK_DIR}"
      announced_wait=1
    fi
    if (( SECONDS - started_seconds >= BUILD_LOCK_TIMEOUT_SECONDS )); then
      echo "build-mlx-metallib.sh: timed out waiting for build lock ${BUILD_LOCK_DIR}" >&2
      return 1
    fi
    sleep 1
  done
}

release_build_lock() {
  local owner_line
  local owner_token=""
  local published_target=""
  local field
  [[ "${BUILD_LOCK_HELD}" == "1" ]] || return 0
  owner_line="$(cat "${BUILD_LOCK_DIR}/owner" 2>/dev/null || true)"
  for field in ${owner_line}; do
    case "${field}" in
      token=*) owner_token="${field#token=}" ;;
    esac
  done
  if [[ -z "${BUILD_LOCK_TOKEN}" || "${owner_token}" != "${BUILD_LOCK_TOKEN}" ]]; then
    echo "build-mlx-metallib.sh: refusing to release a build lock owned by another process" >&2
    return 1
  fi
  if [[ -L "${BUILD_LOCK_DIR}" ]]; then
    published_target="$(readlink "${BUILD_LOCK_DIR}" 2>/dev/null || true)"
    if [[ -z "${BUILD_LOCK_GENERATION_DIR}" \
        || "${published_target}" != "${BUILD_LOCK_GENERATION_DIR}" ]]; then
      echo "build-mlx-metallib.sh: refusing to release a replaced build lock" >&2
      return 1
    fi
    rm -f "${BUILD_LOCK_DIR}" || return 1
    BUILD_LOCK_HELD=0
    BUILD_LOCK_TOKEN=""
    rm -rf "${BUILD_LOCK_GENERATION_DIR}" || return 1
    BUILD_LOCK_GENERATION_DIR=""
  else
    rm -f "${BUILD_LOCK_DIR}/owner" || return 1
    rmdir "${BUILD_LOCK_DIR}" || return 1
    BUILD_LOCK_HELD=0
    BUILD_LOCK_TOKEN=""
  fi
}

cleanup_builder() {
  local status="$?"
  trap - EXIT
  if [[ -n "${PUBLISH_TEMP_PATH}" ]]; then
    rm -f "${PUBLISH_TEMP_PATH}" || true
  fi
  if [[ -n "${BUILD_LOCK_OWNER_TEMP}" ]]; then
    rm -f "${BUILD_LOCK_OWNER_TEMP}" || true
  fi
  if [[ -n "${BUILD_LOCK_INITIALIZING_DIR}" ]]; then
    rm -rf "${BUILD_LOCK_INITIALIZING_DIR}" || true
    BUILD_LOCK_INITIALIZING_DIR=""
  fi
  if [[ "${BUILD_LOCK_HELD}" != "1" && -n "${BUILD_LOCK_GENERATION_DIR}" ]]; then
    remove_build_lock_generation "${BUILD_LOCK_GENERATION_DIR}" || true
    BUILD_LOCK_GENERATION_DIR=""
  fi
  if ! release_build_lock && [[ "${status}" == "0" ]]; then
    status=1
  fi
  exit "${status}"
}

trap cleanup_builder EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

metal_toolchain_identifier() {
  xcodebuild -showComponent MetalToolchain 2>/dev/null \
    | awk -F': ' '/Toolchain Identifier/ { print $2; exit }' \
    || true
}

CMAKE_BIN="$(find_cmake)" || {
  cat >&2 <<EOF
build-mlx-metallib.sh: cmake was not found.

Install CMake, then retry:

  brew install cmake

EOF
  exit 1
}

# POLICY (TOOLCHAINS override) — APPROVED: pin the separately installed Metal
# Toolchain for every xcrun/metal/cmake invocation below so the produced
# mlx.metallib is built by a deterministic compiler. This intentionally
# replaces any TOOLCHAINS value inherited from the caller for the remainder
# of this build, and MLXFAST_METAL_TOOLCHAIN_IDENTIFIER allows the
# environment to select the identifier that gets pinned — supported for
# participants on their own machines. On the ranked M5 runner the identifier
# is not environment-chosen in any meaningful sense: the trusted build step
# in .github/workflows/benchmark.yml injects the operator-pinned value
# through the bench-exec bridge (env -i + controlled assignments), so
# submitted code cannot select a different compiler there. See the matching
# note in setup.sh's configure_metal_toolchain_environment.
METAL_TOOLCHAIN_IDENTIFIER="${MLXFAST_METAL_TOOLCHAIN_IDENTIFIER:-$(metal_toolchain_identifier)}"

if [[ -n "${METAL_TOOLCHAIN_IDENTIFIER}" ]]; then
  export TOOLCHAINS="${METAL_TOOLCHAIN_IDENTIFIER}"
fi

if ! xcrun -sdk macosx metal -v >/dev/null 2>&1; then
  if [[ -n "${METAL_TOOLCHAIN_IDENTIFIER}" ]]; then
    echo "build-mlx-metallib.sh: found Metal Toolchain ${METAL_TOOLCHAIN_IDENTIFIER}, but xcrun could not execute metal" >&2
  fi
  cat >&2 <<EOF
build-mlx-metallib.sh: Xcode's Metal Toolchain is not installed.

Install it, then retry:

  xcodebuild -downloadComponent MetalToolchain

EOF
  exit 1
fi

acquire_build_lock

for required_path in "${MLX_SOURCE}" "${METAL_CPP_SOURCE}" "${JSON_SOURCE}" "${FMT_SOURCE}"; do
  if [[ ! -d "${required_path}" ]]; then
    echo "build-mlx-metallib.sh: required vendored MLX Swift source path is missing: ${required_path}" >&2
    exit 1
  fi
done

MLX_SOURCE="$(cd "${MLX_SOURCE}" && pwd -P)"
METAL_CPP_SOURCE="$(cd "${METAL_CPP_SOURCE}" && pwd -P)"
JSON_SOURCE="$(cd "${JSON_SOURCE}" && pwd -P)"
FMT_SOURCE="$(cd "${FMT_SOURCE}" && pwd -P)"

# Capture the fingerprint BEFORE compiling: the published sidecar must
# describe the sources this build actually consumed, so a source edited
# mid-build yields a sidecar that later verification correctly rejects.
VENDORED_METAL_FINGERPRINT="$(compute_vendored_metal_fingerprint)"

# A CMake cache generated for a different build or source directory (for
# example a build tree restored into a relocated workspace, like the ranked
# job workspace moving from ranked-current to a track-versioned path) makes
# cmake configure hard-fail with "CMakeCache.txt directory ... is different".
# Detect the mismatch while holding the build lock and regenerate the build
# tree instead of failing the build; a matching cache is kept untouched so
# incremental rebuilds stay cheap.
#
# A build tree OWNED BY ANOTHER USER is equally unusable, but fails later
# instead of up front: cmake's configure_file always finishes by chmod()ing
# its output file, and chmod by a non-owner fails with EPERM ("Operation
# not permitted") even where writing the file's data is allowed. That is
# exactly the ranked-runner shape -- the worker build-products cache
# restores as the runner uid, the bench uid builds through an ACL grant
# that deliberately withholds writesecurity, and the first kernel-touching
# submission (metallib cache miss) then reconfigures the restored tree and
# dies inside configure_file (run 30048514684). Regenerate any
# foreign-owned tree: a fresh tree is owned by this uid and configures
# cleanly. Both regeneration paths recreate the Metal compiler HOME, which
# defaults to a subdirectory of the removed build tree.
CMAKE_CACHE_FILE="${CMAKE_BUILD_DIR}/CMakeCache.txt"
if [[ ( -d "${CMAKE_BUILD_DIR}" && ! -O "${CMAKE_BUILD_DIR}" ) \
    || ( -f "${CMAKE_CACHE_FILE}" && ! -O "${CMAKE_CACHE_FILE}" ) ]]; then
  echo "build-mlx-metallib.sh: CMake build tree in ${CMAKE_BUILD_DIR} is owned by another user (e.g. restored build products); regenerating" >&2
  rm -rf "${CMAKE_BUILD_DIR}"
  mkdir -p "${METAL_COMPILER_HOME}"
elif [[ -f "${CMAKE_CACHE_FILE}" ]]; then
  recorded_cache_dir="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "${CMAKE_CACHE_FILE}" | head -n 1)"
  recorded_source_dir="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${CMAKE_CACHE_FILE}" | head -n 1)"
  actual_cache_dir="$(cd "${CMAKE_BUILD_DIR}" && pwd -P)"
  if [[ -z "${recorded_cache_dir}" \
      || "${recorded_cache_dir}" != "${actual_cache_dir}" \
      || "${recorded_source_dir}" != "${MLX_SOURCE}" ]]; then
    echo "build-mlx-metallib.sh: stale CMake cache in ${CMAKE_BUILD_DIR} (recorded build dir '${recorded_cache_dir:-unknown}', source '${recorded_source_dir:-unknown}'); regenerating" >&2
    rm -rf "${CMAKE_BUILD_DIR}"
    mkdir -p "${METAL_COMPILER_HOME}"
  fi
fi

JOBS="${MLXFAST_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

HOME="${METAL_COMPILER_HOME}" "${CMAKE_BIN}" \
  -S "${MLX_SOURCE}" \
  -B "${CMAKE_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_GGUF=OFF \
  -DFETCHCONTENT_SOURCE_DIR_METAL_CPP="${METAL_CPP_SOURCE}" \
  -DFETCHCONTENT_SOURCE_DIR_JSON="${JSON_SOURCE}" \
  -DFETCHCONTENT_SOURCE_DIR_FMT="${FMT_SOURCE}"

HOME="${METAL_COMPILER_HOME}" "${CMAKE_BIN}" \
  --build "${CMAKE_BUILD_DIR}" --target mlx-metallib --parallel "${JOBS}"

METALLIB_PATHS=()
while IFS= read -r path; do
  [[ -n "${path}" ]] && METALLIB_PATHS+=("${path}")
done < <(find "${CMAKE_BUILD_DIR}" -type f -name mlx.metallib -print | LC_ALL=C sort)

if [[ "${#METALLIB_PATHS[@]}" -eq 0 ]]; then
  echo "build-mlx-metallib.sh: CMake finished but no mlx.metallib was found under ${CMAKE_BUILD_DIR}" >&2
  exit 1
fi
if [[ "${#METALLIB_PATHS[@]}" -ne 1 ]]; then
  echo "build-mlx-metallib.sh: CMake produced multiple mlx.metallib files under ${CMAKE_BUILD_DIR}:" >&2
  printf '  %s\n' "${METALLIB_PATHS[@]}" >&2
  exit 1
fi
METALLIB_PATH="${METALLIB_PATHS[0]}"

publish_metallib() {
  local source="$1"
  local destination="$2"
  local destination_parent
  if [[ -d "${destination}" ]]; then
    echo "build-mlx-metallib.sh: output path is a directory: ${destination}" >&2
    return 1
  fi
  destination_parent="$(dirname "${destination}")"
  mkdir -p "${destination_parent}"
  PUBLISH_TEMP_PATH="$(mktemp "${destination}.tmp.XXXXXX")"
  cp "${source}" "${PUBLISH_TEMP_PATH}"
  chmod 0644 "${PUBLISH_TEMP_PATH}"
  if [[ -d "${destination}" ]]; then
    echo "build-mlx-metallib.sh: output path became a directory: ${destination}" >&2
    return 1
  fi
  mv -f "${PUBLISH_TEMP_PATH}" "${destination}"
  PUBLISH_TEMP_PATH=""
}

# The fingerprint sidecar rides next to the metallib so consumers (the
# ranked cache verification and the trusted CLI's pre-spawn check) can tell
# whether this metallib still corresponds to the vendored sources on disk.
publish_fingerprint_record() {
  local destination="$1"
  local record_temp
  record_temp="$(mktemp "${destination}.tmp.XXXXXX")"
  PUBLISH_TEMP_PATH="${record_temp}"
  printf '%s %s\n' "${FINGERPRINT_RECORD_PREFIX}" "${VENDORED_METAL_FINGERPRINT}" \
    > "${record_temp}"
  chmod 0644 "${record_temp}"
  mv -f "${record_temp}" "${destination}"
  PUBLISH_TEMP_PATH=""
}

publish_metallib "${METALLIB_PATH}" "${OUTPUT_PATH}"
publish_fingerprint_record "${OUTPUT_PATH}.fingerprint"
echo "build-mlx-metallib.sh: wrote ${OUTPUT_PATH}"

if [[ "${BUILD_CONFIGURATION}" == "debug" || "${PUBLISH_ALL_BUILD_ROOTS}" == "1" ]]; then
  search_roots=(.build)
  if [[ "${PUBLISH_ALL_BUILD_ROOTS}" == "1" ]]; then
    search_roots=(.build .build-worker)
  fi
  for search_root in "${search_roots[@]}"; do
    [[ -d "${search_root}" ]] || continue
    while IFS= read -r test_binary_dir; do
      publish_metallib "${METALLIB_PATH}" "${test_binary_dir}/mlx.metallib"
      publish_fingerprint_record "${test_binary_dir}/mlx.metallib.fingerprint"
      echo "build-mlx-metallib.sh: wrote ${test_binary_dir}/mlx.metallib"
    done < <(find "${search_root}" -path "*.xctest/Contents/MacOS" -type d)
  done
fi

# The trusted CLI links no MLX today, so it needs no metallib of its own; a
# swift-testing bundle run directly from a build root does. Publish beside every
# built product directory under both roots so no runner location is missing one.
if [[ "${PUBLISH_ALL_BUILD_ROOTS}" == "1" ]]; then
  for product_dir in .build/debug .build/release .build-worker/debug .build-worker/release; do
    [[ -d "${product_dir}" ]] || continue
    # `cond && continue` would abort the whole script under `set -e` whenever the
    # condition is FALSE, because the compound then returns 1 as a standalone
    # statement. Use an if.
    if [[ "$(cd "${product_dir}" && pwd -P)" == "$(dirname "${OUTPUT_PATH}")" ]]; then
      continue
    fi
    publish_metallib "${METALLIB_PATH}" "${product_dir}/mlx.metallib"
    publish_fingerprint_record "${product_dir}/mlx.metallib.fingerprint"
    echo "build-mlx-metallib.sh: wrote ${product_dir}/mlx.metallib"
  done
fi
