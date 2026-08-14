#!/usr/bin/env bash
# Capture or verify content pins over the built artifacts that drive the
# scored phases, as two separately attested artifact sets:
#
#   trusted  the benchmark driver (benchmark.sh) and the trusted CLI binary
#            (.build/release/mlxfast-swift) -- the timer/gates/score surface.
#   worker   the participant runtime worker binary and its MLX metal library
#            (.build-worker/release/{mlxfast-runtime-worker,mlx.metallib}) --
#            the participant-code surface the trusted CLI spawns.
#
# Why two pins: the trusted CLI and the participant worker build from
# independent SwiftPM roots, and their integrity claims differ. The trusted
# pin anchors the timer/gates/score binary against any later in-workspace
# tamper; it is captured immediately after the trusted CLI build and BEFORE
# the participant worker build begins, so participant-build-time code
# execution (manifest evaluation, build plugins) can never alter the trusted
# binary between build and pin. The worker attestation is captured right
# after the worker build and guarantees every scored phase executes exactly
# the worker binary and metallib built from the reviewed source tree -- a
# submitted transform that rewrites the (bench-owned) worker products after
# the build would otherwise run un-reviewed bytes in the scored phases. The
# participant worker deliberately stays OUT of the trusted set: it is
# participant code and must never launder itself into the trusted-artifact
# claim.
#
# Why at all: the submitted transform runs AFTER the builds with write access
# to the bench workspace, and the correctness/gates and timed phases re-run
# ./benchmark.sh and both binaries FROM that same workspace with no rebuild
# in between. A replaced driver or binary could therefore forge a
# self-consistent passing score or exfiltrate hidden material (a TOCTOU on
# the workspace). benchmark.sh is additionally ACL-locked against bench
# writes, but the built binaries and metallib are bench-owned build outputs
# that ACLs cannot protect, so their integrity anchor is these content pins:
# capture each right after its build, then re-verify both immediately before
# every phase that executes them as bench. Any mismatch fails the run closed.
#
# The pin files live under the 0700 runner-only private dir, so the bench uid
# can neither read nor rewrite them.
set -euo pipefail

MODE="${1:?usage: pin-trusted-harness.sh <write|verify> WORKSPACE PIN_FILE [trusted|worker]}"
WORKSPACE="${2:?workspace path is required}"
PIN_FILE="${3:?pin file path is required}"
ARTIFACT_SET="${4:-trusted}"

case "${ARTIFACT_SET}" in
  trusted)
    PINNED_RELATIVE_PATHS=(
      "benchmark.sh"
      ".build/release/mlxfast-swift"
    )
    ;;
  worker)
    PINNED_RELATIVE_PATHS=(
      ".build-worker/release/mlxfast-runtime-worker"
      ".build-worker/release/mlx.metallib"
    )
    ;;
  *)
    echo "::error::unknown artifact set ${ARTIFACT_SET}; expected trusted or worker" >&2
    exit 2
    ;;
esac

compute_pin() {
  local rel path
  for rel in "${PINNED_RELATIVE_PATHS[@]}"; do
    path="${WORKSPACE}/${rel}"
    if [[ -L "${path}" || ! -f "${path}" ]]; then
      echo "::error::${ARTIFACT_SET} harness artifact missing or not a regular file: ${rel}" >&2
      return 1
    fi
    printf '%s  %s\n' "$(shasum -a 256 "${path}" | awk '{print $1}')" "${rel}"
  done
}

case "${MODE}" in
  write)
    tmp="$(mktemp "${PIN_FILE}.XXXXXX")"
    if ! compute_pin > "${tmp}"; then
      rm -f "${tmp}"
      exit 1
    fi
    mv "${tmp}" "${PIN_FILE}"
    chmod 600 "${PIN_FILE}"
    echo "benchmark: pinned ${ARTIFACT_SET} harness artifacts -> ${PIN_FILE}"
    ;;
  verify)
    if [[ ! -s "${PIN_FILE}" ]]; then
      echo "::error::${ARTIFACT_SET} harness pin file missing: ${PIN_FILE}" >&2
      exit 1
    fi
    if ! current="$(compute_pin)"; then
      exit 1
    fi
    if [[ "${current}" != "$(cat "${PIN_FILE}")" ]]; then
      echo "::error::${ARTIFACT_SET} harness artifacts changed since the post-build pin (possible in-workspace tamper of the driver, a binary, or the metallib)" >&2
      exit 1
    fi
    echo "benchmark: ${ARTIFACT_SET} harness artifacts verified against the post-build pin"
    ;;
  *)
    echo "::error::unknown mode ${MODE}; expected write or verify" >&2
    exit 2
    ;;
esac
