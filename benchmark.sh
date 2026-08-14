#!/usr/bin/env bash
# Run the Swift benchmark and emit the benchmark.json scorePath.
set -euo pipefail

# --- Local-mode GPU cool-down gate (--local-iterate / --local-submit) --------
# The ranked runner starts each timed phase only after the GPU has cooled
# below a fixed 40C thermal gate. Local modes expose this trusted shell helper
# to the Swift harness, which invokes it immediately before each measured
# prefill and decode phase. Local-mode only: the ranked/official path keeps its
# operator-side gate.
#
# Knobs (local debugging only; see run_local_cool_gate):
#   MLXFAST_LOCAL_COOL_GATE=0   disable the gate (timings taken hot are not
#                               comparable to gated runs)
#   MLXFAST_MACMON_BIN=...      explicit macmon binary path
#   MLXFAST_GPU_TEMP_CMD=...    testing/portability seam: shell command whose
#                               stdout is the current GPU temperature in C
#   MLXFAST_FAN_CONTROL_HELPER=...  override tools/fan-control.sh, used by the
#                               stalled-cool-down fan-boost offer and by
#                               ./benchmark.sh --fan-speed-normal
#   MLXFAST_FAN_BOOST_STATE_FILE  internal parent->child contract (not a
#                               knob): the top-level local run exports a
#                               state file; the cool-gate child processes the
#                               harness spawns before each timed phase record
#                               an applied fan boost there so the parent can
#                               print the restore reminder on completion and
#                               restore automatic fan control on INT/TERM
readonly COOL_GATE_TEMP_C=40             # start timing only at/below this GPU temp (C); same 40C as the ranked gate
readonly COOL_GATE_POLL_SECONDS=10       # temperature poll / progress-notification interval
readonly COOL_GATE_ABORT_SECONDS=180     # minimum total wait before a stalled cool-down aborts
readonly COOL_GATE_STALL_SECONDS=90      # abort once no new minimum temp has been seen for this long
readonly COOL_GATE_MAX_WAIT_SECONDS=900  # hard wait ceiling even while temp is still (slowly) falling; matches the ranked COOL_TIMEOUT
readonly COOL_GATE_PROGRESS_EPSILON_C=0.25  # a new minimum must drop at least this much to count as progress (sensor jitter is not progress)
readonly COOL_GATE_FAN_OFFER_STALL_SECONDS=60  # offer the one-time fan boost after this long without cooling progress (before the stall abort can fire)

LOCAL_ITERATE=0
LOCAL_SUBMIT=0
OFFICIAL=0
LOCAL_COOL_GATE_ONLY=0
FAN_SPEED_NORMAL=0
# Arguments forwarded to `mlxfast-swift benchmark`. --official is a shell-level
# mode selector only, so it is filtered out here; the Swift CLI does not know it.
FORWARD_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --weights|--weights=*|--golden|--golden=*|--score-path|--score-path=*)
      echo "benchmark.sh: use MLXFAST_WEIGHTS_PATH, MLXFAST_CORRECTNESS_GOLDEN_PATH, or MLXFAST_SCORE_PATH for shell path overrides" >&2
      echo "benchmark.sh: pass --weights/--golden/--score-path only to .build/release/mlxfast-swift benchmark" >&2
      exit 1
      ;;
    --official)
      OFFICIAL=1
      continue
      ;;
    --local-cool-gate-only)
      LOCAL_COOL_GATE_ONLY=1
      continue
      ;;
    --fan-speed-normal)
      FAN_SPEED_NORMAL=1
      continue
      ;;
  esac
  if [[ "${arg}" == "--local-iterate" ]]; then
    LOCAL_ITERATE=1
  fi
  if [[ "${arg}" == "--local-submit" ]]; then
    LOCAL_SUBMIT=1
  fi
  FORWARD_ARGS+=("${arg}")
done

if [[ "${OFFICIAL}" == "1" && ( "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ) ]]; then
  echo "benchmark.sh: --official cannot be combined with --local-iterate/--local-submit" >&2
  exit 1
fi

if [[ "${LOCAL_ITERATE}" == "1" && "${LOCAL_SUBMIT}" == "1" ]]; then
  echo "benchmark.sh: --local-iterate and --local-submit cannot be used together" >&2
  exit 1
fi

# --fan-speed-normal undoes the cool gate's optional 70% fan boost: it hands
# fan control back to macOS's automatic curve (no pinned RPM) and exits.
# Handled before any build/transform work; requires sudo for the SMC write,
# which tools/fan-control.sh explains before sudo prompts.
if [[ "${FAN_SPEED_NORMAL}" == "1" ]]; then
  fan_helper="${MLXFAST_FAN_CONTROL_HELPER:-$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/tools/fan-control.sh}"
  if [[ ! -x "${fan_helper}" ]]; then
    echo "benchmark.sh: fan control helper not found or not executable: ${fan_helper}" >&2
    exit 1
  fi
  exec "${fan_helper}" normal
fi

# Bare invocations default to the participant-friendly local edit loop. The
# ranked full benchmark must be requested explicitly: with --official, by the
# trusted workflow env (MLXFAST_OFFICIAL_BENCHMARK_RUN=1 -- also inherited by
# the pinned paired-baseline checkout's own benchmark.sh), or implicitly by an
# operator pointing MLXFAST_CORRECTNESS_GOLDEN_PATH at a provisioned oracle.
if [[ "${LOCAL_COOL_GATE_ONLY}" == "0" && "${LOCAL_ITERATE}" == "0" && "${LOCAL_SUBMIT}" == "0" && "${OFFICIAL}" == "0" ]]; then
  if [[ "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == "1" || -n "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
    OFFICIAL=1
  else
    echo "benchmark.sh: no mode given; defaulting to --local-iterate (use --official for the ranked entrypoint, which requires the private oracle)"
    LOCAL_ITERATE=1
    FORWARD_ARGS+=("--local-iterate")
  fi
fi

if [[ "${LOCAL_ITERATE}" == "1" && -z "${MLXFAST_SCORE_PATH:-}" ]]; then
  SCORE_PATH="score.local-iterate.json"
else
  SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
fi
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-weights}"
if [[ -z "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" && "${LOCAL_SUBMIT}" == "1" ]]; then
  GOLDEN_PATH="correctness_prompts/public_longcopy_gate_english_512_1024.json"
elif [[ -z "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" && "${LOCAL_ITERATE}" == "1" ]]; then
  GOLDEN_PATH="correctness_prompts/public_longcopy_gate_english_512_256.json"
else
  GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-correctness_golden.json}"
fi

# Fail fast with actionable guidance when the golden fixture is missing,
# BEFORE any build/transform work runs. The official mode needs the private
# oracle, which is never in the public repo -- participants who reached it by
# accident used to burn minutes on the transform and then hit a raw
# file-not-found error from the Swift harness.
if [[ "${LOCAL_COOL_GATE_ONLY}" == "0" && ! -f "${GOLDEN_PATH}" ]]; then
  if [[ "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" || -n "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
    echo "benchmark.sh: correctness golden not found at ${GOLDEN_PATH}" >&2
    echo "benchmark.sh: if you overrode MLXFAST_CORRECTNESS_GOLDEN_PATH, check the path;" >&2
    echo "benchmark.sh: otherwise re-sync the repo (the public fixtures live in correctness_prompts/)." >&2
  else
    cat >&2 <<'EOF'
benchmark.sh: correctness_golden.json is missing.

--official is the RANKED entrypoint: it requires the private benchmark
oracle, which is provisioned only on the official runner and is not part of
the public repository.

For local development use one of the local modes, which run against the
public fixtures checked into correctness_prompts/ (a bare ./benchmark.sh
defaults to --local-iterate):

  ./benchmark.sh --local-iterate   # fast edit-loop signal (~2 minutes)
  ./benchmark.sh --local-submit    # pre-submit gate (longer decode window)

(Operators with a provisioned private oracle: set
MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json.)
EOF
  fi
  exit 1
fi
# Resolve the reference checkpoint the same way setup.sh does. benchmark.sh used
# to hard-default to the reference_weights/ compatibility symlink; when that
# symlink is stale, dangling, or points at a non-directory, the Swift transform
# fails with ENOTDIR ("Not a directory"). Prefer an explicit MLXFAST_REFERENCE_DIR;
# else use reference_weights/ only when it actually holds a checkpoint, resolved to
# its real target so the transform never opens a symlinked directory; else fall
# back to the Hugging Face cache setup.sh downloads into.
REFERENCE_MODEL_REPO="${MLXFAST_REFERENCE_MODEL_REPO:-poolside/Laguna-XS-2.1-NVFP4-mlx}"
REFERENCE_REVISION="${MLXFAST_REFERENCE_REVISION:-841778bda563a36104dd521e37d99218e46f4f25}"
REFERENCE_DEFAULT_DIR="reference_weights/laguna-xs-2.1-nvfp4-mlx"
REFERENCE_HF_HOME="${MLXFAST_HF_HOME:-${HF_HOME:-${HOME:-${PWD}}/.cache/huggingface}}"
REFERENCE_HF_HUB_CACHE="${MLXFAST_HF_HUB_CACHE:-${HF_HUB_CACHE:-${REFERENCE_HF_HOME}/hub}}"
REFERENCE_CACHE_DIR="${MLXFAST_REFERENCE_CACHE_DIR:-${REFERENCE_HF_HUB_CACHE}/models--${REFERENCE_MODEL_REPO//\//--}/snapshots/${REFERENCE_REVISION//\//--}}"
if [[ -n "${MLXFAST_REFERENCE_DIR:-}" ]]; then
  REFERENCE_PATH="${MLXFAST_REFERENCE_DIR}"
elif [[ -f "${REFERENCE_DEFAULT_DIR}/config.json" ]]; then
  REFERENCE_PATH="$(cd -P "${REFERENCE_DEFAULT_DIR}" 2>/dev/null && pwd -P)" \
    || REFERENCE_PATH="${REFERENCE_DEFAULT_DIR}"
elif [[ -f "${REFERENCE_CACHE_DIR}/config.json" ]]; then
  REFERENCE_PATH="${REFERENCE_CACHE_DIR}"
else
  REFERENCE_PATH="${REFERENCE_DEFAULT_DIR}"
fi
SWIFT_BIN="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
# The participant runtime worker builds under its own SwiftPM scratch root
# (.build-worker) so a participant-code build never writes into the trusted
# CLI's build tree (.build). mlx.metallib is a participant artifact (compiled
# from the participant-editable vendored Metal sources) and lives next to the
# worker binary, where Cmlx searches first.
RUNTIME_WORKER_BIN="${MLXFAST_RUNTIME_WORKER_EXECUTABLE:-.build-worker/release/mlxfast-runtime-worker}"
MLX_METALLIB="${MLXFAST_MLX_METALLIB:-$(dirname "${RUNTIME_WORKER_BIN}")/mlx.metallib}"
SANDBOX_PROFILE="${MLXFAST_SANDBOX_PROFILE:-tools/deny-network.sb}"
SOURCE_HASH_PATH="${WEIGHTS_PATH}/.benchmark-source.sha256"
if [[ "${LOCAL_ITERATE}" == "1" && -z "${MLXFAST_INTEGRITY_PATH:-}" ]]; then
  INTEGRITY_PATH="benchmark-integrity.local-iterate.json"
else
  INTEGRITY_PATH="${MLXFAST_INTEGRITY_PATH:-benchmark-integrity.json}"
fi
USE_RUNTIME_WORKER="${MLXFAST_USE_RUNTIME_WORKER:-1}"

report_local_iterate_git_base() {
  if [[ "${LOCAL_ITERATE}" != "1" ]]; then
    return 0
  fi
  if ! command -v git >/dev/null 2>&1; then
    return 0
  fi
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 0
  fi

  local head_sha
  local origin_main_sha
  head_sha="$(git rev-parse --short=12 HEAD 2>/dev/null || true)"
  origin_main_sha="$(git rev-parse --short=12 --verify origin/main 2>/dev/null || true)"

  if [[ -n "${head_sha}" && -n "${origin_main_sha}" ]]; then
    echo "benchmark.sh: local-iterate git_head=${head_sha} origin_main=${origin_main_sha}"
    if ! git merge-base --is-ancestor origin/main HEAD >/dev/null 2>&1; then
      cat >&2 <<EOF
benchmark.sh: warning: HEAD does not contain the currently fetched origin/main.
benchmark.sh: run 'git fetch origin main', rebase or branch from the latest tip,
benchmark.sh: then rerun './benchmark.sh --local-iterate' before trusting local speedups.
EOF
    fi
  else
    cat >&2 <<EOF
benchmark.sh: warning: could not find origin/main for local-iterate baseline context.
benchmark.sh: run 'git fetch origin main' and measure the latest tip locally before comparing changes.
EOF
  fi
}

# Local modes print the same-machine baseline snapshot (when one exists) BEFORE
# the run starts, so the live per-token numbers streaming from the Swift
# harness can be compared against a target from the first second instead of
# only in the final summary. Diagnostic only: any failure here must never fail
# the benchmark run.
report_local_baseline_context() {
  if [[ "${LOCAL_ITERATE}" != "1" && "${LOCAL_SUBMIT}" != "1" ]]; then
    return 0
  fi
  local baseline_path="${SCORE_PATH%.json}.baseline.json"
  if [[ ! -f "${baseline_path}" ]]; then
    return 0
  fi
  local context
  context="$(jq -r '
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    (.metrics // {}) as $m
    | ($m.prefill_seconds_per_token // 0) as $p
    | ($m.decode_seconds_per_token // 0) as $d
    | select($p > 0 and $d > 0)
    | ($m.prefill_speedup // 0) as $ps
    | ($m.decode_speedup // 0) as $ds
    | (if $ps > 0 and $ds > 0 then pow($ds; 0.75) * pow($ps; 0.25) else 0 end) as $est
    | "prefill \($p | r6) s/token, decode \($d | r6) s/token"
      + (if $est > 0 then ", est score \($est | r3)" else "" end)
  ' "${baseline_path}" 2>/dev/null || true)"
  if [[ -n "${context}" ]]; then
    echo "benchmark.sh: local baseline to beat (${baseline_path}): ${context}" >&2
  fi
  return 0
}

# Local modes end with a compact human-readable summary on stderr so the score
# does not have to be dug out of the JSON payload. The estimated score uses the
# official formula against the official baseline constants carried inside the
# score payload; local modes publish that estimate as the payload's score so
# the Yukon participant CLI (`mlxfast run`), which requires a finite numeric
# score at the contract scorePath, can consume local runs. It is a directional
# local estimate (metrics.runtime marks the mode), never the official score,
# which only the ranked runner produces. When a same-machine baseline snapshot
# exists next to the score file (the documented
# `cp score.local-iterate.json score.local-iterate.baseline.json` workflow),
# the summary also prints deltas against it. Diagnostic only: any failure here
# must never fail the benchmark run.
report_local_score_summary() {
  if [[ "${LOCAL_ITERATE}" != "1" && "${LOCAL_SUBMIT}" != "1" ]]; then
    return 0
  fi
  local mode_name="local-submit"
  if [[ "${LOCAL_ITERATE}" == "1" ]]; then
    mode_name="local-iterate"
  fi

  local summary
  summary="$(jq -r '
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    (.metrics // {}) as $m
    | ($m.prefill_seconds_per_token // 0) as $p
    | ($m.decode_seconds_per_token // 0) as $d
    | select($p > 0 and $d > 0)
    | ($m.prefill_speedup // 0) as $ps
    | ($m.decode_speedup // 0) as $ds
    | (if $ps > 0 and $ds > 0 then pow($ds; 0.75) * pow($ps; 0.25) else 0 end) as $est
    | "  prefill \($p | r6) s/token  speedup \($ps | r3)x\n"
      + "  decode  \($d | r6) s/token  speedup \($ds | r3)x"
      + (if $est > 0
         then "\n  est score \($est | r3) (decode_speedup^0.75 * prefill_speedup^0.25; official score comes from the ranked runner)"
         else "" end)
  ' "${SCORE_PATH}" 2>/dev/null || true)"
  if [[ -z "${summary}" ]]; then
    return 0
  fi
  {
    echo "benchmark.sh: ${mode_name} summary"
    printf '%s\n' "${summary}"
  } >&2

  local baseline_path="${SCORE_PATH%.json}.baseline.json"
  if [[ ! -f "${baseline_path}" ]]; then
    if [[ "${LOCAL_ITERATE}" == "1" ]]; then
      echo "benchmark.sh: no local baseline at ${baseline_path}; run 'cp ${SCORE_PATH} ${baseline_path}' to compare future runs" >&2
    fi
    return 0
  fi
  local compare
  compare="$(jq -r -n --slurpfile cur "${SCORE_PATH}" --slurpfile base "${baseline_path}" '
    def r1: . * 10 | round / 10;
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    def sign: if . >= 0 then "+" else "" end;
    ($cur[0].metrics // {}) as $c
    | ($base[0].metrics // {}) as $b
    | ($c.prefill_seconds_per_token // 0) as $cp
    | ($c.decode_seconds_per_token // 0) as $cd
    | ($b.prefill_seconds_per_token // 0) as $bp
    | ($b.decode_seconds_per_token // 0) as $bd
    | select($cp > 0 and $cd > 0 and $bp > 0 and $bd > 0)
    | (($cp - $bp) / $bp * 100) as $pdelta
    | (($cd - $bd) / $bd * 100) as $ddelta
    | ($c.prefill_speedup // 0) as $cps
    | ($c.decode_speedup // 0) as $cds
    | ($b.prefill_speedup // 0) as $bps
    | ($b.decode_speedup // 0) as $bds
    | (if $cps > 0 and $cds > 0 then pow($cds; 0.75) * pow($cps; 0.25) else 0 end) as $cest
    | (if $bps > 0 and $bds > 0 then pow($bds; 0.75) * pow($bps; 0.25) else 0 end) as $best
    | "    prefill \($bp | r6) -> \($cp | r6) s/token (\($pdelta | sign)\($pdelta | r1)%)\n"
      + "    decode  \($bd | r6) -> \($cd | r6) s/token (\($ddelta | sign)\($ddelta | r1)%)"
      + (if $cest > 0 and $best > 0
         then (((($cest - $best) / $best) * 100) as $edelta
           | "\n    est score \($best | r3) -> \($cest | r3) (\($edelta | sign)\($edelta | r1)%)")
         else "" end)
  ' 2>/dev/null || true)"
  if [[ -z "${compare}" ]]; then
    return 0
  fi
  {
    echo "benchmark.sh: vs ${baseline_path} (negative s/token deltas = faster)"
    printf '%s\n' "${compare}"
  } >&2
}

find_macmon() {
  # Prefer an explicit override, then PATH, then the usual install locations
  # (Homebrew, and the ~/bin drop used on the ranked boxes).
  local candidate
  if [[ -n "${MLXFAST_MACMON_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_MACMON_BIN}" ]]; then
      printf '%s\n' "${MLXFAST_MACMON_BIN}"
      return 0
    fi
    echo "benchmark.sh: MLXFAST_MACMON_BIN is set but not executable: ${MLXFAST_MACMON_BIN}" >&2
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

# Prints the current GPU temperature in C (one line), or nothing on failure.
# MLXFAST_GPU_TEMP_CMD is a documented testing/portability seam: any shell
# command whose stdout is a plain Celsius number can stand in for macmon.
COOL_GATE_MACMON_BIN=""
local_gpu_temp() {
  if [[ -n "${MLXFAST_GPU_TEMP_CMD:-}" ]]; then
    bash -c "${MLXFAST_GPU_TEMP_CMD}" 2>/dev/null | head -n 1 | tr -d '[:space:]'
    return 0
  fi
  "${COOL_GATE_MACMON_BIN}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

format_temp_c() {
  awk -v t="$1" 'BEGIN { printf "%.1f", t }'
}

fan_control_helper() {
  if [[ -n "${MLXFAST_FAN_CONTROL_HELPER:-}" ]]; then
    printf '%s\n' "${MLXFAST_FAN_CONTROL_HELPER}"
    return 0
  fi
  printf '%s/tools/fan-control.sh\n' "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
}

# --- Fan-boost run lifecycle --------------------------------------------------
# The boost itself is applied inside the cool-gate child process the Swift
# harness spawns before each timed phase (`benchmark.sh --local-cool-gate-only`),
# but "did THIS run boost the fans?" is a whole-run question that the
# top-level local benchmark.sh must answer at exit. The top-level run creates
# an empty state file and exports its path as MLXFAST_FAN_BOOST_STATE_FILE;
# any gate invocation that applies a boost records it there; ONLY the process
# that created the file (the owner) reads it back, so nested cool-gate
# children never double-handle the same boost.
FAN_BOOST_STATE_FILE_OWNED=""
FAN_BOOST_ABORT_HANDLED=0

# PID of the in-flight `mlxfast-swift benchmark` child. Local modes launch it
# as a monitored background child (see the invocation site) so the INT/TERM
# abort handler and the EXIT cleanup can reap the model-holding process tree
# instead of leaving a ~21.6 GB runtime worker behind on an aborted run.
BENCHMARK_CHILD_PID=""

fan_boost_recorded() {
  [[ -n "${MLXFAST_FAN_BOOST_STATE_FILE:-}" && -s "${MLXFAST_FAN_BOOST_STATE_FILE}" ]]
}

record_fan_boost_applied() {
  if [[ -n "${MLXFAST_FAN_BOOST_STATE_FILE:-}" ]]; then
    printf 'applied\n' >> "${MLXFAST_FAN_BOOST_STATE_FILE}" 2>/dev/null || true
  fi
}

# Creates the run-scoped boost state file and installs the INT/TERM abort
# handler. Only the top-level local run creates (owns) the file: cool-gate
# children the harness spawns inherit MLXFAST_FAN_BOOST_STATE_FILE from the
# environment, skip creation here, and leave signal handling to the owner so
# a group-wide Ctrl-C triggers exactly one restore.
setup_fan_boost_run_tracking() {
  if [[ -n "${MLXFAST_FAN_BOOST_STATE_FILE:-}" ]]; then
    return 0
  fi
  # Install the abort handler before the state-file mktemp: the handler also
  # reaps the in-flight benchmark child tree (see
  # terminate_benchmark_child_tree), which must work even when fan-boost
  # tracking could not be set up.
  trap 'handle_benchmark_abort_signal INT' INT
  trap 'handle_benchmark_abort_signal TERM' TERM
  if ! FAN_BOOST_STATE_FILE_OWNED="$(mktemp "${TMPDIR:-/tmp}/mlxfast-fan-boost-state.XXXXXX")"; then
    FAN_BOOST_STATE_FILE_OWNED=""
    echo "benchmark.sh: warning: could not create the fan-boost state file; an interrupted run will not auto-restore a fan boost" >&2
    return 0
  fi
  export MLXFAST_FAN_BOOST_STATE_FILE="${FAN_BOOST_STATE_FILE_OWNED}"
}

# Printed on the owner's EXIT path (success or failure alike, as long as no
# abort handler ran): after a boost the fans intentionally STAY forced at 70%
# through the later timed phases and past process exit -- restoring mid-run
# would change the thermal conditions being measured -- so the run must end
# by telling the user the fans are still forced and how to undo it. This
# deliberately does NOT restore automatic control itself.
report_fan_boost_restore_reminder() {
  if [[ "${FAN_BOOST_ABORT_HANDLED}" == "1" ]]; then
    return 0
  fi
  if [[ -z "${FAN_BOOST_STATE_FILE_OWNED}" ]] || ! fan_boost_recorded; then
    return 0
  fi
  cat >&2 <<EOF
benchmark.sh: REMINDER: this run boosted the fans; they are still forced to 70% of max.
benchmark.sh: restore macOS automatic fan control with: ./benchmark.sh --fan-speed-normal
EOF
}

# INT/TERM handler for the run that owns the abort traps. Two duties:
# 1) Reap the in-flight benchmark process tree. Local modes run the Swift
#    benchmark as a monitored background child (see the invocation site), and
#    that child's runtime worker holds the ~21.6 GB model. An aborted run that
#    leaves the worker behind makes the NEXT run load a second copy of the
#    model and can out-of-memory the machine, so the tree is torn down here
#    on every abort.
# 2) Fan restore: an aborted run must not leave the machine pinned at 70%,
#    so ONLY IF this run applied a boost, hand fan control back to macOS
#    (the same helper path --fan-speed-normal uses) before exiting.
# Exits with the conventional 128+signal status through the normal EXIT
# cleanup trap; runs that never boosted just reap and exit.
handle_benchmark_abort_signal() {
  local signal_name="$1"
  local exit_status=130
  if [[ "${signal_name}" == "TERM" ]]; then
    exit_status=143
  fi
  FAN_BOOST_ABORT_HANDLED=1
  terminate_benchmark_child_tree
  if [[ -n "${FAN_BOOST_STATE_FILE_OWNED}" ]] && fan_boost_recorded; then
    local helper
    helper="$(fan_control_helper)"
    echo "benchmark.sh: ${signal_name} received after this run boosted the fans; returning them to macOS automatic control (the SMC write needs sudo)" >&2
    if [[ -x "${helper}" ]] && "${helper}" normal; then
      echo "benchmark.sh: fans restored to macOS automatic control before aborting" >&2
    else
      echo "benchmark.sh: WARNING: could not restore the fans; they remain forced at 70% of max" >&2
      echo "benchmark.sh: restore manually with: ./benchmark.sh --fan-speed-normal" >&2
    fi
  fi
  exit "${exit_status}"
}

# --- Local run memory guard and worker teardown --------------------------------
# The Poolside Laguna XS 2.1 NVFP4 text tower is ~21.6 GB and RAM-resident: it lives inside the
# sibling `mlxfast-runtime-worker runtime-worker` subprocess that the trusted
# `mlxfast-swift benchmark` process spawns. ONE resident copy needs roughly a
# 40 GiB machine once KV and workspace are included; TWO do not fit locally.
# Two copies happen when local runs overlap, or when a
# new run starts while an orphaned model-holding process from a previous
# aborted run is still alive. Local modes therefore:
#   1. take a per-user run lock so two local runs cannot overlap
#      (acquire_local_run_lock), with stale-lock reclaim when the recorded
#      holder pid is gone;
#   2. refuse to start while a model-holding mlxfast process is already
#      running (abort_if_model_already_resident) -- WARN-AND-ABORT only,
#      never auto-kill: benchmark.sh cannot prove a detected process is a
#      dead run's orphan rather than someone's legitimate concurrent work;
#   3. reap the spawned benchmark process tree (the Swift benchmark child
#      and its runtime worker) on INT/TERM and on EXIT
#      (terminate_benchmark_child_tree), so an interrupted edit-loop run
#      cannot orphan a 21.6 GB process in the first place.
# Local modes only: the ranked --official path is operator-supervised and is
# not touched by any of this.
#
# Knobs (local debugging/testing only):
#   MLXFAST_LOCAL_RUN_GUARD=0        disable the lock and the resident scan
#   MLXFAST_LOCAL_RUN_LOCK_DIR=...   lock parent directory (default
#                                    ~/.cache/mlxfast; TMPDIR-independent so
#                                    GUI/ssh/sandboxed shells share one lock)
#   MLXFAST_LOCAL_ORPHAN_SCAN_CMD=.. testing seam: shell command whose stdout
#                                    replaces the pgrep/ps resident-process
#                                    listing (empty output = nothing found)
LOCAL_RUN_LOCK_OWNED=""
# Matches the argv of every mlxfast process that holds or is about to hold
# the model: the participant worker in any wrapping (bare or under
# sandbox-exec; the command-name suffix also keeps detection compatible with
# workers launched by an older checkout), plus the trusted mlxfast-swift
# subcommands that own an imminent worker. The trusted binary never holds
# the model in-process, but a run in its pre-worker phase (validation,
# weights digest) is about to spawn a ~21.6 GB worker and would otherwise be
# invisible to this scan until that load has already started. The dflash-*
# subcommands are listed for the same reason: a DFlash residency is the same
# target plus the drafter.
#
# benchmark-dflash.sh reuses THIS guard -- both halves of it -- by extracting
# the pattern below plus local_run_lock_path/acquire_local_run_lock/
# release_local_run_lock/list_resident_model_processes/
# abort_if_model_already_resident rather than reimplementing them, so the two
# scripts exclude each other in both directions AND both refuse to start
# against the case no lock can catch: an ORPHANED model-holding worker from a
# run that died without releasing -- the residency is live but nothing holds
# the lock. Renaming any of those, or moving them off the top level, breaks
# that extraction; DFlashLocalRunLockTests fails when it does.
readonly RESIDENT_MODEL_PROCESS_PATTERN='runtime-worker[[:space:]]+--weights|mlxfast-swift[[:space:]]+(benchmark|correctness|correctness-trace|generate-golden|generate-gpqa-answers|attach-free-run-gate|dflash-benchmark|dflash-probe|dflash-reference)'

local_run_guard_enabled() {
  [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]] || return 1
  [[ "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ]]
}

local_run_lock_path() {
  # Home-anchored default: GUI, ssh, and sandboxed agent shells resolve
  # divergent TMPDIRs, which would shard the per-user lock into per-session
  # locks that cannot see each other. ${HOME} is stable across all of them
  # (and across clones/worktrees, which intentionally share one lock).
  local lock_root="${MLXFAST_LOCAL_RUN_LOCK_DIR:-${HOME:-${TMPDIR:-/tmp}}/.cache/mlxfast}"
  printf '%s/mlxfast-local-benchmark-%s.lock\n' "${lock_root%/}" "$(id -u)"
}

# mkdir-based mutual exclusion between local benchmark runs of the same user.
# The lock directory records the holder's pid; a lock whose holder is no
# longer alive (a killed run never runs its EXIT cleanup) is reclaimed
# instead of wedging the edit loop forever.
acquire_local_run_lock() {
  local_run_guard_enabled || return 0
  local lock_path holder_pid
  lock_path="$(local_run_lock_path)"
  if ! mkdir -p "$(dirname "${lock_path}")" 2>/dev/null; then
    echo "benchmark.sh: ERROR: cannot create the local run lock directory $(dirname "${lock_path}"); set MLXFAST_LOCAL_RUN_LOCK_DIR to a writable directory" >&2
    exit 1
  fi
  for _ in 1 2; do
    if mkdir "${lock_path}" 2>/dev/null; then
      printf '%s\n' "$$" > "${lock_path}/pid" 2>/dev/null || true
      LOCAL_RUN_LOCK_OWNED="${lock_path}"
      return 0
    fi
    holder_pid="$(cat "${lock_path}/pid" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -z "${holder_pid}" ]]; then
      # The holder may be between mkdir and writing its pid; give it a moment
      # before treating the empty lock as debris from a killed run.
      sleep 0.2
      holder_pid="$(cat "${lock_path}/pid" 2>/dev/null | tr -d '[:space:]' || true)"
    fi
    if [[ "${holder_pid}" =~ ^[0-9]+$ ]] && ps -p "${holder_pid}" >/dev/null 2>&1; then
      cat >&2 <<EOF
benchmark.sh: ERROR: another local benchmark run (pid ${holder_pid}) already holds ${lock_path}.
benchmark.sh: two overlapping local runs would hold two ~21.6 GB copies of the model, which can
benchmark.sh: out-of-memory this machine (and they would share one GPU, invalidating both timings).
benchmark.sh: wait for that run to finish and rerun. If pid ${holder_pid} is not a benchmark run,
benchmark.sh: remove the stale lock with: rm -rf "${lock_path}"
benchmark.sh: (MLXFAST_LOCAL_RUN_GUARD=0 disables this guard for harness debugging only; never
benchmark.sh: set it to resolve contention -- wait for the other run to finish instead.)
EOF
      exit 1
    fi
    echo "benchmark.sh: removing stale local run lock ${lock_path} (holder pid ${holder_pid:-unknown} is gone)" >&2
    rm -rf "${lock_path}"
  done
  echo "benchmark.sh: ERROR: could not acquire the local run lock at ${lock_path}; another local run is starting concurrently -- wait for it and rerun" >&2
  exit 1
}

release_local_run_lock() {
  if [[ -n "${LOCAL_RUN_LOCK_OWNED}" ]]; then
    rm -rf "${LOCAL_RUN_LOCK_OWNED}" || true
    LOCAL_RUN_LOCK_OWNED=""
  fi
}

# Prints one `pid ppid rss command` line per same-user process whose argv
# matches RESIDENT_MODEL_PROCESS_PATTERN (excluding this shell and its
# parent). Empty output means no resident model holder was found. NOTE:
# pgrep exits non-zero when nothing matches -- the expected case -- so its
# output is captured with `|| true` rather than piped, or `set -o pipefail`
# would abort the run exactly when the machine is clean.
list_resident_model_processes() {
  if [[ -n "${MLXFAST_LOCAL_ORPHAN_SCAN_CMD:-}" ]]; then
    bash -c "${MLXFAST_LOCAL_ORPHAN_SCAN_CMD}" 2>/dev/null || true
    return 0
  fi
  if ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi
  local matched_pids matched_pid
  matched_pids="$(pgrep -U "$(id -u)" -f -- "${RESIDENT_MODEL_PROCESS_PATTERN}" 2>/dev/null || true)"
  if [[ -z "${matched_pids}" ]]; then
    return 0
  fi
  while IFS= read -r matched_pid; do
    if [[ -z "${matched_pid}" || "${matched_pid}" == "$$" || "${matched_pid}" == "${PPID:-}" ]]; then
      continue
    fi
    ps -o pid=,ppid=,rss=,command= -p "${matched_pid}" 2>/dev/null || true
  done <<< "${matched_pids}"
  return 0
}

abort_if_model_already_resident() {
  local_run_guard_enabled || return 0
  local resident
  resident="$(list_resident_model_processes)"
  if [[ -z "${resident}" ]]; then
    return 0
  fi
  {
    echo "benchmark.sh: ERROR: a model-holding mlxfast process is already running (pid ppid rss_kb command):"
    printf '%s\n' "${resident}" | sed 's/^/benchmark.sh:   /'
    cat <<EOF
benchmark.sh: the Poolside Laguna NVFP4 model is ~21.6 GB RAM-resident per process; starting another local run
benchmark.sh: now would load a second copy and can out-of-memory this machine.
benchmark.sh: - a ppid of 1 usually means an orphan left by a previous aborted run: verify with
benchmark.sh:   'ps -p <pid> -o pid,ppid,rss,command' and stop it with 'kill <pid>'.
benchmark.sh: - a live ppid usually means another run is legitimately in flight: wait for it.
benchmark.sh: rerun once nothing matching the list above is running.
benchmark.sh: (MLXFAST_LOCAL_RUN_GUARD=0 disables this check for harness debugging only; never
benchmark.sh: set it to resolve contention -- a second resident model can OOM this machine.)
EOF
  } >&2
  exit 1
}

# Depth-first TERM/KILL of a process and its descendants, children first so a
# reparented-to-launchd worker cannot slip through between enumerating and
# signaling its parent. Runs from the INT/TERM trap under `set -euo
# pipefail`, so every probe that legitimately "fails" (pgrep finds no
# children for a leaf, the pid already exited) is captured with `|| true` --
# an errexit abort here would skip the kills it exists to deliver.
signal_process_tree() {
  local root_pid="$1"
  local signal="$2"
  local child_pids child_pid
  child_pids="$(pgrep -P "${root_pid}" 2>/dev/null || true)"
  if [[ -n "${child_pids}" ]]; then
    while IFS= read -r child_pid; do
      if [[ -n "${child_pid}" ]]; then
        signal_process_tree "${child_pid}" "${signal}"
      fi
    done <<< "${child_pids}"
  fi
  kill -s "${signal}" "${root_pid}" 2>/dev/null || true
  return 0
}

# Reaps the monitored Swift benchmark child and everything under it (the
# sandbox wrapper and the model-holding runtime worker). TERM first with a
# ~2 s grace, then KILL. No-op when no child is in flight or it already
# exited. Called from the INT/TERM abort handler and, last-resort, from the
# EXIT cleanup.
terminate_benchmark_child_tree() {
  local child_pid="${BENCHMARK_CHILD_PID}"
  [[ -n "${child_pid}" ]] || return 0
  BENCHMARK_CHILD_PID=""
  kill -0 "${child_pid}" 2>/dev/null || return 0
  echo "benchmark.sh: stopping the in-flight benchmark process tree (pid ${child_pid}) so no model-holding worker stays resident" >&2
  signal_process_tree "${child_pid}" TERM
  local waited_deciseconds=0
  while kill -0 "${child_pid}" 2>/dev/null && [[ "${waited_deciseconds}" -lt 20 ]]; do
    sleep 0.1
    waited_deciseconds=$((waited_deciseconds + 1))
  done
  if kill -0 "${child_pid}" 2>/dev/null; then
    signal_process_tree "${child_pid}" KILL
  fi
}

# One-time, opt-in fan boost for a stalled cool-down. Interactive only: the
# user must approve on the terminal because the SMC fan write needs sudo (see
# tools/fan-control.sh for the full sudo/password-handling contract: sudo's
# own secure prompt, never read/stored/logged here, credential dropped with
# `sudo -k` right after the write). The boost target is hard-capped at 70% of
# each fan's maximum speed inside the helper. Returns 0 only when the fans
# are (already or newly) boosted, so the caller can grant the cool-down a
# fresh stall window.
offer_fan_boost() {
  local current_temp="$1"
  local helper fan_status reply=""
  helper="$(fan_control_helper)"
  if [[ ! -x "${helper}" ]]; then
    return 1
  fi
  # One boost per benchmark run: the fans stay forced across the later timed
  # phases, so if a previous gate invocation this run already boosted them,
  # do not prompt for sudo again -- just grant the cool-down a fresh stall
  # window. `status` reports manual when ANY fan's mode bit is set, so a
  # foreign fan controller (or a prior un-restored boost) also lands here:
  # the helper's own `boost` would refuse to overwrite it, so skip the offer
  # with one warning instead of prompting for sudo only to be refused. The
  # foreign hold gets no fresh stall window: unlike our just-boosted fans, it
  # is not new cooling capacity worth waiting on.
  fan_status="$("${helper}" status 2>/dev/null || true)"
  if [[ "${fan_status}" == "manual" ]]; then
    if fan_boost_recorded; then
      echo "benchmark.sh: fans are already boosted by this run; giving the boost more time to cool the GPU" >&2
      return 0
    fi
    echo "benchmark.sh: another fan controller (or a prior un-restored boost) already holds the fans in manual mode; not boosting over it" >&2
    echo "benchmark.sh: if that hold is stale, restore macOS automatic fan control with: ./benchmark.sh --fan-speed-normal" >&2
    return 1
  fi
  if [[ "${fan_status}" != "auto" ]]; then
    return 1
  fi
  if ! { : < /dev/tty; } 2>/dev/null; then
    echo "benchmark.sh: GPU cool-down is stalled and no interactive terminal is attached;" >&2
    echo "benchmark.sh: to force the fans to 70% manually, run: ${helper} boost" >&2
    return 1
  fi
  cat >&2 <<EOF
benchmark.sh: the GPU is stuck at $(format_temp_c "${current_temp}")C (target <=${COOL_GATE_TEMP_C}C) and is not cooling.
benchmark.sh: you can force this Mac's fans to 70% of their maximum speed to help
benchmark.sh: (hard-capped at 70%). Fan targets live in the SMC, and macOS only
benchmark.sh: accepts SMC writes from root, so sudo will ask for your password.
benchmark.sh: that is sudo's own secure prompt: the password is never read, stored,
benchmark.sh: or logged by benchmark.sh or tools/fan-control.sh, and the cached
benchmark.sh: sudo credential is dropped (sudo -k) right after the one-time write.
benchmark.sh: restore macOS automatic fan control later with:
benchmark.sh:   ./benchmark.sh --fan-speed-normal
EOF
  if ! read -r -t 30 -p "benchmark.sh: boost fans to 70% of max now? [y/N] (auto-continues in 30s) " reply < /dev/tty; then
    reply=""
    echo >&2
  fi
  case "${reply}" in
    y|Y|yes|YES|Yes)
      if "${helper}" boost; then
        # The helper read the SMC writes back and verified they took, so this
        # run now owns a live boost: record it for the top-level run's restore
        # reminder and abort-restore handling.
        record_fan_boost_applied
        echo "benchmark.sh: fan boost active for the rest of this run (./benchmark.sh --fan-speed-normal restores automatic control)" >&2
        return 0
      fi
      echo "benchmark.sh: warning: fan boost failed or did not verify; continuing to wait without it" >&2
      return 1
      ;;
    *)
      echo "benchmark.sh: fan boost declined; continuing to wait" >&2
      return 1
      ;;
  esac
}

# Some hardware/OS combinations report frozen or near-zero GPU temperature
# telemetry (observed: macmon 0.7.2 on macOS 26 / M4 Pro reads a constant
# 3.657C for tens of minutes), which makes the local gate pass instantly and
# turns the thermal guarantee decorative. Warn -- once per gate, calmly --
# when the readings look implausible: at or below a 5C plausibility floor, or
# exactly constant across the gate's samples. Never fails the run, and only
# affects this local helper; the ranked M5 box keeps its own operator-side
# gate and contract.
warn_if_gpu_telemetry_implausible() {
  local temp="$1"
  local sample_count="$2"
  local temp_varied="$3"
  local reason=""
  if awk -v t="${temp}" 'BEGIN { exit !(t <= 5) }'; then
    reason="the GPU temperature reads $(format_temp_c "${temp}")C, at or below the 5C plausibility floor"
  elif [[ "${sample_count}" -ge 3 && "${temp_varied}" == "0" ]]; then
    reason="the GPU temperature has read a constant $(format_temp_c "${temp}")C across ${sample_count} samples"
  fi
  if [[ -n "${reason}" ]]; then
    echo "benchmark.sh: warning: ${reason}; the temperature reading looks implausible on this hardware/OS, so the thermal gate may be ineffective and gated timings may effectively be ungated" >&2
    return 0
  fi
  return 1
}

# Block the timed local run until the GPU has cooled to the gate temperature,
# mirroring the ranked runner's thermal gate. Missing-tool policy: warn loudly
# and SKIP (never hard-fail) -- a participant without macmon still gets a
# working local benchmark, just without the thermal guarantee; setup.sh
# installs/instructs about macmon so the tool being present is the normal
# case. Abort policy: if the GPU is hot and NOT trending down, something else
# is loading it and waiting longer will not help -- exit non-zero with an
# actionable message so scripted loops stop instead of measuring a loaded GPU.
run_local_cool_gate() {
  if [[ "${LOCAL_ITERATE}" != "1" && "${LOCAL_SUBMIT}" != "1" ]]; then
    return 0
  fi
  if [[ "${MLXFAST_LOCAL_COOL_GATE:-1}" == "0" ]]; then
    echo "benchmark.sh: warning: local GPU cool-down gate disabled (MLXFAST_LOCAL_COOL_GATE=0); hot-start timings are not comparable to gated runs" >&2
    return 0
  fi

  if [[ -z "${MLXFAST_GPU_TEMP_CMD:-}" ]]; then
    if ! COOL_GATE_MACMON_BIN="$(find_macmon)"; then
      cat >&2 <<EOF
benchmark.sh: warning: skipping the GPU cool-down gate: no GPU temperature reader found.
benchmark.sh: the ranked runner only starts timed runs below a ${COOL_GATE_TEMP_C}C GPU thermal gate;
benchmark.sh: without the same gate, hot back-to-back local runs can look slower than
benchmark.sh: they are. Install macmon (rerunning ./setup.sh does this for you):
benchmark.sh:   brew install macmon
benchmark.sh: or set MLXFAST_MACMON_BIN=/path/to/macmon.
EOF
      return 0
    fi
  fi

  local temp waited=0 min_temp="" last_progress_waited=0 bad_samples=0 fan_boost_offered=0
  local first_temp="" temp_varied=0 sample_count=0 telemetry_warned=0
  while :; do
    temp="$(local_gpu_temp || true)"
    if [[ ! "${temp}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      # Tolerate a couple of flaky reads, then skip the gate rather than hang
      # a participant behind a broken sensor path.
      bad_samples=$((bad_samples + 1))
      if [[ "${bad_samples}" -ge 3 ]]; then
        echo "benchmark.sh: warning: skipping the GPU cool-down gate: temperature reader returned no usable sample (reader: ${MLXFAST_GPU_TEMP_CMD:-${COOL_GATE_MACMON_BIN}})" >&2
        return 0
      fi
      sleep 2
      continue
    fi
    bad_samples=0
    sample_count=$((sample_count + 1))
    if [[ -z "${first_temp}" ]]; then
      first_temp="${temp}"
    elif [[ "${temp}" != "${first_temp}" ]]; then
      temp_varied=1
    fi
    if [[ "${telemetry_warned}" == "0" ]] \
        && warn_if_gpu_telemetry_implausible "${temp}" "${sample_count}" "${temp_varied}"; then
      telemetry_warned=1
    fi

    if awk -v t="${temp}" -v gate="${COOL_GATE_TEMP_C}" 'BEGIN { exit !(t <= gate) }'; then
      echo "benchmark.sh: GPU cool-down gate passed (current $(format_temp_c "${temp}")C, target <=${COOL_GATE_TEMP_C}C, waited ${waited}s)" >&2
      return 0
    fi

    # Progress heuristic: track the minimum temperature seen; only a new
    # minimum at least COOL_GATE_PROGRESS_EPSILON_C below the previous one
    # counts as progress, so sensor jitter around a plateau does not look
    # like cooling.
    if [[ -z "${min_temp}" ]] \
        || awk -v t="${temp}" -v m="${min_temp}" -v e="${COOL_GATE_PROGRESS_EPSILON_C}" 'BEGIN { exit !(t <= m - e) }'; then
      min_temp="${temp}"
      last_progress_waited="${waited}"
    fi

    # Before the stall abort can fire, offer the one-time fan boost: a GPU
    # that has sat hot with no cooling progress for a minute may still be
    # rescued by forcing the fans to 70%. Offered at most once per gate; a
    # boost that engages resets the stall clock so the faster fans get a
    # fresh window to show progress.
    if [[ "${fan_boost_offered}" == "0" \
        && "$((waited - last_progress_waited))" -ge "${COOL_GATE_FAN_OFFER_STALL_SECONDS}" ]]; then
      fan_boost_offered=1
      if offer_fan_boost "${temp}"; then
        last_progress_waited="${waited}"
      fi
    fi

    # Abort when BOTH the total wait exceeded the abort floor AND no progress
    # has been made recently: still hot and not decreasing means an external
    # GPU load, and more waiting will not fix that.
    if [[ "${waited}" -ge "${COOL_GATE_ABORT_SECONDS}" \
        && "$((waited - last_progress_waited))" -ge "${COOL_GATE_STALL_SECONDS}" ]]; then
      cat >&2 <<EOF
benchmark.sh: ERROR: GPU is hot and not cooling down (current $(format_temp_c "${temp}")C, min seen $(format_temp_c "${min_temp}")C, target <=${COOL_GATE_TEMP_C}C, waited ${waited}s).
benchmark.sh: something else appears to be loading the GPU. Close GPU-heavy
benchmark.sh: processes (other benchmarks, ML jobs, games, video encodes),
benchmark.sh: let the machine cool, and rerun. To debug without the gate, set
benchmark.sh: MLXFAST_LOCAL_COOL_GATE=0 (hot-start timings are not comparable).
EOF
      exit 1
    fi
    # Hard ceiling: even a slowly-cooling GPU should not stall the edit loop
    # for more than the ranked runner's own cool timeout.
    if [[ "${waited}" -ge "${COOL_GATE_MAX_WAIT_SECONDS}" ]]; then
      echo "benchmark.sh: ERROR: GPU did not reach ${COOL_GATE_TEMP_C}C within ${COOL_GATE_MAX_WAIT_SECONDS}s (current $(format_temp_c "${temp}")C); reduce GPU load or ambient heat and rerun" >&2
      exit 1
    fi

    echo "benchmark.sh: waiting for GPU to cool down before timing (current $(format_temp_c "${temp}")C, target <=${COOL_GATE_TEMP_C}C, waited ${waited}s)..." >&2
    sleep "${COOL_GATE_POLL_SECONDS}"
    waited=$((waited + COOL_GATE_POLL_SECONDS))
  done
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

canonical_directory_path() {
  local path="$1"
  (cd -P "${path}" 2>/dev/null && pwd -P)
}

path_contains() {
  local parent="$1"
  local child="$2"
  [[ "${child}" == "${parent}" || "${child}" == "${parent}/"* ]]
}

safe_clear_directory_path() {
  local path="$1"
  local label="$2"
  shift 2

  if [[ -z "${path}" ]]; then
    echo "benchmark.sh: refusing to clear empty ${label}" >&2
    return 1
  fi

  local target
  local workspace
  if [[ -L "${path}" ]]; then
    echo "benchmark.sh: refusing to clear ${label} '${path}'; symlink directories are not allowed" >&2
    return 1
  elif [[ -d "${path}" ]]; then
    target="$(canonical_directory_path "${path}")" || {
      echo "benchmark.sh: could not resolve ${label} '${path}'" >&2
      return 1
    }
  elif [[ -e "${path}" || -L "${path}" ]]; then
    echo "benchmark.sh: refusing to clear ${label} '${path}'; it is not a directory" >&2
    return 1
  else
    local parent
    parent="$(dirname "${path}")"
    if [[ ! -d "${parent}" ]]; then
      echo "benchmark.sh: parent directory for ${label} does not exist: ${parent}" >&2
      return 1
    fi
    parent="$(canonical_directory_path "${parent}")" || {
      echo "benchmark.sh: could not resolve parent directory for ${label} '${path}'" >&2
      return 1
    }
    target="${parent}/$(basename "${path}")"
  fi
  workspace="$(pwd -P)"
  if [[ "${target}" == "/" ]] || path_contains "${target}" "${workspace}"; then
    echo "benchmark.sh: refusing to clear unsafe ${label} '${path}' (resolved to ${target})" >&2
    return 1
  fi

  if path_contains "${workspace}" "${target}"; then
    local relative_target="${target#"${workspace}/"}"
    if [[ "${relative_target}" == ".git" || "${relative_target}" == .git/* ]]; then
      echo "benchmark.sh: refusing to clear Git metadata path '${path}'" >&2
      return 1
    fi
    if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      local tracked_path
      local tracked_paths_file
      local unsafe_tracked_path=""
      tracked_paths_file="$(mktemp "${TMPDIR:-/tmp}/mlxfast-tracked-paths.XXXXXX")" || return 1
      if ! git ls-files -z -- "${relative_target}" > "${tracked_paths_file}"; then
        rm -f "${tracked_paths_file}"
        echo "benchmark.sh: could not inspect tracked files under ${label} '${path}'" >&2
        return 1
      fi
      while IFS= read -r -d '' tracked_path; do
        if [[ "${label}" == "weights path" && "${tracked_path}" == "${relative_target}/.gitkeep" ]]; then
          continue
        fi
        unsafe_tracked_path="${tracked_path}"
        break
      done < "${tracked_paths_file}"
      rm -f "${tracked_paths_file}"
      if [[ -n "${unsafe_tracked_path}" ]]; then
        echo "benchmark.sh: refusing to clear ${label} '${path}'; it contains tracked file ${unsafe_tracked_path}" >&2
        return 1
      fi
    fi
  fi

  local protected
  local protected_path
  for protected in "$@"; do
    [[ -n "${protected}" && -d "${protected}" ]] || continue
    protected_path="$(canonical_directory_path "${protected}")" || continue
    if path_contains "${target}" "${protected_path}" || path_contains "${protected_path}" "${target}"; then
      echo "benchmark.sh: refusing to clear ${label} '${path}'; it overlaps protected path ${protected_path}" >&2
      return 1
    fi
  done

  if [[ ! -d "${target}" ]]; then
    mkdir "${target}" || return 1
  fi

  printf '%s\n' "${target}"
}

assert_directory_replacement_owned() {
  local target="$1"
  local label="$2"
  local workspace
  local marker
  local marker_hash
  local first_entry

  [[ -d "${target}" ]] || return 0
  workspace="$(pwd -P)"
  if path_contains "${workspace}" "${target}"; then
    return 0
  fi

  first_entry="$(find "${target}" -mindepth 1 -maxdepth 1 -print -quit)" || return 1
  if [[ -z "${first_entry}" ]]; then
    return 0
  fi

  marker="${target}/.benchmark-source.sha256"
  if [[ -f "${marker}" && ! -L "${marker}" \
      && -f "${target}/config.json" && ! -L "${target}/config.json" \
      && -f "${target}/model.safetensors.index.json" \
      && ! -L "${target}/model.safetensors.index.json" ]]; then
    marker_hash="$(tr -d '[:space:]' < "${marker}")"
    if [[ "${marker_hash}" =~ ^[0-9a-f]{64}$ ]]; then
      return 0
    fi
  fi

  echo "benchmark.sh: refusing to replace ${label} '${target}'; existing directories outside the workspace must be MLXFast-managed" >&2
  echo "benchmark.sh: choose a new/empty MLXFAST_WEIGHTS_PATH instead of reusing an unrelated directory" >&2
  return 1
}

assert_safe_output_file_path() {
  local path="$1"
  local label="$2"
  shift 2
  if [[ -z "${path}" || "${path}" == "/" || -d "${path}" || -L "${path}" ]]; then
    echo "benchmark.sh: refusing unsafe ${label} '${path}'" >&2
    return 1
  fi
  case "/${path}/" in
    *"/.git/"*)
      echo "benchmark.sh: refusing ${label} inside Git metadata: ${path}" >&2
      return 1
      ;;
  esac

  local parent
  local target
  local workspace
  parent="$(dirname "${path}")"
  if [[ ! -d "${parent}" ]]; then
    echo "benchmark.sh: parent directory for ${label} does not exist: ${parent}" >&2
    return 1
  fi
  parent="$(canonical_directory_path "${parent}")" || return 1
  target="${parent}/$(basename "${path}")"
  workspace="$(pwd -P)"

  if path_contains "${workspace}" "${target}"; then
    local relative_target="${target#"${workspace}/"}"
    if [[ "${relative_target}" == ".git" || "${relative_target}" == .git/* ]]; then
      echo "benchmark.sh: refusing ${label} inside Git metadata: ${path}" >&2
      return 1
    fi
    if command -v git >/dev/null 2>&1 \
        && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      local tracked_output
      if ! tracked_output="$(git ls-files -- "${relative_target}")"; then
        echo "benchmark.sh: could not inspect ${label} against tracked files: ${relative_target}" >&2
        return 1
      fi
      if [[ -n "${tracked_output}" ]]; then
        echo "benchmark.sh: refusing ${label} that would overwrite tracked file ${relative_target}" >&2
        return 1
      fi
    fi
  fi

  local protected
  local protected_path
  for protected in "$@"; do
    [[ -n "${protected}" ]] || continue
    if [[ -d "${protected}" ]]; then
      protected_path="$(canonical_directory_path "${protected}")" || return 1
    else
      protected_path="$(absolute_path "${protected}")"
    fi
    if [[ "${target}" == "${protected_path}" ]] \
        || { [[ -d "${protected}" ]] && path_contains "${protected_path}" "${target}"; }; then
      echo "benchmark.sh: refusing ${label} that overlaps protected path ${protected_path}" >&2
      return 1
    fi
  done
}

RUNTIME_WORKER_SANDBOX_PROFILE_OWNED=""
TRANSFORM_STAGING_PARENT_OWNED=""
VERIFY_TRANSFORM_TMP_PARENT_OWNED=""
score_stdout=""
cleanup_benchmark_temporaries() {
  local status="$?"
  # Last-resort reap of the monitored benchmark child tree (normally already
  # empty: the invocation site clears it after wait, and the INT/TERM abort
  # handler reaps it first on aborts). Runs before anything else so a
  # model-holding worker is never left behind by an unexpected exit path.
  terminate_benchmark_child_tree
  release_local_run_lock
  # A completed run that boosted the fans ends with the restore reminder
  # (no-op for runs that never boosted or that already restored on abort).
  report_fan_boost_restore_reminder
  if [[ -n "${FAN_BOOST_STATE_FILE_OWNED}" ]]; then
    rm -f "${FAN_BOOST_STATE_FILE_OWNED}" || true
  fi
  if [[ -n "${RUNTIME_WORKER_SANDBOX_PROFILE_OWNED}" ]]; then
    rm -f "${RUNTIME_WORKER_SANDBOX_PROFILE_OWNED}" || true
  fi
  if [[ -n "${score_stdout}" ]]; then
    rm -f "${score_stdout}" || true
  fi
  if [[ -n "${TRANSFORM_STAGING_PARENT_OWNED}" ]]; then
    rm -rf "${TRANSFORM_STAGING_PARENT_OWNED}" || true
  fi
  if [[ -n "${VERIFY_TRANSFORM_TMP_PARENT_OWNED}" ]]; then
    rm -rf "${VERIFY_TRANSFORM_TMP_PARENT_OWNED}" || true
  fi
  return "${status}"
}

sandbox_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

enforce_official_sandbox() {
  if [[ "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" != "1" ]]; then
    return 0
  fi
  if [[ "${MLXFAST_NO_SANDBOX:-0}" == "1" ]]; then
    echo "benchmark.sh: official GitHub benchmark runs must not set MLXFAST_NO_SANDBOX=1" >&2
    exit 1
  fi
  if [[ "${USE_RUNTIME_WORKER}" != "1" ]]; then
    echo "benchmark.sh: official GitHub benchmark runs must use the runtime worker sandbox" >&2
    exit 1
  fi
}

write_runtime_worker_sandbox_profile() {
  if [[ "${USE_RUNTIME_WORKER}" != "1" || "${MLXFAST_NO_SANDBOX:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -n "${MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE:-}" ]]; then
    return 0
  fi
  if ! command -v sandbox-exec >/dev/null 2>&1; then
    echo "benchmark.sh: sandbox-exec not found for runtime worker sandbox" >&2
    exit 1
  fi

  local profile
  local golden_absolute
  local private_dir_absolute
  local worker_absolute
  profile="$(mktemp "${TMPDIR:-/tmp}/mlxfast-runtime-worker.XXXXXX")"
  RUNTIME_WORKER_SANDBOX_PROFILE_OWNED="${profile}"
  golden_absolute="$(absolute_path "${GOLDEN_PATH}")"
  worker_absolute="$(absolute_path "${RUNTIME_WORKER_BIN}")"
  {
    cat <<EOF
(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny process-exec*)
(allow process-exec (literal "$(sandbox_escape "${worker_absolute}")"))
(deny file-write*)
(allow file-write* (literal "/dev/null"))
(deny file-read* (literal "$(sandbox_escape "${golden_absolute}")"))
EOF
    if [[ -n "${MLXFAST_PRIVATE_DIR:-}" ]]; then
      private_dir_absolute="$(absolute_path "${MLXFAST_PRIVATE_DIR}")"
      cat <<EOF
(deny file-read* (subpath "$(sandbox_escape "${private_dir_absolute}")"))
(deny file-write* (subpath "$(sandbox_escape "${private_dir_absolute}")"))
EOF
    fi
  } > "${profile}"
  export MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE="${profile}"
}

run_offline_writable_command() {
  local writable_paths="$1"
  local status
  shift
  if [[ "${MLXFAST_IN_SANDBOX:-0}" == "1" || "${MLXFAST_NO_SANDBOX:-0}" == "1" ]]; then
    "$@"
    status="$?"
    return "${status}"
  fi
  MLXFAST_OFFLINE_WRITABLE_PATHS="${writable_paths}" .github/scripts/run-offline.sh "$@"
}

source_hash() {
  # This hash gates regeneration of weights/. Keep it limited to the transform
  # target and shared core code so runtime/model-only edits stay fast locally.
  local paths=(
    "Package.swift"
    "Package.resolved"
    "Sources/MLXFastCore"
    "Sources/MLXFastTransform"
  )
  local hash_status

  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git ls-files --cached --others --exclude-standard -z "${paths[@]}" \
      | while IFS= read -r -d '' path; do
      if [[ -f "${path}" ]]; then
        printf '%s\0' "${path}"
        shasum -a 256 "${path}"
      else
        printf '%s\0MISSING\0' "${path}"
      fi
    done | shasum -a 256 | awk '{print $1}'
    hash_status="$?"
    return "${hash_status}"
  fi

  find "${paths[@]}" -type f 2>/dev/null | LC_ALL=C sort | while IFS= read -r path; do
    printf '%s\0' "${path}"
    shasum -a 256 "${path}"
  done | shasum -a 256 | awk '{print $1}'
}

publish_new_staged_metadata_file() {
  local target="$1"
  local contents="$2"
  local label="$3"

  if [[ -e "${target}" || -L "${target}" ]]; then
    echo "benchmark.sh: submitted transform created reserved ${label} path ${target}" >&2
    return 1
  fi
  # noclobber maps this create to O_EXCL, so a submitted symlink or a path
  # raced into place is rejected rather than followed.
  if ! (set -o noclobber; printf '%s' "${contents}" > "${target}"); then
    echo "benchmark.sh: failed to publish reserved ${label} metadata" >&2
    return 1
  fi
  if ! chmod 0644 "${target}"; then
    rm -f "${target}"
    return 1
  fi
}

validate_staged_safetensors_shard() {
  local shard_path="$1"
  local index_path="$2"
  local shard_name
  local file_size
  local header_length
  local data_byte_count
  local header_path
  local extracted_size

  shard_name="$(basename "${shard_path}")"
  file_size="$(wc -c < "${shard_path}" | tr -d ' ')" || return 1
  if ! [[ "${file_size}" =~ ^[0-9]+$ ]] || (( file_size < 9 )); then
    echo "benchmark.sh: staged safetensors shard is too small: ${shard_name}" >&2
    return 1
  fi
  header_length="$(od -An -tu8 -N 8 "${shard_path}" | tr -d '[:space:]')" || return 1
  if ! [[ "${header_length}" =~ ^[0-9]+$ ]] \
      || (( header_length == 0 || header_length > 100000000 )) \
      || (( header_length > file_size - 8 )); then
    echo "benchmark.sh: staged safetensors shard has an invalid header length: ${shard_name}" >&2
    return 1
  fi
  data_byte_count=$((file_size - 8 - header_length))
  header_path="$(mktemp "${TRANSFORM_STAGING_PARENT_OWNED}/.safetensors-header.XXXXXX")" \
    || return 1
  if ! (set +o pipefail; tail -c +9 "${shard_path}" | head -c "${header_length}") \
      > "${header_path}"; then
    rm -f "${header_path}"
    return 1
  fi
  extracted_size="$(wc -c < "${header_path}" | tr -d ' ')" || {
    rm -f "${header_path}"
    return 1
  }
  if [[ "${extracted_size}" != "${header_length}" ]]; then
    rm -f "${header_path}"
    echo "benchmark.sh: staged safetensors shard has a truncated header: ${shard_name}" >&2
    return 1
  fi
  # POLICY (gap-free shard tiling) — APPROVED, gap-free tiling retained:
  # beyond per-tensor bounds/dtype checks, the final clause requires the
  # sorted tensor ranges to tile the data section exactly — first range
  # starts at byte 0, each range ends where the next begins, and the last
  # range ends at the file's final data byte. This rejects overlapping
  # tensors and any unindexed byte spans, so a staged shard cannot smuggle
  # payload bytes that no tensor accounts for past the weights gate. It is
  # intentionally stricter than the safetensors format itself, which
  # tolerates padding/alignment gaps between tensors: shards from writers
  # that emit such gaps fail this gate even though standard parsers accept
  # them. The Swift transform always emits gap-free shards, so this only
  # constrains what hand-crafted staged artifacts can look like —
  # stricter-than-spec is intended.
  if ! jq -e --argjson data_bytes "${data_byte_count}" '
      def integer: type == "number" and . == floor;
      def dtype_width:
        if IN("BOOL", "U8", "I8") then 1
        elif IN("I16", "U16", "F16", "BF16") then 2
        elif IN("I32", "U32", "F32") then 4
        elif IN("I64", "U64", "F64") then 8
        else null
        end;
      type == "object"
      and ([to_entries[] | select(.key != "__metadata__")] | length > 0)
      and all(to_entries[] | select(.key != "__metadata__");
        (.value | type == "object")
        and (.value.dtype | type == "string")
        and ((.value.dtype | dtype_width) as $width | $width != null)
        and (.value.shape | type == "array" and all(.[]; integer and . >= 0))
        and (.value.data_offsets | type == "array" and length == 2)
        and (.value.data_offsets[0] | integer and . >= 0)
        and (.value.data_offsets[1] | integer)
        and (.value.data_offsets[1] > .value.data_offsets[0])
        and (.value.data_offsets[1] <= $data_bytes)
        and ((.value.dtype | dtype_width) as $width
          | (reduce .value.shape[] as $dimension (1; . * $dimension)) as $elements
          | $elements > 0
          and (.value.data_offsets[1] - .value.data_offsets[0]) == ($elements * $width))
      )
      and (([to_entries[] | select(.key != "__metadata__")
        | {start: .value.data_offsets[0], end: .value.data_offsets[1]}]
        | sort_by(.start, .end)) as $ranges
        | ($ranges[0].start == 0)
          and ($ranges[-1].end == $data_bytes)
          and all(range(1; $ranges | length); $ranges[. - 1].end == $ranges[.].start))
    ' "${header_path}" >/dev/null; then
    rm -f "${header_path}"
    echo "benchmark.sh: staged safetensors shard has an invalid header: ${shard_name}" >&2
    return 1
  fi
  if ! jq -e -n \
      --arg shard "${shard_name}" \
      --slurpfile index "${index_path}" \
      --slurpfile header "${header_path}" '
      [$index[0].weight_map | to_entries[] | select(.value == $shard) | .key] as $expected
      | [$header[0] | keys[] | select(. != "__metadata__")] as $actual
      | ($expected | length > 0)
      and (($expected | sort) == ($actual | sort))
    ' >/dev/null; then
    rm -f "${header_path}"
    echo "benchmark.sh: staged safetensors shard tensor inventory disagrees with the index: ${shard_name}" >&2
    return 1
  fi
  rm -f "${header_path}"
}

validate_staged_transform_contents() {
  local staged_weights="$1"
  local config_path="${staged_weights}/config.json"
  local index_path="${staged_weights}/model.safetensors.index.json"
  local shard_list
  local shard_name
  local actual_shard
  local actual_name

  if ! command -v jq >/dev/null 2>&1; then
    echo "benchmark.sh: jq is required to validate transformed weights" >&2
    return 1
  fi
  jq -e 'type == "object"' "${config_path}" >/dev/null || {
    echo "benchmark.sh: staged transform config.json is not a JSON object" >&2
    return 1
  }
  if ! jq -e '
      type == "object"
      and (.weight_map | type == "object" and length > 0)
      and all(.weight_map | to_entries[];
        (.key | type == "string" and length > 0)
        and (.value | type == "string" and test("^[A-Za-z0-9._-]+[.]safetensors$"))
      )
    ' "${index_path}" >/dev/null; then
    echo "benchmark.sh: staged transform index has an invalid weight_map" >&2
    return 1
  fi

  shard_list="$(mktemp "${TRANSFORM_STAGING_PARENT_OWNED}/.safetensors-shards.XXXXXX")" \
    || return 1
  if ! jq -r '.weight_map | [.[]] | unique[]' "${index_path}" > "${shard_list}"; then
    rm -f "${shard_list}"
    return 1
  fi
  # shellcheck disable=SC2094  # error cleanup intentionally unlinks the open list
  while IFS= read -r shard_name; do
    if [[ ! -f "${staged_weights}/${shard_name}" \
        || -L "${staged_weights}/${shard_name}" ]]; then
      rm -f "${shard_list}"
      echo "benchmark.sh: staged transform index references missing shard ${shard_name}" >&2
      return 1
    fi
    if ! validate_staged_safetensors_shard \
        "${staged_weights}/${shard_name}" "${index_path}"; then
      rm -f "${shard_list}"
      return 1
    fi
  done < "${shard_list}"

  while IFS= read -r -d '' actual_shard; do
    actual_name="$(basename "${actual_shard}")"
    if ! jq -e --arg shard "${actual_name}" \
        '[.weight_map[]] | index($shard) != null' "${index_path}" >/dev/null; then
      rm -f "${shard_list}"
      echo "benchmark.sh: staged transform contains unindexed shard ${actual_name}" >&2
      return 1
    fi
  done < <(find "${staged_weights}" -maxdepth 1 -type f -name '*.safetensors' -print0)
  rm -f "${shard_list}"
}

install_transformed_weights() {
  local staged_weights="$1"
  local source_hash="$2"
  local safe_weights_path
  local canonical_staged_weights
  local expected_staged_weights
  local invalid_staged_entry
  local staged_shard
  local previous_weights
  local had_previous=0

  if [[ ! -d "${staged_weights}" || -L "${staged_weights}" ]]; then
    echo "benchmark.sh: submitted transform output is not a regular directory: ${staged_weights}" >&2
    return 1
  fi
  canonical_staged_weights="$(canonical_directory_path "${staged_weights}")" || return 1
  expected_staged_weights="$(canonical_directory_path "${TRANSFORM_STAGING_PARENT_OWNED}")/weights"
  if [[ "${canonical_staged_weights}" != "${expected_staged_weights}" ]]; then
    echo "benchmark.sh: submitted transform output escaped its staging directory" >&2
    return 1
  fi
  if [[ -e "${staged_weights}/.gitkeep" || -L "${staged_weights}/.gitkeep" ]]; then
    echo "benchmark.sh: submitted transform created reserved .gitkeep path ${staged_weights}/.gitkeep" >&2
    return 1
  fi
  if [[ -e "${staged_weights}/.benchmark-source.sha256" \
      || -L "${staged_weights}/.benchmark-source.sha256" ]]; then
    echo "benchmark.sh: submitted transform created reserved .benchmark-source.sha256 path ${staged_weights}/.benchmark-source.sha256" >&2
    return 1
  fi
  invalid_staged_entry="$(find "${staged_weights}" -mindepth 1 \
    ! -type d ! -type f -print -quit)" || return 1
  if [[ -n "${invalid_staged_entry}" ]]; then
    echo "benchmark.sh: submitted transform output contains a symlink or non-regular entry: ${invalid_staged_entry}" >&2
    return 1
  fi
  if [[ ! -f "${staged_weights}/config.json" \
      || -L "${staged_weights}/config.json" \
      || ! -f "${staged_weights}/model.safetensors.index.json" \
      || -L "${staged_weights}/model.safetensors.index.json" ]]; then
    echo "benchmark.sh: submitted transform output is missing regular config/index files" >&2
    return 1
  fi
  staged_shard="$(find "${staged_weights}" -maxdepth 1 -type f \
    -name '*.safetensors' -print -quit)" || return 1
  if [[ -z "${staged_shard}" ]]; then
    echo "benchmark.sh: submitted transform output contains no safetensors shard" >&2
    return 1
  fi
  validate_staged_transform_contents "${staged_weights}" || return 1

  safe_weights_path="$(safe_clear_directory_path "${WEIGHTS_PATH}" "weights path" "${REFERENCE_PATH}")" || return 1
  assert_directory_replacement_owned "${safe_weights_path}" "weights path" || return 1
  previous_weights="${TRANSFORM_STAGING_PARENT_OWNED}/previous-weights"
  if [[ -e "${previous_weights}" || -L "${previous_weights}" ]]; then
    echo "benchmark.sh: submitted transform created reserved rollback path ${previous_weights}" >&2
    return 1
  fi
  publish_new_staged_metadata_file \
    "${staged_weights}/.gitkeep" "" ".gitkeep" || return 1
  publish_new_staged_metadata_file \
    "${staged_weights}/.benchmark-source.sha256" "${source_hash}"$'\n' \
    ".benchmark-source.sha256" || return 1

  if [[ -e "${safe_weights_path}" || -L "${safe_weights_path}" ]]; then
    mv "${safe_weights_path}" "${previous_weights}" || return 1
    had_previous=1
  fi
  if ! mv "${staged_weights}" "${safe_weights_path}"; then
    if [[ "${had_previous}" == "1" ]]; then
      mv "${previous_weights}" "${safe_weights_path}" || {
        echo "benchmark.sh: failed to restore previous weights from ${previous_weights}" >&2
        return 1
      }
    fi
    return 1
  fi
  if [[ "${had_previous}" == "1" ]]; then
    rm -rf "${previous_weights}" || return 1
  fi
}

if [[ "${LOCAL_COOL_GATE_ONLY}" == "1" ]]; then
  LOCAL_ITERATE=1
  # Harness-spawned gate children inherit MLXFAST_FAN_BOOST_STATE_FILE from
  # the top-level run and skip both steps inside; a standalone
  # --local-cool-gate-only invocation owns its boost here, so aborting it
  # restores the fans and completing it prints the restore reminder.
  setup_fan_boost_run_tracking
  trap cleanup_benchmark_temporaries EXIT
  run_local_cool_gate
  exit 0
fi

trap cleanup_benchmark_temporaries EXIT

assert_safe_output_file_path \
  "${SCORE_PATH}" "score path" \
  "${GOLDEN_PATH}" "${INTEGRITY_PATH}" "${WEIGHTS_PATH}" "${REFERENCE_PATH}" || exit 1
assert_safe_output_file_path \
  "${SCORE_PATH}.sha256" "score checksum path" \
  "${GOLDEN_PATH}" "${SCORE_PATH}" "${INTEGRITY_PATH}" "${WEIGHTS_PATH}" "${REFERENCE_PATH}" || exit 1
assert_safe_output_file_path \
  "${INTEGRITY_PATH}" "integrity path" \
  "${GOLDEN_PATH}" "${SCORE_PATH}" "${SCORE_PATH}.sha256" "${WEIGHTS_PATH}" "${REFERENCE_PATH}" || exit 1

enforce_official_sandbox

if [[ ! -s "${MLX_METALLIB}" ]]; then
  # Fail fast when setup never completed (fresh checkout) or completed only
  # partially: without mlx.metallib the runtime worker cannot run, so stop
  # before the automatic `swift build` below spends minutes producing an
  # unusable worker. MLXFAST_CLI_COMMAND only renames the CLI printed in
  # this guidance (wrapper CLIs that drive benchmark.sh under another name
  # set it); it never changes behavior. The Yukon CLI's `setup` subcommand
  # runs this repository's benchmark.json setupCommand, which is ./setup.sh,
  # so both suggested commands are equivalent.
  cli_command="${MLXFAST_CLI_COMMAND:-mlxfast}"
  echo "benchmark.sh: setup is incomplete; MLX metallib is missing at ${MLX_METALLIB}" >&2
  echo "Run setup before benchmarking:" >&2
  echo "  ${cli_command} setup" >&2
  echo "or run the repository setup directly:" >&2
  echo "  ./setup.sh" >&2
  exit 1
fi

# Existence is not freshness for mlx.metallib either. The metallib is a
# CMake/Metal artifact entirely outside SwiftPM's build graph, so the
# automatic `swift build` below can never refresh it: after an edit to an
# AOT-served kernel source (RoPE, RMSNorm, the SDPA vector kernel,
# arg_reduce) an existence-only gate benchmarks the stale library, and in
# local modes the trusted CLI's fingerprint verification deliberately
# downgrades to a single mid-run stderr warning that is easy to scroll past.
# Rebuild whenever anything under the two fingerprinted vendored subtrees
# (the exact mlx.metallib.fingerprint input set -- deliberately an
# over-approximation, matching the builder) is newer than the published
# metallib. Official runs are untouched: the trusted workflow owns that
# build, and the fingerprint check fails closed there instead of warning.
metallib_rebuild_required() {
  # An explicit MLXFAST_MLX_METALLIB override means the caller owns that
  # artifact's lifecycle (test fixtures, operator layouts): never rebuild
  # over it. The trusted CLI's fingerprint verification still warns when an
  # overridden metallib goes stale.
  if [[ -n "${MLXFAST_MLX_METALLIB:-}" ]]; then
    return 1
  fi
  local newer
  newer="$(find \
    Vendor/mlx-swift/Source/Cmlx/mlx \
    Vendor/mlx-swift/Source/Cmlx/mlx-generated \
    -type f -newer "${MLX_METALLIB}" -print -quit 2>/dev/null || true)"
  [[ -n "${newer}" ]]
}

if [[ "${OFFICIAL}" != "1" ]] && metallib_rebuild_required; then
  echo "benchmark.sh: mlx.metallib is stale (vendored kernel sources changed after it was built); rebuilding with tools/build-mlx-metallib.sh"
  if ! tools/build-mlx-metallib.sh; then
    echo "benchmark.sh: mlx.metallib rebuild failed; fix the kernel edit (or rerun ./setup.sh) and retry" >&2
    exit 1
  fi
fi

# Existence is not freshness. The scored binary is
# .build-worker/release/mlxfast-runtime-worker, and an existence-only gate
# timed whatever setup.sh built before the participant's edit: a real decode
# win read as noise, and a correctness-breaking change passed the local public
# gate because it never executed. (`swift build -c release` on its own does
# NOT fix that -- with no --scratch-path it writes .build/release, and the CLI
# deliberately prefers the .build-worker twin.) So also rebuild whenever any
# build input is newer than the product we are about to run. `-print -quit`
# early-exits on the first newer file; the nothing-changed walk over
# Sources+Vendor measures ~35ms, against a build SwiftPM makes a near no-op.
swift_build_required() {
  if [[ ! -x "${SWIFT_BIN}" ]]; then
    return 0
  fi
  if [[ "${USE_RUNTIME_WORKER}" == "1" && ! -x "${RUNTIME_WORKER_BIN}" ]]; then
    return 0
  fi
  local reference="${SWIFT_BIN}"
  if [[ "${USE_RUNTIME_WORKER}" == "1" && "${RUNTIME_WORKER_BIN}" -ot "${reference}" ]]; then
    reference="${RUNTIME_WORKER_BIN}"
  fi
  local newer
  newer="$(find Package.swift Package.resolved Sources Vendor \
    -newer "${reference}" -print -quit 2>/dev/null || true)"
  [[ -n "${newer}" ]]
}

# The dependency graph is frozen by challenge policy: before either build
# begins, Package.swift and Package.resolved must match the committed state
# (SwiftPM re-resolution, a submission, or local edits all show up as a
# work-tree diff). Skips quietly where git is unavailable -- the ranked
# workflows re-verify the manifests byte-for-byte against the trusted ref in
# verify-trusted-source-scope.sh -- and --force-resolved-versions on the
# builds below makes SwiftPM itself fail closed instead of silently
# re-resolving an out-of-date graph.
assert_frozen_dependency_graph() {
  command -v git >/dev/null 2>&1 || return 0
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0
  local manifest
  for manifest in Package.swift Package.resolved; do
    git cat-file -e "HEAD:${manifest}" 2>/dev/null || return 0
    if ! git diff --quiet HEAD -- "${manifest}" 2>/dev/null; then
      echo "benchmark.sh: ${manifest} differs from the committed state; the dependency graph is frozen by challenge policy" >&2
      echo "benchmark.sh: restore it (git checkout -- ${manifest}) and rerun" >&2
      exit 1
    fi
  done
}

if [[ "${MLXFAST_IN_SANDBOX:-0}" != "1" ]] && swift_build_required; then
  echo "benchmark.sh: trusted CLI or participant runtime worker missing or stale; building"
  assert_frozen_dependency_graph
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
fi

# Preserve the legacy no-worker wrapper path under the whole-script sandbox
# while it reaches the trusted CLI's fail-closed rejection. With the worker
# enabled, do not sandbox the parent: supported macOS runners reject nested
# sandbox-exec profiles.
# Submitted transform runs through run-offline.sh below, and submitted model
# execution runs in the participant worker sandbox.
if [[ "${USE_RUNTIME_WORKER}" != "1" && "${MLXFAST_IN_SANDBOX:-0}" != "1" && "${MLXFAST_NO_SANDBOX:-0}" != "1" ]]; then
  if ! command -v sandbox-exec >/dev/null 2>&1; then
    echo "benchmark.sh: sandbox-exec not found (the benchmark requires macOS)." >&2
    echo "Set MLXFAST_NO_SANDBOX=1 to skip the offline sandbox; scores" >&2
    echo "produced that way are not comparable to sandboxed runs." >&2
    exit 1
  fi
  if sandbox-exec -f "${SANDBOX_PROFILE}" \
      curl -fsS --max-time 10 https://example.com -o /dev/null 2>/dev/null; then
    echo "benchmark.sh: sandbox-exec did not block network access; refusing to run" >&2
    exit 1
  fi
  echo "benchmark.sh: network egress is blocked; re-running inside the sandbox"
  # Re-exec with the RESOLVED mode (not the raw "$@"): a bare invocation that
  # defaulted to --local-iterate above must not re-default (and re-print the
  # notice) in the sandboxed child. The ${arr[@]+...} idiom keeps the empty
  # array expansion safe under set -u on macOS's bash 3.2.
  RESOLVED_ARGS=(${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"})
  if [[ "${OFFICIAL}" == "1" ]]; then
    RESOLVED_ARGS+=("--official")
  fi
  exec sandbox-exec -f "${SANDBOX_PROFILE}" env \
    MLXFAST_IN_SANDBOX=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    "$0" ${RESOLVED_ARGS[@]+"${RESOLVED_ARGS[@]}"}
fi

enforce_official_sandbox
report_local_iterate_git_base

if [[ ! -x "${SWIFT_BIN}" ]]; then
  echo "benchmark.sh: trusted Swift CLI missing at ${SWIFT_BIN}" >&2
  exit 1
fi
if [[ "${USE_RUNTIME_WORKER}" == "1" && ! -x "${RUNTIME_WORKER_BIN}" ]]; then
  echo "benchmark.sh: participant runtime worker missing at ${RUNTIME_WORKER_BIN}" >&2
  exit 1
fi

write_runtime_worker_sandbox_profile
export MLXFAST_USE_RUNTIME_WORKER="${USE_RUNTIME_WORKER}"
MLXFAST_RUNTIME_WORKER_EXECUTABLE="$(absolute_path "${RUNTIME_WORKER_BIN}")"
export MLXFAST_RUNTIME_WORKER_EXECUTABLE
export MLXFAST_REFERENCE_DIR="${REFERENCE_PATH}"

# Ranked commit provenance for the sealed score's metrics.commit. The ranked
# workflow's trusted "Capture candidate commit" step writes the dispatched
# commit (validated ^[0-9a-f]{40}$) to candidate.sha before any bench-uid
# execution, and that file is copied into the bench workspace with the rest of
# the tree. Official runs execute as the sandboxed bench uid where
# `git rev-parse` fails (runner-owned .git, dubious ownership) and where the
# measure-job timed path's `sudo env_reset` + `env -i` strip workflow env, so
# recover the trusted commit from the file here and hand it to the harness.
# An explicit MLXFAST_COMMIT_SHA (trusted argv env, as the gates step passes)
# wins; local modes have neither and keep the harness's git fallback. A
# malformed candidate.sha exports nothing, which fails closed downstream at
# the trusted commit-binding checks.
if [[ "${OFFICIAL}" == "1" && -z "${MLXFAST_COMMIT_SHA:-}" && -f candidate.sha ]]; then
  candidate_commit_sha="$(head -c 64 candidate.sha | tr -d '[:space:]')"
  if [[ "${candidate_commit_sha}" =~ ^[0-9a-f]{40}$ ]]; then
    export MLXFAST_COMMIT_SHA="${candidate_commit_sha}"
  fi
fi
if [[ "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ]]; then
  if [[ -z "${MLXFAST_LOCAL_COOL_GATE_HELPER:-}" ]]; then
    MLXFAST_LOCAL_COOL_GATE_HELPER="$(absolute_path "${BASH_SOURCE[0]}")"
  fi
  export MLXFAST_LOCAL_COOL_GATE_HELPER
  # Own the fan-boost state for this run: the cool-gate children the harness
  # spawns record a boost into the exported state file, and this top-level
  # process prints the restore reminder at exit (fans intentionally stay
  # forced through the timed phases) or restores automatic control on
  # INT/TERM so an aborted run is not left pinned at 70%.
  setup_fan_boost_run_tracking
  # Memory guard for the ~21.6 GB RAM-resident model (local modes only, before
  # any transform/model work): serialize local runs behind a per-user lock,
  # then refuse to start while a model-holding process from a previous or
  # parallel run is still alive. See the guard section above for the policy.
  acquire_local_run_lock
  abort_if_model_already_resident
fi

safe_clear_directory_path "${WEIGHTS_PATH}" "weights path" "${REFERENCE_PATH}" >/dev/null || exit 1
wanted_hash="$(source_hash)"
current_hash="$(cat "${SOURCE_HASH_PATH}" 2>/dev/null || true)"

if [[ "${MLXFAST_SKIP_TRANSFORM:-0}" == "1" ]]; then
  if [[ ! -f "${WEIGHTS_PATH}/config.json" ]]; then
    echo "benchmark.sh: MLXFAST_SKIP_TRANSFORM=1 but ${WEIGHTS_PATH}/config.json is missing" >&2
    exit 1
  fi
  echo "benchmark.sh: reusing ${WEIGHTS_PATH}/ because MLXFAST_SKIP_TRANSFORM=1"
elif [[ "${MLXFAST_FORCE_TRANSFORM:-0}" == "1" || ! -f "${WEIGHTS_PATH}/config.json" || "${current_hash}" != "${wanted_hash}" ]]; then
  if [[ -f "${REFERENCE_PATH}/config.json" ]]; then
    echo "benchmark.sh: regenerating weights with Swift transform"
    safe_weights_path="$(safe_clear_directory_path "${WEIGHTS_PATH}" "weights path" "${REFERENCE_PATH}")" || exit 1
    assert_directory_replacement_owned "${safe_weights_path}" "weights path" || exit 1
    TRANSFORM_STAGING_PARENT_OWNED="$(mktemp -d \
      "$(dirname "${safe_weights_path}")/.$(basename "${safe_weights_path}").mlxfast-transform.XXXXXX")"
    staged_weights="${TRANSFORM_STAGING_PARENT_OWNED}/weights"
    run_offline_writable_command "${TRANSFORM_STAGING_PARENT_OWNED}" \
      "${SWIFT_BIN}" transform --reference "${REFERENCE_PATH}" --output "${staged_weights}"
    install_transformed_weights "${staged_weights}" "${wanted_hash}"
    rm -rf "${TRANSFORM_STAGING_PARENT_OWNED}"
    TRANSFORM_STAGING_PARENT_OWNED=""
  else
    cat >&2 <<EOF
benchmark.sh: reference weights not found at ${REFERENCE_PATH}, needed to regenerate weights/.
Run ./setup.sh, or set MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 only after placing the reference checkpoint there.
(If you expected cached weights/, the transform source hash did not match.)
EOF
    exit 1
  fi
else
  echo "benchmark.sh: reusing ${WEIGHTS_PATH}/ for unchanged transform source"
fi

if [[ "${MLXFAST_VERIFY_TRANSFORM:-0}" == "1" ]]; then
  if [[ ! -f "${REFERENCE_PATH}/config.json" ]]; then
    echo "benchmark.sh: MLXFAST_VERIFY_TRANSFORM=1 requires reference weights at ${REFERENCE_PATH}" >&2
    exit 1
  fi
  VERIFY_TRANSFORM_TMP_PARENT="${MLXFAST_VERIFY_TRANSFORM_TMP_PARENT:-.mlxfast-transform-verify}"
  VERIFY_TRANSFORM_TMP_ROOT="$(safe_clear_directory_path \
    "${VERIFY_TRANSFORM_TMP_PARENT}" \
    "transform verification temporary path" \
    "${REFERENCE_PATH}" \
    "${WEIGHTS_PATH}")" || exit 1
  VERIFY_TRANSFORM_TMP_PARENT_OWNED="$(mktemp -d \
    "${VERIFY_TRANSFORM_TMP_ROOT%/}/mlxfast-transform-verify.XXXXXX")"
  echo "benchmark.sh: verifying weights match a fresh run of the submitted Swift transform"
  if run_offline_writable_command "$(absolute_path "${VERIFY_TRANSFORM_TMP_PARENT_OWNED}")" \
    "${SWIFT_BIN}" verify-transform \
    --reference "${REFERENCE_PATH}" \
    --weights "${WEIGHTS_PATH}" \
    --tmp-parent "${VERIFY_TRANSFORM_TMP_PARENT_OWNED}"; then
    rm -rf "${VERIFY_TRANSFORM_TMP_PARENT_OWNED}"
    VERIFY_TRANSFORM_TMP_PARENT_OWNED=""
  else
    status="$?"
    rm -rf "${VERIFY_TRANSFORM_TMP_PARENT_OWNED}"
    VERIFY_TRANSFORM_TMP_PARENT_OWNED=""
    exit "${status}"
  fi
fi

rm -f "${SCORE_PATH}"

# The Swift benchmark process links the editable model code paths, so any
# score.json it leaves on disk is untrusted: submitted code running in that
# unsandboxed process could overwrite the file (e.g. at exit) after the harness
# wrote it. Capture the trusted score payload from the process stdout and, only
# AFTER it has fully exited, re-materialize score.json from that payload here in
# the trusted shell -- discarding any in-process tamper of the on-disk file.
if ! command -v jq >/dev/null 2>&1; then
  echo "benchmark.sh: jq is required to seal score.json from the benchmark process stdout" >&2
  exit 1
fi
score_stdout="$(mktemp "${TMPDIR:-/tmp}/mlxfast-score.XXXXXX")"

report_local_baseline_context

if [[ "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ]]; then
  # Local modes run the Swift benchmark as a monitored background child and
  # wait on it, for two reasons a foreground child cannot provide:
  # 1. bash defers trap handlers until a foreground command completes, so a
  #    plain `kill` of an in-flight local run used to be silently ignored
  #    until the whole run finished on its own;
  # 2. on INT/TERM the abort handler must reap the model-holding process tree
  #    (the Swift benchmark and its ~21.6 GB runtime worker) so an aborted
  #    edit-loop run cannot orphan a resident model and out-of-memory the
  #    next run. The ranked --official invocation below keeps its original
  #    foreground semantics, unchanged.
  "${SWIFT_BIN}" benchmark \
    --weights "${WEIGHTS_PATH}" \
    --golden "${GOLDEN_PATH}" \
    --score-path "${SCORE_PATH}" \
    ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"} > "${score_stdout}" &
  BENCHMARK_CHILD_PID="$!"
  benchmark_status=0
  wait "${BENCHMARK_CHILD_PID}" || benchmark_status="$?"
  BENCHMARK_CHILD_PID=""
  if [[ "${benchmark_status}" != "0" ]]; then
    exit "${benchmark_status}"
  fi
else
  "${SWIFT_BIN}" benchmark \
    --weights "${WEIGHTS_PATH}" \
    --golden "${GOLDEN_PATH}" \
    --score-path "${SCORE_PATH}" \
    ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"} > "${score_stdout}"
fi

# Require exactly one JSON object shaped like a score payload; empty, non-JSON,
# or multiple concatenated objects (an injected extra write) fail closed rather
# than sealing an attacker-controlled or malformed score.
if [[ "$(jq -s 'length' "${score_stdout}" 2>/dev/null)" != "1" ]] \
    || ! jq -e '(.passed | type == "boolean") and has("score") and (.metrics | type == "object")' \
        "${score_stdout}" >/dev/null 2>&1; then
  echo "benchmark.sh: benchmark did not emit a single valid score payload on stdout" >&2
  exit 1
fi
rm -f "${SCORE_PATH}"
cp "${score_stdout}" "${SCORE_PATH}"

if [[ ! -s "${SCORE_PATH}" ]]; then
  echo "benchmark.sh: benchmark did not produce ${SCORE_PATH}" >&2
  exit 1
fi

# stdout was redirected to capture the trusted payload above, so local modes
# must explicitly replay it to the console to keep their existing behavior of
# showing the score there.
if [[ "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ]]; then
  cat "${SCORE_PATH}"
  report_local_score_summary
fi

score_hash="$(shasum -a 256 "${SCORE_PATH}" | awk '{print $1}')"
printf '%s  %s\n' "${score_hash}" "${SCORE_PATH}" > "${SCORE_PATH}.sha256"

if ! score_metrics="$(jq -er '
    .metrics
    | select((.weights_hash | type) == "string" and (.weights_hash | length) > 0)
    | select((.weights_file_count | type) == "number" and .weights_file_count >= 0 and (.weights_file_count | floor) == .weights_file_count)
    | select((.weights_byte_count | type) == "number" and .weights_byte_count >= 0 and (.weights_byte_count | floor) == .weights_byte_count)
    | [.weights_hash, (.weights_file_count | tostring), (.weights_byte_count | tostring)]
    | @tsv
  ' "${SCORE_PATH}")"; then
  echo "benchmark.sh: score payload has invalid weights integrity metrics" >&2
  exit 1
fi
IFS=$'\t' read -r weights_hash weights_file_count weights_byte_count <<< "${score_metrics}"
golden_hash=""
if [[ -f "${GOLDEN_PATH}" ]]; then
  golden_hash="$(shasum -a 256 "${GOLDEN_PATH}" | awk '{print $1}')"
fi

jq -n \
  --arg score_path "${SCORE_PATH}" \
  --arg score_sha256 "${score_hash}" \
  --arg weights_path "${WEIGHTS_PATH}" \
  --arg weights_sha256 "${weights_hash}" \
  --argjson weights_file_count "${weights_file_count}" \
  --argjson weights_byte_count "${weights_byte_count}" \
  --arg golden_sha256 "${golden_hash}" \
  --arg transform_source_sha256 "${wanted_hash}" \
  '{
    score_path: $score_path,
    score_sha256: $score_sha256,
    weights_path: $weights_path,
    weights_sha256: $weights_sha256,
    weights_file_count: $weights_file_count,
    weights_byte_count: $weights_byte_count,
    golden_path: "[private]",
    golden_sha256: $golden_sha256,
    transform_source_sha256: $transform_source_sha256
  }' > "${INTEGRITY_PATH}"

if ! jq -e '.passed == true' "${SCORE_PATH}" >/dev/null; then
  echo "benchmark.sh: benchmark produced a failing score; see ${SCORE_PATH}" >&2
  # Local public-gate token mismatches deserve the non-M5 caveat: the goldens
  # are greedy continuations captured on the ranked M5 box, and near-tie
  # argmaxes diverge deterministically on other Apple Silicon generations, so
  # a first run of unmodified main can fail here on a perfectly good machine.
  # Messaging only; the run still exits non-zero.
  if [[ "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ]] \
      && jq -e '(.metrics.error // "") | test("token mismatch")' "${SCORE_PATH}" >/dev/null 2>&1; then
    echo "benchmark.sh: note: the public goldens are M5-generated, so on non-M5 Apple Silicon a deterministic near-tie token mismatch is expected for a correct build (see \"Correctness fixtures are M5-generated\" in README.md); the ranked M5 runner is the source of truth" >&2
    echo "benchmark.sh: note: if unmodified main diverges at the same token position on this machine, rerun with MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1 to keep the timing estimate (tokens stay unverified and the score records that)" >&2
  fi
  exit 1
fi
