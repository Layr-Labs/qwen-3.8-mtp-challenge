#!/usr/bin/env bash
# Local directional runner for the EXPERIMENTAL Laguna XS 2.1 DFlash block-decode
# track (laguna-xs-2.1-dflash-v1).
#
# Every gate and every measurement below is an EXISTING harness entrypoint; this
# script only wires them together in the right order. In order:
#
#   1. public drift tripwire   `mlxfast-swift correctness` against the checked-in
#                              public fixture -- the same subcommand and the same
#                              fixture the serial local loop uses.
#   2. GPU cool gate           `./benchmark.sh --local-cool-gate-only`, the serial
#                              local loop's own 40C gate, before EACH
#                              model-resident leg.
#   3. reference golden        `mlxfast-swift dflash-reference --generate`, the
#                              only producer of the DFlashReferenceGolden shape
#                              (seed_tokens/rows/reference_self_consistent) that
#                              the two measured legs decode.
#   4. serial K=1 control      `mlxfast-swift dflash-probe`      (denominator)
#   5. block decode            `mlxfast-swift dflash-benchmark`  (numerator)
#
# WHAT THIS SCORE IS NOT. It is DIRECTIONAL and can never be a ranked score:
#   * the reference golden is produced by the CANDIDATE's own build, not by the
#     pinned baseline, so it checks DFlash block/serial parity and speed
#     direction -- not target fidelity;
#   * the ranked run (fixtures/laguna_xs_2_1_dflash_track.json is LIVE:
#     official_scoring_enabled true, 8-entry timed_prompt_pool) samples a
#     hidden pool prompt and scores per-prompt normalised -- none of which
#     this local loop can reproduce;
#   * the ranked pipeline (.github/workflows/dflash-benchmark.yml) measures
#     against organizer-pinned hidden goldens on the M5 box and is the only
#     authority for both fidelity and score.
#
# ONE MODEL-HOLDING RUN AT A TIME. A DFlash residency is the ~21.6 GB NVFP4
# target PLUS the ~0.9 GB drafter, so two overlapping runs do not fit the
# documented ~36 GiB local minimum. This script reuses BOTH halves of
# benchmark.sh's memory guard (see the "local run guard" section below):
#   * the per-user run lock, which excludes a concurrent local run in either
#     direction (serial vs DFlash);
#   * the resident-model scan, which catches the case no lock can -- an
#     ORPHANED model-holding worker from a run that died without releasing,
#     where the residency is live but nothing holds the lock.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./benchmark-dflash.sh [--local-iterate|--local-submit]

Runs the DFlash block-decode path against a candidate-local serial K=1 control
and writes a directional Yukon score payload to MLXFAST_SCORE_PATH
(default: score.json).

Prerequisites (both are reused, never rebuilt here):
  ./setup.sh                     builds mlxfast-swift + the MLX metallib and
                                 provisions the NVFP4 reference checkpoint
  ./setup-dflash.sh              provisions the pinned DFlash drafter
  ./benchmark.sh --local-iterate produces and caches the transformed weights/
                                 tree this script consumes

Environment:
  MLXFAST_SCORE_PATH                     score payload path (default score.json)
  MLXFAST_WEIGHTS_PATH                   transformed weights tree (default weights)
  MLXFAST_SWIFT_BIN                      trusted CLI (default .build/release/mlxfast-swift)
  MLXFAST_DFLASH_BLOCK_SIZE              declared block width, 2..16 (default 2,
                                         the ranked workflow's value)
  MLXFAST_DFLASH_LOCAL_ITERATE_TOKENS    decode tokens for --local-iterate (default 64)
  MLXFAST_DFLASH_LOCAL_SUBMIT_TOKENS     decode tokens for --local-submit (default 128)
  MLXFAST_DFLASH_LOCAL_WORK_DIR          scratch root (default .mlxfast-local-dflash,
                                         removed on exit)
  MLXFAST_DFLASH_DRAFTER_DIR             drafter directory (default: setup-dflash.sh's cache)
  MLXFAST_GPU_TEMP_CMD                   benchmark.sh's documented cool-gate seam
EOF
}

TRACK_ID="laguna-xs-2.1-dflash-v1"

mode="${1:---local-iterate}"
case "${mode}" in
  --local-iterate)
    token_count="${MLXFAST_DFLASH_LOCAL_ITERATE_TOKENS:-64}"
    # Which public fixture the drift tripwire runs against mirrors
    # benchmark.sh's own choice (see its GOLDEN_PATH block): the 256-step
    # golden for the fast edit loop, the 1024-step golden for the pre-submit
    # check. Same fixtures, same calibration, nothing new decided here.
    public_golden_path="${MLXFAST_DFLASH_LOCAL_GOLDEN_FIXTURE:-correctness_prompts/public_longcopy_gate_english_512_256.json}"
    ;;
  --local-submit)
    token_count="${MLXFAST_DFLASH_LOCAL_SUBMIT_TOKENS:-128}"
    public_golden_path="${MLXFAST_DFLASH_LOCAL_GOLDEN_FIXTURE:-correctness_prompts/public_longcopy_gate_english_512_1024.json}"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if ! [[ "${token_count}" =~ ^[1-9][0-9]*$ ]] || (( token_count > 512 )); then
  echo "benchmark-dflash.sh: token count must be an integer in 1...512" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
# The contract fixture is READ here (for the not-rankable statement below), never
# passed to the CLI: the CLI declares no contract option, because the workflow
# interlock and the trusted parent are what enforce the contract.
contract_path="${MLXFAST_DFLASH_CONTRACT_PATH:-fixtures/laguna_xs_2_1_dflash_track.json}"
# ${public_golden_path} (set per mode above) is both the drift-tripwire fixture
# and the source of the 512-token seed: its cases[0].prompt_tokens IS the
# tokenization of correctness_prompts/public_longcopy_gate_english_512.txt, which
# is how the seed is obtained without a tokenizer -- the trusted CLI's DFlash
# subcommands take token ids, never text.
weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"
score_path="${MLXFAST_SCORE_PATH:-score.json}"
work_root="${MLXFAST_DFLASH_LOCAL_WORK_DIR:-.mlxfast-local-dflash}"
# Same env name (and same default) the ranked workflow uses for the declared
# block width, so a local run and a ranked run describe the same schedule
# (ranked dispatches MLXFAST_DFLASH_BLOCK_SIZE: "2" -- Amendment 28's K
# re-derivation; the old local default of 3 predated it).
block_size="${MLXFAST_DFLASH_BLOCK_SIZE:-2}"
workflow_path=".github/workflows/dflash-benchmark.yml"
# Reuse the serial local loop's thermal gate rather than reimplementing it. This
# branch of benchmark.sh returns before the metallib check, the builds, the
# sandbox and the transform, and takes no run lock, so it cannot deadlock here.
cool_gate_command=("./benchmark.sh" "--local-cool-gate-only")

# 2 is the floor the block scheduler enforces for a non-control run (a width-1
# round advances the target without feeding the drafter); 16 is the drafter
# checkpoint's own block size and the compiled-in ceiling.
if ! [[ "${block_size}" =~ ^([2-9]|1[0-6])$ ]]; then
  echo "benchmark-dflash.sh: MLXFAST_DFLASH_BLOCK_SIZE must be an integer in 2...16" >&2
  exit 2
fi

for tool in jq awk; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "benchmark-dflash.sh: required tool not found: ${tool}" >&2
    exit 1
  fi
done
if [[ ! -x "${swift_bin}" ]]; then
  echo "benchmark-dflash.sh: missing ${swift_bin}; run ./setup.sh first" >&2
  exit 1
fi
if [[ ! -s "${contract_path}" ]]; then
  echo "benchmark-dflash.sh: missing DFlash contract fixture ${contract_path}" >&2
  exit 1
fi
if [[ ! -s "${public_golden_path}" ]]; then
  echo "benchmark-dflash.sh: missing public correctness fixture ${public_golden_path}" >&2
  echo "benchmark-dflash.sh: re-sync the repository (public fixtures live in correctness_prompts/)" >&2
  exit 1
fi
if [[ ! -x "${cool_gate_command[0]}" ]]; then
  echo "benchmark-dflash.sh: missing ${cool_gate_command[0]}; the GPU cool gate is not optional" >&2
  exit 1
fi

# --- drafter -----------------------------------------------------------------
# setup-dflash.sh owns the drafter cache layout and prints the env name the CLI
# reads, so the path is never duplicated here.
eval "$(./setup-dflash.sh --print-paths)"
: "${MLXFAST_DFLASH_DRAFTER_DIR:?setup-dflash.sh did not provide the drafter path}"
if [[ ! -s "${MLXFAST_DFLASH_DRAFTER_DIR}/config.json" ]]; then
  echo "benchmark-dflash.sh: DFlash drafter is missing at ${MLXFAST_DFLASH_DRAFTER_DIR}; run ./setup-dflash.sh" >&2
  exit 1
fi

# --- transformed weights -----------------------------------------------------
# The DFlash target IS the serial reference checkpoint, so consume the weights/
# tree benchmark.sh already produced and cached instead of spending ~21.6 GB and
# many minutes re-transforming into a scratch directory that is then deleted.
#
# The staleness test has to be the SAME digest benchmark.sh stamps into
# weights/.benchmark-source.sha256, and two implementations of it would drift
# into a permanent false "stale weights" abort -- so reuse benchmark.sh's single
# definition of source_hash() instead of copying it. Fail closed if the
# extraction stops finding it.
source_hash_definition="$(awk '/^source_hash\(\) \{/,/^\}/' benchmark.sh)"
if [[ "${source_hash_definition}" != *"shasum -a 256"* ]]; then
  echo "benchmark-dflash.sh: could not reuse benchmark.sh's source_hash() definition;" >&2
  echo "benchmark-dflash.sh: benchmark.sh has been refactored -- refusing to guess the transform-source digest" >&2
  exit 1
fi
eval "${source_hash_definition}"

wanted_source_hash="$(source_hash)"
current_source_hash="$(cat "${weights_path}/.benchmark-source.sha256" 2>/dev/null | tr -d '[:space:]' || true)"
if [[ ! -f "${weights_path}/config.json" || "${current_source_hash}" != "${wanted_source_hash}" ]]; then
  cat >&2 <<EOF
benchmark-dflash.sh: no usable transformed weights at ${weights_path}/.
The DFlash target is the serial track's NVFP4 reference checkpoint, so this
script reuses the weights/ tree benchmark.sh produces and caches.
It does not run the ~21.6 GB transform itself.

Produce (or refresh) it first, then rerun this script:

  ./benchmark.sh --local-iterate

(expected transform-source digest ${wanted_source_hash}; found ${current_source_hash:-none})
EOF
  exit 1
fi

# --- local run guard: reuse ---------------------------------------------------
#
# Both local modes hold the ~21.6 GB target AND the drafter, so an overlapping
# local run (serial or DFlash) can out-of-memory the machine, and two runs
# sharing one GPU invalidate both timings. benchmark.sh's guard is TWO
# mechanisms, and reusing only one of them leaves a real hole:
#
#   * the per-user run lock excludes a concurrent local run. It guarded only
#     the serial direction until #814 -- benchmark.sh's resident-model scan
#     lists the dflash-* subcommands, so a serial run refuses to start against
#     a live DFlash one, but this script took no lock, so a DFlash run started
#     happily against a live serial run.
#   * the resident-model scan catches what NO lock can, and #814 did not
#     import it: an ORPHANED model-holding worker from a run that died without
#     releasing. The residency is live, nothing holds the lock, so the lock is
#     free and a DFlash run starts a second ~21.6 GB copy on top of it. That
#     is precisely the ppid-1 case benchmark.sh's own comment calls out.
#
# So reuse BOTH, at benchmark.sh's OWN lock path and with benchmark.sh's OWN
# process pattern, by extracting its definitions rather than copying them.
# What must not drift: the lock path (two implementations that disagree about
# where the lock lives would both "hold a lock" and exclude nothing) and the
# resident-process pattern (a second, staler pattern would miss exactly the
# subcommands benchmark.sh has since learned about). Same awk-extract-and-eval
# idiom the source_hash() reuse above uses.
#
# RESIDENT_MODEL_PROCESS_PATTERN is declared `readonly` in benchmark.sh.
# Re-evaluating that declaration here is fine: `readonly` only errors when the
# name is ALREADY readonly, and this is a fresh shell that has never set it.
#
# `local_run_guard_enabled` is defined HERE, before the eval, because
# benchmark.sh's version tests benchmark.sh's own mode flags. Shell resolves
# function calls at call time, so the extracted acquire/release/scan use this
# one. Both of this script's modes are local and model-holding, so the only
# opt-out is the documented debugging escape hatch, spelled exactly as
# benchmark.sh spells it.
LOCAL_RUN_LOCK_OWNED=""
local_run_guard_enabled() {
  [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]
}
run_lock_definitions="$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^list_resident_model_processes\(\) \{/,/^\}/' benchmark.sh
  awk '/^abort_if_model_already_resident\(\) \{/,/^\}/' benchmark.sh
)"
# Fail closed PER ARM. The previous check tested two sentinel strings against
# the CONCATENATION of the extractions, which is satisfied as long as ANY arm
# still contributes them: if the extraction of one function alone broke, both
# sentinels were still present, the eval succeeded, and the missing function
# surfaced later as `command not found` -- for release_local_run_lock, inside
# the EXIT trap, which under `set -e` abandons the rest of cleanup(), stranding
# the lock and the scratch directory the trap exists to remove. Ask each name
# whether it is actually defined instead, so no arm can be covered by another.
if ! eval "${run_lock_definitions}"; then
  echo "benchmark-dflash.sh: could not evaluate benchmark.sh's local run guard definitions;" >&2
  echo "benchmark-dflash.sh: benchmark.sh has been refactored -- refusing to run unguarded" >&2
  echo "benchmark-dflash.sh: (two overlapping local runs can out-of-memory this machine)" >&2
  exit 1
fi
for reused_definition in \
  local_run_lock_path \
  acquire_local_run_lock \
  release_local_run_lock \
  list_resident_model_processes \
  abort_if_model_already_resident
do
  if ! declare -F "${reused_definition}" >/dev/null 2>&1; then
    echo "benchmark-dflash.sh: could not reuse benchmark.sh's ${reused_definition}();" >&2
    echo "benchmark-dflash.sh: benchmark.sh has been refactored -- refusing to run unguarded" >&2
    echo "benchmark-dflash.sh: (two overlapping local runs can out-of-memory this machine)" >&2
    exit 1
  fi
done
# An empty pattern is worse than a missing one: `pgrep -f ''` under `set -u`
# aborts inside the scan's `|| true`, the scan reports a clean machine, and the
# guard passes silently. Check the value, not just the name.
if [[ -z "${RESIDENT_MODEL_PROCESS_PATTERN:-}" ]]; then
  echo "benchmark-dflash.sh: could not reuse benchmark.sh's RESIDENT_MODEL_PROCESS_PATTERN;" >&2
  echo "benchmark-dflash.sh: benchmark.sh has been refactored -- refusing to run unguarded" >&2
  echo "benchmark-dflash.sh: (an empty pattern matches nothing, so the orphan scan would pass silently)" >&2
  exit 1
fi

# --- scratch and local run guard: acquire -------------------------------------
run_dir=""
score_tmp=""

cleanup() {
  # Released FIRST: the lock is what another run is waiting on, and a failure
  # while removing scratch must not strand it.
  release_local_run_lock
  if [[ -n "${score_tmp}" ]]; then
    rm -f -- "${score_tmp}" || true
  fi
  if [[ -n "${run_dir}" ]]; then
    rm -rf -- "${run_dir}" || true
  fi
  # Leave nothing untracked at the repository root: a surviving work root is
  # exactly what `git add -A` sweeps into a submission diff (see the .gitignore
  # comment above score.json.sha256). rmdir only removes an EMPTY directory, so
  # an operator-chosen work root holding other files is never destroyed.
  rmdir "${work_root}" 2>/dev/null || true
  # Retired name from the MTP-era script; harmless when absent.
  rmdir ".mlxfast-local-mtp" 2>/dev/null || true
}
# Armed BEFORE the scratch directory is created, so an abort between mkdir and
# mktemp still leaves no work root behind.
trap cleanup EXIT

# Scratch, under the trap and only after the reuse above was validated -- so
# the fail-closed arms abort before there is anything to leave behind, and
# every abort after this point is swept up. A surviving work root is exactly
# what `git add -A` sweeps into a submission diff.
mkdir -p "${work_root}"
if [[ "$(dirname "${score_path}")" != "." ]]; then
  mkdir -p "$(dirname "${score_path}")"
fi

# Scanned BEFORE the lock is taken. An orphan holds no lock, so the lock would
# be granted and the abort would come one step later anyway; running the scan
# first means the run never takes a lock it is about to give back, and the
# operator sees the accurate diagnosis (a live pid to inspect and kill) rather
# than a lock message about a run that is already dead. WARN-AND-ABORT only --
# benchmark.sh's rule, and it is benchmark.sh's function doing the aborting.
abort_if_model_already_resident

# Acquired once the trap can release it, and before anything expensive --
# including the thermal gate. Holding the lock while waiting to cool is the
# point: the alternative lets a second run start, heat the GPU, and invalidate
# the wait this run just paid for.
acquire_local_run_lock

run_dir="$(mktemp -d "${work_root%/}/run.XXXXXX")"
score_tmp="${score_path}.tmp"

tripwire_report="${run_dir}/public-gate.json"
plan_path="${run_dir}/seed-plan.json"
golden_path="${run_dir}/dflash-reference-golden.json"
serial_report="${run_dir}/serial-control.json"
dflash_report="${run_dir}/block-decode.json"

# Mirror benchmark.sh's inline seal: require EXACTLY one JSON object. `jq -e`
# alone evaluates its filter per object on a concatenated stream and reports only
# the last object's status, so an injected extra write would pass unnoticed.
require_single_json_object() {
  if [[ "$(jq -s 'length' "$1" 2>/dev/null)" != "1" ]]; then
    echo "benchmark-dflash.sh: $2 did not emit a single JSON object on stdout" >&2
    return 1
  fi
}

run_cool_gate() {
  echo "benchmark-dflash.sh: GPU cool gate before $1 (${cool_gate_command[*]})" >&2
  if ! "${cool_gate_command[@]}"; then
    echo "benchmark-dflash.sh: GPU cool gate failed before $1; free the GPU and rerun" >&2
    return 1
  fi
}

# --- 1. public drift tripwire ------------------------------------------------
# The one gate the local DFlash path lacked entirely. Same subcommand, same
# fixture, no flags and no DFlash awareness: if the submitted model or kernel
# code broke ordinary Laguna decode, this fails before anything is measured.
echo "benchmark-dflash.sh: public drift tripwire (correctness against ${public_golden_path})" >&2
tripwire_status=0
"${swift_bin}" correctness \
  --weights "${weights_path}" \
  --golden "${public_golden_path}" > "${tripwire_report}" || tripwire_status=$?
require_single_json_object "${tripwire_report}" "public drift tripwire"
if [[ "${tripwire_status}" != "0" ]] \
    || ! jq -e '.passed == true' "${tripwire_report}" >/dev/null 2>&1; then
  jq -r '"benchmark-dflash.sh: public gate passed=\(.passed) error=\(.error // "-")"' \
    "${tripwire_report}" >&2 || true
  echo "benchmark-dflash.sh: public drift tripwire FAILED (exit ${tripwire_status}); not measuring a broken build" >&2
  echo "benchmark-dflash.sh: note: the public goldens are M5-generated, so on other Apple Silicon a deterministic near-tie token mismatch can be expected for a correct build; check unmodified main on this machine before assuming a regression" >&2
  exit 1
fi

# --- 2. reference golden -----------------------------------------------------
# `generate-golden` writes a GoldenDocument (cases/prompt_tokens/expected_tokens),
# which the DFlash subcommands cannot decode. `dflash-reference` is the existing
# producer of the shape they do decode.
if ! jq -e '
    (.cases | type == "array") and (.cases | length) > 0
    and (.cases[0].prompt_tokens | type == "array")
    and (.cases[0].prompt_tokens | length) > 0
  ' "${public_golden_path}" >/dev/null 2>&1; then
  echo "benchmark-dflash.sh: ${public_golden_path} carries no prompt tokens to seed the DFlash reference" >&2
  exit 1
fi
jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
  "${public_golden_path}" > "${plan_path}"

run_cool_gate "the DFlash reference pass"
echo "benchmark-dflash.sh: generating the DFlash reference golden (${token_count} rows, K=${block_size})" >&2
"${swift_bin}" dflash-reference \
  --weights "${weights_path}" \
  --drafter "${MLXFAST_DFLASH_DRAFTER_DIR}" \
  --emitted "${plan_path}" \
  --generate "${token_count}" \
  --block-size "${block_size}" \
  --output "${golden_path}" \
  --plan-output "${run_dir}/generated-plan.json"

# Go-live runbook step B / contract Amendment 10: reference_self_consistent alone
# did NOT check that the recorded rows agree with the chain they were generated
# against, and three goldens shipped with exactly that contradiction. Assert
# BOTH, plus a fail-closed row-count preflight: the measured legs require
# rows >= --tokens, and a short golden must be a legible abort here rather than a
# raw range error mid-measurement.
require_single_json_object "${golden_path}" "the DFlash reference golden"
if ! jq -e --argjson tokens "${token_count}" '
    . as $g
    | ($g.rows | length) as $rows
    | $g.reference_self_consistent == true
      and ($g.seed_tokens | type == "array") and ($g.seed_tokens | length) > 0
      and ($g.rows | type == "array") and $rows > 0
      and ($g.emitted_tokens | type == "array")
      and ($g.emitted_tokens | length) == $rows
      and $rows >= $tokens
      and ([
            range(0; $rows)
            | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)
          ] | length) == 0
  ' "${golden_path}" >/dev/null 2>&1; then
  echo "benchmark-dflash.sh: the DFlash reference golden is not usable: it must report" >&2
  echo "benchmark-dflash.sh:   reference_self_consistent == true," >&2
  echo "benchmark-dflash.sh:   emitted_tokens[i] == rows[i].sequential_argmax for every row, and" >&2
  echo "benchmark-dflash.sh:   at least ${token_count} rows (see docs/dflash-go-live-runbook.md step B)" >&2
  exit 1
fi

# --- 3. serial K=1 control (denominator) -------------------------------------
# dflash-probe is dflash-benchmark with the width pinned to 1: same worker, same
# protocol, same forward, drafter loaded and never used. That is what makes the
# ratio like-for-like.
run_cool_gate "the serial K=1 control"
echo "benchmark-dflash.sh: measuring the serial K=1 control (${token_count} tokens)" >&2
"${swift_bin}" dflash-probe \
  --weights "${weights_path}" \
  --drafter "${MLXFAST_DFLASH_DRAFTER_DIR}" \
  --golden "${golden_path}" \
  --tokens "${token_count}" > "${serial_report}"

# --- 4. block decode (numerator) ---------------------------------------------
run_cool_gate "the DFlash block decode"
echo "benchmark-dflash.sh: measuring DFlash block decode (${token_count} tokens, K=${block_size})" >&2
"${swift_bin}" dflash-benchmark \
  --weights "${weights_path}" \
  --drafter "${MLXFAST_DFLASH_DRAFTER_DIR}" \
  --golden "${golden_path}" \
  --tokens "${token_count}" \
  --block-size "${block_size}" > "${dflash_report}"

# --- 5. seal and validate both reports ---------------------------------------
require_single_json_object "${serial_report}" "the serial K=1 control"
jq -e --arg track "${TRACK_ID}" --argjson tokens "${token_count}" '
  .track_id == $track
  and .official_score_produced == false
  and .uses_trained_drafter == false
  and .block_size == 1
  and .decode_token_count == $tokens
  and .all_tokens_matched == true
  and (.parent_measured_seconds_per_token | type == "number" and . > 0)
' "${serial_report}" >/dev/null

require_single_json_object "${dflash_report}" "the DFlash block decode"
jq -e --arg track "${TRACK_ID}" --argjson tokens "${token_count}" '
  .track_id == $track
  and .official_score_produced == false
  and .uses_trained_drafter == true
  and .block_size > 1
  and .decode_token_count == $tokens
  and .all_tokens_matched == true
  and (.parent_measured_seconds_per_token | type == "number" and . > 0)
' "${dflash_report}" >/dev/null

serial_spt="$(jq -r '.parent_measured_seconds_per_token' "${serial_report}")"
dflash_spt="$(jq -r '.parent_measured_seconds_per_token' "${dflash_report}")"
score="$(jq -n --argjson serial "${serial_spt}" --argjson dflash "${dflash_spt}" '$serial / $dflash')"
jq -e 'type == "number" and . > 0' <<<"${score}" >/dev/null

# --- 6. report the RANKED floor, and only ever a directional score ------------
# The floor that actually rejects is the workflow's env value (mirrored by the
# box wrapper's MIN_ACCEPTED_SPEEDUP); benchmark.json's
# scoring.decodeSpeedupFloor is documentation and is not enforced anywhere. Read
# the enforced value straight out of the workflow so this line cannot drift from
# it.
ranked_floor="$(
  awk -F'"' '/^[[:space:]]*MLXFAST_DFLASH_DECODE_SPEEDUP_FLOOR:[[:space:]]*"/ { print $2; exit }' \
    "${workflow_path}" 2>/dev/null || true
)"
if [[ ! "${ranked_floor}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "benchmark-dflash.sh: warning: could not read the ranked decode floor from ${workflow_path}" >&2
  ranked_floor=""
fi

official_scoring_enabled="$(jq -r '.official_scoring_enabled // false' "${contract_path}")"
timed_pool_size="$(jq -r '(.timed_prompt_pool // []) | length' "${contract_path}")"

{
  printf 'benchmark-dflash.sh: local estimated decode speedup %s (serial K=1 %ss/token / DFlash %ss/token)\n' \
    "${score}" "${serial_spt}" "${dflash_spt}"
  if [[ -n "${ranked_floor}" ]]; then
    printf 'benchmark-dflash.sh: the RANKED decode floor enforced on the M5 box is %s (%s)\n' \
      "${ranked_floor}" "${workflow_path}"
    printf 'benchmark-dflash.sh: this local number is shown against that floor for DIRECTION ONLY; it is not a verdict\n'
  fi
  printf 'benchmark-dflash.sh: NOT RANKABLE: official_scoring_enabled=%s, hidden timed prompt pool holds %s entries,\n' \
    "${official_scoring_enabled}" "${timed_pool_size}"
  printf 'benchmark-dflash.sh: and this golden came from the candidate build, not the pinned baseline. Only the\n'
  printf 'benchmark-dflash.sh: ranked %s run can produce a score that ranks.\n' "${workflow_path}"
} >&2

jq -n \
  --arg track "${TRACK_ID}" \
  --arg mode "${mode#--}" \
  --arg ranked_floor "${ranked_floor}" \
  --argjson score "${score}" \
  --argjson token_count "${token_count}" \
  --argjson block_size "${block_size}" \
  --argjson serial_spt "${serial_spt}" \
  --argjson dflash_spt "${dflash_spt}" \
  --argjson accepted_rate "$(jq -r '.accepted_draft_rate' "${dflash_report}")" \
  --argjson residual_divergences "$(jq -r '.residual_divergence_count' "${dflash_report}")" \
  '{
    score: $score,
    passed: true,
    track_id: $track,
    metrics: {
      mode: ("dflash-" + $mode),
      oracle: "candidate-local-dflash-reference-golden",
      official_score: false,
      rankable: false,
      not_rankable_reason: "candidate-generated reference golden; official scoring disabled; ranked M5 run is the only authority",
      ranked_decode_speedup_floor: (if $ranked_floor == "" then null else ($ranked_floor | tonumber) end),
      public_drift_tripwire_passed: true,
      decode_tokens: $token_count,
      block_size: $block_size,
      all_tokens_matched: true,
      uses_trained_drafter: true,
      serial_seconds_per_token: $serial_spt,
      dflash_seconds_per_token: $dflash_spt,
      dflash_decode_speedup: $score,
      accepted_draft_rate: $accepted_rate,
      residual_divergence_count: $residual_divergences
    }
  }' > "${score_tmp}"
mv "${score_tmp}" "${score_path}"
cat "${score_path}"
