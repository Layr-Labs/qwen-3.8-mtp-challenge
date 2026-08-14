#!/usr/bin/env bash
# Local directional runner for the Qwen 3.6 native-MTP speculative-decode track
# (qwen3.8-27b-mtp-v1).
#
# Every gate and every measurement below is an EXISTING harness entrypoint; this
# script only wires them together in the right order. In order:
#
#   1. public drift tripwire   `mlxfast-swift correctness` against the checked-in
#                              public fixture -- the same subcommand and the same
#                              fixture the base local loop uses.
#   2. GPU cool gate           `./benchmark.sh --local-cool-gate-only`, the base
#                              local loop's own 40C gate, before EACH
#                              model-resident leg.
#   3. TRUE serial control     `mlxfast-swift mtp-timed --mtp-depth 0`  (denominator)
#   4. native-MTP decode       `mlxfast-swift mtp-timed`            (numerator)
#
# The reference rows both measured legs decode come from `mlxfast-swift
# mtp-verify --generate`, the producer of the shape they consume.
#
# WHAT THIS SCORE IS NOT. It is DIRECTIONAL and can never be a ranked score:
#   * the reference rows are produced by the CANDIDATE's own build, not by the
#     pinned baseline, so it checks MTP/serial parity and speed direction -- not
#     target fidelity;
#   * the ranked run times ALL 8 hidden pool prompts, normalises each against
#     that prompt's own pinned no-op reference and publishes the MEDIAN of the 8
#     -- none of which this local loop can reproduce (it has one prompt, no
#     pinned references and no median to take);
#   * .github/workflows/qwen-mtp-ranked-benchmark.yml measures against
#     organizer-pinned hidden goldens on the ranked box and is the only authority
#     for both fidelity and score.
#
# ONE MODEL-HOLDING RUN AT A TIME. A native-MTP residency is the Qwen 3.6 27B
# target plus the ~247 MB MTP head. This script reuses BOTH halves of
# benchmark.sh's memory guard -- the per-user run lock (which excludes a
# concurrent local run in either direction) and the resident-model scan (which
# catches what no lock can: an ORPHANED model-holding worker from a run that died
# without releasing) -- by EXTRACTING benchmark.sh's own definitions rather than
# copying them, so the lock path and the process pattern cannot drift into two
# implementations that both "hold a lock" and exclude nothing.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./benchmark-qwen-mtp.sh [--local-iterate|--local-submit]

Runs the native-MTP decode path against a candidate-local serial K=1 control and
writes a directional Yukon score payload to MLXFAST_SCORE_PATH
(default: score.json).

Prerequisites (all reused, never rebuilt here):
  ./setup.sh                     builds mlxfast-swift + the MLX metallib and
                                 provisions the pinned Qwen 3.6 27B checkpoint
  ./setup-qwen-mtp.sh            provisions the pinned MTP head
  ./benchmark.sh --local-iterate produces and caches the transformed weights/
                                 tree this script consumes

Environment:
  MLXFAST_SCORE_PATH                      score payload path (default score.json)
  MLXFAST_WEIGHTS_PATH                    transformed weights tree (default weights)
  MLXFAST_SWIFT_BIN                       trusted CLI (default .build/release/mlxfast-swift)
  MLXFAST_QWEN_MTP_DEPTH                  speculative depth, 2..4 (default 2,
                                          the ranked workflow's value)
  MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS   decode tokens for --local-iterate (default 64)
  MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS    decode tokens for --local-submit (default 128)
  MLXFAST_QWEN_MTP_LOCAL_WORK_DIR         scratch root (default .mlxfast-local-qwen-mtp,
                                          removed on exit)
  MLXFAST_QWEN_MTP_HEAD_DIR               MTP head directory (default: setup-qwen-mtp.sh's cache)
  MLXFAST_GPU_TEMP_CMD                    benchmark.sh's documented cool-gate seam
EOF
}

TRACK_ID="qwen3.8-27b-mtp-v1"
# The two CLI verbs this script drives. Named so that no retired MTP surface
# name survives as a substring: appending "probe", "benchmark" or "reference"
# directly after "mtp-" would reproduce one, and prefixing such a name with
# "qwen-" would NOT clear it.
VERIFY_VERB="mtp-verify"
TIMED_VERB="mtp-timed"

mode="${1:---local-iterate}"
case "${mode}" in
  --local-iterate)
    token_count="${MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS:-64}"
    # Which public fixture the drift tripwire runs against mirrors
    # benchmark.sh's own choice: the 256-step golden for the fast edit loop, the
    # 1024-step golden for the pre-submit check. Same fixtures, same
    # calibration, nothing new decided here.
    public_golden_path="${MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE:-correctness_prompts/public_longcopy_gate_english_512_256.json}"
    ;;
  --local-submit)
    token_count="${MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS:-128}"
    public_golden_path="${MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE:-correctness_prompts/public_longcopy_gate_english_512_1024.json}"
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
  echo "benchmark-qwen-mtp.sh: token count must be an integer in 1...512" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
cli_source="Sources/MLXFastCLI/main.swift"
# The contract fixture is READ here (for the not-rankable statement below),
# never passed to the CLI: the CLI declares no contract option, because the
# workflow interlock and the trusted parent are what enforce the contract.
contract_path="${MLXFAST_QWEN_MTP_CONTRACT_PATH:-fixtures/qwen3_8_27b_mtp_track.json}"
weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"
score_path="${MLXFAST_SCORE_PATH:-score.json}"
work_root="${MLXFAST_QWEN_MTP_LOCAL_WORK_DIR:-.mlxfast-local-qwen-mtp}"
# Same env name the ranked workflow uses for the OFFERED per-round draft
# ceiling, so a local run and a ranked run offer the same width. It is an offer,
# not a schedule: the shipped Qwen36MTPBlockSession.draftPolicy returns
# min(offer, 2) and a submission owns that function. The default here matches
# the workflow's.
depth="${MLXFAST_QWEN_MTP_DEPTH:-8}"
workflow_path=".github/workflows/qwen-mtp-ranked-benchmark.yml"
# Reuse the base local loop's thermal gate rather than reimplementing it. This
# branch of benchmark.sh returns before the metallib check, the builds, the
# sandbox and the transform, and takes no run lock, so it cannot deadlock here.
cool_gate_command=("./benchmark.sh" "--local-cool-gate-only")

# 1 is the floor for a speculative OFFER (0 is the serial control, which this
# script measures separately as the denominator). 8 is the ceiling and is the
# trusted bound MLXFastConstants.qwenMTPMaxDraftDepth -- the parent will not
# offer more, and the ledger is closed against that constant whatever a round
# actually drafts. The old 2...4 range was a bound on a PINNED depth; depth is
# competitive surface now, so the range is the whole legal offer.
if ! [[ "${depth}" =~ ^[1-8]$ ]]; then
  echo "benchmark-qwen-mtp.sh: MLXFAST_QWEN_MTP_DEPTH must be an integer in 1...8 (the offered per-round draft ceiling; 0 is the serial control this script measures itself)" >&2
  exit 2
fi

for tool in jq awk; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "benchmark-qwen-mtp.sh: required tool not found: ${tool}" >&2
    exit 1
  fi
done
# QWEN-MTP-PENDING-PHASE3, checked BEFORE anything expensive and BEFORE the
# binary check: the two verbs this script drives do not exist on the CLI yet
# (harness conversion is phase 3 of QWEN36-MTP-CHALLENGE-PLAN.md), and that is
# the structural blocker -- "run ./setup.sh first" would be a true but useless
# thing to say to someone whose build could not contain these verbs anyway.
# Asserted against the CLI's own dispatch source rather than by invoking the
# binary, so the diagnostic is the same whether or not a stale binary happens to
# be lying around.
missing_verbs=()
for verb in "${VERIFY_VERB}" "${TIMED_VERB}"; do
  if [[ ! -f "${cli_source}" ]] || ! grep -qF "case \"${verb}\":" "${cli_source}"; then
    missing_verbs+=("${verb}")
  fi
done
if (( ${#missing_verbs[@]} > 0 )); then
  cat >&2 <<EOF
benchmark-qwen-mtp.sh: QWEN-MTP-PENDING-PHASE3

The trusted CLI does not declare: ${missing_verbs[*]}

The native-MTP verify and timed subcommands are phase-3 work (the harness
conversion). This runner drives them and the ranked workflow
(${workflow_path}) drives the same two verbs, so both are
blocked on the same change. Nothing is measured until they land -- producing a
number from some other code path would be a directional score for a decode
schedule this track does not run.
EOF
  exit 1
fi

if [[ ! -x "${swift_bin}" ]]; then
  echo "benchmark-qwen-mtp.sh: missing ${swift_bin}; run ./setup.sh first" >&2
  exit 1
fi

if [[ ! -s "${public_golden_path}" ]]; then
  echo "benchmark-qwen-mtp.sh: missing public correctness fixture ${public_golden_path}" >&2
  echo "benchmark-qwen-mtp.sh: re-sync the repository (public fixtures live in correctness_prompts/)" >&2
  exit 1
fi
if [[ ! -x "${cool_gate_command[0]}" ]]; then
  echo "benchmark-qwen-mtp.sh: missing ${cool_gate_command[0]}; the GPU cool gate is not optional" >&2
  exit 1
fi

# --- MTP head ----------------------------------------------------------------
# setup-qwen-mtp.sh owns the head cache layout and prints the env name the CLI
# reads, so the path is never duplicated here.
eval "$(./setup-qwen-mtp.sh --print-paths)"
: "${MLXFAST_QWEN_MTP_HEAD_DIR:?setup-qwen-mtp.sh did not provide the MTP head path}"
if [[ ! -s "${MLXFAST_QWEN_MTP_HEAD_DIR}/config.json" ]]; then
  echo "benchmark-qwen-mtp.sh: MTP head is missing at ${MLXFAST_QWEN_MTP_HEAD_DIR}; run ./setup-qwen-mtp.sh" >&2
  exit 1
fi

# --- transformed weights -----------------------------------------------------
# The MTP track's target IS the base reference checkpoint, so consume the
# weights/ tree benchmark.sh already produced and cached instead of spending
# ~16 GB and many minutes re-transforming into a scratch directory that is then
# deleted.
#
# The staleness test has to be the SAME digest benchmark.sh stamps into
# weights/.benchmark-source.sha256, and two implementations of it would drift
# into a permanent false "stale weights" abort -- so reuse benchmark.sh's single
# definition of source_hash() instead of copying it. Fail closed if the
# extraction stops finding it.
source_hash_definition="$(awk '/^source_hash\(\) \{/,/^\}/' benchmark.sh)"
if [[ "${source_hash_definition}" != *"shasum -a 256"* ]]; then
  echo "benchmark-qwen-mtp.sh: could not reuse benchmark.sh's source_hash() definition;" >&2
  echo "benchmark-qwen-mtp.sh: benchmark.sh has been refactored -- refusing to guess the transform-source digest" >&2
  exit 1
fi
eval "${source_hash_definition}"

wanted_source_hash="$(source_hash)"
current_source_hash="$(cat "${weights_path}/.benchmark-source.sha256" 2>/dev/null | tr -d '[:space:]' || true)"
if [[ ! -f "${weights_path}/config.json" || "${current_source_hash}" != "${wanted_source_hash}" ]]; then
  cat >&2 <<EOF
benchmark-qwen-mtp.sh: no usable transformed weights at ${weights_path}/.
The MTP track's target is the base reference checkpoint, so this script reuses
the weights/ tree benchmark.sh produces and caches. It does not run the ~16 GB
transform itself.

Produce (or refresh) it first, then rerun this script:

  ./benchmark.sh --local-iterate

(expected transform-source digest ${wanted_source_hash}; found ${current_source_hash:-none})
EOF
  exit 1
fi

# --- local run guard: reuse ---------------------------------------------------
# `local_run_guard_enabled` is defined HERE, before the eval, because
# benchmark.sh's version tests benchmark.sh's own mode flags. Shell resolves
# function calls at call time, so the extracted acquire/release/scan use this
# one. Both of this script's modes are local and model-holding, so the only
# opt-out is the documented debugging escape hatch, spelled exactly as
# benchmark.sh spells it.
#
# RESIDENT_MODEL_PROCESS_PATTERN is declared `readonly` in benchmark.sh.
# Re-evaluating that declaration here is fine: `readonly` only errors when the
# name is ALREADY readonly, and this is a fresh shell that has never set it.
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
if ! eval "${run_lock_definitions}"; then
  echo "benchmark-qwen-mtp.sh: could not evaluate benchmark.sh's local run guard definitions;" >&2
  echo "benchmark-qwen-mtp.sh: benchmark.sh has been refactored -- refusing to run unguarded" >&2
  echo "benchmark-qwen-mtp.sh: (two overlapping local runs can out-of-memory this machine)" >&2
  exit 1
fi
# Fail closed PER ARM: asking each name whether it is actually defined, so no
# arm's failure can be masked by another arm still contributing a sentinel
# string. A missing release_local_run_lock would otherwise surface as `command
# not found` inside the EXIT trap, which under `set -e` abandons the rest of
# cleanup() and strands the lock the trap exists to release.
for reused_definition in \
  local_run_lock_path \
  acquire_local_run_lock \
  release_local_run_lock \
  list_resident_model_processes \
  abort_if_model_already_resident
do
  if ! declare -F "${reused_definition}" >/dev/null 2>&1; then
    echo "benchmark-qwen-mtp.sh: could not reuse benchmark.sh's ${reused_definition}();" >&2
    echo "benchmark-qwen-mtp.sh: benchmark.sh has been refactored -- refusing to run unguarded" >&2
    echo "benchmark-qwen-mtp.sh: (two overlapping local runs can out-of-memory this machine)" >&2
    exit 1
  fi
done
# An empty pattern is worse than a missing one: `pgrep -f ''` under `set -u`
# aborts inside the scan's `|| true`, the scan reports a clean machine, and the
# guard passes silently. Check the value, not just the name.
if [[ -z "${RESIDENT_MODEL_PROCESS_PATTERN:-}" ]]; then
  echo "benchmark-qwen-mtp.sh: could not reuse benchmark.sh's RESIDENT_MODEL_PROCESS_PATTERN;" >&2
  echo "benchmark-qwen-mtp.sh: benchmark.sh has been refactored -- refusing to run unguarded" >&2
  echo "benchmark-qwen-mtp.sh: (an empty pattern matches nothing, so the orphan scan would pass silently)" >&2
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
  # exactly what `git add -A` sweeps into a submission diff. rmdir only removes
  # an EMPTY directory, so an operator-chosen work root holding other files is
  # never destroyed.
  rmdir "${work_root}" 2>/dev/null || true
}
# Armed BEFORE the scratch directory is created, so an abort between mkdir and
# mktemp still leaves no work root behind.
trap cleanup EXIT

mkdir -p "${work_root}"
if [[ "$(dirname "${score_path}")" != "." ]]; then
  mkdir -p "$(dirname "${score_path}")"
fi

# Scanned BEFORE the lock is taken. An orphan holds no lock, so the lock would
# be granted and the abort would come one step later anyway; running the scan
# first means the run never takes a lock it is about to give back, and the
# operator sees the accurate diagnosis (a live pid to inspect and kill) rather
# than a lock message about a run that is already dead.
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
golden_path="${run_dir}/qwen-mtp-golden-rows.json"
serial_report="${run_dir}/serial-control.json"
mtp_report="${run_dir}/mtp-decode.json"

# Mirror benchmark.sh's inline seal: require EXACTLY one JSON object. `jq -e`
# alone evaluates its filter per object on a concatenated stream and reports only
# the last object's status, so an injected extra write would pass unnoticed.
require_single_json_object() {
  if [[ "$(jq -s 'length' "$1" 2>/dev/null)" != "1" ]]; then
    echo "benchmark-qwen-mtp.sh: $2 did not emit a single JSON object on stdout" >&2
    return 1
  fi
}

run_cool_gate() {
  echo "benchmark-qwen-mtp.sh: GPU cool gate before $1 (${cool_gate_command[*]})" >&2
  if ! "${cool_gate_command[@]}"; then
    echo "benchmark-qwen-mtp.sh: GPU cool gate failed before $1; free the GPU and rerun" >&2
    return 1
  fi
}

# --- 1. public drift tripwire ------------------------------------------------
# Same subcommand, same fixture, no flags and no MTP awareness: if the submitted
# model or kernel code broke ordinary Qwen decode, this fails before anything is
# measured.
echo "benchmark-qwen-mtp.sh: public drift tripwire (correctness against ${public_golden_path})" >&2
tripwire_status=0
"${swift_bin}" correctness \
  --weights "${weights_path}" \
  --golden "${public_golden_path}" > "${tripwire_report}" || tripwire_status=$?
require_single_json_object "${tripwire_report}" "public drift tripwire"
if [[ "${tripwire_status}" != "0" ]] \
    || ! jq -e '.passed == true' "${tripwire_report}" >/dev/null 2>&1; then
  jq -r '"benchmark-qwen-mtp.sh: public gate passed=\(.passed) error=\(.error // "-")"' \
    "${tripwire_report}" >&2 || true
  echo "benchmark-qwen-mtp.sh: public drift tripwire FAILED (exit ${tripwire_status}); not measuring a broken build" >&2
  echo "benchmark-qwen-mtp.sh: note: the public goldens are M5-generated, so on other Apple Silicon a deterministic near-tie token mismatch can be expected for a correct build; check unmodified main on this machine before assuming a regression" >&2
  exit 1
fi

# --- 2. reference rows -------------------------------------------------------
# The public fixture's cases[0].prompt_tokens IS the tokenization of
# correctness_prompts/public_longcopy_gate_english_512.txt, which is how the
# seed is obtained without a tokenizer -- the trusted CLI's MTP subcommands take
# token ids, never text.
if ! jq -e '
    (.cases | type == "array") and (.cases | length) > 0
    and (.cases[0].prompt_tokens | type == "array")
    and (.cases[0].prompt_tokens | length) > 0
  ' "${public_golden_path}" >/dev/null 2>&1; then
  echo "benchmark-qwen-mtp.sh: ${public_golden_path} carries no prompt tokens to seed the MTP reference" >&2
  exit 1
fi
jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
  "${public_golden_path}" > "${plan_path}"

run_cool_gate "the MTP reference pass"
echo "benchmark-qwen-mtp.sh: generating the MTP reference rows (${token_count} rows, depth=${depth})" >&2
"${swift_bin}" "${VERIFY_VERB}" \
  --weights "${weights_path}" \
  --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
  --emitted "${plan_path}" \
  --generate "$(( token_count + 1 ))" \
  --mtp-depth "${depth}" \
  --output "${golden_path}" \
  --plan-output "${run_dir}/generated-plan.json"

# reference_self_consistent alone does NOT check that the recorded rows agree
# with the chain they were generated against -- three DFlash goldens shipped
# with exactly that contradiction. Assert BOTH, plus a fail-closed row-count
# preflight: the measured legs require rows >= --tokens, and a short reference
# must be a legible abort here rather than a raw range error mid-measurement.
require_single_json_object "${golden_path}" "the MTP reference rows"
if ! jq -e --argjson tokens "${token_count}" '
    . as $g
    | ($g.rows | length) as $rows
    | $g.reference_self_consistent == true
      and ($g.seed_tokens | type == "array") and ($g.seed_tokens | length) > 0
      and ($g.rows | type == "array") and $rows > 0
      and ($g.emitted_tokens | type == "array")
      and ($g.emitted_tokens | length) == $rows
      and $rows >= ($tokens + 1)
      and ([
            range(0; $rows)
            | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)
          ] | length) == 0
  ' "${golden_path}" >/dev/null 2>&1; then
  echo "benchmark-qwen-mtp.sh: the MTP reference rows are not usable: they must report" >&2
  echo "benchmark-qwen-mtp.sh:   reference_self_consistent == true," >&2
  echo "benchmark-qwen-mtp.sh:   emitted_tokens[i] == rows[i].sequential_argmax for every row, and" >&2
  echo "benchmark-qwen-mtp.sh:   at least $(( token_count + 1 )) rows (one per emitted token after the" >&2
  echo "benchmark-qwen-mtp.sh:   seed argmax, plus one for the final round's target tail row)" >&2
  exit 1
fi

# --- 3. TRUE serial control, MTP OFF (denominator) ---------------------------
# Depth 0 is the same binary, the same worker, the same protocol and the same
# target forward with speculation switched OFF: one token per forward, no
# drafting, no head consultation on the hot path. The head stays LOADED so its
# residency is charged to both sides; only the drafting differs.
#
# NOT depth 1. Depth 1 still drafts, verifies and accepts -- box 3 measured 302
# rounds at a 0.699 accept rate over a 512-token window -- so it is an
# already-accelerated baseline, and dividing by it measures depth 2 against
# one-deep speculation instead of against serial decode.
run_cool_gate "the true serial control"
echo "benchmark-qwen-mtp.sh: measuring the TRUE serial control, MTP off (${token_count} tokens)" >&2
"${swift_bin}" "${TIMED_VERB}" \
  --weights "${weights_path}" \
  --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
  --golden "${golden_path}" \
  --tokens "${token_count}" \
  --mtp-depth 0 > "${serial_report}"

# --- 4. native-MTP decode (numerator) ----------------------------------------
run_cool_gate "the native-MTP decode"
echo "benchmark-qwen-mtp.sh: measuring native-MTP decode (${token_count} tokens, depth=${depth})" >&2
"${swift_bin}" "${TIMED_VERB}" \
  --weights "${weights_path}" \
  --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
  --golden "${golden_path}" \
  --tokens "${token_count}" \
  --mtp-depth "${depth}" > "${mtp_report}"

# --- 5. seal and validate both reports ---------------------------------------
require_single_json_object "${serial_report}" "the serial K=1 control"
jq -e --arg track "${TRACK_ID}" --argjson tokens "${token_count}" '
  .track_id == $track
  and .official_score_produced == false
  # `.uses_pinned_mtp_head` is PROVENANCE, not a gate (contract change
  # 2026-08-14) -- it now reports whether the head actually drafted, and head
  # weights plus draft schedule are competitive surface. What identifies the
  # serial control is the depth it was RUN at, which is what these two assert.
  and (.uses_pinned_mtp_head | type == "boolean")
  and .is_serial_control == true
  and .mtp_depth == 0
  and .decode_token_count == $tokens
  and .all_tokens_matched == true
  and (.parent_measured_seconds_per_token | type == "number" and . > 0)
' "${serial_report}" >/dev/null

require_single_json_object "${mtp_report}" "the native-MTP decode"
jq -e --arg track "${TRACK_ID}" --argjson tokens "${token_count}" '
  .track_id == $track
  and .official_score_produced == false
  # Provenance, not a gate: an adaptive policy may legally draft nothing, in
  # which case this is false and the run is still valid. The side identity is
  # `.is_serial_control == false`, which is about the OFFER, not the schedule.
  and (.uses_pinned_mtp_head | type == "boolean")
  and (.head_provenance.sha256 | type == "string")
  and .is_serial_control == false
  and .mtp_depth >= 1
  and .decode_token_count == $tokens
  and .all_tokens_matched == true
  and (.parent_measured_seconds_per_token | type == "number" and . > 0)
' "${mtp_report}" >/dev/null

serial_spt="$(jq -r '.parent_measured_seconds_per_token' "${serial_report}")"
mtp_spt="$(jq -r '.parent_measured_seconds_per_token' "${mtp_report}")"
score="$(jq -n --argjson serial "${serial_spt}" --argjson mtp "${mtp_spt}" '$serial / $mtp')"
jq -e 'type == "number" and . > 0' <<<"${score}" >/dev/null

# --- 6. report the RANKED floor, and only ever a directional score ------------
# The floor that actually rejects is the workflow's env value. Read the enforced
# value straight out of the workflow so this line cannot drift from it. While
# the track is pre-go-live that value is deliberately a placeholder, not a
# number, and the regex below leaves ranked_floor empty rather than printing a
# threshold nobody has measured.
ranked_floor="$(
  awk -F'"' '/^[[:space:]]*MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR:[[:space:]]*"/ { print $2; exit }' \
    "${workflow_path}" 2>/dev/null || true
)"
if [[ ! "${ranked_floor}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "benchmark-qwen-mtp.sh: the ranked decode floor in ${workflow_path} is not yet a number (${ranked_floor:-unset}); it is derived from the hidden pool's own measured no-op references" >&2
  ranked_floor=""
fi

official_scoring_enabled=false
timed_pool_size=0
if [[ -s "${contract_path}" ]]; then
  official_scoring_enabled="$(jq -r '.official_scoring_enabled // false' "${contract_path}")"
  timed_pool_size="$(jq -r '(.timed_prompt_pool // []) | length' "${contract_path}")"
fi

{
  printf 'benchmark-qwen-mtp.sh: local estimated decode speedup %s (serial K=1 %ss/token / MTP %ss/token)\n' \
    "${score}" "${serial_spt}" "${mtp_spt}"
  if [[ -n "${ranked_floor}" ]]; then
    printf 'benchmark-qwen-mtp.sh: the RANKED decode floor enforced on the ranked box is %s (%s)\n' \
      "${ranked_floor}" "${workflow_path}"
    printf 'benchmark-qwen-mtp.sh: this local number is shown against that floor for DIRECTION ONLY; it is not a verdict\n'
  fi
  printf 'benchmark-qwen-mtp.sh: NOT RANKABLE: official_scoring_enabled=%s, hidden timed prompt pool holds %s entries,\n' \
    "${official_scoring_enabled}" "${timed_pool_size}"
  printf 'benchmark-qwen-mtp.sh: and these reference rows came from the candidate build, not the pinned baseline. Only the\n'
  printf 'benchmark-qwen-mtp.sh: ranked %s run can produce a score that ranks.\n' "${workflow_path}"
} >&2

jq -n \
  --arg track "${TRACK_ID}" \
  --arg mode "${mode#--}" \
  --arg ranked_floor "${ranked_floor}" \
  --argjson score "${score}" \
  --argjson token_count "${token_count}" \
  --argjson depth "${depth}" \
  --argjson serial_spt "${serial_spt}" \
  --argjson mtp_spt "${mtp_spt}" \
  --argjson accepted_rate "$(jq -r '.accepted_draft_rate' "${mtp_report}")" \
  --argjson uses_head "$(jq -r '.uses_pinned_mtp_head' "${mtp_report}")" \
  --arg head_sha "$(jq -r '.head_provenance.sha256 // ""' "${mtp_report}")" \
  --argjson effective_depth "$(jq -r '.effective_mean_draft_len // 0' "${mtp_report}")" \
  --argjson residual_divergences "$(jq -r '.residual_divergence_count' "${mtp_report}")" \
  '{
    score: $score,
    passed: true,
    track_id: $track,
    metrics: {
      mode: ("qwen-mtp-" + $mode),
      oracle: "candidate-local-mtp-golden-rows",
      official_score: false,
      rankable: false,
      not_rankable_reason: "candidate-generated reference rows; official scoring disabled; ranked run is the only authority",
      ranked_decode_speedup_floor: (if $ranked_floor == "" then null else ($ranked_floor | tonumber) end),
      public_drift_tripwire_passed: true,
      decode_tokens: $token_count,
      mtp_depth: $depth,
      all_tokens_matched: true,
      uses_pinned_mtp_head: $uses_head,
      head_provenance_sha256: $head_sha,
      effective_mean_draft_len: $effective_depth,
      serial_seconds_per_token: $serial_spt,
      mtp_seconds_per_token: $mtp_spt,
      mtp_decode_speedup: $score,
      accepted_draft_rate: $accepted_rate,
      residual_divergence_count: $residual_divergences
    }
  }' > "${score_tmp}"
mv "${score_tmp}" "${score_path}"
cat "${score_path}"
