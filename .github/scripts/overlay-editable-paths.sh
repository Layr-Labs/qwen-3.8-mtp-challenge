#!/usr/bin/env bash
# Overlay only benchmark.json editablePaths from an untrusted submission checkout.
set -euo pipefail

: "${SUBMISSION_WORKTREE:?SUBMISSION_WORKTREE is required}"

CONTRACT_PATH="${CONTRACT_PATH:-benchmark.json}"

if [[ ! -d "${SUBMISSION_WORKTREE}" ]]; then
  echo "::error::submission worktree not found at ${SUBMISSION_WORKTREE}" >&2
  exit 1
fi
if [[ ! -f "${CONTRACT_PATH}" ]]; then
  echo "::error::trusted benchmark contract missing at ${CONTRACT_PATH}" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "::error::jq is required to read editablePaths from ${CONTRACT_PATH}" >&2
  exit 1
fi

validate_contract_path() {
  local path="$1"
  if [[ -z "${path}" || "${path}" == /* || "${path}" == *\\* ]]; then
    echo "::error::invalid editable path '${path}' in ${CONTRACT_PATH}" >&2
    exit 1
  fi
  case "/${path}/" in
    *"/../"*|*"/./"*)
      echo "::error::invalid editable path '${path}' in ${CONTRACT_PATH}" >&2
      exit 1
      ;;
  esac
}

OPTIONAL_EDITABLE_PATHS=()
while IFS= read -r optional_path; do
  [[ -n "${optional_path}" ]] && OPTIONAL_EDITABLE_PATHS+=("${optional_path}")
done < <(jq -r '(.optionalEditablePaths // [])[]' "${CONTRACT_PATH}")

is_optional_editable_path() {
  local candidate="$1" optional
  for optional in ${OPTIONAL_EDITABLE_PATHS[@]+"${OPTIONAL_EDITABLE_PATHS[@]}"}; do
    [[ "${candidate}" == "${optional}" ]] && return 0
  done
  return 1
}

validate_overlay_tree() {
  local path="$1"
  if find "${path}" -type l -print -quit | grep -q .; then
    echo "::error file=${path}::overlaid editable paths must not contain symlinks" >&2
    exit 1
  fi
  if find "${path}" ! -type f ! -type d -print -quit | grep -q .; then
    echo "::error file=${path}::overlaid editable paths must contain only regular files and directories" >&2
    exit 1
  fi
  if find "${path}" -type f -links +1 -print -quit | grep -q .; then
    echo "::error file=${path}::overlaid editable paths must not contain hardlinked files" >&2
    exit 1
  fi
  if find "${path}" \( -perm -4000 -o -perm -2000 \) -print -quit | grep -q .; then
    echo "::error file=${path}::overlaid editable paths must not contain setuid or setgid files" >&2
    exit 1
  fi
}

### Stale-editable-file detection (report-only) ###############################
#
# The overlay takes the SUBMISSION's copy of every editablePath wholesale. That
# is correct for files the participant actually edited, but it also carries
# over their copy of editable files they never touched -- so an operator change
# to an editable file made AFTER the candidate branched is silently replaced by
# the pre-change content. Two consequences, both observed in production:
#   1. This ranked run builds and measures the older file, not trusted main's.
#   2. On promotion the older content lands back on main, reverting the
#      operator edit (a #791-style guidance banner was lost exactly this way).
# enforce-modifiable-surface.sh cannot catch it: its content rule only checks
# NON-editable paths against trusted main.
#
# Report-only on purpose. Failing here would reject a legitimate submission for
# something the participant did not do, and dropping the file from the overlay
# would change what the ranked run measures -- both are policy calls, not this
# script's. Skipped silently when TRUSTED_MAIN_SHA is unset or the merge base
# cannot be resolved.
TRUSTED_MAIN_SHA="${TRUSTED_MAIN_SHA:-}"

submission_git() {
  local hardened="${GITHUB_WORKSPACE:-.}/.github/scripts/hardened-git.sh"
  if [[ -x "${hardened}" ]]; then
    "${hardened}" -C "${SUBMISSION_WORKTREE}" "$@"
  else
    git -C "${SUBMISSION_WORKTREE}" "$@"
  fi
}

path_is_editable() {
  local candidate="$1" allowed
  while IFS= read -r allowed; do
    [[ -n "${allowed}" ]] || continue
    if [[ "${candidate}" == "${allowed}" || "${candidate}" == "${allowed}/"* ]]; then
      return 0
    fi
  done < <(jq -r '.editablePaths[]' "${CONTRACT_PATH}")
  return 1
}

report_stale_editable_overlays() {
  [[ -n "${TRUSTED_MAIN_SHA}" ]] || return 0
  local merge_base stale=0 path
  merge_base="$(submission_git merge-base HEAD "${TRUSTED_MAIN_SHA}" 2>/dev/null || true)"
  if [[ -z "${merge_base}" ]]; then
    echo "::warning::could not resolve a merge base between the submission and trusted main ${TRUSTED_MAIN_SHA}; skipping the stale-editable-file check" >&2
    return 0
  fi
  # Candidate already sits on the trusted tip: nothing can be stale.
  [[ "${merge_base}" != "${TRUSTED_MAIN_SHA}" ]] || return 0

  while IFS= read -r path; do
    [[ -n "${path}" ]] || continue
    path_is_editable "${path}" || continue
    # Untouched by the submission since it branched -> the overlay is about to
    # replace trusted main's newer content with the pre-branch content.
    if submission_git diff --quiet "${merge_base}" HEAD -- "${path}" 2>/dev/null; then
      echo "::warning file=${path}::${path} changed on trusted main after this submission branched, and the submission still carries the pre-change content; the overlay will use the OLDER file (and promoting this submission would revert the change on main)" >&2
      stale=$((stale + 1))
    fi
  done < <(submission_git diff --name-only "${merge_base}" "${TRUSTED_MAIN_SHA}" 2>/dev/null || true)

  if [[ "${stale}" -gt 0 ]]; then
    echo "::warning::${stale} editable file(s) are overlaid at their pre-branch content because trusted main moved after this submission branched; re-sync the submission with current main to pick the changes up" >&2
  fi
  echo "benchmark: stale-editable-file check complete (merge_base=${merge_base} stale=${stale})"
}

report_stale_editable_overlays

while IFS= read -r editable_path; do
  validate_contract_path "${editable_path}"

  source_path="${SUBMISSION_WORKTREE}/${editable_path}"
  target_path="${editable_path}"

  if [[ ! -e "${source_path}" ]]; then
    # OPTIONAL editable paths may legitimately be absent from a submission, and
    # absence means "keep the trusted checkout's copy" -- never "delete it".
    #
    # WHY THIS EXISTS. Submissions arrive as archives with REPLACE semantics
    # over editablePaths: whatever the archive does not carry is gone. That is
    # correct for source, where a missing file means a stale clone and the loud
    # refusal below is the right answer. It is wrong for the MTP head
    # declaration, which is a small opt-in file most archives will simply not
    # contain -- and the contract says an ABSENT mtp-head.manifest.json means
    # "use the organizer-pinned head". Failing the overlay would turn the
    # default case into a refusal.
    #
    # The set is read from the TRUSTED contract (this script already reads
    # CONTRACT_PATH from the trusted checkout, never the submission worktree),
    # so a submission cannot make its own missing source files optional.
    if is_optional_editable_path "${editable_path}"; then
      echo "benchmark: optional editable path ${editable_path} absent from the submission; keeping the trusted copy (the contract default)"
      continue
    fi
    echo "::error file=${editable_path}::submitted editable path is missing" >&2
    exit 1
  fi
  if find "${source_path}" -type l -print -quit | grep -q .; then
    echo "::error file=${editable_path}::submitted editable paths must not contain symlinks" >&2
    exit 1
  fi

  rm -rf "${target_path}"
  mkdir -p "$(dirname "${target_path}")"
  if [[ -d "${source_path}" ]]; then
    mkdir -p "${target_path}"
    (cd "${source_path}" && tar -cf - .) | (cd "${target_path}" && tar -xf -)
  else
    cp "${source_path}" "${target_path}"
  fi
  validate_overlay_tree "${target_path}"

  echo "benchmark: overlaid editable path ${editable_path}"
done < <(jq -r '.editablePaths[]' "${CONTRACT_PATH}")

echo "benchmark: trusted harness retained; submitted editable paths overlaid"
