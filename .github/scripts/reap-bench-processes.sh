#!/usr/bin/env bash
# Reap lingering bench-uid processes between untrusted benchmark phases.
#
# Why: nothing else kills a daemon that submitted/branch code spawned during the
# unsandboxed build (or transform) step. Such a daemon survives into later
# phases as a bench-uid process that is NOT under the runtime-worker Seatbelt
# profile, so it could read hidden material placed in the workspace during the
# gates phase, or bridge state into the timed measurement. The operator janitor
# reaps bench processes, but only at end-of-job; this runs BETWEEN phases.
#
# How: the runner account cannot signal another uid's processes, and sudo
# permits only the bench-exec / janitor entry points, so this script is invoked
# THROUGH the bench-exec bridge, i.e. as the bench uid itself. A uid can signal
# its own uid's processes, which is exactly the leaked-daemon case. It reaps
# every bench-uid process except this reaper's own tree (its ancestor chain and
# its process group), so it never kills the bench-exec wrapper that launched it.
set -uo pipefail

bench_uid="$(id -u)"
self_pid="$$"

# Never operate on a privileged/system uid: bench-exec must have dropped to the
# unprivileged ephemeral bench account (uid 560 on the ranked box). If we are
# somehow still privileged, refuse rather than signal system processes.
if [[ -z "${bench_uid}" || "${bench_uid}" -lt 500 ]]; then
  echo "reap-bench-processes: refusing to reap privileged/system uid '${bench_uid}'" >&2
  exit 1
fi

# Protect this reaper's ancestor chain (self, parent bench-exec wrapper, ...).
protected_pids=" "
walk_pid="${self_pid}"
for _ in $(seq 1 64); do
  [[ -z "${walk_pid}" || "${walk_pid}" -le 1 ]] && break
  protected_pids+="${walk_pid} "
  walk_pid="$(ps -o ppid= -p "${walk_pid}" 2>/dev/null | tr -d ' ' || true)"
done

# ...and its process group, so short-lived children (ps, awk) are spared.
self_pgid="$(ps -o pgid= -p "${self_pid}" 2>/dev/null | tr -d ' ' || true)"

reap_signal() {
  local signal="$1"
  local pid pgid
  while read -r pid pgid; do
    [[ -z "${pid}" ]] && continue
    [[ "${protected_pids}" == *" ${pid} "* ]] && continue
    [[ -n "${self_pgid}" && "${pgid}" == "${self_pgid}" ]] && continue
    kill -"${signal}" "${pid}" 2>/dev/null || true
  done < <(ps -axo pid=,pgid=,uid= | awk -v u="${bench_uid}" '$3 == u { print $1, $2 }')
}

reap_signal TERM
sleep 1
reap_signal KILL

# Post-KILL residual check: SIGKILL is unblockable, so anything still alive is
# in uninterruptible kernel sleep (or a zombie awaiting reap) -- report it for
# the audit trail rather than failing the run over an unkillable corpse.
residual=""
while read -r pid pgid; do
  [[ -z "${pid}" ]] && continue
  [[ "${protected_pids}" == *" ${pid} "* ]] && continue
  [[ -n "${self_pgid}" && "${pgid}" == "${self_pgid}" ]] && continue
  residual+="${pid} "
done < <(ps -axo pid=,pgid=,uid= | awk -v u="${bench_uid}" '$3 == u { print $1, $2 }')
if [[ -n "${residual}" ]]; then
  echo "reap-bench-processes: WARNING uid=${bench_uid} pids survived SIGKILL (kernel-stuck or zombie): ${residual}" >&2
fi
echo "reap-bench-processes: reaped stray uid=${bench_uid} processes (protected reaper tree)"
