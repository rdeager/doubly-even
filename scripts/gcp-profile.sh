#!/usr/bin/env bash
# Cloud profiling day: measure parallel core utilisation at the N>=28
# frontier and run the frontier-depth A/B that the 2026-06-13 local
# session flagged as the most likely 10-20% lever.
#
# WHY this exists (read markdown/notes/cloud-profiling-wrapup-next-session-2026-06-13.md):
#   - The "d=4 best at every N" finding (V5) was measured at <=24 cores
#     and has NEVER been re-tested at 72. Finer seeds (d=5) cost more
#     seeder overhead but balance 72 workers far better — the tail
#     imbalance is the dominant cloud inefficiency, not the seeder span.
#   - The local 65-95% CFS throttle was a 20-core *cgroup quota* artifact
#     (workers + seeder-helper pool > quota). It does NOT exist on a full
#     VM with no sub-quota, so do NOT tune SEEDER_THREADS to "fix" it.
#   - Steady-state local utilisation was ~98% of the quota; the recoverable
#     headroom is the ramp + tail, surfaced by mpstat %idle over time.
#
# Each arm runs the counts-only kernel (run_counts.py — the N>=30 path,
# no streaming I/O to confound the timing) at full subscription, with
# `mpstat -P ALL` logging per-core utilisation for the whole run. The
# per-core %idle trace IS the "are all cores busy?" answer: a flat low
# %idle = healthy; a high-%idle head = ramp/seeder-feed loss; a high-%idle
# tail = load imbalance (the lever).
#
# Build requirement: kernel built `--features parallel,parallel_profiling`
# (scripts/gcp-setup.sh --features parallel,parallel_profiling). The
# profiling feature also records per-seed enqueue timings into the result
# JSON's kernel_stats for the seeder-timeline join. mpstat needs `sysstat`
# (gcp-setup.sh installs it).
#
# Usage:
#   scripts/gcp-profile.sh                 # N=28, depths 4 & 5, t=nproc
#   scripts/gcp-profile.sh 29              # N=29
#   N=28 DEPTHS="4 5" THREADS=72 scripts/gcp-profile.sh
#
# Honoured env vars (all have sensible defaults):
#   N                  code length            (default: $1 or 28)
#   DEPTHS             frontier depths to A/B (default: "4 5")
#   THREADS            worker threads         (default: nproc)
#   CACHE_CAP          canon cache cap        (default: 300000; cloud value)
#   MPSTAT_INTERVAL    seconds between mpstat samples (default: 10)
#   PROGRESS_INTERVAL  progress.json cadence  (default: 30)
#   OUT_ROOT           output root            (default: $HOME/n<N>-profile)
#
# Run from the repo root (cwd = ~/doubly-even). Long-running — launch
# under tmux/nohup so a dropped ssh session can't SIGHUP it (the local
# session-lifecycle trap that reset two runs on 2026-06-13).

set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-${1:-28}}"
DEPTHS="${DEPTHS:-4 5}"
THREADS="${THREADS:-$(nproc)}"
CACHE_CAP="${CACHE_CAP:-300000}"
MPSTAT_INTERVAL="${MPSTAT_INTERVAL:-10}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
OUT_ROOT="${OUT_ROOT:-$HOME/n${N}-profile}"

TS=$(date -u +%Y%m%dT%H%M%SZ)
GIT_SHA=$(git rev-parse --short HEAD)
CPU_MODEL=$(awk -F: '/^model name/ {print $2; exit}' /proc/cpuinfo | sed 's/^ *//')
[[ -z "$CPU_MODEL" ]] && CPU_MODEL=$(awk -F: '/^Model name/ {print $2; exit}' <(lscpu) | sed 's/^ *//')
MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)

if ! command -v mpstat >/dev/null 2>&1; then
    echo "ERROR: mpstat not found. Install sysstat (sudo apt-get install -y sysstat)." >&2
    exit 2
fi
BUILD_INFO=$(.venv/bin/python -c 'import doubly_even_kernel as k; print(k.kernel_build_info())' 2>/dev/null || echo "kernel not importable")

mkdir -p "$OUT_ROOT"
cat <<EOF
== GCP profile: utilisation + frontier-depth A/B ==
  timestamp:        $TS
  git:              $GIT_SHA
  cpu:              $CPU_MODEL
  cores (nproc):    $(nproc)   threads/arm: $THREADS
  memory:           ${MEM_GB} GiB
  build_info:       $BUILD_INFO
  N:                $N
  depths (A/B):     $DEPTHS
  canon cache cap:  $CACHE_CAP
  mpstat interval:  ${MPSTAT_INTERVAL}s     progress: ${PROGRESS_INTERVAL}s
  output root:      $OUT_ROOT

  Reading the mpstat-d<depth>.log afterwards (the all-CPU "all" row):
    flat low %idle            -> cores fully utilised (healthy)
    high %idle at the START   -> ramp / seeder-feed loss (small at N>=28)
    high %idle at the END      -> tail load imbalance (the lever; finer
                                 seeds = higher depth should shrink it)
EOF

run_arm() {
    local depth="$1"
    local arm_dir="$OUT_ROOT/d${depth}"
    local mlog="$OUT_ROOT/mpstat-d${depth}.log"
    mkdir -p "$arm_dir"
    echo
    echo ">>> ARM depth=$depth  (t=$THREADS)  -> $arm_dir"

    # Start the per-core sampler for the whole run; kill on arm exit.
    mpstat -P ALL "$MPSTAT_INTERVAL" > "$mlog" 2>&1 &
    local mpid=$!
    # shellcheck disable=SC2064
    trap "kill $mpid 2>/dev/null || true" RETURN

    local t0 t1
    t0=$(date +%s)
    DOUBLY_EVEN_THREADS="$THREADS" \
        DOUBLY_EVEN_FRONTIER_DEPTH="$depth" \
        DOUBLY_EVEN_CANON_CACHE_CAP="$CACHE_CAP" \
        .venv/bin/python scripts/run_counts.py \
            --N "$N" \
            --output-dir "$arm_dir" \
            --label "profile-N${N}-d${depth}-t${THREADS}-${GIT_SHA}" \
            --progress-interval "$PROGRESS_INTERVAL"
    t1=$(date +%s)

    kill "$mpid" 2>/dev/null || true
    wait "$mpid" 2>/dev/null || true
    trap - RETURN

    echo "    arm depth=$depth wall: $((t1 - t0)) s   (mpstat: $mlog)"
    # Quick steady-state hint: median %idle of the "all" row, dropping the
    # first 3 samples (ramp). Column count varies by mpstat version, so
    # pull %idle as the last field of "all"-CPU rows.
    awk '
        $2=="all" && $0 !~ /CPU/ {n++; if (n>3) idle[++m]=$NF}
        END {
            if (m>0) {
                asort(idle);
                printf "    steady %%idle (median of post-ramp all-CPU): %.1f%%\n", idle[int((m+1)/2)]
            }
        }' "$mlog" 2>/dev/null || true
}

for d in $DEPTHS; do
    run_arm "$d"
done

echo
echo "== Done. Per-arm results + mpstat logs under $OUT_ROOT =="
ls -la "$OUT_ROOT"/d*/*.json "$OUT_ROOT"/mpstat-*.log 2>/dev/null
cat <<EOF

Compare the arms:
  - Wall per arm is printed above; lower at d=5 => finer seeds won on this
    core count (the hypothesis). If d=4 still wins, the tail wasn't the
    bottleneck — re-check the mpstat tail %idle.
  - kernel_stats in each d<depth>/n${N}.json carries the parallel_profiling
    per-seed enqueue timings (when built with the feature) for the
    seeder-timeline / per-seed finish-spread analysis.
  - mpstat-d<depth>.log: watch the "all" row's %idle over time for the
    ramp (head) vs tail (end) shape described above.

Secondary A/B (manual, if the depth sweep leaves headroom): rerun one
depth at THREADS=$(( THREADS - 4 )) to test leaving cores for the seeder
helper pool + OS during ramp:
  THREADS=$(( THREADS - 4 )) DEPTHS=4 scripts/gcp-profile.sh $N
EOF
