#!/usr/bin/env bash
# Run the GCP shakedown bench: 3 configurations on N=16, 24, 26.
#
# Three runs (each writes its own JSON to scripts/bench-results/):
#   A — seq baseline:  t=1, N=16,24       — sanity floor + per-call cost
#   B — half-sub:      t=24, depth=4,  N=16,24,26
#   C — full-sub:      t=48, depth=5,  N=24,26
#
# Cap stays at 500K for all three (the value that survives N=26 OOM at 20
# workers on the 13700K; per-worker cap × threads ≈ memory ceiling).
#
# Usage:
#   scripts/gcp-bench.sh                 # label defaults to gcp-shake-<host>
#   scripts/gcp-bench.sh shakedown-48    # explicit label suffix
#
# Run from /workspace/src/ (or any repo checkout root).

set -euo pipefail

cd "$(dirname "$0")/.."

LABEL_SUFFIX="${1:-gcp-shake-$(hostname -s)}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
GIT_SHA=$(git rev-parse --short HEAD)
LOGICAL_CORES=$(nproc)
CPU_MODEL=$(awk -F: '/^model name/ {print $2; exit}' /proc/cpuinfo | sed 's/^ *//')
MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)

cat <<EOF
== GCP shakedown bench ==
  timestamp:       $TS
  label-suffix:    $LABEL_SUFFIX
  git:             $GIT_SHA
  cpu:             $CPU_MODEL
  cores:           $LOGICAL_CORES logical
  memory:          ${MEM_GB} GiB
  results dir:     scripts/bench-results/

EOF

# --- Run A: sequential baseline --------------------------------------------
# DOUBLY_EVEN_THREADS unset/1 → sequential driver. Frontier-depth/cache cap
# are ignored on the seq path. N=16 only takes ~0.1 s; N=24 ≈ 60–100 s
# depending on per-call cost (the platform-sanity floor).
echo ">>> Run A — sequential baseline (t=1) — N=16,24"
DOUBLY_EVEN_THREADS=1 \
    scripts/run-bench.sh --label "A-seq-${LABEL_SUFFIX}" --N 16,24

# --- Run B: half-subscribed parallel ---------------------------------------
# 24 threads ≈ "all P-cores oversubscribed" on a 48-vCPU box. Depth=4 matches
# the V5 N=22 sweet spot; depth=5 is preferred for N≥24 but here we hold
# depth fixed to isolate the thread-count effect against Run C.
echo ">>> Run B — half-sub (t=24, depth=4) — N=16,24,26"
DOUBLY_EVEN_THREADS=24 \
    DOUBLY_EVEN_FRONTIER_DEPTH=4 \
    DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
    scripts/run-bench.sh --label "B-t24-d4-${LABEL_SUFFIX}" --N 16,24,26

# --- Run C: fully-subscribed parallel + deeper cut -------------------------
# 48 threads = full logical-core count on c4-standard-48. Depth=5 splits
# work finer — matches the 13700K V5 finding for N≥24. N=16 dropped (its
# wall is dominated by scheduling overhead at this point).
echo ">>> Run C — full-sub (t=48, depth=5) — N=24,26"
DOUBLY_EVEN_THREADS=48 \
    DOUBLY_EVEN_FRONTIER_DEPTH=5 \
    DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
    scripts/run-bench.sh --label "C-t48-d5-${LABEL_SUFFIX}" --N 24,26

echo
echo "== Done. JSON files:"
ls -lt scripts/bench-results/*"${LABEL_SUFFIX}"*.json | head -10
cat <<'EOF'

To pull results back to your local machine (run on your local box, not here):

  gcloud compute scp --recurse \
    <vm-name>:~/doubly-even/scripts/bench-results \
    ./gcp-shake-results --zone=us-east4-a

Then paste any of the three JSON filenames + contents into chat.
EOF
