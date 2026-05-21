#!/usr/bin/env bash
# Run the GCP shakedown bench: 3 configurations on N=16, 24, 26.
#
# Self-adapts to `nproc` so it works on c4-standard-24 / -48 / -288-metal
# without edits. The three runs (each writes its own JSON):
#   A — seq baseline:  t=1                    — sanity floor + per-call cost
#   B — half-sub:      t=nproc/2,  depth=4
#   C — full-sub:      t=nproc,    depth=5
#
# On c4-24 Runs B and C collapse to the same thread count (12 vs 24) — that
# still gives a depth-4-vs-depth-5 comparison at the platform-fully-
# subscribed configuration. On c4-48 they spread into half/full subscription.
#
# Cap stays at 500K for all three (the value that survives N=26 at 20
# workers on the 13700K; per-worker cap × threads ≈ memory ceiling). On
# c4-24 with 90 GiB RAM and 24 workers the worst-case canon cache is
# ~30 GiB — safe. On c4-288-metal this cap should be lowered; that's a
# follow-up climb script, not the shakedown.
#
# Usage:
#   scripts/gcp-bench.sh                 # label defaults to gcp-shake-<host>
#   scripts/gcp-bench.sh shakedown-24    # explicit label suffix
#
# Run from /workspace/src/ (or any repo checkout root).

set -euo pipefail

cd "$(dirname "$0")/.."

LABEL_SUFFIX="${1:-gcp-shake-$(hostname -s)}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
GIT_SHA=$(git rev-parse --short HEAD)
LOGICAL_CORES=$(nproc)
HALF_CORES=$(( LOGICAL_CORES / 2 ))
if (( HALF_CORES < 1 )); then HALF_CORES=1; fi
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
  thread plan:     A=1, B=$HALF_CORES (d=4), C=$LOGICAL_CORES (d=5)

EOF

# --- Run A: sequential baseline --------------------------------------------
# DOUBLY_EVEN_THREADS=1 → sequential driver. Frontier-depth/cache cap are
# ignored on the seq path. N=16 only takes ~0.1 s; N=24 ≈ 60–100 s
# depending on per-call cost (the platform-sanity floor).
echo ">>> Run A — sequential baseline (t=1) — N=16,24"
DOUBLY_EVEN_THREADS=1 \
    scripts/run-bench.sh --label "A-seq-${LABEL_SUFFIX}" --N 16,24

# --- Run B: half-subscribed parallel ---------------------------------------
# t = nproc/2, depth=4. On c4-48 this is 24t (matches the 13700K V5 N=22
# sweet spot); on c4-24 this is 12t (under-subscribed, gives a clean
# scaling baseline against Run C's full subscription).
echo ">>> Run B — half-sub (t=$HALF_CORES, depth=4) — N=16,24,26"
DOUBLY_EVEN_THREADS=$HALF_CORES \
    DOUBLY_EVEN_FRONTIER_DEPTH=4 \
    DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
    scripts/run-bench.sh --label "B-t${HALF_CORES}-d4-${LABEL_SUFFIX}" --N 16,24,26

# --- Run C: fully-subscribed parallel + deeper cut -------------------------
# t = nproc, depth=5. Matches the 13700K V5 finding for N≥24. N=16
# dropped (wall dominated by scheduling overhead at small N).
echo ">>> Run C — full-sub (t=$LOGICAL_CORES, depth=5) — N=24,26"
DOUBLY_EVEN_THREADS=$LOGICAL_CORES \
    DOUBLY_EVEN_FRONTIER_DEPTH=5 \
    DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
    scripts/run-bench.sh --label "C-t${LOGICAL_CORES}-d5-${LABEL_SUFFIX}" --N 24,26

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
