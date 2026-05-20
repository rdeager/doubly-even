#!/usr/bin/env bash
# Run scripts/bench.py via .venv/bin/python (bypasses `uv run`, which
# would resync the venv and drop the locally-installed kernel wheel).
#
# Sets PYTHONUNBUFFERED=1 so per-N output streams as each N completes.
# All other env vars (DOUBLY_EVEN_THREADS, DOUBLY_EVEN_FRONTIER_DEPTH,
# DOUBLY_EVEN_CANON_CACHE_CAP) pass through.
#
# Usage:
#   scripts/run-bench.sh --label foo --N 20,22,24
#   DOUBLY_EVEN_THREADS=22 scripts/run-bench.sh --label parallel --N 22
#
# Run from /workspace/src/.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
exec .venv/bin/python scripts/bench.py "$@"
