"""E1/E2 measurement: bucketed sparsenauty cost histogram.

Runs ``enumerate_doubly_even(N)`` with an instrumented kernel that records
per-call ``(elapsed_ns, numnodes, tctotal, maxlevel, numgenerators,
left_vertices, right_vertices)`` into a global Mutex<Vec>, drained at the
end via ``_kernel.drain_nauty_hist()``.

Bucket by ``numnodes`` (easy = 1 → no backtracking; hard = >1) and report
``mean_cost_easy``, ``mean_cost_hard``, ``easy_fraction``.

Decision rule (see research plan `any-speed-improvement-to-tender-wave`):

- G1: ``easy_fraction >= 0.98``
- G2: ``mean_cost_easy >= 40 µs`` → proceed
- G2-alt: ``mean_cost_easy < 25 µs`` → KILL

Requires the kernel to be built with ``--features parallel,nauty_hist``.

Usage::

    DOUBLY_EVEN_THREADS=16 DOUBLY_EVEN_FRONTIER_DEPTH=4 \\
        uv run python scripts/nauty_hist_experiment.py --N 22
    DOUBLY_EVEN_THREADS=20 DOUBLY_EVEN_FRONTIER_DEPTH=5 \\
        uv run python scripts/nauty_hist_experiment.py --N 24
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import doubly_even_kernel as _kernel  # noqa: E402

from doubly_even.enumerate.augment import (  # noqa: E402
    _parse_thread_env,
    enumerate_doubly_even,
)
from doubly_even.spec.mass import gaborit_sigma  # noqa: E402


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f)


def maxlevel_buckets(rows: list[tuple]) -> dict:
    """Bucket by nauty's tree depth (maxlevel) — the more general
    `1-WL + k individualizations` criterion. maxlevel == 1 means the
    root node alone sufficed (no individualization — equivalent to
    WL-discretizes-with-trivial-Aut). maxlevel <= 2 means at most one
    pivot choice was needed. Higher = deeper tree → worse.
    """
    by_level: dict[int, dict] = {}
    for row in rows:
        elapsed_ns, maxlevel = row[0], row[3]
        b = by_level.setdefault(int(maxlevel), {"n": 0, "ns": []})
        b["n"] += 1
        b["ns"].append(elapsed_ns)
    out = {}
    total = len(rows) if rows else 1
    for level in sorted(by_level):
        ns_list = by_level[level]["ns"]
        n = by_level[level]["n"]
        out[str(level)] = {
            "n": n,
            "fraction": n / total,
            "mean_ns": statistics.fmean(ns_list) if ns_list else 0.0,
            "median_ns": statistics.median(ns_list) if ns_list else 0.0,
        }
    return out


def bucket_stats(rows: list[tuple]) -> dict:
    """Bucket rows by (numgenerators == 0) easy vs (numgenerators > 0) hard.

    numgenerators == 0 means nauty found no non-trivial automorphism, so
    |Aut|=1 and 1-WL on the colored graph must reach a discrete
    partition. This is the fast-path-eligible bucket — see research plan
    `any-speed-improvement-to-tender-wave`.

    rows are tuples (elapsed_ns, numnodes, tctotal, maxlevel,
    numgenerators, left_vertices, right_vertices).
    """
    if not rows:
        return {"total": 0}

    easy_ns: list[int] = []
    hard_ns: list[int] = []
    easy_tctotal: list[int] = []
    hard_tctotal: list[int] = []
    easy_maxlevel: list[int] = []
    hard_maxlevel: list[int] = []
    easy_numnodes: list[int] = []
    hard_numnodes: list[int] = []
    hard_numgens: list[int] = []

    for row in rows:
        elapsed_ns, numnodes, tctotal, maxlevel, numgens = row[:5]
        if numgens == 0:
            easy_ns.append(elapsed_ns)
            easy_tctotal.append(tctotal)
            easy_maxlevel.append(maxlevel)
            easy_numnodes.append(numnodes)
        else:
            hard_ns.append(elapsed_ns)
            hard_tctotal.append(tctotal)
            hard_maxlevel.append(maxlevel)
            hard_numnodes.append(numnodes)
            hard_numgens.append(numgens)

    total = len(rows)
    easy_n = len(easy_ns)
    hard_n = len(hard_ns)

    def stats(values: list[int]) -> dict:
        if not values:
            return {"n": 0, "mean_ns": 0.0, "median_ns": 0.0,
                    "p10_ns": 0.0, "p90_ns": 0.0, "p99_ns": 0.0,
                    "min_ns": 0, "max_ns": 0}
        return {
            "n": len(values),
            "mean_ns": statistics.fmean(values),
            "median_ns": statistics.median(values),
            "p10_ns": percentile(values, 10),
            "p90_ns": percentile(values, 90),
            "p99_ns": percentile(values, 99),
            "min_ns": min(values),
            "max_ns": max(values),
        }

    def stats_dict(values: list[int]) -> dict:
        if not values:
            return {"n": 0, "mean": 0.0, "median": 0.0, "max": 0}
        return {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    return {
        "total": total,
        "easy_fraction": easy_n / total if total else 0.0,
        "hard_fraction": hard_n / total if total else 0.0,
        "easy": {
            "elapsed_ns": stats(easy_ns),
            "tctotal": stats_dict(easy_tctotal),
            "maxlevel": stats_dict(easy_maxlevel),
            "numnodes": stats_dict(easy_numnodes),
        },
        "hard": {
            "elapsed_ns": stats(hard_ns),
            "tctotal": stats_dict(hard_tctotal),
            "maxlevel": stats_dict(hard_maxlevel),
            "numnodes": stats_dict(hard_numnodes),
            "numgenerators": stats_dict(hard_numgens),
        },
    }


def run_one(n: int) -> dict:
    factorial_n = math.factorial(n)
    max_k = (n - (n % 4)) // 2  # k <= n/2 with doubly-even constraint
    # gaborit_sigma quota: σ(N, k) closed-form upper bound on # codes
    quota = [int(gaborit_sigma(n, k)) for k in range(max_k + 1)]
    num_threads = _parse_thread_env(os.environ.get("DOUBLY_EVEN_THREADS"))

    # Drain any leftover from prior runs.
    if hasattr(_kernel, "drain_nauty_hist"):
        _kernel.drain_nauty_hist()
    else:
        raise RuntimeError(
            "kernel was not built with --features nauty_hist; "
            "rebuild with `maturin build --release --features "
            "parallel,nauty_hist -m rust/Cargo.toml`"
        )

    t0 = time.perf_counter()
    classes = list(enumerate_doubly_even(n))
    wall_s = time.perf_counter() - t0

    rows = _kernel.drain_nauty_hist()

    return {
        "N": n,
        "num_threads": num_threads,
        "frontier_depth": int(os.environ.get("DOUBLY_EVEN_FRONTIER_DEPTH", "4")),
        "wall_s": wall_s,
        "classes": len(classes),
        "hist_call_count": len(rows),
        "buckets": bucket_stats(rows),
        "maxlevel_buckets": maxlevel_buckets(rows),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--N",
        type=str,
        default="22",
        help="Comma-separated list of N values (e.g. '22,24')",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="nauty-hist",
        help="Label embedded in the output filename",
    )
    args = parser.parse_args(argv)

    ns = [int(x) for x in args.N.split(",") if x.strip()]

    out_dir = REPO_ROOT / "scripts" / "bench-results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}-{args.label}.json"

    results = []
    for n in ns:
        print(f"\n=== N = {n} ===", flush=True)
        r = run_one(n)
        results.append(r)
        b = r["buckets"]
        print(f"  wall = {r['wall_s']:.2f}s   classes = {r['classes']}", flush=True)
        if "easy" in b:
            ef = b["easy_fraction"]
            em = b["easy"]["elapsed_ns"]["mean_ns"] / 1000.0
            emed = b["easy"]["elapsed_ns"]["median_ns"] / 1000.0
            hm = b["hard"]["elapsed_ns"]["mean_ns"] / 1000.0
            ec_tc = b["easy"]["tctotal"]["mean"]
            ec_ml = b["easy"]["maxlevel"]["mean"]
            ec_nn = b["easy"]["numnodes"]["mean"]
            print(
                f"  easy_fraction (|Aut|=1) = {ef:.4f}   "
                f"mean_easy = {em:.1f} µs   median_easy = {emed:.1f} µs   "
                f"mean_hard = {hm:.1f} µs",
                flush=True,
            )
            print(
                f"  easy: tctotal_mean = {ec_tc:.1f}   "
                f"maxlevel_mean = {ec_ml:.2f}   "
                f"numnodes_mean = {ec_nn:.2f}",
                flush=True,
            )

    payload = {
        "label": args.label,
        "timestamp_utc": ts,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
