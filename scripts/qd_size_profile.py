"""Profile |C_low| distribution and Q_D bail rate across N values.

Runs `enumerate_doubly_even(N)` with the nauty_hist-instrumented kernel,
drains the per-call records, and bucketises by `(rank, qd_path)`. Per
(N, k) bucket we report:

- count: total sparsenauty calls at this rank
- qd_share: fraction that went through `canon_info_qd_native` (vs the
  full-bipartite fallback in `canon_info_native`)
- c_low_p50, c_low_p90, c_low_p99: percentiles of |C_low| over the
  Q_D-path subset (informs whether |C_low| is staying small vs creeping
  toward the 2^(k-1) bail threshold)
- bail_threshold = 2^(k-1)
- bail_headroom_p50 = (bail_threshold - c_low_p50) / bail_threshold —
  positive means we're under the threshold with room to spare, negative
  is impossible (the builder would have bailed)
- median per-call elapsed_ns on Q_D path vs fallback path (the gap
  quantifies how much the reduction is saving)

Requires the kernel built with `--features nauty_hist`. For N>=24 also
build with `--features parallel` and set DOUBLY_EVEN_THREADS.

Usage:

    maturin build --release --features nauty_hist,parallel -m rust/Cargo.toml
    uv pip install rust/target/wheels/doubly_even_kernel-*.whl --force-reinstall

    uv run python scripts/qd_size_profile.py --N 22
    DOUBLY_EVEN_THREADS=16 DOUBLY_EVEN_FRONTIER_DEPTH=4 \\
        uv run python scripts/qd_size_profile.py --N 22,24
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import doubly_even_kernel as _kernel  # noqa: E402

from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f)


def profile_n(n: int) -> dict:
    if not hasattr(_kernel, "drain_nauty_hist"):
        raise RuntimeError(
            "kernel was not built with --features nauty_hist; rebuild and "
            "reinstall the wheel."
        )
    _kernel.drain_nauty_hist()  # clear any prior records

    t0 = time.perf_counter()
    class_count = 0
    for _ec in enumerate_doubly_even(n):
        class_count += 1
    elapsed_s = time.perf_counter() - t0

    rows = _kernel.drain_nauty_hist()

    # Bucket by (rank, qd_path).
    by_rank: dict[int, dict] = {}
    for elapsed_ns, _numnodes, _tctotal, _maxlevel, _numgens, l_v, _r_v, rank, qd in rows:
        bucket = by_rank.setdefault(int(rank), {
            "qd_count": 0,
            "fallback_count": 0,
            "qd_left_vertices": [],
            "qd_elapsed_ns": [],
            "fallback_elapsed_ns": [],
        })
        if qd:
            bucket["qd_count"] += 1
            bucket["qd_left_vertices"].append(int(l_v))
            bucket["qd_elapsed_ns"].append(int(elapsed_ns))
        else:
            bucket["fallback_count"] += 1
            bucket["fallback_elapsed_ns"].append(int(elapsed_ns))

    # Distill to per-rank stats.
    per_rank: dict[str, dict] = {}
    for rank in sorted(by_rank):
        b = by_rank[rank]
        total = b["qd_count"] + b["fallback_count"]
        bail_threshold = 1 << max(rank - 1, 0)
        c_low_sorted = b["qd_left_vertices"]
        per_rank[str(rank)] = {
            "rank": rank,
            "total_calls": total,
            "qd_count": b["qd_count"],
            "fallback_count": b["fallback_count"],
            "qd_share": b["qd_count"] / total if total else 0.0,
            "bail_threshold": bail_threshold,
            "two_to_k": 1 << rank,
            "c_low_p50": percentile(c_low_sorted, 50),
            "c_low_p90": percentile(c_low_sorted, 90),
            "c_low_p99": percentile(c_low_sorted, 99),
            "c_low_max": max(c_low_sorted) if c_low_sorted else 0,
            "qd_elapsed_ns_median": statistics.median(b["qd_elapsed_ns"]) if b["qd_elapsed_ns"] else 0,
            "qd_elapsed_ns_mean": statistics.fmean(b["qd_elapsed_ns"]) if b["qd_elapsed_ns"] else 0,
            "fallback_elapsed_ns_median": statistics.median(b["fallback_elapsed_ns"]) if b["fallback_elapsed_ns"] else 0,
            "fallback_elapsed_ns_mean": statistics.fmean(b["fallback_elapsed_ns"]) if b["fallback_elapsed_ns"] else 0,
        }

    total_calls = sum(b["qd_count"] + b["fallback_count"] for b in by_rank.values())
    total_qd = sum(b["qd_count"] for b in by_rank.values())
    total_fallback = sum(b["fallback_count"] for b in by_rank.values())

    return {
        "N": n,
        "classes": class_count,
        "elapsed_s": elapsed_s,
        "total_calls": total_calls,
        "overall_qd_share": total_qd / total_calls if total_calls else 0.0,
        "overall_fallback_share": total_fallback / total_calls if total_calls else 0.0,
        "per_rank": per_rank,
    }


def fmt_row(rank_stats: dict) -> str:
    s = rank_stats
    return (
        f"  k={s['rank']:>2}  calls={s['total_calls']:>8}  "
        f"qd_share={s['qd_share']*100:5.1f}%  "
        f"|C_low| p50/p90/p99/max="
        f"{s['c_low_p50']:>5.0f}/{s['c_low_p90']:>5.0f}/{s['c_low_p99']:>5.0f}/{s['c_low_max']:>5}  "
        f"bail_thr=2^(k-1)={s['bail_threshold']:>5}  "
        f"2^k={s['two_to_k']:>5}  "
        f"qd_us={s['qd_elapsed_ns_median']/1e3:6.1f}  "
        f"fb_us={s['fallback_elapsed_ns_median']/1e3:6.1f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", required=True, help="comma-separated list of N values")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    ns = [int(x) for x in args.N.split(",")]
    results = []
    for n in ns:
        print(f"\n=== N={n} ===", flush=True)
        r = profile_n(n)
        print(
            f"classes={r['classes']}  elapsed={r['elapsed_s']:.2f}s  "
            f"calls={r['total_calls']}  "
            f"qd_share={r['overall_qd_share']*100:.1f}%  "
            f"fallback_share={r['overall_fallback_share']*100:.1f}%",
            flush=True,
        )
        for rank_key in sorted(r["per_rank"], key=int):
            print(fmt_row(r["per_rank"][rank_key]), flush=True)
        results.append(r)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
