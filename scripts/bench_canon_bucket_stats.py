"""Phase 1 measurement for the cheap-equivalence-verifier plan.

Reads the secondary-cache instrumentation stats out of the kernel's
``enumerate_doubly_even`` and prints:

- mean/total nauty wall per call (the cost budget any verifier must beat)
- bucket size distribution at attempt time (mean, max)
- mean match position within bucket on hit
- secondary-cache hit rate (consistency check against the existing 93.5%
  finding at N=22)

Usage::

    DOUBLY_EVEN_SECONDARY_CACHE_INSTRUMENTATION=1 \\
        uv run python scripts/bench_canon_bucket_stats.py --N 18,20,22

Gate 1 (from the plan): if mean nauty wall at N=22 is < 20 µs, no
Python verifier can plausibly beat it at scale. Abort.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import doubly_even_kernel as _kernel  # noqa: E402
from doubly_even.spec.mass import gaborit_sigma  # noqa: E402


def fmt_ns(ns: int) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1e9:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1e6:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1e3:.3f} us"
    return f"{ns} ns"


def measure(N: int) -> dict:
    cap = N // 2
    quota_vec = [gaborit_sigma(N, k) for k in range(cap + 1)]
    factorial_N = math.factorial(N)
    t0 = time.perf_counter()
    raw, stats, _per_k = _kernel.enumerate_doubly_even(N, cap, quota_vec, factorial_N)
    wall_s = time.perf_counter() - t0
    # Field layout: see rust/src/enumerate.rs::enumerate_doubly_even doc.
    (
        canon_calls,
        _primary_hits,
        secondary_attempts,
        secondary_hits,
        _is_canon_aug_calls,
        _parent_eq_hits,
        _weight_enum_filtered,
        _bfs_calls,
        _bfs_hits,
        _is_canon_aug_ns,
        _bfs_ns,
        nauty_ns,
        bucket_size_sum,
        match_position_sum,
        max_bucket_size,
        *_trailing,
    ) = stats
    return {
        "N": N,
        "cap": cap,
        "classes": len(raw),
        "wall_s": wall_s,
        "canon_calls": canon_calls,
        "secondary_attempts": secondary_attempts,
        "secondary_hits": secondary_hits,
        "nauty_ns": nauty_ns,
        "bucket_size_sum": bucket_size_sum,
        "match_position_sum": match_position_sum,
        "max_bucket_size": max_bucket_size,
    }


def report(m: dict) -> None:
    N = m["N"]
    canon = m["canon_calls"]
    attempts = m["secondary_attempts"]
    hits = m["secondary_hits"]
    nauty_ns = m["nauty_ns"]

    print(f"\n=== N = {N} (cap k ≤ {m['cap']}) ===")
    print(f"  emitted classes:           {m['classes']}")
    print(f"  wall:                      {m['wall_s']:.3f} s")
    print(f"  nauty calls (canon_calls): {canon}")
    if canon:
        mean_nauty = nauty_ns / canon
        print(f"  nauty total wall:          {fmt_ns(nauty_ns)}  "
              f"({100 * nauty_ns / (m['wall_s'] * 1e9):.1f}% of wall)")
        print(f"  mean nauty per call:       {fmt_ns(int(mean_nauty))}")
    if attempts == 0:
        print("  secondary-cache instrumentation OFF — "
              "set DOUBLY_EVEN_SECONDARY_CACHE_INSTRUMENTATION=1")
        return
    print(f"  secondary attempts:        {attempts}  "
          f"({100 * attempts / max(canon, 1):.1f}% of nauty calls had a non-empty bucket)")
    print(f"  secondary hits:            {hits}  "
          f"({100 * hits / max(attempts, 1):.1f}% of attempts found the new code already cached)")
    mean_bucket = m["bucket_size_sum"] / attempts
    print(f"  mean bucket size:          {mean_bucket:.2f}")
    print(f"  max bucket size:           {m['max_bucket_size']}")
    if hits:
        mean_pos = m["match_position_sum"] / hits
        print(f"  mean match position:       {mean_pos:.2f}  "
              "(0 = first entry; for a verifier this is the expected #compares "
              "before a YES is found)")
    # Gate 1: mean nauty per call ≥ 20 µs at N=22 to leave room for a verifier.
    if N == 22 and canon:
        mean_nauty = nauty_ns / canon
        gate = 20_000  # 20 µs in ns
        status = "PASS" if mean_nauty >= gate else "FAIL"
        print(
            f"  >>> Gate 1 (mean nauty ≥ 20 µs at N=22): {status}  "
            f"(mean = {fmt_ns(int(mean_nauty))})"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--N",
        default="18,20,22",
        help="Comma-separated list of N (default: 18,20,22).",
    )
    args = p.parse_args()
    if os.environ.get("DOUBLY_EVEN_SECONDARY_CACHE_INSTRUMENTATION") != "1":
        print(
            "WARNING: DOUBLY_EVEN_SECONDARY_CACHE_INSTRUMENTATION is not set "
            "to '1'. Bucket / match-position fields will be zero. "
            "Re-run as: DOUBLY_EVEN_SECONDARY_CACHE_INSTRUMENTATION=1 "
            "uv run python scripts/bench_canon_bucket_stats.py …",
            file=sys.stderr,
        )
    Ns = [int(s.strip()) for s in args.N.split(",") if s.strip()]
    for N in Ns:
        m = measure(N)
        report(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
