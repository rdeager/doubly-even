"""Profile the share of wall-time spent inside ``subspace_in_orbit``.

Step 0 of the Engine B (Leon §10(i) paired refinement) implementation
plan — load-bearing go/no-go gate. The plan's threshold:

    if BFS time / wall ≥ 15 % at N=22: proceed with Engine B
    else:                              re-plan

This script calls the kernel's ``enumerate_doubly_even`` directly so it
can read the extended stats tuple — the production Python entry-point at
``doubly_even.enumerate.augment.enumerate_doubly_even`` discards it.

Usage::

    uv run python scripts/bench_bfs_profile.py
    uv run python scripts/bench_bfs_profile.py --N 18,20,22
"""

from __future__ import annotations

import argparse
import math
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


def profile_one(N: int) -> None:
    cap = N // 2
    quota_vec = [gaborit_sigma(N, k) for k in range(cap + 1)]
    factorial_N = math.factorial(N)

    t0 = time.perf_counter()
    raw, stats = _kernel.enumerate_doubly_even(N, cap, quota_vec, factorial_N)
    wall_s = time.perf_counter() - t0

    # Stats is now a 15-element list (pyo3 12-tuple limit); see
    # ``enumerate::enumerate_doubly_even`` doc for the layout.
    (
        canon_calls,
        primary_hits,
        secondary_attempts,
        secondary_hits,
        is_canon_aug_calls,
        parent_eq_hits,
        weight_enum_filtered,
        bfs_calls,
        bfs_hits,
        is_canon_aug_ns,
        bfs_ns,
        _nauty_ns,
        _bucket_size_sum,
        _match_position_sum,
        _max_bucket_size,
    ) = stats

    wall_ns = int(wall_s * 1e9)
    bfs_frac = bfs_ns / wall_ns if wall_ns > 0 else 0.0
    is_canon_aug_frac = is_canon_aug_ns / wall_ns if wall_ns > 0 else 0.0
    mean_bfs_ns = bfs_ns / bfs_calls if bfs_calls > 0 else 0
    bfs_miss_frac = 1.0 - (bfs_hits / bfs_calls) if bfs_calls > 0 else 0.0

    print(f"\n=== N = {N} (cap k ≤ {cap}) ===")
    print(f"  emitted classes:                 {len(raw)}")
    print(f"  wall:                            {wall_s:.3f} s")
    print(f"  is_canonical_augmentation total: {fmt_ns(is_canon_aug_ns)}  "
          f"({100 * is_canon_aug_frac:.1f}% of wall)")
    print(f"  subspace_in_orbit  total:        {fmt_ns(bfs_ns)}  "
          f"({100 * bfs_frac:.1f}% of wall)")
    print(f"  is_canon_aug calls:              {is_canon_aug_calls:>10}")
    print(f"    parent-equality short-circuit: {parent_eq_hits:>10}  "
          f"({100 * parent_eq_hits / max(is_canon_aug_calls, 1):.1f}%)")
    print(f"    weight-enum prefilter reject:  {weight_enum_filtered:>10}  "
          f"({100 * weight_enum_filtered / max(is_canon_aug_calls, 1):.1f}%)")
    print(f"    BFS entered:                   {bfs_calls:>10}  "
          f"({100 * bfs_calls / max(is_canon_aug_calls, 1):.1f}%)")
    print(f"      BFS hit ratio:               {bfs_hits} / {bfs_calls}  "
          f"(miss rate {100 * bfs_miss_frac:.1f}%)")
    if bfs_calls:
        print(f"      mean BFS latency:            {fmt_ns(int(mean_bfs_ns))}")
    print(f"  canon nauty calls:               {canon_calls:>10}")
    print(f"  canon primary cache hits:        {primary_hits:>10}")
    print(f"  >>> BFS / wall = {100 * bfs_frac:.1f}%  "
          f"(go/no-go gate: ≥ 15% at N=22)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--N",
        default="18,20,22",
        help="Comma-separated list of N (default: 18,20,22).",
    )
    args = p.parse_args()
    Ns = [int(s.strip()) for s in args.N.split(",") if s.strip()]
    for N in Ns:
        profile_one(N)
    return 0


if __name__ == "__main__":
    sys.exit(main())
