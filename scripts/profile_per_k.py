"""Per-k wall-time breakdown of `enumerate_doubly_even(N)`.

Runs `enumerate(N, max_k=k)` for each k in 0..=N//2 and reports the
cumulative wall, plus the marginal time attributable to including k
(``cumulative[k] - cumulative[k-1]``). The marginal includes BOTH the
candidate-generation + parent-test work done at level k-1 to discover
k-children AND the canon-info work on emitted level-k codes — it's the
"total extra cost of letting the recursion descend to k" rather than
"time spent emitting at k".

Cold cache between rows so the timings compare cleanly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.canon.nauty import canon_info_cache_clear  # noqa: E402
from doubly_even.enumerate.augment import (  # noqa: E402
    enumerate_doubly_even,
    weight_enum_cache_clear,
)
from doubly_even.spec.codes import dual_cache_clear, rref_cache_clear  # noqa: E402


def run(N: int, max_k: int) -> tuple[float, int, dict[int, int]]:
    canon_info_cache_clear()
    rref_cache_clear()
    dual_cache_clear()
    weight_enum_cache_clear()
    per_k_count: dict[int, int] = {}
    t0 = time.perf_counter()
    total = 0
    for ec in enumerate_doubly_even(N, max_k=max_k):
        per_k_count[ec.code.rank] = per_k_count.get(ec.code.rank, 0) + 1
        total += 1
    dt = time.perf_counter() - t0
    return dt, total, per_k_count


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, required=True)
    args = p.parse_args()

    N = args.N
    cap = N // 2

    print(f"N = {N}, cap = N//2 = {cap}")
    print(f"{'max_k':>6} {'wall s':>10} {'classes':>9} {'marginal s':>12} {'classes_at_k':>14}")
    prev = 0.0
    for k in range(cap + 1):
        wall, total, per_k = run(N, max_k=k)
        marginal = wall - prev
        cnt = per_k.get(k, 0)
        print(
            f"{k:>6} {wall:>10.3f} {total:>9} {marginal:>12.3f} {cnt:>14}"
        )
        prev = wall
    return 0


if __name__ == "__main__":
    sys.exit(main())
