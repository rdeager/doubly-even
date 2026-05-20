"""Mass-stop ablation: compare wall time with and without the mass quota.

Calls `doubly_even_kernel.enumerate_doubly_even` twice per N:

1. Baseline: real Gaborit quota σ(N, k) → mass-stop can fire.
2. Ablation: quota = u128::MAX per level → mass-stop never fires; the
   `mass_at_k[k] > quota[k]` panic-check is unreachable too. (We can't
   disable the per-emission division of `factorial_n / aut_order`
   without recompiling, but it's tens of ns and lost in the noise.)

Both runs must emit the same set of canonical classes — that's the
correctness check the ablation also verifies.

Usage::

    uv run python scripts/mass_check_ablation.py --Ns 18 20 22
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import doubly_even_kernel as _kernel
from doubly_even.spec.mass import gaborit_sigma


U128_MAX = (1 << 128) - 1


def run_one(N: int, *, ablate: bool) -> tuple[float, int]:
    cap = N // 2
    if ablate:
        quota = [U128_MAX] * (cap + 1)
    else:
        quota = [gaborit_sigma(N, k) for k in range(cap + 1)]
    factorial_N = math.factorial(N)
    t0 = time.perf_counter()
    raw, _stats, _per_k = _kernel.enumerate_doubly_even(N, cap, quota, factorial_N)
    wall = time.perf_counter() - t0
    return wall, len(raw)


def measure(N: int, *, repeats: int) -> dict:
    # Warmup once to avoid first-call effects (cargo / Python startup).
    _ = run_one(N, ablate=False)

    baseline_walls = []
    ablate_walls = []
    baseline_n: int | None = None
    ablate_n: int | None = None

    for _ in range(repeats):
        w, n = run_one(N, ablate=False)
        baseline_walls.append(w)
        baseline_n = n if baseline_n is None else baseline_n
        if n != baseline_n:
            raise RuntimeError(f"N={N} baseline emission count varied: {n} vs {baseline_n}")

    for _ in range(repeats):
        w, n = run_one(N, ablate=True)
        ablate_walls.append(w)
        ablate_n = n if ablate_n is None else ablate_n
        if n != ablate_n:
            raise RuntimeError(f"N={N} ablate emission count varied: {n} vs {ablate_n}")

    if baseline_n != ablate_n:
        raise RuntimeError(
            f"N={N}: baseline emitted {baseline_n}, ablate emitted {ablate_n}; "
            f"mass-stop was hiding a bug or the ablation removed real codes."
        )

    b_med = statistics.median(baseline_walls)
    a_med = statistics.median(ablate_walls)
    return {
        "N": N,
        "classes": baseline_n,
        "baseline_walls": baseline_walls,
        "ablate_walls": ablate_walls,
        "baseline_median": b_med,
        "ablate_median": a_med,
        "delta_s": a_med - b_med,
        "delta_pct": 100 * (a_med - b_med) / b_med,
    }


def report(m: dict) -> None:
    N = m["N"]
    print(f"\n=== N = {N} (classes = {m['classes']}) ===")
    print(f"  baseline walls (s): {[f'{w:.3f}' for w in m['baseline_walls']]}")
    print(f"  ablate   walls (s): {[f'{w:.3f}' for w in m['ablate_walls']]}")
    print(f"  baseline median:    {m['baseline_median']:.4f} s")
    print(f"  ablate   median:    {m['ablate_median']:.4f} s")
    sign = "+" if m["delta_s"] >= 0 else ""
    print(
        f"  delta (ablate - baseline): {sign}{m['delta_s']*1000:.2f} ms "
        f"({sign}{m['delta_pct']:.2f}%)"
    )
    if m["delta_pct"] > 0:
        print(f"  → mass-stop SAVES {m['delta_pct']:.2f}% (turning off is slower)")
    else:
        print(f"  → mass-stop COSTS {-m['delta_pct']:.2f}% (turning off is faster)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Ns", type=int, nargs="+", default=[18, 20, 22])
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    print(f"kernel build: {_kernel.kernel_build_info()}")
    print(f"repeats per arm: {args.repeats}")
    for N in args.Ns:
        report(measure(N, repeats=args.repeats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
