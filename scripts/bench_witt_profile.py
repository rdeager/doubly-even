"""Profile the share of wall-time spent inside ``doubly_even_candidates_q``.

Stage 0 of the Witt-dispatch tuning plan
(`/home/dev/.claude/plans/the-last-several-sessions-scalable-bear.md`).
Decision gate at N=22:

    candidates_q_ns / wall  <  5 %  ->  close lever
    5 % – 20 %                       ->  run Stage 1 A/B test
    > 20 %                           ->  write a follow-up plan to ship

The script reads the kernel's stats vector (length 21 after Stage 0
instrumentation); fields 19 = `candidates_q_calls`, 20 = `candidates_q_ns`
(per `enumerate::enumerate_doubly_even` doc).

Per-run cap: 60 s wall. If any per-N run exceeds it, later Ns are
skipped.

Usage::

    uv run --no-sync python scripts/bench_witt_profile.py
    uv run --no-sync python scripts/bench_witt_profile.py --N 18,20,22
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


PER_RUN_CAP_S = 60.0


def fmt_ns(ns: int) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1e9:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1e6:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1e3:.3f} us"
    return f"{ns} ns"


def profile_one(N: int) -> dict:
    cap = N // 2
    quota_vec = [gaborit_sigma(N, k) for k in range(cap + 1)]
    factorial_N = math.factorial(N)

    t0 = time.perf_counter()
    raw, stats, _per_k = _kernel.enumerate_doubly_even(N, cap, quota_vec, factorial_N)
    wall_s = time.perf_counter() - t0
    wall_ns = int(wall_s * 1e9)

    if len(stats) != 22:
        raise RuntimeError(
            f"expected stats length 22 (Phase 1 rejection-rate kernel), got {len(stats)}. "
            f"Did you rebuild with `maturin develop --release` after the instrumentation?"
        )

    # Field layout per enumerate::enumerate_doubly_even doc.
    canon_calls = int(stats[0])
    is_canon_aug_ns = int(stats[9])
    bfs_ns = int(stats[10])
    nauty_ns = int(stats[11])
    candidates_q_calls = int(stats[19])
    candidates_q_ns = int(stats[20])

    other_ns = max(0, wall_ns - candidates_q_ns - nauty_ns - is_canon_aug_ns)

    def pct(ns: int) -> str:
        if wall_ns <= 0:
            return "  n/a"
        return f"{100 * ns / wall_ns:5.1f}%"

    print(f"\n=== N = {N} (cap k ≤ {cap}) ===")
    print(f"  emitted classes:           {len(raw):>10}")
    print(f"  wall:                      {wall_s:>10.3f} s")
    print(f"  ---- breakdown ----")
    print(f"  candidates_q  total:       {fmt_ns(candidates_q_ns):>12}  ({pct(candidates_q_ns)} of wall)")
    print(f"  canon_info (nauty) total:  {fmt_ns(nauty_ns):>12}  ({pct(nauty_ns)} of wall)")
    print(f"  is_canon_aug  total:       {fmt_ns(is_canon_aug_ns):>12}  ({pct(is_canon_aug_ns)} of wall)")
    print(f"    (of which BFS):          {fmt_ns(bfs_ns):>12}  ({pct(bfs_ns)} of wall)")
    print(f"  other:                     {fmt_ns(other_ns):>12}  ({pct(other_ns)} of wall)")
    print(f"  ---- counts ----")
    print(f"  canon (nauty) calls:       {canon_calls:>10}")
    print(f"  candidates_q calls:        {candidates_q_calls:>10}")
    if candidates_q_calls:
        mean_cq_ns = candidates_q_ns // candidates_q_calls
        print(f"  mean candidates_q latency: {fmt_ns(mean_cq_ns):>12}")

    return {
        "N": N,
        "wall_s": wall_s,
        "classes": len(raw),
        "candidates_q_ns": candidates_q_ns,
        "candidates_q_pct": (100 * candidates_q_ns / wall_ns) if wall_ns else 0.0,
        "nauty_pct": (100 * nauty_ns / wall_ns) if wall_ns else 0.0,
        "is_canon_aug_pct": (100 * is_canon_aug_ns / wall_ns) if wall_ns else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--N",
        default="18,20,22",
        help="Comma-separated list of N (default: 18,20,22).",
    )
    args = p.parse_args()
    Ns = [int(s.strip()) for s in args.N.split(",") if s.strip()]

    results = []
    for N in Ns:
        r = profile_one(N)
        results.append(r)
        if r["wall_s"] > PER_RUN_CAP_S:
            print(
                f"\n!! N={N} wall {r['wall_s']:.1f}s exceeded {PER_RUN_CAP_S:.0f}s "
                f"cap; skipping remaining Ns for fast iteration"
            )
            break

    print("\n=== Decision-gate summary ===")
    print(f"  {'N':>3}  {'wall':>9}  {'candidates_q':>13}  {'nauty':>7}  {'is_canon_aug':>13}")
    for r in results:
        print(
            f"  {r['N']:>3}  {r['wall_s']:>8.3f}s  "
            f"{r['candidates_q_pct']:>11.1f}%  "
            f"{r['nauty_pct']:>6.1f}%  "
            f"{r['is_canon_aug_pct']:>12.1f}%"
        )

    n22 = next((r for r in results if r["N"] == 22), None)
    if n22 is not None:
        cq = n22["candidates_q_pct"]
        if cq < 5.0:
            verdict = "CLOSE — candidates_q < 5 % of wall at N=22; lever is bounded too tight."
        elif cq <= 20.0:
            verdict = "STAGE 1 — candidates_q is 5–20 % of wall; A/B test is worth running."
        else:
            verdict = "SHIP — candidates_q > 20 % of wall; write follow-up implementation plan."
        print(f"\n>>> N=22 candidates_q = {cq:.1f}% of wall -> {verdict}")
    else:
        print("\n>>> N=22 not benched; cannot apply decision gate.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
