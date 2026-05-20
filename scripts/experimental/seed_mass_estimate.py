"""Phase-0 measurement for gluing-based mass seeding.

For each ``(N, k)``, compute the upper bound on the fraction of
``σ(N, k)`` (Gaborit closed form) that is reachable by enumerating
direct-sum gluings ``C1 ⊕ C2`` where ``C1 ∈ E[n1, k1]``, ``C2 ∈
E[n2, k2]``, ``n1 + n2 = N``, ``k1 + k2 = k``.

A direct sum is doubly-even iff both summands are. The construction
yields one equivalence class per orbit of (ordered pair, decomposition)
under (i) the block-swap symmetry when components are equivalent and
(ii) cross-decomposition equivalences when a summand itself decomposes.
We dedupe by canonicalising each constructed direct sum and tracking
distinct ``canonical_form(C1 ⊕ C2)`` values.

The reported "seed mass" is ``Σ N!/|Aut(D)|`` over distinct canonical
forms. The reported ratio is ``seed_mass / σ(N, k)``.

Decision gate (from
``/home/dev/.claude/plans/we-have-run-out-curious-bentley.md``):

  - All (N, k) on N ∈ {18, 20, 22} have ratio < 5 %: close the lever.
  - Some (N, k) ratio > 10 %: proceed to Phase 1 implementation.

Usage::

    uv run python scripts/seed_mass_estimate.py --N 18,20,22

Writes JSON to ``scripts/bench-results/seed-mass-<timestamp>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.canon.nauty import canon_info, canonical_form  # noqa: E402
from doubly_even.enumerate.augment import (  # noqa: E402
    enumerate_doubly_even,
    enumerate_doubly_even_at,
)
from doubly_even.spec.codes import Code  # noqa: E402
from doubly_even.spec.mass import gaborit_sigma  # noqa: E402


def direct_sum(C1: Code, C2: Code) -> Code:
    """Construct the direct sum ``C1 ⊕ C2``.

    Codeword bits of ``C2`` are shifted left by ``C1.n`` so that the
    block structure is ``(C1_block, C2_block)`` reading left to right.
    """
    n1 = C1.n
    n2 = C2.n
    basis = tuple(C1.basis) + tuple(b << n1 for b in C2.basis)
    return Code(n=n1 + n2, basis=basis)


def enum_table(n_max: int) -> dict[tuple[int, int], list[Code]]:
    """Build ``{(n, k): [canonical reps]}`` for ``n ≤ n_max``.

    The codes returned are exactly the canonical reps yielded by
    ``enumerate_doubly_even`` (one per equivalence class).
    """
    table: dict[tuple[int, int], list[Code]] = {}
    for n in range(0, n_max + 1):
        # n = 0: only the empty code; trivially doubly-even and rank 0.
        if n == 0:
            table[(0, 0)] = [Code(n=0, basis=())]
            continue
        # n >= 1: pull from enumerate_doubly_even. (n, 0) is included
        # because the rank-0 code is the all-zero code and is doubly-even.
        bucket: dict[int, list[Code]] = {}
        for ec in enumerate_doubly_even(n):
            bucket.setdefault(ec.code.rank, []).append(ec.code)
        # Ensure (n, 0) is present even though enumerate_doubly_even may
        # not yield it directly (it yields starting from k=0).
        for k, codes in bucket.items():
            table[(n, k)] = codes
        if (n, 0) not in table:
            table[(n, 0)] = [Code(n=n, basis=())]
    return table


def seed_for_Nk(
    N: int,
    k: int,
    table: dict[tuple[int, int], list[Code]],
    min_n1: int = 1,
) -> dict[str, object]:
    """Compute the seed mass at ``(N, k)`` from direct-sum gluings.

    ``min_n1`` skips the trivial decomposition ``(0, N)`` which is the
    target enumeration itself.
    """
    seen_canons: set[Code] = set()
    seed_mass = 0
    factorial_N = math.factorial(N)
    n_pairs_constructed = 0
    n_canon_calls = 0

    for n1 in range(min_n1, N // 2 + 1):
        n2 = N - n1
        # Iterate (k1, k2) with k1 + k2 = k.
        for k1 in range(0, k + 1):
            k2 = k - k1
            codes_1 = table.get((n1, k1))
            codes_2 = table.get((n2, k2))
            if not codes_1 or not codes_2:
                continue
            # When n1 == n2, avoid (C1, C2) and (C2, C1) double-counting
            # via an index restriction when also k1 == k2 (same bucket);
            # otherwise the two buckets are distinct so all pairs are
            # natural.
            for i, C1 in enumerate(codes_1):
                start = i if (n1 == n2 and k1 == k2) else 0
                for C2 in codes_2[start:]:
                    D = direct_sum(C1, C2)
                    canon = canonical_form(D)
                    n_canon_calls += 1
                    n_pairs_constructed += 1
                    if canon in seen_canons:
                        continue
                    seen_canons.add(canon)
                    info = canon_info(D)
                    seed_mass += factorial_N // info.aut_order

    sigma = gaborit_sigma(N, k)
    ratio = seed_mass / sigma if sigma > 0 else 0.0
    return {
        "N": N,
        "k": k,
        "sigma": sigma,
        "seed_mass": seed_mass,
        "ratio": ratio,
        "seed_classes": len(seen_canons),
        "pairs_constructed": n_pairs_constructed,
        "canon_calls": n_canon_calls,
    }


def run_for_N(
    N: int,
    table: dict[tuple[int, int], list[Code]],
) -> dict[str, object]:
    print(f"\n=== N = {N} ===", flush=True)
    rows: list[dict[str, object]] = []
    for k in range(0, N // 2 + 1):
        t0 = time.time()
        row = seed_for_Nk(N, k, table)
        elapsed = time.time() - t0
        row["seconds"] = elapsed
        rows.append(row)
        pct = 100.0 * row["ratio"]
        print(
            f"  k={k:>2}: {row['seed_classes']:>5} seed classes, "
            f"mass={row['seed_mass']}/{row['sigma']} = {pct:6.2f}%  "
            f"(pairs={row['pairs_constructed']:>7}, t={elapsed:.2f}s)",
            flush=True,
        )
    # Aggregate "best ratio across k" — the gate metric per the plan.
    max_ratio = max((r["ratio"] for r in rows), default=0.0)
    print(f"  max ratio over k: {100*max_ratio:.2f}%", flush=True)
    return {"N": N, "rows": rows, "max_ratio": max_ratio}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=str, default="18,20,22")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    Ns = [int(n) for n in args.N.split(",")]

    # Build enumeration table up to max N - 1 (we need both summands
    # strictly smaller). With min_n1=1, smallest summand is 1 and
    # largest is N-1.
    n_max = max(Ns) - 1
    print(f"Building enum table for n ≤ {n_max} …", flush=True)
    t0 = time.time()
    table = enum_table(n_max)
    t_table = time.time() - t0
    sizes = {nk: len(v) for nk, v in sorted(table.items())}
    print(f"  built in {t_table:.2f}s; non-empty cells:", flush=True)
    by_n: dict[int, list[tuple[int, int]]] = {}
    for (n, k), count in sizes.items():
        by_n.setdefault(n, []).append((k, count))
    for n in sorted(by_n):
        print(f"    n={n:>2}: " + ", ".join(f"k={k}:{c}" for k, c in by_n[n]),
              flush=True)

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Ns": Ns,
        "enum_table_seconds": t_table,
        "enum_table_sizes": {f"{n},{k}": c for (n, k), c in sizes.items()},
        "results": [],
    }
    for N in Ns:
        all_results["results"].append(run_for_N(N, table))

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = HERE / "bench-results"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"seed-mass-{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[saved {out_path}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
