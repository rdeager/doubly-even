"""Leon-style invariant collision experiment (Step 2.0 of the plan).

For each N, enumerate all canonical doubly-even representatives, compute
candidate signatures at four tiers of richness, and count how many distinct
equivalence classes share the same signature at each tier.

Tiers (cumulative):

    T1  weight enumerator of C
    T2  + weight enumerator of C^perp
    T3  + column-degree multiset
    T4  + column-pair-degree multiset
    T5  + V_{c,h} signature for V = C  (only on T4-colliding buckets;
                                       skipped otherwise to keep runtime
                                       bounded)

A collision is two canonical reps with distinct RREFs and identical
signatures. Zero collisions through N = 22 means the invariant chain
uniquely identifies equivalence classes and probabilistic dedup is the
10x lever. Non-zero collisions means we need paired refinement.

Usage::

    uv run python scripts/collision_experiment.py --N 16,18,20,22

Writes JSON to ``scripts/bench-results/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402
from doubly_even.spec.codes import Code  # noqa: E402


# ----------------------------------------------------------------- codeword utils


def codeword_list(code: Code) -> list[int]:
    """Materialise all 2^k codewords of ``code`` in standard order."""
    rows, _ = code.rref_basis()
    cws = [0]
    for r in rows:
        cws = cws + [c ^ r for c in cws]
    return cws


def column_bitmasks(code: Code, cws: list[int]) -> list[int]:
    """For each column j, the bitmask over codeword indices with a 1 there.

    ``masks[j].bit_count()`` is the j-th column degree;
    ``(masks[j1] & masks[j2]).bit_count()`` is the pair degree.
    """
    n = code.n
    masks = [0] * n
    for i, c in enumerate(cws):
        m = c
        while m:
            j = (m & -m).bit_length() - 1
            masks[j] |= 1 << i
            m &= m - 1
    return masks


# ----------------------------------------------------------------- signatures


def weight_enum_sig(code: Code, cws: list[int] | None = None) -> tuple[int, ...]:
    if cws is None:
        cws = codeword_list(code)
    return tuple(sorted(c.bit_count() for c in cws))


def dual_weight_enum_sig(code: Code) -> tuple[int, ...]:
    return weight_enum_sig(code.dual())


def column_degree_sig(code: Code, masks: list[int]) -> tuple[int, ...]:
    return tuple(sorted(m.bit_count() for m in masks))


def pair_degree_sig(code: Code, masks: list[int]) -> tuple[int, ...]:
    n = code.n
    counts: list[int] = []
    for j1 in range(n):
        mj1 = masks[j1]
        for j2 in range(j1 + 1, n):
            counts.append((mj1 & masks[j2]).bit_count())
    return tuple(sorted(counts))


def v_c_h_sig(code: Code, cws: list[int] | None = None) -> tuple[tuple[int, ...], ...]:
    """Leon V_{c,h} signature, V = C: for each v in V, per-Hamming-distance
    multiplicity vector; return the sorted multiset.

    Mathematically degenerate for linear codes: by translation invariance,
    every per-codeword distance distribution equals the weight enumerator,
    so this signature is equivalent to T1. Kept here as a verification of
    that fact; the meaningful Leon variant uses a non-linear V (next).
    """
    if cws is None:
        cws = codeword_list(code)
    n = code.n
    sigs: list[tuple[int, ...]] = []
    for v in cws:
        deg = [0] * (n + 1)
        for w in cws:
            deg[(v ^ w).bit_count()] += 1
        sigs.append(tuple(deg))
    return tuple(sorted(sigs))


def min_weight_codewords(cws: list[int]) -> list[int]:
    """The min-weight nonzero codewords of C — Leon's first V recipe."""
    nz = [c for c in cws if c != 0]
    if not nz:
        return []
    m = min(c.bit_count() for c in nz)
    return [c for c in nz if c.bit_count() == m]


def v_c_h_min_weight_sig(
    code: Code, cws: list[int] | None = None
) -> tuple[tuple[int, ...], ...]:
    """Leon V_{c,h} signature on V = min-weight codewords (non-linear).

    Distance distribution from each v in V_min to all w in V_min;
    return the sorted multiset of those distributions. Non-trivial
    because V_min is not closed under XOR.
    """
    if cws is None:
        cws = codeword_list(code)
    V = min_weight_codewords(cws)
    if not V:
        return ()
    n = code.n
    sigs: list[tuple[int, ...]] = []
    for v in V:
        deg = [0] * (n + 1)
        for w in V:
            deg[(v ^ w).bit_count()] += 1
        sigs.append(tuple(deg))
    return tuple(sorted(sigs))


# ----------------------------------------------------------------- experiment


def signatures_for_code(code: Code) -> dict[str, object]:
    cws = codeword_list(code)
    masks = column_bitmasks(code, cws)
    return {
        "T1": weight_enum_sig(code, cws),
        "T1_only": (weight_enum_sig(code, cws),),
        "dual": dual_weight_enum_sig(code),
        "coldeg": column_degree_sig(code, masks),
        "pairdeg": pair_degree_sig(code, masks),
        "cws": cws,  # held for optional V_{c,h} on collisions
    }


TIER_KEYS = [
    ("T1", ("T1",)),
    ("T2", ("T1", "dual")),
    ("T3", ("T1", "dual", "coldeg")),
    ("T4", ("T1", "dual", "coldeg", "pairdeg")),
]


def run_for_N(N: int) -> dict[str, object]:
    print(f"\n=== N = {N} ===", flush=True)
    t0 = time.time()
    codes = [ec.code for ec in enumerate_doubly_even(N)]
    t_enum = time.time() - t0
    print(f"  enumerated {len(codes)} canonical reps in {t_enum:.2f}s", flush=True)

    t0 = time.time()
    sigs_per_code: list[dict[str, object]] = []
    for code in codes:
        sigs_per_code.append(signatures_for_code(code))
    t_sig = time.time() - t0
    print(f"  signatures (T1-T4) computed in {t_sig:.2f}s", flush=True)

    rrefs = [c.rref_basis()[0] for c in codes]
    ranks = [c.rank for c in codes]

    tiers_out: list[dict[str, object]] = []
    last_tier_collisions: list[list[int]] = []  # list of buckets, each a list of code indices
    for tier_name, keys in TIER_KEYS:
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for i, s in enumerate(sigs_per_code):
            sig = tuple(s[k] for k in keys)
            buckets[sig].append(i)
        n_distinct = len(buckets)
        colliding_buckets = [v for v in buckets.values() if len(v) > 1]
        worst = max((len(v) for v in colliding_buckets), default=1)
        n_collisions = len(codes) - n_distinct
        coll_by_k: Counter = Counter()
        for v in colliding_buckets:
            kcounts = Counter(ranks[i] for i in v)
            for k, cnt in kcounts.items():
                if cnt > 1:
                    coll_by_k[k] += cnt - 1
        print(
            f"  {tier_name}: distinct={n_distinct}/{len(codes)}, "
            f"collisions={n_collisions}, worst_bucket={worst}, "
            f"colliding_buckets={len(colliding_buckets)}",
            flush=True,
        )
        tiers_out.append(
            {
                "name": tier_name,
                "keys": list(keys),
                "distinct_signatures": n_distinct,
                "collisions": n_collisions,
                "worst_bucket": worst,
                "colliding_buckets": len(colliding_buckets),
                "collisions_by_k": dict(coll_by_k),
            }
        )
        last_tier_collisions = colliding_buckets

    def refine_on_colliders(
        name: str, sig_fn, buckets_in: list[list[int]]
    ) -> tuple[list[list[int]], dict]:
        if not buckets_in:
            print(f"  {name}: skipped (no colliders to refine)", flush=True)
            return buckets_in, {"name": name, "skipped": True}
        n_in_codes = sum(len(b) for b in buckets_in)
        t0 = time.time()
        out_collisions = 0
        out_worst = 1
        out_buckets: list[list[int]] = []
        out_coll_by_k: Counter = Counter()
        for bucket in buckets_in:
            sub: dict[object, list[int]] = defaultdict(list)
            for i in bucket:
                sub[sig_fn(i)].append(i)
            colliding = [v for v in sub.values() if len(v) > 1]
            if colliding:
                out_buckets.extend(colliding)
                out_collisions += sum(len(v) - 1 for v in colliding)
                out_worst = max(out_worst, max(len(v) for v in colliding))
                for v in colliding:
                    kcounts = Counter(ranks[i] for i in v)
                    for k, cnt in kcounts.items():
                        if cnt > 1:
                            out_coll_by_k[k] += cnt - 1
        t = time.time() - t0
        print(
            f"  {name}: in={n_in_codes} codes / {len(buckets_in)} buckets; "
            f"out collisions={out_collisions}, worst_bucket={out_worst}, "
            f"colliding_buckets_remaining={len(out_buckets)}, t={t:.2f}s",
            flush=True,
        )
        return out_buckets, {
            "name": name,
            "collisions": out_collisions,
            "worst_bucket": out_worst,
            "colliding_buckets": len(out_buckets),
            "collisions_by_k": dict(out_coll_by_k),
            "compute_seconds": t,
        }

    # T5: V_{c,h} with V = C (expected: tautological for linear codes).
    last_tier_collisions, t5_info = refine_on_colliders(
        "T5 (V=C, linear-degenerate)",
        lambda i: v_c_h_sig(codes[i], sigs_per_code[i]["cws"]),  # type: ignore[arg-type]
        last_tier_collisions,
    )
    tiers_out.append(t5_info)

    # T6: V_{c,h} with V = min-weight codewords (non-linear, meaningful).
    last_tier_collisions, t6_info = refine_on_colliders(
        "T6 (V=min-weight)",
        lambda i: v_c_h_min_weight_sig(codes[i], sigs_per_code[i]["cws"]),  # type: ignore[arg-type]
        last_tier_collisions,
    )
    tiers_out.append(t6_info)

    # Oracle: |Aut| order added to the chain. We won't ship this (computing
    # |Aut| IS what nauty does), but it tells us how strong |Aut| is as an
    # invariant if we had it cheaply.
    if last_tier_collisions:
        from doubly_even.canon.nauty import cached_canon_info  # local import
        last_tier_collisions, aut_info = refine_on_colliders(
            "Oracle (+|Aut|)",
            lambda i: cached_canon_info(codes[i]).aut_order,
            last_tier_collisions,
        )
        tiers_out.append(aut_info)

    return {
        "N": N,
        "total_classes": len(codes),
        "enum_seconds": t_enum,
        "sig_compute_seconds_T1_T4": t_sig,
        "tiers": tiers_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=str, default="16,18,20,22")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    Ns = [int(n) for n in args.N.split(",")]

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Ns": Ns,
        "results": [],
    }
    for N in Ns:
        all_results["results"].append(run_for_N(N))

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = HERE / "bench-results"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"{ts}-leon-collision-experiment.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[saved {out_path}]", flush=True)


if __name__ == "__main__":
    main()
