"""Extended collision experiment — go beyond V=min-weight.

Builds on collision_experiment.py. We've already established (2026-05-17):

  At N=22 (5118 classes):
    T1=T2=T3=T4 (weight enum + dual + col-deg + pair-deg): 3069 collisions
    T6  (V_{c,h}, V=min-weight):                            2269 collisions
    Oracle (+|Aut|):                                         678 collisions

The 3 candidate invariants explored here, refined within T6-collision buckets:

  T7  V_{c,h} on V = weight-8 codewords           (different stratum)
  T8  V_{c,h} on V = weight-12 codewords          (another stratum)
  T9  per-min-weight-codeword: distribution of HAMMING distance to OTHER
      strata (weight-8 and weight-12), not just to itself
  T10 hull dimension: dim(D ∩ D^⊥)                 (a single integer)
  T11 sorted multiset of (weight-w cardinalities ∩ each column's support)
      — a refined column profile keyed by weight strata

These are still cheap iso-invariants (linear-algebra / counting on codewords);
none of them call refinement primitives. Goal: see if any pushes the N=22
collision count below the |Aut| oracle (678) without needing |Aut|.

Usage::

    uv run python scripts/collision_experiment_v2.py --N 16,18,20,22

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


def codeword_list(code: Code) -> list[int]:
    rows, _ = code.rref_basis()
    cws = [0]
    for r in rows:
        cws = cws + [c ^ r for c in cws]
    return cws


def codewords_of_weight(cws: list[int], w: int) -> list[int]:
    return [c for c in cws if c.bit_count() == w]


def column_bitmasks(n: int, cws: list[int]) -> list[int]:
    masks = [0] * n
    for i, c in enumerate(cws):
        m = c
        while m:
            j = (m & -m).bit_length() - 1
            masks[j] |= 1 << i
            m &= m - 1
    return masks


# ---- signatures -----------------------------------------------------------


def weight_enum_sig(cws: list[int]) -> tuple[int, ...]:
    return tuple(sorted(c.bit_count() for c in cws))


def vch_sig(V: list[int], n: int) -> tuple[tuple[int, ...], ...]:
    """V_{c,h}: for each v in V, the distance distribution to W (W=V here)."""
    if not V:
        return ()
    sigs: list[tuple[int, ...]] = []
    for v in V:
        deg = [0] * (n + 1)
        for w in V:
            deg[(v ^ w).bit_count()] += 1
        sigs.append(tuple(deg))
    return tuple(sorted(sigs))


def cross_distance_sig(
    V: list[int], W: list[int], n: int
) -> tuple[tuple[int, ...], ...]:
    """For each v in V, the distance distribution from v to ALL w in W."""
    if not V or not W:
        return ()
    sigs: list[tuple[int, ...]] = []
    for v in V:
        deg = [0] * (n + 1)
        for w in W:
            deg[(v ^ w).bit_count()] += 1
        sigs.append(tuple(deg))
    return tuple(sorted(sigs))


def cross_min_to_strata_sig(
    cws: list[int], n: int, strata: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """For each min-weight cw v, joint distance-dist vector to each weight-w
    stratum. Returns sorted multiset over min-weight cws of these joint
    vectors."""
    V_min_w = min(c.bit_count() for c in cws if c) if any(cws) else None
    if V_min_w is None:
        return ()
    V_min = codewords_of_weight(cws, V_min_w)
    strata_lists = [codewords_of_weight(cws, w) for w in strata]
    sigs: list[tuple[tuple[int, ...], ...]] = []
    for v in V_min:
        per_stratum = []
        for W in strata_lists:
            deg = [0] * (n + 1)
            for w in W:
                deg[(v ^ w).bit_count()] += 1
            per_stratum.append(tuple(deg))
        sigs.append(tuple(per_stratum))
    return tuple(sorted(sigs))


def hull_dim(code: Code) -> int:
    """dim(D ∩ D^perp). Counts # of D-codewords also in D^perp."""
    cws = codeword_list(code)
    dual = code.dual()
    dual_cws = set(codeword_list(dual))
    in_hull = sum(1 for c in cws if c in dual_cws)
    # in_hull is 2^hull_dim
    return in_hull.bit_length() - 1


def column_weight_profile_sig(
    cws: list[int], n: int, weights: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """For each column j, the tuple (#codewords-of-weight-w through col j,
    for w in weights). Returns sorted multiset across columns."""
    per_col: list[tuple[int, ...]] = []
    for j in range(n):
        bit = 1 << j
        through = [c for c in cws if c & bit]
        per_col.append(tuple(sum(1 for c in through if c.bit_count() == w) for w in weights))
    return tuple(sorted(per_col))


def triple_min_weight_sig(cws: list[int], n: int) -> tuple[tuple[int, int, int], ...]:
    """For each unordered triple (a,b,c) of distinct min-weight codewords,
    record sorted (wt(a+b), wt(a+c), wt(b+c)). Returns sorted multiset."""
    if not any(cws):
        return ()
    m = min(c.bit_count() for c in cws if c)
    V = [c for c in cws if c.bit_count() == m]
    nV = len(V)
    out: list[tuple[int, int, int]] = []
    for i in range(nV):
        for j in range(i + 1, nV):
            for k in range(j + 1, nV):
                a, b, c = V[i], V[j], V[k]
                trip = tuple(sorted([(a ^ b).bit_count(), (a ^ c).bit_count(), (b ^ c).bit_count()]))
                out.append(trip)
    out.sort()
    return tuple(out)


# ---- experiment ----------------------------------------------------------


def signatures_for_code(code: Code) -> dict[str, object]:
    cws = codeword_list(code)
    n = code.n
    nz_weights = sorted({c.bit_count() for c in cws if c})
    min_w = nz_weights[0] if nz_weights else None

    V_min = codewords_of_weight(cws, min_w) if min_w is not None else []
    V_8 = codewords_of_weight(cws, 8) if 8 in nz_weights else []
    V_12 = codewords_of_weight(cws, 12) if 12 in nz_weights else []

    return {
        "T1": weight_enum_sig(cws),
        "T6_min": vch_sig(V_min, n),
        "T7_w8": vch_sig(V_8, n) if V_8 else None,
        "T8_w12": vch_sig(V_12, n) if V_12 else None,
        "T9_min_to_strata": cross_min_to_strata_sig(cws, n, (8, 12)),
        "T10_hull": hull_dim(code),
        "T11_col_wprofile": column_weight_profile_sig(cws, n, (4, 8, 12, 16)),
        "T12_triple_min": triple_min_weight_sig(cws, n),
    }


TIER_CHAIN = [
    ("T1", ("T1",)),
    ("T1+T6", ("T1", "T6_min")),
    ("T1+T6+T7", ("T1", "T6_min", "T7_w8")),
    ("T1+T6+T7+T8", ("T1", "T6_min", "T7_w8", "T8_w12")),
    ("T1+T6+T9", ("T1", "T6_min", "T9_min_to_strata")),
    ("T1+T6+T10", ("T1", "T6_min", "T10_hull")),
    ("T1+T6+T11", ("T1", "T6_min", "T11_col_wprofile")),
    ("T1+T6+T12", ("T1", "T6_min", "T12_triple_min")),
    ("ALL", ("T1", "T6_min", "T7_w8", "T8_w12", "T9_min_to_strata", "T10_hull", "T11_col_wprofile", "T12_triple_min")),
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
    per_code_us = (t_sig / max(1, len(codes))) * 1e6
    print(f"  signatures computed in {t_sig:.2f}s ({per_code_us:.0f} µs/code)", flush=True)

    ranks = [c.rank for c in codes]
    tiers_out: list[dict[str, object]] = []
    for tier_name, keys in TIER_CHAIN:
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for i, s in enumerate(sigs_per_code):
            sig = tuple(s[k] for k in keys)
            buckets[sig].append(i)
        n_distinct = len(buckets)
        colliding = [v for v in buckets.values() if len(v) > 1]
        worst = max((len(v) for v in colliding), default=1)
        n_collisions = len(codes) - n_distinct
        n_classes_in_colliding = sum(len(v) for v in colliding)
        n_unique_buckets = n_distinct - len(colliding)
        print(
            f"  {tier_name:30s}: dist={n_distinct:4d}  coll={n_collisions:4d}  "
            f"worst={worst:3d}  uniq_bucket={n_unique_buckets:4d}/{len(codes)} "
            f"({100 * n_unique_buckets / len(codes):.1f}%)",
            flush=True,
        )
        tiers_out.append(
            {
                "name": tier_name,
                "keys": list(keys),
                "distinct_signatures": n_distinct,
                "collisions": n_collisions,
                "worst_bucket": worst,
                "colliding_buckets": len(colliding),
                "unique_buckets": n_unique_buckets,
                "classes_in_colliding_buckets": n_classes_in_colliding,
            }
        )

    return {
        "N": N,
        "total_classes": len(codes),
        "enum_seconds": t_enum,
        "sig_compute_seconds": t_sig,
        "tiers": tiers_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=str, default="16,18,20")
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
        out_path = out_dir / f"{ts}-leon-collision-experiment-v2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[saved {out_path}]", flush=True)


if __name__ == "__main__":
    main()
