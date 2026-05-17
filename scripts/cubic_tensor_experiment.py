"""Phase 2: cubic-tensor / column-triple-degree prefilter experiment.

Extends `scripts/collision_experiment.py` with a new tier:

    T_triple — for each unordered column triple (a, b, c), count the
               number of codewords with a 1 in all three columns.
               Take the sorted multiset of these counts.

This is a 3rd-order column-permutation invariant. Cost: O(C(N, 3))
bitwise-AND-of-three + popcount per code; at N=22 that's 1540 ops.

Background. For doubly-even self-orthogonal codes the 1st-order
invariant `wt(g_i) ≡ 0 (mod 4)` and the 2nd-order
`⟨g_i, g_j⟩ ≡ 0 (mod 2)` are both forced. The triple intersection
T_{abc} (in column form) is NOT determined by lower-order invariants
— `wt(g_i + g_j + g_k) ≡ 0 (mod 4)` reduces to a constraint on
pairwise inner products that is automatically satisfied. So T_triple
carries genuinely new information.

Decision gate (from plan
`for-complete-enumeration-of-proud-meerkat.md` Phase 2):

  - Residual collisions < 100 (vs T1's baseline of 678 with |Aut|
    oracle at N=22) AND prototype cost < 30 µs/code: port to Rust.
  - 100–400 collisions AND cost < 20 µs: borderline; revisit.
  - ≥ 400 collisions OR cost ≥ 50 µs: close.

Usage::

    uv run python scripts/cubic_tensor_experiment.py --N 20,22

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
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402
from doubly_even.spec.codes import Code  # noqa: E402


# ----------------------------------------------------------------- codeword utils


def codeword_list(code: Code) -> list[int]:
    """All 2^k codewords of `code` in standard order."""
    rows, _ = code.rref_basis()
    cws = [0]
    for r in rows:
        cws = cws + [c ^ r for c in cws]
    return cws


def column_bitmasks(code: Code, cws: list[int]) -> list[int]:
    """Per-column bitmask over codeword indices.

    `masks[j] & masks[k]` counts codewords with 1s in both columns.
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


def weight_enum_sig(code: Code, cws: list[int]) -> tuple[int, ...]:
    return tuple(sorted(c.bit_count() for c in cws))


def column_degree_sig(masks: list[int]) -> tuple[int, ...]:
    return tuple(sorted(m.bit_count() for m in masks))


def pair_degree_sig(masks: list[int]) -> tuple[int, ...]:
    n = len(masks)
    counts: list[int] = []
    for j1 in range(n):
        mj1 = masks[j1]
        for j2 in range(j1 + 1, n):
            counts.append((mj1 & masks[j2]).bit_count())
    return tuple(sorted(counts))


def triple_degree_sig(masks: list[int]) -> tuple[int, ...]:
    """The 3rd-order column-permutation invariant.

    For each unordered triple (a, b, c), count codewords with 1s in
    all three. Sorted multiset. C(N, 3) entries.
    """
    n = len(masks)
    counts: list[int] = []
    for a in range(n):
        ma = masks[a]
        for b in range(a + 1, n):
            mab = ma & masks[b]
            if mab == 0:
                # Zero shortcut: any (a, b, c) with empty mab gives 0.
                # Accumulate (n - b - 1) zeros and skip the inner loop.
                counts.extend([0] * (n - b - 1))
                continue
            for c in range(b + 1, n):
                counts.append((mab & masks[c]).bit_count())
    return tuple(sorted(counts))


def min_weight_codewords(cws: list[int]) -> list[int]:
    nz = [c for c in cws if c != 0]
    if not nz:
        return []
    m = min(c.bit_count() for c in nz)
    return [c for c in nz if c.bit_count() == m]


def v_c_h_min_weight_sig(
    cws: list[int], n: int
) -> tuple[tuple[int, ...], ...]:
    """Leon V_{c,h} on V = min-weight codewords (non-linear, meaningful).

    Same as `collision_experiment.v_c_h_min_weight_sig`, reproduced here
    so this script doesn't import from the prior experiment module.
    """
    V = min_weight_codewords(cws)
    if not V:
        return ()
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
        "coldeg": column_degree_sig(masks),
        "pairdeg": pair_degree_sig(masks),
        "tripdeg": triple_degree_sig(masks),
        "cws": cws,
    }


# Tiers, cumulative. Each tier is keyed by a tuple of feature names
# from `signatures_for_code`'s output dict.
TIER_KEYS = [
    ("T1", ("T1",)),
    ("T1+col", ("T1", "coldeg")),
    ("T1+col+pair", ("T1", "coldeg", "pairdeg")),
    ("T1+col+pair+trip", ("T1", "coldeg", "pairdeg", "tripdeg")),
    # T_triple alone (without pair) so we can see the marginal effect of pair.
    ("T1+col+trip", ("T1", "coldeg", "tripdeg")),
]


def bucket_collisions(sigs_per_code, keys, n_codes):
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(sigs_per_code):
        sig = tuple(s[k] for k in keys)
        buckets[sig].append(i)
    distinct = len(buckets)
    colliding = [v for v in buckets.values() if len(v) > 1]
    worst = max((len(v) for v in colliding), default=1)
    collisions = n_codes - distinct
    return {
        "buckets": buckets,
        "distinct": distinct,
        "colliding": colliding,
        "worst": worst,
        "collisions": collisions,
    }


def refine_on_colliders(
    name: str,
    sig_fn,
    buckets_in: list[list[int]],
    ranks: list[int],
):
    if not buckets_in:
        return buckets_in, {
            "name": name,
            "skipped": True,
            "collisions": 0,
            "worst_bucket": 1,
            "colliding_buckets": 0,
            "collisions_by_k": {},
            "compute_seconds": 0.0,
        }
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


def run_for_N(N: int) -> dict[str, object]:
    print(f"\n=== N = {N} ===", flush=True)
    t0 = time.time()
    codes = [ec.code for ec in enumerate_doubly_even(N)]
    t_enum = time.time() - t0
    print(f"  enumerated {len(codes)} canonical reps in {t_enum:.2f}s", flush=True)

    # Per-component signature timing — also serves as the prototype cost
    # estimate against the plan's < 30 µs / code gate.
    t_sig_components: dict[str, float] = {}
    sigs_per_code: list[dict[str, object]] = []

    t0 = time.time()
    cws_list = [codeword_list(c) for c in codes]
    t_sig_components["codeword_list"] = time.time() - t0

    t0 = time.time()
    masks_list = [column_bitmasks(c, cws) for c, cws in zip(codes, cws_list)]
    t_sig_components["column_bitmasks"] = time.time() - t0

    t0 = time.time()
    t1_list = [weight_enum_sig(c, cws) for c, cws in zip(codes, cws_list)]
    t_sig_components["T1_weight_enum"] = time.time() - t0

    t0 = time.time()
    col_list = [column_degree_sig(m) for m in masks_list]
    t_sig_components["coldeg"] = time.time() - t0

    t0 = time.time()
    pair_list = [pair_degree_sig(m) for m in masks_list]
    t_sig_components["pairdeg"] = time.time() - t0

    t0 = time.time()
    trip_list = [triple_degree_sig(m) for m in masks_list]
    t_sig_components["tripdeg"] = time.time() - t0

    for i in range(len(codes)):
        sigs_per_code.append(
            {
                "T1": t1_list[i],
                "coldeg": col_list[i],
                "pairdeg": pair_list[i],
                "tripdeg": trip_list[i],
                "cws": cws_list[i],
            }
        )

    n_codes = len(codes)
    per_code_us = {
        k: 1e6 * v / max(n_codes, 1) for k, v in t_sig_components.items()
    }
    print(
        f"  per-code timing (µs): "
        + ", ".join(f"{k}={per_code_us[k]:.2f}" for k in per_code_us),
        flush=True,
    )

    ranks = [c.rank for c in codes]
    tiers_out: list[dict[str, object]] = []
    for tier_name, keys in TIER_KEYS:
        out = bucket_collisions(sigs_per_code, keys, n_codes)
        coll_by_k: Counter = Counter()
        for v in out["colliding"]:
            kcounts = Counter(ranks[i] for i in v)
            for k, cnt in kcounts.items():
                if cnt > 1:
                    coll_by_k[k] += cnt - 1
        print(
            f"  {tier_name}: distinct={out['distinct']}/{n_codes}, "
            f"collisions={out['collisions']}, worst_bucket={out['worst']}, "
            f"colliding_buckets={len(out['colliding'])}",
            flush=True,
        )
        tiers_out.append(
            {
                "name": tier_name,
                "keys": list(keys),
                "distinct_signatures": out["distinct"],
                "collisions": out["collisions"],
                "worst_bucket": out["worst"],
                "colliding_buckets": len(out["colliding"]),
                "collisions_by_k": dict(coll_by_k),
            }
        )

    # Final refinement: T1+col+pair+trip + V_{c,h} on min-weight codewords
    # (Leon non-linear; the strongest cheap-ish invariant prior to |Aut|).
    base = bucket_collisions(
        sigs_per_code, ("T1", "coldeg", "pairdeg", "tripdeg"), n_codes
    )
    last_buckets = base["colliding"]
    last_buckets, t6_info = refine_on_colliders(
        "T1+col+pair+trip+V_min",
        lambda i: v_c_h_min_weight_sig(sigs_per_code[i]["cws"], N),  # type: ignore[arg-type]
        last_buckets,
        ranks,
    )
    tiers_out.append(t6_info)

    # Oracle |Aut|: where we land if we also had |Aut(C)| for free.
    if last_buckets:
        from doubly_even.canon.nauty import cached_canon_info

        last_buckets, aut_info = refine_on_colliders(
            "Oracle (+|Aut|)",
            lambda i: cached_canon_info(codes[i]).aut_order,
            last_buckets,
            ranks,
        )
        tiers_out.append(aut_info)

    return {
        "N": N,
        "total_classes": n_codes,
        "enum_seconds": t_enum,
        "per_code_microseconds": per_code_us,
        "tiers": tiers_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=str, default="20,22")
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
        out_path = out_dir / f"cubic-tensor-{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[saved {out_path}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
