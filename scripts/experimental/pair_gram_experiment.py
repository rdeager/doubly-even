"""Pair-Gram per-column row-multiset prefilter probe — three M-variants.

Extends the original dense-only probe with two sparse variants matching
the paper's M = weight-w setting and D10's span-aware accumulator. For
each variant we measure (collision-cut %, Python µs/code, rows-in-M)
so Phase A can pick the right M-definition (if any) for the Rust port.

Variants
--------
* **dense**   — M = all 2^k codewords. Reproduces the
  2026-05-17 baseline (`bench-results/pair-gram-20260517T174251Z.json`).
* **sparse**  — M = codewords of the lowest non-zero weight stratum.
  Matches the paper's M = weight-w setting.
* **stacked** — M = exactly the codewords D10's
  ``collect_low_weight_codewords`` (`rust/src/canon.rs:307-355`) keeps:
  span-aware ascending strata, capped at ``2^k / 2``. When the
  accumulator bails (too dense), the Rust feature would be inert on
  that code; we tag those with ``"BAILED"`` so they form a single
  bucket under this variant — equivalent to "T1+coldeg only" on that
  subset.

Mathematical signature is the same per variant: per-column sorted
multiset of ``G[j, j']`` for ``j' ≠ j`` over the chosen M, then sorted
across columns. Strictly finer than the flat C(N, 2) entry multiset.

Decision rule (autonomous): promote variant for Rust port iff
``collision-cut ≥ 30 %`` at N = 22 AND projected Rust µs/code ≤ 20
(Python × 1/3 from the empirical cubic-tensor and D10 port ratios).

Usage::

    uv run python scripts/pair_gram_experiment.py --N 18,20,22

Writes JSON to ``scripts/bench-results/pair-gram-variants-<timestamp>.json``.
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
from doubly_even.spec.codes import Code, _compute_rref  # noqa: E402


# ----------------------------------------------------------------- utils


def codeword_list(code: Code) -> list[int]:
    """All 2^k codewords (including 0) via basis doubling."""
    rows, _ = code.rref_basis()
    cws = [0]
    for r in rows:
        cws = cws + [c ^ r for c in cws]
    return cws


def column_bitmasks_subset(cws: list[int], n: int) -> list[int]:
    """For each column j ∈ [0, n), bit i of masks[j] is set iff
    ``cws[i]`` has a 1 in position j. Works for any subset of
    codewords; bit-position is the index in the subset.
    """
    masks = [0] * n
    for i, c in enumerate(cws):
        m = c
        while m:
            j = (m & -m).bit_length() - 1
            masks[j] |= 1 << i
            m &= m - 1
    return masks


# ----------------------------------------------------------------- M-variants


def sparse_lowest_stratum(cws_nonzero_by_weight: dict[int, list[int]]) -> list[int]:
    """Lowest non-zero weight stratum only."""
    if not cws_nonzero_by_weight:
        return []
    w_min = min(cws_nonzero_by_weight)
    return list(cws_nonzero_by_weight[w_min])


def stacked_span_aware(
    cws_nonzero_by_weight: dict[int, list[int]], k: int, n: int
) -> list[int] | None:
    """Python mirror of ``rust/src/canon.rs:collect_low_weight_codewords``.

    Walk ascending weight strata; after each stratum compute the rank
    of the accumulated set. Stop when rank == k (i.e., the set spans C).
    Bail with ``None`` if the size exceeds ``2^k / 2`` before spanning
    (the full bipartite would be cheaper at that point).
    """
    if k == 0:
        return None
    bail = (1 << k) // 2
    accum: list[int] = []
    for w in sorted(cws_nonzero_by_weight):
        stratum = cws_nonzero_by_weight[w]
        accum.extend(stratum)
        if len(accum) > bail:
            return None
        # rank check via the project's cached RREF helper
        rref_rows, _pivots = _compute_rref(n, tuple(accum))
        if len(rref_rows) == k:
            return accum
    return None


# ----------------------------------------------------------------- signatures


def weight_enum_sig(cws: list[int]) -> tuple[int, ...]:
    return tuple(sorted(c.bit_count() for c in cws))


def column_degree_sig(masks: list[int]) -> tuple[int, ...]:
    return tuple(sorted(m.bit_count() for m in masks))


def pair_degree_flat_sig(masks: list[int]) -> tuple[int, ...]:
    """Flat C(N, 2) multiset of G[j, j'] for j < j'.

    Reproduced for cross-check; measured to add zero refinement
    over T1 at N = 20, 22 (cubic-tensor bench-results).
    """
    n = len(masks)
    counts: list[int] = []
    for j1 in range(n):
        mj1 = masks[j1]
        for j2 in range(j1 + 1, n):
            counts.append((mj1 & masks[j2]).bit_count())
    return tuple(sorted(counts))


def pair_gram_per_col_sig(masks: list[int]) -> tuple[tuple[int, ...], ...]:
    """Per-column row-multiset of G = M^T M.

    For each column j, sorted multiset of G[j, j'] for j' ≠ j.
    Then sort the N tuples across columns. Strictly finer than
    ``pair_degree_flat_sig``.
    """
    n = len(masks)
    per_col: list[tuple[int, ...]] = []
    for j in range(n):
        mj = masks[j]
        row: list[int] = []
        for jp in range(n):
            if jp == j:
                continue
            row.append((mj & masks[jp]).bit_count())
        row.sort()
        per_col.append(tuple(row))
    per_col.sort()
    return tuple(per_col)


# Sentinel for codes where the span-aware accumulator would bail and the
# Rust feature would be inert. Distinct from any real signature tuple.
STACKED_BAILED = ("BAILED",)


# ----------------------------------------------------------------- experiment


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


# Tiers ordered for readability; the per-variant pair_per_col entries are
# the headline rows the decision rule reads.
TIER_KEYS = [
    ("T1", ("T1",)),
    ("T1+col", ("T1", "coldeg")),
    ("T1+col+pair_flat", ("T1", "coldeg", "pair_flat")),
    ("T1+col+pair_per_col[dense]", ("T1", "coldeg", "pair_dense")),
    ("T1+col+pair_per_col[sparse]", ("T1", "coldeg", "pair_sparse")),
    ("T1+col+pair_per_col[stacked]", ("T1", "coldeg", "pair_stacked")),
]


def _row_count_stats(rows: list[int]) -> dict[str, float | int]:
    if not rows:
        return {"mean": 0.0, "max": 0, "min": 0}
    return {
        "mean": sum(rows) / len(rows),
        "max": max(rows),
        "min": min(rows),
    }


def run_for_N(N: int) -> dict[str, object]:
    print(f"\n=== N = {N} ===", flush=True)
    t0 = time.time()
    codes = [ec.code for ec in enumerate_doubly_even(N)]
    t_enum = time.time() - t0
    print(
        f"  enumerated {len(codes)} canonical reps in {t_enum:.2f}s",
        flush=True,
    )

    n_codes = len(codes)
    t_components: dict[str, float] = {}

    # --- Dense M (all 2^k codewords) ------------------------------------
    t0 = time.time()
    cws_dense = [codeword_list(c) for c in codes]
    t_components["codeword_list_dense"] = time.time() - t0

    t0 = time.time()
    masks_dense = [
        column_bitmasks_subset(cws, c.n) for c, cws in zip(codes, cws_dense)
    ]
    t_components["column_bitmasks_dense"] = time.time() - t0

    # --- Group non-zero codewords by Hamming weight (shared by sparse + stacked)
    t0 = time.time()
    cws_by_weight: list[dict[int, list[int]]] = []
    for cws in cws_dense:
        by_w: dict[int, list[int]] = defaultdict(list)
        for c in cws:
            if c != 0:
                by_w[c.bit_count()].append(c)
        cws_by_weight.append(dict(by_w))
    t_components["group_by_weight"] = time.time() - t0

    # --- Sparse M (lowest non-zero weight stratum only) ------------------
    t0 = time.time()
    cws_sparse = [sparse_lowest_stratum(byw) for byw in cws_by_weight]
    t_components["subset_sparse"] = time.time() - t0

    t0 = time.time()
    masks_sparse = [
        column_bitmasks_subset(cws, c.n) for c, cws in zip(codes, cws_sparse)
    ]
    t_components["column_bitmasks_sparse"] = time.time() - t0

    # --- Stacked M (D10 span-aware accumulator) --------------------------
    t0 = time.time()
    cws_stacked: list[list[int] | None] = [
        stacked_span_aware(byw, c.rank, c.n)
        for c, byw in zip(codes, cws_by_weight)
    ]
    t_components["subset_stacked"] = time.time() - t0
    stacked_bailed = sum(1 for s in cws_stacked if s is None)

    t0 = time.time()
    masks_stacked: list[list[int] | None] = [
        column_bitmasks_subset(cws, c.n) if cws is not None else None
        for c, cws in zip(codes, cws_stacked)
    ]
    t_components["column_bitmasks_stacked"] = time.time() - t0

    # --- Signature pipelines (per variant) -------------------------------
    t0 = time.time()
    t1_list = [weight_enum_sig(cws) for cws in cws_dense]
    t_components["T1_weight_enum"] = time.time() - t0

    t0 = time.time()
    col_list = [column_degree_sig(m) for m in masks_dense]
    t_components["coldeg"] = time.time() - t0

    t0 = time.time()
    pair_flat_list = [pair_degree_flat_sig(m) for m in masks_dense]
    t_components["pair_flat"] = time.time() - t0

    # Per-variant pair-gram signatures with their own timings.
    t0 = time.time()
    pair_dense_list = [pair_gram_per_col_sig(m) for m in masks_dense]
    t_components["pair_per_col_dense"] = time.time() - t0

    t0 = time.time()
    pair_sparse_list = [pair_gram_per_col_sig(m) for m in masks_sparse]
    t_components["pair_per_col_sparse"] = time.time() - t0

    # Stacked: skip the bailed codes; for them, use the sentinel tuple
    # so they form a single bucket under this variant (representing
    # "feature inert, falls back to canon_info_native").
    t0 = time.time()
    pair_stacked_list: list[tuple[tuple[int, ...], ...] | tuple[str, ...]] = []
    for m in masks_stacked:
        if m is None:
            pair_stacked_list.append(STACKED_BAILED)
        else:
            pair_stacked_list.append(pair_gram_per_col_sig(m))
    t_stacked_sig = time.time() - t0
    t_components["pair_per_col_stacked"] = t_stacked_sig

    n_stacked_active = n_codes - stacked_bailed

    sigs_per_code = [
        {
            "T1": t1_list[i],
            "coldeg": col_list[i],
            "pair_flat": pair_flat_list[i],
            "pair_dense": pair_dense_list[i],
            "pair_sparse": pair_sparse_list[i],
            "pair_stacked": pair_stacked_list[i],
        }
        for i in range(n_codes)
    ]

    # Per-code µs reported against the number of codes the variant
    # actually does work on. For the stacked sig, that's
    # ``n_stacked_active``; for everything else, it's all codes.
    per_code_us: dict[str, float] = {}
    for k, v in t_components.items():
        denom = (
            n_stacked_active
            if k in ("pair_per_col_stacked", "column_bitmasks_stacked", "subset_stacked")
            else n_codes
        )
        denom = max(denom, 1)
        per_code_us[k] = 1e6 * v / denom

    print(
        "  per-code timing (µs): "
        + ", ".join(f"{k}={per_code_us[k]:.2f}" for k in per_code_us),
        flush=True,
    )

    # --- rows-in-M stats -------------------------------------------------
    rows_dense = [len(c) for c in cws_dense]
    rows_sparse = [len(c) for c in cws_sparse]
    rows_stacked_active = [len(c) for c in cws_stacked if c is not None]

    rows_in_M = {
        "dense": _row_count_stats(rows_dense),
        "sparse": _row_count_stats(rows_sparse),
        "stacked": {
            **_row_count_stats(rows_stacked_active),
            "bailed": stacked_bailed,
            "active": n_stacked_active,
        },
    }
    print(
        f"  rows-in-M: dense mean={rows_in_M['dense']['mean']:.1f} max={rows_in_M['dense']['max']}, "
        f"sparse mean={rows_in_M['sparse']['mean']:.1f} max={rows_in_M['sparse']['max']}, "
        f"stacked mean={rows_in_M['stacked']['mean']:.1f} max={rows_in_M['stacked']['max']} "
        f"bailed={stacked_bailed}/{n_codes}",
        flush=True,
    )

    # --- Tier evaluation -------------------------------------------------
    ranks = [c.rank for c in codes]
    tiers_out: list[dict[str, object]] = []
    t1_collisions = None
    for tier_name, keys in TIER_KEYS:
        out = bucket_collisions(sigs_per_code, keys, n_codes)
        coll_by_k: Counter = Counter()
        for v in out["colliding"]:
            kcounts = Counter(ranks[i] for i in v)
            for k, cnt in kcounts.items():
                if cnt > 1:
                    coll_by_k[k] += cnt - 1
        if tier_name == "T1":
            t1_collisions = out["collisions"]
        cut_pct: float | None = None
        if t1_collisions is not None and t1_collisions > 0 and tier_name != "T1":
            cut_pct = 100.0 * (t1_collisions - out["collisions"]) / t1_collisions
        cut_str = (
            f", cut_vs_T1={cut_pct:.1f}%" if cut_pct is not None else ""
        )
        print(
            f"  {tier_name}: distinct={out['distinct']}/{n_codes}, "
            f"collisions={out['collisions']}, worst_bucket={out['worst']}, "
            f"colliding_buckets={len(out['colliding'])}{cut_str}",
            flush=True,
        )
        tiers_out.append(
            {
                "name": tier_name,
                "keys": list(keys),
                "distinct_signatures": out["distinct"],
                "collisions": out["collisions"],
                "collision_cut_vs_T1_pct": cut_pct,
                "worst_bucket": out["worst"],
                "colliding_buckets": len(out["colliding"]),
                "collisions_by_k": dict(coll_by_k),
            }
        )

    return {
        "N": N,
        "total_classes": n_codes,
        "enum_seconds": t_enum,
        "stacked_bailed_count": stacked_bailed,
        "stacked_active_count": n_stacked_active,
        "per_code_microseconds": per_code_us,
        "rows_in_M": rows_in_M,
        "tiers": tiers_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=str, default="18,20,22")
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
        out_path = out_dir / f"pair-gram-variants-{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[saved {out_path}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
