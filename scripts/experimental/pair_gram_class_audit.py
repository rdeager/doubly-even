"""Phase 0 — empirical pair-gram completeness audit.

Plan: ``/home/dev/.claude/plans/last-several-sessions-have-wild-cupcake.md``.

Asks: is the pair-gram per-column row-multiset tuple a *complete*
invariant of permutation-equivalence classes for doubly-even
self-orthogonal binary codes?

We answer empirically by enumerating every canonical class at
N ∈ {default: 16, 18, 20, 22; with ``--include-N24`` add 24} and
checking whether any two distinct classes share the same pair-gram
tuple.

Canary at N = 16: assert the pair-gram tuples for d_16^+ and
E_8 ⊕ E_8 differ. These share T11 (the only entry in the N=16
blocklist) but should be distinguished by pair-gram.

Outputs JSON to ``scripts/bench-results/pair-gram-class-audit-<ts>.json``
with per-N collision count, worst bucket size, per-call timing
statistics, and a log-linear fit of µs/call vs the code's rank ``k``
for projecting cost at N=26 / 28.

Usage::

    uv run python scripts/experimental/pair_gram_class_audit.py
    uv run python scripts/experimental/pair_gram_class_audit.py --include-N24
    uv run python scripts/experimental/pair_gram_class_audit.py --N 16,18
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402

from pair_gram_experiment import (  # noqa: E402
    codeword_list,
    column_bitmasks_subset,
    pair_gram_per_col_sig,
    sparse_lowest_stratum,
    stacked_span_aware,
)


def codewords_by_weight(cws: list[int]) -> dict[int, list[int]]:
    """Group non-zero codewords by Hamming weight."""
    out: dict[int, list[int]] = defaultdict(list)
    for c in cws:
        if c:
            out[c.bit_count()].append(c)
    return dict(out)


def t11_per_col_tuple(cws: list[int], N: int) -> tuple[tuple[int, ...], ...]:
    """Per-column weight-bucket counts (T11 underlying multiset).

    Same idea as ``rust/src/canon.rs::compute_t11_hash`` but as a tuple
    (so we can compute set-equality, not just a 64-bit hash). Buckets
    weight ∈ {4, 8, 12, 16, 20}.
    """
    BUCKETS = (4, 8, 12, 16, 20)
    profile = [[0] * len(BUCKETS) for _ in range(N)]
    for c in cws:
        w = c.bit_count()
        if w not in BUCKETS:
            continue
        bi = BUCKETS.index(w)
        m = c
        while m:
            j = (m & -m).bit_length() - 1
            profile[j][bi] += 1
            m &= m - 1
    per_col = sorted(tuple(p) for p in profile)
    return tuple(per_col)


def coord_weight_profile(cws: list[int], N: int) -> tuple[tuple[int, ...], ...]:
    """Gemini's suggestion #1 — per-column full weight multiset.

    For each column j: sorted multiset of wt(c) for c ∈ C with c_j = 1.
    No weight bucketing. Strictly at least as fine as T11; equivalent for
    doubly-even codes at N ≤ 22 (only weights are 0, 4, 8, 12, 16, 20).
    """
    per_col: list[list[int]] = [[] for _ in range(N)]
    for c in cws:
        if c == 0:
            continue
        w = c.bit_count()
        m = c
        while m:
            j = (m & -m).bit_length() - 1
            per_col[j].append(w)
            m &= m - 1
    return tuple(sorted(tuple(sorted(p)) for p in per_col))


def weight_enumerator_tuple(cws: list[int], N: int) -> tuple[int, ...]:
    """Full weight enumerator as a tuple (WE[0], WE[1], ..., WE[N]).

    For doubly-even codes, only WE[0], WE[4], WE[8], ... are non-zero.
    """
    out = [0] * (N + 1)
    for c in cws:
        out[c.bit_count()] += 1
    return tuple(out)


def cubic_tensor_per_col_sig(masks: list[int], N: int) -> tuple[tuple[int, ...], ...]:
    """Cubic tensor T[j, j', j''] = popcount(m_j & m_{j'} & m_{j''}) per column.

    Per column j: sorted multiset of T[j, j', j''] over pairs j' < j''
    with both ≠ j. Then sort across columns.
    """
    per_col_rows: list[tuple[int, ...]] = []
    for j in range(N):
        mj = masks[j]
        triples: list[int] = []
        for jp in range(N):
            if jp == j:
                continue
            mjjp = mj & masks[jp]
            for jpp in range(jp + 1, N):
                if jpp == j:
                    continue
                triples.append((mjjp & masks[jpp]).bit_count())
        triples.sort()
        per_col_rows.append(tuple(triples))
    per_col_rows.sort()
    return tuple(per_col_rows)


def compute_signatures(
    code, N: int
) -> dict[str, tuple]:
    """Compute every candidate invariant for one code; return as dict."""
    cws_all = codeword_list(code)
    masks_all = column_bitmasks_subset(cws_all, N)

    cws_nz_by_w = codewords_by_weight(cws_all)
    cws_sparse = sparse_lowest_stratum(cws_nz_by_w)  # weight-4 (Gemini #2)
    masks_sparse = column_bitmasks_subset(cws_sparse, N) if cws_sparse else [0] * N

    cws_stacked = stacked_span_aware(cws_nz_by_w, code.rank, N)
    if cws_stacked is None:
        sig_stacked = ("BAILED",)
    else:
        masks_stacked = column_bitmasks_subset(cws_stacked, N)
        sig_stacked = pair_gram_per_col_sig(masks_stacked)

    sig_dense = pair_gram_per_col_sig(masks_all)
    sig_sparse = (
        pair_gram_per_col_sig(masks_sparse) if cws_sparse else ((0,) * (N - 1),) * N
    )
    sig_t11 = t11_per_col_tuple(cws_all, N)
    sig_cwp = coord_weight_profile(cws_all, N)  # Gemini #1
    sig_we = weight_enumerator_tuple(cws_all, N)  # full WE (weaker than T11)
    if N <= 22:
        sig_cubic = cubic_tensor_per_col_sig(masks_all, N)
    else:
        sig_cubic = ("SKIPPED_N>22",)  # too slow at higher N for audit

    # Cubic on weight-4 codewords only (cheaper at high N).
    if cws_sparse:
        sig_w4_cubic = cubic_tensor_per_col_sig(masks_sparse, N)
    else:
        sig_w4_cubic = ((0,),) * N

    return {
        # Singletons
        "we": sig_we,                       # cheapest — full WE; expected to fail canary
        "t11": sig_t11,
        "cwp": sig_cwp,
        "weight4_pg": sig_sparse,           # Gemini #2
        "stacked_pg": sig_stacked,
        "dense_pg": sig_dense,
        "cubic": sig_cubic,
        "w4_cubic": sig_w4_cubic,           # cubic on weight-4 only
        # Pairwise with T11
        "t11_w4pg": (sig_t11, sig_sparse),
        "t11_stacked": (sig_t11, sig_stacked),
        "t11_dense": (sig_t11, sig_dense),
        "t11_cubic": (sig_t11, sig_cubic),
        "t11_w4cubic": (sig_t11, sig_w4_cubic),
        # Triples (strongest cheap combinations)
        "t11_stacked_w4cubic": (sig_t11, sig_stacked, sig_w4_cubic),
        "t11_w4pg_w4cubic": (sig_t11, sig_sparse, sig_w4_cubic),
    }


VARIANTS = (
    "we", "t11", "cwp", "weight4_pg", "stacked_pg", "dense_pg", "cubic", "w4_cubic",
    "t11_w4pg", "t11_stacked", "t11_dense", "t11_cubic", "t11_w4cubic",
    "t11_stacked_w4cubic", "t11_w4pg_w4cubic",
)


def audit_one_N(N: int) -> dict[str, Any]:
    """Enumerate, compute multiple invariant variants, return collision report
    for each."""

    sigs_by_variant: dict[str, list[tuple]] = {v: [] for v in VARIANTS}
    ranks: list[int] = []
    auts: list[int] = []
    timings_us_by_k_dense: dict[int, list[float]] = defaultdict(list)
    timings_us_by_k_stacked: dict[int, list[float]] = defaultdict(list)
    wall_t0 = time.perf_counter()

    for ec in enumerate_doubly_even(N):
        # Time the two most relevant variants separately for cost modeling.
        cws_all = codeword_list(ec.code)
        masks_all = column_bitmasks_subset(cws_all, N)
        cws_nz_by_w = codewords_by_weight(cws_all)
        cws_stacked = stacked_span_aware(cws_nz_by_w, ec.code.rank, N)

        t0 = time.perf_counter()
        _ = pair_gram_per_col_sig(masks_all)
        dt_dense_us = (time.perf_counter() - t0) * 1e6

        if cws_stacked is not None:
            masks_stacked = column_bitmasks_subset(cws_stacked, N)
            t0 = time.perf_counter()
            _ = pair_gram_per_col_sig(masks_stacked)
            dt_stacked_us = (time.perf_counter() - t0) * 1e6
        else:
            dt_stacked_us = 0.0

        k = ec.code.rank
        timings_us_by_k_dense[k].append(dt_dense_us)
        if dt_stacked_us > 0:
            timings_us_by_k_stacked[k].append(dt_stacked_us)
        ranks.append(k)
        auts.append(ec.aut_order)

        sigs = compute_signatures(ec.code, N)
        for v in VARIANTS:
            sigs_by_variant[v].append(sigs[v])

    wall_s = time.perf_counter() - wall_t0
    n_classes = len(ranks)

    per_variant_report: dict[str, dict[str, Any]] = {}
    for v in VARIANTS:
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for i, s in enumerate(sigs_by_variant[v]):
            buckets[s].append(i)
        n_distinct = len(buckets)
        colliding = [vv for vv in buckets.values() if len(vv) > 1]
        worst = max((len(vv) for vv in colliding), default=1)
        per_variant_report[v] = {
            "n_distinct": n_distinct,
            "n_collisions": n_classes - n_distinct,
            "n_collision_buckets": len(colliding),
            "worst_bucket_size": worst,
        }

    def k_timings(d):
        out: dict[str, dict[str, float]] = {}
        for k, ts in sorted(d.items()):
            if not ts:
                continue
            out[str(k)] = {
                "n": len(ts),
                "mean_us": statistics.fmean(ts),
                "median_us": statistics.median(ts),
                "min_us": min(ts),
                "max_us": max(ts),
            }
        return out

    canary_n16: dict[str, Any] | None = None
    if N == 16:
        k_full = N // 2
        full_rank_indices = [i for i, k in enumerate(ranks) if k == k_full]
        if full_rank_indices:
            canary_per_variant: dict[str, Any] = {}
            for v in VARIANTS:
                sigs_v = sigs_by_variant[v]
                full_rank_sigs_v = {sigs_v[i] for i in full_rank_indices}
                canary_per_variant[v] = {
                    "n_distinct": len(full_rank_sigs_v),
                    "separates": len(full_rank_sigs_v) == len(full_rank_indices),
                }
            canary_n16 = {
                "k_full": k_full,
                "n_full_rank_classes": len(full_rank_indices),
                "full_rank_classes": [
                    {"index": i, "aut_order": auts[i]} for i in full_rank_indices
                ],
                "per_variant": canary_per_variant,
            }

    return {
        "N": N,
        "n_classes": n_classes,
        "per_variant": per_variant_report,
        "per_k_timings_us_dense": k_timings(timings_us_by_k_dense),
        "per_k_timings_us_stacked": k_timings(timings_us_by_k_stacked),
        "canary_n16": canary_n16,
        "wall_s": wall_s,
    }


def fit_loglinear(per_k_timings: dict[int, dict[str, float]]) -> dict[str, Any]:
    """Fit log(median_us) ≈ a + b·k. Returns extrapolated cost at k=13, 14, 16."""
    import math

    xs: list[float] = []
    ys: list[float] = []
    for k_str, stats in per_k_timings.items():
        if stats["median_us"] <= 0:
            continue
        xs.append(float(k_str))
        ys.append(math.log(stats["median_us"]))

    if len(xs) < 2:
        return {"ok": False, "reason": "fewer than 2 data points"}

    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return {"ok": False, "reason": "degenerate x"}
    b = num / den
    a = mean_y - b * mean_x

    projections = {}
    for k_target in [11, 13, 14, 16]:
        projections[f"k={k_target}"] = math.exp(a + b * k_target)

    return {
        "ok": True,
        "log_intercept_a": a,
        "log_slope_b": b,
        "interpretation": f"µs ≈ exp({a:.3f} + {b:.3f}·k)",
        "projected_us": projections,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--N",
        type=str,
        default="",
        help="Comma-separated N values; default {16,18,20,22}; --include-N24 adds 24.",
    )
    ap.add_argument(
        "--include-N24",
        action="store_true",
        help="Include N=24 (adds ~10–30 s).",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.N:
        N_values = [int(s.strip()) for s in args.N.split(",") if s.strip()]
    else:
        N_values = [16, 18, 20, 22]
        if args.include_N24:
            N_values.append(24)

    overall: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "pair_gram_class_audit",
        "N_values": N_values,
        "per_N": {},
    }

    all_per_k_dense: dict[int, dict[str, float]] = {}
    all_per_k_stacked: dict[int, dict[str, float]] = {}

    for N in N_values:
        if not args.quiet:
            print(f"\n== N={N} ==", flush=True)
        rep = audit_one_N(N)
        overall["per_N"][str(N)] = rep
        if not args.quiet:
            print(f"  classes={rep['n_classes']}, wall={rep['wall_s']:.2f}s")
            print(f"  {'variant':<15}{'distinct':>10}{'colls':>8}{'buckets':>10}{'worst':>8}")
            for v in VARIANTS:
                rr = rep["per_variant"][v]
                print(
                    f"  {v:<15}{rr['n_distinct']:>10}{rr['n_collisions']:>8}"
                    f"{rr['n_collision_buckets']:>10}{rr['worst_bucket_size']:>8}"
                )
            if rep["canary_n16"]:
                c = rep["canary_n16"]
                print(
                    f"  N=16 canary: {c['n_full_rank_classes']} k={c['k_full']} classes"
                )
                for fc in c["full_rank_classes"]:
                    print(f"    class idx={fc['index']} |Aut|={fc['aut_order']}")
                print(f"  {'variant':<15}{'sep':>6}{'distinct_full':>14}")
                for v in VARIANTS:
                    cv = c["per_variant"][v]
                    print(
                        f"  {v:<15}{'PASS' if cv['separates'] else 'FAIL':>6}"
                        f"{cv['n_distinct']:>14}"
                    )
        for k_str, ts in rep["per_k_timings_us_dense"].items():
            all_per_k_dense[int(k_str)] = ts
        for k_str, ts in rep["per_k_timings_us_stacked"].items():
            all_per_k_stacked[int(k_str)] = ts

    fit_in_dense = {str(k): v for k, v in all_per_k_dense.items()}
    fit_in_stacked = {str(k): v for k, v in all_per_k_stacked.items()}
    overall["loglinear_fit_dense"] = fit_loglinear(fit_in_dense)
    overall["loglinear_fit_stacked"] = fit_loglinear(fit_in_stacked)
    if not args.quiet:
        for name, f in [
            ("dense", overall["loglinear_fit_dense"]),
            ("stacked", overall["loglinear_fit_stacked"]),
        ]:
            if f.get("ok"):
                print(f"\nlog-linear fit ({name}): {f['interpretation']}")
                for k_target, us in f["projected_us"].items():
                    print(f"  projected {k_target}: {us:.1f} µs/call")

    # Overall decision summary across all variants.
    summary_per_variant: dict[str, dict[str, int]] = {}
    for v in VARIANTS:
        total = sum(r["per_variant"][v]["n_collisions"] for r in overall["per_N"].values())
        worst = max(r["per_variant"][v]["worst_bucket_size"] for r in overall["per_N"].values())
        n16_ok = True
        n16 = overall["per_N"].get("16")
        if n16 and n16["canary_n16"]:
            n16_ok = n16["canary_n16"]["per_variant"][v]["separates"]
        summary_per_variant[v] = {
            "total_collisions": total,
            "worst_bucket_any_N": worst,
            "n16_canary_pass": n16_ok,
        }
    overall["summary"] = summary_per_variant

    if not args.quiet:
        print("\n=== Verdict summary (across all audited N) ===")
        print(f"  {'variant':<15}{'total_colls':>14}{'worst':>8}{'n16_canary':>14}")
        for v in VARIANTS:
            s = summary_per_variant[v]
            print(
                f"  {v:<15}{s['total_collisions']:>14}{s['worst_bucket_any_N']:>8}"
                f"{'PASS' if s['n16_canary_pass'] else 'FAIL':>14}"
            )
        winners = [
            v for v in VARIANTS
            if summary_per_variant[v]["total_collisions"] == 0
            and summary_per_variant[v]["n16_canary_pass"]
        ]
        if winners:
            print(f"\nGO with: {', '.join(winners)}")
        else:
            print("\nNO-GO: every variant has class collisions; need stronger invariant.")

    out_dir = REPO_ROOT / "scripts" / "bench-results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"pair-gram-class-audit-{ts}.json"
    out_path.write_text(json.dumps(overall, indent=2))
    if not args.quiet:
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
