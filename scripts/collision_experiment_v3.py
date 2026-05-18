"""v3: measure per-invariant cost AND test cheaper combinations.

v2 found: T1+T6+T11 gives 99.0% unique at N=22 (25 collisions); T1+T6+T7
gives 97.1% (79 collisions) but is expensive. We want the CHEAPEST combo
that hits ≥99% so the Rust port wins on wall time.

New invariants tested:
  T11 column weight-stratified profile (already in v2)
  T13 per-column-PAIR weight-stratified incidence (richer than T11)
  T14 sorted multiset over min-weight cws of (#w8 cws sharing ≥1 col with v,
       grouped by #shared) — a "min-weight × weight-8" interaction sig

Also tests T11/T1 *without* T6, since T11 alone may already cover what T6
adds.

Per-invariant TIMING in Python is reported so we can extrapolate Rust cost
(Rust roughly 30-100× faster on bit-twiddle).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
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


# --- timed signatures ----------------------------------------------------


def t1_weight_enum(cws, n):
    return tuple(sorted(c.bit_count() for c in cws))


def t6_vch_min_weight(cws, n):
    nz_w = [c.bit_count() for c in cws if c]
    if not nz_w:
        return ()
    m = min(nz_w)
    V = [c for c in cws if c.bit_count() == m]
    sigs = []
    for v in V:
        deg = [0] * (n + 1)
        for w in V:
            deg[(v ^ w).bit_count()] += 1
        sigs.append(tuple(deg))
    return tuple(sorted(sigs))


def t11_col_weight_profile(cws, n):
    """Per-column tuple (#w4, #w8, #w12, #w16) through it; sorted multiset."""
    weights = (4, 8, 12, 16, 20)
    per_col = []
    for j in range(n):
        bit = 1 << j
        cnt = [0] * len(weights)
        for c in cws:
            if c & bit:
                w = c.bit_count()
                for idx, ww in enumerate(weights):
                    if w == ww:
                        cnt[idx] += 1
                        break
        per_col.append(tuple(cnt))
    return tuple(sorted(per_col))


def t13_col_pair_weight_profile(cws, n):
    """Per-column-pair tuple (#w4, #w8, #w12) through BOTH; sorted multiset.
    Costlier than T11 but captures pair-incidence structure."""
    weights = (4, 8, 12, 16)
    per_pair = []
    for i in range(n):
        bi = 1 << i
        for j in range(i + 1, n):
            mask = bi | (1 << j)
            cnt = [0] * len(weights)
            for c in cws:
                if (c & mask) == mask:
                    w = c.bit_count()
                    for idx, ww in enumerate(weights):
                        if w == ww:
                            cnt[idx] += 1
                            break
            per_pair.append(tuple(cnt))
    return tuple(sorted(per_pair))


def t14_min_w8_interaction(cws, n):
    """For each min-weight cw v, the sorted dist of weight-8 cws by #col-overlap
    with v's support. Captures local v-by-stratum incidence."""
    nz_w = [c.bit_count() for c in cws if c]
    if not nz_w:
        return ()
    m = min(nz_w)
    V_min = [c for c in cws if c.bit_count() == m]
    V_8 = codewords_of_weight(cws, 8)
    sigs = []
    for v in V_min:
        overlap_counts: list[int] = []
        for w in V_8:
            overlap_counts.append((v & w).bit_count())
        sigs.append(tuple(sorted(overlap_counts)))
    return tuple(sorted(sigs))


SIG_FNS = [
    ("T1", t1_weight_enum),
    ("T6", t6_vch_min_weight),
    ("T11", t11_col_weight_profile),
    ("T13", t13_col_pair_weight_profile),
    ("T14", t14_min_w8_interaction),
]


COMBOS = [
    ("T1", ("T1",)),
    ("T1+T6", ("T1", "T6")),
    ("T1+T11", ("T1", "T11")),
    ("T11", ("T11",)),
    ("T1+T6+T11", ("T1", "T6", "T11")),
    ("T1+T11+T14", ("T1", "T11", "T14")),
    ("T1+T6+T14", ("T1", "T6", "T14")),
    ("T1+T13", ("T1", "T13")),
    ("T1+T6+T13", ("T1", "T6", "T13")),
    ("T1+T11+T13", ("T1", "T11", "T13")),
    ("ALL", ("T1", "T6", "T11", "T13", "T14")),
]


DUMP_COMBOS = {"T11", "T1+T11", "T1+T6+T11", "T1+T13"}


def run_for_N(N: int, dump_colliding: bool = False) -> dict[str, object]:
    print(f"\n=== N = {N} ===", flush=True)
    t0 = time.time()
    codes = [ec.code for ec in enumerate_doubly_even(N)]
    t_enum = time.time() - t0
    print(f"  enumerated {len(codes)} canonical reps in {t_enum:.2f}s", flush=True)

    # materialise codewords once
    cws_per_code = [codeword_list(c) for c in codes]

    # time and compute each signature
    sigs_per_code: dict[str, list[object]] = {name: [None] * len(codes) for name, _ in SIG_FNS}
    per_sig_time: dict[str, float] = {}
    for name, fn in SIG_FNS:
        t0 = time.time()
        for i, c in enumerate(codes):
            sigs_per_code[name][i] = fn(cws_per_code[i], c.n)
        t = time.time() - t0
        per_sig_time[name] = t
        per_code_us = (t / max(1, len(codes))) * 1e6
        print(f"  {name:5s} per-code: {per_code_us:7.1f} µs  (total {t:.2f}s)", flush=True)

    tiers_out: list[dict[str, object]] = []
    for combo_name, keys in COMBOS:
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for i in range(len(codes)):
            sig = tuple(sigs_per_code[k][i] for k in keys)
            buckets[sig].append(i)
        n_distinct = len(buckets)
        colliding = [(sig, v) for sig, v in buckets.items() if len(v) > 1]
        worst = max((len(v) for _, v in colliding), default=1)
        n_collisions = len(codes) - n_distinct
        n_unique = n_distinct - len(colliding)
        n_classes_in_collision = sum(len(v) for _, v in colliding)
        print(
            f"  {combo_name:25s}: dist={n_distinct:5d}  coll={n_collisions:5d}  "
            f"worst={worst:3d}  uniq_buckets={n_unique:5d}/{len(codes)} "
            f"({100 * n_unique / len(codes):5.1f}%) "
            f"classes_in_coll={n_classes_in_collision}",
            flush=True,
        )
        cost_us_total = sum(per_sig_time[k] for k in keys) / max(1, len(codes)) * 1e6
        tier_entry: dict[str, object] = {
            "name": combo_name,
            "keys": list(keys),
            "distinct_signatures": n_distinct,
            "collisions": n_collisions,
            "worst_bucket": worst,
            "colliding_buckets": len(colliding),
            "unique_buckets": n_unique,
            "classes_in_colliding_buckets": n_classes_in_collision,
            "py_cost_us_per_code": cost_us_total,
        }
        if dump_colliding and combo_name in DUMP_COMBOS:
            tier_entry["colliding_signatures"] = [
                {"sig_repr": repr(sig), "code_indices": v}
                for sig, v in colliding
            ]
        tiers_out.append(tier_entry)

    return {
        "N": N,
        "total_classes": len(codes),
        "enum_seconds": t_enum,
        "per_sig_seconds": per_sig_time,
        "tiers": tiers_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=str, default="18,20,22")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--dump-colliding-hashes",
        action="store_true",
        help="Embed colliding-signature details for diagnostic combos (T11 et al.) into the JSON.",
    )
    args = parser.parse_args()
    Ns = [int(n) for n in args.N.split(",")]

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Ns": Ns,
        "results": [],
    }
    for N in Ns:
        all_results["results"].append(
            run_for_N(N, dump_colliding=args.dump_colliding_hashes)
        )

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = HERE / "bench-results"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"{ts}-leon-collision-experiment-v3.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[saved {out_path}]", flush=True)


if __name__ == "__main__":
    main()
