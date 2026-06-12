"""Phase 3 measurement for the cheap-equivalence-verifier plan.

Times :func:`doubly_even.canon.paired_iso.paired_iso` on representative
``(D, bucket)`` workloads generated from a live enumeration, then
projects the wall savings of a hypothetical "try paired-iso first,
fall back to nauty" dispatch using the Phase 1 numbers
(``scripts/bench_canon_bucket_stats.py``).

Generation: enumerate all classes at the requested N, group their
canonical forms by weight enumerator (= the live secondary-cache key),
then for each class take ``--perms-per-class`` random column
permutations of its RREF as test D's. Each D is run through paired-iso
against the entire weight-enum bucket in insertion order until a match
is found (the bucket-hit path) or all entries are exhausted (the
bucket-miss path, which would fall back to nauty in the live system).

This synthesizes the live traffic statistically — random perms are
exactly the equivalence-class distribution; the bucket structure is
identical to the live cache.

Python wall numbers don't translate 1:1 to Rust. The script also reports
the mean operation counts (refines / branches / leaves) of paired-iso
vs a single Feulner canonicaliser run on the same D — that ratio *does*
translate, since both kernels share the same refinement primitives.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import doubly_even_kernel as _kernel  # noqa: E402
from doubly_even.canon.experimental.feulner import canon_info_feulner  # noqa: E402
from doubly_even.canon.experimental.paired_iso import (  # noqa: E402
    IsoCounters,
    _weight_multiset,
    paired_iso,
)
from doubly_even.spec.codes import Code  # noqa: E402
from doubly_even.spec.mass import gaborit_sigma  # noqa: E402
from doubly_even.spec.vectors import apply_permutation  # noqa: E402


def _enumerate_class_canonical_forms(N: int) -> list[tuple[int, ...]]:
    """Return one canonical RREF per equivalence class at N (via kernel)."""
    cap = N // 2
    quota_vec = [gaborit_sigma(N, k) for k in range(cap + 1)]
    factorial_N = math.factorial(N)
    raw, _stats, _per_k = _kernel.enumerate_doubly_even(N, cap, quota_vec, factorial_N)
    # NOTE (autom-only lever, 2026-06-12): this script consumes the
    # per-class canonical_column_order, which is EMPTY under the default
    # labelling mode — run it with DOUBLY_EVEN_CANON_LABELLING=full.
    # `raw` entries are (rref, canonical_column_order, gens, aut_order_str, orbits).
    # Apply the canonical column order to each rref then re-RREF to get the
    # canonical form (same as what the kernel's secondary cache stores).
    out: list[tuple[int, ...]] = []
    for rref, ccol, *_ in raw:
        permuted = tuple(apply_permutation(b, list(ccol)) for b in rref)
        cf = Code(N, permuted).rref_basis()[0]
        out.append(cf)
    return out


def _bucket_by_weight_enum(
    cfs: list[tuple[int, ...]],
) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    """Group canonical forms by weight multiset (the live cache's key)."""
    buckets: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for cf in cfs:
        k = len(cf)
        we = _weight_multiset(cf, k)
        buckets.setdefault(we, []).append(cf)
    return buckets


def _random_perm_rref(
    cf: tuple[int, ...], n: int, rng: random.Random
) -> tuple[int, ...]:
    """Return RREF of a random column-permutation of cf."""
    perm = list(range(n))
    rng.shuffle(perm)
    permuted = tuple(apply_permutation(b, perm) for b in cf)
    return Code(n, permuted).rref_basis()[0]


def _lookup_in_bucket(
    d_rref: tuple[int, ...],
    bucket: list[tuple[int, ...]],
    n: int,
    counters: IsoCounters,
) -> tuple[int, bool, float]:
    """Simulate the verifier lookup: scan bucket linearly, paired-iso each.

    Returns ``(compares, hit, wall_s)`` — compares is the number of
    paired_iso calls made (positions 0..pos inclusive, pos = match
    position; or full bucket length on miss); hit is whether a match
    was found; wall is the total perf_counter time spent in paired_iso.
    """
    t0 = time.perf_counter()
    compares = 0
    for cf in bucket:
        compares += 1
        if paired_iso(d_rref, cf, n, counters):
            wall = time.perf_counter() - t0
            return compares, True, wall
    wall = time.perf_counter() - t0
    return compares, False, wall


def _measure_paired_iso_canon_ratio(
    d_rref: tuple[int, ...], cf: tuple[int, ...], n: int
) -> tuple[float, float, IsoCounters]:
    """Time a single paired-iso call vs a single Feulner canon call on the
    same D. Used to compute the Python op-count-equivalent ratio that
    should translate to Rust."""
    counters = IsoCounters()
    t0 = time.perf_counter()
    paired_iso(d_rref, cf, n, counters)
    iso_wall = time.perf_counter() - t0
    code_d = Code(n, d_rref)
    t0 = time.perf_counter()
    canon_info_feulner(code_d)
    canon_wall = time.perf_counter() - t0
    return iso_wall, canon_wall, counters


def measure(N: int, perms_per_class: int, seed: int) -> dict:
    rng = random.Random(seed ^ (N << 4))

    print(f"\n=== N = {N} ===", flush=True)
    print(f"  enumerating classes via kernel...", flush=True)
    t0 = time.perf_counter()
    cfs = _enumerate_class_canonical_forms(N)
    enum_s = time.perf_counter() - t0
    print(f"    {len(cfs)} classes in {enum_s:.2f} s", flush=True)

    buckets = _bucket_by_weight_enum(cfs)
    bucket_sizes = sorted((len(v) for v in buckets.values()), reverse=True)
    print(f"  buckets: {len(buckets)} distinct weight enumerators; "
          f"sizes max={bucket_sizes[0]} mean={statistics.mean(bucket_sizes):.2f}",
          flush=True)

    # Per-class fixtures: random permutations of each canonical form.
    # Each D goes through paired-iso against its weight-enum bucket.
    print(f"  generating {perms_per_class} permutations per class...",
          flush=True)
    fixtures: list[tuple[tuple[int, ...], list[tuple[int, ...]]]] = []
    for cf in cfs:
        k = len(cf)
        we = _weight_multiset(cf, k)
        bucket = buckets[we]
        for _ in range(perms_per_class):
            d_rref = _random_perm_rref(cf, N, rng)
            fixtures.append((d_rref, bucket))
    print(f"    {len(fixtures)} (D, bucket) fixtures", flush=True)

    # Workload: scan paired-iso through the bucket. Aggregate compares,
    # hit rate, wall.
    print(f"  running paired-iso through buckets...", flush=True)
    counters = IsoCounters()
    total_compares = 0
    total_wall = 0.0
    hits = 0
    misses = 0
    pos_paired_iso_walls: list[float] = []
    neg_paired_iso_walls: list[float] = []

    sample_for_ratio = min(50, len(fixtures))
    ratio_pairs = rng.sample(range(len(fixtures)), sample_for_ratio)
    ratio_set = set(ratio_pairs)
    iso_canon_pairs: list[tuple[float, float, IsoCounters]] = []

    for i, (d_rref, bucket) in enumerate(fixtures):
        # Per-call: time each paired_iso individually to separate
        # positive (the eventual hit) from negatives (preceding misses
        # in the linear scan).
        t_total0 = time.perf_counter()
        match_pos = -1
        match_counters = IsoCounters()
        for j, cf in enumerate(bucket):
            c0 = IsoCounters()
            t0 = time.perf_counter()
            result = paired_iso(d_rref, cf, N, c0)
            t_wall = time.perf_counter() - t0
            if result:
                pos_paired_iso_walls.append(t_wall)
                match_counters = c0
                match_pos = j
                # Add c0 into the global counters.
                for f in (
                    "refines", "branches", "leaves", "prune_prefix",
                    "prune_shape", "prune_weight_strata",
                ):
                    setattr(
                        counters, f, getattr(counters, f) + getattr(c0, f),
                    )
                break
            neg_paired_iso_walls.append(t_wall)
            for f in (
                "refines", "branches", "leaves", "prune_prefix",
                "prune_shape", "prune_weight_strata",
            ):
                setattr(
                    counters, f, getattr(counters, f) + getattr(c0, f),
                )
        total_wall += time.perf_counter() - t_total0
        if match_pos >= 0:
            hits += 1
            total_compares += match_pos + 1
        else:
            misses += 1
            total_compares += len(bucket)
        if i in ratio_set:
            cf = bucket[match_pos] if match_pos >= 0 else bucket[0]
            iso_canon_pairs.append(
                _measure_paired_iso_canon_ratio(d_rref, cf, N)
            )

    return {
        "N": N,
        "n_classes": len(cfs),
        "n_buckets": len(buckets),
        "n_fixtures": len(fixtures),
        "max_bucket_size": bucket_sizes[0],
        "mean_bucket_size": statistics.mean(bucket_sizes),
        "hits": hits,
        "misses": misses,
        "total_compares": total_compares,
        "total_wall_s": total_wall,
        "pos_walls": pos_paired_iso_walls,
        "neg_walls": neg_paired_iso_walls,
        "iso_canon_pairs": iso_canon_pairs,
        "counters": counters,
    }


def fmt_us(s: float) -> str:
    return f"{s * 1e6:.1f} us"


def report(m: dict, phase1_nauty_us: float | None) -> None:
    N = m["N"]
    print(f"\n--- Phase 3 results for N = {N} ---")
    print(f"  classes: {m['n_classes']}  buckets: {m['n_buckets']}  "
          f"fixtures: {m['n_fixtures']}")
    print(f"  mean bucket size: {m['mean_bucket_size']:.2f}  "
          f"max: {m['max_bucket_size']}")
    print(f"  hit rate: {m['hits']} / {m['n_fixtures']} "
          f"({100 * m['hits'] / max(m['n_fixtures'], 1):.1f}%)")
    print(f"  total compares: {m['total_compares']}")
    print(f"  mean compares per lookup: "
          f"{m['total_compares'] / max(m['n_fixtures'], 1):.2f}")
    if m["pos_walls"]:
        print(f"  positive paired_iso wall (Python): "
              f"mean {fmt_us(statistics.mean(m['pos_walls']))} "
              f"median {fmt_us(statistics.median(m['pos_walls']))} "
              f"p95 {fmt_us(statistics.quantiles(m['pos_walls'], n=20)[-1])}")
    if m["neg_walls"]:
        print(f"  negative paired_iso wall (Python): "
              f"mean {fmt_us(statistics.mean(m['neg_walls']))} "
              f"median {fmt_us(statistics.median(m['neg_walls']))} "
              f"p95 {fmt_us(statistics.quantiles(m['neg_walls'], n=20)[-1])}")
    mean_pos = statistics.mean(m["pos_walls"]) if m["pos_walls"] else 0.0
    mean_neg = statistics.mean(m["neg_walls"]) if m["neg_walls"] else 0.0
    mean_lookup = m["total_wall_s"] / max(m["n_fixtures"], 1)
    print(f"  total paired-iso wall over all fixtures: {m['total_wall_s']:.2f} s")
    print(f"  mean wall per lookup (full bucket scan): {fmt_us(mean_lookup)}")
    c = m["counters"]
    n_calls = m["total_compares"]
    print(f"  paired-iso ops/call: refines={c.refines / n_calls:.2f}  "
          f"branches={c.branches / n_calls:.2f}  "
          f"leaves={c.leaves / n_calls:.2f}  "
          f"prune_prefix={c.prune_prefix / n_calls:.2f}  "
          f"prune_shape={c.prune_shape / n_calls:.2f}")

    if m["iso_canon_pairs"]:
        iso_walls = [p[0] for p in m["iso_canon_pairs"]]
        canon_walls = [p[1] for p in m["iso_canon_pairs"]]
        mean_iso = statistics.mean(iso_walls)
        mean_canon = statistics.mean(canon_walls)
        print(f"  iso vs canon (Python, same fixtures, n={len(iso_walls)}):")
        print(f"    paired_iso mean wall: {fmt_us(mean_iso)}")
        print(f"    canon_info_feulner mean wall: {fmt_us(mean_canon)}")
        if mean_canon > 0:
            print(f"    ratio iso/canon = {mean_iso / mean_canon:.3f}")
            print(f"    >>> Rust projection: at typical Rust scaling, "
                  f"iso ≈ {mean_iso / mean_canon:.3f}× a Rust nauty call")

    if phase1_nauty_us is not None:
        # Use Phase 1 numbers: hit rate ε on attempts, attempt rate α
        # on nauty calls. Approximated as ε=0.96, α=0.974 at N=22.
        epsilon = 0.96
        alpha = 0.974
        match_pos = 5.37  # Phase 1: mean match position at N=22
        bucket_mean = m["mean_bucket_size"]
        # Use ratio scaling: actual Rust per-iso ≈ mean_iso_wall × (canon_wall_rust / canon_wall_py)
        # canon_wall_rust ≈ phase1_nauty_us; canon_wall_py from iso_canon_pairs
        if m["iso_canon_pairs"]:
            mean_canon_py = statistics.mean(p[1] for p in m["iso_canon_pairs"])
            mean_iso_py = statistics.mean(p[0] for p in m["iso_canon_pairs"])
            iso_wall_rust_us = (mean_iso_py / mean_canon_py) * phase1_nauty_us
            print(f"\n  Projected Rust paired_iso per-call: "
                  f"{iso_wall_rust_us:.2f} us")
            verifier_hit_us = (match_pos + 1) * iso_wall_rust_us
            verifier_miss_us = bucket_mean * iso_wall_rust_us + phase1_nauty_us
            mean_verifier_us = (
                (1 - alpha) * phase1_nauty_us
                + alpha * (
                    epsilon * verifier_hit_us
                    + (1 - epsilon) * verifier_miss_us
                )
            )
            saved_us = phase1_nauty_us - mean_verifier_us
            savings_pct = 100 * saved_us / phase1_nauty_us
            print(f"  Projected per-call wall with verifier dispatch: "
                  f"{mean_verifier_us:.2f} us vs nauty {phase1_nauty_us:.2f} us")
            print(f"  >>> Projected savings of canon_info wall: "
                  f"{savings_pct:+.1f}%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--N",
        default="18,20",
        help="Comma-separated list of N (default: 18,20). "
        "N=22 in Python takes minutes.",
    )
    p.add_argument(
        "--perms-per-class",
        type=int,
        default=3,
        help="Number of random permutations per equivalence class.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed.",
    )
    p.add_argument(
        "--phase1-nauty-us",
        type=float,
        default=None,
        help="Mean nauty wall per call in µs from Phase 1 — used to "
        "project Rust savings. Pass 77.05 for N=22, 81.33 for N=20, "
        "70.29 for N=18.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    Ns = [int(s.strip()) for s in args.N.split(",") if s.strip()]
    for N in Ns:
        m = measure(N, args.perms_per_class, args.seed)
        report(m, args.phase1_nauty_us)
    return 0


if __name__ == "__main__":
    sys.exit(main())
