"""Per-rank head-to-head: Feulner vs nauty (full bipartite + Q_D) on
enumerated codes.

For each (N, k) bucket we time both `canon_info_native` (full bipartite,
i.e. the bail-fallback path) and `canon_info_feulner_native` on the same
RREF basis. Output: per-k median µs for both, plus the cost ratio.

The interesting question: at which k does Feulner beat the full-bipartite
fallback? If the crossover exists at low enough k to cover a meaningful
share of fallback calls (per `qd_size_profile.py` output), routing
Q_D-bail → Feulner could be a net win.

Requires the kernel built with `--features nauty_hist` only for the
profile script; this script uses no feature gates. Sample size is
capped at SAMPLE_PER_K per bucket to keep runtime bounded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import doubly_even_kernel as _kernel  # noqa: E402

from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402

SAMPLE_PER_K = 100  # cap samples per (N, k) bucket
WARMUP_ROUNDS = 2   # call each canonicaliser this many times before timing
REPS = 10           # repetitions per code, take median


def time_call(fn, *args) -> float:
    """Return median ns over REPS calls."""
    for _ in range(WARMUP_ROUNDS):
        fn(*args)
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter_ns()
        fn(*args)
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--sample", type=int, default=SAMPLE_PER_K)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n = args.N
    sample_cap = args.sample

    print(f"Enumerating doubly-even codes at N={n}...", flush=True)
    t0 = time.perf_counter()

    # Collect bases per rank, capped at sample_cap.
    by_rank: dict[int, list[list[int]]] = {}
    for ec in enumerate_doubly_even(n):
        k = ec.code.rank
        if k <= 1:
            continue  # rank-0/1 codes are degenerate
        bucket = by_rank.setdefault(k, [])
        if len(bucket) < sample_cap:
            bucket.append(list(ec.code.basis))

    enum_s = time.perf_counter() - t0
    total_samples = sum(len(v) for v in by_rank.values())
    print(f"  Enumerated in {enum_s:.1f}s. Sampled {total_samples} codes "
          f"across k={sorted(by_rank)}", flush=True)

    # Per-rank timing.
    per_rank: dict[str, dict] = {}
    for k in sorted(by_rank):
        codes = by_rank[k]
        full_bip_ns: list[float] = []
        qd_ns: list[float] = []
        feulner_ns: list[float] = []
        qd_fallbacks = 0
        for basis in codes:
            full_bip_ns.append(time_call(_kernel.canon_info_native, basis, n))
            qd_t = time_call(_kernel.canon_info_qd_native, basis, n)
            qd_ns.append(qd_t)
            # The qd_native PyO3 binding returns None on bail; we detect by
            # calling it once and checking. (The timing above includes that
            # call; bails are very cheap when they happen so this is OK.)
            try:
                res = _kernel.canon_info_qd_native(basis, n)
                if res is None:
                    qd_fallbacks += 1
            except Exception:
                pass
            feulner_ns.append(time_call(_kernel.canon_info_feulner_native, basis, n))

        per_rank[str(k)] = {
            "k": k,
            "samples": len(codes),
            "qd_fallbacks_in_sample": qd_fallbacks,
            "full_bip_us_p50": statistics.median(full_bip_ns) / 1e3,
            "qd_us_p50": statistics.median(qd_ns) / 1e3,
            "feulner_us_p50": statistics.median(feulner_ns) / 1e3,
            "full_bip_us_mean": statistics.fmean(full_bip_ns) / 1e3,
            "qd_us_mean": statistics.fmean(qd_ns) / 1e3,
            "feulner_us_mean": statistics.fmean(feulner_ns) / 1e3,
        }

    # Pretty-print.
    print(f"\n=== Per-call medians, N={n} ===")
    print(f"  {'k':>3} {'samples':>8} {'full_bip µs':>13} {'Q_D µs':>10} "
          f"{'Feulner µs':>12} {'Feu/fb':>8} {'Feu/qd':>8}")
    for kkey in sorted(per_rank, key=int):
        r = per_rank[kkey]
        fb = r["full_bip_us_p50"]
        qd = r["qd_us_p50"]
        fe = r["feulner_us_p50"]
        print(f"  {r['k']:>3} {r['samples']:>8} {fb:>13.1f} {qd:>10.1f} "
              f"{fe:>12.1f} {fe/fb if fb else 0:>8.2f} {fe/qd if qd else 0:>8.2f}",
              flush=True)

    out = {
        "N": n,
        "enum_s": enum_s,
        "sample_cap": sample_cap,
        "per_rank": per_rank,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
