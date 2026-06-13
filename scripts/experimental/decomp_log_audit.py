"""Lever-4 measurement: decomposability + twin-column structure of canon-call
inputs, rank-weighted by measured nauty time (bottlenecks §4 lever 4;
external-feedback P1/P2).

Runs the sequential kernel with ``DOUBLY_EVEN_DECOMP_LOG`` set: one JSONL
record per `test_candidate` canon-info consultation —

    {"n":26,"k":7,"out":"acc","ns":71234,"miss":1,"sup":24,
     "comp":[20,4],"tw":[2,2]}

where ``ns`` is the nauty time that consultation cost (0 ⇔ cache hit),
``sup`` the column support, ``comp`` the direct-sum component sizes within
the support (RREF row-support connectivity — exact, see
``rust/core/src/enumerate/decomp.rs``), ``tw`` the twin-column class sizes
(≥ 2 only).

Decision quantities (promotion threshold for the P1/P2 implementation
sprint: ≥ 30 % of rank-weighted nauty time on decomposable inputs):

- ``decomposable`` := ``sup < n`` (zero columns ⇒ trivially C' ⊕ 0) OR
  ``len(comp) ≥ 2`` (proper direct sum within the support). Reported
  jointly and split, since the win route differs: zero-column inputs
  re-key to the support-restricted code; proper sums need component
  canonicalisation + wreath-product |Aut| assembly.
- ``twin-bearing`` := any twin class (the P2 pre-nauty compression target);
  ``Σ(size−1)`` columns would be removed.
- Coverage: Σ logged ns / kernel per-k ``nauty_ns`` total — confirms the
  log captures ~all canon time (the root call and label upgrades are
  outside `test_candidate`; both are negligible).

Behaviour-neutral by construction: the hook only reads the RREF and two
existing counters. Decision identity is implied by the knob-off path being
untouched; the classes count printed per run doubles as the cross-check.

Usage:
    uv run python scripts/experimental/decomp_log_audit.py
    uv run python scripts/experimental/decomp_log_audit.py --Ns 20 24 --keep-logs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import doubly_even_kernel as _kernel
from doubly_even.spec.mass import gaborit_sigma

# DFGHILM Table 3 totals (all ranks) — the decision-identity cross-check.
EXPECTED_CLASSES = {16: 146, 18: 341, 20: 1211, 22: 5118, 24: 37496, 26: 494272}


def run_with_log(N: int, log_path: Path) -> tuple[int, dict[int, int], float]:
    """Sequential enumeration with the decomp log attached.

    Returns (classes, per-rank nauty_ns from the kernel stats, wall s).
    """
    if os.environ.get("DOUBLY_EVEN_THREADS"):
        raise SystemExit("unset DOUBLY_EVEN_THREADS: the decomp log is sequential-only")
    os.environ["DOUBLY_EVEN_DECOMP_LOG"] = str(log_path)
    try:
        cap = N // 2
        quota = [gaborit_sigma(N, k) for k in range(cap + 1)]
        t0 = time.perf_counter()
        raw, stats, per_k = _kernel.enumerate_doubly_even(
            N, cap, quota, math.factorial(N), None
        )
        wall = time.perf_counter() - t0
    finally:
        os.environ.pop("DOUBLY_EVEN_DECOMP_LOG", None)

    _, per_k_names = _kernel.kernel_stats_layout()
    nauty_row = per_k_names.index("nauty_ns")
    nauty_ns_by_k = {k: int(v) for k, v in enumerate(per_k[nauty_row])}
    return len(raw), nauty_ns_by_k, wall


def analyse(N: int, log_path: Path, nauty_ns_by_k: dict[int, int]) -> dict:
    per_k: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "calls": 0,
            "misses": 0,
            "ns": 0,
            "ns_decomp": 0,
            "ns_proper": 0,
            "ns_zerocol": 0,
            "ns_twin": 0,
            "calls_decomp": 0,
            "calls_twin": 0,
        }
    )
    with log_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            k = r["k"]
            row = per_k[k]
            row["calls"] += 1
            row["misses"] += r["miss"]
            row["ns"] += r["ns"]
            zerocol = r["sup"] < r["n"]
            proper = len(r["comp"]) >= 2
            twin = bool(r["tw"])
            if zerocol or proper:
                row["calls_decomp"] += 1
                row["ns_decomp"] += r["ns"]
            if proper:
                row["ns_proper"] += r["ns"]
            if zerocol:
                row["ns_zerocol"] += r["ns"]
            if twin:
                row["calls_twin"] += 1
                row["ns_twin"] += r["ns"]

    tot = {key: sum(row[key] for row in per_k.values()) for key in next(iter(per_k.values()))}
    kernel_total_ns = sum(nauty_ns_by_k.values())

    print(f"\n== N={N} per-rank breakdown (ns = measured nauty time on that call) ==")
    hdr = (
        f"{'k':>3} {'calls':>9} {'misses':>9} {'nauty_ms':>10} "
        f"{'%ns_dec':>8} {'%ns_prop':>9} {'%ns_zcol':>9} {'%ns_twin':>9} {'%calls_dec':>11}"
    )
    print(hdr)
    for k in sorted(per_k):
        row = per_k[k]
        ns = row["ns"] or 1
        print(
            f"{k:>3} {int(row['calls']):>9} {int(row['misses']):>9} "
            f"{row['ns'] / 1e6:>10.1f} "
            f"{100 * row['ns_decomp'] / ns:>8.1f} {100 * row['ns_proper'] / ns:>9.1f} "
            f"{100 * row['ns_zerocol'] / ns:>9.1f} {100 * row['ns_twin'] / ns:>9.1f} "
            f"{100 * row['calls_decomp'] / max(row['calls'], 1):>11.1f}"
        )

    frac_dec = tot["ns_decomp"] / max(tot["ns"], 1)
    frac_proper = tot["ns_proper"] / max(tot["ns"], 1)
    frac_zerocol = tot["ns_zerocol"] / max(tot["ns"], 1)
    frac_twin = tot["ns_twin"] / max(tot["ns"], 1)
    coverage = tot["ns"] / max(kernel_total_ns, 1)
    print(f"\n  rank-weighted nauty time on decomposable inputs: {100 * frac_dec:.1f} %")
    print(f"    … proper direct sums (≥2 components in support): {100 * frac_proper:.1f} %")
    print(f"    … zero columns only (support < N):               {100 * frac_zerocol:.1f} %")
    print(f"  rank-weighted nauty time on twin-bearing inputs:   {100 * frac_twin:.1f} %")
    print(f"  log coverage of kernel nauty_ns: {100 * coverage:.1f} % "
          f"({tot['ns'] / 1e6:.0f} of {kernel_total_ns / 1e6:.0f} ms)")
    return {
        "N": N,
        "frac_ns_decomposable": frac_dec,
        "frac_ns_proper": frac_proper,
        "frac_ns_zerocol": frac_zerocol,
        "frac_ns_twin": frac_twin,
        "coverage": coverage,
        "total_calls": int(tot["calls"]),
        "total_misses": int(tot["misses"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--Ns", type=int, nargs="+", default=[24, 26])
    ap.add_argument("--keep-logs", action="store_true",
                    help="keep the JSONL logs (default: temp files, deleted)")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="directory for kept logs (implies --keep-logs)")
    args = ap.parse_args()

    keep = args.keep_logs or args.log_dir is not None
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="decomp-log-"))
    log_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for N in args.Ns:
        log_path = log_dir / f"decomp-n{N}.jsonl"
        log_path.unlink(missing_ok=True)
        classes, nauty_ns_by_k, wall = run_with_log(N, log_path)
        expected = EXPECTED_CLASSES.get(N)
        tag = ""
        if expected is not None:
            assert classes == expected, f"N={N}: {classes} classes != Table 3 {expected}"
            tag = " (matches DFGHILM Table 3)"
        print(f"N={N}: {classes} classes{tag}, wall {wall:.1f} s, log {log_path}")
        summaries.append(analyse(N, log_path, nauty_ns_by_k))
        if not keep:
            log_path.unlink(missing_ok=True)

    print("\n== VERDICT (promotion threshold: ≥ 30 % of rank-weighted nauty time) ==")
    for s in summaries:
        verdict = "PROMOTE" if s["frac_ns_decomposable"] >= 0.30 else "below threshold"
        print(
            f"  N={s['N']}: decomposable {100 * s['frac_ns_decomposable']:.1f} % "
            f"(proper {100 * s['frac_ns_proper']:.1f} %, zero-col "
            f"{100 * s['frac_ns_zerocol']:.1f} %), twins {100 * s['frac_ns_twin']:.1f} % "
            f"→ {verdict}"
        )


if __name__ == "__main__":
    main()
