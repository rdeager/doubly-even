"""Phase 1: measure the σ_Q-orbit-min → canon-test rejection rate.

For each N in {14, 16, 18, 20, 22}, run the Rust kernel's
`enumerate_doubly_even` and surface the per-k breakdown of the
`is_canonical_augmentation` outcomes:

- `parent_eq_hits` — fast-path accept (canonical_parent(D) == C verbatim).
- `bfs_hits` — BFS in Aut(D)-orbit found C.
- `bfs_rejects` — BFS exhausted (or trivial Aut(D) and parent ≠ C).
- `weight_enum_filtered` — early reject via weight-enumerator mismatch.

The "rejection rate" for the audit is
`bfs_rejects / is_canon_aug_calls` (the fraction of σ_Q-orbit-min
survivors that the canon test ultimately rejects).

See plan `for-complete-enumeration-of-proud-meerkat.md` Phase 1.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import doubly_even_kernel as _kernel
from doubly_even.spec.mass import gaborit_sigma


def measure(N: int) -> dict:
    cap = N // 2
    quota_vec = [gaborit_sigma(N, k) for k in range(cap + 1)]
    factorial_N = math.factorial(N)

    t0 = time.perf_counter()
    raw, stats, per_k = _kernel.enumerate_doubly_even(N, cap, quota_vec, factorial_N)
    wall_s = time.perf_counter() - t0

    # Scalar stats (idx 21 is the new bfs_rejects total).
    is_canon_aug_calls = int(stats[4])
    parent_eq_hits = int(stats[5])
    weight_enum_filtered = int(stats[6])
    bfs_calls = int(stats[7])
    bfs_hits = int(stats[8])
    bfs_rejects = int(stats[21])

    # Sanity: parent_eq + bfs_hits + weight_enum_filtered + bfs_rejects ==
    # is_canon_aug_calls (the only four exits from is_canonical_augmentation).
    accounted = parent_eq_hits + weight_enum_filtered + bfs_hits + bfs_rejects
    if accounted != is_canon_aug_calls:
        raise RuntimeError(
            f"N={N}: accounted={accounted} != is_canon_aug_calls={is_canon_aug_calls}; "
            f"counters drifted (parent_eq={parent_eq_hits}, we_filtered={weight_enum_filtered}, "
            f"bfs_hits={bfs_hits}, bfs_rejects={bfs_rejects})"
        )

    # per_k rows: see enumerate.rs enumerate_doubly_even doc.
    per_k_rows = [
        "is_canon_aug_calls",
        "parent_eq_hits",
        "weight_enum_filtered",
        "bfs_calls",
        "bfs_hits",
        "bfs_rejects",
        "mass_stop_pre_loop",
        "mass_stop_in_loop",
        "candidates_total_seen",
        "candidates_skipped",
    ]
    per_k_table = {row: [int(v) for v in per_k[i]] for i, row in enumerate(per_k_rows)}

    return {
        "N": N,
        "cap": cap,
        "classes": len(raw),
        "wall_s": wall_s,
        "totals": {
            "is_canon_aug_calls": is_canon_aug_calls,
            "parent_eq_hits": parent_eq_hits,
            "weight_enum_filtered": weight_enum_filtered,
            "bfs_calls": bfs_calls,
            "bfs_hits": bfs_hits,
            "bfs_rejects": bfs_rejects,
            "rejection_rate": bfs_rejects / is_canon_aug_calls
            if is_canon_aug_calls
            else 0.0,
        },
        "per_k": per_k_table,
    }


def report(m: dict) -> None:
    N = m["N"]
    t = m["totals"]
    calls = t["is_canon_aug_calls"]

    def pct(x: int) -> str:
        return f"{100 * x / calls:5.1f}%" if calls else "  n/a"

    print(f"\n=== N = {N} (cap k ≤ {m['cap']}) ===")
    print(f"  emitted classes:         {m['classes']:>10}")
    print(f"  wall:                    {m['wall_s']:>10.3f} s")
    print(f"  is_canon_aug calls:      {calls:>10}")
    print(f"    parent_eq_hits:        {t['parent_eq_hits']:>10}  ({pct(t['parent_eq_hits'])})")
    print(f"    weight_enum_filtered:  {t['weight_enum_filtered']:>10}  ({pct(t['weight_enum_filtered'])})")
    print(f"    bfs_hits:              {t['bfs_hits']:>10}  ({pct(t['bfs_hits'])})")
    print(f"    bfs_rejects:           {t['bfs_rejects']:>10}  ({pct(t['bfs_rejects'])})  <<< rejection rate")
    print(f"    --- of bfs_calls only: {t['bfs_calls']} hits={t['bfs_hits']} "
          f"(hit rate {100 * t['bfs_hits'] / t['bfs_calls']:.1f}% if bfs_calls>0)"
          if t["bfs_calls"] else "    --- bfs_calls = 0")

    # Per-k breakdown.
    print(f"  --- per parent rank k ---")
    print(f"   k | is_aug    parent_eq  we_filt   bfs_hit   bfs_rej   rej_rate")
    pk = m["per_k"]
    cap = m["cap"]
    for k in range(cap + 1):
        c = pk["is_canon_aug_calls"][k]
        if c == 0:
            continue
        r = pk["bfs_rejects"][k]
        print(
            f"  {k:>2} | {c:>7}  {pk['parent_eq_hits'][k]:>9}  "
            f"{pk['weight_enum_filtered'][k]:>7}  {pk['bfs_hits'][k]:>8}  "
            f"{r:>8}   {100 * r / c:>5.1f}%"
        )

    # Mass-stop effectiveness.
    print(f"  --- mass-stop per parent rank k ---")
    print(f"   k | pre_loop  in_loop  total_cands  skipped   skip%")
    for k in range(cap + 1):
        ms_pre = pk["mass_stop_pre_loop"][k]
        ms_in = pk["mass_stop_in_loop"][k]
        total = pk["candidates_total_seen"][k]
        skipped = pk["candidates_skipped"][k]
        if total == 0 and ms_pre == 0:
            continue
        skip_pct = 100 * skipped / total if total else 0.0
        print(
            f"  {k:>2} | {ms_pre:>8}  {ms_in:>7}  {total:>11}  {skipped:>7}   {skip_pct:>5.1f}%"
        )
    tot_skipped = sum(pk["candidates_skipped"])
    tot_seen = sum(pk["candidates_total_seen"])
    overall = 100 * tot_skipped / tot_seen if tot_seen else 0.0
    print(f"  OVERALL: skipped {tot_skipped}/{tot_seen} candidates ({overall:.2f}%)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Ns", type=int, nargs="+", default=[14, 16, 18, 20, 22])
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path; default is scripts/bench-results/bfs-rejects-<timestamp>.json.",
    )
    args = p.parse_args()

    print(f"kernel build: {_kernel.kernel_build_info()}")

    measurements = []
    for N in args.Ns:
        m = measure(N)
        report(m)
        measurements.append(m)

    out_path: Path
    if args.out is not None:
        out_path = args.out
    else:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        results_dir = Path(__file__).parent / "bench-results"
        results_dir.mkdir(exist_ok=True)
        out_path = results_dir / f"bfs-rejects-{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "kernel_build": _kernel.kernel_build_info(),
                "measurements": measurements,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
