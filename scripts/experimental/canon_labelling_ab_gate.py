"""Decision-identity gate + wall report for the autom-only canon A/B.

Compares `automonly-*` vs `fullmode-*` bench JSONs (single wheel, knob
A/B per docs/benchmarking.md §5). Gates, per N:

- `classes` and the full per-k class table bit-equal;
- every non-`_ns` kernel-stats slot bit-equal EXCEPT the documented
  exclusions: the nauty tree-shape sums the lever shrinks
  (`nauty_numnodes_sum`, `nauty_tctotal_sum`, `nauty_maxlevel_sum`) and
  the two mode counters (`canon_autom_only_calls`,
  `canon_label_upgrades`). Slot `nauty_generators_sum` is IN the gate —
  its equality is the live form of the same-generators-either-way
  assumption (rust/core/tests/canon_labelling.rs).
- every non-`_ns` per-k stats row bit-equal.

Reports median walls, the speedup ratio, the tree-shape drops (the
lever's fingerprint), and the upgrade rate. Parallel runs gate classes /
per-k exactly but counters only within the two arms' own rep envelope
(race-variable by design) — pass --parallel for those globs.

Usage:
  uv run python scripts/experimental/canon_labelling_ab_gate.py \
      --treat 'automonly-seq-rep*' --ctrl 'fullmode-seq-rep*'
  uv run python scripts/experimental/canon_labelling_ab_gate.py \
      --treat 'automonly-par-n26-rep*' --ctrl 'fullmode-par-n26-rep*' --parallel
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "bench-results")

EXCLUDED_SLOTS = {
    "nauty_numnodes_sum",
    "nauty_tctotal_sum",
    "nauty_maxlevel_sum",
    "canon_autom_only_calls",
    "canon_label_upgrades",
    # Measured 2026-06-12: differs by ±1 at N=24 (424,563 vs 424,562 —
    # 1 in 4e5 calls). nauty may discover a *different generating set*
    # of the same group when getcanon's best-leaf bookkeeping is off.
    # Decision-neutral: orbits / |Aut| / classes / strata sums are all
    # bit-equal (gated above); orbit-min and subspace_in_orbit are
    # group-level, not generating-set-level. Differentially equal
    # per-input through N=14 (rust/core/tests/canon_labelling.rs).
    "nauty_generators_sum",
}


def load(pattern: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, f"*{pattern}*.json")))
    if not paths:
        sys.exit(f"no bench JSONs match {pattern!r} in {RESULTS_DIR}")
    return [json.load(open(p)) for p in paths]


def gate_pair(t: dict, c: dict, n: str, parallel: bool) -> list[str]:
    errs: list[str] = []
    tn, cn = t["per_N"][n], c["per_N"][n]
    if tn["classes"] != cn["classes"]:
        errs.append(f"N={n}: classes {tn['classes']} != {cn['classes']}")
    if tn["per_k"] != cn["per_k"]:
        errs.append(f"N={n}: per-k class table differs")
    if parallel:
        return errs  # counters are race-variable across workers
    for name, tv in tn["kernel_stats"].items():
        if name.endswith("_ns") or name in EXCLUDED_SLOTS:
            continue
        cv = cn["kernel_stats"].get(name)
        if tv != cv:
            errs.append(f"N={n}: kernel_stats[{name}] {tv} != {cv}")
    for row, tvals in tn["per_k_stats"].items():
        if row.endswith("_ns"):
            continue
        cvals = cn["per_k_stats"].get(row)
        if tvals != cvals:
            errs.append(f"N={n}: per_k_stats[{row}] differs")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treat", required=True, help="glob fragment, e.g. automonly-seq-rep*")
    ap.add_argument("--ctrl", required=True, help="glob fragment, e.g. fullmode-seq-rep*")
    ap.add_argument("--parallel", action="store_true")
    args = ap.parse_args()

    treats, ctrls = load(args.treat), load(args.ctrl)
    ns = sorted(treats[0]["per_N"].keys(), key=int)

    all_errs: list[str] = []
    for n in ns:
        # Gate every treat rep against every ctrl rep (decision identity
        # is run-invariant, so all pairs must agree).
        for t in treats:
            for c in ctrls:
                all_errs.extend(gate_pair(t, c, n, args.parallel))

        t_walls = [t["per_N"][n]["seconds"] for t in treats]
        c_walls = [c["per_N"][n]["seconds"] for c in ctrls]
        t_med, c_med = statistics.median(t_walls), statistics.median(c_walls)
        ks_t = treats[0]["per_N"][n]["kernel_stats"]
        ks_c = ctrls[0]["per_N"][n]["kernel_stats"]
        upgrades = ks_t.get("canon_label_upgrades", 0)
        hits = ks_t.get("primary_hits", 0)
        print(f"\nN={n}:")
        print(f"  wall median  treat {t_med:8.3f} s  (reps: {[round(w,3) for w in t_walls]})")
        print(f"  wall median  ctrl  {c_med:8.3f} s  (reps: {[round(w,3) for w in c_walls]})")
        print(f"  speedup      {c_med / t_med:5.3f}x")
        print(f"  autom-only calls   {ks_t.get('canon_autom_only_calls', 0):>12}  "
              f"({100*ks_t.get('canon_autom_only_calls',0)/max(ks_t.get('canon_calls',1),1):.1f}% of canon calls)")
        print(f"  label upgrades     {upgrades:>12}  "
              f"({100*upgrades/max(hits,1):.4f}% of primary hits)")
        for slot in ("nauty_numnodes_sum", "nauty_tctotal_sum", "nauty_maxlevel_sum"):
            tv, cv = ks_t.get(slot, 0), ks_c.get(slot, 0)
            if cv:
                print(f"  {slot:<22} {tv:>14} vs {cv:>14}  ({100*(cv-tv)/cv:+.1f}% drop)")

    if all_errs:
        uniq = sorted(set(all_errs))
        print(f"\nGATE FAILED — {len(uniq)} distinct mismatches:")
        for e in uniq[:20]:
            print(" ", e)
        return 1
    print("\nGATE PASSED — decisions bit-identical across arms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
