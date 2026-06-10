#!/usr/bin/env python3
"""Post-D15 per-phase scaling fit + N=29/32 extrapolation.

Consumes the `post-d15-profile-*` bench JSONs (which carry the per-rank
timing rows shipped in this profiling sprint: `per_k_stats.phi_ns`,
`.candidates_q_ns`, `.nauty_ns`) and produces:

1. A per-phase share table N=18..26 (canon / σ_Q / φ / other vs wall).
2. Per-phase growth ratios per 2-step of N, fitted on the measured range.
3. A post-D15 N=29 wall prediction for c4a-standard-72, cross-checked two
   ways against the pre-D15 actual (12.32 hr, 2026-05-23):
     (a) ratio method — scale the pre-D15 wall by the measured phase-share
         shift wall_new ≈ wall_old · [(1−f_c) + f_c·κ + φ_share], where
         f_c is the pre-D15 canon share (~88 %) and κ the kept-canon-ns
         fraction from the audit runs;
     (b) bottom-up — extrapolate each phase's ns by its fitted growth
         ratio from the N≤26 anchors, sum, divide by effective cores.
   The spread between (a) and (b) is reported as the error bar.
4. An N=32-on-288-cores feasibility sum using the same per-phase model.
5. The headline L1 question: what fraction of TOTAL extrapolated wall at
   N=29/32 sits in φ at child ranks k+1 ≥ 14, where the PhiScratch
   working set (7 B · 2^(k+1)) has outgrown a 48 KB L1d.

Field access is by NAME, never by stats-vector position, and tolerates
old JSONs that predate the φ fields (they are skipped with a notice).

Usage:
  uv run python scripts/experimental/post_d15_scaling_fit.py \
      --glob 'post-d15-profile-*' [--json out.json]
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "bench-results"

# Pre-D15 N=29 anchors (c4a-standard-72, 2026-05-23 run; see
# markdown/architecture/06-scaling-frontier.md and docs/results/n29.json).
PRE_D15_N29_WALL_H = 12.32
PRE_D15_N29_CANON_CALLS = 87.2e9  # legacy: one canon call per candidate tested
PRE_D15_N29_NS_PER_CALL = 29_500  # 29.5 us/call mean on c4a-72
PRE_D15_N29_CLASSES = 239_465_540
PRE_D15_CANON_SHARE = 0.88  # legacy-rule canon share of wall (measured ~88-92 %)
C4A72_CORES = 72
# 13700K single-P-core → c4a-72 single-core clock ratio measured on the
# N=28 deploy (per-call 37.7 µs vs ~30 µs-equivalent on 13700K ≈ 1.3;
# conservative band used below).
X86_TO_AXION_PER_CORE = 1.3
# Post-D15 parallel efficiency band measured this sprint (R5: worker
# active/wall ≈ 0.54 at N=26 t=24 d=5; d=4 recovers to ~0.65; assume
# partial seeder fix for a cloud run).
PAR_EFFICIENCY_BAND = (0.55, 0.85)


def load_runs(pattern: str) -> dict[int, dict]:
    """Median-aggregate bench JSONs by N. Returns
    {N: {wall_s, kernel_stats(median), per_k_stats(median)}}."""
    walls: dict[int, list[float]] = defaultdict(list)
    stats: dict[int, list[dict]] = defaultdict(list)
    perk: dict[int, list[dict]] = defaultdict(list)
    for f in sorted(globmod.glob(str(RESULTS / f"*{pattern}*.json"))):
        d = json.load(open(f))
        for N, r in d.get("per_N", {}).items():
            ks = r.get("kernel_stats") or {}
            if "phi_ns" not in ks:
                print(f"  [skip] {Path(f).name} N={N}: predates phi fields")
                continue
            walls[int(N)].append(r["seconds"])
            stats[int(N)].append(ks)
            if r.get("per_k_stats"):
                perk[int(N)].append(r["per_k_stats"])
    out: dict[int, dict] = {}
    for N in sorted(walls):
        med_i = sorted(range(len(walls[N])), key=lambda i: walls[N][i])[
            len(walls[N]) // 2
        ]
        out[N] = {
            "wall_s": walls[N][med_i],
            "kernel_stats": stats[N][med_i],
            "per_k_stats": perk[N][med_i] if perk[N] else {},
            "reps": len(walls[N]),
        }
    return out


def phase_table(runs: dict[int, dict]) -> dict[int, dict]:
    """Per-N phase ns + shares."""
    rows = {}
    for N, r in runs.items():
        wall_ns = r["wall_s"] * 1e9
        ks = r["kernel_stats"]
        canon, cq, phi = ks["nauty_ns"], ks["candidates_q_ns"], ks["phi_ns"]
        other = wall_ns - canon - cq - phi
        rows[N] = {
            "wall_s": r["wall_s"],
            "canon_ns": canon,
            "cq_ns": cq,
            "phi_ns": phi,
            "other_ns": other,
            "canon_pct": 100 * canon / wall_ns,
            "cq_pct": 100 * cq / wall_ns,
            "phi_pct": 100 * phi / wall_ns,
            "other_pct": 100 * other / wall_ns,
        }
    return rows


def growth_ratios(rows: dict[int, dict], key: str) -> list[float]:
    """ns ratio per 2-step of N over the measured ladder."""
    ns = sorted(rows)
    out = []
    for a, b in zip(ns, ns[1:]):
        if b - a == 2 and rows[a][key] > 0:
            out.append(rows[b][key] / rows[a][key])
    return out


def fit_tail_ratio(ratios: list[float]) -> float:
    """Growth ratio to carry past the measured range: weight the last two
    rungs (the asymptotic trend; small-N rungs are warmup-noise)."""
    tail = ratios[-2:] if len(ratios) >= 2 else ratios
    return statistics.geometric_mean(tail)


def extrapolate(rows: dict[int, dict], targets: list[int]) -> dict[int, dict]:
    """Carry each phase from the largest measured N by its fitted ratio.
    Returns {N: {phase: ns}} of single-thread 13700K-equivalent ns."""
    n_max = max(rows)
    fitted = {}
    for key in ("canon_ns", "cq_ns", "phi_ns", "other_ns"):
        ratios = growth_ratios(rows, key)
        fitted[key] = fit_tail_ratio(ratios)
    out = {}
    for t in targets:
        steps = (t - n_max) / 2
        out[t] = {
            key: rows[n_max][key] * (fitted[key] ** steps)
            for key in fitted
        }
        out[t]["ratios"] = {k: round(v, 3) for k, v in fitted.items()}
    return out


def phi_l1_overflow_share(runs: dict[int, dict], target_n: int) -> dict:
    """Of the extrapolated φ ns at target_n, what fraction sits at child
    ranks k+1 >= 14 (PhiScratch 7B·2^(k+1) > 48 KB L1d)?

    Method: per-k φ ns distributions shift right ~1 rank per 2-step of N
    (the candidate-count peak tracks k ≈ N/2 − 3). Take the largest
    measured per-k φ distribution, shift it by (target_n − N_meas)/2
    ranks, and integrate the k+1 ≥ 14 tail. Crude but honest — flagged
    as the model's weakest joint in the doc."""
    n_meas = max(N for N, r in runs.items() if r["per_k_stats"])
    pk = runs[n_meas]["per_k_stats"]["phi_ns"]
    shift = (target_n - n_meas) // 2
    total = sum(pk)
    if total == 0:
        return {"share": 0.0, "basis_N": n_meas, "shift": shift}
    # child rank = parent k + 1; overflow when parent k + 1 + shift >= 14
    tail = sum(v for k, v in enumerate(pk) if k + 1 + shift >= 14)
    return {"share": tail / total, "basis_N": n_meas, "shift": shift}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="post-d15-profile-")
    p.add_argument("--kappa", type=float, default=None,
                   help="kept-canon-ns fraction from d15_phi_audit (largest N)")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    runs = load_runs(args.glob)
    if not runs:
        print("no usable runs found"); return 1
    rows = phase_table(runs)

    print("\n== Sequential phase profile (median of reps, 13700K P-core) ==")
    print(f"{'N':>3} {'wall_s':>9} {'canon%':>7} {'sigmaQ%':>8} {'phi%':>6} {'other%':>7}")
    for N, r in sorted(rows.items()):
        print(f"{N:>3} {r['wall_s']:>9.3f} {r['canon_pct']:>6.1f}% "
              f"{r['cq_pct']:>7.1f}% {r['phi_pct']:>5.1f}% {r['other_pct']:>6.1f}%")

    print("\n== Per-phase growth ratios (per 2-step of N) ==")
    for key in ("canon_ns", "cq_ns", "phi_ns", "other_ns"):
        rs = growth_ratios(rows, key)
        print(f"  {key:<10} {' '.join(f'{x:6.2f}' for x in rs)}   tail-fit {fit_tail_ratio(rs):.2f}")

    ext = extrapolate(rows, [28, 29, 32])
    print("\n== Bottom-up extrapolation (single-thread 13700K-equivalent) ==")
    for N, e in ext.items():
        tot_h = sum(e[k] for k in ("canon_ns", "cq_ns", "phi_ns", "other_ns")) / 3.6e12
        print(f"  N={N}: total {tot_h:,.1f} core-hours "
              f"(canon {e['canon_ns']/3.6e12:,.1f}, sigmaQ {e['cq_ns']/3.6e12:,.1f}, "
              f"phi {e['phi_ns']/3.6e12:,.1f})")

    # N=29 on c4a-72. Three methods; (c) is the headline because it
    # anchors on the MEASURED candidate count at N=29 (87.2e9 legacy
    # canon calls = candidates tested) instead of extrapolated ratios.
    e29 = ext[29]
    seq_core_h = sum(e29[k] for k in ("canon_ns", "cq_ns", "phi_ns", "other_ns")) / 3.6e12
    bottom_up_h = seq_core_h * X86_TO_AXION_PER_CORE / C4A72_CORES
    print(f"\n== N=29 on c4a-standard-72 (pre-D15 actual: {PRE_D15_N29_WALL_H} h) ==")
    print(f"  (b) geometric carry (UNDERESTIMATE — per-2-step ratios still "
          f"accelerating): {bottom_up_h:,.1f} h")
    if args.kappa is not None:
        n_max = max(rows)
        phi_share_meas = rows[n_max]["phi_pct"] / 100
        f_c = PRE_D15_CANON_SHARE
        ratio = (1 - f_c) + f_c * args.kappa + phi_share_meas * f_c
        print(f"  (a) ratio method (OVERESTIMATE — prices phi at its share of "
              f"the post-D15 N=26 wall, not its N=29 cost): "
              f"{PRE_D15_N29_WALL_H * ratio:,.1f} h")

    # (c) count-anchored: every phase priced per measured event count.
    n_max = max(rows)
    ks26 = runs[n_max]["kernel_stats"]
    cand26 = sum(runs[n_max]["per_k_stats"]["candidates_total_seen"])
    phi_per_call_26 = rows[n_max]["phi_ns"] / cand26
    # phi/call grows ~1.5x per 2-step (0.64/0.81/1.14/1.82/2.74 ladder).
    phi_per_call_29 = phi_per_call_26 * 1.5 ** ((29 - n_max) / 2)
    phi_s = PRE_D15_N29_CANON_CALLS * phi_per_call_29 / 1e9
    # kept canon: kappa(N) falls (0.066@22, 0.039@24); band 0.02-0.04.
    canon_s = (PRE_D15_N29_CANON_CALLS * PRE_D15_N29_NS_PER_CALL / 1e9
               * (args.kappa if args.kappa else 0.03))
    # sigma_Q + other scale with the node count (one candidates_q call
    # per emitted node): classes ratio x flat-ish per-node cost.
    nodes26 = ks26["candidates_q_calls"]
    cq_s = rows[n_max]["cq_ns"] / 1e9 * (PRE_D15_N29_CLASSES / nodes26)
    other_s = rows[n_max]["other_ns"] / 1e9 * (PRE_D15_N29_CLASSES / nodes26)
    total_core_s = (phi_s + canon_s + cq_s + other_s) * X86_TO_AXION_PER_CORE
    lo = total_core_s / C4A72_CORES / PAR_EFFICIENCY_BAND[1] / 3600
    hi = total_core_s / C4A72_CORES / PAR_EFFICIENCY_BAND[0] / 3600
    print(f"  (c) count-anchored: {lo:,.1f}-{hi:,.1f} h wall "
          f"(phi {phi_s/3600:,.0f} + kept-canon {canon_s/3600:,.0f} + "
          f"sigmaQ {cq_s/3600:,.0f} + other {other_s/3600:,.0f} core-h at "
          f"{phi_per_call_29:,.0f} ns/cand x {PRE_D15_N29_CANON_CALLS:.3g} cands, "
          f"x{X86_TO_AXION_PER_CORE} clock, eff {PAR_EFFICIENCY_BAND[0]}-{PAR_EFFICIENCY_BAND[1]})")
    print(f"      => D15 N=29 speedup vs pre-D15 actual: "
          f"{PRE_D15_N29_WALL_H/hi:,.1f}-{PRE_D15_N29_WALL_H/lo:,.1f}x")

    for t in (29, 32):
        ov = phi_l1_overflow_share(runs, t)
        print(f"\n  phi L1-overflow share at N={t}: {100*ov['share']:.0f}% of phi time "
              f"at child rank >= 14 (basis N={ov['basis_N']}, shift {ov['shift']})")

    if args.json:
        args.json.write_text(json.dumps(
            {"rows": rows, "extrapolation": {str(k): v for k, v in ext.items()}},
            indent=2, default=float) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
