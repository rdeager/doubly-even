"""D15 Phase 1 gate: measure the coset-spectrum parent rule against legacy.

Runs the kernel in audit mode (`DOUBLY_EVEN_PARENT_RULE=audit`): the φ
cascade is evaluated for every candidate alongside the unchanged legacy
rule, tallying outcomes and the canon nanoseconds the φ rule would keep.
Enumeration behaviour is byte-identical to legacy; only counters move.

Decision quantities (see plan, "Phase 1 — measurement gate"):

- HARD soundness gate (no-mass-stop runs): per parent rank k,
      phi_accept_unique + phi_tie_accept == parent_eq_hits + bfs_hits
  must hold EXACTLY. Both rules provably accept exactly one
  (parent-class, coset-orbit) pair per class, and the no-mass-stop legacy
  tree tests every pair, so the totals must match. Any deviation = bug or
  unsoundness; stop.
  (With mass-stop ON the identity may drift: skipped candidates can
  include the φ-chosen parent of a class legacy already emitted via a
  different parent. Reported, not gated.)

- κ = nauty_ns_kept / nauty_ns — the fraction of today's canon time the
  φ rule retains (accepts + ties), measured in ns so systematically
  expensive kept calls are priced. Projection:
      wall_new/wall_old ≈ (1 − f_c) + f_c·κ + φ_frac,
  f_c = nauty_ns / wall (sequential), φ_frac = phi_ns / wall.

- GO:    κ ≤ 0.25 and mean φ cost ≤ 5 µs at N=24  (≈ 2.5× at N=26)
- NO-GO: κ > 0.42 at N=24 with no per-rank-cap fix  (< 1.6×)
- else gray zone: tune DOUBLY_EVEN_PHI_MAX_RANK from the per-k table.

Usage:
    uv run python scripts/experimental/d15_phi_audit.py
    uv run python scripts/experimental/d15_phi_audit.py --Ns 18 20 22 --skip-parallel
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import doubly_even_kernel as _kernel
from doubly_even.spec.mass import gaborit_sigma

PER_K_ROWS = [
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
    "phi_reject",
    "phi_accept_unique",
    "phi_tie_accept",
    "phi_tie_reject",
]


def run_audit(N: int, *, no_mass_stop: bool, num_threads: int | None) -> dict:
    cap = N // 2
    quota_vec = [gaborit_sigma(N, k) for k in range(cap + 1)]
    factorial_N = math.factorial(N)

    os.environ["DOUBLY_EVEN_PARENT_RULE"] = "audit"
    if no_mass_stop:
        os.environ["DOUBLY_EVEN_NO_MASS_STOP"] = "1"
    else:
        os.environ.pop("DOUBLY_EVEN_NO_MASS_STOP", None)
    if num_threads:
        os.environ["DOUBLY_EVEN_FRONTIER_DEPTH"] = "5"

    t0 = time.perf_counter()
    raw, stats, per_k = _kernel.enumerate_doubly_even(
        N, cap, quota_vec, factorial_N, num_threads
    )
    wall_s = time.perf_counter() - t0

    per_k_table = {row: [int(v) for v in per_k[i]] for i, row in enumerate(PER_K_ROWS)}

    phi_reject = int(stats[26])
    phi_accept_unique = int(stats[27])
    phi_tie_accept = int(stats[28])
    phi_tie_reject = int(stats[29])
    phi_ns = int(stats[30])
    phi_candidates = phi_reject + phi_accept_unique + phi_tie_accept + phi_tie_reject
    nauty_ns = int(stats[11])
    nauty_ns_kept = int(stats[33])

    kappa = nauty_ns_kept / nauty_ns if nauty_ns else 0.0
    legacy_accepts = int(stats[5]) + int(stats[8])  # parent_eq_hits + bfs_hits
    phi_accepts = phi_accept_unique + phi_tie_accept

    # Hard gate, per rank (meaningful on no-mass-stop runs).
    gate_per_k_ok = True
    gate_rows = []
    for k in range(cap + 1):
        phi_acc_k = (
            per_k_table["phi_accept_unique"][k] + per_k_table["phi_tie_accept"][k]
        )
        legacy_acc_k = per_k_table["parent_eq_hits"][k] + per_k_table["bfs_hits"][k]
        ok = phi_acc_k == legacy_acc_k
        gate_per_k_ok &= ok
        gate_rows.append(
            {"k": k, "phi_accepts": phi_acc_k, "legacy_accepts": legacy_acc_k, "ok": ok}
        )

    return {
        "N": N,
        "cap": cap,
        "num_threads": num_threads or 0,
        "no_mass_stop": no_mass_stop,
        "classes": len(raw),
        "wall_s": wall_s,
        "totals": {
            "candidates_phi_evaluated": phi_candidates,
            "phi_reject": phi_reject,
            "phi_accept_unique": phi_accept_unique,
            "phi_tie_accept": phi_tie_accept,
            "phi_tie_reject": phi_tie_reject,
            "phi_ns": phi_ns,
            "phi_mean_us": phi_ns / phi_candidates / 1e3 if phi_candidates else 0.0,
            "phi_strata_sum": int(stats[31]),
            "phi_m_size_sum": int(stats[32]),
            "nauty_ns": nauty_ns,
            "nauty_ns_kept": nauty_ns_kept,
            "kappa": kappa,
            "legacy_accepts": legacy_accepts,
            "phi_accepts": phi_accepts,
            "is_canon_aug_calls": int(stats[4]),
            "canon_calls": int(stats[0]),
            "wall_ns": int(wall_s * 1e9),
        },
        "gate_per_k_ok": gate_per_k_ok,
        "gate_rows": gate_rows,
        "per_k": per_k_table,
    }


def report(m: dict) -> None:
    t = m["totals"]
    cand = t["candidates_phi_evaluated"]
    mode = f"t={m['num_threads']}" if m["num_threads"] else "seq"
    ms = "mass-stop OFF" if m["no_mass_stop"] else "mass-stop ON"
    print(f"\n=== N={m['N']} ({mode}, {ms}) ===")
    print(f"  wall {m['wall_s']:.3f} s   classes {m['classes']}   φ-evaluated {cand}")

    def pct(x: int) -> str:
        return f"{100 * x / cand:5.1f}%" if cand else "  n/a"

    print(
        f"  φ outcomes: reject {t['phi_reject']} ({pct(t['phi_reject'])})"
        f"  accept_unique {t['phi_accept_unique']} ({pct(t['phi_accept_unique'])})"
        f"  tie_accept {t['phi_tie_accept']} ({pct(t['phi_tie_accept'])})"
        f"  tie_reject {t['phi_tie_reject']} ({pct(t['phi_tie_reject'])})"
    )
    if cand:
        print(
            f"  φ cost: mean {t['phi_mean_us']:.2f} µs"
            f"  (strata mean {t['phi_strata_sum'] / cand:.2f},"
            f" |M| mean {t['phi_m_size_sum'] / cand:.1f})"
        )
    f_c = t["nauty_ns"] / t["wall_ns"] if m["num_threads"] == 0 else float("nan")
    phi_frac = t["phi_ns"] / t["wall_ns"] if m["num_threads"] == 0 else float("nan")
    print(
        f"  κ = {t['kappa']:.4f}  (nauty kept {t['nauty_ns_kept'] / 1e9:.3f} s"
        f" of {t['nauty_ns'] / 1e9:.3f} s)"
    )
    if m["num_threads"] == 0:
        proj = (1 - f_c) + f_c * t["kappa"] + phi_frac
        print(
            f"  f_c = {f_c:.3f}  φ_frac = {phi_frac:.4f}"
            f"  → projected wall ratio {proj:.3f} ({1 / proj:.2f}× speedup)"
        )
    gate = "PASS" if m["gate_per_k_ok"] else "FAIL"
    note = "" if m["no_mass_stop"] else " (advisory only with mass-stop ON)"
    print(f"  accept-identity gate: {gate}{note}"
          f"  [φ {t['phi_accepts']} vs legacy {t['legacy_accepts']}]")
    if not m["gate_per_k_ok"]:
        for row in m["gate_rows"]:
            if not row["ok"]:
                print(
                    f"    k={row['k']}: φ {row['phi_accepts']}"
                    f" != legacy {row['legacy_accepts']}"
                )
    # Per-k κ-relevant table.
    pk = m["per_k"]
    print("   k | cands_φ  reject  acc_uniq tie_acc tie_rej | kept%")
    for k in range(m["cap"] + 1):
        ck = (
            pk["phi_reject"][k]
            + pk["phi_accept_unique"][k]
            + pk["phi_tie_accept"][k]
            + pk["phi_tie_reject"][k]
        )
        if ck == 0:
            continue
        kept = ck - pk["phi_reject"][k]
        print(
            f"  {k:2d} | {ck:8d} {pk['phi_reject'][k]:7d} {pk['phi_accept_unique'][k]:8d}"
            f" {pk['phi_tie_accept'][k]:7d} {pk['phi_tie_reject'][k]:7d}"
            f" | {100 * kept / ck:5.1f}%"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Ns", type=int, nargs="+", default=[18, 20, 22, 24])
    p.add_argument("--threads", type=int, default=24)
    p.add_argument(
        "--skip-parallel",
        action="store_true",
        help="Skip the parallel N=24 validation runs.",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    print(f"kernel build: {_kernel.kernel_build_info()}")
    runs: list[dict] = []
    hard_gate_ok = True
    for N in args.Ns:
        for no_ms in (True, False):
            m = run_audit(N, no_mass_stop=no_ms, num_threads=None)
            report(m)
            runs.append(m)
            if no_ms:
                hard_gate_ok &= m["gate_per_k_ok"]
        if N >= 24 and not args.skip_parallel:
            m = run_audit(N, no_mass_stop=True, num_threads=args.threads)
            report(m)
            runs.append(m)
            hard_gate_ok &= m["gate_per_k_ok"]

    # Verdict from the largest sequential no-mass-stop run.
    gate_runs = [r for r in runs if r["no_mass_stop"] and r["num_threads"] == 0]
    decisive = max(gate_runs, key=lambda r: r["N"])
    kappa = decisive["totals"]["kappa"]
    phi_us = decisive["totals"]["phi_mean_us"]
    if not hard_gate_ok:
        verdict = "HARD-GATE-FAIL"
    elif kappa <= 0.25 and phi_us <= 5.0:
        verdict = "GO"
    elif kappa > 0.42:
        verdict = "NO-GO"
    else:
        verdict = "GRAY-ZONE"
    print(
        f"\nVERDICT: {verdict}  (hard gate {'OK' if hard_gate_ok else 'FAILED'};"
        f" decisive N={decisive['N']}: κ={kappa:.4f}, mean φ {phi_us:.2f} µs)"
    )

    out_path = args.out
    if out_path is None:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        results_dir = Path(__file__).resolve().parent.parent / "bench-results"
        results_dir.mkdir(exist_ok=True)
        out_path = results_dir / f"d15-phi-audit-{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "kernel_build": _kernel.kernel_build_info(),
                "verdict": verdict,
                "hard_gate_ok": hard_gate_ok,
                "runs": runs,
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    return 0 if verdict in ("GO", "GRAY-ZONE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
