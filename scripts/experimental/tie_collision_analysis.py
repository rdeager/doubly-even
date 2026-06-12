"""Analyse DOUBLY_EVEN_TIE_DUMP JSONL files — the φ-invariant collisions.

A tie record holds a parent C (rref, hex rows), a candidate v, the tied
argmin hyperplane functionals `m_set` (u16, over the coordinates of
D = <C, v>: coordinate j < k is basis row j of C, coordinate k is v),
the canon tie-break outcome, |Aut(D)|, and `tie_orbits` — the partition
of `m_set` into Aut(D)-orbits, computed exactly in the kernel.

Classification:
  * single-orbit tie (len(tie_orbits) == 1): the tied strata are
    Aut(D)-equivalent — the invariant tie is benign; ANY choice of
    stratum would be canonical, the canon call just confirms it.
  * multi-orbit tie (len(tie_orbits) >= 2): a TRUE invariant collision —
    pairwise inequivalent codimension-1 subcodes of D share the same
    complement-coset weight spectrum (the Milenkovic-2005 phenomenon
    inside a live classification).

Usage:
  uv run python scripts/experimental/tie_collision_analysis.py ties-n*.jsonl
  uv run python scripts/experimental/tie_collision_analysis.py --examples 3 ties-n24.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict


def coset_spectrum(rows: list[int], u: int) -> tuple[int, ...]:
    """Weight spectrum of the complement coset {x in D : u·coords(x) = 1}.

    `rows` = coordinates basis of D (parent rref rows then v); this is
    the stratum-1 layer of the kernel's φ invariant — tied strata agree
    on it (and on every deeper layer the cascade evaluated).
    """
    kp1 = len(rows)
    spec = Counter()
    for x in range(1, 1 << kp1):
        if bin(u & x).count("1") % 2 == 1:
            w = 0
            for j in range(kp1):
                if (x >> j) & 1:
                    w ^= rows[j]
            spec[bin(w).count("1")] += 1
    return tuple(spec[w] for w in range(max(spec) + 1)) if spec else ()


def hyperplane_rref(rows: list[int], u: int, n: int) -> list[int]:
    """RREF basis of ker(u) in D — mirrors parent_rule::hyperplane_basis."""
    kp1 = len(rows)
    j0 = (u & -u).bit_length() - 1
    basis = []
    for j in range(kp1):
        if j == j0:
            continue
        w = rows[j]
        if (u >> j) & 1:
            w ^= rows[j0]
        basis.append(w)
    # plain GF(2) row reduction
    out = []
    for w in basis:
        for p in out:
            w = min(w, w ^ p)
        if w:
            out.append(w)
            out.sort(reverse=True)
    return sorted(out, reverse=True)


def fmt_vec(w: int, n: int) -> str:
    return "".join("1" if (w >> i) & 1 else "." for i in range(n))


def show_example(rec: dict, idx: int) -> None:
    n = rec["n"]
    rows = [int(r, 16) for r in rec["parent_rref"]] + [int(rec["v"], 16)]
    kp1 = len(rows)
    print(f"\n--- example {idx}: N={n}, child rank {kp1}"
          + ("  ** 2k=N extremal **" if 2 * kp1 == n else ""))
    print(f"    D = <C, v>,  |Aut(D)| = {rec['aut_order']},  "
          f"tie {'ACCEPTED' if rec['accept'] else 'REJECTED'} by canon")
    for j, w in enumerate(rows[:-1]):
        print(f"    C row {j}:  {fmt_vec(w, n)}")
    print(f"    v      :  {fmt_vec(rows[-1], n)}")
    specs = {u: coset_spectrum(rows, u) for u in rec["m_set"]}
    uniq = set(specs.values())
    assert len(uniq) == 1, "tied strata must share the coset spectrum"
    print(f"    shared complement-coset weight spectrum: {next(iter(uniq))}")
    print(f"    tied strata (hyperplane functionals): {rec['m_set']}")
    print(f"    Aut(D)-orbit partition: {rec['tie_orbits']}"
          + ("   <-- TRUE COLLISION (inequivalent subcodes, equal spectra)"
             if len(rec["tie_orbits"]) > 1 else "   (single orbit — benign tie)"))
    if len(rec["tie_orbits"]) > 1:
        for oi, orbit in enumerate(rec["tie_orbits"]):
            u = orbit[0]
            h = hyperplane_rref(rows, u, n)
            print(f"      orbit {oi} representative (u={u}), subcode rref:")
            for w in h:
                print(f"        {fmt_vec(w, n)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="tie-dump JSONL files")
    ap.add_argument("--examples", type=int, default=2,
                    help="exemplars to pretty-print per file (prefers "
                    "multi-orbit and extremal-rank ties)")
    args = ap.parse_args()

    for path in args.dumps:
        recs = [json.loads(line) for line in open(path) if line.strip()]
        if not recs:
            print(f"\n=== {path}: empty (no ties at this N) ===")
            continue
        n = recs[0]["n"]
        by_k = defaultdict(lambda: [0, 0, 0, 0])  # ties, accepts, multi-orbit, max_orbits
        for r in recs:
            row = by_k[r["parent_k"] + 1]  # bucket by CHILD rank
            row[0] += 1
            row[1] += bool(r["accept"])
            row[2] += len(r["tie_orbits"]) > 1
            row[3] = max(row[3], len(r["tie_orbits"]))
        total = len(recs)
        multi = sum(1 for r in recs if len(r["tie_orbits"]) > 1)
        print(f"\n=== {path}: N={n}, {total} ties, "
              f"{multi} true collisions ({100*multi/total:.1f}%) ===")
        print("    child_k   ties   accepts   true-collisions   max-orbits")
        for k in sorted(by_k):
            t, a, m, mo = by_k[k]
            flag = "  <-- 2k=N" if 2 * k == n else ""
            print(f"    {k:7d} {t:6d} {a:9d} {m:17d} {mo:12d}{flag}")

        # Exemplars: multi-orbit first, then highest child rank.
        ranked = sorted(
            recs,
            key=lambda r: (len(r["tie_orbits"]) > 1, r["parent_k"]),
            reverse=True,
        )
        for i, rec in enumerate(ranked[: args.examples], 1):
            show_example(rec, i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
