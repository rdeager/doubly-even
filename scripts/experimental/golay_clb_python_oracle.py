"""Phase-A oracle for the Python CLB prototype in canon/feulner.py.

Finds the extended binary Golay code [24, 12, 8] via the kernel
enumerator, then runs the pure-Python ``canon_info_feulner`` twice on
it: once with ``use_clb=False`` (pre-CLB baseline, equivalent to what
the Rust port currently does), once with ``use_clb=True``.

Expected outcome of this script:
  * Both runs return the same |Aut|, column orbits, and canonical column
    order (correctness — CLB is a search-tree pruning, not a semantic
    change).
  * ``clb_prune_fires`` is strictly positive on the CLB run.
  * ``leaves_visited`` drops materially (target: from ~hundreds to
    single digits / low tens on Golay).
  * Python wall time is *not* a meaningful metric here — pure-Python
    Schreier–Sims rebuilds inside ``_LabelledBranching._rebuild_father``
    dominate, and the Rust port (Phase B) is where wall wins materialise.

Run::

    uv run python scripts/golay_clb_python_oracle.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.canon.experimental.feulner import canon_info_feulner_with_counters  # noqa: E402
from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402
from doubly_even.spec.codes import Code  # noqa: E402

N = 24


def weight_enum(basis: tuple[int, ...], n: int) -> dict[int, int]:
    k = len(basis)
    if k == 0:
        return {0: 1}
    total = 1 << k
    cw = [0] * total
    for mask in range(1, total):
        lo = (mask & -mask).bit_length() - 1
        cw[mask] = cw[mask ^ (1 << lo)] ^ basis[lo]
    counts: dict[int, int] = {}
    for w in cw:
        wt = bin(w).count("1")
        counts[wt] = counts.get(wt, 0) + 1
    return counts


def find_golay() -> Code:
    print(f"Enumerating doubly-even codes at N={N}…", flush=True)
    t0 = time.perf_counter()
    for ec in enumerate_doubly_even(N):
        if ec.code.rank != 12:
            continue
        we = weight_enum(ec.code.basis, N)
        d = min(w for w in we if w > 0 and we[w] > 0)
        if d == 8:
            print(
                f"  Found Golay in {time.perf_counter() - t0:.1f} s "
                f"(|Aut| = {ec.aut_order:,}).",
                flush=True,
            )
            return ec.code
    raise RuntimeError("Golay [24, 12, 8] not encountered in enumeration")


def run_once(C: Code, *, use_clb: bool) -> tuple[float, dict[str, int], object]:
    t0 = time.perf_counter()
    info, counters = canon_info_feulner_with_counters(C, use_clb=use_clb)
    wall = time.perf_counter() - t0
    return wall, counters, info


def main() -> None:
    C = find_golay()

    print("\n--- Python Feulner, CLB OFF (regression baseline) ---", flush=True)
    wall_off, c_off, info_off = run_once(C, use_clb=False)
    print(
        f"  wall = {wall_off:6.3f} s  "
        f"leaves = {c_off['leaves_visited']:>6}  "
        f"prefix_prunes = {c_off['prune_fires']:>6}  "
        f"clb_prunes = {c_off['clb_prune_fires']:>4}",
        flush=True,
    )

    print("\n--- Python Feulner, CLB ON  (Phase A: Lemma 5.9) ---", flush=True)
    wall_on, c_on, info_on = run_once(C, use_clb=True)
    print(
        f"  wall = {wall_on:6.3f} s  "
        f"leaves = {c_on['leaves_visited']:>6}  "
        f"prefix_prunes = {c_on['prune_fires']:>6}  "
        f"clb_prunes = {c_on['clb_prune_fires']:>4}",
        flush=True,
    )

    print("\n--- Correctness cross-check ---", flush=True)
    same_aut = info_off.aut_order == info_on.aut_order
    same_orbits = info_off.column_orbits == info_on.column_orbits
    same_canonical = info_off.canonical_column_order == info_on.canonical_column_order
    print(f"  |Aut| match              : {same_aut}", flush=True)
    print(f"  column-orbit match       : {same_orbits}", flush=True)
    print(f"  canonical-order match    : {same_canonical}", flush=True)

    print("\n--- Summary ---", flush=True)
    leaf_ratio = c_off["leaves_visited"] / max(c_on["leaves_visited"], 1)
    print(
        f"  leaf reduction (CLB off / CLB on) = {leaf_ratio:.1f}×",
        flush=True,
    )

    ok = (
        same_aut
        and same_orbits
        and same_canonical
        and c_on["clb_prune_fires"] > 0
        and c_on["leaves_visited"] <= c_off["leaves_visited"]
    )
    if not ok:
        print("\nPHASE-A ORACLE FAILED — investigate before Phase B.", flush=True)
        sys.exit(1)
    print("\nphase-A oracle PASSED", flush=True)


if __name__ == "__main__":
    main()
