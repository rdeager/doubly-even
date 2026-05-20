"""Direct head-to-head on the extended binary Golay code [24, 12, 8].

Golay is the right stress test for Feulner: extremal Type II self-dual,
|Aut| = M_24 = 244,823,040, |A_4| = 0, |A_8| = 759. Maximally symmetric
code in our enumeration range, where Feulner's automorphism-detection
search-tree pruning should help most.

We find Golay by iterating the doubly-even enumeration at N=24 and
filtering for rank=12, min_weight=8 (Golay is the unique such code).

For each rank-12 [24, 12] code we encounter we report:
- min weight, |A_4|, |A_8|
- |C_low| from the Q_D builder (via nauty_hist)
- per-call median timings: canon_info_native (full bipartite),
  canon_info_qd_native (Q_D), canon_info_feulner_native (Feulner)

Requires the kernel built with `--features nauty_hist,parallel`.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import doubly_even_kernel as _kernel  # noqa: E402

from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402

N = 24
WARMUP = 3
REPS = 20  # heavier for the single-code measurement


def weight_enum(basis: list[int], n: int) -> dict[int, int]:
    """Full weight enumerator. Gray-code walk over 2^k codewords."""
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


def min_weight_nonzero(we: dict[int, int]) -> int:
    return min(w for w in we if w > 0 and we[w] > 0)


def time_call(fn, *args) -> float:
    for _ in range(WARMUP):
        fn(*args)
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter_ns()
        fn(*args)
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples)


def get_clow(rref: list[int], n: int) -> tuple[int, bool]:
    """Return (|C_low|, qd_succeeded). 2^k when bailed."""
    _kernel.drain_nauty_hist()
    res = _kernel.canon_info_qd_native(rref, n)
    qd_succeeded = res is not None
    drained = _kernel.drain_nauty_hist()
    if qd_succeeded and drained:
        for row in drained:
            if row[8] == 1:  # qd_path
                return int(row[5]), True
    return 1 << len(rref), False


def main():
    if not hasattr(_kernel, "drain_nauty_hist"):
        print("ERROR: kernel needs --features nauty_hist", flush=True)
        sys.exit(1)

    print(f"Enumerating doubly-even codes at N={N}...", flush=True)
    t0 = time.perf_counter()
    rank12_codes: list[tuple[list[int], int, dict[int, int]]] = []
    for ec in enumerate_doubly_even(N):
        if ec.code.rank == 12:
            basis = list(ec.code.basis)
            we = weight_enum(basis, N)
            rank12_codes.append((basis, ec.aut_order, we))
    enum_s = time.perf_counter() - t0
    print(f"  Enumerated in {enum_s:.1f}s. Rank-12 classes found: {len(rank12_codes)}", flush=True)

    print(f"\n=== Rank-12 [24, 12] doubly-even self-dual codes ===")
    print(f"  {'min_wt':>7} {'|A_4|':>7} {'|A_8|':>7} {'|Aut|':>16} {'note':>20}")
    for basis, aut, we in rank12_codes:
        d = min_weight_nonzero(we)
        a4 = we.get(4, 0)
        a8 = we.get(8, 0)
        note = "← GOLAY" if d == 8 else ""
        print(f"  {d:>7} {a4:>7} {a8:>7} {aut:>16} {note:>20}", flush=True)

    # Time canon on every rank-12 code.
    print(f"\n=== Per-call timings on all (N=24, k=12) codes (median of {REPS} reps) ===")
    print(f"  {'d':>3} {'|A_4|':>6} {'|A_8|':>6} {'|Aut|':>14} {'|C_low|':>8} "
          f"{'fb µs':>9} {'qd µs':>9} {'feu µs':>10} {'feu/qd':>8} {'feu/fb':>8} {'label':>8}")
    timings = []
    for basis, aut, we in rank12_codes:
        d = min_weight_nonzero(we)
        a4 = we.get(4, 0)
        a8 = we.get(8, 0)
        c_low, qd_ok = get_clow(basis, N)
        fb_ns = time_call(_kernel.canon_info_native, basis, N)
        qd_ns = time_call(_kernel.canon_info_qd_native, basis, N)
        feu_ns = time_call(_kernel.canon_info_feulner_native, basis, N)
        label = "GOLAY" if d == 8 else ""
        print(f"  {d:>3} {a4:>6} {a8:>6} {aut:>14} {c_low:>8} "
              f"{fb_ns/1e3:>9.1f} {qd_ns/1e3:>9.1f} {feu_ns/1e3:>10.1f} "
              f"{feu_ns/qd_ns:>8.2f} {feu_ns/fb_ns:>8.2f} {label:>8}",
              flush=True)
        timings.append({
            "min_weight": d, "a4": a4, "a8": a8,
            "aut_order": str(aut), "c_low": c_low, "qd_ok": qd_ok,
            "fb_us": fb_ns / 1e3, "qd_us": qd_ns / 1e3, "feu_us": feu_ns / 1e3,
            "label": label,
        })

    # Also: Feulner counters (leaves, prunes) on each — see if Golay's huge
    # Aut group prunes the tree dramatically.
    if hasattr(_kernel, "_debug"):
        print(f"\n=== Feulner search-tree counters per rank-12 code ===")
        print(f"  {'d':>3} {'|Aut|':>14} {'leaves':>10} {'prunes':>10} {'leaves/|Aut|':>14}")
        for basis, aut, we in rank12_codes:
            d = min_weight_nonzero(we)
            try:
                leaves, prunes = _kernel._debug.canon_info_feulner_counters(basis, N)
                ratio = leaves / aut if aut else 0.0
                print(f"  {d:>3} {aut:>14} {leaves:>10} {prunes:>10} {ratio:>14.4f}",
                      flush=True)
            except AttributeError:
                pass

    out = {
        "N": N,
        "rank12_classes": [
            {
                "min_weight": min_weight_nonzero(we),
                "a4": we.get(4, 0),
                "a8": we.get(8, 0),
                "aut_order": str(a),
                "we": {str(k): v for k, v in sorted(we.items())},
            }
            for basis, a, we in rank12_codes
        ],
        "timings": timings,
    }
    out_path = Path("scripts/bench-results/golay-canon-bench.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
