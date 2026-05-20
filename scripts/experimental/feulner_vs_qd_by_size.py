"""Head-to-head Feulner vs Q_D nauty as a function of |C_low|.

For each sampled code we:
1. Drain the nauty_hist buffer (clear it).
2. Time `canon_info_qd_native` once and re-drain to read off |C_low|
   (left_vertices in the QD-path record; if the record's qd_path=0
   then Q_D bailed and we record |C_low| = 2^k as the "would-have-been"
   full size).
3. Time `canon_info_feulner_native` on the same RREF.
4. Also time `canon_info_native` (full bipartite) for context.

We then bin codes by |C_low| (and equivalently by |A_4|, the number of
weight-4 codewords — cheap to compute, may be a better dispatch signal
than |C_low| since it doesn't require span-walking).

Output: per-|C_low| bucket median times for Q_D nauty, full-bipartite
nauty, and Feulner. If Feulner crosses below Q_D somewhere on this
axis, that's the dispatch threshold.

Requires the kernel built with `--features nauty_hist`.
"""

from __future__ import annotations

import argparse
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

WARMUP = 2
REPS = 8


def count_weight4_codewords(rref: list[int], n: int) -> int:
    """|A_4| = number of weight-4 codewords. Cheap: full Gray-code walk
    over 2^k, popcount each."""
    k = len(rref)
    if k == 0:
        return 0
    total = 1 << k
    cw = [0] * total
    for mask in range(1, total):
        lo_bit = (mask & -mask).bit_length() - 1
        cw[mask] = cw[mask ^ (1 << lo_bit)] ^ rref[lo_bit]
    return sum(1 for w in cw if bin(w).count("1") == 4)


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
    """Return (|C_low|, qd_succeeded). When Q_D bails, |C_low| is reported
    as 2^k (i.e. the size of the full bipartite graph the fallback uses)."""
    _kernel.drain_nauty_hist()
    # Trigger one Q_D call so it logs.
    res = _kernel.canon_info_qd_native(rref, n)
    qd_succeeded = res is not None
    # When Q_D returns None, it does NOT call sparsenauty so no record
    # was pushed. In that case we record |C_low| = 2^k (the would-be size).
    drained = _kernel.drain_nauty_hist()
    if qd_succeeded and drained:
        # Find the qd_path=1 record (last one is fine; should be the only one).
        for row in drained:
            if row[8] == 1:  # qd_path
                return int(row[5]), True  # left_vertices
    return 1 << len(rref), False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--sample", type=int, default=400,
                    help="cap total codes sampled (across all ranks)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n = args.N
    sample_cap = args.sample

    if not hasattr(_kernel, "drain_nauty_hist"):
        print("ERROR: kernel was not built with --features nauty_hist", flush=True)
        sys.exit(1)

    print(f"Enumerating doubly-even codes at N={n}, sampling up to {sample_cap}...", flush=True)
    t0 = time.perf_counter()

    bases: list[tuple[int, list[int]]] = []  # (rank, basis)
    for ec in enumerate_doubly_even(n):
        if ec.code.rank >= 2:
            bases.append((ec.code.rank, list(ec.code.basis)))
        if len(bases) >= sample_cap * 2:  # gather a few more than needed
            break

    enum_s = time.perf_counter() - t0

    # Stride-sample evenly across the collection (no rank bias).
    if len(bases) > sample_cap:
        stride = len(bases) / sample_cap
        bases = [bases[int(i * stride)] for i in range(sample_cap)]
    print(f"  Enumerated in {enum_s:.1f}s; sampled {len(bases)} codes", flush=True)

    rows = []
    for i, (k, basis) in enumerate(bases):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(bases)}] timing...", flush=True)
        c_low, qd_ok = get_clow(basis, n)
        a4 = count_weight4_codewords(basis, n)
        qd_ns = time_call(_kernel.canon_info_qd_native, basis, n)
        fb_ns = time_call(_kernel.canon_info_native, basis, n)
        feu_ns = time_call(_kernel.canon_info_feulner_native, basis, n)
        rows.append({
            "k": k, "c_low": c_low, "qd_ok": qd_ok, "a4": a4,
            "qd_ns": qd_ns, "fb_ns": fb_ns, "feu_ns": feu_ns,
            "two_to_k": 1 << k,
        })

    # Bin by |C_low|. Use log2-ish bins: 1-2, 3-4, 5-8, 9-16, 17-32, ...
    def bin_for(c_low: int) -> str:
        if c_low <= 2:
            return "1-2"
        edge = 4
        while edge < c_low:
            edge *= 2
        return f"{edge//2+1}-{edge}"

    by_bin: dict[str, list[dict]] = {}
    for r in rows:
        by_bin.setdefault(bin_for(r["c_low"]), []).append(r)

    print(f"\n=== Feulner vs Q_D vs full-bipartite by |C_low| at N={n} ===")
    print(f"  {'|C_low| bin':>12} {'n':>5} {'qd µs':>10} {'fb µs':>10} {'feu µs':>10} "
          f"{'feu/qd':>8} {'feu/fb':>8} {'qd_ok %':>8} {'a4 p50':>8}")
    out_rows: list[dict] = []
    # Sort bins by upper edge.
    def sort_key(bin_label: str) -> int:
        return int(bin_label.split("-")[1])
    for binlbl in sorted(by_bin, key=sort_key):
        items = by_bin[binlbl]
        qd_med = statistics.median(r["qd_ns"] for r in items) / 1e3
        fb_med = statistics.median(r["fb_ns"] for r in items) / 1e3
        fe_med = statistics.median(r["feu_ns"] for r in items) / 1e3
        qd_ok_pct = 100 * sum(1 for r in items if r["qd_ok"]) / len(items)
        a4_med = statistics.median(r["a4"] for r in items)
        print(f"  {binlbl:>12} {len(items):>5} {qd_med:>10.1f} {fb_med:>10.1f} {fe_med:>10.1f} "
              f"{fe_med/qd_med if qd_med else 0:>8.2f} {fe_med/fb_med if fb_med else 0:>8.2f} "
              f"{qd_ok_pct:>7.1f}% {a4_med:>8.0f}")
        out_rows.append({
            "c_low_bin": binlbl, "n": len(items),
            "qd_us_p50": qd_med, "fb_us_p50": fb_med, "feu_us_p50": fe_med,
            "qd_ok_pct": qd_ok_pct, "a4_p50": a4_med,
        })

    # Same again binned by |A_4| (number of weight-4 codewords).
    def a4_bin(a4: int) -> str:
        if a4 == 0:
            return "0"
        edge = 1
        while edge < a4:
            edge *= 2
        return f"{edge//2+1}-{edge}" if edge > 1 else "1"

    by_a4: dict[str, list[dict]] = {}
    for r in rows:
        by_a4.setdefault(a4_bin(r["a4"]), []).append(r)

    print(f"\n=== Same data, binned by |A_4| (number of weight-4 codewords) ===")
    print(f"  {'|A_4| bin':>12} {'n':>5} {'qd µs':>10} {'fb µs':>10} {'feu µs':>10} "
          f"{'feu/qd':>8} {'c_low p50':>10}")
    out_a4: list[dict] = []
    def a4_sort_key(b: str) -> int:
        if b == "0":
            return 0
        if b == "1":
            return 1
        return int(b.split("-")[1])
    for binlbl in sorted(by_a4, key=a4_sort_key):
        items = by_a4[binlbl]
        qd_med = statistics.median(r["qd_ns"] for r in items) / 1e3
        fb_med = statistics.median(r["fb_ns"] for r in items) / 1e3
        fe_med = statistics.median(r["feu_ns"] for r in items) / 1e3
        clow_med = statistics.median(r["c_low"] for r in items)
        print(f"  {binlbl:>12} {len(items):>5} {qd_med:>10.1f} {fb_med:>10.1f} {fe_med:>10.1f} "
              f"{fe_med/qd_med if qd_med else 0:>8.2f} {clow_med:>10.0f}")
        out_a4.append({
            "a4_bin": binlbl, "n": len(items),
            "qd_us_p50": qd_med, "fb_us_p50": fb_med, "feu_us_p50": fe_med,
            "c_low_p50": clow_med,
        })

    out = {
        "N": n, "enum_s": enum_s, "sample_count": len(bases),
        "by_c_low_bin": out_rows,
        "by_a4_bin": out_a4,
        "raw_rows": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
