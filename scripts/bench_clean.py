"""Benchmark ``doubly_even.clean.enumerate_doubly_even(N)`` with a pynauty/
non-pynauty wall split.

Mirror of the production ``scripts/bench.py`` but for the pedagogical clean
module. The clean module's perf budget is "non-nauty Python overhead < 60 %
of pynauty wall" — this script reports the ratio at each N.

Pynauty wall is measured by monkey-patching ``pynauty.autgrp`` and
``pynauty.canon_label`` (the two calls inside
``doubly_even.clean._canon._nauty_canon``). All other Python work (qc
enumeration, McKay parent test, Gray walks, RREF, list building, ...) lands
in the "non-nauty" bucket.

Usage::

    uv run python scripts/bench_clean.py --label baseline --N 12,14,16,18
    uv run python scripts/bench_clean.py --label post-A1 --N 16,18,20

Output JSON lands in ``scripts/bench-results/<UTC>-<label>.json`` alongside
the production bench files.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pynauty  # noqa: E402

from doubly_even.clean import enumerate_doubly_even  # noqa: E402

# DFGHILM Appendix B Table 3 — sanity floor (same values as tests/test_augment.py).
DFGHILM_TABLE_3: dict[int, dict[int, int]] = {
    12: {1: 3, 2: 5, 3: 7, 4: 7, 5: 2},
    13: {1: 3, 2: 5, 3: 8, 4: 8, 5: 4},
    14: {1: 3, 2: 7, 3: 12, 4: 14, 5: 9, 6: 4},
    15: {1: 3, 2: 7, 3: 15, 4: 20, 5: 15, 6: 8, 7: 2},
    16: {1: 4, 2: 10, 3: 23, 4: 38, 5: 36, 6: 23, 7: 9, 8: 2},
    17: {1: 4, 2: 10, 3: 25, 4: 45, 5: 50, 6: 34, 7: 14, 8: 3},
    18: {1: 4, 2: 13, 3: 34, 4: 72, 5: 94, 6: 79, 7: 35, 8: 9},
    20: {1: 5, 2: 16, 3: 49, 4: 119, 5: 199, 6: 232, 7: 175, 8: 84, 9: 23, 10: 4},
}


# --- pynauty timing wrappers -------------------------------------------------

_nauty_ns_total = 0


def _install_nauty_timers():
    global _nauty_ns_total
    _nauty_ns_total = 0
    real_autgrp = pynauty.autgrp
    real_canon_label = pynauty.canon_label

    def timed_autgrp(g):
        global _nauty_ns_total
        t0 = time.perf_counter_ns()
        out = real_autgrp(g)
        _nauty_ns_total += time.perf_counter_ns() - t0
        return out

    def timed_canon_label(g):
        global _nauty_ns_total
        t0 = time.perf_counter_ns()
        out = real_canon_label(g)
        _nauty_ns_total += time.perf_counter_ns() - t0
        return out

    pynauty.autgrp = timed_autgrp
    pynauty.canon_label = timed_canon_label


def _read_and_reset_nauty_ns() -> int:
    global _nauty_ns_total
    out = _nauty_ns_total
    _nauty_ns_total = 0
    return out


# --- bench core --------------------------------------------------------------


def run_one(N: int, workers: int) -> dict:
    _read_and_reset_nauty_ns()
    per_k: Counter[int] = Counter()
    t0 = time.perf_counter_ns()
    for ec in enumerate_doubly_even(N, workers=workers):
        per_k[ec.code.rank] += 1
    total_ns = time.perf_counter_ns() - t0
    nauty_ns = _read_and_reset_nauty_ns()
    other_ns = total_ns - nauty_ns
    ratio = (other_ns / nauty_ns) if nauty_ns > 0 else float("inf")

    # Table-3 check (drop k=0 class which Table 3 doesn't include).
    table = DFGHILM_TABLE_3.get(N)
    table_ok = None
    if table is not None:
        observed = {k: v for k, v in per_k.items() if k >= 1}
        table_ok = observed == table

    return {
        "N": N,
        "workers": workers,
        "wall_s": total_ns / 1e9,
        "nauty_s": nauty_ns / 1e9,
        "other_s": other_ns / 1e9,
        "ratio_other_over_nauty": ratio,
        "classes_by_k": dict(per_k),
        "table_check": table_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--N", default="12,14,16,18",
                    help="Comma-separated N values to bench (sequential by default).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Worker count for clean's parallel path (default 1).")
    args = ap.parse_args()

    Ns = [int(s) for s in args.N.split(",") if s.strip()]
    _install_nauty_timers()

    results = []
    for N in Ns:
        rec = run_one(N, args.workers)
        results.append(rec)
        ratio = rec["ratio_other_over_nauty"]
        tc = rec["table_check"]
        tc_str = ("OK" if tc else "MISMATCH") if tc is not None else "n/a"
        print(
            f"N={N:>2d}  wall={rec['wall_s']:7.2f}s  "
            f"nauty={rec['nauty_s']:7.2f}s  other={rec['other_s']:7.2f}s  "
            f"other/nauty={ratio:5.2f}  table={tc_str}",
            flush=True,
        )

    out = {
        "label": args.label,
        "utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "workers": args.workers,
        "results": results,
    }
    out_dir = HERE / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"{stamp}-clean-{args.label}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    bad = [r for r in results if r["table_check"] is False]
    if bad:
        print(f"TABLE-3 MISMATCH at N={[r['N'] for r in bad]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
