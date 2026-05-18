"""Benchmark ``enumerate_doubly_even(N)`` across a range of ``N``.

Run before and after each optimisation to build an audit trail of wins.
Writes one JSON record per run to ``scripts/bench-results/``.

Usage::

    uv run python scripts/bench.py --label baseline
    uv run python scripts/bench.py --label after-D2 --N 12,14,16,18
    uv run python scripts/bench.py --cprofile 16 --label profile-16

The ``--cprofile N`` flag also dumps a ``.prof`` file in the script
directory for ``snakeviz`` / ``pstats`` inspection.

Sanity floor: equivalence-class counts per ``(N, k)`` are checked
against DFGHILM Appendix B Table 3. The bench writes its JSON
regardless, but exits non-zero and prints the diff if any cell
disagrees.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Allow ``uv run python scripts/bench.py`` without installing the package
# editable on every checkout: prepend the ``src/`` source root.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.canon.nauty import canon_info_cache_clear  # noqa: E402
from doubly_even.enumerate.augment import (  # noqa: E402
    _parse_thread_env,
    enumerate_doubly_even,
    weight_enum_cache_clear,
)
from doubly_even.spec.codes import dual_cache_clear, rref_cache_clear  # noqa: E402
from doubly_even.spec.mass import gaborit_sigma  # noqa: E402

import os as _os  # noqa: E402 — bench env-var read for D13 parallel path

try:  # pragma: no cover -- import-side switch
    import doubly_even_kernel as _kernel
except ImportError:  # pragma: no cover
    _kernel = None

# Layout of the kernel's `stats` vector — kept in sync by hand with
# `rust/src/enumerate.rs::enumerate_doubly_even` doc. Indices into the
# returned 30-element list.
KERNEL_STATS_LAYOUT: tuple[str, ...] = (
    "canon_calls",                # 0
    "primary_hits",               # 1
    "secondary_attempts",         # 2
    "secondary_hits",             # 3
    "is_canon_aug_calls",         # 4
    "parent_eq_hits",             # 5
    "weight_enum_filtered",       # 6
    "bfs_calls",                  # 7
    "bfs_hits",                   # 8
    "is_canon_aug_ns",            # 9
    "bfs_ns",                     # 10
    "nauty_ns",                   # 11
    "bucket_size_sum_at_attempt", # 12
    "match_position_sum",         # 13
    "max_bucket_size",            # 14
    "verifier_attempts",          # 15
    "verifier_hits",              # 16
    "verifier_compares",          # 17
    "verifier_ns",                # 18
    "candidates_q_calls",         # 19
    "candidates_q_ns",            # 20
    "bfs_rejects",                # 21
    "nauty_numnodes_sum",         # 22 — Q6: backtrack tree size
    "nauty_tctotal_sum",          # 23 — Q6: target-cell work
    "nauty_maxlevel_sum",         # 24 — Q6: deepest level reached
    "nauty_generators_sum",       # 25 — Q6: Aut generators found
    "t11_hits",                   # 26 — T11 cache: skipped nauty
    "t11_misses",                 # 27 — T11 cache: new hash, called nauty
    "t11_blocklist_hits",         # 28 — T11 cache: collision hash, forced nauty
    "t11_ns",                     # 29 — T11 cache: hash + lookup + insert time
)


# DFGHILM Appendix B Table 3 — number of permutation-equivalence classes
# of doubly even [N, k] codes. Mirrors ``tests/test_augment.py`` (kept in
# sync by hand; both must agree). Cells absent here are treated as
# "unknown / no check".
DFGHILM_TABLE_3: dict[tuple[int, int], int] = {
    (4, 1): 1,
    (5, 1): 1,
    (6, 1): 1, (6, 2): 1,
    (7, 1): 1, (7, 2): 1, (7, 3): 1,
    (8, 1): 2, (8, 2): 2, (8, 3): 2, (8, 4): 1,
    (9, 1): 2, (9, 2): 2, (9, 3): 2, (9, 4): 1,
    (10, 1): 2, (10, 2): 3, (10, 3): 3, (10, 4): 2,
    (11, 1): 2, (11, 2): 3, (11, 3): 4, (11, 4): 3,
    (12, 1): 3, (12, 2): 5, (12, 3): 7, (12, 4): 7, (12, 5): 2,
    (13, 1): 3, (13, 2): 5, (13, 3): 8, (13, 4): 8, (13, 5): 4,
    (14, 1): 3, (14, 2): 7, (14, 3): 12, (14, 4): 14, (14, 5): 9,
    (14, 6): 4,
    (15, 1): 3, (15, 2): 7, (15, 3): 15, (15, 4): 20, (15, 5): 15,
    (15, 6): 8, (15, 7): 2,
    (16, 1): 4, (16, 2): 10, (16, 3): 23, (16, 4): 38, (16, 5): 36,
    (16, 6): 23, (16, 7): 9, (16, 8): 2,
    (17, 1): 4, (17, 2): 10, (17, 3): 25, (17, 4): 45, (17, 5): 50,
    (17, 6): 34, (17, 7): 14, (17, 8): 3,
    (18, 1): 4, (18, 2): 13, (18, 3): 34, (18, 4): 72, (18, 5): 94,
    (18, 6): 79, (18, 7): 35, (18, 8): 9,
    # N=26 from the DFGHILM table, cross-checked by this enumerator
    # 2026-05-18 (parallel kernel, depth=5, 20 threads, 218 s wall).
    (26, 1): 6, (26, 2): 32, (26, 3): 180, (26, 4): 1114, (26, 5): 6923,
    (26, 6): 37455, (26, 7): 128270, (26, 8): 194626, (26, 9): 103527,
    (26, 10): 20206, (26, 11): 1829, (26, 12): 103,
}


@dataclass
class PerKResult:
    k: int
    classes: int = 0
    mass: int = 0  # sum of N! / |Aut(C)| over emitted classes


@dataclass
class PerNResult:
    N: int
    seconds: float = 0.0
    classes: int = 0
    per_k: dict[int, PerKResult] = field(default_factory=dict)
    # Kernel stats vector keyed by KERNEL_STATS_LAYOUT name. Empty when
    # the kernel isn't loaded.
    kernel_stats: dict[str, int] = field(default_factory=dict)


def run_one(N: int) -> PerNResult:
    factorial_N = math.factorial(N)
    result = PerNResult(N=N)
    # Cold-cache start so the per-N timing isn't contaminated by entries
    # populated during earlier N's. (No cross-N hits in practice since the
    # key includes ``n``, but this is defensive and makes runs comparable.)
    canon_info_cache_clear()
    rref_cache_clear()
    dual_cache_clear()
    weight_enum_cache_clear()
    if _kernel is not None:
        # Call the kernel directly so we can capture the stats vector. The
        # public iterator drops it; for benchmarking we need it.
        cap = N // 2
        quota_vec = [gaborit_sigma(N, k) for k in range(cap + 1)]
        num_threads = _parse_thread_env(_os.environ.get("DOUBLY_EVEN_THREADS"))
        t0 = time.perf_counter()
        if num_threads is not None and num_threads >= 2:
            raw, stats, _per_k = _kernel.enumerate_doubly_even(
                N, cap, quota_vec, factorial_N, num_threads,
            )
        else:
            raw, stats, _per_k = _kernel.enumerate_doubly_even(
                N, cap, quota_vec, factorial_N,
            )
        result.seconds = time.perf_counter() - t0
        # Aggregate per-k from the raw output.
        for rref, _ccol, _gens, aord_str, _orbits in raw:
            k = len(rref)
            slot = result.per_k.setdefault(k, PerKResult(k=k))
            slot.classes += 1
            slot.mass += factorial_N // int(aord_str)
            result.classes += 1
        result.kernel_stats = {
            name: int(stats[i]) for i, name in enumerate(KERNEL_STATS_LAYOUT)
        }
        return result
    # Pure-Python fallback (no kernel; no stats vector available).
    t0 = time.perf_counter()
    for ec in enumerate_doubly_even(N):
        k = ec.code.rank
        slot = result.per_k.setdefault(k, PerKResult(k=k))
        slot.classes += 1
        slot.mass += factorial_N // ec.aut_order
        result.classes += 1
    result.seconds = time.perf_counter() - t0
    return result


def check_against_table(results: list[PerNResult]) -> list[str]:
    """Compare class counts to DFGHILM Table 3; return list of diffs."""
    diffs: list[str] = []
    for r in results:
        for k, expected in DFGHILM_TABLE_3.items():
            if k[0] != r.N:
                continue
            got = r.per_k.get(k[1], PerKResult(k=k[1])).classes
            if got != expected:
                diffs.append(
                    f"  (N={k[0]}, k={k[1]}): got {got}, table {expected}"
                )
    return diffs


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_json(label: str, results: list[PerNResult]) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = HERE / "bench-results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}-{label}.json"
    payload = {
        "label": label,
        "timestamp_utc": ts,
        "git_sha": git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "per_N": {
            r.N: {
                "seconds": r.seconds,
                "classes": r.classes,
                "per_k": {
                    pk.k: {"classes": pk.classes, "mass": pk.mass}
                    for pk in sorted(r.per_k.values(), key=lambda x: x.k)
                },
                "kernel_stats": r.kernel_stats,
            }
            for r in results
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def format_table(results: list[PerNResult]) -> str:
    lines = []
    lines.append(f"{'N':>4} {'seconds':>10} {'classes':>8}   per-k (k:count)")
    for r in results:
        per_k = ", ".join(
            f"{pk.k}:{pk.classes}" for pk in sorted(r.per_k.values(), key=lambda x: x.k)
        )
        lines.append(f"{r.N:>4} {r.seconds:>10.3f} {r.classes:>8}   {per_k}")
    # Nauty Q6 decomposition table (only when kernel stats present).
    have_stats = any(r.kernel_stats for r in results)
    if have_stats:
        lines.append("")
        lines.append(
            f"{'N':>4} {'calls':>8} {'us/call':>8} "
            f"{'nodes/call':>11} {'tc/call':>9} "
            f"{'lvl/call':>9} {'gen/call':>9}"
        )
        for r in results:
            ks = r.kernel_stats
            if not ks:
                continue
            calls = max(ks.get("canon_calls", 0), 1)
            us = ks.get("nauty_ns", 0) / 1_000.0 / calls
            nodes = ks.get("nauty_numnodes_sum", 0) / calls
            tc = ks.get("nauty_tctotal_sum", 0) / calls
            lvl = ks.get("nauty_maxlevel_sum", 0) / calls
            gen = ks.get("nauty_generators_sum", 0) / calls
            lines.append(
                f"{r.N:>4} {calls:>8} {us:>8.2f} "
                f"{nodes:>11.2f} {tc:>9.2f} "
                f"{lvl:>9.2f} {gen:>9.2f}"
            )
        # T11 cache decomposition (only when feature was active, signalled
        # by any of the t11_* fields being non-zero in any run).
        any_t11 = any(
            r.kernel_stats.get("t11_hits", 0)
            or r.kernel_stats.get("t11_misses", 0)
            or r.kernel_stats.get("t11_blocklist_hits", 0)
            for r in results
        )
        if any_t11:
            lines.append("")
            lines.append(
                f"{'N':>4} {'t11_hits':>10} {'t11_miss':>10} "
                f"{'blocklist':>10} {'hit_rate':>10} {'t11_ms':>9}"
            )
            for r in results:
                ks = r.kernel_stats
                hits = ks.get("t11_hits", 0)
                miss = ks.get("t11_misses", 0)
                bl = ks.get("t11_blocklist_hits", 0)
                t11_ms = ks.get("t11_ns", 0) / 1_000_000.0
                total = hits + miss + bl
                rate = 100.0 * hits / max(total, 1)
                lines.append(
                    f"{r.N:>4} {hits:>10} {miss:>10} {bl:>10} "
                    f"{rate:>9.1f}% {t11_ms:>9.2f}"
                )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--label",
        required=True,
        help='Run label, e.g. "baseline", "after-D2"; included in the '
        "output filename and JSON.",
    )
    p.add_argument(
        "--N",
        default="12,14,16,18,20",
        help="Comma-separated list of N to benchmark (default: 12,14,16,18,20).",
    )
    p.add_argument(
        "--cprofile",
        type=int,
        default=None,
        metavar="N",
        help="Run cProfile on a single value of N and dump a .prof file "
        "next to the script. Implies a single-N run.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.cprofile is not None:
        N = args.cprofile
        prof_path = HERE / f"bench-{N}.prof"
        profiler = cProfile.Profile()
        profiler.enable()
        result = run_one(N)
        profiler.disable()
        profiler.dump_stats(str(prof_path))
        results = [result]
        print(f"cProfile output: {prof_path}")
    else:
        Ns = [int(s.strip()) for s in args.N.split(",") if s.strip()]
        results = [run_one(N) for N in Ns]

    print(format_table(results))
    out_path = write_json(args.label, results)
    print(f"\nWrote {out_path}")

    diffs = check_against_table(results)
    if diffs:
        print("\nDFGHILM Table 3 disagreement:")
        print("\n".join(diffs))
        return 1
    print("\nDFGHILM Table 3: all checked cells agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
