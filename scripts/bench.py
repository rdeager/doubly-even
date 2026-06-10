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
# returned 45-element list. Fields 34–44 are the `phase_timers`
# sub-phase split (zero on non-instrumented builds).
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
    "phi_reject",                 # 26 — D15: φ rejects (no canon call)
    "phi_accept_unique",          # 27 — D15: φ unique-min accepts
    "phi_tie_accept",             # 28 — D15: φ ties resolved accept
    "phi_tie_reject",             # 29 — D15: φ ties resolved reject
    "phi_ns",                     # 30 — D15: ns inside the φ cascade
    "phi_strata_sum",             # 31 — D15: Σ strata evaluated
    "phi_m_size_sum",             # 32 — D15: Σ |M| at decision
    "nauty_ns_kept",              # 33 — D15 audit: κ numerator
    "cq_qbasis_ns",               # 34 — σ_Q sub-phase: q_basis
    "cq_autimage_ns",             # 35 — σ_Q sub-phase: aut_image_on_q
    "cq_singular_ns",             # 36 — σ_Q sub-phase: singular_reps_q
    "cq_orbitmin_ns",             # 37 — σ_Q sub-phase: orbit-min BFS
    "cq_lift_sort_ns",            # 38 — σ_Q sub-phase: lift + sort
    "phi_frame_gray_ns",          # 39 — φ sampled: frame + Gray sweep
    "phi_sort_ns",                # 40 — φ sampled: counting sort
    "phi_first_stratum_ns",       # 41 — φ sampled: first-stratum argmin
    "phi_wht_ns",                 # 42 — φ sampled: later-stratum WHT
    "phi_direct_ns",              # 43 — φ sampled: direct parity
    "phi_sampled_calls",          # 44 — φ sampling weights (1-in-64)
)

# Row names of the kernel's `per_k_stats` matrix, in fixed order — kept in
# sync by hand with `rust/src/enumerate.rs::WorkerState::finalize` (and
# `scripts/experimental/d15_phi_audit.py::PER_K_ROWS`). Rows 0–13 are
# counters bucketed by PARENT rank k (rank of C; the child D has rank
# k+1). Rows 14–16 are the post-D15 per-rank timing rows (ns, always-on);
# rows 14–15 are bucketed by parent rank, row 16 (`nauty_ns`) by the rank
# of the code being canonised (= parent_k + 1 for child canon calls).
PER_K_STATS_ROWS: tuple[str, ...] = (
    "is_canon_aug_calls",     # 0
    "parent_eq_hits",         # 1
    "weight_enum_filtered",   # 2
    "bfs_calls",              # 3
    "bfs_hits",               # 4
    "bfs_rejects",            # 5
    "mass_stop_pre_loop",     # 6
    "mass_stop_in_loop",      # 7
    "candidates_total_seen",  # 8
    "candidates_skipped",     # 9
    "phi_reject",             # 10 — audit mode only
    "phi_accept_unique",      # 11 — audit mode only
    "phi_tie_accept",         # 12 — audit mode only
    "phi_tie_reject",         # 13 — audit mode only
    "phi_ns",                 # 14 — per-rank φ-cascade ns
    "candidates_q_ns",        # 15 — per-rank σ_Q candidate-generation ns
    "nauty_ns",               # 16 — per-rank canon-dispatch ns (child rank!)
    "phi_sampled_calls",      # 17 — per-rank φ sampling weights (phase_timers)
)


def per_k_stats_dict(per_k: list[list[int]]) -> dict[str, list[int]]:
    """Name the kernel's per_k_stats rows. Tolerates a kernel that returns
    more rows than we know about (forward-compat) but not fewer."""
    if len(per_k) < len(PER_K_STATS_ROWS):
        raise RuntimeError(
            f"kernel returned {len(per_k)} per_k rows, "
            f"expected >= {len(PER_K_STATS_ROWS)} — rebuild the wheel?"
        )
    return {name: [int(v) for v in per_k[i]] for i, name in enumerate(PER_K_STATS_ROWS)}


def assert_per_k_timing_consistency(
    kernel_stats: dict[str, int], per_k_stats: dict[str, list[int]]
) -> None:
    """The per-rank timing rows must sum to their aggregate stats field
    (same Instant delta accumulated into both) — a >1% gap means a
    bucketing site was missed in the kernel."""
    for row, agg_name in (
        ("phi_ns", "phi_ns"),
        ("candidates_q_ns", "candidates_q_ns"),
        ("nauty_ns", "nauty_ns"),
    ):
        row_sum = sum(per_k_stats[row])
        agg = kernel_stats[agg_name]
        if abs(row_sum - agg) > max(0.01 * agg, 1_000):
            raise RuntimeError(
                f"per_k row '{row}' sums to {row_sum} but aggregate "
                f"{agg_name} is {agg} — kernel bucketing is inconsistent"
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
    # Kernel per_k_stats matrix keyed by PER_K_STATS_ROWS name; each value
    # is a list indexed by rank. Empty when the kernel isn't loaded.
    per_k_stats: dict[str, list[int]] = field(default_factory=dict)
    # --profile-parallel payload, populated only when running through
    # the `enumerate_doubly_even_with_profile` entry (requires the
    # kernel to be built with `--features parallel_profiling`).
    parallel_profile: dict | None = None


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
            raw, stats, per_k = _kernel.enumerate_doubly_even(
                N, cap, quota_vec, factorial_N, num_threads,
            )
        else:
            raw, stats, per_k = _kernel.enumerate_doubly_even(
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
        result.per_k_stats = per_k_stats_dict(per_k)
        assert_per_k_timing_consistency(result.kernel_stats, result.per_k_stats)
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


def run_one_with_profile(N: int, num_threads: int) -> PerNResult:
    """Profiling variant of :func:`run_one`.

    Calls the kernel's `enumerate_doubly_even_with_profile` entry
    (requires the kernel to be built with
    ``--features parallel_profiling``) and captures the per-worker /
    per-seed breakdown into ``PerNResult.parallel_profile``. The
    timing-overhead added by the in-kernel Instant calls is small
    (~tens of ns per traverse), but profile-mode wall times are NOT
    directly comparable to production-mode wall times.
    """
    if _kernel is None:
        raise RuntimeError("--profile-parallel requires the doubly_even_kernel wheel")
    if not hasattr(_kernel, "enumerate_doubly_even_with_profile"):
        raise RuntimeError(
            "--profile-parallel requires the kernel to be built with "
            "`--features parallel_profiling`. The currently installed wheel "
            "does not export `enumerate_doubly_even_with_profile`."
        )
    factorial_N = math.factorial(N)
    cap = N // 2
    quota_vec = [gaborit_sigma(N, k) for k in range(cap + 1)]
    canon_info_cache_clear()
    rref_cache_clear()
    dual_cache_clear()
    weight_enum_cache_clear()
    t0 = time.perf_counter()
    raw, stats, per_k, profile = _kernel.enumerate_doubly_even_with_profile(
        N, cap, quota_vec, factorial_N, num_threads,
    )
    wall = time.perf_counter() - t0
    workers, seeds, frontier_depth, total_wall_ns = profile
    result = PerNResult(N=N, seconds=wall)
    for rref, _ccol, _gens, aord_str, _orbits in raw:
        k = len(rref)
        slot = result.per_k.setdefault(k, PerKResult(k=k))
        slot.classes += 1
        slot.mass += factorial_N // int(aord_str)
        result.classes += 1
    result.kernel_stats = {
        name: int(stats[i]) for i, name in enumerate(KERNEL_STATS_LAYOUT)
    }
    result.per_k_stats = per_k_stats_dict(per_k)
    assert_per_k_timing_consistency(result.kernel_stats, result.per_k_stats)
    result.parallel_profile = {
        "num_threads": int(num_threads),
        "frontier_depth": int(frontier_depth),
        "total_wall_ns": int(total_wall_ns),
        "workers": [
            {
                "worker_id": int(w[0]),
                "active_ns": int(w[1]),
                "idle_ns": int(w[2]),
                "seed_count": int(w[3]),
            }
            for w in workers
        ],
        "seeds": [
            {
                "worker_id": int(s[0]),
                "seed_id": int(s[1]),
                "ns": int(s[2]),
                "nodes": int(s[3]),
                "emitted": int(s[4]),
            }
            for s in seeds
        ],
    }
    return result


def _gini(values: list[int]) -> float:
    """Gini coefficient of a list of non-negative integers (0 = equal,
    → 1 = maximally concentrated). Returns 0.0 for an empty or all-zero
    input."""
    if not values:
        return 0.0
    s = sum(values)
    if s == 0:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cum = 0
    for i, v in enumerate(sorted_vals, start=1):
        cum += i * v
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def render_profile_summary(r: PerNResult) -> str:
    """Human-readable summary of a `parallel_profile` payload."""
    p = r.parallel_profile
    if p is None:
        return ""
    workers = p["workers"]
    seeds = p["seeds"]
    nt = p["num_threads"]
    fd = p["frontier_depth"]
    wall_s = p["total_wall_ns"] / 1e9
    if not workers:
        return f"  (no worker data — sequential fallback at N={r.N})"
    active_ns = [w["active_ns"] for w in workers]
    idle_ns = [w["idle_ns"] for w in workers]
    max_active = max(active_ns)
    min_active = min(active_ns)
    mean_active = sum(active_ns) / len(active_ns)
    idle_pct = 100.0 * sum(idle_ns) / max(sum(active_ns) + sum(idle_ns), 1)
    # Worker imbalance: ratio of max-worker active to mean active. 1.0 =
    # perfectly balanced; > 1.0 = one worker holds more than its share.
    imbalance = max_active / max(mean_active, 1.0)
    seeds_ns = [s["ns"] for s in seeds]
    seed_gini = _gini(seeds_ns)
    top5 = sorted(seeds, key=lambda s: s["ns"], reverse=True)[:5]
    top5_share = sum(s["ns"] for s in top5) / max(sum(seeds_ns), 1) * 100.0
    lines = [
        f"  N={r.N}  threads={nt}  frontier_depth={fd}  wall={wall_s:.3f} s",
        f"  workers: max_active={max_active / 1e6:.1f} ms  "
        f"min_active={min_active / 1e6:.1f} ms  "
        f"mean={mean_active / 1e6:.1f} ms  "
        f"imbalance(max/mean)={imbalance:.2f}×",
        f"  idle: {idle_pct:.1f} % of total worker time spent in recv()",
        f"  seeds: total={len(seeds)}  gini={seed_gini:.3f}  "
        f"top-5 share = {top5_share:.1f} %",
    ]
    if top5:
        lines.append("  top-5 heaviest seeds (worker, seed, ms, nodes, emitted):")
        for s in top5:
            lines.append(
                f"    w{s['worker_id']:>2}  s{s['seed_id']:>3}  "
                f"{s['ns'] / 1e6:>7.1f} ms  "
                f"nodes={s['nodes']:>7}  emitted={s['emitted']:>5}"
            )
    return "\n".join(lines)


def write_profile_json(label: str, results: list[PerNResult]) -> Path:
    """Write the profile payload to a separate JSON next to the main
    bench JSON. Single-file rather than appended so subsequent runs at
    different (N, threads, depth) don't overwrite."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = HERE / "bench-results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}-{label}-profile.json"
    payload = {
        "label": label,
        "timestamp_utc": ts,
        "git_sha": git_sha(),
        "per_N": {
            r.N: {
                "seconds": r.seconds,
                "classes": r.classes,
                "kernel_stats": r.kernel_stats,
                "per_k_stats": r.per_k_stats,
                "parallel_profile": r.parallel_profile,
            }
            for r in results
            if r.parallel_profile is not None
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


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
                "per_k_stats": r.per_k_stats,
            }
            for r in results
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


MAIN_HEADER = f"{'N':>4} {'seconds':>10} {'classes':>8}   per-k (k:count)"
STATS_HEADER = (
    f"{'N':>4} {'calls':>8} {'us/call':>8} "
    f"{'nodes/call':>11} {'tc/call':>9} "
    f"{'lvl/call':>9} {'gen/call':>9}"
)


def format_main_row(r: PerNResult) -> str:
    per_k = ", ".join(
        f"{pk.k}:{pk.classes}" for pk in sorted(r.per_k.values(), key=lambda x: x.k)
    )
    return f"{r.N:>4} {r.seconds:>10.3f} {r.classes:>8}   {per_k}"


def format_stats_row(r: PerNResult) -> str:
    ks = r.kernel_stats
    calls = max(ks.get("canon_calls", 0), 1)
    us = ks.get("nauty_ns", 0) / 1_000.0 / calls
    nodes = ks.get("nauty_numnodes_sum", 0) / calls
    tc = ks.get("nauty_tctotal_sum", 0) / calls
    lvl = ks.get("nauty_maxlevel_sum", 0) / calls
    gen = ks.get("nauty_generators_sum", 0) / calls
    return (
        f"{r.N:>4} {calls:>8} {us:>8.2f} "
        f"{nodes:>11.2f} {tc:>9.2f} "
        f"{lvl:>9.2f} {gen:>9.2f}"
    )


def format_table(results: list[PerNResult]) -> str:
    lines = [MAIN_HEADER]
    for r in results:
        lines.append(format_main_row(r))
    if any(r.kernel_stats for r in results):
        lines.append("")
        lines.append(STATS_HEADER)
        for r in results:
            if r.kernel_stats:
                lines.append(format_stats_row(r))
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
    p.add_argument(
        "--profile-parallel",
        action="store_true",
        help="Call the kernel's `enumerate_doubly_even_with_profile` "
        "entry (requires `--features parallel_profiling`) and dump a "
        "per-worker / per-seed JSON next to the main bench JSON. "
        "DOUBLY_EVEN_THREADS must be set to ≥ 2.",
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
        print(format_table(results), flush=True)
    elif args.profile_parallel:
        nt = _parse_thread_env(_os.environ.get("DOUBLY_EVEN_THREADS"))
        if nt is None or nt < 2:
            raise SystemExit(
                "--profile-parallel needs DOUBLY_EVEN_THREADS >= 2; "
                "otherwise the kernel falls back to the sequential driver."
            )
        Ns = [int(s.strip()) for s in args.N.split(",") if s.strip()]
        results = []
        print(MAIN_HEADER, flush=True)
        for N in Ns:
            r = run_one_with_profile(N, nt)
            results.append(r)
            print(format_main_row(r), flush=True)
        if any(r.kernel_stats for r in results):
            print()
            print(STATS_HEADER, flush=True)
            for r in results:
                if r.kernel_stats:
                    print(format_stats_row(r), flush=True)
        print()
        print("parallel-profile summary:")
        for r in results:
            summary = render_profile_summary(r)
            if summary:
                print(summary)
        profile_path = write_profile_json(args.label, results)
        print(f"\nWrote profile JSON {profile_path}")
    else:
        Ns = [int(s.strip()) for s in args.N.split(",") if s.strip()]
        results = []
        print(MAIN_HEADER, flush=True)
        for N in Ns:
            r = run_one(N)
            results.append(r)
            print(format_main_row(r), flush=True)
        if any(r.kernel_stats for r in results):
            print()
            print(STATS_HEADER, flush=True)
            for r in results:
                if r.kernel_stats:
                    print(format_stats_row(r), flush=True)

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
