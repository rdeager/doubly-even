"""Run the counts-only kernel at a single (potentially very large) N and
write a per-rank result JSON in the ``docs/results/n29.json`` format —
the N >= 30 production mode (no per-class records anywhere; output is
KBs, not TBs).

The 256-bit mass spine makes this the ONLY entry valid at N >= 30:
sigma(30, k) reaches 2^136, which overflows both the u128 quota plumbing
of the older entries and pyo3's int conversion (quota travels as decimal
strings here).

Usage::

    # Local N=24 dry-run (the in-memory path also works at this size):
    DOUBLY_EVEN_THREADS=20 DOUBLY_EVEN_FRONTIER_DEPTH=4 \\
        uv run python scripts/run_counts.py --N 24 \\
        --output-dir /tmp/n24-counts

    # GCP c4a (Axion) N=30:
    DOUBLY_EVEN_THREADS=96 DOUBLY_EVEN_FRONTIER_DEPTH=4 \\
        DOUBLY_EVEN_CANON_CACHE_CAP=300000 \\
        uv run python scripts/run_counts.py --N 30 \\
        --output-dir /mnt/scratch/n30

Progress: the kernel atomically rewrites ``<output-dir>/progress.json``
every ``--progress-interval`` seconds (per-rank mass vs quota). Watch it
live from another terminal with::

    uv run dec progress --output-dir <output-dir>

The in-Rust mass-formula gate (mass[k] == sigma(N, k) at every rank)
runs before the kernel returns — any miscount raises a PanicException
and no result JSON is written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.enumerate.augment import _parse_thread_env  # noqa: E402
from doubly_even.spec.mass import gaborit_sigma  # noqa: E402

try:
    import doubly_even_kernel as _kernel
except ImportError as exc:
    print(f"ERROR: doubly_even_kernel not importable: {exc}", file=sys.stderr)
    sys.exit(2)

# DFGHILM Table 3 row totals (all ranks) where published — cross-checked
# when available; N >= 29 has no published cells (mass formula only).
DFGHILM_TOTALS = {
    16: 146, 18: 341, 20: 1211, 22: 5118, 24: 37496, 26: 494272,
    27: 2673492, 28: 21505546,
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def mem_gib() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--max-k", type=int, default=None, help="Default: N // 2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="counts")
    parser.add_argument("--progress-interval", type=int, default=30,
                        help="progress.json rewrite cadence, seconds (default 30)")
    args = parser.parse_args()

    N = args.N
    max_k = args.max_k if args.max_k is not None else N // 2
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    nt = _parse_thread_env(os.environ.get("DOUBLY_EVEN_THREADS")) or 0
    knobs = {
        var: os.environ.get(var, "default")
        for var in (
            "DOUBLY_EVEN_FRONTIER_DEPTH",
            "DOUBLY_EVEN_CANON_CACHE_CAP",
            "DOUBLY_EVEN_PARENT_RULE",
            "DOUBLY_EVEN_CANON_LABELLING",
            "DOUBLY_EVEN_PHI_MAX_RANK",
            "DOUBLY_EVEN_SEEDER_THREADS",
            "DOUBLY_EVEN_SEEDER_PAR_MIN_L",
            "DOUBLY_EVEN_NO_MASS_STOP",
        )
    }

    print("== run_counts.py ==")
    print(f"  label:           {args.label}")
    print(f"  N, max_k:        {N}, {max_k}")
    print(f"  output_dir:      {output_dir.resolve()}")
    print(f"  threads:         {nt} (from DOUBLY_EVEN_THREADS)")
    for var, val in knobs.items():
        if val != "default":
            print(f"  {var}: {val}")
    print(f"  build_info:      {_kernel.kernel_build_info()}")
    print(f"  target features: {_kernel.kernel_target_features()}")
    print(f"  git_sha:         {git_sha()[:12]}")
    print(f"  cpu:             {cpu_model()}")
    print(f"  memory:          {mem_gib():.1f} GiB")
    print()

    quota = [gaborit_sigma(N, k) for k in range(max_k + 1)]
    quota_strs = [str(q) for q in quota]
    progress_path = output_dir / "progress.json"

    t0 = time.perf_counter()
    res = _kernel.enumerate_doubly_even_counts(
        N, max_k, quota_strs, math.factorial(N),
        nt if nt > 0 else None,
        str(progress_path),
        args.progress_interval,
    )
    wall = time.perf_counter() - t0
    print(f"Kernel wall: {wall:.3f} s (in-Rust mass formula PASSED at every rank)")

    mass = [int(s) for s in res["mass"]]
    classes = list(res["classes"])
    aut_hist = res["aut_hist"]
    stats = [int(s) for s in res["stats"]]
    stats_names, per_k_names = _kernel.kernel_stats_layout()
    stats_by_name = {name: stats[i] for i, name in enumerate(stats_names)}

    total_classes = sum(classes)
    print()
    print("per-k classes + mass certificate:")
    for k in range(max_k + 1):
        ok = "ok" if mass[k] == quota[k] else "MISMATCH"
        print(f"  k={k:2d}: classes={classes[k]:>14d}  mass==sigma {ok}")
    print(f"total classes: {total_classes}")

    table3 = DFGHILM_TOTALS.get(N)
    if table3 is not None and total_classes != table3:
        print(f"ERROR: total {total_classes} != published {table3}", file=sys.stderr)
        return 3

    nauty_s = stats_by_name["nauty_ns"] / 1e9
    payload = {
        "N": N,
        "total_classes": total_classes,
        "mass_formula_ok": all(mass[k] == quota[k] for k in range(max_k + 1)),
        "dfghilm_table3_ok": (total_classes == table3) if table3 is not None else None,
        "per_k": {
            str(k): {
                "classes": classes[k],
                "mass": mass[k],
                "gaborit_sigma": quota[k],
                # |Aut| histogram: {aut_order (decimal string): count}.
                # Counts-mode extra over the n29.json baseline schema.
                "aut_histogram": {aut: count for aut, count in aut_hist[k]},
            }
            for k in range(max_k + 1)
        },
        "run_metadata": {
            "git_sha": git_sha(),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "platform": platform.platform(),
            "machine": cpu_model(),
            "memory_gib": round(mem_gib(), 1),
            "python_version": platform.python_version(),
            "build_info": _kernel.kernel_build_info(),
            "target_features": dict(_kernel.kernel_target_features()),
            "num_threads": nt,
            "env_knobs": knobs,
            "output_mode": "counts-only",
            "kernel_wall_seconds": round(wall, 3),
            "wall_hours": round(wall / 3600, 4),
            "canon_calls": stats_by_name["canon_calls"],
            "nauty_seconds": round(nauty_s, 1),
            "us_per_call_mean": round(
                nauty_s * 1e6 / max(stats_by_name["canon_calls"], 1), 2
            ),
        },
        "kernel_stats": stats_by_name,
        "kernel_per_k_stats": {
            name: res["per_k_stats"][i] for i, name in enumerate(per_k_names)
        },
        "schema_note": (
            "Per-k 'mass' and 'gaborit_sigma' are exact Python ints serialised "
            "as JSON integers; equality at every k is the Gaborit mass-formula "
            "certificate. 'aut_histogram' maps |Aut(C)| (decimal string) to the "
            "number of classes with that automorphism-group order — counts-only "
            "runs retain no per-class records, so this histogram plus the "
            "per-rank totals IS the complete output."
        ),
    }
    result_path = output_dir / f"n{N}.json"
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nWrote {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
