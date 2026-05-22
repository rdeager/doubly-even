"""Run the streaming kernel at a single (potentially large) N and write
per-worker binary files + a stats.json. Designed for N >= 28 production
runs where the in-memory Vec<EnumeratedRaw> output (used by
``scripts/bench.py``) would OOM.

Usage::

    # Local N=24 dry-run (small enough that bench.py also works):
    DOUBLY_EVEN_THREADS=20 DOUBLY_EVEN_FRONTIER_DEPTH=5 \\
        uv run python scripts/run_streaming.py --N 24 \\
        --output-dir /tmp/n24-stream

    # c4d-standard-384-metal N=29:
    DOUBLY_EVEN_THREADS=384 DOUBLY_EVEN_FRONTIER_DEPTH=5 \\
        DOUBLY_EVEN_CANON_CACHE_CAP=500000 \\
        uv run python scripts/run_streaming.py --N 29 \\
        --output-dir /mnt/scratch/n29

The kernel's in-Rust mass-formula gate runs before this script returns
— any mismatch is a fatal PanicException. After the kernel returns,
``scripts/merge_stream.py`` (called automatically unless ``--skip-merge``
is passed) re-walks the binary files to cross-check the assertion and
populate ``stats.json`` with DFGHILM Table 3 cross-checks, per-k mass,
git SHA, and platform info.
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
    parser.add_argument("--max-k", type=int, default=None,
                        help="Default: N // 2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="streaming")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip the post-run merge_stream.py validation")
    args = parser.parse_args()

    N = args.N
    max_k = args.max_k if args.max_k is not None else N // 2
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    nt = _parse_thread_env(os.environ.get("DOUBLY_EVEN_THREADS")) or 0
    frontier_depth = os.environ.get("DOUBLY_EVEN_FRONTIER_DEPTH", "default")
    canon_cap = os.environ.get("DOUBLY_EVEN_CANON_CACHE_CAP", "default (500000)")

    print(f"== run_streaming.py ==")
    print(f"  label:           {args.label}")
    print(f"  N, max_k:        {N}, {max_k}")
    print(f"  output_dir:      {output_dir.resolve()}")
    print(f"  threads:         {nt} (from DOUBLY_EVEN_THREADS)")
    print(f"  frontier depth:  {frontier_depth}")
    print(f"  canon cache cap: {canon_cap}")
    print(f"  build_info:      {_kernel.kernel_build_info()}")
    print(f"  git_sha:         {git_sha()[:12]}")
    print(f"  cpu:             {cpu_model()}")
    print(f"  memory:          {mem_gib():.1f} GiB")
    print()

    quota = [gaborit_sigma(N, k) for k in range(max_k + 1)]
    factorial_N = math.factorial(N)

    t0 = time.perf_counter()
    res = _kernel.enumerate_doubly_even_streaming(
        N, max_k, quota, factorial_N, str(output_dir),
        nt if nt > 0 else None,
    )
    wall = time.perf_counter() - t0
    print(f"Kernel wall: {wall:.3f} s (in-Rust mass formula PASSED)")

    # Convert mass + stats from decimal strings back to int.
    mass = [int(s) for s in res["mass"]]
    stats = [int(s) for s in res["stats"]]
    per_k_stats = res["per_k_stats"]
    print()
    print("per-k mass = N!/|Aut| summed (matches σ(N, k) by in-Rust assertion):")
    for k in range(max_k + 1):
        print(f"  k={k:2d}: mass={mass[k]:>22d}  σ={quota[k]:>22d}")

    # Persist the run-level stats.json (kernel-side data).
    stats_path = output_dir / "stats.json"
    payload = {
        "label": args.label,
        "N": N,
        "max_k": max_k,
        "num_threads": nt,
        "frontier_depth": frontier_depth,
        "canon_cache_cap": canon_cap,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_sha": git_sha(),
        "build_info": _kernel.kernel_build_info(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "memory_gib": mem_gib(),
        "kernel_wall_seconds": wall,
        # mass[k] is the validated sum N!/|Aut| at rank k.
        "mass": mass,
        "quota_sigma": quota,
        # See rust/src/enumerate.rs::enumerate_doubly_even doc for the
        # stats / per_k_stats field layouts (26 fields and 10 rows
        # respectively). Stored verbatim for tuning.
        "kernel_stats_raw": stats,
        "kernel_per_k_stats_raw": per_k_stats,
    }
    stats_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nWrote {stats_path}")

    if not args.skip_merge:
        merge_path = HERE / "merge_stream.py"
        merge_json = output_dir / "merge_stream.json"
        print(f"\n== Cross-check via merge_stream.py ==")
        r = subprocess.run(
            [
                sys.executable, str(merge_path),
                "--N", str(N),
                "--output-dir", str(output_dir),
                "--stats-json", str(merge_json),
            ],
            check=False,
        )
        if r.returncode != 0:
            print("\nERROR: merge_stream.py rejected the run.", file=sys.stderr)
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
