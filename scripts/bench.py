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
from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402
from doubly_even.spec.codes import dual_cache_clear, rref_cache_clear  # noqa: E402


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


def run_one(N: int) -> PerNResult:
    factorial_N = math.factorial(N)
    result = PerNResult(N=N)
    # Cold-cache start so the per-N timing isn't contaminated by entries
    # populated during earlier N's. (No cross-N hits in practice since the
    # key includes ``n``, but this is defensive and makes runs comparable.)
    canon_info_cache_clear()
    rref_cache_clear()
    dual_cache_clear()
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
