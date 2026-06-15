"""End-to-end validation for a streaming-kernel output directory.

Reads every ``out.w*.bin`` file under ``--output-dir``, recomputes:

  - per-rank class counts ``count[k]``
  - per-rank mass ``Σ N!/|Aut(C)|``

and asserts ``mass[k] == gaborit_sigma(N, k)`` for every ``k``. If
``--stats-json`` is also provided, writes a single JSON file with
the validated counts, mass, kernel stats, build info, and git SHA —
the artifact the analyst uses to tune the next-N run.

Usage::

    uv run python scripts/merge_stream.py --N 24 --output-dir /tmp/n24-run
    uv run python scripts/merge_stream.py --N 29 --output-dir /mnt/n29-run \\
        --stats-json /mnt/n29-run/stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.enumerate.stream_reader import count_by_k, sum_mass_by_k  # noqa: E402
from doubly_even.spec.mass import gaborit_sigma  # noqa: E402

# DFGHILM Appendix B Table 3 cells (mirrors scripts/bench.py / tests).
# Cells absent here are unchecked at the table-3 layer; the mass-formula
# check still runs across all k.
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
    (26, 1): 6, (26, 2): 32, (26, 3): 180, (26, 4): 1114, (26, 5): 6923,
    (26, 6): 37455, (26, 7): 128270, (26, 8): 194626, (26, 9): 103527,
    (26, 10): 20206, (26, 11): 1829, (26, 12): 103,
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory containing out.w*.bin files",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="optional path to write the merged stats JSON",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"ERROR: {args.output_dir} is not a directory", file=sys.stderr)
        return 2

    N = args.N
    factorial_N = math.factorial(N)

    counts = count_by_k(args.output_dir, expect_n=N)
    mass = sum_mass_by_k(args.output_dir, factorial_N, expect_n=N)

    # Mass-formula gate: Σ N!/|Aut| at each k must equal σ(N, k).
    max_k = max(counts) if counts else 0
    mass_diffs: list[tuple[int, int, int]] = []
    for k in range(max_k + 1):
        expected = gaborit_sigma(N, k)
        got = mass.get(k, 0)
        if got != expected:
            mass_diffs.append((k, got, expected))

    # DFGHILM Table 3 cross-check on whatever cells we have.
    table_diffs: list[tuple[int, int, int]] = []
    for (n_t, k_t), expected_count in DFGHILM_TABLE_3.items():
        if n_t != N:
            continue
        got = counts.get(k_t, 0)
        if got != expected_count:
            table_diffs.append((k_t, got, expected_count))

    total_classes = sum(counts.values())

    print(f"merge_stream.py: N={N}, output_dir={args.output_dir}")
    print(f"  total classes: {total_classes}")
    print("  per-k counts (k: classes / mass / sigma):")
    for k in sorted(counts):
        print(
            f"    k={k:2d}: {counts[k]:>10d} / "
            f"{mass.get(k, 0):>20d} / {gaborit_sigma(N, k):>20d}"
        )

    ok = True
    if mass_diffs:
        ok = False
        print("\nMASS-FORMULA DISAGREEMENT (Σ N!/|Aut| vs σ(N, k)):")
        for k, got, expected in mass_diffs:
            print(f"  k={k}: got {got}, sigma {expected}")
    else:
        print(f"\nMass formula: all {max_k + 1} ranks match gaborit_sigma(N, k).")

    table3_cells_checked = sum(1 for (n_t, _) in DFGHILM_TABLE_3 if n_t == N)
    if table_diffs:
        ok = False
        print("\nDFGHILM Table 3 DISAGREEMENT:")
        for k, got, expected in table_diffs:
            print(f"  (N={N}, k={k}): got {got}, table {expected}")
    elif table3_cells_checked > 0:
        print(f"DFGHILM Table 3: all {table3_cells_checked} cells for N={N} agree.")
    else:
        print(f"DFGHILM Table 3: no published cells for N={N} (mass formula is the only gate).")

    if args.stats_json is not None:
        payload = {
            "N": N,
            "output_dir": str(args.output_dir.resolve()),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "git_sha": git_sha(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "total_classes": total_classes,
            "per_k": {
                str(k): {
                    "classes": counts[k],
                    "mass": mass.get(k, 0),
                    "gaborit_sigma": gaborit_sigma(N, k),
                }
                for k in sorted(counts)
            },
            "mass_formula_ok": not mass_diffs,
            # null when DFGHILM Table 3 has no cells at this N (nothing to
            # cross-check) — matches scripts/run_counts.py. Avoids a
            # misleading degenerate-True from `not []` at N >= 29.
            "dfghilm_table3_ok": (not table_diffs) if table3_cells_checked > 0 else None,
        }
        args.stats_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print(f"\nWrote {args.stats_json}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
