"""The ``dec`` command-line entry point (``uv run dec …``).

Subcommands:

- ``dec progress --output-dir DIR [--N N] [--interval S]`` — live
  per-rank mass-percentage table for a running enumeration (streaming
  or counts-only run, auto-detected). The full option set is documented
  in :mod:`doubly_even.enumerate.progress`.
- ``dec info`` — installed kernel build/feature/knob report.
"""

from __future__ import annotations

import os
import sys


def _info() -> int:
    try:
        import doubly_even_kernel as kernel
    except ImportError as exc:
        print(f"doubly_even_kernel not importable: {exc}", file=sys.stderr)
        print("build it with: scripts/install-kernel.sh parallel", file=sys.stderr)
        return 2
    print(f"module:          {kernel.__file__}")
    print(f"build_info:      {kernel.kernel_build_info()}")
    print(f"target features: {dict(kernel.kernel_target_features())}")
    names, per_k = kernel.kernel_stats_layout()
    print(f"stats layout:    {len(names)} fields / {len(per_k)} per-k rows")
    print("env knobs (unset = kernel default):")
    for var in (
        "DOUBLY_EVEN_THREADS",
        "DOUBLY_EVEN_FRONTIER_DEPTH",
        "DOUBLY_EVEN_CANON_CACHE_CAP",
        "DOUBLY_EVEN_PARENT_RULE",
        "DOUBLY_EVEN_CANON_LABELLING",
        "DOUBLY_EVEN_PHI_MAX_RANK",
        "DOUBLY_EVEN_SEEDER_THREADS",
        "DOUBLY_EVEN_SEEDER_PAR_MIN_L",
        "DOUBLY_EVEN_NO_MASS_STOP",
        "DOUBLY_EVEN_TIE_DUMP",
        "DOUBLY_EVEN_DECOMP_LOG",
    ):
        val = os.environ.get(var)
        print(f"  {var} = {val if val is not None else '(unset)'}")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "progress":
        from doubly_even.enumerate.progress import main as progress_main

        return progress_main(rest)
    if cmd == "info":
        return _info()
    print(f"unknown subcommand: {cmd!r}\n", file=sys.stderr)
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
