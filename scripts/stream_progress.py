"""Thin wrapper kept for the recipes in ``docs/`` — the sidecar body
moved into the package 2026-06-13 (counts-mode support + the ``dec``
CLI). Equivalent invocations::

    uv run python scripts/stream_progress.py --N 29 --output-dir DIR
    uv run dec progress --N 29 --output-dir DIR

See :mod:`doubly_even.enumerate.progress` for the full documentation
(streaming + counts-only sources, auto-detection, exit signals).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from doubly_even.enumerate.progress import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
