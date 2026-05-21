"""Pedagogical pure-Python doubly-even code enumerator (standalone, ≤ 425 LOC).

A clean companion to the production Rust kernel. Implements DFGHILM Appendix
B.4 canonical augmentation, with the two algorithmic improvements that buy
≥ 2× over the Appendix B baseline carried over verbatim:

  **Improvement #1**: ``Q_C``-coordinate quotient enumeration (``_qc.py``).
      Candidate cosets are enumerated in ``Q_C := C^⊥ / C`` (dim ``N − 2k``),
      not in the ambient ``F_2^N`` (dim ``N``).

  **Improvement #2**: minimum-weight incidence canonicaliser (``_canon.py``).
      The bipartite graph fed to nauty has only minimum-weight codewords on
      the left, not all ``2^k``. Falls back to a min-weight scan for codes
      (like extended Golay ``[24, 12, 8]``) whose min weight exceeds 4.

Production extras intentionally omitted (each only buys ≤ 1.2× in Rust —
together they add up, but not enough to earn their LOC in a pedagogical
Python module): mass-stop via Gaborit σ (D5b), degree-based initial vertex
partition for nauty (D8), weight-enumerator BFS prefilter (D4), two-tier
canon cache (D3). See ``rust/src/`` in the repo for the production versions.

The clean module imports nothing from ``doubly_even.*`` or the Rust kernel —
fully standalone. The Gaborit σ closed form is *copied verbatim* from
``doubly_even/spec/mass.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

from ._augment import EnumeratedCode, enumerate_doubly_even_sequential
from ._canon import CanonInfo
from ._mass import gaborit_sigma
from ._parallel import enumerate_doubly_even_parallel
from ._spec import Code

__all__ = [
    "Code",
    "CanonInfo",
    "EnumeratedCode",
    "enumerate_doubly_even",
    "gaborit_sigma",
]


def enumerate_doubly_even(
    N: int,
    max_k: int | None = None,
    workers: int = 1,
    frontier_depth: int = 3,
) -> Iterator[EnumeratedCode]:
    """Yield one canonical representative per doubly-even ``[N, k]`` equivalence class.

    ``workers <= 1`` runs sequentially. ``workers >= 2`` uses a process pool,
    splitting the McKay tree at ``frontier_depth`` (default 3; bump to 4 for N=24).
    """
    if workers <= 1:
        yield from enumerate_doubly_even_sequential(N, max_k)
    else:
        yield from enumerate_doubly_even_parallel(
            N, max_k, workers=workers, frontier_depth=frontier_depth
        )
