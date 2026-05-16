"""Mass-formula utilities.

``σ(N, k)`` is the number of *labelled* doubly even ``[N, k]`` codes — i.e.
the number of ``k``-dimensional doubly even subspaces of ``F_2^N``. We need
it because completeness of enumeration is certified by

    Σ_{[C]} N! / |Aut(C)| == σ(N, k).

There is a closed-form expression (Gaborit 1996) that handles all ``N`` and
``k``. We do not yet have Gaborit's paper — DFGHILM eq. (B.2) provides a
transcription, but the Mathpix rendering is partially mangled and the
``N ≡ 0 (mod 8)`` branch does not reproduce the small cases. Until we have
the original paper to verify, this module exposes:

* :func:`sigma_brute` — direct enumeration. Slow but correct. Tractable for
  ``N`` up to about 10.
* :func:`gaborit_sigma` — provisional closed form. Currently raises
  ``NotImplementedError`` for cases we have not verified.

When Gaborit's paper arrives we will fill in :func:`gaborit_sigma` and add
agreement tests against :func:`sigma_brute` on the overlap.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from .codes import Code
from .vectors import wt


# ----------------------------------------------------------------- brute force


def sigma_brute(N: int, k: int) -> int:
    """Return ``σ(N, k)`` by direct enumeration. ``O(σ(N, k) * 2^N)`` time.

    Tractable for ``N`` up to about 10. For larger ``N``, this is replaced by
    :func:`gaborit_sigma` once that is verified against Gaborit's paper.
    """
    if k < 0 or N < 0:
        raise ValueError(f"need N, k ≥ 0, got N={N}, k={k}")
    if k == 0:
        return 1
    counts = _sigma_brute_table(N)
    return counts.get(k, 0)


@lru_cache(maxsize=None)
def _sigma_brute_table(N: int) -> dict[int, int]:
    """Return ``{k: σ(N, k)}`` for ``k = 0, 1, …``.

    Walks the augmentation tree from the zero code. Each *subspace* is
    represented by the frozen set of its codewords so that we deduplicate by
    set equality.
    """
    counts: dict[int, int] = {0: 1}
    seen: set[frozenset[int]] = {frozenset({0})}
    frontier: list[tuple[Code, frozenset[int]]] = [
        (Code.zero(N), frozenset({0}))
    ]

    while frontier:
        next_frontier: list[tuple[Code, frozenset[int]]] = []
        for C, words in frontier:
            for v in _candidates(C, words):
                words2 = frozenset(w ^ v for w in words) | words
                if words2 in seen:
                    continue
                seen.add(words2)
                C2 = C.extend(v)
                next_frontier.append((C2, words2))
                counts[C2.rank] = counts.get(C2.rank, 0) + 1
        frontier = next_frontier
    return counts


def _candidates(C: Code, words: frozenset[int]) -> Iterator[int]:
    """Vectors ``v ⊥ C`` with ``wt(v) ≡ 0 (mod 4)`` and ``v ∉ C``."""
    for v in range(1 << C.n):
        if v in words:
            continue
        if wt(v) % 4 != 0:
            continue
        if C.is_orthogonal_to(v):
            yield v


# -------------------------------------------------------------- closed form


def gaborit_sigma(N: int, k: int) -> int:
    """Provisional closed form for ``σ(N, k)``.

    **Currently unimplemented.** The DFGHILM Mathpix transcription of
    Gaborit's formula does not reproduce small cases for ``N ≡ 0 (mod 8)``
    (concretely, the ``ϖ`` correction factor is wrong or the wrong sign).
    Rather than guess, we raise here and let the caller fall back to
    :func:`sigma_brute` until we have Gaborit's paper to verify the formula.

    Cases handled trivially:

    * ``k == 0``: returns 1 (the zero code).
    * ``2k > N``: returns 0 (no doubly even code can be larger than half-rank).
    """
    if k < 0 or N < 0:
        raise ValueError(f"need N, k ≥ 0, got N={N}, k={k}")
    if k == 0:
        return 1
    if 2 * k > N:
        return 0
    raise NotImplementedError(
        "gaborit_sigma needs verification against Gaborit (1996); "
        f"use sigma_brute({N}, {k}) for small N in the meantime"
    )


# ------------------------------------------------------------------- helpers


def known_sigma_values() -> dict[tuple[int, int], int]:
    """A small hand-curated table of ``σ(N, k)`` values for tests.

    Entries here have been cross-checked against either direct counting or
    published literature; see notes in :mod:`doubly_even.spec.mass`.
    """
    return {
        # σ(N, 0) = 1 trivially; we record a few for sanity.
        (4, 0): 1, (5, 0): 1, (6, 0): 1, (7, 0): 1, (8, 0): 1,
        # σ(N, 1) = # nonzero vectors with wt ≡ 0 mod 4.
        (4, 1): 1,      # only the all-ones vector
        (5, 1): 5,      # C(5,4) = 5
        (6, 1): 15,     # C(6,4) = 15
        (7, 1): 35,     # C(7,4) = 35
        (8, 1): 71,     # 1 + C(8,4) + C(8,8) - 1 = 71
        # σ(N, k) for k > 1 enumerated by brute force in this codebase.
        (6, 2): 15,
        (7, 2): 105, (7, 3): 30,
        (8, 2): 455, (8, 3): 345,
        # σ(8, 4) = 30: unique Type II [8,4] code, |Aut| = 1344, orbit 8!/1344.
        (8, 4): 30,
    }
