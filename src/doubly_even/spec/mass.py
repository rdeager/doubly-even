"""Mass-formula utilities.

``σ(N, k)`` is the number of *labelled* doubly even ``[N, k]`` codes — i.e.
the number of ``k``-dimensional doubly even subspaces of ``F_2^N``. We need
it because completeness of enumeration is certified by

    Σ_{[C]} N! / |Aut(C)| == σ(N, k).

There is a closed-form expression (Gaborit 1996) that handles all ``N`` and
``k``. We use the *quadratic-space* recast (cleaner around ``N ≡ 0, 4 mod 8``
than the original DFGHILM B.1 form): doubly even codes are totally singular
subspaces of the binary quadratic form ``q(x) = wt(x)/2 mod 2`` on the
even-weight hyperplane. Letting

    Ω_ε(2m, t) = Π_{i=0}^{t-1} (2^{2m-2i-1} + ε·2^{m-i-1} − 1) / (2^{i+1} − 1)

with the convention "empty product = 1, ``t < 0`` or ``t > m`` is outside
the Witt range and contributes 0", we have

    σ(N, k) =
        Ω_+(N-1, k)                                    if N ≡ 1, 7 (mod 8)
        Ω_-(N-1, k)                                    if N ≡ 3, 5 (mod 8)
        Π_{i=0}^{k-1} (2^{N-2i-2} − 1) / (2^{i+1} − 1)  if N ≡ 2, 6 (mod 8)
        Ω_+(N-2, k-1) + 2^k · Ω_+(N-2, k)              if N ≡ 0 (mod 8)
        Ω_-(N-2, k-1) + 2^k · Ω_-(N-2, k)              if N ≡ 4 (mod 8)

This module exposes:

* :func:`sigma_brute` — direct enumeration. Slow but correct. Tractable for
  ``N`` up to about 10. Stays as the ground-truth oracle.
* :func:`gaborit_sigma` — the closed form above. Cached.

The two functions are cross-checked in ``tests/test_mass_small.py`` for
every ``(N, k)`` with ``N ≤ 10``.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
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


def _omega(m: int, t: int, eps: int) -> int:
    """The ``Ω_ε(2m, t)`` factor of Gaborit's quadratic-space form.

    Returns ``0`` for ``t < 0`` or ``t > m`` (outside the Witt range).
    The product is computed over :class:`fractions.Fraction` so the
    individual (non-integer) factors don't lose precision before the
    whole product simplifies to an integer.
    """
    if t < 0 or t > m:
        return 0
    if t == 0:
        return 1
    result = Fraction(1)
    for i in range(t):
        numer = (1 << (2 * m - 2 * i - 1)) + eps * (1 << (m - i - 1)) - 1
        denom = (1 << (i + 1)) - 1
        result *= Fraction(numer, denom)
    if result.denominator != 1:
        raise RuntimeError(
            f"Ω_{'+' if eps > 0 else '-'}({2 * m}, {t}) did not simplify to "
            f"an integer: got {result}; formula or domain bug"
        )
    return int(result)


@lru_cache(maxsize=None)
def gaborit_sigma(N: int, k: int) -> int:
    """Closed-form ``σ(N, k)``, the number of labelled doubly even
    ``[N, k]`` codes.

    Uses the quadratic-space form documented at the top of this module.
    Cross-checked against :func:`sigma_brute` for ``N ≤ 10`` and against
    the enumerator's mass output ``Σ N!/|Aut(C_i)|`` for ``N ≤ 18``.
    """
    if k < 0 or N < 0:
        raise ValueError(f"need N, k ≥ 0, got N={N}, k={k}")
    if k == 0:
        return 1
    if 2 * k > N:
        return 0

    r = N % 8
    if r in (1, 7):
        return _omega((N - 1) // 2, k, +1)
    if r in (3, 5):
        return _omega((N - 1) // 2, k, -1)
    if r in (2, 6):
        # Π_{i=0}^{k-1} (2^{N-2i-2} − 1) / (2^{i+1} − 1)
        result = Fraction(1)
        for i in range(k):
            result *= Fraction(
                (1 << (N - 2 * i - 2)) - 1,
                (1 << (i + 1)) - 1,
            )
        if result.denominator != 1:
            raise RuntimeError(
                f"σ({N}, {k}) [N≡2,6 mod 8 branch] did not simplify to "
                f"an integer: got {result}"
            )
        return int(result)
    if r == 0:
        m = (N - 2) // 2
        return _omega(m, k - 1, +1) + (1 << k) * _omega(m, k, +1)
    if r == 4:
        m = (N - 2) // 2
        return _omega(m, k - 1, -1) + (1 << k) * _omega(m, k, -1)
    raise AssertionError(f"unreachable: N % 8 = {r}")


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
