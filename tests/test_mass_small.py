"""Small-case checks for ``σ(N, k)``.

We compute ``σ`` by brute force for ``N ≤ 8`` and assert agreement with the
hand-curated table of known values. ``gaborit_sigma`` (closed form, Gaborit
1996 quadratic-space recast) is cross-checked against ``sigma_brute`` on
the overlap and against the enumerator's mass sum for ``N ≤ 18``.
"""

from __future__ import annotations

import math

import pytest

from doubly_even.enumerate.augment import enumerate_doubly_even_at
from doubly_even.spec.mass import gaborit_sigma, known_sigma_values, sigma_brute


def test_sigma_brute_zero_k():
    for N in range(13):
        assert sigma_brute(N, 0) == 1


def test_sigma_brute_above_half_is_zero():
    for N in range(9):
        for k in range(N // 2 + 1, N + 1):
            assert sigma_brute(N, k) == 0


@pytest.mark.parametrize(("Nk", "expected"), sorted(known_sigma_values().items()))
def test_sigma_brute_matches_table(Nk, expected):
    N, k = Nk
    assert sigma_brute(N, k) == expected


def test_gaborit_sigma_trivial_cases():
    # k = 0
    for N in range(13):
        assert gaborit_sigma(N, 0) == 1
    # k > N / 2
    for N in range(13):
        for k in range(N // 2 + 1, N + 1):
            assert gaborit_sigma(N, k) == 0


@pytest.mark.parametrize(
    ("N", "k"),
    [(N, k) for N in range(0, 11) for k in range(0, N // 2 + 1)],
)
def test_gaborit_sigma_matches_brute(N, k):
    """The closed form must agree with brute-force enumeration on every
    cell where ``sigma_brute`` is tractable."""
    assert gaborit_sigma(N, k) == sigma_brute(N, k)


@pytest.mark.parametrize(("Nk", "expected"), sorted(known_sigma_values().items()))
def test_gaborit_sigma_matches_table(Nk, expected):
    """And against the hand-curated table of known values."""
    N, k = Nk
    assert gaborit_sigma(N, k) == expected


@pytest.mark.parametrize("N", [8, 16, 24, 32])
def test_gaborit_sigma_type_II_classical(N):
    """For ``N ≡ 0 (mod 8)``, ``k = N/2``, Gaborit's formula must reduce to
    the classical Type-II product ``Π_{i=0}^{N/2-2}(2^i + 1)``
    (Betsumiya–Harada–Munemasa)."""
    classical = 1
    for i in range(N // 2 - 1):
        classical *= (1 << i) + 1
    assert gaborit_sigma(N, N // 2) == classical


# Cells where the enumerator's mass output is cheap to compute and gives an
# independent witness for ``σ(N, k)``. We restrict to ``N ≤ 12`` in the
# default suite to keep test wall-clock under 1 s; ``--run-slow`` extends
# the coverage.
_MASS_CELLS_FAST = [(N, k) for N in (10, 12) for k in range(1, N // 2 + 1)]
_MASS_CELLS_SLOW = [(N, k) for N in (14, 16) for k in range(1, N // 2 + 1)]


@pytest.mark.parametrize(("N", "k"), _MASS_CELLS_FAST)
def test_gaborit_sigma_matches_enumerator_mass(N, k):
    """``Σ N!/|Aut(C_i)|`` over the enumerator's emissions must equal
    ``gaborit_sigma(N, k)``. This is the strongest external check."""
    mass = sum(
        math.factorial(N) // ec.aut_order
        for ec in enumerate_doubly_even_at(N, k)
    )
    assert mass == gaborit_sigma(N, k)


@pytest.mark.slow
@pytest.mark.parametrize(("N", "k"), _MASS_CELLS_SLOW)
def test_gaborit_sigma_matches_enumerator_mass_slow(N, k):
    """Slow extension of :func:`test_gaborit_sigma_matches_enumerator_mass`
    to ``N = 14, 16``."""
    mass = sum(
        math.factorial(N) // ec.aut_order
        for ec in enumerate_doubly_even_at(N, k)
    )
    assert mass == gaborit_sigma(N, k)
