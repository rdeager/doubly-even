"""Small-case checks for ``σ(N, k)``.

We compute ``σ`` by brute force for ``N ≤ 8`` and assert agreement with the
hand-curated table of known values. When :func:`gaborit_sigma` is eventually
implemented, an additional cross-check will go here.
"""

from __future__ import annotations

import pytest

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


def test_gaborit_sigma_unimplemented_on_real_cases():
    # Until we have Gaborit's paper to verify, the closed form should not be
    # silently returning bogus values.
    with pytest.raises(NotImplementedError):
        gaborit_sigma(8, 4)
