"""Tests for :mod:`doubly_even.enumerate.experimental.witt`.

The witt module's role in phase (b) is intentionally minimal — per the
plan-doc directive that the structural win is in orbit enumeration and
``singular_vectors`` should re-export the existing ``singular_reps_Q``.
Tests here pin down:

1. ``count_singular`` closed-form values agree with hand calculation.
2. ``singular_vectors`` is the same as ``singular_reps_Q`` (alias check).
3. The closed-form count matches direct enumeration for every parent
   where the underlying ``(Q_C, q)`` is **non-degenerate** — this is
   the case when the all-ones vector lies in ``C`` (so ``C⊥`` consists
   entirely of even-weight vectors). For other parents we don't expect
   the formula to apply, since the radical isn't trivial.
"""

from __future__ import annotations

import pytest

from doubly_even.enumerate.augment import enumerate_doubly_even
from doubly_even.enumerate.quotient import Q_basis, singular_reps_Q
from doubly_even.enumerate.experimental.witt import count_singular, singular_vectors


# ---------------------------------------------------------- count_singular


def test_count_singular_known_values():
    # m=1, "+": hyperbolic plane, 2 nonzero singular (the two isotropic lines).
    assert count_singular(1, "+") == 2
    # m=1, "-": anisotropic 2-dim form, 0 nonzero singular.
    assert count_singular(1, "-") == 0
    # m=2, "+": (2^2 − 1)(2^1 + 1) = 3·3 = 9.
    assert count_singular(2, "+") == 9
    # m=2, "-": (2^2 + 1)(2^1 − 1) = 5·1 = 5.
    assert count_singular(2, "-") == 5
    # m=3, "+": 7·5 = 35; m=3, "-": 9·3 = 27.
    assert count_singular(3, "+") == 35
    assert count_singular(3, "-") == 27
    # Parabolic dim 3 (m=1, "0"): 2^2 − 1 = 3.
    assert count_singular(1, "0") == 3
    # Parabolic dim 5 (m=2, "0"): 2^4 − 1 = 15.
    assert count_singular(2, "0") == 15


def test_count_singular_trivial_m():
    for eps in ("+", "-", "0"):
        assert count_singular(0, eps) == 0
        assert count_singular(-1, eps) == 0


def test_count_singular_bad_sign():
    with pytest.raises(ValueError, match="unknown Witt sign"):
        count_singular(2, "?")


# ----------------------------------------------------- singular_vectors


CROSS_CHECK_NS = [4, 6, 8, 10, 12]


def _all_parents(N: int):
    for ec in enumerate_doubly_even(N):
        yield ec.code, ec.info.aut_generators


def test_singular_vectors_is_alias():
    for N in CROSS_CHECK_NS:
        for C, _ in _all_parents(N):
            V_basis, _ = Q_basis(C)
            assert singular_vectors(V_basis) == singular_reps_Q(V_basis)


def test_singular_vectors_empty_L():
    # Self-dual parent has L = 0 → no singular Q-coords.
    assert singular_vectors(()) == []


# ----------------------- closed form vs direct count, non-degenerate parents


def _all_ones_in_code(C) -> bool:
    """``1...1`` is in ``C`` iff every dual basis vector has even weight."""
    all_ones = (1 << C.n) - 1
    return all_ones in C


def test_closed_form_matches_for_non_degenerate_parents():
    """When ``1...1 ∈ C``, ``C⊥ ⊆ even-weight``, so ``(Q_C, q)`` is
    non-degenerate and the singular count matches one of the closed
    forms. (When ``1...1 ∉ C`` the form has a non-trivial radical and
    the formula doesn't apply directly; those parents are skipped.)
    """
    checked = 0
    for N in CROSS_CHECK_NS:
        for C, _ in _all_parents(N):
            if not _all_ones_in_code(C):
                continue
            V_basis, _ = Q_basis(C)
            L = len(V_basis)
            if L == 0:
                continue
            actual = len(singular_reps_Q(V_basis))
            # Try every closed-form type at the matching L.
            candidates: list[int] = []
            if L % 2 == 0:
                m = L // 2
                candidates += [count_singular(m, "+"), count_singular(m, "-")]
            else:
                m = (L - 1) // 2
                candidates += [count_singular(m, "0")]
            assert actual in candidates, (
                f"non-degenerate parent N={N}, k={C.rank}, "
                f"C={list(C.basis)}: singular count {actual} matches no "
                f"closed-form type at L={L}"
            )
            checked += 1
    # Make sure we exercised the formula on at least a few parents.
    assert checked > 0, "no non-degenerate parents found in cross-check range"
