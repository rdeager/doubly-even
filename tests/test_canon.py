"""Tests for the canonicalisation layer."""

from __future__ import annotations

import math

import pytest

from doubly_even.canon.nauty import (
    are_equivalent,
    canon_info,
    canonical_form,
)
from doubly_even.spec.codes import Code
from doubly_even.spec.vectors import apply_permutation


# --------------------------------------------------------- automorphism order


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
def test_zero_code_aut_is_n_factorial(n):
    C = Code.zero(n)
    info = canon_info(C)
    assert info.aut_order == math.factorial(n)


@pytest.mark.parametrize("n", [16, 20, 24, 32])
def test_zero_code_aut_exact_beyond_double_precision(n):
    """At n=20 the rounded float result already drifts from n!. Schreier–Sims
    in :mod:`doubly_even.canon.permutations` recomputes the order exactly."""
    C = Code.zero(n)
    info = canon_info(C)
    assert info.aut_order == math.factorial(n)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
def test_whole_space_aut_is_n_factorial(n):
    C = Code.whole(n)
    info = canon_info(C)
    assert info.aut_order == math.factorial(n)


def test_single_weight4_vector_in_F2_8():
    """Aut of ⟨v⟩ where v has weight 4 in F_2^8 = S_4 × S_4 = 576."""
    C = Code(8, (0b00001111,))
    info = canon_info(C)
    assert info.aut_order == 24 * 24


def test_extended_hamming_8_4():
    """The extended Hamming [8,4,4] code has |Aut| = 1344 = AGL(3, 2)."""
    rows = (
        0b00001111,
        0b00110011,
        0b01010101,
        0b11111111,
    )
    C = Code(8, rows)
    info = canon_info(C)
    assert info.aut_order == 1344


# ------------------------------------------------------- column-orbit basics


def test_zero_code_one_column_orbit():
    C = Code.zero(6)
    info = canon_info(C)
    # All columns are in the same orbit under S_n.
    assert len(set(info.column_orbits)) == 1


def test_weight4_vector_two_column_orbits():
    # Columns 0..3 are the support of v, columns 4..7 are not.
    C = Code(8, (0b00001111,))
    info = canon_info(C)
    # Two orbits of size 4.
    counts = {o: info.column_orbits.count(o) for o in set(info.column_orbits)}
    assert sorted(counts.values()) == [4, 4]


# ---------------------------------------------------------- canonical form


def test_canonical_form_is_idempotent():
    C = Code(8, (0b00001111, 0b00110011))
    once = canonical_form(C)
    twice = canonical_form(once)
    assert once == twice


def test_canonical_form_invariant_under_permutation():
    """Permuting columns of a code should leave the canonical form unchanged."""
    C = Code(8, (0b00001111, 0b00110011, 0b01010101))
    canon = canonical_form(C)
    # A handful of column permutations
    permutations = [
        [1, 0, 2, 3, 4, 5, 6, 7],         # swap 0,1
        [7, 6, 5, 4, 3, 2, 1, 0],         # reverse
        [2, 4, 6, 0, 1, 3, 5, 7],         # arbitrary
    ]
    for sigma in permutations:
        permuted_basis = tuple(apply_permutation(b, sigma) for b in C.basis)
        C_perm = Code(8, permuted_basis)
        assert canonical_form(C_perm) == canon


# ----------------------------------------------------- equivalence detection


def test_are_equivalent_self():
    C = Code(8, (0b00001111,))
    assert are_equivalent(C, C)


def test_are_equivalent_permuted():
    C = Code(8, (0b00001111,))
    # Move the support from columns 0..3 to columns 4..7
    C2 = Code(8, (0b11110000,))
    assert are_equivalent(C, C2)


def test_are_inequivalent_different_rank():
    C1 = Code(8, (0b00001111,))
    C2 = Code(8, (0b00001111, 0b11110000))
    assert not are_equivalent(C1, C2)


def test_are_inequivalent_different_weight_distribution():
    # Same rank (1), but supports of different weights.
    C1 = Code(8, (0b00001111,))   # weight 4
    C2 = Code(8, (0b11111111,))   # weight 8
    assert not are_equivalent(C1, C2)


def test_are_inequivalent_different_n():
    C1 = Code(8, (0b00001111,))
    C2 = Code(7, (0b0001111,))
    assert not are_equivalent(C1, C2)


# ------------------------------------------------- aut generators preserve C


def test_aut_generators_preserve_code():
    """Every claimed automorphism generator must actually map C to itself."""
    rows = (
        0b00001111,
        0b00110011,
        0b01010101,
        0b11111111,
    )
    C = Code(8, rows)
    info = canon_info(C)
    for sigma in info.aut_generators:
        sigma_list = list(sigma)
        for b in C.basis:
            permuted = apply_permutation(b, sigma_list)
            assert permuted in C, (
                f"generator {sigma} sent basis vector {b:#010b} outside C"
            )
