"""Tests for the candidate-extension filters."""

from __future__ import annotations

import pytest

from doubly_even.canon.nauty import canon_info
from doubly_even.enumerate.augment import enumerate_doubly_even
from doubly_even.enumerate.filters import (
    aut_orbit_minima,
    coset_reps_in_dual_mod_code,
    doubly_even_candidates,
    reduce_mod_code,
    standard_form_coset_reps,
    weight_mod_four_zero,
)
from doubly_even.spec.codes import Code
from doubly_even.spec.vectors import wt


# --------------------------------------------------------------- reduction


def test_reduce_mod_code_zero_in():
    C = Code(4, (0b0011,))
    assert reduce_mod_code(0b0011, C) == 0
    assert reduce_mod_code(0, C) == 0


def test_reduce_mod_code_cosets_stable():
    """All elements of the same coset reduce to the same rep."""
    C = Code(8, (0b00001111,))  # weight-4 generator
    # 0b00001111 + 0b11110000 = 0b11111111 is one coset
    # 0b00000001 + 0b00001111 = 0b00001110 is another's element
    r1 = reduce_mod_code(0b11110000, C)
    r2 = reduce_mod_code(0b11110000 ^ 0b00001111, C)
    assert r1 == r2


# ------------------------------------------------------- coset enumeration


def test_coset_reps_count():
    """Number of cosets of C in C⊥ is 2^(n - 2k)."""
    C = Code(8, (0b00001111,))   # k = 1, dual has rank 7, so 2^(8-2) = 64 cosets
    reps = list(coset_reps_in_dual_mod_code(C))
    assert len(reps) == 2 ** (8 - 2 * 1)


def test_coset_reps_include_zero():
    C = Code(6, (0b001111,))
    reps = list(coset_reps_in_dual_mod_code(C))
    assert 0 in reps


# ----------------------------------- standard-form (B.3) quotient enumeration


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_standard_form_matches_dual_enum(N):
    """Across every doubly even code at length N, standard_form_coset_reps
    agrees set-wise with the reduce(coset_reps_in_dual_mod_code) reference.
    """
    for ec in enumerate_doubly_even(N):
        C = ec.code
        old = {reduce_mod_code(v, C) for v in coset_reps_in_dual_mod_code(C)}
        new = set(standard_form_coset_reps(C))
        assert old == new, (
            f"standard_form_coset_reps disagrees with dual enum at "
            f"N={N}, k={C.rank}, basis={list(C.basis)!r}"
        )


@pytest.mark.parametrize(
    "C",
    [
        Code.zero(4),
        Code.zero(8),
        Code(8, (0b00001111,)),
        Code(8, (0b00001111, 0b11110000)),
    ],
)
def test_standard_form_count_matches_quotient_dim(C):
    """For doubly even ``C``, ``standard_form_coset_reps`` enumerates
    exactly ``2^(n - 2k)`` cosets — the dimension of ``C⊥ / C``."""
    reps = list(standard_form_coset_reps(C))
    assert len(reps) == 2 ** (C.n - 2 * C.rank)


def test_standard_form_reps_zero_on_pivots():
    """Every emitted rep should have zero bits at C's pivot columns."""
    C = Code(8, (0b00001111, 0b11110000))
    _, pivots = C.rref_basis()
    mask = sum(1 << p for p in pivots)
    for v in standard_form_coset_reps(C):
        assert v & mask == 0


# ----------------------------------------------------- weight-mod-4 filter


def test_weight_filter_drops_zero_and_bad_weights():
    cands = [0, 0b0011, 0b1111, 0b00110011]
    filtered = list(weight_mod_four_zero(cands))
    assert 0 not in filtered
    assert 0b0011 not in filtered  # weight 2
    assert 0b1111 in filtered      # weight 4
    assert 0b00110011 in filtered  # weight 4


# ---------------------------------------------------------- orbit minimum


def test_orbit_minima_under_trivial_aut():
    """If Aut(C) = {id} (passed as empty generator list), every rep passes."""
    C = Code(4, (0b0011,))
    cands = [0b0011, 0b1100, 0b1111]
    kept = list(aut_orbit_minima(cands, [], C))
    assert kept == cands


def test_orbit_minima_under_S_n():
    """For C = zero code on n=8, Aut(C) = S_8: weight classes are orbits."""
    C = Code.zero(8)
    info = canon_info(C)
    # All weight-4 vectors are in one orbit
    weight4 = [v for v in range(256) if wt(v) == 4]
    kept = list(aut_orbit_minima(weight4, info.aut_generators, C))
    assert len(kept) == 1
    # The orbit min is the lex-min weight-4 vector: 0b00001111 = 15
    assert kept[0] == 0b00001111


# ------------------------------------------------------ end-to-end pipeline


def test_doubly_even_candidates_zero_code_n8():
    """For zero code on n=8 the pipeline yields one rep per allowed weight (4, 8)."""
    C = Code.zero(8)
    info = canon_info(C)
    cands = doubly_even_candidates(C, info.aut_generators)
    # Expected: lex-min weight-4 vector and the all-ones vector
    assert sorted(cands) == [0b00001111, 0b11111111]


def test_doubly_even_candidates_weight4_singleton():
    """For C = ⟨v⟩ with wt(v)=4 on n=8, the candidates should produce
    one extension class per orbit."""
    C = Code(8, (0b00001111,))
    info = canon_info(C)
    cands = doubly_even_candidates(C, info.aut_generators)
    # Sanity: each yields ⟨C, v⟩ doubly even and not in C
    for v in cands:
        assert wt(v) % 4 == 0
        assert v != 0
        assert v not in C
        assert C.is_orthogonal_to(v)
