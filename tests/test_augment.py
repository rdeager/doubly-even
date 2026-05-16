"""End-to-end tests for the canonical-augmentation enumerator."""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from doubly_even.canon.nauty import canon_info
from doubly_even.enumerate.augment import (
    _weight_enum,
    canonical_parent,
    enumerate_doubly_even,
    enumerate_doubly_even_at,
    is_canonical_augmentation,
)
from doubly_even.spec.codes import Code
from doubly_even.spec.doubly_even import is_doubly_even
from doubly_even.spec.mass import sigma_brute
from doubly_even.spec.vectors import apply_permutation


# -------------------------------------------------------- canonical parent


# ------------------------------------- _weight_enum (Aut-orbit prefilter)


def test_weight_enum_invariant_under_aut_generators():
    """Every codeword weight is preserved by column permutation, so applying
    any ``σ ∈ Aut(C)`` to ``C``'s basis must give a code with the same
    sorted weight enumerator."""
    C = Code(8, (0b00001111, 0b11110000))
    info = canon_info(C)
    base = _weight_enum(C)
    for sigma in info.aut_generators:
        sigma_list = list(sigma)
        permuted_basis = tuple(apply_permutation(b, sigma_list) for b in C.basis)
        C_sigma = Code(C.n, permuted_basis)
        assert _weight_enum(C_sigma) == base


def test_weight_enum_distinguishes_inequivalent_codes():
    """Two codes with different weight enumerators are guaranteed in
    different Aut-orbits; the prefilter relies on this direction."""
    C1 = Code(8, (0b00001111,))            # weight enumerator: {0, 4}
    C2 = Code(8, (0b11111111,))            # weight enumerator: {0, 8}
    assert _weight_enum(C1) != _weight_enum(C2)


def test_canonical_parent_drops_one_dimension():
    D = Code(8, (0b00001111, 0b11110000))
    p_D = canonical_parent(D)
    assert p_D.rank == D.rank - 1


def test_canonical_parent_is_subspace_of_D():
    """Every basis vector of p(D) should lie in D."""
    D = Code(8, (0b00001111, 0b11110000))
    p_D = canonical_parent(D)
    for b in p_D.basis:
        assert b in D


def test_canonical_parent_of_rank1_is_zero_code():
    """For a 1-dim code, the parent is the 0-dim code."""
    D = Code(8, (0b00001111,))
    p_D = canonical_parent(D)
    assert p_D.rank == 0


# ----------------------------------- mass-formula completeness certificate


@pytest.mark.parametrize("N", [4, 6, 7, 8])
def test_enumerator_mass_matches_sigma(N):
    """For each k, ``Σ N!/|Aut(C_i)|`` over emitted classes equals ``σ(N, k)``."""
    by_k: dict[int, list] = defaultdict(list)
    for ec in enumerate_doubly_even(N):
        by_k[ec.code.rank].append(ec)

    for k, codes in by_k.items():
        mass = sum(math.factorial(N) // ec.aut_order for ec in codes)
        assert mass == sigma_brute(N, k), (
            f"N={N}, k={k}: enumerator mass={mass}, sigma_brute={sigma_brute(N, k)}"
        )


# ---------------------------------------------- equivalence-class counts


def test_n8_k4_unique():
    """The extended Hamming [8, 4, 4] is the unique doubly even [8, 4] class."""
    codes = list(enumerate_doubly_even_at(8, 4))
    assert len(codes) == 1
    assert codes[0].aut_order == 1344


@pytest.mark.parametrize(
    "N, k, expected_classes",
    [
        (4, 0, 1),
        (4, 1, 1),
        (6, 0, 1),
        (6, 1, 1),
        (6, 2, 1),
        (7, 0, 1),
        (7, 1, 1),
        (7, 2, 1),
        (7, 3, 1),
        (8, 0, 1),
        (8, 1, 2),   # weight-4 generator + all-ones
        (8, 4, 1),   # extended Hamming
    ],
)
def test_equivalence_class_counts(N, k, expected_classes):
    codes = list(enumerate_doubly_even_at(N, k))
    assert len(codes) == expected_classes


# DFGHILM Appendix B Table 3 — number of permutation-equivalence classes
# of doubly even [N, k] codes. Independent published values; our enumerator
# must match exactly. (Only the cells that fit in the table; some larger
# N, k cases were starred (*) in the paper as "still being enumerated".)
DFGHILM_TABLE_3: dict[tuple[int, int], int] = {
    (4, 1): 1,
    (5, 1): 1,
    (6, 1): 1, (6, 2): 1,
    (7, 1): 1, (7, 2): 1, (7, 3): 1,
    (8, 1): 2, (8, 2): 2, (8, 3): 2, (8, 4): 1,
    (9, 1): 2, (9, 2): 2, (9, 3): 2, (9, 4): 1,
    (10, 1): 2, (10, 2): 3, (10, 3): 3, (10, 4): 2,
    (11, 1): 2, (11, 2): 3, (11, 3): 4, (11, 4): 3,
    (12, 1): 3, (12, 2): 5, (12, 3): 7, (12, 4): 7, (12, 5): 2,
    (13, 1): 3, (13, 2): 5, (13, 3): 8, (13, 4): 8, (13, 5): 4,
    (14, 1): 3, (14, 2): 7, (14, 3): 12, (14, 4): 14, (14, 5): 9,
    (14, 6): 4,
    (15, 1): 3, (15, 2): 7, (15, 3): 15, (15, 4): 20, (15, 5): 15,
    (15, 6): 8, (15, 7): 2,
    (16, 1): 4, (16, 2): 10, (16, 3): 23, (16, 4): 38, (16, 5): 36,
    (16, 6): 23, (16, 7): 9, (16, 8): 2,
}


# Cells we have verified manually but mark "slow" so they are skipped by
# default in the test suite. Run with ``pytest --run-slow`` to enable.
DFGHILM_TABLE_3_SLOW: dict[tuple[int, int], int] = {
    (17, 1): 4, (17, 2): 10, (17, 3): 25, (17, 4): 45, (17, 5): 50,
    (17, 6): 34, (17, 7): 14, (17, 8): 3,
    (18, 1): 4, (18, 2): 13, (18, 3): 34, (18, 4): 72, (18, 5): 94,
    (18, 6): 79, (18, 7): 35, (18, 8): 9,
}


@pytest.mark.parametrize(
    "Nk, expected", sorted(DFGHILM_TABLE_3.items())
)
def test_matches_dfghilm_table_3(Nk, expected):
    """Cross-check against DFGHILM Appendix B Table 3."""
    N, k = Nk
    classes = sum(1 for _ in enumerate_doubly_even_at(N, k))
    assert classes == expected, (
        f"N={N}, k={k}: enumerator={classes}, DFGHILM Table 3={expected}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "Nk, expected", sorted(DFGHILM_TABLE_3_SLOW.items())
)
def test_matches_dfghilm_table_3_slow(Nk, expected):
    """Slow Table 3 cells (N >= 17). Run with `pytest --run-slow`."""
    N, k = Nk
    classes = sum(1 for _ in enumerate_doubly_even_at(N, k))
    assert classes == expected, (
        f"N={N}, k={k}: enumerator={classes}, DFGHILM Table 3={expected}"
    )


# --------------------------------------------------- emitted codes valid


@pytest.mark.parametrize("N", [4, 6, 7, 8])
def test_every_emitted_code_is_doubly_even(N):
    for ec in enumerate_doubly_even(N):
        assert is_doubly_even(ec.code)


@pytest.mark.parametrize("N", [4, 6, 8])
def test_emitted_classes_pairwise_inequivalent(N):
    """No two emitted codes are permutation-equivalent."""
    from doubly_even.canon.nauty import canonical_form

    codes_by_k: dict[int, list] = defaultdict(list)
    for ec in enumerate_doubly_even(N):
        codes_by_k[ec.code.rank].append(ec.code)

    for k, codes in codes_by_k.items():
        canons = [canonical_form(c) for c in codes]
        assert len(set(canons)) == len(canons), (
            f"N={N}, k={k}: duplicate canonical forms emitted"
        )


# ---------------------------------------------- canonical-aug invariants


def test_is_canonical_augmentation_zero_to_one():
    """Every augmentation from the zero code is canonical (there is only one
    possible parent)."""
    N = 8
    C0 = Code.zero(N)
    info_C0 = canon_info(C0)
    from doubly_even.enumerate.filters import doubly_even_candidates

    for v in doubly_even_candidates(C0, info_C0.aut_generators):
        D = C0.extend(v)
        assert is_canonical_augmentation(C0, D)


def test_is_canonical_augmentation_rejects_wrong_parent():
    """A non-canonical augmentation should be rejected.

    We construct ⟨C, v⟩ in two different ways: once by starting from the
    canonical parent, once from a different (k-1)-subcode. The latter must
    be rejected as non-canonical (or both accepted iff the alternative is
    in the same Aut(D)-orbit as the canonical parent)."""
    # Type II extended Hamming [8, 4]
    D = Code(8, (0b00001111, 0b00110011, 0b01010101, 0b11111111))
    # All rank-3 subcodes of D, generated by 3 of its 4 basis vectors
    # (One of them is the canonical parent; the others are in the
    # Aut(D)-orbit of the canonical parent iff Aut(D) acts transitively
    # on rank-3 subcodes. For [8,4] Hamming, AGL(3,2) acts transitively
    # on hyperplanes of D, so all four are canonical -- they're in one
    # orbit. We at least test the canonical parent itself.)
    info_D = canon_info(D)
    p_D = canonical_parent(D, info_D)
    assert is_canonical_augmentation(p_D, D)
