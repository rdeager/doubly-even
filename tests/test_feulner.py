"""Differential tests for the Feulner-style canonicaliser.

The new implementation in :mod:`doubly_even.canon.feulner` is verified
against :func:`doubly_even.canon.nauty.canon_info` (whose Rust kernel
path runs nauty under the hood). For every code emitted by the small-N
enumerator we assert:

* the exact `|Aut(C)|` agrees with nauty,
* the column-orbit set-partition agrees (orbit labels are
  path-dependent; compare as sets),
* the returned generators generate a group of the claimed order,
* every returned generator actually fixes `C` as a code,
* `canonical_form(C)` is well-defined: random column permutations of a
  code give the same canonical RREF.
"""

from __future__ import annotations

import math
import random

import pytest

from doubly_even.canon import nauty as nauty_mod
from doubly_even.canon.feulner import canon_info_feulner, canon_info_feulner_native
from doubly_even.canon.nauty import canon_info, canon_info_cache_clear
from doubly_even.canon.permutations import group_order
from doubly_even.enumerate.augment import enumerate_doubly_even, enumerate_doubly_even_at
from doubly_even.spec.codes import Code, _compute_rref
from doubly_even.spec.mass import gaborit_sigma
from doubly_even.spec.vectors import apply_permutation


SMALL_N = [4, 6, 8, 10, 12]


def _orbit_set_partition(orbits: tuple[int, ...]) -> frozenset[frozenset[int]]:
    groups: dict[int, set[int]] = {}
    for col, oid in enumerate(orbits):
        groups.setdefault(oid, set()).add(col)
    return frozenset(frozenset(g) for g in groups.values())


def _permute_code(C: Code, sigma: tuple[int, ...]) -> Code:
    new_basis = tuple(apply_permutation(b, list(sigma)) for b in C.basis)
    return Code(n=C.n, basis=new_basis)


# ----------------------------------------------------------------- aut order


@pytest.mark.parametrize("N", SMALL_N)
def test_aut_order_matches_native_on_enumeration(N: int) -> None:
    for ec in enumerate_doubly_even(N):
        ref = canon_info(ec.code)
        new = canon_info_feulner(ec.code)
        assert new.aut_order == ref.aut_order, (
            f"N={N} k={ec.code.rank} basis={list(ec.code.basis)}: "
            f"feulner={new.aut_order}, native={ref.aut_order}"
        )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
def test_zero_code_aut(n: int) -> None:
    info = canon_info_feulner(Code.zero(n))
    assert info.aut_order == math.factorial(n)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
def test_whole_space_aut(n: int) -> None:
    info = canon_info_feulner(Code.whole(n))
    assert info.aut_order == math.factorial(n)


def test_extended_hamming_8_4() -> None:
    rows = (0b00001111, 0b00110011, 0b01010101, 0b11111111)
    info = canon_info_feulner(Code(8, rows))
    assert info.aut_order == 1344


def test_single_weight4_vector_in_F2_8() -> None:
    info = canon_info_feulner(Code(8, (0b00001111,)))
    assert info.aut_order == 24 * 24


# ----------------------------------------------------------- column orbits


@pytest.mark.parametrize("N", SMALL_N)
def test_column_orbits_set_partition_matches(N: int) -> None:
    for ec in enumerate_doubly_even(N):
        ref = canon_info(ec.code)
        new = canon_info_feulner(ec.code)
        assert _orbit_set_partition(new.column_orbits) == _orbit_set_partition(
            ref.column_orbits
        ), (
            f"N={N} k={ec.code.rank} basis={list(ec.code.basis)}: "
            f"feulner orbits={new.column_orbits}, native={ref.column_orbits}"
        )


# -------------------------------------------------- generator self-consistency


@pytest.mark.parametrize("N", SMALL_N)
def test_generators_generate_group_of_claimed_order(N: int) -> None:
    for ec in enumerate_doubly_even(N):
        info = canon_info_feulner(ec.code)
        derived = group_order(info.aut_generators, N) if info.aut_generators else 1
        assert derived == info.aut_order, (
            f"N={N} k={ec.code.rank} basis={list(ec.code.basis)}: "
            f"gens give order {derived}, info.aut_order={info.aut_order}"
        )


@pytest.mark.parametrize("N", SMALL_N)
def test_generators_preserve_code(N: int) -> None:
    for ec in enumerate_doubly_even(N):
        C = ec.code
        rref_C, _ = C.rref_basis()
        info = canon_info_feulner(C)
        for g in info.aut_generators:
            permuted_basis = tuple(
                apply_permutation(b, list(g)) for b in C.basis
            )
            rref_g, _ = _compute_rref(C.n, permuted_basis)
            assert rref_g == rref_C, (
                f"N={N} k={C.rank}: generator {g} does not preserve code"
            )


# ----------------------------------------------------- Rust kernel parity


@pytest.mark.parametrize("N", SMALL_N)
def test_rust_feulner_matches_python_on_aut_order(N: int) -> None:
    """The Rust port must agree with the Python staging on `|Aut(C)|`."""
    for ec in enumerate_doubly_even(N):
        py = canon_info_feulner(ec.code)
        rust = canon_info_feulner_native(ec.code)
        assert rust.aut_order == py.aut_order, (
            f"N={N} k={ec.code.rank} basis={list(ec.code.basis)}: "
            f"rust={rust.aut_order}, python={py.aut_order}"
        )


@pytest.mark.parametrize("N", SMALL_N)
def test_rust_feulner_matches_python_on_column_orbits(N: int) -> None:
    for ec in enumerate_doubly_even(N):
        py = canon_info_feulner(ec.code)
        rust = canon_info_feulner_native(ec.code)
        assert _orbit_set_partition(rust.column_orbits) == _orbit_set_partition(
            py.column_orbits
        )


@pytest.mark.parametrize("N", SMALL_N)
def test_rust_feulner_generators_preserve_code(N: int) -> None:
    for ec in enumerate_doubly_even(N):
        C = ec.code
        rref_C, _ = C.rref_basis()
        info = canon_info_feulner_native(C)
        for g in info.aut_generators:
            permuted = tuple(apply_permutation(b, list(g)) for b in C.basis)
            rref_g, _ = _compute_rref(C.n, permuted)
            assert rref_g == rref_C, (
                f"Rust generator {g} does not preserve N={N} k={C.rank} code"
            )


@pytest.mark.parametrize("N", SMALL_N)
def test_rust_feulner_matches_native_nauty_on_aut_order(N: int) -> None:
    """Most important: the Rust Feulner port must agree with nauty."""
    for ec in enumerate_doubly_even(N):
        ref = canon_info(ec.code)
        rust = canon_info_feulner_native(ec.code)
        assert rust.aut_order == ref.aut_order, (
            f"N={N} k={ec.code.rank} basis={list(ec.code.basis)}: "
            f"rust feulner={rust.aut_order}, nauty={ref.aut_order}"
        )


# ----------------------------------------------------- canonical-form invariance


# --------------------------------------------------- end-to-end McKay test


# A representative spread across the small enumerator. Each cell is one
# DFGHILM Appendix B Table 3 entry; running the enumerator with feulner
# plugged in must reproduce these counts exactly.
_TABLE_3_SMOKE: dict[tuple[int, int], int] = {
    (8, 1): 2, (8, 2): 2, (8, 3): 2, (8, 4): 1,
    (10, 2): 3, (10, 3): 3, (10, 4): 2,
    (12, 3): 7, (12, 4): 7, (12, 5): 2,
}


@pytest.mark.parametrize("Nk,expected", sorted(_TABLE_3_SMOKE.items()))
def test_enumerator_with_feulner_backend_matches_dfghilm(
    Nk: tuple[int, int], expected: int, monkeypatch
) -> None:
    """Monkeypatch the canon backend to feulner, then re-run the McKay
    enumerator on a representative spread of `(N, k)` cells. Counts must
    still match DFGHILM Appendix B Table 3."""
    N, k = Nk
    monkeypatch.setattr(nauty_mod, "canon_info", canon_info_feulner)
    canon_info_cache_clear()
    try:
        classes = sum(1 for _ in enumerate_doubly_even_at(N, k))
        assert classes == expected, (
            f"feulner+McKay at N={N},k={k}: got {classes}, expected {expected}"
        )
    finally:
        canon_info_cache_clear()


@pytest.mark.parametrize("N", [8, 10, 12])
def test_enumerator_with_feulner_mass_check(N: int, monkeypatch) -> None:
    """With feulner plugged in, the per-rank mass must still equal
    Gaborit's closed form `σ(N, k)`."""
    monkeypatch.setattr(nauty_mod, "canon_info", canon_info_feulner)
    canon_info_cache_clear()
    try:
        by_k: dict[int, int] = {}
        factN = math.factorial(N)
        for ec in enumerate_doubly_even(N):
            k = ec.code.rank
            by_k[k] = by_k.get(k, 0) + factN // ec.aut_order
        for k, mass in by_k.items():
            expected = gaborit_sigma(N, k)
            assert mass == expected, (
                f"feulner+McKay mass at N={N},k={k}: got {mass}, expected {expected}"
            )
    finally:
        canon_info_cache_clear()


@pytest.mark.parametrize("N", SMALL_N)
def test_canonical_form_is_permutation_invariant(N: int) -> None:
    """Two equivalent codes should canonicalise to the same RREF."""
    rng = random.Random(N * 31 + 7)
    for ec in enumerate_doubly_even(N):
        C = ec.code
        if C.rank == 0 or C.rank == C.n:
            # Trivial cases — every permutation gives the same RREF anyway.
            continue
        info_C = canon_info_feulner(C)
        canon_basis_C = tuple(
            apply_permutation(b, list(info_C.canonical_column_order))
            for b in C.basis
        )
        rref_C, _ = _compute_rref(C.n, canon_basis_C)

        for trial in range(3):
            sigma = list(range(N))
            rng.shuffle(sigma)
            C_perm = _permute_code(C, tuple(sigma))
            info_p = canon_info_feulner(C_perm)
            canon_basis_p = tuple(
                apply_permutation(b, list(info_p.canonical_column_order))
                for b in C_perm.basis
            )
            rref_p, _ = _compute_rref(C.n, canon_basis_p)
            assert rref_C == rref_p, (
                f"N={N} k={C.rank} trial={trial}: canonical RREF differs "
                f"between C and a random column permutation. "
                f"sigma={sigma} rref_C={rref_C} rref_perm={rref_p}"
            )
