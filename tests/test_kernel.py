"""Cross-check the Rust kernel against the pure-Python reference.

These tests are skipped entirely if ``doubly_even_kernel`` is not
importable — that is the same import-time gate the production
:mod:`doubly_even.enumerate.filters` uses, so a kernel-less checkout
keeps the rest of the suite green.

What we check:

1. **Stage-level exact-equal** (``debug`` submodule): each pipeline
   stage's Rust output must match the Python reference bit-for-bit on
   every parent of every ``[N, k]`` class for ``N <= 12``.
2. **Top-level exact-equal**: ``kernel.doubly_even_candidates_q``
   matches ``quotient.doubly_even_candidates_Q`` for every parent in
   the full augmentation tree up to ``N = 12``.
3. **End-to-end DFGHILM**: the kernel-active path emits the same class
   counts as DFGHILM Appendix B Table 3 through ``N = 12`` (default
   suite). The larger ``N`` cells are already covered by
   ``test_augment.py``, which now runs through the kernel by default.
"""

from __future__ import annotations

import pytest

doubly_even_kernel = pytest.importorskip("doubly_even_kernel")

from doubly_even.canon.nauty import cached_canon_info
from doubly_even.enumerate.augment import enumerate_doubly_even
from doubly_even.enumerate.quotient import (
    Q_basis,
    aut_image_on_Q,
    aut_orbit_minima_Q,
    aut_orbit_minima_Q_witt,
    doubly_even_candidates_Q,
    singular_reps_Q,
    _sigma_Q_table,
)
from doubly_even.spec.codes import Code


def _iter_parents(N: int):
    """Yield every code that appears as a parent in ``enumerate_doubly_even(N)``.

    Each yielded ``(code, aut_generators)`` is exactly the pair the
    augmentation loop hands to ``doubly_even_candidates``.
    """
    for ec in enumerate_doubly_even(N):
        yield ec.code, ec.info.aut_generators


# --------------------------------------------------------- stage cross-checks


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_kernel_q_basis_matches_python(N: int):
    for C, _aut in _iter_parents(N):
        rref, pivots = C.rref_basis()
        dual_basis = C.dual().basis
        py_basis, py_pivots = Q_basis(C)
        rust_basis, rust_pivots = doubly_even_kernel.debug.q_basis(
            list(rref), list(pivots), list(dual_basis), C.n
        )
        assert tuple(rust_basis) == tuple(py_basis), (
            f"N={N}, k={C.rank}: V_basis mismatch"
        )
        assert tuple(rust_pivots) == tuple(py_pivots), (
            f"N={N}, k={C.rank}: pivots_V mismatch"
        )


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_kernel_aut_image_on_q_matches_python(N: int):
    for C, aut_gens in _iter_parents(N):
        rref, pivots = C.rref_basis()
        v_basis, pivots_v = Q_basis(C)
        py_images = aut_image_on_Q(aut_gens, C, v_basis, pivots_v)
        rust_images = doubly_even_kernel.debug.aut_image_on_q(
            [list(g) for g in aut_gens],
            list(rref),
            list(pivots),
            list(v_basis),
            list(pivots_v),
        )
        assert [tuple(m) for m in rust_images] == [tuple(m) for m in py_images], (
            f"N={N}, k={C.rank}: σ_Q image mismatch"
        )


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_kernel_singular_reps_matches_python(N: int):
    for C, _aut in _iter_parents(N):
        v_basis, _piv = Q_basis(C)
        py_reps = singular_reps_Q(v_basis)
        rust_reps = doubly_even_kernel.debug.singular_reps_q(list(v_basis))
        # singular_reps_Q yields in Gray-code visit order; sort both
        # for the comparison (production code sorts the lift output).
        assert sorted(rust_reps) == sorted(py_reps), (
            f"N={N}, k={C.rank}: singular set mismatch"
        )


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_kernel_sigma_q_table_matches_python(N: int):
    for C, aut_gens in _iter_parents(N):
        v_basis, pivots_v = Q_basis(C)
        sigma_qs = aut_image_on_Q(aut_gens, C, v_basis, pivots_v)
        L = len(v_basis)
        for sigma in sigma_qs:
            py_table = _sigma_Q_table(sigma)
            rust_table = doubly_even_kernel.debug.sigma_q_table(list(sigma), L)
            assert list(rust_table) == py_table, (
                f"N={N}, k={C.rank}: σ_Q table mismatch at L={L}"
            )


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_kernel_orbit_min_table_matches_python(N: int):
    for C, aut_gens in _iter_parents(N):
        v_basis, pivots_v = Q_basis(C)
        sigma_qs = aut_image_on_Q(aut_gens, C, v_basis, pivots_v)
        L = len(v_basis)
        reps_q = singular_reps_Q(v_basis)
        py_mins = aut_orbit_minima_Q(reps_q, sigma_qs)
        rust_mins = doubly_even_kernel.debug.aut_orbit_minima_q_table(
            list(reps_q), [list(m) for m in sigma_qs], L
        )
        assert sorted(rust_mins) == sorted(py_mins), (
            f"N={N}, k={C.rank}: orbit-min (table) mismatch"
        )


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_kernel_orbit_min_witt_matches_python(N: int):
    for C, aut_gens in _iter_parents(N):
        v_basis, pivots_v = Q_basis(C)
        sigma_qs = aut_image_on_Q(aut_gens, C, v_basis, pivots_v)
        L = len(v_basis)
        reps_q = singular_reps_Q(v_basis)
        py_mins = aut_orbit_minima_Q_witt(sigma_qs, reps_q, L)
        rust_mins = doubly_even_kernel.debug.aut_orbit_minima_q_witt(
            list(reps_q), [list(m) for m in sigma_qs], L
        )
        assert sorted(rust_mins) == sorted(py_mins), (
            f"N={N}, k={C.rank}: orbit-min (witt) mismatch"
        )


# -------------------------------------------------- top-level cross-check


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_kernel_doubly_even_candidates_q_matches_python(N: int):
    for C, aut_gens in _iter_parents(N):
        rref, pivots = C.rref_basis()
        dual_basis = C.dual().basis
        rust_out = doubly_even_kernel.doubly_even_candidates_q(
            C.n,
            list(rref),
            list(pivots),
            list(dual_basis),
            [list(g) for g in aut_gens],
        )
        py_out = doubly_even_candidates_Q(C, aut_gens)
        assert list(rust_out) == list(py_out), (
            f"N={N}, k={C.rank}: doubly_even_candidates_q output mismatch"
        )


# ---------------------------------------------- end-to-end mass formula


def test_kernel_path_preserves_mass_formula_at_n12():
    """With the kernel active (default in this checkout), the augmentation
    recursion must still emit the exact Gaborit class counts."""
    from doubly_even.spec.mass import gaborit_sigma
    import math

    N = 12
    factorial_N = math.factorial(N)
    mass_at_k: dict[int, int] = {}
    for ec in enumerate_doubly_even(N):
        k = ec.code.rank
        mass_at_k[k] = mass_at_k.get(k, 0) + factorial_N // ec.aut_order
    for k, mass in mass_at_k.items():
        assert mass == gaborit_sigma(N, k), (
            f"mass at N=12 k={k}: kernel-active path drifted from gaborit_sigma"
        )


def test_kernel_n4_minimal_smoke():
    """Smallest sanity check: N=4 zero code yields one weight-4 augmentation."""
    C = Code.zero(4)
    aut_gens = cached_canon_info(C).aut_generators
    rust_out = doubly_even_kernel.doubly_even_candidates_q(
        4, list(C.rref_basis()[0]), list(C.rref_basis()[1]),
        list(C.dual().basis), [list(g) for g in aut_gens],
    )
    assert rust_out == [0b1111]
