"""Cross-checks for the experimental ``GL(L, F_2)`` Schreier–Sims module.

The matrix-group machinery mirrors :mod:`doubly_even.canon.permutations`;
tests here are structurally analogous to the permutation-group tests
(``test_canon.py``'s ``group_order`` cases) but on the linear action.

Quarantined under ``tests/experimental/`` because
:mod:`doubly_even.canon.experimental.schreier_sims_gl` is the
phase-(b) scaffolding that the active dispatch never reaches.

Validation tiers:

1. ``mat_inv`` round-trip: ``mat_mul(A, mat_inv(A)) == I`` for invertible
   ``A``; raises on singular input.
2. ``group_order_matrix`` agrees with closed-form ``|GL(L, F_2)|`` on the
   full transvection generating set, with ``n!`` on permutation-matrix
   embeddings of ``S_n``, and with a direct BFS enumeration of the
   generated group for small groups.
3. ``group_order_matrix`` on the image of ``Aut(C)`` in ``GL(L, F_2)``
   divides ``|Aut(C)|`` for every parent at small ``N`` (the quotient is
   the kernel size).
"""

from __future__ import annotations

import random
from math import factorial

import pytest

from doubly_even.canon._linalg_f2 import (
    Mat,
    mat_apply,
    mat_identity,
    mat_mul,
)
from doubly_even.canon.experimental.schreier_sims_gl import (
    group_order_matrix,
    mat_inv,
    orbit_and_transversal_matrix,
    stabilizer_chain,
)
from doubly_even.canon.nauty import canon_info
from doubly_even.enumerate.augment import enumerate_doubly_even
from doubly_even.enumerate.quotient import Q_basis, aut_image_on_Q


def _random_invertible(L: int, rng: random.Random) -> Mat:
    """Sample an invertible L×L matrix over F_2 by retry."""
    for _ in range(50):
        cand = tuple(rng.randrange(1 << L) for _ in range(L))
        try:
            mat_inv(cand)
            return cand
        except ValueError:
            continue
    raise RuntimeError(f"could not sample invertible matrix at L={L}")


def _gl_generators(L: int) -> list[Mat]:
    """All elementary transvections ``I + E_{ij}`` (``i ≠ j``).

    These generate ``GL(L, F_2)``: every invertible F_2-matrix is a
    product of elementary matrices, and over F_2 the only non-identity
    elementary type is the shear ``I + E_{ij}``.
    """
    gens: list[Mat] = []
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            cols = list(mat_identity(L))
            cols[j] |= 1 << i  # column j gains row-i bit
            gens.append(tuple(cols))
    return gens


def _perm_to_column_matrix(perm: tuple[int, ...]) -> Mat:
    """Embed a permutation in ``GL(n, F_2)`` as a column-form matrix.

    Column ``j`` of the matrix is the unit vector at row ``perm[j]``.
    """
    return tuple(1 << perm[j] for j in range(len(perm)))


def _bfs_group_order(gens: list[Mat], L: int) -> int:
    """Enumerate ``⟨gens⟩`` by BFS — exact, but exponential. Use only
    for small groups (orders ≲ 10^4) as a cross-check oracle."""
    identity = mat_identity(L)
    seen = {identity}
    queue = [identity]
    while queue:
        next_queue = []
        for g in queue:
            for s in gens:
                h = mat_mul(s, g)
                if h not in seen:
                    seen.add(h)
                    next_queue.append(h)
        queue = next_queue
    return len(seen)


# ------------------------------------------------------------------- mat_inv


@pytest.mark.parametrize("L", [1, 2, 3, 4, 6])
def test_mat_inv_roundtrip(L: int):
    rng = random.Random(0xDEAD + L)
    I = mat_identity(L)
    for _ in range(16):
        A = _random_invertible(L, rng)
        Ainv = mat_inv(A)
        assert mat_mul(A, Ainv) == I
        assert mat_mul(Ainv, A) == I


def test_mat_inv_singular_raises():
    # Zero column ⇒ singular.
    A = (0b01, 0b00)
    with pytest.raises(ValueError, match="singular matrix"):
        mat_inv(A)
    # Two identical columns ⇒ singular.
    B = (0b101, 0b010, 0b101)
    with pytest.raises(ValueError, match="singular matrix"):
        mat_inv(B)


# --------------------------------------------- orbit_and_transversal_matrix


def test_orbit_transversal_takes_base_to_point():
    """For every ``p`` in the orbit, ``mat_apply(transversal[p], base) == p``."""
    # GL(3, F_2) acts transitively on nonzero F_2^3 — orbit size = 7.
    L = 3
    gens = _gl_generators(L)
    base = 1
    orbit, transversal = orbit_and_transversal_matrix(gens, base, L)
    assert sorted(orbit) == list(range(1, 1 << L))
    for p in orbit:
        assert mat_apply(transversal[p], base) == p


def test_orbit_transversal_empty_generators():
    L = 4
    orbit, transversal = orbit_and_transversal_matrix([], 5, L)
    assert orbit == [5]
    assert transversal == {5: mat_identity(L)}


# ----------------------------------------------------- group_order_matrix


def test_group_order_empty():
    assert group_order_matrix([], 3) == 1


@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_group_order_matrix_full_GL(L: int):
    """``|GL(L, F_2)| = Π_{i=0}^{L-1}(2^L − 2^i)``."""
    expected = 1
    for i in range(L):
        expected *= (1 << L) - (1 << i)
    gens = _gl_generators(L)
    assert group_order_matrix(gens, L) == expected


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_group_order_matrix_symmetric_group(n: int):
    """``S_n`` as permutation matrices in ``GL(n, F_2)`` has order ``n!``."""
    # Generators: adjacent transpositions (i, i+1) for i in [0, n-1).
    gens: list[Mat] = []
    for i in range(n - 1):
        perm = list(range(n))
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        gens.append(_perm_to_column_matrix(tuple(perm)))
    assert group_order_matrix(gens, n) == factorial(n)


@pytest.mark.parametrize("L", [2, 3])
def test_group_order_matrix_vs_bfs(L: int):
    """For small ``L`` confirm Schreier–Sims matches direct BFS."""
    gens = _gl_generators(L)
    assert group_order_matrix(gens, L) == _bfs_group_order(gens, L)


def test_stabilizer_chain_orbit_stabiliser_identity():
    """For one parent at N=8 with non-trivial Aut, verify the textbook
    ``|orbit(u)| · |Stab_H(u)| == |H|`` identity using the chain.
    """
    from doubly_even.spec.codes import Code

    # [8, 1]: the unique non-zero doubly even code of rank 1.
    C = Code(8, (0b11111111,))
    info = canon_info(C)
    V_basis, pivots_V = Q_basis(C)
    L = len(V_basis)
    assert L == 8 - 2 * 1
    sigma_Qs = aut_image_on_Q(info.aut_generators, C, V_basis, pivots_V)
    H_order = group_order_matrix(sigma_Qs, L)
    assert H_order >= 1

    # Pick u = e_0 (always in F_2^L \ {0}).
    u = 1
    orbit_u, transversal_u = orbit_and_transversal_matrix(sigma_Qs, u, L)

    # Schreier generators for Stab_H(u).
    identity = mat_identity(L)
    stab_gens: set[Mat] = set()
    for p in orbit_u:
        t_p = transversal_u[p]
        for g in sigma_Qs:
            q = mat_apply(g, p)
            t_q_inv = mat_inv(transversal_u[q])
            sg = mat_mul(t_q_inv, mat_mul(g, t_p))
            if sg != identity:
                stab_gens.add(sg)
    stab_order = group_order_matrix(list(stab_gens), L)
    assert len(orbit_u) * stab_order == H_order, (
        f"orbit-stab identity broken: |orbit|={len(orbit_u)}, "
        f"|Stab|={stab_order}, |H|={H_order}"
    )


# ------------------------------ image of Aut(C) in GL(L, F_2) cross-check


CROSS_CHECK_NS = [4, 6, 8]


def test_image_order_divides_aut_order():
    """For every parent ``C`` at small ``N``, ``|image(Aut(C) → GL(L, F_2))|``
    divides ``|Aut(C)|``. The quotient is the kernel of the action."""
    for N in CROSS_CHECK_NS:
        for ec in enumerate_doubly_even(N):
            C = ec.code
            V_basis, pivots_V = Q_basis(C)
            L = len(V_basis)
            if L == 0:
                continue  # self-dual: trivial image, nothing to check
            sigma_Qs = aut_image_on_Q(ec.info.aut_generators, C, V_basis, pivots_V)
            H_order = group_order_matrix(sigma_Qs, L)
            assert ec.aut_order % H_order == 0, (
                f"|H|={H_order} does not divide |Aut(C)|={ec.aut_order} "
                f"at N={N}, k={C.rank}, C={list(C.basis)}"
            )
