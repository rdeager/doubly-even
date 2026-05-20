"""Tests for the spine ``GL(L, F_2)`` primitives in
:mod:`doubly_even.canon._linalg_f2`.

Covers :data:`Mat`, :func:`mat_identity`, :func:`mat_apply`,
:func:`mat_mul` — the four primitives consumed by
:mod:`doubly_even.enumerate.quotient`. The richer Schreier–Sims
machinery lives under ``canon/experimental/`` and is exercised by
``tests/experimental/test_schreier_sims_gl.py``.
"""

from __future__ import annotations

import random

import pytest

from doubly_even.canon._linalg_f2 import (
    Mat,
    mat_apply,
    mat_identity,
    mat_mul,
)


def _naive_apply(M: Mat, v: int) -> int:
    """Reference matrix-vector product, double loop over (row, col)."""
    L = len(M)
    out = 0
    for i in range(L):
        bit = 0
        for j in range(L):
            if (v >> j) & 1 and (M[j] >> i) & 1:
                bit ^= 1
        if bit:
            out |= 1 << i
    return out


def _random_matrix(L: int, rng: random.Random) -> Mat:
    """Sample a random L×L matrix over F_2 (not necessarily invertible)."""
    return tuple(rng.randrange(1 << L) for _ in range(L))


@pytest.mark.parametrize("L", [3, 5, 8])
def test_mat_apply_matches_naive(L: int):
    rng = random.Random(0xC0DE + L)
    M = _random_matrix(L, rng)
    for _ in range(64):
        v = rng.randrange(1 << L)
        assert mat_apply(M, v) == _naive_apply(M, v), (
            f"mat_apply mismatch at L={L}, M={M}, v={v:#x}"
        )


@pytest.mark.parametrize("L", [2, 4, 6])
def test_mat_mul_action_homomorphism(L: int):
    """``mat_apply(A·B, v) == mat_apply(A, mat_apply(B, v))``."""
    rng = random.Random(0xBEEF + L)
    for _ in range(16):
        A = _random_matrix(L, rng)
        B = _random_matrix(L, rng)
        AB = mat_mul(A, B)
        for _ in range(16):
            v = rng.randrange(1 << L)
            assert mat_apply(AB, v) == mat_apply(A, mat_apply(B, v))


@pytest.mark.parametrize("L", [2, 4, 6, 8])
def test_mat_mul_identity(L: int):
    rng = random.Random(0xFADE + L)
    I = mat_identity(L)
    for _ in range(8):
        A = _random_matrix(L, rng)
        assert mat_mul(I, A) == A
        assert mat_mul(A, I) == A


def test_mat_apply_zero_vector():
    """``mat_apply(M, 0) == 0`` for any matrix."""
    M = (0b110, 0b011, 0b101)
    assert mat_apply(M, 0) == 0


def test_mat_apply_identity_acts_as_identity():
    """``mat_apply(I, v) == v`` for any vector."""
    L = 5
    I = mat_identity(L)
    for v in range(1 << L):
        assert mat_apply(I, v) == v
