"""Rust ↔ Python cross-check for the Witt phase-(b) orbit-min path.

The kernel still exposes ``debug.aut_orbit_minima_q_witt`` even though
the active dispatch never reaches phase (b). This test pins the Rust
implementation against the dormant Python oracle in
:mod:`doubly_even.enumerate.experimental.quotient_witt` so a future
recovery attempt does not start from a broken floor.
"""

from __future__ import annotations

import pytest

doubly_even_kernel = pytest.importorskip("doubly_even_kernel")

from doubly_even.enumerate.augment import enumerate_doubly_even
from doubly_even.enumerate.experimental.quotient_witt import aut_orbit_minima_Q_witt
from doubly_even.enumerate.quotient import (
    Q_basis,
    aut_image_on_Q,
    singular_reps_Q,
)


def _iter_parents(N: int):
    for ec in enumerate_doubly_even(N):
        yield ec.code, ec.info.aut_generators


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
