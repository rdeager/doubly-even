"""Tiny linear-algebra primitives for ``GL(L, F_2)`` actions.

These are the matrix-action helpers the spine needs — specifically,
``aut_image_on_Q`` and the σ_Q-orbit machinery in
:mod:`doubly_even.enumerate.quotient` consume :data:`Mat`,
:func:`mat_apply`, and :func:`mat_identity`. :func:`mat_mul` is the
natural companion and is kept here so the four primitives stay
together.

The richer Schreier–Sims-on-``GL(L, F_2)`` machinery
(``mat_inv``, ``orbit_and_transversal_matrix``, ``stabilizer_chain``,
``group_order_matrix``) lives in
:mod:`doubly_even.canon.experimental.schreier_sims_gl` — phase-(b)
scaffolding that the active dispatch never reaches.

Matrices are stored in **column form**: ``M`` is a tuple of ``L`` ints
where ``M[j]`` is the ``j``-th column of the ``L × L`` matrix over
``F_2``. Bit ``i`` of ``M[j]`` is the entry at row ``i``, column ``j``.
The action on a column vector ``v ∈ F_2^L`` (also an ``L``-bit int) is::

    mat_apply(M, v) = XOR over i of M[i] when bit i of v is set
                    = sum_{i: v_i = 1} M[:, i]

This matches the convention produced by
:func:`doubly_even.enumerate.quotient.aut_image_on_Q` — ``σ_Q[i]`` is
"``σ`` applied to the ``i``-th basis vector", i.e. the ``i``-th column.
"""

from __future__ import annotations


Mat = tuple[int, ...]


def mat_identity(L: int) -> Mat:
    """Identity in column form: column ``j`` is the unit vector ``e_j``."""
    return tuple(1 << j for j in range(L))


def mat_apply(M: Mat, v: int) -> int:
    """Apply ``M`` to column vector ``v ∈ F_2^L``.

    Walks only the *set* bits of ``v`` via the ``v & -v`` trick, so cost
    is ``popcount(v)`` iterations — strictly cheaper than a shift loop
    for sparse vectors and equal in the dense limit.
    """
    out = 0
    while v:
        lsb = v & -v
        out ^= M[lsb.bit_length() - 1]
        v ^= lsb
    return out


def mat_mul(A: Mat, B: Mat) -> Mat:
    """Matrix product over ``F_2``: ``(A·B)[j] = A · B[j]``.

    Composition order matches ``compose(p, q)`` in
    :mod:`doubly_even.canon.permutations`: ``mat_mul(A, B)`` applies
    ``B`` first, then ``A``.
    """
    return tuple(mat_apply(A, B[j]) for j in range(len(B)))
