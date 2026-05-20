"""Direct sum of binary linear codes.

Given codes ``C1`` of length ``n1`` and ``C2`` of length ``n2``, the direct
sum ``C1 ⊕ C2`` is the code of length ``n1 + n2`` with codewords
``{(x1, x2) : x1 ∈ C1, x2 ∈ C2}``. The first ``n1`` bit positions hold
``x1``; positions ``n1 .. n1+n2-1`` hold ``x2``.

This is the simplest gluing operation: it preserves double-evenness
(each codeword's weight is the sum of two doubly-even weights),
self-orthogonality, and rank (``rank(C1 ⊕ C2) = rank(C1) + rank(C2)``).

Used by the mass-seeding pre-population in
:mod:`doubly_even.enumerate.seeds`.
"""

from __future__ import annotations

from .codes import Code


def direct_sum(C1: Code, C2: Code) -> Code:
    """Return the direct sum ``C1 ⊕ C2``.

    Bit layout: positions ``0 .. C1.n - 1`` carry the ``C1`` block;
    positions ``C1.n .. C1.n + C2.n - 1`` carry the ``C2`` block.

    The resulting basis is the concatenation of (i) ``C1``'s basis
    (unchanged) and (ii) ``C2``'s basis shifted left by ``C1.n``.
    The basis is not reduced; ``Code.rref_basis`` reduces on demand.
    """
    n1 = C1.n
    basis = tuple(C1.basis) + tuple(b << n1 for b in C2.basis)
    return Code(n=n1 + C2.n, basis=basis)
