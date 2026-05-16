"""Doubly even predicates and the augmentation criterion.

A binary linear code is **doubly even** if every codeword has Hamming weight
divisible by 4.

The cheap check uses Corollary B.1 of DFGHILM Appendix B.3: if every generator
of ``C`` has weight ``≡ 0 (mod 4)`` and every pair of generators is orthogonal,
then ``C`` is doubly even. So the test is ``O(k^2)`` in the basis size, not
``O(2^k)`` in the codeword count.

The augmentation criterion ("when is ``⟨C, v⟩`` doubly even?") is the engine
of our search. It says: ``v ∈ C⊥`` and ``wt(v) ≡ 0 (mod 4)``.
"""

from __future__ import annotations

from .codes import Code
from .vectors import BinVec, dot, wt


def is_doubly_even(C: Code) -> bool:
    """Test whether every codeword of ``C`` has weight ``≡ 0 (mod 4)``.

    Uses Corollary B.1: a code is doubly even iff some (equivalently, every)
    generating set ``v_1, …, v_k`` satisfies

    * ``wt(v_i) ≡ 0 (mod 4)`` for all ``i``, and
    * ``⟨v_i, v_j⟩ = 0`` for all ``i ≠ j``.

    The check is ``O(k^2)`` and works on the stored basis directly (no need to
    reduce or enumerate codewords).
    """
    basis = C.basis
    for b in basis:
        if wt(b) % 4 != 0:
            return False
    for i, bi in enumerate(basis):
        for bj in basis[i + 1 :]:
            if dot(bi, bj) != 0:
                return False
    return True


def doubly_even_extension(C: Code, v: BinVec) -> bool:
    """Return True iff ``⟨C, v⟩`` is doubly even, assuming ``C`` already is.

    This is the augmentation criterion (DFGHILM Appendix B):

    * ``v ∈ C⊥`` — that is, ``v`` is orthogonal to every codeword of ``C``,
      equivalently orthogonal to every basis vector;
    * ``wt(v) ≡ 0 (mod 4)``.

    The caller's responsibility to ensure ``C`` is already doubly even; we do
    not re-check.
    """
    if wt(v) % 4 != 0:
        return False
    return C.is_orthogonal_to(v)


def all_codewords_doubly_even(C: Code) -> bool:
    """Brute-force version of :func:`is_doubly_even`.

    Enumerates every codeword and checks ``wt mod 4 == 0`` directly.
    ``O(2^k * k)`` and only suitable for tests. Used to validate
    :func:`is_doubly_even` against the literal definition.
    """
    return all(wt(c) % 4 == 0 for c in C.codewords())
