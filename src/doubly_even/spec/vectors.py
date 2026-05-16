"""Binary vectors over GF(2), represented as Python ints.

A vector ``v ∈ F_2^N`` is stored as an integer whose bit ``i`` is ``v_i``.
With this representation:

* addition is XOR (``u ^ v``),
* Hamming weight is ``int.bit_count()``,
* the standard basis vector ``e_i`` is ``1 << i``.

We deliberately do not wrap the int in a class. The cost of attribute access
and method dispatch is significant at the sizes we care about, and the
operations we need are *exactly* Python's built-in int operations. Code that
reads ``u ^ v`` instead of ``add(u, v)`` is easier to check against the math.

The length ``N`` is carried separately (in :class:`doubly_even.spec.codes.Code`).
A vector value alone has no notion of length.
"""

from __future__ import annotations

from collections.abc import Iterable

BinVec = int
"""Type alias: a binary vector is an int with bit i = component i."""


def wt(v: BinVec) -> int:
    """Hamming weight: number of 1 bits."""
    return v.bit_count()


def dot(u: BinVec, v: BinVec) -> int:
    """Inner product over GF(2): ``Σ u_i v_i mod 2``."""
    return (u & v).bit_count() & 1


def add(u: BinVec, v: BinVec) -> BinVec:
    """Vector addition over GF(2). Provided for readability; equivalent to ``u ^ v``."""
    return u ^ v


def is_zero(v: BinVec) -> bool:
    return v == 0


def basis_vector(i: int) -> BinVec:
    """The standard basis vector ``e_i`` — bit ``i`` set, all others zero."""
    return 1 << i


def from_bits(bits: Iterable[int]) -> BinVec:
    """Pack an iterable of bits (LSB first) into a vector.

    >>> from_bits([1, 0, 1, 1])  # = 1 + 0 + 4 + 8
    13
    """
    v = 0
    for i, b in enumerate(bits):
        if b & 1:
            v |= 1 << i
    return v


def to_bits(v: BinVec, n: int) -> list[int]:
    """Unpack a vector to a length-``n`` list of bits (LSB first)."""
    return [(v >> i) & 1 for i in range(n)]


def support(v: BinVec) -> list[int]:
    """Indices ``i`` with ``v_i = 1``, in increasing order."""
    out: list[int] = []
    i = 0
    while v:
        if v & 1:
            out.append(i)
        v >>= 1
        i += 1
    return out


def apply_permutation(v: BinVec, sigma: list[int]) -> BinVec:
    """Apply a column permutation to a vector.

    ``sigma`` is a list of length ``n`` such that the new vector's bit
    ``sigma[i]`` equals the old vector's bit ``i``. (I.e. ``sigma`` maps "old
    index" → "new index"; equivalently, column ``i`` is sent to column
    ``sigma[i]``.)
    """
    out = 0
    for i, j in enumerate(sigma):
        if (v >> i) & 1:
            out |= 1 << j
    return out
