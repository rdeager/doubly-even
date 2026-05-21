"""Binary vectors as Python ints; the Code dataclass; GF(2) RREF and dual basis.

A vector ``v ∈ F_2^n`` is an int with bit ``i`` = component ``i``. XOR is
addition, ``int.bit_count()`` is Hamming weight, ``(1 << i)`` is ``e_i``.
"""

from __future__ import annotations
from dataclasses import dataclass

BinVec = int


def apply_perm(v: BinVec, sigma: list[int]) -> BinVec:
    """Column permutation: bit ``i`` of ``v`` lands at bit ``sigma[i]``."""
    out = 0
    for i, j in enumerate(sigma):
        if (v >> i) & 1:
            out |= 1 << j
    return out


def rref_gf2(rows: list[BinVec], n: int) -> tuple[list[BinVec], list[int]]:
    """In-place GF(2) Gauss-Jordan; returns ``(reduced rows, pivot cols)``."""
    pivots: list[int] = []
    r = 0
    for c in range(n):
        pivot = -1
        for i in range(r, len(rows)):
            if (rows[i] >> c) & 1:
                pivot = i
                break
        if pivot == -1:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        for i in range(len(rows)):
            if i != r and (rows[i] >> c) & 1:
                rows[i] ^= rows[r]
        pivots.append(c)
        r += 1
    return rows[:r], pivots


def dual_basis_from_rref(rref, pivots, n):
    """Basis of ``C^⊥`` given ``C`` in RREF: one dual generator per free column."""
    pset = set(pivots)
    out: list[BinVec] = []
    for j in range(n):
        if j in pset:
            continue
        v = 1 << j
        for row, p in zip(rref, pivots):
            if (row >> j) & 1:
                v |= 1 << p
        out.append(v)
    return out


@dataclass(frozen=True)
class Code:
    n: int
    basis: tuple[BinVec, ...] = ()

    @classmethod
    def zero(cls, n: int) -> "Code":
        return cls(n=n, basis=())

    def rref(self):
        rows, pivots = rref_gf2(list(self.basis), self.n)
        return tuple(rows), tuple(pivots)

    @property
    def rank(self) -> int:
        return len(self.rref()[0])

    def dual_basis(self) -> tuple[BinVec, ...]:
        rref, pivots = self.rref()
        return tuple(dual_basis_from_rref(list(rref), list(pivots), self.n))

    def codewords(self) -> list[BinVec]:
        """All ``2^k`` codewords via Gray-code walk (one XOR per step)."""
        rows, _ = self.rref()
        k = len(rows)
        out = [0] * (1 << k)
        for i in range(1, 1 << k):
            flip = (i & -i).bit_length() - 1
            out[i] = out[i ^ (1 << flip)] ^ rows[flip]
        return out

    def codewords_by_weight(self) -> dict[int, list[BinVec]]:
        """All ``2^k`` codewords bucketed by weight in one Gray-code walk."""
        rows, _ = self.rref()
        out: dict[int, list[BinVec]] = {0: [0]}
        c = 0
        for i in range(1, 1 << len(rows)):
            flip = (i & -i).bit_length() - 1
            c ^= rows[flip]
            out.setdefault(c.bit_count(), []).append(c)
        return out

    def __contains__(self, v: BinVec) -> bool:
        rref, pivots = self.rref()
        residue = v
        for row, c in zip(rref, pivots):
            if (residue >> c) & 1:
                residue ^= row
        return residue == 0

    def is_orthogonal_to(self, v: BinVec) -> bool:
        return all((b & v).bit_count() & 1 == 0 for b in self.basis)

    def extend(self, v: BinVec) -> "Code":
        return Code(self.n, self.basis + (v,))
