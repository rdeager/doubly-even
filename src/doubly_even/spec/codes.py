"""Linear codes over GF(2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .vectors import BinVec, dot


@dataclass(frozen=True)
class Code:
    """A binary linear code, stored as a length and a basis.

    The basis is a tuple of int-encoded vectors (see :mod:`.vectors`).
    Basis vectors are not required to be reduced; :meth:`rref_basis`
    returns the row-reduced form. The empty basis is allowed (this is
    the zero-dimensional code).

    ``Code`` is intentionally immutable: derived data (RREF, pivots,
    dual) is recomputed on demand. Cache it externally if a hot loop
    needs it.
    """

    n: int
    basis: tuple[BinVec, ...] = ()

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"n must be non-negative, got {self.n}")
        mask = (1 << self.n) - 1 if self.n > 0 else 0
        for v in self.basis:
            if v < 0 or v & ~mask:
                raise ValueError(
                    f"basis vector {v:#x} has bits outside [0, {self.n})"
                )

    # ------------------------------------------------------------------ basics

    @classmethod
    def zero(cls, n: int) -> "Code":
        """The zero-dimensional code in ``F_2^n``."""
        return cls(n=n, basis=())

    @classmethod
    def whole(cls, n: int) -> "Code":
        """The whole space ``F_2^n``."""
        return cls(n=n, basis=tuple(1 << i for i in range(n)))

    # ----------------------------------------------------------------- rref +

    def rref_basis(self) -> tuple[tuple[BinVec, ...], tuple[int, ...]]:
        """Return ``(reduced_rows, pivot_columns)``.

        ``reduced_rows`` is the unique row-reduced echelon form basis of the
        rowspace; its length is the code's rank. ``pivot_columns`` lists the
        leading-1 column for each row in order.

        Process columns left to right (LSB to MSB). The reduction is over GF(2),
        so subtraction is XOR.
        """
        rows = list(self.basis)
        pivots: list[int] = []
        r = 0
        for c in range(self.n):
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
        return tuple(rows[:r]), tuple(pivots)

    @property
    def rank(self) -> int:
        return len(self.rref_basis()[0])

    # -------------------------------------------------------------- membership

    def __contains__(self, v: BinVec) -> bool:
        """Membership test: is ``v`` in the rowspace?"""
        if self.n == 0:
            return v == 0
        rows, pivots = self.rref_basis()
        residue = v
        for row, c in zip(rows, pivots):
            if (residue >> c) & 1:
                residue ^= row
        return residue == 0

    # -------------------------------------------------------------- enumeration

    def codewords(self) -> Iterator[BinVec]:
        """Yield every codeword exactly once. Use only when 2^rank is small."""
        rows, _ = self.rref_basis()
        k = len(rows)
        for mask in range(1 << k):
            w = 0
            m = mask
            i = 0
            while m:
                if m & 1:
                    w ^= rows[i]
                m >>= 1
                i += 1
            yield w

    # -------------------------------------------------------------------- dual

    def dual(self) -> "Code":
        """The orthogonal complement under the standard inner product.

        Construction: put ``self`` in RREF with pivot columns ``P`` and free
        columns ``Q``. For each free column ``j`` we build a dual basis vector
        ``v_j`` with bit ``j`` set and, for each ``i``, bit ``P[i]`` set iff
        the RREF row ``i`` has bit ``j`` set. Then ``{v_j : j ∈ Q}`` is a
        basis of ``C⊥``.
        """
        rows, pivots = self.rref_basis()
        pivot_set = set(pivots)
        free_cols = [c for c in range(self.n) if c not in pivot_set]
        dual_basis: list[BinVec] = []
        for j in free_cols:
            v = 1 << j
            for row, p in zip(rows, pivots):
                if (row >> j) & 1:
                    v |= 1 << p
            dual_basis.append(v)
        return Code(n=self.n, basis=tuple(dual_basis))

    # ----------------------------------------------------------- augmentation

    def extend(self, v: BinVec) -> "Code":
        """Return the code spanned by ``self`` together with ``v``.

        No check is made that ``v`` is independent of ``self``; if it isn't,
        :attr:`rank` of the result still equals :attr:`rank` of ``self``.
        Callers that want to enforce dimension growth should test
        ``v not in self`` first.
        """
        return Code(n=self.n, basis=self.basis + (v,))

    def is_orthogonal_to(self, v: BinVec) -> bool:
        """True iff ``v`` is in ``self.dual()``. Equivalently, ``⟨b, v⟩ = 0``
        for every basis vector ``b``."""
        return all(dot(b, v) == 0 for b in self.basis)

    # -------------------------------------------------------------- niceties

    def __repr__(self) -> str:  # pragma: no cover
        return f"Code(n={self.n}, k={self.rank}, basis={list(self.basis)!r})"
