"""McKay canonical augmentation (DFGHILM Appendix B.4): the recursion driver.

Accept ``D = ⟨C, v⟩`` iff ``C`` lies in the Aut(D)-orbit of ``D``'s canonical
parent ``p(D)``. ``p(D)`` is built by applying ``D``'s canonical column
permutation, RREF-ing, dropping the last row, then un-permuting. The orbit
test is a BFS comparing subspace RREFs (the canonical subspace key).
"""

from __future__ import annotations
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ._canon import CanonInfo, canon_info, perm_inverse
from ._spec import Code, apply_perm, rref_gf2


@dataclass(frozen=True)
class EnumeratedCode:
    code: Code
    info: CanonInfo

    @property
    def aut_order(self) -> int:
        return self.info.aut_order


def canonical_parent(D: Code, info_D: CanonInfo) -> Code:
    """``p(D)``: apply σ_D, RREF, drop last row, apply σ_D⁻¹."""
    if D.rank == 0:
        raise ValueError("canonical_parent undefined for the zero code")
    sigma = list(info_D.canonical_column_order)
    permuted_basis = [apply_perm(b, sigma) for b in D.basis]
    rref_rows, _ = rref_gf2(permuted_basis, D.n)
    parent_in_canon = rref_rows[:-1]
    inv_sigma = list(perm_inverse(tuple(sigma)))
    return Code(D.n, tuple(apply_perm(b, inv_sigma) for b in parent_in_canon))


def _subspace_in_orbit(C, target, aut_generators, n):
    """BFS on RREF keys: does some ``g`` send subspace ``C`` to ``target``?"""
    target_key = target.rref()[0]
    start_key = C.rref()[0]
    if start_key == target_key:
        return True
    sigma_lists = [list(g) for g in aut_generators]
    if not sigma_lists:
        return False
    seen = {start_key}
    queue = [start_key]
    while queue:
        next_q: list = []
        for basis in queue:
            for sigma in sigma_lists:
                new_basis = [apply_perm(b, sigma) for b in basis]
                key = tuple(rref_gf2(new_basis, n)[0])
                if key == target_key:
                    return True
                if key in seen:
                    continue
                seen.add(key)
                next_q.append(key)
        queue = next_q
    return False


def is_canonical_augmentation(C: Code, D: Code, info_D: CanonInfo) -> bool:
    parent = canonical_parent(D, info_D)
    return _subspace_in_orbit(C, parent, info_D.aut_generators, D.n)


def traverse(C: Code, max_k: int, info_C: CanonInfo | None = None) -> Iterator["EnumeratedCode"]:
    """Depth-first canonical-augmentation traversal rooted at ``C``."""
    from ._qc import qc_candidates  # avoid circular import at module load

    if info_C is None:
        info_C = canon_info(C)
    yield EnumeratedCode(code=C, info=info_C)
    if C.rank >= max_k:
        return
    for v in qc_candidates(C, info_C.aut_generators):
        D = C.extend(v)
        info_D = canon_info(D)
        if not is_canonical_augmentation(C, D, info_D):
            continue
        yield from traverse(D, max_k, info_D)


def enumerate_doubly_even_sequential(N: int, max_k: int | None = None):
    cap = N // 2 if max_k is None else max_k
    yield from traverse(Code.zero(N), cap)
