"""Canonical form and automorphism group of a code, via pynauty.

Given a :class:`~doubly_even.spec.codes.Code` ``C`` we build the bipartite
encoding (see :mod:`.bipartite`) and ask nauty for:

1. a *canonical label* — a permutation of the vertices that maps ``G(C)``
   to its canonical form; restricted to the right (column) side this gives
   a canonical column permutation ``π`` for ``C``;
2. *generators of* ``Aut(G(C))``, restricted to their column-side action,
   which together generate the column-permutation automorphism group
   ``Aut(C)``;
3. the *order* ``|Aut(C)|``.

These three pieces are returned via :func:`canon_info`.

**On precision of** ``|Aut(C)|``. pynauty returns the order as a pair
``(grpsize1, grpsize2)`` with ``order = grpsize1 * 10**grpsize2``, where
``grpsize1`` is a float in ``[1, 10)``. Beyond about ``10^16`` the rounded
result diverges from the true integer (e.g. at ``n = 20`` the zero code's
order ``20!`` is off by ~512). We therefore *recompute* the order from
the generator permutations using :func:`.permutations.group_order`
(Schreier–Sims) so that the value stored in :attr:`CanonInfo.aut_order`
is always exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import pynauty

from ..spec.codes import Code
from ..spec.vectors import apply_permutation
from .bipartite import bipartite_graph
from .permutations import group_order


@dataclass(frozen=True)
class CanonInfo:
    """Output of :func:`canon_info` for a code ``C``.

    Attributes
    ----------
    canonical_column_order
        A permutation ``π`` of ``range(n)`` such that applying ``π`` to
        the columns of ``C`` produces a canonical form. Stored as a tuple
        where ``π[i] = j`` means "old column ``i`` becomes new column ``j``"
        — the same convention used by
        :func:`doubly_even.spec.vectors.apply_permutation`.
    aut_generators
        Generators of ``Aut(C)``, each a permutation of ``range(n)`` in the
        same convention. The generators are the column-restrictions of the
        bipartite-graph automorphism generators returned by nauty.
    aut_order
        The order of ``Aut(C)``, rounded to the nearest integer from
        pynauty's floating-point representation. See module docstring.
    column_orbits
        ``column_orbits[i]`` is the orbit identifier of column ``i`` under
        ``Aut(C)`` — two columns are in the same orbit iff they share an
        identifier. Useful for cheap pre-filtering.
    """

    canonical_column_order: tuple[int, ...]
    aut_generators: tuple[tuple[int, ...], ...]
    aut_order: int
    column_orbits: tuple[int, ...]


def canon_info(C: Code) -> CanonInfo:
    """Compute canonical form and automorphism group of ``C``."""
    enc = bipartite_graph(C)

    # ---- automorphism group --------------------------------------------------
    gens, _grpsize1, _grpsize2, orbits, _numorbits = pynauty.autgrp(enc.graph)

    # Project graph-automorphism generators onto column actions. The induced
    # left-side action is determined by the right-side action, so the column
    # projection generates the full code-automorphism group.
    aut_generators: list[tuple[int, ...]] = []
    for sigma in gens:
        col_perm = _column_part(sigma, enc.L, enc.R)
        aut_generators.append(col_perm)

    # pynauty's grpsize1 is a double — beyond ~10^16 the rounded product
    # drifts from the true integer (20! is already off by ~512). Recompute
    # exactly from the projected generators via Schreier–Sims.
    aut_order = group_order(aut_generators, enc.R)

    column_orbits = tuple(orbits[enc.L:])

    # ---- canonical column ordering ------------------------------------------
    # pynauty.canon_label returns a list ``π`` such that vertex π[i] in the
    # original graph becomes vertex i in the canonical labeling — i.e. it is
    # the inverse of the permutation that takes original → canonical. We want
    # the "old → new" form for the column part.
    canon_perm = pynauty.canon_label(enc.graph)
    # canon_perm[new_index] = old_index. We want old → new for the right side.
    # Build inverse on the right side.
    new_to_old_right: list[int] = []
    for new_index, old_vertex in enumerate(canon_perm):
        if old_vertex >= enc.L:
            new_to_old_right.append(old_vertex - enc.L)  # old column index
    # new_to_old_right[k] = "old column landing at the k-th right-side slot".
    # The "old → new" map for columns is the inverse.
    old_to_new = [0] * enc.R
    for new_col, old_col in enumerate(new_to_old_right):
        old_to_new[old_col] = new_col
    canonical_column_order = tuple(old_to_new)

    return CanonInfo(
        canonical_column_order=canonical_column_order,
        aut_generators=tuple(aut_generators),
        aut_order=aut_order,
        column_orbits=column_orbits,
    )


def canonical_form(C: Code) -> Code:
    """Return the canonically-relabelled code, in RREF.

    Two codes are permutation-equivalent iff their :func:`canonical_form`
    outputs are equal as :class:`Code` values (same ``n``, same RREF basis).
    """
    info = canon_info(C)
    permuted_basis = tuple(
        apply_permutation(b, list(info.canonical_column_order))
        for b in C.basis
    )
    # Re-RREF after permutation so that the *basis* is also canonical.
    relabeled = Code(n=C.n, basis=permuted_basis)
    rref_rows, _ = relabeled.rref_basis()
    return Code(n=C.n, basis=rref_rows)


def are_equivalent(C1: Code, C2: Code) -> bool:
    """Test permutation equivalence by comparing canonical forms."""
    if C1.n != C2.n:
        return False
    if C1.rank != C2.rank:
        return False
    return canonical_form(C1) == canonical_form(C2)


def _column_part(sigma: list[int], L: int, R: int) -> tuple[int, ...]:
    """Extract the column action of a bipartite graph automorphism.

    ``sigma`` is a permutation of ``range(L + R)`` returned by pynauty.
    Vertices ``[L, L+R)`` are columns. Returns a tuple of length ``R``
    giving the "old → new" column permutation, in the convention used by
    :func:`doubly_even.spec.vectors.apply_permutation`.
    """
    out = [0] * R
    for old_vertex in range(L, L + R):
        new_vertex = sigma[old_vertex]
        old_col = old_vertex - L
        new_col = new_vertex - L
        out[old_col] = new_col
    return tuple(out)
