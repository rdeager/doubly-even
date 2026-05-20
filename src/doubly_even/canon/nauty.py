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

import os
from dataclasses import dataclass
from functools import lru_cache

import pynauty

from ..spec.codes import Code
from ..spec.vectors import apply_permutation
from .bipartite import bipartite_graph
from .permutations import group_order

try:  # pragma: no cover -- import-side switch
    import doubly_even_kernel as _kernel
except ImportError:  # pragma: no cover
    _kernel = None

# Selecting `feulner` makes `canon_info` route through the Rust Feulner
# canonicaliser instead of nauty. Defaults to nauty for safety until
# benchmarks confirm Feulner is faster on the doubly-even shape.
_CANON_BACKEND = os.environ.get("DOUBLY_EVEN_CANON_BACKEND", "nauty").lower()


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


_FLOAT_INT_LIMIT = 1 << 53  # 2^53; largest integer exactly representable in float64.


def _trustable_pynauty_order(grpsize1: float, grpsize2: int) -> int | None:
    """Return the pynauty-reported order as an exact int when it can be
    trusted, otherwise ``None``.

    pynauty reports ``|Aut| = grpsize1 * 10**grpsize2`` with ``grpsize1`` a
    Python ``float``. Empirically ``grpsize1`` is **not** normalised to
    ``[1, 10)``: pynauty may return e.g. ``(2092.27..., 10)`` at ``n =
    16`` whose product is still exactly ``16!``. The precision limit that
    matters is whether ``grpsize1 * 10**grpsize2`` fits in float64's
    exact-integer range (``< 2^53 ≈ 9.0e15``). Above that, the rounded
    product is wrong; below, it is exact.

    If we can't trust the float, we punt to :func:`group_order`
    (Schreier–Sims), which gives an exact integer regardless of the size.
    """
    raw = grpsize1 * (10 ** grpsize2)
    if raw < _FLOAT_INT_LIMIT:
        return int(round(raw))
    return None


def canon_info(C: Code) -> CanonInfo:
    """Compute canonical form and automorphism group of ``C``.

    Backend dispatch (in order):

    1. ``DOUBLY_EVEN_CANON_BACKEND=feulner`` with the Rust kernel built →
       :func:`doubly_even.canon.feulner.canon_info_feulner_native`.
    2. ``DOUBLY_EVEN_CANON_BACKEND=sage_partn_ref`` → bench-only daemon
       path that proxies to a long-lived Sage subprocess running
       :class:`LinearBinaryCodeStruct` (Robert Miller's binary-specialised
       partition refinement). See
       :mod:`doubly_even.canon.experimental.sage_proxy`.
    3. Rust kernel built → :func:`_canon_info_via_kernel` (nauty bipartite path).
    4. Pure Python fallback → :func:`_canon_info_via_pynauty`.
    """
    if _CANON_BACKEND == "feulner" and _kernel is not None:
        # Local import to avoid a top-level circular dep with .feulner.
        from .experimental.feulner import canon_info_feulner_native
        return canon_info_feulner_native(C)
    if _CANON_BACKEND == "sage_partn_ref":
        from .experimental.sage_proxy import canon_info_via_sage
        return canon_info_via_sage(C)
    if _kernel is not None:
        return _canon_info_via_kernel(C)
    return _canon_info_via_pynauty(C)


def _canon_info_via_kernel(C: Code) -> CanonInfo:
    """Same contract as :func:`canon_info`, via the Rust kernel.

    Builds the bipartite codeword × column sparsegraph entirely in Rust and
    runs `sparsenauty` directly — no Python `bipartite_graph` dict, no
    pynauty wrapper. One FFI call per parent vs pynauty's two (`autgrp` +
    `canon_label`).
    """
    rref, _ = C.rref_basis()
    canonical_column_order, aut_generators, grpsize1, grpsize2, column_orbits = (
        _kernel.canon_info_native(rref, C.n)
    )
    aut_generators_tuple = tuple(tuple(g) for g in aut_generators)
    aut_order = _trustable_pynauty_order(grpsize1, grpsize2)
    if aut_order is None:
        aut_order = group_order(aut_generators_tuple, C.n)
    return CanonInfo(
        canonical_column_order=tuple(canonical_column_order),
        aut_generators=aut_generators_tuple,
        aut_order=aut_order,
        column_orbits=tuple(column_orbits),
    )


def _canon_info_via_pynauty(C: Code) -> CanonInfo:
    """Reference Python path; preserved as the cross-check oracle."""
    enc = bipartite_graph(C)

    # ---- automorphism group --------------------------------------------------
    gens, grpsize1, grpsize2, orbits, _numorbits = pynauty.autgrp(enc.graph)

    # Project graph-automorphism generators onto column actions. The induced
    # left-side action is determined by the right-side action, so the column
    # projection generates the full code-automorphism group.
    aut_generators: list[tuple[int, ...]] = []
    for sigma in gens:
        col_perm = _column_part(sigma, enc.L, enc.R)
        aut_generators.append(col_perm)

    # Prefer pynauty's reported order when it is within float precision —
    # that's the case for the vast majority of codes in the recursion,
    # which have ``|Aut|`` well below ``2^53``. Fall back to Schreier–Sims
    # only when the float order is too large to be exact (e.g. the zero
    # code at ``N ≥ 19`` has ``|Aut| = N!``).
    aut_order = _trustable_pynauty_order(grpsize1, grpsize2)
    if aut_order is None:
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


@lru_cache(maxsize=None)
def _canon_info_by_rref(n: int, rref: tuple[int, ...]) -> CanonInfo:
    """Internal cache hook for :func:`cached_canon_info`.

    Keyed by the canonical RREF subspace identifier so that two ``Code``
    instances representing the same subspace via different bases collide
    on the same entry.
    """
    return canon_info(Code(n=n, basis=rref))


def cached_canon_info(C: Code) -> CanonInfo:
    """Memoised :func:`canon_info`, keyed by ``C``'s RREF subspace.

    The McKay parent test re-enters ``canon_info`` for every candidate
    extension; many candidates share ancestors via the subspace orbit
    BFS in :func:`doubly_even.enumerate.augment._in_aut_orbit_of_subspace`
    and the recursion re-encounters the same subspace from different
    sibling branches. Each cache hit replaces a pynauty call plus
    Schreier–Sims with a dict lookup.

    The cache grows unbounded; call :func:`canon_info_cache_clear` to
    reset (e.g. between benchmark runs that want cold-cache timings).
    """
    return _canon_info_by_rref(C.n, C.rref_basis()[0])


def canon_info_cache_clear() -> None:
    """Drop every entry from the :func:`cached_canon_info` cache."""
    _canon_info_by_rref.cache_clear()


def canon_info_cache_info():
    """Return ``functools.lru_cache.cache_info()`` for inspection."""
    return _canon_info_by_rref.cache_info()


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
