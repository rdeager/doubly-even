"""Feulner-style column-side canonicaliser for binary linear codes.

The Feulner / Sage `codecan` algorithm searches an `N`-leaf tree of
column individualisations rather than building a `2^k + N`-vertex
bipartite graph for nauty. For our `[N, k≤N/2]` doubly-even shape this
trades a `2^k` blow-up for backtracking with cheap partition-refinement
work per node.

This module is the Python staging of that algorithm — the correctness
oracle for an eventual Rust port. It is deliberately written for
readability, not speed. The contract matches
:class:`doubly_even.canon.nauty.CanonInfo` exactly so callers can swap
this in place of :func:`canon_info`.

Binary + permutation-only specialisation drops every layer of Sage's
generic implementation that handles `(F_q*)^n` scaling, `Aut(F_q)`
Frobenius, and the semilinear/projective machinery. Concretely:

* The equivalence is `C ∼ π(C)` for `π ∈ S_N`.
* The canonical form is the lex-minimum `rref(π·G)` over `π`.
* Partition refinement is column-only, using low-weight codewords (per
  Bouyukliev 2019 §3.3 and Sage's choice) as `Aut(C)`-invariant
  refiners. The signature of column `j` is built from how each refiner
  intersects the current cells.
* Automorphism generators are discovered at search leaves: any two
  leaves with the same canonical `rref` differ by an element of
  `Aut(C)`. Schreier-Sims (reused from :mod:`.permutations`) turns the
  generator set into an exact order.

Pruning is "first orbit element" under the subgroup of currently-known
generators that fix every column already individualised on the path.
This is safe — generators we skip are reachable through composition
with what we discover — and is what keeps the search from visiting
`Aut(C)`-orbit-equivalent siblings.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..spec.codes import Code
from ..spec.vectors import apply_permutation
from .nauty import CanonInfo
from .permutations import Perm, compose, group_order, inverse

try:  # pragma: no cover -- import-side switch
    import doubly_even_kernel as _kernel
except ImportError:  # pragma: no cover
    _kernel = None


def canon_info_feulner_native(C: Code) -> CanonInfo:
    """`CanonInfo` via the Rust Feulner kernel.

    Returns the same contract as :func:`canon_info_feulner` (and as the
    nauty-based :func:`doubly_even.canon.nauty.canon_info`). Falls back
    to the pure-Python implementation if the kernel isn't available.
    """
    if _kernel is None:
        return canon_info_feulner(C)
    rref, _ = C.rref_basis()
    canonical_column_order, aut_generators, aut_order_str, column_orbits = (
        _kernel.canon_info_feulner_native(rref, C.n)
    )
    return CanonInfo(
        canonical_column_order=tuple(canonical_column_order),
        aut_generators=tuple(tuple(g) for g in aut_generators),
        aut_order=int(aut_order_str),
        column_orbits=tuple(column_orbits),
    )


@dataclass
class _SearchState:
    """Mutable state carried through the recursive search."""

    n: int
    rref_to_pi: dict[tuple[int, ...], Perm] = field(default_factory=dict)
    best_rref: tuple[int, ...] | None = None
    aut_gens: list[Perm] = field(default_factory=list)
    _seen_gens: set[Perm] = field(default_factory=set)

    def push_aut(self, g: Perm) -> None:
        """Dedupe before pushing — a leaf that matches the canonical
        re-yields the same permutation through different paths often
        enough that uniqueness saves both memory and per-iteration
        ``fixing_gens`` filter work."""
        if g in self._seen_gens:
            return
        self._seen_gens.add(g)
        self.aut_gens.append(g)


def canon_info_feulner(C: Code) -> CanonInfo:
    """Compute :class:`CanonInfo` via column-side partition refinement.

    Same contract as :func:`doubly_even.canon.nauty.canon_info`: returns
    canonical column order, automorphism generators, exact `|Aut(C)|`,
    and a column-orbit assignment.
    """
    n = C.n
    rref, _ = C.rref_basis()
    k = len(rref)

    if n == 0:
        return CanonInfo((), (), 1, ())
    if k == 0 or k == n:
        return _sn_canon_info(n)

    refiners = _invariant_refiners(rref, k)
    state = _SearchState(n=n)
    _search(_initial_partition(rref, n), rref, refiners, state, path=())

    aut_gens = tuple(state.aut_gens)
    aut_order = group_order(aut_gens, n) if aut_gens else 1
    assert state.best_rref is not None
    transporter = state.rref_to_pi[state.best_rref]
    column_orbits = _column_orbits(aut_gens, n)

    return CanonInfo(
        canonical_column_order=transporter,
        aut_generators=aut_gens,
        aut_order=aut_order,
        column_orbits=column_orbits,
    )


def _sn_canon_info(n: int) -> CanonInfo:
    """`CanonInfo` for codes with full `S_n` automorphism (zero / whole space)."""
    gens: list[Perm] = []
    if n >= 2:
        swap = list(range(n))
        swap[0], swap[1] = 1, 0
        gens.append(tuple(swap))
    if n >= 3:
        gens.append(tuple(list(range(1, n)) + [0]))
    return CanonInfo(
        canonical_column_order=tuple(range(n)),
        aut_generators=tuple(gens),
        aut_order=math.factorial(n),
        column_orbits=(0,) * n,
    )


def _initial_partition(rref: tuple[int, ...], n: int) -> list[list[int]]:
    """Aut-invariant initial column partition.

    A column `j` is "covered" by `C` iff some codeword has bit `j` set —
    equivalently iff some rref row does (rref spans the same support set
    as the code). Aut(C) is a subgroup of the support's symmetric group
    direct sum with the non-support's symmetric group, so this two-cell
    split costs nothing and is strictly finer than `{0..n-1}` whenever
    `C` has any non-covered column.
    """
    support_mask = 0
    for row in rref:
        support_mask |= row
    nonzero = [j for j in range(n) if (support_mask >> j) & 1]
    zero = [j for j in range(n) if not (support_mask >> j) & 1]
    cells: list[list[int]] = []
    if nonzero:
        cells.append(nonzero)
    if zero:
        cells.append(zero)
    return cells or [list(range(n))]


def _invariant_refiners(rref: tuple[int, ...], k: int) -> list[int]:
    """Enumerate `Aut(C)`-invariant refiner codewords.

    Prefers weight-4 codewords (the chromotopology strata for doubly-even
    codes; unusually discriminative per Bouyukliev–Bouyuklieva 2019).
    Falls back to the lowest nonzero weight stratum if no weight-4 word
    exists — for very low-dimensional codes this can be weight 8 or
    higher. Empty for the zero code (handled by the special-case path).
    """
    by_weight: dict[int, list[int]] = defaultdict(list)
    for mask in range(1, 1 << k):
        w = 0
        for i in range(k):
            if (mask >> i) & 1:
                w ^= rref[i]
        by_weight[w.bit_count()].append(w)
    if 4 in by_weight:
        return by_weight[4]
    if not by_weight:
        return []
    return by_weight[min(by_weight)]


def _refine(P: list[list[int]], refiners: list[int]) -> list[list[int]]:
    """Equitable refinement of column partition `P` using `refiners`.

    A column `j`'s signature is the tuple, over types of refiner, of how
    many refiners of that type include column `j`. Refiner types are
    determined by the multiset of how the refiner intersects the current
    cells of `P` — so when `P` becomes finer (via individualisation in
    the caller) the refiner types split too, and the column signatures
    distinguish more columns.

    Returns a new partition. The input is not mutated.
    """
    while True:
        cell_masks = [_mask_of_cell(c) for c in P]
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for w in refiners:
            t = tuple((w & m).bit_count() for m in cell_masks)
            groups[t].append(w)
        ordered_types = sorted(groups.keys())

        sig_of: dict[int, tuple[int, ...]] = {}
        for cell in P:
            for j in cell:
                s = []
                for t in ordered_types:
                    count = 0
                    for w in groups[t]:
                        if (w >> j) & 1:
                            count += 1
                    s.append(count)
                sig_of[j] = tuple(s)

        new_P: list[list[int]] = []
        changed = False
        for cell in P:
            if len(cell) == 1:
                new_P.append(cell)
                continue
            buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for j in cell:
                buckets[sig_of[j]].append(j)
            if len(buckets) > 1:
                changed = True
            for s in sorted(buckets.keys()):
                new_P.append(sorted(buckets[s]))
        if not changed:
            return P
        P = new_P


def _mask_of_cell(cell: list[int]) -> int:
    m = 0
    for j in cell:
        m |= 1 << j
    return m


def _individualise(
    P: list[list[int]], cell_idx: int, col: int
) -> list[list[int]]:
    """Pull `col` out of `P[cell_idx]` as a singleton placed before the rest."""
    cell = P[cell_idx]
    rest = [c for c in cell if c != col]
    new_cells: list[list[int]] = [[col]]
    if rest:
        new_cells.append(rest)
    return P[:cell_idx] + new_cells + P[cell_idx + 1 :]


def _rref(rows: list[int], n: int) -> tuple[int, ...]:
    """Row-reduce in place, return the nonzero rows as a tuple.

    A local copy of the RREF kernel from `spec.codes` — we use a local
    routine here instead of the module-level lru-cached one because the
    permuted bases we feed are almost always unique (would pollute the
    cache with one-shot entries).
    """
    pivots = 0
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
        pivots += 1
        r += 1
    return tuple(rows[:pivots])


def _search(
    P: list[list[int]],
    G: tuple[int, ...],
    refiners: list[int],
    state: _SearchState,
    path: tuple[int, ...],
) -> None:
    P = _refine(P, refiners)

    if all(len(cell) == 1 for cell in P):
        pi_list = [0] * state.n
        for new_pos, cell in enumerate(P):
            pi_list[cell[0]] = new_pos
        pi: Perm = tuple(pi_list)

        permuted = [apply_permutation(g, pi_list) for g in G]
        rref_tuple = _rref(permuted, state.n)

        prior = state.rref_to_pi.get(rref_tuple)
        if prior is not None:
            # Two leaves with the same canonical RREF ⇒ pi · prior^-1 ∈ Aut.
            # In our convention this is compose(inverse(prior), pi).
            state.push_aut(compose(inverse(prior), pi))
        else:
            state.rref_to_pi[rref_tuple] = pi
            if state.best_rref is None or rref_tuple < state.best_rref:
                state.best_rref = rref_tuple
        return

    cell_idx = next(i for i, c in enumerate(P) if len(c) > 1)
    cell = P[cell_idx]

    # Iterate cell columns, refreshing orbit_rep every step so newly
    # discovered automorphisms (added by descendant leaves of earlier
    # iterations) prune later siblings that lie in their orbit.
    seen: set[int] = set()
    last_n_gens = -1
    orbit_rep: dict[int, int] = {c: c for c in cell}
    for col in cell:
        if len(state.aut_gens) != last_n_gens:
            last_n_gens = len(state.aut_gens)
            fixing_gens = [
                g for g in state.aut_gens if all(g[p] == p for p in path)
            ]
            orbit_rep = (
                _orbits_on_subset(fixing_gens, cell)
                if fixing_gens
                else {c: c for c in cell}
            )
        rep = orbit_rep[col]
        if rep in seen:
            continue
        seen.add(rep)
        _search(
            _individualise(P, cell_idx, col),
            G,
            refiners,
            state,
            path + (col,),
        )


def _orbits_on_subset(gens: list[Perm], subset: list[int]) -> dict[int, int]:
    """Orbit-rep map: column → smallest column in its orbit under `<gens>`.

    Assumes `gens` already fix every column outside `subset` (i.e. they
    permute `subset` to itself). Used during search after filtering
    generators by "fixes the path", which preserves the cell.
    """
    parent: dict[int, int] = {c: c for c in subset}

    def find(c: int) -> int:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for g in gens:
        for c in subset:
            j = g[c]
            if j != c:
                union(c, j)

    return {c: find(c) for c in subset}


def _column_orbits(aut_gens: tuple[Perm, ...], n: int) -> tuple[int, ...]:
    """Orbits of `{0, …, n-1}` under `<aut_gens>`, labelled by smallest member."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for g in aut_gens:
        for i, j in enumerate(g):
            if i != j:
                union(i, j)

    return tuple(find(i) for i in range(n))
