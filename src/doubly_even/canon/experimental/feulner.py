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

from ...spec.codes import Code
from ..nauty import CanonInfo
from ..permutations import (
    Perm,
    compose,
    group_order,
    identity,
    inverse,
    orbit_and_transversal,
)

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
    k: int
    key_to_pi: dict[tuple[int, ...], Perm] = field(default_factory=dict)
    best_key: tuple[int, ...] | None = None
    aut_gens: list[Perm] = field(default_factory=list)
    _seen_gens: set[Perm] = field(default_factory=set)
    # Counters (Phase A diagnostic; tiny overhead, kept for Phase B tuning).
    leaves_visited: int = 0
    prune_fires: int = 0
    clb_prune_fires: int = 0
    use_clb: bool = True
    clb: _LabelledBranching = field(init=False)

    def __post_init__(self) -> None:
        self.clb = _LabelledBranching(n=self.n)

    def push_aut(self, g: Perm) -> None:
        """Dedupe before pushing — a leaf that matches the canonical
        re-yields the same permutation through different paths often
        enough that uniqueness saves both memory and per-iteration
        ``fixing_gens`` filter work."""
        if g in self._seen_gens:
            return
        self._seen_gens.add(g)
        self.aut_gens.append(g)
        if self.use_clb:
            self.clb.add_gen(g)


@dataclass
class _PartialKey:
    """Incremental canonical-form key threaded through the search.

    Encodes Sage's lex-from-low column trace: at each singleton emission
    we Gaussian-eliminate that column to a unit vector and **swap the
    pivot row to position `depth`** so the column trace is invariant
    under the inner group `GL_k(F_2)` (i.e. independent of which
    original row happened to be the pivot). The key is compared
    **lex-from-low** (entry 0 most significant), so partial information
    at depth `d` sits in the most-significant prefix and the prefix
    prune is structurally strong.

    Invariant: rows `0..depth-1` of `work` are pivot rows in pivot
    order; rows `depth..k-1` are uncovered. For an absorbed pivot column
    `c` the post-swap layout has bit `c` set exactly in row `depth-1`
    (the just-placed pivot), so its key entry is `1 << (k-1-(depth-1))`.
    For a non-pivot column, bits are confined to rows `0..depth-1`.

    Recursive descents receive a fresh copy via `copy()`, so the
    caller's state survives backtrack without explicit restore.
    """

    k: int
    work: list[int]
    depth: int = 0  # rows 0..depth-1 are pivot rows in pivot order
    key: list[int] = field(default_factory=list)
    absorbed_cols: int = 0  # bitmask: bit c set iff col c is absorbed

    def copy(self) -> "_PartialKey":
        return _PartialKey(
            k=self.k,
            work=list(self.work),
            depth=self.depth,
            key=list(self.key),
            absorbed_cols=self.absorbed_cols,
        )

    def absorb(self, c: int) -> None:
        """Absorb column `c`: pivot-swap + Gaussian elim + append to key."""
        pivot = -1
        for r in range(self.depth, self.k):
            if (self.work[r] >> c) & 1:
                pivot = r
                break
        if pivot >= 0:
            if pivot != self.depth:
                self.work[self.depth], self.work[pivot] = (
                    self.work[pivot],
                    self.work[self.depth],
                )
            pivot_val = self.work[self.depth]
            for r in range(self.k):
                if r != self.depth and (self.work[r] >> c) & 1:
                    self.work[r] ^= pivot_val
            self.depth += 1
        col_bits = 0
        for r in range(self.k):
            if (self.work[r] >> c) & 1:
                col_bits |= 1 << (self.k - 1 - r)
        self.key.append(col_bits)
        self.absorbed_cols |= 1 << c


@dataclass
class _LabelledBranching:
    """Jerrum's complete labelled branching of `<gens> ≤ S_n` (Feulner §5.2).

    The branching is encoded by the parent array ``father``:
    ``father[j] = i`` means there is an arc ``(i → j)`` in `B` with
    ``i < j``; ``father[j] = -1`` means ``j`` is a root of the forest.

    Two operations:

    * :meth:`add_gen` records a new generator. The ``father`` array is
      rebuilt lazily on the next :meth:`has_empty_intersection` call.
    * :meth:`has_empty_intersection` is Lemma 5.9: given the current
      ordered partition `π = [α_0, …, α_m]`, return ``True`` iff the
      coset ``S_π·π`` contains **no** topological sort of `B` — in which
      case the entire subtree at this search node is unreachable from
      any canonical leaf and can be pruned.

    Mirrors Sage's ``LabelledBranching`` from
    ``sage/groups/perm_gps/partn_ref2/refinement_generic.pyx``. Sage
    delegates the stabilizer chain to GAP; we use the local
    Schreier–Sims walk (same one as :func:`permutations.group_order`).
    """

    n: int
    father: list[int] = field(default_factory=list)
    gens: list[Perm] = field(default_factory=list)
    _seen: set[Perm] = field(default_factory=set)
    dirty: bool = False

    def __post_init__(self) -> None:
        if not self.father:
            self.father = [-1] * self.n

    def add_gen(self, g: Perm) -> None:
        """Record `g` as a known automorphism; defer ``father`` rebuild."""
        if g in self._seen:
            return
        self._seen.add(g)
        self.gens.append(g)
        self.dirty = True

    def _rebuild_father(self) -> None:
        """Walk a Schreier–Sims chain on points 0..n-1, fill ``father``.

        At each base point ``i`` (in increasing order), if the residual
        group acts non-trivially on ``i``: set ``father[j] = i`` for every
        ``j ≠ i`` in the orbit of ``i``. Then recurse on the stabiliser
        (Schreier generators). Same procedure as Sage's GAP-driven walk,
        only with our hand-rolled Schreier–Sims.
        """
        self.father = [-1] * self.n
        if not self.gens:
            self.dirty = False
            return
        gens: list[Perm] = list(self.gens)
        id_p = identity(self.n)
        for base in range(self.n):
            if not gens:
                break
            orbit, transversal = orbit_and_transversal(gens, base, self.n)
            if len(orbit) == 1:
                continue
            for j in orbit:
                if j != base:
                    self.father[j] = base
            new_gens: set[Perm] = set()
            for p in orbit:
                t_p = transversal[p]
                for g in gens:
                    q = g[p]
                    t_q_inv = inverse(transversal[q])
                    sg = compose(t_q_inv, compose(g, t_p))
                    if sg != id_p:
                        new_gens.add(sg)
            if not new_gens:
                break
            gens = list(new_gens)
        self.dirty = False

    def has_empty_intersection(self, partition: list[list[int]]) -> bool:
        """Lemma 5.9 topological-sort test.

        ``partition`` is interpreted as the ordered cell list of a
        Young-subgroup coset: position ``k`` in the flattened cell
        sequence corresponds to column ``flatten[k]``. Returns ``True``
        iff some arc ``(i → j)`` in `B` has its target ``j`` placed
        **before** its source ``i`` in a strictly earlier cell — i.e.
        no topological sort of `B` exists in the coset, so the subtree
        cannot contain a canonical representative.
        """
        if self.dirty:
            self._rebuild_father()
        if not self.gens:
            return False

        pos = [0] * self.n
        cell_of = [0] * self.n
        k = 0
        for ci, cell in enumerate(partition):
            for col in cell:
                pos[col] = k
                cell_of[col] = ci
                k += 1

        for j in range(self.n):
            i = self.father[j]
            if i == -1:
                continue
            if cell_of[j] < cell_of[i]:
                return True
        return False


def canon_info_feulner(C: Code, *, use_clb: bool = True) -> CanonInfo:
    """Compute :class:`CanonInfo` via column-side partition refinement.

    Same contract as :func:`doubly_even.canon.nauty.canon_info`: returns
    canonical column order, automorphism generators, exact `|Aut(C)|`,
    and a column-orbit assignment.

    ``use_clb`` (default ``True``): apply Jerrum's complete labelled
    branching + Lemma 5.9 topological-sort pruning (Feulner §5.2). The
    pre-CLB orbit-rep filter is kept available behind ``use_clb=False``
    as a regression-only fallback.
    """
    n = C.n
    rref, _ = C.rref_basis()
    k = len(rref)

    if n == 0:
        return CanonInfo((), (), 1, ())
    if k == 0 or k == n:
        return _sn_canon_info(n)

    refiners = _invariant_refiners(rref, k)
    state = _SearchState(n=n, k=k, use_clb=use_clb)
    partial = _PartialKey(k=k, work=list(rref))
    _search(
        _initial_partition(rref, n), rref, refiners, state, path=(), partial=partial
    )

    aut_gens = tuple(state.aut_gens)
    aut_order = group_order(aut_gens, n) if aut_gens else 1
    assert state.best_key is not None
    transporter = state.key_to_pi[state.best_key]
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

    Phase B: returns the **two lowest non-zero weight strata** of the
    code — these are the most discriminative invariants under
    Aut(C) (Bouyukliev–Bouyuklieva 2019 §3.3). The Phase-A choice of
    only weight-4 codewords lost discriminative power on (n, k) cells
    where weight-8 codewords also discriminate orbits. The cost is at
    most ~2× more refiner-incidence work in ``_refine``, well worth the
    finer refinement.
    """
    by_weight: dict[int, list[int]] = defaultdict(list)
    for mask in range(1, 1 << k):
        w = 0
        for i in range(k):
            if (mask >> i) & 1:
                w ^= rref[i]
        by_weight[w.bit_count()].append(w)
    if not by_weight:
        return []
    weights = sorted(by_weight.keys())[:2]
    out: list[int] = []
    for w in weights:
        out.extend(by_weight[w])
    return out


def _refine_naive(P: list[list[int]], refiners: list[int]) -> list[list[int]]:
    """Equitable refinement of column partition `P` using `refiners`.

    A column `j`'s signature is the tuple, over types of refiner, of how
    many refiners of that type include column `j`. Refiner types are
    determined by the multiset of how the refiner intersects the current
    cells of `P` — so when `P` becomes finer (via individualisation in
    the caller) the refiner types split too, and the column signatures
    distinguish more columns.

    Returns a new partition. The input is not mutated. Kept as the
    correctness oracle for :func:`_refine_incremental` — at this point
    it is not on the hot path of :func:`_search`.
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


def _refine_incremental(
    P: list[list[int]], refiners: list[int]
) -> list[list[int]]:
    """Equitable refinement via a nauty-style worklist of refiner groups.

    Same contract as :func:`_refine_naive` (the correctness oracle): the
    output partition is equitable under the refiner-incidence signature
    and is a deterministic function of `(P, refiners)`. The cell-order in
    the output is **not** guaranteed to match `_refine_naive` — it's the
    lex-by-signature order of an equitable partition, which is invariant
    under code permutation (so the canonical form is still canonical,
    and aut order / orbits / mass / Table 3 counts are unchanged).

    State maintained across the worklist loop:

    * ``groups[g]``: list of refiner indices in group ``g``.
    * ``p[g][j]``: count of refiners in ``g`` whose bit ``j`` is set.
    * ``worklist`` / ``in_worklist``: pending groups that still need to
      act as splitters.
    * ``cells`` / ``cell_of`` / ``cell_mask``: the current partition, an
      inverse map, and per-cell bitmasks.

    Initial groups are produced by bucketing refiners by their
    cell-histogram (same equivalence as the first iteration of
    :func:`_refine_naive`).
    """
    n_cols_max = 1
    for cell in P:
        if cell:
            n_cols_max = max(n_cols_max, max(cell) + 1)

    # cells / cell_of / cell_mask / cell_lineage. Lineage is the
    # caller's input-cell index; sub-cells inherit their parent's
    # lineage so :func:`_emit_sorted` can group output cells by input
    # cell (preserving the McKay convention that the partition order at
    # any node is "input cells in order, each replaced by its split
    # products").
    cells: list[list[int]] = [list(c) for c in P]
    cell_of: list[int] = [0] * n_cols_max
    cell_mask: list[int] = []
    cell_lineage: list[int] = []
    for ci, cell in enumerate(cells):
        m = 0
        for j in cell:
            cell_of[j] = ci
            m |= 1 << j
        cell_mask.append(m)
        cell_lineage.append(ci)

    if not refiners:
        return _emit_sorted(cells, p=[], cell_lineage=cell_lineage)

    # Initial refiner groups by cell-histogram tuple.
    init_buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for ri, w in enumerate(refiners):
        t = tuple((w & m).bit_count() for m in cell_mask)
        init_buckets[t].append(ri)
    groups: list[list[int]] = [init_buckets[t] for t in sorted(init_buckets.keys())]
    group_of: list[int] = [0] * len(refiners)
    for gi, g in enumerate(groups):
        for ri in g:
            group_of[ri] = gi

    # p[g][j] for each initial group.
    p: list[list[int]] = []
    for g in groups:
        row = [0] * n_cols_max
        for ri in g:
            w = refiners[ri]
            bits = w
            while bits:
                lsb = bits & -bits
                j = lsb.bit_length() - 1
                row[j] += 1
                bits ^= lsb
        p.append(row)

    worklist: list[int] = list(range(len(groups)))
    in_worklist: list[bool] = [True] * len(groups)

    while worklist:
        g = worklist.pop()
        in_worklist[g] = False

        # Find all non-singleton cells that split under p[g][·].
        # Snapshot current cells (we'll mutate `cells` as we go); but
        # cells we haven't touched in this iteration keep stable indices.
        # We iterate by index over a snapshot length; appended sub-cells
        # are processed in subsequent worklist iterations (their own
        # groups go on the worklist).
        ci = 0
        n_snapshot = len(cells)
        while ci < n_snapshot:
            cell = cells[ci]
            if len(cell) > 1:
                buckets: dict[int, list[int]] = defaultdict(list)
                for j in cell:
                    buckets[p[g][j]].append(j)
                if len(buckets) > 1:
                    _apply_cell_split(
                        ci,
                        buckets,
                        cells,
                        cell_of,
                        cell_mask,
                        cell_lineage,
                        groups,
                        group_of,
                        p,
                        worklist,
                        in_worklist,
                        refiners,
                        n_cols_max,
                    )
            ci += 1

    return _emit_sorted(cells, p, cell_lineage)


def _apply_cell_split(
    ci: int,
    buckets: dict[int, list[int]],
    cells: list[list[int]],
    cell_of: list[int],
    cell_mask: list[int],
    cell_lineage: list[int],
    groups: list[list[int]],
    group_of: list[int],
    p: list[list[int]],
    worklist: list[int],
    in_worklist: list[bool],
    refiners: list[int],
    n_cols_max: int,
) -> None:
    """Replace ``cells[ci]`` with its bucketed sub-cells and propagate.

    The largest sub-cell keeps slot ``ci`` (smallest-fragment-style
    bookkeeping at the cell level); the others append to ``cells``. Then
    every refiner group whose refiners had bits in old cell ``ci`` is
    re-bucketed by per-sub-cell histogram; any group that splits is
    propagated to the worklist using the Hopcroft smallest-fragment
    rule.
    """
    old_mask = cell_mask[ci]
    sub_cells_sorted = [sorted(buckets[k]) for k in sorted(buckets.keys())]
    sub_masks = [_mask_of_cell(c) for c in sub_cells_sorted]

    # Largest fragment keeps slot ci.
    sizes = [len(c) for c in sub_cells_sorted]
    largest = max(range(len(sizes)), key=lambda i: (sizes[i], -i))
    parent_lineage = cell_lineage[ci]
    new_cell_ids: list[int] = [0] * len(sub_cells_sorted)
    new_cell_ids[largest] = ci
    cells[ci] = sub_cells_sorted[largest]
    cell_mask[ci] = sub_masks[largest]
    for idx, sub in enumerate(sub_cells_sorted):
        if idx == largest:
            continue
        new_cell_ids[idx] = len(cells)
        cells.append(sub)
        cell_mask.append(sub_masks[idx])
        cell_lineage.append(parent_lineage)
    # cell_of for the (possibly moved) columns.
    for idx, sub in enumerate(sub_cells_sorted):
        cid = new_cell_ids[idx]
        for j in sub:
            cell_of[j] = cid

    # Propagate to groups. Touching groups: any g' with some refiner
    # having a bit in old_mask.
    n_groups_snapshot = len(groups)
    for gi in range(n_groups_snapshot):
        g_refs = groups[gi]
        if not g_refs:
            continue
        if not any((refiners[ri] & old_mask) for ri in g_refs):
            continue

        # Bucket refiners by their histogram across the new sub-cells.
        # Histogram is a tuple of length len(sub_cells_sorted); since
        # the largest sub-cell stayed in slot ci, the histogram is in
        # *sub-cell-order* (ascending by bucket key).
        rbuckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for ri in g_refs:
            w = refiners[ri]
            hist = tuple((w & sm).bit_count() for sm in sub_masks)
            rbuckets[hist].append(ri)
        if len(rbuckets) == 1:
            continue

        # Group gi splits. Replace gi's content with the largest
        # fragment; allocate fresh slots for the others.
        sorted_keys = sorted(rbuckets.keys())
        frag_lists = [rbuckets[k] for k in sorted_keys]
        frag_sizes = [len(f) for f in frag_lists]
        largest_g = max(
            range(len(frag_sizes)), key=lambda i: (frag_sizes[i], -i)
        )

        new_group_ids: list[int] = [0] * len(frag_lists)
        new_group_ids[largest_g] = gi
        groups[gi] = frag_lists[largest_g]
        # Recompute p[gi] from scratch for the largest fragment.
        p[gi] = _recompute_p(frag_lists[largest_g], refiners, n_cols_max)
        for fi, frag in enumerate(frag_lists):
            if fi == largest_g:
                continue
            new_gi = len(groups)
            new_group_ids[fi] = new_gi
            groups.append(frag)
            p.append(_recompute_p(frag, refiners, n_cols_max))
            in_worklist.append(False)
            for ri in frag:
                group_of[ri] = new_gi

        # Smallest-fragment rule: if gi was on worklist, add all new
        # groups (incl. gi); else add all but the largest.
        if in_worklist[gi]:
            for fi in range(len(frag_lists)):
                gid = new_group_ids[fi]
                if not in_worklist[gid]:
                    in_worklist[gid] = True
                    worklist.append(gid)
        else:
            for fi in range(len(frag_lists)):
                if fi == largest_g:
                    continue
                gid = new_group_ids[fi]
                if not in_worklist[gid]:
                    in_worklist[gid] = True
                    worklist.append(gid)


def _recompute_p(
    refs: list[int], refiners: list[int], n_cols_max: int
) -> list[int]:
    """``p[g][j] = #{r in refs : (refiners[r] >> j) & 1}``.

    Iterates set bits of each refiner — cost proportional to weight,
    not ``n_cols_max``.
    """
    row = [0] * n_cols_max
    for ri in refs:
        w = refiners[ri]
        bits = w
        while bits:
            lsb = bits & -bits
            j = lsb.bit_length() - 1
            row[j] += 1
            bits ^= lsb
    return row


def _emit_sorted(
    cells: list[list[int]], p: list[list[int]], cell_lineage: list[int]
) -> list[list[int]]:
    """Sort cells deterministically before returning.

    Cell order: lex by ``(input_lineage, signature, min_col)``. The
    lineage primary key preserves the McKay convention that input cell
    order is respected at the top level — each input cell's split
    products form a contiguous block in the output. Within a block,
    sub-cells are ordered by full signature, then by min column for
    ties. Within each cell, columns are sorted. Empty cells are
    dropped.
    """
    n_groups = len(p)

    def cell_key(idx: int) -> tuple:
        cell = cells[idx]
        rep = min(cell)
        sig = tuple(p[g][rep] for g in range(n_groups))
        return (cell_lineage[idx], sig, rep)

    keyed = [
        (cell_key(i), sorted(cells[i]))
        for i in range(len(cells))
        if cells[i]
    ]
    keyed.sort(key=lambda x: x[0])
    return [c for _, c in keyed]


# Public alias: the search uses the incremental implementation. The
# naive form is retained for the differential oracle test only.
_refine = _refine_incremental


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


def _search(
    P: list[list[int]],
    G: tuple[int, ...],
    refiners: list[int],
    state: _SearchState,
    path: tuple[int, ...],
    partial: _PartialKey,
) -> None:
    """One node of the McKay search.

    Refines `P`, absorbs any new singletons into `partial.key` in cell
    order, then checks the lex-from-low prefix against `state.best_key`.
    Pruning fires when the partial column-trace prefix strictly exceeds
    the best's prefix. Target cell for individualisation is the
    **smallest** non-trivial cell (Phase B heuristic; Feulner/nauty
    default — smaller branching factor at each level).

    CLB pruning (Feulner §5.2, Lemma 5.9) is applied at two points: just
    after partition refinement, and again after singleton absorption (the
    partition order has shifted, so re-test). Either check fires only
    when ``state.best_key is not None`` — before the first leaf, all
    leaves are still candidates and we have no automorphisms to act on.
    """
    P = _refine(P, refiners)

    # CLB topological-sort prune (Feulner §5.2 / Sage's _cut_by_known_automs,
    # call site #1 — top of every backtrack node, after refinement).
    if (
        state.use_clb
        and state.best_key is not None
        and state.clb.has_empty_intersection(P)
    ):
        state.clb_prune_fires += 1
        return

    # Absorb every singleton not yet in the key, in the order they appear
    # in the refined partition. Check the prefix prune after each
    # absorption to fail fast.
    for cell in P:
        if len(cell) == 1:
            c = cell[0]
            if not (partial.absorbed_cols >> c) & 1:
                partial.absorb(c)
                bk = state.best_key
                if bk is not None:
                    d = len(partial.key)
                    if d <= len(bk):
                        for i in range(d):
                            a = partial.key[i]
                            b = bk[i]
                            if a < b:
                                break  # winning; full key will be < best
                            if a > b:
                                state.prune_fires += 1
                                return

    # Note: Sage's `_cut_by_known_automs` runs *twice* per node — once at the
    # top of `_backtrack`, once inside `_one_refinement` after each refinement
    # step changes the partition. Our `_refine()` is a single-shot
    # equitable-refinement call, so `P` is unchanged between the top-of-node
    # check above and here. A second check on the same partition would never
    # fire, so we skip it.

    if all(len(cell) == 1 for cell in P):
        state.leaves_visited += 1
        pi_list = [0] * state.n
        for new_pos, cell in enumerate(P):
            pi_list[cell[0]] = new_pos
        pi: Perm = tuple(pi_list)
        key_tuple = tuple(partial.key)

        prior = state.key_to_pi.get(key_tuple)
        if prior is not None:
            # Two leaves with the same canonical key ⇒ pi · prior^-1 ∈ Aut.
            state.push_aut(compose(inverse(prior), pi))
        else:
            state.key_to_pi[key_tuple] = pi
            if state.best_key is None or key_tuple < state.best_key:
                state.best_key = key_tuple
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
            partial.copy(),
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


def canon_info_feulner_with_counters(
    C: Code, *, use_clb: bool = True
) -> tuple[CanonInfo, dict[str, int]]:
    """Wrapper around :func:`canon_info_feulner` that also returns the
    Phase-A diagnostic counters (leaves visited, prefix-prune fires, CLB
    topological-sort prune fires).

    Used by the Golay regression script and any other A/B harness that
    wants to confirm CLB is actually pruning. Not on the hot path —
    pure-Python only; the Rust kernel exposes its own counters.
    """
    n = C.n
    rref, _ = C.rref_basis()
    k = len(rref)
    if n == 0:
        return CanonInfo((), (), 1, ()), {
            "leaves_visited": 0,
            "prune_fires": 0,
            "clb_prune_fires": 0,
        }
    if k == 0 or k == n:
        return _sn_canon_info(n), {
            "leaves_visited": 0,
            "prune_fires": 0,
            "clb_prune_fires": 0,
        }

    refiners = _invariant_refiners(rref, k)
    state = _SearchState(n=n, k=k, use_clb=use_clb)
    partial = _PartialKey(k=k, work=list(rref))
    _search(
        _initial_partition(rref, n), rref, refiners, state, path=(), partial=partial
    )

    aut_gens = tuple(state.aut_gens)
    aut_order = group_order(aut_gens, n) if aut_gens else 1
    assert state.best_key is not None
    transporter = state.key_to_pi[state.best_key]
    column_orbits = _column_orbits(aut_gens, n)

    info = CanonInfo(
        canonical_column_order=transporter,
        aut_generators=aut_gens,
        aut_order=aut_order,
        column_orbits=column_orbits,
    )
    counters = {
        "leaves_visited": state.leaves_visited,
        "prune_fires": state.prune_fires,
        "clb_prune_fires": state.clb_prune_fires,
    }
    return info, counters


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
