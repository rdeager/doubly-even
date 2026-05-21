"""**Improvement #2 over DFGHILM Appendix B**: low-weight-incidence canonicaliser.

Restrict nauty's bipartite graph to (low-weight codewords) × (columns) instead
of (all ``2^k`` codewords) × (columns). **Math invariant**: the column-side
stabiliser of the low-weight-incidence graph equals ``Aut(C)`` *iff the
included codewords span C*. So we walk weight strata in ascending order
(4, 8, 12, ...) and accumulate them until the running set spans C.

For most codes the weight-4 stratum alone spans (e.g. Reed–Muller-like at
``k ≤ 8``); extended Golay ``[24, 12, 8]`` jumps to weight 8 since it has no
weight-4 codewords. Bail to the full bipartite graph once the accumulated
set exceeds ``2^k / 2`` — at that point the full graph is cheaper.

Schreier–Sims gives exact ``|Aut|`` when pynauty's float overflows ``2^53``
(the zero code at ``N ≥ 19``).
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import pynauty

from ._spec import Code, rref_gf2


def perm_inverse(p):
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def perm_compose(p, q):
    """``(p ∘ q)[i] = p[q[i]]`` — apply ``q`` first, then ``p``."""
    return tuple(p[q[i]] for i in range(len(p)))


def group_order(generators, n: int) -> int:
    """Exact ``|⟨generators⟩|`` via textbook Schreier–Sims, base 0, 1, ..., n-1."""
    gens = [tuple(g) for g in generators]
    if not gens:
        return 1
    identity = tuple(range(n))
    order = 1
    for base in range(n):
        orbit = [base]
        transversal = {base: identity}
        queue = [base]
        while queue:
            p = queue.pop()
            for g in gens:
                q = g[p]
                if q not in transversal:
                    transversal[q] = perm_compose(g, transversal[p])
                    orbit.append(q)
                    queue.append(q)
        order *= len(orbit)
        if len(orbit) == 1:
            continue
        new_gens: set = set()
        for p in orbit:
            for g in gens:
                t_qi = perm_inverse(transversal[g[p]])
                s = perm_compose(t_qi, perm_compose(g, transversal[p]))
                if s != identity:
                    new_gens.add(s)
        if not new_gens:
            break
        gens = list(new_gens)
    return order


@dataclass(frozen=True)
class CanonInfo:
    canonical_column_order: tuple[int, ...]
    aut_generators: tuple[tuple[int, ...], ...]
    aut_order: int
    column_orbits: tuple[int, ...]

    @classmethod
    def trivial_sn(cls, n: int) -> "CanonInfo":
        """``Aut = S_n``; used for the rank-0 zero code (no left vertices in graph)."""
        ident = tuple(range(n))
        if n <= 1:
            return cls(ident, (), 1, ident)
        swap = (1, 0) + tuple(range(2, n))
        ncycle = tuple((i + 1) % n for i in range(n))
        return cls(ident, (swap, ncycle), math.factorial(n), (0,) * n)


_FLOAT_INT_LIMIT = 1 << 53


def _trustable_order(g1: float, g2: int):
    raw = g1 * (10**g2)
    return int(round(raw)) if raw < _FLOAT_INT_LIMIT else None


def canon_info(C: Code) -> CanonInfo:
    """Canonical form + Aut(C) via the low-weight spanning stratum (Improvement #2)."""
    if C.rank == 0:
        return CanonInfo.trivial_sn(C.n)
    k = C.rank
    bail = (1 << k) // 2
    accum: list[int] = []
    for w in range(4, C.n + 1, 4):
        stratum = C.codewords_of_weight(w)
        if not stratum:
            continue
        accum.extend(stratum)
        if len(accum) > bail:
            return _nauty_canon(C.n, C.codewords()[1:])  # fall back to full graph
        if len(rref_gf2(list(accum), C.n)[0]) == k:
            return _nauty_canon(C.n, accum)
    return _nauty_canon(C.n, C.codewords()[1:])


def _nauty_canon(n: int, codewords: list[int]) -> CanonInfo:
    L, R = len(codewords), n
    adjacency: dict[int, list[int]] = {}
    for i, w in enumerate(codewords):
        nbrs = [L + j for j in range(R) if (w >> j) & 1]
        if nbrs:
            adjacency[i] = nbrs
    # Two-cell coloring enforces bipartition; we skip degree sub-partitioning
    # (the D8 lever, +19 % in production) for clarity.
    g = pynauty.Graph(
        number_of_vertices=L + R, directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=[set(range(L)), set(range(L, L + R))],
    )
    gens, g1, g2, orbits, _ = pynauty.autgrp(g)
    aut_generators = tuple(_column_part(s, L, R) for s in gens)
    aut_order = _trustable_order(g1, g2)
    if aut_order is None:
        aut_order = group_order(list(aut_generators), R)
    # pynauty.canon_label returns π_inv: π_inv[new] = old. Invert on right side.
    canon_perm = pynauty.canon_label(g)
    new_to_old_col = [v - L for v in canon_perm if v >= L]
    old_to_new = [0] * R
    for new_col, old_col in enumerate(new_to_old_col):
        old_to_new[old_col] = new_col
    return CanonInfo(
        canonical_column_order=tuple(old_to_new),
        aut_generators=aut_generators,
        aut_order=aut_order,
        column_orbits=tuple(orbits[L:]),
    )


def _column_part(sigma, L: int, R: int) -> tuple[int, ...]:
    out = [0] * R
    for v in range(L, L + R):
        out[v - L] = sigma[v] - L
    return tuple(out)
