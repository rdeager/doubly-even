"""Bipartite-graph encoding of a binary linear code.

To compute ``Aut(C)`` and a canonical form for ``C``, we encode the code as a
bipartite graph ``G(C)``:

* **Left vertices** are the ``2^k`` codewords of ``C``.
* **Right vertices** are the ``N`` column indices.
* There is an edge between codeword ``w`` and column ``j`` iff ``w_j = 1``.

The column-side stabiliser of ``Aut(G(C))`` is canonically isomorphic to the
permutation-automorphism group of the code. We enforce the bipartite
structure by passing a two-block ``vertex_coloring`` to pynauty, so the
returned automorphisms necessarily map left vertices to left vertices and
right vertices to right vertices.

The left action is determined by the right action — if a column permutation
``σ`` is in ``Aut(C)`` then the induced action on codewords is
``w ↦ w ∘ σ^{-1}`` — so projecting onto the right-vertex action loses no
information, and the size of the bipartite group equals the size of
``Aut(C)``.

See ``/workspace/markdown/algorithm/05-automorphism-group.md`` for the math.
"""

from __future__ import annotations

from dataclasses import dataclass

import pynauty

from ..spec.codes import Code


@dataclass(frozen=True)
class BipartiteEncoding:
    """The graph ``G(C)`` plus the index ranges used to encode it.

    Vertices ``[0, L)`` are codewords (left side); vertices ``[L, L + R)`` are
    column indices (right side). ``codewords`` lists the actual codeword
    values in the order they were assigned to left-vertex indices, so
    callers can map a vertex back to a codeword if they need to. ``R == n``
    always; we store it for symmetry.
    """

    graph: pynauty.Graph
    codewords: tuple[int, ...]
    L: int
    R: int

    @property
    def total_vertices(self) -> int:
        return self.L + self.R

    def left_vertex(self, codeword_index: int) -> int:
        return codeword_index

    def right_vertex(self, column: int) -> int:
        return self.L + column

    def is_left(self, vertex: int) -> bool:
        return vertex < self.L

    def column_of(self, vertex: int) -> int:
        if vertex < self.L:
            raise ValueError(f"vertex {vertex} is on the left side")
        return vertex - self.L


def bipartite_graph(C: Code) -> BipartiteEncoding:
    """Build ``G(C)``, the bipartite codeword/column graph of ``C``."""
    codewords = tuple(C.codewords())  # length 2^rank, all distinct
    L = len(codewords)
    R = C.n
    total = L + R

    # Two-block colouring forces automorphisms to respect the bipartition.
    vertex_coloring = [set(range(L)), set(range(L, total))]
    g = pynauty.Graph(total, vertex_coloring=vertex_coloring)

    for i, w in enumerate(codewords):
        # Neighbours on the right side: column j iff bit j of w is set.
        neighbours = []
        for j in range(R):
            if (w >> j) & 1:
                neighbours.append(L + j)
        if neighbours:
            g.connect_vertex(i, neighbours)

    return BipartiteEncoding(graph=g, codewords=codewords, L=L, R=R)
