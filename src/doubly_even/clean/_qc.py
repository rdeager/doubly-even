"""**Improvement #1 over DFGHILM Appendix B**: ``Q_C``-coordinate quotient enumeration.

Candidate extensions live in cosets of ``C`` in ``C^⊥``, parameterised by an
``L = N - 2k``-bit Q-coordinate via ``lift(u) := ⊕_i V_basis[i] · u_i``. At
``N = 22, k = 8`` this shrinks the search from ``2^22`` to ``2^6`` candidates.

Aut(C)'s column action descends to ``GL(L, F_2)`` on Q-coords. Per generator
we precompute ``σ_Q[i] = Q-coord of σ(V_basis[i]) mod C`` and a lookup table
``T_σ[u] = σ_Q · u`` (Gray-code built, one XOR per entry). Each orbit-BFS
step is then one list index. The wt-mod-4 filter runs *before* orbit-min:
Aut(C) preserves coset weight on doubly-even ``C``, so orbits don't split.
"""

from __future__ import annotations
from collections.abc import Iterable

from ._spec import Code, apply_perm


def Q_basis(C: Code):
    """``(V_basis, pivots_V)`` for ``V := C^⊥ ∩ {v : v_p = 0 ∀ p ∈ pivots(C)}``."""
    rref_rows, pivots = C.rref()
    reduced: list[int] = []
    for v in C.dual_basis():
        rep = v
        for row, p in zip(rref_rows, pivots):
            if (rep >> p) & 1:
                rep ^= row
        if rep != 0:
            reduced.append(rep)
    return Code(C.n, tuple(reduced)).rref()


def aut_image_on_Q(aut_generators, C, V_basis, pivots_V):
    """Per generator σ: an ``L``-tuple of ``L``-bit ints giving σ_Q's action on Q."""
    rref_rows, pivots = C.rref()
    rref_pivots = list(zip(rref_rows, pivots))
    out: list[tuple[int, ...]] = []
    for sigma in aut_generators:
        images: list[int] = []
        sigma_list = list(sigma)
        for b in V_basis:
            permuted = apply_perm(b, sigma_list)
            for row, p in rref_pivots:
                if (permuted >> p) & 1:
                    permuted ^= row
            coord = 0
            for k, p in enumerate(pivots_V):
                if (permuted >> p) & 1:
                    coord |= 1 << k
            images.append(coord)
        out.append(tuple(images))
    return out


def _sigma_Q_table(sigma_Q):
    """``T[u] = σ_Q · u`` for all ``u ∈ [0, 2^L)``, built via Gray code."""
    L = len(sigma_Q)
    size = 1 << L
    table = [0] * size
    if L == 0:
        return table
    val = 0
    u = 0
    for i in range(1, size):
        flip = (i & -i).bit_length() - 1
        u ^= 1 << flip
        val ^= sigma_Q[flip]
        table[u] = val
    return table


def _singular_reps_Q(V_basis):
    """Q-coords ``u ∈ [1, 2^L)`` whose lift has ``wt ≡ 0 (mod 4)``."""
    L = len(V_basis)
    out: list[int] = []
    if L == 0:
        return out
    u = 0
    v = 0
    for i in range(1, 1 << L):
        flip = (i & -i).bit_length() - 1
        u ^= 1 << flip
        v ^= V_basis[flip]
        if v.bit_count() & 3 == 0:
            out.append(u)
    return out


def _aut_orbit_minima_Q(reps_Q, sigma_Qs):
    """One Q-coord per ⟨σ_Qs⟩-orbit, taking the min in sorted iteration order."""
    if not sigma_Qs:
        return sorted(reps_Q)
    tables = [_sigma_Q_table(s) for s in sigma_Qs]
    seen: set[int] = set()
    minima: list[int] = []
    for v in sorted(reps_Q):
        if v in seen:
            continue
        minima.append(v)
        seen.add(v)
        queue = [v]
        while queue:
            next_q: list[int] = []
            for current in queue:
                for table in tables:
                    new_v = table[current]
                    if new_v not in seen:
                        seen.add(new_v)
                        next_q.append(new_v)
            queue = next_q
    return minima


def _lift(u_Q, V_basis):
    v = 0
    u = u_Q
    i = 0
    while u:
        if u & 1:
            v ^= V_basis[i]
        u >>= 1
        i += 1
    return v


def qc_candidates(C: Code, aut_generators: Iterable) -> list[int]:
    """Sorted ``F_2^N`` reps of Aut(C)-orbits of doubly-even 1-dim extensions of C."""
    V_basis, pivots_V = Q_basis(C)
    sigma_Qs = aut_image_on_Q(aut_generators, C, V_basis, pivots_V)
    reps_Q = _singular_reps_Q(V_basis)
    orbit_min = _aut_orbit_minima_Q(reps_Q, sigma_Qs)
    return sorted(_lift(u, V_basis) for u in orbit_min)
