"""Pure-Python permutation helpers and a small Schreier–Sims.

Permutations here are tuples of length ``n`` with the convention
``p[i] = j`` meaning "point ``i`` is mapped to point ``j``". This matches
:func:`doubly_even.spec.vectors.apply_permutation` and the column-action
convention used by :mod:`.nauty`.

Composition follows ``(p ∘ q)[i] = p[q[i]]``.

We provide :func:`group_order` to compute the order of a permutation group
from a set of generators *exactly* (as a Python ``int``). pynauty returns
the order as a double-precision float, which loses accuracy beyond about
``2^53 ≈ 10^16``. Since we will assert ``Σ N!/|Aut| == σ(N, k)`` at sizes
where ``N!`` is much larger than that, every ``|Aut|`` that enters the
asserted sum must be exact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


Perm = tuple[int, ...]


def identity(n: int) -> Perm:
    return tuple(range(n))


def compose(p: Perm, q: Perm) -> Perm:
    """``(p ∘ q)[i] = p[q[i]]`` — apply ``q`` first, then ``p``."""
    return tuple(p[q[i]] for i in range(len(p)))


def inverse(p: Perm) -> Perm:
    n = len(p)
    inv = [0] * n
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def orbit_and_transversal(
    generators: Sequence[Perm], base: int, n: int
) -> tuple[list[int], dict[int, Perm]]:
    """Compute the orbit of ``base`` under ``generators`` and a transversal.

    Returns ``(orbit, transversal)`` where:

    * ``orbit`` is the list of points reachable from ``base``;
    * ``transversal[p]`` is a permutation taking ``base`` to ``p`` (the
      identity at ``base`` itself).

    BFS with the generators; Schreier–Sims later uses the transversal to
    build the stabiliser of ``base``.
    """
    orbit = [base]
    transversal: dict[int, Perm] = {base: identity(n)}
    queue: list[int] = [base]
    while queue:
        next_queue: list[int] = []
        for p in queue:
            for g in generators:
                q = g[p]
                if q not in transversal:
                    transversal[q] = compose(g, transversal[p])
                    orbit.append(q)
                    next_queue.append(q)
        queue = next_queue
    return orbit, transversal


def group_order(generators: Iterable[Perm], n: int) -> int:
    """Exact order of the permutation group ``⟨generators⟩`` on ``n`` points.

    Implements a textbook Schreier–Sims using points ``0, 1, …, n-1`` as
    base in that order. For each base point we compute the orbit under the
    current generators (this gives one factor of the order), then produce
    Schreier generators for the stabiliser of that point and pass them to
    the next level. No filtering / sifting is done, so the generator set
    can grow geometrically; for the sizes we use (``n ≤ 32``) this is fine.

    Returns ``1`` if ``generators`` is empty (trivial group).
    """
    gens: list[Perm] = [tuple(g) for g in generators]
    if not gens:
        return 1
    order = 1
    for base in range(n):
        orbit, transversal = orbit_and_transversal(gens, base, n)
        order *= len(orbit)
        if len(orbit) == 1:
            # Every generator already fixes ``base``; same stabiliser.
            continue
        # Schreier generators for the stabiliser of ``base``:
        # for each p in orbit and each g in gens, form t[g(p)]^{-1} ∘ g ∘ t[p].
        new_gens: set[Perm] = set()
        id_perm = identity(n)
        for p in orbit:
            t_p = transversal[p]
            for g in gens:
                q = g[p]
                t_q_inv = inverse(transversal[q])
                schreier_gen = compose(t_q_inv, compose(g, t_p))
                if schreier_gen != id_perm:
                    new_gens.add(schreier_gen)
        if not new_gens:
            break
        gens = list(new_gens)
    return order
