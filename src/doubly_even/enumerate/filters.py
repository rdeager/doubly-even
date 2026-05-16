"""Candidate-extension enumeration and pre-canonical filters.

To build the recursion ``C ↦ ⟨C, v⟩`` we need, for each doubly even code
``C`` of rank ``k``, the set of vectors ``v`` that produce a doubly even
``[N, k+1]`` extension. By the augmentation theorem these are exactly

    v ∈ C⊥,  wt(v) ≡ 0 (mod 4),  v ∉ C.

Adding any ``c ∈ C`` to such a ``v`` gives the same extended code (since
``⟨C, v + c⟩ = ⟨C, v⟩``), so the *meaningful* candidates are nonzero
cosets of ``C`` in ``C⊥``. For ``C`` doubly even the weight ``wt(v) mod 4``
is constant on each coset (polarisation plus ``v ⟂ C``), so the weight
filter applies at the coset level.

Furthermore, two cosets in the same ``Aut(C)``-orbit produce
permutation-equivalent extensions. Taking one representative per orbit is
the cheap pre-canonical filter.

What this module exposes:

* :func:`coset_reps_in_dual_mod_code` — one element per coset of ``C`` in
  ``C⊥``.
* :func:`weight_mod_four_zero` — keeps cosets whose representative has
  weight ``≡ 0 (mod 4)``.
* :func:`aut_orbit_minima` — among a set of candidate cosets, keeps only
  those that are the lex-min in their ``Aut(C)``-orbit.

The chain ``coset_reps → weight_mod_four_zero → aut_orbit_minima`` is the
candidate set that feeds the canonical-augmentation test in
:mod:`.augment`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..spec.codes import Code
from ..spec.vectors import apply_permutation, wt


def reduce_mod_code(v: int, C: Code) -> int:
    """Return the canonical representative of the coset ``v + C`` in ``C⊥``.

    Implementation: subtract (XOR) any RREF basis row of ``C`` whose pivot
    column is set in ``v``. The result has bit ``c`` cleared at every pivot
    column ``c``, making cosets uniquely keyed by their value at non-pivot
    columns.
    """
    rref, pivots = C.rref_basis()
    rep = v
    for row, c in zip(rref, pivots):
        if (rep >> c) & 1:
            rep ^= row
    return rep


def coset_reps_in_dual_mod_code(C: Code) -> Iterator[int]:
    """Yield each coset of ``C`` in ``C⊥`` exactly once, by canonical rep.

    The zero coset (``v ∈ C``) is included and is the first element when
    iteration order matters; callers that don't want it should skip ``0``.
    """
    dual = C.dual()
    seen: set[int] = set()
    for v in dual.codewords():
        rep = reduce_mod_code(v, C)
        if rep in seen:
            continue
        seen.add(rep)
        yield rep


def weight_mod_four_zero(reps: Iterable[int]) -> Iterator[int]:
    """Keep only ``v`` with ``wt(v) ≡ 0 (mod 4)``."""
    for v in reps:
        if v != 0 and wt(v) % 4 == 0:
            yield v


def aut_orbit_minima(
    reps: Iterable[int],
    aut_generators: Iterable[tuple[int, ...]],
    C: Code,
) -> Iterator[int]:
    """Keep only reps that are the lex-min of their ``Aut(C)``-orbit (mod ``C``).

    The action is column permutation followed by reduction mod ``C`` (so we
    compare *cosets*, not vectors). A rep ``v`` is kept iff BFS through its
    orbit never produces a coset rep strictly less than ``v`` itself.
    """
    aut_gens = [tuple(g) for g in aut_generators]
    if not aut_gens:
        # Trivial automorphism group: every rep is its own orbit min.
        yield from reps
        return

    for v in reps:
        if _is_orbit_min(v, aut_gens, C):
            yield v


def _is_orbit_min(
    v: int, aut_gens: list[tuple[int, ...]], C: Code
) -> bool:
    """Lex-min check for ``v`` under ``Aut(C)`` acting on cosets of ``C``."""
    seen = {v}
    queue: list[int] = [v]
    while queue:
        next_queue: list[int] = []
        for current in queue:
            for sigma in aut_gens:
                sigma_list = list(sigma)
                new_v = reduce_mod_code(apply_permutation(current, sigma_list), C)
                if new_v < v:
                    return False
                if new_v not in seen:
                    seen.add(new_v)
                    next_queue.append(new_v)
        queue = next_queue
    return True


def doubly_even_candidates(
    C: Code, aut_generators: Iterable[tuple[int, ...]]
) -> list[int]:
    """All ``Aut(C)``-orbit reps of doubly even 1-dim extensions of ``C``.

    Composes the three filters in order. Returns a sorted list so callers
    that don't care about determinism don't have to think about it.
    """
    reps = coset_reps_in_dual_mod_code(C)
    weight_filtered = weight_mod_four_zero(reps)
    orbit_minima = aut_orbit_minima(weight_filtered, aut_generators, C)
    return sorted(orbit_minima)
