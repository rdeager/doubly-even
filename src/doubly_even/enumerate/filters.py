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

Module layout
-------------

This module is in **two halves**:

1. **F_2^N reference oracle** (this docstring through
   :func:`aut_orbit_minima`). The historical implementation that
   builds canonical coset reps in the full ``F_2^N`` space — readable,
   spec-shaped, and preserved as the cross-check oracle for the
   ``Q_C``-pipeline tests. Not on the hot path.

2. **Hot-path dispatcher** (from :func:`doubly_even_candidates`
   downward). Routes to the Rust kernel's
   ``doubly_even_candidates_q`` when the extension module is loaded,
   otherwise delegates to
   :func:`doubly_even.enumerate.quotient.doubly_even_candidates_Q`
   (the σ_Q-coordinate pipeline).

The oracle exposes:

* :func:`coset_reps_in_dual_mod_code` — one element per coset of ``C`` in
  ``C⊥`` via the full-dual deduplication path.
* :func:`standard_form_coset_reps` — DFGHILM B.3 quotient-space
  enumeration: directly emits each of the ``2^(n-2k)`` coset reps.
* :func:`weight_mod_four_zero` — coset weight filter.
* :func:`aut_orbit_minima` — orbit-min representative selection.

The chain ``standard_form_coset_reps → weight_mod_four_zero →
aut_orbit_minima`` is the F_2^N candidate set used to validate the
quotient-space pipeline's output in the tests.
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

    Shared primitive: used by both the F_2^N reference oracle below and
    the cross-check tests against the Q_C-coordinate pipeline.
    """
    rref, pivots = C.rref_basis()
    rep = v
    for row, c in zip(rref, pivots):
        if (rep >> c) & 1:
            rep ^= row
    return rep


# ---------------------------------------------------------------------------
# F_2^N reference oracle (cross-check only; not on the hot path).
# ---------------------------------------------------------------------------


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


def standard_form_coset_reps(C: Code) -> Iterator[int]:
    """Yield each coset of ``C`` in ``C⊥`` exactly once, by canonical rep.

    Direct quotient-space construction (DFGHILM Appendix B.3): for doubly
    even ``C`` (so ``C ⊆ C⊥``) the unique canonical reps of ``C`` in
    ``C⊥`` are exactly the vectors of ``C⊥`` that are zero at every pivot
    column of ``C``. They form a subspace ``V`` of dimension ``n - 2k``,
    and this function enumerates its ``2^(n - 2k)`` elements via
    Gray-code XOR — strictly faster than
    :func:`coset_reps_in_dual_mod_code`, which iterates the full
    ``2^(n-k)`` dual and dedupes.

    Precondition: ``C ⊆ C⊥``. (Always satisfied along the doubly-even
    augmentation tree.) If ``C ⊄ C⊥`` the function still terminates but
    the enumerated set will not be the C-coset reps in C⊥ — use
    :func:`coset_reps_in_dual_mod_code` instead.

    The zero coset is yielded first (matching the convention of
    :func:`coset_reps_in_dual_mod_code`).
    """
    rref_rows, pivots = C.rref_basis()
    dual_basis = C.dual().basis

    # Reduce each dual basis vector mod ``C`` — clears pivot column bits.
    # The result lies in ``V = C⊥ ∩ {v : v_p = 0 ∀ p ∈ pivots(C)}``.
    reduced: list[int] = []
    for v in dual_basis:
        rep = v
        for row, p in zip(rref_rows, pivots):
            if (rep >> p) & 1:
                rep ^= row
        if rep != 0:
            reduced.append(rep)

    # Row-reduce to get a basis of V. The reduced set may be linearly
    # dependent: ``dual.basis`` has ``n - k`` vectors, but ``dim V = n - 2k``.
    V_basis, _ = Code(C.n, tuple(reduced)).rref_basis()
    L = len(V_basis)

    # Gray-code enumeration of the ``2^L`` elements of V. Each step toggles
    # the basis vector at the lowest set bit position of the step counter,
    # so we perform a single XOR per yielded value.
    yield 0
    if L == 0:
        return
    current = 0
    for i in range(1, 1 << L):
        flip = (i & -i).bit_length() - 1
        current ^= V_basis[flip]
        yield current


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
    # Hoist work that depends only on ``C`` or on the generators out of the
    # per-``v`` loop. Without this, profiling showed ``C.rref_basis()`` being
    # recomputed millions of times per ``aut_orbit_minima`` call from inside
    # the inner ``reduce_mod_code`` invocation.
    sigma_lists = [list(g) for g in aut_generators]
    if not sigma_lists:
        # Trivial automorphism group: every rep is its own orbit min.
        yield from reps
        return
    rref_rows, pivots = C.rref_basis()
    rref_pivots = list(zip(rref_rows, pivots))

    for v in reps:
        if _is_orbit_min(v, sigma_lists, rref_pivots):
            yield v


def _is_orbit_min(
    v: int,
    sigma_lists: list[list[int]],
    rref_pivots: list[tuple[int, int]],
) -> bool:
    """Lex-min check for ``v`` under a permutation group acting on cosets.

    ``sigma_lists`` is the list of generators of the acting group, each as
    a Python list (so we don't pay ``list(sigma)`` per inner-loop step).
    ``rref_pivots`` is ``list(zip(rref_rows, pivots))`` of the underlying
    code's RREF; reduction is inlined here to avoid per-call
    ``Code.rref_basis()`` work.
    """
    seen = {v}
    queue: list[int] = [v]
    while queue:
        next_queue: list[int] = []
        for current in queue:
            for sigma_list in sigma_lists:
                # Inlined apply_permutation(current, sigma_list).
                permuted = 0
                for i, j in enumerate(sigma_list):
                    if (current >> i) & 1:
                        permuted |= 1 << j
                # Inlined reduce_mod_code with hoisted RREF.
                new_v = permuted
                for row, p in rref_pivots:
                    if (new_v >> p) & 1:
                        new_v ^= row
                if new_v < v:
                    return False
                if new_v not in seen:
                    seen.add(new_v)
                    next_queue.append(new_v)
        queue = next_queue
    return True


# ---------------------------------------------------------------------------
# Hot path: kernel dispatcher.
# ---------------------------------------------------------------------------


try:  # pragma: no cover -- import-side switch
    import doubly_even_kernel as _kernel
except ImportError:  # pragma: no cover
    _kernel = None


def doubly_even_candidates(
    C: Code, aut_generators: Iterable[tuple[int, ...]]
) -> list[int]:
    """All ``Aut(C)``-orbit reps of doubly even 1-dim extensions of ``C``.

    Returns a sorted list so callers that don't care about determinism
    don't have to think about it.

    Dispatch:

    * If ``doubly_even_kernel`` is importable (Rust extension built via
      ``maturin develop``), the call marshals ``C`` and ``aut_generators``
      across the FFI and runs the whole `Q_C`-pipeline natively.
    * Otherwise falls back to
      :func:`doubly_even.enumerate.quotient.doubly_even_candidates_Q`.

    The Python path runs the orbit-min BFS in ``Q_C := C⊥/C`` coordinates
    (``L = n - 2k`` bits per element, no per-step reduce-mod-C). The
    return type and call convention match the previous ``F_2^N``-bit
    pipeline; ``F_2^N`` reps are produced by lifting orbit-min
    survivors via the ``V_basis`` of ``Q_C``.

    The previous pipeline (``standard_form_coset_reps →
    weight_mod_four_zero → aut_orbit_minima``) is preserved verbatim
    in this module as an oracle for cross-check tests.
    """
    if _kernel is not None:
        rref, pivots = C.rref_basis()
        dual_basis = C.dual().basis
        # PyO3 converts iterables (tuples included) into Vec<T> directly;
        # pass tuples through as-is to avoid the per-call list rebuild.
        return _kernel.doubly_even_candidates_q(
            C.n,
            rref,
            pivots,
            dual_basis,
            aut_generators,
        )

    from .quotient import doubly_even_candidates_Q

    return doubly_even_candidates_Q(C, aut_generators)
