"""Canonical augmentation for doubly even codes.

Implements McKay's 1998 canonical-augmentation algorithm specialised to the
problem of doubly even binary linear codes (DFGHILM Appendix B.4).

Recursion shape::

    enumerate(N):
        traverse(zero_code(N))

    traverse(C):
        yield C
        for v in candidates(C):                # filters.doubly_even_candidates
            D = ⟨C, v⟩
            if not is_canonical_augmentation(C, D):
                continue
            traverse(D)

Where ``is_canonical_augmentation(C, D)`` is the McKay parent test: compute
the canonical *parent* ``p(D)`` of ``D`` by

1. applying the canonical column permutation of ``D`` (from nauty) to its
   basis;
2. taking the RREF of the result and dropping the last row (a fixed
   choice of canonical generator);
3. mapping back to the original column ordering.

Then ``(C, D)`` is canonical iff ``C`` lies in the ``Aut(D)``-orbit of
``p(D)``. Because both ``C`` and ``p(D)`` are ``(rank D − 1)``-dimensional
subcodes of ``D``, this orbit has at most ``2^{rank D} − 1`` elements; BFS
through the orbit is cheap.

The driver :func:`enumerate_doubly_even` yields one canonical
representative per permutation-equivalence class. The default ordering of
yields is the depth-first order induced by sorted candidate cosets — this
is stable across runs and useful for debugging.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache

from ..canon.nauty import CanonInfo, cached_canon_info
from ..spec.codes import Code, _compute_rref
from ..spec.mass import gaborit_sigma
from ..spec.vectors import apply_permutation
from .filters import doubly_even_candidates

try:  # pragma: no cover -- import-side switch
    import doubly_even_kernel as _kernel
except ImportError:  # pragma: no cover
    _kernel = None


# ----------------------------------------------------------- canonical parent


def canonical_parent(D: Code, info_D: CanonInfo | None = None) -> Code:
    """Return ``p(D)`` — the canonical parent of ``D`` in the McKay sense.

    Algorithm: apply ``D``'s canonical column permutation, RREF the result,
    drop the last RREF row, then undo the column permutation. The output
    is a ``(rank D − 1)``-dimensional subspace of ``F_2^N``.

    ``info_D`` may be passed if it has already been computed (it usually
    has, by the caller of :func:`is_canonical_augmentation`).
    """
    if D.rank == 0:
        raise ValueError("canonical_parent undefined for the zero code")
    if info_D is None:
        info_D = cached_canon_info(D)

    sigma = list(info_D.canonical_column_order)
    permuted_basis = tuple(apply_permutation(b, sigma) for b in D.basis)
    permuted = Code(D.n, permuted_basis)

    rref_rows, _ = permuted.rref_basis()
    if len(rref_rows) != D.rank:
        raise RuntimeError(
            "canonical_parent: RREF rank mismatch with code rank"
        )
    parent_in_canon = rref_rows[:-1]

    inv_sigma = [0] * D.n
    for i, j in enumerate(sigma):
        inv_sigma[j] = i
    parent_basis = tuple(
        apply_permutation(b, inv_sigma) for b in parent_in_canon
    )
    return Code(n=D.n, basis=parent_basis)


# --------------------------------------------------- canonical augmentation


def _subspace_key(C: Code) -> tuple[int, ...]:
    """Stable subspace identifier: the RREF basis as a tuple."""
    return C.rref_basis()[0]


@lru_cache(maxsize=None)
def _weight_enum_by_rref(rref: tuple[int, ...]) -> tuple[int, ...]:
    """Sorted codeword-weight tuple, keyed by RREF basis.

    Pure function of the subspace (the RREF basis is the canonical
    subspace identifier), so memoising here collapses repeated calls
    from sibling expansions of the parent-test BFS.
    """
    k = len(rref)
    weights: list[int] = []
    for mask in range(1 << k):
        w = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                w ^= rref[i]
            m >>= 1
            i += 1
        weights.append(w.bit_count())
    weights.sort()
    return tuple(weights)


def _weight_enum(C: Code) -> tuple[int, ...]:
    """Sorted tuple of codeword weights. ``Aut``-orbit invariant on subspaces.

    Used as a cheap necessary-condition prefilter in
    :func:`_in_aut_orbit_of_subspace`: two subspaces with different
    weight enumerators are guaranteed to lie in different orbits of any
    column-permutation group, so we can short-circuit the orbit BFS
    without doing any ``apply_permutation`` work.

    Delegates to the RREF-keyed cache :func:`_weight_enum_by_rref`.
    """
    return _weight_enum_by_rref(C.rref_basis()[0])


def weight_enum_cache_clear() -> None:
    """Drop every entry from the :func:`_weight_enum_by_rref` cache."""
    _weight_enum_by_rref.cache_clear()


def is_canonical_augmentation(
    C: Code, D: Code, info_D: CanonInfo | None = None
) -> bool:
    """Return True iff ``(C, D)`` is McKay-canonical.

    ``C`` must be a subcode of ``D`` of one less dimension; we do not
    verify that here, callers are responsible.
    """
    if info_D is None:
        info_D = cached_canon_info(D)
    p_D = canonical_parent(D, info_D)
    return _in_aut_orbit_of_subspace(C, p_D, info_D.aut_generators, D.n)


def _in_aut_orbit_of_subspace(
    C: Code,
    target: Code,
    aut_generators: Iterable[tuple[int, ...]],
    n: int,
) -> bool:
    """Test whether some element of ``⟨aut_generators⟩`` maps ``C`` to ``target``.

    BFS through the orbit of ``C``. Since both ``C`` and ``target`` are
    subspaces of fixed dimension in ``F_2^n``, the action is via the
    induced map on subspaces and the orbit has size at most ``2^{rank C}``.

    Two prefilters before the BFS:

    * ``_subspace_key`` equality short-circuits when ``C == target`` as
      subspaces (the common case at the root of the McKay test).
    * Sorted-codeword-weight tuples must match — every column-permutation
      preserves weights, so unequal weight enumerators guarantee that
      ``C`` and ``target`` are in different orbits.

    When the Rust kernel is available the BFS itself runs natively via
    :func:`doubly_even_kernel.subspace_in_orbit`; the Python body below
    is the fallback / diff oracle.
    """
    target_key = _subspace_key(target)
    start_key = _subspace_key(C)
    if start_key == target_key:
        return True
    if _weight_enum_by_rref(start_key) != _weight_enum_by_rref(target_key):
        return False
    sigma_lists = [list(g) for g in aut_generators]
    if not sigma_lists:
        return False

    if _kernel is not None:
        return _kernel.subspace_in_orbit(
            n, list(start_key), list(target_key), sigma_lists
        )

    seen = {start_key}
    # Queue holds RREF basis tuples (subspace identifiers). Going through
    # ``_compute_rref`` directly skips the ``Code`` wrapper churn and lets
    # repeated sibling expansions hit the module-level RREF cache.
    queue: list[tuple[int, ...]] = [start_key]
    while queue:
        next_queue: list[tuple[int, ...]] = []
        for current_basis in queue:
            for sigma_list in sigma_lists:
                new_basis = tuple(
                    apply_permutation(b, sigma_list) for b in current_basis
                )
                key, _ = _compute_rref(n, new_basis)
                if key == target_key:
                    return True
                if key in seen:
                    continue
                seen.add(key)
                next_queue.append(key)
        queue = next_queue
    return False


# ------------------------------------------------------------------ driver


@dataclass(frozen=True)
class EnumeratedCode:
    """One canonical representative produced by :func:`enumerate_doubly_even`.

    Carries the code itself plus the cached ``CanonInfo``, so callers who
    want the automorphism order or canonical column ordering don't have to
    recompute it.
    """

    code: Code
    info: CanonInfo

    @property
    def aut_order(self) -> int:
        return self.info.aut_order


def enumerate_doubly_even(N: int, max_k: int | None = None) -> Iterator[EnumeratedCode]:
    """Yield one canonical representative per equivalence class of doubly even
    binary codes ``[N, k]``, for ``k = 0, 1, …, max_k`` (default: until the
    augmentation tree dies, i.e. ``k = ⌊N / 2⌋`` for self-orthogonal codes).

    Yields in depth-first canonical-augmentation order.

    Uses the verified closed-form ``σ(N, k)`` from
    :func:`doubly_even.spec.mass.gaborit_sigma` as a mass-stopping
    shortcut: once the running tally ``Σ N!/|Aut(C_i)|`` over emitted
    rank-``k`` classes reaches ``σ(N, k)``, no further candidates at
    level ``k`` can be canonical augmentations, so the recursion at the
    parent level skips its remaining work. Correctness is preserved
    because the McKay parent test already guarantees one canonical
    representative per equivalence class — the shortcut just lets us
    return as soon as we know we've found them all.

    When the Rust kernel is loaded the entire canonical-augmentation
    recursion runs natively via :func:`doubly_even_kernel.enumerate_doubly_even`
    — no Python ↔ Rust boundary crossing inside the loop. The Python body
    below is the fallback / oracle.
    """
    cap = N // 2 if max_k is None else max_k

    if _kernel is not None:
        quota_vec = [gaborit_sigma(N, k) for k in range(cap + 1)]
        factorial_N = math.factorial(N)
        raw, _stats, _per_k = _kernel.enumerate_doubly_even(N, cap, quota_vec, factorial_N)
        for rref, ccol, gens, aord_str, orbits in raw:
            c = Code(N, tuple(rref))
            info = CanonInfo(
                canonical_column_order=tuple(ccol),
                aut_generators=tuple(tuple(g) for g in gens),
                aut_order=int(aord_str),
                column_orbits=tuple(orbits),
            )
            yield EnumeratedCode(code=c, info=info)
        return

    # Quotas per rank from Gaborit's closed form. ``mass_at_k`` accumulates
    # ``N! // aut_order`` over emitted classes at each rank.
    quota: dict[int, int] = {k: gaborit_sigma(N, k) for k in range(cap + 1)}
    mass_at_k: dict[int, int] = dict.fromkeys(range(cap + 1), 0)
    factorial_N = math.factorial(N)
    yield from _traverse(Code.zero(N), cap, quota, mass_at_k, factorial_N)


def _traverse(
    C: Code,
    max_k: int,
    quota: dict[int, int],
    mass_at_k: dict[int, int],
    factorial_N: int,
    info_C: CanonInfo | None = None,
) -> Iterator[EnumeratedCode]:
    if info_C is None:
        info_C = cached_canon_info(C)
    k = C.rank
    yield EnumeratedCode(code=C, info=info_C)
    mass_at_k[k] += factorial_N // info_C.aut_order
    if mass_at_k[k] > quota[k]:
        # Should be impossible: McKay parent test gives one canonical rep
        # per equivalence class, and Gaborit gives the exact labelled count.
        # Exceeding the quota means either a classification bug or a wrong
        # closed-form value — surface loudly rather than silently miscount.
        raise RuntimeError(
            f"level-{k} mass {mass_at_k[k]} exceeded σ(N={C.n}, k={k}) "
            f"= {quota[k]}; classification bug or σ formula off."
        )
    if k >= max_k:
        return
    if mass_at_k[k + 1] >= quota[k + 1]:
        return
    for v in doubly_even_candidates(C, info_C.aut_generators):
        if mass_at_k[k + 1] >= quota[k + 1]:
            return
        D = C.extend(v)
        # Compute D's canon info once and forward it: both the McKay test
        # and the recursive _traverse on D need it. Without threading it
        # through, the recursion's entry would re-fetch via the LRU cache.
        info_D = cached_canon_info(D)
        if not is_canonical_augmentation(C, D, info_D=info_D):
            continue
        yield from _traverse(
            D, max_k, quota, mass_at_k, factorial_N, info_C=info_D
        )


def enumerate_doubly_even_at(N: int, k: int) -> Iterator[EnumeratedCode]:
    """Yield only the dimension-``k`` slice of :func:`enumerate_doubly_even`."""
    for ec in enumerate_doubly_even(N, max_k=k):
        if ec.code.rank == k:
            yield ec
