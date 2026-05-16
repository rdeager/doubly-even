"""Q_C-coordinate candidate pipeline (Milestone 4 phase (a)).

For a doubly even code ``C`` of length ``N`` and rank ``k`` we want the
``Aut(C)``-orbit reps of nonzero cosets ``v + C ⊆ C⊥`` with
``wt(v) ≡ 0 (mod 4)`` — the valid doubly even augmentations.

The reference pipeline in :mod:`.filters` does this in ``F_2^N``: it
enumerates the ``2^(n-2k)`` canonical coset reps via
``standard_form_coset_reps``, filters by ``wt mod 4``, then runs an
orbit-min BFS that per step *applies* a column permutation and then
*reduces mod C*.

This module does the same work, but with the cosets re-coordinatised as
``(n - 2k)``-bit integers in a fixed basis of
``V := C⊥ ∩ {v : v_p = 0 ∀ p ∈ pivots(C)}`` (the canonical-rep
subspace). The BFS body becomes one XOR per set bit of the current
element, with no reduction step — the quotient is built into the
coordinates. We then lift orbit-min survivors back to ``F_2^N`` and
apply the ``wt mod 4`` filter on the lift.

Math note. The plan doc framed this in terms of the quadratic form
``q(v + C) := wt(v)/2 mod 2`` and claimed its polarisation is zero
because ``⟨u, v⟩ ≡ 0 (mod 2)`` for all ``u, v ∈ C⊥``. The second
claim only holds when ``C`` is self-dual; for ``C ⊊ C⊥`` the
polarisation is generally nonzero, so a single ``q_vector & u`` bit
parity does not classify singular vectors. We therefore keep the
weight check at the lift step rather than encoding ``q`` in
``Q``-coordinates. Lifting is one XOR per set bit and runs only on
the orbit-min survivor list, so the cost is negligible.

The output type of :func:`doubly_even_candidates_Q` matches the
existing ``doubly_even_candidates``: a sorted ``list[int]`` of
``F_2^N`` vectors. ``filters.doubly_even_candidates`` delegates here.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..spec.codes import Code
from ..spec.vectors import wt


def Q_basis(C: Code) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(V_basis, pivots_V)`` for ``Q_C := C⊥/C``.

    ``V_basis``: tuple of ``L = n - 2k`` integers in ``F_2^N``, each in
    ``C⊥`` and zero on every pivot column of ``C``. The tuple is
    row-reduced; ``pivots_V[i]`` is the column of ``V_basis[i]``'s
    leading 1.

    Precondition: ``C ⊆ C⊥`` (always holds on the doubly even
    augmentation tree).
    """
    rref_rows, pivots = C.rref_basis()
    dual_basis = C.dual().basis

    # Reduce each dual basis vector mod C — clears the bits at C's
    # pivot columns, sending each dual generator to V.
    reduced: list[int] = []
    for v in dual_basis:
        rep = v
        for row, p in zip(rref_rows, pivots):
            if (rep >> p) & 1:
                rep ^= row
        if rep != 0:
            reduced.append(rep)

    # ``dual.basis`` has n - k generators; ``dim V = n - 2k``. Row-reduce
    # to recover a basis of V and its pivot columns (which live in
    # ``[0, N) \ pivots(C)``).
    V_basis, pivots_V = Code(C.n, tuple(reduced)).rref_basis()
    return V_basis, pivots_V


def lift(u_Q: int, V_basis: tuple[int, ...]) -> int:
    """Map a ``Q``-coordinate int back to its ``F_2^N`` rep.

    ``u_Q`` is an ``L``-bit integer with ``L = len(V_basis)``. The
    output is the XOR of ``V_basis[i]`` over set bits ``i`` of ``u_Q``.
    """
    v = 0
    u = u_Q
    i = 0
    while u:
        if u & 1:
            v ^= V_basis[i]
        u >>= 1
        i += 1
    return v


def project(v_in_V: int, pivots_V: tuple[int, ...]) -> int:
    """Map an ``F_2^N`` vector already in ``V`` to its ``Q``-coordinate.

    ``V_basis`` is in RREF, so bit ``i`` of the output is the bit of
    ``v_in_V`` at column ``pivots_V[i]``. Callers must ensure
    ``v_in_V ∈ V`` (i.e. is zero on ``C``'s pivot columns); the
    function does not check.
    """
    out = 0
    for i, p in enumerate(pivots_V):
        if (v_in_V >> p) & 1:
            out |= 1 << i
    return out


def aut_image_on_Q(
    aut_generators: Iterable[tuple[int, ...]],
    C: Code,
    V_basis: tuple[int, ...],
    pivots_V: tuple[int, ...],
) -> list[tuple[int, ...]]:
    """Image of each automorphism generator in ``End(Q_C)``.

    For each ``σ`` in ``aut_generators``, returns a length-``L`` tuple
    ``sigma_Q`` where ``sigma_Q[i]`` is the ``Q``-coordinate of
    ``σ(V_basis[i])`` reduced mod ``C``. Applying ``sigma_Q`` to a
    ``Q``-coordinate int ``u`` is then ``XOR sigma_Q[i] for each bit i
    set in u``.
    """
    rref_rows, pivots = C.rref_basis()
    rref_pivots = list(zip(rref_rows, pivots))

    out: list[tuple[int, ...]] = []
    for sigma in aut_generators:
        sigma_list = list(sigma)
        # Image of each Q-basis vector.
        images: list[int] = []
        for b in V_basis:
            # Apply column permutation σ.
            permuted = 0
            for i, j in enumerate(sigma_list):
                if (b >> i) & 1:
                    permuted |= 1 << j
            # Reduce mod C (zero out bits at C's pivot columns).
            for row, p in rref_pivots:
                if (permuted >> p) & 1:
                    permuted ^= row
            # Project to Q-coordinates by reading V's pivot bits.
            coord = 0
            for k, p in enumerate(pivots_V):
                if (permuted >> p) & 1:
                    coord |= 1 << k
            images.append(coord)
        out.append(tuple(images))
    return out


def _sigma_Q_table(sigma_Q: tuple[int, ...]) -> list[int]:
    """Precompute ``table[u] = σ_Q · u`` for every ``u ∈ [0, 2^L)``.

    Built in Gray-code order so each entry costs one XOR vs. the
    previous: total build cost is ``2^L`` XORs (vs. ``2^L · L`` for
    the naïve per-cell computation). The BFS then applies ``σ_Q`` via
    a single list index, eliminating the per-step Python bit-walk
    that previously dominated the profile.
    """
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


def aut_orbit_minima_Q(
    reps_Q: Iterable[int],
    sigma_Qs: list[tuple[int, ...]],
) -> list[int]:
    """One ``Q``-coordinate rep per ``⟨sigma_Qs⟩``-orbit on ``reps_Q``.

    Algorithm: decompose ``reps_Q`` into orbits in a single sweep. Each
    coset is visited exactly once across all orbits (vs. the naïve
    per-rep ``_is_orbit_min`` BFS which visits each orbit element up to
    ``orbit_size`` times). We iterate ``reps_Q`` in sorted order and
    BFS-expand the orbit of the first unseen element, yielding that
    element as the orbit min.

    Inner step: ``σ_Q · current`` is a single list index into a
    precomputed table (see :func:`_sigma_Q_table`). Memory cost is
    ``2^L`` ints per generator per call.

    Precondition for correctness of the "min in iteration order = orbit
    min" trick: ``reps_Q`` must be closed under the action (so each
    orbit is fully inside it). The caller satisfies this by passing
    only ``wt mod 4 = 0`` cosets, which is closed because ``Aut(C)``
    preserves ``wt mod 4`` on cosets of doubly even ``C``.

    With an empty generator set every rep is its own orbit — return a
    sorted list of the input.
    """
    if not sigma_Qs:
        return sorted(reps_Q)

    tables = [_sigma_Q_table(tuple(s)) for s in sigma_Qs]
    reps_sorted = sorted(reps_Q)
    seen: set[int] = set()
    minima: list[int] = []

    for v in reps_sorted:
        if v in seen:
            continue
        minima.append(v)
        seen.add(v)
        queue: list[int] = [v]
        while queue:
            next_queue: list[int] = []
            for current in queue:
                for table in tables:
                    new_v = table[current]
                    if new_v not in seen:
                        seen.add(new_v)
                        next_queue.append(new_v)
            queue = next_queue
    return minima


def singular_reps_Q(V_basis: tuple[int, ...]) -> list[int]:
    """Yield ``Q``-coordinate ints ``u`` whose lift has ``wt ≡ 0 (mod 4)``.

    Gray-code walk through ``[1, 2^L)`` maintaining ``(u, lift(u))``
    incrementally: each step toggles one ``Q``-bit and XORs the
    corresponding ``V_basis`` row into the lift. We keep ``u`` whenever
    the lift's popcount is divisible by 4 — this is exactly the singular
    coset condition.

    Returned as a list (the caller iterates it twice: once for orbit-min
    BFS, once via the lift step that sorts the survivors).
    """
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


def doubly_even_candidates_Q(
    C: Code, aut_generators: Iterable[tuple[int, ...]]
) -> list[int]:
    """``Aut(C)``-orbit reps of doubly even 1-dim extensions of ``C``.

    Drop-in replacement for
    :func:`doubly_even.enumerate.filters.doubly_even_candidates` —
    returns the same sorted ``list[int]`` of ``F_2^N`` representatives.

    Pipeline:

    1. Build ``V_basis, pivots_V = Q_basis(C)``.
    2. Image each ``Aut(C)`` generator into ``End(Q_C)``.
    3. Enumerate the singular ``Q``-coordinates (lift weight ≡ 0 mod 4)
       via a Gray-code walk that maintains the lift incrementally.
    4. Orbit-min filter in ``Q``.
    5. Lift each survivor back to ``F_2^N``.

    The weight filter must run *before* orbit-min (not after) so the
    orbit BFS only visits the singular cosets — otherwise we'd burn
    BFS time on the ~3/4 of cosets that fail ``wt mod 4`` and never
    yield an output. ``Aut(C)`` preserves ``wt mod 4`` on cosets
    (column permutation preserves weight; coset reduction by ``c ∈ C``
    doubly even shifts weight by ``wt(c) - 2⟨w, c⟩ ≡ 0 (mod 4)`` for
    ``w ∈ C⊥``), so every orbit lies entirely inside or entirely
    outside the singular set — pre-filtering does not split orbits.
    """
    V_basis, pivots_V = Q_basis(C)
    sigma_Qs = aut_image_on_Q(aut_generators, C, V_basis, pivots_V)
    reps_Q = singular_reps_Q(V_basis)
    orbit_min = aut_orbit_minima_Q(reps_Q, sigma_Qs)
    return sorted(lift(u, V_basis) for u in orbit_min)
