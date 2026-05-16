"""Pure-Python Schreier–Sims on ``GL(L, F_2)``.

Matrices are stored in **column form**: ``M`` is a tuple of ``L`` ints
where ``M[j]`` is the ``j``-th column of the ``L × L`` matrix over
``F_2``. Bit ``i`` of ``M[j]`` is the entry at row ``i``, column ``j``.
The action on a column vector ``v ∈ F_2^L`` (also an ``L``-bit int) is::

    mat_apply(M, v) = XOR over i of M[i] when bit i of v is set
                    = sum_{i: v_i = 1} M[:, i]

This matches the convention produced by
:func:`doubly_even.enumerate.quotient.aut_image_on_Q` and consumed by
its callers — ``σ_Q[i]`` is "``σ`` applied to the ``i``-th basis vector",
i.e. the ``i``-th column.

Schreier–Sims structure mirrors :mod:`.permutations` one-to-one:
:func:`orbit_and_transversal_matrix` is the matrix analogue of
:func:`.permutations.orbit_and_transversal`, and
:func:`group_order_matrix` mirrors :func:`.permutations.group_order`
with base points the unit vectors ``1 << 0, …, 1 << (L-1)`` of
``F_2^L``. Those form a base for any subgroup of ``GL(L, F_2)`` since
fixing every standard basis vector forces the identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


Mat = tuple[int, ...]


def mat_identity(L: int) -> Mat:
    """Identity in column form: column ``j`` is the unit vector ``e_j``."""
    return tuple(1 << j for j in range(L))


def mat_apply(M: Mat, v: int) -> int:
    """Apply ``M`` to column vector ``v ∈ F_2^L``.

    Walks only the *set* bits of ``v`` via the ``v & -v`` trick, so cost
    is ``popcount(v)`` iterations — strictly cheaper than a shift loop
    for sparse vectors and equal in the dense limit.
    """
    out = 0
    while v:
        lsb = v & -v
        out ^= M[lsb.bit_length() - 1]
        v ^= lsb
    return out


def mat_mul(A: Mat, B: Mat) -> Mat:
    """Matrix product over ``F_2``: ``(A·B)[j] = A · B[j]``.

    Composition order matches ``compose(p, q)`` in :mod:`.permutations`:
    ``mat_mul(A, B)`` applies ``B`` first, then ``A``.
    """
    return tuple(mat_apply(A, B[j]) for j in range(len(B)))


def _transpose(M: Mat, L: int) -> list[int]:
    """Transpose column form → row form (or back; the map is involutive).

    Returns a list (mutable) so :func:`mat_inv` can do in-place row swaps.
    """
    out = [0] * L
    for j in range(L):
        col = M[j]
        i = 0
        while col:
            if col & 1:
                out[i] |= 1 << j
            col >>= 1
            i += 1
    return out


def mat_inv(A: Mat) -> Mat:
    """Inverse over ``F_2``. Raises ``ValueError`` if ``A`` is singular.

    Transposes to row form, runs Gauss–Jordan on ``[rows | I]``, transposes
    the right-hand block back to column form. The transpose helper is
    involutive so one routine handles both directions.

    Defensive: σ_Q values from :func:`aut_image_on_Q` are images of
    column permutations and so always lie in ``O(Q_C, q) ⊂ GL(L, F_2)``;
    a singular input signals an upstream bug, not user error.
    """
    L = len(A)
    rows = _transpose(A, L)
    inv = list(mat_identity(L))
    for j in range(L):
        pivot = -1
        for i in range(j, L):
            if (rows[i] >> j) & 1:
                pivot = i
                break
        if pivot < 0:
            raise ValueError(f"singular matrix at pivot column {j}")
        if pivot != j:
            rows[j], rows[pivot] = rows[pivot], rows[j]
            inv[j], inv[pivot] = inv[pivot], inv[j]
        for i in range(L):
            if i != j and (rows[i] >> j) & 1:
                rows[i] ^= rows[j]
                inv[i] ^= inv[j]
    return tuple(_transpose(tuple(inv), L))


def orbit_and_transversal_matrix(
    generators: Sequence[Mat], base: int, L: int
) -> tuple[list[int], dict[int, Mat]]:
    """BFS orbit of ``base ∈ F_2^L`` under ``⟨generators⟩ ⊆ GL(L, F_2)``.

    Returns ``(orbit, transversal)`` where ``transversal[p]`` is a matrix
    that maps ``base`` to ``p`` (identity at ``base``). Mirrors
    :func:`.permutations.orbit_and_transversal`.
    """
    orbit = [base]
    transversal: dict[int, Mat] = {base: mat_identity(L)}
    queue: list[int] = [base]
    while queue:
        next_queue: list[int] = []
        for p in queue:
            t_p = transversal[p]
            for g in generators:
                q = mat_apply(g, p)
                if q not in transversal:
                    transversal[q] = mat_mul(g, t_p)
                    orbit.append(q)
                    next_queue.append(q)
        queue = next_queue
    return orbit, transversal


def stabilizer_chain(
    generators: Iterable[Mat], L: int
) -> list[tuple[list[int], dict[int, Mat]]]:
    """Schreier–Sims chain with base ``e_0, e_1, …, e_{L-1}``.

    Returns one ``(orbit, transversal)`` entry per non-trivial level
    (entries with orbit size 1 are skipped from continuing the recursion
    but still appended for shape consistency). The product of orbit
    sizes equals ``|⟨generators⟩|``.

    Same structure as :func:`.permutations.group_order` lifted to the
    linear action.
    """
    gens: list[Mat] = [tuple(g) for g in generators]
    chain: list[tuple[list[int], dict[int, Mat]]] = []
    if not gens:
        return chain
    for i in range(L):
        base = 1 << i
        orbit, transversal = orbit_and_transversal_matrix(gens, base, L)
        chain.append((orbit, transversal))
        if len(orbit) == 1:
            continue
        identity = mat_identity(L)
        new_gens: set[Mat] = set()
        for p in orbit:
            t_p = transversal[p]
            for g in gens:
                q = mat_apply(g, p)
                t_q_inv = mat_inv(transversal[q])
                schreier_gen = mat_mul(t_q_inv, mat_mul(g, t_p))
                if schreier_gen != identity:
                    new_gens.add(schreier_gen)
        if not new_gens:
            break
        gens = list(new_gens)
    return chain


def group_order_matrix(generators: Iterable[Mat], L: int) -> int:
    """Exact ``|⟨generators⟩|`` for a subgroup of ``GL(L, F_2)``.

    Returns 1 for empty generator set. Same algorithm as
    :func:`.permutations.group_order`, adapted to the linear action via
    :func:`stabilizer_chain`.
    """
    order = 1
    for orbit, _ in stabilizer_chain(generators, L):
        order *= len(orbit)
    return order
