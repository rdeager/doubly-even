"""Leon §10(i)-style paired-refinement iso test between two codes.

Phase 2 of the cheap-equivalence-verifier plan
(`/home/dev/.claude/plans/let-s-implement-the-plan-nifty-kahn.md`).
Built on the existing Feulner primitives in :mod:`.feulner` so the
algorithmic primitives stay in one place.

Contract: ``paired_iso(d_rref, cf_rref, n)`` returns ``True`` iff D and
cf are S_n-equivalent as binary linear codes. The two RREFs are assumed
already row-reduced; if a caller passes a non-RREF basis, the iso test
is still correct but unnecessarily expensive on the D side (D's basis
is fed straight into ``_PartialKey``).

The function is **deliberately written for readability**, not speed.
This is the prototype that lets us measure operation counts vs nauty's
per-call cost; if the projection to Rust looks favourable, a Rust port
follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .feulner import (
    _column_orbits,
    _PartialKey,
    _individualise,
    _initial_partition,
    _invariant_refiners,
    _refine,
)
from .nauty import CanonInfo
from .permutations import Perm, compose, inverse


@dataclass
class IsoCounters:
    """Diagnostic counters carried through the paired search.

    Used by ``scripts/bench_paired_iso_vs_nauty.py`` to extrapolate
    Rust-port wall cost from Python operation counts.
    """

    refines: int = 0
    branches: int = 0
    leaves: int = 0
    prune_prefix: int = 0
    prune_shape: int = 0
    prune_weight_strata: int = 0


def _weight_multiset(rref: tuple[int, ...], k: int) -> tuple[int, ...]:
    """Sorted weights of all 2^k codewords spanned by ``rref``."""
    weights: list[int] = [0]
    w = 0
    for mask in range(1, 1 << k):
        flip = (mask & -mask).bit_length() - 1
        w ^= rref[flip]
        weights.append(w.bit_count())
    weights.sort()
    return tuple(weights)


def paired_iso(
    d_rref: tuple[int, ...],
    cf_rref: tuple[int, ...],
    n: int,
    counters: IsoCounters | None = None,
) -> bool:
    """Return ``True`` iff D and cf are permutation-equivalent.

    ``d_rref`` and ``cf_rref`` must each be the RREF basis of an ``[N, k]``
    code (same ``k``). If they have different rank, returns ``False``
    immediately.

    ``counters`` (optional) is mutated in place if provided; otherwise a
    fresh counter is used internally. The counters are diagnostic only —
    they don't affect the answer.
    """
    return paired_iso_with_witness(d_rref, cf_rref, n, counters=counters) is not None


def paired_iso_with_witness(
    d_rref: tuple[int, ...],
    cf_rref: tuple[int, ...],
    n: int,
    counters: IsoCounters | None = None,
) -> Optional[Perm]:
    """Return a witness permutation π if D and cf are equivalent, else ``None``.

    Convention: ``π[i] = j`` means D-column ``i`` is sent to cf-column ``j``
    (same left-action convention as :func:`spec.vectors.apply_permutation`).
    Applying π to D's RREF and re-row-reducing produces cf's row-span, i.e.
    ``Code(n, [apply_permutation(b, π) for b in d_rref]).rref_basis()[0] ==
    cf_rref``.

    The witness is one of possibly many — any element of the coset
    ``π · Aut(cf)`` is also a witness. Callers that need a canonical
    witness must canonicalise separately.
    """
    if counters is None:
        counters = IsoCounters()
    if len(d_rref) != len(cf_rref):
        return None
    k = len(d_rref)
    if n == 0:
        return ()
    if k == 0:
        # Zero codes on both sides — every permutation is a witness; pick id.
        return tuple(range(n))
    if k == n:
        # Whole space on both sides — every permutation is a witness; pick id.
        return tuple(range(n))

    # Cheap invariant reject: same weight multiset is necessary for iso.
    if _weight_multiset(d_rref, k) != _weight_multiset(cf_rref, k):
        counters.prune_weight_strata += 1
        return None

    refiners_d = _invariant_refiners(d_rref, k)
    refiners_cf = _invariant_refiners(cf_rref, k)
    # Refiner counts should match if the weight multisets match — the
    # _invariant_refiners contract picks the two lowest non-zero weight
    # strata, identical on both sides under iso.
    if len(refiners_d) != len(refiners_cf):
        counters.prune_weight_strata += 1
        return None

    P_d = _initial_partition(d_rref, n)
    P_cf = _initial_partition(cf_rref, n)
    partial_d = _PartialKey(k=k, work=list(d_rref))
    partial_cf = _PartialKey(k=k, work=list(cf_rref))

    return _paired_search(
        P_d, P_cf, refiners_d, refiners_cf, partial_d, partial_cf, counters, n
    )


def _paired_search(
    P_d: list[list[int]],
    P_cf: list[list[int]],
    refiners_d: list[int],
    refiners_cf: list[int],
    partial_d: _PartialKey,
    partial_cf: _PartialKey,
    counters: IsoCounters,
    n: int,
) -> Optional[Perm]:
    """One node of the paired Leon §10(i) search.

    Both partitions are refined under their own refiners. If the cell
    *shapes* (lengths) disagree at any point, the codes are non-iso on
    this branch. Singletons are absorbed in lockstep into their
    respective ``_PartialKey``s; the prefix prune (Leon Prop 8(i)–(iii))
    fires the moment the two partial keys diverge.

    Target-cell choice: the first non-singleton cell on the D side.
    Anchor: the lex-smallest column on the D side. Branching variable:
    the column on the cf side that the anchor maps to.

    Returns ``Some(π)`` at the first matching discrete leaf, where π is
    the witness permutation D-col → cf-col. Returns ``None`` if every
    branch fails.
    """
    counters.refines += 1
    P_d = _refine(P_d, refiners_d)
    P_cf = _refine(P_cf, refiners_cf)

    # Cell-shape check: lengths must match positionally for iso to be
    # possible. The _refine output is ordered by (lineage, signature,
    # min-col); under iso the signatures and lineages match, so equal
    # shapes are necessary at every level.
    if len(P_d) != len(P_cf):
        counters.prune_shape += 1
        return None
    for cd, cc in zip(P_d, P_cf):
        if len(cd) != len(cc):
            counters.prune_shape += 1
            return None

    # Absorb all singletons in lockstep, comparing partial keys after
    # each pair of absorbs.
    for cd, cc in zip(P_d, P_cf):
        if len(cd) == 1:
            col_d = cd[0]
            col_cf = cc[0]
            new_d = not (partial_d.absorbed_cols >> col_d) & 1
            new_cf = not (partial_cf.absorbed_cols >> col_cf) & 1
            if new_d:
                partial_d.absorb(col_d)
            if new_cf:
                partial_cf.absorb(col_cf)
            if new_d or new_cf:
                # Check whatever common prefix exists.
                d_key = partial_d.key
                cf_key = partial_cf.key
                common = min(len(d_key), len(cf_key))
                for i in range(common):
                    if d_key[i] != cf_key[i]:
                        counters.prune_prefix += 1
                        return None

    if all(len(c) == 1 for c in P_d):
        counters.leaves += 1
        # Both partitions discrete; iso iff full keys are equal.
        if partial_d.key != partial_cf.key:
            return None
        # Extract the witness: each D-col cell_d[0] at position i corresponds
        # to cf-col cell_cf[0]. π[d_col] = cf_col.
        pi = [0] * n
        for cell_d, cell_cf in zip(P_d, P_cf):
            pi[cell_d[0]] = cell_cf[0]
        return tuple(pi)

    cell_idx = next(i for i, c in enumerate(P_d) if len(c) > 1)
    cell_d = P_d[cell_idx]
    cell_cf = P_cf[cell_idx]

    # Anchor on the lex-smallest column of cell_d; branch over every
    # column of cell_cf for the cf-side image. This corresponds to
    # "where does the anchor column map to in cf?". If iso exists, some
    # col_cf will lead to a matching leaf.
    col_d = cell_d[0]
    for col_cf in cell_cf:
        counters.branches += 1
        new_P_d = _individualise(P_d, cell_idx, col_d)
        new_P_cf = _individualise(P_cf, cell_idx, col_cf)
        new_partial_d = partial_d.copy()
        new_partial_cf = partial_cf.copy()
        witness = _paired_search(
            new_P_d,
            new_P_cf,
            refiners_d,
            refiners_cf,
            new_partial_d,
            new_partial_cf,
            counters,
            n,
        )
        if witness is not None:
            return witness
    return None


def reconstruct_canon_info(
    cf_info: CanonInfo, pi: Perm, n: int
) -> CanonInfo:
    """Reconstruct ``CanonInfo`` for D from cached ``CanonInfo`` for cf + witness π.

    ``π`` is the witness from :func:`paired_iso_with_witness`: π[i] = j means
    "D-column i becomes cf-column j". Reconstruction formulas:

    * ``aut_order`` — invariant under permutation equivalence.
    * ``canonical_column_order`` — ``σ_d[i] = σ_cf[π[i]]`` (=
      ``compose(σ_cf, π)``). Goes D-col → cf-col → canonical pos.
    * ``aut_generators`` — each ``g ∈ Aut(cf)`` conjugates to
      ``π⁻¹ · g · π ∈ Aut(D)``.
    * ``column_orbits`` — recompute via union-find on the conjugated
      generators; orbit labels become the smallest column in D's frame
      naturally.
    """
    pi_inv = inverse(pi)
    sigma_d = compose(cf_info.canonical_column_order, pi)
    aut_d = tuple(
        compose(pi_inv, compose(g, pi)) for g in cf_info.aut_generators
    )
    orbits_d = _column_orbits(aut_d, n)
    return CanonInfo(
        canonical_column_order=sigma_d,
        aut_generators=aut_d,
        aut_order=cf_info.aut_order,
        column_orbits=orbits_d,
    )
