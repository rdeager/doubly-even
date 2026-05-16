"""Cross-check the Q_C-coordinate pipeline against the F_2^N oracle.

The reference pipeline in ``enumerate.filters`` (``standard_form_coset_reps``
→ ``weight_mod_four_zero`` → ``aut_orbit_minima``) is preserved and acts
as the oracle. ``enumerate.quotient`` runs the same logic in
``Q_C := C⊥/C`` coordinates; tests assert agreement across every
parent code emitted by the enumerator at small ``N``.
"""

from __future__ import annotations

import pytest

from doubly_even.canon.nauty import canon_info
from doubly_even.enumerate.augment import enumerate_doubly_even
from doubly_even.enumerate.filters import (
    aut_orbit_minima,
    reduce_mod_code,
    standard_form_coset_reps,
    weight_mod_four_zero,
)
from doubly_even.enumerate.quotient import (
    Q_basis,
    aut_image_on_Q,
    aut_orbit_minima_Q,
    doubly_even_candidates_Q,
    lift,
    project,
)
from doubly_even.spec.codes import Code
from doubly_even.spec.vectors import apply_permutation, wt


# Parents to cross-check against — every doubly even class at small N.
CROSS_CHECK_NS = [4, 6, 8, 10, 12]


def _all_parents(N: int):
    """Yield (code, aut_generators) for every doubly even class at length N."""
    for ec in enumerate_doubly_even(N):
        yield ec.code, ec.info.aut_generators


# ---------------------------------------------------------- Q_basis / lift / project


def test_Q_basis_rref_structure():
    """V_basis is RREF: each row has a 1 at its own pivot and 0 at others'.
    Every row is also zero at every pivot column of C."""
    for N in CROSS_CHECK_NS:
        for C, _ in _all_parents(N):
            V_basis, pivots_V = Q_basis(C)
            _, c_pivots = C.rref_basis()
            c_pivot_set = set(c_pivots)
            for i, b in enumerate(V_basis):
                # Own pivot is set.
                assert (b >> pivots_V[i]) & 1 == 1, (
                    f"V_basis[{i}] has no bit at its pivot {pivots_V[i]} "
                    f"(N={N}, C={list(C.basis)})"
                )
                # Other rows' pivots are clear.
                for j, p_j in enumerate(pivots_V):
                    if i != j:
                        assert (b >> p_j) & 1 == 0, (
                            f"V_basis[{i}] has a bit at V_basis[{j}]'s pivot "
                            f"{p_j} (N={N})"
                        )
                # No bits at C's pivot columns.
                for p in c_pivot_set:
                    assert (b >> p) & 1 == 0, (
                        f"V_basis[{i}] has a bit at C's pivot {p} (N={N})"
                    )


def test_lift_project_roundtrip():
    """For every u in [0, 2^L), project(lift(u, V_basis), pivots_V) == u."""
    for N in CROSS_CHECK_NS:
        for C, _ in _all_parents(N):
            V_basis, pivots_V = Q_basis(C)
            L = len(V_basis)
            # Cap exhaustive sweep to avoid combinatorial blow-ups; L stays
            # small (≤ N - 2k, and we only iterate up to k = N/2). At N=12
            # the worst case is L=12 (zero code) → 4096, still fine.
            for u in range(1 << L):
                v = lift(u, V_basis)
                assert project(v, pivots_V) == u, (
                    f"roundtrip failed at u={u}, N={N}, C={list(C.basis)}, "
                    f"V_basis={list(V_basis)}, pivots_V={pivots_V}"
                )


def test_lift_image_lies_in_V():
    """Every lift(u) must lie in C⊥ and be zero on C's pivot columns."""
    for N in CROSS_CHECK_NS:
        for C, _ in _all_parents(N):
            V_basis, _ = Q_basis(C)
            _, c_pivots = C.rref_basis()
            mask = sum(1 << p for p in c_pivots)
            L = len(V_basis)
            for u in range(1 << L):
                v = lift(u, V_basis)
                assert v & mask == 0, "lift has bits on C's pivots"
                assert C.is_orthogonal_to(v), "lift is not in C⊥"


# ---------------------------------------------- coset-rep set equals oracle


def test_singular_reps_set_equals_standard_form():
    """The set { lift(u) : u in 1..2^L-1, wt(lift(u)) % 4 == 0 } equals the
    oracle { v in standard_form_coset_reps(C) : v != 0, wt(v) % 4 == 0 }."""
    for N in CROSS_CHECK_NS:
        for C, _ in _all_parents(N):
            V_basis, _ = Q_basis(C)
            L = len(V_basis)
            new_set = {
                lift(u, V_basis)
                for u in range(1, 1 << L)
                if wt(lift(u, V_basis)) % 4 == 0
            }
            oracle = {
                v
                for v in standard_form_coset_reps(C)
                if v != 0 and wt(v) % 4 == 0
            }
            assert new_set == oracle, (
                f"singular set disagrees at N={N}, C={list(C.basis)}: "
                f"new \\ oracle = {sorted(new_set - oracle)}, "
                f"oracle \\ new = {sorted(oracle - new_set)}"
            )


def test_lift_set_equals_standard_form_full():
    """Lifts of all Q-coordinates (including zero, no weight filter) equal
    the full set of canonical coset reps from standard_form_coset_reps."""
    for N in CROSS_CHECK_NS:
        for C, _ in _all_parents(N):
            V_basis, _ = Q_basis(C)
            L = len(V_basis)
            new_set = {lift(u, V_basis) for u in range(1 << L)}
            oracle = set(standard_form_coset_reps(C))
            assert new_set == oracle, (
                f"full coset-rep set disagrees at N={N}, C={list(C.basis)}"
            )


# ----------------------------------------------- aut_image_on_Q correctness


def test_aut_image_on_Q_lifts_correctly():
    """For each generator σ and each Q-basis vector b_i,
    lift(sigma_Q[i]) == reduce_mod_code(apply_permutation(b_i, σ), C)."""
    for N in CROSS_CHECK_NS:
        for C, gens in _all_parents(N):
            V_basis, pivots_V = Q_basis(C)
            sigma_Qs = aut_image_on_Q(gens, C, V_basis, pivots_V)
            assert len(sigma_Qs) == len(gens)
            for sigma, sigma_Q in zip(gens, sigma_Qs):
                sigma_list = list(sigma)
                for i, b in enumerate(V_basis):
                    permuted = apply_permutation(b, sigma_list)
                    expected = reduce_mod_code(permuted, C)
                    actual = lift(sigma_Q[i], V_basis)
                    assert actual == expected, (
                        f"sigma_Q image wrong at N={N}, gen={sigma}, "
                        f"basis index {i}: expected {expected:#x}, got {actual:#x}"
                    )


def test_aut_image_on_Q_is_linear():
    """sigma_Q acts F_2-linearly on Q-coordinates: applying it to u (XOR of
    basis bits) equals XOR of applying it to each basis bit individually.

    Equivalently, lift commutes with the actions: lift(sigma_Q · u) reduced
    mod C equals reduce_mod_code(apply_permutation(lift(u), σ), C).
    """
    for N in CROSS_CHECK_NS:
        for C, gens in _all_parents(N):
            V_basis, pivots_V = Q_basis(C)
            sigma_Qs = aut_image_on_Q(gens, C, V_basis, pivots_V)
            L = len(V_basis)
            for sigma, sigma_Q in zip(gens, sigma_Qs):
                sigma_list = list(sigma)
                for u in range(1 << L):
                    # New: apply sigma_Q in Q-coords, then lift.
                    new_v = 0
                    tmp = u
                    bit_idx = 0
                    while tmp:
                        if tmp & 1:
                            new_v ^= sigma_Q[bit_idx]
                        tmp >>= 1
                        bit_idx += 1
                    new_lift = lift(new_v, V_basis)
                    # Reference: lift u, permute, reduce mod C.
                    ref = reduce_mod_code(
                        apply_permutation(lift(u, V_basis), sigma_list), C
                    )
                    assert new_lift == ref, (
                        f"linearity mismatch at N={N}, u={u}, gen={sigma}"
                    )


# ----------------------------------------------- orbit-min set agreement


def _orbit_of_coset(start: int, gens: list[tuple[int, ...]], C: Code) -> set[int]:
    """Compute the full orbit of ``start`` (an F_2^N coset rep in V) under
    column-permutation by ``gens`` followed by reduce_mod_code."""
    seen = {start}
    queue: list[int] = [start]
    while queue:
        next_queue: list[int] = []
        for current in queue:
            for sigma in gens:
                permuted = apply_permutation(current, list(sigma))
                reduced = reduce_mod_code(permuted, C)
                if reduced not in seen:
                    seen.add(reduced)
                    next_queue.append(reduced)
        queue = next_queue
    return seen


def test_orbit_min_Q_partitions_match_oracle():
    """The set of orbit-min reps in Q coords (lifted) partitions the cosets
    into the same orbits as the F_2^N pipeline."""
    for N in CROSS_CHECK_NS:
        for C, gens in _all_parents(N):
            V_basis, pivots_V = Q_basis(C)
            sigma_Qs = aut_image_on_Q(gens, C, V_basis, pivots_V)
            L = len(V_basis)

            # New: Q-coord orbit-min, lifted to F_2^N reps.
            new_reps_lift = {
                lift(u, V_basis)
                for u in aut_orbit_minima_Q(range(1, 1 << L), sigma_Qs)
            }

            # Oracle: collect every orbit in F_2^N space and check both
            # pipelines produce one rep per orbit, with the same orbit
            # partition. Compare orbits by canonical key = sorted tuple.
            all_reps = [v for v in standard_form_coset_reps(C) if v != 0]
            covered: set[int] = set()
            oracle_orbits: list[frozenset[int]] = []
            gens_list = list(gens)
            for v in all_reps:
                if v in covered:
                    continue
                orbit = _orbit_of_coset(v, gens_list, C)
                covered |= orbit
                oracle_orbits.append(frozenset(orbit))

            # Every orbit must have exactly one new-pipeline rep.
            for orbit in oracle_orbits:
                hits = orbit & new_reps_lift
                assert len(hits) == 1, (
                    f"orbit at N={N}, C={list(C.basis)} has {len(hits)} reps "
                    f"in new pipeline (expected 1); orbit size={len(orbit)}"
                )

            # And every new rep must lie in some orbit.
            union = set().union(*oracle_orbits) if oracle_orbits else set()
            assert new_reps_lift <= union, (
                f"new pipeline produced a rep outside any oracle orbit"
            )


# ----------------------------------------- end-to-end candidate-set parity


def test_doubly_even_candidates_orbit_equivalence_to_oracle():
    """The new pipeline's candidate list represents the same orbits as
    the oracle pipeline: identical orbit count, and each orbit gets
    exactly one rep from each pipeline. Reps within an orbit may differ
    (the two pipelines minimise over different total orders — Q-coord
    int vs F_2^N int — but McKay only cares about orbit identity)."""
    for N in CROSS_CHECK_NS:
        for C, gens in _all_parents(N):
            # Oracle: standard_form_coset_reps → weight_mod_four_zero →
            # aut_orbit_minima.
            oracle_pipeline = sorted(
                aut_orbit_minima(
                    weight_mod_four_zero(standard_form_coset_reps(C)), gens, C
                )
            )
            new_pipeline = doubly_even_candidates_Q(C, gens)

            # Same number of orbits.
            assert len(oracle_pipeline) == len(new_pipeline), (
                f"orbit count differs at N={N}, C={list(C.basis)}: "
                f"oracle={len(oracle_pipeline)}, new={len(new_pipeline)}"
            )

            # Group oracle reps by their orbit and check each orbit is
            # hit by exactly one new-pipeline rep.
            gens_list = list(gens)
            for v in oracle_pipeline:
                orbit = _orbit_of_coset(v, gens_list, C)
                hits = orbit & set(new_pipeline)
                assert len(hits) == 1, (
                    f"oracle orbit at N={N}, C={list(C.basis)} containing {v} "
                    f"has {len(hits)} new-pipeline reps (expected 1)"
                )


def test_doubly_even_candidates_empty_aut_is_passthrough():
    """With an empty generator set, every singular nonzero coset rep is its
    own orbit, so the pipeline returns every singular standard-form rep."""
    # Pick a small non-trivial doubly even code.
    C = Code(8, (0b00001111,))
    out = doubly_even_candidates_Q(C, [])
    expected = sorted(
        v for v in standard_form_coset_reps(C) if v != 0 and wt(v) % 4 == 0
    )
    assert out == expected
