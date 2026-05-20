"""Cross-check the Witt phase-(b) orbit-min against the F_2^N oracle.

The implementation in
:mod:`doubly_even.enumerate.experimental.quotient_witt` is dormant
scaffolding (phase (a) wins at every measured ``L`` in pure Python).
But the correctness of the orbit-min decomposition must be preserved
in case a future port re-enables it.
"""

from __future__ import annotations

from doubly_even.enumerate.augment import enumerate_doubly_even
from doubly_even.enumerate.experimental.quotient_witt import aut_orbit_minima_Q_witt
from doubly_even.enumerate.filters import reduce_mod_code, standard_form_coset_reps
from doubly_even.enumerate.quotient import Q_basis, aut_image_on_Q, lift
from doubly_even.spec.codes import Code
from doubly_even.spec.vectors import apply_permutation


CROSS_CHECK_NS = [4, 6, 8, 10, 12]


def _all_parents(N: int):
    for ec in enumerate_doubly_even(N):
        yield ec.code, ec.info.aut_generators


def _orbit_of_coset(start: int, gens: list[tuple[int, ...]], C: Code) -> set[int]:
    """BFS orbit of ``start`` (an F_2^N coset rep in V) under column-permutation
    by ``gens`` followed by ``reduce_mod_code``."""
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


def test_witt_orbit_min_partitions_match_oracle():
    """The phase-(b) ``aut_orbit_minima_Q_witt`` produces the same orbit
    partition over the F_2^N oracle orbits as phase (a)."""
    for N in CROSS_CHECK_NS:
        for C, gens in _all_parents(N):
            V_basis, pivots_V = Q_basis(C)
            sigma_Qs = aut_image_on_Q(gens, C, V_basis, pivots_V)
            L = len(V_basis)

            orbit_min = aut_orbit_minima_Q_witt(sigma_Qs, range(1, 1 << L), L)
            new_reps_lift = {lift(u, V_basis) for u in orbit_min}

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

            for orbit in oracle_orbits:
                hits = orbit & new_reps_lift
                assert len(hits) == 1, (
                    f"orbit at N={N}, C={list(C.basis)} (witt) has "
                    f"{len(hits)} reps in new pipeline (expected 1); "
                    f"orbit size={len(orbit)}"
                )

            union = set().union(*oracle_orbits) if oracle_orbits else set()
            assert new_reps_lift <= union, (
                "witt pipeline produced a rep outside any oracle orbit"
            )
