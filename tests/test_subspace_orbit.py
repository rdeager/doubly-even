"""Differential tests for the Rust ``subspace_in_orbit`` BFS.

The Rust path replaces the inner BFS of
:func:`doubly_even.enumerate.augment._in_aut_orbit_of_subspace`. We compare
against the pure-Python fallback at small ``N`` over random code/generator
combinations to verify the boolean result agrees.
"""

from __future__ import annotations

import random

import pytest

from doubly_even.canon.nauty import cached_canon_info
from doubly_even.enumerate.augment import (
    _compute_rref,
    _in_aut_orbit_of_subspace,
    _subspace_key,
)
from doubly_even.enumerate.augment import _kernel as kernel
from doubly_even.enumerate.augment import enumerate_doubly_even
from doubly_even.spec.codes import Code
from doubly_even.spec.vectors import apply_permutation


pytestmark = pytest.mark.skipif(
    kernel is None,
    reason="Rust kernel not built",
)


def _python_bfs(
    start_rref: tuple[int, ...],
    target_rref: tuple[int, ...],
    generators: list[list[int]],
    n: int,
) -> bool:
    """Pure-Python reference BFS, mirroring the Rust subspace_in_orbit."""
    if start_rref == target_rref:
        return True
    if not generators:
        return False
    seen = {start_rref}
    queue: list[tuple[int, ...]] = [start_rref]
    while queue:
        next_queue: list[tuple[int, ...]] = []
        for current in queue:
            for sigma in generators:
                new_basis = tuple(apply_permutation(b, sigma) for b in current)
                key, _ = _compute_rref(n, new_basis)
                if key == target_rref:
                    return True
                if key in seen:
                    continue
                seen.add(key)
                next_queue.append(key)
        queue = next_queue
    return False


@pytest.mark.parametrize("N", [6, 8, 10, 12, 14])
def test_kernel_matches_python_on_canonical_pairs(N: int) -> None:
    """Across every canonical (C, D) pair in `enumerate_doubly_even(N)`,
    the Rust BFS must agree with the Python BFS on the McKay parent test.
    """
    rng = random.Random(0xDE_2026 + N)
    # Walk parents at level k≥1 and probe candidates' canonicality.
    parents = [ec for ec in enumerate_doubly_even(N) if ec.code.rank >= 1]
    rng.shuffle(parents)
    # Limit per-N sample size to keep the suite fast.
    parents = parents[:30]

    for ec in parents:
        D = ec.code
        # Re-derive via the direct canon API: under the default autom-only
        # labelling mode (2026-06-12) the enumeration record's
        # canonical_column_order is empty — only the direct `canon_info`
        # call guarantees the label `canonical_parent` consumes.
        info_D = cached_canon_info(D)
        # Use C := every rank-(k-1) subspace obtained by dropping a basis
        # row — at least one must be the canonical parent.
        rref_rows, _ = D.rref_basis()
        if len(rref_rows) == 0:
            continue
        for drop in range(len(rref_rows)):
            sub_basis = tuple(rref_rows[:drop] + rref_rows[drop + 1 :])
            if not sub_basis:
                continue
            C = Code(N, sub_basis)
            # Target: D's canonical parent p_D, as a rank-(k-1) subspace.
            from doubly_even.enumerate.augment import canonical_parent

            p_D = canonical_parent(D, info_D)
            start_key = _subspace_key(C)
            target_key = _subspace_key(p_D)
            gens = [list(g) for g in info_D.aut_generators]

            py_result = _python_bfs(start_key, target_key, gens, N)
            rs_result = kernel.subspace_in_orbit(
                N, list(start_key), list(target_key), gens
            )
            assert py_result == rs_result, (
                f"N={N}, C/D mismatch: python={py_result}, rust={rs_result}, "
                f"D.basis={list(D.basis)}, drop={drop}"
            )


@pytest.mark.parametrize("seed", range(20))
def test_kernel_matches_python_on_random_inputs(seed: int) -> None:
    """Random codes + random aut-generator subsets; the Rust path must
    agree with the Python BFS bool-for-bool."""
    rng = random.Random(0xC0DE_2026 + seed)
    N = rng.randint(6, 12)
    # Build a random doubly-even-ish code via enumerate up to a small k.
    pool = list(enumerate_doubly_even(N))
    rng.shuffle(pool)
    pool = [ec for ec in pool if 1 <= ec.code.rank <= 3]
    if not pool:
        pytest.skip(f"no rank 1-3 codes at N={N}")
    ec = pool[0]
    C = ec.code
    info_C = cached_canon_info(C)

    # Target = some permuted version of C (so it's in the orbit).
    sigma = list(range(N))
    rng.shuffle(sigma)
    permuted_basis = tuple(apply_permutation(b, sigma) for b in C.basis)
    target = Code(N, permuted_basis)

    start_key = _subspace_key(C)
    target_key = _subspace_key(target)
    gens = [list(g) for g in info_C.aut_generators]

    py_result = _python_bfs(start_key, target_key, gens, N)
    rs_result = kernel.subspace_in_orbit(
        N, list(start_key), list(target_key), gens
    )
    assert py_result == rs_result, (
        f"seed={seed}, N={N}, mismatch: python={py_result}, rust={rs_result}"
    )
