"""Differential test: ``_refine_naive`` vs ``_refine_incremental``.

The naive form is the readability oracle from Phase B; the incremental
form is the worklist-driven nauty-style port that now drives the
search. They must produce the same *set-partition* of columns when fed
the same input — cell-order in the output may differ (the incremental
emit is by lineage + signature + min-col), but the equivalence
classes themselves must agree.
"""

from __future__ import annotations

import random
from collections import defaultdict

import pytest

from doubly_even.canon.experimental.feulner import (
    _initial_partition,
    _invariant_refiners,
    _refine_incremental,
    _refine_naive,
)
from doubly_even.spec.codes import Code


def _as_set_partition(P: list[list[int]]) -> frozenset[frozenset[int]]:
    return frozenset(frozenset(cell) for cell in P)


def _random_code(rng: random.Random, n: int, k: int) -> Code:
    """Random `[n, k]` binary code with full row rank."""
    while True:
        rows = tuple(rng.randrange(1, 1 << n) for _ in range(k))
        C = Code(n=n, basis=rows)
        if C.rank == k:
            return C


def _random_refinement_of(
    rng: random.Random, P: list[list[int]]
) -> list[list[int]]:
    """Random equitable-style refinement: split each cell at random."""
    out: list[list[int]] = []
    for cell in P:
        if len(cell) <= 1:
            out.append(list(cell))
            continue
        # With prob 0.5, split into 2-3 chunks at random.
        if rng.random() < 0.5:
            shuffled = list(cell)
            rng.shuffle(shuffled)
            n_chunks = rng.randint(2, min(3, len(shuffled)))
            chunks: list[list[int]] = [[] for _ in range(n_chunks)]
            for j in shuffled:
                chunks[rng.randrange(n_chunks)].append(j)
            for ch in chunks:
                if ch:
                    out.append(sorted(ch))
        else:
            out.append(list(cell))
    return out


@pytest.mark.parametrize("n", [6, 8, 10, 12, 14])
def test_random_codes_same_set_partition(n: int) -> None:
    rng = random.Random(0xFE0F17 ^ n)
    for trial in range(40):
        k = rng.randint(1, min(n - 1, n // 2 + 1))
        C = _random_code(rng, n, k)
        rref, _ = C.rref_basis()
        refiners = _invariant_refiners(rref, len(rref))
        P0 = _initial_partition(rref, n)
        # Two depths of random refinement to stress the worklist.
        for _depth in range(rng.randint(0, 2)):
            P0 = _random_refinement_of(rng, P0)

        naive = _refine_naive(P0, refiners)
        inc = _refine_incremental([list(c) for c in P0], refiners)

        # Same set-partition.
        assert _as_set_partition(naive) == _as_set_partition(inc), (
            f"n={n} k={k} basis={list(C.basis)}: "
            f"naive={naive}, incremental={inc}"
        )

        # Each output cell is internally sorted.
        for cell in inc:
            assert cell == sorted(cell)
            assert len(cell) > 0


def test_empty_refiners_returns_input_partition() -> None:
    """No refiners → cells emerge equitable trivially; only sort + lineage."""
    P = [[0, 1, 2, 3], [4, 5]]
    naive = _refine_naive(P, [])
    inc = _refine_incremental([list(c) for c in P], [])
    assert _as_set_partition(naive) == _as_set_partition(inc)


def test_singleton_only_partition() -> None:
    """All singletons: refinement is the identity."""
    P = [[0], [1], [2], [3]]
    refiners = [0b1010, 0b0101]
    naive = _refine_naive(P, refiners)
    inc = _refine_incremental([list(c) for c in P], refiners)
    assert _as_set_partition(naive) == _as_set_partition(inc)
    assert all(len(c) == 1 for c in inc)


def test_extended_hamming_8_4_partition_invariants() -> None:
    """Extended Hamming [8,4,4]: a fully transitive code — initial cell
    [0..7] should remain a single cell after refinement (the code's
    Aut group is 2-transitive on columns)."""
    rows = (0b00001111, 0b00110011, 0b01010101, 0b11111111)
    C = Code(n=8, basis=rows)
    rref, _ = C.rref_basis()
    refiners = _invariant_refiners(rref, len(rref))
    P0 = _initial_partition(rref, 8)
    naive = _refine_naive(P0, refiners)
    inc = _refine_incremental([list(c) for c in P0], refiners)
    assert _as_set_partition(naive) == _as_set_partition(inc)
    # Sanity: refinement doesn't break the single-cell start.
    assert len(inc) == 1
    assert sorted(inc[0]) == list(range(8))


def test_input_not_mutated() -> None:
    """Incremental must not mutate the caller's partition."""
    P = [[0, 1, 2, 3], [4, 5, 6, 7]]
    P_copy = [list(c) for c in P]
    _refine_incremental(P, [0b00001111, 0b00110011])
    assert P == P_copy


def test_signature_equitable_at_fixpoint() -> None:
    """Every column in a returned cell must have the same incidence
    signature against every refiner-group (cell-histogram type)."""
    rng = random.Random(7)
    for _ in range(20):
        n = rng.randint(6, 12)
        k = rng.randint(1, min(n - 1, n // 2 + 1))
        C = _random_code(rng, n, k)
        rref, _ = C.rref_basis()
        refiners = _invariant_refiners(rref, len(rref))
        P0 = _initial_partition(rref, n)
        out = _refine_incremental([list(c) for c in P0], refiners)

        # Group refiners by their cell-histogram type at fixpoint.
        cell_masks = [sum(1 << j for j in c) for c in out]
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for w in refiners:
            t = tuple(bin(w & m).count("1") for m in cell_masks)
            groups[t].append(w)
        ordered_types = sorted(groups.keys())

        for cell in out:
            sigs = set()
            for j in cell:
                sig = tuple(
                    sum(1 for w in groups[t] if (w >> j) & 1)
                    for t in ordered_types
                )
                sigs.add(sig)
            assert len(sigs) <= 1, (
                f"cell {cell} is not equitable: sigs={sigs}"
            )
