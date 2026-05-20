"""Tests for :mod:`doubly_even.enumerate.seeds`."""

from __future__ import annotations

import math

import pytest

from doubly_even.canon.nauty import canon_info, canonical_form
from doubly_even.enumerate.augment import enumerate_doubly_even
from doubly_even.enumerate.seeds import (
    direct_sum_seeds,
    seed_mass,
)
from doubly_even.spec.doubly_even import is_doubly_even
from doubly_even.spec.mass import gaborit_sigma


def test_all_seeds_doubly_even():
    seeds = direct_sum_seeds(10)
    for k, code_seeds in seeds.items():
        for s in code_seeds:
            assert is_doubly_even(s.canonical), (
                f"Seed at k={k} fails double-evenness: {s.canonical.basis}"
            )
            assert s.canonical.rank == k


def test_all_seeds_distinct_canonical_forms():
    seeds = direct_sum_seeds(12)
    for k, code_seeds in seeds.items():
        canons = [s.canonical for s in code_seeds]
        assert len(canons) == len(set(canons)), (
            f"Duplicate canonical forms at k={k}"
        )


def test_seeded_codes_are_actually_present_in_enumeration():
    """Every seeded code must appear in `enumerate_doubly_even(N)`.

    Otherwise we'd be pre-crediting mass to non-existent equivalence
    classes, which would over-count and confuse mass-stop.
    """
    N = 10
    seeds = direct_sum_seeds(N)
    enum_canons = {canonical_form(ec.code) for ec in enumerate_doubly_even(N)}
    seed_canons = {s.canonical for ks in seeds.values() for s in ks}
    missing = seed_canons - enum_canons
    assert not missing, (
        f"Seeded codes not found by enumerate_doubly_even: {len(missing)}"
    )


def test_seed_mass_does_not_exceed_sigma():
    """Σ N!/|Aut| over seeds at rank k must not exceed σ(N, k).

    Mathematical sanity: seeds are a *subset* of the equivalence
    classes; their total mass is bounded by the closed-form total.
    """
    for N in [8, 10, 12]:
        seeds = direct_sum_seeds(N)
        factorial_N = math.factorial(N)
        mass = seed_mass(seeds, factorial_N)
        for k, m in mass.items():
            sigma = gaborit_sigma(N, k)
            assert m <= sigma, (
                f"N={N}, k={k}: seed mass {m} > σ(N, k) = {sigma}"
            )


def test_seed_aut_orders_match_canon_info():
    """SeededCode.aut_order must equal canon_info(canonical).aut_order."""
    seeds = direct_sum_seeds(10)
    for k, code_seeds in seeds.items():
        for s in code_seeds:
            ci = canon_info(s.canonical)
            assert s.aut_order == ci.aut_order, (
                f"Aut mismatch at k={k}: seed={s.aut_order}, canon_info={ci.aut_order}"
            )


@pytest.mark.parametrize(
    "N,expected_counts",
    [
        # From scripts/experimental/seed_mass_estimate.py — seed class counts per k.
        # These are determined by the constituent enumeration tables;
        # locking these down catches behaviour regressions in the
        # canonicaliser-or-dedup path.
        (8, {0: 1, 1: 1, 2: 2, 3: 1, 4: 0}),
        (10, {0: 1, 1: 2, 2: 2, 3: 3, 4: 1, 5: 0}),
        (12, {0: 1, 1: 2, 2: 4, 3: 5, 4: 5, 5: 1, 6: 0}),
    ],
)
def test_seed_counts_per_k_small(N, expected_counts):
    seeds = direct_sum_seeds(N)
    for k, expected in expected_counts.items():
        actual = len(seeds.get(k, []))
        assert actual == expected, (
            f"N={N}, k={k}: seed builder gave {actual}, expected {expected}"
        )
