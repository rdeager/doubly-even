"""Correctness tests for ``doubly_even.canon.paired_iso``.

Phase 2 of the cheap-equivalence-verifier plan. The oracle is
:func:`doubly_even.canon.nauty.canon_info` — its canonical form decides
S_n-equivalence, and ``paired_iso`` must return the same verdict.

Strategy: enumerate every equivalence class at a small (N, k), compute
each class's canonical form, then for each random alternate-basis code
in that class:

* positive cell — ``paired_iso(d_rref, cf_rref)`` must be ``True``.
* negative cell — ``paired_iso(d_rref, cf_rref_of_different_class)``
  must be ``False`` whenever both share a weight enumerator (so the
  cheap weight-multiset reject doesn't shortcut to ``False``).

Small N keeps the suite fast (< 1 s).
"""

from __future__ import annotations

import random

import pytest

from doubly_even.canon.nauty import canon_info
from doubly_even.canon.paired_iso import IsoCounters, paired_iso
from doubly_even.enumerate.augment import enumerate_doubly_even_at
from doubly_even.spec.codes import Code
from doubly_even.spec.vectors import apply_permutation


def _canonical_form(rref: tuple[int, ...], n: int) -> tuple[int, ...]:
    """Apply ``canon_info`` to recover the canonical RREF (subspace identifier)."""
    C = Code(n, rref)
    info = canon_info(C)
    permuted = tuple(
        apply_permutation(b, list(info.canonical_column_order)) for b in C.basis
    )
    return Code(n, permuted).rref_basis()[0]


def _random_permute(rref: tuple[int, ...], n: int, rng: random.Random) -> tuple[int, ...]:
    """Apply a random column permutation and return the result's RREF."""
    perm = list(range(n))
    rng.shuffle(perm)
    permuted = tuple(apply_permutation(b, perm) for b in rref)
    return Code(n, permuted).rref_basis()[0]


# (N, k) cells over which to test. Small enough to keep total runtime <1 s.
_CELLS = [
    (6, 1),
    (8, 1), (8, 2), (8, 3), (8, 4),
    (10, 1), (10, 2), (10, 3),
    (12, 1), (12, 2), (12, 3), (12, 4),
    (14, 3), (14, 4),
]


@pytest.mark.parametrize(("N", "k"), _CELLS)
def test_paired_iso_positive(N: int, k: int) -> None:
    """For every emitted (N, k) class, paired_iso(d, cf) == True for
    several random permutations d of cf."""
    rng = random.Random((N << 8) | k)
    for ec in enumerate_doubly_even_at(N, k):
        cf = _canonical_form(ec.code.basis, N)
        # Test the class's own rref and 3 random permutations.
        candidates = [ec.code.basis] + [
            _random_permute(ec.code.basis, N, rng) for _ in range(3)
        ]
        for d_rref in candidates:
            assert paired_iso(d_rref, cf, N), (
                f"paired_iso missed a YES at N={N}, k={k}: "
                f"d={d_rref}, cf={cf}"
            )


@pytest.mark.parametrize(("N", "k"), _CELLS)
def test_paired_iso_negative(N: int, k: int) -> None:
    """For pairs of distinct classes that share a weight enumerator,
    paired_iso must return False — i.e. the iso test must reject
    non-equivalent codes that aren't caught by the cheap weight-multiset
    prefilter."""
    classes = list(enumerate_doubly_even_at(N, k))
    if len(classes) < 2:
        pytest.skip("need ≥ 2 classes to test negatives")
    # Group by weight enumerator. Within each group, all pairs must
    # cross-reject under paired_iso.
    from collections import defaultdict

    from doubly_even.canon.paired_iso import _weight_multiset

    by_we: dict[tuple[int, ...], list[tuple[int, ...]]] = defaultdict(list)
    for ec in classes:
        cf = _canonical_form(ec.code.basis, N)
        by_we[_weight_multiset(cf, k)].append(cf)
    found_pair = False
    for group in by_we.values():
        if len(group) < 2:
            continue
        found_pair = True
        for i in range(len(group)):
            for j in range(len(group)):
                if i == j:
                    continue
                assert not paired_iso(group[i], group[j], N), (
                    f"paired_iso falsely accepted at N={N}, k={k}: "
                    f"d={group[i]}, cf={group[j]}"
                )
    if not found_pair:
        pytest.skip("no two classes shared a weight enumerator at this cell")


def test_paired_iso_counters_increment() -> None:
    """The diagnostic counters tick on a non-trivial pair."""
    classes = list(enumerate_doubly_even_at(8, 3))
    assert len(classes) >= 2
    cf0 = _canonical_form(classes[0].code.basis, 8)
    counters = IsoCounters()
    paired_iso(classes[0].code.basis, cf0, 8, counters=counters)
    assert counters.refines >= 1
