"""D15 coset-spectrum parent rule: kernel-level equivalence checks.

The rule changes WHICH (parent-class, coset-orbit) pair emits each class,
so emitted representatives may differ at rank >= 2. What may not change:

- per-rank class counts (oracle: the legacy rule, itself pinned to
  DFGHILM Table 3 by ``test_augment.py``),
- the per-rank sorted multiset of |Aut| values (hence the mass
  ``sum N!/|Aut|`` per rank, which we re-check against ``gaborit_sigma``
  here as an independent certificate).

The deeper Rust-side coverage (rank-cap mixing sweep, audit byte
identity, parallel-vs-sequential identity) lives in
``rust/tests/parent_rule_equivalence.rs``.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

kernel = pytest.importorskip("doubly_even_kernel")

from doubly_even.spec.mass import gaborit_sigma

N = 14


def _enumerate(monkeypatch, rule: str):
    """Per-rank {rank: sorted |Aut| list} under the given parent rule.

    The kernel resolves ``DOUBLY_EVEN_PARENT_RULE`` once per driver call,
    so flipping the env between calls in one process is supported (the
    same mechanism ``scripts/experimental/d15_phi_audit.py`` uses).
    """
    monkeypatch.setenv("DOUBLY_EVEN_PARENT_RULE", rule)
    cap = N // 2
    quota = [gaborit_sigma(N, k) for k in range(cap + 1)]
    raw, _stats, _per_k = kernel.enumerate_doubly_even(
        N, cap, quota, math.factorial(N)
    )
    profile: dict[int, list[int]] = defaultdict(list)
    for rref, _cco, _gens, aut_order, _orbits in raw:
        profile[len(rref)].append(int(aut_order))
    return {k: sorted(v) for k, v in profile.items()}


def test_coset_spectrum_matches_legacy_class_profile(monkeypatch):
    legacy = _enumerate(monkeypatch, "legacy")
    phi = _enumerate(monkeypatch, "coset-spectrum")
    assert phi == legacy


def test_coset_spectrum_satisfies_mass_formula(monkeypatch):
    """Independent certificate: sum N!/|Aut| == gaborit_sigma per rank."""
    phi = _enumerate(monkeypatch, "coset-spectrum")
    fact = math.factorial(N)
    for k, auts in phi.items():
        assert sum(fact // a for a in auts) == gaborit_sigma(N, k), f"rank {k}"


def test_unknown_rule_value_fails_loudly(monkeypatch):
    monkeypatch.setenv("DOUBLY_EVEN_PARENT_RULE", "cosetspectrum")  # typo
    quota = [gaborit_sigma(8, k) for k in range(5)]
    with pytest.raises(BaseException, match="DOUBLY_EVEN_PARENT_RULE"):
        kernel.enumerate_doubly_even(8, 4, quota, math.factorial(8))
