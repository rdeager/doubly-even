"""Tie-dump hook (``DOUBLY_EVEN_TIE_DUMP``, 2026-06-12).

The sequential kernel drivers append one JSONL record per φ-tie —
the raw material for the invariant-collision analysis
(``scripts/experimental/tie_collision_analysis.py``). Checks:

1. Record count equals ``phi_tie_accept + phi_tie_reject`` from the same
   run's stats (oracle: the kernel's own tie counters).
2. ``tie_orbits`` is a partition of ``m_set``; accepted/rejected flags and
   parent ranks are well-formed.
3. The parallel drivers refuse the knob (sequential-only contract).
"""

from __future__ import annotations

import json
import math

import pytest

from doubly_even.spec.mass import gaborit_sigma

kernel = pytest.importorskip("doubly_even_kernel")

# N=18 is the smallest N with a measured nonzero tie count under the
# default coset-spectrum rule (1 tie-reject; decision-deterministic).
N_WITH_TIES = 18


def _run_sequential(N: int):
    cap = N // 2
    quota = [gaborit_sigma(N, k) for k in range(cap + 1)]
    raw, stats, _per_k = kernel.enumerate_doubly_even(
        N, cap, quota, math.factorial(N)
    )
    layout, _per_k_rows = kernel.kernel_stats_layout()
    return raw, {name: stats[i] for i, name in enumerate(layout)}


def test_tie_dump_records_match_tie_counters(tmp_path, monkeypatch) -> None:
    dump = tmp_path / "ties.jsonl"
    monkeypatch.setenv("DOUBLY_EVEN_TIE_DUMP", str(dump))
    monkeypatch.delenv("DOUBLY_EVEN_THREADS", raising=False)
    _raw, stats = _run_sequential(N_WITH_TIES)

    ties_expected = stats["phi_tie_accept"] + stats["phi_tie_reject"]
    assert ties_expected > 0, (
        f"N={N_WITH_TIES} produced no ties — pick a larger N for this test"
    )

    records = [json.loads(line) for line in dump.read_text().splitlines()]
    assert len(records) == ties_expected

    accepts = 0
    for rec in records:
        assert rec["n"] == N_WITH_TIES
        assert 1 <= rec["parent_k"] <= N_WITH_TIES // 2
        assert len(rec["m_set"]) >= 2, "a tie involves >= 2 argmin functionals"
        # tie_orbits partitions m_set.
        flattened = sorted(u for orbit in rec["tie_orbits"] for u in orbit)
        assert flattened == sorted(rec["m_set"])
        assert all(orbit for orbit in rec["tie_orbits"])
        # aut_order is a decimal string (u128-safe).
        assert int(rec["aut_order"]) >= 1
        # parent_rref rows are hex strings.
        assert all(int(row, 16) >= 0 for row in rec["parent_rref"])
        int(rec["v"], 16)
        accepts += bool(rec["accept"])
    assert accepts == stats["phi_tie_accept"]


def test_tie_dump_absent_when_env_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DOUBLY_EVEN_TIE_DUMP", raising=False)
    _raw, stats = _run_sequential(12)
    assert not (tmp_path / "ties.jsonl").exists()
    assert stats["canon_autom_only_calls"] > 0  # the lever is on by default


def test_parallel_driver_rejects_tie_dump(tmp_path, monkeypatch) -> None:
    # Probe for the parallel feature: with it off, num_threads >= 2 raises
    # ValueError before any tie-dump check can fire.
    monkeypatch.delenv("DOUBLY_EVEN_TIE_DUMP", raising=False)
    N = 14  # cap 7 > frontier depth 4, so the parallel path really engages
    cap = N // 2
    quota = [gaborit_sigma(N, k) for k in range(cap + 1)]
    try:
        kernel.enumerate_doubly_even(N, cap, quota, math.factorial(N), 2)
    except ValueError:
        pytest.skip("kernel built without the parallel feature")

    monkeypatch.setenv("DOUBLY_EVEN_TIE_DUMP", str(tmp_path / "ties.jsonl"))
    with pytest.raises(BaseException, match="sequential-only"):
        kernel.enumerate_doubly_even(N, cap, quota, math.factorial(N), 2)
