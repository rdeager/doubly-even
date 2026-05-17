"""Differential tests for the Rust ``canon_info_qd_native`` Q_D-graph path.

The Q_D-graph canonicaliser feeds nauty a smaller bipartite graph built
from only the low-weight codewords of C (each Hamming-weight stratum is
its own colour class), with a span check ensuring its column-side
stabiliser equals Aut(C).

For every doubly-even code at ``N <= 14`` enumerated by the existing
pipeline, we cross-check that the Q_D-graph result agrees with the
production ``canon_info_native`` (nauty over the full bipartite) on:

* exact ``|Aut(C)|``;
* column-orbit set partition (orbit ids may differ between the two
  graphs, but the partition they encode must be identical);
* the aut generators returned by Q_D actually generate a group of the
  same order — guarding against accidental "extra" permutations that
  could arise if the low-weight set fails to span.
"""

from __future__ import annotations

import pytest

from doubly_even.canon.nauty import _trustable_pynauty_order
from doubly_even.canon.permutations import group_order
from doubly_even.enumerate.augment import enumerate_doubly_even
from doubly_even.enumerate.augment import _kernel as kernel


pytestmark = pytest.mark.skipif(
    kernel is None,
    reason="Rust kernel not built",
)


def _orbit_partition(orbits: tuple[int, ...]) -> frozenset[frozenset[int]]:
    """Set partition view of an orbit-id vector. Stable under orbit relabelling."""
    groups: dict[int, list[int]] = {}
    for i, o in enumerate(orbits):
        groups.setdefault(o, []).append(i)
    return frozenset(frozenset(g) for g in groups.values())


def _aut_order_from_native(
    grpsize1: float, grpsize2: int, gens: list[tuple[int, ...]], n: int
) -> int:
    trusted = _trustable_pynauty_order(grpsize1, grpsize2)
    if trusted is not None:
        return trusted
    return group_order(gens, n)


def _all_doubly_even_codes(N: int) -> list[tuple[tuple[int, ...], int]]:
    """Enumerate every doubly-even code at length ``N`` and return its
    RREF basis plus rank. The McKay enumerator already yields one per
    equivalence class — that's enough cross-section for a differential
    check, and using the production enumerator means the Q_D dispatch is
    actually exercised on the same inputs the recursion will hit.
    """
    out: list[tuple[tuple[int, ...], int]] = []
    for ec in enumerate_doubly_even(N):
        rref = ec.code.rref_basis()[0]
        out.append((rref, ec.code.rank))
    return out


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12, 14])
def test_qd_native_matches_canon_info_native(N: int) -> None:
    """Q_D-graph canon agrees with full bipartite on aut order + column orbits."""
    codes = _all_doubly_even_codes(N)
    assert codes, f"no doubly-even codes enumerated at N={N}"

    compared = 0
    bailed = 0
    for rref, _k in codes:
        qd_out = kernel.canon_info_qd_native(list(rref), N)
        # Always cross-check against the full bipartite canon — that's our
        # oracle regardless of whether the Q_D builder bailed.
        ccol_n, gens_n, gs1_n, gs2_n, orb_n = kernel.canon_info_native(list(rref), N)
        gens_n_t = [tuple(g) for g in gens_n]
        aut_native = _aut_order_from_native(gs1_n, gs2_n, gens_n_t, N)

        if qd_out is None:
            # Builder bailed (low-weight set too dense). Nothing to compare
            # against — the production dispatch will fall back to native.
            bailed += 1
            continue

        ccol_q, gens_q, gs1_q, gs2_q, orb_q = qd_out
        gens_q_t = [tuple(g) for g in gens_q]
        aut_qd = _aut_order_from_native(gs1_q, gs2_q, gens_q_t, N)

        assert aut_qd == aut_native, (
            f"N={N}, rref={list(rref)}: aut order differs "
            f"(qd={aut_qd}, native={aut_native})"
        )

        # Compute the group order *from the generators* — if the Q_D graph
        # ever admitted a column permutation outside Aut(C) we'd see it
        # here, since the returned generators would span a larger group
        # than nauty's reported order.
        if gens_q_t:
            gen_order_qd = group_order(gens_q_t, N)
            assert gen_order_qd == aut_qd, (
                f"N={N}, rref={list(rref)}: Q_D generators span order "
                f"{gen_order_qd} but nauty reported {aut_qd}"
            )

        assert _orbit_partition(tuple(orb_q)) == _orbit_partition(tuple(orb_n)), (
            f"N={N}, rref={list(rref)}: column-orbit partition differs"
        )
        compared += 1

    # Sanity: at least some codes should hit the Q_D path. Below N=10 most
    # codes have rank too small to clear the bail threshold, so we relax
    # this guard for the smallest sizes.
    if N >= 10:
        assert compared > 0, f"N={N}: every code bailed — Q_D path not exercised"
