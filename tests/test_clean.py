"""Cross-checks for ``doubly_even.clean`` (the pedagogical reference module).

Covers:
- Rank-0 shortcut in ``qc_candidates`` (Speedup A1): closed-form ``(1<<4ℓ)-1``
  must agree with the full Q_C orbit-min pipeline for the zero code.
- End-to-end class counts match DFGHILM Table 3 cells through N=12.
- ``codewords_by_weight`` parity vs the ``codewords()`` Gray walk (Speedup B1,
  only runs once the method exists).
"""

from __future__ import annotations

from collections import Counter

import pytest

from doubly_even.clean import enumerate_doubly_even
from doubly_even.clean._canon import CanonInfo, canon_info
from doubly_even.clean._qc import qc_candidates
from doubly_even.clean._spec import Code


# DFGHILM Table 3 cells we replicate end-to-end (small N only — keep the test
# default-suite-fast). Larger N cells are covered by the production-path
# ``tests/test_augment.py``.
DFGHILM_SMALL: dict[int, dict[int, int]] = {
    8: {1: 2, 2: 2, 3: 2, 4: 1},
    10: {1: 2, 2: 3, 3: 3, 4: 2},
    12: {1: 3, 2: 5, 3: 7, 4: 7, 5: 2},
}


@pytest.mark.parametrize("N", [4, 6, 8, 12, 16, 20, 22, 24])
def test_rank0_shortcut_matches_closed_form(N: int):
    """``qc_candidates(zero code)`` returns the ⌊N/4⌋ all-ones-prefix vectors."""
    C = Code.zero(N)
    aut_gens = CanonInfo.trivial_sn(N).aut_generators
    reps = qc_candidates(C, aut_gens)
    expected = [(1 << w) - 1 for w in range(4, N + 1, 4)]
    assert reps == expected


@pytest.mark.parametrize("N, expected", sorted(DFGHILM_SMALL.items()))
def test_enumerate_matches_dfghilm_table_3(N: int, expected: dict[int, int]):
    """End-to-end class counts at small N match DFGHILM Table 3 cells."""
    classes: Counter[int] = Counter()
    for ec in enumerate_doubly_even(N):
        classes[ec.code.rank] += 1
    observed = {k: v for k, v in classes.items() if k >= 1}
    assert observed == expected


def _hamming_8_4_4_basis() -> tuple[int, ...]:
    # Standard [8, 4, 4] extended-Hamming generator (rows).
    return (0b00001111, 0b00110011, 0b01010101, 0b11111111)


def test_codewords_by_weight_matches_filter():
    """``codewords_by_weight`` (if present) buckets the same multiset as
    ``codewords()`` filtered by ``bit_count()``. Skips if Speedup B not yet
    landed."""
    C = Code(8, _hamming_8_4_4_basis())
    method = getattr(C, "codewords_by_weight", None)
    if method is None:
        pytest.skip("Code.codewords_by_weight not yet implemented (Speedup B)")
    by_w = method()
    all_cw = C.codewords()
    for w in range(0, C.n + 1):
        expected = sorted(c for c in all_cw if c.bit_count() == w)
        assert sorted(by_w.get(w, [])) == expected


def test_canon_info_zero_code_is_full_symmetric():
    """``CanonInfo.trivial_sn(N)`` returns N! for the zero code without
    invoking nauty."""
    import math

    for N in (4, 6, 8, 16):
        info = canon_info(Code.zero(N))
        assert info.aut_order == math.factorial(N)
        assert info.aut_generators  # nonempty for N >= 2
