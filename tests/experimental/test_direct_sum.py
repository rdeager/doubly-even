"""Tests for :mod:`doubly_even.spec.experimental.direct_sum`."""

from __future__ import annotations

import pytest

from doubly_even.canon.nauty import are_equivalent, canon_info
from doubly_even.spec.codes import Code
from doubly_even.spec.experimental.direct_sum import direct_sum
from doubly_even.spec.doubly_even import is_doubly_even


def test_zero_codes_compose_to_zero():
    Z3 = Code.zero(3)
    Z4 = Code.zero(4)
    D = direct_sum(Z3, Z4)
    assert D.n == 7
    assert D.rank == 0


def test_basis_shift_layout():
    # C1 = {0000, 1111} (rep code of length 4, doubly-even).
    # C2 = {0, e1} of length 2 (singly-even, but we just test bit layout).
    C1 = Code(n=4, basis=(0b1111,))
    C2 = Code(n=2, basis=(0b01,))
    D = direct_sum(C1, C2)
    assert D.n == 6
    # Codewords: 000000, 001111 (C1 set, C2 not), 010000 (C2 set, C1 not),
    # 011111 (both set). Bit 4 is the C2's bit 0 (i.e., position n1=4).
    cws = sorted(D.codewords())
    assert cws == [0b000000, 0b001111, 0b010000, 0b011111]


def test_preserves_double_evenness():
    # Two doubly-even length-4 codes ⊕ should be doubly-even.
    C = Code(n=4, basis=(0b1111,))
    assert is_doubly_even(C)
    D = direct_sum(C, C)
    assert is_doubly_even(D)
    assert D.rank == 2
    assert D.n == 8


def test_rank_additivity():
    C1 = Code(n=4, basis=(0b1111,))
    C2 = Code(n=8, basis=(0b11110000, 0b00001111))
    D = direct_sum(C1, C2)
    assert D.rank == C1.rank + C2.rank == 3


def test_aut_order_distinct_components():
    """|Aut(C1 ⊕ C2)| = |Aut(C1)| * |Aut(C2)| when C1 ≇ C2."""
    # C1: [4, 1] rep code, Aut = S_4, order 24.
    C1 = Code(n=4, basis=(0b1111,))
    # C2: [6, 1] rep code of length 6, doubly-even? Weight 6 is not div by 4.
    # Use C2 = {0, 11111111} on 8 columns — also rep code, Aut = S_8.
    C2 = Code(n=8, basis=(0b11111111,))
    D = direct_sum(C1, C2)
    aut_C1 = canon_info(C1).aut_order
    aut_C2 = canon_info(C2).aut_order
    aut_D = canon_info(D).aut_order
    # C1 and C2 are non-equivalent (different lengths), so block-swap
    # factor doesn't apply.
    assert aut_D == aut_C1 * aut_C2


def test_aut_order_identical_components_includes_block_swap():
    """|Aut(C ⊕ C)| = |Aut(C)|^2 * 2 (block-swap factor)."""
    C = Code(n=4, basis=(0b1111,))
    D = direct_sum(C, C)
    aut_C = canon_info(C).aut_order
    aut_D = canon_info(D).aut_order
    assert aut_D == aut_C * aut_C * 2


def test_direct_sum_commutes_under_equivalence():
    """C1 ⊕ C2 ≡ C2 ⊕ C1 as permutation-equivalent codes."""
    C1 = Code(n=4, basis=(0b1111,))
    C2 = Code(n=8, basis=(0b11111111,))
    D12 = direct_sum(C1, C2)
    D21 = direct_sum(C2, C1)
    assert D12 != D21  # different bit layouts; not literally equal
    assert are_equivalent(D12, D21)


def test_direct_sum_associative_up_to_equivalence():
    """(A ⊕ B) ⊕ C ≡ A ⊕ (B ⊕ C)."""
    A = Code(n=4, basis=(0b1111,))
    B = Code(n=4, basis=(0b1111,))
    C = Code(n=8, basis=(0b11111111,))
    left = direct_sum(direct_sum(A, B), C)
    right = direct_sum(A, direct_sum(B, C))
    assert left.n == right.n == 16
    assert are_equivalent(left, right)


@pytest.mark.parametrize("n", [0, 1, 4])
def test_direct_sum_with_zero_code_is_padding(n):
    """C ⊕ zero_code(n) has the same codeword multiset as C padded with n zero bits."""
    C = Code(n=4, basis=(0b1111,))
    D = direct_sum(C, Code.zero(n))
    assert D.n == 4 + n
    assert D.rank == C.rank
    expected_cws = sorted(c for c in C.codewords())
    actual_cws = sorted(c for c in D.codewords())
    assert expected_cws == actual_cws
