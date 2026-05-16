"""Tests for the doubly-even predicate and augmentation criterion."""

from hypothesis import given
from hypothesis import strategies as st

from doubly_even.spec.codes import Code
from doubly_even.spec.doubly_even import (
    all_codewords_doubly_even,
    doubly_even_extension,
    is_doubly_even,
)
from doubly_even.spec.vectors import wt


# ---------------------------------------------------------------- examples


def test_zero_code_is_doubly_even():
    assert is_doubly_even(Code.zero(8))


def test_extended_hamming_8_4_is_doubly_even():
    # H_8 = [8, 4, 4] extended Hamming. Standard basis (8 columns):
    #   1 1 1 1 0 0 0 0
    #   1 1 0 0 1 1 0 0
    #   1 0 1 0 1 0 1 0
    #   1 1 1 1 1 1 1 1   (the all-ones vector, weight 8)
    # We encode bit i ↔ column i (LSB at position 0).
    rows = (
        0b00001111,
        0b00110011,
        0b01010101,
        0b11111111,
    )
    C = Code(8, rows)
    assert C.rank == 4
    assert is_doubly_even(C)
    # Verify via the brute definition as well:
    assert all_codewords_doubly_even(C)


def test_single_weight_2_vector_not_doubly_even():
    C = Code(4, (0b0011,))
    assert not is_doubly_even(C)


def test_single_weight_4_vector_is_doubly_even():
    C = Code(4, (0b1111,))
    assert is_doubly_even(C)


def test_orthogonal_weight4_pair_is_doubly_even():
    # Two weight-4 vectors in F_2^8 that are orthogonal (disjoint supports)
    C = Code(8, (0b00001111, 0b11110000))
    assert is_doubly_even(C)


def test_nonorthogonal_weight4_pair_is_not_doubly_even():
    # Two weight-4 vectors with overlap 1 -- their sum has weight 6.
    # 0b00001111 has support {0,1,2,3}; 0b00011110 has support {1,2,3,4}; overlap = 3 (odd)
    # so their dot product is 1 (odd), not 0.
    C = Code(8, (0b00001111, 0b00011110))
    assert not is_doubly_even(C)


# ----------------------------------------------------------- augmentation


def test_augmentation_criterion_accepts_compatible_vector():
    C = Code(8, (0b00001111,))
    assert is_doubly_even(C)
    # Extend with a disjoint weight-4 vector
    assert doubly_even_extension(C, 0b11110000)


def test_augmentation_criterion_rejects_wrong_weight():
    C = Code(8, (0b00001111,))
    # wt = 2, not ≡ 0 mod 4
    assert not doubly_even_extension(C, 0b00000011)


def test_augmentation_criterion_rejects_non_orthogonal():
    C = Code(8, (0b00001111,))
    # wt(v) = 4 but v overlaps C with parity 1
    v = 0b00011110  # support {1,2,3,4}; dot with 0b00001111 = bits 1,2,3 = parity 1
    assert wt(v) == 4
    assert not doubly_even_extension(C, v)


# ----------------------------------------------------------- agreement


@given(
    n=st.integers(min_value=0, max_value=8),
    basis_count=st.integers(min_value=0, max_value=3),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_is_doubly_even_agrees_with_brute_force(n, basis_count, seed):
    # Generate a few random basis vectors of weight ≡ 0 mod 4 to bias toward
    # interesting examples, then check both routines agree.
    import random
    rng = random.Random(seed)
    basis = []
    mask = (1 << n) - 1 if n > 0 else 0
    for _ in range(basis_count):
        v = rng.randint(0, mask)
        basis.append(v)
    C = Code(n, tuple(basis))
    assert is_doubly_even(C) == all_codewords_doubly_even(C)
