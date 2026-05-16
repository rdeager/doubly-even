from hypothesis import given
from hypothesis import strategies as st

from doubly_even.spec.vectors import (
    add,
    apply_permutation,
    basis_vector,
    dot,
    from_bits,
    support,
    to_bits,
    wt,
)


def test_wt_zero():
    assert wt(0) == 0


def test_wt_examples():
    assert wt(0b1011) == 3
    assert wt(0b1111_1111) == 8
    assert wt((1 << 32) - 1) == 32


def test_dot_examples():
    assert dot(0b1010, 0b0101) == 0
    assert dot(0b1011, 0b0011) == 0  # two shared 1s, parity 0
    assert dot(0b1011, 0b0001) == 1  # one shared 1


def test_basis_vector_round_trip():
    for i in range(10):
        e = basis_vector(i)
        assert wt(e) == 1
        assert support(e) == [i]


def test_from_to_bits_round_trip():
    bits = [1, 0, 1, 1, 0, 0, 1]
    v = from_bits(bits)
    assert to_bits(v, 7) == bits


@given(u=st.integers(min_value=0, max_value=2**16 - 1),
       v=st.integers(min_value=0, max_value=2**16 - 1))
def test_dot_symmetric(u, v):
    assert dot(u, v) == dot(v, u)


@given(u=st.integers(min_value=0, max_value=2**16 - 1),
       v=st.integers(min_value=0, max_value=2**16 - 1))
def test_dot_bilinear(u, v):
    # ⟨u, v⟩ + ⟨0, v⟩ = ⟨u + 0, v⟩  -- trivial; nontrivial bilinearity:
    # ⟨u ^ v, u⟩ = ⟨u, u⟩ ^ ⟨v, u⟩
    assert dot(u ^ v, u) == (dot(u, u) ^ dot(v, u))


@given(v=st.integers(min_value=0, max_value=2**12 - 1))
def test_dot_self_equals_wt_mod_2(v):
    assert dot(v, v) == wt(v) % 2


@given(u=st.integers(min_value=0, max_value=2**12 - 1),
       v=st.integers(min_value=0, max_value=2**12 - 1))
def test_polarization(u, v):
    # Hamming polarisation over the integers:
    # wt(u ^ v) = wt(u) + wt(v) - 2 * (number of shared 1 bits)
    shared = (u & v).bit_count()
    assert wt(u ^ v) == wt(u) + wt(v) - 2 * shared


def test_apply_permutation_identity():
    sigma = list(range(8))
    assert apply_permutation(0b1011_0110, sigma) == 0b1011_0110


def test_apply_permutation_swap():
    # swap columns 0 and 1
    sigma = [1, 0, 2, 3]
    # 0b0001 (bit 0 set) should become 0b0010 (bit 1 set)
    assert apply_permutation(0b0001, sigma) == 0b0010
    assert apply_permutation(0b0010, sigma) == 0b0001
    assert apply_permutation(0b0100, sigma) == 0b0100


@given(v=st.integers(min_value=0, max_value=2**8 - 1))
def test_add_is_xor(v):
    assert add(v, v) == 0
    assert add(v, 0) == v
