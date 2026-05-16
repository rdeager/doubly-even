from doubly_even.spec.codes import Code
from doubly_even.spec.vectors import wt


def test_zero_code():
    C = Code.zero(8)
    assert C.rank == 0
    assert 0 in C
    assert 1 not in C
    assert list(C.codewords()) == [0]


def test_whole_space():
    C = Code.whole(4)
    assert C.rank == 4
    for v in range(16):
        assert v in C


def test_rref_basic():
    # basis [0b0011, 0b0101] over F_2^4
    C = Code(4, (0b0011, 0b0101))
    rows, pivots = C.rref_basis()
    # RREF: 0b0011, 0b0110 (then 0b0011 stays since lowest col 0; second row eliminates col 0)
    # Let's check the actual columns: column 0 (LSB) bits set in row 0 and row 1.
    # Pivot at column 0 from row 0 (value 0b0011), then eliminate column 0 in row 1:
    # row1 ^= row0 = 0b0101 ^ 0b0011 = 0b0110. Next pivot column 1 (bit 1) -- row 0 has bit 1, row 1 doesn't.
    # Eliminate bit 1 in row 0 against row 1: row 0 ^= row 1 if row 0 bit 1 is set... wait we look at column-by-column.
    # Actually the algorithm scans columns. Column 0 finds pivot row 0 (bit 0 of 0b0011 = 1).
    # Eliminate column 0 in row 1: row1 has bit 0 = 1 (0b0101), so row1 ^= row0 = 0b0110.
    # Column 1: pivot search from row 1; row 1 = 0b0110, bit 1 = 1. So swap nothing.
    # Eliminate column 1 in row 0: row 0 = 0b0011 has bit 1 = 1, so row 0 ^= row 1 = 0b0011 ^ 0b0110 = 0b0101.
    # Now: rows = [0b0101, 0b0110], pivots = [0, 1].
    assert rows == (0b0101, 0b0110)
    assert pivots == (0, 1)


def test_rank_with_dependent_basis():
    # Same vector twice
    C = Code(4, (0b1010, 0b1010))
    assert C.rank == 1


def test_rank_empty():
    assert Code.zero(8).rank == 0


def test_membership():
    # 2-dim code in F_2^4: span(0b0011, 0b1100)
    C = Code(4, (0b0011, 0b1100))
    assert 0 in C
    assert 0b0011 in C
    assert 0b1100 in C
    assert 0b1111 in C  # sum
    assert 0b0001 not in C
    assert 0b0010 not in C
    assert 0b1010 not in C


def test_codewords_enumerates_2k():
    C = Code(4, (0b0011, 0b1100))
    words = sorted(set(C.codewords()))
    assert words == [0b0000, 0b0011, 0b1100, 0b1111]


def test_dual_of_zero_is_whole():
    C = Code.zero(5)
    D = C.dual()
    assert D.rank == 5
    for v in range(32):
        assert v in D


def test_dual_of_whole_is_zero():
    C = Code.whole(4)
    D = C.dual()
    assert D.rank == 0
    assert 0 in D


def test_dual_dual_is_self():
    # span(0b0011, 0b1100) ⊆ F_2^4 — repetition code in two blocks
    C = Code(4, (0b0011, 0b1100))
    DD = C.dual().dual()
    # Same rowspace, not necessarily same basis
    for v in range(16):
        assert (v in C) == (v in DD)


def test_dual_dimension_sums_to_n():
    # Any code: dim C + dim C⊥ = n
    for basis in [
        (0b0001,),
        (0b0011, 0b0101),
        (0b0001, 0b0010, 0b0100, 0b1000),
        (),
    ]:
        C = Code(4, basis)
        D = C.dual()
        assert C.rank + D.rank == 4


def test_dual_orthogonal():
    C = Code(6, (0b000111, 0b011001, 0b101010))
    D = C.dual()
    # Every codeword of D is orthogonal to every codeword of C
    for c in C.codewords():
        for d in D.codewords():
            assert wt(c & d) % 2 == 0


def test_extend_adds_vector():
    C = Code(4, (0b0011,))
    C2 = C.extend(0b1100)
    assert C2.rank == 2
    assert 0b1111 in C2


def test_extend_with_dependent_vector():
    C = Code(4, (0b0011,))
    C2 = C.extend(0b0011)  # already in C
    assert C2.rank == 1
    assert 0b1100 not in C2


def test_is_orthogonal_to():
    C = Code(4, (0b0011, 0b1100))
    # 0b1010 has dot 1 with 0b0011 (shared bit 1) -- actually 0b1010 & 0b0011 = 0b0010, wt 1, dot 1
    assert not C.is_orthogonal_to(0b1010)
    assert C.is_orthogonal_to(0b1111)  # dot with both basis = 0
    assert C.is_orthogonal_to(0b0011)


def test_validation_rejects_out_of_range_basis():
    import pytest

    with pytest.raises(ValueError):
        Code(4, (0b10000,))  # bit 4 set in length-4 vector
    with pytest.raises(ValueError):
        Code(-1, ())
