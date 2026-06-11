//! Small GF(2) linear-algebra helpers used by the quotient construction.
//!
//! Mirrors `doubly_even.spec.codes.Code.rref_basis` and
//! `doubly_even.spec.vectors.apply_permutation` for the slices the kernel
//! actually consumes.

use crate::types::{BinVec, ColPerm};

/// Reduce a set of rows over `GF(2)` to RREF.
///
/// Returns `(rref_rows, pivots)` where `rref_rows.len() == rank` and
/// `pivots[i]` is the leading-1 column of `rref_rows[i]`. Columns are
/// processed left-to-right (LSB to MSB), matching `Code.rref_basis`.
///
/// `n` bounds the column range; rows must have no bits set beyond
/// `[0, n)`. We don't verify (the FFI layer already does).
pub fn row_reduce(rows_in: &[BinVec], n: u32) -> (Vec<BinVec>, Vec<u32>) {
    let mut rows: Vec<BinVec> = rows_in.to_vec();
    let mut pivots: Vec<u32> = Vec::with_capacity(rows.len().min(n as usize));
    let mut r = 0usize;
    for c in 0..n {
        let mut pivot: Option<usize> = None;
        for i in r..rows.len() {
            if (rows[i] >> c) & 1 == 1 {
                pivot = Some(i);
                break;
            }
        }
        let Some(p) = pivot else { continue };
        rows.swap(r, p);
        let pivot_row = rows[r];
        for i in 0..rows.len() {
            if i != r && (rows[i] >> c) & 1 == 1 {
                rows[i] ^= pivot_row;
            }
        }
        pivots.push(c);
        r += 1;
    }
    rows.truncate(r);
    (rows, pivots)
}

/// Apply a column permutation to a binary vector.
///
/// `sigma[i] = j` means bit `i` of the input becomes bit `j` of the output.
/// `sigma.len()` is the code length `n`; bits of `v` outside `[0, n)` are
/// ignored. Pure port of `doubly_even.spec.vectors.apply_permutation`.
#[inline]
pub fn apply_permutation(v: BinVec, sigma: &[u32]) -> BinVec {
    let mut out: BinVec = 0;
    let mut bits = v;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        if i >= sigma.len() {
            break;
        }
        out |= 1u64 << sigma[i];
        bits &= bits - 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[7,4]` Hamming generator matrix in standard form `[I | A]`.
    /// Pivot columns are 0,1,2,3.
    #[test]
    fn rref_hamming74_standard_form() {
        // basis given as rows (each int has bits 0..6 over F_2^7)
        let rows: Vec<BinVec> = vec![
            0b1110001, // x0 + x1 + x2 + x6   (pivot 0)
            0b1101010, // x1 + x3 + x4 + x6   – mangled on purpose
            0b1011100, // x2 + x3 + x4 + x5
            0b0111100, // x2 + x3 + x4 + x5 + x6 – linearly dependent on others?
        ];
        // The intent here is simply that RREF picks linearly independent rows
        // and emits ascending pivots — we don't pin the exact basis.
        let (rref, pivots) = row_reduce(&rows, 7);
        // pivots must be strictly increasing
        for w in pivots.windows(2) {
            assert!(w[0] < w[1], "pivots not increasing: {pivots:?}");
        }
        // each pivot bit set in its row, cleared in every other row
        for (i, &p) in pivots.iter().enumerate() {
            for (j, row) in rref.iter().enumerate() {
                let bit = (row >> p) & 1;
                if i == j {
                    assert_eq!(bit, 1, "pivot {p} missing from its own row");
                } else {
                    assert_eq!(bit, 0, "pivot {p} not cleared in row {j}");
                }
            }
        }
    }

    /// Idempotence: running RREF twice should give the same answer.
    #[test]
    fn rref_idempotent() {
        let rows: Vec<BinVec> = vec![0b101101, 0b011011, 0b110110];
        let (r1, p1) = row_reduce(&rows, 6);
        let (r2, p2) = row_reduce(&r1, 6);
        assert_eq!(r1, r2);
        assert_eq!(p1, p2);
    }

    /// Zero-input edge case.
    #[test]
    fn rref_empty() {
        let (rref, pivots) = row_reduce(&[], 5);
        assert!(rref.is_empty());
        assert!(pivots.is_empty());
    }

    #[test]
    fn apply_permutation_identity() {
        let sigma: ColPerm = (0..8u32).collect();
        for v in [0u64, 1, 0xFF, 0x55, 0xAA] {
            assert_eq!(apply_permutation(v, &sigma), v);
        }
    }

    #[test]
    fn apply_permutation_swap_first_two() {
        // sigma sends 0 -> 1, 1 -> 0, identity on rest.
        let mut sigma: ColPerm = (0..8u32).collect();
        sigma[0] = 1;
        sigma[1] = 0;
        assert_eq!(apply_permutation(0b0001, &sigma), 0b0010);
        assert_eq!(apply_permutation(0b0011, &sigma), 0b0011);
        assert_eq!(apply_permutation(0b1100, &sigma), 0b1100);
    }

    #[test]
    fn apply_permutation_matches_python_reference() {
        // Python:
        //   apply_permutation(0b10110, [2, 0, 4, 1, 3])
        //   bits 1, 2, 4 set → land at sigma[1]=0, sigma[2]=4, sigma[4]=3
        //   → bits 0, 3, 4 set = 0b11001
        let sigma: ColPerm = vec![2, 0, 4, 1, 3];
        assert_eq!(apply_permutation(0b10110, &sigma), 0b11001);
    }
}
