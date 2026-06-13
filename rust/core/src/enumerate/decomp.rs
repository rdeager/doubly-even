//! Direct-sum / twin-column structure of a canon-call input — the
//! read-only metrics behind the `DOUBLY_EVEN_DECOMP_LOG` experiment
//! (bottlenecks §4 lever 4; external-feedback P1/P2).
//!
//! Soundness of the component computation: connected components of the
//! RREF rows' supports give exactly the finest direct-sum decomposition
//! of the code. One direction is immediate (rows partitioned by support
//! ⇒ the code is the direct sum of the per-part row spans). For the
//! other, let `C = C_A ⊕ C_B` over a coordinate partition and let `r` be
//! an RREF row, `r = a + b` with `a ∈ C_A, b ∈ C_B`. If `b ≠ 0` its
//! leading column is some pivot column `p_b ∈ B` (every codeword leads
//! at a pivot), and `r[p_b] = b[p_b] = 1` since `a` vanishes on `B` —
//! contradicting RREF reducedness unless `r` is itself the pivot row of
//! `p_b`, in which case `a = 0` symmetrically. So every RREF row lies
//! wholly inside one part of ANY direct-sum partition, in particular the
//! finest one, and row-support connectivity recovers it exactly.
//!
//! Twin columns: column `i` and `j` are twins of the *code* iff they are
//! equal as columns of the RREF basis matrix — row operations preserve
//! column equality, so this is basis-independent (the P2 caution about
//! "twins of the code, not of G_low" is satisfied by construction).

use crate::types::BinVec;

/// OR of all rows: the column support of the code as a bitmask.
pub(crate) fn column_support(rref: &[BinVec]) -> BinVec {
    rref.iter().fold(0, |acc, &r| acc | r)
}

/// Sizes of the connected components of the column-support graph
/// (union-find over each row's support), restricted to support columns,
/// sorted descending. Zero columns are NOT counted as components — they
/// are reported separately via `n − popcount(support)` (each is a
/// trivial `0` summand). A result with `len() ≥ 2` means the code is a
/// nontrivial direct sum even within its support.
pub(crate) fn component_sizes(rref: &[BinVec], n: u32) -> Vec<u32> {
    let n = n as usize;
    let mut parent: Vec<u32> = (0..n as u32).collect();

    fn find(parent: &mut [u32], i: u32) -> u32 {
        let mut cur = i;
        while parent[cur as usize] != cur {
            let next = parent[cur as usize];
            let gp = parent[next as usize];
            parent[cur as usize] = gp;
            cur = gp;
        }
        cur
    }

    for &row in rref {
        let mut bits = row;
        if bits == 0 {
            continue;
        }
        let first = bits.trailing_zeros();
        bits &= bits - 1;
        while bits != 0 {
            let j = bits.trailing_zeros();
            bits &= bits - 1;
            let ra = find(&mut parent, first);
            let rb = find(&mut parent, j);
            if ra != rb {
                let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
                parent[hi as usize] = lo;
            }
        }
    }

    let support = column_support(rref);
    let mut counts = vec![0u32; n];
    for j in 0..n as u32 {
        if (support >> j) & 1 == 1 {
            counts[find(&mut parent, j) as usize] += 1;
        }
    }
    let mut sizes: Vec<u32> = counts.into_iter().filter(|&c| c > 0).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes
}

/// Sizes (≥ 2 only) of the twin-column classes among support columns,
/// sorted descending: groups of identical k-bit columns of the RREF
/// basis matrix. `Σ (size − 1)` is the number of columns pre-nauty twin
/// compression would remove.
pub(crate) fn twin_class_sizes(rref: &[BinVec], n: u32) -> Vec<u32> {
    let support = column_support(rref);
    // Column signature: bit i of `sig` = row i's entry in this column.
    // k ≤ 16 in every reachable run (rank cap), but build a u32 to be safe.
    let mut sig_counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for j in 0..n {
        if (support >> j) & 1 == 0 {
            continue;
        }
        let mut sig: u32 = 0;
        for (i, &row) in rref.iter().enumerate() {
            sig |= (((row >> j) & 1) as u32) << i;
        }
        *sig_counts.entry(sig).or_insert(0) += 1;
    }
    let mut sizes: Vec<u32> = sig_counts.into_values().filter(|&c| c >= 2).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::row_reduce;

    /// e8: [8,4] extended Hamming, the unique doubly even [8,4] code.
    fn e8_rows() -> Vec<BinVec> {
        vec![0b1111_0000, 0b1100_1100, 0b1010_1010, 0b1111_1111]
    }

    #[test]
    fn e8_is_indecomposable_with_full_support() {
        let (rr, _) = row_reduce(&e8_rows(), 8);
        assert_eq!(column_support(&rr).count_ones(), 8);
        assert_eq!(component_sizes(&rr, 8), vec![8]);
    }

    #[test]
    fn e8_plus_e8_splits_into_two_components() {
        // e8 ⊕ e8 on 16 columns: second copy shifted by 8.
        let mut rows = e8_rows();
        rows.extend(e8_rows().iter().map(|&r| r << 8));
        let (rr, _) = row_reduce(&rows, 16);
        assert_eq!(component_sizes(&rr, 16), vec![8, 8]);
    }

    #[test]
    fn zero_column_shrinks_support_but_not_components() {
        // e8 viewed inside N=10: columns 8, 9 are zero — trivially C' ⊕ 0.
        let (rr, _) = row_reduce(&e8_rows(), 10);
        let support = column_support(&rr);
        assert_eq!(support.count_ones(), 8);
        assert_eq!(10 - support.count_ones(), 2);
        assert_eq!(component_sizes(&rr, 10), vec![8]);
    }

    #[test]
    fn d4_style_repeated_columns_are_twins() {
        // [1111] on 4 columns: all four columns equal (signature 1) —
        // one twin class of size 4. The k=1 analogue of d4's doubled
        // coordinates.
        let rows = vec![0b1111u64 as BinVec];
        assert_eq!(twin_class_sizes(&rows, 4), vec![4]);

        // Doubling every coordinate of e8 ([16,4], all weights doubled —
        // still doubly even): 8 twin pairs.
        let doubled: Vec<BinVec> = e8_rows()
            .iter()
            .map(|&r| {
                let mut d: BinVec = 0;
                for j in 0..8 {
                    if (r >> j) & 1 == 1 {
                        d |= 0b11 << (2 * j);
                    }
                }
                d
            })
            .collect();
        let (rr, _) = row_reduce(&doubled, 16);
        assert_eq!(twin_class_sizes(&rr, 16), vec![2; 8]);
        // Twins force the doubled pairs into single components pairwise,
        // and e8's connectivity glues them all together.
        assert_eq!(component_sizes(&rr, 16), vec![16]);
    }

    #[test]
    fn e8_has_no_twins() {
        let (rr, _) = row_reduce(&e8_rows(), 8);
        assert_eq!(twin_class_sizes(&rr, 8), Vec::<u32>::new());
    }
}
