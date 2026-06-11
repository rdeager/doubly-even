//! Small Schreier-Sims for exact `|Aut|` and dual-basis construction,
//! plus the column-permutation utilities (`perm_compose`, `perm_inverse`,
//! `compute_column_orbits`) shared by the hot path and the dormant
//! Feulner / paired-iso substrate.
//!
//! Direct port of `doubly_even.canon.permutations.group_order` — exact
//! permutation-group order via base-point orbits + Schreier generators.

use std::collections::{HashMap, HashSet};

use crate::types::{BinVec, ColPerm};

/// Compose two column permutations: `(p ∘ q)[i] = p[q[i]]` — apply `q`
/// first, then `p`. Mirrors `doubly_even.canon.permutations.compose`.
pub(crate) fn perm_compose(p: &[u32], q: &[u32]) -> Vec<u32> {
    q.iter().map(|&i| p[i as usize]).collect()
}

/// Inverse of a column permutation.
pub(crate) fn perm_inverse(p: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; p.len()];
    for (i, &j) in p.iter().enumerate() {
        inv[j as usize] = i as u32;
    }
    inv
}

/// Union-find over `n` points: each generator's non-fixed `(i, g[i])`
/// pair gets unioned. Returns the root of each column's orbit.
pub(crate) fn compute_column_orbits(aut_gens: &[Vec<u32>], n: u32) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..n).collect();

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

    fn union(parent: &mut [u32], a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return;
        }
        if ra < rb {
            parent[rb as usize] = ra;
        } else {
            parent[ra as usize] = rb;
        }
    }

    for g in aut_gens {
        for (i, &j) in g.iter().enumerate() {
            if i as u32 != j {
                union(&mut parent, i as u32, j);
            }
        }
    }
    (0..n).map(|i| find(&mut parent, i)).collect()
}

/// Exact order of `⟨generators⟩` acting on `n` points.
///
/// Schreier–Sims with base `(0, 1, …, n-1)`. No sifting; for our `n ≤ 32`
/// codes the generator set is small enough that this is fine.
///
/// Returns `1` for the empty generator set.
pub fn group_order(generators: &[ColPerm], n: u32) -> u128 {
    if generators.is_empty() {
        return 1;
    }
    let mut gens: Vec<Vec<u32>> = generators.iter().cloned().collect();
    let mut order: u128 = 1;
    for base in 0..n {
        // Orbit + transversal of `base` under current generators.
        let identity: Vec<u32> = (0..n).collect();
        let mut orbit: Vec<u32> = vec![base];
        let mut transversal: HashMap<u32, Vec<u32>> = HashMap::new();
        transversal.insert(base, identity.clone());
        let mut queue: Vec<u32> = vec![base];
        while !queue.is_empty() {
            let mut next: Vec<u32> = Vec::new();
            for &p in &queue {
                for g in &gens {
                    let q = g[p as usize];
                    if !transversal.contains_key(&q) {
                        let t_p = transversal[&p].clone();
                        transversal.insert(q, perm_compose(g, &t_p));
                        orbit.push(q);
                        next.push(q);
                    }
                }
            }
            queue = next;
        }
        order *= orbit.len() as u128;
        if orbit.len() == 1 {
            continue;
        }
        // Schreier generators for the stabiliser of `base`.
        let mut new_gens: HashSet<Vec<u32>> = HashSet::new();
        for p in &orbit {
            let t_p = transversal[p].clone();
            for g in &gens {
                let q = g[*p as usize];
                let t_q_inv = perm_inverse(&transversal[&q]);
                let inner = perm_compose(g, &t_p);
                let schreier = perm_compose(&t_q_inv, &inner);
                if schreier != identity {
                    new_gens.insert(schreier);
                }
            }
        }
        if new_gens.is_empty() {
            break;
        }
        gens = new_gens.into_iter().collect();
    }
    order
}

/// Dual basis of a code in RREF: for each free (non-pivot) column `j` emit
/// a vector with bit `j` set, plus bit `pivots[i]` set whenever RREF row
/// `i` has bit `j` set.
///
/// Direct port of `doubly_even.spec.codes._compute_dual_basis`.
pub fn dual_basis(rref: &[BinVec], pivots: &[u32], n: u32) -> Vec<BinVec> {
    let pivot_set: HashSet<u32> = pivots.iter().copied().collect();
    let mut dual: Vec<BinVec> = Vec::with_capacity((n as usize).saturating_sub(rref.len()));
    for j in 0..n {
        if pivot_set.contains(&j) {
            continue;
        }
        let mut v: BinVec = 1u64 << j;
        for (row, &p) in rref.iter().zip(pivots) {
            if (row >> j) & 1 == 1 {
                v |= 1u64 << p;
            }
        }
        dual.push(v);
    }
    dual
}

/// Magnitude cap on the float fast path. Beyond `2^52`, `grpsize1 * 10^k`
/// can round into the wrong integer even when `M < 2^53` is exactly
/// representable, because nauty's `MULTIPLY` macro (`nauty.h:1147`)
/// normalises grpsize1 by `s1 /= 1e10` and rounds to nearest f64 — that
/// ½-ULP error is baked into grpsize1 itself; no reconstruction recovers
/// the bit. The post-rounding divisibility check below catches the
/// remaining off-by-one cases below `2^52` (which arise when f64 round
/// lands on a half-integer at magnitudes M ≥ 2^51 where ULP = 0.5).
const FLOAT_INT_LIMIT_F64: f64 = (1u64 << 52) as f64;

/// Convert nauty's `(grpsize1, grpsize2)` to an exact `u128`, falling back
/// to Schreier-Sims when the float product can't be trusted.
///
/// Trust the float fast path only if all of the following hold:
/// - `grpsize2 <= 22` (`10^k` is exact in f64 only for `k <= 22`)
/// - `raw` is finite, ≥ 1, and below `2^52` (worst-case round error
///   stays under ½ ULP, modulo the half-integer rounding hazard
///   handled next)
/// - the rounded result divides `factorial_n` (Lagrange: `|Aut|` MUST
///   divide `N!`). This catches the half-integer hazard: e.g. for the
///   N=21 [4,1] code, `raw = grpsize1 * 10^15` lands at
///   `8536498274304000.5` exactly, and Rust's half-away-from-zero
///   rounds to `8536498274304001` — but `21! mod 8536498274304001 != 0`,
///   so we fall through to Schreier-Sims which returns the true
///   `8536498274304000` (= 4! · 17!).
pub fn aut_order_exact(
    grpsize1: f64,
    grpsize2: i32,
    aut_generators: &[ColPerm],
    n: u32,
    factorial_n: u128,
) -> u128 {
    if grpsize2 <= 22 {
        let raw = grpsize1 * 10f64.powi(grpsize2);
        if raw.is_finite() && raw >= 1.0 && raw < FLOAT_INT_LIMIT_F64 {
            let candidate = raw.round() as u128;
            if candidate > 0 && factorial_n % candidate == 0 {
                return candidate;
            }
        }
    }
    group_order(aut_generators, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_group_is_order_one() {
        let gens: Vec<ColPerm> = vec![];
        assert_eq!(group_order(&gens, 4), 1);
    }

    #[test]
    fn s3_order_six() {
        // Standard generators of S_3.
        let gens: Vec<ColPerm> = vec![vec![1, 0, 2], vec![1, 2, 0]];
        assert_eq!(group_order(&gens, 3), 6);
    }

    #[test]
    fn s5_order_one_twenty() {
        let gens: Vec<ColPerm> = vec![
            vec![1, 0, 2, 3, 4],
            vec![1, 2, 3, 4, 0],
        ];
        assert_eq!(group_order(&gens, 5), 120);
    }

    #[test]
    fn dual_basis_repetition_code() {
        // Code <11> in F_2^2: dual = <11>.
        let dual = dual_basis(&[0b11], &[0], 2);
        assert_eq!(dual, vec![0b11]);
    }

    #[test]
    fn dual_basis_zero_code() {
        // Zero code: dual = whole space.
        let dual = dual_basis(&[], &[], 4);
        assert_eq!(dual, vec![1, 2, 4, 8]);
    }

    /// Regression: aut_order_exact must round-trip exactly for the
    /// N=21 weight-4 single-codeword code C = ⟨(1)^4 (0)^17⟩, whose
    /// |Aut| = 4! · 17! = 8,536,498,274,304,000. nauty returns
    /// (grpsize1=8.536498274304, grpsize2=15); the float fast path
    /// computes raw = 8536498274304000.5 (exact half-integer) and the
    /// pre-fix code rounded up to ...4001, breaking the mass formula
    /// at N=21 by exactly 1.
    ///
    /// The post-fix code:
    /// - keeps grpsize2 ≤ 22 (10^15 is exact)
    /// - raw < 2^52? NO (8.5e15 > 4.5e15 = 2^52) → falls through to
    ///   Schreier-Sims, which returns the exact 8,536,498,274,304,000
    /// - alternative half-integer cases below 2^52 are caught by the
    ///   Lagrange divisibility check
    ///
    /// We can't drive nauty directly from here, but we can confirm
    /// the function returns the Schreier-Sims order when given the
    /// half-integer-prone float pair, because the divisibility check
    /// (factorial_21 mod 8536498274304001 != 0) forces the fallback.
    #[test]
    fn aut_order_exact_n21_w4_regression() {
        let factorial_21: u128 = (1..=21u128).product();
        // The 4! · 17! generators for ⟨(1)^4 (0)^17⟩: swap any two of
        // the first 4 columns, and any two of the last 17. We exercise
        // a small subset; group_order via Schreier-Sims gives the full
        // order regardless of generator choice.
        let gens: Vec<ColPerm> = vec![
            // S_4 on cols 0..4: transposition + 4-cycle
            (0..21u32).map(|i| if i == 0 { 1 } else if i == 1 { 0 } else { i }).collect(),
            {
                let mut g: Vec<u32> = (0..21u32).collect();
                g[0] = 1; g[1] = 2; g[2] = 3; g[3] = 0;
                g
            },
            // S_17 on cols 4..21: transposition + 17-cycle
            (0..21u32).map(|i| if i == 4 { 5 } else if i == 5 { 4 } else { i }).collect(),
            {
                let mut g: Vec<u32> = (0..21u32).collect();
                for i in 4..20 { g[i] = (i + 1) as u32; }
                g[20] = 4;
                g
            },
        ];
        // Even with the buggy float pair, divisibility forces fallback
        // (21! mod 8_536_498_274_304_001 != 0).
        let order = aut_order_exact(
            8.536498274304, 15, &gens, 21, factorial_21,
        );
        // Expected: 4! · 17! = 24 · 355687428096000 = 8536498274304000.
        assert_eq!(order, 24u128 * (1..=17u128).product::<u128>());
    }

    /// Small-magnitude case: |Aut| well below 2^52 should still take
    /// the float fast path and return the exact value.
    #[test]
    fn aut_order_exact_small_magnitude_fast_path() {
        let factorial_8: u128 = (1..=8u128).product();
        // S_4 × S_4 has order 24·24 = 576.
        let gens: Vec<ColPerm> = vec![];  // unused on fast path
        let order = aut_order_exact(5.76, 2, &gens, 8, factorial_8);
        assert_eq!(order, 576);
    }
}
