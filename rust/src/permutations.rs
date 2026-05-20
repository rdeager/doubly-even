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

/// Float64 exact-integer limit (`2^53`); above this `f64 * 10^i32` can no
/// longer represent every integer exactly.
const FLOAT_INT_LIMIT_F64: f64 = (1u64 << 53) as f64;

/// Convert nauty's `(grpsize1, grpsize2)` to an exact `u128`, falling back
/// to Schreier-Sims when the float product would exceed `2^53`.
pub fn aut_order_exact(
    grpsize1: f64,
    grpsize2: i32,
    aut_generators: &[ColPerm],
    n: u32,
) -> u128 {
    let raw = grpsize1 * 10f64.powi(grpsize2);
    if raw.is_finite() && raw < FLOAT_INT_LIMIT_F64 {
        return raw.round() as u128;
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
}
