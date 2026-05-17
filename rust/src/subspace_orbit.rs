//! Subspace-orbit BFS for the McKay parent test.
//!
//! Mirrors `doubly_even.enumerate.augment._in_aut_orbit_of_subspace`:
//! given a starting subspace (by RREF basis), a target subspace, and
//! column-permutation generators of some group `G`, decide whether the
//! target is reachable from the start under the induced `G`-action on
//! subspaces.
//!
//! BFS state is the subspace's RREF basis (a canonical subspace
//! identifier), so equal subspaces with different bases collide on the
//! same hash key. Each step applies one generator to every basis row
//! and re-RREFs the result; the seen-set bounds the orbit at most by
//! `2^rank` distinct subspaces.

use std::collections::HashSet;

use crate::linalg::{apply_permutation, row_reduce};
use crate::types::{BinVec, ColPerm};

/// Return `true` iff some element of `⟨generators⟩` maps `start_rref`
/// (as a subspace of `F_2^n`) to `target_rref`.
///
/// `start_rref` and `target_rref` must already be in RREF (the caller
/// ensures this). The function applies generators to basis rows and
/// re-RREFs the result to obtain the new subspace identifier.
///
/// Returns `false` when generators are empty unless start == target.
pub fn subspace_in_orbit(
    n: u32,
    start_rref: &[BinVec],
    target_rref: &[BinVec],
    generators: &[ColPerm],
) -> bool {
    if start_rref == target_rref {
        return true;
    }
    if generators.is_empty() {
        return false;
    }

    let start_key: Vec<BinVec> = start_rref.to_vec();
    let target_key: &[BinVec] = target_rref;

    let mut seen: HashSet<Vec<BinVec>> = HashSet::new();
    seen.insert(start_key.clone());

    let mut queue: Vec<Vec<BinVec>> = vec![start_key];
    let mut scratch: Vec<BinVec> = Vec::with_capacity(target_rref.len().max(1));

    while !queue.is_empty() {
        let mut next_queue: Vec<Vec<BinVec>> = Vec::new();
        for current in &queue {
            for gen in generators {
                scratch.clear();
                for &b in current {
                    scratch.push(apply_permutation(b, gen));
                }
                let (key, _) = row_reduce(&scratch, n);
                if key.as_slice() == target_key {
                    return true;
                }
                if seen.contains(&key) {
                    continue;
                }
                seen.insert(key.clone());
                next_queue.push(key);
            }
        }
        queue = next_queue;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_start_equals_target() {
        let rref = vec![0b011u64, 0b101u64];
        let gens: Vec<ColPerm> = vec![];
        assert!(subspace_in_orbit(3, &rref, &rref, &gens));
    }

    #[test]
    fn no_gens_no_orbit() {
        let start = vec![0b001u64];
        let target = vec![0b010u64];
        let gens: Vec<ColPerm> = vec![];
        assert!(!subspace_in_orbit(3, &start, &target, &gens));
    }

    #[test]
    fn swap_columns_reaches_target() {
        // Subspace ⟨e_0⟩ = {0, e_0}; under (0 1) it becomes ⟨e_1⟩ = {0, e_1}.
        let start = vec![0b001u64];
        let target = vec![0b010u64];
        // Permutation swap: bit 0 → 1, bit 1 → 0, identity on 2.
        let gen: ColPerm = vec![1, 0, 2];
        assert!(subspace_in_orbit(3, &start, &target, std::slice::from_ref(&gen)));
    }

    #[test]
    fn unreachable_subspace_returns_false() {
        // <e_0> can't reach <e_0+e_1+e_2> under just swap (0 1).
        let start = vec![0b001u64];
        let target = vec![0b111u64];
        let gen: ColPerm = vec![1, 0, 2];
        assert!(!subspace_in_orbit(3, &start, &target, std::slice::from_ref(&gen)));
    }
}
