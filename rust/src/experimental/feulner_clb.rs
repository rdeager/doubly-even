//! `crate::experimental::feulner_clb` — Jerrum's complete labelled
//! branching (CLB) + Feulner §5.2 Lemma 5.9 topological-sort test.
//! **EXPERIMENTAL / Feulner reference substrate**, driven only by the
//! dormant `feulner.rs` canonicaliser; not reached by the active
//! dispatch.
//!
//! Direct port of the Python staging in
//! `doubly_even.canon.experimental.feulner._LabelledBranching`; the
//! Python module is the correctness oracle for
//! this code. Both mirror Sage's `LabelledBranching` in
//! `sage/groups/perm_gps/partn_ref2/refinement_generic.pyx`. Sage's
//! implementation delegates the stabiliser chain to libgap; we reuse
//! our local Schreier–Sims (same one `feulner::group_order` already
//! drives).
//!
//! ## Data structure
//!
//! Given a permutation group `A ≤ S_n` and a Schreier–Sims chain of `A`
//! on the natural base `(0, 1, …, n-1)`, the CLB stores a parent array
//! `father` with:
//!
//! * `father[j] = i` (`i < j`) ⇔ there is an arc `(i → j)` in the
//!   branching `B` of Feulner §5.2.
//! * `father[j] = u32::MAX` ⇔ `j` is a root of `B` (column `j`
//!   participates in no orbit beyond itself at any chain level).
//!
//! The branching encodes a unique generator set for `A`. Concretely:
//! at level `i` of the Schreier–Sims chain, every `j ≠ i` in the orbit
//! of `i` under the residual group satisfies `father[j] = i`.
//!
//! ## Lemma 5.9 (the prune)
//!
//! Given a current ordered partition `P = [α_0, …, α_m]` (the search
//! node's column partition), define `cell_of[c]` = index of the cell
//! containing column `c`. Then for every arc `(i → j)` in `B`:
//!
//! * The arc is *feasible in `P`* iff `cell_of[i] ≤ cell_of[j]`.
//! * If any arc is infeasible (i.e. `cell_of[j] < cell_of[i]`), the
//!   coset `S_P · π` contains no topological sort of `B`, so the entire
//!   subtree rooted here cannot contain a canonical leaf and can be
//!   pruned.
//!
//! ## Lazy rebuild
//!
//! Generators are pushed via [`LabelledBranching::add_gen`] without
//! immediately rebuilding `father`. The rebuild fires once on the next
//! [`LabelledBranching::has_empty_intersection`] call. This amortises
//! the Schreier–Sims walk across burst-discoveries of generators (the
//! common case: a leaf composes one generator, and the parent loop is
//! about to test the next sibling).

use std::collections::{BTreeSet, HashSet};

use crate::experimental::feulner::{perm_identity, Perm};
use crate::permutations::{perm_compose, perm_inverse};

const ROOT: u32 = u32::MAX;

/// Reuses [`crate::experimental::feulner::orbit_and_transversal`] semantics — a small
/// re-implementation kept private to avoid widening the parent module's
/// API. Same BFS as Python `permutations.orbit_and_transversal`.
fn orbit_and_transversal(
    gens: &[Perm],
    base: u32,
    n: u32,
) -> (Vec<u32>, std::collections::HashMap<u32, Perm>) {
    let mut orbit = vec![base];
    let mut transversal: std::collections::HashMap<u32, Perm> =
        std::collections::HashMap::new();
    transversal.insert(base, perm_identity(n));
    let mut frontier = vec![base];
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &p in &frontier {
            for g in gens {
                let q = g[p as usize];
                if transversal.contains_key(&q) {
                    continue;
                }
                let new_perm = perm_compose(g, &transversal[&p]);
                transversal.insert(q, new_perm);
                orbit.push(q);
                next.push(q);
            }
        }
        frontier = next;
    }
    (orbit, transversal)
}

pub struct LabelledBranching {
    n: u32,
    /// `father[j] = i` means arc `(i → j)`; `ROOT` (`u32::MAX`) means root.
    father: Vec<u32>,
    gens: Vec<Perm>,
    seen: HashSet<Perm>,
    dirty: bool,
}

impl LabelledBranching {
    pub fn new(n: u32) -> Self {
        Self {
            n,
            father: vec![ROOT; n as usize],
            gens: Vec::new(),
            seen: HashSet::new(),
            dirty: false,
        }
    }

    /// Record `g` as a known automorphism; defer the `father[]` rebuild.
    /// No-op if `g` was already inserted.
    pub fn add_gen(&mut self, g: Perm) {
        if self.seen.insert(g.clone()) {
            self.gens.push(g);
            self.dirty = true;
        }
    }

    /// Number of generators accumulated (for diagnostics / tests).
    pub fn num_gens(&self) -> usize {
        self.gens.len()
    }

    /// Cleared if `father[]` has been rebuilt since the last `add_gen`.
    /// `pub(crate)` for use by Rust-side tests.
    #[cfg(test)]
    pub(crate) fn father(&self) -> &[u32] {
        &self.father
    }

    /// Rebuild `father[]` from `gens` via Schreier–Sims on the natural
    /// base `(0, 1, …, n-1)`. Cleared `dirty`.
    fn rebuild_father(&mut self) {
        for f in self.father.iter_mut() {
            *f = ROOT;
        }
        if self.gens.is_empty() {
            self.dirty = false;
            return;
        }
        let id = perm_identity(self.n);
        let mut working: Vec<Perm> = self.gens.clone();
        for base in 0..self.n {
            if working.is_empty() {
                break;
            }
            let (orbit, transversal) = orbit_and_transversal(&working, base, self.n);
            if orbit.len() == 1 {
                continue;
            }
            for &j in &orbit {
                if j != base {
                    self.father[j as usize] = base;
                }
            }
            // Schreier generators for Stab(base).
            let mut next_gens: BTreeSet<Perm> = BTreeSet::new();
            for &p in &orbit {
                let t_p = &transversal[&p];
                for g in &working {
                    let q = g[p as usize];
                    let t_q_inv = perm_inverse(&transversal[&q]);
                    let inner = perm_compose(g, t_p);
                    let sg = perm_compose(&t_q_inv, &inner);
                    if sg != id {
                        next_gens.insert(sg);
                    }
                }
            }
            if next_gens.is_empty() {
                break;
            }
            working = next_gens.into_iter().collect();
        }
        self.dirty = false;
    }

    /// Lemma 5.9: return `true` iff the coset described by `partition`
    /// contains **no** topological sort of `B` and the subtree can be
    /// pruned.
    ///
    /// `partition` is the ordered cell list of the current search node:
    /// column `c` lives in cell `cell_of[c]`, and the pruning condition
    /// is `cell_of[j] < cell_of[i]` for some arc `(i → j)`.
    pub fn has_empty_intersection(&mut self, partition: &[Vec<u32>]) -> bool {
        if self.dirty {
            self.rebuild_father();
        }
        if self.gens.is_empty() {
            return false;
        }
        // Fill cell_of[c] for every column.
        let mut cell_of = vec![0u32; self.n as usize];
        for (ci, cell) in partition.iter().enumerate() {
            for &c in cell {
                cell_of[c as usize] = ci as u32;
            }
        }
        // Walk arcs and apply Lemma 5.9.
        for j in 0..self.n {
            let i = self.father[j as usize];
            if i == ROOT {
                continue;
            }
            if cell_of[j as usize] < cell_of[i as usize] {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_group_never_prunes() {
        let mut clb = LabelledBranching::new(4);
        let part = vec![vec![0u32], vec![1], vec![2], vec![3]];
        assert!(!clb.has_empty_intersection(&part));
    }

    #[test]
    fn s3_father_via_schreier_sims() {
        // S_3 = ⟨(0 1), (1 2)⟩ on 3 points.
        let mut clb = LabelledBranching::new(3);
        clb.add_gen(vec![1, 0, 2]);
        clb.add_gen(vec![0, 2, 1]);
        clb.rebuild_father();
        // Base 0: orbit = {0,1,2} → father[1]=0, father[2]=0.
        // Stab(0) on {1,2}: orbit of 1 = {1,2} → father[2] overwritten to 1.
        assert_eq!(clb.father(), &[ROOT, 0, 1]);
    }

    #[test]
    fn lemma_5_9_natural_order_is_feasible() {
        let mut clb = LabelledBranching::new(3);
        clb.add_gen(vec![1, 0, 2]);
        clb.add_gen(vec![0, 2, 1]);
        // Partition [{0},{1},{2}]: 0 in cell 0, 1 in cell 1, 2 in cell 2.
        // Arcs (0→1) and (1→2) both feasible → no prune.
        let part = vec![vec![0u32], vec![1], vec![2]];
        assert!(!clb.has_empty_intersection(&part));
    }

    #[test]
    fn lemma_5_9_reversed_order_prunes() {
        let mut clb = LabelledBranching::new(3);
        clb.add_gen(vec![1, 0, 2]);
        clb.add_gen(vec![0, 2, 1]);
        // Partition [{2},{1},{0}]: 0 in cell 2, 1 in cell 1, 2 in cell 0.
        // Arc (0→1): cell_of[1]=1 < cell_of[0]=2 → prune.
        let part = vec![vec![2u32], vec![1], vec![0]];
        assert!(clb.has_empty_intersection(&part));
    }

    #[test]
    fn lemma_5_9_same_cell_is_feasible() {
        let mut clb = LabelledBranching::new(3);
        clb.add_gen(vec![1, 0, 2]);
        clb.add_gen(vec![0, 2, 1]);
        // Partition [{0,1,2}]: all in cell 0 → no violation (all arcs OK).
        let part = vec![vec![0u32, 1, 2]];
        assert!(!clb.has_empty_intersection(&part));
    }

    #[test]
    fn lazy_rebuild_after_add_gen() {
        let mut clb = LabelledBranching::new(3);
        // No generators yet — never prunes.
        let part = vec![vec![2u32], vec![1], vec![0]];
        assert!(!clb.has_empty_intersection(&part));
        // Add a generator; next call should trigger rebuild and prune.
        clb.add_gen(vec![1, 0, 2]); // (0 1)
        clb.add_gen(vec![0, 2, 1]); // (1 2)
        assert!(clb.has_empty_intersection(&part));
    }
}
