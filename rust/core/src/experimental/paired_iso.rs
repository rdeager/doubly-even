//! `crate::experimental::paired_iso` — Leon §10(i) paired-refinement
//! isomorphism test for binary linear codes. **EXPERIMENTAL / D12 dormant
//! verifier**, active only under cargo feature `equivalence_verifier`
//! (default OFF); closed 2026-05-20 as the per-probe cost explodes with
//! N (see `project_verifier_dormant.md`). Kept as the worked example of
//! the algorithm and as a possible recovery substrate.
//!
//! Port of `doubly_even.canon.experimental.paired_iso` with the witness
//! permutation π
//! returned. Built on the Feulner partition-refinement primitives so the
//! algorithm lives in one place. The witness is needed by the secondary-
//! cache verifier dispatch in `enumerate.rs`: given a primary-cache miss
//! whose subspace is permutation-equivalent to a bucketed canonical-form
//! `cf`, we reconstruct `CachedInfo` for `D` from cached `cf_info` and π
//! without re-calling nauty.
//!
//! Convention for π: `π[i] = j` means "D-column `i` is sent to cf-column
//! `j`" — same left-action convention as
//! [`crate::linalg::apply_permutation`]. Applying π to D and re-row-
//! reducing yields cf's row-span.

use crate::experimental::feulner::{
    individualise, initial_partition, invariant_refiners, refine, PartialKey, Perm,
};
use crate::permutations::{compute_column_orbits, perm_compose, perm_inverse};
use crate::types::BinVec;

/// Sorted weight multiset of all `2^k` codewords spanned by `rref`. Cheap
/// necessary-condition reject before any partition refinement.
fn weight_multiset(rref: &[BinVec]) -> Vec<u32> {
    let k = rref.len();
    if k == 0 {
        return vec![0];
    }
    let total = 1usize << k;
    let mut weights: Vec<u32> = Vec::with_capacity(total);
    let mut w: BinVec = 0;
    weights.push(0);
    for mask in 1..total {
        let flip = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        w ^= rref[flip];
        weights.push(w.count_ones());
    }
    weights.sort_unstable();
    weights
}

/// Return a witness π such that applying π to `d_rref` and re-row-reducing
/// yields `cf_rref`'s row-span. `None` if D and cf are not permutation-
/// equivalent. See module docs for the π convention.
pub fn paired_iso(d_rref: &[BinVec], cf_rref: &[BinVec], n: u32) -> Option<Perm> {
    if d_rref.len() != cf_rref.len() {
        return None;
    }
    let k = d_rref.len();
    let n_us = n as usize;
    if n == 0 {
        return Some(Vec::new());
    }
    if k == 0 || k == n_us {
        // Whole space (or zero code) on both sides — identity is a witness.
        return Some((0..n).collect());
    }
    if weight_multiset(d_rref) != weight_multiset(cf_rref) {
        return None;
    }
    let cached = PairedIsoCachedCf::new(cf_rref, n);
    paired_iso_with_cached(d_rref, &cached, n)
}

/// cf-side precomputed data, shared across all verifier comparisons against
/// the same canonical form. Each verifier dispatch reuses these allocations,
/// saving `invariant_refiners` + `initial_partition` per call.
pub struct PairedIsoCachedCf {
    pub rref: Vec<BinVec>,
    pub refiners: Vec<BinVec>,
    pub initial_p: Vec<Vec<u32>>,
}

impl PairedIsoCachedCf {
    pub fn new(cf_rref: &[BinVec], n: u32) -> Self {
        Self {
            rref: cf_rref.to_vec(),
            refiners: invariant_refiners(cf_rref),
            initial_p: initial_partition(cf_rref, n),
        }
    }
}

/// Verifier-dispatch entry point — uses cached cf-side state. Skips the
/// weight-multiset prefilter (the bucket key already enforced equality)
/// and reuses the bucket-precomputed refiners + initial partition. The
/// D-side work (refiners, initial partition, partial key) is fresh per call.
pub fn paired_iso_with_cached(
    d_rref: &[BinVec],
    cf: &PairedIsoCachedCf,
    n: u32,
) -> Option<Perm> {
    if d_rref.len() != cf.rref.len() {
        return None;
    }
    let k = d_rref.len();
    let n_us = n as usize;
    if n == 0 {
        return Some(Vec::new());
    }
    if k == 0 || k == n_us {
        return Some((0..n).collect());
    }
    let refiners_d = invariant_refiners(d_rref);
    if refiners_d.len() != cf.refiners.len() {
        return None;
    }
    let p_d = initial_partition(d_rref, n);
    let p_cf = cf.initial_p.clone();
    let mut partial_d = PartialKey::new(d_rref);
    let mut partial_cf = PartialKey::new(&cf.rref);
    paired_search(
        p_d,
        p_cf,
        &refiners_d,
        &cf.refiners,
        &mut partial_d,
        &mut partial_cf,
        n,
    )
}

/// One node of the paired Leon §10(i) search. Returns `Some(π)` at the
/// first matching discrete leaf along this branch, `None` if every
/// descendant branch fails.
///
/// `partial_d` / `partial_cf` are mutated in place and **always restored
/// to their entry state** before returning (mirrors the Feulner
/// `search`'s snap+restore pattern). The witness π returned at a leaf is
/// computed from the cell-positions of `p_d` / `p_cf` and does not depend
/// on the post-restore partial state.
fn paired_search(
    p_d: Vec<Vec<u32>>,
    p_cf: Vec<Vec<u32>>,
    refiners_d: &[BinVec],
    refiners_cf: &[BinVec],
    partial_d: &mut PartialKey,
    partial_cf: &mut PartialKey,
    n: u32,
) -> Option<Perm> {
    let p_d = refine(p_d, refiners_d);
    let p_cf = refine(p_cf, refiners_cf);

    if p_d.len() != p_cf.len() {
        return None;
    }
    for (cd, cc) in p_d.iter().zip(p_cf.iter()) {
        if cd.len() != cc.len() {
            return None;
        }
    }

    let snap_d = partial_d.snapshot();
    let snap_cf = partial_cf.snapshot();
    let mut log_d: Vec<(u32, BinVec)> = Vec::new();
    let mut log_cf: Vec<(u32, BinVec)> = Vec::new();

    // Absorb singletons in lockstep; check the lex prefix prune.
    for (cd, cc) in p_d.iter().zip(p_cf.iter()) {
        if cd.len() == 1 {
            let col_d = cd[0];
            let col_cf = cc[0];
            let new_d = (partial_d.absorbed_cols >> col_d) & 1 == 0;
            let new_cf = (partial_cf.absorbed_cols >> col_cf) & 1 == 0;
            if new_d {
                partial_d.absorb(col_d, &mut log_d);
            }
            if new_cf {
                partial_cf.absorb(col_cf, &mut log_cf);
            }
            if new_d || new_cf {
                let common = partial_d.key.len().min(partial_cf.key.len());
                for i in 0..common {
                    if partial_d.key[i] != partial_cf.key[i] {
                        partial_d.restore(snap_d, &log_d);
                        partial_cf.restore(snap_cf, &log_cf);
                        return None;
                    }
                }
            }
        }
    }

    // Leaf check: both partitions discrete + full keys equal ⇒ witness exists.
    if p_d.iter().all(|c| c.len() == 1) {
        let witness = if partial_d.key == partial_cf.key {
            let mut pi: Perm = vec![0u32; n as usize];
            for (cell_d, cell_cf) in p_d.iter().zip(p_cf.iter()) {
                pi[cell_d[0] as usize] = cell_cf[0];
            }
            Some(pi)
        } else {
            None
        };
        partial_d.restore(snap_d, &log_d);
        partial_cf.restore(snap_cf, &log_cf);
        return witness;
    }

    // Anchor on the lex-smallest D-col; iterate every cf-col it could map to.
    let cell_idx = p_d.iter().position(|c| c.len() > 1).unwrap();
    let col_d = p_d[cell_idx][0];
    let cell_cf_cols: Vec<u32> = p_cf[cell_idx].clone();
    for col_cf in cell_cf_cols {
        let new_p_d = individualise(&p_d, cell_idx, col_d);
        let new_p_cf = individualise(&p_cf, cell_idx, col_cf);
        let witness = paired_search(
            new_p_d,
            new_p_cf,
            refiners_d,
            refiners_cf,
            partial_d,
            partial_cf,
            n,
        );
        if witness.is_some() {
            partial_d.restore(snap_d, &log_d);
            partial_cf.restore(snap_cf, &log_cf);
            return witness;
        }
    }

    partial_d.restore(snap_d, &log_d);
    partial_cf.restore(snap_cf, &log_cf);
    None
}

// ----------------------------------------------- equitable-only prefilter

/// Outcome of the cheap equitable-partition-only iso prefilter.
#[derive(Debug, PartialEq)]
pub enum EquitableResult {
    /// Both refined partitions are all-singletons and their canonical keys
    /// agree positionally. π carries the witness D-col → cf-col.
    Iso(Vec<u32>),
    /// Refinement produced a positional cell-shape disagreement, so the
    /// codes are definitely not permutation-equivalent. Cheaper version of
    /// the same prune `paired_search` performs at every node.
    NotIso,
    /// Refinement converged but at least one side has a non-singleton cell.
    /// The full paired backtrack might still find an iso; here we let the
    /// caller fall through to nauty without spending more work.
    Inconclusive,
}

/// "Weaker but much faster" sibling of `paired_iso`. Runs only the
/// equitable-partition refinement once on each side (no individualisation,
/// no branching) and decides iso iff both sides land at the discrete leaf
/// with matching `PartialKey`. Otherwise returns `Inconclusive` so the
/// caller can fall through to nauty.
///
/// Per-probe cost ≈ one `refine` pair + absorbs ≈ 17 µs at N=18 vs ~70 µs
/// for full `paired_iso`. **Empirical finding for doubly-even codes: hit
/// rate = 0 %** (initial equitable refinement never fully discretises
/// because of non-trivial Aut). Kept as the cheapest possible filter; not
/// useful as a standalone yes-decider for our code shape.
pub fn paired_iso_equitable(
    d_rref: &[BinVec],
    cf: &PairedIsoCachedCf,
    n: u32,
) -> EquitableResult {
    if d_rref.len() != cf.rref.len() {
        return EquitableResult::NotIso;
    }
    let k = d_rref.len();
    let n_us = n as usize;
    if n == 0 {
        return EquitableResult::Iso(Vec::new());
    }
    if k == 0 || k == n_us {
        return EquitableResult::Iso((0..n).collect());
    }
    let refiners_d = invariant_refiners(d_rref);
    if refiners_d.len() != cf.refiners.len() {
        return EquitableResult::NotIso;
    }

    let p_d = refine(initial_partition(d_rref, n), &refiners_d);
    let p_cf = refine(cf.initial_p.clone(), &cf.refiners);

    if p_d.len() != p_cf.len() {
        return EquitableResult::NotIso;
    }
    for (cd, cc) in p_d.iter().zip(p_cf.iter()) {
        if cd.len() != cc.len() {
            return EquitableResult::NotIso;
        }
    }
    if !p_d.iter().all(|c| c.len() == 1) {
        return EquitableResult::Inconclusive;
    }

    let mut partial_d = PartialKey::new(d_rref);
    let mut partial_cf = PartialKey::new(&cf.rref);
    let mut log_d: Vec<(u32, BinVec)> = Vec::new();
    let mut log_cf: Vec<(u32, BinVec)> = Vec::new();
    for (cd, cc) in p_d.iter().zip(p_cf.iter()) {
        partial_d.absorb(cd[0], &mut log_d);
        partial_cf.absorb(cc[0], &mut log_cf);
    }
    if partial_d.key != partial_cf.key {
        return EquitableResult::NotIso;
    }
    let mut pi: Vec<u32> = vec![0u32; n as usize];
    for (cell_d, cell_cf) in p_d.iter().zip(p_cf.iter()) {
        pi[cell_d[0] as usize] = cell_cf[0];
    }
    EquitableResult::Iso(pi)
}

// --------------------------------------------------------- reconstruction

/// `σ_d[i] = σ_cf[π[i]]` — D-col → cf-col → canonical pos.
pub fn reconstruct_canonical_column_order(sigma_cf: &[u32], pi: &[u32]) -> Vec<u32> {
    perm_compose(sigma_cf, pi)
}

/// For each `g ∈ Aut(cf)`, produce `π⁻¹ · g · π ∈ Aut(D)`.
pub fn reconstruct_aut_generators(gens_cf: &[Vec<u32>], pi: &[u32]) -> Vec<Vec<u32>> {
    let pi_inv = perm_inverse(pi);
    gens_cf
        .iter()
        .map(|g| perm_compose(&pi_inv, &perm_compose(g, pi)))
        .collect()
}

/// Recompute column orbits via union-find on the (already-conjugated)
/// generators. Thin wrapper to keep callers from importing `feulner` directly.
pub fn reconstruct_column_orbits(aut_gens_d: &[Vec<u32>], n: u32) -> Vec<u32> {
    compute_column_orbits(aut_gens_d, n)
}

// ----------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{apply_permutation, row_reduce};

    fn rref_of(rows: &[BinVec], n: u32) -> Vec<BinVec> {
        let (rr, _) = row_reduce(rows, n);
        rr
    }

    /// Apply π to each row of `rref` and re-row-reduce.
    fn permute_and_rref(rref: &[BinVec], pi: &[u32], n: u32) -> Vec<BinVec> {
        let permuted: Vec<BinVec> =
            rref.iter().map(|&b| apply_permutation(b, pi)).collect();
        rref_of(&permuted, n)
    }

    /// Apply σ inverse-style: deterministic relabel `i ← σ[i]` to produce
    /// a "permuted copy" of cf in a different basis. (Same operation as
    /// apply_permutation — the convention is symmetric for our purposes.)
    fn make_permuted_copy(cf: &[BinVec], sigma: &[u32], n: u32) -> Vec<BinVec> {
        permute_and_rref(cf, sigma, n)
    }

    /// A standard `[8, 1]` doubly-even code (the weight-8 repetition word).
    fn code_8_1_repetition() -> Vec<BinVec> {
        rref_of(&[0b11111111u64], 8)
    }

    /// A standard `[8, 4]` doubly-even code (the extended Hamming `e8`).
    fn code_8_4_e8() -> Vec<BinVec> {
        rref_of(
            &[
                0b00001111u64,
                0b00110011u64,
                0b01010101u64,
                0b11111111u64,
            ],
            8,
        )
    }

    #[test]
    fn identity_returns_identity_witness() {
        let cf = code_8_4_e8();
        let pi = paired_iso(&cf, &cf, 8).expect("self-iso");
        assert_eq!(permute_and_rref(&cf, &pi, 8), cf);
    }

    #[test]
    fn permuted_copy_round_trips() {
        let cf = code_8_4_e8();
        let sigmas: &[&[u32]] = &[
            &[7, 0, 1, 2, 3, 4, 5, 6],
            &[3, 1, 4, 1, 5, 9, 2, 6], // not a perm; skip — replaced below
            &[2, 5, 1, 7, 0, 3, 6, 4],
            &[0, 1, 2, 3, 4, 5, 6, 7], // identity
        ];
        for sigma in sigmas {
            // Skip the deliberately-malformed entry.
            let mut sorted = sigma.to_vec();
            sorted.sort();
            if sorted != (0u32..8).collect::<Vec<_>>() {
                continue;
            }
            let d_rref = make_permuted_copy(&cf, sigma, 8);
            let pi = paired_iso(&d_rref, &cf, 8)
                .expect("permuted copy must be iso to cf");
            // Verify π is a valid witness.
            assert_eq!(
                permute_and_rref(&d_rref, &pi, 8),
                cf,
                "witness π did not transport d → cf for sigma={:?}",
                sigma
            );
        }
    }

    /// Stress test: many random column perms applied to a non-trivial
    /// `[10, 3]` doubly-even code. Every returned witness must round-trip:
    /// `row_reduce(apply_perm(d, π)) == cf`.
    #[test]
    fn witness_round_trips_under_random_perms() {
        // A [10, 3] code from the actual N=10 enumeration.
        let cf = rref_of(&[771u64, 780u64, 928u64], 10);
        // Simple deterministic PRNG (no extra crates).
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = || -> u64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for trial in 0..200 {
            let mut perm: Vec<u32> = (0..10).collect();
            for i in (1..10).rev() {
                let j = (next() % ((i + 1) as u64)) as usize;
                perm.swap(i, j);
            }
            let d_rref = permute_and_rref(&cf, &perm, 10);
            let pi = paired_iso(&d_rref, &cf, 10).expect("must find iso");
            let round_tripped = permute_and_rref(&d_rref, &pi, 10);
            assert_eq!(
                round_tripped, cf,
                "trial {}: witness π = {:?} did not transport d_rref = {:?} \
                 back to cf = {:?}; got {:?}",
                trial, pi, d_rref, cf, round_tripped
            );
        }
    }

    #[test]
    fn rejects_non_equivalent_codes() {
        // [8,1] repetition vs an [8,1] code with a different weight enum.
        // The trivial weight-4 word [0,1,2,3] is also doubly-even.
        let cf_rep = code_8_1_repetition();
        let cf_w4 = rref_of(&[0b00001111u64], 8);
        assert!(paired_iso(&cf_rep, &cf_w4, 8).is_none());
        assert!(paired_iso(&cf_w4, &cf_rep, 8).is_none());
    }

    #[test]
    fn reconstruct_canonical_column_order_composes_correctly() {
        let sigma_cf: Vec<u32> = vec![3, 0, 1, 2];
        let pi: Vec<u32> = vec![2, 3, 0, 1];
        let sigma_d = reconstruct_canonical_column_order(&sigma_cf, &pi);
        // σ_d[i] = σ_cf[π[i]]: [σ_cf[2], σ_cf[3], σ_cf[0], σ_cf[1]] = [1, 2, 3, 0]
        assert_eq!(sigma_d, vec![1, 2, 3, 0]);
    }

    #[test]
    fn reconstruct_aut_generators_conjugates_correctly() {
        // g = swap(0,1) in cf-coords; π = identity ⇒ conjugate = g unchanged.
        let g: Vec<u32> = vec![1, 0, 2, 3];
        let pi: Vec<u32> = vec![0, 1, 2, 3];
        let conj = reconstruct_aut_generators(&[g.clone()], &pi);
        assert_eq!(conj, vec![g]);

        // π = swap(0,2), g = swap(0,1) ⇒ π⁻¹ ∘ g ∘ π = swap(1, 2).
        let g: Vec<u32> = vec![1, 0, 2, 3];
        let pi: Vec<u32> = vec![2, 1, 0, 3];
        let conj = reconstruct_aut_generators(&[g], &pi);
        // π⁻¹ = pi (it's an involution). For each i, conj[i] = π⁻¹[g[π[i]]]:
        //   i=0: π[0]=2; g[2]=2; π⁻¹[2]=0  -> 0
        //   i=1: π[1]=1; g[1]=0; π⁻¹[0]=2  -> 2
        //   i=2: π[2]=0; g[0]=1; π⁻¹[1]=1  -> 1
        //   i=3: π[3]=3; g[3]=3; π⁻¹[3]=3  -> 3
        // i.e. swap(1, 2).
        assert_eq!(conj, vec![vec![0, 2, 1, 3]]);
    }
}
