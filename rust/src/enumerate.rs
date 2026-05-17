//! Native canonical-augmentation enumerator for doubly-even codes.
//!
//! Port of `doubly_even.enumerate.augment._traverse` + helpers. Holds the
//! recursion state, canon-info cache, and mass accumulators inside one
//! struct so the hot loop has no Python ↔ Rust boundary crossings beyond
//! the single entry call.
//!
//! Two caches live here:
//!
//! - **Primary RREF cache**: `HashMap<Vec<BinVec>, CachedInfo>` keyed by
//!   subspace identifier — same shape as the Python LRU.
//! - **Secondary permutation-equivalence cache**: `HashMap<weight_enum,
//!   Vec<(rref, CachedInfo)>>`. On primary miss, scan the bucket; verify
//!   equivalence via [`subspace_orbit::subspace_in_orbit`] and transfer
//!   the cached info through the witnessing permutation.

use std::collections::HashMap;
use std::rc::Rc;

use crate::canon::{canon_info_native, canon_info_qd_native, NativeCanonInfo, QD_GRAPH_THRESHOLD};
use crate::candidates::doubly_even_candidates_q;
use crate::linalg::{apply_permutation, row_reduce};
use crate::permutations::{aut_order_exact, dual_basis};
use crate::subspace_orbit::subspace_in_orbit;
use crate::types::BinVec;

/// Output row for one canonical code emitted by [`enumerate_doubly_even`].
pub struct EnumeratedRaw {
    pub rref: Vec<BinVec>,
    pub canonical_column_order: Vec<u32>,
    pub aut_generators: Vec<Vec<u32>>,
    pub aut_order: u128,
    pub column_orbits: Vec<u32>,
}

/// Internal cache row for canon info; mirrors `EnumeratedRaw` minus the
/// `rref` (which is the cache key). Stored behind `Rc` so cache hits and
/// recursion-state hand-off don't clone the heavy `Vec<Vec<u32>>` aut
/// generators field.
struct CachedInfo {
    canonical_column_order: Vec<u32>,
    aut_generators: Vec<Vec<u32>>,
    aut_order: u128,
    column_orbits: Vec<u32>,
}

struct State {
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    mass_at_k: Vec<u128>,
    factorial_n: u128,
    canon_cache: HashMap<Vec<BinVec>, Rc<CachedInfo>>,
    /// Counter of true cache misses (one nauty call apiece).
    pub stats_canon_calls: u64,
    pub stats_primary_hits: u64,
    output: Vec<EnumeratedRaw>,
}

/// Sorted weight enumerator of the code with the given RREF basis.
fn weight_enum(rref: &[BinVec]) -> Vec<u32> {
    let k = rref.len();
    if k == 0 {
        return vec![0];
    }
    let size = 1usize << k;
    let mut weights: Vec<u32> = Vec::with_capacity(size);
    // Gray-code walk so each step is one XOR.
    let mut w: BinVec = 0;
    weights.push(0);
    for mask in 1..size {
        let flip = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        w ^= rref[flip];
        weights.push(w.count_ones());
    }
    weights.sort_unstable();
    weights
}

/// Compose two column permutations: `(p ∘ q)[i] = p[q[i]]`.
fn compose_perm(p: &[u32], q: &[u32]) -> Vec<u32> {
    q.iter().map(|&i| p[i as usize]).collect()
}

/// Inverse of a column permutation.
fn inverse_perm(p: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; p.len()];
    for (i, &j) in p.iter().enumerate() {
        inv[j as usize] = i as u32;
    }
    inv
}

impl State {
    fn new(n: u32, max_k: u32, quota: Vec<u128>, factorial_n: u128) -> Self {
        let len = (max_k + 1) as usize;
        Self {
            n,
            max_k,
            quota,
            mass_at_k: vec![0u128; len],
            factorial_n,
            canon_cache: HashMap::new(),
            stats_canon_calls: 0,
            stats_primary_hits: 0,
            output: Vec::new(),
        }
    }

    /// Compute canon info for the code given by `rref`, or recover from cache.
    fn canon_info(&mut self, rref: &[BinVec]) -> Rc<CachedInfo> {
        // Primary cache.
        if let Some(info) = self.canon_cache.get(rref) {
            self.stats_primary_hits += 1;
            return Rc::clone(info);
        }

        // True miss: call nauty.
        // (A secondary perm-equivalence cache was tried here, keyed on
        // `weight_enum`. It cannot work as a verification cache: the BFS
        // `Aut(D') · D' = {D'}` because aut generators preserve D' as a
        // subspace — they never reach an equivalent-but-distinct subspace.
        // Without a cheap permutation-equivalence test, there is no path
        // around running nauty.)
        self.stats_canon_calls += 1;
        // Dispatch: when 2^k is large the low-weight-incidence graph (Q_D)
        // beats the full bipartite by a factor that grows with 2^k / |C_low|.
        // For small k the full graph is already cheap, so we skip the span
        // check. See plan file's "Dispatch" section and canon.rs's
        // `QD_GRAPH_THRESHOLD`.
        let native: NativeCanonInfo = if (1u32 << rref.len()) > QD_GRAPH_THRESHOLD {
            canon_info_qd_native(rref, self.n)
                .unwrap_or_else(|| canon_info_native(rref, self.n))
        } else {
            canon_info_native(rref, self.n)
        };
        let aut_order = aut_order_exact(
            native.grpsize1,
            native.grpsize2,
            &native.aut_generators,
            self.n,
        );
        let info = Rc::new(CachedInfo {
            canonical_column_order: native.canonical_column_order,
            aut_generators: native.aut_generators,
            aut_order,
            column_orbits: native.column_orbits,
        });
        self.canon_cache.insert(rref.to_vec(), Rc::clone(&info));
        info
    }

    /// Compute the canonical parent of `D` as a rank-(k-1) subspace.
    /// Mirrors `doubly_even.enumerate.augment.canonical_parent`.
    fn canonical_parent(
        &self,
        d_rref: &[BinVec],
        canonical_column_order: &[u32],
    ) -> Vec<BinVec> {
        // Apply σ to each RREF row.
        let permuted: Vec<BinVec> = d_rref
            .iter()
            .map(|&b| apply_permutation(b, canonical_column_order))
            .collect();
        let (rref_rows, _) = row_reduce(&permuted, self.n);
        // Drop last row.
        if rref_rows.is_empty() {
            return Vec::new();
        }
        let parent_in_canon: &[BinVec] = &rref_rows[..rref_rows.len() - 1];
        // Apply inverse σ.
        let inv_sigma: Vec<u32> = inverse_perm(canonical_column_order);
        let parent_basis: Vec<BinVec> = parent_in_canon
            .iter()
            .map(|&b| apply_permutation(b, &inv_sigma))
            .collect();
        // Re-RREF — parent is now in some basis; canonicalize as subspace.
        let (parent_rref, _) = row_reduce(&parent_basis, self.n);
        parent_rref
    }

    /// True iff `(c, d)` is a McKay-canonical augmentation, given d's canon info.
    fn is_canonical_augmentation(
        &self,
        c_rref: &[BinVec],
        d_rref: &[BinVec],
        info_d: &CachedInfo,
    ) -> bool {
        let p_d = self.canonical_parent(d_rref, &info_d.canonical_column_order);
        if c_rref == p_d.as_slice() {
            return true;
        }
        // Weight-enum prefilter.
        let we_c = weight_enum(c_rref);
        let we_p = weight_enum(&p_d);
        if we_c != we_p {
            return false;
        }
        // BFS in the orbit of p_d under Aut(D).
        if info_d.aut_generators.is_empty() {
            return false;
        }
        subspace_in_orbit(self.n, c_rref, &p_d, &info_d.aut_generators)
    }

    fn traverse(&mut self, rref: Vec<BinVec>, pivots: Vec<u32>, info: Rc<CachedInfo>) {
        let k = rref.len() as u32;
        // Emit. Cloning Vec fields directly into the output row (instead of
        // through Rc) avoids retaining the Rc beyond emission.
        self.output.push(EnumeratedRaw {
            rref: rref.clone(),
            canonical_column_order: info.canonical_column_order.clone(),
            aut_generators: info.aut_generators.clone(),
            aut_order: info.aut_order,
            column_orbits: info.column_orbits.clone(),
        });
        // Update mass.
        let mass_contribution = self.factorial_n / info.aut_order;
        self.mass_at_k[k as usize] = self.mass_at_k[k as usize]
            .checked_add(mass_contribution)
            .expect("mass overflow");
        if self.mass_at_k[k as usize] > self.quota[k as usize] {
            panic!(
                "level-{k} mass {} exceeded quota {}",
                self.mass_at_k[k as usize], self.quota[k as usize]
            );
        }
        if k >= self.max_k {
            return;
        }
        if self.mass_at_k[k as usize + 1] >= self.quota[k as usize + 1] {
            return;
        }
        // Generate candidates.
        let dual = dual_basis(&rref, &pivots, self.n);
        let candidates = doubly_even_candidates_q(
            self.n,
            &rref,
            &pivots,
            &dual,
            &info.aut_generators,
        );
        for v in candidates {
            if self.mass_at_k[k as usize + 1] >= self.quota[k as usize + 1] {
                return;
            }
            // D = C.extend(v): append v, re-RREF.
            let mut new_basis = rref.clone();
            new_basis.push(v);
            let (d_rref, d_pivots) = row_reduce(&new_basis, self.n);
            let info_d = self.canon_info(&d_rref);
            if !self.is_canonical_augmentation(&rref, &d_rref, &info_d) {
                continue;
            }
            self.traverse(d_rref, d_pivots, info_d);
        }
    }
}

/// Driver: enumerate canonical-augmentation representatives of doubly-even
/// codes of length `n` up to rank `max_k`.
///
/// `quota[k]` must be `σ(N, k)`; `factorial_n` must be `N!`. Both are
/// passed in (Python computes them via `gaborit_sigma` / `math.factorial`).
///
/// The result is a `Vec<EnumeratedRaw>` in DFS order.
pub fn enumerate_doubly_even(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
) -> (Vec<EnumeratedRaw>, (u64, u64)) {
    let mut state = State::new(n, max_k, quota, factorial_n);
    // Zero code: rref empty, pivots empty.
    let zero_rref: Vec<BinVec> = Vec::new();
    let zero_pivots: Vec<u32> = Vec::new();
    let info = state.canon_info(&zero_rref);
    state.traverse(zero_rref, zero_pivots, info);
    let stats = (state.stats_canon_calls, state.stats_primary_hits);
    (state.output, stats)
}
