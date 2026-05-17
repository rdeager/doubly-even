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
    /// Secondary cache: weight enumerator → list of canonical forms already
    /// computed for that bucket. Instrumentation only — we still call nauty
    /// on every primary miss, but we record whether the new RREF turns out
    /// to be permutation-equivalent to an entry already in its bucket. If
    /// the would-be-hit rate is high, there's room for a real verification
    /// scheme that skips the nauty call when a bucket entry matches.
    secondary_cache: HashMap<Vec<u32>, Vec<Vec<BinVec>>>,
    /// Times we consulted the secondary cache (primary missed AND bucket non-empty).
    pub stats_secondary_attempts: u64,
    /// Times the new RREF's canonical form already lived in the bucket —
    /// i.e., we just paid for a nauty call on something we'd recognised.
    pub stats_secondary_hits: u64,
    /// Whether to populate the secondary cache (env-gated).
    instrument_secondary: bool,
    /// Times `is_canonical_augmentation` was entered.
    pub stats_is_canon_aug_calls: u64,
    /// Times the candidate already equalled `p(D)` (zero-cost branch).
    pub stats_parent_eq_hits: u64,
    /// Times the weight-enum prefilter rejected before BFS.
    pub stats_weight_enum_filtered: u64,
    /// Times the BFS was actually entered.
    pub stats_bfs_calls: u64,
    /// Times the BFS returned `true`.
    pub stats_bfs_hits: u64,
    /// Cumulative ns spent in `is_canonical_augmentation`.
    pub stats_is_canon_aug_ns: u128,
    /// Cumulative ns spent inside `subspace_in_orbit`.
    pub stats_bfs_ns: u128,
    /// Cumulative ns spent inside the nauty / Q_D-graph canon dispatch
    /// (and `aut_order_exact`). Bounds the budget any "cheap equivalence
    /// verifier" must beat to be a net win — see plan file
    /// `let-s-implement-the-plan-nifty-kahn.md` Phase 1.
    pub stats_nauty_ns: u128,
    /// Sum of bucket sizes at secondary-cache attempt time. Env-gated.
    pub stats_bucket_size_sum_at_attempt: u64,
    /// Sum of match positions at secondary-cache *hits* (0-based index
    /// into the bucket). Env-gated. Combined with `stats_secondary_hits`
    /// gives the mean comparisons a verifier would do per hit.
    pub stats_match_position_sum: u64,
    /// Max bucket size seen at attempt time. Env-gated.
    pub stats_max_bucket_size: u64,
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
            secondary_cache: HashMap::new(),
            stats_secondary_attempts: 0,
            stats_secondary_hits: 0,
            instrument_secondary: std::env::var("DOUBLY_EVEN_SECONDARY_CACHE_INSTRUMENTATION")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            stats_is_canon_aug_calls: 0,
            stats_parent_eq_hits: 0,
            stats_weight_enum_filtered: 0,
            stats_bfs_calls: 0,
            stats_bfs_hits: 0,
            stats_is_canon_aug_ns: 0,
            stats_bfs_ns: 0,
            stats_nauty_ns: 0,
            stats_bucket_size_sum_at_attempt: 0,
            stats_match_position_sum: 0,
            stats_max_bucket_size: 0,
            output: Vec::new(),
        }
    }

    /// Compute the canonical-form RREF of `rref` given its canonical column
    /// order: apply σ to each row, then RREF. Two permutation-equivalent
    /// subspaces produce the same canonical form, so this acts as an
    /// equivalence-class identifier.
    fn canonical_form(&self, rref: &[BinVec], canonical_column_order: &[u32]) -> Vec<BinVec> {
        let permuted: Vec<BinVec> = rref
            .iter()
            .map(|&b| apply_permutation(b, canonical_column_order))
            .collect();
        let (rr, _) = row_reduce(&permuted, self.n);
        rr
    }

    /// Compute canon info for the code given by `rref`, or recover from cache.
    fn canon_info(&mut self, rref: &[BinVec]) -> Rc<CachedInfo> {
        // Primary cache.
        if let Some(info) = self.canon_cache.get(rref) {
            self.stats_primary_hits += 1;
            return Rc::clone(info);
        }

        // True miss: call nauty.
        self.stats_canon_calls += 1;

        // Secondary-cache instrumentation: opt-in via the
        // `DOUBLY_EVEN_SECONDARY_CACHE_INSTRUMENTATION=1` env var. When
        // disabled (the default), the recursion has no secondary-cache
        // overhead. When enabled, we count how often a new RREF is
        // permutation-equivalent to one already cached under a different
        // basis — useful for measuring headroom of a future cheap-
        // verification dedup scheme. We still call nauty unconditionally;
        // the counter answers "what fraction of nauty calls land on a code
        // we already canonised under another basis?".
        let we_key = if self.instrument_secondary {
            Some(weight_enum(rref))
        } else {
            None
        };
        let bucket_size_at_attempt: usize = if let Some(key) = we_key.as_ref() {
            let size = self
                .secondary_cache
                .get(key)
                .map(|v| v.len())
                .unwrap_or(0);
            if size > 0 {
                self.stats_secondary_attempts += 1;
                self.stats_bucket_size_sum_at_attempt += size as u64;
                if (size as u64) > self.stats_max_bucket_size {
                    self.stats_max_bucket_size = size as u64;
                }
            }
            size
        } else {
            0
        };
        let bucket_was_nonempty = bucket_size_at_attempt > 0;

        // Time nauty (including `aut_order_exact`, which can fall back to
        // Schreier–Sims for groups past float64 precision). This bounds
        // the cost any cheap equivalence verifier must beat. Always-on:
        // one `Instant::now()` pair per call ≈ 40 ns, negligible vs
        // nauty's tens-of-µs work.
        let nauty_t0 = std::time::Instant::now();
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
        self.stats_nauty_ns += nauty_t0.elapsed().as_nanos();

        if let Some(key) = we_key {
            // Compute canonical form to check secondary-cache membership.
            let canonical = self.canonical_form(rref, &native.canonical_column_order);
            if bucket_was_nonempty {
                if let Some(bucket) = self.secondary_cache.get(&key) {
                    if let Some(pos) = bucket.iter().position(|cf| cf == &canonical) {
                        self.stats_secondary_hits += 1;
                        self.stats_match_position_sum += pos as u64;
                    }
                }
            }
            self.secondary_cache.entry(key).or_default().push(canonical);
        }

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
    ///
    /// Timer instrumentation (Step 0 of Engine B plan): the `&mut self`
    /// signature lets us bump cumulative ns + call counters on each path
    /// through the function. Cost of one `Instant::now()` is ~20 ns on
    /// Linux — negligible against the BFS / canon-form work it bounds.
    fn is_canonical_augmentation(
        &mut self,
        c_rref: &[BinVec],
        d_rref: &[BinVec],
        info_d: &CachedInfo,
    ) -> bool {
        let t0 = std::time::Instant::now();
        self.stats_is_canon_aug_calls += 1;

        let p_d = self.canonical_parent(d_rref, &info_d.canonical_column_order);
        if c_rref == p_d.as_slice() {
            self.stats_parent_eq_hits += 1;
            self.stats_is_canon_aug_ns += t0.elapsed().as_nanos();
            return true;
        }
        // Weight-enum prefilter.
        let we_c = weight_enum(c_rref);
        let we_p = weight_enum(&p_d);
        if we_c != we_p {
            self.stats_weight_enum_filtered += 1;
            self.stats_is_canon_aug_ns += t0.elapsed().as_nanos();
            return false;
        }
        // BFS in the orbit of p_d under Aut(D).
        if info_d.aut_generators.is_empty() {
            self.stats_is_canon_aug_ns += t0.elapsed().as_nanos();
            return false;
        }
        let bfs_t0 = std::time::Instant::now();
        self.stats_bfs_calls += 1;
        let hit = subspace_in_orbit(self.n, c_rref, &p_d, &info_d.aut_generators);
        self.stats_bfs_ns += bfs_t0.elapsed().as_nanos();
        if hit {
            self.stats_bfs_hits += 1;
        }
        self.stats_is_canon_aug_ns += t0.elapsed().as_nanos();
        hit
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
///
/// Stats vector layout (15 u128 fields, packed for pyo3 tuple-arity
/// limits — pyo3 0.23 caps `IntoPyObject` tuples at 12 elements):
///
/// ```text
///  idx  field
///   0   canon_calls
///   1   primary_hits
///   2   secondary_attempts
///   3   secondary_hits
///   4   is_canon_aug_calls
///   5   parent_eq_hits
///   6   weight_enum_filtered
///   7   bfs_calls
///   8   bfs_hits
///   9   is_canon_aug_ns
///  10   bfs_ns
///  11   nauty_ns                       (always-on)
///  12   bucket_size_sum_at_attempt     (env-gated)
///  13   match_position_sum             (env-gated)
///  14   max_bucket_size                (env-gated)
/// ```
///
/// Fields 4–10 came from the Engine B BFS-cost profile (see
/// `markdown/notes/engine-b-bfs-profile.md`); fields 11–14 came from
/// Phase 1 of the cheap-equivalence-verifier plan
/// (`/home/dev/.claude/plans/let-s-implement-the-plan-nifty-kahn.md`).
pub fn enumerate_doubly_even(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
) -> (Vec<EnumeratedRaw>, Vec<u128>) {
    let mut state = State::new(n, max_k, quota, factorial_n);
    // Zero code: rref empty, pivots empty.
    let zero_rref: Vec<BinVec> = Vec::new();
    let zero_pivots: Vec<u32> = Vec::new();
    let info = state.canon_info(&zero_rref);
    state.traverse(zero_rref, zero_pivots, info);
    let stats: Vec<u128> = vec![
        state.stats_canon_calls as u128,
        state.stats_primary_hits as u128,
        state.stats_secondary_attempts as u128,
        state.stats_secondary_hits as u128,
        state.stats_is_canon_aug_calls as u128,
        state.stats_parent_eq_hits as u128,
        state.stats_weight_enum_filtered as u128,
        state.stats_bfs_calls as u128,
        state.stats_bfs_hits as u128,
        state.stats_is_canon_aug_ns,
        state.stats_bfs_ns,
        state.stats_nauty_ns,
        state.stats_bucket_size_sum_at_attempt as u128,
        state.stats_match_position_sum as u128,
        state.stats_max_bucket_size as u128,
    ];
    (state.output, stats)
}
