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
use std::num::NonZeroUsize;
use std::rc::Rc;

use lru::LruCache;

/// Per-worker primary canon cache capacity. Read once at `WorkerState::new`
/// from `DOUBLY_EVEN_CANON_CACHE_CAP` (parsed as a positive integer); else
/// defaults to 500,000 entries, which is ~500 MB at N=26 and keeps a
/// 20-worker run well under a 50 GB cgroup ceiling. Hit rate measured at
/// 3–5 % across N=18–24, so eviction barely affects wall time.
fn canon_cache_capacity() -> NonZeroUsize {
    const DEFAULT_CAP: usize = 500_000;
    let cap = std::env::var("DOUBLY_EVEN_CANON_CACHE_CAP")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_CAP);
    NonZeroUsize::new(cap).expect("canon cache capacity must be non-zero")
}

#[cfg(all(feature = "parallel", feature = "traces_qd"))]
compile_error!(
    "feature `parallel` is incompatible with `traces_qd`: Traces uses \
     non-TLS static work queues and is not thread-safe even with HAVE_TLS=1"
);

use crate::canon::{canon_info_native, canon_info_qd_native, NativeCanonInfo, QD_GRAPH_THRESHOLD};
#[cfg(feature = "dense_qd")]
use crate::canon::canon_info_qd_dense;
#[cfg(feature = "traces_qd")]
use crate::canon::canon_info_qd_traces;
use crate::candidates::doubly_even_candidates_q;
use crate::feulner::{perm_compose, perm_inverse};
use crate::linalg::{apply_permutation, row_reduce};
#[cfg(feature = "equivalence_verifier")]
use crate::paired_iso::{
    paired_iso_equitable, reconstruct_aut_generators,
    reconstruct_canonical_column_order, reconstruct_column_orbits,
    EquitableResult, PairedIsoCachedCf,
};
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
/// Secondary-cache bucket value. The `cached_cf` field is only populated
/// when the `equivalence_verifier` feature is on; otherwise it's `None`
/// and the entry behaves as an instrumentation-only `(canonical_form, info)`
/// pair.
struct BucketEntry {
    canonical: Vec<BinVec>,
    info: Rc<CachedInfo>,
    #[cfg(feature = "equivalence_verifier")]
    cached_cf: Rc<PairedIsoCachedCf>,
}

#[derive(Clone)]
pub(crate) struct CachedInfo {
    pub(crate) canonical_column_order: Vec<u32>,
    pub(crate) aut_generators: Vec<Vec<u32>>,
    pub(crate) aut_order: u128,
    pub(crate) column_orbits: Vec<u32>,
}

/// Class-fingerprint cache entry, keyed externally by `T11(D)`.
///
/// `parent_class_t11 := T11(canonical_parent(D))` is determined by
/// `class(D)` alone — independent of which RREF of D populated the
/// entry. On a future visit with the same `T11(D)`, the caller compares
/// the cached value against its own `hash_C = T11(C)`. A mismatch
/// proves `class(canonical_parent(D)) ≠ class(C)`, so D cannot be a
/// canonical augmentation of C — cheap-reject without invoking nauty.
///
/// Plan: `/home/dev/.claude/plans/implement-and-bench-the-logical-lovelace.md`.
#[cfg(feature = "t11_cache")]
#[derive(Clone, Copy)]
struct ClassEntry {
    parent_class_t11: u64,
}

/// A canonical node captured at the depth-cut frontier of the parallel
/// seeder. By-value `CachedInfo` (rather than `Rc<…>`) so the entire
/// struct is `Send` and can travel through a crossbeam channel into a
/// worker thread. Workers reconstruct an `Rc` on receipt.
#[cfg(feature = "parallel")]
pub(crate) struct SeedFrontier {
    pub rref: Vec<BinVec>,
    pub pivots: Vec<u32>,
    pub info: CachedInfo,
    /// `T11(rref)` of the seed; the worker receives this and threads it
    /// into its `traverse` call as the starting `hash_c`. Always present
    /// (0 when `t11_cache` is off) to avoid cfg-proliferation across the
    /// crossbeam channel boundary.
    pub hash: u64,
}

struct WorkerState {
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    mass_at_k: Vec<u128>,
    factorial_n: u128,
    canon_cache: LruCache<Vec<BinVec>, Rc<CachedInfo>>,
    /// Per-k breakdown of `is_canonical_augmentation` outcomes. Indexed by
    /// the parent rank (i.e., rank of C; the child D has rank k+1). Used
    /// by the σ_Q-orbit-min rejection-rate audit
    /// (plan `for-complete-enumeration-of-proud-meerkat.md` Phase 1).
    /// `parent_eq_hits + bfs_hits` is "the canon test accepted"; the
    /// remainder (`is_canon_aug_calls - parent_eq - bfs_hits - weight_enum_filtered`)
    /// is the BFS-exhausted-rejection count surfaced as
    /// `stats_bfs_rejects_by_k`.
    pub stats_is_canon_aug_calls_by_k: Vec<u64>,
    pub stats_parent_eq_hits_by_k: Vec<u64>,
    pub stats_weight_enum_filtered_by_k: Vec<u64>,
    pub stats_bfs_calls_by_k: Vec<u64>,
    pub stats_bfs_hits_by_k: Vec<u64>,
    pub stats_bfs_rejects_by_k: Vec<u64>,
    /// Mass-stop counters bucketed by parent rank k. Together they tell
    /// us how effective the Gaborit closed-form quota is at pruning the
    /// recursion. Audit added for the "Conway–Pless gluing would fill
    /// mass early" question — see
    /// `for-complete-enumeration-of-proud-meerkat.md`.
    ///
    /// `pre_loop`: fired at the top of `traverse` for the child level
    ///   k+1 (quota already met when we entered this parent — entire
    ///   subtree skipped).
    /// `in_loop`: fired mid-candidate-loop in `traverse` (some children
    ///   processed, then the remaining candidates skipped).
    /// `candidates_total_seen`: total candidates we *generated* via
    ///   `doubly_even_candidates_q` (denominator for skip rates).
    /// `candidates_skipped`: of those, how many were left when mass-stop
    ///   fired mid-loop. `candidates_skipped / candidates_total_seen`
    ///   is the fraction of generated candidate-work the quota avoided.
    pub stats_mass_stop_pre_loop_by_k: Vec<u64>,
    pub stats_mass_stop_in_loop_by_k: Vec<u64>,
    pub stats_candidates_total_seen_by_k: Vec<u64>,
    pub stats_candidates_skipped_by_k: Vec<u64>,
    /// Counter of true cache misses (one nauty call apiece).
    pub stats_canon_calls: u64,
    pub stats_primary_hits: u64,
    /// Secondary cache: weight enumerator → list of (canonical form, cached
    /// `CachedInfo` for that canonical form). Each entry is dedup'd on
    /// insertion. Populated whenever either the `equivalence_verifier`
    /// Cargo feature is on or the env-var instrumentation is set; otherwise
    /// stays empty.
    ///
    /// When the verifier feature is on, this cache is the input to the
    /// paired-iso fast path that skips `nauty` on primary-cache miss.
    secondary_cache: HashMap<Vec<u32>, Vec<BucketEntry>>,
    /// Times we consulted the secondary cache (primary missed AND bucket non-empty).
    pub stats_secondary_attempts: u64,
    /// Times the new RREF's canonical form already lived in the bucket —
    /// i.e., we just paid for a nauty call on something we'd recognised
    /// (only bumped on the nauty-fallback path; under the verifier
    /// feature the hits happen earlier and bump `stats_verifier_hits`).
    pub stats_secondary_hits: u64,
    /// Whether to populate the secondary cache. True when either the
    /// `equivalence_verifier` feature is enabled or the env-var
    /// instrumentation is set.
    maintain_secondary_cache: bool,
    /// Class-fingerprint cache. **Shared (D14-V2)** via `Arc<DashMap>`
    /// so the seeder + every parallel worker feed one cache, avoiding
    /// the ~7× per-worker populate duplication observed in V1
    /// (35,988 vs 5,118 distinct classes at N=22 16t). Keyed by
    /// `T11(D)`; the entry stores the class-invariant
    /// `parent_class_t11 = T11(canonical_parent(D))`. Populated on the
    /// FIRST `canon_info(D)` call (regardless of McKay outcome) so
    /// subsequent siblings can cheap-reject when their parent's T11
    /// disagrees. The populate path uses `entry().match Vacant/Occupied`
    /// so concurrent populates of the same hash collapse to one closure
    /// execution. See `ClassEntry` doc.
    #[cfg(feature = "t11_cache")]
    class_cache: std::sync::Arc<
        dashmap::DashMap<u64, ClassEntry, rustc_hash::FxBuildHasher>,
    >,
    /// Per-N set of known T11 collision hashes (see `t11_blocklist::*`).
    /// Blocklisted hashes are never populated and always fall through
    /// to `canon_info` (two classes could share the key — the cached
    /// `parent_class_t11` would be ambiguous).
    #[cfg(feature = "t11_cache")]
    t11_blocklist: rustc_hash::FxHashSet<u64>,
    /// `false` when `t11_blocklist::t11_blocklist_for_n(n)` returns `None`
    /// (e.g., N=26+ — no audited blocklist available). All T11 work is
    /// skipped by the class-fingerprint dispatch when this is false.
    #[cfg(feature = "t11_cache")]
    t11_enabled: bool,
    /// Times the class-fingerprint cache cheap-rejected a candidate
    /// (parent_class_t11 mismatch → skipped canon_info entirely).
    #[cfg(feature = "t11_cache")]
    pub stats_t11_hits: u64,
    /// Times this worker *reached* the populate path (its local view
    /// said the hash was absent). Under V2's shared cache a concurrent
    /// worker may have inserted first; the closure may not actually
    /// run. Equivalent to V1's `stats_t11_misses`; kept in stats-slot
    /// 27 with the label `t11_cache_populates` for bench-JSON
    /// continuity. Sum across workers ≥ `class_cache.len()`.
    #[cfg(feature = "t11_cache")]
    pub stats_t11_lookup_misses: u64,
    /// Times this worker actually inserted into the shared cache (i.e.,
    /// its `Entry::Vacant` arm fired). Sum across workers equals
    /// `class_cache.len()`; the diff vs `stats_t11_lookup_misses`
    /// shows per-worker duplication. Reported in new stats-slot 31
    /// with label `t11_cache_unique_inserts`.
    #[cfg(feature = "t11_cache")]
    pub stats_t11_populates_inserted: u64,
    /// Times the T11 hash matched a known collision and we forced nauty
    /// (never populated; the entry would be ambiguous).
    #[cfg(feature = "t11_cache")]
    pub stats_t11_blocklist_hits: u64,
    /// Cumulative ns spent inside the class-fingerprint fast path (hash
    /// + lookup + populate).
    #[cfg(feature = "t11_cache")]
    pub stats_t11_ns: u128,
    /// Times the class-fingerprint cache hit with `parent_class_t11`
    /// matching the caller's `hash_c` — fell through to `canon_info`
    /// for the Aut(D)-orbit refinement that the class cache cannot
    /// answer cheaply.
    #[cfg(feature = "t11_cache")]
    pub stats_t11_class_match: u64,
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
    /// Times the BFS was entered but exhausted without finding C in the
    /// orbit of `canonical_parent(D)` — i.e., σ_Q orbit-min survivors that
    /// the final canon test rejects. Equals
    /// `stats_bfs_calls - stats_bfs_hits` but materialised so the Python
    /// harness doesn't have to recompute it.
    pub stats_bfs_rejects: u64,
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
    /// Times the verifier was dispatched (primary miss + non-empty bucket
    /// + feature enabled). Always 0 when the feature is off.
    pub stats_verifier_attempts: u64,
    /// Times the verifier confirmed equivalence and we reused cached info,
    /// skipping `nauty` entirely.
    pub stats_verifier_hits: u64,
    /// Sum of `paired_iso` calls made across all verifier dispatches —
    /// gives mean compares per bucket scan when divided by attempts.
    pub stats_verifier_compares: u64,
    /// Cumulative ns spent inside the verifier path (scan + reconstruct).
    pub stats_verifier_ns: u128,
    /// Times `doubly_even_candidates_q` was invoked.
    pub stats_candidates_q_calls: u64,
    /// Cumulative ns spent inside `doubly_even_candidates_q`. Bounds the
    /// upper limit of any candidate-side optimisation (e.g. flipping the
    /// witt-path dispatch). See plan file
    /// `the-last-several-sessions-scalable-bear.md` Stage 0.
    pub stats_candidates_q_ns: u128,
    /// Sums of nauty `statsblk` tree-shape counters across all
    /// `canon_info_*_native` calls. Each summand is one call; divide by
    /// `stats_canon_calls` for per-call averages. Recorded to decompose
    /// the 78 µs/call cost (Q6 in `expert-review/05-nauty-traces-audit.md`)
    /// without an external profiler: high `numnodes` ⇒ backtrack-heavy
    /// (schreier should help), high `tctotal` with low `numnodes` ⇒
    /// refinement-heavy (tc_level=0 candidate), trivial Aut (numgenerators
    /// ≈ 1, maxlevel ≈ 1 per call) ⇒ FFI/setup-dominated (densenauty
    /// cache win likely).
    pub stats_nauty_numnodes: u64,
    pub stats_nauty_tctotal: u64,
    pub stats_nauty_maxlevel_sum: u64,
    pub stats_nauty_generators_sum: u64,
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

impl WorkerState {
    fn new(
        n: u32,
        max_k: u32,
        quota: Vec<u128>,
        factorial_n: u128,
        #[cfg(feature = "t11_cache")]
        class_cache: std::sync::Arc<
            dashmap::DashMap<u64, ClassEntry, rustc_hash::FxBuildHasher>,
        >,
    ) -> Self {
        let len = (max_k + 1) as usize;
        let env_instrument = std::env::var("DOUBLY_EVEN_SECONDARY_CACHE_INSTRUMENTATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let feature_on = cfg!(feature = "equivalence_verifier");
        Self {
            n,
            max_k,
            quota,
            mass_at_k: vec![0u128; len],
            factorial_n,
            canon_cache: LruCache::new(canon_cache_capacity()),
            stats_is_canon_aug_calls_by_k: vec![0u64; len],
            stats_parent_eq_hits_by_k: vec![0u64; len],
            stats_weight_enum_filtered_by_k: vec![0u64; len],
            stats_bfs_calls_by_k: vec![0u64; len],
            stats_bfs_hits_by_k: vec![0u64; len],
            stats_bfs_rejects_by_k: vec![0u64; len],
            stats_mass_stop_pre_loop_by_k: vec![0u64; len],
            stats_mass_stop_in_loop_by_k: vec![0u64; len],
            stats_candidates_total_seen_by_k: vec![0u64; len],
            stats_candidates_skipped_by_k: vec![0u64; len],
            stats_canon_calls: 0,
            stats_primary_hits: 0,
            secondary_cache: HashMap::new(),
            stats_secondary_attempts: 0,
            stats_secondary_hits: 0,
            maintain_secondary_cache: env_instrument || feature_on,
            stats_is_canon_aug_calls: 0,
            stats_parent_eq_hits: 0,
            stats_weight_enum_filtered: 0,
            stats_bfs_calls: 0,
            stats_bfs_hits: 0,
            stats_bfs_rejects: 0,
            stats_is_canon_aug_ns: 0,
            stats_bfs_ns: 0,
            stats_nauty_ns: 0,
            stats_bucket_size_sum_at_attempt: 0,
            stats_match_position_sum: 0,
            stats_max_bucket_size: 0,
            stats_verifier_attempts: 0,
            stats_verifier_hits: 0,
            stats_verifier_compares: 0,
            stats_verifier_ns: 0,
            stats_candidates_q_calls: 0,
            stats_candidates_q_ns: 0,
            stats_nauty_numnodes: 0,
            stats_nauty_tctotal: 0,
            stats_nauty_maxlevel_sum: 0,
            stats_nauty_generators_sum: 0,
            #[cfg(feature = "t11_cache")]
            class_cache,
            #[cfg(feature = "t11_cache")]
            t11_blocklist: crate::t11_blocklist::t11_blocklist_for_n(n).unwrap_or_default(),
            #[cfg(feature = "t11_cache")]
            t11_enabled: crate::t11_blocklist::t11_blocklist_for_n(n).is_some(),
            #[cfg(feature = "t11_cache")]
            stats_t11_hits: 0,
            #[cfg(feature = "t11_cache")]
            stats_t11_lookup_misses: 0,
            #[cfg(feature = "t11_cache")]
            stats_t11_populates_inserted: 0,
            #[cfg(feature = "t11_cache")]
            stats_t11_blocklist_hits: 0,
            #[cfg(feature = "t11_cache")]
            stats_t11_ns: 0,
            #[cfg(feature = "t11_cache")]
            stats_t11_class_match: 0,
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

        // True miss. Decide whether to maintain the secondary cache for
        // either the verifier dispatch or the env-gated instrumentation.
        self.stats_canon_calls += 1;

        let we_key = if self.maintain_secondary_cache {
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

        // --- Feature-gated paired-iso verifier dispatch ---
        //
        // If the bucket is non-empty, try the Leon §10(i) paired-iso test
        // against each entry. On the first match, reconstruct CachedInfo
        // for D from the cached cf info + witness π — skipping nauty
        // entirely. See `paired_iso.rs` and the design plan
        // `/home/dev/.claude/plans/let-s-implement-the-previous-memoized-simon.md`.
        #[cfg(feature = "equivalence_verifier")]
        {
            if bucket_was_nonempty {
                let v_t0 = std::time::Instant::now();
                self.stats_verifier_attempts += 1;
                let we_key_ref = we_key.as_ref().unwrap();
                // Iterate immutably; if a hit, clone the matching Rc and
                // exit so the bucket borrow is released before we mutate
                // `canon_cache`. Cheaper than snapshotting the whole bucket.
                let mut compares: u64 = 0;
                // Equitable-partition-only prefilter: cheap, but
                // INCONCLUSIVE if refinement doesn't fully discretise.
                // On INCONCLUSIVE the caller falls through to nauty
                // immediately (we don't pay full paired_iso cost).
                let hit: Option<(Vec<BinVec>, Rc<CachedInfo>, Vec<u32>)> = {
                    let bucket = self.secondary_cache.get(we_key_ref).unwrap();
                    let mut found = None;
                    let mut any_inconclusive = false;
                    for entry in bucket {
                        compares += 1;
                        match paired_iso_equitable(rref, &entry.cached_cf, self.n) {
                            EquitableResult::Iso(pi) => {
                                found = Some((
                                    entry.canonical.clone(),
                                    Rc::clone(&entry.info),
                                    pi,
                                ));
                                break;
                            }
                            EquitableResult::NotIso => continue,
                            EquitableResult::Inconclusive => {
                                any_inconclusive = true;
                            }
                        }
                    }
                    let _ = any_inconclusive;
                    found
                };
                self.stats_verifier_compares += compares;
                if let Some((cf, cf_info, pi)) = hit {
                    let sigma_d = reconstruct_canonical_column_order(
                        &cf_info.canonical_column_order,
                        &pi,
                    );
                    let gens_d = reconstruct_aut_generators(
                        &cf_info.aut_generators,
                        &pi,
                    );
                    #[cfg(debug_assertions)]
                    {
                        let recon_cf = self.canonical_form(rref, &sigma_d);
                        if recon_cf != cf {
                            eprintln!(
                                "verifier mismatch: rref={:?} cf_bucket={:?} \
                                 π={:?} σ_cf={:?} σ_d={:?} recon_cf={:?}",
                                rref, cf, pi,
                                cf_info.canonical_column_order, sigma_d, recon_cf
                            );
                            panic!("verifier reconstruction produced wrong canonical form");
                        }
                    }
                    let _ = cf;
                    let orbits_d = reconstruct_column_orbits(&gens_d, self.n);
                    let new_info = Rc::new(CachedInfo {
                        canonical_column_order: sigma_d,
                        aut_generators: gens_d,
                        aut_order: cf_info.aut_order,
                        column_orbits: orbits_d,
                    });
                    self.canon_cache.put(rref.to_vec(), Rc::clone(&new_info));
                    self.stats_verifier_hits += 1;
                    self.stats_verifier_ns += v_t0.elapsed().as_nanos();
                    return new_info;
                }
                self.stats_verifier_ns += v_t0.elapsed().as_nanos();
            }
        }

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
            // Q6 audit Phase 2/3: feature-gated alternative engines for
            // the Q_D-graph canon. `traces_qd` wins over `dense_qd` if both
            // are enabled (Traces is a different algorithm; dense vs sparse
            // is just a representation tweak).
            #[cfg(feature = "traces_qd")]
            {
                canon_info_qd_traces(rref, self.n)
                    .unwrap_or_else(|| canon_info_native(rref, self.n))
            }
            #[cfg(all(feature = "dense_qd", not(feature = "traces_qd")))]
            {
                canon_info_qd_dense(rref, self.n)
                    .unwrap_or_else(|| canon_info_native(rref, self.n))
            }
            #[cfg(not(any(feature = "dense_qd", feature = "traces_qd")))]
            {
                canon_info_qd_native(rref, self.n)
                    .unwrap_or_else(|| canon_info_native(rref, self.n))
            }
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
        self.stats_nauty_numnodes += native.numnodes;
        self.stats_nauty_tctotal += native.tctotal;
        self.stats_nauty_maxlevel_sum += native.maxlevel.max(0) as u64;
        self.stats_nauty_generators_sum += native.numgenerators.max(0) as u64;

        let info = Rc::new(CachedInfo {
            canonical_column_order: native.canonical_column_order,
            aut_generators: native.aut_generators,
            aut_order,
            column_orbits: native.column_orbits,
        });
        self.canon_cache.put(rref.to_vec(), Rc::clone(&info));

        if let Some(key) = we_key {
            // Compute canonical form for secondary-cache membership.
            let canonical = self.canonical_form(rref, &info.canonical_column_order);
            let bucket = self.secondary_cache.entry(key).or_default();
            if let Some(pos) = bucket
                .iter()
                .position(|e| e.canonical == canonical)
            {
                self.stats_secondary_hits += 1;
                self.stats_match_position_sum += pos as u64;
                let _ = bucket_was_nonempty;
            } else {
                // The bucket value carries `CachedInfo` *for the canonical
                // form itself*, not for `rref`. Conjugate `info`'s data into
                // `canonical`'s column frame by the cached σ:
                //
                //   σ_canonical = identity (canonical is already in RREF).
                //   gens_canonical = σ · g · σ⁻¹ for each g ∈ Aut(rref).
                //
                // This makes the verifier reconstruction
                // `σ_d = compose(σ_canonical, π) = π` correct — π is exactly
                // the witness D → canonical from `paired_iso`.
                let sigma: &[u32] = &info.canonical_column_order;
                let sigma_inv = perm_inverse(sigma);
                let gens_canonical: Vec<Vec<u32>> = info
                    .aut_generators
                    .iter()
                    .map(|g| perm_compose(sigma, &perm_compose(g, &sigma_inv)))
                    .collect();
                let orbits_canonical = crate::feulner::compute_column_orbits(
                    &gens_canonical,
                    self.n,
                );
                let info_canonical = Rc::new(CachedInfo {
                    canonical_column_order: (0..self.n).collect(),
                    aut_generators: gens_canonical,
                    aut_order: info.aut_order,
                    column_orbits: orbits_canonical,
                });
                #[cfg(feature = "equivalence_verifier")]
                {
                    let cached_cf = Rc::new(PairedIsoCachedCf::new(&canonical, self.n));
                    bucket.push(BucketEntry {
                        canonical,
                        info: info_canonical,
                        cached_cf,
                    });
                }
                #[cfg(not(feature = "equivalence_verifier"))]
                {
                    bucket.push(BucketEntry {
                        canonical,
                        info: info_canonical,
                    });
                }
            }
        }

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
        parent_k: usize,
        c_rref: &[BinVec],
        d_rref: &[BinVec],
        info_d: &CachedInfo,
    ) -> bool {
        let t0 = std::time::Instant::now();
        self.stats_is_canon_aug_calls += 1;
        self.stats_is_canon_aug_calls_by_k[parent_k] += 1;

        let p_d = self.canonical_parent(d_rref, &info_d.canonical_column_order);
        if c_rref == p_d.as_slice() {
            self.stats_parent_eq_hits += 1;
            self.stats_parent_eq_hits_by_k[parent_k] += 1;
            self.stats_is_canon_aug_ns += t0.elapsed().as_nanos();
            return true;
        }
        // Weight-enum prefilter.
        let we_c = weight_enum(c_rref);
        let we_p = weight_enum(&p_d);
        if we_c != we_p {
            self.stats_weight_enum_filtered += 1;
            self.stats_weight_enum_filtered_by_k[parent_k] += 1;
            self.stats_is_canon_aug_ns += t0.elapsed().as_nanos();
            return false;
        }
        // BFS in the orbit of p_d under Aut(D).
        if info_d.aut_generators.is_empty() {
            // Trivial Aut(D): BFS would be a single-step rejection. Count
            // this as a BFS-style reject for the rejection-rate audit so
            // every non-trivial-orbit candidate that doesn't accept lands
            // somewhere in the per-k breakdown.
            self.stats_bfs_rejects += 1;
            self.stats_bfs_rejects_by_k[parent_k] += 1;
            self.stats_is_canon_aug_ns += t0.elapsed().as_nanos();
            return false;
        }
        let bfs_t0 = std::time::Instant::now();
        self.stats_bfs_calls += 1;
        self.stats_bfs_calls_by_k[parent_k] += 1;
        let hit = subspace_in_orbit(self.n, c_rref, &p_d, &info_d.aut_generators);
        self.stats_bfs_ns += bfs_t0.elapsed().as_nanos();
        if hit {
            self.stats_bfs_hits += 1;
            self.stats_bfs_hits_by_k[parent_k] += 1;
        } else {
            self.stats_bfs_rejects += 1;
            self.stats_bfs_rejects_by_k[parent_k] += 1;
        }
        self.stats_is_canon_aug_ns += t0.elapsed().as_nanos();
        hit
    }

    fn traverse(
        &mut self,
        rref: Vec<BinVec>,
        pivots: Vec<u32>,
        info: Rc<CachedInfo>,
        #[cfg(feature = "t11_cache")] hash_c: u64,
    ) {
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
            self.stats_mass_stop_pre_loop_by_k[k as usize] += 1;
            return;
        }
        // Generate candidates. Time the call: Stage 0 of the Witt-dispatch
        // plan wants the candidates_q / wall ratio at N ∈ {18, 20, 22}.
        let dual = dual_basis(&rref, &pivots, self.n);
        let cq_t0 = std::time::Instant::now();
        let candidates = doubly_even_candidates_q(
            self.n,
            &rref,
            &pivots,
            &dual,
            &info.aut_generators,
        );
        self.stats_candidates_q_ns += cq_t0.elapsed().as_nanos();
        self.stats_candidates_q_calls += 1;
        let total = candidates.len() as u64;
        self.stats_candidates_total_seen_by_k[k as usize] += total;
        for (idx, v) in candidates.iter().enumerate() {
            if self.mass_at_k[k as usize + 1] >= self.quota[k as usize + 1] {
                let remaining = total - idx as u64;
                self.stats_mass_stop_in_loop_by_k[k as usize] += 1;
                self.stats_candidates_skipped_by_k[k as usize] += remaining;
                return;
            }
            // D = C.extend(v): append v, re-RREF.
            let mut new_basis = rref.clone();
            new_basis.push(*v);
            let (d_rref, d_pivots) = row_reduce(&new_basis, self.n);

            // --- CLASS-FINGERPRINT CACHE FAST PATH ---
            // Hash D; on a cache hit where the cached
            // parent_class_t11 disagrees with hash_c, cheap-reject
            // without calling canon_info(D). See `ClassEntry` doc.
            #[cfg(feature = "t11_cache")]
            let hash_d: Option<u64> = if self.t11_enabled {
                let t0 = std::time::Instant::now();
                let h = crate::canon::compute_t11_hash(&d_rref, self.n);
                if self.t11_blocklist.contains(&h) {
                    self.stats_t11_blocklist_hits += 1;
                    self.stats_t11_ns += t0.elapsed().as_nanos();
                    Some(h)
                } else if let Some(entry) = self.class_cache.get(&h).map(|r| *r.value()) {
                    if entry.parent_class_t11 != hash_c {
                        self.stats_t11_hits += 1;
                        self.stats_t11_ns += t0.elapsed().as_nanos();
                        #[cfg(debug_assertions)]
                        {
                            // A cheap-reject must agree with a full
                            // recomputation. If this trips: either a
                            // T11 cross-class collision is missing from
                            // the blocklist, or the populate hook is
                            // broken (sibling RREFs of D would also
                            // disagree). Diagnostic prints the hash so
                            // `dump_t11_blocklist.py` can reproduce.
                            let info_check = self.canon_info(&d_rref);
                            let parent_check = self.canonical_parent(
                                &d_rref,
                                &info_check.canonical_column_order,
                            );
                            let t11_check = crate::canon::compute_t11_hash(
                                &parent_check, self.n,
                            );
                            debug_assert_eq!(
                                entry.parent_class_t11, t11_check,
                                "class-cache reject divergence at hash {:#x}: \
                                 cached {:#x} vs recomputed {:#x}; \
                                 likely missing blocklist entry at N={}",
                                h, entry.parent_class_t11, t11_check, self.n,
                            );
                        }
                        continue;
                    }
                    // Class matches: nauty still needed for the
                    // Aut(D)-orbit refinement in `is_canonical_augmentation`.
                    self.stats_t11_class_match += 1;
                    self.stats_t11_ns += t0.elapsed().as_nanos();
                    Some(h)
                } else {
                    self.stats_t11_ns += t0.elapsed().as_nanos();
                    Some(h)
                }
            } else {
                None
            };

            let info_d = self.canon_info(&d_rref);

            // --- POPULATE ON FIRST canon_info ---
            // Compute parent_class_t11(D) = T11(canonical_parent(D)) and
            // insert. The "populate on first call" timing (vs "on
            // accept") is empirically critical: see prototype memory
            // `project_class_cache_prototype.md` — accept-only delivered
            // only 2.6× at N=22 because non-canonical-parent visits
            // come first.
            #[cfg(feature = "t11_cache")]
            if let Some(h) = hash_d {
                if !self.t11_blocklist.contains(&h) {
                    let t0 = std::time::Instant::now();
                    self.stats_t11_lookup_misses += 1;
                    // D14-V2: `match Entry::{Vacant,Occupied}` collapses
                    // concurrent populates of the same hash to one
                    // closure execution. The Vacant arm holds the shard
                    // write lock for ~95 µs (canonical_parent +
                    // compute_t11_hash); under 4×ncpus = 64 default
                    // shards, same-shard contention is rare.
                    let mut inserted = false;
                    {
                        use dashmap::mapref::entry::Entry as DashEntry;
                        match self.class_cache.entry(h) {
                            DashEntry::Vacant(slot) => {
                                let parent_d = self.canonical_parent(
                                    &d_rref, &info_d.canonical_column_order,
                                );
                                let parent_class_t11 =
                                    crate::canon::compute_t11_hash(
                                        &parent_d, self.n,
                                    );
                                slot.insert(ClassEntry { parent_class_t11 });
                                inserted = true;
                            }
                            DashEntry::Occupied(_) => {
                                // Race lost (or this worker populated
                                // earlier). Nothing to do.
                            }
                        }
                    }
                    if inserted {
                        self.stats_t11_populates_inserted += 1;
                    }
                    self.stats_t11_ns += t0.elapsed().as_nanos();
                }
            }

            if !self.is_canonical_augmentation(k as usize, &rref, &d_rref, &info_d) {
                continue;
            }
            #[cfg(feature = "t11_cache")]
            self.traverse(d_rref, d_pivots, info_d, hash_d.unwrap_or(0));
            #[cfg(not(feature = "t11_cache"))]
            self.traverse(d_rref, d_pivots, info_d);
        }
    }

    /// Variant of [`Self::traverse`] used by the parallel seeder.
    ///
    /// Behaves identically to `traverse` for `k < frontier_depth`. At
    /// `k == frontier_depth` the node is captured into `frontier` (as a
    /// by-value, `Send`-friendly [`SeedFrontier`]) — it is NOT pushed
    /// into `self.output` and recursion stops. The worker that receives
    /// this seed will call `traverse` on it, which will emit the node
    /// and recurse through the rest of the subtree.
    ///
    /// Mass-stop is intentionally disabled during seeding: stopping based
    /// on partial seed mass would preempt valid worker seeds.
    #[cfg(feature = "parallel")]
    fn traverse_seed(
        &mut self,
        rref: Vec<BinVec>,
        pivots: Vec<u32>,
        info: Rc<CachedInfo>,
        frontier_depth: u32,
        frontier: &mut Vec<SeedFrontier>,
        #[cfg(feature = "t11_cache")] hash_c: u64,
    ) {
        let k = rref.len() as u32;
        if k == frontier_depth {
            let info_owned: CachedInfo = match Rc::try_unwrap(info) {
                Ok(info) => info,
                Err(rc) => (*rc).clone(),
            };
            #[cfg(feature = "t11_cache")]
            let seed_hash = hash_c;
            #[cfg(not(feature = "t11_cache"))]
            let seed_hash = 0u64;
            frontier.push(SeedFrontier { rref, pivots, info: info_owned, hash: seed_hash });
            return;
        }
        self.output.push(EnumeratedRaw {
            rref: rref.clone(),
            canonical_column_order: info.canonical_column_order.clone(),
            aut_generators: info.aut_generators.clone(),
            aut_order: info.aut_order,
            column_orbits: info.column_orbits.clone(),
        });
        let mass_contribution = self.factorial_n / info.aut_order;
        self.mass_at_k[k as usize] = self.mass_at_k[k as usize]
            .checked_add(mass_contribution)
            .expect("mass overflow");
        if k >= self.max_k {
            return;
        }
        let dual = dual_basis(&rref, &pivots, self.n);
        let cq_t0 = std::time::Instant::now();
        let candidates = doubly_even_candidates_q(
            self.n,
            &rref,
            &pivots,
            &dual,
            &info.aut_generators,
        );
        self.stats_candidates_q_ns += cq_t0.elapsed().as_nanos();
        self.stats_candidates_q_calls += 1;
        let total = candidates.len() as u64;
        self.stats_candidates_total_seen_by_k[k as usize] += total;
        for v in candidates.iter() {
            let mut new_basis = rref.clone();
            new_basis.push(*v);
            let (d_rref, d_pivots) = row_reduce(&new_basis, self.n);

            // Class-fingerprint fast path — mirrors `traverse`.
            #[cfg(feature = "t11_cache")]
            let hash_d: Option<u64> = if self.t11_enabled {
                let t0 = std::time::Instant::now();
                let h = crate::canon::compute_t11_hash(&d_rref, self.n);
                if self.t11_blocklist.contains(&h) {
                    self.stats_t11_blocklist_hits += 1;
                    self.stats_t11_ns += t0.elapsed().as_nanos();
                    Some(h)
                } else if let Some(entry) = self.class_cache.get(&h).map(|r| *r.value()) {
                    if entry.parent_class_t11 != hash_c {
                        self.stats_t11_hits += 1;
                        self.stats_t11_ns += t0.elapsed().as_nanos();
                        #[cfg(debug_assertions)]
                        {
                            let info_check = self.canon_info(&d_rref);
                            let parent_check = self.canonical_parent(
                                &d_rref,
                                &info_check.canonical_column_order,
                            );
                            let t11_check = crate::canon::compute_t11_hash(
                                &parent_check, self.n,
                            );
                            debug_assert_eq!(
                                entry.parent_class_t11, t11_check,
                                "class-cache reject divergence at hash {:#x}: \
                                 cached {:#x} vs recomputed {:#x}; \
                                 likely missing blocklist entry at N={}",
                                h, entry.parent_class_t11, t11_check, self.n,
                            );
                        }
                        continue;
                    }
                    self.stats_t11_class_match += 1;
                    self.stats_t11_ns += t0.elapsed().as_nanos();
                    Some(h)
                } else {
                    self.stats_t11_ns += t0.elapsed().as_nanos();
                    Some(h)
                }
            } else {
                None
            };

            let info_d = self.canon_info(&d_rref);

            #[cfg(feature = "t11_cache")]
            if let Some(h) = hash_d {
                if !self.t11_blocklist.contains(&h) {
                    let t0 = std::time::Instant::now();
                    self.stats_t11_lookup_misses += 1;
                    // D14-V2: `match Entry::{Vacant,Occupied}` collapses
                    // concurrent populates of the same hash to one
                    // closure execution. The Vacant arm holds the shard
                    // write lock for ~95 µs (canonical_parent +
                    // compute_t11_hash); under 4×ncpus = 64 default
                    // shards, same-shard contention is rare.
                    let mut inserted = false;
                    {
                        use dashmap::mapref::entry::Entry as DashEntry;
                        match self.class_cache.entry(h) {
                            DashEntry::Vacant(slot) => {
                                let parent_d = self.canonical_parent(
                                    &d_rref, &info_d.canonical_column_order,
                                );
                                let parent_class_t11 =
                                    crate::canon::compute_t11_hash(
                                        &parent_d, self.n,
                                    );
                                slot.insert(ClassEntry { parent_class_t11 });
                                inserted = true;
                            }
                            DashEntry::Occupied(_) => {
                                // Race lost (or this worker populated
                                // earlier). Nothing to do.
                            }
                        }
                    }
                    if inserted {
                        self.stats_t11_populates_inserted += 1;
                    }
                    self.stats_t11_ns += t0.elapsed().as_nanos();
                }
            }

            if !self.is_canonical_augmentation(k as usize, &rref, &d_rref, &info_d) {
                continue;
            }
            #[cfg(feature = "t11_cache")]
            self.traverse_seed(
                d_rref, d_pivots, info_d, frontier_depth, frontier,
                hash_d.unwrap_or(0),
            );
            #[cfg(not(feature = "t11_cache"))]
            self.traverse_seed(d_rref, d_pivots, info_d, frontier_depth, frontier);
        }
    }

    /// Consume this WorkerState and produce the `(output, stats, per_k_stats)`
    /// tuple used by both the sequential and parallel drivers. Stats layout
    /// is documented on [`enumerate_doubly_even`].
    fn finalize(self) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>) {
        #[cfg(feature = "t11_cache")]
        let (
            t11_hits,
            t11_lookup_misses,
            t11_blocklist_hits,
            t11_ns,
            t11_class_match,
            t11_inserts,
        ) = (
            self.stats_t11_hits as u128,
            self.stats_t11_lookup_misses as u128,
            self.stats_t11_blocklist_hits as u128,
            self.stats_t11_ns,
            self.stats_t11_class_match as u128,
            self.stats_t11_populates_inserted as u128,
        );
        #[cfg(not(feature = "t11_cache"))]
        let (
            t11_hits,
            t11_lookup_misses,
            t11_blocklist_hits,
            t11_ns,
            t11_class_match,
            t11_inserts,
        ) = (0u128, 0u128, 0u128, 0u128, 0u128, 0u128);

        let stats: Vec<u128> = vec![
            self.stats_canon_calls as u128,
            self.stats_primary_hits as u128,
            self.stats_secondary_attempts as u128,
            self.stats_secondary_hits as u128,
            self.stats_is_canon_aug_calls as u128,
            self.stats_parent_eq_hits as u128,
            self.stats_weight_enum_filtered as u128,
            self.stats_bfs_calls as u128,
            self.stats_bfs_hits as u128,
            self.stats_is_canon_aug_ns,
            self.stats_bfs_ns,
            self.stats_nauty_ns,
            self.stats_bucket_size_sum_at_attempt as u128,
            self.stats_match_position_sum as u128,
            self.stats_max_bucket_size as u128,
            self.stats_verifier_attempts as u128,
            self.stats_verifier_hits as u128,
            self.stats_verifier_compares as u128,
            self.stats_verifier_ns,
            self.stats_candidates_q_calls as u128,
            self.stats_candidates_q_ns,
            self.stats_bfs_rejects as u128,
            self.stats_nauty_numnodes as u128,
            self.stats_nauty_tctotal as u128,
            self.stats_nauty_maxlevel_sum as u128,
            self.stats_nauty_generators_sum as u128,
            t11_hits,
            t11_lookup_misses,
            t11_blocklist_hits,
            t11_ns,
            t11_class_match,
            t11_inserts,
        ];
        let per_k_stats: Vec<Vec<u64>> = vec![
            self.stats_is_canon_aug_calls_by_k,
            self.stats_parent_eq_hits_by_k,
            self.stats_weight_enum_filtered_by_k,
            self.stats_bfs_calls_by_k,
            self.stats_bfs_hits_by_k,
            self.stats_bfs_rejects_by_k,
            self.stats_mass_stop_pre_loop_by_k,
            self.stats_mass_stop_in_loop_by_k,
            self.stats_candidates_total_seen_by_k,
            self.stats_candidates_skipped_by_k,
        ];
        (self.output, stats, per_k_stats)
    }
}

/// Driver: enumerate canonical-augmentation representatives of doubly-even
/// codes of length `n` up to rank `max_k`.
///
/// `quota[k]` must be `σ(N, k)`; `factorial_n` must be `N!`. Both are
/// passed in (Python computes them via `gaborit_sigma` / `math.factorial`).
///
/// Returns `(output, stats, per_k_stats)`:
///
/// - `output` — `Vec<EnumeratedRaw>` in DFS order.
/// - `stats` — flat vector (32 u128 fields). See layout below.
/// - `per_k_stats` — rectangular `[10][max_k+1]` matrix of u64 counters
///   bucketed by the *parent* rank k (i.e., the rank of C; D has rank
///   k+1). Rows in fixed order:
///   `[is_canon_aug_calls, parent_eq_hits, weight_enum_filtered,
///   bfs_calls, bfs_hits, bfs_rejects, mass_stop_pre_loop,
///   mass_stop_in_loop, candidates_total_seen, candidates_skipped]`.
///   Rows 0–5 from Phase 1 of `for-complete-enumeration-of-proud-meerkat.md`
///   (σ_Q-orbit-min rejection-rate audit). Rows 6–9 from the mass-stop
///   audit (same plan, Conway–Pless gluing follow-up).
///
/// Stats vector layout (32 u128 fields, packed for pyo3 tuple-arity
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
///  12   bucket_size_sum_at_attempt     (cache-maintained)
///  13   match_position_sum             (cache-maintained)
///  14   max_bucket_size                (cache-maintained)
///  15   verifier_attempts              (feature equivalence_verifier)
///  16   verifier_hits                  (feature equivalence_verifier)
///  17   verifier_compares              (feature equivalence_verifier)
///  18   verifier_ns                    (feature equivalence_verifier)
///  19   candidates_q_calls             (always-on)
///  20   candidates_q_ns                (always-on)
///  21   bfs_rejects                    (always-on; Phase 1 audit)
///  22   nauty_numnodes_sum             (Q6 audit: backtrack tree size)
///  23   nauty_tctotal_sum              (Q6 audit: total target-cell work)
///  24   nauty_maxlevel_sum             (Q6 audit: deepest level reached)
///  25   nauty_generators_sum           (Q6 audit: Aut generators found)
///  26   t11_cheap_rejects              (feature t11_cache; was `t11_hits`)
///  27   t11_cache_populates            (feature t11_cache; per-worker
///                                       lookup misses — the worker
///                                       reached the populate path.
///                                       Concurrent workers may have
///                                       inserted first.)
///  28   t11_blocklist_hits             (feature t11_cache)
///  29   t11_ns                         (feature t11_cache)
///  30   t11_class_match                (feature t11_cache)
///  31   t11_cache_unique_inserts       (feature t11_cache; D14-V2 —
///                                       per-worker successful inserts
///                                       into the shared cache; sum
///                                       across workers equals
///                                       `class_cache.len()`. Compare
///                                       to slot 27 for duplication
///                                       ratio: V1 ≈ 7×, V2 target 1×.)
/// ```
///
/// Fields 4–10 came from the Engine B BFS-cost profile (see
/// `markdown/notes/engine-b-bfs-profile.md`); 11–14 from Phase 1 of the
/// cheap-equivalence-verifier plan; 15–18 from the verifier-dispatch
/// integration (see
/// `/home/dev/.claude/plans/let-s-implement-the-previous-memoized-simon.md`);
/// 19–20 from Stage 0 of the Witt-dispatch plan
/// (`the-last-several-sessions-scalable-bear.md`); 21 + per_k_stats from
/// Phase 1 of `for-complete-enumeration-of-proud-meerkat.md`.
pub fn enumerate_doubly_even(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>) {
    #[cfg(feature = "t11_cache")]
    let class_cache = std::sync::Arc::new(
        dashmap::DashMap::<u64, ClassEntry, rustc_hash::FxBuildHasher>::default(),
    );
    let mut state = WorkerState::new(
        n,
        max_k,
        quota,
        factorial_n,
        #[cfg(feature = "t11_cache")]
        class_cache,
    );
    // Zero code: rref empty, pivots empty.
    let zero_rref: Vec<BinVec> = Vec::new();
    let zero_pivots: Vec<u32> = Vec::new();
    let info = state.canon_info(&zero_rref);
    #[cfg(feature = "t11_cache")]
    let hash_zero = crate::canon::compute_t11_hash(&zero_rref, n);
    state.traverse(
        zero_rref,
        zero_pivots,
        info,
        #[cfg(feature = "t11_cache")] hash_zero,
    );
    state.finalize()
}

/// Parallel driver: outer-DFS subtree fan-out across a worker pool.
///
/// Traverses sequentially down to `frontier_depth = 3`, then dispatches each
/// accepted depth-3 canonical code as an independent subtree task to one of
/// `num_threads` long-lived workers (crossbeam `bounded` channel). Each worker
/// runs the existing recursion on its assigned seed with its own canon
/// cache and stat counters; mass-stop is disabled inside workers (per-worker
/// quota = `u128::MAX`) so the V1 path loses the global Gaborit quota
/// pruning (4–11 % regression measured per `architecture/04-optimisations.md`
/// §D5). The trade is a coarse-grained parallelism win typically dominant
/// at N ≥ 18.
///
/// V1 conditions for falling back to [`enumerate_doubly_even`]:
///
/// - `num_threads <= 1`, OR
/// - `max_k <= frontier_depth` (no work for workers — the seeder covered it),
/// - any of the above; the function quietly forwards to the sequential
///   driver to avoid the worker-pool overhead.
///
/// Output ordering: not DFS order. Codes of depth `< frontier_depth` appear
/// in DFS order (seeder), followed by per-task concatenated output (worker
/// receive order). Callers that need deterministic order must sort
/// downstream — Python's `augment.enumerate_doubly_even` already turns the
/// raw row stream into a generator and downstream consumers (bench,
/// tests) treat the set as unordered.
///
/// **Safety**: workers call into `sparsenauty` via the bundled
/// `nauty-Traces-sys` build. The `tls` Cargo feature on
/// `nauty-Traces-sys` (`/workspace/src/rust/Cargo.toml:36`) enables
/// `HAVE_TLS=1`, which qualifies sparsenauty's mutable globals
/// (`workspace`, `dnwork`, etc.) with `_Thread_local`. Traces is NOT
/// thread-safe even under `HAVE_TLS=1` — the `traces_qd` feature is
/// barred by `compile_error!` at the top of this file when `parallel`
/// is on.
#[cfg(feature = "parallel")]
pub fn enumerate_doubly_even_parallel(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    num_threads: usize,
) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>) {
    use crossbeam_channel::{bounded, unbounded};

    // D13-V2: deeper frontier breaks the tail. With depth 3 (V1) at N=22
    // the heaviest subtree was ~30 % of total work; bumping the cut to
    // depth 4 splits it into ~3–5 pieces, pushing the ceiling from
    // ~3.3× toward ~6–8×. Configurable via env so heavy users (large N)
    // can tune. Default = 4 at runtime, clamped to ≥ 2.
    let frontier_depth: u32 = std::env::var("DOUBLY_EVEN_FRONTIER_DEPTH")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&d| d >= 2)
        .unwrap_or(4);

    if num_threads <= 1 || max_k <= frontier_depth {
        return enumerate_doubly_even(n, max_k, quota, factorial_n);
    }

    // D14-V2: one shared Arc<DashMap> for the seeder + every worker.
    // Seeder populates depths 0..frontier_depth which are then reused
    // by workers when their subtrees revisit those hashes.
    #[cfg(feature = "t11_cache")]
    let class_cache = std::sync::Arc::new(
        dashmap::DashMap::<u64, ClassEntry, rustc_hash::FxBuildHasher>::default(),
    );

    // Phase 1: sequential seeder, walks depths 0..frontier_depth-1, emits
    // them, and collects the frontier_depth-depth nodes as worker seeds.
    let mut seed_state = WorkerState::new(
        n,
        max_k,
        quota.clone(),
        factorial_n,
        #[cfg(feature = "t11_cache")]
        std::sync::Arc::clone(&class_cache),
    );
    let mut frontier: Vec<SeedFrontier> = Vec::new();
    {
        let zero_rref: Vec<BinVec> = Vec::new();
        let zero_pivots: Vec<u32> = Vec::new();
        let zero_info = seed_state.canon_info(&zero_rref);
        #[cfg(feature = "t11_cache")]
        let hash_zero = crate::canon::compute_t11_hash(&zero_rref, n);
        seed_state.traverse_seed(
            zero_rref,
            zero_pivots,
            zero_info,
            frontier_depth,
            &mut frontier,
            #[cfg(feature = "t11_cache")] hash_zero,
        );
    }
    if frontier.is_empty() {
        // No work past frontier_depth (e.g. N tiny). Already finished.
        return seed_state.finalize();
    }

    // Phase 2: worker pool.
    let cap = (num_threads * 4).max(8);
    let (task_tx, task_rx) = bounded::<SeedFrontier>(cap);
    let (result_tx, result_rx) =
        unbounded::<(Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>)>();

    let mut handles = Vec::with_capacity(num_threads);
    for _ in 0..num_threads {
        let task_rx = task_rx.clone();
        let result_tx = result_tx.clone();
        let mk = max_k;
        let nn = n;
        let fact = factorial_n;
        #[cfg(feature = "t11_cache")]
        let cache_clone = std::sync::Arc::clone(&class_cache);
        handles.push(std::thread::spawn(move || {
            // Per-worker mass-stop disabled: u128::MAX quota means the
            // check never fires. Lose 4–11 % vs sequential mass-stop;
            // gain ≫1× from parallelism. Tracked under V2 of D13.
            let inf_quota = vec![u128::MAX; (mk + 1) as usize];
            let mut worker = WorkerState::new(
                nn,
                mk,
                inf_quota,
                fact,
                #[cfg(feature = "t11_cache")]
                cache_clone,
            );
            while let Ok(seed) = task_rx.recv() {
                #[cfg(feature = "t11_cache")]
                worker.traverse(seed.rref, seed.pivots, Rc::new(seed.info), seed.hash);
                #[cfg(not(feature = "t11_cache"))]
                worker.traverse(seed.rref, seed.pivots, Rc::new(seed.info));
            }
            let _ = result_tx.send(worker.finalize());
        }));
    }
    drop(task_rx);
    drop(result_tx);

    for seed in frontier {
        task_tx.send(seed).expect("worker pool closed unexpectedly");
    }
    drop(task_tx);

    // Drain results.
    let mut combined = seed_state.finalize();
    for _ in 0..num_threads {
        let r = result_rx
            .recv()
            .expect("worker thread hung up before sending result");
        merge_finalized(&mut combined, r);
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }

    // D14-V2 invariant: `sum(stats_t11_populates_inserted)` across
    // seed_state + workers must equal `class_cache.len()`. Each worker
    // increments slot 31 once per Vacant arm; that arm fires once per
    // hash globally. Catches refactor bugs that double-count inserts.
    #[cfg(all(feature = "t11_cache", debug_assertions))]
    {
        let total_inserts: u128 = combined.1[31];
        let actual = class_cache.len() as u128;
        debug_assert_eq!(
            total_inserts, actual,
            "sum(stats_t11_populates_inserted) ({}) must equal \
             class_cache.len() ({})",
            total_inserts, actual,
        );
    }

    combined
}

/// Index of `stats_max_bucket_size` inside the flat stats vector — `.max()`
/// not `+=` on merge. See doc on [`enumerate_doubly_even`] for full layout.
#[cfg(feature = "parallel")]
const STATS_MAX_BUCKET_SIZE_IDX: usize = 14;

#[cfg(feature = "parallel")]
fn merge_finalized(
    a: &mut (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>),
    b: (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>),
) {
    a.0.extend(b.0);
    debug_assert_eq!(a.1.len(), b.1.len(), "stats vector length mismatch");
    for (i, (x, y)) in a.1.iter_mut().zip(b.1.iter()).enumerate() {
        if i == STATS_MAX_BUCKET_SIZE_IDX {
            *x = (*x).max(*y);
        } else {
            *x = x
                .checked_add(*y)
                .expect("stats vector merge overflow");
        }
    }
    debug_assert_eq!(a.2.len(), b.2.len(), "per_k rows count mismatch");
    for (row_a, row_b) in a.2.iter_mut().zip(b.2.iter()) {
        debug_assert_eq!(row_a.len(), row_b.len(), "per_k row length mismatch");
        for (xa, yb) in row_a.iter_mut().zip(row_b.iter()) {
            *xa = xa
                .checked_add(*yb)
                .expect("per_k merge overflow");
        }
    }
}

/// Build identifier — returns `"verifier"` when compiled with the
/// `equivalence_verifier` feature, otherwise `"baseline"`. Used by the
/// Python A/B harness to assert which kernel binary is loaded.
pub fn build_info() -> &'static str {
    if cfg!(feature = "equivalence_verifier") {
        "verifier"
    } else {
        "baseline"
    }
}

#[cfg(test)]
#[cfg(feature = "equivalence_verifier")]
mod verifier_tests {
    use super::*;

    fn factorial(n: u128) -> u128 {
        (1..=n).product()
    }

    fn gaborit_sigma_n10() -> Vec<u128> {
        // σ(10, k) for k = 0..5 from doubly_even.spec.mass.gaborit_sigma.
        vec![1, 255, 5355, 11475, 2295, 0]
    }

    /// Regression: N=10, max_k=3 must emit `1 (k=0) + 2 (k=1) + 3 (k=2) + 3 (k=3) = 9`
    /// canonical classes (per DFGHILM Table 3). Baseline does; ensure verifier
    /// matches. This test catches reconstruction bugs in the bucket-stored
    /// `CachedInfo` — see the algebra fix in `canon_info`.
    #[test]
    fn enumerate_n10_max_k3_count() {
        let quota = gaborit_sigma_n10();
        let (out, _stats, _per_k) = enumerate_doubly_even(10, 3, quota, factorial(10));
        let mut per_k = vec![0usize; 4];
        for e in &out {
            per_k[e.rref.len()] += 1;
        }
        assert_eq!(per_k, vec![1, 2, 3, 3], "per-k emission count mismatch");
        assert_eq!(out.len(), 9);
    }
}
