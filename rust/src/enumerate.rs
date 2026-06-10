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

use crate::canon::{canon_info_native, NativeCanonInfo, QD_GRAPH_THRESHOLD};
use crate::qd_graph::canon_info_qd_native;
#[cfg(feature = "dense_qd")]
use crate::experimental::canon_dense_qd::canon_info_qd_dense;
#[cfg(feature = "traces_qd")]
use crate::experimental::canon_traces_qd::canon_info_qd_traces;
use crate::candidates::doubly_even_candidates_q;
use crate::linalg::{apply_permutation, row_reduce};
use crate::parent_rule::{
    phi_cascade_shared, tie_break_parent, ParentRule, PhiOutcome, PhiParentSlot, PhiResult,
};
#[cfg(feature = "equivalence_verifier")]
use crate::experimental::paired_iso::PairedIsoCachedCf;
use crate::permutations::{
    aut_order_exact, compute_column_orbits, dual_basis, perm_compose, perm_inverse,
};
use crate::streaming::BinaryWriter;
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
pub(crate) struct BucketEntry {
    pub(crate) canonical: Vec<BinVec>,
    pub(crate) info: Rc<CachedInfo>,
    #[cfg(feature = "equivalence_verifier")]
    pub(crate) cached_cf: Rc<PairedIsoCachedCf>,
}

#[derive(Clone)]
pub(crate) struct CachedInfo {
    pub(crate) canonical_column_order: Vec<u32>,
    pub(crate) aut_generators: Vec<Vec<u32>>,
    pub(crate) aut_order: u128,
    pub(crate) column_orbits: Vec<u32>,
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
}

/// D13-V4 cut 4: shared cross-worker mass tracker for the parallel path.
///
/// Each worker (and the seeder) atomically adds `N!/|Aut|` to `mass[k]`
/// when it emits a code at rank `k`. Workers then consult
/// `is_full(k+1)` before descending — once any subset of workers has
/// collectively emitted enough mass at rank `k+1` to hit `quota[k+1]`,
/// the remaining workers can prune that subtree just as the sequential
/// mass-stop does. The invariant is one-directional: skipping when
/// `global_mass >= quota` can only over-search relative to the optimum
/// (workers may briefly continue past the tipping point before noticing),
/// never under-search. The class count is still correct.
///
/// `std::sync::Mutex<Vec<u128>>` rather than a per-rank atomic because
/// (a) u128 has no native atomic op on x86-64 and (b) the lock is held
/// for nanoseconds per emission — well under nauty's ≥ 78 µs/call cost.
#[cfg(feature = "parallel")]
pub(crate) struct GlobalMassTracker {
    mass: std::sync::Mutex<Vec<u128>>,
    quota: Vec<u128>,
}

#[cfg(feature = "parallel")]
impl GlobalMassTracker {
    pub(crate) fn new(quota: Vec<u128>) -> Self {
        let len = quota.len();
        Self {
            mass: std::sync::Mutex::new(vec![0u128; len]),
            quota,
        }
    }

    /// Atomically add `delta` to `mass[k]`.
    pub(crate) fn add(&self, k: usize, delta: u128) {
        let mut m = self.mass.lock().expect("global mass tracker poisoned");
        m[k] = m[k].checked_add(delta).expect("global mass overflow");
    }

    /// True iff `mass[k] >= quota[k]`. Returns `false` for out-of-range `k`.
    pub(crate) fn is_full(&self, k: usize) -> bool {
        if k >= self.quota.len() {
            return false;
        }
        let m = self.mass.lock().expect("global mass tracker poisoned");
        m[k] >= self.quota[k]
    }

    /// Snapshot of the per-k mass totals. Called after all workers have
    /// joined; used by the streaming drivers as the in-Rust correctness
    /// gate (`mass[k] == quota[k]` for `k < max_k` must hold, mirroring
    /// the Python-side `sigma_brute` / `gaborit_sigma` assertion the
    /// in-memory path runs post-collection).
    pub(crate) fn snapshot(&self) -> Vec<u128> {
        let m = self.mass.lock().expect("global mass tracker poisoned");
        m.clone()
    }
}

pub(crate) struct WorkerState {
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    mass_at_k: Vec<u128>,
    factorial_n: u128,
    /// D13-V4 cut 4: when `Some`, every emit atomically increments
    /// `global_mass.mass[k]` and the candidate-loop mass-stop checks
    /// consult `global_mass.is_full(k+1)` instead of the local
    /// `self.mass_at_k`. Set by the parallel driver; `None` for the
    /// sequential path (no-op cost when absent).
    #[cfg(feature = "parallel")]
    global_mass: Option<std::sync::Arc<GlobalMassTracker>>,
    /// D16 lever B: helper pool the SEEDER's σ_Q calls fan out onto
    /// (`doubly_even_candidates_q_pooled`). Installed only on the
    /// seeder's `WorkerState` by the parallel drivers and dropped right
    /// after `traverse_seed` returns; workers keep `None` (they are
    /// already saturated). `None` ⇒ exact pre-D16 sequential calls.
    #[cfg(feature = "parallel")]
    seeder_pool: Option<std::sync::Arc<crate::seeder_pool::SeederPool>>,
    /// When true, the two mass-stop branches in [`traverse`] (`mass_at_k[k+1]
    /// >= quota[k+1]` checks before and inside the candidate loop) become
    /// no-ops. Read once from `DOUBLY_EVEN_NO_MASS_STOP` at construction;
    /// kept off the parallel workers regardless because they already have
    /// `quota = u128::MAX`. Ablation knob for the refactor profiling pass.
    skip_mass_stop: bool,
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
    /// Active parent-selection rule (see [`crate::parent_rule`]). Resolved
    /// once per driver invocation and passed in, so the seeder and every
    /// worker of a parallel run agree. `Legacy` = σ-based rule, one canon
    /// call per candidate. `CosetSpectrum` = D15 φ cascade decides most
    /// rejects with no canon call (the >2× lever). `Audit` = legacy
    /// behaviour byte-for-byte plus φ tallies + κ accounting (gate driver:
    /// `scripts/experimental/d15_phi_audit.py`).
    parent_rule: ParentRule,
    /// φ outcome tallies (audit mode only; always 0 otherwise).
    pub stats_phi_reject: u64,
    pub stats_phi_accept_unique: u64,
    pub stats_phi_tie_accept: u64,
    pub stats_phi_tie_reject: u64,
    /// Cumulative ns inside `phi_cascade` (+ tie resolution).
    pub stats_phi_ns: u128,
    /// Sum of strata evaluated per cascade (mean = /candidates).
    pub stats_phi_strata_sum: u64,
    /// Sum of `|M|` at the cascade decision point.
    pub stats_phi_m_size_sum: u64,
    /// Canon-dispatch ns spent on candidates the φ rule would KEEP
    /// (accepts + ties). κ = nauty_ns_kept / nauty_ns is the fraction of
    /// today's canon time the new rule cannot remove — measured in ns,
    /// not call counts, so systematically-expensive kept calls are priced.
    pub stats_nauty_ns_kept: u128,
    /// Per-parent-rank φ outcome tallies (audit mode only).
    pub stats_phi_reject_by_k: Vec<u64>,
    pub stats_phi_accept_unique_by_k: Vec<u64>,
    pub stats_phi_tie_accept_by_k: Vec<u64>,
    pub stats_phi_tie_reject_by_k: Vec<u64>,
    /// Times `doubly_even_candidates_q` was invoked.
    pub stats_candidates_q_calls: u64,
    /// Cumulative ns spent inside `doubly_even_candidates_q`. Bounds the
    /// upper limit of any candidate-side optimisation (e.g. flipping the
    /// witt-path dispatch). See plan file
    /// `the-last-several-sessions-scalable-bear.md` Stage 0.
    pub stats_candidates_q_ns: u128,
    /// Per-rank timing rows (post-D15 profile). Always-on at zero new
    /// timer cost: each bucketed delta is an `Instant` pair that already
    /// exists for the matching aggregate field — bucketing adds one
    /// array store per event. `phi_ns_by_k` and `candidates_q_ns_by_k`
    /// are indexed by PARENT rank (rank of C); `nauty_ns_by_k` is
    /// indexed by the rank of the code being canonised (the child D, so
    /// parent_k + 1 — off-by-one vs the other rows by design, since one
    /// canon result is shared by every candidate probing the same D).
    pub stats_phi_ns_by_k: Vec<u64>,
    pub stats_candidates_q_ns_by_k: Vec<u64>,
    pub stats_nauty_ns_by_k: Vec<u64>,
    /// σ_Q sub-phase ns (`phase_timers` builds; always-present fields,
    /// zero otherwise — the `verifier_ns` precedent). Sum over the five
    /// stages ≤ `stats_candidates_q_ns`: the k=0/k=1 closed-form fast
    /// paths and inter-stage glue are untimed. See
    /// `candidates::phase_timers`.
    pub stats_cq_qbasis_ns: u128,
    pub stats_cq_autimage_ns: u128,
    pub stats_cq_singular_ns: u128,
    pub stats_cq_orbitmin_ns: u128,
    pub stats_cq_lift_sort_ns: u128,
    /// Sampled φ sub-phase ns (`phase_timers` builds; every 64th cascade
    /// per thread is fully timed via the cycle counter). Scale by the
    /// sampling factor at analysis time, reweighting per rank via
    /// `stats_phi_sampled_calls_by_k` (φ cost grows with k, so uniform
    /// scaling would bias low). Diagnostic ±10 %. See
    /// `parent_rule::phi_sample`.
    pub stats_phi_frame_gray_ns: u128,
    pub stats_phi_sort_ns: u128,
    pub stats_phi_first_stratum_ns: u128,
    pub stats_phi_wht_ns: u128,
    pub stats_phi_direct_ns: u128,
    /// How many cascades were fully timed (the sampling weights).
    pub stats_phi_sampled_calls: u64,
    pub stats_phi_sampled_calls_by_k: Vec<u64>,
    /// D16 split-frame φ: ns spent building per-parent shared contexts
    /// (eager C-half tables + lazy per-stratum WHTs). Always-on. NOTE:
    /// a SUBSET of `stats_phi_ns` — builds happen inside the timed
    /// cascade call — so the φ wall share is still just `phi_ns`; this
    /// field splits out the amortised-per-parent component.
    pub stats_phi_ctx_ns: u128,
    pub stats_phi_ctx_ns_by_k: Vec<u64>,
    /// Number of per-parent ctx builds (= φ-tested parents).
    pub stats_phi_ctx_builds: u64,
    /// Cascades whose first-stratum decision needed no per-candidate
    /// WHT (coset-only / C-only stratum fast paths, k = 0 frames).
    pub stats_phi_s1_fastpath: u64,
    /// Cascades decided O(1) on the D17 E-chain at stratum ≥ 2 (v-only
    /// reject, E-restricted amax reject, or chain-filter unique accept).
    /// Slot 48 — formerly the never-wired `phi_killer_rejects` reserve
    /// (the D16 amax reject made the killer pre-check moot).
    pub stats_phi_chain_fastpath: u64,
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
    pub(crate) output: Vec<EnumeratedRaw>,
    /// Streaming output sink. When `Some`, every emit writes
    /// `(k, aut_order, basis)` to the per-worker binary file and skips
    /// the per-class clones (`canonical_column_order`, `aut_generators`,
    /// `column_orbits`) that the in-memory `EnumeratedRaw` would carry.
    /// `None` is the legacy in-memory path used by the small-N test
    /// suite and bench harness. Set via [`Self::install_output_writer`].
    pub(crate) output_writer: Option<BinaryWriter<std::fs::File>>,
}

/// `D = ⟨C, v⟩`: append `v` to the parent basis and re-RREF.
fn extend_rref(rref: &[BinVec], v: BinVec, n: u32) -> (Vec<BinVec>, Vec<u32>) {
    let mut new_basis = Vec::with_capacity(rref.len() + 1);
    new_basis.extend_from_slice(rref);
    new_basis.push(v);
    row_reduce(&new_basis, n)
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

impl WorkerState {
    pub(crate) fn new(
        n: u32,
        max_k: u32,
        quota: Vec<u128>,
        factorial_n: u128,
        parent_rule: ParentRule,
    ) -> Self {
        let len = (max_k + 1) as usize;
        let env_instrument = std::env::var("DOUBLY_EVEN_SECONDARY_CACHE_INSTRUMENTATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let feature_on = cfg!(feature = "equivalence_verifier");
        let skip_mass_stop = std::env::var("DOUBLY_EVEN_NO_MASS_STOP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        Self {
            n,
            max_k,
            quota,
            mass_at_k: vec![0u128; len],
            factorial_n,
            skip_mass_stop,
            #[cfg(feature = "parallel")]
            global_mass: None,
            #[cfg(feature = "parallel")]
            seeder_pool: None,
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
            parent_rule,
            stats_phi_reject: 0,
            stats_phi_accept_unique: 0,
            stats_phi_tie_accept: 0,
            stats_phi_tie_reject: 0,
            stats_phi_ns: 0,
            stats_phi_strata_sum: 0,
            stats_phi_m_size_sum: 0,
            stats_nauty_ns_kept: 0,
            stats_phi_reject_by_k: vec![0u64; len],
            stats_phi_accept_unique_by_k: vec![0u64; len],
            stats_phi_tie_accept_by_k: vec![0u64; len],
            stats_phi_tie_reject_by_k: vec![0u64; len],
            stats_candidates_q_calls: 0,
            stats_candidates_q_ns: 0,
            stats_phi_ns_by_k: vec![0u64; len],
            stats_candidates_q_ns_by_k: vec![0u64; len],
            stats_nauty_ns_by_k: vec![0u64; len],
            stats_cq_qbasis_ns: 0,
            stats_cq_autimage_ns: 0,
            stats_cq_singular_ns: 0,
            stats_cq_orbitmin_ns: 0,
            stats_cq_lift_sort_ns: 0,
            stats_phi_frame_gray_ns: 0,
            stats_phi_sort_ns: 0,
            stats_phi_first_stratum_ns: 0,
            stats_phi_wht_ns: 0,
            stats_phi_direct_ns: 0,
            stats_phi_sampled_calls: 0,
            stats_phi_sampled_calls_by_k: vec![0u64; len],
            stats_phi_ctx_ns: 0,
            stats_phi_ctx_ns_by_k: vec![0u64; len],
            stats_phi_ctx_builds: 0,
            stats_phi_s1_fastpath: 0,
            stats_phi_chain_fastpath: 0,
            stats_nauty_numnodes: 0,
            stats_nauty_tctotal: 0,
            stats_nauty_maxlevel_sum: 0,
            stats_nauty_generators_sum: 0,
            output: Vec::new(),
            output_writer: None,
        }
    }

    /// Attach a per-worker binary streaming sink. After this, every
    /// emit writes the compact `(k, aut_order, basis)` triple to the
    /// sink instead of pushing an `EnumeratedRaw` into `self.output`.
    /// The streaming and in-memory paths are mutually exclusive —
    /// callers wire either one or the other.
    pub(crate) fn install_output_writer(
        &mut self,
        writer: BinaryWriter<std::fs::File>,
    ) {
        self.output_writer = Some(writer);
    }

    /// Snapshot of `mass_at_k` (cumulative `Σ N!/|Aut|` per rank). Used
    /// by the sequential streaming driver as the input to the in-Rust
    /// mass-formula assertion (the parallel driver uses
    /// `GlobalMassTracker::snapshot` instead — same role, cross-worker).
    pub(crate) fn mass_snapshot(&self) -> Vec<u128> {
        self.mass_at_k.clone()
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
    pub(crate) fn canon_info(&mut self, rref: &[BinVec]) -> Rc<CachedInfo> {
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

        // Feature-gated paired-iso verifier dispatch — see
        // `crate::experimental::verifier_dispatch`. The helper walks the
        // bucket, runs the equitable-partition prefilter, and on a hit
        // returns a freshly reconstructed CachedInfo for `rref` (bypasses
        // nauty entirely). The caller just updates stats and short-circuits.
        #[cfg(feature = "equivalence_verifier")]
        if bucket_was_nonempty {
            self.stats_verifier_attempts += 1;
            let bucket = self
                .secondary_cache
                .get(we_key.as_ref().unwrap())
                .unwrap();
            let outcome =
                crate::experimental::verifier_dispatch::try_dispatch(rref, self.n, bucket);
            self.stats_verifier_compares += outcome.compares;
            self.stats_verifier_ns += outcome.elapsed_ns;
            if let Some(new_info) = outcome.hit {
                #[cfg(debug_assertions)]
                {
                    let recon_cf =
                        self.canonical_form(rref, &new_info.canonical_column_order);
                    let expected_cf =
                        outcome.hit_cf.as_ref().expect("hit_cf set when hit Some");
                    assert_eq!(
                        &recon_cf, expected_cf,
                        "verifier reconstruction produced wrong canonical form"
                    );
                }
                self.canon_cache.put(rref.to_vec(), Rc::clone(&new_info));
                self.stats_verifier_hits += 1;
                return new_info;
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
            self.factorial_n,
        );
        let nauty_delta = nauty_t0.elapsed().as_nanos();
        self.stats_nauty_ns += nauty_delta;
        self.stats_nauty_ns_by_k[rref.len()] += nauty_delta as u64;
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
                let orbits_canonical = compute_column_orbits(&gens_canonical, self.n);
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
        let inv_sigma: Vec<u32> = perm_inverse(canonical_column_order);
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

    /// Drain the σ_Q sub-phase accumulator after a
    /// `doubly_even_candidates_q` call (`phase_timers` builds; no-op
    /// otherwise). Per-call drain keeps attribution on the right thread
    /// and WorkerState.
    #[cfg(feature = "phase_timers")]
    #[inline]
    fn drain_cq_phase_timers(&mut self) {
        let ph = crate::candidates::phase_timers::drain();
        self.stats_cq_qbasis_ns += ph[0] as u128;
        self.stats_cq_autimage_ns += ph[1] as u128;
        self.stats_cq_singular_ns += ph[2] as u128;
        self.stats_cq_orbitmin_ns += ph[3] as u128;
        self.stats_cq_lift_sort_ns += ph[4] as u128;
    }

    #[cfg(not(feature = "phase_timers"))]
    #[inline(always)]
    fn drain_cq_phase_timers(&mut self) {}

    /// Collect the sampled φ sub-phase split of the cascade that just
    /// returned, if that call was the 1-in-64 fully-timed one
    /// (`phase_timers` builds; no-op otherwise).
    #[cfg(feature = "phase_timers")]
    #[inline]
    fn drain_phi_sample(&mut self, parent_k: usize) {
        if let Some(ph) = crate::parent_rule::phi_sample::take_last() {
            self.stats_phi_frame_gray_ns += ph[0] as u128;
            self.stats_phi_sort_ns += ph[1] as u128;
            self.stats_phi_first_stratum_ns += ph[2] as u128;
            self.stats_phi_wht_ns += ph[3] as u128;
            self.stats_phi_direct_ns += ph[4] as u128;
            self.stats_phi_sampled_calls += 1;
            self.stats_phi_sampled_calls_by_k[parent_k] += 1;
        }
    }

    #[cfg(not(feature = "phase_timers"))]
    #[inline(always)]
    fn drain_phi_sample(&mut self, _parent_k: usize) {}

    /// Decide one candidate augmentation `(C → D = ⟨C, v⟩)` under the
    /// active parent rule. Returns the child's `(rref, pivots, canon
    /// info)` when the augmentation is canonical; `None` to skip. On the
    /// coset-spectrum Reject path no RREF, no cache probe, and no canon
    /// call happen at all — that is the D15 lever (φ costs a Gray sweep
    /// plus per-stratum WHTs, ~1–4 µs, against the ~30–75 µs canon call
    /// the legacy rule pays on every reject).
    ///
    /// Shared by `traverse` and `traverse_seed` so the seeder and the
    /// workers apply the identical rule.
    fn test_candidate(
        &mut self,
        parent_k: usize,
        rref: &[BinVec],
        v: BinVec,
        phi: &mut PhiParentSlot,
    ) -> Option<(Vec<BinVec>, Vec<u32>, Rc<CachedInfo>)> {
        let use_phi = match self.parent_rule {
            ParentRule::CosetSpectrum { max_rank } => parent_k as u32 + 1 <= max_rank,
            _ => false,
        };
        if use_phi {
            let t0 = std::time::Instant::now();
            let res = phi_cascade_shared(phi, rref, v, self.n);
            let phi_delta = t0.elapsed().as_nanos();
            self.stats_phi_ns += phi_delta;
            self.stats_phi_ns_by_k[parent_k] += phi_delta as u64;
            self.drain_phi_sample(parent_k);
            self.drain_phi_ctx_stats(parent_k, phi, &res);
            self.stats_phi_strata_sum += res.strata_used as u64;
            self.stats_phi_m_size_sum += res.m_size_at_decision as u64;
            return match res.outcome {
                PhiOutcome::Reject => {
                    self.stats_phi_reject += 1;
                    self.stats_phi_reject_by_k[parent_k] += 1;
                    None
                }
                PhiOutcome::AcceptUnique => {
                    // m(D) is the orbit of C itself; canon is still paid
                    // here because the recursion needs Aut(D) + |Aut(D)|
                    // (the same cost any rule pays on accepts).
                    self.stats_phi_accept_unique += 1;
                    self.stats_phi_accept_unique_by_k[parent_k] += 1;
                    let (d_rref, d_pivots) = extend_rref(rref, v, self.n);
                    let info_d = self.canon_info(&d_rref);
                    Some((d_rref, d_pivots, info_d))
                }
                PhiOutcome::Tie(m_set) => {
                    let (d_rref, d_pivots) = extend_rref(rref, v, self.n);
                    let info_d = self.canon_info(&d_rref);
                    let t1 = std::time::Instant::now();
                    let target = tie_break_parent(
                        rref,
                        v,
                        self.n,
                        &m_set,
                        &info_d.canonical_column_order,
                    );
                    let accept =
                        subspace_in_orbit(self.n, rref, &target, &info_d.aut_generators);
                    let tie_delta = t1.elapsed().as_nanos();
                    self.stats_phi_ns += tie_delta;
                    self.stats_phi_ns_by_k[parent_k] += tie_delta as u64;
                    if accept {
                        self.stats_phi_tie_accept += 1;
                        self.stats_phi_tie_accept_by_k[parent_k] += 1;
                        Some((d_rref, d_pivots, info_d))
                    } else {
                        self.stats_phi_tie_reject += 1;
                        self.stats_phi_tie_reject_by_k[parent_k] += 1;
                        None
                    }
                }
            };
        }
        // Legacy path (also taken by CosetSpectrum above its rank cap);
        // Audit additionally tallies the φ outcome post-hoc.
        let phi_res = self.phi_audit_evaluate(parent_k, rref, v, phi);
        let (d_rref, d_pivots) = extend_rref(rref, v, self.n);
        let nauty_ns_before = self.stats_nauty_ns;
        let info_d = self.canon_info(&d_rref);
        let accept = self.is_canonical_augmentation(parent_k, rref, &d_rref, &info_d);
        if let Some(res) = phi_res {
            let delta = self.stats_nauty_ns - nauty_ns_before;
            self.phi_audit_resolve(parent_k, rref, v, res, &info_d, delta);
        }
        if accept {
            Some((d_rref, d_pivots, info_d))
        } else {
            None
        }
    }

    /// D15 Phase 1 audit: run the φ cascade for one candidate, timed.
    /// `None` unless audit mode is on. Behaviour-neutral — the result is
    /// only tallied by [`Self::phi_audit_resolve`] after the legacy path.
    #[inline]
    fn phi_audit_evaluate(
        &mut self,
        parent_k: usize,
        c_rref: &[BinVec],
        v: BinVec,
        phi: &mut PhiParentSlot,
    ) -> Option<PhiResult> {
        if self.parent_rule != ParentRule::Audit {
            return None;
        }
        let t0 = std::time::Instant::now();
        let res = phi_cascade_shared(phi, c_rref, v, self.n);
        let phi_delta = t0.elapsed().as_nanos();
        self.stats_phi_ns += phi_delta;
        self.stats_phi_ns_by_k[parent_k] += phi_delta as u64;
        self.drain_phi_sample(parent_k);
        self.drain_phi_ctx_stats(parent_k, phi, &res);
        Some(res)
    }

    /// D16: drain the per-parent ctx build accounting from the slot and
    /// tally the first-stratum fast-path hit, right after a cascade call.
    #[inline]
    fn drain_phi_ctx_stats(
        &mut self,
        parent_k: usize,
        phi: &mut PhiParentSlot,
        res: &PhiResult,
    ) {
        let (ctx_ns, ctx_builds) = phi.take_build_stats();
        if ctx_ns != 0 {
            self.stats_phi_ctx_ns += ctx_ns as u128;
            self.stats_phi_ctx_ns_by_k[parent_k] += ctx_ns;
        }
        self.stats_phi_ctx_builds += ctx_builds;
        if res.s1_fastpath {
            self.stats_phi_s1_fastpath += 1;
        }
        if res.chain_fastpath {
            self.stats_phi_chain_fastpath += 1;
        }
    }

    /// D15 Phase 1 audit: tally a cascade result against the per-candidate
    /// ground truth, resolving φ-ties with the `info_d` the legacy path
    /// already paid for. `nauty_ns_delta` is the canon-dispatch time this
    /// candidate cost (0 on a cache hit) — accumulated into
    /// `stats_nauty_ns_kept` for φ-kept candidates only, so
    /// κ = kept/total prices exactly the canon time the φ rule retains.
    fn phi_audit_resolve(
        &mut self,
        parent_k: usize,
        c_rref: &[BinVec],
        v: BinVec,
        res: PhiResult,
        info_d: &CachedInfo,
        nauty_ns_delta: u128,
    ) {
        self.stats_phi_strata_sum += res.strata_used as u64;
        self.stats_phi_m_size_sum += res.m_size_at_decision as u64;
        match res.outcome {
            PhiOutcome::Reject => {
                self.stats_phi_reject += 1;
                self.stats_phi_reject_by_k[parent_k] += 1;
            }
            PhiOutcome::AcceptUnique => {
                self.stats_phi_accept_unique += 1;
                self.stats_phi_accept_unique_by_k[parent_k] += 1;
                self.stats_nauty_ns_kept += nauty_ns_delta;
            }
            PhiOutcome::Tie(m_set) => {
                let t0 = std::time::Instant::now();
                let target = tie_break_parent(
                    c_rref,
                    v,
                    self.n,
                    &m_set,
                    &info_d.canonical_column_order,
                );
                let accept =
                    subspace_in_orbit(self.n, c_rref, &target, &info_d.aut_generators);
                let tie_delta = t0.elapsed().as_nanos();
                self.stats_phi_ns += tie_delta;
                self.stats_phi_ns_by_k[parent_k] += tie_delta as u64;
                if accept {
                    self.stats_phi_tie_accept += 1;
                    self.stats_phi_tie_accept_by_k[parent_k] += 1;
                } else {
                    self.stats_phi_tie_reject += 1;
                    self.stats_phi_tie_reject_by_k[parent_k] += 1;
                }
                self.stats_nauty_ns_kept += nauty_ns_delta;
            }
        }
    }

    /// D13-V4 cut 4: shared mass-stop predicate. When `global_mass` is
    /// `Some` (parallel path), consult the cross-worker counter; otherwise
    /// fall back to the local `mass_at_k` (sequential, byte-identical
    /// behaviour to pre-V4).
    #[inline]
    fn next_rank_full(&self, k: usize) -> bool {
        #[cfg(feature = "parallel")]
        if let Some(gm) = self.global_mass.as_ref() {
            return gm.is_full(k + 1);
        }
        self.mass_at_k[k + 1] >= self.quota[k + 1]
    }

    /// D13-V4 cut 4: wire a worker (or the seeder) into the shared
    /// cross-worker mass tracker. Called by `enumerate_doubly_even_parallel`
    /// before the worker pulls its first seed.
    #[cfg(feature = "parallel")]
    pub(crate) fn install_global_mass(&mut self, gm: std::sync::Arc<GlobalMassTracker>) {
        self.global_mass = Some(gm);
    }

    /// D16 lever B: wire the seeder helper pool into this state (seeder
    /// only). Cleared via [`Self::clear_seeder_pool`] before the run's
    /// worker-dominated tail so helper threads exit promptly.
    #[cfg(feature = "parallel")]
    pub(crate) fn install_seeder_pool(
        &mut self,
        pool: std::sync::Arc<crate::seeder_pool::SeederPool>,
    ) {
        self.seeder_pool = Some(pool);
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn clear_seeder_pool(&mut self) {
        self.seeder_pool = None;
    }

    pub(crate) fn traverse(&mut self, rref: Vec<BinVec>, pivots: Vec<u32>, info: Rc<CachedInfo>) {
        let k = rref.len() as u32;
        // Emit. Streaming path skips the per-class clones of
        // `canonical_column_order`, `aut_generators`, `column_orbits` —
        // the merge script reconstructs everything it needs from rref +
        // aut_order. In-memory path keeps the legacy shape.
        if let Some(w) = self.output_writer.as_mut() {
            w.write_class(info.aut_order, &rref)
                .expect("BinaryWriter::write_class failed");
        } else {
            self.output.push(EnumeratedRaw {
                rref: rref.clone(),
                canonical_column_order: info.canonical_column_order.clone(),
                aut_generators: info.aut_generators.clone(),
                aut_order: info.aut_order,
                column_orbits: info.column_orbits.clone(),
            });
        }
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
        // D13-V4 cut 4: contribute to the shared cross-worker mass counter
        // so peer workers can mass-stop based on the global total.
        #[cfg(feature = "parallel")]
        if let Some(gm) = self.global_mass.as_ref() {
            gm.add(k as usize, mass_contribution);
        }
        if k >= self.max_k {
            return;
        }
        if !self.skip_mass_stop && self.next_rank_full(k as usize) {
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
        let cq_delta = cq_t0.elapsed().as_nanos();
        self.stats_candidates_q_ns += cq_delta;
        self.stats_candidates_q_ns_by_k[k as usize] += cq_delta as u64;
        self.drain_cq_phase_timers();
        self.stats_candidates_q_calls += 1;
        let total = candidates.len() as u64;
        self.stats_candidates_total_seen_by_k[k as usize] += total;
        // D16: one shared φ context per parent, amortised across the
        // sibling candidates below (built lazily on the first cascade).
        let mut phi_slot = PhiParentSlot::new();
        for (idx, v) in candidates.iter().enumerate() {
            if !self.skip_mass_stop && self.next_rank_full(k as usize) {
                let remaining = total - idx as u64;
                self.stats_mass_stop_in_loop_by_k[k as usize] += 1;
                self.stats_candidates_skipped_by_k[k as usize] += remaining;
                return;
            }
            // D = C.extend(v), accepted or skipped under the active
            // parent rule (φ rejects never touch RREF/cache/canon).
            let Some((d_rref, d_pivots, info_d)) =
                self.test_candidate(k as usize, &rref, *v, &mut phi_slot)
            else {
                continue;
            };
            self.traverse(d_rref, d_pivots, info_d);
        }
    }

    /// Variant of [`Self::traverse`] used by the parallel seeder.
    ///
    /// Behaves identically to `traverse` for `k < frontier_depth`. At
    /// `k == frontier_depth` the node is sent on `task_tx` (as a by-value,
    /// `Send`-friendly [`SeedFrontier`]) — it is NOT pushed into
    /// `self.output` and recursion stops. The worker that receives this
    /// seed will call `traverse` on it, emitting the node and recursing
    /// through the rest of the subtree.
    ///
    /// Pipelined: workers consume seeds from the channel as the seeder
    /// discovers them, overlapping seeder DFS with worker recursion. See
    /// `architecture/07-parallel-scaling-profile.md` for the Amdahl
    /// analysis that motivated this design.
    ///
    /// Mass-stop is intentionally disabled during seeding: stopping based
    /// on partial seed mass would preempt valid worker seeds.
    #[cfg(feature = "parallel")]
    pub(crate) fn traverse_seed(
        &mut self,
        rref: Vec<BinVec>,
        pivots: Vec<u32>,
        info: Rc<CachedInfo>,
        frontier_depth: u32,
        task_tx: &crossbeam_channel::Sender<SeedFrontier>,
    ) {
        let k = rref.len() as u32;
        if k == frontier_depth {
            let info_owned: CachedInfo = match Rc::try_unwrap(info) {
                Ok(info) => info,
                Err(rc) => (*rc).clone(),
            };
            task_tx
                .send(SeedFrontier { rref, pivots, info: info_owned })
                .expect("worker pool closed before seeder finished");
            return;
        }
        if let Some(w) = self.output_writer.as_mut() {
            w.write_class(info.aut_order, &rref)
                .expect("BinaryWriter::write_class failed (seeder)");
        } else {
            self.output.push(EnumeratedRaw {
                rref: rref.clone(),
                canonical_column_order: info.canonical_column_order.clone(),
                aut_generators: info.aut_generators.clone(),
                aut_order: info.aut_order,
                column_orbits: info.column_orbits.clone(),
            });
        }
        let mass_contribution = self.factorial_n / info.aut_order;
        self.mass_at_k[k as usize] = self.mass_at_k[k as usize]
            .checked_add(mass_contribution)
            .expect("mass overflow");
        // D13-V4 cut 4: seeder emissions also contribute to the shared
        // mass counter so workers' mass-stop sees the full picture.
        // (Seeder itself does not mass-stop; see comment on traverse_seed.)
        if let Some(gm) = self.global_mass.as_ref() {
            gm.add(k as usize, mass_contribution);
        }
        if k >= self.max_k {
            return;
        }
        let dual = dual_basis(&rref, &pivots, self.n);
        // NOTE: with the helper pool active, this seeder-side
        // candidates_q_ns delta records WALL time of a parallel region,
        // not CPU time — that shrinkage is the point of lever B.
        let cq_t0 = std::time::Instant::now();
        let candidates = match self.seeder_pool.as_deref() {
            Some(pool) => crate::candidates::doubly_even_candidates_q_pooled(
                self.n,
                &rref,
                &pivots,
                &dual,
                &info.aut_generators,
                pool,
            ),
            None => doubly_even_candidates_q(
                self.n,
                &rref,
                &pivots,
                &dual,
                &info.aut_generators,
            ),
        };
        let cq_delta = cq_t0.elapsed().as_nanos();
        self.stats_candidates_q_ns += cq_delta;
        self.stats_candidates_q_ns_by_k[k as usize] += cq_delta as u64;
        self.drain_cq_phase_timers();
        self.stats_candidates_q_calls += 1;
        let total = candidates.len() as u64;
        self.stats_candidates_total_seen_by_k[k as usize] += total;
        let mut phi_slot = PhiParentSlot::new();
        for v in candidates.iter() {
            // Same shared candidate test as `traverse` — the seeder must
            // apply the identical parent rule the workers do.
            let Some((d_rref, d_pivots, info_d)) =
                self.test_candidate(k as usize, &rref, *v, &mut phi_slot)
            else {
                continue;
            };
            self.traverse_seed(d_rref, d_pivots, info_d, frontier_depth, task_tx);
        }
    }

    /// Consume this WorkerState and produce the `(output, stats, per_k_stats)`
    /// tuple used by both the sequential and parallel drivers. Stats layout
    /// is documented on [`enumerate_doubly_even`].
    ///
    /// When `output_writer` is set (streaming mode), `output` is empty and
    /// the writer is explicitly flushed here so any disk-write failure
    /// surfaces as a panic rather than being swallowed by `BufWriter::Drop`.
    pub(crate) fn finalize(mut self) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>) {
        if let Some(mut w) = self.output_writer.take() {
            w.flush().expect("BinaryWriter final flush failed");
        }
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
            self.stats_phi_reject as u128,
            self.stats_phi_accept_unique as u128,
            self.stats_phi_tie_accept as u128,
            self.stats_phi_tie_reject as u128,
            self.stats_phi_ns,
            self.stats_phi_strata_sum as u128,
            self.stats_phi_m_size_sum as u128,
            self.stats_nauty_ns_kept,
            self.stats_cq_qbasis_ns,
            self.stats_cq_autimage_ns,
            self.stats_cq_singular_ns,
            self.stats_cq_orbitmin_ns,
            self.stats_cq_lift_sort_ns,
            self.stats_phi_frame_gray_ns,
            self.stats_phi_sort_ns,
            self.stats_phi_first_stratum_ns,
            self.stats_phi_wht_ns,
            self.stats_phi_direct_ns,
            self.stats_phi_sampled_calls as u128,
            self.stats_phi_ctx_ns,
            self.stats_phi_ctx_builds as u128,
            self.stats_phi_s1_fastpath as u128,
            self.stats_phi_chain_fastpath as u128,
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
            self.stats_phi_reject_by_k,
            self.stats_phi_accept_unique_by_k,
            self.stats_phi_tie_accept_by_k,
            self.stats_phi_tie_reject_by_k,
            self.stats_phi_ns_by_k,
            self.stats_candidates_q_ns_by_k,
            self.stats_nauty_ns_by_k,
            self.stats_phi_sampled_calls_by_k,
            self.stats_phi_ctx_ns_by_k,
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
/// - `stats` — flat vector (49 u128 fields). See layout below; the
///   canonical Python mirror is `scripts/bench.py::KERNEL_STATS_LAYOUT`.
/// - `per_k_stats` — rectangular `[19][max_k+1]` matrix of u64 counters
///   bucketed by the *parent* rank k (i.e., the rank of C; D has rank
///   k+1), EXCEPT row 16 (`nauty_ns_by_k`) which is bucketed by the rank
///   of the code being canonised. Rows in fixed order:
///   `[is_canon_aug_calls, parent_eq_hits, weight_enum_filtered,
///   bfs_calls, bfs_hits, bfs_rejects, mass_stop_pre_loop,
///   mass_stop_in_loop, candidates_total_seen, candidates_skipped,
///   phi_reject, phi_accept_unique, phi_tie_accept, phi_tie_reject,
///   phi_ns, candidates_q_ns, nauty_ns, phi_sampled_calls,
///   phi_ctx_ns]`.
///   Rows 0–5 from Phase 1 of `for-complete-enumeration-of-proud-meerkat.md`
///   (σ_Q-orbit-min rejection-rate audit). Rows 6–9 from the mass-stop
///   audit (same plan, Conway–Pless gluing follow-up). Rows 10–13 from
///   the D15 coset-spectrum parent-rule audit (zero unless
///   `DOUBLY_EVEN_PARENT_RULE=audit`). Rows 14–16 are the post-D15
///   per-rank timing rows (ns; always-on — they bucket deltas the
///   aggregate fields already pay for); row 17 counts the 1-in-64
///   sampled φ cascades per rank (`phase_timers` builds only); row 18
///   (`phi_ctx_ns_by_k`, always-on) buckets the D16 per-parent ctx
///   build time — a SUBSET of row 14, not additive with it.
///
/// Stats vector layout (34 u128 fields, packed for pyo3 tuple-arity
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
///  26   phi_reject                     (D15 audit; 0 unless audit mode)
///  27   phi_accept_unique              (D15 audit)
///  28   phi_tie_accept                 (D15 audit)
///  29   phi_tie_reject                 (D15 audit)
///  30   phi_ns                         (D15 audit; timer)
///  31   phi_strata_sum                 (D15 audit)
///  32   phi_m_size_sum                 (D15 audit)
///  33   nauty_ns_kept                  (D15 audit; timer — κ numerator)
///  34   cq_qbasis_ns                   (phase_timers; σ_Q 5-way split)
///  35   cq_autimage_ns                 (phase_timers)
///  36   cq_singular_ns                 (phase_timers)
///  37   cq_orbitmin_ns                 (phase_timers)
///  38   cq_lift_sort_ns                (phase_timers)
///  39   phi_vhalf_ns                   (phase_timers; sampled φ split —
///                                       pre-D16: frame+Gray sweep)
///  40   phi_members_ns                 (phase_timers; pre-D16: sort)
///  41   phi_first_stratum_ns           (phase_timers)
///  42   phi_wht_ns                     (phase_timers; later-stratum WHT)
///  43   phi_direct_ns                  (phase_timers; later direct)
///  44   phi_sampled_calls              (phase_timers; sampling weights)
///  45   phi_ctx_ns                     (D16; always-on; SUBSET of phi_ns)
///  46   phi_ctx_builds                 (D16; always-on)
///  47   phi_s1_fastpath                (D16; always-on)
///  48   phi_chain_fastpath             (D17; O(1) E-chain decisions at stratum >= 2)
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
    enumerate_doubly_even_with_rule(n, max_k, quota, factorial_n, ParentRule::from_env())
}

/// [`enumerate_doubly_even`] with an explicit parent rule — the
/// env-independent entry the rule-equivalence tests drive (integration
/// tests must not mutate process env: the cargo harness is threaded).
pub fn enumerate_doubly_even_with_rule(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    rule: ParentRule,
) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>) {
    let mut state = WorkerState::new(n, max_k, quota, factorial_n, rule);
    // Zero code: rref empty, pivots empty.
    let zero_rref: Vec<BinVec> = Vec::new();
    let zero_pivots: Vec<u32> = Vec::new();
    let info = state.canon_info(&zero_rref);
    state.traverse(zero_rref, zero_pivots, info);
    state.finalize()
}

/// Parallel driver: pipelined outer-DFS subtree fan-out across a worker pool.
///
/// `num_threads` long-lived workers spawn first and wait on a bounded
/// crossbeam channel. The seeder runs on the main thread, walking the
/// canonical-augmentation DFS sequentially through depths `0 ..
/// frontier_depth - 1` (emitting those codes into the seeder's own
/// `output`), and **pushing each depth-`frontier_depth` accepted seed
/// directly into the channel as it discovers it**. Workers pick up
/// seeds as soon as they arrive, so seeder DFS and worker recursion
/// overlap on wall time. Bounded channel capacity `(num_threads * 4)
/// .max(8)` provides backpressure: if workers are saturated, the seeder
/// blocks on `send` rather than racing ahead.
///
/// This is the producer-consumer model endorsed by DFGHILM Appendix B.4
/// ("survivors become fodder for the next worker"). It replaces the
/// pre-2026-05-21 design that built a complete frontier `Vec` before
/// any worker started — that design left the seeder as a serial
/// Amdahl ceiling (~44 % of wall at N=22, t=16, d=4, per
/// `architecture/07-parallel-scaling-profile.md`).
///
/// Each worker holds its own canon cache and stat counters; mass-stop
/// is disabled inside workers (per-worker quota = `u128::MAX`) so the
/// V1 path loses the global Gaborit quota pruning (4–11 % regression
/// measured per `architecture/04-optimisations.md` §D5). The trade is
/// a coarse-grained parallelism win dominant at N ≥ 18.
///
/// Fallbacks to [`enumerate_doubly_even`] (the sequential driver):
///
/// - `num_threads <= 1`, OR
/// - `max_k <= frontier_depth` (no work for workers — the seeder
///   would cover everything itself).
///
/// Output ordering: not DFS order. Seeder emissions (`k < frontier_depth`)
/// come in DFS order; worker outputs append in receive order. Callers
/// that need deterministic order must sort downstream — Python's
/// `augment.enumerate_doubly_even` already treats the raw stream as
/// unordered.
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
    enumerate_doubly_even_parallel_with_rule(
        n,
        max_k,
        quota,
        factorial_n,
        num_threads,
        ParentRule::from_env(),
    )
}

/// [`enumerate_doubly_even_parallel`] with an explicit parent rule. The
/// rule is resolved once here and handed to the seeder and every worker,
/// guaranteeing rule agreement across the frontier. Seeder-pool knobs
/// come from the environment (`DOUBLY_EVEN_SEEDER_THREADS`,
/// `DOUBLY_EVEN_SEEDER_PAR_MIN_L`).
#[cfg(feature = "parallel")]
pub fn enumerate_doubly_even_parallel_with_rule(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    num_threads: usize,
    rule: ParentRule,
) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>) {
    let (seeder_threads, seeder_min_l) =
        crate::seeder_pool::SeederPool::env_defaults(num_threads);
    enumerate_doubly_even_parallel_with_seeder(
        n,
        max_k,
        quota,
        factorial_n,
        num_threads,
        rule,
        seeder_threads,
        seeder_min_l,
    )
}

/// [`enumerate_doubly_even_parallel_with_rule`] with explicit seeder-pool
/// knobs — the env-independent entry the pooled-seeder determinism tests
/// drive (integration tests must not mutate process env: the cargo
/// harness is threaded). `seeder_threads <= 1` disables the helper pool
/// (exact pre-D16 seeder behaviour).
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn enumerate_doubly_even_parallel_with_seeder(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    num_threads: usize,
    rule: ParentRule,
    seeder_threads: usize,
    seeder_min_l: u32,
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
        return enumerate_doubly_even_with_rule(n, max_k, quota, factorial_n, rule);
    }

    // Worker pool spawns first and waits on the (initially empty)
    // task channel. The bounded cap doubles as backpressure once the
    // seeder is producing — saturated workers stall the seeder rather
    // than letting the queue grow without bound.
    let cap = (num_threads * 4).max(8);
    let (task_tx, task_rx) = bounded::<SeedFrontier>(cap);
    let (result_tx, result_rx) =
        unbounded::<(Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>)>();

    // D13-V4 cut 4: shared cross-worker mass tracker. Workers consult
    // `gm.is_full(k+1)` instead of local `mass_at_k`, recovering the
    // 4–11 % sequential mass-stop win that V2/V3 left on the floor.
    // The worker's *local* quota stays u128::MAX (the local panic check
    // would otherwise race; global counter is the authoritative one).
    let global_mass = std::sync::Arc::new(GlobalMassTracker::new(quota.clone()));

    let mut handles = Vec::with_capacity(num_threads);
    for _ in 0..num_threads {
        let task_rx = task_rx.clone();
        let result_tx = result_tx.clone();
        let mk = max_k;
        let nn = n;
        let fact = factorial_n;
        let gm = std::sync::Arc::clone(&global_mass);
        handles.push(std::thread::spawn(move || {
            let inf_quota = vec![u128::MAX; (mk + 1) as usize];
            let mut worker = WorkerState::new(nn, mk, inf_quota, fact, rule);
            worker.install_global_mass(gm);
            while let Ok(seed) = task_rx.recv() {
                worker.traverse(seed.rref, seed.pivots, Rc::new(seed.info));
            }
            let _ = result_tx.send(worker.finalize());
        }));
    }
    drop(task_rx);
    drop(result_tx);

    // Seeder runs on the main thread, pushing each frontier-depth seed
    // into task_tx as the DFS discovers it. Workers (already blocking
    // on task_rx.recv()) pick them up immediately, overlapping the
    // seeder's traversal cost with worker recursion. Codes at depths
    // 0..frontier_depth-1 land in seed_state.output here on the main
    // thread; depth-`frontier_depth` seeds travel by channel.
    let mut seed_state = WorkerState::new(n, max_k, quota.clone(), factorial_n, rule);
    seed_state.install_global_mass(std::sync::Arc::clone(&global_mass));
    // D16 lever B: helper pool for the seeder's σ_Q calls, alive only
    // for the seeding span (helpers exit before the worker-dominated
    // tail; the main worker pool is starved during this span anyway).
    if seeder_threads >= 2 {
        seed_state.install_seeder_pool(std::sync::Arc::new(
            crate::seeder_pool::SeederPool::new(seeder_threads, seeder_min_l),
        ));
    }
    {
        let zero_rref: Vec<BinVec> = Vec::new();
        let zero_pivots: Vec<u32> = Vec::new();
        let zero_info = seed_state.canon_info(&zero_rref);
        seed_state.traverse_seed(
            zero_rref,
            zero_pivots,
            zero_info,
            frontier_depth,
            &task_tx,
        );
    }
    seed_state.clear_seeder_pool();
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

    combined
}

/// Assert `mass[k] == quota[k]` for `k = 0..=max_k`. The streaming
/// kernel's only correctness signal — without the in-memory
/// `Vec<EnumeratedRaw>`, Python can't recompute the mass formula
/// without reading the binary stream back, and we want to catch any
/// kernel-side miscount before paying the egress cost.
///
/// Panics with a per-k diff on any mismatch (excess or shortfall).
fn assert_mass_matches_quota(mass: &[u128], quota: &[u128], max_k: u32) {
    let limit = ((max_k as usize) + 1).min(mass.len()).min(quota.len());
    let mut diffs: Vec<String> = Vec::new();
    for k in 0..limit {
        if mass[k] != quota[k] {
            let line = if mass[k] > quota[k] {
                format!("  k={k}: mass={}, quota={}, excess=+{}", mass[k], quota[k], mass[k] - quota[k])
            } else {
                format!("  k={k}: mass={}, quota={}, shortfall=-{}", mass[k], quota[k], quota[k] - mass[k])
            };
            diffs.push(line);
        }
    }
    if !diffs.is_empty() {
        panic!(
            "streaming kernel mass-formula assertion failed (max_k={max_k}):\n{}",
            diffs.join("\n")
        );
    }
}

/// Output of the streaming drivers. The `mass` field is the per-k
/// `Σ N!/|Aut|` snapshot — already validated against `quota` in the
/// driver, but returned so the Python caller can also surface it in
/// `stats.json` for human-readable post-run inspection.
pub struct StreamingResult {
    pub stats: Vec<u128>,
    pub per_k_stats: Vec<Vec<u64>>,
    pub mass: Vec<u128>,
}

/// Streaming sequential driver. Mirrors [`enumerate_doubly_even`] but
/// writes each emitted class to `output_dir/out.w000.bin` in the binary
/// format defined in [`crate::streaming`]. After traversal completes,
/// asserts the in-Rust mass formula `mass[k] == quota[k]` for every k
/// — replaces the Python-side `sigma_brute` / `gaborit_sigma` check
/// that the in-memory path runs post-collection.
///
/// The output file is truncated if it exists. The caller is responsible
/// for ensuring `output_dir` exists and is writable.
pub fn enumerate_doubly_even_streaming(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    output_dir: &std::path::Path,
) -> StreamingResult {
    let path = output_dir.join("out.w000.bin");
    let writer = BinaryWriter::create(&path, n, 0)
        .expect("failed to create streaming output file");
    let quota_for_assert = quota.clone();
    let rule = ParentRule::from_env();
    let mut state = WorkerState::new(n, max_k, quota, factorial_n, rule);
    state.install_output_writer(writer);
    let zero_rref: Vec<BinVec> = Vec::new();
    let zero_pivots: Vec<u32> = Vec::new();
    let info = state.canon_info(&zero_rref);
    state.traverse(zero_rref, zero_pivots, info);
    let mass = state.mass_snapshot();
    let (_empty, stats, per_k_stats) = state.finalize();
    debug_assert!(_empty.is_empty(), "streaming mode must produce empty in-memory output");
    assert_mass_matches_quota(&mass, &quota_for_assert, max_k);
    StreamingResult { stats, per_k_stats, mass }
}

/// Streaming parallel driver. Mirrors [`enumerate_doubly_even_parallel`] but
/// hands each worker (and the seeder) its own [`BinaryWriter`] over a file
/// `output_dir/out.w{:03}.bin`. The seeder uses `worker_id = num_threads`,
/// so the merge script can glob `out.w*.bin` and pick up everything.
///
/// Falls back to [`enumerate_doubly_even_streaming`] when `num_threads <= 1`
/// or `max_k <= frontier_depth` (same conditions as the in-memory parallel
/// driver).
#[cfg(feature = "parallel")]
pub fn enumerate_doubly_even_parallel_streaming(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    num_threads: usize,
    output_dir: &std::path::Path,
) -> StreamingResult {
    use crossbeam_channel::{bounded, unbounded};

    let frontier_depth: u32 = std::env::var("DOUBLY_EVEN_FRONTIER_DEPTH")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&d| d >= 2)
        .unwrap_or(4);

    if num_threads <= 1 || max_k <= frontier_depth {
        return enumerate_doubly_even_streaming(n, max_k, quota, factorial_n, output_dir);
    }

    let cap = (num_threads * 4).max(8);
    let (task_tx, task_rx) = bounded::<SeedFrontier>(cap);
    let (result_tx, result_rx) = unbounded::<(Vec<u128>, Vec<Vec<u64>>)>();

    let quota_for_assert = quota.clone();
    let global_mass = std::sync::Arc::new(GlobalMassTracker::new(quota.clone()));

    // One rule resolution for seeder + all workers (rule agreement).
    let rule = ParentRule::from_env();

    let mut handles = Vec::with_capacity(num_threads);
    for worker_id in 0..num_threads {
        let task_rx = task_rx.clone();
        let result_tx = result_tx.clone();
        let mk = max_k;
        let nn = n;
        let fact = factorial_n;
        let gm = std::sync::Arc::clone(&global_mass);
        let path = output_dir.join(format!("out.w{:03}.bin", worker_id));
        handles.push(std::thread::spawn(move || {
            let writer = BinaryWriter::create(&path, nn, worker_id as u32)
                .expect("failed to create per-worker streaming file");
            let inf_quota = vec![u128::MAX; (mk + 1) as usize];
            let mut worker = WorkerState::new(nn, mk, inf_quota, fact, rule);
            worker.install_global_mass(gm);
            worker.install_output_writer(writer);
            while let Ok(seed) = task_rx.recv() {
                worker.traverse(seed.rref, seed.pivots, Rc::new(seed.info));
            }
            let (_empty, stats, per_k) = worker.finalize();
            let _ = result_tx.send((stats, per_k));
        }));
    }
    drop(task_rx);
    drop(result_tx);

    // Seeder gets worker_id = num_threads so the merge glob is uniform.
    let seeder_id = num_threads;
    let seeder_path = output_dir.join(format!("out.w{:03}.bin", seeder_id));
    let seeder_writer = BinaryWriter::create(&seeder_path, n, seeder_id as u32)
        .expect("failed to create seeder streaming file");
    let mut seed_state = WorkerState::new(n, max_k, quota.clone(), factorial_n, rule);
    seed_state.install_global_mass(std::sync::Arc::clone(&global_mass));
    seed_state.install_output_writer(seeder_writer);
    // D16 lever B (same wiring as the in-memory parallel driver).
    let (seeder_threads, seeder_min_l) =
        crate::seeder_pool::SeederPool::env_defaults(num_threads);
    if seeder_threads >= 2 {
        seed_state.install_seeder_pool(std::sync::Arc::new(
            crate::seeder_pool::SeederPool::new(seeder_threads, seeder_min_l),
        ));
    }
    {
        let zero_rref: Vec<BinVec> = Vec::new();
        let zero_pivots: Vec<u32> = Vec::new();
        let zero_info = seed_state.canon_info(&zero_rref);
        seed_state.traverse_seed(
            zero_rref,
            zero_pivots,
            zero_info,
            frontier_depth,
            &task_tx,
        );
    }
    seed_state.clear_seeder_pool();
    drop(task_tx);

    let (_empty, mut combined_stats, mut combined_per_k) = seed_state.finalize();
    for _ in 0..num_threads {
        let (stats, per_k) = result_rx
            .recv()
            .expect("worker thread hung up before sending result");
        merge_stats_only(&mut combined_stats, &mut combined_per_k, stats, per_k);
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }

    let mass = global_mass.snapshot();
    assert_mass_matches_quota(&mass, &quota_for_assert, max_k);
    StreamingResult {
        stats: combined_stats,
        per_k_stats: combined_per_k,
        mass,
    }
}

/// In-place merge of stats vectors (no `Vec<EnumeratedRaw>` to splice).
/// Used by the streaming parallel driver.
#[cfg(feature = "parallel")]
fn merge_stats_only(
    a_stats: &mut Vec<u128>,
    a_per_k: &mut Vec<Vec<u64>>,
    b_stats: Vec<u128>,
    b_per_k: Vec<Vec<u64>>,
) {
    debug_assert_eq!(a_stats.len(), b_stats.len(), "stats vector length mismatch");
    for (i, (x, y)) in a_stats.iter_mut().zip(b_stats.iter()).enumerate() {
        if i == STATS_MAX_BUCKET_SIZE_IDX {
            *x = (*x).max(*y);
        } else {
            *x = x.checked_add(*y).expect("stats merge overflow");
        }
    }
    debug_assert_eq!(a_per_k.len(), b_per_k.len(), "per_k rows count mismatch");
    for (row_a, row_b) in a_per_k.iter_mut().zip(b_per_k.iter()) {
        debug_assert_eq!(row_a.len(), row_b.len(), "per_k row length mismatch");
        for (xa, yb) in row_a.iter_mut().zip(row_b.iter()) {
            *xa = xa.checked_add(*yb).expect("per_k merge overflow");
        }
    }
}

/// Index of `stats_max_bucket_size` inside the flat stats vector — `.max()`
/// not `+=` on merge. See doc on [`enumerate_doubly_even`] for full layout.
#[cfg(feature = "parallel")]
const STATS_MAX_BUCKET_SIZE_IDX: usize = 14;

#[cfg(feature = "parallel")]
pub(crate) fn merge_finalized(
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
