//! `WorkerState` — the per-worker recursion state — plus the traversal,
//! canonical-parent / canonical-augmentation tests and the candidate
//! test. Bodies are verbatim from the original `enumerate.rs`.

use std::collections::HashMap;
use std::rc::Rc;

use lru::LruCache;

use crate::candidates::doubly_even_candidates_q;
use crate::linalg::{apply_permutation, row_reduce};
use crate::parent_rule::{
    hyperplane_basis, phi_cascade_shared, tie_break_parent, ParentRule, PhiOutcome,
    PhiParentSlot, PhiResult,
};
use crate::permutations::{dual_basis, perm_inverse};
use crate::streaming::BinaryWriter;
use crate::subspace_orbit::subspace_in_orbit;
use crate::types::BinVec;
use crate::u256::U256;

use super::cache::{canon_cache_capacity, weight_enum, BucketEntry, CachedInfo, LabelMode};
#[cfg(feature = "parallel")]
use super::drivers::{GlobalMassTracker, SeedFrontier};

/// Output row for one canonical code emitted by [`enumerate_doubly_even`].
pub struct EnumeratedRaw {
    pub rref: Vec<BinVec>,
    pub canonical_column_order: Vec<u32>,
    pub aut_generators: Vec<Vec<u32>>,
    pub aut_order: u128,
    pub column_orbits: Vec<u32>,
}

pub(crate) struct WorkerState {
    pub(crate) n: u32,
    pub(crate) max_k: u32,
    pub(crate) quota: Vec<U256>,
    pub(crate) mass_at_k: Vec<U256>,
    /// Counts-only mode (N ≥ 30 frontier): emits fold into per-rank
    /// aggregates instead of materialising `EnumeratedRaw` records, so
    /// memory stays O(ranks) rather than O(classes).
    pub(crate) counts_only: bool,
    /// Per-rank `|Aut| → class-count` histogram. Populated only when
    /// `counts_only`; drained by [`Self::take_counts`].
    counts_hist: Vec<HashMap<u128, u64>>,
    pub(crate) factorial_n: u128,
    /// D13-V4 cut 4: when `Some`, every emit atomically increments
    /// `global_mass.mass[k]` and the candidate-loop mass-stop checks
    /// consult `global_mass.is_full(k+1)` instead of the local
    /// `self.mass_at_k`. Set by the parallel driver; `None` for the
    /// sequential path (no-op cost when absent).
    #[cfg(feature = "parallel")]
    pub(crate) global_mass: Option<std::sync::Arc<GlobalMassTracker>>,
    /// Per-rank mass emitted by this worker but not yet flushed to
    /// `global_mass`. Batched to avoid the per-emit lock that became a
    /// 96-thread futex storm (see `GlobalMassTracker`). U256 because a
    /// full batch of per-class `N!/|Aut|` overflows u128 at N ≥ 31.
    #[cfg(feature = "parallel")]
    pending_mass: Vec<U256>,
    /// Emissions since the last `flush_global_mass`; a flush fires at
    /// `mass_flush_interval`.
    #[cfg(feature = "parallel")]
    emits_since_flush: u64,
    /// Batch size for `global_mass` flushes
    /// (`DOUBLY_EVEN_MASS_FLUSH_INTERVAL`, default 2048), resolved once.
    #[cfg(feature = "parallel")]
    mass_flush_interval: u64,
    /// D16 lever B: helper pool the SEEDER's σ_Q calls fan out onto
    /// (`doubly_even_candidates_q_pooled`). Installed only on the
    /// seeder's `WorkerState` by the parallel drivers and dropped right
    /// after `traverse_seed` returns; workers keep `None` (they are
    /// already saturated). `None` ⇒ exact pre-D16 sequential calls.
    #[cfg(feature = "parallel")]
    pub(crate) seeder_pool: Option<std::sync::Arc<crate::seeder_pool::SeederPool>>,
    /// D20 demand-driven self-subdivision. When `self_subdivide` is set
    /// (the `DOUBLY_EVEN_SELF_SUBDIVIDE` knob), a busy worker at shallow
    /// depth (`k <= frontier_depth + subdivide_delta`) donates an accepted
    /// child onto `task_tx` instead of recursing it, but only when
    /// `balancer.idle_workers > 0`. On the seeder these are installed too
    /// (with `task_tx = None`) so its send path reserves `outstanding`.
    /// All `None`/`false` ⇒ exact pre-D20 behaviour.
    #[cfg(feature = "parallel")]
    pub(crate) task_tx: Option<crossbeam_channel::Sender<SeedFrontier>>,
    #[cfg(feature = "parallel")]
    pub(crate) balancer: Option<std::sync::Arc<super::drivers::LoadBalancer>>,
    #[cfg(feature = "parallel")]
    pub(crate) frontier_depth: u32,
    #[cfg(feature = "parallel")]
    pub(crate) self_subdivide: bool,
    #[cfg(feature = "parallel")]
    pub(crate) subdivide_delta: u32,
    /// Seeder-timeline instrumentation (profiling builds only). When the
    /// profile driver arms `profile_epoch`, [`traverse_seed`] records
    /// epoch-relative seed-enqueue timestamps and per-rank σ_Q call spans
    /// into the two Vecs. `None` on workers, on the sequential path, and
    /// on production drivers — the recording branches are never taken.
    #[cfg(feature = "parallel_profiling")]
    pub(crate) profile_epoch: Option<std::time::Instant>,
    /// One `(ready_ns, sent_ns)` pair per enqueued seed, in send order;
    /// the index is the seed's `seed_id`. `sent_ns − ready_ns` is the
    /// bounded-channel backpressure wait (workers saturated).
    #[cfg(feature = "parallel_profiling")]
    pub(crate) profile_enqueues: Vec<(u64, u64)>,
    /// One `(k, l, pooled, start_ns, end_ns)` row per seeder σ_Q call.
    #[cfg(feature = "parallel_profiling")]
    pub(crate) profile_sigma_spans: Vec<(u32, u32, bool, u64, u64)>,
    /// When true, the two mass-stop branches in [`traverse`] (`mass_at_k[k+1]
    /// >= quota[k+1]` checks before and inside the candidate loop) become
    /// no-ops. Read once from `DOUBLY_EVEN_NO_MASS_STOP` at construction;
    /// kept off the parallel workers regardless because they already have
    /// `quota = u128::MAX`. Ablation knob for the refactor profiling pass.
    pub(crate) skip_mass_stop: bool,
    pub(crate) canon_cache: LruCache<Vec<BinVec>, Rc<CachedInfo>>,
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
    pub(crate) secondary_cache: HashMap<Vec<u32>, Vec<BucketEntry>>,
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
    pub(crate) maintain_secondary_cache: bool,
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
    pub(crate) parent_rule: ParentRule,
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
    /// Canonical-labelling mode (the autom-only lever, 2026-06-12).
    /// Resolved once per driver invocation; see [`LabelMode`].
    pub(crate) label_mode: LabelMode,
    /// True misses computed with `getcanon = FALSE` (slot 49).
    pub stats_canon_autom_only_calls: u64,
    /// Autom-only cache entries recomputed full on a label-needing hit
    /// (slot 50). Expected ~0.005 % of `primary_hits`.
    pub stats_canon_label_upgrades: u64,
    /// Tie-dump sink (`DOUBLY_EVEN_TIE_DUMP`, sequential drivers only):
    /// one JSONL record per φ-tie, for the invariant-collision analysis.
    /// `None` ⇒ zero-cost no-op. See [`Self::dump_tie`].
    pub(crate) tie_dump: Option<std::io::BufWriter<std::fs::File>>,
    /// Decomposability-log sink (`DOUBLY_EVEN_DECOMP_LOG`, sequential
    /// drivers only): one JSONL record per `test_candidate` canon-info
    /// consultation — support / direct-sum components / twin columns of
    /// the child RREF, plus the nauty ns the consultation cost (0 ⇒
    /// cache hit). The lever-4 measurement (bottlenecks §4); `None` ⇒
    /// zero-cost no-op. Flushed in `finalize`.
    pub(crate) decomp_log: Option<std::io::BufWriter<std::fs::File>>,
    pub(crate) output: Vec<EnumeratedRaw>,
    /// Streaming output sink. When `Some`, every emit writes
    /// `(k, aut_order, basis)` to the per-worker binary file and skips
    /// the per-class clones (`canonical_column_order`, `aut_generators`,
    /// `column_orbits`) that the in-memory `EnumeratedRaw` would carry.
    /// `None` is the legacy in-memory path used by the small-N test
    /// suite and bench harness. Set via [`Self::install_output_writer`].
    pub(crate) output_writer: Option<BinaryWriter<std::fs::File>>,
}

/// Default `global_mass` flush batch size (emissions per worker between
/// shared-tracker pushes). Overridable via `DOUBLY_EVEN_MASS_FLUSH_INTERVAL`.
/// Larger = fewer tracker locks but the shared full-flag flips later →
/// more mass-stop over-search. Parallel path only.
#[cfg(feature = "parallel")]
const DEFAULT_MASS_FLUSH_INTERVAL: u64 = 2048;

/// Resolve the flush interval once at `WorkerState::new` (workers + seeder
/// all read the same env). Floor of 1 (flush every emission) is legal and
/// is the tightest-pruning, most-lock ablation point.
#[cfg(feature = "parallel")]
fn mass_flush_interval_from_env() -> u64 {
    std::env::var("DOUBLY_EVEN_MASS_FLUSH_INTERVAL")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(DEFAULT_MASS_FLUSH_INTERVAL)
}

/// `D = ⟨C, v⟩`: append `v` to the parent basis and re-RREF.
fn extend_rref(rref: &[BinVec], v: BinVec, n: u32) -> (Vec<BinVec>, Vec<u32>) {
    let mut new_basis = Vec::with_capacity(rref.len() + 1);
    new_basis.extend_from_slice(rref);
    new_basis.push(v);
    row_reduce(&new_basis, n)
}

impl WorkerState {
    pub(crate) fn new(
        n: u32,
        max_k: u32,
        quota: Vec<U256>,
        factorial_n: u128,
        parent_rule: ParentRule,
        label_mode: LabelMode,
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
            mass_at_k: vec![U256::ZERO; len],
            counts_only: false,
            counts_hist: vec![HashMap::new(); len],
            factorial_n,
            skip_mass_stop,
            #[cfg(feature = "parallel")]
            global_mass: None,
            #[cfg(feature = "parallel")]
            pending_mass: vec![U256::ZERO; len],
            #[cfg(feature = "parallel")]
            emits_since_flush: 0,
            #[cfg(feature = "parallel")]
            mass_flush_interval: mass_flush_interval_from_env(),
            #[cfg(feature = "parallel")]
            seeder_pool: None,
            #[cfg(feature = "parallel")]
            task_tx: None,
            #[cfg(feature = "parallel")]
            balancer: None,
            #[cfg(feature = "parallel")]
            frontier_depth: 0,
            #[cfg(feature = "parallel")]
            self_subdivide: false,
            #[cfg(feature = "parallel")]
            subdivide_delta: 1,
            #[cfg(feature = "parallel_profiling")]
            profile_epoch: None,
            #[cfg(feature = "parallel_profiling")]
            profile_enqueues: Vec::new(),
            #[cfg(feature = "parallel_profiling")]
            profile_sigma_spans: Vec::new(),
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
            label_mode,
            stats_canon_autom_only_calls: 0,
            stats_canon_label_upgrades: 0,
            tie_dump: None,
            decomp_log: None,
            output: Vec::new(),
            output_writer: None,
        }
    }

    /// Attach the tie-dump sink (`DOUBLY_EVEN_TIE_DUMP`). Installed by
    /// the sequential drivers only; the parallel drivers panic when the
    /// env var is set (interleaved appends from many workers are not
    /// sound for arbitrary record sizes — per-worker files are the
    /// documented future alternative if ever needed).
    pub(crate) fn install_tie_dump(&mut self, writer: std::io::BufWriter<std::fs::File>) {
        self.tie_dump = Some(writer);
    }

    /// Attach the decomposability-log sink (`DOUBLY_EVEN_DECOMP_LOG`).
    /// Sequential drivers only, for the same interleaved-append reason
    /// as the tie dump.
    pub(crate) fn install_decomp_log(&mut self, writer: std::io::BufWriter<std::fs::File>) {
        self.decomp_log = Some(writer);
    }

    /// Decomposability-log hook (no-op when `DOUBLY_EVEN_DECOMP_LOG` is
    /// unset): one JSONL record per canon-info consultation in
    /// [`Self::test_candidate`], carrying the direct-sum / twin-column
    /// structure of the child RREF (see [`super::decomp`] for the math)
    /// and the nauty ns the consultation cost (`ns = 0` ⇔ cache hit,
    /// `miss = 0`). Records are buffered; [`flush_decomp_log`] runs in
    /// `finalize` — unlike ties, these are high-volume (one per accept
    /// or tie at every rank), so no per-record flush.
    ///
    /// [`flush_decomp_log`]: Self::flush_decomp_log
    fn log_decomp(&mut self, d_rref: &[BinVec], nauty_ns: u128, miss: u64, outcome: &str) {
        if self.decomp_log.is_none() {
            return;
        }
        let support = super::decomp::column_support(d_rref);
        let comps = super::decomp::component_sizes(d_rref, self.n);
        let twins = super::decomp::twin_class_sizes(d_rref, self.n);
        let comps_json: Vec<String> = comps.iter().map(|c| c.to_string()).collect();
        let twins_json: Vec<String> = twins.iter().map(|c| c.to_string()).collect();
        use std::io::Write;
        let w = self.decomp_log.as_mut().expect("checked above");
        writeln!(
            w,
            "{{\"n\":{},\"k\":{},\"out\":\"{}\",\"ns\":{},\"miss\":{},\"sup\":{},\
             \"comp\":[{}],\"tw\":[{}]}}",
            self.n,
            d_rref.len(),
            outcome,
            nauty_ns,
            miss,
            support.count_ones(),
            comps_json.join(","),
            twins_json.join(","),
        )
        .expect("decomp-log write failed");
    }

    /// Flush the decomp-log sink (called from `finalize`; the BufWriter
    /// would also flush on drop, but explicitly so write errors surface).
    pub(crate) fn flush_decomp_log(&mut self) {
        if let Some(w) = self.decomp_log.as_mut() {
            use std::io::Write;
            w.flush().expect("decomp-log flush failed");
        }
    }

    /// Emitted-record label under the active [`LabelMode`]: `Full` clones
    /// the (always-present) label; `AutomOnly` emits an empty Vec
    /// UNCONDITIONALLY — even for tie-accepted classes whose cache entry
    /// happens to carry one — so the output contract is deterministic
    /// (whether a given entry has a label depends on cache history, which
    /// is timing-dependent in parallel runs).
    fn emitted_label(&self, info: &CachedInfo) -> Vec<u32> {
        match self.label_mode {
            LabelMode::Full => info
                .canonical_column_order
                .as_ref()
                .expect("Full mode computes the label on every call")
                .clone(),
            LabelMode::AutomOnly => Vec::new(),
        }
    }

    /// Tie-dump hook (no-op when `DOUBLY_EVEN_TIE_DUMP` is unset): one
    /// JSONL record per φ-tie, including the partition of the tied
    /// hyperplane functionals into Aut(D)-orbits. A `tie_orbits` with
    /// more than one part is a TRUE invariant collision — inequivalent
    /// codimension-1 subcodes with identical complement-coset spectra
    /// (the Milenkovic-2005 phenomenon); a single part means the tied
    /// strata are Aut(D)-equivalent (the invariant tie is benign).
    /// Hand-rolled JSON: every value is a hex string / int / bool, no
    /// escaping needed. Cost is irrelevant — ties are ~10⁻³–10⁻⁴ of
    /// candidates and the hook is sequential-only.
    fn dump_tie(
        &mut self,
        parent_k: usize,
        c_rref: &[BinVec],
        v: BinVec,
        m_set: &[u16],
        info_d: &CachedInfo,
        accept: bool,
    ) {
        if self.tie_dump.is_none() {
            return;
        }
        // Materialise each tied stratum as a subspace (RREF) and group
        // them into Aut(D)-orbits by greedy representative matching via
        // the exact subspace-orbit BFS (orbit membership is symmetric).
        let bases: Vec<Vec<BinVec>> = m_set
            .iter()
            .map(|&u| {
                let b = hyperplane_basis(c_rref, v, u);
                let (rr, _) = row_reduce(&b, self.n);
                rr
            })
            .collect();
        let mut orbit_of: Vec<usize> = Vec::with_capacity(m_set.len());
        let mut reps: Vec<usize> = Vec::new();
        for (i, b) in bases.iter().enumerate() {
            let mut assigned = None;
            for (oid, &rep_i) in reps.iter().enumerate() {
                if bases[rep_i] == *b
                    || subspace_in_orbit(self.n, b, &bases[rep_i], &info_d.aut_generators)
                {
                    assigned = Some(oid);
                    break;
                }
            }
            match assigned {
                Some(oid) => orbit_of.push(oid),
                None => {
                    orbit_of.push(reps.len());
                    reps.push(i);
                }
            }
        }
        let mut orbits: Vec<Vec<u16>> = vec![Vec::new(); reps.len()];
        for (i, &u) in m_set.iter().enumerate() {
            orbits[orbit_of[i]].push(u);
        }

        // Decomposability tags for the tied CHILD `D = ⟨C, v⟩` (lever-4
        // joint evidence: are invariant collisions direct-sum-rich?).
        let (d_rref, _) = extend_rref(c_rref, v, self.n);
        let d_support = super::decomp::column_support(&d_rref).count_ones();
        let d_comps = super::decomp::component_sizes(&d_rref, self.n);
        let d_twins = super::decomp::twin_class_sizes(&d_rref, self.n);

        use std::io::Write;
        let w = self.tie_dump.as_mut().expect("checked above");
        let rref_hex: Vec<String> = c_rref.iter().map(|r| format!("\"{r:x}\"")).collect();
        let m_set_json: Vec<String> = m_set.iter().map(|u| u.to_string()).collect();
        let orbits_json: Vec<String> = orbits
            .iter()
            .map(|o| {
                let inner: Vec<String> = o.iter().map(|u| u.to_string()).collect();
                format!("[{}]", inner.join(","))
            })
            .collect();
        let comps_json: Vec<String> = d_comps.iter().map(|c| c.to_string()).collect();
        let twins_json: Vec<String> = d_twins.iter().map(|c| c.to_string()).collect();
        writeln!(
            w,
            "{{\"n\":{},\"parent_k\":{},\"parent_rref\":[{}],\"v\":\"{:x}\",\
             \"m_set\":[{}],\"accept\":{},\"aut_order\":\"{}\",\"tie_orbits\":[{}],\
             \"d_sup\":{},\"d_comp\":[{}],\"d_tw\":[{}]}}",
            self.n,
            parent_k,
            rref_hex.join(","),
            v,
            m_set_json.join(","),
            accept,
            info_d.aut_order,
            orbits_json.join(","),
            d_support,
            comps_json.join(","),
            twins_json.join(","),
        )
        .expect("tie-dump write failed");
        // Flush per record: ties are rare and partial runs stay inspectable.
        w.flush().expect("tie-dump flush failed");
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
    pub(crate) fn mass_snapshot(&self) -> Vec<U256> {
        self.mass_at_k.clone()
    }

    /// Drain the per-rank aggregates accumulated in `counts_only` mode:
    /// `(classes_per_rank, |Aut|-histogram_per_rank)`. The class count is
    /// the sum of the histogram's values.
    pub(crate) fn take_counts(&mut self) -> (Vec<u64>, Vec<HashMap<u128, u64>>) {
        let hist = std::mem::take(&mut self.counts_hist);
        let classes = hist.iter().map(|h| h.values().sum()).collect();
        (classes, hist)
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

        let p_d = self.canonical_parent(
            d_rref,
            info_d
                .canonical_column_order
                .as_ref()
                .expect("legacy parent test always requests the label"),
        );
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
                    // (the same cost any rule pays on accepts) — but NOT
                    // the canonical labelling: no decision on this path
                    // consumes it, so `need_label = false` (the autom-only
                    // lever; nauty runs with getcanon=FALSE by default).
                    self.stats_phi_accept_unique += 1;
                    self.stats_phi_accept_unique_by_k[parent_k] += 1;
                    let (d_rref, d_pivots) = extend_rref(rref, v, self.n);
                    let (ns0, calls0) = (self.stats_nauty_ns, self.stats_canon_calls);
                    let info_d = self.canon_info(&d_rref, false);
                    self.log_decomp(
                        &d_rref,
                        self.stats_nauty_ns - ns0,
                        self.stats_canon_calls - calls0,
                        "acc",
                    );
                    Some((d_rref, d_pivots, info_d))
                }
                PhiOutcome::Tie(m_set) => {
                    let (d_rref, d_pivots) = extend_rref(rref, v, self.n);
                    let (ns0, calls0) = (self.stats_nauty_ns, self.stats_canon_calls);
                    let info_d = self.canon_info(&d_rref, true);
                    let (tie_ns, tie_calls) =
                        (self.stats_nauty_ns - ns0, self.stats_canon_calls - calls0);
                    let t1 = std::time::Instant::now();
                    let target = tie_break_parent(
                        rref,
                        v,
                        self.n,
                        &m_set,
                        info_d
                            .canonical_column_order
                            .as_ref()
                            .expect("tie path requests the label"),
                    );
                    let accept =
                        subspace_in_orbit(self.n, rref, &target, &info_d.aut_generators);
                    let tie_delta = t1.elapsed().as_nanos();
                    self.stats_phi_ns += tie_delta;
                    self.stats_phi_ns_by_k[parent_k] += tie_delta as u64;
                    self.dump_tie(parent_k, rref, v, &m_set, &info_d, accept);
                    self.log_decomp(
                        &d_rref,
                        tie_ns,
                        tie_calls,
                        if accept { "tie_acc" } else { "tie_rej" },
                    );
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
        // Audit additionally tallies the φ outcome post-hoc. All three
        // consume the canonical labelling (`canonical_parent` /
        // `tie_break_parent` in the audit resolve), so `need_label = true`.
        let phi_res = self.phi_audit_evaluate(parent_k, rref, v, phi);
        let (d_rref, d_pivots) = extend_rref(rref, v, self.n);
        let nauty_ns_before = self.stats_nauty_ns;
        let calls_before = self.stats_canon_calls;
        let info_d = self.canon_info(&d_rref, true);
        let accept = self.is_canonical_augmentation(parent_k, rref, &d_rref, &info_d);
        self.log_decomp(
            &d_rref,
            self.stats_nauty_ns - nauty_ns_before,
            self.stats_canon_calls - calls_before,
            if accept { "leg_acc" } else { "leg_rej" },
        );
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
                    info_d
                        .canonical_column_order
                        .as_ref()
                        .expect("audit path always requests the label"),
                );
                let accept =
                    subspace_in_orbit(self.n, c_rref, &target, &info_d.aut_generators);
                let tie_delta = t0.elapsed().as_nanos();
                self.stats_phi_ns += tie_delta;
                self.stats_phi_ns_by_k[parent_k] += tie_delta as u64;
                self.dump_tie(parent_k, c_rref, v, &m_set, info_d, accept);
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

    /// D20: arm demand-driven self-subdivision on this state. Workers pass
    /// `task_tx = Some(channel clone)` (they donate accepted children); the
    /// seeder passes `None` (it is the producer — it only needs `balancer`
    /// so its send path can reserve `outstanding`). Sets `self_subdivide`,
    /// which flips on the donation gate in [`Self::traverse`] and the
    /// reservation in [`Self::traverse_seed`].
    #[cfg(feature = "parallel")]
    pub(crate) fn install_self_subdivide(
        &mut self,
        balancer: std::sync::Arc<super::drivers::LoadBalancer>,
        task_tx: Option<crossbeam_channel::Sender<SeedFrontier>>,
        frontier_depth: u32,
        delta: u32,
    ) {
        self.balancer = Some(balancer);
        self.task_tx = task_tx;
        self.frontier_depth = frontier_depth;
        self.subdivide_delta = delta;
        self.self_subdivide = true;
    }

    /// Accumulate a per-class mass contribution into the local pending
    /// buffer, flushing to the shared `global_mass` tracker every
    /// `mass_flush_interval` emissions. No-op on the sequential path
    /// (`global_mass` is `None`). Replaces the old per-emit `gm.add`,
    /// which locked the tracker on every class.
    #[cfg(feature = "parallel")]
    #[inline]
    fn contribute_global_mass(&mut self, k: usize, delta: u128) {
        if self.global_mass.is_none() {
            return;
        }
        self.pending_mass[k] = self.pending_mass[k].add_u128(delta);
        self.emits_since_flush += 1;
        if self.emits_since_flush >= self.mass_flush_interval {
            self.flush_global_mass();
        }
    }

    /// Push this worker's pending per-rank mass into the shared tracker
    /// under a single lock and reset the buffer. Called at the flush
    /// interval and **once more before the worker/seeder's result is
    /// read** by the driver, so the post-join `snapshot()` (the
    /// mass-formula correctness gate) is exact. No-op when `global_mass`
    /// is `None`.
    #[cfg(feature = "parallel")]
    pub(crate) fn flush_global_mass(&mut self) {
        if let Some(gm) = self.global_mass.as_ref() {
            gm.add_batch(&self.pending_mass);
            for p in self.pending_mass.iter_mut() {
                *p = U256::ZERO;
            }
            self.emits_since_flush = 0;
        }
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
        if self.counts_only {
            *self.counts_hist[k as usize]
                .entry(info.aut_order)
                .or_insert(0) += 1;
        } else if let Some(w) = self.output_writer.as_mut() {
            w.write_class(info.aut_order, &rref)
                .expect("BinaryWriter::write_class failed");
        } else {
            self.output.push(EnumeratedRaw {
                rref: rref.clone(),
                canonical_column_order: self.emitted_label(&info),
                aut_generators: info.aut_generators.clone(),
                aut_order: info.aut_order,
                column_orbits: info.column_orbits.clone(),
            });
        }
        // Update mass.
        let mass_contribution = self.factorial_n / info.aut_order;
        self.mass_at_k[k as usize] = self.mass_at_k[k as usize].add_u128(mass_contribution);
        if self.mass_at_k[k as usize] > self.quota[k as usize] {
            panic!(
                "level-{k} mass {} exceeded quota {}",
                self.mass_at_k[k as usize], self.quota[k as usize]
            );
        }
        // D13-V4 cut 4: contribute to the shared cross-worker mass counter
        // (batched — see `contribute_global_mass`) so peer workers can
        // mass-stop based on the global total.
        #[cfg(feature = "parallel")]
        self.contribute_global_mass(k as usize, mass_contribution);
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
            // D20 demand-driven self-subdivision: donate this accepted
            // child to an idle peer instead of recursing it ourselves —
            // but only shallow (near a seed root, where a child is a
            // substantial subtree) and only when a peer is actually idle.
            // Cold path: one relaxed load per shallow parent when armed.
            #[cfg(feature = "parallel")]
            if self.self_subdivide
                && k <= self.frontier_depth + self.subdivide_delta
                && self
                    .balancer
                    .as_ref()
                    .map_or(false, |b| {
                        b.idle_workers.load(std::sync::atomic::Ordering::Relaxed) > 0
                    })
            {
                // Arc + Sender clones (cheap) so we don't hold a borrow of
                // `self` across the `self.traverse(...)` on the Full path.
                let bal = self.balancer.clone().unwrap();
                let tx = self.task_tx.clone().unwrap();
                // Own the CachedInfo (same idiom traverse_seed uses): the
                // Rc wrapper isn't Send, but CachedInfo is.
                let info_owned: CachedInfo = match Rc::try_unwrap(info_d) {
                    Ok(i) => i,
                    Err(rc) => (*rc).clone(),
                };
                // Reserve BEFORE the seed becomes visible: a peer must not
                // be able to dequeue + complete it (fetch_sub) before our
                // fetch_add lands, or `outstanding` underflows.
                bal.outstanding
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let seed = SeedFrontier {
                    rref: d_rref,
                    pivots: d_pivots,
                    info: info_owned,
                    #[cfg(feature = "parallel_profiling")]
                    seed_id: 0,
                };
                match tx.try_send(seed) {
                    // Donated; a peer will emit + recurse it.
                    Ok(()) => {
                        #[cfg(feature = "parallel_profiling")]
                        {
                            bal.donations
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            bal.donation_max_k
                                .fetch_max(k as usize, std::sync::atomic::Ordering::Relaxed);
                        }
                        continue;
                    }
                    // Channel full (already balanced) or disconnected: roll
                    // back the reservation and recurse the child ourselves.
                    Err(e) => {
                        bal.outstanding
                            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        let seed = e.into_inner();
                        self.traverse(seed.rref, seed.pivots, Rc::new(seed.info));
                        continue;
                    }
                }
            }
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
            #[cfg(feature = "parallel_profiling")]
            let (seed_id, ready_ns) = match self.profile_epoch {
                Some(epoch) => (
                    self.profile_enqueues.len() as u32,
                    epoch.elapsed().as_nanos() as u64,
                ),
                None => (0, 0),
            };
            // D20: reserve this seed's termination count BEFORE the
            // (blocking) send makes it visible to a worker. Mirrors the
            // donor's reserve-before-send; harmless when self-subdivision
            // is off (the count is just never consulted).
            if self.self_subdivide {
                self.balancer
                    .as_ref()
                    .expect("self_subdivide set without balancer")
                    .outstanding
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            task_tx
                .send(SeedFrontier {
                    rref,
                    pivots,
                    info: info_owned,
                    #[cfg(feature = "parallel_profiling")]
                    seed_id,
                })
                .expect("worker pool closed before seeder finished");
            #[cfg(feature = "parallel_profiling")]
            if let Some(epoch) = self.profile_epoch {
                let sent_ns = epoch.elapsed().as_nanos() as u64;
                self.profile_enqueues.push((ready_ns, sent_ns));
            }
            return;
        }
        if self.counts_only {
            *self.counts_hist[k as usize]
                .entry(info.aut_order)
                .or_insert(0) += 1;
        } else if let Some(w) = self.output_writer.as_mut() {
            w.write_class(info.aut_order, &rref)
                .expect("BinaryWriter::write_class failed (seeder)");
        } else {
            self.output.push(EnumeratedRaw {
                rref: rref.clone(),
                canonical_column_order: self.emitted_label(&info),
                aut_generators: info.aut_generators.clone(),
                aut_order: info.aut_order,
                column_orbits: info.column_orbits.clone(),
            });
        }
        let mass_contribution = self.factorial_n / info.aut_order;
        self.mass_at_k[k as usize] = self.mass_at_k[k as usize].add_u128(mass_contribution);
        // D13-V4 cut 4: seeder emissions also contribute to the shared
        // (batched) mass counter so workers' mass-stop sees the full
        // picture. (Seeder itself does not mass-stop; see traverse_seed.)
        self.contribute_global_mass(k as usize, mass_contribution);
        if k >= self.max_k {
            return;
        }
        let dual = dual_basis(&rref, &pivots, self.n);
        // NOTE: with the helper pool active, this seeder-side
        // candidates_q_ns delta records WALL time of a parallel region,
        // not CPU time — that shrinkage is the point of lever B.
        let cq_t0 = std::time::Instant::now();
        #[cfg(feature = "parallel_profiling")]
        let span_start_ns = self
            .profile_epoch
            .map(|epoch| epoch.elapsed().as_nanos() as u64);
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
        #[cfg(feature = "parallel_profiling")]
        if let (Some(epoch), Some(start_ns)) = (self.profile_epoch, span_start_ns) {
            // Quotient dimension L = dim(C^⊥/C) = N − 2k; `pooled`
            // replicates the dispatch gate in
            // `candidates::doubly_even_candidates_q_pooled`.
            let l = self.n.saturating_sub(2 * k);
            let pooled = self
                .seeder_pool
                .as_deref()
                .is_some_and(|p| p.size() >= 2 && l >= p.min_l);
            let end_ns = epoch.elapsed().as_nanos() as u64;
            self.profile_sigma_spans.push((k, l, pooled, start_ns, end_ns));
        }
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
}
