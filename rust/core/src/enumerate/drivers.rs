//! Sequential / parallel / streaming drivers, the seed-frontier plumbing
//! and the in-Rust mass-formula gate. Bodies are verbatim from the
//! original `enumerate.rs`.

#[cfg(feature = "parallel")]
use std::rc::Rc;

use crate::parent_rule::ParentRule;
use crate::streaming::BinaryWriter;
use crate::types::BinVec;
use crate::u256::U256;

#[cfg(feature = "parallel")]
use super::cache::CachedInfo;
use super::cache::LabelMode;
use super::worker::{EnumeratedRaw, WorkerState};
#[cfg(feature = "parallel")]
use super::stats::{merge_finalized, merge_stats_only};

/// A canonical node captured at the depth-cut frontier of the parallel
/// seeder. By-value `CachedInfo` (rather than `Rc<…>`) so the entire
/// struct is `Send` and can travel through a crossbeam channel into a
/// worker thread. Workers reconstruct an `Rc` on receipt.
#[cfg(feature = "parallel")]
pub(crate) struct SeedFrontier {
    pub rref: Vec<BinVec>,
    pub pivots: Vec<u32>,
    pub info: CachedInfo,
    /// Stable enqueue index (== index into the seeder's
    /// `profile_enqueues`), so worker-side seed timings can be joined
    /// with the seeder timeline. Profiling builds only; always 0 when
    /// the epoch isn't armed.
    #[cfg(feature = "parallel_profiling")]
    pub seed_id: u32,
}

/// D13-V4 cut 4: shared cross-worker mass tracker for the parallel path.
///
/// Each worker (and the seeder) accumulates `N!/|Aut|` into a *local*
/// per-rank pending buffer as it emits, and flushes that buffer into
/// `mass[k]` here in batches (see `WorkerState::flush_global_mass`).
/// Workers consult the lock-free `is_full(k+1)` before descending —
/// once the collective mass at rank `k+1` hits `quota[k+1]`, the
/// remaining workers prune that subtree just as the sequential
/// mass-stop does. The invariant is one-directional: a *stale* full
/// flag (batching delays when it flips) can only over-search relative
/// to the optimum, never under-search — the class count and the final
/// mass are exact regardless of flush timing.
///
/// Batched writes + atomic read flags replaced a per-emit
/// `Mutex<Vec<U256>>` lock (2026-06-14): post-D15→D19 each class costs
/// ~50 µs, so the per-class write-lock + per-candidate read-lock became
/// a futex storm past ~24 threads (85 % `sy` at 96 threads on
/// c4a-96-metal — effective ~14 of 96 cores). The lock now fires once
/// per ~`DOUBLY_EVEN_MASS_FLUSH_INTERVAL` emissions; the per-candidate
/// read is a relaxed atomic load. U256 since 2026-06-13: σ(30, ·) ≈
/// 2^136 overflows u128 (and a 72-way worker split of it still does).
#[cfg(feature = "parallel")]
pub(crate) struct GlobalMassTracker {
    mass: std::sync::Mutex<Vec<U256>>,
    quota: Vec<U256>,
    /// Monotonic per-rank `mass[k] >= quota[k]` flags. Set during
    /// `add_batch` (mass only grows, so once true they stay true) and
    /// read lock-free by `is_full` — this is what removes the
    /// per-candidate read-lock that caused the 96-thread futex storm.
    full: Vec<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "parallel")]
impl GlobalMassTracker {
    pub(crate) fn new(quota: Vec<U256>) -> Self {
        let len = quota.len();
        // A rank whose quota is zero (σ(N, k) == 0) is "full" from the
        // start, matching the pre-batch `mass[k] (== 0) >= quota[k]`.
        let full = quota
            .iter()
            .map(|q| std::sync::atomic::AtomicBool::new(*q == U256::ZERO))
            .collect();
        Self {
            mass: std::sync::Mutex::new(vec![U256::ZERO; len]),
            quota,
            full,
        }
    }

    /// Add a worker's accumulated per-rank `deltas` under a single lock,
    /// then refresh the monotonic `full` flags. Called once per
    /// ~`DOUBLY_EVEN_MASS_FLUSH_INTERVAL` emissions per worker (plus a
    /// final flush before its result is read), not once per emitted
    /// class — ~2000× fewer lock acquisitions than the pre-2026-06-14
    /// per-emit `add`. `deltas` is U256 because a 2048-batch sum of
    /// per-class `N!/|Aut|` overflows u128 at N ≥ 31.
    pub(crate) fn add_batch(&self, deltas: &[U256]) {
        let mut m = self.mass.lock().expect("global mass tracker poisoned");
        for (k, d) in deltas.iter().enumerate() {
            if *d == U256::ZERO {
                continue;
            }
            m[k] = m[k].checked_add(*d);
            if m[k] >= self.quota[k] {
                self.full[k].store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// True iff rank `k`'s mass has reached `quota[k]`. Lock-free: reads
    /// the monotonic atomic flag set by `add_batch`. Returns `false` for
    /// out-of-range `k`.
    pub(crate) fn is_full(&self, k: usize) -> bool {
        if k >= self.full.len() {
            return false;
        }
        self.full[k].load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Snapshot of the per-k mass totals. Called after all workers have
    /// joined; used by the streaming drivers as the in-Rust correctness
    /// gate (`mass[k] == quota[k]` for `k < max_k` must hold, mirroring
    /// the Python-side `sigma_brute` / `gaborit_sigma` assertion the
    /// in-memory path runs post-collection). Also polled live by the
    /// counts-mode progress watcher.
    pub(crate) fn snapshot(&self) -> Vec<U256> {
        let m = self.mass.lock().expect("global mass tracker poisoned");
        m.clone()
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
/// - `stats` — flat vector (51 u128 fields). See layout below; the
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
///  49   canon_autom_only_calls         (autom-only lever; true misses with getcanon=FALSE)
///  50   canon_label_upgrades           (autom-only entries recomputed full on a label-needing hit)
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
/// Labelling mode and the tie-dump sink still come from the environment
/// here; use [`enumerate_doubly_even_with_opts`] for full injection.
pub fn enumerate_doubly_even_with_rule(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    rule: ParentRule,
) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>) {
    let quota = widen_quota(quota);
    let mut state =
        WorkerState::new(n, max_k, quota, factorial_n, rule, LabelMode::from_env());
    install_tie_dump_from_env(&mut state);
    run_sequential(state)
}

/// u128 → U256 quota widening at the public-API boundary. The legacy
/// `Vec<u128>` signatures stay (every value through N = 29 fits); the
/// N ≥ 30 counts entry takes decimal strings instead.
fn widen_quota(quota: Vec<u128>) -> Vec<U256> {
    quota.into_iter().map(U256::from).collect()
}

/// [`enumerate_doubly_even_with_rule`] with an explicit labelling mode —
/// the env-independent entry the autom-only equivalence tests drive
/// (same threaded-harness rationale). No tie-dump.
pub fn enumerate_doubly_even_with_opts(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    rule: ParentRule,
    labelling: LabelMode,
) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>) {
    let quota = widen_quota(quota);
    run_sequential(WorkerState::new(n, max_k, quota, factorial_n, rule, labelling))
}

/// Shared sequential body: root call + traversal + finalize. The root
/// (zero code) never needs the canonical labelling — it is no candidate's
/// child `D`, so no tie-break or legacy parent test ever reads it
/// (`need_label = false`; `LabelMode::Full` forces the label anyway).
fn run_sequential(
    mut state: WorkerState,
) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>) {
    let zero_rref: Vec<BinVec> = Vec::new();
    let zero_pivots: Vec<u32> = Vec::new();
    let info = state.canon_info(&zero_rref, false);
    state.traverse(zero_rref, zero_pivots, info);
    state.finalize()
}

/// Install the tie-dump sink when `DOUBLY_EVEN_TIE_DUMP` is set.
/// Sequential drivers only — the parallel drivers call
/// [`reject_tie_dump_in_parallel`] instead.
fn install_tie_dump_from_env(state: &mut WorkerState) {
    if let Ok(path) = std::env::var("DOUBLY_EVEN_TIE_DUMP") {
        let path = path.trim();
        if !path.is_empty() {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap_or_else(|e| {
                    panic!("DOUBLY_EVEN_TIE_DUMP: cannot open {path:?}: {e}")
                });
            state.install_tie_dump(std::io::BufWriter::new(f));
        }
    }
    if let Ok(path) = std::env::var("DOUBLY_EVEN_DECOMP_LOG") {
        let path = path.trim();
        if !path.is_empty() {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap_or_else(|e| {
                    panic!("DOUBLY_EVEN_DECOMP_LOG: cannot open {path:?}: {e}")
                });
            state.install_decomp_log(std::io::BufWriter::new(f));
        }
    }
}

/// Interleaved appends from many workers are not sound for arbitrary
/// record sizes, so the tie dump is sequential-only; per-worker suffixed
/// files are the documented future alternative if ever needed.
#[cfg(feature = "parallel")]
fn reject_tie_dump_in_parallel() {
    for var in ["DOUBLY_EVEN_TIE_DUMP", "DOUBLY_EVEN_DECOMP_LOG"] {
        let set = std::env::var(var)
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        if set {
            panic!(
                "{var} is sequential-only; unset DOUBLY_EVEN_THREADS \
                 or run the sequential driver"
            );
        }
    }
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
    let quota = widen_quota(quota);
    reject_tie_dump_in_parallel();
    // One labelling-mode resolution for seeder + all workers.
    let labelling = LabelMode::from_env();

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
    // The worker's *local* quota stays U256::MAX (the local panic check
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
            let inf_quota = vec![U256::MAX; (mk + 1) as usize];
            let mut worker = WorkerState::new(nn, mk, inf_quota, fact, rule, labelling);
            worker.install_global_mass(gm);
            while let Ok(seed) = task_rx.recv() {
                worker.traverse(seed.rref, seed.pivots, Rc::new(seed.info));
            }
            worker.flush_global_mass();
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
    let mut seed_state =
        WorkerState::new(n, max_k, quota.clone(), factorial_n, rule, labelling);
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
        let zero_info = seed_state.canon_info(&zero_rref, false);
        seed_state.traverse_seed(
            zero_rref,
            zero_pivots,
            zero_info,
            frontier_depth,
            &task_tx,
        );
    }
    seed_state.clear_seeder_pool();
    seed_state.flush_global_mass();
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
fn assert_mass_matches_quota(mass: &[U256], quota: &[U256], max_k: u32) {
    let limit = ((max_k as usize) + 1).min(mass.len()).min(quota.len());
    let mut diffs: Vec<String> = Vec::new();
    for k in 0..limit {
        if mass[k] != quota[k] {
            let line = if mass[k] > quota[k] {
                format!("  k={k}: mass={}, quota={}, excess=+{}", mass[k], quota[k], mass[k].checked_sub(quota[k]))
            } else {
                format!("  k={k}: mass={}, quota={}, shortfall=-{}", mass[k], quota[k], quota[k].checked_sub(mass[k]))
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
    /// U256 since 2026-06-13 (σ(30, ·) overflows u128); the pyo3 layer
    /// already serialised mass as decimal strings, so the Python-facing
    /// format is unchanged.
    pub mass: Vec<U256>,
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
    let quota = widen_quota(quota);
    let path = output_dir.join("out.w000.bin");
    let writer = BinaryWriter::create(&path, n, 0)
        .expect("failed to create streaming output file");
    let quota_for_assert = quota.clone();
    let rule = ParentRule::from_env();
    let mut state =
        WorkerState::new(n, max_k, quota, factorial_n, rule, LabelMode::from_env());
    state.install_output_writer(writer);
    install_tie_dump_from_env(&mut state);
    let zero_rref: Vec<BinVec> = Vec::new();
    let zero_pivots: Vec<u32> = Vec::new();
    let info = state.canon_info(&zero_rref, false);
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
    let quota = widen_quota(quota);

    let cap = (num_threads * 4).max(8);
    let (task_tx, task_rx) = bounded::<SeedFrontier>(cap);
    let (result_tx, result_rx) = unbounded::<(Vec<u128>, Vec<Vec<u64>>)>();

    reject_tie_dump_in_parallel();
    let quota_for_assert = quota.clone();
    let global_mass = std::sync::Arc::new(GlobalMassTracker::new(quota.clone()));

    // One rule + labelling-mode resolution for seeder + all workers.
    let rule = ParentRule::from_env();
    let labelling = LabelMode::from_env();

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
            let inf_quota = vec![U256::MAX; (mk + 1) as usize];
            let mut worker = WorkerState::new(nn, mk, inf_quota, fact, rule, labelling);
            worker.install_global_mass(gm);
            worker.install_output_writer(writer);
            while let Ok(seed) = task_rx.recv() {
                worker.traverse(seed.rref, seed.pivots, Rc::new(seed.info));
            }
            worker.flush_global_mass();
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
    let mut seed_state =
        WorkerState::new(n, max_k, quota.clone(), factorial_n, rule, labelling);
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
        let zero_info = seed_state.canon_info(&zero_rref, false);
        seed_state.traverse_seed(
            zero_rref,
            zero_pivots,
            zero_info,
            frontier_depth,
            &task_tx,
        );
    }
    seed_state.clear_seeder_pool();
    seed_state.flush_global_mass();
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

/// Output of the counts-only drivers (the N ≥ 30 mode — eval plan §5).
/// No per-class records exist anywhere: the only artefacts are the
/// per-rank aggregates below (~KBs at any N) and the in-Rust
/// mass-formula gate, which remains the correctness certificate.
pub struct CountsResult {
    pub stats: Vec<u128>,
    pub per_k_stats: Vec<Vec<u64>>,
    /// Per-rank Σ N!/|Aut| — validated `== quota[k]` before returning.
    pub mass: Vec<U256>,
    /// Per-rank emitted-class counts.
    pub classes: Vec<u64>,
    /// Per-rank |Aut| histogram, ascending by |Aut|.
    pub aut_hist: Vec<Vec<(u128, u64)>>,
}

/// Progress sink for the counts drivers: every `interval_s` seconds the
/// watcher snapshots the shared mass tracker and atomically rewrites
/// `<path>` (write to `.tmp`, rename) with per-rank mass vs quota as
/// decimal strings — exactly the "fraction of σ(N, k) mass found"
/// signal `dec progress` renders. One final snapshot is written when
/// the run completes.
#[cfg(feature = "parallel")]
pub struct ProgressSink {
    pub path: std::path::PathBuf,
    pub interval_s: u64,
}

#[cfg(feature = "parallel")]
fn write_progress_json(
    path: &std::path::Path,
    n: u32,
    max_k: u32,
    elapsed_s: f64,
    done: bool,
    mass: &[U256],
    quota: &[U256],
) {
    let rows: Vec<String> = mass
        .iter()
        .zip(quota.iter())
        .map(|(m, q)| format!("[\"{m}\",\"{q}\"]"))
        .collect();
    let body = format!(
        "{{\"n\":{n},\"max_k\":{max_k},\"elapsed_s\":{elapsed_s:.1},\"done\":{done},\
         \"mass_quota\":[{}]}}\n",
        rows.join(",")
    );
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, &body).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// Spawn the progress watcher. Returns a stop flag + handle; set the
/// flag and join once the traversal is done — the watcher writes a
/// final `done: true` snapshot on exit.
#[cfg(feature = "parallel")]
fn spawn_progress_watcher(
    tracker: std::sync::Arc<GlobalMassTracker>,
    quota: Vec<U256>,
    n: u32,
    max_k: u32,
    sink: ProgressSink,
) -> (
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    std::thread::JoinHandle<()>,
) {
    use std::sync::atomic::{AtomicBool, Ordering};
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let stop_w = std::sync::Arc::clone(&stop);
    let handle = std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        let mut last_write = std::time::Instant::now();
        write_progress_json(&sink.path, n, max_k, 0.0, false, &tracker.snapshot(), &quota);
        while !stop_w.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(250));
            if last_write.elapsed().as_secs() >= sink.interval_s.max(1) {
                write_progress_json(
                    &sink.path,
                    n,
                    max_k,
                    t0.elapsed().as_secs_f64(),
                    false,
                    &tracker.snapshot(),
                    &quota,
                );
                last_write = std::time::Instant::now();
            }
        }
        write_progress_json(
            &sink.path,
            n,
            max_k,
            t0.elapsed().as_secs_f64(),
            true,
            &tracker.snapshot(),
            &quota,
        );
    });
    (stop, handle)
}

/// Merge per-worker counts folds (element-wise classes, histogram union).
fn merge_counts(
    classes: &mut [u64],
    hist: &mut [std::collections::HashMap<u128, u64>],
    other_classes: Vec<u64>,
    other_hist: Vec<std::collections::HashMap<u128, u64>>,
) {
    for (a, b) in classes.iter_mut().zip(other_classes) {
        *a += b;
    }
    for (a, b) in hist.iter_mut().zip(other_hist) {
        for (aut, count) in b {
            *a.entry(aut).or_insert(0) += count;
        }
    }
}

fn sort_hist(hist: Vec<std::collections::HashMap<u128, u64>>) -> Vec<Vec<(u128, u64)>> {
    hist.into_iter()
        .map(|h| {
            let mut v: Vec<(u128, u64)> = h.into_iter().collect();
            v.sort_unstable_by_key(|&(aut, _)| aut);
            v
        })
        .collect()
}

/// Counts-only sequential driver. Takes quota as `Vec<U256>` directly —
/// this is the one entry that must work at N ≥ 30, where σ(N, k)
/// overflows both u128 and pyo3's int conversion (the Python layer
/// passes decimal strings). With the `parallel` feature a progress sink
/// can be attached (the watcher polls a shared tracker); without it the
/// sink is ignored beyond a final write.
pub fn enumerate_doubly_even_counts(
    n: u32,
    max_k: u32,
    quota: Vec<U256>,
    factorial_n: u128,
    #[cfg(feature = "parallel")] progress: Option<ProgressSink>,
) -> CountsResult {
    let quota_for_assert = quota.clone();
    let rule = ParentRule::from_env();
    let mut state =
        WorkerState::new(n, max_k, quota.clone(), factorial_n, rule, LabelMode::from_env());
    state.counts_only = true;

    #[cfg(feature = "parallel")]
    let watcher = progress.map(|sink| {
        let tracker = std::sync::Arc::new(GlobalMassTracker::new(quota.clone()));
        state.install_global_mass(std::sync::Arc::clone(&tracker));
        spawn_progress_watcher(tracker, quota, n, max_k, sink)
    });

    let zero_rref: Vec<BinVec> = Vec::new();
    let zero_pivots: Vec<u32> = Vec::new();
    let info = state.canon_info(&zero_rref, false);
    state.traverse(zero_rref, zero_pivots, info);

    let mass = state.mass_snapshot();
    let (classes, hist) = state.take_counts();
    let (_empty, stats, per_k_stats) = state.finalize();
    debug_assert!(_empty.is_empty(), "counts mode must produce empty in-memory output");

    #[cfg(feature = "parallel")]
    if let Some((stop, handle)) = watcher {
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        handle.join().expect("progress watcher panicked");
    }

    assert_mass_matches_quota(&mass, &quota_for_assert, max_k);
    CountsResult {
        stats,
        per_k_stats,
        mass,
        classes,
        aut_hist: sort_hist(hist),
    }
}

/// Counts-only parallel driver. Same fan-out as
/// [`enumerate_doubly_even_parallel_streaming`], but workers fold
/// per-rank aggregates instead of writing class records; the shared
/// mass tracker doubles as the live progress source.
#[cfg(feature = "parallel")]
pub fn enumerate_doubly_even_parallel_counts(
    n: u32,
    max_k: u32,
    quota: Vec<U256>,
    factorial_n: u128,
    num_threads: usize,
    progress: Option<ProgressSink>,
) -> CountsResult {
    use crossbeam_channel::{bounded, unbounded};

    let frontier_depth: u32 = std::env::var("DOUBLY_EVEN_FRONTIER_DEPTH")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&d| d >= 2)
        .unwrap_or(4);

    if num_threads <= 1 || max_k <= frontier_depth {
        return enumerate_doubly_even_counts(n, max_k, quota, factorial_n, progress);
    }

    type WorkerResult = (
        Vec<u128>,
        Vec<Vec<u64>>,
        Vec<u64>,
        Vec<std::collections::HashMap<u128, u64>>,
    );

    let cap = (num_threads * 4).max(8);
    let (task_tx, task_rx) = bounded::<SeedFrontier>(cap);
    let (result_tx, result_rx) = unbounded::<WorkerResult>();

    reject_tie_dump_in_parallel();
    let quota_for_assert = quota.clone();
    let global_mass = std::sync::Arc::new(GlobalMassTracker::new(quota.clone()));
    let watcher = progress.map(|sink| {
        spawn_progress_watcher(
            std::sync::Arc::clone(&global_mass),
            quota.clone(),
            n,
            max_k,
            sink,
        )
    });

    let rule = ParentRule::from_env();
    let labelling = LabelMode::from_env();

    let mut handles = Vec::with_capacity(num_threads);
    for _ in 0..num_threads {
        let task_rx = task_rx.clone();
        let result_tx = result_tx.clone();
        let mk = max_k;
        let nn = n;
        let fact = factorial_n;
        let gm = std::sync::Arc::clone(&global_mass);
        handles.push(std::thread::spawn(move || {
            let inf_quota = vec![U256::MAX; (mk + 1) as usize];
            let mut worker = WorkerState::new(nn, mk, inf_quota, fact, rule, labelling);
            worker.counts_only = true;
            worker.install_global_mass(gm);
            while let Ok(seed) = task_rx.recv() {
                worker.traverse(seed.rref, seed.pivots, Rc::new(seed.info));
            }
            worker.flush_global_mass();
            let (classes, hist) = worker.take_counts();
            let (_empty, stats, per_k) = worker.finalize();
            let _ = result_tx.send((stats, per_k, classes, hist));
        }));
    }
    drop(task_rx);
    drop(result_tx);

    let mut seed_state =
        WorkerState::new(n, max_k, quota.clone(), factorial_n, rule, labelling);
    seed_state.counts_only = true;
    seed_state.install_global_mass(std::sync::Arc::clone(&global_mass));
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
        let zero_info = seed_state.canon_info(&zero_rref, false);
        seed_state.traverse_seed(
            zero_rref,
            zero_pivots,
            zero_info,
            frontier_depth,
            &task_tx,
        );
    }
    seed_state.clear_seeder_pool();
    seed_state.flush_global_mass();
    drop(task_tx);

    let (mut classes, mut hist) = seed_state.take_counts();
    let (_empty, mut combined_stats, mut combined_per_k) = seed_state.finalize();
    for _ in 0..num_threads {
        let (stats, per_k, w_classes, w_hist) = result_rx
            .recv()
            .expect("worker thread hung up before sending result");
        merge_stats_only(&mut combined_stats, &mut combined_per_k, stats, per_k);
        merge_counts(&mut classes, &mut hist, w_classes, w_hist);
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }

    let mass = global_mass.snapshot();
    if let Some((stop, handle)) = watcher {
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        handle.join().expect("progress watcher panicked");
    }
    assert_mass_matches_quota(&mass, &quota_for_assert, max_k);
    CountsResult {
        stats: combined_stats,
        per_k_stats: combined_per_k,
        mass,
        classes,
        aut_hist: sort_hist(hist),
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
