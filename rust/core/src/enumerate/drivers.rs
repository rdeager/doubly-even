//! Sequential / parallel / streaming drivers, the seed-frontier plumbing
//! and the in-Rust mass-formula gate. Bodies are verbatim from the
//! original `enumerate.rs`.

#[cfg(feature = "parallel")]
use std::rc::Rc;

use crate::parent_rule::ParentRule;
use crate::streaming::BinaryWriter;
use crate::types::BinVec;

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
    let mut state =
        WorkerState::new(n, max_k, quota, factorial_n, rule, LabelMode::from_env());
    install_tie_dump_from_env(&mut state);
    run_sequential(state)
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
}

/// Interleaved appends from many workers are not sound for arbitrary
/// record sizes, so the tie dump is sequential-only; per-worker suffixed
/// files are the documented future alternative if ever needed.
#[cfg(feature = "parallel")]
fn reject_tie_dump_in_parallel() {
    let set = std::env::var("DOUBLY_EVEN_TIE_DUMP")
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if set {
        panic!(
            "DOUBLY_EVEN_TIE_DUMP is sequential-only; unset DOUBLY_EVEN_THREADS \
             or run the sequential driver"
        );
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
            let mut worker = WorkerState::new(nn, mk, inf_quota, fact, rule, labelling);
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
            let inf_quota = vec![u128::MAX; (mk + 1) as usize];
            let mut worker = WorkerState::new(nn, mk, inf_quota, fact, rule, labelling);
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
