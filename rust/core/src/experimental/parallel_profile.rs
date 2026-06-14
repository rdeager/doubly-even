//! crate::experimental::parallel_profile — parallel-scaling profile
//! entry point (Phase 3, re-pipelined for the seeder timeline 2026-06-10).
//!
//! Mirrors `enumerate::enumerate_doubly_even_parallel_with_seeder` —
//! workers spawn first and block on the bounded task channel, the seeder
//! runs pipelined on the main thread with the `GlobalMassTracker` and the
//! D16 seeder helper pool installed — and wraps each worker's seed loop
//! with `Instant`-based timing plus snapshots of the in-flight stats
//! counters so we can recover per-seed work. On top of the original
//! per-worker/per-seed rows it captures the SEEDER TIMELINE: per-seed
//! enqueue timestamps (ready/sent, exposing bounded-channel backpressure)
//! and per-rank σ_Q call spans, all relative to one run-start epoch.
//!
//! HISTORY NOTE: before 2026-06-10 this harness was a two-phase shim
//! (eager frontier collection, then dispatch; no seeder pool, no global
//! mass tracker) — see git history. Profile payloads recorded by that
//! shim (`d16-par-profile-*`) are NOT comparable to payloads from this
//! pipelined driver: the old `total_wall_ns` serialized the seeder span
//! ahead of all worker activity and recv-idle was structurally ~0.
//!
//! Gated behind Cargo feature `parallel_profiling` (default OFF) so the
//! production parallel build stays byte-identical. The findings driven by
//! this entry live in `/workspace/markdown/architecture/07-parallel-
//! scaling-profile.md` and `memory/project_parallel_profile_2026_05_20.md`.

use std::rc::Rc;

use crate::enumerate::{
    enumerate_doubly_even, merge_finalized, EnumeratedRaw, GlobalMassTracker, LoadBalancer,
    SeedFrontier, SelfSubdivideCfg, WorkerState,
};
use crate::types::BinVec;
use crate::u256::U256;

#[derive(Clone, Debug)]
pub struct WorkerProfile {
    pub worker_id: u32,
    pub active_ns: u64,
    pub idle_ns: u64,
    pub seed_count: u32,
}

#[derive(Clone, Debug)]
pub struct SeedProfile {
    pub worker_id: u32,
    pub seed_id: u32,
    /// Epoch-relative time at which this worker's `recv()` returned the
    /// seed — the moment work on it could begin. Joined with the seeder's
    /// `enqueues[seed_id]` this gives per-seed queue latency, and the
    /// interval `[start_ns, start_ns + ns)` feeds the busy-worker
    /// sweep-line that locates the starved window.
    pub start_ns: u64,
    pub ns: u64,
    /// Stand-in for "recursion nodes visited under this seed" — the
    /// `is_canonical_augmentation` call counter, snapshotted before and
    /// after the seed's `traverse(...)` call.
    pub nodes: u64,
    /// Classes emitted under this seed (output Vec growth delta).
    pub emitted: u32,
}

/// Seeder-side timeline, all timestamps relative to the run-start epoch
/// (the same epoch `total_wall_ns` and `SeedProfile::start_ns` use).
#[derive(Default, Debug)]
pub struct SeederTimeline {
    /// `(ready_ns, sent_ns)` per seed, in send order; index == `seed_id`.
    /// `sent − ready` is the bounded-channel backpressure wait — the
    /// direct "workers saturated" signal.
    pub enqueues: Vec<(u64, u64)>,
    /// `(k, l, pooled, start_ns, end_ns)` per seeder σ_Q call.
    pub sigma_spans: Vec<(u32, u32, bool, u64, u64)>,
    /// Epoch-relative instant at which `traverse_seed` returned and the
    /// helper pool was dropped (start of the pure drain tail).
    pub seeder_done_ns: u64,
}

#[derive(Default, Debug)]
pub struct ParallelProfile {
    pub workers: Vec<WorkerProfile>,
    pub seeds: Vec<SeedProfile>,
    pub frontier_depth: u32,
    pub total_wall_ns: u64,
    pub seeder: SeederTimeline,
    /// D20 behavioral check: total children donated and the deepest parent
    /// depth at which a donation fired. With self-subdivision off both are
    /// 0; on, `donation_max_k <= frontier_depth + delta` must hold.
    pub donations: u64,
    pub donation_max_k: u32,
}

/// Profiling variant of `enumerate::enumerate_doubly_even_parallel`.
/// Returns the usual `(output, stats_flat, per_k_stats)` plus a
/// [`ParallelProfile`] with one row per worker (active/idle ns, seed
/// count), one row per seed (worker id, start, ns, nodes-visited proxy,
/// classes-emitted), and the seeder timeline.
///
/// All algorithmic decisions match the production entry — pipelined
/// seeder, bounded channel of the same capacity, shared mass tracker,
/// env-resolved seeder helper pool — so the timing data reflects
/// production overlap and contention within timing-overhead noise.
pub fn enumerate_doubly_even_parallel_with_profile(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    num_threads: usize,
) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>, ParallelProfile) {
    use crossbeam_channel::{bounded, unbounded};
    use std::time::Instant;

    let frontier_depth: u32 = std::env::var("DOUBLY_EVEN_FRONTIER_DEPTH")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&d| d >= 2)
        .unwrap_or(4);

    let total_t0 = Instant::now();

    if num_threads <= 1 || max_k <= frontier_depth {
        // Fall back to sequential; report a single "worker 0" rolled-up
        // entry so downstream tooling can assume `workers` is non-empty.
        let (out, stats, per_k) = enumerate_doubly_even(n, max_k, quota, factorial_n);
        let total_ns = total_t0.elapsed().as_nanos() as u64;
        let profile = ParallelProfile {
            workers: vec![WorkerProfile {
                worker_id: 0,
                active_ns: total_ns,
                idle_ns: 0,
                seed_count: 0,
            }],
            seeds: Vec::new(),
            frontier_depth,
            total_wall_ns: total_ns,
            seeder: SeederTimeline::default(),
            donations: 0,
            donation_max_k: 0,
        };
        return (out, stats, per_k, profile);
    }

    let rule = crate::parent_rule::ParentRule::from_env();
    let labelling = crate::enumerate::LabelMode::from_env();

    // The U256 migration (2026-06-13) widened GlobalMassTracker / WorkerState
    // quotas; this experimental profiling driver was missed, so widen here too
    // (the sequential fallback above already consumed the u128 `quota`).
    let quota: Vec<U256> = quota.into_iter().map(U256::from).collect();

    // Worker pool spawns first and waits on the (initially empty) bounded
    // channel — identical shape to the production driver, so seeder DFS
    // overlaps worker recursion and backpressure is real.
    let cap = (num_threads * 4).max(8);
    let (task_tx, task_rx) = bounded::<SeedFrontier>(cap);
    type WorkerResult = (
        Vec<EnumeratedRaw>,
        Vec<u128>,
        Vec<Vec<u64>>,
        WorkerProfile,
        Vec<SeedProfile>,
    );
    let (result_tx, result_rx) = unbounded::<WorkerResult>();

    let global_mass = std::sync::Arc::new(GlobalMassTracker::new(quota.clone()));
    let balancer = std::sync::Arc::new(LoadBalancer::new());
    // D20 demand-driven self-subdivision (default OFF ⇒ original loop).
    let ss = SelfSubdivideCfg::from_env();

    let mut handles = Vec::with_capacity(num_threads);
    for worker_id in 0..num_threads as u32 {
        let task_rx = task_rx.clone();
        let result_tx = result_tx.clone();
        let mk = max_k;
        let nn = n;
        let fact = factorial_n;
        let gm = std::sync::Arc::clone(&global_mass);
        let bal = std::sync::Arc::clone(&balancer);
        let donor_tx = if ss.enabled { Some(task_tx.clone()) } else { None };
        let (ss_on, ss_poll, ss_delta, fd) = (ss.enabled, ss.poll, ss.delta, frontier_depth);
        let epoch = total_t0; // Instant is Copy; one shared epoch.
        handles.push(std::thread::spawn(move || {
            use std::sync::atomic::Ordering;
            let inf_quota = vec![U256::MAX; (mk + 1) as usize];
            let mut worker = WorkerState::new(nn, mk, inf_quota, fact, rule, labelling);
            worker.install_global_mass(gm);
            if ss_on {
                worker.install_self_subdivide(std::sync::Arc::clone(&bal), donor_tx, fd, ss_delta);
            }
            let mut seed_profiles: Vec<SeedProfile> = Vec::new();
            let mut active_ns: u64 = 0;
            let mut idle_ns: u64 = 0;
            let mut seed_count: u32 = 0;
            loop {
                if ss_on {
                    bal.idle_workers.fetch_add(1, Ordering::Relaxed);
                }
                let recv_t0 = Instant::now();
                // OFF: blocking recv (ends on seeder's drop, original shape).
                // ON: timed poll; exit only when seeder_done && outstanding==0.
                let item = if ss_on {
                    task_rx.recv_timeout(ss_poll)
                } else {
                    task_rx
                        .recv()
                        .map_err(|_| crossbeam_channel::RecvTimeoutError::Disconnected)
                };
                let idle_dt = recv_t0.elapsed().as_nanos() as u64;
                idle_ns += idle_dt;
                if ss_on {
                    bal.idle_workers.fetch_sub(1, Ordering::Relaxed);
                }
                let seed = match item {
                    Ok(x) => x,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        if bal.seeder_done.load(Ordering::Acquire)
                            && bal.outstanding.load(Ordering::Acquire) == 0
                        {
                            break;
                        }
                        continue;
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                };
                let start_ns = epoch.elapsed().as_nanos() as u64;
                let seed_id = seed.seed_id;
                let nodes_before = worker.stats_is_canon_aug_calls;
                let emitted_before = worker.output.len() as u32;
                let active_t0 = Instant::now();
                worker.traverse(seed.rref, seed.pivots, Rc::new(seed.info));
                let active_dt = active_t0.elapsed().as_nanos() as u64;
                active_ns += active_dt;
                if ss_on {
                    bal.outstanding.fetch_sub(1, Ordering::AcqRel);
                }
                seed_count += 1;
                seed_profiles.push(SeedProfile {
                    worker_id,
                    seed_id,
                    start_ns,
                    ns: active_dt,
                    nodes: worker.stats_is_canon_aug_calls - nodes_before,
                    emitted: worker.output.len() as u32 - emitted_before,
                });
            }
            worker.flush_global_mass();
            let (out, stats, per_k) = worker.finalize();
            let profile = WorkerProfile {
                worker_id,
                active_ns,
                idle_ns,
                seed_count,
            };
            let _ = result_tx.send((out, stats, per_k, profile, seed_profiles));
        }));
    }
    drop(task_rx);
    drop(result_tx);

    // Seeder on the main thread, pipelined into the bounded channel,
    // with the epoch armed so traverse_seed records the timeline.
    let mut seed_state = WorkerState::new(n, max_k, quota.clone(), factorial_n, rule, labelling);
    seed_state.install_global_mass(std::sync::Arc::clone(&global_mass));
    if ss.enabled {
        seed_state.install_self_subdivide(std::sync::Arc::clone(&balancer), None, frontier_depth, ss.delta);
    }
    let (seeder_threads, seeder_min_l) = crate::seeder_pool::SeederPool::env_defaults(num_threads);
    if seeder_threads >= 2 {
        seed_state.install_seeder_pool(std::sync::Arc::new(
            crate::seeder_pool::SeederPool::new(seeder_threads, seeder_min_l),
        ));
    }
    seed_state.profile_epoch = Some(total_t0);
    {
        let zero_rref: Vec<BinVec> = Vec::new();
        let zero_pivots: Vec<u32> = Vec::new();
        let zero_info = seed_state.canon_info(&zero_rref, false);
        seed_state.traverse_seed(zero_rref, zero_pivots, zero_info, frontier_depth, &task_tx);
    }
    seed_state.clear_seeder_pool();
    seed_state.flush_global_mass();
    if ss.enabled {
        balancer
            .seeder_done
            .store(true, std::sync::atomic::Ordering::Release);
    }
    let seeder_done_ns = total_t0.elapsed().as_nanos() as u64;
    drop(task_tx);

    let seeder_timeline = SeederTimeline {
        enqueues: std::mem::take(&mut seed_state.profile_enqueues),
        sigma_spans: std::mem::take(&mut seed_state.profile_sigma_spans),
        seeder_done_ns,
    };

    let mut combined = seed_state.finalize();
    let mut worker_profiles: Vec<WorkerProfile> = Vec::with_capacity(num_threads);
    let mut all_seeds: Vec<SeedProfile> = Vec::new();
    for _ in 0..num_threads {
        let (out, stats, per_k, wprof, sprofs) = result_rx
            .recv()
            .expect("worker thread hung up before sending result");
        merge_finalized(&mut combined, (out, stats, per_k));
        worker_profiles.push(wprof);
        all_seeds.extend(sprofs);
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }

    worker_profiles.sort_by_key(|w| w.worker_id);
    all_seeds.sort_by_key(|s| (s.worker_id, s.seed_id));

    let total_wall_ns = total_t0.elapsed().as_nanos() as u64;
    let donations = balancer
        .donations
        .load(std::sync::atomic::Ordering::Relaxed) as u64;
    let donation_max_k = balancer
        .donation_max_k
        .load(std::sync::atomic::Ordering::Relaxed) as u32;
    let profile = ParallelProfile {
        workers: worker_profiles,
        seeds: all_seeds,
        frontier_depth,
        total_wall_ns,
        seeder: seeder_timeline,
        donations,
        donation_max_k,
    };
    (combined.0, combined.1, combined.2, profile)
}
