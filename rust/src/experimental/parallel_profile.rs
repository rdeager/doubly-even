//! crate::experimental::parallel_profile — Phase 3 parallel-scaling
//! profile entry point.
//!
//! Mirrors `enumerate::enumerate_doubly_even_parallel` but wraps each
//! worker's seed loop with `Instant`-based timing and snapshots the
//! in-flight stats counters so we can recover per-seed work. The kernel
//! is conceptually identical — every algorithmic decision (frontier_depth,
//! per-worker inf_quota, channel cap) matches the production entry.
//! Diverges only in the channel item type (`(u32 seed_id, SeedFrontier)`)
//! and the extra `(WorkerProfile, Vec<SeedProfile>)` shipped back through
//! the result channel.
//!
//! Gated behind Cargo feature `parallel_profiling` (default OFF) so the
//! production parallel build stays byte-identical. The findings driven by
//! this entry live in `/workspace/markdown/architecture/07-parallel-
//! scaling-profile.md` and `memory/project_parallel_profile_2026_05_20.md`.

use std::rc::Rc;

use crate::enumerate::{
    enumerate_doubly_even, merge_finalized, EnumeratedRaw, SeedFrontier, WorkerState,
};
use crate::types::BinVec;

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
    pub ns: u64,
    /// Stand-in for "recursion nodes visited under this seed" — the
    /// `is_canonical_augmentation` call counter, snapshotted before and
    /// after the seed's `traverse(...)` call.
    pub nodes: u64,
    /// Classes emitted under this seed (output Vec growth delta).
    pub emitted: u32,
}

#[derive(Default, Debug)]
pub struct ParallelProfile {
    pub workers: Vec<WorkerProfile>,
    pub seeds: Vec<SeedProfile>,
    pub frontier_depth: u32,
    pub total_wall_ns: u64,
}

/// Profiling variant of `enumerate::enumerate_doubly_even_parallel`.
/// Returns the usual `(output, stats_flat, per_k_stats)` plus a
/// [`ParallelProfile`] with one row per worker (active/idle ns, seed
/// count) and one row per seed (worker id, ns, nodes-visited proxy,
/// classes-emitted).
///
/// All algorithmic decisions match the production entry so the timing
/// data is comparable to a production run within timing-overhead noise.
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
        };
        return (out, stats, per_k, profile);
    }

    let mut seed_state = WorkerState::new(n, max_k, quota.clone(), factorial_n);
    let mut frontier: Vec<SeedFrontier> = Vec::new();
    {
        let zero_rref: Vec<BinVec> = Vec::new();
        let zero_pivots: Vec<u32> = Vec::new();
        let zero_info = seed_state.canon_info(&zero_rref);
        seed_state.traverse_seed(
            zero_rref,
            zero_pivots,
            zero_info,
            frontier_depth,
            &mut frontier,
        );
    }
    if frontier.is_empty() {
        let (out, stats, per_k) = seed_state.finalize();
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
        };
        return (out, stats, per_k, profile);
    }

    let cap = (num_threads * 4).max(8);
    // Channel carries (seed_id, SeedFrontier) so workers can attribute
    // timing back to a stable identifier across runs.
    let (task_tx, task_rx) = bounded::<(u32, SeedFrontier)>(cap);
    type WorkerResult = (
        Vec<EnumeratedRaw>,
        Vec<u128>,
        Vec<Vec<u64>>,
        WorkerProfile,
        Vec<SeedProfile>,
    );
    let (result_tx, result_rx) = unbounded::<WorkerResult>();

    let mut handles = Vec::with_capacity(num_threads);
    for worker_id in 0..num_threads as u32 {
        let task_rx = task_rx.clone();
        let result_tx = result_tx.clone();
        let mk = max_k;
        let nn = n;
        let fact = factorial_n;
        handles.push(std::thread::spawn(move || {
            let inf_quota = vec![u128::MAX; (mk + 1) as usize];
            let mut worker = WorkerState::new(nn, mk, inf_quota, fact);
            let mut seed_profiles: Vec<SeedProfile> = Vec::new();
            let mut active_ns: u64 = 0;
            let mut idle_ns: u64 = 0;
            let mut seed_count: u32 = 0;
            loop {
                let recv_t0 = Instant::now();
                let item = task_rx.recv();
                let idle_dt = recv_t0.elapsed().as_nanos() as u64;
                idle_ns += idle_dt;
                let (seed_id, seed) = match item {
                    Ok(x) => x,
                    Err(_) => break,
                };
                let nodes_before = worker.stats_is_canon_aug_calls;
                let emitted_before = worker.output.len() as u32;
                let active_t0 = Instant::now();
                worker.traverse(seed.rref, seed.pivots, Rc::new(seed.info));
                let active_dt = active_t0.elapsed().as_nanos() as u64;
                active_ns += active_dt;
                seed_count += 1;
                seed_profiles.push(SeedProfile {
                    worker_id,
                    seed_id,
                    ns: active_dt,
                    nodes: worker.stats_is_canon_aug_calls - nodes_before,
                    emitted: worker.output.len() as u32 - emitted_before,
                });
            }
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

    for (seed_id, seed) in frontier.into_iter().enumerate() {
        task_tx
            .send((seed_id as u32, seed))
            .expect("worker pool closed unexpectedly");
    }
    drop(task_tx);

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
    let profile = ParallelProfile {
        workers: worker_profiles,
        seeds: all_seeds,
        frontier_depth,
        total_wall_ns,
    };
    (combined.0, combined.1, combined.2, profile)
}
