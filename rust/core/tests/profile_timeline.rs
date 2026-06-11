//! Seeder-timeline profile-entry harness (2026-06-10 re-pipeline).
//!
//! Runs `enumerate_doubly_even_parallel_with_profile` at small N and
//! asserts (1) correctness is unchanged — per-rank class counts equal the
//! sequential driver's — and (2) the timeline payload is internally
//! consistent: enqueue timestamps monotone in send order, seed counts
//! agree across the three places they're recorded, σ_Q spans ordered and
//! inside the seeder span, and worker seed start/durations inside the
//! run wall.
//!
//! σ(N, k) constants from `doubly_even.spec.mass.gaborit_sigma`, same
//! pattern as `parallel_determinism.rs`.

#![cfg(feature = "parallel_profiling")]

use doubly_even_core::enumerate::enumerate_doubly_even;
use doubly_even_core::experimental::parallel_profile::enumerate_doubly_even_parallel_with_profile;

/// σ(14, k) for k = 0..7.
const SIGMA_N14: [u128; 8] = [
    1,
    4_095,
    1_396_395,
    50_868_675,
    213_648_435,
    103_378_275,
    4_922_775,
    0,
];
const FACT_N14: u128 = 87_178_291_200;

fn per_rank_counts(out: &[doubly_even_core::enumerate::EnumeratedRaw]) -> Vec<usize> {
    let max_k = out.iter().map(|e| e.rref.len()).max().unwrap_or(0);
    let mut counts = vec![0usize; max_k + 1];
    for e in out {
        counts[e.rref.len()] += 1;
    }
    counts
}

#[test]
fn timeline_payload_consistent_n14() {
    let (seq_out, _, _) =
        enumerate_doubly_even(14, 7, SIGMA_N14.to_vec(), FACT_N14);
    for threads in [2usize, 4, 8] {
        let (out, _stats, _per_k, profile) = enumerate_doubly_even_parallel_with_profile(
            14,
            7,
            SIGMA_N14.to_vec(),
            FACT_N14,
            threads,
        );
        assert_eq!(
            per_rank_counts(&out),
            per_rank_counts(&seq_out),
            "per-rank class counts diverge at threads={threads}"
        );

        let tl = &profile.seeder;
        // Seed counts agree: enqueues == worker seed rows == Σ seed_count.
        assert_eq!(tl.enqueues.len(), profile.seeds.len());
        let total_seed_count: u32 = profile.workers.iter().map(|w| w.seed_count).sum();
        assert_eq!(tl.enqueues.len() as u32, total_seed_count);
        // seed_ids are exactly 0..len.
        let mut ids: Vec<u32> = profile.seeds.iter().map(|s| s.seed_id).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..tl.enqueues.len() as u32).collect::<Vec<_>>());

        // Enqueue timestamps: ready <= sent, monotone in send order, all
        // within the seeder span.
        let mut prev_sent = 0u64;
        for &(ready, sent) in &tl.enqueues {
            assert!(ready <= sent, "ready_ns > sent_ns");
            assert!(sent >= prev_sent, "sent_ns not monotone in send order");
            assert!(sent <= tl.seeder_done_ns, "enqueue after seeder done");
            prev_sent = sent;
        }

        // σ_Q spans: ordered, inside the seeder span, plausible (k, L).
        for &(k, l, _pooled, start, end) in &tl.sigma_spans {
            assert!(start <= end);
            assert!(end <= tl.seeder_done_ns, "sigma span after seeder done");
            assert!(k < 7, "sigma span at impossible rank");
            assert_eq!(l, 14 - 2 * k, "L != N - 2k");
        }

        // Worker seed rows: inside the wall; a seed can only start after
        // it became ready on the seeder. (NOT after `sent_ns`: with a
        // waiting receiver, crossbeam hands the item to the worker
        // directly, and the worker may stamp `start_ns` before the
        // seeder returns from `send()` and stamps `sent_ns`.)
        let ready_by_id: Vec<u64> = tl.enqueues.iter().map(|e| e.0).collect();
        for s in &profile.seeds {
            assert!(s.start_ns + s.ns <= profile.total_wall_ns);
            assert!(
                s.start_ns >= ready_by_id[s.seed_id as usize],
                "seed started before it was ready to enqueue"
            );
        }
    }
}
