//! D13: parallel-driver correctness harness.
//!
//! Runs `enumerate_doubly_even_parallel` at a handful of thread counts and
//! asserts that:
//!
//! 1. The set of canonical RREFs emitted is identical to the sequential
//!    driver's output (regardless of thread count).
//! 2. Per-rank class counts match between sequential and parallel runs.
//! 3. The `aut_order` field per canonical RREF matches.
//!
//! Coverage: `N = 12, 14`. `N = 12, max_k = 6` has 7 ranks and 21 depth-3
//! frontier nodes, enough to exercise the worker pool. `N = 14, max_k = 7`
//! is larger and stresses load balance across workers.
//!
//! σ(N, k) and N! constants below are from
//! `doubly_even.spec.mass.gaborit_sigma` (Python source); see the
//! `enumerate_n10_max_k3_count` test in `enumerate.rs` for the same
//! pattern at N = 10.

#![cfg(feature = "parallel")]

use doubly_even_kernel::enumerate::{
    enumerate_doubly_even, enumerate_doubly_even_parallel,
};

type Row = (Vec<u64>, u128);

/// Reduce the rich `EnumeratedRaw` rows to `(rref, aut_order)` and sort by
/// `rref` so set equality compares cleanly across runs.
fn canonical_rows(
    out: Vec<doubly_even_kernel::enumerate::EnumeratedRaw>,
) -> Vec<Row> {
    let mut rows: Vec<Row> = out
        .into_iter()
        .map(|e| (e.rref, e.aut_order))
        .collect();
    rows.sort();
    rows
}

/// σ(12, k) for k = 0..6, from `doubly_even.spec.mass.gaborit_sigma`.
const SIGMA_N12: [u128; 7] = [1, 991, 79035, 625515, 479655, 25245, 0];
const FACT_N12: u128 = 479_001_600;

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

fn check_parallel_matches_sequential(
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
) {
    let (out_seq, _stats_seq, _pk_seq) =
        enumerate_doubly_even(n, max_k, quota.clone(), factorial_n);
    let seq_rows = canonical_rows(out_seq);

    for &nt in &[2usize, 4, 8] {
        let (out_par, _stats_par, _pk_par) = enumerate_doubly_even_parallel(
            n,
            max_k,
            quota.clone(),
            factorial_n,
            nt,
        );
        let par_rows = canonical_rows(out_par);
        assert_eq!(
            seq_rows.len(),
            par_rows.len(),
            "N={n} max_k={max_k} threads={nt}: class count diverged \
             (seq={} par={})",
            seq_rows.len(),
            par_rows.len(),
        );
        assert_eq!(
            seq_rows, par_rows,
            "N={n} max_k={max_k} threads={nt}: canonical-row set diverged"
        );
    }
}

#[test]
fn parallel_matches_sequential_n12() {
    check_parallel_matches_sequential(
        12,
        6,
        SIGMA_N12.to_vec(),
        FACT_N12,
    );
}

#[test]
fn parallel_matches_sequential_n14() {
    check_parallel_matches_sequential(
        14,
        7,
        SIGMA_N14.to_vec(),
        FACT_N14,
    );
}

/// `num_threads <= 1` must transparently dispatch to the sequential driver
/// and produce identical output. Timing-field indices in the stats vector
/// (`*_ns`) are excluded from the byte-identity comparison because they
/// legitimately vary across runs.
#[test]
fn parallel_with_one_thread_falls_back_to_sequential() {
    let (out_seq, stats_seq, pk_seq) =
        enumerate_doubly_even(14, 7, SIGMA_N14.to_vec(), FACT_N14);
    let (out_par, stats_par, pk_par) = enumerate_doubly_even_parallel(
        14,
        7,
        SIGMA_N14.to_vec(),
        FACT_N14,
        1,
    );
    assert_eq!(canonical_rows(out_seq), canonical_rows(out_par));
    // Indices 9, 10, 11, 18, 20 in the stats vector are *_ns timers and
    // are inherently noisy; compare the deterministic counters only.
    let timing_idx: &[usize] = &[9, 10, 11, 18, 20];
    for (i, (a, b)) in stats_seq.iter().zip(stats_par.iter()).enumerate() {
        if timing_idx.contains(&i) {
            continue;
        }
        assert_eq!(a, b, "stats[{i}] diverged between sequential and parallel(nt=1)");
    }
    assert_eq!(pk_seq, pk_par);
}

/// `max_k <= frontier_depth` (currently 3) must also fall back to the
/// sequential driver — workers would have nothing to do otherwise. We
/// don't actually rely on the byte-identity assertion here because the
/// fallback path is the very same `enumerate_doubly_even` call.
#[test]
fn parallel_with_shallow_max_k_falls_back() {
    let (out_seq, _, _) =
        enumerate_doubly_even(14, 3, SIGMA_N14[..4].to_vec(), FACT_N14);
    let (out_par, _, _) = enumerate_doubly_even_parallel(
        14,
        3,
        SIGMA_N14[..4].to_vec(),
        FACT_N14,
        8,
    );
    assert_eq!(canonical_rows(out_seq), canonical_rows(out_par));
}
