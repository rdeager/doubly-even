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
//! Coverage: `N = 12, 14, 16`. `N = 12, max_k = 6` has 7 ranks and 21
//! depth-3 frontier nodes, enough to exercise the worker pool.
//! `N = 14, max_k = 7` is larger and stresses load balance.
//! `N = 16, max_k = 8` is the smallest case where the pipelined seeder
//! sees a frontier large enough that worker-overlap is visibly non-trivial
//! — added when the pipelined-seeder splice landed.
//!
//! σ(N, k) and N! constants below are from
//! `doubly_even.spec.mass.gaborit_sigma` (Python source); see the
//! `enumerate_n10_max_k3_count` test in `enumerate.rs` for the same
//! pattern at N = 10.

#![cfg(feature = "parallel")]

use doubly_even_kernel::enumerate::{
    enumerate_doubly_even, enumerate_doubly_even_parallel,
    enumerate_doubly_even_parallel_with_seeder,
};
use doubly_even_kernel::parent_rule::ParentRule;

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

/// σ(16, k) for k = 0..8.
const SIGMA_N16: [u128; 9] = [
    1,
    16_511,
    22_891_115,
    3_451_225_635,
    62_449_776_675,
    143_919_296_235,
    44_388_662_175,
    1_885_422_825,
    9_845_550,
];
const FACT_N16: u128 = 20_922_789_888_000;

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

#[test]
fn parallel_matches_sequential_n16() {
    check_parallel_matches_sequential(
        16,
        8,
        SIGMA_N16.to_vec(),
        FACT_N16,
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
    // Indices 9, 10, 11, 18, 20, 30, 33 in the stats vector are *_ns
    // timers and are inherently noisy; compare the deterministic
    // counters only. (30 = phi_ns, 33 = nauty_ns_kept — both 0 unless
    // DOUBLY_EVEN_PARENT_RULE=audit, but excluded for future-proofing.)
    // 34–44 are the phase_timers sub-phase fields: 34–43 are timers and
    // 44 (phi_sampled_calls) depends on a process-lifetime thread-local
    // call counter, so none are run-deterministic when the feature is on.
    // 45 (phi_ctx_ns, D16) is a timer; 46–48 (ctx_builds, s1_fastpath,
    // killer_rejects) are deterministic counters and stay compared.
    let timing_idx: &[usize] = &[
        9, 10, 11, 18, 20, 30, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    ];
    for (i, (a, b)) in stats_seq.iter().zip(stats_par.iter()).enumerate() {
        if timing_idx.contains(&i) {
            continue;
        }
        assert_eq!(a, b, "stats[{i}] diverged between sequential and parallel(nt=1)");
    }
    // per_k rows 14–16 (phi_ns, candidates_q_ns, nauty_ns) are per-rank
    // timers — same noise argument as the flat *_ns fields above. Compare
    // the count rows exactly, the timing rows structurally.
    const PER_K_TIMING_ROWS_START: usize = 14;
    assert_eq!(pk_seq.len(), pk_par.len(), "per_k row count diverged");
    assert_eq!(
        pk_seq[..PER_K_TIMING_ROWS_START],
        pk_par[..PER_K_TIMING_ROWS_START],
        "per_k count rows diverged between sequential and parallel(nt=1)"
    );
    for (i, (a, b)) in pk_seq[PER_K_TIMING_ROWS_START..]
        .iter()
        .zip(pk_par[PER_K_TIMING_ROWS_START..].iter())
        .enumerate()
    {
        assert_eq!(
            a.len(),
            b.len(),
            "per_k timing row {} length diverged",
            PER_K_TIMING_ROWS_START + i
        );
    }
}

/// D16 lever B: the pooled-seeder σ_Q path must produce the identical
/// output set. The default `min_l = 16` never fires at N ≤ 16
/// (`L = N − 2k ≤ 12` for k ≥ 2), so this test forces `min_l = 2` via
/// the env-free `_with_seeder` entry — the pooled Gray walk and pooled
/// orbit-min BFS then execute at every reachable L.
#[test]
fn parallel_matches_sequential_pooled_seeder() {
    for (n, max_k, sigma, fact) in [
        (14u32, 7u32, SIGMA_N14.to_vec(), FACT_N14),
        (16, 8, SIGMA_N16.to_vec(), FACT_N16),
    ] {
        let (out_seq, _, _) =
            enumerate_doubly_even(n, max_k, sigma.clone(), fact);
        let want = canonical_rows(out_seq);
        for &nt in &[2usize, 4, 8] {
            let (out_par, _, _) = enumerate_doubly_even_parallel_with_seeder(
                n,
                max_k,
                sigma.clone(),
                fact,
                nt,
                ParentRule::from_env(),
                4, // seeder_threads
                2, // seeder_min_l — force the pooled path at small L
            );
            assert_eq!(
                want,
                canonical_rows(out_par),
                "pooled-seeder output diverged at N={n}, threads={nt}"
            );
        }
    }
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
