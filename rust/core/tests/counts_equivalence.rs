//! Counts-only mode correctness harness (the N ≥ 30 driver).
//!
//! Asserts that `enumerate_doubly_even_counts` (sequential) and
//! `enumerate_doubly_even_parallel_counts` produce per-rank class
//! counts, mass and |Aut| histograms identical to aggregates folded
//! from the in-memory reference `enumerate_doubly_even` output, at
//! N = 12, 14, 16. Also exercises the progress sink (file appears,
//! final snapshot has `done: true`, mass == quota rows).

#![cfg(feature = "parallel")]

use std::collections::HashMap;
use std::path::PathBuf;

use doubly_even_core::enumerate::{
    enumerate_doubly_even, enumerate_doubly_even_counts,
    enumerate_doubly_even_parallel_counts, ProgressSink,
};
use doubly_even_core::u256::U256;

const SIGMA_N12: [u128; 7] = [1, 991, 79035, 625515, 479655, 25245, 0];
const FACT_N12: u128 = 479_001_600;
const SIGMA_N14: [u128; 8] = [
    1, 4_095, 1_396_395, 50_868_675, 213_648_435, 103_378_275, 4_922_775, 0,
];
const FACT_N14: u128 = 87_178_291_200;
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

/// Reference aggregates from the in-memory driver.
fn reference_fold(
    n: u32,
    max_k: u32,
    quota: &[u128],
    fact: u128,
) -> (Vec<u64>, Vec<Vec<(u128, u64)>>) {
    let (raw, _, _) = enumerate_doubly_even(n, max_k, quota.to_vec(), fact);
    let len = (max_k + 1) as usize;
    let mut classes = vec![0u64; len];
    let mut hist: Vec<HashMap<u128, u64>> = vec![HashMap::new(); len];
    for e in &raw {
        let k = e.rref.len();
        classes[k] += 1;
        *hist[k].entry(e.aut_order).or_insert(0) += 1;
    }
    let hist_sorted: Vec<Vec<(u128, u64)>> = hist
        .into_iter()
        .map(|h| {
            let mut v: Vec<(u128, u64)> = h.into_iter().collect();
            v.sort_unstable_by_key(|&(aut, _)| aut);
            v
        })
        .collect();
    (classes, hist_sorted)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("counts_eq_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn check_one(n: u32, quota: &[u128], fact: u128) {
    let max_k = (quota.len() - 1) as u32;
    let (ref_classes, ref_hist) = reference_fold(n, max_k, quota, fact);
    let quota_wide: Vec<U256> = quota.iter().map(|&q| U256::from(q)).collect();

    // Sequential counts.
    let seq = enumerate_doubly_even_counts(n, max_k, quota_wide.clone(), fact, None);
    assert_eq!(seq.classes, ref_classes, "N={n}: seq counts classes diverged");
    assert_eq!(seq.aut_hist, ref_hist, "N={n}: seq counts |Aut| hist diverged");
    for k in 0..=(max_k as usize) {
        assert_eq!(seq.mass[k], U256::from(quota[k]), "N={n}: seq mass[k={k}]");
    }

    // Parallel counts at several thread counts, with the progress sink.
    for &nt in &[2usize, 4, 8] {
        let dir = scratch(&format!("n{n}_t{nt}"));
        let progress_path = dir.join("progress.json");
        let par = enumerate_doubly_even_parallel_counts(
            n,
            max_k,
            quota_wide.clone(),
            fact,
            nt,
            Some(ProgressSink {
                path: progress_path.clone(),
                interval_s: 1,
            }),
        );
        assert_eq!(
            par.classes, ref_classes,
            "N={n} threads={nt}: parallel counts classes diverged"
        );
        assert_eq!(
            par.aut_hist, ref_hist,
            "N={n} threads={nt}: parallel counts |Aut| hist diverged"
        );
        let body = std::fs::read_to_string(&progress_path)
            .expect("final progress.json must exist");
        assert!(body.contains("\"done\":true"), "final snapshot not marked done");
        assert!(body.contains("\"mass_quota\""), "snapshot missing mass_quota rows");
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn counts_mode_matches_in_memory_reference() {
    check_one(12, &SIGMA_N12, FACT_N12);
    check_one(14, &SIGMA_N14, FACT_N14);
    check_one(16, &SIGMA_N16, FACT_N16);
}
