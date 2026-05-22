//! Streaming-output correctness harness.
//!
//! Runs `enumerate_doubly_even_streaming` (sequential) and
//! `enumerate_doubly_even_parallel_streaming` at N=12, 14, 16 and asserts
//! the parsed binary stream's `(rref, aut_order)` set equals the
//! reference set emitted by `enumerate_doubly_even`.
//!
//! Format under test: `crate::streaming::BinaryWriter`'s wire layout —
//!
//! ```text
//! header := b"DEKS" version:u32 n:u32 worker_id:u32       (16 bytes LE)
//! record := k:u8 aut_order:u128 basis:[u64; k]            (1+16+8k LE)
//! ```

#![cfg(feature = "parallel")]

use std::fs;
use std::path::{Path, PathBuf};

use doubly_even_kernel::enumerate::{
    enumerate_doubly_even, enumerate_doubly_even_parallel_streaming,
    enumerate_doubly_even_streaming,
};
use doubly_even_kernel::streaming::{HEADER_LEN, MAGIC, VERSION};

type Row = (Vec<u64>, u128);

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

/// Parse one `out.w*.bin` file. Returns `(worker_id, [(rref, aut_order)])`.
fn parse_stream_file(path: &Path) -> (u32, Vec<Row>) {
    let buf = fs::read(path).expect("read stream file");
    assert!(buf.len() >= HEADER_LEN, "file shorter than header: {path:?}");
    assert_eq!(&buf[0..4], &MAGIC, "magic mismatch in {path:?}");
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    assert_eq!(version, VERSION, "version mismatch in {path:?}");
    let _n = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let wid = u32::from_le_bytes(buf[12..16].try_into().unwrap());

    let mut out = Vec::new();
    let mut i = HEADER_LEN;
    while i < buf.len() {
        let k = buf[i] as usize;
        i += 1;
        let aut = u128::from_le_bytes(buf[i..i + 16].try_into().unwrap());
        i += 16;
        let mut basis = Vec::with_capacity(k);
        for _ in 0..k {
            basis.push(u64::from_le_bytes(buf[i..i + 8].try_into().unwrap()));
            i += 8;
        }
        out.push((basis, aut));
    }
    assert_eq!(i, buf.len(), "trailing bytes in {path:?}");
    (wid, out)
}

fn parse_all_streams(dir: &Path) -> Vec<Row> {
    let mut rows = Vec::new();
    let mut seen_ids = Vec::new();
    for entry in fs::read_dir(dir).expect("read_dir") {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !name.starts_with("out.w") || !name.ends_with(".bin") {
            continue;
        }
        let (wid, mut chunk) = parse_stream_file(&path);
        seen_ids.push(wid);
        rows.append(&mut chunk);
    }
    rows.sort();
    rows
}

fn reference_rows(n: u32, max_k: u32, quota: &[u128], fact: u128) -> Vec<Row> {
    let (out, _, _) = enumerate_doubly_even(n, max_k, quota.to_vec(), fact);
    let mut rows: Vec<Row> = out.into_iter().map(|e| (e.rref, e.aut_order)).collect();
    rows.sort();
    rows
}

/// Run streaming sequential + parallel(2, 4, 8 threads) at the given
/// problem size; assert their parsed output equals the reference.
fn check_streaming_matches_sequential(n: u32, max_k: u32, quota: &[u128], fact: u128) {
    let reference = reference_rows(n, max_k, quota, fact);

    // Sequential streaming.
    let tmp_seq = scratch_dir(&format!("streaming_seq_n{n}_k{max_k}"));
    let res_seq = enumerate_doubly_even_streaming(n, max_k, quota.to_vec(), fact, &tmp_seq);
    let seq_rows = parse_all_streams(&tmp_seq);
    assert_eq!(
        seq_rows, reference,
        "N={n}: sequential streaming output diverged from reference"
    );
    // Mass-formula gate is also checked inside the kernel — this just
    // confirms the snapshot is surfaced to the caller for stats.json.
    assert_eq!(
        res_seq.mass.len(),
        quota.len(),
        "mass snapshot length mismatch"
    );
    for k in 0..=(max_k as usize) {
        assert_eq!(res_seq.mass[k], quota[k], "seq mass[k={k}] vs quota mismatch");
    }
    fs::remove_dir_all(&tmp_seq).ok();

    // Parallel streaming at several thread counts.
    for &nt in &[2usize, 4, 8] {
        let tmp_par = scratch_dir(&format!("streaming_par_n{n}_k{max_k}_t{nt}"));
        let res_par = enumerate_doubly_even_parallel_streaming(
            n, max_k, quota.to_vec(), fact, nt, &tmp_par,
        );
        let par_rows = parse_all_streams(&tmp_par);
        assert_eq!(
            par_rows.len(),
            reference.len(),
            "N={n} threads={nt}: streaming class count diverged"
        );
        assert_eq!(
            par_rows, reference,
            "N={n} threads={nt}: streaming output diverged from reference"
        );
        for k in 0..=(max_k as usize) {
            assert_eq!(
                res_par.mass[k], quota[k],
                "par mass[k={k}] threads={nt} vs quota mismatch"
            );
        }
        fs::remove_dir_all(&tmp_par).ok();
    }
}

fn scratch_dir(suffix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("doubly_even_{}_{}", std::process::id(), suffix));
    fs::create_dir_all(&p).expect("create scratch dir");
    p
}

#[test]
fn streaming_matches_sequential_n12() {
    check_streaming_matches_sequential(12, 6, &SIGMA_N12, FACT_N12);
}

#[test]
fn streaming_matches_sequential_n14() {
    check_streaming_matches_sequential(14, 7, &SIGMA_N14, FACT_N14);
}

#[test]
fn streaming_matches_sequential_n16() {
    check_streaming_matches_sequential(16, 8, &SIGMA_N16, FACT_N16);
}
