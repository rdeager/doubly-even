//! Criterion microbenches: σ_Q table vs witt bit-walk for orbit-min BFS.
//!
//! This is the rethink-while-porting validation step from
//! `we-now-have-a-luminous-sunbeam.md` (5(b)-4). In pure Python the
//! table-based BFS dominated witt bit-walk at every `L`
//! ([[phase-b-empirical-finding]]); in Rust the per-step cost of bit-walk
//! may flip the verdict. We measure here, not after-the-fact in the
//! enumeration profile.
//!
//! The benches are synthetic: random `Mat`s and full `[1, 2^L)` rep sets.
//! That's enough to compare per-step costs — the *real* workload is
//! `singular-set ⊂ [1, 2^L)` with `|gens|` ~ small, which has the same
//! BFS shape just smaller.
//!
//! Run with `cargo bench --manifest-path rust/Cargo.toml` (HTML reports
//! at `rust/target/criterion/report/index.html`).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use doubly_even_core::orbit::{aut_orbit_minima_q_table, aut_orbit_minima_q_witt};
use doubly_even_core::types::{BinVec, Mat};

/// Deterministic xorshift64 — keeps the benches reproducible across runs
/// without pulling in `rand`.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Build a random invertible matrix in `GL(L, F_2)` by composing random
/// elementary row ops on the identity. Good enough for the BFS bench —
/// the exact group structure doesn't matter, only the per-step cost.
fn random_gl_matrix(seed: u64, l: u32) -> Mat {
    let l = l as usize;
    let mut m: Mat = (0..l as u32).map(|j| 1u64 << j).collect();
    let mut state = seed;
    let mask = (1u64 << l) - 1;
    // 4L random elementary ops: m[i] ^= m[j] for i != j.
    for _ in 0..(4 * l) {
        let r = xorshift64(&mut state);
        let i = (r as usize) % l;
        let j = ((r >> 32) as usize) % l;
        if i != j {
            m[i] ^= m[j];
            m[i] &= mask;
        }
    }
    m
}

fn bench_orbit_min(c: &mut Criterion) {
    let mut group = c.benchmark_group("orbit_min");
    for &l in &[10u32, 14, 18, 22] {
        // Three random generators — typical column-permutation Aut size
        // produces a handful, not hundreds.
        let gens: Vec<Mat> = (0..3)
            .map(|i| random_gl_matrix(0xDEAD_BEEF + i, l))
            .collect();
        // Use the full [1, 2^L) as reps — worst-case singular set.
        let reps: Vec<BinVec> = (1u64..(1u64 << l)).collect();

        group.bench_with_input(BenchmarkId::new("table", l), &l, |b, &l| {
            b.iter(|| {
                let out = aut_orbit_minima_q_table(black_box(&reps), black_box(&gens), l);
                black_box(out);
            });
        });
        group.bench_with_input(BenchmarkId::new("witt", l), &l, |b, &l| {
            b.iter(|| {
                let out = aut_orbit_minima_q_witt(black_box(&reps), black_box(&gens), l);
                black_box(out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_orbit_min);
criterion_main!(benches);
