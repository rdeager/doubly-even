//! Cache-cliff sweep of the low-rank σ_Q pipeline (post-D15 profile,
//! plan `last-session-we-had-sequential-fiddle.md` Phase 3).
//!
//! The per-k timing rows show σ_Q candidate generation concentrates at
//! LOW parent ranks, where `L = N − 2k` is large and one call walks all
//! `2^L` quotient coordinates (`singular_reps_q` Gray sweep) and then
//! BFS-decomposes the survivors under ⟨σ_Q⟩ against a `2^L`-bit
//! FixedBitSet. Both arms call `doubly-even-core` directly (the
//! hand-copied clones are retired): the Gray sweep is
//! `orbit::singular_reps_q`, the BFS is the production legacy walk
//! body `orbit::orbit_minima_walk` — pinned to the WALK on purpose so
//! the sweep keeps measuring the chained-walk shape across the cache
//! cliff (since D18 the production entry `aut_orbit_minima_q_witt`
//! dispatches to the m4r body at L ≥ 14).
//!
//! Working sets vs L: bitset 2^L bits (L=18 → 32 KB, L=21 → 256 KB,
//! L=24 → 2 MB, L=27 → 16 MB) + reps vec ≈ 8 B · 2^L / 4. At N=32, k=2
//! ⇒ L=28: a 32 MB bitset probed in BFS order — this bin measures how
//! per-step cost degrades L1 → L2 → L3 → DRAM, which is the dominant
//! input to the N=29/32 low-k extrapolation.
//!
//! Reported per L:
//!   gray_ns/step  — singular_reps_q cost / 2^L  (streaming + push)
//!   bfs_ns/rep    — witt orbit-min cost / |reps| (random bitset probes)
//!
//! Build & run (portable x86_64 / aarch64):
//!   cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     --bin singular_walk -- --min-l 10 --max-l 26
//! Pin it: `taskset -c 4 ...`. L=26 needs ~700 MB peak (reps + queues);
//! default max is 24 for quick iteration.

use doubly_even_core::orbit::{orbit_minima_walk, singular_reps_q};
use microbench::timing::{cycles_to_ns, mono_cycles, ns_per_cycle};
use microbench::XorShift64;
use std::env;
use std::hint::black_box;

/// Sort + production walk body — the same sort the production entry
/// performs before dispatching to a BFS body. The random-GL generators
/// here are never the identity, so the entry's identity filter is a
/// no-op for these inputs.
fn orbit_min_walk(reps_q: &[u64], gens: &[Vec<u64>], l: u32) -> Vec<u64> {
    let mut reps_sorted = reps_q.to_vec();
    reps_sorted.sort_unstable();
    let gens_ref: Vec<&Vec<u64>> = gens.iter().collect();
    orbit_minima_walk(&reps_sorted, &gens_ref, l)
}

// ----- synthetic inputs

/// Random invertible L×L column matrix over F_2: start from identity,
/// apply L² random elementary row-adds + column swaps. Models a σ_Q
/// image of an Aut(C) generator (invertible, unstructured).
fn random_gl(l: usize, rng: &mut XorShift64) -> Vec<u64> {
    let mut m: Vec<u64> = (0..l).map(|i| 1u64 << i).collect();
    for _ in 0..l * l {
        let a = (rng.next() as usize) % l;
        let b = (rng.next() as usize) % l;
        if a != b {
            if rng.next() & 1 == 0 {
                m[a] ^= m[b];
            } else {
                m.swap(a, b);
            }
        }
    }
    m
}

fn arg(name: &str, default: u32) -> u32 {
    let args: Vec<String> = env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let min_l = arg("--min-l", 10);
    let max_l = arg("--max-l", 24);
    let n_gens = arg("--gens", 4) as usize;
    let n_bits = 48u64;

    println!("# singular_walk: singular_reps_q Gray sweep + witt orbit-min BFS");
    println!("# ns_per_cycle = {:.4}, gens = {n_gens}", ns_per_cycle());
    println!(
        "{:>3} {:>10} {:>10} {:>9} {:>12} {:>12} {:>10} {:>10}",
        "L", "bitset_B", "reps", "minima", "gray_ns/step", "bfs_ns/rep", "gray_ms", "bfs_ms"
    );

    let mut rng = XorShift64::new(0x51_4d15);

    for l in min_l..=max_l {
        let mask = (1u64 << n_bits) - 1;
        let v_basis: Vec<u64> = (0..l).map(|_| rng.next() & mask).collect();
        let gens: Vec<Vec<u64>> = (0..n_gens)
            .map(|_| random_gl(l as usize, &mut rng))
            .collect();

        // Repeat small-L runs so every row has ≥ ~10 ms of measured work.
        let reps_iters = ((1u64 << 24) >> l).max(1);

        let mut reps: Vec<u64> = Vec::new();
        let c0 = mono_cycles();
        for _ in 0..reps_iters {
            reps = singular_reps_q(&v_basis);
            black_box(reps.len());
        }
        let gray_cyc = mono_cycles().wrapping_sub(c0) / reps_iters;

        let mut minima_len = 0usize;
        let c0 = mono_cycles();
        for _ in 0..reps_iters {
            let minima = orbit_min_walk(&reps, &gens, l);
            minima_len = minima.len();
            black_box(minima_len);
        }
        let bfs_cyc = mono_cycles().wrapping_sub(c0) / reps_iters;

        let steps = (1u64 << l) as f64;
        println!(
            "{:>3} {:>10} {:>10} {:>9} {:>12.3} {:>12.2} {:>10.2} {:>10.2}",
            l,
            (1u64 << l) / 8,
            reps.len(),
            minima_len,
            cycles_to_ns(gray_cyc) / steps,
            cycles_to_ns(bfs_cyc) / reps.len().max(1) as f64,
            cycles_to_ns(gray_cyc) / 1e6,
            cycles_to_ns(bfs_cyc) / 1e6,
        );
    }
}
