//! Cache-cliff sweep of the low-rank σ_Q pipeline (post-D15 profile,
//! plan `last-session-we-had-sequential-fiddle.md` Phase 3).
//!
//! The per-k timing rows show σ_Q candidate generation concentrates at
//! LOW parent ranks, where `L = N − 2k` is large and one call walks all
//! `2^L` quotient coordinates (`singular_reps_q` Gray sweep) and then
//! BFS-decomposes the survivors under ⟨σ_Q⟩ against a `2^L`-bit
//! FixedBitSet (`aut_orbit_minima_q_witt`). Both clones below are
//! verbatim copies of `orbit.rs` (kept in copy-sync by hand).
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

use microbench::timing::{cycles_to_ns, mono_cycles, ns_per_cycle};
use microbench::XorShift64;
use std::env;
use std::hint::black_box;

// ----- verbatim orbit.rs clones (FixedBitSet replaced by a Vec<u64>
// ----- bitset with identical probe/insert behaviour: one word load /
// ----- store per bit, no dependency needed).

struct BitSet(Vec<u64>);

impl BitSet {
    fn with_capacity(bits: usize) -> Self {
        BitSet(vec![0u64; bits.div_ceil(64)])
    }
    #[inline]
    fn contains(&self, i: usize) -> bool {
        (self.0[i >> 6] >> (i & 63)) & 1 == 1
    }
    #[inline]
    fn insert(&mut self, i: usize) {
        self.0[i >> 6] |= 1 << (i & 63);
    }
}

#[inline]
fn mat_apply(m: &[u64], v: u64) -> u64 {
    let mut out: u64 = 0;
    let mut u = v;
    while u != 0 {
        let i = u.trailing_zeros() as usize;
        out ^= m[i];
        u &= u - 1;
    }
    out
}

fn singular_reps_q(v_basis: &[u64]) -> Vec<u64> {
    let l = v_basis.len();
    let mut out: Vec<u64> = Vec::new();
    if l == 0 {
        return out;
    }
    let size: u64 = 1u64 << l;
    out.reserve(size as usize / 2);
    let mut u: u64 = 0;
    let mut v: u64 = 0;
    for i in 1..size {
        let flip = i.trailing_zeros() as usize;
        u ^= 1u64 << flip;
        v ^= v_basis[flip];
        if v.count_ones() & 3 == 0 {
            out.push(u);
        }
    }
    out
}

fn aut_orbit_minima_q_witt(reps_q: &[u64], gens: &[Vec<u64>], l: u32) -> Vec<u64> {
    let mut reps_sorted = reps_q.to_vec();
    reps_sorted.sort_unstable();
    let universe = 1usize << l;
    let mut seen = BitSet::with_capacity(universe);
    let mut minima: Vec<u64> = Vec::new();
    let cap = reps_q.len();
    let mut queue: Vec<u64> = Vec::with_capacity(cap);
    let mut next: Vec<u64> = Vec::with_capacity(cap);
    for &v in &reps_sorted {
        if seen.contains(v as usize) {
            continue;
        }
        minima.push(v);
        seen.insert(v as usize);
        queue.clear();
        queue.push(v);
        while !queue.is_empty() {
            next.clear();
            for &current in &queue {
                for g in gens {
                    let new_v = mat_apply(g, current);
                    if !seen.contains(new_v as usize) {
                        seen.insert(new_v as usize);
                        next.push(new_v);
                    }
                }
            }
            std::mem::swap(&mut queue, &mut next);
        }
    }
    minima
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
            let minima = aut_orbit_minima_q_witt(&reps, &gens, l);
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
