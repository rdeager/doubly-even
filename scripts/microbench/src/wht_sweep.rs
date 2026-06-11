//! Cache-cliff sweep of the φ-cascade WHT (post-D15 profile, plan
//! `last-session-we-had-sequential-fiddle.md` Phase 3).
//!
//! Measures the fused `fill_indicator` + `wht_in_place` operation —
//! exactly what `parent_rule.rs` pays once per evaluated stratum — over
//! buffer sizes 2^8..2^17 (1 KB → 512 KB of i32). The question: where
//! does ns-per-butterfly jump as the `f` buffer outgrows L1d
//! (48 KB on 13700K P-cores ⇒ between 2^13 and 2^14; 64 KB on Axion
//! Neoverse V2 ⇒ between 2^14 and 2^15 — a moved cliff is the causal
//! confirmation), and how big is the multiplier?
//!
//! Modes per size:
//!   hot  — back-to-back calls, buffer stays cache-resident (matches
//!          production at small k where the scratch survives between
//!          candidates).
//!   cold — a 4 MB eviction sweep between calls (upper-bounds the
//!          production interleaving with ~210 KB CanonScratch canon
//!          calls).
//!
//! Butterfly count per call = size · log2(size); ns/bfly is
//! size-comparable. ns/call is what the extrapolation model consumes.
//!
//! Build & run (portable x86_64 / aarch64):
//!   cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     --bin wht_sweep -- --min-log2 8 --max-log2 17
//! Pin it: `taskset -c 4 cargo run ...` (x86 dev box).

use doubly_even_core::parent_rule::wht_in_place;
use microbench::timing::{cycles_to_ns, mono_cycles, ns_per_cycle};
use microbench::{evict_l1_l2, XorShift64};
use std::env;
use std::hint::black_box;

// `wht_in_place` is the production transform, linked directly from
// `doubly-even-core` (the hand-copied clone is retired).

/// `fill_indicator` shape: clear + scatter |T| ones + transform.
fn fill_and_transform(f: &mut Vec<i32>, size: usize, members: &[u16]) {
    f.clear();
    f.resize(size, 0);
    for &x in members {
        f[x as usize] = 1;
    }
    wht_in_place(f);
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
    let min_log2 = arg("--min-log2", 8);
    let max_log2 = arg("--max-log2", 17);
    // Per-size iteration budget: ~2^27 butterflies per mode keeps every
    // row past 10 ms of measured work.
    let bfly_budget: u64 = 1 << 27;

    println!("# wht_sweep: fused fill_indicator + wht_in_place");
    println!("# ns_per_cycle = {:.4}", ns_per_cycle());
    println!(
        "{:>5} {:>9} {:>7} {:>12} {:>12} {:>12} {:>12} {:>7}",
        "log2", "f_bytes", "iters", "hot_ns/call", "cold_ns/call", "hot_ns/bfly", "cold_ns/bfly", "ratio"
    );

    let mut rng = XorShift64::new(0xd15);
    let mut junk = vec![0u8; 4 << 20];

    for log2 in min_log2..=max_log2 {
        let size = 1usize << log2;
        let bfly_per_call = (size as u64) * log2 as u64;
        let iters = (bfly_budget / bfly_per_call).clamp(20, 200_000);

        // Stratum members: |T| = size/4 distinct coordinates (production
        // first strata carry a constant fraction of the 2^(k+1) space).
        let n_members = (size / 4).max(1);
        let members: Vec<u16> = (0..n_members)
            .map(|_| (rng.next() as usize % size) as u16)
            .collect();

        let mut f: Vec<i32> = Vec::with_capacity(size);

        // Warmup + hot.
        for _ in 0..5 {
            fill_and_transform(&mut f, size, &members);
        }
        let c0 = mono_cycles();
        for _ in 0..iters {
            fill_and_transform(&mut f, size, &members);
            black_box(f[size / 2]);
        }
        let hot_cyc = mono_cycles().wrapping_sub(c0);

        // Cold: evict L1/L2 between calls; subtract the eviction cost
        // measured separately so only the WHT-side penalty is reported.
        let c0 = mono_cycles();
        for _ in 0..iters {
            evict_l1_l2(&mut junk);
            black_box(junk[0]);
        }
        let evict_cyc = mono_cycles().wrapping_sub(c0);

        let c0 = mono_cycles();
        for _ in 0..iters {
            evict_l1_l2(&mut junk);
            fill_and_transform(&mut f, size, &members);
            black_box(f[size / 2]);
        }
        let cold_cyc = mono_cycles()
            .wrapping_sub(c0)
            .saturating_sub(evict_cyc);

        let hot_call = cycles_to_ns(hot_cyc) / iters as f64;
        let cold_call = cycles_to_ns(cold_cyc) / iters as f64;
        println!(
            "{:>5} {:>9} {:>7} {:>12.1} {:>12.1} {:>12.4} {:>12.4} {:>7.2}",
            log2,
            size * 4,
            iters,
            hot_call,
            cold_call,
            hot_call / bfly_per_call as f64,
            cold_call / bfly_per_call as f64,
            cold_call / hot_call,
        );
    }
}
