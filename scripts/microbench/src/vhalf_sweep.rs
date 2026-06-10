//! v-half sweep microbench — THE SIMD target loop, isolated
//! (pre-SIMD checklist item 3, 2026-06-10).
//!
//! Replays φ phase 0 exactly as production runs it
//! (`parent_rule.rs::phi_cascade_split` lines "Phase 0"): per candidate
//! `v`, over the parent's shared C-half codeword table of `h = 2^k`
//! u64 words,
//!
//!   wt_v[x'] = popcount(cwords[x'] ^ v)            (XOR + popcount)
//!   counts4[x' & 3][wt_v[x']] += 1                 (4-way split hist)
//!
//! This loop is 67.3 % of residual φ at N=26 (`d17-pt2-postchain`
//! sampled split) and the aimed SIMD shapes are AVX2 Harley-Seal /
//! AVX-512 VPOPCNTDQ / NEON cnt+addv. Reported per k+1:
//!
//!   pop_ns/elem    popcount-only loop (no histogram)
//!   fused_ns/elem  production shape (weights store + histogram)
//!   GB/s           effective traffic in fused mode (9 B/elem: 8 read
//!                  + 1 written) — compare against ~DRAM/L2 streams to
//!                  see how far from the bandwidth roof the scalar
//!                  loop sits (= the SIMD headroom)
//!
//! hot = table L1/L2-resident; cold = 4 MB eviction between candidates.
//!
//! Run (from /workspace/src):
//!   cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     --bin vhalf_sweep -- --min-kp1 8 --max-kp1 16
//! Pin it: `taskset -c 4 ...`.

use microbench::timing::{cycles_to_ns, mono_cycles, ns_per_cycle};
use microbench::{evict_l1_l2, XorShift64};
use std::env;
use std::hint::black_box;

/// Production phase-0 clone: weights + 4-way split histogram.
#[inline(never)]
fn vhalf_fused(cwords: &[u64], v: u64, wt_v: &mut Vec<u8>) -> [u32; 65] {
    let h = cwords.len();
    wt_v.clear();
    wt_v.resize(h, 0);
    for (wt, &cw) in wt_v.iter_mut().zip(cwords.iter()) {
        *wt = (cw ^ v).count_ones() as u8;
    }
    let mut counts4 = [[0u32; 65]; 4];
    let mut chunks = wt_v.chunks_exact(4);
    for c in &mut chunks {
        counts4[0][c[0] as usize] += 1;
        counts4[1][c[1] as usize] += 1;
        counts4[2][c[2] as usize] += 1;
        counts4[3][c[3] as usize] += 1;
    }
    for &wt in chunks.remainder() {
        counts4[0][wt as usize] += 1;
    }
    let mut counts_v = [0u32; 65];
    for w in 0..65 {
        counts_v[w] = counts4[0][w] + counts4[1][w] + counts4[2][w] + counts4[3][w];
    }
    counts_v
}

/// Popcount-only variant (no weights store, no histogram): the floor a
/// pure popcount SIMD kernel could reach if the histogram were free.
#[inline(never)]
fn vhalf_pop_only(cwords: &[u64], v: u64) -> u32 {
    let mut acc = 0u32;
    for &cw in cwords {
        acc = acc.wrapping_add((cw ^ v).count_ones());
    }
    acc
}

fn arg(name: &str, default: u64) -> u64 {
    let args: Vec<String> = env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let min_kp1 = arg("--min-kp1", 8) as usize;
    let max_kp1 = arg("--max-kp1", 16) as usize;
    let n_bits = arg("--n", 48);

    println!("# vhalf_sweep: phi phase-0 XOR+popcount+histogram (the SIMD target)");
    println!("# ns_per_cycle = {:.4}, N = {n_bits}", ns_per_cycle());
    println!(
        "{:>4} {:>9} {:>12} {:>12} {:>14} {:>14} {:>8}",
        "k+1", "tbl_KB", "pop_ns/el", "fused_ns/el", "fused_cold/el", "hot_GB/s", "ratio"
    );

    let mut rng = XorShift64::new(0x5EED_F00D);
    let mask = (1u64 << n_bits) - 1;
    let mut junk = vec![0u8; 4 << 20];
    let mut wt_v: Vec<u8> = Vec::new();

    for kp1 in min_kp1..=max_kp1 {
        let h = 1usize << (kp1 - 1);
        let cwords: Vec<u64> = (0..h).map(|_| rng.next() & mask).collect();
        let cands: Vec<u64> = (0..64).map(|_| rng.next() & mask).collect();
        let iters = ((1u64 << 26) / (h as u64 * 64)).max(1);

        // popcount-only, hot
        let c0 = mono_cycles();
        for _ in 0..iters {
            for &v in &cands {
                black_box(vhalf_pop_only(&cwords, black_box(v)));
            }
        }
        let pop_cyc = mono_cycles().wrapping_sub(c0) / (iters * 64);

        // fused, hot
        let c0 = mono_cycles();
        for _ in 0..iters {
            for &v in &cands {
                black_box(vhalf_fused(&cwords, black_box(v), &mut wt_v)[4]);
            }
        }
        let fused_cyc = mono_cycles().wrapping_sub(c0) / (iters * 64);

        // fused, cold
        let cold_iters = iters.min(4);
        let c0 = mono_cycles();
        for _ in 0..cold_iters {
            for &v in &cands {
                evict_l1_l2(&mut junk);
                black_box(vhalf_fused(&cwords, black_box(v), &mut wt_v)[4]);
            }
        }
        let mut cold_cyc = mono_cycles().wrapping_sub(c0) / (cold_iters * 64);
        let c0 = mono_cycles();
        for _ in 0..32 {
            evict_l1_l2(&mut junk);
        }
        let evict_cyc = mono_cycles().wrapping_sub(c0) / 32;
        cold_cyc = cold_cyc.saturating_sub(evict_cyc);

        let pop_ns = cycles_to_ns(pop_cyc) / h as f64;
        let fused_ns = cycles_to_ns(fused_cyc) / h as f64;
        let cold_ns = cycles_to_ns(cold_cyc) / h as f64;
        // 8 B read (cwords) + 1 B written (wt_v) per element.
        let gbs = 9.0 / fused_ns;
        println!(
            "{:>4} {:>9.1} {:>12.3} {:>12.3} {:>14.3} {:>14.2} {:>8.2}",
            kp1,
            (h * 8) as f64 / 1024.0,
            pop_ns,
            fused_ns,
            cold_ns,
            gbs,
            fused_ns / pop_ns.max(1e-9),
        );
    }
}
