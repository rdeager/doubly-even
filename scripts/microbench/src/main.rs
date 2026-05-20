//! Micro-bench probes for the doubly-even kernel hot loops.
//!
//! Targets the inner work of `rust/src/canon.rs::build_sparsegraph` and
//! `rust/src/canon.rs::build_low_weight_sparsegraph`: Gray-code codeword
//! enumeration with per-codeword popcount AND per-column AND+popcount.
//!
//! Reports `ns/op` and `cycles/op` (via `rdtsc`) for three popcount
//! variants:
//!   A. scalar `u64::count_ones` (the current Rust kernel).
//!   B. inline `popcnt` intrinsic (essentially identical on x86 — sanity).
//!   C. AVX2 Harley-Seal popcount over 256-bit lanes.
//!
//! And for the per-column accumulation (Probe B in the plan): given the
//! Gray-walked codeword array, compute `degree[j] = popcount(AND over
//! codewords of (cw & bit_j))` for j in 0..n. We measure scalar vs AVX2
//! tree-popcount.
//!
//! Raptor Lake P-cores (13700K) have AVX-512 fused off → VPOPCNTDQ is
//! NOT available; the realistic SIMD ceiling for popcount is AVX2
//! Harley-Seal, which is what variant C measures.
//!
//! Usage:
//!   cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     -- --n 22 --k 11 --iters 1000

use std::arch::x86_64::*;
use std::env;
use std::hint::black_box;
use std::time::Instant;

#[inline(always)]
unsafe fn rdtsc() -> u64 {
    _rdtsc()
}

/// Build the codeword array via Gray-code walk. Used as fixed input for
/// the popcount probes so we measure the popcount path, not the walk.
fn build_codewords(rref: &[u64]) -> Vec<u64> {
    let k = rref.len();
    let l = 1usize << k;
    let mut cw = vec![0u64; l];
    for mask in 1..l {
        let lo_bit = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        cw[mask] = cw[mask ^ (1 << lo_bit)] ^ rref[lo_bit];
    }
    cw
}

/// Probe A1: scalar `count_ones`. Per-codeword weight, summed.
/// `black_box` on each load defeats the optimiser from realising the input
/// doesn't change between iterations and hoisting the work out.
#[inline(never)]
fn probe_a1_scalar_count_ones(cw: &[u64]) -> u64 {
    let mut sum: u64 = 0;
    for &w in cw {
        sum = sum.wrapping_add(black_box(w).count_ones() as u64);
    }
    sum
}

/// Probe A2: hardware popcnt via inline intrinsic (target-feature attr).
#[target_feature(enable = "popcnt")]
#[inline(never)]
unsafe fn probe_a2_popcnt_intrinsic(cw: &[u64]) -> u64 {
    let mut sum: u64 = 0;
    for &w in cw {
        sum = sum.wrapping_add(_popcnt64(black_box(w) as i64) as u64);
    }
    sum
}

/// Harley-Seal popcount of a 256-bit lane (4 × u64) via AVX2.
/// Reference: Mula, Kurz, Lemire 2018 "Faster Population Counts Using AVX2
/// Instructions"; implements the nibble-LUT version which beats scalar by
/// ~3-4x on Skylake-and-newer.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn popcnt256_lut(v: __m256i) -> __m256i {
    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    let lo_cnt = _mm256_shuffle_epi8(lookup, lo);
    let hi_cnt = _mm256_shuffle_epi8(lookup, hi);
    _mm256_add_epi8(lo_cnt, hi_cnt)
}

/// Probe A3: AVX2 Harley-Seal popcount over 256-bit lanes, summed.
/// Tail u64s handled with scalar fallback.
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn probe_a3_avx2_harley_seal(cw: &[u64]) -> u64 {
    let chunks = cw.len() / 4;
    let mut acc = _mm256_setzero_si256();
    let ptr = cw.as_ptr() as *const __m256i;
    for i in 0..chunks {
        let v = _mm256_loadu_si256(black_box(ptr.add(i)));
        let pc = popcnt256_lut(v);
        // Horizontally collapse u8 lane counts via psadbw against zero.
        let collapsed = _mm256_sad_epu8(pc, _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, collapsed);
    }
    // Extract 4 × u64 from acc and sum.
    let mut buf = [0u64; 4];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
    let mut sum: u64 = buf.iter().sum();
    // Tail.
    for &w in &cw[chunks * 4..] {
        sum = sum.wrapping_add(w.count_ones() as u64);
    }
    sum
}

/// Probe B1: per-column degree fill, scalar.
/// Mirrors `canon.rs::build_sparsegraph` lines 113-122: for j in 0..n,
/// degree[j] = #{ i : (cw[i] >> j) & 1 == 1 }.
#[inline(never)]
fn probe_b1_per_column_scalar(cw: &[u64], n: u32) -> Vec<u32> {
    let mut deg = vec![0u32; n as usize];
    for j in 0..n {
        let bit = 1u64 << j;
        let mut d: u32 = 0;
        for &w in cw {
            if w & bit != 0 {
                d += 1;
            }
        }
        deg[j as usize] = d;
    }
    deg
}

/// Probe B2: per-column degree by transpose + popcount.
/// Pack codewords' bit j into a bit-packed column vector, then popcount.
/// For |cw| up to 2^k, the packed column needs ⌈|cw|/64⌉ u64 words.
#[inline(never)]
fn probe_b2_per_column_transpose(cw: &[u64], n: u32) -> Vec<u32> {
    let words = (cw.len() + 63) / 64;
    let n = n as usize;
    let mut packed: Vec<u64> = vec![0u64; n * words];
    // Transpose: for each codeword i, for each j with bit set, set bit
    // (i % 64) of packed[j * words + i / 64].
    for (i, &w) in cw.iter().enumerate() {
        let i_word = i / 64;
        let i_bit = 1u64 << (i % 64);
        let mut bits = w;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            packed[j * words + i_word] |= i_bit;
            bits &= bits - 1;
        }
    }
    let mut deg = vec![0u32; n];
    for j in 0..n {
        let mut s: u32 = 0;
        for w in 0..words {
            s += packed[j * words + w].count_ones();
        }
        deg[j] = s;
    }
    deg
}

fn fmt_thousands(x: u64) -> String {
    let s = x.to_string();
    let b = s.as_bytes();
    let mut out = String::new();
    for (idx, &c) in b.iter().enumerate() {
        if idx > 0 && (b.len() - idx) % 3 == 0 {
            out.push('_');
        }
        out.push(c as char);
    }
    out
}

fn run<F: FnMut() -> u64>(label: &str, n_ops: u64, iters: usize, mut f: F) -> (f64, f64, u64) {
    // Warm-up.
    let mut acc: u64 = 0;
    for _ in 0..(iters.max(1) / 4).max(1) {
        acc = acc.wrapping_add(f());
    }
    // Measure.
    let t0 = Instant::now();
    let c0 = unsafe { rdtsc() };
    for _ in 0..iters {
        acc = acc.wrapping_add(black_box(f()));
    }
    let c1 = unsafe { rdtsc() };
    let elapsed = t0.elapsed();
    let total_ops = n_ops * iters as u64;
    let ns = elapsed.as_nanos() as f64;
    let cycles = (c1 - c0) as f64;
    let ns_per_op = ns / total_ops as f64;
    let cyc_per_op = cycles / total_ops as f64;
    println!(
        "{label:>32}: {iters} iter × {n_ops_str:>10} ops = {total:>14} ops in {wall:>8.3} ms; \
         {ns_per_op:>7.3} ns/op, {cyc_per_op:>6.2} cyc/op  (sink={acc})",
        label = label,
        iters = iters,
        n_ops_str = fmt_thousands(n_ops),
        total = fmt_thousands(total_ops),
        wall = elapsed.as_secs_f64() * 1000.0,
        ns_per_op = ns_per_op,
        cyc_per_op = cyc_per_op,
        acc = acc,
    );
    (ns_per_op, cyc_per_op, acc)
}

fn main() {
    // Defaults shadow the N=22, k=11 / N=18, k=9 regimes the kernel hits.
    let args: Vec<String> = env::args().collect();
    let mut n: u32 = 22;
    let mut k: u32 = 11;
    let mut iters: usize = 5000;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => { n = args[i + 1].parse().unwrap(); i += 2; }
            "--k" => { k = args[i + 1].parse().unwrap(); i += 2; }
            "--iters" => { iters = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(2); }
        }
    }
    println!("# popcount probe — N={n}, k={k}, iters={iters}");
    println!("# CPU: 13th Gen Intel Core i7-13700K (Raptor Lake, P-cores; AVX-512 fused off)");
    println!("# is_x86_feature_detected: popcnt={}, avx2={}, avx512vpopcntdq={}",
             is_x86_feature_detected!("popcnt"),
             is_x86_feature_detected!("avx2"),
             is_x86_feature_detected!("avx512vpopcntdq"));

    // Fixed input mimicking canon.rs::build_sparsegraph: an RREF of width n
    // and rank k. Use a simple deterministic basis = first k bits of one
    // pattern + a sprinkling — the workload is the popcount, not what's
    // canonicalised.
    let mut rref: Vec<u64> = (0..k as usize)
        .map(|i| ((1u64 << i) | (0xA5A5A5A5A5A5A5A5u64 << (i & 7))) & ((1u64 << n) - 1))
        .collect();
    // Ensure non-degeneracy.
    for i in 1..rref.len() { rref[i] |= 1u64 << ((n - 1) as usize - (i % n as usize)); }
    let cw = build_codewords(&rref);
    let l = cw.len() as u64;

    println!("# codeword array: {} entries ({} KiB)", fmt_thousands(l), (l * 8) / 1024);

    println!("\n## Probe A — per-codeword popcount (mirrors canon.rs:110-112)");
    let (a1_ns, a1_cyc, _) = run("A1 scalar count_ones", l, iters, || probe_a1_scalar_count_ones(&cw));
    let (a2_ns, a2_cyc, _) = if is_x86_feature_detected!("popcnt") {
        run("A2 popcnt intrinsic", l, iters, || unsafe { probe_a2_popcnt_intrinsic(&cw) })
    } else { (f64::NAN, f64::NAN, 0) };
    let (a3_ns, a3_cyc, _) = if is_x86_feature_detected!("avx2") {
        run("A3 AVX2 Harley-Seal", l, iters, || unsafe { probe_a3_avx2_harley_seal(&cw) })
    } else { (f64::NAN, f64::NAN, 0) };

    println!("\n## Probe B — per-column degree (mirrors canon.rs:113-122)");
    // Per-column probe: 1 op = 1 codeword × 1 column visited (n × l per call).
    let ops_b = (n as u64) * l;
    // Use fewer iters because each call is more expensive.
    let iters_b = (iters / 5).max(50);
    let (b1_ns, b1_cyc, _) = run("B1 scalar per-column", ops_b, iters_b, || {
        let v = probe_b1_per_column_scalar(&cw, n);
        v.iter().map(|&x| x as u64).sum()
    });
    let (b2_ns, b2_cyc, _) = run("B2 transpose + popcount", ops_b, iters_b, || {
        let v = probe_b2_per_column_transpose(&cw, n);
        v.iter().map(|&x| x as u64).sum()
    });

    println!("\n## Summary (ratio = baseline / variant; >1 means variant is faster)");
    let pr = |label: &str, base: f64, x: f64| {
        if x.is_nan() {
            println!("{label}: variant unavailable on this CPU");
        } else {
            let r = base / x;
            println!("{label}: {ratio:.2}× ({x:.3} ns/op)", label = label, ratio = r, x = x);
        }
    };
    println!("# Probe A baseline = A1 scalar count_ones @ {a1_ns:.3} ns/op ({a1_cyc:.2} cyc/op)");
    pr("  vs A2 popcnt intrinsic", a1_ns, a2_ns);
    pr("  vs A3 AVX2 Harley-Seal", a1_ns, a3_ns);
    println!("# Probe B baseline = B1 scalar per-column @ {b1_ns:.3} ns/op ({b1_cyc:.2} cyc/op)");
    pr("  vs B2 transpose + popcount", b1_ns, b2_ns);
}
