//! SIMD go/no-go probes (SIMD design session, 2026-06-11).
//!
//! EXPLORATORY — not a committed harness like `orbit_probe`/`vhalf_sweep`;
//! numbers here feed the SIMD design note only. Two modes:
//!
//! `--sigma`: replay the real rank-2/3 parent dumps through the m4r BFS
//!   with the apply and probe phases TIMED SEPARATELY (two-pass per
//!   chunk, minima asserted equal to the production gen-major body).
//!   This is the Amdahl split for any bitsliced-apply idea: the probe
//!   phase (random bitset RMW + push) is inherently scalar.
//!   Also reports a bitsliced 64-image matmul throughput microkernel
//!   (transpose-in amortised across gens + row-walk matmul + butterfly
//!   transpose-out) against the m4r apply ns/image on the same real
//!   generators — values are synthetic (arithmetic is value-blind),
//!   shape/L/gens are real.
//!
//! `--vhalf`: φ phase-0 fused loop with the histogram split widened
//!   4 → 8 → 16 sub-counts, plus a weights-only/histogram-only split
//!   timing. Run under both baseline and -C target-cpu=x86-64-v3 to
//!   see what codegen alone does before any hand intrinsics.
//!
//! Build & run (from /workspace/src):
//!   RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=scripts/microbench/target-v3 \
//!     cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     --bin simd_probe -- --sigma
//! Pin it: `taskset -c 4 ...`.

use doubly_even_core::orbit::{m4r_build, orbit_minima_m4r, singular_reps_q};
use microbench::timing::{cycles_to_ns, mono_cycles, ns_per_cycle};
use microbench::XorShift64;
use std::env;
use std::hint::black_box;

// ----- production-shape reference arms (ground-truth minima, baseline
// ----- timing) come straight from doubly-even-core; the local M4rGen /
// ----- BitSet below exist only for the EXPERIMENTAL split-timed body.

struct BitSet(Vec<u64>);

impl BitSet {
    fn with_capacity(bits: usize) -> Self {
        BitSet(vec![0u64; bits.div_ceil(64)])
    }
    #[inline]
    fn put(&mut self, i: usize) -> bool {
        let w = &mut self.0[i >> 6];
        let m = 1u64 << (i & 63);
        let old = *w & m != 0;
        *w |= m;
        old
    }
}

fn is_identity_mat(m: &[u64]) -> bool {
    m.iter().enumerate().all(|(i, &c)| c == 1u64 << i)
}

struct M4rGen {
    tables: Vec<[u64; 256]>,
}

impl M4rGen {
    fn build(m: &[u64]) -> Self {
        let l = m.len();
        let n_chunks = l.div_ceil(8);
        let mut tables = vec![[0u64; 256]; n_chunks];
        for (c, t) in tables.iter_mut().enumerate() {
            let base = c * 8;
            let width = (l - base).min(8);
            for b in 1usize..1 << width {
                t[b] = t[b & (b - 1)] ^ m[base + (b.trailing_zeros() as usize)];
            }
        }
        M4rGen { tables }
    }

    #[inline]
    fn apply(&self, x: u64) -> u64 {
        let mut out = 0u64;
        for (c, t) in self.tables.iter().enumerate() {
            out ^= t[((x >> (c * 8)) & 0xff) as usize];
        }
        out
    }
}

const CHUNK: usize = 1024;

/// m4r BFS with apply and probe phases timed separately (two passes per
/// chunk through a flat image buffer). Per-level new-element set is
/// probe-order-independent, so minima match the fused body exactly.
/// Returns (minima, apply_cycles, probe_cycles, images).
fn bfs_m4r_split(
    reps_sorted: &[u64],
    gens: &[&Vec<u64>],
    l: u32,
) -> (Vec<u64>, u64, u64, u64) {
    let m4r: Vec<M4rGen> = gens.iter().map(|g| M4rGen::build(g)).collect();
    let universe = 1usize << l;
    let mut seen = BitSet::with_capacity(universe);
    let mut minima: Vec<u64> = Vec::new();
    let cap = reps_sorted.len();
    let mut queue: Vec<u64> = Vec::with_capacity(cap);
    let mut next: Vec<u64> = Vec::with_capacity(cap);
    let mut images: Vec<u64> = Vec::with_capacity(CHUNK * gens.len().max(1));
    let mut apply_cyc = 0u64;
    let mut probe_cyc = 0u64;
    let mut n_images = 0u64;
    for &v in reps_sorted {
        if seen.put(v as usize) {
            continue;
        }
        minima.push(v);
        queue.clear();
        queue.push(v);
        while !queue.is_empty() {
            next.clear();
            for cur_chunk in queue.chunks(CHUNK) {
                let c0 = mono_cycles();
                images.clear();
                for g in &m4r {
                    for &current in cur_chunk {
                        images.push(g.apply(current));
                    }
                }
                let c1 = mono_cycles();
                for &new_v in &images {
                    if !seen.put(new_v as usize) {
                        next.push(new_v);
                    }
                }
                let c2 = mono_cycles();
                apply_cyc += c1.wrapping_sub(c0);
                probe_cyc += c2.wrapping_sub(c1);
                n_images += images.len() as u64;
            }
            std::mem::swap(&mut queue, &mut next);
        }
    }
    (minima, apply_cyc, probe_cyc, n_images)
}

// ----- bitsliced 64-image matmul microkernel (throughput only)

/// 64x64 bit-matrix in-place transpose (Hacker's Delight 7-3 shape).
#[inline]
fn transpose64(a: &mut [u64; 64]) {
    const MASKS: [(usize, u64); 6] = [
        (32, 0x0000_0000_FFFF_FFFF),
        (16, 0x0000_FFFF_0000_FFFF),
        (8, 0x00FF_00FF_00FF_00FF),
        (4, 0x0F0F_0F0F_0F0F_0F0F),
        (2, 0x3333_3333_3333_3333),
        (1, 0x5555_5555_5555_5555),
    ];
    for &(j, m) in &MASKS {
        let mut k = 0usize;
        while k < 64 {
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= t << j;
            a[k + j] ^= t;
            k = (k + j + 1) & !j;
        }
    }
}

/// transpose64 self-check: involution + a hand case.
fn transpose64_selfcheck() {
    let mut rng = XorShift64::new(7);
    let mut a = [0u64; 64];
    for w in a.iter_mut() {
        *w = rng.next();
    }
    let orig = a;
    transpose64(&mut a);
    // bit j of a[i] must equal bit i of orig[j]
    for i in 0..64 {
        for j in 0..64 {
            assert_eq!((a[i] >> j) & 1, (orig[j] >> i) & 1, "transpose wrong at ({i},{j})");
        }
    }
    transpose64(&mut a);
    assert_eq!(a, orig, "transpose64 is not an involution");
}

/// rows[j] over input index i: bit i set iff gen column m[i] has bit j.
fn gen_rows(m: &[u64], l: usize) -> Vec<u64> {
    let mut rows = vec![0u64; l];
    for (i, &col) in m.iter().enumerate() {
        let mut c = col;
        while c != 0 {
            let j = c.trailing_zeros() as usize;
            rows[j] |= 1u64 << i;
            c &= c - 1;
        }
    }
    rows
}

/// Bitsliced batch-of-64 apply: transpose-in (once, shared across gens)
/// + per-gen row-walk matmul + per-gen butterfly transpose-out.
/// Returns cycles split (tin, matmul, tout) over `iters` sweeps.
fn bitslice_throughput(
    gens_rows: &[Vec<u64>],
    batch: &[u64; 64],
    l: usize,
    iters: u64,
) -> (u64, u64, u64) {
    let mut tin_cyc = 0u64;
    let mut mm_cyc = 0u64;
    let mut tout_cyc = 0u64;
    let mut slice = [0u64; 64];
    let mut out = [0u64; 64];
    for _ in 0..iters {
        // transpose-in: element-major -> bit-plane-major
        let c0 = mono_cycles();
        slice = *black_box(batch);
        transpose64(&mut slice);
        let c1 = mono_cycles();
        tin_cyc += c1.wrapping_sub(c0);
        for rows in gens_rows {
            // matmul: out bit-plane j = XOR of in planes in rows[j]
            let c2 = mono_cycles();
            out = [0u64; 64];
            for (j, &r) in rows.iter().enumerate().take(l) {
                let mut acc = 0u64;
                let mut u = r;
                while u != 0 {
                    let i = u.trailing_zeros() as usize;
                    acc ^= slice[i];
                    u &= u - 1;
                }
                out[j] = acc;
            }
            let c3 = mono_cycles();
            // transpose-out: bit-plane-major -> element-major (probe-ready)
            transpose64(&mut out);
            black_box(&out);
            let c4 = mono_cycles();
            mm_cyc += c3.wrapping_sub(c2);
            tout_cyc += c4.wrapping_sub(c3);
        }
    }
    black_box(slice);
    (tin_cyc, mm_cyc, tout_cyc)
}

// ----- dump-file loading (hand-copied from orbit_probe)

struct ParentInput {
    name: String,
    n: u32,
    k: u32,
    l: u32,
    v_basis: Vec<u64>,
    gens: Vec<Vec<u64>>,
}

fn parse_dump(path: &std::path::Path) -> ParentInput {
    let text = std::fs::read_to_string(path).expect("read dump file");
    let mut n = 0u32;
    let mut k = 0u32;
    let mut l = 0u32;
    let mut v_basis: Vec<u64> = Vec::new();
    let mut gens: Vec<Vec<u64>> = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("n") => n = it.next().unwrap().parse().unwrap(),
            Some("k") => k = it.next().unwrap().parse().unwrap(),
            Some("l") => l = it.next().unwrap().parse().unwrap(),
            Some("v_basis") => {
                v_basis = it.map(|w| u64::from_str_radix(w, 16).unwrap()).collect()
            }
            Some("gen") => {
                gens.push(it.map(|w| u64::from_str_radix(w, 16).unwrap()).collect())
            }
            _ => {}
        }
    }
    assert_eq!(v_basis.len() as u32, l);
    gens.retain(|g| !is_identity_mat(g));
    ParentInput {
        name: path.file_stem().unwrap().to_string_lossy().into_owned(),
        n,
        k,
        l,
        v_basis,
        gens,
    }
}

fn sigma_mode(dir: &str, filter: &str) {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .filter(|p| filter.is_empty() || p.to_string_lossy().contains(filter))
        .collect();
    files.sort();
    println!("# {} parents from {dir} (filter {filter:?})", files.len());

    // (n, k) -> [m4r_fused, apply, probe, images, calls, tin, mm, tout, bs_batches]
    let mut agg: std::collections::BTreeMap<(u32, u32), [u64; 9]> = Default::default();
    let mut rng = XorShift64::new(0xD15C);

    for path in &files {
        let p = parse_dump(path);
        let gens: Vec<&Vec<u64>> = p.gens.iter().collect();
        let reps = singular_reps_q(&p.v_basis);
        let mut reps_sorted = reps;
        reps_sorted.sort_unstable();

        // Production fused body (doubly-even-core), build included in
        // the timed region — ground truth minima + baseline number.
        let c0 = mono_cycles();
        let tables = m4r_build(&gens, p.l);
        let minima_fused = orbit_minima_m4r(&reps_sorted, &tables, p.l);
        let fused_cyc = mono_cycles().wrapping_sub(c0);

        let (minima_split, apply_cyc, probe_cyc, images) =
            bfs_m4r_split(&reps_sorted, &gens, p.l);
        assert_eq!(minima_split, minima_fused, "split minima diverge on {}", p.name);

        // bitsliced throughput on this parent's real generator set
        let lmask = (1u64 << p.l) - 1;
        let mut batch = [0u64; 64];
        for b in batch.iter_mut() {
            *b = rng.next() & lmask;
        }
        let gens_rows: Vec<Vec<u64>> =
            p.gens.iter().map(|g| gen_rows(g, p.l as usize)).collect();
        // ~the same image count per parent as one BFS level sweep
        let bs_iters = (images / (64 * gens.len().max(1) as u64)).clamp(1, 20_000);
        let (tin, mm, tout) =
            bitslice_throughput(&gens_rows, &batch, p.l as usize, bs_iters);

        let e = agg.entry((p.n, p.k)).or_default();
        e[0] += fused_cyc;
        e[1] += apply_cyc;
        e[2] += probe_cyc;
        e[3] += images;
        e[4] += 1;
        e[5] += tin;
        e[6] += mm;
        e[7] += tout;
        e[8] += bs_iters * 64 * gens.len() as u64;
    }

    println!(
        "{:>3} {:>2} {:>6} {:>9} {:>9} {:>9} {:>7} {:>11} {:>9} {:>9} | {:>9} {:>9} {:>9} {:>9}",
        "N", "k", "calls", "fused_ms", "apply_ms", "probe_ms", "ap_shr", "images", "ap_ns/im", "pr_ns/im",
        "tin_ns/im", "mm_ns/im", "tout_ns/im", "bs_ns/im"
    );
    for ((n, k), e) in &agg {
        let fused_ms = cycles_to_ns(e[0]) / 1e6;
        let apply_ms = cycles_to_ns(e[1]) / 1e6;
        let probe_ms = cycles_to_ns(e[2]) / 1e6;
        let img = e[3] as f64;
        let bs_img = e[8] as f64;
        let tin = cycles_to_ns(e[5]) / bs_img;
        let mm = cycles_to_ns(e[6]) / bs_img;
        let tout = cycles_to_ns(e[7]) / bs_img;
        println!(
            "{:>3} {:>2} {:>6} {:>9.1} {:>9.1} {:>9.1} {:>7.2} {:>11} {:>9.2} {:>9.2} | {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            n, k, e[4], fused_ms, apply_ms, probe_ms,
            apply_ms / (apply_ms + probe_ms),
            e[3],
            cycles_to_ns(e[1]) / img,
            cycles_to_ns(e[2]) / img,
            tin, mm, tout, tin + mm + tout,
        );
    }
}

// ----- vhalf histogram-widening probes

/// Production clone: 4-way split histogram (vhalf_sweep::vhalf_fused).
#[inline(never)]
fn vhalf_fused4(cwords: &[u64], v: u64, wt_v: &mut Vec<u8>) -> [u32; 65] {
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

/// 8-way split histogram.
#[inline(never)]
fn vhalf_fused8(cwords: &[u64], v: u64, wt_v: &mut Vec<u8>) -> [u32; 65] {
    let h = cwords.len();
    wt_v.clear();
    wt_v.resize(h, 0);
    for (wt, &cw) in wt_v.iter_mut().zip(cwords.iter()) {
        *wt = (cw ^ v).count_ones() as u8;
    }
    let mut counts8 = [[0u32; 65]; 8];
    let mut chunks = wt_v.chunks_exact(8);
    for c in &mut chunks {
        for lane in 0..8 {
            counts8[lane][c[lane] as usize] += 1;
        }
    }
    for &wt in chunks.remainder() {
        counts8[0][wt as usize] += 1;
    }
    let mut counts_v = [0u32; 65];
    for w in 0..65 {
        let mut s = 0u32;
        for lane in counts8.iter() {
            s += lane[w];
        }
        counts_v[w] = s;
    }
    counts_v
}

/// Weights-only pass (store, no histogram) + histogram-only pass,
/// timed separately by the caller.
#[inline(never)]
fn vhalf_weights_only(cwords: &[u64], v: u64, wt_v: &mut Vec<u8>) {
    let h = cwords.len();
    wt_v.clear();
    wt_v.resize(h, 0);
    for (wt, &cw) in wt_v.iter_mut().zip(cwords.iter()) {
        *wt = (cw ^ v).count_ones() as u8;
    }
}

#[inline(never)]
fn hist_only4(wt_v: &[u8]) -> [u32; 65] {
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

fn vhalf_mode(min_kp1: usize, max_kp1: usize, n_bits: u64) {
    println!("# simd_probe --vhalf: histogram widening + weights/hist split");
    println!("# ns_per_cycle = {:.4}, N = {n_bits}", ns_per_cycle());
    println!(
        "{:>4} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "k+1", "tbl_KB", "f4_ns/el", "f8_ns/el", "wts_ns/el", "hist_ns/el", "w+h_ns/el"
    );
    let mut rng = XorShift64::new(0x5EED_F00D);
    let mask = (1u64 << n_bits) - 1;
    let mut wt_v: Vec<u8> = Vec::new();

    for kp1 in min_kp1..=max_kp1 {
        let h = 1usize << (kp1 - 1);
        let cwords: Vec<u64> = (0..h).map(|_| rng.next() & mask).collect();
        let cands: Vec<u64> = (0..64).map(|_| rng.next() & mask).collect();
        let iters = ((1u64 << 26) / (h as u64 * 64)).max(1);

        let c0 = mono_cycles();
        for _ in 0..iters {
            for &v in &cands {
                black_box(vhalf_fused4(&cwords, black_box(v), &mut wt_v)[4]);
            }
        }
        let f4 = mono_cycles().wrapping_sub(c0) / (iters * 64);

        let c0 = mono_cycles();
        for _ in 0..iters {
            for &v in &cands {
                black_box(vhalf_fused8(&cwords, black_box(v), &mut wt_v)[4]);
            }
        }
        let f8 = mono_cycles().wrapping_sub(c0) / (iters * 64);

        // sanity: same counts
        let a = vhalf_fused4(&cwords, cands[0], &mut wt_v);
        let b = vhalf_fused8(&cwords, cands[0], &mut wt_v);
        assert_eq!(a, b, "histogram widening changed counts");

        let c0 = mono_cycles();
        for _ in 0..iters {
            for &v in &cands {
                vhalf_weights_only(&cwords, black_box(v), &mut wt_v);
                black_box(wt_v[0]);
            }
        }
        let wts = mono_cycles().wrapping_sub(c0) / (iters * 64);

        let c0 = mono_cycles();
        for _ in 0..iters {
            for _ in &cands {
                black_box(hist_only4(black_box(&wt_v))[4]);
            }
        }
        let hist = mono_cycles().wrapping_sub(c0) / (iters * 64);

        let hf = h as f64;
        println!(
            "{:>4} {:>9.1} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            kp1,
            (h * 8) as f64 / 1024.0,
            cycles_to_ns(f4) / hf,
            cycles_to_ns(f8) / hf,
            cycles_to_ns(wts) / hf,
            cycles_to_ns(hist) / hf,
            cycles_to_ns(wts + hist) / hf,
        );
    }
}

fn flag(name: &str) -> bool {
    env::args().any(|a| a == name)
}

fn arg_str(name: &str, default: &str) -> String {
    let args: Vec<String> = env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    println!("# simd_probe (exploratory; SIMD design session 2026-06-11)");
    println!("# ns_per_cycle = {:.4}", ns_per_cycle());
    transpose64_selfcheck();
    if flag("--sigma") {
        let dir = arg_str("--inputs", "scripts/bench-results/sigma-inputs");
        let filter = arg_str("--filter", "");
        sigma_mode(&dir, &filter);
    }
    if flag("--vhalf") {
        vhalf_mode(8, 16, 48);
    }
}
