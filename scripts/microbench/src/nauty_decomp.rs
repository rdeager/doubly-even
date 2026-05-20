//! Decomposition of sparsenauty's per-call cost (Q6 audit).
//!
//! No root needed (samply / perf require perf_event_paranoid <= 1).
//! Times three variants on a fixed representative graph using rdtsc:
//!
//!   A. full sparsenauty   (production:  getcanon=TRUE,  produces canong)
//!   B. autom-only         (              getcanon=FALSE,  no canong)
//!   C. graph construction only (no nauty call)
//!
//! Deltas decompose the cost:
//!   setup            = C
//!   autom backtrack  = B - C
//!   canonical pass   = A - B
//!
//! The graph is a synthetic Q_D-style bipartite — 16 codeword vertices of
//! ascending weight + 22 column vertices — matching the shape that
//! production hits at N = 22 (mean |C_low| ~ 16 per audit numbers).
//!
//! Build & run:
//!   cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     --bin nauty_decomp -- --iters 5000

use std::arch::x86_64::_rdtsc;
use std::ffi::c_int;
use std::ptr;

use nauty_Traces_sys::{
    optionblk, sparsegraph, sparsenauty, statsblk, FALSE, TRUE,
};

#[inline(always)]
unsafe fn rdtsc() -> u64 {
    _rdtsc()
}

/// Build a Q_D-style bipartite sparse graph: L codeword vertices, R column
/// vertices. Codewords are the row sums of a synthetic [22, 4] code: the
/// first 4 rows are random 22-bit basis vectors, the remaining 11 are all
/// XOR combinations of subsets of weight 2-3. The exact code doesn't
/// matter — we want the shape (L+R, edges).
fn build_qd_sparsegraph(
    codewords: &[u64],
    n: u32,
) -> (Vec<usize>, Vec<i32>, Vec<i32>, Vec<c_int>, Vec<c_int>, usize) {
    let l = codewords.len();
    let r = n as usize;
    let total = l + r;

    let mut d = vec![0i32; total];
    for (i, &cw) in codewords.iter().enumerate() {
        d[i] = cw.count_ones() as i32;
    }
    for j in 0..r {
        let bit = 1u64 << j;
        let mut deg = 0i32;
        for &cw in codewords {
            if cw & bit != 0 {
                deg += 1;
            }
        }
        d[l + j] = deg;
    }

    let nde: usize = d.iter().map(|&x| x as usize).sum();
    let mut v = vec![0usize; total];
    let mut acc = 0usize;
    for i in 0..total {
        v[i] = acc;
        acc += d[i] as usize;
    }
    let mut e = vec![0i32; nde];
    let mut write = v.clone();
    let mut right_lists: Vec<Vec<i32>> =
        (0..r).map(|j| Vec::with_capacity(d[l + j] as usize)).collect();
    for (i, &cw) in codewords.iter().enumerate() {
        let mut bits = cw;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            e[write[i]] = (l + j) as i32;
            write[i] += 1;
            right_lists[j].push(i as i32);
            bits &= bits - 1;
        }
    }
    for j in 0..r {
        let base = v[l + j];
        for (offset, &nb) in right_lists[j].iter().enumerate() {
            e[base + offset] = nb;
        }
    }

    // Initial partition: (side, degree) — same shape as production.
    let mut by_cell: Vec<(u8, i32, c_int)> = (0..total as c_int)
        .map(|vid| {
            let side: u8 = if (vid as usize) < l { 0 } else { 1 };
            (side, d[vid as usize], vid)
        })
        .collect();
    by_cell.sort_unstable_by_key(|&(s, deg, _)| (s, deg));
    let lab: Vec<c_int> = by_cell.iter().map(|&(_, _, vid)| vid).collect();
    let mut ptn = vec![1i32; total];
    for i in 0..total.saturating_sub(1) {
        let (s1, d1, _) = by_cell[i];
        let (s2, d2, _) = by_cell[i + 1];
        if s1 != s2 || d1 != d2 {
            ptn[i] = 0;
        }
    }
    if total > 0 {
        ptn[total - 1] = 0;
    }

    (v, d, e, lab, ptn, l)
}

// Last-call stats from `run_sparsenauty`; lets `main` print tree-shape
// after the rdtsc loop without changing the timed-region signature.
thread_local! {
    static LAST_NUMNODES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static LAST_TCTOTAL: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static LAST_MAXLEVEL: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    static LAST_NUMGEN: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
}

fn run_sparsenauty(
    codewords: &[u64],
    n: u32,
    getcanon: c_int,
) -> u64 {
    let (mut v, mut d, mut e, mut lab, mut ptn, _l) = build_qd_sparsegraph(codewords, n);
    let total = lab.len();

    let mut sg = sparsegraph {
        nde: e.len(),
        v: v.as_mut_ptr(),
        nv: total as c_int,
        d: d.as_mut_ptr(),
        e: e.as_mut_ptr(),
        w: ptr::null_mut(),
        vlen: v.len(),
        dlen: d.len(),
        elen: e.len(),
        wlen: 0,
    };
    let mut orbits = vec![0i32; total];
    let mut options = optionblk::default_sparse();
    options.getcanon = getcanon;
    options.defaultptn = FALSE;
    let mut stats = statsblk::default();

    // canong buffers: allocated even when getcanon=FALSE so the cost
    // difference reflects only nauty's internal work, not the allocations.
    let mut cg_v = vec![0usize; total];
    let mut cg_d = vec![0i32; total];
    let mut cg_e = vec![0i32; e.len()];
    let mut canon_sg = sparsegraph {
        nde: 0,
        v: cg_v.as_mut_ptr(),
        nv: total as c_int,
        d: cg_d.as_mut_ptr(),
        e: cg_e.as_mut_ptr(),
        w: ptr::null_mut(),
        vlen: cg_v.len(),
        dlen: cg_d.len(),
        elen: cg_e.len(),
        wlen: 0,
    };
    let t0 = unsafe { rdtsc() };
    unsafe {
        sparsenauty(
            &mut sg,
            lab.as_mut_ptr(),
            ptn.as_mut_ptr(),
            orbits.as_mut_ptr(),
            &mut options,
            &mut stats,
            &mut canon_sg,
        );
    }
    let t1 = unsafe { rdtsc() };
    LAST_NUMNODES.with(|c| c.set(stats.numnodes as u64));
    LAST_TCTOTAL.with(|c| c.set(stats.tctotal as u64));
    LAST_MAXLEVEL.with(|c| c.set(stats.maxlevel as i32));
    LAST_NUMGEN.with(|c| c.set(stats.numgenerators as i32));
    // Return stats so the optimiser can't elide the work.
    std::hint::black_box(stats.grpsize1);
    std::hint::black_box(orbits.iter().sum::<i32>());
    t1.wrapping_sub(t0)
}

fn time_construction_only(codewords: &[u64], n: u32) -> u64 {
    let t0 = unsafe { rdtsc() };
    let (v, d, e, lab, ptn, _l) = build_qd_sparsegraph(codewords, n);
    let t1 = unsafe { rdtsc() };
    std::hint::black_box(&v);
    std::hint::black_box(&d);
    std::hint::black_box(&e);
    std::hint::black_box(&lab);
    std::hint::black_box(&ptn);
    t1.wrapping_sub(t0)
}

/// Synthetic [22, k]-style codeword set: random but deterministic
/// codewords with low weight. Generates ~L codewords by enumerating
/// linear combinations of `k` random basis vectors with at most `max_pop`
/// bits set in the mask (Gray-code walk over the 2^k subsets, truncated).
fn synth_codewords(n: u32, k: u32, target_l: usize, seed: u64) -> Vec<u64> {
    let mask = (1u64 << n) - 1;
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut basis: Vec<u64> = Vec::with_capacity(k as usize);
    for _ in 0..k {
        // xorshift64
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        basis.push(s & mask);
    }
    let total = 1usize << k;
    let mut out = Vec::with_capacity(target_l);
    // Enumerate via Gray code so codewords come in some weight order.
    let mut w: u64 = 0;
    out.push(w);
    for mask_idx in 1..total {
        let lo_bit = (mask_idx & mask_idx.wrapping_neg()).trailing_zeros() as usize;
        w ^= basis[lo_bit];
        out.push(w);
        if out.len() >= target_l {
            break;
        }
    }
    // Drop the zero codeword if any duplicates appear; cap at target_l.
    out.truncate(target_l);
    out
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn run_arm<F: FnMut() -> u64>(label: &str, iters: usize, warmup: usize, mut f: F) {
    for _ in 0..warmup {
        std::hint::black_box(f());
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        samples.push(f());
    }
    samples.sort_unstable();
    let median = percentile(&samples, 0.5);
    let p25 = percentile(&samples, 0.25);
    let p75 = percentile(&samples, 0.75);
    let p95 = percentile(&samples, 0.95);
    let mean: u64 = (samples.iter().sum::<u64>()) / (samples.len() as u64);
    // Convert cycles → ns assuming a 3.5 GHz boost clock (13700K
    // P-core). The exact frequency only affects unit conversion; the
    // relative decomposition is independent of it.
    let ns = |cyc: u64| (cyc as f64) / 3.5;
    println!(
        "{label:>22}  iters={iters}  median={median:>8} cyc ({:.0} ns)  p25={p25} p75={p75} p95={p95}  mean={mean}",
        ns(median)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut iters = 5_000usize;
    let mut warmup = 200usize;
    let mut n = 22u32;
    let mut k = 4u32;
    let mut target_l = 16usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => { iters = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup" => { warmup = args[i + 1].parse().unwrap(); i += 2; }
            "--n" => { n = args[i + 1].parse().unwrap(); i += 2; }
            "--k" => { k = args[i + 1].parse().unwrap(); i += 2; }
            "--l" => { target_l = args[i + 1].parse().unwrap(); i += 2; }
            other => panic!("unknown arg: {other}"),
        }
    }

    println!("# Q6 sparsenauty per-call decomposition");
    println!("# graph shape: N={n}, k={k}, target |C_low|={target_l}");
    println!("# cycles → ns assumes 3.5 GHz (13700K P-core)");
    println!();

    let codewords = synth_codewords(n, k, target_l, 0xc0ffee);
    println!(
        "# generated {} codewords, |edges|={:.0}, vertices={}",
        codewords.len(),
        codewords.iter().map(|c| c.count_ones() as f64).sum::<f64>(),
        codewords.len() + n as usize,
    );
    println!();

    run_arm("A_full_canon", iters, warmup, || run_sparsenauty(&codewords, n, TRUE));
    let a_nodes = LAST_NUMNODES.with(|c| c.get());
    let a_tc = LAST_TCTOTAL.with(|c| c.get());
    let a_lvl = LAST_MAXLEVEL.with(|c| c.get());
    let a_gen = LAST_NUMGEN.with(|c| c.get());
    run_arm("B_autom_only", iters, warmup, || run_sparsenauty(&codewords, n, FALSE));
    let b_nodes = LAST_NUMNODES.with(|c| c.get());
    let b_tc = LAST_TCTOTAL.with(|c| c.get());
    let b_lvl = LAST_MAXLEVEL.with(|c| c.get());
    let b_gen = LAST_NUMGEN.with(|c| c.get());
    run_arm("C_construction_only", iters, warmup, || time_construction_only(&codewords, n));

    println!();
    println!("# nauty statsblk on this graph (last call of each arm):");
    println!(
        "  A_full_canon  numnodes={a_nodes}  tctotal={a_tc}  maxlevel={a_lvl}  numgens={a_gen}"
    );
    println!(
        "  B_autom_only  numnodes={b_nodes}  tctotal={b_tc}  maxlevel={b_lvl}  numgens={b_gen}"
    );
}
