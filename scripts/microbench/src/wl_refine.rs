//! Bit-parallel vs sparse WL-refinement micro-benchmark.
//!
//! Direct test of the question "can the bit-vector encoding of our codeword
//! × column incidence beat sparsenauty's sparse-adjacency refinement?" —
//! Direction B of the plan in
//! `~/.claude/plans/our-calls-to-nauty-luminous-scroll.md`.
//!
//! We implement two WL-refine-to-fixpoint kernels on the same bipartite
//! graph that production hands to sparsenauty:
//!
//!   SPARSE:    refine via sparsegraph-style adjacency-list walks. This
//!              is what nauty's `refine_sg` does (modulo cleverer cell-
//!              selection heuristics; we use the same "scan all unprocessed
//!              witness cells" loop for both arms so the comparison is
//!              kernel-vs-kernel, not heuristic-vs-heuristic).
//!
//!   BITPAR:    refine via popcount(mask AND mask). Codewords stored as
//!              u64 (column-membership); column-incidence stored as
//!              bit-packed `[u64; words]` (codeword-membership). Each
//!              signature query is one popcount per word (1 cycle on
//!              Raptor Lake P-core).
//!
//! Correctness check: both kernels must produce the same final equitable
//! partition (the coarsest equitable refinement is canonical given a
//! starting partition).
//!
//! What this microbench does NOT measure:
//!   - Full canonicalisation (no individualisation, no backtrack).
//!   - Generator extraction (only refinement to fixpoint).
//!   - Canonical-form selection / leaf comparison.
//! So this gives the *kernel ratio* between sparse and bit-parallel refine,
//! not the full nauty replacement ratio. Per the plan's outcome gates:
//!   < 1.5× per refine step → STOP (per-probe trap closes the lever)
//!   1.5–3× → contingent GO
//!   ≥ 3× → strong GO for Direction C.
//!
//! Build & run:
//!   cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     --bin wl_refine -- --n 22 --k 11 --iters 2000

use std::arch::x86_64::_rdtsc;
use std::env;
use std::hint::black_box;

#[inline(always)]
unsafe fn rdtsc() -> u64 { _rdtsc() }

// --------------------------------------------------------------- Inputs

/// Build a synthetic Q_D-low-weight codeword set: enumerate all 2^k linear
/// combinations of a random rank-k basis over n bits, then keep only
/// weight-1 and weight-2 codewords (matching `canon.rs::build_low_weight_*`
/// at the default `QD_GRAPH_THRESHOLD = 0`). Returns codewords + n.
fn synth_qd_codewords(n: u32, k: u32, seed: u64) -> Vec<u64> {
    let mask = (1u64 << n) - 1;
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut basis: Vec<u64> = Vec::with_capacity(k as usize);
    for _ in 0..k {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        basis.push(s & mask);
    }
    let total = 1usize << k;
    let mut all = Vec::with_capacity(total);
    let mut w: u64 = 0;
    all.push(w);
    for mask_idx in 1..total {
        let lo_bit = (mask_idx & mask_idx.wrapping_neg()).trailing_zeros() as usize;
        w ^= basis[lo_bit];
        all.push(w);
    }
    // Keep weights 1 and 2 — these are the "low-weight" codewords whose
    // bipartite incidence the Q_D-graph canonicaliser uses.
    let mut lw: Vec<u64> = all.into_iter()
        .filter(|&c| {
            let w = c.count_ones();
            w == 1 || w == 2
        })
        .collect();
    // Dedup (random basis can produce repeats).
    lw.sort_unstable();
    lw.dedup();
    // Cap at a reasonable size; production sees ~14-89 low-weight codewords
    // at k=9-11. If the random basis happens to produce too few, pad with
    // random columns of weight 1 to ensure realistic L.
    if lw.len() < (n as usize).min(20) {
        for j in 0..n {
            let cw = 1u64 << j;
            if !lw.contains(&cw) { lw.push(cw); }
        }
        lw.sort_unstable();
        lw.dedup();
    }
    lw
}

/// Synthetic FULL bipartite codeword set: enumerate all 2^k codewords of
/// a random rank-k basis. This is what the pre-D10 canonicaliser fed to
/// nauty, and what bit-parallel refinement is *theoretically* better
/// suited to (large L → long sparse-adjacency walks for column-side
/// signature, while bit-parallel stays at ceil(L/64) popcounts/query).
fn synth_full_codewords(n: u32, k: u32, seed: u64) -> Vec<u64> {
    let mask = (1u64 << n) - 1;
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut basis: Vec<u64> = Vec::with_capacity(k as usize);
    for _ in 0..k {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        basis.push(s & mask);
    }
    let total = 1usize << k;
    let mut all = Vec::with_capacity(total);
    let mut w: u64 = 0;
    all.push(w);
    for mask_idx in 1..total {
        let lo_bit = (mask_idx & mask_idx.wrapping_neg()).trailing_zeros() as usize;
        w ^= basis[lo_bit];
        all.push(w);
    }
    all
}

// ----------------------------------------------------- Common datatypes

/// Partition represented as (lab, ptn) à la nauty: `lab` is a permutation
/// of `0..total`; `ptn[i] = 0` marks the end of a cell, `ptn[i] = 1` marks
/// an interior position. Identical layout in both kernels for the
/// correctness check.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Partition {
    lab: Vec<u32>,
    ptn: Vec<u8>,
    /// cell_of[v] = the start index (in `lab`) of the cell containing v.
    /// Both kernels maintain this for O(1) cell-lookup; the final value
    /// is what we compare for correctness.
    cell_of: Vec<u32>,
}

impl Partition {
    fn new(total: usize) -> Self {
        Partition {
            lab: (0..total as u32).collect(),
            ptn: vec![1u8; total],
            cell_of: vec![0u32; total],
        }
    }

    /// Sort the partition cells canonically by their *content* (vertex
    /// sets), so two equitable refinements that arrived at the same set
    /// partition compare equal even if `lab` order within cells differs.
    fn canonicalise(&self) -> Vec<Vec<u32>> {
        let total = self.lab.len();
        let mut cells: Vec<Vec<u32>> = Vec::new();
        let mut i = 0;
        while i < total {
            let mut c = Vec::new();
            loop {
                c.push(self.lab[i]);
                if self.ptn[i] == 0 { i += 1; break; }
                i += 1;
            }
            c.sort_unstable();
            cells.push(c);
        }
        cells.sort();
        cells
    }
}

// ----------------------------------- Initial partition (shared by both)

/// Initial partition by (side, degree) — matches `canon.rs:248-269`.
/// `l` = codeword-side count; columns are vertices `[l, l + n)`.
fn initial_partition(cw: &[u64], n: u32) -> Partition {
    let l = cw.len();
    let total = l + n as usize;
    let mut deg = vec![0i32; total];
    for (i, &w) in cw.iter().enumerate() { deg[i] = w.count_ones() as i32; }
    for j in 0..n as usize {
        let bit = 1u64 << j;
        let mut d = 0i32;
        for &w in cw { if w & bit != 0 { d += 1; } }
        deg[l + j] = d;
    }
    let mut by_cell: Vec<(u8, i32, u32)> = (0..total as u32)
        .map(|v| {
            let side: u8 = if (v as usize) < l { 0 } else { 1 };
            (side, deg[v as usize], v)
        })
        .collect();
    by_cell.sort_unstable_by_key(|&(s, d, _)| (s, d));
    let lab: Vec<u32> = by_cell.iter().map(|&(_, _, v)| v).collect();
    let mut ptn = vec![1u8; total];
    for i in 0..total.saturating_sub(1) {
        if by_cell[i].0 != by_cell[i + 1].0 || by_cell[i].1 != by_cell[i + 1].1 {
            ptn[i] = 0;
        }
    }
    if total > 0 { ptn[total - 1] = 0; }
    let mut cell_of = vec![0u32; total];
    let mut start = 0u32;
    for (i, &p) in ptn.iter().enumerate() {
        cell_of[lab[i] as usize] = start;
        if p == 0 { start = (i as u32) + 1; }
    }
    Partition { lab, ptn, cell_of }
}

// ----------------------------------------------------- SPARSE refinement

/// Sparse adjacency lists: `adj[v]` = sorted list of neighbours of v.
/// (Two-way: codewords list their column neighbours and vice versa.)
fn build_sparse_adj(cw: &[u64], n: u32) -> Vec<Vec<u32>> {
    let l = cw.len();
    let total = l + n as usize;
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); total];
    for (i, &w) in cw.iter().enumerate() {
        let mut bits = w;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            adj[i].push((l + j) as u32);
            adj[l + j].push(i as u32);
            bits &= bits - 1;
        }
    }
    adj
}

/// Refine the partition to its coarsest equitable refinement using the
/// sparse adjacency-list kernel. Worklist algorithm: repeatedly pop a
/// witness cell W, and for each cell C, compute deg(v in W) for each v
/// in C; if degs vary, split C.
///
/// Returns the number of refinement *steps* (one step = one signature
/// query for one vertex), for instrumentation.
fn refine_sparse(p: &mut Partition, adj: &[Vec<u32>], l: usize) -> u64 {
    let total = p.lab.len();
    let mut step_count: u64 = 0;
    let mut worklist: Vec<u32> = cell_starts(p);
    let mut sig: Vec<i32> = vec![0; total];
    let mut in_worklist: Vec<bool> = vec![false; total];
    for &s in &worklist { in_worklist[s as usize] = true; }
    let mut mark = vec![false; total];

    let mut witness_buf: Vec<u32> = Vec::with_capacity(total);
    while let Some(w_start) = worklist.pop() {
        in_worklist[w_start as usize] = false;
        let w_end = cell_end(p, w_start as usize);
        witness_buf.clear();
        witness_buf.extend_from_slice(&p.lab[w_start as usize..=w_end]);
        let witness_is_codeword = (witness_buf[0] as usize) < l;
        for &v in &witness_buf { mark[v as usize] = true; }

        let starts = cell_starts(p);
        for c_start in starts {
            let c_end = cell_end(p, c_start as usize);
            if c_start as usize == c_end { continue; }
            let first_v = p.lab[c_start as usize] as usize;
            let cell_is_codeword = first_v < l;
            if witness_is_codeword == cell_is_codeword { continue; }

            for i in c_start as usize..=c_end {
                let v = p.lab[i] as usize;
                let mut s: i32 = 0;
                for &nb in &adj[v] {
                    if mark[nb as usize] { s += 1; }
                }
                sig[v] = s;
                step_count += 1;
            }
            split_cell_by_sig(p, c_start, c_end, &sig, &mut worklist, &mut in_worklist);
        }
        for &v in &witness_buf { mark[v as usize] = false; }
    }
    step_count
}

// --------------------------------------------------- BITPAR refinement

/// Bit-parallel adjacency representation.
///
/// `cw_mask[i]` = column-membership of codeword i (one u64 if n ≤ 64).
/// `col_inc[j]` = codeword-membership of column j, bit-packed across
/// `cw_words = ceil(L/64)` u64s — bit i of `col_inc[j][i / 64]` is set
/// iff codeword i has column j set.
struct BitparAdj {
    cw_mask: Vec<u64>,        // length L, one per codeword (low n bits used)
    col_inc: Vec<Vec<u64>>,   // length N; each is cw_words u64s
    cw_words: usize,
    l: usize,
    n: usize,
}

fn build_bitpar_adj(cw: &[u64], n: u32) -> BitparAdj {
    let l = cw.len();
    let cw_words = (l + 63) / 64;
    let mut col_inc: Vec<Vec<u64>> = (0..n as usize).map(|_| vec![0u64; cw_words]).collect();
    for (i, &w) in cw.iter().enumerate() {
        let i_word = i / 64;
        let i_bit = 1u64 << (i % 64);
        let mut bits = w;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            col_inc[j][i_word] |= i_bit;
            bits &= bits - 1;
        }
    }
    BitparAdj {
        cw_mask: cw.to_vec(),
        col_inc,
        cw_words,
        l,
        n: n as usize,
    }
}

/// Refine via bit-parallel popcount. Same worklist algorithm as the
/// sparse kernel — only the inner signature query changes.
///
/// Two cases:
///   - witness cell on the COLUMN side: for each codeword v in cell C,
///     sig(v) = popcount(cw_mask[v] AND witness_col_mask). One popcount.
///   - witness cell on the CODEWORD side: for each column v in cell C,
///     sig(v) = sum over words of popcount(col_inc[v][w] AND
///     witness_cw_mask[w]). cw_words popcounts.
fn refine_bitpar(p: &mut Partition, adj: &BitparAdj) -> u64 {
    let total = p.lab.len();
    let l = adj.l;
    let mut step_count: u64 = 0;
    let mut worklist: Vec<u32> = cell_starts(p);
    let mut sig: Vec<i32> = vec![0; total];
    let mut in_worklist: Vec<bool> = vec![false; total];
    for &s in &worklist { in_worklist[s as usize] = true; }

    while let Some(w_start) = worklist.pop() {
        in_worklist[w_start as usize] = false;
        let w_end = cell_end(p, w_start as usize);
        let witness: &[u32] = &p.lab[w_start as usize..=w_end];
        // Cheap dispatch: witness is uniformly on one side (initial
        // partition splits by side; subsequent splits preserve side).
        let witness_is_codeword = (witness[0] as usize) < l;

        let starts = cell_starts(p);
        if witness_is_codeword {
            // Build witness_cw_mask: packed bitmask over codewords.
            let mut wmask = vec![0u64; adj.cw_words];
            for &v in witness {
                let vi = v as usize;
                wmask[vi / 64] |= 1u64 << (vi % 64);
            }
            for c_start in starts {
                let c_end = cell_end(p, c_start as usize);
                if c_start as usize == c_end { continue; }
                // Only column cells respond to a codeword-side witness.
                let first_v = p.lab[c_start as usize] as usize;
                if first_v < l { continue; } // codeword-side cell, signature ≡ 0

                for i in c_start as usize..=c_end {
                    let v = p.lab[i] as usize;
                    let col_j = v - l;
                    let inc = &adj.col_inc[col_j];
                    let mut s: i32 = 0;
                    // Bit-parallel popcount over cw_words words.
                    for w in 0..adj.cw_words {
                        s += (inc[w] & wmask[w]).count_ones() as i32;
                    }
                    sig[v] = s;
                    step_count += 1;
                }
                split_cell_by_sig(p, c_start, c_end, &sig, &mut worklist, &mut in_worklist);
            }
        } else {
            // Witness is a column cell. Build column-side witness mask.
            let mut wmask: u64 = 0;
            for &v in witness {
                let col_j = (v as usize) - l;
                wmask |= 1u64 << col_j;
            }
            for c_start in starts {
                let c_end = cell_end(p, c_start as usize);
                if c_start as usize == c_end { continue; }
                // Only codeword cells respond to a column-side witness.
                let first_v = p.lab[c_start as usize] as usize;
                if first_v >= l { continue; }

                for i in c_start as usize..=c_end {
                    let v = p.lab[i] as usize;
                    let s = (adj.cw_mask[v] & wmask).count_ones() as i32;
                    sig[v] = s;
                    step_count += 1;
                }
                split_cell_by_sig(p, c_start, c_end, &sig, &mut worklist, &mut in_worklist);
            }
        }
    }
    step_count
}

// ---------------------------------------------------- Shared mechanics

fn cell_starts(p: &Partition) -> Vec<u32> {
    let mut out = Vec::new();
    let mut start = 0u32;
    out.push(start);
    for (i, &pp) in p.ptn.iter().enumerate() {
        if pp == 0 && (i as u32) + 1 < p.ptn.len() as u32 {
            start = (i as u32) + 1;
            out.push(start);
        }
    }
    out
}

fn cell_end(p: &Partition, start: usize) -> usize {
    let mut i = start;
    while p.ptn[i] != 0 { i += 1; }
    i
}

/// Sort vertices in `[c_start, c_end]` by `sig[v]`, then write back into
/// `p.lab` and set `p.ptn` markers where `sig` changes. New (non-singleton)
/// sub-cells get pushed onto the worklist.
fn split_cell_by_sig(
    p: &mut Partition,
    c_start: u32, c_end: usize,
    sig: &[i32],
    worklist: &mut Vec<u32>,
    in_worklist: &mut [bool],
) {
    let cs = c_start as usize;
    let mut buf: Vec<(i32, u32)> = (cs..=c_end)
        .map(|i| (sig[p.lab[i] as usize], p.lab[i]))
        .collect();
    buf.sort_by_key(|&(s, _)| s);
    // Detect whether the cell actually splits.
    let first_sig = buf[0].0;
    let splits = buf.iter().any(|&(s, _)| s != first_sig);
    for (offset, &(_, v)) in buf.iter().enumerate() {
        p.lab[cs + offset] = v;
    }
    if !splits { return; }
    // Re-set ptn markers and cell_of for the new sub-cells.
    let mut sub_start = cs as u32;
    for offset in cs..c_end {
        let here = buf[offset - cs].0;
        let next = buf[offset - cs + 1].0;
        if here != next {
            p.ptn[offset] = 0;
            // New cell starts at offset+1.
            // Update cell_of for the just-closed sub-cell.
            for k in sub_start as usize..=offset {
                p.cell_of[p.lab[k] as usize] = sub_start;
            }
            // Push the closed sub-cell to the worklist (its size > 0).
            if !in_worklist[sub_start as usize] {
                worklist.push(sub_start);
                in_worklist[sub_start as usize] = true;
            }
            sub_start = (offset as u32) + 1;
        } else {
            p.ptn[offset] = 1;
        }
    }
    // ptn[c_end] = 0 already (it was the original cell end); update
    // cell_of for the final sub-cell and push it too.
    for k in sub_start as usize..=c_end {
        p.cell_of[p.lab[k] as usize] = sub_start;
    }
    if !in_worklist[sub_start as usize] {
        worklist.push(sub_start);
        in_worklist[sub_start as usize] = true;
    }
}

// ------------------------------------------------------- Timing harness

fn time_one<F: FnMut() -> u64>(label: &str, iters: usize, warmup: usize, mut f: F) -> (f64, u64) {
    let mut sink: u64 = 0;
    for _ in 0..warmup { sink = sink.wrapping_add(f()); }
    let mut samples: Vec<u64> = Vec::with_capacity(iters);
    let mut step_total: u64 = 0;
    for _ in 0..iters {
        let t0 = unsafe { rdtsc() };
        let steps = f();
        let t1 = unsafe { rdtsc() };
        samples.push(t1.wrapping_sub(t0));
        step_total += steps;
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let p25 = samples[samples.len() / 4];
    let p75 = samples[3 * samples.len() / 4];
    let mean: u64 = samples.iter().sum::<u64>() / samples.len() as u64;
    // 13700K P-core boost: 5.3 GHz. Adjust for your CPU; relative ratios
    // are clock-independent.
    let ghz = 5.3;
    let ns = |c: u64| (c as f64) / ghz;
    let avg_steps = step_total as f64 / iters as f64;
    println!(
        "{label:>20}  median={median:>7} cyc ({:.0} ns)  p25={p25} p75={p75}  mean={mean}  \
         avg_steps={avg_steps:.0}  ns/step={:.2}  sink={}",
        ns(median),
        ns(median) / avg_steps,
        sink,
    );
    (ns(median), step_total / iters as u64)
}

// ------------------------------------------------------------------ main

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut n: u32 = 22;
    let mut k: u32 = 11;
    let mut iters: usize = 2000;
    let mut warmup: usize = 100;
    let mut seed: u64 = 0xc0ffee;
    let mut mode = "qd".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => { n = args[i + 1].parse().unwrap(); i += 2; }
            "--k" => { k = args[i + 1].parse().unwrap(); i += 2; }
            "--iters" => { iters = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup" => { warmup = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "--mode" => { mode = args[i + 1].clone(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(2); }
        }
    }
    println!("# bit-parallel vs sparse WL-refinement microbench");
    println!("# N={n}, k={k}, iters={iters}, seed={seed:#x}, mode={mode}");
    assert!(n <= 64, "this microbench packs columns into one u64");

    let cw = match mode.as_str() {
        "qd" => synth_qd_codewords(n, k, seed),
        "full" => synth_full_codewords(n, k, seed),
        other => { eprintln!("unknown mode: {other}"); std::process::exit(2); }
    };
    let l = cw.len();
    let total = l + n as usize;
    let edges: u64 = cw.iter().map(|c| c.count_ones() as u64).sum();
    println!("# graph: L={l} codewords, N={n} columns, |V|={total}, |E|={edges}");

    let sparse_adj = build_sparse_adj(&cw, n);
    let bitpar_adj = build_bitpar_adj(&cw, n);

    // Correctness: both kernels reach the same final partition.
    let init = initial_partition(&cw, n);
    let mut p_sparse = init.clone();
    let _ = refine_sparse(&mut p_sparse, &sparse_adj, l);
    let mut p_bitpar = init.clone();
    let _ = refine_bitpar(&mut p_bitpar, &bitpar_adj);
    let c_sparse = p_sparse.canonicalise();
    let c_bitpar = p_bitpar.canonicalise();
    if c_sparse == c_bitpar {
        println!("# correctness: OK ({} cells)", c_sparse.len());
    } else {
        println!("# CORRECTNESS FAIL: sparse and bitpar disagree!");
        println!("  sparse cells: {:?}", c_sparse);
        println!("  bitpar cells: {:?}", c_bitpar);
        std::process::exit(1);
    }

    println!();
    println!("# Refinement to fixpoint (one call = full WL-refine of a fresh init partition)");
    let init1 = init.clone();
    let (ns_sparse, steps_sparse) = time_one("SPARSE adj-list", iters, warmup, || {
        let mut p = init1.clone();
        refine_sparse(&mut p, &sparse_adj, l)
    });
    let init2 = init.clone();
    let (ns_bitpar, steps_bitpar) = time_one("BITPAR popcount", iters, warmup, || {
        let mut p = init2.clone();
        refine_bitpar(&mut p, &bitpar_adj)
    });

    println!();
    println!("# Summary");
    println!("# SPARSE: {ns_sparse:.0} ns/call  ({steps_sparse} sig-queries/call, \
              {:.2} ns/query)",
              ns_sparse / steps_sparse as f64);
    println!("# BITPAR: {ns_bitpar:.0} ns/call  ({steps_bitpar} sig-queries/call, \
              {:.2} ns/query)",
              ns_bitpar / steps_bitpar as f64);
    let ratio = ns_sparse / ns_bitpar;
    println!("# Speedup BITPAR vs SPARSE: {ratio:.2}× wall, \
              per-query {:.2}×",
              (ns_sparse / steps_sparse as f64) / (ns_bitpar / steps_bitpar as f64));
    println!();
    let nauty_floor_ns: f64 = 78_000.0 / (67.0 * 163.0); // 78 µs / (nodes × tc/node) ≈ 7.1 ns
    println!("# Reference: sparsenauty production budget at N=22 is ~{nauty_floor_ns:.1} ns/refine-step");
    println!("#   (78 µs/call ÷ (67 backtrack nodes × 163 tc-evals/node), from Q6 audit)");
    println!("# BITPAR per-query vs nauty per-refine-step: {:.2}×",
              nauty_floor_ns / (ns_bitpar / steps_bitpar as f64));
    let _ = black_box(ns_sparse);
    let _ = black_box(ns_bitpar);
}
