//! σ_Q orbit-min BFS probe — REAL-INPUT replay + portable restructures
//! (pre-SIMD checklist item 4, 2026-06-10).
//!
//! The seeder timeline located the parallel bottleneck in the rank-2/3
//! orbit-min BFS calls (`aut_orbit_minima_q_witt`, 86–89 % of σ_Q), and
//! the min_l re-measure showed the D16 *intra-call* pool never beats the
//! sequential BFS at L = 20/21. This bin asks the platform-agnostic
//! questions that come before any SIMD work:
//!
//!   (a) is the BFS latency/dependency-bound or bandwidth-bound?
//!       (`--poles`: dependent-chase vs independent-probe ns on the
//!       same-size bitset = the available memory-level parallelism)
//!   (b) do portable restructures win? Variants, all minima-identical:
//!         base   — the PRODUCTION legacy walk body
//!                  (`doubly_even_core::orbit::orbit_minima_walk`)
//!         put    — single read-modify-write probe per image
//!         batch  — two passes per level chunk: compute all images into
//!                  a flat buffer, then probe; gives the OoO core
//!                  independent loads to overlap (MLP)
//!         bucket — batch + radix-bucket images by high bits before
//!                  probing (page/TLB locality; only plausibly useful
//!                  at L ≥ 24 where the bitset spans many pages)
//!         m4r    — the PRODUCTION D18 body (`m4r_build` +
//!                  `orbit_minima_m4r`; fixed internal chunk = 1024 —
//!                  `--batch-chunk` only affects the local variants)
//!       Production arms link `doubly-even-core` directly (clones
//!       retired); local variants are asserted minima-equal AGAINST
//!       production on every parent.
//!   (c) is there cross-parent structure worth sharing? (`--stats`:
//!       per-parent generators / reps / minima / probes / orbit sizes)
//!
//! Real inputs come from the kernel dump test
//! (`cargo test --release dump_sigma_inputs -- --ignored` in rust/),
//! one file per rank-2/3 parent at N = 26, 27 under
//! `scripts/bench-results/sigma-inputs/`. Random-GL synthetics are NOT
//! used here: real aut images are permutation-induced with many small
//! orbits, and restructure verdicts on few-giant-orbit synthetics would
//! not transfer.
//!
//! Every variant's minima Vec is asserted equal to base's on every
//! parent (the set of newly-seen elements per BFS level is independent
//! of probe order, so reordering probes within a level cannot change
//! the orbit closure or the ascending-rep minima scan).
//!
//! Build & run (portable x86_64 / aarch64), from /workspace/src:
//!   cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     --bin orbit_probe -- --inputs scripts/bench-results/sigma-inputs
//!   ... -- --poles            # diagnosis sweep only
//!   ... -- --filter n26-k3    # subset of parents
//!   ... -- --stats            # per-parent cross-parent table
//! Pin it: `taskset -c 4 ...`.

use doubly_even_core::orbit::{m4r_build, mat_apply, orbit_minima_m4r, orbit_minima_walk, singular_reps_q};
use microbench::timing::{cycles_to_ns, mono_cycles, ns_per_cycle};
use microbench::XorShift64;
use std::env;
use std::hint::black_box;

// ----- bitset (one u64 word per probe, same shape as FixedBitSet) for
// ----- the LOCAL experimental variants; production arms use the core
// ----- bodies (FixedBitSet) directly.

struct BitSet(Vec<u64>);

impl BitSet {
    fn with_capacity(bits: usize) -> Self {
        BitSet(vec![0u64; bits.div_ceil(64)])
    }
    /// Single read-modify-write: set bit i, return whether it was set.
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

// ----- BFS variants (minima provably identical; see module doc)

/// base with the contains+insert pair replaced by one `put`.
fn bfs_put(reps_sorted: &[u64], gens: &[&Vec<u64>], l: u32) -> Vec<u64> {
    let universe = 1usize << l;
    let mut seen = BitSet::with_capacity(universe);
    let mut minima: Vec<u64> = Vec::new();
    let cap = reps_sorted.len();
    let mut queue: Vec<u64> = Vec::with_capacity(cap);
    let mut next: Vec<u64> = Vec::with_capacity(cap);
    for &v in reps_sorted {
        if seen.put(v as usize) {
            continue;
        }
        minima.push(v);
        queue.clear();
        queue.push(v);
        while !queue.is_empty() {
            next.clear();
            for &current in &queue {
                for g in gens {
                    let new_v = mat_apply(g, current);
                    if !seen.put(new_v as usize) {
                        next.push(new_v);
                    }
                }
            }
            std::mem::swap(&mut queue, &mut next);
        }
    }
    minima
}

/// Two passes per level chunk: compute `chunk × gens` images into a
/// flat buffer (independent XOR walks, sequential writes), then probe
/// them (independent RMWs the OoO core can overlap).
fn bfs_batch(reps_sorted: &[u64], gens: &[&Vec<u64>], l: u32, chunk: usize) -> Vec<u64> {
    let universe = 1usize << l;
    let mut seen = BitSet::with_capacity(universe);
    let mut minima: Vec<u64> = Vec::new();
    let cap = reps_sorted.len();
    let mut queue: Vec<u64> = Vec::with_capacity(cap);
    let mut next: Vec<u64> = Vec::with_capacity(cap);
    let mut images: Vec<u64> = Vec::with_capacity(chunk * gens.len());
    for &v in reps_sorted {
        if seen.put(v as usize) {
            continue;
        }
        minima.push(v);
        queue.clear();
        queue.push(v);
        while !queue.is_empty() {
            next.clear();
            for cur_chunk in queue.chunks(chunk) {
                images.clear();
                for &current in cur_chunk {
                    for g in gens {
                        images.push(mat_apply(g, current));
                    }
                }
                for &new_v in &images {
                    if !seen.put(new_v as usize) {
                        next.push(new_v);
                    }
                }
            }
            std::mem::swap(&mut queue, &mut next);
        }
    }
    minima
}

/// Generalised m4r with a runtime chunk width `t` ∈ {4, 8, 16} — the
/// classic M4R locality trade: per-chunk table is `2^t` u64s (t=4:
/// 128 B, t=8: 2 KB, t=16: 512 KB — past L1), images cost
/// `ceil(L/t)` lookups. Dynamic tables for all widths so the
/// comparison isolates t, not the container.
fn bfs_m4r_t(reps_sorted: &[u64], gens: &[&Vec<u64>], l: u32, chunk: usize, t: u32) -> Vec<u64> {
    let n_chunks = (l as usize).div_ceil(t as usize);
    let tsize = 1usize << t;
    let tmask = (tsize - 1) as u64;
    let tables: Vec<Vec<u64>> = gens
        .iter()
        .map(|g| {
            let m: &[u64] = g.as_slice();
            let mut tab = vec![0u64; n_chunks * tsize];
            for c in 0..n_chunks {
                let base = c * t as usize;
                let width = (l as usize - base).min(t as usize);
                let tc = &mut tab[c * tsize..(c + 1) * tsize];
                for b in 1usize..1 << width {
                    tc[b] = tc[b & (b - 1)] ^ m[base + b.trailing_zeros() as usize];
                }
            }
            tab
        })
        .collect();
    let apply = |tab: &[u64], x: u64| -> u64 {
        let mut out = 0u64;
        for c in 0..n_chunks {
            out ^= tab[c * tsize + ((x >> (c as u32 * t)) & tmask) as usize];
        }
        out
    };
    let universe = 1usize << l;
    let mut seen = BitSet::with_capacity(universe);
    let mut minima: Vec<u64> = Vec::new();
    let cap = reps_sorted.len();
    let mut queue: Vec<u64> = Vec::with_capacity(cap);
    let mut next: Vec<u64> = Vec::with_capacity(cap);
    for &v in reps_sorted {
        if seen.put(v as usize) {
            continue;
        }
        minima.push(v);
        queue.clear();
        queue.push(v);
        while !queue.is_empty() {
            next.clear();
            for cur_chunk in queue.chunks(chunk) {
                for tab in &tables {
                    for &current in cur_chunk {
                        let new_v = apply(tab, current);
                        if !seen.put(new_v as usize) {
                            next.push(new_v);
                        }
                    }
                }
            }
            std::mem::swap(&mut queue, &mut next);
        }
    }
    minima
}

/// Exact, cheap generator reduction: drop duplicates and inverse-pairs.
/// Sound because a finite group contains every element's inverse —
/// `⟨S⟩ = ⟨S ∪ {g⁻¹}⟩` — so removing `g_j == g_i⁻¹` (or `g_j == g_i`)
/// never changes the generated group or its orbits.
fn reduce_gens(gens: Vec<Vec<u64>>) -> Vec<Vec<u64>> {
    let mut kept: Vec<Vec<u64>> = Vec::with_capacity(gens.len());
    for g in gens {
        let dup = kept.iter().any(|h| *h == g);
        if dup {
            continue;
        }
        let inv_of_kept = kept.iter().any(|h| {
            // h ∘ g == identity? Column form: (h∘g)[i] = mat_apply(h, g[i]).
            g.iter()
                .enumerate()
                .all(|(i, &col)| mat_apply(h, col) == 1u64 << i)
        });
        if inv_of_kept {
            continue;
        }
        kept.push(g);
    }
    kept
}

/// batch + radix-bucket the images by their top bits before probing, so
/// consecutive probes land on nearby bitset pages (TLB locality).
fn bfs_bucket(reps_sorted: &[u64], gens: &[&Vec<u64>], l: u32, chunk: usize) -> Vec<u64> {
    const NBUCKETS: usize = 64;
    let universe = 1usize << l;
    let shift = l.saturating_sub(6); // top 6 bits
    let mut seen = BitSet::with_capacity(universe);
    let mut minima: Vec<u64> = Vec::new();
    let cap = reps_sorted.len();
    let mut queue: Vec<u64> = Vec::with_capacity(cap);
    let mut next: Vec<u64> = Vec::with_capacity(cap);
    let mut buckets: Vec<Vec<u64>> = (0..NBUCKETS).map(|_| Vec::new()).collect();
    for &v in reps_sorted {
        if seen.put(v as usize) {
            continue;
        }
        minima.push(v);
        queue.clear();
        queue.push(v);
        while !queue.is_empty() {
            next.clear();
            for cur_chunk in queue.chunks(chunk) {
                for b in buckets.iter_mut() {
                    b.clear();
                }
                for &current in cur_chunk {
                    for g in gens {
                        let img = mat_apply(g, current);
                        buckets[(img >> shift) as usize & (NBUCKETS - 1)].push(img);
                    }
                }
                for b in &buckets {
                    for &new_v in b {
                        if !seen.put(new_v as usize) {
                            next.push(new_v);
                        }
                    }
                }
            }
            std::mem::swap(&mut queue, &mut next);
        }
    }
    minima
}

// ----- dump-file loading

struct ParentInput {
    name: String,
    n: u32,
    k: u32,
    l: u32,
    v_basis: Vec<u64>,
    gens: Vec<Vec<u64>>, // identity gens already filtered
    gens_total: usize,
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
    let gens_total = gens.len();
    gens.retain(|g| !is_identity_mat(g));
    ParentInput {
        name: path.file_stem().unwrap().to_string_lossy().into_owned(),
        n,
        k,
        l,
        v_basis,
        gens,
        gens_total,
    }
}

// ----- poles diagnosis: latency vs MLP on a bitset of 2^L bits

fn poles(min_l: u32, max_l: u32) {
    println!("# poles: dependent-chase vs independent-probe ns/probe on a 2^L bitset");
    println!(
        "{:>3} {:>10} {:>12} {:>12} {:>8}",
        "L", "bitset_B", "dep_ns/p", "indep_ns/p", "MLP"
    );
    let mut rng = XorShift64::new(0xb0b);
    for l in min_l..=max_l {
        let words = (1usize << l) / 64;
        let mut bits: Vec<u64> = (0..words).map(|_| rng.next()).collect();
        let mask = (1u64 << l) - 1;
        let probes = 1u64 << 22;

        // Dependent chase: next index derived from the loaded word.
        let mut idx: u64 = rng.next() & mask;
        let c0 = mono_cycles();
        let mut acc = 0u64;
        for _ in 0..probes {
            let w = bits[(idx >> 6) as usize];
            acc ^= w;
            idx = (idx.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ w) & mask;
        }
        black_box(acc);
        let dep_cyc = mono_cycles().wrapping_sub(c0);

        // Independent probes: indices precomputed, RMW like the BFS.
        let idxs: Vec<u64> = (0..probes).map(|_| rng.next() & mask).collect();
        let c0 = mono_cycles();
        for &i in &idxs {
            let w = &mut bits[(i >> 6) as usize];
            *w |= 1 << (i & 63);
        }
        black_box(bits[0]);
        let ind_cyc = mono_cycles().wrapping_sub(c0);

        let dep_ns = cycles_to_ns(dep_cyc) / probes as f64;
        let ind_ns = cycles_to_ns(ind_cyc) / probes as f64;
        println!(
            "{:>3} {:>10} {:>12.2} {:>12.2} {:>8.1}",
            l,
            (1u64 << l) / 8,
            dep_ns,
            ind_ns,
            dep_ns / ind_ns
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

fn arg_usize(name: &str, default: usize) -> usize {
    arg_str(name, &default.to_string()).parse().unwrap_or(default)
}

fn main() {
    println!("# orbit_probe: real-input sigma_Q orbit-min BFS replay");
    println!("# ns_per_cycle = {:.4}", ns_per_cycle());

    if flag("--poles") {
        poles(16, 27);
        return;
    }

    let dir = arg_str("--inputs", "scripts/bench-results/sigma-inputs");
    let filter = arg_str("--filter", "");
    let chunk = arg_usize("--batch-chunk", 1024);
    let show_stats = flag("--stats");

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {dir}: {e} — run the kernel dump test first"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .filter(|p| filter.is_empty() || p.to_string_lossy().contains(&filter))
        .collect();
    files.sort();
    println!("# {} parents from {dir} (filter: {:?})", files.len(), filter);

    if show_stats {
        println!(
            "{:<16} {:>2} {:>2} {:>5} {:>5} {:>9} {:>8} {:>10} {:>9} {:>9}",
            "parent", "k", "L", "gtot", "gnid", "reps", "minima", "probes", "bfs_ms", "p/rep"
        );
    }

    // (n, k) -> [base, put, batch, bucket, m4r, m4r_red, t4, t8dyn, t16,
    // gray, calls] cycles
    let mut agg: std::collections::BTreeMap<(u32, u32), [u64; 11]> = Default::default();
    let mut gens_in_total = 0usize;
    let mut gens_red_total = 0usize;

    for path in &files {
        let p = parse_dump(path);
        let gens: Vec<&Vec<u64>> = p.gens.iter().collect();
        let reduced = reduce_gens(p.gens.clone());
        let gens_red: Vec<&Vec<u64>> = reduced.iter().collect();
        gens_in_total += gens.len();
        gens_red_total += gens_red.len();

        let c0 = mono_cycles();
        let reps = singular_reps_q(&p.v_basis);
        let gray_cyc = mono_cycles().wrapping_sub(c0);
        let mut reps_sorted = reps.clone();
        reps_sorted.sort_unstable();

        // base — production walk body (reps pre-sorted, identity gens
        // pre-filtered, matching the production entry's contract).
        let c0 = mono_cycles();
        let minima_base = orbit_minima_walk(&reps_sorted, &gens, p.l);
        let base_cyc = mono_cycles().wrapping_sub(c0);
        // The singular rep set is ⟨gens⟩-closed (Aut(C) preserves
        // wt mod 4 on lifts), so the BFS dequeues each rep exactly once
        // and probes it once per generator: probes = |reps| · |gens|.
        // (The retired instrumented clone counted exactly this.)
        let probes = reps_sorted.len() as u64 * gens.len() as u64;

        // put
        let c0 = mono_cycles();
        let minima_put = bfs_put(&reps_sorted, &gens, p.l);
        let put_cyc = mono_cycles().wrapping_sub(c0);
        assert_eq!(minima_put, minima_base, "put minima diverge on {}", p.name);

        // batch
        let c0 = mono_cycles();
        let minima_batch = bfs_batch(&reps_sorted, &gens, p.l, chunk);
        let batch_cyc = mono_cycles().wrapping_sub(c0);
        assert_eq!(minima_batch, minima_base, "batch minima diverge on {}", p.name);

        // bucket
        let c0 = mono_cycles();
        let minima_bucket = bfs_bucket(&reps_sorted, &gens, p.l, chunk);
        let bucket_cyc = mono_cycles().wrapping_sub(c0);
        assert_eq!(minima_bucket, minima_base, "bucket minima diverge on {}", p.name);

        // m4r (full gen set) — production D18 body, table build included
        // in the timed region (as the production entry pays it per call).
        let c0 = mono_cycles();
        let tables = m4r_build(&gens, p.l);
        let minima_m4r = orbit_minima_m4r(&reps_sorted, &tables, p.l);
        let m4r_cyc = mono_cycles().wrapping_sub(c0);
        assert_eq!(minima_m4r, minima_base, "m4r minima diverge on {}", p.name);

        // m4r + dedupe/inverse-reduced gen set (same group, same orbits)
        let c0 = mono_cycles();
        let tables_red = m4r_build(&gens_red, p.l);
        let minima_m4r_red = orbit_minima_m4r(&reps_sorted, &tables_red, p.l);
        let m4r_red_cyc = mono_cycles().wrapping_sub(c0);
        assert_eq!(
            minima_m4r_red, minima_base,
            "reduced-gens minima diverge on {}",
            p.name
        );

        // chunk-width sweep (--tsweep): t = 4 / 8 / 16, dynamic tables
        if flag("--tsweep") {
            for (slot, t) in [(6usize, 4u32), (7, 8), (8, 16)] {
                let c0 = mono_cycles();
                let m = bfs_m4r_t(&reps_sorted, &gens, p.l, chunk, t);
                let cyc = mono_cycles().wrapping_sub(c0);
                assert_eq!(m, minima_base, "t={t} minima diverge on {}", p.name);
                let e = agg.entry((p.n, p.k)).or_default();
                e[slot] += cyc;
            }
        }

        if show_stats {
            println!(
                "{:<16} {:>2} {:>2} {:>5} {:>5} {:>9} {:>8} {:>10} {:>9.2} {:>9.1}",
                p.name,
                p.k,
                p.l,
                p.gens_total,
                gens.len(),
                reps.len(),
                minima_base.len(),
                probes,
                cycles_to_ns(base_cyc) / 1e6,
                probes as f64 / reps.len().max(1) as f64,
            );
        }

        let e = agg.entry((p.n, p.k)).or_default();
        e[0] += base_cyc;
        e[1] += put_cyc;
        e[2] += batch_cyc;
        e[3] += bucket_cyc;
        e[4] += m4r_cyc;
        e[5] += m4r_red_cyc;
        e[9] += gray_cyc;
        e[10] += 1;
    }

    println!();
    println!(
        "# gens: {} loaded (identity-filtered) -> {} after dedupe+inverse-drop",
        gens_in_total, gens_red_total
    );
    println!(
        "{:>3} {:>2} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "N", "k", "calls", "gray_ms", "base_ms", "put_ms", "batch_ms", "buckt_ms",
        "m4r_ms", "m4rR_ms", "put_x", "batch_x", "buckt_x", "m4r_x", "m4rR_x"
    );
    for ((n, k), e) in &agg {
        let base = e[0];
        println!(
            "{:>3} {:>2} {:>6} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>8.2}",
            n,
            k,
            e[10],
            cycles_to_ns(e[9]) / 1e6,
            cycles_to_ns(base) / 1e6,
            cycles_to_ns(e[1]) / 1e6,
            cycles_to_ns(e[2]) / 1e6,
            cycles_to_ns(e[3]) / 1e6,
            cycles_to_ns(e[4]) / 1e6,
            cycles_to_ns(e[5]) / 1e6,
            base as f64 / e[1].max(1) as f64,
            base as f64 / e[2].max(1) as f64,
            base as f64 / e[3].max(1) as f64,
            base as f64 / e[4].max(1) as f64,
            base as f64 / e[5].max(1) as f64,
        );
    }
    if flag("--tsweep") {
        println!();
        println!(
            "# m4r chunk-width sweep (dynamic tables; per-gen table = 2^t x 8B x ceil(L/t)/chunks)"
        );
        println!(
            "{:>3} {:>2} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7}",
            "N", "k", "base_ms", "t4_ms", "t8_ms", "t16_ms", "t4_x", "t8_x", "t16_x"
        );
        for ((n, k), e) in &agg {
            let base = e[0];
            println!(
                "{:>3} {:>2} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>7.2} {:>7.2} {:>7.2}",
                n,
                k,
                cycles_to_ns(base) / 1e6,
                cycles_to_ns(e[6]) / 1e6,
                cycles_to_ns(e[7]) / 1e6,
                cycles_to_ns(e[8]) / 1e6,
                base as f64 / e[6].max(1) as f64,
                base as f64 / e[7].max(1) as f64,
                base as f64 / e[8].max(1) as f64,
            );
        }
    }
}
