//! End-to-end φ-cascade replay across child ranks (post-D15 profile,
//! plan `last-session-we-had-sequential-fiddle.md` Phase 3).
//!
//! **STALE since D16 (2026-06-10)**: this clones the PRE-D16
//! full-frame cascade (Gray sweep + sort + full-size WHT per
//! candidate). The production cascade now shares the C-half per
//! parent and decides ~99 % of first strata via the amax O(1) bound
//! (`parent_rule.rs::PhiParentCtx`), so these numbers bound only the
//! historical kernel. Kept as the record behind the 08-doc §4
//! L1-cliff analysis; re-sync before reusing for new conclusions.
//!
//! Clone of pre-D16 `parent_rule::phi_cascade_with` (filters, counting
//! sort, Gray sweep — was kept in copy-sync by hand) driven by
//! synthetic frames, k+1 = 8..16. Two modes:
//!
//!   hot  — back-to-back cascades, PhiScratch L1/L2-resident (matches
//!          production bursts of candidates against one parent).
//!   cold — 4 MB eviction sweep between cascades (upper-bounds the
//!          interleaving with ~210 KB CanonScratch canon calls).
//!
//! The hot/cold gap per rank BOUNDS what any L1-fit optimisation of the
//! φ working set (wt u8 + f i32 + sorted_idx u16 ≈ 7 B · 2^(k+1)) can
//! possibly recover at that rank. Sub-phase cycle marks mirror the
//! kernel's `phase_timers` φ split so in-vivo sampled splits and this
//! standalone replay are directly comparable.
//!
//! Synthetic-frame caveat: rows are random even-weight words, giving
//! denser strata ladders than true doubly-even codes (whose strata sit
//! at multiples of 4) and near-always first-stratum resolution. The
//! replay is a CACHE-COST model of the dominant path (frame + Gray +
//! sort + first-stratum WHT) — production outcome rates come from the
//! kernel's always-on per-k φ stats, not from here.
//!
//! Build & run (portable x86_64 / aarch64):
//!   cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     --bin phi_replay -- --min-kp1 8 --max-kp1 16
//! Pin it: `taskset -c 4 ...`.

use microbench::timing::{cycles_to_ns, mono_cycles, ns_per_cycle};
use microbench::{evict_l1_l2, XorShift64};
use std::env;
use std::hint::black_box;

const DIRECT_THRESHOLD: usize = 64;
const N_BITS: u32 = 48;

#[derive(Default)]
struct PhiScratch {
    rows: Vec<u64>,
    wt: Vec<u8>,
    f: Vec<i32>,
    sorted_idx: Vec<u16>,
    m_buf: Vec<u16>,
}

enum Outcome {
    Reject,
    AcceptUnique,
    Tie(usize),
}

/// Sub-phase cycle accumulator: [frame+gray, sort, first, wht, direct].
struct Marks {
    t: u64,
    acc: [u64; 5],
}

impl Marks {
    #[inline]
    fn start() -> Self {
        Marks { t: mono_cycles(), acc: [0; 5] }
    }
    #[inline]
    fn mark(&mut self, idx: usize) {
        let now = mono_cycles();
        self.acc[idx] += now.wrapping_sub(self.t);
        self.t = now;
    }
}

fn wht_in_place(f: &mut [i32]) {
    let size = f.len();
    let mut h = 1;
    while h < size {
        let mut i = 0;
        while i < size {
            for j in i..i + h {
                let x = f[j];
                let y = f[j + h];
                f[j] = x + y;
                f[j + h] = x - y;
            }
            i += h << 1;
        }
        h <<= 1;
    }
}

fn fill_indicator(s: &mut PhiScratch, size: usize, t_begin: usize, t_end: usize) {
    s.f.clear();
    s.f.resize(size, 0);
    for i in t_begin..t_end {
        s.f[s.sorted_idx[i] as usize] = 1;
    }
    wht_in_place(&mut s.f);
}

fn filter_first_stratum(s: &mut PhiScratch, size: usize, t_begin: usize, t_end: usize, u_c: u16) -> bool {
    fill_indicator(s, size, t_begin, t_end);
    let mut best = i32::MIN;
    for u in 1..size {
        if s.f[u] > best {
            best = s.f[u];
        }
    }
    s.m_buf.clear();
    let mut u_c_in = false;
    for u in 1..size {
        if s.f[u] == best {
            s.m_buf.push(u as u16);
            u_c_in |= u as u16 == u_c;
        }
    }
    u_c_in
}

fn filter_by_wht(s: &mut PhiScratch, size: usize, t_begin: usize, t_end: usize, u_c: u16) -> bool {
    fill_indicator(s, size, t_begin, t_end);
    let mut best = i32::MIN;
    for &u in &s.m_buf {
        if s.f[u as usize] > best {
            best = s.f[u as usize];
        }
    }
    let f = std::mem::take(&mut s.f);
    s.m_buf.retain(|&u| f[u as usize] == best);
    s.f = f;
    s.m_buf.contains(&u_c)
}

fn filter_direct(s: &mut PhiScratch, t_begin: usize, t_end: usize, u_c: u16) -> bool {
    let members = &s.sorted_idx[t_begin..t_end];
    let mut best = u32::MAX;
    let mut counts: Vec<u32> = Vec::with_capacity(s.m_buf.len());
    for &u in &s.m_buf {
        let mut c = 0u32;
        for &x in members {
            c += ((u & x).count_ones() & 1) as u32;
        }
        counts.push(c);
        if c < best {
            best = c;
        }
    }
    let mut i = 0;
    s.m_buf.retain(|_| {
        let keep = counts[i] == best;
        i += 1;
        keep
    });
    s.m_buf.contains(&u_c)
}

/// `phi_cascade_with` clone with cycle marks.
fn phi_cascade(s: &mut PhiScratch, c_rref: &[u64], v: u64, n: u32, marks: &mut Marks) -> Outcome {
    let kp1 = c_rref.len() + 1;
    let size = 1usize << kp1;
    let u_c: u16 = 1 << (kp1 - 1);

    s.rows.clear();
    s.rows.extend_from_slice(c_rref);
    s.rows.push(v);

    s.wt.clear();
    s.wt.resize(size, 0);
    let mut counts = [0u32; 65];
    counts[0] = 1;
    let mut cur: u64 = 0;
    for i in 1..size {
        let flip = i.trailing_zeros() as usize;
        cur ^= s.rows[flip];
        let w = cur.count_ones() as usize;
        s.wt[i ^ (i >> 1)] = w as u8;
        counts[w] += 1;
    }
    marks.mark(0);

    let mut start = [0u32; 66];
    for w in 0..65 {
        start[w + 1] = start[w] + counts[w];
    }
    s.sorted_idx.clear();
    s.sorted_idx.resize(size, 0);
    let mut cursor = start;
    for idx in 0..size {
        let w = s.wt[idx] as usize;
        s.sorted_idx[cursor[w] as usize] = idx as u16;
        cursor[w] += 1;
    }
    marks.mark(1);

    let mut first = true;
    let n_cap = (n as usize).min(64);
    for w in 1..=n_cap {
        if counts[w] == 0 {
            continue;
        }
        let t_begin = start[w] as usize;
        let t_end = t_begin + counts[w] as usize;
        let u_c_in = if first {
            first = false;
            let r = filter_first_stratum(s, size, t_begin, t_end, u_c);
            marks.mark(2);
            r
        } else if s.m_buf.len() > DIRECT_THRESHOLD {
            let r = filter_by_wht(s, size, t_begin, t_end, u_c);
            marks.mark(3);
            r
        } else {
            let r = filter_direct(s, t_begin, t_end, u_c);
            marks.mark(4);
            r
        };
        if !u_c_in {
            return Outcome::Reject;
        }
        if s.m_buf.len() == 1 {
            return Outcome::AcceptUnique;
        }
    }
    Outcome::Tie(s.m_buf.len())
}

fn arg(name: &str, default: u32) -> u32 {
    let args: Vec<String> = env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Random even-weight word over N_BITS (frames must be independent with
/// overwhelming probability at these dims; evenness keeps strata sane).
fn even_word(rng: &mut XorShift64) -> u64 {
    let mask = (1u64 << N_BITS) - 1;
    let mut w = rng.next() & mask;
    if w.count_ones() & 1 == 1 {
        w ^= 1 << (rng.next() % N_BITS as u64);
    }
    w
}

fn main() {
    let min_kp1 = arg("--min-kp1", 8);
    let max_kp1 = arg("--max-kp1", 16);
    let iters = arg("--iters", 4000) as usize;

    println!("# phi_replay: cloned phi_cascade on synthetic frames, N={N_BITS}");
    println!("# ns_per_cycle = {:.4}", ns_per_cycle());
    println!(
        "{:>4} {:>9} {:>12} {:>12} {:>6} | {:>10} {:>9} {:>10} {:>9} {:>9}",
        "k+1", "ws_bytes", "hot_ns/call", "cold_ns/call", "ratio",
        "gray_ns", "sort_ns", "first_ns", "wht_ns", "direct_ns"
    );

    let mut rng = XorShift64::new(0xf1_d15);
    let mut junk = vec![0u8; 4 << 20];
    let mut s = PhiScratch::default();

    for kp1 in min_kp1..=max_kp1 {
        let k = (kp1 - 1) as usize;
        let size = 1usize << kp1;
        // 7 B per coordinate: wt u8 + f i32 + sorted_idx u16.
        let ws_bytes = size * 7;

        let c_rref: Vec<u64> = (0..k).map(|_| even_word(&mut rng)).collect();
        let vs: Vec<u64> = (0..64).map(|_| even_word(&mut rng)).collect();

        // Warmup.
        let mut m = Marks::start();
        for &v in &vs {
            black_box(phi_cascade(&mut s, &c_rref, v, N_BITS, &mut m));
        }

        // Hot.
        let mut marks = Marks { t: 0, acc: [0; 5] };
        let c0 = mono_cycles();
        for i in 0..iters {
            marks.t = mono_cycles();
            black_box(phi_cascade(&mut s, &c_rref, vs[i % vs.len()], N_BITS, &mut marks));
        }
        let hot_cyc = mono_cycles().wrapping_sub(c0);

        // Cold (eviction cost subtracted).
        let c0 = mono_cycles();
        for _ in 0..iters {
            evict_l1_l2(&mut junk);
            black_box(junk[0]);
        }
        let evict_cyc = mono_cycles().wrapping_sub(c0);
        let mut cold_marks = Marks { t: 0, acc: [0; 5] };
        let c0 = mono_cycles();
        for i in 0..iters {
            evict_l1_l2(&mut junk);
            cold_marks.t = mono_cycles();
            black_box(phi_cascade(&mut s, &c_rref, vs[i % vs.len()], N_BITS, &mut cold_marks));
        }
        let cold_cyc = mono_cycles().wrapping_sub(c0).saturating_sub(evict_cyc);

        let hot_call = cycles_to_ns(hot_cyc) / iters as f64;
        let cold_call = cycles_to_ns(cold_cyc) / iters as f64;
        let ph: Vec<f64> = marks
            .acc
            .iter()
            .map(|&c| cycles_to_ns(c) / iters as f64)
            .collect();
        println!(
            "{:>4} {:>9} {:>12.1} {:>12.1} {:>6.2} | {:>10.1} {:>9.1} {:>10.1} {:>9.1} {:>9.1}",
            kp1, ws_bytes, hot_call, cold_call, cold_call / hot_call,
            ph[0], ph[1], ph[2], ph[3], ph[4]
        );
    }
}
