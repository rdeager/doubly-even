//! φ-cascade replay on synthetic frames — POST-D16/D17 cascade.
//!
//! Drives the PRODUCTION split-frame cascade directly from
//! `doubly-even-core`: per parent a `parent_rule::PhiParentSlot`, per
//! candidate `parent_rule::phi_cascade_shared` (eager C-half tables;
//! lazy per-stratum WHTs `F̂_C^{(w)}`, argmax sets `E_w`, amax bounds;
//! D17 E-chain), exactly as `enumerate.rs::test_candidate` runs it. The
//! hand-copied clone this bin carried before the workspace restructure
//! is retired (git history has it) — the arm under test IS production
//! code.
//!
//! Per-phase rows come from the production sampled `PhaseClock`
//! (`phase_timers` feature, enabled by this crate's core dependency):
//! the marks pass forces a sample per call via
//! `phi_sample::force_next_sample()` and drains it with `take_last()`,
//! so the phase indices ARE the production ones and rows stay
//! column-comparable with the kernel's sampled splits (stats 39–43):
//!   [0] v-half weights+histogram   [1] s1 indicator fill
//!   [2] s1 decision (fastpaths / Ĝ_v WHT + fused scan)
//!   [3] later-stratum WHT          [4] direct parity + chain arms
//! The timing passes do NOT force samples — they pay exactly the
//! production sampling cost (1-in-64) and nothing more.
//!
//! Frame generators (local, synthetic):
//!   --mode mixed  random even-weight rows + candidates (default;
//!                 exercises amax + general first strata — the chain
//!                 under-fires here, as in any unstructured frame)
//!   --mode conly  low-weight rows (wt 8) + heavy candidates (wt ≥ 20)
//!                 so the lowest strata are C-only and the D17 chain
//!                 arms fire (the production-dominant accept shape)
//!
//! --validate runs a brute-force full-frame spectrum argmin per
//! candidate (kp1 ≤ 10) and asserts decision equality — validating the
//! REAL production cascade against the local oracle.
//!
//! Run (from /workspace/src):
//!   cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     --bin phi_replay -- --min-kp1 8 --max-kp1 16 [--mode conly]
//!     [--validate] [--n 48] [--parents 8] [--cands 64]
//! Pin it: `taskset -c 4 ...`.

use doubly_even_core::parent_rule::{
    phi_cascade_shared, phi_sample, PhiOutcome, PhiParentSlot,
};
use microbench::timing::{cycles_to_ns, mono_cycles, ns_per_cycle};
use microbench::{evict_l1_l2, XorShift64};
use std::env;
use std::hint::black_box;

// ───────────────────────────── brute-force reference (--validate)

#[derive(PartialEq, Debug, Clone)]
enum Outcome {
    Reject,
    AcceptUnique,
    Tie(Vec<u16>),
}

fn outcome_of(o: &PhiOutcome) -> Outcome {
    match o {
        PhiOutcome::Reject => Outcome::Reject,
        PhiOutcome::AcceptUnique => Outcome::AcceptUnique,
        PhiOutcome::Tie(m) => Outcome::Tie(m.clone()),
    }
}

/// Full-frame spectrum argmin, straight from the definition: for every
/// nonzero functional u over the (k+1)-frame `[C-rows, v]`, the spectrum
/// `φ_w(u) = #{x : wt(word(x)) = w, u·x odd}`, lex-compared ascending.
fn reference_decision(c_rref: &[u64], v: u64, n: u32) -> Outcome {
    let kp1 = c_rref.len() + 1;
    let size = 1usize << kp1;
    let mut rows = c_rref.to_vec();
    rows.push(v);
    let mut words = vec![0u64; size];
    for x in 1..size {
        let flip = x.trailing_zeros() as usize;
        // x & (x - 1) clears the lowest set bit; that index is < x, so
        // its word is already filled.
        words[x] = words[x & (x - 1)] ^ rows[flip];
    }
    let n_cap = (n as usize).min(64);
    let mut specs: Vec<Vec<u32>> = Vec::with_capacity(size - 1);
    for u in 1..size {
        let mut spec = vec![0u32; n_cap];
        for (x, &word) in words.iter().enumerate() {
            if (u & x).count_ones() & 1 == 1 {
                let w = word.count_ones() as usize;
                if (1..=n_cap).contains(&w) {
                    spec[w - 1] += 1;
                }
            }
        }
        specs.push(spec);
    }
    let min_spec = specs.iter().min().expect("nonempty").clone();
    let argmin: Vec<u16> = (1..size)
        .filter(|&u| specs[u - 1] == min_spec)
        .map(|u| u as u16)
        .collect();
    let u_c = 1u16 << (kp1 - 1);
    if !argmin.contains(&u_c) {
        Outcome::Reject
    } else if argmin.len() == 1 {
        Outcome::AcceptUnique
    } else {
        Outcome::Tie(argmin)
    }
}

// ───────────────────────────── frame generators

fn random_word_weight(rng: &mut XorShift64, n: u32, target_wt: u32) -> u64 {
    let mut w = 0u64;
    while w.count_ones() < target_wt {
        w |= 1u64 << (rng.next() % n as u64);
    }
    w
}

/// True iff `w` is outside span(rows) — eliminate against the rows.
fn independent(rows: &[u64], w: u64) -> bool {
    let mut basis: Vec<u64> = rows.to_vec();
    basis.sort_unstable_by_key(|r| std::cmp::Reverse(*r));
    let mut x = w;
    for r in &basis {
        if *r == 0 {
            continue;
        }
        let hb = 63 - r.leading_zeros();
        if x >> hb & 1 == 1 {
            x ^= r;
        }
    }
    x != 0
}

/// `mixed`: random even-weight rows (old replay's scheme).
/// `conly`: weight-8 rows so the low strata are C-populated.
fn gen_frame(rng: &mut XorShift64, n: u32, k: usize, conly: bool) -> Vec<u64> {
    let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    let mut rows: Vec<u64> = Vec::with_capacity(k);
    while rows.len() < k {
        let mut r = if conly {
            random_word_weight(rng, n, 8)
        } else {
            rng.next() & mask
        };
        if r.count_ones() & 1 == 1 {
            r ^= r & r.wrapping_neg(); // clear lowest set bit → even weight
        }
        if r == 0 || !independent(&rows, r) {
            continue;
        }
        rows.push(r);
    }
    rows
}

fn gen_candidate(rng: &mut XorShift64, n: u32, rows: &[u64], conly: bool) -> u64 {
    let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    loop {
        let mut v = if conly {
            let extra = (rng.next() % 8) as u32;
            random_word_weight(rng, n, 20 + extra)
        } else {
            rng.next() & mask
        };
        if v.count_ones() & 1 == 1 {
            v ^= v & v.wrapping_neg();
        }
        if v != 0 && independent(rows, v) {
            return v;
        }
    }
}

// ───────────────────────────── harness

fn arg(name: &str, default: u64) -> u64 {
    let args: Vec<String> = env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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
    let min_kp1 = arg("--min-kp1", 8) as usize;
    let max_kp1 = arg("--max-kp1", 16) as usize;
    let n = arg("--n", 48) as u32;
    let n_parents = arg("--parents", 8) as usize;
    let n_cands = arg("--cands", 64) as usize;
    let validate = flag("--validate");
    let conly = arg_str("--mode", "mixed") == "conly";

    println!("# phi_replay (post-D16/D17 split-frame + chain cascade)");
    println!(
        "# ns_per_cycle = {:.4}, N = {n}, mode = {}, parents = {n_parents}, cands/parent = {n_cands}",
        ns_per_cycle(),
        if conly { "conly" } else { "mixed" }
    );
    println!(
        "{:>4} {:>9} {:>9} {:>9} {:>6} {:>6} {:>6} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "k+1", "ws_KB", "hot_ns", "cold_ns", "rej%", "acc%", "tie%", "s1f%", "chain%",
        "ph0%", "ph1%", "ph2%", "ph3%", "ph4%"
    );

    let mut junk = vec![0u8; 4 << 20];

    for kp1 in min_kp1..=max_kp1 {
        let k = kp1 - 1;
        let h = 1usize << k;
        let mut rng = XorShift64::new(0x00D1_6D17 ^ ((kp1 as u64) << 32));
        let frames: Vec<Vec<u64>> = (0..n_parents)
            .map(|_| gen_frame(&mut rng, n, k, conly))
            .collect();
        let cands: Vec<Vec<u64>> = frames
            .iter()
            .map(|rows| {
                (0..n_cands)
                    .map(|_| gen_candidate(&mut rng, n, rows, conly))
                    .collect()
            })
            .collect();

        if validate && kp1 <= 10 {
            for (rows, cs) in frames.iter().zip(cands.iter()) {
                let mut slot = PhiParentSlot::new();
                for &v in cs {
                    let got = phi_cascade_shared(&mut slot, rows, v, n);
                    let want = reference_decision(rows, v, n);
                    assert_eq!(
                        outcome_of(&got.outcome),
                        want,
                        "cascade != reference at kp1={kp1}"
                    );
                }
            }
        }

        // Timing passes (no forced samples). Repeat to ≥ ~10 ms per row.
        // One fresh slot per parent per sweep, so the ctx builds once per
        // parent per iteration — same accounting as the retired clone's
        // explicit `ctx.build` (boxes recycle through the core pool).
        let iters = ((1u64 << 24) / (h as u64 * (n_parents * n_cands) as u64).max(1)).max(1);

        let c0 = mono_cycles();
        for _ in 0..iters {
            for (rows, cs) in frames.iter().zip(cands.iter()) {
                let mut slot = PhiParentSlot::new();
                for &v in cs {
                    black_box(
                        phi_cascade_shared(&mut slot, rows, black_box(v), n).chain_fastpath,
                    );
                }
            }
        }
        let hot_cyc = mono_cycles().wrapping_sub(c0) / (iters * (n_parents * n_cands) as u64);

        let cold_iters = iters.min(4);
        let c0 = mono_cycles();
        for _ in 0..cold_iters {
            for (rows, cs) in frames.iter().zip(cands.iter()) {
                let mut slot = PhiParentSlot::new();
                for &v in cs {
                    evict_l1_l2(&mut junk);
                    black_box(
                        phi_cascade_shared(&mut slot, rows, black_box(v), n).chain_fastpath,
                    );
                }
            }
        }
        let mut cold_cyc =
            mono_cycles().wrapping_sub(c0) / (cold_iters * (n_parents * n_cands) as u64);
        let c0 = mono_cycles();
        for _ in 0..32 {
            evict_l1_l2(&mut junk);
        }
        let evict_cyc = mono_cycles().wrapping_sub(c0) / 32;
        cold_cyc = cold_cyc.saturating_sub(evict_cyc);

        // Marks + outcome pass: every call force-sampled through the
        // production PhaseClock, drained via phi_sample::take_last.
        let mut acc = [0u64; phi_sample::N_PHASES];
        let mut outc = [0u64; 3]; // rej, acc, tie
        let mut s1f = 0u64;
        let mut chainf = 0u64;
        for (rows, cs) in frames.iter().zip(cands.iter()) {
            let mut slot = PhiParentSlot::new();
            for &v in cs {
                phi_sample::force_next_sample();
                let r = phi_cascade_shared(&mut slot, rows, v, n);
                let ph = phi_sample::take_last().expect("forced sample not recorded");
                for (a, m) in acc.iter_mut().zip(ph.iter()) {
                    *a += m;
                }
                match &r.outcome {
                    PhiOutcome::Reject => outc[0] += 1,
                    PhiOutcome::AcceptUnique => outc[1] += 1,
                    PhiOutcome::Tie(_) => outc[2] += 1,
                }
                s1f += r.s1_fastpath as u64;
                chainf += r.chain_fastpath as u64;
            }
        }
        let total_cands = (n_parents * n_cands) as f64;
        let acc_total = acc.iter().sum::<u64>().max(1) as f64;
        // Eager ctx working set per coord: cwords 8B + wt_c 1B +
        // sorted_c 2B, plus scratch wt_v 1B + g 4B.
        let ws = h * (8 + 1 + 2 + 1 + 4);
        println!(
            "{:>4} {:>9.1} {:>9.1} {:>9.1} {:>6.1} {:>6.1} {:>6.1} {:>5.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1}",
            kp1,
            ws as f64 / 1024.0,
            cycles_to_ns(hot_cyc),
            cycles_to_ns(cold_cyc),
            100.0 * outc[0] as f64 / total_cands,
            100.0 * outc[1] as f64 / total_cands,
            100.0 * outc[2] as f64 / total_cands,
            100.0 * s1f as f64 / total_cands,
            100.0 * chainf as f64 / total_cands,
            100.0 * acc[0] as f64 / acc_total,
            100.0 * acc[1] as f64 / acc_total,
            100.0 * acc[2] as f64 / acc_total,
            100.0 * acc[3] as f64 / acc_total,
            100.0 * acc[4] as f64 / acc_total,
        );
    }
    if validate {
        println!("# validate: cascade decisions == brute-force reference (kp1 <= 10) — OK");
    }
}
