//! φ-cascade replay on synthetic frames — POST-D16/D17 cascade
//! (re-synced 2026-06-10; the pre-D16 full-frame version this file used
//! to hold is in git history).
//!
//! Kernel-free, portable (x86_64 / aarch64) iteration harness for the
//! SIMD sprint: clones the production split-frame cascade from
//! `rust/src/parent_rule.rs` — per-parent `PhiParentCtx` (eager C-half
//! tables; lazy per-stratum WHTs `F̂_C^{(w)}`, argmax sets `E_w`, amax
//! bounds; D17 E-chain `chain_e`/`chain_bound`) shared across sibling
//! candidates, per-candidate `PhiScratch`, the amax / chain O(1)
//! rejects, and the WHT / direct later-stratum arms. Phase mark indices
//! mirror the production `PhaseClock` exactly, so rows here are
//! column-comparable with the kernel's sampled splits (stats 39–43):
//!   [0] v-half weights+histogram   [1] s1 indicator fill
//!   [2] s1 decision (fastpaths / Ĝ_v WHT + fused scan)
//!   [3] later-stratum WHT          [4] direct parity + chain arms
//!
//! Clones are hand-copied per the established microbench pattern
//! (`src/lib.rs` — the kernel crate can't be linked here). Strip-downs
//! vs production: no ctx pool / build-ns accounting, no sampling (the
//! marks pass is separate from the timing pass so totals stay clean).
//!
//! Frame generators:
//!   --mode mixed  random even-weight rows + candidates (default;
//!                 exercises amax + general first strata — the chain
//!                 under-fires here, as in any unstructured frame)
//!   --mode conly  low-weight rows (wt 8) + heavy candidates (wt ≥ 20)
//!                 so the lowest strata are C-only and the D17 chain
//!                 arms fire (the production-dominant accept shape)
//!
//! --validate runs a brute-force full-frame spectrum argmin per
//! candidate (kp1 ≤ 10) and asserts decision equality — the same
//! reference shape as the kernel's `check_against_reference` unit test.
//!
//! Run (from /workspace/src):
//!   cargo run --release --manifest-path scripts/microbench/Cargo.toml \
//!     --bin phi_replay -- --min-kp1 8 --max-kp1 16 [--mode conly]
//!     [--validate] [--n 48] [--parents 8] [--cands 64]
//! Pin it: `taskset -c 4 ...`.

use microbench::timing::{cycles_to_ns, mono_cycles, ns_per_cycle};
use microbench::{evict_l1_l2, XorShift64};
use std::env;
use std::hint::black_box;

const DIRECT_THRESHOLD: usize = 64;

// ───────────────────────────── production clones (parent_rule.rs)

#[derive(PartialEq, Debug, Clone)]
enum Outcome {
    Reject,
    AcceptUnique,
    Tie(Vec<u16>),
}

struct ReplayResult {
    outcome: Outcome,
    s1_fastpath: bool,
    chain_fastpath: bool,
}

/// Per-parent shared ctx — `PhiParentCtx` minus pool/stats plumbing.
struct Ctx {
    kp1: usize,
    cwords: Vec<u64>,
    wt_c: Vec<u8>,
    counts_c: [u32; 65],
    start_c: [u32; 66],
    sorted_c: Vec<u16>,
    fhat_c: Vec<Vec<i32>>,
    e_set: Vec<Vec<u16>>,
    amax: Vec<i32>,
    fhat_built: u128,
    chain_e: Vec<Vec<u16>>,
    chain_bound: Vec<i32>,
    chain_w: Vec<u8>,
    chain_len: usize,
}

impl Default for Ctx {
    fn default() -> Self {
        Self {
            kp1: 0,
            cwords: Vec::new(),
            wt_c: Vec::new(),
            counts_c: [0; 65],
            start_c: [0; 66],
            sorted_c: Vec::new(),
            fhat_c: Vec::new(),
            e_set: Vec::new(),
            amax: Vec::new(),
            fhat_built: 0,
            chain_e: Vec::new(),
            chain_bound: Vec::new(),
            chain_w: Vec::new(),
            chain_len: 0,
        }
    }
}

impl Ctx {
    fn build(&mut self, c_rref: &[u64]) {
        let kp1 = c_rref.len() + 1;
        assert!(kp1 <= 16, "u16 coordinate vectors need k+1 <= 16");
        let h = 1usize << (kp1 - 1);
        self.kp1 = kp1;
        self.cwords.clear();
        self.cwords.resize(h, 0);
        let mut cur: u64 = 0;
        for i in 1..h {
            let flip = i.trailing_zeros() as usize;
            cur ^= c_rref[flip];
            self.cwords[i ^ (i >> 1)] = cur;
        }
        self.wt_c.clear();
        self.wt_c.resize(h, 0);
        for x in 0..h {
            self.wt_c[x] = self.cwords[x].count_ones() as u8;
        }
        self.counts_c = [0u32; 65];
        for x in 0..h {
            self.counts_c[self.wt_c[x] as usize] += 1;
        }
        assert_eq!(self.counts_c[0], 1, "parent rows must be independent");
        self.start_c[0] = 0;
        for w in 0..65 {
            self.start_c[w + 1] = self.start_c[w] + self.counts_c[w];
        }
        self.sorted_c.clear();
        self.sorted_c.resize(h, 0);
        let mut cursor = self.start_c;
        for x in 0..h {
            let w = self.wt_c[x] as usize;
            self.sorted_c[cursor[w] as usize] = x as u16;
            cursor[w] += 1;
        }
        self.fhat_built = 0;
        self.chain_len = 0;
        if self.fhat_c.is_empty() {
            self.fhat_c = vec![Vec::new(); 65];
            self.e_set = vec![Vec::new(); 65];
            self.amax = vec![i32::MIN; 65];
        }
    }

    fn ensure_fhat(&mut self, w: usize) {
        if self.fhat_built >> w & 1 == 1 {
            return;
        }
        let h = 1usize << (self.kp1 - 1);
        let f = &mut self.fhat_c[w];
        f.clear();
        f.resize(h, 0);
        let b = self.start_c[w] as usize;
        let e = b + self.counts_c[w] as usize;
        for &x in &self.sorted_c[b..e] {
            f[x as usize] = 1;
        }
        wht_in_place(f);
        let es = &mut self.e_set[w];
        es.clear();
        let tc = self.counts_c[w] as i32;
        let mut amax = i32::MIN;
        for (u, &fu) in f.iter().enumerate().take(h).skip(1) {
            amax = amax.max(fu);
            if tc > 0 && fu == tc {
                es.push(u as u16);
            }
        }
        self.amax[w] = amax;
        self.fhat_built |= 1 << w;
    }

    fn ensure_chain(&mut self, j: usize, w: usize) {
        if j < self.chain_len {
            debug_assert_eq!(self.chain_w[j], w as u8);
            return;
        }
        debug_assert_eq!(j, self.chain_len);
        self.ensure_fhat(w);
        let mut out = if self.chain_e.len() > j {
            std::mem::take(&mut self.chain_e[j])
        } else {
            Vec::new()
        };
        out.clear();
        let mut bound = i32::MIN;
        if j == 0 {
            out.extend_from_slice(&self.e_set[w]);
        } else {
            let fc = self.fhat_c[w].as_slice();
            let tc = self.counts_c[w] as i32;
            for &u in &self.chain_e[j - 1] {
                let f = fc[u as usize];
                bound = bound.max(f);
                if f == tc {
                    out.push(u);
                }
            }
        }
        if self.chain_e.len() > j {
            self.chain_e[j] = out;
            self.chain_bound[j] = bound;
            self.chain_w[j] = w as u8;
        } else {
            self.chain_e.push(out);
            self.chain_bound.push(bound);
            self.chain_w.push(w as u8);
        }
        self.chain_len = j + 1;
    }
}

struct Scratch {
    wt_v: Vec<u8>,
    g: Vec<i32>,
    sorted_v: Vec<u16>,
    start_v: [u32; 66],
    sorted_v_built: bool,
    counts_buf: Vec<u32>,
    m_buf: Vec<u16>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            wt_v: Vec::new(),
            g: Vec::new(),
            sorted_v: Vec::new(),
            start_v: [0; 66],
            sorted_v_built: false,
            counts_buf: Vec::new(),
            m_buf: Vec::new(),
        }
    }
}

impl Scratch {
    fn ensure_sorted_v(&mut self, h: usize, counts_v: &[u32; 65]) {
        if self.sorted_v_built {
            return;
        }
        self.start_v[0] = 0;
        for w in 0..65 {
            self.start_v[w + 1] = self.start_v[w] + counts_v[w];
        }
        self.sorted_v.clear();
        self.sorted_v.resize(h, 0);
        let mut cursor = self.start_v;
        for (x, &wt) in self.wt_v.iter().enumerate().take(h) {
            let w = wt as usize;
            self.sorted_v[cursor[w] as usize] = x as u16;
            cursor[w] += 1;
        }
        self.sorted_v_built = true;
    }
}

/// Always-on cycle marks, same indices as production `PhaseClock`.
/// `enabled = false` is a well-predicted branch (used by the timing
/// pass so totals aren't distorted by rdtsc pairs).
struct Marks {
    enabled: bool,
    t: u64,
    acc: [u64; 5],
}

impl Marks {
    fn start(enabled: bool) -> Self {
        Self {
            enabled,
            t: if enabled { mono_cycles() } else { 0 },
            acc: [0; 5],
        }
    }
    #[inline]
    fn mark(&mut self, idx: usize) {
        if self.enabled {
            let now = mono_cycles();
            self.acc[idx] += now.wrapping_sub(self.t);
            self.t = now;
        }
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

/// `phi_cascade_split` clone (decision logic verbatim; PhaseClock →
/// Marks, PhiResult → ReplayResult).
fn phi_cascade_split(
    s: &mut Scratch,
    ctx: &mut Ctx,
    v: u64,
    n: u32,
    clock: &mut Marks,
) -> ReplayResult {
    let kp1 = ctx.kp1;
    let h = 1usize << (kp1 - 1);
    let u_c: u16 = 1 << (kp1 - 1);

    // Phase 0: v-half weights + 4-way split histogram.
    s.wt_v.clear();
    s.wt_v.resize(h, 0);
    s.sorted_v_built = false;
    for (wt, &cw) in s.wt_v.iter_mut().zip(ctx.cwords.iter()) {
        *wt = (cw ^ v).count_ones() as u8;
    }
    let mut counts4 = [[0u32; 65]; 4];
    let mut chunks = s.wt_v.chunks_exact(4);
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
    debug_assert_eq!(counts_v[0], 0, "candidate v must lie outside C");
    clock.mark(0);

    let mut first = true;
    let mut s1_fastpath = false;
    let mut chain: Option<usize> = None;
    let n_cap = (n as usize).min(64);
    for w in 1..=n_cap {
        let tc = ctx.counts_c[w];
        let tv = counts_v[w];
        if tc + tv == 0 {
            continue;
        }

        let u_c_in = if first {
            first = false;
            if h == 1 {
                clock.mark(2);
                return ReplayResult {
                    outcome: Outcome::AcceptUnique,
                    s1_fastpath: true,
                    chain_fastpath: false,
                };
            }
            if tc == 0 {
                clock.mark(2);
                return ReplayResult {
                    outcome: Outcome::Reject,
                    s1_fastpath: true,
                    chain_fastpath: false,
                };
            }
            if tv == 0 {
                s1_fastpath = true;
                ctx.ensure_fhat(w);
                if ctx.e_set[w].is_empty() {
                    clock.mark(2);
                    return ReplayResult {
                        outcome: Outcome::AcceptUnique,
                        s1_fastpath: true,
                        chain_fastpath: false,
                    };
                }
                ctx.ensure_chain(0, w);
                chain = Some(0);
                clock.mark(2);
                continue;
            }
            ctx.ensure_fhat(w);
            if ctx.amax[w] > tc as i32 - tv as i32 {
                clock.mark(2);
                return ReplayResult {
                    outcome: Outcome::Reject,
                    s1_fastpath: true,
                    chain_fastpath: false,
                };
            }
            first_stratum_split(s, ctx, clock, w, tc, tv, u_c)
        } else if let Some(j) = chain {
            if tc == 0 {
                clock.mark(4);
                return ReplayResult {
                    outcome: Outcome::Reject,
                    s1_fastpath,
                    chain_fastpath: true,
                };
            }
            ctx.ensure_chain(j + 1, w);
            if tv == 0 {
                if ctx.chain_e[j + 1].is_empty() {
                    clock.mark(4);
                    return ReplayResult {
                        outcome: Outcome::AcceptUnique,
                        s1_fastpath,
                        chain_fastpath: true,
                    };
                }
                chain = Some(j + 1);
                clock.mark(4);
                continue;
            }
            if ctx.chain_bound[j + 1] > tc as i32 - tv as i32 {
                clock.mark(4);
                return ReplayResult {
                    outcome: Outcome::Reject,
                    s1_fastpath,
                    chain_fastpath: true,
                };
            }
            let e = &ctx.chain_e[j];
            s.m_buf.clear();
            s.m_buf.extend_from_slice(e);
            s.m_buf.push(u_c);
            for &u in e.iter() {
                s.m_buf.push(u_c + u);
            }
            chain = None;
            if s.m_buf.len() > DIRECT_THRESHOLD {
                let r = later_stratum_wht_split(s, ctx, w, tc, tv, u_c);
                clock.mark(3);
                r
            } else {
                let r = later_stratum_direct_split(s, ctx, w, &counts_v, u_c);
                clock.mark(4);
                r
            }
        } else if s.m_buf.len() > DIRECT_THRESHOLD {
            let r = later_stratum_wht_split(s, ctx, w, tc, tv, u_c);
            clock.mark(3);
            r
        } else {
            let r = later_stratum_direct_split(s, ctx, w, &counts_v, u_c);
            clock.mark(4);
            r
        };

        if !u_c_in {
            return ReplayResult {
                outcome: Outcome::Reject,
                s1_fastpath,
                chain_fastpath: false,
            };
        }
        if s.m_buf.len() == 1 {
            return ReplayResult {
                outcome: Outcome::AcceptUnique,
                s1_fastpath,
                chain_fastpath: false,
            };
        }
    }
    if let Some(j) = chain {
        let e = &ctx.chain_e[j];
        s.m_buf.clear();
        s.m_buf.extend_from_slice(e);
        s.m_buf.push(u_c);
        for &u in e.iter() {
            s.m_buf.push(u_c + u);
        }
    }
    ReplayResult {
        outcome: Outcome::Tie(s.m_buf.clone()),
        s1_fastpath,
        chain_fastpath: false,
    }
}

fn first_stratum_split(
    s: &mut Scratch,
    ctx: &mut Ctx,
    clock: &mut Marks,
    w: usize,
    tc: u32,
    tv: u32,
    u_c: u16,
) -> bool {
    let h = 1usize << (ctx.kp1 - 1);
    s.m_buf.clear();
    s.g.clear();
    s.g.resize(h, 0);
    for (gx, &wt) in s.g.iter_mut().zip(s.wt_v.iter()) {
        *gx = (wt as usize == w) as i32;
    }
    clock.mark(1);
    wht_in_place(&mut s.g);
    let fc = ctx.fhat_c[w].as_slice();
    let g = s.g.as_slice();
    let target = tc as i32 - tv as i32;
    let mut i = 1usize;
    while i < h {
        let end = (i + 256).min(h);
        let mut mx = i32::MIN;
        for u in i..end {
            let val = fc[u] + g[u].abs();
            mx = mx.max(val);
        }
        if mx > target {
            clock.mark(2);
            return false;
        }
        i = end;
    }
    for (u, (&f, &gv)) in fc.iter().zip(g.iter()).enumerate().take(h).skip(1) {
        if f + gv == target {
            s.m_buf.push(u as u16);
        }
    }
    for (u, (&f, &gv)) in fc.iter().zip(g.iter()).enumerate().take(h) {
        if f - gv == target {
            s.m_buf.push(u_c + u as u16);
        }
    }
    debug_assert!(s.m_buf.contains(&u_c));
    clock.mark(2);
    true
}

fn later_stratum_wht_split(
    s: &mut Scratch,
    ctx: &mut Ctx,
    w: usize,
    tc: u32,
    tv: u32,
    u_c: u16,
) -> bool {
    let h = 1usize << (ctx.kp1 - 1);
    let hmask = u_c - 1;
    s.g.clear();
    s.g.resize(h, 0);
    if tv > 0 {
        for (gx, &wt) in s.g.iter_mut().zip(s.wt_v.iter()) {
            *gx = (wt as usize == w) as i32;
        }
        wht_in_place(&mut s.g);
    }
    let fc: Option<&[i32]> = if tc > 0 {
        ctx.ensure_fhat(w);
        Some(ctx.fhat_c[w].as_slice())
    } else {
        None
    };
    let g = std::mem::take(&mut s.g);
    let val = |u: u16| -> i32 {
        let up = (u & hmask) as usize;
        let f = fc.map_or(0, |f| f[up]);
        if u & u_c == 0 {
            f + g[up]
        } else {
            f - g[up]
        }
    };
    let mut best = i32::MIN;
    for &u in &s.m_buf {
        best = best.max(val(u));
    }
    s.m_buf.retain(|&u| val(u) == best);
    s.g = g;
    s.m_buf.contains(&u_c)
}

fn later_stratum_direct_split(
    s: &mut Scratch,
    ctx: &mut Ctx,
    w: usize,
    counts_v: &[u32; 65],
    u_c: u16,
) -> bool {
    let h = 1usize << (ctx.kp1 - 1);
    let hmask = u_c - 1;
    let tv = counts_v[w];
    s.ensure_sorted_v(h, counts_v);
    let vb = s.start_v[w] as usize;
    let mv = &s.sorted_v[vb..vb + tv as usize];
    let tc = ctx.counts_c[w];
    let kh = (ctx.kp1 as u32 - 1) * (1u32 << (ctx.kp1 - 1));
    if ctx.fhat_built >> w & 1 == 0 && s.m_buf.len() as u32 * tc > kh {
        ctx.ensure_fhat(w);
    }
    let b = ctx.start_c[w] as usize;
    let e = b + ctx.counts_c[w] as usize;
    let mc = &ctx.sorted_c[b..e];
    let fc: Option<&[i32]> = if ctx.fhat_built >> w & 1 == 1 {
        Some(ctx.fhat_c[w].as_slice())
    } else {
        None
    };
    s.counts_buf.clear();
    let mut best = u32::MAX;
    for &u in &s.m_buf {
        let up = u & hmask;
        let p_c: u32 = match fc {
            Some(f) => ((tc as i32 - f[up as usize]) / 2) as u32,
            None => mc
                .iter()
                .filter(|&&x| (up & x).count_ones() & 1 == 1)
                .count() as u32,
        };
        let p_v: u32 = mv
            .iter()
            .filter(|&&x| (up & x).count_ones() & 1 == 1)
            .count() as u32;
        let c = p_c + if u & u_c == 0 { p_v } else { tv - p_v };
        s.counts_buf.push(c);
        best = best.min(c);
    }
    let counts = std::mem::take(&mut s.counts_buf);
    let mut i = 0;
    s.m_buf.retain(|_| {
        let keep = counts[i] == best;
        i += 1;
        keep
    });
    s.counts_buf = counts;
    s.m_buf.contains(&u_c)
}

// ───────────────────────────── brute-force reference (--validate)

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
            let mut ctx = Ctx::default();
            let mut s = Scratch::default();
            for (rows, cs) in frames.iter().zip(cands.iter()) {
                ctx.build(rows);
                for &v in cs {
                    let mut marks = Marks::start(false);
                    let got = phi_cascade_split(&mut s, &mut ctx, v, n, &mut marks);
                    let want = reference_decision(rows, v, n);
                    assert_eq!(got.outcome, want, "cascade != reference at kp1={kp1}");
                }
            }
        }

        // Timing passes (marks disabled). Repeat to ≥ ~10 ms per row.
        let iters = ((1u64 << 24) / (h as u64 * (n_parents * n_cands) as u64).max(1)).max(1);
        let mut ctx = Ctx::default();
        let mut s = Scratch::default();

        let c0 = mono_cycles();
        for _ in 0..iters {
            for (rows, cs) in frames.iter().zip(cands.iter()) {
                ctx.build(rows);
                for &v in cs {
                    let mut marks = Marks::start(false);
                    black_box(
                        phi_cascade_split(&mut s, &mut ctx, black_box(v), n, &mut marks)
                            .chain_fastpath,
                    );
                }
            }
        }
        let hot_cyc = mono_cycles().wrapping_sub(c0) / (iters * (n_parents * n_cands) as u64);

        let cold_iters = iters.min(4);
        let c0 = mono_cycles();
        for _ in 0..cold_iters {
            for (rows, cs) in frames.iter().zip(cands.iter()) {
                ctx.build(rows);
                for &v in cs {
                    evict_l1_l2(&mut junk);
                    let mut marks = Marks::start(false);
                    black_box(
                        phi_cascade_split(&mut s, &mut ctx, black_box(v), n, &mut marks)
                            .chain_fastpath,
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

        // Marks + outcome pass.
        let mut acc = [0u64; 5];
        let mut outc = [0u64; 3]; // rej, acc, tie
        let mut s1f = 0u64;
        let mut chainf = 0u64;
        for (rows, cs) in frames.iter().zip(cands.iter()) {
            ctx.build(rows);
            for &v in cs {
                let mut marks = Marks::start(true);
                let r = phi_cascade_split(&mut s, &mut ctx, v, n, &mut marks);
                for (a, m) in acc.iter_mut().zip(marks.acc.iter()) {
                    *a += m;
                }
                match r.outcome {
                    Outcome::Reject => outc[0] += 1,
                    Outcome::AcceptUnique => outc[1] += 1,
                    Outcome::Tie(_) => outc[2] += 1,
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
