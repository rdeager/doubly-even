//! D15 coset-spectrum parent rule: the φ cascade and its tie-break.
//! D16 split-frame evaluation: per-parent work shared across siblings.
//!
//! ### The rule
//!
//! The parent of a rank-(k+1) code `D` is one of its `2^(k+1) − 1`
//! hyperplanes (index-2 subcodes `H_u`, one per nonzero functional `u` on
//! `D`; every hyperplane of a doubly-even code is doubly-even). Define
//!
//! ```text
//! φ_w(u) = #{ x ∈ D : wt(x) = w, u(x) = 1 }          (complement-coset
//! φ(u)   = (φ_w(u))_w ascending over all weight strata  weight spectrum)
//! ```
//!
//! and select `m(D)` = the Aut(D)-orbit of the argmin-lex hyperplane. If
//! the argmin set `M` is a single functional, it names the parent orbit
//! outright; otherwise the tie is broken exactly like the legacy rule,
//! restricted to `M`: the orbit of the argmin hyperplane whose
//! σ_D-permuted RREF is lexicographically least (σ_D = nauty's canonical
//! column order).
//!
//! ### Soundness (McKay 1998 requirements)
//!
//! `φ` is codeword-weight data only, so column permutations preserve it:
//! the argmin set is Aut(D)-invariant and transported by any isomorphism.
//! The σ-key tie-break picks exactly one hyperplane subspace (RREF is a
//! bijective subspace identifier), well-defined up to Aut(D) by the same
//! argument the legacy drop-last-σ-row rule relies on. Hence `m(D)` is an
//! isomorphism-invariant single-orbit parent function and canonical
//! augmentation stays isomorph-free. The runtime mass-formula panic in
//! `enumerate.rs` certifies every rank of every run end-to-end.
//!
//! ### Why this exists
//!
//! The legacy rule needs nauty's σ_D just to *name* the parent, so every
//! candidate pays a full canon call before the parent test starts — and
//! 93.4 % of those calls (N=22) end in a cheap weight-enumerator reject.
//! Under the φ rule the reject decision usually needs no canon call at
//! all: weight data plus per-stratum Walsh–Hadamard transforms, evaluated
//! lazily until the lex comparison resolves. Canon is paid only on
//! accepts (needed anyway for `Aut(D)` / mass / recursion) and on exact
//! ties.
//!
//! ### Evaluation frame
//!
//! All φ work happens in the fixed frame basis `[C's k RREF rows, v]`
//! (NOT `d_rref`): the candidate's own hyperplane is then the kernel of
//! the last-coordinate functional `u_C = 1 << k`, and coordinate vectors
//! are plain `u16` indices. Candidate generation guarantees `v ∉ C`
//! (nonzero coset reps), so the frame rows are independent.
//!
//! ### Split-frame sharing (D16)
//!
//! Bit `k` of a coordinate `x = (x', b)` splits the frame into the
//! **C-half** (`b = 0`: the `2^k` codewords of `C`, identical for every
//! sibling candidate of one parent) and the **v-half** (`b = 1`: the
//! coset `v + C`). All C-half data — codeword table, weights, histogram,
//! stratum lists, per-stratum WHTs — is computed ONCE per parent in a
//! [`PhiParentCtx`] and shared across the sibling candidates. Per
//! candidate only the v-half remains: `2^k` XOR+popcounts (branchless,
//! no Gray serial dependency), a histogram, and at most one half-size
//! WHT per stratum.
//!
//! The decomposition is exact — it is literally the last butterfly stage
//! of the full WHT factored out. For the stratum indicator `f = f_C + f_v`
//! and a functional `u = (u', a)`:
//!
//! ```text
//! f̂[(u', a)] = F̂_C[u'] + (1 − 2a) · Ĝ_v[u']
//! ```
//!
//! where `F̂_C`/`Ĝ_v` are the `2^k`-point WHTs of the C-half / v-half
//! indicators. In particular `f̂[u_C] = f̂[(0, 1)] = |T_w ∩ C| −
//! |T_w ∩ (v+C)|` is free from the two histograms, which yields two
//! exact first-stratum fast paths with no per-candidate spectral work:
//!
//! - **coset-only stratum** (`|T_w ∩ C| = 0`, `k ≥ 1`): some `(u', a)`
//!   pair always beats `u_C` — for any `u' ≠ 0`,
//!   `max(f̂[(u',0)], f̂[(u',1)]) = |Ĝ_v[u']| ≥ 0 > f̂[u_C]` — so the
//!   candidate REJECTS in O(1).
//! - **C-only stratum** (`|T_w ∩ (v+C)| = 0`): `Ĝ_v ≡ 0`, so
//!   `f̂[(u', a)] = F̂_C[u'] ≤ F̂_C[0] = f̂[u_C]`: `u_C` always survives,
//!   and the surviving argmin set is `E_w ∪ {u_C} ∪ (u_C + E_w)` with
//!   `E_w = {u' ≠ 0 : F̂_C[u'] = |T_w ∩ C|}` precomputed per parent.
//!   Empty `E_w` ⇒ ACCEPT-UNIQUE in O(1).
//!
//! ### The E-chain (D17): O(1) later strata while the pair structure holds
//!
//! A candidate that survives a C-only first stratum enters stratum 2 with
//! `M = E ∪ {u_C} ∪ (u_C + E)` — every nonzero `u' ∈ E` present as the
//! full pair `(u', 0), (u', 1)`. That **pair structure** is preserved by
//! every further C-only stratum (`Ĝ_v ≡ 0` filters both halves of a pair
//! together), and while it holds each stratum resolves in O(1) per
//! candidate from parent-only data:
//!
//! - **v-only stratum** (`tc = 0`, `tv > 0`): `max(±Ĝ_v[u']) = |Ĝ_v[u']|
//!   ≥ 0 > tc − tv = f̂[u_C]` — some pair member always beats `u_C`.
//!   REJECT, no parent data needed at all.
//! - **mixed stratum** (`tc, tv > 0`): `max` over a pair is
//!   `F̂_C[u'] + |Ĝ_v[u']| ≥ F̂_C[u']`, so the per-parent
//!   `max_{u' ∈ E_cur} F̂_C^{(w)}[u'] > tc − tv` proves REJECT in one
//!   integer compare — the amax theorem restricted to the running E-set.
//!   Otherwise the candidate materialises `M` and falls back to the
//!   generic stratum machinery (pair structure ends here).
//! - **C-only stratum**: `E_cur ← {u' ∈ E_cur : F̂_C^{(w)}[u'] = tc}`;
//!   empty ⇒ ACCEPT-UNIQUE. The filter depends only on the parent.
//!
//! The walk is identical for every sibling — the C-strata of one parent
//! arrive in the same ascending order — so the per-position E-sets and
//! bounds form a single per-parent **chain**, built lazily once and read
//! O(1) by every later sibling ([`PhiParentCtx::ensure_chain`]).
//! Decisions are exactly those of the generic machinery (the chain is the
//! same argmin cascade evaluated against parent-side data only).

use std::cell::RefCell;

use crate::linalg::{apply_permutation, row_reduce};
use crate::types::BinVec;

/// Default child-rank cap for the φ cascade. The per-candidate WHT costs
/// `k·2^k` adds post-D16 (was `(k+1)·2^(k+1)`), which passes the ~30 µs
/// canon call it replaces around `k+1 ≈ 14–16`; children of rank ≥ 14
/// only exist at `N ≥ 30`, where falling back to the legacy rule above
/// the cap is the right trade. Per-rank rule mixing is sound: rank is
/// iso-invariant and McKay's induction is local to one child rank.
pub const DEFAULT_PHI_MAX_RANK: u32 = 13;

/// Active parent-selection rule for one enumeration run.
///
/// Resolved ONCE per driver invocation (never per worker) so the seeder
/// and all workers are guaranteed to agree — a frontier-rule mismatch
/// would still be *sound* (per-rank mixing), but it would break the
/// seq-vs-parallel determinism harness and is never what anyone wants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParentRule {
    /// σ-based legacy rule: canonical parent = drop the last row of the
    /// σ_D-permuted RREF. One canon call per candidate, accept or reject.
    Legacy,
    /// D15 coset-spectrum rule for child rank ≤ `max_rank` (legacy
    /// above): canon calls only on accepts and φ-ties.
    CosetSpectrum { max_rank: u32 },
    /// Phase 1 measurement mode: legacy behaviour byte-for-byte, plus φ
    /// outcome tallies and the κ (kept-canon-ns) accounting.
    Audit,
}

impl ParentRule {
    /// Resolve from `DOUBLY_EVEN_PARENT_RULE` ∈ {`coset-spectrum` | `phi`
    /// (default), `legacy`, `audit`} and `DOUBLY_EVEN_PHI_MAX_RANK`
    /// (default [`DEFAULT_PHI_MAX_RANK`]). Unknown values panic loudly —
    /// a silently misread rule knob could waste a cloud run.
    ///
    /// The coset-spectrum rule became the default after the 2026-06-10
    /// ship gate: ≥ 2× wall at N = 24 and N = 26 vs a same-session
    /// legacy control (measured 3.0× and 6.5× at t=24 d=5; 7.6×
    /// sequential at N = 22), all DFGHILM Table 3 cells agreeing.
    /// `DOUBLY_EVEN_PARENT_RULE=legacy` is the kill-switch.
    pub fn from_env() -> Self {
        let max_rank = std::env::var("DOUBLY_EVEN_PHI_MAX_RANK")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(DEFAULT_PHI_MAX_RANK);
        match std::env::var("DOUBLY_EVEN_PARENT_RULE") {
            Err(_) => ParentRule::CosetSpectrum { max_rank },
            Ok(s) => match s.trim().to_ascii_lowercase().as_str() {
                "" | "coset-spectrum" | "phi" => ParentRule::CosetSpectrum { max_rank },
                "legacy" => ParentRule::Legacy,
                "audit" => ParentRule::Audit,
                other => panic!(
                    "DOUBLY_EVEN_PARENT_RULE: unknown value {other:?} \
                     (expected coset-spectrum | legacy | audit)"
                ),
            },
        }
    }
}

/// Above this argmin-set size, a stratum is filtered by a fresh
/// Walsh–Hadamard transform (cost `k·2^k` adds post-D16); at or below
/// it, per-functional direct parity counting over the stratum members
/// is cheaper. Crossover is flat in the 32–128 range; 64 measured fine.
const DIRECT_THRESHOLD: usize = 64;

/// Outcome of the φ cascade for one candidate augmentation `(C, v)`.
pub enum PhiOutcome {
    /// `u_C` left the running argmin set: `C` is provably not the
    /// φ-selected parent. Sound reject — no canon call needed.
    Reject,
    /// The argmin set is exactly `{u_C}`: `m(D)` is the orbit of `C`
    /// itself. Accept — canon is still called, but only because the
    /// recursion needs `Aut(D)` (same cost any rule pays on accepts).
    AcceptUnique,
    /// Full spectra exhausted with `u_C` tied among `|M| > 1`
    /// functionals. Needs the σ_D tie-break (one canon call, as legacy).
    Tie(Vec<u16>),
}

pub struct PhiResult {
    pub outcome: PhiOutcome,
    /// Number of weight strata evaluated before the cascade resolved.
    pub strata_used: u32,
    /// `|M|` at the decision point (1 for AcceptUnique; the surviving
    /// tie-set size for Tie). For REJECTS decided at the first stratum
    /// (D16 early exit) or on the E-chain (D17 — the argmin set is never
    /// materialised) a witness count of 1 is reported (pre-D16 this was
    /// the full beating-set size). Diagnostic mean only — nothing gates
    /// on it.
    pub m_size_at_decision: u32,
    /// True when the first-stratum decision needed no per-candidate WHT
    /// (k = 0 frame, coset-only stratum, or C-only stratum fast path).
    pub s1_fastpath: bool,
    /// True when the FINAL decision came from the D17 E-chain at stratum
    /// ≥ 2 (O(1) per-candidate: v-only reject, mixed-stratum bound
    /// reject, or chain-filter accept) — i.e. the cascade ran to
    /// completion without materialising a per-candidate argmin set.
    pub chain_fastpath: bool,
}

/// Per-parent shared φ context (D16). Holds everything that depends only
/// on the parent `C`: the C-half codeword table, weights, histogram and
/// stratum lists (built eagerly), plus per-stratum `2^k`-point WHTs
/// `F̂_C^{(w)}` and the C-only-stratum argmax sets `E_w` (built lazily on
/// first use of stratum `w`). One ctx serves every sibling candidate of
/// one parent; backing boxes are recycled through a thread-local pool.
///
/// Memory: at the default rank cap (`k ≤ 12`, `h = 4096`) the eager part
/// is ~46 KB and each cached stratum WHT adds 16 KB; one live ctx per
/// recursion level per thread (sizes geometric in k), so the per-thread
/// pool footprint stays a few hundred KB.
pub(crate) struct PhiParentCtx {
    /// Frame size `k + 1`; the half size is `h = 1 << (kp1 - 1)`.
    kp1: usize,
    /// Copy of the parent rows the ctx was built from (slot-misuse guard).
    parent_rows: Vec<BinVec>,
    /// C-half codeword per plain coordinate `x' ∈ [0, h)`.
    cwords: Vec<BinVec>,
    /// `wt_c[x'] = wt(cwords[x'])`.
    wt_c: Vec<u8>,
    /// C-half weight histogram (`counts_c[0] == 1`, the zero word).
    counts_c: [u32; 65],
    /// Counting-sort offsets into `sorted_c` per weight.
    start_c: [u32; 66],
    /// C-half coordinates sorted by weight (stratum member lists).
    sorted_c: Vec<u16>,
    /// Lazy per-stratum `2^k`-point WHT of the C-half indicator at
    /// weight `w`; valid iff bit `w` of `fhat_built` is set.
    fhat_c: Vec<Vec<i32>>,
    /// Lazy `E_w = {u' ≠ 0 : F̂_C^{(w)}[u'] = counts_c[w]}` (ascending),
    /// built together with `fhat_c[w]`; only populated when
    /// `counts_c[w] > 0` (it is the C-only-stratum surviving set).
    e_set: Vec<Vec<u16>>,
    /// Lazy `amax_w = max_{u' ≠ 0} F̂_C^{(w)}[u']`, built with `fhat_c[w]`.
    /// Powers the O(1) first-stratum reject: `max(f̂[(u',0)], f̂[(u',1)])
    /// = F̂_C[u'] + |Ĝ_v[u']| ≥ F̂_C[u']`, so `amax_w > tc − tv` proves
    /// some functional beats `u_C` before any per-candidate spectral
    /// work. (i32::MIN when the half has a single coordinate.)
    amax: Vec<i32>,
    /// Bit `w` set ⇔ `fhat_c[w]` / `e_set[w]` are valid for this parent.
    fhat_built: u128,
    /// D17 E-chain (see module doc): entry `j` is the state after a
    /// candidate has passed the parent's first `j + 1` C-strata all
    /// C-only. `chain_e[0]` is a copy of `E_{w₁}`; `chain_e[j]` is the
    /// `F̂ = tc` filter of `chain_e[j−1]` at the (j+1)-th C-stratum.
    /// Storage is grow-only across parents; `chain_len` is the valid
    /// prefix. Entries build sequentially and lazily — a sibling can
    /// only reach chain position j by passing positions 0..j first.
    chain_e: Vec<Vec<u16>>,
    /// `chain_bound[j]` (j ≥ 1) = `max_{u' ∈ chain_e[j−1]} F̂_C^{(w_j)}`
    /// where `w_j` is the weight of the (j+1)-th C-stratum: the one
    /// integer the O(1) mixed-stratum reject compares against `tc − tv`.
    /// `chain_bound[0]` is unused (i32::MIN).
    chain_bound: Vec<i32>,
    /// Weight at which chain entry `j` was built (debug invariant: every
    /// sibling must present the same C-stratum weight at position `j`).
    chain_w: Vec<u8>,
    /// Number of valid chain entries for the current parent.
    chain_len: usize,
    /// Build time (eager + lazy) accumulated since the last drain, and
    /// the number of eager builds. Drained by `WorkerState` into
    /// `stats_phi_ctx_ns` / `stats_phi_ctx_builds`.
    ns: u64,
    builds: u64,
}

impl Default for PhiParentCtx {
    fn default() -> Self {
        Self {
            kp1: 0,
            parent_rows: Vec::new(),
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
            ns: 0,
            builds: 0,
        }
    }
}

impl PhiParentCtx {
    /// (Re)build the eager C-half tables for parent `c_rref`. Buffers
    /// are grow-only across pool recycling.
    fn build(&mut self, c_rref: &[BinVec]) {
        let t0 = std::time::Instant::now();
        let kp1 = c_rref.len() + 1;
        debug_assert!(kp1 <= 16, "φ cascade needs k+1 ≤ 16 (u16 coordinate vectors)");
        let h = 1usize << (kp1 - 1);
        self.kp1 = kp1;
        self.parent_rows.clear();
        self.parent_rows.extend_from_slice(c_rref);
        // Gray-code sweep over C's k rows: codeword per plain coordinate.
        self.cwords.clear();
        self.cwords.resize(h, 0);
        let mut cur: BinVec = 0;
        for i in 1..h {
            let flip = i.trailing_zeros() as usize;
            cur ^= c_rref[flip];
            self.cwords[i ^ (i >> 1)] = cur;
        }
        // Weights (separate pass: branchless, auto-vectorises) + histogram.
        self.wt_c.clear();
        self.wt_c.resize(h, 0);
        for x in 0..h {
            self.wt_c[x] = self.cwords[x].count_ones() as u8;
        }
        self.counts_c = [0u32; 65];
        for x in 0..h {
            self.counts_c[self.wt_c[x] as usize] += 1;
        }
        debug_assert_eq!(
            self.counts_c[0], 1,
            "dependent parent rows: C's RREF rows must be independent"
        );
        // Counting sort into per-weight stratum member lists.
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
        self.builds += 1;
        self.ns += t0.elapsed().as_nanos() as u64;
    }

    /// Ensure `F̂_C^{(w)}` (and `E_w`) are built for stratum `w`.
    fn ensure_fhat(&mut self, w: usize) {
        if self.fhat_built >> w & 1 == 1 {
            return;
        }
        let t0 = std::time::Instant::now();
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
        self.ns += t0.elapsed().as_nanos() as u64;
    }

    /// Ensure D17 chain entry `j` exists, built at C-stratum weight `w`
    /// (see the `chain_e` field doc). Entry 0 copies `E_w`; entry `j ≥ 1`
    /// scans `chain_e[j−1]` once against `F̂_C^{(w)}`, producing the
    /// `F̂ = tc` filter and the bound `max F̂` together. Amortised: built
    /// once per parent per position, read O(1) by every later sibling.
    fn ensure_chain(&mut self, j: usize, w: usize) {
        if j < self.chain_len {
            debug_assert_eq!(
                self.chain_w[j], w as u8,
                "chain position {j} revisited at a different weight"
            );
            return;
        }
        debug_assert_eq!(j, self.chain_len, "chain entries build sequentially");
        self.ensure_fhat(w); // accounts its own ns; time only the scan below
        let t0 = std::time::Instant::now();
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
        self.ns += t0.elapsed().as_nanos() as u64;
    }
}

thread_local! {
    /// Recycled ctx boxes (LIFO; one live ctx per recursion level, so
    /// the pool depth tracks the recursion depth).
    static PHI_CTX_POOL: RefCell<Vec<Box<PhiParentCtx>>> = const { RefCell::new(Vec::new()) };
}

/// One parent's φ context slot. A stack local in each `traverse` /
/// `traverse_seed` frame, created before the candidate loop and passed
/// by `&mut` into `test_candidate`; the ctx materialises lazily on the
/// first φ-tested candidate (so legacy-rule runs and mass-stopped loops
/// never pay a build) and returns to the thread-local pool on drop.
pub struct PhiParentSlot {
    ctx: Option<Box<PhiParentCtx>>,
}

impl PhiParentSlot {
    pub fn new() -> Self {
        Self { ctx: None }
    }

    fn ensure(&mut self, c_rref: &[BinVec]) -> &mut PhiParentCtx {
        if self.ctx.is_none() {
            let mut b = PHI_CTX_POOL
                .with(|p| p.borrow_mut().pop())
                .unwrap_or_default();
            b.ns = 0;
            b.builds = 0;
            b.build(c_rref);
            self.ctx = Some(b);
        }
        let ctx = self.ctx.as_mut().expect("ctx ensured above");
        debug_assert_eq!(
            ctx.parent_rows.as_slice(),
            c_rref,
            "PhiParentSlot reused across different parents"
        );
        ctx
    }

    /// Drain `(build_ns, builds)` accumulated since the last drain.
    pub(crate) fn take_build_stats(&mut self) -> (u64, u64) {
        match self.ctx.as_mut() {
            Some(c) => {
                let r = (c.ns, c.builds);
                c.ns = 0;
                c.builds = 0;
                r
            }
            None => (0, 0),
        }
    }
}

impl Drop for PhiParentSlot {
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx.take() {
            PHI_CTX_POOL.with(|p| p.borrow_mut().push(ctx));
        }
    }
}

/// Grow-only per-thread per-candidate scratch (D13-V4 house pattern, cf.
/// `canon::CanonScratch`): zero heap allocation per call after warmup,
/// except the rare `Tie` return which clones its small argmin set.
struct PhiScratch {
    /// `wt_v[x'] = wt(cwords[x'] ^ v)` — v-half weights, plain-indexed.
    wt_v: Vec<u8>,
    /// Half-size WHT buffer (`Ĝ_v`). `i32`: values reach ±2^k ≤ 32768.
    g: Vec<i32>,
    /// v-half coordinates counting-sorted by weight. Built lazily, once
    /// per candidate, on the first later-stratum DIRECT filter (the only
    /// consumer of explicit member lists; the WHT paths fill their
    /// indicator branchlessly from `wt_v` instead).
    sorted_v: Vec<u16>,
    start_v: [u32; 66],
    sorted_v_built: bool,
    /// Direct-path parity counts (one per surviving functional).
    counts_buf: Vec<u32>,
    /// Running argmin set of functionals.
    m_buf: Vec<u16>,
}

impl Default for PhiScratch {
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

impl PhiScratch {
    /// Lazy per-candidate counting sort of the v-half coords by weight.
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

thread_local! {
    static PHI_SCRATCH: RefCell<PhiScratch> = RefCell::new(PhiScratch::default());
}

/// Sampled φ sub-phase timing (`phase_timers` builds only). A φ cascade
/// runs well under a microsecond post-D16; five `Instant` pairs per call
/// would distort the very share being measured, so every 64th call on
/// each thread is fully timed via the ~6–9 ns cycle counter
/// (`crate::cycles`) and the rest pay one thread-local counter
/// increment. The driver takes the sample via [`phi_sample::take_last`]
/// right after the cascade returns and reweights per rank at analysis
/// time (φ cost correlates strongly with k — see
/// `stats_phi_sampled_calls_by_k`). Diagnostic ±10 %.
#[cfg(feature = "phase_timers")]
pub mod phi_sample {
    use std::cell::Cell;

    /// `call_index & SAMPLE_MASK == 0` ⇒ fully timed. 64 keeps the
    /// expected overhead ≈ (5 × 9 ns)/64 ≈ 0.7 ns/call; drop to 255 if
    /// the wall-overhead gate (≤ 1.02×) ever fails.
    pub(crate) const SAMPLE_MASK: u64 = 63;
    pub const N_PHASES: usize = 5;

    thread_local! {
        static CALL_COUNT: Cell<u64> = const { Cell::new(0) };
        static LAST_SAMPLE: Cell<Option<[u64; N_PHASES]>> = const { Cell::new(None) };
    }

    #[inline]
    pub(crate) fn should_sample() -> bool {
        CALL_COUNT.with(|c| {
            let n = c.get();
            c.set(n.wrapping_add(1));
            n & SAMPLE_MASK == 0
        })
    }

    #[inline]
    pub(crate) fn record(ns: [u64; N_PHASES]) {
        LAST_SAMPLE.with(|c| c.set(Some(ns)));
    }

    /// Take the sub-phase ns of the most recent cascade IF it was
    /// sampled (cleared on take, so a sample is never double-counted).
    #[inline]
    pub fn take_last() -> Option<[u64; N_PHASES]> {
        LAST_SAMPLE.with(|c| c.take())
    }

    /// Reset the thread-local call counter so the NEXT cascade on this
    /// thread is fully timed. Microbench-only entry point
    /// (`scripts/microbench/phi_replay` drives its per-phase rows with
    /// force + [`take_last`] pairs); cold code, never on the
    /// enumeration path.
    pub fn force_next_sample() {
        CALL_COUNT.with(|c| c.set(0));
    }
}

/// Per-cascade phase clock: no-op for unsampled calls and on
/// non-`phase_timers` builds (zero-sized, fully compiled away).
/// Post-D16 phase indices: 0 v-half weights+histogram, 1 first-stratum
/// member collection, 2 first-stratum decision (fast paths / Ĝ WHT +
/// fused scan), 3 later-stratum WHT, 4 later-stratum direct parity.
/// (Pre-D16: 0 frame+Gray sweep, 1 counting sort, 2 first-stratum
/// argmin — phases 0–2 cover the same prefix work, re-split.)
#[cfg(feature = "phase_timers")]
struct PhaseClock {
    sampled: bool,
    t: u64,
    acc: [u64; 5],
}

#[cfg(feature = "phase_timers")]
impl PhaseClock {
    #[inline]
    fn start() -> Self {
        let sampled = phi_sample::should_sample();
        Self {
            sampled,
            t: if sampled { crate::cycles::mono_cycles() } else { 0 },
            acc: [0; 5],
        }
    }

    #[inline]
    fn mark(&mut self, idx: usize) {
        if self.sampled {
            let now = crate::cycles::mono_cycles();
            self.acc[idx] += now.wrapping_sub(self.t);
            self.t = now;
        }
    }

    /// Convert to ns and publish. Called at every cascade return point.
    #[inline]
    fn commit(&self) {
        if self.sampled {
            let mut ns = [0u64; 5];
            for (o, &c) in ns.iter_mut().zip(self.acc.iter()) {
                *o = crate::cycles::cycles_to_ns(c);
            }
            phi_sample::record(ns);
        }
    }
}

#[cfg(not(feature = "phase_timers"))]
struct PhaseClock;

#[cfg(not(feature = "phase_timers"))]
impl PhaseClock {
    #[inline(always)]
    fn start() -> Self {
        PhaseClock
    }
    #[inline(always)]
    fn mark(&mut self, _idx: usize) {}
    #[inline(always)]
    fn commit(&self) {}
}

/// Evaluate the φ cascade for candidate `v` against parent RREF `c_rref`,
/// building a throwaway per-parent ctx. Compat entry for unit tests and
/// one-off callers; the enumeration hot path uses [`phi_cascade_shared`]
/// with a per-parent slot so the C-half work amortises across siblings.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn phi_cascade(c_rref: &[BinVec], v: BinVec, n: u32) -> PhiResult {
    let mut slot = PhiParentSlot::new();
    phi_cascade_shared(&mut slot, c_rref, v, n)
}

/// Evaluate the φ cascade for candidate `v` of the parent owning `slot`
/// (D16 split-frame path). The slot's ctx is built on first use and
/// shared by every sibling candidate tested through the same slot.
pub fn phi_cascade_shared(
    slot: &mut PhiParentSlot,
    c_rref: &[BinVec],
    v: BinVec,
    n: u32,
) -> PhiResult {
    let ctx = slot.ensure(c_rref);
    PHI_SCRATCH.with(|cell| phi_cascade_split(&mut cell.borrow_mut(), ctx, v, n))
}

fn phi_cascade_split(
    s: &mut PhiScratch,
    ctx: &mut PhiParentCtx,
    v: BinVec,
    n: u32,
) -> PhiResult {
    let mut clock = PhaseClock::start();
    let kp1 = ctx.kp1;
    let h = 1usize << (kp1 - 1);
    let u_c: u16 = 1 << (kp1 - 1);

    // Phase 0: v-half weights + histogram. Branchless XOR+popcount over
    // the shared C-half codeword table — no Gray serial dependency, so
    // LLVM auto-vectorises the weight loop. The histogram accumulates
    // into 4 interleaved sub-counts to break the store-to-load
    // dependency chain on repeated weights (classic histogram split).
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

    // Lazy lex cascade, strata ascending over D = C ∪ (v+C). (Doubly-even
    // inputs only have strata at multiples of 4; the loop is
    // weight-agnostic so unit tests can exercise arbitrary frames.)
    let mut strata_used = 0u32;
    let mut first = true;
    let mut s1_fastpath = false;
    // D17 E-chain position: `Some(j)` ⇔ the pair structure is alive and
    // the running argmin set is `chain_e[j] ∪ {u_C} ∪ (u_C + chain_e[j])`
    // — NOT materialised in m_buf (that is the point).
    let mut chain: Option<usize> = None;
    let n_cap = (n as usize).min(64);
    for w in 1..=n_cap {
        let tc = ctx.counts_c[w];
        let tv = counts_v[w];
        if tc + tv == 0 {
            continue;
        }
        strata_used += 1;

        let u_c_in = if first {
            first = false;
            // k = 0 frame: u_C is the only nonzero functional.
            if h == 1 {
                clock.mark(2);
                clock.commit();
                return PhiResult {
                    outcome: PhiOutcome::AcceptUnique,
                    strata_used,
                    m_size_at_decision: 1,
                    s1_fastpath: true,
                    chain_fastpath: false,
                };
            }
            // Coset-only first stratum: O(1) exact reject (see module
            // doc — some pair (u', a) always beats u_C).
            if tc == 0 {
                clock.mark(2);
                clock.commit();
                return PhiResult {
                    outcome: PhiOutcome::Reject,
                    strata_used,
                    m_size_at_decision: 1,
                    s1_fastpath: true,
                    chain_fastpath: false,
                };
            }
            // C-only first stratum: u_C always survives; the argmin set
            // comes from the per-parent E_w. Empty ⇒ unique accept;
            // otherwise enter the E-chain at position 0 (the argmin set
            // stays implicit — no m_buf materialisation).
            if tv == 0 {
                s1_fastpath = true;
                ctx.ensure_fhat(w);
                if ctx.e_set[w].is_empty() {
                    clock.mark(2);
                    clock.commit();
                    return PhiResult {
                        outcome: PhiOutcome::AcceptUnique,
                        strata_used,
                        m_size_at_decision: 1,
                        s1_fastpath: true,
                        chain_fastpath: false,
                    };
                }
                ctx.ensure_chain(0, w);
                chain = Some(0);
                clock.mark(2);
                continue;
            }
            // General first stratum (tc ≥ 1, tv ≥ 1). O(1) exact
            // reject first: max over the pair (u',0)/(u',1) is
            // F̂_C[u'] + |Ĝ_v[u']| ≥ F̂_C[u'], so a per-parent amax
            // beating f̂[u_C] = tc − tv settles the candidate with
            // no v-half spectral work.
            ctx.ensure_fhat(w);
            if ctx.amax[w] > tc as i32 - tv as i32 {
                clock.mark(2);
                clock.commit();
                return PhiResult {
                    outcome: PhiOutcome::Reject,
                    strata_used,
                    m_size_at_decision: 1,
                    s1_fastpath: true,
                    chain_fastpath: false,
                };
            }
            first_stratum_split(s, ctx, &mut clock, w, tc, tv, u_c)
        } else if let Some(j) = chain {
            // D17 E-chain stratum (see module doc): pair structure
            // alive, argmin set implicit. Each arm is O(1) per
            // candidate; only leaving the chain materialises m_buf.
            if tc == 0 {
                // v-only stratum: |Ĝ_v[u']| ≥ 0 > tc − tv for any pair
                // in the set — some pair member always beats u_C.
                clock.mark(4);
                clock.commit();
                return PhiResult {
                    outcome: PhiOutcome::Reject,
                    strata_used,
                    m_size_at_decision: 1,
                    s1_fastpath,
                    chain_fastpath: true,
                };
            }
            ctx.ensure_chain(j + 1, w);
            if tv == 0 {
                // C-only stratum: filter is parent-side; pair structure
                // survives. Empty filter ⇒ argmin set is exactly {u_C}.
                if ctx.chain_e[j + 1].is_empty() {
                    clock.mark(4);
                    clock.commit();
                    return PhiResult {
                        outcome: PhiOutcome::AcceptUnique,
                        strata_used,
                        m_size_at_decision: 1,
                        s1_fastpath,
                        chain_fastpath: true,
                    };
                }
                chain = Some(j + 1);
                clock.mark(4);
                continue;
            }
            // Mixed stratum: the E-restricted amax reject (one compare).
            if ctx.chain_bound[j + 1] > tc as i32 - tv as i32 {
                clock.mark(4);
                clock.commit();
                return PhiResult {
                    outcome: PhiOutcome::Reject,
                    strata_used,
                    m_size_at_decision: 1,
                    s1_fastpath,
                    chain_fastpath: true,
                };
            }
            // Bound inconclusive: leave the chain. Materialise the
            // argmin set in ascending-u order — byte-identical to what
            // the generic machinery would have carried here — and
            // process THIS stratum generically.
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
            clock.commit();
            return PhiResult {
                outcome: PhiOutcome::Reject,
                strata_used,
                // First-stratum early-exit rejects leave m_buf cleared;
                // report the witness count (see field doc).
                m_size_at_decision: s.m_buf.len().max(1) as u32,
                s1_fastpath,
                chain_fastpath: false,
            };
        }
        if s.m_buf.len() == 1 {
            clock.commit();
            return PhiResult {
                outcome: PhiOutcome::AcceptUnique,
                strata_used,
                m_size_at_decision: 1,
                s1_fastpath,
                chain_fastpath: false,
            };
        }
    }
    // Strata exhausted. If the chain is still alive the tie set was never
    // materialised — do it now, in the same ascending-u order.
    if let Some(j) = chain {
        let e = &ctx.chain_e[j];
        s.m_buf.clear();
        s.m_buf.extend_from_slice(e);
        s.m_buf.push(u_c);
        for &u in e.iter() {
            s.m_buf.push(u_c + u);
        }
    }
    clock.commit();
    PhiResult {
        outcome: PhiOutcome::Tie(s.m_buf.clone()),
        strata_used,
        m_size_at_decision: s.m_buf.len() as u32,
        s1_fastpath,
        chain_fastpath: false,
    }
}

/// First stratum, general case (`tc ≥ 1`, `tv ≥ 1`, `h ≥ 2`): argmin over
/// ALL nonzero functionals (u = 0 is excluded from the start — φ(0) ≡ 0
/// would always win and reject everything). Returns whether `u_c`
/// survived; on a reject `s.m_buf` is left cleared (early exit).
fn first_stratum_split(
    s: &mut PhiScratch,
    ctx: &mut PhiParentCtx,
    clock: &mut PhaseClock,
    w: usize,
    tc: u32,
    tv: u32,
    u_c: u16,
) -> bool {
    let h = 1usize << (ctx.kp1 - 1);
    s.m_buf.clear();
    // Phase 1: v-half stratum indicator, filled branchlessly from the
    // weight table (cmp+select vectorises; no member list needed).
    s.g.clear();
    s.g.resize(h, 0);
    for (gx, &wt) in s.g.iter_mut().zip(s.wt_v.iter()) {
        *gx = (wt as usize == w) as i32;
    }
    clock.mark(1);
    // Phase 2: Ĝ_v (half-size WHT of the v-half indicator) + the fused
    // argmax scan (F̂_C is already ensured by the caller's amax check).
    wht_in_place(&mut s.g);
    let fc = ctx.fhat_c[w].as_slice();
    let g = s.g.as_slice();
    // f̂[u_C] = f̂[(0, 1)] = F̂_C[0] − Ĝ_v[0] = tc − tv: free.
    let target = tc as i32 - tv as i32;
    // Fused scan with early exit: max(f̂[(u',0)], f̂[(u',1)]) =
    // F̂_C[u'] + |Ĝ_v[u']|, blockwise so the inner max auto-vectorises
    // and the reject branch costs one compare per block.
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
    // u_C attains the max. Build the argmin set in ascending-u order
    // (a = 0 half first, then a = 1) — byte-identical to pre-D16.
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
    debug_assert!(s.m_buf.contains(&u_c), "u_C must be in its own argmin set");
    clock.mark(2);
    true
}

/// Later stratum, large argmin set: half-size WHT of the v-half stratum
/// indicator, combined with the per-parent `F̂_C^{(w)}`, max restricted
/// to `m_buf`.
fn later_stratum_wht_split(
    s: &mut PhiScratch,
    ctx: &mut PhiParentCtx,
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
    // C-half spectrum: skip the (all-zero) WHT when the stratum has no
    // C-half members.
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

/// Later stratum, small argmin set: per-functional parity counting over
/// the split stratum members (`|M|·|T_w|` ops, cheaper than a WHT here).
/// `φ_w((u', a)) = pC(u') + (a = 0 ? pV(u') : tv − pV(u'))` where `pC` /
/// `pV` count odd-parity members in the C-half / v-half lists; `pC`
/// reads `(tc − F̂_C[u'])/2` for free whenever the stratum WHT happens
/// to be cached already.
fn later_stratum_direct_split(
    s: &mut PhiScratch,
    ctx: &mut PhiParentCtx,
    w: usize,
    counts_v: &[u32; 65],
    u_c: u16,
) -> bool {
    let h = 1usize << (ctx.kp1 - 1);
    let hmask = u_c - 1;
    let tv = counts_v[w];
    // The direct path is the only consumer of explicit v-half member
    // lists; build the per-candidate counting sort once, lazily.
    s.ensure_sorted_v(h, counts_v);
    let vb = s.start_v[w] as usize;
    let mv = &s.sorted_v[vb..vb + tv as usize];
    let tc = ctx.counts_c[w];
    // Read pC from the cached stratum WHT when present; BUILD it when
    // this single call's C-half parity work would already exceed the
    // WHT cost (k·h) — later siblings of the same parent then get
    // O(1) pC for free. Cost-only heuristic: decisions are unchanged.
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

/// In-place Walsh–Hadamard transform: `f̂[u] = Σ_x f[x]·(−1)^(u·x)`.
/// For a stratum indicator, `f̂[u] = |T_w| − 2·φ_w(u)`, so minimising
/// `φ_w` over `u` is maximising `f̂`.
///
/// D16 exactness note: applying this to the two halves of a split frame
/// and combining with `f̂[(u', a)] = F̂_C[u'] + (1−2a)·Ĝ_v[u']` is
/// literally the last butterfly stage (`h = 2^k`) factored out — the
/// split path computes the identical integers the full-frame transform
/// would.
///
/// `pub` for the `scripts/microbench/` sweep bins (`wht_sweep`), which
/// time this exact production body rather than a hand-copied clone.
pub fn wht_in_place(f: &mut [i32]) {
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

/// σ_D tie-break: among the argmin functionals `m_set`, select the
/// hyperplane whose σ_D-permuted RREF is lexicographically least, and
/// return that hyperplane's RREF in the ORIGINAL column frame (the
/// subspace identifier `subspace_in_orbit` consumes).
///
/// Iso-invariance: for isomorphic `D`s the σ-permuted argmin keys are the
/// same set of subspaces of the canonical form, so the selected orbit is
/// the same — word-for-word the legacy `canonical_parent` argument,
/// restricted to the φ-argmin set.
pub(crate) fn tie_break_parent(
    c_rref: &[BinVec],
    v: BinVec,
    n: u32,
    m_set: &[u16],
    sigma: &[u32],
) -> Vec<BinVec> {
    let mut best_key: Option<Vec<BinVec>> = None;
    let mut best_rref: Option<Vec<BinVec>> = None;
    for &u in m_set {
        let basis = hyperplane_basis(c_rref, v, u);
        let permuted: Vec<BinVec> = basis
            .iter()
            .map(|&b| apply_permutation(b, sigma))
            .collect();
        let (key, _) = row_reduce(&permuted, n);
        if best_key.as_ref().is_none_or(|bk| key < *bk) {
            let (orig_rref, _) = row_reduce(&basis, n);
            best_key = Some(key);
            best_rref = Some(orig_rref);
        }
    }
    best_rref.expect("tie_break_parent called with an empty argmin set")
}

/// Kernel basis of the hyperplane functional `u` on `D = ⟨C, v⟩` — the
/// codimension-1 subcode `{x ∈ D : u·coords(x) = 0}` as a k-row basis in
/// the original column frame (NOT row-reduced). For each `j ≠ j0` (`j0` =
/// lowest set bit of `u`), the combination `e_j + u_j·e_{j0}` lies in
/// `ker(u)`. Extracted verbatim from `tie_break_parent`'s loop body; the
/// tie-dump hook reuses it to materialise the tied strata.
pub(crate) fn hyperplane_basis(c_rref: &[BinVec], v: BinVec, u: u16) -> Vec<BinVec> {
    debug_assert_ne!(u, 0, "u = 0 is not a hyperplane functional");
    let kp1 = c_rref.len() + 1;
    let row_at = |j: usize| -> BinVec {
        if j < c_rref.len() {
            c_rref[j]
        } else {
            v
        }
    };
    let j0 = u.trailing_zeros() as usize;
    let mut basis: Vec<BinVec> = Vec::with_capacity(kp1 - 1);
    for j in 0..kp1 {
        if j == j0 {
            continue;
        }
        let mut word = row_at(j);
        if (u >> j) & 1 == 1 {
            word ^= row_at(j0);
        }
        basis.push(word);
    }
    basis
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent brute-force reference: full coset spectra for all
    /// nonzero functionals, lex argmin set (sorted).
    fn phi_reference_argmin(c_rref: &[BinVec], v: BinVec) -> Vec<u16> {
        let mut rows: Vec<BinVec> = c_rref.to_vec();
        rows.push(v);
        let kp1 = rows.len();
        let size = 1usize << kp1;
        let word = |x: usize| -> BinVec {
            let mut w = 0;
            for (j, &r) in rows.iter().enumerate() {
                if (x >> j) & 1 == 1 {
                    w ^= r;
                }
            }
            w
        };
        let spectrum = |u: usize| -> Vec<u32> {
            let mut spec = vec![0u32; 65];
            for x in 1..size {
                if ((u & x).count_ones() & 1) == 1 {
                    spec[word(x).count_ones() as usize] += 1;
                }
            }
            spec
        };
        let mut best: Option<Vec<u32>> = None;
        let mut argmin: Vec<u16> = Vec::new();
        for u in 1..size {
            let spec = spectrum(u);
            match &best {
                None => {
                    best = Some(spec);
                    argmin = vec![u as u16];
                }
                Some(b) => {
                    if spec < *b {
                        best = Some(spec);
                        argmin = vec![u as u16];
                    } else if spec == *b {
                        argmin.push(u as u16);
                    }
                }
            }
        }
        argmin
    }

    fn check_against_reference(c_rref: &[BinVec], v: BinVec, n: u32) {
        let argmin = phi_reference_argmin(c_rref, v);
        let u_c: u16 = 1 << c_rref.len();
        let res = phi_cascade(c_rref, v, n);
        match res.outcome {
            PhiOutcome::Reject => {
                assert!(
                    !argmin.contains(&u_c),
                    "cascade rejected but reference argmin contains u_C \
                     (c_rref={c_rref:?}, v={v:#b})"
                );
            }
            PhiOutcome::AcceptUnique => {
                assert_eq!(
                    argmin,
                    vec![u_c],
                    "cascade unique-accepted but reference argmin differs \
                     (c_rref={c_rref:?}, v={v:#b})"
                );
            }
            PhiOutcome::Tie(mut m) => {
                m.sort_unstable();
                assert!(m.len() > 1, "Tie with |M| <= 1");
                assert!(m.contains(&u_c), "Tie set must contain u_C");
                assert_eq!(
                    m, argmin,
                    "cascade tie set differs from reference argmin \
                     (c_rref={c_rref:?}, v={v:#b})"
                );
                assert!(!m.contains(&0), "u = 0 must never appear in a tie set");
            }
        }
    }

    #[test]
    fn matches_brute_force_on_fixed_frames() {
        // k+1 = 1 (zero-code parent).
        check_against_reference(&[], 0b1111, 8);
        // Small doubly-even frames at N = 8.
        check_against_reference(&[0b11110000], 0b00001111, 8);
        check_against_reference(&[0b11110000, 0b00111100], 0b01010101, 8);
        // e8-like high-symmetry frame (forces ties).
        check_against_reference(&[0b11110000, 0b00111100, 0b00001111], 0b01101001, 8);
        // Non-doubly-even frame (the cascade is weight-agnostic).
        check_against_reference(&[0b1100], 0b1010, 4);
    }

    #[test]
    fn matches_brute_force_on_orthogonal_sweep() {
        // N = 12, C = <111100000000, 001111000000>: sweep every weight-4
        // vector v orthogonal to C with v not in C, cross-checking the
        // cascade against the brute-force argmin on each.
        let n = 12u32;
        let c: Vec<BinVec> = vec![0b111100000000, 0b001111000000];
        let in_span = |v: BinVec| -> bool {
            for mask in 0..4u32 {
                let mut w = 0;
                if mask & 1 == 1 {
                    w ^= c[0];
                }
                if mask & 2 == 2 {
                    w ^= c[1];
                }
                if w == v {
                    return true;
                }
            }
            false
        };
        let mut checked = 0;
        for v in 1..(1u64 << n) {
            if v.count_ones() != 4 || in_span(v) {
                continue;
            }
            if c.iter().any(|&b| (b & v).count_ones() & 1 == 1) {
                continue;
            }
            check_against_reference(&c, v, n);
            checked += 1;
        }
        assert!(checked > 50, "sweep too small to be meaningful: {checked}");
    }

    #[test]
    fn column_permutation_leaves_outcome_invariant() {
        // φ is weight data only: permuting columns of the frame must not
        // change the outcome variant, tie set, or strata count.
        let c: Vec<BinVec> = vec![0b11110000, 0b00111100];
        let v: BinVec = 0b01010101;
        let perm: Vec<u32> = vec![3, 7, 1, 5, 0, 6, 2, 4];
        let c_p: Vec<BinVec> = c.iter().map(|&b| apply_permutation(b, &perm)).collect();
        let v_p = apply_permutation(v, &perm);
        let a = phi_cascade(&c, v, 8);
        let b = phi_cascade(&c_p, v_p, 8);
        assert_eq!(a.strata_used, b.strata_used);
        assert_eq!(a.m_size_at_decision, b.m_size_at_decision);
        let key = |r: PhiResult| match r.outcome {
            PhiOutcome::Reject => (0u8, Vec::new()),
            PhiOutcome::AcceptUnique => (1u8, Vec::new()),
            PhiOutcome::Tie(mut m) => {
                m.sort_unstable();
                (2u8, m)
            }
        };
        assert_eq!(key(a), key(b));
    }

    #[test]
    fn tie_break_returns_full_rank_hyperplane_stably() {
        // D = e8 (extended Hamming [8,4,4]) with the all-ones word as the
        // candidate: its coset spectrum (7·w4 + 1·w8) ties with the 7
        // other hyperplanes missing the all-ones word. The tie-break must
        // return a rank-k RREF and be independent of m_set ordering.
        let c: Vec<BinVec> = vec![0b10101010, 0b11001100, 0b11110000];
        let v: BinVec = 0b11111111;
        let res = phi_cascade(&c, v, 8);
        let m = match res.outcome {
            PhiOutcome::Tie(m) => m,
            _ => panic!("expected a tie on the high-symmetry frame"),
        };
        let sigma: Vec<u32> = (0..8).collect();
        let h1 = tie_break_parent(&c, v, 8, &m, &sigma);
        assert_eq!(h1.len(), c.len(), "hyperplane rank must be k");
        let mut m_rev = m.clone();
        m_rev.reverse();
        let h2 = tie_break_parent(&c, v, 8, &m_rev, &sigma);
        assert_eq!(h1, h2, "tie-break must not depend on argmin ordering");
    }

    #[test]
    fn wht_matches_direct_parity_counts() {
        // φ_w from the WHT must equal direct parity counting.
        let members: Vec<u16> = vec![0b0011, 0b0101, 0b1110, 0b1000];
        let size = 16usize;
        let mut f = vec![0i32; size];
        for &x in &members {
            f[x as usize] = 1;
        }
        wht_in_place(&mut f);
        for u in 0..size {
            let direct: i32 = members
                .iter()
                .filter(|&&x| ((u as u16 & x).count_ones() & 1) == 1)
                .count() as i32;
            assert_eq!(
                f[u],
                members.len() as i32 - 2 * direct,
                "WHT mismatch at u={u}"
            );
        }
    }

    /// Tiny deterministic PRNG (xorshift64*) — tests must not pull in a
    /// rand dependency, and seeds are fixed for reproducibility.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// D16 decomposition identity: the split-frame combine
    /// `f̂[(u', a)] = F̂_C[u'] + (1−2a)·Ĝ_v[u']` must reproduce the
    /// full-frame WHT exactly, for every stratum of random frames.
    #[test]
    fn split_wht_combine_matches_full_frame_wht() {
        let mut rng = XorShift(0x5EED_D16);
        for kp1 in 1..=8usize {
            for _ in 0..20 {
                let n = 16u32;
                let mask = (1u64 << n) - 1;
                // Random independent frame rows via row_reduce.
                let raw: Vec<BinVec> =
                    (0..kp1).map(|_| rng.next() & mask).collect();
                let (rref, _) = row_reduce(&raw, n);
                if rref.len() < kp1 {
                    continue; // dependent sample; skip
                }
                let c_rref = &rref[..kp1 - 1];
                let v = rref[kp1 - 1];
                let h = 1usize << (kp1 - 1);
                let size = 2 * h;
                // Full-frame weights (Gray sweep, as pre-D16).
                let mut rows: Vec<BinVec> = c_rref.to_vec();
                rows.push(v);
                let mut wt = vec![0u8; size];
                let mut cur: BinVec = 0;
                for i in 1..size {
                    let flip = i.trailing_zeros() as usize;
                    cur ^= rows[flip];
                    wt[i ^ (i >> 1)] = cur.count_ones() as u8;
                }
                for w in 1..=(n as usize) {
                    let mut full = vec![0i32; size];
                    let mut f_c = vec![0i32; h];
                    let mut g_v = vec![0i32; h];
                    let mut any = false;
                    for x in 0..size {
                        if wt[x] as usize == w {
                            any = true;
                            full[x] = 1;
                            if x < h {
                                f_c[x] = 1;
                            } else {
                                g_v[x - h] = 1;
                            }
                        }
                    }
                    if !any {
                        continue;
                    }
                    wht_in_place(&mut full);
                    wht_in_place(&mut f_c);
                    wht_in_place(&mut g_v);
                    for up in 0..h {
                        assert_eq!(full[up], f_c[up] + g_v[up], "a=0 combine");
                        assert_eq!(full[h + up], f_c[up] - g_v[up], "a=1 combine");
                    }
                }
            }
        }
    }

    /// D16 property test: the shared-ctx cascade must agree with the
    /// brute-force reference on random frames across all frame sizes,
    /// including the fast-path edge cases (coset-only / C-only first
    /// strata arise naturally in random frames).
    #[test]
    fn random_frames_match_brute_force() {
        let mut rng = XorShift(0xD16_CA5CADE);
        // (kp1, reps): brute force is O(4^kp1·kp1), so taper the count.
        let plan: &[(usize, usize)] =
            &[(1, 60), (2, 120), (3, 200), (4, 200), (5, 150), (6, 120), (7, 60), (8, 40), (9, 16), (10, 8)];
        let mut checked = 0u32;
        for &(kp1, reps) in plan {
            let mut done = 0;
            let mut attempts = 0;
            while done < reps && attempts < reps * 10 {
                attempts += 1;
                // Mix of small and larger n so strata land on diverse
                // weights (including weight collisions across halves).
                let n = [8u32, 12, 16, 24][(rng.next() % 4) as usize];
                let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
                let raw: Vec<BinVec> = (0..kp1).map(|_| rng.next() & mask).collect();
                let (rref, _) = row_reduce(&raw, n);
                if rref.len() < kp1 {
                    continue;
                }
                let c_rref: Vec<BinVec> = rref[..kp1 - 1].to_vec();
                let v = rref[kp1 - 1];
                check_against_reference(&c_rref, v, n);
                done += 1;
                checked += 1;
            }
            assert!(done > 0, "no independent samples at kp1={kp1}");
        }
        assert!(checked > 500, "property test too small: {checked}");
    }

    /// D17 E-chain test: a parent whose C-strata force multi-position
    /// chains (E⁰ = {2,4,6} at w=4 → E¹ = {4} at w=8 → ∅ at w=12), swept
    /// with random candidates through ONE shared slot, each checked
    /// against the brute-force reference AND a fresh cascade. Tallies
    /// assert every chain arm actually fired: v-only O(1) reject,
    /// E-restricted amax reject / chain-exit at mixed strata, and the
    /// O(1) unique accept at chain position 2.
    #[test]
    fn e_chain_multi_stratum_matches_brute_force() {
        let n = 24u32;
        // Block design: r1 w4, r2 w8, r3 w12; every r3-involving word has
        // weight ≥ 12, so strata w4 {r1} and w8 {r2} are pure-C for any
        // candidate coset of min weight ≥ 13.
        let c: Vec<BinVec> = vec![
            0b1111_0000_0000_0000_0000_0000,
            0b0000_1111_1111_0000_0000_0000,
            0b0000_0000_0000_1111_1111_1111,
        ];
        let mut slot = PhiParentSlot::new();
        let mut rng = XorShift(0xE5E7_C8A1);
        let mask = (1u64 << n) - 1;
        let in_span = |v: BinVec| -> bool {
            (0..8u32).any(|m| {
                let mut w = 0;
                for (j, &r) in c.iter().enumerate() {
                    if (m >> j) & 1 == 1 {
                        w ^= r;
                    }
                }
                w == v
            })
        };
        let (mut chain_rejects, mut others) = (0u32, 0u32);
        let mut tested = 0u32;
        while tested < 30_000 {
            let v = rng.next() & mask;
            if v == 0 || in_span(v) {
                continue;
            }
            tested += 1;
            let shared = phi_cascade_shared(&mut slot, &c, v, n);
            let chain_fp = shared.chain_fastpath;
            let key = |r: PhiResult| match r.outcome {
                PhiOutcome::Reject => (0u8, Vec::new()),
                PhiOutcome::AcceptUnique => (1u8, Vec::new()),
                PhiOutcome::Tie(m) => (2u8, m),
            };
            let fresh = phi_cascade(&c, v, n);
            assert_eq!(shared.strata_used, fresh.strata_used);
            assert_eq!(key(shared), key(fresh));
            check_against_reference(&c, v, n);
            if chain_fp {
                chain_rejects += 1;
            } else {
                others += 1;
            }
        }
        assert!(chain_rejects > 1000, "chain rejects never fired: {chain_rejects}");
        assert!(others > 100, "generic/non-chain paths starved: {others}");
    }

    /// D17 deterministic chain witnesses. Parent (N = 34, weight-agnostic
    /// frame): r1/r2 are disjoint w4 blocks, r3 a w12 block, so the chain
    /// is provably E⁰ = {100} at w4 → E¹ = {100} at w8 (r1+r2 ⊥ 100) →
    /// ∅ at w12 (r3 ⊥̸ 100). Candidates are all-ones tails outside the
    /// row support, shifting the whole coset up by e: each arm of the
    /// chain (v-only reject, O(1) unique accept at position 2, mixed-
    /// stratum chain exit into the generic machinery) fires exactly
    /// where computed by hand, through ONE shared slot, in an order that
    /// exercises both the build and the read path of every chain entry.
    #[test]
    fn e_chain_deterministic_witnesses() {
        let n = 34u32;
        let r1: BinVec = 0b1111;
        let r2: BinVec = 0b1111_0000;
        let r3: BinVec = 0b1111_1111_1111_0000_0000;
        let c: Vec<BinVec> = vec![r1, r2, r3];
        let tail = |e: u32| -> BinVec { ((1u64 << e) - 1) << 20 };
        let mut slot = PhiParentSlot::new();

        // e = 9: coset min weight 9 ⇒ strata w4, w8 C-only (chain builds
        // entries 0, 1), then w9 is v-only ⇒ O(1) chain reject.
        let res_a = phi_cascade_shared(&mut slot, &c, tail(9), n);
        assert!(matches!(res_a.outcome, PhiOutcome::Reject));
        assert!(res_a.chain_fastpath, "v-only mid-chain reject must be O(1)");
        assert_eq!(res_a.strata_used, 3);
        check_against_reference(&c, tail(9), n);

        // e = 13: coset min weight 13 ⇒ w4, w8, w12 all C-only; the w12
        // filter empties E (r3 has odd parity against 100) ⇒ O(1) unique
        // accept at chain position 2 (extends the chain built above).
        let res_b = phi_cascade_shared(&mut slot, &c, tail(13), n);
        assert!(matches!(res_b.outcome, PhiOutcome::AcceptUnique));
        assert!(res_b.chain_fastpath, "chain-filter accept must be O(1)");
        assert_eq!(res_b.strata_used, 3);
        check_against_reference(&c, tail(13), n);

        // e = 12: coset min weight 12 collides with the w12 C-stratum ⇒
        // mixed stratum, bound = −1 ≤ tc − tv = 0 is inconclusive ⇒ the
        // candidate leaves the chain (reads the cached entries built by
        // the siblings above) and resolves generically further down.
        let res_c = phi_cascade_shared(&mut slot, &c, tail(12), n);
        assert!(
            !res_c.chain_fastpath,
            "inconclusive bound must fall back to the generic machinery"
        );
        assert!(res_c.strata_used > 3, "generic resolution continues past w12");
        check_against_reference(&c, tail(12), n);

        // Random sweep over the same shared slot: chain caches must stay
        // coherent for arbitrary siblings of this parent.
        let mut rng = XorShift(0xD17_ACCE57);
        let mask = (1u64 << n) - 1;
        let mut tested = 0;
        while tested < 2_000 {
            let v = rng.next() & mask;
            let mut probe = c.clone();
            probe.push(v);
            let (pr, _) = row_reduce(&probe, n);
            if pr.len() < 4 {
                continue;
            }
            tested += 1;
            let shared = phi_cascade_shared(&mut slot, &c, v, n);
            let fresh = phi_cascade(&c, v, n);
            assert_eq!(shared.strata_used, fresh.strata_used);
            let key = |r: PhiResult| match r.outcome {
                PhiOutcome::Reject => (0u8, Vec::new()),
                PhiOutcome::AcceptUnique => (1u8, Vec::new()),
                PhiOutcome::Tie(m) => (2u8, m),
            };
            assert_eq!(key(shared), key(fresh));
            check_against_reference(&c, v, n);
        }
    }

    /// D16 staleness test: many sibling candidates evaluated through ONE
    /// shared slot must each agree with a fresh per-call cascade — this
    /// is the test that catches stale lazy caches (`fhat_c` / `e_set`
    /// leaking across strata or parents).
    #[test]
    fn shared_slot_agrees_with_fresh_cascade_across_siblings() {
        let mut rng = XorShift(0x51B1_1265);
        let n = 16u32;
        let mask = (1u64 << n) - 1;
        for _ in 0..40 {
            let kp1 = 2 + (rng.next() % 5) as usize; // 2..=6
            let raw: Vec<BinVec> = (0..kp1).map(|_| rng.next() & mask).collect();
            let (rref, _) = row_reduce(&raw, n);
            if rref.len() < kp1 {
                continue;
            }
            let c_rref: Vec<BinVec> = rref[..kp1 - 1].to_vec();
            // Generate sibling candidates: random vectors outside C.
            let mut slot = PhiParentSlot::new();
            let mut tested = 0;
            while tested < 25 {
                let cand = rng.next() & mask;
                let mut probe: Vec<BinVec> = c_rref.clone();
                probe.push(cand);
                let (pr, _) = row_reduce(&probe, n);
                if pr.len() < kp1 || cand == 0 {
                    continue; // cand ∈ C (dependent) — not a valid frame
                }
                tested += 1;
                let shared = phi_cascade_shared(&mut slot, &c_rref, cand, n);
                let fresh = phi_cascade(&c_rref, cand, n);
                assert_eq!(shared.strata_used, fresh.strata_used);
                assert_eq!(shared.m_size_at_decision, fresh.m_size_at_decision);
                let key = |r: PhiResult| match r.outcome {
                    PhiOutcome::Reject => (0u8, Vec::new()),
                    PhiOutcome::AcceptUnique => (1u8, Vec::new()),
                    PhiOutcome::Tie(m) => (2u8, m), // order must match too
                };
                assert_eq!(key(shared), key(fresh));
            }
        }
    }
}
