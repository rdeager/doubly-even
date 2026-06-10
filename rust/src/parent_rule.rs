//! D15 coset-spectrum parent rule: the φ cascade and its tie-break.
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
//! all: one Gray-code sweep of `D`'s `2^(k+1)` codewords plus a
//! Walsh–Hadamard transform per weight stratum, evaluated lazily until
//! the lex comparison resolves. Canon is paid only on accepts (needed
//! anyway for `Aut(D)` / mass / recursion) and on exact ties.
//!
//! ### Evaluation frame
//!
//! All φ work happens in the fixed frame basis `[C's k RREF rows, v]`
//! (NOT `d_rref`): the candidate's own hyperplane is then the kernel of
//! the last-coordinate functional `u_C = 1 << k`, and coordinate vectors
//! are plain `u16` indices. Candidate generation guarantees `v ∉ C`
//! (nonzero coset reps), so the frame rows are independent.

use std::cell::RefCell;

use crate::linalg::{apply_permutation, row_reduce};
use crate::types::BinVec;

/// Default child-rank cap for the φ cascade. The per-candidate WHT costs
/// `(k+1)·2^(k+1)` adds, which passes the ~30 µs canon call it replaces
/// around `k+1 ≈ 14–16`; children of rank ≥ 14 only exist at `N ≥ 30`,
/// where falling back to the legacy rule above the cap is the right
/// trade. Per-rank rule mixing is sound: rank is iso-invariant and
/// McKay's induction is local to one child rank.
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
/// Walsh–Hadamard transform (cost `(k+1)·2^(k+1)` adds); at or below it,
/// per-functional direct parity counting over the stratum members is
/// cheaper. Crossover is flat in the 32–128 range; 64 measured fine.
const DIRECT_THRESHOLD: usize = 64;

/// Outcome of the φ cascade for one candidate augmentation `(C, v)`.
pub(crate) enum PhiOutcome {
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

pub(crate) struct PhiResult {
    pub(crate) outcome: PhiOutcome,
    /// Number of weight strata evaluated before the cascade resolved.
    pub(crate) strata_used: u32,
    /// `|M|` at the decision point (1 for AcceptUnique; the surviving
    /// tie-set size for Tie; the set size that excluded `u_C` for Reject).
    pub(crate) m_size_at_decision: u32,
}

/// Grow-only per-thread scratch (D13-V4 house pattern, cf.
/// `canon::CanonScratch`): zero heap allocation per call after warmup,
/// except the rare `Tie` return which clones its small argmin set.
#[derive(Default)]
struct PhiScratch {
    /// Frame rows `[C's RREF rows, v]`.
    rows: Vec<BinVec>,
    /// `wt[x]` = Hamming weight of the codeword with coordinate vector
    /// `x` in the frame basis. Indexed by plain (non-Gray) coordinates.
    wt: Vec<u8>,
    /// WHT buffer. `i32` is required: values reach ±2^(k+1) (≤ 65536).
    f: Vec<i32>,
    /// Coordinate vectors counting-sorted by weight (stratum member lists).
    sorted_idx: Vec<u16>,
    /// Running argmin set of functionals.
    m_buf: Vec<u16>,
}

thread_local! {
    static PHI_SCRATCH: RefCell<PhiScratch> = RefCell::new(PhiScratch::default());
}

/// Evaluate the φ cascade for candidate `v` against parent RREF `c_rref`.
pub(crate) fn phi_cascade(c_rref: &[BinVec], v: BinVec, n: u32) -> PhiResult {
    PHI_SCRATCH.with(|cell| phi_cascade_with(&mut cell.borrow_mut(), c_rref, v, n))
}

fn phi_cascade_with(s: &mut PhiScratch, c_rref: &[BinVec], v: BinVec, n: u32) -> PhiResult {
    let kp1 = c_rref.len() + 1;
    debug_assert!(kp1 <= 16, "φ cascade needs k+1 ≤ 16 (u16 coordinate vectors)");
    let size = 1usize << kp1;
    let u_c: u16 = 1 << (kp1 - 1);

    s.rows.clear();
    s.rows.extend_from_slice(c_rref);
    s.rows.push(v);

    // One Gray-code sweep of all 2^(k+1) codewords: weight per coordinate
    // vector + per-weight counts. Same arithmetic volume as the two
    // weight_enum() passes the legacy reject path pays.
    s.wt.clear();
    s.wt.resize(size, 0);
    let mut counts = [0u32; 65];
    counts[0] = 1; // the zero codeword (coordinate 0)
    let mut cur: BinVec = 0;
    for i in 1..size {
        let flip = i.trailing_zeros() as usize;
        cur ^= s.rows[flip];
        let w = cur.count_ones() as usize;
        s.wt[i ^ (i >> 1)] = w as u8;
        counts[w] += 1;
    }
    debug_assert_eq!(
        counts[0], 1,
        "dependent frame rows: candidate v must lie outside C"
    );

    // Counting sort: stratum member lists without per-stratum rescans.
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

    // Lazy lex cascade, strata ascending. (Doubly-even inputs only have
    // strata at multiples of 4; the loop is weight-agnostic so unit tests
    // can exercise arbitrary frames.)
    let mut strata_used = 0u32;
    let mut first = true;
    let n_cap = (n as usize).min(64);
    for w in 1..=n_cap {
        if counts[w] == 0 {
            continue;
        }
        let t_begin = start[w] as usize;
        let t_end = t_begin + counts[w] as usize;
        strata_used += 1;

        let u_c_in = if first {
            first = false;
            filter_first_stratum(s, size, t_begin, t_end, u_c)
        } else if s.m_buf.len() > DIRECT_THRESHOLD {
            filter_by_wht(s, size, t_begin, t_end, u_c)
        } else {
            filter_direct(s, t_begin, t_end, u_c)
        };

        if !u_c_in {
            return PhiResult {
                outcome: PhiOutcome::Reject,
                strata_used,
                m_size_at_decision: s.m_buf.len() as u32,
            };
        }
        if s.m_buf.len() == 1 {
            return PhiResult {
                outcome: PhiOutcome::AcceptUnique,
                strata_used,
                m_size_at_decision: 1,
            };
        }
    }
    PhiResult {
        outcome: PhiOutcome::Tie(s.m_buf.clone()),
        strata_used,
        m_size_at_decision: s.m_buf.len() as u32,
    }
}

/// In-place Walsh–Hadamard transform: `f̂[u] = Σ_x f[x]·(−1)^(u·x)`.
/// For a stratum indicator, `f̂[u] = |T_w| − 2·φ_w(u)`, so minimising
/// `φ_w` over `u` is maximising `f̂`.
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

/// First stratum: argmin over ALL nonzero functionals (u = 0 is excluded
/// from the start — φ(0) ≡ 0 would always win and reject everything).
/// Returns whether `u_c` survived.
fn filter_first_stratum(
    s: &mut PhiScratch,
    size: usize,
    t_begin: usize,
    t_end: usize,
    u_c: u16,
) -> bool {
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

/// Later stratum, large argmin set: fresh WHT, max restricted to `m_buf`.
fn filter_by_wht(
    s: &mut PhiScratch,
    size: usize,
    t_begin: usize,
    t_end: usize,
    u_c: u16,
) -> bool {
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

/// Later stratum, small argmin set: per-functional parity counting over
/// the stratum members (`|M|·|T_w|` ops, cheaper than a WHT here).
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
    let kp1 = c_rref.len() + 1;
    let row_at = |j: usize| -> BinVec {
        if j < c_rref.len() {
            c_rref[j]
        } else {
            v
        }
    };
    let mut best_key: Option<Vec<BinVec>> = None;
    let mut best_rref: Option<Vec<BinVec>> = None;
    for &u in m_set {
        debug_assert_ne!(u, 0, "u = 0 is not a hyperplane functional");
        // Kernel basis of u: for each j ≠ j0 (j0 = lowest set bit of u),
        // the coordinate e_j + u_j·e_{j0} lies in ker(u).
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
}
