//! Top-level orchestrator: `doubly_even_candidates_q`.
//!
//! Direct port of `doubly_even.enumerate.quotient.doubly_even_candidates_Q`.
//! One fat call per parent during the canonical-augmentation recursion.

use crate::orbit::{aut_orbit_minima_q_table, aut_orbit_minima_q_witt, singular_reps_q};
use crate::quotient::{aut_image_on_q, lift, q_basis};
use crate::types::{BinVec, ColPerm, Mat};

/// σ_Q sub-phase accumulator (`phase_timers` builds only; post-D15
/// profiling, plan `last-session-we-had-sequential-fiddle.md` § 1.3).
/// The generic pipeline below is one fat call per parent; the five
/// stages are timed with `Instant` pairs (~25 ns each against a ~70 µs
/// call) into a per-thread accumulator that the enumerate driver drains
/// immediately after every call, so attribution never crosses calls or
/// threads. The k=0 / k=1 closed-form fast paths are NOT timed — their
/// cost is negligible and lands only in the aggregate
/// `stats_candidates_q_ns`, so sum(sub-phases) ≤ aggregate by design.
#[cfg(feature = "phase_timers")]
pub mod phase_timers {
    use std::cell::Cell;

    pub const N_PHASES: usize = 5;
    pub const PHASE_NAMES: [&str; N_PHASES] = [
        "q_basis",
        "aut_image_on_q",
        "singular_reps_q",
        "orbit_min",
        "lift_sort",
    ];

    thread_local! {
        static CQ_PHASE_NS: Cell<[u64; N_PHASES]> = const { Cell::new([0; N_PHASES]) };
    }

    #[inline]
    pub(crate) fn add(idx: usize, ns: u64) {
        CQ_PHASE_NS.with(|c| {
            let mut a = c.get();
            a[idx] += ns;
            c.set(a);
        });
    }

    /// Return-and-zero the per-thread accumulator.
    pub fn drain() -> [u64; N_PHASES] {
        CQ_PHASE_NS.with(|c| c.replace([0; N_PHASES]))
    }
}

/// Wrap one pipeline stage in an `Instant` pair under `phase_timers`;
/// compiles to the bare expression otherwise.
macro_rules! cq_timed {
    ($idx:expr, $e:expr) => {{
        #[cfg(feature = "phase_timers")]
        {
            let t0 = std::time::Instant::now();
            let r = $e;
            phase_timers::add($idx, t0.elapsed().as_nanos() as u64);
            r
        }
        #[cfg(not(feature = "phase_timers"))]
        {
            $e
        }
    }};
}

/// `Aut(C)`-orbit reps of doubly-even 1-dim extensions of `C`, returned
/// sorted as `F_2^N` integers.
///
/// Inputs mirror the Python call shape exactly (`Code` becomes
/// `(code_rref, pivots)` plus `dual_basis`; `aut_generators` is a slice
/// of column permutations).
///
/// Pipeline:
///
/// 1. Build `(V_basis, pivots_V)` from `(rref, pivots, dual_basis)`.
/// 2. Image each `Aut(C)` generator into `End(Q_C)`.
/// 3. Enumerate singular `Q`-coords (lift weight `≡ 0 mod 4`).
/// 4. Orbit-min decompose in `Q`.
/// 5. Lift survivors back to `F_2^N` and sort.
pub fn doubly_even_candidates_q(
    n: u32,
    code_rref: &[BinVec],
    pivots: &[u32],
    dual_basis: &[BinVec],
    aut_generators: &[ColPerm],
) -> Vec<BinVec> {
    if pivots.is_empty() {
        // Aut(zero code) = S_N: weight is a complete invariant, so the only
        // canonical doubly-even k=1 extensions are `(1 << 4ℓ) - 1` for
        // ℓ = 1, ..., ⌊N/4⌋. Skip the 2^N Q-walk + S_N orbit BFS.
        return (1..=n / 4)
            .map(|l| (1u64 << (4 * l)) - 1)
            .collect();
    }
    if pivots.len() == 1 {
        let v = code_rref[0];
        let w = v.count_ones();
        if w >= 4 && w % 4 == 0 && v == (1u64 << w) - 1 {
            // Young-subgroup parent `⟨(1)^{4ℓ}(0)^{N−4ℓ}⟩`: every rank-1
            // node reached by the recursion flows through the k=0 fast-path,
            // so the basis vector here is the all-ones prefix.
            // `Aut = S_{4ℓ} × S_{N−4ℓ}` makes the weight pair
            // `(wt_a, wt_b)` a complete orbit invariant of doubly-even
            // k=2 extensions, modulo the quotient `(wt_a, wt_b) ~ (4ℓ−wt_a, wt_b)`.
            return young_subgroup_k2_reps(n, w);
        }
    }
    let (v_basis, pivots_v) = cq_timed!(0, q_basis(code_rref, pivots, dual_basis, n));
    let sigma_qs = cq_timed!(
        1,
        aut_image_on_q(aut_generators, code_rref, pivots, &v_basis, &pivots_v)
    );
    let l = v_basis.len() as u32;
    let reps_q = cq_timed!(2, singular_reps_q(&v_basis));
    let orbit_min = cq_timed!(
        3,
        if use_witt_path(&sigma_qs, l) {
            aut_orbit_minima_q_witt(&reps_q, &sigma_qs, l)
        } else {
            aut_orbit_minima_q_table(&reps_q, &sigma_qs, l)
        }
    );
    cq_timed!(4, {
        let mut out: Vec<BinVec> = orbit_min.iter().map(|&u| lift(u, &v_basis)).collect();
        out.sort_unstable();
        out
    })
}

/// [`doubly_even_candidates_q`] with the σ_Q heavy stages (singular-rep
/// Gray walk + orbit-min BFS) fanned out on the seeder helper pool when
/// the quotient dimension clears `pool.min_l` (D16 lever B). Output is
/// identical to the sequential pipeline — see the determinism notes on
/// [`crate::orbit::aut_orbit_minima_q_witt_pooled`] and
/// [`crate::orbit::singular_reps_q_pooled`]. Only the parallel seeder
/// calls this; workers stay on the sequential entry (they are already
/// saturated).
#[cfg(feature = "parallel")]
pub fn doubly_even_candidates_q_pooled(
    n: u32,
    code_rref: &[BinVec],
    pivots: &[u32],
    dual_basis: &[BinVec],
    aut_generators: &[ColPerm],
    pool: &crate::seeder_pool::SeederPool,
) -> Vec<BinVec> {
    use crate::orbit::{aut_orbit_minima_q_witt_pooled, singular_reps_q_pooled};

    if pivots.len() <= 1 {
        // k = 0 / Young k = 1 closed forms — no σ_Q pipeline to pool.
        return doubly_even_candidates_q(n, code_rref, pivots, dual_basis, aut_generators);
    }
    let (v_basis, pivots_v) = cq_timed!(0, q_basis(code_rref, pivots, dual_basis, n));
    let sigma_qs = cq_timed!(
        1,
        aut_image_on_q(aut_generators, code_rref, pivots, &v_basis, &pivots_v)
    );
    let l = v_basis.len() as u32;
    let pooled = pool.size() >= 2 && l >= pool.min_l;
    let reps_q = cq_timed!(
        2,
        if pooled {
            singular_reps_q_pooled(&v_basis, pool)
        } else {
            singular_reps_q(&v_basis)
        }
    );
    let orbit_min = cq_timed!(
        3,
        if pooled {
            aut_orbit_minima_q_witt_pooled(&reps_q, &sigma_qs, l, pool)
        } else if use_witt_path(&sigma_qs, l) {
            aut_orbit_minima_q_witt(&reps_q, &sigma_qs, l)
        } else {
            aut_orbit_minima_q_table(&reps_q, &sigma_qs, l)
        }
    );
    cq_timed!(4, {
        let mut out: Vec<BinVec> = orbit_min.iter().map(|&u| lift(u, &v_basis)).collect();
        out.sort_unstable();
        out
    })
}

/// Closed-form `Aut(⟨v_ℓ⟩)`-orbit reps of doubly-even k=2 extensions, where
/// `v_ℓ = (1)^{4ℓ}(0)^{N−4ℓ}`. Both `wt_a`, `wt_b` must be even and
/// `(wt_a + wt_b) ≡ 0 (mod 4)`; the canonical rep takes `wt_a ≤ 2ℓ` to fold
/// the `w ↔ v_ℓ ⊕ w` quotient. Each emitted rep is then `⊕ v_ℓ`-flipped
/// to standard form (bit 0 cleared) so the output lies in the same
/// `pivots = [0]`-cleared subspace as `singular_reps_q ∘ lift`.
fn young_subgroup_k2_reps(n: u32, four_l: u32) -> Vec<BinVec> {
    let two_l = four_l / 2;
    let nb = n - four_l;
    let v_l: BinVec = (1u64 << four_l) - 1;
    let mut out: Vec<BinVec> = Vec::new();
    let mut wa = 0u32;
    while wa <= two_l {
        let mut wb = 0u32;
        while wb <= nb {
            if (wa != 0 || wb != 0) && (wa + wb) % 4 == 0 {
                let mut w = ((1u64 << wa) - 1) | (((1u64 << wb) - 1) << four_l);
                if w & 1 != 0 {
                    w ^= v_l;
                }
                out.push(w);
            }
            wb += 2;
        }
        wa += 2;
    }
    out.sort_unstable();
    out
}

/// Pick the structural witt-path BFS over the `2^L` σ_Q lookup table.
///
/// In Rust the per-step cost of the `mat_apply` bit-walk inside the
/// witt BFS is ~2× cheaper than building the per-generator `2^L` table
/// and walking it. Measured at N ∈ {18, 20, 22} via
/// `scripts/experimental/bench_witt_profile.py` (see `04-optimisations.md` §D13):
/// mean `doubly_even_candidates_q` latency 245 → 116 µs at N=22,
/// 1.08–1.11× total wall reduction. Phase (b) wins at every benched
/// `(N, L)`, so dispatch is unconditional — no `L` threshold needed.
///
/// Note: this is the *inverted* finding from CPython phase-(b), where
/// the table beat the bit-walk by 13–15 % (see D7). The flip happens
/// because Rust closes the per-step interpreter overhead that made the
/// table's precompute-once model dominate in CPython.
fn use_witt_path(sigma_qs: &[Mat], l: u32) -> bool {
    let _ = sigma_qs;
    let _ = l;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permutations::dual_basis as compute_dual_basis;

    /// Dump REAL σ_Q inputs `(n, k, l, v_basis, sigma_qs)` for every
    /// rank-2/3 parent at N = 26, 27 — the seeder-exclusive ranks whose
    /// orbit-min BFS dominates the parallel seeder span (timeline capture
    /// 2026-06-10). The `orbit_probe` microbench replays these instead of
    /// random-GL synthetics: real aut groups are small permutation-induced
    /// subgroups with many tiny orbits, and restructure verdicts on
    /// random-GL inputs (few giant orbits) would not transfer.
    ///
    /// Not a test of anything — quarantined behind `#[ignore]` so the
    /// suite stays fast. Run explicitly:
    ///
    /// ```sh
    /// cargo test --release dump_sigma_inputs -- --ignored --nocapture
    /// ```
    ///
    /// Output: one line-oriented text file per parent under
    /// `scripts/bench-results/sigma-inputs/` (gitignored). `reps_q` is
    /// NOT dumped — the replay regenerates it with its `singular_reps_q`
    /// clone, keeping files < 1 KB each.
    ///
    /// σ(N, k) and N! constants from `doubly_even.spec.mass.gaborit_sigma`.
    #[test]
    #[ignore]
    fn dump_sigma_inputs_n26_n27() {
        use std::io::Write;

        let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../scripts/bench-results/sigma-inputs");
        std::fs::create_dir_all(&out_dir).expect("create sigma-inputs dir");

        let cases: [(u32, [u128; 4], u128); 2] = [
            (
                26,
                [1, 16_777_215, 23_456_241_068_715, 3_513_661_139_803_975_875],
                403_291_461_126_605_635_584_000_000,
            ),
            (
                27,
                [1, 33_550_335, 93_790_621_315_755, 28_085_293_383_818_511_555],
                10_888_869_450_418_352_160_768_000_000,
            ),
        ];
        for (n, sigma, fact) in cases {
            let (out, _stats, _per_k) =
                crate::enumerate::enumerate_doubly_even(n, 3, sigma.to_vec(), fact);
            let mut idx_by_k = [0usize; 4];
            for e in &out {
                let k = e.rref.len();
                if k < 2 {
                    continue;
                }
                let (rref, pivots) = crate::linalg::row_reduce(&e.rref, n);
                debug_assert_eq!(rref, e.rref, "emitted basis must already be RREF");
                let dual = crate::permutations::dual_basis(&rref, &pivots, n);
                let (v_basis, pivots_v) = crate::quotient::q_basis(&rref, &pivots, &dual, n);
                let sigma_qs = crate::quotient::aut_image_on_q(
                    &e.aut_generators,
                    &rref,
                    &pivots,
                    &v_basis,
                    &pivots_v,
                );
                let l = v_basis.len();
                assert_eq!(l as u32, n - 2 * k as u32, "L != N - 2k");
                let idx = idx_by_k[k];
                idx_by_k[k] += 1;
                let path = out_dir.join(format!("n{n}-k{k}-p{idx:03}.txt"));
                let mut f = std::fs::File::create(&path).expect("create dump file");
                writeln!(f, "n {n}").unwrap();
                writeln!(f, "k {k}").unwrap();
                writeln!(f, "l {l}").unwrap();
                writeln!(f, "aut_order {}", e.aut_order).unwrap();
                let vb: Vec<String> = v_basis.iter().map(|w| format!("{w:x}")).collect();
                writeln!(f, "v_basis {}", vb.join(" ")).unwrap();
                writeln!(f, "gens {}", sigma_qs.len()).unwrap();
                for g in &sigma_qs {
                    let cols: Vec<String> = g.iter().map(|w| format!("{w:x}")).collect();
                    writeln!(f, "gen {}", cols.join(" ")).unwrap();
                }
            }
            println!(
                "N={n}: dumped {} rank-2 + {} rank-3 parents",
                idx_by_k[2], idx_by_k[3]
            );
        }
    }

    /// Run the post-fast-path generic pipeline directly.
    fn generic_candidates(
        n: u32,
        rref: &[BinVec],
        pivots: &[u32],
        dual: &[BinVec],
        aut_gens: &[ColPerm],
    ) -> Vec<BinVec> {
        let (v_basis, pivots_v) = crate::quotient::q_basis(rref, pivots, dual, n);
        let sigma_qs = crate::quotient::aut_image_on_q(aut_gens, rref, pivots, &v_basis, &pivots_v);
        let l = v_basis.len() as u32;
        let reps_q = crate::orbit::singular_reps_q(&v_basis);
        let orbit_min = crate::orbit::aut_orbit_minima_q_witt(&reps_q, &sigma_qs, l);
        let mut out: Vec<BinVec> = orbit_min.iter().map(|&u| crate::quotient::lift(u, &v_basis)).collect();
        out.sort_unstable();
        out
    }

    /// Young-subgroup `S_{4ℓ} × S_{N−4ℓ}` generators: swap-in-block + cyclic-in-block
    /// for each factor. Sufficient to generate `S_m` whenever `m ≥ 2`.
    fn young_gens(n: u32, four_l: u32) -> Vec<ColPerm> {
        let mut gens: Vec<ColPerm> = Vec::new();
        if four_l >= 2 {
            let mut swap: Vec<u32> = (0..n).collect();
            swap.swap(0, 1);
            gens.push(swap);
            if four_l >= 3 {
                let cyc: Vec<u32> = (0..n).map(|i| if i < four_l { (i + 1) % four_l } else { i }).collect();
                gens.push(cyc);
            }
        }
        let m = n - four_l;
        if m >= 2 {
            let mut swap: Vec<u32> = (0..n).collect();
            swap.swap(four_l as usize, (four_l + 1) as usize);
            gens.push(swap);
            if m >= 3 {
                let cyc: Vec<u32> = (0..n)
                    .map(|i| if i >= four_l { four_l + (i - four_l + 1) % m } else { i })
                    .collect();
                gens.push(cyc);
            }
        }
        gens
    }

    /// Canonical `Aut(⟨v_ℓ⟩) × ⟨v_ℓ⟩`-orbit label of a k=2 extension `w`:
    /// `(min(wt_a, 4ℓ−wt_a), wt_b)`. Two w's share a label iff they generate
    /// the same Aut(⟨v_ℓ⟩)-orbit (folding the `w ↔ v_ℓ ⊕ w` quotient).
    fn orbit_label(w: BinVec, four_l: u32) -> (u32, u32) {
        let mask: BinVec = (1u64 << four_l) - 1;
        let wa = (w & mask).count_ones();
        let wb = (w >> four_l).count_ones();
        let wa = wa.min(four_l - wa);
        (wa, wb)
    }

    /// For every Young-subgroup parent at every reachable `(N, ℓ)`,
    /// the fast-path's emitted set of candidates must cover the same
    /// `Aut(⟨v_ℓ⟩)`-orbits (modulo the `⟨v_ℓ⟩`-quotient) as the generic
    /// pipeline — i.e. the multisets of orbit labels match. The fast-path
    /// picks a different in-orbit representative than the generic
    /// V_basis-driven lift, which is functionally equivalent (the
    /// downstream canonical-augmentation test is independent of the
    /// in-orbit representative choice).
    #[test]
    fn k2_young_fast_path_matches_generic_pipeline() {
        for &n in &[8u32, 12, 16, 20, 22] {
            for four_l in (4..=n).step_by(4) {
                let v_l: BinVec = (1u64 << four_l) - 1;
                let rref: Vec<BinVec> = vec![v_l];
                let pivots: Vec<u32> = vec![0];
                let dual = compute_dual_basis(&rref, &pivots, n);
                let gens = young_gens(n, four_l);
                let fast = doubly_even_candidates_q(n, &rref, &pivots, &dual, &gens);
                let slow = generic_candidates(n, &rref, &pivots, &dual, &gens);
                assert_eq!(
                    fast.len(), slow.len(),
                    "candidate count mismatch at N={}, 4ℓ={}: fast={}, slow={}",
                    n, four_l, fast.len(), slow.len(),
                );
                let mut fast_labels: Vec<_> = fast.iter().map(|&w| orbit_label(w, four_l)).collect();
                let mut slow_labels: Vec<_> = slow.iter().map(|&w| orbit_label(w, four_l)).collect();
                fast_labels.sort();
                slow_labels.sort();
                assert_eq!(
                    fast_labels, slow_labels,
                    "orbit labels mismatch at N={}, 4ℓ={}: fast={:?}, slow={:?}",
                    n, four_l, fast, slow
                );
            }
        }
    }


    /// `C = ⟨11⟩` in `F_2^2` (`[2, 1]` even-weight code) has `C⊥ = C`, so the
    /// quotient is trivial and there are no doubly-even augmentations.
    #[test]
    fn no_candidates_for_self_dual_repetition_code() {
        let rref: Vec<BinVec> = vec![0b11];
        let pivots: Vec<u32> = vec![0];
        let dual_basis: Vec<BinVec> = vec![0b11];
        let aut_gens: Vec<ColPerm> = vec![vec![1, 0]];
        let out = doubly_even_candidates_q(2, &rref, &pivots, &dual_basis, &aut_gens);
        assert!(out.is_empty());
    }

    /// `C = {0}` in `F_2^4` (zero code): every weight-4 vector is a
    /// doubly-even augmentation; the only one is `0b1111` itself.
    /// `Aut({0}) = S_4` so all weight-4 vectors are a single orbit.
    #[test]
    fn zero_code_n4_yields_single_weight4_rep() {
        let rref: Vec<BinVec> = vec![];
        let pivots: Vec<u32> = vec![];
        // C.dual() = F_2^4 so dual_basis is the four unit vectors.
        let dual_basis: Vec<BinVec> = vec![1, 2, 4, 8];
        // Aut(zero code) = S_4. Two generators suffice: swap(0,1) and cyclic.
        let aut_gens: Vec<ColPerm> = vec![
            vec![1, 0, 2, 3], // swap columns 0 and 1
            vec![1, 2, 3, 0], // cyclic shift
        ];
        let out = doubly_even_candidates_q(4, &rref, &pivots, &dual_basis, &aut_gens);
        assert_eq!(out, vec![0b1111]);
    }
}
