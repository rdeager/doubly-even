//! Top-level orchestrator: `doubly_even_candidates_q`.
//!
//! Direct port of `doubly_even.enumerate.quotient.doubly_even_candidates_Q`.
//! One fat call per parent during the canonical-augmentation recursion.

use crate::orbit::{aut_orbit_minima_q_table, aut_orbit_minima_q_witt, singular_reps_q};
use crate::quotient::{aut_image_on_q, lift, q_basis};
use crate::types::{BinVec, ColPerm, Mat};

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
    let (v_basis, pivots_v) = q_basis(code_rref, pivots, dual_basis, n);
    let sigma_qs = aut_image_on_q(aut_generators, code_rref, pivots, &v_basis, &pivots_v);
    let l = v_basis.len() as u32;
    let reps_q = singular_reps_q(&v_basis);
    let orbit_min = if use_witt_path(&sigma_qs, l) {
        aut_orbit_minima_q_witt(&reps_q, &sigma_qs, l)
    } else {
        aut_orbit_minima_q_table(&reps_q, &sigma_qs, l)
    };
    let mut out: Vec<BinVec> = orbit_min.iter().map(|&u| lift(u, &v_basis)).collect();
    out.sort_unstable();
    out
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
