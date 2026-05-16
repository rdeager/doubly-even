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

/// Cheap predicate deciding which orbit-min path to take.
///
/// In Rust the per-step costs of the table-lookup BFS vs. the
/// `mat_apply` bit-walk are much closer than they were in CPython
/// ([[phase-b-empirical-finding]]). The 5(b) criterion benches in
/// `benches/candidates.rs` measure this directly; the dispatch
/// threshold here is set from that data in 5(b)-7.
///
/// Default for 5(b): table path everywhere. A profile-driven `L`
/// threshold can flip the witt branch on later.
fn use_witt_path(sigma_qs: &[Mat], l: u32) -> bool {
    let _ = sigma_qs;
    let _ = l;
    false
}

#[cfg(test)]
mod tests {
    use super::*;

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
