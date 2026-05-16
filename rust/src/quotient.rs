//! `Q_C := C⊥ / C` operations.
//!
//! Direct port of `doubly_even.enumerate.quotient.{Q_basis, lift, project,
//! aut_image_on_Q}`. The kernel receives `C`'s RREF basis + pivots and
//! `C.dual().basis` from the Python side (Python's
//! `Code.rref_basis()` / `Code.dual()` are the readable reference); the
//! kernel re-derives `V_basis` and `pivots_V` so callers don't have to
//! marshal them.

use crate::linalg::{apply_permutation, row_reduce};
use crate::types::{BinVec, ColPerm, Mat};

/// Reduce a vector `v` modulo a code given by `(rref, pivots)`.
///
/// Clears every bit of `v` at a pivot column of `C`. The result is the
/// unique coset representative of `v + C` whose bits at `C`'s pivots are
/// all zero.
///
/// Inlined hot helper — used by [`q_basis`] and [`aut_image_on_q`].
#[inline]
pub fn reduce_mod_c(mut v: BinVec, rref: &[BinVec], pivots: &[u32]) -> BinVec {
    for (row, &p) in rref.iter().zip(pivots.iter()) {
        if (v >> p) & 1 == 1 {
            v ^= *row;
        }
    }
    v
}

/// Return `(V_basis, pivots_V)` for `Q_C = C⊥ / C`.
///
/// `V_basis` is the RREF basis of the canonical-rep subspace
/// `V = C⊥ ∩ {v : v_p = 0 ∀ p ∈ pivots(C)}`; `pivots_V[i]` is the leading-1
/// column of `V_basis[i]`. `pivots_V[i] ∈ [0, n) \ pivots(C)` by construction.
///
/// Precondition: `C ⊆ C⊥` (always holds on the doubly even augmentation tree).
pub fn q_basis(
    rref: &[BinVec],
    pivots: &[u32],
    dual_basis: &[BinVec],
    n: u32,
) -> (Vec<BinVec>, Vec<u32>) {
    let mut reduced: Vec<BinVec> = Vec::with_capacity(dual_basis.len());
    for &v in dual_basis {
        let rep = reduce_mod_c(v, rref, pivots);
        if rep != 0 {
            reduced.push(rep);
        }
    }
    row_reduce(&reduced, n)
}

/// Map a `Q`-coordinate `u_q` back to an `F_2^N` representative.
///
/// `out = XOR_{i: bit i of u_q is set} V_basis[i]`.
#[inline]
pub fn lift(u_q: BinVec, v_basis: &[BinVec]) -> BinVec {
    let mut out: BinVec = 0;
    let mut u = u_q;
    while u != 0 {
        let i = u.trailing_zeros() as usize;
        out ^= v_basis[i];
        u &= u - 1;
    }
    out
}

/// Project an `F_2^N` vector already lying in `V` to its `Q`-coordinate.
///
/// `V_basis` is in RREF, so bit `i` of the output is the bit of `v_in_v` at
/// column `pivots_v[i]`. Callers must guarantee `v_in_v ∈ V`; we don't check.
#[inline]
pub fn project(v_in_v: BinVec, pivots_v: &[u32]) -> BinVec {
    let mut out: BinVec = 0;
    for (i, &p) in pivots_v.iter().enumerate() {
        if (v_in_v >> p) & 1 == 1 {
            out |= 1u64 << i;
        }
    }
    out
}

/// Image of each automorphism generator in `End(Q_C)`.
///
/// For each `σ` in `aut_generators` returns a length-`L` column-form matrix
/// `sigma_q` where `sigma_q[i]` is the `Q`-coordinate of `σ(V_basis[i])`
/// reduced mod `C`. Applying `sigma_q` to a `Q`-coordinate `u` is then
/// `XOR_{i: u_i = 1} sigma_q[i]`.
pub fn aut_image_on_q(
    aut_generators: &[ColPerm],
    rref: &[BinVec],
    pivots: &[u32],
    v_basis: &[BinVec],
    pivots_v: &[u32],
) -> Vec<Mat> {
    let mut out: Vec<Mat> = Vec::with_capacity(aut_generators.len());
    let l = v_basis.len();
    for sigma in aut_generators {
        let mut sigma_q: Mat = Vec::with_capacity(l);
        for &b in v_basis {
            let permuted = apply_permutation(b, sigma);
            let reduced = reduce_mod_c(permuted, rref, pivots);
            sigma_q.push(project(reduced, pivots_v));
        }
        out.push(sigma_q);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: `project(lift(u, V_basis), pivots_V) == u`.
    #[test]
    fn project_lift_round_trip() {
        // Any RREF basis with strictly increasing pivots works here.
        // We pin a small one: dim 3 in F_2^5, pivots at columns 0, 2, 3.
        let v_basis: Vec<BinVec> = vec![
            0b00001, // bit 0
            0b00100, // bit 2
            0b01000, // bit 3
        ];
        let pivots_v: Vec<u32> = vec![0, 2, 3];
        for u in 0u64..(1 << 3) {
            let lifted = lift(u, &v_basis);
            let reprojected = project(lifted, &pivots_v);
            assert_eq!(reprojected, u, "round trip failed at u={u:b}");
        }
    }

    /// `lift(0, _) == 0` and `lift(u, single) == single` when popcount(u) == 1.
    #[test]
    fn lift_single_bits() {
        let v_basis: Vec<BinVec> = vec![0b00101, 0b11000, 0b00010];
        assert_eq!(lift(0, &v_basis), 0);
        assert_eq!(lift(0b001, &v_basis), 0b00101);
        assert_eq!(lift(0b010, &v_basis), 0b11000);
        assert_eq!(lift(0b100, &v_basis), 0b00010);
        // Two bits → XOR of the corresponding rows.
        assert_eq!(lift(0b011, &v_basis), 0b00101 ^ 0b11000);
        assert_eq!(lift(0b111, &v_basis), 0b00101 ^ 0b11000 ^ 0b00010);
    }

    /// `reduce_mod_c(c, _) == 0` for every `c ∈ C`. For `[I_2 | 11]`,
    /// `C = {000, 101, 110, 011}`; the pivots are 0 and 1.
    #[test]
    fn reduce_mod_c_kills_codewords() {
        let rref: Vec<BinVec> = vec![0b101, 0b110]; // pivots 0, 1
        let pivots: Vec<u32> = vec![0, 1];
        for &c in &[0b000u64, 0b101, 0b110, 0b011] {
            assert_eq!(reduce_mod_c(c, &rref, &pivots), 0);
        }
        // A non-codeword reduces to its unique pivot-free coset rep.
        let v = 0b100; // = pivot-1 codeword + bit 2; mod C → bit 2 only
        // Actually 0b100 has bit 2 set, no pivot bits, so already reduced.
        assert_eq!(reduce_mod_c(v, &rref, &pivots), 0b100);
    }

    /// `[8,4]` extended Hamming (self-dual, doubly even) — the canonical
    /// small doubly even code. We hard-code its RREF, dual, and confirm
    /// Q_basis is empty since C = C⊥ ⇒ dim V = n - 2k = 0.
    #[test]
    fn q_basis_extended_hamming_8_4_is_empty() {
        // Extended [8,4] Hamming, self-dual:
        // rows: 1 0 0 0 0 1 1 1
        //       0 1 0 0 1 0 1 1
        //       0 0 1 0 1 1 0 1
        //       0 0 0 1 1 1 1 0
        // Bit 0 = column 0 = LSB.
        // Pivots: 0, 1, 2, 3.
        let rref: Vec<BinVec> = vec![
            0b1110_0001, // c0 = 1 + c5 + c6 + c7
            0b1101_0010, // c1 = 1 + c4 + c6 + c7  (bit 1 is the LSB after?)
            0b1011_0100, // c2 = ...
            0b0111_1000, // c3 = ...
        ];
        let pivots: Vec<u32> = vec![0, 1, 2, 3];
        // For a self-dual code C = C⊥, so dual_basis ⊆ C and every entry
        // reduces to 0; V_basis is empty.
        let dual_basis = rref.clone();
        let (v_basis, pivots_v) = q_basis(&dual_basis, &pivots, &dual_basis, 8);
        assert!(v_basis.is_empty(), "expected empty V, got {v_basis:?}");
        assert!(pivots_v.is_empty(), "expected empty pivots_V, got {pivots_v:?}");
    }

    /// `aut_image_on_q` for an identity automorphism preserves V-basis identity.
    #[test]
    fn aut_image_of_identity_is_identity() {
        // Same V_basis as the round-trip test.
        let v_basis: Vec<BinVec> = vec![0b00001, 0b00100, 0b01000];
        let pivots_v: Vec<u32> = vec![0, 2, 3];
        // C is trivial (zero code) for simplicity: empty rref, empty pivots.
        let rref: Vec<BinVec> = vec![];
        let pivots: Vec<u32> = vec![];
        // identity permutation on 5 columns
        let identity_perm: ColPerm = (0..5u32).collect();
        let aut_generators = vec![identity_perm];
        let images = aut_image_on_q(&aut_generators, &rref, &pivots, &v_basis, &pivots_v);
        assert_eq!(images.len(), 1);
        // Image of e_0, e_1, e_2 under identity is the L×L identity.
        let identity_mat: Mat = vec![0b001, 0b010, 0b100];
        assert_eq!(images[0], identity_mat);
    }

    /// Swap permutation on the two free columns of `[2,1]` code C = {00, 11}.
    /// `V_basis = (0b10,)` (one free column, bit 1), `pivots_V = (1,)`.
    /// Swapping columns 0 and 1 in F_2^2 should send `b = 0b10` to `0b01`,
    /// which reduces mod C (XOR with `0b11`) to `0b10`. So σ_Q is the
    /// identity on Q.
    #[test]
    fn aut_image_swap_columns_of_repetition_code() {
        let rref: Vec<BinVec> = vec![0b11]; // C = ⟨11⟩
        let pivots: Vec<u32> = vec![0];
        let v_basis: Vec<BinVec> = vec![0b10]; // V = ⟨e_1⟩
        let pivots_v: Vec<u32> = vec![1];
        // swap columns 0 and 1
        let swap: ColPerm = vec![1, 0];
        let images = aut_image_on_q(&vec![swap], &rref, &pivots, &v_basis, &pivots_v);
        assert_eq!(images, vec![vec![0b1]]); // 1x1 identity
    }
}
