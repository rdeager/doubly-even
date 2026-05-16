//! Singular-vector enumeration and orbit-min decomposition in `Q`-coordinates.
//!
//! Two BFS variants land in this module (5(b)-4), kept side-by-side so the
//! Rust profile, not the inherited Python verdict, picks the default:
//!
//! - [`aut_orbit_minima_q_table`] — Gray-code-initialised `σ_Q` lookup tables,
//!   one list-index per BFS step. This is the post-D6 Python winner.
//! - [`aut_orbit_minima_q_witt`] — apply `σ_Q` via `mat_apply` (popcount-many
//!   XORs); skips the `2^L`-per-generator table build. Lost to the table path
//!   in pure Python ([[phase-b-empirical-finding]]); the open question is
//!   whether per-step bit-walk in Rust is cheap enough to flip the verdict.

use crate::types::{BinVec, Mat};
use fixedbitset::FixedBitSet;

/// Apply a column-form `GL(L, F_2)` matrix to a column vector.
///
/// `out = XOR_{i : bit i of v is set} m[i]`. Bit-walk via
/// `trailing_zeros` + `v & (v - 1)` — strictly cheaper than a shift loop
/// for sparse `v` and equal in the dense limit.
#[inline]
pub fn mat_apply(m: &[BinVec], v: BinVec) -> BinVec {
    let mut out: BinVec = 0;
    let mut u = v;
    while u != 0 {
        let i = u.trailing_zeros() as usize;
        out ^= m[i];
        u &= u - 1;
    }
    out
}

/// True iff `m` is the `L × L` identity in column form (`m[i] == 1 << i`).
#[inline]
fn is_identity_mat(m: &[BinVec]) -> bool {
    for (i, &col) in m.iter().enumerate() {
        if col != (1u64 << i) {
            return false;
        }
    }
    true
}

/// Precompute `table[u] = σ_Q · u` for every `u ∈ [0, 2^L)`.
///
/// Built in Gray-code order — each entry costs one XOR vs. the previous,
/// so total build cost is `2^L` XORs (vs. `2^L · L` for the naïve
/// per-cell computation). The BFS body then applies `σ_Q` via a single
/// indexed load.
///
/// Direct port of `doubly_even.enumerate.quotient._sigma_Q_table`.
pub fn sigma_q_table(sigma_q: &[BinVec], l: u32) -> Vec<BinVec> {
    let l = l as usize;
    let size = 1usize << l;
    let mut table = vec![0u64; size];
    if l == 0 {
        return table;
    }
    let mut val: BinVec = 0;
    let mut u: BinVec = 0;
    for i in 1u64..(size as u64) {
        let flip = i.trailing_zeros() as usize;
        u ^= 1u64 << flip;
        val ^= sigma_q[flip];
        table[u as usize] = val;
    }
    table
}

/// Orbit-min decomposition of `reps_q` under `⟨sigma_qs⟩`, via Gray-code
/// `σ_Q` lookup tables.
///
/// Algorithm: process `reps_q` in sorted order; the first unseen rep is the
/// orbit min, then BFS-expand its orbit with the precomputed tables. Each
/// element of `reps_q` is visited exactly once across all orbits — vs. the
/// naïve per-rep `_is_orbit_min` BFS that visits each element up to
/// `orbit_size` times.
///
/// Precondition: `reps_q` is closed under `⟨sigma_qs⟩`. The caller satisfies
/// this by passing `wt(lift) ≡ 0 (mod 4)` cosets — `Aut(C)` preserves
/// `wt mod 4` on cosets of doubly-even codes.
///
/// `seen` is a `FixedBitSet` of size `2^L`. At `L ≤ 28` (32 MB) this stays
/// in a small range of host RAM; beyond that the table path is memory-
/// infeasible anyway (per-generator table is `2^L · 8` bytes).
pub fn aut_orbit_minima_q_table(
    reps_q: &[BinVec],
    sigma_qs: &[Mat],
    l: u32,
) -> Vec<BinVec> {
    if sigma_qs.is_empty() {
        let mut out = reps_q.to_vec();
        out.sort_unstable();
        return out;
    }
    let tables: Vec<Vec<BinVec>> =
        sigma_qs.iter().map(|s| sigma_q_table(s, l)).collect();
    let mut reps_sorted = reps_q.to_vec();
    reps_sorted.sort_unstable();
    let universe = 1usize << l;
    let mut seen = FixedBitSet::with_capacity(universe);
    let mut minima: Vec<BinVec> = Vec::new();
    let cap = reps_q.len();
    let mut queue: Vec<BinVec> = Vec::with_capacity(cap);
    let mut next: Vec<BinVec> = Vec::with_capacity(cap);
    for &v in &reps_sorted {
        if seen.contains(v as usize) {
            continue;
        }
        minima.push(v);
        seen.insert(v as usize);
        queue.clear();
        queue.push(v);
        while !queue.is_empty() {
            next.clear();
            for &current in &queue {
                let idx = current as usize;
                for table in &tables {
                    let new_v = table[idx];
                    if !seen.contains(new_v as usize) {
                        seen.insert(new_v as usize);
                        next.push(new_v);
                    }
                }
            }
            std::mem::swap(&mut queue, &mut next);
        }
    }
    minima
}

/// Orbit-min decomposition without the `σ_Q` lookup table — applies each
/// generator via [`mat_apply`] on every BFS step.
///
/// Skips the `2^L`-per-generator build cost of [`aut_orbit_minima_q_table`].
/// Per-step cost is one `popcount`-many `XOR` walk (so cheap for sparse
/// `current`); the open question is whether that flips the verdict against
/// the table path in Rust — in pure Python it did not
/// ([[phase-b-empirical-finding]]).
///
/// Identity generators are filtered at the entry; the inner BFS loop has
/// no identity check.
pub fn aut_orbit_minima_q_witt(
    reps_q: &[BinVec],
    sigma_qs: &[Mat],
    l: u32,
) -> Vec<BinVec> {
    let gens: Vec<&Mat> = sigma_qs.iter().filter(|m| !is_identity_mat(m)).collect();
    if gens.is_empty() {
        let mut out = reps_q.to_vec();
        out.sort_unstable();
        return out;
    }
    let mut reps_sorted = reps_q.to_vec();
    reps_sorted.sort_unstable();
    let universe = 1usize << l;
    let mut seen = FixedBitSet::with_capacity(universe);
    let mut minima: Vec<BinVec> = Vec::new();
    let cap = reps_q.len();
    let mut queue: Vec<BinVec> = Vec::with_capacity(cap);
    let mut next: Vec<BinVec> = Vec::with_capacity(cap);
    for &v in &reps_sorted {
        if seen.contains(v as usize) {
            continue;
        }
        minima.push(v);
        seen.insert(v as usize);
        queue.clear();
        queue.push(v);
        while !queue.is_empty() {
            next.clear();
            for &current in &queue {
                for g in &gens {
                    let new_v = mat_apply(g, current);
                    if !seen.contains(new_v as usize) {
                        seen.insert(new_v as usize);
                        next.push(new_v);
                    }
                }
            }
            std::mem::swap(&mut queue, &mut next);
        }
    }
    minima
}

/// Gray-code walk of `(2^L)` `Q`-coordinates, yielding those whose `F_2^N`
/// lift has weight `≡ 0 (mod 4)`.
///
/// We maintain `(u, lift(u))` incrementally: each step toggles one
/// `Q`-bit `flip` and XORs `v_basis[flip]` into the running lift. The
/// returned vector excludes `u = 0` (the trivial coset).
///
/// Direct port of `doubly_even.enumerate.quotient.singular_reps_Q`.
pub fn singular_reps_q(v_basis: &[BinVec]) -> Vec<BinVec> {
    let l = v_basis.len();
    let mut out: Vec<BinVec> = Vec::new();
    if l == 0 {
        return out;
    }
    debug_assert!(l < 64, "L = {l} exceeds the u64 shift range");
    let size: u64 = 1u64 << l;
    // Heuristic: roughly 1/2 of cosets satisfy wt(lift) ≡ 0 (mod 4) in the
    // typical doubly-even case; reserve that to avoid re-allocation churn.
    out.reserve(size as usize / 2);
    let mut u: BinVec = 0;
    let mut v: BinVec = 0;
    for i in 1..size {
        let flip = i.trailing_zeros() as usize;
        u ^= 1u64 << flip;
        v ^= v_basis[flip];
        if v.count_ones() & 3 == 0 {
            out.push(u);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `L = 0` returns the empty list.
    #[test]
    fn singular_reps_empty_v_basis() {
        let out = singular_reps_q(&[]);
        assert!(out.is_empty());
    }

    /// `L = 1` with `v_basis = (e_0,)`: only u=1 ⇒ lift = e_0 (weight 1, not ≡ 0 mod 4) ⇒ no output.
    #[test]
    fn singular_reps_l1_unit_basis() {
        let v_basis: Vec<BinVec> = vec![0b001];
        let out = singular_reps_q(&v_basis);
        assert!(out.is_empty());
    }

    /// `L = 2` with a `V_basis` whose XOR has weight 4. Lifts:
    /// u=01 → wt=2; u=10 → wt=2; u=11 → wt=4. Only u=3 is singular.
    #[test]
    fn singular_reps_l2_weight4_xor() {
        // v_basis[0] has weight 2 (bits 0,1); v_basis[1] has weight 2 (bits 2,3);
        // XOR has weight 4 (bits 0..3).
        let v_basis: Vec<BinVec> = vec![0b0011, 0b1100];
        let out = singular_reps_q(&v_basis);
        assert_eq!(out, vec![0b11]);
    }

    /// At `L = 4` with the doubly-even `[8,4]` extended Hamming basis lifted
    /// into Q, every codeword has weight 0, 4, or 8 — so every coset is
    /// singular. The count must be `2^L - 1 = 15` (excluding `u = 0`).
    #[test]
    fn singular_reps_extended_hamming_all_singular() {
        // Use the four extended-Hamming basis vectors directly as a V_basis.
        let v_basis: Vec<BinVec> = vec![
            0b1110_0001,
            0b1101_0010,
            0b1011_0100,
            0b0111_1000,
        ];
        let out = singular_reps_q(&v_basis);
        assert_eq!(out.len(), 15);
        // every u in [1, 16) must be present, in Gray-code visit order
        let mut sorted = out.clone();
        sorted.sort();
        let expected: Vec<u64> = (1..16).collect();
        assert_eq!(sorted, expected);
    }

    /// Reference oracle for cross-checks: re-implement the loop body the
    /// same way the Python version does (using `1 << flip` etc.) so we can
    /// verify the trailing-zeros trick is wired right. This is the test
    /// the kernel-vs-Python cross-check in 5(b)-6 piggybacks on.
    #[test]
    fn singular_reps_matches_python_recipe() {
        let v_basis: Vec<BinVec> = vec![0b0011, 0b1100, 0b0101_0101];
        // Reference: brute-force every u from 1 to 2^L - 1.
        let l = v_basis.len();
        let mut expected: Vec<BinVec> = Vec::new();
        for u in 1u64..(1 << l) {
            let mut v: BinVec = 0;
            let mut bits = u;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                v ^= v_basis[i];
                bits &= bits - 1;
            }
            if v.count_ones() & 3 == 0 {
                expected.push(u);
            }
        }
        let mut got = singular_reps_q(&v_basis);
        got.sort();
        expected.sort();
        assert_eq!(got, expected);
    }

    /// `mat_apply` on the identity returns its input.
    #[test]
    fn mat_apply_identity() {
        let m: Mat = vec![1, 2, 4, 8]; // identity over 4 dims
        for v in [0u64, 1, 2, 3, 0b1010, 0b1111] {
            assert_eq!(mat_apply(&m, v), v);
        }
    }

    /// `mat_apply` on a swap-columns matrix swaps the first two bits.
    #[test]
    fn mat_apply_swap() {
        // Column 0 → e_1, column 1 → e_0, columns 2..3 → identity.
        let m: Mat = vec![0b0010, 0b0001, 0b0100, 0b1000];
        assert_eq!(mat_apply(&m, 0b0001), 0b0010);
        assert_eq!(mat_apply(&m, 0b0010), 0b0001);
        assert_eq!(mat_apply(&m, 0b0011), 0b0011);
    }

    /// `sigma_q_table[u]` == manual XOR of σ_Q columns at u's bits, for every u.
    #[test]
    fn sigma_q_table_matches_brute_force() {
        // Take a non-trivial σ_Q of dim L = 4.
        let sigma: Mat = vec![0b0011, 0b0101, 0b1001, 0b1110];
        let table = sigma_q_table(&sigma, 4);
        for u in 0u64..(1 << 4) {
            let want = mat_apply(&sigma, u);
            assert_eq!(table[u as usize], want, "table[{u:04b}] mismatch");
        }
    }

    /// Empty generator set: both BFS variants return reps_q sorted.
    #[test]
    fn orbit_min_no_generators_returns_sorted() {
        let reps: Vec<BinVec> = vec![5, 1, 3, 2];
        let gens: Vec<Mat> = vec![];
        let mins_table = aut_orbit_minima_q_table(&reps, &gens, 3);
        let mins_witt = aut_orbit_minima_q_witt(&reps, &gens, 3);
        assert_eq!(mins_table, vec![1, 2, 3, 5]);
        assert_eq!(mins_witt, vec![1, 2, 3, 5]);
    }

    /// One generator that swaps bit 0 ↔ bit 1: the orbit-min of {01, 10, 11}
    /// is {01, 11}, since 11 is fixed and {01, 10} fuse into one orbit with
    /// min 01.
    #[test]
    fn orbit_min_swap_bit0_bit1() {
        let swap: Mat = vec![0b10, 0b01]; // col0 -> e1, col1 -> e0
        let gens = vec![swap];
        let reps: Vec<BinVec> = vec![1, 2, 3];
        let mins_table = aut_orbit_minima_q_table(&reps, &gens, 2);
        let mins_witt = aut_orbit_minima_q_witt(&reps, &gens, 2);
        assert_eq!(mins_table, vec![1, 3]);
        assert_eq!(mins_witt, vec![1, 3]);
    }

    /// Both BFS variants must agree exact-equal on a non-trivial input.
    /// This is the "table vs witt produce identical output" oracle.
    #[test]
    fn table_and_witt_paths_agree() {
        // L = 5; a permutation matrix (cyclic shift of basis vectors).
        let cyclic: Mat = vec![0b00010, 0b00100, 0b01000, 0b10000, 0b00001];
        // A second generator: swap bits 0 ↔ 1.
        let swap: Mat = vec![0b00010, 0b00001, 0b00100, 0b01000, 0b10000];
        let gens = vec![cyclic, swap];
        // Use all of [1, 2^5) as reps.
        let reps: Vec<BinVec> = (1u64..(1 << 5)).collect();
        let mut mins_table = aut_orbit_minima_q_table(&reps, &gens, 5);
        let mut mins_witt = aut_orbit_minima_q_witt(&reps, &gens, 5);
        mins_table.sort_unstable();
        mins_witt.sort_unstable();
        assert_eq!(mins_table, mins_witt);
        // The full S_5 action on basis vectors makes weight the only invariant,
        // so the orbit-min set is one rep per weight in [1, 5]: weights 1..5.
        // Sorted, those are {0b00001, 0b00011, 0b00111, 0b01111, 0b11111}.
        assert_eq!(
            mins_table,
            vec![0b00001, 0b00011, 0b00111, 0b01111, 0b11111]
        );
    }

    /// Identity generators are silently filtered by the witt path.
    #[test]
    fn witt_filters_identity_generators() {
        let identity: Mat = vec![1, 2, 4, 8];
        let reps: Vec<BinVec> = vec![1, 2, 3];
        let mins = aut_orbit_minima_q_witt(&reps, &[identity], 4);
        // With only-identity gens the witt path should behave like
        // empty-gen-set: return sorted reps.
        assert_eq!(mins, vec![1, 2, 3]);
    }
}
