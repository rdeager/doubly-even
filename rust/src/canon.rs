//! Native bipartite-graph canonicaliser via `nauty-Traces-sys`.
//!
//! Replaces the Python `canon.bipartite.bipartite_graph` + `pynauty.autgrp` +
//! `pynauty.canon_label` chain. The hot loop builds a fresh bipartite graph
//! per parent in the canonical-augmentation recursion; doing the whole thing
//! in Rust eliminates both the Python adjacency-dict construction and
//! pynauty's internal dict→sparsegraph conversion (measured: ~43% of the
//! per-call cost is Python↔C marshalling).
//!
//! API mirrors `canon.nauty.canon_info`: takes `(n, rref, pivots)` →
//! `(canonical_column_order, aut_generators, grpsize1, grpsize2,
//! column_orbits)`. The float pair `(grpsize1, grpsize2)` is nauty's native
//! group-order representation; Python applies the same trust-the-float-when-
//! small / Schreier-Sims-fallback logic as before, so the API contract with
//! `CanonInfo` is unchanged.

use std::cell::RefCell;
use std::ffi::c_int;
use std::ptr;

use nauty_Traces_sys::{
    optionblk, sparsegraph, sparsenauty, statsblk, FALSE, TRUE,
};

use crate::types::BinVec;

/// Output of [`canon_info_native`], mirroring `canon.nauty.CanonInfo` minus
/// the exact `aut_order` (Python computes that from `grpsize1`/`grpsize2`
/// or via Schreier-Sims on the returned generators).
pub struct NativeCanonInfo {
    /// `canonical_column_order[old_col] = new_col` — the column permutation
    /// that puts `C` in canonical form. Length `n`.
    pub canonical_column_order: Vec<u32>,
    /// Column-restricted generators of `Aut(C)`, each a permutation of
    /// `range(n)` in `apply_permutation` convention (`g[i]=j` means old
    /// column `i` becomes new column `j`).
    pub aut_generators: Vec<Vec<u32>>,
    /// `column_orbits[col]` is the orbit identifier of column `col` under
    /// `Aut(C)`. Two columns share an identifier iff they are in the same
    /// orbit.
    pub column_orbits: Vec<u32>,
    /// `|Aut(G(C))| = grpsize1 * 10^grpsize2`. The full bipartite-graph
    /// automorphism group order; equals `|Aut(C)|` (the column-side
    /// stabiliser is the whole group, since the left action is determined
    /// by the right action under our two-block colouring).
    pub grpsize1: f64,
    pub grpsize2: i32,
}

// nauty's callback API hands generators to a function pointer one at a time.
// We collect them into a thread-local Vec; the callback runs synchronously
// inside `sparsenauty`, so the lifetime is bounded by the call site.
thread_local! {
    static AUT_BUFFER: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
    /// Number of left-side (codeword) vertices; the callback needs to know
    /// where to slice the column part out of the full permutation.
    static LEFT_VERTEX_COUNT: RefCell<u32> = const { RefCell::new(0) };
}

extern "C" fn auto_callback(
    _count: c_int,
    perm: *mut c_int,
    _orbits: *mut c_int,
    _numorbits: c_int,
    _stabvertex: c_int,
    n: c_int,
) {
    let l = LEFT_VERTEX_COUNT.with(|cell| *cell.borrow());
    let n = n as usize;
    let mut col_perm = vec![0u32; n - l as usize];
    // `perm[i] = j` means "old vertex i → new vertex j" in nauty's convention.
    // Columns live in `[l, n)`; restrict the action and re-zero the index.
    for old in (l as usize)..n {
        let new_v = unsafe { *perm.add(old) } as u32;
        col_perm[old - l as usize] = new_v - l;
    }
    AUT_BUFFER.with(|cell| cell.borrow_mut().push(col_perm));
}

/// Build the bipartite codeword × column sparsegraph adjacency arrays.
///
/// Left vertices `[0, L)` are codewords (`L = 2^rank`); right vertices
/// `[L, L+R)` are columns (`R = n`). Edge `(codeword i, column j)` iff bit
/// `j` of codeword `i` is set. The graph is stored undirected: each edge
/// contributes to both endpoints' neighbour lists.
fn build_sparsegraph(rref: &[BinVec], n: u32) -> (Vec<usize>, Vec<i32>, Vec<i32>) {
    let k = rref.len();
    let l: usize = 1usize << k;
    let r: usize = n as usize;
    let total = l + r;

    // Enumerate codewords once via a Gray-code walk so XOR is constant-time
    // per word; `cw[mask]` is the codeword whose linear combination uses the
    // bits of `mask`. (Same layout as `Code.codewords()`.)
    let mut cw = vec![0u64; l];
    for mask in 1..l {
        let lo_bit = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        cw[mask] = cw[mask ^ (1 << lo_bit)] ^ rref[lo_bit];
    }

    // Degrees on both sides.
    let mut d = vec![0i32; total];
    for (i, &w) in cw.iter().enumerate().take(l) {
        d[i] = w.count_ones() as i32;
    }
    for j in 0..r {
        let bit = 1u64 << j;
        let mut deg = 0i32;
        for &w in cw.iter().take(l) {
            if w & bit != 0 {
                deg += 1;
            }
        }
        d[l + j] = deg;
    }

    // Offsets via prefix sum.
    let nde: usize = d.iter().map(|&x| x as usize).sum();
    let mut v = vec![0usize; total];
    let mut acc = 0usize;
    for i in 0..total {
        v[i] = acc;
        acc += d[i] as usize;
    }

    // Edge lists. Walk each codeword's set bits once; emit (left → right)
    // immediately, push (right → left) into per-column buffers we splice in
    // afterwards (cheaper than seeking into `e` from both directions).
    let mut e = vec![0i32; nde];
    let mut left_write = v.clone();
    let mut right_lists: Vec<Vec<i32>> = (0..r).map(|j| Vec::with_capacity(d[l + j] as usize)).collect();
    for (i, &w) in cw.iter().enumerate().take(l) {
        let mut bits = w;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            e[left_write[i]] = (l + j) as i32;
            left_write[i] += 1;
            right_lists[j].push(i as i32);
            bits &= bits - 1;
        }
    }
    for j in 0..r {
        let base = v[l + j];
        for (offset, &neighbour) in right_lists[j].iter().enumerate() {
            e[base + offset] = neighbour;
        }
    }

    (v, d, e)
}

/// Run nauty on the bipartite encoding of `(rref, n)` and return the
/// canonical column order, automorphism generators (column-restricted),
/// orbit assignment, and group-order float pair.
///
/// Single FFI call per parent (vs pynauty's two — one for `autgrp`, one for
/// `canon_label` — each of which also rebuilds the adjacency from a Python
/// dict). The bipartite-graph automorphism group on the right side is
/// `Aut(C)` exactly; see `canon/bipartite.py` module docstring for the
/// math.
pub fn canon_info_native(rref: &[BinVec], n: u32) -> NativeCanonInfo {
    let l: usize = 1usize << rref.len();
    let r: usize = n as usize;
    let total = l + r;

    let (mut v, mut d, mut e) = build_sparsegraph(rref, n);

    let mut sg = sparsegraph {
        nde: e.len(),
        v: v.as_mut_ptr(),
        nv: total as c_int,
        d: d.as_mut_ptr(),
        e: e.as_mut_ptr(),
        w: ptr::null_mut(),
        vlen: v.len(),
        dlen: d.len(),
        elen: e.len(),
        wlen: 0,
    };

    // Two-block colouring: codewords (left) form colour 0, columns (right)
    // form colour 1. `lab` is the identity ordering; `ptn[i] = 0` marks the
    // last vertex of its colour cell.
    let mut lab: Vec<c_int> = (0..total as c_int).collect();
    let mut ptn = vec![1i32; total];
    if l > 0 {
        ptn[l - 1] = 0;
    }
    if r > 0 {
        ptn[total - 1] = 0;
    }
    let mut orbits = vec![0i32; total];

    let mut options = optionblk::default_sparse();
    options.getcanon = TRUE;
    options.defaultptn = FALSE;
    options.userautomproc = Some(auto_callback);
    let mut stats = statsblk::default();

    // Canonical-graph output: nauty needs an allocated sparsegraph to write
    // into. We discard the contents — only the canonical labelling matters.
    let mut cg_v = vec![0usize; total];
    let mut cg_d = vec![0i32; total];
    let mut cg_e = vec![0i32; e.len()];
    let mut canon_sg = sparsegraph {
        nde: 0,
        v: cg_v.as_mut_ptr(),
        nv: total as c_int,
        d: cg_d.as_mut_ptr(),
        e: cg_e.as_mut_ptr(),
        w: ptr::null_mut(),
        vlen: cg_v.len(),
        dlen: cg_d.len(),
        elen: cg_e.len(),
        wlen: 0,
    };

    AUT_BUFFER.with(|cell| cell.borrow_mut().clear());
    LEFT_VERTEX_COUNT.with(|cell| *cell.borrow_mut() = l as u32);

    unsafe {
        sparsenauty(
            &mut sg,
            lab.as_mut_ptr(),
            ptn.as_mut_ptr(),
            orbits.as_mut_ptr(),
            &mut options,
            &mut stats,
            &mut canon_sg,
        );
    }

    // `lab` now holds the canonical-form vertex order: `lab[new_index] =
    // old_vertex`. Restrict to the right side (columns) and invert into
    // "old column → new column" so it matches `apply_permutation`'s
    // convention (and pynauty's `canonical_column_order` field).
    let mut new_to_old_right: Vec<u32> = Vec::with_capacity(r);
    for new_index in 0..total {
        let old_vertex = lab[new_index] as usize;
        if old_vertex >= l {
            new_to_old_right.push((old_vertex - l) as u32);
        }
    }
    let mut canonical_column_order = vec![0u32; r];
    for (new_col, &old_col) in new_to_old_right.iter().enumerate() {
        canonical_column_order[old_col as usize] = new_col as u32;
    }

    let column_orbits: Vec<u32> = orbits[l..].iter().map(|&x| x as u32).collect();
    let aut_generators = AUT_BUFFER.with(|cell| std::mem::take(&mut *cell.borrow_mut()));

    NativeCanonInfo {
        canonical_column_order,
        aut_generators,
        column_orbits,
        grpsize1: stats.grpsize1,
        grpsize2: stats.grpsize2 as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_code_has_full_symmetric_group() {
        // Rank-0 code: bipartite graph has L=1 (only the zero codeword),
        // R=n, no edges. Aut is the full S_n on the right side.
        let info = canon_info_native(&[], 4);
        assert_eq!(info.canonical_column_order.len(), 4);
        // 4! = 24
        assert_eq!(info.grpsize1.round() as u32, 24);
        assert_eq!(info.grpsize2, 0);
        // All columns in one orbit.
        let first = info.column_orbits[0];
        assert!(info.column_orbits.iter().all(|&x| x == first));
    }

    #[test]
    fn repetition_code_n4_k1() {
        // Code spanned by 0b1111: one nonzero codeword of weight 4.
        // Aut is S_4 (any permutation preserves the all-ones codeword).
        let info = canon_info_native(&[0b1111u64], 4);
        assert_eq!(info.grpsize1.round() as u32, 24);
        assert_eq!(info.grpsize2, 0);
    }

    #[test]
    fn extended_hamming_8_4_has_known_aut_order() {
        // [8, 4, 4] extended Hamming code in standard form G = [I_4 | P]
        // with P the unique 4x4 weight-3 parity matrix.
        //   row i = bit i (the identity part) ∪ bits {4..7} of P[i].
        //   P = ((0,1,1,1), (1,0,1,1), (1,1,0,1), (1,1,1,0)).
        // Each row has weight 4 (doubly even). |Aut| = 1344 = |AGL(3, 2)|,
        // the affine extension of GL(3, 2) ≅ PSL(2, 7); equivalently
        // 2^3 ⋊ PSL(2, 7) (order 8 × 168).
        let basis = vec![
            0xE1u64, // bits {0, 5, 6, 7}
            0xD2u64, // bits {1, 4, 6, 7}
            0xB4u64, // bits {2, 4, 5, 7}
            0x78u64, // bits {3, 4, 5, 6}
        ];
        let info = canon_info_native(&basis, 8);
        let raw = info.grpsize1 * 10f64.powi(info.grpsize2);
        assert_eq!(raw.round() as u32, 1344);
    }
}
