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

/// Codeword-side vertex count (2^k) above which the dispatch in
/// `enumerate::State::canon_info` prefers `canon_info_qd_native` over
/// `canon_info_native`. Below this, the full bipartite graph is small
/// enough that the encoding savings don't pay for the span check.
pub const QD_GRAPH_THRESHOLD: u32 = 0;

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
    /// Tree-shape counters from nauty's `statsblk`. Recorded per call so
    /// the aggregate decomposes the 78 µs/call cost: backtrack size
    /// (`numnodes`), total target-cell work (`tctotal`), deepest level
    /// reached (`maxlevel`), automorphism generators discovered
    /// (`numgenerators`). See `expert-review/05-nauty-traces-audit.md` Q6.
    pub numnodes: u64,
    pub tctotal: u64,
    pub maxlevel: i32,
    pub numgenerators: i32,
}

// nauty's callback API hands generators to a function pointer one at a time.
// We collect them into a thread-local Vec; the callback runs synchronously
// inside `sparsenauty`, so the lifetime is bounded by the call site.
thread_local! {
    pub(crate) static AUT_BUFFER: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
    /// Number of left-side (codeword) vertices; the callback needs to know
    /// where to slice the column part out of the full permutation.
    pub(crate) static LEFT_VERTEX_COUNT: RefCell<u32> = const { RefCell::new(0) };
    /// D13-V4 cut 2: per-call scratch buffers for `canon_info_native` +
    /// `canon_info_qd_native`. Holds ~210 KB of working storage (cw, d, v, e,
    /// lab, ptn, orbits, cg_v, cg_d, cg_e, right_lists, by_cell, …).
    /// Reused across calls via `.clear() + .resize()` so the Vec capacities
    /// settle at the high-water mark after warmup and subsequent calls do
    /// zero heap allocations. Keeps the working set hot in L2 across
    /// successive canon calls on the same worker.
    pub(crate) static SCRATCH: RefCell<CanonScratch> = RefCell::new(CanonScratch::default());
}

/// Shared scratch buffers; see `SCRATCH` thread-local. All fields are
/// grow-only across the worker's lifetime — `.clear()` resets length to 0
/// without releasing capacity, `.resize(n, 0)` re-zeros the prefix.
#[derive(Default)]
pub(crate) struct CanonScratch {
    // Bipartite-graph adjacency (input to sparsenauty).
    pub(crate) v: Vec<usize>,
    pub(crate) d: Vec<i32>,
    pub(crate) e: Vec<i32>,
    // Initial colouring (input to sparsenauty).
    pub(crate) lab: Vec<c_int>,
    pub(crate) ptn: Vec<c_int>,
    pub(crate) orbits: Vec<i32>,
    // Canonical-graph storage (nauty writes here, we discard).
    pub(crate) cg_v: Vec<usize>,
    pub(crate) cg_d: Vec<i32>,
    pub(crate) cg_e: Vec<i32>,
    // `canon_info_native` helpers.
    pub(crate) cw: Vec<u64>,
    pub(crate) right_lists: Vec<Vec<i32>>,
    pub(crate) by_cell: Vec<(u8, i32, c_int)>,
    // `canon_info_qd_native` helpers.
    pub(crate) by_weight: Vec<Vec<BinVec>>,
    pub(crate) accum: Vec<BinVec>,
    pub(crate) stratum_sizes: Vec<usize>,
    pub(crate) col_by_deg: Vec<(i32, c_int)>,
}

// E1/E2 sparsenauty histogram (Cargo feature `nauty_hist`, default OFF)
// lives in `crate::experimental::canon_hist`. The two call sites below
// push records via `canon_hist::push(...)`.
#[cfg(feature = "nauty_hist")]
use crate::experimental::canon_hist::{push as nauty_hist_push, NautyCallRecord};

pub(crate) extern "C" fn auto_callback(
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

/// Build the bipartite codeword × column sparsegraph adjacency arrays into
/// `scratch`. Writes `scratch.cw`, `.d`, `.v`, `.e`; uses `.right_lists` as
/// internal scratch. After return: `scratch.v.as_mut_ptr()`, `.d.as_mut_ptr()`,
/// `.e.as_mut_ptr()` are ready to hand to `sparsenauty`. Returns `nde`
/// (= `e.len()`) for convenience.
///
/// Left vertices `[0, L)` are codewords (`L = 2^rank`); right vertices
/// `[L, L+R)` are columns (`R = n`). Edge `(codeword i, column j)` iff bit
/// `j` of codeword `i` is set. The graph is stored undirected: each edge
/// contributes to both endpoints' neighbour lists.
fn build_sparsegraph(rref: &[BinVec], n: u32, scratch: &mut CanonScratch) -> usize {
    let k = rref.len();
    let l: usize = 1usize << k;
    let r: usize = n as usize;
    let total = l + r;

    // Enumerate codewords once via a Gray-code walk so XOR is constant-time
    // per word; `cw[mask]` is the codeword whose linear combination uses the
    // bits of `mask`. (Same layout as `Code.codewords()`.)
    scratch.cw.clear();
    scratch.cw.resize(l, 0);
    for mask in 1..l {
        let lo_bit = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        scratch.cw[mask] = scratch.cw[mask ^ (1 << lo_bit)] ^ rref[lo_bit];
    }

    // Degrees on both sides.
    scratch.d.clear();
    scratch.d.resize(total, 0);
    for (i, &w) in scratch.cw.iter().enumerate().take(l) {
        scratch.d[i] = w.count_ones() as i32;
    }
    for j in 0..r {
        let bit = 1u64 << j;
        let mut deg = 0i32;
        for &w in scratch.cw.iter().take(l) {
            if w & bit != 0 {
                deg += 1;
            }
        }
        scratch.d[l + j] = deg;
    }

    // Offsets via prefix sum.
    let nde: usize = scratch.d.iter().map(|&x| x as usize).sum();
    scratch.v.clear();
    scratch.v.resize(total, 0);
    let mut acc = 0usize;
    for i in 0..total {
        scratch.v[i] = acc;
        acc += scratch.d[i] as usize;
    }

    // Edge lists. Walk each codeword's set bits once; emit (left → right)
    // immediately, push (right → left) into per-column buffers we splice in
    // afterwards (cheaper than seeking into `e` from both directions).
    //
    // Cut 3: write into the left half of `e` directly using a stack-local
    // cursor (initialised from `v[0..l]`), so we no longer clone the whole
    // `v` vector on every call.
    scratch.e.clear();
    scratch.e.resize(nde, 0);
    // Grow right_lists outer Vec if needed; reuse inner Vecs via .clear().
    if scratch.right_lists.len() < r {
        scratch.right_lists.resize_with(r, Vec::new);
    }
    for j in 0..r {
        scratch.right_lists[j].clear();
        scratch.right_lists[j].reserve(scratch.d[l + j] as usize);
    }
    // Disjoint borrows: cw, d, v, e, right_lists are separate fields.
    let cw = &scratch.cw;
    let d = &scratch.d;
    let v = &scratch.v;
    let e = &mut scratch.e;
    let right_lists = &mut scratch.right_lists;
    for (i, &w) in cw.iter().enumerate().take(l) {
        let mut cursor = v[i];
        let mut bits = w;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            e[cursor] = (l + j) as i32;
            cursor += 1;
            right_lists[j].push(i as i32);
            bits &= bits - 1;
        }
        debug_assert_eq!(cursor, v[i] + d[i] as usize);
    }
    for j in 0..r {
        let base = v[l + j];
        for (offset, &neighbour) in right_lists[j].iter().enumerate() {
            e[base + offset] = neighbour;
        }
    }

    nde
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
    SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        canon_info_native_impl(rref, n, &mut scratch)
    })
}

fn canon_info_native_impl(rref: &[BinVec], n: u32, scratch: &mut CanonScratch) -> NativeCanonInfo {
    let l: usize = 1usize << rref.len();
    let r: usize = n as usize;
    let total = l + r;

    let nde = build_sparsegraph(rref, n, scratch);

    let mut sg = sparsegraph {
        nde,
        v: scratch.v.as_mut_ptr(),
        nv: total as c_int,
        d: scratch.d.as_mut_ptr(),
        e: scratch.e.as_mut_ptr(),
        w: ptr::null_mut(),
        vlen: scratch.v.len(),
        dlen: scratch.d.len(),
        elen: scratch.e.len(),
        wlen: 0,
    };

    // Initial partition by (side, degree). Side first keeps codewords (left)
    // and columns (right) in separate cells — automorphisms must respect the
    // bipartition. Within each side, sub-cells by degree: codeword weight
    // on the left, column-incidence count on the right. Degree is Aut-
    // invariant so distinct degrees imply distinct orbits, letting nauty
    // skip its own initial degree-refinement pass (Bouyukliev §3.3,
    // matches Sage's default behaviour).
    scratch.by_cell.clear();
    scratch.by_cell.extend((0..total as c_int).map(|vid| {
        let side: u8 = if (vid as usize) < l { 0 } else { 1 };
        (side, scratch.d[vid as usize], vid)
    }));
    scratch.by_cell.sort_unstable_by_key(|&(side, deg, _)| (side, deg));

    scratch.lab.clear();
    scratch.lab.extend(scratch.by_cell.iter().map(|&(_, _, v)| v));
    scratch.ptn.clear();
    scratch.ptn.resize(total, 1);
    // ptn[i] = 0 marks the last vertex of a cell. Set 0 wherever the
    // (side, degree) key changes at position i+1; always set 0 at the end.
    for i in 0..total.saturating_sub(1) {
        let (s1, d1, _) = scratch.by_cell[i];
        let (s2, d2, _) = scratch.by_cell[i + 1];
        if s1 != s2 || d1 != d2 {
            scratch.ptn[i] = 0;
        }
    }
    if total > 0 {
        scratch.ptn[total - 1] = 0;
    }
    scratch.orbits.clear();
    scratch.orbits.resize(total, 0);

    let mut options = optionblk::default_sparse();
    options.getcanon = TRUE;
    options.defaultptn = FALSE;
    options.userautomproc = Some(auto_callback);
    // Q6 audit (`expert-review/05-nauty-traces-audit.md` Phase 1):
    // `options.schreier = TRUE` measured +3–6 % wall regression at
    // N = 20, 22 with zero change to `numnodes` — sparsenauty's tree
    // is already small enough (≈ 67 nodes/call at N = 22) that the
    // Schreier bookkeeping outweighs the pruning. Knob closed.
    let mut stats = statsblk::default();

    // Canonical-graph output: nauty needs an allocated sparsegraph to write
    // into. We discard the contents — only the canonical labelling matters.
    scratch.cg_v.clear();
    scratch.cg_v.resize(total, 0);
    scratch.cg_d.clear();
    scratch.cg_d.resize(total, 0);
    scratch.cg_e.clear();
    scratch.cg_e.resize(nde, 0);
    let mut canon_sg = sparsegraph {
        nde: 0,
        v: scratch.cg_v.as_mut_ptr(),
        nv: total as c_int,
        d: scratch.cg_d.as_mut_ptr(),
        e: scratch.cg_e.as_mut_ptr(),
        w: ptr::null_mut(),
        vlen: scratch.cg_v.len(),
        dlen: scratch.cg_d.len(),
        elen: scratch.cg_e.len(),
        wlen: 0,
    };

    AUT_BUFFER.with(|cell| cell.borrow_mut().clear());
    LEFT_VERTEX_COUNT.with(|cell| *cell.borrow_mut() = l as u32);

    #[cfg(feature = "nauty_hist")]
    let nauty_t0 = std::time::Instant::now();

    unsafe {
        sparsenauty(
            &mut sg,
            scratch.lab.as_mut_ptr(),
            scratch.ptn.as_mut_ptr(),
            scratch.orbits.as_mut_ptr(),
            &mut options,
            &mut stats,
            &mut canon_sg,
        );
    }

    #[cfg(feature = "nauty_hist")]
    {
        let elapsed_ns = nauty_t0.elapsed().as_nanos() as u64;
        nauty_hist_push(NautyCallRecord {
            elapsed_ns,
            numnodes: stats.numnodes as u64,
            tctotal: stats.tctotal as u64,
            maxlevel: stats.maxlevel as i32,
            numgenerators: stats.numgenerators as i32,
            left_vertices: l as u32,
            right_vertices: r as u32,
            rank: rref.len() as u32,
            qd_path: false,
        });
    }

    // `lab` now holds the canonical-form vertex order: `lab[new_index] =
    // old_vertex`. Restrict to the right side (columns) and invert into
    // "old column → new column" so it matches `apply_permutation`'s
    // convention (and pynauty's `canonical_column_order` field).
    // Result Vec sizes are O(n) (~22 entries at N=22); keep these fresh-
    // allocated rather than scratch-reused — they're handed to Python and
    // outlive the scratch borrow anyway.
    let mut canonical_column_order = vec![0u32; r];
    let mut new_col_counter = 0u32;
    for new_index in 0..total {
        let old_vertex = scratch.lab[new_index] as usize;
        if old_vertex >= l {
            canonical_column_order[old_vertex - l] = new_col_counter;
            new_col_counter += 1;
        }
    }

    let column_orbits: Vec<u32> = scratch.orbits[l..].iter().map(|&x| x as u32).collect();
    let aut_generators = AUT_BUFFER.with(|cell| std::mem::take(&mut *cell.borrow_mut()));

    NativeCanonInfo {
        canonical_column_order,
        aut_generators,
        column_orbits,
        grpsize1: stats.grpsize1,
        grpsize2: stats.grpsize2 as i32,
        numnodes: stats.numnodes as u64,
        tctotal: stats.tctotal as u64,
        maxlevel: stats.maxlevel as i32,
        numgenerators: stats.numgenerators as i32,
    }
}

// `collect_low_weight_codewords`, `build_low_weight_sparsegraph`, and
// `canon_info_qd_native` live in `crate::qd_graph` since Phase 2 Cut 6.
// The dormant audit substrate (canon_dense_qd, canon_traces_qd) imports
// `build_low_weight_sparsegraph` from there as well.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qd_graph::canon_info_qd_native;

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

    #[test]
    fn qd_native_extended_hamming_8_4_bails() {
        // The [8, 4, 4] extended Hamming code has 14 weight-4 codewords out
        // of 15 nonzero — almost no vertex savings vs the full bipartite.
        // The builder's bail threshold (|C_low| ≥ 2^(k-1)) correctly fires
        // here; the dispatch in `enumerate.rs` falls back to
        // `canon_info_native`. Pinning this behaviour so a future tweak to
        // the bail threshold doesn't silently change it.
        let basis = vec![0xE1u64, 0xD2u64, 0xB4u64, 0x78u64];
        assert!(canon_info_qd_native(&basis, 8).is_none());
    }

    #[test]
    fn qd_native_sparse_doubly_even_n8_k2() {
        // Two disjoint-support all-ones blocks: {0b0000_1111, 0b1111_0000}.
        // 4 codewords: 0, two of weight 4, one of weight 8. The weight-4
        // stratum has 2 codewords and spans the code; bail threshold is 2
        // (no bail at |accum| == 2). Aut(C) is S_4 × S_4 ⋊ Z_2, order
        // (4!)^2 * 2 = 1152.
        let basis = vec![0x0Fu64, 0xF0u64];
        let info = canon_info_qd_native(&basis, 8).expect("qd builder should succeed");
        let raw = info.grpsize1 * 10f64.powi(info.grpsize2);
        // Cross-check against canon_info_native on the same code.
        let native = canon_info_native(&basis, 8);
        let native_raw = native.grpsize1 * 10f64.powi(native.grpsize2);
        assert_eq!(raw.round() as u64, native_raw.round() as u64);
        // Orbit partition (as a set partition) must agree.
        assert_eq!(
            orbit_partition(&info.column_orbits),
            orbit_partition(&native.column_orbits),
        );
    }

    /// Canonicalise an orbit-id vector into the set partition it represents,
    /// so two orbit vectors with different label conventions still compare
    /// equal when they describe the same partition.
    fn orbit_partition(orbits: &[u32]) -> Vec<Vec<usize>> {
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, &o) in orbits.iter().enumerate() {
            groups.entry(o).or_default().push(i);
        }
        let mut parts: Vec<Vec<usize>> = groups.into_values().collect();
        parts.sort();
        parts
    }

    #[test]
    fn qd_native_repetition_n4_k1() {
        // Code spanned by 0b1111: span of weight-4 stratum is the whole
        // 1-dim code. Aut should be S_4 (24).
        let info = canon_info_qd_native(&[0b1111u64], 4).expect("qd builder should succeed");
        assert_eq!(info.grpsize1.round() as u32, 24);
        assert_eq!(info.grpsize2, 0);
    }

    #[test]
    fn qd_native_zero_code_falls_back() {
        // Rank-0 code: no nonzero codewords. Builder returns None so the
        // caller falls back to canon_info_native.
        assert!(canon_info_qd_native(&[], 5).is_none());
    }
}
