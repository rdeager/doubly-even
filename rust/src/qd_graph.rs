//! Q_D-graph canonicaliser: nauty over the low-weight-codeword × column
//! bipartite encoding (D10, 2026-04-23 ship; the active default for the
//! production hot path).
//!
//! Math invariant: the column-side stabiliser of the low-weight-incidence
//! graph equals Aut(C) iff the included codewords span C; the builder
//! (`collect_low_weight_codewords`) walks weight strata in ascending order
//! and stops once their span equals C. If the accumulated set grows past
//! `2^k / 2` before spanning, we bail and the caller falls back to the
//! full bipartite encoding (`canon::canon_info_native`).
//!
//! Sized accumulator → sparsegraph build → sparsenauty call → restrict
//! canonical labelling to the column block. ~1.91× faster than full G(C)
//! at N=22 by virtue of the smaller left-side vertex count.

use std::ffi::c_int;
use std::ptr;

use nauty_Traces_sys::{optionblk, sparsegraph, sparsenauty, statsblk, FALSE, TRUE};

use crate::canon::{
    auto_callback, CanonScratch, NativeCanonInfo, AUT_BUFFER, LEFT_VERTEX_COUNT, SCRATCH,
};
use crate::linalg::row_reduce;
use crate::types::BinVec;

/// Walks codewords by ascending weight (Gray code) and accumulates strata
/// until their span equals C. Stops early and returns `None` if the
/// accumulated set reaches the "no-win" size of `(2^k) / 2` before
/// spanning — at that point the full bipartite graph is cheaper than the
/// low-weight one plus its overhead, so the caller falls back.
///
/// Returns `(low_weight_codewords, stratum_sizes)` where `stratum_sizes[i]`
/// is the count of codewords belonging to the `i`-th included weight
/// stratum, in ascending weight order. The flat list is the strata
/// concatenated.
/// Returns `Some(L)` (the codeword-side vertex count) on success; the
/// caller reads the populated codeword set from `scratch.accum` and the
/// stratum sizes from `scratch.stratum_sizes`. Reuses
/// `scratch.by_weight` / `.accum` / `.stratum_sizes` across calls.
fn collect_low_weight_codewords(rref: &[BinVec], n: u32, scratch: &mut CanonScratch) -> Option<usize> {
    let k = rref.len();
    if k == 0 {
        // Rank-0 code: only the zero codeword. No useful low-weight set;
        // fall back to the native encoding (which handles this case fine).
        return None;
    }
    let total_codewords: usize = 1usize << k;
    let bail_threshold = total_codewords / 2;

    // Gray-code walk: enumerate all nonzero codewords with one XOR per step.
    // Group by Hamming weight into `by_weight[w] = Vec<codeword>`. We use a
    // BTreeMap-like structure but cap at n+1 distinct possible weights.
    if scratch.by_weight.len() < (n as usize) + 1 {
        scratch.by_weight.resize((n as usize) + 1, Vec::new());
    }
    for bucket in scratch.by_weight.iter_mut().take((n as usize) + 1) {
        bucket.clear();
    }
    let mut w: BinVec = 0;
    for mask in 1..total_codewords {
        let lo_bit = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        w ^= rref[lo_bit];
        scratch.by_weight[w.count_ones() as usize].push(w);
    }

    // Walk strata in ascending weight, growing the accumulated low-weight
    // set until its span equals C. We rebuild RREF each round on the
    // accumulated set; cheap relative to the nauty call we're about to skip.
    scratch.accum.clear();
    scratch.stratum_sizes.clear();
    for weight in 1..=(n as usize) {
        if scratch.by_weight[weight].is_empty() {
            continue;
        }
        let stratum_len = scratch.by_weight[weight].len();
        // Drain the stratum into accum without re-allocating the bucket.
        let drain = scratch.by_weight[weight].drain(..);
        scratch.accum.extend(drain);
        scratch.stratum_sizes.push(stratum_len);
        if scratch.accum.len() > bail_threshold {
            return None;
        }
        let (rr, _) = row_reduce(&scratch.accum, n);
        if rr.len() == k {
            return Some(scratch.accum.len());
        }
    }
    // If we exhaust strata without spanning, something is wrong with the
    // RREF (it should span all of C); be safe and fall back.
    None
}

/// Build the bipartite (low-weight codeword) × column sparsegraph plus the
/// initial partition that keeps each weight stratum as its own colour
/// class and segregates the column block by its incidence-degree.
///
/// Returns `None` if the span-aware low-weight set isn't usefully smaller
/// than 2^k (see [`collect_low_weight_codewords`]).
///
/// Layout:
/// - Vertices `[0, L)` are codewords, grouped by ascending weight stratum.
/// - Vertices `[L, L + R)` are columns.
/// - `lab` lists vertices in the order each cell is laid out; `ptn[i] = 0`
///   marks cell boundaries (nauty convention).
#[allow(clippy::type_complexity)]
/// Returns `Some((nde, l))` on success and populates
/// `scratch.{v,d,e,lab,ptn}`. Returns `None` (and leaves scratch in a
/// best-effort partial state) when the span-aware builder bails. Reuses
/// `scratch.{by_weight,accum,stratum_sizes,col_by_deg,right_lists}` as
/// internal storage.
pub(crate) fn build_low_weight_sparsegraph(
    rref: &[BinVec],
    n: u32,
    scratch: &mut CanonScratch,
) -> Option<(usize, usize)> {
    let l: usize = collect_low_weight_codewords(rref, n, scratch)?;
    let r: usize = n as usize;
    let total = l + r;

    // Per-vertex degrees.
    scratch.d.clear();
    scratch.d.resize(total, 0);
    for (i, &cw) in scratch.accum.iter().enumerate() {
        scratch.d[i] = cw.count_ones() as i32;
    }
    for j in 0..r {
        let bit = 1u64 << j;
        let mut deg = 0i32;
        for &cw in &scratch.accum {
            if cw & bit != 0 {
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

    // Edge lists: walk each codeword's set bits once.
    //
    // Cut 3: avoid cloning `v` for the left-side write cursor — use a
    // stack-local that's initialised from `v[i]` per outer iteration.
    scratch.e.clear();
    scratch.e.resize(nde, 0);
    if scratch.right_lists.len() < r {
        scratch.right_lists.resize_with(r, Vec::new);
    }
    for j in 0..r {
        scratch.right_lists[j].clear();
        scratch.right_lists[j].reserve(scratch.d[l + j] as usize);
    }
    let codewords = &scratch.accum;
    let d_ref = &scratch.d;
    let v_ref = &scratch.v;
    let e_ref = &mut scratch.e;
    let right_lists = &mut scratch.right_lists;
    for (i, &cw) in codewords.iter().enumerate() {
        let mut cursor = v_ref[i];
        let mut bits = cw;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            e_ref[cursor] = (l + j) as i32;
            cursor += 1;
            right_lists[j].push(i as i32);
            bits &= bits - 1;
        }
        debug_assert_eq!(cursor, v_ref[i] + d_ref[i] as usize);
    }
    for j in 0..r {
        let base = scratch.v[l + j];
        for (offset, &neighbour) in scratch.right_lists[j].iter().enumerate() {
            scratch.e[base + offset] = neighbour;
        }
    }

    // Initial partition. Codeword side: cells in stratum order (each weight
    // stratum is one cell). Column side: subdivide by degree (the existing
    // bipartite encoding does the same — see `canon_info_native` in canon.rs).
    // `ptn[i] = 0` marks the end of a cell.
    scratch.lab.clear();
    scratch.lab.reserve(total);
    scratch.ptn.clear();
    scratch.ptn.reserve(total);
    let mut cursor = 0usize;
    for &size in &scratch.stratum_sizes {
        for idx in cursor..(cursor + size) {
            scratch.lab.push(idx as c_int);
            scratch.ptn.push(1);
        }
        // mark end of stratum
        if !scratch.ptn.is_empty() {
            let last = scratch.ptn.len() - 1;
            scratch.ptn[last] = 0;
        }
        cursor += size;
    }
    // Columns sub-cells by degree. Stable sort by degree, then put each
    // distinct-degree run into its own cell. (Tried Bouyukliev §3.3's
    // per-stratum-incidence fingerprint here as a strictly finer initial
    // partition; the fingerprint-construction cost outweighed nauty's
    // refinement savings at every N from 18 through 22 — measured ~1-5%
    // regression. Reverted to degree-only.)
    scratch.col_by_deg.clear();
    scratch.col_by_deg.extend(
        (0..r as c_int).map(|j| (scratch.d[l + j as usize], (l as c_int) + j)),
    );
    scratch.col_by_deg.sort_unstable_by_key(|&(deg, _)| deg);
    let col_start = scratch.lab.len();
    for &(_, vid) in &scratch.col_by_deg {
        scratch.lab.push(vid);
        scratch.ptn.push(1);
    }
    for i in 0..r.saturating_sub(1) {
        if scratch.col_by_deg[i].0 != scratch.col_by_deg[i + 1].0 {
            scratch.ptn[col_start + i] = 0;
        }
    }
    if !scratch.ptn.is_empty() {
        let last = scratch.ptn.len() - 1;
        scratch.ptn[last] = 0;
    }

    Some((nde, l))
}

/// Q_D-graph canonicaliser: nauty over the low-weight-codeword × column
/// bipartite encoding.
///
/// Returns `Some(NativeCanonInfo)` on success; `None` when the span-aware
/// builder gave up (caller falls back to `canon::canon_info_native`). The
/// shape of `NativeCanonInfo` is identical so the dispatch site needs no
/// extra plumbing.
pub fn canon_info_qd_native(rref: &[BinVec], n: u32) -> Option<NativeCanonInfo> {
    SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        canon_info_qd_native_impl(rref, n, &mut scratch)
    })
}

fn canon_info_qd_native_impl(
    rref: &[BinVec],
    n: u32,
    scratch: &mut CanonScratch,
) -> Option<NativeCanonInfo> {
    let (nde, l) = build_low_weight_sparsegraph(rref, n, scratch)?;
    let r: usize = n as usize;
    let total = l + r;

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

    scratch.orbits.clear();
    scratch.orbits.resize(total, 0);
    let mut options = optionblk::default_sparse();
    options.getcanon = TRUE;
    options.defaultptn = FALSE;
    options.userautomproc = Some(auto_callback);
    // schreier=TRUE measured net regression on the Q_D-graph too — see
    // `canon_info_native` for the audit reference.
    let mut stats = statsblk::default();

    // Canonical-graph output buffers (nauty needs writable storage even
    // when we discard the canonical sparsegraph).
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
        use crate::experimental::canon_hist::{push as nauty_hist_push, NautyCallRecord};
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
            qd_path: true,
        });
    }

    // Restrict canonical labelling to columns and invert into old→new form
    // in a single pass. Result Vec sizes are O(n) — leave fresh-allocated.
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

    Some(NativeCanonInfo {
        canonical_column_order,
        aut_generators,
        column_orbits,
        grpsize1: stats.grpsize1,
        grpsize2: stats.grpsize2 as i32,
        numnodes: stats.numnodes as u64,
        tctotal: stats.tctotal as u64,
        maxlevel: stats.maxlevel as i32,
        numgenerators: stats.numgenerators as i32,
    })
}
