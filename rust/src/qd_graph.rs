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

use crate::canon::{auto_callback, NativeCanonInfo, AUT_BUFFER, LEFT_VERTEX_COUNT};
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
fn collect_low_weight_codewords(rref: &[BinVec], n: u32) -> Option<(Vec<BinVec>, Vec<usize>)> {
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
    let mut by_weight: Vec<Vec<BinVec>> = vec![Vec::new(); (n as usize) + 1];
    let mut w: BinVec = 0;
    for mask in 1..total_codewords {
        let lo_bit = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        w ^= rref[lo_bit];
        by_weight[w.count_ones() as usize].push(w);
    }

    // Walk strata in ascending weight, growing the accumulated low-weight
    // set until its span equals C. We rebuild RREF each round on the
    // accumulated set; cheap relative to the nauty call we're about to skip.
    let mut accum: Vec<BinVec> = Vec::new();
    let mut stratum_sizes: Vec<usize> = Vec::new();
    for weight in 1..=(n as usize) {
        if by_weight[weight].is_empty() {
            continue;
        }
        let stratum = std::mem::take(&mut by_weight[weight]);
        let stratum_len = stratum.len();
        accum.extend(stratum);
        stratum_sizes.push(stratum_len);
        if accum.len() > bail_threshold {
            return None;
        }
        let (rr, _) = row_reduce(&accum, n);
        if rr.len() == k {
            return Some((accum, stratum_sizes));
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
pub(crate) fn build_low_weight_sparsegraph(
    rref: &[BinVec],
    n: u32,
) -> Option<(
    Vec<usize>, // v
    Vec<i32>,   // d
    Vec<i32>,   // e
    Vec<c_int>, // lab
    Vec<c_int>, // ptn
    usize,      // L (number of codeword-side vertices)
)> {
    let (codewords, stratum_sizes) = collect_low_weight_codewords(rref, n)?;
    let l: usize = codewords.len();
    let r: usize = n as usize;
    let total = l + r;

    // Per-vertex degrees.
    let mut d = vec![0i32; total];
    for (i, &cw) in codewords.iter().enumerate() {
        d[i] = cw.count_ones() as i32;
    }
    for j in 0..r {
        let bit = 1u64 << j;
        let mut deg = 0i32;
        for &cw in &codewords {
            if cw & bit != 0 {
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

    // Edge lists: walk each codeword's set bits once.
    let mut e = vec![0i32; nde];
    let mut left_write = v.clone();
    let mut right_lists: Vec<Vec<i32>> =
        (0..r).map(|j| Vec::with_capacity(d[l + j] as usize)).collect();
    for (i, &cw) in codewords.iter().enumerate() {
        let mut bits = cw;
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

    // Initial partition. Codeword side: cells in stratum order (each weight
    // stratum is one cell). Column side: subdivide by degree (the existing
    // bipartite encoding does the same — see `canon_info_native` in canon.rs).
    // `ptn[i] = 0` marks the end of a cell.
    let mut lab: Vec<c_int> = Vec::with_capacity(total);
    let mut ptn: Vec<c_int> = Vec::with_capacity(total);
    let mut cursor = 0usize;
    for &size in &stratum_sizes {
        for idx in cursor..(cursor + size) {
            lab.push(idx as c_int);
            ptn.push(1);
        }
        // mark end of stratum
        if !ptn.is_empty() {
            let last = ptn.len() - 1;
            ptn[last] = 0;
        }
        cursor += size;
    }
    // Columns sub-cells by degree. Stable sort by degree, then put each
    // distinct-degree run into its own cell. (Tried Bouyukliev §3.3's
    // per-stratum-incidence fingerprint here as a strictly finer initial
    // partition; the fingerprint-construction cost outweighed nauty's
    // refinement savings at every N from 18 through 22 — measured ~1-5%
    // regression. Reverted to degree-only.)
    let mut col_by_deg: Vec<(i32, c_int)> = (0..r as c_int)
        .map(|j| (d[l + j as usize], (l as c_int) + j))
        .collect();
    col_by_deg.sort_unstable_by_key(|&(deg, _)| deg);
    let col_start = lab.len();
    for &(_, vid) in &col_by_deg {
        lab.push(vid);
        ptn.push(1);
    }
    for i in 0..r.saturating_sub(1) {
        if col_by_deg[i].0 != col_by_deg[i + 1].0 {
            ptn[col_start + i] = 0;
        }
    }
    if !ptn.is_empty() {
        let last = ptn.len() - 1;
        ptn[last] = 0;
    }

    Some((v, d, e, lab, ptn, l))
}

/// Q_D-graph canonicaliser: nauty over the low-weight-codeword × column
/// bipartite encoding.
///
/// Returns `Some(NativeCanonInfo)` on success; `None` when the span-aware
/// builder gave up (caller falls back to `canon::canon_info_native`). The
/// shape of `NativeCanonInfo` is identical so the dispatch site needs no
/// extra plumbing.
pub fn canon_info_qd_native(rref: &[BinVec], n: u32) -> Option<NativeCanonInfo> {
    let (mut v, mut d, mut e, mut lab, mut ptn, l) = build_low_weight_sparsegraph(rref, n)?;
    let r: usize = n as usize;
    let total = l + r;

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

    let mut orbits = vec![0i32; total];
    let mut options = optionblk::default_sparse();
    options.getcanon = TRUE;
    options.defaultptn = FALSE;
    options.userautomproc = Some(auto_callback);
    // schreier=TRUE measured net regression on the Q_D-graph too — see
    // `canon_info_native` for the audit reference.
    let mut stats = statsblk::default();

    // Canonical-graph output buffers (nauty needs writable storage even
    // when we discard the canonical sparsegraph).
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

    #[cfg(feature = "nauty_hist")]
    let nauty_t0 = std::time::Instant::now();

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

    // Restrict canonical labelling to columns and invert into old→new form.
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
