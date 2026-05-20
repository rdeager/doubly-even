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
#[cfg(feature = "dense_qd")]
use nauty_Traces_sys::{densenauty, empty_graph, graph, ADDONEEDGE, SETWORDSNEEDED};
// `refinvar` is not re-exported by `nauty-Traces-sys` (only `adjacencies`
// and `adjacencies_sg` are). It is in libnauty under its C symbol name;
// declare a Rust extern that matches `optionblk::invarproc`'s signature
// — see the bindings: 11 args, no return.
#[cfg(feature = "dense_qd_refinvar")]
use nauty_Traces_sys::{boolean, graph as nauty_graph};
#[cfg(feature = "dense_qd_refinvar")]
extern "C" {
    fn refinvar(
        g: *mut nauty_graph,
        lab: *mut c_int,
        ptn: *mut c_int,
        level: c_int,
        numcells: c_int,
        tvpos: c_int,
        invar: *mut c_int,
        invararg: c_int,
        digraph: boolean,
        m: c_int,
        n: c_int,
    );
}
#[cfg(feature = "traces_qd")]
use nauty_Traces_sys::{Traces, TracesOptions, TracesStats};

use crate::linalg::row_reduce;
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
    static AUT_BUFFER: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
    /// Number of left-side (codeword) vertices; the callback needs to know
    /// where to slice the column part out of the full permutation.
    static LEFT_VERTEX_COUNT: RefCell<u32> = const { RefCell::new(0) };
}

// E1/E2 measurement substrate. Off by default. See Cargo feature
// `nauty_hist` doc + research plan
// `any-speed-improvement-to-tender-wave`.
#[cfg(feature = "nauty_hist")]
#[derive(Clone, Copy)]
pub struct NautyCallRecord {
    pub elapsed_ns: u64,
    pub numnodes: u64,
    pub tctotal: u64,
    pub maxlevel: i32,
    pub numgenerators: i32,
    pub left_vertices: u32,
    pub right_vertices: u32,
    /// Code rank `k = rref.len()`. With this and `left_vertices` we can
    /// reconstruct the Q_D-vs-fallback bucketing: `qd_path` iff
    /// `left_vertices < (1 << rank)` (the low-weight subset is strictly
    /// smaller than the full codeword set) or `qd_path` is `true` and
    /// `left_vertices == (1 << rank) - 1` only in the degenerate
    /// rank-0 → fallback case.
    pub rank: u32,
    /// `true` if this record came from `canon_info_qd_native` (Q_D
    /// low-weight path); `false` if from `canon_info_native` — either
    /// because Q_D bailed (`collect_low_weight_codewords` returned
    /// `None`) or because the dispatch in `enumerate::canon_info`
    /// chose the full-bipartite path directly.
    pub qd_path: bool,
}

#[cfg(feature = "nauty_hist")]
static NAUTY_HIST: std::sync::Mutex<Vec<NautyCallRecord>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(feature = "nauty_hist")]
#[inline]
fn nauty_hist_push(rec: NautyCallRecord) {
    if let Ok(mut h) = NAUTY_HIST.lock() {
        h.push(rec);
    }
}

#[cfg(feature = "nauty_hist")]
pub fn nauty_hist_drain() -> Vec<NautyCallRecord> {
    let mut h = NAUTY_HIST.lock().expect("nauty hist mutex poisoned");
    std::mem::take(&mut *h)
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

// Traces' userautomproc has a different signature than sparsenauty's:
// (count, perm, n) instead of (count, perm, orbits, numorbits, stabvertex, n).
#[cfg(feature = "traces_qd")]
unsafe extern "C" fn auto_callback_traces(
    _count: c_int,
    perm: *mut c_int,
    n: c_int,
) {
    let l = LEFT_VERTEX_COUNT.with(|cell| *cell.borrow());
    let n = n as usize;
    let mut col_perm = vec![0u32; n - l as usize];
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

    // Initial partition by (side, degree). Side first keeps codewords (left)
    // and columns (right) in separate cells — automorphisms must respect the
    // bipartition. Within each side, sub-cells by degree: codeword weight
    // on the left, column-incidence count on the right. Degree is Aut-
    // invariant so distinct degrees imply distinct orbits, letting nauty
    // skip its own initial degree-refinement pass (Bouyukliev §3.3,
    // matches Sage's default behaviour).
    let mut by_cell: Vec<(u8, i32, c_int)> = (0..total as c_int)
        .map(|v| {
            let side: u8 = if (v as usize) < l { 0 } else { 1 };
            (side, d[v as usize], v)
        })
        .collect();
    by_cell.sort_unstable_by_key(|&(side, deg, _)| (side, deg));

    let mut lab: Vec<c_int> = by_cell.iter().map(|&(_, _, v)| v).collect();
    let mut ptn = vec![1i32; total];
    // ptn[i] = 0 marks the last vertex of a cell. Set 0 wherever the
    // (side, degree) key changes at position i+1; always set 0 at the end.
    for i in 0..total.saturating_sub(1) {
        let (s1, d1, _) = by_cell[i];
        let (s2, d2, _) = by_cell[i + 1];
        if s1 != s2 || d1 != d2 {
            ptn[i] = 0;
        }
    }
    if total > 0 {
        ptn[total - 1] = 0;
    }
    let mut orbits = vec![0i32; total];

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
        numnodes: stats.numnodes as u64,
        tctotal: stats.tctotal as u64,
        maxlevel: stats.maxlevel as i32,
        numgenerators: stats.numgenerators as i32,
    }
}

// ---------------------------------------------------------------- Q_D-graph
//
// Low-weight-incidence variant of the bipartite encoding. Instead of all 2^k
// codewords on the left side, we use only enough low-weight codewords to
// span C; each Hamming-weight stratum is its own colour class. The column-
// side stabiliser of this graph equals Aut(C) whenever the chosen codewords
// span C (see plan file, "Math invariant we depend on"); the span check is
// part of the build.

/// Span-aware low-weight-codeword set.
///
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
fn collect_low_weight_codewords(
    rref: &[BinVec],
    n: u32,
) -> Option<(Vec<BinVec>, Vec<usize>)> {
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
fn build_low_weight_sparsegraph(
    rref: &[BinVec],
    n: u32,
) -> Option<(
    Vec<usize>,   // v
    Vec<i32>,     // d
    Vec<i32>,     // e
    Vec<c_int>,   // lab
    Vec<c_int>,   // ptn
    usize,        // L (number of codeword-side vertices)
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
    // bipartite encoding does the same — see `canon_info_native` lines
    // 188–209). `ptn[i] = 0` marks the end of a cell.
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
    let mut col_by_deg: Vec<(i32, c_int)> =
        (0..r as c_int).map(|j| (d[l + j as usize], (l as c_int) + j)).collect();
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
/// builder gave up (caller falls back to [`canon_info_native`]). The shape
/// of `NativeCanonInfo` is identical so the dispatch site needs no extra
/// plumbing.
///
/// Math: the column-side stabiliser of the low-weight-incidence graph
/// equals Aut(C) iff the included codewords span C; the builder enforces
/// this. See plan file's "Math invariant" section and
/// `memory/project-gpt-suggestions-audit.md` for context.
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

// ---------------------------------------------------- densenauty Q_D-graph
//
// Q6 audit Phase 2: at our graph size (`|C_low| + N ≤ ~54` vertices), the
// dense adjacency is a single 64-bit setword per row (m=1) — total ≤ 432
// bytes, fits in 7 cache lines. Sparse representation forces nauty's
// `refine_sg` to pointer-chase through `e[]`/`v[]`/`d[]` triples for every
// cell-membership query; dense `refine` operates on bitwise AND of two
// setwords. Both for cache footprint and refinement primitive cost, dense
// can win at this size — never measured.
//
// Build feature `dense_qd` swaps the dispatch in `enumerate.rs`. The sparse
// implementation above (`canon_info_qd_native`) stays compiled in as the
// fallback.

#[cfg(feature = "dense_qd")]
pub fn canon_info_qd_dense(rref: &[BinVec], n: u32) -> Option<NativeCanonInfo> {
    let (v, d, e, mut lab, mut ptn, l) = build_low_weight_sparsegraph(rref, n)?;
    let r: usize = n as usize;
    let total = l + r;

    // Dense adjacency: m setwords per row, total*m setwords total.
    let m = SETWORDSNEEDED(total);
    let mut g: Vec<graph> = empty_graph(m, total);
    // Walk left-side adjacency (codeword → column); ADDONEEDGE bidirects.
    for u in 0..l {
        let start = v[u];
        let end = start + d[u] as usize;
        for &w in &e[start..end] {
            ADDONEEDGE(&mut g, u, w as usize, m);
        }
    }
    let mut canong: Vec<graph> = empty_graph(m, total);

    let mut orbits = vec![0i32; total];

    let mut options = optionblk::default();
    options.getcanon = TRUE;
    options.defaultptn = FALSE;
    options.userautomproc = Some(auto_callback);
    #[cfg(feature = "dense_qd_tc0")]
    {
        options.tc_level = 0;
    }
    #[cfg(feature = "dense_qd_refinvar")]
    {
        options.invarproc = Some(refinvar);
        options.mininvarlevel = 1;
        options.maxinvarlevel = 1;
    }
    let mut stats = statsblk::default();

    AUT_BUFFER.with(|cell| cell.borrow_mut().clear());
    LEFT_VERTEX_COUNT.with(|cell| *cell.borrow_mut() = l as u32);

    unsafe {
        densenauty(
            g.as_mut_ptr(),
            lab.as_mut_ptr(),
            ptn.as_mut_ptr(),
            orbits.as_mut_ptr(),
            &mut options,
            &mut stats,
            m as c_int,
            total as c_int,
            canong.as_mut_ptr(),
        );
    }

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

// ------------------------------------------------------- Traces Q_D-graph
//
// Q6 audit Phase 3: Traces is "generally recommended for sparse graphs
// with many automorphisms, which describes our case" (audit §2(e)) but
// has never been benched. Identical sparse-graph input; different
// canonicalisation engine. `TracesOptions` lacks `invarproc` /
// `mininvarlevel` / `maxinvarlevel`.

#[cfg(feature = "traces_qd")]
pub fn canon_info_qd_traces(rref: &[BinVec], n: u32) -> Option<NativeCanonInfo> {
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
    let mut options = TracesOptions {
        getcanon: TRUE,
        defaultptn: FALSE,
        userautomproc: Some(auto_callback_traces),
        ..TracesOptions::default()
    };
    let mut stats = TracesStats::default();

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
        Traces(
            &mut sg,
            lab.as_mut_ptr(),
            ptn.as_mut_ptr(),
            orbits.as_mut_ptr(),
            &mut options,
            &mut stats,
            &mut canon_sg,
        );
    }

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

    // Traces doesn't expose numnodes/tctotal/maxlevel — emit zeros for
    // those Q6 counters; numgenerators is still in TracesStats.
    Some(NativeCanonInfo {
        canonical_column_order,
        aut_generators,
        column_orbits,
        grpsize1: stats.grpsize1,
        grpsize2: stats.grpsize2 as i32,
        numnodes: 0,
        tctotal: 0,
        maxlevel: 0,
        numgenerators: stats.numgenerators as i32,
    })
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
