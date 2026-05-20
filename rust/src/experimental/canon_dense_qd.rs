//! crate::experimental::canon_dense_qd — dense Q_D-graph canonicaliser.
//!
//! Dormant audit substrate behind Cargo feature `dense_qd` (Q6 audit
//! Phase 2). At our graph size (`|C_low| + N ≤ ~54` vertices), the dense
//! adjacency is a single 64-bit setword per row — ≤ 432 bytes, fits in 7
//! cache lines. Sparse `refine_sg` pointer-chases through `e[]`/`v[]`/`d[]`
//! triples per cell-membership query; dense `refine` operates on bitwise
//! AND of two setwords. Cache footprint and refinement-primitive cost
//! could plausibly favour dense at this size — every benched variant
//! regressed against sparsenauty's 78 µs/call floor (see
//! `memory/project_q6_audit_closed.md`).
//!
//! The optional `dense_qd_refinvar` feature plugs nauty's `refinvar` into
//! `options.invarproc` (one level of vertex-invariant refinement); also
//! regressed in audit. The extern declaration lives here because the
//! symbol isn't re-exported by `nauty-Traces-sys`.

use std::ffi::c_int;

use nauty_Traces_sys::{
    densenauty, empty_graph, graph, optionblk, statsblk, ADDONEEDGE, FALSE, SETWORDSNEEDED, TRUE,
};
#[cfg(feature = "dense_qd_refinvar")]
use nauty_Traces_sys::{boolean, graph as nauty_graph};

use crate::canon::{auto_callback, NativeCanonInfo, AUT_BUFFER, LEFT_VERTEX_COUNT};
use crate::qd_graph::build_low_weight_sparsegraph;
use crate::types::BinVec;

// `refinvar` is not re-exported by `nauty-Traces-sys` (only `adjacencies`
// and `adjacencies_sg` are). It is in libnauty under its C symbol name;
// declare a Rust extern that matches `optionblk::invarproc`'s signature.
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
