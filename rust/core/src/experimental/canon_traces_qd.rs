//! crate::experimental::canon_traces_qd — Traces engine over Q_D-graph.
//!
//! Dormant audit substrate behind Cargo feature `traces_qd` (Q6 audit
//! Phase 3). Traces is "generally recommended for sparse graphs with
//! many automorphisms, which describes our case" (audit §2(e)) but
//! never beat sparsenauty in practice — see
//! `memory/project_q6_audit_closed.md`. Identical sparse-graph input;
//! different canonicalisation engine. `TracesOptions` lacks
//! `invarproc` / `mininvarlevel` / `maxinvarlevel`.
//!
//! Incompatible with the `parallel` feature — Traces uses non-TLS
//! static work queues; the `compile_error!` guard is in `enumerate.rs`.

use std::ffi::c_int;
use std::ptr;

use nauty_Traces_sys::{sparsegraph, Traces, TracesOptions, TracesStats, FALSE, TRUE};

use crate::canon::{NativeCanonInfo, AUT_BUFFER, LEFT_VERTEX_COUNT, SCRATCH};
use crate::qd_graph::build_low_weight_sparsegraph;
use crate::types::BinVec;

// Traces' userautomproc has a different signature than sparsenauty's:
// (count, perm, n) instead of (count, perm, orbits, numorbits, stabvertex, n).
unsafe extern "C" fn auto_callback_traces(_count: c_int, perm: *mut c_int, n: c_int) {
    let l = LEFT_VERTEX_COUNT.with(|cell| *cell.borrow());
    let n = n as usize;
    let mut col_perm = vec![0u32; n - l as usize];
    for old in (l as usize)..n {
        let new_v = unsafe { *perm.add(old) } as u32;
        col_perm[old - l as usize] = new_v - l;
    }
    AUT_BUFFER.with(|cell| cell.borrow_mut().push(col_perm));
}

pub fn canon_info_qd_traces(rref: &[BinVec], n: u32) -> Option<NativeCanonInfo> {
    // Dormant audit substrate: snapshot the sparsegraph out of the shared
    // scratch into owned Vecs. Traces holds its own per-call state and
    // doesn't benefit from scratch reuse here.
    let (mut v, mut d, mut e, mut lab, mut ptn, l) = SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        let (_, l) = build_low_weight_sparsegraph(rref, n, &mut sc)?;
        Some((sc.v.clone(), sc.d.clone(), sc.e.clone(), sc.lab.clone(), sc.ptn.clone(), l))
    })?;
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
        canonical_column_order: Some(canonical_column_order),
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
