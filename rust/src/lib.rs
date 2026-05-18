//! Native hot kernel for the doubly-even enumerator.
//!
//! Milestone 5(b): exposes `doubly_even_candidates_q` — the fat per-parent
//! call that replaces the Python `enumerate.quotient.doubly_even_candidates_Q`
//! pipeline (Q_basis → σ_Q tables → singular BFS → orbit-min → lift). A
//! `debug` submodule exposes each stage individually for cross-check tests.

pub mod candidates;
pub mod canon;
pub mod enumerate;
pub mod feulner;
pub mod linalg;
pub mod orbit;
pub mod paired_iso;
pub mod permutations;
pub mod quotient;
pub mod subspace_orbit;
#[cfg(feature = "t11_cache")]
pub mod t11_blocklist;
pub mod types;

use pyo3::prelude::*;

use crate::types::{BinVec, ColPerm, Mat};

// --------------------------------------------------- main FFI entry point

/// Production hot path: see `candidates::doubly_even_candidates_q`.
///
/// Per-parent call shape:
///
/// - `n`: code length (must satisfy `n <= types::MAX_N`).
/// - `code_rref`: RREF rows of `C`, length-`rank(C)`.
/// - `pivots`: pivot columns of `code_rref` (parallel array).
/// - `dual_basis`: `C.dual().basis` from the Python side, raw — the kernel
///   re-reduces inside `Q_basis`.
/// - `aut_generators`: column permutations from `canon.nauty.canon_info`.
///
/// Returns the sorted `F_2^N` reps of doubly-even 1-dim extensions of `C`.
#[pyfunction]
#[pyo3(name = "doubly_even_candidates_q")]
fn py_doubly_even_candidates_q(
    n: u32,
    code_rref: Vec<BinVec>,
    pivots: Vec<u32>,
    dual_basis: Vec<BinVec>,
    aut_generators: Vec<ColPerm>,
) -> PyResult<Vec<BinVec>> {
    if n > types::MAX_N {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n = {n} exceeds MAX_N = {}; the u64 kernel supports N up to 64",
            types::MAX_N,
        )));
    }
    Ok(candidates::doubly_even_candidates_q(
        n,
        &code_rref,
        &pivots,
        &dual_basis,
        &aut_generators,
    ))
}

// ----------------------------------------------- debug-submodule wrappers
//
// These mirror the public Rust API one-to-one so `tests/test_kernel.py`
// can compare each pipeline stage against its Python reference. They are
// not on the hot path — production code calls `doubly_even_candidates_q`.

#[pyfunction]
#[pyo3(name = "kernel_version")]
fn py_kernel_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
#[pyo3(name = "popcount_batch")]
fn py_popcount_batch(words: Vec<u64>) -> Vec<u32> {
    words.iter().map(|w| w.count_ones()).collect()
}

#[pyfunction]
#[pyo3(name = "q_basis")]
fn py_q_basis(
    rref: Vec<BinVec>,
    pivots: Vec<u32>,
    dual_basis: Vec<BinVec>,
    n: u32,
) -> (Vec<BinVec>, Vec<u32>) {
    quotient::q_basis(&rref, &pivots, &dual_basis, n)
}

#[pyfunction]
#[pyo3(name = "lift")]
fn py_lift(u_q: BinVec, v_basis: Vec<BinVec>) -> BinVec {
    quotient::lift(u_q, &v_basis)
}

#[pyfunction]
#[pyo3(name = "project")]
fn py_project(v_in_v: BinVec, pivots_v: Vec<u32>) -> BinVec {
    quotient::project(v_in_v, &pivots_v)
}

#[pyfunction]
#[pyo3(name = "aut_image_on_q")]
fn py_aut_image_on_q(
    aut_generators: Vec<ColPerm>,
    rref: Vec<BinVec>,
    pivots: Vec<u32>,
    v_basis: Vec<BinVec>,
    pivots_v: Vec<u32>,
) -> Vec<Mat> {
    quotient::aut_image_on_q(&aut_generators, &rref, &pivots, &v_basis, &pivots_v)
}

#[pyfunction]
#[pyo3(name = "singular_reps_q")]
fn py_singular_reps_q(v_basis: Vec<BinVec>) -> Vec<BinVec> {
    orbit::singular_reps_q(&v_basis)
}

#[pyfunction]
#[pyo3(name = "sigma_q_table")]
fn py_sigma_q_table(sigma_q: Vec<BinVec>, l: u32) -> Vec<BinVec> {
    orbit::sigma_q_table(&sigma_q, l)
}

#[pyfunction]
#[pyo3(name = "aut_orbit_minima_q_table")]
fn py_aut_orbit_minima_q_table(
    reps_q: Vec<BinVec>,
    sigma_qs: Vec<Mat>,
    l: u32,
) -> Vec<BinVec> {
    orbit::aut_orbit_minima_q_table(&reps_q, &sigma_qs, l)
}

#[pyfunction]
#[pyo3(name = "aut_orbit_minima_q_witt")]
fn py_aut_orbit_minima_q_witt(
    reps_q: Vec<BinVec>,
    sigma_qs: Vec<Mat>,
    l: u32,
) -> Vec<BinVec> {
    orbit::aut_orbit_minima_q_witt(&reps_q, &sigma_qs, l)
}

// ------------------------------------------------------ canon FFI surface

/// Native replacement for `canon.nauty.canon_info` + `canon.bipartite.bipartite_graph`.
///
/// Takes the RREF basis of a code and its length; runs nauty on the bipartite
/// codeword × column encoding internally (no Python dict, no double FFI hop).
/// Returns a tuple matching the fields of `CanonInfo`:
///
///   `(canonical_column_order, aut_generators, grpsize1, grpsize2, column_orbits)`
///
/// `grpsize1`/`grpsize2` are nauty's native group-order float pair; Python is
/// already set up to convert these to an exact int via `_trustable_pynauty_order`
/// (falling back to Schreier-Sims when the float is past `2^53`).
#[pyfunction]
#[pyo3(name = "canon_info_native")]
fn py_canon_info_native(
    rref: Vec<BinVec>,
    n: u32,
) -> (Vec<u32>, Vec<Vec<u32>>, f64, i32, Vec<u32>) {
    let info = canon::canon_info_native(&rref, n);
    (
        info.canonical_column_order,
        info.aut_generators,
        info.grpsize1,
        info.grpsize2,
        info.column_orbits,
    )
}

/// Q_D-graph canonicaliser: nauty over the low-weight-codeword × column
/// bipartite encoding. Returns `None` when the span-aware build gives up
/// (caller falls back to [`py_canon_info_native`]). Used for differential
/// tests; the production hot path is inside `enumerate_doubly_even`.
#[pyfunction]
#[pyo3(name = "canon_info_qd_native")]
fn py_canon_info_qd_native(
    rref: Vec<BinVec>,
    n: u32,
) -> Option<(Vec<u32>, Vec<Vec<u32>>, f64, i32, Vec<u32>)> {
    canon::canon_info_qd_native(&rref, n).map(|info| {
        (
            info.canonical_column_order,
            info.aut_generators,
            info.grpsize1,
            info.grpsize2,
            info.column_orbits,
        )
    })
}

/// Feulner-style canonicaliser — column-side partition refinement, no nauty.
///
/// Same contract as [`py_canon_info_native`] but the algorithm avoids
/// materialising the `2^k + N`-vertex bipartite graph. Returns
/// `(canonical_column_order, aut_generators, aut_order_decimal,
/// column_orbits)`; Python converts the decimal string to an int.
#[pyfunction]
#[pyo3(name = "canon_info_feulner_native")]
fn py_canon_info_feulner_native(
    rref: Vec<BinVec>,
    n: u32,
) -> PyResult<(Vec<u32>, Vec<Vec<u32>>, String, Vec<u32>)> {
    if n > types::MAX_N {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n = {n} exceeds MAX_N = {}; the u64 kernel supports N up to 64",
            types::MAX_N,
        )));
    }
    let info = feulner::canon_info_feulner(&rref, n);
    Ok((
        info.canonical_column_order,
        info.aut_generators,
        info.aut_order_decimal,
        info.column_orbits,
    ))
}

/// Feulner-style canonicaliser — returns (leaves_visited, prunes_fired)
/// for diagnostics (Phase A).
#[pyfunction]
#[pyo3(name = "canon_info_feulner_counters")]
fn py_canon_info_feulner_counters(
    rref: Vec<BinVec>,
    n: u32,
) -> PyResult<(u64, u64)> {
    if n > types::MAX_N {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n = {n} exceeds MAX_N = {}", types::MAX_N,
        )));
    }
    let info = feulner::canon_info_feulner(&rref, n);
    Ok((info.leaves, info.prunes))
}

// ---------------------------------------- McKay subspace-orbit BFS surface

/// BFS test for `_in_aut_orbit_of_subspace` — the inner loop of the McKay
/// parent test.
///
/// Returns `true` iff some element of `⟨generators⟩` maps the subspace
/// with RREF basis `start_rref` to the subspace with RREF basis
/// `target_rref` under the column-permutation action on `F_2^n`.
///
/// Both `start_rref` and `target_rref` are already in RREF (the Python
/// side ensures it). The BFS applies each generator to every basis row,
/// re-RREFs, and compares; the seen-set is keyed on the RREF tuple.
#[pyfunction]
#[pyo3(name = "subspace_in_orbit")]
fn py_subspace_in_orbit(
    n: u32,
    start_rref: Vec<BinVec>,
    target_rref: Vec<BinVec>,
    generators: Vec<ColPerm>,
) -> PyResult<bool> {
    if n > types::MAX_N {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n = {n} exceeds MAX_N = {}; the u64 kernel supports N up to 64",
            types::MAX_N,
        )));
    }
    Ok(subspace_orbit::subspace_in_orbit(
        n,
        &start_rref,
        &target_rref,
        &generators,
    ))
}

// ----------------------------------------- enumerate_doubly_even (native)

/// Native canonical-augmentation enumerator for doubly-even codes.
///
/// Returns one tuple per canonical class:
///
///   `(rref, canonical_column_order, aut_generators, aut_order_decimal, column_orbits)`
///
/// Plus a `stats: Vec[int]` (length 26) and a `per_k_stats: list[list[int]]`
/// — see `enumerate::enumerate_doubly_even` doc for the field layout.
/// Packed as flat lists because pyo3 0.23 caps `IntoPyObject` tuples at 12
/// elements.
///
/// `quota[k]` must be `σ(N, k)`; `factorial_n` must be `N!`. Python computes
/// these via `gaborit_sigma` / `math.factorial` before the call.
///
/// `num_threads`: when `None` or `Some(0)` or `Some(1)` runs the sequential
/// driver (default — byte-identical to pre-D13 baseline). When the
/// `parallel` Cargo feature is on and `num_threads >= 2`, dispatches to
/// the outer-DFS worker-pool driver. When the feature is off, any value
/// > 1 raises `ValueError`.
#[pyfunction]
#[pyo3(name = "enumerate_doubly_even", signature = (n, max_k, quota, factorial_n, num_threads=None))]
fn py_enumerate_doubly_even(
    py: Python<'_>,
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    num_threads: Option<u32>,
) -> PyResult<(
    Vec<(Vec<BinVec>, Vec<u32>, Vec<Vec<u32>>, String, Vec<u32>)>,
    Vec<u128>,
    Vec<Vec<u64>>,
)> {
    if n > types::MAX_N {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n = {n} exceeds MAX_N = {}; the u64 kernel supports N up to 64",
            types::MAX_N,
        )));
    }
    let nt = num_threads.unwrap_or(0);
    let (out, stats, per_k) = if nt >= 2 {
        #[cfg(feature = "parallel")]
        {
            // Release the GIL so the worker threads (which never touch
            // Python state) can run in parallel with the main thread.
            py.allow_threads(|| {
                enumerate::enumerate_doubly_even_parallel(
                    n,
                    max_k,
                    quota,
                    factorial_n,
                    nt as usize,
                )
            })
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = py;
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_threads >= 2 requires the kernel to be built with \
                 --features parallel (D13 outer-DFS parallelism)",
            ));
        }
    } else {
        let _ = py;
        enumerate::enumerate_doubly_even(n, max_k, quota, factorial_n)
    };
    debug_assert_eq!(stats.len(), 30, "stats vector length mismatch");
    let result: Vec<_> = out
        .into_iter()
        .map(|e| {
            (
                e.rref,
                e.canonical_column_order,
                e.aut_generators,
                e.aut_order.to_string(),
                e.column_orbits,
            )
        })
        .collect();
    Ok((result, stats, per_k))
}

/// Build identifier — `"verifier"` when compiled with the
/// `equivalence_verifier` feature, otherwise `"baseline"`. Used by the
/// Python A/B harness to confirm which kernel is loaded.
#[pyfunction]
#[pyo3(name = "kernel_build_info")]
fn py_kernel_build_info() -> &'static str {
    enumerate::build_info()
}

/// T11 column-stratified weight-profile hash (canon.rs::compute_t11_hash).
///
/// Returned as Python `int` (u64-safe). Exposed for the
/// `scripts/dump_t11_blocklist.py` helper that generates the per-N
/// blocklist const arrays for `t11_blocklist.rs`.
#[cfg(feature = "t11_cache")]
#[pyfunction]
#[pyo3(name = "compute_t11_hash")]
fn py_compute_t11_hash(rref: Vec<BinVec>, n: u32) -> u64 {
    canon::compute_t11_hash(&rref, n)
}

// ------------------------------------------------------ module assembly

#[pymodule]
fn doubly_even_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Production entry points.
    m.add_function(wrap_pyfunction!(py_doubly_even_candidates_q, m)?)?;
    m.add_function(wrap_pyfunction!(py_canon_info_native, m)?)?;
    m.add_function(wrap_pyfunction!(py_canon_info_qd_native, m)?)?;
    m.add_function(wrap_pyfunction!(py_canon_info_feulner_native, m)?)?;
    m.add_function(wrap_pyfunction!(py_subspace_in_orbit, m)?)?;
    m.add_function(wrap_pyfunction!(py_enumerate_doubly_even, m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_build_info, m)?)?;
    #[cfg(feature = "t11_cache")]
    m.add_function(wrap_pyfunction!(py_compute_t11_hash, m)?)?;

    // Stage-level helpers under `doubly_even_kernel.debug`.
    let debug = PyModule::new(m.py(), "debug")?;
    debug.add_function(wrap_pyfunction!(py_kernel_version, &debug)?)?;
    debug.add_function(wrap_pyfunction!(py_popcount_batch, &debug)?)?;
    debug.add_function(wrap_pyfunction!(py_q_basis, &debug)?)?;
    debug.add_function(wrap_pyfunction!(py_lift, &debug)?)?;
    debug.add_function(wrap_pyfunction!(py_project, &debug)?)?;
    debug.add_function(wrap_pyfunction!(py_aut_image_on_q, &debug)?)?;
    debug.add_function(wrap_pyfunction!(py_singular_reps_q, &debug)?)?;
    debug.add_function(wrap_pyfunction!(py_sigma_q_table, &debug)?)?;
    debug.add_function(wrap_pyfunction!(py_aut_orbit_minima_q_table, &debug)?)?;
    debug.add_function(wrap_pyfunction!(py_aut_orbit_minima_q_witt, &debug)?)?;
    debug.add_function(wrap_pyfunction!(py_canon_info_feulner_counters, &debug)?)?;
    m.add_submodule(&debug)?;
    Ok(())
}
