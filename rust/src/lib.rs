//! Native hot kernel for the doubly-even enumerator.
//!
//! Milestone 5(b): exposes `doubly_even_candidates_q` — the fat per-parent
//! call that replaces the Python `enumerate.quotient.doubly_even_candidates_Q`
//! pipeline (Q_basis → σ_Q tables → singular BFS → orbit-min → lift). A
//! `debug` submodule exposes each stage individually for cross-check tests.

pub mod candidates;
pub mod linalg;
pub mod orbit;
pub mod quotient;
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

// ------------------------------------------------------ module assembly

#[pymodule]
fn doubly_even_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Production entry point.
    m.add_function(wrap_pyfunction!(py_doubly_even_candidates_q, m)?)?;

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
    m.add_submodule(&debug)?;
    Ok(())
}
