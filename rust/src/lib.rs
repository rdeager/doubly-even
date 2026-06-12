//! Native hot kernel for the doubly-even enumerator.
//!
//! Production hot path:
//!
//! - `doubly_even_candidates_q` — the fat per-parent call that runs the
//!   Q_basis → σ_Q tables → singular BFS → orbit-min → lift pipeline in
//!   Rust (replaces the Python `enumerate.quotient.doubly_even_candidates_Q`
//!   spine).
//! - `canon_info_native` / `canon_info_qd_native` — native sparsenauty
//!   canonicalisers (D10 Q_D-graph default, full G(C) fallback).
//! - `subspace_in_orbit` — the McKay parent test inner-loop BFS.
//! - `enumerate_doubly_even` — the full canonical-augmentation recursion;
//!   parallel-16 entry when built with `--features parallel`.
//! - `kernel_build_info` — A/B harness identifier.
//!
//! This crate is the thin pyo3 wrapper (FFI conversion + the mimalloc
//! global allocator); every algorithm lives in the workspace member
//! `core/` (`doubly_even_core`). Dormant Python-facing wrappers (debug
//! submodule, Feulner port, WL + T11/T12/T13 invariants,
//! parallel_profiling, nauty_hist) live in `crate::py_exports` and are
//! registered at module init via `py_exports::register(m)`.

mod py_exports;

use doubly_even_core::{candidates, canon, enumerate, qd_graph, subspace_orbit, types};

// D13-V4: see Cargo.toml mimalloc dep for rationale. Per-thread arena
// allocator; replaces glibc ptmalloc which becomes the contention surface
// under 20 workers each allocating ~210 KB / call into canon_info_*.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::prelude::*;

use doubly_even_core::types::{BinVec, ColPerm};

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
    // The direct API always computes the canonical labelling — tests and
    // Python-side consumers depend on it (the autom-only lever applies
    // only inside the enumeration recursion).
    let info = canon::canon_info_native(&rref, n, true);
    (
        info.canonical_column_order
            .expect("get_canon=true always yields the label"),
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
    qd_graph::canon_info_qd_native(&rref, n, true).map(|info| {
        (
            info.canonical_column_order
                .expect("get_canon=true always yields the label"),
            info.aut_generators,
            info.grpsize1,
            info.grpsize2,
            info.column_orbits,
        )
    })
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
/// Plus a `stats: Vec[int]` (length 51) and a `per_k_stats: list[list[int]]`
/// (19 rows) — see `enumerate::enumerate_doubly_even` doc for the field
/// layout, mirrored in `scripts/bench.py` (`KERNEL_STATS_LAYOUT`,
/// `PER_K_STATS_ROWS`).
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
    debug_assert_eq!(
        stats.len(),
        doubly_even_core::enumerate::KERNEL_STATS_LAYOUT.len(),
        "stats vector length mismatch"
    );
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

/// Streaming variant of [`py_enumerate_doubly_even`] for the N >= 28
/// frontier. Writes one binary file per worker under `output_dir`
/// (format: `crate::streaming`). Returns a dict with the validated
/// `mass[k]` snapshot + flat stats vector + per-k stats matrix.
///
/// The kernel runs `assert mass[k] == quota[k]` for every `k = 0..=max_k`
/// before returning; any mismatch raises (via panic surfaced through
/// pyo3 as `PanicException`).
///
/// `num_threads` semantics mirror `py_enumerate_doubly_even`:
/// `None`/`0`/`1` → sequential; `>= 2` → parallel (requires
/// `--features parallel`).
#[pyfunction]
#[pyo3(name = "enumerate_doubly_even_streaming",
       signature = (n, max_k, quota, factorial_n, output_dir, num_threads=None))]
fn py_enumerate_doubly_even_streaming(
    py: Python<'_>,
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    output_dir: std::path::PathBuf,
    num_threads: Option<u32>,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    use pyo3::types::PyDict;
    if n > types::MAX_N {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n = {n} exceeds MAX_N = {}; the u64 kernel supports N up to 64",
            types::MAX_N,
        )));
    }
    if !output_dir.is_dir() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "output_dir does not exist or is not a directory: {output_dir:?}"
        )));
    }
    let nt = num_threads.unwrap_or(0);
    let result = if nt >= 2 {
        #[cfg(feature = "parallel")]
        {
            py.allow_threads(|| {
                enumerate::enumerate_doubly_even_parallel_streaming(
                    n,
                    max_k,
                    quota,
                    factorial_n,
                    nt as usize,
                    &output_dir,
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
        py.allow_threads(|| {
            enumerate::enumerate_doubly_even_streaming(
                n,
                max_k,
                quota,
                factorial_n,
                &output_dir,
            )
        })
    };

    let dict = PyDict::new(py);
    // mass[k] as decimal strings so Python receives unbounded ints
    // without u128 -> int conversion losing precision at N >= 21.
    let mass_strs: Vec<String> = result.mass.iter().map(|m| m.to_string()).collect();
    dict.set_item("mass", mass_strs)?;
    // stats vector: same 51-field layout as the in-memory entry. Convert
    // to decimal strings for the same precision reason (some fields are
    // cumulative ns — easily > 2^53 at N = 26).
    let stats_strs: Vec<String> = result.stats.iter().map(|s| s.to_string()).collect();
    dict.set_item("stats", stats_strs)?;
    dict.set_item("per_k_stats", result.per_k_stats)?;
    dict.set_item("n", n)?;
    dict.set_item("max_k", max_k)?;
    dict.set_item("num_threads", nt)?;
    Ok(dict.into())
}

/// Names of the kernel stats vector / per-rank rows, in kernel order.
/// The Rust consts in `core::enumerate::stats` are the single source of
/// truth; `scripts/bench.py` consumes this at import (with a frozen
/// fallback for pre-workspace wheels). Doubles as a "new wheel actually
/// installed" probe for the bench harness.
#[pyfunction]
#[pyo3(name = "kernel_stats_layout")]
fn py_kernel_stats_layout() -> (Vec<&'static str>, Vec<&'static str>) {
    (
        doubly_even_core::enumerate::KERNEL_STATS_LAYOUT.to_vec(),
        doubly_even_core::enumerate::PER_K_STATS_ROWS.to_vec(),
    )
}

/// Compile-time target features of the installed wheel. The gate for the
/// x86-64-v3 codegen ship: after an install, `avx2` must be `true` on a
/// v3 build (cargo config discovery is cwd-based, so a silently-ignored
/// `.cargo/config.toml` would otherwise produce an unflagged wheel that
/// looks identical).
#[pyfunction]
#[pyo3(name = "kernel_target_features")]
fn py_kernel_target_features() -> Vec<(&'static str, bool)> {
    vec![
        ("x86_64", cfg!(target_arch = "x86_64")),
        ("aarch64", cfg!(target_arch = "aarch64")),
        ("avx2", cfg!(target_feature = "avx2")),
        ("bmi2", cfg!(target_feature = "bmi2")),
        ("popcnt", cfg!(target_feature = "popcnt")),
    ]
}

/// Build identifier — `"verifier"` when compiled with the
/// `equivalence_verifier` feature, otherwise `"baseline"`. Used by the
/// Python A/B harness to confirm which kernel is loaded.
#[pyfunction]
#[pyo3(name = "kernel_build_info")]
fn py_kernel_build_info() -> &'static str {
    enumerate::build_info()
}

// ------------------------------------------------------ module assembly

#[pymodule]
fn doubly_even_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Production entry points.
    m.add_function(wrap_pyfunction!(py_doubly_even_candidates_q, m)?)?;
    m.add_function(wrap_pyfunction!(py_canon_info_native, m)?)?;
    m.add_function(wrap_pyfunction!(py_canon_info_qd_native, m)?)?;
    m.add_function(wrap_pyfunction!(py_subspace_in_orbit, m)?)?;
    m.add_function(wrap_pyfunction!(py_enumerate_doubly_even, m)?)?;
    m.add_function(wrap_pyfunction!(py_enumerate_doubly_even_streaming, m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_stats_layout, m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_target_features, m)?)?;

    // Dormant wrappers (debug submodule, Feulner, invariants, audit features).
    py_exports::register(m)?;

    Ok(())
}
