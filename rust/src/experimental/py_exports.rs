//! crate::experimental::py_exports — dormant Python-facing wrappers.
//!
//! Quarantined out of `lib.rs` (Phase 2 Cut 1, 2026-05-20). None of these
//! wrappers are on the production hot path; they exist as differential-test
//! substrate (the `debug` submodule mirrors each pipeline stage), as the
//! alternative canonicaliser entry points (Feulner port — dispatch closed
//! 2026-05-20; see `memory/project_feulner_dispatch_closed.md`), as
//! collision-experiment surface (WL + T11/T12/T13; see
//! `memory/project_1wl_collision_experiment.md` and
//! `memory/project_bitpar_refine_microbench.md`), and as audit-feature
//! entries (`parallel_profiling`, `nauty_hist`).
//!
//! All registration goes through [`register`]. `lib.rs` only adds the
//! seven production wrappers directly and delegates here for the rest.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::types::{BinVec, ColPerm, Mat};
use crate::{experimental, orbit, quotient, types};

// ----------------------------------------- debug-submodule wrappers
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

// ----------------------------------------- Feulner canonicaliser
//
// Dormant since 2026-05-20. The Feulner port (D9) remains in tree as a
// differential / diff-oracle substrate; the production canonicaliser is
// the Q_D-graph (D10) in `crate::canon`. See
// `memory/project_feulner_dispatch_closed.md`.

/// Feulner-style canonicaliser — column-side partition refinement, no nauty.
///
/// Same contract as `canon_info_native` but the algorithm avoids
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
    let info = experimental::feulner::canon_info_feulner(&rref, n);
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
fn py_canon_info_feulner_counters(rref: Vec<BinVec>, n: u32) -> PyResult<(u64, u64)> {
    if n > types::MAX_N {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n = {n} exceeds MAX_N = {}",
            types::MAX_N,
        )));
    }
    let info = experimental::feulner::canon_info_feulner(&rref, n);
    Ok((info.leaves, info.prunes))
}

/// Extended Feulner counters — `(leaves, prefix_prunes, clb_prunes)`.
/// Companion to the v1 entry above; new tests use this one. Kept
/// alongside the v1 entry so existing callers (and
/// `golay_canon_bench.py`) stay green during the CLB rollout.
#[pyfunction]
#[pyo3(name = "canon_info_feulner_counters_v2")]
fn py_canon_info_feulner_counters_v2(rref: Vec<BinVec>, n: u32) -> PyResult<(u64, u64, u64)> {
    if n > types::MAX_N {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n = {n} exceeds MAX_N = {}",
            types::MAX_N,
        )));
    }
    let info = experimental::feulner::canon_info_feulner(&rref, n);
    Ok((info.leaves, info.prunes, info.clb_prunes))
}

// ───────────────────────────────────────────── invariants (collision experiment)
//
// Substrate for `scripts/experimental/wl_collision_experiment.py`'s
// N=24/26 Rust port — 1-WL on (full G(C) | G_min) bipartite graphs +
// T11/T12/T13 cheap invariants. Not on the kernel hot path; pure
// standalone signatures.

/// 1-WL signature on the codeword × column bipartite graph.
///
/// - `graph`: `"min"` (low-weight spanning subset, falls back to full G(C)
///   when low-weight set exceeds 2^(k-1)) or `"full"` (all 2^k − 1 nonzero
///   codewords).
/// - `init`: `"vanilla"` (codewords colour 0, columns colour 1) or
///   `"degree_init"` (codewords by weight stratum, columns by degree).
///
/// Returns the sorted multiset of final per-vertex content-hashes
/// (side-tagged so left/right colour spaces are disjoint). Permutation
/// invariant; bucketing-equivalent to (but byte-incompatible with) the
/// Python wl_signature.
#[pyfunction]
#[pyo3(name = "wl_signature")]
fn py_wl_signature(
    rref: Vec<BinVec>,
    n: u32,
    graph: &str,
    init: &str,
) -> PyResult<(Vec<u64>, u32, bool)> {
    let g_opt = match graph {
        "min" => match experimental::invariants::build_low_weight_bipartite(&rref, n) {
            Some(g) => (g, false),
            None => (experimental::invariants::build_full_bipartite(&rref, n), true),
        },
        "full" => (experimental::invariants::build_full_bipartite(&rref, n), false),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "graph must be 'min' or 'full', got {other:?}"
            )))
        }
    };
    let init_mode = match init {
        "vanilla" => experimental::invariants::InitMode::Vanilla,
        "degree_init" => experimental::invariants::InitMode::DegreeAndWeight,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "init must be 'vanilla' or 'degree_init', got {other:?}"
            )))
        }
    };
    let (g, fallback) = g_opt;
    let (sig, rounds) = experimental::invariants::wl_refine(&g, init_mode);
    Ok((sig, rounds, fallback))
}

/// T11: per-column profile of (#wt-w cws through col j, for w in weights).
/// Computed over the full nonzero codeword set. Per-column tuple packed
/// losslessly into u128 (16 bits per count × ≤ 8 weights).
#[pyfunction]
#[pyo3(name = "t11_signature")]
fn py_t11_signature(rref: Vec<BinVec>, n: u32, weights: Vec<u32>) -> Vec<u128> {
    let cws = experimental::invariants::full_codewords(&rref, n);
    experimental::invariants::t11_signature(&cws, n, &weights)
}

/// T11 on the low-weight spanning subset (the "G_min" T11). Falls back to
/// full G(C) when low-weight bails.
#[pyfunction]
#[pyo3(name = "t11_signature_gmin")]
fn py_t11_signature_gmin(rref: Vec<BinVec>, n: u32, weights: Vec<u32>) -> Vec<u128> {
    let cws = match experimental::invariants::low_weight_codewords_pub(&rref, n) {
        Some((c, _)) => c,
        None => experimental::invariants::full_codewords(&rref, n),
    };
    experimental::invariants::t11_signature(&cws, n, &weights)
}

/// T12: sorted multiset over min-weight-codeword triples.
#[pyfunction]
#[pyo3(name = "t12_signature")]
fn py_t12_signature(rref: Vec<BinVec>, n: u32) -> Vec<u64> {
    let cws = experimental::invariants::full_codewords(&rref, n);
    let _ = n;
    experimental::invariants::t12_signature(&cws)
}

/// T13 (pair-gram): per-column-pair tuple of #wt-w cws containing both.
#[pyfunction]
#[pyo3(name = "t13_signature")]
fn py_t13_signature(rref: Vec<BinVec>, n: u32, weights: Vec<u32>) -> Vec<u64> {
    let cws = experimental::invariants::full_codewords(&rref, n);
    experimental::invariants::t13_signature(&cws, n, &weights)
}

/// Compute every signature for one code in a single Rust call.
///
/// Returns `(digests, fallback, metadata, component_nanos)`:
/// - `digests` is a list of 9 u128 hashes in this order:
///   `[wl_min_vanilla, wl_min_degree, wl_full_vanilla, wl_full_degree,
///     t11_full, t11_gmin, t12, t13, t13_min]`. Each entry is a 128-bit
///   content hash of the sorted multiset signature — collision
///   probability is ~10⁻²³ over a 494K-class N=26 run (negligible), and
///   per-code memory is fixed at 9 × 16 = 144 bytes regardless of |L+R|.
/// - `fallback` is true iff the low-weight builder bailed (full G(C) used
///   for wl_min variants).
/// - `metadata` is a 6-element list:
///   `[rounds_min_vanilla, rounds_min_degree, rounds_full_vanilla,
///     rounds_full_degree, l_min, l_full]`
/// - `component_nanos` is a 9-element list of per-component wall-time
///   nanoseconds (same index order as `digests`). Caller accumulates
///   these across codes to report per-component µs/code. Excludes the
///   shared codeword-set / bipartite-graph builds (those would
///   double-count across components).
///
/// `weights` is used for T11_full and T13 (typically (4, 8, 12, 16));
/// `gmin_weights` is used for T11_gmin (typically multiples of 4 up to N).
/// `t13_min` is T13 over the auto-detected minimum-weight stratum.
#[pyfunction]
#[pyo3(name = "all_invariants")]
fn py_all_invariants(
    rref: Vec<BinVec>,
    n: u32,
    weights: Vec<u32>,
    gmin_weights: Vec<u32>,
) -> (Vec<u128>, bool, Vec<u32>, Vec<u64>) {
    let r = experimental::invariants::compute_all_invariants(&rref, n, &weights, &gmin_weights);
    let digests = vec![
        r.wl_min_vanilla,
        r.wl_min_degree,
        r.wl_full_vanilla,
        r.wl_full_degree,
        r.t11_full,
        r.t11_gmin,
        r.t12,
        r.t13,
        r.t13_min,
    ];
    let metadata = vec![
        r.rounds_min_vanilla,
        r.rounds_min_degree,
        r.rounds_full_vanilla,
        r.rounds_full_degree,
        r.l_min,
        r.l_full,
    ];
    (digests, r.fallback, metadata, r.component_nanos.to_vec())
}

// ───────────────────────────────────────────── parallel_profiling entry
//
// Phase 3 (2026-05-20) audit entry. Mirrors `enumerate_doubly_even_parallel`
// but accumulates per-worker / per-seed timing. Only registered when the
// `parallel_profiling` Cargo feature is on.

/// Phase 3 parallel profiling entry. Same return shape as
/// `enumerate_doubly_even` plus a profile payload:
///
///   `(workers: list[(worker_id, active_ns, idle_ns, seed_count)],
///     seeds:   list[(worker_id, seed_id, ns, nodes, emitted)],
///     frontier_depth, total_wall_ns)`
///
/// Only available when the kernel is built with
/// `--features parallel_profiling`. `num_threads` must be >= 2 to
/// exercise the worker pool.
#[cfg(feature = "parallel_profiling")]
#[pyfunction]
#[pyo3(
    name = "enumerate_doubly_even_with_profile",
    signature = (n, max_k, quota, factorial_n, num_threads)
)]
fn py_enumerate_doubly_even_with_profile(
    py: Python<'_>,
    n: u32,
    max_k: u32,
    quota: Vec<u128>,
    factorial_n: u128,
    num_threads: u32,
) -> PyResult<(
    Vec<(Vec<BinVec>, Vec<u32>, Vec<Vec<u32>>, String, Vec<u32>)>,
    Vec<u128>,
    Vec<Vec<u64>>,
    (
        Vec<(u32, u64, u64, u32)>,
        Vec<(u32, u32, u64, u64, u32)>,
        u32,
        u64,
    ),
)> {
    if n > types::MAX_N {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n = {n} exceeds MAX_N = {}; the u64 kernel supports N up to 64",
            types::MAX_N,
        )));
    }
    let (out, stats, per_k, profile) = py.allow_threads(|| {
        crate::enumerate::enumerate_doubly_even_parallel_with_profile(
            n,
            max_k,
            quota,
            factorial_n,
            num_threads as usize,
        )
    });
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
    let workers: Vec<(u32, u64, u64, u32)> = profile
        .workers
        .iter()
        .map(|w| (w.worker_id, w.active_ns, w.idle_ns, w.seed_count))
        .collect();
    let seeds: Vec<(u32, u32, u64, u64, u32)> = profile
        .seeds
        .iter()
        .map(|s| (s.worker_id, s.seed_id, s.ns, s.nodes, s.emitted))
        .collect();
    Ok((
        result,
        stats,
        per_k,
        (workers, seeds, profile.frontier_depth, profile.total_wall_ns),
    ))
}

// ───────────────────────────────────────────── nauty_hist drain
//
// E1/E2 measurement: drain the per-call sparsenauty histogram. Only
// registered when the kernel is built with `--features nauty_hist`.

/// Drain the per-call sparsenauty histogram. Each tuple is
/// `(elapsed_ns, numnodes, tctotal, maxlevel, numgenerators,
///   left_vertices, right_vertices, rank, qd_path)` — one record per
/// sparsenauty call since the last drain. `qd_path` is `1` if the call
/// went through `canon_info_qd_native` (low-weight-incidence path) and
/// `0` if it went through `canon_info_native` (full bipartite —
/// either Q_D dispatch skipped or `collect_low_weight_codewords`
/// bailed).
#[cfg(feature = "nauty_hist")]
#[pyfunction]
#[pyo3(name = "drain_nauty_hist")]
fn py_drain_nauty_hist() -> Vec<(u64, u64, u64, i32, i32, u32, u32, u32, u8)> {
    experimental::canon_hist::drain()
        .into_iter()
        .map(|r| {
            (
                r.elapsed_ns,
                r.numnodes,
                r.tctotal,
                r.maxlevel,
                r.numgenerators,
                r.left_vertices,
                r.right_vertices,
                r.rank,
                if r.qd_path { 1u8 } else { 0u8 },
            )
        })
        .collect()
}

// ───────────────────────────────────────────── registration
//
// Called from `lib.rs#[pymodule] doubly_even_kernel(m)` so that dormant
// wrappers and their feature-gated sub-set come up at module init time.

/// Register every dormant wrapper into the supplied module. lib.rs adds
/// the production wrappers itself and then calls this for the rest.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Top-level — Feulner main entry (callable as
    // `doubly_even_kernel.canon_info_feulner_native(...)`).
    m.add_function(wrap_pyfunction!(py_canon_info_feulner_native, m)?)?;

    // Top-level — invariants (collision experiment substrate).
    m.add_function(wrap_pyfunction!(py_wl_signature, m)?)?;
    m.add_function(wrap_pyfunction!(py_t11_signature, m)?)?;
    m.add_function(wrap_pyfunction!(py_t11_signature_gmin, m)?)?;
    m.add_function(wrap_pyfunction!(py_t12_signature, m)?)?;
    m.add_function(wrap_pyfunction!(py_t13_signature, m)?)?;
    m.add_function(wrap_pyfunction!(py_all_invariants, m)?)?;

    // Conditional top-level wrappers.
    #[cfg(feature = "nauty_hist")]
    m.add_function(wrap_pyfunction!(py_drain_nauty_hist, m)?)?;
    #[cfg(feature = "parallel_profiling")]
    m.add_function(wrap_pyfunction!(py_enumerate_doubly_even_with_profile, m)?)?;

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
    debug.add_function(wrap_pyfunction!(py_canon_info_feulner_counters_v2, &debug)?)?;
    m.add_submodule(&debug)?;

    Ok(())
}
