//! Native hot kernel for the doubly-even enumerator.
//!
//! Scaffolding only. Real entry points (σ_Q table build, singular-set BFS,
//! orbit-min decomposition; eventually `doubly_even_candidates_q`) land in
//! follow-up commits. For now this exposes two functions:
//!
//! - `kernel_version()` — version string from `Cargo.toml`, smoke-tests that
//!   the FFI bridge is wired up.
//! - `popcount_batch(words)` — `Vec<u64> -> Vec<u32>`, smoke-tests bulk
//!   data movement across the FFI (the shape every real entry point will
//!   need).

use pyo3::prelude::*;

/// Version string read from `Cargo.toml` at build time.
#[pyfunction]
fn kernel_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Per-element popcount of a `u64` vector.
#[pyfunction]
fn popcount_batch(words: Vec<u64>) -> Vec<u32> {
    words.iter().map(|w| w.count_ones()).collect()
}

#[pymodule]
fn doubly_even_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kernel_version, m)?)?;
    m.add_function(wrap_pyfunction!(popcount_batch, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popcount_matches_count_ones() {
        let xs = vec![0u64, 1, 2, 3, 0xFFFF_FFFF_FFFF_FFFF, 0x5555_5555_5555_5555];
        let got = popcount_batch(xs.clone());
        let want: Vec<u32> = xs.iter().map(|w| w.count_ones()).collect();
        assert_eq!(got, want);
    }
}
