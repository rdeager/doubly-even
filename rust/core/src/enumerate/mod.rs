//! Native canonical-augmentation enumerator for doubly-even codes.
//!
//! Port of `doubly_even.enumerate.augment._traverse` + helpers. Holds the
//! recursion state, canon-info cache, and mass accumulators inside one
//! struct so the hot loop has no Python ↔ Rust boundary crossings beyond
//! the single entry call.
//!
//! Two caches live here:
//!
//! - **Primary RREF cache**: `HashMap<Vec<BinVec>, CachedInfo>` keyed by
//!   subspace identifier — same shape as the Python LRU.
//! - **Secondary permutation-equivalence cache**: `HashMap<weight_enum,
//!   Vec<(rref, CachedInfo)>>`. On primary miss, scan the bucket; verify
//!   equivalence via [`subspace_orbit::subspace_in_orbit`] and transfer
//!   the cached info through the witnessing permutation.
//!
//! Split across four submodules (pure moves — bodies are verbatim from
//! the original single file):
//!
//! - [`worker`]  — `WorkerState` (recursion state) + the traversal,
//!   parent-test and candidate-test methods.
//! - [`cache`]   — the two-tier canon-info cache (primary RREF LRU +
//!   secondary weight-enum buckets).
//! - [`stats`]   — the stats-vector layout (single source of truth:
//!   [`KERNEL_STATS_LAYOUT`] / [`PER_K_STATS_ROWS`]), `finalize`, and
//!   the merge helpers.
//! - [`drivers`] — sequential / parallel / streaming entry points and
//!   the in-Rust mass gate.

mod cache;
mod decomp;
mod drivers;
mod stats;
mod worker;

#[cfg(all(feature = "parallel", feature = "traces_qd"))]
compile_error!(
    "feature `parallel` is incompatible with `traces_qd`: Traces uses \
     non-TLS static work queues and is not thread-safe even with HAVE_TLS=1"
);

pub use cache::LabelMode;
pub use drivers::{
    enumerate_doubly_even, enumerate_doubly_even_counts, enumerate_doubly_even_streaming,
    enumerate_doubly_even_with_opts, enumerate_doubly_even_with_rule, CountsResult,
    StreamingResult,
};
#[cfg(feature = "parallel")]
pub use drivers::{
    enumerate_doubly_even_parallel, enumerate_doubly_even_parallel_counts,
    enumerate_doubly_even_parallel_streaming, enumerate_doubly_even_parallel_with_rule,
    enumerate_doubly_even_parallel_with_seeder, ProgressSink,
};
pub use stats::{KERNEL_STATS_LAYOUT, PER_K_STATS_ROWS};
pub use worker::EnumeratedRaw;

#[cfg(feature = "equivalence_verifier")]
pub(crate) use cache::{BucketEntry, CachedInfo};
#[cfg(feature = "parallel_profiling")]
pub(crate) use drivers::{GlobalMassTracker, LoadBalancer, SeedFrontier, SelfSubdivideCfg};
#[cfg(feature = "parallel_profiling")]
pub(crate) use stats::merge_finalized;
#[cfg(feature = "parallel_profiling")]
pub(crate) use worker::WorkerState;

/// Build identifier — returns `"verifier"` when compiled with the
/// `equivalence_verifier` feature, otherwise `"baseline"`. Used by the
/// Python A/B harness to assert which kernel binary is loaded.
pub fn build_info() -> &'static str {
    if cfg!(feature = "equivalence_verifier") {
        "verifier"
    } else {
        "baseline"
    }
}
