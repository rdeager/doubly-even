//! Algorithm core of the doubly-even enumerator.
//!
//! Pure Rust — no pyo3, no global allocator. The Python surface (and the
//! mimalloc `#[global_allocator]`, which must stay out of test binaries
//! and microbench bins) lives in the workspace-root wrapper crate
//! `doubly-even-kernel`; the microbench bins under `scripts/microbench/`
//! depend on this crate directly so their "production arms" are
//! production code.
//!
//! Production hot path:
//!
//! - [`candidates::doubly_even_candidates_q`] — the fat per-parent call
//!   that runs the Q_basis → σ_Q tables → singular BFS → orbit-min →
//!   lift pipeline.
//! - [`canon::canon_info_native`] / [`qd_graph::canon_info_qd_native`] —
//!   native sparsenauty canonicalisers (Q_D-graph default, full G(C)
//!   fallback).
//! - [`subspace_orbit::subspace_in_orbit`] — the McKay parent test
//!   inner-loop BFS.
//! - [`parent_rule`] — the coset-spectrum parent rule (φ cascade,
//!   split-frame sharing, pair-max fast paths, E-chain); the formal
//!   statements live in `docs/theory.md`.
//! - [`enumerate`] — the full canonical-augmentation recursion;
//!   parallel worker-pool drivers under `--features parallel`.
//!
//! Dormant / audit substrate is quarantined under [`experimental`] —
//! a one-way barrier no hot-path module imports from.

pub mod candidates;
pub mod canon;
#[cfg(feature = "phase_timers")]
pub mod cycles;
pub mod enumerate;
pub mod experimental;
pub mod linalg;
pub mod orbit;
pub mod parent_rule;
pub mod permutations;
pub mod qd_graph;
pub mod quotient;
#[cfg(feature = "parallel")]
pub mod seeder_pool;
pub mod streaming;
pub mod subspace_orbit;
pub mod types;
pub mod u256;
