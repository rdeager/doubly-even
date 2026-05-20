//! Dormant canonicaliser substrate.
//!
//! These modules are not on the kernel hot path. They are kept in tree as
//! reference oracles, diff substrate, and revivable audit material. See
//! `/workspace/src/EXPERIMENTAL.md` for the full index and the per-module
//! memory notes for the close-out reasoning:
//!
//! - `feulner`     — Feulner column-side canonicaliser (D9, ~1000 LOC).
//!   Dispatch closed 2026-05-20; see
//!   `memory/project_feulner_dispatch_closed.md`.
//! - `feulner_clb` — Jerrum CLB + Lemma 5.9 (Feulner §5.2 substrate).
//! - `paired_iso`  — Leon §10(i) verifier (D12). Dormant behind the
//!   `equivalence_verifier` Cargo feature; see
//!   `memory/project_verifier_dormant.md`.
//! - `invariants`  — WL + T11/T12/T13 collision-experiment substrate.
//!
//! Production builds compile these unconditionally (the pyfunctions that
//! expose them are useful for ad-hoc audit scripts), but no hot-path
//! module imports from them — the `experimental` namespace is a one-way
//! barrier the reader can ignore when tracing the active flow.

#[cfg(feature = "nauty_hist")]
pub mod canon_hist;
pub mod feulner;
pub mod feulner_clb;
pub mod invariants;
pub mod paired_iso;
pub mod py_exports;
#[cfg(feature = "equivalence_verifier")]
pub mod verifier_dispatch;
