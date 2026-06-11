//! Smoke test: every dormant module is reachable under
//! `crate::experimental::…`. Compiles iff the quarantine layout is
//! intact — protects against an accidental revert of the Phase 1 move.

#[test]
fn experimental_modules_accessible() {
    // One public symbol per moved module. Function-item references are
    // resolved at compile time; the closures below would not type-check
    // if the paths broke.
    let _feulner: fn(&[u64], u32) -> doubly_even_core::experimental::feulner::FeulnerCanonInfo =
        doubly_even_core::experimental::feulner::canon_info_feulner;

    let _clb_construct =
        doubly_even_core::experimental::feulner_clb::LabelledBranching::new;

    let _paired_iso: fn(
        &[u64],
        &[u64],
        u32,
    ) -> Option<Vec<u32>> =
        doubly_even_core::experimental::paired_iso::paired_iso;

    let _wl_refine: fn(
        &doubly_even_core::experimental::invariants::Bipartite,
        doubly_even_core::experimental::invariants::InitMode,
    ) -> (Vec<u64>, u32) = doubly_even_core::experimental::invariants::wl_refine;

    let _ = (_feulner, _clb_construct, _paired_iso, _wl_refine);
}
