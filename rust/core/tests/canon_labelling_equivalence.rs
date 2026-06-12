//! Autom-only canon (2026-06-12): `LabelMode::AutomOnly` (default) vs
//! `LabelMode::Full` (kill-switch) must be DECISION-IDENTICAL — not just
//! the same classes, but the same DFS emission *sequence*, the same
//! per-rank counts and mass, and bit-equal decision counters. The only
//! permitted stats differences are the nauty tree-shape sums
//! (numnodes / tctotal / maxlevel — their drop IS the lever) and the
//! two new slots (49 autom-only calls, 50 label upgrades), and
//! `nauty_generators_sum` — equal at N<=16 here, but measured ±1 at
//! N=24 (nauty may emit a different generating set of the same group
//! without best-leaf bookkeeping; decision-neutral, see
//! canon_labelling.rs and docs/benchmarking.md §5).
//!
//! Modeled on parent_rule_equivalence.rs.

use doubly_even_core::enumerate::{
    enumerate_doubly_even_with_opts, EnumeratedRaw, LabelMode, KERNEL_STATS_LAYOUT,
};
use doubly_even_core::parent_rule::ParentRule;

const SIGMA_N10: [u128; 6] = [1, 255, 5355, 11475, 2295, 0];
const FACT_N10: u128 = 3_628_800;
const SIGMA_N12: [u128; 7] = [1, 991, 79035, 625515, 479655, 25245, 0];
const FACT_N12: u128 = 479_001_600;
const SIGMA_N14: [u128; 8] = [
    1,
    4095,
    1_396_395,
    50_868_675,
    213_648_435,
    103_378_275,
    4_922_775,
    0,
];
const FACT_N14: u128 = 87_178_291_200;
const SIGMA_N16: [u128; 9] = [
    1,
    16_511,
    22_891_115,
    3_451_225_635,
    62_449_776_675,
    143_919_296_235,
    44_388_662_175,
    1_885_422_825,
    9_845_550,
];
const FACT_N16: u128 = 20_922_789_888_000;

fn run(
    n: u32,
    sigma: &[u128],
    fact: u128,
    rule: ParentRule,
    labelling: LabelMode,
) -> (Vec<EnumeratedRaw>, Vec<u128>) {
    let max_k = (sigma.len() - 1) as u32;
    let (out, stats, _per_k) =
        enumerate_doubly_even_with_opts(n, max_k, sigma.to_vec(), fact, rule, labelling);
    (out, stats)
}

fn slot(name: &str) -> usize {
    KERNEL_STATS_LAYOUT
        .iter()
        .position(|&s| s == name)
        .unwrap_or_else(|| panic!("unknown stats slot {name}"))
}

/// Stats slots allowed to differ between the two labelling modes:
/// `_ns` timers (nondeterministic), the nauty tree-shape sums the
/// lever shrinks, and the two new mode counters.
fn slot_may_differ(name: &str) -> bool {
    name.ends_with("_ns")
        || matches!(
            name,
            "nauty_numnodes_sum"
                | "nauty_tctotal_sum"
                | "nauty_maxlevel_sum"
                | "nauty_generators_sum"
                | "canon_autom_only_calls"
                | "canon_label_upgrades"
        )
}

fn assert_mode_equivalence(n: u32, sigma: &[u128], fact: u128, rule: ParentRule) {
    let (out_a, stats_a) = run(n, sigma, fact, rule, LabelMode::AutomOnly);
    let (out_f, stats_f) = run(n, sigma, fact, rule, LabelMode::Full);

    // Sequential DFS is deterministic: exact emission-sequence equality
    // of (rref, aut_order) — stronger than any per-rank profile.
    assert_eq!(out_a.len(), out_f.len(), "class count differs (N={n})");
    for (a, f) in out_a.iter().zip(out_f.iter()) {
        assert_eq!(a.rref, f.rref, "emission sequence diverged (N={n})");
        assert_eq!(a.aut_order, f.aut_order, "aut_order differs (N={n})");
        assert_eq!(a.aut_generators, f.aut_generators);
        assert_eq!(a.column_orbits, f.column_orbits);
        // Output contract: AutomOnly emits the empty label, Full the real one.
        assert!(a.canonical_column_order.is_empty());
        assert_eq!(f.canonical_column_order.len(), n as usize);
    }

    // Decision counters bit-equal except the documented exclusions.
    for (i, name) in KERNEL_STATS_LAYOUT.iter().enumerate() {
        if slot_may_differ(name) {
            continue;
        }
        assert_eq!(
            stats_a[i], stats_f[i],
            "stats slot {i} ({name}) differs between labelling modes (N={n})"
        );
    }

    // The lever actually fired: under the φ rule, autom-only calls are
    // the accept-unique calls plus the root; Full mode forces the label
    // on every call.
    let autom_calls = stats_a[slot("canon_autom_only_calls")];
    let accept_unique = stats_a[slot("phi_accept_unique")];
    match rule {
        ParentRule::CosetSpectrum { .. } => {
            assert_eq!(
                autom_calls,
                accept_unique + 1,
                "autom-only calls != accept_unique + root (N={n})"
            );
        }
        // Legacy / Audit consume the label on every candidate call —
        // only the root can run autom-only.
        _ => assert_eq!(autom_calls, 1, "legacy/audit should autom-only the root only"),
    }
    assert_eq!(stats_f[slot("canon_autom_only_calls")], 0);
    assert_eq!(stats_f[slot("canon_label_upgrades")], 0);
}

#[test]
fn autom_only_equals_full_n10() {
    assert_mode_equivalence(
        10,
        &SIGMA_N10,
        FACT_N10,
        ParentRule::CosetSpectrum { max_rank: 13 },
    );
}

#[test]
fn autom_only_equals_full_n12() {
    assert_mode_equivalence(
        12,
        &SIGMA_N12,
        FACT_N12,
        ParentRule::CosetSpectrum { max_rank: 13 },
    );
}

#[test]
fn autom_only_equals_full_n14() {
    assert_mode_equivalence(
        14,
        &SIGMA_N14,
        FACT_N14,
        ParentRule::CosetSpectrum { max_rank: 13 },
    );
}

#[test]
fn autom_only_equals_full_n16() {
    assert_mode_equivalence(
        16,
        &SIGMA_N16,
        FACT_N16,
        ParentRule::CosetSpectrum { max_rank: 13 },
    );
}

/// Pins the call-site table: under the legacy rule every candidate call
/// passes `need_label = true`, so only the root runs autom-only.
#[test]
fn legacy_rule_autom_onlys_root_only() {
    assert_mode_equivalence(12, &SIGMA_N12, FACT_N12, ParentRule::Legacy);
}

/// Rank-cap mixing: children above the φ cap take the legacy path
/// (label forced) — the boundary must stay decision-identical too.
#[test]
fn rank_cap_mixing_stays_identical() {
    let (out_a, _) = run(
        14,
        &SIGMA_N14,
        FACT_N14,
        ParentRule::CosetSpectrum { max_rank: 3 },
        LabelMode::AutomOnly,
    );
    let (out_f, _) = run(
        14,
        &SIGMA_N14,
        FACT_N14,
        ParentRule::CosetSpectrum { max_rank: 3 },
        LabelMode::Full,
    );
    assert_eq!(out_a.len(), out_f.len());
    for (a, f) in out_a.iter().zip(out_f.iter()) {
        assert_eq!(a.rref, f.rref);
        assert_eq!(a.aut_order, f.aut_order);
    }
}
