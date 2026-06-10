//! D15: the coset-spectrum parent rule and the legacy σ-rule must
//! enumerate the SAME equivalence classes.
//!
//! What changes under the rule is *which* (parent-class, coset-orbit)
//! pair emits each class, so the emitted RREF representatives may differ
//! at rank ≥ 2. What may NOT change:
//!
//!   - per-rank class counts (DFGHILM Table 3 cells),
//!   - the per-rank sorted multiset of |Aut| values (hence the mass
//!     Σ N!/|Aut|, which the kernel additionally panics on internally
//!     against the Gaborit quota — excess aborts the run).
//!
//! Audit mode must be byte-identical to legacy (it only tallies).
//!
//! Also pinned: per-rank rule MIXING (φ below a rank cap, legacy above)
//! is sound — rank is iso-invariant and McKay's induction is local to
//! one child rank.

use std::collections::BTreeMap;

use doubly_even_kernel::enumerate::{enumerate_doubly_even_with_rule, EnumeratedRaw};
use doubly_even_kernel::parent_rule::ParentRule;

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

/// Per-rank profile: rank → sorted |Aut| multiset. Equal profiles ⇔
/// equal class counts per rank AND equal mass per rank.
fn class_profile(out: &[EnumeratedRaw]) -> BTreeMap<usize, Vec<u128>> {
    let mut profile: BTreeMap<usize, Vec<u128>> = BTreeMap::new();
    for raw in out {
        profile.entry(raw.rref.len()).or_default().push(raw.aut_order);
    }
    for orders in profile.values_mut() {
        orders.sort_unstable();
    }
    profile
}

/// Sorted canonical RREF sets — only for the audit-vs-legacy byte
/// identity (representatives must match there, unlike across rules).
fn rref_set(out: &[EnumeratedRaw]) -> Vec<Vec<u64>> {
    let mut rows: Vec<Vec<u64>> = out.iter().map(|r| r.rref.clone()).collect();
    rows.sort();
    rows
}

fn run(
    n: u32,
    sigma: &[u128],
    fact: u128,
    rule: ParentRule,
) -> Vec<EnumeratedRaw> {
    let max_k = (sigma.len() - 1) as u32;
    let (out, _stats, _per_k) =
        enumerate_doubly_even_with_rule(n, max_k, sigma.to_vec(), fact, rule);
    out
}

fn check_equivalence(n: u32, sigma: &[u128], fact: u128) {
    let legacy = run(n, sigma, fact, ParentRule::Legacy);
    let phi = run(n, sigma, fact, ParentRule::CosetSpectrum { max_rank: 13 });
    assert_eq!(
        class_profile(&legacy),
        class_profile(&phi),
        "N={n}: per-rank class/|Aut| profile diverged between legacy and \
         coset-spectrum rules"
    );
}

#[test]
fn coset_spectrum_matches_legacy_n10() {
    check_equivalence(10, &SIGMA_N10, FACT_N10);
}

#[test]
fn coset_spectrum_matches_legacy_n12() {
    check_equivalence(12, &SIGMA_N12, FACT_N12);
}

#[test]
fn coset_spectrum_matches_legacy_n14() {
    check_equivalence(14, &SIGMA_N14, FACT_N14);
}

#[test]
fn coset_spectrum_matches_legacy_n16() {
    check_equivalence(16, &SIGMA_N16, FACT_N16);
}

#[test]
fn audit_mode_is_byte_identical_to_legacy() {
    let legacy = run(14, &SIGMA_N14, FACT_N14, ParentRule::Legacy);
    let audit = run(14, &SIGMA_N14, FACT_N14, ParentRule::Audit);
    assert_eq!(
        rref_set(&legacy),
        rref_set(&audit),
        "audit mode changed the emitted representative set"
    );
}

/// Per-rank rule mixing: φ for child rank ≤ cap, legacy above. Sweep the
/// cap through the whole rank range at N=14 — every mix must produce the
/// same class profile.
#[test]
fn rank_cap_mixing_is_sound() {
    let reference = class_profile(&run(14, &SIGMA_N14, FACT_N14, ParentRule::Legacy));
    for cap in 0..=7u32 {
        let mixed = run(
            14,
            &SIGMA_N14,
            FACT_N14,
            ParentRule::CosetSpectrum { max_rank: cap },
        );
        assert_eq!(
            reference,
            class_profile(&mixed),
            "rank cap {cap}: mixed-rule enumeration diverged"
        );
    }
}

/// The φ rule under the parallel driver must agree with the sequential
/// φ rule (seeder and workers share one rule).
#[cfg(feature = "parallel")]
#[test]
fn coset_spectrum_parallel_matches_sequential() {
    use doubly_even_kernel::enumerate::enumerate_doubly_even_parallel_with_rule;
    let rule = ParentRule::CosetSpectrum { max_rank: 13 };
    let seq = run(16, &SIGMA_N16, FACT_N16, rule);
    let (par, _, _) = enumerate_doubly_even_parallel_with_rule(
        16,
        8,
        SIGMA_N16.to_vec(),
        FACT_N16,
        8,
        rule,
    );
    // Same rule ⇒ same parent choices ⇒ identical representative sets,
    // not merely identical profiles.
    assert_eq!(rref_set(&seq), rref_set(&par));
    assert_eq!(class_profile(&seq), class_profile(&par));
}
