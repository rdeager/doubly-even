//! Autom-only canon (2026-06-12): differential test of the
//! `get_canon = false` native path against `get_canon = true`.
//!
//! The lever's exactness rests on one assumption: nauty's automorphism
//! search produces the same group-level outputs — generators, group
//! order, orbits — whether or not the canonical labelling is requested
//! (`getcanon` only adds the best-leaf bookkeeping; the automorphism
//! discovery walks the same tree). This file states that assumption as
//! a test over REAL production inputs: every class emitted by full
//! enumerations at N ∈ {8, 10, 12, 14} (exercising both the Q_D-graph
//! and full-bipartite builders across ranks), plus the zero code and
//! the canon.rs fixed bases.
//!
//! Generator-list equality holds per-input through N=14 (this file)
//! but is NOT guaranteed in general: the 2026-06-12 N=24 bench A/B
//! found `nauty_generators_sum` off by 1 (1 in 4e5 calls) — nauty can
//! emit a different generating set of the same group without the
//! best-leaf bookkeeping. That is decision-neutral (orbits and orbit
//! minima are generating-set independent); slot 25 is accordingly
//! excluded from the bench gate (docs/benchmarking.md §5). If THIS
//! test ever fails on the N<=14 corpus, the same reasoning applies —
//! downgrade the generator assert to group-level checks, don't ship-block.

use doubly_even_core::canon::canon_info_native;
use doubly_even_core::enumerate::{enumerate_doubly_even_with_opts, LabelMode};
use doubly_even_core::parent_rule::ParentRule;
use doubly_even_core::qd_graph::canon_info_qd_native;

const SIGMA_N8: [u128; 5] = [1, 71, 455, 345, 30];
const FACT_N8: u128 = 40_320;
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

/// All class representatives from a full (Full-labelling) enumeration —
/// real production canon inputs at every rank.
fn corpus(n: u32, sigma: &[u128], fact: u128) -> Vec<Vec<u64>> {
    let max_k = (sigma.len() - 1) as u32;
    let (out, _stats, _per_k) = enumerate_doubly_even_with_opts(
        n,
        max_k,
        sigma.to_vec(),
        fact,
        ParentRule::CosetSpectrum { max_rank: 13 },
        LabelMode::Full,
    );
    out.into_iter().map(|r| r.rref).collect()
}

fn assert_identical_group_outputs(rref: &[u64], n: u32) {
    // Full-bipartite engine.
    let full = canon_info_native(rref, n, true);
    let autom = canon_info_native(rref, n, false);
    assert!(
        full.canonical_column_order.is_some(),
        "get_canon=true must yield the label"
    );
    assert!(
        autom.canonical_column_order.is_none(),
        "get_canon=false must not yield a label"
    );
    assert_eq!(
        full.aut_generators, autom.aut_generators,
        "generator lists differ between getcanon modes (full bipartite, rref={rref:?})"
    );
    assert_eq!(full.column_orbits, autom.column_orbits);
    assert_eq!(full.grpsize1.to_bits(), autom.grpsize1.to_bits());
    assert_eq!(full.grpsize2, autom.grpsize2);
    assert_eq!(full.numgenerators, autom.numgenerators);

    // Q_D-graph engine (when its builder succeeds — bail parity must match).
    let qd_full = canon_info_qd_native(rref, n, true);
    let qd_autom = canon_info_qd_native(rref, n, false);
    assert_eq!(qd_full.is_some(), qd_autom.is_some(), "bail parity differs");
    if let (Some(f), Some(a)) = (qd_full, qd_autom) {
        assert!(f.canonical_column_order.is_some());
        assert!(a.canonical_column_order.is_none());
        assert_eq!(
            f.aut_generators, a.aut_generators,
            "generator lists differ between getcanon modes (Q_D graph, rref={rref:?})"
        );
        assert_eq!(f.column_orbits, a.column_orbits);
        assert_eq!(f.grpsize1.to_bits(), a.grpsize1.to_bits());
        assert_eq!(f.grpsize2, a.grpsize2);
        assert_eq!(f.numgenerators, a.numgenerators);
    }
}

#[test]
fn autom_only_matches_full_on_fixed_bases() {
    // Zero code, repetition code, extended Hamming [8,4,4].
    assert_identical_group_outputs(&[], 4);
    assert_identical_group_outputs(&[0b1111u64], 4);
    assert_identical_group_outputs(&[0xE1u64, 0xD2, 0xB4, 0x78], 8);
}

#[test]
fn autom_only_matches_full_on_enumeration_corpus() {
    for (n, sigma, fact) in [
        (8u32, &SIGMA_N8[..], FACT_N8),
        (10, &SIGMA_N10[..], FACT_N10),
        (12, &SIGMA_N12[..], FACT_N12),
        (14, &SIGMA_N14[..], FACT_N14),
    ] {
        for rref in corpus(n, sigma, fact) {
            assert_identical_group_outputs(&rref, n);
        }
    }
}
