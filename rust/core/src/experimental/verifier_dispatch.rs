//! crate::experimental::verifier_dispatch — paired-iso fast-path helper.
//!
//! Dormant audit substrate behind Cargo feature `equivalence_verifier`.
//! Walks a secondary-cache bucket and, on a Leon §10(i) equitable-partition
//! hit, reconstructs `CachedInfo` for the new code via the witnessing
//! column permutation π — bypassing nauty entirely. Closed direction; see
//! `memory/project_verifier_dormant.md` (cost explodes 43→68→363 µs
//! N=18→20→22; net negative even at 100 % hit rate).
//!
//! The function returns an `AttemptOutcome`; the caller (WorkerState in
//! `crate::enumerate::canon_info`) updates its own `stats_verifier_*`
//! counters and short-circuits if the dispatch hit.

use std::rc::Rc;
use std::time::Instant;

use crate::enumerate::{BucketEntry, CachedInfo};
use crate::experimental::paired_iso::{
    paired_iso_equitable, reconstruct_aut_generators, reconstruct_canonical_column_order,
    reconstruct_column_orbits, EquitableResult,
};
use crate::types::BinVec;

pub(crate) struct AttemptOutcome {
    /// Number of bucket entries scanned (≤ bucket.len(); short-circuits on hit).
    pub(crate) compares: u64,
    /// Total wall-time spent inside the helper (prefilter + reconstruction).
    pub(crate) elapsed_ns: u128,
    /// On hit: freshly reconstructed `CachedInfo` for `rref` — ready for
    /// the caller's primary canon cache. `None` on miss.
    pub(crate) hit: Option<Rc<CachedInfo>>,
    /// On hit: canonical-form bytes of the matched bucket entry — used by
    /// the debug-only reconstruction assert in the caller. Unused in
    /// release builds; the field exists so the assert costs nothing extra
    /// to evaluate when on.
    #[allow(dead_code)]
    pub(crate) hit_cf: Option<Vec<BinVec>>,
}

/// Walk `bucket`, looking for an entry whose cached canonical-form
/// `PairedIsoCachedCf` matches `rref` under the equitable-partition test.
/// On the first hit, reconstruct `CachedInfo` for `rref` via the witnessing
/// permutation π. `Inconclusive` results are treated as misses — the caller
/// falls through to nauty.
pub(crate) fn try_dispatch(rref: &[BinVec], n: u32, bucket: &[BucketEntry]) -> AttemptOutcome {
    let t0 = Instant::now();
    let mut compares: u64 = 0;
    let mut found: Option<(Vec<BinVec>, Rc<CachedInfo>, Vec<u32>)> = None;
    for entry in bucket {
        compares += 1;
        match paired_iso_equitable(rref, &entry.cached_cf, n) {
            EquitableResult::Iso(pi) => {
                found = Some((entry.canonical.clone(), Rc::clone(&entry.info), pi));
                break;
            }
            EquitableResult::NotIso | EquitableResult::Inconclusive => continue,
        }
    }
    let (hit, hit_cf) = if let Some((cf, cf_info, pi)) = found {
        let sigma_d = reconstruct_canonical_column_order(&cf_info.canonical_column_order, &pi);
        let gens_d = reconstruct_aut_generators(&cf_info.aut_generators, &pi);
        let orbits_d = reconstruct_column_orbits(&gens_d, n);
        let new_info = Rc::new(CachedInfo {
            canonical_column_order: sigma_d,
            aut_generators: gens_d,
            aut_order: cf_info.aut_order,
            column_orbits: orbits_d,
        });
        (Some(new_info), Some(cf))
    } else {
        (None, None)
    };
    AttemptOutcome {
        compares,
        elapsed_ns: t0.elapsed().as_nanos(),
        hit,
        hit_cf,
    }
}
