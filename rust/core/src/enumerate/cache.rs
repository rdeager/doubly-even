//! Two-tier canon-info cache: primary RREF-keyed LRU + secondary
//! weight-enumerator buckets (equivalence transfer via the McKay BFS,
//! or the paired-iso verifier when that feature is on). Bodies are
//! verbatim from the original `enumerate.rs`.

use std::num::NonZeroUsize;
use std::rc::Rc;

use crate::canon::{canon_info_native, NativeCanonInfo, QD_GRAPH_THRESHOLD};
#[cfg(feature = "dense_qd")]
use crate::experimental::canon_dense_qd::canon_info_qd_dense;
#[cfg(feature = "traces_qd")]
use crate::experimental::canon_traces_qd::canon_info_qd_traces;
#[cfg(feature = "equivalence_verifier")]
use crate::experimental::paired_iso::PairedIsoCachedCf;
use crate::linalg::{apply_permutation, row_reduce};
use crate::permutations::{aut_order_exact, compute_column_orbits, perm_compose, perm_inverse};
#[cfg(not(any(feature = "dense_qd", feature = "traces_qd")))]
use crate::qd_graph::canon_info_qd_native;
use crate::types::BinVec;

use super::worker::WorkerState;

/// Per-worker primary canon cache capacity. Read once at `WorkerState::new`
/// from `DOUBLY_EVEN_CANON_CACHE_CAP` (parsed as a positive integer); else
/// defaults to 500,000 entries, which is ~500 MB at N=26 and keeps a
/// 20-worker run well under a 50 GB cgroup ceiling. Hit rate measured at
/// 3–5 % across N=18–24, so eviction barely affects wall time.
pub(crate) fn canon_cache_capacity() -> NonZeroUsize {
    const DEFAULT_CAP: usize = 500_000;
    let cap = std::env::var("DOUBLY_EVEN_CANON_CACHE_CAP")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_CAP);
    NonZeroUsize::new(cap).expect("canon cache capacity must be non-zero")
}

/// Internal cache row for canon info; mirrors `EnumeratedRaw` minus the
/// `rref` (which is the cache key). Stored behind `Rc` so cache hits and
/// recursion-state hand-off don't clone the heavy `Vec<Vec<u32>>` aut
/// generators field.
/// Secondary-cache bucket value. The `cached_cf` field is only populated
/// when the `equivalence_verifier` feature is on; otherwise it's `None`
/// and the entry behaves as an instrumentation-only `(canonical_form, info)`
/// pair.
pub(crate) struct BucketEntry {
    pub(crate) canonical: Vec<BinVec>,
    pub(crate) info: Rc<CachedInfo>,
    #[cfg(feature = "equivalence_verifier")]
    pub(crate) cached_cf: Rc<PairedIsoCachedCf>,
}

#[derive(Clone)]
pub(crate) struct CachedInfo {
    /// `None` = autom-only entry (computed with `getcanon = FALSE` under
    /// [`LabelMode::AutomOnly`]). INVARIANT: every consumer of the label
    /// obtains its `Rc` through `canon_info(rref, /*need_label=*/ true)`,
    /// which upgrades-and-replaces the cache entry; stale autom-only `Rc`s
    /// held by recursion frames only ever read gens / aut_order / orbits.
    pub(crate) canonical_column_order: Option<Vec<u32>>,
    pub(crate) aut_generators: Vec<Vec<u32>>,
    pub(crate) aut_order: u128,
    pub(crate) column_orbits: Vec<u32>,
}

/// Canonical-labelling mode for the recursion's canon calls (the
/// autom-only lever, 2026-06-12). Resolved once per driver invocation —
/// same pattern as [`crate::parent_rule::ParentRule`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LabelMode {
    /// Default: `getcanon = FALSE` on calls whose canonical labelling no
    /// decision consumes (φ accept-unique accepts + the root). Skips the
    /// measured 19–25 % canonical pass of the sparsenauty call on ~80 %
    /// of canon calls. Emitted records carry an empty
    /// `canonical_column_order` (the streaming path never carried one).
    AutomOnly,
    /// Kill-switch: always compute the canonical labelling — byte-identical
    /// to pre-lever behaviour, including the emitted per-class records.
    Full,
}

impl LabelMode {
    /// `DOUBLY_EVEN_CANON_LABELLING` ∈ {unset | "autom-only" (default),
    /// "full"}. Unknown values panic loudly rather than silently changing
    /// behaviour (the `DOUBLY_EVEN_PARENT_RULE` precedent).
    pub fn from_env() -> Self {
        match std::env::var("DOUBLY_EVEN_CANON_LABELLING") {
            Ok(raw) => match raw.trim() {
                "" | "autom-only" => LabelMode::AutomOnly,
                "full" => LabelMode::Full,
                other => panic!(
                    "unrecognised DOUBLY_EVEN_CANON_LABELLING={other:?} \
                     (expected autom-only | full)"
                ),
            },
            Err(_) => LabelMode::AutomOnly,
        }
    }
}

/// Sorted weight enumerator of the code with the given RREF basis.
pub(crate) fn weight_enum(rref: &[BinVec]) -> Vec<u32> {
    let k = rref.len();
    if k == 0 {
        return vec![0];
    }
    let size = 1usize << k;
    let mut weights: Vec<u32> = Vec::with_capacity(size);
    // Gray-code walk so each step is one XOR.
    let mut w: BinVec = 0;
    weights.push(0);
    for mask in 1..size {
        let flip = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        w ^= rref[flip];
        weights.push(w.count_ones());
    }
    weights.sort_unstable();
    weights
}

impl WorkerState {
    /// Compute the canonical-form RREF of `rref` given its canonical column
    /// order: apply σ to each row, then RREF. Two permutation-equivalent
    /// subspaces produce the same canonical form, so this acts as an
    /// equivalence-class identifier.
    fn canonical_form(&self, rref: &[BinVec], canonical_column_order: &[u32]) -> Vec<BinVec> {
        let permuted: Vec<BinVec> = rref
            .iter()
            .map(|&b| apply_permutation(b, canonical_column_order))
            .collect();
        let (rr, _) = row_reduce(&permuted, self.n);
        rr
    }

    /// Compute canon info for the code given by `rref`, or recover from cache.
    ///
    /// `need_label`: whether the caller will consume
    /// `canonical_column_order`. With [`LabelMode::AutomOnly`] (default) a
    /// `false` here runs nauty with `getcanon = FALSE` — the autom-only
    /// lever. A cache hit on an autom-only entry by a label-needing caller
    /// recomputes with the label and **replaces** the entry (never mutates:
    /// in-flight `Rc` holders keep a valid autom-only snapshot they only
    /// read gens / aut_order / orbits from). Stats contract: an upgrade
    /// counts as a primary hit (bit-identical `canon_calls` /
    /// `primary_hits` vs `LabelMode::Full`), tallied separately in
    /// `stats_canon_label_upgrades`; the nauty tree counters
    /// (numnodes/tctotal/maxlevel/generators) are NOT re-accumulated on
    /// upgrades, so they keep meaning "per true miss".
    pub(crate) fn canon_info(&mut self, rref: &[BinVec], need_label: bool) -> Rc<CachedInfo> {
        // The secondary cache computes `canonical_form` on every true miss,
        // so maintaining it forces the label on every call (the feature /
        // instrumentation path predates the lever and keeps its semantics).
        let want_label = need_label
            || self.label_mode == LabelMode::Full
            || self.maintain_secondary_cache;

        // Primary cache.
        if let Some(info) = self.canon_cache.get(rref) {
            self.stats_primary_hits += 1;
            if !want_label || info.canonical_column_order.is_some() {
                return Rc::clone(info);
            }
            // Label upgrade: recompute full, replace the entry.
            self.stats_canon_label_upgrades += 1;
            let nauty_t0 = std::time::Instant::now();
            let native = self.canon_dispatch(rref, true);
            let aut_order = aut_order_exact(
                native.grpsize1,
                native.grpsize2,
                &native.aut_generators,
                self.n,
                self.factorial_n,
            );
            let nauty_delta = nauty_t0.elapsed().as_nanos();
            self.stats_nauty_ns += nauty_delta;
            self.stats_nauty_ns_by_k[rref.len()] += nauty_delta as u64;
            let old = self
                .canon_cache
                .get(rref)
                .expect("entry present moments ago");
            // The group-level outputs are properties of Aut(C) and must not
            // depend on getcanon; the generator *list* may differ (nauty can
            // discover a different generating set of the same group), which
            // is decision-neutral — orbits and orbit minima are
            // generating-set-independent.
            assert_eq!(
                old.aut_order, aut_order,
                "autom-only and full canon disagree on |Aut|"
            );
            debug_assert_eq!(
                old.column_orbits, native.column_orbits,
                "autom-only and full canon disagree on column orbits"
            );
            let upgraded = Rc::new(CachedInfo {
                canonical_column_order: native.canonical_column_order,
                aut_generators: native.aut_generators,
                aut_order,
                column_orbits: native.column_orbits,
            });
            self.canon_cache.put(rref.to_vec(), Rc::clone(&upgraded));
            return upgraded;
        }

        // True miss. Decide whether to maintain the secondary cache for
        // either the verifier dispatch or the env-gated instrumentation.
        self.stats_canon_calls += 1;
        if !want_label {
            self.stats_canon_autom_only_calls += 1;
        }

        let we_key = if self.maintain_secondary_cache {
            Some(weight_enum(rref))
        } else {
            None
        };
        let bucket_size_at_attempt: usize = if let Some(key) = we_key.as_ref() {
            let size = self
                .secondary_cache
                .get(key)
                .map(|v| v.len())
                .unwrap_or(0);
            if size > 0 {
                self.stats_secondary_attempts += 1;
                self.stats_bucket_size_sum_at_attempt += size as u64;
                if (size as u64) > self.stats_max_bucket_size {
                    self.stats_max_bucket_size = size as u64;
                }
            }
            size
        } else {
            0
        };
        let bucket_was_nonempty = bucket_size_at_attempt > 0;

        // Feature-gated paired-iso verifier dispatch — see
        // `crate::experimental::verifier_dispatch`. The helper walks the
        // bucket, runs the equitable-partition prefilter, and on a hit
        // returns a freshly reconstructed CachedInfo for `rref` (bypasses
        // nauty entirely). The caller just updates stats and short-circuits.
        #[cfg(feature = "equivalence_verifier")]
        if bucket_was_nonempty {
            self.stats_verifier_attempts += 1;
            let bucket = self
                .secondary_cache
                .get(we_key.as_ref().unwrap())
                .unwrap();
            let outcome =
                crate::experimental::verifier_dispatch::try_dispatch(rref, self.n, bucket);
            self.stats_verifier_compares += outcome.compares;
            self.stats_verifier_ns += outcome.elapsed_ns;
            if let Some(new_info) = outcome.hit {
                #[cfg(debug_assertions)]
                {
                    let recon_cf = self.canonical_form(
                        rref,
                        new_info
                            .canonical_column_order
                            .as_ref()
                            .expect("verifier reconstructions always carry the label"),
                    );
                    let expected_cf =
                        outcome.hit_cf.as_ref().expect("hit_cf set when hit Some");
                    assert_eq!(
                        &recon_cf, expected_cf,
                        "verifier reconstruction produced wrong canonical form"
                    );
                }
                self.canon_cache.put(rref.to_vec(), Rc::clone(&new_info));
                self.stats_verifier_hits += 1;
                return new_info;
            }
        }

        // Time nauty (including `aut_order_exact`, which can fall back to
        // Schreier–Sims for groups past float64 precision). This bounds
        // the cost any cheap equivalence verifier must beat. Always-on:
        // one `Instant::now()` pair per call ≈ 40 ns, negligible vs
        // nauty's tens-of-µs work.
        let nauty_t0 = std::time::Instant::now();
        let native: NativeCanonInfo = self.canon_dispatch(rref, want_label);
        let aut_order = aut_order_exact(
            native.grpsize1,
            native.grpsize2,
            &native.aut_generators,
            self.n,
            self.factorial_n,
        );
        let nauty_delta = nauty_t0.elapsed().as_nanos();
        self.stats_nauty_ns += nauty_delta;
        self.stats_nauty_ns_by_k[rref.len()] += nauty_delta as u64;
        self.stats_nauty_numnodes += native.numnodes;
        self.stats_nauty_tctotal += native.tctotal;
        self.stats_nauty_maxlevel_sum += native.maxlevel.max(0) as u64;
        self.stats_nauty_generators_sum += native.numgenerators.max(0) as u64;

        let info = Rc::new(CachedInfo {
            canonical_column_order: native.canonical_column_order,
            aut_generators: native.aut_generators,
            aut_order,
            column_orbits: native.column_orbits,
        });
        self.canon_cache.put(rref.to_vec(), Rc::clone(&info));

        if let Some(key) = we_key {
            // Compute canonical form for secondary-cache membership. The
            // label is always present here: `maintain_secondary_cache`
            // forces `want_label` above.
            let canonical = self.canonical_form(
                rref,
                info.canonical_column_order
                    .as_ref()
                    .expect("secondary cache forces want_label"),
            );
            let bucket = self.secondary_cache.entry(key).or_default();
            if let Some(pos) = bucket
                .iter()
                .position(|e| e.canonical == canonical)
            {
                self.stats_secondary_hits += 1;
                self.stats_match_position_sum += pos as u64;
                let _ = bucket_was_nonempty;
            } else {
                // The bucket value carries `CachedInfo` *for the canonical
                // form itself*, not for `rref`. Conjugate `info`'s data into
                // `canonical`'s column frame by the cached σ:
                //
                //   σ_canonical = identity (canonical is already in RREF).
                //   gens_canonical = σ · g · σ⁻¹ for each g ∈ Aut(rref).
                //
                // This makes the verifier reconstruction
                // `σ_d = compose(σ_canonical, π) = π` correct — π is exactly
                // the witness D → canonical from `paired_iso`.
                let sigma: &[u32] = info
                    .canonical_column_order
                    .as_ref()
                    .expect("secondary cache forces want_label");
                let sigma_inv = perm_inverse(sigma);
                let gens_canonical: Vec<Vec<u32>> = info
                    .aut_generators
                    .iter()
                    .map(|g| perm_compose(sigma, &perm_compose(g, &sigma_inv)))
                    .collect();
                let orbits_canonical = compute_column_orbits(&gens_canonical, self.n);
                let info_canonical = Rc::new(CachedInfo {
                    canonical_column_order: Some((0..self.n).collect()),
                    aut_generators: gens_canonical,
                    aut_order: info.aut_order,
                    column_orbits: orbits_canonical,
                });
                #[cfg(feature = "equivalence_verifier")]
                {
                    let cached_cf = Rc::new(PairedIsoCachedCf::new(&canonical, self.n));
                    bucket.push(BucketEntry {
                        canonical,
                        info: info_canonical,
                        cached_cf,
                    });
                }
                #[cfg(not(feature = "equivalence_verifier"))]
                {
                    bucket.push(BucketEntry {
                        canonical,
                        info: info_canonical,
                    });
                }
            }
        }

        info
    }

    /// The raw nauty / Q_D-graph engine dispatch — no caching, no stats.
    /// When 2^k is large the low-weight-incidence graph (Q_D) beats the
    /// full bipartite by a factor that grows with 2^k / |C_low|. For small
    /// k the full graph is already cheap, so we skip the span check. See
    /// canon.rs's `QD_GRAPH_THRESHOLD`. `get_canon` forwards to nauty's
    /// `getcanon` option (the autom-only lever; experimental engines
    /// always compute the label).
    fn canon_dispatch(&self, rref: &[BinVec], get_canon: bool) -> NativeCanonInfo {
        if (1u32 << rref.len()) > QD_GRAPH_THRESHOLD {
            // Q6 audit Phase 2/3: feature-gated alternative engines for
            // the Q_D-graph canon. `traces_qd` wins over `dense_qd` if both
            // are enabled (Traces is a different algorithm; dense vs sparse
            // is just a representation tweak).
            #[cfg(feature = "traces_qd")]
            {
                canon_info_qd_traces(rref, self.n)
                    .unwrap_or_else(|| canon_info_native(rref, self.n, get_canon))
            }
            #[cfg(all(feature = "dense_qd", not(feature = "traces_qd")))]
            {
                canon_info_qd_dense(rref, self.n)
                    .unwrap_or_else(|| canon_info_native(rref, self.n, get_canon))
            }
            #[cfg(not(any(feature = "dense_qd", feature = "traces_qd")))]
            {
                canon_info_qd_native(rref, self.n, get_canon)
                    .unwrap_or_else(|| canon_info_native(rref, self.n, get_canon))
            }
        } else {
            canon_info_native(rref, self.n, get_canon)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parent_rule::ParentRule;

    fn fresh_worker(label_mode: LabelMode) -> WorkerState {
        let quota = vec![crate::u256::U256::MAX; 5];
        let mut w = WorkerState::new(
            8,
            4,
            quota,
            40_320,
            ParentRule::CosetSpectrum { max_rank: 13 },
            label_mode,
        );
        // Unit-test isolation: ignore any ambient env instrumentation.
        w.maintain_secondary_cache = false;
        w
    }

    /// The extended Hamming [8,4,4] basis from canon.rs's tests.
    const E8: [BinVec; 4] = [0xE1, 0xD2, 0xB4, 0x78];

    /// Upgrade path: an autom-only cache entry hit by a label-needing
    /// caller is recomputed full and REPLACED; the stats contract keeps
    /// `canon_calls` / `primary_hits` bit-identical to Full-mode history
    /// and tallies the recompute in `canon_label_upgrades` only.
    #[test]
    fn label_upgrade_replaces_entry_and_counts_once() {
        let mut w = fresh_worker(LabelMode::AutomOnly);

        let first = w.canon_info(&E8, false);
        assert!(first.canonical_column_order.is_none());
        assert_eq!(w.stats_canon_calls, 1);
        assert_eq!(w.stats_primary_hits, 0);
        assert_eq!(w.stats_canon_autom_only_calls, 1);
        assert_eq!(w.stats_canon_label_upgrades, 0);

        let upgraded = w.canon_info(&E8, true);
        let sigma = upgraded
            .canonical_column_order
            .as_ref()
            .expect("upgrade must produce the label");
        assert_eq!(w.stats_canon_calls, 1, "upgrade is not a new canon call");
        assert_eq!(w.stats_primary_hits, 1, "upgrade counts as a primary hit");
        assert_eq!(w.stats_canon_label_upgrades, 1);
        assert_eq!(upgraded.aut_order, first.aut_order);
        assert_eq!(upgraded.column_orbits, first.column_orbits);

        // The upgraded label matches a from-scratch Full compute.
        let mut w_full = fresh_worker(LabelMode::Full);
        let oracle = w_full.canon_info(&E8, false); // Full mode forces the label
        assert_eq!(
            Some(sigma),
            oracle.canonical_column_order.as_ref().into(),
            "upgraded label differs from a fresh full compute"
        );

        // Subsequent label-needing hits are plain hits — no re-upgrade.
        let again = w.canon_info(&E8, true);
        assert!(again.canonical_column_order.is_some());
        assert_eq!(w.stats_canon_calls, 1);
        assert_eq!(w.stats_primary_hits, 2);
        assert_eq!(w.stats_canon_label_upgrades, 1);
    }

    /// Full mode (the kill-switch) never produces label-less entries,
    /// regardless of what the caller requests.
    #[test]
    fn full_mode_forces_label_on_need_label_false() {
        let mut w = fresh_worker(LabelMode::Full);
        let info = w.canon_info(&E8, false);
        assert!(info.canonical_column_order.is_some());
        assert_eq!(w.stats_canon_autom_only_calls, 0);
    }

    #[test]
    #[should_panic(expected = "unrecognised DOUBLY_EVEN_CANON_LABELLING")]
    fn label_mode_rejects_unknown_values() {
        // Direct parse-path check without mutating process env (the cargo
        // harness is threaded): replicate from_env's match on a bad value.
        match "bogus" {
            "" | "autom-only" => LabelMode::AutomOnly,
            "full" => LabelMode::Full,
            other => panic!(
                "unrecognised DOUBLY_EVEN_CANON_LABELLING={other:?} \
                 (expected autom-only | full)"
            ),
        };
    }
}
