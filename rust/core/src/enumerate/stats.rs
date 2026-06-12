//! Stats-vector layout (the single source of truth), `finalize`, and the
//! cross-worker merge helpers. The layout constants are exported to
//! Python via the wrapper crate's `kernel_stats_layout()` and consumed
//! by `scripts/bench.py`; the unit tests below pin vector lengths to
//! the constants so the mirror can never silently drift.
//! `finalize` / merge bodies are verbatim from the original
//! `enumerate.rs`.

use super::worker::{EnumeratedRaw, WorkerState};

/// Field names of the kernel stats vector, in the exact order
/// `WorkerState::finalize` emits them. **This is the single source of
/// truth** for the layout: the wrapper crate exports it to Python as
/// `kernel_stats_layout()` and `scripts/bench.py` consumes it from the
/// installed kernel (keeping only a frozen fallback for old wheels).
/// Evolution is append-only — downstream JSONs are compared across
/// sessions, so never reorder or repurpose a slot (slot 48 is the one
/// historical exception, an unwired reserve repurposed before any JSON
/// recorded it). The unit tests below pin `finalize` to these lengths.
pub const KERNEL_STATS_LAYOUT: [&str; 51] = [
    "canon_calls",                // 0
    "primary_hits",               // 1
    "secondary_attempts",         // 2
    "secondary_hits",             // 3
    "is_canon_aug_calls",         // 4
    "parent_eq_hits",             // 5
    "weight_enum_filtered",       // 6
    "bfs_calls",                  // 7
    "bfs_hits",                   // 8
    "is_canon_aug_ns",            // 9
    "bfs_ns",                     // 10
    "nauty_ns",                   // 11
    "bucket_size_sum_at_attempt", // 12
    "match_position_sum",         // 13
    "max_bucket_size",            // 14 — merges as max, not sum
    "verifier_attempts",          // 15
    "verifier_hits",              // 16
    "verifier_compares",          // 17
    "verifier_ns",                // 18
    "candidates_q_calls",         // 19
    "candidates_q_ns",            // 20
    "bfs_rejects",                // 21
    "nauty_numnodes_sum",         // 22 — nauty statsblk: backtrack tree size
    "nauty_tctotal_sum",          // 23 — nauty statsblk: target-cell work
    "nauty_maxlevel_sum",         // 24 — nauty statsblk: deepest level
    "nauty_generators_sum",       // 25 — nauty statsblk: generators found
    "phi_reject",                 // 26 — parent rule: rejects (no canon call)
    "phi_accept_unique",          // 27 — parent rule: unique-min accepts
    "phi_tie_accept",             // 28 — parent rule: ties resolved accept
    "phi_tie_reject",             // 29 — parent rule: ties resolved reject
    "phi_ns",                     // 30 — ns inside the phi cascade
    "phi_strata_sum",             // 31 — sum of strata evaluated
    "phi_m_size_sum",             // 32 — sum of |M| at decision
    "nauty_ns_kept",              // 33 — audit: kappa numerator
    "cq_qbasis_ns",               // 34 — sigma_Q sub-phase (phase_timers)
    "cq_autimage_ns",             // 35
    "cq_singular_ns",             // 36
    "cq_orbitmin_ns",             // 37
    "cq_lift_sort_ns",            // 38
    "phi_vhalf_ns",               // 39 — sampled phi sub-phase (phase_timers)
    "phi_members_ns",             // 40
    "phi_first_stratum_ns",       // 41
    "phi_wht_ns",                 // 42
    "phi_direct_ns",              // 43
    "phi_sampled_calls",          // 44 — phi sampling weights (1-in-64)
    "phi_ctx_ns",                 // 45 — per-parent ctx builds (subset of phi_ns)
    "phi_ctx_builds",             // 46 — number of phi-tested parents
    "phi_s1_fastpath",            // 47 — first-stratum O(1) decisions
    "phi_chain_fastpath",         // 48 — E-chain O(1) decisions at stratum >= 2
    "canon_autom_only_calls",     // 49 — true misses run with getcanon=FALSE
    "canon_label_upgrades",       // 50 — autom-only entries recomputed full on a label-needing hit
];

/// Row names of the per-rank stats matrix, in the exact order
/// `WorkerState::finalize` emits them. Rows 0-13 are counters bucketed
/// by PARENT rank; rows 14-15 and 17-18 too; row 16 (`nauty_ns`) is
/// bucketed by the rank of the code being canonised (the child, so
/// parent_k + 1 — off-by-one by design, one canon result is shared by
/// every candidate probing the same D).
pub const PER_K_STATS_ROWS: [&str; 19] = [
    "is_canon_aug_calls",    // 0
    "parent_eq_hits",        // 1
    "weight_enum_filtered",  // 2
    "bfs_calls",             // 3
    "bfs_hits",              // 4
    "bfs_rejects",           // 5
    "mass_stop_pre_loop",    // 6
    "mass_stop_in_loop",     // 7
    "candidates_total_seen", // 8
    "candidates_skipped",    // 9
    "phi_reject",            // 10 — audit mode only
    "phi_accept_unique",     // 11 — audit mode only
    "phi_tie_accept",        // 12 — audit mode only
    "phi_tie_reject",        // 13 — audit mode only
    "phi_ns",                // 14 — per-rank phi-cascade ns (always-on)
    "candidates_q_ns",       // 15 — per-rank sigma_Q ns (always-on)
    "nauty_ns",              // 16 — per-rank canon ns (CHILD rank)
    "phi_sampled_calls",     // 17 — per-rank sampling weights
    "phi_ctx_ns",            // 18 — per-rank ctx-build ns (subset of row 14)
];

impl WorkerState {
    /// Consume this WorkerState and produce the `(output, stats, per_k_stats)`
    /// tuple used by both the sequential and parallel drivers. Stats layout
    /// is documented on [`enumerate_doubly_even`].
    ///
    /// When `output_writer` is set (streaming mode), `output` is empty and
    /// the writer is explicitly flushed here so any disk-write failure
    /// surfaces as a panic rather than being swallowed by `BufWriter::Drop`.
    pub(crate) fn finalize(mut self) -> (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>) {
        if let Some(mut w) = self.output_writer.take() {
            w.flush().expect("BinaryWriter final flush failed");
        }
        let stats: Vec<u128> = vec![
            self.stats_canon_calls as u128,
            self.stats_primary_hits as u128,
            self.stats_secondary_attempts as u128,
            self.stats_secondary_hits as u128,
            self.stats_is_canon_aug_calls as u128,
            self.stats_parent_eq_hits as u128,
            self.stats_weight_enum_filtered as u128,
            self.stats_bfs_calls as u128,
            self.stats_bfs_hits as u128,
            self.stats_is_canon_aug_ns,
            self.stats_bfs_ns,
            self.stats_nauty_ns,
            self.stats_bucket_size_sum_at_attempt as u128,
            self.stats_match_position_sum as u128,
            self.stats_max_bucket_size as u128,
            self.stats_verifier_attempts as u128,
            self.stats_verifier_hits as u128,
            self.stats_verifier_compares as u128,
            self.stats_verifier_ns,
            self.stats_candidates_q_calls as u128,
            self.stats_candidates_q_ns,
            self.stats_bfs_rejects as u128,
            self.stats_nauty_numnodes as u128,
            self.stats_nauty_tctotal as u128,
            self.stats_nauty_maxlevel_sum as u128,
            self.stats_nauty_generators_sum as u128,
            self.stats_phi_reject as u128,
            self.stats_phi_accept_unique as u128,
            self.stats_phi_tie_accept as u128,
            self.stats_phi_tie_reject as u128,
            self.stats_phi_ns,
            self.stats_phi_strata_sum as u128,
            self.stats_phi_m_size_sum as u128,
            self.stats_nauty_ns_kept,
            self.stats_cq_qbasis_ns,
            self.stats_cq_autimage_ns,
            self.stats_cq_singular_ns,
            self.stats_cq_orbitmin_ns,
            self.stats_cq_lift_sort_ns,
            self.stats_phi_frame_gray_ns,
            self.stats_phi_sort_ns,
            self.stats_phi_first_stratum_ns,
            self.stats_phi_wht_ns,
            self.stats_phi_direct_ns,
            self.stats_phi_sampled_calls as u128,
            self.stats_phi_ctx_ns,
            self.stats_phi_ctx_builds as u128,
            self.stats_phi_s1_fastpath as u128,
            self.stats_phi_chain_fastpath as u128,
            self.stats_canon_autom_only_calls as u128,
            self.stats_canon_label_upgrades as u128,
        ];
        let per_k_stats: Vec<Vec<u64>> = vec![
            self.stats_is_canon_aug_calls_by_k,
            self.stats_parent_eq_hits_by_k,
            self.stats_weight_enum_filtered_by_k,
            self.stats_bfs_calls_by_k,
            self.stats_bfs_hits_by_k,
            self.stats_bfs_rejects_by_k,
            self.stats_mass_stop_pre_loop_by_k,
            self.stats_mass_stop_in_loop_by_k,
            self.stats_candidates_total_seen_by_k,
            self.stats_candidates_skipped_by_k,
            self.stats_phi_reject_by_k,
            self.stats_phi_accept_unique_by_k,
            self.stats_phi_tie_accept_by_k,
            self.stats_phi_tie_reject_by_k,
            self.stats_phi_ns_by_k,
            self.stats_candidates_q_ns_by_k,
            self.stats_nauty_ns_by_k,
            self.stats_phi_sampled_calls_by_k,
            self.stats_phi_ctx_ns_by_k,
        ];
        debug_assert_eq!(
            stats.len(),
            KERNEL_STATS_LAYOUT.len(),
            "finalize stats vector out of sync with KERNEL_STATS_LAYOUT"
        );
        debug_assert_eq!(
            per_k_stats.len(),
            PER_K_STATS_ROWS.len(),
            "finalize per_k matrix out of sync with PER_K_STATS_ROWS"
        );
        (self.output, stats, per_k_stats)
    }
}

/// In-place merge of stats vectors (no `Vec<EnumeratedRaw>` to splice).
/// Used by the streaming parallel driver.
#[cfg(feature = "parallel")]
pub(crate) fn merge_stats_only(
    a_stats: &mut Vec<u128>,
    a_per_k: &mut Vec<Vec<u64>>,
    b_stats: Vec<u128>,
    b_per_k: Vec<Vec<u64>>,
) {
    debug_assert_eq!(a_stats.len(), b_stats.len(), "stats vector length mismatch");
    for (i, (x, y)) in a_stats.iter_mut().zip(b_stats.iter()).enumerate() {
        if i == STATS_MAX_BUCKET_SIZE_IDX {
            *x = (*x).max(*y);
        } else {
            *x = x.checked_add(*y).expect("stats merge overflow");
        }
    }
    debug_assert_eq!(a_per_k.len(), b_per_k.len(), "per_k rows count mismatch");
    for (row_a, row_b) in a_per_k.iter_mut().zip(b_per_k.iter()) {
        debug_assert_eq!(row_a.len(), row_b.len(), "per_k row length mismatch");
        for (xa, yb) in row_a.iter_mut().zip(row_b.iter()) {
            *xa = xa.checked_add(*yb).expect("per_k merge overflow");
        }
    }
}

/// Index of `stats_max_bucket_size` inside the flat stats vector — `.max()`
/// not `+=` on merge. See doc on [`enumerate_doubly_even`] for full layout.
/// (Un-gated so the layout test can pin it on every build.)
const STATS_MAX_BUCKET_SIZE_IDX: usize = 14;

#[cfg(feature = "parallel")]
pub(crate) fn merge_finalized(
    a: &mut (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>),
    b: (Vec<EnumeratedRaw>, Vec<u128>, Vec<Vec<u64>>),
) {
    a.0.extend(b.0);
    debug_assert_eq!(a.1.len(), b.1.len(), "stats vector length mismatch");
    for (i, (x, y)) in a.1.iter_mut().zip(b.1.iter()).enumerate() {
        if i == STATS_MAX_BUCKET_SIZE_IDX {
            *x = (*x).max(*y);
        } else {
            *x = x
                .checked_add(*y)
                .expect("stats vector merge overflow");
        }
    }
    debug_assert_eq!(a.2.len(), b.2.len(), "per_k rows count mismatch");
    for (row_a, row_b) in a.2.iter_mut().zip(b.2.iter()) {
        debug_assert_eq!(row_a.len(), row_b.len(), "per_k row length mismatch");
        for (xa, yb) in row_a.iter_mut().zip(row_b.iter()) {
            *xa = xa
                .checked_add(*yb)
                .expect("per_k merge overflow");
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    /// The layout constants and the vectors `finalize` actually emits
    /// must agree — this is what makes the constants a trustworthy
    /// single source for Python (`kernel_stats_layout()` / bench.py).
    #[test]
    fn finalize_lengths_match_layout_consts() {
        let quota = vec![u128::MAX; 4];
        let mut w = WorkerState::new(
            10,
            3,
            quota,
            3_628_800,
            crate::parent_rule::ParentRule::from_env(),
            super::super::cache::LabelMode::AutomOnly,
        );
        let (rref, pivots) = crate::linalg::row_reduce(&[], 10);
        let info = w.canon_info(&rref, false);
        w.traverse(rref, pivots, info);
        let (_, stats, per_k) = w.finalize();
        assert_eq!(stats.len(), KERNEL_STATS_LAYOUT.len());
        assert_eq!(per_k.len(), PER_K_STATS_ROWS.len());
    }

    /// Slot 14 merges as max (not sum) in the cross-worker merge fns;
    /// guard the special-cased index against any future reordering.
    #[test]
    fn max_bucket_size_index_is_pinned() {
        assert_eq!(KERNEL_STATS_LAYOUT[STATS_MAX_BUCKET_SIZE_IDX], "max_bucket_size");
    }
}
