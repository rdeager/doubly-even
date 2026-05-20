//! crate::experimental::canon_hist — E1/E2 sparsenauty per-call histogram.
//!
//! Dormant audit substrate behind Cargo feature `nauty_hist` (default
//! OFF). Records one [`NautyCallRecord`] per sparsenauty call in a
//! process-wide Mutex; Python drains the buffer via the
//! `drain_nauty_hist` pyfunction registered from `experimental::py_exports`.
//!
//! The records expose tree-shape counters from nauty's `statsblk`
//! (`numnodes`, `tctotal`, `maxlevel`, `numgenerators`) alongside the
//! input graph shape (left/right vertex counts, code rank, Q_D-path bit).
//! See the `nauty_hist` audit memory entries (`project_wl_canonical_closed.md`,
//! `project_q6_audit_closed.md`) for the analyses this powered.

#[derive(Clone, Copy)]
pub struct NautyCallRecord {
    pub elapsed_ns: u64,
    pub numnodes: u64,
    pub tctotal: u64,
    pub maxlevel: i32,
    pub numgenerators: i32,
    pub left_vertices: u32,
    pub right_vertices: u32,
    /// Code rank `k = rref.len()`. With this and `left_vertices` we can
    /// reconstruct the Q_D-vs-fallback bucketing: `qd_path` iff
    /// `left_vertices < (1 << rank)` (the low-weight subset is strictly
    /// smaller than the full codeword set) or `qd_path` is `true` and
    /// `left_vertices == (1 << rank) - 1` only in the degenerate
    /// rank-0 → fallback case.
    pub rank: u32,
    /// `true` if this record came from `canon_info_qd_native` (Q_D
    /// low-weight path); `false` if from `canon_info_native` — either
    /// because Q_D bailed (`collect_low_weight_codewords` returned
    /// `None`) or because the dispatch in `enumerate::canon_info`
    /// chose the full-bipartite path directly.
    pub qd_path: bool,
}

static NAUTY_HIST: std::sync::Mutex<Vec<NautyCallRecord>> =
    std::sync::Mutex::new(Vec::new());

#[inline]
pub fn push(rec: NautyCallRecord) {
    if let Ok(mut h) = NAUTY_HIST.lock() {
        h.push(rec);
    }
}

pub fn drain() -> Vec<NautyCallRecord> {
    let mut h = NAUTY_HIST.lock().expect("nauty hist mutex poisoned");
    std::mem::take(&mut *h)
}
