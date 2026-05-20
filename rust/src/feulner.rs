//! **EXPERIMENTAL / reference oracle.** Feulner-style column-side
//! canonicaliser, binary + permutation-only.
//!
//! Active dispatch uses `canon::canon_info_qd_native` (sparsenauty on the
//! Q_D bipartite graph). This module is kept in tree as the diff oracle
//! for the Python staging and as the substrate for `paired_iso`. See
//! `/workspace/src/EXPERIMENTAL.md` and the memory bullet
//! `project_feulner_dispatch_closed.md` for why the dispatch was closed.
//!
//! Direct port of `doubly_even.canon.experimental.feulner` — the Python
//! staging is the
//! correctness oracle for this module. See that file for the algorithmic
//! write-up; behaviour here mirrors it line-for-line.
//!
//! Replaces the bipartite-graph `nauty` call in `canon::canon_info_native`.
//! Where nauty searches a `2^k + N`-vertex graph, this backtracks on an
//! `N`-leaf coordinate-individualisation tree with cheap partition-refinement
//! work per node. For `[N=22, k≤11]` doubly-even codes this is much
//! better-sized than the graph view.
//!
//! Aut-order is returned as a decimal string so values exceeding `u64` (e.g.
//! `22! ≈ 1.1e21` for the zero code) round-trip exactly through PyO3 to
//! Python's arbitrary-precision int. Internally we accumulate in `u128`,
//! which fits factorials up to `34!`.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::feulner_clb::LabelledBranching;
use crate::permutations::{compute_column_orbits, perm_compose, perm_inverse};
use crate::types::BinVec;

/// Output of [`canon_info_feulner`], mirroring `canon.feulner.CanonInfo`.
pub struct FeulnerCanonInfo {
    pub canonical_column_order: Vec<u32>,
    pub aut_generators: Vec<Vec<u32>>,
    /// `|Aut(C)|` as a decimal string. Empty for the zero-dim case is `"1"`.
    pub aut_order_decimal: String,
    pub column_orbits: Vec<u32>,
    pub leaves: u64,
    pub prunes: u64,
    /// Number of CLB topological-sort prunes (Feulner §5.2, Lemma 5.9).
    pub clb_prunes: u64,
}

// ----------------------------------------------------- permutation utilities

pub(crate) type Perm = Vec<u32>;

pub(crate) fn perm_identity(n: u32) -> Perm {
    (0..n).collect()
}

// ------------------------------------- Schreier–Sims for permutation groups

fn orbit_and_transversal(
    gens: &[Perm],
    base: u32,
    n: u32,
) -> (Vec<u32>, HashMap<u32, Perm>) {
    let mut orbit = vec![base];
    let mut transversal: HashMap<u32, Perm> = HashMap::new();
    transversal.insert(base, perm_identity(n));
    let mut frontier = vec![base];
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &p in &frontier {
            for g in gens {
                let q = g[p as usize];
                if transversal.contains_key(&q) {
                    continue;
                }
                let new_perm = perm_compose(g, &transversal[&p]);
                transversal.insert(q, new_perm);
                orbit.push(q);
                next.push(q);
            }
        }
        frontier = next;
    }
    (orbit, transversal)
}

/// Exact order of `⟨gens⟩` on `n` points via textbook Schreier–Sims.
///
/// Mirrors `doubly_even.canon.permutations.group_order`. No sifting; for the
/// `N ≤ 32` sizes we use this is fine. Returns the product of orbit sizes
/// down the natural base `0, 1, …, n-1`.
pub fn group_order(gens: &[Perm], n: u32) -> u128 {
    if gens.is_empty() {
        return 1;
    }
    let mut working: Vec<Perm> = gens.to_vec();
    let mut order: u128 = 1;
    let id = perm_identity(n);
    for base in 0..n {
        let (orbit, transversal) = orbit_and_transversal(&working, base, n);
        order = order
            .checked_mul(orbit.len() as u128)
            .expect("aut_order overflows u128");
        if orbit.len() == 1 {
            continue;
        }
        let mut next_gens: std::collections::BTreeSet<Perm> =
            std::collections::BTreeSet::new();
        for &p in &orbit {
            let t_p = &transversal[&p];
            for g in &working {
                let q = g[p as usize];
                let t_q_inv = perm_inverse(&transversal[&q]);
                let inner = perm_compose(g, t_p);
                let sg = perm_compose(&t_q_inv, &inner);
                if sg != id {
                    next_gens.insert(sg);
                }
            }
        }
        if next_gens.is_empty() {
            break;
        }
        working = next_gens.into_iter().collect();
    }
    order
}

// ----------------------------------------------------- column-orbit helpers

/// Orbit-rep map under `⟨gens⟩` restricted to `subset`. Assumes each `g`
/// maps `subset` to itself (the caller filters gens by "fixes path", which
/// preserves the cell, so all images land in the cell).
fn orbits_on_subset(gens: &[&Perm], subset: &[u32]) -> HashMap<u32, u32> {
    let mut parent: HashMap<u32, u32> = subset.iter().map(|&c| (c, c)).collect();
    let in_subset: HashSet<u32> = subset.iter().copied().collect();

    fn find(parent: &mut HashMap<u32, u32>, c: u32) -> u32 {
        let mut cur = c;
        loop {
            let p = parent[&cur];
            if p == cur {
                return cur;
            }
            let gp = parent[&p];
            parent.insert(cur, gp);
            cur = gp;
        }
    }

    fn union(parent: &mut HashMap<u32, u32>, a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return;
        }
        if ra < rb {
            parent.insert(rb, ra);
        } else {
            parent.insert(ra, rb);
        }
    }

    for g in gens {
        for &c in subset {
            let j = g[c as usize];
            if j != c && in_subset.contains(&j) {
                union(&mut parent, c, j);
            }
        }
    }
    subset.iter().map(|&c| (c, find(&mut parent, c))).collect()
}

// ------------------------------------------------- initial column partition

/// Aut-invariant 2-cell start: split columns by "lies in `support(C)`".
///
/// A column `j` is covered by the code iff some RREF row has bit `j` set
/// (equivalent to: some codeword has bit `j` set, since the RREF spans
/// the support). `Aut(C)` permutes covered and uncovered columns
/// separately, so this split is free and strictly finer than `{0..n-1}`
/// whenever `C` has any uncovered column.
pub(crate) fn initial_partition(rref: &[BinVec], n: u32) -> Vec<Vec<u32>> {
    let mut support: BinVec = 0;
    for &row in rref {
        support |= row;
    }
    let nonzero: Vec<u32> =
        (0..n).filter(|&j| (support >> j) & 1 == 1).collect();
    let zero: Vec<u32> =
        (0..n).filter(|&j| (support >> j) & 1 == 0).collect();
    let mut cells: Vec<Vec<u32>> = Vec::new();
    if !nonzero.is_empty() {
        cells.push(nonzero);
    }
    if !zero.is_empty() {
        cells.push(zero);
    }
    if cells.is_empty() {
        cells.push((0..n).collect());
    }
    cells
}

// ------------------------------------- refiner enumeration + partition refine

/// Enumerate `Aut(C)`-invariant refiner codewords.
///
/// Phase B: returns the **two lowest non-zero weight strata** of the
/// code (Bouyukliev–Bouyuklieva 2019 §3.3 anchor). The Phase-A choice
/// of weight-4 only left discriminative power on the table whenever
/// weight-8 (or the next stratum up) also distinguished orbits. The
/// extra refiner-incidence work in `refine` is paid back many-fold by
/// the smaller search tree at large `N`.
pub(crate) fn invariant_refiners(rref: &[BinVec]) -> Vec<BinVec> {
    let k = rref.len();
    let mut by_weight: BTreeMap<u32, Vec<BinVec>> = BTreeMap::new();
    if k == 0 {
        return Vec::new();
    }
    let total = 1u64 << k;
    for mask in 1u64..total {
        let mut w: BinVec = 0;
        let mut m = mask;
        let mut i = 0usize;
        while m != 0 {
            if m & 1 == 1 {
                w ^= rref[i];
            }
            m >>= 1;
            i += 1;
        }
        let wt = w.count_ones();
        by_weight.entry(wt).or_default().push(w);
    }
    let mut out: Vec<BinVec> = Vec::new();
    for (_, words) in by_weight.into_iter().take(2) {
        out.extend(words);
    }
    out
}

fn mask_of_cell(cell: &[u32]) -> u64 {
    let mut m: u64 = 0;
    for &j in cell {
        m |= 1u64 << j;
    }
    m
}

/// Equitable refinement of column partition `p` by refiner-incidence
/// signatures, via a nauty-style worklist of refiner groups.
///
/// Mirrors `_refine_incremental` in `doubly_even.canon.feulner` — see
/// that docstring for the algorithm. State: refiner groups indexed by
/// stable slot; `p_table[g][j]` = count of refiners in group `g` with
/// bit `j` set; `worklist` of pending groups that may still cause cell
/// splits.
///
/// Output cell-order: lex by `(input_lineage, signature, min_col)` so
/// each input cell's split products form a contiguous block — this
/// preserves the McKay convention the search expects (individualised
/// singleton inherits the parent's slot index in the output).
pub(crate) fn refine(p: Vec<Vec<u32>>, refiners: &[BinVec]) -> Vec<Vec<u32>> {
    if p.is_empty() {
        return p;
    }
    // n_cols_max is "one past the highest column index" — column
    // indices live in [0, n_cols_max), used to size the per-column
    // p_table rows.
    let mut n_cols_max: usize = 1;
    for cell in &p {
        if let Some(&max_j) = cell.iter().max() {
            n_cols_max = n_cols_max.max(max_j as usize + 1);
        }
    }

    let mut cells: Vec<Vec<u32>> = p;
    let mut cell_mask: Vec<u64> =
        cells.iter().map(|c| mask_of_cell(c)).collect();
    let mut cell_lineage: Vec<u32> = (0..cells.len() as u32).collect();

    if refiners.is_empty() {
        return emit_sorted(cells, &Vec::new(), &cell_lineage);
    }

    // Initial groups by cell-histogram tuple.
    let mut init_buckets: BTreeMap<Vec<u32>, Vec<u32>> = BTreeMap::new();
    for (ri, &w) in refiners.iter().enumerate() {
        let t: Vec<u32> =
            cell_mask.iter().map(|m| (w & m).count_ones()).collect();
        init_buckets.entry(t).or_default().push(ri as u32);
    }
    let mut groups: Vec<Vec<u32>> =
        init_buckets.into_iter().map(|(_, v)| v).collect();

    // p_table[g][j] = #{r in groups[g] : (refiners[r] >> j) & 1}
    let mut p_table: Vec<Vec<u32>> = groups
        .iter()
        .map(|g| recompute_p(g, refiners, n_cols_max))
        .collect();

    let mut worklist: Vec<u32> = (0..groups.len() as u32).collect();
    let mut in_worklist: Vec<bool> = vec![true; groups.len()];

    while let Some(g) = worklist.pop() {
        in_worklist[g as usize] = false;

        // Snapshot length; sub-cells appended during this iteration
        // are processed in later worklist iterations (their own groups
        // are pushed onto the worklist when they split).
        let n_snapshot = cells.len();
        let mut ci = 0usize;
        while ci < n_snapshot {
            if cells[ci].len() > 1 {
                let mut buckets: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
                for &j in &cells[ci] {
                    buckets
                        .entry(p_table[g as usize][j as usize])
                        .or_default()
                        .push(j);
                }
                if buckets.len() > 1 {
                    apply_cell_split(
                        ci,
                        buckets,
                        &mut cells,
                        &mut cell_mask,
                        &mut cell_lineage,
                        &mut groups,
                        &mut p_table,
                        &mut worklist,
                        &mut in_worklist,
                        refiners,
                        n_cols_max,
                    );
                }
            }
            ci += 1;
        }
    }

    emit_sorted(cells, &p_table, &cell_lineage)
}

/// Replace `cells[ci]` with its bucketed sub-cells (largest fragment
/// stays in slot `ci`; others appended). Then propagate the cell split
/// to every refiner group whose refiners had any bit in the old cell
/// mask: re-bucket their refiners by per-sub-cell histogram, split
/// groups that distinguish sub-cells, and queue new groups onto the
/// worklist via the Hopcroft smallest-fragment rule.
#[allow(clippy::too_many_arguments)]
fn apply_cell_split(
    ci: usize,
    buckets: BTreeMap<u32, Vec<u32>>,
    cells: &mut Vec<Vec<u32>>,
    cell_mask: &mut Vec<u64>,
    cell_lineage: &mut Vec<u32>,
    groups: &mut Vec<Vec<u32>>,
    p_table: &mut Vec<Vec<u32>>,
    worklist: &mut Vec<u32>,
    in_worklist: &mut Vec<bool>,
    refiners: &[BinVec],
    n_cols_max: usize,
) {
    let old_mask = cell_mask[ci];
    // Sub-cells in lex-by-bucket-key order, each internally sorted.
    let mut sub_cells: Vec<Vec<u32>> = buckets
        .into_iter()
        .map(|(_, mut v)| {
            v.sort();
            v
        })
        .collect();
    let sub_masks: Vec<u64> =
        sub_cells.iter().map(|c| mask_of_cell(c)).collect();

    // Largest fragment (ties broken by earliest sub-cell index) keeps
    // slot ci; the others append.
    let largest_idx: usize = (0..sub_cells.len())
        .max_by_key(|&i| (sub_cells[i].len(), std::cmp::Reverse(i)))
        .unwrap();
    let parent_lineage = cell_lineage[ci];

    cells[ci] = std::mem::take(&mut sub_cells[largest_idx]);
    cell_mask[ci] = sub_masks[largest_idx];
    for (idx, sub) in sub_cells.into_iter().enumerate() {
        if idx == largest_idx {
            continue;
        }
        cell_mask.push(sub_masks[idx]);
        cell_lineage.push(parent_lineage);
        cells.push(sub);
    }

    // Propagate to groups. A group `gi` "touches" the old cell iff any
    // of its refiners has a bit in `old_mask`. Iterate by index over a
    // snapshot since we push new groups inside the loop.
    let n_groups_snapshot = groups.len();
    for gi in 0..n_groups_snapshot {
        if groups[gi].is_empty() {
            continue;
        }
        let touches = groups[gi]
            .iter()
            .any(|&ri| refiners[ri as usize] & old_mask != 0);
        if !touches {
            continue;
        }

        // Bucket refiners in this group by their per-sub-cell histogram.
        let mut rbuckets: BTreeMap<Vec<u32>, Vec<u32>> = BTreeMap::new();
        for &ri in &groups[gi] {
            let w = refiners[ri as usize];
            let hist: Vec<u32> = sub_masks
                .iter()
                .map(|m| (w & m).count_ones())
                .collect();
            rbuckets.entry(hist).or_default().push(ri);
        }
        if rbuckets.len() == 1 {
            continue;
        }

        // Group gi splits. Largest fragment keeps slot gi; others append.
        let frag_lists: Vec<Vec<u32>> =
            rbuckets.into_values().collect();
        let largest_g_idx: usize = (0..frag_lists.len())
            .max_by_key(|&i| {
                (frag_lists[i].len(), std::cmp::Reverse(i))
            })
            .unwrap();

        let mut new_group_ids: Vec<u32> = vec![0; frag_lists.len()];
        for (fi, frag) in frag_lists.into_iter().enumerate() {
            if fi == largest_g_idx {
                p_table[gi] = recompute_p(&frag, refiners, n_cols_max);
                groups[gi] = frag;
                new_group_ids[fi] = gi as u32;
            } else {
                let new_gi = groups.len();
                p_table.push(recompute_p(&frag, refiners, n_cols_max));
                groups.push(frag);
                in_worklist.push(false);
                new_group_ids[fi] = new_gi as u32;
            }
        }

        // Smallest-fragment rule: if gi was on worklist, push all new
        // groups; else push all but the largest.
        if in_worklist[gi] {
            for &gid in &new_group_ids {
                if !in_worklist[gid as usize] {
                    in_worklist[gid as usize] = true;
                    worklist.push(gid);
                }
            }
        } else {
            for (fi, &gid) in new_group_ids.iter().enumerate() {
                if fi == largest_g_idx {
                    continue;
                }
                if !in_worklist[gid as usize] {
                    in_worklist[gid as usize] = true;
                    worklist.push(gid);
                }
            }
        }
    }
}

/// `p[g][j] = #{r in refs : (refiners[r] >> j) & 1}`. Iterates set
/// bits of each refiner; cost is `O(sum_r weight(r))`, not `O(N · |refs|)`.
fn recompute_p(refs: &[u32], refiners: &[BinVec], n_cols_max: usize) -> Vec<u32> {
    let mut row = vec![0u32; n_cols_max];
    for &ri in refs {
        let mut bits = refiners[ri as usize];
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            row[j] += 1;
            bits &= bits - 1;
        }
    }
    row
}

/// Emit cells in deterministic order. Primary key: input lineage
/// (preserves the McKay convention that each input cell's split
/// products form a contiguous block in the output, with cells before
/// the parent slot appearing first). Secondary: full final signature
/// across all groups. Tertiary: min column.
fn emit_sorted(
    cells: Vec<Vec<u32>>,
    p_table: &[Vec<u32>],
    cell_lineage: &[u32],
) -> Vec<Vec<u32>> {
    let n_groups = p_table.len();
    let mut keyed: Vec<(u32, Vec<u32>, u32, Vec<u32>)> = cells
        .into_iter()
        .enumerate()
        .filter(|(_, c)| !c.is_empty())
        .map(|(i, mut c)| {
            c.sort();
            let rep = c[0];
            let sig: Vec<u32> =
                (0..n_groups).map(|g| p_table[g][rep as usize]).collect();
            (cell_lineage[i], sig, rep, c)
        })
        .collect();
    keyed.sort_by(|a, b| {
        (a.0, &a.1, a.2).cmp(&(b.0, &b.1, b.2))
    });
    keyed.into_iter().map(|(_, _, _, c)| c).collect()
}

pub(crate) fn individualise(p: &[Vec<u32>], cell_idx: usize, col: u32) -> Vec<Vec<u32>> {
    let mut new_p: Vec<Vec<u32>> = Vec::with_capacity(p.len() + 1);
    new_p.extend_from_slice(&p[..cell_idx]);
    new_p.push(vec![col]);
    let rest: Vec<u32> =
        p[cell_idx].iter().copied().filter(|&c| c != col).collect();
    if !rest.is_empty() {
        new_p.push(rest);
    }
    new_p.extend_from_slice(&p[cell_idx + 1..]);
    new_p
}

// ------------------------------------------------------------- search engine

/// Incremental canonical-form key threaded through the search.
///
/// Mirrors `_PartialKey` in `doubly_even.canon.feulner`: at each
/// singleton emission we Gaussian-eliminate that column to a unit
/// vector and **swap the pivot row to position `depth`** so the column
/// trace is invariant under the inner group `GL_k(F_2)`. The key is
/// compared **lex-from-low** (entry 0 most significant), so partial
/// information at depth `d` sits in the most-significant prefix and
/// the prefix prune is structurally strong.
///
/// Invariant: rows `0..depth` of `work` are pivot rows in pivot order;
/// rows `depth..k` are uncovered.
///
/// The state is mutated in place across the recursion; each `search`
/// call captures a `Snapshot` before descending and `restore`s with
/// the accumulated row-mutation `log` on the way back up. This avoids
/// the per-descent `Vec` clones that dominated the cost otherwise.
pub(crate) struct PartialKey {
    k: u32,
    work: Vec<BinVec>,
    pub(crate) depth: u32,
    pub(crate) key: Vec<u64>,
    pub(crate) absorbed_cols: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct Snapshot {
    depth: u32,
    absorbed_cols: u64,
    key_len: usize,
}

impl PartialKey {
    pub(crate) fn new(rref: &[BinVec]) -> Self {
        Self {
            k: rref.len() as u32,
            work: rref.to_vec(),
            depth: 0,
            key: Vec::new(),
            absorbed_cols: 0,
        }
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            depth: self.depth,
            absorbed_cols: self.absorbed_cols,
            key_len: self.key.len(),
        }
    }

    /// Undo every absorb performed since `snap` was taken. `log` holds
    /// `(row_index, prior_value)` entries pushed by absorb in order;
    /// we restore them in reverse so a swap-then-XOR sequence unwinds
    /// to the original layout.
    pub(crate) fn restore(&mut self, snap: Snapshot, log: &[(u32, BinVec)]) {
        for &(r, val) in log.iter().rev() {
            self.work[r as usize] = val;
        }
        self.depth = snap.depth;
        self.absorbed_cols = snap.absorbed_cols;
        self.key.truncate(snap.key_len);
    }

    /// Absorb column `c`. Each row of `work` that gets mutated is first
    /// pushed to `log` (as `(row_index, prior_value)`); pivot-swap is
    /// expressed as two such mutations.
    pub(crate) fn absorb(&mut self, c: u32, log: &mut Vec<(u32, BinVec)>) {
        let mut pivot: i32 = -1;
        for r in self.depth..self.k {
            if (self.work[r as usize] >> c) & 1 == 1 {
                pivot = r as i32;
                break;
            }
        }
        if pivot >= 0 {
            let pivot = pivot as u32;
            let depth_usize = self.depth as usize;
            if pivot != self.depth {
                let pivot_usize = pivot as usize;
                log.push((self.depth, self.work[depth_usize]));
                log.push((pivot, self.work[pivot_usize]));
                self.work.swap(depth_usize, pivot_usize);
            }
            let pivot_val = self.work[depth_usize];
            for r in 0..self.k as usize {
                if r != depth_usize && (self.work[r] >> c) & 1 == 1 {
                    log.push((r as u32, self.work[r]));
                    self.work[r] ^= pivot_val;
                }
            }
            self.depth += 1;
        }
        let mut col_bits: u64 = 0;
        for r in 0..self.k as usize {
            if (self.work[r] >> c) & 1 == 1 {
                col_bits |= 1u64 << (self.k as usize - 1 - r);
            }
        }
        self.key.push(col_bits);
        self.absorbed_cols |= 1u64 << c;
    }
}

struct SearchState {
    n: u32,
    key_to_pi: HashMap<Vec<u64>, Perm>,
    best_key: Option<Vec<u64>>,
    aut_gens: Vec<Perm>,
    seen_gens: HashSet<Perm>,
    leaves: u64,
    prunes: u64,
    clb_prunes: u64,
    clb: LabelledBranching,
}

impl SearchState {
    fn new(n: u32) -> Self {
        Self {
            n,
            key_to_pi: HashMap::new(),
            best_key: None,
            aut_gens: Vec::new(),
            seen_gens: HashSet::new(),
            leaves: 0,
            prunes: 0,
            clb_prunes: 0,
            clb: LabelledBranching::new(n),
        }
    }

    /// Push an aut generator only if it hasn't already been collected.
    /// Mirrors the Python `push_aut`: also informs the CLB so Lemma 5.9
    /// can use the new generator immediately.
    fn push_aut(&mut self, g: Perm) {
        if self.seen_gens.insert(g.clone()) {
            self.aut_gens.push(g.clone());
            self.clb.add_gen(g);
        }
    }
}

fn search(
    p: Vec<Vec<u32>>,
    refiners: &[BinVec],
    state: &mut SearchState,
    path: &mut Vec<u32>,
    partial: &mut PartialKey,
) {
    let p = refine(p, refiners);

    // CLB topological-sort prune (Feulner §5.2 / Sage's
    // `_cut_by_known_automs`, call site #1 — top of every backtrack
    // node, after refinement). Only meaningful once we have a candidate
    // canonical leaf (so before then, `best_key` is `None` and we skip).
    if state.best_key.is_some() && state.clb.has_empty_intersection(&p) {
        state.clb_prunes += 1;
        return;
    }

    // Absorb every singleton not yet in the key, in cell order. Mutates
    // `partial` in place; `log` records every row mutation so we can
    // restore on return. Check the prefix prune after each absorb.
    let snap = partial.snapshot();
    let mut log: Vec<(u32, BinVec)> = Vec::new();
    for cell in &p {
        if cell.len() == 1 {
            let c = cell[0];
            if (partial.absorbed_cols >> c) & 1 == 0 {
                partial.absorb(c, &mut log);
                if let Some(bk) = state.best_key.as_ref() {
                    let d = partial.key.len();
                    if d <= bk.len() {
                        for i in 0..d {
                            let a = partial.key[i];
                            let b = bk[i];
                            if a < b {
                                break;
                            }
                            if a > b {
                                state.prunes += 1;
                                partial.restore(snap, &log);
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    // Note: Sage's `_cut_by_known_automs` is called *twice* per node — once
    // at the top of `_backtrack`, once inside `_one_refinement` after each
    // refinement step changes the partition. Our `refine()` is a single-shot
    // equitable-refinement call (we don't iterate; the result is already
    // fully equitable), so `p` is unchanged between the top-of-node check
    // above and here. A second check on the same partition would never fire,
    // so we skip it.

    if p.iter().all(|c| c.len() == 1) {
        state.leaves += 1;
        let mut pi: Perm = vec![0u32; state.n as usize];
        for (new_pos, cell) in p.iter().enumerate() {
            pi[cell[0] as usize] = new_pos as u32;
        }

        if let Some(prior) = state.key_to_pi.get(&partial.key) {
            let inv_prior = perm_inverse(prior);
            let aut_gen = perm_compose(&inv_prior, &pi);
            state.push_aut(aut_gen);
        } else {
            let key = partial.key.clone();
            let is_new_best =
                state.best_key.as_ref().map_or(true, |b| &key < b);
            state.key_to_pi.insert(key.clone(), pi);
            if is_new_best {
                state.best_key = Some(key);
            }
        }
        partial.restore(snap, &log);
        return;
    }

    let cell_idx = p.iter().position(|c| c.len() > 1).unwrap();
    let cell = p[cell_idx].clone();

    let mut seen: HashSet<u32> = HashSet::new();
    let mut orbit_rep: HashMap<u32, u32> =
        cell.iter().map(|&c| (c, c)).collect();
    let mut last_n_gens: usize = usize::MAX;
    for &col in &cell {
        if state.aut_gens.len() != last_n_gens {
            last_n_gens = state.aut_gens.len();
            let fixing_gens: Vec<&Perm> = state
                .aut_gens
                .iter()
                .filter(|g| path.iter().all(|&pj| g[pj as usize] == pj))
                .collect();
            orbit_rep = if !fixing_gens.is_empty() {
                orbits_on_subset(&fixing_gens, &cell)
            } else {
                cell.iter().map(|&c| (c, c)).collect()
            };
        }
        let rep = orbit_rep[&col];
        if !seen.insert(rep) {
            continue;
        }
        let new_p = individualise(&p, cell_idx, col);
        path.push(col);
        search(new_p, refiners, state, path, partial);
        path.pop();
    }

    partial.restore(snap, &log);
}

// ------------------------------------------------------- S_n fast paths

fn factorial_u128(n: u128) -> u128 {
    let mut r: u128 = 1;
    for i in 1..=n {
        r = r.checked_mul(i).expect("factorial overflows u128");
    }
    r
}

fn sn_canon_info(n: u32) -> FeulnerCanonInfo {
    let mut gens: Vec<Perm> = Vec::new();
    if n >= 2 {
        let mut swap: Perm = (0..n).collect();
        swap[0] = 1;
        swap[1] = 0;
        gens.push(swap);
    }
    if n >= 3 {
        let cycle: Perm = (1..n).chain(std::iter::once(0)).collect();
        gens.push(cycle);
    }
    let order = factorial_u128(n as u128);
    FeulnerCanonInfo {
        canonical_column_order: (0..n).collect(),
        aut_generators: gens,
        aut_order_decimal: order.to_string(),
        column_orbits: vec![0u32; n as usize],
        leaves: 0,
        prunes: 0,
        clb_prunes: 0,
    }
}

// ------------------------------------------------------------- public entry

/// Compute `CanonInfo`-equivalent data via Feulner-style search.
///
/// `rref` is the RREF basis of the code (already row-reduced; pivots in
/// strictly increasing order). `n` is the code length.
pub fn canon_info_feulner(rref: &[BinVec], n: u32) -> FeulnerCanonInfo {
    if n == 0 {
        return FeulnerCanonInfo {
            canonical_column_order: Vec::new(),
            aut_generators: Vec::new(),
            aut_order_decimal: "1".to_string(),
            column_orbits: Vec::new(),
            leaves: 0,
            prunes: 0,
            clb_prunes: 0,
        };
    }
    let k = rref.len();
    if k == 0 || k == n as usize {
        return sn_canon_info(n);
    }

    let refiners = invariant_refiners(rref);
    let mut state = SearchState::new(n);
    let initial = initial_partition(rref, n);
    let mut partial = PartialKey::new(rref);
    let mut path: Vec<u32> = Vec::new();
    search(initial, &refiners, &mut state, &mut path, &mut partial);

    let aut_order = group_order(&state.aut_gens, n);
    let best_key =
        state.best_key.expect("search produced no discrete partition");
    let transporter = state
        .key_to_pi
        .get(&best_key)
        .expect("best_key not present in key_to_pi")
        .clone();
    let column_orbits = compute_column_orbits(&state.aut_gens, n);

    FeulnerCanonInfo {
        canonical_column_order: transporter,
        aut_generators: state.aut_gens,
        aut_order_decimal: aut_order.to_string(),
        column_orbits,
        leaves: state.leaves,
        prunes: state.prunes,
        clb_prunes: state.clb_prunes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_code_n4() {
        let info = canon_info_feulner(&[], 4);
        assert_eq!(info.aut_order_decimal, "24");
        assert_eq!(info.column_orbits, vec![0u32; 4]);
    }

    #[test]
    fn extended_hamming_8_4() {
        let rref: Vec<BinVec> = vec![
            0b00001111,
            0b00110011,
            0b01010101,
            0b11111111,
        ];
        // Re-RREF in case input isn't already (it is, but be safe).
        let (rref, _) = crate::linalg::row_reduce(&rref, 8);
        let info = canon_info_feulner(&rref, 8);
        assert_eq!(info.aut_order_decimal, "1344");
    }

    #[test]
    fn single_weight4_in_f2_8() {
        let info = canon_info_feulner(&[0b00001111], 8);
        // |S_4 x S_4| = 24 * 24 = 576
        assert_eq!(info.aut_order_decimal, "576");
    }
}
