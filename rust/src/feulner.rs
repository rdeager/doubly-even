//! Feulner-style column-side canonicaliser, binary + permutation-only.
//!
//! Direct port of `doubly_even.canon.feulner` — the Python staging is the
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
}

// ----------------------------------------------------- permutation utilities

type Perm = Vec<u32>;

fn perm_identity(n: u32) -> Perm {
    (0..n).collect()
}

/// `(p ∘ q)[i] = p[q[i]]` — apply `q` first, then `p`. Same convention as
/// `doubly_even.canon.permutations.compose`.
fn perm_compose(p: &[u32], q: &[u32]) -> Perm {
    q.iter().map(|&qi| p[qi as usize]).collect()
}

fn perm_inverse(p: &[u32]) -> Perm {
    let mut inv = vec![0u32; p.len()];
    for (i, &j) in p.iter().enumerate() {
        inv[j as usize] = i as u32;
    }
    inv
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

fn compute_column_orbits(aut_gens: &[Perm], n: u32) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..n).collect();

    fn find(parent: &mut [u32], i: u32) -> u32 {
        let mut cur = i;
        while parent[cur as usize] != cur {
            let next = parent[cur as usize];
            let gp = parent[next as usize];
            parent[cur as usize] = gp;
            cur = gp;
        }
        cur
    }

    fn union(parent: &mut [u32], a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return;
        }
        if ra < rb {
            parent[rb as usize] = ra;
        } else {
            parent[ra as usize] = rb;
        }
    }

    for g in aut_gens {
        for (i, &j) in g.iter().enumerate() {
            if i as u32 != j {
                union(&mut parent, i as u32, j);
            }
        }
    }
    (0..n).map(|i| find(&mut parent, i)).collect()
}

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
fn initial_partition(rref: &[BinVec], n: u32) -> Vec<Vec<u32>> {
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
/// Prefers weight-4 (the chromotopology stratum for doubly-even codes; most
/// discriminative per Bouyukliev–Bouyuklieva §3.3). Falls back to the
/// lowest non-zero weight stratum otherwise.
fn invariant_refiners(rref: &[BinVec]) -> Vec<BinVec> {
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
    if let Some(wt4) = by_weight.remove(&4) {
        return wt4;
    }
    // first entry is the minimum weight
    by_weight
        .into_iter()
        .next()
        .map(|(_, v)| v)
        .unwrap_or_default()
}

fn mask_of_cell(cell: &[u32]) -> u64 {
    let mut m: u64 = 0;
    for &j in cell {
        m |= 1u64 << j;
    }
    m
}

/// Equitable refinement of column partition `p` by refiner-incidence
/// signatures. Returns a new partition (sorted within each cell).
fn refine(mut p: Vec<Vec<u32>>, refiners: &[BinVec]) -> Vec<Vec<u32>> {
    loop {
        let cell_masks: Vec<u64> = p.iter().map(|c| mask_of_cell(c)).collect();

        // Group refiners by their cell-histogram type (Aut-invariant under
        // the current partition).
        let mut groups: BTreeMap<Vec<u32>, Vec<BinVec>> = BTreeMap::new();
        for &w in refiners {
            let t: Vec<u32> =
                cell_masks.iter().map(|m| (w & m).count_ones()).collect();
            groups.entry(t).or_default().push(w);
        }
        // Ordered (BTreeMap iterates by key); flatten to a Vec for indexed
        // signature construction.
        let ordered: Vec<Vec<BinVec>> =
            groups.into_iter().map(|(_, v)| v).collect();

        // Per-column signature: for each refiner type group, count how many
        // members include column j.
        let n_cols: usize = p.iter().map(|c| c.len()).sum();
        let mut sig_of: HashMap<u32, Vec<u32>> = HashMap::with_capacity(n_cols);
        for cell in &p {
            for &j in cell {
                let mut s = Vec::with_capacity(ordered.len());
                for words in &ordered {
                    let mut count: u32 = 0;
                    for &w in words {
                        if (w >> j) & 1 == 1 {
                            count += 1;
                        }
                    }
                    s.push(count);
                }
                sig_of.insert(j, s);
            }
        }

        let mut new_p: Vec<Vec<u32>> = Vec::with_capacity(p.len());
        let mut changed = false;
        for cell in &p {
            if cell.len() == 1 {
                new_p.push(cell.clone());
                continue;
            }
            let mut buckets: BTreeMap<Vec<u32>, Vec<u32>> = BTreeMap::new();
            for &j in cell {
                buckets.entry(sig_of[&j].clone()).or_default().push(j);
            }
            if buckets.len() > 1 {
                changed = true;
            }
            for (_, mut bucket) in buckets {
                bucket.sort();
                new_p.push(bucket);
            }
        }
        if !changed {
            return p;
        }
        p = new_p;
    }
}

fn individualise(p: &[Vec<u32>], cell_idx: usize, col: u32) -> Vec<Vec<u32>> {
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
struct PartialKey {
    k: u32,
    work: Vec<BinVec>,
    depth: u32,
    key: Vec<u64>,
    absorbed_cols: u64,
}

#[derive(Clone, Copy)]
struct Snapshot {
    depth: u32,
    absorbed_cols: u64,
    key_len: usize,
}

impl PartialKey {
    fn new(rref: &[BinVec]) -> Self {
        Self {
            k: rref.len() as u32,
            work: rref.to_vec(),
            depth: 0,
            key: Vec::new(),
            absorbed_cols: 0,
        }
    }

    fn snapshot(&self) -> Snapshot {
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
    fn restore(&mut self, snap: Snapshot, log: &[(u32, BinVec)]) {
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
    fn absorb(&mut self, c: u32, log: &mut Vec<(u32, BinVec)>) {
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
        }
    }

    /// Push an aut generator only if it hasn't already been collected.
    fn push_aut(&mut self, g: Perm) {
        if self.seen_gens.insert(g.clone()) {
            self.aut_gens.push(g);
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
