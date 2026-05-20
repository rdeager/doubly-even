//! `crate::experimental::invariants` — WL + T11/T12/T13 collision-
//! experiment substrate. **EXPERIMENTAL**, cheap permutation-invariant
//! signatures for doubly-even codes — Rust port of the per-code primitives
//! in `scripts/experimental/wl_collision_experiment.py`. Not on the kernel
//! hot path; exposed as standalone pyfunctions for the collision-rate
//! audit. See `project_1wl_collision_experiment.md` and
//! `feedback_no_offline_blocklist.md` for why the entire family closed.
//!
//! What's here and why:
//!
//! - `Bipartite` — CSR adjacency for the codeword × column bipartite graph
//!   that the D10 Q_D-graph canonicaliser hands to nauty (`canon.rs:432`).
//!   Supports two constructions: `build_full_bipartite` over all `2^k - 1`
//!   nonzero codewords (full G(C)), and `build_low_weight_bipartite` over
//!   the span-aware low-weight subset (G_min, mirrors
//!   `canon.rs:369-417`).
//! - `wl_refine` — 1-Weisfeiler-Leman colour refinement to fixed point on a
//!   `Bipartite`, with two initial partitions (`InitMode::Vanilla` and
//!   `InitMode::DegreeAndWeight`, matching the Python `vanilla` /
//!   `degree_init` variants).
//! - `t11_signature` — per-column profile of `#wt-w codewords through col j`
//!   for a fixed weight tuple, returned as a sorted multiset.
//! - `t12_signature` — per unordered min-weight-codeword triple `(a,b,c)`,
//!   sorted `(wt(a⊕b), wt(a⊕c), wt(b⊕c))`. Triangle-like; provably not
//!   captured by 1-WL on the incidence graph.
//! - `t13_signature` — pair-gram: per unordered column-pair `(i,j)`, the
//!   tuple `(#wt-w cws containing both i and j)` for a fixed weight tuple.
//!   Bitmask implementation: `popcount(col_mask[i] & col_mask[j] &
//!   wt_mask[w])`.
//!
//! These are exposed via PyO3 in `lib.rs`. They do NOT touch the kernel hot
//! path (`enumerate.rs`); they are a standalone substrate for the collision
//! experiment.
//!
//! ## Hash construction (NOT byte-equal to Python)
//!
//! Python's WL implementation uses the built-in `hash` builtin on tuples;
//! that hash is per-process-randomised (PYTHONHASHSEED) so Python ↔ Rust
//! cross-validation is on **bucketing** (the partition codes → signature
//! buckets), not byte-equal output. Internally we use:
//!
//! - `mix64` — SplitMix64-style integer mixer (Stafford 13).
//! - An **order-independent additive multiset hash**:
//!   `multiset_hash(s) = Σ mix64(x) for x in s` (wrapping). This avoids
//!   sorting the neighbour-colour slice per vertex per round — at N = 26
//!   full G(C) right-side has ~4 K-degree columns where sort_unstable
//!   would dominate.
//!
//! Multiset collisions (two distinct multisets summing to the same u64) are
//! possible but astronomically rare for the populations we run (~10⁷ hash
//! invocations / experiment); we accept the theoretical loss.
//!
//! For the cheap invariants (T11/T13) we pack per-column / per-pair tuples
//! losslessly into u64 (each component ≤ 2^k ≤ 8192 ≤ 16 bits, 4 components
//! fits in 64 bits), so those are *exact* — no hash collision possible.

use std::collections::HashMap;

use crate::linalg::row_reduce;
use crate::types::BinVec;

// ───────────────────────────────────────────────────────────────── helpers

#[inline]
fn mix64(x: u64) -> u64 {
    // SplitMix64 (Stafford 13) with an FNV-offset XOR bias on the input.
    // The bias kills the `mix64(0) == 0` fixed point — without it, any
    // colour that ever becomes 0 acts as an additive identity in the
    // multiset hash, collapsing entire neighbourhoods to a single bucket.
    let mut x = x ^ 0xcbf29ce484222325;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

#[inline]
fn combine3(side: u64, old: u64, multiset_hash: u64) -> u64 {
    mix64(
        mix64(side)
            .wrapping_add(mix64(old).wrapping_mul(0x9e3779b97f4a7c15))
            .wrapping_add(multiset_hash.wrapping_mul(0xbf58476d1ce4e5b9)),
    )
}

fn distinct_count(colors: &[u64]) -> usize {
    let mut s: Vec<u64> = colors.to_vec();
    s.sort_unstable();
    s.dedup();
    s.len()
}

// ──────────────────────────────────────────────────────── codeword enumeration

/// Gray-code walk over all `2^k - 1` nonzero codewords, grouped by Hamming
/// weight in ascending order. Returns `(flat_codewords, stratum_sizes)`.
fn all_codewords_by_stratum(rref: &[BinVec], n: u32) -> (Vec<BinVec>, Vec<u32>) {
    let k = rref.len();
    if k == 0 {
        return (Vec::new(), Vec::new());
    }
    let total = 1usize << k;
    let mut by_weight: Vec<Vec<BinVec>> = vec![Vec::new(); (n as usize) + 1];
    let mut w: BinVec = 0;
    for mask in 1..total {
        let lo_bit = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        w ^= rref[lo_bit];
        by_weight[w.count_ones() as usize].push(w);
    }
    let mut flat: Vec<BinVec> = Vec::with_capacity(total - 1);
    let mut strata: Vec<u32> = Vec::new();
    for weight in 1..=(n as usize) {
        if !by_weight[weight].is_empty() {
            let stratum = std::mem::take(&mut by_weight[weight]);
            strata.push(stratum.len() as u32);
            flat.extend(stratum);
        }
    }
    (flat, strata)
}

/// Span-aware low-weight codeword set (mirrors `canon.rs:369-417`).
///
/// Walks codewords by ascending weight (Gray code) and accumulates strata
/// until their span equals C. Returns `None` if the accumulated set reaches
/// `2^(k-1)` before spanning — caller falls back to full G(C).
pub fn low_weight_codewords_pub(
    rref: &[BinVec],
    n: u32,
) -> Option<(Vec<BinVec>, Vec<u32>)> {
    let k = rref.len();
    if k == 0 {
        return None;
    }
    let total = 1usize << k;
    let bail = total / 2;

    let mut by_weight: Vec<Vec<BinVec>> = vec![Vec::new(); (n as usize) + 1];
    let mut w: BinVec = 0;
    for mask in 1..total {
        let lo_bit = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        w ^= rref[lo_bit];
        by_weight[w.count_ones() as usize].push(w);
    }

    let mut accum: Vec<BinVec> = Vec::new();
    let mut strata: Vec<u32> = Vec::new();
    for weight in 1..=(n as usize) {
        if by_weight[weight].is_empty() {
            continue;
        }
        let stratum = std::mem::take(&mut by_weight[weight]);
        let stratum_len = stratum.len();
        accum.extend(stratum);
        strata.push(stratum_len as u32);
        if accum.len() > bail {
            return None;
        }
        let (rr, _) = row_reduce(&accum, n);
        if rr.len() == k {
            return Some((accum, strata));
        }
    }
    None
}

/// Public helper used by the cross-language T-tests: full nonzero codewords
/// (Gray-code order), flat.
pub fn full_codewords(rref: &[BinVec], _n: u32) -> Vec<BinVec> {
    let k = rref.len();
    if k == 0 {
        return Vec::new();
    }
    let total = 1usize << k;
    let mut out: Vec<BinVec> = Vec::with_capacity(total - 1);
    let mut w: BinVec = 0;
    for mask in 1..total {
        let lo_bit = (mask & mask.wrapping_neg()).trailing_zeros() as usize;
        w ^= rref[lo_bit];
        out.push(w);
    }
    out
}

// ─────────────────────────────────────────────────────────────── bipartite

/// CSR bipartite-graph adjacency. Left side = codewords (grouped into
/// `left_strata`), right side = columns.
pub struct Bipartite {
    pub n_left: u32,
    pub n_right: u32,
    /// Sizes of left-side strata in ascending-weight order. Sum =
    /// `n_left`. Used to build the `degree_init` initial partition.
    pub left_strata: Vec<u32>,
    /// Left CSR offsets, length `n_left + 1`. Vertex `i`'s neighbours are
    /// `left_nbr[left_off[i]..left_off[i+1]]` (column indices in
    /// `[0, n_right)`).
    pub left_off: Vec<u32>,
    pub left_nbr: Vec<u32>,
    /// Right CSR offsets, length `n_right + 1`. Vertex `j`'s neighbours are
    /// `right_nbr[right_off[j]..right_off[j+1]]` (codeword indices in
    /// `[0, n_left)`).
    pub right_off: Vec<u32>,
    pub right_nbr: Vec<u32>,
}

fn build_bipartite_from_codewords(
    codewords: &[BinVec],
    strata: &[u32],
    n: u32,
) -> Bipartite {
    let l = codewords.len();
    let r = n as usize;

    // Left CSR.
    let mut left_off: Vec<u32> = Vec::with_capacity(l + 1);
    let mut left_nbr: Vec<u32> = Vec::new();
    left_off.push(0);
    for &cw in codewords {
        let mut bits = cw;
        while bits != 0 {
            let j = bits.trailing_zeros() as u32;
            left_nbr.push(j);
            bits &= bits - 1;
        }
        left_off.push(left_nbr.len() as u32);
    }

    // Right CSR (invert left).
    let mut right_deg = vec![0u32; r];
    for &j in &left_nbr {
        right_deg[j as usize] += 1;
    }
    let mut right_off: Vec<u32> = Vec::with_capacity(r + 1);
    right_off.push(0);
    let mut acc = 0u32;
    for &d in &right_deg {
        acc += d;
        right_off.push(acc);
    }
    let mut right_nbr = vec![0u32; acc as usize];
    let mut write_ptr: Vec<u32> = right_off[..r].to_vec();
    for i in 0..l {
        for k in left_off[i]..left_off[i + 1] {
            let j = left_nbr[k as usize] as usize;
            right_nbr[write_ptr[j] as usize] = i as u32;
            write_ptr[j] += 1;
        }
    }

    Bipartite {
        n_left: l as u32,
        n_right: r as u32,
        left_strata: strata.to_vec(),
        left_off,
        left_nbr,
        right_off,
        right_nbr,
    }
}

pub fn build_full_bipartite(rref: &[BinVec], n: u32) -> Bipartite {
    let (codewords, strata) = all_codewords_by_stratum(rref, n);
    build_bipartite_from_codewords(&codewords, &strata, n)
}

pub fn build_low_weight_bipartite(rref: &[BinVec], n: u32) -> Option<Bipartite> {
    let (codewords, strata) = low_weight_codewords_pub(rref, n)?;
    Some(build_bipartite_from_codewords(&codewords, &strata, n))
}

// ────────────────────────────────────────────────────────────────────── 1-WL

#[derive(Copy, Clone, Debug)]
pub enum InitMode {
    /// Codewords all colour 0, columns all colour 1. Textbook test.
    Vanilla,
    /// Codewords by weight stratum, columns by incidence degree.
    /// Matches the partition `build_low_weight_sparsegraph` hands nauty.
    DegreeAndWeight,
}

/// 1-WL colour refinement to fixed point. Returns the final per-side colour
/// sequence as a single sorted `Vec<u64>` with side-tag mixed in so
/// left/right colour spaces can't collide.
pub fn wl_refine(g: &Bipartite, init: InitMode) -> (Vec<u64>, u32) {
    let l = g.n_left as usize;
    let r = g.n_right as usize;

    if l == 0 || r == 0 {
        // Degenerate: just return per-side counts as a single u64 each.
        let sig: Vec<u64> = vec![
            combine3(0, 0, l as u64),
            combine3(1, 0, r as u64),
        ];
        return (sig, 0);
    }

    // Initial colours.
    let (mut cw_colors, mut col_colors): (Vec<u64>, Vec<u64>) = match init {
        InitMode::Vanilla => (vec![mix64(0); l], vec![mix64(1); r]),
        InitMode::DegreeAndWeight => {
            let mut cw = vec![0u64; l];
            let mut cursor = 0usize;
            for (stratum_idx, &size) in g.left_strata.iter().enumerate() {
                let s = size as usize;
                for slot in cw.iter_mut().skip(cursor).take(s) {
                    *slot = mix64(stratum_idx as u64);
                }
                cursor += s;
            }
            let stratum_count = g.left_strata.len() as u64;
            let mut col = vec![0u64; r];
            for (j, slot) in col.iter_mut().enumerate() {
                let deg = (g.right_off[j + 1] - g.right_off[j]) as u64;
                *slot = mix64(stratum_count.wrapping_add(deg));
            }
            (cw, col)
        }
    };

    let mut prev_total = distinct_count(&cw_colors) + distinct_count(&col_colors);
    let max_rounds = 2 * (l + r);

    let mut new_cw = vec![0u64; l];
    let mut new_col = vec![0u64; r];
    let mut rounds_used: u32 = 0;

    for round_idx in 0..max_rounds {
        // Codeword-side update.
        for i in 0..l {
            let start = g.left_off[i] as usize;
            let end = g.left_off[i + 1] as usize;
            let mut ms: u64 = 0;
            for k in start..end {
                let nbr = g.left_nbr[k] as usize;
                ms = ms.wrapping_add(mix64(col_colors[nbr]));
            }
            new_cw[i] = combine3(0, cw_colors[i], ms);
        }
        // Column-side update.
        for j in 0..r {
            let start = g.right_off[j] as usize;
            let end = g.right_off[j + 1] as usize;
            let mut ms: u64 = 0;
            for k in start..end {
                let nbr = g.right_nbr[k] as usize;
                ms = ms.wrapping_add(mix64(cw_colors[nbr]));
            }
            new_col[j] = combine3(1, col_colors[j], ms);
        }
        std::mem::swap(&mut cw_colors, &mut new_cw);
        std::mem::swap(&mut col_colors, &mut new_col);
        rounds_used = (round_idx + 1) as u32;

        let total = distinct_count(&cw_colors) + distinct_count(&col_colors);
        if total == prev_total {
            break;
        }
        prev_total = total;
    }

    // Side-tag and sort.
    let mut sig: Vec<u64> = Vec::with_capacity(l + r);
    for c in &cw_colors {
        sig.push(combine3(0, *c, 0));
    }
    for c in &col_colors {
        sig.push(combine3(1, *c, 0));
    }
    sig.sort_unstable();
    (sig, rounds_used)
}

// ─────────────────────────────────────────────────────────────────── T11

/// T11: per column j, tuple `(#wt-w cws through j, for w in weights)`.
/// Returns sorted multiset across columns.
///
/// Encoding: each per-column tuple is packed losslessly into a u128 —
/// 16 bits per count × up to 8 weights = 128 bits. Counts are bounded by
/// 2^k ≤ 2^32 ≤ 65535 at N ≤ 32 (k ≤ 16), so 16 bits per slot is
/// sufficient for our entire enumeration range. The fixed-weight (4, 8,
/// 12, 16) T11_full path uses only 4 slots (64 bits used, top 64 bits
/// zero); T11_gmin at N=32 uses all 8 slots.
///
/// **Why u128, not u64 + hashing**: u64 packing only fits 4 slots × 16
/// bits = 64 bits, so a hash would be required for `nw > 4`. Hash
/// collisions are vanishingly rare (~5e-6 across the run at N=26) but not
/// impossible; u128 sidesteps this with negligible cost (modern CPUs add/
/// compare/sort u128 in 1–2 cycles per element). The headline `wl + T12
/// + T13` metric doesn't depend on T11, so this is a robustness upgrade
/// for the per-component reporting, not a correctness fix for the
/// headline.
pub fn t11_signature(codewords: &[BinVec], n: u32, weights: &[u32]) -> Vec<u128> {
    let r = n as usize;
    let nw = weights.len();
    assert!(
        nw <= 8,
        "t11_signature: pack assumes ≤ 8 weights (16 bits each, 128 bits total)"
    );
    let weight_idx: HashMap<u32, usize> =
        weights.iter().enumerate().map(|(i, &w)| (w, i)).collect();
    let mut counts = vec![0u32; r * nw];
    for &cw in codewords {
        if let Some(&wi) = weight_idx.get(&cw.count_ones()) {
            let mut bits = cw;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                counts[j * nw + wi] += 1;
                bits &= bits - 1;
            }
        }
    }
    let mut sig: Vec<u128> = Vec::with_capacity(r);
    for j in 0..r {
        let mut packed: u128 = 0;
        for wi in 0..nw {
            packed |= ((counts[j * nw + wi] as u128) & 0xFFFF) << (16 * wi);
        }
        sig.push(packed);
    }
    sig.sort_unstable();
    sig
}

// ─────────────────────────────────────────────────────────────────── T12

/// T12: per unordered triple `(a,b,c)` of distinct min-weight codewords,
/// sorted `(wt(a⊕b), wt(a⊕c), wt(b⊕c))`. Returns sorted multiset of triples.
/// Each weight ≤ 64 (6 bits), packed losslessly into a u64.
pub fn t12_signature(codewords: &[BinVec]) -> Vec<u64> {
    if codewords.is_empty() {
        return Vec::new();
    }
    let m: u32 = codewords
        .iter()
        .filter(|c| **c != 0)
        .map(|c| c.count_ones())
        .min()
        .unwrap_or(0);
    let mins: Vec<BinVec> = codewords
        .iter()
        .copied()
        .filter(|c| c.count_ones() == m)
        .collect();
    let nv = mins.len();
    let mut out: Vec<u64> = Vec::new();
    for i in 0..nv {
        for j in (i + 1)..nv {
            let xab = (mins[i] ^ mins[j]).count_ones();
            for k in (j + 1)..nv {
                let xac = (mins[i] ^ mins[k]).count_ones();
                let xbc = (mins[j] ^ mins[k]).count_ones();
                // Sort the triple ascending and pack.
                let mut t = [xab as u64, xac as u64, xbc as u64];
                t.sort_unstable();
                let packed = t[0] | (t[1] << 8) | (t[2] << 16);
                out.push(packed);
            }
        }
    }
    out.sort_unstable();
    out
}

// ─────────────────────────────────────────────────────────────────── T13

/// T13 (pair-gram): per unordered column pair `(i, j)`, tuple
/// `(#wt-w cws containing both i and j, for w in weights)`.
/// Returns sorted multiset across pairs. Each component ≤ 2^k ≤ 65535
/// (16 bits), packed losslessly into u64 with ≤ 4 weights.
pub fn t13_signature(codewords: &[BinVec], n: u32, weights: &[u32]) -> Vec<u64> {
    let l = codewords.len();
    let r = n as usize;
    let nw = weights.len();
    assert!(nw <= 4, "t13_signature: pack assumes ≤ 4 weights");
    if l == 0 {
        return Vec::new();
    }

    // Bitmasks over codeword indices, in 64-bit chunks.
    let num_words = l.div_ceil(64);

    let weight_idx: HashMap<u32, usize> =
        weights.iter().enumerate().map(|(i, &w)| (w, i)).collect();

    // col_mask[j * num_words + word] : codewords through column j.
    let mut col_mask = vec![0u64; r * num_words];
    // wt_mask[wi * num_words + word] : codewords of weight weights[wi].
    let mut wt_mask = vec![0u64; nw * num_words];

    for (i, &cw) in codewords.iter().enumerate() {
        let word = i / 64;
        let bit = 1u64 << (i % 64);
        if let Some(&wi) = weight_idx.get(&cw.count_ones()) {
            wt_mask[wi * num_words + word] |= bit;
        }
        let mut bits = cw;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            col_mask[j * num_words + word] |= bit;
            bits &= bits - 1;
        }
    }

    let num_pairs = r * r.saturating_sub(1) / 2;
    let mut sig: Vec<u64> = Vec::with_capacity(num_pairs);
    for i in 0..r {
        for j in (i + 1)..r {
            let mut packed: u64 = 0;
            for wi in 0..nw {
                let mut count: u32 = 0;
                let wstart = wi * num_words;
                let istart = i * num_words;
                let jstart = j * num_words;
                for w in 0..num_words {
                    let a = col_mask[istart + w];
                    let b = col_mask[jstart + w];
                    let m = wt_mask[wstart + w];
                    count += (a & b & m).count_ones();
                }
                packed |= ((count as u64) & 0xFFFF) << (16 * wi);
            }
            sig.push(packed);
        }
    }
    sig.sort_unstable();
    sig
}

// ──────────────────────────────────── 128-bit digest of a sorted multiset
//
// For the N=24/26 collision experiment, per-code WL signatures are large
// (max L ≈ 8K at N=26), so storing them across 494K codes in Python tuples
// would cost ~440 GB. Instead we hash each sorted multiset to a single
// u128 digest on the Rust side and ship just the digest. Bucketing is on
// digest equality. Collision probability over the run is ~(494K)² / 2¹²⁸
// ≈ 10⁻²³ — completely negligible.
//
// The digest uses two parallel polynomial hashes with different seed/
// update schedules, so two distinct sorted multisets collide only when
// BOTH hashes collide. Inputs are pre-mixed with mix64 to randomise away
// any pathological u64 patterns.

#[inline]
fn digest_finalize(h1: u64, h2: u64) -> u128 {
    ((h1 as u128) << 64) | (h2 as u128)
}

/// 128-bit digest of a sorted `&[u64]` multiset. The slice MUST be sorted
/// (callers always sort before digesting). Output depends only on the
/// multiset value, not on memory layout.
pub fn hash_sorted_u64(values: &[u64]) -> u128 {
    let mut h1: u64 = mix64(0x123456789abcdef0);
    let mut h2: u64 = mix64(0xfedcba9876543210);
    for &v in values {
        let mv = mix64(v);
        h1 = mix64(h1.wrapping_add(mv));
        h2 = mix64(h2.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(mv));
    }
    digest_finalize(h1, h2)
}

/// 128-bit digest of a sorted `&[u128]` multiset (used for T11 which packs
/// 8-tuples losslessly into u128).
pub fn hash_sorted_u128(values: &[u128]) -> u128 {
    let mut h1: u64 = mix64(0x0fedcba987654321);
    let mut h2: u64 = mix64(0x0123456789abcdef);
    for &v in values {
        // Mix the 128-bit input asymmetrically in lo/hi — `lo XOR mix64(hi)`
        // distinguishes `(lo=1, hi=0)` from `(lo=0, hi=1)` (which would
        // otherwise collide under a symmetric `mix64(lo) + mix64(hi)`).
        // This collision was observed at N=20 between codes 70 (per-col
        // tuple (1,0,0,0,0)) and 1210 ((0,0,0,0,1)).
        let lo = v as u64;
        let hi = (v >> 64) as u64;
        let mv = mix64(lo ^ mix64(hi));
        h1 = mix64(h1.wrapping_add(mv));
        h2 = mix64(h2.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(mv));
    }
    digest_finalize(h1, h2)
}

// ─────────────────────────────────────────────────────────── combined entry

/// Per-code invariant digests. Each component is hashed to a u128 so the
/// caller can bucket without paying memory for the full sorted multiset.
pub struct InvariantDigests {
    pub wl_min_vanilla: u128,
    pub wl_min_degree: u128,
    pub wl_full_vanilla: u128,
    pub wl_full_degree: u128,
    pub t11_full: u128,
    pub t11_gmin: u128,
    pub t12: u128,
    pub t13: u128,
    /// T13 restricted to minimum-weight codewords only (single-weight slice).
    /// Diagnostic: tests whether the min-weight stratum alone carries enough
    /// pair-incidence signal to match full T13. Strictly weaker than `t13`
    /// — higher-weight pairs contribute zero rows here.
    pub t13_min: u128,
    /// True iff the low-weight builder bailed (full G(C) was used for wl_min).
    pub fallback: bool,
    pub rounds_min_vanilla: u32,
    pub rounds_min_degree: u32,
    pub rounds_full_vanilla: u32,
    pub rounds_full_degree: u32,
    pub l_min: u32,
    pub l_full: u32,
    /// Per-component wall time in nanoseconds (this single call's slice).
    /// Order:
    /// `[wl_min_vanilla, wl_min_degree, wl_full_vanilla, wl_full_degree,
    ///   t11_full, t11_gmin, t12, t13, t13_min]`. Caller accumulates
    /// across codes for per-component µs/code stats. Includes the digest
    /// hash; excludes the upfront codeword-set + bipartite-graph builds
    /// (those are shared across multiple components and would double-count).
    pub component_nanos: [u64; 9],
}

pub fn compute_all_invariants(
    rref: &[BinVec],
    n: u32,
    weights: &[u32],
    gmin_weights: &[u32],
) -> InvariantDigests {
    use std::time::Instant;

    let full_cws = full_codewords(rref, n);
    let full_g = build_full_bipartite(rref, n);

    let (gmin_g, fallback): (Bipartite, bool) =
        match build_low_weight_bipartite(rref, n) {
            Some(g) => (g, false),
            None => (build_full_bipartite(rref, n), true),
        };

    // Codewords actually used for the G_min T11 (full set when fallback fired).
    let gmin_cws: Vec<BinVec> = if fallback {
        full_cws.clone()
    } else {
        match low_weight_codewords_pub(rref, n) {
            Some((cws, _)) => cws,
            None => full_cws.clone(),
        }
    };

    let t0 = Instant::now();
    let (wl_min_van, rmv) = wl_refine(&gmin_g, InitMode::Vanilla);
    let d_mv = hash_sorted_u64(&wl_min_van);
    let ns_wl_min_van = t0.elapsed().as_nanos() as u64;

    let t0 = Instant::now();
    let (wl_min_deg, rmd) = wl_refine(&gmin_g, InitMode::DegreeAndWeight);
    let d_md = hash_sorted_u64(&wl_min_deg);
    let ns_wl_min_deg = t0.elapsed().as_nanos() as u64;

    let t0 = Instant::now();
    let (wl_full_van, rfv) = wl_refine(&full_g, InitMode::Vanilla);
    let d_fv = hash_sorted_u64(&wl_full_van);
    let ns_wl_full_van = t0.elapsed().as_nanos() as u64;

    let t0 = Instant::now();
    let (wl_full_deg, rfd) = wl_refine(&full_g, InitMode::DegreeAndWeight);
    let d_fd = hash_sorted_u64(&wl_full_deg);
    let ns_wl_full_deg = t0.elapsed().as_nanos() as u64;

    let t0 = Instant::now();
    let t11_full = t11_signature(&full_cws, n, weights);
    let d_t11f = hash_sorted_u128(&t11_full);
    let ns_t11_full = t0.elapsed().as_nanos() as u64;

    let t0 = Instant::now();
    let t11_gmin = t11_signature(&gmin_cws, n, gmin_weights);
    let d_t11g = hash_sorted_u128(&t11_gmin);
    let ns_t11_gmin = t0.elapsed().as_nanos() as u64;

    let t0 = Instant::now();
    let t12 = t12_signature(&full_cws);
    let d_t12 = hash_sorted_u64(&t12);
    let ns_t12 = t0.elapsed().as_nanos() as u64;

    let t0 = Instant::now();
    let t13 = t13_signature(&full_cws, n, weights);
    let d_t13 = hash_sorted_u64(&t13);
    let ns_t13 = t0.elapsed().as_nanos() as u64;

    // t13_min: same pair-gram, restricted to the minimum-weight stratum.
    // Doubly-even codes always have min weight in {4, 8, 12, ...}; for
    // the all-zero (k=0) corner case full_cws is empty and t13 returns
    // an empty multiset, which hashes deterministically.
    let t0 = Instant::now();
    let min_w: u32 = full_cws
        .iter()
        .filter(|c| **c != 0)
        .map(|c| c.count_ones())
        .min()
        .unwrap_or(0);
    let t13_min = t13_signature(&full_cws, n, &[min_w]);
    let d_t13_min = hash_sorted_u64(&t13_min);
    let ns_t13_min = t0.elapsed().as_nanos() as u64;

    InvariantDigests {
        wl_min_vanilla: d_mv,
        wl_min_degree: d_md,
        wl_full_vanilla: d_fv,
        wl_full_degree: d_fd,
        t11_full: d_t11f,
        t11_gmin: d_t11g,
        t12: d_t12,
        t13: d_t13,
        t13_min: d_t13_min,
        fallback,
        rounds_min_vanilla: rmv,
        rounds_min_degree: rmd,
        rounds_full_vanilla: rfv,
        rounds_full_degree: rfd,
        l_min: gmin_g.n_left,
        l_full: full_g.n_left,
        component_nanos: [
            ns_wl_min_van,
            ns_wl_min_deg,
            ns_wl_full_van,
            ns_wl_full_deg,
            ns_t11_full,
            ns_t11_gmin,
            ns_t12,
            ns_t13,
            ns_t13_min,
        ],
    }
}

// ───────────────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;

    fn rep_code_n8_k1() -> (Vec<BinVec>, u32) {
        // [8, 1] repetition code (the all-ones row). All-ones is wt-8 ≡ 0 mod 4.
        (vec![0xFFu64], 8)
    }

    #[test]
    fn full_codewords_n8_k1() {
        let (rref, n) = rep_code_n8_k1();
        let cws = full_codewords(&rref, n);
        assert_eq!(cws.len(), 1);
        assert_eq!(cws[0].count_ones(), 8);
    }

    #[test]
    fn full_bipartite_n8_k1() {
        let (rref, n) = rep_code_n8_k1();
        let g = build_full_bipartite(&rref, n);
        assert_eq!(g.n_left, 1);
        assert_eq!(g.n_right, 8);
        // The one codeword (all-ones) is connected to every column.
        assert_eq!(g.left_off, vec![0, 8]);
        assert_eq!(g.left_nbr, (0u32..8).collect::<Vec<_>>());
        // Every column has degree 1.
        for j in 0..8 {
            assert_eq!(g.right_off[j + 1] - g.right_off[j], 1);
            assert_eq!(g.right_nbr[g.right_off[j] as usize], 0);
        }
    }

    #[test]
    fn wl_signature_is_permutation_invariant() {
        // Build a small doubly-even code and verify that permuting columns
        // leaves the wl signature unchanged.
        // [4, 1] code with basis (1111): the single repetition row.
        let rref_a: Vec<BinVec> = vec![0b1111];
        // Same code under column perm — but a 1-row k=1 code is its own
        // permutation, so use a [6, 2] code instead.
        // Basis: e1 + e2 + e3 + e4 = 0b001111 = 15
        //        e3 + e4 + e5 + e6 = 0b111100 = 60
        let rref_b: Vec<BinVec> = vec![15, 60];
        let n = 6;
        let g1 = build_full_bipartite(&rref_b, n);
        let (sig1, _) = wl_refine(&g1, InitMode::DegreeAndWeight);

        // Apply a column permutation by re-ordering bits, then verify the
        // bipartite graph and WL sig change in step (or rather, same sig).
        let perm = [3u32, 0, 1, 5, 4, 2]; // arbitrary
        let permute = |v: BinVec| -> BinVec {
            let mut out = 0u64;
            for (old_bit, &new_bit) in perm.iter().enumerate() {
                if (v >> old_bit) & 1 != 0 {
                    out |= 1u64 << new_bit;
                }
            }
            out
        };
        let rref_p: Vec<BinVec> = rref_b.iter().copied().map(permute).collect();
        let g2 = build_full_bipartite(&rref_p, n);
        let (sig2, _) = wl_refine(&g2, InitMode::DegreeAndWeight);

        // Sigs must match — permutation equivalence.
        assert_eq!(sig1, sig2, "WL signature is not permutation invariant");

        // Use rref_a so it doesn't go unused in this test.
        let _ = rref_a;
    }

    #[test]
    fn t11_and_t13_pack_within_u64() {
        // Verify the packing keeps full information (no overflow): at k=13
        // counts can reach 4096. 4 weights × 16 bits = 64 bits → fits.
        let counts = [4096u32, 4095, 1, 4096];
        let mut packed: u64 = 0;
        for (i, &c) in counts.iter().enumerate() {
            packed |= ((c as u64) & 0xFFFF) << (16 * i);
        }
        let unpack: [u32; 4] = [
            (packed & 0xFFFF) as u32,
            ((packed >> 16) & 0xFFFF) as u32,
            ((packed >> 32) & 0xFFFF) as u32,
            ((packed >> 48) & 0xFFFF) as u32,
        ];
        assert_eq!(unpack, counts);
    }
}
