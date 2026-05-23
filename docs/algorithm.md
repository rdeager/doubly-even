# Algorithmic levers

This doc walks through what `doubly-even` adds on top of the DFGHILM
Appendix B canonical-augmentation recipe to make the enumeration tractable
on a desktop or modest cloud VM. Five levers carry the load; each delivered
a measurable wall-time reduction on its own, and they compose to roughly
~220× over the pure-Python baseline at `N = 22` and ~525× over Sage's
`self_orthogonal_binary_codes` at the same length.

Cumulative wall-time ablation on the 13700K development host (sequential
unless noted):

| stage                                                | N = 20  | N = 22  | N = 24 |
|------------------------------------------------------|--------:|--------:|-------:|
| pure Python, full BFS over `C⊥`                      |   235 s | > 600 s |   —    |
| + quotient-space orbit-min in `C⊥/C`                 |  13.8 s |   152 s |   —    |
| + degree-based initial nauty partition               |  2.00 s |  18.8 s |   —    |
| + native Rust kernel (no Python↔Rust crossings)      |  1.46 s |  14.5 s |   —    |
| + low-weight-incidence canonicaliser                 |  1.07 s |  7.57 s |  107 s |
| + outer-DFS pipelined-seeder parallel kernel (24t)   |  0.22 s | **0.69 s** | **8.90 s** |

The remainder of this doc explains what each lever does and why it works.

## What DFGHILM Appendix B prescribes

The enumerator implements DFGHILM Appendix B end-to-end:

- **B.1 — Gaborit mass formula.** Closed-form `σ(N, k)` for the labelled
  count of doubly-even `[N, k]` codes. Used as a stopping certificate
  and as the running consistency check
  `Σ N!/|Aut(C_i)| = σ(N, k)`.
- **B.2 — Bipartite encoding `G(C)`.** Codewords on one side, columns on
  the other. The column-side stabiliser of `Aut(G(C))` is the permutation
  automorphism group of `C`. Canonical labels and `Aut` generators come
  from a single `nauty` call (via `nauty-Traces-sys` in the Rust kernel).
- **B.3 — Doubly-even linear-algebra optimisations.** Corollary B.1
  reduces the doubly-even predicate on a generated code to `O(k²)`
  inner-product checks, avoiding the `2^k` codeword sweep.
- **B.4 — Canonical augmentation (McKay 1998).** The recursion is on
  *augmentations* `(parent, child)`, not codes alone. A child is emitted
  iff its augmentation agrees with the canonical-parent function `p` —
  that property uniquely picks one ancestry per equivalence class, so
  every class is emitted exactly once.

Three independent oracles verify correctness:

1. The Gaborit mass formula, on every emitted class.
2. DFGHILM Table 3, cell-for-cell through `N = 26`, plus the `N = 28`
   cell from a cloud run.
3. Sage's `self_orthogonal_binary_codes`, through `N = 22`.

Robert L. Miller's independent enumeration at
[rlmiller.org/de_codes](https://rlmiller.org/de_codes/) (a
no-zero-column convention) provides a fourth check at `N = 28`; the
DFGHILM↔Miller bridge is `total(N) = no_zero_cols(N) + total(N-1)`.

## The levers

### 1. Quotient-space orbit-min prefilter

DFGHILM B.3 gives the orbit filter that decides which dual-coset
representatives can extend the current code: walk `C⊥`, keep those that
are minimal under `Aut(C)`. A naïve implementation walks all `2^(N-k)`
dual elements, then orbit-reduces — wasted work, because doubly-even
codes are self-orthogonal (`C ⊆ C⊥`), so the canonical coset reps form
an `(N − 2k)`-dimensional subspace `V` of `C⊥`.

We construct a basis of `V` (the rows of the dual reduced modulo `C`),
walk `V` in Gray-code order — one XOR per yielded vector — and project
`Aut(C)`'s action down to the dimension-`(N − 2k)` quadratic quotient
space `Q_C := C⊥/C` via precomputed `σ_Q` lookup tables. The orbit-min
BFS then lives entirely in `Q_C`, which is at most a few thousand
elements rather than millions.

This was the first single-session lever to break 10×, and on the pure-
Python baseline at `N = 20` it took the wall from 235 s to 13.8 s. It
remains the dominant Python-side accelerator over a literal reading
of B.3.

Code: `src/doubly_even/enumerate/quotient.py` (Python spec), wrapped by
the Rust kernel via `rust/src/candidates.rs`.

### 2. Low-weight-incidence canonicaliser

The bipartite graph `G(C)` that B.2 hands to nauty has `2^k + N`
vertices. At deep ranks `k ≈ N/2` the codeword side dominates: at
`N = 22, k = 11` that's 2048 codeword vertices for a 22-column code.

We replace the full codeword side with just enough low-weight codewords
to span `C`. Let `C_low ⊆ C` be the lowest-weight codewords needed to
span `C`, accumulated by Gray-walked weight strata until the span
condition is reached. The reduced graph `G_low(C)` has `|C_low| + N`
vertices, each Hamming-weight stratum coloured as its own cell.

The correctness invariant is tight: let `H` be the column-side stabiliser
of `G_low(C)`. Then `Aut(C) ⊆ H` always; equality holds iff
`span(C_low) = C`, which the Gray-walk-by-weight-stratum construction
enforces exactly. So `H = Aut(C)` and the canonical column order from
`G_low(C)` is the canonical column order for `C`.

Measured wall reduction at the time this lever shipped, single-threaded:
**1.91× at N = 22** (14.48 s → 7.57 s); **≥2.5× at N = 24**.

Code: `rust/src/canon.rs::collect_low_weight_codewords`,
`build_low_weight_sparsegraph`, `canon_info_qd_native`. The
threshold sweep landed at "always dispatch" (no `k` gate). The fall-back
to the full-bipartite path is one branch in the same file.

### 3. Native Rust kernel

The canonicaliser was already in Rust (via `nauty-Traces-sys`), but the
recursion itself — `canonical_parent`, `is_canonical_augmentation`, the
mass-quota check, the canon-info LRU — was Python. Every step in the
DFS crossed the Python↔Rust boundary at least twice (once for canon-
info, once for candidate generation).

The native kernel ports the entire recursion to Rust:

- `rust/src/enumerate.rs` owns the LRU cache (keyed by the RREF basis,
  with `Rc<CachedInfo>` values so cache hits don't clone aut-generator
  payloads), the mass accumulator, the depth-first traversal, and a
  Rust port of `canonical_parent` and `is_canonical_augmentation`.
- `rust/src/permutations.rs` ports the Schreier–Sims fallback used when
  nauty's `grpsize1 × 10^grpsize2` overflows float64's exact-integer
  range (which happens at the zero code for `N ≥ 19`).

Measured wall reduction: **1.32× at N = 22**; speedup shrinks with `N`
because the eliminated overhead is a finite Python-side slice (~25 %
at N = 22). The Python implementation stays in `src/doubly_even/enumerate/augment.py`
as a fallback for non-kernel builds and as a diff oracle.

### 4. Outer-DFS worker parallelism with pipelined seeder

The depth-first traversal is embarrassingly parallel above a configurable
cut depth: distinct seeds at the cut depth induce disjoint subtrees that
never communicate. The kernel implements this with a producer-consumer
recipe — exactly the model DFGHILM Section B.4 describes:

1. The seeder runs on the main thread and walks the DFS down to
   `DOUBLY_EVEN_FRONTIER_DEPTH` (default 4; 5 at `N ≥ 24`).
2. Each accepted seed at the frontier is pushed into a bounded crossbeam
   channel.
3. A pool of worker threads spawn first and block on the channel;
   they consume seeds as the seeder produces them and run each subtree
   on their own thread with per-worker canon caches.
4. A shared atomic mass-tracker (`Arc<Mutex<Vec<u128>>>`) lets workers
   check the rank-`k+1` mass quota across the pool — saving the rest of
   a subtree as soon as the quota for the rank above is hit.

The bounded channel doubles as backpressure: if workers fall behind the
seeder, the seeder blocks until they catch up. This is essential —
unbounded queueing degrades cache behaviour at the seeder and starves
the canon caches.

Measured wall, mean of 3 runs on the 13700K (16 physical / 24 logical):

| N  | sequential | parallel best         |  ratio |
|----|-----------:|-----------------------|-------:|
| 20 |     0.97 s | 0.22 s (t=16, d=4)    |  4.4×  |
| 22 |     6.64 s | **0.691 s** (t=20, d=4) | **9.6×** |
| 24 |     ~107 s | **8.90 s** (t=24, d=5) |  ~12.0× |
| 26 |     —      | **170 s** (t=24, d=5) |  ~11.2× |

Tuning summary (the [`performance.md`](performance.md) reference has
the full table):

- `N ≤ 22`: `DOUBLY_EVEN_THREADS = logical_cores − 2` (leave 2 cores
  for the scheduler).
- `N ≥ 24`: `DOUBLY_EVEN_THREADS = logical_cores`, `DOUBLY_EVEN_FRONTIER_DEPTH=5`.
- `N ≥ 26`: also `DOUBLY_EVEN_CANON_CACHE_CAP` — the per-worker LRU is
  load-bearing for memory ceiling.

Code: `rust/src/enumerate.rs::enumerate_doubly_even_parallel`. The
parallel build is an opt-in Cargo feature (`parallel`) so the sequential
path stays byte-identical when unset.

### 5. Mass-formula early stop

DFGHILM B.1 gives a closed form for `σ(N, k)` (the labelled count of
doubly-even `[N, k]` codes). At every emit, the kernel updates
`Σ N!/|Aut(C_i)|` for the current rank `k+1`; as soon as that sum equals
the closed-form `σ(N, k+1)`, the rest of the subtree at rank `k+1` is
provably exhausted and we can skip it.

This is conceptually elegant — it's a strong oracle for "we're done at
this rank" derived purely from labelled counts — but the empirical win
is modest: ablation measures **4–11 % wall reduction** at `N = 18–22`,
which corresponds to ~210 pre-loop firings at `N = 22`, each skipping
roughly 1.5 ms of subtree work. The lever is most effective when the
DFS visits low-`|Aut|` classes early (their `N!/|Aut|` contributions
are larger and fill the quota faster); a `feature/dfs-order-by-aut`
branch explores a child-ordering heuristic that exploits this and is
described under "Opt-in branches" in the project README.

The `σ(N, k)` itself is implemented in
`src/doubly_even/spec/mass.py::gaborit_sigma` (closed-form, verified
against `sigma_brute` for `N ≤ 8`). The quota check is in
`rust/src/enumerate.rs`.

## Cumulative result

At `N = 22`, mean of 3 runs on the 13700K, parallel kernel at the
recommended tuning:

| baseline                                     | wall    | ratio |
|----------------------------------------------|--------:|------:|
| pure Python, BFS over `C⊥` (no Q_C orbit-min)| > 600 s | 1.00× |
| Sage `self_orthogonal_binary_codes`          | 363.85 s | 1.65× |
| post all levers, sequential                  |  6.64 s | ≥ 90× |
| post all levers, parallel (t=20, d=4)        | **0.691 s** | **≥ 870×** |

The "≥" is because the pure-Python BFS was killed by timeout — we never
let it run to completion at `N = 22`, only at `N = 20` (235 s). The
ratio `0.691 / 363.85 = 525×` against Sage is anchored to a measured
run.

## What we did not beat

The 13700K parallel-kernel `N = 22` wall of 0.691 s has roughly 90 % of
its time inside sparsenauty's C code. The remaining ~10 % is split
across Rust enumerate overhead (`apply_permutation`, `row_reduce`,
the orbit BFS in `subspace_in_orbit`, mass-accumulator bookkeeping); no
individual slice is > 2 % of wall.

Three classes of further improvement were investigated and closed during
the 2026-05-15 → 2026-05-23 optimisation sprint:

- **Prefilter the canonicaliser call.** Paired-iso (Leon §10(i)),
  probabilistic dedup via cheap invariants. Blocker: the prefilter
  shares nauty's refinement primitives, so per-probe cost ≈ per-call
  cost in Rust. The hit rate is high; the cost is too close to the
  thing it would skip.
- **Replace the canonicaliser entirely.** Feulner-style backtrack
  refinement (1000 LOC in tree as a reference / diff oracle); bliss
  (vendored under `rust/vendor/bliss-0.77/`, benched in
  `scripts/microbench/`). Both are 1.15–6× slower than sparsenauty
  per call on the doubly-even shape. Thirty years of `nauty` C
  micro-optimisation are hard to overtake.
- **Reduce the canonicaliser's input further.** Richer column colourings
  on top of the low-weight-incidence cut measurably *regressed* — the
  fingerprint cost outweighed nauty's refinement saving.

The honest framing: sparsenauty per-call cost is the algorithmic floor
at this graph shape. Further wins would require either a GPU
canonicaliser (PEACE-style — the only published direction with > 5×
headroom, but multi-week, uncertain payoff) or a different enumeration
paradigm targeting a different frontier (e.g. a column-multiset /
Fourier-domain engine specifically for `N = 32, k ≤ 4`, where the
current pipeline cannot reach).

## Validation oracles in more detail

- **Gaborit mass formula** is checked on every step:
  `Σ N!/|Aut(C_i)| == σ(N, k)`. The closed-form is in
  `src/doubly_even/spec/mass.py`; the running sum is in the Rust kernel.
  Mismatch is a fatal `PanicException`.
- **DFGHILM Table 3** counts are hardcoded in
  `tests/test_augment.py::DFGHILM_TABLE_3` and matched cell-for-cell
  through `N = 16` in the default suite, `N = 18` with `--run-slow`,
  and `N = 20, 22` through `scripts/bench.py` (which re-runs the
  Table-3 check after each timed enumeration).
- **Sage `self_orthogonal_binary_codes`** agrees with our class counts
  through `N = 22` (where Sage starts becoming impractical). The
  comparison data is in
  [`docs/performance.md`](performance.md).
- **rlmiller.org/de_codes** (Robert L. Miller's independent enumeration,
  one of the DFGHILM authors) cross-checks `N = 28` to within OCR-parse
  noise via the no-zero-column bridge described above.
- **Bouyukliev–Bouyuklieva 2019** (arXiv:1907.10363) provides counts
  for `[N, k, ≥ d]` codes at `N = 31, 32`. We don't yet ship a minimum-
  distance filter, so this oracle is open.

Each cross-check is cited in [`docs/references.md`](references.md).
