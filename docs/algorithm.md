# Algorithmic levers

This doc walks through what `doubly-even` adds on top of the DFGHILM
Appendix B canonical-augmentation recipe to make the enumeration tractable
on a desktop or modest cloud VM. Ten levers carry the load; each delivered
a measurable wall-time reduction on its own, and they compose to roughly
~640× over the pure-Python baseline at `N = 22` and ~1500× over Sage's
`self_orthogonal_binary_codes` at the same length.

Cumulative wall-time ablation on the 13700K development host (sequential
unless noted; rows from the coset-spectrum rule down are a
13700K-equivalent container, each a same-session A/B against the row
above it):

| stage                                                | N = 20  | N = 22  | N = 24 | N = 26 |
|------------------------------------------------------|--------:|--------:|-------:|-------:|
| pure Python, full BFS over `C⊥`                      |   235 s | > 600 s |   —    |   —    |
| + quotient-space orbit-min in `C⊥/C`                 |  13.8 s |   152 s |   —    |   —    |
| + degree-based initial nauty partition               |  2.00 s |  18.8 s |   —    |   —    |
| + native Rust kernel (no Python↔Rust crossings)      |  1.46 s |  14.5 s |   —    |   —    |
| + low-weight-incidence canonicaliser                 |  1.07 s |  7.57 s |  107 s |   —    |
| + outer-DFS pipelined-seeder parallel kernel (24t)   |  0.22 s |  0.69 s |  8.90 s | 170 s  |
| + coset-spectrum parent rule (parallel, same knobs)  | 0.053 s |  0.24 s |  2.60 s | 21.4 s |
| + split-frame φ sharing + one-comparison reject      | 0.051 s | 0.245 s |  1.67 s | 13.2 s |
| + pair-structure chain (O(1) later strata)           | 0.051 s |  0.24 s |  1.77 s | 11.5 s |
| + method-of-four-Russians orbit BFS                  | 0.051 s |  0.24 s |  1.7 s  | 9.8 s  |
| + x86-64-v3 codegen (and workspace split)            | 0.051 s |  0.24 s |  1.7 s  | 9.7 s  |
| + automorphism-only canonicalisation on accepts      | 0.051 s | **0.24 s** | **1.7 s** | **9.2 s** |

The parallel `N ≤ 24` cells stopped moving at the chain row (each
row's same-hour control re-measures within noise of the row above):
parallel `N ≤ 26` is bounded by the serial seeder, not by
per-candidate work, so the last three levers' wins are clearest
sequentially. The chain takes `N = 26` sequential from 126.6 to
97.2 s (1.30×, spectrum evaluation itself 3.78×), the four-Russians
orbit BFS to 85.6 s (1.13×; 1.17× at `N = 24`), the codegen flag
to 81.4 s (1.044× on a cool box), and automorphism-only
canonicalisation to **74.6 s** (1.092×; 1.097×/1.096× at
`N = 22`/`24`). The orbit BFS is the one post-chain lever that also
moves the parallel wall at `N = 26` substantially (11.5 → 9.8 s,
1.18×), because the serial seeder span it shortens is exactly the
parallel bottleneck; autom-only adds a further 1.035× there.

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

Validation hierarchy, by how far each oracle reaches:

1. **Gaborit mass formula** — fires on every emitted class at every
   `N`, including the new `N = 29` result. This is the load-bearing
   correctness oracle: `Σ N!/|Aut(C_i)| == σ(N, k)` exactly at every
   rank. Mismatch is a fatal kernel panic.
2. **DFGHILM Table 3** — cell-by-cell through `N = 26` in the
   default suite + bench script, plus the `N = 28` cells from the
   c4a-72 cloud run. **No published cells at `N = 29`**, so this
   oracle is silent there.
3. **Sage `self_orthogonal_binary_codes`** — cross-checked through
   `N = 22` (Sage becomes impractical beyond that).
4. **rlmiller.org/de_codes** — Robert L. Miller's independent
   enumeration (no-zero-column convention) provides a fourth check
   at `N = 28` via the bridge `total(N) = no_zero_cols(N) + total(N-1)`.

For `N = 29` specifically, only oracle 1 fires (and does, at every
rank `k = 0..13`). The certificate is at
[`docs/results/n29.json`](results/n29.json).

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

**Comparison with Sage.** A 2026-05-23 audit of Sage's
`self_orthogonal_binary_codes` (the implementation at
`sage/coding/binary_code.pyx`, Miller 2007) confirmed that Sage's
enumerator already uses the quotient-space coordinates
(`binary_code.pyx:4017`), a Gray-code walk through the orthogonal
complement (`binary_code.pyx:4123–4133`), a weight-mod-4 filter on
the lift (`binary_code.pyx:4016`), and a visited-set bitmap to skip
already-processed cosets (`binary_code.pyx:4001, 4018`). What is
specifically novel in our lever is the combination of (a) precomputed
`σ_Q ∈ End(Q_C)` action tables built incrementally in Gray-code
order at one XOR per entry per generator, and (b) a single global
O(2^L) sweep that decomposes all orbits at once via these tables —
replacing the per-candidate orbit-BFS through generators that Sage
runs at `binary_code.pyx:4109–4121`. That trades O(N) word-permutation
cost per orbit hop for an O(1) table lookup, at the price of
`|gens| × 2^L` memory per active code. Not present in DFGHILM B,
Bouyukliev–Bouyuklieva 2019, Sage `codecan` (Feulner port — operates
in full `F_q^n`), or Sage `binary_code.pyx`. Full audit at
[`/workspace/markdown/notes/qc-sage-audit-2026-05-23.md`](../../markdown/notes/qc-sage-audit-2026-05-23.md).

Code: `src/doubly_even/enumerate/quotient.py` (Python spec), wrapped by
the Rust kernel via `rust/core/src/candidates.rs`.

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

Code: `rust/core/src/canon.rs::collect_low_weight_codewords`,
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

- `rust/core/src/enumerate/` owns the LRU cache (keyed by the RREF basis,
  with `Rc<CachedInfo>` values so cache hits don't clone aut-generator
  payloads), the mass accumulator, the depth-first traversal, and a
  Rust port of `canonical_parent` and `is_canonical_augmentation`.
- `rust/core/src/permutations.rs` ports the Schreier–Sims fallback used when
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
- `N ≥ 24`: `DOUBLY_EVEN_THREADS = logical_cores`. (The
  `FRONTIER_DEPTH=5` advice from this lever's era is obsolete — the
  default `d = 4` is the measured best everywhere since the
  coset-spectrum rule.)
- `N ≥ 26`: also `DOUBLY_EVEN_CANON_CACHE_CAP` — the per-worker LRU is
  load-bearing for memory ceiling.

Code: `rust/core/src/enumerate/drivers.rs::enumerate_doubly_even_parallel`. The
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
`rust/core/src/enumerate/`.

### 6. Coset-spectrum parent rule

Formal statements and proofs: [`theory.md`](theory.md) §2–§3 (the rule,
its McKay soundness, the spectral calculus).

The biggest lever, and the only one that attacks canon **call count**
rather than per-call cost. Background: in McKay's canonical-augmentation
framework, each candidate child `D = ⟨C, v⟩` is accepted iff `C` is the
"canonical parent" of `D` — and the parent-selection function is *ours
to choose*, subject only to being isomorphism-invariant and selecting a
single `Aut(D)`-orbit of subcodes (McKay 1998; the canonical form is
needed only to break ties). The textbook choice — also DFGHILM's and
our original one — derives the parent from the canonical labeling, so
**every candidate pays a full nauty call before the test even starts**,
and at scale ~94–99.7 % of those calls reject the candidate. Worse,
profiling showed 93.4 % of those rejections (N = 22) were ultimately
decided by a 1.5 µs weight-enumerator comparison made *after* the
~75 µs canon call.

The fix is to select the parent by weight data directly. The parent of
a rank-`(k+1)` code `D` is one of its `2^(k+1) − 1` hyperplanes (index-2
subcodes `H_u`, one per nonzero functional `u`; every hyperplane of a
doubly-even code is doubly-even). Define the complement-coset weight
spectrum

```
φ_w(u) = #{ x ∈ D : wt(x) = w, u(x) = 1 },     φ(u) = (φ_4(u), φ_8(u), …)
```

and let the canonical parent be the `Aut(D)`-orbit of the hyperplane
with lexicographically minimal `φ` — ties broken exactly like the old
rule, but restricted to the argmin set. Both definitions are valid
McKay parent functions; they enumerate the same equivalence classes.

What makes φ cheap is that it is computable for **all hyperplanes at
once** without nauty: working in the frame basis `[C's rows, v]` (where
the candidate's own functional is the last coordinate), one Gray-code
sweep collects the weight of all `2^(k+1)` codewords, and then a
Walsh–Hadamard transform per weight stratum yields `φ_w(u)` for every
`u` simultaneously in `(k+1)·2^(k+1)` integer adds. The lex comparison
is evaluated lazily, stratum by stratum, shrinking the argmin set `M`
until one of three exits:

- the candidate's functional drops out of `M` → **reject, with no RREF,
  no cache probe, and no nauty call at all** (the common case:
  94–97 % of candidates at `N = 20–24`);
- `M` is exactly the candidate → **accept** (nauty is then called as
  before — the recursion genuinely needs `Aut(D)` and `|Aut(D)|`, so
  accepts cost what they always cost);
- the full spectra tie across several functionals (high-symmetry codes,
  e.g. extended Hamming) → nauty + the old σ-based tie-break restricted
  to the tied set, i.e. exactly the legacy cost path (~2 % of
  candidates).

The φ evaluation costs 1–4 µs per candidate against the ~30–75 µs canon
call it replaces, uses ~100 KB of grow-only thread-local scratch, and
involves no caching or cross-code state. Above a configurable child-rank
cap (default 13, only relevant at `N ≥ 30`) the kernel falls back to the
legacy rule per rank — sound, because rank is isomorphism-invariant and
the exactly-once argument is local to one child rank.

Measured same-session A/B (13700K-equivalent host, both rules, all
DFGHILM Table 3 cells verified on every run):

| N  | mode      | legacy   | coset-spectrum | wall ratio | canon calls     |
|----|-----------|---------:|---------------:|-----------:|----------------:|
| 22 | seq       |  6.48 s  |       0.85 s   |   **7.6×** | 78,750 → 5,248  |
| 24 | 24t, d=5  |  7.98 s  |       2.60 s   |   **3.1×** | 1.27 M → 39.5 K |
| 26 | 24t, d=5  | 168.3 s  |      26.4 s    |   **6.5×** | 45.5 M → 523 K  |

The canon-call reduction grows with `N` (15× → 32× → **87×**), tracking
exactly the calls-per-class explosion that motivated the lever, so it
strengthens toward the `N ≥ 28` frontier. Correctness is enforced at
run time by the mass-formula panic (any duplicate or missed class at
any rank aborts), and the test suite pins rule-equivalence on per-rank
class counts and `|Aut|` multisets at `N = 10–16`, audit-mode byte
identity, per-rank rule mixing, and parallel-vs-sequential agreement.

Knobs: `DOUBLY_EVEN_PARENT_RULE` = `coset-spectrum` (default) |
`legacy` (kill-switch / A-B control) | `audit` (instrumented legacy);
`DOUBLY_EVEN_PHI_MAX_RANK` (default 13).

Code: `rust/core/src/parent_rule.rs` (cascade + tie-break),
`rust/core/src/enumerate/worker.rs::test_candidate` (dispatch); Python prototype
`scripts/experimental/d15_phi_rule_check.py`; measurement harness
`scripts/experimental/d15_phi_audit.py`.

### 7. Split-frame spectrum sharing and the one-comparison reject

Formal statements and proofs: [`theory.md`](theory.md) §3–§4 (the
split-frame factorisation, the pair-max bound and its corollaries).

Once the coset-spectrum rule became the default, the φ evaluation
itself became the bottleneck — 60 % of the `N = 26` sequential wall,
~79 % of the projected `N = 29` core-hours, because the per-candidate
cost Θ((k+1)·2^(k+1)) multiplies an exploding candidate count. Two
exact refinements (decisions provably bit-identical — the audit
harness re-verifies the per-rank accept identity against the legacy
rule on every candidate of full N = 22/24 runs) cut φ by ~2.8×:

**Split-frame sharing.** Bit `k` of a frame coordinate splits the frame
into the *C-half* — the `2^k` codewords of the parent `C`, identical
for every sibling candidate — and the *v-half* (the coset `v + C`).
Everything C-half-dependent is computed once per parent and shared
across its siblings: the codeword table, weights, histogram, stratum
lists, and (lazily, per stratum) a half-size Walsh–Hadamard transform
`F̂_C`. The full-frame WHT then factors exactly as its last butterfly
stage, `f̂[(u′, a)] = F̂_C[u′] + (1−2a)·Ĝ_v[u′]`, so a candidate pays
only one `2^k`-point transform — and its weight sweep degenerates to a
branchless XOR + popcount over the shared table, with no Gray-code
serial dependency.

**The one-comparison reject.** For any `u′ ≠ 0`, the better of the
functional pair `(u′, 0)/(u′, 1)` scores `F̂_C[u′] + |Ĝ_v[u′]| ≥
F̂_C[u′]`. So the per-parent constant `amax = max_{u′≠0} F̂_C[u′]`
yields a sufficient reject test that needs **no per-candidate spectral
work at all**: if `amax > f̂[u_C]` — and `f̂[u_C]` is free from the two
weight histograms — some hyperplane provably beats the candidate's own.
At `N = 24–26` this single integer comparison (plus two histogram-only
fast paths for strata lying entirely in one half) decides ~99 % of
first strata.

A companion change parallelises the seeder's candidate generation in
the multi-threaded kernel: the low-rank orbit-min BFS and Gray sweep
fan out level-by-level onto a short-lived helper pool with exactly-once
claiming on an atomic bitset (output provably identical to sequential —
orbit closures are schedule-independent). The pool engages only on the
large early calls (quotient dimension ≥ 22 by default) where the worker
pool is still idle; pooling smaller calls measurably loses to
helper-vs-worker contention.

Measured same-session A/B against the coset-spectrum baseline (mean of
3, all DFGHILM Table 3 cells verified per run): `N = 24` sequential
9.34 → 7.80 s; `N = 26` sequential 207.1 → **126.6 s** (1.64×);
`N = 26` parallel (t=24, d=4) 21.4 → **13.2 s** (1.62×). With lever 8
on top, the count-anchored `N = 29` forecast drops to **1.0–1.5 h** on
the same 72-core cloud box that took 12.32 h pre-coset-spectrum.

Knobs: `DOUBLY_EVEN_SEEDER_THREADS` (helper-pool size; default =
worker count, `0`/`1` disables), `DOUBLY_EVEN_SEEDER_PAR_MIN_L`
(default 22). Code: `rust/core/src/parent_rule.rs` (`PhiParentCtx`, the
amax bound), `rust/core/src/seeder_pool.rs`, pooled variants in
`rust/core/src/orbit.rs` / `rust/core/src/candidates.rs`.

### 8. The pair-structure chain: O(1) later strata

Formal statements and proofs: [`theory.md`](theory.md) §5 (the E-set
chain invariant and its three decision arms).

After the one-comparison reject, a fresh sub-phase profile showed
**76 %** of the remaining spectrum-evaluation time at `N = 26` sat in
*later* strata — the per-candidate transforms and parity counts paid by
candidates that survive their first stratum. Almost all of those
candidates enter stratum 2 by the same door: a first stratum lying
entirely in the parent half (`tv = 0`), which leaves the running argmin
set in the special form `E ∪ {u_C} ∪ (u_C + E)` — every functional
`u′ ∈ E` present as its full pair `(u′, 0), (u′, 1)`.

That **pair structure** is an invariant: any further parent-only
stratum filters both halves of a pair together (the candidate-half
spectrum is identically zero there), so the structure survives. And
while it holds, every stratum decides in O(1) per candidate from
parent-only data:

- a stratum with no parent-half words rejects outright — the better
  half of any pair scores `|Ĝ_v[u′]| ≥ 0`, while the candidate's own
  functional scores `−tv < 0`;
- a mixed stratum applies the one-comparison reject *restricted to the
  running E-set*: a per-parent bound `max_{u′∈E} F̂_C[u′]` beating
  `tc − tv` proves some hyperplane wins; an inconclusive bound falls
  back to the generic machinery at exactly the same stratum it would
  have reached anyway;
- a parent-only stratum shrinks `E` by a parent-side filter; an empty
  result proves the candidate's own hyperplane is the unique argmin —
  accept.

Every sibling candidate of one parent walks the parent's strata in the
same ascending order, so the per-position E-sets and bounds form a
single lazy per-parent **chain**, built once and read O(1) by all later
siblings. The argmin set is never materialised while on the chain, so
the per-candidate stratum transforms and the per-candidate counting
sort disappear with it.

The chain is the same argmin cascade evaluated against cached
parent-side data — decisions are provably identical, and the A/B run
confirms it the hard way: class counts, canonicalisation-call counts
and even the total number of strata visited are bit-equal to the
control at every benched `N`. The audit harness re-verifies the
per-rank accept identity against the legacy rule on full `N = 22` / `24`
runs, mass-stop on and off.

Measured same-session A/B (median of 3, all DFGHILM Table 3 cells
verified per run): the chain decides **34 / 41 / 43 / 38 %** of *all*
candidates at `N = 22 / 24 / 26 / 27` in O(1) at stratum ≥ 2; spectrum
evaluation drops 3.78× at `N = 26` (43.7 → 11.6 s of a 126.6 s
sequential wall → **97.2 s**, 1.30×) and 4.62× at `N = 27`, where the
*parallel* wall drops 94.1 → **66.1 s** (1.42× — large enough that a
24-thread desktop beats the pre-parent-rule 72-core cloud `N = 27` row
5.7×); `N = 24` sequential 7.75 → 7.52 s. What remains of the spectrum
evaluation is ~67 % the unconditional XOR + popcount sweep over the
shared codeword table — a pure SIMD shape. The vectorisation question
that raised was settled afterwards by compiler codegen, not hand
intrinsics — see lever 9's coda.

No knobs (exact, bit-identical; `DOUBLY_EVEN_PARENT_RULE=legacy`
bypasses the whole spectrum rule if ever needed). Code:
`rust/core/src/parent_rule.rs` (`ensure_chain`, the chain arm of the
cascade); deterministic-witness and brute-force-sweep tests in the
same file.

### 9. Method-of-four-Russians orbit BFS

Formal statements and proofs: [`theory.md`](theory.md) §6 (linearity
of the byte-table decomposition; BFS schedule-independence).

With the spectrum cascade tamed, quotient-space candidate generation
was the next-largest consumer (~41 % of the `N = 26` sequential wall,
89 % of it the orbit-min BFS that closes each candidate orbit under
the `σ_Q` generator matrices). A focused profile — replaying 446
rank-2/3 parents dumped from real `N = 26 / 27` runs, not synthetic
inputs — showed the BFS is **compute-bound on image generation**, not
memory-bound: the seen-bitset is L2/L3-resident at quotient dimensions
`L = 20–23`, the singular set is group-invariant so the closure visits
all ~`2^(L−2)` representatives, and the orbits are few and giant.
Probe-side restructures (deferred probing, batched lookups,
radix-bucketed frontiers) all failed a 1.15× ship bar; the image
computation itself was the cost.

The fix is the classic method of four Russians. For each generator
matrix, precompute 256-entry byte tables `tables[j][b] = σ·(b ≪ 8j)`
for each of the `⌈L/8⌉` byte positions; by linearity, one image is
then `⌈L/8⌉` L1 table loads + XORs instead of a chained per-set-bit
walk. The BFS applies tables generator-major over 1024-element
frontier chunks, so one generator's ≤ 8 KB of tables stays L1-resident
across the whole chunk. The table build amortises whenever the
universe is large; below the crossover the original walk is kept.

Measured: **1.84–1.94×** on the BFS itself on the real-input replay;
on the wall, vs the pair-structure-chain epoch: `N = 22` sequential
0.796 → 0.698 s (1.14×), `N = 24` 7.52 → 6.44 s (1.17×), `N = 26`
97.2 → 85.6 s (1.13×), and `N = 26` *parallel* 11.5 → 9.81 s (1.18× —
the low-rank BFS lives in the serial seeder span, which is the
parallel bottleneck). `N = 27` parallel is flat: that wall is
worker-bound, not seeder-bound, so a seeder-side lever cannot move it.

Exactness is structural: the byte-table image is the same linear map
(linearity), and the per-level new-element set of the BFS is
independent of the order in which images are generated or probed, so
the emitted orbit minima are **byte-identical** to the original walk's
([`theory.md`](theory.md) §6, Lemmas 7–8). The A/B run confirms
classes, canon-call counts and strata sums bit-equal to the control.

No knobs (exact; the table/walk crossover is the `M4R_MIN_L = 14`
const). Code: `rust/core/src/orbit.rs` (`m4r_build`,
`orbit_minima_m4r`); the sequential and pooled BFS bodies share it.

**Codegen coda.** The vectorisation pass that lever 8 left on the
table was made moot by a one-line compiler flag:
`-C target-cpu=x86-64-v3` (in `rust/.cargo/config.toml`) lets LLVM
auto-vectorise the v-half popcount sweep (`vpshufb`/`vpsadbw`), the
Gray sweeps and the WHT butterflies, worth 1.01–1.05× sequential on a
cool box with decisions bit-identical — and hand intrinsics beyond
the flag measured dead. (The flag was measured as noise in 2026-05;
that was true while ~90 % of the wall was nauty's C, and reversed
once the parent-rule levers made Rust-side code ~50 % of the wall.)
x86 wheels consequently require AVX2 — any x86 CPU since ~2013;
aarch64 builds are unaffected (NEON is baseline and already
auto-vectorised). The live ranked-lever list is
[`bottlenecks.md`](bottlenecks.md).

### 10. Automorphism-only canonicalisation on accepts

With the parent rule deciding rejects invariant-first (lever 6), the
canon call count sits at its floor: one call per emitted class (the
recursion needs `Aut(D)` and `|Aut(D)|` per class for the mass
certificate and the next level's candidate generation) plus the ~6 %
of calls that break invariant ties. The remaining canon lever is
therefore not *fewer* calls but *cheaper requested output* — and nauty
decomposes its work accordingly: the automorphism-group search and the
canonical-labelling ("best leaf") bookkeeping are separable, with the
canonical pass measured at 19–25 % of the sparsenauty call on our
graph shapes.

The decision architecture makes the split free to exploit: the parent
rule's outcome is known *before* the canonicaliser is invoked, so the
kernel knows at call time whether the canonical labelling will be
consumed. Only tie-breaks (and the legacy-rule path) read it; on the
~80 % of calls that are unique-accepts, nauty now runs with
`getcanon = FALSE` — same generators, same group order, same column
orbits, no labelling. A cache entry written without a labelling that
is later hit by a tie-break on the same subspace is transparently
recomputed in full (measured: never fired in any benched run).

Measured (same-session knob A/B, median of 3): `N = 22` sequential
0.683 → 0.623 s, `N = 24` 6.27 → 5.72 s, `N = 26` 81.5 → **74.6 s** —
**1.092–1.097×** across the board — and `N = 26` parallel
9.53 → 9.21 s (1.035×; the parallel wall is seeder-bound). Decisions
are bit-identical: classes, per-rank counts, canon-call counts and
strata sums all gate-equal; the emitted search tree cannot move
because no decision ever read the labelling on these calls. Expected
to matter more on cloud runs at `N ≥ 28`, where canonicalisation is
~68 % of forecast core-hours.

One empirical footnote: in roughly one call in 4×10⁵, nauty without
the best-leaf bookkeeping reports a *different generating set* of the
same automorphism group. This is decision-neutral — orbit computations
are generating-set independent — but it is why the A/B gate compares
group-level counters rather than generator counts.

Knob: `DOUBLY_EVEN_CANON_LABELLING=full` restores the labelling on
every call (and per-class `canonical_column_order` in the in-memory
output; the streaming format never carried it). Code:
`rust/core/src/canon.rs`, `qd_graph.rs` (the `get_canon` flag),
`rust/core/src/enumerate/cache.rs` (label modes, upgrade-on-hit).

## Cumulative result

At `N = 22`, mean of 3 runs, parallel kernel at the recommended tuning
(rows above the last: 13700K; last row: 13700K-equivalent container,
same-session A/B against the legacy rule):

| baseline                                     | wall    | ratio |
|----------------------------------------------|--------:|------:|
| pure Python, BFS over `C⊥` (no Q_C orbit-min)| > 600 s | 1.00× |
| Sage `self_orthogonal_binary_codes`          | 363.85 s | 1.65× |
| legacy parent rule, sequential               |  6.64 s | ≥ 90× |
| legacy parent rule, parallel (t=20, d=4)     |  0.691 s | ≥ 870× |
| coset-spectrum rule, sequential              |  0.848 s | ≥ 700× |
| coset-spectrum rule, parallel (t=24, d=4)    | **0.237 s** | **≥ 2500×** |

The "≥" is because the pure-Python BFS was killed by timeout — we never
let it run to completion at `N = 22`, only at `N = 20` (235 s). The
ratio `0.237 / 363.85 ≈ 1535×` against Sage is anchored to a measured
run. The three levers shipped since (split-frame sharing, the
pair-structure chain, the four-Russians orbit BFS) plus the codegen
flag leave the `N = 22` parallel wall essentially unchanged at 0.24 s —
they bind at `N ≥ 24`, where the ablation table at the top of this doc
tracks them — though `N = 22` *sequential* did drop 0.85 → 0.69 s.

## What we beat last, and what is left

Until 2026-06 the framing was: ~90 % of the parallel `N = 22` wall sat
inside sparsenauty's C code, per-call cost was already at its
algorithmic floor, and at `N = 29` the kernel made ~**364 canon calls
per emitted class** (15.6 at `N = 22`, growing ~2.6× per 2-step — see
[`performance.md`](performance.md#the-scaling-story-per-call-cost-drops-calls-per-class-explodes)).
The conclusion — *a real >2× lever must cut the number of canon calls
per class, not the cost per call* — is exactly what the coset-spectrum
parent rule (lever 6) then did: ~94–97 % of candidates now resolve
without any canon call, and the per-class call multiplicity collapses
(87× fewer calls at `N = 26`).

What remains after that lever and the three passes that followed it
(levers 7–9), at `N = 26` sequential: canon ~50 %, quotient-space
candidate generation ~33 %, spectrum evaluation ~13.5 %. Canon is the
largest consumer again, but nothing dominates outright — future levers
have to re-profile first. The spectrum evaluation is at its popcount
floor: the x86-64-v3 codegen flag banked what vectorisation had to
offer (lever 9's coda), and hand SIMD beyond it measured dead. The
parallel kernel's frontier at `N ≤ 26` is the serial seeder span, not
per-candidate work. The live ranked-lever list is
[`bottlenecks.md`](bottlenecks.md).

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
  Mismatch is a fatal `PanicException`. This is the universal oracle —
  it fires for every `N`, every `k`, every emitted class, and is the
  only oracle that fires at `N = 29` (no published table exists).
- **DFGHILM Table 3** counts are hardcoded in
  `tests/test_augment.py::DFGHILM_TABLE_3` and matched cell-for-cell
  through `N = 16` in the default suite, `N = 18` with `--run-slow`,
  and `N = 20, 22` through `scripts/bench.py` (which re-runs the
  Table-3 check after each timed enumeration). For `N = 24, 26, 28`
  cell-by-cell agreement is verified by `scripts/merge_stream.py` on
  the cloud-run output; the `dfghilm_table3_ok` flag in
  [`docs/results/n29.json`](results/n29.json) is degenerate-True at
  `N = 29` because the cross-check has no published cells to fail.
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
