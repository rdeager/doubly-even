# Project context for Claude

You're working on the **`doubly-even`** package — an enumerator of doubly
even binary linear codes `[N, k]` up to permutation equivalence. The
classification underlies the Adinkra chromotopology problem from
supersymmetric representation theory (Doran–Faux–Gates–Hübsch–Iga–
Landweber–Miller, henceforth **DFGHILM**, whose Appendix B is the
algorithmic spec we implement).

The repo is one of three peer directories on the host:

- `/workspace/src/` (this repo) — Python code, `uv`-managed.
- `/workspace/markdown/` — design documentation. Read
  `markdown/README.md` first.
- `/workspace/inbox/mathpix/` — papers, Mathpix-converted to markdown.
  `DFGHILM_ATMP.md` is the primary spec; Appendix B is the recipe.

## Quick orientation

The package is split into three layers; depend only on the layers above.

```
spec/        readable executable spec (math, no perf work)
  vectors.py     BinVec=int with bit ops, wt, dot, polarisation
  codes.py       Code dataclass (n, basis), rref, dual, contains, extend
  doubly_even.py is_doubly_even via Corollary B.1; augmentation predicate
  mass.py        sigma_brute (works), gaborit_sigma (closed form)
  direct_sum.py  C1 ⊕ C2 operator (BHM2012 mass-seeding scaffolding)

canon/       canonical labels + Aut(C); active backend is the Rust kernel
  bipartite.py   G(C) bipartite encoding (codewords × columns)
  nauty.py       canon_info, canonical_form, are_equivalent — dispatches
                  to doubly_even_kernel when available (Rust + sparsenauty
                  via nauty-Traces-sys); pynauty kept as Python fallback
  permutations.py hand-rolled Schreier-Sims for exact |Aut| when nauty's
                  float grpsize would lose precision past ~2^53
  _linalg_f2.py  Spine GL(L, F_2) primitives (Mat, mat_apply, mat_mul,
                  mat_identity) consumed by enumerate/quotient.py
  experimental/  Dormant canonicalisers — see /workspace/src/EXPERIMENTAL.md

enumerate/   the canonical-augmentation search
  filters.py     Hot dispatcher (doubly_even_candidates) + F_2^N reference
                  oracle (cross-check only); two-half layout, section
                  banner inside.
  quotient.py    Q_C-coordinate orbit-min with σ_Q lookup tables (the
                  hot path).
  augment.py     canonical_parent(D); is_canonical_augmentation;
                  enumerate_doubly_even(N) -> EnumeratedCode iter.
                  Dispatches the whole recursion to the Rust kernel
                  via _kernel.enumerate_doubly_even (D11) when loaded.
  seeds.py       Direct-sum mass-seeding scaffolding (BHM2012); not
                  yet wired into active dispatch.
  experimental/  Witt phase-(b) scaffolding — see EXPERIMENTAL.md

rust/        Rust kernel (doubly_even_kernel), built with maturin
  src/canon.rs       Q_D-graph low-weight-incidence canonicaliser (D10)
                      + native sparsenauty path; default dispatch.
  src/enumerate.rs   Native enumerate_doubly_even recursion (D11).
                      EXPERIMENTAL banners on the modules below.
  src/feulner.rs     Feulner column-side canonicaliser (D9, ~1000 LOC,
                      reference / diff oracle; dispatch closed).
  src/feulner_clb.rs Jerrum CLB + Lemma 5.9 (Feulner §5.2 substrate).
  src/paired_iso.rs  Leon §10(i) verifier (D12, dormant under
                      `equivalence_verifier` feature).
  src/invariants.rs  WL + T11/T12/T13 collision-experiment substrate.
  Cargo.toml          Feature flags (all default OFF): parallel,
                      equivalence_verifier, dense_qd, dense_qd_tc0,
                      dense_qd_refinvar, traces_qd, nauty_hist.
```

## Quarantine layout

Dormant / experimental / audit-substrate code is kept in tree under
`canon/experimental/`, `enumerate/experimental/`, and
`scripts/experimental/`. **Quarantine, don't delete** — every
closed optimisation direction has its code preserved so a future
contributor can revisit it. Top-level index:
`/workspace/src/EXPERIMENTAL.md`.

## Conventions

- **`uv` for everything.** `uv add`, `uv run pytest`, `uv run dec`.
  No `pip install` outside of `uv`. No `requirements.txt`.
- **Binary vectors are Python `int`s.** Bit `i` is component `i`. XOR is
  addition, `int.bit_count()` is weight. Don't wrap them in a class.
- **Codes are immutable.** `Code(n, basis)` is frozen; derived data
  (RREF, dual) is recomputed on demand. If you need to cache derived
  data, use an external cache, not mutation.
- **`spec/` is the reference.** Every `spec/` function should be readable
  by someone who knows GF(2) linear algebra. Optimise elsewhere if you
  need to.
- **Tests cite their oracle.** Each `(N, k)` check names what it's
  matching (DFGHILM Table 3, `sigma_brute`, AGL(3, 2), etc.).
- **Slow tests are marked.** `@pytest.mark.slow` is skipped by default;
  enable with `uv run pytest --run-slow`.

## Validation oracles

Three independent checks the test suite relies on:

1. **Mass formula:** `Σ N!/|Aut(C_i)|` over emitted classes must equal
   `sigma_brute(N, k)`. Verified for `N ≤ 8`. This is internal
   consistency.
2. **DFGHILM Table 3:** published equivalence-class counts of doubly
   even `[N, k]` codes. The enumerator matches every cell exactly
   through `N = 16` in the default suite, `N = 18` with `--run-slow`,
   and `N = 20, 22` via `scripts/bench.py` (which runs the same
   Table-3 check after each timed enumeration).
   Hardcoded in `tests/test_augment.py::DFGHILM_TABLE_3`.
3. **Bouyukliev–Bouyuklieva 2019** (`inbox/mathpix/1907.10363v1.md`)
   gives counts for `[N, k, ≥ d]` codes at `N = 31, 32`. Not yet wired
   into tests — we don't have a minimum-distance filter.

## Ruled-out approaches (do not propose again)

- **T11 class-fingerprint cache (D14 V1 / V2).** Tried and removed
  2026-05-18. The approach used a permutation-invariant hash
  (`compute_t11_hash`, per-column weight multiset) as a cache key with
  cheap-rejects on class mismatch, made sound at N ≤ 24 by per-N
  precomputed blocklists of T11 multiset collisions. The blocklists
  presuppose a full enumeration to generate, so the approach **cannot
  scale to N ≥ 26** under a single-pass constraint. Audit
  (`scripts/experimental/pair_gram_class_audit.py`, in tree as a standalone) showed
  every cheap class invariant tested has tuple collisions that grow
  rapidly with N (t11_stacked_w4cubic: 2 at N=22, 137 at N=24,
  thousands extrapolated for N=26). At N=32 the "cheap" invariants
  themselves cost ~ms (comparable to nauty), eliminating the cache
  benefit. **Reverted** in a single revert chain (formerly commits
  `9625b4b`, `56f3466`, `3393f63`, `270f4f0`, `b3d9a98`, `62b2174`).
  Do not propose any variant — stored-tuple verifier, two-pass
  protocol, pair-gram-on-every-hit, "shared concurrent V3", etc.
  The single-pass + mass-formula-verified constraint
  ([[no-offline-blocklist]] in user memory) closes the entire family.

## Open / parked items

- `/workspace/markdown/` is not under git. If we want it versioned,
  either move it under `src/markdown/` or `git init` `/workspace/`.
- Bouyukliev–Bouyuklieva 2019 has `[N, k, ≥ d]` validation counts at
  `N = 31, 32`. Not yet wired into tests — we don't have a
  minimum-distance filter.
- **D12 paired-iso verifier shipped dormant.** Behind
  `cargo feature equivalence_verifier`, default OFF; per-probe cost ties
  nauty in Rust so net −29 % at `N = 18`. Recovery needs a flat-array
  Feulner refactor + static cf refinement tree to drop per-probe
  to ~40 µs. See `architecture/04-optimisations.md` §D12 and
  `architecture/05-retrospective.md`.
- **Phase (c) cheap-invariant prefilter (Q_C-recursion lever)** is
  formally open but now classified as a **blocked-prefilter category**
  alongside D12: every prefilter we have measured shares nauty's
  refinement primitives so per-probe cost ≈ per-call cost. Not
  expected to break 2× without a per-probe-cost unlock.
- **Engine A (§5 column-multiset / Fourier-domain) Rust port** is
  the unblocked next deliverable — Python prototype exists in
  `scripts/experimental/multiset_*.py`,
  `scripts/experimental/collision_experiment.py`. Target:
  `N = 32, k ≤ 4` frontier (the existing pipeline dies at `N ≥ 28`).
  Does not speed up `N ≤ 22`.
- **GPU canonicaliser (PEACE-style)** is the only published
  direction with > 5× headroom at `N ≤ 22`. Multi-week.
- **Audit 2026-05-17** (kernel instrumentation in `rust/src/enumerate.rs`
  per_k_stats matrix, scripts `experimental/bfs_rejects_measurement.py`,
  `experimental/cubic_tensor_experiment.py`,
  `experimental/mass_check_ablation.py`):
  - σ_Q + weight-enum prefilter is near-perfect — final BFS rejects
    only 0.4 % of survivors at `N = 22` (308 / 82 413). The canon test
    is mostly confirmatory; "direct canonical generation" lever is
    closed.
  - Cubic-tensor (column-triple-degree) is the strongest cheap
    invariant found — cuts T1 residual collisions by 37 % at `N = 22`
    (1918 vs 3069). But 95 µs Python / ~25 µs projected Rust falls
    into the D12 per-probe-cost trap; not ported.
  - Mass-stop is a 4–11 % win (ablation-measured: `N = 18` 10.9 %,
    `N = 22` 4.4 %). Each of 210 pre-loop firings at `N = 22` skips a
    whole subtree (~1.5 ms each). Code kept; could ~2× if we add
    low-|Aut|-first tree ordering.

## Performance state (post-D10/D11/D12 sprint)

See `/workspace/markdown/architecture/04-optimisations.md` for the
per-lever write-up and `architecture/05-retrospective.md` for the
sign-off retrospective.

Headline: **~129× cumulative at `N = 22`** since the pre-kernel
D6 baseline (152 s → 1.18 s with 16t, depth=4); **> 500×** since
the pre-Q_C original baseline (> 600 s → 1.18 s at `N = 22`).
Parallel kernel: `N = 20` in 0.22 s; `N = 22` in 1.18 s; `N = 24`
in 12.3 s; **`N = 26` in 218 s** (494 K classes, all 12 non-trivial
DFGHILM cells match exactly). Sequential at `N = 22` is 6.87 s.
We beat Sage's `self_orthogonal_binary_codes` by **~308× end-to-end
at `N = 22`** (Sage 363.85 s, single-threaded; parallelising Sage
would be weeks of Cython surgery).

Scaling forecast for `N ≥ 28` lives at
`/workspace/markdown/architecture/06-scaling-frontier.md`: N=28
reachable today (~1 hr at 20 threads), N=29 needs the streaming-
output refactor (1–2 days), N=30 needs streaming + ≥256 GB RAM or
a small cluster.

Cumulative levers, oldest first:

1. **D2** Quotient-space candidate enumeration (DFGHILM B.3).
2. **D3** Two-tier canon cache (`rust/src/enumerate.rs:111, 157`):
   primary LRU keyed by RREF (capacity tunable via
   `DOUBLY_EVEN_CANON_CACHE_CAP`, default 1M) + secondary
   weight-enumerator-keyed bucket cache. **The cap is load-bearing
   at N ≥ 26** — without it, N=26 OOMs at ~500K cached entries per
   worker × 20 workers. Commit `fd530cb`.
3. **D4** Weight-enumerator prefilter on the orbit BFS.
4. **D5a/b** Verified closed-form `gaborit_sigma` + mass-stop
   shortcut in the recursion.
5. **D6** `Q_C`-coordinate orbit-min with σ_Q lookup tables.
6. **D7** Witt-structured orbit infrastructure (kept; not active).
7. **D8** Degree-based initial vertex partition for nauty.
8. **D9** Feulner Rust+Python canonicaliser (reference, 6× slower
   than nauty per call; substrate for D12).
9. **D10** Q_D-graph low-weight-incidence canonicaliser
   (**1.91× at `N = 22`**, active default).
10. **D11** Native Rust `enumerate_doubly_even` + incremental Feulner
    (**1.32× at `N = 22`**; eliminates Python ↔ Rust crossings).
11. **D12** Paired-iso (Leon §10(i)) prefilter — shipped **dormant**.
12. **Q6 sparsenauty audit** (2026-05-17) — full `expert-review/
    05-nauty-traces-audit.md` §5 priority list executed:
    `schreier = TRUE` (+5 %), densenauty (+ 39 / + 42 / + 260 %),
    Traces (+ 2 %) all regress (all Δ are wall-time slower than
    baseline 6.97 s). `78 µs/call` is the sparsenauty
    algorithmic floor at this graph shape. Phase 0 `statsblk`
    counters (`numnodes`, `tctotal`, `maxlevel`, `numgenerators`)
    added to the kernel for future audits; four dormant Cargo
    features (`dense_qd`, `dense_qd_tc0`, `dense_qd_refinvar`,
    `traces_qd`) kept as reproducibility substrate.
13. **D13 Outer-DFS parallelism** (2026-05-18) — sequential traversal
    to a configurable cut depth, then a crossbeam-channel worker
    pool runs each accepted subtree on its own thread. Sparsenauty
    is parallel-safe under the `tls` feature on `nauty-Traces-sys`
    (already on; `USE_TLS` → `_Thread_local` on mutable globals).
    Per-worker canon caches; per-worker mass-stop disabled
    (loses 4–11 % vs sequential; gained outright by parallelism).
    V1 (depth=3): 3.1× at N=22. **V2 (depth=4, the new default):
    5.8× at N=22** (6.87 s → 1.18 s, 16 threads on 13700K); 4.4×
    at N=20. **8.7× at N=24** (107 s → 12.3 s, 20 threads,
    depth=5). The deeper frontier breaks the tail-task ceiling:
    at depth=3 the heaviest of 83 subtrees held ~30 % of total
    work; at depth=4 the ~300 tasks divide that into 3–5 pieces.
    Behind off-by-default Cargo feature `parallel`; enabled at
    runtime via `DOUBLY_EVEN_THREADS` env var (and tunable via
    `DOUBLY_EVEN_FRONTIER_DEPTH`, default 4) or `num_threads=`
    kwarg on the kernel entry. Conflicts with `traces_qd` (Traces
    uses non-TLS static work queues) — guarded by `compile_error!`.
    Determinism harness in `rust/tests/parallel_determinism.rs`.

`bench.py` lives in `scripts/`; per-step JSON records are in
`scripts/bench-results/` (gitignored). Rust microbench for
sparsenauty internals: `scripts/microbench/src/nauty_decomp.rs`.

**The `N ≤ 22` frontier is saturated at the pure-algorithmic level**
with this canonicaliser — see `architecture/05-retrospective.md` for
the failure-mode analysis of recent 2× attempts and the recommended
next-direction decision (push the `N ≥ 28` frontier via Engine A; or
multi-week GPU port; or ship-what-we-have). D13 (outer-DFS
parallelism, 2026-05-18) sits *orthogonally* to that statement: it
buys ~3× wall-time via infrastructure (worker pool) without changing
the algorithmic ceiling.

## D13 quickstart

Enable in the wheel build:

```sh
maturin build --release --features parallel -m rust/Cargo.toml \
  && uv pip install rust/target/wheels/doubly_even_kernel-*.whl
```

At runtime, set `DOUBLY_EVEN_THREADS` (16 on a 13700K is the sweet
spot for N ≤ 22; 20 at N ≥ 24). `DOUBLY_EVEN_FRONTIER_DEPTH`
defaults to 4 and rarely needs tweaking — at N ≥ 24 a value of 5
is better because the tree is bigger and the deeper split gives
finer load balance.

For `N ≥ 26`, also cap the per-worker canon cache via
`DOUBLY_EVEN_CANON_CACHE_CAP` (default 1,000,000 entries; lower if
RAM-constrained — at N = 26 with 20 workers, ~500 K cap × 20 = ~64 GB
canon-cache footprint plus the output Vec). Without a cap, N = 26
OOMs; this env var is the unlock that landed in `fd530cb`.

```sh
DOUBLY_EVEN_THREADS=16 uv run python scripts/bench.py \
  --label parallel-t16 --N 18,20,22
# Larger N — try the deeper cut for finer balance:
DOUBLY_EVEN_THREADS=20 DOUBLY_EVEN_FRONTIER_DEPTH=5 \
  uv run python scripts/bench.py --label parallel-t20-d5 --N 24
# N = 26 — cap canon cache to avoid OOM at 20 workers:
DOUBLY_EVEN_THREADS=20 DOUBLY_EVEN_FRONTIER_DEPTH=5 \
  DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
  uv run python scripts/bench.py --label parallel-t20-d5-n26 --N 26
```

V3+ improvements still deferred: per-worker mass-stop using
Gaborit residual (recovers the 4–11 % sequential mass-stop win);
heuristic per-seed depth selection (light seeds at depth 3, heavy
seeds at depth 5+) once depth-4 hits a ceiling at larger N.

## Useful commands

```sh
uv sync --all-extras --dev               # bootstrap a fresh checkout
maturin build --release -m rust/Cargo.toml \
  && uv pip install rust/target/wheels/doubly_even_kernel-*.whl
# verifier build (D12, dormant): add --features equivalence_verifier
# parallel build (D13, opt-in): add --features parallel
#   then set DOUBLY_EVEN_THREADS=16 at runtime (or pass num_threads= to
#   the kernel entry directly). Sequential path is byte-identical when
#   the env var is unset.
uv run pytest                            # 517 fast tests + 41 slow-skipped (~7 s)
uv run pytest --run-slow                 # adds N=17, 18 Table 3 cells (~10 s total)
uv run python scripts/bench.py --label baseline  # benchmark; writes JSON
DOUBLY_EVEN_THREADS=16 uv run python scripts/bench.py --label parallel-t16 --N 20,22

# Enumerate doubly even codes of length N (yields EnumeratedCode objects)
uv run python -c '
from doubly_even.enumerate.augment import enumerate_doubly_even
for ec in enumerate_doubly_even(8):
    print(f"k={ec.code.rank} |Aut|={ec.aut_order} basis={list(ec.code.basis)}")
'
```

## Git etiquette

- Commits are authored by the user (email + name from local
  `git config`). Don't touch `~/.gitconfig` from inside the agent.
- `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
  trailer goes on every assistant-driven commit.
- The repo had its history rewritten once (`git filter-repo`) to fix
  author emails; further rewrites should be similarly explicit and
  user-approved.
