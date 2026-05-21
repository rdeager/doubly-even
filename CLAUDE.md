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
  experimental/  BHM2012 direct_sum operator (deferred) — see EXPERIMENTAL.md

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
  experimental/  Witt phase-(b) scaffolding + BHM2012 direct-sum
                  mass-seeding (seeds.py) — see EXPERIMENTAL.md

rust/        Rust kernel (doubly_even_kernel), built with maturin
  src/canon.rs       Q_D-graph low-weight-incidence canonicaliser (D10)
                      + native sparsenauty path; default dispatch.
  src/enumerate.rs   Native enumerate_doubly_even recursion (D11).
  src/permutations.rs Schreier-Sims + perm_compose / perm_inverse /
                      compute_column_orbits shared utilities.
  src/experimental/  Dormant audit substrate — see EXPERIMENTAL.md
    feulner.rs       Feulner column-side canonicaliser (D9, ~1000 LOC,
                      reference / diff oracle; dispatch closed).
    feulner_clb.rs   Jerrum CLB + Lemma 5.9 (Feulner §5.2 substrate).
    paired_iso.rs    Leon §10(i) verifier (D12, dormant under
                      `equivalence_verifier` feature).
    invariants.rs    WL + T11/T12/T13 collision-experiment substrate.
  Cargo.toml         Feature flags (all default OFF): parallel,
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

Headline (post-D13-V4 mass-stop + memory work, 2026-05-22, 13700K):
**~220× cumulative at `N = 22`** since the pre-kernel D6 baseline
(152 s → 0.691 s parallel-20t); **> 880×** since the pre-Q_C original
baseline. Parallel kernel: `N = 20` in 0.22 s; **`N = 22` in 0.691 s**
(20 threads, depth=4, mean of 3); **`N = 24` in 9.13 s** (22 threads,
depth=5); **`N = 26` in 184.8 s** (20 threads, depth=5, cap=500K —
down from 218 s pre-V4). Sequential at `N = 22` is **6.64 s** post-V4
(was 7.01 s on dev box, 6.87 s on 13700K pre-V4). We beat Sage's
`self_orthogonal_binary_codes` by **~525× end-to-end at `N = 22`**
(Sage 363.85 s, single-threaded; parallelising Sage would be weeks of
Cython surgery).

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
13. **D13 Outer-DFS parallelism** (2026-05-18, V3 2026-05-21) —
    sequential traversal to a configurable cut depth, then a
    crossbeam-channel worker pool runs each accepted subtree on its
    own thread. Sparsenauty is parallel-safe under the `tls` feature
    on `nauty-Traces-sys` (already on; `USE_TLS` → `_Thread_local`
    on mutable globals). Per-worker canon caches; per-worker
    mass-stop disabled (loses 4–11 % vs sequential; gained outright
    by parallelism). V1 (depth=3): 3.1× at N=22. V2 (depth=4):
    5.8× at N=22 (6.87 s → 1.18 s, 16t on 13700K). **V3 pipelined
    seeder (2026-05-21, the new default):** workers spawn first and
    block on `task_rx.recv()`; the seeder runs on the main thread
    and pushes each depth-`frontier_depth` seed into the channel as
    it's discovered, overlapping seeder DFS with worker recursion
    (DFGHILM Appendix B.4 producer-consumer recipe). Closes the
    serial-seeder Amdahl ceiling diagnosed in
    `architecture/07-parallel-scaling-profile.md` (~44 % of V2 wall
    at N=22 t=16 d=4). **8.55× at N=22** (7.01 s → 0.82 s, 20t d=4
    on 22-core); **11.2× at N=24** (107 s → 9.54 s, 22t d=5).
    Bounded channel cap = `num_threads * 4` doubles as backpressure
    once the seeder pipes directly into it. Behind off-by-default
    Cargo feature `parallel`; enabled at runtime via
    `DOUBLY_EVEN_THREADS` env var (and tunable via
    `DOUBLY_EVEN_FRONTIER_DEPTH`, default 4) or `num_threads=`
    kwarg on the kernel entry. Conflicts with `traces_qd` (Traces
    uses non-TLS static work queues) — guarded by `compile_error!`.
    Determinism harness in `rust/tests/parallel_determinism.rs`
    (covers N = 12, 14, 16 at threads 2, 4, 8).
14. **D13-V4 sprint** (2026-05-22, 13700K): four cuts attacking
    per-call memory pressure + recovering the worker mass-stop V2/V3
    left disabled. Net: N=22 0.82 → 0.691 s (-16 %), N=26 218 →
    184.8 s (-15 %), seq 7.01 → 6.64 s (-5.3 %).
    - **Cut 1 (`40dc251`)**: `mimalloc` global allocator. Per-thread
      arenas eliminate ptmalloc cross-thread contention on the ~210 KB
      of fresh allocs per `canon_info_*` call. -3.6 % seq, -4.8 %
      parallel-unbound, -3.8 % N=24. One line in `lib.rs`.
    - **Cut 2+3 (`ec6e65f`)**: thread-local `CanonScratch` struct
      hoists all 13 per-call Vecs (`cw`, `d`, `v`, `e`, `lab`, `ptn`,
      `orbits`, `cg_v`, `cg_d`, `cg_e`, `right_lists`, `by_cell`,
      `by_weight`) shared between `canon_info_native` +
      `canon_info_qd_native`. Grow-only via `.clear()`/`.resize()`;
      zero per-call heap allocs after warmup. Also drops the
      `left_write = v.clone()` (~8 KB memcpy/call) in both
      `build_sparsegraph` and `build_low_weight_sparsegraph` via a
      stack-local cursor. -2.2 % seq wall, -3 % per-call parallel.
    - **Cut 4 (`0059089`)**: per-worker mass-stop via shared
      `GlobalMassTracker` (Mutex<Vec<u128>> + Arc). Workers atomically
      add `N!/|Aut|` on every emit; consult `is_full(k+1)` before
      descending. Invariant: skip-when-full can only over-search,
      never under-search (canonical augmentation forbids duplicates).
      **The single biggest V4 lever**: -20.7 % wall at N=22 PEEP,
      -20.3 % unbound, -0.8 % at N=24. Also fixes pre-existing
      `parallel_profiling` feature breakage from V3 ship.

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

At runtime, set `DOUBLY_EVEN_THREADS`. Post-D13-V5 measurements on
the 13700K (16 physical / 24 logical) confirm:

- **N = 22**: sweet spot at `t = 20` (mean 0.691 s over 3 runs;
  t=18 = 0.720, t=22 = 0.717, both ~4 % worse). On the 13700K this
  uses all 8 P-core SMT pairs + 4 E-cores, leaving 4 E-cores idle
  for scheduler migration. On a symmetric N-core host the rough rule
  remains `t = N − 2`; on hybrid the actual sweet spot is empirical
  per topology — see `architecture/07-parallel-scaling-profile.md`
  § "V4 contention/memory sprint".
- **N ≥ 24**: sweet spot at **`t = 24` (full logical core count)**.
  D13-V5 re-measurement (2026-05-23) on the 13700K:
  N=24 t=24 = 8.90 s vs t=22 = 9.12 s (**-2.4 %**, mean of 3);
  N=26 t=24 = 169.8 s vs t=20 = 184.2 s (**-7.8 %**). V4's
  per-call thread-local scratch + mimalloc cut SMT contention enough
  that oversubscribed P-core siblings now pay back the latency cost.
  Going above 24 (t=26, t=28) regresses — true logical-core
  count is the ceiling, not a target to exceed. See
  `architecture/07-parallel-scaling-profile.md` § "V5 thread-count
  recalibration".

`DOUBLY_EVEN_FRONTIER_DEPTH` defaults to 4 and rarely needs
tweaking — at N ≥ 24 a value of 5 is better because the tree is
bigger and the deeper split gives finer load balance.

For `N ≥ 26`, also cap the per-worker canon cache via
`DOUBLY_EVEN_CANON_CACHE_CAP` (default 1,000,000 entries; lower if
RAM-constrained — at N = 26 with 20 workers, ~500 K cap × 20 = ~64 GB
canon-cache footprint plus the output Vec). Without a cap, N = 26
OOMs; this env var is the unlock that landed in `fd530cb`.

```sh
# N ≤ 22 — leave 2 cores headroom (use t = cores − 2):
DOUBLY_EVEN_THREADS=20 uv run python scripts/bench.py \
  --label parallel-t20 --N 18,20,22
# N ≥ 24 — full logical-core count (V5 finding) + deeper cut:
DOUBLY_EVEN_THREADS=24 DOUBLY_EVEN_FRONTIER_DEPTH=5 \
  uv run python scripts/bench.py --label parallel-t24-d5 --N 24
# N = 26 — cap canon cache to avoid OOM at 24 workers:
DOUBLY_EVEN_THREADS=24 DOUBLY_EVEN_FRONTIER_DEPTH=5 \
  DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
  uv run python scripts/bench.py --label parallel-t24-d5-n26 --N 26
```

D13-V5 sprint (2026-05-23) investigated and closed the two named
post-V4 candidates:
- **Adaptive per-seed depth**: doesn't apply at N ≥ 24. Channel-based
  load balancing already absorbs the heavy-seed tail (N=24 t=22
  max/mean = 1.01× per `parallel_profiling` measurement; the N=22
  1.28× imbalance is an N ≤ 22-shape artifact). Forcing adaptive
  recursion at the boundary regresses N=24 by **+43-56 %** when
  enabled at any threshold — the depth-4 → 5 split explodes the
  seed set, the bounded channel backpressures, and the seeder runs
  finely-grained work that workers could otherwise parallelise.
  Profile data + analysis in
  `architecture/07-parallel-scaling-profile.md` § "V5 thread-count
  recalibration". No code change shipped.
- **Shared canon LRU**: intra-worker primary-cache hit rate at N=24
  is only **3.13 %** (41 K hits / 1.31 M is_canon_aug calls). The
  cross-worker dedup opportunity is capped by the same rate; even
  if every per-worker miss became a shared hit, the wall reduction
  ceiling is ~4 %, and DashMap shard contention plus `Rc → Arc`
  surgery would eat most of it. Below the engineering threshold for
  "final lever" status. Closed.

The shipped V5 win is **the threading recommendation update**: use
`t = 24` (full logical-core count) at N ≥ 24, not `t = 22` as the
post-V4 docs suggested. Pure env-var change, no recompile. V4's
per-call thread-local scratch + mimalloc cut SMT contention enough
that oversubscribed P-cores now pay back the latency cost.

## GCP quickstart (cloud port, validated 2026-05-21)

The kernel ports to Google Cloud Emerald Rapids with **zero per-IPC
penalty** — per-thread wall is 1.65× slower than the 13700K, which
is purely the clock ratio. Validated on `c4-standard-24` (Intel
Xeon Platinum 8581C in us-east4): N=26 at t=24 d=5 in 285 s, all
DFGHILM cells match. 86 % nauty utilisation at N=26 confirms the
parallel kernel saturates cloud silicon cleanly. Full writeup:
`markdown/notes/gcp-shakedown-2026-05-21.md`. Refined
c4-288-metal estimates: `markdown/architecture/06-scaling-frontier.md`
§"GCP Emerald Rapids validation 2026-05-21".

Two scripts in tree (nproc-aware, work unchanged on c4-24 / c4-48 /
c4-288-metal):

```sh
# On the GCP VM, fresh Ubuntu 24.04, one-liner bootstrap:
curl -fsSL https://raw.githubusercontent.com/<gh-user>/doubly-even/main/scripts/gcp-setup.sh \
  | bash -s -- --repo https://github.com/<gh-user>/doubly-even
# (~10 min: apt → rustup → uv → clone → uv sync → maturin --release --features parallel → smoke pytest)

# Then run the three-stage bench (Run A seq, Run B t=nproc/2 d=4, Run C t=nproc d=5):
cd ~/doubly-even
scripts/gcp-bench.sh shakedown-c4-24
```

**Gotchas**:
- `kernel_build_info()` returns `"baseline"` even with `--features
  parallel` (it only distinguishes verifier vs not). To verify
  parallel is in: check `inspect.signature(k.enumerate_doubly_even)`
  contains `num_threads`.
- Fresh GCP projects cap at **24 global vCPUs**; first quota
  request for more is denied with a 48-h cooldown for billing
  history. Plan ahead.
- On c4-288-metal, set `DOUBLY_EVEN_CANON_CACHE_CAP=200000` (not
  the local 500K) — 288 workers × 500K × 5 KB = 720 GB would eat
  the 2160 GB ceiling without enough headroom for output Vec +
  scratch. 200K leaves ~1.8 TB free.

## Useful commands

```sh
uv sync --all-extras --dev               # bootstrap a fresh checkout
maturin build --release -m rust/Cargo.toml \
  && uv pip install rust/target/wheels/doubly_even_kernel-*.whl
# verifier build (D12, dormant): add --features equivalence_verifier
# parallel build (D13-V3 pipelined, opt-in): add --features parallel
#   then set DOUBLY_EVEN_THREADS=<cores − 2> for N ≤ 22 or <cores>
#   for N ≥ 24 (or pass num_threads= to the kernel entry directly).
#   Sequential path is byte-identical when the env var is unset.
uv run pytest                            # 527 fast tests + 41 slow-skipped (~6 s)
uv run pytest --run-slow                 # adds N=17, 18 Table 3 cells (~7 s total)
uv run python scripts/bench.py --label baseline  # benchmark; writes JSON
DOUBLY_EVEN_THREADS=20 uv run python scripts/bench.py --label parallel-t20 --N 20,22

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
