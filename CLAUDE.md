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

rust/        Rust kernel — Cargo WORKSPACE since 2026-06-11:
  Cargo.toml         workspace root = the thin pyo3 wrapper crate
                      (doubly-even-kernel, cdylib; mimalloc allocator
                      lives here). Feature flags forward 1:1 to core
                      (all default OFF): parallel, equivalence_verifier,
                      dense_qd[_tc0,_refinvar], traces_qd, nauty_hist,
                      parallel_profiling, phase_timers. NEVER
                      --all-features (parallel × traces_qd is a
                      deliberate compile_error).
  .cargo/config.toml x86-64-v3 codegen (cwd-discovered — build via
                      scripts/install-kernel.sh; verify with
                      kernel_target_features(), avx2 must be True).
  src/lib.rs         the 9 pyfunctions + py_exports.rs (dormant FFI).
  core/              doubly-even-core (rlib) — ALL algorithms:
    src/enumerate/   the recursion, split by concern: worker.rs
                      (traversal + candidate test), cache.rs (two-tier
                      canon cache), stats.rs (stats layout — SINGLE
                      SOURCE OF TRUTH, exported via
                      kernel_stats_layout()), drivers.rs (seq/parallel/
                      streaming entries).
    src/parent_rule.rs coset-spectrum φ cascade (D15–D17) — module
                      doc-comment states the math; proofs in
                      docs/theory.md.
    src/orbit.rs     σ_Q orbit-min BFS (D18 m4r) + singular reps.
    src/canon.rs     Q_D-graph low-weight-incidence canonicaliser (D10)
                      + native sparsenauty path; default dispatch.
    src/seeder_pool.rs D16 helper pool for seeder σ_Q calls.
    src/permutations.rs Schreier-Sims + perm utilities.
    src/experimental/ Dormant audit substrate — see EXPERIMENTAL.md
                      (feulner D9, paired_iso D12, invariants, …).

scripts/microbench/  standalone crate; links doubly-even-core DIRECTLY
                      (since 2026-06-11) — production arms ARE
                      production code, no hand-copied clones to sync.
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

Post-2026-06 closures (hand SIMD intrinsics, bitsliced BFS, probe
restructures, L1-fit, adaptive seeder gate, …) live in the
**`docs/bottlenecks.md` §6 dead list** — check it before proposing any
performance idea. Note the category distinction recorded there: cheap
*rejectors* died of per-probe cost; the parent-rule *redefinition* won.

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

## Public docs

Landing docs for GitHub readers are under `/workspace/src/docs/`:
- `algorithm.md` — the nine-lever narrative with cumulative ablation table.
- `theory.md` — the parent-rule math: definitions, theorems, proofs,
  theorem-to-code index.
- `bottlenecks.md` — **the LIVING performance state** (walls, phase
  shares, ranked next levers, dead list). Single home of "where the
  time goes"; carries the when-a-lever-ships doc-maintenance checklist
  at the top — follow it.
- `benchmarking.md` — measurement runbook (install discipline, gates,
  stats schema, microbench suite, samply).
- `performance.md` — measured walls; cloud rows; tuning knobs.
- `reproducing.md` — clean-checkout-to-cloud-run recipe.
- `cluster-deployment.md` — honest multi-node sketch (untested at scale).
- `references.md` — credits (DFGHILM, RL Miller, Bouyukliev,
  Gaborit, McKay, McKay-Piperno, Sage).

Public docs **never use internal session labels (D5/D6/D10/D11/D13/V3/V4/V5)**.
Those labels stay only in `/workspace/markdown/architecture/*.md` and in
the section below.

## Performance state

**Live numbers, phase shares, ranked next levers and the dead list are
in `docs/bottlenecks.md` — that file, not this one, is the single
source for "where the time goes".** Desktop seq headline (2026-06-12,
**D19 autom-only canon SHIPPED**: getcanon=FALSE on the 80.7 % of
canon calls whose labelling no decision reads; 1.09–1.10× seq,
decisions bit-identical): N=22 seq 0.623 s / par 0.24 s; N=24 seq
5.72 s; N=26 seq 74.6 s / par 9.21 s (t=24 d=4 cap=500K).

**Cloud days DONE 2026-06-14.** The parallel mass-mutex futex storm
(D13 single global `Mutex`, 85 % `sy` at 96t) is FIXED and
cloud-confirmed on c4a-highmem-96-metal (`a957200`, batched writes +
atomic full-flags; `sy`~0 %/`us`~92 %). Counts sweep N=24→29 at
**d=5/96t** done: **N=29 in 33.1 min** (239,465,540 classes, 22× over
the 12.32 h pre-fix run, all ranks `mass==σ` certified). **d=5 is the
96-core knee** (N=28 d=4/5/6 = 301.6/139.8/220.1 s); the old "d=4 best /
d=5 loses" was a ≤24-core + mutex-storm artifact. The live N>27 frontier
is now **tail load-imbalance** (N=29: ~92 % mass by 13 min, 33 min total
— the heavy-subtree tail ~doubled the wall). Chosen lever (DESIGNED
2026-06-14): **demand-driven self-subdivision** (victim-initiated
work-sharing — idle workers fed via `try_send` off the existing channel,
not classic work-stealing), to be built next session on
`feature/tail-self-subdivision` (entry point:
`markdown/notes/self-subdivision-build-next-session-2026-06-14.md`). 2026-06-13
session: **counts-only output mode SHIPPED** (`run_counts.py` +
`dec progress`; the ONLY N≥30-capable entry — σ(30,·) ≈ 2^136
overflows the old u128 mass spine, now 256-bit, decisions
gate-identical); lever-4 decomp/twin experiment RAN (P1 component
canon DEAD at 2.2 % of N=26 nauty time; survivor = twin compression,
93 % of nauty time twin-bearing, gated on a nauty A−B —
`markdown/notes/decomp-twin-logging-2026-06-13.md`); SVE2 histogram
ceiling measured (1.04×/1.07×/1.18–1.20× e2e at N=26/29/32 —
`markdown/notes/sve2-ablation-2026-06-13.md`). External GPT/Gemini
feedback adjudicated in
`markdown/notes/external-feedback-review-2026-06-12.md`; φ-tie
collision exposition: `markdown/notes/tie-collisions-2026-06-12.md`.

Knob quick-reference (details in `docs/benchmarking.md` §3):

| knob | default | note |
|---|---|---|
| `DOUBLY_EVEN_THREADS` | unset (seq) | cores−2 at N≤22, full count at N≥24 |
| `DOUBLY_EVEN_FRONTIER_DEPTH` | 4 | d=4 best ≤24 cores; **d=5 is the knee at 96 cores** (N=28 d=4/5/6 = 301.6/139.8/220.1 s, cloud 2026-06-14). Granularity knob — deeper splits the tail finer at the cost of a longer serial seeder walk |
| `DOUBLY_EVEN_CANON_CACHE_CAP` | 1M | **INERT for speed** (0.003 % hit rate post-D15 — `canon_calls` ≈ classes+ties). A memory knob only: 100K on 96-core cloud, 200K on ≥200-core boxes |
| `DOUBLY_EVEN_PARENT_RULE` | coset-spectrum | `legacy` = kill-switch, `audit` = measurement |
| `DOUBLY_EVEN_CANON_LABELLING` | autom-only | `full` = D19 kill-switch (labelling on every call + per-class ccol in output) |
| `DOUBLY_EVEN_TIE_DUMP` | unset | JSONL tie dump, sequential only (collision analysis) |
| `DOUBLY_EVEN_DECOMP_LOG` | unset | JSONL decomp/twin record per canon call, sequential only (lever-4 measurement, 2026-06-13) |
| `DOUBLY_EVEN_PHI_MAX_RANK` | 13 | legacy rule above (sound mixing) |
| `DOUBLY_EVEN_SEEDER_THREADS` | = threads | 0 disables the seeder pool |
| `DOUBLY_EVEN_SEEDER_PAR_MIN_L` | 22 | load-bearing; don't lower |
| `DOUBLY_EVEN_NO_MASS_STOP` | off | ablation knob (mass-stop measured ~inert post-D19: ≤26 candidates pruned at N=27) |
| `DOUBLY_EVEN_MASS_FLUSH_INTERVAL` | 2048 | parallel only: emissions/worker between shared-mass-tracker flushes; the 96-thread futex-storm fix. Contention knob, not pruning — classes+mass identical at any value |
| `DOUBLY_EVEN_SELF_SUBDIVIDE` | **off** | **D20 demand-driven self-subdivision** (MERGED to main, default OFF = byte-identical). On: a busy worker at shallow depth donates an accepted child onto the seed channel when peers are idle (victim-initiated work-sharing) — adaptive tail depth for the N>27 load-imbalance. Local ladder PASS; **first scaled signal N=27 1.30× / −23 %** (20-core, contended, single run). Not yet the production default — `(d, δ)` co-optimised on cloud (`scripts/cloud_depth_sweep.py`) then flip ON |
| `DOUBLY_EVEN_SELF_SUBDIVIDE_DELTA` | 1 | D20 donatable depth past `frontier_depth` (parent gate `k ≤ frontier_depth+δ`); sets adaptive granularity `frontier_depth+δ+1`. **Decouples seed granularity from seeder span** — co-optimise `(FRONTIER_DEPTH × δ)` on cloud; the old "d=5 knee" is a no-lever result |
| `DOUBLY_EVEN_SELF_SUBDIVIDE_POLL_MS` | 2 | D20 worker `recv_timeout` poll for the donation-aware termination loop |
| `M4R_MIN_L = 14` | const in `core/src/orbit.rs` | orbit-BFS byte-table crossover |

Internal-label index (the label↔name map; one line per lever, full
writeups in `markdown/architecture/04-optimisations.md` §-by-§):
D2 quotient-space candidates · D3 two-tier canon cache · D4 weight-enum
prefilter · D5 gaborit_sigma + mass-stop · D6 σ_Q orbit-min tables ·
D7 Witt infra (parked) · D8 degree initial partition · D9 Feulner
reference · D10 Q_D-graph canonicaliser · D11 native Rust recursion ·
D12 paired-iso (dormant) · D13(+V3–V5) outer-DFS parallelism ·
D14 fingerprint cache (REVERTED) · **D15 coset-spectrum parent rule** ·
**D16 split-frame φ + amax + pooled seeder** · **D17 E-chain** ·
**D18 m4r orbit BFS** · 2026-06-11: workspace restructure +
x86-64-v3 codegen (no label) · **D19 autom-only canon**
(getcanon=FALSE on accepts; public name "automorphism-only
canonicalisation", algorithm.md lever 10) · **D20 demand-driven
self-subdivision** (victim-initiated tail work-sharing,
`feature/tail-self-subdivision`, default OFF; public name
"demand-driven self-subdivision"). Public docs use the descriptive names
only.

History of this section: the long narrative that lived here was
relocated verbatim to
`markdown/notes/claude-md-relocated-2026-06-11.md` (most of it
duplicates 04-optimisations.md §D15–D18 / 08-post-d15-profile.md /
the wrapup notes). Next-session entry points live in
`markdown/notes/<latest>-wrapup-*.md`.

## Quickstart

```sh
scripts/install-kernel.sh parallel       # build + install the wheel
                                         # (cd's into rust/ for the v3
                                         # config; probes avx2 + module)
uv run pytest                            # 549 passed + 41 slow-skipped
uv run python scripts/bench.py --label <arm>-seq --N 18,22,24
DOUBLY_EVEN_THREADS=24 DOUBLY_EVEN_FRONTIER_DEPTH=4 \
  DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
  uv run python scripts/bench.py --label <arm>-par-n26 --N 26
```

The full bench-a-change protocol (one-wheel-one-label, the uv-cwd
wheel trap, decision-identity gates, microbench suite, samply) is
`docs/benchmarking.md` — read it before benching anything.

GCP gotchas (full recipe: `docs/reproducing.md`):
- `kernel_build_info()` says "baseline" even with `--features
  parallel`; check `num_threads` in
  `inspect.signature(k.enumerate_doubly_even)` instead, and
  `kernel_target_features()` for the codegen.
- Fresh GCP projects cap at 24 global vCPUs; the first quota bump is
  denied with a 48-h cooldown. Plan ahead.
- On c4-288-metal set `DOUBLY_EVEN_CANON_CACHE_CAP=200000` (288
  workers × 500K would eat the 2160 GB ceiling).

## Useful commands

```sh
uv sync --all-extras --dev               # bootstrap a fresh checkout
scripts/install-kernel.sh parallel       # standard build (use this, not
                                         # bare maturin — it cd's into
                                         # rust/ so .cargo/config.toml
                                         # is discovered, and probes the
                                         # installed wheel)
# other feature sets: scripts/install-kernel.sh parallel,phase_timers
# sequential path is byte-identical when DOUBLY_EVEN_THREADS is unset
uv run pytest                            # 549 passed + 41 slow-skipped (~47 s)
uv run pytest --run-slow                 # 580 passed + 10 skipped
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
- A `Co-Authored-By: Claude <model> <noreply@anthropic.com>` trailer
  goes on every assistant-driven commit (use the current model name).
- The repo had its history rewritten once (`git filter-repo`) to fix
  author emails; further rewrites should be similarly explicit and
  user-approved.
