# Outsider's tour: `doubly-even` for cold-start agents

This document is for an LLM / agent that has never seen this project and
needs to do useful work after one read-through. It is **not** the same as
`CLAUDE.md` (which assumes you are already doing engineering work here)
or `README.md` (which is the GitHub landing page).

Read top-to-bottom in order. Citations are file:line.

---

## 1. What this is, in one paragraph

`doubly-even` enumerates **binary linear codes of length `N`, dimension `k`,
all codeword weights divisible by 4**, up to permutation equivalence of
coordinates. The underlying motivation is the Adinkra chromotopology
problem in supersymmetric representation theory (Doran–Faux–Gates–Hübsch–
Iga–Landweber–Miller, **DFGHILM**). The canonical algorithmic spec is
[`/workspace/inbox/mathpix/DFGHILM_ATMP.md`](../inbox/mathpix/DFGHILM_ATMP.md)
Appendix B. The project replicates DFGHILM Table 3 exactly through
`N = 28` and published the first publicly reproducible `N = 29` and
`N = 30` enumerations (239,465,540 and 3,786,528,214 classes,
mass-formula certified at every rank — certificates at
[`docs/results/n29.json`](docs/results/n29.json) and
[`docs/results/n30.json`](docs/results/n30.json)), verified against four
independent oracles (mass formula, DFGHILM published counts, Sage's
`self_orthogonal_binary_codes`, and rlmiller.org's `N = 28` tables).

---

## 2. The problem in 30 seconds

A **binary linear code** `C` is a subspace of `F_2^N` (binary vectors of
length `N` under XOR). Two codes are **permutation-equivalent** if one
becomes the other under a column permutation `σ ∈ S_N`. **Doubly even**
means every codeword has Hamming weight divisible by 4. The job:
**count equivalence classes** of doubly-even codes for each `(N, k)`.

DFGHILM Table 3 is the published reference: `N=8, k=1 → 2 classes`;
`N=22, k=11 → 5118 classes`; `N=26 → 494,272 classes`; etc. The exact
sequence we reproduce.

---

## 3. The three code paradigms and when to read each

The repo carries **three implementations on purpose** — they answer
different questions:

| Paradigm | Where | When to read it |
|---|---|---|
| **Clean Python** (pedagogical) | `src/doubly_even/clean/` | "What does the algorithm actually do?" Closest to executable pseudocode. 651 LOC standalone, no perf clutter. `N=22` in 31 s. **One deliberate divergence**: clean implements the legacy σ-derived parent rule; production uses the coset-spectrum rule (same classes, different representatives — see [`docs/theory.md`](docs/theory.md)). |
| **Production Rust** (optimised) | `rust/` (workspace: `core/` = algorithms, root = pyo3 wrapper) | "Why is the production code shaped this way?" ~1500× faster than Sage at `N=22`. Heavy perf engineering. FFI to nauty/sparsenauty. `N=22` in 0.24 s parallel / 0.69 s sequential; `N=26` in ~6.3 s parallel (24t, default d=3/δ=3 with D20 self-subdivision). |
| **Lean 4 spec** (formal) | `lean/` | "What is the *exact* mathematical object?" Dependent types pin down the definitions; you can verify your understanding against `Code N k`'s type signature. **Spec scaffold only as of 2026-05-21**, no enumerator yet. |

**Each is a different lens**. The clean Python rewrite (2026-05-21)
surfaced the k=1,2 Young-subgroup pre-seed via profiling — an
algorithmic insight that landed in production (commits `a9e2894`,
preceded by `7bb6ea6`). The Lean 4 scaffold (2026-05-21) is a third
paradigm bet on the same principle: stating invariants precisely may
expose redundant or strengthenable checks.

---

## 4. How the algorithm works

The algorithm is **canonical augmentation** (DFGHILM Appendix B.4): a
DFS over `(N, k)` codes built by adding one basis vector at a time,
where a child node is kept iff the added vector is the *canonical*
choice among its orbit under `Aut(parent)`.

Read this with [`src/doubly_even/clean/_augment.py`](src/doubly_even/clean/_augment.py)
open. The shape:

```
enumerate_doubly_even(N):           # entry point
  for each k in 1..N/2:             # dimension
    for each canonical k-code C:    # canonical-augmentation tree
      yield C

  recursion:
    parent C of dim k → children { C + ⟨v⟩ : v makes child doubly even }
    keep child D iff is_canonical_augmentation(D, parent=C)
```

Four load-bearing optimisations in the recursion (the clean Python
shows the first three in their simplest form; the fourth lives only in
the Rust kernel):

1. **σ_Q quotient-space orbit-min** ([`clean/_canon.py`](src/doubly_even/clean/_canon.py))
   — restrict child search to one rep per orbit of `Aut(C)` on
   `F_2^N / C`. Massive prefilter.
2. **Q_D-graph canonical labelling**
   ([`clean/_canon.py`](src/doubly_even/clean/_canon.py)) — for the
   canonicality test, build a low-weight-incidence bipartite graph and
   canonicalise with nauty. Cheaper than the natural bipartite encoding.
3. **Closed-form k=1, k=2 seeds**
   ([`clean/_canon.py`](src/doubly_even/clean/_canon.py)) — at the
   first one or two levels, codes have Young-subgroup-product Auts;
   you can generate canonical reps directly without recursing.
4. **The coset-spectrum parent rule** (Rust kernel only,
   [`rust/core/src/parent_rule.rs`](rust/core/src/parent_rule.rs)) —
   the headline 2026-06 lever family. McKay's framework lets any
   iso-invariant function pick the canonical parent; selecting by the
   complement-coset weight spectrum φ (with canon only on accepts and
   ties) removes ~94–97 % of canonicaliser calls, and three follow-on
   theorems (split-frame sharing, the one-comparison pair-max reject,
   the E-set chain) decide most candidates in O(1). Narrative:
   [`docs/algorithm.md`](docs/algorithm.md) levers 6–8; definitions,
   theorems and proofs: [`docs/theory.md`](docs/theory.md).

The validation oracle in tests is the **mass formula**:
`Σ N! / |Aut(C_i)| = σ(N, k)` (Gaborit closed form). If your class
list is wrong, this almost always catches it. See
[`tests/test_augment.py`](tests/test_augment.py) for the
DFGHILM Table 3 cells; the formula check is internal.

---

## 5. The Rust kernel in one paragraph

Production lives at [`rust/`](rust/), a two-crate Cargo workspace:
`rust/core/` (`doubly-even-core`) holds every algorithm, the root
package is the thin pyo3 wrapper built by maturin as
`doubly_even_kernel`. The kernel owns the **entire**
canonical-augmentation recursion plus the canonicaliser (`canon_info`)
via [`nauty-Traces-sys`](https://crates.io/crates/nauty-Traces-sys).
The microbenches under `scripts/microbench/` link `doubly-even-core`
directly, so their "production arms" are production code. Key files
(all under `rust/core/src/`):

- [`enumerate/`](rust/core/src/enumerate/) — the recursion, split by
  concern: `worker.rs` (traversal + candidate test), `cache.rs`
  (canon-info computation + the secondary weight-enumerator cache; the
  primary RREF-keyed per-worker LRU was removed 2026-06-14), `stats.rs`
  (the stats layout — single source of truth, exported to Python),
  `drivers.rs` (sequential / parallel / streaming entries; crossbeam
  worker pool fed by a pipelined seeder with demand-driven
  self-subdivision).
- [`parent_rule.rs`](rust/core/src/parent_rule.rs) — the coset-spectrum
  φ cascade, where most candidates die. The module doc-comment is a
  compact statement of the math; proofs in [`docs/theory.md`](docs/theory.md).
- [`orbit.rs`](rust/core/src/orbit.rs) — σ_Q orbit-min BFS
  (four-Russians image kernel) + singular-reps enumeration.
- [`canon.rs`](rust/core/src/canon.rs) — Q_D-graph canon
  + sparsenauty integration.
- [`seeder_pool.rs`](rust/core/src/seeder_pool.rs) — helper pool for
  the parallel seeder's large σ_Q calls.
- [`permutations.rs`](rust/core/src/permutations.rs) —
  Schreier-Sims + perm utilities.

**The live performance state — walls, phase shares, ranked next
levers, dead list — is [`docs/bottlenecks.md`](docs/bottlenecks.md)**;
how to measure is [`docs/benchmarking.md`](docs/benchmarking.md). A
2026-05 sign-off once declared the `N ≤ 22` frontier "saturated at the
algorithmic level"; the coset-spectrum parent rule (2026-06) broke it
by 7.6× sequential — the claim held only inside the fixed-parent-rule
frame. Treat every "closed" claim as scoped to the bottleneck profile
that produced it. With `N = 30` now complete (2026-06-15), the open
frontier is `N = 31..32`; scaling plans:
[`/workspace/markdown/architecture/06-scaling-frontier.md`](../markdown/architecture/06-scaling-frontier.md).

---

## 6. Dead ends — **do not propose these**

These directions were tried, measured, and closed. If you re-propose
any of them, your suggestion will be rejected with "see X". Read the
named memory entry before going further.

| Direction | Why closed | Memory entry |
|---|---|---|
| **T11 / pair-gram class-fingerprint cache** | Requires offline blocklists; circular for `N ≥ 26` single-pass. Cheap invariants have collisions that grow rapidly with N. | `feedback_no_offline_blocklist.md`, `project_pair_gram_initpart_dormant.md` |
| **Custom Rust canonicaliser to beat nauty** | Three independent attempts (Feulner D9, paired-iso D12, bit-parallel WL) all confirm Rust loses to nauty's C on refinement. `78 µs/call` is the sparsenauty floor at this graph shape. | `project_bitpar_refine_microbench.md`, `project_q6_audit_closed.md` |
| **WL fast-path / WL-discretises shortcut** | `0 of 5,118` `N=22` codes have `\|Aut\|=1`. Doubly-even canonical augmentation only visits codes with rich Aut, so WL discretisation is rare. | `project_wl_canonical_closed.md` |
| **Probabilistic dedup with cheap invariants** | `678/5,118` collisions at `N=22` even with `\|Aut\|` oracle. Step 4 paired refinement is load-bearing. | `project_collision_experiment.md` |
| **Adaptive per-seed depth** | Regresses `N=24` by +43-56%. The N=22-shape load imbalance is an N≤22 artefact; channel balances `N≥24` already. | `project_d13_v5_thread_recalibration.md` |
| **Shared canon LRU across workers** | Intra-worker hit rate at `N=24` is only 3.13%, capping cross-worker dedup at ~4%. | `project_d13_v5_thread_recalibration.md` |
| **`target-cpu=native`** | ~~Measured ±0.14% (noise) at `N=22` (2026-05-21)~~ **REVERSED 2026-06-11**: true while the wall was ~90 % sparsenauty C, false once the parent-rule levers made Rust code ~50 % of the wall — `x86-64-v3` then measured 1.06–1.11× and **shipped** (`rust/.cargo/config.toml`). Object lesson: a "measured dead" verdict is scoped to the bottleneck profile that produced it. | `simd-codegen-verdict.md` |
| **Witt-orbit phase-(b) scaffolding** | Shipped but no Python speedup. Lesson: complete an implementation before optimising it. | `phase_b_empirical_finding.md` |
| **Post-2026-06 dead list** (hand SIMD intrinsics, bitsliced BFS, probe restructures, L1-fit, …) | One row each in the living profile doc, with evidence pointers. | [`docs/bottlenecks.md`](docs/bottlenecks.md) §6 |

The general principle: **per-probe cost is the trap**. Cheap-looking
invariants share nauty's refinement primitives, so per-probe cost ≈
per-call cost, killing the cache benefit. But note the counterpoint
that broke five sprints of closure analysis: every *cheaper rejector
bolted onto the old parent rule* died of that trap, while the
coset-spectrum rule won by **changing which question the parent test
asks** (a different parent function, not a prefilter). Judge new ideas
by which of those two categories they fall in. See
[`/workspace/markdown/architecture/04-optimisations.md`](../markdown/architecture/04-optimisations.md)
for the lever-by-lever writeup.

---

## 7. For specific tasks, read

| If you want to … | Start here |
|---|---|
| Understand the math precisely | `lean/DoublyEven/Code.lean` (definitions as types); `docs/theory.md` (the parent-rule theorems, with proofs and a theorem-to-code index) |
| Understand the algorithm | `src/doubly_even/clean/_augment.py` + `clean/_canon.py`; then `docs/algorithm.md` for the lever narrative |
| Find the current bottleneck / next lever | `docs/bottlenecks.md` (living doc — single home of "where the time goes") |
| Bench or profile a change | `docs/benchmarking.md` (install discipline, gates, microbench suite, samply) |
| Reproduce DFGHILM Table 3 | `scripts/bench.py`; oracles in `tests/test_augment.py` |
| Optimise production | `rust/core/src/{enumerate/,parent_rule.rs,orbit.rs,canon.rs}` + `markdown/architecture/04-optimisations.md` |
| Run on a bigger machine | `markdown/architecture/06-scaling-frontier.md` + `scripts/gcp-{setup,bench}.sh` |
| Extend the Lean spec | `lean/README.md` + `lean/DoublyEven/` modules |
| Avoid dead ends | Section 6 above + `docs/bottlenecks.md` §6; `EXPERIMENTAL.md` indexes all quarantined code |
| See what was tried & why it failed | `/home/dev/.claude/projects/-workspace-src/memory/` (auto-memory) |

---

## 8. Files-at-a-glance

```
/workspace/
├── inbox/mathpix/DFGHILM_ATMP.md      ← spec; Appendix B is the algorithm
├── inbox/mathpix/1907.10363v1.md      ← Bouyukliev-Bouyuklieva validation counts
├── markdown/
│   ├── README.md                       ← markdown-side entry
│   └── architecture/
│       ├── 04-optimisations.md         ← per-lever writeup
│       ├── 05-retrospective.md         ← why N≤22 is saturated
│       ├── 06-scaling-frontier.md      ← N=28,29,30 plan
│       └── 07-parallel-scaling-profile.md
└── src/                                ← git repo
    ├── CLAUDE.md                       ← agent working-context (read for engineering)
    ├── AGENT_TOUR.md                   ← this file
    ├── EXPERIMENTAL.md                 ← quarantined-code index
    ├── README.md                       ← GitHub landing page
    ├── docs/                           ← public docs
    │   ├── algorithm.md                ← lever-by-lever narrative
    │   ├── theory.md                   ← parent-rule math: defs, theorems, proofs
    │   ├── bottlenecks.md              ← LIVING bottleneck profile + dead list
    │   ├── benchmarking.md             ← measurement runbook
    │   └── {performance,reproducing,cluster-deployment,references}.md
    ├── src/                            ← Python package dir (PEP src/ layout)
    │   └── doubly_even/                ← production Python (calls Rust kernel)
    │       ├── spec/                   ← reference impl (no perf work)
    │       ├── canon/                  ← canonicalisation; dispatches to Rust
    │       ├── enumerate/              ← canonical-augmentation recursion
    │       └── clean/                  ← pedagogical Python (standalone, 651 LOC)
    ├── rust/                           ← production kernel (Cargo workspace)
    │   ├── Cargo.toml + src/lib.rs     ← thin pyo3 wrapper (root package)
    │   ├── .cargo/config.toml          ← x86-64-v3 codegen (cwd-discovered!)
    │   └── core/                       ← doubly-even-core: ALL algorithms
    │       ├── src/enumerate/{worker,cache,stats,drivers}.rs
    │       ├── src/{parent_rule,orbit,canon,seeder_pool,…}.rs
    │       └── src/experimental/       ← dormant audit substrate
    ├── lean/                           ← Lean 4 formal spec (scaffold only)
    │   ├── README.md
    │   ├── lakefile.lean
    │   └── DoublyEven/{Vectors,Code,Equivalence,FFI}.lean
    ├── scripts/
    │   ├── bench.py                    ← oracle-checked benchmark
    │   ├── install-kernel.sh           ← wheel build+install (cd's into rust/)
    │   ├── microbench/                 ← perf bins; link doubly-even-core directly
    │   ├── gcp-{setup,bench}.sh        ← cloud reproducer
    │   └── experimental/               ← audit scripts
    └── tests/                          ← 590 tests (~48 s with --run-slow)
```

---

## 9. Where the formal spec ends — terminal node

If you read only one file to understand what we mean by "doubly-even
binary linear code of length `N` and dimension `k` up to permutation
equivalence", read [`lean/DoublyEven/Code.lean`](lean/DoublyEven/Code.lean).
The core lines:

```lean
abbrev BinVec (N : ℕ) : Type := Fin N → ZMod 2

def Code (N k : ℕ) : Type :=
  { C : Submodule (ZMod 2) (BinVec N) // Module.finrank (ZMod 2) C = k }

def IsDoublyEven (C : Code N k) : Prop :=
  ∀ v ∈ (C : Submodule (ZMod 2) (BinVec N)), hammingWt v % 4 = 0
```

Plus [`lean/DoublyEven/Equivalence.lean`](lean/DoublyEven/Equivalence.lean):

```lean
def Equivalent (C C' : Code N k) : Prop :=
  ∃ σ : Equiv.Perm (Fin N), permAct σ C.1 = C'.1
```

If your interpretation of "doubly-even code" / "equivalence" doesn't
match these, you are working from a different definition than this
project does. The math is in the types; you can verify your
understanding statically.

---

## 10. Last words

If you find a contradiction between this tour and the code, **the code
wins, and please update this tour**. This document is a roadmap; the
implementations are the truth. If you find a contradiction between this
tour and the auto-memory entries, **the memory entries win** for facts
about what was tried and why; this tour wins for orientation. The
markdown architecture/* notes are the canonical writeup of each lever.
