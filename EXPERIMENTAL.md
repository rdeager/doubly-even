# Quarantined / experimental code

This file indexes everything in the repo that is **dormant, ablation
substrate, or a deferred-direction prototype** — kept in tree so future
contributors (or future-us) can revisit any closed optimisation attempt
without having to reconstruct the post-mortem from scratch.

The active hot path lives in:

- `src/doubly_even/spec/` — readable executable spec.
- `src/doubly_even/canon/` (excluding `experimental/`) — the
  canonicaliser dispatcher (`nauty.py`) + spine primitives.
- `src/doubly_even/enumerate/` (excluding `experimental/`) — the
  canonical-augmentation recursion + Q_C-coordinate pipeline.
- `rust/src/` — the Rust kernel; see "Rust Cargo features" below for
  which modules are dormant.

Everything below is quarantined.

---

## Feulner D9 (column-side partition-refinement reference oracle)

- Python: `src/doubly_even/canon/experimental/feulner.py`
- Rust: `rust/src/feulner.rs`, `rust/src/feulner_clb.rs`
- Reached only via `DOUBLY_EVEN_CANON_BACKEND=feulner` env var; the
  spine `canon/nauty.py` lazily imports the module under that flag.

**What was tried.** A full Python+Rust port of Feulner's column-side
partition-refinement canonicaliser (Feulner §§3–5.2 including the
complete labelled branching / Lemma 5.9 topological-sort prune).
Intended as a possible replacement for nauty's bipartite path.

**Why parked.** Per-call cost is structurally ~6× slower than
sparsenauty's Q_D bipartite at every measured `(N, k)`. The Q_D graph
is sublinear in `2^k / |C_low|` while Feulner is linear in `k · n`, so
the dispatch ratio gets worse with `k`, not better. No bucket-size
threshold flipped the dispatch.

**Where to read.**
- `~/.claude/projects/-workspace-src/memory/project_feulner_dispatch_closed.md`
  — 2026-05-20 close on the dispatch question.
- `~/.claude/projects/-workspace-src/memory/project_feulner_clb_implemented.md`
  — CLB + Lemma 5.9 port that finished off the "improve Feulner to be
  competitive" line.

**Why we kept it.** Reference oracle for the Rust kernel + substrate
for `paired_iso` (Leon §10(i) builds on Feulner's partition
primitives). If the canonicaliser per-call cost ever becomes the
bottleneck again at very large N > 28, Feulner is the cleanest
starting point.

---

## Paired-iso D12 (Leon §10(i) verifier)

- Python: `src/doubly_even/canon/experimental/paired_iso.py`
- Rust: `rust/src/paired_iso.rs`
- Cargo feature `equivalence_verifier` (default OFF). Spine dispatch
  only when the feature is enabled.

**What was tried.** A "cheap" paired-isomorphism prefilter that, when
the secondary cache has a matching weight-enumerator bucket, tries to
verify equivalence without calling nauty.

**Why parked.** Per-probe cost explodes 43 → 68 → 363 µs at
N = 18 → 20 → 22. Even with perfect singleton buckets the full path
is 19× slower than the production canon at N=22 (see
`project_verifier_dormant.md`, 2026-05-20). Recovery would need a
flat-array Feulner rewrite + static refinement tree to drop per-probe
to ~40 µs.

**Where to read.**
- `~/.claude/projects/-workspace-src/memory/project_verifier_dormant.md`

**Why we kept it.** The Rust prefilter + Python prototype together
form a worked example of the Leon §10(i) algorithm against the
doubly-even shape; useful audit substrate if anyone revisits the
"avoid nauty calls" line.

---

## WL / T11 / T12 / T13 cheap invariants (research substrate)

- Rust: `rust/src/invariants.rs`
- Bench scripts: `scripts/experimental/wl_collision_experiment.py`,
  `scripts/experimental/collision_experiment*.py`,
  `scripts/experimental/cubic_tensor_experiment.py`,
  `scripts/experimental/pair_gram_class_audit.py`.

**What was tried.** A family of permutation-invariant fingerprints
(1-WL hash, weight-multiset, T11/T12/T13 stacked statistics, cubic
tensor, pair-gram) intended as cheap pre-rejects for the canon cache.

**Why parked.** Every cheap invariant tested has tuple collisions that
grow rapidly with N (thousands at N = 26 for the strongest tested
candidates). The "cache the invariant, skip nauty on mismatch" angle
requires offline per-N blocklist generation that defeats the
single-pass enumeration constraint.

**Where to read.**
- `~/.claude/projects/-workspace-src/memory/project_1wl_collision_experiment.md`
  — 1-WL completeness audit at N=22, 24, 26.
- `~/.claude/projects/-workspace-src/memory/feedback_no_offline_blocklist.md`
  — the permanent rule that closes this entire family.
- `~/.claude/projects/-workspace-src/memory/project_bitpar_refine_microbench.md`
  — bit-parallel WL refine vs sparse nauty (Rust ratio 1.2×, below
  the 1.5× gate; 17× slower than nauty's C).

---

## Witt phase-(b) scaffolding (Q_C orbit-min alternative)

- Python: `src/doubly_even/enumerate/experimental/witt.py`,
  `src/doubly_even/enumerate/experimental/quotient_witt.py`,
  `src/doubly_even/canon/experimental/schreier_sims_gl.py`.

**What was tried.** A drop-in alternative to the σ_Q lookup-table
orbit-min that applies each generator via `mat_apply` on every BFS
step, skipping the `2^L`-per-generator table build.

**Why parked.** Pure-Python `mat_apply` per-step cost outweighs the
`O(2^L)` table-build savings at every measured L. Hard-wired
`use_witt_path(...) → False` in the experimental module marks the
single decision point a native-code port could re-tune.

**Where to read.**
- `~/.claude/projects/-workspace-src/memory/phase_b_empirical_finding.md`

---

## Sage daemon canon proxy (benchmark only)

- Python: `src/doubly_even/canon/experimental/sage_proxy.py`
- Reached via `DOUBLY_EVEN_CANON_BACKEND=sage_partn_ref`.

**What it is.** Long-lived Sage subprocess that proxies
`LinearBinaryCodeStruct` partition-refinement. Used to compare wall
time vs Sage's `self_orthogonal_binary_codes`.

**Why parked.** Not a perf path — Sage is ~300× slower than our
kernel at N=22. Kept so the comparison number is reproducible.

**Where to read.**
- `~/.claude/projects/-workspace-src/memory/project_sage_comparison.md`

---

## Engine A multiset / Fourier-domain prototype (deferred direction)

- Python: `scripts/experimental/multiset_enum.py`,
  `scripts/experimental/multiset_fourier.py`.

**What it is.** Python prototype of the §5 column-multiset (Fourier
domain) enumeration. Target: the `N = 32, k ≤ 4` slice that the
existing canonical-augmentation pipeline cannot reach (σ_Q BFS dies
at `N ≥ 28` for small k).

**Status.** Not parked — **deferred**. The Rust port is the
unblocked next deliverable for the `k ≤ 4, N = 32` frontier; the
Python prototype is too slow to be useful directly but is the only
in-tree implementation of the Fourier-domain framing.

**Where to read.**
- `~/.claude/projects/-workspace-src/memory/project_small_k_engine.md`
- `/workspace/markdown/architecture/06-scaling-frontier.md`

---

## Rust Cargo features (audit substrate)

All in `rust/Cargo.toml`. Each is OFF by default and acts as a
self-contained quarantine of one closed audit direction.

| Feature | Module(s) | What | Why parked |
|---------|-----------|------|-----------|
| `equivalence_verifier` | `paired_iso.rs` | D12 verifier dispatch | See "Paired-iso D12" above |
| `dense_qd` | `canon.rs` (densenauty path) | Audit Q6: dense Q_D | +39–42 % at N=22 |
| `dense_qd_tc0` | `canon.rs` | Audit Q6: zero target-cell | +42 % at N=22 |
| `dense_qd_refinvar` | `canon.rs` | Audit Q6: refinvar FFI | +260 % at N=22 |
| `traces_qd` | `canon.rs` (Traces path) | Audit Q6: Traces vs sparsenauty | +2 %, within noise; conflicts with `parallel` |
| `nauty_hist` | `enumerate.rs` (statsblk histogram) | Research-phase instrumentation | Off by default; zero overhead when off |

The conflict guard `parallel ↔ traces_qd` lives at
`rust/src/enumerate.rs:38-42` (Traces uses non-TLS static work
queues; not thread-safe under HAVE_TLS=1).

**Where to read.**
- `~/.claude/projects/-workspace-src/memory/project_q6_audit_closed.md`
- `/workspace/markdown/expert-review/05-nauty-traces-audit.md`

---

## Runtime ablation knobs (re-measurement substrate)

| Env var | What it disables | When set |
|---------|-----------------|----------|
| `DOUBLY_EVEN_NO_MASS_STOP=1` | The D5 mass-stop branches in `WorkerState::traverse` (sequential path only; parallel workers already get `u128::MAX` quota) | Step 0 of this refactor; 4-11 % regression confirmed |
| `DOUBLY_EVEN_SECONDARY_CACHE_INSTRUMENTATION=1` | (Inverse — *enables* the secondary weight-enum bucket cache; off in default builds without the `equivalence_verifier` feature) | Verifier-mode plumbing |
| `DOUBLY_EVEN_CANON_CACHE_CAP=<N>` | LRU canon cache capacity (default 500 000) | Memory-bound at N ≥ 26 |
| `DOUBLY_EVEN_THREADS=<N>` | Worker count for the parallel kernel | Active perf knob |
| `DOUBLY_EVEN_FRONTIER_DEPTH=<N>` | Outer-DFS seed depth (default 4) | Tune at N ≥ 24 |
| `DOUBLY_EVEN_CANON_BACKEND=feulner` / `=sage_partn_ref` | Spine canonicaliser dispatch | Opt into a quarantined backend for cross-check |

---

## Tests that exercise quarantined code

- `tests/experimental/test_witt.py` — closed-form singular counts.
- `tests/experimental/test_quotient_witt.py` — Witt orbit-min vs
  F_2^N oracle.
- `tests/experimental/test_kernel_witt.py` — Rust ↔ Python cross-check
  for `aut_orbit_minima_q_witt`.
- `tests/experimental/test_schreier_sims_gl.py` — GL(L, F_2)
  Schreier–Sims oracle tests.
- `tests/test_feulner.py`, `tests/test_feulner_refine_incremental.py`,
  `tests/test_paired_iso.py`, `tests/test_canon_info_reconstruction.py`
  — Feulner / paired-iso tests that exercise the quarantined modules
  through their public APIs; kept under `tests/` rather than
  `tests/experimental/` because they pin invariants that any future
  Rust port has to honour.

All run by default with `uv run pytest`.
