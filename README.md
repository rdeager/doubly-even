# doubly-even

Enumerate doubly even binary linear codes `[N, k]` up to permutation
equivalence — one canonical representative and `|Aut(C)|` per class,
verified against the Gaborit mass formula on every step.

To our knowledge, this is the first reproducible open-source replication
of the DFGHILM Appendix B Table 3 counts through `N = 26`.

## Doubly even codes

A **binary linear code** of length `N` is a subspace `C ⊆ F₂^N`; its
codewords are 0/1 strings of length `N` that form a vector space under
bitwise XOR. The code is

- **even** if every codeword has Hamming weight `≡ 0 (mod 2)`;
- **doubly even** if every codeword has Hamming weight `≡ 0 (mod 4)`;
- **self-dual** if `C = C⊥` under the standard inner product (forces `N = 2k`);
- **Type II** if it is doubly even *and* self-dual (forces `8 | N`).

Two codes are *equivalent* if one is obtained from the other by permuting
the `N` coordinate positions. The classification problem is: list one
representative per equivalence class of doubly even `[N, k]` codes,
together with its automorphism group `Aut(C) ≤ S_N`.

Type II codes are a classical object of coding theory — they contain the
extended Hamming `[8, 4]`, the binary Golay `[24, 12]`, and the
second-order Reed–Muller codes `RM(2, m)` — and they are exactly the
even self-dual lattices' mod-2 reductions, by Construction A. The mass
formula due to Gaborit gives a closed-form labelled count

    σ(N, k) = (number of doubly even [N, k] codes, not modded out)

which acts as a stopping certificate: enumeration is complete iff
`Σ_{C in classes} N! / |Aut(C)| = σ(N, k)`.

## Adinkras and supersymmetry

The motivation we follow comes from one-dimensional **`N`-extended
worldline supersymmetry** — a quantum-mechanical toy model where time is
the only spacetime coordinate but there are `N` real supercharges
`Q_1, …, Q_N` satisfying the Clifford-like relations

    {Q_I, Q_J} = 2 δ_{IJ} ∂_t,    [Q_I, ∂_t] = 0.

A *multiplet* is a representation: a finite list of real component fields
split into bosons `φ_a` and fermions `ψ_i`, such that each `Q_I` exchanges
bosons with fermions, the algebra above closes on derivatives, and the
fields balance dimensionally.

An **Adinkra**, introduced by Faux–Gates and developed by
Doran–Faux–Gates–Hübsch–Iga–Landweber–Miller (**DFGHILM**), is a graph
that draws this data:

- one vertex per component field; bosons and fermions form the two halves
  of a bipartition;
- one edge of *color* `I ∈ {1, …, N}` between `φ_a` and `ψ_i` whenever
  `Q_I φ_a = ± ψ_i` (or vice versa);
- a *dashing* on each edge that records the sign;
- a *height* on each vertex that records the engineering dimension (each
  `Q_I` step bumps it by `½`).

Forget the signs and the heights and you are left with the
**chromotopology** — a bipartite multigraph whose edges are colored by the
`N` supercharges. The central observation (DFGHILM §2 and §5) is that
chromotopologies are extremely rigid: every connected one on `2^{k+1}`
vertices is the Cayley graph of the quotient group `Z_2^N / C`, where
`C ⊆ Z_2^N = F_2^N` is some `k`-dimensional binary linear code, and the
algebra `{Q_I, Q_J} = 2δ_{IJ} ∂_t` closes consistently **iff `C` is doubly
even**. So:

> chromotopologies on `2^{k+1}` vertices, up to color-preserving graph
> isomorphism
> ≡  doubly even codes `[N, k]`, up to column permutation.

Geometrically, `k = 0` is the bare `N`-cube on `2^N` vertices (the
fundamental "isoscalar" multiplet); each additional codeword identifies a
pair of opposite faces, collapsing the cube by another factor of two. The
Golay code `[24, 12]` collapses the 24-cube down to a `2^{12} = 4096`-
vertex chromotopology — this is the largest doubly-even Adinkra that fits
inside `N ≤ 32`.

DFGHILM enumerated 60,000+ codes by the time the paper was written but
did not release software. The relevant target is `N ≤ 32`: the labelled
count grows to `σ(32, 10) ≈ 1.6 × 10⁴⁷`, with on the order of `10¹²`
equivalence classes. Enumeration to `N = 32` is a real computational
problem, not a textbook exercise.

## How the enumerator works

The naïve approach — list every doubly even code, canonicalise each, dedupe
the canonical forms — is hopeless past about `N = 12`: even at `N = 22`
the labelled count is `σ(22, 7) ≈ 1.3 × 10¹⁵`. The enumerator instead
follows **McKay's canonical augmentation** (1998), which builds codes
incrementally and decides at each step whether the *augmentation* is
canonical, pruning entire subtrees that aren't.

The high-level recursion is

```
enumerate(N):
    traverse(zero_code(N))

traverse(C):
    emit (C, |Aut(C)|)
    for v in C⊥ \ C with wt(v) ≡ 0 (mod 4):       # doubly-even-preserving extensions
        D := span(C, v)
        if v is the minimum of its Aut(C)-orbit:   # cheap orbit filter
            if (C, D) is the canonical augmentation of D:   # expensive nauty call
                traverse(D)
```

The recursion is depth-first; at depth `k` we are looking at `[N, k]`
codes. Two filters carry the load:

1. **Orbit minimum (cheap).** A child code `D = span(C, v)` is determined
   by the new row `v ∈ C⊥`. Two choices `v, v'` give equivalent children
   if they lie in the same `Aut(C)`-orbit on `C⊥ \ C`; we keep only the
   lexicographic minimum of each orbit. `Aut(C)` is already in hand from
   canonicalising `C`, so this filter is essentially free.
2. **Canonical augmentation (expensive).** Each `D` of dimension `k+1`
   has many `k`-dimensional doubly-even subcodes — any of them could be
   "the parent". McKay's trick is to fix one parent per code by the rule
   *the canonical parent is what you get by canonicalising `D` and
   dropping the last row of the canonical generator matrix*. We accept
   the edge `(C, D)` only if `C` matches that canonical parent (up to
   `Aut(D)`). Every equivalence class then has exactly one ancestry to
   the zero code, so it is emitted exactly once.

Canonicalising `D` is the hot operation. We do it by building the
**bipartite incidence graph** `G(D)` — codewords on one side, the `N`
column indices on the other, with edges where codewords have a `1` — and
handing it to nauty (specifically the sparse variant, via
`nauty-Traces-sys`). The column-side stabiliser of `Aut(G(D))` is exactly
the permutation automorphism group `Aut(D)`, so one nauty call delivers
both the canonical label and `|Aut(D)|`. The bipartite encoding is what
DFGHILM Appendix B.2 prescribes; everything in this repo other than
nauty is open-coded.

Two further accelerations are essential at the scales this repo targets:

- **`Q_C`-coordinate orbit-min prefilter.** Computing each candidate `v`
  in the full `F_2^N` and then orbit-reducing is wasteful. Instead we
  work in the `(N − 2k)`-dimensional quotient space `Q_C = C⊥ / C` where
  the orbit reduction lives, looking up minima via precomputed
  `σ_Q`-tables. This is the dominant Python-side optimisation
  (`enumerate/quotient.py`).
- **Mass-formula early stop.** `σ(N, k)` is known in closed form
  (Gaborit). As soon as the running sum `Σ N!/|Aut(C_i)|` over emitted
  classes reaches `σ(N, k)` for the current depth, we have provably hit
  every equivalence class and can short-circuit the rest of the tree.

For `N ≥ 22` the bipartite-graph nauty call dominates wall time; the
parallel kernel splits the DFS at a configurable cut depth and processes
each subtree on its own worker thread, with per-worker canon caches and a
shared atomic mass-stop tracker. The full per-lever optimisation history
(D1–D13-V5) lives in the design-docs tree referenced under
[Project documentation](#project-documentation).

## What this package implements

**DFGHILM Appendix B**, end-to-end. The appendix is a short, self-contained
algorithm description; this repo turns it into a working enumerator.

- **B.1 — Gaborit mass formula.** Closed form for `σ(N, k)`, the labelled
  count of doubly even `[N, k]` codes. Used both as a stopping certificate
  and as the internal consistency check `Σ N!/|Aut(C_i)| = σ(N, k)`.
- **B.2 — Bipartite-graph encoding `G(C)`.** Codewords on one side,
  columns on the other; the column-side stabiliser of `Aut(G(C))` is the
  permutation automorphism group of `C`. Canonical labels and `Aut`
  generators come from a single nauty call (via `nauty-Traces-sys` in the
  Rust kernel).
- **B.3 — Doubly-even linear-algebra optimisations.** Corollary B.1
  reduces the doubly-even predicate on a generated code to `O(k²)`
  pairwise inner-product checks (`⟨v_i, v_j⟩ = 0` plus `wt(v_i) ≡ 0 mod 4`),
  avoiding the `2^k` codeword sweep. A `Q_C`-coordinate orbit-min
  precanonical filter narrows the augmentation candidate set further.
- **B.4 — McKay 1998 canonical augmentation.** The recursion is on
  *augmentations* `(parent, child)`, not codes alone. A child is emitted
  iff its augmentation agrees with the parent function `p` derived from
  the canonical labeller — that property uniquely picks one ancestry per
  equivalence class, so every class is emitted exactly once.

Correctness is checked against three independent oracles:

1. **Gaborit mass formula** — internal consistency on every emitted class.
2. **DFGHILM Table 3** — published equivalence-class counts; matched
   cell-for-cell through `N = 26`.
3. **Sage** `self_orthogonal_binary_codes` — agrees through `N = 22`.

## Performance

Headline numbers on a 13700K (8 P-cores × SMT2 + 8 E-cores = 24 logical /
16 physical), parallel Rust kernel, mean of 3 runs:

| `N` | sequential | parallel (best)        | classes  |
|----:|-----------:|-----------------------:|---------:|
|  20 |    0.97 s  | 0.22 s (t=16, d=4)     |    1,211 |
|  22 |    6.64 s  | **0.69 s** (t=20, d=4) |    5,118 |
|  24 |   ~107 s   | **8.90 s** (t=24, d=5) |   37,496 |
|  26 |       —    | **170 s** (t=24, d=5)  |  494,272 |

At `N = 22`, end-to-end ~525× faster than Sage's
`self_orthogonal_binary_codes` (363.85 s, single-threaded). Cloud-validated
on GCP `c4-standard-24` (Emerald Rapids 8581C, us-east4): `N = 26` in
285 s, DFGHILM cells match exactly, zero algorithmic surprise — per-thread
wall is pure clock-ratio slower than the 13700K.

A per-lever writeup, sign-off retrospective at `N ≤ 22`, and `N ≥ 28`
scaling forecast live in a separate design-docs tree (not currently
published; see [Project documentation](#project-documentation) below).

## Layout

Three layers (depend only on the layer above):

- `doubly_even.spec` — executable specification (math, readable).
- `doubly_even.canon` — wrapper around an external canonical labeller.
  Active backend is the Rust kernel (`doubly_even_kernel`) calling
  sparsenauty via `nauty-Traces-sys`; pure-Python `pynauty` is kept as a
  fallback.
- `doubly_even.enumerate` — the canonical-augmentation search loop and
  pre-canonical filters. When the Rust kernel is installed, the entire
  recursion runs in Rust.

The Rust kernel lives under [`rust/`](rust/) and is built with `maturin`.
Dormant / experimental / audit substrate is quarantined under
`*/experimental/` subpackages and documented in
[`EXPERIMENTAL.md`](EXPERIMENTAL.md).

## Build and run

Requires Python 3.12+, Rust toolchain (rustup), and `uv`.

```sh
# Python deps
uv sync --all-extras --dev

# Build the Rust kernel and install the wheel into the project's venv
maturin build --release -m rust/Cargo.toml \
  && uv pip install rust/target/wheels/doubly_even_kernel-*.whl

# Tests (568 collected; ~7 s default, ~10 s with --run-slow)
uv run pytest
uv run pytest --run-slow

# Benchmark
uv run python scripts/bench.py --label baseline
```

For the parallel kernel (opt-in feature flag):

```sh
maturin build --release --features parallel -m rust/Cargo.toml \
  && uv pip install --force-reinstall rust/target/wheels/doubly_even_kernel-*.whl

# N ≤ 22: leave 2 cores headroom (t = logical_cores − 2)
DOUBLY_EVEN_THREADS=20 uv run python scripts/bench.py --label par-t20 --N 22

# N ≥ 24: full logical-core count, deeper cut depth
DOUBLY_EVEN_THREADS=24 DOUBLY_EVEN_FRONTIER_DEPTH=5 \
  uv run python scripts/bench.py --label par-t24-d5 --N 24

# N = 26: also cap the per-worker canon cache to avoid OOM
DOUBLY_EVEN_THREADS=24 DOUBLY_EVEN_FRONTIER_DEPTH=5 \
  DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
  uv run python scripts/bench.py --label par-t24-d5-n26 --N 26
```

## Quick example

```python
from doubly_even.enumerate.augment import enumerate_doubly_even

for ec in enumerate_doubly_even(8):
    print(f"k={ec.code.rank} |Aut|={ec.aut_order} basis={list(ec.code.basis)}")
```

Each yielded `EnumeratedCode` carries a canonical representative of an
equivalence class plus `|Aut(C)|`.

## Cloud / GCP

The kernel ports cleanly to GCP Emerald Rapids; bootstrap and bench
scripts in tree work unchanged on `c4-standard-24`, `c4-standard-48`, and
`c4-288-metal`:

```sh
# Fresh Ubuntu 24.04 VM, one-liner bootstrap (apt + rustup + uv + clone
# + uv sync + maturin --release --features parallel + smoke pytest):
curl -fsSL https://raw.githubusercontent.com/rdeager/doubly-even/main/scripts/gcp-setup.sh \
  | bash -s -- --repo https://github.com/rdeager/doubly-even

# Three-stage bench (Run A seq, Run B t=nproc/2 d=4, Run C t=nproc d=5):
cd ~/doubly-even && scripts/gcp-bench.sh shakedown-c4-24
```

## Project documentation

Long-form design docs live alongside this repo at
[`/workspace/markdown/`](../markdown/) (not yet versioned in this repo):

- `algorithm/` — what the enumerator does, in math.
- `architecture/` — engineering decisions, per-lever writeups, scaling
  forecast.
- `notes/` — paper summaries, cloud-shakedown notes.
- `references/` — bibliography.

## Status

- Phases 0–3 + Milestones 4–5 + D9–D13-V5 sprints complete.
- `N ≤ 22` exhausted at the pure-algorithmic level with this canonicaliser
  (sparsenauty is the algorithmic floor at this graph shape).
- `N ≥ 28` is gated on infrastructure: `N = 28` is reachable today (~1 hr,
  20 threads); `N ≥ 29` needs a streaming-output refactor; `N = 30` needs
  ≥ 256 GB RAM or a small cluster. See `architecture/06-scaling-frontier.md`.

## License

MIT. See the `license` field in [`rust/Cargo.toml`](rust/Cargo.toml); the
Python package inherits the same terms.

## Citation

If you use this in academic work, please cite DFGHILM (the algorithmic
spec) and this repository. A `CITATION.cff` will land before the first
tagged release.
