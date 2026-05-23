# doubly-even

Enumerate doubly even binary linear codes `[N, k]` up to permutation
equivalence — one canonical representative and `|Aut(C)|` per class,
mass-formula-verified at every step.

To our knowledge this is the first reproducible open-source enumerator
matching [Doran–Faux–Gates–Hübsch–Iga–Landweber–Miller (DFGHILM)
Appendix B](docs/references.md#dfghilm-2011--the-algorithmic-spec)
Table 3 through `N = 28` — the 21,505,546 equivalence classes at
`N = 28` reproduced in 61 minutes on a single GCP `c4a-standard-72` VM
(~$3 of on-demand compute) — and the first publicly reproducible
enumeration at `N = 29` (**239,465,540 equivalence classes** in
**12.3 hr** on the same VM, ~$35; mass-formula certified at every
rank, see [`docs/results/n29.json`](docs/results/n29.json)).

## Prior work

The algorithm is DFGHILM Appendix B (2011); the canonicaliser is
`sparsenauty` (McKay–Piperno 2014); the mass formula is Gaborit (1996);
the canonical-augmentation framework is McKay (1998). Bouyukliev and
Bouyuklieva's [arXiv:1907.10363](https://arxiv.org/abs/1907.10363)
(2019) is an independent column-by-column canonical-augmentation engine
with `[N, k, ≥ d]` validation counts at `N = 31, 32`. Sage's
`self_orthogonal_binary_codes` is the prior open-source bar
(single-threaded Cython; we run ~525× faster at `N = 22`).
[Robert L. Miller](https://rlmiller.org/de_codes/) — one of the
DFGHILM authors — maintains an independent reference enumeration
(no-zero-column convention) that cross-checks our `N = 28` result.

Detailed credits and validation oracles in
[`docs/references.md`](docs/references.md). DFGHILM's original
enumeration ran on the OSU Glenn supercomputer in 2011; sixteen years
of hardware and the algorithmic improvements documented in
[`docs/algorithm.md`](docs/algorithm.md) put their `N ≤ 28` results
within reach of a desktop or a single cloud VM.

## Doubly even codes

A **binary linear code** of length `N` is a subspace `C ⊆ F₂^N`; its
codewords are 0/1 strings of length `N` that form a vector space under
bitwise XOR. The code is

- **even** if every codeword has Hamming weight `≡ 0 (mod 2)`,
- **doubly even** if every codeword has Hamming weight `≡ 0 (mod 4)`,
- **self-dual** if `C = C⊥` under the standard inner product
  (forces `N = 2k`),
- **Type II** if it is doubly even *and* self-dual (forces `8 | N`).

Two codes are *equivalent* if one is obtained from the other by
permuting the `N` coordinate positions. The classification problem
solved by this package is: list one representative per equivalence
class of doubly even `[N, k]` codes, together with its automorphism
group `Aut(C) ≤ S_N`.

Type II codes are classical — they contain the extended Hamming
`[8, 4]`, the binary Golay `[24, 12]`, and the second-order
Reed–Muller codes `RM(2, m)`. The Gaborit mass formula gives a
closed-form labelled count

    σ(N, k) = (number of doubly even [N, k] codes, not modded out)

which doubles as a stopping certificate: enumeration is complete iff
`Σ_{C in classes} N! / |Aut(C)| = σ(N, k)`.

## Adinkras and supersymmetry

The motivation is one-dimensional `N`-extended worldline
supersymmetry: a quantum-mechanical toy model where time is the only
spacetime coordinate but there are `N` real supercharges
`Q_1, …, Q_N` satisfying

    {Q_I, Q_J} = 2 δ_{IJ} ∂_t,    [Q_I, ∂_t] = 0.

An **Adinkra** (Faux–Gates; developed by DFGHILM) is a graph encoding
a multiplet: vertices are component fields (bosons / fermions on the
two halves of a bipartition), edges of color `I ∈ {1, …, N}` record
`Q_I φ_a = ± ψ_i`, plus a dashing for signs and heights for
engineering dimension. Forget the signs and the heights and you are
left with the **chromotopology** — a bipartite multigraph whose edges
are colored by the `N` supercharges. DFGHILM §2 and §5 prove:

> chromotopologies on `2^{k+1}` vertices, up to color-preserving
> graph isomorphism ≡ doubly even codes `[N, k]`, up to column
> permutation.

So enumerating Adinkra chromotopologies is exactly enumerating doubly
even codes. The relevant target is `N ≤ 32`; the labelled count
grows to `σ(32, 10) ≈ 1.6 × 10⁴⁷` with on the order of `10¹²`
equivalence classes. Enumeration to `N = 32` is a real computational
problem.

## What this package does

DFGHILM Appendix B, end-to-end:

- **B.1** — Gaborit mass formula `σ(N, k)`, used as both stopping
  certificate and running consistency check.
- **B.2** — Bipartite-graph encoding `G(C)` (codewords × columns)
  fed to `sparsenauty` for canonical label + `Aut(C)`.
- **B.3** — Doubly-even linear-algebra optimisations (Corollary B.1
  inner-product check + dual-coset reps).
- **B.4** — Canonical augmentation (McKay 1998): the recursion is on
  *augmentations* `(parent, child)`, with a canonical-parent function
  `p` that uniquely picks one ancestry per equivalence class.

Validated against four independent oracles:

1. Gaborit mass formula, on every emitted class.
2. DFGHILM Table 3, cell-for-cell through `N = 28`.
3. Sage `self_orthogonal_binary_codes`, through `N = 22`.
4. [rlmiller.org/de_codes](https://rlmiller.org/de_codes/),
   no-zero-column convention, at `N = 28`.

## The algorithmic levers

Five accelerations on top of DFGHILM Appendix B carry the load. Each
delivered a measurable wall-time reduction; they compose to
~220× over the pure-Python baseline and ~525× over Sage at `N = 22`.

- **Quotient-space orbit-min prefilter** — work in the
  `(N − 2k)`-dimensional quotient `C⊥/C` (Gray-code walk over
  `2^(N − 2k)` reps) instead of the full `2^(N − k)` BFS of B.3.
  The dominant Python-side win. The quotient-space framing, Gray-walk,
  and weight-mod-4 filter overlap with Sage's `binary_code.pyx`; the
  specifically novel sub-technique is precomputed `σ_Q` action tables
  (Gray-code-built, one XOR per entry) plus a single global
  `O(2^L)` sweep that decomposes all orbits at once — not present in
  Sage, DFGHILM B, or Bouyukliev–Bouyuklieva 2019. See
  [`docs/algorithm.md` §1](docs/algorithm.md#1-quotient-space-orbit-min-prefilter)
  for the prior-art comparison.
- **Low-weight-incidence canonicaliser** — feed nauty a sparse
  bipartite graph on `|C_low| + N` vertices (the lowest-weight
  codewords needed to span `C`, plus the `N` columns) instead of the
  full `2^k + N` graph. `1.91×` at `N = 22`, `≥ 2.5×` at `N = 24`.
- **Native Rust kernel** — the whole DFS in Rust (canonical-parent,
  is-canonical-augmentation, canon-info LRU, mass quota), no
  Python↔Rust crossings. `1.32×` at `N = 22`.
- **Outer-DFS worker parallelism with pipelined seeder** — DFGHILM
  B.4's producer-consumer model: seeder pushes seeds into a bounded
  channel as the DFS discovers them, workers consume in parallel,
  per-worker canon caches, shared atomic mass-tracker for per-rank
  early termination. `9.6×` at `N = 22`, `~12×` at `N = 24`,
  `~11.2×` at `N = 26`.
- **Mass-formula early stop** — closed-form `σ(N, k)` plus monotone
  quota check skips the rest of a rank as soon as
  `Σ N!/|Aut| = σ(N, k+1)`. Modest `4–11%` in isolation, but
  conceptually clean.

Detailed writeup with ablation table in
[`docs/algorithm.md`](docs/algorithm.md). The remaining ~90% of
`N = 22` wall is inside `sparsenauty`'s C code — the algorithmic
floor at this graph shape.

## Performance

Mean of 3 runs on a 13700K (8 P-cores × SMT2 + 8 E-cores =
24 logical / 16 physical), parallel Rust kernel:

| `N` | sequential | parallel best          | classes  |
|----:|-----------:|-----------------------:|---------:|
|  20 |    0.97 s  | 0.22 s (t=16, d=4)     |    1,211 |
|  22 |    6.64 s  | **0.691 s** (t=20, d=4)|    5,118 |
|  24 |   ~107 s   | **8.90 s** (t=24, d=5) |   37,496 |
|  26 |       —    | **170 s** (t=24, d=5)  |  494,272 |
|  28 |       —    | **3669 s** (61.2 min) (t=72, d=5, GCP c4a-72) | 21,505,546 |
|  29 |       —    | **44 356 s** (12.3 hr) (t=72, d=5, GCP c4a-72) | 239,465,540 |

Sage `self_orthogonal_binary_codes` at `N = 22` runs in 363.85 s
single-threaded (Cython, GIL-bound). End-to-end ratio:
`0.691 / 363.85 ≈ 525×`.

At `N = 29` the kernel made **87.2 billion canon calls** at a mean
**29.5 µs/call** — equal to **~364 canon calls per emitted
equivalence class**. The calls-per-class ratio is the dominant
scaling factor at this point: it grew from 15.6 at `N = 22` to 33.7
at `N = 24` to 91.2 at `N = 26` to 249 at `N = 28`. Per-call cost is
*dropping* with `N` (the canonicaliser exploits more structure in
larger codes), but the canonical-augmentation candidate count grows
much faster than the class count. Any future >2× algorithmic
improvement has to reduce the *number* of canon calls, not the cost
per call — that's already inside sparsenauty's C floor.

Full table and tuning knobs in
[`docs/performance.md`](docs/performance.md).

## Quick start

```sh
# Python deps
uv sync --all-extras --dev

# Rust kernel (parallel)
maturin build --release --features parallel -m rust/Cargo.toml
uv pip install --force-reinstall rust/target/wheels/doubly_even_kernel-*.whl

# Test suite (568 tests, ~7 s default; ~10 s with --run-slow)
uv run pytest

# Benchmark at N = 22, parallel
DOUBLY_EVEN_THREADS=20 uv run python scripts/bench.py --label par-t20 --N 22
```

Programmatic use:

```python
from doubly_even.enumerate.augment import enumerate_doubly_even

for ec in enumerate_doubly_even(8):
    print(f"k={ec.code.rank} |Aut|={ec.aut_order} basis={list(ec.code.basis)}")
```

Each yielded `EnumeratedCode` carries a canonical representative and
`|Aut(C)|`. Full reproducibility recipe (including the N=28 cloud
run) in [`docs/reproducing.md`](docs/reproducing.md).

## Long-running jobs (local or cloud)

For `N ≥ 26` the in-memory Vec output approach used by `bench.py`
becomes memory-heavy and (at `N ≥ 29`) infeasible. The streaming
output path writes per-worker binary files to a local directory and
runs the mass-formula gate in-Rust, with peak RSS dominated by canon
caches:

```sh
# Long-running kernel (locally or in a cloud VM):
DOUBLY_EVEN_THREADS=24 DOUBLY_EVEN_FRONTIER_DEPTH=5 \
    DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
    uv run python scripts/run_streaming.py --N 26 \
    --output-dir /tmp/n26-out

# In another terminal, sidecar that polls progress:
uv run python scripts/stream_progress.py --N 26 \
    --output-dir /tmp/n26-out --interval 30
```

The sidecar prints a per-`k` mass-vs-σ table at the configured
interval and exits when the kernel finishes. It's equally useful for a
30-minute local `N = 24` run and a multi-hour cloud `N = 28` run — the
output directory is the only thing that changes.

On a developer-class Apple Silicon laptop (M5 / M5 Pro, 64 GB unified
memory, ARM64) the streaming path is expected to handle `N = 27` in
under half an hour and `N = 28` overnight at
`CANON_CACHE_CAP ≈ 200000`; per-thread throughput on M5 P-cores is
competitive with the 13700K and 64 GB of unified memory fits the
per-worker LRU comfortably. **We have not run on Apple Silicon
ourselves** — these are extrapolations from the 13700K (16 physical)
and c4a-72 (72 Neoverse V2) measurements, not measurements. The same
target-conditional `Cargo.toml` patch used on `c4a` (popcnt feature
disabled on aarch64) is already on `main` and is the only build
adjustment required.

For multi-node / >256-core deployment hints, see
[`docs/cluster-deployment.md`](docs/cluster-deployment.md) — we
haven't tested the kernel above 72 single-NUMA cores, so this doc is
a code-pointer design sketch rather than a working cluster
implementation.

## Status

- Phases 0–3, Milestones 4–5, all D-series optimisation sprints
  complete.
- `N ≤ 26` reproducible on a 13700K desktop in seconds-to-minutes;
  on an M5 / M5 Pro MacBook with 64 GB RAM (prediction, not measured)
  expect a similar ladder with per-thread throughput competitive with
  the 13700K.
- `N = 27` ~20 min on a 16-core desktop or M5-class laptop
  (predicted from cloud measurement: 6.2 min on c4a-72 with 72
  threads, scaled to laptop core count and per-thread throughput).
- `N = 28` reproducible on GCP `c4a-standard-72` in 61 minutes (~$3);
  predicted to fit overnight on a 64 GB M5 Pro using the streaming
  output path.
- **`N = 29` complete** — 239,465,540 equivalence classes in 12.3 hr
  on `c4a-standard-72` (~$35 of compute); the first publicly
  reproducible enumeration at this length. Mass-formula certified
  at every rank `k = 0..13`. Full per-`k` table and certificate in
  [`docs/results/n29.json`](docs/results/n29.json). At current
  per-thread throughput, `N = 29` is past single-laptop wall
  budgets — the c4a-72 / cloud-VM target.
- `N ≥ 30` requires either a much bigger single machine
  (`c4-standard-288-metal` or similar) or a small cluster. The
  per-node streaming-output path is shipped; the cross-node
  coordinator is not.
- `N ≤ 22` is exhausted at the pure-algorithmic level with this
  canonicaliser — sparsenauty per-call cost is the floor at this
  graph shape.

## Project layout

Three Python layers (depend only on the layer above), plus the Rust
kernel:

- `src/doubly_even/spec/` — executable specification: `Code`,
  `BinVec`, `is_doubly_even`, Gaborit mass formula.
- `src/doubly_even/canon/` — wrapper around `sparsenauty` (via the
  Rust kernel) with `pynauty` as a fallback.
- `src/doubly_even/enumerate/` — canonical-augmentation search loop
  and quotient-space pre-canonical filters. When the Rust kernel is
  installed, the entire recursion runs in Rust.
- `rust/` — Rust kernel (built with `maturin`). Sparsenauty via
  `nauty-Traces-sys`; the producer-consumer parallel kernel via
  `crossbeam-channel`.

Dormant / experimental / audit-substrate code is quarantined under
`*/experimental/` subpackages — indexed in
[`EXPERIMENTAL.md`](EXPERIMENTAL.md).

## Opt-in branches

Two branches carry small, low-risk improvements that are not on
`main` so users can choose:

- **[`feature/dfs-order-by-aut`](https://github.com/rdeager/doubly-even/tree/feature/dfs-order-by-aut)**
  — sort sibling children by `|Aut(D)|` descending in the DFS.
  Measured wins on the 13700K: `-12 %` at `N = 24`, `-10 %` at
  `N = 25`, `-8 %` at `N = 26`. Regresses at `N ≤ 22` (small-N
  shape; mass-stop has less room to bite when |Aut| is already
  large). Enabled via `DOUBLY_EVEN_DFS_ORDER=aut_desc` (or the
  branch's default), reverted with `DOUBLY_EVEN_DFS_ORDER=off`.
- **[`feature/dfs-speedup-bliss-pq`](https://github.com/rdeager/doubly-even/tree/feature/dfs-speedup-bliss-pq)**
  — `feature/dfs-order-by-aut` plus a cross-worker priority-queue
  pull at the depth-5 seeder frontier. Additional `-2 %` at
  `N = 24, 26` on top of `aut_desc`. Toggled via
  `DOUBLY_EVEN_SEED_ORDER=fifo` (the priority pull is the default on
  the branch).

Each branch can be checked out and measured directly:
`git checkout feature/dfs-order-by-aut && maturin build --release …`.

## License

MIT. See the `license` field in
[`rust/Cargo.toml`](rust/Cargo.toml); the Python package inherits the
same terms.

## Citation

If you use this in academic work, please cite DFGHILM (the algorithmic
spec) and this repository. See [`CITATION.cff`](CITATION.cff) for the
machine-readable citation pinned to the `N = 29` build SHA, including
references to DFGHILM 2011, Gaborit 1996, McKay–Piperno 2014, and
McKay 1998. The `N = 29` result certificate is at
[`docs/results/n29.json`](docs/results/n29.json).
