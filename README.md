# doubly-even

`doubly-even` enumerates doubly even binary linear codes `[N, k]` up to
coordinate permutation. It emits one canonical representative per
equivalence class together with `|Aut(C)|`, and checks each rank against
Gaborit's mass formula.

The current release reproduces the [DFGHILM Appendix
B](docs/references.md#dfghilm-2011--the-algorithmic-spec) Table 3 cells
through `N = 28` — 21,505,546 equivalence classes at `N = 28`, in
61 minutes on a single GCP `c4a-standard-72` VM. It also adds a
mass-certified `N = 29` enumeration: **239,465,540 equivalence
classes** in **12.3 hours** on the same VM, the first publicly
reproducible enumeration at this length. The per-rank certificate is
at [`docs/results/n29.json`](docs/results/n29.json).

## Highlights

- Reproduces DFGHILM Table 3 cell-for-cell through `N = 28`.
- First publicly reproducible `N = 29` enumeration, mass-formula
  certified at every rank `k = 0..13`.
- Emits one canonical representative and `|Aut(C)|` per class.
- Parallel Rust kernel with a streaming-output path for long runs.
- Cross-checked against Gaborit's mass formula, DFGHILM Table 3, Sage
  `self_orthogonal_binary_codes`, and Robert L. Miller's independent
  no-zero-column enumeration.

## What is being enumerated

A **binary linear code** of length `N` is a subspace `C ⊆ F₂^N`; its
codewords are 0/1 strings of length `N` that form a vector space under
bitwise XOR. The code is

- **even** if every codeword has Hamming weight `≡ 0 (mod 2)`,
- **doubly even** if every codeword has Hamming weight `≡ 0 (mod 4)`,
- **self-dual** if `C = C⊥` under the standard inner product
  (forces `N = 2k`),
- **Type II** if it is doubly even *and* self-dual (forces `8 | N`).

Two codes are *equivalent* if one is obtained from the other by
permuting the `N` coordinate positions. This package enumerates one
representative of each equivalence class of doubly even `[N, k]` codes
together with its coordinate-permutation automorphism group
`Aut(C) ≤ S_N`. The mass check at each rank is

    Σ_C N! / |Aut(C)| = σ(N, k),

where `σ(N, k)` is Gaborit's labelled count (a closed form). The
classical examples — the extended Hamming `[8, 4]`, the binary Golay
`[24, 12]`, and the second-order Reed–Muller codes `RM(2, m)` — all
live inside this universe.

The motivation is supersymmetric representation theory: DFGHILM §2 and
§5 prove that

> chromotopologies on `2^{k+1}` vertices, up to color-preserving graph
> isomorphism, are in bijection with doubly even codes `[N, k]`, up to
> column permutation.

So enumerating Adinkra chromotopologies is exactly enumerating doubly
even codes. The relevant target is `N ≤ 32`, with on the order of
`10¹²` equivalence classes at the top.

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
`|Aut(C)|`. The full reproducibility recipe (including the `N = 28`
cloud run) is in [`docs/reproducing.md`](docs/reproducing.md).

## Performance

Single platform for reproducibility: **GCP `c4a-standard-72`** (Axion /
Arm Neoverse V2, aarch64, 72 physical cores, 288 GB), parallel Rust
kernel at `t=72 d=5`. All wall times below are kernel-only:

All cloud rows below predate the **coset-spectrum parent rule** and its
two follow-on spectrum-evaluation levers (2026-06-10, now the default —
see [Algorithmic levers](#algorithmic-levers)), which together cut
canonicalisation calls 15–87× at `N = 22–26` and walls ~9–15× on the
development host; a re-run of the large-`N` rows is pending (the
count-anchored `N = 29` forecast is **1.0–1.5 h** on the same box that
took 12.3 h below).

| `N` | wall                        | classes       |
|----:|----------------------------:|--------------:|
|  20 |                     0.21 s  |         1,211 |
|  22 |                     0.87 s  |         5,118 |
|  24 |                     5.34 s  |        37,496 |
|  26 |                     53.5 s  |       494,272 |
|  27 |             374 s (6.2 min) |     2,673,492 |
|  28 |    **3669 s (61.2 min)**    |  **21,505,546** |
|  29 |    **44 356 s (12.3 hr)**   | **239,465,540** |

`N ≤ 26` uses `CANON_CACHE_CAP=500 000`; the `N = 28` run used
`CAP=300 000`; the `N = 29` run used `CAP=200 000` to stay inside the
288 GB ceiling. The `N = 28` result was measured 2026-05-21 in a
dedicated run; `N ≤ 27` and `N = 29` come from the 2026-05-22/23 sweep.

At `N = 29` the kernel made 87.2 billion canonicalisation calls at a
mean of 29.5 µs/call — about 364 calls per emitted equivalence class.
The scaling bottleneck at this length is the number of calls, not the
per-call cost — which is precisely what the coset-spectrum parent rule
now attacks: most candidates are rejected by an exact weight-spectrum
comparison before any canonicalisation (canon calls drop 87× at
`N = 26`), and two follow-on levers cut the spectrum evaluation itself
~7× per candidate (split-frame sharing with a one-comparison reject,
then a pair-structure chain that decides ~43 % of all candidates in
O(1) past the first stratum). Desktop measurements (`N = 22` in 0.24 s
parallel / 0.80 s sequential, `N = 26` in **11.5 s** parallel / 97.2 s
sequential) and the cross-platform Sage comparison (~1500× at `N = 22`)
live in [`docs/performance.md`](docs/performance.md).

### `N = 29` per-rank class counts

Every row mass-formula certified (`Σ N!/|Aut(C_i)| == σ(29, k)`). Full
integer-precision mass values, σ values, and the audit recipe are in
[`docs/results/n29.json`](docs/results/n29.json).

|     `k` | classes              |
|--------:|---------------------:|
|       0 |                    1 |
|       1 |                    7 |
|       2 |                   39 |
|       3 |                  287 |
|       4 |                2,693 |
|       5 |               34,233 |
|       6 |              555,804 |
|       7 |            8,084,014 |
|       8 |           57,432,707 |
|       9 |          116,908,496 |
|      10 |           51,474,285 |
|      11 |            4,837,471 |
|      12 |              133,563 |
|      13 |                1,940 |
| **total** |    **239,465,540** |

(`σ(29, 14) = 0` — no `[29, 14]` doubly-even codes exist.)

## Validation

Four independent checks back every emitted class:

1. **Gaborit mass formula** — every emitted class, every rank, every `N`.
2. **DFGHILM Table 3** — cell-for-cell through `N = 28`.
3. **Sage `self_orthogonal_binary_codes`** — through `N = 22`.
4. **[rlmiller.org/de_codes](https://rlmiller.org/de_codes/)** —
   Robert L. Miller's no-zero-column enumeration, at `N = 28`.

## Algorithm sketch

The enumerator implements DFGHILM Appendix B end-to-end: Gaborit's mass
formula, a bipartite-graph encoding fed to `sparsenauty`, the
doubly-even linear-algebra optimisations of Corollary B.1, and McKay
1998 canonical augmentation. The production kernel adds eight
engineering changes on top of that recipe. The per-lever multipliers
below are desktop measurements (the development platform used for
ablation); the cumulative effect is roughly 640× over the
pure-Python baseline at `N = 22` and ~1535× faster than Sage
`self_orthogonal_binary_codes`.

- **Quotient-space orbit-min prefilter** — work in the
  `(N − 2k)`-dimensional quotient `C⊥/C` (Gray-code walk over
  `2^(N − 2k)` reps) instead of the full `2^(N − k)` BFS of B.3. The
  dominant Python-side win. Uses precomputed `σ_Q` action tables plus
  a single global `O(2^L)` sweep that decomposes all orbits at once.
- **Low-weight-incidence canonicaliser** — feed nauty a sparse
  bipartite graph on `|C_low| + N` vertices (the lowest-weight
  codewords needed to span `C`, plus the `N` columns) instead of the
  full `2^k + N` graph. `1.91×` at `N = 22`, `≥ 2.5×` at `N = 24`.
- **Native Rust kernel** — the whole DFS in Rust (canonical-parent,
  is-canonical-augmentation, canon-info LRU, mass quota), no
  Python ↔ Rust crossings. `1.32×` at `N = 22`.
- **Outer-DFS worker parallelism with pipelined seeder** — DFGHILM
  B.4's producer-consumer model: the seeder pushes seeds into a
  bounded channel as the DFS discovers them, workers consume in
  parallel, each with its own canon cache, and share an atomic
  mass-tracker for per-rank early termination. `9.6×` at `N = 22`,
  `~12×` at `N = 24`, `~11.2×` at `N = 26`.
- **Mass-formula early stop** — the closed-form `σ(N, k)` plus a
  monotone quota check skips the rest of a rank as soon as
  `Σ N!/|Aut| = σ(N, k+1)`. `4–11%` in isolation.
- **Coset-spectrum parent rule** — the McKay canonical parent of a
  child code is selected by the lex-min complement-coset weight
  spectrum over all `2^(k+1) − 1` hyperplanes (computed for all of
  them at once by per-stratum Walsh–Hadamard transforms, 1–4 µs), so
  ~94–97 % of candidates are rejected with **no canonicalisation call
  at all**; nauty runs only on accepts and exact ties. `7.6×`
  sequential at `N = 22`, `3.1×` at `N = 24` parallel, `6.5×` at
  `N = 26` parallel — growing with `N` (canon-call count −15×/−32×/−87×).
  Kill-switch: `DOUBLY_EVEN_PARENT_RULE=legacy`.
- **Split-frame spectrum sharing + one-comparison reject** — all
  parent-half spectrum work is computed once per parent and shared
  across its sibling candidates (the full-frame Walsh–Hadamard
  transform factors exactly as its last butterfly stage), and a
  per-parent bound decides ~99 % of first strata in a single integer
  compare. Spectrum evaluation `2.8×`; `N = 26` sequential
  207 → 127 s over the plain rule.
- **Pair-structure chain** — candidates surviving a parent-only first
  stratum carry an argmin set with an exploitable pair structure that
  survives every further parent-only stratum; per-parent cached
  E-sets and bounds then decide ~43 % of *all* candidates at `N = 26`
  in O(1) past the first stratum. Spectrum evaluation a further
  `3.8×` at `N = 26`; sequential 126.6 → **97.2 s**.

The cumulative ablation table and per-lever writeup are in
[`docs/algorithm.md`](docs/algorithm.md). Before the coset-spectrum
rule, ~90 % of the `N = 22` parallel wall was inside `sparsenauty`'s C
code; after the three parent-rule levers, the `N = 26` sequential wall
splits canon ~44 % / quotient-space candidate generation ~41 % /
spectrum evaluation ~12 % — and the remaining spectrum time is almost
entirely a popcount stream, i.e. SIMD-shaped.

## Long-running jobs (local or cloud)

For `N ≥ 26` the in-memory `Vec` output approach used by `bench.py`
becomes memory-heavy, and at `N ≥ 29` it is infeasible. The streaming
output path writes per-worker binary files to a local directory and
runs the mass-formula gate in-Rust; peak RSS is dominated by the
per-worker canon caches.

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

The sidecar prints a per-`k` mass-vs-σ table at the configured interval
and exits when the kernel finishes. The same recipe applies to a
30-minute local `N = 24` run and a multi-hour cloud `N = 28` run; only
the output directory changes.

On Apple Silicon (e.g. M5 / M5 Pro, 64 GB unified memory) the same
streaming path should comfortably handle `N = 27` and `N = 28`
overnight at `CANON_CACHE_CAP ≈ 200 000`. This is an extrapolation
from the 13700K and c4a-72 measurements, not a measurement.

For multi-node or `>256`-core deployment hints, see
[`docs/cluster-deployment.md`](docs/cluster-deployment.md). The kernel
has only been tested up to 72 single-NUMA cores, so that doc is a
code-pointer design sketch rather than a working cluster
implementation.

## Status

- `N ≤ 26` is reproducible on a 13700K desktop in seconds to minutes
  (or on `c4a-standard-72` in under a minute).
- `N = 28` is reproducible on GCP `c4a-standard-72` in 61 minutes.
- `N = 29` is complete: 239,465,540 classes in 12.3 hr on the same VM,
  mass-formula certified at every rank
  ([`docs/results/n29.json`](docs/results/n29.json)).
- `N ≥ 30` requires either a much bigger single machine
  (`c4-standard-288-metal` or similar) or a small cluster. The per-node
  streaming-output path is shipped; the cross-node coordinator is not.

At `N ≤ 22` the wall-time frontier is exhausted at the algorithmic
level with this canonicaliser. Roughly 90 % of the parallel wall sits
inside `sparsenauty`'s C code, which is the per-call floor at this
graph shape.

## Project layout

Three Python layers (depend only on the layer above) plus the Rust
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

Dormant and experimental audit-substrate code is quarantined under
`*/experimental/` subpackages — indexed in
[`EXPERIMENTAL.md`](EXPERIMENTAL.md).

## Prior work

This package follows several foundational works:

- **Algorithm** — DFGHILM Appendix B (2011), with Gaborit's mass
  formula (1996) and McKay's canonical-augmentation framework (1998).
- **Canonicaliser** — `sparsenauty` (McKay–Piperno 2014), via the
  `nauty-Traces-sys` Rust binding.
- **Independent enumerations** — Bouyukliev–Bouyuklieva 2019
  ([arXiv:1907.10363](https://arxiv.org/abs/1907.10363)) is an
  independent column-by-column canonical-augmentation engine with
  `[N, k, ≥ d]` validation counts at `N = 31, 32`. Robert L. Miller —
  one of the DFGHILM authors — maintains an independent
  [no-zero-column reference enumeration](https://rlmiller.org/de_codes/)
  that cross-checks our `N = 28` result.

Full credits and the validation hierarchy are in
[`docs/references.md`](docs/references.md). Sage's
`self_orthogonal_binary_codes` is the prior open-source bar
(single-threaded Cython); the parallel Rust kernel here runs ~1535×
faster at `N = 22`.

## Opt-in branches

Two small wins are kept on branches rather than `main` so users can
choose:

- **[`feature/dfs-order-by-aut`](https://github.com/rdeager/doubly-even/tree/feature/dfs-order-by-aut)**
  sorts sibling children by `|Aut(D)|` descending in the DFS. Measured:
  `-12 %` at `N = 24`, `-10 %` at `N = 25`, `-8 %` at `N = 26`;
  regresses at `N ≤ 22`. Toggle: `DOUBLY_EVEN_DFS_ORDER=aut_desc` (or
  `=off`).
- **[`feature/dfs-speedup-bliss-pq`](https://github.com/rdeager/doubly-even/tree/feature/dfs-speedup-bliss-pq)**
  adds a cross-worker priority-queue pull at the depth-5 seeder
  frontier on top of `aut_desc`. Additional `-2 %` at `N = 24, 26`.
  Toggle: `DOUBLY_EVEN_SEED_ORDER=fifo` reverts to the default.

Each branch can be checked out and measured directly:
`git checkout feature/dfs-order-by-aut && maturin build --release …`.
Both branches were measured against the legacy parent rule; their
deltas have not been re-validated on top of the coset-spectrum rule
(which reshuffles where the wall goes), so re-bench before stacking.

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
