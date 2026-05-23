# Performance reference

Single home for the measured wall-time numbers and tuning knobs. The
algorithmic *why* lives in [`algorithm.md`](algorithm.md); this doc is
tables.

## Headline (13700K, parallel kernel, mean of 3 runs)

13700K = Intel Core i7-13700K (8 P-cores × SMT2 + 8 E-cores =
24 logical / 16 physical). Each run starts from a cold canon cache.

| N  | classes  | sequential | parallel best          | speedup |
|----|---------:|-----------:|------------------------|--------:|
| 18 |      341 |     0.15 s | —                      | —       |
| 20 |    1,211 |     0.97 s | 0.22 s (t=16, d=4)     |  4.4×   |
| 22 |    5,118 |     6.64 s | **0.691 s** (t=20, d=4)| **9.6×**|
| 24 |   37,496 |    ~107 s  | **8.90 s** (t=24, d=5) | ~12.0×  |
| 26 |  494,272 |     —      | **170 s** (t=24, d=5)  | ~11.2×  |

Sequential at `N = 26` was not measured directly; the extrapolation
from `N = 24` is ~30 min.

## Cloud runs

| platform                  | cores               | RAM    | N      | wall      | notes |
|---------------------------|---------------------|--------|--------|-----------|-------|
| GCP `c4-standard-24` (Emerald Rapids 8581C, x86_64) | 12 phys + SMT | 90 GB  | 26 | 285 s     | per-thread 1.65× slower than 13700K = pure clock ratio |
| GCP `c4a-standard-72` (Axion / Neoverse V2, aarch64) | 72 phys (no SMT) | 288 GB | 28 | **3669 s (61.2 min)** | first reproducible N=28 enumeration; ~$3 of compute |
| GCP `c4a-standard-72` (Axion, aarch64) | 72 phys | 288 GB | 29 | *in flight at publication time* | placeholder — to be updated |

The Emerald-Rapids cross-port has zero per-IPC penalty; the Axion
port needs a one-line `Cargo.toml` patch to disable the x86-only
`popcnt` feature of `nauty-Traces-sys` (see the
[`reproducing.md`](reproducing.md) ARM section), after which builds
work unchanged.

## The `N = 28` cloud run (first reproducible result)

DFGHILM Appendix B Table 3 publishes class counts for `[N, k]`
doubly-even codes up to `N = 28`. Until 2026-05-21, the `N = 28` row
had not been independently reproduced from a published implementation;
DFGHILM's enumeration ran on the OSU Glenn supercomputer in 2011 and
no enumerator was released.

We reproduced the `N = 28` row on a single GCP `c4a-standard-72` VM
in 61.2 min of wall time, ~$3 of on-demand compute:

- **classes:** 21,505,546
- **canon calls:** 5,358,750,799
- **per-call cost:** 37.7 µs (the surprise — *lower* than at N=22; see below)
- **DFGHILM Table 3:** all 14 (N=28, k) cells agree exactly
- **rlmiller cross-check:** predicted no-zero-col count
  `total(28) - total(27) = 18,832,025` agrees with
  [rlmiller.org/de_codes](https://rlmiller.org/de_codes/)'s
  ~18,832,054 to within OCR-parse noise.

Per-rank breakdown at `N = 28`:

```
k= 0:           1     k= 1:           7     k= 2:          39
k= 3:         263     k= 4:       2,136     k= 5:      20,812
k= 6:     224,825     k= 7:   1,917,212     k= 8:   7,631,323
k= 9:   8,948,070     k=10:   2,550,127     k=11:     203,178
k=12:       7,402     k=13:         151
```

(Heaviest at `k = 9`; no `[28, 14]` half-rank doubly-even codes exist.)

## `N = 29` status (placeholder)

The `N = 29` enumeration started 2026-05-22 on `c4a-standard-72` with
the streaming output path (per-worker binary files, no in-memory `Vec`).
At session time it was ~53 % through by mass with an ETA of ~3.8 h
remaining. **This section will be updated with the final wall, class
count, and per-k cells once the run lands.**

The reproducible recipe is in [`reproducing.md`](reproducing.md);
the streaming output path is described under
[long-running jobs](../README.md#long-running-jobs-local-or-cloud)
in the README.

## The surprise: per-call cost drops with N

A reasonable a-priori prediction is that nauty's per-call cost grows
with `N` (the bipartite graph has more vertices). Empirically the
opposite holds:

| N  | µs/call | nauty `nodes/call` | `maxlevel` | gen/call |
|----|--------:|-------------------:|-----------:|---------:|
| 22 |   80.43 |               67.5 |       9.25 |    10.62 |
| 24 |   77.96 |               49.4 |       7.64 |     8.67 |
| 26 |   58.84 |               28.6 |       5.60 |     5.89 |
| 28 |   37.68 |               13.9 |       3.80 |     3.47 |

Every metric of nauty's internal search tree shrinks monotonically. The
larger codes have more exploitable internal structure — the low-weight-
incidence refinement discretises faster, so per-call cost drops by ~2×
across `N = 22 → 28`.

The implication for forecasting: the dominant wall-time multiplier per
2-step is `(call-count ratio) × (per-call cost ratio)`. Across
`N = 26 → 28` this was `118.9× × 0.64× = 76×` (expected) and `66×`
(measured); the ~10 % gap is end-of-run tail imbalance.

## Sage comparison

`sage.coding.databases.self_orthogonal_binary_codes(N, N//2, 4)` is
the comparable Sage entry point. Measured on the same 13700K host,
single-threaded:

| N  | Sage         | `doubly-even` seq | `doubly-even` parallel | ratio (par)   |
|----|-------------:|------------------:|-----------------------:|--------------:|
| 22 |    363.85 s  |            6.64 s |             **0.691 s** | **~525×**    |

Sage is inherently single-threaded as shipped:
`sage.coding.binary_code.BinaryCodeClassifier.generate_children` is
Cython, holds the GIL, uses Python-object refcounting for partition
stacks and orbit machinery, and has no `nogil` blocks. Parallelising
it would require non-trivial Cython surgery; nobody has done it.

At `N ≥ 24` Sage is impractical in our experience; the ratio comparison
extrapolates rather than measures.

## Tuning knobs reference

All knobs are environment variables, read by the Rust kernel. They have
no effect when unset; defaults are the recommended values for `N ≤ 22`.

| env var                              | default       | description |
|--------------------------------------|---------------|-------------|
| `DOUBLY_EVEN_THREADS`                | unset (seq)   | `≥ 2` enables the parallel path; recommended `logical_cores − 2` at `N ≤ 22`, `logical_cores` at `N ≥ 24` |
| `DOUBLY_EVEN_FRONTIER_DEPTH`         | 4             | DFS depth at which the seeder yields seeds to the worker pool; raise to 5 for `N ≥ 24` |
| `DOUBLY_EVEN_CANON_CACHE_CAP`        | 1,000,000     | per-worker LRU size (entries); load-bearing at `N ≥ 26` to keep the per-worker × N-workers footprint under host memory |
| `DOUBLY_EVEN_NO_MASS_STOP`           | unset         | set to `1` to disable mass-stop pruning (ablation only) |

Recommended config table (after the 2026-05-23 thread-count recalibration):

| host                            | N    | THREADS | FRONTIER_DEPTH | CANON_CACHE_CAP |
|---------------------------------|------|--------:|---------------:|----------------:|
| 13700K (24 logical / 16 phys)   | ≤22  |      20 |              4 |       1,000,000 |
| 13700K                          |   24 |      24 |              5 |       1,000,000 |
| 13700K                          |   26 |      24 |              5 |         500,000 |
| 13700K (N=28 — cgroup-tight)    |   28 |      20 |              5 |         200,000 |
| c4-standard-24 (Intel, 12 phys) |   26 |      24 |              5 |         500,000 |
| c4a-standard-72 (Axion, 72 phys)|   28 |      72 |              5 |         300,000 |

At `N = 28` on the 13700K, the per-worker canon caches `× 22 threads`
exceeded the 52 GiB cgroup limit on the dev host; the local run OOM'd
under the default `CANON_CACHE_CAP = 500,000`. Drop to `200,000` to
fit, or run on a memory-larger machine. The c4a-72 run used
`CAP = 300,000` against 288 GB RAM and peaked at 71 GB.

## How to reproduce these numbers

See [`reproducing.md`](reproducing.md). The short version:

```sh
uv sync --all-extras --dev
maturin build --release --features parallel -m rust/Cargo.toml
uv pip install --force-reinstall rust/target/wheels/doubly_even_kernel-*.whl
DOUBLY_EVEN_THREADS=20 uv run python scripts/bench.py --label local-22 --N 22
```

This should produce `N=22 wall ≈ 0.69 s` and 5,118 classes (mass-formula
verified in-Rust) on a 16-physical-core machine.
