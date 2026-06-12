# Performance reference

Single home for the measured wall-time numbers and tuning knobs. The
algorithmic *why* lives in [`algorithm.md`](algorithm.md); this doc is
tables.

## Headline (automorphism-only canonicalisation epoch, parallel kernel, median of 3)

Measured 2026-06-12 on a 13700K-equivalent dev container — a
single-wheel knob A/B whose `DOUBLY_EVEN_CANON_LABELLING=full` control
reproduced the 2026-06-11 walls (0.683 / 6.27 / 81.5 s sequential), so
the "vs full labelling" column is exactly
[`algorithm.md`](algorithm.md) lever 10's same-session delta. Each run
starts from a cold canon cache; all DFGHILM Table 3 cells verified per
run. `d = 4` is the best frontier depth at every benched `N`.

| N  | classes   | sequential | parallel best           | vs full labelling |
|----|----------:|-----------:|-------------------------|------------------------:|
| 18 |       341 |    0.029 s | —                       | ~1.0× (small N) |
| 20 |     1,211 |    0.147 s | 0.051 s (t=16, d=4)     | ~1.0× (small N) |
| 22 |     5,118 |    0.623 s | **0.24 s** (t=24, d=4)  | 1.097× (seq) |
| 24 |    37,496 |     5.72 s | **~1.7 s** (t=24, d=4)  | 1.096× (seq) |
| 26 |   494,272 | **74.6 s** | **9.21 s** (t=24, d=4)  | 1.092× (seq), 1.035× (par) |
| 27 | 2,673,492 |          — | **~63–66 s** (t=24, d=4, cap=500K) | ~1.0× (par; worker-bound, session noise ±10 %) |

`N = 26` sequential needs `DOUBLY_EVEN_CANON_CACHE_CAP=500000` (the
uncapped run OOMs). The four-Russians + codegen epoch (2026-06-11):
`N = 22` 0.685 s seq / 0.24 s par; `N = 24` 6.26 s seq; `N = 26`
81.4 s seq / 9.70 s par; `N = 27` 63.0 s par.
The pair-structure-chain numbers (2026-06-10, same
container, kept for cross-reference): `N = 22` 0.796 s seq / 0.24 s
par; `N = 24` 7.52 s seq / 1.77 s par; `N = 26` 97.2 s seq / 11.5 s
par; `N = 27` 66.1 s par. The split-frame numbers (2026-06-10):
`N = 24` 7.75 s seq / 1.77 s par (same-hour control); `N = 26`
126.6 s seq / 11.8 s par (same-hour control; 13.2 s in its own cooler
session). Plain coset-spectrum: `N = 22` 0.848 s seq / 0.237 s par;
`N = 24` 9.40 s seq / 2.60 s par; `N = 26` 207.1 s seq / 21.4 s par
(t=24 d=4). The legacy σ-based parent rule remains available as
`DOUBLY_EVEN_PARENT_RULE=legacy`; its 13700K record numbers: `N = 22`
6.64 s seq / 0.691 s t=20 d=4; `N = 24` 8.90 s t=24 d=5; `N = 26`
169.8 s t=24 d=5.

Parallel `N ≤ 24` is bounded by the serial seeder span (worker
active/wall is 44 % at `N = 26`, 19 % at `N = 24`, t=24 d=4), which is
why the chain's 1.30× sequential win showed up as only ~1.03× parallel
on the desktop at those sizes — and why the four-Russians orbit BFS,
which shortens the seeder span itself, *does* land 1.18× parallel at
`N = 26`. One step up the chain's win emerged in full: at **`N = 27`**
(same-hour A/B, median of 3, t=24 d=4 cap=500K) it took the parallel
wall from 94.1 s to **66.1 s (1.42×)** — total work grows ~6× per step
while the seeder span only roughly doubles, so the worker share
recovers and the per-candidate saving (4.6× on spectrum evaluation at
`N = 27`) lands on the wall. The orbit BFS left `N = 27` flat (that
wall is worker-bound, not seeder-bound); the codegen flag brings the
current record to **63.0 s**. The 2,673,492 classes reproduce the
`c4a-standard-72` record exactly — a 24-thread desktop now beats that
pre-parent-rule 72-core cloud row (374 s) ~5.9×. On a many-core cloud
run the sequential savings carry into the core-hours directly.

## Cloud runs

| platform                  | cores               | RAM    | N      | wall      | notes |
|---------------------------|---------------------|--------|--------|-----------|-------|
| GCP `c4-standard-24` (Emerald Rapids 8581C, x86_64) | 12 phys + SMT | 90 GB  | 26 | 285 s     | per-thread 1.65× slower than 13700K = pure clock ratio |
| GCP `c4a-standard-72` (Axion / Neoverse V2, aarch64) | 72 phys (no SMT) | 288 GB | 28 | **3669 s (61.2 min)** | first reproducible N=28 enumeration; ~$3 of compute; `CAP=300K` |
| GCP `c4a-standard-72` (Axion, aarch64) | 72 phys | 288 GB | 29 | **44 356 s (12.3 hr)** | **first publicly reproducible N=29 enumeration**; mass-formula certified; ~$35 of compute; `CAP=200K` (cgroup-tight) |

The Emerald-Rapids cross-port has zero per-IPC penalty; the Axion
port builds unchanged — the x86-only `popcnt` feature of
`nauty-Traces-sys` is target-conditional in the manifest
(`rust/core/Cargo.toml`; see the [`reproducing.md`](reproducing.md)
ARM section). At `N = 29` on c4a-72 the per-worker LRU times 72
workers grew tighter against the 288 GB headroom, so we ran with
`DOUBLY_EVEN_CANON_CACHE_CAP = 200000` rather than the 300 K used at
`N = 28` — this likely costs ~5–10 % wall to canon-cache thrash but
fits comfortably.

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

## The `N = 29` result (first publicly reproducible)

DFGHILM Table 3 stops at `N = 28`. Our `N = 29` enumeration is new
ground — verified at every emitted class by the Gaborit mass-formula
oracle (`Σ N!/|Aut(C_i)| == σ(N, k)`), but with no published
table to cross-check class counts cell-by-cell.

Run on a single GCP `c4a-standard-72` VM, 2026-05-23, in 12.32 hr of
wall (~$35 of on-demand compute), git SHA
[`5a13414`](https://github.com/rdeager/doubly-even/commit/5a1341468c98d2e58e39a37c9dc3ac62b7d25f46):

- **total equivalence classes:** 239,465,540
- **canon calls:** 87,205,911,575
- **per-call cost:** 29.5 µs (continuing the dropping trend)
- **nauty utilisation:** 80.5 % of wall × threads (in nauty C code)
- **mass-formula certificate:** `mass == σ(29, k)` exactly at every
  `k = 0..13`

Per-rank breakdown:

| `k`  | classes      | `mass` = `σ(29, k)`                                             |
|----:|-------------:|----------------------------------------------------------------:|
|   0 |            1 | 1                                                               |
|   1 |            7 | 134,209,535                                                     |
|   2 |           39 | 1,500,924,953,148,075                                           |
|   3 |          287 | 1,798,227,953,449,795,246,275                                   |
|   4 |        2,693 | 251,287,611,025,390,597,345,911,795                             |
|   5 |       34,233 | 4,245,747,369,833,030,971,769,514,529,875                       |
|   6 |      555,804 | 8,815,991,145,789,015,024,952,841,955,961,875                   |
|   7 |    8,084,014 | 2,265,709,724,467,776,861,412,880,382,682,201,875               |
|   8 |   57,432,707 | 72,209,501,689,214,206,089,029,328,902,189,233,875              |
|   9 |**116,908,496** | 284,740,011,553,359,344,949,890,602,226,832,301,875           |
|  10 |   51,474,285 | 137,777,424,945,173,876,588,656,743,012,983,371,875             |
|  11 |    4,837,471 | 8,009,532,764,277,328,438,715,267,424,789,946,875               |
|  12 |      133,563 | 52,810,106,138,092,275,420,100,664,339,274,375                  |
|  13 |        1,940 | 32,236,665,937,060,356,134,843,526,028,125                      |

Heaviest at `k = 9` (~117 M classes); no `[29, 14]` doubly-even codes
exist (`σ(29, 14) = 0`).

The full machine-readable certificate, with integer `mass` and
`gaborit_sigma` values per `k` and a one-second offline audit recipe,
is at [`docs/results/n29.json`](results/n29.json). The complete
per-class binary stream (5.2 GB compressed, ~21 GB uncompressed) and
the JSONL extracts for the heaviest ranks are
[available on request](results/README.md#on-request).

The reproducible recipe is in [`reproducing.md`](reproducing.md);
the streaming output path is described under
[long-running jobs](../README.md#long-running-jobs-local-or-cloud)
in the README.

## The scaling story: per-call cost drops, calls-per-class explodes

A reasonable a-priori prediction is that nauty's per-call cost grows
with `N` (the bipartite graph has more vertices). Empirically the
opposite holds — *but* the total number of canon calls grows much
faster than the class count, and that growth is the dominant scaling
factor:

| N  | classes      | canon calls         | calls/class | µs/call | nauty `nodes/call` | `maxlevel` | gen/call |
|---:|-------------:|--------------------:|------------:|--------:|-------------------:|-----------:|---------:|
| 22 |        5,118 |              80,011 |      **15.6** |   80.43 |               67.5 |       9.25 |    10.62 |
| 24 |       37,496 |           1,264,155 |      **33.7** |   77.96 |               49.4 |       7.64 |     8.67 |
| 26 |      494,272 |          45,085,645 |      **91.2** |   58.84 |               28.6 |       5.60 |     5.89 |
| 28 |   21,505,546 |       5,358,750,799 |      **249**  |   37.68 |               13.9 |       3.80 |     3.47 |
| 29 |  239,465,540 |      87,205,911,575 |      **364**  |   29.47 |                  — |          — |        — |

Two trends pulling in opposite directions:

- **Per-call cost is dropping** ~2.7× across `N = 22 → 29` (80 → 29 µs).
  Every metric of nauty's internal search tree (`nodes/call`,
  `maxlevel`, generators emitted) shrinks monotonically — larger codes
  have more exploitable internal structure, so the low-weight-incidence
  refinement discretises faster.
- **Calls per emitted class is exploding**: 15.6 at `N = 22`,
  growing by ~2.6× per 2-step to 249 at `N = 28` and 364 at `N = 29`.
  This is the canonical-augmentation tax — most candidates fail the
  is-canonical-augmentation test and the candidate set per
  augmentation step grows with `N`.

Net for wall-time forecasting: the dominant multiplier per 2-step is
`(class-count ratio) × (calls/class ratio) × (per-call cost ratio)`.
Across `N = 26 → 28` this was `43.5× × 2.73× × 0.64× = 76×` (expected)
and `66×` (measured); the ~10 % gap is end-of-run tail imbalance.

**Resolution (2026-06-10).** The conclusion this table forced — the
leverage point is calls per class, and a successful lever must reject
candidates *before* canonicalising them — is what the
**coset-spectrum parent rule** delivered (lever 6 in
[`algorithm.md`](algorithm.md#6-coset-spectrum-parent-rule)). Unlike
the cheap *rejectors* measured earlier (paired-iso, cubic tensor, T11
— all costlier per probe than the canon call), it changes the parent
*definition* so that an exact 1–4 µs weight-spectrum computation
decides ~94–97 % of candidates with no canon call at all. Measured
canon-call reduction: 15× at `N = 22`, 32× at `N = 24`, **87× at
`N = 26`** — growing with `N` because it cancels precisely the
calls-per-class explosion in the table above. The table's per-call
column still describes the calls that remain (accepts + rare φ-ties).
The two spectrum-evaluation passes that followed (levers 7–8) then cut
the per-candidate cost of the new parent test itself ~7× end-to-end
(1.82 µs → 257 ns per candidate at `N = 24–26`); the count-anchored
`N = 29` forecast on the same 72-core box that took 12.32 h pre-rule
is now **1.0–1.5 h** (~$3–5) — the cheapest high-value validation run
on the books.

## Full `c4a-standard-72` sweep — DFGHILM Table 3 reproduction

Single machine, single configuration, ten enumerations on
`c4a-standard-72` over 2026-05-22 → 2026-05-23. Each row's total
class count was checked cell-by-cell against the corresponding row
of DFGHILM Table 3 (where it exists) and against the
Gaborit mass formula at every rank. All checks passed:

| `N`  | wall      | total classes      | check                    |
|----:|----------:|-------------------:|--------------------------|
|  16 |   0.02 s  |                146 | DFGHILM Table 3 ✓        |
|  20 |   0.21 s  |              1,211 | DFGHILM Table 3 ✓        |
|  21 |   0.36 s  |              2,078 | DFGHILM Table 3 ✓        |
|  22 |   0.87 s  |              5,118 | DFGHILM Table 3 ✓        |
|  23 |   1.81 s  |             11,783 | DFGHILM Table 3 ✓        |
|  24 |   5.34 s  |             37,496 | DFGHILM Table 3 ✓        |
|  25 |  11.74 s  |            113,223 | DFGHILM Table 3 ✓        |
|  26 |  53.51 s  |            494,272 | DFGHILM Table 3 ✓        |
|  27 |   374.0 s |          2,673,492 | DFGHILM Table 3 ✓        |
|  29 |  12.32 hr |        239,465,540 | mass-formula certificate (DFGHILM has no `N = 29` cells) |

Configuration: `DOUBLY_EVEN_THREADS=72 DOUBLY_EVEN_FRONTIER_DEPTH=5
DOUBLY_EVEN_CANON_CACHE_CAP=500000` for `N ≤ 27`, `CAP=200000` for
`N = 29` (cgroup-tight RAM headroom). `N = 28` is documented
separately above — the c4a-72 N=28 run on 2026-05-21 was used for
the original DFGHILM Table 3 reproduction milestone and predates
this sweep; its `CAP=300000` and class counts match.

The corresponding per-`k` cells for `N ≤ 28` are in DFGHILM Table 3;
the `N = 29` per-`k` breakdown is in
[the `N = 29` section above](#the-n--29-result-first-publicly-reproducible)
and in [`docs/results/n29.json`](results/n29.json). Cross-checks
against [Robert L. Miller's `de_codes` site](https://rlmiller.org/de_codes/)
(no-zero-column convention) are documented at `N = 28`
in [`references.md`](references.md#robert-l-millers-de_codes-site).

## Sage comparison

`sage.coding.databases.self_orthogonal_binary_codes(N, N//2, 4)` is
the comparable Sage entry point. Measured on the same 13700K host,
single-threaded:

| N  | Sage         | `doubly-even` seq | `doubly-even` parallel | ratio (par)   |
|----|-------------:|------------------:|-----------------------:|--------------:|
| 22 |    363.85 s  |            6.64 s |             **0.691 s** | **~525×**    |

The 525× is not "Sage runs the naive canon-everything baseline" — a
2026-05-23 prior-art audit (see
[`algorithm.md` §1](algorithm.md#1-quotient-space-orbit-min-prefilter))
confirmed that Sage's `binary_code.pyx` already has the quotient-space
prefilter, the Gray-walked lift, the weight-mod-4 filter, and a
visited-set bitmap to skip processed cosets. The honest decomposition
of the wall-time gap is a stack of compounding wins, approximately:
~6× from the precomputed `σ_Q` action tables + global single-sweep
orbit decomposition (O(1) table lookup vs Sage's O(N) per-candidate
word-permutation hop, `binary_code.pyx:4109–4121`); ~10–20× from the
Rust kernel + native sparsenauty replacing per-candidate Python
canonicalisation under the GIL; ~5–9× from the outer-DFS worker
parallelism with pipelined seeder (`DOUBLY_EVEN_THREADS=20`); ~2×
from the Q_D low-weight-incidence canonicaliser handing nauty
smaller graphs at high `k`. No single mythical lever.

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
| `DOUBLY_EVEN_FRONTIER_DEPTH`         | 4             | DFS depth at which the seeder yields seeds to the worker pool; `4` is the measured best at every benched `N` since the split-frame lever (the older "raise to 5 for `N ≥ 24`" guidance is obsolete) |
| `DOUBLY_EVEN_CANON_CACHE_CAP`        | 1,000,000     | per-worker LRU size (entries); load-bearing at `N ≥ 26` to keep the per-worker × N-workers footprint under host memory |
| `DOUBLY_EVEN_NO_MASS_STOP`           | unset         | set to `1` to disable mass-stop pruning (ablation only) |
| `DOUBLY_EVEN_PARENT_RULE`            | `coset-spectrum` | parent-selection rule: `coset-spectrum` (default), `legacy` (σ-based rule, kill-switch / A-B control), `audit` (legacy behaviour + φ instrumentation) |
| `DOUBLY_EVEN_PHI_MAX_RANK`           | 13            | child-rank cap for the coset-spectrum cascade; above it the kernel uses the legacy rule per rank (only relevant at `N ≥ 30`) |
| `DOUBLY_EVEN_SEEDER_THREADS`         | = THREADS     | seeder helper-pool size for pooled σ_Q candidate generation; `0`/`1` disables the pool entirely |
| `DOUBLY_EVEN_SEEDER_PAR_MIN_L`       | 22            | minimum quotient dimension for pooled σ_Q stages; the default pools only the large early calls where the worker pool is still idle — lowering it measurably loses to helper-vs-worker contention |

Recommended config table (after the 2026-06-10 split-frame re-tune —
`d = 4` everywhere; the seeder-pool knobs are best left at defaults):

| host                            | N    | THREADS | FRONTIER_DEPTH | CANON_CACHE_CAP |
|---------------------------------|------|--------:|---------------:|----------------:|
| 13700K (24 logical / 16 phys)   | ≤22  |      20 |              4 |       1,000,000 |
| 13700K                          |   24 |      24 |              4 |       1,000,000 |
| 13700K                          |   26 |      24 |              4 |         500,000 |
| 13700K (N=28 — cgroup-tight)    |   28 |      20 |              4 |         200,000 |
| c4-standard-24 (Intel, 12 phys) |   26 |      24 |              4 |         500,000 |
| c4a-standard-72 (Axion, 72 phys)|   28 |      72 |              4 |         300,000 |

(The cloud rows above were *measured* at `d = 5` pre-split-frame; the
`d = 4` recommendation extrapolates the 13700K re-tune and should be
spot-checked in the first post-split-frame cloud shakedown.)

At `N = 28` on the 13700K, the per-worker canon caches `× 22 threads`
exceeded the 52 GiB cgroup limit on the dev host; the local run OOM'd
under the default `CANON_CACHE_CAP = 500,000`. Drop to `200,000` to
fit, or run on a memory-larger machine. The c4a-72 run used
`CAP = 300,000` against 288 GB RAM and peaked at 71 GB.

**Build note (x86 codegen).** x86 wheels are built with
`-C target-cpu=x86-64-v3` via `rust/.cargo/config.toml` — cargo's
config discovery is working-directory-based, so the flag only applies
to builds run from inside `rust/`; `scripts/install-kernel.sh` does
this for you. Verify a wheel with
`doubly_even_kernel.kernel_target_features()` — `avx2` must be `True`
on x86. The resulting wheel **requires AVX2** (any x86 CPU since
~2013; both GCP x86 families qualify). aarch64 builds are unaffected.

## How to reproduce these numbers

See [`reproducing.md`](reproducing.md). The short version:

```sh
uv sync --all-extras --dev
scripts/install-kernel.sh parallel   # builds from inside rust/ so the x86-64-v3 flag applies
DOUBLY_EVEN_THREADS=20 uv run python scripts/bench.py --label local-22 --N 22
```

This should produce `N=22 wall ≈ 0.25 s` and 5,118 classes (mass-formula
verified in-Rust) on a 16-physical-core machine.
