# Current bottleneck profile (living document)

> **As of:** 2026-06-11 (consolidation session: workspace restructure +
> x86-64-v3 codegen flag, both shipped and re-benched) · **wheel:**
> `--features parallel`, AVX2 build · **box:** 13700K-equivalent dev
> container (24 logical, AVX2, no AVX-512), 62 GB.
>
> **Maintenance checklist — when a lever ships, touch these in order:**
> 1. The internal maintainers' writeup (see the note at the bottom of this
>    file) — that tree holds the only mapping from internal change labels
>    to the descriptive names used here.
> 2. `algorithm.md` — new lever section under its descriptive name, plus a
>    cumulative-ablation row.
> 3. `performance.md` — headline table re-measured (same-session A/B,
>    median of 3, Table 3 + mass green).
> 4. **This file** — header (date + kernel commit), phase shares,
>    sub-phase splits if re-captured, ranked levers re-ranked, dead list
>    extended if anything was closed en route.
> 5. `../README.md` — only if the headline desktop/cloud numbers moved.
> 6. `theory.md` — only if decision semantics changed (new theorem or
>    lemma) or functions in the code index moved.
> 7. `../CLAUDE.md` — only if a knob, convention, or trap changed
>    (performance *state* lives here, not there).
> 8. `../AGENT_TOUR.md` — only if an orientation-level claim changed
>    (bottleneck model, frontier, file map).
> 9. Stats schema changed? The kernel is the single source
>    (`kernel_stats_layout()`); re-check the `scripts/bench.py` mirror.
>
> Never copy internal session labels into items 2–8.

This is the one document that describes the *current* performance state
of the enumerator. Everything quantitative below was measured on the dev
box above unless a cloud platform is named. How these numbers are
produced: [`benchmarking.md`](benchmarking.md). Why the levers work:
[`algorithm.md`](algorithm.md) (narrative) and [`theory.md`](theory.md)
(proofs).

## 1. Headline walls

| N  | sequential | parallel (t=24, d=4) | classes |
|----|-----------:|---------------------:|--------:|
| 22 | 0.685 s    | 0.24 s               | 5,118 |
| 24 | 6.26 s     | ~1.7 s               | 37,496 |
| 26 | 81.4 s     | 9.70 s               | 494,272 |
| 27 | —          | 63.0 s               | 2,673,492 |

Medians of 3 (N=27 par: single rep) on the v3 wheel, 2026-06-11. Two
structural changes shipped the same session, both decision-bit-identical:
the **Cargo workspace restructure** (perf-neutral, 0.98–0.99× control
A/B) and the **x86-64-v3 codegen flag** (same-session A/B: N=26 seq
1.044×, N=24 seq 1.014×, N≤22 and N=26 par flat; an earlier warmer-box
session recorded up to 1.09–1.11× — treat 1.01–1.05× as the cool-box
value, with the N=27 row suggesting more at larger N). N=26 sequential
needs `DOUBLY_EVEN_CANON_CACHE_CAP=500000` (uncapped OOMs); even capped
it is OOM-adjacent on a 62 GB box (silent mid-run SIGKILLs observed —
use an RSS poller if a rep goes missing).

Cloud records (both predate every lever shipped since 2026-06-10, so a
re-run is cheap): N=28 in 61 min on GCP c4a-standard-72 (~$3); N=29 in
12.32 h on the same platform (~$35), 239,465,540 classes, certificate at
[`results/n29.json`](results/n29.json).

## 2. Phase shares (sequential N=26; shares from the pre-v3 85.6 s profile, shape unchanged at 81.4 s)

| phase | seconds | share | trend |
|---|---:|---:|---|
| canonicalisation (nauty) | 43.0 | 50.3 % | largest again; κ (kept-canon fraction) ≈ 0.039 and falling — a *better canonicaliser* is not the lever, fewer calls already won |
| σ_Q candidate generation | 28.0 | 32.8 % | was ~41 % before the four-Russians BFS; residual is close to its scalar floor |
| φ cascade (parent rule)  | 11.6 | 13.5 % | beaten down from 59.9 % by the sharing/amax/chain levers |
| other (recursion, RREF, cache) | ~3 | ~3.4 % | |

Sub-phase splits:

| φ residual (sampled) | share | | σ_Q | share |
|---|---:|---|---|---:|
| v-half XOR+popcount+histogram | 67.3 % | | orbit-min BFS | ~87–89 % |
| first-stratum decision | 16.7 % | | singular-reps Gray walk | ~6 % |
| direct parity counting | 10.8 % | | basis/lift/sort | rest |
| per-stratum WHT | 5.1 % | | | |

Character of each: the φ v-half sweep is **compute-bound** (no cache
cliff anywhere — streams at 12 GB/s scalar, 21–22 GB/s under AVX2; its
histogram half is scalar-bound at ~0.19 ns/elem and is the floor). The
σ_Q orbit-min BFS is **compute-bound on image generation** (the
seen-bitset is L2/L3-resident at L=20–23; probes are 2–5× cheaper than
the old chained walk); the byte-table method left it ~L1-load-bound.
Canon is sparsenauty at its measured algorithmic floor for this graph
shape (~78 µs/call shape; per-call cost *falls* with N).

## 3. Parallel state (the seeder Amdahl wall)

Workers are healthy; the **seeder is the binding constraint at
N ≤ 27**:

| N | worker busy / threads | seeder facts |
|---|---|---|
| 24 | 19 % avg (early-run idle) | seeder span small; pool a no-op by construction (L < 22) |
| 26 | 13.9/24 during k=3 σ_Q spans | unpooled k=3 σ_Q ≈ 27–49 % of par wall (with N=27) |
| 27 | 19.1/24 | seeder itself blocked 30.3 s of its 51 s span on a full channel — wall is worker-throughput-bound, so seeder-side σ_Q speedups don't move it |

Standing results: frontier depth **d=4 beats d=5 at every benched N**
(the deeper cut resurrects the serial-seeder ceiling); the pooled-seeder
gate `DOUBLY_EVEN_SEEDER_PAR_MIN_L=22` is load-bearing (lowering to 20
loses 6–7 % at N=26 — pooling past the workers-idle window contends with
saturated workers); an adaptive workers-starved gate was measured dead
(workers are *not* starved during the k=3 spans). The measured next
parallel lever is **fanning out across the ~200 independent per-parent
k=3 σ_Q calls** (~20–30 ms each), not parallelising inside one BFS.

## 4. Ranked next levers

1. ~~x86-64-v3 codegen flag~~ — **SHIPPED 2026-06-11**
   (`rust/.cargo/config.toml`; 1.01–1.05× sequential on a cool box,
   decisions bit-identical; aarch64 unaffected, NEON is baseline).
2. **N ≥ 28 cloud re-run** — the highest-value datapoint. N=29
   count-anchored forecast: **1.0–1.5 h on c4a-72** (~$3–5) vs the
   12.32 h actually paid pre-levers; the forecast predates the codegen
   flag and only gets safer. Re-verify d on the first N ≥ 28 run.
3. **Parallel k=3 σ_Q fan-out** (across calls, not within a BFS) — the
   measured seeder lever at N=26/27; see §3.
4. **Dense-level direction-switch BFS** (sequential σ_Q candidate):
   blind-mark + extract on dense BFS levels (Beamer-style bottom-up).
   Portable, exact, ~1.15–1.25× BFS-local ≈ 1.04–1.06× end-to-end,
   ~a day in `orbit.rs`. The only surviving σ_Q idea from the SIMD
   investigation; stacks with the flag.

## 5. Scaling forecast N=28 → 32

Method: count-anchored extrapolation
(`scripts/experimental/post_d15_scaling_fit.py --glob <clean-label> --kappa <κ>`)
— phases priced per *event count* (candidates, kept-canon calls) rather
than geometric wall fits; use only same-wheel same-session label globs
(a polluted glob quietly shifts the fit).

| target | forecast | confidence / blocker |
|---|---|---|
| N=28 | well under the 61-min record; single c4a-72 | re-run is routine |
| N=29 | 1.0–1.5 h on c4a-72 | count-anchored against the true 87.2 B pre-lever candidate count |
| N=30 | streaming + ≥256 GB RAM or a small cluster; ~5–20 h on ~10× c4a-72 | cluster coordinator is the unshipped piece ([`cluster-deployment.md`](cluster-deployment.md)) |
| N=31 | between N=30 and N=32; no fit run yet | candidate-count growth is the unknown |
| N=32 | **floor ~63 core-h** single-thread by geometric carry — but the same method gave 475 and "174" on earlier data vintages; treat every figure as a floor, not an estimate | cluster territory; within reach post-levers. The k ≤ 4 slice alone has a separate published path (column-multiset engine, `algorithm.md`) |

A >10× algorithmic speedup is banked since the N=29 run; machines with
~4× the cores of c4a-72 exist (c4-288-metal, r8g.metal-48xl) — that
product is the credible path to N=30/31 on a single machine. N=32
planning should wait for a post-lever cloud datapoint to re-anchor the
fit.

## 6. Measured dead — do not revisit without new evidence

| idea | why it's dead | vintage |
|---|---|---|
| Hand SIMD intrinsics for the φ v-half loop beyond the codegen flag | post-flag the loop is histogram-bound (scalar store-to-load chains); free weights pass caps gain at ~1 % e2e | 2026-06-11 |
| Histogram widening (4→8 sub-counts) | wash-to-loss at production sizes (−12 % at k+1=8) | 2026-06-11 |
| Bitsliced orbit-BFS image batches | matmul is 7× cheaper but the mandatory transpose-out is 82 % of the cost, doesn't auto-vectorise; per-ISA hand transpose ⇒ ~1.05× e2e | 2026-06-11 |
| AVX2 gathers for the byte-table loads | L1-resident scalar loads already sustain ~2/cycle | 2026-06-11 |
| Any codegen lever on the orbit-BFS body | flat (±1.5 %) under v3; L1-load + bitset bound | 2026-06-11 |
| Probe-side BFS restructures (single-access put / batch-then-probe / radix-bucket) | 0.82–1.04× on real inputs; BFS is compute-bound on image generation, not probe-bound | 2026-06-10 |
| Generator dedupe / inverse-drop for the BFS | sound but near-empty (9,590 → 9,410 images) | 2026-06-10 |
| Generating-subset BFS (use 2–3 generators) | exactness needs orbit-equality verification that costs back the savings | 2026-06-11 (parked, needs a cheap group-equality certificate) |
| Probe-free closures (sliced hashes, Bloom prefilters, label propagation) | random-access membership is the irreducible scalar core | 2026-06-11 |
| "Fit the hot loops in L1" | no cache cliff exists anywhere in the kernel (WHT/Gray sweeps stream; hot-vs-evicted replay < 2 % apart) | 2026-06-10 |
| Adaptive seeder-pool gate (pool when workers starve) | workers are not starved during the k=3 spans; the static min_l=22 gate is correct | 2026-06-10 |
| Shared cross-worker canon LRU | intra-worker hit rate 3.13 % caps the win at ~4 %, under the engineering bar | 2026-05-23 |
| Adaptive per-seed frontier depth | +43–56 % regression at N=24 (seed-set explosion + channel backpressure) | 2026-05-23 |
| Class-fingerprint caches / cheap equivalence rejectors (whole family) | per-probe cost ≈ per-call cost; collision blocklists presuppose the enumeration | 2026-05-17/18, closed permanently |

The one *category* distinction worth remembering: every "cheaper
rejector bolted onto the old parent rule" died of per-probe cost; the
rule-redefinition lever won by changing **which question the parent test
asks**. Judge new ideas by which category they're in.

---

*Maintainers: the session-by-session development history behind these
results — including the mapping from internal change labels to the
descriptive names used here — lives in the unversioned maintainers' tree
under `markdown/architecture/` (not shipped with this repository).*
