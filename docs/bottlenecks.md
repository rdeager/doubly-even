# Current bottleneck profile (living document)

> **As of:** 2026-06-12 (speedup-evaluation session: no levers shipped;
> phase shares re-measured on the v3 wheel, canon phase decomposed,
> build levers A/B'd dead, two new levers ranked) · **wheel:**
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

## 2. Phase shares (sequential N=26; re-measured 2026-06-12 from the v3 wheel's own counters, wall 81.4 s)

| phase | seconds | share | trend |
|---|---:|---:|---|
| canonicalisation (nauty) | 42.9 | 52.7 % | at a *double floor* — see the call arithmetic and the per-call split below |
| σ_Q candidate generation | 26.9 | 33.0 % | was ~41 % before the four-Russians BFS; residual is close to its scalar floor |
| φ cascade (parent rule)  | 8.9 (+1.7 ctx) | 10.9 % (+2.1 %) | beaten down from 59.9 % by the sharing/amax/chain levers; fastest-growing phase with N (×20 per 2-step vs ×10 for the others) |
| other (recursion, RREF, cache) | ~1.1 | ~1.3 % | |

**Canon-call arithmetic (2026-06-12).** At every N:
`canon_calls + cache_hits = classes + tie_rejects`, exactly. calls/class =
1.003 / 1.025 / 1.053 / 1.059 at N=18/22/24/26; tie-rejects are 5.6 % of
calls at N=26. **The call count is at the one-call-per-class floor** —
every emitted class needs Aut(C) for the mass certificate and the next
level's candidate generation. Any further "call nauty less often" idea is
dead by arithmetic (ceiling = the 5.6 % tie-reject slice ≈ 2.9 % e2e).

**Inside the canon phase (per-call timers + sampling profiler,
2026-06-12).** sparsenauty proper is **94 %** of the phase (75.7 µs/call
mean at N=26); the wrapper (codeword collect, graph build, canonical
extract, exact |Aut|) is 6 %. Whole-process self-time at N=24:
`refine_sg` 21.4 % + `targetcell_sg` 12.5 % + `sortints` 4.5 % ⇒
**partition refinement ≈ 38 % of total wall ≈ 81 % of nauty time**;
automorphism verification only ~3.6 %. 81 % of sparsenauty time sits at
ranks k=7–9 where graphs are small (|C_low| median 24–56); the per-call
tail at k ≥ 10 is size-driven but low-volume. Consequences: PGO barely
moves it (measured, §6), and refinement-side invariant hooks add work
exactly where the time already goes (§6).

Sub-phase splits:

| φ residual (sampled) | share | | σ_Q | share |
|---|---:|---|---|---:|
| v-half XOR+popcount+histogram | 67.3 % | | orbit-min BFS | ~87–89 % |
| first-stratum decision | 16.7 % | | singular-reps Gray walk | ~6 % |
| direct parity counting | 10.8 % | | basis/lift/sort | rest |
| per-stratum WHT | 5.1 % | | | |

Character of each: the φ v-half sweep is **compute-bound** (no cache
cliff anywhere — streams at 12 GB/s scalar, 21–22 GB/s under AVX2; its
histogram half is scalar-bound at ~0.19 ns/elem and is the floor;
disassembly re-certified 2026-06-12: weights pass is vpshufb/vpsadbw
AVX2, histogram is scalar increments). The σ_Q orbit-min BFS is
**compute-bound on image generation** (the seen-bitset is L2/L3-resident
at L=20–23; probes are 2–5× cheaper than the old chained walk); the
byte-table method left it ~L1-load-bound, and a u64 comparison sort in
the candidate path is a further measured 5.5 % of N=24 wall (lever 5,
§4). Canon is sparsenauty at its measured algorithmic floor for this
graph shape (~76 µs/call at N=26; per-call cost *falls* with N).

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

## 4. Ranked next levers (re-ranked 2026-06-12; ceilings from §2 shares)

1. **Autom-only canon on non-tie accepts** (NEW, found 2026-06-12 via the
   per-call decomposition): call sparsenauty with `getcanon=FALSE` on the
   80.7 % of canon calls whose canonical labelling no decision consumes —
   the φ outcome (accept-unique vs tie) is known *before* `canon_info` is
   invoked; only ties feed the labelling into `tie_break_parent`. The
   canonical pass is 19–25 % of the sparsenauty call (synthetic A−B;
   re-measure on real graphs first). Est. **1.06–1.09× e2e** at N=26 seq,
   more at N ≥ 28 where canon is ~68 % of forecast core-hours. Zero
   decision risk by construction (same search tree, same generators /
   orbits / |Aut|); cache entries lacking the labelling recompute on a
   tie-hit (hit rate 0.005 %). ~1 day.
2. **N ≥ 28 cloud re-run** — the highest-value datapoint. N=29
   count-anchored forecast: **0.9–1.3 h on c4a-72** (~$3–5) vs the
   12.32 h actually paid pre-levers. Re-verify d on the first N ≥ 28 run;
   A/B the aarch64 `target-cpu=neoverse-v2` pin (SVE2) while there.
3. **Counts-only compact output mode** — required for the N=30 run
   (output budget 100–200 MB; no per-class stream): fold per-rank
   {classes, mass (bigint), |Aut| histogram} in the driver, emit one JSON.
   Composes with lever 1 (counts mode needs no canonical labelling on any
   accept). ~1–2 days.
4. **Dense-level direction-switch BFS** (sequential σ_Q candidate):
   blind-mark + extract on dense BFS levels (Beamer-style bottom-up).
   Portable, exact, ~1.15–1.25× BFS-local ≈ 1.04–1.06× end-to-end,
   ~a day in `orbit.rs`. The only surviving σ_Q idea from the SIMD
   investigation; stacks with the flag.
5. **Radix sort for the σ_Q lift sort** (NEW): 5.5 % of N=24 wall is
   Rust's comparison sort on u64 keys inside the orbit-minima path
   (sampling profile, 95.6 % attributed). LSD radix on u64 at these sizes
   is typically 2–4×. Est. ~1.03–1.04× e2e, portable, exact. ~0.5–1 day.
6. **Parallel k=3 σ_Q fan-out** (across calls, not within a BFS) — the
   measured seeder lever, but **desktop-parallel only**: N ≥ 27 is
   worker-throughput-bound (§3), so its cloud N ≥ 28 value is ~nil.
   Demoted accordingly.

Realistic stack of 1+4+5 ≈ **1.15–1.2× sequential**; no remaining lever
family reaches 2× locally (the call-count arithmetic in §2 closed the
last one). The 2×-class ideas left are research-grade: incremental
Aut-group transfer along the augmentation tree (replacing nauty on
accepts entirely) and GPU canonicalisation — both parked.

## 5. Scaling forecast N=28 → 32

Method: count-anchored extrapolation
(`scripts/experimental/post_d15_scaling_fit.py --glob <clean-label> --kappa <κ>`)
— phases priced per *event count* (candidates, kept-canon calls) rather
than geometric wall fits; use only same-wheel same-session label globs
(a polluted glob quietly shifts the fit).

| target | forecast | confidence / blocker |
|---|---|---|
| N=28 | well under the 61-min record; single c4a-72 | re-run is routine |
| N=29 | 0.9–1.3 h on c4a-72 (41 core-h: kept-canon 28, φ 9, σ_Q 4) | count-anchored, re-fit 2026-06-12 on the v3 glob |
| N=30 | **single Axion box, counts-only output**: ~250 ± 100 core-h (×6/step from N=29) ⇒ ~4–6 h on a 96-core c4a, ~$20–50 | needs the counts-only mode (§4 lever 3); the old "streaming + cluster, ~3 TB" plan is obsolete — the output budget is now 100–200 MB (per-rank table + certificate) |
| N=31 | ~×6 again ⇒ ~1,500 core-h; feasible single-box over days, or a small cluster | candidate-count growth is the unknown |
| N=32 | **floor ~44 core-h** single-thread by geometric carry (re-fit 2026-06-12; earlier vintages gave 63/475/174 — the method is a floor, not an estimate; count-anchored ×6/step says ~9,000 core-h) | cluster coordinator + the column-multiset engine for k ≤ 4 (σ_Q BFS universe/time explodes at low rank); re-anchor after the first post-lever cloud datapoint |

A >10× algorithmic speedup is banked since the N=29 run. Two structural
notes from the 2026-06-12 re-fit: **φ grows ~×20 per 2-step of N vs ~×10
for canon and σ_Q** — at N ≥ 30 the parent-rule sweep overtakes canon as
the dominant phase, and φ is ~47 % of the N=32 floor, so the histogram
scalar floor (not canon) is the long-term algorithmic wall. φ
L1-overflow at N=32 is only 4 % of φ time (the L1-fit idea stays dead
even at scale). N=32 planning should wait for a post-lever cloud
datapoint to re-anchor the fit.

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
| Any further "fewer canon calls" idea | call count is exactly classes + tie-rejects (one call per emitted class is the floor — Aut is needed per class); ceiling = the 5.6 % tie-reject slice ≈ 2.9 % e2e | 2026-06-12, closed by arithmetic |
| Fat LTO | 1.004×/0.997× at N=22/24 — noise (thin-LTO + codegen-units=1 already ship) | 2026-06-12 |
| PGO on the nauty C side (gcc, trained N=22/24) | 1.012–1.016× (N=22/24/26), decisions bit-identical — below the ~5 % bar; refinement cost is data-dependent pointer-chasing, not mispredicts | 2026-06-12 |
| Rust-side PGO | blocked locally (no llvm-profdata for rustc's LLVM); ceiling is the non-nauty 45 % in branch-light loops, expected ≲2 % | 2026-06-12 (cloud-idle curiosity at best) |
| Refinement-side nauty invariants (`distances_sg` etc.) | partition refinement is already 81 % of nauty time — invariants add per-node work exactly there; same failure mode as the measured refinement-invariant hook (+259 %) | 2026-06-12, dead on paper |
| Tie-rate reduction as a standalone lever | tie-*accepts* still need their per-class call; only tie-rejects (5.6 % of calls) are removable ≈ 2.9 % e2e, and it would be decision-changing (audit-gated) | 2026-06-12 |

The one *category* distinction worth remembering: every "cheaper
rejector bolted onto the old parent rule" died of per-probe cost; the
rule-redefinition lever won by changing **which question the parent test
asks**. Judge new ideas by which category they're in.

---

*Maintainers: the session-by-session development history behind these
results — including the mapping from internal change labels to the
descriptive names used here — lives in the unversioned maintainers' tree
under `markdown/architecture/` (not shipped with this repository).*
