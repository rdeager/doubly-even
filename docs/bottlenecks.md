# Current bottleneck profile (living document)

> **As of:** 2026-06-12 evening (external-feedback review session:
> **autom-only canon SHIPPED** — `getcanon=FALSE` on the 80.7 % of
> canon calls whose labelling no decision consumes; 1.09× seq across
> N=22–26, decisions bit-identical, kill-switch
> `DOUBLY_EVEN_CANON_LABELLING=full`) · **wheel:** `--features
> parallel`, AVX2 build · **box:** 13700K-equivalent dev container
> (24 logical, AVX2, no AVX-512), 62 GB.
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
| 22 | 0.623 s    | 0.24 s               | 5,118 |
| 24 | 5.72 s     | ~1.7 s               | 37,496 |
| 26 | 74.6 s     | 9.21 s               | 494,272 |
| 27 | —          | ~64 s                | 2,673,492 |

Medians of 3, single-wheel knob A/B, 2026-06-12 (autom-only canon
arm; the `DOUBLY_EVEN_CANON_LABELLING=full` control reproduced the
2026-06-11 walls — 0.683 / 6.27 / 81.5 s seq — so the lever is the
whole delta: **1.097× / 1.096× / 1.092× seq at N=22/24/26, 1.035×
par N=26**; the seq win exceeds the par win because parallel N≤26 is
seeder-bound. N=27 par measured flat within a noisy session — rep
spread ±10 % on both arms; the 63.0 s record stands as ~63–66 s). Earlier same-session context: the **Cargo workspace
restructure** (perf-neutral) and the **x86-64-v3 codegen flag**
(1.01–1.05× cool-box). N=26 sequential needs
`DOUBLY_EVEN_CANON_CACHE_CAP=500000` (uncapped OOMs); even capped
it is OOM-adjacent on a 62 GB box (silent mid-run SIGKILLs observed —
use an RSS poller if a rep goes missing).

Cloud records (both predate every lever shipped since 2026-06-10, so a
re-run is cheap): N=28 in 61 min on GCP c4a-standard-72 (~$3); N=29 in
12.32 h on the same platform (~$35), 239,465,540 classes, certificate at
[`results/n29.json`](results/n29.json).

## 2. Phase shares (sequential N=26; post-autom-only, 2026-06-12 evening, wall 74.6 s)

| phase | seconds | share | trend |
|---|---:|---:|---|
| canonicalisation (nauty) | 36.0 | 48.3 % | autom-only cut it 42.9 → 36.0 s (per-call 80.6 → 68.9 µs); the call count stays at the one-per-class floor — see the arithmetic below |
| σ_Q candidate generation | 26.9 | 36.0 % | unchanged in seconds; now the second-largest phase by a thinner margin |
| φ cascade (parent rule)  | 8.9 | 11.9 % | beaten down from 59.9 % by the sharing/amax/chain levers; fastest-growing phase with N (×20 per 2-step vs ×10 for the others) |
| other (recursion, RREF, cache) | ~2.8 | ~3.7 % | |

**Canon-call arithmetic (2026-06-12).** At every N:
`canon_calls + cache_hits = classes + tie_rejects`, exactly. calls/class =
1.003 / 1.025 / 1.053 / 1.059 at N=18/22/24/26; tie-rejects are 5.6 % of
calls at N=26. **The call count is at the one-call-per-class floor** —
every emitted class needs Aut(C) for the mass certificate and the next
level's candidate generation. Any further "call nauty less often" idea is
dead by arithmetic (ceiling = the 5.6 % tie-reject slice ≈ 2.9 % e2e).

**Inside the canon phase (per-call timers + sampling profiler,
2026-06-12, pre-autom-only).** sparsenauty proper is **94 %** of the
phase (75.7 µs/call mean at N=26 with the canonical pass; 68.9 µs/call
post-autom-only — the 19–25 %-of-call canonical pass is now paid only
on the 19.3 % tie calls); the wrapper (codeword collect, graph build,
canonical extract, exact |Aut|) is 6 %. Whole-process self-time at N=24:
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
the candidate path is a further measured 5.5 % of N=24 wall (lever 6,
§4). For canon: among measured generic nauty/Traces/dense encodings of
this graph shape, sparsenauty is currently the floor (~76 µs/call at
N=26; per-call cost *falls* with N) — remaining canon wins likely
require shrinking/factoring the object (decomposition, twin
compression — see lever 4), changing the requested output (autom-only,
lever 1), or reusing structure across the augmentation tree
(stabilizer-chain reuse, parked).

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

1. ✅ **Autom-only canon on non-tie accepts — SHIPPED 2026-06-12
   evening.** `getcanon=FALSE` on the 80.7 % of canon calls whose
   canonical labelling no decision consumes (the φ outcome is known
   *before* `canon_info`; only ties feed `tie_break_parent`). Measured
   at the top of the 1.06–1.09× estimate: **1.097× / 1.096× / 1.092×
   seq at N=22/24/26**, 1.035× par N=26 (seeder-bound); decisions
   bit-identical (gate `scripts/experimental/canon_labelling_ab_gate.py`
   PASSED; label upgrades 0). Larger effect expected at N ≥ 28 where
   canon is ~68 % of forecast core-hours. Kill-switch
   `DOUBLY_EVEN_CANON_LABELLING=full`. Writeup: algorithm.md lever 10.
2. **N ≥ 28 cloud re-run** — the highest-value datapoint. N=29
   count-anchored forecast: **0.9–1.3 h on c4a-72** (~$3–5) vs the
   12.32 h actually paid pre-levers. Re-verify d on the first N ≥ 28 run;
   A/B the aarch64 `target-cpu=neoverse-v2` pin (SVE2) while there. If
   the pin alone moves nothing, the queued follow-up is an SVE2
   in-register histogram prototype for the φ v-half (HISTCNT/TBL — the
   65-bin histogram fits in Z-registers; external suggestion, see the
   2026-06-12 feedback review): irrelevant at N ≤ 27 (~1.05× ceiling)
   but φ is ~47 % of the N=32 floor.
3. **Counts-only compact output mode** — required for the N=30 run
   (output budget 100–200 MB; no per-class stream): fold per-rank
   {classes, mass (bigint), |Aut| histogram} in the driver, emit one JSON.
   Composes with lever 1 (counts mode needs no canonical labelling on any
   accept). ~1–2 days.
4. **Decomposability + twin-column logging experiment** (NEW 2026-06-12,
   from the external-feedback review — the strongest external idea):
   for every canon-call input at N=24/26 log support size, connected
   components of the column matroid on the RREF, and twin-column
   (identical k-bit column) classes; report fractions by rank, weighted
   by per-rank nauty cost. Read-only, ~0.5 day. Promotes or kills two
   exact structural canon levers at once — direct-sum component
   canonicalisation (component canonical forms are *exact* keys, so the
   dead cheap-rejector arithmetic does not apply; |Aut| assembles as
   wreath products) and pre-nauty twin compression — plus the
   component-convolution φ option at N=32. Promotion threshold: ≥ 30 %
   of rank-weighted nauty time on decomposable inputs. Implementation
   (if promoted) is decision-changing → D15-style audit gate.
5. **Dense-level direction-switch BFS** (sequential σ_Q candidate):
   blind-mark + extract on dense BFS levels (Beamer-style bottom-up).
   Portable, exact, ~1.15–1.25× BFS-local ≈ 1.04–1.06× end-to-end,
   ~a day in `orbit.rs`. The only surviving σ_Q idea from the SIMD
   investigation; stacks with the flag.
6. **Radix sort for the σ_Q lift sort**: 5.5 % of N=24 wall is
   Rust's comparison sort on u64 keys inside the orbit-minima path
   (sampling profile, 95.6 % attributed). LSD radix on u64 at these sizes
   is typically 2–4×. Est. ~1.03–1.04× e2e, portable, exact. ~0.5–1 day.
   (The external "delete the sort via natural ordering" variant is dead —
   §6; the sort is on *lifted* F_2^N values and the lift is not monotone.)
7. **Parallel k=3 σ_Q fan-out** (across calls, not within a BFS) — the
   measured seeder lever, but **desktop-parallel only**: N ≥ 27 is
   worker-throughput-bound (§3), so its cloud N ≥ 28 value is ~nil.
   Demoted accordingly.

Realistic stack of 1+5+6 ≈ **1.15–1.2× sequential**; no remaining lever
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
| Gleason-polynomial pre-φ candidate reject (external) | category error: candidate *validity* is decided in σ_Q generation (candidates are exactly the singular vectors = doubly-even extensions); φ rejection is parent *selection*, not validity. Gleason's ring also only constrains *self-dual* Type II enumerators, not general [N,k] doubly even. As a parent-test prefilter it's the dead cheap-rejector family | 2026-06-12, feedback review |
| Cache-oblivious linear-sweep orbit closure (external) | attacks latency that isn't there: BFS is compute-bound on image generation (probe restructures 0.82–1.04×) and no cache cliff exists; the full-2^L sweep does strictly *more* image work than the BFS frontier | 2026-06-12, feedback review |
| Delete the σ_Q lift sort via natural BFS ordering (external) | minima do emerge ascending in Q-coordinates, but the 5.5 % sort is on the *lifted* F_2^N values and the lift is not monotone; DFS consumes F_2^N order (decision-bearing). Make it faster (radix, lever 6), not absent | 2026-06-12, feedback review |
| Trellis / split-DP φ histograms (external) | near-zero on generic full-width codes (trellis width ≈ k); residual φ is a compute-bound stream after the E-chain decides 43 % O(1). Conditional revival: only if the decomposability experiment (lever 4) finds components common AND N=32 is committed — and then component-spectrum *convolution* supersedes a trellis | 2026-06-12, feedback review (parked w/ condition) |
| Singular-only σ_Q restriction (external, not dead — **already shipped**) | listed here to stop re-proposal: BFS seeds are exactly the singular reps (`orbit.rs::singular_reps_q` → `candidates.rs`); the "2^L universe" in older docs referred to the bitset index space, not the seed set | 2026-06-12 |

External-feedback adjudications above are argued in full in
`markdown/notes/external-feedback-review-2026-06-12.md` (12 proposals,
2 admitted as the lever-4 measurement, 4 dead, 1 already shipped,
1 = lever 1 independently re-derived).

The one *category* distinction worth remembering: every "cheaper
rejector bolted onto the old parent rule" died of per-probe cost; the
rule-redefinition lever won by changing **which question the parent test
asks**. Judge new ideas by which category they're in.

---

*Maintainers: the session-by-session development history behind these
results — including the mapping from internal change labels to the
descriptive names used here — lives in the unversioned maintainers' tree
under `markdown/architecture/` (not shipped with this repository).*
