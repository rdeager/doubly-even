# Current bottleneck profile (living document)

> **As of:** 2026-06-14 (**parallel mass-mutex contention fixed** —
> batched per-worker writes + atomic full-flags replace the per-emit
> `GlobalMassTracker` lock that capped scaling at ~24 threads (85 % `sy`
> at 96t on c4a-96-metal). Correctness gated locally; cure is cloud-only.
> Side finding: mass-stop pruning is ~inert post-D19. See §3
> "High-core wall". Previous: 2026-06-13 — **counts-only output mode SHIPPED** —
> `enumerate_doubly_even_counts` + `scripts/run_counts.py` +
> `dec progress`, per-rank {classes, mass, |Aut| histogram} with no
> per-class records, the N ≥ 30 prerequisite; **mass spine widened to
> 256-bit** — σ(30, ·) ≈ 2^136 overflows u128 (σ(29, ·) fit by ONE
> bit), so the pre-existing streaming/in-memory entries were never
> N ≥ 30-capable; decisions bit-identical, gate-verified. Same session:
> lever-4 decomp/twin measurement + SVE2 ablation verdicts — see §4/§6.
> Previous milestone 2026-06-12: autom-only canon, 1.09× seq,
> kill-switch `DOUBLY_EVEN_CANON_LABELLING=full`.) · **wheel:**
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

### High-core wall: the global mass mutex (found 2026-06-14, c4a-96-metal)

The §3 table is a ≤24-core (desktop) picture. The first 96-core cloud
outing exposed a *different* binding constraint above ~24 threads: the
single `GlobalMassTracker` mutex. Every emitted class write-locked it and
every candidate read-locked it; harmless until D15→D19 cut compute/class
to ~50 µs, after which the lock is hit a few µs apart globally. Thread
ladder at N=28/29 counts d=4 on c4a-96-metal: **24t → 2.8 % `sy`, 48t →
34 %, 96t → 85 %** (effective ~14 of 96 cores — *slower* than the 24-thread
desktop). Diagnose by `top` **us vs sy**, not load-average: high `sy` =
futex contention, not work.

**Fix (2026-06-14, this branch):** batched per-worker writes (flush every
`DOUBLY_EVEN_MASS_FLUSH_INTERVAL=2048` emissions) + a monotonic
`Vec<AtomicBool>` "full" flag the per-candidate read consults lock-free.
~2000× fewer lock acquisitions; the read is a relaxed load. Classes + the
mass-formula certificate stay exact (gated parallel-vs-sequential at
N=18/20 and the N=12/14/16 determinism suites). The dev container is
20-core CFS-capped so it cannot reproduce the `sy` collapse — the
contention cure is confirmed on a fresh c4a-96-metal redeploy (low `sy`
at 96t), the local box only proves correctness.

**Side finding (deterministic counters, contention-immune):** mass-stop
*pruning* is now **~inert** in the parallel path. At N=27, flushing every
emission (interval=1, tightest pruning) skips **26 candidates / 5 canon
calls** out of 2.85 M; at interval ≥ 2048 it skips **0**. canon_calls are
byte-identical OFF / i256 / i2048 / i16384 (calls ≈ classes + ties post-D19
— nothing left to prune). So `MASS_FLUSH_INTERVAL` is a *contention* knob,
not a pruning knob, and the batch delay erodes no real win. The tracker
stays (it carries the live progress signal + the correctness certificate);
only the *mass-stop* could be dropped from the parallel path as dead weight
— a future simplification, not done here. The legacy "4–11 % mass-stop win"
(D5) was sequential + pre-D15.

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
2. **N ≥ 28 cloud re-run** — the highest-value datapoint, now unblocked
   above 24 cores by the §3 mass-mutex fix (redeploy + confirm low `sy`
   at 96t first). N=29 count-anchored forecast: **0.9–1.3 h on c4a-72**
   (~$3–5) vs the 12.32 h actually paid pre-levers. Re-verify d on the
   first N ≥ 28 run;
   A/B the aarch64 `target-cpu=neoverse-v2` pin (SVE2) while there. If
   the pin alone moves nothing, the queued follow-up is an SVE2
   in-register histogram prototype for the φ v-half (HISTCNT/TBL — the
   65-bin histogram fits in Z-registers; external suggestion, see the
   2026-06-12 feedback review). **Ceiling measured 2026-06-13** (ablation
   arm in `vhalf_sweep`, architecture-independent bound): the histogram
   is 47–48 % of phase 0 ⇒ hist-free e2e ceiling **1.04× at N=26 /
   ~1.07× at N=29 / 1.18–1.20× at N=32**; realistic (hist 3×) ~1.11× at
   N=32. Prototype stays queued conditional on an N=32 commitment; the
   external ">1.5×" figure is refuted. See
   `markdown/notes/sve2-ablation-2026-06-13.md`.
3. ✅ **Counts-only compact output mode — SHIPPED 2026-06-13.**
   Fold-only drivers (`enumerate_doubly_even_counts` seq/parallel) emit
   per-rank {classes, Σ N!/|Aut|, |Aut| histogram} and NO per-class
   records; runner `scripts/run_counts.py` writes the n29.json-format
   result; live progress via an in-kernel watcher (`progress.json`,
   atomic rewrite) rendered by `uv run dec progress` (works for
   streaming runs too). En-route fix that was a hard N=30 blocker:
   **σ(30, ·) ≈ 2^136 overflows the u128 mass spine** (quota,
   `mass_at_k`, `GlobalMassTracker`; σ(29, ·) fit by one bit) — all
   mass accumulation is now 256-bit (`core/src/u256.rs`); quota travels
   as decimal strings through pyo3 in the counts entry. Decisions
   bit-identical (bench gate: canon calls 342/5248/39491 at N=18/22/24,
   walls within noise). Equality-tested against the in-memory driver at
   N=12/14/16, seq + par.
4. **Twin-column compression pre-nauty — gated on a per-call A−B**
   (re-scoped 2026-06-13: the logging experiment RAN; full verdict in
   `markdown/notes/decomp-twin-logging-2026-06-13.md`). Measured at
   N=24/26 (`DOUBLY_EVEN_DECOMP_LOG` hook + 
   `scripts/experimental/decomp_log_audit.py`, 100 % nauty-ns coverage):
   component canonicalisation and zero-column special-casing are DEAD
   (§6 — proper direct sums carry 2.2 % of nauty time at N=26, falling
   in N and k; zero-col deficit is 0.4 columns ns-weighted), but
   **93.2 % of nauty time sits on twin-bearing inputs** with an
   ns-weighted mean of 4.2/26 removable columns. If per-call cost tracks
   column count: ~5–7 % e2e seq at N=26, ~8–10 % at N ≥ 28 — the
   automorphism-only-canonicalisation tier. Next gate (~0.5 day): extend `nauty_decomp` to A−B original
   vs twin-compressed colored graphs on real dumped inputs. Implementation
   (if it clears ~5 % e2e) composes with automorphism-only canonicalisation — compress only on
   autom-only calls so tie-break labelling is untouched — but orbits
   feed σ_Q, so it still takes a coset-spectrum-style audit gate.
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
| N=30 | **single Axion box, counts-only output**: ~250 ± 100 core-h (×6/step from N=29) ⇒ ~4–6 h on a 96-core c4a, ~$20–50 | counts-only mode SHIPPED 2026-06-13 (§4 lever 3) — `run_counts.py` is the ONLY N=30-capable entry (the u128 mass spine of the older entries overflows at σ(30, ·) ≈ 2^136); the old "streaming + cluster, ~3 TB" plan is obsolete — output is one JSON (per-rank table + |Aut| histograms + certificate) |
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
| Any further "fewer canon calls" idea **via dedup** | call count is exactly classes + tie-rejects (one *Aut computation* per emitted class is the floor — Aut is needed per class for the mass certificate + next-level σ_Q); dedup ceiling = the 5.6 % tie-reject slice ≈ 2.9 % e2e. **Scope (2026-06-13): floors *dedup*, not *nauty-per-class*** — "need Aut" ≠ "need a fresh nauty call"; computing Aut without one (incremental Aut-group transfer along the augmentation tree, §4 tail) is the open escape — same `need-Aut ≠ need-canon` distinction the coset-spectrum parent rule used. | 2026-06-12, closed by arithmetic *for dedup* |
| Fat LTO | 1.004×/0.997× at N=22/24 — noise (thin-LTO + codegen-units=1 already ship) | 2026-06-12 |
| PGO on the nauty C side (gcc, trained N=22/24) | 1.012–1.016× (N=22/24/26), decisions bit-identical — below the ~5 % bar; refinement cost is data-dependent pointer-chasing, not mispredicts | 2026-06-12 |
| Rust-side PGO | blocked locally (no llvm-profdata for rustc's LLVM); ceiling is the non-nauty 45 % in branch-light loops, expected ≲2 % | 2026-06-12 (cloud-idle curiosity at best) |
| Refinement-side nauty invariants (`distances_sg` etc.) | partition refinement is already 81 % of nauty time — invariants add per-node work exactly there; same failure mode as the measured refinement-invariant hook (+259 %) | 2026-06-12, dead on paper |
| Tie-rate reduction as a standalone lever | tie-*accepts* still need their per-class call; only tie-rejects (5.6 % of calls) are removable ≈ 2.9 % e2e, and it would be decision-changing (audit-gated) | 2026-06-12 |
| Gleason-polynomial pre-φ candidate reject (external) | category error: candidate *validity* is decided in σ_Q generation (candidates are exactly the singular vectors = doubly-even extensions); φ rejection is parent *selection*, not validity. Gleason's ring also only constrains *self-dual* Type II enumerators, not general [N,k] doubly even. As a parent-test prefilter it's the dead cheap-rejector family | 2026-06-12, feedback review |
| Cache-oblivious linear-sweep orbit closure (external) | attacks latency that isn't there: BFS is compute-bound on image generation (probe restructures 0.82–1.04×) and no cache cliff exists; the full-2^L sweep does strictly *more* image work than the BFS frontier | 2026-06-12, feedback review |
| Delete the σ_Q lift sort via natural BFS ordering (external) | minima do emerge ascending in Q-coordinates, but the 5.5 % sort is on the *lifted* F_2^N values and the lift is not monotone; DFS consumes F_2^N order (decision-bearing). Make it faster (radix, lever 6), not absent | 2026-06-12, feedback review |
| Trellis / split-DP φ histograms (external) | near-zero on generic full-width codes (trellis width ≈ k); residual φ is a compute-bound stream after the E-chain decides 43 % O(1). The 2026-06-13 revival condition FAILED: proper components are 2.2 % of nauty time at N=26 and falling — component-spectrum convolution has nothing to convolve | 2026-06-12; condition closed 2026-06-13 |
| Direct-sum component canonicalisation (P1) | measured (decomp-log, N=24/26): proper direct sums carry 5.8 % / **2.2 %** of rank-weighted nauty time, falling in N and into the hot ranks (k=7–9: 1.6–2.0 %); the "decomposables are common" hypothesis holds only at low ranks, which carry no nauty time | 2026-06-13, `decomp-twin-logging` note |
| Zero-column (support < N) canon special-casing | 23.5 % of nauty ns sits on support-deficient inputs but the ns-weighted mean deficit is 0.4 columns — smaller-graph win < 1 % e2e | 2026-06-13, same note |
| SVE2 in-register φ histogram at N ≤ 29 | ablation arm (`vhalf_weights_nohist`): histogram = 47–48 % of phase 0 ⇒ hist-FREE ceiling 1.04×/1.07× e2e at N=26/29; N=32 ceiling 1.18–1.20× keeps the prototype queued there only | 2026-06-13, `sve2-ablation` note |
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
