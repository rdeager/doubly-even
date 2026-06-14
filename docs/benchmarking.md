# Benchmarking and profiling runbook

How to measure this enumerator without fooling yourself. This is the
developer's reference; the outsider's clean-checkout-to-result recipe is
[`reproducing.md`](reproducing.md), and the *current* numbers live in
[`bottlenecks.md`](bottlenecks.md).

## 1. Bench-a-change checklist (TL;DR)

1. Build + install the wheel: `scripts/install-kernel.sh parallel`
   (always from the repo root).
2. **Verify the new wheel is actually loaded** (§2) — probe a stat or
   symbol your change owns before trusting any number.
3. Correctness gates: `uv run pytest --run-slow`; for any φ/parent-rule
   change additionally the audit gate (§5).
4. A/B bench with fresh labels (§3): same session, both arms, median of
   3 — compare ratios, not absolutes.
5. Decision-identity check between arms (§5).
6. Keep the JSONs (`scripts/bench-results/`, gitignored); delete
   wrong-artifact runs immediately — stale labels poison later
   `--glob` fits.
7. If a lever shipped: walk the maintenance checklist at the top of
   `bottlenecks.md`.

## 2. Build and install discipline

**One wheel, one label.** Same-version wheels have identical filenames
and silently overwrite each other in `rust/target/wheels/`. For an A/B,
copy each arm's wheel aside (e.g. `scripts/bench-results/wheels-<arm>/`)
so either arm can be reinstalled exactly.

**The uv working-directory trap.** Run `uv` (and the install script)
only from the repo root. Running `uv` from `rust/` creates a stray
`rust/.venv`, can trigger a default-features rebuild, and leaves a
second wheel that breaks the install glob — the symptom is benching a
*stale kernel while believing it's new*. Before every gated measurement,
prove the installed kernel is the one you built:

```sh
.venv/bin/python -c "
import doubly_even_kernel as k
print(k.kernel_build_info(), k.__file__)
print(hasattr(k, 'kernel_stats_layout'))   # present since the workspace restructure
"
```

and, for a code change, check that a counter only the new code can move
is nonzero in a small-N smoke run.

**The static-TLS landmine.** The wheel is dlopen'd by Python and must
stay free of the `STATIC_TLS` ELF flag (a TLS-model regression once made
imports fail and produced a spurious 75× slowdown). After any
allocator/TLS-adjacent change:

```sh
readelf -d "$(.venv/bin/python -c 'import doubly_even_kernel as k; print(k.__file__)')" | grep -c STATIC_TLS
# expect 0
```

**Smoke check:** N=18 must give 341 classes in ~0.03 s.

## 3. bench.py usage and standard configurations

```sh
# sequential, small N
uv run python scripts/bench.py --label <arm>-seq --N 18,22,24
# sequential N=26 — the cache cap is mandatory (uncapped OOMs)
DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
  uv run python scripts/bench.py --label <arm>-seq-n26 --N 26
# parallel N>=24 — full logical-core count, frontier depth 4
DOUBLY_EVEN_THREADS=24 DOUBLY_EVEN_FRONTIER_DEPTH=4 \
  DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
  uv run python scripts/bench.py --label <arm>-par-n26 --N 26
```

Standard knob choices (re-verify on new hardware, especially the first
N ≥ 28 cloud run):

| knob | value | note |
|---|---|---|
| `DOUBLY_EVEN_THREADS` | cores − 2 at N ≤ 22; full logical count at N ≥ 24 | hybrid topologies are empirical |
| `DOUBLY_EVEN_FRONTIER_DEPTH` | 4 | d=4 best ≤24 cores; **d=5 is the knee at 96 cores** (N=28 d=4/5/6 = 301.6/139.8/220.1 s, cloud 2026-06-14) — a granularity knob, deeper splits the tail finer for a longer serial seeder walk |
| `DOUBLY_EVEN_CANON_CACHE_CAP` | 500000 at N=26 local; 100000 on 96-core cloud; 200000 on ≥200-core machines | per-worker cap; **inert for speed** (0.003 % hit post-D15), a memory/OOM knob only |
| `DOUBLY_EVEN_SEEDER_THREADS` | default (= threads) | 0 disables the seeder pool (A/B control) |
| `DOUBLY_EVEN_SEEDER_PAR_MIN_L` | 22 (default) | load-bearing; lowering it loses to helper-vs-worker contention |
| `DOUBLY_EVEN_MASS_FLUSH_INTERVAL` | 2048 | parallel only: emissions per worker between shared-mass-tracker flushes. Larger = fewer tracker locks (the 96-thread contention fix). A **contention** knob, not a pruning knob — mass-stop is ~inert post-D19 (≤26 candidates / 5 canon calls pruned at N=27 even at interval=1), so classes + mass are identical at any value; tune only for cloud `sy`-time |
| `DOUBLY_EVEN_PARENT_RULE` | default | `legacy` is the whole-rule kill-switch, `audit` the measurement mode |
| `DOUBLY_EVEN_CANON_LABELLING` | default (`autom-only`) | `full` is the autom-only-lever kill-switch: computes nauty's canonical labelling on every call and restores per-class `canonical_column_order` in the output |
| `DOUBLY_EVEN_TIE_DUMP` | unset | path to a JSONL sink for φ-tie records (collision analysis). **Sequential drivers only** — parallel drivers panic. Analysis: `scripts/experimental/tie_collision_analysis.py` |
| `DOUBLY_EVEN_SELF_SUBDIVIDE` | off (merged to main, OFF default) | demand-driven self-subdivision (D20): a busy worker at shallow depth donates an accepted child onto the seed channel when peers are idle, deepening the frontier adaptively in the heavy N>27 tail. Victim-initiated work-sharing, not work-stealing. **OFF byte-identical to main.** Local ladder PASS; first scaled signal N=27 **1.30×** (contended, single run). Cloud `(d, δ)` sweep determines the production default |
| `DOUBLY_EVEN_SELF_SUBDIVIDE_DELTA` | 1 | D20: donatable depth past `frontier_depth` (parent gate `k ≤ frontier_depth+δ`; finest granularity `frontier_depth+δ+1`). Decouples granularity from seeder span — co-optimise via `scripts/cloud_depth_sweep.py` |
| `DOUBLY_EVEN_SELF_SUBDIVIDE_POLL_MS` | 2 | D20: `recv_timeout` poll interval for the donation-aware worker loop (shutdown latency, negligible vs a minutes-long run) |

The **`(frontier_depth × delta)` cloud sweep** is `scripts/cloud_depth_sweep.py`:
a 2-D grid at the cheap N=27 size with a **hard ≤2 min per-arm budget**
(kill-on-overrun), reporting wall + the tail metric (wall − time-to-90 %-mass,
read live from `progress.json`) and a correctness gate vs the known class count.
Run it on the clean 96-core box (the "d=5 knee" was a no-lever 1-D result;
self-subdivision makes the optimum 2-D and likely shallower-static).

Each invocation runs each `N` once and writes one JSON
(`scripts/bench-results/<timestamp>-<label>.json`) keyed by `per_N`,
containing `seconds`, `classes`, `per_k`, `kernel_stats` (by field
name), and `per_k_stats`. **Every run re-verifies the DFGHILM Table 3
cells and the mass formula** — a bench that finishes is also a
correctness witness. Median-of-3 means three invocations with `-rep1/2/3`
label suffixes.

## 4. Kernel stats schema

The stats vector (51 fields) and per-rank matrix (19 rows) are
**single-sourced from the kernel**: the layout constants live next to
the code that fills them (`enumerate::stats` in the kernel crate) and
are exported to Python as `doubly_even_kernel.kernel_stats_layout()`.
`scripts/bench.py` consumes that at import time (with a frozen fallback
for pre-restructure wheels and an append-only prefix assertion). Don't
hand-maintain a copy anywhere else.

Semantics of the load-bearing fields:

| field(s) | meaning |
|---|---|
| `canon_calls`, `nauty_ns`, `nauty_ns_kept` | canonicaliser calls / total ns / ns on kept (accepted) candidates — κ = `nauty_ns_kept / legacy nauty_ns` |
| `phi_reject`, `phi_accept_unique`, `phi_tie_*` | parent-rule cascade outcomes |
| `phi_ns`, `phi_ctx_ns` | total φ time; per-parent context build time (subset of `phi_ns`) |
| `phi_s1_fastpath`, `phi_chain_fastpath` | O(1) decisions at the first stratum / at later strata via the chain |
| `phi_strata_sum` | total strata walked — a sensitive decision-path fingerprint, used in identity gates |
| `candidates_q_ns`, `cq_*_ns` | σ_Q candidate generation total + 5-way sub-phase split (`phase_timers` builds) |
| per-rank rows 14–16 | `phi_ns` / `candidates_q_ns` (by parent rank), `nauty_ns` (by **child** rank — off-by-one vs the others) |

Timer fields (`*_ns`) are populated always-on for rows 14–16 and the
aggregates; the finer sub-phase splits are zero unless the kernel was
built with `--features phase_timers` (§6).

## 5. Correctness and decision-identity gates

Always-on: the Table 3 check and the mass-formula panic (every bench,
every production run).

**For any change touching the parent rule / φ cascade**, the hard gate
is the audit harness: `DOUBLY_EVEN_PARENT_RULE=audit` runs legacy
behaviour with φ tallies, and
`scripts/experimental/d15_phi_audit.py` asserts per-rank φ-accepts ==
legacy accepts *exactly* (integer decisions, no tolerance), mass-stop on
and off.

**A/B decision identity** (the refactor gate — "exact" levers must not
move any decision):

- *Sequential arms*: `classes` (total and per-k) and **every non-`_ns`
  counter** must be bit-equal — in particular `canon_calls`,
  `phi_strata_sum`, `phi_s1_fastpath`, `phi_chain_fastpath`, and all
  per-k counter rows. **Exception (autom-only labelling A/Bs only)**:
  the nauty tree-shape sums (`nauty_numnodes_sum`, `nauty_tctotal_sum`,
  `nauty_maxlevel_sum`) legitimately SHRINK under
  `DOUBLY_EVEN_CANON_LABELLING=autom-only` — that drop is the lever —
  and the two mode counters (`canon_autom_only_calls`,
  `canon_label_upgrades`) only exist on one arm. `nauty_generators_sum`
  is also excluded — measured ±1 at N=24 (1 in 4e5 calls): nauty may
  emit a *different generating set* of the same group without the
  best-leaf bookkeeping, which is decision-neutral (orbits / |Aut| /
  classes / strata sums stay bit-equal — those ARE the gate). Stock
  gate: `scripts/experimental/canon_labelling_ab_gate.py`.
- *Parallel arms*: `classes`, per-k classes, and the mass certificate
  must be exact; call counters are race-variable by design (the shared
  mass-stop lets workers briefly over-search depending on timing) —
  compare them against the min/max envelope of two same-wheel runs
  instead of demanding equality.

```sh
uv run python - <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for n in a["per_N"]:
    ka, kb = a["per_N"][n]["kernel_stats"], b["per_N"][n]["kernel_stats"]
    diff = {f: (ka[f], kb[f]) for f in ka
            if not f.endswith("_ns") and ka[f] != kb.get(f)}
    print(n, "OK" if not diff else diff)
EOF
```

## 6. Measurement Cargo features

| feature | gives | cost / caveat |
|---|---|---|
| `phase_timers` | σ_Q 5-way and φ 5-way sub-phase splits; φ sampled 1-in-64 via the portable cycle counter | overhead gate ≤ 1.02× (measured 1.008×/0.998×); production *shares* still come from feature-OFF builds |
| `parallel_profiling` | per-worker active/idle + per-seed timeline; seeder timeline is a 7-tuple payload | JSONs produced before 2026-06-10 used a two-phase shim and are **not comparable** |
| `parallel` | the worker-pool build benched everywhere | sequential path is byte-identical with the env var unset |

## 7. Microbench suite (`scripts/microbench/`)

Standalone Cargo project; bins link the kernel's core crate directly, so
the "production arm" of each bin **is** production code — experimental
variants stay local to the bin and are asserted equal where exactness is
claimed. Pin a core when timing: `taskset -c 4 …`.

| bin | measures |
|---|---|
| `phi_replay` | the φ cascade on synthetic frames; `--mode conly` exercises the chain arms; `--validate` checks production decisions against a local brute-force spectrum oracle |
| `vhalf_sweep` | φ phase 0 (the v-half XOR+popcount+histogram loop) in isolation — the SIMD-shaped target |
| `wht_sweep` | the WHT butterfly across table sizes (cache-cliff sweep) |
| `orbit_probe` | the σ_Q orbit-min BFS replayed on **real dumped inputs** (446 rank-2/3 parents from N=26/27); variants asserted minima-identical to production |
| `singular_walk` | the singular-reps Gray walk (cache-cliff sweep) |
| `simd_probe` | exploratory SIMD prototypes (apply/probe splits, bitsliced microkernels) vs production shapes |
| `nauty_decomp` | sparsenauty internals via statsblk counters |

Real-input dumps regenerate via the ignored kernel test:

```sh
cd rust && cargo test --release dump_sigma_inputs -- --ignored --nocapture
```

(writes `scripts/bench-results/sigma-inputs/`, one file per parent).

Synthetic inputs are a last resort for σ_Q work: random-GL generators
have the wrong orbit/generator structure; replay real dumps.

## 8. Instruction-level profiling

The dev container needs two host-side grants: `--cap-add PERFMON` *and*
`--ulimit memlock=-1:-1` (CAP_PERFMON does not bypass mlock accounting;
the spawn script's `--perf` mode sets both). Host
`kernel.perf_event_paranoid=1` suffices. Then:

```sh
# build with line tables + frame pointers (codegen identical to release)
RUSTFLAGS="-C force-frame-pointers=yes" CFLAGS="-fno-omit-frame-pointer" \
  maturin build --profile profiling -m rust/Cargo.toml --features parallel
samply record -r 1000 -- .venv/bin/python scripts/bench.py --label prof --N 24
```

(`CFLAGS` reaches the bundled nauty C; needs a clean rebuild of
`nauty-Traces-sys` to take effect.) samply 0.13.1 verified end-to-end in
this container.

Traps (hit 2026-06-12 — all three will silently break this recipe):

- `rust/pyproject.toml` sets `[tool.maturin] strip = true`, which strips
  even `--profile profiling` wheels (symbol count 0, no attribution).
  Workaround: `cargo build --profile profiling --features parallel` from
  `rust/`, then copy `target/profiling/libdoubly_even_kernel.so` over the
  installed module `.so` (verify import + `nm | wc -l` > 0 after).
- An env `RUSTFLAGS` **replaces** (does not merge with) the
  `rust/.cargo/config.toml` rustflags — the recipe above as written drops
  the v3 flag. Use
  `RUSTFLAGS="-C target-cpu=x86-64-v3 -C force-frame-pointers=yes"` and
  re-verify `kernel_target_features()` afterwards.
- `perf_event_paranoid` defaults to **4** on this host (the container
  holds no CAP_PERFMON); ask the host to set it to 1 when a profiling
  session is planned — don't assume the grant persists across spawns.

PGO experiments (closed 2026-06-12, see `bottlenecks.md` §6 — recipe kept
for reproducibility): instrumented arm needs
`RUSTFLAGS="-C target-cpu=x86-64-v3 -C link-arg=-lgcov" CFLAGS="-fprofile-generate"`;
the profile-use arm must use **identical RUSTFLAGS** (the
nauty-Traces-sys OUT_DIR hash includes them — different flags ⇒ gcc
silently finds no `.gcda`; check the `.gcda` files are present in the
out-dir the rebuild actually used) plus
`CFLAGS="-fprofile-use -fprofile-correction -Wno-coverage-mismatch"`.

## 9. Scaling extrapolation

`scripts/experimental/post_d15_scaling_fit.py --glob '<label-prefix>'
--kappa <κ>` produces the count-anchored N ≥ 28 forecasts: phases are
priced per event count (candidates, kept-canon calls) against the known
true candidate counts, rather than fitting wall-time geometrically.
Feed it only a **clean glob** — one wheel, one session, sequential
labels; mixed-wheel globs shift the fit silently (this happened once;
the polluted-glob numbers had to be superseded). Quote outputs as
floors/ranges, never point estimates: the candidate-count growth rate is
the dominant unknown.

## 10. Environment conventions

- Same-session A/B, interleaved arms; the box drifts a few percent
  between sessions, so **ratios are the result, absolutes are not**.
- Median of 3 per configuration; pin microbenches with `taskset`.
- Sequential N=26 is OOM-adjacent on a 62 GB box even with
  `CANON_CACHE_CAP=500000`: a mid-enumeration SIGKILL is *silent* in
  non-interactive shells (output just stops, no JSON). If a rep goes
  missing, that's what happened — re-run it, and use an RSS poller if it
  recurs.
- Cloud quirks live in [`reproducing.md`](reproducing.md) (build
  bootstrap) and `../CLAUDE.md` (GCP gotchas: quota, per-machine cache
  caps, `kernel_build_info()` reporting).
