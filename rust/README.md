# doubly-even-kernel

Native (Rust + PyO3) hot kernel for the `doubly_even` Python enumerator.

## Status

Milestone 5(c) lands. Two production hot paths:

1. `doubly_even_candidates_q(n, code_rref, pivots, dual_basis, aut_generators)
   → list[u64]` — one fat FFI call per parent in the canonical-augmentation
   recursion (Q-pipeline: σ_Q tables + singular BFS + orbit-min + lift).
   Shipped in 5(b).
2. `canon_info_native(rref, n)
   → (canonical_column_order, aut_generators, grpsize1, grpsize2, column_orbits)`
   — Replaces the per-parent `canon.bipartite.bipartite_graph` + `pynauty.autgrp`
   + `pynauty.canon_label` chain with a single FFI call into nauty via
   `nauty-Traces-sys`. The bipartite codeword × column sparsegraph is built
   directly in Rust, eliminating the Python adjacency dict and pynauty's
   internal dict→sparsegraph conversion. Shipped in 5(c).

A `debug` submodule exposes each stage of the Q-pipeline (`q_basis`,
`aut_image_on_q`, `singular_reps_q`, `sigma_q_table`,
`aut_orbit_minima_q_table`, `aut_orbit_minima_q_witt`, `lift`, `project`)
for the cross-check tests in `/workspace/src/tests/test_kernel.py`.

The Python enumerator's two entry points dispatch to the kernel when the
extension is built:
- `enumerate/filters.py::doubly_even_candidates` → `doubly_even_candidates_q`
- `canon/nauty.py::canon_info` → `canon_info_native`

If the wheel isn't built the Python implementations in
`enumerate/quotient.py` and `canon/nauty.py::_canon_info_via_pynauty`
take over (cross-check oracles).

## Layout

```
rust/
├── Cargo.toml         crate manifest (cdylib + rlib) + criterion dev-dep
├── Cargo.lock         committed for reproducible builds
├── pyproject.toml     maturin build-backend config
├── README.md          this file
├── benches/
│   └── candidates.rs  criterion microbenches: table vs witt orbit-min BFS
└── src/
    ├── lib.rs         PyO3 module bindings (top-level + debug submodule)
    ├── types.rs       BinVec = u64; ColPerm = Vec<u32>; Mat = Vec<u64>; MAX_N
    ├── linalg.rs      row_reduce, apply_permutation
    ├── quotient.rs    Q_basis, lift, project, aut_image_on_q, reduce_mod_c
    ├── orbit.rs       singular_reps_q, sigma_q_table, orbit-min BFS (×2)
    ├── candidates.rs  doubly_even_candidates_q — the Q-pipeline orchestrator
    └── canon.rs       canon_info_native — bipartite encoding + sparsenauty
```

## Build dependencies

`nauty-Traces-sys` compiles nauty from vendored C during `cargo build`
(via the `bundled` feature). It uses `bindgen` to generate the Rust FFI,
which requires `libclang` on the host (the system package, e.g.
`libclang-18-dev`, not the LLVM-only runtime). The crate's `tls` feature
must stay enabled or nauty cannot be called concurrently (e.g. cargo's
parallel test runner SIGSEGV's without it).

## Build / install for development

From the project root (`/workspace/src/`):

```sh
CARGO_HOME=$HOME/.cache/claude-cargo \
    uv run maturin develop --release --manifest-path rust/Cargo.toml

# Smoke test.
uv run python -c "
import doubly_even_kernel as k
print('version:', k.debug.kernel_version())
print('zero[4]:', k.doubly_even_candidates_q(4, [], [], [1,2,4,8],
    [[1,0,2,3],[1,2,3,0]]))
"
```

`maturin develop` builds the cdylib and drops it into the active venv as
an importable module. Use `--release` for benched runs; debug builds are
~10× slower.

### CARGO_HOME note

On this host `/home/dev/.cargo/` was created by the `root`-run `rustup`
installer and is not writable by the `dev` user. Use the host-owned
shared cache under `~/.cache/claude-cargo` for every `cargo` and
`maturin` invocation:

```sh
CARGO_HOME=$HOME/.cache/claude-cargo …
```

`~/.cache/claude-cargo/` lives on the host (not the workspace volume),
so the registry survives container rebuilds and is shared across
sessions.

## Rust-side tests

```sh
CARGO_HOME=$HOME/.cache/claude-cargo cargo test \
    --manifest-path rust/Cargo.toml
```

Pure-Rust unit tests under each module's `#[cfg(test)]` block. The
Python-side cross-check (`tests/test_kernel.py`) tests the FFI surface
end-to-end and runs as part of the normal `uv run pytest`.

## Microbenchmarks

```sh
CARGO_HOME=$HOME/.cache/claude-cargo cargo bench \
    --manifest-path rust/Cargo.toml
```

`benches/candidates.rs` measures the table-based vs. witt bit-walk BFS at
`L = 10, 14, 18, 22` with synthetic `GL(L, F_2)` generators. Results land
in `rust/target/criterion/report/index.html`.

5(b) measurement on the dev host (release build):

| L  | table    | witt     | winner            |
|----|----------|----------|-------------------|
| 10 | 4.7 µs   | 6.8 µs   | table (1.45×)     |
| 14 | 175 µs   | 308 µs   | table (1.76×)     |
| 18 | 5.4 ms   | 5.9 ms   | table (1.09×)     |
| 22 | 168 ms   | 116 ms   | witt (1.45×)      |

Crossover happens around `L = 20`. The current dispatch in
`candidates.rs::use_witt_path` uses the table path everywhere; an
`L`-threshold switch will land once a real workload at `L ≥ 22` shows
up on the flamegraph.

## Flame graphs

Two paths, both rooted at this host:

```sh
# Bench-level (Rust only, criterion harness).
CARGO_HOME=$HOME/.cache/claude-cargo cargo flamegraph \
    --manifest-path rust/Cargo.toml --bench candidates

# End-to-end (Python + Rust). Build release first, then perf-record
# the bench script — captures both sides of the FFI boundary.
CARGO_HOME=$HOME/.cache/claude-cargo \
    uv run maturin develop --release --manifest-path rust/Cargo.toml
perf record -g uv run python scripts/bench.py --label flamegraph --N 18
perf report
```

`perf record` needs `kernel.perf_event_paranoid <= 1` (set via
`sysctl` as root, or use `--call-graph dwarf` to avoid the kernel
sample). On a sandboxed host install with `apt install linux-tools-`
matching `uname -r`.

## End-to-end bench (Python orchestrator + Rust kernel)

```sh
uv run python scripts/bench.py --label kernel-5b --N 14,16,18,20,22
```

5(b) → 5(c) wall progression on the dev host:

| N  | pre-kernel | kernel-5b | easy-wins (T1) | native-canon (T3) | weight-cache (T4) | total speedup |
|----|-----------:|----------:|---------------:|------------------:|------------------:|--------------:|
| 14 |   0.63 s   |  0.018 s  |    0.017 s     |     0.010 s       |    0.010 s        |   63×         |
| 16 |   4.75 s   |  0.130 s  |    0.121 s     |     0.063 s       |    0.061 s        |   78×         |
| 18 |   1.83 s   |  0.671 s  |    0.639 s     |     0.296 s       |    0.286 s        |   6.4×        |
| 20 |   13.8 s   |  6.83 s   |    6.63 s      |     2.33 s        |    2.22 s         |   6.2×        |
| 22 |    152 s   |   108 s   |     107 s      |     24.8 s        |    23.3 s         |   6.5×        |

(Pre-kernel column is the post-D7 pure-Python baseline from the project
memory; N=14,16 also include the recursion fixed-cost overhead that the
larger `N` measurements amortise away.)

## What's next (post-5(c))

The N=22 cProfile under the kernel-active path is now dominated by
nauty's own C algorithm:

```
canon_info_native (nauty C core) .............. ~52% of wall
codewords()/weight_enum chain (cacheable, T4) ~6-10%
_compute_rref ................................. ~8%
apply_permutation ............................. ~5%
doubly_even_candidates_q (Q-pipeline FFI) ..... ~4%
```

Beyond Milestone 5, the levers are:

1. **Custom canonicaliser** in Rust (no nauty) — the only single port
   that breaks the ~52% nauty ceiling. ~1-2 KLOC; oracle is
   `pynauty.autgrp`/`canon_label` on every existing test case. Risk
   concentrates on high-symmetry codes (which dominate at large N).
2. **Length induction** (`(N, k) → (N+1, k|k+1)`, Bouyukliev style) —
   structural restructure of the recursion. Different mass-formula
   shape; one enumeration produces results across `N`. Larger surface
   than canon, naturally the Milestone 6.
3. **Sibling/cousin sharing** within `(N, k)` — cache `Q_basis` and
   σ_Q tables across parents with isomorphic `Q_C`; cache
   `_in_aut_orbit_of_subspace` BFS state across cousins. Plausibly
   1.5-3× at large `N`, needs A/B measurement.
4. **Mass-formula early pruning** — current pruning is *late*
   (`augment.py:235-245`); predictive pruning at level `k` would need
   a combinatorial bound on per-parent contribution.

`scripts/bench-results/*.json` is the audit trail; compare
`20260516T125916Z-kernel-5b.json` (5(b) baseline) →
`20260516T141532Z-weight-enum-cache.json` (post-5(c) tip) for the
session's wall-time arc.
