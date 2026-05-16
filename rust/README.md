# doubly-even-kernel

Native (Rust + PyO3) hot kernel for the `doubly_even` Python enumerator.

## Status

Milestone 5(b) lands. Production hot path is
`doubly_even_candidates_q(n, code_rref, pivots, dual_basis, aut_generators)
→ list[u64]` — one fat FFI call per parent in the canonical-augmentation
recursion. A `debug` submodule exposes each stage of the inner pipeline
(`q_basis`, `aut_image_on_q`, `singular_reps_q`, `sigma_q_table`,
`aut_orbit_minima_q_table`, `aut_orbit_minima_q_witt`, `lift`, `project`)
for the cross-check tests in `/workspace/src/tests/test_kernel.py`.

The Python enumerator at `src/doubly_even/enumerate/filters.py` tries to
import this kernel at module load; if present it marshals `Code` and
`aut_generators` across the FFI. If the wheel isn't built the Python
implementation in `enumerate/quotient.py` takes over.

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
    └── candidates.rs  doubly_even_candidates_q — the orchestrator
```

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

5(b) results on the dev host:

| N  | pre-kernel | kernel-5b | speedup |
|----|-----------:|----------:|--------:|
| 14 |   0.63 s   |  0.018 s  |  35×    |
| 16 |   4.75 s   |  0.130 s  |  37×    |
| 18 |   1.83 s   |  0.671 s  |  2.7×   |
| 20 |   13.8 s   |  6.83 s   |  2.0×   |
| 22 |    152 s   |   108 s   |  1.4×   |

(Pre-kernel column is the post-D7 pure-Python baseline from the project
memory; N=14,16 also include the recursion fixed-cost overhead that the
larger `N` measurements amortise away.)

## What's next (5(c) and beyond)

The N=18 cProfile under the kernel-active path shifts the dominant
share to **pynauty + the canonical-augmentation parent test**:

```
is_canonical_augmentation chain ............... 94% of wall
  cached_canon_info  (pynauty.autgrp + canon_label) .. 66%
  bipartite_graph encoding ........................... 20%
  _in_aut_orbit_of_subspace BFS ...................... 19%
doubly_even_candidates (kernel call) ........... 4.6%
```

This makes the 5(c) targets data-driven:

1. **Replace pynauty for our bipartite encoding** (`canon_info_fast` in
   the kernel) — directly attacks the 66% slice. The biggest single win
   available, but a big surface; estimate ~1 KLOC in Rust plus a custom
   bipartite-graph canonicaliser.
2. **Port `_in_aut_orbit_of_subspace`** — straightforward Rust port of
   the column-permutation orbit BFS, eats the 19% slice.
3. **Reduce per-parent marshalling** (the FFI overhead on the Q-pipeline
   side is now visible) — pass `Code` as a frozen handle that lives on
   the Rust side across multiple stages, or batch siblings.

`scripts/bench-results/*.json` is the audit trail; compare to
`20260516T125916Z-kernel-5b.json` for the 5(b) baseline.
