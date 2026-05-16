# doubly-even-kernel

Native (Rust + PyO3) hot kernel for the `doubly_even` Python enumerator.

## Status

Scaffold only. Exposes two smoke-test functions:

- `kernel_version()` — version from `Cargo.toml`.
- `popcount_batch(words: list[int]) -> list[int]` — proves bulk `Vec<u64>`
  round-trips across the FFI.

Real entry points (σ_Q table build, singular-set BFS, orbit-min
decomposition, then `doubly_even_candidates_q`) land in follow-ups. See
`/workspace/markdown/architecture/04-optimisations.md` for the kernel
target list and `/home/dev/.claude/plans/last-session-we-tried-synchronous-spring.md`
for the audit + decision context.

## Layout

```
rust/
├── Cargo.toml         crate manifest (cdylib + rlib)
├── Cargo.lock         committed for reproducible builds
├── pyproject.toml     maturin build-backend config
├── README.md          this file
└── src/
    └── lib.rs         PyO3 module
```

The crate produces a top-level Python module named `doubly_even_kernel`.
The pure-Python package at `/workspace/src/src/doubly_even/` imports it
with a try/except fallback so the Python-only path keeps working when the
kernel is not built.

## Build / install for development

From the project root (`/workspace/src/`):

```sh
# Build + install into the project venv in one command.
uv run maturin develop --release --manifest-path rust/Cargo.toml

# Quick smoke test.
uv run python -c \
    "import doubly_even_kernel as k; \
     print('kernel', k.kernel_version()); \
     print('popcount', k.popcount_batch([0, 1, 0xff, (1<<64)-1]))"
```

`maturin develop` builds the cdylib and drops it into the active venv as
an importable module. Use `--release` for benched runs; debug builds are
~10× slower.

### CARGO_HOME note

On this host `/home/dev/.cargo/` was created by the `root`-run `rustup`
installer and is not writable by the `dev` user. Use the host-owned
shared cache under `~/.cache/claude-cargo` instead:

```sh
CARGO_HOME=$HOME/.cache/claude-cargo uv run maturin develop --release --manifest-path rust/Cargo.toml
```

`~/.cache/claude-cargo/` lives on the host (not the workspace volume),
so the registry survives container rebuilds and is shared across
sessions. It is tool cache — no need to back it up or check it in.

## Rust-side tests

```sh
cargo test
```

Runs pure-Rust unit tests (no Python). Add tests under `#[cfg(test)]`
blocks; for now there's one (`popcount_matches_count_ones`).

## Toolchain pinning

No `rust-toolchain.toml` yet — using whatever `rustup` provides. Pin
once the kernel grows enough that stable-vs-nightly matters.
