# Reproducing the published numbers

This doc walks through reproducing every wall-time figure in
[`performance.md`](performance.md) from a clean checkout. Each step
ends with the expected output so you can confirm you're on track.

## Requirements

- **Python 3.12+** (3.13 also tested). Earlier 3.x versions are not
  supported.
- **Rust toolchain** via [rustup](https://rustup.rs/) (stable channel
  is fine). `cargo` and `rustc` on `PATH`.
- **[`uv`](https://github.com/astral-sh/uv)** for Python project
  management. The repo does not support `pip` outside of `uv`.
- **`libclang`** for `bindgen` (used by `nauty-Traces-sys`). On
  Debian/Ubuntu: `apt install libclang-dev`. If you can't install
  system packages, the Python wheel works as a fallback —
  `uv pip install libclang` and set
  `LIBCLANG_PATH=$(python -c 'import clang, os; print(os.path.join(os.path.dirname(clang.__file__), "native"))')`.
- **RAM:** ~4 GB for `N ≤ 22`, ~16 GB for `N = 24`, ~60 GB for
  `N = 26`, ~80 GB for `N = 28` with streaming output (much more
  without). 200+ GB recommended for `N = 29` and beyond.

## Build

```sh
# 1. Python env + dependencies
uv sync --all-extras --dev

# 2. Build the Rust kernel (parallel feature) and install the wheel
scripts/install-kernel.sh parallel
```

The parallel feature is opt-in; without it, the kernel still works
but only sequentially. The build takes ~5–8 minutes on a modern
desktop (most of it is sparsenauty's C compile via
`nauty-Traces-sys`).

If you only want to run the spec-side Python code (no kernel), skip
step 2 — the package will fall back to `pynauty` and pure-Python
recursion, ~50× slower but correct.

### Build requirements (x86)

x86 builds use `-C target-cpu=x86-64-v3` via `rust/.cargo/config.toml`,
so the wheel **requires AVX2** — any x86 CPU since ~2013 qualifies,
including both GCP x86 families. Cargo's config discovery is
working-directory-based: the flag only applies to builds run from
inside `rust/`, which `scripts/install-kernel.sh` handles for you.
Verify an installed wheel with

```sh
uv run python -c 'import doubly_even_kernel as k; print(dict(k.kernel_target_features()))'
```

— `avx2` must be `True` on x86. aarch64 builds are unaffected (NEON
is baseline; no config needed).

## Reproduce the validation table

```sh
# Fast tests (590 collected: 549 pass + 41 slow-skipped, ~7 s)
uv run pytest

# All tests including N=17, N=18 Table-3 cells (580 pass / 10 skipped)
uv run pytest --run-slow
```

Expected: all green. The `--run-slow` suite includes the cell-by-cell
DFGHILM Table 3 check at `N = 18` plus the mass-formula consistency
oracle at every `(N, k)` pair through `N = 22`.

## Reproduce the headline wall times

Each invocation writes a JSON record to `scripts/bench-results/`
(gitignored). Class counts in the output should match the headline
table in [`performance.md`](performance.md).

```sh
# Sequential at N=22 (~0.7 s on a 13700K)
uv run python scripts/bench.py --label seq-n22 --N 22

# Parallel at N=22 (~0.24 s on a 13700K, 16 physical / 24 logical)
DOUBLY_EVEN_THREADS=24 \
    uv run python scripts/bench.py --label par-t24-n22 --N 22

# Parallel at N=24 (~1.7 s)
DOUBLY_EVEN_THREADS=24 \
    uv run python scripts/bench.py --label par-t24-n24 --N 24

# Parallel at N=26 (~10 s with the canon-cache cap)
DOUBLY_EVEN_THREADS=24 DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
    uv run python scripts/bench.py --label par-t24-n26 --N 26
```

The default frontier depth (`DOUBLY_EVEN_FRONTIER_DEPTH=4`) is the
measured best at every benched `N` — the older "raise to 5 at
`N ≥ 24`" advice predates the coset-spectrum parent rule and is
obsolete. Each run prints the wall time, the per-`k` class count, and
(after the run) cross-checks DFGHILM Table 3. The expected class
counts:

| N  | classes  |
|----|---------:|
| 18 |      341 |
| 20 |    1,211 |
| 22 |    5,118 |
| 24 |   37,496 |
| 26 |  494,272 |

## Reproduce the N = 28 cloud run

The N = 28 enumeration was done on a GCP `c4a-standard-72`
(Axion / Arm Neoverse V2, us-east4-a). The recipe is one-liner
bootstrap + one-liner benchmark; both scripts auto-adapt to `nproc`
and machine type.

### Step 1 — provision the VM

Any cloud will work; the scripts make no GCP-specific assumptions
once the VM is up. The example below is for GCP. Substitute your
project, zone, and chosen image:

```sh
# On your local box (one-time setup):
gcloud compute instances create doubly-even-n28 \
    --machine-type=c4a-standard-72 \
    --zone=<your-zone> \
    --image-family=ubuntu-2404-lts-arm64 \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced

# ssh in
gcloud compute ssh doubly-even-n28 --zone=<your-zone>
```

For x86 platforms (`c4-standard-24`, `c4-standard-288-metal`, etc.)
use `ubuntu-2404-lts-amd64` and a corresponding `--machine-type`.

### Step 2 — bootstrap

```sh
# Inside the VM:
curl -fsSL https://raw.githubusercontent.com/rdeager/doubly-even/main/scripts/gcp-setup.sh \
    | bash -s -- --repo https://github.com/rdeager/doubly-even
```

This installs apt deps, rustup, uv, clones the repo, builds the
parallel kernel (`maturin build --release --features parallel`), and
runs a smoke pytest. Takes ~7–10 min on c4a-72; longer on smaller
hosts because Rust compile parallelism scales with cores.

The script prints a `READY` banner with rustc + python versions, the
git SHA, CPU model, and cores once the build succeeds.

### Step 3 — run the shakedown benchmark (any size machine)

`scripts/gcp-bench.sh` runs three configurations (sequential, half-
subscribed parallel, full-subscribed parallel) on `N = 16, 24, 26`.
On c4a-72 it takes ~5 min total and validates that the parallel path
works end-to-end before you commit to a longer run.

```sh
cd ~/doubly-even
scripts/gcp-bench.sh shakedown-c4a-72
```

### Step 4 — N = 28 with streaming output

```sh
cd ~/doubly-even
DOUBLY_EVEN_THREADS=72 DOUBLY_EVEN_FRONTIER_DEPTH=4 \
    DOUBLY_EVEN_CANON_CACHE_CAP=300000 \
    uv run python scripts/run_streaming.py \
    --N 28 \
    --output-dir /home/$USER/n28-out \
    --label n28-c4a72-shakedown
```

(The original 2026-05-21 record used `FRONTIER_DEPTH=5`; depth 4 has
been the better setting at every benched `N` since the parent-rule
levers landed — see `bottlenecks.md` §3. Walls quoted below are from
the depth-5 era and should improve.)

Per-worker binary files are written to `~/n28-out/out.w<wid>.bin`;
peak RSS is ~71 GB. Wall time on c4a-72: ~61 min. On a smaller host
expect ~110 × `(72 / physical_cores)` minutes plus a clock-rate
factor.

### Step 5 — watch progress (sidecar)

In a separate `ssh` session against the same VM:

```sh
cd ~/doubly-even
uv run python scripts/stream_progress.py \
    --N 28 \
    --output-dir /home/$USER/n28-out \
    --interval 60
```

The sidecar prints the per-`k` progress table (classes emitted,
mass-vs-σ ratio, ETA estimate) every 60 s. It exits when
`stats.json` appears in the output directory, signalling kernel
completion.

The same sidecar works against any local directory — see
[`cluster-deployment.md` §"Streaming output is not just for cloud"](cluster-deployment.md#streaming-output-is-not-just-for-cloud)
for a local-only example.

### Step 6 — verify

```sh
cd ~/doubly-even
cat /home/$USER/n28-out/stats.json | jq '.mass[]' | head -20
# Should match σ(28, k) for k = 0..13:
# 1, ..., (15 values, all equal to their σ entry)
```

`merge_stream.py` is invoked automatically by `run_streaming.py` and
fails loudly if any per-`k` mass doesn't match `σ(28, k)`.

### Step 7 — pull results back, tear down

```sh
# On your local box:
gcloud compute scp --recurse \
    doubly-even-n28:~/n28-out \
    ./n28-results --zone=<your-zone>

gcloud compute instances delete doubly-even-n28 --zone=<your-zone>
```

## Sidecar for long local runs

The streaming pipeline isn't cloud-specific. For local `N = 26` /
`N = 28` runs that would otherwise tie up your shell:

```sh
# Terminal 1 — long-running kernel:
DOUBLY_EVEN_THREADS=24 DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
    uv run python scripts/run_streaming.py --N 26 \
    --output-dir /tmp/n26-local

# Terminal 2 — sidecar, polls every 30 s:
uv run python scripts/stream_progress.py --N 26 \
    --output-dir /tmp/n26-local --interval 30
```

Both terminals exit cleanly when the kernel finishes; the cross-check
runs automatically against DFGHILM Table 3.

## ARM (aarch64) build note

`nauty-Traces-sys 0.11`'s `popcnt` feature is x86-only. On aarch64
hosts (Graviton, Axion, Raspberry Pi 5, M-series under Linux) the
build would fail with a `is_x86_feature_detected!` macro error. The
target-conditional dependency is already in the manifest
(`rust/core/Cargo.toml`), so no patch is needed:

```toml
[target.'cfg(any(target_arch = "x86", target_arch = "x86_64"))'.dependencies]
nauty-Traces-sys = { version = "0.11", default-features = false,
    features = ["bundled", "tls", "popcnt"] }

[target.'cfg(not(any(target_arch = "x86", target_arch = "x86_64")))'.dependencies]
nauty-Traces-sys = { version = "0.11", default-features = false,
    features = ["bundled", "tls"] }
```

Without `popcnt`, nauty's C code uses `__builtin_popcountl`, which on
Neoverse V2 compiles to the single-cycle `CNT` instruction —
zero perf cost on ARM.

## Troubleshooting

**Build fails on `nauty-Traces-sys` with libclang errors.** The
fallback Python wheel: `uv pip install libclang` then
`LIBCLANG_PATH=$(python -c 'import clang, os; print(os.path.join(os.path.dirname(clang.__file__), "native"))') maturin build --release …`.

**Wall time at `N = 22` parallel is much worse than 0.24 s.** Check:

- The kernel was built `--release` (not debug). `maturin develop`
  without `--release` is ~20× slower.
  `scripts/install-kernel.sh parallel` builds release with the
  parallel feature and force-reinstalls the wheel (else a stale wheel
  from a previous build wins).
- On x86, the x86-64-v3 flag took effect:
  `kernel_target_features()` must report `avx2` as `True` (see
  "Build requirements (x86)" above).
- mimalloc is the allocator. Check the build info via
  `python -c 'import doubly_even_kernel; print(doubly_even_kernel.kernel_build_info())'`.
- `DOUBLY_EVEN_THREADS` is set ≥ 2; the sequential path is taken
  when it's unset.

**OOM at `N = 26` or `N = 28`.** Lower `DOUBLY_EVEN_CANON_CACHE_CAP`.
The per-worker cache × thread count is the dominant memory user; at
`N = 28` on a memory-tight host (e.g. cgroup-limited dev box), drop
to `CAP = 200000`.

**Class counts don't match DFGHILM Table 3.** This is a kernel bug —
file an issue with the bench JSON output, your platform info, and
the kernel build info string.
