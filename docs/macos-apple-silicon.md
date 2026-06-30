# Building & running on Apple Silicon (MacBook M1 / M5 Pro)

A step-by-step walkthrough for cloning, building, testing, and running the
`doubly-even` enumerator natively on an Apple Silicon Mac. It assumes you
set up `uv` and `rust` yourself (no bootstrap script) and works for **M1
through M5 Pro** — they are all the same Rust target,
`aarch64-apple-darwin`.

> Status: this is the first-pass walkthrough. The kernel already targets
> aarch64 cleanly — the mimalloc and nauty dependencies are OS/arch-gated
> in-tree, so there are **no source edits to make** (see step 3). Treat the
> perf numbers as *to be measured on your machine* until a row lands in
> [`performance.md`](performance.md).

## At a glance

```sh
# prerequisites (once): Xcode CLT, rustup, uv  — see step 1
git clone <repo-url> doubly-even && cd doubly-even
# (no source edits needed — deps are already OS/arch-gated, see step 3)
export MACOSX_DEPLOYMENT_TARGET=11.0
uv python install 3.12
uv sync --all-extras --dev
scripts/install-kernel.sh parallel          # builds + installs the arm64 wheel
uv run pytest                                # expect 552 passed + 41 skipped
# run + watch mass% climb — see step 6
```

## 0. Why M1 and M5 Pro are one target

Every Apple Silicon chip (M1, M1 Pro/Max, … M5, M5 Pro) compiles to the
single Rust triple `aarch64-apple-darwin`. `rustc` already defaults that
triple's `target-cpu` to `apple-m1` — the portable Apple-Silicon baseline
that runs on every M-series chip — so there is **nothing per-chip to
configure**. NEON is the baseline SIMD on aarch64 and is already
auto-vectorised; the x86-only `-C target-cpu=x86-64-v3` flag in
`rust/.cargo/config.toml` does not apply here. (If you only ever run on the
same Mac you build on, you can opt into native codegen — see step 4.)

## 1. Prerequisites (native, by hand)

1. **Xcode Command Line Tools** — provides `clang`, the macOS SDK, and
   **`libclang`** (which `bindgen` needs to build the bundled `nauty`). No
   `libclang` Python-wheel fallback is required on macOS (unlike Linux).
   ```sh
   xcode-select --install      # skip if `xcode-select -p` already prints a path
   ```

2. **rustup** (stable toolchain). On Apple Silicon the host default toolchain
   is already `aarch64-apple-darwin`.
   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
   . "$HOME/.cargo/env"
   ```

3. **uv** (Astral), then a native arm64 Python 3.12.
   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.12      # arm64 python-build-standalone
   ```

4. **Arch sanity check** — make sure you are *not* in a Rosetta/x86 shell or
   pointed at an x86 Homebrew Python (that would build an x86_64 extension
   that mismatches the kernel):
   ```sh
   uname -m                                          # must print: arm64
   uv run python -c 'import platform; print(platform.machine())'   # must print: arm64
   ```

## 2. Clone

```sh
git clone <repo-url> doubly-even
cd doubly-even
uv sync --all-extras --dev      # creates .venv with Python 3.12
```

## 3. mimalloc TLS flag — already scoped to Linux (no action needed)

`rust/Cargo.toml` pins mimalloc with `local_dynamic_tls`, a **Linux/glibc
static-TLS workaround** (it keeps the dlopen'd `cdylib` off glibc's ~1.6 KB
static-TLS surplus). Mach-O has no such surplus, so the flag is irrelevant
and untested on macOS. The dependency is **already split by `target_os`** in
the tree (mirroring the `nauty-Traces-sys` split in `rust/core/Cargo.toml`),
so there is nothing to edit — an Apple Silicon build takes the non-Linux
branch automatically:

```toml
[target.'cfg(target_os = "linux")'.dependencies]
mimalloc = { version = "0.1", default-features = false, features = ["local_dynamic_tls"] }

[target.'cfg(not(target_os = "linux"))'.dependencies]
mimalloc = { version = "0.1", default-features = false }
```

The gate is on `target_os`, **not** architecture, so every Linux target keeps
the flag byte-identical — x86_64 and the aarch64 Axion (`c4a`) fleet alike;
only non-Linux (Mach-O) drops it. Confirm the resolution on your Mac (no
compiler needed):

```sh
cd rust && cargo tree --target aarch64-apple-darwin -e features -i mimalloc
# mimalloc appears WITHOUT the local_dynamic_tls feature
```

## 4. Build the kernel

```sh
export MACOSX_DEPLOYMENT_TARGET=11.0    # oldest Apple-Silicon macOS; honored by
                                        # rustc AND the cc-built nauty. Avoids the
                                        # "object file built for newer macOS" warning.
scripts/install-kernel.sh parallel
```

`install-kernel.sh` `cd`s into `rust/`, builds a `…-macosx_*_arm64.whl` via
maturin, installs it into `.venv`, and probes the module. Its x86 AVX2 guard
is skipped on aarch64.

Expected probe output — the import **succeeding** is itself the proof the
mimalloc change is correct (no "cannot allocate memory in static TLS block"):

```
target: {'x86_64': False, 'aarch64': True, 'avx2': False, 'bmi2': False, 'popcnt': False}
```

Optional, if you only run on the machine you build on, pin native codegen
(slightly faster, but the wheel then won't run on an older Apple chip):

```sh
scripts/install-kernel.sh parallel --target-cpu native
```

## 5. Test

```sh
uv run pytest                 # expect: 552 passed + 41 skipped (slow)
uv run pytest --run-slow      # optional: 580 passed (~longer)
```

The mass-formula and DFGHILM Table 3 oracles are platform-independent, so a
green suite here certifies the Apple Silicon kernel is bit-correct.

## 6. Run, and watch progress as mass found

Set `DOUBLY_EVEN_THREADS` to the **performance**-core count — Apple Silicon
is heterogeneous, and including the efficiency cores worsens the tail
load-imbalance:

```sh
PCORES=$(sysctl -n hw.perflevel0.logicalcpu)   # e.g. 4 on M1, 8+ on M-series Pro
```

**Small/medium N (in-memory timing):**

```sh
DOUBLY_EVEN_THREADS=$PCORES uv run python scripts/bench.py --label m1-par --N 22,24
```

**N ≥ 26 — counts mode + the live mass-progress display.** This is the
existing `dec progress` tool; it shows, per rank, accumulated mass
`Σ N!/|Aut|` against the Gaborit target `σ(N,k)`, the total fraction, and an
ETA projected from the mass fraction.

```sh
# terminal 1 — the run (writes progress.json every 5 s):
DOUBLY_EVEN_THREADS=$PCORES uv run python scripts/run_counts.py \
    --N 26 --output-dir /tmp/n26 --progress-interval 5

# terminal 2 — watch mass% climb to 100%:
uv run dec progress --output-dir /tmp/n26
```

Notes:
- N ≥ 26 in-memory needs roughly 16–32 GiB; counts mode (above) is the
  memory-frugal, N ≥ 30-capable path.
- The current shipped scheduling defaults are best — `frontier_depth=3`,
  `δ=3`, with demand-driven self-subdivision **on**. Do **not** set the
  stale `DOUBLY_EVEN_FRONTIER_DEPTH=4`.
- On macOS the progress display's memory line needs the small polish in
  step 7 to show real GiB instead of `mem: —`.

## 7. Small macOS polish (optional, None-safe)

Each is tiny and degrades gracefully, so the build/run works without them —
they just make the *existing* tooling display correctly on macOS:

- **`src/doubly_even/enumerate/progress.py`** — `_read_meminfo_bytes()`
  reads `/proc/meminfo`, which doesn't exist on macOS, so `dec progress`
  shows `mem: —`. Add a `platform.system() == "Darwin"` branch: total via
  `os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")`, used via a
  best-effort `vm_stat` parse, returning `None` on any failure (both
  `render()` and `render_counts()` already handle `None`).
- **`scripts/run_counts.py` / `scripts/run_streaming.py`** — give the
  `cpu_model()` / `mem_gib()` helpers a non-Linux fallback so result-JSON
  metadata is correct on Mac: `mem_gib` via `os.sysconf` (as above);
  `cpu_model` via `sysctl -n machdep.cpu.brand_string` (returns e.g.
  "Apple M5 Pro"), falling back to `platform.processor()`.

## 8. Troubleshooting

- **`import doubly_even_kernel` fails with a static-TLS error** — your build
  picked up the Linux `local_dynamic_tls` mimalloc feature on macOS. The §3
  `target_os` split (which ships in-tree) prevents this; confirm it's present
  in `rust/Cargo.toml`, then rebuild `scripts/install-kernel.sh parallel`.
- **`bindgen` / "libclang not found"** — Xcode CLT isn't installed or
  `xcrun` can't find it. `xcode-select --install`, then verify
  `xcrun --find clang`.
- **maturin: "couldn't find any python interpreters from python3"** — the
  stale macOS system `/usr/bin/python3` (3.9) is on `PATH` and doesn't
  satisfy `requires-python`, so maturin reports none. `install-kernel.sh`
  pins maturin to `.venv/bin/python`, which avoids this; on an older
  checkout, `source .venv/bin/activate` before rebuilding. Confirm the venv
  is the arm64 3.12 (not the system 3.9): `.venv/bin/python --version`.
- **The extension is x86_64 / Rosetta surprises** — recheck step 1.4; you're
  in an x86 shell or using an x86 Python. Use a native arm64 shell and the
  uv-provisioned arm64 Python.
- **"object file was built for newer macOS version than being linked"** —
  set `export MACOSX_DEPLOYMENT_TARGET=11.0` before building (step 4).
- **Gatekeeper blocks a *downloaded* wheel** — locally built + `uv pip
  install`ed wheels are auto ad-hoc-signed and not quarantined, so this only
  affects wheels you downloaded from a browser. Clear the quarantine xattr:
  `xattr -dr com.apple.quarantine <wheel-or-.so>`.

## 9. After a successful run

Record the measured N=24 / N=26 walls and per-call µs, then update the docs:
`README.md` (Apple-Silicon note → measured), `docs/references.md` (drop
"predicted"), and add `m1` / `m5pro` rows to `docs/performance.md` /
`docs/benchmarking.md`. See also the general
[`reproducing.md`](reproducing.md) recipe.
