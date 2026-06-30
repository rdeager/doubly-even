#!/usr/bin/env bash
# Build and install the doubly_even_kernel Rust wheel into the venv.
# Bypasses uv's dependency lifecycle so the kernel survives `uv sync` /
# `uv run` (which would otherwise drop it — see CLAUDE.md).
#
# Usage:
#   scripts/install-kernel.sh                   # baseline
#   scripts/install-kernel.sh parallel          # D13 parallel
#   scripts/install-kernel.sh parallel,equivalence_verifier
#   scripts/install-kernel.sh parallel --target-cpu neoverse-v2
#
# --target-cpu <cpu> overrides the codegen CPU via RUSTFLAGS for this
# build only (the cloud-day aarch64 pin A/B — eval §3 row 10). NOTE:
# RUSTFLAGS takes precedence over rust/.cargo/config.toml, so on x86 an
# override REPLACES the x86-64-v3 default rather than adding to it.
#
# Run from /workspace/src/.

set -euo pipefail

cd "$(dirname "$0")/.."

FEATURES="${1:-}"
TARGET_CPU=""
if [[ "${1:-}" == "--target-cpu" ]]; then
    FEATURES=""
    TARGET_CPU="${2:?--target-cpu needs a value}"
elif [[ "${2:-}" == "--target-cpu" ]]; then
    TARGET_CPU="${3:?--target-cpu needs a value}"
fi
# Pin maturin to the project venv's interpreter. Without --interpreter,
# maturin discovers via `python3` on PATH, which fails on a fresh macOS
# (uv's Python isn't exposed as `python3` unless the venv is activated:
# "couldn't find any python interpreters from python3"). The rest of the
# script is already venv-centric (installs into .venv, probes with
# .venv/bin/python), so pin the build interpreter to match on every OS.
VENV_PYTHON="$PWD/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "error: $VENV_PYTHON not found — run 'uv sync --all-extras --dev' first" >&2
    exit 1
fi
MATURIN_ARGS=(build --release -m Cargo.toml --interpreter "$VENV_PYTHON")
if [[ -n "$FEATURES" ]]; then
    MATURIN_ARGS+=(--features "$FEATURES")
fi

# Build from inside rust/: cargo config discovery is CWD-based, so the
# per-target rustflags in rust/.cargo/config.toml (x86-64-v3) are silently
# ignored when maturin runs from the repo root. The avx2 probe below is
# the backstop.
if [[ -n "$TARGET_CPU" ]]; then
    echo "target-cpu override: $TARGET_CPU (replaces .cargo/config.toml rustflags)"
    (cd rust && RUSTFLAGS="-C target-cpu=$TARGET_CPU" ../.venv/bin/maturin "${MATURIN_ARGS[@]}")
else
    (cd rust && ../.venv/bin/maturin "${MATURIN_ARGS[@]}")
fi

WHEEL=$(ls -t rust/target/wheels/doubly_even_kernel-*.whl | head -1)
# uv-managed venvs ship without pip; `uv pip` targets ./.venv from the repo
# root (project convention: uv for everything).
uv pip install --quiet --reinstall --no-deps "$WHEEL"

echo "Installed: $WHEEL"
.venv/bin/python -c '
import doubly_even_kernel as k
print(f"  build_info: {k.kernel_build_info()}")
print(f"  module:     {k.__file__}")
tf = dict(k.kernel_target_features())
print(f"  target:     {tf}")
if tf.get("x86_64") and not tf.get("avx2"):
    raise SystemExit(
        "x86_64 wheel built WITHOUT avx2 — rust/.cargo/config.toml was "
        "ignored (cwd-based discovery). Rebuild via this script."
    )
'
