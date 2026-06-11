#!/usr/bin/env bash
# Build and install the doubly_even_kernel Rust wheel into the venv.
# Bypasses uv's dependency lifecycle so the kernel survives `uv sync` /
# `uv run` (which would otherwise drop it — see CLAUDE.md).
#
# Usage:
#   scripts/install-kernel.sh                   # baseline
#   scripts/install-kernel.sh parallel          # D13 parallel
#   scripts/install-kernel.sh parallel,equivalence_verifier
#
# Run from /workspace/src/.

set -euo pipefail

cd "$(dirname "$0")/.."

FEATURES="${1:-}"
MATURIN_ARGS=(build --release -m Cargo.toml)
if [[ -n "$FEATURES" ]]; then
    MATURIN_ARGS+=(--features "$FEATURES")
fi

# Build from inside rust/: cargo config discovery is CWD-based, so the
# per-target rustflags in rust/.cargo/config.toml (x86-64-v3) are silently
# ignored when maturin runs from the repo root. The avx2 probe below is
# the backstop.
(cd rust && ../.venv/bin/maturin "${MATURIN_ARGS[@]}")

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
