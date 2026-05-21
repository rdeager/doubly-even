#!/usr/bin/env bash
# Bootstrap a fresh GCP Ubuntu VM for the doubly-even benchmark campaign.
#
# Designed to be either:
#   (a) run inside the VM after `git clone`-ing this repo, OR
#   (b) `curl | bash`-ed from a raw GitHub URL (see --repo/--branch flags).
#
# What it does, in order:
#   1. apt deps (build-essential, python3.12, git, …)
#   2. rustup (stable toolchain)
#   3. uv (Astral installer)
#   4. clone the repo into ~/doubly-even (if not already there)
#   5. python venv + `uv sync`
#   6. libclang probe; fall back to `uv pip install libclang` if missing
#   7. scripts/install-kernel.sh parallel    — D13 outer-DFS worker pool
#   8. smoke pytest -k "n_12 or n_14"        — confirms the kernel loads
#   9. print a READY banner with git SHA + uname
#
# Usage (already cloned, cwd = repo root):
#   scripts/gcp-setup.sh
#
# Usage (curl from a raw URL on a fresh VM):
#   curl -fsSL https://raw.githubusercontent.com/<gh-user>/doubly-even/main/scripts/gcp-setup.sh \
#     | bash -s -- --repo https://github.com/<gh-user>/doubly-even --branch main
#
# Honoured env vars:
#   REPO_DIR    target clone directory (default: $HOME/doubly-even)
#   PY_VERSION  python version to install (default: 3.12)

set -euo pipefail

REPO_URL=""
BRANCH="main"
REPO_DIR="${REPO_DIR:-$HOME/doubly-even}"
PY_VERSION="${PY_VERSION:-3.12}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)   REPO_URL="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --dir)    REPO_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

log() { printf '\n=== %s ===\n' "$*"; }

log "Step 1/9 — apt deps"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential pkg-config git ca-certificates curl \
    "python${PY_VERSION}" "python${PY_VERSION}-venv" "python${PY_VERSION}-dev" \
    libclang-dev clang

log "Step 2/9 — rustup (stable)"
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
fi
# shellcheck source=/dev/null
. "$HOME/.cargo/env"

log "Step 3/9 — uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

log "Step 4/9 — clone repo into $REPO_DIR"
if [[ ! -d "$REPO_DIR/.git" ]]; then
    if [[ -z "$REPO_URL" ]]; then
        echo "REPO_DIR=$REPO_DIR has no .git and --repo not given." >&2
        echo "Either pass --repo <url> or cd into an existing checkout." >&2
        exit 2
    fi
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

log "Step 5/9 — python venv + uv sync"
if [[ ! -d .venv ]]; then
    "python${PY_VERSION}" -m venv .venv
fi
uv sync --all-extras --dev

log "Step 6/9 — libclang probe"
# Stock ubuntu-2404-lts-amd64 has libclang-dev (installed above) → bindgen
# finds /usr/lib/x86_64-linux-gnu/libclang.so.1 on its own. Fallback for
# images without libclang-dev: install the python wheel, point LIBCLANG_PATH
# at its native/ directory (the recipe in /home/dev/.claude/CLAUDE.md).
if ! ldconfig -p | grep -q libclang; then
    .venv/bin/pip install --quiet libclang
    LIBCLANG_DIR=$(.venv/bin/python -c \
        'import clang, os; print(os.path.join(os.path.dirname(clang.__file__), "native"))')
    export LIBCLANG_PATH="$LIBCLANG_DIR"
    echo "  using fallback LIBCLANG_PATH=$LIBCLANG_PATH"
fi

log "Step 7/9 — build & install kernel (parallel feature)"
scripts/install-kernel.sh parallel

log "Step 8/9 — smoke pytest (n12 or n14)"
.venv/bin/python -m pytest -x -q -k "n12 or n14"

log "Step 9/9 — READY"
GIT_SHA=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
LOGICAL_CORES=$(nproc)
PHYSICAL_CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/ {c=$2} /^Socket\(s\)/ {s=$2} END {print c*s}' | tr -d ' ')
MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
CPU_MODEL=$(awk -F: '/^model name/ {print $2; exit}' /proc/cpuinfo | sed 's/^ *//')

cat <<EOF

READY
  repo:           $REPO_DIR
  git:            $GIT_BRANCH @ $GIT_SHA
  python:         $("python${PY_VERSION}" --version)
  rustc:          $(rustc --version)
  cpu:            $CPU_MODEL
  cores:          $LOGICAL_CORES logical / $PHYSICAL_CORES physical
  memory:         ${MEM_GB} GiB
  kernel module:  $(.venv/bin/python -c 'import doubly_even_kernel; print(doubly_even_kernel.__file__)')
  build info:     $(.venv/bin/python -c 'import doubly_even_kernel; print(doubly_even_kernel.kernel_build_info())')

Next:  scripts/gcp-bench.sh <label>
EOF
