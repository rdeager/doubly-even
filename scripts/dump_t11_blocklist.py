"""Dump T11 collision hashes for embedding into rust/src/t11_blocklist.rs.

Run with a `t11_cache`-featured kernel wheel installed (the wheel must export
the `compute_t11_hash` pyo3 function). Enumerates at the requested N, runs
the Rust T11 hash on every canonical RREF, and prints any hash that maps to
≥ 2 distinct canonical forms — formatted as Rust source for paste into
`T11_BLOCKLIST_N{N}`.

Usage:
    uv run python scripts/dump_t11_blocklist.py --N 22
    uv run python scripts/dump_t11_blocklist.py --N 22,24

Both forms write the blocklist code to stdout. Use `--out path.json` to
also dump a machine-readable record of the collision pairs (canonical
RREFs grouped by hash) for audit.

Phase 1c follow-up of `last-time-we-profiled-floofy-lollipop.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Bootstrap: T11 cache is off by default in the kernel (see notes in
# `rust/src/enumerate.rs::WorkerState::new`). We rely on that default
# so this dump produces ground-truth canonical RREFs; explicitly
# overriding here in case the kernel default ever flips.
os.environ.pop("DOUBLY_EVEN_ENABLE_T11", None)

from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402

try:
    from doubly_even_kernel import compute_t11_hash  # type: ignore
except ImportError as exc:
    sys.stderr.write(
        "ERROR: kernel does not export compute_t11_hash. Build with\n"
        "  maturin build --release --features t11_cache -m rust/Cargo.toml\n"
        "  uv pip install rust/target/wheels/doubly_even_kernel-*.whl --force-reinstall\n"
    )
    raise


def dump_for_n(n: int) -> tuple[list[int], dict[int, list[tuple[int, ...]]]]:
    """Return (sorted collision hashes, hash → list of canonical RREFs)."""
    by_hash: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    count = 0
    for ec in enumerate_doubly_even(n):
        rref = tuple(int(b) for b in ec.code.basis)
        h = compute_t11_hash(list(rref), n)
        by_hash[h].append(rref)
        count += 1

    collisions = {h: rs for h, rs in by_hash.items() if len(set(rs)) > 1}
    sorted_hashes = sorted(collisions.keys())
    print(
        f"# N={n}: enumerated {count} classes; "
        f"{len(by_hash)} unique T11 hashes; "
        f"{len(collisions)} colliding hashes; "
        f"worst bucket = {max((len(set(rs)) for rs in collisions.values()), default=1)}",
        file=sys.stderr,
        flush=True,
    )
    return sorted_hashes, collisions


def rust_const_decl(n: int, hashes: list[int]) -> str:
    """Format the collision hashes as a Rust `&[u64]` const array body."""
    lines = [f"pub const T11_BLOCKLIST_N{n}: &[u64] = &["]
    # 2 hashes per line (~36 hex chars + commas).
    for i in range(0, len(hashes), 2):
        chunk = hashes[i : i + 2]
        rendered = ", ".join(f"0x{h:016x}" for h in chunk)
        lines.append(f"    {rendered},")
    lines.append("];")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=str, default="22")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path: also write JSON record of collision sets.")
    args = parser.parse_args()
    Ns = [int(n) for n in args.N.split(",")]

    audit: dict[str, object] = {"Ns": Ns, "blocklists": {}}
    for n in Ns:
        hashes, collisions = dump_for_n(n)
        print()
        print(rust_const_decl(n, hashes))
        audit["blocklists"][str(n)] = {
            "hashes": [f"0x{h:016x}" for h in hashes],
            "collisions": {
                f"0x{h:016x}": [list(r) for r in rs]
                for h, rs in collisions.items()
            },
        }

    if args.out:
        Path(args.out).write_text(json.dumps(audit, indent=2))
        print(f"\n# audit JSON written to {args.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
