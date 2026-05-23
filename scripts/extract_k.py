"""Extract canonical classes at one or more ranks from a streaming-kernel
output directory, including from a **still-running** enumeration.

Reads every ``out.w*.bin`` file under ``--output-dir`` once via mmap,
filters records by their ``k:u8`` prefix, and writes one JSON-lines
file per requested rank.

Safe to run mid-run: per-worker files are append-only with a 256 KB
``BufWriter`` (see ``rust/src/streaming.rs``), and POSIX allows
concurrent O_RDONLY readers against a file held open for write. The
existing ``scripts/stream_progress.py`` sidecar uses the same pattern.
A partially-flushed trailing record is silently skipped.

Usage::

    uv run python scripts/extract_k.py \\
        --N 29 --output-dir ~/sweep/n29 --k 12,13 \\
        --out-prefix ~/sweep/n29/partial_

    # writes ~/sweep/n29/partial_k12.jsonl
    #    and ~/sweep/n29/partial_k13.jsonl

Each output line is::

    {"aut": <int>, "basis": [<int>, ...]}

``aut`` is ``|Aut(C)|`` as a Python int (the on-disk ``u128``).
``basis`` is the ``k`` basis rows as integers (each a ``u64`` bitmask
where bit ``i`` is column ``i``).

The mass formula (``Σ N!/|Aut| == gaborit_sigma(N, k)``) is **not**
checked here — it only holds once a rank is fully enumerated. For a
completed run, use ``scripts/merge_stream.py`` instead.
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

MAGIC = b"DEKS"
VERSION = 1
HEADER_LEN = 16

_HDR_TAIL = struct.Struct("<III")     # version, N, worker_id
_AUT_STRUCT = struct.Struct("<QQ")    # u128 -> (lo:u64, hi:u64), LE


def _extract_file(
    path: Path,
    expect_n: int,
    wanted_k: set[int],
    handles: dict[int, "object"],
    counts: dict[int, int],
) -> None:
    """Walk one ``out.w*.bin`` file, appending matching records to
    ``handles[k]`` and bumping ``counts[k]``. Stops cleanly at a
    partially-flushed tail record."""
    with path.open("rb") as fh:
        size = fh.seek(0, 2)
        if size < HEADER_LEN:
            return
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if mm[0:4] != MAGIC:
                raise SystemExit(
                    f"{path}: bad magic {bytes(mm[0:4])!r} (expected {MAGIC!r})"
                )
            version, n_hdr, _worker_id = _HDR_TAIL.unpack(mm[4:HEADER_LEN])
            if version != VERSION:
                raise SystemExit(
                    f"{path}: format version {version} unsupported (expected {VERSION})"
                )
            if n_hdr != expect_n:
                raise SystemExit(
                    f"{path}: header N={n_hdr} but --N={expect_n}"
                )
            i = HEADER_LEN
            end = len(mm)
            while i < end:
                k = mm[i]
                need = 1 + 16 + 8 * k
                if i + need > end:
                    break  # writer hasn't flushed the trailing record yet
                if k in wanted_k:
                    lo, hi = _AUT_STRUCT.unpack_from(mm, i + 1)
                    aut = (hi << 64) | lo
                    if k > 0:
                        basis = list(struct.unpack_from(f"<{k}Q", mm, i + 17))
                    else:
                        basis = []
                    handles[k].write(json.dumps({"aut": aut, "basis": basis}) + "\n")
                    counts[k] += 1
                i += need


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory holding out.w*.bin files (e.g. ~/sweep/n29).")
    p.add_argument("--k", type=str, required=True,
                   help="Comma-separated ranks to extract, e.g. '12,13'.")
    p.add_argument("--out-prefix", type=Path, required=True,
                   help="Output path prefix; produces <prefix>k{K}.jsonl per K.")
    args = p.parse_args()

    try:
        wanted_k = sorted({int(x) for x in args.k.split(",") if x.strip()})
    except ValueError:
        print(f"ERROR: --k must be comma-separated ints, got {args.k!r}", file=sys.stderr)
        return 2
    if not wanted_k:
        print("ERROR: --k produced an empty set", file=sys.stderr)
        return 2
    if not args.output_dir.is_dir():
        print(f"ERROR: {args.output_dir} is not a directory", file=sys.stderr)
        return 2

    bin_files = sorted(args.output_dir.glob("out.w*.bin"))
    if not bin_files:
        print(f"ERROR: no out.w*.bin files in {args.output_dir}", file=sys.stderr)
        return 2

    out_parent = args.out_prefix.parent
    if str(out_parent) != "":
        out_parent.mkdir(parents=True, exist_ok=True)

    out_paths = {k: Path(f"{args.out_prefix}k{k}.jsonl") for k in wanted_k}
    handles = {k: out_paths[k].open("w") for k in wanted_k}
    counts = {k: 0 for k in wanted_k}
    wanted_set = set(wanted_k)
    try:
        for path in bin_files:
            _extract_file(path, args.N, wanted_set, handles, counts)
    finally:
        for fh in handles.values():
            fh.close()

    print(f"extract_k.py: N={args.N}, scanned {len(bin_files)} worker file(s) in {args.output_dir}")
    for k in wanted_k:
        print(f"  k={k:2d}: {counts[k]:>10d} classes -> {out_paths[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
