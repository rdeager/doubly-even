"""Reader for the binary streaming format written by
`doubly_even_kernel.enumerate_doubly_even_streaming`.

File layout (one per worker, defined in `rust/src/streaming.rs`)::

    header := b"DEKS" version:u32  n:u32  worker_id:u32         (16 bytes LE)
    record := k:u8    aut_order:u128    basis:[u64; k]          (1+16+8k LE)

The reader yields :class:`EnumeratedCode` so downstream code that only
touches ``code.rref`` + ``info.aut_order`` works without modification —
the dropped fields (``canonical_column_order``, ``aut_generators``,
``column_orbits``) are returned as empty tuples.

At N=29 a worker file is ~30 MB; mmap is sufficient (no need to chunk).
"""

from __future__ import annotations

import mmap
import struct
from collections.abc import Iterator
from pathlib import Path

from ..canon.nauty import CanonInfo
from ..spec.codes import Code
from .augment import EnumeratedCode

MAGIC = b"DEKS"
VERSION = 1
HEADER_LEN = 16

# Little-endian unpackers for the per-record fields.
_AUT_STRUCT = struct.Struct("<Q Q")  # u128 split into two u64 (low, high)
_U64_STRUCT = struct.Struct("<Q")


class StreamFormatError(ValueError):
    """Raised when a binary stream file has an unexpected magic, version,
    or truncated payload."""


def _parse_header(mm: mmap.mmap, path: Path) -> tuple[int, int]:
    """Read + validate the 16-byte file header from an open mmap."""
    if len(mm) < HEADER_LEN:
        raise StreamFormatError(f"{path}: file shorter than {HEADER_LEN}-byte header")
    if mm[0:4] != MAGIC:
        raise StreamFormatError(f"{path}: bad magic {mm[0:4]!r} (expected {MAGIC!r})")
    version, n, worker_id = struct.unpack("<III", mm[4:16])
    if version != VERSION:
        raise StreamFormatError(
            f"{path}: format version {version} not supported (expected {VERSION})"
        )
    return n, worker_id


def _check_n(n_hdr: int, expect_n: int | None, path: Path) -> None:
    if expect_n is not None and n_hdr != expect_n:
        raise StreamFormatError(
            f"{path}: header N={n_hdr} but caller expected N={expect_n}"
        )


def iter_stream_file(path: Path, expect_n: int | None = None) -> Iterator[EnumeratedCode]:
    """Iterate over canonical classes in a single ``out.w*.bin`` file.

    Yields :class:`EnumeratedCode` with ``info.aut_order`` populated and
    the other ``CanonInfo`` fields set to empty tuples (the streaming
    format drops them — see module docstring).

    ``expect_n``: if not ``None``, asserts the file's header N matches.
    """
    with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        n, _worker_id = _parse_header(mm, path)
        _check_n(n, expect_n, path)
        # Materialise records into a list before mmap closes — yielding
        # while holding the mmap would let the caller leak references.
        records: list[tuple[int, tuple[int, ...]]] = []
        i = HEADER_LEN
        end = len(mm)
        while i < end:
            k = mm[i]
            need = 1 + 16 + 8 * k
            if i + need > end:
                raise StreamFormatError(
                    f"{path}: truncated record at byte {i} (need {need} bytes for k={k})"
                )
            lo, hi = _AUT_STRUCT.unpack_from(mm, i + 1)
            aut_order = (hi << 64) | lo
            if k == 0:
                basis: tuple[int, ...] = ()
            else:
                basis = struct.unpack_from(f"<{k}Q", mm, i + 17)
            records.append((aut_order, basis))
            i += need
        if i != end:  # pragma: no cover -- guarded above
            raise StreamFormatError(f"{path}: trailing {end - i} bytes after last record")
    for aut_order, basis in records:
        code = Code(n, basis)
        info = CanonInfo(
            canonical_column_order=(),
            aut_generators=(),
            aut_order=aut_order,
            column_orbits=(),
        )
        yield EnumeratedCode(code=code, info=info)


def iter_streamed_codes(output_dir: Path, expect_n: int | None = None) -> Iterator[EnumeratedCode]:
    """Iterate over canonical classes across every ``out.w*.bin`` file in
    ``output_dir``, in sorted-by-filename order (deterministic, but not
    DFS — workers complete in arbitrary order at N >= 22)."""
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("out.w*.bin"))
    if not files:
        raise FileNotFoundError(f"no out.w*.bin files in {output_dir}")
    for path in files:
        yield from iter_stream_file(path, expect_n=expect_n)


def count_by_k(output_dir: Path, expect_n: int | None = None) -> dict[int, int]:
    """Cheap pass that counts classes per rank without constructing
    :class:`Code` / :class:`CanonInfo` objects — used by the merge /
    validation harness on N=29-sized streams."""
    counts: dict[int, int] = {}
    output_dir = Path(output_dir)
    for path in sorted(output_dir.glob("out.w*.bin")):
        with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            n, _ = _parse_header(mm, path)
            _check_n(n, expect_n, path)
            i = HEADER_LEN
            end = len(mm)
            while i < end:
                k = mm[i]
                counts[k] = counts.get(k, 0) + 1
                i += 1 + 16 + 8 * k
    return counts


def sum_mass_by_k(
    output_dir: Path, factorial_n: int, expect_n: int | None = None
) -> dict[int, int]:
    """Compute ``Σ N!/|Aut(C)|`` per rank — the mass-formula quantity to
    compare against ``gaborit_sigma(N, k)``. One mmap pass per file."""
    mass: dict[int, int] = {}
    output_dir = Path(output_dir)
    for path in sorted(output_dir.glob("out.w*.bin")):
        with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            n, _ = _parse_header(mm, path)
            _check_n(n, expect_n, path)
            i = HEADER_LEN
            end = len(mm)
            while i < end:
                k = mm[i]
                lo, hi = _AUT_STRUCT.unpack_from(mm, i + 1)
                aut = (hi << 64) | lo
                mass[k] = mass.get(k, 0) + factorial_n // aut
                i += 1 + 16 + 8 * k
    return mass
