"""Live progress for long enumeration runs — the per-rank
"fraction of σ(N, k) mass found" table, with ETA projected from the
mass fraction.

Two run types are supported, auto-detected from the output directory:

- **Streaming runs** (``scripts/run_streaming.py``): tails the
  per-worker ``out.w*.bin`` files incrementally (per-file byte offsets
  are cached so each tick only scans freshly-flushed bytes; the 256 KB
  BufWriter in ``rust/core/src/streaming.rs`` bounds the lag at ~2000
  records per worker). Exit signal: ``stats.json`` appears.
- **Counts-only runs** (``scripts/run_counts.py``, the N ≥ 30 mode):
  reads the ``progress.json`` snapshot the kernel's watcher thread
  atomically rewrites (per-rank mass vs quota as decimal strings).
  No per-class records exist in this mode, so the classes column shows
  "—". Exit signal: the snapshot's ``done`` flag.

Entry points: ``uv run dec progress`` (the CLI) and
``scripts/stream_progress.py`` (kept as a thin wrapper for the recipes
in ``docs/``). History: this module is the former
``scripts/stream_progress.py`` body, moved into the package 2026-06-13
when counts-mode support was added.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

from doubly_even.enumerate.stream_reader import (
    HEADER_LEN,
    MAGIC,
    VERSION,
    StreamFormatError,
)
from doubly_even.spec.mass import gaborit_sigma

_AUT_STRUCT = struct.Struct("<QQ")
_HDR_TAIL_STRUCT = struct.Struct("<III")


class WorkerTail:
    """Incremental record reader for one ``out.w*.bin`` file.

    Maintains a byte offset between ticks; on each :meth:`update` call,
    seeks past the last fully-parsed record and walks forward until it
    hits EOF or a partially-flushed tail. Counts and per-k mass are
    accumulated in place.
    """

    __slots__ = ("path", "expect_n", "offset", "header_parsed", "counts", "mass")

    def __init__(self, path: Path, expect_n: int) -> None:
        self.path = path
        self.expect_n = expect_n
        self.offset = 0
        self.header_parsed = False
        self.counts: dict[int, int] = {}
        self.mass: dict[int, int] = {}

    def update(self, factorial_n: int) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size <= self.offset:
            return
        with self.path.open("rb") as fh:
            if not self.header_parsed:
                if size < HEADER_LEN:
                    return
                header = fh.read(HEADER_LEN)
                if header[0:4] != MAGIC:
                    raise StreamFormatError(f"{self.path}: bad magic {header[0:4]!r}")
                version, n_hdr, _worker_id = _HDR_TAIL_STRUCT.unpack(header[4:HEADER_LEN])
                if version != VERSION:
                    raise StreamFormatError(
                        f"{self.path}: version {version} unsupported (expected {VERSION})"
                    )
                if n_hdr != self.expect_n:
                    raise StreamFormatError(
                        f"{self.path}: header N={n_hdr} != expected N={self.expect_n}"
                    )
                self.offset = HEADER_LEN
                self.header_parsed = True

            fh.seek(self.offset)
            buf = fh.read()
        pos = 0
        end = len(buf)
        while pos < end:
            if pos + 17 > end:
                break
            k = buf[pos]
            need = 1 + 16 + 8 * k
            if pos + need > end:
                break
            lo, hi = _AUT_STRUCT.unpack_from(buf, pos + 1)
            aut = (hi << 64) | lo
            self.counts[k] = self.counts.get(k, 0) + 1
            self.mass[k] = self.mass.get(k, 0) + factorial_n // aut
            pos += need
        self.offset += pos


def _fmt_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _fmt_bytes(b: int) -> str:
    x = float(b)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0:
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} PB"


def _fmt_count(n: int) -> str:
    if n >= 10_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n}"


def _fmt_eta(elapsed: float, fraction: float) -> str:
    """Project remaining wall time from mass-fraction progress. Returns
    ``"—"`` while the fraction is below 0.1 % (extrapolation unreliable)
    and ``"~done"`` once the mass formula has filled."""
    if fraction <= 0.001:
        return "—"
    if fraction >= 0.9999:
        return "~done"
    remaining = elapsed * (1.0 - fraction) / fraction
    return _fmt_elapsed(remaining)


def _read_meminfo_bytes() -> tuple[int, int] | None:
    """Return (used, total) bytes from /proc/meminfo, or None if
    unreadable (non-Linux host). Uses MemAvailable when present —
    matches what `free -h` reports as "available" / "used"."""
    try:
        with open("/proc/meminfo") as fh:
            info = {}
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    try:
                        info[parts[0][:-1]] = int(parts[1]) * 1024
                    except ValueError:
                        continue
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if avail is None:
            free = info.get("MemFree", 0)
            cached = info.get("Cached", 0)
            buffers = info.get("Buffers", 0)
            avail = free + cached + buffers
        if total is None:
            return None
        return max(0, total - avail), total
    except OSError:
        return None


def _disk_usage_safe(path: Path) -> tuple[int, int] | None:
    """(free, total) bytes for the filesystem holding ``path``."""
    try:
        du = shutil.disk_usage(path)
        return du.free, du.total
    except OSError:
        return None


class CountsProgress:
    """Reader for the counts-mode ``progress.json`` snapshot (written
    atomically by the kernel's watcher thread)."""

    __slots__ = ("path", "n", "max_k", "elapsed_s", "done", "mass", "quota")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.n: int | None = None
        self.max_k: int | None = None
        self.elapsed_s = 0.0
        self.done = False
        self.mass: list[int] = []
        self.quota: list[int] = []

    def update(self) -> bool:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        self.n = data["n"]
        self.max_k = data["max_k"]
        self.elapsed_s = float(data["elapsed_s"])
        self.done = bool(data["done"])
        self.mass = [int(m) for m, _q in data["mass_quota"]]
        self.quota = [int(q) for _m, q in data["mass_quota"]]
        return True


def render(
    *,
    N: int,
    max_k: int,
    quotas: list[int],
    quota_total: int,
    tails: dict[str, WorkerTail],
    last_per_k: dict[int, int],
    last_total: int,
    last_tick_ts: float | None,
    last_offsets: dict[str, int],
    elapsed: float,
    output_dir: Path,
) -> tuple[list[str], dict[int, int], int, dict[str, int]]:
    counts: dict[int, int] = {}
    mass: dict[int, int] = {}
    for t in tails.values():
        for k, v in t.counts.items():
            counts[k] = counts.get(k, 0) + v
        for k, v in t.mass.items():
            mass[k] = mass.get(k, 0) + v

    total_classes = sum(counts.values())
    total_mass = sum(mass.values())
    total_bytes = 0
    for name in tails:
        try:
            total_bytes += (output_dir / name).stat().st_size
        except FileNotFoundError:
            pass

    now = time.time()
    delta_total = total_classes - last_total
    if last_tick_ts is not None and now > last_tick_ts:
        rate = delta_total / (now - last_tick_ts)
        rate_str = f"{_fmt_count(int(rate))}/s"
    else:
        rate_str = "—"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pct_total = (100.0 * total_mass / quota_total) if quota_total else 0.0
    eta_str = _fmt_eta(elapsed, total_mass / quota_total if quota_total else 0.0)

    cur_offsets: dict[str, int] = {name: t.offset for name, t in tails.items()}
    active = sum(
        1 for name, off in cur_offsets.items() if off > last_offsets.get(name, 0)
    )

    mem = _read_meminfo_bytes()
    if mem is not None:
        used_b, total_b = mem
        mem_str = (
            f"mem: {used_b / (1024**3):.1f} / {total_b / (1024**3):.1f} GiB "
            f"({100.0 * used_b / total_b:.1f}%)"
        )
    else:
        mem_str = "mem: —"

    disk = _disk_usage_safe(output_dir if output_dir.exists() else output_dir.parent)
    if disk is not None:
        free_b, _disk_total = disk
        disk_str = (
            f"disk: {_fmt_bytes(total_bytes)} written / "
            f"{free_b / (1024**3):.1f} GiB free"
        )
    else:
        disk_str = f"disk: {_fmt_bytes(total_bytes)} written"

    lines = [
        f"[{ts}] N={N}  workers={len(tails)} ({active} active)  "
        f"elapsed={_fmt_elapsed(elapsed)}  rate={rate_str}  ETA={eta_str}",
        f"  {mem_str}    {disk_str}",
        f"   k    classes        mass / sigma                    %    Δclasses",
    ]
    for k in range(max_k + 1):
        c = counts.get(k, 0)
        q = quotas[k]
        if c == 0 and q == 0:
            continue
        m = mass.get(k, 0)
        pct = (100.0 * m / q) if q else (100.0 if m == 0 else float("inf"))
        delta = c - last_per_k.get(k, 0)
        lines.append(
            f"  {k:>2d}  {c:>9d}  {m:>12.4e} / {q:<12.4e}  {pct:>6.2f}  {delta:+10d}"
        )
    lines.append(
        f"  total: {_fmt_count(total_classes)} classes  "
        f"({total_mass:.3e} / {quota_total:.3e} mass = {pct_total:.2f}%)"
    )
    return lines, counts, total_classes, cur_offsets


def render_counts(cp: CountsProgress) -> list[str]:
    """Per-rank mass-percentage table for a counts-mode snapshot. The
    kernel keeps no live per-class counts (workers fold locally), so
    only the mass columns — the actual progress signal — appear."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quota_total = sum(cp.quota)
    mass_total = sum(cp.mass)
    frac = mass_total / quota_total if quota_total else 0.0
    eta = "~done" if cp.done else _fmt_eta(cp.elapsed_s, frac)

    mem = _read_meminfo_bytes()
    mem_str = (
        f"mem: {mem[0] / (1024**3):.1f} / {mem[1] / (1024**3):.1f} GiB"
        if mem
        else "mem: —"
    )
    lines = [
        f"[{ts}] N={cp.n}  counts-only run  "
        f"elapsed={_fmt_elapsed(cp.elapsed_s)}  ETA={eta}"
        f"{'  [DONE]' if cp.done else ''}",
        f"  {mem_str}    (snapshot: {cp.path.name})",
        f"   k          mass / sigma                    %",
    ]
    for k, (m, q) in enumerate(zip(cp.mass, cp.quota)):
        if m == 0 and q == 0:
            continue
        pct = (100.0 * m / q) if q else (100.0 if m == 0 else float("inf"))
        lines.append(f"  {k:>2d}  {m:>12.4e} / {q:<12.4e}  {pct:>6.2f}")
    lines.append(
        f"  total mass: {mass_total:.3e} / {quota_total:.3e} = {100.0 * frac:.2f}%"
    )
    return lines


def discover_and_update(
    output_dir: Path,
    expect_n: int,
    factorial_n: int,
    tails: dict[str, WorkerTail],
) -> None:
    if not output_dir.is_dir():
        return
    for p in sorted(output_dir.glob("out.w*.bin")):
        t = tails.get(p.name)
        if t is None:
            t = WorkerTail(p, expect_n)
            tails[p.name] = t
        t.update(factorial_n)


def _infer_n_from_stream(output_dir: Path) -> int | None:
    """Peek at the first ``out.w*.bin`` header for N."""
    for p in sorted(output_dir.glob("out.w*.bin")):
        try:
            with p.open("rb") as fh:
                header = fh.read(HEADER_LEN)
            if len(header) == HEADER_LEN and header[0:4] == MAGIC:
                _version, n_hdr, _wid = _HDR_TAIL_STRUCT.unpack(header[4:HEADER_LEN])
                return n_hdr
        except OSError:
            continue
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--N", type=int, default=None,
        help="Code length. Optional: inferred from progress.json (counts "
             "runs) or the first stream header once either appears.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--interval", type=float, default=30.0,
        help="Seconds between snapshots (default 30; use 300+ for N>=28).",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append each tick to stdout instead of refreshing in place. "
             "Use for long runs where you want scrollable history.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Print one snapshot and exit (useful for scripting / CI).",
    )
    parser.add_argument(
        "--max-k", type=int, default=None,
        help="Override max_k (default N//2 — matches the run scripts).",
    )
    args = parser.parse_args(argv)

    counts_path = args.output_dir / "progress.json"
    t_start = time.time()

    # --- Counts-mode loop: render the kernel's snapshot file. ---------
    def counts_loop() -> int:
        cp = CountsProgress(counts_path)
        try:
            while True:
                if cp.update():
                    body = "\n".join(render_counts(cp))
                    if args.append or args.once:
                        print(body)
                        print()
                        sys.stdout.flush()
                    else:
                        sys.stdout.write("\033[H\033[2J" + body + "\n")
                        sys.stdout.flush()
                    if args.once or cp.done:
                        return 0
                time.sleep(min(args.interval, 5.0) if not cp.mass else args.interval)
        except KeyboardInterrupt:
            print()
            return 130

    # Wait for either source to materialise, then dispatch.
    N = args.N
    while True:
        if counts_path.exists():
            return counts_loop()
        inferred = _infer_n_from_stream(args.output_dir)
        if inferred is not None:
            if N is None:
                N = inferred
            break
        if N is not None and (args.output_dir / "stats.json").exists():
            break
        if args.once:
            print(f"no run artefacts in {args.output_dir} yet")
            return 1
        time.sleep(min(args.interval, 2.0))

    # --- Streaming-mode loop (the original sidecar). -------------------
    max_k = args.max_k if args.max_k is not None else N // 2
    factorial_N = math.factorial(N)
    quotas = [gaborit_sigma(N, k) for k in range(max_k + 1)]
    quota_total = sum(quotas)

    tails: dict[str, WorkerTail] = {}
    last_per_k: dict[int, int] = {}
    last_total = 0
    last_tick_ts: float | None = None
    last_offsets: dict[str, int] = {}

    def snapshot(extra_blank: bool = True) -> tuple[dict[int, int], int, dict[str, int]]:
        discover_and_update(args.output_dir, N, factorial_N, tails)
        lines, counts, total, cur_offsets = render(
            N=N,
            max_k=max_k,
            quotas=quotas,
            quota_total=quota_total,
            tails=tails,
            last_per_k=last_per_k,
            last_total=last_total,
            last_tick_ts=last_tick_ts,
            last_offsets=last_offsets,
            elapsed=time.time() - t_start,
            output_dir=args.output_dir,
        )
        body = "\n".join(lines)
        if args.append or args.once:
            print(body)
            if extra_blank:
                print()
            sys.stdout.flush()
        else:
            sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(body + "\n")
            sys.stdout.flush()
        return counts, total, cur_offsets

    try:
        while True:
            last_per_k, last_total, last_offsets = snapshot()
            last_tick_ts = time.time()

            if args.once:
                return 0

            if (args.output_dir / "stats.json").exists():
                # Kernel returned; one more pass to pick up any post-flush
                # bytes the in-place tick missed, then exit.
                time.sleep(0.5)
                if args.append:
                    print("# stats.json detected — kernel finished; final snapshot:")
                snapshot(extra_blank=False)
                return 0

            time.sleep(args.interval)
    except KeyboardInterrupt:
        if not args.append:
            sys.stdout.write("\n")
        else:
            print("# interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
