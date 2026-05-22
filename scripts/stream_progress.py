"""Tail-reader sidecar for streaming-kernel runs. Polls per-worker
``out.w*.bin`` files and prints a per-k progress table on a fixed
interval.

Usage::

    # Local prototype (N=22..25), 30 s refresh in place:
    uv run python scripts/stream_progress.py --N 24 \\
        --output-dir /tmp/n24-stream

    # Production N=29 on c4a-72, 5-min interval, scrollable history:
    uv run python scripts/stream_progress.py --N 29 \\
        --output-dir /mnt/scratch/n29 \\
        --interval 300 --append

Runs alongside ``scripts/run_streaming.py``. Reads each ``out.w*.bin``
incrementally — per-file byte offsets are cached so each tick only
scans freshly-flushed bytes. The 256 KB BufWriter (see
``rust/src/streaming.rs``) bounds the lag at ~2000 records per worker.

Exit signal: when ``stats.json`` appears in ``--output-dir`` (written
by ``run_streaming.py`` after the kernel returns), the sidecar prints
one final post-flush snapshot and exits 0. Ctrl-C also exits cleanly.
"""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.enumerate.stream_reader import (  # noqa: E402
    HEADER_LEN,
    MAGIC,
    VERSION,
    StreamFormatError,
)
from doubly_even.spec.mass import gaborit_sigma  # noqa: E402

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--N", type=int, required=True)
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
        help="Override max_k (default N//2 — matches run_streaming.py).",
    )
    args = parser.parse_args()

    N = args.N
    max_k = args.max_k if args.max_k is not None else N // 2
    factorial_N = math.factorial(N)
    quotas = [gaborit_sigma(N, k) for k in range(max_k + 1)]
    quota_total = sum(quotas)

    tails: dict[str, WorkerTail] = {}
    t_start = time.time()
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
