"""Bench-only proxy to a Sage daemon for ``canon_info``.

Spawns ``scripts/sage_canon_daemon.py`` on first use and proxies per-code
canon_info requests over stdin/stdout. Returns the same :class:`CanonInfo`
shape as the in-tree backends so :func:`enumerate_doubly_even` can use it
unchanged.

Enable via ``DOUBLY_EVEN_CANON_BACKEND=sage_partn_ref``. The daemon uses
Sage's ``LinearBinaryCodeStruct`` (Robert Miller's binary-specialised
partition refinement), the fastest Sage canonicaliser for binary codes.

This module is for benchmarking only; it is not on any hot path and has
no test coverage beyond the bench scripts that invoke it.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from pathlib import Path

from ..spec.codes import Code
from .nauty import CanonInfo

_SAGE_BIN = os.environ.get("SAGE_BIN", "/usr/local/bin/sage")
_DAEMON_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "sage_canon_daemon.py"

_proc: subprocess.Popen | None = None
_call_count = 0
_ipc_seconds = 0.0


def _start() -> subprocess.Popen:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc
    _proc = subprocess.Popen(
        [_SAGE_BIN, str(_DAEMON_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )
    # Wait for ready signal
    ready = _proc.stdout.readline()
    if not ready:
        err = _proc.stderr.read()
        raise RuntimeError(f"Sage daemon failed to start: {err}")
    if "ready" not in ready:
        raise RuntimeError(f"Sage daemon sent unexpected greeting: {ready!r}")
    print(f"[sage_proxy] daemon started, pid={_proc.pid}", file=sys.stderr,
          flush=True)
    atexit.register(_stop)
    return _proc


def _stop() -> None:
    global _proc
    if _proc is None:
        return
    try:
        _proc.stdin.write(json.dumps({"req": "quit"}) + "\n")
        _proc.stdin.flush()
        _proc.wait(timeout=5)
    except Exception:
        _proc.kill()
    _proc = None


def canon_info_via_sage(C: Code) -> CanonInfo:
    """Compute canon_info via the Sage daemon."""
    global _call_count, _ipc_seconds
    import time
    proc = _start()
    rref, _ = C.rref_basis()
    req = {"req": "canon", "rref": list(rref), "n": C.n}
    t0 = time.perf_counter()
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    _ipc_seconds += time.perf_counter() - t0
    _call_count += 1
    if not line:
        err = proc.stderr.read()
        raise RuntimeError(f"Sage daemon died: {err}")
    resp = json.loads(line)
    if "error" in resp:
        raise RuntimeError(f"Sage daemon error: {resp['error']}")
    return CanonInfo(
        canonical_column_order=tuple(resp["col_order"]),
        aut_generators=tuple(tuple(g) for g in resp["aut_gens"]),
        aut_order=int(resp["aut_order"]),
        column_orbits=tuple(resp["column_orbits"]),
    )


def stats() -> dict:
    """Return per-process counters."""
    return {"calls": _call_count, "ipc_seconds": _ipc_seconds,
            "avg_ipc_us_per_call": (_ipc_seconds * 1e6 / _call_count)
                                    if _call_count else 0.0}
