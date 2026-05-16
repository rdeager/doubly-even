"""End-to-end enumeration bench: ours (Rust+nauty) vs Sage backend.

For each N, runs `enumerate_doubly_even(N)` twice:

1. **Baseline**: kernel + nauty (the default; what `scripts/bench.py` measures).
2. **Sage**: forces the Python fallback (disables the Rust kernel) and
   routes every `canon_info` call through a Sage daemon
   (`scripts/sage_canon_daemon.py`) using
   ``DOUBLY_EVEN_CANON_BACKEND=sage_partn_ref``.

We disable the kernel for the Sage path so the comparison is honest:
both runs use the *same algorithmic recursion* (the Python `_traverse`),
differing only in which canonicaliser they call. That isolates the
canonicaliser cost.

Caveat: the Python `_traverse` is itself slower than the Rust kernel for
non-canonicaliser work (subspace orbit BFS, etc.). To separate the two
effects we also measure a "Python + nauty" baseline so we can
attribute the gap to canon vs non-canon work.

Usage:
    uv run python scripts/bench_sage_e2e.py --N 14,16,18

For N=20, 22 use ``--timeout 1800`` since Sage paths can take many
minutes.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def run_one(N: int, backend: str, disable_kernel: bool, timeout_s: float) -> dict:
    """Run enumerate_doubly_even(N) under a given backend config.

    To avoid module-import side effects spanning runs, this is invoked in
    a subprocess via ``uv run python -c …``.
    """
    import subprocess
    code = f"""
import os, sys, time, json
os.environ['DOUBLY_EVEN_CANON_BACKEND'] = {backend!r}
sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
{'import doubly_even.enumerate.augment as A; A._kernel = None' if disable_kernel else ''}
from doubly_even.enumerate.augment import enumerate_doubly_even
t0 = time.perf_counter()
classes = 0
per_k = {{}}
for ec in enumerate_doubly_even({N}):
    classes += 1
    per_k[ec.code.rank] = per_k.get(ec.code.rank, 0) + 1
wall = time.perf_counter() - t0
out = {{"N": {N}, "wall": wall, "classes": classes, "per_k": per_k}}
# If using the sage backend, dump per-process IPC stats too.
if {backend!r} == 'sage_partn_ref':
    from doubly_even.canon.sage_proxy import stats
    out['sage_stats'] = stats()
print('@@RESULT@@' + json.dumps(out))
"""
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["uv", "run", "python", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(REPO_ROOT),
    )
    wall_outer = time.perf_counter() - t0

    # Extract the JSON result line
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            result = json.loads(line[len("@@RESULT@@"):])
            break
    if result is None:
        return {
            "error": "no_result",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1500:],
            "stderr_tail": proc.stderr[-1500:],
            "wall_outer": wall_outer,
        }
    result["wall_outer"] = wall_outer
    result["returncode"] = proc.returncode
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", default="14,16,18",
                    help="comma-separated list of N (default 14,16,18)")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-run timeout in seconds (default 1800)")
    ap.add_argument("--label", default="sage-e2e",
                    help="output JSON label (default sage-e2e)")
    ap.add_argument("--configs", default="kernel_nauty,py_nauty,py_sage",
                    help="comma-separated configs to run")
    args = ap.parse_args()

    Ns = [int(s) for s in args.N.split(",") if s.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    runners = {
        "kernel_nauty": ("nauty", False),   # default (Rust kernel + nauty)
        "py_nauty":     ("nauty", True),    # Python fallback + nauty (isolates non-canon work)
        "py_sage":      ("sage_partn_ref", True),  # Python fallback + Sage daemon
    }

    results: dict[int, dict] = {}
    for N in Ns:
        results[N] = {}
        for cfg in configs:
            backend, disable_kernel = runners[cfg]
            print(f"\n=== N={N} | config={cfg} (backend={backend}, "
                  f"disable_kernel={disable_kernel}) ===", flush=True)
            t = time.perf_counter()
            r = run_one(N, backend, disable_kernel, args.timeout)
            print(f"  wall_outer={time.perf_counter()-t:.2f}s", flush=True)
            if "error" in r:
                print(f"  ERROR: {r['error']}", flush=True)
                if "stderr_tail" in r:
                    print(r["stderr_tail"], flush=True)
            else:
                print(f"  wall={r['wall']:.3f}s  classes={r['classes']}  "
                      f"per_k={r['per_k']}", flush=True)
                if "sage_stats" in r:
                    s = r["sage_stats"]
                    print(f"  sage: {s['calls']} canon calls, "
                          f"{s['ipc_seconds']:.2f}s in IPC "
                          f"({s['avg_ipc_us_per_call']:.0f} µs/call)",
                          flush=True)
            results[N][cfg] = r

    # Write JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = HERE / "bench-results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{ts}-{args.label}.json"
    payload = {
        "label": args.label,
        "timestamp_utc": ts,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "configs": configs,
        "per_N": {str(N): results[N] for N in Ns},
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}", flush=True)

    # Print summary
    print(f"\n{'N':>3}", end="")
    for cfg in configs:
        print(f" {cfg:>16}", end="")
    print()
    for N in Ns:
        print(f"{N:>3}", end="")
        for cfg in configs:
            r = results[N][cfg]
            if "error" in r:
                print(f" {'ERROR':>16}", end="")
            else:
                print(f" {r['wall']:>14.3f}s", end="")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
