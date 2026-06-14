#!/usr/bin/env python3
"""Cloud (frontier_depth x self-subdivide-delta) sweep for the D20 tail lever.

WHY this exists
---------------
The cloud finding "**d=5 is the 96-core knee**" (N=28 d=4/5/6 =
301.6/139.8/220.1 s, 2026-06-14) was measured **without** the
self-subdivision lever. That lever (`DOUBLY_EVEN_SELF_SUBDIVIDE`, D20)
*decouples* the two jobs `FRONTIER_DEPTH` used to do with one number:

  * `frontier_depth`            -> the serial **seeder span** (grows with depth)
  * `frontier_depth + delta`    -> the finest **adaptive granularity** the
                                   lever can reach (a worker donates while its
                                   parent depth `k <= frontier_depth + delta`,
                                   so the deepest donated subtree is at
                                   `frontier_depth + delta + 1`)

So the optimum is now a 2-D point, and likely shifts *shallower*: a cheap
shallow static frontier + an adequate delta should match a deep static
frontier's balance without paying its seeder span. This sweep finds that
point empirically on the clean 96-core box.

DISCIPLINE: **no single run may exceed `--timeout` seconds** (default 120 =
2 min). Each arm runs the counts-only kernel (run_counts.py) at the cheap
sweep size (default N=27, ~20-40 s on c4a-96-metal); any config that blows
the budget is killed and recorded as OVER_BUDGET (it's a loser anyway).
N=28/N=29 are *confirmation* runs, not sweep arms -- run the winning
(d, delta) at those sizes separately (they intentionally exceed 2 min).

The tail metric is read live from `<out>/progress.json` (the kernel's
per-rank mass-vs-quota snapshot): `t90`/`t99` = wall-clock to 90 %/99 % of
total mass, and `tail = wall - t90`. A small tail means the run finished
flat (the lever kept cores fed); a large tail is the `96->37->16->4->1`
straggler collapse the lever exists to remove.

USAGE (on the cloud box, cwd = repo root, kernel built --features parallel)
---------------------------------------------------------------------------
    python3 scripts/cloud_depth_sweep.py                  # N=27, default grid
    python3 scripts/cloud_depth_sweep.py --N 27 --threads 96
    python3 scripts/cloud_depth_sweep.py --timeout 90 --repeats 2
    python3 scripts/cloud_depth_sweep.py --grid "off:4 off:5 on:4:1 on:4:2 on:5:1"

Launch under tmux/nohup so a dropped ssh can't SIGHUP it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Known equivalence-class totals (correctness gate). Sources: DFGHILM
# Table 3 / the running mass formula / prior cloud records.
EXPECTED_CLASSES = {
    22: 5_118,
    24: 37_496,
    26: 494_272,
    27: 2_673_492,
    28: 21_505_546,
    29: 239_465_540,
}

# Mass-fraction thresholds whose first-crossing wall-clock we record.
THRESHOLDS = (0.50, 0.90, 0.95, 0.99)

WALL_RE = re.compile(r"Kernel wall:\s*([0-9.]+)\s*s")
CLASSES_RE = re.compile(r"total classes:\s*([0-9]+)")


def parse_grid(spec: str) -> list[dict]:
    """`off:D` / `on:D:DELTA` tokens -> config dicts."""
    cfgs = []
    for tok in spec.split():
        parts = tok.split(":")
        mode = parts[0].lower()
        if mode == "off":
            cfgs.append({"mode": "off", "depth": int(parts[1]), "delta": None})
        elif mode == "on":
            cfgs.append(
                {"mode": "on", "depth": int(parts[1]), "delta": int(parts[2])}
            )
        else:
            raise SystemExit(f"bad grid token {tok!r} (want off:D or on:D:DELTA)")
    return cfgs


def default_grid() -> list[dict]:
    """OFF baselines at d=4/5/6, plus the lever ON across the plausible
    shallow-static x delta-reach region. ~11 arms; at N=27 on 96 cores
    each is well under 2 min."""
    return parse_grid(
        "off:4 off:5 off:6 "
        "on:4:1 on:4:2 on:4:3 "
        "on:5:1 on:5:2 "
        "on:3:2 on:3:3"
    )


def label(cfg: dict) -> str:
    if cfg["mode"] == "off":
        return f"d{cfg['depth']}-OFF"
    return f"d{cfg['depth']}-ON-delta{cfg['delta']}"


def mass_fraction(progress_path: Path) -> float | None:
    """Sum(mass)/Sum(quota) from the kernel's progress.json snapshot."""
    try:
        d = json.loads(progress_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    pairs = d.get("mass_quota")
    if not pairs:
        return None
    tot_mass = tot_quota = 0
    for m, q in pairs:
        tot_mass += int(m)
        tot_quota += int(q)
    if tot_quota == 0:
        return None
    return tot_mass / tot_quota


def run_one(cfg: dict, args, out_dir: Path) -> dict:
    """Launch one arm, poll progress.json for the tail metric, enforce the
    per-run timeout. Returns a result dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    if progress_path.exists():
        progress_path.unlink()

    env = os.environ.copy()
    env["DOUBLY_EVEN_THREADS"] = str(args.threads)
    env["DOUBLY_EVEN_FRONTIER_DEPTH"] = str(cfg["depth"])
    if cfg["mode"] == "on":
        env["DOUBLY_EVEN_SELF_SUBDIVIDE"] = "1"
        env["DOUBLY_EVEN_SELF_SUBDIVIDE_DELTA"] = str(cfg["delta"])
        if args.poll_ms is not None:
            env["DOUBLY_EVEN_SELF_SUBDIVIDE_POLL_MS"] = str(args.poll_ms)
    else:
        env.pop("DOUBLY_EVEN_SELF_SUBDIVIDE", None)

    cmd = [
        "uv", "run", "python", "scripts/run_counts.py",
        "--N", str(args.N),
        "--output-dir", str(out_dir),
        "--label", label(cfg),
        "--progress-interval", str(args.progress_interval),
    ]
    log_path = out_dir / "run.log"
    log_f = log_path.open("w")
    t0 = time.monotonic()
    # New session so we can kill the whole process group on timeout (uv ->
    # python -> kernel threads).
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    crossings: dict[float, float] = {}
    timed_out = False
    while True:
        rc = proc.poll()
        now = time.monotonic() - t0
        frac = mass_fraction(progress_path)
        if frac is not None:
            for thr in THRESHOLDS:
                if thr not in crossings and frac >= thr:
                    crossings[thr] = round(now, 2)
        if rc is not None:
            break
        if now > args.timeout:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(2)
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            break
        time.sleep(0.25)
    wall_observed = round(time.monotonic() - t0, 2)
    log_f.close()
    text = log_path.read_text()

    res = {
        "label": label(cfg), **cfg,
        "status": "OK",
        "wall_observed_s": wall_observed,
        "t50_s": crossings.get(0.50),
        "t90_s": crossings.get(0.90),
        "t95_s": crossings.get(0.95),
        "t99_s": crossings.get(0.99),
        "kernel_wall_s": None,
        "classes": None,
        "classes_ok": None,
    }
    if res["t90_s"] is not None:
        res["tail_s"] = round(wall_observed - res["t90_s"], 2)

    if timed_out:
        res["status"] = "OVER_BUDGET"
        return res
    if proc.returncode != 0:
        res["status"] = f"FAILED(rc={proc.returncode})"
        return res
    if "MISMATCH" in text:
        res["status"] = "MASS_MISMATCH"
    m = WALL_RE.search(text)
    if m:
        res["kernel_wall_s"] = float(m.group(1))
    c = CLASSES_RE.search(text)
    if c:
        res["classes"] = int(c.group(1))
        exp = EXPECTED_CLASSES.get(args.N)
        if exp is not None:
            res["classes_ok"] = res["classes"] == exp
            if not res["classes_ok"]:
                res["status"] = f"WRONG_COUNT({res['classes']}!={exp})"
    return res


def fmt(v, width, prec=1):
    if v is None:
        return "-".rjust(width)
    if isinstance(v, float):
        return f"{v:.{prec}f}".rjust(width)
    return str(v).rjust(width)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=27,
                   help="sweep size (default 27 ~ <2 min/arm on 96 cores)")
    p.add_argument("--threads", type=int, default=os.cpu_count(),
                   help="worker threads (default nproc)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="HARD per-run budget in seconds (default 120 = 2 min)")
    p.add_argument("--progress-interval", type=int, default=1,
                   help="progress.json cadence; sets tail-metric resolution")
    p.add_argument("--poll-ms", type=int, default=None,
                   help="override DOUBLY_EVEN_SELF_SUBDIVIDE_POLL_MS")
    p.add_argument("--repeats", type=int, default=1,
                   help="runs per config; medians reported")
    p.add_argument("--grid", type=str, default=None,
                   help='custom grid, e.g. "off:5 on:4:2 on:3:3"')
    p.add_argument("--out-root", type=Path, default=None)
    args = p.parse_args()

    grid = parse_grid(args.grid) if args.grid else default_grid()
    out_root = args.out_root or (Path.home() / f"depth-sweep-N{args.N}")
    out_root.mkdir(parents=True, exist_ok=True)

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True
    ).stdout.strip()

    print(f"# D20 depth x delta sweep | N={args.N} threads={args.threads} "
          f"timeout={args.timeout:g}s repeats={args.repeats} git={git_sha}")
    print(f"# expected classes N={args.N}: "
          f"{EXPECTED_CLASSES.get(args.N, 'UNKNOWN')}")
    print(f"# tail metric from progress.json (interval={args.progress_interval}s); "
          f"OVER_BUDGET = killed at the {args.timeout:g}s cap\n")

    all_results = []
    for cfg in grid:
        walls, samples = [], []
        for r in range(args.repeats):
            sub = f"{label(cfg)}" + (f".r{r}" if args.repeats > 1 else "")
            res = run_one(cfg, args, out_root / sub)
            samples.append(res)
            tag = (f"wall={res['kernel_wall_s']}s t90={res['t90_s']} "
                   f"tail={res.get('tail_s')} [{res['status']}]")
            print(f"  {sub:<18} {tag}")
            if res["status"] == "OK" and res["kernel_wall_s"] is not None:
                walls.append(res["kernel_wall_s"])
        agg = dict(samples[0])
        if walls:
            agg["kernel_wall_s"] = round(statistics.median(walls), 2)
            agg["wall_samples"] = walls
        t90s = [s["t90_s"] for s in samples if s["t90_s"] is not None]
        agg["t90_s"] = round(statistics.median(t90s), 2) if t90s else None
        # tail = wall_observed - t90 (both process-relative); median over runs.
        tails = [s["tail_s"] for s in samples if s.get("tail_s") is not None]
        agg["tail_s"] = round(statistics.median(tails), 2) if tails else None
        all_results.append(agg)

    # Ranked table (finished arms by wall, then the rest).
    ok = [r for r in all_results if r["status"] == "OK" and r["kernel_wall_s"]]
    bad = [r for r in all_results if r not in ok]
    ok.sort(key=lambda r: r["kernel_wall_s"])

    print("\n" + "=" * 74)
    print(f"{'config':<18}{'wall_s':>9}{'t50':>8}{'t90':>8}{'t99':>8}"
          f"{'tail':>8}{'ok':>5}")
    print("-" * 74)
    for r in ok + bad:
        ok_mark = ("y" if r["classes_ok"] else "N") if r["classes_ok"] is not None else "?"
        status = "" if r["status"] == "OK" else f"  <- {r['status']}"
        print(f"{r['label']:<18}{fmt(r['kernel_wall_s'],9)}{fmt(r['t50_s'],8)}"
              f"{fmt(r['t90_s'],8)}{fmt(r['t99_s'],8)}{fmt(r.get('tail_s'),8)}"
              f"{ok_mark:>5}{status}")
    print("=" * 74)
    if ok:
        best = ok[0]
        off_baseline = next((r for r in ok if r["mode"] == "off"), None)
        line = f"WINNER: {best['label']}  wall={best['kernel_wall_s']}s"
        if off_baseline and off_baseline is not best:
            spd = off_baseline["kernel_wall_s"] / best["kernel_wall_s"]
            line += (f"  vs best-OFF {off_baseline['label']} "
                     f"{off_baseline['kernel_wall_s']}s = {spd:.2f}x")
        print(line)
        print("Then confirm the winner at N=28 / N=29 (separate, >2 min runs) "
              "before flipping the production default ON.")

    ts = subprocess.run(["date", "-u", "+%Y%m%dT%H%M%SZ"],
                        capture_output=True, text=True).stdout.strip()
    out_json = out_root / f"sweep-N{args.N}-{git_sha}-{ts}.json"
    out_json.write_text(json.dumps(
        {"N": args.N, "threads": args.threads, "timeout_s": args.timeout,
         "git_sha": git_sha, "results": all_results}, indent=2))
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
