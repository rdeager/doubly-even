"""Per-call canonicaliser micro-benchmark: ours vs SageMath.

For each N in --N (default 14,16,18,20,22):

1. Sample doubly-even code classes by running our enumerator and taking
   up to --sample-per-n representatives, stratified across k.
2. Time our `canon_info_native` on each sample (median of --reps reps).
3. Spawn one Sage subprocess per N, feed it the sample matrices, time
   three Sage backends:
   - `LinearCode.canonical_representative('permutation')`  — Feulner codecan
   - `LinearCode.permutation_automorphism_group(algorithm='partition').order()`
        — Robert Miller's binary-specialised partn_ref
   - `PartitionRefinementLinearCode.get_canonical_form()` — low-level Feulner
4. Sanity-check: |Aut| must agree across backends for every sample.
5. Write JSON to scripts/bench-results/<timestamp>-sage-percall.json.

The Sage child script is `scripts/_bench_sage_child.py` (one Sage subprocess
per N to amortise Sage's ~3 s startup cost).

Usage:
    uv run python scripts/bench_sage.py --N 14,16,18,20,22
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.canon.nauty import canon_info_cache_clear  # noqa: E402
from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402

SAGE = "/usr/local/bin/sage"
CHILD = HERE / "_bench_sage_child.py"


def sample_codes(N: int, max_per_k: int) -> list[tuple[int, int, tuple[int, ...]]]:
    """Return up to max_per_k codes per k as (N, k, rref_rows) triples.

    Skips the k=0 (zero) code — Sage's LinearCode rejects empty matrices,
    and the trivial code is uninteresting for canonicaliser comparison.
    """
    canon_info_cache_clear()
    buckets: dict[int, list[tuple[int, int, tuple[int, ...]]]] = defaultdict(list)
    for ec in enumerate_doubly_even(N):
        k = ec.code.rank
        if k == 0:
            continue
        if len(buckets[k]) >= max_per_k:
            continue
        rref, _ = ec.code.rref_basis()
        buckets[k].append((N, k, tuple(rref)))
    out: list[tuple[int, int, tuple[int, ...]]] = []
    for k in sorted(buckets):
        out.extend(buckets[k])
    return out


def time_ours(samples: list[tuple[int, int, tuple[int, ...]]], reps: int) -> dict:
    """Time our canon_info_native on each sample; return per-call medians + aut orders."""
    import doubly_even_kernel as kern

    medians_us: list[float] = []
    aut_orders: list[int] = []
    for n, k, rref in samples:
        rref_list = list(rref)
        # warmup
        kern.canon_info_native(rref_list, n)
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            _, _, gs1, gs2, _ = kern.canon_info_native(rref_list, n)
            times.append((time.perf_counter() - t) * 1e6)
        medians_us.append(statistics.median(times))
        # Aut order from grpsize1 * 10^grpsize2 — exact for small enough values
        # (we don't need it exact, just consistent for the sanity check)
        aut_orders.append(int(round(gs1 * (10 ** gs2))))
    return {
        "medians_us": medians_us,
        "aut_orders": aut_orders,
        "total_seconds": sum(medians_us) / 1e6,
        "median_us": statistics.median(medians_us) if medians_us else 0.0,
        "mean_us": statistics.mean(medians_us) if medians_us else 0.0,
    }


def run_sage_child(N: int, samples: list[tuple[int, int, tuple[int, ...]]],
                   reps: int, backends: list[str], timeout_s: float) -> dict:
    """Run the Sage child script in a subprocess; return its JSON result."""
    payload = {
        "N": N,
        "reps": reps,
        "backends": backends,
        "samples": [
            {"n": n, "k": k, "rref": list(rref)} for (n, k, rref) in samples
        ],
    }
    proc = subprocess.run(
        [SAGE, str(CHILD)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        return {"error": "sage_failed", "stderr": proc.stderr[-2000:]}
    # Child writes JSON on the LAST line of stdout (preceded by progress lines).
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", default="14,16,18,20,22",
                    help="comma-separated list of N (default 14,16,18,20,22)")
    ap.add_argument("--sample-per-k", type=int, default=20,
                    help="max codes sampled per (N, k) cell (default 20)")
    ap.add_argument("--reps", type=int, default=3,
                    help="reps per code for median (default 3)")
    ap.add_argument("--label", default="sage-percall",
                    help="output JSON label (default 'sage-percall')")
    ap.add_argument("--backends", default="partition,codecan,low_level",
                    help="comma-separated Sage backends to bench "
                         "(partition,codecan,low_level)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-N Sage subprocess timeout in seconds")
    args = ap.parse_args()

    Ns = [int(s) for s in args.N.split(",") if s.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    all_results: dict[int, dict] = {}
    for N in Ns:
        print(f"\n=== N={N} ===", flush=True)
        samples = sample_codes(N, args.sample_per_k)
        print(f"  sampled {len(samples)} codes "
              f"(by k: {[len([s for s in samples if s[1]==k]) for k in sorted(set(s[1] for s in samples))]})",
              flush=True)
        if not samples:
            all_results[N] = {"error": "no samples"}
            continue

        # Time ours
        print(f"  timing ours ({args.reps} reps each)…", flush=True)
        ours = time_ours(samples, args.reps)
        print(f"    median {ours['median_us']:.1f} µs/call, "
              f"total {ours['total_seconds']*1000:.1f} ms over {len(samples)} codes",
              flush=True)

        # Time Sage
        print(f"  spawning Sage…", flush=True)
        sage = run_sage_child(N, samples, args.reps, backends, args.timeout)
        if "error" in sage:
            print(f"    SAGE ERROR: {sage['error']}", flush=True)
            print(sage.get("stderr", ""), flush=True)
            all_results[N] = {"ours": ours, "sage": sage}
            continue

        # Sanity: |Aut| must match
        sage_auts = sage.get("aut_orders", {})
        mismatches = []
        for backend, orders in sage_auts.items():
            for i, (ours_a, sage_a) in enumerate(zip(ours["aut_orders"], orders)):
                if int(ours_a) != int(sage_a):
                    mismatches.append((backend, i, ours_a, sage_a))
        if mismatches:
            print(f"  WARNING: |Aut| mismatches: {len(mismatches)} (showing first 3)",
                  flush=True)
            for m in mismatches[:3]:
                print(f"    backend={m[0]} sample[{m[1]}]: ours={m[2]} sage={m[3]}",
                      flush=True)

        print(f"  Sage medians:", flush=True)
        for backend, b_res in sage.get("backends", {}).items():
            print(f"    {backend:>12}: median {b_res['median_us']:.1f} µs/call, "
                  f"total {b_res['total_ms']:.1f} ms", flush=True)

        all_results[N] = {"ours": ours, "sage": sage, "mismatches": mismatches}

    # Write JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = HERE / "bench-results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{ts}-{args.label}.json"

    # Slim payload for disk (don't dump per-sample arrays)
    slim: dict[str, dict] = {}
    for N, r in all_results.items():
        if "error" in r:
            slim[str(N)] = r
            continue
        ours = r["ours"]
        sage = r["sage"]
        slim[str(N)] = {
            "n_samples": len(ours["medians_us"]),
            "ours_median_us": ours["median_us"],
            "ours_mean_us": ours["mean_us"],
            "ours_total_ms": ours["total_seconds"] * 1000,
            "backends": {
                b: {
                    "median_us": v["median_us"],
                    "mean_us": v["mean_us"],
                    "total_ms": v["total_ms"],
                    "ratio_vs_ours": v["median_us"] / max(ours["median_us"], 1e-9),
                }
                for b, v in sage.get("backends", {}).items()
            },
            "n_mismatches": len(r.get("mismatches", [])),
        }

    payload = {
        "label": args.label,
        "timestamp_utc": ts,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config": {
            "sample_per_k": args.sample_per_k,
            "reps": args.reps,
            "backends": backends,
        },
        "per_N": slim,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}", flush=True)

    # Final table
    print(f"\n{'N':>3} {'samples':>7} {'ours µs':>9}", end="")
    for b in backends:
        print(f"  {b[:10]:>10} {'ratio':>6}", end="")
    print()
    for N in Ns:
        if "error" in all_results[N]:
            print(f"{N:>3}  ERROR", flush=True)
            continue
        s = slim[str(N)]
        print(f"{N:>3} {s['n_samples']:>7} {s['ours_median_us']:>9.1f}", end="")
        for b in backends:
            bd = s["backends"].get(b, {})
            print(f"  {bd.get('median_us', 0):>10.1f} {bd.get('ratio_vs_ours', 0):>6.1f}x",
                  end="")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
