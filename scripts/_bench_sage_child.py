"""Sage-side benchmark child. Reads JSON from stdin, writes JSON on last stdout line.

Invoked by bench_sage.py via:
    sage scripts/_bench_sage_child.py < payload.json

Payload schema:
    {"N": int, "reps": int, "backends": ["partition","codecan","low_level"],
     "samples": [{"n": int, "k": int, "rref": [int...]}, ...]}

The rref rows are encoded as Python ints where bit i is column i (same convention
as our BinVec). The child decodes them into GF(2) matrix rows.

Output: JSON on the LAST line of stdout (progress lines may precede it).
    {"backends": {<name>: {"median_us":..., "mean_us":..., "total_ms":...,
                            "per_sample_us":[...]}}, "aut_orders": {<name>: [...]}}
"""
from __future__ import annotations

import json
import statistics
import sys
import time

from sage.all import GF, Matrix
from sage.coding.linear_code import LinearCode
from sage.coding.codecan.codecan import PartitionRefinementLinearCode
from sage.groups.perm_gps.partn_ref.refinement_binary import LinearBinaryCodeStruct


def rref_to_matrix(rref: list, n: int):
    """Decode our int-bitmask rows into a Sage GF(2) matrix."""
    rows = []
    for word in rref:
        rows.append([(word >> i) & 1 for i in range(n)])
    return Matrix(GF(2), rows)


def time_one(call_fn, reps: int) -> tuple[float, list[float]]:
    """Run call_fn() reps times; return (median_us, all_us)."""
    times_us = []
    for _ in range(reps):
        t = time.perf_counter()
        call_fn()
        times_us.append((time.perf_counter() - t) * 1e6)
    return statistics.median(times_us), times_us


def main() -> None:
    payload = json.loads(sys.stdin.read())
    N = payload["N"]
    reps = payload["reps"]
    backends = payload["backends"]
    samples = payload["samples"]

    # Pre-build matrices once (matrix construction itself is not what we're timing;
    # we want the canonicaliser kernel cost).
    matrices = [rref_to_matrix(s["rref"], s["n"]) for s in samples]

    # Warmup: ensure all Sage modules are JIT-imported
    if matrices:
        G = matrices[0]
        n0 = samples[0]["n"]
        LinearCode(G).canonical_representative(equivalence="permutation")
        LinearCode(G).permutation_automorphism_group(algorithm="partition").order()
        PartitionRefinementLinearCode(n0, G).get_canonical_form()
        S = LinearBinaryCodeStruct(G); S.automorphism_group(); S.canonical_relabeling()

    print(f"[child] warmed up; N={N} samples={len(samples)} reps={reps}", flush=True)

    results: dict = {"backends": {}, "aut_orders": {}}

    for backend in backends:
        per_sample_us: list[float] = []
        aut_orders: list[int] = []

        for sample, G in zip(samples, matrices):
            n = sample["n"]
            if backend == "partition":
                def call():
                    C = LinearCode(G)
                    grp = C.permutation_automorphism_group(algorithm="partition")
                    return grp.order()
                # Get aut order on a one-off call (outside timed loop is fine —
                # we re-do the same work inside call() anyway).
                med, _ = time_one(call, reps)
                C = LinearCode(G)
                ao = int(C.permutation_automorphism_group(algorithm="partition").order())
            elif backend == "codecan":
                def call():
                    C = LinearCode(G)
                    return C.canonical_representative(equivalence="permutation")
                med, _ = time_one(call, reps)
                C = LinearCode(G)
                # codecan path also gives aut order via the inner_stabilizer call:
                # but easier: use partition for the sanity check
                ao = int(C.permutation_automorphism_group(algorithm="partition").order())
            elif backend == "low_level":
                def call():
                    P = PartitionRefinementLinearCode(n, G)
                    cf = P.get_canonical_form()
                    return cf
                med, _ = time_one(call, reps)
                P = PartitionRefinementLinearCode(n, G)
                P.get_canonical_form()
                ao = int(P.get_autom_order_permutation())
            elif backend == "raw_partn_ref":
                # Cython LinearBinaryCodeStruct directly — skips LinearCode
                # and PermutationGroup wrapping. This is the most apples-to-
                # apples comparison against our nauty Rust wrapper:
                # both are thin native wrappers around a partition-
                # refinement canonicaliser, return (gens, order, base) plus
                # canonical relabeling in two Cython/Rust calls.
                def call():
                    S = LinearBinaryCodeStruct(G)
                    gens, order, _base = S.automorphism_group()
                    relabel = S.canonical_relabeling()
                    return (gens, order, relabel)
                med, _ = time_one(call, reps)
                S = LinearBinaryCodeStruct(G)
                _gens, order, _ = S.automorphism_group()
                ao = int(order)
            else:
                raise ValueError(f"unknown backend {backend!r}")

            per_sample_us.append(med)
            aut_orders.append(ao)

        results["backends"][backend] = {
            "per_sample_us": per_sample_us,
            "median_us": statistics.median(per_sample_us) if per_sample_us else 0.0,
            "mean_us": statistics.mean(per_sample_us) if per_sample_us else 0.0,
            "total_ms": sum(per_sample_us) / 1000.0,
        }
        results["aut_orders"][backend] = aut_orders

        print(f"[child] {backend}: median {results['backends'][backend]['median_us']:.1f} µs/call",
              flush=True)

    # JSON on the last line for parent to parse
    print(json.dumps(results), flush=True)


# Sage runs us via eval(compile(...)) without setting __name__ == "__main__",
# so call main() unconditionally instead of guarding.
main()
