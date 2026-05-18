"""Audit: how often does ``class(canonical_parent(D)) == class(C)`` during enumeration?

Motivation (post-T11 retrospective in
`/home/dev/.claude/plans/last-time-we-profiled-floofy-lollipop.md`):
the T11 cache shipped dormant because storing σ per T11-hash is unsound
(σ is RREF-specific, not class-invariant). The candidate redesign caches
class-level info only — canonical form + |Aut| + parent class id. On a
cache hit, the **fast reject** path applies when D's canonical-parent
class is different from C's class; we instant-reject without computing
σ_D (no nauty call). Only when classes match do we fall back to nauty
to disambiguate the Aut(D)-orbit test.

This audit measures the fraction of candidate visits (post-σ_Q +
weight-enum prefilters) for which ``class(canonical_parent(D)) ==
class(C)``. Lower fraction → larger speedup potential.

Estimated wall-time savings at N=22 (current 88% of wall is nauty,
~82K canon_info calls):
  mismatch fraction = X  →  ~X · 82K nauty calls skipped
  X = 90 %  →  ~5× wall speedup
  X = 70 %  →  ~3× wall speedup
  X = 30 %  →  ~1.4× wall speedup

Python-only re-implementation; slow at N ≥ 20. Run at small N as a
proxy; trust the trend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Python-driven traversal: we call `doubly_even_candidates`, `canon_info`,
# and `canonical_parent` directly so we can instrument each candidate.
# Each of those still dispatches to the Rust kernel internally (Rust does
# the actual nauty / σ_Q work); only the loop is in Python.

from doubly_even.canon.nauty import canon_info, canonical_form  # noqa: E402
from doubly_even.enumerate.augment import (  # noqa: E402
    canonical_parent,
    is_canonical_augmentation,
)
from doubly_even.enumerate.filters import doubly_even_candidates  # noqa: E402
from doubly_even.spec.codes import Code  # noqa: E402
from doubly_even.spec.mass import gaborit_sigma  # noqa: E402


def _canonical_form_rref(C: Code) -> tuple[int, ...]:
    """Class identifier: the RREF of C after applying its canonical σ."""
    if C.rank == 0:
        return ()
    return tuple(canonical_form(C).basis)


def audit(N: int, max_k: int | None = None) -> dict:
    """Run a pure-Python canonical-augmentation traversal at length ``N``,
    tallying ``class(canonical_parent(D)) == class(C)`` per parent rank.
    """
    cap = N // 2 if max_k is None else max_k
    quota = {k: gaborit_sigma(N, k) for k in range(cap + 1)}
    mass_at_k: dict[int, int] = dict.fromkeys(range(cap + 1), 0)

    import math

    factorial_N = math.factorial(N)

    match_by_k: Counter[int] = Counter()
    mismatch_by_k: Counter[int] = Counter()
    accepts_by_k: Counter[int] = Counter()  # subset of match_by_k

    # Cache: class_id -> True (just to detect "first-time class visits").
    classes_seen: set[tuple[int, ...]] = set()

    # Count "first-time class first-time-pop" — proxy for cache misses.
    first_visits_by_k: Counter[int] = Counter()

    def traverse(C: Code, info_C):
        k = C.rank
        mass_at_k[k] += factorial_N // info_C.aut_order
        if k >= cap:
            return
        if mass_at_k[k + 1] >= quota[k + 1]:
            return

        class_C_id = _canonical_form_rref(C)

        for v in doubly_even_candidates(C, info_C.aut_generators):
            if mass_at_k[k + 1] >= quota[k + 1]:
                return
            D = C.extend(v)
            # This is the candidate that would reach canon_info(D) in the
            # Rust kernel. Compute its canon info (= nauty call) so we can
            # ask about class equality.
            info_D = canon_info(D)
            parent_D = canonical_parent(D, info_D)
            class_parent_D_id = _canonical_form_rref(parent_D)

            classes_match = class_parent_D_id == class_C_id
            if classes_match:
                match_by_k[k] += 1
            else:
                mismatch_by_k[k] += 1

            # First-time-class measurement: track D's class.
            class_D_id = _canonical_form_rref(D)
            if class_D_id not in classes_seen:
                classes_seen.add(class_D_id)
                first_visits_by_k[k + 1] += 1

            if not is_canonical_augmentation(C, D, info_D=info_D):
                continue
            accepts_by_k[k] += 1
            traverse(D, info_D)

    t0 = time.time()
    zero = Code.zero(N)
    traverse(zero, canon_info(zero))
    wall = time.time() - t0

    total_match = sum(match_by_k.values())
    total_mismatch = sum(mismatch_by_k.values())
    total = total_match + total_mismatch
    total_accepts = sum(accepts_by_k.values())
    total_first_visits = sum(first_visits_by_k.values())

    # Projected nauty calls under the class-fingerprint cache:
    #   - one canon_info per first-class-visit (to populate cache) plus
    #     one canon_info for that class's parent canonical form
    #     (~= 2 × first_visits, ignoring overlap when parent already cached)
    #   - one canon_info per match-class candidate (need σ for Aut(D)-
    #     orbit disambiguation)
    #   - zero per mismatch-class candidate (cheap reject)
    projected_nauty = 2 * total_first_visits + total_match
    speedup_estimate = total / max(projected_nauty, 1)

    return {
        "N": N,
        "max_k": cap,
        "wall_seconds": wall,
        "match_by_k": dict(match_by_k),
        "mismatch_by_k": dict(mismatch_by_k),
        "accepts_by_k": dict(accepts_by_k),
        "first_visits_by_k": dict(first_visits_by_k),
        "total_canon_calls": total,
        "total_match": total_match,
        "total_mismatch": total_mismatch,
        "total_accepts": total_accepts,
        "total_first_visits": total_first_visits,
        "mismatch_fraction": total_mismatch / max(total, 1),
        "match_fraction": total_match / max(total, 1),
        "projected_nauty_under_class_cache": projected_nauty,
        "nauty_call_reduction_estimate": speedup_estimate,
    }


def print_summary(rec: dict) -> None:
    N = rec["N"]
    print(f"\n=== N = {N} (max_k={rec['max_k']}) ===", flush=True)
    print(
        f"  wall: {rec['wall_seconds']:.2f}s  total_candidates: "
        f"{rec['total_canon_calls']}  accepts: {rec['total_accepts']}  "
        f"first_visits: {rec['total_first_visits']}",
        flush=True,
    )
    print(
        f"  parent-class match:    {rec['total_match']:>7} "
        f"({100*rec['match_fraction']:5.1f}%)  ← still need nauty",
        flush=True,
    )
    print(
        f"  parent-class mismatch: {rec['total_mismatch']:>7} "
        f"({100*rec['mismatch_fraction']:5.1f}%)  ← cheap-reject candidates",
        flush=True,
    )
    print(
        f"  projected nauty calls under class-cache: "
        f"{rec['projected_nauty_under_class_cache']} "
        f"(vs {rec['total_canon_calls']} today)",
        flush=True,
    )
    print(
        f"  estimated nauty-call reduction: "
        f"{rec['nauty_call_reduction_estimate']:.2f}×",
        flush=True,
    )
    print("  per-k breakdown (parent rank k → match/mismatch):")
    ks = sorted(set(rec["match_by_k"]) | set(rec["mismatch_by_k"]))
    for k in ks:
        m = rec["match_by_k"].get(k, 0)
        mm = rec["mismatch_by_k"].get(k, 0)
        a = rec["accepts_by_k"].get(k, 0)
        tot = m + mm
        frac = (100 * mm / tot) if tot else 0
        print(
            f"    k={k}: match={m:>6}  mismatch={mm:>6}  "
            f"({frac:5.1f}% mismatch)  accepts={a}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--N",
        type=str,
        default="12,14,16,18",
        help="Comma-separated lengths to audit (default: 12,14,16,18).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="JSON output path; default writes under scripts/bench-results.",
    )
    args = parser.parse_args()
    Ns = [int(n) for n in args.N.split(",")]

    results = []
    for n in Ns:
        rec = audit(n)
        print_summary(rec)
        results.append(rec)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Ns": Ns,
        "results": results,
        "note": (
            "Phase 0 audit for the class-fingerprint cache proposal — "
            "measures the fraction of canon_info candidates that have "
            "class(canonical_parent(D)) != class(C). High mismatch rate "
            "→ class-fingerprint cache can reject cheaply, large speedup."
        ),
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = HERE / "bench-results"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"{ts}-class-fingerprint-audit.json"

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[saved {out_path}]", flush=True)


if __name__ == "__main__":
    main()
