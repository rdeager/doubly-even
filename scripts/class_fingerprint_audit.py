"""Audit + prototype: class-fingerprint cache for canonical augmentation.

Two modes:

* ``--mode audit`` (the original behaviour): instrument a Python-driven
  canonical-augmentation traversal to count ``class(canonical_parent(D))
  == class(C)`` per candidate. No behavioural change vs. baseline.
* ``--mode prototype``: actually *use* a class-fingerprint cache keyed
  by ``T11(D)`` to skip ``canon_info(D)`` calls on cheap-reject hits
  (where ``cached.parent_class_t11 != T11(C)``). Measures empirical
  Python wall speedup.
* ``--mode both`` (default): runs both back-to-back, reports the ratio.

Motivation (post-T11 retrospective in
`/home/dev/.claude/plans/last-time-we-profiled-floofy-lollipop.md`):
the T11 cache shipped dormant because storing σ per T11-hash is unsound
(σ is RREF-specific, not class-invariant). The candidate redesign caches
class-level info only — canonical form + |Aut| + parent class id. On a
cache hit, the **fast reject** path applies when D's canonical-parent
class is different from C's class; we instant-reject without computing
σ_D (no nauty call). Only when classes match do we fall back to nauty
to disambiguate the Aut(D)-orbit test.

The prototype is the empirical go/no-go gate for the Rust port — last
time we built first and audited never, costing ~1 day. This time the
Python prototype proves the win first.

Status notes
------------
* This is a **research/benchmark script**, NOT a reference implementation.
  The Python reference canonical-augmentation path remains in
  ``src/doubly_even/enumerate/augment.py::_traverse`` — unmodified, no
  class cache. The Rust kernel handles all production enumeration; this
  script's traversal is a Python-driven instrumentation for measurement.
* The T11 hash here is computed via ``doubly_even_kernel.compute_t11_hash``
  — a Rust call across the PyO3 FFI. There is **no pure-Python T11
  implementation in tree**. If we want either (a) a clean Python reference
  with the same optimisation, or (b) something upstreamable to SageMath,
  we need to add a pure-Python ``compute_t11_hash`` (≈30 LOC: enumerate
  codewords via Gray-code walk, accumulate per-column weight-bucket
  counts, sort per-column tuples to a multiset, hash deterministically).
  See ``rust/src/canon.rs::compute_t11_hash`` for the spec.
* To make the cache the Python default reference, port the prototype's
  ``traverse`` (with early populate on first canon_info using
  ``T11(canonical_parent(D))``) into ``augment.py::_traverse``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Python-driven traversal: we call `doubly_even_candidates`, `canon_info`,
# and `canonical_parent` directly so we can instrument each candidate.
# Each of those still dispatches to the Rust kernel internally (Rust does
# the actual nauty / σ_Q work); only the loop is in Python.

from doubly_even.canon.nauty import (  # noqa: E402
    canon_info,
    canon_info_cache_clear,
    canonical_form,
)
from doubly_even.enumerate.augment import (  # noqa: E402
    canonical_parent,
    is_canonical_augmentation,
)
from doubly_even.enumerate.filters import doubly_even_candidates  # noqa: E402
from doubly_even.spec.codes import Code  # noqa: E402
from doubly_even.spec.mass import gaborit_sigma  # noqa: E402

try:
    from doubly_even_kernel import compute_t11_hash as _t11_hash  # type: ignore
except ImportError as exc:
    sys.stderr.write(
        "ERROR: kernel does not export compute_t11_hash. Build with\n"
        "  maturin build --release --features t11_cache -m rust/Cargo.toml\n"
        "  uv pip install rust/target/wheels/doubly_even_kernel-*.whl --force-reinstall\n"
    )
    raise


# DFGHILM Table 3 totals (sum of equivalence classes at k ≥ 1; the
# trivial zero code at k=0 is excluded). Matches what the audit
# traversal counts via ``accepts_by_k``: each accept is a child code D
# at rank k+1, so the zero code is the root, not an accept.
# Verified by independent ``enumerate_doubly_even`` runs.
DFGHILM_ACCEPT_TOTALS = {
    12: 24,
    14: 49,
    16: 145,
    18: 340,
    20: 1210,
    22: 5117,
    24: 37495,
}


# ----------------------------------------------------------------- blocklist

_BLOCKLIST_BY_N: dict[int, frozenset[int]] = {}


# Empirically-discovered small-N collisions. The Rust `t11_blocklist_for_n`
# returns an empty set for 4 ≤ N ≤ 20 with a comment claiming "no collisions
# observed", but rerunning `dump_t11_blocklist.py` shows that's stale: there
# is 1 collision at N=16, 1 at N=18, and 19 at N=20 (all worst bucket = 2).
# These have been verified via `dump_t11_blocklist.py --N 16,18,20`.
# If the Rust source is regenerated to fix this, this Python dict can be
# replaced with regex parsing alone.
_SMALL_N_BLOCKLISTS: dict[int, tuple[int, ...]] = {
    4: (),
    8: (),
    14: (),
    16: (0xd4ce2a4c6231244c,),
    18: (0x08dc2b1849b7c7bc,),
    20: (
        0x02609515b0bb1e85, 0x0f030e7db349cf14,
        0x1e9344756a1bf4a5, 0x20175221680a016d,
        0x2485a7696bb5795a, 0x291016c08fa312d9,
        0x55a106de8738c991, 0x579c6ce7e840a3bd,
        0x79a51831e89f5f13, 0x88b175b28ea68313,
        0x8ad357c245419e55, 0x947c186fe8280fbc,
        0xa46af291c74326cf, 0xb8ee3cd0689781d1,
        0xd40c3a3c698d7e0b, 0xd9181c2f696f52f7,
        0xdae323dfc7c91d68, 0xe19c8f3a475c304f,
        0xf096b645d7f326fb,
    ),
}


def _load_blocklists() -> None:
    """Build per-N T11 blocklists.

    Sources, in priority order:
    1. ``_SMALL_N_BLOCKLISTS`` (hard-coded empirical hashes for N ≤ 20).
    2. ``rust/src/t11_blocklist.rs`` regex parse for N = 22 const array.
    3. ``rust/src/t11_blocklist_n24.in`` parse for N = 24.
    """
    for n, hashes in _SMALL_N_BLOCKLISTS.items():
        _BLOCKLIST_BY_N[n] = frozenset(hashes)
    rust_src = REPO_ROOT / "rust" / "src" / "t11_blocklist.rs"
    n24_src = REPO_ROOT / "rust" / "src" / "t11_blocklist_n24.in"
    hex_re = re.compile(r"0x([0-9a-fA-F]{16})")
    text = rust_src.read_text()
    for m in re.finditer(
        r"T11_BLOCKLIST_N(\d+):\s*&\[u64\]\s*=\s*&\[(.*?)\];",
        text,
        re.DOTALL,
    ):
        n = int(m.group(1))
        hashes = [int(h, 16) for h in hex_re.findall(m.group(2))]
        _BLOCKLIST_BY_N[n] = frozenset(hashes)
    if n24_src.exists():
        n24_text = n24_src.read_text()
        n24_hashes = [int(h, 16) for h in hex_re.findall(n24_text)]
        _BLOCKLIST_BY_N[24] = frozenset(n24_hashes)


_load_blocklists()


# ------------------------------------------------------------- cache structs


@dataclass(frozen=True)
class _ClassEntry:
    """One slot in the class-fingerprint cache.

    Keyed (externally) by the class invariant ``T11(D)``. ``parent_class_t11``
    is set at the moment ``D`` is first accepted as a canonical augmentation;
    by construction this is ``T11(C)`` where ``C`` is whichever parent
    accepted ``D``. Since ``class(canonical_parent(D))`` depends only on
    ``class(D)``, this value is class-invariant — multiple sibling RREFs of
    ``class(D)`` later look up the same entry and see the same parent class.

    ``canonical_form_rref`` is optional and populated only with
    ``--verify-collisions`` for catching inter-class hash collisions that
    aren't on the blocklist (i.e. correctness debugging).
    """

    parent_class_t11: int
    canonical_form_rref: tuple[int, ...] | None = None


# --------------------------------------------------------------- class IDs


def _canonical_form_rref(C: Code) -> tuple[int, ...]:
    """Class identifier: the RREF of C after applying its canonical σ."""
    if C.rank == 0:
        return ()
    return tuple(canonical_form(C).basis)


def _t11_hash_code(C: Code, n: int) -> int:
    """T11 hash of ``C``'s RREF basis. ``T11`` is permutation-invariant
    so this is a class invariant.
    """
    if C.rank == 0:
        return _t11_hash([], n)
    rref = list(C.rref_basis()[0])
    return _t11_hash(rref, n)


# ----------------------------------------------------------------- traversal


def audit(N: int, mode: str = "audit", *, max_k: int | None = None,
          verify_collisions: bool = False) -> dict:
    """Run a pure-Python canonical-augmentation traversal at length ``N``.

    Modes (each shares the same skeleton, differs only in instrumentation):

    * ``"audit"``: count match/mismatch per parent rank. Calls extra
      ``canonical_form`` work per candidate for ground-truth class IDs —
      this inflates baseline wall, so it's not the right comparison
      target for prototype speedup. Kept for backwards compatibility
      with the original audit-mode JSON shape.
    * ``"baseline"``: pure traversal (no instrumentation, no cache).
      The minimum-overhead Python loop; the right wall to compare
      ``prototype`` against.
    * ``"prototype"``: same skeleton as ``baseline`` plus the class
      fingerprint cache. Cheap-rejects skip ``canon_info(D)`` entirely.
    """
    cap = N // 2 if max_k is None else max_k
    quota = {k: gaborit_sigma(N, k) for k in range(cap + 1)}
    mass_at_k: dict[int, int] = dict.fromkeys(range(cap + 1), 0)

    import math

    factorial_N = math.factorial(N)

    match_by_k: Counter[int] = Counter()
    mismatch_by_k: Counter[int] = Counter()
    accepts_by_k: Counter[int] = Counter()  # subset of match_by_k

    classes_seen: set[tuple[int, ...]] = set()
    first_visits_by_k: Counter[int] = Counter()

    # Prototype-mode cache + counters.
    class_cache: dict[int, _ClassEntry] = {}
    cache_hit_match_by_k: Counter[int] = Counter()
    cache_hit_reject_by_k: Counter[int] = Counter()
    cache_miss_by_k: Counter[int] = Counter()
    blocklist_hits_by_k: Counter[int] = Counter()
    canon_info_calls = 0  # actual canon_info calls during this run

    blocklist = _BLOCKLIST_BY_N.get(N, frozenset())

    def traverse(C: Code, info_C, hash_C: int) -> None:
        nonlocal canon_info_calls

        k = C.rank
        mass_at_k[k] += factorial_N // info_C.aut_order
        if k >= cap:
            return
        if mass_at_k[k + 1] >= quota[k + 1]:
            return

        # The canonical-form ID for C; only computed in audit mode (it
        # itself triggers canon_info, which would pollute prototype timing).
        class_C_id = _canonical_form_rref(C) if mode == "audit" else None

        for v in doubly_even_candidates(C, info_C.aut_generators):
            if mass_at_k[k + 1] >= quota[k + 1]:
                return
            D = C.extend(v)

            # T11 hash of D — needed in both modes (audit uses for stats,
            # prototype for cache key). Cheap (~5 µs Rust, +Py FFI ovh).
            rref_D = list(D.rref_basis()[0])
            hash_D = _t11_hash(rref_D, N)

            # ----------------- PROTOTYPE FAST PATH -----------------
            if mode == "prototype":
                if hash_D in blocklist:
                    blocklist_hits_by_k[k] += 1
                    # Fall through to canon_info(D): collision-prone hash.
                else:
                    entry = class_cache.get(hash_D)
                    if entry is not None:
                        if entry.parent_class_t11 != hash_C:
                            # CHEAP REJECT: skip canon_info(D), skip
                            # canonical_parent, skip McKay test.
                            cache_hit_reject_by_k[k] += 1
                            continue
                        cache_hit_match_by_k[k] += 1
                        # Fall through; nauty still needed for Aut(D)-orbit.
                    else:
                        cache_miss_by_k[k] += 1

            # canon_info(D) — the expensive call we're trying to skip.
            info_D = canon_info(D)
            canon_info_calls += 1

            # Audit-mode bookkeeping (still useful in prototype mode for
            # the post-traversal correctness gate; we only skip the
            # canon-form computation, not the accept count).
            if mode == "audit":
                parent_D = canonical_parent(D, info_D)
                class_parent_D_id = _canonical_form_rref(parent_D)
                classes_match = class_parent_D_id == class_C_id
                if classes_match:
                    match_by_k[k] += 1
                else:
                    mismatch_by_k[k] += 1

                class_D_id = _canonical_form_rref(D)
                if class_D_id not in classes_seen:
                    classes_seen.add(class_D_id)
                    first_visits_by_k[k + 1] += 1

            # PROTOTYPE: populate the class cache on the FIRST canon_info
            # for this hash (regardless of McKay outcome). The
            # parent_class_t11 comes from canonical_parent(D), which is
            # cheap given info_D in hand. This earlier population means
            # subsequent rejects of class(D) hit the cache and cheap-
            # reject instead of falling through to another canon_info.
            if (
                mode == "prototype"
                and hash_D not in blocklist
                and hash_D not in class_cache
            ):
                parent_D = canonical_parent(D, info_D)
                parent_class_t11 = _t11_hash_code(parent_D, N)
                cf_rref = None
                if verify_collisions:
                    cf_rref = _canonical_form_rref(D)
                class_cache[hash_D] = _ClassEntry(
                    parent_class_t11=parent_class_t11,
                    canonical_form_rref=cf_rref,
                )

            if not is_canonical_augmentation(C, D, info_D=info_D):
                continue
            accepts_by_k[k] += 1

            if mode == "prototype" and verify_collisions:
                # Already populated above; double-check the canonical-form
                # invariant didn't drift across siblings (catches T11
                # collisions not covered by the blocklist).
                cached = class_cache.get(hash_D)
                if cached is not None and cached.canonical_form_rref:
                    cf_now = _canonical_form_rref(D)
                    if cached.canonical_form_rref != cf_now:
                        raise RuntimeError(
                            f"T11 collision at N={N} hash={hash_D:#x}: "
                            f"two distinct canonical forms map to same hash. "
                            f"Add to blocklist."
                        )

            traverse(D, info_D, hash_D)

    # Clear LRU caches in canon module between runs to ensure fair timing
    # comparison (the lru_cache on `_canon_info_by_rref` is process-global).
    canon_info_cache_clear()

    t0 = time.time()
    zero = Code.zero(N)
    info_zero = canon_info(zero)
    canon_info_calls += 1
    hash_zero = _t11_hash([], N)
    traverse(zero, info_zero, hash_zero)
    wall = time.time() - t0

    total_match = sum(match_by_k.values())
    total_mismatch = sum(mismatch_by_k.values())
    total = total_match + total_mismatch
    total_accepts = sum(accepts_by_k.values())
    total_first_visits = sum(first_visits_by_k.values())

    rec: dict = {
        "N": N,
        "max_k": cap,
        "mode": mode,
        "wall_seconds": wall,
        "accepts_by_k": dict(accepts_by_k),
        "total_accepts": total_accepts,
        "actual_canon_info_calls": canon_info_calls,
    }
    if mode == "audit":
        # Audit-mode metrics (unchanged shape from previous releases).
        projected_nauty = 2 * total_first_visits + total_match
        speedup_estimate = total / max(projected_nauty, 1)
        rec.update({
            "match_by_k": dict(match_by_k),
            "mismatch_by_k": dict(mismatch_by_k),
            "first_visits_by_k": dict(first_visits_by_k),
            "total_canon_calls": total,
            "total_match": total_match,
            "total_mismatch": total_mismatch,
            "total_first_visits": total_first_visits,
            "mismatch_fraction": total_mismatch / max(total, 1),
            "match_fraction": total_match / max(total, 1),
            "projected_nauty_under_class_cache": projected_nauty,
            "nauty_call_reduction_estimate": speedup_estimate,
        })
    else:
        rec.update({
            "cache_hit_match_by_k": dict(cache_hit_match_by_k),
            "cache_hit_reject_by_k": dict(cache_hit_reject_by_k),
            "cache_miss_by_k": dict(cache_miss_by_k),
            "blocklist_hits_by_k": dict(blocklist_hits_by_k),
            "cache_hits_match": sum(cache_hit_match_by_k.values()),
            "cache_hits_reject": sum(cache_hit_reject_by_k.values()),
            "cache_misses": sum(cache_miss_by_k.values()),
            "blocklist_hits": sum(blocklist_hits_by_k.values()),
            "class_cache_size": len(class_cache),
            "blocklist_size": len(blocklist),
        })

    expected = DFGHILM_ACCEPT_TOTALS.get(N)
    if expected is not None:
        rec["expected_accepts"] = expected
        rec["accepts_match_oracle"] = (total_accepts == expected)

    return rec


# ------------------------------------------------------------------- reporting


def _print_audit_summary(rec: dict) -> None:
    N = rec["N"]
    print(f"\n=== N = {N} [mode=audit] (max_k={rec['max_k']}) ===", flush=True)
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


def _print_baseline_summary(rec: dict) -> None:
    N = rec["N"]
    print(f"\n=== N = {N} [mode=baseline] ===", flush=True)
    print(
        f"  wall: {rec['wall_seconds']:.2f}s  "
        f"actual_canon_info_calls: {rec['actual_canon_info_calls']}  "
        f"accepts: {rec['total_accepts']}",
        flush=True,
    )
    if "expected_accepts" in rec:
        ok = "✓" if rec["accepts_match_oracle"] else "✗ MISMATCH"
        print(
            f"  DFGHILM oracle: {rec['total_accepts']} vs expected "
            f"{rec['expected_accepts']}  {ok}",
            flush=True,
        )


def _print_prototype_summary(rec: dict) -> None:
    N = rec["N"]
    print(f"\n=== N = {N} [mode=prototype] ===", flush=True)
    print(
        f"  wall: {rec['wall_seconds']:.2f}s  "
        f"actual_canon_info_calls: {rec['actual_canon_info_calls']}  "
        f"accepts: {rec['total_accepts']}",
        flush=True,
    )
    print(
        f"  cache misses (first visits):  {rec['cache_misses']:>7}",
        flush=True,
    )
    print(
        f"  cache hits — match (still nauty):  {rec['cache_hits_match']:>7}",
        flush=True,
    )
    print(
        f"  cache hits — REJECT (skipped nauty): {rec['cache_hits_reject']:>7}",
        flush=True,
    )
    print(
        f"  blocklist hits (forced canon):  {rec['blocklist_hits']:>7}  "
        f"(blocklist size = {rec['blocklist_size']})",
        flush=True,
    )
    print(
        f"  class_cache_size: {rec['class_cache_size']}",
        flush=True,
    )
    if "expected_accepts" in rec:
        ok = "✓" if rec["accepts_match_oracle"] else "✗ MISMATCH"
        print(
            f"  DFGHILM oracle: {rec['total_accepts']} vs expected "
            f"{rec['expected_accepts']}  {ok}",
            flush=True,
        )


def _print_pair_comparison(label_a: str, rec_a: dict,
                           label_b: str, rec_b: dict) -> None:
    a = rec_a["wall_seconds"]
    b = rec_b["wall_seconds"]
    speedup = a / b if b > 0 else float("inf")
    nauty_a = rec_a["actual_canon_info_calls"]
    nauty_b = rec_b["actual_canon_info_calls"]
    nauty_reduction = nauty_a / max(nauty_b, 1)
    print(
        f"\n  >>> wall speedup ({label_a} / {label_b}): {speedup:.2f}×  "
        f"({a:.2f}s → {b:.2f}s)",
        flush=True,
    )
    print(
        f"  >>> nauty-call reduction: {nauty_reduction:.2f}× "
        f"({nauty_a} → {nauty_b})",
        flush=True,
    )


# ----------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--N",
        type=str,
        default="18,20,22",
        help="Comma-separated lengths to audit (default: 18,20,22).",
    )
    parser.add_argument(
        "--mode",
        choices=["audit", "baseline", "prototype", "fair", "all"],
        default="fair",
        help=(
            "audit: measure-only (extra canonical_form calls — biased "
            "baseline). "
            "baseline: clean Python traversal, no cache, no metrics. "
            "prototype: clean traversal + class-fingerprint cache. "
            "fair (default): run baseline + prototype, report fair "
            "wall speedup. "
            "all: run audit + baseline + prototype."
        ),
    )
    parser.add_argument(
        "--verify-collisions",
        action="store_true",
        help=(
            "In prototype mode, also store canonical form RREF in each "
            "cache entry and verify on hit. Catches inter-class T11 "
            "collisions not covered by the blocklist."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="JSON output path; default writes under scripts/bench-results.",
    )
    args = parser.parse_args()
    Ns = [int(n) for n in args.N.split(",")]

    results: list[dict] = []
    comparisons: list[dict] = []

    run_audit = args.mode in ("audit", "all")
    run_baseline = args.mode in ("baseline", "fair", "all")
    run_prototype = args.mode in ("prototype", "fair", "all")

    for n in Ns:
        arec = brec = prec = None
        if run_audit:
            arec = audit(n, mode="audit")
            _print_audit_summary(arec)
            results.append(arec)
        if run_baseline:
            brec = audit(n, mode="baseline")
            _print_baseline_summary(brec)
            results.append(brec)
        if run_prototype:
            prec = audit(
                n,
                mode="prototype",
                verify_collisions=args.verify_collisions,
            )
            _print_prototype_summary(prec)
            results.append(prec)

        if brec is not None and prec is not None:
            _print_pair_comparison("baseline", brec, "prototype", prec)
        if arec is not None and prec is not None:
            _print_pair_comparison("audit", arec, "prototype", prec)

        if prec is not None:
            comp: dict = {
                "N": n,
                "prototype_wall": prec["wall_seconds"],
                "prototype_canon_calls": prec["actual_canon_info_calls"],
                "accepts_match_oracle": prec.get("accepts_match_oracle"),
            }
            if brec is not None:
                comp["baseline_wall"] = brec["wall_seconds"]
                comp["baseline_canon_calls"] = brec["actual_canon_info_calls"]
                comp["fair_wall_speedup"] = (
                    brec["wall_seconds"] / prec["wall_seconds"]
                    if prec["wall_seconds"] > 0 else None
                )
                comp["nauty_call_reduction"] = (
                    brec["actual_canon_info_calls"] /
                    max(prec["actual_canon_info_calls"], 1)
                )
            if arec is not None:
                comp["audit_wall"] = arec["wall_seconds"]
                comp["audit_canon_calls"] = arec.get("total_canon_calls", 0)
                comp["audit_vs_prototype_wall"] = (
                    arec["wall_seconds"] / prec["wall_seconds"]
                    if prec["wall_seconds"] > 0 else None
                )
            comparisons.append(comp)

    # Correctness summary across all Ns.
    any_oracle_failure = False
    for rec in results:
        if "accepts_match_oracle" in rec and not rec["accepts_match_oracle"]:
            any_oracle_failure = True
            sys.stderr.write(
                f"\nFAILED: N={rec['N']} mode={rec['mode']} accepts={rec['total_accepts']} "
                f"vs expected {rec['expected_accepts']}\n"
            )

    if comparisons:
        print("\n=== summary ===", flush=True)
        for c in comparisons:
            ok = ""
            if c["accepts_match_oracle"] is False:
                ok = " ✗ ORACLE MISMATCH"
            elif c["accepts_match_oracle"] is True:
                ok = " ✓"
            sp = c.get("fair_wall_speedup")
            nr = c.get("nauty_call_reduction")
            if sp is not None and nr is not None:
                print(
                    f"  N={c['N']:>2}  fair_wall_speedup={sp:.2f}×  "
                    f"nauty_reduction={nr:.2f}×{ok}",
                    flush=True,
                )
            else:
                print(
                    f"  N={c['N']:>2}  (no fair comparison; mode={args.mode}){ok}",
                    flush=True,
                )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Ns": Ns,
        "mode": args.mode,
        "verify_collisions": args.verify_collisions,
        "results": results,
        "comparisons": comparisons,
        "note": (
            "Class-fingerprint cache prototype + audit. The audit-mode "
            "records preserve the original mismatch-fraction shape; the "
            "prototype-mode records measure actual canon_info calls and "
            "wall speedup. DFGHILM_ACCEPT_TOTALS provides the oracle "
            "for canonical-augmentation correctness."
        ),
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = HERE / "bench-results"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"{ts}-class-fingerprint-prototype.json"

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[saved {out_path}]", flush=True)

    if any_oracle_failure:
        sys.exit(2)


if __name__ == "__main__":
    main()
