"""1-Weisfeiler-Leman collision audit for doubly-even codes.

Scientific question: does the combined `wl_{min,full} + T12 + T13` invariant
remain complete (zero collisions among inequivalent codes) as N grows past
22? The Python prototype showed 0 collisions at N=22 on 5118 codes; this
script ports the per-code primitives to Rust (via the
``doubly_even_kernel.all_invariants`` entry point) so the experiment is
tractable at N=24 (37,496 classes) and N=26 (494,272 classes).

Builds the bipartite ``(|codewords|, N)`` graph in two flavours:

  G_min       — span-aware low-weight subset (matches the D10 Q_D-graph
                canonicaliser at ``rust/src/canon.rs:432``; falls back to
                full G(C) when the low-weight set exceeds 2^(k-1)).
  full G(C)   — every nonzero codeword (strictly more refining than G_min).

Runs 1-WL colour refinement under two initial partitions:

  vanilla       — codewords colour 0, columns colour 1.
  degree_init   — codewords by weight stratum, columns by incidence degree.

Plus three cheap orthogonal invariants (T11/T12/T13; see docstrings below).
The combined-invariant report groups every code by tuple of selected
component signatures and reports the resulting collision counts.

Backend selection:

  --backend rust     (default)  call doubly_even_kernel.all_invariants
  --backend python              pure-Python reference path (slow at N≥24)
  --backend both                run both and assert bucketing-equivalence

Bucketing-equivalence (NOT byte-equal): Python's ``hash`` is per-process
randomised and Rust uses deterministic mix64, so the two backends emit
different actual signature values. The cross-check is on the partition
``code_idx → bucket_id`` induced by each backend — codes that group
together under Python must group together under Rust, and vice versa.
Mismatches indicate a port bug.

Output JSON: ``scripts/bench-results/{ts}-wl-collision-experiment.json``.
The per-variant and combined entries gain a ``backend`` field (``rust`` |
``python``). Combined entries also gain ``colliding_bucket_indices`` — the
list of (code_idx, ...) tuples for each colliding bucket — used by
``scripts/inspect_collisions.py`` for the post-mortem dump.

Usage::

    uv run python scripts/wl_collision_experiment.py --N 12,14,16,18,20,22
    DOUBLY_EVEN_THREADS=20 DOUBLY_EVEN_FRONTIER_DEPTH=5 \\
        uv run python scripts/wl_collision_experiment.py \\
            --N 24 --cache-codes scripts/bench-results/codes-cache
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.enumerate.augment import enumerate_doubly_even  # noqa: E402
from doubly_even.spec.codes import Code  # noqa: E402

try:
    import doubly_even_kernel as _kernel
    _HAVE_KERNEL = hasattr(_kernel, "all_invariants")
except ImportError:  # pragma: no cover
    _kernel = None
    _HAVE_KERNEL = False


# ─────────────────────────────────────────────────────────────── data classes


@dataclass(frozen=True)
class CodeInfo:
    """Just enough about each code to compute signatures and report on
    collisions. Pickle-friendly (unlike EnumeratedCode + CanonInfo)."""
    rref: tuple[int, ...]
    rank: int
    aut_order: int
    n: int


# ──────────────────────────────────────────────────────── enumeration & cache


def _enumerate_to_code_infos(N: int) -> list[CodeInfo]:
    out: list[CodeInfo] = []
    for ec in enumerate_doubly_even(N):
        rows, _ = ec.code.rref_basis()
        out.append(
            CodeInfo(
                rref=tuple(rows),
                rank=ec.code.rank,
                aut_order=ec.aut_order,
                n=N,
            )
        )
    return out


def codes_for_N(
    N: int, cache_dir: Path | None
) -> tuple[list[CodeInfo], float]:
    """Return the canonical-class list for length ``N``.

    With ``--cache-codes PATH``: on first run pickle the list to
    ``PATH/codes-N{N}.pkl``; on subsequent runs reload from disk and skip
    the (potentially expensive) enumeration.  Cached files are small —
    only the RREF + Aut order + rank is stored, not the CanonInfo.
    """
    if cache_dir is not None:
        cache_path = cache_dir / f"codes-N{N}.pkl"
        if cache_path.exists():
            t0 = time.time()
            with open(cache_path, "rb") as f:
                codes = pickle.load(f)
            dt = time.time() - t0
            print(
                f"  loaded {len(codes)} classes from cache in {dt:.2f}s "
                f"({cache_path.name})",
                flush=True,
            )
            return codes, dt
    t0 = time.time()
    codes = _enumerate_to_code_infos(N)
    dt = time.time() - t0
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"codes-N{N}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(codes, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  cached → {cache_path}", flush=True)
    return codes, dt


# ────────────────────────────────────────────────────────── Python reference
#
# Mirrored verbatim from the pre-refactor version of this script — kept as
# the byte-for-byte reference path for ``--backend both`` bucketing checks.


def _rank_gf2(rows: list[int], n: int) -> int:
    work = list(rows)
    r = 0
    for c in range(n):
        pivot = -1
        for i in range(r, len(work)):
            if (work[i] >> c) & 1:
                pivot = i
                break
        if pivot == -1:
            continue
        work[r], work[pivot] = work[pivot], work[r]
        for i in range(r + 1, len(work)):
            if (work[i] >> c) & 1:
                work[i] ^= work[r]
        r += 1
    return r


def _full_codewords_from_rref(rref: tuple[int, ...]) -> list[int]:
    k = len(rref)
    if k == 0:
        return []
    out: list[int] = []
    w = 0
    for mask in range(1, 1 << k):
        lo = (mask & -mask).bit_length() - 1
        w ^= rref[lo]
        out.append(w)
    return out


def _low_weight_from_rref(
    rref: tuple[int, ...], n: int
) -> tuple[list[int], list[int]] | None:
    """Span-aware low-weight set; mirror of canon.rs:369-417."""
    k = len(rref)
    if k == 0:
        return None
    total = 1 << k
    bail = total // 2
    by_weight: list[list[int]] = [[] for _ in range(n + 1)]
    w = 0
    for mask in range(1, total):
        lo = (mask & -mask).bit_length() - 1
        w ^= rref[lo]
        by_weight[w.bit_count()].append(w)
    accum: list[int] = []
    strata: list[int] = []
    for weight in range(1, n + 1):
        if not by_weight[weight]:
            continue
        stratum = by_weight[weight]
        accum.extend(stratum)
        strata.append(len(stratum))
        if len(accum) > bail:
            return None
        if _rank_gf2(accum, n) == k:
            return accum, strata
    return None


def _all_codewords_by_stratum_from_rref(
    rref: tuple[int, ...], n: int
) -> tuple[list[int], list[int]]:
    k = len(rref)
    by_weight: list[list[int]] = [[] for _ in range(n + 1)]
    w = 0
    for mask in range(1, 1 << k):
        lo = (mask & -mask).bit_length() - 1
        w ^= rref[lo]
        by_weight[w.bit_count()].append(w)
    flat: list[int] = []
    strata: list[int] = []
    for weight in range(1, n + 1):
        if by_weight[weight]:
            flat.extend(by_weight[weight])
            strata.append(len(by_weight[weight]))
    return flat, strata


def _build_bipartite(
    cws: list[int], n: int
) -> tuple[list[list[int]], list[list[int]]]:
    cw_nbrs: list[list[int]] = []
    col_nbrs: list[list[int]] = [[] for _ in range(n)]
    for i, cw in enumerate(cws):
        nbrs: list[int] = []
        bits = cw
        while bits:
            j = (bits & -bits).bit_length() - 1
            nbrs.append(j)
            col_nbrs[j].append(i)
            bits &= bits - 1
        cw_nbrs.append(nbrs)
    return cw_nbrs, col_nbrs


def _wl_refine(
    cw_nbrs, col_nbrs, init_cw, init_col, max_rounds
):
    cw_colors = list(init_cw)
    col_colors = list(init_col)
    L = len(cw_colors)
    R = len(col_colors)
    prev_total = len(set(cw_colors)) + len(set(col_colors))
    for round_idx in range(max_rounds):
        new_cw = [
            hash((0, cw_colors[i],
                  tuple(sorted(col_colors[j] for j in cw_nbrs[i]))))
            for i in range(L)
        ]
        new_col = [
            hash((1, col_colors[j],
                  tuple(sorted(cw_colors[i] for i in col_nbrs[j]))))
            for j in range(R)
        ]
        cw_colors = new_cw
        col_colors = new_col
        total = len(set(cw_colors)) + len(set(col_colors))
        if total == prev_total:
            return cw_colors, col_colors, round_idx + 1
        prev_total = total
    return cw_colors, col_colors, max_rounds


def _wl_signature(cw_colors, col_colors) -> tuple:
    return (tuple(sorted(cw_colors)), tuple(sorted(col_colors)))


def _signatures_for_rref_python(
    rref: tuple[int, ...], n: int, variant: str
) -> tuple[tuple, int, bool, int]:
    if variant.startswith("full_"):
        cws, strata = _all_codewords_by_stratum_from_rref(rref, n)
        fallback = False
        init_kind = variant[len("full_"):]
    else:
        low = _low_weight_from_rref(rref, n)
        if low is None:
            cws, strata = _all_codewords_by_stratum_from_rref(rref, n)
            fallback = True
        else:
            cws, strata = low
            fallback = False
        init_kind = variant
    L = len(cws)
    if L == 0:
        return (((), (n,)), 0, fallback, 0)
    cw_nbrs, col_nbrs = _build_bipartite(cws, n)
    if init_kind == "vanilla":
        init_cw = [0] * L
        init_col = [1] * n
    elif init_kind == "degree_init":
        init_cw = []
        for stratum_idx, size in enumerate(strata):
            init_cw.extend([stratum_idx] * size)
        col_offset = len(strata)
        init_col = [col_offset + len(nbrs) for nbrs in col_nbrs]
    else:
        raise ValueError(f"unknown init {init_kind!r}")
    final_cw, final_col, rounds = _wl_refine(
        cw_nbrs, col_nbrs, init_cw, init_col, max_rounds=2 * (L + n)
    )
    return _wl_signature(final_cw, final_col), rounds, fallback, L


def _t11_python(
    cws: list[int], n: int, weights: tuple[int, ...] = (4, 8, 12, 16)
) -> tuple:
    per_col: list[tuple[int, ...]] = []
    for j in range(n):
        bit = 1 << j
        through = [c for c in cws if c & bit]
        per_col.append(
            tuple(sum(1 for c in through if c.bit_count() == w) for w in weights)
        )
    return tuple(sorted(per_col))


def _t13_python(
    cws: list[int], n: int, weights: tuple[int, ...] = (4, 8, 12, 16)
) -> tuple:
    L = len(cws)
    if L == 0:
        return ()
    col_mask = [0] * n
    for k, c in enumerate(cws):
        bits = c
        while bits:
            j = (bits & -bits).bit_length() - 1
            col_mask[j] |= 1 << k
            bits &= bits - 1
    weight_mask = [0] * len(weights)
    weight_index = {w: idx for idx, w in enumerate(weights)}
    for k, c in enumerate(cws):
        idx = weight_index.get(c.bit_count())
        if idx is not None:
            weight_mask[idx] |= 1 << k
    per_pair: list[tuple] = []
    for i in range(n):
        mi = col_mask[i]
        for j in range(i + 1, n):
            mij = mi & col_mask[j]
            per_pair.append(
                tuple((mij & weight_mask[idx]).bit_count() for idx in range(len(weights)))
            )
    return tuple(sorted(per_pair))


def _t12_python(cws: list[int]) -> tuple:
    if not cws:
        return ()
    nonzero = [c for c in cws if c]
    if not nonzero:
        return ()
    m = min(c.bit_count() for c in nonzero)
    V = [c for c in cws if c.bit_count() == m]
    nV = len(V)
    out: list[tuple[int, int, int]] = []
    for i in range(nV):
        for j in range(i + 1, nV):
            for k in range(j + 1, nV):
                a, b, c = V[i], V[j], V[k]
                trip = tuple(sorted([
                    (a ^ b).bit_count(),
                    (a ^ c).bit_count(),
                    (b ^ c).bit_count(),
                ]))
                out.append(trip)
    out.sort()
    return tuple(out)


VARIANTS = ("vanilla", "degree_init", "full_vanilla", "full_degree_init")


# ─────────────────────────────────────────────── per-backend signature batches


@dataclass
class SignatureBatch:
    """Per-variant lists of signatures for all codes. Keys = component name."""
    sigs: dict[str, list]
    backend: str
    extras_seconds: float
    variant_seconds: dict[str, float]
    variant_rounds: dict[str, list[int]]
    variant_L: dict[str, list[int]]
    variant_fallbacks: dict[str, int]
    # Cumulative per-component wall-time nanoseconds (Rust backend only;
    # Python leaves this empty). Index order matches the kernel:
    #   0 wl_min_vanilla, 1 wl_min_degree, 2 wl_full_vanilla,
    #   3 wl_full_degree, 4 t11_full, 5 t11_gmin, 6 t12, 7 t13
    component_nanos: list[int] | None = None


def _compute_signatures_python(
    code_infos: list[CodeInfo], N: int
) -> SignatureBatch:
    weights = (4, 8, 12, 16)
    gmin_weights = tuple(range(4, N + 1, 4))

    sigs: dict[str, list] = {
        "t11_full": [],
        "t11_gmin": [],
        "t12": [],
        "t13": [],
        "t13_min": [],
        "vanilla": [],
        "degree_init": [],
        "full_vanilla": [],
        "full_degree_init": [],
    }

    variant_rounds: dict[str, list[int]] = {v: [] for v in VARIANTS}
    variant_L: dict[str, list[int]] = {v: [] for v in VARIANTS}
    variant_fallbacks: dict[str, int] = {v: 0 for v in VARIANTS}
    variant_seconds: dict[str, float] = {v: 0.0 for v in VARIANTS}

    t0 = time.time()
    for ci in code_infos:
        full_cws = _full_codewords_from_rref(ci.rref)
        sigs["t11_full"].append(_t11_python(full_cws, N, weights))
        sigs["t12"].append(_t12_python(full_cws))
        sigs["t13"].append(_t13_python(full_cws, N, weights))
        # T13 restricted to the minimum-weight stratum (single-weight slice).
        nonzero = [c for c in full_cws if c]
        min_w = min((c.bit_count() for c in nonzero), default=0)
        sigs["t13_min"].append(_t13_python(full_cws, N, (min_w,)))
        low = _low_weight_from_rref(ci.rref, N)
        gmin_cws = low[0] if low is not None else full_cws
        sigs["t11_gmin"].append(_t11_python(gmin_cws, N, gmin_weights))
    extras_seconds = time.time() - t0

    for variant in VARIANTS:
        t0 = time.time()
        for ci in code_infos:
            sig, rounds, fallback, L = _signatures_for_rref_python(
                ci.rref, N, variant
            )
            sigs[variant].append(sig)
            variant_rounds[variant].append(rounds)
            variant_L[variant].append(L)
            if fallback:
                variant_fallbacks[variant] += 1
        variant_seconds[variant] = time.time() - t0

    return SignatureBatch(
        sigs=sigs,
        backend="python",
        extras_seconds=extras_seconds,
        variant_seconds=variant_seconds,
        variant_rounds=variant_rounds,
        variant_L=variant_L,
        variant_fallbacks=variant_fallbacks,
    )


def _compute_signatures_rust(
    code_infos: list[CodeInfo], N: int
) -> SignatureBatch:
    if not _HAVE_KERNEL:
        raise RuntimeError(
            "Rust backend requested but doubly_even_kernel.all_invariants is "
            "unavailable. Rebuild with: maturin build --release --features "
            "parallel -m rust/Cargo.toml"
        )

    weights = [4, 8, 12, 16]
    gmin_weights = list(range(4, N + 1, 4))

    # Rust returns one 128-bit digest per code per component. Stored as
    # Python ints. These are directly hashable & comparable, so we don't
    # need tuple-conversion. Compact: 16 bytes/digest × 8 × 494K ≈ 63 MB
    # at N=26 (vs ~440 GB for full sorted Vec<u64> tuples).
    sigs: dict[str, list[int]] = {
        "t11_full": [],
        "t11_gmin": [],
        "t12": [],
        "t13": [],
        "t13_min": [],
        "vanilla": [],
        "degree_init": [],
        "full_vanilla": [],
        "full_degree_init": [],
    }
    variant_rounds: dict[str, list[int]] = {v: [] for v in VARIANTS}
    variant_L: dict[str, list[int]] = {v: [] for v in VARIANTS}
    variant_fallbacks: dict[str, int] = {v: 0 for v in VARIANTS}

    variant_seconds = {v: 0.0 for v in VARIANTS}

    # Digest order from py_all_invariants:
    #   0 wl_min_vanilla, 1 wl_min_degree, 2 wl_full_vanilla,
    #   3 wl_full_degree, 4 t11_full, 5 t11_gmin, 6 t12, 7 t13, 8 t13_min
    component_nanos = [0] * 9
    t_total_start = time.time()
    for ci in code_infos:
        digests, fallback, meta, nanos = _kernel.all_invariants(
            list(ci.rref), N, weights, gmin_weights
        )
        sigs["vanilla"].append(digests[0])
        sigs["degree_init"].append(digests[1])
        sigs["full_vanilla"].append(digests[2])
        sigs["full_degree_init"].append(digests[3])
        sigs["t11_full"].append(digests[4])
        sigs["t11_gmin"].append(digests[5])
        sigs["t12"].append(digests[6])
        sigs["t13"].append(digests[7])
        sigs["t13_min"].append(digests[8])
        # Metadata layout: [r_mv, r_md, r_fv, r_fd, l_min, l_full]
        variant_rounds["vanilla"].append(meta[0])
        variant_rounds["degree_init"].append(meta[1])
        variant_rounds["full_vanilla"].append(meta[2])
        variant_rounds["full_degree_init"].append(meta[3])
        variant_L["vanilla"].append(meta[4])
        variant_L["degree_init"].append(meta[4])
        variant_L["full_vanilla"].append(meta[5])
        variant_L["full_degree_init"].append(meta[5])
        if fallback:
            variant_fallbacks["vanilla"] += 1
            variant_fallbacks["degree_init"] += 1
        # full_* variants never fall back (they always use full G(C))
        for i in range(9):
            component_nanos[i] += nanos[i]
    extras_seconds = time.time() - t_total_start

    return SignatureBatch(
        sigs=sigs,
        backend="rust",
        extras_seconds=extras_seconds,
        variant_seconds=variant_seconds,
        variant_rounds=variant_rounds,
        variant_L=variant_L,
        variant_fallbacks=variant_fallbacks,
        component_nanos=component_nanos,
    )


# ──────────────────────────────────── grouping & combined-invariant reporting


def _bucket_by_sig(sigs: list) -> dict[object, list[int]]:
    buckets: dict[object, list[int]] = defaultdict(list)
    for i, s in enumerate(sigs):
        buckets[s].append(i)
    return buckets


def _partitions_equivalent(sigs_a: list, sigs_b: list) -> tuple[bool, str]:
    """Bucket-equivalence check: ``sigs_a[i] == sigs_a[j]`` iff
    ``sigs_b[i] == sigs_b[j]`` for every pair (i, j)."""
    if len(sigs_a) != len(sigs_b):
        return False, f"length mismatch: {len(sigs_a)} vs {len(sigs_b)}"
    ids_a: dict[object, int] = {}
    ids_b: dict[object, int] = {}
    a_to_b: dict[int, int] = {}
    b_to_a: dict[int, int] = {}
    for i, (sa, sb) in enumerate(zip(sigs_a, sigs_b)):
        if sa not in ids_a:
            ids_a[sa] = len(ids_a)
        if sb not in ids_b:
            ids_b[sb] = len(ids_b)
        a, b = ids_a[sa], ids_b[sb]
        if a in a_to_b and a_to_b[a] != b:
            return (
                False,
                f"at i={i}: a-bucket {a} maps to both {a_to_b[a]} and {b}",
            )
        if b in b_to_a and b_to_a[b] != a:
            return (
                False,
                f"at i={i}: b-bucket {b} maps to both {b_to_a[b]} and {a}",
            )
        a_to_b[a] = b
        b_to_a[b] = a
    return True, "ok"


def _cross_check_backends(py: SignatureBatch, rust: SignatureBatch) -> dict:
    out: dict = {}
    for key in py.sigs:
        ok, msg = _partitions_equivalent(py.sigs[key], rust.sigs[key])
        out[key] = {"ok": ok, "msg": msg}
        marker = "OK" if ok else "FAIL"
        print(f"    bucket-equiv {key:20s}: {marker} ({msg})", flush=True)
        if not ok:
            raise AssertionError(
                f"Python ↔ Rust bucketing mismatch on '{key}': {msg}"
            )
    return out


# ───────────────────────────────────────────────────────────────── runner


COMBO_SPECS: list[tuple[str, tuple[str, ...]]] = [
    ("t11_gmin", ("t11_gmin",)),
    ("t11", ("t11_full",)),
    ("t12", ("t12",)),
    ("t13_pairgram", ("t13",)),
    ("t13_min", ("t13_min",)),
    ("t11+t12", ("t11_full", "t12")),
    ("t11+t13", ("t11_full", "t13")),
    ("t11+t12+t13", ("t11_full", "t12", "t13")),
    ("wl_min", ("degree_init",)),
    ("wl_full", ("full_degree_init",)),
    ("wl_min+t11+t12", ("degree_init", "t11_full", "t12")),
    ("wl_min+t13", ("degree_init", "t13")),
    ("wl_min+t13_min", ("degree_init", "t13_min")),
    ("wl_min+t11+t13", ("degree_init", "t11_full", "t13")),
    ("wl_min+t12+t13", ("degree_init", "t12", "t13")),
    ("wl_min+t12+t13_min", ("degree_init", "t12", "t13_min")),
    ("t13_min+t12", ("t13_min", "t12")),
    ("wl_min+all", ("degree_init", "t11_full", "t12", "t13")),
    ("wl_full+t12", ("full_degree_init", "t12")),
    ("wl_full+t13", ("full_degree_init", "t13")),
    ("wl_full+t12+t13", ("full_degree_init", "t12", "t13")),
    ("wl_full+all", ("full_degree_init", "t11_full", "t12", "t13")),
]


def run_for_N(
    N: int,
    backend: str,
    cache_dir: Path | None,
    save_collisions: bool = True,
) -> dict:
    print(f"\n=== N = {N} (backend={backend}) ===", flush=True)
    codes, t_enum = codes_for_N(N, cache_dir)
    print(f"  {len(codes)} classes (enum/load {t_enum:.2f}s)", flush=True)

    ranks = [ci.rank for ci in codes]
    out: dict = {
        "N": N,
        "backend": backend,
        "total_classes": len(codes),
        "enum_seconds": t_enum,
        "classes_by_k": dict(Counter(ranks)),
        "variants": {},
    }

    # Compute signature batches per backend.
    batches: dict[str, SignatureBatch] = {}

    if backend in ("python", "both"):
        t0 = time.time()
        py_batch = _compute_signatures_python(codes, N)
        t_py = time.time() - t0
        batches["python"] = py_batch
        print(
            f"  python signatures in {t_py:.2f}s "
            f"({(t_py / max(1, len(codes))) * 1e6:.1f} µs/code total)",
            flush=True,
        )

    if backend in ("rust", "both"):
        t0 = time.time()
        rust_batch = _compute_signatures_rust(codes, N)
        t_rust = time.time() - t0
        batches["rust"] = rust_batch
        print(
            f"  rust signatures in {t_rust:.2f}s "
            f"({(t_rust / max(1, len(codes))) * 1e6:.1f} µs/code total)",
            flush=True,
        )
        if rust_batch.component_nanos:
            comp_names = (
                "wl_min_vanilla",
                "wl_min_degree",
                "wl_full_vanilla",
                "wl_full_degree",
                "t11_full",
                "t11_gmin",
                "t12",
                "t13",
                "t13_min",
            )
            n_codes = max(1, len(codes))
            print("    per-component µs/code (Rust, excludes shared graph build):", flush=True)
            comp_us: dict[str, float] = {}
            for name, ns in zip(comp_names, rust_batch.component_nanos):
                us = ns / 1000.0 / n_codes
                comp_us[name] = us
                print(f"      {name:18s}: {us:7.2f} µs", flush=True)
            # Convenience aggregates the user asks about.
            wl_min = comp_us["wl_min_vanilla"] + comp_us["wl_min_degree"]
            wl_full = comp_us["wl_full_vanilla"] + comp_us["wl_full_degree"]
            print(
                f"      [combined]          wl_min (both inits): {wl_min:7.2f} µs, "
                f"wl_full (both inits): {wl_full:7.2f} µs",
                flush=True,
            )
            out["component_us_per_code"] = comp_us

    if backend == "both":
        print("  cross-check Python ↔ Rust bucketing:", flush=True)
        out["cross_check"] = _cross_check_backends(
            batches["python"], batches["rust"]
        )

    # Primary batch for the reporting below.
    primary: SignatureBatch = batches.get("rust") or batches["python"]

    # Per-variant report.
    for variant in VARIANTS:
        sigs_v = primary.sigs[variant]
        buckets = _bucket_by_sig(sigs_v)
        n_distinct = len(buckets)
        colliding = [v for v in buckets.values() if len(v) > 1]
        worst = max((len(v) for v in colliding), default=1)
        n_collisions = len(codes) - n_distinct
        n_unique_buckets = n_distinct - len(colliding)
        pct_unique = 100 * n_unique_buckets / max(1, len(codes))

        rounds = primary.variant_rounds[variant]
        Lvals = primary.variant_L[variant]
        avg_rounds = sum(rounds) / max(1, len(rounds))
        max_round = max(rounds, default=0)
        avg_L = sum(Lvals) / max(1, len(Lvals))
        max_L = max(Lvals, default=0)

        # Per-k breakdown.
        per_k: dict[str, dict] = {}
        codes_by_k: dict[int, list[int]] = defaultdict(list)
        for i, k in enumerate(ranks):
            codes_by_k[k].append(i)
        for k, idxs in sorted(codes_by_k.items()):
            sb: dict[object, list[int]] = defaultdict(list)
            for i in idxs:
                sb[sigs_v[i]].append(i)
            sb_coll = [v for v in sb.values() if len(v) > 1]
            per_k[str(k)] = {
                "total": len(idxs),
                "distinct": len(sb),
                "collisions": len(idxs) - len(sb),
                "worst": max((len(v) for v in sb_coll), default=1),
            }

        dt = primary.variant_seconds.get(variant, 0.0) or primary.extras_seconds
        us_per_code = (dt / max(1, len(codes))) * 1e6

        print(
            f"  {variant:18s}: dist={n_distinct:6d} coll={n_collisions:5d} "
            f"worst={worst:4d} uniq={n_unique_buckets:6d} "
            f"({pct_unique:5.1f}%) fb={primary.variant_fallbacks[variant]:4d} "
            f"avgL={avg_L:6.1f} maxL={max_L:5d} "
            f"rounds={avg_rounds:4.2f}/{max_round} "
            f"({us_per_code:5.1f} µs/code)",
            flush=True,
        )

        out["variants"][variant] = {
            "distinct_signatures": n_distinct,
            "collisions": n_collisions,
            "colliding_buckets": len(colliding),
            "worst_bucket": worst,
            "unique_buckets": n_unique_buckets,
            "unique_pct": pct_unique,
            "fallback_count": primary.variant_fallbacks[variant],
            "avg_L": avg_L,
            "max_L": max_L,
            "avg_rounds": avg_rounds,
            "max_rounds": max_round,
            "us_per_code": us_per_code,
            "sig_compute_seconds": dt,
            "per_k": per_k,
        }

    # Combined-invariant report (the headline metric).
    combined_out: dict[str, dict] = {}
    print("  combined invariants:", flush=True)
    for name, components in COMBO_SPECS:
        component_sigs = [primary.sigs[c] for c in components]
        cb: dict[tuple, list[int]] = defaultdict(list)
        for i in range(len(codes)):
            sig = tuple(sl[i] for sl in component_sigs)
            cb[sig].append(i)
        n_d = len(cb)
        col_b = [v for v in cb.values() if len(v) > 1]
        wb = max((len(v) for v in col_b), default=1)
        nc = len(codes) - n_d
        n_uniq = n_d - len(col_b)
        pct = 100 * n_uniq / max(1, len(codes))
        entry: dict = {
            "distinct_signatures": n_d,
            "collisions": nc,
            "colliding_buckets": len(col_b),
            "worst_bucket": wb,
            "unique_buckets": n_uniq,
            "unique_pct": pct,
        }
        if save_collisions and col_b:
            # List the code-index tuples for each collision bucket. Used by
            # scripts/inspect_collisions.py to dump per-bucket structural
            # details (|Aut|, RREF, weight enum).
            entry["colliding_bucket_indices"] = [
                sorted(v) for v in col_b
            ]
        combined_out[name] = entry
        marker = " *" if nc == 0 else ""
        print(
            f"    {name:18s}: dist={n_d:6d} coll={nc:5d} worst={wb:4d} "
            f"uniq={n_uniq:6d} ({pct:5.1f}%){marker}",
            flush=True,
        )
    out["combined"] = combined_out

    return out


# ─────────────────────────────────────────────────────────────────── main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=str, default="12,14,16,18,20,22")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--backend",
        choices=("rust", "python", "both"),
        default="rust",
        help="rust (default), python (slow reference), or both (cross-check)",
    )
    parser.add_argument(
        "--cache-codes",
        type=str,
        default=None,
        help="Directory for pickled CodeInfo lists. Re-runs skip enumeration.",
    )
    parser.add_argument(
        "--no-save-collisions",
        action="store_true",
        help="Skip saving colliding bucket indices in the JSON (slimmer output).",
    )
    args = parser.parse_args()

    Ns = [int(n) for n in args.N.split(",")]
    cache_dir = Path(args.cache_codes) if args.cache_codes else None

    results: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Ns": Ns,
        "backend": args.backend,
        "cache_codes_dir": str(cache_dir) if cache_dir else None,
        "results": [],
    }
    for N in Ns:
        results["results"].append(
            run_for_N(
                N,
                backend=args.backend,
                cache_dir=cache_dir,
                save_collisions=not args.no_save_collisions,
            )
        )

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = HERE / "bench-results"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"{ts}-wl-collision-experiment.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[saved {out_path}]", flush=True)


if __name__ == "__main__":
    # Ensure ``Code`` import is exercised so static checkers don't flag it.
    _ = Code
    main()
