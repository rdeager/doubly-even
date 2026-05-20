"""Direct-sum seed table for mass-seeded canonical augmentation.

Phase 1 of the gluing-based mass-seeding plan. For each rank ``k`` of a
target length ``N``, compute the set of doubly-even codes reachable as
direct sums ``C1 ⊕ C2`` of strictly-smaller doubly-even codes. Each
seeded code's mass ``N! / |Aut|`` will be pre-credited to the
``mass_at_k[k]`` accumulator before the canonical-augmentation BFS
starts, so mass-stop fires sooner.

The dedup is exact: every seeded code is canonicalised via the active
canonicaliser (``canon_info`` → Rust kernel by default) and indexed by
its canonical RREF. Cross-split equivalences (e.g.,
``(A ⊕ B) ⊕ C ≡ A ⊕ (B ⊕ C)``) are handled by this dedup rather than
by a Krull–Schmidt pre-pass.

See ``/home/dev/.claude/plans/we-have-run-out-curious-bentley.md`` and
``markdown/architecture/04-optimisations.md`` "Audit — gluing-based
mass-seeding (Phase 0)" for the expected coverage and weighted-savings
upper bounds.

**Status**: not yet wired into the active enumeration. Consumed only
by :mod:`tests.test_seeds`. Phase 2 (gluing-based mass-stop credit)
is the unblocking step.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ...canon.nauty import canon_info, canonical_form
from ...spec.codes import Code
from ...spec.experimental.direct_sum import direct_sum
from ..augment import enumerate_doubly_even


@dataclass(frozen=True)
class SeededCode:
    """A pre-canonicalised seed code, ready to be credited at rank `k`.

    `canonical` is the canonical-RREF representative (the value used
    for set-membership tests against codes the BFS later generates).
    `aut_order` is `|Aut(canonical)|` from the canonicaliser.
    """

    canonical: Code
    aut_order: int

    @property
    def k(self) -> int:
        return self.canonical.rank


def _build_enum_table(
    n_max: int,
) -> dict[tuple[int, int], list[Code]]:
    """Build `{(n, k): [canonical reps]}` for `n ≤ n_max`.

    Empty code at `(0, 0)` included explicitly because
    ``enumerate_doubly_even(0)`` yields nothing.
    """
    table: dict[tuple[int, int], list[Code]] = {(0, 0): [Code(n=0, basis=())]}
    for n in range(1, n_max + 1):
        bucket: dict[int, list[Code]] = {}
        for ec in enumerate_doubly_even(n):
            bucket.setdefault(ec.code.rank, []).append(ec.code)
        for k, codes in bucket.items():
            table[(n, k)] = codes
        # Rank-0 (all-zero code) is always a valid representative.
        table.setdefault((n, 0), [Code(n=n, basis=())])
    return table


def direct_sum_seeds(
    N: int,
    max_k: int | None = None,
    *,
    table: dict[tuple[int, int], list[Code]] | None = None,
) -> dict[int, list[SeededCode]]:
    """Build seed table indexed by rank `k` for direct-sum gluings at `N`.

    ``table`` can be supplied to short-circuit the recursive enumeration
    of constituent codes (useful when seeding multiple `N` from a shared
    pre-built cache).

    Returns ``{k: [SeededCode, …]}`` covering ``k = 0 .. max_k`` (default
    ``N // 2``). Each entry is one equivalence class per canonical form;
    no duplicates across decomposition splits.
    """
    cap = N // 2 if max_k is None else max_k
    if table is None:
        table = _build_enum_table(N - 1)

    seeds_by_k: dict[int, list[SeededCode]] = defaultdict(list)
    seen_canons: dict[int, set[Code]] = defaultdict(set)

    for k in range(cap + 1):
        for n1 in range(1, N // 2 + 1):
            n2 = N - n1
            for k1 in range(0, k + 1):
                k2 = k - k1
                codes_1 = table.get((n1, k1))
                codes_2 = table.get((n2, k2))
                if not codes_1 or not codes_2:
                    continue
                for i, C1 in enumerate(codes_1):
                    # Avoid double-counting symmetric ordered pairs when
                    # both summand buckets are identical.
                    start = i if (n1 == n2 and k1 == k2) else 0
                    for C2 in codes_2[start:]:
                        D = direct_sum(C1, C2)
                        canon = canonical_form(D)
                        if canon in seen_canons[k]:
                            continue
                        seen_canons[k].add(canon)
                        info = canon_info(D)
                        seeds_by_k[k].append(
                            SeededCode(
                                canonical=canon, aut_order=info.aut_order
                            )
                        )

    return dict(seeds_by_k)


def seed_mass(seeds_by_k: dict[int, list[SeededCode]], factorial_N: int) -> dict[int, int]:
    """Sum ``N! // |Aut|`` per rank over the seed table."""
    out: dict[int, int] = {}
    for k, seeds in seeds_by_k.items():
        out[k] = sum(factorial_N // s.aut_order for s in seeds)
    return out
