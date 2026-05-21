"""Process-pool frontier-split parallelism with **pipelined seeder**.

The seeder (sequential DFS to ``frontier_depth``) runs in the main process
but submits each at-depth ``Code`` to the worker pool **as it's discovered**,
overlapping seeder work with worker recursion. Without pipelining, the seeder
dominates ~half the wall time at N ≥ 22 (matches the production D13 V2 → V3
upgrade documented in ``architecture/07-parallel-scaling-profile.md``).

Workers complete out of order; we yield with ``as_completed`` so the first
finished subtree produces output immediately. ProcessPool, not threads —
pynauty holds the GIL.
"""

from __future__ import annotations
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed

from ._augment import EnumeratedCode, is_canonical_augmentation, traverse
from ._canon import CanonInfo, canon_info
from ._qc import qc_candidates
from ._spec import Code


def _walk(C: Code, info: CanonInfo, target_depth: int, max_k: int):
    """Generator: yields ``('above', (C, info))`` or ``('seed', (C, info))``."""
    k = C.rank
    if k >= max_k or k == target_depth:
        yield ('seed', (C, info))
        return
    yield ('above', (C, info))
    for v in qc_candidates(C, info.aut_generators):
        D = C.extend(v)
        info_D = canon_info(D)
        if is_canonical_augmentation(C, D, info_D):
            yield from _walk(D, info_D, target_depth, max_k)


def _subtree_worker(args):
    C, info, max_k = args
    return list(traverse(C, max_k, info))


def enumerate_doubly_even_parallel(
    N: int, max_k: int | None = None, workers: int = 4, frontier_depth: int = 3,
) -> Iterator[EnumeratedCode]:
    cap = N // 2 if max_k is None else max_k
    root = Code.zero(N)
    info_root = canon_info(root)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for kind, payload in _walk(root, info_root, frontier_depth, cap):
            if kind == 'above':
                C, info = payload
                yield EnumeratedCode(code=C, info=info)
            else:
                C, info = payload
                futures.append(ex.submit(_subtree_worker, (C, info, cap)))
        for fut in as_completed(futures):
            yield from fut.result()
