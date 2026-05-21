# `doubly_even.clean` — pedagogical pure-Python enumerator

A self-contained companion to the production Rust kernel. Implements
DFGHILM Appendix B.4 canonical augmentation plus the two algorithmic
improvements over Appendix B that buy ≥ 2× (production has more, but
each of those buys ≤ 1.2× and adds LOC).

- **Improvement #1** (`_qc.py`): Q_C-coordinate quotient enumeration —
  candidate cosets live in `Q_C := C^⊥ / C` (dim `N − 2k`), not in
  `F_2^N`. At `N=22, k=8` that's `2^6` candidates instead of `2^22`.
- **Improvement #2** (`_canon.py`): low-weight-incidence canonicaliser —
  feed nauty only the codewords from the lowest weight strata that
  jointly span C (rather than all `2^k`). ~2× faster at `N=22`. Math
  invariant: column-stabiliser equals `Aut(C)` *iff the included
  codewords span C* — so the algorithm walks strata 4, 8, 12, ... and
  stops once the running set spans.

The module is **standalone**: imports nothing from `doubly_even.*` or
the Rust kernel. Only outside dep is `pynauty` (already in the project).
Schreier–Sims is hand-rolled in `_canon.py` for exact `|Aut|` when
pynauty's float overflows `2^53`.

## Usage

```python
from doubly_even.clean import enumerate_doubly_even

# Sequential
for ec in enumerate_doubly_even(N=18):
    print(ec.code.rank, ec.aut_order)

# Process-pool frontier split (workers >= 2)
for ec in enumerate_doubly_even(N=20, workers=4, frontier_depth=3):
    ...
```

## Verification

```bash
# DFGHILM Table 3 (hardcoded reference values, N ≤ 16)
uv run python - <<'PY'
from doubly_even.clean import enumerate_doubly_even
from collections import Counter
DFGHILM = {
    8: {1: 2, 2: 2, 3: 2, 4: 1},
    10: {1: 2, 2: 3, 3: 3, 4: 2},
    12: {1: 3, 2: 5, 3: 7, 4: 7, 5: 2},
    14: {1: 3, 2: 7, 3: 12, 4: 14, 5: 9, 6: 4},
    16: {1: 4, 2: 10, 3: 23, 4: 38, 5: 36, 6: 23, 7: 9, 8: 2},
}
for N, expected in DFGHILM.items():
    c = Counter(ec.code.rank for ec in enumerate_doubly_even(N))
    got = {k: c.get(k, 0) for k in expected}
    print(f'N={N}: {"OK" if got == expected else "MISMATCH"}')
PY

# Cross-check vs production enumerator (N ≤ 20)
uv run python - <<'PY'
from doubly_even.clean import enumerate_doubly_even as clean_enum
from doubly_even.enumerate.augment import enumerate_doubly_even as prod_enum
from collections import Counter
for N in [12, 14, 16, 18, 20]:
    a = Counter(ec.code.rank for ec in clean_enum(N, workers=4))
    b = Counter(ec.code.rank for ec in prod_enum(N))
    print(f'N={N}: {"OK" if a == b else "MISMATCH"}')
PY

# Mass formula Σ N!/|Aut| == σ(N, k)
uv run python - <<'PY'
import math
from doubly_even.clean import enumerate_doubly_even, gaborit_sigma
from collections import defaultdict
for N in [6, 8, 10, 12]:
    by_k = defaultdict(int)
    for ec in enumerate_doubly_even(N):
        by_k[ec.code.rank] += math.factorial(N) // ec.aut_order
    for k in sorted(by_k):
        s = gaborit_sigma(N, k)
        print(f'N={N} k={k}: {"OK" if by_k[k] == s else "MISMATCH"}')
PY

# Hamming [8, 4, 4]: |Aut| = 1344
uv run python - <<'PY'
from doubly_even.clean._spec import Code
from doubly_even.clean._canon import canon_info
C = Code(8, (0xE1, 0xD2, 0xB4, 0x78))
print('Hamming |Aut| =', canon_info(C).aut_order, '(expected 1344)')
PY

# Parallel determinism
uv run python - <<'PY'
from doubly_even.clean import enumerate_doubly_even
for N in [12, 14, 16]:
    seq = sorted((e.code.rank, e.code.rref()[0]) for e in enumerate_doubly_even(N))
    par = sorted((e.code.rank, e.code.rref()[0]) for e in enumerate_doubly_even(N, workers=4))
    print(f'N={N}: {"OK" if seq == par else "MISMATCH"}')
PY

# LOC check
wc -l doubly_even/clean/*.py
```

## What's intentionally NOT here

These ship in the production Rust kernel; each buys ≤ 1.2× and would
cost LOC that obscures the algorithm:

- Mass-stop via Gaborit σ (D5b, +4–11 %): the σ closed form is in
  `_mass.py` for *verification* but isn't wired into the recursion.
- Degree-based initial vertex partition for nauty (D8, +19 %):
  pynauty gets the bare bipartite colouring.
- Weight-enumerator BFS prefilter (D4): subspace-orbit BFS skips it.
- Two-tier canon cache (D3): no global memoisation across recursion.
- Streaming output, env-var knobs, GCP/cluster split, etc.

For all of these, see `rust/src/` and
`markdown/architecture/04-optimisations.md` in the repo.
