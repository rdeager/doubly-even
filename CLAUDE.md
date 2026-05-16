# Project context for Claude

You're working on the **`doubly-even`** package — an enumerator of doubly
even binary linear codes `[N, k]` up to permutation equivalence. The
classification underlies the Adinkra chromotopology problem from
supersymmetric representation theory (Doran–Faux–Gates–Hübsch–Iga–
Landweber–Miller, henceforth **DFGHILM**, whose Appendix B is the
algorithmic spec we implement).

The repo is one of three peer directories on the host:

- `/workspace/src/` (this repo) — Python code, `uv`-managed.
- `/workspace/markdown/` — design documentation. Read
  `markdown/README.md` first.
- `/workspace/inbox/mathpix/` — papers, Mathpix-converted to markdown.
  `DFGHILM_ATMP.md` is the primary spec; Appendix B is the recipe.

## Quick orientation

The package is split into three layers; depend only on the layers above.

```
spec/        readable executable spec (math, no perf work)
  vectors.py     BinVec=int with bit ops, wt, dot, polarisation
  codes.py       Code dataclass (n, basis), rref, dual, contains, extend
  doubly_even.py is_doubly_even via Corollary B.1; augmentation predicate
  mass.py        sigma_brute (works), gaborit_sigma (stub, see below)

canon/       wraps pynauty for canonical labels + Aut(C)
  bipartite.py   G(C) bipartite encoding (codewords × columns)
  nauty.py       canon_info, canonical_form, are_equivalent
  permutations.py hand-rolled Schreier-Sims for exact |Aut| (pynauty
                  returns a float that loses precision past ~2^53)

enumerate/   the canonical-augmentation search
  filters.py     coset reps in C-perp/C, weight-mod-4, Aut(C)-orbit-min
  augment.py     canonical_parent(D); is_canonical_augmentation;
                  enumerate_doubly_even(N) -> EnumeratedCode iter
```

## Conventions

- **`uv` for everything.** `uv add`, `uv run pytest`, `uv run dec`.
  No `pip install` outside of `uv`. No `requirements.txt`.
- **Binary vectors are Python `int`s.** Bit `i` is component `i`. XOR is
  addition, `int.bit_count()` is weight. Don't wrap them in a class.
- **Codes are immutable.** `Code(n, basis)` is frozen; derived data
  (RREF, dual) is recomputed on demand. If you need to cache derived
  data, use an external cache, not mutation.
- **`spec/` is the reference.** Every `spec/` function should be readable
  by someone who knows GF(2) linear algebra. Optimise elsewhere if you
  need to.
- **Tests cite their oracle.** Each `(N, k)` check names what it's
  matching (DFGHILM Table 3, `sigma_brute`, AGL(3, 2), etc.).
- **Slow tests are marked.** `@pytest.mark.slow` is skipped by default;
  enable with `uv run pytest --run-slow`.

## Validation oracles

Three independent checks the test suite relies on:

1. **Mass formula:** `Σ N!/|Aut(C_i)|` over emitted classes must equal
   `sigma_brute(N, k)`. Verified for `N ≤ 8`. This is internal
   consistency.
2. **DFGHILM Table 3:** published equivalence-class counts of doubly
   even `[N, k]` codes. The enumerator matches every cell exactly
   through `N = 16` in the default suite, `N = 18` with `--run-slow`.
   Hardcoded in `tests/test_augment.py::DFGHILM_TABLE_3`.
3. **Bouyukliev–Bouyuklieva 2019** (`inbox/mathpix/1907.10363v1.md`)
   gives counts for `[N, k, ≥ d]` codes at `N = 31, 32`. Not yet wired
   into tests — we don't have a minimum-distance filter.

## Open / parked items

- `/workspace/markdown/` is not under git. If we want it versioned,
  either move it under `src/markdown/` or `git init` `/workspace/`.
- Bouyukliev–Bouyuklieva 2019 has `[N, k, ≥ d]` validation counts at
  `N = 31, 32`. Not yet wired into tests — we don't have a
  minimum-distance filter.
- Performance ceiling in pure Python is now ≈ 28 s at `N = 20` (down
  from 235 s in baseline). At the post-optimisation growth rate of
  ~7×/`N+2`, `N = 22` runs in ~3 minutes. To get past that comfortably
  we'd need C++/Rust hot kernels — see the next-session plan.

## Recent algorithmic + representation wins (Python session, post-Phase 3)

See `/workspace/markdown/architecture/04-optimisations.md` for the
full write-up. Headline: cumulative 7–8× speedup vs pre-session
baseline, achieved by (1) quotient-space candidate enumeration
(DFGHILM B.3), (2) LRU-cached `canon_info`, (3) verified closed-form
`gaborit_sigma` + mass-stopping shortcut in the recursion, (4)
hoisting `Code.rref_basis()` out of the orbit-min BFS, (5) weight-
enumerator prefilter on the subspace orbit BFS, and (6) trusting
pynauty's float order whenever it's within float64's exact-integer
range. `bench.py` lives in `scripts/`; baseline + per-step JSON
records are in `scripts/bench-results/` (gitignored).

## Useful commands

```sh
uv sync --all-extras --dev               # bootstrap a fresh checkout
uv run pytest                            # 263 fast tests, ~10 s
uv run pytest --run-slow                 # adds slow mass + Table 3 cells (~45 s total)
uv run python scripts/bench.py --label baseline  # benchmark; writes JSON

# Enumerate doubly even codes of length N (yields EnumeratedCode objects)
uv run python -c '
from doubly_even.enumerate.augment import enumerate_doubly_even
for ec in enumerate_doubly_even(8):
    print(f"k={ec.code.rank} |Aut|={ec.aut_order} basis={list(ec.code.basis)}")
'
```

## Git etiquette

- Commits are authored by the user (email + name from local
  `git config`). Don't touch `~/.gitconfig` from inside the agent.
- `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
  trailer goes on every assistant-driven commit.
- The repo had its history rewritten once (`git filter-repo`) to fix
  author emails; further rewrites should be similarly explicit and
  user-approved.
