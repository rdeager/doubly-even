# References, credits, validation oracles

The enumerator implements algorithms developed over thirty years of
combinatorics and computational group theory. This page collects the
sources we follow, validate against, and credit.

## DFGHILM (2011) — the algorithmic spec

C. F. Doran, M. G. Faux, S. J. Gates Jr., T. Hübsch, K. M. Iga,
G. D. Landweber, and **Robert L. Miller**, *Codes and Supersymmetry in
One Dimension*, Advances in Theoretical and Mathematical Physics 15(6)
(2011), 1909–1970.

- arXiv: [1108.4124](https://arxiv.org/abs/1108.4124).
- Appendix B is the recipe we implement, end-to-end:
  - **B.1** — Gaborit mass formula `σ(N, k)`.
  - **B.2** — Bipartite-graph encoding `G(C)` for nauty.
  - **B.3** — Doubly-even linear-algebra optimisations.
  - **B.4** — Canonical augmentation framework (McKay 1998 applied to
    doubly-even `[N, k]` codes).
- Table 3 is the source of our cell-by-cell class-count check at every
  `N ≤ 28`. We name this oracle "DFGHILM Table 3" throughout the codebase.
- DFGHILM's own enumeration ran on the OSU
  [Glenn supercomputer](https://www.osc.edu/supercomputing/computing/glenn)
  in 2011. No enumerator was released.

## Robert L. Miller's `de_codes` site

[rlmiller.org/de_codes](https://rlmiller.org/de_codes/) is Robert L.
Miller's independent reference enumeration of doubly even codes. The
site uses a **no-zero-column** convention — any code of length `N`
with ≥ 1 zero column is permutation-equivalent to a shorter code, and
the table counts only those without zero columns. The conventions are
related by

```
total(N) = no_zero_cols(N) + total(N - 1)
```

so DFGHILM-convention totals (which include codes with zero columns,
trivially counted via embedding into the shorter-`N` enumeration) and
Miller-convention totals differ by a known offset.

At `N = 28` this site was the second independent cross-check for our
cloud result (DFGHILM Table 3 was the first):

- DFGHILM `total(28)` (our run) = 21,505,546
- DFGHILM `total(27)` (Table 3) = 2,673,521
- predicted `no_zero_cols(28)` = 21,505,546 − 2,673,521 = 18,832,025
- Miller-reported `no_zero_cols(28)` ≈ 18,832,054 (agreement within
  parse noise)

We credit Robert L. Miller as a DFGHILM author *and* as the maintainer
of the independent reference enumeration.

## Bouyukliev–Bouyuklieva (2019) — a second canonical augmentation

I. Bouyukliev and S. Bouyuklieva, *Classification of linear codes using
canonical augmentation*, arXiv:[1907.10363](https://arxiv.org/abs/1907.10363)
(2019).

A second canonical-augmentation engine for linear codes, with a
column-by-column traversal (rather than the row-by-row recursion we use)
and an industrial implementation in their `Generation` program. The
paper publishes validation counts for `[N, k, ≥ d]` doubly-even codes
at `N = 31, 32` with `k ∈ {4, 5, 6}` — this is the oracle for a future
session that adds a minimum-distance filter to our enumerator.

## Sage `self_orthogonal_binary_codes` — the prior open-source bar

[Sage](https://www.sagemath.org/)'s
`sage.coding.databases.self_orthogonal_binary_codes(N, k, d)` is the
public-domain implementation we directly compare against. Sage is more
general — it enumerates all self-orthogonal binary codes, not just
doubly-even ones — but the doubly-even slice is reachable with
`d = 4`.

Comparisons (single-threaded on a 13700K, the only platform Sage runs
practically):

| N  | Sage wall  | `doubly-even` seq | `doubly-even` parallel |
|----|-----------:|------------------:|-----------------------:|
| 22 |   363.85 s |            6.64 s |                0.691 s |

Sage's enumeration in `sage/coding/binary_code.pyx` (Miller 2007,
NICE-based partition refinement) is **column-augmentation**: it adds
one column at a time to an existing code, the dual of our row-by-row
DFGHILM B.4 recursion. Inside its candidate loop it has competent
prefiltering: quotient-space coordinates (`binary_code.pyx:4017`),
a Gray-code walk over the orthogonal complement
(`binary_code.pyx:4123–4133`), a weight-mod-4 filter on the lift
(`binary_code.pyx:4016`), and a visited-set bitmap to skip
already-processed cosets (`binary_code.pyx:4001, 4018`) — the
shared design space we exploit too. The mechanism difference at the
orbit-rep selection step is what we improve over (see
[`algorithm.md` §1](algorithm.md#1-quotient-space-orbit-min-prefilter)
and the [2026-05-23 prior-art audit](../../markdown/notes/qc-sage-audit-2026-05-23.md)).

Sage is also inherently single-threaded as shipped: the hot loop is
Cython that holds the GIL and uses Python-object refcounting through
its partition-stack machinery. Parallelising it would require
multi-week Cython surgery; nobody has done it.

The full ratio breakdown is in
[`performance.md`](performance.md#sage-comparison).

## Gaborit (1996) — the mass formula

P. Gaborit, *Mass formulas for self-dual codes over `Z₄` and
`F_q + uF_q` rings*, IEEE Transactions on Information Theory **42**(4)
(1996), 1222–1228. DOI:
[10.1109/18.508848](https://doi.org/10.1109/18.508848).

Closed form for `σ(N, k)`, the labelled count of doubly-even `[N, k]`
codes. We use it as the verification oracle on every emitted class
(`Σ N!/|Aut(C_i)| == σ(N, k)`) and as the stopping certificate for the
mass-stop early-termination (see [`algorithm.md` §5](algorithm.md#5-mass-formula-early-stop)).

The implementation is in `src/doubly_even/spec/mass.py`; verified
against the brute-force `sigma_brute` reference at `N ≤ 8`.

## McKay (1998) — canonical augmentation

B. McKay, *Isomorph-free exhaustive generation*, Journal of Algorithms
**26** (1998), 306–324.

The canonical-augmentation framework: every recursion step is checked
against the canonical-parent function `p` so that each equivalence
class is reached via exactly one ancestry. DFGHILM Appendix B.4 applies
this directly to doubly-even codes.

## McKay–Piperno (2014) — nauty / Traces

B. McKay and A. Piperno, *Practical graph isomorphism, II*, Journal of
Symbolic Computation **60** (2014), 94–112.

The modern reference for `nauty` and Traces. We use the sparse variant
(`sparsenauty`) via Rust's [`nauty-Traces-sys`](https://crates.io/crates/nauty-Traces-sys)
crate. At `N = 22`, roughly **90 % of the parallel-kernel wall** is
inside `sparsenauty`'s C code; the per-call cost of ~80 µs is the
algorithmic floor at our graph shape (see
[`algorithm.md` §"What we did not beat"](algorithm.md#what-we-did-not-beat)
for the audit details).

McKay's earlier *Practical graph isomorphism* (Congr. Numer. **30**
(1981), 45–87) introduced the partition-refinement framework that
underlies `nauty`.

## Conway–Pless–Sloane (1992)

J. H. Conway, V. Pless, N. J. A. Sloane, *The binary self-dual codes of
length up to 32: a revised enumeration*, J. Comb. Theory A **60**
(1992), 183–195.

Published class counts for binary self-dual codes at `N ≤ 32`. We don't
enumerate self-dual codes directly (we enumerate doubly-even; the
self-dual slice is `k = N/2`), but the overlap at `N = 24` and `N = 32`
is a useful sanity check for future work.

## Compute environments

DFGHILM's 2011 enumeration ran on the [OSU Glenn
supercomputer](https://www.osc.edu/supercomputing/computing/glenn). The
`N = 28` row in the published Table 3 was not independently reproduced
from a public implementation until 2026-05-21. There is no published
table at `N = 29`; our 2026-05-23 enumeration is the first publicly
reproducible result at that length, mass-formula certified (see
[`docs/results/n29.json`](results/n29.json)).

We reproduce DFGHILM Table 3 with:

- A single 13700K desktop (cumulative ~525× faster than Sage at
  `N = 22`) for `N ≤ 26` in seconds-to-minutes; `N = 27` in tens of
  minutes.
- A single GCP `c4a-standard-72` cloud VM (~$3 of on-demand compute,
  72 Neoverse V2 cores) for `N = 28` in 61 min and (~$35) for
  `N = 29` in 12.3 hr.
- An Apple Silicon M5 / M5 Pro MacBook with 64 GB of unified memory
  is *predicted* (not measured) to handle the same workload up to
  `N = 28` overnight: per-thread throughput on M5 P-cores is
  competitive with the 13700K, the unified memory comfortably fits
  the per-worker LRU at `CANON_CACHE_CAP ≈ 200 000`, and the same
  target-conditional `Cargo.toml` patch used on the c4a-72 ARM
  build is already on `main`. Total throughput on a 14-core M5 Pro
  is still well below the 72-core c4a-72, so `N = 29` remains a
  cloud-scale problem.

The hardware gap between 2011 and 2026, combined with the algorithmic
levers documented in [`algorithm.md`](algorithm.md), is large enough
that DFGHILM's Table 3 is now reachable from a developer laptop in
seconds-to-hours for `N ≤ 28` and from a cloud VM in an hour for
`N = 28` (12 hr for the new `N = 29` cell). The `N = 30` and `N = 32`
frontiers remain hard; see
[`cluster-deployment.md`](cluster-deployment.md) for the honest
sketch of what would push them further.

## Other background

- W. C. Huffman, V. Pless, *Fundamentals of Error-Correcting Codes*,
  Cambridge University Press, 2003. Textbook reference for self-dual
  code theory.
- J. S. Leon, *Permutation group algorithms based on partition, I*, J.
  Symbolic Comput. **12** (1991), 533–583. The substrate for sparse
  partition refinement; relevant if you want to roll your own
  canonicaliser (we tried; sparsenauty wins).
- T. Junttila, P. Kaski, [bliss](https://users.aalto.fi/~tjunttil/bliss/).
  Another sparse graph canonicaliser; we A/B-benched it against
  sparsenauty on our graph shape and it was 1.15–2.28× slower
  uniformly (vendored under `rust/vendor/bliss-0.77/` for the
  reproducibility of that result).
- R. T. Bilous, G. H. J. van Rees, *An enumeration of binary self-dual
  codes of length 32*, Des. Codes Cryptogr. **26** (2002), 61–86.
  Independent self-dual enumeration at `N = 32`.
