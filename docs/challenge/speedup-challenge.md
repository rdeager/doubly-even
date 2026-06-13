# Open problem: a >2× speedup for enumerating doubly-even codes

You are a strong researcher in algorithmic combinatorics and high-performance
code. Below is a faithful description of a working enumerator, where its time
goes, and what has already been measured and ruled out. At the end there is one
task.

This document is deliberately *neutral*. It states the problem, the current
implementation, the profile, and the genuine dead ends — and nothing about
where a solution might lie. Some prior conclusions about this code were stated
with more confidence than the evidence supported; treat every "closed" claim
below as something to re-derive from its stated justification, not to defer to.
A justification that is a *measurement* deserves trust; a justification that is
a *mathematical argument* deserves to be re-checked, because if the argument is
subtly wrong the door it claims to close is still open.

---

## 1. The problem

A binary linear code `C ⊆ F₂^N` of dimension `k` is **doubly even** if every
codeword has Hamming weight divisible by 4. (Doubly-even codes are automatically
self-orthogonal.) Two codes are **equivalent** if some permutation of the `N`
coordinates maps one onto the other; `Aut(C) ≤ S_N` is the stabiliser of `C`
under this action.

The task the enumerator performs: **count (and emit one representative of) every
equivalence class of doubly-even `[N, k]` codes**, for all `k`, at a given
length `N`. This classification underlies a problem in supersymmetric
representation theory; the published reference counts go up to `N = 28`, and we
have independently reproduced `N = 28` and `N = 29`. The goal is to go faster at
the lengths we can already reach and, eventually, to reach `N = 30`–`32`.

Correctness is non-negotiable and is checked three independent ways: against
published class counts, against an independent classifier's counts, and — on
every production run — against the **Gaborit mass formula**

```
Σ_classes  N! / |Aut(C_i)|  =  σ(N, k)
```

where `σ(N, k)` is a known closed form for the *labelled* count of doubly-even
`[N, k]` codes. Any dedup error (emitting a class twice, or missing one, or
miscomputing an automorphism group) makes this identity fail loudly. A proposed
speedup that cannot keep this identity exact is not a candidate.

## 2. The method: McKay canonical augmentation

The enumerator is an orderly-generation / canonical-augmentation search in the
sense of McKay, *Isomorph-free exhaustive generation* (J. Algorithms, 1998).
Codes are built by rank: each dimension-`(k+1)` code `D = ⟨C, v⟩` is generated
by extending a dimension-`k` code `C` (the *parent*) with a coset
representative `v ∉ C`.

The mechanism that makes this isomorph-free is a **parent function** `m(·)`.
For a code `D`, `m(D)` names a distinguished dimension-`k` subcode of `D` (a
*hyperplane* — an index-2 subcode). A generated `D = ⟨C, v⟩` is **accepted**
exactly when:

1. the parent it was built from is the canonical one: `C ∈ m(D)`; and
2. `v` lies in the canonical `Aut(D)`-orbit of admissible coset extensions
   (this second condition is discharged by the candidate generator, which
   enumerates extensions already reduced to orbit minima — see §5).

McKay's theorem requires of the parent function `m(·)` **exactly two
properties**, and nothing more:

- **(i) Isomorphism-invariance.** For any coordinate permutation `π`,
  `m(πD) = π · m(D)`.
- **(ii) Single-orbit value.** `m(D)` is a single `Aut(D)`-orbit of deletable
  sub-objects (here: one orbit of hyperplanes of `D`).

Given any `m(·)` with these two properties, the augmentation tree visits each
equivalence class exactly once: each class of dimension-`(k+1)` codes is reached
through one parent class and one coset orbit, so exactly one representative is
emitted per class. This is the entire correctness argument for the dedup.

## 3. How the parent function is currently computed

The implementation realises `m(D)` through a **canonical form** of `D`. Concretely:

1. `D` is encoded as a vertex-coloured graph (codewords on one side, coordinates
   on the other), and a graph canonicalisation library (`nauty`, via
   `sparsenauty`) returns a canonical ordering `σ_D` of `D`'s coordinates,
   together with generators of `Aut(D)`.
2. The canonical parent is then read off mechanically from the canonical form —
   the dimension-`k` subcode obtained by dropping the last row of `D`'s
   reduced row-echelon basis under the canonical coordinate order `σ_D`.

This satisfies (i) and (ii): the canonical form is isomorphism-invariant by
construction, and `σ_D` is well-defined up to `Aut(D)`, so the named subcode is
a single `Aut(D)`-orbit.

The consequence that matters for performance: **to evaluate `m(D)` for a single
candidate child `D`, the implementation computes a full graph canonicalisation
of `D`.** Every candidate that the generator produces is canonicalised in order
to test whether it was reached through its canonical parent.

## 4. Where the time goes

The number of *candidates* generated vastly exceeds the number of *classes*
emitted, and every candidate triggers one canonical-form computation. The
canonicalisation phase therefore dominates:

- Canonicalisation (`nauty`) is roughly **90 % of wall-clock** at the lengths we
  profile.
- The cost is driven by **call count**, not by per-call cost. The per-call cost
  is already near the floor for general graph canonicalisation of this object
  (partition refinement inside `nauty` is ~80 % of it, and it is
  data-dependent pointer-chasing that resists micro-optimisation, PGO, and
  vectorisation — all measured, all flat).
- Call count grows fast with `N`. At `N = 29` the run made on the order of
  **3.6 × 10² canonicalisation calls per emitted class** (≈ 87 billion calls
  total for 2.39 × 10⁸ classes), versus ≈ 16 calls per class at `N = 22` —
  roughly a **2.6× growth in calls-per-class every two steps of `N`**.

Representative wall-clock at the current state (one 24-thread desktop unless a
cloud platform is named):

| N  | sequential | parallel | classes |
|----|-----------:|---------:|--------:|
| 22 | 6.64 s | 0.69 s | 5,118 |
| 24 | — | 8.90 s | 37,496 |
| 26 | — | 169.8 s | 494,272 |
| 28 | — | 61 min (72-core cloud) | 21,505,546 |
| 29 | — | 12.32 h (72-core cloud) | 239,465,540 |

The operative conclusion from the `N = 29` profile: **any lever that aims for
>2× must reduce the number of canonicalisation calls — reducing the per-call
cost cannot get there**, because per-call cost is already near the
general-purpose floor and falls, not rises, with `N`.

## 5. The candidate generator (context, not the bottleneck)

For completeness: condition (2) of §2 — that `v` is the canonical coset choice —
is handled before canonicalisation. Extensions of `C` live in an
`L = N − 2k`-dimensional `F₂` quotient on which `Aut(C)` acts by induced
matrices; the generator enumerates only the orbit-minimum representatives of the
admissible ("singular", i.e. doubly-even-preserving) cosets under that action,
by a breadth-first orbit closure. This phase is real but is the *second*-largest
cost, well behind canonicalisation, and it does not produce canonicalisation
calls. It is described here only so that "candidate" is well-defined; it is not
the subject of the task.

## 6. What has already been ruled out (measured)

These are dead. Each is recorded with the evidence that killed it.

- **Cheap invariants used as rejectors.** Several permutation-invariant
  signatures of a candidate `D` were tried as a *pre-filter*: compute the cheap
  signature, and if it proves `D` is not reached through its canonical parent,
  skip the canonicalisation call. Every one tested (per-column weight multisets;
  a column-triple "cubic tensor" degree; a paired-isomorphism certificate; a
  pairwise Gram signature) **lost**. Two independent reasons, both measured:
  (a) the cheap signatures share `nauty`'s own refinement primitives, so the
  cost *per probe* comes out ≈ the cost *per canonicalisation call* — there is no
  cheap-rejector regime; and (b) making such a filter *sound* required, for each
  one, a precomputed blocklist of signature collisions, and generating that
  blocklist presupposes having already run the full enumeration — circular for a
  single-pass goal, and the collisions grow rapidly with `N` regardless.

- **Engineering levers are done.** A native (non-interpreted) kernel, a fast
  allocator, thread-local scratch, outer-DFS parallelism across cores, and
  aggressive codegen flags are all already in place. They cut the *constant*,
  not the *call count*, and are not where a >2× lives.

- **Per-call canonicalisation cost is at the general-purpose floor.** Among
  measured graph encodings and canonicalisers for this object shape,
  `sparsenauty` is the cheapest; PGO on the C side, fat LTO, and
  refinement-side invariant hooks were each measured and are flat-to-negative.

## 7. Constraints any solution must respect

- **Single pass, no oracle.** No precomputed table that presupposes having
  already enumerated the codes (that is circular for the `N ≥ 30` goal).
- **Exact.** No sampling, no probabilistic dedup. The mass-formula identity of
  §1 must hold at every rank `k`, on every run.
- **Scales with `N`.** A lever whose own cost grows as fast as the thing it
  removes is not a lever. State how your proposal's cost scales.

## 8. The task

Propose the **next lever** — the change with the best plausible payoff for the
effort — that would cut the dominant cost (§4) by more than 2×, subject to the
constraints (§7).

Be concrete:

1. **Mechanism.** What exactly changes, and at which step of §2–§5.
2. **Why it is sound.** Tie it back to the correctness argument in §2 and the
   mass-formula identity in §1 — explain why your change still emits exactly one
   representative per equivalence class.
3. **Why it scales.** How its cost grows with `N` relative to the cost it
   removes.
4. **What it does *not* help.** Be honest about the regime where it fails.

If, after working through §2–§7, you conclude that no such lever exists and the
frontier is genuinely exhausted at the algorithmic level, say so — but make the
argument explicit and say which of §2–§7 forces the conclusion. "It is
exhausted" and "I found the lever" are both acceptable answers; an unjustified
guess in either direction is not.
