# Problem: faster exhaustive enumeration of doubly-even codes

You are given a working enumerator, its performance profile, and its measured
dead ends. The task is at the end (§8).

## 1. Definitions and goal

A binary linear code `C ⊆ F₂^N` of dimension `k` is **doubly even** if every
codeword has Hamming weight divisible by 4 (such codes are self-orthogonal).
Two codes are **equivalent** if a permutation of the `N` coordinates maps one
onto the other; `Aut(C) ≤ S_N` is the stabiliser.

Goal: count, and emit one representative of, every equivalence class of
doubly-even `[N, k]` codes, for all `k`, at a given length `N`. Reference counts
are published to `N = 28`; `N = 28` and `N = 29` have been reproduced. We want
the existing lengths faster and eventually `N = 30`–`32`.

Correctness is checked against published counts, against an independent
classifier, and on every run against the mass formula
`Σ_classes N!/|Aut(C_i)| = σ(N, k)`, where `σ(N, k)` is the known labelled count.
Any dedup error breaks this identity. A speedup that cannot keep it exact is not
a candidate.

## 2. Method

Canonical augmentation (McKay, *Isomorph-free exhaustive generation*, 1998).
Codes are built by rank: a dimension-`(k+1)` code `D = ⟨C, v⟩` is generated from
a dimension-`k` code `C` (the parent) and a coset representative `v ∉ C`. A
**parent function** `m` assigns to each `D` a distinguished dimension-`k`
subcode — a hyperplane (index-2 subcode) of `D`; `m` is a fixed
isomorphism-invariant rule whose value is a single `Aut(D)`-orbit of hyperplanes.
A generated `D = ⟨C, v⟩` is accepted iff (1) `C` is the parent selected by
`m(D)`, and (2) `v` is the canonical coset choice within `D` (condition (2) is
discharged by the generator, §5). Under these rules the search emits exactly one
representative per equivalence class.

## 3. Parent computation as implemented

`m(D)` is evaluated from a canonical form of `D`. `D` is encoded as a
vertex-coloured graph (codewords on one side, coordinates on the other); nauty
(sparsenauty) returns a canonical coordinate order `σ_D` together with generators
of `Aut(D)`; the selected parent is the dimension-`k` subcode obtained by
dropping the last row of `D`'s reduced row-echelon basis taken under `σ_D`. Each
generated candidate child `D` is canonicalised in order to evaluate `m(D)`.

## 4. Profile

The number of candidates generated greatly exceeds the number of classes
emitted, and each candidate triggers one canonicalisation. Canonicalisation is
approximately **90% of wall-clock** time. Its per-call cost is near the floor for
graph canonicalisation of this object — about 80% of a call is nauty's partition
refinement, which is data-dependent pointer chasing; micro-optimisation, PGO, and
vectorisation of it are all measured flat — and per-call cost falls slowly as `N`
grows. Call count grows the other way: about **16 canonicalisations per emitted
class at `N = 22` and about 360 at `N = 29`** (≈2.6× per two-step increase in
`N`), i.e. ≈8.7×10¹⁰ calls for 2.39×10⁸ classes at `N = 29`.

| N  | wall | classes |
|----|------|--------:|
| 22 | 6.6 s seq / 0.7 s parallel | 5,118 |
| 24 | 8.9 s parallel | 37,496 |
| 26 | 170 s parallel | 494,272 |
| 28 | 61 min (72-core cloud) | 21,505,546 |
| 29 | 12.3 h (72-core cloud) | 239,465,540 |

## 5. Candidate generator (context, not the subject)

Condition (2) of §2 is handled before canonicalisation. Extensions of `C` live in
an `L = N − 2k` dimensional `F₂` quotient on which `Aut(C)` acts by induced
matrices; the generator enumerates only the orbit-minimum representatives of the
admissible ("singular", doubly-even-preserving) cosets under that action, by a
breadth-first orbit closure. This is real work — the second-largest cost, well
behind canonicalisation — but it produces no canonicalisation calls and is not
the subject of the task.

## 6. Measured negative results

- Cheap isomorphism-invariant signatures of a candidate `D` — per-column weight
  multisets, a column-triple degree tensor, a paired-isomorphism certificate, a
  pairwise Gram signature — were computed in order to **skip the canonicalisation
  on candidates the signature could rule out**. None paid off: each signature's
  cost came out comparable to a canonicalisation call (they exercise the same
  refinement primitives), and making the skip sound required a precomputed table
  of signature collisions whose construction presupposes the finished
  enumeration, with collisions growing rapidly in `N`.
- A native kernel, a fast allocator, thread-local scratch, multicore
  parallelism, and aggressive codegen flags are already in place. They reduce the
  constant factor, not the number of canonicalisations.
- Among graph encodings and canonicalisers measured for this object, sparsenauty
  is the cheapest; PGO, fat LTO, and refinement-side invariant hooks are
  flat-to-negative.

## 7. Constraints any solution must respect

- **Single pass, no oracle:** no precomputed table that presupposes having
  already enumerated the codes (circular for `N ≥ 30`).
- **Exact:** no sampling or probabilistic dedup; the §1 identity holds at every
  `k` on every run.
- **Scales with `N`:** a lever whose own cost grows as fast as the cost it
  removes is not a lever; state how your proposal scales.

## 8. Task

Propose the next lever — the change with the best payoff-to-effort — that reduces
the dominant cost in §4, subject to §7. State: (1) the **mechanism** — what
changes, and at which step of §2–§5; (2) the **soundness argument** — why the
search still emits exactly one representative per equivalence class, tied to §2
and the §1 identity; (3) how the cost **scales** with `N`; (4) where it does
**not** help. If instead you conclude no such lever exists and the frontier is
exhausted at the algorithmic level, state that and identify which facts in
§2–§7 force the conclusion.
