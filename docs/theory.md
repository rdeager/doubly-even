# The coset-spectrum parent rule: definitions, theorems, proofs

This is the formal layer behind the enumerator's biggest lever family.
[`algorithm.md`](algorithm.md) tells the story — what each lever does,
why it is fast, and what it measured; this document states the
definitions, proves the theorems, and maps each statement to the Rust
function that implements it and the test that pins it. No benchmark
numbers appear here; for those see `algorithm.md` and
[`bottlenecks.md`](bottlenecks.md).

The four results that matter, in one breath: a **weight-spectrum parent
rule** is a sound replacement for the canonical-form parent rule
(Theorem 1), its evaluation factors over a **parent/candidate frame
split** (Lemma 5), a **one-integer-compare bound** decides most
candidates at the first weight stratum (Theorem 2), and a **per-parent
chain** extends that one-compare decision to all later strata while a
pair-structure invariant holds (Lemma 6, Theorem 3). A separate section
covers the orbit-minimisation BFS in the candidate generator: the
**method-of-four-Russians** image kernel (Lemma 7) and why every BFS
scheduling variant produces identical minima (Lemma 8).

## 1. Setting and notation

A binary linear code `C ⊆ F_2^N` of rank `k` is **doubly even** if every
codeword weight is divisible by 4 (such codes are automatically
self-orthogonal). Two codes are **equivalent** if a coordinate (column)
permutation maps one onto the other; `Aut(C) ≤ S_N` is the stabiliser.
The enumerator counts equivalence classes of doubly even `[N, k]` codes
by McKay-style canonical augmentation: rank-`(k+1)` codes `D = ⟨C, v⟩`
are generated from rank-`k` parents `C`, and a generated `D` is kept
exactly when the parent it was built from is *the* canonical parent of
`D` (and `v` is the canonical coset choice within that parent — the
orbit condition handled by the surrounding machinery; see
`algorithm.md`).

| symbol | meaning |
|---|---|
| `C`, `k` | the parent code and its rank |
| `D = ⟨C, v⟩` | the candidate child, rank `k+1` |
| `u` | a nonzero linear functional on `D`; `D` has `2^(k+1) − 1` of them |
| `H_u = ker u` | the hyperplane (index-2 subcode) named by `u` |
| `φ_w(u)` | `#{x ∈ D : wt(x) = w, u(x) = 1}` — the complement-coset weight spectrum |
| `φ(u)` | the tuple `(φ_4(u), φ_8(u), …)` over ascending weight strata |
| frame | the basis `[C's k RREF rows, v]` of `D`; coordinates `x = (x', b) ∈ F_2^k × F_2` |
| `u_C` | the last-coordinate functional `(0, 1)`; `ker u_C = C` in the frame |
| `T_w` | the stratum `{x ∈ D : wt(codeword of x) = w}` (as a set of frame coordinates) |
| `tc, tv` | `|T_w ∩ C-half|`, `|T_w ∩ v-half|` (the two stratum histogram entries) |
| `f̂` | Walsh–Hadamard transform of the stratum indicator `f = 1_{T_w}` |
| `F̂_C, Ĝ_v` | half-frame WHTs of the C-half / v-half indicators (Lemma 5) |
| `E_w` | `{u' ≠ 0 : F̂_C^{(w)}[u'] = tc}` — the C-half argmax set of stratum `w` |
| `amax_w` | `max_{u' ≠ 0} F̂_C^{(w)}[u']` — the per-parent bound of Theorem 2 |
| `M` | the running argmin set of the lex cascade |
| `σ_D` | nauty's canonical column order for `D` (used only in tie-breaks) |
| `L = N − 2k` | dimension of the quotient space the candidate generator works in |
| `σ_Q` | the `GL(L, 2)` matrix induced on that quotient by an `Aut(C)` generator |

Sign convention for the WHT of a set indicator `f : F_2^m → {0, 1}`:

```
f̂[u] = Σ_x (−1)^{u·x} f(x).
```

## 2. The parent rule and its soundness

**Definition 1 (hyperplanes).** The index-2 subcodes of `D` are exactly
the kernels `H_u = ker u` of the `2^(k+1) − 1` nonzero functionals
`u : D → F_2`, and the assignment `u ↦ H_u` is a bijection.

**Lemma 1.** Every subcode of a doubly even code is doubly even.

*Proof.* Its codewords are a subset of `D`'s, so every weight is still
divisible by 4. ∎

So every hyperplane of `D` is an admissible rank-`k` parent.

**Definition 2 (the spectrum).** For a nonzero functional `u` on `D` and
a weight `w`, `φ_w(u) = #{x ∈ D : wt(x) = w, u(x) = 1}` counts the
stratum's codewords in the complement coset `D \ H_u`. The spectrum
`φ(u)` is the tuple of `φ_w(u)` over ascending strata, and spectra are
compared lexicographically.

**Definition 3 (the rule).** The canonical parent of `D` is

```
m(D) = the Aut(D)-orbit of H_u* , where
u*   = the φ-lex-minimal functional, ties broken by: among the argmin
       set M, the u whose hyperplane's σ_D-permuted RREF is
       lexicographically least.
```

A child-rank cap applies (`DEFAULT_PHI_MAX_RANK`, default 13): above it
the legacy rule (drop the last row of the σ_D-permuted RREF) is used
instead. Both pieces are individually sound; mixing is covered by
Lemma 3.

**Lemma 2 (iso-invariance of φ).** If `π` is a coordinate permutation
with `π(D) = D'`, then `φ_w(u ∘ π^{-1}) = φ_w(u)` for every `u` on `D`
and every `w`. In particular (taking `D' = D`) the argmin set `M` is a
union of `Aut(D)`-orbits of functionals, and isomorphisms transport
argmin sets to argmin sets.

*Proof.* `π` restricts to a weight-preserving bijection `D → D'`, and
`x ∈ D` satisfies `u(x) = 1` iff `π(x)` satisfies `(u ∘ π^{-1})(π x) = 1`.
The two counted sets are in bijection. ∎

**Theorem 1 (soundness).** `m(D)` is an isomorphism-invariant function
of `D` whose value is a single `Aut(D)`-orbit of hyperplanes.
Consequently, canonical augmentation under `m` generates exactly one
representative per equivalence class: each class of rank-`(k+1)` codes
is accepted from exactly one (parent class, coset orbit) pair.

*Proof.* McKay's framework requires of a parent function exactly two
properties: (i) it is transported by isomorphisms
(`m(π D) = π · m(D)`), and (ii) its value is one `Aut(D)`-orbit of
deletable sub-objects.

For (ii): the argmin set `M` is `Aut(D)`-invariant (Lemma 2). The
tie-break selects from `M` a single hyperplane *subspace*: an RREF basis
is a bijective identifier of a subspace, so "σ_D-permuted RREF,
lex-least" picks exactly one element of `M` for each choice of σ_D.
nauty's σ_D is itself well-defined only up to `Aut(D)` — replacing σ_D
by σ_D ∘ α for α ∈ Aut(D) maps the selected hyperplane to its α-image —
so the *orbit* of the selected hyperplane is independent of the
representative σ_D, and `m(D)` is a well-defined single orbit. (This is
the same argument the legacy σ-derived rule relies on; the φ rule only
shrinks the candidate set from "all hyperplanes" to `M` first.)

For (i): an isomorphism `π : D → D'` transports `M` to `M'` (Lemma 2)
and σ_{D'} can be chosen as σ_D composed with `π`; the lex-least
σ-permuted RREF over `M'` is then the `π`-image of the one over `M`,
hence `m(D') = π · m(D)`.

Given (i) and (ii), McKay's generic argument applies verbatim: in the
augmentation tree, each isomorphism class of rank-`(k+1)` codes is
reachable by an accepted edge only through the parent class `m(D)`, and
within that parent through one `Aut`-orbit of extensions, so exactly one
representative is emitted. Selecting parents by invariants and reserving
the canonical form for ties is the same design Bouyukliev–Bouyuklieva
use in their column-augmentation classifier (see
[`references.md`](references.md)). As defence in depth, every production
run also verifies the Gaborit mass formula
`Σ N!/|Aut(C_i)| = σ(N, k)` at every rank — a counting certificate that
would fail loudly if the orbit-uniqueness argument were ever violated in
code. ∎

**Lemma 3 (rank-cap mixing).** Using the φ rule for child ranks
`k+1 ≤ r` and the legacy rule above `r` is sound.

*Proof.* Rank is an isomorphism invariant, so "which rule applies" is
itself iso-invariant, and the McKay uniqueness argument is local to one
child rank: it needs *a* sound parent function for the codes of that
rank, not the same one globally. Each rank therefore independently emits
one representative per class. ∎

## 3. The spectral calculus

All φ evaluation happens in the frame `[C's RREF rows, v]`, where the
candidate's own hyperplane is `ker u_C` and frame coordinates are small
integers. Candidate generation guarantees `v ∉ C`, so the frame is a
basis.

**Lemma 4 (WHT identity).** For the stratum indicator `f = 1_{T_w}`,

```
f̂[u] = |T_w| − 2 φ_w(u),   equivalently   φ_w(u) = (|T_w| − f̂[u]) / 2.
```

Hence minimising `φ_w` over functionals is maximising `f̂`, and the
lex-minimisation of `φ` is the stratum-by-stratum cascade: initialise
`M` to all nonzero functionals; at each stratum in ascending weight
order replace `M` by `argmax_{u ∈ M} f̂[u]`; stop when `u_C ∉ M`
(**reject**), `M = {u_C}` (**accept, unique**), or strata are exhausted
with `|M| > 1` (**tie** — invoke the σ_D tie-break).

*Proof.* `f̂[u] = #{x ∈ T_w : u(x) = 0} − #{x ∈ T_w : u(x) = 1}
= (|T_w| − φ_w(u)) − φ_w(u)`. Lex order compares tuples coordinate by
coordinate, which is exactly the iterated argmin (= argmax of `f̂`)
intersection. ∎

**Lemma 5 (split-frame factorisation).** Write a frame coordinate as
`x = (x', b)` and a functional as `u = (u', a)`, with `u(x) = u'·x' ⊕ ab`.
Let `g_C(x') = f(x', 0)` and `g_v(x') = f(x', 1)` be the C-half and
v-half indicators of a stratum, with `2^k`-point WHTs `F̂_C` and `Ĝ_v`.
Then

```
f̂[(u', a)] = F̂_C[u'] + (−1)^a Ĝ_v[u'] ,
```

and in particular `f̂[u_C] = f̂[(0,1)] = F̂_C[0] − Ĝ_v[0] = tc − tv` —
available from the two histogram entries with no transform at all.

*Proof.*
`f̂[(u',a)] = Σ_{x',b} (−1)^{u'·x' ⊕ ab} f(x',b)
= Σ_{x'} (−1)^{u'·x'} (g_C(x') + (−1)^a g_v(x'))`. This is literally the
last butterfly stage of the full `2^(k+1)`-point WHT, factored out. ∎

The computational meaning: everything C-half (`g_C`, its weights,
histograms, stratum lists, the transforms `F̂_C^{(w)}`, the sets `E_w`,
the bounds `amax_w`) is **identical for every sibling candidate of one
parent** and is computed once per parent; per candidate only the v-half
remains.

## 4. First-stratum decisions in O(1)

Throughout, "first stratum" means the lowest weight `w` with
`T_w ≠ ∅`, where `M` is still the full functional set.

**Theorem 2 (the pair-max bound).** For any `u' ≠ 0`,

```
max( f̂[(u',0)], f̂[(u',1)] ) = F̂_C[u'] + |Ĝ_v[u']|  ≥  F̂_C[u'] .
```

Consequently, with the per-parent constant
`amax_w = max_{u'≠0} F̂_C^{(w)}[u']`:

> if `amax_w > tc − tv`, some functional strictly beats `u_C` in the
> first stratum, so `u_C` leaves the argmax set and the candidate is
> **rejected** — decided by one integer comparison against
> precomputed parent data, with no per-candidate transform.

*Proof.* `max(a + g, a − g) = a + |g|` for integers, applied with
`a = F̂_C[u']`, `g = Ĝ_v[u']`; dropping `|g| ≥ 0` gives the bound. If
`amax_w > tc − tv`, pick `u'*` attaining `amax_w`: one member of the
pair over `u'*` has `f̂ ≥ F̂_C[u'*] > tc − tv = f̂[u_C]` (Lemma 5), and
both pair members are in `M` at the first stratum. ∎

The test is one-sided: `amax_w ≤ tc − tv` proves nothing, and the
cascade falls through to the generic machinery (compute `Ĝ_v` by one
half-size WHT, intersect `M`, continue).

**Corollary 2.1 (coset-only first stratum).** If `tc = 0` (and
`k ≥ 1`), the candidate rejects in O(1).

*Proof.* The stratum is nonempty so `tv > 0` and
`f̂[u_C] = −tv < 0`. `g_C ≡ 0` gives `F̂_C ≡ 0`, so for any `u' ≠ 0`
(one exists since `k ≥ 1`) the pair max is `|Ĝ_v[u']| ≥ 0 > −tv`. ∎

**Corollary 2.2 (C-only first stratum).** If `tv = 0`, then `u_C`
survives the first stratum, and the surviving argmax set is exactly

```
M = E_w ∪ {u_C} ∪ (u_C + E_w),    E_w = {u' ≠ 0 : F̂_C[u'] = tc},
```

where `u'` is identified with `(u', 0)` and `u_C + u'` with `(u', 1)`.
If `E_w = ∅` the candidate is **accepted with `M = {u_C}`**, in O(1).

*Proof.* `g_v ≡ 0` gives `Ĝ_v ≡ 0`, so `f̂[(u', a)] = F̂_C[u']`
independent of `a`. For an indicator, `|F̂_C[u']| ≤ F̂_C[0] = tc`, and
`f̂[u_C] = tc` is the maximum. The argmax set is `u_C` together with
*both* functionals over each `u' ∈ E_w`. ∎

Note the shape in Corollary 2.2: every `u' ∈ E_w` enters `M` as the
complete pair `{(u',0), (u',1)}`. That **pair structure** is the
invariant the next section exploits.

## 5. The E-set chain: O(1) decisions at all later strata

Suppose a candidate survived a C-only first stratum, so
`M = E ∪ {u_C} ∪ (u_C + E)` with `E ≠ ∅` and all pairs complete. The
cascade now walks the remaining strata in ascending weight order. Each
later stratum is one of three kinds, by its histogram `(tc, tv)`.

**Lemma 6 (pair preservation and O(1) stratum decisions).** While `M`
has the form `E_cur ∪ {u_C} ∪ (u_C + E_cur)` with complete pairs and
`E_cur ≠ ∅`:

1. **v-only stratum** (`tc = 0 < tv`): the candidate **rejects**, using
   no parent data at all.
2. **mixed stratum** (`tc, tv > 0`): if
   `max_{u' ∈ E_cur} F̂_C^{(w)}[u'] > tc − tv`, the candidate
   **rejects** — one comparison against a per-parent bound. Otherwise
   the chain ends: `M` is materialised and the generic stratum machinery
   continues from this exact stratum (the pair structure may break here,
   because `±Ĝ_v` can separate the two halves of a pair).
3. **C-only stratum** (`tv = 0`): the new argmax set keeps `u_C` and
   filters the pairs *together*:
   `E_cur ← {u' ∈ E_cur : F̂_C^{(w)}[u'] = tc}` — a parent-side
   computation. The pair structure is preserved. If `E_cur` becomes
   empty, the candidate is **accepted with `M = {u_C}`**.

*Proof.* In each case compare `f̂` over the current `M` (Lemma 5).

(1) `f̂[u_C] = −tv < 0`, while any pair has max `|Ĝ_v[u']| ≥ 0`; with
`E_cur ≠ ∅` some pair member beats `u_C`, which therefore leaves the
argmax set.

(2) The pair over `u'` has max `F̂_C[u'] + |Ĝ_v[u']| ≥ F̂_C[u']`
(Theorem 2). If the stated bound exceeds `tc − tv = f̂[u_C]`, the
maximising pair member beats `u_C`. The converse direction is *not*
claimed; on failure of the test, the cascade proceeds generically with
the same `M`, so no decision is altered.

(3) `Ĝ_v ≡ 0` on this stratum, so both members of the pair over `u'`
score `F̂_C[u']`, while `u_C` scores `tc`; since `F̂_C[u'] ≤ tc`
always (indicator bound), `u_C` stays, and a pair stays iff
`F̂_C[u'] = tc` — both members in or out together, which is exactly the
stated filter and preserves the invariant. Empty filter result means
`M = {u_C}`: by Lemma 4's cascade semantics that is an accept, and it is
reached without ever materialising `M`. ∎

**Theorem 3 (per-parent chain, shared across siblings).** Index the
parent's C-strata (weights with `tc > 0`) in ascending order
`w_1 < w_2 < …`. Define inductively

```
E_1 = E_{w_1},   E_{i+1} = {u' ∈ E_i : F̂_C^{(w_{i+1})}[u'] = tc_{i+1}},
B_i = max_{u' ∈ E_i} F̂_C^{(w_i)}[u']  (per-position bound, over the
                                        E-set current when w_i is met)
```

— all functions of the parent alone. Then for every sibling candidate
that is still undecided when it reaches C-stratum position `i`, the
cascade state is exactly `M = E_i ∪ {u_C} ∪ (u_C + E_i)`, and the
decisions of Lemma 6 evaluated against the chain data `(E_i, B_i)` are
identical to the decisions of the generic argmin cascade. The chain can
therefore be built lazily once per parent and read in O(1) by every
sibling.

*Proof.* By induction on the strata walked. A candidate's stratum
sequence interleaves the parent's C-strata with candidate-specific
v-only strata. A v-only stratum rejects (Lemma 6.1) — the candidate
never advances past it, so reaching position `i` undecided means every
previous stratum it met was one of the parent's C-strata `w_j (j < i)`
handled by case (3) — case (2) either rejects (decided) or ends the
chain (no longer "on the chain"). Case (3)'s filter at `w_j` maps
`E_j ↦ E_{j+1}` independently of the candidate, hence the state at
position `i` is `E_i` for *every* such sibling, and the case-(2) bound
at `w_i` is `B_i`. Lemma 6 already established that each chain decision
agrees with the generic cascade on the same `M`. ∎

The cascade-state diagram, in summary:

```
first stratum:  coset-only → REJECT (O(1), Cor 2.1)
                amax bound fires → REJECT (O(1), Thm 2)
                C-only, E empty → ACCEPT-UNIQUE (O(1), Cor 2.2)
                C-only, E ≠ ∅ → enter chain (Thm 3)
                else → generic machinery (half-WHTs per stratum)
on chain:       v-only → REJECT (O(1))
                mixed, bound fires → REJECT (O(1))
                mixed, bound silent → leave chain, generic machinery
                C-only → advance chain; E empties → ACCEPT-UNIQUE (O(1))
exhausted with |M| > 1 → TIE → σ_D tie-break (one canon call)
```

A practical remark: on real parents every low stratum tends to be
populated on the C side, so the v-only chain reject (case 1) almost
never fires in production — the bound reject and the parent-side filter
do the work. The case is still required for exactness (and dominates on
random frames, which is what the property tests generate).

## 6. Orbit minimisation in the candidate generator

Candidate cosets `v` for extending a parent `C` are enumerated in an
`L = N − 2k`-dimensional `F_2` quotient space, on which `Aut(C)` acts
through induced matrices `σ_Q ∈ GL(L, 2)`; the doubly-even-compatible
("singular") vectors form a union of `Aut(C)`-orbits, and the generator
needs the minimum representative of each orbit (see `algorithm.md` for
the derivation of the quotient and the singular condition). The
orbit-minimum computation is a BFS: scan singular representatives in
ascending order; each unseen one is an orbit minimum, and its orbit is
closed by repeatedly applying the generator matrices to the frontier.

**Definition 4 (orbit-min BFS).** With `seen₀` = ∅ and, for an orbit
seed `r`, `frontier₀ = {r}`:

```
frontier_{t+1} = ( ⋃_{σ ∈ gens} σ · frontier_t ) \ seen_t ,
seen_{t+1}     = seen_t ∪ frontier_{t+1} .
```

**Lemma 7 (method-of-four-Russians image kernel).** Split `x ∈ F_2^L`
into bytes `x = (x_0, …, x_{c−1})`, `c = ⌈L/8⌉`. For each generator
matrix `σ` precompute `tables[j][b] = σ · (b « 8j)` for all 256 byte
values `b`. Then

```
σ · x = ⊕_j tables[j][x_j] ,
```

so one image costs `c` table loads and XORs. Per generator the table
costs `c · 256` precomputed images; the BFS visits on the order of
`2^(L−2)` elements per generator, so the build amortises whenever the
universe is large (the kernel gates the method at `L ≥ M4R_MIN_L`).

*Proof.* `x = ⊕_j (x_j « 8j)` and `σ` is linear. ∎

**Lemma 8 (schedule independence).** The sets `frontier_t` and
`seen_t` of Definition 4 — and therefore the set of orbit minima — do
not depend on the order in which images are generated or probed within
a level, nor on how a level is partitioned among threads, provided
each element's first insertion is recorded exactly once.

*Proof.* Induction on `t`. `frontier_{t+1}` is defined from
`frontier_t` and `seen_t` as a set union followed by a set difference,
both order-free; exactly-once insertion makes the implemented "claimed
this element" relation coincide with membership in
`frontier_{t+1} \ seen_t`. The ascending outer scan then encounters
exactly the same unseen seeds, in the same order. ∎

Lemma 8 is what licenses three interchangeable BFS bodies — the chained
bit-walk, the four-Russians gen-major chunked body, and the pooled
(range-split, atomic-bitset) parallel body — to be **bit-identical** in
output, which the equivalence tests assert directly.

## 7. Theorem-to-code index

Anchors are `module::function` paths in the kernel crate; they are
stable across file moves. Tests live in the same module unless a path
is given.

| statement | implementation | pinned by |
|---|---|---|
| Def 2/3 — φ, the rule, cascade semantics | `parent_rule::phi_cascade_shared`, `parent_rule::PhiOutcome` | `matches_brute_force_on_fixed_frames`, `matches_brute_force_on_orthogonal_sweep`, `random_frames_match_brute_force` |
| Def 3 — tie-break, rank cap, rule dispatch | `parent_rule::tie_break_parent`, `parent_rule::ParentRule::from_env`, `parent_rule::DEFAULT_PHI_MAX_RANK`; dispatch in `enumerate::WorkerState::test_candidate` | `tests/parent_rule_equivalence.rs::coset_spectrum_matches_legacy_n{10,12,14,16}` |
| Lem 2 — iso-invariance | (property of the definition) | `column_permutation_leaves_outcome_invariant` |
| Thm 1 — McKay soundness | the accept/reject contract of `parent_rule` + `enumerate` | per-rank accept-identity audit (`scripts/experimental/d15_phi_audit.py`), the always-on mass-formula panic, `tests/parent_rule_equivalence.rs` |
| Lem 3 — rank-cap mixing | rank dispatch in `enumerate::WorkerState::test_candidate` | `tests/parent_rule_equivalence.rs::rank_cap_mixing_is_sound` |
| Lem 4 — WHT identity | `parent_rule::wht_in_place` + cascade loop | `wht_matches_direct_parity_counts` |
| Lem 5 — split-frame factorisation | `parent_rule::PhiParentCtx` (per-parent C-half data, `ensure_fhat`) | `split_wht_combine_matches_full_frame_wht` |
| Thm 2 + Cor 2.1/2.2 — first-stratum fast paths | first-stratum arm of `parent_rule::phi_cascade_split` (`amax_w`, `E_w` from the ctx) | brute-force property tests above; production counter `phi_s1_fastpath` |
| Lem 6 + Thm 3 — E-chain | `parent_rule::PhiParentCtx::ensure_chain` (`chain_e`, `chain_bound`), chain arm of `phi_cascade_split` | `e_chain_deterministic_witnesses` (hand-computed frame), `e_chain_multi_stratum_matches_brute_force`; production counter `phi_chain_fastpath`; A/B bit-equality of `phi_strata_sum` |
| Lem 7 — four-Russians kernel | `orbit::m4r_build`, `orbit::m4r_apply`, gate `orbit::M4R_MIN_L` | `m4r_body_matches_legacy_walk_exactly` |
| Lem 8 — schedule independence | `orbit::orbit_minima_m4r` vs the walk body; pooled arms in `orbit` + `seeder_pool` | `m4r_body_matches_legacy_walk_exactly`, `pooled_orbit_min_matches_sequential_exact`, `pooled_gray_walk_matches_sequential_exact` |

## 8. References

- B. D. McKay, *Isomorph-free exhaustive generation*, J. Algorithms 26
  (1998) — the canonical-augmentation framework and the two parent-rule
  requirements used in Theorem 1.
- I. Bouyukliev, S. Bouyuklieva, 2019 (arXiv:1907.10363) — §3.3:
  invariant-first parent selection with canonical-form tie-breaks, the
  design precedent for Definition 3; also an independent validation
  oracle for the counts.
- DFGHILM (Adinkra chromotopology), Appendix B — the algorithmic
  specification this enumerator implements.
- V. Arlazarov, E. Dinic, M. Kronrod, I. Faradžev, 1970 — the "method
  of four Russians" (Lemma 7).
- Full bibliography with links: [`references.md`](references.md).

---

*Maintainers: the session-by-session development history behind these
results — including the mapping from internal change labels to the
descriptive names used here — lives in the unversioned maintainers' tree
under `markdown/architecture/` (not shipped with this repository).*
