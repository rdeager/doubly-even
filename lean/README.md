# Lean 4 spec + reference for the doubly-even enumerator

This is a third paradigm alongside the production Rust kernel (`../rust/`) and
the pedagogical Python reference (`../doubly_even/clean/`). Where:

- **Rust** optimises wall-time. Production. Owns nauty/sparsenauty FFI.
- **clean Python** optimises readability of the *algorithm*. Pedagogical.
- **Lean 4** (this directory) optimises readability of the *spec*, and gives
  machine-checked correctness for small N.

## Why a third deliverable

The clean-Python rewrite (2026-05-21) surfaced the k=1,2 Young-subgroup
pre-seed idea via profiling — an algorithmic insight that landed in
production. The Lean 4 deliverable bets on a different lens: writing the
spec forces every invariant the enumerator preserves to become an explicit
`theorem`, which has a chance of exposing either redundant checks or
strengthened checks. Same idea, different surface area.

## Type-level design

Codes are **rank-k submodules of F_2^N** by construction:

```lean
def Code (N k : ℕ) :=
  { C : Submodule (ZMod 2) (Fin N → ZMod 2) // Module.finrank (ZMod 2) C = k }
```

The wrapper makes `rank = k` a *type-level* assertion — every value of
`Code N k` has rank k by construction, no separate invariant to thread.
Predicates like `IsDoublyEven` and `IsSelfOrthogonal` are then `Prop`s on
`Code N k`. Permutation equivalence lifts the S_N action on coordinates to
submodules via `Submodule.map (permLinearEquiv σ)`.

A `Submodule` carries no basis. To talk to nauty we materialise a basis
(`Basis (Fin k) (ZMod 2) C.1`); the choice does not affect the canonical
form because nauty canonicalises the *submodule*, not the matrix.

## Module map

| Module                          | Purpose                                          |
|---------------------------------|--------------------------------------------------|
| `DoublyEven.lean`               | Umbrella; imports the spine.                     |
| `DoublyEven/Vectors.lean`       | `BinVec N`, Hamming weight, F_2 inner product.   |
| `DoublyEven/Code.lean`          | `Code N k`; `IsDoublyEven`, `IsSelfOrthogonal`. |
| `DoublyEven/Equivalence.lean`   | S_N action on submodules; `Equivalent`.          |
| `DoublyEven/FFI.lean`           | Stub `@[extern]` declaration for nauty.          |

## Three-tier verification story

1. **Spec layer.** `Code`, `IsDoublyEven`, `Equivalent` etc. — math-natural
   definitions. Used as the reference everything else is judged against.
2. **Executable brute force.** A `decide`-friendly enumerator over `Finset`s
   for tiny N. Run via `native_decide`. Lands DFGHILM Table 3 cells as
   actual `theorem` statements for small N.
3. **FFI bridge (eventual).** Rust kernel's nauty canon_info exposed via
   `@[extern]`. **The Rust implementation is in the trust base** — we
   state, but do not prove, that the FFI matches its declared spec
   (`FFI.lean`). A future certificate-checker direction would tighten
   this gap.

## Status

Scaffold only. The five spine modules type-check (validated via
`lean_run_code`). No `sorry`s in current files; brute-force enumerator
and table-cell theorems are TODO.

## Build (when toolchain is local)

```sh
elan default leanprover/lean4:v4.28.0
lake exe cache get       # fetch Mathlib oleans (avoids 30+ min recompile)
lake build
```

## MCP-server filesystem note

The `lean-lsp` MCP runs in a separate container and does not share this
filesystem. While iterating, every module's source can be validated by
pasting its content into `lean_run_code` (with explicit imports). For
full builds the user must either run lake locally or sync this directory
into the MCP-server's view.
