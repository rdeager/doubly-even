/-
FFI shim for nauty/sparsenauty (stub).

The plan: expose the production Rust kernel's `canon_info` via `@[extern]`,
returning a canonical-form basis (as bytes) and the order of the
automorphism group. This puts the Rust implementation in the trust base —
we *state*, but do not *prove*, that the FFI matches the spec below. A
future direction would tighten this gap with a Lean-side certificate
checker.

Build wiring (TODO): a Rust crate compiled as `cdylib` exposing
`extern "C"` functions; lakefile teaches `lean_lib` how to link the `.so`.

This file is **stub only** right now: the externs are declared but not
backed by the Rust crate. Don't `#eval` them — they will linker-fail.
-/
import DoublyEven.Code

namespace DoublyEven.FFI

/--
Canonical-form output: the canonical generator matrix flattened as
`k` rows of `N` bits each (one `UInt8` per bit), plus `|Aut(C)|` as a
`UInt64`. A future, richer return type can split this into a structure;
for the stub we keep the marshalling minimal.
-/
structure CanonResult where
  canonRows : Array (Array UInt8)
  autOrder  : UInt64
  deriving Inhabited

/--
**SPEC** (what the Rust implementation is contracted to deliver):

Given a generator matrix `rows : Array (Array UInt8)` of shape k × N
representing some basis of a code `C ⊆ F_2^N`, return a `CanonResult`
such that:

  1. `result.canonRows` is a basis of a code `C'` permutation-equivalent
     to `C`, depending only on the equivalence class of `C` (the
     canonical form).
  2. `result.autOrder` is `|Aut(C)|` — the order of the stabiliser of
     `C` under the S_N action.

This is a **trust-base statement**, not a Lean theorem. Verifying it
formally would mean proving the Rust matches the spec; out of scope here.
-/
@[extern "doubly_even_canon_info"]
opaque canonInfo (rows : @& Array (Array UInt8)) : CanonResult

end DoublyEven.FFI
