import Lake
open Lake DSL

package «doubly_even_lean» where
  -- Lean 4 reference + spec for the doubly-even enumerator.
  -- Pin the Lean toolchain in `lean-toolchain`; Mathlib's branch must match.

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.28.0"

@[default_target]
lean_lib «DoublyEven» where
  -- All Lean modules live under `DoublyEven/`.
