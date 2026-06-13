/-
Codes as rank-`k` submodules of F_2^N, with doubly-even and self-orthogonal
predicates.

The math-natural definition: an [N, k] binary linear code is a k-dimensional
subspace of F_2^N. We capture this directly:

    Code N k := { C : Submodule … // Module.finrank … C = k }

so that the rank is a *type-level* assertion — every value of `Code N k` has
rank k by construction. This is the payoff the user asked for: dependent
types making the algorithm self-documenting.
-/
import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import DoublyEven.Vectors

namespace DoublyEven

/-- An `[N, k]` binary linear code: a rank-`k` submodule of `F_2^N`. -/
def Code (N k : ℕ) : Type :=
  { C : Submodule (ZMod 2) (BinVec N) // Module.finrank (ZMod 2) C = k }

namespace Code

variable {N k : ℕ}

instance : CoeOut (Code N k) (Submodule (ZMod 2) (BinVec N)) := ⟨Subtype.val⟩

/-- The doubly-even predicate: every codeword has Hamming weight divisible by 4. -/
def IsDoublyEven (C : Code N k) : Prop :=
  ∀ v ∈ (C : Submodule (ZMod 2) (BinVec N)), hammingWt v % 4 = 0

/-- The self-orthogonal predicate: every pair of codewords has zero inner product. -/
def IsSelfOrthogonal (C : Code N k) : Prop :=
  ∀ u ∈ (C : Submodule (ZMod 2) (BinVec N)),
  ∀ v ∈ (C : Submodule (ZMod 2) (BinVec N)),
    dot u v = 0

end Code

end DoublyEven
