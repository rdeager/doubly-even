/-
Permutation equivalence on codes.

The symmetric group `S_N` acts on `BinVec N` by permuting coordinates;
this lifts to submodules via `Submodule.map (permLinearEquiv σ)`. Two
codes are *equivalent* if some coordinate permutation maps one to the
other.
-/
import Mathlib.LinearAlgebra.Pi
import Mathlib.GroupTheory.Perm.Basic
import DoublyEven.Code

namespace DoublyEven

variable {N : ℕ}

/-- The permutation `σ` acts on `BinVec N` by `v ↦ v ∘ σ⁻¹`. Realised as a
`LinearEquiv` via Mathlib's `LinearEquiv.funCongrLeft`. -/
def permLinearEquiv (σ : Equiv.Perm (Fin N)) :
    BinVec N ≃ₗ[ZMod 2] BinVec N :=
  LinearEquiv.funCongrLeft (ZMod 2) (ZMod 2) σ.symm

/-- The image of a submodule under a coordinate permutation. -/
def permAct (σ : Equiv.Perm (Fin N))
    (C : Submodule (ZMod 2) (BinVec N)) : Submodule (ZMod 2) (BinVec N) :=
  Submodule.map (permLinearEquiv σ).toLinearMap C

namespace Code

variable {k : ℕ}

/-- Two codes are equivalent iff some coordinate permutation maps one to the
other. -/
def Equivalent (C C' : Code N k) : Prop :=
  ∃ σ : Equiv.Perm (Fin N), permAct σ (C : Submodule (ZMod 2) (BinVec N)) = C'.1

end Code

end DoublyEven
