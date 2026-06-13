/-
Binary vectors over F_2 = ZMod 2.

A `BinVec N` is just `Fin N → ZMod 2`; we keep it as an `abbrev` so that
Mathlib's `Pi`-instance machinery applies transparently.
-/
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.BigOperators.Basic

namespace DoublyEven

/-- Binary vectors of length `N`. -/
abbrev BinVec (N : ℕ) : Type := Fin N → ZMod 2

/-- Hamming weight: number of nonzero coordinates. -/
def hammingWt {N : ℕ} (v : BinVec N) : ℕ :=
  (Finset.univ.filter (fun i => v i ≠ 0)).card

/-- F_2 inner product. -/
def dot {N : ℕ} (u v : BinVec N) : ZMod 2 :=
  ∑ i, u i * v i

@[simp] lemma hammingWt_zero {N : ℕ} : hammingWt (0 : BinVec N) = 0 := by
  unfold hammingWt
  simp

@[simp] lemma dot_zero_left {N : ℕ} (v : BinVec N) : dot 0 v = 0 := by
  unfold dot
  simp

@[simp] lemma dot_zero_right {N : ℕ} (v : BinVec N) : dot v 0 = 0 := by
  unfold dot
  simp

end DoublyEven
