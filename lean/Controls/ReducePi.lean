import Multishoot.ReducePi

/-!
# Controls of `ReducePi.lean`

The reduction with its rounding to the nearest multiple of `π/2` made a truncation, and `quadrant` with a sign
changed: each must fail what the theorems state.
-/

namespace Multishoot

open Real

/-- truncating `2p/q` in place of rounding it leaves `(1/3)π` a remainder outside `[-π/4, π/4)` -/
example : ¬ 2 * (2 * (1 % 6) - 4 * (1 % 6) / 6 * 3) < (3 : ℤ) := by decide

/-- `(c, s)` for `n ≡ 1` is not sin and cos of `r + π/2` -/
example : ¬ ∀ r : ℝ, (cos r, sin r) = (sin (r + π / 2), cos (r + π / 2)) := by
  intro h
  have e := congrArg Prod.snd (h (π / 2))
  rw [show π / 2 + π / 2 = π by ring, cos_pi, sin_pi_div_two] at e
  norm_num at e

end Multishoot
