import Multishoot.PiOD

/-!
# Controls of `PiOD.lean`

The kernel's checks of `Machin.lean` and `PiOD.lean` with a part, a bound or a literal moved, and the kernel proves
each negation; then the theorems' own statements made false the same ways.
-/

namespace Multishoot

open Finset

/-- `π`'s quad-double, the first four parts, is not within `2^{-438}` of the enclosure -/
example : ¬ (piSum 4 - 1 / 2 ^ 438 < piLo ∧ piHi < piSum 4 + 1 / 2 ^ 438) := by decide +kernel

/-- the last part one ulp up is not within half an ulp of its remainder -/
example : ¬ (piSum 7 + (piM 7 + 1) / 2 ^ piK 7 - 1 / 2 ^ (piK 7 + 1) < piLo ∧
    piHi < piSum 7 + (piM 7 + 1) / 2 ^ piK 7 + 1 / 2 ^ (piK 7 + 1)) := by decide +kernel

/-- nor one ulp down -/
example : ¬ (piSum 7 + (piM 7 - 1) / 2 ^ piK 7 - 1 / 2 ^ (piK 7 + 1) < piLo ∧
    piHi < piSum 7 + (piM 7 - 1) / 2 ^ piK 7 + 1 / 2 ^ (piK 7 + 1)) := by decide +kernel

/-- the lower bound of `π`, `2^{-460}` up, is not below the series' lower sum -/
example : ¬ piLo + 1 / 2 ^ 460 ≤ 16 * atSQ (1 / 5) 100 - 4 * atSQ (1 / 239) 31 := by decide +kernel

/-- the upper sum with 91 terms at `1/5` is not below the upper bound -/
example : ¬ 16 * atSQ (1 / 5) 91 - 4 * atSQ (1 / 239) 30 ≤ piHi := by decide +kernel

/-- part 7's literal with its last digit one down does not round to it -/
example : ¬ |(191012354687998 : ℚ) / 10 ^ 130 - piPart 7| < 1 / 2 ^ (piK 7 + 1) := by decide +kernel

/-- the last part one ulp up is not the binary64 number nearest its remainder -/
theorem pi_od_control_ulp :
    ¬ IsNearest (Real.pi - ∑ j ∈ range 7, ((piPart j : ℚ) : ℝ)) (((piM 7 + 1 : ℤ) : ℝ) * 2 ^ (-(piK 7 : ℤ))) := by
  intro h
  have e := (pi_od_greedy 7 (by norm_num)).unique h
  rw [cast_piPart] at e
  have : (piM 7 : ℝ) = ((piM 7 + 1 : ℤ) : ℝ) := mul_right_cancel₀ (zpow_ne_zero _ two_ne_zero) e
  norm_num at this

/-- `π`'s quad-double is not within `2^{-438}` of `π` -/
theorem pi_od_control_quad : ¬ |Real.pi - ∑ j ∈ range 4, ((piPart j : ℚ) : ℝ)| < 1 / 2 ^ 438 := by
  have c : piSum 4 + 1 / 2 ^ 438 < piLo := by decide +kernel
  have c' := Rat.cast_lt (K := ℝ).mpr c
  rw [Rat.cast_add, cast_piSum, Rat.cast_div, Rat.cast_one, Rat.cast_pow, Rat.cast_ofNat] at c'
  have l := pi_enclosure.1
  rw [abs_lt, not_and_or]
  right
  -- linarith's normalization would evaluate `2^438`
  generalize (1 : ℝ) / 2 ^ 438 = t at c' ⊢
  linarith

end Multishoot
