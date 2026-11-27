import Multishoot.Binary64
import Multishoot.Machin

/-!
# `PI_OD` is the greedy round-to-nearest expansion of `π`

`PI_OD` in src/periodic.rs holds eight binary64 parts `p_0, …, p_7`. Part `i` is the binary64 number nearest
`π - (p_0 + ⋯ + p_{i-1})`: `p_0 = RN(π)`, and each later part rounds what the parts before it leave. The parts are
written here as `m_i/2^{k_i}`. The kernel checks `2^52 < |m_i| < 2^53`, and that `p_0 + ⋯ + p_i` lies within half of
`2^{-k_i}` of both ends of `π`'s enclosure, so of `π`. The source writes parts 1 to 7 as decimal literals, and each
literal rounds to its part.
-/

namespace Multishoot

open Finset

/-- the parts' 53-bit integers -/
def piM : ℕ → ℤ
  | 0 => 7074237752028440
  | 1 => 4967757600021511
  | 2 => -8753721960665020
  | 3 => 5857755168774013
  | 4 => 5380502254069925
  | 5 => 4770902957472735
  | 6 => 5939234543911707
  | 7 => 6779061338189706
  | _ => 0

/-- the parts' exponents, negated: `p_i = m_i/2^{k_i}`, of ulp `2^{-k_i}` -/
def piK : ℕ → ℕ
  | 0 => 51
  | 1 => 105
  | 2 => 161
  | 3 => 215
  | 4 => 269
  | 5 => 327
  | 6 => 382
  | 7 => 437
  | _ => 0

/-- part `i` of `PI_OD` -/
def piPart (i : ℕ) : ℚ := piM i / 2 ^ piK i

/-- `p_0 + ⋯ + p_{n-1}` -/
def piSum : ℕ → ℚ
  | 0 => 0
  | n + 1 => piSum n + piPart n

theorem cast_piSum (n : ℕ) : ((piSum n : ℚ) : ℝ) = ∑ j ∈ range n, ((piPart j : ℚ) : ℝ) := by
  induction n with
  | zero => simp [piSum]
  | succ n ih => rw [sum_range_succ, ← ih, piSum, Rat.cast_add]

theorem cast_piPart (i : ℕ) : ((piPart i : ℚ) : ℝ) = (piM i : ℝ) * 2 ^ (-(piK i : ℤ)) := by
  rw [piPart, zpow_neg, zpow_natCast]
  push_cast
  ring

/-- half the ulp of part `i` -/
theorem half_ulp (i : ℕ) : (2 : ℝ) ^ (-(piK i : ℤ)) / 2 = ((1 / 2 ^ (piK i + 1) : ℚ) : ℝ) := by
  rw [zpow_neg, zpow_natCast]
  push_cast
  ring

theorem pi_od_check : ∀ i < 8, 2 ^ 52 < |piM i| ∧ |piM i| < 2 ^ 53 ∧ piK i ≤ 1074 ∧
    piSum (i + 1) - 1 / 2 ^ (piK i + 1) < piLo ∧ piHi < piSum (i + 1) + 1 / 2 ^ (piK i + 1) := by
  decide +kernel

/-- `π - p_0 - ⋯ - p_i` is below half the ulp of `p_i` -/
theorem pi_od_remainder (i : ℕ) (hi : i < 8) :
    |Real.pi - ((piSum (i + 1) : ℚ) : ℝ)| < ((1 / 2 ^ (piK i + 1) : ℚ) : ℝ) := by
  obtain ⟨-, -, -, s1, s2⟩ := pi_od_check i hi
  obtain ⟨l, u⟩ := pi_enclosure
  have s1' := Rat.cast_lt (K := ℝ).mpr s1
  have s2' := Rat.cast_lt (K := ℝ).mpr s2
  rw [Rat.cast_sub] at s1'
  rw [Rat.cast_add] at s2'
  rw [abs_lt]
  constructor <;> linarith

/-- part `i` of `PI_OD` is the binary64 number nearest `π` less the parts before it -/
theorem pi_od_greedy (i : ℕ) (hi : i < 8) :
    IsNearest (Real.pi - ∑ j ∈ range i, ((piPart j : ℚ) : ℝ)) (piPart i) := by
  obtain ⟨m1, m2, k1, -, -⟩ := pi_od_check i hi
  rw [cast_piPart]
  refine isNearest_of_half_ulp m1 m2 (by omega) (by omega) ?_
  rw [half_ulp, sub_sub, ← cast_piPart, ← sum_range_succ, ← cast_piSum]
  exact pi_od_remainder i hi

/-- the eight parts sum to `π` within `2^{-438}`, half the ulp of the last -/
theorem pi_od_error : |Real.pi - ∑ j ∈ range 8, ((piPart j : ℚ) : ℝ)| < 1 / 2 ^ 438 := by
  have h := pi_od_remainder 7 (by norm_num)
  rw [cast_piSum] at h
  norm_num [piK] at h ⊢
  exact h

/-! ## The source's decimal literals -/

/-- the digits of the decimal literals of parts 1 to 7 -/
def piDecN : ℕ → ℤ
  | 1 => 12246467991473532
  | 2 => -29947698097183397
  | 3 => 11124542208633653
  | 4 => 5672231979640316
  | 5 => 17449862161352486
  | 6 => 602937273224954
  | 7 => 191012354687999
  | _ => 0

/-- their scales: the literal of part `i` is `N_i/10^{K_i}` -/
def piDecK : ℕ → ℕ
  | 1 => 32
  | 2 => 49
  | 3 => 65
  | 4 => 81
  | 5 => 99
  | 6 => 114
  | 7 => 130
  | _ => 0

/-- the decimal literal of part `i` -/
def piDec (i : ℕ) : ℚ := piDecN i / 10 ^ piDecK i

theorem pi_dec_check : ∀ i < 8, 0 < i → |piDec i - piPart i| < 1 / 2 ^ (piK i + 1) := by
  decide +kernel

/-- the decimal literal of part `i`, `1 ≤ i ≤ 7`, rounds to it -/
theorem pi_od_literal (i : ℕ) (hi : i < 8) (h0 : 0 < i) : IsNearest (piDec i) (piPart i) := by
  obtain ⟨m1, m2, k1, -, -⟩ := pi_od_check i hi
  rw [cast_piPart]
  refine isNearest_of_half_ulp m1 m2 (by omega) (by omega) ?_
  rw [half_ulp, ← cast_piPart, ← Rat.cast_sub, ← Rat.cast_abs]
  exact Rat.cast_lt.mpr (pi_dec_check i hi h0)

end Multishoot
