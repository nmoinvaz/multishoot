import Mathlib.Analysis.SpecialFunctions.Complex.Arctan
import Mathlib.Analysis.SpecificLimits.Normed

/-!
# `π` to 470 bits

Machin's formula `π = 16 arctan(1/5) - 4 arctan(1/239)` with the series of `arctan`, whose partial sums of an even
and an odd number of terms bracket it for `0 ≤ x < 1`. The kernel sums 100 and 101 terms at `1/5` and 30 and 31 at
`1/239` over `ℚ` and checks that the bracket lies within `[piLo, piHi]`, a width of `800·2^{-480}`.
-/

namespace Multishoot

open Finset Filter Topology

/-- the term `x^(2i+1)/(2i+1)` of the series of `arctan x` -/
noncomputable def atT (x : ℝ) (i : ℕ) : ℝ := x ^ (2 * i + 1) / ((2 * i + 1 : ℕ) : ℝ)

theorem atT_antitone {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1) : Antitone (atT x) := by
  refine antitone_nat_of_succ_le fun i => ?_
  unfold atT
  exact div_le_div₀ (pow_nonneg h0 _) (pow_le_pow_of_le_one h0 h1 (by omega)) (by positivity)
    (by push_cast; linarith)

/-- partial sums of an even and an odd number of terms bracket `arctan x` -/
theorem arctan_bracket {x : ℝ} (h0 : 0 ≤ x) (h1 : x < 1) (k : ℕ) :
    ∑ i ∈ range (2 * k), (-1) ^ i * atT x i ≤ Real.arctan x ∧
      Real.arctan x ≤ ∑ i ∈ range (2 * k + 1), (-1) ^ i * atT x i := by
  have hs := (Real.hasSum_arctan (x := x) (by rw [Real.norm_eq_abs, abs_of_nonneg h0]; exact h1)).tendsto_sum_nat
  have he : (fun n => ∑ i ∈ range n, (-1) ^ i * atT x i) =
      fun n => ∑ i ∈ range n, (-1 : ℝ) ^ i * x ^ (2 * i + 1) / ((2 * i + 1 : ℕ) : ℝ) := by
    funext n
    refine sum_congr rfl fun i _ => ?_
    unfold atT
    ring
  rw [← he] at hs
  exact ⟨(atT_antitone h0 h1.le).alternating_series_le_tendsto hs k,
    (atT_antitone h0 h1.le).tendsto_le_alternating_series hs k⟩

/-- `Σ_{i < n} (-1)^i x^(2i+1)/(2i+1)` over `ℚ` -/
def atSQ (x : ℚ) : ℕ → ℚ
  | 0 => 0
  | n + 1 => atSQ x n + (-1) ^ n * x ^ (2 * n + 1) / (2 * n + 1)

theorem cast_atSQ (x : ℚ) (n : ℕ) : ((atSQ x n : ℚ) : ℝ) = ∑ i ∈ range n, (-1) ^ i * atT (x : ℝ) i := by
  induction n with
  | zero => simp [atSQ]
  | succ n ih =>
    rw [sum_range_succ, ← ih, atSQ, atT]
    push_cast
    ring

/-- a lower bound of `π` -/
def piLo : ℚ :=
  9807262312027308209670433188918835507998242348779285001939787782752841194690228188362411542853276791620015003000463704887612936232611341761041077 / 2 ^ 480

/-- an upper bound of `π` -/
def piHi : ℚ :=
  9807262312027308209670433188918835507998242348779285001939787782752841194690228188362411542853276791620015003000463704887612936232611341761041877 / 2 ^ 480

theorem pi_check : piLo ≤ 16 * atSQ (1 / 5) 100 - 4 * atSQ (1 / 239) 31 ∧
    16 * atSQ (1 / 5) 101 - 4 * atSQ (1 / 239) 30 ≤ piHi := by
  decide +kernel

/-- `π` within `800·2^{-480}` -/
theorem pi_enclosure : ((piLo : ℚ) : ℝ) ≤ Real.pi ∧ Real.pi ≤ ((piHi : ℚ) : ℝ) := by
  have hm := Real.four_mul_arctan_inv_5_sub_arctan_inv_239
  obtain ⟨a1, a2⟩ := arctan_bracket (x := 1 / 5) (by norm_num) (by norm_num) 50
  obtain ⟨b1, b2⟩ := arctan_bracket (x := 1 / 239) (by norm_num) (by norm_num) 15
  obtain ⟨c1, c2⟩ := pi_check
  have c1' := Rat.cast_le (K := ℝ).mpr c1
  have c2' := Rat.cast_le (K := ℝ).mpr c2
  push_cast [cast_atSQ] at c1' c2'
  norm_num at a1 a2 b1 b2 hm ⊢
  constructor <;> linarith

end Multishoot
