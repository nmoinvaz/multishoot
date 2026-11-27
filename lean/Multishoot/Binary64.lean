import Mathlib.Tactic

/-!
# The binary64 numbers and the one nearest a real

A finite binary64 number is `n·2^f` with `|n| < 2^53` and `-1074 ≤ f ≤ 971`, the subnormals included. A number
`m·2^e` with `2^52 < |m| < 2^53` within half its ulp `2^e` of `x` is nearer `x` than every other binary64 number, so
round-to-nearest gives it under every tie rule.
-/

namespace Multishoot

/-- `x` is a finite binary64 number -/
def IsBinary64 (x : ℝ) : Prop :=
  ∃ n f : ℤ, |n| < 2 ^ 53 ∧ -1074 ≤ f ∧ f ≤ 971 ∧ x = n * 2 ^ f

/-- `p` is a binary64 number nearer `x` than every other: `RN(x) = p` under every tie rule -/
def IsNearest (x p : ℝ) : Prop :=
  IsBinary64 p ∧ ∀ q, IsBinary64 q → q ≠ p → |x - p| < |x - q|

/-- a binary64 number other than `m·2^e`, `2^52 < |m|`, is at least `2^e` from it -/
theorem spacing {m e n f : ℤ} (hm : 2 ^ 52 < |m|) (hn : |n| < 2 ^ 53) (hne : (n : ℝ) * 2 ^ f ≠ m * 2 ^ e) :
    (2 : ℝ) ^ e ≤ |(m : ℝ) * 2 ^ e - n * 2 ^ f| := by
  have he : (0 : ℝ) < 2 ^ e := zpow_pos two_pos e
  rcases le_or_gt e f with h | h
  -- `f ≥ e`: `n·2^f` is a multiple of `2^e`, and not `m·2^e`
  · obtain ⟨k, hk⟩ := Int.eq_ofNat_of_zero_le (sub_nonneg.mpr h)
    have h2 : (2 : ℝ) ^ f = 2 ^ k * 2 ^ e := by
      rw [← zpow_natCast, ← zpow_add₀ two_ne_zero, ← hk, sub_add_cancel]
    have hd : (m : ℝ) * 2 ^ e - n * 2 ^ f = ((m - n * 2 ^ k : ℤ) : ℝ) * 2 ^ e := by
      rw [h2]
      push_cast
      ring
    have hd0 : m - n * 2 ^ k ≠ 0 := by
      intro h0
      apply hne
      have : (m : ℝ) = n * 2 ^ k := by exact_mod_cast sub_eq_zero.mp h0
      rw [h2, this]
      ring
    have h1 : (1 : ℝ) ≤ |((m - n * 2 ^ k : ℤ) : ℝ)| := by exact_mod_cast Int.one_le_abs hd0
    rw [hd, abs_mul, abs_of_pos he]
    nlinarith
  -- `f < e`: `|n·2^f| < 2^53·2^(e-1) = 2^52·2^e < |m|·2^e`
  · have hf : (2 : ℝ) ^ f ≤ 2 ^ e / 2 := by
      have : (2 : ℝ) ^ f ≤ 2 ^ (e - 1) := zpow_le_zpow_right₀ (by norm_num) (by omega)
      rwa [zpow_sub₀ two_ne_zero, zpow_one] at this
    have hq : |(n : ℝ) * 2 ^ f| ≤ (2 ^ 53 - 1) * (2 ^ e / 2) := by
      rw [abs_mul, abs_of_pos (zpow_pos (two_pos : (0 : ℝ) < 2) f)]
      have : |(n : ℝ)| ≤ 2 ^ 53 - 1 := by exact_mod_cast Int.le_sub_one_of_lt hn
      exact mul_le_mul this hf (zpow_pos two_pos f).le (by norm_num)
    have hp : (2 ^ 52 + 1) * 2 ^ e ≤ |(m : ℝ) * 2 ^ e| := by
      rw [abs_mul, abs_of_pos he]
      have : (2 ^ 52 + 1 : ℝ) ≤ |(m : ℝ)| := by exact_mod_cast Int.add_one_le_of_lt hm
      exact mul_le_mul_of_nonneg_right this he.le
    have := abs_sub_abs_le_abs_sub ((m : ℝ) * 2 ^ e) (n * 2 ^ f)
    nlinarith

/-- `m·2^e`, `2^52 < |m| < 2^53`, within half of `2^e` of `x` is the binary64 number nearest `x` -/
theorem isNearest_of_half_ulp {x : ℝ} {m e : ℤ} (hm1 : 2 ^ 52 < |m|) (hm2 : |m| < 2 ^ 53) (he1 : -1074 ≤ e)
    (he2 : e ≤ 971) (hx : |x - m * 2 ^ e| < 2 ^ e / 2) : IsNearest x (m * 2 ^ e) := by
  refine ⟨⟨m, e, hm2, he1, he2, rfl⟩, ?_⟩
  rintro q ⟨n, f, hn, -, -, rfl⟩ hne
  have hs := spacing hm1 hn hne
  have ht := abs_sub_le ((m : ℝ) * 2 ^ e) x (n * 2 ^ f)
  rw [abs_sub_comm (m * 2 ^ e : ℝ) x] at ht
  linarith

/-- at most one binary64 number is nearest `x` -/
theorem IsNearest.unique {x p q : ℝ} (hp : IsNearest x p) (hq : IsNearest x q) : p = q := by
  by_contra h
  exact lt_asymm (hp.2 q hq.1 (Ne.symm h)) (hq.2 p hp.1 h)

end Multishoot
