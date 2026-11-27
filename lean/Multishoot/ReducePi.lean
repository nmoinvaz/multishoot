import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# The reduction of `(p/q)π` in `sin_cos_pi`

`reduce_pi` in src/periodic.rs takes `p` modulo `2q`, a period, then `n = ⌊(4p + q)/(2q)⌋`, the nearest multiple of
`π/2`, and returns `(n, 2p - nq, 2q)`; `quadrant` maps sin and cos of the remainder to those of the angle. Here, for
`q > 0`: `(p/q)π = nπ/2 + (2p - nq)π/(2q)` modulo `2π`, with `0 ≤ n ≤ 4` and the remainder in `[-π/4, π/4)`, and
`quadrant` gives sin and cos of `(p/q)π` exactly from those of the remainder.
-/

namespace Multishoot

open Real

/-- `reduce_pi(p, q)`: for a positive divisor, Rust's `rem_euclid` and `div_euclid` are `%` and `/` on `ℤ` -/
def reducePi (p q : ℤ) : ℤ × ℤ × ℤ :=
  (((4 * (p % (2 * q)) + q) / (2 * q)), 2 * (p % (2 * q)) - (4 * (p % (2 * q)) + q) / (2 * q) * q, 2 * q)

/-- `quadrant(n, (s, c))`: sin and cos of `r + nπ/2` from `s = sin r` and `c = cos r` -/
noncomputable def quadrant (n : ℤ) (s c : ℝ) : ℝ × ℝ :=
  if n % 4 = 0 then (s, c) else if n % 4 = 1 then (c, -s) else if n % 4 = 2 then (-s, -c) else (-c, s)

/-- `0 ≤ n ≤ 4`, `-q ≤ 2 num < q` and the divisor `2q` -/
theorem reducePi_bounds (p q : ℤ) (hq : 0 < q) :
    0 ≤ (reducePi p q).1 ∧ (reducePi p q).1 ≤ 4 ∧ -q ≤ 2 * (reducePi p q).2.1 ∧ 2 * (reducePi p q).2.1 < q ∧
      (reducePi p q).2.2 = 2 * q := by
  have h2q : 0 < 2 * q := by omega
  have r0 : 0 ≤ p % (2 * q) := Int.emod_nonneg p h2q.ne'
  have r1 : p % (2 * q) < 2 * q := Int.emod_lt_of_pos p h2q
  have a := Int.ediv_mul_le (4 * (p % (2 * q)) + q) h2q.ne'
  have b := Int.lt_ediv_add_one_mul_self (4 * (p % (2 * q)) + q) h2q
  have n0 : 0 ≤ (4 * (p % (2 * q)) + q) / (2 * q) := Int.ediv_nonneg (by omega) h2q.le
  have n4 : (4 * (p % (2 * q)) + q) / (2 * q) ≤ 4 := by
    by_contra h
    nlinarith
  refine ⟨n0, n4, ?_, ?_, rfl⟩ <;> dsimp only [reducePi] <;> nlinarith

/-- `(p/q)π = nπ/2 + num π/(2q) + 2kπ` with `k = ⌊p/(2q)⌋` -/
theorem reducePi_angle (p q : ℤ) (hq : 0 < q) :
    (p / q : ℝ) * π = (reducePi p q).1 * (π / 2) + (reducePi p q).2.1 * π / (reducePi p q).2.2 +
      (p / (2 * q) : ℤ) * (2 * π) := by
  have hp : ((2 * q * (p / (2 * q)) + p % (2 * q) : ℤ) : ℝ) = p := by rw [Int.mul_ediv_add_emod]
  have hq' : (q : ℝ) ≠ 0 := by exact_mod_cast hq.ne'
  simp only [reducePi]
  rw [← hp]
  push_cast
  field_simp
  ring

/-- sin and cos of `r + nπ/2` -/
theorem quadrant_eq (n : ℤ) (r : ℝ) : quadrant n (sin r) (cos r) = (sin (r + n * (π / 2)), cos (r + n * (π / 2))) := by
  have hn : (n : ℝ) * (π / 2) = (n % 4 : ℤ) * (π / 2) + (n / 4 : ℤ) * (2 * π) := by
    have : ((4 * (n / 4) + n % 4 : ℤ) : ℝ) = n := by rw [Int.mul_ediv_add_emod]
    rw [← this]
    push_cast
    ring
  have hs : sin (r + n * (π / 2)) = sin (r + (n % 4 : ℤ) * (π / 2)) := by
    rw [hn, ← add_assoc, sin_add_int_mul_two_pi]
  have hc : cos (r + n * (π / 2)) = cos (r + (n % 4 : ℤ) * (π / 2)) := by
    rw [hn, ← add_assoc, cos_add_int_mul_two_pi]
  rw [hs, hc]
  have h0 : 0 ≤ n % 4 := Int.emod_nonneg n (by norm_num)
  have h4 : n % 4 < 4 := Int.emod_lt_of_pos n (by norm_num)
  unfold quadrant
  rcases (by omega : n % 4 = 0 ∨ n % 4 = 1 ∨ n % 4 = 2 ∨ n % 4 = 3) with h | h | h | h <;> rw [h]
  · rw [ite_eq_left rfl]
    simp
  · rw [ite_eq_right (show (1 : ℤ) ≠ 0 by decide), ite_eq_left rfl, Int.cast_one, one_mul, sin_add_pi_div_two,
      cos_add_pi_div_two]
  · rw [ite_eq_right (show (2 : ℤ) ≠ 0 by decide), ite_eq_right (show (2 : ℤ) ≠ 1 by decide), ite_eq_left rfl,
      show ((2 : ℤ) : ℝ) * (π / 2) = π by push_cast; ring, sin_add_pi, cos_add_pi]
  · rw [ite_eq_right (show (3 : ℤ) ≠ 0 by decide), ite_eq_right (show (3 : ℤ) ≠ 1 by decide),
      ite_eq_right (show (3 : ℤ) ≠ 2 by decide), show ((3 : ℤ) : ℝ) * (π / 2) = π / 2 + π by push_cast; ring,
      ← add_assoc, sin_add_pi, cos_add_pi, sin_add_pi_div_two, cos_add_pi_div_two, neg_neg]

/-- `sin_cos_pi`'s reduction: sin and cos of `(p/q)π` are `quadrant` of those of `r = num π/(2q)`, `|r| ≤ π/4` -/
theorem sin_cos_reducePi (p q : ℤ) (hq : 0 < q) :
    let r := (reducePi p q).2.1 * π / (reducePi p q).2.2
    (sin ((p / q : ℝ) * π), cos ((p / q : ℝ) * π)) = quadrant (reducePi p q).1 (sin r) (cos r) ∧
      -(π / 4) ≤ r ∧ r < π / 4 := by
  intro r
  obtain ⟨-, -, b1, b2, b3⟩ := reducePi_bounds p q hq
  refine ⟨?_, ?_, ?_⟩
  · rw [quadrant_eq, reducePi_angle p q hq, add_comm ((reducePi p q).1 * (π / 2) : ℝ), sin_add_int_mul_two_pi,
      cos_add_int_mul_two_pi, add_comm]
  all_goals
    have hq' : (0 : ℝ) < 2 * q := by exact_mod_cast (by omega : 0 < 2 * q)
    have b1' : (-q : ℝ) ≤ 2 * (reducePi p q).2.1 := by exact_mod_cast b1
    have b2' : (2 * (reducePi p q).2.1 : ℝ) < q := by exact_mod_cast b2
    simp only [r, b3]
    push_cast
  · rw [le_div_iff₀ hq']
    nlinarith [pi_pos]
  · rw [div_lt_iff₀ hq']
    nlinarith [pi_pos]

end Multishoot
