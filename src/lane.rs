//! The lane types the engine is generic over: multidouble's `dual::Lane` (values, tangents, jets) with its
//! `field::MatLane` (the product with a constant matrix such as d/dtheta), on the compile-time backend `Native`.
//!
//! Every lane type of multidouble's is one: `DV` (double-double values), `Du` (value and one tangent), `Dc` (value
//! and a complex tangent), `QV`, `Qu`, `OV`, `Ou` (quad- and octo-double), `Jet<N>`, `QJet<N>`, `OJet<N>` (truncated
//! power series in a parameter). The equations are written once over [`Lane`], and the engine runs them with values,
//! exact derivatives or series.
//!
//! Constants enter the lanes in double-double (`Fl::muls`, `Lane::P`). A constant that is not exact in double-double
//! and multiplies a quad- or octo-double lane is a [`LaneConst`]: held in octo-double, applied in the lane's own
//! precision.
use multidouble::dd::DD;
pub use multidouble::dual::{Lane as BaseLane, Param};
use multidouble::field::Fl;
pub use multidouble::field::MatLane;
pub use multidouble::native::{DMat, Dc, Dtn, Du, Fu, Fun, Jet, OJet, Ou, QJet, Qu, DV, OV, QV};
use multidouble::od::OD;
use multidouble::qd::QD;
use multidouble::simd::Native;

/// multidouble's lanes on the compile-time backend, with the product by a constant double-double matrix
pub trait Lane: MatLane<S = Native> {}

impl<T: MatLane<S = Native>> Lane for T {}

/// A real constant held as the eight parts of an octo-double and applied in each lane's own precision: the leading
/// two parts on the f64 and double-double lanes (an f64 value reads the first), four on the quad-double lanes, all
/// eight on the octo-double lanes, a jet's coefficients one by one. A double-double constant (`Fl::muls`) caps a
/// quad- or octo-double lane at 2^-106 relative wherever the constant is not exact in double-double (1/3, 1/x0, the
/// coefficients of a change of variable).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaneConst(pub [f64; 8]);

/// 2^e, exactly, for e in the normal range
fn pow2(e: i64) -> f64 {
    assert!((-1022..=1023).contains(&e), "pow2: 2^{e} is not a normal f64");
    f64::from_bits(((e + 1023) as u64) << 52)
}

impl LaneConst {
    /// the octo-double v
    pub fn new(v: OD) -> Self {
        LaneConst(v.x)
    }

    /// an f64, exactly
    pub fn from_f64(x: f64) -> Self {
        LaneConst([x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }

    /// a double-double, exactly
    pub fn from_dd(x: DD) -> Self {
        LaneConst([x.hi, x.lo, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }

    /// 1/k, 1 <= k < 2^60, in eight parts by greedy round-to-nearest: r_0 = RN(1/k), r_i = RN(1/k - r_0 - ... -
    /// r_(i-1)), each part rounded once from the exact remainder N / (k 2^E) in integers. A prefix of P parts is
    /// then 1/k rounded to P parts, the nearest double-double for P = 2.
    pub fn recip(k: u64) -> Self {
        assert!((1..1u64 << 60).contains(&k), "LaneConst::recip: k = {k}");
        let d = k as u128;
        let kb = 128 - d.leading_zeros() as i64;
        let mut out = [0.0; 8];
        let (mut n, mut neg, mut e) = (1u128, false, 0i64);
        for r in out.iter_mut() {
            if n == 0 {
                break;
            }
            // n 2^s / k in [2^52, 2^53): n <= k, so s >= 52 and n 2^s < 2^(kb + 53)
            let mut s = 52 - ((128 - n.leading_zeros() as i64) - kb);
            let mut a = n << s;
            if a / d < 1 << 52 {
                s += 1;
                a <<= 1;
            }
            let q0 = a / d;
            let rem = a - q0 * d;
            let q = q0 + (2 * rem > d || (2 * rem == d && q0 & 1 == 1)) as u128;
            let m = q as f64 * pow2(-(e + s));
            *r = if neg { -m } else { m };
            // the remainder (a - q k) / (k 2^(e + s)), of either sign
            let rr = a as i128 - (q * d) as i128;
            neg ^= rr < 0;
            n = rr.unsigned_abs();
            e += s;
        }
        LaneConst(out)
    }

    /// the constant in octo-double
    pub fn od(&self) -> OD {
        OD { x: self.0 }
    }

    /// its leading two parts
    pub fn dd(&self) -> DD {
        DD {
            hi: self.0[0],
            lo: self.0[1],
        }
    }

    /// the parts a lane of type T reads: 8 octo-double, 4 quad-double, 1 an f64 value, else 2
    pub fn parts<T: Lane>() -> usize {
        if T::VAL_OD {
            8
        } else if T::VAL_QD {
            4
        } else if T::VAL_F64 {
            1
        } else {
            2
        }
    }

    /// The constant as a lane scalar with zero tangent: the pairs of parts splat as double-doubles and added in the
    /// lane's arithmetic, (r_0 + r_1) + (r_2 + r_3) and on the octo-double lanes then + (r_4 + r_5) + (r_6 + r_7),
    /// sums that are exact where the parts do not overlap.
    pub fn lane<T: Lane>(&self) -> T {
        let r = &self.0;
        let dd = |i: usize| T::splat(DD { hi: r[i], lo: r[i + 1] });
        if T::VAL_OD {
            ((dd(0) + dd(2)) + dd(4)) + dd(6)
        } else {
            dd(0) + dd(2)
        }
    }

    /// x times the constant: `muls` of its leading pair on the f64 and double-double lanes, the lane's own product
    /// with [`LaneConst::lane`] on the quad- and octo-double lanes (`mul_t`, which scales the tangents by the same
    /// constant), and coefficient by coefficient in quad- or octo-double on their jets
    #[inline]
    pub fn mul<T: Lane, const B: usize>(&self, x: Fl<T, B>) -> Fl<T, B> {
        let r = &self.0;
        if Self::parts::<T>() <= 2 {
            return x.muls(self.dd());
        }
        if T::NJ == 0 {
            return x.mul_t(self.lane::<T>());
        }
        let mut y = x;
        if T::VAL_OD {
            let c = OV::splat(OD { x: *r });
            for (yb, xb) in y.0.iter_mut().zip(&x.0) {
                for i in 0..T::NJ {
                    yb.set_coef_od(i, xb.coef_od(i) * c);
                }
            }
        } else {
            let c = QV::splat(QD {
                x: [r[0], r[1], r[2], r[3]],
            });
            for (yb, xb) in y.0.iter_mut().zip(&x.0) {
                for i in 0..T::NJ {
                    yb.set_coef_qd(i, xb.coef_qd(i) * c);
                }
            }
        }
        y
    }

    /// x plus the constant, [`LaneConst::lane`] at every point
    #[inline]
    pub fn add<T: Lane, const B: usize>(&self, x: Fl<T, B>) -> Fl<T, B> {
        x + Fl::<T, B>([self.lane::<T>(); B])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use multidouble::exact::Dy;

    use std::cmp::Ordering;

    /// whether the parts p are 1/k rounded greedily: each nonzero part's remainder 1/k - (p_0 + ... + p_i) at most
    /// half an ulp of p_i, exactly (`Dy`), and zero after the first zero part
    fn greedy_recip(p: &[f64; 8], k: u64) -> bool {
        // k exactly, as the sum of its nearest f64 and the remainder
        let hi = k as f64;
        let kd = Dy::sum(&[hi, (k as i128 - hi as i128) as f64]);
        (0..8).all(|i| {
            let err = Dy::from_f64(1.0).sub(&Dy::sum(&p[..=i]).mul(&kd));
            if p[i] == 0.0 {
                return err.is_zero();
            }
            let ulp = p[i].abs().next_up() - p[i].abs();
            err.cmp_abs(&Dy::from_f64(0.5 * ulp).mul(&kd)) != Ordering::Greater
        })
    }

    /// 1/k in eight parts for k = 1..2000 and three large k: pass, every prefix is 1/k rounded greedily (its
    /// remainder at most half an ulp of its last part, exactly), so that a lane of P parts reads 1/k rounded to P.
    #[test]
    fn recip_parts_are_one_over_k() {
        let bad: Vec<u64> = (1..2000)
            .chain([3_000_017, 1 << 40, (1 << 59) + 7])
            .filter(|&k| !greedy_recip(&LaneConst::recip(k).0, k))
            .collect();
        assert!(bad.is_empty(), "not greedy at k {bad:?}");
    }

    /// The control of `recip_parts_are_one_over_k`: the eighth part one ulp up must fail at every odd k in 3..200,
    /// the second part one ulp up likewise.
    #[test]
    fn recip_parts_control_one_ulp_fails() {
        for i in [1, 7] {
            let passed = (3..200u64).step_by(2).find(|&k| {
                let mut p = LaneConst::recip(k).0;
                p[i] = p[i].next_up();
                greedy_recip(&p, k)
            });
            assert_eq!(passed, None, "control did not fire, part {i}");
        }
    }
}
