//! Series at a regular singular point whose coefficients are periodic in theta: each order is one periodic linear
//! system, solved in every lane type the engine evaluates.
//!
//! A field f(x, theta) = sum_k f_k(theta) x^k that is slaved at a regular singular point x = 0 meets at order k a
//! system
//!
//! ```text
//! L_k f_k = (omega D + diag kappa_k) f_k = r_k,
//! ```
//!
//! D the derivative of the field's parity sector on the M points ([`crate::periodic::dmat`]), omega a frequency, and
//! kappa_k a field: the indicial term k times a field (x d/dx gives k) plus the field's own coefficient at order 0.
//! r_k holds the orders below. The same system is a slaved field's order k when kappa depends on x: with
//! kappa(x) = kappa_0 + sum_j kappa_j x^j, order k of (omega D + kappa(x)) f = r(x) is
//! L_0 f_k = r_k - sum_{j=1..k} kappa_j f_(k-j).
//!
//! The factors are formed once per evaluation from the double-double values ([`PeriodicLu::factor`],
//! [`series_lus`], in `Equations::prepare`), and [`PeriodicLu::solve`] solves the lane's own system with them:
//!
//! * values in double-double by the factors, in quad- and octo-double refined with the residual in the lane's
//!   precision (Moler, J. ACM 14, 316 (1967));
//! * a tangent (`Du`, `Qu`, `Ou`, `Oun`, `Dc`, the f64 tangents of `Fu`, `Fun`, `Dtn`) by the differentiated system,
//!   L f_d = r_d - (omega_d D + diag kappa_d) f_v;
//! * a jet (`Jet`, `QJet`, `OJet`) order by order, L f_j = r_j - sum_{i=1..j} (omega_i D + diag kappa_i) f_(j-i);
//! * at the growth rate lambda of a mode-matrix evaluation ([`crate::mode`]), whose t-derivative omega D shifts a
//!   tangent by lambda times itself, the tangents by L + lambda (a real rate) or by the real form of the complex
//!   system (a complex one).
//!
//! The lanes' constants that are not exact in double-double (1/k, the coefficients of a change of variable) enter
//! through [`crate::lane::LaneConst`]; the scaled variable x / 2^s that keeps a series' coefficients near one
//! (multidouble's `dd::pow2_near`) and Horner's rule (`horner`) are multidouble's.
#[cfg(feature = "mutate")]
use crate::control::Control;
use crate::lane::{DMat, Dc, Lane, DV, OV, QV};
use crate::mode::Rate;
use multidouble::dd::DD;
use multidouble::field::{FieldSolve, Fl};
use multidouble::linalg::{ld_for, DDLu};
use multidouble::native::F8;
use rayon::prelude::*;
use std::any::Any;

/// Refinement steps of a periodic solve: `qd` for quad-double values and jets, `od` for octo-double ones, one more
/// for a tangent, whose first solve leaves out the matrix's derivative. With factors to double-double each step
/// gains about cond(L) 2^-104, so the defaults reach the lane's precision where cond(L) is below about 10^10
/// (quad-double) and 10^6 (octo-double).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Refine {
    pub qd: usize,
    pub od: usize,
}

impl Default for Refine {
    fn default() -> Self {
        Refine { qd: 2, od: 4 }
    }
}

/// The factors of L = omega D + diag kappa on the M = 8 B points at the double-double values, the preconditioner of
/// the lane solves ([`PeriodicLu::solve`]), with those of the tangents' systems at a mode matrix's rate.
#[derive(Clone)]
pub struct PeriodicLu {
    /// L at the values, M x M
    pub lu: DDLu,
    /// L + re(lambda), the tangents' matrix at a real rate (`None` at rate zero and at a complex rate)
    pub lu_t: Option<DDLu>,
    /// [[L + re, -im], [im, L + re]] on (f_re, f_im), the complex tangents' matrix at a complex rate, 2M x 2M
    pub lu_c: Option<DDLu>,
    pub rate: Rate,
    pub refine: Refine,
    #[cfg(feature = "mutate")]
    control: Control,
}

/// omega D column-major with the leading dimension of the factors, hi and lo parts: the matrix every order's L adds
/// its diagonal to, formed once for a table of orders
struct Base {
    m: usize,
    ld: usize,
    hi: Vec<f64>,
    lo: Vec<f64>,
}

impl Base {
    fn new<const B: usize>(d: &DMat<B>, omega: DD) -> Self {
        let m = 8 * B;
        let ld = ld_for(m);
        let (mut hi, mut lo) = (vec![0.0; ld * m], vec![0.0; ld * m]);
        for (j, c) in d.cols.iter().enumerate() {
            c.muls(omega)
                .store(&mut hi[j * ld..j * ld + m], &mut lo[j * ld..j * ld + m]);
        }
        Base { m, ld, hi, lo }
    }

    /// the factors of the base plus diag(kappa + shift)
    fn factor(&self, kappa: &[DD], shift: DD) -> Result<DDLu, String> {
        let (m, ld) = (self.m, self.ld);
        assert_eq!(
            kappa.len(),
            m,
            "PeriodicLu: kappa has {} values for {m} points",
            kappa.len()
        );
        let (mut h, mut l) = (self.hi.clone(), self.lo.clone());
        for (i, &k) in kappa.iter().enumerate() {
            let v = DD {
                hi: h[i * ld + i],
                lo: l[i * ld + i],
            } + (k + shift);
            h[i * ld + i] = v.hi;
            l[i * ld + i] = v.lo;
        }
        DDLu::factor(m, ld, h, l, false)
    }

    /// the factors of the real 2M x 2M form [[A + re, -im], [im, A + re]], A the base plus diag kappa
    fn factor_complex(&self, kappa: &[DD], rate: Rate) -> Result<DDLu, String> {
        let (m, ld) = (self.m, self.ld);
        let m2 = 2 * m;
        let ld2 = ld_for(m2);
        let (mut h, mut l) = (vec![0.0; ld2 * m2], vec![0.0; ld2 * m2]);
        for j in 0..m {
            for i in 0..m {
                let (a, b) = (self.hi[j * ld + i], self.lo[j * ld + i]);
                h[j * ld2 + i] = a;
                l[j * ld2 + i] = b;
                h[(j + m) * ld2 + i + m] = a;
                l[(j + m) * ld2 + i + m] = b;
            }
            h[(j + m) * ld2 + j] = -rate.im.hi;
            l[(j + m) * ld2 + j] = -rate.im.lo;
            h[j * ld2 + j + m] = rate.im.hi;
            l[j * ld2 + j + m] = rate.im.lo;
        }
        for (i, &k) in kappa.iter().enumerate() {
            for ii in [i, i + m] {
                let v = DD {
                    hi: h[ii * ld2 + ii],
                    lo: l[ii * ld2 + ii],
                } + (k + rate.re);
                h[ii * ld2 + ii] = v.hi;
                l[ii * ld2 + ii] = v.lo;
            }
        }
        DDLu::factor(m2, ld2, h, l, false)
    }

    /// the factors at kappa and at the rate's tangent systems
    fn lu(&self, kappa: &[DD], rate: Rate) -> Result<PeriodicLu, String> {
        let lu = self.factor(kappa, DD::ZERO)?;
        let (lu_t, lu_c) = if rate.im.hi != 0.0 {
            (None, Some(self.factor_complex(kappa, rate)?))
        } else if rate.re.hi != 0.0 {
            (Some(self.factor(kappa, rate.re)?), None)
        } else {
            (None, None)
        };
        Ok(PeriodicLu {
            lu,
            lu_t,
            lu_c,
            rate,
            refine: Refine::default(),
            #[cfg(feature = "mutate")]
            control: Control::None,
        })
    }
}

/// The factors of the orders' systems L_k = omega D + diag kappa(k), k = 0..=order, at the double-double values, the
/// diagonals from `kappa(k)` (M values each), at the rate (zero but for a mode-matrix evaluation), the orders on
/// rayon tasks under `par`. omega D is formed once. An error names the order whose factorization failed.
pub fn series_lus<const B: usize>(
    d: &DMat<B>,
    omega: DD,
    order: usize,
    kappa: impl Fn(usize) -> Vec<DD> + Sync,
    rate: Rate,
    par: bool,
) -> Result<Vec<PeriodicLu>, String> {
    let base = Base::new(d, omega);
    let one = |k: usize| {
        base.lu(&kappa(k), rate)
            .map_err(|e| format!("series_lus: order {k}: {e}"))
    };
    if par {
        (0..=order).into_par_iter().map(one).collect()
    } else {
        (0..=order).map(one).collect()
    }
}

impl PeriodicLu {
    /// the factors of omega D + diag kappa (M values) at rate zero
    pub fn factor<const B: usize>(d: &DMat<B>, omega: DD, kappa: &[DD]) -> Result<Self, String> {
        Self::factor_rate(d, omega, kappa, Rate::ZERO)
    }

    /// the factors of omega D + diag kappa and of its tangents' systems at the rate
    pub fn factor_rate<const B: usize>(d: &DMat<B>, omega: DD, kappa: &[DD], rate: Rate) -> Result<Self, String> {
        Base::new(d, omega).lu(kappa, rate)
    }

    /// Sets the deliberate error of the solves ([`Control::PeriodicCoupling`], [`Control::RateUnshifted`],
    /// [`Control::RateResidual`]; `Control::None` to clear it), for a control.
    #[cfg(feature = "mutate")]
    pub fn set_control(&mut self, c: Control) {
        self.control = c;
    }

    /// whether the control c is set
    #[cfg(feature = "mutate")]
    fn is(&self, c: Control) -> bool {
        self.control == c
    }

    /// whether the right-hand sides of tangents and jet orders keep the matrix's derivative (not under
    /// `Control::PeriodicCoupling`)
    fn coupled(&self) -> bool {
        #[cfg(feature = "mutate")]
        if self.is(Control::PeriodicCoupling) {
            return false;
        }
        true
    }

    /// whether the tangents are solved by the factors at rate zero (`Control::RateUnshifted`)
    fn unshifted(&self) -> bool {
        #[cfg(feature = "mutate")]
        if self.is(Control::RateUnshifted) {
            return true;
        }
        false
    }

    /// the rate of the refinement residual: the factors' own, zero under `Control::RateResidual`
    fn residual_rate(&self) -> Rate {
        #[cfg(feature = "mutate")]
        if self.is(Control::RateResidual) {
            return Rate::ZERO;
        }
        self.rate
    }

    /// The solution f of (omega D + diag kappa) f = rhs in the lane T, omega and kappa the lane's (with their
    /// tangents or series), these factors the double-double values': refined to quad- or octo-double on those lanes,
    /// a tangent's system differentiated, a jet's orders in turn, at the rate of the factors (the t-derivative omega
    /// D shifted by `Rate::shift`). d is the matrix the factors were formed from.
    ///
    /// The values' solve reads only the leading two parts of omega and kappa through the factors, so a lane whose
    /// omega or kappa differs from the factored ones by more than the double-double rounding needs the refinement's
    /// steps to absorb it: the quad- and octo-double lanes refine, the double-double lanes take the factored matrix as
    /// theirs. A jet or an f64 lane at a nonzero rate panics (`Rate::shift` admits neither).
    pub fn solve<T: Lane, const B: usize>(&self, d: &DMat<B>, omega: T, kappa: &Fl<T, B>, rhs: &Fl<T, B>) -> Fl<T, B> {
        let lu = &self.lu;
        let lu_t = match &self.lu_t {
            Some(t) if !self.unshifted() => t,
            _ => lu,
        };
        let rate = self.residual_rate();
        let coupled = self.coupled();
        if T::NJ > 0 {
            assert!(self.rate.is_zero(), "PeriodicLu::solve: a jet at a nonzero rate");
        }
        if T::NJ > 0 && T::VAL_OD {
            let om: Vec<OV> = (0..T::NJ).map(|j| omega.coef_od(j)).collect();
            let k0 = kappa.coef_od(0);
            return lu.solve_lanes_jet_refine_od(
                rhs,
                self.refine.od,
                |j, x, r| {
                    if coupled {
                        (r - d.apply(x).mul_t(om[j])) - kappa.coef_od(j) * *x
                    } else {
                        r
                    }
                },
                |r, x| *r - (d.apply(x).mul_t(om[0]) + k0 * *x),
            );
        }
        if T::VAL_OD {
            return lu.solve_lanes_refine_t(lu_t, rhs, self.refine.od + T::TANGENT as usize, |f| {
                *rhs - (rate.shift(f, d.apply(f).mul_t(omega)) + *kappa * *f)
            });
        }
        if T::NJ > 0 && T::VAL_QD {
            let om: Vec<QV> = (0..T::NJ).map(|j| omega.coef_qd(j)).collect();
            let k0 = kappa.coef_qd(0);
            return lu.solve_lanes_jet_refine(
                rhs,
                self.refine.qd,
                |j, x, r| {
                    if coupled {
                        (r - d.apply(x).mul_t(om[j])) - kappa.coef_qd(j) * *x
                    } else {
                        r
                    }
                },
                |r, x| *r - (d.apply(x).mul_t(om[0]) + k0 * *x),
            );
        }
        if T::VAL_QD {
            return lu.solve_lanes_refine_t(lu_t, rhs, self.refine.qd + T::TANGENT as usize, |f| {
                *rhs - (rate.shift(f, d.apply(f).mul_t(omega)) + *kappa * *f)
            });
        }
        if T::NJ > 0 {
            return lu.solve_lanes_jet(rhs, |j, x, r| {
                if !coupled {
                    return r;
                }
                let dfv = d.apply::<DV>(x);
                let od = Fl::<DV, B>([omega.coef(j); B]);
                (r - dfv * od) - kappa.coef(j) * *x
            });
        }
        if T::VAL_F64 || T::TAN_F64 {
            // f64 values by the factors' hi parts, double-double values (Dtn) by the factors; f64 tangents by the
            // hi parts, each tangent's differentiated system
            let fv = if T::VAL_F64 {
                lu.solve_lanes_hi(&rhs.val())
            } else {
                lu.solve_lanes(&rhs.val())
            };
            let z = Fl::<DV, B>::zero();
            let mut out = Fl::<T, B>::from_vt(&fv, &z);
            if !T::TANGENT {
                return out;
            }
            let fv8 = Fl::<F8, B>::from_vt(&fv, &z);
            let dfv = d.apply(&fv8);
            for k in 0..T::NT {
                let od = multidouble::dual::F8(omega.tan64(k));
                let r = if coupled {
                    (rhs.tan64(k) - dfv.mul_t(od)) - kappa.tan64(k) * fv8
                } else {
                    rhs.tan64(k)
                };
                let fd = lu_t.solve_lanes_hi(&r.val());
                for b in 0..B {
                    out.0[b].set_tan64(k, fd.0[b].hi);
                }
            }
            return out;
        }
        if T::TANGENT {
            if let Some(f) = self.solve_dc(d, lu_t, omega, kappa, rhs, coupled) {
                return f;
            }
        }
        let fv = lu.solve_lanes(&rhs.val());
        if !T::TANGENT {
            return Fl::from_vt(&fv, &Fl::zero());
        }
        let r = if coupled {
            let dfv = d.apply::<DV>(&fv);
            let od = Fl::<DV, B>([omega.tan(); B]);
            (rhs.tan() - dfv * od) - kappa.tan() * fv
        } else {
            rhs.tan()
        };
        Fl::from_vt(&fv, &lu_t.solve_lanes(&r))
    }

    /// `solve` on complex tangents (`Dc`, the mode matrix's lanes; `None` for any other lane): the value by the
    /// factors, the real and imaginary tangents by their differentiated systems, with L + re at a real rate and the
    /// 2M x 2M real form at a complex one
    fn solve_dc<T: Lane, const B: usize>(
        &self,
        d: &DMat<B>,
        lu_t: &DDLu,
        omega: T,
        kappa: &Fl<T, B>,
        rhs: &Fl<T, B>,
        coupled: bool,
    ) -> Option<Fl<T, B>> {
        let kappa = (kappa as &dyn Any).downcast_ref::<Fl<Dc, B>>()?;
        let rhs = (rhs as &dyn Any).downcast_ref::<Fl<Dc, B>>()?;
        let omega = (&omega as &dyn Any).downcast_ref::<Dc>()?;
        let m = 8 * B;
        let fv = self.lu.solve_lanes(&rhs.val());
        let (rr, ri) = if coupled {
            let dfv = d.apply::<DV>(&fv);
            let (or, oi) = (Fl::<DV, B>([omega.r; B]), Fl::<DV, B>([omega.i; B]));
            (
                (rhs.tan_re() - dfv * or) - kappa.tan_re() * fv,
                (rhs.tan_im() - dfv * oi) - kappa.tan_im() * fv,
            )
        } else {
            (rhs.tan_re(), rhs.tan_im())
        };
        let out = match &self.lu_c {
            Some(lu2) if !self.unshifted() => {
                let (mut hh, mut ll) = (vec![0.0; m], vec![0.0; m]);
                let mut b = vec![DD::ZERO; 2 * m];
                for (part, f) in [&rr, &ri].into_iter().enumerate() {
                    f.store(&mut hh, &mut ll);
                    for i in 0..m {
                        b[part * m + i] = DD { hi: hh[i], lo: ll[i] };
                    }
                }
                lu2.solve(&mut b);
                let part = |o: usize| {
                    let (h, l): (Vec<f64>, Vec<f64>) = b[o..o + m].iter().map(|v| (v.hi, v.lo)).unzip();
                    Fl::<DV, B>::load(&h, &l)
                };
                Fl::<Dc, B>::from_vri(&fv, &part(0), &part(m))
            }
            _ => Fl::<Dc, B>::from_vri(&fv, &lu_t.solve_lanes(&rr), &lu_t.solve_lanes(&ri)),
        };
        (&out as &dyn Any).downcast_ref::<Fl<T, B>>().copied()
    }

    /// `solve` for the coefficient of order j of a jet lane alone, into coefficient j of f, whose coefficients below
    /// j hold the orders solved before: r_j = rhs_j - sum_{i=1..j} (omega_i D + diag kappa_i) f_(j-i), then L f_j = r_j
    /// refined as `solve` refines each order, so that every coefficient is `solve`'s bit for bit. For a residual's
    /// series formed one order at a time (`Equations::ORDER_BY_ORDER`); jet lanes at rate zero only.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_at<T: Lane, const B: usize>(
        &self,
        d: &DMat<B>,
        omega: &T,
        kappa: &Fl<T, B>,
        rhs: &Fl<T, B>,
        f: &mut Fl<T, B>,
        j: usize,
    ) {
        assert!(
            T::NJ > 0 && self.rate.is_zero(),
            "PeriodicLu::solve_at: jet lanes at rate zero"
        );
        let lu = &self.lu;
        let coupled = self.coupled();
        if T::VAL_OD {
            let mut r = rhs.coef_od(j);
            if coupled {
                for i in 1..=j {
                    let x = f.coef_od(j - i);
                    r = (r - d.apply(&x).mul_t(omega.coef_od(i))) - kappa.coef_od(i) * x;
                }
            }
            let (om0, k0) = (omega.coef_od(0), kappa.coef_od(0));
            let x = lu.solve_lanes_refine(&r, self.refine.od, |y| r - (d.apply(y).mul_t(om0) + k0 * *y));
            f.set_coef_od(j, &x);
        } else if T::VAL_QD {
            let mut r = rhs.coef_qd(j);
            if coupled {
                for i in 1..=j {
                    let x = f.coef_qd(j - i);
                    r = (r - d.apply(&x).mul_t(omega.coef_qd(i))) - kappa.coef_qd(i) * x;
                }
            }
            let (om0, k0) = (omega.coef_qd(0), kappa.coef_qd(0));
            let x = lu.solve_lanes_refine(&r, self.refine.qd, |y| r - (d.apply(y).mul_t(om0) + k0 * *y));
            f.set_coef_qd(j, &x);
        } else {
            let mut r = rhs.coef(j);
            if coupled {
                for i in 1..=j {
                    let x = f.coef(j - i);
                    let dfv = d.apply::<DV>(&x);
                    let od = Fl::<DV, B>([omega.coef(i); B]);
                    r = (r - dfv * od) - kappa.coef(i) * x;
                }
            }
            f.set_coef(j, &lu.solve_lanes(&r));
        }
    }
}
