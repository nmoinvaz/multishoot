//! The mode matrix: the linearization of the shooting system about a background, for perturbations that grow as
//! exp(lambda t) in a variable t of the equations, and its determinant, whose zeros in lambda are the growth rates
//! of the background's modes.
//!
//! A perturbation exp(lambda t) g, with g of the background's form (periodic in t, say), has the t-derivative
//! exp(lambda t) (g_t + lambda g): every t-derivative of a perturbation gains lambda times the perturbation, while
//! the background's values are unchanged. The equations mark their t-derivatives with [`Rate::shift`], in the Taylor
//! recursion through `Work::rate` and in the endpoint maps through their preparation
//! ([`Equations::prepare_rate`]). The engine evaluates the linearization with complex tangents (multidouble's
//! `Dc`), so lambda may be complex.
//!
//! The mode matrix M(lambda) is that linearization over the block rows and the unknowns that are not globals: the
//! globals (an eigenvalue, a period) and the extra rows (phase conditions) belong to the background and are held
//! fixed. At lambda = 0 it is the Jacobian's leading block. det M(lambda) vanishes exactly where a mode
//! exp(lambda t) g solves the linearized problem, so the winding number of det M along a closed contour counts the
//! modes inside it (the argument principle).
use crate::equations::{Equations, State};
use crate::jacobian::Seed;
use crate::lane::{BaseLane, Dc, Du, Lane, DV};
use crate::shooting::Shooting;
use multidouble::dd::DD;
use multidouble::field::Fl;
use multidouble::linalg::{clu_logdet, ld_for};
use multidouble::precision::Double;
use rayon::prelude::*;
use std::any::{Any, TypeId};

/// The growth rate lambda = re + i im of a mode-matrix evaluation, zero for the residual, the Jacobian and the jets
/// (the default)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rate {
    pub re: DD,
    pub im: DD,
}

impl Rate {
    pub const ZERO: Rate = Rate {
        re: DD::ZERO,
        im: DD::ZERO,
    };

    /// the rate re + i im
    pub fn new(re: DD, im: DD) -> Self {
        Rate { re, im }
    }

    /// whether the rate is zero (both parts' leading words)
    pub fn is_zero(&self) -> bool {
        self.re.hi == 0.0 && self.im.hi == 0.0
    }

    /// The t-derivative dx of the field x, shifted for the rate: dx plus lambda times x's tangent, its value
    /// unchanged. It is dx itself on lanes without a tangent and at rate zero, bit for bit. A real rate shifts a
    /// double-double tangent (`Du`) and a complex one a complex tangent (`Dc`: re += lr re - li im,
    /// im += lr im + li re). Every other lane with a nonzero rate panics: the mode matrix is in double-double.
    pub fn shift<T: Lane, const B: usize>(&self, x: &Fl<T, B>, dx: Fl<T, B>) -> Fl<T, B> {
        if self.is_zero() || !T::TANGENT {
            return dx;
        }
        let mut out = dx;
        if let Some(oc) = (&mut out as &mut dyn Any).downcast_mut::<Fl<Dc, B>>() {
            let xc = (x as &dyn Any)
                .downcast_ref::<Fl<Dc, B>>()
                .expect("x and dx have one lane type");
            let (a, b) = (DV::splat(self.re), DV::splat(self.im));
            for k in 0..B {
                let t = xc.0[k];
                oc.0[k].r = oc.0[k].r + (t.r * a - t.i * b);
                oc.0[k].i = oc.0[k].i + (t.i * a + t.r * b);
            }
            return out;
        }
        assert!(
            TypeId::of::<T>() == TypeId::of::<Du>() && self.im.hi == 0.0,
            "Rate::shift: a real rate needs Du lanes and a complex one Dc lanes"
        );
        out + Fl::from_vt(&Fl::zero(), &x.tan().muls(self.re))
    }
}

/// The mode matrix M(lambda), n x n complex double-double, column-major with leading dimension ld: part q (hi, lo)
/// of the real part of entry (i, j) at `re[q][j ld + i]`, of the imaginary part at `im[q][j ld + i]`. Row i is
/// residual row i and column j unknown j, for the block rows and the unknowns before the globals.
#[derive(Clone, Debug)]
pub struct ModeMatrix {
    pub n: usize,
    pub ld: usize,
    pub re: Vec<Vec<f64>>,
    pub im: Vec<Vec<f64>>,
}

impl ModeMatrix {
    /// entry (i, j): (real part, imaginary part)
    pub fn at(&self, i: usize, j: usize) -> (DD, DD) {
        let k = j * self.ld + i;
        (
            DD {
                hi: self.re[0][k],
                lo: self.re[1][k],
            },
            DD {
                hi: self.im[0][k],
                lo: self.im[1][k],
            },
        )
    }

    /// (ln|det M|, arg det M in (-pi, pi]) by multidouble's complex double-double LU (`clu_logdet`); the sums of
    /// ln|u_kk| and arg u_kk are formed in f64
    pub fn logdet(self, par: bool) -> Result<(f64, f64), String> {
        let [rh, rl]: [Vec<f64>; 2] = self.re.try_into().map_err(|_| "ModeMatrix: 2 real parts".to_string())?;
        let [ih, il]: [Vec<f64>; 2] = self
            .im
            .try_into()
            .map_err(|_| "ModeMatrix: 2 imaginary parts".to_string())?;
        let (la, arg, _, _) = clu_logdet(self.n, self.ld, rh, rl, ih, il, par)?;
        Ok((la, arg))
    }
}

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// the rate the engine hands the equations: the given one, its real part under `Control::RateReal`
    fn rate_used(&self, rate: Rate) -> Rate {
        #[cfg(feature = "mutate")]
        if self.control == crate::control::Control::RateReal {
            return Rate::new(rate.re, DD::ZERO);
        }
        rate
    }

    /// The mode matrix at the background z (double-double, 2 part arrays of length `len()`) and the growth rate:
    /// the block rows' derivatives along every unknown before the globals, each column one evaluation with complex
    /// tangents (`Dc`, a real unit tangent on the column's unknown; one per input field for a pointwise problem),
    /// the globals held at z, the Taylor steps given the rate (`Work::rate`) and the endpoint maps the preparation of
    /// `Equations::prepare_rate`; with `row_scale` each row is scaled as the Jacobian's. It needs as many block rows as
    /// such unknowns (NL + NR = NF).
    pub fn mode_matrix(&self, z: &[&[f64]], rate: Rate, par: bool) -> Result<ModeMatrix, String> {
        self.check_z::<Double>(z)?;
        let n = self.off_global();
        if self.row_extra() != n {
            return Err(format!(
                "mode_matrix: {} block rows for {n} unknowns before the globals",
                self.row_extra()
            ));
        }
        let m = Self::M;
        let ld = ld_for(n);
        let rate = self.rate_used(rate);
        let g: Vec<Dc> = self.globals::<Double>(z).into_iter().map(Dc::splat).collect();
        #[cfg(feature = "mutate")]
        let prep = if self.control == crate::control::Control::RateNotPrepared {
            self.eq.prepare(z[0], z[1])
        } else {
            self.eq.prepare_rate(z[0], z[1], rate)
        };
        #[cfg(not(feature = "mutate"))]
        let prep = self.eq.prepare_rate(z[0], z[1], rate);
        #[cfg(feature = "mutate")]
        let steps = if self.control == crate::control::Control::RateNotInSteps {
            Rate::ZERO
        } else {
            rate
        };
        #[cfg(not(feature = "mutate"))]
        let steps = rate;
        let tasks: Vec<(usize, Seed)> = self
            .jac_tasks()
            .into_iter()
            .filter(|t| !matches!(t.1, Seed::Global(_)))
            .collect();
        let column = |&(b, seed): &(usize, Seed)| -> Vec<(DD, DD)> {
            let inp: Vec<Fl<Dc, B>> = self
                .block_inputs(b)
                .iter()
                .enumerate()
                .map(|(fi, &o)| {
                    let t = match seed {
                        Seed::Field(f, j) if f == fi => Fl::<DV, B>::unit(j),
                        Seed::Points(f) if f == fi => Fl::<DV, B>::splat(DD::ONE),
                        _ => Fl::zero(),
                    };
                    Fl::from_vt(&self.load::<Double>(z, o), &t)
                })
                .collect();
            let s: State<Dc, B> = self.block_state_at::<Dc>(b, &inp, &g, &prep, steps);
            let mut out = Vec::with_capacity(E::NF * m);
            for sq in &s {
                let (tr, ti) = (sq.tan_re(), sq.tan_im());
                for i in 0..m {
                    out.push((tr.get(i), ti.get(i)));
                }
            }
            out
        };
        let cols: Vec<Vec<(DD, DD)>> = if par {
            tasks.par_iter().map(column).collect()
        } else {
            tasks.iter().map(column).collect()
        };
        let mut re = vec![vec![0.0; ld * n], vec![0.0; ld * n]];
        let mut im = vec![vec![0.0; ld * n], vec![0.0; ld * n]];
        let nf = E::NF;
        for (&(b, seed), col) in tasks.iter().zip(&cols) {
            let r0 = self.row_block(b);
            for (jj, cj) in self.seed_columns(b, seed).into_iter().enumerate() {
                #[cfg(feature = "mutate")]
                if self.control == crate::control::Control::DropColumn(cj) {
                    continue;
                }
                // a pointwise task's column jj has point jj of every output field; any other task's column is whole
                let rows: Vec<usize> = if let Seed::Points(_) = seed {
                    (0..nf).map(|q| q * m + jj).collect()
                } else {
                    (0..nf * m).collect()
                };
                for r in rows {
                    let k = cj * ld + r0 + r;
                    let (vr, vi) = col[r];
                    (re[0][k], re[1][k]) = (vr.hi, vr.lo);
                    (im[0][k], im[1][k]) = (vi.hi, vi.lo);
                }
            }
        }
        for b in 0..self.k() + 1 {
            let t = self.block_target(b);
            let r0 = self.row_block(b);
            for r in 0..nf * m {
                let k = (t + r) * ld + r0 + r;
                let v = DD {
                    hi: re[0][k],
                    lo: re[1][k],
                } - DD::ONE;
                (re[0][k], re[1][k]) = (v.hi, v.lo);
            }
        }
        if !self.row_scale.is_empty() {
            if self.row_scale.len() != self.rows() {
                return Err(format!(
                    "row_scale has {} factors for {} rows",
                    self.row_scale.len(),
                    self.rows()
                ));
            }
            for (i, &sr) in self.row_scale.iter().enumerate().take(n) {
                let s = DD::from_f64(sr);
                for j in 0..n {
                    let k = j * ld + i;
                    for part in [&mut re, &mut im] {
                        let v = DD {
                            hi: part[0][k],
                            lo: part[1][k],
                        } * s;
                        (part[0][k], part[1][k]) = (v.hi, v.lo);
                    }
                }
            }
        }
        Ok(ModeMatrix { n, ld, re, im })
    }

    /// (ln|det M(lambda)|, arg det M(lambda)) of [`Shooting::mode_matrix`], by [`ModeMatrix::logdet`]
    pub fn mode_logdet(&self, z: &[&[f64]], rate: Rate, par: bool) -> Result<(f64, f64), String> {
        self.mode_matrix(z, rate, par)?.logdet(par)
    }
}
