//! The residual's Taylor series in a parameter: the residual on a jet state `z(h) = sum_k z_k h^k`, in the lanes
//! `P::J<N>` (multidouble's `Jet`, `QJet`, `OJet`), with the equations' parameter entered as the jet variable.
//!
//! The equations form a parameter's constants through the lane's parameter arithmetic (`Lane::P`): `Param::var(c)`
//! is c on every other lane and the series c + h on a jet, so every block's coefficients carry their own Taylor
//! coefficients in it. With z_k for k >= 1 absent (zero) the coefficient of order k is the right-hand side of order k
//! of the Taylor continuation in the parameter; with z_1 given, order 1 is J z_1 + dF/dparameter.
use crate::equations::{Equations, State};
use crate::lane::BaseLane;
use crate::shooting::Shooting;
use multidouble::field::Fl;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use rayon::prelude::*;

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// The residual's coefficients of orders 0..nout - 1 on the jet state with coefficient vectors zs (each PARTS
    /// part arrays of length `len()`, at most N of them, z_k zero past them), nout <= N, one scalar per row. The
    /// extra rows are linear: order 0 is `sum w_i z_0,i - rhs`, order k >= 1 `sum w_i z_k,i`.
    pub fn residual_jet<P: Precision<Native>, const N: usize>(
        &self,
        zs: &[Vec<&[f64]>],
        par: bool,
        nout: usize,
    ) -> Result<Vec<Vec<P::Scalar>>, String> {
        if zs.is_empty() || zs.len() > N || nout == 0 || nout > N {
            return Err(format!(
                "residual_jet: 1..={N} coefficient vectors and 1..={N} outputs, not {} and {nout}",
                zs.len()
            ));
        }
        for z in zs {
            self.check_z::<P>(z)?;
        }
        let m = Self::M;
        let fld_k = |k: usize, o: usize| -> Fl<P::V, B> {
            if k < zs.len() {
                self.load::<P>(&zs[k], o)
            } else {
                Fl::zero()
            }
        };
        let fld_jet = |o: usize| -> Fl<P::J<N>, B> {
            let mut f = Fl::<P::J<N>, B>::zero();
            for k in 0..zs.len() {
                P::field_set_coef(&mut f, k, &fld_k(k, o));
            }
            f
        };
        let og = self.off_global();
        let g: Vec<P::J<N>> = (0..E::NG)
            .map(|i| {
                let mut gi = <P::J<N> as BaseLane>::zero();
                for (k, z) in zs.iter().enumerate() {
                    P::set_coef(&mut gi, k, P::splat(self.scalar::<P>(z, og + i)));
                }
                gi
            })
            .collect();
        let prep = self.prep(&zs[0])?;
        let one = |b: usize| -> State<P::J<N>, B> {
            let inp: Vec<Fl<P::J<N>, B>> = self.block_inputs(b).iter().map(|&o| fld_jet(o)).collect();
            self.block_state::<P::J<N>>(b, &inp, &g, &prep)
        };
        let nb = self.k() + 1;
        let states: Vec<State<P::J<N>, B>> = if par {
            (0..nb).into_par_iter().map(one).collect()
        } else {
            (0..nb).map(one).collect()
        };
        let mut out = vec![vec![P::ZERO; self.rows()]; nout];
        for (b, s) in states.iter().enumerate() {
            let t = self.block_target(b);
            let r0 = self.row_block(b);
            for q in 0..E::NF {
                for (k, ok) in out.iter_mut().enumerate() {
                    let d = P::field_coef(&s[q], k) - fld_k(k, t + q * m);
                    for i in 0..m {
                        ok[r0 + q * m + i] = P::field_get(&d, i);
                    }
                }
            }
        }
        for (k, ok) in out.iter_mut().enumerate() {
            self.project_residual::<P>(ok, |p| {
                (0..m)
                    .map(|j| {
                        if k < zs.len() {
                            self.scalar::<P>(&zs[k], p.pin + j)
                        } else {
                            P::ZERO
                        }
                    })
                    .collect()
            });
        }
        let re = self.row_extra();
        for (k, ok) in out.iter_mut().enumerate() {
            for (r, row) in self.extra.iter().enumerate() {
                ok[re + r] = if k == 0 {
                    self.extra_row::<P>(&zs[0], r)
                } else if k < zs.len() {
                    P::dot(
                        row.terms
                            .iter()
                            .map(|&(i, w)| (P::from_dd(w), self.scalar::<P>(&zs[k], i))),
                    )
                } else {
                    P::ZERO
                };
            }
            self.scale_rows::<P>(ok)?;
        }
        Ok(out)
    }
}
