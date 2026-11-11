//! The Jacobian-vector product J v without the Jacobian: one forward-mode pass per block (Wengert, Commun. ACM 7,
//! 463 (1964)), with the block's inputs and the globals carrying v's entries as their tangents, in any precision.
//!
//! Block b's rows of J v are the tangent of its state along v minus v at its target node; the extra rows are linear,
//! `sum_i w_i v_i`. A product costs K + 1 dual evaluations, about as many as K + 1 Jacobian columns, against one per
//! unknown for the Jacobian, so a Krylov solver or a directional derivative needs no matrix. The residual comes with
//! it, from the values of the same pass.
use crate::equations::{Equations, State};
use crate::shooting::Shooting;
use multidouble::field::Fl;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use rayon::prelude::*;

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// the field of v at offset off as a seed (`load`), without the point of unknown c under `Control::DropColumn(c)`
    fn seed_field<P: Precision<Native>>(&self, v: &[&[f64]], off: usize) -> Fl<P::V, B> {
        #[cfg(feature = "mutate")]
        if let crate::control::Control::DropColumn(c) = self.control {
            if c >= off && c < off + Self::M {
                let held: Vec<Vec<f64>> = v
                    .iter()
                    .map(|p| {
                        let mut q = p[off..off + Self::M].to_vec();
                        q[c - off] = 0.0;
                        q
                    })
                    .collect();
                let parts: Vec<&[f64]> = held.iter().map(|p| p.as_slice()).collect();
                return self.load::<P>(&parts, 0);
            }
        }
        self.load::<P>(v, off)
    }

    /// unknown i of v as a seed (`scalar`), zero under `Control::DropColumn(i)`
    fn seed_scalar<P: Precision<Native>>(&self, v: &[&[f64]], i: usize) -> P::Scalar {
        #[cfg(feature = "mutate")]
        if self.control == crate::control::Control::DropColumn(i) {
            return P::ZERO;
        }
        self.scalar::<P>(v, i)
    }

    /// The product J v and the residual F at z, one scalar per row each: (J v, F). z and v are PARTS part arrays of
    /// length `len()`. Every block runs once in the dual lanes `P::U` (`Du`, `Qu`, `Ou`), each input field with v's
    /// values at its offsets as its tangent and each global with v's entry; with `par` each block on its own rayon
    /// task, the result not depending on it. J v equals the exact Jacobian times v to the precision's rounding, and F
    /// is the residual's value.
    #[allow(clippy::type_complexity)]
    pub fn jvp<P: Precision<Native>>(
        &self,
        z: &[&[f64]],
        v: &[&[f64]],
        par: bool,
    ) -> Result<(Vec<P::Scalar>, Vec<P::Scalar>), String> {
        self.check_z::<P>(z)?;
        self.check_z::<P>(v)?;
        let m = Self::M;
        let og = self.off_global();
        let g: Vec<P::U> = (0..E::NG)
            .map(|i| {
                P::dual(
                    P::splat(self.scalar::<P>(z, og + i)),
                    P::splat(self.seed_scalar::<P>(v, og + i)),
                )
            })
            .collect();
        let prep = self.eq.prepare(z[0], z[1]);
        let one = |b: usize| -> State<P::U, B> {
            let inp: Vec<Fl<P::U, B>> = self
                .block_inputs(b)
                .iter()
                .map(|&o| P::field_dual(&self.load::<P>(z, o), &self.seed_field::<P>(v, o)))
                .collect();
            self.block_state::<P::U>(b, &inp, &g, &prep)
        };
        let nb = self.k() + 1;
        // the blocks are few (K + 1) and unequal; with par, each runs on its own task
        let states: Vec<State<P::U, B>> = if par {
            (0..nb).into_par_iter().map(one).collect()
        } else {
            (0..nb).map(one).collect()
        };
        let mut jv = vec![P::ZERO; self.rows()];
        let mut f = vec![P::ZERO; self.rows()];
        for (b, s) in states.iter().enumerate() {
            let t = self.block_target(b);
            let r0 = self.row_block(b);
            for q in 0..E::NF {
                let d = P::field_coef(&s[q], 1) - self.load::<P>(v, t + q * m);
                let e = P::field_coef(&s[q], 0) - self.load::<P>(z, t + q * m);
                for i in 0..m {
                    jv[r0 + q * m + i] = P::field_get(&d, i);
                    f[r0 + q * m + i] = P::field_get(&e, i);
                }
            }
        }
        let re = self.row_extra();
        for (r, row) in self.extra.iter().enumerate() {
            jv[re + r] = P::dot(row.terms.iter().map(|&(i, w)| (P::from_dd(w), self.scalar::<P>(v, i))));
            f[re + r] = self.extra_row::<P>(z, r);
        }
        Ok((jv, f))
    }
}
