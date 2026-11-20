//! The residual F(z) in any of multidouble's precisions (`Precision`): every block's state from its inputs, minus its target node, then the
//! extra rows. F is zero exactly when the pieces join into one solution that meets the extra conditions.
//!
//! The blocks are independent, so with `par` each runs on its own rayon task; the result does not depend on it.
use crate::equations::{Equations, State};
use crate::shooting::Shooting;
use multidouble::field::Fl;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use rayon::prelude::*;

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// Checks that z has PARTS part arrays of the problem's length.
    pub fn check_z<P: Precision<Native>>(&self, z: &[&[f64]]) -> Result<(), String> {
        let l = self.len();
        if z.len() != P::PARTS || z.iter().any(|p| p.len() != l) {
            return Err(format!("z must be {} part arrays of length {l}", P::PARTS));
        }
        Ok(())
    }

    /// the number of parts of z that are read: all of them, or half under `Control::HalfParts`
    fn kept(&self, parts: usize) -> usize {
        #[cfg(feature = "mutate")]
        if self.control == crate::control::Control::HalfParts {
            return parts / 2;
        }
        parts
    }

    /// The field of z at offset off, its parts as they are
    pub fn load<P: Precision<Native>>(&self, z: &[&[f64]], off: usize) -> Fl<P::V, B> {
        let m = Self::M;
        let keep = self.kept(z.len());
        if keep == z.len() {
            let parts: Vec<&[f64]> = z.iter().map(|p| &p[off..off + m]).collect();
            return P::load_field(&parts);
        }
        let zero = vec![0.0; m];
        let parts: Vec<&[f64]> = z
            .iter()
            .enumerate()
            .map(|(q, p)| if q < keep { &p[off..off + m] } else { &zero[..] })
            .collect();
        P::load_field(&parts)
    }

    /// Unknown i of z, its parts as they are and zero past the kept ones, as `load` reads a field: the extra rows, the
    /// globals and the pinned unknowns see the same value of an unknown as the blocks do, normalized or not
    pub fn scalar<P: Precision<Native>>(&self, z: &[&[f64]], i: usize) -> P::Scalar {
        let keep = self.kept(z.len()).min(z.len());
        let parts: Vec<f64> = (0..P::PARTS).map(|q| if q < keep { z[q][i] } else { 0.0 }).collect();
        #[cfg(feature = "mutate")]
        if self.control == crate::control::Control::RenormScalar {
            return P::new(&parts);
        }
        P::from_raw(&parts)
    }

    /// The globals of z
    pub fn globals<P: Precision<Native>>(&self, z: &[&[f64]]) -> Vec<P::Scalar> {
        (0..E::NG).map(|i| self.scalar::<P>(z, self.off_global() + i)).collect()
    }

    /// The states of every block at z, in value lanes, or the error of z's check or of its preparation
    /// (`Equations::try_prepare`)
    pub fn try_block_states<P: Precision<Native>>(
        &self,
        z: &[&[f64]],
        par: bool,
    ) -> Result<Vec<State<P::V, B>>, String> {
        self.check_z::<P>(z)?;
        let g: Vec<P::V> = self.globals::<P>(z).into_iter().map(P::splat).collect();
        let prep = self.prep(z)?;
        let one = |b: usize| -> State<P::V, B> {
            let inp: Vec<Fl<P::V, B>> = self.block_inputs(b).iter().map(|&o| self.load::<P>(z, o)).collect();
            self.block_state::<P::V>(b, &inp, &g, &prep)
        };
        let nb = self.k() + 1;
        Ok(if par {
            (0..nb).into_par_iter().map(one).collect()
        } else {
            (0..nb).map(one).collect()
        })
    }

    /// The states of every block at z, in value lanes (`try_block_states`). Panics when the preparation fails.
    pub fn block_states<P: Precision<Native>>(&self, z: &[&[f64]], par: bool) -> Vec<State<P::V, B>> {
        self.try_block_states::<P>(z, par)
            .unwrap_or_else(|e| panic!("block_states: {e}"))
    }

    /// Extra row r at z: `sum_i w_i z_(idx_i) - rhs`
    pub fn extra_row<P: Precision<Native>>(&self, z: &[&[f64]], r: usize) -> P::Scalar {
        let row = &self.extra[r];
        let s = P::dot(row.terms.iter().map(|&(i, w)| (P::from_dd(w), self.scalar::<P>(z, i))));
        if row.rhs.hi == 0.0 && row.rhs.lo == 0.0 {
            s
        } else {
            s - P::from_dd(row.rhs)
        }
    }

    /// An a-posteriori estimate of the Taylor paths' truncation error: the residual at z with every segment's path
    /// recomputed with each step halved (twice the steps at the same order), and per residual block (0..=K, the
    /// endpoints' blocks first and last) the largest change of its rows, |F(z) - F_halved(z)| in hi parts. With an
    /// error of order h^(kt+1) per step the halved path's error is about 2^-kt of the original's, so the change
    /// estimates the original path's error; the endpoint maps take no steps and show zero. `segs` is restored. A step
    /// too short to halve in f64 (its midpoint rounds to an end) is an error, not a zero estimate.
    pub fn taylor_check<P: Precision<Native>>(&mut self, z: &[&[f64]], par: bool) -> Result<Vec<f64>, String> {
        let mut fine: Vec<Vec<f64>> = Vec::with_capacity(self.segs.len());
        for (b, s) in self.segs.iter().enumerate() {
            let mut out = Vec::with_capacity(2 * s.len());
            for w in s.windows(2) {
                let mid = 0.5 * (w[0] + w[1]);
                if !(mid > w[0].min(w[1]) && mid < w[0].max(w[1])) {
                    return Err(format!(
                        "taylor_check: the step {} to {} of segment {} has no midpoint in f64",
                        w[0],
                        w[1],
                        b + 1
                    ));
                }
                out.push(w[0]);
                out.push(mid);
            }
            out.extend(s.last());
            fine.push(out);
        }
        let f0 = self.residual::<P>(z, par)?;
        let saved = std::mem::replace(&mut self.segs, fine);
        let f1 = self.residual::<P>(z, par);
        self.segs = saved;
        let f1 = f1?;
        let rows = E::NF * Self::M;
        Ok((0..=self.k())
            .map(|b| {
                (self.row_block(b)..self.row_block(b) + rows)
                    .map(|i| P::to_dd(f0[i] - f1[i]).hi.abs())
                    .fold(0.0, multisolver::prec::nan_max)
            })
            .collect())
    }

    /// The residual at z (PARTS part arrays of length `len()`), one scalar per row.
    pub fn residual<P: Precision<Native>>(&self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        self.check_z::<P>(z)?;
        let m = Self::M;
        let states = self.try_block_states::<P>(z, par)?;
        let mut f = vec![P::ZERO; self.rows()];
        let nb = E::NF * m;
        for (b, s) in states.iter().enumerate() {
            let r0 = self.row_block(b);
            f[r0..r0 + nb].copy_from_slice(&self.block_rows::<P>(z, b, s));
        }
        self.project_residual::<P>(&mut f, |p| self.pinned::<P>(z, p));
        let re = self.row_extra();
        for r in 0..self.extra.len() {
            f[re + r] = self.extra_row::<P>(z, r);
        }
        self.scale_rows::<P>(&mut f)?;
        Ok(f)
    }

    /// Block b's NF M rows of the residual from its state s: the state minus its target node's fields (row r matched
    /// to the unknown after its target under `Control::ShiftRow(r)`)
    pub(crate) fn block_rows<P: Precision<Native>>(
        &self,
        z: &[&[f64]],
        b: usize,
        s: &State<P::V, B>,
    ) -> Vec<P::Scalar> {
        let m = Self::M;
        let t = self.block_target(b);
        let mut f = Vec::with_capacity(E::NF * m);
        for (q, sq) in s.iter().enumerate().take(E::NF) {
            let d = *sq - self.load::<P>(z, t + q * m);
            f.extend((0..m).map(|i| P::field_get(&d, i)));
            #[cfg(feature = "mutate")]
            if let crate::control::Control::ShiftRow(r) = self.control {
                let r0 = self.row_block(b);
                if r >= r0 + q * m && r < r0 + (q + 1) * m {
                    let i = r - r0 - q * m;
                    f[q * m + i] = P::field_get(sq, i) - self.scalar::<P>(z, t + q * m + i + 1);
                }
            }
        }
        f
    }

    /// The values in z of a projection's M pinned unknowns, as the residual's projected rows read them
    pub(crate) fn pinned<P: Precision<Native>>(
        &self,
        z: &[&[f64]],
        p: &crate::projected::ProjectedRows,
    ) -> Vec<P::Scalar> {
        (0..Self::M).map(|j| self.scalar::<P>(z, p.pin + j)).collect()
    }

    /// The residual's factor of a row whose `row_scale` entry is s
    pub(crate) fn row_factor<P: Precision<Native>>(s: f64) -> P::Scalar {
        P::from_dd(multidouble::dd::DD::from_f64(s))
    }

    /// Each row times its `row_scale` factor (nothing when there are none)
    pub(crate) fn scale_rows<P: Precision<Native>>(&self, f: &mut [P::Scalar]) -> Result<(), String> {
        if self.row_scale.is_empty() {
            return Ok(());
        }
        if self.row_scale.len() != f.len() {
            return Err(format!(
                "row_scale has {} factors for {} rows",
                self.row_scale.len(),
                f.len()
            ));
        }
        for (v, &s) in f.iter_mut().zip(&self.row_scale) {
            *v = *v * Self::row_factor::<P>(s);
        }
        Ok(())
    }
}
