//! The layout as a family of systems in one of its globals, for multisolver's continuation through folds
//! ([`multisolver::arclength`]): the parameter p is global g, z the other unknowns, and the extra row that fixes g is
//! dropped, so that F(z, p) = 0 is square in z.
//!
//! [`InGlobal`] implements multisolver's [`Family`]. F(z, p) is the layout's residual with p written into global g,
//! less the dropped row; dF/dz is the layout's Jacobian less that row and g's column, entry for entry; and dF/dp is one
//! J v with v the unit vector on g, which needs no Jacobian. A continuation then follows the branch of the layout's
//! own discrete system.
//!
//! The corrector's bordered matrix [[J, F_p], [t^T]] is the layout's Jacobian with the dropped row replaced by the
//! tangent t, its rows and columns reordered: g's column is F_p, and the row t^T. [`InGlobal::bordered_lu`] factors it
//! in that form, in f64 from the f64-tangent lanes (`Shooting::jacobian_f64`). At a simple fold J alone is singular
//! and the bordered matrix is not, so it is factored whole rather than by elimination against factors of J.
use crate::equations::Equations;
use crate::lane::Lane;
use crate::shooting::Shooting;
use multidouble::dd::DD;
use multidouble::linalg::{ld_for, DDLu};
use multidouble::precision::Precision;
use multidouble::simd::Native;
use multisolver::arclength::Bordered;
use multisolver::newton::Factored;
use multisolver::prec::parts;
use multisolver::Family;

/// A layout as a family in its global g, the extra row that fixes g dropped ([`Shooting::in_global`]). z holds the
/// layout's unknowns in their order without g's, and F's rows are the layout's rows in their order without the
/// dropped one.
pub struct InGlobal<'a, E: Equations<B>, const B: usize> {
    sh: &'a Shooting<E, B>,
    /// the parameter's unknown in the layout, `off_global() + g`
    col: usize,
    /// the dropped row of the layout, `row_extra() + pin`
    row: usize,
}

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// The layout as a family in its global g ([`InGlobal`]), with the extra row `pin` (an index into `extra`), the
    /// row that fixes g, dropped. The layout must be square, g below `NG` and pin below `extra.len()`.
    pub fn in_global(&self, g: usize, pin: usize) -> Result<InGlobal<'_, E, B>, String> {
        if !self.is_square() {
            return Err(format!("in_global: {} rows for {} unknowns", self.rows(), self.len()));
        }
        if g >= E::NG || pin >= self.extra.len() {
            return Err(format!(
                "in_global: global {g} of {} and extra row {pin} of {}",
                E::NG,
                self.extra.len()
            ));
        }
        Ok(InGlobal {
            sh: self,
            col: self.off_global() + g,
            row: self.row_extra() + pin,
        })
    }
}

impl<E: Equations<B>, const B: usize> InGlobal<'_, E, B> {
    /// the layout
    pub fn shooting(&self) -> &Shooting<E, B> {
        self.sh
    }

    /// the parameter's unknown in the layout and the dropped row: (col, row)
    pub fn col_row(&self) -> (usize, usize) {
        (self.col, self.row)
    }

    /// the family's unknowns z, one fewer than the layout's
    fn n(&self) -> usize {
        self.sh.len() - 1
    }

    /// The layout's state from the family's z (P::PARTS part arrays of `len()`) and p: p's parts written into global
    /// g's place
    pub fn layout_z<P: Precision<Native>>(&self, z: &[&[f64]], p: P::Scalar) -> Result<Vec<Vec<f64>>, String> {
        let n = self.n();
        if z.len() != P::PARTS || z.iter().any(|q| q.len() != n) {
            return Err(format!(
                "InGlobal: z of {} part arrays of lengths {:?}, not {} of {n}",
                z.len(),
                z.iter().map(|q| q.len()).collect::<Vec<_>>(),
                P::PARTS
            ));
        }
        Ok(z.iter()
            .enumerate()
            .map(|(q, zq)| {
                let mut v = Vec::with_capacity(n + 1);
                v.extend_from_slice(&zq[..self.col]);
                v.push(P::part(p, q));
                v.extend_from_slice(&zq[self.col..]);
                v
            })
            .collect())
    }

    /// The family's z and p from a layout state (P::PARTS part arrays of the layout's `len()`)
    pub fn family_z<P: Precision<Native>>(&self, z: &[&[f64]]) -> (Vec<Vec<f64>>, P::Scalar) {
        let p: Vec<f64> = z.iter().map(|q| q[self.col]).collect();
        let zf = z
            .iter()
            .map(|q| q[..self.col].iter().chain(&q[self.col + 1..]).copied().collect())
            .collect();
        (zf, P::from_raw(&p))
    }

    /// the layout's column read as the family's column j: j's own, past g's; under `Control::FamilyColumn` the column
    /// before g's is skipped instead, so that g's column stands among z's
    fn layout_col(&self, j: usize) -> usize {
        #[allow(unused_mut)]
        let mut skip = self.col;
        #[cfg(feature = "mutate")]
        if self.sh.control == crate::control::Control::FamilyColumn {
            skip = self.col.saturating_sub(1);
        }
        j + usize::from(j >= skip)
    }

    /// The bordered matrix [[J, F_p], [t^T]] of the corrector `b` at w = (z, p) (P::PARTS part arrays of `len() +
    /// 1`, p last), in f64 from the lanes T (`Fu`, `Fun<N>`, `Dtn<N>`), factored dense ([`BorderedLu`]): the layout's
    /// f64 Jacobian (`Shooting::jacobian_f64`, with `row_scale`) at w's layout state, the dropped row replaced by `b.t`
    /// rounded to f64. For multisolver's `arclength::continuation`, `start`, `step` and `locate_fold`, as `|b, w|
    /// fam.bordered_lu::<P, T>(b, w, par)`. With f64 factors the corrector converges linearly, each step contracting
    /// by about the factors' relative error times the conditioning, and the tangent is refined to P's accuracy against
    /// the bordered matrix in P.
    pub fn bordered_lu<P: Precision<Native>, T: Lane>(
        &self,
        b: &Bordered<'_, P, impl Family<P>>,
        w: &[&[f64]],
        par: bool,
    ) -> Result<BorderedLu, String> {
        let n = self.n();
        if b.t.len() != n + 1 || w.len() != P::PARTS || w.iter().any(|q| q.len() != n + 1) {
            return Err(format!(
                "bordered_lu: a tangent of {} and w of {} part arrays, not {} and {} of that length",
                b.t.len(),
                w.len(),
                n + 1,
                P::PARTS
            ));
        }
        let p: Vec<f64> = w.iter().map(|q| q[n]).collect();
        let z: Vec<&[f64]> = w.iter().map(|q| &q[..n]).collect();
        let zl = self.layout_z::<P>(&z, P::from_raw(&p))?;
        let (mut c, ld) = self.sh.jacobian_f64::<T>(&parts(&zl[..2]), par)?;
        #[allow(unused_mut)]
        let mut border = true;
        #[cfg(feature = "mutate")]
        if self.sh.control == crate::control::Control::BorderPin {
            border = false;
        }
        if border {
            for j in 0..=n {
                c[j * ld + self.row] = P::to_dd(b.t[self.w_of(j)]).hi;
            }
        }
        Ok(BorderedLu {
            lu: DDLu::factor_f64(n + 1, ld, c, par)?,
            col: self.col,
            row: self.row,
        })
    }

    /// the place in w = (z, p) of the layout's unknown j
    fn w_of(&self, j: usize) -> usize {
        match j.cmp(&self.col) {
            std::cmp::Ordering::Less => j,
            std::cmp::Ordering::Equal => self.n(),
            std::cmp::Ordering::Greater => j - 1,
        }
    }
}

/// The family F(z, p): the layout's residual, Jacobian and J v at the layout state of (z, p), re-indexed
impl<E: Equations<B>, const B: usize, P: Precision<Native>> Family<P> for InGlobal<'_, E, B> {
    fn len(&self) -> usize {
        self.n()
    }

    /// the layout's residual at (z, p) without the dropped row
    fn residual_at(&self, z: &[&[f64]], p: P::Scalar, par: bool) -> Result<Vec<P::Scalar>, String> {
        let zl = self.layout_z::<P>(z, p)?;
        let mut f = self.sh.residual::<P>(&parts(&zl), par)?;
        f.remove(self.row);
        Ok(f)
    }

    /// the layout's Jacobian at (z, p) without the dropped row and g's column, entry for entry
    fn jacobian_at(&self, z: &[&[f64]], p: P::Scalar, par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let zl = self.layout_z::<P>(z, p)?;
        let (c, ld0) = self.sh.jacobian::<P>(&parts(&zl), par)?;
        let (n, r) = (self.n(), self.row);
        let ld = ld_for(n);
        let out = c
            .iter()
            .map(|cq| {
                let mut o = vec![0.0; ld * n];
                for j in 0..n {
                    let s = self.layout_col(j) * ld0;
                    o[j * ld..j * ld + r].copy_from_slice(&cq[s..s + r]);
                    o[j * ld + r..j * ld + n].copy_from_slice(&cq[s + r + 1..s + n + 1]);
                }
                o
            })
            .collect();
        Ok((out, ld))
    }

    /// J v at (z, p) with v the unit vector on g, without the dropped row: one dual pass per block
    fn dp_at(&self, z: &[&[f64]], p: P::Scalar, par: bool) -> Result<Vec<P::Scalar>, String> {
        let zl = self.layout_z::<P>(z, p)?;
        let mut v = vec![vec![0.0; self.n() + 1]; P::PARTS];
        v[0][self.col] = 1.0;
        let (mut jv, _) = self.sh.jvp::<P>(&parts(&zl), &parts(&v), par)?;
        jv.remove(self.row);
        Ok(jv)
    }
}

/// The f64 factors of a corrector's bordered matrix ([`InGlobal::bordered_lu`]): the dense LU of the layout's
/// Jacobian with the dropped row replaced by the tangent. Solves in double-double.
pub struct BorderedLu {
    lu: DDLu,
    col: usize,
    row: usize,
}

impl<P: Precision<Native>> Factored<P> for BorderedLu {
    /// -M^-1 r in P, r in the bordered rows' order (F's rows, then the tangent's): r rounded to double-double and put
    /// in the layout's rows (the tangent's in the dropped row's place), solved with the f64 factors in double-double,
    /// and the solution read in w's order (z, then p from g's place), taken into P exactly
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        let n = r.len() - 1;
        let mut b: Vec<DD> = (0..=n)
            .map(|i| {
                let ib = match i.cmp(&self.row) {
                    std::cmp::Ordering::Less => i,
                    std::cmp::Ordering::Equal => n,
                    std::cmp::Ordering::Greater => i - 1,
                };
                -P::to_dd(r[ib])
            })
            .collect();
        self.lu.solve(&mut b);
        (0..=n)
            .map(|jw| {
                let j = if jw == n {
                    self.col
                } else {
                    jw + usize::from(jw >= self.col)
                };
                P::from_dd(b[j])
            })
            .collect()
    }
}
