//! The exact Jacobian dF/dz by forward-mode duals (Wengert, Commun. ACM 7, 463 (1964)), in any precision: one
//! evaluation of a block per column, with a unit tangent on one of its inputs or on a global, through the same code
//! that forms the residual.
//!
//! Block b's rows depend on its inputs, its target node (through the -1 of the mismatch) and the globals; the extra
//! rows on the unknowns they name, with their constant weights. Every other entry is zero. The matrix is stored
//! column-major with leading dimension `ld_for(L)`, part q of entry (i, j) at `c[q][j ld + i]`.
//!
//! For a pointwise problem (`Equations::POINTWISE`) the lanes are independent, so one evaluation seeds a unit
//! tangent on every point of an input field and gives that field's M columns at once: point j's tangent reaches only
//! point j's rows. A column's entries are those of the evaluation seeded on its point alone, lane for lane.
use crate::equations::{Equations, State};
use crate::lane::BaseLane;
use crate::prec::Prec;
use crate::shooting::Shooting;
use multidouble::field::Fl;
use multidouble::linalg::ld_for;
use rayon::prelude::*;

/// What a Jacobian column task differentiates along: point j of a block's input field f, every point of input
/// field f at once (pointwise problems), or global i
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seed {
    Field(usize, usize),
    Points(usize),
    Global(usize),
}

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// The Jacobian's column tasks: (block, seed), every point of every input field of every block (one task per
    /// field for a pointwise problem), then its globals.
    pub fn jac_tasks(&self) -> Vec<(usize, Seed)> {
        let mut tasks = Vec::new();
        for b in 0..self.k() + 1 {
            for f in 0..self.block_inputs(b).len() {
                if E::POINTWISE {
                    tasks.push((b, Seed::Points(f)));
                    continue;
                }
                for j in 0..Self::M {
                    tasks.push((b, Seed::Field(f, j)));
                }
            }
            for i in 0..E::NG {
                tasks.push((b, Seed::Global(i)));
            }
        }
        tasks
    }

    /// The z indices of a task's columns
    pub fn seed_columns(&self, b: usize, s: Seed) -> Vec<usize> {
        match s {
            Seed::Field(f, j) => vec![self.block_inputs(b)[f] + j],
            Seed::Points(f) => (0..Self::M).map(|j| self.block_inputs(b)[f] + j).collect(),
            Seed::Global(i) => vec![self.off_global() + i],
        }
    }

    /// One column task: block b's state (NF M values) differentiated along the seed, with dual lanes, at the values
    /// v of its inputs and the globals g.
    pub fn jac_column<P: Prec>(&self, z: &[&[f64]], g: &[P::Scalar], (b, seed): (usize, Seed)) -> Vec<P::Scalar> {
        let m = Self::M;
        let inp: Vec<Fl<P::U, B>> = self
            .block_inputs(b)
            .iter()
            .enumerate()
            .map(|(fi, &o)| {
                let t = match seed {
                    Seed::Field(f, j) if f == fi => Fl::<P::V, B>::unit(j),
                    Seed::Points(f) if f == fi => Fl::<P::V, B>::splat(multidouble::dd::DD::ONE),
                    _ => Fl::zero(),
                };
                P::field_dual(&self.load::<P>(z, o), &t)
            })
            .collect();
        let gu: Vec<P::U> = g
            .iter()
            .enumerate()
            .map(|(i, &gi)| {
                let d = if seed == Seed::Global(i) {
                    P::splat(P::ONE)
                } else {
                    <P::V as BaseLane>::zero()
                };
                P::dual(P::splat(gi), d)
            })
            .collect();
        let s: State<P::U, B> = self.block_state::<P::U>(b, &inp, &gu);
        let mut out = Vec::with_capacity(E::NF * m);
        for sq in &s {
            let t = P::field_coef(sq, 1);
            for i in 0..m {
                out.push(P::field_get(&t, i));
            }
        }
        out
    }

    /// The exact Jacobian at z: (PARTS column-major part arrays, ld). The system must be square.
    pub fn jacobian<P: Prec>(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        self.check_z::<P>(z)?;
        if !self.is_square() {
            return Err(format!("jacobian: {} rows for {} unknowns", self.rows(), self.len()));
        }
        let l = self.len();
        let m = Self::M;
        let ld = ld_for(l);
        let g = self.globals::<P>(z);
        let tasks = self.jac_tasks();
        let column = |t: &(usize, Seed)| self.jac_column::<P>(z, &g, *t);
        let cols: Vec<Vec<P::Scalar>> = if par {
            tasks.par_iter().map(column).collect()
        } else {
            tasks.iter().map(column).collect()
        };
        let mut c: Vec<Vec<f64>> = (0..P::PARTS).map(|_| vec![0.0; ld * l]).collect();
        let add = |c: &mut [Vec<f64>], idx: usize, v: P::Scalar| {
            let cur: Vec<f64> = (0..P::PARTS).map(|q| c[q][idx]).collect();
            let s = P::from_raw(&cur) + v;
            for (q, cq) in c.iter_mut().enumerate() {
                cq[idx] = P::part(s, q);
            }
        };
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
                    for (q, cq) in c.iter_mut().enumerate() {
                        cq[cj * ld + r0 + r] = P::part(col[r], q);
                    }
                }
            }
        }
        for b in 0..self.k() + 1 {
            let t = self.block_target(b);
            let r0 = self.row_block(b);
            for r in 0..E::NF * m {
                add(&mut c, (t + r) * ld + r0 + r, -P::ONE);
            }
        }
        let re = self.row_extra();
        for (r, row) in self.extra.iter().enumerate() {
            for &(i, w) in &row.terms {
                add(&mut c, i * ld + re + r, P::from_dd(w));
            }
        }
        Ok((c, ld))
    }
}
