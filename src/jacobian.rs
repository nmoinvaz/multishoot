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
use crate::lane::{BaseLane, Lane};
use crate::shooting::Shooting;
use multidouble::dual::Tangents;
use multidouble::field::Fl;
use multidouble::linalg::ld_for;
use multidouble::native::{Ou, Oun};
use multidouble::precision::{Octo, Precision};
use multidouble::simd::Native;
use multisolver::Problem;
use rayon::prelude::*;

/// A row's factor of the Jacobian's scaling in P and its product by +0
pub(crate) type RowFactor<P> = (<P as Precision<Native>>::Scalar, <P as Precision<Native>>::Scalar);

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
    /// of its inputs in z, the globals g and the evaluation's preparation.
    pub fn jac_column<P: Precision<Native>>(
        &self,
        z: &[&[f64]],
        g: &[P::Scalar],
        prep: &E::Prep,
        (b, seed): (usize, Seed),
    ) -> Vec<P::Scalar> {
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
        let s: State<P::U, B> = self.block_state::<P::U>(b, &inp, &gu, prep);
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
    pub fn jacobian<P: Precision<Native>>(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        self.check_z::<P>(z)?;
        if !self.is_square() {
            return Err(format!("jacobian: {} rows for {} unknowns", self.rows(), self.len()));
        }
        let g = self.globals::<P>(z);
        let tasks = self.jac_tasks();
        let prep = self.prep(z)?;
        let column = |t: &(usize, Seed)| self.jac_column::<P>(z, &g, &prep, *t);
        let cols: Vec<Vec<P::Scalar>> = if par {
            tasks.par_iter().map(column).collect()
        } else {
            tasks.iter().map(column).collect()
        };
        self.assemble::<P>(&tasks, &cols)
    }

    /// `jacobian`, bit for bit, with dual lanes T that carry `T::NTD` tangents each (multidouble's `Tangents`, such as
    /// `Oun<N>` for octo-double): one evaluation of a block for up to NTD of its column tasks, the value formed once
    /// for all of them and each tangent by the operations of its own single-tangent evaluation.
    pub fn jacobian_tangents<P: Precision<Native>, T: Lane + Tangents<V = P::V>>(
        &self,
        z: &[&[f64]],
        par: bool,
    ) -> Result<(Vec<Vec<f64>>, usize), String> {
        self.check_z::<P>(z)?;
        if !self.is_square() {
            return Err(format!("jacobian: {} rows for {} unknowns", self.rows(), self.len()));
        }
        let n = T::NTD.max(1);
        let g = self.globals::<P>(z);
        let tasks = self.jac_tasks();
        let prep = self.prep(z)?;
        // consecutive tasks of one block, at most n to an evaluation
        let mut groups: Vec<(usize, Vec<Seed>)> = Vec::new();
        for &(b, seed) in &tasks {
            match groups.last_mut() {
                Some((gb, seeds)) if *gb == b && seeds.len() < n => seeds.push(seed),
                _ => groups.push((b, vec![seed])),
            }
        }
        let eval = |(b, seeds): &(usize, Vec<Seed>)| self.jac_columns_tangents::<P, T>(z, &g, &prep, *b, seeds);
        let cols: Vec<Vec<P::Scalar>> = if par {
            groups.par_iter().flat_map_iter(eval).collect()
        } else {
            groups.iter().flat_map(eval).collect()
        };
        self.assemble::<P>(&tasks, &cols)
    }

    /// The columns of block b's tasks `seeds` from one evaluation with T lanes, tangent k seeded as task k's
    /// `jac_column` seeds its one tangent; each column NF M values
    fn jac_columns_tangents<P: Precision<Native>, T: Lane + Tangents<V = P::V>>(
        &self,
        z: &[&[f64]],
        g: &[P::Scalar],
        prep: &E::Prep,
        b: usize,
        seeds: &[Seed],
    ) -> Vec<Vec<P::Scalar>> {
        let m = Self::M;
        let inp: Vec<Fl<T, B>> = self
            .block_inputs(b)
            .iter()
            .enumerate()
            .map(|(fi, &o)| {
                let t: Vec<Fl<P::V, B>> = seeds
                    .iter()
                    .map(|seed| match *seed {
                        Seed::Field(f, j) if f == fi => Fl::<P::V, B>::unit(j),
                        Seed::Points(f) if f == fi => Fl::<P::V, B>::splat(multidouble::dd::DD::ONE),
                        _ => Fl::zero(),
                    })
                    .collect();
                Fl::<T, B>::from_tangents(&self.load::<P>(z, o), &t)
            })
            .collect();
        let gu: Vec<T> = g
            .iter()
            .enumerate()
            .map(|(i, &gi)| {
                let t: Vec<P::V> = seeds
                    .iter()
                    .map(|seed| {
                        if *seed == Seed::Global(i) {
                            P::splat(P::ONE)
                        } else {
                            <P::V as BaseLane>::zero()
                        }
                    })
                    .collect();
                T::from_tangents(P::splat(gi), &t)
            })
            .collect();
        let s: State<T, B> = self.block_state::<T>(b, &inp, &gu, prep);
        #[cfg(feature = "mutate")]
        let swap = self.control == crate::control::Control::TangentOrder && seeds.len() > 1;
        #[cfg(not(feature = "mutate"))]
        let swap = false;
        #[cfg(feature = "mutate")]
        let shift = self.control == crate::control::Control::TangentShift;
        #[cfg(not(feature = "mutate"))]
        let shift = false;
        (0..seeds.len())
            .map(|k| {
                let k = if swap { seeds.len() - 1 - k } else { k };
                let mut out = Vec::with_capacity(E::NF * m);
                for sq in &s {
                    let t = match (shift, k) {
                        (true, 0) => sq.value(),
                        (true, k) => sq.tangent(k - 1),
                        _ => sq.tangent(k),
                    };
                    for i in 0..m {
                        out.push(P::field_get(&t, i));
                    }
                }
                out
            })
            .collect()
    }

    /// The Jacobian from its column tasks' columns: the entries, the targets' -1, the projected rows, the extra rows
    /// and the row scaling, in `jacobian`'s order
    fn assemble<P: Precision<Native>>(
        &self,
        tasks: &[(usize, Seed)],
        cols: &[Vec<P::Scalar>],
    ) -> Result<(Vec<Vec<f64>>, usize), String> {
        let l = self.len();
        let m = Self::M;
        let ld = ld_for(l);
        let mut c: Vec<Vec<f64>> = (0..P::PARTS).map(|_| vec![0.0; ld * l]).collect();
        let add = |c: &mut [Vec<f64>], idx: usize, v: P::Scalar| {
            let cur: Vec<f64> = (0..P::PARTS).map(|q| c[q][idx]).collect();
            let s = P::from_raw(&cur) + v;
            for (q, cq) in c.iter_mut().enumerate() {
                cq[idx] = P::part(s, q);
            }
        };
        let nf = E::NF;
        for (&(b, seed), col) in tasks.iter().zip(cols) {
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
        self.project_jacobian::<P>(&mut c, ld);
        let re = self.row_extra();
        for (r, row) in self.extra.iter().enumerate() {
            for &(i, w) in &row.terms {
                add(&mut c, i * ld + re + r, P::from_dd(w));
            }
        }
        let scale = self.jac_row_factors::<P>()?;
        if !scale.is_empty() {
            let mut cur = vec![0.0; P::PARTS];
            for j in 0..l {
                for (r, &sr) in scale.iter().enumerate() {
                    let idx = j * ld + r;
                    for q in 0..P::PARTS {
                        cur[q] = c[q][idx];
                    }
                    let v = Self::jac_scaled::<P>(&cur, sr);
                    for (q, cq) in c.iter_mut().enumerate() {
                        cq[idx] = P::part(v, q);
                    }
                }
            }
        }
        Ok((c, ld))
    }

    /// The Jacobian's row factors (`row_scale`) in P, each with its product by +0; empty without a scaling. An entry
    /// of +0 parts scales to the same bits in every column, so that product is formed once per row.
    pub(crate) fn jac_row_factors<P: Precision<Native>>(&self) -> Result<Vec<RowFactor<P>>, String> {
        let l = self.len();
        if self.row_scale.is_empty() {
            return Ok(Vec::new());
        }
        if self.row_scale.len() != l {
            return Err(format!("row_scale has {} factors for {l} rows", self.row_scale.len()));
        }
        Ok(self
            .row_scale
            .iter()
            .map(|&sr| {
                let s = P::from_dd(multidouble::dd::DD::from_f64(sr));
                (s, P::ZERO * s)
            })
            .collect())
    }

    /// A Jacobian entry of parts `cur` times its row's factor (s, +0 s) from `jac_row_factors`
    pub(crate) fn jac_scaled<P: Precision<Native>>(cur: &[f64], (s, zs): RowFactor<P>) -> P::Scalar {
        if cur.iter().all(|x| x.to_bits() == 0) {
            zs
        } else {
            P::from_raw(cur) * s
        }
    }

    /// The layout as an octo-double problem ([`multisolver::Problem`] at [`Octo`]) whose exact Jacobian takes n
    /// tangents per evaluation: `jacobian_tangents` with multidouble's `Ou` lanes for n = 1 and `Oun<n>` for n = 2,
    /// 4 or 8; any other n is an error. Its residual is `residual::<Octo>`'s and its Jacobian `jacobian::<Octo>`'s,
    /// bit for bit, so multisolver's solvers take the same steps on it as on the layout; only the Jacobian's cost
    /// changes, one evaluation of a block serving up to n of its columns. Octo-double only: multidouble's lanes with
    /// several tangents in the value's precision (`Tangents`) are `Ou` and `Oun<N>`, octo-double's.
    pub fn with_tangents(&self, n: usize) -> Result<WithTangents<'_, E, B>, String> {
        match n {
            1 | 2 | 4 | 8 => Ok(WithTangents { sh: self, n }),
            _ => Err(format!("with_tangents: 1, 2, 4 or 8 tangents per evaluation, not {n}")),
        }
    }
}

/// A layout as an octo-double problem whose Jacobian takes several tangents per evaluation
/// ([`Shooting::with_tangents`])
pub struct WithTangents<'a, E: Equations<B>, const B: usize> {
    sh: &'a Shooting<E, B>,
    n: usize,
}

impl<E: Equations<B>, const B: usize> WithTangents<'_, E, B> {
    /// the tangents per evaluation: 1, 2, 4 or 8
    pub fn tangents(&self) -> usize {
        self.n
    }
}

impl<E: Equations<B>, const B: usize> Problem<Octo> for WithTangents<'_, E, B> {
    fn len(&self) -> usize {
        self.sh.len()
    }
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<<Octo as Precision<Native>>::Scalar>, String> {
        self.sh.residual::<Octo>(z, par)
    }
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        match self.n {
            1 => self.sh.jacobian_tangents::<Octo, Ou>(z, par),
            2 => self.sh.jacobian_tangents::<Octo, Oun<2>>(z, par),
            4 => self.sh.jacobian_tangents::<Octo, Oun<4>>(z, par),
            8 => self.sh.jacobian_tangents::<Octo, Oun<8>>(z, par),
            n => Err(format!("with_tangents: 1, 2, 4 or 8 tangents per evaluation, not {n}")),
        }
    }
}
