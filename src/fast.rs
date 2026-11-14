//! The fast path of Newton's method: the Jacobian in f64 from lanes with f64 tangents, several columns per
//! evaluation, stored dense or factored in staircase storage, and the mixed-precision Newton step built on it.
//!
//! A Newton step needs its Jacobian only to the accuracy that keeps the iteration contracting: with an f64 Jacobian
//! J~ and the residual in double-double (or quad-, octo-double), z <- z - J~^-1 F converges to the same root, linearly,
//! with a contraction of about |J~ - J| |J^-1| per step (iterative refinement, Moler, J. ACM 14, 316 (1967)). The
//! columns come from multidouble's lanes with f64 tangents (`Lane::TAN_F64`): `Fu` (f64 value and tangent), `Fun<N>`
//! (f64 value, N tangents) and `Dtn<N>` (double-double value, N f64 tangents, the base trajectory to double-double
//! accuracy once for N columns). Each evaluation carries `T::NT` column tasks of one block.
//!
//! The staircase storage keeps only the nonzero blocks: residual block b depends on two neighbouring groups of
//! unknowns (the left data and node 0, nodes b - 1 and b, node K - 1 and the right data) and on the global, so the
//! system is almost block diagonal. multidouble's `StaircaseLu` factors it step by step with partial pivoting on a
//! system augmented by one copy of the global per group, without forming the dense matrix, and solves in
//! double-double with the f64 factors. It needs one global and one extra row on the left data, the global and node 0.
use crate::equations::Equations;
use crate::jacobian::Seed;
use crate::lane::{Lane, DV};
use crate::newton::{newton_with, Factored, NewtonOpts, NewtonReport};
use crate::shooting::Shooting;
use multidouble::dd::DD;
use multidouble::field::Fl;
use multidouble::linalg::{ld_for, ColumnGroup, DDLu, StaircaseControl, StaircaseLu};
use multidouble::native::V8;
use multidouble::precision::{Double, Precision};
use multidouble::simd::Native;
use rayon::prelude::*;

/// Where the f64 Jacobian of a mixed step is kept and factored
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// the dense matrix, factored by the blocked f64 LU (`DDLu::factor_f64`)
    Dense,
    /// the nonzero blocks only, factored by the staircase LU (`StaircaseLu`)
    Staircase,
}

/// The f64 factors of a mixed step, dense or in staircase storage; both solve in double-double
pub enum MixedLu {
    Dense(DDLu),
    Staircase(StaircaseLu),
}

impl MixedLu {
    /// Solve J x = b in place, in double-double with the f64 factors
    pub fn solve(&self, b: &mut [DD]) {
        match self {
            MixedLu::Dense(lu) => lu.solve(b),
            MixedLu::Staircase(lu) => lu.solve(b),
        }
    }

    /// min and max |u_kk| of the factors
    pub fn pivot_range(&self) -> (f64, f64) {
        match self {
            MixedLu::Dense(lu) => lu.pivot_range(),
            MixedLu::Staircase(lu) => lu.pivot_range(),
        }
    }
}

impl<P: Precision<Native>> Factored<P> for MixedLu {
    /// -J^-1 r in P: r rounded to double-double, solved with the f64 factors in double-double, the solution taken
    /// into P exactly
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        let mut b: Vec<DD> = r.iter().map(|&v| -P::to_dd(v)).collect();
        self.solve(&mut b);
        b.into_iter().map(P::from_dd).collect()
    }
}

/// The f64 values of a field's f64 tangent k
fn tangent<T: Lane, const B: usize>(f: &Fl<T, B>, k: usize, out: &mut Vec<f64>) {
    for b in 0..B {
        out.extend_from_slice(&f.0[b].tan64(k).to_array());
    }
}

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// The column groups of an f64 Jacobian with T::NT tangents per evaluation: each block's tasks (`jac_tasks`) in
    /// chunks of NT, in task order.
    pub fn f64_groups<T: Lane>(&self) -> Vec<Vec<(usize, Seed)>> {
        assert!(T::TAN_F64 && T::NT >= 1, "f64_groups: a lane type with f64 tangents");
        let mut groups: Vec<Vec<(usize, Seed)>> = Vec::new();
        for t in self.jac_tasks() {
            match groups.last_mut() {
                Some(g) if g.len() < T::NT && g[0].0 == t.0 => g.push(t),
                _ => groups.push(vec![t]),
            }
        }
        groups
    }

    /// One evaluation of a block with f64 tangents (lanes T, `Lane::TAN_F64`): tangent k carries task `group[k]`, all
    /// of one block, at the double-double values of z (2 part arrays), the globals g and the preparation. Returns, in
    /// the order of the group, each task's NF M derivatives of the block's state.
    pub fn jac_columns_f64<T: Lane>(
        &self,
        z: &[&[f64]],
        g: &[DD],
        prep: &E::Prep,
        group: &[(usize, Seed)],
    ) -> Vec<Vec<f64>> {
        assert!(
            T::TAN_F64 && !group.is_empty() && group.len() <= T::NT && group.iter().all(|t| t.0 == group[0].0),
            "jac_columns_f64: 1..=NT tasks of one block"
        );
        let b = group[0].0;
        let inp: Vec<Fl<T, B>> = self
            .block_inputs(b)
            .iter()
            .enumerate()
            .map(|(fi, &o)| {
                let mut x = Fl::<T, B>::from_vt(&self.load::<Double>(z, o), &Fl::zero());
                for (k, &(_, seed)) in group.iter().enumerate() {
                    match seed {
                        Seed::Field(f, j) if f == fi => {
                            let mut e = [0.0; 8];
                            e[j % 8] = 1.0;
                            x.0[j / 8].set_tan64(k, V8::from_array(e));
                        }
                        Seed::Points(f) if f == fi => {
                            for xb in x.0.iter_mut() {
                                xb.set_tan64(k, V8::splat(1.0));
                            }
                        }
                        _ => {}
                    }
                }
                x
            })
            .collect();
        let gl: Vec<T> = g
            .iter()
            .enumerate()
            .map(|(i, &gi)| {
                let mut t = T::from_vt(DV::splat(gi), DV::zero());
                for (k, &(_, seed)) in group.iter().enumerate() {
                    if seed == Seed::Global(i) {
                        t.set_tan64(k, V8::splat(1.0));
                    }
                }
                t
            })
            .collect();
        let s = self.block_state::<T>(b, &inp, &gl, prep);
        (0..group.len())
            .map(|k| {
                let mut out = Vec::with_capacity(E::NF * Self::M);
                for sq in &s {
                    tangent(sq, k, &mut out);
                }
                out
            })
            .collect()
    }

    /// the f64 columns of the groups, in parallel with `par`, flattened in task order, or the preparation's error
    fn f64_columns<T: Lane>(
        &self,
        z: &[&[f64]],
        groups: &[Vec<(usize, Seed)>],
        par: bool,
    ) -> Result<Vec<Vec<f64>>, String> {
        let g = self.globals::<Double>(z);
        let prep = self.prep(z)?;
        let one = |gr: &Vec<(usize, Seed)>| self.jac_columns_f64::<T>(z, &g, &prep, gr);
        let cols: Vec<Vec<Vec<f64>>> = if par {
            groups.par_iter().map(one).collect()
        } else {
            groups.iter().map(one).collect()
        };
        Ok(cols.into_iter().flatten().collect())
    }

    /// The rows (within its block) of a task's column jj: point jj of every output field for a pointwise task, every
    /// row otherwise
    fn task_rows(seed: Seed, jj: usize) -> Vec<usize> {
        let (nf, m) = (E::NF, Self::M);
        if let Seed::Points(_) = seed {
            (0..nf).map(|q| q * m + jj).collect()
        } else {
            (0..nf * m).collect()
        }
    }

    /// whether the Jacobian's column c is dropped by `Control::DropColumn`
    fn dropped(&self, _c: usize) -> bool {
        #[cfg(feature = "mutate")]
        if self.control == crate::control::Control::DropColumn(_c) {
            return true;
        }
        false
    }

    /// The Jacobian in f64 at z (2 part arrays, double-double) from the lanes T (`Fu`, `Fun<N>`, `Dtn<N>`): the same
    /// columns as `jacobian`, T::NT per evaluation, rounded to f64, with the target's -1, the projected rows (formed in
    /// double-double and rounded), the extra rows' weights (hi parts) and `row_scale`. Column-major with leading dimension ld: (J, ld). The system must be square.
    pub fn jacobian_f64<T: Lane>(&self, z: &[&[f64]], par: bool) -> Result<(Vec<f64>, usize), String> {
        self.check_z::<Double>(z)?;
        if !self.is_square() {
            return Err(format!(
                "jacobian_f64: {} rows for {} unknowns",
                self.rows(),
                self.len()
            ));
        }
        let (l, m) = (self.len(), Self::M);
        let ld = ld_for(l);
        let groups = self.f64_groups::<T>();
        let cols = self.f64_columns::<T>(z, &groups, par)?;
        let mut c = vec![0.0; ld * l];
        for (&(b, seed), col) in groups.iter().flatten().zip(&cols) {
            let r0 = self.row_block(b);
            for (jj, cj) in self.seed_columns(b, seed).into_iter().enumerate() {
                if self.dropped(cj) {
                    continue;
                }
                for r in Self::task_rows(seed, jj) {
                    c[cj * ld + r0 + r] = col[r];
                }
            }
        }
        for b in 0..self.k() + 1 {
            let (t, r0) = (self.block_target(b), self.row_block(b));
            for r in 0..E::NF * m {
                c[(t + r) * ld + r0 + r] -= 1.0;
            }
            self.project_f64_block(b, &mut c, ld, r0, l, Some, |_| None)?;
        }
        let re = self.row_extra();
        for (r, row) in self.extra.iter().enumerate() {
            for &(i, w) in &row.terms {
                c[i * ld + re + r] += w.hi;
            }
        }
        if !self.row_scale.is_empty() {
            if self.row_scale.len() != l {
                return Err(format!("row_scale has {} factors for {l} rows", self.row_scale.len()));
            }
            for j in 0..l {
                for (r, &s) in self.row_scale.iter().enumerate() {
                    c[j * ld + r] *= s;
                }
            }
        }
        Ok((c, ld))
    }

    /// the staircase LU's deliberate breakage under the controls `StaircasePivot` and `StaircaseLink`
    fn staircase_control(&self) -> StaircaseControl {
        #[cfg(feature = "mutate")]
        match self.control {
            crate::control::Control::StaircasePivot => return StaircaseControl::Pivot,
            crate::control::Control::StaircaseLink => return StaircaseControl::Link,
            _ => {}
        }
        StaircaseControl::None
    }

    /// The column groups of the staircase storage: the left data, then each node with a copy of the global first,
    /// then the right data with one
    fn staircase_groups(&self) -> Vec<ColumnGroup> {
        let m = Self::M;
        let mut gs = vec![ColumnGroup {
            param: false,
            zoff: 0,
            zlen: E::NL * m,
        }];
        for t in 0..self.k() {
            gs.push(ColumnGroup {
                param: true,
                zoff: self.off_node(t),
                zlen: E::NF * m,
            });
        }
        gs.push(ColumnGroup {
            param: true,
            zoff: self.off_right(),
            zlen: E::NR * m,
        });
        gs
    }

    /// The f64 Jacobian of `jacobian_f64` built block by block and factored in staircase storage (multidouble's
    /// `StaircaseLu`), never forming the dense matrix. Every block's columns are computed first (in parallel with
    /// `par`) by the same evaluations as the dense Jacobian's, so the blocks hold its entries. It needs a square system
    /// with one global (`NG` = 1) and one extra row, whose terms lie on the left data, the global and node 0.
    pub fn jacobian_staircase<T: Lane>(&self, z: &[&[f64]], par: bool) -> Result<StaircaseLu, String> {
        self.check_z::<Double>(z)?;
        if !self.is_square() || E::NG != 1 || self.extra.len() != 1 {
            return Err(format!(
                "jacobian_staircase: a square system with one global and one extra row, not {} globals and {} rows",
                E::NG,
                self.extra.len()
            ));
        }
        let (l, m, og) = (self.len(), Self::M, self.off_global());
        let row0 = &self.extra[0];
        if let Some(&(i, _)) = row0.terms.iter().find(|&&(i, _)| i >= self.off_node(1) && i != og) {
            return Err(format!(
                "jacobian_staircase: the extra row's unknown {i} is not on the left data, the global or node 0"
            ));
        }
        if !self.row_scale.is_empty() && self.row_scale.len() != l {
            return Err(format!("row_scale has {} factors for {l} rows", self.row_scale.len()));
        }
        let scale = |r: usize| {
            if self.row_scale.is_empty() {
                1.0
            } else {
                self.row_scale[r]
            }
        };
        let gs = self.staircase_groups();
        let rblocks: Vec<(usize, usize)> = (0..=self.k()).map(|b| (self.row_block(b), E::NF * m)).collect();
        // every block's columns before the factorization, which asks for the blocks in turn, so the evaluations of all
        // blocks run in parallel as the dense Jacobian's do
        let groups = self.f64_groups::<T>();
        let cols = self.f64_columns::<T>(z, &groups, par)?;
        let tasks: Vec<(usize, Seed)> = groups.into_iter().flatten().collect();
        // each block's task indices, in task order (jac_tasks emits the blocks in turn)
        let mut by_block: Vec<Vec<usize>> = vec![Vec::new(); self.k() + 1];
        for (i, t) in tasks.iter().enumerate() {
            by_block[t.0].push(i);
        }
        let gc = gs.clone();
        let fill = |b: usize, w: &mut [f64], ld: usize, r0: usize| -> Result<(), String> {
            let (p, q) = (gc[b], gc[b + 1]);
            let np = p.len();
            // the column of z index zi in this step: group p, then group q; the global is q's copy
            let pos = |zi: usize| -> Result<usize, String> {
                if zi == og {
                    Ok(np)
                } else if zi >= p.zoff && zi < p.zoff + p.zlen {
                    Ok(p.param as usize + zi - p.zoff)
                } else if zi >= q.zoff && zi < q.zoff + q.zlen {
                    Ok(np + q.param as usize + zi - q.zoff)
                } else {
                    Err(format!(
                        "jacobian_staircase: block {b}'s unknown {zi} outside its two groups"
                    ))
                }
            };
            let rb = self.row_block(b);
            for &i in &by_block[b] {
                let (seed, col) = (tasks[i].1, &cols[i]);
                for (jj, cj) in self.seed_columns(b, seed).into_iter().enumerate() {
                    if self.dropped(cj) {
                        continue;
                    }
                    let c = pos(cj)?;
                    for r in Self::task_rows(seed, jj) {
                        w[c * ld + r0 + r] = col[r];
                    }
                }
            }
            let t = self.block_target(b);
            for r in 0..E::NF * m {
                w[pos(t + r)? * ld + r0 + r] -= 1.0;
            }
            // projected rows: their pinned unknowns must lie in the block's two groups
            self.project_f64_block(b, w, ld, r0, np + q.len(), |zi| pos(zi).ok(), |_| None)?;
            for r in 0..E::NF * m {
                let s = scale(rb + r);
                if s != 1.0 {
                    for c in 0..np + q.len() {
                        w[c * ld + r0 + r] *= s;
                    }
                }
            }
            if b == 0 {
                let s = scale(self.row_extra());
                for &(i, wt) in &row0.terms {
                    w[pos(i)? * ld + r0 - 1] += wt.hi * s;
                }
            }
            Ok(())
        };
        StaircaseLu::factor_control(
            l,
            gs,
            &rblocks,
            self.row_extra(),
            og,
            fill,
            par,
            self.staircase_control(),
        )
    }

    /// The mixed-precision Newton step at z (PARTS part arrays of P): F in P, the f64 Jacobian of lanes T at z's
    /// double-double parts factored in the given storage, and dz = -J~^-1 F solved in double-double with its factors.
    /// Returns (dz as P's part arrays, F, the factors).
    #[allow(clippy::type_complexity)]
    pub fn newton_step_mixed<P: Precision<Native>, T: Lane>(
        &self,
        z: &[&[f64]],
        storage: Storage,
        par: bool,
    ) -> Result<(Vec<Vec<f64>>, Vec<P::Scalar>, MixedLu), String> {
        let f = self.residual::<P>(z, par)?;
        let lu = self.mixed_lu::<T>(z, storage, par)?;
        let dz = Factored::<P>::solve_neg(&lu, &f);
        let parts = (0..P::PARTS)
            .map(|q| dz.iter().map(|&v| P::part(v, q)).collect())
            .collect();
        Ok((parts, f, lu))
    }

    /// Newton's method with the mixed step, damped as `newton` (`NewtonOpts::damping`): the residual in P, the Newton
    /// matrix the f64 Jacobian of lanes T in the given storage, formed afresh at each step, and the simplified
    /// corrections of the correction rule solved with the same factors. It converges to the root of `newton`,
    /// linearly, each step contracting by about the f64 Jacobian's relative error times the conditioning.
    pub fn newton_mixed<P: Precision<Native>, T: Lane>(
        &self,
        z: &mut Vec<Vec<f64>>,
        opts: &NewtonOpts,
        storage: Storage,
    ) -> Result<NewtonReport, String> {
        newton_with::<P, Self, MixedLu>(self, z, opts, |z| self.mixed_lu::<T>(z, storage, opts.par))
    }

    /// The f64 Jacobian of lanes T at z (its double-double parts), factored in the given storage
    pub fn mixed_lu<T: Lane>(&self, z: &[&[f64]], storage: Storage, par: bool) -> Result<MixedLu, String> {
        let zd = &z[..2.min(z.len())];
        match storage {
            Storage::Dense => {
                let (c, ld) = self.jacobian_f64::<T>(zd, par)?;
                Ok(MixedLu::Dense(DDLu::factor_f64(self.len(), ld, c, par)?))
            }
            Storage::Staircase => Ok(MixedLu::Staircase(self.jacobian_staircase::<T>(zd, par)?)),
        }
    }
}
