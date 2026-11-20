//! Sub-blocks of the exact Jacobian and of the residual: `J[rows, cols]` and `F[rows]` at z from the evaluations that
//! reach them, bit for bit the entries of [`Shooting::jacobian`] and [`Shooting::residual`] there.
//!
//! Block b's rows depend on block b's evaluations alone. Entry (r, c) of the Jacobian in them is the value of the
//! column task of block b that seeds c (zero when there is none), plus the -1 of b's target; in a projected field's
//! rows it also reads the task's other rows of that field, which the same evaluation gives. An extra row's entries
//! are its weights. A sub-block then costs the column tasks (or the blocks) it meets, not the whole Jacobian (or
//! residual), and each entry is formed by the operations, in the order, that the whole one forms it by.
use crate::equations::Equations;
use crate::jacobian::Seed;
use crate::shooting::Shooting;
use multidouble::field::Fl;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use rayon::prelude::*;
use std::collections::HashMap;

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// The blocks whose rows meet `rows`, increasing; the extra rows meet none
    fn blocks_of(&self, rows: &[usize]) -> Vec<usize> {
        let re = self.row_extra();
        let mut blocks: Vec<usize> = rows
            .iter()
            .filter(|&&r| r < re)
            .map(|&r| self.block_of_row(r))
            .collect();
        blocks.sort_unstable();
        blocks.dedup();
        blocks
    }

    /// the block whose rows hold row r, below the extra rows
    fn block_of_row(&self, r: usize) -> usize {
        r / (E::NF * Self::M)
    }

    /// The first of the evaluations a sub-block needs dropped, under `Control::SubSkip`
    fn sub_skip(&self, _need: &mut Vec<usize>) {
        #[cfg(feature = "mutate")]
        if self.control == crate::control::Control::SubSkip && !_need.is_empty() {
            _need.remove(0);
        }
    }

    /// `J[rows[i], cols[j]]` at z, as PARTS part arrays of rows.len() x cols.len() entries, row-major (part q of entry
    /// (i, j) at `out[q][i * cols.len() + j]`), bit for bit the entries of `jacobian::<P>` there. Indices may repeat
    /// and come in any order.
    ///
    /// Only the column tasks ([`Shooting::jac_tasks`]) whose block's rows meet `rows` and whose columns meet `cols`
    /// are evaluated, by [`Shooting::jac_column`] with `jacobian`'s globals and preparation, on rayon tasks with
    /// `par`. Each entry is formed in the order of `jacobian`'s assembly: the task's value (zero where no task
    /// writes), the target's -1, in a projected field's rows the projector Pi on the column's rows of the field and
    /// I - Pi in the pinned unknowns' columns, the extra rows' weights, and the row scaling (`row_scale`). Projected
    /// rows and a row scaling are supported, by `jacobian`'s operations on the same values.
    ///
    /// The system must be square and every index below `len()`. The preparation (`Equations::try_prepare`) is
    /// formed, and its error returned, only when a task is evaluated.
    pub fn jacobian_sub<P: Precision<Native>>(
        &self,
        z: &[&[f64]],
        rows: &[usize],
        cols: &[usize],
        par: bool,
    ) -> Result<Vec<Vec<f64>>, String> {
        self.check_z::<P>(z)?;
        if !self.is_square() {
            return Err(format!(
                "jacobian_sub: {} rows for {} unknowns",
                self.rows(),
                self.len()
            ));
        }
        let l = self.len();
        if let Some(&bad) = rows.iter().chain(cols).find(|&&i| i >= l) {
            return Err(format!("jacobian_sub: index {bad} out of range for {l} unknowns"));
        }
        let scale = self.jac_row_factors::<P>()?;
        let m = Self::M;
        let re = self.row_extra();
        let tasks = self.jac_tasks();
        let blocks = self.blocks_of(rows);
        // (block, column) -> (task, the column's place among the task's columns), for the blocks the rows meet
        let mut owner: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        for (ti, &(b, seed)) in tasks.iter().enumerate() {
            if blocks.binary_search(&b).is_ok() {
                for (jj, c) in self.seed_columns(b, seed).into_iter().enumerate() {
                    owner.insert((b, c), (ti, jj));
                }
            }
        }
        let owner = &owner;
        let mut need: Vec<usize> = blocks
            .iter()
            .flat_map(|&b| cols.iter().filter_map(move |&c| owner.get(&(b, c)).map(|&(ti, _)| ti)))
            .collect();
        need.sort_unstable();
        need.dedup();
        self.sub_skip(&mut need);
        let vals: HashMap<usize, Vec<P::Scalar>> = if need.is_empty() {
            HashMap::new()
        } else {
            let g = self.globals::<P>(z);
            let prep = self.prep(z)?;
            let column = |&ti: &usize| (ti, self.jac_column::<P>(z, &g, &prep, tasks[ti]));
            if par {
                need.par_iter().map(column).collect()
            } else {
                need.iter().map(column).collect()
            }
        };
        let put = |v: &mut [f64], x: P::Scalar| {
            for (q, vq) in v.iter_mut().enumerate() {
                *vq = P::part(x, q);
            }
        };
        // entry (row rr of block b, column c) before the projection: the task's value, then the target's -1
        let unprojected = |b: usize, rr: usize, c: usize| -> Vec<f64> {
            let mut v = vec![0.0; P::PARTS];
            if let Some(&(ti, jj)) = owner.get(&(b, c)) {
                // a pointwise task's column jj holds point jj of every output field only
                let hit = !matches!(tasks[ti].1, Seed::Points(_)) || rr % m == jj;
                #[cfg(feature = "mutate")]
                let hit = hit && self.control != crate::control::Control::DropColumn(c);
                if let (true, Some(col)) = (hit, vals.get(&ti)) {
                    put(&mut v, col[rr]);
                }
            }
            if c == self.block_target(b) + rr {
                let s = P::from_raw(&v) + -P::ONE;
                put(&mut v, s);
            }
            v
        };
        let dd = self.projector_dd();
        // (projection, column) -> the column's M rows of the field after Pi, as `project_columns` leaves them
        let mut projected: HashMap<(usize, usize), Vec<Vec<f64>>> = HashMap::new();
        let (nr, nc) = (rows.len(), cols.len());
        let mut out: Vec<Vec<f64>> = (0..P::PARTS).map(|_| vec![0.0; nr * nc]).collect();
        for (i, &r) in rows.iter().enumerate() {
            // (block, row in it, projection of its field) for a block's row; None for an extra row
            let at = (r < re).then(|| {
                let b = self.block_of_row(r);
                let rr = r - self.row_block(b);
                (
                    b,
                    rr,
                    self.projected.iter().position(|p| p.block == b && p.field == rr / m),
                )
            });
            for (j, &c) in cols.iter().enumerate() {
                let mut v = match at {
                    Some((b, rr, Some(pi))) => {
                        let p = &self.projected[pi];
                        let ii = rr % m;
                        let x = projected.entry((pi, c)).or_insert_with(|| {
                            let x: Vec<Vec<f64>> = (0..m).map(|k| unprojected(b, p.field * m + k, c)).collect();
                            // only the block's own columns and its target's have entries in these rows
                            if x.iter().all(|v| v[0] == 0.0) {
                                return x;
                            }
                            let xs: Vec<P::Scalar> = x.iter().map(|v| P::from_raw(v)).collect();
                            p.apply::<P>(&xs, None, dd)
                                .into_iter()
                                .map(|y| (0..P::PARTS).map(|q| P::part(y, q)).collect())
                                .collect()
                        });
                        let mut v = x[ii].clone();
                        if c >= p.pin && c < p.pin + m {
                            let s = P::from_raw(&v) + p.entry::<P>(false, ii, c - p.pin, dd);
                            put(&mut v, s);
                        }
                        v
                    }
                    Some((b, rr, None)) => unprojected(b, rr, c),
                    None => {
                        let mut v = vec![0.0; P::PARTS];
                        for &(ci, w) in &self.extra[r - re].terms {
                            if ci == c {
                                let s = P::from_raw(&v) + P::from_dd(w);
                                put(&mut v, s);
                            }
                        }
                        v
                    }
                };
                if !scale.is_empty() {
                    let s = Self::jac_scaled::<P>(&v, scale[r]);
                    put(&mut v, s);
                }
                for (q, oq) in out.iter_mut().enumerate() {
                    oq[i * nc + j] = v[q];
                }
            }
        }
        Ok(out)
    }

    /// `F[rows[i]]` at z, as PARTS part arrays of rows.len() entries, bit for bit the entries of `residual::<P>` there.
    /// Indices may repeat and come in any order.
    ///
    /// Only the blocks whose rows meet `rows` are evaluated, by [`Shooting::block_state`] with `residual`'s globals
    /// and preparation, on rayon tasks with `par`. Each row is formed by `residual`'s operations: its block's state
    /// minus the target node's field, in a projected field's rows Pi on the field's rows plus I - Pi on the pinned
    /// unknowns, an extra row's [`Shooting::extra_row`], then its `row_scale` factor. Projected rows and a row scaling
    /// are supported.
    ///
    /// Every index is below `rows()`. The preparation (`Equations::try_prepare`) is formed, and its error returned,
    /// only when a block is evaluated.
    pub fn residual_sub<P: Precision<Native>>(
        &self,
        z: &[&[f64]],
        rows: &[usize],
        par: bool,
    ) -> Result<Vec<Vec<f64>>, String> {
        self.check_z::<P>(z)?;
        let n = self.rows();
        if let Some(&bad) = rows.iter().find(|&&r| r >= n) {
            return Err(format!("residual_sub: row {bad} out of range for {n} rows"));
        }
        if !self.row_scale.is_empty() && self.row_scale.len() != n {
            return Err(format!("row_scale has {} factors for {n} rows", self.row_scale.len()));
        }
        let m = Self::M;
        let re = self.row_extra();
        let mut blocks = self.blocks_of(rows);
        self.sub_skip(&mut blocks);
        let mut f: HashMap<usize, Vec<P::Scalar>> = if blocks.is_empty() {
            HashMap::new()
        } else {
            let g: Vec<P::V> = self.globals::<P>(z).into_iter().map(P::splat).collect();
            let prep = self.prep(z)?;
            let one = |&b: &usize| {
                let inp: Vec<Fl<P::V, B>> = self.block_inputs(b).iter().map(|&o| self.load::<P>(z, o)).collect();
                (
                    b,
                    self.block_rows::<P>(z, b, &self.block_state::<P::V>(b, &inp, &g, &prep)),
                )
            };
            if par {
                blocks.par_iter().map(one).collect()
            } else {
                blocks.iter().map(one).collect()
            }
        };
        let dd = self.projector_dd();
        for p in &self.projected {
            if let Some(fb) = f.get_mut(&p.block) {
                let r0 = p.field * m;
                let y = p.apply::<P>(&fb[r0..r0 + m], Some(&self.pinned::<P>(z, p)), dd);
                fb[r0..r0 + m].copy_from_slice(&y);
            }
        }
        let mut out: Vec<Vec<f64>> = (0..P::PARTS).map(|_| vec![0.0; rows.len()]).collect();
        for (i, &r) in rows.iter().enumerate() {
            let mut v = if r < re {
                let b = self.block_of_row(r);
                f.get(&b).map_or(P::ZERO, |fb| fb[r - self.row_block(b)])
            } else {
                self.extra_row::<P>(z, r - re)
            };
            if !self.row_scale.is_empty() {
                v = v * Self::row_factor::<P>(self.row_scale[r]);
            }
            for (q, oq) in out.iter_mut().enumerate() {
                oq[i] = P::part(v, q);
            }
        }
        Ok(out)
    }
}
