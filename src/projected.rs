//! Projected rows: one field's rows of one block replaced by a projected condition, a constraint of the layout.
//!
//! With Pi the orthogonal projector onto the harmonics k <= kmax of the field's periodic grid and Pi' = I - Pi, the
//! M rows of field q of block b become
//!
//! ```text
//! Pi (state_q - target_q) + Pi' y
//! ```
//!
//! with y the M unknowns at `pin`: the matching of the field's high harmonics is dropped, and the high harmonics of
//! the pinned field are held at zero in its place. As Pi and Pi' are complementary orthogonal projectors, the rows
//! vanish exactly when the mismatch has no low harmonics and y no high ones. It suits high harmonics that reach the
//! matching only through exponentially small couplings, whose matching is ill-conditioned.
//!
//! On the M points of the grid of `parity` (the full grid theta_j = 2 pi j / M, or a half grid theta_j = pi j / M
//! with the field's symmetry under theta -> theta + pi), the discrete orthogonality of the harmonics gives
//! `Pi_ij = (1 / M) sum_k w_k cos(k (theta_i - theta_j))`, over the harmonics of the parity up to kmax, with w_k = 1
//! for k = 0 and the Nyquist harmonic and 2 otherwise. It depends on |i - j| only. Its entries are formed once in
//! octo-double, from cosines of exact multiples of pi ([`crate::periodic::sin_cos_pi_od`]), and each precision
//! reads their leading parts, so the rows are exact to the precision of the solve. A projector rounded to
//! double-double would leave a quad- or octo-double solution about 2^-106 |y| off the exact rows.
use crate::equations::Equations;
use crate::periodic::{sin_cos_pi_od, Parity};
use crate::shooting::Shooting;
use multidouble::dd::DD;
use multidouble::od::OD;
use multidouble::precision::{Double, Precision};
use multidouble::simd::Native;

/// The projected rows of one field of one block ([`crate::projected`])
#[derive(Clone, Debug)]
pub struct ProjectedRows {
    /// the residual block
    pub block: usize,
    /// the field of the block whose rows are projected
    pub field: usize,
    /// the offset in z of the M unknowns whose high harmonics are held at zero
    pub pin: usize,
    /// the grid of the field and of the pinned unknowns
    pub parity: Parity,
    /// the highest harmonic kept by Pi
    pub kmax: usize,
    /// Pi_ij = low[|i - j|] and (I - Pi)_ij = high[|i - j|], in octo-double
    low: Vec<OD>,
    high: Vec<OD>,
}

/// the harmonics of the grid of `parity` with M points, up to kmax, and their weights w_k
fn harmonics(parity: Parity, m: usize, kmax: usize) -> Vec<(usize, f64)> {
    let (top, step, start) = match parity {
        Parity::Full => (m / 2, 1, 0),
        Parity::Odd => (m - 1, 2, 1),
        Parity::Even => (m, 2, 0),
    };
    (start..=top.min(kmax))
        .step_by(step)
        .map(|k| {
            (
                k,
                if k == 0 || (k == top && parity != Parity::Odd) {
                    1.0
                } else {
                    2.0
                },
            )
        })
        .collect()
}

impl ProjectedRows {
    /// The projected rows of field `field` of block `block` on M points, pinning the unknowns at `pin`, with the
    /// projector onto the harmonics k <= kmax of the grid of `parity`.
    pub fn new(m: usize, block: usize, field: usize, pin: usize, parity: Parity, kmax: usize) -> Self {
        // cos(t pi / M), t < 2M: every angle of the entries, k (theta_i - theta_j) = (k or 2k) |i - j| pi / M
        let cosines: Vec<OD> = (0..2 * m).map(|t| sin_cos_pi_od(t as i64, m as i64).1).collect();
        let mult = if parity == Parity::Full { 2 } else { 1 };
        let hs = harmonics(parity, m, kmax);
        let inv_m = OD::from_f64(1.0) / OD::from_f64(m as f64);
        let low: Vec<OD> = (0..m)
            .map(|l| {
                let s = hs.iter().fold(OD::from_f64(0.0), |a, &(k, w)| {
                    a + cosines[(mult * k * l) % (2 * m)].mul_f(w)
                });
                s * inv_m
            })
            .collect();
        let high = low
            .iter()
            .enumerate()
            .map(|(l, &p)| if l == 0 { OD::from_f64(1.0) - p } else { -p })
            .collect();
        ProjectedRows {
            block,
            field,
            pin,
            parity,
            kmax,
            low,
            high,
        }
    }

    /// Entry (i, j) of Pi (`low`) or I - Pi in P: the leading PARTS parts of its octo-double value, or two of them
    /// under the control `ProjectorDD` (`dd`)
    pub fn entry<P: Precision<Native>>(&self, low: bool, i: usize, j: usize, dd: bool) -> P::Scalar {
        let v = if low { &self.low } else { &self.high }[i.abs_diff(j)];
        P::new(&v.x[..if dd { 2 } else { P::PARTS }])
    }

    /// Pi x + (I - Pi) y in P (x, y of M values; y absent: Pi x), each row one sum of products in P's accumulator
    pub fn apply<P: Precision<Native>>(&self, x: &[P::Scalar], y: Option<&[P::Scalar]>, dd: bool) -> Vec<P::Scalar> {
        let m = self.low.len();
        assert!(
            x.len() == m && y.is_none_or(|y| y.len() == m),
            "ProjectedRows::apply: values for {m} points"
        );
        // an entry depends on |i - j| only: each of the m values of Pi (and of I - Pi) is converted to P once
        let conv = |low: bool| -> Vec<P::Scalar> { (0..m).map(|d| self.entry::<P>(low, d, 0, dd)).collect() };
        let pl = conv(true);
        let ph = if y.is_some() { conv(false) } else { Vec::new() };
        (0..m)
            .map(|i| {
                let lo = (0..m).map(|j| (pl[i.abs_diff(j)], x[j]));
                match y {
                    Some(y) => P::dot(lo.chain((0..m).map(|j| (ph[i.abs_diff(j)], y[j])))),
                    None => P::dot(lo),
                }
            })
            .collect()
    }
}

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// whether the projectors are read in double-double, under `Control::ProjectorDD`
    pub(crate) fn projector_dd(&self) -> bool {
        #[cfg(feature = "mutate")]
        if self.control == crate::control::Control::ProjectorDD {
            return true;
        }
        false
    }

    /// Projects field `field`'s rows of block `block` ([`crate::projected`]): Pi onto the harmonics k <= kmax of
    /// the grid of `parity`, the unknowns at `pin` (M of them, before the globals, apart from any other projection's)
    /// held to Pi'.
    pub fn project_rows(
        &mut self,
        block: usize,
        field: usize,
        pin: usize,
        parity: Parity,
        kmax: usize,
    ) -> Result<(), String> {
        let m = Self::M;
        if block > self.k() || field >= E::NF {
            return Err(format!("project_rows: no field {field} of block {block}"));
        }
        if pin.checked_add(m).is_none_or(|end| end > self.off_global()) {
            return Err(format!(
                "project_rows: the pinned unknowns from {pin} pass the fields (they end at {})",
                self.off_global()
            ));
        }
        if self.projected.iter().any(|p| p.block == block && p.field == field) {
            return Err(format!(
                "project_rows: field {field} of block {block} is projected already"
            ));
        }
        // two projections holding the same unknowns would give them two conditions and leave others without one
        if let Some(p) = self.projected.iter().find(|p| p.pin < pin + m && pin < p.pin + m) {
            return Err(format!(
                "project_rows: the pinned unknowns {pin}..{} overlap those of field {} of block {}",
                pin + m,
                p.field,
                p.block
            ));
        }
        self.projected
            .push(ProjectedRows::new(m, block, field, pin, parity, kmax));
        Ok(())
    }

    /// The projected rows of a residual, or of one coefficient of the residual's series: rows d of each projected
    /// field become Pi d + Pi' y, y the values `pinned(p)` of its pinned unknowns.
    pub(crate) fn project_residual<P: Precision<Native>>(
        &self,
        f: &mut [P::Scalar],
        pinned: impl Fn(&ProjectedRows) -> Vec<P::Scalar>,
    ) {
        let m = Self::M;
        for p in &self.projected {
            let r0 = self.row_block(p.block) + p.field * m;
            let rows = p.apply::<P>(&f[r0..r0 + m], Some(&pinned(p)), self.projector_dd());
            f[r0..r0 + m].copy_from_slice(&rows);
        }
    }

    /// The projected rows of a Jacobian (PARTS column-major part arrays, ld): each column's rows d of a projected
    /// field become Pi d (the target's -1 becomes -Pi), and the pinned unknowns' columns gain Pi'.
    pub(crate) fn project_jacobian<P: Precision<Native>>(&self, c: &mut [Vec<f64>], ld: usize) {
        self.project_columns::<P>(c, ld, self.len(), true);
    }

    /// The projected rows of block b in an f64 matrix, as [`Shooting::jacobian_f64`] and
    /// [`Shooting::jacobian_staircase`] form them, for a caller that assembles the matrix from its own column
    /// evaluations ([`Shooting::jac_column`], [`Shooting::jac_columns_f64`]). The matrix is column-major with leading
    /// dimension ld, its rows rel..rel + NF M are block b's and it has `cols` columns; `pos` maps an unknown to its
    /// column, None when it has none (an error for a pinned unknown, as is a column two pinned unknowns share). For
    /// every projection of block b, each column's rows of the projected field become Pi times them, formed in
    /// double-double and rounded to f64, and the pinned unknowns' columns gain Pi'.
    ///
    /// `dd_column(j)` gives column j's NF M entries in block b's rows in double-double, when the caller formed them so
    /// (`jac_column`): Pi then acts on those values in place of the matrix's entries, and the result is rounded only
    /// after Pi' is added when the column is a pinned unknown's, so the rows are the double-double Jacobian's rounded
    /// to f64. None reads the matrix's f64 entries.
    #[allow(clippy::too_many_arguments)]
    pub fn project_f64_block<'a>(
        &self,
        b: usize,
        w: &mut [f64],
        ld: usize,
        rel: usize,
        cols: usize,
        pos: impl Fn(usize) -> Option<usize>,
        dd_column: impl Fn(usize) -> Option<&'a [DD]>,
    ) -> Result<(), String> {
        let m = Self::M;
        let nr = E::NF * m;
        if b > self.k() {
            return Err(format!("project_f64_block: no block {b}"));
        }
        let end = rel.checked_add(nr);
        if end.is_none_or(|e| e > ld) || cols.checked_mul(ld).is_none_or(|n| w.len() < n) {
            return Err(format!(
                "project_f64_block: {nr} rows from {rel} and {cols} columns of leading dimension {ld} in {} entries",
                w.len()
            ));
        }
        let dd = self.projector_dd();
        for p in self.projected.iter().filter(|p| p.block == b) {
            let r0 = rel + p.field * m;
            // the projected rows of the columns given in double-double, kept to add the pinned unknowns' Pi'
            let mut kept: Vec<Option<Vec<DD>>> = vec![None; cols];
            for j in 0..cols {
                // only the block's own columns and its target's have entries in these rows
                let given = dd_column(j);
                let x: Vec<DD> = match given {
                    Some(col) if col.len() == nr => {
                        let x = &col[p.field * m..(p.field + 1) * m];
                        if x.iter().all(|v| v.hi == 0.0 && v.lo == 0.0) {
                            // a zero column, stored as given in place of the matrix's entries
                            for (i, v) in x.iter().enumerate() {
                                w[j * ld + r0 + i] = v.hi;
                            }
                            continue;
                        }
                        x.to_vec()
                    }
                    Some(col) => {
                        return Err(format!(
                            "project_f64_block: column {j} of block {b} has {} double-double entries, not {nr}",
                            col.len()
                        ))
                    }
                    None => {
                        if (0..m).all(|i| w[j * ld + r0 + i] == 0.0) {
                            continue;
                        }
                        (0..m).map(|i| DD::from_f64(w[j * ld + r0 + i])).collect()
                    }
                };
                let rows = p.apply::<Double>(&x, None, dd);
                for (i, v) in rows.iter().enumerate() {
                    w[j * ld + r0 + i] = v.hi;
                }
                if given.is_some() {
                    kept[j] = Some(rows);
                }
            }
            // each pinned unknown's own column, as the double-double Jacobian has
            let mut pinned: Vec<usize> = Vec::with_capacity(m);
            for jj in 0..m {
                let c = pos(p.pin + jj).filter(|&c| c < cols).ok_or_else(|| {
                    format!(
                        "projected rows of block {b}: the pinned unknown {} has no column here",
                        p.pin + jj
                    )
                })?;
                if pinned.contains(&c) {
                    return Err(format!(
                        "projected rows of block {b}: the pinned unknown {} shares column {c}",
                        p.pin + jj
                    ));
                }
                pinned.push(c);
            }
            for (jj, &c) in pinned.iter().enumerate() {
                for i in 0..m {
                    let k = c * ld + r0 + i;
                    let v = kept[c].as_ref().map_or_else(|| DD::from_f64(w[k]), |rows| rows[i]);
                    w[k] = (v + p.entry::<Double>(false, i, jj, dd)).hi;
                }
            }
        }
        Ok(())
    }

    /// `project_jacobian` on the first `ncols` columns of a matrix in P's part arrays, the pinned columns' Pi' added
    /// only with `pinned` (a mode matrix's imaginary part has none)
    pub(crate) fn project_columns<P: Precision<Native>>(
        &self,
        c: &mut [Vec<f64>],
        ld: usize,
        ncols: usize,
        pinned: bool,
    ) {
        let m = Self::M;
        let dd = self.projector_dd();
        let at = |c: &[Vec<f64>], k: usize| P::from_raw(&c.iter().map(|cq| cq[k]).collect::<Vec<_>>());
        let put = |c: &mut [Vec<f64>], k: usize, v: P::Scalar| {
            for (q, cq) in c.iter_mut().enumerate() {
                cq[k] = P::part(v, q);
            }
        };
        for p in &self.projected {
            let r0 = self.row_block(p.block) + p.field * m;
            for j in 0..ncols {
                // only the block's own columns and its target's have entries in these rows
                if (0..m).all(|i| c[0][j * ld + r0 + i] == 0.0) {
                    continue;
                }
                let x: Vec<P::Scalar> = (0..m).map(|i| at(c, j * ld + r0 + i)).collect();
                for (i, v) in p.apply::<P>(&x, None, dd).into_iter().enumerate() {
                    put(c, j * ld + r0 + i, v);
                }
            }
            for jj in 0..if pinned { m } else { 0 } {
                for i in 0..m {
                    let k = (p.pin + jj) * ld + r0 + i;
                    put(c, k, at(c, k) + p.entry::<P>(false, i, jj, dd));
                }
            }
        }
    }
}
