//! Moving a solution to another layout of the same problem: new node positions ([`Shooting::renode`]) or another
//! number of collocation points ([`Shooting::resample`]), as the start of a Newton solve there. Both work in
//! double-double on the hi and lo parts of z; [`Shooting::resample_in`] moves z in any precision, between uniform or
//! mapped grids ([`crate::interp`]). [`Shooting::split_segments`] adds nodes at existing step points, so
//! that long segments become several blocks that run on separate tasks, in any precision and without changing the
//! discrete problem.
use crate::equations::{Equations, State, Work};
use crate::interp::{interp_between, ArctanMap, Interp};
use crate::lane::DV;
use crate::periodic::{Parity, Resampler};
use crate::shooting::Shooting;
use crate::taylor::{plan_path, taylor_path, Plan};
use multidouble::dd::DD;
use multidouble::field::Fl;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use rayon::prelude::*;

/// The parity of each field ([`Parity`]): of the left endpoint's data (NL), of a node's state (NF) and of the right
/// endpoint's data (NR), which fixes its grid and harmonics. The equations know these from their derivative matrices;
/// the layout does not.
#[derive(Clone, Debug)]
pub struct Parities {
    pub left: Vec<Parity>,
    pub node: Vec<Parity>,
    pub right: Vec<Parity>,
}

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// The unknowns (hi, lo) for the nodes `nodes` (increasing) from the solution z = (zh, zl) of this layout: a new
    /// node's state is the old one where the node is unchanged, the left endpoint's map summed to it below the first
    /// old node and the right endpoint's beyond the last (meaningful where those maps are series valid there, as at a
    /// singular end), and otherwise the Taylor path from the nearest old node below, its steps planned by `plan`.
    /// The endpoints' data and the globals are kept. The new layout is `Shooting::new` on these nodes, planned again.
    /// The new nodes' states are formed on rayon tasks; the result does not depend on it.
    pub fn renode(&self, zh: &[f64], zl: &[f64], nodes: &[f64], plan: &Plan) -> Result<(Vec<f64>, Vec<f64>), String> {
        if zh.len() != self.len() || zl.len() != self.len() {
            return Err(format!("renode: z of length {}, not {}", zh.len(), self.len()));
        }
        if nodes.is_empty() || nodes.windows(2).any(|w| w[0] >= w[1]) {
            return Err("renode: the nodes must be one or more, increasing".into());
        }
        let m = Self::M;
        let (k0, og) = (self.k(), self.off_global());
        let g: Vec<DV> = (0..E::NG).map(|i| DV::splat(DD::new(zh[og + i], zl[og + i]))).collect();
        let prep = self.prep(&[zh, zl])?;
        let fld = |o: usize| Fl::<DV, B>::load(&zh[o..o + m], &zl[o..o + m]);
        let fields = |o: usize, n: usize| -> Vec<Fl<DV, B>> { (0..n).map(|f| fld(o + f * m)).collect() };
        let (left, right) = (fields(0, E::NL), fields(self.off_right(), E::NR));
        let nl = E::NL * m;
        let len = nl + nodes.len() * E::NF * m + E::NR * m + E::NG;
        let (mut oh, mut ol) = (vec![0.0; len], vec![0.0; len]);
        oh[..nl].copy_from_slice(&zh[..nl]);
        ol[..nl].copy_from_slice(&zl[..nl]);
        let tail = nl + nodes.len() * E::NF * m;
        oh[tail..].copy_from_slice(&zh[self.off_right()..]);
        ol[tail..].copy_from_slice(&zl[self.off_right()..]);
        // every new node's state is independent of the others': each on its own rayon task
        let states: Vec<State<DV, B>> = nodes
            .par_iter()
            .map(|&x| {
                if let Some(i) = self.nodes.iter().position(|&y| y == x) {
                    fields(self.off_node(i), E::NF)
                } else if x < self.nodes[0] {
                    self.eq.left_state(x, &left, &g, &prep)
                } else if x > self.nodes[k0 - 1] {
                    self.eq.right_state(x, &right, &g, &prep)
                } else {
                    let i = (0..k0).rev().find(|&i| self.nodes[i] <= x).unwrap_or(0);
                    let s0 = fields(self.off_node(i), E::NF);
                    let xs = plan_path(&self.eq, &s0, self.nodes[i], x, &g, plan);
                    let mut w = Work::<DV, B>::new(E::NF, E::NAUX, plan.kt);
                    taylor_path(&self.eq, &s0, &xs, &g, &mut w)
                }
            })
            .collect();
        for (ix, st) in states.iter().enumerate() {
            let o = nl + ix * E::NF * m;
            for (f, v) in st.iter().enumerate() {
                v.store(&mut oh[o + f * m..o + (f + 1) * m], &mut ol[o + f * m..o + (f + 1) * m]);
            }
        }
        Ok((oh, ol))
    }

    /// The unknowns (hi, lo) of the layout `to`, with M2 = 8 B2 collocation points, from the solution z = (zh, zl) of
    /// this one: every field moved by trigonometric interpolation in its parity ([`Resampler`]), the globals kept.
    /// `to` must have the same nodes, fields and globals.
    pub fn resample<E2: Equations<B2>, const B2: usize>(
        &self,
        zh: &[f64],
        zl: &[f64],
        parities: &Parities,
        to: &Shooting<E2, B2>,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        if zh.len() != self.len() || zl.len() != self.len() {
            return Err(format!("resample: z of length {}, not {}", zh.len(), self.len()));
        }
        if (E2::NL, E2::NF, E2::NR, E2::NG, to.k()) != (E::NL, E::NF, E::NR, E::NG, self.k()) {
            return Err("resample: the layouts differ in their fields, globals or nodes".into());
        }
        if (parities.left.len(), parities.node.len(), parities.right.len()) != (E::NL, E::NF, E::NR) {
            return Err("resample: one parity per field of the left data, a node and the right data".into());
        }
        let (m, m2) = (Self::M, Shooting::<E2, B2>::M);
        let rs = |p: Parity| Resampler::new(p, m, m2);
        let (full, odd, even) = (rs(Parity::Full), rs(Parity::Odd), rs(Parity::Even));
        let pick = |p: Parity| match p {
            Parity::Full => &full,
            Parity::Odd => &odd,
            Parity::Even => &even,
        };
        // the parity of each old field block, in z's order
        let mut kinds: Vec<Parity> = parities.left.clone();
        for _ in 0..self.k() {
            kinds.extend(&parities.node);
        }
        kinds.extend(&parities.right);
        let mut oh = Vec::with_capacity(to.len());
        let mut ol = Vec::with_capacity(to.len());
        for (b, &p) in kinds.iter().enumerate() {
            let f: Vec<DD> = (b * m..(b + 1) * m).map(|i| DD::new(zh[i], zl[i])).collect();
            for v in pick(p).apply(&f) {
                oh.push(v.hi);
                ol.push(v.lo);
            }
        }
        let og = self.off_global();
        oh.extend(&zh[og..]);
        ol.extend(&zl[og..]);
        Ok((oh, ol))
    }

    /// The unknowns of the layout `to` (M2 = 8 B2 collocation points) from the solution z of this one in P (P::PARTS
    /// part arrays, given and returned): every field moved by [`crate::interp::interp_between`] in its parity, from
    /// this layout's grid (the map `from`, None: uniform) to `to`'s (the map `to_map`), its entries in octo-double read
    /// in P's leading parts, so that a quad- or octo-double solution keeps its precision; the globals kept. Between
    /// uniform grids it is [`Shooting::resample`]'s map, to rounding. `to` must have the same nodes, fields and
    /// globals.
    pub fn resample_in<P: Precision<Native>, E2: Equations<B2>, const B2: usize>(
        &self,
        z: &[&[f64]],
        parities: &Parities,
        to: &Shooting<E2, B2>,
        from: Option<&ArctanMap>,
        to_map: Option<&ArctanMap>,
    ) -> Result<Vec<Vec<f64>>, String> {
        self.check_z::<P>(z)?;
        if (E2::NL, E2::NF, E2::NR, E2::NG, to.k()) != (E::NL, E::NF, E::NR, E::NG, self.k()) {
            return Err("resample_in: the layouts differ in their fields, globals or nodes".into());
        }
        if (parities.left.len(), parities.node.len(), parities.right.len()) != (E::NL, E::NF, E::NR) {
            return Err("resample_in: one parity per field of the left data, a node and the right data".into());
        }
        let (m, m2) = (Self::M, Shooting::<E2, B2>::M);
        let maps: Vec<(Parity, Interp)> = [Parity::Full, Parity::Odd, Parity::Even]
            .into_iter()
            .map(|p| (p, interp_between(p, m, m2, from, to_map)))
            .collect();
        let pick = |p: Parity| &maps.iter().find(|q| q.0 == p).expect("every parity").1;
        let mut kinds: Vec<Parity> = parities.left.clone();
        for _ in 0..self.k() {
            kinds.extend(&parities.node);
        }
        kinds.extend(&parities.right);
        let scalar = |i: usize| P::from_raw(&z.iter().map(|q| q[i]).collect::<Vec<_>>());
        let mut out: Vec<P::Scalar> = Vec::with_capacity(to.len());
        for (b, &p) in kinds.iter().enumerate() {
            let f: Vec<P::Scalar> = (b * m..(b + 1) * m).map(scalar).collect();
            out.extend(pick(p).apply::<P>(&f));
        }
        out.extend((self.off_global()..self.len()).map(scalar));
        Ok((0..P::PARTS)
            .map(|q| out.iter().map(|&v| P::part(v, q)).collect())
            .collect())
    }

    /// Splits every segment of more than `max_steps` Taylor steps at its own step points into near-equal pieces of at
    /// most `max_steps` steps, each a block of its own, and returns z on the new layout (P::PARTS part arrays).
    ///
    /// The residual, J v and the jets run one task per block, so a few long segments leave threads idle; with
    /// `max_steps` about the total steps over the threads, the blocks balance. Piece i of segment b starts where
    /// piece i - 1 ends, so a new node's state is the segment's path from its source in z to that point in P, each
    /// piece with work of its own, as its block forms it. At the returned z the new layout's residual has the old
    /// one's rows, bit for bit, at their new places (a segment's rows at its last piece, whose target is the old
    /// one), and +0 in the rows of the pieces before it; its Jacobian and Newton's iterates differ, as the unknowns do.
    ///
    /// The layout is moved with the nodes: `nodes`, `segs` and `match_node` (pieces keep their segment's direction),
    /// the extra rows' unknowns, the projected rows' blocks (a segment's last piece) and pinned unknowns, and
    /// `row_scale` (each piece takes its segment's factors). A projection whose pinned unknowns span two nodes that the
    /// split moves apart is an error, and so is a `row_scale` of the wrong length; on an error the layout is unchanged.
    /// The globals are read from z as the residual reads them; the segments are split on rayon tasks with `par`, the
    /// result the same.
    pub fn split_segments<P: Precision<Native>>(
        &mut self,
        z: &[&[f64]],
        max_steps: usize,
        par: bool,
    ) -> Result<Vec<Vec<f64>>, String> {
        self.check_z::<P>(z)?;
        if max_steps == 0 {
            return Err("split_segments: max_steps must be at least 1".into());
        }
        let (m, k) = (Self::M, self.k());
        let nfm = E::NF * m;
        // boundary step indices of segment b's pieces, 0 = idx_0 < ... < idx_p = steps
        let cuts = |b: usize| -> Vec<usize> {
            let s = self.segs[b - 1].len() - 1;
            let p = s.div_ceil(max_steps).max(1);
            (0..=p).map(|i| i * s / p).collect()
        };
        let g: Vec<P::V> = self.globals::<P>(z).into_iter().map(P::splat).collect();
        // the states at segment b's interior cuts, in the segment's direction, each piece's path from the state the
        // piece before it ends at, stored to P's parts and read back as its block reads it
        let interior = |b: usize| -> Vec<Vec<Vec<f64>>> {
            let (c, pts) = (cuts(b), &self.segs[b - 1]);
            let o = self.off_node(self.seg_source(b));
            let mut s: Vec<Fl<P::V, B>> = (0..E::NF).map(|f| self.load::<P>(z, o + f * m)).collect();
            let mut out = Vec::with_capacity(c.len().saturating_sub(2));
            for w in c.windows(2).take(c.len() - 2) {
                let mut work = Work::<P::V, B>::new(E::NF, E::NAUX, self.kt);
                let st = taylor_path(&self.eq, &s, &pts[w[0]..=w[1]], &g, &mut work);
                let parts: Vec<Vec<f64>> = (0..P::PARTS)
                    .map(|q| {
                        st.iter()
                            .flat_map(|f| (0..m).map(move |i| P::part(P::field_get(f, i), q)))
                            .collect()
                    })
                    .collect();
                s = (0..E::NF)
                    .map(|f| {
                        let pf: Vec<&[f64]> = parts.iter().map(|p| &p[f * m..(f + 1) * m]).collect();
                        P::load_field(&pf)
                    })
                    .collect();
                out.push(parts);
            }
            out
        };
        let states: Vec<Vec<Vec<Vec<f64>>>> = if par {
            (1..k).into_par_iter().map(interior).collect()
        } else {
            (1..k).map(interior).collect()
        };
        // the new nodes, segments, and each old node's new index
        let mut nodes = vec![self.nodes[0]];
        let mut segs: Vec<Vec<f64>> = Vec::new();
        // per new node: Ok(n) for old node n, Err((b, i)) for interior cut i of segment b
        let mut src: Vec<Result<usize, (usize, usize)>> = vec![Ok(0)];
        let mut new_idx = vec![0usize];
        for b in 1..k {
            let (c, pts) = (cuts(b), &self.segs[b - 1]);
            let p = c.len() - 1;
            let pieces: Vec<Vec<f64>> = c.windows(2).map(|w| pts[w[0]..=w[1]].to_vec()).collect();
            // in increasing x: an outward segment's pieces in order, an inward one's reversed
            let inward = self.seg_inward(b);
            let order: Vec<usize> = if inward {
                (0..p).rev().collect()
            } else {
                (0..p).collect()
            };
            for (n, &i) in order.iter().enumerate() {
                segs.push(pieces[i].clone());
                if n + 1 < p {
                    // the node after this piece in x: an interior cut, pieces[i]'s end outward, its start inward
                    let cut = if inward { i - 1 } else { i };
                    nodes.push(pts[c[cut + 1]]);
                    src.push(Err((b, cut)));
                }
            }
            nodes.push(self.nodes[b]);
            src.push(Ok(b));
            new_idx.push(nodes.len() - 1);
        }
        let k2 = nodes.len();
        let shift = (k2 - k) * nfm;
        let (o0, oright) = (self.off_node(0), self.off_right());
        let inward: Vec<bool> = (0..=k).map(|b| b > 0 && b < k && self.seg_inward(b)).collect();
        let map_unknown = |i: usize| -> usize {
            if i < o0 {
                i
            } else if i < oright {
                let n = (i - o0) / nfm;
                i + (new_idx[n] - n) * nfm
            } else {
                i + shift
            }
        };
        // old block b -> the new block that carries its rows
        let map_block = |b: usize| -> usize {
            if b == 0 {
                0
            } else if b == k {
                k2
            } else if inward[b] {
                new_idx[b - 1] + 1
            } else {
                new_idx[b]
            }
        };
        // the new z
        let len2 = self.len() + shift;
        let mut out = vec![vec![0.0; len2]; P::PARTS];
        for (q, oq) in out.iter_mut().enumerate() {
            oq[..o0].copy_from_slice(&z[q][..o0]);
            oq[o0 + k2 * nfm..].copy_from_slice(&z[q][self.off_right()..]);
            for (n2, s) in src.iter().enumerate() {
                let dst = &mut oq[o0 + n2 * nfm..o0 + (n2 + 1) * nfm];
                match *s {
                    Ok(n) => dst.copy_from_slice(&z[q][self.off_node(n)..self.off_node(n) + nfm]),
                    Err((b, i)) => dst.copy_from_slice(&states[b - 1][i][q]),
                }
            }
        }
        // a projection's M pinned unknowns stay consecutive only if no new node falls inside them; checked before the
        // layout changes, so that an error leaves it as it was
        if let Some(p) = self
            .projected
            .iter()
            .find(|p| (0..m).any(|j| map_unknown(p.pin + j) != map_unknown(p.pin) + j))
        {
            return Err(format!(
                "split_segments: the pinned unknowns {}..{} of field {} of block {} span a node the split moves",
                p.pin,
                p.pin + m,
                p.field,
                p.block
            ));
        }
        // the layout's rows and conditions, moved
        let row_scale = if self.row_scale.is_empty() {
            Vec::new()
        } else {
            if self.row_scale.len() != self.rows() {
                return Err(format!(
                    "split_segments: row_scale has {} factors for {} rows",
                    self.row_scale.len(),
                    self.rows()
                ));
            }
            // new block j's factors: its old block's, a piece taking its segment's
            let mut old_of = vec![0usize; k2 + 1];
            for b in 0..=k {
                old_of[map_block(b)] = b;
            }
            for b in 1..k {
                let (lo, hi) = (new_idx[b - 1], new_idx[b]);
                for o in old_of.iter_mut().take(hi + 1).skip(lo + 1) {
                    *o = b;
                }
            }
            let mut rs = Vec::with_capacity(self.rows() + shift);
            for &b in &old_of {
                rs.extend_from_slice(&self.row_scale[self.row_block(b)..self.row_block(b) + nfm]);
            }
            rs.extend_from_slice(&self.row_scale[self.row_extra()..]);
            rs
        };
        for row in &mut self.extra {
            for t in &mut row.terms {
                t.0 = map_unknown(t.0);
            }
        }
        for p in &mut self.projected {
            p.block = map_block(p.block);
            p.pin = map_unknown(p.pin);
        }
        self.match_node = new_idx[self.match_node];
        self.row_scale = row_scale;
        self.nodes = nodes;
        self.segs = segs;
        Ok(out)
    }
}
