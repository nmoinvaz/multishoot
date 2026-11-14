//! Moving a solution to another layout of the same problem: new node positions ([`Shooting::renode`]) or another
//! number of collocation points ([`Shooting::resample`]), as the start of a Newton solve there. Both work in
//! double-double on the hi and lo parts of z.
use crate::equations::{Equations, State, Work};
use crate::lane::DV;
use crate::periodic::{Parity, Resampler};
use crate::shooting::Shooting;
use crate::taylor::{plan_path, taylor_path, Plan};
use multidouble::dd::DD;
use multidouble::field::Fl;

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
        for (ix, &x) in nodes.iter().enumerate() {
            let st: State<DV, B> = if let Some(i) = self.nodes.iter().position(|&y| y == x) {
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
            };
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
}
