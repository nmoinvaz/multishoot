//! The layout of a multiple-shooting problem: the nodes, the segments of Taylor steps between them, the unknown
//! vector and the residual's blocks.
//!
//! With K nodes x_0 < ... < x_(K-1), M = 8 B collocation points and the sizes of [`Equations`], the unknowns are
//!
//! ```text
//! z = [ left data (NL M) | node 0 (NF M) | ... | node K-1 (NF M) | right data (NR M) | globals (NG) ]
//! ```
//!
//! and the residual has K + 1 blocks of NF M rows, then one row per extra condition:
//!
//! * block 0: the state at node 0 from the left data, minus node 0;
//! * block b, 0 < b < K: the Taylor path of segment b from its source node, minus its target node;
//! * block K: the state at node K-1 from the right data, minus node K-1;
//! * the extra rows ([`ExtraRow`]): linear conditions on z, such as a phase condition.
//!
//! One field's rows of a block can be replaced by a projected condition ([`crate::projected`]), which drops the
//! matching of the field's high harmonics and holds those of other unknowns at zero in its place.
//!
//! Segment b runs outward, node b-1 to node b, for b <= `match_node`, and inward, node b to node b-1, after it. The
//! system is square when NL + NR = NF and there is one extra row per global.
use crate::equations::{Equations, State, Work};
use crate::lane::{Lane, DV};
use crate::mode::Rate;
use crate::taylor::{plan_path_within, steps_within, taylor_path, OrderChoice, OrderCost, Plan};
use multidouble::dd::DD;
use multidouble::field::Fl;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use multisolver::newton::{NewtonOpts, NewtonReport, Step};
use multisolver::{JetOrders, Jvp, Problem, Recompute, ResidualJet};
use rayon::prelude::*;

/// A linear condition on the unknowns, `sum_i w_i z_(idx_i) - rhs = 0`: a phase condition, a normalization, a pinned
/// unknown
#[derive(Clone, Debug)]
pub struct ExtraRow {
    pub terms: Vec<(usize, DD)>,
    pub rhs: DD,
}

/// A multiple-shooting problem: the equations and the layout of their unknowns and residual.
pub struct Shooting<E: Equations<B>, const B: usize> {
    pub eq: E,
    /// node positions, in the order of the unknowns
    pub nodes: Vec<f64>,
    /// segs[b-1]: the step points of segment b, from its source node to its target node, both included
    pub segs: Vec<Vec<f64>>,
    /// Taylor order of the steps
    pub kt: usize,
    /// the last segment that runs outward (default K - 1: all of them)
    pub match_node: usize,
    pub extra: Vec<ExtraRow>,
    /// factors of the residual's rows, one per row (empty: none), applied to the residual, the Jacobian and the jet;
    /// powers of two keep them exact. Newton's correction rule takes the same steps with or without them, but its
    /// residual stop (max |F| at most `tol`) reads the scaled rows; with `tol` 0 and `tol_dz` the stop does not
    /// depend on them either.
    pub row_scale: Vec<f64>,
    /// projected rows (`Shooting::project_rows`, [`crate::projected`]), at most one per field of a block
    pub projected: Vec<crate::projected::ProjectedRows>,
    #[cfg(feature = "mutate")]
    pub(crate) control: crate::control::Control,
}

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// collocation points per field
    pub const M: usize = 8 * B;

    /// A problem on the given nodes (at least one, increasing), every segment outward and a single step each (plan
    /// them with [`Shooting::plan_segments`] or set `segs`), no extra rows.
    pub fn new(eq: E, nodes: Vec<f64>, kt: usize) -> Self {
        assert!(!nodes.is_empty(), "Shooting::new: no nodes");
        assert!(
            nodes.windows(2).all(|w| w[0] < w[1]),
            "Shooting::new: nodes not increasing"
        );
        let segs = nodes.windows(2).map(|w| vec![w[0], w[1]]).collect();
        let match_node = nodes.len() - 1;
        Shooting {
            eq,
            nodes,
            segs,
            kt,
            match_node,
            extra: Vec::new(),
            row_scale: Vec::new(),
            projected: Vec::new(),
            #[cfg(feature = "mutate")]
            control: crate::control::Control::None,
        }
    }

    /// Sets the deliberate error of the residual or the Jacobian, for a control (`Control::None` to clear it).
    #[cfg(feature = "mutate")]
    pub fn set_control(&mut self, c: crate::control::Control) {
        self.control = c;
    }

    /// the number of nodes K
    pub fn k(&self) -> usize {
        self.nodes.len()
    }

    /// the number of unknowns
    pub fn len(&self) -> usize {
        self.off_global() + E::NG
    }

    /// whether there are no unknowns (never: there is at least one node)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// the number of residual rows
    pub fn rows(&self) -> usize {
        (self.k() + 1) * E::NF * Self::M + self.extra.len()
    }

    /// whether the residual has as many rows as there are unknowns
    pub fn is_square(&self) -> bool {
        self.rows() == self.len()
    }

    /// offset of node i's first field
    pub fn off_node(&self, i: usize) -> usize {
        E::NL * Self::M + E::NF * Self::M * i
    }

    /// offset of the right endpoint's data
    pub fn off_right(&self) -> usize {
        self.off_node(self.k())
    }

    /// offset of the globals
    pub fn off_global(&self) -> usize {
        self.off_right() + E::NR * Self::M
    }

    /// first row of residual block b
    pub fn row_block(&self, b: usize) -> usize {
        b * E::NF * Self::M
    }

    /// first extra row
    pub fn row_extra(&self) -> usize {
        self.row_block(self.k() + 1)
    }

    /// whether segment b (0 < b < K) runs inward, node b to node b-1
    pub fn seg_inward(&self, b: usize) -> bool {
        b > self.match_node && b < self.k()
    }

    /// the source node of segment b (0 < b < K)
    pub fn seg_source(&self, b: usize) -> usize {
        if self.seg_inward(b) {
            b
        } else {
            b - 1
        }
    }

    /// the offsets in z of the fields block b reads (besides the globals): the left data, the segment's source node,
    /// or the right data
    pub fn block_inputs(&self, b: usize) -> Vec<usize> {
        let m = Self::M;
        let k = self.k();
        if b == 0 {
            (0..E::NL).map(|f| f * m).collect()
        } else if b < k {
            let o = self.off_node(self.seg_source(b));
            (0..E::NF).map(|f| o + f * m).collect()
        } else {
            (0..E::NR).map(|f| self.off_right() + f * m).collect()
        }
    }

    /// the offset of the node whose fields block b's state is matched to
    pub fn block_target(&self, b: usize) -> usize {
        let k = self.k();
        if b == 0 {
            self.off_node(0)
        } else if b < k {
            self.off_node(if self.seg_inward(b) { b - 1 } else { b })
        } else {
            self.off_node(k - 1)
        }
    }

    /// The preparation of an evaluation at z (`Equations::try_prepare` at its double-double parts), or `prepare`'s
    /// under `Control::InfalliblePrepare`
    pub(crate) fn prep(&self, z: &[&[f64]]) -> Result<E::Prep, String> {
        #[cfg(feature = "mutate")]
        if self.control == crate::control::Control::InfalliblePrepare {
            return Ok(self.eq.prepare(z[0], z[1]));
        }
        self.eq.try_prepare(z[0], z[1])
    }

    /// The preparation of a mode-matrix evaluation at z and the rate (`Equations::try_prepare_rate`), or
    /// `prepare_rate`'s under `Control::InfalliblePrepare`
    pub(crate) fn prep_rate(&self, z: &[&[f64]], rate: Rate) -> Result<E::Prep, String> {
        #[cfg(feature = "mutate")]
        if self.control == crate::control::Control::InfalliblePrepare {
            return Ok(self.eq.prepare_rate(z[0], z[1], rate));
        }
        self.eq.try_prepare_rate(z[0], z[1], rate)
    }

    /// The state block b produces from its inputs (fields in the order of `block_inputs`), the globals and the
    /// evaluation's preparation (`Equations::try_prepare`): the left endpoint's state at node 0, segment b's Taylor
    /// path, or the right endpoint's state at node K-1.
    pub fn block_state<T: Lane>(&self, b: usize, inp: &[Fl<T, B>], g: &[T], prep: &E::Prep) -> State<T, B> {
        self.block_state_at(b, inp, g, prep, Rate::ZERO)
    }

    /// `block_state` with the Taylor steps at the growth rate of a mode-matrix evaluation (`Work::rate`); the
    /// endpoint maps take theirs from the preparation
    pub fn block_state_at<T: Lane>(
        &self,
        b: usize,
        inp: &[Fl<T, B>],
        g: &[T],
        prep: &E::Prep,
        rate: Rate,
    ) -> State<T, B> {
        let k = self.k();
        if b == 0 {
            self.eq.left_state(self.nodes[0], inp, g, prep)
        } else if b < k {
            let mut w = Work::<T, B>::new(E::NF, E::NAUX, self.kt);
            w.rate = rate;
            taylor_path(&self.eq, inp, &self.segs[b - 1], g, &mut w)
        } else {
            self.eq.right_state(self.nodes[k - 1], inp, g, prep)
        }
    }

    /// Plans every segment's steps (`plan_path`) from its source node's state in z (double-double, hi and lo parts)
    /// toward its target node, with the globals of z, at the order `plan.kt`, which becomes the problem's. The
    /// segments are independent and are planned on rayon tasks; the points do not depend on it, and a panic is the
    /// first segment's to panic, as in a serial loop.
    pub fn plan_segments(&mut self, zh: &[f64], zl: &[f64], plan: &Plan) {
        let segs = self.plan_each(zh, zl, plan, usize::MAX);
        self.segs = segs
            .into_iter()
            .map(|s| s.expect("plan_segments: no step limit"))
            .collect();
        self.kt = plan.kt;
    }

    /// Every segment's plan (`plan_path_within`, at most `max_steps` steps each) from z, on rayon tasks
    fn plan_each(&self, zh: &[f64], zl: &[f64], plan: &Plan, max_steps: usize) -> Vec<Option<Vec<f64>>> {
        assert!(
            zh.len() == self.len() && zl.len() == self.len(),
            "plan_segments: z of length {}",
            self.len()
        );
        let m = Self::M;
        let og = self.off_global();
        let g: Vec<DV> = (0..E::NG).map(|i| DV::splat(DD::new(zh[og + i], zl[og + i]))).collect();
        (1..self.k())
            .into_par_iter()
            .map(|b| {
                let src = self.seg_source(b);
                let dst = if self.seg_inward(b) { b - 1 } else { b };
                let o = self.off_node(src);
                let s0: Vec<Fl<DV, B>> = (0..E::NF)
                    .map(|f| Fl::load(&zh[o + f * m..o + (f + 1) * m], &zl[o + f * m..o + (f + 1) * m]))
                    .collect();
                plan_path_within(&self.eq, &s0, self.nodes[src], self.nodes[dst], &g, plan, max_steps)
            })
            .collect()
    }

    /// Plans the segments as [`Shooting::plan_segments`] does at each candidate order of `orders` (each at least 2,
    /// last terms below tol, steps at most hmax), and keeps the cheapest: the order of least estimated cost, the sum of
    /// the steps over the segments times `cost(k)`, the cost of one step at order k ([`crate::taylor::step_cost`] by
    /// default). A tie goes to the lower order. The order kept becomes `kt`, its points `segs`, and the result lists
    /// every candidate's steps and cost by increasing order. The default search is two passes, a ladder and the
    /// orders near its choice:
    ///
    /// ```ignore
    /// let c = sh.plan_segments_auto(&zh, &zl, tol, hmax, &Plan::candidate_orders(tol), step_cost);
    /// let c = sh.plan_segments_auto(&zh, &zl, tol, hmax, &c.refined_orders(), step_cost);
    /// ```
    ///
    /// The residual, the Jacobian (each column runs every segment) and the other evaluations take one recursion and
    /// one Horner sum per step, so their cost follows the estimate. The candidates are planned from the highest order
    /// down; once one segment of a candidate takes more steps than would cost the cheapest estimate so far, the
    /// candidate is dropped (its `steps` and `cost` are None), so that a low order with many steps is not planned in
    /// full. The choice depends on step counts and `cost` alone, never on timing, threads or the SIMD backend. It
    /// panics as `plan_segments` does on any candidate, on an empty `orders` and on a cost that is not positive and
    /// finite.
    pub fn plan_segments_auto(
        &mut self,
        zh: &[f64],
        zl: &[f64],
        tol: f64,
        hmax: f64,
        orders: &[usize],
        cost: impl Fn(usize) -> f64,
    ) -> OrderChoice {
        assert!(!orders.is_empty(), "plan_segments_auto: no candidate orders");
        let mut ks = orders.to_vec();
        ks.sort_unstable();
        ks.dedup();
        // the cheapest so far: its estimate, its order and its points
        let mut best: Option<(f64, usize, Vec<Vec<f64>>)> = None;
        let mut candidates = Vec::with_capacity(ks.len());
        for &kt in ks.iter().rev() {
            let c = cost(kt);
            assert!(c > 0.0 && c.is_finite(), "plan_segments_auto: cost {c:e} of order {kt}");
            let budget = best.as_ref().map_or(f64::INFINITY, |b| b.0);
            let plan = Plan { kt, tol, hmax };
            let segs: Option<Vec<Vec<f64>>> = self
                .plan_each(zh, zl, &plan, steps_within(budget, c))
                .into_iter()
                .collect();
            let Some(segs) = segs else {
                candidates.push(OrderCost {
                    kt,
                    steps: None,
                    cost: None,
                });
                continue;
            };
            let steps: Vec<usize> = segs.iter().map(|s| s.len() - 1).collect();
            let est = steps.iter().sum::<usize>() as f64 * c;
            // from the highest order down, so `<=` gives a tie to the lower order
            if est <= budget {
                best = Some((est, kt, segs));
            }
            candidates.push(OrderCost {
                kt,
                steps: Some(steps),
                cost: Some(est),
            });
        }
        candidates.reverse();
        let (_, kt, segs) = best.expect("the first candidate has no budget");
        #[cfg(feature = "mutate")]
        let (kt, segs) = if self.control == crate::control::Control::FirstOrder && kt != ks[0] {
            let plan = Plan { kt: ks[0], tol, hmax };
            (
                ks[0],
                self.plan_each(zh, zl, &plan, usize::MAX)
                    .into_iter()
                    .flatten()
                    .collect(),
            )
        } else {
            (kt, segs)
        };
        self.segs = segs;
        self.kt = kt;
        OrderChoice { kt, candidates }
    }
}

/// The layout as a problem for the solvers of [`multisolver`]: its residual and exact Jacobian in any precision
impl<E: Equations<B>, const B: usize, P: Precision<Native>> Problem<P> for Shooting<E, B> {
    fn len(&self) -> usize {
        Shooting::len(self)
    }
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        Shooting::residual::<P>(self, z, par)
    }
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        Shooting::jacobian::<P>(self, z, par)
    }
}

impl<E: Equations<B>, const B: usize, P: Precision<Native>> Jvp<P> for Shooting<E, B> {
    fn jvp(&self, z: &[&[f64]], v: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        Ok(Shooting::jvp::<P>(self, z, v, par)?.0)
    }
}

/// The residual's series in the equations' parameter ([`Shooting::residual_jet`]) on jet lanes of 8, 16 or 32
/// coefficients, the smallest that holds the orders asked for and the coefficient vectors given: each size is one
/// instance of the residual's code, and a jet's products cost about the square of its size.
impl<E: Equations<B>, const B: usize, P: Precision<Native>> ResidualJet<P> for Shooting<E, B> {
    fn residual_jet(&self, zs: &[Vec<&[f64]>], nout: usize, par: bool) -> Result<Vec<Vec<P::Scalar>>, String> {
        match nout.max(zs.len()) {
            0..=8 => Shooting::residual_jet::<P, 8>(self, zs, par, nout),
            9..=16 => Shooting::residual_jet::<P, 16>(self, zs, par, nout),
            17..=32 => Shooting::residual_jet::<P, 32>(self, zs, par, nout),
            n => Err(format!(
                "residual_jet: at most 32 orders and coefficient vectors, not {n}"
            )),
        }
    }

    /// A [`crate::jet::JetSession`] on jets of 8, 16 or 32 coefficients, the smallest that holds orders 0..=order,
    /// when the equations form one order from the kept ones (`Equations::ORDER_BY_ORDER`); otherwise multisolver's
    /// `Recompute`, each order by `residual_jet`. Both give `residual_jet`'s bits.
    fn jet_orders<'a>(&'a self, order: usize) -> Result<Box<dyn JetOrders<P> + 'a>, String>
    where
        P: 'a,
    {
        if !E::ORDER_BY_ORDER {
            return Ok(Box::new(Recompute::new(self)));
        }
        match order + 1 {
            0..=8 => Ok(Box::new(self.jet_session::<P, 8>())),
            9..=16 => Ok(Box::new(self.jet_session::<P, 16>())),
            17..=32 => Ok(Box::new(self.jet_session::<P, 32>())),
            n => Err(format!("jet_orders: at most 32 orders, not {n}")),
        }
    }
}

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// The Newton step at z ([`multisolver::newton::newton_step`] on this problem)
    pub fn newton_step<P: Precision<Native>>(&self, z: &[&[f64]], par: bool) -> Result<Step<P>, String> {
        multisolver::newton::newton_step::<P, Self>(self, z, par)
    }

    /// Newton's method from z, damped by `opts.damping` ([`multisolver::newton::newton`] on this problem)
    pub fn newton<P: Precision<Native>>(
        &self,
        z: &mut Vec<Vec<f64>>,
        opts: &NewtonOpts,
    ) -> Result<NewtonReport, String> {
        multisolver::newton::newton::<P, Self>(self, z, opts)
    }
}
