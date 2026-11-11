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
//! Segment b runs outward, node b-1 to node b, for b <= `match_node`, and inward, node b to node b-1, after it. The
//! system is square when NL + NR = NF and there is one extra row per global.
use crate::equations::{Equations, State, Work};
use crate::lane::{Lane, DV};
use crate::taylor::{plan_path, taylor_path, Plan};
use multidouble::dd::DD;
use multidouble::field::Fl;

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

    /// The state block b produces from its inputs (fields in the order of `block_inputs`), the globals and the
    /// evaluation's preparation (`Equations::prepare`): the left endpoint's state at node 0, segment b's Taylor path,
    /// or the right endpoint's state at node K-1.
    pub fn block_state<T: Lane>(&self, b: usize, inp: &[Fl<T, B>], g: &[T], prep: &E::Prep) -> State<T, B> {
        let k = self.k();
        if b == 0 {
            self.eq.left_state(self.nodes[0], inp, g, prep)
        } else if b < k {
            let mut w = Work::<T, B>::new(E::NF, E::NAUX, self.kt);
            taylor_path(&self.eq, inp, &self.segs[b - 1], g, &mut w)
        } else {
            self.eq.right_state(self.nodes[k - 1], inp, g, prep)
        }
    }

    /// Plans every segment's steps (`plan_path`) from its source node's state in z (double-double, hi and lo parts)
    /// toward its target node, with the globals of z, at the order `plan.kt`, which becomes the problem's.
    pub fn plan_segments(&mut self, zh: &[f64], zl: &[f64], plan: &Plan) {
        assert!(
            zh.len() == self.len() && zl.len() == self.len(),
            "plan_segments: z of length {}",
            self.len()
        );
        let m = Self::M;
        let og = self.off_global();
        let g: Vec<DV> = (0..E::NG).map(|i| DV::splat(DD::new(zh[og + i], zl[og + i]))).collect();
        let segs = (1..self.k())
            .map(|b| {
                let src = self.seg_source(b);
                let dst = if self.seg_inward(b) { b - 1 } else { b };
                let o = self.off_node(src);
                let s0: Vec<Fl<DV, B>> = (0..E::NF)
                    .map(|f| Fl::load(&zh[o + f * m..o + (f + 1) * m], &zl[o + f * m..o + (f + 1) * m]))
                    .collect();
                plan_path(&self.eq, &s0, self.nodes[src], self.nodes[dst], &g, plan)
            })
            .collect();
        self.segs = segs;
        self.kt = plan.kt;
    }
}
