//! Taylor stepping in x: one step from the coefficients of [`Equations::taylor_coeffs`] summed by Horner's rule, a
//! path of steps through given points, and the planner that chooses those points so that each step's last terms stay
//! below a tolerance.
//!
//! A step from x0 to x1 sums `sum_k c_k h^k`, k = 0..=kt, with h = x1 - x0 exact in double-double (both points are
//! f64). The planner bounds the truncation error by the last two terms: with `e_k = max |c_k|` over the fields and
//! collocation points, the step is the largest h with `e_(kt-1) h^(kt-1) <= tol` and `e_kt h^kt <= tol`, capped at
//! `hmax`. For a series whose terms fall geometrically past order kt-1, the dropped tail is then below tol times the
//! ratio of consecutive terms.
use crate::equations::{Equations, State, Work};
use crate::lane::{Lane, DV};
use multidouble::dd::DD;
use multidouble::field::{horner, Fl};

/// The planner's settings: Taylor order, tolerance on the last terms (absolute), largest step.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub kt: usize,
    pub tol: f64,
    pub hmax: f64,
}

/// One Taylor step of the state s from x0 to x1 (w sized for E's fields and w.kt).
pub fn taylor_step<E: Equations<B>, T: Lane, const B: usize>(
    eq: &E,
    s: &[Fl<T, B>],
    x0: f64,
    x1: f64,
    g: &[T],
    w: &mut Work<T, B>,
) -> State<T, B> {
    for (f, sf) in s.iter().enumerate().take(E::NF) {
        w.c[f][0] = *sf;
    }
    eq.taylor_coeffs(x0, g, w);
    let h = DD::from_f64(x1) - DD::from_f64(x0);
    let kt = w.kt;
    w.c.iter().map(|cf| horner(&cf[..=kt], h)).collect()
}

/// Taylor steps through the points xs, the first the start, from the state s0.
pub fn taylor_path<E: Equations<B>, T: Lane, const B: usize>(
    eq: &E,
    s0: &[Fl<T, B>],
    xs: &[f64],
    g: &[T],
    w: &mut Work<T, B>,
) -> State<T, B> {
    let mut s = s0.to_vec();
    for i in 0..xs.len().saturating_sub(1) {
        s = taylor_step(eq, &s, xs[i], xs[i + 1], g, w);
    }
    s
}

/// max over the fields and collocation points of |coefficient k| (hi parts)
pub fn coef_norm<const B: usize>(w: &Work<DV, B>, k: usize) -> f64 {
    w.c.iter().map(|cf| cf[k].amax()).fold(0.0, f64::max)
}

/// The step the planner takes from coefficients w (orders kt - 1 and kt), before the cap at the end of the segment.
pub fn step_size<const B: usize>(w: &Work<DV, B>, plan: &Plan) -> f64 {
    let kt = w.kt;
    let e1 = coef_norm(w, kt - 1).max(1e-300);
    let e2 = coef_norm(w, kt).max(1e-300);
    let h1 = (plan.tol / e1).powf(1.0 / (kt - 1) as f64);
    let h2 = (plan.tol / e2).powf(1.0 / kt as f64);
    h1.min(h2).min(plan.hmax)
}

/// Step points from xa to xb (either direction) for the state s0, both ends included and the last exactly xb: each
/// step from [`step_size`] at the state reached so far; a step that would leave less than 40% of the rest is halved,
/// so the segment ends in two near-equal steps instead of a sliver. A plan with tol or hmax not positive, or a step
/// that is not positive or does not move x (coefficients that are not finite, or a singularity at x), panics.
pub fn plan_path<E: Equations<B>, const B: usize>(
    eq: &E,
    s0: &[Fl<DV, B>],
    xa: f64,
    xb: f64,
    g: &[DV],
    plan: &Plan,
) -> Vec<f64> {
    assert!(plan.kt >= 2, "plan_path: Taylor order {} < 2", plan.kt);
    assert!(
        plan.tol > 0.0 && plan.hmax > 0.0,
        "plan_path: tol {:e} and hmax {:e} must be positive",
        plan.tol,
        plan.hmax
    );
    let mut w = Work::<DV, B>::new(E::NF, E::NAUX, plan.kt);
    let mut xs = vec![xa];
    let mut x = xa;
    let mut s = s0.to_vec();
    let dir = if xb > xa { 1.0 } else { -1.0 };
    while (xb - x) * dir > 0.0 {
        for (f, sf) in s.iter().enumerate() {
            w.c[f][0] = *sf;
        }
        eq.taylor_coeffs(x, g, &mut w);
        let mut h = step_size(&w, plan);
        assert!(h > 0.0, "plan_path: step {h:e} at x = {x}");
        let rest = (xb - x) * dir;
        if h >= rest {
            h = rest;
        } else if h > 0.6 * rest {
            h = 0.5 * rest;
        }
        let xn = if h == rest { xb } else { x + dir * h };
        assert!(xn != x, "plan_path: step {h:e} does not move x = {x}");
        let hd = DD::from_f64(xn) - DD::from_f64(x);
        s = w.c.iter().map(|cf| horner(&cf[..=plan.kt], hd)).collect();
        x = xn;
        xs.push(x);
    }
    xs
}
