//! Taylor stepping and the planner on the linear model `Rotor`, whose solution is known in closed form.
mod common;

use common::{max_diff, Rotor, Wrong};
use multishoot::lane::DV;
use multishoot::taylor::{taylor_step, Plan};
use multishoot::{Equations, Shooting, State, Work};

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

/// z with the exact node states of the rotor (the endpoint data and globals are not read here)
fn exact_z(sh: &Shooting<Rotor<B>, B>) -> (Vec<f64>, Vec<f64>) {
    let m = 8 * B;
    let mut zh = vec![0.0; sh.len()];
    let mut zl = vec![0.0; sh.len()];
    for (i, &x) in sh.nodes.iter().enumerate() {
        let o = sh.off_node(i);
        for (f, fl) in sh.eq.exact(x).iter().enumerate() {
            fl.store(&mut zh[o + f * m..o + (f + 1) * m], &mut zl[o + f * m..o + (f + 1) * m]);
        }
    }
    (zh, zl)
}

/// The state carried through every segment in the direction of the layout, each from the state the previous
/// segment delivered, starting from the exact state at the first segment's source; the largest error at the nodes.
fn carry_error(sh: &Shooting<Rotor<B>, B>) -> f64 {
    let k = sh.k();
    let order: Vec<usize> = if sh.match_node == 0 {
        (1..k).rev().collect()
    } else {
        (1..k).collect()
    };
    let mut s: State<DV, B> = sh.eq.exact(sh.nodes[sh.seg_source(order[0])]);
    let mut err = 0.0f64;
    for &b in &order {
        s = sh.block_state::<DV>(b, &s, &[]);
        let dst = if sh.seg_inward(b) { b - 1 } else { b };
        err = err.max(max_diff(&s, &sh.eq.exact(sh.nodes[dst])));
    }
    err
}

fn rotor_shooting(wrong: Wrong, match_node: usize) -> Shooting<Rotor<B>, B> {
    let mut eq = Rotor::<B>::new();
    eq.wrong = wrong;
    let mut sh = Shooting::new(eq, NODES.to_vec(), 30);
    sh.match_node = match_node;
    let (zh, zl) = exact_z(&sh);
    // the plan is made on the correct equations, so that a control changes the recursion only
    let wrong_saved = sh.eq.wrong;
    sh.eq.wrong = Wrong::None;
    sh.plan_segments(
        &zh,
        &zl,
        &Plan {
            kt: 30,
            tol: 1e-32,
            hmax: 0.5,
        },
    );
    sh.eq.wrong = wrong_saved;
    sh
}

/// The layout's sizes and offsets for NL = 1, NF = 2, NR = 1 at M = 16 and five nodes, with segments 3 and 4 inward.
/// Pass: L = rows = 192, nodes at 16 + 32 i, the right data at 176, the inward blocks reading node b and matched to
/// node b - 1.
#[test]
fn layout_offsets() {
    let mut sh = Shooting::new(Rotor::<B>::new(), NODES.to_vec(), 20);
    sh.match_node = 2;
    assert_eq!((sh.len(), sh.rows(), sh.is_square()), (192, 192, true));
    assert_eq!(
        (0..5).map(|i| sh.off_node(i)).collect::<Vec<_>>(),
        vec![16, 48, 80, 112, 144]
    );
    assert_eq!((sh.off_right(), sh.off_global()), (176, 192));
    assert_eq!(sh.block_inputs(0), vec![0]);
    assert_eq!(sh.block_inputs(2), vec![48, 64]);
    assert_eq!(sh.block_inputs(3), vec![112, 128]);
    assert_eq!(sh.block_inputs(5), vec![176]);
    let targets: Vec<usize> = (0..6).map(|b| sh.block_target(b)).collect();
    assert_eq!(targets, vec![16, 48, 80, 80, 112, 144]);
    assert_eq!(
        (sh.seg_inward(2), sh.seg_inward(3), sh.seg_inward(5)),
        (false, true, false)
    );
}

/// The rotor carried from x = 0 to 2 outward, and from 2 to 0 inward, through planned segments (order 30, last terms
/// below 1e-32, steps at most 0.5), each segment from the state the previous one delivered. Pass: every node state
/// within 1e-28 of the exact one (double-double rounding is 2^-106 = 1.2e-32 per operation, accumulated over the
/// steps and the d/dtheta products).
#[test]
fn rotor_carried_to_double_double() {
    for match_node in [4, 0] {
        let sh = rotor_shooting(Wrong::None, match_node);
        let steps: usize = sh.segs.iter().map(|s| s.len() - 1).sum();
        let err = carry_error(&sh);
        println!("match_node {match_node}: {steps} steps, max error {err:.3e}");
        assert!(err < 1e-28, "match_node {match_node}: error {err:e}");
    }
}

/// The control of `rotor_carried_to_double_double`: one coefficient of the recursion (order 10) wrong by a relative
/// 2^-60 must fail it (error above 1e-28).
#[test]
fn rotor_carried_control_wrong_coefficient_fails() {
    for match_node in [4, 0] {
        let sh = rotor_shooting(Wrong::Order9, match_node);
        let err = carry_error(&sh);
        println!("control, match_node {match_node}: max error {err:.3e}");
        assert!(err > 1e-28, "control did not fire: error {err:e}");
    }
}

/// One step of each planned segment from the exact state at its start, against the exact state at its end: the
/// largest error, relative to the tolerance. `stretch` scales the planned steps.
fn plan_ratios(plan: &Plan, stretch: f64) -> (f64, usize) {
    let eq = Rotor::<B>::new();
    let mut sh = Shooting::new(Rotor::<B>::new(), NODES.to_vec(), plan.kt);
    let (zh, zl) = exact_z(&sh);
    sh.plan_segments(&zh, &zl, plan);
    let mut w = Work::<DV, B>::new(Rotor::<B>::NF, Rotor::<B>::NAUX, plan.kt);
    let mut worst = 0.0f64;
    let mut n = 0;
    for seg in &sh.segs {
        for p in seg.windows(2) {
            let x1 = p[0] + stretch * (p[1] - p[0]);
            let s = taylor_step(&eq, &eq.exact(p[0]), p[0], x1, &[], &mut w);
            worst = worst.max(max_diff(&s, &eq.exact(x1)) / plan.tol);
            n += 1;
        }
    }
    (worst, n)
}

/// The planner's bound against the true error: order 20, tolerance 1e-20 (far above double-double rounding, so the
/// error measured is the truncation), steps at most 1. Pass: every planned step's error from the exact start is at
/// most tol (the bound holds), and the largest is at least 1e-3 tol (the bound is not loose by more than that).
#[test]
fn planner_bound_holds() {
    let plan = Plan {
        kt: 20,
        tol: 1e-20,
        hmax: 1.0,
    };
    let (worst, n) = plan_ratios(&plan, 1.0);
    println!("{n} steps, largest error / tol {worst:.3e}");
    assert!(worst <= 1.0, "a planned step's error is {worst:e} tol");
    assert!(worst >= 1e-3, "the largest error is only {worst:e} tol");
}

/// The control of `planner_bound_holds`: the planned steps doubled must break the bound (some error above tol).
#[test]
fn planner_bound_control_doubled_steps_fails() {
    let plan = Plan {
        kt: 20,
        tol: 1e-20,
        hmax: 1.0,
    };
    let (worst, _) = plan_ratios(&plan, 2.0);
    println!("doubled steps: largest error / tol {worst:.3e}");
    assert!(worst > 1.0, "control did not fire: {worst:e} tol");
}
