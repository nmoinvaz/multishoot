//! Taylor stepping and the planner on the linear model `Rotor`, whose solution is known in closed form.
mod common;

use multishoot::prec::nan_max;

use common::{max_diff, Fnv, Rotor, Wrong};
use multishoot::lane::DV;
use multishoot::taylor::{plan_path, taylor_step, Plan};
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
/// segment delivered, starting from the exact state at the first segment's source with `tweak` applied to it: the
/// states delivered at the nodes, with the node of each.
fn carry(sh: &Shooting<Rotor<B>, B>, tweak: impl Fn(&mut State<DV, B>)) -> Vec<(usize, State<DV, B>)> {
    let k = sh.k();
    let order: Vec<usize> = if sh.match_node == 0 {
        (1..k).rev().collect()
    } else {
        (1..k).collect()
    };
    let mut s: State<DV, B> = sh.eq.exact(sh.nodes[sh.seg_source(order[0])]);
    tweak(&mut s);
    let mut out = Vec::new();
    for &b in &order {
        s = sh.block_state::<DV>(b, &s, &[]);
        out.push((if sh.seg_inward(b) { b - 1 } else { b }, s.clone()));
    }
    out
}

/// The largest error of the carried states at the nodes
fn carry_error(sh: &Shooting<Rotor<B>, B>) -> f64 {
    carry(sh, |_| {})
        .iter()
        .map(|(n, s)| max_diff(s, &sh.eq.exact(sh.nodes[*n])))
        .fold(0.0, nan_max)
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

/// The fingerprint of the Taylor stepping: the planned step points and the carried node states, outward and inward,
/// every part's bits, with `tweak` applied to each carry's starting state
fn stepping_hash(tweak: impl Fn(&mut State<DV, B>) + Copy) -> u64 {
    let mut h = Fnv::new();
    for match_node in [4, 0] {
        let sh = rotor_shooting(Wrong::None, match_node);
        sh.segs.iter().flatten().for_each(|&x| h.f64(x));
        for (_, s) in carry(&sh, tweak) {
            s.iter().for_each(|f| h.field(f));
        }
    }
    h.0
}

/// the reference fingerprint, recorded on aarch64 (NEON) and on the portable backend
const STEPPING_REFERENCE: u64 = 0xc8d40d2c1841b8cd;

/// The Taylor stepping's results do not depend on multidouble's SIMD backend. Pass: `stepping_hash` equals the
/// reference on every build of the CI matrix (SSE2, AVX2, NEON, portable, MSVC).
#[test]
fn stepping_bytes_match_reference() {
    let h = stepping_hash(|_| {});
    println!("stepping hash {h:#018x}");
    assert_eq!(
        h, STEPPING_REFERENCE,
        "the Taylor stepping's bytes differ from the reference"
    );
}

/// The control of `stepping_bytes_match_reference`: one ulp of one value of the starting state must change the
/// fingerprint.
#[test]
fn stepping_bytes_control_one_ulp_differs() {
    let h = stepping_hash(|s| {
        let mut v: Vec<f64> = (0..8 * B).map(|j| s[1].get(j).hi).collect();
        let l: Vec<f64> = (0..8 * B).map(|j| s[1].get(j).lo).collect();
        v[5] = f64::from_bits(v[5].to_bits() + 1);
        s[1] = multishoot::Fl::load(&v, &l);
    });
    assert_ne!(h, STEPPING_REFERENCE, "control did not fire");
    assert_ne!(h, stepping_hash(|_| {}), "control did not fire");
}

/// Every planned segment ends exactly at its target node. Pass: the last step point equals the node in f64.
#[test]
fn planned_segments_end_at_their_nodes() {
    for match_node in [4, 0] {
        let sh = rotor_shooting(Wrong::None, match_node);
        for b in 1..sh.k() {
            let dst = if sh.seg_inward(b) { b - 1 } else { b };
            assert_eq!(sh.segs[b - 1].first(), Some(&sh.nodes[sh.seg_source(b)]));
            assert_eq!(sh.segs[b - 1].last(), Some(&sh.nodes[dst]));
        }
    }
}

/// A plan with a zero tolerance would make no progress: the planner must refuse it rather than loop.
#[test]
#[should_panic(expected = "must be positive")]
fn planner_rejects_zero_tolerance() {
    let mut sh = Shooting::new(Rotor::<B>::new(), NODES.to_vec(), 20);
    let (zh, zl) = exact_z(&sh);
    sh.plan_segments(
        &zh,
        &zl,
        &Plan {
            kt: 20,
            tol: 0.0,
            hmax: 1.0,
        },
    );
}

/// The rotor's plan from x = 0 to 0.5 with the recursion's `wrong`
fn plan_rotor(wrong: Wrong) -> Vec<f64> {
    let mut eq = Rotor::<B>::new();
    eq.wrong = wrong;
    let s0 = eq.exact(0.0);
    plan_path(
        &eq,
        &s0,
        0.0,
        0.5,
        &[],
        &Plan {
            kt: 20,
            tol: 1e-20,
            hmax: 1.0,
        },
    )
}

/// A recursion that yields a NaN at one collocation point must stop the planner (`Fl::amax` alone would skip it).
#[test]
#[should_panic(expected = "is not finite")]
fn planner_rejects_nan_coefficient() {
    plan_rotor(Wrong::Nan);
}

/// The control of `planner_rejects_nan_coefficient`: the same plan with finite coefficients must not panic and must
/// reach its end.
#[test]
fn planner_nan_control_finite_plans() {
    assert_eq!(plan_rotor(Wrong::None).last(), Some(&0.5));
}

/// An infinite end must stop the planner before it starts.
#[test]
#[should_panic(expected = "must be finite")]
fn planner_rejects_infinite_end() {
    let eq = Rotor::<B>::new();
    plan_path(
        &eq,
        &eq.exact(0.0),
        0.0,
        f64::INFINITY,
        &[],
        &Plan {
            kt: 20,
            tol: 1e-20,
            hmax: 1.0,
        },
    );
}

/// whether a plan is exactly the one step [xa, xb]
fn is_one_step(xs: &[f64], xa: f64, xb: f64) -> bool {
    xs == [xa, xb]
}

/// A path whose end is one representable f64 from its start, outward and inward, within the old exit threshold of
/// 1e-15. Pass: the plan is exactly [xa, xb].
#[test]
fn planner_reaches_an_end_one_ulp_away() {
    let eq = Rotor::<B>::new();
    let plan = Plan {
        kt: 20,
        tol: 1e-20,
        hmax: 1.0,
    };
    for (xa, xb) in [(0.5, 0.5f64.next_up()), (0.5, 0.5f64.next_down())] {
        let xs = plan_path(&eq, &eq.exact(xa), xa, xb, &[], &plan);
        assert!(is_one_step(&xs, xa, xb), "{xa} -> {xb}: {xs:?}");
    }
}

/// The control of `planner_reaches_an_end_one_ulp_away`: the same check fed the plan stopped one step early, [xa], as
/// the exit at 1e-15 left it, must fail.
#[test]
fn planner_one_ulp_control_early_stop_fails() {
    let eq = Rotor::<B>::new();
    let (xa, xb) = (0.5, 0.5f64.next_up());
    let xs = plan_path(
        &eq,
        &eq.exact(xa),
        xa,
        xb,
        &[],
        &Plan {
            kt: 20,
            tol: 1e-20,
            hmax: 1.0,
        },
    );
    assert!(!is_one_step(&xs[..xs.len() - 1], xa, xb), "control did not fire");
}

/// The fingerprint of the planner alone: the step points of the rotor's segments, outward and inward, at orders 30
/// and 20, with the tolerance scaled by `tol_scale`
fn plan_hash(tol_scale: f64) -> u64 {
    let mut h = Fnv::new();
    for (kt, tol) in [(30, 1e-32), (20, 1e-20)] {
        for match_node in [4, 0] {
            let mut sh = Shooting::new(Rotor::<B>::new(), NODES.to_vec(), kt);
            sh.match_node = match_node;
            let (zh, zl) = exact_z(&sh);
            sh.plan_segments(
                &zh,
                &zl,
                &Plan {
                    kt,
                    tol: tol * tol_scale,
                    hmax: 0.5,
                },
            );
            sh.segs.iter().flatten().for_each(|&x| h.f64(x));
        }
    }
    h.0
}

/// the reference fingerprint of the planned points, recorded on aarch64 (NEON) and on the portable backend
const PLAN_REFERENCE: u64 = 0x59775fc9dcbb802a;

/// The planner's step points are the same bits on every platform (its roots use no libm): pass, `plan_hash` equals
/// the reference on Linux, macOS and Windows, every backend.
#[test]
fn plan_points_match_reference() {
    let h = plan_hash(1.0);
    println!("plan hash {h:#018x}");
    assert_eq!(h, PLAN_REFERENCE, "the planned points differ from the reference");
}

/// The control of `plan_points_match_reference`: the tolerance scaled by 1 + 2^-30 must change the fingerprint.
#[test]
fn plan_points_control_tolerance_differs() {
    assert_ne!(plan_hash(1.0 + 2f64.powi(-30)), plan_hash(1.0), "control did not fire");
}

/// a plan of the rotor at order 2 over one tiny segment, x = 0 to 1e-12, at the tolerance tol
fn plan_tiny(tol: f64) -> Vec<f64> {
    let eq = Rotor::<B>::new();
    plan_path(&eq, &eq.exact(0.0), 0.0, 1e-12, &[], &Plan { kt: 2, tol, hmax: 1.0 })
}

/// A tolerance so small that tol / e underflows to zero (the smallest subnormal over coefficients above 2) must stop
/// the planner with its own message about the step, not inside the root.
#[test]
#[should_panic(expected = "plan_path: step 0e0")]
fn planner_reports_an_underflowing_tolerance() {
    plan_tiny(f64::from_bits(1));
}

/// The control of `planner_reports_an_underflowing_tolerance`: a tolerance whose quotient does not underflow plans
/// the segment without panicking, in one step (at order 2 the first root is of degree 1, h = tol / e_1, about 2e-11
/// at tol = 1e-10 and above the segment's 1e-12).
#[test]
fn planner_underflow_control_normal_tolerance_plans() {
    assert_eq!(plan_tiny(1e-10), vec![0.0, 1e-12]);
}
