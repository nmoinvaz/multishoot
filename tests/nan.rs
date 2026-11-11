//! Non-finite values must never pass a check: max |F| and the planner's coefficient norms propagate NaN, and Newton
//! never reports a NaN residual as converged.
mod common;

use common::Planted;
use multishoot::lane::DV;
use multishoot::prec::{max_abs, parts};
use multishoot::taylor::{coef_norm, step_size, Plan};
use multishoot::{Double, Fl, NewtonOpts, Shooting, Work, DD};

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

fn planted() -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    Planted::<B>::shooting(&NODES, 30, 1e-32)
}

/// the fold the crate used before, the control: `f64::max` returns the other operand for a NaN
fn old_max_abs(x: &[DD]) -> f64 {
    x.iter().map(|v| v.hi.abs()).fold(0.0, f64::max)
}

/// z* with one unknown NaN, and with every unknown NaN: the residuals
fn nan_residuals() -> (Vec<DD>, Vec<DD>) {
    let (sh, zs) = planted();
    let mut one = zs.clone();
    one[0][sh.off_node(2) + 3] = f64::NAN;
    let all: Vec<Vec<f64>> = zs.iter().map(|p| vec![f64::NAN; p.len()]).collect();
    (
        sh.residual::<Double>(&parts(&one), true).unwrap(),
        sh.residual::<Double>(&parts(&all), true).unwrap(),
    )
}

/// Pass: max |F| is NaN for a residual with a NaN row, and for one that is NaN in every row.
#[test]
fn max_abs_propagates_nan() {
    let (one, all) = nan_residuals();
    assert!(max_abs::<Double>(&one).is_nan() && max_abs::<Double>(&all).is_nan());
}

/// The control of `max_abs_propagates_nan`: the old fold must fail it, scoring both finite and the all-NaN one 0.
#[test]
fn max_abs_control_old_fold_drops_nan() {
    let (one, all) = nan_residuals();
    assert!(old_max_abs(&one).is_finite() && old_max_abs(&all) == 0.0);
}

/// Newton from a state whose residual is NaN in every row. Pass: it returns not converged, with the NaN recorded.
#[test]
fn newton_on_nan_residual_does_not_converge() {
    let (sh, zs) = planted();
    let mut z: Vec<Vec<f64>> = zs.iter().map(|p| vec![f64::NAN; p.len()]).collect();
    let rep = sh.newton::<Double>(&mut z, &NewtonOpts::default()).unwrap();
    println!(
        "Newton on a NaN state: converged {}, max |F| {:?}",
        rep.converged, rep.fnorm
    );
    assert!(!rep.converged && rep.fnorm.iter().all(|v| v.is_nan()));
}

/// The control of `newton_on_nan_residual_does_not_converge`: under the old fold that residual scores 0, at or below
/// any tolerance, which is how Newton reported it converged.
#[test]
fn newton_nan_control_old_fold_scores_zero() {
    let (_, all) = nan_residuals();
    assert!(old_max_abs(&all) <= NewtonOpts::default().tol);
}

/// a Taylor work array of order 4 with every coefficient 0.5 but one value of order 4, NaN when `nan`
fn work(nan: bool) -> Work<DV, 1> {
    let mut w = Work::<DV, 1>::new(1, 0, 4);
    for k in 0..=4 {
        w.c[0][k] = Fl::splat(DD::from_f64(0.5));
    }
    if nan {
        let v: Vec<f64> = (0..8).map(|j| if j == 6 { f64::NAN } else { 0.5 }).collect();
        w.c[0][4] = Fl::load(&v, &[0.0; 8]);
    }
    w
}

/// Pass: the planner's coefficient norm and step are NaN when a coefficient is, and finite otherwise.
#[test]
fn planner_norm_propagates_nan() {
    let plan = Plan {
        kt: 4,
        tol: 1e-20,
        hmax: 1.0,
    };
    assert!(coef_norm(&work(true), 4).is_nan() && step_size(&work(true), &plan).is_nan());
    assert!(coef_norm(&work(false), 4).is_finite() && step_size(&work(false), &plan).is_finite());
}

/// The control of `planner_norm_propagates_nan`: `Fl::amax` alone, the old norm, must fail it by skipping the NaN.
#[test]
fn planner_norm_control_amax_skips_nan() {
    let w = work(true);
    assert!(w.c.iter().map(|cf| cf[4].amax()).fold(0.0, f64::max).is_finite());
}
