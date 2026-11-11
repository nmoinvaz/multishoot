//! A singular endpoint: Bessel's equation for the orders 0..7, one per lane, started at x = 0 from its Frobenius
//! series, against the closed form of its regular solution.
mod common;

use common::{perturbed, Bessel, PerColumn};
use multishoot::prec::{nan_max, parts, widen};
use multishoot::{Double, Native, NewtonOpts, Precision, Quad, Shooting, DD};
use std::sync::atomic::Ordering;

const KT: usize = 30;

fn opts(tol: f64, max_iter: usize) -> NewtonOpts {
    NewtonOpts {
        tol,
        max_iter,
        ..Default::default()
    }
}

fn bessel(wrong: bool) -> (Shooting<Bessel, 1>, Vec<Vec<f64>>) {
    let mut eq = Bessel::new(0.5, 2.0);
    eq.wrong = wrong;
    Bessel::shooting(eq, KT)
}

/// Newton in double-double from the solution perturbed by 1e-3, then in quad-double: (dd converged, qd max |F|, max
/// |z - z*| in quad-double rounded, |a - 1| the largest over the orders)
fn solve(sh: &Shooting<Bessel, 1>, zs: &[Vec<f64>]) -> (bool, f64, f64, f64) {
    let mut z = perturbed(zs, 1e-3);
    let dd = sh.newton::<Double>(&mut z, &opts(1e-28, 10)).unwrap();
    let mut zq = widen(&z, 4);
    let qd = sh.newton::<Quad>(&mut zq, &opts(1e-60, 4)).unwrap();
    let q = |i: usize| <Quad as Precision<Native>>::new(&[zq[0][i], zq[1][i], zq[2][i], zq[3][i]]).to_dd();
    let zerr = (0..sh.len())
        .map(|i| (q(i) - DD::new(zs[0][i], zs[1][i])).hi.abs())
        .fold(0.0, nan_max);
    let aerr = (0..8).map(|i| (q(i) - DD::ONE).hi.abs()).fold(0.0, nan_max);
    let fq = *qd.fnorm.last().unwrap();
    println!(
        "dd max |F| {:?}, qd {:?}; max |a - 1| {aerr:.3e}, max |z - z*| {zerr:.3e}",
        dd.fnorm.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>(),
        qd.fnorm.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>()
    );
    (dd.converged, fq, zerr, aerr)
}

/// dd converged, qd max |F| below 1e-60, and every unknown, the amplitudes a included, within 1e-27 of the closed
/// form (its double-double coefficients, the Frobenius sum and the steps' truncation each about 1e-32)
fn passes(r: (bool, f64, f64, f64)) -> bool {
    r.0 && r.1 < 1e-60 && r.2 < 1e-27 && r.3 < 1e-27
}

/// The eight orders' boundary-value problem, started at x = 0.5 from the Frobenius series and matched to y*(2) at
/// x = 2, from the solution perturbed by 1e-3: Newton in double-double, then quad-double. Pass: `passes`.
#[test]
fn bessel_converges_to_its_closed_form() {
    let (sh, zs) = bessel(false);
    let r = solve(&sh, &zs);
    assert!(passes(r), "{r:?}");
}

/// The control of `bessel_converges_to_its_closed_form`: the Frobenius coefficient c_3 off by 2^-60 must fail it.
#[test]
fn bessel_control_wrong_frobenius_coefficient_fails() {
    let (sh, zs) = bessel(true);
    assert!(!passes(solve(&sh, &zs)), "control did not fire");
}

/// The preparation runs once per evaluation, not per block or column: pass, the counter rises by one for a
/// residual, a Jacobian and a jet each, and by two for a Newton step (its residual and its Jacobian).
#[test]
fn prepare_runs_once_per_evaluation() {
    let (sh, zs) = bessel(false);
    let z = parts(&zs);
    let n = || sh.eq.prepared.load(Ordering::Relaxed);
    let n0 = n();
    sh.residual::<Double>(&z, true).unwrap();
    let n1 = n();
    sh.jacobian::<Double>(&z, true).unwrap();
    let n2 = n();
    sh.residual_jet::<Double, 2>(std::slice::from_ref(&z), true, 2).unwrap();
    let n3 = n();
    sh.newton_step::<Double>(&z, true).unwrap();
    let n4 = n();
    println!("prepare calls: {} {} {} {}", n1 - n0, n2 - n1, n3 - n2, n4 - n3);
    assert_eq!((n1 - n0, n2 - n1, n3 - n2, n4 - n3), (1, 1, 1, 2));
}

/// The control of `prepare_runs_once_per_evaluation`: two residuals must count two.
#[test]
fn prepare_control_counts_each_evaluation() {
    let (sh, zs) = bessel(false);
    let n0 = sh.eq.prepared.load(Ordering::Relaxed);
    sh.residual::<Double>(&parts(&zs), true).unwrap();
    sh.residual::<Double>(&parts(&zs), true).unwrap();
    assert_eq!(sh.eq.prepared.load(Ordering::Relaxed) - n0, 2);
}

/// The number of entries (in any part) in which the pointwise Jacobian, one dual evaluation per input field,
/// differs from the column-by-column one of the problem with the Frobenius coefficient c_3 wrong when `wrong_per_column`,
/// at the perturbed solution; and the two evaluation counts
fn jacobian_differences(wrong_per_column: bool) -> (usize, usize, usize) {
    let (sh, zs) = bessel(false);
    let z = perturbed(&zs, 1e-3);
    let (c1, _) = sh.jacobian::<Double>(&parts(&z), true).unwrap();
    let mut eq = Bessel::new(0.5, 2.0);
    eq.wrong = wrong_per_column;
    let mut pc = Shooting::new(PerColumn(eq), sh.nodes.clone(), sh.kt);
    pc.segs = sh.segs.clone();
    let (c2, _) = pc.jacobian::<Double>(&parts(&z), true).unwrap();
    let differ = c1
        .iter()
        .zip(&c2)
        .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
        .sum::<usize>();
    (differ, sh.jac_tasks().len(), pc.jac_tasks().len())
}

/// The pointwise Jacobian against the same problem column by column. Pass: every entry equal in every part (as f64
/// values).
#[test]
fn bessel_pointwise_jacobian_equals_per_column() {
    let (differ, n1, n2) = jacobian_differences(false);
    println!("Bessel Jacobian: {n1} evaluations against {n2}, {differ} entries differ");
    assert_eq!(differ, 0);
}

/// The control of `bessel_pointwise_jacobian_equals_per_column`: with c_3 wrong on the column-by-column side only,
/// the same comparison must find entries that differ.
#[test]
fn bessel_pointwise_jacobian_control_one_side_wrong_differs() {
    let (differ, ..) = jacobian_differences(true);
    println!("control, c_3 wrong on one side: {differ} entries differ");
    assert!(differ > 0, "control did not fire");
}
