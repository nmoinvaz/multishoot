//! The Levenberg-Marquardt method (multisolver's) on the planted multiple-shooting problem. Its model-problem tests are
//! multisolver's.
mod common;

use common::{perturbed, Planted};
use multishoot::{Double, DD};
use multisolver::lm::{levenberg_marquardt, LmOpts};
use multisolver::prec::nan_max;

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

/// max |z - z*| in double-double
fn planted_err(z: &[Vec<f64>], zs: &[Vec<f64>]) -> f64 {
    (0..z[0].len())
        .map(|i| (DD::new(z[0][i], z[1][i]) - DD::new(zs[0][i], zs[1][i])).hi.abs())
        .fold(0.0, nan_max)
}

/// Pass, written before the first run: on the planted multiple-shooting problem from z* perturbed by 1e-3, LM
/// converges to max |F| <= 1e-28 (with its handover to Newton) and reaches z* to 1e-27. The control: its solution
/// with 1e-20 added to the lo part of the first unknown fails the accuracy check.
#[test]
fn lm_converges_on_the_planted_problem() {
    let (sh, zs) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let mut z = perturbed(&zs, 1e-3);
    let rep = levenberg_marquardt::<Double, _>(&sh, &mut z, &LmOpts::default()).unwrap();
    let err = planted_err(&z, &zs);
    println!(
        "planted: converged {} after {} LM steps and {} factorizations, max |z - z*| {err:.3e}",
        rep.converged,
        rep.mu.len(),
        rep.factorizations
    );
    assert!(rep.converged && err <= 1e-27);
    let mut zw = z.clone();
    zw[1][0] += 1e-20;
    let err = planted_err(&zw, &zs);
    println!("control, first unknown off by 1e-20: max |z - z*| {err:.3e}");
    assert!(err > 1e-27, "control did not fire");
}
