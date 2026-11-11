//! Newton's damping on the planted problem: the error-oriented (affine-covariant) rule against the max |F| rule, on
//! a copy whose rows of block 3 are scaled by 2^100 (about 1.27e30; a power of two, so the scaling is exact).
mod common;

use common::{wave, Planted};
use multishoot::lane::DV;
use multishoot::newton::Damping;
use multishoot::prec::{max_abs, nan_max, parts};
use multishoot::{Double, NewtonOpts, NewtonReport, Shooting, DD};

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];
const SCALE: f64 = 1.2676506002282294e30; // 2^100

/// The planted problem, with block 3's rows (segment 3, node 2 to node 3) scaled by 2^100 when `scaled`
fn planted(scaled: bool) -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    let (mut sh, zs) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    if scaled {
        let (r0, n) = (sh.row_block(3), sh.row_block(4) - sh.row_block(3));
        sh.row_scale = (0..sh.rows())
            .map(|r| if r >= r0 && r < r0 + n { SCALE } else { 1.0 })
            .collect();
    }
    (sh, zs)
}

/// z* perturbed by 1e-2 (relative, from `wave`) on every unknown, then node 3 set to segment 3's Taylor path from the
/// perturbed node 2: block 3 starts matched to rounding, while the first Newton step moves its input by about 1e-2,
/// so its quadratic remainder, times 2^100 in the scaled copy, dominates max |F| after that step
fn start(sh: &Shooting<Planted<B>, B>, zs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = 8 * B;
    let mut z = zs.to_vec();
    for (i, v) in z[0].iter_mut().enumerate() {
        *v += 1e-2 * wave(i) * v.abs().max(0.1);
    }
    z[1].iter_mut().for_each(|v| *v = 0.0);
    let og = sh.off_global();
    let g = [DV::splat(DD::new(z[0][og], z[1][og]))];
    let o2 = sh.off_node(2);
    let inp: Vec<_> = (0..2).map(|f| sh.load::<Double>(&parts(&z), o2 + f * m)).collect();
    let s3 = sh.block_state::<DV>(3, &inp, &g, &());
    let o3 = sh.off_node(3);
    let (hi, lo) = z.split_at_mut(1);
    for (f, fl) in s3.iter().enumerate() {
        fl.store(
            &mut hi[0][o3 + f * m..o3 + (f + 1) * m],
            &mut lo[0][o3 + f * m..o3 + (f + 1) * m],
        );
    }
    z
}

/// Newton with the damping from `start`: (report, max |z - z*|, max |F| of the unscaled problem at the result)
fn run(scaled: bool, damping: Damping) -> (NewtonReport, f64, f64, Vec<Vec<f64>>) {
    let (sh, zs) = planted(scaled);
    let mut z = start(&sh, &zs);
    let opts = NewtonOpts {
        tol: 0.0,
        tol_dz: 1e-26,
        max_iter: 12,
        damping,
        ..Default::default()
    };
    let rep = sh.newton::<Double>(&mut z, &opts).unwrap();
    let err = (0..sh.len())
        .map(|i| (DD::new(z[0][i], z[1][i]) - DD::new(zs[0][i], zs[1][i])).hi.abs())
        .fold(0.0, nan_max);
    let (plain, _) = planted(false);
    let f = max_abs::<Double>(&plain.residual::<Double>(&parts(&z), true).unwrap());
    println!(
        "{damping:?}, scaled {scaled}: lambda {:?}, converged {}, max |z - z*| {err:.3e}, unscaled max |F| {f:.3e}",
        rep.scale.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(),
        rep.converged
    );
    (rep, err, f, z)
}

/// From `start` on the scaled copy. Pass: the max |F| rule refuses the first full step (its first lambda below 1),
/// and the correction rule converges (its simplified correction below 1e-26 after a full step) within 12 steps to
/// within 1e-27 of z*, with the unscaled residual there below 1e-28.
#[test]
fn correction_rule_converges_where_residual_rule_refuses() {
    let (res, ..) = run(true, Damping::Residual);
    let (cor, err, f, _) = run(true, Damping::Correction);
    assert!(
        res.scale[0] < 1.0,
        "the max |F| rule took the first full step: {:?}",
        res.scale
    );
    assert!(
        cor.converged && err < 1e-27 && f < 1e-28,
        "correction rule: {err:e} {f:e} {:?}",
        cor.scale
    );
}

/// whether two runs took the same steps: the same lambdas, thetas within 1e-10 relative, and final iterates within
/// 1e-27
fn same_steps(a: &(NewtonReport, f64, f64, Vec<Vec<f64>>), b: &(NewtonReport, f64, f64, Vec<Vec<f64>>)) -> bool {
    let th = a.0.theta.len() == b.0.theta.len()
        && a.0
            .theta
            .iter()
            .zip(&b.0.theta)
            .all(|(x, y)| (x - y).abs() <= 1e-10 * x.abs().max(1e-300));
    let dz = (0..a.3[0].len())
        .map(|i| (DD::new(a.3[0][i], a.3[1][i]) - DD::new(b.3[0][i], b.3[1][i])).hi.abs())
        .fold(0.0, nan_max);
    a.0.scale == b.0.scale && th && dz < 1e-27
}

/// Affine covariance: the correction rule on the copy with block 3 scaled by 2^100 and on the plain problem, from
/// the same start. Pass: `same_steps`.
#[test]
fn correction_rule_is_affine_covariant() {
    let (a, b) = (run(false, Damping::Correction), run(true, Damping::Correction));
    assert!(same_steps(&a, &b), "the scaled copy took other steps");
}

/// The control of `correction_rule_is_affine_covariant`: the max |F| rule on the same two problems must not pass
/// `same_steps`.
#[test]
fn affine_covariance_control_residual_rule_differs() {
    let (a, b) = (run(false, Damping::Residual), run(true, Damping::Residual));
    assert!(!same_steps(&a, &b), "control did not fire");
}
