//! The asymptotic numerical method (multisolver's) on the eigenproblem `Cylinder`, whose eigenvalue pi^2 + mu is
//! linear in its parameter mu while its mode does not change. Its model-problem tests are multisolver's.
mod common;

use common::{perturbed, Cylinder};
use multishoot::{Double, DD};
use multisolver::anm::{series, Branch};
use multisolver::prec::{nan_max, parts};
use multisolver::NewtonOpts;

const B: usize = 1;

/// The series of order `order` of `Cylinder`'s branch in mu at its double-double root, refined once, and the index
/// of the eigenvalue among the unknowns
fn cylinder(freeze: bool, order: usize) -> (Branch<Double>, usize) {
    let (mut sh, zs) = Cylinder::<B>::shooting(Cylinder::new(), 30);
    let mut z = perturbed(&zs, 1e-3);
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 12,
        ..Default::default()
    };
    assert!(sh.newton::<Double>(&mut z, &opts).unwrap().converged);
    sh.eq.freeze_mu = freeze;
    (
        series::<Double, _>(&sh, &parts(&z), order, 1, true).unwrap(),
        sh.off_global(),
    )
}

/// [|z_1 - e_lambda|, max over k >= 2 of |z_k|], NaN where any entry is
fn cylinder_errors(br: &Branch<Double>, og: usize) -> [f64; 2] {
    let e1 = br.z[1]
        .iter()
        .enumerate()
        .map(|(i, &v)| (v - if i == og { DD::ONE } else { DD::ZERO }).hi.abs())
        .fold(0.0, nan_max);
    let high = (2..=br.order())
        .flat_map(|k| br.z[k].iter().map(|v| v.hi.abs()))
        .fold(0.0, nan_max);
    [e1, high]
}

/// the accuracy checks of Cylinder's branch: [z_1 within 1e-28, the higher orders within 1e-26]
fn cylinder_checks(e: [f64; 2]) -> [bool; 2] {
    [e[0] <= 1e-28, e[1] <= 1e-26]
}

/// Pass, written before the first run: through the multiple-shooting problem, the series of order 10 (jets of 8 and
/// 16 coefficients) gives z_1 = e_lambda, one in the eigenvalue and zero elsewhere, to 1e-28, and every higher order
/// below 1e-26. The controls: with mu frozen in the residual's series, z_1 misses e_lambda by all of it (its higher
/// orders are zero); z_2 = 1e-20 in one entry, and a NaN in z_4, fail the higher orders.
#[test]
fn anm_follows_the_eigenvalue_through_shooting() {
    let (br, og) = cylinder(false, 10);
    let e = cylinder_errors(&br, og);
    let res = br.residual.iter().copied().fold(0.0, nan_max);
    println!(
        "cylinder: |z_1 - e_lambda| {:.3e}, z_2..z_10 {:.3e}, solve residuals up to {res:.3e}",
        e[0], e[1]
    );
    assert_eq!(cylinder_checks(e), [true; 2]);
    let control = |name: &str, br: &Branch<Double>, want: [bool; 2]| {
        let e = cylinder_errors(br, og);
        println!(
            "control, {name}: |z_1 - e_lambda| {:.3e}, higher orders {:.3e}",
            e[0], e[1]
        );
        assert_eq!(cylinder_checks(e), want, "control {name} did not fire");
    };
    control("mu frozen", &cylinder(true, 3).0, [false, true]);
    let mut bad = br.clone();
    bad.z[2][0] = DD::from_f64(1e-20);
    control("z_2 = 1e-20", &bad, [true, false]);
    let mut bad = br;
    bad.z[4][0] = DD::from_f64(f64::NAN);
    control("NaN in z_4", &bad, [true, false]);
}
