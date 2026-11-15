//! The exact tangent (multisolver's predictor) on the eigenproblem `Cylinder`. The predictors' model-problem tests are
//! multisolver's.
mod common;

use common::{perturbed, Cylinder};
use multishoot::{Double, DD};
use multisolver::prec::{nan_max, parts};
use multisolver::predict::tangent;
use multisolver::NewtonOpts;

/// max |t - e_lambda| over the unknowns (NaN where any entry is), og the eigenvalue's index
fn tangent_err(t: &[Vec<f64>], og: usize) -> f64 {
    (0..t[0].len())
        .map(|i| {
            (DD::new(t[0][i], t[1][i]) - if i == og { DD::ONE } else { DD::ZERO })
                .hi
                .abs()
        })
        .fold(0.0, nan_max)
}

/// Pass, written before the first run: the tangent dz/dmu of `Cylinder` at its double-double root is e_lambda, one in
/// the eigenvalue and zero elsewhere, to 1e-28. The controls: with mu frozen in the residual's series it is zero, and
/// the tangent with one component NaN fails the check.
#[test]
fn tangent_is_the_eigenvalue_direction() {
    let run = |freeze: bool| {
        let (mut sh, zs) = Cylinder::<1>::shooting(Cylinder::new(), 30);
        let mut z = perturbed(&zs, 1e-3);
        let opts = NewtonOpts {
            tol: 1e-28,
            max_iter: 12,
            ..Default::default()
        };
        assert!(sh.newton::<Double>(&mut z, &opts).unwrap().converged);
        sh.eq.freeze_mu = freeze;
        (tangent::<Double, _>(&sh, &parts(&z), 1, true).unwrap(), sh.off_global())
    };
    let ((t, og), (tc, _)) = (run(false), run(true));
    let (e, c) = (tangent_err(&t, og), tangent_err(&tc, og));
    println!("tangent against e_lambda {e:.3e}; control, mu frozen {c:.3e}");
    assert!(e <= 1e-28);
    assert!(c > 0.5, "control did not fire");
    let mut tn = t.clone();
    let k = (og + 1) % tn[0].len();
    tn[0][k] = f64::NAN;
    let e = tangent_err(&tn, og);
    println!("control, one component NaN: {e:.3e}");
    let accurate = |e: f64| e <= 1e-28;
    assert!(!accurate(e), "control did not fire");
}
