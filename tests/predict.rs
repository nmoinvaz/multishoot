//! The continuation predictors on planted branches: Lagrange extrapolation and the cubic Hermite on polynomial
//! branches, in double-double and quad-double, and the exact tangent on the eigenproblem `Cylinder`.
mod common;

use common::{perturbed, Cylinder};
use multishoot::prec::parts;
use multishoot::predict::{hermite, lagrange, tangent};
use multishoot::{Double, Native, NewtonOpts, Precision, Quad, DD};

const N: usize = 6;

/// coefficient d of unknown i of the planted branch, a double-double with a nonzero lo part
fn coef(i: usize, d: usize) -> DD {
    let v = ((3 * i + 5 * d) as f64).sin() / (d + 1) as f64;
    DD::new(v, v * 1e-17 * (1.0 + i as f64 / 7.0))
}

/// the branch z_i(p) = sum_(d <= deg) coef(i, d) p^d (or its derivative) at p, as part arrays of P
fn branch<P: Precision<Native>>(deg: usize, p: f64, deriv: bool, scale: f64) -> Vec<Vec<f64>> {
    let pp = P::from_dd(DD::from_f64(p));
    let v: Vec<P::Scalar> = (0..N)
        .map(|i| {
            let terms = (0..=deg).rev().filter(|&d| !deriv || d > 0);
            let mut s = P::ZERO;
            for d in terms {
                let c = P::from_dd(coef(i, d))
                    * if deriv {
                        P::from_dd(DD::from_f64(d as f64))
                    } else {
                        P::ONE
                    };
                s = s * pp + c;
            }
            s * P::from_dd(DD::new(1.0, scale))
        })
        .collect();
    (0..P::PARTS)
        .map(|q| v.iter().map(|&x| P::part(x, q)).collect())
        .collect()
}

/// max |a - b| / max |b| over the unknowns, in P
fn rel<P: Precision<Native>>(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let sc = |z: &[Vec<f64>], i: usize| P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>());
    let d = (0..N)
        .map(|i| P::to_dd(sc(a, i) - sc(b, i)).hi.abs())
        .fold(0.0, f64::max);
    d / (0..N).map(|i| P::to_dd(sc(b, i)).hi.abs()).fold(0.0, f64::max)
}

/// Lagrange through the branch of degree deg at p = 0.1, 0.2, 0.3, 0.4, extrapolated to 0.55: relative error
fn lagrange_err<P: Precision<Native>>(deg: usize) -> f64 {
    let ps = [0.1, 0.2, 0.3, 0.4];
    let zs: Vec<Vec<Vec<f64>>> = ps.iter().map(|&p| branch::<P>(deg, p, false, 0.0)).collect();
    let refs: Vec<Vec<&[f64]>> = zs.iter().map(|z| parts(z)).collect();
    let pts: Vec<(f64, &[&[f64]])> = ps.iter().zip(&refs).map(|(&p, r)| (p, &r[..])).collect();
    rel::<P>(&lagrange::<P>(&pts, 0.55), &branch::<P>(deg, 0.55, false, 0.0))
}

/// Pass, written before the first run: through four points of a cubic branch, Lagrange extrapolation to 0.55 is exact
/// to 1e-30 relative in double-double and 1e-62 in quad-double. The control: a quartic branch misses by more than
/// 1e-8.
#[test]
fn lagrange_is_exact_on_a_cubic_branch() {
    let (d, q, c) = (
        lagrange_err::<Double>(3),
        lagrange_err::<Quad>(3),
        lagrange_err::<Double>(4),
    );
    println!("lagrange: double-double {d:.3e}, quad-double {q:.3e}; control, quartic {c:.3e}");
    assert!(d <= 1e-30 && q <= 1e-62);
    assert!(c > 1e-8, "control did not fire");
}

/// the cubic Hermite through p = 0.2 and 0.5 of the cubic branch, at 0.7, the tangent at 0.5 scaled by 1 + bump:
/// relative error
fn hermite_err<P: Precision<Native>>(bump: f64) -> f64 {
    let (z0, t0) = (branch::<P>(3, 0.2, false, 0.0), branch::<P>(3, 0.2, true, 0.0));
    let (z1, t1) = (branch::<P>(3, 0.5, false, 0.0), branch::<P>(3, 0.5, true, bump));
    let h = hermite::<P>((0.2, &parts(&z0), &parts(&t0)), (0.5, &parts(&z1), &parts(&t1)), 0.7);
    rel::<P>(&h, &branch::<P>(3, 0.7, false, 0.0))
}

/// Pass, written before the first run: the cubic Hermite through two points of a cubic branch and its tangents
/// there is exact at 0.7, to 1e-30 relative in double-double and 1e-62 in quad-double. The control: one tangent off
/// by 1e-10 relative misses by more than 1e-14.
#[test]
fn hermite_is_exact_on_a_cubic_branch() {
    let (d, q, c) = (
        hermite_err::<Double>(0.0),
        hermite_err::<Quad>(0.0),
        hermite_err::<Double>(1e-10),
    );
    println!("hermite: double-double {d:.3e}, quad-double {q:.3e}; control, tangent off by 1e-10 {c:.3e}");
    assert!(d <= 1e-30 && q <= 1e-62);
    assert!(c > 1e-14, "control did not fire");
}

/// Pass, written before the first run: the tangent dz/dmu of `Cylinder` at its double-double root is e_lambda, one in
/// the eigenvalue and zero elsewhere, to 1e-28. The control: with mu frozen in the residual's series it is zero.
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
        let t = tangent::<Double, _>(&sh, &parts(&z), 1, true).unwrap();
        let og = sh.off_global();
        (0..sh.len())
            .map(|i| {
                (DD::new(t[0][i], t[1][i]) - if i == og { DD::ONE } else { DD::ZERO })
                    .hi
                    .abs()
            })
            .fold(0.0, f64::max)
    };
    let (e, c) = (run(false), run(true));
    println!("tangent against e_lambda {e:.3e}; control, mu frozen {c:.3e}");
    assert!(e <= 1e-28);
    assert!(c > 0.5, "control did not fire");
}
