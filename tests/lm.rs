//! The Levenberg-Marquardt method: its Gram matrix against scalar dot products, a start where the Jacobian is singular
//! and Newton's method cannot step, its handover to Newton, and the planted multiple-shooting problem.
mod common;

use common::{perturbed, Planted};
use multishoot::lm::{gram, levenberg_marquardt, LmOpts};
use multishoot::multidouble::linalg::ld_for;
use multishoot::newton::{newton, NewtonOpts};
use multishoot::prec::{max_abs, nan_max, parts};
use multishoot::{Double, Native, Precision, Problem, Quad, DD};

/// a J of n x n with entries sin(3 i + 7 j) + i j / n^2 in double-double (nonzero lo parts), as `Problem::jacobian`
/// lays it out, rows n..ld filled with 1e300 (which `gram` must ignore), and b_i = cos(i)
fn matrix(n: usize, bump: f64) -> (Vec<Vec<f64>>, usize, Vec<DD>) {
    let ld = ld_for(n);
    let mut c = vec![vec![1e300; ld * n]; 2];
    for j in 0..n {
        for i in 0..n {
            let mut v = DD::new(((3 * i + 7 * j) as f64).sin(), 1e-17 * ((i + j) as f64).cos())
                + DD::from_f64((i * j) as f64 / (n * n) as f64);
            if (i, j) == (2, 3) {
                v = v * DD::new(1.0, bump);
            }
            (c[0][j * ld + i], c[1][j * ld + i]) = (v.hi, v.lo);
        }
    }
    (c, ld, (0..n).map(|i| DD::from_f64((i as f64).cos())).collect())
}

/// a wrong input given to `gram`
#[derive(Clone, Copy, Debug)]
enum Fault {
    None,
    /// entry (2, 3) of J off by 1e-20 relative
    J,
    /// b_3 off by 1e-20 relative
    B,
    /// b_0 NaN
    Nan,
}

/// the largest difference of `gram`, given the fault, from scalar dot products of the true columns and b (padding
/// excluded), relative to max |G|: (G's, g's), NaN where any difference is
fn gram_vs_scalar(fault: Fault) -> (f64, f64) {
    let n = 21;
    let (c, ld, b) = matrix(n, 0.0);
    let (cb, _, _) = matrix(n, if matches!(fault, Fault::J) { 1e-20 } else { 0.0 });
    let mut bb = b.clone();
    match fault {
        Fault::B => bb[3] = bb[3] * DD::new(1.0, 1e-20),
        Fault::Nan => bb[0] = DD::from_f64(f64::NAN),
        _ => {}
    }
    let (g, jb) = gram::<Double>(n, ld, &cb, &bb, true);
    let col = |j: usize| -> Vec<DD> { (0..n).map(|i| DD::new(c[0][j * ld + i], c[1][j * ld + i])).collect() };
    let dot = |x: &[DD], y: &[DD]| <Double as Precision<Native>>::dot(x.iter().copied().zip(y.iter().copied()));
    let (mut eg, mut ej, mut scale) = (0.0f64, 0.0f64, 0.0f64);
    for j in 0..n {
        for i in 0..n {
            let want = dot(&col(i), &col(j));
            let got = DD::new(g[0][j * ld + i], g[1][j * ld + i]);
            eg = nan_max(eg, (got - want).hi.abs());
            scale = nan_max(scale, want.hi.abs());
        }
        ej = nan_max(ej, (jb[j] - dot(&col(j), &b)).hi.abs());
    }
    (eg / scale, ej / scale)
}

/// the checks of `gram`: [G, g] within 1e-30 relative of the scalar dot products
fn gram_checks((eg, ej): (f64, f64)) -> [bool; 2] {
    [eg <= 1e-30, ej <= 1e-30]
}

/// Pass, written before the first run: G = J^T J and g = J^T b from `gram` agree with scalar double-double dot
/// products of the columns to 1e-30 relative, with the padding rows full of 1e300 ignored. The controls: one entry of
/// J off by 1e-20 relative fails both checks, b_3 off by 1e-20 relative fails g's, and a NaN in b fails g's.
#[test]
fn gram_matches_scalar_dot_products() {
    let (eg, ej) = gram_vs_scalar(Fault::None);
    println!("gram against scalar dots: G {eg:.3e}, g {ej:.3e}");
    assert_eq!(gram_checks((eg, ej)), [true; 2]);
    for (fault, want) in [
        (Fault::J, [false; 2]),
        (Fault::B, [true, false]),
        (Fault::Nan, [true, false]),
    ] {
        let (eg, ej) = gram_vs_scalar(fault);
        println!("control, {fault:?}: G {eg:.3e}, g {ej:.3e}");
        assert_eq!(gram_checks((eg, ej)), want, "control {fault:?} did not fire");
    }
}

/// F(z) = (z0^2 + z1^2 - 1, z0 - z1): the circle and the diagonal, roots +-(1, 1) / sqrt(2). Its Jacobian
/// [[2 z0, 2 z1], [1, -1]] is singular on z1 = -z0.
struct Circle;

fn sc<P: Precision<Native>>(z: &[&[f64]], i: usize) -> P::Scalar {
    P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>())
}

impl<P: Precision<Native>> Problem<P> for Circle {
    fn len(&self) -> usize {
        2
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let (a, b) = (sc::<P>(z, 0), sc::<P>(z, 1));
        Ok(vec![a * a + b * b - P::ONE, a - b])
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let (a, b) = (sc::<P>(z, 0), sc::<P>(z, 1));
        let two = P::from_dd(DD::from_f64(2.0));
        let ld = ld_for(2);
        let mut c = vec![vec![0.0; ld * 2]; P::PARTS];
        for (q, cq) in c.iter_mut().enumerate() {
            cq[0] = P::part(two * a, q);
            cq[1] = P::part(P::ONE, q);
            cq[ld] = P::part(two * b, q);
            cq[ld + 1] = P::part(-P::ONE, q);
        }
        Ok((c, ld))
    }
}

/// (0.5, -0.5) in P's parts: on the line where the Jacobian is singular, with J^T F = (0.5, -0.5) nonzero
fn singular_start<P: Precision<Native>>() -> Vec<Vec<f64>> {
    (0..P::PARTS)
        .map(|q| if q == 0 { vec![0.5, -0.5] } else { vec![0.0, 0.0] })
        .collect()
}

/// LM in P from the singular start, handing over to Newton below 1e-12: (converged, LM steps, final z)
fn circle<P: Precision<Native>>(tol: f64) -> (bool, usize, Vec<Vec<f64>>) {
    let mut z = singular_start::<P>();
    let opts = LmOpts {
        tol,
        ..Default::default()
    };
    let rep = levenberg_marquardt::<P, _>(&Circle, &mut z, &opts).unwrap();
    (rep.converged, rep.mu.len(), z)
}

/// (max |F|, |z0 - z1|, |2 z0^2 - 1|) of Circle at z in P
fn circle_errors<P: Precision<Native>>(z: &[Vec<f64>]) -> (f64, f64, f64) {
    let (a, b) = (sc::<P>(&parts(z), 0), sc::<P>(&parts(z), 1));
    let f = max_abs::<P>(&Problem::<P>::residual(&Circle, &parts(z), false).unwrap());
    let two = P::from_dd(DD::from_f64(2.0));
    (f, P::to_dd(a - b).hi.abs(), P::to_dd(two * a * a - P::ONE).hi.abs())
}

/// the accuracy checks of a root of Circle: [max |F|, |z0 - z1|, |2 z0^2 - 1|] each at most tol
fn circle_checks((f, d, r): (f64, f64, f64), tol: f64) -> [bool; 3] {
    [f <= tol, d <= tol, r <= tol]
}

/// LM from the singular start in P passes the checks at tol, and its root with eps added to z0 (the control) fails
/// all three
fn circle_case<P: Precision<Native>>(name: &str, tol: f64, eps: f64) {
    let (ok, steps, z) = circle::<P>(tol);
    let (f, d, r) = circle_errors::<P>(&z);
    println!("{name}: converged {ok} after {steps} LM steps, max |F| {f:.3e}, |z0 - z1| {d:.3e}, |2 z0^2 - 1| {r:.3e}");
    assert!(ok && circle_checks((f, d, r), tol) == [true; 3]);
    let mut zw = z.clone();
    let a = sc::<P>(&parts(&z), 0) + P::from_dd(DD::from_f64(eps));
    for (q, p) in zw.iter_mut().enumerate() {
        p[0] = P::part(a, q);
    }
    let (f, d, r) = circle_errors::<P>(&zw);
    println!("control, z0 off by {eps:.0e}: max |F| {f:.3e}, |z0 - z1| {d:.3e}, |2 z0^2 - 1| {r:.3e}");
    assert_eq!(circle_checks((f, d, r), tol), [false; 3], "control did not fire");
}

/// Pass, written before the first run: from (0.5, -0.5), where the Jacobian is singular, LM reaches the root to
/// max |F| <= 1e-28 in double-double and 1e-60 in quad-double (with its handover to Newton), within 50 LM steps, with
/// z0 = z1 and 2 z0^2 = 1 to the same accuracy. The controls: the root with z0 off by 1e-20 (1e-50 in quad-double)
/// fails all three checks, and Newton's method from the same start fails there, its LU singular or its steps not
/// converging.
#[test]
fn lm_starts_where_the_jacobian_is_singular() {
    circle_case::<Double>("double-double", 1e-28, 1e-20);
    circle_case::<Quad>("quad-double", 1e-60, 1e-50);
    let mut z = singular_start::<Double>();
    let nr = newton::<Double, _>(&Circle, &mut z, &NewtonOpts::default());
    println!(
        "control, Newton from the same start: {:?}",
        nr.as_ref().map(|r| r.converged)
    );
    assert!(nr.map_or(true, |r| !r.converged), "control did not fire");
}

/// F(z) = (1e-14 (z0 - 1), 0): the root z0 = 1 for every z1, with the Jacobian [[1e-14, 0], [0, 0]] singular
/// everywhere
struct Weak;

impl Problem<Double> for Weak {
    fn len(&self) -> usize {
        2
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        Ok(vec![
            DD::from_f64(1e-14) * (DD::new(z[0][0], z[1][0]) - DD::ONE),
            DD::ZERO,
        ])
    }
    fn jacobian(&self, _z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(2);
        let mut c = vec![vec![0.0; ld * 2]; 2];
        c[0][0] = 1e-14;
        Ok((c, ld))
    }
}

/// Pass, written before the first run: from (0, 0), where max |F| = 1e-14 is below the handover's 1e-12 and the
/// Jacobian is singular, LM returns, converged, with max |F| <= 1e-28 and no Newton report (Newton returned an
/// error). The control: Newton from the same start returns that error, which the handover used to pass on, and the
/// check rejects it.
#[test]
fn lm_goes_on_where_the_handover_fails() {
    let check = |r: &Result<(bool, f64), String>| matches!(r, Ok((true, f)) if *f <= 1e-28);
    let mut z = vec![vec![0.0; 2]; 2];
    let lm = levenberg_marquardt::<Double, _>(&Weak, &mut z, &LmOpts::default());
    let newton_report = lm.as_ref().map(|r| r.newton.is_some()).map_err(|e| e.clone());
    let lm = lm.map(|r| {
        (
            r.converged,
            max_abs::<Double>(&Weak.residual(&parts(&z), false).unwrap()),
        )
    });
    println!("LM from (0, 0): {lm:?}, Newton report {newton_report:?}");
    assert!(check(&lm) && newton_report == Ok(false));
    let mut z = vec![vec![0.0; 2]; 2];
    let nr =
        newton::<Double, _>(&Weak, &mut z, &NewtonOpts::default()).map(|r| (r.converged, *r.fnorm.last().unwrap()));
    println!("control, Newton from (0, 0): {nr:?}");
    assert!(!check(&nr), "control did not fire");
}

/// F(z) = z, whose Newton step lands on the root 0 exactly
struct Ident;

impl Problem<Quad> for Ident {
    fn len(&self) -> usize {
        1
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<<Quad as Precision<Native>>::Scalar>, String> {
        Ok(vec![sc::<Quad>(z, 0)])
    }
    fn jacobian(&self, _z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(1);
        let mut c = vec![vec![0.0; ld]; 4];
        c[0][0] = 1.0;
        Ok((c, ld))
    }
}

/// Pass, written before the first run: from z = 1e-30 in quad-double with tol 1e-60, LM hands over to Newton at once
/// and reports convergence with max |F| <= 1e-60 at its final state, the last max |F| of its report. The controls:
/// Newton with its own default tolerance (1e-28), as the handover used to run it, reports convergence at max |F| =
/// 1e-30, and LM's report without Newton's final entry ends at 1e-30; the check rejects both.
#[test]
fn lm_hands_over_with_its_own_tolerance() {
    // (converged, the report's last max |F|, max |F| at the final state)
    let check = |converged: bool, reported: f64, f: f64| converged && f <= 1e-60 && reported == f;
    let start = || vec![vec![1e-30], vec![0.0], vec![0.0], vec![0.0]];
    let fmax = |z: &[Vec<f64>]| max_abs::<Quad>(&Ident.residual(&parts(z), false).unwrap());
    let mut z = start();
    let opts = LmOpts {
        tol: 1e-60,
        ..Default::default()
    };
    let rep = levenberg_marquardt::<Quad, _>(&Ident, &mut z, &opts).unwrap();
    let last = *rep.fnorm.last().unwrap();
    println!(
        "LM to 1e-60: converged {}, max |F| {:.3e}, reported {last:.3e}, handed over {}",
        rep.converged,
        fmax(&z),
        rep.newton.is_some()
    );
    assert!(check(rep.converged, last, fmax(&z)) && rep.newton.is_some());
    let before = rep.fnorm[rep.fnorm.len() - 2];
    println!("control, report without Newton's final entry: reported {before:.3e}");
    assert!(!check(rep.converged, before, fmax(&z)), "control did not fire");
    let mut z = start();
    let nr = newton::<Quad, _>(&Ident, &mut z, &NewtonOpts::default()).unwrap();
    println!(
        "control, Newton at its own tolerance: converged {}, max |F| {:.3e}",
        nr.converged,
        fmax(&z)
    );
    assert!(
        !check(nr.converged, *nr.fnorm.last().unwrap(), fmax(&z)),
        "control did not fire"
    );
}

/// F(z) = z - 1, its residual an error past `limit` (an integration that fails away from the start)
struct Fragile {
    limit: f64,
}

impl Problem<Double> for Fragile {
    fn len(&self) -> usize {
        1
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        let a = DD::new(z[0][0], z[1][0]);
        if a.hi > self.limit {
            return Err("integration failed".into());
        }
        Ok(vec![a - DD::ONE])
    }
    fn jacobian(&self, _z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(1);
        let mut c = vec![vec![0.0; ld]; 2];
        c[0][0] = 1.0;
        Ok((c, ld))
    }
}

/// Pass, written before the first run: with the residual an error past z = 0.5, LM from 0 returns that error at its
/// first trial state, near 1. The control: with the residual valid up to 2, LM converges, and the check, which wants
/// the error, rejects that result.
#[test]
fn lm_returns_the_problems_error() {
    let check = |r: &Result<bool, String>| matches!(r, Err(e) if e == "integration failed");
    let run = |limit: f64| {
        let mut z = vec![vec![0.0], vec![0.0]];
        levenberg_marquardt::<Double, _>(&Fragile { limit }, &mut z, &LmOpts::default()).map(|r| r.converged)
    };
    let r = run(0.5);
    println!("residual an error past 0.5: {r:?}");
    assert!(check(&r));
    let r = run(2.0);
    println!("control, residual valid up to 2: {r:?}");
    assert!(!check(&r), "control did not fire");
}

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
