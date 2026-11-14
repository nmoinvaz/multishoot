//! The Levenberg-Marquardt method: its Gram matrix against scalar dot products, a start where the Jacobian is singular
//! and Newton's method cannot step, and the planted multiple-shooting problem.
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

/// the largest difference of `gram` from scalar dot products of the columns (padding excluded), relative to max |G|:
/// (G's, g's)
fn gram_vs_scalar(bump: f64) -> (f64, f64) {
    let n = 21;
    let (c, ld, b) = matrix(n, 0.0);
    let (cb, _, _) = matrix(n, bump);
    let (g, jb) = gram::<Double>(n, ld, &cb, &b, true);
    let col = |j: usize| -> Vec<DD> { (0..n).map(|i| DD::new(c[0][j * ld + i], c[1][j * ld + i])).collect() };
    let dot = |x: &[DD], y: &[DD]| <Double as Precision<Native>>::dot(x.iter().copied().zip(y.iter().copied()));
    let (mut eg, mut ej, mut scale) = (0.0f64, 0.0f64, 0.0f64);
    for j in 0..n {
        for i in 0..n {
            let want = dot(&col(i), &col(j));
            let got = DD::new(g[0][j * ld + i], g[1][j * ld + i]);
            eg = eg.max((got - want).hi.abs());
            scale = scale.max(want.hi.abs());
        }
        ej = ej.max((jb[j] - dot(&col(j), &b)).hi.abs());
    }
    (eg / scale, ej / scale)
}

/// Pass, written before the first run: G = J^T J and g = J^T b from `gram` agree with scalar double-double dot
/// products of the columns to 1e-30 relative, with the padding rows full of 1e300 ignored. The control: one entry of J
/// off by 1e-20 relative must show in G above 1e-25.
#[test]
fn gram_matches_scalar_dot_products() {
    let (eg, ej) = gram_vs_scalar(0.0);
    println!("gram against scalar dots: G {eg:.3e}, g {ej:.3e}");
    assert!(eg <= 1e-30 && ej <= 1e-30);
    let (eg, _) = gram_vs_scalar(1e-20);
    println!("control, entry (2, 3) off by 1e-20: G {eg:.3e}");
    assert!(eg > 1e-25, "control did not fire");
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

/// LM in P from the singular start, handing over to Newton below 1e-12: (converged, LM steps, max |F|, |z0 - z1|,
/// |2 z0^2 - 1|)
fn circle<P: Precision<Native>>(tol: f64) -> (bool, usize, f64, f64, f64) {
    let mut z = singular_start::<P>();
    let opts = LmOpts {
        tol,
        newton: NewtonOpts {
            tol,
            ..Default::default()
        },
        ..Default::default()
    };
    let rep = levenberg_marquardt::<P, _>(&Circle, &mut z, &opts).unwrap();
    let (a, b) = (sc::<P>(&parts(&z), 0), sc::<P>(&parts(&z), 1));
    let f = max_abs::<P>(&Problem::<P>::residual(&Circle, &parts(&z), false).unwrap());
    let two = P::from_dd(DD::from_f64(2.0));
    (
        rep.converged,
        rep.mu.len(),
        f,
        P::to_dd(a - b).hi.abs(),
        P::to_dd(two * a * a - P::ONE).hi.abs(),
    )
}

/// Pass, written before the first run: from (0.5, -0.5), where the Jacobian is singular, LM reaches the root to
/// max |F| <= 1e-28 in double-double and 1e-60 in quad-double (with its handover to Newton), within 50 LM steps, with
/// z0 = z1 and 2 z0^2 = 1 to the same accuracy. The control: Newton's method from the same start fails there, its LU
/// singular or its steps not converging.
#[test]
fn lm_starts_where_the_jacobian_is_singular() {
    let (ok, steps, f, d, r) = circle::<Double>(1e-28);
    println!("double-double: converged {ok} after {steps} LM steps, max |F| {f:.3e}, |z0 - z1| {d:.3e}, |2 z0^2 - 1| {r:.3e}");
    assert!(ok && f <= 1e-28 && d <= 1e-28 && r <= 1e-28);
    let (ok, steps, f, d, r) = circle::<Quad>(1e-60);
    println!(
        "quad-double: converged {ok} after {steps} LM steps, max |F| {f:.3e}, |z0 - z1| {d:.3e}, |2 z0^2 - 1| {r:.3e}"
    );
    assert!(ok && f <= 1e-60 && d <= 1e-60 && r <= 1e-60);
    let mut z = singular_start::<Double>();
    let nr = newton::<Double, _>(&Circle, &mut z, &NewtonOpts::default());
    println!(
        "control, Newton from the same start: {:?}",
        nr.as_ref().map(|r| r.converged)
    );
    assert!(nr.map_or(true, |r| !r.converged), "control did not fire");
}

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

/// Pass, written before the first run: on the planted multiple-shooting problem from z* perturbed by 1e-3, LM
/// converges to max |F| <= 1e-28 (with its handover to Newton) and reaches z* to 1e-27.
#[test]
fn lm_converges_on_the_planted_problem() {
    let (sh, zs) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let mut z = perturbed(&zs, 1e-3);
    let rep = levenberg_marquardt::<Double, _>(&sh, &mut z, &LmOpts::default()).unwrap();
    let err = (0..sh.len())
        .map(|i| (DD::new(z[0][i], z[1][i]) - DD::new(zs[0][i], zs[1][i])).hi.abs())
        .fold(0.0, nan_max);
    println!(
        "planted: converged {} after {} LM steps and {} factorizations, max |z - z*| {err:.3e}",
        rep.converged,
        rep.mu.len(),
        rep.factorizations
    );
    assert!(rep.converged && err <= 1e-27);
}
