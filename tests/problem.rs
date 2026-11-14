//! Newton's method on a problem that is not multiple shooting: a cyclic cubic system written directly as a
//! [`Problem`], with a manufactured root, in double-double and quad-double.
use multishoot::multidouble::linalg::ld_for;
use multishoot::newton::{newton, NewtonOpts};
use multishoot::prec::{max_abs, nan_max, parts};
use multishoot::{Double, Native, Precision, Problem, Quad, DD};

const N: usize = 12;

/// F_i(z) = z_i^3 + z_i + c z_(i+1) - b_i, the index cyclic, with b = F's terms at the root z*_i = 0.5 + 0.1 sin(i),
/// so that F(z*) = 0 exactly in P. The Jacobian has 3 z_i^2 + 1 on the diagonal and c on the cyclic superdiagonal.
/// `diag_err` scales the Jacobian's diagonal, 1 for the exact one.
struct Cubic<P: Precision<Native>> {
    c: P::Scalar,
    b: Vec<P::Scalar>,
    diag_err: f64,
}

impl<P: Precision<Native>> Cubic<P> {
    fn new(diag_err: f64) -> (Self, Vec<Vec<f64>>) {
        let c = P::from_dd(DD::from_f64(0.3));
        let zs: Vec<P::Scalar> = (0..N)
            .map(|i| P::from_dd(DD::from_f64(0.5 + 0.1 * (i as f64).sin())))
            .collect();
        let mut pr = Cubic {
            c,
            b: vec![P::ZERO; N],
            diag_err,
        };
        pr.b = pr.terms(&zs);
        (pr, to_parts::<P>(&zs))
    }

    /// z_i^3 + z_i + c z_(i+1)
    fn terms(&self, z: &[P::Scalar]) -> Vec<P::Scalar> {
        (0..N)
            .map(|i| z[i] * z[i] * z[i] + z[i] + self.c * z[(i + 1) % N])
            .collect()
    }
}

/// the scalars of a state held as part arrays
fn scalars<P: Precision<Native>>(z: &[&[f64]]) -> Vec<P::Scalar> {
    (0..z[0].len())
        .map(|i| P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>()))
        .collect()
}

/// part arrays of scalars
fn to_parts<P: Precision<Native>>(v: &[P::Scalar]) -> Vec<Vec<f64>> {
    (0..P::PARTS)
        .map(|q| v.iter().map(|&x| P::part(x, q)).collect())
        .collect()
}

impl<P: Precision<Native>> Problem<P> for Cubic<P> {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let z = scalars::<P>(z);
        Ok(self.terms(&z).iter().zip(&self.b).map(|(&t, &b)| t - b).collect())
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        let three = P::from_dd(DD::from_f64(3.0));
        let err = P::from_dd(DD::from_f64(self.diag_err));
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; P::PARTS];
        for i in 0..N {
            let d = (three * z[i] * z[i] + P::ONE) * err;
            for (q, cq) in c.iter_mut().enumerate() {
                cq[i * ld + i] = P::part(d, q);
                cq[((i + 1) % N) * ld + i] = P::part(self.c, q);
            }
        }
        Ok((c, ld))
    }
}

/// Newton from z* + 1e-2 wave in P: (converged, steps, max |F| at the end, max |z - z*|)
fn run<P: Precision<Native>>(diag_err: f64, tol: f64) -> (bool, usize, f64, f64) {
    let (pr, zs) = Cubic::<P>::new(diag_err);
    let mut z = zs.clone();
    for (i, v) in z[0].iter_mut().enumerate() {
        *v += 1e-2 * (1.0 + 0.5 * (3.0 * i as f64).cos());
    }
    let opts = NewtonOpts {
        tol,
        max_iter: 8,
        ..Default::default()
    };
    let rep = newton::<P, _>(&pr, &mut z, &opts).unwrap();
    let (zn, zt) = (scalars::<P>(&parts(&z)), scalars::<P>(&parts(&zs)));
    let err = zn
        .iter()
        .zip(&zt)
        .map(|(&a, &b)| P::to_dd(a - b).hi.abs())
        .fold(0.0, nan_max);
    let f = max_abs::<P>(&pr.residual(&parts(&z), false).unwrap());
    (rep.converged, rep.scale.len(), f, err)
}

/// Pass, written before the first run: from a start 1e-2 off the root, Newton converges within 8 steps in
/// double-double to max |F| <= 1e-28 and in quad-double to max |F| <= 1e-60, with max |z - z*| below 1e-27 and
/// 1e-59. Eight steps leave room for the quadratic rate (about five from 1e-2), not for a linear one.
#[test]
fn newton_converges_on_a_problem_that_is_not_shooting() {
    let (ok, steps, f, err) = run::<Double>(1.0, 1e-28);
    println!("double-double: converged {ok} in {steps} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}");
    assert!(ok && f <= 1e-28 && err <= 1e-27);
    let (ok, steps, f, err) = run::<Quad>(1.0, 1e-60);
    println!("quad-double: converged {ok} in {steps} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}");
    assert!(ok && f <= 1e-60 && err <= 1e-59);
}

/// The control: with the Jacobian's diagonal 10% off, Newton contracts only linearly, by about 0.1 per step, and must
/// fail to converge to 1e-28 within the 8 steps.
#[test]
fn newton_control_wrong_jacobian_does_not_converge() {
    let (ok, steps, f, _) = run::<Double>(1.1, 1e-28);
    println!("control, diagonal 10% off: converged {ok} in {steps} steps, max |F| {f:.3e}");
    assert!(!ok, "control did not fire");
}
