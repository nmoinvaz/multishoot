//! Newton's method on a problem that is not multiple shooting: a cyclic cubic system written directly as a
//! [`Problem`], with a manufactured root, in double-double and quad-double, and the chord and Shamanskii variants that
//! reuse its factorization.
use multishoot::multidouble::linalg::ld_for;
use multishoot::newton::{newton, newton_chord, NewtonOpts, NewtonReport};
use multishoot::prec::{max_abs, nan_max, parts};
use multishoot::{Double, Native, Precision, Problem, Quad, DD};
use std::cell::Cell;

const N: usize = 12;

/// F_i(z) = z_i^3 + z_i + c z_(i+1) - b_i, the index cyclic, with b = F's terms at the root z*_i = 0.5 + 0.1 sin(i),
/// so that F(z*) = 0 exactly in P. The Jacobian has 3 z_i^2 + 1 on the diagonal and c on the cyclic superdiagonal.
/// `diag_err` scales the Jacobian's diagonal, 1 for the exact one.
struct Cubic<P: Precision<Native>> {
    c: P::Scalar,
    b: Vec<P::Scalar>,
    diag_err: f64,
    /// the Jacobians formed
    jacobians: Cell<usize>,
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
            jacobians: Cell::new(0),
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
        self.jacobians.set(self.jacobians.get() + 1);
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
    let mut z = start(&zs);
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
    assert_eq!(checks((ok, f, err), 1e-28, 1e-27), [true; 3]);
    let (ok, steps, f, err) = run::<Quad>(1.0, 1e-60);
    println!("quad-double: converged {ok} in {steps} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}");
    assert_eq!(checks((ok, f, err), 1e-60, 1e-59), [true; 3]);
}

/// the three checks of a run: converged, max |F| <= ftol and max |z - z*| <= etol
fn checks((ok, f, err): (bool, f64, f64), ftol: f64, etol: f64) -> [bool; 3] {
    [ok, f <= ftol, err <= etol]
}

/// The control: with the Jacobian's diagonal 10% off, Newton contracts only linearly, by about 0.1 per step, and must
/// fail every check of the run within the 8 steps, in double-double and in quad-double: it does not converge, and both
/// max |F| and max |z - z*| stay above their bounds.
#[test]
fn newton_control_wrong_jacobian_does_not_converge() {
    let (ok, steps, f, err) = run::<Double>(1.1, 1e-28);
    println!("control, diagonal 10% off: converged {ok} in {steps} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}");
    assert_eq!(checks((ok, f, err), 1e-28, 1e-27), [false; 3], "control did not fire");
    let (ok, steps, f, err) = run::<Quad>(1.1, 1e-60);
    println!("control, quad-double: converged {ok} in {steps} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}");
    assert_eq!(
        checks((ok, f, err), 1e-60, 1e-59),
        [false; 3],
        "quad-double control did not fire"
    );
}

/// the start of the runs: z* + 1e-2 wave
fn start(zs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut z = zs.to_vec();
    for (i, v) in z[0].iter_mut().enumerate() {
        *v += 1e-2 * (1.0 + 0.5 * (3.0 * i as f64).cos());
    }
    z
}

/// `newton_chord` in double-double from the start, its factors formed every `every` steps (`every` 0: plain
/// `newton`): (report, final z, Jacobians formed)
fn chord(every: usize, max_iter: usize) -> (NewtonReport, Vec<Vec<f64>>, usize) {
    let (pr, zs) = Cubic::<Double>::new(1.0);
    let mut z = start(&zs);
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter,
        ..Default::default()
    };
    let rep = if every == 0 {
        newton::<Double, _>(&pr, &mut z, &opts).unwrap()
    } else {
        newton_chord::<Double, _>(&pr, &mut z, &opts, every).unwrap()
    };
    (rep, z, pr.jacobians.get())
}

/// the bits of a report and a state
fn bits(rep: &NewtonReport, z: &[Vec<f64>]) -> Vec<u64> {
    rep.fnorm
        .iter()
        .chain(&rep.scale)
        .chain(&rep.theta)
        .chain(z.iter().flatten())
        .map(|v| v.to_bits())
        .chain([rep.converged as u64])
        .collect()
}

/// the two checks that a run is plain Newton: [same report and state bits, same Jacobian count]
fn same_as_newton(every: usize) -> [bool; 2] {
    let (ra, za, ja) = chord(0, 8);
    let (rb, zb, jb) = chord(every, 8);
    [bits(&ra, &za) == bits(&rb, &zb), ja == jb]
}

/// Pass, written before the first run: `newton_chord` with its factors formed at every step is `newton` bit for bit,
/// in the report and the solution, and forms as many Jacobians. The control: formed every third step, it fails both
/// checks.
#[test]
fn chord_every_step_is_newton() {
    assert_eq!(same_as_newton(1), [true; 2], "newton_chord(1) differs from newton");
    assert_eq!(same_as_newton(3), [false; 2], "control did not fire");
}

/// the reuse check: converged, with ceil(steps / every) Jacobians, fewer than its steps
fn reuses(rep: &NewtonReport, jac: usize, every: usize) -> bool {
    let steps = rep.scale.len();
    rep.converged && jac == steps.div_ceil(every) && jac < steps
}

/// Pass, written before the first run: with the factors formed every third step, Newton converges to max |F| <= 1e-28
/// within 12 steps, and forms ceil(steps / 3) Jacobians, fewer than its steps. The control: formed at every step, it
/// forms one Jacobian per step, so the count sees the reuse, and the reuse check rejects it.
#[test]
fn chord_reuses_the_factors() {
    let (rep, _, jac) = chord(3, 12);
    let steps = rep.scale.len();
    println!(
        "every 3: converged {} in {steps} steps, {jac} Jacobians, max |F| {:.3e}",
        rep.converged,
        rep.fnorm.last().unwrap()
    );
    assert!(reuses(&rep, jac, 3));
    let (rep, _, jac) = chord(1, 12);
    println!("control, every 1: {} steps, {jac} Jacobians", rep.scale.len());
    assert_eq!(jac, rep.scale.len(), "control: every step forms its Jacobian");
    assert!(!reuses(&rep, jac, 3), "control did not fire");
}

/// `every` 0 is refused.
#[test]
fn chord_refuses_every_zero() {
    let (pr, zs) = Cubic::<Double>::new(1.0);
    let mut z = start(&zs);
    assert!(newton_chord::<Double, _>(&pr, &mut z, &NewtonOpts::default(), 0).is_err());
}
