//! Newton's method on F(z) = 0 in any precision, for any [`Problem`]: the step dz = -J^-1 F with the exact Jacobian
//! and a dense LU, and a damped loop.
//!
//! The default damping is affine covariant (Deuflhard, Newton Methods for Nonlinear Problems, Springer 2004, section
//! 3.3, the error-oriented global Newton method NLEQ-ERR). A trial z + lambda dz is accepted when its simplified
//! correction dz_bar = -J(z)^-1 F(z + lambda dz), formed with the old LU by one solve, passes the restricted
//! monotonicity test |dz_bar| < (1 - lambda / 4) |dz|. lambda is predicted from the last step's contraction and
//! corrected from the trial's. No scaling of the residual's rows changes these tests, so a problem whose rows differ by
//! orders of magnitude takes the steps of its unscaled copy. The residual rule, halving while max |F| does not fall,
//! is kept for comparison.
use super::Problem;
use crate::prec::{max_abs, parts};
use multidouble::dd::DD;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

/// A Newton step at z: dz = -J^-1 F (PARTS part arrays), the residual F, the LU of J, and the seconds of the
/// residual, the Jacobian, the factorization and the solve.
pub struct Step<P: Precision<Native>> {
    pub dz: Vec<Vec<f64>>,
    pub f: Vec<P::Scalar>,
    pub lu: P::Lu,
    pub t: [f64; 4],
}

/// How [`newton`] damps its steps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Damping {
    /// Deuflhard's error-oriented rule: the simplified correction must contract (the module's documentation)
    Correction,
    /// halve the step while max |F| does not fall, at most `max_halvings` times, then take it
    Residual,
}

/// The settings of [`newton`]
#[derive(Clone, Copy, Debug)]
pub struct NewtonOpts {
    /// stop when max |F| is at most this
    pub tol: f64,
    /// with the correction rule, also stop after a full step whose simplified correction has a norm at most this (0:
    /// off); unlike max |F| it does not depend on the rows' scaling
    pub tol_dz: f64,
    /// Newton steps at most
    pub max_iter: usize,
    /// the correction rule's smallest lambda is 2^-max_halvings, below which it stops, not converged; the residual
    /// rule halves at most this many times
    pub max_halvings: usize,
    /// the correction rule's first lambda (1 for a mildly nonlinear problem)
    pub lambda0: f64,
    pub damping: Damping,
    pub par: bool,
}

impl Default for NewtonOpts {
    fn default() -> Self {
        NewtonOpts {
            tol: 1e-28,
            tol_dz: 0.0,
            max_iter: 20,
            max_halvings: 10,
            lambda0: 1.0,
            damping: Damping::Correction,
            par: true,
        }
    }
}

/// What [`newton`] did: max |F| before each step and at the end, each step's lambda, the correction rule's
/// contraction theta = |dz_bar| / |dz| for each step taken (empty with the residual rule), and whether it converged.
#[derive(Clone, Debug)]
pub struct NewtonReport {
    pub fnorm: Vec<f64>,
    pub scale: Vec<f64>,
    pub theta: Vec<f64>,
    pub converged: bool,
}

impl NewtonReport {
    fn new() -> Self {
        NewtonReport {
            fnorm: Vec::new(),
            scale: Vec::new(),
            theta: Vec::new(),
            converged: false,
        }
    }
}

/// z + s dz in the precision, part by part, dz given as scalars
fn axpy<P: Precision<Native>>(z: &[&[f64]], dz: &[P::Scalar], s: f64) -> Vec<Vec<f64>> {
    let l = z[0].len();
    let mut out: Vec<Vec<f64>> = (0..P::PARTS).map(|_| vec![0.0; l]).collect();
    let sc = P::from_dd(DD::from_f64(s));
    let mut a = vec![0.0; P::PARTS];
    for i in 0..l {
        for q in 0..P::PARTS {
            a[q] = z[q][i];
        }
        let v = P::from_raw(&a) + sc * dz[i];
        for (q, o) in out.iter_mut().enumerate() {
            o[i] = P::part(v, q);
        }
    }
    out
}

/// the Euclidean norm of the scalars, from the hi parts of their double-double roundings, scaled by their largest
/// magnitude so that no square overflows; NaN when any is NaN
pub(crate) fn norm2<P: Precision<Native>>(v: &[P::Scalar]) -> f64 {
    let m = max_abs::<P>(v);
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    m * v.iter().map(|&x| P::to_dd(x).hi / m).map(|x| x * x).sum::<f64>().sqrt()
}

/// -J^-1 r with the factored J
fn solve_neg<P: Precision<Native>>(lu: &P::Lu, r: &[P::Scalar]) -> Vec<P::Scalar> {
    let mut b: Vec<P::Scalar> = r.iter().map(|&v| -v).collect();
    P::lu_solve(lu, &mut b);
    b
}

/// A factored Newton matrix, exact or approximate: the loops' corrections -J^-1 r in P
pub trait Factored<P: Precision<Native>> {
    /// -J^-1 r
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar>;
    /// -J^-1 r, or the error of a solve that can fail, such as an iterative one that evaluates the problem
    /// ([`crate::solve::gcr::Refined`]); the damped loops call this one
    fn try_solve_neg(&self, r: &[P::Scalar]) -> Result<Vec<P::Scalar>, String> {
        Ok(self.solve_neg(r))
    }
}

impl<P: Precision<Native>, F: Factored<P>> Factored<P> for Rc<F> {
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        (**self).solve_neg(r)
    }
    fn try_solve_neg(&self, r: &[P::Scalar]) -> Result<Vec<P::Scalar>, String> {
        (**self).try_solve_neg(r)
    }
}

/// The exact Jacobian's LU in P
pub struct Exact<P: Precision<Native>>(pub P::Lu);

impl<P: Precision<Native>> Factored<P> for Exact<P> {
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        solve_neg::<P>(&self.0, r)
    }
}

/// The exact Jacobian of `pr` at z, factored
pub fn exact<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &[&[f64]],
    par: bool,
) -> Result<Exact<P>, String> {
    let (c, ld) = pr.jacobian(z, par)?;
    Ok(Exact(P::lu_factor(pr.len(), ld, c, par)?))
}

/// A factorization reused for `every` steps: the returned closure calls `factor` at the first step and again every
/// `every` steps, and hands out the last factors in between. The loops of [`newton_with`] call it once per step, so
/// with it they run Shamanskii's method, the chord method for `every` at least the step count (Kelley, Iterative
/// Methods for Linear and Nonlinear Equations, SIAM 1995, section 5.4). Each reuse saves a Jacobian and a
/// factorization, and the steps within a run converge linearly, contracting by about the change of J since its
/// factorization. `every` is at least 1.
pub fn reuse<F>(
    every: usize,
    factor: impl Fn(&[&[f64]]) -> Result<F, String>,
) -> impl Fn(&[&[f64]]) -> Result<Rc<F>, String> {
    assert!(every >= 1, "reuse: every must be at least 1");
    let last: RefCell<Option<(usize, Rc<F>)>> = RefCell::new(None);
    move |z| {
        let mut last = last.borrow_mut();
        if let Some((uses, f)) = last.as_mut() {
            if *uses < every {
                *uses += 1;
                return Ok(f.clone());
            }
        }
        let f = Rc::new(factor(z)?);
        *last = Some((1, f.clone()));
        Ok(f)
    }
}

/// The Newton step at z.
pub fn newton_step<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &[&[f64]],
    par: bool,
) -> Result<Step<P>, String> {
    let l = pr.len();
    let mut t = [0.0f64; 4];
    let t0 = Instant::now();
    let f = pr.residual(z, par)?;
    t[0] = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let (c, ld) = pr.jacobian(z, par)?;
    t[1] = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let lu = P::lu_factor(l, ld, c, par)?;
    t[2] = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let b = solve_neg::<P>(&lu, &f);
    t[3] = t0.elapsed().as_secs_f64();
    let dz = (0..P::PARTS)
        .map(|q| b.iter().map(|&v| P::part(v, q)).collect())
        .collect();
    Ok(Step { dz, f, lu, t })
}

/// Newton's method from z (PARTS part arrays, updated in place), damped by `opts.damping`. A residual that is not
/// finite (NaN or infinite) is never converged: at the start of a step it stops the method; in a trial it fails
/// the correction rule's test (a NaN theta), and stops the residual rule if its last halving still has one.
pub fn newton<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
) -> Result<NewtonReport, String> {
    newton_with::<P, Pr, _>(pr, z, opts, |z| exact::<P, Pr>(pr, z, opts.par))
}

/// Newton's method with the exact Jacobian formed and factored only every `every` steps and reused in between
/// ([`reuse`]): Shamanskii's method, the chord method for `every` >= `opts.max_iter`, and `newton` bit for bit for
/// `every` = 1. Damped by `opts.damping` as `newton`, the correction rule's simplified corrections solved with the
/// step's factors.
pub fn newton_chord<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    every: usize,
) -> Result<NewtonReport, String> {
    if every == 0 {
        return Err("newton_chord: every must be at least 1".into());
    }
    newton_with::<P, Pr, _>(pr, z, opts, reuse(every, |z| exact::<P, Pr>(pr, z, opts.par)))
}

/// The damped loop of `newton` with the Newton matrix that `factor` forms and factors at each step's z: the exact
/// Jacobian for `newton`, an approximate one for a mixed-precision Newton (`Shooting::newton_mixed`).
pub fn newton_with<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    factor: impl Fn(&[&[f64]]) -> Result<F, String>,
) -> Result<NewtonReport, String> {
    if !(opts.lambda0 > 0.0 && opts.lambda0 <= 1.0) {
        return Err(format!("newton: lambda0 must lie in (0, 1], not {}", opts.lambda0));
    }
    // 0.5^1074 is the smallest positive f64; past it the minimum step length would be zero
    if opts.max_halvings > 1074 {
        return Err(format!(
            "newton: max_halvings {} is beyond f64's range (at most 1074)",
            opts.max_halvings
        ));
    }
    match opts.damping {
        Damping::Correction => newton_correction::<P, Pr, F>(pr, z, opts, factor),
        Damping::Residual => newton_residual::<P, Pr, F>(pr, z, opts, factor),
    }
}

/// The error-oriented global Newton method (Deuflhard 2004, section 3.3.3): the prediction `lambda = min(1,
/// |dz_prev| |dz_bar_prev| lambda_prev / (|dz_bar_prev - dz| |dz|))`, the trial's estimate `mu = lambda^2 |dz| / (2
/// |dz_bar - (1 - lambda) dz|)`, a trial accepted when `theta < 1 - lambda / 4` and otherwise retried at
/// `min(mu, lambda / 2)`, and one retry per step at `min(1, mu)` when that is 4 lambda or more.
fn newton_correction<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    factor: impl Fn(&[&[f64]]) -> Result<F, String>,
) -> Result<NewtonReport, String> {
    let mut rep = NewtonReport::new();
    let lam_min = 0.5f64.powi(opts.max_halvings as i32);
    let mut f = pr.residual(&parts(z), opts.par)?;
    // |dz|, dz_bar and lambda of the last step taken
    let mut prev: Option<(f64, Vec<P::Scalar>, f64)> = None;
    for _ in 0..opts.max_iter {
        let fmax = max_abs::<P>(&f);
        rep.fnorm.push(fmax);
        if !fmax.is_finite() {
            return Ok(rep);
        }
        if fmax <= opts.tol {
            rep.converged = true;
            return Ok(rep);
        }
        let lu = factor(&parts(z))?;
        let dz = lu.try_solve_neg(&f)?;
        let ndz = norm2::<P>(&dz);
        let mut lam = match &prev {
            None => opts.lambda0,
            Some((n0, bar0, lam0)) => {
                let diff: Vec<P::Scalar> = bar0.iter().zip(&dz).map(|(&a, &b)| a - b).collect();
                let mu = n0 * norm2::<P>(bar0) * lam0 / (norm2::<P>(&diff) * ndz);
                if mu.is_nan() {
                    1.0
                } else {
                    mu.min(1.0)
                }
            }
        };
        let mut raised = false;
        loop {
            if lam.is_nan() || lam < lam_min || lam == 0.0 {
                rep.fnorm.push(fmax);
                return Ok(rep);
            }
            let zt = axpy::<P>(&parts(z), &dz, lam);
            let ft = pr.residual(&parts(&zt), opts.par)?;
            let bar = lu.try_solve_neg(&ft)?;
            let nbar = norm2::<P>(&bar);
            let theta = nbar / ndz;
            let rest = P::from_dd(DD::from_f64(1.0 - lam));
            let w: Vec<P::Scalar> = bar.iter().zip(&dz).map(|(&b, &d)| b - d * rest).collect();
            let mu = 0.5 * ndz * lam * lam / norm2::<P>(&w);
            if theta.is_nan() || theta >= 1.0 - lam / 4.0 {
                lam = mu.min(lam / 2.0);
                continue;
            }
            let lam_new = mu.min(1.0);
            if !raised && lam < 1.0 && lam_new >= 4.0 * lam {
                raised = true;
                lam = lam_new;
                continue;
            }
            rep.scale.push(lam);
            rep.theta.push(theta);
            *z = zt;
            f = ft;
            if opts.tol_dz > 0.0 && lam == 1.0 && nbar <= opts.tol_dz {
                rep.fnorm.push(max_abs::<P>(&f));
                rep.converged = true;
                return Ok(rep);
            }
            prev = Some((ndz, bar, lam));
            break;
        }
    }
    let fmax = max_abs::<P>(&f);
    rep.fnorm.push(fmax);
    rep.converged = fmax <= opts.tol;
    Ok(rep)
}

/// The residual rule: steps halved while max |F| does not fall, at most `max_halvings` times, then taken.
fn newton_residual<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    factor: impl Fn(&[&[f64]]) -> Result<F, String>,
) -> Result<NewtonReport, String> {
    let mut rep = NewtonReport::new();
    // the residual at z, kept from the trial that reached z
    let mut f = pr.residual(&parts(z), opts.par)?;
    let mut fn0 = max_abs::<P>(&f);
    for _ in 0..opts.max_iter {
        rep.fnorm.push(fn0);
        if !fn0.is_finite() {
            return Ok(rep);
        }
        if fn0 <= opts.tol {
            rep.converged = true;
            return Ok(rep);
        }
        let dz = factor(&parts(z))?.try_solve_neg(&f)?;
        let mut s = 1.0;
        let (mut zn, mut ft, mut fnn);
        let mut h = 0;
        loop {
            zn = axpy::<P>(&parts(z), &dz, s);
            ft = pr.residual(&parts(&zn), opts.par)?;
            fnn = max_abs::<P>(&ft);
            if fnn < fn0 || h == opts.max_halvings {
                break;
            }
            s *= 0.5;
            h += 1;
        }
        if !fnn.is_finite() {
            rep.fnorm.push(fnn);
            return Ok(rep);
        }
        rep.scale.push(s);
        *z = zn;
        f = ft;
        fn0 = fnn;
    }
    rep.fnorm.push(fn0);
    rep.converged = fn0 <= opts.tol;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Double, DD};

    /// The norm of two entries of 1e160 is 1.414e160, finite. The control is the unscaled sum of squares, which
    /// overflows to infinity, so the scaling is what the check sees.
    #[test]
    fn norm2_does_not_overflow() {
        let v = [DD::from_f64(1e160), DD::from_f64(1e160)];
        let n = norm2::<Double>(&v);
        assert!((n / (1e160 * 2f64.sqrt()) - 1.0).abs() < 1e-15, "norm {n}");
        let naive = v.iter().map(|x| x.hi * x.hi).sum::<f64>().sqrt();
        assert!(naive.is_infinite());
    }

    /// A NaN entry makes the norm NaN, and zeros give zero.
    #[test]
    fn norm2_nan_and_zero() {
        assert!(norm2::<Double>(&[DD::from_f64(1.0), DD::from_f64(f64::NAN)]).is_nan());
        assert_eq!(norm2::<Double>(&[DD::ZERO, DD::ZERO]), 0.0);
    }
}
