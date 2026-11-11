//! Newton's method on F(z) = 0 in any precision: the step dz = -J^-1 F with the exact Jacobian and a dense LU, and
//! a loop that halves a step which does not lower max |F|.
use crate::equations::Equations;
use crate::prec::{max_abs, parts};
use crate::shooting::Shooting;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use std::time::Instant;

/// A Newton step at z: dz = -J^-1 F (PARTS part arrays), the residual F, the LU of J, and the seconds of the
/// residual, the Jacobian, the factorization and the solve.
pub struct Step<P: Precision<Native>> {
    pub dz: Vec<Vec<f64>>,
    pub f: Vec<P::Scalar>,
    pub lu: P::Lu,
    pub t: [f64; 4],
}

/// The settings of [`Shooting::newton`]
#[derive(Clone, Copy, Debug)]
pub struct NewtonOpts {
    /// stop when max |F| is at most this
    pub tol: f64,
    /// Newton steps at most
    pub max_iter: usize,
    /// halvings of a step at most, before it is taken as it is
    pub max_halvings: usize,
    pub par: bool,
}

impl Default for NewtonOpts {
    fn default() -> Self {
        NewtonOpts {
            tol: 1e-28,
            max_iter: 20,
            max_halvings: 10,
            par: true,
        }
    }
}

/// What [`Shooting::newton`] did: max |F| before each step and at the end, the scale of each step taken (1 or a
/// power of 1/2), and whether it reached the tolerance.
#[derive(Clone, Debug)]
pub struct NewtonReport {
    pub fnorm: Vec<f64>,
    pub scale: Vec<f64>,
    pub converged: bool,
}

/// z + s dz in the precision, part by part (s a power of two)
fn axpy<P: Precision<Native>>(z: &[&[f64]], dz: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    let l = z[0].len();
    let mut out: Vec<Vec<f64>> = (0..P::PARTS).map(|_| vec![0.0; l]).collect();
    let sc = P::from_dd(multidouble::dd::DD::from_f64(s));
    let mut a = vec![0.0; P::PARTS];
    let mut d = vec![0.0; P::PARTS];
    for i in 0..l {
        for q in 0..P::PARTS {
            a[q] = z[q][i];
            d[q] = dz[q][i];
        }
        let v = P::from_raw(&a) + sc * P::from_raw(&d);
        for (q, o) in out.iter_mut().enumerate() {
            o[i] = P::part(v, q);
        }
    }
    out
}

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// The Newton step at z.
    pub fn newton_step<P: Precision<Native>>(&self, z: &[&[f64]], par: bool) -> Result<Step<P>, String> {
        let l = self.len();
        let mut t = [0.0f64; 4];
        let t0 = Instant::now();
        let f = self.residual::<P>(z, par)?;
        t[0] = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let (c, ld) = self.jacobian::<P>(z, par)?;
        t[1] = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let lu = P::lu_factor(l, ld, c, par)?;
        t[2] = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let mut b: Vec<P::Scalar> = f.iter().map(|&v| -v).collect();
        P::lu_solve(&lu, &mut b);
        t[3] = t0.elapsed().as_secs_f64();
        let dz = (0..P::PARTS)
            .map(|q| b.iter().map(|&v| P::part(v, q)).collect())
            .collect();
        Ok(Step { dz, f, lu, t })
    }

    /// Newton's method from z (PARTS part arrays, updated in place): steps until max |F| <= tol, each halved while it
    /// does not lower max |F|, at most `max_halvings` times. A residual that is not finite (NaN or infinite) is never
    /// converged: at the start it stops the method, and a trial with one is halved, and stops the method if the last
    /// halving still has one.
    pub fn newton<P: Precision<Native>>(
        &self,
        z: &mut Vec<Vec<f64>>,
        opts: &NewtonOpts,
    ) -> Result<NewtonReport, String> {
        let mut rep = NewtonReport {
            fnorm: Vec::new(),
            scale: Vec::new(),
            converged: false,
        };
        let mut fn0 = max_abs::<P>(&self.residual::<P>(&parts(z), opts.par)?);
        for _ in 0..opts.max_iter {
            rep.fnorm.push(fn0);
            if !fn0.is_finite() {
                return Ok(rep);
            }
            if fn0 <= opts.tol {
                rep.converged = true;
                return Ok(rep);
            }
            let step = self.newton_step::<P>(&parts(z), opts.par)?;
            let mut s = 1.0;
            let (mut zn, mut fnn);
            let mut h = 0;
            loop {
                zn = axpy::<P>(&parts(z), &step.dz, s);
                fnn = max_abs::<P>(&self.residual::<P>(&parts(&zn), opts.par)?);
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
            fn0 = fnn;
        }
        rep.fnorm.push(fn0);
        rep.converged = fn0 <= opts.tol;
        Ok(rep)
    }
}
