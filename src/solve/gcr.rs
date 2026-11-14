//! A preconditioned Krylov solver on exact matrix-vector products, and the Newton method whose steps it refines.
//!
//! [`gcr`] solves A x = b with GMRES on A M^-1 (right preconditioning; Saad and Schultz, SIAM J. Sci. Stat. Comput. 7
//! (1986) 856) in its GCR form (Eisenstat, Elman and Schultz, SIAM J. Numer. Anal. 20 (1983) 345), where the products
//! A u are exact in the precision P and M is an approximation of A that is cheap to solve with, such as A's f64 LU.
//! Each iteration takes the preconditioned residual u = M^-1 r as its direction, forms w = A u (one product),
//! orthogonalizes w against the earlier w's twice, applying the same coefficients to u, and updates x += c u and
//! r -= c w. The vectors are held in P and updated with the same coefficients, so r stays the residual b - A x to P's
//! rounding without being recomputed. In exact arithmetic the iterates are GMRES's.
//!
//! The loop stops on the change of x, not on the residual, since an error along a weak direction of A barely shows in
//! the residual: after `confirm` iterations in a row with |c u| <= step_tol |x|, or after `max_products` products.
//!
//! [`Refined`] is a Newton matrix whose solves are this loop on the exact Jacobian-vector products of a [`Jvp`]
//! problem, preconditioned by an approximate factorization, and [`newton_refined`] runs the damped Newton loops with
//! it. The steps are then exact Newton steps to the loop's tolerance, at the cost of a few products each, where the
//! approximate factorization alone converges only linearly.
use super::newton::{newton_with, norm2, Factored, NewtonOpts, NewtonReport};
use super::Jvp;
use crate::prec::parts;
use multidouble::dd::DD;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use std::cell::{Cell, RefCell};

/// How [`gcr`] forms its coefficients
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coef {
    /// in P: c = (w . r) / (w . w) and the orthogonalization's (W_j . w) / (W_j . W_j), the dot products in P's
    /// accumulator; x reaches P's accuracy
    Full,
    /// in f64, from the vectors rounded to f64, with |w| = 1 assumed. w is scaled by an f64 1 / |w|, so w . w = 1 only
    /// to about 2^-53, and each update leaves a component of about 2^-53 |r| along its direction in r, which the later
    /// directions never remove: x stalls near f64 accuracy relative to b, which a Newton step tolerates and an
    /// accurate solve does not. Fewer operations.
    F64,
}

/// The settings of [`gcr`]
#[derive(Clone, Copy, Debug)]
pub struct GcrOpts {
    /// products A u at most
    pub max_products: usize,
    /// a change |c u| <= step_tol |x| counts towards convergence (1e-28 asks for double-double accuracy; inside a
    /// Newton iteration about 1e-10 saves products)
    pub step_tol: f64,
    /// changes in a row below step_tol that end the loop
    pub confirm: usize,
    pub coef: Coef,
}

impl Default for GcrOpts {
    fn default() -> Self {
        GcrOpts {
            max_products: 30,
            step_tol: 1e-28,
            confirm: 1,
            coef: Coef::Full,
        }
    }
}

/// What [`gcr`] did: the products formed, the relative change |c u| / |x| of each iteration, and whether it converged
#[derive(Clone, Debug, Default)]
pub struct GcrInfo {
    pub products: usize,
    pub changes: Vec<f64>,
    pub converged: bool,
}

/// 1 / a in P: the f64 reciprocal refined by Newton's iteration x + x (1 - a x), each step doubling the correct bits,
/// one step per doubling from 53 bits to P's and one more
pub(crate) fn recip<P: Precision<Native>>(a: P::Scalar) -> P::Scalar {
    let mut x = P::from_dd(DD::from_f64(1.0 / P::to_dd(a).hi));
    let steps = P::PARTS.trailing_zeros() as usize + 1;
    for _ in 0..steps {
        x = x + x * (P::ONE - a * x);
    }
    x
}

/// a . b in P's accumulator
fn dot<P: Precision<Native>>(a: &[P::Scalar], b: &[P::Scalar]) -> P::Scalar {
    P::dot(a.iter().copied().zip(b.iter().copied()))
}

/// a . b in f64, from the vectors rounded to f64
fn dot64<P: Precision<Native>>(a: &[P::Scalar], b: &[P::Scalar]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| P::to_dd(x).hi * P::to_dd(y).hi).sum()
}

/// y += c x in P
fn axpy<P: Precision<Native>>(c: P::Scalar, x: &[P::Scalar], y: &mut [P::Scalar]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = *yi + c * xi;
    }
}

/// a direction of [`gcr`]: u, its image w = A u (both scaled so that |w| = 1 in f64), and 1 / (w . w)
struct Dir<P: Precision<Native>> {
    u: Vec<P::Scalar>,
    w: Vec<P::Scalar>,
    inv: P::Scalar,
}

/// x A-solved by GMRES on A M^-1 in its GCR form (the module's documentation): `apply(u)` = A u, exact in P, and
/// `precond(r)` = M^-1 r. Returns x and what the loop did. The first iteration's x is M^-1 b, the plain
/// preconditioned solve.
pub fn gcr<P: Precision<Native>>(
    mut apply: impl FnMut(&[P::Scalar]) -> Result<Vec<P::Scalar>, String>,
    precond: impl Fn(&[P::Scalar]) -> Result<Vec<P::Scalar>, String>,
    b: &[P::Scalar],
    opts: &GcrOpts,
) -> Result<(Vec<P::Scalar>, GcrInfo), String> {
    let n = b.len();
    let mut x = vec![P::ZERO; n];
    let mut r = b.to_vec();
    let mut dirs: Vec<Dir<P>> = Vec::new();
    let mut info = GcrInfo::default();
    let mut run = 0;
    while info.products < opts.max_products {
        let mut u = precond(&r)?;
        let mut w = apply(&u)?;
        info.products += 1;
        for _ in 0..2 {
            for d in &dirs {
                let beta = match opts.coef {
                    Coef::Full => dot::<P>(&d.w, &w) * d.inv,
                    Coef::F64 => P::from_dd(DD::from_f64(dot64::<P>(&d.w, &w))),
                };
                axpy::<P>(-beta, &d.w, &mut w);
                axpy::<P>(-beta, &d.u, &mut u);
            }
        }
        let nw = norm2::<P>(&w);
        if nw.is_nan() || nw <= 0.0 {
            break;
        }
        let s = P::from_dd(DD::from_f64(1.0 / nw));
        w.iter_mut().for_each(|v| *v = *v * s);
        u.iter_mut().for_each(|v| *v = *v * s);
        let (c, inv) = match opts.coef {
            Coef::Full => {
                let inv = recip::<P>(dot::<P>(&w, &w));
                (dot::<P>(&w, &r) * inv, inv)
            }
            Coef::F64 => (P::from_dd(DD::from_f64(dot64::<P>(&w, &r))), P::ONE),
        };
        axpy::<P>(c, &u, &mut x);
        axpy::<P>(-c, &w, &mut r);
        let ch = P::to_dd(c).hi.abs() * norm2::<P>(&u) / norm2::<P>(&x).max(1e-300);
        info.changes.push(ch);
        dirs.push(Dir { u, w, inv });
        run = if dirs.len() > 1 && ch <= opts.step_tol {
            run + 1
        } else {
            0
        };
        if run >= opts.confirm {
            info.converged = true;
            break;
        }
    }
    Ok((x, info))
}

/// A Newton matrix whose solves are [`gcr`] on the exact Jacobian-vector products of `pr` at z, preconditioned by an
/// approximate factorization `inner` of the Jacobian there. It counts the products of all its solves.
pub struct Refined<'a, P: Precision<Native>, Pr: Jvp<P> + ?Sized, F: Factored<P>> {
    pub pr: &'a Pr,
    pub z: Vec<Vec<f64>>,
    pub inner: F,
    pub opts: GcrOpts,
    pub par: bool,
    /// the products of all solves so far
    pub products: Cell<usize>,
    /// the last solve's report
    pub last: RefCell<GcrInfo>,
    _p: std::marker::PhantomData<P>,
}

impl<'a, P: Precision<Native>, Pr: Jvp<P> + ?Sized, F: Factored<P>> Refined<'a, P, Pr, F> {
    pub fn new(pr: &'a Pr, z: &[&[f64]], inner: F, opts: GcrOpts, par: bool) -> Self {
        Refined {
            pr,
            z: z.iter().map(|p| p.to_vec()).collect(),
            inner,
            opts,
            par,
            products: Cell::new(0),
            last: RefCell::new(GcrInfo::default()),
            _p: std::marker::PhantomData,
        }
    }
}

impl<P: Precision<Native>, Pr: Jvp<P> + ?Sized, F: Factored<P>> Factored<P> for Refined<'_, P, Pr, F> {
    /// panics where `try_solve_neg` returns an error
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        self.try_solve_neg(r).expect("Refined::solve_neg")
    }
    fn try_solve_neg(&self, r: &[P::Scalar]) -> Result<Vec<P::Scalar>, String> {
        let z = parts(&self.z);
        // x = -J^-1 r solves J x = -r, with M^-1 v = -(inner's -M^-1 v)
        let b: Vec<P::Scalar> = r.iter().map(|&v| -v).collect();
        let apply = |u: &[P::Scalar]| {
            let up: Vec<Vec<f64>> = (0..P::PARTS)
                .map(|q| u.iter().map(|&v| P::part(v, q)).collect())
                .collect();
            self.pr.jvp(&z, &parts(&up), self.par)
        };
        let precond = |v: &[P::Scalar]| -> Result<Vec<P::Scalar>, String> {
            Ok(self.inner.try_solve_neg(v)?.into_iter().map(|x| -x).collect())
        };
        let (x, info) = gcr::<P>(apply, precond, &b, &self.opts)?;
        self.products.set(self.products.get() + info.products);
        *self.last.borrow_mut() = info;
        Ok(x)
    }
}

/// Newton's method, damped as [`crate::newton::newton`], with each step's solves refined by [`gcr`] on the exact
/// Jacobian-vector products of `pr` and preconditioned by the factors `factor` forms at the step's z (an f64 LU, or
/// one reused over several steps with [`crate::newton::reuse`]). Returns the report and the products of all solves.
/// With the correction rule (`Damping::Correction`) each trial's simplified correction is a refined solve as well;
/// the residual rule (`Damping::Residual`) spends no products on its trials.
pub fn newton_refined<P: Precision<Native>, Pr: Jvp<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    gcr_opts: &GcrOpts,
    factor: impl Fn(&[&[f64]]) -> Result<F, String>,
) -> Result<(NewtonReport, usize), String> {
    let products = Cell::new(0);
    let refined = |z: &[&[f64]]| -> Result<CountOnDrop<'_, P, Pr, F>, String> {
        Ok(CountOnDrop(
            Refined::new(pr, z, factor(z)?, *gcr_opts, opts.par),
            &products,
        ))
    };
    let rep = newton_with::<P, Pr, _>(pr, z, opts, refined)?;
    Ok((rep, products.get()))
}

/// a step's `Refined` that adds its products to the run's count when the loop drops it
struct CountOnDrop<'a, P: Precision<Native>, Pr: Jvp<P> + ?Sized, F: Factored<P>>(
    Refined<'a, P, Pr, F>,
    &'a Cell<usize>,
);

impl<P: Precision<Native>, Pr: Jvp<P> + ?Sized, F: Factored<P>> Drop for CountOnDrop<'_, P, Pr, F> {
    fn drop(&mut self) {
        self.1.set(self.1.get() + self.0.products.get());
    }
}

impl<P: Precision<Native>, Pr: Jvp<P> + ?Sized, F: Factored<P>> Factored<P> for CountOnDrop<'_, P, Pr, F> {
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        self.0.solve_neg(r)
    }
    fn try_solve_neg(&self, r: &[P::Scalar]) -> Result<Vec<P::Scalar>, String> {
        self.0.try_solve_neg(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Double, Octo, Quad};

    /// 1 / 3 to each precision's accuracy: |3 x - 1| below 2^-104, 2^-208 and 2^-416. The control: the f64 start alone
    /// misses the double-double bound.
    #[test]
    fn recip_reaches_the_precision() {
        let e2 = (DD::from_f64(3.0) * recip::<Double>(DD::from_f64(3.0)) - DD::ONE)
            .hi
            .abs();
        assert!(e2 < 2f64.powi(-104), "double-double {e2:e}");
        let q = recip::<Quad>(<Quad as Precision<Native>>::from_dd(DD::from_f64(3.0)));
        let eq = <Quad as Precision<Native>>::to_dd(
            <Quad as Precision<Native>>::from_dd(DD::from_f64(3.0)) * q - <Quad as Precision<Native>>::ONE,
        )
        .hi
        .abs();
        assert!(eq < 2f64.powi(-208), "quad-double {eq:e}");
        let o = recip::<Octo>(<Octo as Precision<Native>>::from_dd(DD::from_f64(3.0)));
        let eo = <Octo as Precision<Native>>::to_dd(
            <Octo as Precision<Native>>::from_dd(DD::from_f64(3.0)) * o - <Octo as Precision<Native>>::ONE,
        )
        .hi
        .abs();
        assert!(eo < 2f64.powi(-416), "octo-double {eo:e}");
        let x0 = DD::from_f64(1.0 / 3.0);
        assert!(
            (DD::from_f64(3.0) * x0 - DD::ONE).hi.abs() > 2f64.powi(-104),
            "control did not fire"
        );
    }
}
