//! Predictors for continuation in a parameter p: the start of the corrector at the next value of p, from the
//! solutions already found. [`lagrange`] extrapolates through the last few (p_i, z_i), [`hermite`] is the cubic
//! through two solutions and their tangents dz/dp, and [`tangent`] gives dz/dp at a solution exactly, as order 1 of
//! the branch's series ([`crate::anm`]): -J^-1 dF/dp with dF/dp from the residual's series, not a difference quotient.
//!
//! The weights are formed in the precision P from the parameters' exact differences (two f64 values differ by a
//! double-double exactly), so a polynomial branch of degree below the number of points is reproduced to P's rounding.
use super::anm::series;
use super::gcr::recip;
use super::{Jvp, ResidualJet};
use multidouble::dd::DD;
use multidouble::precision::Precision;
use multidouble::simd::Native;

/// a - b in P, exact
fn diff<P: Precision<Native>>(a: f64, b: f64) -> P::Scalar {
    P::from_dd(DD::from_f64(a) - DD::from_f64(b))
}

/// sum_i w_i z_i for states held as P::PARTS part arrays, in P
fn combine<P: Precision<Native>>(w: &[P::Scalar], zs: &[&[&[f64]]]) -> Vec<Vec<f64>> {
    let n = zs[0][0].len();
    let mut buf = vec![0.0; P::PARTS];
    let out: Vec<P::Scalar> = (0..n)
        .map(|i| {
            w.iter().zip(zs).fold(P::ZERO, |t, (&wk, z)| {
                for (q, b) in buf.iter_mut().enumerate() {
                    *b = z[q][i];
                }
                t + wk * P::from_raw(&buf)
            })
        })
        .collect();
    (0..P::PARTS)
        .map(|q| out.iter().map(|&v| P::part(v, q)).collect())
        .collect()
}

/// The Lagrange polynomial through the points (p_i, z_i) evaluated at p, with z_i as P::PARTS part arrays: exact for
/// a branch that is a polynomial of degree below the number of points. The p_i must differ.
pub fn lagrange<P: Precision<Native>>(pts: &[(f64, &[&[f64]])], p: f64) -> Vec<Vec<f64>> {
    let w: Vec<P::Scalar> = pts
        .iter()
        .enumerate()
        .map(|(i, &(pi, _))| {
            pts.iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .fold(P::ONE, |t, (_, &(pj, _))| {
                    t * diff::<P>(p, pj) * recip::<P>(diff::<P>(pi, pj))
                })
        })
        .collect();
    let zs: Vec<&[&[f64]]> = pts.iter().map(|&(_, z)| z).collect();
    combine::<P>(&w, &zs)
}

/// The cubic Hermite polynomial through (p0, z0) and (p1, z1) with the tangents t0 and t1 there, evaluated at p
/// (part arrays of P): exact for a branch that is a cubic in p.
pub fn hermite<P: Precision<Native>>(
    (p0, z0, t0): (f64, &[&[f64]], &[&[f64]]),
    (p1, z1, t1): (f64, &[&[f64]], &[&[f64]]),
    p: f64,
) -> Vec<Vec<f64>> {
    let h = diff::<P>(p1, p0);
    let u = diff::<P>(p, p0) * recip::<P>(h);
    let (u2, one) = (u * u, P::ONE);
    let u3 = u2 * u;
    let (two, three) = (P::from_dd(DD::from_f64(2.0)), P::from_dd(DD::from_f64(3.0)));
    let w = [
        two * u3 - three * u2 + one,
        (u3 - two * u2 + u) * h,
        three * u2 - two * u3,
        (u3 - u2) * h,
    ];
    combine::<P>(&w, &[z0, t0, z1, t1])
}

/// dz/dp at the solution z (part arrays of P): -J^-1 dF/dp, order 1 of the branch's series ([`series`]), its solve
/// refined `refine` times by the exact products J x
pub fn tangent<P: Precision<Native>, Pr: ResidualJet<P> + Jvp<P> + ?Sized>(
    pr: &Pr,
    z: &[&[f64]],
    refine: usize,
    par: bool,
) -> Result<Vec<Vec<f64>>, String> {
    let br = series::<P, Pr>(pr, z, 1, refine, par)?;
    Ok((0..P::PARTS)
        .map(|q| br.z[1].iter().map(|&v| P::part(v, q)).collect())
        .collect())
}
