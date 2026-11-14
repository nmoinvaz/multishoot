//! The asymptotic numerical method (ANM): the Taylor series z(h) = sum_k z_k h^k of a solution branch of
//! F(z, p0 + h) = 0 in the problem's parameter, from a solution z_0 at p0 (Cochelin, Comput. Struct. 53 (1994) 1181;
//! Cochelin, Damil and Potier-Ferry, Int. J. Numer. Methods Eng. 37 (1994) 1187).
//!
//! Order k of F(z(h), p0 + h) is J z_k plus what the lower orders give, with J the Jacobian at z_0, so each
//! coefficient solves J z_k = -F_k, where F_k is order k of the residual's series with z_k left out
//! ([`ResidualJet`]). One factorization of J serves every order. With `refine` > 0 each solve is refined by the exact
//! products J x ([`Jvp`]): the residual b - J x, solved with the same factors, corrects x.
//!
//! The series is valid up to about [`Branch::radius`], Cochelin's estimate (delta |z_1| / |z_N|)^(1 / (N - 1)), where
//! the last term would be delta times the first.
use super::newton::exact;
use super::{Jvp, ResidualJet};
use crate::prec::{max_abs, parts};
use multidouble::dd::DD;
use multidouble::precision::Precision;
use multidouble::simd::Native;

/// The Taylor coefficients z_0..z_N of a branch, and for each order k >= 1 the relative residual |J z_k + F_k| /
/// |F_k| of its solve after refinement (max norms, NaN where either holds a NaN, and NaN without refinement)
#[derive(Clone, Debug)]
pub struct Branch<P: Precision<Native>> {
    pub z: Vec<Vec<P::Scalar>>,
    pub residual: Vec<f64>,
}

/// part arrays of scalars
fn to_parts<P: Precision<Native>>(v: &[P::Scalar]) -> Vec<Vec<f64>> {
    (0..P::PARTS)
        .map(|q| v.iter().map(|&x| P::part(x, q)).collect())
        .collect()
}

/// The branch's coefficients z_1..z_order at the solution z0 (P::PARTS part arrays), the module's documentation,
/// each order's solve refined `refine` times.
pub fn series<P: Precision<Native>, Pr: ResidualJet<P> + Jvp<P> + ?Sized>(
    pr: &Pr,
    z0: &[&[f64]],
    order: usize,
    refine: usize,
    par: bool,
) -> Result<Branch<P>, String> {
    let lu = exact::<P, Pr>(pr, z0, par)?.0;
    let mut zp: Vec<Vec<Vec<f64>>> = vec![z0.iter().map(|p| p.to_vec()).collect()];
    let mut br = Branch {
        z: vec![(0..pr.len())
            .map(|i| P::from_raw(&z0.iter().map(|p| p[i]).collect::<Vec<_>>()))
            .collect()],
        residual: Vec::new(),
    };
    for k in 1..=order {
        let refs: Vec<Vec<&[f64]>> = zp.iter().map(|z| parts(z)).collect();
        let f = pr.residual_jet(&refs, k + 1, par)?;
        let b: Vec<P::Scalar> = f[k].iter().map(|&v| -v).collect();
        let mut x = b.clone();
        P::lu_solve(&lu, &mut x);
        let resid = |x: &[P::Scalar]| -> Result<Vec<P::Scalar>, String> {
            let jx = pr.jvp(z0, &parts(&to_parts::<P>(x)), par)?;
            Ok(b.iter().zip(&jx).map(|(&bi, &ji)| bi - ji).collect())
        };
        for _ in 0..refine {
            let mut r = resid(&x)?;
            P::lu_solve(&lu, &mut r);
            x.iter_mut().zip(&r).for_each(|(xi, &ri)| *xi = *xi + ri);
        }
        let nb = max_abs::<P>(&b);
        br.residual.push(if refine > 0 {
            max_abs::<P>(&resid(&x)?) / if nb.is_nan() { nb } else { nb.max(f64::MIN_POSITIVE) }
        } else {
            f64::NAN
        });
        zp.push(to_parts::<P>(&x));
        br.z.push(x);
    }
    Ok(br)
}

impl<P: Precision<Native>> Branch<P> {
    /// the order N of the series
    pub fn order(&self) -> usize {
        self.z.len() - 1
    }

    /// z(a) = sum_k z_k a^k by Horner's rule in P
    pub fn eval(&self, a: f64) -> Vec<P::Scalar> {
        let a = P::from_dd(DD::from_f64(a));
        let mut out = self.z[self.order()].clone();
        for zk in self.z.iter().rev().skip(1) {
            out.iter_mut().zip(zk).for_each(|(o, &c)| *o = *o * a + c);
        }
        out
    }

    /// z(a) as P::PARTS part arrays, the start of the corrector at p0 + a
    pub fn eval_parts(&self, a: f64) -> Vec<Vec<f64>> {
        to_parts::<P>(&self.eval(a))
    }

    /// Cochelin's step (delta |z_1| / |z_N|)^(1 / (N - 1)) (max norms): where the series' last term is delta times its
    /// first. Infinite when z_N is zero (a polynomial or constant branch), NaN when a coefficient holds a NaN; needs
    /// N >= 2.
    pub fn radius(&self, delta: f64) -> f64 {
        let n = self.order();
        assert!(n >= 2, "radius: the series needs order 2 or more, not {n}");
        let last = max_abs::<P>(&self.z[n]);
        if last == 0.0 {
            return f64::INFINITY;
        }
        (delta * max_abs::<P>(&self.z[1]) / last).powf(1.0 / (n - 1) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Double;

    /// For z_k = r^-k the radius is delta^(1 / (N - 1)) r, and z(a) is the geometric sum. The controls: a series one
    /// order shorter has another radius, and its sum at a misses by its last term.
    #[test]
    fn radius_and_eval_of_a_geometric_series() {
        let r = 0.25f64;
        let z: Vec<Vec<DD>> = (0..=6).map(|k| vec![DD::from_f64(r.powi(-k))]).collect();
        let br = Branch::<Double> { z, residual: vec![] };
        let want = 1e-8f64.powf(1.0 / 5.0) * r;
        let radius_ok = |b: &Branch<Double>| (b.radius(1e-8) / want - 1.0).abs() < 1e-14;
        let a = 0.1;
        let sum: f64 = (0..=6).map(|k| (a / r).powi(k)).sum();
        let eval_ok = |b: &Branch<Double>| (b.eval(a)[0].hi / sum - 1.0).abs() < 1e-15;
        assert!(radius_ok(&br) && eval_ok(&br), "{}", br.radius(1e-8));
        let short = Branch::<Double> {
            z: br.z[..6].to_vec(),
            residual: vec![],
        };
        assert!(!radius_ok(&short), "radius control did not fire");
        assert!(!eval_ok(&short), "eval control did not fire");
    }

    /// The series of a constant branch, z_1 = z_2 = 0, has an infinite radius, and one with a NaN in z_1 a NaN
    /// radius. The control: a last coefficient of 1e-300 gives a finite radius.
    #[test]
    fn radius_of_a_constant_branch() {
        let br = |z1: f64, z2: f64| Branch::<Double> {
            z: vec![vec![DD::ONE], vec![DD::from_f64(z1)], vec![DD::from_f64(z2)]],
            residual: vec![],
        };
        assert_eq!(br(0.0, 0.0).radius(1e-8), f64::INFINITY);
        assert!(br(f64::NAN, 1.0).radius(1e-8).is_nan());
        assert!(br(0.0, 1e-300).radius(1e-8).is_finite(), "control did not fire");
    }
}
