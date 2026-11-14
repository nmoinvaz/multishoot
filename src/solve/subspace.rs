//! The weakly determined directions of a problem at a point, and what uses them: the w weakest right singular
//! directions of the Jacobian by inverse subspace iteration on (J^T J)^-1 and Rayleigh-Ritz, Newton's method with its
//! steps' components along such a subspace removed, and the sensitivities y = J^-T e_k of single unknowns to the
//! residual (a change dF of the residual moves unknown k by -y . dF).
//!
//! The transpose solves use a factorization of J^T formed beside J's, since multidouble's `Precision` has no
//! transpose solve; it doubles the factorizations of these routines, not their solves.
//!
//! Removing a weak subspace from Newton's steps keeps the well-determined part of the solution converging without its
//! second-order error leaking into directions the equations fix only weakly (the reduced Newton step of Deuflhard,
//! Numer. Math. 22 (1974) 289, with the weak subspace given rather than found from the rank).
use super::gcr::recip;
use super::newton::{newton_with, Factored, NewtonOpts, NewtonReport};
use super::Problem;
use crate::prec::nan_max;
use multidouble::dd::DD;
use multidouble::precision::Precision;
use multidouble::simd::Native;

/// The factors of J and of J^T at a point
pub struct Factors<P: Precision<Native>> {
    pub n: usize,
    pub lu: P::Lu,
    pub lut: P::Lu,
}

impl<P: Precision<Native>> Factors<P> {
    /// J and J^T of `pr` at z, factored
    pub fn new<Pr: Problem<P> + ?Sized>(pr: &Pr, z: &[&[f64]], par: bool) -> Result<Self, String> {
        let n = pr.len();
        let (c, ld) = pr.jacobian(z, par)?;
        let mut ct = vec![vec![0.0; ld * n]; P::PARTS];
        for (cq, tq) in c.iter().zip(ct.iter_mut()) {
            for j in 0..n {
                for i in 0..n {
                    tq[i * ld + j] = cq[j * ld + i];
                }
            }
        }
        Ok(Factors {
            n,
            lu: P::lu_factor(n, ld, c, par)?,
            lut: P::lu_factor(n, ld, ct, par)?,
        })
    }

    /// J^-1 b
    pub fn solve(&self, b: &[P::Scalar]) -> Vec<P::Scalar> {
        let mut x = b.to_vec();
        P::lu_solve(&self.lu, &mut x);
        x
    }

    /// J^-T b
    pub fn solve_t(&self, b: &[P::Scalar]) -> Vec<P::Scalar> {
        let mut x = b.to_vec();
        P::lu_solve(&self.lut, &mut x);
        x
    }

    /// (J^T J)^-1 v for an f64 vector, rounded to f64
    fn normal(&self, v: &[f64]) -> Vec<f64> {
        let b: Vec<P::Scalar> = v.iter().map(|&x| P::from_dd(DD::from_f64(x))).collect();
        self.solve(&self.solve_t(&b)).iter().map(|&x| P::to_dd(x).hi).collect()
    }

    /// y = J^-T e_k: the sensitivity of unknown k to the residual, a change dF moving it by -y . dF
    pub fn sensitivity(&self, k: usize) -> Vec<P::Scalar> {
        let mut e = vec![P::ZERO; self.n];
        e[k] = P::ONE;
        self.solve_t(&e)
    }
}

/// a deterministic sequence in [-1, 1) from a seed (xorshift)
fn wave(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// |v|_2 with the squares taken of v / max |v|, so that they neither overflow nor underflow (NaN propagates)
fn norm(v: &[f64]) -> f64 {
    let m = v.iter().map(|x| x.abs()).fold(0.0, nan_max);
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    m * v.iter().map(|&x| (x / m) * (x / m)).sum::<f64>().sqrt()
}

/// the columns made orthonormal by modified Gram-Schmidt, twice
fn orthonormalize(v: &mut [Vec<f64>]) {
    for _ in 0..2 {
        for j in 0..v.len() {
            for i in 0..j {
                let c = dot(&v[i], &v[j]);
                let (head, tail) = v.split_at_mut(j);
                tail[0].iter_mut().zip(&head[i]).for_each(|(x, y)| *x -= c * y);
            }
            let nrm = norm(&v[j]);
            v[j].iter_mut().for_each(|x| *x /= nrm);
        }
    }
}

/// eigenvalues and eigenvectors (columns of q) of a small symmetric matrix by cyclic Jacobi rotations
fn jacobi(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let w = a.len();
    let mut q: Vec<Vec<f64>> = (0..w)
        .map(|i| (0..w).map(|j| (i == j) as u8 as f64).collect())
        .collect();
    for _ in 0..100 {
        // the off-diagonal and diagonal sums of squares of a / max |a_ij|, which neither overflow nor underflow
        let s = a.iter().flatten().fold(0.0f64, |m, &x| m.max(x.abs()));
        if s == 0.0 || !s.is_finite() {
            break;
        }
        let sq = |x: f64| (x / s) * (x / s);
        let off: f64 = (0..w)
            .flat_map(|i| (0..w).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| sq(a[i][j]))
            .sum();
        let diag: f64 = (0..w).map(|i| sq(a[i][i])).sum();
        if off <= 1e-32 * diag {
            break;
        }
        for p in 0..w {
            for r in p + 1..w {
                if a[p][r] == 0.0 {
                    continue;
                }
                let th = (a[r][r] - a[p][p]) / (2.0 * a[p][r]);
                let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
                let t = if th == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akr) = (row[p], row[r]);
                    row[p] = c * akp - s * akr;
                    row[r] = s * akp + c * akr;
                }
                let (lo, hi) = a.split_at_mut(r);
                for (x, y) in lo[p].iter_mut().zip(hi[0].iter_mut()) {
                    let (apk, ark) = (*x, *y);
                    *x = c * apk - s * ark;
                    *y = s * apk + c * ark;
                }
                for row in q.iter_mut() {
                    let (qp, qr) = (row[p], row[r]);
                    row[p] = c * qp - s * qr;
                    row[r] = s * qp + c * qr;
                }
            }
        }
    }
    ((0..w).map(|i| a[i][i]).collect(), q)
}

/// The w weakest right singular directions of J at the factors: an orthonormal basis (f64 vectors, weakest first) and
/// its singular values. Inverse subspace iteration on (J^T J)^-1 from a deterministic start (`seed`), `iters`
/// products, then Rayleigh-Ritz on the subspace: the eigenvalues lambda of V^T (J^T J)^-1 V give sigma = lambda^-1/2.
pub fn weak_subspace<P: Precision<Native>>(
    f: &Factors<P>,
    w: usize,
    iters: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut st = seed.max(1);
    let mut v: Vec<Vec<f64>> = (0..w).map(|_| (0..f.n).map(|_| wave(&mut st)).collect()).collect();
    orthonormalize(&mut v);
    for _ in 0..iters {
        v = v.iter().map(|x| f.normal(x)).collect();
        orthonormalize(&mut v);
    }
    let av: Vec<Vec<f64>> = v.iter().map(|x| f.normal(x)).collect();
    let t: Vec<Vec<f64>> = (0..w).map(|i| (0..w).map(|j| dot(&v[i], &av[j])).collect()).collect();
    let (lam, q) = jacobi(t);
    let mut order: Vec<usize> = (0..w).collect();
    order.sort_by(|&a, &b| lam[b].total_cmp(&lam[a]));
    let basis = order
        .iter()
        .map(|&k| (0..f.n).map(|i| (0..w).map(|j| v[j][i] * q[j][k]).sum()).collect())
        .collect();
    let sigma = order.iter().map(|&k| 1.0 / lam[k].sqrt()).collect();
    (basis, sigma)
}

/// x minus its component in the span of the linearly independent f64 vectors of `basis`, in P. The vectors, orthonormal
/// only to f64's rounding, are made orthogonal in P by Gram-Schmidt, and each component is divided by q . q, so x's
/// component along the span is removed to P's rounding.
pub fn remove<P: Precision<Native>>(x: &mut [P::Scalar], basis: &[Vec<f64>]) {
    let dot = |a: &[P::Scalar], b: &[P::Scalar]| P::dot(a.iter().copied().zip(b.iter().copied()));
    // (q, 1 / (q . q)) of each basis vector
    let mut qs: Vec<(Vec<P::Scalar>, P::Scalar)> = Vec::with_capacity(basis.len());
    for b in basis {
        let mut q: Vec<P::Scalar> = b.iter().map(|&v| P::from_dd(DD::from_f64(v))).collect();
        for (p, inv) in &qs {
            let c = dot(p, &q) * *inv;
            q.iter_mut().zip(p).for_each(|(qi, &pi)| *qi = *qi - c * pi);
        }
        let inv = recip::<P>(dot(&q, &q));
        qs.push((q, inv));
    }
    for (q, inv) in &qs {
        let c = dot(q, x) * *inv;
        x.iter_mut().zip(q).for_each(|(xi, &qi)| *xi = *xi - c * qi);
    }
}

/// A Newton matrix whose corrections have their components along a subspace removed
pub struct Deflated<F> {
    pub inner: F,
    pub basis: Vec<Vec<f64>>,
}

impl<P: Precision<Native>, F: Factored<P>> Factored<P> for Deflated<F> {
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        let mut x = self.inner.solve_neg(r);
        remove::<P>(&mut x, &self.basis);
        x
    }
    fn try_solve_neg(&self, r: &[P::Scalar]) -> Result<Vec<P::Scalar>, String> {
        let mut x = self.inner.try_solve_neg(r)?;
        remove::<P>(&mut x, &self.basis);
        Ok(x)
    }
}

/// Newton's method, damped as [`crate::newton::newton`], with every step's and every simplified correction's
/// components along the orthonormal `basis` removed: the unknowns never move along it.
pub fn newton_deflated<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    basis: &[Vec<f64>],
) -> Result<NewtonReport, String> {
    newton_with::<P, Pr, _>(pr, z, opts, |z| {
        Ok(Deflated {
            inner: super::newton::exact::<P, Pr>(pr, z, opts.par)?,
            basis: basis.to_vec(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the checks of a decomposition of s [[2, 1], [1, 2]]: [the eigenvalues s and 3 s to 1e-15 relative, the
    /// eigenvector of s along (1, -1) to 1e-15]
    fn jacobi_checks(lam: &[f64], q: &[Vec<f64>], s: f64) -> [bool; 2] {
        let mut l: Vec<f64> = lam.iter().map(|&x| x / s).collect();
        l.sort_by(f64::total_cmp);
        let k = if lam[0] < lam[1] { 0 } else { 1 };
        [
            (l[0] - 1.0).abs() < 1e-15 && (l[1] - 3.0).abs() < 1e-15,
            (q[0][k] + q[1][k]).abs() < 1e-15,
        ]
    }

    /// The eigenvalues of s [[2, 1], [1, 2]] are s and 3 s, with eigenvectors (1, -1) and (1, 1) over sqrt 2, at s = 1
    /// and at s = 1e200 and 1e-200, where the squares of the entries overflow and underflow. The controls: wrong
    /// eigenvalues (3 s off by 1e-12) fail the first check, the identity as eigenvectors the second.
    #[test]
    fn jacobi_on_a_known_matrix() {
        for s in [1.0, 1e200, 1e-200] {
            let (lam, q) = jacobi(vec![vec![2.0 * s, s], vec![s, 2.0 * s]]);
            assert_eq!(jacobi_checks(&lam, &q, s), [true; 2], "s {s:e}: {lam:?} {q:?}");
        }
        let (lam, q) = jacobi(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        let wrong: Vec<f64> = lam.iter().map(|&x| if x > 2.0 { x + 3e-12 } else { x }).collect();
        assert_eq!(
            jacobi_checks(&wrong, &q, 1.0),
            [false, true],
            "eigenvalue control did not fire"
        );
        let id = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            jacobi_checks(&lam, &id, 1.0),
            [true, false],
            "eigenvector control did not fire"
        );
    }

    /// Two vectors scaled by 1e200 and 1e-200, where their squares overflow and underflow, come out orthonormal to
    /// 1e-15. The control: the same check fails for the vectors before orthonormalization.
    #[test]
    fn orthonormalize_at_extreme_scales() {
        let check =
            |v: &[Vec<f64>]| (0..2).all(|i| (0..2).all(|j| (dot(&v[i], &v[j]) - (i == j) as u8 as f64).abs() < 1e-15));
        for s in [1e200, 1e-200] {
            let start = vec![vec![s, 2.0 * s, 0.0], vec![s, 0.0, 3.0 * s]];
            let mut v = start.clone();
            orthonormalize(&mut v);
            assert!(check(&v), "s {s:e}: {v:?}");
            assert!(!check(&start), "control did not fire");
        }
    }

    /// the largest component of x along the f64 vectors of `basis`, each formed in P
    fn along<P: Precision<Native>>(x: &[P::Scalar], basis: &[Vec<f64>]) -> f64 {
        basis
            .iter()
            .map(|b| {
                P::to_dd(P::dot(
                    b.iter().map(|&v| P::from_dd(DD::from_f64(v))).zip(x.iter().copied()),
                ))
                .hi
                .abs()
            })
            .fold(0.0, f64::max)
    }

    /// Pass: with the basis (1, 1, 0) / sqrt 2 and (1, -1, 1) / sqrt 3 rounded to f64, orthonormal only to about
    /// 1e-16, `remove` leaves x = 3 b_1 - 2 b_2 + (1, -1, -2) with components along both below 1e-30 in
    /// double-double, and its part (1, -1, -2) off the span to 1e-15. The control: x itself fails the first check.
    #[test]
    fn remove_is_exact_in_p_for_an_f64_basis() {
        let basis = vec![
            vec![1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt(), 0.0],
            vec![1.0 / 3f64.sqrt(), -1.0 / 3f64.sqrt(), 1.0 / 3f64.sqrt()],
        ];
        let e = [1.0, -1.0, -2.0];
        let x0: Vec<DD> = (0..3)
            .map(|i| {
                DD::from_f64(3.0) * DD::from_f64(basis[0][i]) - DD::from_f64(2.0) * DD::from_f64(basis[1][i])
                    + DD::from_f64(e[i])
            })
            .collect();
        let mut x = x0.clone();
        remove::<crate::Double>(&mut x, &basis);
        let rest = (0..3).map(|i| (x[i] - DD::from_f64(e[i])).hi.abs()).fold(0.0, f64::max);
        assert!(along::<crate::Double>(&x, &basis) <= 1e-30 && rest <= 1e-15, "{x:?}");
        assert!(along::<crate::Double>(&x0, &basis) > 1e-30, "control did not fire");
    }
}
