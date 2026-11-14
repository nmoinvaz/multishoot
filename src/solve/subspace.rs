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
use super::newton::{newton_with, Factored, NewtonOpts, NewtonReport};
use super::Problem;
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

/// the columns made orthonormal by modified Gram-Schmidt, twice
fn orthonormalize(v: &mut [Vec<f64>]) {
    for _ in 0..2 {
        for j in 0..v.len() {
            for i in 0..j {
                let c = dot(&v[i], &v[j]);
                let (head, tail) = v.split_at_mut(j);
                tail[0].iter_mut().zip(&head[i]).for_each(|(x, y)| *x -= c * y);
            }
            let nrm = dot(&v[j], &v[j]).sqrt();
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
        let off: f64 = (0..w)
            .flat_map(|i| (0..w).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let diag: f64 = (0..w).map(|i| a[i][i] * a[i][i]).sum();
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

/// x minus its components along the orthonormal f64 vectors of `basis`, in P
pub fn remove<P: Precision<Native>>(x: &mut [P::Scalar], basis: &[Vec<f64>]) {
    for b in basis {
        let bp: Vec<P::Scalar> = b.iter().map(|&v| P::from_dd(DD::from_f64(v))).collect();
        let c = P::dot(bp.iter().copied().zip(x.iter().copied()));
        x.iter_mut().zip(&bp).for_each(|(xi, &bi)| *xi = *xi - c * bi);
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

    /// The eigenvalues of [[2, 1], [1, 2]] are 1 and 3, with eigenvectors (1, -1) and (1, 1) over sqrt 2.
    #[test]
    fn jacobi_on_a_known_matrix() {
        let (lam, q) = jacobi(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        let mut l = lam.clone();
        l.sort_by(f64::total_cmp);
        assert!((l[0] - 1.0).abs() < 1e-15 && (l[1] - 3.0).abs() < 1e-15, "{lam:?}");
        let k = if lam[0] < lam[1] { 0 } else { 1 };
        assert!((q[0][k] + q[1][k]).abs() < 1e-15, "{q:?}");
    }
}
