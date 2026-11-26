//! Trigonometric interpolation of a periodic field in any precision, on uniform and mapped grids.
//!
//! [`Interp`] is a linear map from a field's values on the grid of m points of its parity to m2 values. Its entries
//! are formed once in octo-double from sines and cosines whose reductions are exact, and each precision reads their
//! leading parts ([`Interp::apply`]), so a quad- or octo-double field keeps its precision. A double-double resampler
//! would leave it about 2^-106 of its size off. [`Interp::uniform`] is [`crate::periodic::Resampler`]'s map, from one
//! uniform grid to another with the harmonics past the coarser grid's Nyquist dropped. [`Interp::at`] evaluates the
//! field's interpolant at any points.
//!
//! A mapped grid is uniform in s and has its points at theta = g(s). With Boyd's periodic arctan map ([`ArctanMap`];
//! J. P. Boyd, J. Comput. Phys. 98 (1992) 181, doi:10.1016/0021-9991(92)90136-M),
//!
//! ```text
//! g(s) = s0 + atan2(L sin(s - s0), cos(s - s0)),   g'(s) = L / (cos^2(s - s0) + L^2 sin^2(s - s0)),
//! ```
//!
//! the points are densest at theta = s0 and s0 + pi (g' = L there). Since g(s + pi) = g(s) + pi, the odd and even
//! fields keep their parities in s. A field on a mapped grid is a trigonometric polynomial in s, so moving it to
//! another grid evaluates its interpolant in s at the targets' s = g^-1(theta). d/dtheta there is diag(1 / g'(s_i))
//! times d/ds ([`dmat_mapped`]).
use crate::periodic::{dmat_od, sin_cos_pi_od, DMatC, Parity, PI_OD};
use multidouble::od::OD;
use multidouble::precision::Precision;
use multidouble::simd::Native;

/// the number of grid points per period: 2m on the half grids, m on the full one
fn period_points(parity: Parity, m: usize) -> i64 {
    match parity {
        Parity::Full => m as i64,
        _ => 2 * m as i64,
    }
}

/// whether the harmonic k belongs to the parity
fn allowed(parity: Parity, k: i64) -> bool {
    match parity {
        Parity::Full => true,
        Parity::Odd => k % 2 == 1,
        Parity::Even => k % 2 == 0,
    }
}

/// atan2(y, x) in octo-double: f64's atan2, then Newton's method on y cos t - x sin t = 0 with octo-double sines and
/// cosines, each step doubling the bits (53, 106, 212, 424) and one more
pub fn atan2_od(y: OD, x: OD) -> OD {
    let mut t = OD::from_f64(y.to_f64().atan2(x.to_f64()));
    for _ in 0..4 {
        let (s, c) = t.sin_cos();
        t = t + (y * c - x * s) / (x * c + y * s);
    }
    t
}

/// Boyd's periodic arctan map theta = g(s) of a grid uniform in s (the module's documentation): L = 1 is the identity,
/// L > 1 crowds the points about theta = s0 and s0 + pi
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArctanMap {
    pub s0: f64,
    pub l: f64,
}

impl ArctanMap {
    /// (g(s), g'(s)) in octo-double, written as the identity plus deviations with the factor L - 1, so that L = 1 gives
    /// s and 1 exactly: g - s = atan2((L - 1) sin u cos u, cos^2 u + L sin^2 u), the angle of the rotation from u to
    /// v with tan v = L tan u, and g' - 1 = (L - 1)(1 - (1 + L) sin^2 u) / (cos^2 u + L^2 sin^2 u), u = s - s0
    pub fn g(&self, s: OD) -> (OD, OD) {
        let (l, one) = (OD::from_f64(self.l), OD::ONE);
        let lm = l - one;
        if lm.to_f64() == 0.0 {
            return (s, one);
        }
        let (su, cu) = (s - OD::from_f64(self.s0)).sin_cos();
        let (s2, c2) = (su * su, cu * cu);
        let dg = atan2_od(lm * su * cu, c2 + l * s2);
        let dgp = lm * (one - (one + l) * s2) / (c2 + l * l * s2);
        (s + dg, one + dgp)
    }

    /// s with g(s) = theta, on the branch continuous with s = theta: s0 + v + atan2((1 - L) sin v cos v, L cos^2 v +
    /// sin^2 v), v = theta - s0
    pub fn inverse(&self, theta: OD) -> OD {
        let (l, one) = (OD::from_f64(self.l), OD::ONE);
        if (l - one).to_f64() == 0.0 {
            return theta;
        }
        let s0 = OD::from_f64(self.s0);
        let v = theta - s0;
        let (sv, cv) = v.sin_cos();
        s0 + v + atan2_od((one - l) * sv * cv, l * cv * cv + sv * sv)
    }
}

/// s_j of the grid of m points uniform in s: 2 pi j / m on the full grid, pi j / m on the half grids, in octo-double
pub fn grid_s(parity: Parity, m: usize, j: usize) -> OD {
    PI_OD.mul_f(2.0 * j as f64) / OD::from_f64(period_points(parity, m) as f64)
}

/// the points theta_j = g(s_j) of the grid of m points with the map (None: the uniform grid), in octo-double
pub fn grid_theta(parity: Parity, m: usize, map: Option<&ArctanMap>) -> Vec<OD> {
    (0..m)
        .map(|j| {
            let s = grid_s(parity, m, j);
            map.map_or(s, |g| g.g(s).0)
        })
        .collect()
}

/// d/dtheta on the M = 8 B points of the mapped grid of `parity`: row i of [`dmat_od`] (d/ds) divided by g'(s_i) in
/// octo-double, held as [`DMatC`]. The identity map gives [`crate::periodic::dmat_c`] bit for bit.
pub fn dmat_mapped<const B: usize>(parity: Parity, map: &ArctanMap) -> DMatC<B> {
    let m = 8 * B;
    let mut a = dmat_od::<B>(parity);
    for i in 0..m {
        let gp = map.g(grid_s(parity, m, i)).1;
        for v in &mut a[i * m..(i + 1) * m] {
            *v = *v / gp;
        }
    }
    DMatC::from_od(&a)
}

/// x as P: its leading min(8, P::PARTS) parts (the parts of an octo-double do not overlap)
fn to_p<P: Precision<Native>>(x: OD) -> P::Scalar {
    P::new(&x.x[..P::PARTS.min(8)])
}

/// below this many grid steps from a grid point a target counts as on it, and gets the unit row
const ON_POINT: f64 = 1e-60;

/// A linear map from a field's m values on its grid to m2 values, its entries in octo-double (the module's
/// documentation)
#[derive(Clone, Debug)]
pub struct Interp {
    m: usize,
    m2: usize,
    /// row-major (m2, m)
    w: Vec<OD>,
}

impl Interp {
    /// the number of values taken and given
    pub fn shape(&self) -> (usize, usize) {
        (self.m, self.m2)
    }

    /// entry (i, j), the weight of source point j in target i
    pub fn entry(&self, i: usize, j: usize) -> OD {
        self.w[i * self.m + j]
    }

    /// The map of [`crate::periodic::Resampler::new`] as a matrix: from the uniform grid of m points of `parity` to
    /// that of m2, through the harmonics up to the coarser grid's Nyquist (at the finer grid's Nyquist the cosine
    /// part alone, at the coarser one's the cosine counted once when it is the source's), each entry the sum over the
    /// harmonics in octo-double of sines and cosines from [`sin_cos_pi_od`].
    pub fn uniform(parity: Parity, m: usize, m2: usize) -> Interp {
        let (n, n2) = (period_points(parity, m), period_points(parity, m2));
        let kcut = n.min(n2) / 2;
        // per harmonic: (k, cosine weight, sine weight), the analysis factor 1 / N (2 / N on a half grid) applied below
        let ks: Vec<(i64, f64, f64)> = (0..=kcut)
            .filter(|&k| allowed(parity, k))
            .map(|k| {
                let (wc, ws) = if k == 0 || 2 * k == n {
                    (1.0, 0.0)
                } else if 2 * k == n2 {
                    (2.0, 0.0)
                } else {
                    (2.0, 2.0)
                };
                (k, wc, ws)
            })
            .collect();
        let fac = match parity {
            Parity::Full => OD::ONE,
            _ => OD::from_f64(2.0),
        } / OD::from_f64(n as f64);
        let tab = |mm: usize, nn: i64| -> Vec<Vec<(OD, OD)>> {
            ks.iter()
                .map(|&(k, ..)| (0..mm).map(|j| sin_cos_pi_od(2 * k * j as i64, nn)).collect())
                .collect()
        };
        let (old, new) = (tab(m, n), tab(m2, n2));
        let mut w = vec![OD::ZERO; m2 * m];
        for i in 0..m2 {
            for j in 0..m {
                let mut t = OD::ZERO;
                for (q, &(_, wc, ws)) in ks.iter().enumerate() {
                    let ((sn, cn), (so, co)) = (new[q][i], old[q][j]);
                    t = t + (cn * co).mul_f(wc);
                    if ws != 0.0 {
                        t = t + (sn * so).mul_f(ws);
                    }
                }
                w[i * m + j] = t * fac;
            }
        }
        Interp { m, m2, w }
    }

    /// The interpolant of a field on the grid of m points of `parity`, uniform in s, evaluated at the points s (in
    /// that grid's coordinate), its entries in octo-double from the interpolant's kernel. On the half grids (m even),
    /// with x = s - s_j and (-1)^j sin(M s) = sin(M x): the odd fold sin(M x) / (M sin x) and the even one
    /// sin(M x) cot(x) / M. On the full grid of N points: sin(N x / 2) cot(x / 2) / N for even N (the Nyquist cosine
    /// counted once) and sin(N x / 2) / (N sin(x / 2)) for odd N. A target within 1e-60 grid steps of a grid point
    /// gets the unit row exactly (with the odd field's sign past pi), so the grid's own points give its values bit for
    /// bit.
    pub fn at(parity: Parity, m: usize, s: &[OD]) -> Interp {
        assert!(
            parity == Parity::Full || m.is_multiple_of(2),
            "Interp::at: the half grids' kernels need an even m, not {m}"
        );
        let n = period_points(parity, m);
        let step = PI_OD.mul_f(2.0) / OD::from_f64(n as f64);
        let mut w = vec![OD::ZERO; s.len() * m];
        for (i, &si) in s.iter().enumerate() {
            let row = &mut w[i * m..(i + 1) * m];
            // s in grid steps, and the grid point nearest it
            let q = si / step;
            let near = q.to_f64().round();
            if (q - OD::from_f64(near)).to_f64().abs() < ON_POINT {
                let p = near as i64;
                let j = p.rem_euclid(m as i64) as usize;
                // the wraps by the m points of a half period flip an odd field's sign
                let wraps = p.div_euclid(m as i64);
                row[j] = if parity == Parity::Odd && wraps % 2 != 0 {
                    -OD::ONE
                } else {
                    OD::ONE
                };
                continue;
            }
            for (j, r) in row.iter_mut().enumerate() {
                let x = si - step.mul_f(j as f64);
                *r = match parity {
                    Parity::Odd => {
                        let (sx, _) = x.sin_cos();
                        (x.mul_f(m as f64)).sin() / (sx.mul_f(m as f64))
                    }
                    Parity::Even => {
                        let (sx, cx) = x.sin_cos();
                        (x.mul_f(m as f64)).sin() * cx / (sx.mul_f(m as f64))
                    }
                    Parity::Full => {
                        let h = x.mul_f(0.5);
                        let (sh, ch) = h.sin_cos();
                        let top = h.mul_f(n as f64).sin();
                        if n % 2 == 0 {
                            top * ch / sh.mul_f(n as f64)
                        } else {
                            top / sh.mul_f(n as f64)
                        }
                    }
                };
            }
        }
        Interp { m, m2: s.len(), w }
    }

    /// The map in P: value i is the sum over j of entry (i, j), read in P's leading parts, times f_j, in P's dot
    /// product. f has m values.
    pub fn apply<P: Precision<Native>>(&self, f: &[P::Scalar]) -> Vec<P::Scalar> {
        assert_eq!(
            f.len(),
            self.m,
            "Interp::apply: {} values for {} points",
            f.len(),
            self.m
        );
        let w: Vec<P::Scalar> = self.w.iter().map(|&x| to_p::<P>(x)).collect();
        (0..self.m2)
            .map(|i| P::dot(w[i * self.m..(i + 1) * self.m].iter().copied().zip(f.iter().copied())))
            .collect()
    }
}

/// The map from the grid of m points of `parity` (with the map `from`, None: uniform) to the grid of m2 points (with
/// `to`): [`Interp::uniform`] between uniform grids, else [`Interp::at`] at the targets' points in the source's s,
/// s = g_from^-1(g_to(s_i))
pub fn interp_between(parity: Parity, m: usize, m2: usize, from: Option<&ArctanMap>, to: Option<&ArctanMap>) -> Interp {
    if from.is_none() && to.is_none() {
        return Interp::uniform(parity, m, m2);
    }
    let s: Vec<OD> = grid_theta(parity, m2, to)
        .into_iter()
        .map(|t| from.map_or(t, |g| g.inverse(t)))
        .collect();
    Interp::at(parity, m, &s)
}
