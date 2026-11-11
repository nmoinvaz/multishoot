//! The periodic direction: collocation grids in theta, the spectral derivative d/dtheta on them, and sines and
//! cosines in double-double for their constants.
//!
//! A field is held on M = 8 B points. Without symmetry ([`Parity::Full`]) the grid is theta_j = 2 pi j / M over the
//! period. A field odd under theta -> theta + pi (odd harmonics only) or even (even harmonics only) is held on the
//! half grid theta_j = pi j / M, j < M, of the 2M-point grid; the other half follows from the symmetry, and the
//! derivative matrix is the full one folded with it. Pointwise products keep the parities (odd times odd is even).
//!
//! The derivative matrix of the N-point grid (N even) is D_ij = (-1)^(i-j) cot((i-j) pi / N) / 2, i != j, and zero
//! on the diagonal: the derivative of the trigonometric interpolant, exact for harmonics below N/2 (Trefethen,
//! Spectral Methods in MATLAB, SIAM 2000, ch. 3).
use crate::lane::DMat;
use multidouble::dd::DD;

/// pi in double-double
pub const PI: DD = DD {
    hi: std::f64::consts::PI,
    lo: 1.2246467991473532e-16,
};

/// pi / 2 in double-double
const PI_2: DD = DD {
    hi: std::f64::consts::FRAC_PI_2,
    lo: 6.123233995736766e-17,
};

/// The symmetry of a field under theta -> theta + pi, which fixes its grid and derivative matrix
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parity {
    /// odd harmonics only, on the half grid
    Odd,
    /// even harmonics only, on the half grid
    Even,
    /// all harmonics, on the full grid
    Full,
}

/// sin and cos of r by their Taylor series, |r| <= pi / 4 (+ rounding): terms to below 2^-115 of the first
fn sin_cos_series(r: DD) -> (DD, DD) {
    let r2 = r * r;
    let (mut s, mut c) = (r, DD::ONE);
    let (mut ts, mut tc) = (r, DD::ONE);
    for k in 1..30 {
        let kf = k as f64;
        ts = -(ts * r2) / DD::from_f64((2.0 * kf) * (2.0 * kf + 1.0));
        tc = -(tc * r2) / DD::from_f64((2.0 * kf - 1.0) * (2.0 * kf));
        s = s + ts;
        c = c + tc;
        if tc.hi.abs() < 2e-35 && ts.hi.abs() < 2e-35 * r.hi.abs().max(1e-300) {
            break;
        }
    }
    (s, c)
}

/// (sin, cos) of r + n pi / 2 from those of r
fn quadrant(n: i64, (s, c): (DD, DD)) -> (DD, DD) {
    match n.rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// sin x and cos x in double-double: x reduced by the nearest multiple of pi / 2 (an absolute error of about
/// |x| 2^-106 from pi's rounding), then the Taylor series
pub fn sin_cos(x: DD) -> (DD, DD) {
    let n = (x.hi / std::f64::consts::FRAC_PI_2).round();
    let r = x - PI_2.mul_f(n);
    quadrant(n as i64, sin_cos_series(r))
}

/// sin and cos of (p / q) pi, q > 0, with the reduction in integers: the remainder (2p - n q) pi / (2q) after the
/// nearest multiple n of pi / 2 is formed from an exact integer, so the result is accurate to about 2^-106 relative
/// for every p
pub fn sin_cos_pi(p: i64, q: i64) -> (DD, DD) {
    assert!(q > 0, "sin_cos_pi: q = {q}");
    // n = round(2p / q), the nearest multiple of pi / 2
    let n = (4 * p + q).div_euclid(2 * q);
    let num = 2 * p - n * q;
    let r = PI.mul_f(num as f64) / DD::from_f64((2 * q) as f64);
    quadrant(n, sin_cos_series(r))
}

/// The number of grid points per period: 2M on the half grids, M on the full one
fn period_points(parity: Parity, m: usize) -> i64 {
    match parity {
        Parity::Full => m as i64,
        _ => 2 * m as i64,
    }
}

/// theta_j of the grid with m points: 2 pi j / m on the full grid, pi j / m on the half grids
pub fn theta(parity: Parity, m: usize, j: usize) -> DD {
    PI.mul_f(2.0 * j as f64) / DD::from_f64(period_points(parity, m) as f64)
}

/// sin and cos of k theta_j on the grid with m points, from the integer multiple of pi
pub fn sin_cos_grid(parity: Parity, m: usize, j: usize, k: i64) -> (DD, DD) {
    sin_cos_pi(2 * k * j as i64, period_points(parity, m))
}

/// entry (i, j) of the N-point derivative matrix, i != j mod N: (-1)^(i-j) cot((i-j) pi / N) / 2
fn d_full(d: i64, n: i64) -> DD {
    let (s, c) = sin_cos_pi(d, n);
    let v = (c / s).mul_f(0.5);
    if d.rem_euclid(2) == 0 {
        v
    } else {
        -v
    }
}

/// d/dtheta on the M = 8 B points of the grid of `parity`, in double-double: on the half grids the 2M-point matrix
/// folded with the symmetry, `D_ij = D2M_ij -+ D2M_i,j+M`
pub fn dmat<const B: usize>(parity: Parity) -> DMat<B> {
    let m = 8 * B;
    let n = period_points(parity, m);
    let mut hi = vec![0.0; m * m];
    let mut lo = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut v = DD::ZERO;
            let images: &[(usize, f64)] = match parity {
                Parity::Full => &[(j, 1.0)],
                Parity::Odd => &[(j, 1.0), (j + m, -1.0)],
                Parity::Even => &[(j, 1.0), (j + m, 1.0)],
            };
            for &(jj, sgn) in images {
                if jj != i {
                    v = v + d_full(i as i64 - jj as i64, n).mul_f(sgn);
                }
            }
            hi[i * m + j] = v.hi;
            lo[i * m + j] = v.lo;
        }
    }
    DMat::<B>::from_rows(&hi, &lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane::DV;
    use crate::prec::nan_max;
    use multidouble::field::Fl;

    /// sin_cos against sin_cos_pi at multiples of pi / 7 and against f64 at random points; pass: 1e-30 and 1e-15
    #[test]
    fn sin_cos_agree() {
        for p in -30..30 {
            let x = PI.mul_f(p as f64) / DD::from_f64(7.0);
            let (s, c) = sin_cos(x);
            let (s2, c2) = sin_cos_pi(p, 7);
            assert!((s - s2).hi.abs() < 1e-30 && (c - c2).hi.abs() < 1e-30, "p {p}");
            assert!((s.hi - x.hi.sin()).abs() < 1e-15 && (c.hi - x.hi.cos()).abs() < 1e-15);
            // sin^2 + cos^2 = 1
            assert!((s * s + c * c - DD::ONE).hi.abs() < 1e-31);
        }
    }

    /// D applied to sin(k theta) and cos(k theta), k below the Nyquist harmonic, against k cos and -k sin; pass:
    /// 1e-29 (D's entries are O(M), each rounded to 2^-106)
    #[test]
    fn dmat_differentiates_harmonics() {
        const B: usize = 2;
        let m = 8 * B;
        for (parity, ks) in [
            (Parity::Odd, vec![1i64, 3, 5, 15]),
            (Parity::Even, vec![0i64, 2, 8, 14]),
            (Parity::Full, vec![0i64, 1, 4, 7]),
        ] {
            let d = dmat::<B>(parity);
            for &k in &ks {
                let samples = |f: &dyn Fn(usize) -> DD| {
                    let v: Vec<DD> = (0..m).map(f).collect();
                    let (h, l) = multidouble::dd::split(&v);
                    Fl::<DV, B>::load(&h, &l)
                };
                let s = samples(&|j| sin_cos_grid(parity, m, j, k).0);
                let c = samples(&|j| sin_cos_grid(parity, m, j, k).1);
                let ds = d.apply(&s);
                let dc = d.apply(&c);
                for j in 0..m {
                    let kf = k as f64;
                    let e1 = (ds.get(j) - c.get(j).mul_f(kf)).hi.abs();
                    let e2 = (dc.get(j) + s.get(j).mul_f(kf)).hi.abs();
                    assert!(e1 < 1e-29 && e2 < 1e-29, "{parity:?} k {k} j {j}: {e1:e} {e2:e}");
                }
            }
        }
    }

    /// the control: a harmonic of the wrong parity is not differentiated by the folded matrix (cos 2 theta on the
    /// odd grid)
    #[test]
    fn dmat_control_wrong_parity_fails() {
        const B: usize = 1;
        let m = 8 * B;
        let d = dmat::<B>(Parity::Odd);
        let v: Vec<DD> = (0..m).map(|j| sin_cos_grid(Parity::Odd, m, j, 2).1).collect();
        let (h, l) = multidouble::dd::split(&v);
        let c = Fl::<DV, B>::load(&h, &l);
        let s: Vec<DD> = (0..m).map(|j| sin_cos_grid(Parity::Odd, m, j, 2).0).collect();
        let dc = d.apply(&c);
        let err = (0..m)
            .map(|j| (dc.get(j) + s[j].mul_f(2.0)).hi.abs())
            .fold(0.0, nan_max);
        assert!(err > 1e-3, "{err:e}");
    }
}
