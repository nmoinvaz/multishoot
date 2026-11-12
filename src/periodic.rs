//! The periodic direction: collocation grids in theta, the spectral derivative d/dtheta on them, and sines and
//! cosines in double-double, and in octo-double for constants that must be exact in every precision.
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
use multidouble::od::OD;

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

/// The reduction of the angle (p / q) pi, q > 0, in integers without overflow: p modulo 2q (a period), then
/// (n, num, 2q) with n = round(2p / q), the nearest multiple of pi / 2, and the remainder (p / q) pi - n pi / 2 =
/// num pi / 2q, |num| <= q. The results are those of the unreduced p wherever its products fit in i64.
fn reduce_pi(p: i64, q: i64) -> (i64, i128, i128) {
    let (p, q) = (p as i128, q as i128);
    let p = p.rem_euclid(2 * q);
    let n = (4 * p + q).div_euclid(2 * q);
    (n as i64, 2 * p - n * q, 2 * q)
}

/// whether the integer is an f64 exactly (|x| <= 2^53)
fn f64_exact(x: i128) -> bool {
    x.unsigned_abs() <= 1 << 53
}

/// x as an exact sum of two f64s, |x| < 2^106: its nearest f64 and the integer remainder (below 2^53)
fn split_int(x: i128) -> (f64, f64) {
    let hi = x as f64;
    (hi, (x - hi as i128) as f64)
}

/// the remainder num pi / 2q in double-double: by f64 operations where num and 2q are f64s exactly, as before, and
/// otherwise from their exact double-double values
fn remainder_dd(num: i128, q2: i128) -> DD {
    if f64_exact(num) && f64_exact(q2) {
        return PI.mul_f(num as f64) / DD::from_f64(q2 as f64);
    }
    let dd = |x: i128| {
        let (hi, lo) = split_int(x);
        DD { hi, lo }
    };
    PI * dd(num) / dd(q2)
}

/// sin and cos of (p / q) pi, q > 0, with the reduction in integers: the remainder (2p - n q) pi / (2q) after the
/// nearest multiple n of pi / 2 is formed from an exact integer, and divided exactly, so the result is accurate to
/// about 2^-106 relative for every p and q
pub fn sin_cos_pi(p: i64, q: i64) -> (DD, DD) {
    assert!(q > 0, "sin_cos_pi: q = {q}");
    let (n, num, q2) = reduce_pi(p, q);
    quadrant(n, sin_cos_series(remainder_dd(num, q2)))
}

/// pi in octo-double: the parts of its greedy round-to-nearest expansion, from 150 decimal digits (checked against
/// `OD::parse` and Machin's formula in the tests)
pub const PI_OD: OD = OD {
    x: [
        std::f64::consts::PI,
        1.2246467991473532e-16,
        -2.9947698097183397e-33,
        1.1124542208633653e-49,
        5.672231979640316e-66,
        1.7449862161352486e-83,
        6.02937273224954e-100,
        1.91012354687999e-116,
    ],
};

/// sin and cos of r in octo-double by their Taylor series, |r| <= pi / 4 (+ rounding): terms to below tol (1e-136
/// for octo-double)
fn sin_cos_series_od(r: OD, tol: f64) -> (OD, OD) {
    let r2 = r * r;
    let (mut s, mut c) = (r, OD::from_f64(1.0));
    let (mut ts, mut tc) = (r, OD::from_f64(1.0));
    for k in 1..80 {
        let kf = k as f64;
        ts = -(ts * r2) / OD::from_f64((2.0 * kf) * (2.0 * kf + 1.0));
        tc = -(tc * r2) / OD::from_f64((2.0 * kf - 1.0) * (2.0 * kf));
        s = s + ts;
        c = c + tc;
        if tc.to_f64().abs() < tol && ts.to_f64().abs() < tol {
            break;
        }
    }
    (s, c)
}

/// sin and cos of (p / q) pi, q > 0, in octo-double, reduced in integers as [`sin_cos_pi`]: accurate to about 2^-424
/// relative for every p and q, so the constants made from them are exact in every precision of the solve
pub fn sin_cos_pi_od(p: i64, q: i64) -> (OD, OD) {
    assert!(q > 0, "sin_cos_pi_od: q = {q}");
    let (n, num, q2) = reduce_pi(p, q);
    let r = if f64_exact(num) && f64_exact(q2) {
        PI_OD.mul_f(num as f64) / OD::from_f64(q2 as f64)
    } else {
        let od = |x: i128| {
            let (hi, lo) = split_int(x);
            OD::new([hi, lo, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        };
        PI_OD * od(num) / od(q2)
    };
    let (s, c) = sin_cos_series_od(r, 1e-136);
    match n.rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
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
    // an entry depends on i - jj only, which runs over -(2m - 1)..m: one d_full per offset (none at 0)
    let off = 2 * m as i64 - 1;
    let tab: Vec<DD> = (-off..m as i64)
        .map(|d| if d == 0 { DD::ZERO } else { d_full(d, n) })
        .collect();
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
                    v = v + tab[(i as i64 - jj as i64 + off) as usize].mul_f(sgn);
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

    /// |a - b| in octo-double, as an f64
    fn od_diff(a: OD, b: OD) -> f64 {
        (a - b).to_f64().abs()
    }

    /// arctan(1 / n) in octo-double by its series, n >= 5: terms to below 1e-140
    fn atan_inv(n: f64) -> OD {
        let x = OD::from_f64(1.0) / OD::from_f64(n);
        let x2 = x * x;
        let (mut s, mut p) = (x, x);
        for k in 1..200 {
            p = -(p * x2);
            let t = p / OD::from_f64((2 * k + 1) as f64);
            s = s + t;
            if t.to_f64().abs() < 1e-140 {
                break;
            }
        }
        s
    }

    /// PI_OD against the parse of its 150 decimal digits (every part), and against Machin's formula
    /// pi = 16 arctan(1/5) - 4 arctan(1/239) in octo-double; pass: equal parts, and 1e-125
    #[test]
    fn pi_od_is_pi() {
        let parsed = OD::parse(
            "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940813",
        )
        .unwrap();
        assert_eq!(parsed.x, PI_OD.x);
        let machin = atan_inv(5.0).mul_f(16.0) - atan_inv(239.0).mul_f(4.0);
        let e = od_diff(machin, PI_OD);
        println!("|Machin - PI_OD| = {e:.3e}");
        assert!(e < 1e-125, "{e:e}");
    }

    /// The control of `pi_od_is_pi`: pi in quad-double (its first four parts) must fail Machin's bound.
    #[test]
    fn pi_od_control_quad_parts_fail() {
        let mut q = PI_OD;
        q.x[4..].iter_mut().for_each(|v| *v = 0.0);
        let machin = atan_inv(5.0).mul_f(16.0) - atan_inv(239.0).mul_f(4.0);
        assert!(od_diff(machin, q) > 1e-125);
    }

    /// sin_cos_pi_od against sin_cos_pi in double-double (1e-31), sin^2 + cos^2 = 1 (1e-126), and the double angle
    /// sin 2x = 2 sin x cos x against the value at 2p (1e-125), for p / q over three periods and three q
    #[test]
    fn sin_cos_pi_od_agrees() {
        let mut worst = 0.0f64;
        for q in [7i64, 16, 40] {
            for p in -3 * q..3 * q {
                let (s, c) = sin_cos_pi_od(p, q);
                let (sd, cd) = sin_cos_pi(p, q);
                assert!(
                    (s.to_dd() - sd).hi.abs() < 1e-31 && (c.to_dd() - cd).hi.abs() < 1e-31,
                    "{p}/{q}"
                );
                let one = od_diff(s * s + c * c, OD::from_f64(1.0));
                let (s2, _) = sin_cos_pi_od(2 * p, q);
                let dbl = od_diff((s * c).mul_f(2.0), s2);
                worst = worst.max(one).max(dbl);
                assert!(one < 1e-126 && dbl < 1e-125, "{p}/{q}: {one:e} {dbl:e}");
            }
        }
        println!("sin_cos_pi_od: worst identity error {worst:.3e}");
    }

    /// The reduction modulo 2q: pass, sin and cos of (p / q) pi in octo-double and double-double equal those of the
    /// reduced p bit for bit for p near i64::MAX and i64::MIN and q up to 10^15, where 4p + q would overflow.
    #[test]
    fn sin_cos_pi_reduces_large_p() {
        for q in [7i64, 40, 1_000_000_000_000_000] {
            for p in [i64::MAX, i64::MAX - 5, i64::MIN, i64::MIN + 3] {
                let pr = (p as i128).rem_euclid(2 * q as i128) as i64;
                assert_eq!(sin_cos_pi_od(p, q), sin_cos_pi_od(pr, q), "{p}/{q}");
                assert_eq!(sin_cos_pi(p, q), sin_cos_pi(pr, q), "{p}/{q}");
            }
        }
    }

    /// The control of `sin_cos_pi_reduces_large_p`: the angle half a period on (p + q) must differ, its sine and
    /// cosine negated, in octo-double and in double-double.
    #[test]
    fn sin_cos_pi_reduction_control_half_period_differs() {
        let (p, q) = (i64::MAX - 5, 40);
        let pr = (p as i128).rem_euclid(2 * q as i128) as i64;
        let (s, c) = sin_cos_pi_od(pr + q, q);
        assert_ne!(sin_cos_pi_od(p, q), (s, c));
        assert_eq!(sin_cos_pi_od(p, q), (-s, -c));
        let (s, c) = sin_cos_pi(pr + q, q);
        assert_ne!(sin_cos_pi(p, q), (s, c));
        assert_eq!(sin_cos_pi(p, q), (-s, -c));
    }

    /// sin and cos of x, |x| <= 2^-50, in octo-double by their Taylor series to x^21 (the next term is below 2^-1100
    /// relative), formed here from x alone
    fn small_angle(x: OD) -> (OD, OD) {
        let (mut s, mut c) = (OD::from_f64(0.0), OD::from_f64(0.0));
        let mut t = OD::from_f64(1.0);
        for k in 0..22 {
            if k % 2 == 0 {
                c = c + if k % 4 == 0 { t } else { -t };
            } else {
                s = s + if k % 4 == 1 { t } else { -t };
            }
            t = t * x / OD::from_f64((k + 1) as f64);
        }
        (s, c)
    }

    /// the denominators past f64's integers: 2^53 + 1 (2q is not an f64), 2^62 (2q = 2^63 overflows i64) and
    /// i64::MAX, with p = 1, 3 and -5
    const LARGE_Q: [i64; 3] = [(1 << 53) + 1, 1 << 62, i64::MAX];

    /// the largest error of a (sin, cos) evaluation of p pi / q against `small_angle`, relative, for each large q
    fn large_q_error(f: impl Fn(i64, i64) -> (OD, OD)) -> [f64; 3] {
        LARGE_Q.map(|q| {
            let mut worst = 0.0f64;
            for p in [1i64, 3, -5] {
                let qd = OD::new([
                    q as f64,
                    (q as i128 - (q as f64) as i128) as f64,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]);
                let x = PI_OD.mul_f(p as f64) / qd;
                let (s0, c0) = small_angle(x);
                let (s, c) = f(p, q);
                worst = worst.max((od_diff(s, s0) / s0.to_f64().abs()).max(od_diff(c, c0)));
            }
            worst
        })
    }

    /// Large denominators: pass, sin_cos_pi_od within 1e-125 and sin_cos_pi within 1e-31 of the small-angle series
    /// at p / q for q = 2^53 + 1, 2^62 and i64::MAX (relative for sin).
    #[test]
    fn sin_cos_pi_takes_large_denominators() {
        let eo = large_q_error(sin_cos_pi_od);
        let ed = large_q_error(|p, q| {
            let (s, c) = sin_cos_pi(p, q);
            (OD::from_dd(s), OD::from_dd(c))
        });
        println!("large q: od {eo:?}, dd {ed:?}");
        assert!(
            eo.iter().all(|&e| e < 1e-125) && ed.iter().all(|&e| e < 1e-31),
            "{eo:?} {ed:?}"
        );
    }

    /// The controls of `sin_cos_pi_takes_large_denominators`: the former remainder, num and 2q rounded to f64 (and 2q
    /// formed in i64, wrapping where it overflows), must fail both bounds at every q: by the rounding of 2q at
    /// 2^53 + 1, by the overflow at 2^62 and i64::MAX.
    #[test]
    fn sin_cos_pi_large_q_control_former_remainder_fails() {
        let former = |p: i64, q: i64| {
            let (n, num, _) = reduce_pi(p, q);
            let r = PI_OD.mul_f(num as f64) / OD::from_f64(q.wrapping_mul(2) as f64);
            let (s, c) = sin_cos_series_od(r, 1e-136);
            match n.rem_euclid(4) {
                0 => (s, c),
                1 => (c, -s),
                2 => (-s, -c),
                _ => (-c, s),
            }
        };
        let eo = large_q_error(former);
        let ed = large_q_error(|p, q| {
            let (n, num, _) = reduce_pi(p, q);
            let r = PI.mul_f(num as f64) / DD::from_f64(q.wrapping_mul(2) as f64);
            let (s, c) = quadrant(n, sin_cos_series(r));
            (OD::from_dd(s), OD::from_dd(c))
        });
        println!("control, former remainder: od {eo:?}, dd {ed:?}");
        assert!(
            eo.iter().all(|&e| e >= 1e-125) && ed.iter().all(|&e| e >= 1e-31),
            "control did not fire: {eo:?} {ed:?}"
        );
    }

    /// The control of `sin_cos_pi_od_agrees`: the series stopped at terms below 1e-90 must fail sin^2 + cos^2 = 1 at
    /// some angle p pi / 160.
    #[test]
    fn sin_cos_pi_od_control_short_series_fails() {
        let worst = (1..40)
            .map(|p| {
                let (s, c) = sin_cos_series_od(PI_OD.mul_f(p as f64) / OD::from_f64(160.0), 1e-90);
                od_diff(s * s + c * c, OD::from_f64(1.0))
            })
            .fold(0.0f64, f64::max);
        println!("control, short series: sin^2 + cos^2 - 1 up to {worst:.3e}");
        assert!(worst > 1e-126, "{worst:e}");
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

    /// entry (i, j) of the folded matrix as the direct sum of d_full over its images, j's and j + M's
    fn direct(parity: Parity, m: usize, i: usize, j: usize) -> DD {
        let n = period_points(parity, m);
        let images: &[(usize, f64)] = match parity {
            Parity::Full => &[(j, 1.0)],
            Parity::Odd => &[(j, 1.0), (j + m, -1.0)],
            Parity::Even => &[(j, 1.0), (j + m, 1.0)],
        };
        images
            .iter()
            .filter(|&&(jj, _)| jj != i)
            .fold(DD::ZERO, |v, &(jj, sgn)| v + d_full(i as i64 - jj as i64, n).mul_f(sgn))
    }

    /// the number of entries of dmat::<B> whose hi or lo bits differ from `direct` at (i, j), or at (j, i) when
    /// `transposed`, over the three grids
    fn differing<const B: usize>(transposed: bool) -> usize {
        let m = 8 * B;
        let mut n = 0;
        for parity in [Parity::Odd, Parity::Even, Parity::Full] {
            let d = dmat::<B>(parity);
            for i in 0..m {
                for j in 0..m {
                    let (a, b) = (
                        d.cols[j].get(i),
                        if transposed {
                            direct(parity, m, j, i)
                        } else {
                            direct(parity, m, i, j)
                        },
                    );
                    n += usize::from(a.hi.to_bits() != b.hi.to_bits() || a.lo.to_bits() != b.lo.to_bits());
                }
            }
        }
        n
    }

    /// dmat forms each offset's d_full once; pass: every entry equals the direct sum over its images, hi and lo bits,
    /// on the three grids at M = 8, 16 and 32
    #[test]
    fn dmat_equals_direct_sums() {
        assert_eq!(
            (differing::<1>(false), differing::<2>(false), differing::<4>(false)),
            (0, 0, 0)
        );
    }

    /// The control of `dmat_equals_direct_sums`: against the direct sums of the transposed matrix the entries must
    /// differ at every M.
    #[test]
    fn dmat_control_transposed_differs() {
        assert!(differing::<1>(true) > 0 && differing::<2>(true) > 0 && differing::<4>(true) > 0);
    }
}
