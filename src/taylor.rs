//! Taylor stepping in x: one step from the coefficients of [`Equations::taylor_coeffs`] summed by Horner's rule, a
//! path of steps through given points, and the planner that chooses those points so that each step's last terms stay
//! below a tolerance.
//!
//! A step from x0 to x1 sums `sum_k c_k h^k`, k = 0..=kt, with h = x1 - x0 exact in double-double (both points are
//! f64). The planner bounds the truncation error by the last two terms: with `e_k = max |c_k|` over the fields and
//! collocation points, the step is the largest h with `e_(kt-1) h^(kt-1) <= tol` and `e_kt h^kt <= tol`, capped at
//! `hmax`. For a series whose terms fall geometrically past order kt-1, the dropped tail is then below tol times the
//! ratio of consecutive terms.
use crate::equations::{Equations, State, Work};
use crate::lane::{Lane, DV};
use multidouble::dd::DD;
use multidouble::field::{horner, Fl};
use multidouble::native::V8;

/// The planner's settings: Taylor order, tolerance on the last terms (absolute), largest step.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub kt: usize,
    pub tol: f64,
    pub hmax: f64,
}

/// One Taylor step of the state s from x0 to x1 (w sized for E's fields and w.kt).
pub fn taylor_step<E: Equations<B>, T: Lane, const B: usize>(
    eq: &E,
    s: &[Fl<T, B>],
    x0: f64,
    x1: f64,
    g: &[T],
    w: &mut Work<T, B>,
) -> State<T, B> {
    for (f, sf) in s.iter().enumerate().take(E::NF) {
        w.c[f][0] = *sf;
    }
    eq.taylor_coeffs(x0, g, w);
    let h = DD::from_f64(x1) - DD::from_f64(x0);
    let kt = w.kt;
    w.c.iter().map(|cf| horner(&cf[..=kt], h)).collect()
}

/// Taylor steps through the points xs, the first the start, from the state s0.
pub fn taylor_path<E: Equations<B>, T: Lane, const B: usize>(
    eq: &E,
    s0: &[Fl<T, B>],
    xs: &[f64],
    g: &[T],
    w: &mut Work<T, B>,
) -> State<T, B> {
    let mut s = s0.to_vec();
    for i in 0..xs.len().saturating_sub(1) {
        s = taylor_step(eq, &s, xs[i], xs[i + 1], g, w);
    }
    s
}

/// whether every value of the field is finite, hi and lo parts, tested lane-wise: x - x is zero exactly when x is
/// finite (NaN and infinities give NaN)
pub fn all_finite<const B: usize>(f: &Fl<DV, B>) -> bool {
    let zero = V8::zero();
    f.0.iter()
        .all(|v| v.hi.sub(v.hi).eq(zero).bits() == 0xff && v.lo.sub(v.lo).eq(zero).bits() == 0xff)
}

/// max over the fields and collocation points of |coefficient k| (hi parts)
pub fn coef_norm<const B: usize>(w: &Work<DV, B>, k: usize) -> f64 {
    w.c.iter().map(|cf| cf[k].amax()).fold(0.0, f64::max)
}

/// 2^n, n in -1022..=1023
fn pow2(n: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&n));
    f64::from_bits(((n + 1023) as u64) << 52)
}

/// x 2^n, exact barring overflow and underflow of the result: two factors, so n may exceed one factor's range
fn scale2(x: f64, n: i32) -> f64 {
    let h = n / 2;
    x * pow2(h) * pow2(n - h)
}

/// floor(log2 a) for finite a > 0, from the exponent bits (a subnormal a scaled up by 2^64 first)
fn ilog2(a: f64) -> i32 {
    let (a, off) = if a < f64::MIN_POSITIVE {
        (a * pow2(64), -64)
    } else {
        (a, 0)
    };
    ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023 + off
}

/// t^n in double-double by binary powering
fn powi_dd(t: DD, n: u32) -> DD {
    let (mut r, mut b, mut n) = (DD::ONE, t, n);
    while n > 0 {
        if n & 1 == 1 {
            r = r * b;
        }
        b = b * b;
        n >>= 1;
    }
    r
}

/// t^n in f64 by repeated multiplication
fn powi_f64(t: f64, n: u32) -> f64 {
    (0..n).fold(1.0, |r, _| r * t)
}

/// The k-th root of a, for finite a > 0 and k >= 1, from IEEE arithmetic alone (add, subtract, multiply, divide and
/// the fused multiply-add of multidouble's double-double; no libm), so its bits are the same on every platform.
///
/// The exponent bits give a = b 2^(k q) with b in [1, 2^k), so the root is t 2^q with t = b^(1/k) in [1, 2). Twelve
/// bisections of [1, 2] in f64 bracket t, five Newton steps in double-double, t <- ((k - 1) t + b / t^(k-1)) / k,
/// refine it, and the result is rounded to f64 and moved to the neighbour whose rounding interval holds the root, by
/// the signs of (y -+ ulp / 2)^k - b in double-double. The result is the correctly rounded root, except when the root
/// lies within about k 2^-104 (relative) of a midpoint between two f64s, where it is within one ulp of it.
pub fn kth_root(a: f64, k: u32) -> f64 {
    assert!(a > 0.0 && a.is_finite() && k >= 1, "kth_root: a = {a:e}, k = {k}");
    if k == 1 {
        return a;
    }
    let q = ilog2(a).div_euclid(k as i32);
    let b = scale2(a, -(k as i32) * q);
    let (mut lo, mut hi) = (1.0f64, 2.0f64);
    for _ in 0..12 {
        let mid = 0.5 * (lo + hi);
        if powi_f64(mid, k) > b {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let bd = DD::from_f64(b);
    let kd = DD::from_f64(k as f64);
    let mut t = DD::from_f64(0.5 * (lo + hi));
    for _ in 0..5 {
        t = (t.mul_f((k - 1) as f64) + bd / powi_dd(t, k - 1)) / kd;
    }
    let mut y = t.hi;
    for _ in 0..3 {
        let (up, dn) = (y.next_up(), y.next_down());
        // (m^k - b) at the midpoints m = y + ulp / 2 and y - ulp / 2, each exact as a double-double
        let above = (powi_dd(
            DD {
                hi: y,
                lo: 0.5 * (up - y),
            },
            k,
        ) - bd)
            .hi;
        let below = (powi_dd(
            DD {
                hi: y,
                lo: -0.5 * (y - dn),
            },
            k,
        ) - bd)
            .hi;
        if above <= 0.0 {
            y = up;
        } else if below > 0.0 {
            y = dn;
        } else {
            break;
        }
    }
    scale2(y, q)
}

/// The step the planner takes from coefficients w (orders kt - 1 and kt), before the cap at the end of the segment:
/// the k-th roots of tol / e_k by [`kth_root`], the same on every platform.
pub fn step_size<const B: usize>(w: &Work<DV, B>, plan: &Plan) -> f64 {
    let kt = w.kt;
    let e1 = coef_norm(w, kt - 1).max(1e-300);
    let e2 = coef_norm(w, kt).max(1e-300);
    let root = |a: f64, k: usize| {
        if a.is_finite() {
            kth_root(a, k as u32)
        } else {
            f64::INFINITY
        }
    };
    let h1 = root(plan.tol / e1, kt - 1);
    let h2 = root(plan.tol / e2, kt);
    h1.min(h2).min(plan.hmax)
}

/// Step points from xa to xb (either direction) for the state s0, both ends included and the last exactly xb: each
/// step from [`step_size`] at the state reached so far; a step that would leave less than 40% of the rest is halved,
/// so the segment ends in two near-equal steps instead of a sliver. It panics on an end that is not finite, a plan
/// with tol or hmax not positive, a Taylor coefficient that is not finite (every order 0..=kt, both parts, as
/// Horner's rule uses them all), and a step that is not positive or does not move x (a singularity at x).
pub fn plan_path<E: Equations<B>, const B: usize>(
    eq: &E,
    s0: &[Fl<DV, B>],
    xa: f64,
    xb: f64,
    g: &[DV],
    plan: &Plan,
) -> Vec<f64> {
    assert!(
        xa.is_finite() && xb.is_finite(),
        "plan_path: ends {xa} and {xb} must be finite"
    );
    assert!(plan.kt >= 2, "plan_path: Taylor order {} < 2", plan.kt);
    assert!(
        plan.tol > 0.0 && plan.hmax > 0.0,
        "plan_path: tol {:e} and hmax {:e} must be positive",
        plan.tol,
        plan.hmax
    );
    let mut w = Work::<DV, B>::new(E::NF, E::NAUX, plan.kt);
    let mut xs = vec![xa];
    let mut x = xa;
    let mut s = s0.to_vec();
    let dir = if xb > xa { 1.0 } else { -1.0 };
    while (xb - x) * dir > 0.0 {
        for (f, sf) in s.iter().enumerate() {
            w.c[f][0] = *sf;
        }
        eq.taylor_coeffs(x, g, &mut w);
        for (f, cf) in w.c.iter().enumerate() {
            if let Some(k) = cf[..=plan.kt].iter().position(|c| !all_finite(c)) {
                panic!("plan_path: Taylor coefficient {k} of field {f} at x = {x} is not finite");
            }
        }
        let mut h = step_size(&w, plan);
        assert!(h > 0.0, "plan_path: step {h:e} at x = {x}");
        let rest = (xb - x) * dir;
        if h >= rest {
            h = rest;
        } else if h > 0.6 * rest {
            h = 0.5 * rest;
        }
        let xn = if h == rest { xb } else { x + dir * h };
        assert!(xn != x, "plan_path: step {h:e} does not move x = {x}");
        let hd = DD::from_f64(xn) - DD::from_f64(x);
        s = w.c.iter().map(|cf| horner(&cf[..=plan.kt], hd)).collect();
        x = xn;
        xs.push(x);
    }
    xs
}

#[cfg(test)]
mod tests {
    use super::*;
    use multidouble::qd::QD;

    /// the fixed sample of (a, k): mantissas 1 + j / 97, exponents from -1000 to 900, orders 1 to 60
    fn sample() -> Vec<(f64, u32)> {
        let mut v = Vec::new();
        for e in [-1000, -500, -77, -30, -1, 0, 1, 52, 300, 900] {
            for j in [0.0, 1.0, 13.0, 50.0, 96.0] {
                for k in [1u32, 2, 3, 7, 19, 29, 35, 36, 60] {
                    v.push((scale2(1.0 + j / 97.0, e), k));
                }
            }
        }
        v
    }

    /// q^k in quad-double by repeated multiplication
    fn powi_qd(q: QD, k: u32) -> QD {
        (0..k).fold(QD::from_dd(DD::ONE), |r, _| r * q)
    }

    /// whether y is the correctly rounded k-th root of a, tested in quad-double (code apart from `kth_root`'s
    /// double-double): (y - ulp_below / 2)^k <= a < (y + ulp_above / 2)^k
    fn correctly_rounded(y: f64, a: f64, k: u32) -> bool {
        let (up, dn) = (y.next_up(), y.next_down());
        let hi = QD::new(y, 0.5 * (up - y), 0.0, 0.0);
        let lo = QD::new(y, -0.5 * (y - dn), 0.0, 0.0);
        let aq = QD::from_dd(DD::from_f64(a));
        (powi_qd(lo, k) - aq).to_dd().hi <= 0.0 && (powi_qd(hi, k) - aq).to_dd().hi > 0.0
    }

    /// Pass: the root is the correctly rounded one on every sample point.
    #[test]
    fn kth_root_is_correctly_rounded() {
        for (a, k) in sample() {
            let y = kth_root(a, k);
            assert!(correctly_rounded(y, a, k), "a = {a:e}, k = {k}: {y:e}");
        }
    }

    /// The control of `kth_root_is_correctly_rounded`: its neighbours one ulp away must fail the check.
    #[test]
    fn kth_root_control_neighbours_fail() {
        for (a, k) in sample() {
            let y = kth_root(a, k);
            assert!(!correctly_rounded(y.next_up(), a, k) && !correctly_rounded(y.next_down(), a, k));
        }
    }

    /// Against libm's powf(a, 1 / k). powf cannot be held to one ulp: it raises a to fl(1 / k) = (1 / k)(1 + d),
    /// |d| <= 2^-53, and a^((1 + d) / k) = a^(1 / k) exp(d ln(a) / k), a relative error of up to 2^-53 |ln a| / k
    /// before its own rounding (about 170 ulp at a = 2^-1000, k = 2). Pass: within one ulp plus that bound, with
    /// |ln a| bounded by (|floor(log2 a)| + 1) ln 2, on every sample point.
    #[test]
    fn kth_root_agrees_with_powf() {
        let mut worst = 0.0f64;
        for (a, k) in sample() {
            let y = kth_root(a, k);
            let p = a.powf(1.0 / k as f64);
            let ulp = y.next_up() - y;
            let bound = ulp + y * (ilog2(a).abs() + 1) as f64 * std::f64::consts::LN_2 * 2f64.powi(-53) / k as f64;
            worst = worst.max((y - p).abs() / ulp);
            assert!((y - p).abs() <= bound, "a = {a:e}, k = {k}: {y:e} against powf {p:e}");
        }
        println!("largest |kth_root - powf| {worst:.1} ulp");
    }

    /// The root follows its input at the resolution a k-th root has: an input 2k ulps larger moves it by about two
    /// ulps, so it must change (one ulp of input moves it by about 1 / k ulp, and for k = 1 that must change it).
    #[test]
    fn kth_root_follows_its_input() {
        for (a, k) in sample() {
            let mut b = a;
            for _ in 0..(if k == 1 { 1 } else { 2 * k }) {
                b = b.next_up();
            }
            assert_ne!(kth_root(b, k), kth_root(a, k), "a = {a:e}, k = {k}");
        }
    }
}
