//! Model problems with known solutions, and helpers, shared by the integration tests.
#![allow(dead_code)]
use multishoot::lane::{DMat, Lane, DV};
use multishoot::periodic::{dmat, sin_cos, theta, Parity};
use multishoot::{Equations, Fl, State, Work, DD};

/// a double-double field from its values at the collocation points
pub fn field<const B: usize>(f: impl Fn(usize) -> DD) -> Fl<DV, B> {
    let v: Vec<DD> = (0..8 * B).map(f).collect();
    let (h, l) = multidouble::dd::split(&v);
    Fl::load(&h, &l)
}

/// max over the fields and points of |a - b| (hi part of the double-double difference)
pub fn max_diff<const B: usize>(a: &[Fl<DV, B>], b: &[Fl<DV, B>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (*x - *y).amax()).fold(0.0, f64::max)
}

/// Mutations of the model problems' recursions, the controls of the accuracy tests
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrong {
    None,
    /// the coefficient of order 10 (made at k = 9) scaled by 1 + 2^-60
    Order9,
    /// the coefficient of order 10 of u made NaN at one collocation point (3), the others unchanged
    Nan,
}

/// The linear model: two odd fields u, v on the half grid, carried in x by
///
/// ```text
/// (1 + x) u_x = (1 + x)(-omega u_theta + beta v) - a u
/// (1 + x) v_x = (1 + x)(-omega v_theta - beta u) - a v
/// ```
///
/// whose solution is `(1 + x)^-a R(beta x) (F, G)(theta - omega x)`, R the rotation by beta x: the shift in theta
/// tests d/dtheta, the rotation the coupling, and (1 + x)^-a a series of radius 1 + x0 about x0 (pole at x = -1)
/// that the planner must respect. F = sin th + 0.3 cos 3 th, G = 0.5 cos th - 0.2 sin 5 th: harmonics below the
/// half grid's limit, so the collocation solution is the samples of the exact one.
pub struct Rotor<const B: usize> {
    pub d: DMat<B>,
    pub omega: f64,
    pub beta: f64,
    pub a: f64,
    pub wrong: Wrong,
}

impl<const B: usize> Rotor<B> {
    pub fn new() -> Self {
        Rotor {
            d: dmat::<B>(Parity::Odd),
            omega: 1.3,
            beta: 0.7,
            a: 2.0,
            wrong: Wrong::None,
        }
    }

    /// the exact state (u, v) at x
    pub fn exact(&self, x: f64) -> State<DV, B> {
        let m = 8 * B;
        let xd = DD::from_f64(x);
        let q = DD::ONE / ((DD::ONE + xd) * (DD::ONE + xd));
        assert_eq!(self.a, 2.0);
        let wx = DD::from_f64(self.omega) * xd;
        let (sb, cb) = sin_cos(DD::from_f64(self.beta) * xd);
        let fg = |j: usize| -> (DD, DD) {
            let th = theta(Parity::Odd, m, j);
            let s1 = sin_cos(th - wx);
            let s3 = sin_cos(th.mul_f(3.0) - wx.mul_f(3.0));
            let s5 = sin_cos(th.mul_f(5.0) - wx.mul_f(5.0));
            let f = s1.0 + s3.1.mul_f(0.3);
            let g = s1.1.mul_f(0.5) - s5.0.mul_f(0.2);
            (f, g)
        };
        let u = field::<B>(|j| {
            let (f, g) = fg(j);
            q * (cb * f + sb * g)
        });
        let v = field::<B>(|j| {
            let (f, g) = fg(j);
            q * (cb * g - sb * f)
        });
        vec![u, v]
    }
}

impl<const B: usize> Default for Rotor<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const B: usize> Equations<B> for Rotor<B> {
    const NF: usize = 2;
    const NL: usize = 1;
    const NR: usize = 1;
    const NAUX: usize = 2;

    fn taylor_coeffs<T: Lane>(&self, x0: f64, _g: &[T], w: &mut Work<T, B>) {
        // s (k+1) u_(k+1) + k u_k = s R_k + R_(k-1) - a u_k, s = 1 + x0, R the right-hand side without the decay
        let s = DD::ONE + DD::from_f64(x0);
        let is = DD::ONE / s;
        let om = DD::from_f64(-self.omega);
        let be = DD::from_f64(self.beta);
        for k in 0..w.kt {
            let (u, v) = (w.c[0][k], w.c[1][k]);
            let ru = self.d.apply(&u).muls(om) + v.muls(be);
            let rv = self.d.apply(&v).muls(om) - u.muls(be);
            w.aux[0][k] = ru;
            w.aux[1][k] = rv;
            let (pu, pv) = if k > 0 {
                (w.aux[0][k - 1], w.aux[1][k - 1])
            } else {
                (Fl::zero(), Fl::zero())
            };
            let ka = DD::from_f64(k as f64 + self.a);
            let mut f = is / DD::from_f64((k + 1) as f64);
            if self.wrong == Wrong::Order9 && k == 9 {
                f = f + f.mul_f(2f64.powi(-60));
            }
            w.c[0][k + 1] = (ru.muls(s) + pu - u.muls(ka)).muls(f);
            if self.wrong == Wrong::Nan && k == 9 {
                let nan = field::<B>(|j| if j == 3 { DD::from_f64(f64::NAN) } else { DD::ONE });
                w.c[0][k + 1] = w.c[0][k + 1] * Fl::cst(&nan);
            }
            w.c[1][k + 1] = (rv.muls(s) + pv - v.muls(ka)).muls(f);
        }
    }

    /// u at the left end is the exact one at x; the data is v
    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], _g: &[T]) -> State<T, B> {
        vec![Fl::cst(&self.exact(x)[0]), data[0]]
    }

    /// v at the right end is the exact one at x; the data is u
    fn right_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], _g: &[T]) -> State<T, B> {
        vec![data[0], Fl::cst(&self.exact(x)[1])]
    }
}

/// FNV-1a over the bits of f64s: a fingerprint of results that must be the same on every SIMD backend
pub struct Fnv(pub u64);

impl Fnv {
    pub fn new() -> Self {
        Fnv(0xcbf29ce484222325)
    }

    pub fn f64(&mut self, x: f64) {
        for b in x.to_bits().to_le_bytes() {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }

    /// every value's hi and lo parts
    pub fn field<const B: usize>(&mut self, f: &Fl<DV, B>) {
        for j in 0..8 * B {
            let v = f.get(j);
            self.f64(v.hi);
            self.f64(v.lo);
        }
    }
}

impl Default for Fnv {
    fn default() -> Self {
        Self::new()
    }
}
