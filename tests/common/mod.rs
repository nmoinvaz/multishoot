//! Model problems with known solutions, and helpers, shared by the integration tests.
#![allow(dead_code)]
use multishoot::lane::{DMat, Lane, DV};
use multishoot::periodic::{dmat, sin_cos, theta, Parity};
use multishoot::prec::nan_max;
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
    a.iter().zip(b).map(|(x, y)| (*x - *y).amax()).fold(0.0, nan_max)
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
    type Prep = ();

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
    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], _g: &[T], _prep: &()) -> State<T, B> {
        vec![Fl::cst(&self.exact(x)[0]), data[0]]
    }

    /// v at the right end is the exact one at x; the data is u
    fn right_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], _g: &[T], _prep: &()) -> State<T, B> {
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

/// The planted nonlinear problem: an odd field u and an even field v on the half grid, with the frequency omega a
/// global unknown,
///
/// ```text
/// u_x = -omega u_theta + u v + s_u(x)
/// v_x = u^2 - v + s_v(x)
/// ```
///
/// u given at the left end and v at the right, and an extra row omega = omega*. The sources are made from the
/// manufactured solution `u* = g A + B`, `v* = g^2 C + E`, g = 1 / (1 + x), with the grid's own d/dtheta, so the
/// samples of (u*, v*) solve the collocation system to rounding: `s = sum_n g^n S_n`, n = 0..3, whose Taylor
/// coefficients about x0 are exact, `(g^n)_k = (-1)^k C(n + k - 1, k) (1 + x0)^-(n + k)`.
pub struct Planted<const B: usize> {
    pub d: DMat<B>,
    pub omega: f64,
    /// whether the points are coupled through -omega u_theta; without it the term is -omega u, pointwise
    pub coupled: bool,
    /// the manufactured solution's fields A, B (odd), C, E (even)
    pub a: Fl<DV, B>,
    pub b: Fl<DV, B>,
    pub c: Fl<DV, B>,
    pub e: Fl<DV, B>,
    /// su[n], sv[n]: the sources' fields of g^n
    pub su: Vec<Fl<DV, B>>,
    pub sv: Vec<Fl<DV, B>>,
}

impl<const B: usize> Planted<B> {
    pub fn new() -> Self {
        Self::with_coupling(true)
    }

    /// the problem with -omega u_theta (coupled) or -omega u (pointwise) in the u equation
    pub fn with_coupling(coupled: bool) -> Self {
        let m = 8 * B;
        let sc = |j: usize, k: i64| multishoot::periodic::sin_cos_grid(Parity::Odd, m, j, k);
        let a = field::<B>(|j| sc(j, 1).0 + sc(j, 3).1.mul_f(0.3));
        let b = field::<B>(|j| sc(j, 1).1.mul_f(0.2) - sc(j, 3).0.mul_f(0.1));
        let c = field::<B>(|j| DD::from_f64(0.5) + sc(j, 2).1.mul_f(0.25));
        let e = field::<B>(|j| DD::ONE + sc(j, 2).0.mul_f(0.1));
        let d = dmat::<B>(Parity::Odd);
        let omega = 1.3;
        let om = DD::from_f64(omega);
        let th = |f: &Fl<DV, B>| if coupled { d.apply(f) } else { *f };
        let su = vec![th(&b).muls(om) - b * e, th(&a).muls(om) - a * e, -a - b * c, -(a * c)];
        let sv = vec![e - b * b, -(a * b).mulf(2.0), c - a * a, -c.mulf(2.0)];
        Planted {
            d,
            omega,
            coupled,
            a,
            b,
            c,
            e,
            su,
            sv,
        }
    }

    /// the manufactured solution (u*, v*) at x
    pub fn exact(&self, x: f64) -> State<DV, B> {
        let g = DD::ONE / (DD::ONE + DD::from_f64(x));
        vec![self.a.muls(g) + self.b, self.c.muls(g * g) + self.e]
    }

    /// The shooting problem on the nodes with its extra row omega = omega*, planned from the manufactured solution
    /// (order kt, last terms below tol, steps at most 0.5), and that solution's z.
    pub fn shooting(nodes: &[f64], kt: usize, tol: f64) -> (multishoot::Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
        Self::shooting_of(Planted::<B>::new(), nodes, kt, tol)
    }

    /// `shooting` for a given problem
    pub fn shooting_of(
        eq: Planted<B>,
        nodes: &[f64],
        kt: usize,
        tol: f64,
    ) -> (multishoot::Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
        let mut sh = multishoot::Shooting::new(eq, nodes.to_vec(), kt);
        let og = sh.off_global();
        sh.extra.push(multishoot::ExtraRow {
            terms: vec![(og, DD::ONE)],
            rhs: DD::from_f64(sh.eq.omega),
        });
        let z = planted_z(&sh);
        sh.plan_segments(&z[0], &z[1], &multishoot::Plan { kt, tol, hmax: 0.5 });
        (sh, z)
    }
}

impl<const B: usize> Default for Planted<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// z of the manufactured solution (hi, lo): the left data v*(x_0), the node states, the right data u*(x_(K-1)) and
/// omega*
pub fn planted_z<const B: usize>(sh: &multishoot::Shooting<Planted<B>, B>) -> Vec<Vec<f64>> {
    let m = 8 * B;
    let l = sh.len();
    let (mut zh, mut zl) = (vec![0.0; l], vec![0.0; l]);
    let mut put = |o: usize, f: &Fl<DV, B>| f.store(&mut zh[o..o + m], &mut zl[o..o + m]);
    let k = sh.k();
    put(0, &sh.eq.exact(sh.nodes[0])[1]);
    for i in 0..k {
        let s = sh.eq.exact(sh.nodes[i]);
        put(sh.off_node(i), &s[0]);
        put(sh.off_node(i) + m, &s[1]);
    }
    put(sh.off_right(), &sh.eq.exact(sh.nodes[k - 1])[0]);
    zh[sh.off_global()] = sh.eq.omega;
    vec![zh, zl]
}

/// A deterministic sequence in [-1, 1] from integer arithmetic and one IEEE division, so the same on every platform
/// (a libm `sin` is not correctly rounded everywhere)
pub fn wave(i: usize) -> f64 {
    let n = (i as u64).wrapping_mul(2654435761).wrapping_add(12345) % 2001;
    n as f64 / 1000.0 - 1.0
}

/// z plus a perturbation of relative size eps on every unknown from `wave`, the same on every platform
pub fn perturbed(z: &[Vec<f64>], eps: f64) -> Vec<Vec<f64>> {
    let mut out = z.to_vec();
    for (i, v) in out[0].iter_mut().enumerate() {
        *v += eps * wave(i) * v.abs().max(0.1);
    }
    for p in out.iter_mut().skip(1) {
        p.iter_mut().for_each(|v| *v = 0.0);
    }
    out
}

impl<const B: usize> Equations<B> for Planted<B> {
    const NF: usize = 2;
    const NL: usize = 1;
    const NR: usize = 1;
    const NG: usize = 1;
    const NAUX: usize = 0;
    type Prep = ();

    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        let kt = w.kt;
        let sig = DD::ONE / (DD::ONE + DD::from_f64(x0));
        // gn[n][k]: Taylor coefficient k of g^n about x0
        let mut gn = vec![vec![DD::ZERO; kt + 1]; 4];
        for (n, gk) in gn.iter_mut().enumerate() {
            let mut c = DD::ONE;
            for _ in 0..n {
                c = c * sig;
            }
            gk[0] = c;
            for (k, gkk) in gk.iter_mut().enumerate().skip(1) {
                c = (c * sig).mul_f(-((n + k - 1) as f64)) / DD::from_f64(k as f64);
                *gkk = c;
            }
        }
        let om = -g[0];
        let rate = w.rate;
        for k in 0..kt {
            let mut su = Fl::<T, B>::zero();
            let mut sv = Fl::<T, B>::zero();
            for (n, gk) in gn.iter().enumerate() {
                if gk[k].hi != 0.0 {
                    su = su + Fl::cst(&self.su[n].muls(gk[k]));
                    sv = sv + Fl::cst(&self.sv[n].muls(gk[k]));
                }
            }
            let (u, v) = (&w.c[0], &w.c[1]);
            // -omega u_theta is -u_t, t = theta / omega, whose derivative a mode matrix shifts by its rate; the
            // pointwise variant's -omega u has no derivative
            let th = if self.coupled {
                -rate.shift(&u[k], self.d.apply(&u[k]).mul_t(g[0]))
            } else {
                u[k].mul_t(om)
            };
            let du = th + multishoot::cs(u, v, k) + su;
            let dv = (multishoot::cs(u, u, k) - v[k]) + sv;
            let ik = DD::ONE / DD::from_f64((k + 1) as f64);
            w.c[0][k + 1] = du.muls(ik);
            w.c[1][k + 1] = dv.muls(ik);
        }
    }

    /// u at the left end is u*; the data is v
    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], _g: &[T], _prep: &()) -> State<T, B> {
        vec![Fl::cst(&self.exact(x)[0]), data[0]]
    }

    /// v at the right end is v*; the data is u
    fn right_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], _g: &[T], _prep: &()) -> State<T, B> {
        vec![data[0], Fl::cst(&self.exact(x)[1])]
    }
}

/// A problem declared pointwise (`Equations::POINTWISE`), the methods of E unchanged: the batched Jacobian columns of
/// a problem that is pointwise, or the control of a coupled one that is not.
pub struct Pointwise<E>(pub E);

impl<E: Equations<B>, const B: usize> Equations<B> for Pointwise<E> {
    const NF: usize = E::NF;
    const NL: usize = E::NL;
    const NR: usize = E::NR;
    const NG: usize = E::NG;
    const NAUX: usize = E::NAUX;
    const POINTWISE: bool = true;
    type Prep = E::Prep;

    fn prepare(&self, zh: &[f64], zl: &[f64]) -> E::Prep {
        self.0.prepare(zh, zl)
    }
    fn prepare_rate(&self, zh: &[f64], zl: &[f64], rate: multishoot::Rate) -> E::Prep {
        self.0.prepare_rate(zh, zl, rate)
    }
    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        self.0.taylor_coeffs(x0, g, w)
    }
    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &E::Prep) -> State<T, B> {
        self.0.left_state(x, data, g, prep)
    }
    fn right_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &E::Prep) -> State<T, B> {
        self.0.right_state(x, data, g, prep)
    }
}

/// The same problem declared pointwise, with the layout (segments, extra rows) of sh
pub fn as_pointwise<const B: usize>(
    sh: multishoot::Shooting<Planted<B>, B>,
) -> multishoot::Shooting<Pointwise<Planted<B>>, B> {
    let mut p = multishoot::Shooting::new(Pointwise(sh.eq), sh.nodes, sh.kt);
    p.segs = sh.segs;
    p.extra = sh.extra;
    p.match_node = sh.match_node;
    p
}

/// The eigenproblem on a cylinder: u odd in theta on the half grid, x in [0, 1],
///
/// ```text
/// u_xx + mu u_theta_theta + lambda u + nu u_theta = 0,   u(0, theta) = u(1, theta) = 0
/// ```
///
/// as the first-order system u_x = v, v_x = -mu D^2 u - lambda u - nu D u, with two globals, lambda and nu, and two
/// extra rows on v(0): an amplitude row `sum (2/M) cos theta_j v_j(0) = 1` and a phase row `sum sin theta_j v_j(0) =
/// 0`. The lowest mode is `u = sin(pi x) cos theta / pi` with lambda = pi^2 + mu and nu = 0. Without nu and the
/// phase row its shifts in theta would be solutions too; nu unfolds that family, as a period does for a periodic orbit.
/// mu = 1 is the equations' parameter, entered through the lanes' parameter arithmetic.
pub struct Cylinder<const B: usize> {
    pub d: DMat<B>,
    pub mu: DD,
    /// the control of the jet: mu a constant instead of the parameter
    pub freeze_mu: bool,
    /// the control of the eigenvalue: lambda's coefficient scaled by 1 + 2^-50
    pub wrong_lambda: bool,
}

impl<const B: usize> Cylinder<B> {
    pub fn new() -> Self {
        Cylinder {
            d: dmat::<B>(Parity::Odd),
            mu: DD::ONE,
            freeze_mu: false,
            wrong_lambda: false,
        }
    }

    /// the eigenvalue of the lowest mode, pi^2 + mu, in double-double
    pub fn lambda(&self) -> DD {
        multishoot::periodic::PI * multishoot::periodic::PI + self.mu
    }

    /// the lowest mode (u, v) at x
    pub fn exact(&self, x: f64) -> State<DV, B> {
        let m = 8 * B;
        let px = multishoot::periodic::PI * DD::from_f64(x);
        let (s, c) = sin_cos(px);
        let ipi = DD::ONE / multishoot::periodic::PI;
        let cth = |j: usize| multishoot::periodic::sin_cos_grid(Parity::Odd, m, j, 1).1;
        vec![field::<B>(|j| s * ipi * cth(j)), field::<B>(|j| c * cth(j))]
    }

    /// The shooting problem on five nodes with its two extra rows, planned from the lowest mode (order kt, last
    /// terms below 1e-34, steps at most 0.25), and that mode's z.
    pub fn shooting(eq: Cylinder<B>, kt: usize) -> (multishoot::Shooting<Cylinder<B>, B>, Vec<Vec<f64>>) {
        let m = 8 * B;
        let mut sh = multishoot::Shooting::new(eq, vec![0.0, 0.25, 0.5, 0.75, 1.0], kt);
        let sc = |j: usize| multishoot::periodic::sin_cos_grid(Parity::Odd, m, j, 1);
        let two_m = DD::from_f64(2.0 / m as f64);
        sh.extra.push(multishoot::ExtraRow {
            terms: (0..m).map(|j| (j, sc(j).1 * two_m)).collect(),
            rhs: DD::ONE,
        });
        sh.extra.push(multishoot::ExtraRow {
            terms: (0..m).map(|j| (j, sc(j).0)).collect(),
            rhs: DD::ZERO,
        });
        let l = sh.len();
        let (mut zh, mut zl) = (vec![0.0; l], vec![0.0; l]);
        let mut put = |o: usize, f: &Fl<DV, B>| f.store(&mut zh[o..o + m], &mut zl[o..o + m]);
        put(0, &sh.eq.exact(0.0)[1]);
        for i in 0..sh.k() {
            let s = sh.eq.exact(sh.nodes[i]);
            put(sh.off_node(i), &s[0]);
            put(sh.off_node(i) + m, &s[1]);
        }
        put(sh.off_right(), &sh.eq.exact(1.0)[1]);
        let og = sh.off_global();
        let lam = sh.eq.lambda();
        (zh[og], zl[og]) = (lam.hi, lam.lo);
        let z = vec![zh, zl];
        sh.plan_segments(
            &z[0],
            &z[1],
            &multishoot::Plan {
                kt,
                tol: 1e-34,
                hmax: 0.25,
            },
        );
        (sh, z)
    }
}

impl<const B: usize> Default for Cylinder<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const B: usize> Equations<B> for Cylinder<B> {
    const NF: usize = 2;
    const NL: usize = 1;
    const NR: usize = 1;
    const NG: usize = 2;
    type Prep = ();

    fn taylor_coeffs<T: Lane>(&self, _x0: f64, g: &[T], w: &mut Work<T, B>) {
        let mu = if self.freeze_mu {
            <T::P as multishoot::lane::Param>::from_dd(self.mu)
        } else {
            <T::P as multishoot::lane::Param>::var(self.mu)
        };
        let (lam, nu) = (g[0], g[1]);
        let lam = if self.wrong_lambda {
            lam.muls(DD::ONE + DD::from_f64(2f64.powi(-50)))
        } else {
            lam
        };
        for k in 0..w.kt {
            let ik = DD::ONE / DD::from_f64((k + 1) as f64);
            let u = w.c[0][k];
            let du = self.d.apply(&u);
            let rhs = -(self.d.apply(&du).mulp(mu) + u.mul_t(lam) + du.mul_t(nu));
            w.c[0][k + 1] = w.c[1][k].muls(ik);
            w.c[1][k + 1] = rhs.muls(ik);
        }
    }

    /// u(0) = 0; the data is v(0)
    fn left_state<T: Lane>(&self, _x: f64, data: &[Fl<T, B>], _g: &[T], _prep: &()) -> State<T, B> {
        vec![Fl::zero(), data[0]]
    }

    /// u(1) = 0; the data is v(1)
    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, B>], _g: &[T], _prep: &()) -> State<T, B> {
        vec![Fl::zero(), data[0]]
    }
}

/// Bessel's equation `x^2 y'' + x y' + (x^2 - nu^2) y = 0` for the orders nu = 0..7, one per lane (B = 1, pointwise),
/// as the system `x y' = p`, `x p' = (nu^2 - x^2) y` in y and p = x y'.
///
/// x = 0 is a regular singular point. The regular solution starts from its Frobenius series `y = a sum_m c_m
/// x^(nu + 2m)`, c_0 = 1, `c_m = -c_(m-1) / (4 m (m + nu))`, summed to the first node once per evaluation
/// (`prepare`); its amplitude a is the left data. At the right end y = y*(x_R) and p is the data. The solution is
/// a = 1 and y = y* = 2^nu nu! J_nu(x), whose values `exact` takes from the closed form `c_m = (-1)^m / D_m`, D_m =
/// 4^m m! (nu + 1)...(nu + m) as one product.
pub struct Bessel {
    pub nu: Fl<DV, 1>,
    pub nu2: Fl<DV, 1>,
    pub x_left: f64,
    pub x_right: f64,
    /// the control: the Frobenius coefficient c_3 off by 2^-60
    pub wrong: bool,
    /// the number of `prepare` calls
    pub prepared: std::sync::atomic::AtomicUsize,
    /// 4 m (m + nu) for m = 1..=40, and x_left where nu > p (else 1) for p = 0..7, lane-wise constants
    dens: Vec<Fl<DV, 1>>,
    xsel: Vec<Fl<DV, 1>>,
    y_right: Fl<DV, 1>,
}

/// The Frobenius series at the first node: y and p = x y' of the solution with a = 1
pub struct BesselPrep {
    pub x: f64,
    pub phi: Fl<DV, 1>,
    pub psi: Fl<DV, 1>,
}

impl Default for BesselPrep {
    fn default() -> Self {
        BesselPrep {
            x: f64::NAN,
            phi: Fl::zero(),
            psi: Fl::zero(),
        }
    }
}

impl Bessel {
    pub fn new(x_left: f64, x_right: f64) -> Self {
        let nu = field::<1>(|j| DD::from_f64(j as f64));
        let dens = (1..=40)
            .map(|m| field::<1>(|j| DD::from_f64((4 * m * (m + j)) as f64)))
            .collect();
        let xsel = (0..7)
            .map(|p| field::<1>(|j| if j > p { DD::from_f64(x_left) } else { DD::ONE }))
            .collect();
        let mut b = Bessel {
            nu,
            nu2: nu * nu,
            x_left,
            x_right,
            wrong: false,
            prepared: std::sync::atomic::AtomicUsize::new(0),
            dens,
            xsel,
            y_right: Fl::zero(),
        };
        b.y_right = b.exact(x_right)[0];
        b
    }

    /// (y*, p*) at x from the closed-form coefficients, lane by lane
    pub fn exact(&self, x: f64) -> State<DV, 1> {
        let xd = DD::from_f64(x);
        let lane = |j: usize| -> (DD, DD) {
            let mut xp = DD::ONE;
            for _ in 0..j {
                xp = xp * xd;
            }
            let (mut y, mut p) = (DD::ZERO, DD::ZERO);
            let mut d = DD::ONE;
            for m in 0..40usize {
                if m > 0 {
                    d = d * DD::from_f64((4 * m * (j + m)) as f64);
                    xp = xp * xd * xd;
                }
                let t = xp / d;
                let t = if m % 2 == 1 { -t } else { t };
                y = y + t;
                p = p + t.mul_f((j + 2 * m) as f64);
            }
            (y, p)
        };
        vec![field::<1>(|j| lane(j).0), field::<1>(|j| lane(j).1)]
    }

    /// The shooting problem on five nodes from x_left to x_right, planned from the solution (order kt, last terms
    /// below 1e-32, steps at most 0.25), and the solution's z
    pub fn shooting(eq: Bessel, kt: usize) -> (multishoot::Shooting<Bessel, 1>, Vec<Vec<f64>>) {
        let (xl, xr) = (eq.x_left, eq.x_right);
        let nodes: Vec<f64> = (0..5).map(|i| xl + (xr - xl) * i as f64 / 4.0).collect();
        let mut sh = multishoot::Shooting::new(eq, nodes, kt);
        let l = sh.len();
        let (mut zh, mut zl) = (vec![0.0; l], vec![0.0; l]);
        let mut put = |o: usize, f: &Fl<DV, 1>| f.store(&mut zh[o..o + 8], &mut zl[o..o + 8]);
        put(0, &Fl::splat(DD::ONE));
        for i in 0..sh.k() {
            let s = sh.eq.exact(sh.nodes[i]);
            put(sh.off_node(i), &s[0]);
            put(sh.off_node(i) + 8, &s[1]);
        }
        put(sh.off_right(), &sh.eq.exact(xr)[1]);
        let z = vec![zh, zl];
        sh.plan_segments(
            &z[0],
            &z[1],
            &multishoot::Plan {
                kt,
                tol: 1e-32,
                hmax: 0.25,
            },
        );
        (sh, z)
    }
}

impl Equations<1> for Bessel {
    const NF: usize = 2;
    const NL: usize = 1;
    const NR: usize = 1;
    const POINTWISE: bool = true;
    type Prep = BesselPrep;

    /// the Frobenius series at x_left, lane by lane in the vector arithmetic: x^nu, then the terms by their ratio
    fn prepare(&self, _zh: &[f64], _zl: &[f64]) -> BesselPrep {
        self.prepared.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let x = DD::from_f64(self.x_left);
        let mut t = Fl::<DV, 1>::splat(DD::ONE);
        for s in &self.xsel {
            t = t * *s;
        }
        let x2 = Fl::<DV, 1>::splat(x * x);
        let (mut phi, mut psi) = (t, t * self.nu);
        for (i, den) in self.dens.iter().enumerate() {
            let m = i + 1;
            t = -(t * x2) / *den;
            if self.wrong && m == 3 {
                t = t + t.mulf(2f64.powi(-60));
            }
            phi = phi + t;
            psi = psi + t * (self.nu + Fl::splat(DD::from_f64(2.0 * m as f64)));
        }
        BesselPrep {
            x: self.x_left,
            phi,
            psi,
        }
    }

    fn taylor_coeffs<T: Lane>(&self, x0: f64, _g: &[T], w: &mut Work<T, 1>) {
        // x0 (k+1) y_(k+1) + k y_k = p_k;  x0 (k+1) p_(k+1) + k p_k = nu^2 y_k - (x^2 y)_k
        let x0d = DD::from_f64(x0);
        let (x0sq, x02) = (x0d * x0d, x0d.mul_f(2.0));
        let nu2 = Fl::<T, 1>::cst(&self.nu2);
        for k in 0..w.kt {
            let ik = DD::ONE / (x0d * DD::from_f64((k + 1) as f64));
            let (y, p) = (&w.c[0], &w.c[1]);
            let mut x2y = y[k].muls(x0sq);
            if k >= 1 {
                x2y = x2y + y[k - 1].muls(x02);
            }
            if k >= 2 {
                x2y = x2y + y[k - 2];
            }
            let yn = (p[k] - y[k].mulf(k as f64)).muls(ik);
            let pn = ((nu2 * y[k] - x2y) - p[k].mulf(k as f64)).muls(ik);
            w.c[0][k + 1] = yn;
            w.c[1][k + 1] = pn;
        }
    }

    /// the Frobenius series times the amplitude a, at the first node
    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, 1>], _g: &[T], prep: &BesselPrep) -> State<T, 1> {
        assert_eq!(x, prep.x, "the preparation is for x = {}", prep.x);
        vec![data[0] * Fl::cst(&prep.phi), data[0] * Fl::cst(&prep.psi)]
    }

    /// y = y*(x_R); the data is p
    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], _g: &[T], _prep: &BesselPrep) -> State<T, 1> {
        vec![Fl::cst(&self.y_right), data[0]]
    }
}

/// A problem declared not pointwise, the methods of E unchanged: its Jacobian column by column
pub struct PerColumn<E>(pub E);

impl<E: Equations<B>, const B: usize> Equations<B> for PerColumn<E> {
    const NF: usize = E::NF;
    const NL: usize = E::NL;
    const NR: usize = E::NR;
    const NG: usize = E::NG;
    const NAUX: usize = E::NAUX;
    const POINTWISE: bool = false;
    type Prep = E::Prep;

    fn prepare(&self, zh: &[f64], zl: &[f64]) -> E::Prep {
        self.0.prepare(zh, zl)
    }
    fn prepare_rate(&self, zh: &[f64], zl: &[f64], rate: multishoot::Rate) -> E::Prep {
        self.0.prepare_rate(zh, zl, rate)
    }
    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        self.0.taylor_coeffs(x0, g, w)
    }
    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &E::Prep) -> State<T, B> {
        self.0.left_state(x, data, g, prep)
    }
    fn right_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &E::Prep) -> State<T, B> {
        self.0.right_state(x, data, g, prep)
    }
}
