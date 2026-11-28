//! The plane stepper (`multishoot::plane`) on planted systems with known solutions, in f64, double-double and
//! quad-double (P1o: octo-double). The reference of each is a closed form that shares no code with the stepper.
//!
//! Common to P1, P2 and P3: Taylor tolerance 100 eps_lane, order `Plan::optimal_kt(tol)`, jets to the same order,
//! `hmax` 1, from x = 0. Pass criteria, written before the first run, with eps_lane the lane's unit roundoff
//! (`Real::EPS`):
//!
//! * the plane: the Frobenius norm of (I - Pi) Y, Pi the projector onto the exact plane formed in the lane from the
//!   exact basis, at most 1e4 eps (it bounds the sine of the largest principal angle);
//! * the log-volume: |ln_vol - exact| at most 1e4 eps max(1, |exact|);
//! * the returned columns orthonormal: max |Y^T Y - I| at most 1e2 eps.
//!
//! P1: J = P diag(24, 8, -8, -24) P^-1, P = L U unimodular (an integer J), k = 2, the plane P[:, :2] (not orthonormal),
//! to x = 12: the plane stays span P[:, :2] and ln_vol = 32 x + (1/2) ln 5.
//! P2: J = Q'Q^T + Q L Q^T with Q(x) = H Rot12(theta1(x)) Rot23(theta2(x)), a rotation, L = diag(20, 4, -8, -20), k = 2,
//! the plane Q(0)[:, :2] M with det M = 2, to x = 12: the fundamental matrix is Q(x) e^(L x), so the plane is
//! span Q(x)[:, :2] and ln_vol = 24 x + ln 2.
//! P3: J = P diag(24, 8, -8, -24) P^-1 + x A1 + x^2 A2, k = n = 4, the plane U (unimodular, not orthonormal), to x = 4:
//! by Liouville ln_vol = int tr J = 3 x^2 / 2 - 2 x^3 / 3 = -56 / 3.
//!
//! The controls (each fails its check; the library's under the feature `mutate`, the wrong inputs always):
//! `J_gamma` (P1 with gamma_2 scaled by 1 + 1e-6), `J_noQdot` (P2 without the Q'Q^T term), `J_trace` (P3 with
//! tr A1 off by 1), `wrong_plane` (P1's plane compared with span P[:, 1:3]), `ref_no_gram` (P1's and P2's exact
//! log-volume without the starting basis' term); `LowSum`, `RecipDD`, `ScaleOnly`, `NoGram`, `LogDrop`, `SkipQr`; and
//! the orthonormality check's own, a column duplicated and a column scaled by 1 + 1e5 eps. Bases of columns near
//! 1e200 and 1e-200, whose sums of squares leave f64's range, are orthonormalized with their log-volume kept.
use multishoot::multidouble::dd::DD;
use multishoot::multidouble::od::OD;
use multishoot::multidouble::qd::QD;
use multishoot::plane::{JetSource, Jets, Plane, Real};
use multishoot::taylor::Plan;

/// the lane's parts, and a number from a decimal string (the references' constants, from mpmath)
trait Lane: Real {
    const NAME: &'static str;
    fn parts(self) -> Vec<f64>;
    fn parse(s: &str) -> Self;
}

impl Lane for f64 {
    const NAME: &'static str = "f64";
    fn parts(self) -> Vec<f64> {
        vec![self]
    }
    fn parse(s: &str) -> f64 {
        s.parse().unwrap()
    }
}

impl Lane for DD {
    const NAME: &'static str = "dd";
    fn parts(self) -> Vec<f64> {
        vec![self.hi, self.lo]
    }
    fn parse(s: &str) -> DD {
        QD::parse(s).unwrap().to_dd()
    }
}

impl Lane for QD {
    const NAME: &'static str = "qd";
    fn parts(self) -> Vec<f64> {
        self.x.to_vec()
    }
    fn parse(s: &str) -> QD {
        QD::parse(s).unwrap()
    }
}

impl Lane for OD {
    const NAME: &'static str = "od";
    fn parts(self) -> Vec<f64> {
        self.x.to_vec()
    }
    fn parse(s: &str) -> OD {
        OD::parse(s).unwrap()
    }
}

/// ln 2 and (1/2) ln 5 to 140 digits (mpmath)
const LN2: &str = "0.69314718055994530941723212145817656807550013436025525412068000949339362196969471560586332699641868754200148102057068573368552023575813055703";
const HALF_LN5: &str = "0.80471895621705018730037966661309381976280067713425886095632394573708949385382888231506693904658980539998315151085778144986200261466233809982";

const N: usize = 4;
type Mat = [[f64; N]; N];

/// P = L U, unimodular, and its inverse (checked in `p_and_its_inverse`)
const P: Mat = [[1., 1., 0., 1.], [1., 2., 1., 1.], [0., 1., 2., 1.], [1., 1., 1., 3.]];
const PINV: Mat = [
    [7., -4., 3., -2.],
    [-4., 3., -2., 1.],
    [3., -2., 2., -1.],
    [-2., 1., -1., 1.],
];
/// the upper factor of P: a unimodular, not orthonormal, basis for P3's plane
const U: Mat = [[1., 1., 0., 1.], [0., 1., 1., 0.], [0., 0., 1., 1.], [0., 0., 0., 1.]];
const GAMMA: [f64; N] = [24., 8., -8., -24.];
const A1: Mat = [[1., 0., 1., 0.], [0., 1., 0., -1.], [1., 0., 1., 0.], [0., 1., 0., 0.]];
const A2: Mat = [[0., 1., 0., 0.], [0., -1., 0., 1.], [1., 0., -1., 0.], [0., 0., 0., 0.]];
const LAMBDA: [f64; N] = [20., 4., -8., -20.];

fn matmul(a: &Mat, b: &Mat) -> Mat {
    let mut c = [[0.0; N]; N];
    for (i, ci) in c.iter_mut().enumerate() {
        for (j, cij) in ci.iter_mut().enumerate() {
            *cij = (0..N).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

/// P diag(g) P^-1, exact in f64 for integer g
fn p_diag_pinv(g: &[f64; N]) -> Mat {
    let mut pd = P;
    for row in pd.iter_mut() {
        for (j, x) in row.iter_mut().enumerate() {
            *x *= g[j];
        }
    }
    matmul(&pd, &PINV)
}

/// The entries of a matrix polynomial J(x) = sum_m A_m x^m, with the jets about x0 in the lane
struct Poly {
    a: Vec<Mat>,
}

impl<T: Lane> JetSource<T> for Poly {
    fn pattern(&self) -> Vec<(usize, usize)> {
        let mut v = vec![];
        for i in 0..N {
            for j in 0..N {
                if self.a.iter().any(|m| m[i][j] != 0.0) {
                    v.push((i, j));
                }
            }
        }
        v
    }
    fn jets(&self, x0: f64, kj: usize, out: &mut Jets<T>) {
        let d = self.a.len();
        let x = T::from_f64(x0);
        // x0^p in the lane
        let mut pw = vec![T::one(); d];
        for p in 1..d {
            pw[p] = pw[p - 1] * x;
        }
        for en in out.ent.iter_mut() {
            let n = d.min(kj);
            for j in 0..n {
                // coefficient j: sum_(m >= j) C(m, j) x0^(m - j) A_m[i][j]
                let mut s = T::zero();
                for m in j..d {
                    let mut binom = 1.0;
                    for t in 0..j {
                        binom = binom * (m - t) as f64 / (t + 1) as f64;
                    }
                    s = s + T::from_f64(binom * self.a[m][en.i][en.j]) * pw[m - j];
                }
                en.c[j] = s;
            }
            en.n = n;
        }
    }
}

/// a truncated power series in t, `len` coefficients
type Ser<T> = Vec<T>;

fn s_add<T: Lane>(a: &[T], b: &[T]) -> Ser<T> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

fn s_sub<T: Lane>(a: &[T], b: &[T]) -> Ser<T> {
    a.iter().zip(b).map(|(&x, &y)| x - y).collect()
}

fn s_scale<T: Lane>(a: &[T], f: T) -> Ser<T> {
    a.iter().map(|&x| x * f).collect()
}

fn s_mul<T: Lane>(a: &[T], b: &[T]) -> Ser<T> {
    let n = a.len();
    (0..n)
        .map(|k| {
            let mut s = T::zero();
            for i in 0..=k {
                s = s + a[i] * b[k - i];
            }
            s
        })
        .collect()
}

fn s_recip<T: Lane>(a: &[T]) -> Ser<T> {
    let n = a.len();
    let b0 = a[0].recip();
    let mut b = vec![T::zero(); n];
    b[0] = b0;
    for k in 1..n {
        let mut s = T::zero();
        for i in 1..=k {
            s = s + a[i] * b[k - i];
        }
        b[k] = -(b0 * s);
    }
    b
}

/// the derivative; the last coefficient is not known and is returned zero
fn s_deriv<T: Lane>(a: &[T]) -> Ser<T> {
    let n = a.len();
    let mut d = vec![T::zero(); n];
    for k in 0..n - 1 {
        d[k] = a[k + 1] * T::from_f64((k + 1) as f64);
    }
    d
}

/// the constant series c
fn s_const<T: Lane>(c: f64, len: usize) -> Ser<T> {
    let mut s = vec![T::zero(); len];
    s[0] = T::from_f64(c);
    s
}

type SMat<T> = Vec<Vec<Ser<T>>>;

fn sm_mul<T: Lane>(a: &SMat<T>, b: &SMat<T>, len: usize) -> SMat<T> {
    (0..N)
        .map(|i| {
            (0..N)
                .map(|j| {
                    let mut s = vec![T::zero(); len];
                    for k in 0..N {
                        s = s_add(&s, &s_mul(&a[i][k], &b[k][j]));
                    }
                    s
                })
                .collect()
        })
        .collect()
}

fn sm_t<T: Lane>(a: &SMat<T>) -> SMat<T> {
    (0..N).map(|i| (0..N).map(|j| a[j][i].clone()).collect()).collect()
}

/// H = (1/2) [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]], orthogonal, exact in binary
const H: Mat = [
    [0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5, 0.5],
];

/// the rotation by theta in the (a, b) plane of R^4 with cos theta = c, sin theta = s (scalars)
fn rot<T: Lane>(a: usize, b: usize, c: T, s: T) -> [[T; N]; N] {
    let mut r = [[T::zero(); N]; N];
    for (i, row) in r.iter_mut().enumerate() {
        row[i] = T::one();
    }
    r[a][a] = c;
    r[b][b] = c;
    r[a][b] = -s;
    r[b][a] = s;
    r
}

/// (cos theta, sin theta) = ((1 - u^2) / (1 + u^2), 2 u / (1 + u^2)) as series, u = (x0 + t) / tau
fn cs_series<T: Lane>(x0: f64, tau: f64, len: usize) -> (Ser<T>, Ser<T>) {
    let mut u = vec![T::zero(); len];
    u[0] = T::from_f64(x0 / tau);
    u[1] = T::from_f64(1.0 / tau);
    let u2 = s_mul(&u, &u);
    let one = s_const::<T>(1.0, len);
    let r = s_recip(&s_add(&one, &u2));
    (s_mul(&s_sub(&one, &u2), &r), s_scale(&s_mul(&u, &r), T::from_f64(2.0)))
}

/// Q(x0 + t) = H Rot12(theta1) Rot23(theta2) as a matrix of series of `len` coefficients
fn q_series<T: Lane>(x0: f64, len: usize) -> SMat<T> {
    let (c1, s1) = cs_series::<T>(x0, 2.0, len);
    let (c2, s2) = cs_series::<T>(x0, 4.0, len);
    let zero = vec![T::zero(); len];
    let one = s_const::<T>(1.0, len);
    let neg = |a: &Ser<T>| s_scale(a, T::from_f64(-1.0));
    let mut a: SMat<T> = (0..N).map(|_| (0..N).map(|_| zero.clone()).collect()).collect();
    let mut b = a.clone();
    for i in 0..N {
        a[i][i] = one.clone();
        b[i][i] = one.clone();
    }
    a[0][0] = c1.clone();
    a[1][1] = c1;
    a[0][1] = neg(&s1);
    a[1][0] = s1;
    b[1][1] = c2.clone();
    b[2][2] = c2;
    b[1][2] = neg(&s2);
    b[2][1] = s2;
    let h: SMat<T> = (0..N)
        .map(|i| (0..N).map(|j| s_const::<T>(H[i][j], len)).collect())
        .collect();
    sm_mul(&sm_mul(&h, &a, len), &b, len)
}

/// P2's system: J = Q'Q^T + Q L Q^T from the series of Q (the first term left out when `no_qdot`)
struct Rotating {
    no_qdot: bool,
}

impl<T: Lane> JetSource<T> for Rotating {
    fn pattern(&self) -> Vec<(usize, usize)> {
        (0..N).flat_map(|i| (0..N).map(move |j| (i, j))).collect()
    }
    fn jets(&self, x0: f64, kj: usize, out: &mut Jets<T>) {
        // one coefficient more than J's, for the derivative of Q
        let len = kj + 1;
        let q = q_series::<T>(x0, len);
        let qt = sm_t(&q);
        let mut ql = q.clone();
        for row in ql.iter_mut() {
            for (j, s) in row.iter_mut().enumerate() {
                *s = s_scale(s, T::from_f64(LAMBDA[j]));
            }
        }
        let mut j = sm_mul(&ql, &qt, len);
        if !self.no_qdot {
            let qd: SMat<T> = q.iter().map(|r| r.iter().map(|s| s_deriv(s)).collect()).collect();
            let t1 = sm_mul(&qd, &qt, len);
            for i in 0..N {
                for k in 0..N {
                    j[i][k] = s_add(&j[i][k], &t1[i][k]);
                }
            }
        }
        for en in out.ent.iter_mut() {
            en.c[..kj].copy_from_slice(&j[en.i][en.j][..kj]);
            en.n = kj;
        }
    }
}

/// the exact Q(x) in the lane, by the closed forms
fn q_exact<T: Lane>(x: f64) -> [[T; N]; N] {
    let cs = |tau: f64| {
        let u = T::from_f64(x / tau);
        let d = (T::one() + u * u).recip();
        ((T::one() - u * u) * d, T::from_f64(2.0) * u * d)
    };
    let (c1, s1) = cs(2.0);
    let (c2, s2) = cs(4.0);
    let a = rot::<T>(0, 1, c1, s1);
    let b = rot::<T>(1, 2, c2, s2);
    let mut ha = [[T::zero(); N]; N];
    let mut q = [[T::zero(); N]; N];
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                ha[i][j] = ha[i][j] + T::from_f64(H[i][k]) * a[k][j];
            }
        }
    }
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                q[i][j] = q[i][j] + ha[i][k] * b[k][j];
            }
        }
    }
    q
}

/// What a case reports
struct Out {
    /// ||(I - Pi) Y||_F, NaN for P3
    plane: f64,
    /// |ln_vol - exact|
    ln_err: f64,
    /// the bar of the log-volume
    ln_bar: f64,
    /// max |Y^T Y - I|
    orth: f64,
    steps: usize,
    hmin: f64,
    hmax: f64,
    /// a digest of every part of the returned columns and the log-volume, for the comparison across backends
    digest: u64,
    /// the parts of the computed and of the exact log-volume
    ln_parts: (Vec<f64>, Vec<f64>),
}

/// the wrong input of a control, if any
#[derive(Clone, Copy, PartialEq, Debug)]
enum Wrong {
    None,
    Gamma,
    NoQdot,
    Trace,
    OtherPlane,
    RefNoGram,
}

#[derive(Clone, Copy)]
struct Opts {
    wrong: Wrong,
    #[cfg(feature = "mutate")]
    control: multishoot::control::PlaneControl,
}

impl Opts {
    fn new() -> Self {
        Opts {
            wrong: Wrong::None,
            #[cfg(feature = "mutate")]
            control: multishoot::control::PlaneControl::None,
        }
    }
    fn wrong(w: Wrong) -> Self {
        let mut o = Self::new();
        o.wrong = w;
        o
    }
    #[cfg(feature = "mutate")]
    fn control(c: multishoot::control::PlaneControl) -> Self {
        Opts {
            control: c,
            ..Self::new()
        }
    }
}

fn plan_for<T: Lane>() -> (Plan, usize) {
    let tol = 100.0 * T::EPS;
    let kt = Plan::optimal_kt(tol);
    (Plan { kt, tol, hmax: 1.0 }, kt)
}

fn digest<T: Lane>(p: &Plane<T>) -> u64 {
    // FNV-1a over the bits of every part
    let mut h = 0xcbf29ce484222325u64;
    let mut eat = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for &x in p.columns() {
        x.parts().into_iter().for_each(&mut eat);
    }
    p.ln_vol().parts().into_iter().for_each(&mut eat);
    h
}

/// max |Y^T Y - I| of the k columns of the column-major n x k `y`, in the lane
fn orthonormality_of<T: Lane>(y: &[T], n: usize, k: usize) -> f64 {
    let mut worst = 0.0f64;
    for a in 0..k {
        for b in 0..=a {
            let mut d = T::zero();
            for i in 0..n {
                d = d + y[a * n + i] * y[b * n + i];
            }
            let e = if a == b { d - T::one() } else { d };
            worst = worst.max(e.to_f64().abs());
        }
    }
    worst
}

/// max |Y^T Y - I| of the plane's columns, in the lane
fn orthonormality<T: Lane>(p: &Plane<T>) -> f64 {
    orthonormality_of(p.columns(), p.n(), p.k())
}

/// the orthonormality check: max |Y^T Y - I| at most 1e2 eps (false for NaN)
fn orth_ok<T: Lane>(orth: f64) -> bool {
    orth <= 1e2 * T::EPS
}

/// ||(I - Pi) Y||_F for Y the plane's columns and Pi the projector onto the span of the columns of `b` (n x k, the
/// columns given as rows of `cols`), with Pi = B (B^T B)^-1 B^T formed in the lane from the 2 x 2 Gram matrix inverse
fn off_plane<T: Lane>(p: &Plane<T>, cols: &[[T; N]]) -> f64 {
    assert_eq!(cols.len(), 2);
    let g = |a: usize, b: usize| (0..N).fold(T::zero(), |s, i| s + cols[a][i] * cols[b][i]);
    let (g00, g01, g11) = (g(0, 0), g(0, 1), g(1, 1));
    let det = (g00 * g11 - g01 * g01).recip();
    let mut sum = 0.0f64;
    for c in 0..p.k() {
        let y = p.column(c);
        let t: Vec<T> = (0..2)
            .map(|a| (0..N).fold(T::zero(), |s, i| s + cols[a][i] * y[i]))
            .collect();
        // (B^T B)^-1 t
        let w = [(g11 * t[0] - g01 * t[1]) * det, (g00 * t[1] - g01 * t[0]) * det];
        for i in 0..N {
            let r = y[i] - (cols[0][i] * w[0] + cols[1][i] * w[1]);
            let r = r.to_f64();
            sum += r * r;
        }
    }
    sum.sqrt()
}

fn finish<T: Lane>(pl: &Plane<T>, st: multishoot::plane::Stats, plane: f64, exact: T) -> Out {
    let ln_err = (pl.ln_vol() - exact).to_f64().abs();
    Out {
        plane,
        ln_err,
        ln_bar: 1e4 * T::EPS * exact.to_f64().abs().max(1.0),
        orth: orthonormality(pl),
        steps: st.steps,
        hmin: st.hmin,
        hmax: st.hmax,
        digest: digest(pl),
        ln_parts: (pl.ln_vol().parts(), exact.parts()),
    }
}

/// P1: the constant system, the plane P[:, :2]
fn p1<T: Lane>(o: Opts) -> Out {
    let mut gamma = GAMMA;
    if o.wrong == Wrong::Gamma {
        gamma[1] *= 1.0 + 1e-6;
    }
    let src = Poly {
        a: vec![p_diag_pinv(&gamma)],
    };
    let basis: Vec<T> = (0..2).flat_map(|c| (0..N).map(move |i| T::from_f64(P[i][c]))).collect();
    let mut pl = Plane::<T>::new(N, 2, &basis);
    #[cfg(feature = "mutate")]
    pl.set_control(o.control);
    let (plan, kj) = plan_for::<T>();
    let st = pl.advance(&src, 0.0, 12.0, &plan, kj);
    // the exact plane: span P[:, :2], or the wrong control's P[:, 1:3]
    let (a, b) = if o.wrong == Wrong::OtherPlane { (1, 2) } else { (0, 1) };
    let cols: Vec<[T; N]> = [a, b]
        .iter()
        .map(|&c| std::array::from_fn(|i| T::from_f64(P[i][c])))
        .collect();
    let plane = off_plane(&pl, &cols);
    let gram = if o.wrong == Wrong::RefNoGram {
        T::zero()
    } else {
        T::parse(HALF_LN5)
    };
    let exact = T::from_f64(32.0 * 12.0) + gram;
    finish(&pl, st, plane, exact)
}

/// P2: the rotating system, the plane Q(0)[:, :2] M
fn p2<T: Lane>(o: Opts) -> Out {
    let src = Rotating {
        no_qdot: o.wrong == Wrong::NoQdot,
    };
    let m = [[3.0, 1.0], [1.0, 1.0]];
    let q0 = q_exact::<T>(0.0);
    let mut basis = vec![T::zero(); 2 * N];
    for c in 0..2 {
        for i in 0..N {
            basis[c * N + i] = (0..2).fold(T::zero(), |s, j| s + q0[i][j] * T::from_f64(m[j][c]));
        }
    }
    let mut pl = Plane::<T>::new(N, 2, &basis);
    #[cfg(feature = "mutate")]
    pl.set_control(o.control);
    let (plan, kj) = plan_for::<T>();
    let st = pl.advance(&src, 0.0, 12.0, &plan, kj);
    let q = q_exact::<T>(12.0);
    let cols: Vec<[T; N]> = (0..2).map(|c| std::array::from_fn(|i| q[i][c])).collect();
    let plane = off_plane(&pl, &cols);
    let gram = if o.wrong == Wrong::RefNoGram {
        T::zero()
    } else {
        T::parse(LN2)
    };
    let exact = T::from_f64(24.0 * 12.0) + gram;
    finish(&pl, st, plane, exact)
}

/// P3: Liouville, k = n
fn p3<T: Lane>(o: Opts) -> Out {
    let mut a1 = A1;
    if o.wrong == Wrong::Trace {
        a1[3][3] += 1.0;
    }
    let src = Poly {
        a: vec![p_diag_pinv(&GAMMA), a1, A2],
    };
    let basis: Vec<T> = (0..N).flat_map(|c| (0..N).map(move |i| T::from_f64(U[i][c]))).collect();
    let mut pl = Plane::<T>::new(N, N, &basis);
    #[cfg(feature = "mutate")]
    pl.set_control(o.control);
    let (plan, kj) = plan_for::<T>();
    let st = pl.advance(&src, 0.0, 4.0, &plan, kj);
    let exact = T::from_f64(-56.0) / T::from_f64(3.0);
    finish(&pl, st, f64::NAN, exact)
}

fn report(name: &str, lane: &str, o: &Out) {
    println!(
        "{name} {lane}: ln_vol parts {:?}, exact parts {:?}",
        o.ln_parts.0, o.ln_parts.1
    );
    println!(
        "{name} {lane}: plane {:.3e}, ln_vol error {:.3e} (bar {:.3e}), orthonormality {:.3e}, steps {} hmin {:.3e} hmax {:.3e}, digest {:016x}",
        o.plane, o.ln_err, o.ln_bar, o.orth, o.steps, o.hmin, o.hmax, o.digest
    );
}

/// whether the plane check, the log-volume check and the orthonormality of an output pass in the lane
fn checks<T: Lane>(o: &Out) -> [bool; 3] {
    [
        o.plane.is_nan() || o.plane <= 1e4 * T::EPS,
        o.ln_err <= o.ln_bar,
        orth_ok::<T>(o.orth),
    ]
}

fn pass<T: Lane>(name: &str, o: &Out) {
    report(name, T::NAME, o);
    assert_eq!(checks::<T>(o), [true; 3], "{name} {}: a check failed", T::NAME);
}

/// whether any of the plane and log-volume checks fails (the orthonormality is the stepper's own)
fn fires<T: Lane>(name: &str, ctl: &str, o: &Out) -> bool {
    let c = checks::<T>(o);
    println!(
        "control {ctl} on {name} {}: plane {:.3e}, ln_vol error {:.3e}, bars {:.3e} {:.3e}, {} steps: {}",
        T::NAME,
        o.plane,
        o.ln_err,
        1e4 * T::EPS,
        o.ln_bar,
        o.steps,
        if c[0] && c[1] { "NOT FIRED" } else { "fired" }
    );
    !(c[0] && c[1])
}

#[test]
fn p_and_its_inverse() {
    assert_eq!(
        matmul(&P, &PINV),
        [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    );
    let j = p_diag_pinv(&GAMMA);
    let tr: f64 = (0..N).map(|i| j[i][i]).sum();
    assert_eq!(tr, 0.0);
}

#[test]
fn p1_f64() {
    pass::<f64>("P1", &p1::<f64>(Opts::new()));
}

#[test]
fn p1_dd() {
    pass::<DD>("P1", &p1::<DD>(Opts::new()));
}

#[test]
fn p1_qd() {
    pass::<QD>("P1", &p1::<QD>(Opts::new()));
}

#[test]
fn p1_od() {
    pass::<OD>("P1o", &p1::<OD>(Opts::new()));
}

#[test]
fn p2_f64() {
    pass::<f64>("P2", &p2::<f64>(Opts::new()));
}

#[test]
fn p2_dd() {
    pass::<DD>("P2", &p2::<DD>(Opts::new()));
}

#[test]
fn p2_qd() {
    pass::<QD>("P2", &p2::<QD>(Opts::new()));
}

#[test]
fn p3_f64() {
    pass::<f64>("P3", &p3::<f64>(Opts::new()));
}

#[test]
fn p3_dd() {
    pass::<DD>("P3", &p3::<DD>(Opts::new()));
}

#[test]
fn p3_qd() {
    pass::<QD>("P3", &p3::<QD>(Opts::new()));
}

/// P1's wrong inputs, `J_gamma`, `wrong_plane` and `ref_no_gram`, each failing its check (octo-double's too, as P1o
/// is its case)
fn wrong_inputs_p1<T: Lane>() {
    assert!(fires::<T>("P1", "J_gamma", &p1::<T>(Opts::wrong(Wrong::Gamma))));
    assert!(fires::<T>(
        "P1",
        "wrong_plane",
        &p1::<T>(Opts::wrong(Wrong::OtherPlane))
    ));
    assert!(fires::<T>("P1", "ref_no_gram", &p1::<T>(Opts::wrong(Wrong::RefNoGram))));
}

/// The wrong inputs, each failing its check in every lane (no feature needed): P1's, `J_noQdot` and `ref_no_gram` on
/// P2, `J_trace` on P3.
fn wrong_inputs<T: Lane>() {
    wrong_inputs_p1::<T>();
    assert!(fires::<T>("P2", "J_noQdot", &p2::<T>(Opts::wrong(Wrong::NoQdot))));
    assert!(fires::<T>("P2", "ref_no_gram", &p2::<T>(Opts::wrong(Wrong::RefNoGram))));
    assert!(fires::<T>("P3", "J_trace", &p3::<T>(Opts::wrong(Wrong::Trace))));
}

#[test]
fn controls_wrong_inputs_f64() {
    wrong_inputs::<f64>();
}

#[test]
fn controls_wrong_inputs_dd() {
    wrong_inputs::<DD>();
}

#[test]
fn controls_wrong_inputs_qd() {
    wrong_inputs::<QD>();
}

#[test]
fn controls_wrong_inputs_od() {
    wrong_inputs_p1::<OD>();
}

/// The control of the orthonormality check, apart from the plane and log-volume checks, in every lane: the
/// orthonormalized basis P[:, :2] passes it, and fails it with its second column replaced by its first, or with its
/// first scaled by 1 + 1e5 eps (1e3 times the check's bound).
fn orthonormality_control<T: Lane>() {
    let basis: Vec<T> = (0..2).flat_map(|c| (0..N).map(move |i| T::from_f64(P[i][c]))).collect();
    let y = Plane::<T>::new(N, 2, &basis).columns().to_vec();
    assert!(
        orth_ok::<T>(orthonormality_of(&y, N, 2)),
        "the orthonormal basis fails in {}",
        T::NAME
    );
    let mut dup = y.clone();
    dup.copy_within(0..N, N);
    let mut long = y.clone();
    let f = T::one() + T::from_f64(1e5 * T::EPS);
    long[..N].iter_mut().for_each(|x| *x = *x * f);
    for (what, cols) in [("duplicate", &dup), ("scaled", &long)] {
        let e = orthonormality_of(cols, N, 2);
        println!("orthonormality control {what} {}: {e:.3e}", T::NAME);
        assert!(!orth_ok::<T>(e), "control {what} did not fire in {}", T::NAME);
    }
}

#[test]
fn controls_orthonormality() {
    orthonormality_control::<f64>();
    orthonormality_control::<DD>();
    orthonormality_control::<QD>();
    orthonormality_control::<OD>();
}

/// Pass, written before the first run: the basis (1, 2, 2) s of one column in R^3, s = 1e200 and 1e-200, whose sum
/// of squares overflows or underflows f64, in every lane: each entry of the column is that of (1, 2, 2) / 3 within
/// 10 eps, and ln_vol is ln(3 s) within 10 eps |ln(3 s)|. Control: ln 3, the log-volume of the rescaled column
/// without its scale back, fails the log-volume check.
fn extreme_scales<T: Lane>() {
    for s in [1e200, 1e-200] {
        let basis: Vec<T> = [1.0, 2.0, 2.0].iter().map(|&x| T::from_f64(x * s)).collect();
        let pl = Plane::<T>::new(3, 1, &basis);
        let three = T::from_f64(3.0);
        let errs: Vec<f64> = (pl.column(0).iter().zip([1.0, 2.0, 2.0]))
            .map(|(&y, x)| (y - T::from_f64(x) / three).to_f64().abs())
            .collect();
        let exact = (three * T::from_f64(s)).ln();
        let ln_ok = |v: T| (v - exact).to_f64().abs() <= 10.0 * T::EPS * exact.to_f64().abs();
        println!(
            "scale {s:e} {}: column errors {errs:?}, ln_vol {:.17e}, exact {:.17e}",
            T::NAME,
            pl.ln_vol().to_f64(),
            exact.to_f64()
        );
        assert!(
            errs.iter().all(|&e| e <= 10.0 * T::EPS) && ln_ok(pl.ln_vol()),
            "scale {s:e} in {}",
            T::NAME
        );
        assert!(!ln_ok(three.ln()), "control did not fire in {}", T::NAME);
    }
}

#[test]
fn extreme_scales_in_every_lane() {
    extreme_scales::<f64>();
    extreme_scales::<DD>();
    extreme_scales::<QD>();
    extreme_scales::<OD>();
}

/// The library's controls (feature `mutate`), each on the cells it is declared to fail (README amendment A1): `LowSum(8)`,
/// `ScaleOnly` and `LogDrop` on every case and lane (a stepper that refuses, its columns having become dependent, counts
/// as a failed check); `NoGram` on P1 and P2 (P3's starting basis has volume 1); `RecipDD` in
/// qd only (in f64 and dd its widened reciprocals are the lane's own).
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::PlaneControl as C;

    /// a case: its options in, its output out
    type Case = fn(Opts) -> Out;

    /// the check fails, or the stepper refuses (columns that became dependent: a panic) and returns no plane
    fn cell<T: Lane>(case: &str, f: fn(Opts) -> Out, ctl: C, name: &str) {
        match std::panic::catch_unwind(|| f(Opts::control(ctl))) {
            Ok(o) => assert!(fires::<T>(case, name, &o), "{name} did not fire on {case} {}", T::NAME),
            Err(_) => println!(
                "control {name} on {case} {}: the stepper panicked (dependent columns): fired",
                T::NAME
            ),
        }
    }

    fn all_cases<T: Lane>(ctl: C, name: &str) {
        cell::<T>("P1", p1::<T>, ctl, name);
        cell::<T>("P2", p2::<T>, ctl, name);
        cell::<T>("P3", p3::<T>, ctl, name);
    }

    fn on_every_case<T: Lane>() {
        all_cases::<T>(C::LowSum(8), "LowSum(8)");
        all_cases::<T>(C::ScaleOnly, "ScaleOnly");
        all_cases::<T>(C::LogDrop, "LogDrop");
        cell::<T>("P1", p1::<T>, C::NoGram, "NoGram");
        cell::<T>("P2", p2::<T>, C::NoGram, "NoGram");
    }

    /// RESULT, not a check: skipping one orthonormalization (after the first step, the middle one and the second to
    /// last) on every case and lane. In exact arithmetic it changes neither the plane nor the log-volume. Recorded
    /// outcome (README amendment A1): the errors stay at the baseline's on P1 and P2 in every lane and on P3 in f64 and
    /// dd (dd within 7% of the bar at the first step), and exceed the bar on P3 in qd by roundoff scaled with the
    /// condition number of the unnormalized product (cond P = 96), not by the growth spread, which is 1.2 to 6.7
    /// against ln(1/eps) of 36.7, 73.5 and 147 over one planned step. Nothing is asserted but that the stepper runs.
    fn skip_one<T: Lane>() {
        let cases: [(&str, Case); 3] = [("P1", p1::<T>), ("P2", p2::<T>), ("P3", p3::<T>)];
        for (case, f) in cases {
            let base = f(Opts::new());
            for i in [0, base.steps / 2, base.steps - 2] {
                let o = f(Opts::control(C::SkipQr(i, 1)));
                let c = checks::<T>(&o);
                println!(
                    "skip one after step {i} of {} on {case} {}: plane {:.3e}, ln_vol error {:.3e} (baseline {:.3e}, bar {:.3e}) -> {}",
                    base.steps,
                    T::NAME,
                    o.plane,
                    o.ln_err,
                    base.ln_err,
                    o.ln_bar,
                    if c[0] && c[1] { "within the bars" } else { "past a bar" }
                );
            }
        }
    }

    /// A perturbed shared method (feature `mutate`): multidouble's `real::Real::from_f64` one unit of f64 up, which the
    /// stepper reaches through `Real::from_f64` (the step's start and length, the series' constants). The plane and
    /// log-volume checks fire on every case in the multi-double lanes; f64's bars (1e4 eps) are above the perturbation
    /// and nothing is asserted there.
    fn shared_method<T: Lane>(must_fire: bool) {
        use multishoot::multidouble::real::mutate;
        mutate::set(mutate::FROM_F64);
        let r = std::panic::catch_unwind(|| {
            let cases: [(&str, Case); 3] = [("P1", p1::<T>), ("P2", p2::<T>), ("P3", p3::<T>)];
            cases.map(|(name, f)| fires::<T>(name, "FROM_F64", &f(Opts::new())))
        });
        mutate::set(0);
        let fired = r.expect("the stepper panicked under the perturbed method");
        if must_fire {
            assert_eq!(fired, [true; 3], "{}: FROM_F64 fired on P1 P2 P3 = {fired:?}", T::NAME);
        }
    }

    #[test]
    fn shared_method_f64() {
        shared_method::<f64>(false);
    }

    #[test]
    fn shared_method_dd() {
        shared_method::<DD>(true);
    }

    #[test]
    fn shared_method_qd() {
        shared_method::<QD>(true);
    }

    #[test]
    fn skip_one_f64() {
        skip_one::<f64>();
    }

    #[test]
    fn skip_one_dd() {
        skip_one::<DD>();
    }

    #[test]
    fn skip_one_qd() {
        skip_one::<QD>();
    }

    #[test]
    fn controls_f64() {
        on_every_case::<f64>();
    }

    #[test]
    fn controls_dd() {
        on_every_case::<DD>();
    }

    #[test]
    fn controls_qd() {
        on_every_case::<QD>();
        all_cases::<QD>(C::RecipDD, "RecipDD");
    }
}
