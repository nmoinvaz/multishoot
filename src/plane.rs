//! The plane stepper: a k-dimensional plane of solutions of the linear system `y' = J(x) y`, carried through Taylor
//! steps and orthonormalized after every step.
//!
//! The columns of an n x k matrix Y solve the system when `Y' = J Y`. Taylor steps from x0 give
//! `y_(m+1) = (1 / (m + 1)) sum_(j <= m) J_j y_(m-j)`, with `J_j` the Taylor coefficients of J about x0, supplied by the
//! caller as the jets of J's nonzero entries ([`JetSource`]); the library knows nothing of any one problem. A step is
//! planned by multishoot's rule ([`Plan`], [`crate::taylor::step_from_norms`]) from the coefficients of orders kt - 1 and
//! kt of the plane's columns, and the state is the Horner sum.
//!
//! Left alone, the columns of Y line up with the fastest growing mode, each column's component along the others
//! lost below the lane's roundoff once their growth rates differ by more than ln(1 / eps) over the interval. After
//! each step the plane is therefore replaced by an orthonormal basis of the same span, `Y = Q R`, R upper triangular
//! with a positive diagonal (Gram-Schmidt, each column's projections removed in turn and the pass repeated once). The
//! plane Q is the same plane, and the sum of `ln R_ii` over all steps is the log of the volume the plane gained:
//! [`Plane::ln_vol`]. By the Liouville formula that volume grows like `exp(Re int tr J dx)` when k = n (the integral is real for a real system), and a plane
//! spanned by the k fastest growing modes with rates `g_1 > ... > g_k` gains `(g_1 + ... + g_k) x`.
//!
//! The scalar is one lane's number type ([`Real`]): `f64`, `DD`, `QD` or `OD`, with the usual
//! operations. [`Real`] is not [`crate::Precision`]'s scalar: that has no division, square root or logarithm, and `f64`
//! is no [`crate::Precision`].
//!
//! A complex system is the same stepper over [`Complex`]`<R>`, a pair of lane values that is a scalar of the plane too
//! (the trait keeps the name `Real` of the real lanes it began with). The orthonormalization then uses the Hermitian product
//! `sum conj(y_p) y_c` and the norm `sqrt(sum |y|^2)`; R keeps a real positive diagonal, so [`Plane::ln_vol`] stays real
//! (a `Complex` with a zero imaginary part) and **the phase the flow gives the columns stays in Q**, where the plane's
//! orthonormal basis is the unique one with R positive: for a determinant of two planes, `det[Q_1 | Q_2]` carries the phase
//! of the determinant of the unnormalized columns, and the volumes only its modulus. Real data on a `Complex` lane give
//! the real lane's bits: the extra methods of [`Real`] (`conj`, `abs2`, `mag`, `re_f64`) are the identity, `z * z`, `|x|`
//! and `x` on the real lanes, and `Complex` takes the real lane's own operation for quotient, reciprocal, square root and
//! logarithm when its imaginary part is zero.
//!
//! The coefficients are scalars of one lane, so the plane does not use [`crate::Work`] or [`crate::Fl`] (fields on
//! collocation points, eight lanes per value) or the series lanes (`Jet<N>`, a series in the eight lanes of a vector);
//! it shares the planner's [`Plan`] and its step rule.
use crate::taylor::{step_from_norms, Plan};
use multidouble::dd::DD;
use multidouble::od::OD;
use multidouble::qd::QD;
use multisolver::prec::nan_max;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The number types of the plane's lanes: `f64`, double-double, quad-double and octo-double
pub trait Real:
    Copy
    + Debug
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// the unit roundoff of the lane
    const EPS: f64;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(x: f64) -> Self;
    /// from double-double, exactly (rounded in `f64`)
    fn from_dd(x: DD) -> Self;
    fn to_f64(self) -> f64;
    /// 1 / x in the lane
    fn recip(self) -> Self;
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    /// The complex conjugate; the identity on the real lanes.
    fn conj(self) -> Self {
        self
    }
    /// |z|^2 as a scalar of the lane; `z * z` on the real lanes.
    fn abs2(self) -> Self {
        self * self
    }
    /// |z| as `f64`; |x| on the real lanes.
    fn mag(self) -> f64 {
        self.to_f64().abs()
    }
    /// Re z as `f64`; the value on the real lanes.
    fn re_f64(self) -> f64 {
        self.to_f64()
    }
}

impl Real for f64 {
    const EPS: f64 = 1.1102230246251565e-16;
    fn zero() -> f64 {
        0.0
    }
    fn one() -> f64 {
        1.0
    }
    fn from_f64(x: f64) -> f64 {
        x
    }
    fn from_dd(x: DD) -> f64 {
        x.hi + x.lo
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn recip(self) -> f64 {
        1.0 / self
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn ln(self) -> f64 {
        f64::ln(self)
    }
}

macro_rules! real_multidouble {
    ($t:ty, $eps:expr, $from_dd:expr) => {
        impl Real for $t {
            const EPS: f64 = $eps;
            fn zero() -> $t {
                <$t>::from_f64(0.0)
            }
            fn one() -> $t {
                <$t>::from_f64(1.0)
            }
            fn from_f64(x: f64) -> $t {
                <$t>::from_f64(x)
            }
            fn from_dd(x: DD) -> $t {
                $from_dd(x)
            }
            fn to_f64(self) -> f64 {
                <$t>::to_f64(self)
            }
            fn recip(self) -> $t {
                <$t>::recip(self)
            }
            fn sqrt(self) -> $t {
                <$t>::sqrt(self)
            }
            fn ln(self) -> $t {
                <$t>::ln(self)
            }
        }
    };
}

real_multidouble!(DD, 1.232595164407831e-32, |x| x);
real_multidouble!(QD, 1.504632769052528e-64, QD::from_dd);
real_multidouble!(OD, 2.3e-128, OD::from_dd);

/// A complex number as a pair of lane values, a scalar of the plane like the real lanes: a plane of a complex system
/// `y' = J y` carries complex columns. With `im = 0` it behaves bit for bit as its real part: the product is
/// `(ac - bd, ad + bc)`, and the quotient, reciprocal, square root and logarithm take the real lane's own operation when
/// `im = 0`, so a complex plane whose data are real reproduces the real plane's bits.
///
/// `ln` is defined for a positive real only (the diagonal of R is real and positive, the only logarithm the stepper
/// takes); `to_f64` is defined for `im = 0` only, since a silent drop of an imaginary part is a bug (`re_f64` and `mag` read
/// the real part and the modulus). The square root is the principal one (Re >= 0).
#[derive(Clone, Copy, Debug)]
pub struct Complex<R: Real> {
    pub re: R,
    pub im: R,
}

impl<R: Real> Complex<R> {
    pub fn new(re: R, im: R) -> Self {
        Complex { re, im }
    }

    fn is_real(self) -> bool {
        self.im.to_f64() == 0.0
    }
}

impl<R: Real> Add for Complex<R> {
    type Output = Complex<R>;
    fn add(self, o: Complex<R>) -> Complex<R> {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl<R: Real> Sub for Complex<R> {
    type Output = Complex<R>;
    fn sub(self, o: Complex<R>) -> Complex<R> {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl<R: Real> Neg for Complex<R> {
    type Output = Complex<R>;
    fn neg(self) -> Complex<R> {
        Complex::new(-self.re, -self.im)
    }
}

impl<R: Real> Mul for Complex<R> {
    type Output = Complex<R>;
    fn mul(self, o: Complex<R>) -> Complex<R> {
        Complex::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

impl<R: Real> Div for Complex<R> {
    type Output = Complex<R>;
    /// the quotient by Smith's algorithm, which forms no squared denominator (a real divisor takes the lane's own quotient)
    fn div(self, o: Complex<R>) -> Complex<R> {
        if o.is_real() {
            return Complex::new(self.re / o.re, self.im / o.re);
        }
        if o.re.to_f64().abs() >= o.im.to_f64().abs() {
            let r = o.im / o.re;
            let den = o.re + o.im * r;
            Complex::new((self.re + self.im * r) / den, (self.im - self.re * r) / den)
        } else {
            let r = o.re / o.im;
            let den = o.re * r + o.im;
            Complex::new((self.re * r + self.im) / den, (self.im * r - self.re) / den)
        }
    }
}

impl<R: Real> Real for Complex<R> {
    const EPS: f64 = R::EPS;
    fn zero() -> Self {
        Complex::new(R::zero(), R::zero())
    }
    fn one() -> Self {
        Complex::new(R::one(), R::zero())
    }
    fn from_f64(x: f64) -> Self {
        Complex::new(R::from_f64(x), R::zero())
    }
    fn from_dd(x: DD) -> Self {
        Complex::new(R::from_dd(x), R::zero())
    }
    fn to_f64(self) -> f64 {
        assert!(
            self.is_real(),
            "Complex::to_f64 of a number with a nonzero imaginary part: use re_f64 or mag"
        );
        self.re.to_f64()
    }
    fn recip(self) -> Self {
        if self.is_real() {
            return Complex::new(self.re.recip(), R::zero());
        }
        Complex::one() / self
    }
    fn sqrt(self) -> Self {
        let (a, b) = (self.re, self.im);
        if self.is_real() {
            return if a.to_f64() >= 0.0 {
                Complex::new(a.sqrt(), R::zero())
            } else {
                Complex::new(R::zero(), (-a).sqrt())
            };
        }
        // components outside 2^+-250 would overflow or underflow in a^2 + b^2: scaled by an even power of two 2^-2k
        // (exactly), the root taken, and scaled back by 2^k
        let big = a.to_f64().abs().max(b.to_f64().abs());
        let k = if big.is_finite() && !(pow2(-250)..=pow2(250)).contains(&big) {
            ((((big.to_bits() >> 52) & 0x7ff) as i32) - 1023).clamp(-1022, 1022) / 2
        } else {
            0
        };
        let (a, b) = if k == 0 {
            (a, b)
        } else {
            let p = R::from_f64(pow2(-2 * k));
            (a * p, b * p)
        };
        let m = (a * a + b * b).sqrt();
        let abs_a = if a.to_f64() < 0.0 { -a } else { a };
        let t = ((m + abs_a) * R::from_f64(0.5)).sqrt();
        let two_t = t + t;
        let (re, im) = if a.to_f64() >= 0.0 {
            (t, b / two_t)
        } else {
            let abs_b = if b.to_f64() < 0.0 { -b } else { b };
            (abs_b / two_t, if b.to_f64() < 0.0 { -t } else { t })
        };
        if k == 0 {
            Complex::new(re, im)
        } else {
            let q = R::from_f64(pow2(k));
            Complex::new(re * q, im * q)
        }
    }
    fn ln(self) -> Self {
        assert!(
            self.is_real() && self.re.to_f64() > 0.0,
            "Complex::ln is defined for a positive real number only (the diagonal of R)"
        );
        Complex::new(self.re.ln(), R::zero())
    }
    fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }
    fn abs2(self) -> Self {
        if self.is_real() {
            return self * self;
        }
        Complex::new(self.re * self.re + self.im * self.im, R::zero())
    }
    fn mag(self) -> f64 {
        if self.is_real() {
            self.re.to_f64().abs()
        } else {
            self.re.to_f64().hypot(self.im.to_f64())
        }
    }
    fn re_f64(self) -> f64 {
        self.re.to_f64()
    }
}

/// The jets of one nonzero entry (i, j) of J about a step's start: `c[..n]` are the Taylor coefficients, the rest of
/// `c` is scratch
#[derive(Clone, Debug)]
pub struct Entry<T: Real> {
    pub i: usize,
    pub j: usize,
    /// the leading coefficients that may be nonzero (0: the entry is zero here), at most `c.len()`
    pub n: usize,
    pub c: Vec<T>,
}

/// The jets of the nonzero entries of J, one [`Entry`] per entry of the source's pattern
#[derive(Clone, Debug)]
pub struct Jets<T: Real> {
    pub ent: Vec<Entry<T>>,
}

impl<T: Real> Jets<T> {
    /// jets of the entries of `pattern`, each with room for `kj` coefficients, all zero
    pub fn new(pattern: &[(usize, usize)], kj: usize) -> Self {
        Jets {
            ent: pattern
                .iter()
                .map(|&(i, j)| Entry {
                    i,
                    j,
                    n: 0,
                    c: vec![T::zero(); kj],
                })
                .collect(),
        }
    }
}

/// The problem's side: J's nonzero pattern and its Taylor coefficients about any point
pub trait JetSource<T: Real> {
    /// the nonzero entries (row, column) of J, fixed for the source, none twice
    fn pattern(&self) -> Vec<(usize, usize)>;
    /// Coefficients 0..kj of every entry of the pattern about x0, in the same order as `pattern`: `out.ent[e].c[..n]` and
    /// `out.ent[e].n`, `n <= kj`. A constant entry has n = 1.
    fn jets(&self, x0: f64, kj: usize, out: &mut Jets<T>);
}

/// What an [`Plane::advance`] did
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    /// Taylor steps taken
    pub steps: usize,
    /// shortest and longest step (infinity and zero without a step)
    pub hmin: f64,
    pub hmax: f64,
}

/// Gram-Schmidt of the `k` columns of the column-major n x k matrix `y`, in place: each column has its projections on
/// the columns before it removed in turn, twice, and is scaled to unit length; the positive diagonal of R goes to
/// `diag[..k]`.
///
/// # Panics
/// A column whose norm after the projections is zero or not finite (the columns are dependent, or hold a NaN).
pub fn orthonormalize<T: Real>(y: &mut [T], n: usize, k: usize, diag: &mut [T]) {
    orthonormalize_with(y, n, k, diag, false)
}

/// `orthonormalize`; with `bilinear` the projections use the product without the conjugate (a control: on a complex
/// plane it loses the phase of the projection coefficients and of the norms)
fn orthonormalize_with<T: Real>(y: &mut [T], n: usize, k: usize, diag: &mut [T], bilinear: bool) {
    for c in 0..k {
        for _pass in 0..2 {
            for p in 0..c {
                let mut d = T::zero();
                for i in 0..n {
                    let yp = if bilinear { y[p * n + i] } else { y[p * n + i].conj() };
                    d = d + yp * y[c * n + i];
                }
                for i in 0..n {
                    y[c * n + i] = y[c * n + i] - d * y[p * n + i];
                }
            }
        }
        let nrm = normalize(&mut y[c * n..(c + 1) * n], bilinear);
        let v = nrm.re_f64();
        assert!(
            v > 0.0 && v.is_finite(),
            "orthonormalize: column {c} has norm {v:e} after its projections (dependent columns, or not finite)"
        );
        diag[c] = nrm;
    }
}

/// 2^e for e in -1022..=1023, exactly
fn pow2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// The length of the column `col`, which is scaled to unit length in place. A length outside 2^+-256 would square out
/// of the normal range in some part of the lane: the column is first scaled by the power of two 2^-e of its largest
/// entry (exactly, e clamped to 1022 in magnitude), and the length scaled back by 2^e. Inside that range the column is
/// not rescaled, and the result is bitwise that of the plain sum of squares.
fn normalize<T: Real>(col: &mut [T], bilinear: bool) -> T {
    let length = |col: &[T]| {
        col.iter()
            .fold(T::zero(), |s, &x| s + if bilinear { x * x } else { x.abs2() })
            .sqrt()
    };
    let mut nrm = length(col);
    let mut back = None;
    let v = nrm.re_f64();
    if !(pow2(-256)..=pow2(256)).contains(&v) {
        let m = col.iter().fold(0.0, |m, x| nan_max(m, x.mag()));
        if m > 0.0 && m.is_finite() {
            // the unbiased exponent of m; a subnormal m reads as -1023, clamped
            let e = ((((m.to_bits() >> 52) & 0x7ff) as i32) - 1023).clamp(-1022, 1022);
            let p = T::from_f64(pow2(-e));
            col.iter_mut().for_each(|x| *x = *x * p);
            nrm = length(col);
            back = Some(T::from_f64(pow2(e)));
        }
    }
    let inv = nrm.recip();
    col.iter_mut().for_each(|x| *x = *x * inv);
    back.map_or(nrm, |b| nrm * b)
}

/// Each of the k columns of the column-major n x k matrix `y` scaled to unit length in place, the lengths to
/// `diag[..k]` (a control's stand-in for `orthonormalize`)
fn scale_columns<T: Real>(y: &mut [T], n: usize, k: usize, diag: &mut [T]) {
    for c in 0..k {
        diag[c] = normalize(&mut y[c * n..(c + 1) * n], false);
    }
}

/// A k-plane in R^n of solutions of y' = J y, held as an orthonormal basis, with the log of its volume
#[derive(Clone, Debug)]
pub struct Plane<T: Real> {
    n: usize,
    k: usize,
    /// column-major n x k, orthonormal columns
    y: Vec<T>,
    ln_gram: T,
    ln_vol: T,
    #[cfg(feature = "mutate")]
    control: crate::control::PlaneControl,
}

impl<T: Real> Plane<T> {
    /// The plane spanned by the k columns of `basis` (column-major n x k, any basis of full rank), orthonormalized. Its
    /// log-volume starts at the log of the volume of the basis, `sum ln R_ii` of its orthonormalization.
    ///
    /// # Panics
    /// `basis.len() != n k`, k = 0, k > n, or dependent columns.
    pub fn new(n: usize, k: usize, basis: &[T]) -> Self {
        assert!(k >= 1 && k <= n, "Plane::new: k = {k} must be in 1..={n}");
        assert_eq!(
            basis.len(),
            n * k,
            "Plane::new: {n} x {k} basis, not {} values",
            basis.len()
        );
        let mut y = basis.to_vec();
        let mut diag = vec![T::zero(); k];
        orthonormalize(&mut y, n, k, &mut diag);
        let mut ln_gram = T::zero();
        for d in &diag {
            ln_gram = ln_gram + d.ln();
        }
        Plane {
            n,
            k,
            y,
            ln_gram,
            ln_vol: ln_gram,
            #[cfg(feature = "mutate")]
            control: crate::control::PlaneControl::None,
        }
    }

    /// Sets the deliberate error of the stepper, for a control (`PlaneControl::None` to clear it).
    #[cfg(feature = "mutate")]
    pub fn set_control(&mut self, c: crate::control::PlaneControl) {
        self.control = c;
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// the orthonormal basis, column-major n x k
    pub fn columns(&self) -> &[T] {
        &self.y
    }

    /// column c of the basis
    pub fn column(&self, c: usize) -> &[T] {
        &self.y[c * self.n..(c + 1) * self.n]
    }

    /// The log of the volume of the plane as carried from the basis `new` was given: the volume of that basis plus
    /// every step's `sum ln R_ii`. For the k fastest growing modes of rates `g_i` it is `(g_1 + ... + g_k) x` plus
    /// the log volume of the starting basis; for k = n, `int tr J dx` (its real part for a complex system) plus the log of |det| of the starting basis.
    pub fn ln_vol(&self) -> T {
        #[cfg(feature = "mutate")]
        if self.control == crate::control::PlaneControl::NoGram {
            return self.ln_vol - self.ln_gram;
        }
        self.ln_vol
    }

    /// the log of the volume of the basis `new` was given
    pub fn ln_gram(&self) -> T {
        self.ln_gram
    }

    /// Carries the plane from xa to xb (either direction) by Taylor steps of order `plan.kt`, the jets of J supplied
    /// by `src` to order `kj` (`1 <= kj <= plan.kt`; the recursion at order m uses the coefficients of J below m), each
    /// step planned by `plan` ([`step_from_norms`], at most `plan.hmax`) and the last ending exactly at xb: a step
    /// that would leave less than 40% of the rest becomes half the rest, as [`crate::taylor::plan_path`]'s. After
    /// every step the plane is orthonormalized and `sum ln R_ii` added to the log-volume.
    ///
    /// # Panics
    /// An end that is not finite, `plan.kt < 2`, `kj` outside 1..=kt, a tol or hmax not positive, a Taylor
    /// coefficient that is not finite, a step that is not positive or does not move x, and columns that become
    /// dependent.
    pub fn advance<S: JetSource<T>>(&mut self, src: &S, xa: f64, xb: f64, plan: &Plan, kj: usize) -> Stats {
        assert!(
            xa.is_finite() && xb.is_finite(),
            "Plane::advance: ends {xa} and {xb} must be finite"
        );
        let kt = plan.kt;
        assert!(kt >= 2, "Plane::advance: Taylor order {kt} < 2");
        assert!(kj >= 1 && kj <= kt, "Plane::advance: jets {kj} must be in 1..={kt}");
        assert!(
            plan.tol > 0.0 && plan.hmax > 0.0,
            "Plane::advance: tol {:e} and hmax {:e} must be positive",
            plan.tol,
            plan.hmax
        );
        let (n, k) = (self.n, self.k);
        let inv = reciprocals::<T>(kt, self.recip_in_dd());
        let mut jets = Jets::<T>::new(&src.pattern(), kj);
        let mut by_row: Vec<Vec<usize>> = vec![vec![]; n];
        for (e, en) in jets.ent.iter().enumerate() {
            assert!(
                en.i < n && en.j < n,
                "Plane::advance: pattern entry ({}, {}) outside {n} x {n}",
                en.i,
                en.j
            );
            by_row[en.i].push(e);
        }
        // co[(c n + i) w + m]: coefficient m of component i of column c
        let w = kt + 1;
        let mut co = vec![T::zero(); n * k * w];
        let mut diag = vec![T::zero(); k];
        let dir = if xb >= xa { 1.0 } else { -1.0 };
        let mut x0 = xa;
        let mut st = Stats {
            steps: 0,
            hmin: f64::INFINITY,
            hmax: 0.0,
        };
        while (xb - x0) * dir > 0.0 {
            src.jets(x0, kj, &mut jets);
            for cn in 0..n * k {
                co[cn * w] = self.y[cn];
            }
            for m in 0..kt {
                for c in 0..k {
                    for i in 0..n {
                        let mut s = T::zero();
                        for &e in &by_row[i] {
                            let en = &jets.ent[e];
                            if en.n == 0 {
                                continue;
                            }
                            let top = m.min(en.n - 1);
                            for j in 0..=top {
                                s = s + en.c[j] * co[(c * n + en.j) * w + m - j];
                            }
                        }
                        co[(c * n + i) * w + m + 1] = s * inv[m + 1];
                    }
                }
            }
            let norm = |m: usize| {
                let mut e = 0.0f64;
                for cn in 0..n * k {
                    e = nan_max(e, co[cn * w + m].mag());
                }
                e
            };
            let mut h = step_from_norms(norm(kt - 1), norm(kt), kt, plan);
            assert!(
                h > 0.0,
                "Plane::advance: step {h:e} at x = {x0} (a Taylor coefficient not finite, or tol too small)"
            );
            let rest = (xb - x0) * dir;
            if h >= rest {
                h = rest;
            } else if h > 0.6 * rest {
                h = 0.5 * rest;
            }
            let x1 = if h == rest { xb } else { x0 + dir * h };
            assert!(x1 != x0, "Plane::advance: step {h:e} does not move x = {x0}");
            let hh = T::from_f64(x1) - T::from_f64(x0);
            let top = self.horner_top(kt);
            for cn in 0..n * k {
                let mut s = co[cn * w + top];
                for m in (0..top).rev() {
                    s = s * hh + co[cn * w + m];
                }
                self.y[cn] = s;
            }
            if self.scale_only() {
                scale_columns(&mut self.y, n, k, &mut diag);
                self.add_ln(&diag);
            } else if self.orthonormalize_after(st.steps, x1 == xb) {
                let bilinear = self.bilinear();
                orthonormalize_with(&mut self.y, n, k, &mut diag, bilinear);
                self.add_ln(&diag);
            }
            st.steps += 1;
            st.hmin = st.hmin.min((x1 - x0).abs());
            st.hmax = st.hmax.max((x1 - x0).abs());
            x0 = x1;
        }
        if self.scale_only() && st.steps > 0 {
            let bilinear = self.bilinear();
            orthonormalize_with(&mut self.y, n, k, &mut diag, bilinear);
            self.add_ln(&diag);
        }
        st
    }

    /// adds the logs of an orthonormalization's diagonal to the log-volume
    fn add_ln(&mut self, diag: &[T]) {
        #[allow(unused_mut)]
        let mut last = self.k;
        #[cfg(feature = "mutate")]
        if self.control == crate::control::PlaneControl::LogDrop {
            last -= 1;
        }
        let bilinear = self.bilinear();
        for d in &diag[..last] {
            // the control's "norm" is complex: it takes the log of its modulus, in the lane
            self.ln_vol = self.ln_vol + if bilinear { d.abs2().sqrt().ln() } else { d.ln() };
        }
    }

    /// whether the step number `step` is followed by an orthonormalization
    #[allow(unused_variables)]
    fn orthonormalize_after(&self, step: usize, last: bool) -> bool {
        #[cfg(feature = "mutate")]
        match self.control {
            // a skip after the last step would leave the plane unnormalized: no control of the same plane
            crate::control::PlaneControl::SkipQr(i, m) if step >= i && step < i + m && !last => return false,
            _ => {}
        }
        true
    }

    /// whether the projections use the product without the conjugate (a control)
    fn bilinear(&self) -> bool {
        #[cfg(feature = "mutate")]
        {
            self.control == crate::control::PlaneControl::NoConj
        }
        #[cfg(not(feature = "mutate"))]
        {
            false
        }
    }

    /// whether the columns are only scaled between steps (a control)
    fn scale_only(&self) -> bool {
        #[cfg(feature = "mutate")]
        {
            self.control == crate::control::PlaneControl::ScaleOnly
        }
        #[cfg(not(feature = "mutate"))]
        {
            false
        }
    }

    /// the order of the Horner sum: kt
    #[allow(unused_variables)]
    fn horner_top(&self, kt: usize) -> usize {
        #[cfg(feature = "mutate")]
        if let crate::control::PlaneControl::LowSum(m) = self.control {
            return m.min(kt);
        }
        kt
    }

    /// whether 1 / m is formed in double-double
    fn recip_in_dd(&self) -> bool {
        #[cfg(feature = "mutate")]
        {
            self.control == crate::control::PlaneControl::RecipDD
        }
        #[cfg(not(feature = "mutate"))]
        {
            false
        }
    }
}

/// 1 / m for m = 1..=kt at the lane's precision (a quotient formed in the lane), index m; `in_dd` forms it in
/// double-double and widens it, for a control
fn reciprocals<T: Real>(kt: usize, in_dd: bool) -> Vec<T> {
    let mut v = vec![T::zero(); kt + 1];
    for (m, x) in v.iter_mut().enumerate().skip(1) {
        *x = if in_dd {
            T::from_dd(DD::ONE / DD::from_f64(m as f64))
        } else {
            T::one() / T::from_f64(m as f64)
        };
    }
    v
}
