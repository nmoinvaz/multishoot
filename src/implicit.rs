//! Taylor steps for an implicit second-order ODE `N(x, y, y', y'') = 0`, a polynomial in y'', with the root branch tracked
//! by continuity and the places where `dN/dy''` vanishes reported.
//!
//! `N = sum_g A_g(x) y^a y'^b y''^c`: the caller gives the monomials (a, b, c) and, at the start x0 of each step, the Taylor
//! coefficients of every A_g about x0 ([`Coefficients`]); the library knows nothing of any one problem. About x0 the fields
//! are series in `s = (x - x0) / l`, l a power of two. Order k >= 1 of N is set to zero and solved for the coefficient
//! `y''_k`, which enters it linearly with the factor `A0 = dN/dy''` at the order-0 values: `y''_k = -N_k / A0`, one division
//! per step ([`Recursion`]). `y_k` and `y'_k` follow from `y'_(k-1)` and `y''_(k-1)`. Order 0 of N is not imposed, so N is
//! constant along the flow and the data's `N = 0` carries through.
//!
//! The branch is chosen once, by the data (y, y', y'') at the start, and carried: no root is ever re-selected by its order
//! among the roots. [`Branch::Project`] makes one Newton correction of y'' at each step's start, [`Branch::Conserve`]
//! none. Two branches can be confused only where roots meet, which is where `A0 = 0`, and there the walk reports: a step
//! that starts at `A0 = 0` is [`Stop::ZeroPivot`], one that starts with `|A0| <= pivot_abs` is [`Stop::Pivot`] with an
//! estimate of the zero of A0 ([`Zero`]), and a sign change of A0 between two step starts is a [`Crossing`], a double root
//! that the branch passes with its analytic continuation intact (the zero of A0's series inside the step is located).
//!
//! Near a double root of N in y'' use [`Branch::Project`]; `Conserve` turns the crossing into an avoided one (see [`Branch`]).
//!
//! The scalar is [`Scalar`]: `f64`, double-, quad- and octo-double, and a field of any lane (`Fl<T, 1>`, with duals).
//! A walk ([`Walker`]) plans each step by multishoot's rule ([`Plan`], [`crate::taylor::step_from_norms`]) on the largest
//! coefficients of orders kt - 1 and kt, relative to each field's own size by default, and sums the series by Horner's rule.
use crate::lane::{BaseLane, Lane, DV, OV, QV};
use crate::taylor::{step_from_norms, Plan};
use multidouble::dd::DD;
use multidouble::field::Fl;
use multidouble::od::OD;
use multidouble::qd::QD;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The monomial `y^a y'^b y''^c`
pub type Mono = (usize, usize, usize);

/// The numbers a recursion runs on
pub trait Scalar:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    /// an `f64`, exactly
    fn from_f64(x: f64) -> Self;
    /// a double-double, exactly (rounded in `f64`)
    fn from_dd(x: DD) -> Self;
    /// times an `f64`, exact for a power of two
    fn mulf(self, s: f64) -> Self;
    /// rounded to double-double and back: the identity on `f64`, double-double and the lanes' fields (a control's tool)
    fn round_dd(self) -> Self;
    /// `sum_(j = 0..=k) a[j] b[k - j]`, the Cauchy product's coefficient. The default is the plain sum of products; the
    /// multi-double scalars and the lanes' fields accumulate in multidouble's dot accumulators (`Lane::acc`), as
    /// `field::cs` does, which keeps the rounding of a sum that cancels (as the orders of N do) below the plain sum's.
    fn cauchy(a: &[Self], b: &[Self], k: usize) -> Self {
        plain_cauchy(a, b, k)
    }
}

/// The scalars of a walk: a number with a leading `f64` and a unit roundoff
pub trait Real: Scalar + Debug + Send + Sync + 'static {
    /// the unit roundoff
    const EPS: f64;
    /// the leading part
    fn to_f64(self) -> f64;
}

impl Scalar for f64 {
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
    fn mulf(self, s: f64) -> f64 {
        self * s
    }
    fn round_dd(self) -> f64 {
        self
    }
}

impl Real for f64 {
    const EPS: f64 = 1.1102230246251565e-16;
    fn to_f64(self) -> f64 {
        self
    }
}

macro_rules! scalar_multidouble {
    ($t:ty, $eps:expr, $from_dd:expr, $to_dd:expr, $mulf:expr, $lane:ident) => {
        impl Scalar for $t {
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
            fn mulf(self, s: f64) -> $t {
                $mulf(self, s)
            }
            fn round_dd(self) -> $t {
                $from_dd($to_dd(self))
            }
            fn cauchy(a: &[$t], b: &[$t], k: usize) -> $t {
                let mut acc = $lane::acc0();
                for j in 0..=k {
                    acc = $lane::acc(acc, $lane::splat(a[j]), $lane::splat(b[k - j]));
                }
                Fl([$lane::fin(acc)]).get(0)
            }
        }
        impl Real for $t {
            const EPS: f64 = $eps;
            fn to_f64(self) -> f64 {
                <$t>::to_f64(self)
            }
        }
    };
}

scalar_multidouble!(DD, 1.232595164407831e-32, |x| x, |x| x, |x: DD, s| x.mul_f(s), DV);
scalar_multidouble!(
    QD,
    1.504632769052528e-64,
    QD::from_dd,
    QD::to_dd,
    |x: QD, s| x.mul_f(s),
    QV
);
scalar_multidouble!(OD, 2.3e-128, OD::from_dd, OD::to_dd, |x: OD, s| x.mul_f(s), OV);

impl<T: Lane> Scalar for Fl<T, 1> {
    fn zero() -> Self {
        Fl::zero()
    }
    fn one() -> Self {
        Fl::splat(DD::ONE)
    }
    fn from_f64(x: f64) -> Self {
        Fl::splat(DD::from_f64(x))
    }
    fn from_dd(x: DD) -> Self {
        Fl::splat(x)
    }
    fn mulf(self, s: f64) -> Self {
        Fl::mulf(self, s)
    }
    fn round_dd(self) -> Self {
        self
    }
    fn cauchy(a: &[Self], b: &[Self], k: usize) -> Self {
        multidouble::field::cs(a, b, k)
    }
}

/// The Taylor coefficients about x0 of one A_g: `c[..n]` are those that may be nonzero (`n <= c.len()`), the rest of `c`
/// is scratch
#[derive(Clone, Debug)]
pub struct Entry<T> {
    pub n: usize,
    pub c: Vec<T>,
}

/// The jets of the coefficients A_g about a step's start, one entry per monomial of the pattern
#[derive(Clone, Debug)]
pub struct Jets<T> {
    pub ent: Vec<Entry<T>>,
}

impl<T: Scalar> Jets<T> {
    /// jets of `groups` coefficients with room for `kt + 1` coefficients each, all zero
    pub fn new(groups: usize, kt: usize) -> Self {
        Jets {
            ent: (0..groups)
                .map(|_| Entry {
                    n: 0,
                    c: vec![T::zero(); kt + 1],
                })
                .collect(),
        }
    }
}

/// The problem's side: the monomials of N and the Taylor coefficients of their factors A_g(x) about any point
pub trait Coefficients<T: Scalar> {
    /// the monomials (a, b, c) of `N = sum_g A_g(x) y^a y'^b y''^c`, each once, fixed for the source; at least one has c >= 1
    fn pattern(&self) -> Vec<Mono>;
    /// `A_g(x0 + h) = sum_j out.ent[g].c[j] h^j`, g in the order of `pattern`: set `c[..n]` and `n` of every entry
    /// (`n <= kt + 1`; the entries past `n` are taken as zero)
    fn jets(&self, x0: f64, kt: usize, out: &mut Jets<T>);
}

/// The coefficients of `p(x0 + h)` for `p(x) = sum_m poly[m] x^m`, in the scalar's precision, into `out[..poly.len()]`:
/// `out[j] = sum_(m >= j) poly[m] C(m, j) x0^(m - j)`. Binomials are exact in `f64` up to degree 56.
pub fn taylor_shift<T: Scalar>(poly: &[T], x0: f64, out: &mut [T]) {
    let deg = poly.len().saturating_sub(1);
    assert!(deg <= 56, "taylor_shift: degree {deg} > 56");
    assert!(
        out.len() >= poly.len(),
        "taylor_shift: out holds {} < {}",
        out.len(),
        poly.len()
    );
    let mut pw = vec![T::one()];
    for i in 1..=deg {
        pw.push(pw[i - 1] * T::from_f64(x0));
    }
    for o in out.iter_mut().take(poly.len()) {
        *o = T::zero();
    }
    let mut row = vec![1u64];
    for m in 0..=deg {
        if m > 0 {
            let mut next = vec![1u64; m + 1];
            for j in 1..m {
                next[j] = row[j - 1] + row[j];
            }
            row = next;
        }
        for (j, &b) in row.iter().enumerate() {
            out[j] = out[j] + poly[m] * T::from_f64(b as f64) * pw[m - j];
        }
    }
}

/// A deliberate error of the recursion or the walk (feature `mutate`): each of the planted checks is run once with one of
/// these and must fail
#[cfg(feature = "mutate")]
pub use crate::control::ImplicitControl;

/// `sum_(j = 0..=k) a[j] b[k - j]` by the plain sum of products
fn plain_cauchy<T: Scalar>(a: &[T], b: &[T], k: usize) -> T {
    let mut s = a[0] * b[k];
    for j in 1..=k {
        s = s + a[j] * b[k - j];
    }
    s
}

/// `x * base^n`, with `base^n` applied in pieces that stay normal in `f64` where it does not: the pieces move `x` the
/// same way, so none leaves the range that `x` and the product are in
fn mul_pow<T: Scalar>(x: T, base: f64, n: usize) -> T {
    let f = base.powi(n as i32);
    if n <= 1 || f.is_normal() {
        x.mulf(f)
    } else {
        mul_pow(mul_pow(x, base, n / 2), base, n - n / 2)
    }
}

/// The series of one step: the fields in `s = (x - x0) / l` to order kt, and `A0 = dN/dy''` at order 0
#[derive(Clone, Debug)]
pub struct Series<T> {
    /// `c[f][k]`, f = y, y', y''; k = 0..=kt (entries kt + 1 and beyond are scratch)
    pub c: [Vec<T>; 3],
    /// the scale of the step variable
    pub l: f64,
    /// `dN/dy''` at the step's start, from the data in `c[.][0]`
    pub a0: T,
}

/// Where a monomial's `y^a y'^b` series is held
#[derive(Clone, Copy, Debug)]
enum Base {
    Unit,
    P0(usize),
    P1(usize),
    Pair(usize),
}

/// The Taylor recursion of the implicit equation about one point, with its work arrays kept between steps
#[derive(Clone, Debug)]
pub struct Recursion<T: Scalar> {
    pat: Vec<Mono>,
    kt: usize,
    amax: usize,
    bmax: usize,
    cmax: usize,
    base: Vec<Base>,
    pair_keys: Vec<(usize, usize)>,
    /// 1 / k, k = 0..=kt + 1 (index 0 unused)
    rk: Vec<T>,
    pw0: Vec<Vec<T>>,
    pw1: Vec<Vec<T>>,
    pw2: Vec<Vec<T>>,
    unit: Vec<T>,
    pairs: Vec<Vec<T>>,
    q: Vec<Vec<T>>,
    ex: Vec<Vec<T>>,
    nex: Vec<usize>,
    /// the series formed by the last [`Recursion::form`]
    pub series: Series<T>,
    #[cfg(feature = "mutate")]
    control: ImplicitControl,
}

impl<T: Scalar> Recursion<T> {
    /// A recursion of order `kt >= 1` for the monomials of `pattern`.
    ///
    /// # Panics
    /// `kt = 0`, a monomial twice, or no monomial with c >= 1.
    pub fn new(pattern: &[Mono], kt: usize) -> Self {
        assert!(kt >= 1, "Recursion::new: order kt = 0");
        for (i, p) in pattern.iter().enumerate() {
            assert!(!pattern[..i].contains(p), "Recursion::new: monomial {p:?} twice");
        }
        assert!(
            pattern.iter().any(|&(_, _, c)| c >= 1),
            "Recursion::new: no monomial with c >= 1"
        );
        let amax = pattern.iter().map(|m| m.0).max().unwrap();
        let bmax = pattern.iter().map(|m| m.1).max().unwrap();
        let cmax = pattern.iter().map(|m| m.2).max().unwrap();
        let len = kt + 2;
        let z = || vec![T::zero(); len];
        let mut pair_keys: Vec<(usize, usize)> = Vec::new();
        let mut base = Vec::new();
        for &(a, b, _) in pattern {
            base.push(match (a, b) {
                (0, 0) => Base::Unit,
                (a, 0) => Base::P0(a),
                (0, b) => Base::P1(b),
                (a, b) => {
                    let i = pair_keys.iter().position(|&k| k == (a, b)).unwrap_or_else(|| {
                        pair_keys.push((a, b));
                        pair_keys.len() - 1
                    });
                    Base::Pair(i)
                }
            });
        }
        let mut unit = z();
        unit[0] = T::one();
        let mut rk = vec![T::zero(); kt + 2];
        for (k, r) in rk.iter_mut().enumerate().skip(1) {
            *r = T::one() / T::from_f64(k as f64);
        }
        Recursion {
            pat: pattern.to_vec(),
            kt,
            amax,
            bmax,
            cmax,
            base,
            pairs: pair_keys.iter().map(|_| z()).collect(),
            pair_keys,
            rk,
            pw0: (0..=amax).map(|_| z()).collect(),
            pw1: (0..=bmax).map(|_| z()).collect(),
            pw2: (0..=cmax.max(1)).map(|_| z()).collect(),
            unit,
            q: pattern.iter().map(|m| if m.2 > 0 { z() } else { Vec::new() }).collect(),
            ex: pattern.iter().map(|_| z()).collect(),
            nex: vec![0; pattern.len()],
            series: Series {
                c: [z(), z(), z()],
                l: 1.0,
                a0: T::zero(),
            },
            #[cfg(feature = "mutate")]
            control: ImplicitControl::None,
        }
    }

    /// the order kt
    pub fn kt(&self) -> usize {
        self.kt
    }

    /// the monomials, in the order of the jets
    pub fn pattern(&self) -> &[Mono] {
        &self.pat
    }

    #[cfg(feature = "mutate")]
    pub fn set_control(&mut self, c: ImplicitControl) {
        self.control = c;
    }

    fn base_of(&self, g: usize) -> &[T] {
        match self.base[g] {
            Base::Unit => &self.unit,
            Base::P0(a) => &self.pw0[a],
            Base::P1(b) => &self.pw1[b],
            Base::Pair(i) => &self.pairs[i],
        }
    }

    /// orders k of the powers, the products and the y'' power products, with `y''_k` as it stands (zero for k >= 1)
    fn partial(&mut self, k: usize) {
        for a in 1..=self.amax {
            self.pw0[a][k] = if a == 1 {
                self.series.c[0][k]
            } else {
                T::cauchy(&self.pw0[a - 1], &self.series.c[0], k)
            };
        }
        for b in 1..=self.bmax {
            self.pw1[b][k] = if b == 1 {
                self.series.c[1][k]
            } else {
                T::cauchy(&self.pw1[b - 1], &self.series.c[1], k)
            };
        }
        self.pw2[1][k] = self.series.c[2][k];
        for c in 2..=self.cmax {
            let v = T::cauchy(&self.pw2[c - 1], &self.series.c[2], k);
            self.pw2[c][k] = v;
        }
        for i in 0..self.pair_keys.len() {
            let (a, b) = self.pair_keys[i];
            let v = T::cauchy(&self.pw0[a], &self.pw1[b], k);
            self.pairs[i][k] = v;
        }
        for g in 0..self.pat.len() {
            let c = self.pat[g].2;
            if c > 0 {
                let v = T::cauchy(self.base_of(g), &self.pw2[c], k);
                self.q[g][k] = v;
            }
        }
    }

    /// order k of N from the current series
    fn n_order(&self, k: usize) -> T {
        let mut r = T::zero();
        for g in 0..self.pat.len() {
            let src: &[T] = if self.pat[g].2 > 0 { &self.q[g] } else { self.base_of(g) };
            for j in 0..self.nex[g].min(k + 1) {
                r = r + src[k - j] * self.ex[g][j];
            }
        }
        r
    }

    /// `dN/dy''` at order 0
    fn a0(&self) -> T {
        let mut a0 = T::zero();
        for g in 0..self.pat.len() {
            let c = self.pat[g].2;
            if c == 0 || self.nex[g] == 0 {
                continue;
            }
            let pc = if c == 1 { T::one() } else { self.pw2[c - 1][0] };
            a0 = a0 + (self.base_of(g)[0] * pc * self.ex[g][0]).mulf(c as f64);
        }
        a0
    }

    /// set `y''_k = dk` and add the part of the y'' powers and their products that holds it
    fn fix(&mut self, k: usize, dk: T) {
        self.series.c[2][k] = dk;
        let y0 = self.series.c[2][0];
        // d(y''^c)_k / d y''_k = c y''_0^(c - 1)
        let mut corr = vec![T::zero(); self.cmax + 1];
        let mut t = dk;
        corr[1] = dk;
        for (c, cc) in corr.iter_mut().enumerate().skip(2) {
            t = t * y0;
            *cc = t.mulf(c as f64);
        }
        self.pw2[1][k] = dk;
        for (c, &cc) in corr.iter().enumerate().skip(2) {
            self.pw2[c][k] = self.pw2[c][k] + cc;
        }
        for g in 0..self.pat.len() {
            let c = self.pat[g].2;
            if c > 0 {
                let b0 = self.base_of(g)[0];
                self.q[g][k] = self.q[g][k] + b0 * corr[c];
            }
        }
    }

    /// Taylor coefficients 1..=kt of (y, y', y'') in `s = (x - x0) / l` from the data `y0` and the jets of the coefficients
    /// about x0 (`l` a power of two, so the scaling of the jets is exact, and `l^j` is applied in pieces where it is not an
    /// `f64`). Returns the series; `a0` is zero or not finite
    /// where the division fails, and the coefficients are then not finite.
    pub fn form(&mut self, jets: &Jets<T>, l: f64, y0: [T; 3]) -> &Series<T> {
        let kt = self.kt;
        assert_eq!(
            jets.ent.len(),
            self.pat.len(),
            "Recursion::form: {} jets for {} monomials",
            jets.ent.len(),
            self.pat.len()
        );
        #[cfg(feature = "mutate")]
        let plain = self.control == ImplicitControl::PlainScale;
        #[cfg(not(feature = "mutate"))]
        let plain = false;
        let mut lp = 1.0f64;
        let mut lps = vec![1.0f64; kt + 2];
        for v in lps.iter_mut().skip(1) {
            lp *= l;
            *v = lp;
        }
        for g in 0..self.pat.len() {
            let n = jets.ent[g].n.min(kt + 1).min(jets.ent[g].c.len());
            self.nex[g] = n;
            for (j, e) in self.ex[g].iter_mut().enumerate().take(n) {
                let c = jets.ent[g].c[j];
                // `l^j` itself may underflow or overflow where `c l^j` does not
                *e = if j == 0 {
                    c
                } else if plain || lps[j].is_normal() {
                    c.mulf(lps[j])
                } else {
                    mul_pow(c, l, j)
                };
            }
        }
        for (cf, &y) in self.series.c.iter_mut().zip(y0.iter()) {
            for v in cf.iter_mut() {
                *v = T::zero();
            }
            cf[0] = y;
        }
        self.series.l = l;
        for v in self
            .pw0
            .iter_mut()
            .chain(self.pw1.iter_mut())
            .chain(self.pw2.iter_mut())
            .chain(self.pairs.iter_mut())
        {
            for e in v.iter_mut() {
                *e = T::zero();
            }
        }
        for v in self.q.iter_mut() {
            for e in v.iter_mut() {
                *e = T::zero();
            }
        }
        self.pw0[0][0] = T::one();
        self.pw1[0][0] = T::one();
        self.pw2[0][0] = T::one();
        self.partial(0);
        self.series.a0 = self.a0();
        #[allow(unused_mut)]
        let mut inv = T::one() / self.series.a0;
        #[cfg(feature = "mutate")]
        if self.control == ImplicitControl::RecipDD {
            inv = inv.round_dd();
        }
        for k in 1..=kt {
            #[allow(unused_mut)]
            let mut ik = self.rk[k].mulf(l);
            #[cfg(feature = "mutate")]
            if self.control == ImplicitControl::RecipDD {
                ik = ik.round_dd();
            }
            self.series.c[0][k] = self.series.c[1][k - 1] * ik;
            self.series.c[1][k] = self.series.c[2][k - 1] * ik;
            self.series.c[2][k] = T::zero();
            self.partial(k);
            let r = self.n_order(k);
            #[allow(unused_mut)]
            let mut dk = -(r * inv);
            #[cfg(feature = "mutate")]
            if self.control == ImplicitControl::SkipOrder(k) {
                dk = T::zero();
            }
            self.fix(k, dk);
        }
        &self.series
    }

    /// Orders 0..=nmax of `A0 = dN/dy''` along the branch, in `s`, from the series formed last (`nmax <= kt`):
    /// `sum_g c A_g y^a y'^b y''^(c - 1)`.
    pub fn a0_series(&self, nmax: usize) -> Vec<T> {
        assert!(nmax <= self.kt, "a0_series: order {nmax} > kt = {}", self.kt);
        let mut out = vec![T::zero(); nmax + 1];
        for g in 0..self.pat.len() {
            let c = self.pat[g].2;
            if c == 0 || self.nex[g] == 0 {
                continue;
            }
            let low: &[T] = &self.pw2[c - 1];
            let base = self.base_of(g);
            for (k, o) in out.iter_mut().enumerate() {
                let mut t = T::zero();
                // (ex * base * y''^(c-1))_k as the double Cauchy sum
                for (j, &e) in self.ex[g].iter().enumerate().take(self.nex[g].min(k + 1)) {
                    t = t + T::cauchy(base, low, k - j) * e;
                }
                *o = *o + t.mulf(c as f64);
            }
        }
        out
    }
}

/// How the walk keeps the state on its branch at each step's start.
///
/// Which one near a double root of N in y'' (where the branch meets another root and `A0 = 0`): `Project`. `Conserve`
/// keeps N at its start value, a rounding eps and not 0, and for N = eps a double root is two roots: the analytic
/// branch becomes an avoided crossing (or a fold). A Taylor series about a point at distance d from the crossing then has
/// radius about d, the planner's steps shrink toward it and the walk ends with [`Stop::Tiny`], or it follows the other
/// branch. `Project` makes one Newton correction at each step's start, which brings N back to about roundoff when the
/// state is near the branch, so the double root stays a double root. It is one correction, not a projection onto N = 0
/// (for `N = y''^2 - 1` at `y'' = 2` it gives `y'' = 1.25`, where N is still 0.5625): it promises neither a zero of N
/// nor a crossing.
/// Where `|A0|` stays away from 0, both give the branch to roundoff; `Conserve` is then the cheaper (no evaluation of N
/// per step) and its drift of N is bounded by the roundoff of the steps (`WalkOut::n_rel_max`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Branch {
    /// one Newton correction of y'': `y'' -= N / A0` at the values of the state
    Project,
    /// none: the recursion keeps N constant along the flow
    Conserve,
}

/// What the step planner compares with the tolerance
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepNorm {
    /// each field's coefficients divided by the field's |c_0| (by 1 where c_0 = 0)
    Rel,
    /// the coefficients as they are
    Abs,
}

/// The settings of a walk
#[derive(Clone, Debug)]
pub struct WalkOpts {
    /// order, tolerance on the last two terms, largest step (x units)
    pub plan: Plan,
    pub branch: Branch,
    pub norm: StepNorm,
    /// every planned step is multiplied by this (0.5 halves every step)
    pub h_scale: f64,
    /// a step shorter than `tiny * max(|x|, 1e-6)` stops the walk ([`Stop::Tiny`])
    pub tiny: f64,
    pub max_steps: usize,
    /// a step that starts with `|A0| <= pivot_abs` stops the walk ([`Stop::Pivot`]); 0 stops at `A0 = 0` only
    pub pivot_abs: f64,
    /// a step whose start has `|N| / sum |terms| > drift_tol` before the correction stops the walk ([`Stop::Drift`])
    pub drift_tol: f64,
}

impl WalkOpts {
    /// order `kt`, tolerance `tol`, no cap on the step, `Project`, `Rel`, no guards but `A0 = 0`
    pub fn new(kt: usize, tol: f64) -> Self {
        WalkOpts {
            plan: Plan { kt, tol, hmax: 1e300 },
            branch: Branch::Project,
            norm: StepNorm::Rel,
            h_scale: 1.0,
            tiny: 1e-13,
            max_steps: 1_000_000,
            pivot_abs: 0.0,
            drift_tol: f64::INFINITY,
        }
    }
}

/// Why a walk ended
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// reached the end of the interval
    End,
    /// `|A0| <= pivot_abs` at a step's start: `WalkOut::zero` holds the estimate of the zero
    Pivot,
    /// `A0 = 0` at a step's start
    ZeroPivot,
    /// `|N| / sum |terms|` at a step's start above `drift_tol`
    Drift,
    /// a coefficient, the step or the state is not finite
    NonFinite,
    /// a step shorter than `tiny * max(|x|, 1e-6)`
    Tiny,
    StepLimit,
    /// [`Walker::advance`] without a [`Walker::form`] that succeeded for the current state
    NotFormed,
}

/// A sign change of A0 between two step starts: the branch passes a double root of N in y''. `x` is the zero of A0's
/// series inside the step (`located`), else the step's start.
#[derive(Clone, Debug)]
pub struct Crossing<T> {
    pub x: T,
    pub step: usize,
    pub located: bool,
}

/// An estimate of the zero of A0 from its first three Taylor coefficients: for `A0 ~ c |x* - x|^beta`,
/// `beta = 1 / (1 - rho)` with `rho = A0 A0'' / A0'^2`, and `x* = x - beta A0 / A0'`. Exact for a pure power; an estimate
/// otherwise, whose error is the departure of A0 from a power.
#[derive(Clone, Debug)]
pub struct Zero<T> {
    pub x: T,
    pub beta: f64,
}

/// What one step did
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Advance {
    /// the step landed on the target
    pub landed: bool,
    /// the step taken (x units), after the landing rule
    pub h: f64,
}

/// A walk's result
#[derive(Clone, Debug)]
pub struct WalkOut<T> {
    /// the end point and the state there
    pub x: f64,
    pub y: [T; 3],
    pub steps: usize,
    pub stop: Stop,
    /// the state on each checkpoint passed, with the step count
    pub cps: Vec<(f64, [T; 3], usize)>,
    pub crossings: Vec<Crossing<T>>,
    /// with [`Stop::Pivot`]: the estimate of the zero of A0
    pub zero: Option<Zero<T>>,
    /// the largest `|N| / sum |terms|` at a step's start after the correction (before it for `Conserve`), the first step
    /// excluded: the start data's own residual is not the walk's
    pub n_rel_max: f64,
    /// the same before the correction
    pub n_rel_pre_max: f64,
}

/// The step before: what a crossing's location needs
#[derive(Clone, Debug)]
struct Prev<T> {
    x: f64,
    y: [T; 3],
    l: f64,
    /// the step in s
    hs: T,
    a0_sign: f64,
}

/// A walk along x of an implicit equation from the state (y, y', y'') at x0
pub struct Walker<'a, T: Real, S: Coefficients<T>> {
    src: &'a S,
    opts: WalkOpts,
    rec: Recursion<T>,
    scratch: Option<Recursion<T>>,
    jets: Jets<T>,
    jets_prev: Jets<T>,
    x: f64,
    y: [T; 3],
    l: f64,
    steps: usize,
    first: bool,
    /// the series is that of the last successful `form` at the current state: `advance`, `set_y` and a `form` that stops
    /// clear it
    ready: bool,
    /// the crossing of the step before has been looked for (reported or not): once per step, however many times `form` runs
    checked: bool,
    /// the sign of A0 at the current step's start
    cur_sign: f64,
    prev: Option<Prev<T>>,
    crossings: Vec<Crossing<T>>,
    zero: Option<Zero<T>>,
    n_rel_max: f64,
    n_rel_pre_max: f64,
    #[cfg(feature = "mutate")]
    control: ImplicitControl,
}

fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `sum_k c[k] h^k` by Horner's rule
fn horner_t<T: Scalar>(c: &[T], h: T) -> T {
    c.iter().rev().fold(T::zero(), |r, &ck| r * h + ck)
}

/// `d/dh sum_k c[k] h^k`
fn horner_d<T: Scalar>(c: &[T], h: T) -> T {
    let mut r = T::zero();
    for k in (1..c.len()).rev() {
        r = r * h + c[k].mulf(k as f64);
    }
    r
}

impl<'a, T: Real, S: Coefficients<T>> Walker<'a, T, S> {
    /// A walk from `x0` with the state `y0`. The first step's scale is the power of two at or below |x0| (1 below 1).
    ///
    /// # Panics
    /// order below 2, tolerance or `h_scale` not positive.
    pub fn new(src: &'a S, x0: f64, y0: [T; 3], opts: WalkOpts) -> Self {
        assert!(opts.plan.kt >= 2, "Walker::new: order {} < 2", opts.plan.kt);
        assert!(
            opts.plan.tol > 0.0 && opts.h_scale > 0.0,
            "Walker::new: tol and h_scale must be positive"
        );
        let pat = src.pattern();
        let rec = Recursion::new(&pat, opts.plan.kt);
        let jets = Jets::new(pat.len(), opts.plan.kt);
        let l = if x0.abs() >= 1.0 {
            2f64.powi(x0.abs().log2().floor() as i32)
        } else {
            1.0
        };
        Walker {
            src,
            opts,
            rec,
            scratch: None,
            jets_prev: jets.clone(),
            jets,
            x: x0,
            y: y0,
            l,
            steps: 0,
            first: true,
            ready: false,
            checked: true,
            cur_sign: 0.0,
            prev: None,
            crossings: Vec::new(),
            zero: None,
            n_rel_max: 0.0,
            n_rel_pre_max: 0.0,
            #[cfg(feature = "mutate")]
            control: ImplicitControl::None,
        }
    }

    #[cfg(feature = "mutate")]
    pub fn set_control(&mut self, c: ImplicitControl) {
        self.control = c;
        self.rec.set_control(c);
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    /// the state, taken onto the branch by the last [`Walker::form`] under [`Branch::Project`]
    pub fn y(&self) -> [T; 3] {
        self.y
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// the series formed by the last [`Walker::form`]
    pub fn series(&self) -> &Series<T> {
        &self.rec.series
    }

    /// Replace the state (the data of a restart on a given root); the step before is forgotten, so no crossing is
    /// reported across the change.
    pub fn set_y(&mut self, y: [T; 3]) {
        self.y = y;
        self.prev = None;
        self.checked = true;
        self.ready = false;
    }

    /// N, `sum |terms|` (leading parts) and A0 at the state from the jets' order 0
    fn eval(&self, y: &[T; 3]) -> (T, f64, T) {
        let (mut v, mut sm, mut d) = (T::zero(), 0.0f64, T::zero());
        let pw = |x: T, n: usize| (0..n).fold(T::one(), |r, _| r * x);
        for (g, &(a, b, c)) in self.rec.pat.iter().enumerate() {
            if self.jets.ent[g].n == 0 {
                continue;
            }
            let base = pw(y[0], a) * pw(y[1], b);
            let co = self.jets.ent[g].c[0];
            let t = co * base * pw(y[2], c);
            v = v + t;
            sm += t.to_f64().abs();
            if c >= 1 {
                d = d + (co * base * pw(y[2], c - 1)).mulf(c as f64);
            }
        }
        (v, sm, d)
    }

    /// The jets about the current point, the state taken onto the branch (`Project`), and the series about the point.
    /// A stop leaves the state as the projection made it, and [`Walker::advance`] then refuses until a `form` succeeds.
    pub fn form(&mut self) -> Result<&Series<T>, Stop> {
        self.ready = false;
        let kt = self.opts.plan.kt;
        self.src.jets(self.x, kt, &mut self.jets);
        let (nv, sm, a0v) = self.eval(&self.y);
        let n_pre = if sm == 0.0 { 0.0 } else { nv.to_f64().abs() / sm };
        if n_pre.is_nan() {
            return Err(Stop::NonFinite);
        }
        if n_pre > self.opts.drift_tol {
            return Err(Stop::Drift);
        }
        if !self.first {
            self.n_rel_pre_max = self.n_rel_pre_max.max(n_pre);
        }
        let mut y = self.y;
        if self.opts.branch == Branch::Project && a0v.to_f64() != 0.0 {
            y[2] = y[2] - nv / a0v;
        }
        if !self.first {
            let n_post = match self.opts.branch {
                Branch::Project => {
                    let (nv2, sm2, _) = self.eval(&y);
                    if sm2 == 0.0 {
                        0.0
                    } else {
                        nv2.to_f64().abs() / sm2
                    }
                }
                Branch::Conserve => n_pre,
            };
            self.n_rel_max = self.n_rel_max.max(n_post);
        }
        self.y = y;
        self.rec.form(&self.jets, self.l, y);
        let a0 = self.rec.series.a0;
        let a0h = a0.to_f64();
        #[cfg(feature = "mutate")]
        let check_zero = self.control != ImplicitControl::NoPivotCheck;
        #[cfg(not(feature = "mutate"))]
        let check_zero = true;
        if check_zero && a0h == 0.0 {
            return Err(Stop::ZeroPivot);
        }
        if a0h.is_nan()
            || a0h.is_infinite()
            || self
                .rec
                .series
                .c
                .iter()
                .any(|cf| cf[..=kt].iter().any(|v| !v.to_f64().is_finite()))
        {
            return Err(Stop::NonFinite);
        }
        let sg = sign(a0h);
        #[cfg(feature = "mutate")]
        let crossings_on = self.control != ImplicitControl::NoCrossing;
        #[cfg(not(feature = "mutate"))]
        let crossings_on = true;
        if crossings_on {
            self.check_crossing(sg);
        }
        self.cur_sign = sg;
        #[cfg(feature = "mutate")]
        let pivot_on = self.control != ImplicitControl::NoPivotStop;
        #[cfg(not(feature = "mutate"))]
        let pivot_on = true;
        if pivot_on && a0h.abs() <= self.opts.pivot_abs {
            self.zero = self.estimate_zero();
            return Err(Stop::Pivot);
        }
        self.ready = true;
        Ok(&self.rec.series)
    }

    /// The crossing of the step before, if A0 (of sign `sg` here) changed sign across it: located and recorded once per step
    fn check_crossing(&mut self, sg: f64) {
        #[cfg(feature = "mutate")]
        let again = self.control == ImplicitControl::RepeatCrossing;
        #[cfg(not(feature = "mutate"))]
        let again = false;
        if self.checked && !again {
            return;
        }
        self.checked = true;
        if let Some(p) = &self.prev {
            if p.a0_sign != 0.0 && sg != 0.0 && p.a0_sign != sg {
                let c = self.locate_crossing();
                self.crossings.push(c);
            }
        }
    }

    /// The crossing of the last accepted step when no `form` has looked at its end (the walk ended there): A0 at the end
    /// state as it stands, with the jets about the end. The state is not projected and no step is taken.
    fn check_end(&mut self) {
        #[cfg(feature = "mutate")]
        if matches!(
            self.control,
            ImplicitControl::NoEndCrossing | ImplicitControl::NoCrossing
        ) {
            return;
        }
        if self.checked || self.prev.is_none() {
            return;
        }
        let kt = self.opts.plan.kt;
        self.src.jets(self.x, kt, &mut self.jets);
        let (_, _, a0) = self.eval(&self.y);
        self.check_crossing(sign(a0.to_f64()));
    }

    /// The zero of A0's series in the step before, between its start and its end (A0 changes sign there)
    fn locate_crossing(&mut self) -> Crossing<T> {
        let p = self.prev.clone().expect("a step before");
        let kt = self.opts.plan.kt;
        let pat = self.rec.pat.clone();
        let scratch = self.scratch.get_or_insert_with(|| Recursion::new(&pat, kt));
        self.src.jets(p.x, kt, &mut self.jets_prev);
        scratch.form(&self.jets_prev, p.l, p.y);
        let a = scratch.a0_series(kt);
        let at = |s: T| horner_t(&a, s).to_f64();
        let hs = p.hs;
        let (mut lo, mut hi) = (0.0f64, hs.to_f64());
        let s_lo = sign(at(T::zero()));
        let bracketed = s_lo != 0.0 && sign(at(hs)) == -s_lo;
        let step = self.steps;
        if !bracketed {
            return Crossing {
                x: T::from_f64(p.x),
                step,
                located: false,
            };
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if sign(at(T::from_f64(mid))) == s_lo {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let mut s = T::from_f64(0.5 * (lo + hi));
        for _ in 0..6 {
            let d = horner_d(&a, s);
            if d.to_f64() == 0.0 {
                break;
            }
            s = s - horner_t(&a, s) / d;
        }
        Crossing {
            x: T::from_f64(p.x) + s.mulf(p.l),
            step,
            located: true,
        }
    }

    /// The estimate of the zero of A0 from its orders 0 to 2 in the series just formed
    fn estimate_zero(&self) -> Option<Zero<T>> {
        let a = self.rec.a0_series(2);
        if a[1].to_f64() == 0.0 {
            return None;
        }
        // in s: A0' = a1, A0'' = 2 a2
        let rho = (a[0] * a[2]).mulf(2.0) / (a[1] * a[1]);
        #[allow(unused_mut)]
        let mut beta = T::one() / (T::one() - rho);
        #[cfg(feature = "mutate")]
        if self.control == ImplicitControl::ExponentHalf {
            beta = T::from_f64(0.5);
        }
        let b = beta.to_f64();
        if !(b.is_finite() && b > 0.0) {
            return None;
        }
        let s = -(beta * a[0] / a[1]);
        Some(Zero {
            x: T::from_f64(self.x) + s.mulf(self.l),
            beta: b,
        })
    }

    /// The planner's step in x from the series just formed: [`step_from_norms`] on the largest coefficient of orders kt - 1
    /// and kt (each field relative to its |c_0| under [`StepNorm::Rel`]), in the step variable, times l and `h_scale`.
    /// NaN where a coefficient is NaN.
    pub fn planned(&self) -> f64 {
        let kt = self.opts.plan.kt;
        let c = &self.rec.series.c;
        let norm = |k: usize| {
            let mut e = 0.0f64;
            for cf in c.iter() {
                let sc = match self.opts.norm {
                    StepNorm::Rel => {
                        let s0 = cf[0].to_f64().abs();
                        if s0 > 0.0 {
                            s0
                        } else {
                            1.0
                        }
                    }
                    StepNorm::Abs => 1.0,
                };
                let v = cf[k].to_f64().abs() / sc;
                if v.is_nan() {
                    return f64::NAN;
                }
                e = e.max(v);
            }
            e
        };
        let plan = Plan {
            kt,
            tol: self.opts.plan.tol,
            hmax: self.opts.plan.hmax / self.l,
        };
        step_from_norms(norm(kt - 1), norm(kt), kt, &plan) * self.l * self.opts.h_scale
    }

    /// Take a step of `h` (as planned) toward `target`, no further: a step that would leave less than 40% of the rest
    /// becomes half the rest, and one that reaches it lands on it. The state is summed by Horner's rule in s. Returns
    /// the step taken and whether it landed on `target`.
    ///
    /// The series must be the one [`Walker::form`] returned for the current state: [`Stop::NotFormed`] after a step, a
    /// [`Walker::set_y`] or a `form` that stopped, with the walk unchanged.
    pub fn advance(&mut self, h: f64, target: f64) -> Result<Advance, Stop> {
        let kt = self.opts.plan.kt;
        #[cfg(feature = "mutate")]
        let stale_ok = self.control == ImplicitControl::StaleSeries;
        #[cfg(not(feature = "mutate"))]
        let stale_ok = false;
        if !self.ready && !stale_ok {
            return Err(Stop::NotFormed);
        }
        if !(h.is_finite() && h > 0.0) {
            return Err(Stop::NonFinite);
        }
        let dir = if target > self.x { 1.0 } else { -1.0 };
        let rest = (target - self.x) * dir;
        let mut h = h;
        if h >= rest {
            h = rest;
        } else if h > 0.6 * rest {
            h = 0.5 * rest;
        }
        let xn = if h == rest { target } else { self.x + dir * h };
        if xn == self.x || h < self.opts.tiny * self.x.abs().max(1e-6) {
            return Err(Stop::Tiny);
        }
        let hd = (T::from_f64(xn) - T::from_f64(self.x)).mulf(1.0 / self.l);
        let c = &self.rec.series.c;
        let ns = [
            horner_t(&c[0][..=kt], hd),
            horner_t(&c[1][..=kt], hd),
            horner_t(&c[2][..=kt], hd),
        ];
        if ns.iter().any(|v| !v.to_f64().is_finite()) {
            return Err(Stop::NonFinite);
        }
        self.prev = Some(Prev {
            x: self.x,
            y: self.y,
            l: self.l,
            hs: hd,
            a0_sign: self.cur_sign,
        });
        self.y = ns;
        self.x = xn;
        // the next step's scale: the power of two nearest this step's length (the series' radius is a few of them)
        self.l = 2f64.powi((h / self.opts.h_scale).log2().round() as i32).max(1e-300);
        self.steps += 1;
        self.first = false;
        self.ready = false;
        self.checked = false;
        Ok(Advance {
            landed: xn == target,
            h,
        })
    }

    /// Walk to `x1`, landing on each checkpoint between on the way, until the end or a stop. The end of the last step
    /// is checked for a crossing too, whether the walk ended there or stopped on a guard, without projecting it.
    pub fn run(&mut self, x1: f64, cps: &[f64]) -> WalkOut<T> {
        let x0 = self.x;
        let dir = if x1 > x0 { 1.0 } else { -1.0 };
        let mut targets: Vec<f64> = cps
            .iter()
            .copied()
            .filter(|&c| (c - x0) * dir > 0.0 && (x1 - c) * dir > 0.0)
            .collect();
        targets.sort_by(|a, b| (a * dir).partial_cmp(&(b * dir)).unwrap());
        // a checkpoint given twice is one landing: the second would ask for a step of length 0
        #[cfg(feature = "mutate")]
        let dedup = self.control != ImplicitControl::NoDedup;
        #[cfg(not(feature = "mutate"))]
        let dedup = true;
        if dedup {
            targets.dedup();
        }
        targets.push(x1);
        let mut ti = 0;
        let mut cp = Vec::new();
        let mut stop = Stop::End;
        loop {
            if (x1 - self.x) * dir <= 0.0 {
                break;
            }
            if self.steps >= self.opts.max_steps {
                stop = Stop::StepLimit;
                break;
            }
            if let Err(s) = self.form() {
                stop = s;
                break;
            }
            let h = self.planned();
            match self.advance(h, targets[ti]) {
                Err(s) => {
                    stop = s;
                    break;
                }
                Ok(adv) => {
                    if adv.landed && ti + 1 < targets.len() {
                        cp.push((self.x, self.y, self.steps));
                        ti += 1;
                    }
                }
            }
        }
        self.check_end();
        WalkOut {
            x: self.x,
            y: self.y,
            steps: self.steps,
            stop,
            cps: cp,
            crossings: self.crossings.clone(),
            zero: self.zero.clone(),
            n_rel_max: self.n_rel_max,
            n_rel_pre_max: self.n_rel_pre_max,
        }
    }
}
