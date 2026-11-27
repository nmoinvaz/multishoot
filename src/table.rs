//! Taylor tables: the series of a problem's coefficient functions about the start of each step of a path, every step
//! cut by the width at which the *stored* series have converged, and a lookup that shifts a step's series to any point
//! inside it.
//!
//! A system `y' = J(x) y` whose coefficients are known only numerically (a background that was itself integrated) needs
//! the Taylor coefficients of every entry of J about the start of each of the integrator's steps. The caller supplies
//! them at any point ([`SeriesSource`]); the table keeps them at chosen points and serves any point in between by
//! shifting. The library knows nothing of any one problem.
//!
//! # The step rule
//!
//! The series of a function about a point converge only within the distance to its nearest singularity, and the
//! functions the table stores can have singularities that the source's own data lack: a product of a regular background
//! with `1/r^2` has a pole at `r = 0` where the background is regular, so a step as long as the background's runs past
//! the radius of the stored function and the shifted series are wrong (by hundreds of percent near the pole).
//! The table therefore cuts each step by the stored functions' own series.
//!
//! With the coefficients `c_0 ..= c_(nb-1)` of one function about the step start, `t_k(h) = |c_k| h^k` and
//! `S(h) = sum_j t_j(h)`, the **width** `w` of the step is the largest `h <= cap` with
//!
//! ```text
//! t_k(h) <= tail * S(h) + atol    for k = nb - last ..= nb - 1, for every stored function,
//! ```
//!
//! and the step is `min(fraction * w, limit)`, `limit` the least of the distance to the end of the path, to the end of
//! the source's piece ([`SeriesSource::series`] returns it) and [`Opts::hmax`], and `cap = limit / fraction`. The
//! fraction leaves the rest of the width for a consumer's own step from inside the table step: its reach is
//! `|x0 - start| + h`, at most `fraction * w + h`. **A fraction that is right for one consumer is wrong for a looser
//! one**: the consumer's longest step must stay within `(1 - fraction) w`, or the series it sums are past `tail`.
//!
//! The test measures the dropped tail by the last `last` terms, as the planner of [`crate::taylor`] does. It is blind to
//! a series whose last `last` coefficients vanish by a symmetry of the function about the step start: `1 / (1 - x^3)`
//! about 0 has `c_k = 0` unless `k` is a multiple of 3, and with `last` = 2 the test sees only zeros and passes at
//! any h. A function with a symmetry of period p needs `last` > p; the default is 4.
//!
//! **Why a bisection.** With `k_bar(h)` the mean order of the terms weighted by `t_j(h)`,
//! `d ln(t_k / S) / d ln h = k - k_bar(h)`. `k_bar <= nb - 1`, so the ratio of the term of order `nb - 1` is
//! nondecreasing and the set where it is at most `tail` is an interval `(0, h_1]`; the ratio of order `nb - 2` is
//! nondecreasing while `k_bar <= nb - 2`, and so on. That is all that is shown: where `k_bar` exceeds `nb - j` the test of
//! order `nb - j` can pass again beyond a failure (a series dominated by its last terms). The search halves `h` down
//! from `cap` until the test passes at some `h` and fails at `2h`, then bisects between them (53 steps), so the step
//! ends at a point where the test passed and the one just beyond it failed, whether or not the passing set is an
//! interval, and its resolution is relative to `w`. It is **not** a bound on the truncation error: the test reads the
//! last terms of series that are only as good as the source's, and `tail` is the share of the sum they may carry.
//!
//! The terms are formed from a mantissa and a power of two (aligned to the largest), so `|c_k| h^k` neither overflows
//! nor underflows where `c_k` and `h^k` are in range of `f64` and their product is not; the plain products would give
//! zero last terms, and a step to the end of the path, for a function with `c_58` of order `1e290` (the control
//! `TableControl::Unscaled` of the feature `mutate`). Only IEEE operations are used, so the step points are the same
//! bits on every platform and backend.
//!
//! # Limits
//!
//! * **A singularity on or near the path.** A pole at distance d gives coefficients of size `d^-(k+1)`, and every part of
//!   a multidouble shares `f64`'s range, so the series of order `nb - 1` overflows at `d = f64::MAX^(-1/nb)` times the
//!   function's scale (5.6e-11 at nb 30, 7.4e-6 at nb 60) and [`Error::NotFinite`] is returned there; in `f64`, where
//!   the step stops moving `x` first, [`Error::Stall`]. A table cannot be built closer to a singularity than that.
//! * **The consumer's reach**, above: the table does not know the consumer's step.
//! * **`last`**, above: a symmetry of the function about a step start.
//!
//! # The lookup
//!
//! [`Table::shifted`] takes the stored series of the step that holds `x0` (a boundary belongs to the step that starts
//! there) and shifts it by `delta = x0 - start`, formed in the lane's own type: the difference of two `f64` numbers is
//! exact in `DD`, `QD` and `OD`, and rounded in `f64` where the two are not within a factor of 2 of each other.
//!
//! [`Table::headroom`] gives `width - |x0 - start|`, how far a step from `x0` stays inside the converged width.
use crate::plane::{JetSource, Jets, Real};
use std::fmt;

#[cfg(feature = "mutate")]
use crate::control::TableControl;

/// What the caller supplies: the Taylor series of its functions about any point of the path
pub trait SeriesSource<T: Real> {
    /// the number of functions F
    fn functions(&self) -> usize;
    /// The coefficients `0..nb` of every function about `x0`, `out[f * nb + k]` (`out.len() == F nb`), for a path that
    /// runs in the direction `dir` (+1 or -1). At a boundary of the source's own pieces, the piece that starts at `x0`
    /// in that direction is read. Returns the farthest point, in that direction, up to which these series hold (the end
    /// of the piece), or `dir` times infinity where they hold without limit.
    fn series(&self, x0: f64, dir: f64, nb: usize, out: &mut [T]) -> f64;
}

/// The settings of [`Table::build`]
#[derive(Clone, Debug)]
pub struct Opts {
    /// coefficients stored per function, 2 ..= 1000
    pub nb: usize,
    /// the share of the sum that the last `last` terms may carry, positive
    pub tail: f64,
    /// how many of the last coefficients the test reads, 1 ..= nb. 4: a series whose last three coefficients vanish
    /// by a symmetry of period 3 is still seen
    pub last: usize,
    /// a step is this share of the width, in (0, 1]
    pub fraction: f64,
    /// the longest step
    pub hmax: f64,
    /// an absolute floor under the test, in the units of the function (0: none), for a function that is zero up to
    /// noise
    pub atol: f64,
    /// the most steps before `Error::TooManySteps`
    pub max_steps: usize,
    /// the deliberate error of a control
    #[cfg(feature = "mutate")]
    pub control: TableControl,
}

impl Opts {
    /// `nb` coefficients and `tail`, with `last` 4, `fraction` 0.3, no `hmax`, no `atol` and a million steps at most
    pub fn new(nb: usize, tail: f64) -> Opts {
        Opts {
            nb,
            tail,
            last: 4,
            fraction: 0.3,
            hmax: f64::INFINITY,
            atol: 0.0,
            max_steps: 1_000_000,
            #[cfg(feature = "mutate")]
            control: TableControl::None,
        }
    }
}

/// Why a table could not be built
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error {
    /// the series of function `f` at `x` has a coefficient of order `k` that is NaN or infinite: a source that is wrong, or
    /// a singularity so near that the coefficient overflows `f64` (see the module's limits)
    NotFinite { x: f64, f: usize, k: usize },
    /// no step from `x` moves it: a singularity at `x` (or within rounding of it), or a source that ends its piece there
    Stall { x: f64 },
    /// `Opts::max_steps` steps were taken and the path was not at its end, at `x`
    TooManySteps { x: f64 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Error::NotFinite { x, f: func, k } => {
                write!(
                    f,
                    "table: coefficient {k} of function {func} about x = {x} is not finite"
                )
            }
            Error::Stall { x } => write!(f, "table: no step from x = {x} moves it (a singularity there)"),
            Error::TooManySteps { x } => write!(f, "table: too many steps, at x = {x}"),
        }
    }
}

impl std::error::Error for Error {}

/// 2^e for e in -1022..=1023, exactly
fn pow2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// `x` times 2^e, by exact powers: an IEEE product at every stage, saturating at 0 and infinity
fn scale2(mut x: f64, e: i64) -> f64 {
    let mut e = e.clamp(-2300, 2300);
    while e > 1000 {
        x *= pow2(1000);
        e -= 1000;
    }
    while e < -1000 {
        x *= pow2(-1000);
        e += 1000;
    }
    x * pow2(e as i32)
}

/// the mantissa in [1, 2) and the exponent of a positive finite `a`, exactly (a subnormal is normalized first)
fn frexp2(a: f64) -> (f64, i32) {
    let (a, adj) = if a < f64::MIN_POSITIVE {
        (a * pow2(64), -64)
    } else {
        (a, 0)
    };
    let bits = a.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    (
        f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023u64 << 52)),
        e + adj,
    )
}

/// The coefficients of one function as mantissas in [1, 2) (0 for a zero coefficient) and exponents
struct Terms {
    m: Vec<f64>,
    e: Vec<i32>,
}

impl Terms {
    fn of<T: Real>(c: &[T]) -> Terms {
        let mut m = Vec::with_capacity(c.len());
        let mut e = Vec::with_capacity(c.len());
        for x in c {
            let a = x.to_f64().abs();
            if a == 0.0 {
                m.push(0.0);
                e.push(0);
            } else {
                let (mm, ee) = frexp2(a);
                m.push(mm);
                e.push(ee);
            }
        }
        Terms { m, e }
    }

    /// whether `t_k(h) <= tail S(h) + atol` for the last `last` orders, the terms scaled by the power of two of the
    /// largest
    fn passes(&self, h: f64, last: usize, tail: f64, atol: f64) -> bool {
        let nb = self.m.len();
        let (mh, q) = frexp2(h);
        let q = q as i64;
        let mut emax = i64::MIN;
        for k in 0..nb {
            if self.m[k] != 0.0 {
                emax = emax.max(self.e[k] as i64 + q * k as i64);
            }
        }
        if emax == i64::MIN {
            return true;
        }
        let mut hk = 1.0;
        let mut sum = 0.0;
        let mut worst = 0.0f64;
        for k in 0..nb {
            if self.m[k] != 0.0 {
                let t = scale2(self.m[k] * hk, self.e[k] as i64 + q * k as i64 - emax);
                sum += t;
                if k + last >= nb {
                    worst = worst.max(t);
                }
            }
            hk *= mh;
        }
        let floor = if atol > 0.0 { scale2(atol, -emax) } else { 0.0 };
        worst <= tail * sum + floor
    }

    /// the same test in plain `f64` products, which underflow (a control)
    #[cfg(feature = "mutate")]
    fn passes_plain(&self, h: f64, last: usize, tail: f64, atol: f64) -> bool {
        let nb = self.m.len();
        let mut hk = 1.0;
        let mut sum = 0.0;
        let mut worst = 0.0f64;
        for k in 0..nb {
            let t = if self.m[k] == 0.0 {
                0.0
            } else {
                scale2(self.m[k], self.e[k] as i64) * hk
            };
            sum += t;
            if k + last >= nb {
                worst = worst.max(t);
            }
            hk *= h;
        }
        worst <= tail * sum + atol
    }
}

/// The width of a step: the largest `h` up to `cap` at which every function passes (module docs)
fn width_of(terms: &[Terms], cap: f64, o: &Opts, tail: f64) -> f64 {
    let ok = |h: f64| {
        #[cfg(feature = "mutate")]
        if o.control == TableControl::Unscaled {
            return terms.iter().all(|t| t.passes_plain(h, o.last, tail, o.atol));
        }
        terms.iter().all(|t| t.passes(h, o.last, tail, o.atol))
    };
    if ok(cap) {
        return cap;
    }
    let mut hi = cap;
    let mut lo = 0.5 * cap;
    while !ok(lo) {
        hi = lo;
        lo *= 0.5;
        if lo < 1e-300 {
            return 0.0;
        }
    }
    for _ in 0..53 {
        let mid = lo + 0.5 * (hi - lo);
        if ok(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The first `kj` coefficients of the polynomial `c` (about its start) about the point `delta` from it: the repeated
/// synthetic division, `c.len() kj` multiply-adds. Entries of `c` past `kj` hold intermediate values on return.
pub fn shift<T: Real>(c: &mut [T], delta: T, kj: usize) {
    let nb = c.len();
    if nb == 0 {
        return;
    }
    for i in 0..kj.min(nb) {
        for m in (i..nb - 1).rev() {
            c[m] = c[m] + delta * c[m + 1];
        }
    }
}

/// The series of F functions about the start of each step of a path, every step cut by the stored functions' own width
#[derive(Clone, Debug)]
pub struct Table<T: Real> {
    dir: f64,
    nb: usize,
    nf: usize,
    start: Vec<f64>,
    end: Vec<f64>,
    width: Vec<f64>,
    /// `c[(step * nf + f) * nb + k]`
    c: Vec<T>,
    #[cfg(feature = "mutate")]
    control: TableControl,
}

impl<T: Real> Table<T> {
    /// The table of `src`'s functions along the path from `xa` to `xb` (either order): each step starts where the last
    /// ended, its series is the source's about the start, and it ends at `min(fraction w, limit)` ([module docs](self)).
    ///
    /// # Panics
    /// A path that is not finite or has no length, `nb` outside 2..=1000, `last` outside 1..=nb, a `tail` that is not in
    /// (0, 1], `fraction` outside (0, 1], an `hmax` that is not positive, a source with no function.
    pub fn build<S: SeriesSource<T>>(src: &S, xa: f64, xb: f64, opts: &Opts) -> Result<Table<T>, Error> {
        assert!(
            xa.is_finite() && xb.is_finite() && xa != xb,
            "Table::build: the path {xa} to {xb} has no length"
        );
        let nb = opts.nb;
        assert!((2..=1000).contains(&nb), "Table::build: nb = {nb} must be in 2..=1000");
        assert!(
            (1..=nb).contains(&opts.last),
            "Table::build: last = {} must be in 1..={nb}",
            opts.last
        );
        assert!(
            opts.tail > 0.0 && opts.tail <= 1.0,
            "Table::build: tail {:e} must be in (0, 1]",
            opts.tail
        );
        assert!(
            opts.atol >= 0.0,
            "Table::build: atol {:e} must not be negative",
            opts.atol
        );
        assert!(
            opts.fraction > 0.0 && opts.fraction <= 1.0,
            "Table::build: fraction {} must be in (0, 1]",
            opts.fraction
        );
        assert!(opts.hmax > 0.0, "Table::build: hmax {:e} must be positive", opts.hmax);
        let nf = src.functions();
        assert!(nf >= 1, "Table::build: the source has no function");
        #[allow(unused_mut)]
        let mut fraction = opts.fraction;
        #[allow(unused_mut)]
        let mut tail = opts.tail;
        #[cfg(feature = "mutate")]
        {
            if opts.control == TableControl::FractionOne {
                fraction = 1.0;
            }
            if opts.control == TableControl::LooseTail {
                tail = 1e-5;
            }
        }
        let dir = if xb > xa { 1.0 } else { -1.0 };
        let mut t = Table {
            dir,
            nb,
            nf,
            start: Vec::new(),
            end: Vec::new(),
            width: Vec::new(),
            c: Vec::new(),
            #[cfg(feature = "mutate")]
            control: opts.control,
        };
        let mut buf = vec![T::zero(); nf * nb];
        let mut x = xa;
        while (xb - x) * dir > 0.0 {
            if t.start.len() >= opts.max_steps {
                return Err(Error::TooManySteps { x });
            }
            let valid = src.series(x, dir, nb, &mut buf);
            for (i, v) in buf.iter().enumerate() {
                if !v.to_f64().is_finite() {
                    return Err(Error::NotFinite {
                        x,
                        f: i / nb,
                        k: i % nb,
                    });
                }
            }
            let rest = (xb - x).abs();
            let to_valid = (valid - x) * dir;
            if to_valid.is_nan() || to_valid <= 0.0 {
                return Err(Error::Stall { x });
            }
            #[allow(unused_mut)]
            let mut limit = rest.min(opts.hmax);
            #[allow(unused_mut)]
            let mut end_valid = to_valid;
            #[cfg(feature = "mutate")]
            if opts.control == TableControl::IgnoreValidTo {
                end_valid = f64::INFINITY;
            }
            limit = limit.min(end_valid);
            let terms: Vec<Terms> = (0..nf).map(|f| Terms::of(&buf[f * nb..(f + 1) * nb])).collect();
            let w = width_of(&terms, limit / fraction, opts, tail);
            #[allow(unused_mut)]
            let mut h = (fraction * w).min(limit);
            #[cfg(feature = "mutate")]
            if opts.control == TableControl::Wide {
                h = limit;
            }
            let x1 = if h >= rest {
                xb
            } else if h >= end_valid {
                valid
            } else {
                x + dir * h
            };
            if x1 == x || (x1 - x) * dir <= 0.0 {
                return Err(Error::Stall { x });
            }
            t.start.push(x);
            t.end.push(x1);
            t.width.push(w);
            t.c.extend_from_slice(&buf);
            x = x1;
        }
        Ok(t)
    }

    /// the number of steps
    pub fn steps(&self) -> usize {
        self.start.len()
    }

    /// the number of functions
    pub fn functions(&self) -> usize {
        self.nf
    }

    /// the coefficients stored per function
    pub fn nb(&self) -> usize {
        self.nb
    }

    /// the direction of the path: +1 or -1
    pub fn direction(&self) -> f64 {
        self.dir
    }

    /// the start (the series' centre) and the end of step `i`, in path order
    pub fn bounds(&self, i: usize) -> (f64, f64) {
        (self.start[i], self.end[i])
    }

    /// The width found at the start of step `i`: the distance within which every stored series passes the tail test, at
    /// most the search cap `limit / fraction`
    pub fn width(&self, i: usize) -> f64 {
        self.width[i]
    }

    /// The stored coefficients `0..nb` of function `f` about the start of step `i`
    pub fn series(&self, i: usize, f: usize) -> &[T] {
        let o = (i * self.nf + f) * self.nb;
        &self.c[o..o + self.nb]
    }

    /// The step that holds `x`, a boundary belonging to the step that starts there; `None` outside the path (and for a
    /// NaN)
    pub fn find(&self, x: f64) -> Option<usize> {
        let d = self.dir;
        let k = self.start.partition_point(|&s| (x - s) * d >= 0.0);
        let n = self.start.len();
        if k == 0 || (self.end[n - 1] - x) * d < 0.0 {
            return None;
        }
        Some(k - 1)
    }

    /// The distance from the start of step `i` to `x0`, in the lane
    fn delta(&self, i: usize, x0: f64) -> T {
        #[cfg(feature = "mutate")]
        match self.control {
            TableControl::NoShift => return T::zero(),
            TableControl::ShiftF64 => return T::from_f64(x0 - self.start[i]),
            _ => {}
        }
        T::from_f64(x0) - T::from_f64(self.start[i])
    }

    /// The jets of function `f` about `x0`: coefficients `0..kj` of the stored series of the step that holds `x0`,
    /// shifted by `x0 - start` (exact in the lane's type where the lane is wider than `f64`) into `out[..kj]`.
    ///
    /// # Panics
    /// `x0` outside the path, `f` outside the functions, `kj` outside 1..=nb, `out` shorter than `kj`.
    pub fn shifted(&self, x0: f64, f: usize, kj: usize, out: &mut [T]) {
        assert!(f < self.nf, "Table::shifted: function {f} of {}", self.nf);
        assert!(
            (1..=self.nb).contains(&kj),
            "Table::shifted: kj = {kj} must be in 1..={}",
            self.nb
        );
        assert!(
            out.len() >= kj,
            "Table::shifted: out holds {} coefficients, not {kj}",
            out.len()
        );
        let i = self
            .find(x0)
            .unwrap_or_else(|| panic!("Table::shifted: x0 = {x0} is outside the table"));
        let mut buf = self.series(i, f).to_vec();
        shift(&mut buf, self.delta(i, x0), kj);
        out[..kj].copy_from_slice(&buf[..kj]);
    }

    /// How far a step from `x0` stays inside the converged width of its table step: `width - |x0 - start|`, at least 0
    ///
    /// # Panics
    /// `x0` outside the path.
    pub fn headroom(&self, x0: f64) -> f64 {
        let i = self
            .find(x0)
            .unwrap_or_else(|| panic!("Table::headroom: x0 = {x0} is outside the table"));
        (self.width[i] - (x0 - self.start[i]).abs()).max(0.0)
    }

    /// The same table in another scalar, each coefficient through `conv` (a table built in `DD` and used in `f64`:
    /// `|x| f64::from_dd(x)`)
    pub fn map<U: Real>(&self, conv: impl Fn(T) -> U) -> Table<U> {
        Table {
            dir: self.dir,
            nb: self.nb,
            nf: self.nf,
            start: self.start.clone(),
            end: self.end.clone(),
            width: self.width.clone(),
            c: self.c.iter().map(|&x| conv(x)).collect(),
            #[cfg(feature = "mutate")]
            control: self.control,
        }
    }

    /// Sets the deliberate error of the lookup (`NoShift`, `ShiftF64`; the others act when the table is built).
    #[cfg(feature = "mutate")]
    pub fn set_control(&mut self, c: TableControl) {
        self.control = c;
    }
}

/// A table as the jets of a plane stepper: function `e` of the table is the entry `pattern[e]` of J
#[derive(Clone, Debug)]
pub struct TableJets<'a, T: Real> {
    table: &'a Table<T>,
    pattern: Vec<(usize, usize)>,
}

impl<'a, T: Real> TableJets<'a, T> {
    /// # Panics
    /// A pattern that does not have one entry per function of the table.
    pub fn new(table: &'a Table<T>, pattern: Vec<(usize, usize)>) -> Self {
        assert_eq!(
            pattern.len(),
            table.functions(),
            "TableJets::new: {} pattern entries for {} functions",
            pattern.len(),
            table.functions()
        );
        TableJets { table, pattern }
    }
}

impl<T: Real> JetSource<T> for TableJets<'_, T> {
    fn pattern(&self) -> Vec<(usize, usize)> {
        self.pattern.clone()
    }

    fn jets(&self, x0: f64, kj: usize, out: &mut Jets<T>) {
        for (e, en) in out.ent.iter_mut().enumerate() {
            self.table.shifted(x0, e, kj, &mut en.c[..kj]);
            en.n = kj;
        }
    }
}
