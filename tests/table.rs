//! Taylor tables (`multishoot::table`) on planted functions with known singularities, in f64, double-double and
//! quad-double (octo-double: one smoke case). The references are the functions' closed forms in the lane (rational
//! functions, so a quotient or a power by repeated multiplication) and their exact coefficients by recurrences; none
//! shares code with the table. Pass criteria, written before the first run (the planted cases, their lanes and the
//! prototype that fixed the step lists are in negative-mass-raw `I192-mac`: README.md, design.md, proto_table.py), with
//! `tail` the lane's table setting (f64 (nb 20, 1e-13), dd (30, 1e-25), qd (60, 1e-55), od (100, 1e-115)), `last` 4,
//! `fraction` 0.3:
//!
//! * **C1, the jets.** For every step, every function, x0 = start + t len (t in 0, 1/4, 1/2, 3/4 and the last f64 before
//!   the end) and h = u max(0, min(0.7 w, V - x0)) (u in 0, 1/4, 1/2, 3/4, 1; w the width the table reports, V the
//!   source's `valid_to` at the step start; V - x0 in the path's direction): with the jets of the function about x0
//!   from `Table::shifted`, `|sum_k jets_k h^k - f(x0 + h)| / sum_k |jets_k| h^k <= 10 tail`.
//! * **C2, the structure.** The steps are contiguous (end == next start, bitwise), the first starts at xa and the last
//!   ends at xb, each runs in the path's direction, none ends past `valid_to`, and with w recomputed from the stored
//!   series by a bisection of its own (logarithms, not the library's scaled products) at the nominal fraction, tail and
//!   `last`: `len <= 0.3 w (1 + 1e-12)` always and `len >= 0.3 w (1 - 1e-6)` unless the step is cut by `valid_to`,
//!   `hmax` or xb.
//! * **C3, an independent step list.** The count and every 20th step start of each case and lane equal those of the
//!   prototype (`table_steps.in`, 90-digit decimal arithmetic) to 1e-9 relative.
//! * **C4, a plane through the table (PH).** `y' = (M / x) y`, M = P diag(-1, -2, -3, -4) P^-1 (integer), the plane
//!   P[:, :2], x from 1e-3 to 40, the table of the entries of M / x with nb = kt + 12 and tail = eps, `Plane::advance` at
//!   its default plan (tol 100 eps, kt = `Plan::optimal_kt`, kj = kt): `||(I - Pi) Y||_F <= 1e4 eps` and
//!   `|ln_vol - exact| <= 1e4 eps max(1, |exact|)`, exact = (1/2) ln 5 - 3 ln(40 / 1e-3).
//! * **C5, the errors.** `1/(x-5)` on [0, 10] gives `Err(Stall)` at an x within 1e-9 of 5 and below it (**met in f64 only**:
//!   in dd and qd the coefficient of order nb - 1 overflows first and `NotFinite` is returned, recorded in its own test);
//!   with the pole moved to 15 the table is built; a NaN coefficient (function 2, order 7) gives `NotFinite { f: 2, k: 7 }`.
//!
//! The planted cases: PA `1/x^2, 1/x, 1/(30-x)` on [1e-3, 25] with source pieces of length 6; PB `1/(30-x) + 3 (x-5)_+^2`
//! on [0, 20] (`valid_to` 5, a piece boundary the series cannot cross); PC `1/((x-12)^2 + 0.04)` on [10, 14] (a conjugate
//! pair ahead); PD `(1 + x/4)^40` on [0, 3] (no singularity: the width is set by nb); PE `1/(1-x^3)` on [0, 0.9] with
//! `hmax` 0.5 (the first series has c_k = 0 unless k is a multiple of 3: `last` 2 sees only zeros); PF PA backward; PP
//! `1 + 2x + 3x^2 - x^4` on [0.1, 3] (one step: an exact polynomial from a start that is not dyadic); PS `1/(1 + 1e5 x)`
//! on [0, 1] in qd (c_58 = 1e290 and h^58 underflows in plain f64 products).
//!
//! The controls (feature `mutate`; each is run once and must fail its targets, the targets having been declared from the
//! prototype's development runs): `Wide`, `FractionOne`, `LooseTail`, `IgnoreValidTo`, `Unscaled`, `NoShift`, `ShiftF64`
//! (silent on every case but PP in dd and qd, where the shift is longer than the start: the f64 difference of two f64
//! numbers within a factor of 2 is exact), and the wrong inputs `ref_scale` (the reference times 1 + 1e-4) and
//! `pole_off_path` (C5's).
use multishoot::multidouble::dd::DD;
use multishoot::multidouble::od::OD;
use multishoot::multidouble::qd::QD;
use multishoot::plane::{Plane, Real};
use multishoot::table::{shift, Error, Opts, SeriesSource, Table, TableJets};
use multishoot::taylor::Plan;

#[cfg(feature = "mutate")]
use multishoot::control::TableControl;

include!("table_steps.in");

/// the lane's parts, and a number from a decimal string
trait Lane: Real {
    const NAME: &'static str;
    /// coefficients per function, the tail test's `tail`
    const NB: usize;
    const TAIL: f64;
    fn parts(self) -> Vec<f64>;
    fn parse(s: &str) -> Self;
}

impl Lane for f64 {
    const NAME: &'static str = "f64";
    const NB: usize = 20;
    const TAIL: f64 = 1e-13;
    fn parts(self) -> Vec<f64> {
        vec![self]
    }
    fn parse(s: &str) -> f64 {
        s.parse().unwrap()
    }
}

impl Lane for DD {
    const NAME: &'static str = "dd";
    const NB: usize = 30;
    const TAIL: f64 = 1e-25;
    fn parts(self) -> Vec<f64> {
        vec![self.hi, self.lo]
    }
    fn parse(s: &str) -> DD {
        QD::parse(s).unwrap().to_dd()
    }
}

impl Lane for QD {
    const NAME: &'static str = "qd";
    const NB: usize = 60;
    const TAIL: f64 = 1e-55;
    fn parts(self) -> Vec<f64> {
        self.x.to_vec()
    }
    fn parse(s: &str) -> QD {
        QD::parse(s).unwrap()
    }
}

impl Lane for OD {
    const NAME: &'static str = "od";
    const NB: usize = 100;
    const TAIL: f64 = 1e-115;
    fn parts(self) -> Vec<f64> {
        self.x.to_vec()
    }
    fn parse(s: &str) -> OD {
        OD::parse(s).unwrap()
    }
}

/// (1/2) ln 5 to 140 digits (mpmath)
const HALF_LN5: &str = "0.80471895621705018730037966661309381976280067713425886095632394573708949385382888231506693904658980539998315151085778144986200261466233809982";

// ---------------------------------------------------------------- the planted functions

/// A planted function: its coefficients about any point and its value, in the lane
#[derive(Clone, Copy, Debug, PartialEq)]
enum Func {
    /// 1 / x^2
    Inv2,
    /// 1 / x
    Inv1,
    /// 1 / (30 - x)
    Far30,
    /// 1 / (30 - x) + 3 max(x - 5, 0)^2
    Ramp,
    /// 1 / ((x - 12)^2 + 0.04)
    Pair,
    /// (1 + x / 4)^40
    Poly40,
    /// 1 / (1 - x^3)
    Cube,
    /// 1 / (x - 5)
    Pole5,
    /// 1 + 2 x + 3 x^2 - x^4
    Pp,
    /// 1 / (1 + 1e5 x)
    Ps,
    /// m / x
    Scaled(f64),
}

impl Func {
    fn coef<T: Lane>(self, x0: f64, out: &mut [T]) {
        let nb = out.len();
        let c = T::from_f64;
        let x = c(x0);
        match self {
            Func::Inv2 => {
                out[0] = (x * x).recip();
                for k in 0..nb - 1 {
                    out[k + 1] = -out[k] * c((k + 2) as f64) / (c((k + 1) as f64) * x);
                }
            }
            Func::Inv1 => {
                out[0] = x.recip();
                for k in 0..nb - 1 {
                    out[k + 1] = -out[k] / x;
                }
            }
            Func::Scaled(m) => {
                out[0] = c(m) * x.recip();
                for k in 0..nb - 1 {
                    out[k + 1] = -out[k] / x;
                }
            }
            Func::Far30 | Func::Ramp => {
                let d = c(30.0) - x;
                out[0] = d.recip();
                for k in 0..nb - 1 {
                    out[k + 1] = out[k] / d;
                }
                if self == Func::Ramp && x0 >= 5.0 {
                    let r = x - c(5.0);
                    out[0] = out[0] + c(3.0) * r * r;
                    out[1] = out[1] + c(6.0) * r;
                    out[2] = out[2] + c(3.0);
                }
            }
            Func::Pair => {
                let d = x - c(12.0);
                let q = d * d + c(0.04);
                out[0] = q.recip();
                out[1] = -(c(2.0) * d * out[0]) / q;
                for k in 2..nb {
                    out[k] = (-(c(2.0) * d * out[k - 1]) - out[k - 2]) / q;
                }
            }
            Func::Poly40 => {
                let u = c(1.0) + x / c(4.0);
                let mut p = c(1.0);
                for _ in 0..40 {
                    p = p * u;
                }
                out[0] = p;
                for k in 0..nb - 1 {
                    out[k + 1] = out[k] * c((40.0 - k as f64).max(0.0)) / (c((k + 1) as f64) * c(4.0) * u);
                }
            }
            Func::Cube => {
                let den = c(1.0) - x * x * x;
                for k in 0..nb {
                    let mut r = if k == 0 { c(1.0) } else { c(0.0) };
                    if k >= 1 {
                        r = r + c(3.0) * x * x * out[k - 1];
                    }
                    if k >= 2 {
                        r = r + c(3.0) * x * out[k - 2];
                    }
                    if k >= 3 {
                        r = r + out[k - 3];
                    }
                    out[k] = r / den;
                }
            }
            Func::Pole5 => {
                let d = x - c(5.0);
                out[0] = d.recip();
                for k in 0..nb - 1 {
                    out[k + 1] = -out[k] / d;
                }
            }
            Func::Pp => {
                out.iter_mut().for_each(|v| *v = c(0.0));
                out[0] = c(1.0) + c(2.0) * x + c(3.0) * x * x - x * x * x * x;
                out[1] = c(2.0) + c(6.0) * x - c(4.0) * x * x * x;
                out[2] = c(3.0) - c(6.0) * x * x;
                out[3] = -(c(4.0) * x);
                out[4] = -c(1.0);
            }
            Func::Ps => {
                let den = c(1.0) + c(1e5) * x;
                out[0] = den.recip();
                for k in 0..nb - 1 {
                    out[k + 1] = -(c(1e5) * out[k]) / den;
                }
            }
        }
    }

    fn val<T: Lane>(self, x: T) -> T {
        let c = T::from_f64;
        match self {
            Func::Inv2 => (x * x).recip(),
            Func::Inv1 => x.recip(),
            Func::Scaled(m) => c(m) * x.recip(),
            Func::Far30 => (c(30.0) - x).recip(),
            Func::Ramp => {
                let base = (c(30.0) - x).recip();
                if x.to_f64() > 5.0 {
                    let r = x - c(5.0);
                    base + c(3.0) * r * r
                } else {
                    base
                }
            }
            Func::Pair => ((x - c(12.0)) * (x - c(12.0)) + c(0.04)).recip(),
            Func::Poly40 => {
                let u = c(1.0) + x / c(4.0);
                (0..40).fold(c(1.0), |p, _| p * u)
            }
            Func::Cube => (c(1.0) - x * x * x).recip(),
            Func::Pole5 => (x - c(5.0)).recip(),
            Func::Pp => c(1.0) + c(2.0) * x + c(3.0) * x * x - x * x * x * x,
            Func::Ps => (c(1.0) + c(1e5) * x).recip(),
        }
    }
}

/// Where a source's series stop holding
#[derive(Clone, Copy, Debug, PartialEq)]
enum Valid {
    /// nowhere
    Never,
    /// at the ends of pieces of length 6
    Pieces6,
    /// at 5, before which the ramp's jets are zero
    Ramp5,
}

#[derive(Clone, Debug)]
struct Src {
    funcs: Vec<Func>,
    valid: Valid,
    /// a NaN in function f at order k (C5)
    nan: Option<(usize, usize)>,
}

impl Src {
    fn valid_to(&self, x0: f64, dir: f64) -> f64 {
        match self.valid {
            Valid::Never => dir * f64::INFINITY,
            Valid::Pieces6 => {
                if dir > 0.0 {
                    (x0 / 6.0).floor() * 6.0 + 6.0
                } else {
                    (x0 / 6.0).ceil() * 6.0 - 6.0
                }
            }
            Valid::Ramp5 => {
                assert!(dir > 0.0);
                if x0 < 5.0 {
                    5.0
                } else {
                    f64::INFINITY
                }
            }
        }
    }
}

impl<T: Lane> SeriesSource<T> for Src {
    fn functions(&self) -> usize {
        self.funcs.len()
    }
    fn series(&self, x0: f64, dir: f64, nb: usize, out: &mut [T]) -> f64 {
        for (f, fun) in self.funcs.iter().enumerate() {
            fun.coef(x0, &mut out[f * nb..(f + 1) * nb]);
        }
        if let Some((f, k)) = self.nan {
            out[f * nb + k] = T::from_f64(f64::NAN);
        }
        self.valid_to(x0, dir)
    }
}

// ---------------------------------------------------------------- the cases

struct Case {
    name: &'static str,
    funcs: Vec<Func>,
    valid: Valid,
    xa: f64,
    xb: f64,
    hmax: f64,
    lanes: &'static [&'static str],
}

impl Case {
    fn src(&self) -> Src {
        Src {
            funcs: self.funcs.clone(),
            valid: self.valid,
            nan: None,
        }
    }
}

const ALL: &[&str] = &["f64", "dd", "qd"];

fn pa_funcs() -> Vec<Func> {
    vec![Func::Inv2, Func::Inv1, Func::Far30]
}

fn cases() -> Vec<Case> {
    let case = |name, funcs, valid, xa, xb, hmax, lanes| Case {
        name,
        funcs,
        valid,
        xa,
        xb,
        hmax,
        lanes,
    };
    let inf = f64::INFINITY;
    vec![
        case("PA", pa_funcs(), Valid::Pieces6, 1e-3, 25.0, inf, ALL),
        case("PB", vec![Func::Ramp], Valid::Ramp5, 0.0, 20.0, inf, ALL),
        case("PC", vec![Func::Pair], Valid::Never, 10.0, 14.0, inf, ALL),
        case("PD", vec![Func::Poly40], Valid::Never, 0.0, 3.0, inf, ALL),
        case("PE", vec![Func::Cube], Valid::Never, 0.0, 0.9, 0.5, &["dd", "qd"]),
        case("PF", pa_funcs(), Valid::Pieces6, 25.0, 1e-3, inf, ALL),
        case("PP", vec![Func::Pp], Valid::Never, 0.1, 3.0, inf, ALL),
        case("PS", vec![Func::Ps], Valid::Never, 0.0, 1.0, inf, &["qd"]),
    ]
}

fn case_named(name: &str) -> Case {
    cases().into_iter().find(|c| c.name == name).unwrap()
}

/// The deliberate errors, with the library's under the feature `mutate`
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)]
enum Ctl {
    None,
    Wide,
    FractionOne,
    LooseTail,
    IgnoreValidTo,
    Unscaled,
    NoShift,
    ShiftF64,
}

#[cfg(feature = "mutate")]
impl Ctl {
    fn lib(self) -> TableControl {
        match self {
            Ctl::None => TableControl::None,
            Ctl::Wide => TableControl::Wide,
            Ctl::FractionOne => TableControl::FractionOne,
            Ctl::LooseTail => TableControl::LooseTail,
            Ctl::IgnoreValidTo => TableControl::IgnoreValidTo,
            Ctl::Unscaled => TableControl::Unscaled,
            Ctl::NoShift => TableControl::NoShift,
            Ctl::ShiftF64 => TableControl::ShiftF64,
        }
    }
}

/// The options of a lane for a case; the control goes to the build or, for the two of the lookup, to the table
fn opts_for<T: Lane>(case: &Case, last: usize, ctl: Ctl) -> Opts {
    let mut o = Opts::new(T::NB, T::TAIL);
    o.last = last;
    o.hmax = case.hmax;
    #[cfg(feature = "mutate")]
    {
        o.control = ctl.lib();
    }
    #[cfg(not(feature = "mutate"))]
    assert_eq!(ctl, Ctl::None);
    o
}

fn build<T: Lane>(case: &Case, last: usize, ctl: Ctl) -> Table<T> {
    let o = opts_for::<T>(case, last, ctl);
    #[allow(unused_mut)]
    let mut t = Table::<T>::build(&case.src(), case.xa, case.xb, &o)
        .unwrap_or_else(|e| panic!("{} {}: {e}", case.name, T::NAME));
    #[cfg(feature = "mutate")]
    t.set_control(ctl.lib());
    t
}

// ---------------------------------------------------------------- C1

const T_SAMPLES: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

/// the largest E of C1 (NaN if any is) and where; `scale` multiplies the reference (the wrong input `ref_scale`)
fn c1<T: Lane>(case: &Case, tab: &Table<T>, scale: f64) -> (f64, String) {
    let src = case.src();
    let nb = tab.nb();
    let dir = tab.direction();
    let mut worst = 0.0f64;
    let mut at = String::from("-");
    let mut jets = vec![T::zero(); nb];
    for i in 0..tab.steps() {
        let (s, e) = tab.bounds(i);
        let v = src.valid_to(s, dir);
        let w = tab.width(i);
        for &ts in &T_SAMPLES {
            let x0 = if ts == 0.0 {
                s
            } else if ts == 1.0 {
                // the last f64 inside the step
                if dir > 0.0 {
                    e.next_down()
                } else {
                    e.next_up()
                }
            } else {
                s + ts * (e - s)
            };
            for (f, fun) in case.funcs.iter().enumerate() {
                tab.shifted(x0, f, nb, &mut jets);
                let room = ((v - x0) * dir).max(0.0);
                let hm = (0.7 * w).min(room);
                for &us in &T_SAMPLES {
                    let h = us * hm;
                    let tt = T::from_f64(dir * h);
                    let p = jets.iter().rev().fold(T::zero(), |p, &c| p * tt + c);
                    let mut sc = 0.0;
                    let mut hk = 1.0;
                    for c in &jets {
                        sc += c.to_f64().abs() * hk;
                        hk *= h;
                    }
                    let r = fun.val(T::from_f64(x0) + tt) * T::from_f64(scale);
                    let d = (p - r).to_f64().abs();
                    let err = if sc != 0.0 { d / sc } else { d };
                    if err.is_nan() {
                        worst = f64::NAN;
                        at = format!("step {i} f {f} x0 {x0} h {h}: NaN");
                    } else if !worst.is_nan() && err > worst {
                        worst = err;
                        at = format!("step {i} f {f} x0 {x0:.17e} h {h:.3e}");
                    }
                }
            }
        }
    }
    (worst, at)
}

// ---------------------------------------------------------------- C2

/// the width by the test's own bisection, in logarithms: the largest h up to `cap` with `t_k <= tail S` for the last
/// `last` orders of every series
fn width_log(series: &[Vec<f64>], cap: f64, tail: f64, last: usize) -> f64 {
    let ok = |h: f64| {
        let lh = h.ln();
        series.iter().all(|c| {
            let nb = c.len();
            let lt: Vec<f64> = c
                .iter()
                .enumerate()
                .map(|(k, x)| {
                    if *x == 0.0 {
                        f64::NEG_INFINITY
                    } else {
                        x.abs().ln() + k as f64 * lh
                    }
                })
                .collect();
            let lmax = lt.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if lmax == f64::NEG_INFINITY {
                return true;
            }
            let sum: f64 = lt.iter().map(|l| (l - lmax).exp()).sum();
            let worst = lt[nb - last..].iter().map(|l| (l - lmax).exp()).fold(0.0, f64::max);
            worst <= tail * sum
        })
    };
    if ok(cap) {
        return cap;
    }
    let mut hi = cap.ln();
    let mut lo = hi - 1.0;
    while !ok(lo.exp()) {
        hi = lo;
        lo -= 1.0;
        if lo < -700.0 {
            return 0.0;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if ok(mid.exp()) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo.exp()
}

/// the violations of C2 (none: the structure is right), at the nominal fraction 0.3, `tail` and `last`
fn c2<T: Lane>(case: &Case, tab: &Table<T>, last: usize) -> Vec<String> {
    let src = case.src();
    let dir = tab.direction();
    let n = tab.steps();
    let mut v = vec![];
    for i in 0..n {
        let (s, e) = tab.bounds(i);
        if i == 0 && s != case.xa {
            v.push(format!("first start {s} is not xa"));
        }
        if i + 1 < n && e != tab.bounds(i + 1).0 {
            v.push(format!(
                "step {i} ends at {e}, the next starts at {}",
                tab.bounds(i + 1).0
            ));
        }
        if i == n - 1 && e != case.xb {
            v.push(format!("last end {e} is not xb"));
        }
        if (e - s) * dir <= 0.0 {
            v.push(format!("step {i} [{s}, {e}] runs against the path"));
            continue;
        }
        let vt = src.valid_to(s, dir);
        if (e - vt) * dir > 0.0 {
            v.push(format!("step {i} ends at {e}, past valid_to {vt}"));
        }
        let limit = (case.xb - s).abs().min(case.hmax).min((vt - s) * dir);
        let cap = limit / 0.3;
        let series: Vec<Vec<f64>> = (0..tab.functions())
            .map(|f| tab.series(i, f).iter().map(|x| x.to_f64()).collect())
            .collect();
        let w = width_log(&series, cap, T::TAIL, last).min(cap);
        let len = (e - s).abs();
        if len > 0.3 * w * (1.0 + 1e-12) {
            v.push(format!("step {i}: length {len:e} > 0.3 w = {:e} (w {w:e})", 0.3 * w));
        }
        let cut = e == case.xb || e == vt || len >= case.hmax * (1.0 - 1e-12);
        if !cut && len < 0.3 * w * (1.0 - 1e-6) {
            v.push(format!(
                "step {i}: length {len:e} < 0.3 w = {:e} (w {w:e}), not cut",
                0.3 * w
            ));
        }
    }
    v
}

// ---------------------------------------------------------------- C3 and the digest

/// the violations of C3 against the records `rows` (a missing record is one: C3 requires a record for every case and lane
/// it covers)
fn c3_rows<T: Lane>(rows: &[StepRow], name: &str, tab: &Table<T>) -> Vec<String> {
    let mut v = vec![];
    let Some(&(_, _, count, samples)) = rows.iter().find(|r| r.0 == name && r.1 == T::NAME) else {
        v.push(format!("{name} {}: no record in the prototype's step lists", T::NAME));
        return v;
    };
    if tab.steps() != count {
        v.push(format!(
            "{name} {}: {} steps, the prototype's list has {count}",
            T::NAME,
            tab.steps()
        ));
        return v;
    }
    for &(i, x) in samples {
        let s = tab.bounds(i).0;
        let err = if x == 0.0 { s.abs() } else { ((s - x) / x).abs() };
        if err > 1e-9 {
            v.push(format!(
                "{name} {}: step {i} starts at {s:.17e}, the prototype's {x:.17e}",
                T::NAME
            ));
        }
    }
    v
}

/// C3 against `table_steps.in`
fn c3<T: Lane>(name: &str, tab: &Table<T>) -> Vec<String> {
    c3_rows(STEPS, name, tab)
}

/// FNV-1a over the bits of every step point and every part of every coefficient
fn digest<T: Lane>(tab: &Table<T>) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut eat = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for i in 0..tab.steps() {
        let (s, e) = tab.bounds(i);
        eat(s);
        eat(e);
        eat(tab.width(i));
        for f in 0..tab.functions() {
            for c in tab.series(i, f) {
                c.parts().into_iter().for_each(&mut eat);
            }
        }
    }
    h
}

/// the digests of the planted tables at `last` 4, as printed on aarch64 NEON; the same bits on every backend
const DIGESTS: &[(&str, &str, u64)] = &[
    ("PA", "dd", 0x6939d511435ecced),
    ("PA", "f64", 0x280f62a3ab45ae7a),
    ("PA", "qd", 0xa2bbea4542aea07b),
    ("PB", "dd", 0xa994767068eb1a1c),
    ("PB", "f64", 0x899abaa14b883dc3),
    ("PB", "qd", 0x179001490fee523e),
    ("PC", "dd", 0x365da269c7e2948d),
    ("PC", "f64", 0xd151afba86408879),
    ("PC", "qd", 0x3ce8f712140f1d2c),
    ("PD", "dd", 0x54072c80538c44cf),
    ("PD", "f64", 0x799052772125188b),
    ("PD", "qd", 0x8af916f973af2bc8),
    ("PE", "dd", 0x6b406f74f5e49477),
    ("PE", "qd", 0xddf7c37de676db5e),
    ("PF", "dd", 0xa274ccd7001851bf),
    ("PF", "f64", 0xaa43a114b56a5755),
    ("PF", "qd", 0x40c69cd73f665084),
    ("PP", "dd", 0x014221d6d8a975d8),
    ("PP", "f64", 0x366eaf099a2e72de),
    ("PP", "qd", 0xd3ad435d9866e420),
    ("PS", "qd", 0xea5fd49419737db1),
];

// ---------------------------------------------------------------- one run

struct Run {
    steps: usize,
    c1: f64,
    c1_at: String,
    c2: Vec<String>,
    c3: Vec<String>,
    digest: u64,
}

fn run<T: Lane>(case: &Case, last: usize, ctl: Ctl) -> Run {
    let tab = build::<T>(case, last, ctl);
    let (c1v, c1_at) = c1(case, &tab, 1.0);
    Run {
        steps: tab.steps(),
        c1: c1v,
        c1_at,
        c2: c2(case, &tab, last),
        c3: if last == 4 && ctl == Ctl::None {
            c3(case.name, &tab)
        } else {
            vec![]
        },
        digest: digest(&tab),
    }
}

fn bar<T: Lane>() -> f64 {
    10.0 * T::TAIL
}

/// whether E is above the bar, or not a number
fn exceeds(e: f64, bar: f64) -> bool {
    e.is_nan() || e > bar
}

fn report<T: Lane>(case: &Case, last: usize, r: &Run) {
    println!(
        "{} {} last={last}: steps {}, C1 maxE {:.3e} (bar {:.1e}) at {}, C2 violations {}, C3 violations {}, digest {:016x}",
        case.name,
        T::NAME,
        r.steps,
        r.c1,
        bar::<T>(),
        r.c1_at,
        r.c2.len(),
        r.c3.len(),
        r.digest
    );
}

/// every planted case of the lane at the default `last`: C1, C2 and C3 pass; returns the digests
fn planted<T: Lane>() -> Vec<(String, u64)> {
    let mut out = vec![];
    for case in cases().iter().filter(|c| c.lanes.contains(&T::NAME)) {
        let r = run::<T>(case, 4, Ctl::None);
        report::<T>(case, 4, &r);
        assert!(
            r.c1 <= bar::<T>(),
            "{} {}: C1 {:e} > {:e} at {}",
            case.name,
            T::NAME,
            r.c1,
            bar::<T>(),
            r.c1_at
        );
        assert!(
            r.c2.is_empty(),
            "{} {}: C2 {:?}",
            case.name,
            T::NAME,
            &r.c2[..r.c2.len().min(5)]
        );
        assert!(
            r.c3.is_empty(),
            "{} {}: C3 {:?}",
            case.name,
            T::NAME,
            &r.c3[..r.c3.len().min(5)]
        );
        out.push((case.name.to_string(), r.digest));
    }
    out
}

fn check_digests<T: Lane>(got: &[(String, u64)]) {
    for (name, d) in got {
        let &(_, _, want) = DIGESTS
            .iter()
            .find(|r| r.0 == name && r.1 == T::NAME)
            .unwrap_or_else(|| panic!("{name} {}: no digest recorded", T::NAME));
        assert_eq!(*d, want, "{name} {}: digest {d:016x}, expected {want:016x}", T::NAME);
    }
}

#[test]
fn planted_f64() {
    check_digests::<f64>(&planted::<f64>());
}

#[test]
fn planted_dd() {
    check_digests::<DD>(&planted::<DD>());
}

#[test]
fn planted_qd() {
    check_digests::<QD>(&planted::<QD>());
}

/// PC in octo-double, a smoke test of the lane (nb 100, tail 1e-115)
#[test]
fn planted_od_pc() {
    let case = case_named("PC");
    let r = run::<OD>(&case, 4, Ctl::None);
    report::<OD>(&case, 4, &r);
    assert!(r.c1 <= bar::<OD>(), "PC od: C1 {:e} at {}", r.c1, r.c1_at);
    assert!(r.c2.is_empty(), "PC od: C2 {:?}", &r.c2[..r.c2.len().min(5)]);
}

/// PE: the first series has c_k = 0 unless k is a multiple of 3, so `last` 2 sees zeros and steps to `hmax`, past the
/// pole (C1 fails, or is not a number at x = 1); `last` 3 and 4 see c_27 and pass
fn pe<T: Lane>() {
    let case = case_named("PE");
    for last in [2usize, 3, 4] {
        let r = run::<T>(&case, last, Ctl::None);
        report::<T>(&case, last, &r);
        let pass = r.c1 <= bar::<T>();
        assert_eq!(pass, last >= 3, "PE {} last {last}: C1 {:e}", T::NAME, r.c1);
        assert!(
            r.c2.is_empty(),
            "PE {} last {last}: C2 {:?}",
            T::NAME,
            &r.c2[..r.c2.len().min(5)]
        );
    }
}

#[test]
fn pe_last_dd() {
    pe::<DD>();
}

#[test]
fn pe_last_qd() {
    pe::<QD>();
}

// ---------------------------------------------------------------- C5

/// `1/(x-5)` on [0, 10]: the table runs into the pole
fn pg<T: Lane>() -> Result<Table<T>, Error> {
    let src = Src {
        funcs: vec![Func::Pole5],
        valid: Valid::Never,
        nan: None,
    };
    Table::<T>::build(&src, 0.0, 10.0, &Opts::new(T::NB, T::TAIL))
}

/// C5 as written: `Err(Stall)` at an x below 5 and within 1e-9 of it. It holds in f64, whose coefficients of order 19
/// are finite where a step no longer moves x; **in dd and qd it is not met** (below): the coefficient of order nb - 1
/// leaves the range of f64 first.
#[test]
fn pole_on_the_path_f64() {
    match pg::<f64>() {
        Err(Error::Stall { x }) => {
            println!("PG f64: Stall at {x:.17}");
            assert!(x < 5.0 && 5.0 - x < 1e-9, "stalled at {x}");
        }
        other => panic!("PG f64: {other:?}"),
    }
}

/// What dd and qd do instead (a recorded result, not the criterion): the series about x holds `d^-(k+1)` at order k for
/// a pole at distance d, so the coefficient of order nb - 1 overflows `f64` (the range every part of a multidouble
/// shares) at d = f64::MAX^(-1/nb): 5.6e-11 at nb 30, 7.4e-6 at nb 60, and `NotFinite` is returned there, before a step
/// fails to move x. The error names the function and the order; x is below the pole within ten times that distance.
#[test]
fn pole_on_the_path_wide_lanes_recorded() {
    fn go<T: Lane>() {
        let d_ovf = f64::MAX.powf(-1.0 / T::NB as f64);
        match pg::<T>() {
            Err(Error::NotFinite { x, f: 0, k }) => {
                println!(
                    "PG {}: NotFinite at x = {x:.17}, order {k}; 5 - x = {:.3e}, overflow distance {d_ovf:.3e}",
                    T::NAME,
                    5.0 - x
                );
                assert!(x < 5.0 && 5.0 - x <= 10.0 * d_ovf, "{}: stopped at {x}", T::NAME);
                assert_eq!(k, T::NB - 1);
            }
            other => panic!("PG {}: {other:?}", T::NAME),
        }
    }
    go::<DD>();
    go::<QD>();
}

/// the check's input broken once (the pole moved to 15: no error), and a NaN coefficient
#[test]
fn pole_off_the_path_and_nan() {
    struct Moved;
    impl<U: Lane> SeriesSource<U> for Moved {
        fn functions(&self) -> usize {
            1
        }
        fn series(&self, x0: f64, dir: f64, nb: usize, out: &mut [U]) -> f64 {
            let d = U::from_f64(x0 - 15.0);
            out[0] = d.recip();
            for k in 0..nb - 1 {
                out[k + 1] = -out[k] / d;
            }
            dir * f64::INFINITY
        }
    }
    fn go<T: Lane>() {
        let t = Table::<T>::build(&Moved, 0.0, 10.0, &Opts::new(T::NB, T::TAIL)).expect("pole at 15: no error");
        println!("PG {} with the pole at 15: {} steps", T::NAME, t.steps());
        // a NaN at function 2, order 7
        let nan = Src {
            funcs: vec![Func::Inv1, Func::Far30, Func::Pp],
            valid: Valid::Never,
            nan: Some((2, 7)),
        };
        match Table::<T>::build(&nan, 1.0, 2.0, &Opts::new(T::NB, T::TAIL)) {
            Err(Error::NotFinite { f: 2, k: 7, .. }) => {}
            other => panic!("NaN {}: {other:?}", T::NAME),
        }
    }
    go::<f64>();
    go::<DD>();
    go::<QD>();
}

// ---------------------------------------------------------------- the API

#[test]
fn shift_of_a_polynomial_is_exact() {
    // (1 + t)^5 shifted by 2 is (3 + t)^5: coefficients C(5, k) 3^(5 - k)
    let mut c = vec![1.0, 5.0, 10.0, 10.0, 5.0, 1.0f64];
    shift(&mut c, 2.0, 6);
    assert_eq!(c, vec![243.0, 405.0, 270.0, 90.0, 15.0, 1.0]);
    // only the first kj are final
    let mut d = vec![1.0, 5.0, 10.0, 10.0, 5.0, 1.0f64];
    shift(&mut d, 2.0, 2);
    assert_eq!(&d[..2], &[243.0, 405.0]);
    let mut dd = vec![
        DD::from_f64(1.0),
        DD::from_f64(5.0),
        DD::from_f64(10.0),
        DD::from_f64(10.0),
    ];
    shift(&mut dd, DD::from_f64(-0.5), 4);
    // the cubic 1 + 5 t + 10 t^2 + 10 t^3 about -0.5, summed at u = 1, is its value at t = 0.5
    let direct = |x: f64| 1.0 + 5.0 * x + 10.0 * x * x + 10.0 * x * x * x;
    let sum: f64 = dd.iter().map(|c| c.to_f64()).sum();
    assert_eq!(sum, direct(0.5));
}

/// the lookup of a table of an exact polynomial: boundaries, direction, outside, NaN
#[test]
fn find_and_headroom() {
    for (xa, xb) in [(0.1, 3.0), (3.0, 0.1)] {
        let src = Src {
            funcs: vec![Func::Far30],
            valid: Valid::Pieces6,
            nan: None,
        };
        let mut o = Opts::new(20, 1e-13);
        o.hmax = 0.4;
        let t = Table::<f64>::build(&src, xa, xb, &o).unwrap();
        let n = t.steps();
        assert!(n > 5);
        assert_eq!(t.direction(), if xb > xa { 1.0 } else { -1.0 });
        for i in 0..n {
            let (s, e) = t.bounds(i);
            // a boundary belongs to the step that starts there
            assert_eq!(t.find(s), Some(i));
            if i + 1 < n {
                assert_eq!(t.find(e), Some(i + 1));
            } else {
                assert_eq!(t.find(e), Some(i), "the end of the path belongs to the last step");
            }
            let mid = 0.5 * (s + e);
            assert_eq!(t.find(mid), Some(i));
            assert!(t.headroom(mid) <= t.width(i));
            assert!(t.headroom(s) == t.width(i));
        }
        assert_eq!(t.find(f64::NAN), None);
        assert_eq!(t.find(xb + (xb - xa).signum()), None);
        assert_eq!(t.find(xa - (xb - xa).signum()), None);
        // the lane change keeps the steps and rounds the coefficients
        let d = t.map(DD::from_f64);
        assert_eq!(d.steps(), n);
        assert_eq!(d.series(0, 0)[3].to_f64(), t.series(0, 0)[3]);
    }
}

// ---------------------------------------------------------------- C4: a plane through the table

const N: usize = 4;
type Mat = [[f64; N]; N];
const P: Mat = [[1., 1., 0., 1.], [1., 2., 1., 1.], [0., 1., 2., 1.], [1., 1., 1., 3.]];
const PINV: Mat = [
    [7., -4., 3., -2.],
    [-4., 3., -2., 1.],
    [3., -2., 2., -1.],
    [-2., 1., -1., 1.],
];
/// the exponents a of the solutions x^a P[:, j] of y' = (M / x) y: the plane P[:, :2] carries the two slowest decaying
/// modes, the dominant plane, so that rounding in the lane does not grow against it (a plane of (1, -2) among (3, -1)
/// amplifies it by x^5 over the path: see amendment A2 of the raw README)
const EXPONENTS: [f64; N] = [-1., -2., -3., -4.];

fn matmul(a: &Mat, b: &Mat) -> Mat {
    let mut c = [[0.0; N]; N];
    for (i, ci) in c.iter_mut().enumerate() {
        for (j, cij) in ci.iter_mut().enumerate() {
            *cij = (0..N).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

/// M = P diag(1, -2, 3, -1) P^-1, an integer matrix
fn m_matrix() -> Mat {
    let mut pd = P;
    for row in pd.iter_mut() {
        for (j, x) in row.iter_mut().enumerate() {
            *x *= EXPONENTS[j];
        }
    }
    matmul(&pd, &PINV)
}

/// ||(I - Pi) Y||_F for the plane's columns Y and Pi the projector onto span P[:, :2], formed in the lane
fn off_plane<T: Lane>(p: &Plane<T>) -> f64 {
    let cols: Vec<[T; N]> = (0..2).map(|c| std::array::from_fn(|i| T::from_f64(P[i][c]))).collect();
    let g = |a: usize, b: usize| (0..N).fold(T::zero(), |s, i| s + cols[a][i] * cols[b][i]);
    let (g00, g01, g11) = (g(0, 0), g(0, 1), g(1, 1));
    let det = (g00 * g11 - g01 * g01).recip();
    let mut sum = 0.0f64;
    for c in 0..p.k() {
        let y = p.column(c);
        let t: Vec<T> = (0..2)
            .map(|a| (0..N).fold(T::zero(), |s, i| s + cols[a][i] * y[i]))
            .collect();
        let w = [(g11 * t[0] - g01 * t[1]) * det, (g00 * t[1] - g01 * t[0]) * det];
        for i in 0..N {
            let r = (y[i] - (cols[0][i] * w[0] + cols[1][i] * w[1])).to_f64();
            sum += r * r;
        }
    }
    sum.sqrt()
}

struct Ph {
    plane: f64,
    ln_err: f64,
    ln_bar: f64,
    table_steps: usize,
    plane_steps: usize,
    c3: Vec<String>,
}

fn ph<T: Lane>(ctl: Ctl) -> Ph {
    let (xa, xb) = (1e-3, 40.0);
    let m = m_matrix();
    let mut pattern = vec![];
    let mut funcs = vec![];
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                pattern.push((i, j));
                funcs.push(Func::Scaled(v));
            }
        }
    }
    let tol = 100.0 * T::EPS;
    let kt = Plan::optimal_kt(tol);
    #[allow(unused_mut)]
    let mut o = Opts::new(kt + 12, T::EPS);
    #[cfg(feature = "mutate")]
    {
        o.control = ctl.lib();
    }
    #[cfg(not(feature = "mutate"))]
    assert_eq!(ctl, Ctl::None);
    let src = Src {
        funcs,
        valid: Valid::Never,
        nan: None,
    };
    let tab = Table::<T>::build(&src, xa, xb, &o).unwrap();
    let jets = TableJets::new(&tab, pattern);
    let basis: Vec<T> = (0..2).flat_map(|c| (0..N).map(move |i| T::from_f64(P[i][c]))).collect();
    let mut pl = Plane::<T>::new(N, 2, &basis);
    let st = pl.advance(&jets, xa, xb, &Plan { kt, tol, hmax: 1.0 }, kt);
    let plane = off_plane(&pl);
    // the plane's volume gains (-1 - 2) ln(xb / xa) from the basis' (1/2) ln 5
    let exact = T::parse(HALF_LN5) - T::from_f64(3.0) * (T::from_f64(xb) / T::from_f64(xa)).ln();
    let ln_err = (pl.ln_vol() - exact).to_f64().abs();
    let c3 = if ctl == Ctl::None {
        // the prototype's table of M / x has the same widths as the table of 1/x (the entries only scale it)
        c3::<T>("PH", &tab)
    } else {
        vec![]
    };
    Ph {
        plane,
        ln_err,
        ln_bar: 1e4 * T::EPS * exact.to_f64().abs().max(1.0),
        table_steps: tab.steps(),
        plane_steps: st.steps,
        c3,
    }
}

fn ph_passes<T: Lane>(o: &Ph) -> bool {
    o.plane <= 1e4 * T::EPS && o.ln_err <= o.ln_bar
}

fn ph_report<T: Lane>(ctl: &str, o: &Ph) {
    println!(
        "PH {} {ctl}: table steps {}, plane steps {}, plane error {:.3e} (bar {:.3e}), ln_vol error {:.3e} (bar {:.3e}), C3 violations {}",
        T::NAME,
        o.table_steps,
        o.plane_steps,
        o.plane,
        1e4 * T::EPS,
        o.ln_err,
        o.ln_bar,
        o.c3.len()
    );
}

fn ph_pass<T: Lane>() {
    let o = ph::<T>(Ctl::None);
    ph_report::<T>("clean", &o);
    assert!(
        ph_passes::<T>(&o),
        "PH {}: plane {:e}, ln_vol {:e} (bar {:e})",
        T::NAME,
        o.plane,
        o.ln_err,
        o.ln_bar
    );
    assert!(o.c3.is_empty(), "PH {}: C3 {:?}", T::NAME, &o.c3[..o.c3.len().min(5)]);
}

#[test]
fn ph_f64() {
    ph_pass::<f64>();
}

#[test]
fn ph_dd() {
    ph_pass::<DD>();
}

#[test]
fn ph_qd() {
    ph_pass::<QD>();
}

// ---------------------------------------------------------------- the controls

/// The wrong input `ref_scale`: the reference function times 1 + 1e-4; C1 fails in every lane
#[test]
fn ref_scale_fires() {
    fn go<T: Lane>() {
        let case = case_named("PC");
        let tab = build::<T>(&case, 4, Ctl::None);
        let (e, _) = c1(&case, &tab, 1.0 + 1e-4);
        println!("ref_scale PC {}: C1 maxE {e:.3e} (bar {:.1e})", T::NAME, bar::<T>());
        assert!(exceeds(e, bar::<T>()));
    }
    go::<f64>();
    go::<DD>();
    go::<QD>();
}

/// what a library control must do on a case and lane: Some((C1 fires, C2 fires)), None where not declared
#[cfg(feature = "mutate")]
fn declared(ctl: Ctl, case: &str, lane: &str) -> Option<(bool, bool)> {
    // PD in qd and PP have no radius to cut by: their width is the cap (an exact polynomial), so the controls that act on
    // the width leave them as they are
    let exact_poly = (case == "PD" && lane == "qd") || case == "PP";
    // PD at `last` 4 has room: a fraction of 1 stays within the bar in C1 and fires C2 only (below, FractionOne)
    match ctl {
        Ctl::Wide => match case {
            "PA" | "PB" | "PC" | "PF" => Some((true, true)),
            "PD" | "PP" => Some(if exact_poly { (false, false) } else { (true, true) }),
            _ => None,
        },
        Ctl::FractionOne => match case {
            "PA" | "PB" | "PC" | "PF" => Some((true, true)),
            "PD" | "PP" => Some((false, !exact_poly)),
            _ => None,
        },
        Ctl::LooseTail => match case {
            "PA" | "PB" | "PC" | "PF" => Some((true, true)),
            "PD" | "PP" => Some(if exact_poly { (false, false) } else { (true, true) }),
            _ => None,
        },
        Ctl::IgnoreValidTo => match case {
            "PB" => Some((true, true)),
            "PA" | "PF" => Some((false, true)),
            _ => None,
        },
        Ctl::NoShift => match case {
            "PA" | "PB" | "PC" | "PD" | "PF" | "PP" => Some((true, false)),
            _ => None,
        },
        Ctl::ShiftF64 => match case {
            "PP" => Some((lane != "f64", false)),
            "PA" | "PB" | "PC" | "PD" | "PF" => Some((false, false)),
            _ => None,
        },
        Ctl::Unscaled => match case {
            "PS" => Some((true, true)),
            _ => None,
        },
        Ctl::None => None,
    }
}

#[cfg(feature = "mutate")]
fn control_cells<T: Lane>() {
    let ctls = [
        Ctl::Wide,
        Ctl::FractionOne,
        Ctl::LooseTail,
        Ctl::IgnoreValidTo,
        Ctl::NoShift,
        Ctl::ShiftF64,
        Ctl::Unscaled,
    ];
    let mut wrong = vec![];
    for ctl in ctls {
        for case in cases().iter().filter(|c| c.lanes.contains(&T::NAME)) {
            let Some((f1, f2)) = declared(ctl, case.name, T::NAME) else {
                continue;
            };
            let r = run::<T>(case, 4, ctl);
            let c1_fires = exceeds(r.c1, bar::<T>());
            let c2_fires = !r.c2.is_empty();
            println!(
                "control {ctl:?} on {} {}: C1 maxE {:.3e} ({}), C2 violations {} ({}), steps {}",
                case.name,
                T::NAME,
                r.c1,
                if c1_fires { "fires" } else { "silent" },
                r.c2.len(),
                if c2_fires { "fires" } else { "silent" },
                r.steps
            );
            if (c1_fires, c2_fires) != (f1, f2) {
                wrong.push(format!(
                    "{ctl:?} on {} {}: C1 fires {c1_fires} (declared {f1}), C2 fires {c2_fires} (declared {f2})",
                    case.name,
                    T::NAME
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[cfg(feature = "mutate")]
#[test]
fn controls_f64() {
    control_cells::<f64>();
}

#[cfg(feature = "mutate")]
#[test]
fn controls_dd() {
    control_cells::<DD>();
}

#[cfg(feature = "mutate")]
#[test]
fn controls_qd() {
    control_cells::<QD>();
}

/// the controls through the plane (PH): `FractionOne` and `LooseTail` fail the log-volume bar in dd and qd and are
/// silent in f64, where the bar (1e4 eps = 1e-12) is above their effect; `Wide` is not run (the plane's steps collapse
/// on the table's garbage and the integration does not finish)
#[cfg(feature = "mutate")]
fn ph_controls<T: Lane>(fires: bool) {
    for ctl in [Ctl::FractionOne, Ctl::LooseTail] {
        let o = ph::<T>(ctl);
        ph_report::<T>(&format!("{ctl:?}"), &o);
        assert_eq!(
            !ph_passes::<T>(&o),
            fires,
            "PH {} {ctl:?}: plane {:e}, ln_vol {:e} (bar {:e})",
            T::NAME,
            o.plane,
            o.ln_err,
            o.ln_bar
        );
    }
}

#[cfg(feature = "mutate")]
#[test]
fn ph_controls_f64() {
    ph_controls::<f64>(false);
}

#[cfg(feature = "mutate")]
#[test]
fn ph_controls_dd() {
    ph_controls::<DD>(true);
}

#[cfg(feature = "mutate")]
#[test]
fn ph_controls_qd() {
    ph_controls::<QD>(true);
}

// ---------------------------------------------------------------- the review's edge cases, and C3's control

/// C3's control: the comparator is fed a wrong prototype (a count one too many, a start moved by 1e-6, no record) and
/// must report each
#[test]
fn c3_fires() {
    let case = case_named("PA");
    let tab = build::<DD>(&case, 4, Ctl::None);
    assert!(c3::<DD>("PA", &tab).is_empty());
    let count: Vec<StepRow> = STEPS.iter().map(|&(a, b, n, s)| (a, b, n + 1, s)).collect();
    assert_eq!(c3_rows::<DD>(&count, "PA", &tab).len(), 1);
    let moved: Vec<StepRow> = STEPS
        .iter()
        .map(|&(a, b, n, s)| {
            let v: Vec<StepStart> = s
                .iter()
                .map(|&(i, x)| (i, if i == 20 { x * (1.0 + 1e-6) } else { x }))
                .collect();
            (a, b, n, &*Box::leak(v.into_boxed_slice()))
        })
        .collect();
    assert_eq!(c3_rows::<DD>(&moved, "PA", &tab).len(), 1);
    assert_eq!(c3_rows::<DD>(&[], "PA", &tab).len(), 1);
}

/// a source of one function with given coefficients (the rest zero) about every point, whose series stop holding at
/// `valid_fwd` when the path runs forward from below it
struct Fixed {
    c: Vec<f64>,
    valid_fwd: f64,
}

impl<T: Lane> SeriesSource<T> for Fixed {
    fn functions(&self) -> usize {
        1
    }
    fn series(&self, x0: f64, dir: f64, nb: usize, out: &mut [T]) -> f64 {
        for (k, o) in out[..nb].iter_mut().enumerate() {
            *o = T::from_f64(self.c.get(k).copied().unwrap_or(0.0));
        }
        if dir > 0.0 && x0 < self.valid_fwd {
            self.valid_fwd
        } else {
            dir * f64::INFINITY
        }
    }
}

/// two coefficients per function and `last` 1 set outright (`last` 2 would read c_0, which no tail passes; the default
/// is tested by itself)
fn opts2(tail: f64) -> Opts {
    let mut o = Opts::new(2, tail);
    o.last = 1;
    o
}

fn fixed(c: &[f64]) -> Fixed {
    Fixed {
        c: c.to_vec(),
        valid_fwd: f64::INFINITY,
    }
}

/// `Opts::new(2, ..)` is a valid setting: the default `last` is at most nb - 1 (a `last` of nb reads c_0, which no tail passes)
#[test]
fn default_last_fits_small_nb() {
    let o = Opts::new(2, 1e-6);
    assert!(o.last <= 1);
    let t = Table::<f64>::build(&fixed(&[1.0, 0.0]), 0.0, 1.0, &o).unwrap();
    assert_eq!(t.bounds(t.steps() - 1).1, 1.0);
}

/// a path shorter than the widths the search used to give up at: coefficients [1, 1e300] on [0, 1e-300], where the test
/// fails at 5e-301 and passes at 6.25e-302
#[test]
fn tiny_path_is_not_a_stall() {
    let mut o = Opts::new(2, 0.1);
    o.last = 1;
    o.fraction = 1.0;
    let t = Table::<f64>::build(&fixed(&[1.0, 1e300]), 0.0, 1e-300, &o).expect("a step of 6e-302 moves x");
    assert_eq!(t.bounds(t.steps() - 1).1, 1e-300);
    assert!(t.steps() > 1);
}

/// a path of 1e308 with fraction 0.5: `limit / fraction` overflows, and the search must still end
#[test]
fn huge_path_ends() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut o = Opts::new(2, 0.1);
        o.last = 1;
        o.fraction = 0.5;
        // the function 1 + x: its series about x0 is [1 + x0, 1]
        struct Linear;
        impl SeriesSource<f64> for Linear {
            fn functions(&self) -> usize {
                1
            }
            fn series(&self, x0: f64, dir: f64, nb: usize, out: &mut [f64]) -> f64 {
                out[..nb].iter_mut().for_each(|o| *o = 0.0);
                out[0] = 1.0 + x0;
                out[1] = 1.0;
                dir * f64::INFINITY
            }
        }
        let r = Table::<f64>::build(&Linear, 0.0, 1e308, &o).map(|t| (t.steps(), t.bounds(t.steps() - 1).1));
        let _ = tx.send(r);
    });
    let r = rx
        .recv_timeout(std::time::Duration::from_secs(20))
        .expect("the width search does not end");
    let (steps, end) = r.unwrap();
    println!("path of 1e308: {steps} steps");
    assert_eq!(end, 1e308);
}

/// the source's piece ends at 0, the path at 1, and x = -2^54: both distances round to 2^54, and the step must end at the
/// nearer, 0
#[test]
fn piece_end_before_path_end() {
    let x = -(2.0f64.powi(54));
    let src = Fixed {
        c: vec![1.0, 0.0],
        valid_fwd: 0.0,
    };
    let mut o = opts2(1e-6);
    o.fraction = 1.0;
    let t = Table::<f64>::build(&src, x, 1.0, &o).unwrap();
    assert_eq!(t.bounds(0), (x, 0.0), "the first step ends at the piece end");
    assert_eq!(t.bounds(t.steps() - 1).1, 1.0);
}

/// `hmax` bounds the stored endpoint: from 1 to the next f64, one ulp of 2^-52, with `hmax` 3 2^-54 no step fits
#[test]
fn hmax_bounds_the_stored_step() {
    let ulp = 2.0f64.powi(-52);
    let mut o = opts2(1e-6);
    o.hmax = 3.0 * 2.0f64.powi(-54);
    match Table::<f64>::build(&fixed(&[1.0, 0.0]), 1.0, 1.0 + ulp, &o) {
        Err(Error::Stall { x }) => assert_eq!(x, 1.0),
        other => panic!("{other:?}"),
    }
    // and with room for the ulp the step is the ulp
    o.hmax = 2.0 * ulp;
    let t = Table::<f64>::build(&fixed(&[1.0, 0.0]), 1.0, 1.0 + ulp, &o).unwrap();
    assert_eq!(t.bounds(0), (1.0, 1.0 + ulp));
}

/// a step or function index outside the table is not read as another step's series
#[test]
#[should_panic(expected = "function")]
fn series_index_is_checked() {
    let mut o = opts2(1e-6);
    o.hmax = 0.25;
    let t = Table::<f64>::build(&fixed(&[1.0, 0.0]), 0.0, 1.0, &o).unwrap();
    assert!(t.steps() >= 4);
    let _ = t.series(0, 1);
}

#[test]
#[should_panic(expected = "twice")]
fn tablejets_rejects_a_repeated_entry() {
    struct Two;
    impl<T: Lane> SeriesSource<T> for Two {
        fn functions(&self) -> usize {
            2
        }
        fn series(&self, _x0: f64, dir: f64, nb: usize, out: &mut [T]) -> f64 {
            out.iter_mut().for_each(|o| *o = T::zero());
            out[0] = T::one();
            out[nb] = T::one();
            dir * f64::INFINITY
        }
    }
    let t = Table::<f64>::build(&Two, 0.0, 1.0, &opts2(1e-6)).unwrap();
    let _ = TableJets::new(&t, vec![(0, 0), (0, 0)]);
}

/// a request for more coefficients than the table stores stops with a message, not by silently dropping orders
#[test]
#[should_panic(expected = "exceeds the table's nb")]
fn tablejets_names_an_order_beyond_nb() {
    use multishoot::plane::{JetSource, Jets};
    let t = Table::<f64>::build(&fixed(&[1.0, 0.0]), 0.0, 1.0, &opts2(1e-6)).unwrap();
    let j = TableJets::new(&t, vec![(0, 0)]);
    let mut out = Jets::<f64>::new(&[(0, 0)], 3);
    j.jets(0.5, 3, &mut out);
}

/// The width is where the search ended, not an interval: a series [1, 0, 10, 1] with `last` 2 and tail 0.6 passes at 10
/// (the last two terms are 1000 and 1000 of 2001) and fails at 2 (40 and 8 of 49). With fraction 1 the width is 10, and
/// `headroom` is the distance to it; the module's docs say this is pointwise outside the cases they derive.
#[test]
fn width_is_a_point_where_the_test_passes() {
    let mut o = Opts::new(4, 0.6);
    o.last = 2;
    o.fraction = 1.0;
    let t = Table::<f64>::build(&fixed(&[1.0, 0.0, 10.0, 1.0]), 0.0, 10.0, &o).unwrap();
    assert_eq!(t.width(0), 10.0);
    assert_eq!(t.headroom(0.0), 10.0);
    let c = [vec![1.0, 0.0, 10.0, 1.0]];
    assert_eq!(width_log(&c, 10.0, 0.6, 2), 10.0);
    assert!(width_log(&c, 2.0, 0.6, 2) < 2.0, "the test fails at 2");
}
