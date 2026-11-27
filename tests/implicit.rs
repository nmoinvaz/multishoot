//! Planted checks of `multishoot::implicit`: implicit second-order ODEs N(x, y, y', y'') = 0, a polynomial in y'', whose
//! solutions are known in closed form, in f64, double-double and quad-double.
//!
//! Pass criteria, written before the first run (the unit roundoff eps of the lane: f64 1.1e-16, dd 1.2e-32, qd 1.5e-64;
//! bar = 1e6 eps; Taylor order kt = `Plan::optimal_kt(tol)`: f64 22 at tol 1e-14, dd 42 at 1e-30, qd 80 at 1e-62;
//! err = max_f |computed_f - exact_f| / max_f |exact_f| over (y, y', y'')):
//!
//! - `coeffs`: Q-A's series about x0 = 0.7, scaled by l = 1 and l = 1/4, against the Taylor coefficients of exp(x^2 / 2):
//!   gated, sum_k |dc_k| <= bar sum_k |c_k| for each of y, y', y'' (the series summed at s = 1); observed and not gated
//!   (NOT MET, README): every c_k within bar, relative, `coeffs_each`.
//! - `closed_form`: nine branches, from x = 0 to 3, hmax 0.3, landing on 0.5, 1.5 and 3, in both branch modes: err <= bar
//!   at each landing, the walk ends with `Stop::End`, and `n_rel_max` <= bar.
//! - `crossings`: the sign changes of A0 equal the exact list in number and each is located within bar max(1, x):
//!   Q-B {1}, K-A {1}, K-B {1, sqrt 3}, K-C {x_C}, R-A {1}, R-B {1, sqrt 3}, R-C {x_C}, u tanh u = 1 at u = sqrt 2 x_C.
//! - `fold`: from x0 = -1 with pivot_abs 1e-4 (F1) and 1e-9 (F2), hmax 1: `Stop::Pivot`, x < 0, and the estimate of the zero
//!   x* = 0 within 1e-3 (-x) of it (the trivial estimate x_zero = x misses by the full distance).
//! - `zero_pivot`: a walk from (1, -1, 0) of N1 at x = 1 stops with `Stop::ZeroPivot` at x = 1.
//! - `planner`: `planned()` equals the planner's rule applied to the series' own coefficients, bitwise, at five steps of F1.
//! - `lanes`: the series on the lane field `Fl<DV, 1>` and the dual lane `Fl<Du, 1>` (value part) equals double-double's
//!   within 1e2 eps_dd per coefficient (Q-A and K-C from x0 = 0.7).
//!
//! Controls (feature `mutate`, plus two that need none): each must fail the check it names; the targets are declared after
//! a development run (raw README, amendment), the prediction being the name's own check. `println!` lines are the log.
use multishoot::implicit::*;
use multishoot::lane::{Du, DV};
use multishoot::multidouble::od::OD;
use multishoot::multidouble::qd::QD;
use multishoot::taylor::{step_from_norms, Plan};
use multishoot::{Fl, DD};

/// the number types of the planted checks, with the closed forms' functions
trait TestReal: Real {
    const NAME: &'static str;
    const KT: usize;
    const TOL: f64;
    /// how many f64 parts of a value the digests hash: its whole value, which is four for f64, dd and qd (hashed with
    /// four from the first version of the digests, which #190 and #214 quote) and eight for od
    const PARTS: usize;
    /// the `digests` of the lane's planted cells: [base, full], as printed on aarch64 NEON, both backends of multidouble
    /// there agreeing (README, SIMD backends); a different value on any backend or target is a failure of the engine's
    /// claim that results do not depend on it, and a deliberate change of the numerics changes these
    const DIGEST: [u64; 2];
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
}

impl TestReal for f64 {
    const NAME: &'static str = "f64";
    const KT: usize = 22;
    const TOL: f64 = 1e-14;
    const PARTS: usize = 4;
    const DIGEST: [u64; 2] = [0xb124d44e9758f1a1, 0x50775575530f4fa9];
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn sin(self) -> f64 {
        f64::sin(self)
    }
    fn cos(self) -> f64 {
        f64::cos(self)
    }
}

macro_rules! test_real {
    ($t:ty, $name:expr, $kt:expr, $tol:expr, $parts:expr, $digest:expr) => {
        impl TestReal for $t {
            const NAME: &'static str = $name;
            const KT: usize = $kt;
            const TOL: f64 = $tol;
            const PARTS: usize = $parts;
            const DIGEST: [u64; 2] = $digest;
            fn exp(self) -> $t {
                <$t>::exp(self)
            }
            fn sqrt(self) -> $t {
                <$t>::sqrt(self)
            }
            fn sin(self) -> $t {
                <$t>::sin(self)
            }
            fn cos(self) -> $t {
                <$t>::cos(self)
            }
        }
    };
}
test_real!(DD, "dd", 42, 1e-30, 4, [0x92ec461781147d77, 0x13ac06786e54ccb1]);
test_real!(QD, "qd", 80, 1e-62, 4, [0x75902b0bf76ced2a, 0xb37fa8d9cb20094f]);
// the digests of `od_digests`' cells, not of all of them
test_real!(OD, "od", 154, 1e-126, 8, [0x10f1b89a823ad3c3, 0x21d526b7c72a4319]);

fn bar<T: TestReal>() -> f64 {
    1e6 * T::EPS
}

/// N = sum_g A_g(x) y^a y'^b y''^c with integer polynomial coefficients A_g
#[derive(Clone)]
struct PolyEq {
    pat: Vec<Mono>,
    polys: Vec<Vec<i64>>,
}

impl PolyEq {
    fn new(terms: &[(Mono, &[i64])]) -> Self {
        PolyEq {
            pat: terms.iter().map(|t| t.0).collect(),
            polys: terms.iter().map(|t| t.1.to_vec()).collect(),
        }
    }
    /// N1 = y''^2 - (y + x y')^2
    fn q() -> Self {
        Self::new(&[
            ((0, 0, 2), &[1]),
            ((2, 0, 0), &[-1]),
            ((1, 1, 0), &[0, -2]),
            ((0, 2, 0), &[0, 0, -1]),
        ])
    }
    /// N3 = N1 (y'' - 2 y)
    fn k() -> Self {
        Self::new(&[
            ((0, 0, 3), &[1]),
            ((1, 0, 2), &[-2]),
            ((2, 0, 1), &[-1]),
            ((1, 1, 1), &[0, -2]),
            ((0, 2, 1), &[0, 0, -1]),
            ((3, 0, 0), &[2]),
            ((2, 1, 0), &[0, 4]),
            ((1, 2, 0), &[0, 0, 2]),
        ])
    }
    /// N4 = N1 (y''^2 - 4 y^2)
    fn r() -> Self {
        Self::new(&[
            ((0, 0, 4), &[1]),
            ((2, 0, 2), &[-5]),
            ((1, 1, 2), &[0, -2]),
            ((0, 2, 2), &[0, 0, -1]),
            ((4, 0, 0), &[4]),
            ((3, 1, 0), &[0, 8]),
            ((2, 2, 0), &[0, 0, 4]),
        ])
    }
    /// 16 y''^2 + 4 y'^2 + 225 x + 25 x^3
    fn f1() -> Self {
        Self::new(&[((0, 0, 2), &[16]), ((0, 2, 0), &[4]), ((0, 0, 0), &[0, 225, 0, 25])])
    }
    /// 3 y' + 2 y''^3
    fn f2() -> Self {
        Self::new(&[((0, 1, 0), &[3]), ((0, 0, 3), &[2])])
    }
}

impl<T: Scalar> Coefficients<T> for PolyEq {
    fn pattern(&self) -> Vec<Mono> {
        self.pat.clone()
    }
    fn jets(&self, x0: f64, _kt: usize, out: &mut Jets<T>) {
        for (g, p) in self.polys.iter().enumerate() {
            let poly: Vec<T> = p.iter().map(|&c| T::from_f64(c as f64)).collect();
            taylor_shift(&poly, x0, &mut out.ent[g].c);
            out.ent[g].n = poly.len();
        }
    }
}

/// N with `f64` coefficients, for the equations whose coefficients are beyond `PolyEq`'s integers (2^1000, 2^-1030)
struct RealEq {
    pat: Vec<Mono>,
    polys: Vec<Vec<f64>>,
}

impl RealEq {
    fn new(terms: &[(Mono, &[f64])]) -> Self {
        RealEq {
            pat: terms.iter().map(|t| t.0).collect(),
            polys: terms.iter().map(|t| t.1.to_vec()).collect(),
        }
    }
}

impl<T: Scalar> Coefficients<T> for RealEq {
    fn pattern(&self) -> Vec<Mono> {
        self.pat.clone()
    }
    fn jets(&self, x0: f64, _kt: usize, out: &mut Jets<T>) {
        for (g, p) in self.polys.iter().enumerate() {
            let poly: Vec<T> = p.iter().map(|&c| T::from_f64(c)).collect();
            taylor_shift(&poly, x0, &mut out.ent[g].c);
            out.ent[g].n = poly.len();
        }
    }
}

/// 2^e exactly, for -1074 <= e <= 1023
fn pow2(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// the branches of the planted equations
#[derive(Clone, Copy, Debug, PartialEq)]
enum Br {
    QA,
    QB,
    KA,
    KB,
    KC,
    RA,
    RB,
    RC,
    RD,
}

const BRANCHES: [Br; 9] = [Br::QA, Br::QB, Br::KA, Br::KB, Br::KC, Br::RA, Br::RB, Br::RC, Br::RD];

fn eq_of(b: Br) -> PolyEq {
    match b {
        Br::QA | Br::QB => PolyEq::q(),
        Br::KA | Br::KB | Br::KC => PolyEq::k(),
        _ => PolyEq::r(),
    }
}

fn sqrt2<T: TestReal>() -> T {
    T::from_f64(2.0).sqrt()
}

/// (cosh u, sinh u)
fn cosh_sinh<T: TestReal>(u: T) -> (T, T) {
    let (a, b) = (u.exp(), (-u).exp());
    ((a + b).mulf(0.5), (a - b).mulf(0.5))
}

/// (y, y', y'') of a branch at x in the lane
fn exact<T: TestReal>(b: Br, x: T) -> [T; 3] {
    let one = T::one();
    match b {
        Br::QA | Br::KA | Br::RA => {
            let e = (x * x).mulf(0.5).exp();
            [e, x * e, (one + x * x) * e]
        }
        Br::QB | Br::KB | Br::RB => {
            let e = (-(x * x).mulf(0.5)).exp();
            [e, -(x * e), (x * x - one) * e]
        }
        Br::KC | Br::RC => {
            let r = sqrt2::<T>();
            let (c, s) = cosh_sinh(r * x);
            [c, r * s, c.mulf(2.0)]
        }
        Br::RD => {
            let r = sqrt2::<T>();
            let u = r * x;
            let (c, s) = (u.cos(), u.sin());
            [c, -(r * s), -c.mulf(2.0)]
        }
    }
}

/// the root of u tanh u = 1 over sqrt 2, by Newton on u sinh u - cosh u in the lane
fn x_c<T: TestReal>() -> T {
    let mut u = T::from_f64(1.2);
    for _ in 0..14 {
        let (c, s) = cosh_sinh(u);
        u = u - (u * s - c) / (u * c);
    }
    u / sqrt2::<T>()
}

/// where the branch meets another root of N in y''
fn crossing_list<T: TestReal>(b: Br) -> Vec<T> {
    let one = T::one();
    match b {
        Br::QB | Br::KA | Br::RA => vec![one],
        Br::KB | Br::RB => vec![one, T::from_f64(3.0).sqrt()],
        Br::KC | Br::RC => vec![x_c::<T>()],
        _ => vec![],
    }
}

/// a deliberate error of a control
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ctl {
    None,
    #[cfg(feature = "mutate")]
    Lib(ImplicitControl),
    /// K-A started on K-B's root
    WrongStart,
    /// K-A with y'' replaced by K-C's root at x = 0.6
    SwitchMid,
    /// the walk compared with another branch's closed form
    WrongForm,
    /// one computed coefficient of the series replaced by a NaN
    NanCoeff,
    /// one coefficient of the series off by 10 bar of the series' size
    BadCoeff,
    /// the position of every crossing found replaced by a NaN
    NanPos,
    /// the end state of the walk replaced by a NaN
    NanState,
    /// the planned step of `planner` with its last bit flipped, before the comparison
    FlipBit,
    /// one coefficient of each lane of `lanes` off by 1e-20
    LanePerturb,
    /// one coefficient of each lane of `lanes` a NaN
    NanLane,
    /// a planted path that ends after n accepted steps, with `Stop::StepLimit` (n = 0: its first `form` fails)
    Cut(usize),
}

/// the larger of two numbers, NaN if either is: `f64::max` returns the other number, which hides a NaN measurement
fn nanmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// max_f |a_f - b_f| / max_f |b_f|
fn rel_err<T: TestReal>(a: &[T; 3], b: &[T; 3]) -> f64 {
    let d = (0..3).map(|f| (a[f] - b[f]).to_f64().abs()).fold(0.0, nanmax);
    let n = (0..3).map(|f| b[f].to_f64().abs()).fold(0.0, nanmax);
    d / n
}

struct Case<T: TestReal> {
    out: WalkOut<T>,
    /// err at each landing (0.5, 1.5, 3)
    errs: Vec<f64>,
}

const CPS: [f64; 2] = [0.5, 1.5];

fn opts<T: TestReal>(mode: Branch, hmax: f64) -> WalkOpts {
    let mut o = WalkOpts::new(T::KT, T::TOL);
    o.plan.hmax = hmax;
    o.branch = mode;
    o
}

/// the walk of a branch from x = 0 to 3, with a control
fn run_case<T: TestReal>(br: Br, mode: Branch, ctl: Ctl) -> Case<T> {
    let eq = eq_of(br);
    let y0 = exact::<T>(br, T::zero());
    let mut w = Walker::<T, _>::new(&eq, 0.0, y0, opts::<T>(mode, 0.3));
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        w.set_control(c);
    }
    if ctl == Ctl::WrongStart {
        w.set_y([y0[0], y0[1], -y0[2]]);
    }
    let mut cps = Vec::new();
    let mut out = if ctl == Ctl::SwitchMid {
        let o1 = w.run(0.6, &CPS);
        cps.extend(o1.cps.iter().cloned());
        let y = o1.y;
        w.set_y([y[0], y[1], y[0].mulf(2.0)]);
        w.run(3.0, &CPS)
    } else {
        w.run(3.0, &CPS)
    };
    cps.extend(out.cps.iter().cloned());
    if ctl == Ctl::NanState {
        out.y[0] = T::from_f64(f64::NAN);
    }
    let against = if ctl == Ctl::WrongForm { Br::KB } else { br };
    let mut errs = Vec::new();
    for (x, y, _) in &cps {
        errs.push(rel_err(y, &exact::<T>(against, T::from_f64(*x))));
    }
    errs.push(rel_err(&out.y, &exact::<T>(against, T::from_f64(out.x))));
    Case { out, errs }
}

struct Check {
    pass: bool,
    detail: String,
}

fn closed_form<T: TestReal>(br: Br, mode: Branch, ctl: Ctl) -> Check {
    let c = run_case::<T>(br, mode, ctl);
    let b = bar::<T>();
    let worst = c.errs.iter().cloned().fold(0.0, nanmax);
    let pass = c.out.stop == Stop::End && c.errs.len() == 3 && worst <= b && c.out.n_rel_max <= b;
    Check {
        pass,
        detail: format!(
            "{} {:?} {:?} {:?}: stop {:?}, {} steps, err {:.2e} (bar {:.1e}), n_rel_max {:.2e}",
            T::NAME,
            br,
            mode,
            ctl,
            c.out.stop,
            c.out.steps,
            worst,
            b,
            c.out.n_rel_max
        ),
    }
}

fn crossings<T: TestReal>(br: Br, mode: Branch, ctl: Ctl) -> Check {
    let mut c = run_case::<T>(br, mode, ctl);
    if ctl == Ctl::NanPos {
        for k in c.out.crossings.iter_mut() {
            k.x = T::from_f64(f64::NAN);
        }
    }
    let list = crossing_list::<T>(br);
    let b = bar::<T>();
    let found: Vec<T> = c.out.crossings.iter().map(|k| k.x).collect();
    let mut worst = 0.0f64;
    for (a, e) in found.iter().zip(&list) {
        worst = nanmax(worst, (*a - *e).to_f64().abs() / e.to_f64().abs().max(1.0));
    }
    let pass = found.len() == list.len() && c.out.crossings.iter().all(|k| k.located) && worst <= b;
    Check {
        pass,
        detail: format!(
            "{} {:?} {:?} {:?}: {} crossings found, {} expected, worst position error {:.2e} (bar {:.1e})",
            T::NAME,
            br,
            mode,
            ctl,
            found.len(),
            list.len(),
            worst,
            b
        ),
    }
}

/// 'F1' (exponent 1/2) and 'F2' (exponent 1)
fn fold<T: TestReal>(which: u8, mode: Branch, ctl: Ctl) -> Check {
    let (eq, y0, pivot) = if which == 1 {
        (PolyEq::f1(), [T::one(), T::from_f64(-2.5), T::from_f64(3.75)], 1e-4)
    } else {
        let (a, c) = (
            T::from_f64(4.0) / T::from_f64(15.0),
            T::from_f64(-2.0) / T::from_f64(3.0),
        );
        (PolyEq::f2(), [a, c, T::one()], 1e-9)
    };
    let mut o = opts::<T>(mode, 1.0);
    o.pivot_abs = pivot;
    let mut w = Walker::<T, _>::new(&eq, -1.0, y0, o);
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        w.set_control(c);
    }
    let out = w.run(1.0, &[]);
    let (pass, extra) = match (&out.stop, &out.zero) {
        (Stop::Pivot, Some(z)) => {
            let e = z.x.to_f64().abs();
            (
                out.x < 0.0 && e <= 1e-3 * (-out.x),
                format!("x_last {:.3e}, |x_zero| {:.2e}, beta {:.4}", out.x, e, z.beta),
            )
        }
        _ => (false, format!("no estimate, x_last {:.3e}", out.x)),
    };
    Check {
        pass,
        detail: format!(
            "{} F{} {:?} {:?}: stop {:?}, {} steps, {}",
            T::NAME,
            which,
            mode,
            ctl,
            out.stop,
            out.steps,
            extra
        ),
    }
}

fn zero_pivot<T: TestReal>(ctl: Ctl) -> Check {
    let eq = PolyEq::q();
    let mut w = Walker::<T, _>::new(
        &eq,
        1.0,
        [T::one(), -T::one(), T::zero()],
        opts::<T>(Branch::Project, 0.3),
    );
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        w.set_control(c);
    }
    let out = w.run(2.0, &[]);
    let finite = out.y.iter().all(|v| v.to_f64().is_finite());
    Check {
        pass: out.stop == Stop::ZeroPivot && out.x == 1.0 && finite,
        detail: format!(
            "{} {:?}: stop {:?} at x = {}, state finite {}",
            T::NAME,
            ctl,
            out.stop,
            out.x,
            finite
        ),
    }
}

fn planner<T: TestReal>(ctl: Ctl) -> Check {
    let eq = PolyEq::f1();
    let mut o = opts::<T>(Branch::Project, 1.0);
    o.h_scale = 0.5;
    let y0 = [T::one(), T::from_f64(-2.5), T::from_f64(3.75)];
    let mut w = Walker::<T, _>::new(&eq, -1.0, y0, o.clone());
    let kt = T::KT;
    let mut bad = 0;
    for _ in 0..5 {
        w.form().expect("form");
        let h = w.planned();
        let s = w.series();
        let e = |k: usize| {
            (0..3)
                .map(|f| {
                    s.c[f][k].to_f64().abs()
                        / if s.c[f][0].to_f64() != 0.0 {
                            s.c[f][0].to_f64().abs()
                        } else {
                            1.0
                        }
                })
                .fold(0.0, f64::max)
        };
        let plan = Plan {
            kt,
            tol: T::TOL,
            hmax: 1.0 / s.l,
        };
        let want = step_from_norms(e(kt - 1), e(kt), kt, &plan) * s.l * 0.5;
        let got = if ctl == Ctl::FlipBit {
            f64::from_bits(h.to_bits() ^ 1)
        } else {
            h
        };
        if got.to_bits() != want.to_bits() {
            bad += 1;
        }
        w.advance(h, 1.0).expect("advance");
    }
    Check {
        pass: bad == 0,
        detail: format!("{} F1 planner {:?}: {} of 5 steps differ in bits", T::NAME, ctl, bad),
    }
}

/// Q-A's series about x0 = 0.7 in the variable s = (x - x0) / l, and the Taylor coefficients of exp(x^2 / 2) in the same
/// variable: y, y', y'' by order
fn coeffs_data<T: TestReal>(l: f64, ctl: Ctl) -> (Series<T>, [Vec<T>; 3]) {
    let eq = PolyEq::q();
    let kt = T::KT;
    let x0 = 0.7;
    let y0 = exact::<T>(Br::QA, T::from_f64(x0));
    let mut rec = Recursion::<T>::new(&eq.pat, kt);
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        rec.set_control(c);
    }
    let mut jets = Jets::<T>::new(eq.pat.len(), kt);
    eq.jets(x0, kt, &mut jets);
    let mut s = rec.form(&jets, l, y0).clone();
    // exp(x0^2/2) exp(x0 h) exp(h^2/2): a_k = e0 sum_(2i + j = k) x0^j / (j! 2^i i!)
    let n = kt + 3;
    let mut fact = vec![T::one(); n + 1];
    for k in 1..=n {
        fact[k] = fact[k - 1] * T::from_f64(k as f64);
    }
    let xt = T::from_f64(x0);
    let e0 = (xt * xt).mulf(0.5).exp();
    let a: Vec<T> = (0..=n)
        .map(|k| {
            let mut t = T::zero();
            for i in 0..=k / 2 {
                let j = k - 2 * i;
                let xj = (0..j).fold(T::one(), |r, _| r * xt);
                t = t + xj / (fact[j] * fact[i] * T::from_f64(2f64.powi(i as i32)));
            }
            e0 * t
        })
        .collect();
    let ex = [
        (0..=kt).map(|k| a[k].mulf(l.powi(k as i32))).collect::<Vec<T>>(),
        (0..=kt)
            .map(|k| (a[k + 1] * T::from_f64((k + 1) as f64)).mulf(l.powi(k as i32)))
            .collect(),
        (0..=kt)
            .map(|k| (a[k + 2] * T::from_f64(((k + 1) * (k + 2)) as f64)).mulf(l.powi(k as i32)))
            .collect(),
    ];
    if ctl == Ctl::BadCoeff {
        // an error of 10 bar of the series' size in one coefficient of y
        let size: f64 = ex[0].iter().map(|e| e.to_f64().abs()).sum();
        s.c[0][kt / 2] = s.c[0][kt / 2] + T::from_f64(10.0 * bar::<T>() * size);
    }
    (s, ex)
}

/// The gated criterion of `coeffs`: for each field, sum_k |dc_k| <= bar sum_k |c_k| (the series summed at s = 1, the step
/// the scale l is for, with its terms' signs dropped): the accuracy a step uses, which `step_sum` measures on planned
/// steps. The relative error of each c_k is `coeffs_each`'s observation.
fn coeffs<T: TestReal>(l: f64, ctl: Ctl) -> Check {
    let (s, ex) = coeffs_data::<T>(l, ctl);
    let mut worst = 0.0f64;
    for (cf, ef) in s.c.iter().zip(ex.iter()) {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (c, e) in cf.iter().zip(ef.iter()) {
            num += (*c - *e).to_f64().abs();
            den += c.to_f64().abs();
        }
        worst = nanmax(worst, num / (T::EPS * den));
    }
    Check {
        pass: worst <= 1e6,
        detail: format!(
            "{} Q-A l = {} {:?}: worst sum|dc_k| / (eps sum|c_k|) {:.2e} (bar 1e6)",
            T::NAME,
            l,
            ctl,
            worst
        ),
    }
}

/// An observation, not a gated check: Q-A's series against the Taylor coefficients of exp(x^2 / 2), the relative error of
/// every c_k, met when the worst is within the bar. It is NOT MET in every lane (README, the coordinator's ruling): the
/// relative error of c_k grows about threefold per order, as order k of N is a cancelling convolution.
fn coeffs_each<T: TestReal>(l: f64) -> Check {
    let (s, ex) = coeffs_data::<T>(l, Ctl::None);
    let mut worst = 0.0f64;
    for k in 0..=T::KT {
        for (cf, ef) in s.c.iter().zip(ex.iter()) {
            worst = worst.max(((cf[k] - ef[k]) / ef[k]).to_f64().abs());
        }
    }
    let b = bar::<T>();
    Check {
        pass: worst <= b,
        detail: format!(
            "{} Q-A l = {}: observation, worst relative error of c_k, k <= {}: {:.2e} (bar {:.1e})",
            T::NAME,
            l,
            T::KT,
            worst,
            b
        ),
    }
}

/// a planted path: a branch of N1, N3 or N4 from x = 0, or the fold F1 or F2 from x = -1
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pth {
    B(Br),
    F1,
    F2,
}

/// the Taylor coefficients about x0 of the closed form's y in the variable s = (x - x0) / l, A_j = a_j l^j, j = 0..=n,
/// without forming a_j (which overflows near a fold) when the path is a fold
fn exact_a<T: TestReal>(p: Pth, x0: f64, l: f64, n: usize) -> Vec<T> {
    let x = T::from_f64(x0);
    let mut fact = vec![T::one(); n + 1];
    for k in 1..=n {
        fact[k] = fact[k - 1] * T::from_f64(k as f64);
    }
    let lp = |k: usize| (0..k).fold(1.0f64, |r, _| r * l);
    match p {
        Pth::B(br @ (Br::QA | Br::KA | Br::RA | Br::QB | Br::KB | Br::RB)) => {
            let neg = matches!(br, Br::QB | Br::KB | Br::RB);
            let sg = if neg { -1.0 } else { 1.0 };
            let e0 = (x * x).mulf(0.5 * sg).exp();
            // exp(sg x0^2 / 2) exp(sg x0 h) exp(sg h^2 / 2)
            (0..=n)
                .map(|k| {
                    let mut t = T::zero();
                    for i in 0..=k / 2 {
                        let j = k - 2 * i;
                        let xj = (0..j).fold(T::one(), |r, _| r * x.mulf(sg));
                        let term = xj / (fact[j] * fact[i] * T::from_f64(2f64.powi(i as i32)));
                        t = t + if (sg < 0.0) && i % 2 == 1 { -term } else { term };
                    }
                    (e0 * t).mulf(lp(k))
                })
                .collect()
        }
        Pth::B(Br::KC | Br::RC) => {
            let r = sqrt2::<T>();
            let (c0, s0) = cosh_sinh(r * x);
            (0..=n)
                .map(|k| {
                    let rk = (0..k).fold(T::one(), |q, _| q * r);
                    (if k % 2 == 0 { c0 } else { s0 } * rk / fact[k]).mulf(lp(k))
                })
                .collect()
        }
        Pth::B(Br::RD) => {
            let r = sqrt2::<T>();
            let u = r * x;
            let (c0, s0) = (u.cos(), u.sin());
            (0..=n)
                .map(|k| {
                    let rk = (0..k).fold(T::one(), |q, _| q * r);
                    let base = if k % 2 == 0 { c0 } else { -s0 };
                    let sgn = if (k / 2) % 2 == 1 { -1.0 } else { 1.0 };
                    (base * rk / fact[k]).mulf(sgn * lp(k))
                })
                .collect()
        }
        Pth::F1 | Pth::F2 => {
            // c (u0 - h)^(5/2), u0 = -x0: c u0^(5/2) b_k (-l / u0)^k, b_k = C(5/2, k)
            let u0 = -x;
            let q = T::from_f64(-l) / u0;
            let c = if p == Pth::F2 {
                T::from_f64(4.0) / T::from_f64(15.0)
            } else {
                T::one()
            };
            let lead = c * u0 * u0 * u0.sqrt();
            let mut b = T::one();
            let mut qk = T::one();
            (0..=n)
                .map(|k| {
                    if k > 0 {
                        b = b * (T::from_f64(2.5 - (k - 1) as f64) / T::from_f64(k as f64));
                        qk = qk * q;
                    }
                    lead * b * qk
                })
                .collect()
        }
    }
}

/// the Taylor series of (y, y', y'') in s from A_j = a_j l^j: y_k = A_k, y'_k = (k + 1) A_(k+1) / l, y''_k = (k + 1) (k + 2) A_(k+2) / l^2
fn exact_series<T: TestReal>(p: Pth, x0: f64, l: f64, kt: usize) -> [Vec<T>; 3] {
    let a = exact_a::<T>(p, x0, l, kt + 2);
    let il = 1.0 / l;
    [
        (0..=kt).map(|k| a[k]).collect(),
        (0..=kt).map(|k| a[k + 1].mulf((k + 1) as f64).mulf(il)).collect(),
        (0..=kt)
            .map(|k| a[k + 2].mulf(((k + 1) * (k + 2)) as f64).mulf(il).mulf(il))
            .collect(),
    ]
}

/// The accepted steps of a planted path, (x_i, l_i, planned h_i) at each step start, the equation, and the stop that ended
/// the path: `Stop::End` for a branch, the `Stop::Pivot` of a fold
fn path_steps<T: TestReal>(p: Pth, ctl: Ctl) -> (Vec<(f64, f64, f64)>, PolyEq, Stop) {
    let (eq, x0, y0, x1, o) = match p {
        Pth::B(br) => (
            eq_of(br),
            0.0,
            exact::<T>(br, T::zero()),
            3.0,
            opts::<T>(Branch::Project, 0.3),
        ),
        Pth::F1 => {
            let mut o = opts::<T>(Branch::Project, 1.0);
            o.pivot_abs = 1e-4;
            (
                PolyEq::f1(),
                -1.0,
                [T::one(), T::from_f64(-2.5), T::from_f64(3.75)],
                1.0,
                o,
            )
        }
        Pth::F2 => {
            let mut o = opts::<T>(Branch::Project, 1.0);
            o.pivot_abs = 1e-9;
            let (a, c) = (
                T::from_f64(4.0) / T::from_f64(15.0),
                T::from_f64(-2.0) / T::from_f64(3.0),
            );
            (PolyEq::f2(), -1.0, [a, c, T::one()], 1.0, o)
        }
    };
    let mut w = Walker::<T, _>::new(&eq, x0, y0, o);
    let mut pts = Vec::new();
    let mut stop = Stop::End;
    while (x1 - w.x()) > 0.0 {
        if ctl == Ctl::Cut(pts.len()) {
            stop = Stop::StepLimit;
            break;
        }
        if let Err(s) = w.form() {
            stop = s;
            break;
        }
        let h = w.planned();
        let at = (w.x(), w.series().l, h);
        if let Err(s) = w.advance(h, x1) {
            stop = s;
            break;
        }
        pts.push(at);
    }
    (pts, eq, stop)
}

/// The recursion's series against the closed form's at the planned step of each step of a path, started from the exact
/// state at the step's start: sum_k |dc_k| h^k <= 1e6 eps sum_k |c_k| h^k for each field (README, amendment A1). The
/// path must have ended where it is planned to (a branch with `Stop::End`, a fold with `Stop::Pivot`) after at least one
/// accepted step: a path that stopped early has been measured on a prefix, or on nothing.
fn step_sum<T: TestReal>(p: Pth, ctl: Ctl) -> Check {
    step_sum_run::<T>(p, ctl).0
}

/// `step_sum` and what it measured, for the digests: the stop and the step count, then for each planned step its x, l and
/// every coefficient of the series (all parts), and last `worst`
fn step_sum_run<T: TestReal>(p: Pth, ctl: Ctl) -> (Check, Vec<f64>) {
    let (pts, eq, stop) = path_steps::<T>(p, ctl);
    let mut data = vec![stop as u8 as f64, pts.len() as f64];
    let want = if matches!(p, Pth::B(_)) { Stop::End } else { Stop::Pivot };
    let kt = T::KT;
    let mut rec = Recursion::<T>::new(&eq.pat, kt);
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        rec.set_control(c);
    }
    let mut jets = Jets::<T>::new(eq.pat.len(), kt);
    let mut worst = 0.0f64;
    for &(x, l, h) in &pts {
        let ex = exact_series::<T>(p, x, l, kt);
        let y0 = [ex[0][0], ex[1][0], ex[2][0]];
        eq.jets(x, kt, &mut jets);
        let mut s = rec.form(&jets, l, y0).clone();
        if ctl == Ctl::NanCoeff {
            s.c[1][kt / 2] = T::from_f64(f64::NAN);
        }
        let sh = h / l;
        data.extend([x, l]);
        for c in s.c.iter().flatten() {
            data.extend(parts(*c));
        }
        for (cf, ef) in s.c.iter().zip(ex.iter()) {
            let (mut num, mut den, mut sk) = (0.0f64, 0.0f64, 1.0f64);
            for (c, e) in cf.iter().zip(ef.iter()).take(kt + 1) {
                num += (*c - *e).to_f64().abs() * sk;
                den += c.to_f64().abs() * sk;
                sk *= sh;
            }
            worst = nanmax(worst, num / (T::EPS * den));
        }
    }
    data.push(worst);
    let check = Check {
        pass: stop == want && !pts.is_empty() && worst <= 1e6,
        detail: format!(
            "{} {:?} {:?}: stop {:?} (wanted {:?}), {} steps, worst sum|dc_k| h^k / (eps sum|c_k| h^k) {:.2e} (bar 1e6)",
            T::NAME,
            p,
            ctl,
            stop,
            want,
            pts.len(),
            worst
        ),
    };
    (check, data)
}

fn show(c: &Check) {
    println!("[{}] {}", if c.pass { "PASS" } else { "FAIL" }, c.detail);
}

fn all_modes() -> [Branch; 2] {
    [Branch::Project, Branch::Conserve]
}

/// every cell is run and printed; the test fails if any is not met
fn finish(bad: Vec<String>) {
    assert!(bad.is_empty(), "{} cells not met:\n{}", bad.len(), bad.join("\n"));
}

/// the cells that are NOT MET by the criteria as written and are recorded, not gated (README, rulings of the coordinator):
/// `Conserve` on a branch that meets a double root with its mirror or with another root (K-B, R-B, R-D), where N = eps
/// splits the double root and the walk follows an avoided crossing; `Project` crosses
fn recorded(br: Br, mode: Branch) -> bool {
    mode == Branch::Conserve && matches!(br, Br::KB | Br::RB | Br::RD)
}

/// a cell that is gated must be met; a recorded one is printed and may be either
fn tally(bad: &mut Vec<String>, c: Check, recorded: bool) {
    if recorded {
        println!(
            "[{}] (recorded, not gated) {}",
            if c.pass { "met" } else { "NOT MET" },
            c.detail
        );
    } else {
        show(&c);
        if !c.pass {
            bad.push(c.detail);
        }
    }
}

fn closed_form_lane<T: TestReal>() {
    let mut bad = Vec::new();
    for br in BRANCHES {
        for mode in all_modes() {
            tally(&mut bad, closed_form::<T>(br, mode, Ctl::None), recorded(br, mode));
        }
    }
    finish(bad);
}

fn crossings_lane<T: TestReal>() {
    let mut bad = Vec::new();
    for br in [Br::QB, Br::KA, Br::KB, Br::KC, Br::RA, Br::RB, Br::RC] {
        for mode in all_modes() {
            tally(&mut bad, crossings::<T>(br, mode, Ctl::None), recorded(br, mode));
        }
    }
    finish(bad);
}

fn fold_lane<T: TestReal>() {
    let mut bad = Vec::new();
    for which in [1, 2] {
        for mode in all_modes() {
            // N drifts by about eps (sum |terms|) per step and moves the fold of N = const by that over dN/dx
            let rec = T::NAME == "f64" && which == 1 && mode == Branch::Conserve;
            tally(&mut bad, fold::<T>(which, mode, Ctl::None), rec);
        }
    }
    finish(bad);
}

fn misc_lane<T: TestReal>() {
    let mut bad = Vec::new();
    for l in [1.0, 0.25] {
        tally(&mut bad, coeffs::<T>(l, Ctl::None), false);
        // as worded (the relative error of every c_k) it is NOT MET in every lane: recorded, not gated
        tally(&mut bad, coeffs_each::<T>(l), true);
    }
    tally(&mut bad, zero_pivot::<T>(Ctl::None), false);
    tally(&mut bad, planner::<T>(Ctl::None), false);
    for br in BRANCHES {
        tally(&mut bad, step_sum::<T>(Pth::B(br), Ctl::None), false);
    }
    tally(&mut bad, step_sum::<T>(Pth::F1, Ctl::None), false);
    tally(&mut bad, step_sum::<T>(Pth::F2, Ctl::None), false);
    finish(bad);
}

/// Pass: `closed_form` on every branch, both modes (f64)
#[test]
fn closed_form_f64() {
    closed_form_lane::<f64>();
}

#[test]
fn closed_form_dd() {
    closed_form_lane::<DD>();
}

#[test]
fn closed_form_qd() {
    closed_form_lane::<QD>();
}

#[test]
fn crossings_f64() {
    crossings_lane::<f64>();
}

#[test]
fn crossings_dd() {
    crossings_lane::<DD>();
}

#[test]
fn crossings_qd() {
    crossings_lane::<QD>();
}

#[test]
fn fold_f64() {
    fold_lane::<f64>();
}

#[test]
fn fold_dd() {
    fold_lane::<DD>();
}

#[test]
fn fold_qd() {
    fold_lane::<QD>();
}

/// Pass, octo-double smoke (bar 1e6 eps = 2.3e-122, order 154 at tol 1e-126): K-A on both modes, K-B on `Project` with its
/// two crossings, the step sum of K-A, and F1 with its estimate
#[test]
fn octo_double() {
    let mut bad = Vec::new();
    for mode in all_modes() {
        tally(&mut bad, closed_form::<OD>(Br::KA, mode, Ctl::None), false);
    }
    tally(&mut bad, closed_form::<OD>(Br::KB, Branch::Project, Ctl::None), false);
    tally(&mut bad, crossings::<OD>(Br::KB, Branch::Project, Ctl::None), false);
    tally(&mut bad, step_sum::<OD>(Pth::B(Br::KA), Ctl::None), false);
    tally(&mut bad, fold::<OD>(1, Branch::Project, Ctl::None), false);
    finish(bad);
}

#[test]
fn misc_f64() {
    misc_lane::<f64>();
}

#[test]
fn misc_dd() {
    misc_lane::<DD>();
}

#[test]
fn misc_qd() {
    misc_lane::<QD>();
}

fn coeff_vec<T: Scalar>(eq: &PolyEq, x0: f64, y0: [T; 3], kt: usize, l: f64) -> [Vec<T>; 3] {
    let mut rec = Recursion::<T>::new(&eq.pat, kt);
    let mut jets = Jets::<T>::new(eq.pat.len(), kt);
    Coefficients::<T>::jets(eq, x0, kt, &mut jets);
    rec.form(&jets, l, y0).c.clone()
}

/// Pass: the recursion on `Fl<DV, 1>` and on the dual lane `Fl<Du, 1>` (value part) gives double-double's coefficients to
/// 1e2 eps_dd, for Q-A and K-C from x0 = 0.7. Every coefficient must be within it: a NaN is not.
fn lanes_check(br: Br, ctl: Ctl) -> Check {
    let kt = <DD as TestReal>::KT;
    let x0 = 0.7;
    let tol = 1e2 * <DD as Real>::EPS;
    let eq = eq_of(br);
    let y = exact::<DD>(br, DD::from_f64(x0));
    let c0 = coeff_vec::<DD>(&eq, x0, y, kt, 0.25);
    let yv = [
        Fl::<DV, 1>::splat(y[0]),
        Fl::<DV, 1>::splat(y[1]),
        Fl::<DV, 1>::splat(y[2]),
    ];
    let yu = [
        Fl::<Du, 1>::splat(y[0]),
        Fl::<Du, 1>::splat(y[1]),
        Fl::<Du, 1>::splat(y[2]),
    ];
    let mut cv = coeff_vec::<Fl<DV, 1>>(&eq, x0, yv, kt, 0.25);
    let mut cu = coeff_vec::<Fl<Du, 1>>(&eq, x0, yu, kt, 0.25);
    match ctl {
        // an error far above the tolerance in one coefficient of each lane
        Ctl::LanePerturb => {
            cv[0][kt / 2] = cv[0][kt / 2] + Fl::<DV, 1>::splat(DD::from_f64(1e-20));
            cu[0][kt / 2] = cu[0][kt / 2] + Fl::<Du, 1>::splat(DD::from_f64(1e-20));
        }
        Ctl::NanLane => {
            cv[1][kt / 2] = cv[1][kt / 2] + Fl::<DV, 1>::splat(DD::from_f64(f64::NAN));
            cu[2][kt / 2] = cu[2][kt / 2] + Fl::<Du, 1>::splat(DD::from_f64(f64::NAN));
        }
        _ => {}
    }
    let (mut wv, mut wu) = (0.0f64, 0.0f64);
    for f in 0..3 {
        for k in 0..=kt {
            let r = c0[f][k];
            let sc = r.to_f64().abs().max(1e-300);
            wv = nanmax(wv, ((cv[f][k].get(0) - r).to_f64()).abs() / sc);
            wu = nanmax(wu, ((cu[f][k].val().get(0) - r).to_f64()).abs() / sc);
        }
    }
    Check {
        pass: wv <= tol && wu <= tol,
        detail: format!(
            "lanes {:?} {:?}: Fl<DV,1> {:.2e}, Fl<Du,1> value {:.2e} (bar {:.1e})",
            br, ctl, wv, wu, tol
        ),
    }
}

#[test]
fn lanes() {
    let mut bad = Vec::new();
    for br in [Br::QA, Br::KC] {
        tally(&mut bad, lanes_check(br, Ctl::None), false);
    }
    finish(bad);
}

/// A coefficient of a lane off by 1e-20, or a NaN, fails the lane check
#[test]
fn lanes_controls() {
    for br in [Br::QA, Br::KC] {
        for ctl in [Ctl::LanePerturb, Ctl::NanLane] {
            let c = lanes_check(br, ctl);
            show(&c);
            assert!(!c.pass, "control {ctl:?} did not fire: {}", c.detail);
        }
    }
}

/// the `n` leading parts of a lane value: each is the leading f64 of what the ones before leave
fn parts_n<T: TestReal>(v: T, n: usize) -> Vec<f64> {
    let mut r = v;
    (0..n)
        .map(|_| {
            let o = r.to_f64();
            r = r - T::from_f64(o);
            o
        })
        .collect()
}

/// all the parts of a lane value that the digests hash
fn parts<T: TestReal>(v: T) -> Vec<f64> {
    parts_n(v, T::PARTS)
}

/// FNV-1a over the bits of f64s
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf29ce484222325)
    }
    fn put(&mut self, x: f64) {
        for b in x.to_bits().to_le_bytes() {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    fn put_all(&mut self, xs: &[f64]) {
        xs.iter().for_each(|&x| self.put(x));
    }
}

/// What a control of `digests` changes in the result before it is hashed
#[derive(Clone, Copy, Debug, PartialEq)]
enum Poke {
    None,
    /// the last bit of the first checkpoint's y: `full` must change, and `base` does not
    Checkpoint,
    /// the first cell's end y[0] times (1 + 2^-300), a change below its fourth part: `base` and `full` must change
    EndLower,
    /// the last bit of the last part of the first coefficient of the first step-sum series: `full` must change, and
    /// `base` does not
    SumLast,
}

/// The digests of the planted results of some cells, folds and step-sum paths, two of them, to compare across the SIMD
/// backends bit for bit:
/// - `base` hashes the end state, the crossings' positions and the step count of each cell, and the end, step count and
///   zero estimate of each fold, all parts of a value (`TestReal::PARTS`): the first version of this check, whose values
///   #190 and #214 quote for f64, dd and qd;
/// - `full` hashes those too, and what `base` leaves out: the state and step count at every checkpoint, each stop, each
///   crossing's step and whether it was located, `n_rel_max` and `n_rel_pre_max`, a fold's `beta` and stop, and for each
///   path of `sums` what `step_sum_run` measured, every coefficient of every planned step's series included.
///
/// `poke` is a control: it changes one thing in the result.
fn digests<T: TestReal>(cells: &[(Br, Branch)], folds: &[u8], sums: &[Pth], poke: Poke) -> (u64, u64) {
    let (mut base, mut full) = (Fnv::new(), Fnv::new());
    for (ci, &(br, mode)) in cells.iter().enumerate() {
        let c = run_case::<T>(br, mode, Ctl::None);
        let mut y = c.out.y;
        if poke == Poke::EndLower && ci == 0 {
            y[0] = y[0] + y[0].mulf(2f64.powi(-300));
        }
        for h in [&mut base, &mut full] {
            for v in y {
                h.put_all(&parts(v));
            }
            for k in &c.out.crossings {
                h.put_all(&parts(k.x));
            }
            h.put(c.out.steps as f64);
        }
        full.put(c.out.stop as u8 as f64);
        full.put_all(&[c.out.n_rel_max, c.out.n_rel_pre_max]);
        for k in &c.out.crossings {
            full.put_all(&[k.step as f64, k.located as u8 as f64]);
        }
        for (i, (x, y, n)) in c.out.cps.iter().enumerate() {
            full.put(*x);
            for (f, v) in y.iter().enumerate() {
                let mut p = parts(*v);
                if poke == Poke::Checkpoint && i == 0 && f == 0 {
                    p[0] = f64::from_bits(p[0].to_bits() ^ 1);
                }
                full.put_all(&p);
            }
            full.put(*n as f64);
        }
    }
    for &which in folds {
        let (eq, y0, pivot) = if which == 1 {
            (PolyEq::f1(), [T::one(), T::from_f64(-2.5), T::from_f64(3.75)], 1e-4)
        } else {
            let (a, c) = (
                T::from_f64(4.0) / T::from_f64(15.0),
                T::from_f64(-2.0) / T::from_f64(3.0),
            );
            (PolyEq::f2(), [a, c, T::one()], 1e-9)
        };
        let mut o = opts::<T>(Branch::Project, 1.0);
        o.pivot_abs = pivot;
        let out = Walker::<T, _>::new(&eq, -1.0, y0, o).run(1.0, &[]);
        for h in [&mut base, &mut full] {
            h.put(out.x);
            h.put(out.steps as f64);
            if let Some(z) = &out.zero {
                h.put_all(&parts(z.x));
            }
        }
        full.put(out.stop as u8 as f64);
        if let Some(z) = &out.zero {
            full.put(z.beta);
        }
    }
    for &p in sums {
        let (_, mut data) = step_sum_run::<T>(p, Ctl::None);
        if poke == Poke::SumLast {
            // data: stop, steps, then x, l and the parts of c[0][0] of the first step
            let at = 3 + T::PARTS;
            data[at] = f64::from_bits(data[at].to_bits() ^ 1);
        }
        full.put_all(&data);
    }
    (base.0, full.0)
}

/// every cell of the planted walks, and both folds
fn all_cells() -> Vec<(Br, Branch)> {
    BRANCHES
        .iter()
        .flat_map(|&br| all_modes().into_iter().map(move |m| (br, m)))
        .collect()
}

/// Pass: the two digests of a lane (`digest NAME base`, `digest-full NAME full`) equal `TestReal::DIGEST`, on every backend
/// and target the suite runs on (the CI's SSE2, portable, AVX2 with `-C target-feature=+avx2,+fma`, NEON and MSVC jobs)
fn digest_lane<T: TestReal>() {
    let (base, full) = digests::<T>(&all_cells(), &[1, 2], &[], Poke::None);
    println!("digest {} {:016x}", T::NAME, base);
    println!("digest-full {} {:016x}", T::NAME, full);
    check_digests::<T>(base, full);
}

fn check_digests<T: TestReal>(base: u64, full: u64) {
    assert!(
        [base, full] == T::DIGEST,
        "{}: digests {base:016x} {full:016x}, pinned {:016x} {:016x}",
        T::NAME,
        T::DIGEST[0],
        T::DIGEST[1]
    );
}

#[test]
fn digest_f64() {
    digest_lane::<f64>();
}

#[test]
fn digest_dd() {
    digest_lane::<DD>();
}

#[test]
fn digest_qd() {
    digest_lane::<QD>();
}

/// the octo-double results of `octo_double`: the cells K-A on both modes and K-B on `Project`, the fold F1 and the step
/// sum of K-A
fn od_digests(poke: Poke) -> (u64, u64) {
    let cells = [
        (Br::KA, Branch::Project),
        (Br::KA, Branch::Conserve),
        (Br::KB, Branch::Project),
    ];
    digests::<OD>(&cells, &[1], &[Pth::B(Br::KA)], poke)
}

/// Pass: the two digests of the octo-double results, all eight parts of each value, equal `TestReal::DIGEST`
#[test]
fn digest_od() {
    let (base, full) = od_digests(Poke::None);
    println!("digest {} {:016x}", <OD as TestReal>::NAME, base);
    println!("digest-full {} {:016x}", <OD as TestReal>::NAME, full);
    check_digests::<OD>(base, full);
}

/// Control of the pinned digests: the flipped checkpoint bit makes the comparison with the pinned `full` fail, and the
/// pinned `base` still matches
#[test]
fn digest_pinned_fails_on_a_flip() {
    let (base, full) = digests::<DD>(&all_cells(), &[1, 2], &[], Poke::Checkpoint);
    assert_eq!(base, <DD as TestReal>::DIGEST[0]);
    assert_ne!(
        full,
        <DD as TestReal>::DIGEST[1],
        "the pinned digest does not see one flipped bit"
    );
}

/// Control of the digests: one flipped bit of one checkpoint's state changes `full` and not `base`, which cannot see a
/// checkpoint; it is the case `full` was added for
#[test]
fn digest_sees_a_checkpoint() {
    let cells = [(Br::KA, Branch::Project)];
    let (b0, f0) = digests::<DD>(&cells, &[], &[], Poke::None);
    let (b1, f1) = digests::<DD>(&cells, &[], &[], Poke::Checkpoint);
    println!("flipped bit of a checkpoint: base {b0:016x} -> {b1:016x}, full {f0:016x} -> {f1:016x}");
    assert_eq!(b0, b1, "base hashes a checkpoint");
    assert_ne!(f0, f1, "full does not see a checkpoint");
}

/// Control of `parts` for an octo-double: a value changed below its fourth part has the same first four parts and
/// different eight (`TestReal::PARTS`), and the eight are the whole value (the ninth is 0, the eighth is not)
#[test]
fn parts_see_a_lower_part() {
    let v = OD::one() / OD::from_f64(3.0);
    let w = v + v.mulf(2f64.powi(-300));
    assert_eq!(parts_n(v, 4), parts_n(w, 4), "the change reaches the first four parts");
    assert_ne!(
        parts(v),
        parts(w),
        "the parts hashed do not see a change below the fourth"
    );
    let all = parts_n(w, 9);
    assert!(all[7] != 0.0 && all[8] == 0.0, "an od is not its eight parts: {all:?}");
}

/// Control of the octo-double digests, from one run to the next: a change below the fourth part of an end state changes
/// `base` and `full`, and one bit of a step-sum series' coefficient changes `full` and not `base`
#[test]
fn digest_od_sees_the_lower_parts() {
    let (b0, f0) = od_digests(Poke::None);
    let (b1, f1) = od_digests(Poke::EndLower);
    let (b2, f2) = od_digests(Poke::SumLast);
    println!("lower part of an end state: base {b0:016x} -> {b1:016x}, full {f0:016x} -> {f1:016x}");
    println!("last part of a step-sum coefficient: base {b0:016x} -> {b2:016x}, full {f0:016x} -> {f2:016x}");
    assert_ne!(b0, b1, "base does not see a lower part of an end state");
    assert_ne!(f0, f1, "full does not see a lower part of an end state");
    assert_eq!(b0, b2, "base hashes a step sum");
    assert_ne!(f0, f2, "full does not see a step sum");
}

/// Pass (review of `Recursion::form`, `l^j` in `f64`): N = y'' - 2^1000 x^2 about x0 = 0 with l = 2^-600 has
/// `y''_2 = 2^1000 l^2 = 2^-200` within the bar of f64, though `l^2 = 2^-1200` is not an `f64`
fn underflow(ctl: Ctl) -> Check {
    let eq = RealEq::new(&[((0, 0, 1), &[1.0]), ((0, 0, 0), &[0.0, 0.0, -pow2(1000)])]);
    let kt = 4;
    let mut rec = Recursion::<f64>::new(&eq.pat, kt);
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        rec.set_control(c);
    }
    let mut jets = Jets::<f64>::new(eq.pat.len(), kt);
    Coefficients::<f64>::jets(&eq, 0.0, kt, &mut jets);
    let got = rec.form(&jets, pow2(-600), [0.0; 3]).c[2][2];
    let want = pow2(-200);
    let err = ((got - want) / want).abs();
    Check {
        pass: err <= bar::<f64>(),
        detail: format!("f64 {ctl:?}: y''_2 = {got:e} for 2^-200 = {want:e}, relative error {err:.2e}"),
    }
}

/// y'' = 1
fn unit_eq() -> PolyEq {
    PolyEq::new(&[((0, 0, 1), &[1]), ((0, 0, 0), &[-1])])
}

/// N = y''^2 - x^2, whose branch y'' = x meets its mirror y'' = -x at x = 0 (a double root of N in y'' there)
fn mirror_eq() -> PolyEq {
    PolyEq::new(&[((0, 0, 2), &[1]), ((0, 0, 0), &[0, 0, -1])])
}

/// the state of its branch y'' = x at x: y = x^3 / 6, y' = x^2 / 2
fn mirror_state<T: TestReal>(x: f64) -> [T; 3] {
    let x = T::from_f64(x);
    [x * x * x / T::from_f64(6.0), (x * x).mulf(0.5), x]
}

/// whether two states are equal in every part
fn same<T: TestReal>(a: &[T; 3], b: &[T; 3]) -> bool {
    (0..3).all(|f| parts(a[f]) == parts(b[f]))
}

/// Pass (review of `Walker::advance`): a step uses the series of the `form` for its state and no other. For y'' = 1 from
/// [0, 0, 1] at x = 0: `form`, `advance` to 1, and `advance` again without a `form` is `Stop::NotFormed` and leaves
/// the walk where it was (a stale series would sum the series about 0 from 1 to 2 and give [1/2, 1, 1], not the state at
/// 2); with a `form` before each `advance` the walk reaches [2, 2, 1] within the bar. `advance` after `set_y`, and after
/// a `form` that stopped at a pivot, is `NotFormed` as well.
fn stale<T: TestReal>(ctl: Ctl) -> Check {
    let eq = unit_eq();
    let mut w = Walker::<T, _>::new(
        &eq,
        0.0,
        [T::zero(), T::zero(), T::one()],
        opts::<T>(Branch::Project, 1.0),
    );
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        w.set_control(c);
    }
    let mut ok = w.form().is_ok() && w.advance(1.0, 3.0).is_ok();
    let (x1, y1) = (w.x(), w.y());
    let refused = w.advance(1.0, 3.0) == Err(Stop::NotFormed) && w.x() == x1 && same(&w.y(), &y1);
    ok &= w.form().is_ok() && w.advance(1.0, 3.0).is_ok();
    let want = [T::from_f64(2.0), T::from_f64(2.0), T::one()];
    let err = rel_err(&w.y(), &want);
    ok &= w.form().is_ok();
    w.set_y(w.y());
    let after_set = w.advance(1.0, 3.0) == Err(Stop::NotFormed);
    let mut o = opts::<T>(Branch::Project, 1.0);
    o.pivot_abs = 1e-4;
    let y0 = [T::one(), T::from_f64(-2.5), T::from_f64(3.75)];
    let f1 = PolyEq::f1();
    let mut f = Walker::<T, _>::new(&f1, -1.0, y0, o);
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        f.set_control(c);
    }
    let stopped = f.run(1.0, &[]).stop == Stop::Pivot;
    let h = f.planned();
    let after_stop = stopped && f.advance(h, 1.0) == Err(Stop::NotFormed);
    Check {
        pass: ok && refused && err <= bar::<T>() && after_set && after_stop,
        detail: format!(
            "{} {:?}: advance without form refused {}, walk to 2 off by {:.2e}, refused after set_y {}, after a stop {}",
            T::NAME,
            ctl,
            refused,
            err,
            after_set,
            after_stop
        ),
    }
}

/// Pass (review of `Walker::form`, a crossing once per step): on the branch y'' = x of N = y''^2 - x^2 a single step from -1
/// to 1 crosses the double root at 0. After `form`, `advance`, then `form` twice at x = 1, the walk reports one crossing,
/// located at 0 within the bar.
fn repeat_form<T: TestReal>(ctl: Ctl) -> Check {
    let eq = mirror_eq();
    let mut w = Walker::<T, _>::new(&eq, -1.0, mirror_state::<T>(-1.0), opts::<T>(Branch::Project, 4.0));
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        w.set_control(c);
    }
    let mut ok = w.form().is_ok();
    let h = w.planned();
    ok &= w.advance(h, 1.0).is_ok() && w.steps() == 1;
    ok &= w.form().is_ok() && w.form().is_ok();
    let out = w.run(1.0, &[]);
    let n = out.crossings.len();
    let pos = out.crossings.first().map_or(f64::NAN, |k| k.x.to_f64().abs());
    Check {
        pass: ok && n == 1 && out.crossings[0].located && pos <= bar::<T>(),
        detail: format!(
            "{} {:?}: {} crossings after two forms at the end of one step, the first at {:.2e} from 0",
            T::NAME,
            ctl,
            n,
            pos
        ),
    }
}

/// Pass (review of `Walker::run`, the last step): the step of `repeat_form`'s walk from -1 to 1 is the last one. `run`
/// ends with `Stop::End` in one step, to the state of the branch within the bar, and reports the crossing at 0 (located,
/// within the bar), which no step start sees.
fn last_crossing<T: TestReal>(ctl: Ctl) -> Check {
    let eq = mirror_eq();
    let mut w = Walker::<T, _>::new(&eq, -1.0, mirror_state::<T>(-1.0), opts::<T>(Branch::Project, 4.0));
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        w.set_control(c);
    }
    let out = w.run(1.0, &[]);
    let err = rel_err(&out.y, &mirror_state::<T>(1.0));
    let n = out.crossings.len();
    let pos = out.crossings.first().map_or(f64::NAN, |k| k.x.to_f64().abs());
    Check {
        pass: out.stop == Stop::End
            && out.steps == 1
            && out.x == 1.0
            && err <= bar::<T>()
            && n == 1
            && out.crossings[0].located
            && pos <= bar::<T>(),
        detail: format!(
            "{} {:?}: stop {:?} after {} steps, state off by {:.2e}, {} crossings, the first {:.2e} from 0",
            T::NAME,
            ctl,
            out.stop,
            out.steps,
            err,
            n,
            pos
        ),
    }
}

/// Pass (review of `Walker::run`, equal checkpoints): `run(2, [1, 1])` of y'' = 1 from x = 0 ends with `Stop::End`, lands
/// once on 1 and gives the state and the step count of `run(2, [1])`, bit for bit.
fn dup_checkpoints<T: TestReal>(ctl: Ctl) -> Check {
    let eq = unit_eq();
    let run = |cps: &[f64]| {
        let y0 = [T::zero(), T::zero(), T::one()];
        let mut w = Walker::<T, _>::new(&eq, 0.0, y0, opts::<T>(Branch::Project, 0.3));
        #[cfg(feature = "mutate")]
        if let Ctl::Lib(c) = ctl {
            w.set_control(c);
        }
        w.run(2.0, cps)
    };
    let (twice, once) = (run(&[1.0, 1.0]), run(&[1.0]));
    Check {
        pass: twice.stop == Stop::End
            && twice.x == 2.0
            && twice.cps.len() == 1
            && twice.cps[0].0 == 1.0
            && twice.steps == once.steps
            && same(&twice.y, &once.y),
        detail: format!(
            "{} {:?}: stop {:?} at {}, {} landings, {} steps against {} with the checkpoint once",
            T::NAME,
            ctl,
            twice.stop,
            twice.x,
            twice.cps.len(),
            twice.steps,
            once.steps
        ),
    }
}

/// Pass (review of `Walker::run`, the estimate of the zero): the fold F1 stops with `Stop::Pivot` and the estimate; a second
/// `run` from where it stopped, to the same x, ends with `Stop::End` and no estimate.
fn zero_only_pivot<T: TestReal>(ctl: Ctl) -> Check {
    let eq = PolyEq::f1();
    let mut o = opts::<T>(Branch::Project, 1.0);
    o.pivot_abs = 1e-4;
    let y0 = [T::one(), T::from_f64(-2.5), T::from_f64(3.75)];
    let mut w = Walker::<T, _>::new(&eq, -1.0, y0, o);
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        w.set_control(c);
    }
    let first = w.run(1.0, &[]);
    let again = w.run(w.x(), &[]);
    Check {
        pass: first.stop == Stop::Pivot && first.zero.is_some() && again.stop == Stop::End && again.zero.is_none(),
        detail: format!(
            "{} {:?}: first run {:?} with an estimate {}, second run {:?} with one {}",
            T::NAME,
            ctl,
            first.stop,
            first.zero.is_some(),
            again.stop,
            again.zero.is_some()
        ),
    }
}

/// Pass (review of `taylor_shift`): an empty polynomial writes nothing and does not panic, a constant keeps its value, and
/// 1 + 2x + 3x^2 about 2 is 17 + 14 h + 3 h^2
fn shift_edges<T: TestReal>(ctl: Ctl) -> Check {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let call = |poly: &[T], x0: f64, out: &mut [T]| {
        #[cfg(feature = "mutate")]
        if let Ctl::Lib(c) = ctl {
            return taylor_shift_with(poly, x0, out, c);
        }
        taylor_shift(poly, x0, out)
    };
    let empty = catch_unwind(AssertUnwindSafe(|| call(&[], 0.5, &mut []))).is_ok();
    let mut one = [T::from_f64(-1.0)];
    call(&[T::from_f64(7.0)], 3.0, &mut one);
    let mut quad = [T::zero(); 3];
    call(&[T::one(), T::from_f64(2.0), T::from_f64(3.0)], 2.0, &mut quad);
    let want = [17.0, 14.0, 3.0];
    let (const_ok, quad_ok) = (one[0].to_f64() == 7.0, (0..3).all(|j| quad[j].to_f64() == want[j]));
    Check {
        pass: empty && const_ok && quad_ok,
        detail: format!(
            "{} {:?}: empty polynomial returns {}, constant kept {}, 1 + 2x + 3x^2 about 2 exact {}",
            T::NAME,
            ctl,
            empty,
            const_ok,
            quad_ok
        ),
    }
}

/// Pass (review of `taylor_shift`, a shift by 0): [0, 0, 2^1023] about 0 and about -0 is itself, not a NaN from the
/// binomial 2 times 2^1023 times 0
fn shift_zero_point<T: TestReal>(ctl: Ctl) -> Check {
    let poly = [T::zero(), T::zero(), T::from_f64(pow2(1023))];
    let ok: Vec<bool> = [0.0, -0.0]
        .iter()
        .map(|&x0| {
            let mut out = [T::from_f64(-1.0); 3];
            taylor_shift(&poly, x0, &mut out);
            (0..3).all(|j| out[j].to_f64() == poly[j].to_f64())
        })
        .collect();
    Check {
        pass: ok.iter().all(|&b| b),
        detail: format!(
            "{} {:?}: [0, 0, 2^1023] kept about 0 {} and about -0 {}",
            T::NAME,
            ctl,
            ok[0],
            ok[1]
        ),
    }
}

/// N = A(x) y'' with A = -1 before x = 1 and +inf from there: the pole of a pivot -1/(1 - x) read at 1, where the pivot
/// has been negative throughout
struct PoleEq;

impl<T: Scalar> Coefficients<T> for PoleEq {
    fn pattern(&self) -> Vec<Mono> {
        vec![(0, 0, 1)]
    }
    fn jets(&self, x0: f64, _kt: usize, out: &mut Jets<T>) {
        out.ent[0].c[0] = T::from_f64(if x0 < 1.0 { -1.0 } else { f64::INFINITY });
        out.ent[0].n = 1;
    }
}

/// Pass (review of `Walker::check_end`, a non-finite pivot at the end): the zero branch of N = A(x) y'' with `PoleEq`'s A
/// from 0 to 1 is one step to the pole. The pivot is negative throughout, so the walk ends with `Stop::End` at 1 and
/// reports no crossing.
fn end_pole<T: TestReal>(ctl: Ctl) -> Check {
    let mut w = Walker::<T, _>::new(&PoleEq, 0.0, [T::zero(); 3], opts::<T>(Branch::Project, 4.0));
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        w.set_control(c);
    }
    let out = w.run(1.0, &[]);
    Check {
        pass: out.stop == Stop::End && out.x == 1.0 && out.steps == 1 && out.crossings.is_empty(),
        detail: format!(
            "{} {:?}: stop {:?} at {} after {} steps, {} crossings (none expected)",
            T::NAME,
            ctl,
            out.stop,
            out.x,
            out.steps,
            out.crossings.len()
        ),
    }
}

/// the edge cases of the review of multishoot#73 that hold in every lane
fn edges_lane<T: TestReal>() {
    let mut bad = Vec::new();
    tally(&mut bad, stale::<T>(Ctl::None), false);
    tally(&mut bad, repeat_form::<T>(Ctl::None), false);
    tally(&mut bad, last_crossing::<T>(Ctl::None), false);
    tally(&mut bad, dup_checkpoints::<T>(Ctl::None), false);
    tally(&mut bad, zero_only_pivot::<T>(Ctl::None), false);
    tally(&mut bad, shift_edges::<T>(Ctl::None), false);
    tally(&mut bad, shift_zero_point::<T>(Ctl::None), false);
    tally(&mut bad, end_pole::<T>(Ctl::None), false);
    finish(bad);
}

/// Pass (review of `Recursion::form`, an overflowing reciprocal): N = 2^-1030 (y'' - x) about x0 = 0, l = 1, from the zero
/// state. The pivot 2^-1030 is a finite f64 whose reciprocal is not; the series of y'' = x (y''_1 = 1, y'_2 = 1/2,
/// y_3 = 1/6, the rest 0) is found within the bar of f64.
fn reciprocal(ctl: Ctl) -> Check {
    let c = pow2(-1030);
    let eq = RealEq::new(&[((0, 0, 1), &[c]), ((0, 0, 0), &[0.0, -c])]);
    let kt = 4;
    let mut rec = Recursion::<f64>::new(&eq.pat, kt);
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        rec.set_control(c);
    }
    let mut jets = Jets::<f64>::new(eq.pat.len(), kt);
    Coefficients::<f64>::jets(&eq, 0.0, kt, &mut jets);
    let s = rec.form(&jets, 1.0, [0.0; 3]).clone();
    let want = [
        [0.0, 0.0, 0.0, 1.0 / 6.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
    ];
    let mut worst = 0.0f64;
    for (cf, wf) in s.c.iter().zip(want.iter()) {
        for (c, w) in cf.iter().zip(wf.iter()) {
            worst = nanmax(worst, (c - w).abs());
        }
    }
    Check {
        pass: worst <= bar::<f64>(),
        detail: format!(
            "f64 {ctl:?}: pivot 2^-1030, y''_1 = {:e}, worst difference from y'' = x's series {worst:.2e}",
            s.c[2][1]
        ),
    }
}

/// Pass (review of `Walker::form`, the zero estimate's exponent): N = 2^-550 (1 - x) y'' from the zero state at x = 0 has
/// A0 = 2^-550 (1 - x), linear, zero at x = 1. With `pivot_abs` 1e-100 the walk stops at once with `Stop::Pivot`; the
/// estimate is x = 1 with beta = 1, though the square of A0's slope, 2^-1100, is not an `f64`.
fn rho_scale(ctl: Ctl) -> Check {
    let c = pow2(-550);
    let eq = RealEq::new(&[((0, 0, 1), &[c, -c])]);
    let mut o = opts::<f64>(Branch::Project, 1.0);
    o.pivot_abs = 1e-100;
    let mut w = Walker::<f64, _>::new(&eq, 0.0, [0.0; 3], o);
    #[cfg(feature = "mutate")]
    if let Ctl::Lib(c) = ctl {
        w.set_control(c);
    }
    let out = w.run(0.5, &[]);
    let (x, beta) = out.zero.as_ref().map_or((f64::NAN, f64::NAN), |z| (z.x, z.beta));
    Check {
        pass: out.stop == Stop::Pivot && (x - 1.0).abs() <= bar::<f64>() && (beta - 1.0).abs() <= bar::<f64>(),
        detail: format!(
            "f64 {ctl:?}: stop {:?}, estimate x = {x}, beta = {beta} for x = 1, beta = 1",
            out.stop
        ),
    }
}

/// the edge cases of the review of multishoot#73 whose exponents are beyond what the multi-doubles' splits take: f64 only
#[test]
fn edges_f64() {
    edges_lane::<f64>();
    let mut bad = Vec::new();
    tally(&mut bad, underflow(Ctl::None), false);
    tally(&mut bad, reciprocal(Ctl::None), false);
    tally(&mut bad, rho_scale(Ctl::None), false);
    finish(bad);
}

#[test]
fn edges_dd() {
    edges_lane::<DD>();
}

#[test]
fn edges_qd() {
    edges_lane::<QD>();
}

/// a NaN coefficient fails the step sum, on a branch and on a fold
fn nan_coeff_fires<T: TestReal>() {
    for p in [Pth::B(Br::KA), Pth::F1] {
        let c = step_sum::<T>(p, Ctl::NanCoeff);
        show(&c);
        assert!(!c.pass, "control NanCoeff did not fire: {}", c.detail);
        assert!(step_sum::<T>(p, Ctl::None).pass);
    }
}

/// a path that stops early, at its first `form` or after three steps, fails the step sum, on a branch and on a fold
fn cut_fires<T: TestReal>() {
    for p in [Pth::B(Br::KA), Pth::F1] {
        for n in [0, 3] {
            let c = step_sum::<T>(p, Ctl::Cut(n));
            show(&c);
            assert!(!c.pass, "control Cut({n}) did not fire: {}", c.detail);
        }
    }
}

/// the gated `coeffs` fails on a coefficient that is off by 10 bar of the series' size
fn bad_coeff_fires<T: TestReal>() {
    for l in [1.0, 0.25] {
        let c = coeffs::<T>(l, Ctl::BadCoeff);
        show(&c);
        assert!(!c.pass, "control BadCoeff did not fire: {}", c.detail);
    }
}

/// the planner check fails when the planned step differs from the rule's in the last bit, in every step
fn flip_bit_fires<T: TestReal>() {
    let c = planner::<T>(Ctl::FlipBit);
    show(&c);
    assert!(
        !c.pass && c.detail.contains("5 of 5"),
        "control FlipBit did not fire: {}",
        c.detail
    );
    assert!(planner::<T>(Ctl::None).pass);
}

/// a NaN position of a crossing fails `crossings`, and a NaN end state fails `closed_form`
fn nan_position_fires<T: TestReal>() {
    for (c, name) in [
        (crossings::<T>(Br::KA, Branch::Project, Ctl::NanPos), "NanPos"),
        (closed_form::<T>(Br::KA, Branch::Project, Ctl::NanState), "NanState"),
    ] {
        show(&c);
        assert!(!c.pass, "control {name} did not fire: {}", c.detail);
    }
    assert!(crossings::<T>(Br::KA, Branch::Project, Ctl::None).pass);
    assert!(closed_form::<T>(Br::KA, Branch::Project, Ctl::None).pass);
}

/// The controls that need no hook: the check must fail on its stated case
#[test]
fn controls_without_hooks() {
    flip_bit_fires::<f64>();
    flip_bit_fires::<DD>();
    flip_bit_fires::<QD>();
    nan_position_fires::<f64>();
    nan_position_fires::<DD>();
    nan_position_fires::<QD>();
    bad_coeff_fires::<f64>();
    bad_coeff_fires::<DD>();
    bad_coeff_fires::<QD>();
    cut_fires::<f64>();
    cut_fires::<DD>();
    nan_coeff_fires::<f64>();
    nan_coeff_fires::<DD>();
    for mode in all_modes() {
        for ctl in [Ctl::WrongStart, Ctl::SwitchMid, Ctl::WrongForm] {
            let c = closed_form::<DD>(Br::KA, mode, ctl);
            show(&c);
            assert!(!c.pass, "control {ctl:?} did not fire: {}", c.detail);
        }
    }
}

#[cfg(feature = "mutate")]
mod controls {
    use super::*;

    /// the checks a control is run against, in the order of the declared rows
    const CHECKS: [&str; 6] = [
        "step_sum",
        "closed_form",
        "crossings",
        "fold F1",
        "fold F2",
        "zero_pivot",
    ];

    /// whether each check fails with the control (K-A, R-D, K-B on `Project`, the folds on `Project`)
    fn fires<T: TestReal>(c: ImplicitControl) -> [bool; 6] {
        let l = Ctl::Lib(c);
        [
            !step_sum::<T>(Pth::B(Br::KA), l).pass,
            !closed_form::<T>(Br::KA, Branch::Project, l).pass || !closed_form::<T>(Br::RD, Branch::Project, l).pass,
            !crossings::<T>(Br::KB, Branch::Project, l).pass,
            !fold::<T>(1, Branch::Project, l).pass,
            !fold::<T>(2, Branch::Project, l).pass,
            !zero_pivot::<T>(l).pass,
        ]
    }

    /// The rows declared after the development run `dev_03_implicit` (README, amendment A2): the checks each control must
    /// fail, and no others. `RecipDD` rounds to double-double: a no-op in f64 and double-double, an error in quad-double.
    fn declared<T: TestReal>(c: ImplicitControl) -> [bool; 6] {
        match c {
            ImplicitControl::SkipOrder(_) => [true, true, true, true, true, false],
            ImplicitControl::RecipDD => {
                if T::NAME == "qd" {
                    [true, true, true, false, false, false]
                } else {
                    [false; 6]
                }
            }
            ImplicitControl::NoPivotCheck => [false, false, false, false, false, true],
            ImplicitControl::NoCrossing => [false, false, true, false, false, false],
            ImplicitControl::ExponentHalf => [false, false, false, false, true, false],
            ImplicitControl::NoPivotStop => [false, false, false, true, true, false],
            ImplicitControl::None
            | ImplicitControl::PlainScale
            | ImplicitControl::StaleSeries
            | ImplicitControl::RepeatCrossing
            | ImplicitControl::NoEndCrossing
            | ImplicitControl::NoDedup
            | ImplicitControl::KeepZero
            | ImplicitControl::EmptyShift
            | ImplicitControl::PlainReciprocal
            | ImplicitControl::PlainRho => [false; 6],
        }
    }

    /// Pass: every control fails exactly the checks declared for it, in each lane
    fn controls_lane<T: TestReal>() {
        let mut bad = Vec::new();
        for c in [
            ImplicitControl::SkipOrder(3),
            ImplicitControl::RecipDD,
            ImplicitControl::NoPivotCheck,
            ImplicitControl::NoCrossing,
            ImplicitControl::ExponentHalf,
            ImplicitControl::NoPivotStop,
        ] {
            let (got, want) = (fires::<T>(c), declared::<T>(c));
            let on: Vec<&str> = (0..6).filter(|&i| got[i]).map(|i| CHECKS[i]).collect();
            let ok = got == want;
            println!(
                "[{}] control {:?} in {}: fails {:?}",
                if ok { "FIRED" } else { "MISMATCH" },
                c,
                T::NAME,
                on
            );
            if !ok {
                let w: Vec<&str> = (0..6).filter(|&i| want[i]).map(|i| CHECKS[i]).collect();
                bad.push(format!("{c:?} in {}: fails {on:?}, declared {w:?}", T::NAME));
            }
        }
        finish(bad);
    }

    /// a control, the name of the check it is run against, and the check
    type EdgeRow<'a> = (ImplicitControl, &'a str, fn(Ctl) -> Check);

    /// Pass: each edge check is met without its control and fails with it
    fn edge_controls(rows: &[EdgeRow]) {
        let mut bad = Vec::new();
        for &(c, name, f) in rows {
            let (clean, wrong) = (f(Ctl::None), f(Ctl::Lib(c)));
            let ok = clean.pass && !wrong.pass;
            println!(
                "[{}] control {:?} against {}: met without {}, with: {}",
                if ok { "FIRED" } else { "MISMATCH" },
                c,
                name,
                clean.pass,
                wrong.detail
            );
            if !ok {
                bad.push(format!(
                    "{c:?} against {name}: met without it {}, met with it {}",
                    clean.pass, wrong.pass
                ));
            }
        }
        finish(bad);
    }

    /// the controls of the edge cases that hold in every lane
    fn edge_rows<T: TestReal>() -> Vec<EdgeRow<'static>> {
        vec![
            (ImplicitControl::StaleSeries, "stale", stale::<T>),
            (ImplicitControl::RepeatCrossing, "repeat_form", repeat_form::<T>),
            (ImplicitControl::NoEndCrossing, "last_crossing", last_crossing::<T>),
            (ImplicitControl::NoDedup, "dup_checkpoints", dup_checkpoints::<T>),
            (ImplicitControl::KeepZero, "zero_only_pivot", zero_only_pivot::<T>),
            (ImplicitControl::EmptyShift, "shift_edges", shift_edges::<T>),
        ]
    }

    #[test]
    fn edge_controls_f64() {
        let mut rows = edge_rows::<f64>();
        rows.push((ImplicitControl::PlainScale, "underflow", underflow));
        rows.push((ImplicitControl::PlainReciprocal, "reciprocal", reciprocal));
        rows.push((ImplicitControl::PlainRho, "rho_scale", rho_scale));
        edge_controls(&rows);
    }

    #[test]
    fn edge_controls_dd() {
        edge_controls(&edge_rows::<DD>());
    }

    #[test]
    fn edge_controls_qd() {
        edge_controls(&edge_rows::<QD>());
    }

    /// the gated `coeffs` fails on a recursion that skips order 3 of N
    fn skip_order_fires<T: TestReal>() {
        for l in [1.0, 0.25] {
            let c = coeffs::<T>(l, Ctl::Lib(ImplicitControl::SkipOrder(3)));
            show(&c);
            assert!(!c.pass, "control SkipOrder did not fire: {}", c.detail);
        }
    }

    #[test]
    fn coeffs_controls() {
        skip_order_fires::<f64>();
        skip_order_fires::<DD>();
        skip_order_fires::<QD>();
    }

    #[test]
    fn controls_f64() {
        controls_lane::<f64>();
    }

    #[test]
    fn controls_dd() {
        controls_lane::<DD>();
    }

    #[test]
    fn controls_qd() {
        controls_lane::<QD>();
    }
}
