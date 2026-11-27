//! The plane stepper over a complex scalar (`multishoot::plane::Complex`) on a planted system with a known solution, in f64,
//! double-double and quad-double. The reference is a closed form that shares no code with the stepper.
//!
//! Taylor tolerance 100 eps_lane, order `Plan::optimal_kt`, jets to the same order, `hmax` 1, from x = 0. Pass criteria, written
//! before the first run, with eps_lane the lane's unit roundoff (`Real::EPS`):
//!
//! P1c: P1 of `tests/plane.rs` with a rotating phase. J = P diag(24 + 7i, 8 - 3i, -8 + 5i, -24 - 11i) P^-1 (Gaussian
//! integer entries; P = L U unimodular, cond 96), k = 2, the plane P[:, :2] (not orthonormal), to x = 12. The plane stays
//! span P[:, :2]; Y(x) = P[:, :2] diag(e^(lambda_1 x), e^(lambda_2 x)), whose orthonormal basis with R positive is
//! Q(x) = Q0 diag(e^(7 i x), e^(-3 i x)) with Q0 the one of P[:, :2] at x = 0, and ln_vol = 32 x + (1/2) ln 5. The phase stays in
//! Q (R has a real positive diagonal), so the complex log-volume of the plane is Lambda = ln_vol + ln det(Q0^T Q) =
//! (32 + 4 i) x + (1/2) ln 5, a phase of 48 rad at x = 12 (84 rad and -36 rad on the columns).
//!
//! * the plane: ||(I - Pi) Y||_F at most 1e4 eps, Pi the real projector onto the exact plane formed in the lane from the exact
//!   basis, applied to the real and imaginary parts of Y;
//! * the gauge: max |Q - Q_exact| at most 1e4 eps (the columns with their phases, Q_exact = Q0 diag(e^(7 i x), e^(-3 i x)));
//! * the log-volume: |ln_vol - exact| at most 1e4 eps max(1, |exact|), its imaginary part zero;
//! * the phase of the volume: |det(Q0^T Q) - e^(48 i)| at most 1e4 eps (Lambda's imaginary part);
//! * the returned columns orthonormal in the Hermitian product: max |Q^H Q - I| at most 1e2 eps;
//! * real data: the same system with f = 0 on `Complex<R>` gives the digest (every part of the columns and of the
//!   log-volume) of the real `Plane<R>`, with every imaginary part zero.
//!
//! The controls (each fails its check; the library's under the feature `mutate`, the wrong inputs always): `NoConj` (the
//! orthonormalization without the conjugate: it keeps plane and log-volume's real part and loses the phase: gauge and phase fail),
//! `f_scaled` (f_2 scaled by 1 + 1e-6 in the system, not in the reference), `ref_no_phase` (the reference without the phases
//! e^(i f x)), `wrong_plane` (the plane compared with span P[:, 1:3]).
use multishoot::multidouble::dd::DD;
use multishoot::multidouble::qd::QD;
use multishoot::plane::{Complex, JetSource, Jets, Plane, Real};
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

/// (1/2) ln 5, and cos and sin of 84, -36 and 48 to 140 digits (mpmath)
const HALF_LN5: &str = "0.80471895621705018730037966661309381976280067713425886095632394573708949385382888231506693904658980539998315151085778144986200261466233809982";
const COS84: &str = "-0.68002349558733879542720377532040326778164799412571980676264605772854880166725271039500102689300364475124824204634245575894419681617774962105";
const SIN84: &str = "0.73319032007329216636320780871381280204518512546944786510004230734329688609986905753505489934620906874163622849282196330972683644911838578007";
const COS36: &str = "-0.12796368962740468102832797791437121294042538189146929439527991586488979705497812165107701872297170352802088045880737850262845741706779732915";
// sin(-36)
const SINM36: &str = "0.99177885344311573683528896884081577859591980659077822976320143546279363096769486966381659828832478295465196892659920831031551125594460908894";
const COS48: &str = "-0.64014433946919973131294141939307717951754277405406299215589874781751461348123400975315743571060292120540825716279420500564471746665114277163";
const SIN48: &str = "-0.76825466132366679904496566484589349479809267158803529602008513420815383686676365514659448814092604132906674919422045680381138265907508556906";

const N: usize = 4;
type Mat = [[f64; N]; N];
const P: Mat = [[1., 1., 0., 1.], [1., 2., 1., 1.], [0., 1., 2., 1.], [1., 1., 1., 3.]];
const PINV: Mat = [
    [7., -4., 3., -2.],
    [-4., 3., -2., 1.],
    [3., -2., 2., -1.],
    [-2., 1., -1., 1.],
];
const G: [f64; N] = [24., 8., -8., -24.];
const F: [f64; N] = [7., -3., 5., -11.];
const X: f64 = 12.0;

/// J = P diag(g + i f) P^-1 as (re, im) pairs of integers
fn j_entries(f: &[f64; N]) -> [[(f64, f64); N]; N] {
    let mut j = [[(0.0, 0.0); N]; N];
    for (i, row) in j.iter_mut().enumerate() {
        for (jj, e) in row.iter_mut().enumerate() {
            for k in 0..N {
                e.0 += P[i][k] * G[k] * PINV[k][jj];
                e.1 += P[i][k] * f[k] * PINV[k][jj];
            }
        }
    }
    j
}

/// constant J on a scalar `T` made from (re, im)
struct Const<T: Real> {
    j: [[(f64, f64); N]; N],
    mk: fn(f64, f64) -> T,
}

impl<T: Real> JetSource<T> for Const<T> {
    fn pattern(&self) -> Vec<(usize, usize)> {
        (0..N).flat_map(|i| (0..N).map(move |j| (i, j))).collect()
    }
    fn jets(&self, _x0: f64, _kj: usize, out: &mut Jets<T>) {
        for en in out.ent.iter_mut() {
            en.c[0] = (self.mk)(self.j[en.i][en.j].0, self.j[en.i][en.j].1);
            en.n = 1;
        }
    }
}

fn plan_for<T: Real>() -> (Plan, usize) {
    let tol = 100.0 * T::EPS;
    let kt = Plan::optimal_kt(tol);
    (Plan { kt, tol, hmax: 1.0 }, kt)
}

fn mk_c<T: Real>(re: f64, im: f64) -> Complex<T> {
    Complex::new(T::from_f64(re), T::from_f64(im))
}

fn mk_r<T: Real>(re: f64, im: f64) -> T {
    assert!(im == 0.0, "a real lane needs f = 0");
    T::from_f64(re)
}

fn basis<T: Real>(cols: [usize; 2], mk: fn(f64) -> T) -> Vec<T> {
    cols.iter()
        .flat_map(|&c| (0..N).map(move |i| (c, i)))
        .map(|(c, i)| mk(P[i][c]))
        .collect()
}

/// a digest of every part of the columns and of the log-volume, for the comparison across backends and with the real plane
fn digest(parts: impl Iterator<Item = f64>) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in parts {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// What a case reports
#[derive(Clone, Copy)]
struct Out {
    plane: f64,
    gauge: f64,
    ln_err: f64,
    ln_bar: f64,
    ln_im: f64,
    phase: f64,
    orth: f64,
    steps: usize,
    digest: u64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Wrong {
    None,
    FScaled,
    RefNoPhase,
    RefNoGram,
    OtherPlane,
}

/// textbook Gram-Schmidt of the real columns P[:, a], P[:, b] in the lane, twice, positive diagonal: Q0
fn q0<T: Lane>(cols: [usize; 2]) -> [[T; N]; 2] {
    let mut q: [[T; N]; 2] = [[T::zero(); N]; 2];
    for (c, &pc) in cols.iter().enumerate() {
        let mut v: [T; N] = std::array::from_fn(|i| T::from_f64(P[i][pc]));
        for _ in 0..2 {
            for qp in q.iter().take(c) {
                let d = (0..N).fold(T::zero(), |s, i| s + qp[i] * v[i]);
                for (vi, &qi) in v.iter_mut().zip(qp.iter()) {
                    *vi = *vi - d * qi;
                }
            }
        }
        let nrm = v.iter().fold(T::zero(), |s, &x| s + x * x).sqrt();
        let inv = nrm.recip();
        for (qi, &vi) in q[c].iter_mut().zip(v.iter()) {
            *qi = vi * inv;
        }
    }
    q
}

/// the plane's complex columns as (re, im) lane vectors
fn columns_c<T: Lane>(pl: &Plane<Complex<T>>) -> Vec<([T; N], [T; N])> {
    (0..pl.k())
        .map(|c| {
            let col = pl.column(c);
            (std::array::from_fn(|i| col[i].re), std::array::from_fn(|i| col[i].im))
        })
        .collect()
}

/// P1c on `Complex<T>`
#[allow(unused_variables)]
fn p1c<T: Lane>(wrong: Wrong, nconj: bool, f: [f64; N]) -> Out {
    let mut f_run = f;
    if wrong == Wrong::FScaled {
        f_run[1] *= 1.0 + 1e-6;
    }
    let src = Const::<Complex<T>> {
        j: j_entries(&f_run),
        mk: mk_c::<T>,
    };
    let (a, b) = if wrong == Wrong::OtherPlane { (1, 2) } else { (0, 1) };
    let mut pl = Plane::<Complex<T>>::new(N, 2, &basis([0, 1], Complex::from_f64));
    #[cfg(feature = "mutate")]
    if nconj {
        pl.set_control(multishoot::control::PlaneControl::NoConj);
    }
    let (plan, kj) = plan_for::<T>();
    let st = pl.advance(&src, 0.0, X, &plan, kj);
    let y = columns_c::<T>(&pl);
    let q0 = q0::<T>([a, b]);
    // the real projector onto the exact plane: Pi = B (B^T B)^-1 B^T from the exact (not orthonormal) basis, in the lane
    let cols: Vec<[T; N]> = [a, b]
        .iter()
        .map(|&c| std::array::from_fn(|i| T::from_f64(P[i][c])))
        .collect();
    let g = |p: usize, q: usize| (0..N).fold(T::zero(), |s, i| s + cols[p][i] * cols[q][i]);
    let (g00, g01, g11) = (g(0, 0), g(0, 1), g(1, 1));
    let det = (g00 * g11 - g01 * g01).recip();
    let mut plane = 0.0f64;
    for (re, im) in &y {
        for part in [re, im] {
            let t: Vec<T> = (0..2)
                .map(|p| (0..N).fold(T::zero(), |s, i| s + cols[p][i] * part[i]))
                .collect();
            let w = [(g11 * t[0] - g01 * t[1]) * det, (g00 * t[1] - g01 * t[0]) * det];
            for i in 0..N {
                let r = part[i] - (cols[0][i] * w[0] + cols[1][i] * w[1]);
                plane += r.to_f64() * r.to_f64();
            }
        }
    }
    let plane = plane.sqrt();
    // the exact gauge: column j of Q0 times e^(i f_j x), the phasors from mpmath (f = 7, -3 at x = 12)
    let ph: [(T, T); 2] = if wrong == Wrong::RefNoPhase || f == [0.0; N] {
        [(T::one(), T::zero()), (T::one(), T::zero())]
    } else {
        [(T::parse(COS84), T::parse(SIN84)), (T::parse(COS36), T::parse(SINM36))]
    };
    let mut gauge = 0.0f64;
    for c in 0..2 {
        for (i, &q) in q0[c].iter().enumerate() {
            let (er, ei) = (q * ph[c].0, q * ph[c].1);
            gauge = gauge
                .max((y[c].0[i] - er).to_f64().abs())
                .max((y[c].1[i] - ei).to_f64().abs());
        }
    }
    // the phase of the volume: det(Q0^T Q) against e^(48 i)
    let m = |p: usize, q: usize| {
        let re = (0..N).fold(T::zero(), |s, i| s + q0[p][i] * y[q].0[i]);
        let im = (0..N).fold(T::zero(), |s, i| s + q0[p][i] * y[q].1[i]);
        Complex::new(re, im)
    };
    let dq = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    let (er, ei) = if f == [0.0; N] || wrong == Wrong::RefNoPhase {
        (T::one(), T::zero())
    } else {
        (T::parse(COS48), T::parse(SIN48))
    };
    let phase = ((dq.re - er).to_f64()).hypot((dq.im - ei).to_f64());
    // orthonormality in the Hermitian product
    let mut orth = 0.0f64;
    for a_ in 0..2 {
        for b_ in 0..2 {
            let mut re = T::zero();
            let mut im = T::zero();
            for i in 0..N {
                let (ar, ai, br, bi) = (y[a_].0[i], y[a_].1[i], y[b_].0[i], y[b_].1[i]);
                re = re + ar * br + ai * bi;
                im = im + ar * bi - ai * br;
            }
            if a_ == b_ {
                re = re - T::one();
            }
            orth = orth.max(re.to_f64().abs()).max(im.to_f64().abs());
        }
    }
    let lnv = pl.ln_vol();
    let gram = if wrong == Wrong::RefNoGram {
        T::zero()
    } else {
        T::parse(HALF_LN5)
    };
    let exact = T::from_f64(32.0 * X) + gram;
    let ln_err = (lnv.re - exact).to_f64().abs();
    let ln_im = lnv.im.to_f64().abs();
    let parts = pl
        .columns()
        .iter()
        .flat_map(|z| z.re.parts().into_iter().chain(z.im.parts()))
        .chain(lnv.re.parts())
        .chain(lnv.im.parts());
    Out {
        plane,
        gauge,
        ln_err,
        ln_bar: 1e4 * T::EPS * exact.to_f64().abs().max(1.0),
        ln_im,
        phase,
        orth,
        steps: st.steps,
        digest: digest(parts),
    }
}

/// the same system with f = 0 on the real lane `T`: a digest comparable to `p1c(f = 0)`'s, and its ln_vol
fn p1_real<T: Lane>() -> (u64, usize) {
    let src = Const::<T> {
        j: j_entries(&[0.0; N]),
        mk: mk_r::<T>,
    };
    let mut pl = Plane::<T>::new(N, 2, &basis([0, 1], T::from_f64));
    let (plan, kj) = plan_for::<T>();
    let st = pl.advance(&src, 0.0, X, &plan, kj);
    // every real part, every imaginary part zero (as the complex plane lists them), then ln_vol and its zero
    let parts: Vec<f64> = pl
        .columns()
        .iter()
        .flat_map(|z| z.parts().into_iter().chain(T::zero().parts()))
        .chain(pl.ln_vol().parts())
        .chain(T::zero().parts())
        .collect();
    (digest(parts.into_iter()), st.steps)
}

/// x <= bar, false for a NaN
fn within(x: f64, bar: f64) -> bool {
    x <= bar
}

/// the checks of a case that fail: each acceptance predicate by name (shared by `pass` and the controls)
fn verdict<T: Lane>(o: &Out) -> Vec<&'static str> {
    let eps = T::EPS;
    let mut bad = vec![];
    // a NaN fails every predicate (the comparisons are written so that it does)
    if !within(o.plane, 1e4 * eps) {
        bad.push("plane");
    }
    if !within(o.gauge, 1e4 * eps) {
        bad.push("gauge");
    }
    if !within(o.ln_err, o.ln_bar) {
        bad.push("ln_err");
    }
    if o.ln_im != 0.0 {
        bad.push("ln_im");
    }
    if !within(o.phase, 1e4 * eps) {
        bad.push("phase");
    }
    if !within(o.orth, 1e2 * eps) {
        bad.push("orth");
    }
    bad
}

fn pass<T: Lane>(name: &str, o: &Out) {
    println!(
        "{name} {}: steps {} plane {:.2e} gauge {:.2e} ln_err {:.2e} (bar {:.2e}) ln_im {:.1e} phase {:.2e} orth {:.2e} digest {:016x}",
        T::NAME,
        o.steps,
        o.plane,
        o.gauge,
        o.ln_err,
        o.ln_bar,
        o.ln_im,
        o.phase,
        o.orth,
        o.digest
    );
    let bad = verdict::<T>(o);
    assert!(bad.is_empty(), "{name} {}: failed {bad:?}", T::NAME);
}

fn all_pass<T: Lane>() {
    pass::<T>("P1c", &p1c::<T>(Wrong::None, false, F));
}

#[test]
fn p1c_f64() {
    all_pass::<f64>();
}

#[test]
fn p1c_dd() {
    all_pass::<DD>();
}

#[test]
fn p1c_qd() {
    all_pass::<QD>();
}

/// real data on a complex lane give the real plane's digest, every imaginary part zero
fn real_embedding<T: Lane>() {
    let o = p1c::<T>(Wrong::None, false, [0.0; N]);
    let (d, steps) = p1_real::<T>();
    println!(
        "P1c f = 0 {}: complex digest {:016x}, real digest {:016x}, steps {} and {}",
        T::NAME,
        o.digest,
        d,
        o.steps,
        steps
    );
    assert_eq!(o.steps, steps);
    assert_eq!(
        o.digest, d,
        "the complex lane with real data is not the real lane bit for bit"
    );
    pass::<T>("P1c(f = 0)", &o);
}

#[test]
fn real_embedding_f64() {
    real_embedding::<f64>();
}

#[test]
fn real_embedding_dd() {
    real_embedding::<DD>();
}

#[test]
fn real_embedding_qd() {
    real_embedding::<QD>();
}

/// the wrong inputs, each failing the check it targets in every lane (no feature needed), and only the checks it should
fn wrong_inputs<T: Lane>() {
    let show = |name: &str, o: &Out, bad: &[&str]| {
        println!(
            "control {name} {}: failed {bad:?}; plane {:.2e} gauge {:.2e} ln_err {:.2e} phase {:.2e} orth {:.2e}",
            T::NAME,
            o.plane,
            o.gauge,
            o.ln_err,
            o.phase,
            o.orth
        )
    };
    // f_2 scaled by 1 + 1e-6 in the system, not in the reference: the phase of the volume and the gauge (the scaled entries of J are rounded
    // to f64, so in the wide lanes the plane and the log-volume move at 1e-16 too)
    let o = p1c::<T>(Wrong::FScaled, false, F);
    let bad = verdict::<T>(&o);
    show("f_scaled", &o, &bad);
    assert!(bad.contains(&"gauge") && bad.contains(&"phase"), "f_scaled: {bad:?}");
    assert!(!bad.contains(&"orth"));
    // the reference without the phases: exactly the gauge and the phase of the volume
    let o = p1c::<T>(Wrong::RefNoPhase, false, F);
    let bad = verdict::<T>(&o);
    show("ref_no_phase", &o, &bad);
    assert_eq!(bad, vec!["gauge", "phase"]);
    // the reference log-volume without the starting basis' term: exactly the log-volume
    let o = p1c::<T>(Wrong::RefNoGram, false, F);
    let bad = verdict::<T>(&o);
    show("ref_no_gram", &o, &bad);
    assert_eq!(bad, vec!["ln_err"]);
    // the other plane: the plane check
    let o = p1c::<T>(Wrong::OtherPlane, false, F);
    let bad = verdict::<T>(&o);
    show("wrong_plane", &o, &bad);
    assert!(bad.contains(&"plane"), "wrong_plane: {bad:?}");
}

/// a way of spoiling one metric of a case, given the lane's epsilon
type Wrong1 = fn(&mut Out, f64);

/// each reported metric fed a wrong value in turn, from a case that passes: the predicate of that metric, and only it, rejects
fn one_metric_at_a_time<T: Lane>() {
    let base = p1c::<T>(Wrong::None, false, F);
    assert!(verdict::<T>(&base).is_empty());
    let eps = T::EPS;
    let cases: [(&str, Wrong1); 6] = [
        ("plane", |o, e| o.plane = 2e4 * e),
        ("gauge", |o, e| o.gauge = 2e4 * e),
        ("ln_err", |o, _| o.ln_err = 2.0 * o.ln_bar),
        ("ln_im", |o, e| o.ln_im = e),
        ("phase", |o, e| o.phase = 2e4 * e),
        ("orth", |o, e| o.orth = 2e2 * e),
    ];
    for (name, wrong) in cases {
        let mut o = base;
        wrong(&mut o, eps);
        assert_eq!(verdict::<T>(&o), vec![name], "metric {name}");
    }
    let mut o = base;
    o.plane = f64::NAN;
    assert_eq!(verdict::<T>(&o), vec!["plane"], "a NaN plane error passed");
}

#[test]
fn metrics_one_at_a_time() {
    one_metric_at_a_time::<f64>();
    one_metric_at_a_time::<DD>();
    one_metric_at_a_time::<QD>();
}

#[test]
fn wrong_inputs_f64() {
    wrong_inputs::<f64>();
}

#[test]
fn wrong_inputs_dd() {
    wrong_inputs::<DD>();
}

#[test]
fn wrong_inputs_qd() {
    wrong_inputs::<QD>();
}

/// the library control `NoConj`: the plane, the log-volume and the orthonormality stay, the phase goes: exactly the gauge and the phase of the volume fail
#[cfg(feature = "mutate")]
fn no_conj<T: Lane>() {
    let o = p1c::<T>(Wrong::None, true, F);
    let bad = verdict::<T>(&o);
    println!(
        "control NoConj {}: failed {bad:?}; plane {:.2e} gauge {:.2e} ln_err {:.2e} phase {:.2e} orth {:.2e}",
        T::NAME,
        o.plane,
        o.gauge,
        o.ln_err,
        o.phase,
        o.orth
    );
    assert_eq!(bad, vec!["gauge", "phase"], "NoConj");
}

#[cfg(feature = "mutate")]
#[test]
fn no_conj_f64() {
    no_conj::<f64>();
}

#[cfg(feature = "mutate")]
#[test]
fn no_conj_dd() {
    no_conj::<DD>();
}

#[cfg(feature = "mutate")]
#[test]
fn no_conj_qd() {
    no_conj::<QD>();
}

// the scalar itself
type C = Complex<f64>;

#[test]
fn complex_real_embedding_is_exact() {
    for &a in &[0.3f64, -2.75, 1e-7, 123456.789] {
        for &c in &[0.1f64, -3.5, 7.25e5] {
            let (x, y) = (C::from_f64(a), C::from_f64(c));
            assert_eq!((x * y).re.to_bits(), (a * c).to_bits());
            assert_eq!((x / y).re.to_bits(), (a / c).to_bits());
            assert_eq!(x.recip().re.to_bits(), (1.0 / a).to_bits());
            assert!((x * y).im == 0.0 && (x / y).im == 0.0);
        }
        if a > 0.0 {
            assert_eq!(C::from_f64(a).sqrt().re.to_bits(), a.sqrt().to_bits());
            assert_eq!(C::from_f64(a).ln().re.to_bits(), a.ln().to_bits());
            assert_eq!(C::from_f64(a).abs2().re.to_bits(), (a * a).to_bits());
        }
    }
}

#[test]
fn complex_sqrt_recip_div() {
    for &(a, b) in &[
        (3.0f64, 4.0),
        (-3.0, 4.0),
        (-3.0, -4.0),
        (3.0, -4.0),
        (0.0, 2.0),
        (-5.0, 0.0),
        (1e-3, 7.0),
    ] {
        let z = C::new(a, b);
        let r = z.sqrt();
        let back = r * r;
        let tol = 1e-14 * (a.abs() + b.abs() + 1.0);
        assert!((back.re - a).abs() < tol && (back.im - b).abs() < tol, "sqrt({a}, {b})");
        assert!(r.re >= 0.0);
        let one = z / z;
        assert!((one.re - 1.0).abs() < 1e-15 && one.im.abs() < 1e-15);
        let w = z * z.recip();
        assert!((w.re - 1.0).abs() < 1e-15 && w.im.abs() < 1e-15);
    }
    // components whose squares leave f64's range: quotient, reciprocal and square root stay finite and right
    for &big in &[1e200f64, 1e-200] {
        let z = C::new(big, big);
        let one = z / z;
        assert!(
            (one.re - 1.0).abs() < 1e-15 && one.im.abs() < 1e-15,
            "z / z at {big:e}: {one:?}"
        );
        let w = z * z.recip();
        assert!(
            (w.re - 1.0).abs() < 1e-15 && w.im.abs() < 1e-15,
            "z * 1/z at {big:e}: {w:?}"
        );
        let i = C::new(big, big) / C::new(big, -big);
        assert!(
            i.re.abs() < 1e-15 && (i.im - 1.0).abs() < 1e-15,
            "(1 + i) / (1 - i) at {big:e}: {i:?}"
        );
        let r = z.sqrt();
        assert!(
            r.re.is_finite() && r.im.is_finite() && r.re > 0.0,
            "sqrt at {big:e}: {r:?}"
        );
        let back = r * r;
        assert!(
            ((back.re - big) / big).abs() < 1e-14 && ((back.im - big) / big).abs() < 1e-14,
            "sqrt squared at {big:e}: {back:?}"
        );
    }
    let z = C::new(3.0, -4.0);
    assert_eq!((z.conj().re, z.conj().im), (3.0, 4.0));
    assert_eq!((z.abs2().re, z.abs2().im), (25.0, 0.0));
    assert_eq!((z.mag(), z.re_f64()), (5.0, 3.0));
}

#[test]
#[should_panic(expected = "nonzero imaginary part")]
fn complex_to_f64_of_a_complex_panics() {
    let _ = C::new(1.0, 2.0).to_f64();
}

#[test]
#[should_panic(expected = "positive real")]
fn complex_ln_of_a_complex_panics() {
    let _ = C::new(1.0, 2.0).ln();
}
