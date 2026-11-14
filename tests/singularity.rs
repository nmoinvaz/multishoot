//! The singularity estimators on planted series with known answers: conjugate pole pairs and square-root branch points
//! hidden below a background, series with no singularity, a geometric series and a power law. The pass criteria of
//! each test were fixed before its first run.
use multishoot::singularity::{domb_sykes, recurrence, root_radius, stable_poles, Cx, PoleOpts};
use multishoot::{Double, Native, Precision, Quad, DD};

type Qs = <Quad as Precision<Native>>::Scalar;

const N: usize = 24;

fn q(x: f64) -> Qs {
    <Quad as Precision<Native>>::from_dd(DD::from_f64(x))
}

fn inv(x: Qs) -> Qs {
    Cx::<Quad>::real(x).inv().re
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    /// a conjugate pole pair at |s| 1, arg 60 degrees, residue 1e-14
    P1,
    /// a conjugate pole pair at |s| 2.5, arg 25 degrees, residue 1e-11
    P2,
    /// square-root branch points at |s| 1.5, arg 45 degrees
    P3,
    /// the background alone
    N1,
    /// the background and an entire oscillating component 2 Re C exp((0.1 + 2.3 i) a)
    N3,
}

/// a deterministic sequence of unit variance (xorshift, uniform on [-sqrt 3, sqrt 3))
fn noise(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * 3f64.sqrt()
}

/// c_1..c_N in quad-double of the planted series, and its planted singularity (Im >= 0). The background is
/// -1.7e-4 ln(1 + a / 243) / (1 / 243), whose singularity at -243 lies far out, plus a pair at radius 7 of weight
/// 1.9e-8; each planted singularity hides below it at orders 1 to 4 and dominates from order 5 on. Every coefficient
/// carries relative noise `rel` (seeded).
fn planted(kind: Kind, seed: u64, rel: f64, drop: bool) -> (Vec<Qs>, Option<(f64, f64)>) {
    let s243 = inv(q(243.0));
    let st = Cx::<Quad>::from_f64(7.0 * 2.2f64.cos(), 7.0 * 2.2f64.sin()).inv();
    let (mut p243, mut pst) = (Qs::from(q(1.0)), Cx::<Quad>::real(q(1.0)));
    let mut c: Vec<Qs> = (1..=N)
        .map(|k| {
            p243 = p243 * s243;
            pst = pst * st;
            let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
            q(-1.7e-4 * 243.0 * sign) * p243 * inv(q(k as f64)) + q(1.9e-8 * 2.0) * pst.re
        })
        .collect();
    let mut plant = None;
    let pole = |r: f64, deg: f64| (r * deg.to_radians().cos(), r * deg.to_radians().sin());
    match kind {
        Kind::P1 | Kind::P2 => {
            let ((re, im), w) = if kind == Kind::P1 {
                (pole(1.0, 60.0), 1e-14)
            } else {
                (pole(2.5, 25.0), 1e-11)
            };
            plant = Some((re, im));
            if !drop {
                // 1 / (s - a) + conjugate: c_k = 2 Re s^-(k+1)
                let si = Cx::<Quad>::from_f64(re, im).inv();
                let mut p = si;
                for ck in c.iter_mut() {
                    p = p * si;
                    *ck = *ck + q(2.0 * w) * p.re;
                }
            }
        }
        Kind::P3 => {
            let (re, im) = pole(1.5, 45.0);
            plant = Some((re, im));
            // (1 - a / s)^(1/2) + conjugate: c_k = 2 Re binom(1/2, k) (-1 / s)^k
            let ms = Cx::<Quad>::from_f64(-re, -im).inv();
            let (mut b, mut p) = (q(1.0), Cx::<Quad>::real(q(1.0)));
            for (i, ck) in c.iter_mut().enumerate() {
                let k = (i + 1) as f64;
                b = b * q(0.5 - (k - 1.0)) * inv(q(k));
                p = p * ms;
                *ck = *ck + q(2.0 * 1e-12) * (Cx::<Quad>::real(b) * p).re;
            }
        }
        Kind::N3 => {
            let lam = Cx::<Quad>::from_f64(0.1, 2.3);
            let mut p = Cx::<Quad>::real(q(1.0));
            for (i, ck) in c.iter_mut().enumerate() {
                p = p * lam * Cx::<Quad>::real(inv(q((i + 1) as f64)));
                *ck = *ck + q(2.0 * 1e-11) * p.re;
            }
        }
        Kind::N1 => {}
    }
    let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1);
    for ck in c.iter_mut() {
        *ck = *ck * q(1.0 + rel * noise(&mut st));
    }
    (c, plant)
}

fn deg(s: (f64, f64)) -> f64 {
    s.1.atan2(s.0).to_degrees()
}

/// [the nearest stable Pade pole within tol_r in |s| and tol_deg degrees of the plant, the recurrence fit within
/// tol_r2 and tol_deg2]
fn judge(c: &[Qs], plant: (f64, f64), tol: (f64, f64, f64, f64)) -> [bool; 2] {
    let pr = plant.0.hypot(plant.1);
    let sp = stable_poles::<Quad>(c, &PoleOpts::default());
    let c64: Vec<f64> = c.iter().map(|&v| <Quad as Precision<Native>>::to_dd(v).hi).collect();
    let rec = recurrence(&c64, 5, 6).s;
    let pade = sp.first().map(|p| {
        println!(
            "    Pade |s| {:.4} arg {:.2} (M {:?}, spread {:.1e})",
            p.abs,
            deg(p.s),
            p.ms,
            p.spread
        );
        (p.abs / pr - 1.0).abs() <= tol.0 && (deg(p.s) - deg(plant)).abs() <= tol.1
    });
    let rfit = rec.map(|s| {
        println!("    recurrence |s| {:.4} arg {:.2}", s.0.hypot(s.1), deg(s));
        (s.0.hypot(s.1) / pr - 1.0).abs() <= tol.2 && (deg(s) - deg(plant)).abs() <= tol.3
    });
    [pade.unwrap_or(false), rfit.unwrap_or(false)]
}

/// Pass, written before the first run: with relative noise 1e-6, in quad-double, the nearest stable Pade pole recovers
/// P1 and P2 to 3% in |s| and 3 degrees, and the recurrence fit to 5% and 5 degrees; P3's branch points to 10% and 10
/// degrees by both. The control: P1 built without its pole fails both.
#[test]
fn poles_of_planted_pairs_and_branch_points() {
    for (kind, tol) in [
        (Kind::P1, (0.03, 3.0, 0.05, 5.0)),
        (Kind::P2, (0.03, 3.0, 0.05, 5.0)),
        (Kind::P3, (0.10, 10.0, 0.10, 10.0)),
    ] {
        let (c, plant) = planted(kind, 1, 1e-6, false);
        let plant = plant.unwrap();
        println!(
            "{kind:?}: planted |s| {:.4} arg {:.2}",
            plant.0.hypot(plant.1),
            deg(plant)
        );
        assert_eq!(judge(&c, plant, tol), [true; 2], "{kind:?}");
    }
    let (c, plant) = planted(Kind::P1, 1, 1e-6, true);
    println!("control, P1 without its pole:");
    assert_eq!(
        judge(&c, plant.unwrap(), (0.03, 3.0, 0.05, 5.0)),
        [false; 2],
        "control did not fire"
    );
}

/// the stable Pade poles within |s| <= 4
fn near_poles(c: &[Qs]) -> Vec<(f64, f64)> {
    stable_poles::<Quad>(c, &PoleOpts::default())
        .into_iter()
        .filter(|p| p.abs <= 4.0)
        .map(|p| (p.abs, deg(p.s)))
        .collect()
}

/// Pass, written before the first run: the background alone (N1) and with an entire oscillating component (N3), each
/// with relative noise 1e-6 at seeds 1 to 5, give no stable Pade pole within |s| <= 4. The control: P1 gives one.
#[test]
fn no_stable_pole_without_a_singularity() {
    for kind in [Kind::N1, Kind::N3] {
        for seed in 1..=5 {
            let (c, _) = planted(kind, seed, 1e-6, false);
            let near = near_poles(&c);
            println!("{kind:?} seed {seed}: stable poles within 4: {near:?}");
            assert!(near.is_empty(), "{kind:?} seed {seed}");
        }
    }
    let (c, _) = planted(Kind::P1, 1, 1e-6, false);
    assert!(!near_poles(&c).is_empty(), "control did not fire");
}

/// Pass, written before the first run: the root test's radius of c_k = 1.7^-k is 1.7 to 1e-12 (control: the series
/// of 2^-k misses 1.7 by more than 0.1); the ratio method on (1 - a / 2)^(-1/2) gives s = 2 to 1% and gamma = 1/2 to
/// 0.05 (control: (1 - a / 2)^(-3/2) misses gamma = 1/2 by more than 0.5); and the stable Pade pole of P1 in
/// double-double is reported, not judged.
#[test]
fn root_radius_and_ratio_method() {
    let geo = |r: f64| (1..=N).map(|k| r.powi(-(k as i32))).collect::<Vec<f64>>();
    let rr = root_radius(&geo(1.7), None);
    let rc = root_radius(&geo(2.0), None);
    println!("root radius: {rr:.15} (control {rc:.3})");
    assert!((rr - 1.7).abs() <= 1e-12);
    assert!((rc - 1.7).abs() > 0.1, "control did not fire");
    // (1 - a / s)^(-gamma): c_k = (gamma)_k / k! s^-k
    let power = |gamma: f64| {
        let mut c = 1.0;
        (1..=N)
            .map(|k| {
                c *= (gamma + k as f64 - 1.0) / k as f64 / 2.0;
                c
            })
            .collect::<Vec<f64>>()
    };
    let (s, g) = domb_sykes(&power(0.5)).unwrap();
    let (_, gc) = domb_sykes(&power(1.5)).unwrap();
    println!("Domb-Sykes: s {s:.5}, gamma {g:.4} (control gamma {gc:.4})");
    assert!((s / 2.0 - 1.0).abs() <= 0.01 && (g - 0.5).abs() <= 0.05);
    assert!((gc - 0.5).abs() > 0.5, "control did not fire");
    let (c, _) = planted(Kind::P1, 1, 1e-6, false);
    let cd: Vec<DD> = c.iter().map(|&v| <Quad as Precision<Native>>::to_dd(v)).collect();
    let sp = stable_poles::<Double>(&cd, &PoleOpts::default());
    println!(
        "P1 in double-double (reported): {:?}",
        sp.first().map(|p| (p.abs, deg(p.s), p.ms.clone()))
    );
}
