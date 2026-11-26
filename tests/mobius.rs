//! A series about a singular point summed in the Moebius variable (`mobius::Mobius`) against its closed form in every
//! lane type, where the series in the affine variable diverges.
//!
//! The planted function f(w) = w^-alpha exp(beta (w - 1)) exp(-gamma (1/w - 1)), f(1) = 1, solves
//!
//! ```text
//! w^2 f' = (gamma - alpha w + beta w^2) f,
//! ```
//!
//! singular at w = 0 (an essential singularity) and at w = infinity. Its series about w = 1 in the affine variable
//! 1 - w has radius 1, so at w_l = 5/2 it diverges like 1.5^k. In tau, with a = 1, w = 0 and w = infinity sit at
//! tau = 1/t_l = -7/3 and -1/(a t_l) = 7/3, so the series converges at the node tau = 1 like (3/7)^k. Order k of the
//! equation uses all three helpers: w^2 d/dw by its polynomial in tau (`deriv_poly`, `deriv_lower`), and the
//! products w f and w^2 f by the two-term recurrence (`mul_w`). The lanes are pointwise (B = 1), alpha = 1/2 + j/8 on
//! lane j, and the closed form's derivatives in the data f(1) = A and in alpha are A f and -ln(w) f.
mod common;

use common::probe::{rel_err, Probe};
use multidouble::native::Oun;
use multidouble::od::OD;
use multishoot::lane::{Dc, Dtn, Du, Fu, Fun, Jet, Lane, OJet, Ou, QJet, Qu, DV, OV, QV};
use multishoot::{Fl, LaneConst, Mobius, DD};
use multisolver::prec::nan_max;

const BETA: f64 = 0.3;
const GAMMA: f64 = 0.2;
const A: f64 = 1.0;
const WL: f64 = 2.5;
/// the series' order: (3/7)^360 is about 1e-132
const ORDER: usize = 360;

/// alpha on lane j
fn alpha0(j: usize) -> f64 {
    0.5 + j as f64 / 8.0
}

/// a pointwise field of the lane values f(j)
fn lanes(f: impl Fn(usize) -> f64) -> Fl<DV, 1> {
    let v: Vec<DD> = (0..8).map(|j| DD::from_f64(f(j))).collect();
    let (h, l) = multidouble::dd::split(&v);
    Fl::load(&h, &l)
}

/// the series' coefficients f_0..=f_order in tau from f_0 and alpha: order k of the equation solved for f_(k+1),
/// (gamma f_k - alpha (w f)_k + beta (w^2 f)_k - [the orders k, k - 1 of w^2 f']) / (q_0 (k + 1))
fn series<T: Lane>(mb: &Mobius, alpha: &Fl<T, 1>, f0: &Fl<T, 1>, order: usize) -> Vec<Fl<T, 1>> {
    let iq = LaneConst::new(OD::ONE / mb.deriv_poly(2)[0].od());
    let (mut f, mut wf, mut wwf) = (vec![*f0], Vec::with_capacity(order), Vec::with_capacity(order));
    for k in 0..order {
        wf.push(mb.mul_w(&f, &wf, k));
        wwf.push(mb.mul_w(&wf, &wwf, k));
        let rhs = (f[k].mulf(GAMMA) - *alpha * wf[k]) + wwf[k].mulf(BETA);
        let next = LaneConst::recip(k as u64 + 1).mul(iq.mul(rhs - mb.deriv_lower(2, &f, k)));
        f.push(next);
    }
    f
}

/// the closed form's coefficient n in h of f(w) along f(1) = A (1 + sigma h), alpha = alpha_0 + tau h, on lane j:
/// F (-tau L)^n / n! + sigma F (-tau L)^(n-1) / (n-1)!, F = f(w) and L = ln w
fn closed(w: OD, j: usize, n: usize, sigma: OD, tau: OD) -> OD {
    let l = w.ln();
    let f = (-(OD::from_f64(alpha0(j)) * l)).exp()
        * (OD::from_f64(BETA) * (w - OD::ONE)).exp()
        * (-(OD::from_f64(GAMMA) * (OD::ONE / w - OD::ONE))).exp()
        * OD::from_f64(A);
    let x = -(tau * l);
    let mut pw = OD::ONE;
    let mut fact = OD::ONE;
    let mut prev = OD::from_f64(0.0);
    for i in 1..=n {
        prev = pw / fact;
        pw = pw * x;
        fact = fact * OD::from_f64(i as f64);
    }
    let cur = pw / fact;
    if n == 0 {
        f
    } else {
        f * cur + sigma * f * prev
    }
}

/// The Moebius sum of one lane type to the order against the closed form, every part (value, then the derivative
/// parts along A with seed sigma_p and alpha with seed tau_p); `mb` the variable. Returns the relative error of each
/// part.
fn mobius_errors<T: Probe>(mb: &Mobius, order: usize) -> Vec<f64> {
    let tp = T::parts();
    let seeds = |which: usize| -> Vec<Fl<DV, 1>> {
        tp.iter()
            .map(|&(n, s, t)| {
                let c = if which == 0 { s } else { t };
                if n == 1 {
                    Fl::splat(DD::from_f64(c))
                } else {
                    Fl::zero()
                }
            })
            .collect()
    };
    let f0 = T::make(&Fl::splat(DD::from_f64(A)), &seeds(0));
    let alpha = T::make(&lanes(alpha0), &seeds(1));
    let f = series(mb, &alpha, &f0, order);
    let s = Mobius::sum(&f);
    (0..=tp.len())
        .map(|p| {
            let (n, sigma, tau) = if p == 0 { (0, 0.0, 0.0) } else { tp[p - 1] };
            // a jet's coefficient p is order p along the seeds of coefficient 1
            let reference: Vec<OD> = (0..8)
                .map(|j| closed(mb.wl, j, n, OD::from_f64(sigma), OD::from_f64(tau)))
                .collect();
            rel_err(&T::get(&s, p), &reference)
        })
        .collect()
}

fn passes<T: Probe>(e: &[f64]) -> bool {
    e[0] <= T::VAL.bound() && e[1..].iter().all(|&x| x <= T::DER.bound())
}

/// (lane type, passes, errors) of `mobius_errors` on every lane type
fn all_lanes(mb: &Mobius, order: usize) -> Vec<(&'static str, bool, Vec<f64>)> {
    macro_rules! each {
        ($($t:ty),*) => {
            vec![$({
                let e = mobius_errors::<$t>(mb, order);
                (<$t as Probe>::NAME, passes::<$t>(&e), e)
            }),*]
        };
    }
    each!(
        DV,
        QV,
        OV,
        Du,
        Qu,
        Ou,
        Oun<2>,
        Dc,
        Jet<4>,
        QJet<4>,
        OJet<4>,
        Fu,
        Fun<2>,
        Dtn<2>
    )
}

fn show(label: &str, r: &[(&'static str, bool, Vec<f64>)]) {
    for (name, ok, e) in r {
        let es: Vec<String> = e.iter().map(|x| format!("{x:.1e}")).collect();
        println!(
            "{label} {name:>7}: {} [{}]",
            if *ok { "pass" } else { "FAIL" },
            es.join(", ")
        );
    }
}

fn variable() -> Mobius {
    Mobius::new(1.0, OD::from_f64(WL))
}

/// The series in tau summed at the node against the closed form in every lane type, the value and the derivative
/// parts in the data and in alpha, for three variables: a = 1 at w_l = 5/2 (order 360, terms falling like (3/7)^k),
/// the affine a = 0 at w_l = 3/5 inside its disc (order 340, like 0.4^k), and a = 8 at w_l = 1/2 (order 1500, like
/// 0.8^k, w = infinity the nearest singularity). Pass: each part within its precision's bound relative to the closed
/// form (1e-27, 1e-57, 1e-115, 1e-11), written before the first run.
#[test]
fn mobius_sum_is_its_closed_form() {
    for (a, wl, order) in [(1.0, WL, ORDER), (0.0, 0.6, 340), (8.0, 0.5, 1500)] {
        let r = all_lanes(&Mobius::new(a, OD::from_f64(wl)), order);
        show(&format!("a {a} w_l {wl}"), &r);
        assert!(r.iter().all(|x| x.1), "a {a}, w_l {wl}: a lane type fails");
    }
}

/// the largest |f_k| over the orders of the series in double-double, and the last |f_k|, in the variable
fn coefficient_sizes(mb: &Mobius) -> (f64, f64) {
    let f = series::<DV>(mb, &Fl::cst(&lanes(alpha0)), &Fl::splat(DD::from_f64(A)), ORDER);
    let big = f.iter().map(|c| c.val().amax()).fold(0.0, nan_max);
    (big, f[ORDER].val().amax())
}

/// The same series in the affine variable (a = 0, t = 1 - w): at w_l = 5/2 its coefficients grow like 1.5^k, where the
/// Moebius variable's stay at most f_0 = 1 and fall to below 1e-120 by order 360. Pass: the affine series' last
/// coefficient above 1e30, the Moebius series' largest at most 1 and its last below 1e-120.
#[test]
fn affine_series_diverges_where_mobius_converges() {
    let (mb_big, mb_last) = coefficient_sizes(&variable());
    let (af_big, af_last) = coefficient_sizes(&Mobius::new(0.0, OD::from_f64(WL)));
    println!("max |f_k|, |f_360|: Moebius {mb_big:.3e}, {mb_last:.3e}; affine {af_big:.3e}, {af_last:.3e}");
    assert!(af_last > 1e30 && mb_big <= 1.0 && mb_last < 1e-120);
}

/// mul_w's recurrence against the Cauchy product with w's coefficients (`w_coef`) on the series' coefficients, in
/// octo-double: (largest difference relative to the product's largest coefficient)
fn recurrence_vs_cauchy(mb: &Mobius) -> f64 {
    let f = series::<OV>(mb, &Fl::cst(&lanes(alpha0)), &Fl::splat(DD::from_f64(A)), 200);
    let mut wf: Vec<Fl<OV, 1>> = Vec::new();
    let mut worst = 0.0f64;
    let big = f.iter().map(|c| c.val().amax()).fold(0.0, nan_max);
    for k in 0..f.len() {
        wf.push(mb.mul_w(&f, &wf, k));
        let mut c = Fl::<OV, 1>::zero();
        for j in 0..=k {
            c = c + mb.w_coef(j).mul(f[k - j]);
        }
        worst = nan_max(worst, (wf[k] - c).val().amax() / big);
    }
    worst
}

/// The two-term recurrence of the products with w gives the Cauchy product with w's coefficients, in octo-double, to
/// order 200. Pass: within 1e-115 of the product's largest coefficient.
#[test]
fn mul_w_is_the_cauchy_product() {
    let e = recurrence_vs_cauchy(&variable());
    println!("w f by its recurrence against the Cauchy product: {e:.3e}");
    assert!(e < 1e-115, "{e:e}");
}

/// the Moebius sum of one lane type (a = 1, w_l = 5/2, order 360; alpha on lane 0 moved by `ulps` ulps) into the hash
fn mobius_hash<T: Probe>(h: &mut common::Fnv, ulps: u64) {
    let np = T::parts().len();
    let f0 = T::make(&Fl::splat(DD::from_f64(A)), &vec![Fl::splat(DD::ONE); np]);
    let al = lanes(|j| {
        if j == 0 {
            f64::from_bits(alpha0(0).to_bits() + ulps)
        } else {
            alpha0(j)
        }
    });
    let alpha = T::make(&al, &vec![Fl::splat(DD::ONE); np]);
    common::probe::hash_parts(h, &Mobius::sum(&series(&variable(), &alpha, &f0, ORDER)));
}

/// the hash of the Moebius sums over the lane types
fn mobius_hashes(ulps: u64) -> u64 {
    let mut h = common::Fnv::new();
    mobius_hash::<DV>(&mut h, ulps);
    mobius_hash::<QV>(&mut h, ulps);
    mobius_hash::<OV>(&mut h, ulps);
    mobius_hash::<Du>(&mut h, ulps);
    mobius_hash::<Qu>(&mut h, ulps);
    mobius_hash::<Ou>(&mut h, ulps);
    mobius_hash::<Dc>(&mut h, ulps);
    mobius_hash::<Jet<4>>(&mut h, ulps);
    mobius_hash::<QJet<4>>(&mut h, ulps);
    mobius_hash::<OJet<4>>(&mut h, ulps);
    mobius_hash::<Fu>(&mut h, ulps);
    mobius_hash::<Dtn<2>>(&mut h, ulps);
    h.0
}

/// the hash of `mobius_hashes(0)` on every backend
const MOBIUS_REFERENCE: u64 = 0x267c9e127aaa83f3;

/// The Moebius sums' bytes (every part in twelve lane types) do not depend on multidouble's SIMD backend. Pass: the
/// hash equals the reference, which CI checks on SSE2, AVX2, portable and NEON.
#[test]
fn mobius_bytes_match_reference() {
    let h = mobius_hashes(0);
    println!("Moebius sums hash {h:#018x}");
    assert_eq!(h, MOBIUS_REFERENCE, "the Moebius sums' bytes differ from the reference");
}

/// The control of `mobius_bytes_match_reference`: alpha one ulp up on one lane must change the hash.
#[test]
fn mobius_bytes_control_one_ulp_differs() {
    assert_ne!(mobius_hashes(1), mobius_hashes(0), "control did not fire");
}

/// The controls (feature `mutate`), each against the check it breaks.
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::Control;

    fn with(c: Control) -> Mobius {
        Mobius::with_control(1.0, OD::from_f64(WL), c)
    }

    /// a's sign flipped in w(tau) and the products fails `mobius_sum_is_its_closed_form` on every lane type
    #[test]
    fn sign_flipped_fails() {
        let r = all_lanes(&with(Control::MobiusSign), ORDER);
        show("sign", &r);
        assert!(r.iter().all(|x| !x.1), "control did not fire");
    }

    /// the recurrence's coefficient off by 2^-60 fails `mobius_sum_is_its_closed_form` on every lane type of double-
    /// double precision or more, and `mul_w_is_the_cauchy_product`
    #[test]
    fn recurrence_off_fails() {
        let mb = with(Control::MobiusRecurrence);
        let r = all_lanes(&mb, ORDER);
        show("recurrence", &r);
        for (name, ok, _) in &r {
            if !name.starts_with('F') {
                assert!(!ok, "control did not fire on {name}");
            }
        }
        let e = recurrence_vs_cauchy(&mb);
        println!("control, recurrence off: against the Cauchy product {e:.3e}");
        assert!(e >= 1e-115, "control did not fire");
    }

    /// the constants rounded to double-double fail `mobius_sum_is_its_closed_form` on the quad- and octo-double lane
    /// types
    #[test]
    fn constants_in_double_double_fail() {
        let r = all_lanes(&with(Control::MobiusConstDD), ORDER);
        show("constants dd", &r);
        for name in ["QV", "OV", "Qu", "Ou", "Oun<2>", "QJet<4>", "OJet<4>"] {
            assert!(
                !r.iter().find(|x| x.0 == name).unwrap().1,
                "control did not fire on {name}"
            );
        }
    }
}
