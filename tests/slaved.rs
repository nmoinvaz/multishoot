//! A series at a regular singular point with coefficients periodic in theta, against its closed form in every lane
//! type, and as the left end of a boundary-value problem solved by Newton's method.
//!
//! The model: fields u (slaved at x = 0) and v (its data V = v(0)) on the even half grid,
//!
//! ```text
//! x u_x + omega u_theta + kappa u = v,      kappa = c0 + omega (1 - (D E) / E),
//! v_x = alpha u + beta v,
//! ```
//!
//! E = 2 + cos 2 theta and D the grid's d/dtheta, so that (omega D + kappa) E = (c0 + omega) E on the grid for every
//! omega, to the lane's rounding: E is an exact eigenvector of the discrete operator, and the coefficient kappa
//! depends on theta. With V = E the solution is u = E g(x), v = E h(x), where
//!
//! ```text
//! (k + c0 + omega) g_k = h_k,    k h_k = alpha g_(k-1) + beta h_(k-1),    h_0 = 1:
//! ```
//!
//! order k of u is the periodic system (k + omega D + kappa) u_k = v_k, one solve per order with the factors of that
//! order, and g, h are entire. The closed form's derivatives in V and omega follow from the same recursion on
//! series in h.
mod common;

use common::probe::{rel_err, Probe};
use common::{field, perturbed};
use multidouble::native::Oun;
use multidouble::od::OD;
use multishoot::lane::{DMat, Dc, Dtn, Du, Fu, Fun, Jet, Lane, OJet, Ou, QJet, Qu, DV, OV, QV};
use multishoot::periodic::{dmat, sin_cos_grid, Parity};
use multishoot::series::{series_lus, PeriodicLu, Refine};
use multishoot::{horner, Double, Equations, ExtraRow, Fl, LaneConst, Plan, Quad, Rate, Shooting, State, Work, DD};
use multisolver::prec::{nan_max, parts, widen};
use multisolver::NewtonOpts;

const B: usize = 2;
const M: usize = 8 * B;
const C0: f64 = 0.75;
const ALPHA: f64 = 0.6;
const BETA: f64 = -0.4;
/// the series' order
const ORDER: usize = 70;

/// E = 2 + cos 2 theta on the even grid
fn e_field() -> Fl<DV, B> {
    field::<B>(|j| DD::from_f64(2.0) + sin_cos_grid(Parity::Even, M, j, 2).1)
}

/// kappa = c0 + omega (1 - (D E) / E) in the lane
fn kappa<T: Lane>(d: &DMat<B>, e: &Fl<T, B>, omega: T) -> Fl<T, B> {
    let one = Fl::<T, B>::splat(DD::ONE);
    (one - d.apply(e) / *e).mul_t(omega).adds(DD::from_f64(C0))
}

/// the factors of every order, L_k = omega D + diag(kappa + k), at the double-double omega; `shift` takes order k's
/// factors from order k - shift (the control of the wrong order)
fn factors(d: &DMat<B>, omega: DD, shift: usize) -> Vec<PeriodicLu> {
    let e = Fl::<DV, B>::cst(&e_field());
    let kv = kappa::<DV>(d, &e, DV::splat(omega));
    series_lus(
        d,
        omega,
        ORDER,
        |k| {
            (0..M)
                .map(|i| kv.get(i) + DD::from_f64(k.saturating_sub(shift) as f64))
                .collect()
        },
        Rate::ZERO,
        true,
    )
    .unwrap()
}

/// the series' orders (u_k, v_k), k = 0..=ORDER, in the lane from v_0 and omega: v_k by the recursion, then u_k by
/// order k's periodic solve
fn series<T: Lane>(lus: &[PeriodicLu], d: &DMat<B>, omega: T, v0: &Fl<T, B>) -> (Vec<Fl<T, B>>, Vec<Fl<T, B>>) {
    let e = Fl::<T, B>::cst(&e_field());
    let kap = kappa(d, &e, omega);
    let (mut u, mut v): (Vec<Fl<T, B>>, _) = (Vec::with_capacity(ORDER + 1), vec![*v0]);
    for (k, lu) in lus.iter().enumerate() {
        if k > 0 {
            let vk = LaneConst::recip(k as u64).mul(u[k - 1].mulf(ALPHA) + v[k - 1].mulf(BETA));
            v.push(vk);
        }
        u.push(lu.solve(d, omega, &kap.adds(DD::from_f64(k as f64)), &v[k]));
    }
    (u, v)
}

/// a truncated series in h with four octo-double coefficients
type S4 = [OD; 4];

/// the closed form's (g_k, h_k), k = 0..=order, as series in h along V = E (1 + sigma h), omega = omega_0 + tau h
fn closed(omega0: OD, sigma: OD, tau: OD, order: usize) -> Vec<(S4, S4)> {
    let z = OD::from_f64(0.0);
    let mut out: Vec<(S4, S4)> = Vec::with_capacity(order + 1);
    let mut h: S4 = [OD::ONE, sigma, z, z];
    for k in 0..=order {
        if k > 0 {
            let (gp, hp) = out[k - 1];
            let ik = OD::ONE / OD::from_f64(k as f64);
            h = std::array::from_fn(|n| (OD::from_f64(ALPHA) * gp[n] + OD::from_f64(BETA) * hp[n]) * ik);
        }
        // g = h / (k + c0 + omega0 + tau h)
        let b0 = OD::from_f64(k as f64 + C0) + omega0;
        let mut g = [z; 4];
        for n in 0..4 {
            let prev = if n > 0 { tau * g[n - 1] } else { z };
            g[n] = (h[n] - prev) / b0;
        }
        out.push((g, h));
    }
    out
}

/// The series of one lane type against the closed form: V = E with seed sigma_p E on each part p, omega = 1.3 / 3
/// formed in the lane with seed tau_p, the factors at the double-double omega. Returns the largest relative error over
/// the orders of each part of u_k and v_k (value, then the derivative parts), and of the sums at x = 1/2.
fn slaved_errors<T: Probe>(refine: Refine, shift: usize, setup: &dyn Fn(&mut PeriodicLu)) -> Vec<f64> {
    let d = dmat::<B>(Parity::Even);
    let e = e_field();
    let tp = T::parts();
    let seed = |c: f64, f: &Fl<DV, B>| -> Vec<Fl<DV, B>> {
        tp.iter()
            .map(|&(n, s, t)| {
                if n == 1 {
                    f.muls(DD::from_f64(c * s + (1.0 - c) * t))
                } else {
                    Fl::zero()
                }
            })
            .collect()
    };
    let v0 = T::make(&e, &seed(1.0, &e));
    let three = Fl::<T, B>::splat(DD::from_f64(3.0));
    let omega = (T::make(&Fl::splat(DD::from_f64(1.3)), &seed(0.0, &Fl::splat(DD::from_f64(3.0)))) / three).0[0];
    let mut lus = factors(&d, omega.val().lane(0), shift);
    for lu in lus.iter_mut() {
        lu.refine = refine;
        setup(lu);
    }
    let (u, v) = series(&lus, &d, omega, &v0);
    let om = Fl::<T, B>([omega; B]);
    let om0 = T::get(&om, 0)[0];
    let x = DD::from_f64(0.5);
    let (us, vs) = (horner(&u, x), horner(&v, x));
    let mut errs = Vec::new();
    for p in 0..=tp.len() {
        let (n, sigma) = if p == 0 { (0, 0.0) } else { (tp[p - 1].0, tp[p - 1].1) };
        let tau = if p == 0 {
            OD::from_f64(0.0)
        } else {
            T::get(&om, if n == 1 { p } else { 1 })[0]
        };
        let cf = closed(om0, OD::from_f64(sigma), tau, ORDER);
        let ep: Vec<OD> = (0..M).map(|i| OD::from_dd(e.get(i))).collect();
        let at = |k: usize, gh: usize| -> Vec<OD> {
            let c = if gh == 0 { cf[k].0[n] } else { cf[k].1[n] };
            ep.iter().map(|&ei| ei * c).collect()
        };
        let mut worst = 0.0f64;
        for k in 0..=ORDER {
            worst = nan_max(worst, rel_err(&T::get(&u[k], p), &at(k, 0)));
            worst = nan_max(worst, rel_err(&T::get(&v[k], p), &at(k, 1)));
        }
        let xo = OD::from_f64(0.5);
        let sum = |gh: usize| -> Vec<OD> {
            let mut s = vec![OD::from_f64(0.0); M];
            for k in (0..=ORDER).rev() {
                let a = at(k, gh);
                for i in 0..M {
                    s[i] = s[i] * xo + a[i];
                }
            }
            s
        };
        worst = nan_max(worst, rel_err(&T::get(&us, p), &sum(0)));
        worst = nan_max(worst, rel_err(&T::get(&vs, p), &sum(1)));
        errs.push(worst);
    }
    errs
}

/// whether the errors are within the lane type's bounds
fn passes<T: Probe>(e: &[f64]) -> bool {
    e[0] <= T::VAL.bound() && e[1..].iter().all(|&x| x <= T::DER.bound())
}

/// (lane type, passes, errors) of `slaved_errors` on every lane type
fn all_lanes(refine: Refine, shift: usize, setup: &dyn Fn(&mut PeriodicLu)) -> Vec<(&'static str, bool, Vec<f64>)> {
    macro_rules! each {
        ($($t:ty),*) => {
            vec![$({
                let e = slaved_errors::<$t>(refine, shift, setup);
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

/// The slaved series to order 70 in every lane type against the closed form: every order of u and v and their sums at
/// x = 1/2, the values and the derivative parts in V and omega (tangents, complex and f64 tangents, jet
/// coefficients). Pass: each part's largest error over the orders and sums, relative to that order's closed form,
/// within its precision's bound (1e-27, 1e-57, 1e-115, 1e-11), written before the first run.
#[test]
fn slaved_series_is_its_closed_form() {
    let r = all_lanes(Refine::default(), 0, &|_| {});
    show("slaved", &r);
    assert!(r.iter().all(|x| x.1), "a lane type fails");
}

/// The controls of `slaved_series_is_its_closed_form`: without refinement the quad- and octo-double lane types must
/// fail; with order k's factors taken from order k - 1 every lane type must fail.
#[test]
fn slaved_series_controls_fail() {
    let r = all_lanes(Refine { qd: 0, od: 0 }, 0, &|_| {});
    show("unrefined", &r);
    for name in ["QV", "OV", "Qu", "Ou", "Oun<2>", "QJet<4>", "OJet<4>"] {
        assert!(
            !r.iter().find(|x| x.0 == name).unwrap().1,
            "unrefined: control did not fire on {name}"
        );
    }
    let w = all_lanes(Refine::default(), 1, &|_| {});
    show("wrong order", &w);
    assert!(w.iter().all(|x| !x.1), "wrong order: control did not fire");
}

/// the slaved series' sums at x = 1/2 of one lane type (`slaved_errors`'s inputs, omega's value moved by `ulps` ulps
/// of its leading part) into the hash
fn slaved_hash<T: Probe>(h: &mut common::Fnv, ulps: u64) {
    let d = dmat::<B>(Parity::Even);
    let e = e_field();
    let np = T::parts().len();
    let v0 = T::make(&e, &vec![e; np]);
    let w = f64::from_bits(1.3f64.to_bits() + ulps);
    let omega = (T::make(&Fl::splat(DD::from_f64(w)), &vec![Fl::splat(DD::ONE); np])
        / Fl::<T, B>::splat(DD::from_f64(3.0)))
    .0[0];
    let lus = factors(&d, omega.val().lane(0), 0);
    let (u, v) = series(&lus, &d, omega, &v0);
    let x = DD::from_f64(0.5);
    common::probe::hash_parts(h, &horner(&u, x));
    common::probe::hash_parts(h, &horner(&v, x));
}

/// the hash of the slaved series' sums over the lane types
fn slaved_hashes(ulps: u64) -> u64 {
    let mut h = common::Fnv::new();
    slaved_hash::<DV>(&mut h, ulps);
    slaved_hash::<QV>(&mut h, ulps);
    slaved_hash::<OV>(&mut h, ulps);
    slaved_hash::<Du>(&mut h, ulps);
    slaved_hash::<Qu>(&mut h, ulps);
    slaved_hash::<Ou>(&mut h, ulps);
    slaved_hash::<Dc>(&mut h, ulps);
    slaved_hash::<Jet<4>>(&mut h, ulps);
    slaved_hash::<QJet<4>>(&mut h, ulps);
    slaved_hash::<OJet<4>>(&mut h, ulps);
    slaved_hash::<Fu>(&mut h, ulps);
    slaved_hash::<Dtn<2>>(&mut h, ulps);
    h.0
}

/// the hash of `slaved_hashes(0)` on every backend
const SLAVED_REFERENCE: u64 = 0xafd7689b8868507a;

/// The slaved series' bytes (every part of both sums in twelve lane types) do not depend on multidouble's SIMD
/// backend. Pass: the hash equals the reference, which CI checks on SSE2, AVX2, portable and NEON.
#[test]
fn slaved_bytes_match_reference() {
    let h = slaved_hashes(0);
    println!("slaved series hash {h:#018x}");
    assert_eq!(
        h, SLAVED_REFERENCE,
        "the slaved series' bytes differ from the reference"
    );
}

/// The control of `slaved_bytes_match_reference`: omega one ulp up must change the hash.
#[test]
fn slaved_bytes_control_one_ulp_differs() {
    assert_ne!(slaved_hashes(1), slaved_hashes(0), "control did not fire");
}

/// The slaved model as a boundary-value problem on [1/4, 1]: the series at x = 0 summed at the first node, Taylor
/// steps between the nodes, v given at x = 1. Unknowns: V (the left data), the nodes' (u, v), u(1) (the right data)
/// and omega (a global, fixed by an extra row), so that the series' derivative in omega enters the Jacobian.
pub struct Slaved {
    pub d: DMat<B>,
    pub e: Fl<DV, B>,
    /// v at the right end, E h(1)
    pub v_r: Fl<DV, B>,
    /// the control: order k's factors from order k - 1
    pub shift: usize,
    /// the control of the Jacobian: the periodic solves' coupling dropped (feature `mutate`)
    pub uncoupled: bool,
}

impl Slaved {
    pub fn new() -> Self {
        let e = e_field();
        let hr = closed_sum(OD::from_f64(0.9), 1.0).1;
        let v_r = field::<B>(|i| (OD::from_dd(e.get(i)) * hr).to_dd());
        Slaved {
            d: dmat::<B>(Parity::Even),
            e,
            v_r,
            shift: 0,
            uncoupled: false,
        }
    }
}

impl Default for Slaved {
    fn default() -> Self {
        Self::new()
    }
}

/// (g(x), h(x)) of the closed form at omega, summed to order 200
fn closed_sum(omega: OD, x: f64) -> (OD, OD) {
    let cf = closed(omega, OD::from_f64(0.0), OD::from_f64(0.0), 200);
    let xo = OD::from_f64(x);
    cf.iter()
        .rev()
        .fold((OD::from_f64(0.0), OD::from_f64(0.0)), |(g, h), c| {
            (g * xo + c.0[0], h * xo + c.1[0])
        })
}

impl Equations<B> for Slaved {
    const NF: usize = 2;
    const NL: usize = 1;
    const NR: usize = 1;
    const NG: usize = 1;
    type Prep = Vec<PeriodicLu>;

    /// every order's factors at the state's double-double omega (the last unknown)
    fn try_prepare(&self, zh: &[f64], zl: &[f64]) -> Result<Vec<PeriodicLu>, String> {
        let og = zh.len() - 1;
        #[allow(unused_mut)]
        let mut lus = factors(&self.d, DD::new(zh[og], zl[og]), self.shift);
        #[cfg(feature = "mutate")]
        if self.uncoupled {
            lus.iter_mut()
                .for_each(|lu| lu.set_control(multishoot::control::Control::PeriodicCoupling));
        }
        Ok(lus)
    }

    fn prepare(&self, zh: &[f64], zl: &[f64]) -> Vec<PeriodicLu> {
        self.try_prepare(zh, zl).unwrap()
    }

    /// x0 (k + 1) u_(k+1) = v_k - (omega D + kappa + k) u_k, (k + 1) v_(k+1) = alpha u_k + beta v_k, with 1 / x0 and
    /// 1 / (k + 1) in the lane's precision
    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        let omega = g[0];
        let kap = kappa(&self.d, &Fl::<T, B>::cst(&self.e), omega);
        let ix0 = LaneConst::new(OD::ONE / OD::from_f64(x0));
        for k in 0..w.kt {
            let (u, v) = (w.c[0][k], w.c[1][k]);
            let lu = self.d.apply(&u).mul_t(omega) + kap.adds(DD::from_f64(k as f64)) * u;
            let r = LaneConst::recip(k as u64 + 1);
            w.c[0][k + 1] = r.mul(ix0.mul(v - lu));
            w.c[1][k + 1] = r.mul(u.mulf(ALPHA) + v.mulf(BETA));
        }
    }

    /// the series from V = data[0], summed at the first node
    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &Vec<PeriodicLu>) -> State<T, B> {
        let (u, v) = series(prep, &self.d, g[0], &data[0]);
        vec![horner(&u, DD::from_f64(x)), horner(&v, DD::from_f64(x))]
    }

    /// u is the data, v = E h(1)
    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, B>], _g: &[T], _prep: &Vec<PeriodicLu>) -> State<T, B> {
        vec![data[0], Fl::cst(&self.v_r)]
    }
}

const OMEGA: f64 = 0.9;
const NODES: [f64; 4] = [0.25, 0.5, 0.75, 1.0];

/// the layout on NODES with the row omega = 0.9, planned from the closed form, and the closed form's z
fn slaved_bvp(eq: Slaved) -> (Shooting<Slaved, B>, Vec<Vec<f64>>) {
    let mut sh = Shooting::new(eq, NODES.to_vec(), 30);
    let og = sh.off_global();
    sh.extra.push(ExtraRow {
        terms: vec![(og, DD::ONE)],
        rhs: DD::from_f64(OMEGA),
    });
    let l = sh.len();
    let (mut zh, mut zl) = (vec![0.0; l], vec![0.0; l]);
    let e = sh.eq.e;
    let mut put = |o: usize, c: OD| {
        let f = field::<B>(|i| (OD::from_dd(e.get(i)) * c).to_dd());
        f.store(&mut zh[o..o + M], &mut zl[o..o + M]);
    };
    put(0, OD::ONE);
    for (i, &x) in NODES.iter().enumerate() {
        let (g, h) = closed_sum(OD::from_f64(OMEGA), x);
        put(sh.off_node(i), g);
        put(sh.off_node(i) + M, h);
    }
    put(sh.off_right(), closed_sum(OD::from_f64(OMEGA), 1.0).0);
    zh[og] = OMEGA;
    let z = vec![zh, zl];
    sh.plan_segments(
        &z[0],
        &z[1],
        &Plan {
            kt: 30,
            tol: 1e-34,
            hmax: 0.125,
        },
    );
    (sh, z)
}

fn opts(tol: f64, max_iter: usize) -> NewtonOpts {
    NewtonOpts {
        tol,
        max_iter,
        ..Default::default()
    }
}

/// Newton in double-double from z* perturbed by 1e-3, then in quad-double: (dd converged, qd max |F|, max |z - z*|
/// from the quad-double root, dd steps)
fn solve(sh: &Shooting<Slaved, B>, zs: &[Vec<f64>]) -> (bool, f64, f64, usize) {
    let mut z = perturbed(zs, 1e-3);
    let dd = sh.newton::<Double>(&mut z, &opts(1e-28, 8)).unwrap();
    let mut zq = widen(&z, 4);
    let qd = sh.newton::<Quad>(&mut zq, &opts(1e-60, 4)).unwrap();
    let err = (0..sh.len())
        .map(|i| (DD::new(zq[0][i], zq[1][i]) - DD::new(zs[0][i], zs[1][i])).hi.abs())
        .fold(0.0, nan_max);
    let fq = *qd.fnorm.last().unwrap();
    println!(
        "dd max |F| {:?}, qd {:?}; max |z - z*| {err:.3e}",
        dd.fnorm.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>(),
        qd.fnorm.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>()
    );
    (dd.converged, fq, err, dd.fnorm.len())
}

/// The boundary-value problem with the slaved series at its left end, from the closed form perturbed by 1e-3: Newton
/// in double-double, then in quad-double. Pass: double-double converged within 8 steps, quad-double max |F| below
/// 1e-60, and every unknown within 1e-27 of the closed form (its values rounded to double-double, and D's entries).
#[test]
fn slaved_bvp_converges_to_its_closed_form() {
    let (sh, zs) = slaved_bvp(Slaved::new());
    let r = solve(&sh, &zs);
    assert!(r.0 && r.1 < 1e-60 && r.2 < 1e-27, "{r:?}");
}

/// The control of `slaved_bvp_converges_to_its_closed_form`: with order k's factors from order k - 1 in the series the
/// root must miss the closed form.
#[test]
fn slaved_bvp_control_wrong_order_fails() {
    let mut eq = Slaved::new();
    eq.shift = 1;
    let (sh, zs) = slaved_bvp(eq);
    let r = solve(&sh, &zs);
    assert!(!(r.0 && r.1 < 1e-60 && r.2 < 1e-27), "control did not fire: {r:?}");
}

/// The exact Jacobian's columns of the left data and of omega, which reach the series' tangent solves, against central
/// differences of the residual at z* perturbed by 1e-2, h = 2^-30: (worst relative difference, its column)
fn jacobian_vs_central(sh: &Shooting<Slaved, B>, zs: &[Vec<f64>]) -> (f64, usize) {
    let z = perturbed(zs, 1e-2);
    let l = sh.len();
    let h = 2f64.powi(-30);
    let (c, ld) = sh.jacobian::<Double>(&parts(&z), true).unwrap();
    let cols: Vec<usize> = (0..M).chain([sh.off_global()]).collect();
    let mut worst = (0.0f64, 0usize);
    for &j in &cols {
        let (mut zp, mut zm) = (z.clone(), z.clone());
        let vj = DD::new(z[0][j], z[1][j]);
        let (p, m) = (vj + DD::from_f64(h), vj - DD::from_f64(h));
        (zp[0][j], zp[1][j]) = (p.hi, p.lo);
        (zm[0][j], zm[1][j]) = (m.hi, m.lo);
        let fp = sh.residual::<Double>(&parts(&zp), true).unwrap();
        let fm = sh.residual::<Double>(&parts(&zm), true).unwrap();
        let col: Vec<DD> = (0..l).map(|i| DD::new(c[0][j * ld + i], c[1][j * ld + i])).collect();
        let scale = col.iter().map(|v| v.hi.abs()).fold(1.0, nan_max);
        let err = (0..l)
            .map(|i| (col[i] - (fp[i] - fm[i]) / DD::from_f64(2.0 * h)).hi.abs())
            .fold(0.0, nan_max);
        if err / scale > worst.0 {
            worst = (err / scale, j);
        }
    }
    worst
}

/// The exact Jacobian through the series' tangent solves (columns of V and omega) against central differences: the
/// differences' truncation is about h^2 = 1e-18 and their rounding 1e-32 / h = 1e-23. Pass: within 1e-13 relative to
/// max(1, the column's largest entry).
#[test]
fn slaved_jacobian_matches_central_differences() {
    let (sh, zs) = slaved_bvp(Slaved::new());
    let (worst, col) = jacobian_vs_central(&sh, &zs);
    println!("Jacobian vs central differences: worst {worst:.3e} at column {col}");
    assert!(worst < 1e-13, "column {col}: {worst:e}");
}

/// The control of `slaved_jacobian_matches_central_differences` (feature `mutate`): with the periodic solves' coupling
/// dropped the omega column must fail.
#[cfg(feature = "mutate")]
#[test]
fn slaved_jacobian_control_uncoupled_fails() {
    let mut eq = Slaved::new();
    eq.uncoupled = true;
    let (sh, zs) = slaved_bvp(eq);
    let (worst, col) = jacobian_vs_central(&sh, &zs);
    println!("control, coupling dropped: worst {worst:.3e} at column {col}");
    assert!(
        worst >= 1e-13 && col == sh.off_global(),
        "control did not fire: {worst:e} at {col}"
    );
}
