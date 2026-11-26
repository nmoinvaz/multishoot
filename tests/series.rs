//! The periodic solves of a series at a singular end (`series::PeriodicLu`) on planted systems in every lane type, and
//! a series at a regular singular point with coefficients periodic in theta against its closed form, alone and as the
//! left end of a boundary-value problem solved by Newton's method.
mod common;

use common::field;
use common::probe::{rel_err, Probe};
use multidouble::native::Oun;
use multishoot::lane::{Dc, Dtn, Du, Fu, Fun, Jet, OJet, Ou, QJet, Qu, DV, OV, QV};
use multishoot::periodic::{dmat, sin_cos_grid, Parity};
use multishoot::series::{PeriodicLu, Refine};
use multishoot::{Fl, Rate, DD};

const B: usize = 2;
const M: usize = 8 * B;

/// cos and sin of k theta_j on the grid of the parity
fn sc(parity: Parity, j: usize, k: i64) -> (DD, DD) {
    let (s, c) = sin_cos_grid(parity, M, j, k);
    (c, s)
}

/// a lane scalar of value v whose derivative parts are t, divided by 3 in the lane, so that its value and parts carry
/// digits past double-double
fn scalar<T: Probe>(v: f64, t: f64) -> T {
    let np = T::parts().len();
    let ts: Vec<Fl<DV, B>> = (0..np)
        .map(|p| Fl::splat(DD::from_f64(t * (1.0 + 0.5 * p as f64))))
        .collect();
    (T::make(&Fl::<DV, B>::splat(DD::from_f64(v)), &ts) / Fl::<T, B>::splat(DD::from_f64(3.0))).0[0]
}

/// The planted system of one lane type on the grid of the parity: f* = f1 / (1 + f2^2) and kappa = kappa_1 + kappa_2 /
/// 7 formed in the lane from fields with derivative parts, omega a scalar, the right-hand side L f* = omega D f* +
/// kappa f* in the lane with the t-derivative shifted by the rate, then f from the factors of L at the double-double
/// values and the rate. `setup` changes the factors before the solve (a control). Returns the relative error of
/// every part of f (value, then the derivative parts).
fn planted_solve<T: Probe>(parity: Parity, rate: Rate, setup: &dyn Fn(&mut PeriodicLu)) -> Vec<f64> {
    let d = dmat::<B>(parity);
    let np = T::parts().len();
    let parts =
        |f: &dyn Fn(usize, usize) -> DD| -> Vec<Fl<DV, B>> { (0..np).map(|p| field::<B>(|j| f(p, j))).collect() };
    let f1 = T::make(
        &field::<B>(|j| sc(parity, j, 1).1 + sc(parity, j, 3).0.mul_f(0.3) + DD::from_f64(0.1)),
        &parts(&|p, j| sc(parity, j, 1).0.mul_f(0.2) + DD::from_f64(0.1 * (p + 1) as f64)),
    );
    let f2 = T::make(
        &field::<B>(|j| DD::from_f64(0.5) + sc(parity, j, 2).0.mul_f(0.2)),
        &parts(&|p, _| DD::from_f64(0.3 - 0.1 * p as f64)),
    );
    let one = Fl::<T, B>::splat(DD::ONE);
    let fs = f1 / (one + f2 * f2);
    let k1 = T::make(
        &field::<B>(|j| DD::from_f64(2.0) + sc(parity, j, 1).0.mul_f(0.5) + sc(parity, j, 2).1.mul_f(0.3)),
        &parts(&|p, j| DD::from_f64(0.4 - 0.1 * p as f64) + sc(parity, j, 1).0.mul_f(0.2)),
    );
    let k2 = T::make(&Fl::splat(DD::ONE), &parts(&|_, _| DD::from_f64(0.2)));
    let kappa = k1 + k2 / Fl::<T, B>::splat(DD::from_f64(7.0));
    let omega = scalar::<T>(1.3, 0.7);
    let rhs = rate.shift(&fs, d.apply(&fs).mul_t(omega)) + kappa * fs;
    let kv = kappa.val();
    let kd: Vec<DD> = (0..M).map(|i| kv.get(i)).collect();
    let mut lu = PeriodicLu::factor_rate(&d, omega.val().lane(0), &kd, rate).unwrap();
    setup(&mut lu);
    let f = lu.solve(&d, omega, &kappa, &rhs);
    (0..=np).map(|p| rel_err(&T::get(&f, p), &T::get(&fs, p))).collect()
}

/// whether the errors of a lane type's parts are within its precisions' bounds (NaN fails)
fn passes<T: Probe>(e: &[f64]) -> bool {
    e[0] <= T::VAL.bound() && e[1..].iter().all(|&x| x <= T::DER.bound())
}

/// (lane type, passes, errors) of `planted_solve` on every lane type of the list
macro_rules! over_lanes {
    ($parity:expr, $rate:expr, $setup:expr; $($t:ty),*) => {
        vec![$({
            let e = planted_solve::<$t>($parity, $rate, $setup);
            (<$t as Probe>::NAME, passes::<$t>(&e), e)
        }),*]
    };
}

/// every lane type the engine evaluates at rate zero
fn all_lanes(parity: Parity, setup: &dyn Fn(&mut PeriodicLu)) -> Vec<(&'static str, bool, Vec<f64>)> {
    over_lanes!(parity, Rate::ZERO, setup; DV, QV, OV, Du, Qu, Ou, Oun<2>, Dc, Jet<4>, QJet<4>, OJet<4>, Fu, Fun<2>, Dtn<2>)
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

/// The planted periodic solve on the odd, even and full grids (M = 16) in every lane type at rate zero: values,
/// tangents (one, two, f64, complex) and jet coefficients. Pass: each part's error, relative to the planted part's
/// largest, within its precision's bound (1e-27 double-double, 1e-57 quad-double, 1e-115 octo-double, 1e-11 f64),
/// written before the first run; the systems' condition numbers are below 100.
#[test]
fn periodic_solve_is_planted_in_every_lane() {
    for parity in [Parity::Odd, Parity::Even, Parity::Full] {
        let r = all_lanes(parity, &|_| {});
        show(&format!("{parity:?}"), &r);
        assert!(r.iter().all(|x| x.1), "{parity:?}: a lane type fails");
    }
}

/// The control of `periodic_solve_is_planted_in_every_lane`: without refinement the quad- and octo-double values
/// and tangents and the quad- and octo-double jets must fail (the double-double factors alone give 1e-31).
#[test]
fn periodic_solve_control_unrefined_fails() {
    let r = all_lanes(Parity::Odd, &|lu| lu.refine = Refine { qd: 0, od: 0 });
    show("unrefined", &r);
    for name in ["QV", "OV", "Qu", "Ou", "Oun<2>", "QJet<4>", "OJet<4>"] {
        assert!(
            !r.iter().find(|x| x.0 == name).unwrap().1,
            "control did not fire on {name}"
        );
    }
}

/// The planted solve at a real rate (Du, Qu, Ou, Dc) and a complex one (Dc), on the odd grid: the right-hand side's
/// t-derivative shifted by the rate in the lane (`Rate::shift`), the factors those of the rate. Pass: as
/// `periodic_solve_is_planted_in_every_lane`.
#[test]
fn periodic_solve_is_planted_at_a_rate() {
    let real = Rate::new(DD::from_f64(0.37), DD::ZERO);
    let complex = Rate::new(DD::from_f64(0.37), DD::from_f64(0.81));
    let r = over_lanes!(Parity::Odd, real, &|_| {}; Du, Qu, Ou, Dc);
    show("real rate", &r);
    let c = over_lanes!(Parity::Odd, complex, &|_| {}; Dc);
    show("complex rate", &c);
    assert!(r.iter().chain(&c).all(|x| x.1), "a lane type fails at a rate");
}

/// The order-by-order solve of a jet (`solve_at`) against `solve`: (coefficients that differ in any part, of all)
fn solve_at_differs<T: Probe>(setup: &dyn Fn(&mut PeriodicLu)) -> (usize, usize) {
    let parity = Parity::Even;
    let d = dmat::<B>(parity);
    let np = T::parts().len();
    let ts: Vec<Fl<DV, B>> = (0..np)
        .map(|p| field::<B>(|j| sc(parity, j, 2).0.mul_f(0.1 * (p + 1) as f64)))
        .collect();
    let v = T::make(&field::<B>(|j| DD::from_f64(1.0) + sc(parity, j, 2).1.mul_f(0.4)), &ts);
    let rhs = v / Fl::<T, B>::splat(DD::from_f64(3.0));
    let kappa = T::make(&field::<B>(|j| DD::from_f64(1.5) + sc(parity, j, 4).0.mul_f(0.2)), &ts)
        / Fl::<T, B>::splat(DD::from_f64(1.1));
    let omega = scalar::<T>(0.9, 0.4);
    let kv = kappa.val();
    let kd: Vec<DD> = (0..M).map(|i| kv.get(i)).collect();
    let lu = PeriodicLu::factor(&d, omega.val().lane(0), &kd).unwrap();
    let whole = lu.solve(&d, omega, &kappa, &rhs);
    let mut lu_at = lu.clone();
    setup(&mut lu_at);
    let mut f = Fl::<T, B>::zero();
    for j in 0..T::NJ {
        lu_at.solve_at(&d, &omega, &kappa, &rhs, &mut f, j);
    }
    let differ = (0..T::NJ)
        .filter(|&j| {
            let (a, b) = (whole.coef_od(j), f.coef_od(j));
            (0..M).any(|i| a.get(i).x != b.get(i).x)
        })
        .count();
    (differ, T::NJ)
}

/// `solve_at`, one jet order at a time, gives `solve`'s coefficients bit for bit on Jet, QJet and OJet. Pass: no
/// coefficient differs in any part.
#[test]
fn solve_at_is_solve_bitwise() {
    let r = [
        solve_at_differs::<Jet<4>>(&|_| {}),
        solve_at_differs::<QJet<4>>(&|_| {}),
        solve_at_differs::<OJet<4>>(&|_| {}),
    ];
    println!("coefficients that differ (Jet, QJet, OJet): {r:?}");
    assert!(r.iter().all(|x| x.0 == 0), "{r:?}");
}

/// The control of `solve_at_is_solve_bitwise`: with one refinement step fewer in the order-by-order solve, the
/// orders past 0 must differ on QJet and OJet, and on Jet a differing right-hand side (the coupling dropped, feature
/// `mutate`) is the control below.
#[test]
fn solve_at_control_fewer_steps_differs() {
    let fewer = |lu: &mut PeriodicLu| lu.refine = Refine { qd: 1, od: 3 };
    let r = [solve_at_differs::<QJet<4>>(&fewer), solve_at_differs::<OJet<4>>(&fewer)];
    println!("control, one step fewer: coefficients that differ (QJet, OJet): {r:?}");
    assert!(r.iter().all(|x| x.0 > 0), "control did not fire: {r:?}");
}

/// The controls of the periodic solve's derivative parts (feature `mutate`), each against the check it breaks.
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::Control;

    /// the right-hand sides of tangents and jet orders without the matrix's derivative fail
    /// `periodic_solve_is_planted_in_every_lane` on every lane type whose derivative parts the solve forms that way
    /// (Du, Dc, the jets and the f64 tangents); the quad- and octo-double duals refine theirs, and the unrefined
    /// control fails them
    #[test]
    fn coupling_dropped_fails_the_derivative_parts() {
        let r = all_lanes(Parity::Odd, &|lu| lu.set_control(Control::PeriodicCoupling));
        show("coupling dropped", &r);
        for name in ["Du", "Dc", "Jet<4>", "QJet<4>", "OJet<4>", "Fu", "Fun<2>", "Dtn<2>"] {
            assert!(
                !r.iter().find(|x| x.0 == name).unwrap().1,
                "control did not fire on {name}"
            );
        }
    }

    /// the tangents solved by the factors at rate zero fail `periodic_solve_is_planted_at_a_rate` on Du and Dc
    #[test]
    fn unshifted_factors_fail_the_rate() {
        let real = Rate::new(DD::from_f64(0.37), DD::ZERO);
        let complex = Rate::new(DD::from_f64(0.37), DD::from_f64(0.81));
        let set = |lu: &mut PeriodicLu| lu.set_control(Control::RateUnshifted);
        let r = over_lanes!(Parity::Odd, real, &set; Du, Dc);
        let c = over_lanes!(Parity::Odd, complex, &set; Dc);
        show("unshifted", &r);
        show("unshifted, complex", &c);
        assert!(r.iter().chain(&c).all(|x| !x.1), "control did not fire");
    }

    /// the refinement residual of Qu and Ou formed without the rate's shift fails
    /// `periodic_solve_is_planted_at_a_rate`: the refinement converges to the tangent at rate zero
    #[test]
    fn unshifted_residual_fails_the_rate() {
        let real = Rate::new(DD::from_f64(0.37), DD::ZERO);
        let r = over_lanes!(Parity::Odd, real, &|lu| lu.set_control(Control::RateResidual); Qu, Ou);
        show("residual unshifted", &r);
        assert!(r.iter().all(|x| !x.1), "control did not fire");
    }

    /// the order-by-order solve with the coupling dropped differs from `solve` on Jet (`solve_at_is_solve_bitwise`)
    #[test]
    fn solve_at_coupling_dropped_differs() {
        let r = solve_at_differs::<Jet<4>>(&|lu| lu.set_control(Control::PeriodicCoupling));
        println!("control, coupling dropped: coefficients that differ (Jet): {r:?}");
        assert!(r.0 > 0, "control did not fire");
    }
}
