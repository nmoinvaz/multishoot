//! The asymptotic numerical method (multisolver's) on the eigenproblem `Cylinder`, whose eigenvalue pi^2 + mu is
//! linear in its parameter mu while its mode does not change. Its model-problem tests are multisolver's.
mod common;

use common::{perturbed, Cylinder};
use multishoot::lane::Lane;
use multishoot::{Double, Equations, Fl, Rate, Shooting, State, Work, DD};
use multisolver::anm::{series, Branch};
use multisolver::prec::{nan_max, parts};
use multisolver::NewtonOpts;

const B: usize = 1;

/// The series of order `order` of `Cylinder`'s branch in mu at its double-double root, refined once, and the index
/// of the eigenvalue among the unknowns
fn cylinder(freeze: bool, order: usize) -> (Branch<Double>, usize) {
    let (mut sh, zs) = Cylinder::<B>::shooting(Cylinder::new(), 30);
    let mut z = perturbed(&zs, 1e-3);
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 12,
        ..Default::default()
    };
    assert!(sh.newton::<Double>(&mut z, &opts).unwrap().converged);
    sh.eq.freeze_mu = freeze;
    (
        series::<Double, _>(&sh, &parts(&z), order, 1, true).unwrap(),
        sh.off_global(),
    )
}

/// [|z_1 - e_lambda|, max over k >= 2 of |z_k|], NaN where any entry is
fn cylinder_errors(br: &Branch<Double>, og: usize) -> [f64; 2] {
    let e1 = br.z[1]
        .iter()
        .enumerate()
        .map(|(i, &v)| (v - if i == og { DD::ONE } else { DD::ZERO }).hi.abs())
        .fold(0.0, nan_max);
    let high = (2..=br.order())
        .flat_map(|k| br.z[k].iter().map(|v| v.hi.abs()))
        .fold(0.0, nan_max);
    [e1, high]
}

/// the accuracy checks of Cylinder's branch: [z_1 within 1e-28, the higher orders within 1e-26]
fn cylinder_checks(e: [f64; 2]) -> [bool; 2] {
    [e[0] <= 1e-28, e[1] <= 1e-26]
}

/// Pass, written before the first run: through the multiple-shooting problem, the series of order 10 (jets of 8 and
/// 16 coefficients) gives z_1 = e_lambda, one in the eigenvalue and zero elsewhere, to 1e-28, and every higher order
/// below 1e-26. The controls: with mu frozen in the residual's series, z_1 misses e_lambda by all of it (its higher
/// orders are zero); z_2 = 1e-20 in one entry, and a NaN in z_4, fail the higher orders.
#[test]
fn anm_follows_the_eigenvalue_through_shooting() {
    let (br, og) = cylinder(false, 10);
    let e = cylinder_errors(&br, og);
    let res = br.residual.iter().copied().fold(0.0, nan_max);
    println!(
        "cylinder: |z_1 - e_lambda| {:.3e}, z_2..z_10 {:.3e}, solve residuals up to {res:.3e}",
        e[0], e[1]
    );
    assert_eq!(cylinder_checks(e), [true; 2]);
    let control = |name: &str, br: &Branch<Double>, want: [bool; 2]| {
        let e = cylinder_errors(br, og);
        println!(
            "control, {name}: |z_1 - e_lambda| {:.3e}, higher orders {:.3e}",
            e[0], e[1]
        );
        assert_eq!(cylinder_checks(e), want, "control {name} did not fire");
    };
    control("mu frozen", &cylinder(true, 3).0, [false, true]);
    let mut bad = br.clone();
    bad.z[2][0] = DD::from_f64(1e-20);
    control("z_2 = 1e-20", &bad, [true, false]);
    let mut bad = br;
    bad.z[4][0] = DD::from_f64(f64::NAN);
    control("NaN in z_4", &bad, [true, false]);
}

/// Equations E declared order by order (`Equations::ORDER_BY_ORDER`) with E's default order-k methods, so that the layout's
/// `jet_orders` gives its session; the session over the defaults forms every order, so its bits are `residual_jet`'s
struct OrderByOrder<E>(E);

impl<E: Equations<B>, const B: usize> Equations<B> for OrderByOrder<E> {
    const NF: usize = E::NF;
    const NL: usize = E::NL;
    const NR: usize = E::NR;
    const NG: usize = E::NG;
    const NAUX: usize = E::NAUX;
    const POINTWISE: bool = E::POINTWISE;
    const ORDER_BY_ORDER: bool = true;
    type Prep = E::Prep;
    fn prepare(&self, zh: &[f64], zl: &[f64]) -> E::Prep {
        self.0.prepare(zh, zl)
    }
    fn prepare_rate(&self, zh: &[f64], zl: &[f64], rate: Rate) -> E::Prep {
        self.0.prepare_rate(zh, zl, rate)
    }
    fn try_prepare(&self, zh: &[f64], zl: &[f64]) -> Result<E::Prep, String> {
        self.0.try_prepare(zh, zl)
    }
    fn try_prepare_rate(&self, zh: &[f64], zl: &[f64], rate: Rate) -> Result<E::Prep, String> {
        self.0.try_prepare_rate(zh, zl, rate)
    }
    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        self.0.taylor_coeffs(x0, g, w)
    }
    fn left_state<T: Lane>(&self, x: f64, d: &[Fl<T, B>], g: &[T], p: &E::Prep) -> State<T, B> {
        self.0.left_state(x, d, g, p)
    }
    fn right_state<T: Lane>(&self, x: f64, d: &[Fl<T, B>], g: &[T], p: &E::Prep) -> State<T, B> {
        self.0.right_state(x, d, g, p)
    }
}

/// `Cylinder`'s layout at its double-double root, and the same layout over `OrderByOrder<Cylinder>`
fn cylinder_pair() -> (
    Shooting<Cylinder<B>, B>,
    Shooting<OrderByOrder<Cylinder<B>>, B>,
    Vec<Vec<f64>>,
) {
    let (sh, zs) = Cylinder::<B>::shooting(Cylinder::new(), 30);
    let mut z = perturbed(&zs, 1e-3);
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 12,
        ..Default::default()
    };
    assert!(sh.newton::<Double>(&mut z, &opts).unwrap().converged);
    let mut on = Shooting::new(OrderByOrder(Cylinder::new()), sh.nodes.clone(), sh.kt);
    on.segs = sh.segs.clone();
    on.extra = sh.extra.clone();
    (sh, on, z)
}

/// the number of coefficients, over every order, whose parts differ in any bit
fn coefficients_differ(a: &Branch<Double>, b: &Branch<Double>) -> usize {
    if a.z.len() != b.z.len() {
        return usize::MAX;
    }
    a.z.iter()
        .zip(&b.z)
        .map(|(x, y)| {
            if x.len() != y.len() {
                return x.len().max(y.len());
            }
            x.iter()
                .zip(y)
                .filter(|(u, v)| u.hi.to_bits() != v.hi.to_bits() || u.lo.to_bits() != v.lo.to_bits())
                .count()
        })
        .sum()
}

/// Pass, written before the first run: the series of order 10 of `Cylinder`'s branch through equations declared
/// order by order, whose orders come from the layout's `JetSession` (jets of 16 coefficients), is the series through
/// `Recompute` by `residual_jet` (jets of 8 and 16), bit for bit at every order, with and without threads.
#[test]
fn anm_series_through_the_session_is_bitwise() {
    let (sh, on, z) = cylinder_pair();
    for par in [false, true] {
        let a = series::<Double, _>(&sh, &parts(&z), 10, 1, par).unwrap();
        let b = series::<Double, _>(&on, &parts(&z), 10, 1, par).unwrap();
        let d = coefficients_differ(&a, &b);
        println!("par {par}: the session's series against residual_jet's, {d} coefficients differ");
        assert_eq!(d, 0, "par {par}: the session changed the series");
    }
}

/// The control of `anm_series_through_the_session_is_bitwise` (feature `mutate`): with the session's order k left as
/// it was with z_k zero (`Control::JetStale`), the series differs.
#[cfg(feature = "mutate")]
#[test]
fn anm_series_control_stale_session_differs() {
    let (sh, mut on, z) = cylinder_pair();
    on.set_control(multishoot::control::Control::JetStale);
    let a = series::<Double, _>(&sh, &parts(&z), 10, 1, true).unwrap();
    let b = series::<Double, _>(&on, &parts(&z), 10, 1, true).unwrap();
    let d = coefficients_differ(&a, &b);
    println!("control, stale session: {d} coefficients differ");
    assert!(d > 0, "control did not fire");
}
