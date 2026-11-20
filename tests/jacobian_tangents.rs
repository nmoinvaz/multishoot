//! The octo-double Jacobian with several tangents per evaluation (`Shooting::jacobian_tangents` with multidouble's
//! `Oun<N>`) against the Jacobian with one (`Shooting::jacobian` at `Octo`, `Ou` lanes): the eigenproblem `Cylinder`
//! (two globals, two extra rows, planned segments) and the planted problem with projected rows, at states perturbed
//! off the solution; and the layout as a problem with n tangents per evaluation (`Shooting::with_tangents`) against
//! the layout, through multisolver's Newton step.
mod common;

use common::{as_pointwise, perturbed, Cylinder, Planted};
use multishoot::multidouble::native::{Ou, Oun};
use multishoot::periodic::Parity;
use multishoot::{Equations, Native, Octo, Precision, Problem, Shooting};
use multisolver::newton::Step;
use multisolver::prec::{parts, widen};

/// the number of entries, over every part, in which two Jacobians differ (every entry when the shapes differ)
fn entries_differ(a: &(Vec<Vec<f64>>, usize), b: &(Vec<Vec<f64>>, usize)) -> usize {
    if a.1 != b.1 || a.0.len() != b.0.len() {
        return usize::MAX;
    }
    a.0.iter()
        .zip(&b.0)
        .map(|(p, q)| {
            if p.len() != q.len() {
                return p.len().max(q.len());
            }
            p.iter().zip(q).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
        })
        .sum()
}

/// (Ou, Oun<4>, Oun<8>) against `jacobian::<Octo>` at the state, entries that differ
fn compare<E: Equations<B>, const B: usize>(sh: &Shooting<E, B>, z8: &[Vec<f64>], par: bool) -> [usize; 3] {
    let z = parts(z8);
    let one = sh.jacobian::<Octo>(&z, par).unwrap();
    [
        entries_differ(&one, &sh.jacobian_tangents::<Octo, Ou>(&z, par).unwrap()),
        entries_differ(&one, &sh.jacobian_tangents::<Octo, Oun<4>>(&z, par).unwrap()),
        entries_differ(&one, &sh.jacobian_tangents::<Octo, Oun<8>>(&z, par).unwrap()),
    ]
}

fn cylinder() -> (Shooting<Cylinder<1>, 1>, Vec<Vec<f64>>) {
    let (sh, z) = Cylinder::<1>::shooting(Cylinder::new(), 30);
    (sh, widen(&perturbed(&z, 1e-3), 8))
}

fn projected() -> (Shooting<Planted<2>, 2>, Vec<Vec<f64>>) {
    let (mut sh, z) = Planted::<2>::shooting(&[0.0, 0.5, 1.0, 1.5, 2.0], 30, 1e-32);
    let k = sh.k();
    sh.project_rows(k, 1, 0, Parity::Even, 2).unwrap();
    (sh, widen(&perturbed(&z, 1e-3), 8))
}

/// Pass, written before the first run: on `Cylinder`, on the projected planted problem and on the planted problem as a
/// pointwise one (one evaluation seeding every point of a field, `Seed::Points`), at states perturbed by 1e-3, the
/// octo-double Jacobian with one tangent per evaluation through `jacobian_tangents` (Ou), with four and with eight
/// (Oun<4>, Oun<8>) is `jacobian::<Octo>`'s, every part of every entry, with and without threads (0 entries differ).
#[test]
fn jacobian_tangents_is_jacobian_bitwise() {
    let (cy, zc) = cylinder();
    let (pr, zp) = projected();
    let (pl, zl) = Planted::<2>::shooting(&[0.0, 0.5, 1.0, 1.5, 2.0], 30, 1e-32);
    let pw = as_pointwise(pl);
    let zw = widen(&perturbed(&zl, 1e-3), 8);
    for par in [false, true] {
        let c = compare(&cy, &zc, par);
        let p = compare(&pr, &zp, par);
        let w = compare(&pw, &zw, par);
        println!("par {par}: entries that differ (Ou, Oun<4>, Oun<8>): cylinder {c:?}, projected planted {p:?}, pointwise planted {w:?}");
        assert_eq!(
            (c, p, w),
            ([0; 3], [0; 3], [0; 3]),
            "par {par}: a multi-tangent Jacobian is not jacobian's"
        );
    }
}

/// The control of `jacobian_tangents_is_jacobian_bitwise` (feature `mutate`): with each evaluation's tangents read in
/// reverse order (`Control::TangentOrder`), the four- and eight-tangent Jacobians differ, while the one-tangent one,
/// which has nothing to reverse, does not.
#[cfg(feature = "mutate")]
#[test]
fn jacobian_tangents_control_order() {
    let (mut cy, zc) = cylinder();
    cy.set_control(multishoot::control::Control::TangentOrder);
    let c = compare(&cy, &zc, true);
    println!("control, tangents reversed: entries that differ (Ou, Oun<4>, Oun<8>) {c:?}");
    assert!(
        c[0] == 0 && c[1] > 0 && c[2] > 0,
        "control did not fire as written: {c:?}"
    );
}

/// the parts of dz and of F in which two Newton steps differ (every one when the shapes differ)
fn steps_differ(a: &Step<Octo>, b: &Step<Octo>) -> usize {
    let part = <Octo as Precision<Native>>::part;
    if a.f.len() != b.f.len() || a.dz.len() != b.dz.len() || a.dz.iter().zip(&b.dz).any(|(p, q)| p.len() != q.len()) {
        return usize::MAX;
    }
    let dz: usize =
        a.dz.iter()
            .zip(&b.dz)
            .map(|(p, q)| p.iter().zip(q).filter(|(x, y)| x.to_bits() != y.to_bits()).count())
            .sum();
    let f: usize =
        a.f.iter()
            .zip(&b.f)
            .map(|(&x, &y)| (0..8).filter(|&q| part(x, q).to_bits() != part(y, q).to_bits()).count())
            .sum();
    dz + f
}

/// Per n of 1, 2, 4 and 8, the view `with_tangents(n)` against the layout at the state: (entries of its Jacobian
/// that differ from `jacobian::<Octo>`'s, without and with threads; parts of dz and F of multisolver's `newton_step`
/// on it that differ from `Shooting::newton_step::<Octo>`'s, with threads), usize::MAX for an error
fn compare_view<E: Equations<B>, const B: usize>(sh: &Shooting<E, B>, z8: &[Vec<f64>]) -> [(usize, usize, usize); 4] {
    let z = parts(z8);
    let one = [false, true].map(|par| sh.jacobian::<Octo>(&z, par).unwrap());
    let step = sh.newton_step::<Octo>(&z, true).unwrap();
    [1, 2, 4, 8].map(|n| {
        let v = sh.with_tangents(n).unwrap();
        assert_eq!(v.tangents(), n);
        let jac = |par: bool| {
            Problem::<Octo>::jacobian(&v, &z, par).map_or(usize::MAX, |j| entries_differ(&one[par as usize], &j))
        };
        let st =
            multisolver::newton::newton_step::<Octo, _>(&v, &z, true).map_or(usize::MAX, |s| steps_differ(&step, &s));
        (jac(false), jac(true), st)
    })
}

/// Pass, written before the first run: on `Cylinder`, the projected planted problem and the planted problem without
/// its theta coupling declared pointwise, at states perturbed by 1e-3, the view `with_tangents(n)` for n = 1, 2, 4 and
/// 8 has `jacobian::<Octo>`'s Jacobian in every part of every entry, with and without threads, and multisolver's
/// `newton_step` on it gives `Shooting::newton_step::<Octo>`'s dz and F in every part (0 differ in each count); n = 0,
/// 3, 5 and 16 are errors.
#[test]
fn with_tangents_is_the_layout_bitwise() {
    let (cy, zc) = cylinder();
    let (pr, zp) = projected();
    let (pl, zl) = Planted::<2>::shooting_of(Planted::with_coupling(false), &[0.0, 0.5, 1.0, 1.5, 2.0], 30, 1e-32);
    let pw = as_pointwise(pl);
    let zw = widen(&perturbed(&zl, 1e-3), 8);
    let c = compare_view(&cy, &zc);
    let p = compare_view(&pr, &zp);
    let w = compare_view(&pw, &zw);
    let refused = [0, 3, 5, 16].map(|n| cy.with_tangents(n).is_err());
    println!(
        "n = 1, 2, 4, 8: (J entries that differ serial, threaded; Newton step parts that differ): cylinder {c:?}, \
         projected planted {p:?}, pointwise planted {w:?}; n = 0, 3, 5, 16 refused {refused:?}"
    );
    assert_eq!(
        (c, p, w),
        ([(0, 0, 0); 4], [(0, 0, 0); 4], [(0, 0, 0); 4]),
        "the view is not the layout"
    );
    assert_eq!(refused, [true; 4], "a count of tangents without lanes was taken");
}

/// The control of `with_tangents_is_the_layout_bitwise` (feature `mutate`): with each evaluation's columns read one
/// tangent early (`Control::TangentShift`), the view's Jacobian, with and without threads, and its Newton step differ
/// from the layout's on `Cylinder` for every n, one tangent included. A step that fails counts as differing: n = 1
/// and 2 do, at a zero pivot of the LU.
#[cfg(feature = "mutate")]
#[test]
fn with_tangents_control_shift() {
    let (mut cy, zc) = cylinder();
    cy.set_control(multishoot::control::Control::TangentShift);
    let c = compare_view(&cy, &zc);
    println!("control, tangents shifted: n = 1, 2, 4, 8: {c:?}");
    assert!(
        c.iter().all(|&(a, b, s)| a > 0 && b > 0 && s > 0),
        "control did not fire as written: {c:?}"
    );
}

/// Seconds of the octo-double Jacobian on the cylinder and on the projected planted problem with one, two, four and
/// eight tangents per evaluation, serial (the best of three each); reported, not asserted
#[test]
#[ignore]
fn jacobian_tangents_timing() {
    fn run<E: Equations<B>, const B: usize>(name: &str, sh: &Shooting<E, B>, z8: &[Vec<f64>]) {
        let z = parts(z8);
        let time = |f: &dyn Fn()| {
            (0..3)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64()
                })
                .fold(f64::MAX, f64::min)
        };
        let one = time(&|| {
            sh.jacobian::<Octo>(&z, false).unwrap();
        });
        let t2 = time(&|| {
            sh.jacobian_tangents::<Octo, Oun<2>>(&z, false).unwrap();
        });
        let t4 = time(&|| {
            sh.jacobian_tangents::<Octo, Oun<4>>(&z, false).unwrap();
        });
        let t8 = time(&|| {
            sh.jacobian_tangents::<Octo, Oun<8>>(&z, false).unwrap();
        });
        println!(
            "{name}, serial: one tangent {one:.3} s; two {t2:.3} s ({:.2}x), four {t4:.3} s ({:.2}x), eight {t8:.3} s ({:.2}x)",
            one / t2,
            one / t4,
            one / t8
        );
    }
    let (cy, zc) = cylinder();
    run("cylinder (M 8)", &cy, &zc);
    let (pr, zp) = projected();
    run("projected planted (M 16)", &pr, &zp);
}
