//! Scaled rows: the residual, the Jacobian, the jet and the Jacobian-vector product with `row_scale` are the unscaled
//! ones times the factors.
mod common;

use common::{in_pool, perturbed, Fnv, Planted};
use multishoot::prec::{parts, widen};
use multishoot::{Double, Quad, Rate, Shooting, DD};

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

/// the planted problem with block 3's rows scaled by s (power of two), and z* perturbed by 1e-3
fn scaled(s: f64) -> (Shooting<Planted<B>, B>, Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    let (plain, zs) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let (mut sc, _) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let (r0, r1) = (sc.row_block(3), sc.row_block(4));
    sc.row_scale = (0..sc.rows())
        .map(|r| if (r0..r1).contains(&r) { s } else { 1.0 })
        .collect();
    (plain, sc, perturbed(&zs, 1e-3))
}

/// Whether the residual, every Jacobian entry, both orders of the jet and the product J v with its residual of `sc`
/// equal `plain`'s times `t` on the scaled rows and times 1 on the others, bit for bit: one verdict per comparison.
fn comparisons(plain: &Shooting<Planted<B>, B>, sc: &Shooting<Planted<B>, B>, z: &[Vec<f64>], t: f64) -> [bool; 4] {
    let p = parts(z);
    let f = |r: usize| {
        if sc.row_scale[r] == 1.0 {
            DD::ONE
        } else {
            DD::from_f64(t)
        }
    };
    let (f0, f1) = (
        plain.residual::<Double>(&p, true).unwrap(),
        sc.residual::<Double>(&p, true).unwrap(),
    );
    let res = (0..f0.len()).all(|r| f0[r] * f(r) == f1[r]);
    let ((c0, ld), (c1, _)) = (
        plain.jacobian::<Double>(&p, true).unwrap(),
        sc.jacobian::<Double>(&p, true).unwrap(),
    );
    let l = plain.len();
    let jac = (0..l).all(|j| {
        (0..l).all(|r| {
            DD::new(c0[0][j * ld + r], c0[1][j * ld + r]) * f(r) == DD::new(c1[0][j * ld + r], c1[1][j * ld + r])
        })
    });
    let v = perturbed(z, 1e-1);
    let (j0, j1) = (
        plain
            .residual_jet::<Double, 2>(&[p.clone(), parts(&v)], true, 2)
            .unwrap(),
        sc.residual_jet::<Double, 2>(&[p.clone(), parts(&v)], true, 2).unwrap(),
    );
    let jet = (0..2).all(|k| (0..l).all(|r| j0[k][r] * f(r) == j1[k][r]));
    let ((p0, g0), (p1, g1)) = (
        plain.jvp::<Double>(&p, &parts(&v), true).unwrap(),
        sc.jvp::<Double>(&p, &parts(&v), true).unwrap(),
    );
    let jvp = (0..l).all(|r| p0[r] * f(r) == p1[r] && g0[r] * f(r) == g1[r]);
    [res, jac, jet, jvp]
}

/// Pass: with block 3 scaled by 2^100 the residual, every Jacobian entry, both orders of the jet and the product J v
/// equal the unscaled ones times the factor, bit for bit (a power of two scales exactly), and the other rows are
/// unchanged.
#[test]
fn row_scale_scales_residual_jacobian_and_jet() {
    let s = 2f64.powi(100);
    let (plain, sc, z) = scaled(s);
    assert_eq!(comparisons(&plain, &sc, &z, s), [true, true, true, true]);
}

/// The control of the comparisons: told the factor is 2^101 where it is 2^100, each of the four must fail.
#[test]
fn row_scale_control_wrong_factor_fails() {
    let s = 2f64.powi(100);
    let (plain, sc, z) = scaled(s);
    assert_eq!(comparisons(&plain, &sc, &z, 2.0 * s), [false, false, false, false]);
}

/// The control: a factor list of the wrong length must be refused, not applied.
#[test]
fn row_scale_control_wrong_length_fails() {
    let (_, mut sc, z) = scaled(2.0);
    sc.row_scale.pop();
    assert!(sc.residual::<Double>(&parts(&z), true).is_err() && sc.jacobian::<Double>(&parts(&z), true).is_err());
    assert!(sc.jvp::<Double>(&parts(&z), &parts(&z), true).is_err());
}

/// The hashes of the planted problem's Jacobian in quad-double and of its mode matrix at the rate 0.3 + 0.2 i, at z*
/// perturbed by 1e-3, with every row scaled by a factor that rounds (1.5, 0.7, 3 and 1.1 in turn, starting `rot` rows
/// on), with `par`
fn scaled_hashes(rot: usize, par: bool) -> [u64; 2] {
    let (mut sh, zs) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    sh.row_scale = (0..sh.rows()).map(|r| [1.5, 0.7, 3.0, 1.1][(r + rot) % 4]).collect();
    let z = perturbed(&zs, 1e-3);
    let mut hj = Fnv::new();
    let (c, _) = sh.jacobian::<Quad>(&parts(&widen(&z, 4)), par).unwrap();
    c.iter().flatten().for_each(|&x| hj.f64(x));
    let mut hm = Fnv::new();
    let mm = sh
        .mode_matrix(&parts(&z), Rate::new(DD::from_f64(0.3), DD::from_f64(0.2)), par)
        .unwrap();
    mm.re.iter().chain(&mm.im).flatten().for_each(|&x| hm.f64(x));
    [hj.0, hm.0]
}

/// the reference hashes of `scaled_hashes`, recorded with the rows scaled in a serial loop on aarch64 (NEON)
const SCALED_REFERENCE: [u64; 2] = [0x077c6dd5fddf8816, 0x449d6c551020ab51];

/// Rows scaled by factors that round keep their bits however the columns are scaled: pass, `scaled_hashes` with
/// `par` in pools of 1, 3 and 6 threads, and without it, equals the reference of the serial loop. The control: the
/// factors one row on change both hashes.
#[test]
fn scaled_rows_do_not_depend_on_threads() {
    for (t, par) in [(1, true), (3, true), (6, true), (3, false)] {
        let h = in_pool(t, || scaled_hashes(0, par));
        println!(
            "scaled hashes, {t} threads, par {par}: [{:#018x}, {:#018x}]",
            h[0], h[1]
        );
        assert_eq!(h, SCALED_REFERENCE, "{t} threads, par {par}");
    }
    let c = in_pool(3, || scaled_hashes(1, true));
    assert!(
        c[0] != SCALED_REFERENCE[0] && c[1] != SCALED_REFERENCE[1],
        "control did not fire"
    );
}
