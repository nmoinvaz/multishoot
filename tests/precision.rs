//! The residual, the Jacobian and Newton's method in quad-double and octo-double (multidouble's `Precision`) on the
//! planted problem, and the residual's bytes, which must not depend on multidouble's SIMD backend.
mod common;

use common::{perturbed, Fnv, Planted};
use multishoot::prec::{max_abs, parts, widen};
use multishoot::{Double, NewtonOpts, Octo, Prec, Quad, Shooting};

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

fn planted() -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    Planted::<B>::shooting(&NODES, 30, 1e-32)
}

/// Newton's settings: the tolerance on max |F| and the most steps
fn opts(tol: f64, max_iter: usize) -> NewtonOpts {
    NewtonOpts {
        tol,
        max_iter,
        ..Default::default()
    }
}

/// a list of numbers in scientific notation
fn sci(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x:.2e}")).collect::<Vec<_>>().join(" ")
}

/// max |a_i - b_i| over the unknowns of two states in P's parts
fn state_diff<P: Prec>(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let pick = |z: &[Vec<f64>], i: usize| -> P::Scalar {
        let p: Vec<f64> = z.iter().map(|q| q[i]).collect();
        P::new(&p)
    };
    let d: Vec<P::Scalar> = (0..a[0].len()).map(|i| pick(a, i) - pick(b, i)).collect();
    max_abs::<P>(&d)
}

/// Newton in double-double from z* perturbed by 1e-3, then in quad-double from its result, then in octo-double from
/// that: the three reports; the octo-double root must lie within 1e-27 of z*
fn chain(sh: &Shooting<Planted<B>, B>, zs: &[Vec<f64>]) -> [multishoot::NewtonReport; 3] {
    let mut z = perturbed(zs, 1e-3);
    let dd = sh.newton::<Double>(&mut z, &opts(1e-28, 10)).unwrap();
    let mut zq = widen(&z, 4);
    let qd = sh.newton::<Quad>(&mut zq, &opts(1e-60, 4)).unwrap();
    let mut zo = widen(&zq, 8);
    let od = sh.newton::<Octo>(&mut zo, &opts(1e-110, 4)).unwrap();
    println!(
        "max |F|: dd {}, qd {}, od {}; max |z_od - z*| {:.3e}",
        sci(&dd.fnorm),
        sci(&qd.fnorm),
        sci(&od.fnorm),
        state_diff::<Octo>(&zo, &widen(zs, 8))
    );
    assert!(
        state_diff::<Octo>(&zo, &widen(zs, 8)) < 1e-27,
        "the octo-double root is not z*"
    );
    [dd, qd, od]
}

/// The planted problem's Newton carried from double-double to quad-double and octo-double, each starting from the
/// last one's result, at most 4 steps each. Pass: max |F| below 1e-60 in quad-double and below 1e-110 in
/// octo-double, and the octo-double root within 1e-27 of z* (the manufactured solution solves the discrete system
/// only to its double-double sources' rounding).
#[test]
fn planted_converges_in_quad_and_octo() {
    let (sh, zs) = planted();
    let [dd, qd, od] = chain(&sh, &zs);
    assert!(dd.converged, "double-double: {:?}", dd.fnorm);
    assert!(
        qd.converged && *qd.fnorm.last().unwrap() < 1e-60,
        "quad-double: {:?}",
        qd.fnorm
    );
    assert!(
        od.converged && *od.fnorm.last().unwrap() < 1e-110,
        "octo-double: {:?}",
        od.fnorm
    );
}

/// The three precisions evaluate one discrete system: at a double-double z (z* perturbed by 1e-3), the quad- and
/// octo-double residuals agree to 1e-60 and the double-double one with them to 1e-29.
#[test]
fn precisions_agree_on_the_residual() {
    let (sh, zs) = planted();
    let z = perturbed(&zs, 1e-3);
    let fd = sh.residual::<Double>(&parts(&z), true).unwrap();
    let fq = sh.residual::<Quad>(&parts(&widen(&z, 4)), true).unwrap();
    let fo = sh.residual::<Octo>(&parts(&widen(&z, 8)), true).unwrap();
    // the quad-double values exactly in octo-double, from their four parts
    let qo: Vec<_> = fq
        .iter()
        .zip(&fo)
        .map(|(q, o)| <Octo as Prec>::new(&q.x) - *o)
        .collect();
    let qo = max_abs::<Octo>(&qo);
    let dq = fd
        .iter()
        .zip(&fq)
        .map(|(d, q)| (*d - q.to_dd()).hi.abs())
        .fold(0.0, f64::max);
    println!("|F_qd - F_od| = {qo:.3e}, |F_dd - F_qd| = {dq:.3e}");
    assert!(qo < 1e-60 && dq < 1e-29, "{qo:e} {dq:e}");
}

/// FNV-1a over the bits of every part of the scalars
fn fnv<P: Prec>(f: &[P::Scalar]) -> u64 {
    let mut h = Fnv::new();
    for &v in f {
        for q in 0..P::PARTS {
            h.f64(P::part(v, q));
        }
    }
    h.0
}

/// A state with every part in use: z* perturbed, times 1 + 1e-3 / (3 + i) in octo-double
fn full_parts_z(zs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let z = perturbed(zs, 1e-3);
    let l = z[0].len();
    let mut out: Vec<Vec<f64>> = (0..8).map(|_| vec![0.0; l]).collect();
    for i in 0..l {
        let x = <Octo as Prec>::new(&[z[0][i]]);
        let r = <Octo as Prec>::ONE + <Octo as Prec>::new(&[1e-3]) * <Octo as Prec>::new(&[1.0 / (3.0 + i as f64)]);
        let v = x * r * r;
        for (q, o) in out.iter_mut().enumerate() {
            o[i] = <Octo as Prec>::part(v, q);
        }
    }
    out
}

/// The hashes of the residual in double-, quad- and octo-double at `full_parts_z` (each precision reading its first
/// PARTS parts), and of the quad-double Newton step there
fn hashes(sh: &Shooting<Planted<B>, B>, z8: &[Vec<f64>]) -> [u64; 4] {
    let p = parts(z8);
    let fd = sh.residual::<Double>(&p[..2], true).unwrap();
    let fq = sh.residual::<Quad>(&p[..4], true).unwrap();
    let fo = sh.residual::<Octo>(&p, true).unwrap();
    let st = sh.newton_step::<Quad>(&p[..4], true).unwrap();
    let dz: Vec<_> = (0..sh.len())
        .map(|i| <Quad as Prec>::from_raw(&[st.dz[0][i], st.dz[1][i], st.dz[2][i], st.dz[3][i]]))
        .collect();
    [fnv::<Double>(&fd), fnv::<Quad>(&fq), fnv::<Octo>(&fo), fnv::<Quad>(&dz)]
}

/// The reference hashes, recorded on aarch64 (NEON) and checked on every backend of the CI matrix
const REFERENCE: [u64; 4] = [
    0x644aab6b8168def1,
    0xcc72caf157ee5c3a,
    0x6eee158d2c9606a3,
    0xc033077e35583804,
];

/// The residual's bytes in every precision, and the quad-double Newton step's, at a fixed state with every part in
/// use. Pass: the FNV-1a hashes of their bits equal the reference, on every SIMD backend (native, portable, AVX2,
/// NEON) and with any number of threads.
#[test]
fn residual_bytes_match_reference() {
    let (sh, zs) = planted();
    let h = hashes(&sh, &full_parts_z(&zs));
    println!(
        "hashes [{}]",
        h.iter().map(|v| format!("{v:#018x}")).collect::<Vec<_>>().join(", ")
    );
    assert_eq!(h, REFERENCE, "residual bytes differ from the reference");
}

/// The control of `residual_bytes_match_reference`: part 0, 1 or 3 of one unknown changed by one ulp must change the
/// hash of every precision that reads that part. (One ulp of part 7 is 2^-424 relative, below octo-double's rounding,
/// and need not reach the result.)
#[test]
fn residual_bytes_control_one_ulp_differs() {
    let (sh, zs) = planted();
    let z = full_parts_z(&zs);
    let base = hashes(&sh, &z);
    let i = sh.off_node(1) + 3;
    for q in [0usize, 1, 3] {
        let mut zm = z.clone();
        zm[q][i] = f64::from_bits(zm[q][i].to_bits() + 1);
        let h = hashes(&sh, &zm);
        // a change in part q reaches the precisions that read it
        for (k, parts) in [2usize, 4, 8, 4].iter().enumerate() {
            if q < *parts {
                assert_ne!(h[k], base[k], "hash {k} unchanged by part {q}");
            }
        }
    }
}

/// The controls (feature `mutate`): the state read in half its parts must keep the quad-double residual above 1e-60
/// and the octo-double one above 1e-110.
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::Control;

    #[test]
    fn half_parts_stall_above_the_floors() {
        let (mut sh, zs) = planted();
        let mut z = perturbed(&zs, 1e-3);
        sh.newton::<Double>(&mut z, &opts(1e-28, 10)).unwrap();
        sh.set_control(Control::HalfParts);
        let mut zq = widen(&z, 4);
        let qd = sh.newton::<Quad>(&mut zq, &opts(1e-60, 4)).unwrap();
        sh.set_control(Control::None);
        let mut zq = widen(&z, 4);
        sh.newton::<Quad>(&mut zq, &opts(1e-60, 4)).unwrap();
        sh.set_control(Control::HalfParts);
        let mut zo = widen(&zq, 8);
        let od = sh.newton::<Octo>(&mut zo, &opts(1e-110, 4)).unwrap();
        println!("control, half parts: qd {}, od {}", sci(&qd.fnorm), sci(&od.fnorm));
        assert!(!qd.converged && !od.converged, "control did not fire");
    }
}
