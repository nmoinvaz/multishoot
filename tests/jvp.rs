//! The Jacobian-vector product on the planted problem, in double-, quad- and octo-double, against the exact Jacobian
//! times the vector, and its bytes, which must not depend on multidouble's SIMD backend.
mod common;

use common::{perturbed, wave, Fnv, Planted};
use multishoot::prec::{nan_max, parts};
use multishoot::{Double, Native, Octo, Precision, Quad, Shooting};

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

fn planted() -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    Planted::<B>::shooting(&NODES, 30, 1e-32)
}

/// The octo-double parts of x_i (1 + 1e-3 / (3 + i))^2, every part in use
fn full_parts(x: &[f64]) -> Vec<Vec<f64>> {
    type O = Octo;
    let mut out: Vec<Vec<f64>> = (0..8).map(|_| vec![0.0; x.len()]).collect();
    for (i, &xi) in x.iter().enumerate() {
        let r = <O as Precision<Native>>::ONE
            + <O as Precision<Native>>::new(&[1e-3]) * <O as Precision<Native>>::new(&[1.0 / (3.0 + i as f64)]);
        let v = <O as Precision<Native>>::new(&[xi]) * r * r;
        for (q, o) in out.iter_mut().enumerate() {
            o[i] = <O as Precision<Native>>::part(v, q);
        }
    }
    out
}

/// The state (z* perturbed by 1e-3) and the direction (`wave`, every unknown, the global included), both with
/// every octo-double part in use; a precision reads their first PARTS parts
fn state_and_direction(sh: &Shooting<Planted<B>, B>, zs: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let z = full_parts(&perturbed(zs, 1e-3)[0]);
    let v = full_parts(&(0..sh.len()).map(|i| wave(i + 7)).collect::<Vec<_>>());
    (z, v)
}

/// max_i |(J v)_i - jvp_i| over the rows, relative to max_i sum_j |J_ij| |v_j|, with J the given Jacobian in P
/// (part arrays, ld) and J v its product with v in P's accumulator
fn relative_error<P: Precision<Native>>(
    sh: &Shooting<Planted<B>, B>,
    z: &[&[f64]],
    v: &[&[f64]],
    (c, ld): &(Vec<Vec<f64>>, usize),
) -> f64 {
    let l = sh.len();
    let (jv, _) = sh.jvp::<P>(z, v, true).unwrap();
    let entry = |i: usize, j: usize| P::from_raw(&c.iter().map(|cq| cq[j * ld + i]).collect::<Vec<_>>());
    let vj: Vec<P::Scalar> = (0..l)
        .map(|j| P::from_raw(&v.iter().map(|q| q[j]).collect::<Vec<_>>()))
        .collect();
    let (mut scale, mut err) = (0.0f64, 0.0f64);
    for (i, &ji) in jv.iter().enumerate() {
        let prod = P::dot((0..l).map(|j| (entry(i, j), vj[j])));
        scale = nan_max(
            scale,
            (0..l).map(|j| P::part(entry(i, j), 0).abs() * v[0][j].abs()).sum(),
        );
        err = nan_max(err, P::to_dd(prod - ji).hi.abs());
    }
    err / scale
}

/// `relative_error` against the exact Jacobian of sh at z, and whether jvp's residual equals `residual` in every part
fn jvp_against_jacobian<P: Precision<Native>>(sh: &Shooting<Planted<B>, B>, z: &[&[f64]], v: &[&[f64]]) -> (f64, bool) {
    let e = relative_error::<P>(sh, z, v, &sh.jacobian::<P>(z, true).unwrap());
    let (_, f) = sh.jvp::<P>(z, v, true).unwrap();
    let fr = sh.residual::<P>(z, true).unwrap();
    let same = (0..sh.rows()).all(|i| (0..P::PARTS).all(|q| P::part(f[i], q) == P::part(fr[i], q)));
    (e, same)
}

/// the three precisions' relative errors of `jvp_against_jacobian` and whether each residual matched
fn all_precisions(sh: &Shooting<Planted<B>, B>, z8: &[Vec<f64>], v8: &[Vec<f64>]) -> [(f64, bool); 3] {
    let (z, v) = (parts(z8), parts(v8));
    [
        jvp_against_jacobian::<Double>(sh, &z[..2], &v[..2]),
        jvp_against_jacobian::<Quad>(sh, &z[..4], &v[..4]),
        jvp_against_jacobian::<Octo>(sh, &z, &v),
    ]
}

/// the pass bounds of `jvp_matches_the_jacobian`, relative: double-, quad- and octo-double
const BOUNDS: [f64; 3] = [1e-28, 1e-58, 1e-115];

/// The product from one pass per block against the exact Jacobian times v, in each precision, at a state and a
/// direction with every part in use (the global's entry included). Both differentiate the same arithmetic, so they
/// differ by rounding only, accumulated along the Taylor paths. Pass: max |J v - jvp| below 1e-28, 1e-58 and 1e-115
/// of max_i sum_j |J_ij| |v_j| in double-, quad- and octo-double; and the residual the product comes with equal to
/// `residual`'s in every part.
#[test]
fn jvp_matches_the_jacobian() {
    let (sh, zs) = planted();
    let (z, v) = state_and_direction(&sh, &zs);
    let r = all_precisions(&sh, &z, &v);
    println!(
        "|J v - jvp| / scale: dd {:.3e}, qd {:.3e}, od {:.3e}; residual equal {:?}",
        r[0].0,
        r[1].0,
        r[2].0,
        r.map(|x| x.1)
    );
    for (k, &(e, same)) in r.iter().enumerate() {
        assert!(e < BOUNDS[k], "precision {k}: {e:e}");
        assert!(same, "precision {k}: jvp's residual differs from residual's");
    }
}

/// The control of the residual comparison in `jvp_matches_the_jacobian`: the residual at z with one unknown one
/// ulp away must differ from jvp's residual at z, in every precision.
#[test]
fn jvp_residual_control_one_ulp_differs() {
    let (sh, zs) = planted();
    let (z, v) = state_and_direction(&sh, &zs);
    let mut zu = z.clone();
    let i = sh.off_node(2) + 3;
    zu[0][i] = f64::from_bits(zu[0][i].to_bits() + 1);
    fn differs<P: Precision<Native>>(sh: &Shooting<Planted<B>, B>, z: &[&[f64]], zu: &[&[f64]], v: &[&[f64]]) -> bool {
        let (_, f) = sh.jvp::<P>(z, v, true).unwrap();
        let fu = sh.residual::<P>(zu, true).unwrap();
        (0..sh.rows()).any(|r| (0..P::PARTS).any(|q| P::part(f[r], q) != P::part(fu[r], q)))
    }
    let (z, zu, v) = (parts(&z), parts(&zu), parts(&v));
    assert!(
        differs::<Double>(&sh, &z[..2], &zu[..2], &v[..2]),
        "control did not fire in dd"
    );
    assert!(
        differs::<Quad>(&sh, &z[..4], &zu[..4], &v[..4]),
        "control did not fire in qd"
    );
    assert!(differs::<Octo>(&sh, &z, &zu, &v), "control did not fire in od");
}

/// FNV-1a over the bits of every part of the scalars
fn fnv<P: Precision<Native>>(f: &[P::Scalar]) -> u64 {
    let mut h = Fnv::new();
    for &x in f {
        for q in 0..P::PARTS {
            h.f64(P::part(x, q));
        }
    }
    h.0
}

/// The hashes of J v in double-, quad- and octo-double at the state and direction of `state_and_direction`, the
/// blocks on rayon tasks with `par` or one after another without
fn hashes(sh: &Shooting<Planted<B>, B>, z8: &[Vec<f64>], v8: &[Vec<f64>], par: bool) -> [u64; 3] {
    let (z, v) = (parts(z8), parts(v8));
    [
        fnv::<Double>(&sh.jvp::<Double>(&z[..2], &v[..2], par).unwrap().0),
        fnv::<Quad>(&sh.jvp::<Quad>(&z[..4], &v[..4], par).unwrap().0),
        fnv::<Octo>(&sh.jvp::<Octo>(&z, &v, par).unwrap().0),
    ]
}

/// The reference hashes, recorded on aarch64 (NEON) and on the portable backend
const REFERENCE: [u64; 3] = [0x228f16c859d1cca4, 0x93abd5378a01aca3, 0x09994890e025f5e3];

/// The product's bytes in every precision. Pass: the FNV-1a hashes of their bits equal the reference, on every SIMD
/// backend of the CI matrix, with any number of threads, and with the blocks in parallel or in sequence.
#[test]
fn jvp_bytes_match_reference() {
    let (sh, zs) = planted();
    let (z, v) = state_and_direction(&sh, &zs);
    let h = hashes(&sh, &z, &v, true);
    println!(
        "jvp hashes [{}]",
        h.iter().map(|x| format!("{x:#018x}")).collect::<Vec<_>>().join(", ")
    );
    assert_eq!(h, REFERENCE, "the product's bytes differ from the reference");
    assert_eq!(
        hashes(&sh, &z, &v, false),
        REFERENCE,
        "the sequential product's bytes differ"
    );
}

/// The control of `jvp_bytes_match_reference`: one ulp of the direction's part 0 at one unknown must change every
/// precision's hash.
#[test]
fn jvp_bytes_control_one_ulp_differs() {
    let (sh, zs) = planted();
    let (z, v) = state_and_direction(&sh, &zs);
    let base = hashes(&sh, &z, &v, true);
    let mut vu = v.clone();
    let i = sh.off_node(1) + 5;
    vu[0][i] = f64::from_bits(vu[0][i].to_bits() + 1);
    let h = hashes(&sh, &z, &vu, true);
    for k in 0..3 {
        assert_ne!(h[k], base[k], "hash {k} unchanged");
    }
}

/// The controls (feature `mutate`): the product without one column's block derivative, a node unknown's or the
/// global's, must fail `jvp_matches_the_jacobian`'s bound in every precision.
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::Control;

    fn dropped(col: impl Fn(&Shooting<Planted<B>, B>) -> usize, what: &str) {
        let (mut sh, zs) = planted();
        let (z, v) = state_and_direction(&sh, &zs);
        let (z, v) = (parts(&z), parts(&v));
        // the exact Jacobians, before the control, which is for the product alone
        let jd = sh.jacobian::<Double>(&z[..2], true).unwrap();
        let jq = sh.jacobian::<Quad>(&z[..4], true).unwrap();
        let jo = sh.jacobian::<Octo>(&z, true).unwrap();
        let c = col(&sh);
        sh.set_control(Control::DropColumn(c));
        let r = [
            relative_error::<Double>(&sh, &z[..2], &v[..2], &jd),
            relative_error::<Quad>(&sh, &z[..4], &v[..4], &jq),
            relative_error::<Octo>(&sh, &z, &v, &jo),
        ];
        println!(
            "control, {what} (column {c}) dropped: dd {:.3e}, qd {:.3e}, od {:.3e}",
            r[0], r[1], r[2]
        );
        for (k, &e) in r.iter().enumerate() {
            assert!(e >= BOUNDS[k], "control did not fire in precision {k}: {e:e}");
        }
    }

    #[test]
    fn dropped_node_column_fails() {
        dropped(|sh| sh.off_node(2) + 5, "a node unknown");
    }

    #[test]
    fn dropped_global_column_fails() {
        dropped(|sh| sh.off_global(), "the global");
    }
}
