//! The fallible preparation (`Equations::try_prepare`, `try_prepare_rate`): the engine forms every evaluation's
//! preparation through it, a failed preparation comes back as every entry point's error, and the default, which calls
//! `prepare`, gives the same bits as before.
mod common;

use common::{perturbed, wave, Bessel, Planted};
use multishoot::lane::{Dtn, Fu, Lane};
use multishoot::prec::{parts, widen};
use multishoot::{
    Double, Equations, Fl, Native, NewtonOpts, Precision, Quad, Rate, Shooting, State, Storage, Work, DD,
};

/// E with a fallible preparation: `try_prepare` and `try_prepare_rate` give E's preparation marked as formed by them,
/// or the error `fail` when it is set. Its infallible `prepare` and `prepare_rate` keep their defaults, an unmarked
/// preparation, with which the endpoint maps give twice their data (the left data in the last NL fields, the right
/// data in the first NR) and zero elsewhere, so an entry point that prepared through them would return other values
/// and no error.
struct Fallible<E> {
    eq: E,
    fail: Option<String>,
}

/// E's preparation and whether `try_prepare` or `try_prepare_rate` formed it
#[derive(Default)]
struct Marked<P> {
    formed: bool,
    prep: P,
}

impl<E: Equations<B>, const B: usize> Equations<B> for Fallible<E> {
    const NF: usize = E::NF;
    const NL: usize = E::NL;
    const NR: usize = E::NR;
    const NG: usize = E::NG;
    const NAUX: usize = E::NAUX;
    const POINTWISE: bool = E::POINTWISE;
    type Prep = Marked<E::Prep>;

    fn try_prepare(&self, zh: &[f64], zl: &[f64]) -> Result<Marked<E::Prep>, String> {
        match &self.fail {
            Some(e) => Err(e.clone()),
            None => Ok(Marked {
                formed: true,
                prep: self.eq.try_prepare(zh, zl)?,
            }),
        }
    }
    fn try_prepare_rate(&self, zh: &[f64], zl: &[f64], rate: Rate) -> Result<Marked<E::Prep>, String> {
        match &self.fail {
            Some(e) => Err(e.clone()),
            None => Ok(Marked {
                formed: true,
                prep: self.eq.try_prepare_rate(zh, zl, rate)?,
            }),
        }
    }
    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        self.eq.taylor_coeffs(x0, g, w)
    }
    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &Self::Prep) -> State<T, B> {
        if prep.formed {
            return self.eq.left_state(x, data, g, &prep.prep);
        }
        let mut s = vec![Fl::zero(); E::NF];
        for (i, d) in data.iter().enumerate() {
            s[E::NF - E::NL + i] = d.mulf(2.0);
        }
        s
    }
    fn right_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &Self::Prep) -> State<T, B> {
        if prep.formed {
            return self.eq.right_state(x, data, g, &prep.prep);
        }
        let mut s = vec![Fl::zero(); E::NF];
        for (i, d) in data.iter().enumerate() {
            s[i] = d.mulf(2.0);
        }
        s
    }
}

/// the error of the failing preparation
const FAIL: &str = "a zero pivot";

/// The problem of sh with a fallible preparation that fails with `fail`, in sh's layout
fn fallible<E: Equations<1>>(sh: Shooting<E, 1>, fail: Option<&str>) -> Shooting<Fallible<E>, 1> {
    let eq = Fallible {
        eq: sh.eq,
        fail: fail.map(String::from),
    };
    let mut f = Shooting::new(eq, sh.nodes, sh.kt);
    f.segs = sh.segs;
    f.extra = sh.extra;
    f.match_node = sh.match_node;
    f.row_scale = sh.row_scale;
    f.projected = sh.projected;
    f
}

/// the planted problem on four nodes, M = 8, and its solution perturbed by 1e-3
fn planted() -> (Shooting<Planted<1>, 1>, Vec<Vec<f64>>) {
    let (sh, zs) = Planted::<1>::shooting(&[0.0, 0.5, 1.0, 1.5], 30, 1e-32);
    (sh, perturbed(&zs, 1e-3))
}

/// Bessel's problem, whose left endpoint map reads its preparation (the Frobenius series), and its solution perturbed
/// by 1e-3
fn bessel() -> (Shooting<Bessel, 1>, Vec<Vec<f64>>) {
    let (sh, zs) = Bessel::shooting(Bessel::new(0.5, 2.0), 30);
    (sh, perturbed(&zs, 1e-3))
}

/// the bits of every part of the scalars
fn bits<P: Precision<Native>>(v: &[P::Scalar]) -> Vec<u64> {
    v.iter()
        .flat_map(|&x| (0..P::PARTS).map(move |q| P::part(x, q).to_bits()))
        .collect()
}

/// the bits of part arrays
fn bits_f64(c: &[Vec<f64>]) -> Vec<u64> {
    c.iter().flatten().map(|x| x.to_bits()).collect()
}

/// Every entry point at z, each by name with the bits of its output or its error: the residual in double- and
/// quad-double, the block states, the Jacobian, the Newton step and loop, the f64 Jacobian, the mixed step (dense),
/// with `staircase` the staircase LU (its solve of a fixed vector) and the mixed loop in it, the Jacobian-vector
/// product, the jet, the mode matrix and its determinant at a complex rate
fn outputs<E: Equations<1>>(
    sh: &Shooting<E, 1>,
    z: &[Vec<f64>],
    staircase: bool,
) -> Vec<(&'static str, Result<Vec<u64>, String>)> {
    let zp = parts(z);
    let zq = widen(z, 4);
    let v: Vec<Vec<f64>> = vec![(0..sh.len()).map(wave).collect(), vec![0.0; sh.len()]];
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 4,
        ..Default::default()
    };
    let rate = Rate::new(DD::from_f64(0.3), DD::from_f64(0.2));
    let mut out: Vec<(&'static str, Result<Vec<u64>, String>)> = vec![
        (
            "residual dd",
            sh.residual::<Double>(&zp, true).map(|f| bits::<Double>(&f)),
        ),
        (
            "residual qd",
            sh.residual::<Quad>(&parts(&zq), true).map(|f| bits::<Quad>(&f)),
        ),
        (
            "block states",
            sh.try_block_states::<Double>(&zp, true).map(|s| {
                s.iter()
                    .flatten()
                    .flat_map(|f| (0..8).flat_map(|j| [f.get(j).hi.to_bits(), f.get(j).lo.to_bits()]))
                    .collect()
            }),
        ),
        ("jacobian", sh.jacobian::<Double>(&zp, true).map(|(c, _)| bits_f64(&c))),
        (
            "newton step",
            sh.newton_step::<Double>(&zp, true)
                .map(|s| [bits_f64(&s.dz), bits::<Double>(&s.f)].concat()),
        ),
        (
            "jacobian f64",
            sh.jacobian_f64::<Fu>(&zp, true).map(|(c, _)| bits_f64(&[c])),
        ),
        (
            "mixed step",
            sh.newton_step_mixed::<Double, Fu>(&zp, Storage::Dense, true)
                .map(|(dz, f, _)| [bits_f64(&dz), bits::<Double>(&f)].concat()),
        ),
        (
            "jvp",
            sh.jvp::<Double>(&zp, &parts(&v), true)
                .map(|(jv, f)| [bits::<Double>(&jv), bits::<Double>(&f)].concat()),
        ),
        (
            "jet",
            sh.residual_jet::<Double, 2>(&[zp.clone(), parts(&v)], true, 2)
                .map(|o| o.iter().flat_map(|f| bits::<Double>(f)).collect()),
        ),
        (
            "mode matrix",
            sh.mode_matrix(&zp, rate, true)
                .map(|mm| [bits_f64(&mm.re), bits_f64(&mm.im)].concat()),
        ),
        (
            "mode logdet",
            sh.mode_logdet(&zp, rate, true)
                .map(|(a, b)| vec![a.to_bits(), b.to_bits()]),
        ),
    ];
    let mut zn = z.to_vec();
    let r = sh.newton::<Double>(&mut zn, &opts);
    out.push(("newton", r.map(|r| [bits_f64(&zn), bits_f64(&[r.fnorm])].concat())));
    if staircase {
        let lu = sh.jacobian_staircase::<Dtn<4>>(&zp, true).map(|lu| {
            let mut b: Vec<DD> = v[0].iter().map(|&x| DD::from_f64(x)).collect();
            lu.solve(&mut b);
            b.iter().flat_map(|x| [x.hi.to_bits(), x.lo.to_bits()]).collect()
        });
        out.push(("staircase", lu));
        let mut zn = z.to_vec();
        let r = sh.newton_mixed::<Double, Fu>(&mut zn, &opts, Storage::Staircase);
        out.push((
            "mixed newton",
            r.map(|r| [bits_f64(&zn), bits_f64(&[r.fnorm])].concat()),
        ));
    }
    out
}

/// the names of the entry points whose outputs differ between a and b, or are errors in either
fn differing(
    a: &[(&'static str, Result<Vec<u64>, String>)],
    b: &[(&'static str, Result<Vec<u64>, String>)],
) -> Vec<&'static str> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.1.is_err() || y.1.is_err() || x.1 != y.1)
        .map(|(x, _)| x.0)
        .collect()
}

/// the names of the entry points that did not return the error FAIL
fn not_failing(out: &[(&'static str, Result<Vec<u64>, String>)]) -> Vec<&'static str> {
    out.iter()
        .filter(|(_, r)| r.as_ref().err().is_none_or(|e| e != FAIL))
        .map(|(n, _)| n)
        .copied()
        .collect()
}

/// The default `try_prepare` is `prepare`'s: pass, every entry point of the planted problem and of Bessel's gives the
/// same bits through a `try_prepare` that returns E's preparation as through E's own, the default that calls
/// `prepare`. The wrapper's infallible `prepare` gives zero endpoint states, so an entry point that prepared by it
/// rather than by `try_prepare` would differ (`controls::infallible_prepare_changes_the_bits`).
#[test]
fn try_prepare_gives_the_bits_of_prepare() {
    let (sh, z) = planted();
    let a = outputs(&sh, &z, true);
    let b = outputs(&fallible(sh, None), &z, true);
    let (sb, zb) = bessel();
    let c = outputs(&sb, &zb, false);
    let d = outputs(&fallible(sb, None), &zb, false);
    let (dp, db) = (differing(&a, &b), differing(&c, &d));
    println!(
        "{} planted and {} Bessel outputs; differing: planted {dp:?}, Bessel {db:?}",
        a.len(),
        c.len()
    );
    assert!(dp.is_empty() && db.is_empty(), "planted {dp:?}, Bessel {db:?}");
}

/// A failed preparation: pass, every entry point of both problems returns the preparation's error, its message
/// unchanged, with no panic.
#[test]
fn failed_preparation_is_every_entry_points_error() {
    let (sh, z) = planted();
    let a = outputs(&fallible(sh, Some(FAIL)), &z, true);
    let (sb, zb) = bessel();
    let b = outputs(&fallible(sb, Some(FAIL)), &zb, false);
    let (np, nb) = (not_failing(&a), not_failing(&b));
    println!(
        "{} planted and {} Bessel entry points; without the error: planted {np:?}, Bessel {nb:?}",
        a.len(),
        b.len()
    );
    assert!(np.is_empty() && nb.is_empty(), "planted {np:?}, Bessel {nb:?}");
}

/// `block_states` has no error to return: it panics with the preparation's message.
#[test]
#[should_panic(expected = "block_states: a zero pivot")]
fn block_states_panics_on_a_failed_preparation() {
    let (sh, z) = planted();
    fallible(sh, Some(FAIL)).block_states::<Double>(&parts(&z), true);
}

/// `try_block_states` checks z before preparing: pass, no part arrays, one part and parts one short are each refused
/// with an error, and the planted z is accepted.
#[test]
fn try_block_states_refuses_a_malformed_z() {
    let (sh, z) = planted();
    let short: Vec<&[f64]> = z.iter().map(|p| &p[..p.len() - 1]).collect();
    let refused = [
        sh.try_block_states::<Double>(&[], true).is_err(),
        sh.try_block_states::<Double>(&parts(&z)[..1], true).is_err(),
        sh.try_block_states::<Double>(&short, true).is_err(),
    ];
    println!("refused: {refused:?}");
    assert_eq!(refused, [true; 3]);
    assert!(sh.try_block_states::<Double>(&parts(&z), true).is_ok());
}

/// The controls (feature `mutate`): every evaluation prepared by the infallible methods.
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::Control;

    /// `failed_preparation_is_every_entry_points_error` fails: every entry point returns values, none an error.
    #[test]
    fn infallible_prepare_hides_the_error() {
        let (sh, z) = planted();
        let mut f = fallible(sh, Some(FAIL));
        f.set_control(Control::InfalliblePrepare);
        let out = outputs(&f, &z, true);
        let erring: Vec<(&str, &String)> = out
            .iter()
            .filter_map(|(n, r)| r.as_ref().err().map(|e| (*n, e)))
            .collect();
        println!(
            "control, infallible preparation: {} entry points, erring {erring:?}",
            out.len()
        );
        assert!(erring.is_empty(), "control did not fire for {erring:?}");
    }

    /// `try_prepare_gives_the_bits_of_prepare` fails: every entry point's bits differ.
    #[test]
    fn infallible_prepare_changes_the_bits() {
        let (sh, z) = planted();
        let a = outputs(&sh, &z, true);
        let mut f = fallible(sh, None);
        f.set_control(Control::InfalliblePrepare);
        let b = outputs(&f, &z, true);
        let same: Vec<&str> = a.iter().zip(&b).filter(|(x, y)| x.1 == y.1).map(|(x, _)| x.0).collect();
        println!(
            "control, infallible preparation: {} entry points, unchanged {same:?}",
            a.len()
        );
        assert!(same.is_empty(), "control did not fire for {same:?}");
    }
}
