//! The extra rows read their unknowns as the blocks read fields, with the parts as they are (`Shooting::scalar`).
//!
//! On the planted problem with a second extra row, a weighted sum over the left data (as a phase condition is), at two
//! states whose parts are not normalized (states in the precision's parts divided by 1000 part by part, their second
//! and last parts overlapping the first), four comparisons are made bit for bit in double-, quad- and octo-double: the
//! extra row of the residual at the first state, of the jet's orders 0 and 1 at the two states, and of the
//! Jacobian-vector product along the second, each against the weighted sum of the unknowns as `load` reads them
//! (`P::field_get` of the loaded field).
//!
//! Controls: the same comparisons against the sum of the unknowns renormalized by `P::new`, the read `scalar` had
//! before, must all fail; and with the feature `mutate`, `Control::RenormScalar` puts that read back in the engine,
//! and the comparisons against the fields' read must all fail.
mod common;

use common::{perturbed, wave, Planted};
use multishoot::prec::{parts, widen};
use multishoot::{Double, ExtraRow, Native, Octo, Precision, Quad, Shooting, DD};

const B: usize = 1;
const M: usize = 8 * B;
const NODES: [f64; 3] = [0.0, 0.5, 1.0];

/// the planted problem with the second extra row sum_j w_j z_j over the left data, w_j = 1 / (j + 3)
fn problem() -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    let (mut sh, z) = Planted::<B>::shooting(&NODES, 20, 1e-30);
    sh.extra.push(ExtraRow {
        terms: (0..M).map(|j| (j, DD::ONE / DD::from_f64(j as f64 + 3.0))).collect(),
        rhs: DD::ZERO,
    });
    (sh, z)
}

/// z in P's parts with every part in use (the lower ones small multiples of the leading one), each part divided by
/// 1000, and the second and last parts moved into the first one's bits (2^-40 and 2^-30 of it), which leaves the
/// parts unnormalized; with the last part's overlap octo-double's renormalization changes the rows too
fn unnormalized<P: Precision<Native>>(z: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut w = widen(z, P::PARTS);
    for i in 0..w[0].len() {
        for q in 2..P::PARTS {
            w[q][i] = w[0][i] * wave(i + 17 * q) * 2f64.powi(-53 * q as i32);
        }
        for p in w.iter_mut() {
            p[i] /= 1000.0;
        }
        w[1][i] += w[0][i] * wave(i + 5) * 2f64.powi(-40);
        w[P::PARTS - 1][i] += w[0][i] * wave(i + 3) * 2f64.powi(-30);
    }
    w
}

/// the bits of every part of a scalar
fn bits<P: Precision<Native>>(x: P::Scalar) -> Vec<u64> {
    (0..P::PARTS).map(|q| P::part(x, q).to_bits()).collect()
}

/// How the expected row reads its unknowns
#[derive(Clone, Copy)]
enum Read {
    /// as `load` reads a field, the parts as they are
    AsLoaded,
    /// renormalized by `P::new`, the control's wrong result
    Renormalized,
}

/// the bits of the row's sum over its unknowns, read as `read` says
fn expected<P: Precision<Native>>(sh: &Shooting<Planted<B>, B>, z: &[&[f64]], row: &ExtraRow, read: Read) -> Vec<u64> {
    bits::<P>(P::dot(row.terms.iter().map(|&(i, w)| {
        let x = match read {
            Read::AsLoaded => P::field_get(&sh.load::<P>(z, i - i % M), i % M),
            Read::Renormalized => P::new(&z.iter().map(|p| p[i]).collect::<Vec<f64>>()),
        };
        (P::from_dd(w), x)
    })))
}

/// Whether the residual's extra row at zu, the jet's orders 0 and 1 at (zu, zv) and the Jacobian-vector product along
/// zv equal the row's sum over their unknowns read as `read` says, bit for bit: one verdict per comparison.
fn comparisons<P: Precision<Native>>(sh: &Shooting<Planted<B>, B>, z: &[Vec<f64>], read: Read) -> [bool; 4] {
    let row = sh.extra[1].clone();
    let r = sh.row_extra() + 1;
    let (zu, zv) = (unnormalized::<P>(z), unnormalized::<P>(&perturbed(z, 1e-1)));
    let (u, v) = (parts(&zu), parts(&zv));
    let (wu, wv) = (expected::<P>(sh, &u, &row, read), expected::<P>(sh, &v, &row, read));
    let f = sh.residual::<P>(&u, false).unwrap();
    let jet = sh.residual_jet::<P, 4>(&[u.clone(), v.clone()], false, 2).unwrap();
    let (jv, _) = sh.jvp::<P>(&u, &v, false).unwrap();
    [
        bits::<P>(f[r]) == wu,
        bits::<P>(jet[0][r]) == wu,
        bits::<P>(jet[1][r]) == wv,
        bits::<P>(jv[r]) == wv,
    ]
}

/// Pass: all four equal the sum of the unknowns as the fields read them.
fn pass<P: Precision<Native>>() {
    let (sh, z) = problem();
    assert_eq!(
        comparisons::<P>(&sh, &z, Read::AsLoaded),
        [true; 4],
        "{} parts",
        P::PARTS
    );
}

/// The control of the comparisons: against the renormalized sum, each of the four must fail.
fn control<P: Precision<Native>>() {
    let (sh, z) = problem();
    assert_eq!(
        comparisons::<P>(&sh, &z, Read::Renormalized),
        [false; 4],
        "{} parts",
        P::PARTS
    );
}

#[test]
fn extra_rows_read_parts_as_they_are_dd() {
    pass::<Double>();
}

#[test]
fn extra_rows_read_parts_as_they_are_qd() {
    pass::<Quad>();
}

#[test]
fn extra_rows_read_parts_as_they_are_od() {
    pass::<Octo>();
}

#[test]
fn extra_rows_control_renormalized_sum_fails_dd() {
    control::<Double>();
}

#[test]
fn extra_rows_control_renormalized_sum_fails_qd() {
    control::<Quad>();
}

#[test]
fn extra_rows_control_renormalized_sum_fails_od() {
    control::<Octo>();
}

#[cfg(feature = "mutate")]
mod mutate {
    use super::*;
    use multishoot::control::Control;

    /// The engine's control: with `scalar` renormalizing, each of the four must fail against the fields' read.
    fn renorm_scalar<P: Precision<Native>>() {
        let (mut sh, z) = problem();
        sh.set_control(Control::RenormScalar);
        assert_eq!(
            comparisons::<P>(&sh, &z, Read::AsLoaded),
            [false; 4],
            "{} parts",
            P::PARTS
        );
    }

    #[test]
    fn extra_rows_control_renorm_scalar_fails_dd() {
        renorm_scalar::<Double>();
    }

    #[test]
    fn extra_rows_control_renorm_scalar_fails_qd() {
        renorm_scalar::<Quad>();
    }

    #[test]
    fn extra_rows_control_renorm_scalar_fails_od() {
        renorm_scalar::<Octo>();
    }
}
