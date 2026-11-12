//! The extra rows read their unknowns as the blocks read fields, with the parts as they are (`Shooting::scalar`).
//!
//! On the planted problem with a second extra row, a weighted sum over the left data (as a phase condition is), at a
//! state whose parts are not normalized (a state in the precision's parts divided by 1000 part by part, its second and
//! last parts overlapping the first), the extra row of the residual, of each coefficient of the residual's jet and of
//! the Jacobian-vector product must equal the weighted sum of the unknowns as `load` reads them (`P::field_get` of the
//! loaded field), bit for bit, in double-, quad- and octo-double.
//!
//! The check can only fail if renormalizing changes the state: the test asserts that `P::new` of the parts differs
//! from the parts as they are on at least one of the row's unknowns. With `scalar` renormalizing (`P::new`), the
//! comparisons fail.
mod common;

use common::{wave, Planted};
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

/// the row's sum over the unknowns as `load` reads them
fn field_read<P: Precision<Native>>(sh: &Shooting<Planted<B>, B>, z: &[&[f64]], row: &ExtraRow) -> P::Scalar {
    P::dot(
        row.terms
            .iter()
            .map(|&(i, w)| (P::from_dd(w), P::field_get(&sh.load::<P>(z, i - i % M), i % M))),
    )
}

fn check<P: Precision<Native>>() {
    let (sh, z) = problem();
    let row = sh.extra[1].clone();
    let r = sh.row_extra() + 1;
    let z0 = widen(&z, P::PARTS);
    let zu = unnormalized::<P>(&z);
    let changed = row.terms.iter().any(|&(i, _)| {
        let p: Vec<f64> = zu.iter().map(|q| q[i]).collect();
        bits::<P>(P::new(&p)) != bits::<P>(P::from_raw(&p))
    });
    assert!(
        changed,
        "{} parts: renormalizing leaves the state unchanged, so the check cannot fail",
        P::PARTS
    );
    let want_u = bits::<P>(field_read::<P>(&sh, &parts(&zu), &row));
    let want_0 = bits::<P>(field_read::<P>(&sh, &parts(&z0), &row));
    let f = sh.residual::<P>(&parts(&zu), false).unwrap();
    assert_eq!(bits::<P>(f[r]), want_u, "{} parts: the residual's extra row", P::PARTS);
    let jet = sh.residual_jet::<P, 4>(&[parts(&z0), parts(&zu)], false, 2).unwrap();
    assert_eq!(bits::<P>(jet[0][r]), want_0, "{} parts: the jet's order 0", P::PARTS);
    assert_eq!(bits::<P>(jet[1][r]), want_u, "{} parts: the jet's order 1", P::PARTS);
    let (jv, _) = sh.jvp::<P>(&parts(&z0), &parts(&zu), false).unwrap();
    assert_eq!(
        bits::<P>(jv[r]),
        want_u,
        "{} parts: the Jacobian-vector product",
        P::PARTS
    );
}

#[test]
fn extra_rows_read_parts_as_they_are_dd() {
    check::<Double>();
}

#[test]
fn extra_rows_read_parts_as_they_are_qd() {
    check::<Quad>();
}

#[test]
fn extra_rows_read_parts_as_they_are_od() {
    check::<Octo>();
}
