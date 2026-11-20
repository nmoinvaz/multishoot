//! Sub-blocks of the Jacobian and of the residual (`Shooting::jacobian_sub`, `Shooting::residual_sub`) against the
//! whole ones, entry for entry: the planted problem, the planted problem with inward segments, projected rows and a
//! row scaling, the planted problem without its theta coupling declared pointwise (`Seed::Points`), the eigenproblem
//! `Cylinder` and Bessel's equation (pointwise, with a preparation), at states perturbed off their solutions with
//! nonzero lower parts, in double-, quad- and octo-double.
mod common;

use common::{as_pointwise, perturbed, wave, Bessel, Cylinder, Planted};
use multishoot::lane::Lane;
use multishoot::periodic::Parity;
use multishoot::{Double, Equations, Fl, Native, Octo, Plan, Precision, Quad, Shooting, State, Work};
use multisolver::prec::parts;
use std::sync::atomic::{AtomicUsize, Ordering};

const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

/// z perturbed by 1e-3 in its leading part, in `n` parts with nonzero lower ones: part q > 0 of unknown i is the
/// leading part times 2^(-56 q) (0.5 + 0.25 wave), a normalized expansion
fn deep(z: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let hi = perturbed(z, 1e-3).swap_remove(0);
    (0..n)
        .map(|q| {
            hi.iter()
                .enumerate()
                .map(|(i, &x)| {
                    if q == 0 {
                        x
                    } else {
                        x * 2f64.powi(-56 * q as i32) * (0.5 + 0.25 * wave(31 * i + q))
                    }
                })
                .collect()
        })
        .collect()
}

/// The planted problem with segments 3 and 4 inward (match_node 2, planned so), u's rows of the right block
/// projected (pinning the right data, the block's own input) and v's of segment block 2 (pinning the left data, which
/// the block does not read), and its rows scaled by factors of which some round (3, 0.1)
fn decorated() -> (Shooting<Planted<2>, 2>, Vec<Vec<f64>>) {
    let (mut sh, z) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    sh.match_node = 2;
    sh.plan_segments(
        &z[0],
        &z[1],
        &Plan {
            kt: 30,
            tol: 1e-32,
            hmax: 0.5,
        },
    );
    let (k, pin) = (sh.k(), sh.off_right());
    sh.project_rows(k, 0, pin, Parity::Odd, 3).unwrap();
    sh.project_rows(2, 1, 0, Parity::Even, 2).unwrap();
    sh.row_scale = (0..sh.rows())
        .map(|r| [1.0, 2f64.powi(-40), 3.0, 0.1, 2f64.powi(70)][r % 5])
        .collect();
    (sh, z)
}

/// The planted problem without its theta coupling, declared pointwise
fn pointwise() -> (Shooting<common::Pointwise<Planted<2>>, 2>, Vec<Vec<f64>>) {
    let (sh, z) = Planted::<2>::shooting_of(Planted::with_coupling(false), &NODES, 30, 1e-32);
    (as_pointwise(sh), z)
}

type Case = (String, Vec<usize>, Vec<usize>);

/// The (name, rows, columns) cases of a layout: segment block 2's rows by its columns (its inputs, its target and the
/// globals); rows and columns scattered over the blocks and the last row, in no order and with repeats, among them
/// the left block's row of field 1 at point 0 and column 0; the extra rows and one segment row by every column; every
/// row but the left block's by the globals' columns; each projected field's rows by its block's columns and the
/// pinned unknowns'; and the whole matrix.
fn cases<E: Equations<B>, const B: usize>(sh: &Shooting<E, B>) -> Vec<Case> {
    let m = Shooting::<E, B>::M;
    let (nb, k, l, og) = (E::NF * m, sh.k(), sh.len(), sh.off_global());
    let block_cols = |b: usize| -> Vec<usize> {
        let mut c: Vec<usize> = sh.block_inputs(b).iter().flat_map(|&o| o..o + m).collect();
        c.extend(sh.block_target(b)..sh.block_target(b) + nb);
        c.extend(og..l);
        c
    };
    let mut out: Vec<Case> = vec![
        (
            "segment block 2".into(),
            (sh.row_block(2)..sh.row_block(3)).collect(),
            block_cols(2),
        ),
        (
            "scattered".into(),
            vec![
                sh.row_block(k) + 3,
                m,
                sh.row_block(1) + nb - 1,
                sh.row_block(3) + m + 2,
                m,
                l - 1,
                sh.row_block(2),
            ],
            vec![
                l - 1,
                sh.off_node(1) + 3,
                0,
                sh.off_right() + 1,
                sh.off_node(k - 1) + m + 7,
                sh.off_node(2) + 2,
                0,
                sh.off_node(3) + 1,
            ],
        ),
        (
            "extra rows".into(),
            std::iter::once(sh.row_block(1) + 1).chain(sh.row_extra()..l).collect(),
            (0..l).collect(),
        ),
    ];
    if E::NG > 0 {
        out.push((
            "globals".into(),
            (sh.row_block(1)..l).rev().collect(),
            (og..l).collect(),
        ));
    }
    for p in &sh.projected {
        let r0 = sh.row_block(p.block) + p.field * m;
        let mut cols = block_cols(p.block);
        cols.extend(p.pin..p.pin + m);
        out.push((
            format!("projected field {} of block {}", p.field, p.block),
            (r0..r0 + m).collect(),
            cols,
        ));
    }
    out.push(("whole".into(), (0..l).collect(), (0..l).collect()));
    out
}

/// Per case, the entries of J[rows, cols] from `jacobian_sub` whose bits differ from `jacobian`'s in any part
/// (usize::MAX for a block of the wrong shape)
fn jac_differ<P: Precision<Native>, E: Equations<B>, const B: usize>(
    sh: &Shooting<E, B>,
    z0: &[Vec<f64>],
    par: bool,
) -> Vec<usize> {
    let z = deep(z0, P::PARTS);
    let zv = parts(&z);
    let (full, ld) = sh.jacobian::<P>(&zv, par).unwrap();
    cases(sh)
        .iter()
        .map(|(_, rows, cols)| {
            let sub = sh.jacobian_sub::<P>(&zv, rows, cols, par).unwrap();
            let nc = cols.len();
            if sub.len() != P::PARTS || sub.iter().any(|s| s.len() != rows.len() * nc) {
                return usize::MAX;
            }
            let mut n = 0;
            for (i, &r) in rows.iter().enumerate() {
                for (j, &c) in cols.iter().enumerate() {
                    n += (0..P::PARTS).any(|q| sub[q][i * nc + j].to_bits() != full[q][c * ld + r].to_bits()) as usize;
                }
            }
            n
        })
        .collect()
}

/// Per case, the rows of F[rows] from `residual_sub` whose bits differ from `residual`'s in any part (usize::MAX for
/// the wrong shape)
fn res_differ<P: Precision<Native>, E: Equations<B>, const B: usize>(
    sh: &Shooting<E, B>,
    z0: &[Vec<f64>],
    par: bool,
) -> Vec<usize> {
    let z = deep(z0, P::PARTS);
    let zv = parts(&z);
    let full = sh.residual::<P>(&zv, par).unwrap();
    cases(sh)
        .iter()
        .map(|(_, rows, _)| {
            let sub = sh.residual_sub::<P>(&zv, rows, par).unwrap();
            if sub.len() != P::PARTS || sub.iter().any(|s| s.len() != rows.len()) {
                return usize::MAX;
            }
            rows.iter()
                .enumerate()
                .filter(|&(i, &r)| (0..P::PARTS).any(|q| sub[q][i].to_bits() != P::part(full[r], q).to_bits()))
                .count()
        })
        .collect()
}

/// `jac_differ` and `res_differ` in double-, quad- and octo-double, with and without threads, printed per problem;
/// the counts of every case in that order
fn report<E: Equations<B>, const B: usize>(name: &str, sh: &Shooting<E, B>, z: &[Vec<f64>]) -> Vec<usize> {
    let names: Vec<String> = cases(sh).into_iter().map(|c| c.0).collect();
    let mut all = Vec::new();
    for par in [false, true] {
        let j = [
            jac_differ::<Double, E, B>(sh, z, par),
            jac_differ::<Quad, E, B>(sh, z, par),
            jac_differ::<Octo, E, B>(sh, z, par),
        ];
        let f = [
            res_differ::<Double, E, B>(sh, z, par),
            res_differ::<Quad, E, B>(sh, z, par),
            res_differ::<Octo, E, B>(sh, z, par),
        ];
        println!("{name}, par {par}, cases {names:?}: J entries that differ (dd, qd, od) {j:?}, F rows {f:?}");
        all.extend(j.into_iter().flatten());
        all.extend(f.into_iter().flatten());
    }
    all
}

/// Pass, written before the first run: on the planted problem, the decorated one (inward segments, two projected
/// fields, a row scaling with rounding factors), the pointwise planted problem, `Cylinder` and Bessel's equation, at
/// states perturbed by 1e-3 with nonzero lower parts, J[rows, cols] from `jacobian_sub` is `jacobian`'s and F[rows]
/// from `residual_sub` is `residual`'s in every part of every entry, for every case of `cases` (a segment block,
/// scattered rows and columns with repeats, the extra rows, the globals' columns, the projected fields' rows, the
/// whole matrix), in double-, quad- and octo-double, with and without threads: 0 entries differ in every count.
#[test]
fn sub_blocks_are_the_whole_bitwise() {
    let mut all = Vec::new();
    let (pl, zp) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    all.extend(report("planted", &pl, &zp));
    let (dc, zd) = decorated();
    all.extend(report("decorated planted", &dc, &zd));
    let (pw, zw) = pointwise();
    all.extend(report("pointwise planted", &pw, &zw));
    let (cy, zc) = Cylinder::<1>::shooting(Cylinder::new(), 30);
    all.extend(report("cylinder", &cy, &zc));
    let (be, zb) = Bessel::shooting(Bessel::new(0.5, 2.0), 30);
    all.extend(report("bessel", &be, &zb));
    assert!(
        all.iter().all(|&n| n == 0),
        "a sub-block entry is not the whole one's: {all:?}"
    );
}

/// The control of `sub_blocks_are_the_whole_bitwise` (feature `mutate`): with the first column task the sub-Jacobian
/// needs and the first block the sub-residual needs left out (`Control::SubSkip`), every case of every problem
/// differs from the whole Jacobian and residual in at least one entry, in double-, quad- and octo-double, with threads.
#[cfg(feature = "mutate")]
#[test]
fn sub_blocks_control_skip() {
    use multishoot::control::Control;
    fn fired<E: Equations<B>, const B: usize>(name: &str, mut sh: Shooting<E, B>, z: &[Vec<f64>]) -> bool {
        sh.set_control(Control::SubSkip);
        let j = [
            jac_differ::<Double, E, B>(&sh, z, true),
            jac_differ::<Quad, E, B>(&sh, z, true),
            jac_differ::<Octo, E, B>(&sh, z, true),
        ];
        let f = [
            res_differ::<Double, E, B>(&sh, z, true),
            res_differ::<Quad, E, B>(&sh, z, true),
            res_differ::<Octo, E, B>(&sh, z, true),
        ];
        println!("control, {name}: J entries that differ (dd, qd, od) {j:?}, F rows {f:?}");
        j.iter().chain(&f).flatten().all(|&n| n > 0 && n != usize::MAX)
    }
    let (pl, zp) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    let (dc, zd) = decorated();
    let (pw, zw) = pointwise();
    let (cy, zc) = Cylinder::<1>::shooting(Cylinder::new(), 30);
    let (be, zb) = Bessel::shooting(Bessel::new(0.5, 2.0), 30);
    let got = [
        fired("planted", pl, &zp),
        fired("decorated planted", dc, &zd),
        fired("pointwise planted", pw, &zw),
        fired("cylinder", cy, &zc),
        fired("bessel", be, &zb),
    ];
    assert_eq!(got, [true; 5], "the control did not fire in every case");
}

/// Pass, written before the first run: an index of `len()` among `jacobian_sub`'s rows or columns, or of `rows()`
/// among `residual_sub`'s rows, is an error; so is `jacobian_sub` on a system that is not square (the planted problem
/// without its extra row), on which `residual_sub` gives `residual`'s rows; and a row scaling of the wrong length is
/// an error of both. Control of each error: the same call with the index one lower, or on the square system,
/// succeeds.
#[test]
fn sub_blocks_check_their_input() {
    let (mut sh, z) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    let z = deep(&z, 2);
    let zv = parts(&z);
    let l = sh.len();
    let bad = [
        sh.jacobian_sub::<Double>(&zv, &[l], &[0], false).is_err(),
        sh.jacobian_sub::<Double>(&zv, &[0], &[l], false).is_err(),
        sh.residual_sub::<Double>(&zv, &[l], false).is_err(),
    ];
    let good = [
        sh.jacobian_sub::<Double>(&zv, &[l - 1], &[0], false).is_ok(),
        sh.jacobian_sub::<Double>(&zv, &[0], &[l - 1], false).is_ok(),
        sh.residual_sub::<Double>(&zv, &[l - 1], false).is_ok(),
    ];
    sh.row_scale = vec![2.0; l - 1];
    let short_scale = [
        sh.jacobian_sub::<Double>(&zv, &[0], &[0], false).is_err(),
        sh.residual_sub::<Double>(&zv, &[0], false).is_err(),
    ];
    sh.row_scale = vec![2.0; l];
    let scale = [
        sh.jacobian_sub::<Double>(&zv, &[0], &[0], false).is_ok(),
        sh.residual_sub::<Double>(&zv, &[0], false).is_ok(),
    ];
    sh.row_scale.clear();
    let extra = sh.extra.pop().unwrap();
    let rows: Vec<usize> = (0..sh.rows()).rev().collect();
    let sub = sh.residual_sub::<Double>(&zv, &rows, true).unwrap();
    let full = sh.residual::<Double>(&zv, true).unwrap();
    let non_square = [
        sh.jacobian_sub::<Double>(&zv, &[0], &[0], false).is_err(),
        rows.iter().enumerate().all(|(i, &r)| {
            (0..2).all(|q| sub[q][i].to_bits() == <Double as Precision<Native>>::part(full[r], q).to_bits())
        }),
    ];
    sh.extra.push(extra);
    let square = sh.jacobian_sub::<Double>(&zv, &[0], &[0], false).is_ok();
    println!(
        "out of range refused {bad:?}, in range taken {good:?}; short row_scale refused {short_scale:?}, full taken \
         {scale:?}; not square: Jacobian refused and residual rows taken {non_square:?}, square taken {square}"
    );
    assert_eq!(bad, [true; 3]);
    assert_eq!(good, [true; 3]);
    assert_eq!(short_scale, [true; 2]);
    assert_eq!(scale, [true; 2]);
    assert_eq!(non_square, [true; 2]);
    assert!(square);
}

/// A problem whose endpoint maps and Taylor steps are counted, the methods of E unchanged
struct Tally<E> {
    eq: E,
    left: AtomicUsize,
    right: AtomicUsize,
    steps: AtomicUsize,
}

impl<E: Equations<B>, const B: usize> Equations<B> for Tally<E> {
    const NF: usize = E::NF;
    const NL: usize = E::NL;
    const NR: usize = E::NR;
    const NG: usize = E::NG;
    const NAUX: usize = E::NAUX;
    const POINTWISE: bool = E::POINTWISE;
    type Prep = E::Prep;

    fn prepare(&self, zh: &[f64], zl: &[f64]) -> E::Prep {
        self.eq.prepare(zh, zl)
    }
    fn try_prepare(&self, zh: &[f64], zl: &[f64]) -> Result<E::Prep, String> {
        self.eq.try_prepare(zh, zl)
    }
    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        self.steps.fetch_add(1, Ordering::Relaxed);
        self.eq.taylor_coeffs(x0, g, w)
    }
    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &E::Prep) -> State<T, B> {
        self.left.fetch_add(1, Ordering::Relaxed);
        self.eq.left_state(x, data, g, prep)
    }
    fn right_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &E::Prep) -> State<T, B> {
        self.right.fetch_add(1, Ordering::Relaxed);
        self.eq.right_state(x, data, g, prep)
    }
}

impl<E> Tally<E> {
    /// (left maps, right maps, Taylor steps) since the last call, the counters then cleared
    fn take(&self) -> (usize, usize, usize) {
        (
            self.left.swap(0, Ordering::Relaxed),
            self.right.swap(0, Ordering::Relaxed),
            self.steps.swap(0, Ordering::Relaxed),
        )
    }
}

/// Pass, written before the first run: on the planted problem with its evaluations counted, J[rows, cols] over the
/// right block's rows by that block's columns (the right data and the global) evaluates the right block's M + 1
/// column tasks only, (left maps, right maps, Taylor steps) = (0, M + 1, 0), and F over the same rows evaluates that
/// block once, (0, 1, 0). Control, that the counters see the other blocks: the whole Jacobian counts NL M + NG = M + 1
/// left maps, M + 1 right maps and at least one step, and the whole residual (1, 1, at least one).
#[test]
fn sub_blocks_evaluate_only_what_they_meet() {
    let (pl, z) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    let mut sh = Shooting::new(
        Tally {
            eq: pl.eq,
            left: AtomicUsize::new(0),
            right: AtomicUsize::new(0),
            steps: AtomicUsize::new(0),
        },
        pl.nodes,
        pl.kt,
    );
    sh.segs = pl.segs;
    sh.extra = pl.extra;
    let m = Shooting::<Tally<Planted<2>>, 2>::M;
    let (k, l) = (sh.k(), sh.len());
    let z = deep(&z, 2);
    let zv = parts(&z);
    let rows: Vec<usize> = (sh.row_block(k)..sh.row_extra()).collect();
    let cols: Vec<usize> = (sh.off_right()..l).collect();
    sh.jacobian_sub::<Double>(&zv, &rows, &cols, true).unwrap();
    let jac_sub = sh.eq.take();
    sh.residual_sub::<Double>(&zv, &rows, true).unwrap();
    let res_sub = sh.eq.take();
    sh.jacobian::<Double>(&zv, true).unwrap();
    let jac = sh.eq.take();
    sh.residual::<Double>(&zv, true).unwrap();
    let res = sh.eq.take();
    println!(
        "(left maps, right maps, Taylor steps): J sub {jac_sub:?}, F sub {res_sub:?}; whole J {jac:?}, whole F {res:?}"
    );
    assert_eq!(
        (jac_sub, res_sub),
        ((0, m + 1, 0), (0, 1, 0)),
        "a sub-block evaluated what it does not meet"
    );
    assert!(
        (jac.0, jac.1, res.0, res.1) == (m + 1, m + 1, 1, 1) && jac.2 > 0 && res.2 > 0,
        "the counters did not see the whole evaluations: {jac:?}, {res:?}"
    );
}
