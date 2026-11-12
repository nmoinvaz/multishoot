//! The fast path on the planted problem: the f64 Jacobian from f64-tangent lanes against the exact one, the staircase
//! LU against the dense f64 LU, and the mixed-precision Newton against the planted solution.
mod common;

use common::{as_pointwise, perturbed, Fnv, Planted};
use multishoot::lane::{Dtn, Fu, Fun, Lane};
use multishoot::prec::{max_abs, nan_max, parts, widen};
use multishoot::{Double, Equations, ExtraRow, MixedLu, NewtonOpts, Quad, Shooting, Storage, DD};
use std::collections::HashMap;

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

fn planted() -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    Planted::<B>::shooting(&NODES, 30, 1e-32)
}

/// The largest |J_f64 - J| over each column, relative to max(1, the column's largest |J|), J the exact double-double
/// Jacobian, over every column: (worst, its column)
fn f64_vs_exact<T: Lane>(sh: &Shooting<Planted<B>, B>, z: &[Vec<f64>]) -> (f64, usize) {
    let (c, ld) = sh.jacobian::<Double>(&parts(z), true).unwrap();
    let (f, ldf) = sh.jacobian_f64::<T>(&parts(z), true).unwrap();
    let l = sh.len();
    let mut worst = (0.0f64, 0usize);
    for j in 0..l {
        let scale = (0..l).map(|i| c[0][j * ld + i].abs()).fold(1.0, nan_max);
        let e = (0..l)
            .map(|i| {
                (DD::new(c[0][j * ld + i], c[1][j * ld + i]) - DD::from_f64(f[j * ldf + i]))
                    .hi
                    .abs()
            })
            .fold(0.0, nan_max);
        if e / scale > worst.0 || e.is_nan() {
            worst = (e / scale, j);
        }
    }
    worst
}

/// The f64 Jacobians of `Fu` (f64 values and tangent), `Fun<4>` (f64 values, four tangents) and `Dtn<4>`
/// (double-double values, four f64 tangents) against the exact Jacobian at z* perturbed by 1e-2. Their tangents are
/// f64 sums along the Taylor paths, accurate to about 1e-16 times the paths' growth. Pass: every column within 1e-11,
/// relative.
#[test]
fn f64_jacobians_match_the_exact_one() {
    let (sh, zs) = planted();
    let z = perturbed(&zs, 1e-2);
    let r = [
        f64_vs_exact::<Fu>(&sh, &z),
        f64_vs_exact::<Fun<4>>(&sh, &z),
        f64_vs_exact::<Dtn<4>>(&sh, &z),
    ];
    println!(
        "f64 Jacobians against the exact one: Fu {:.3e}, Fun<4> {:.3e}, Dtn<4> {:.3e}",
        r[0].0, r[1].0, r[2].0
    );
    for (k, &(e, col)) in r.iter().enumerate() {
        assert!(e < 1e-11, "lane {k}, column {col}: {e:e}");
    }
}

/// the f64 Jacobian of lanes T at z
fn f64_jac<T: Lane>(sh: &Shooting<Planted<B>, B>, z: &[Vec<f64>]) -> Vec<f64> {
    sh.jacobian_f64::<T>(&parts(z), true).unwrap().0
}

/// Four tangents per evaluation give the columns of one: pass, `Fun<4>` equal to `Fun<1>` and `Dtn<4>` to `Dtn<1>`
/// in every entry, with four columns per evaluation where a block has them (46 evaluations against 166).
#[test]
fn four_tangents_equal_one() {
    let (sh, zs) = planted();
    let z = perturbed(&zs, 1e-2);
    let (n4, n1) = (sh.f64_groups::<Fun<4>>().len(), sh.f64_groups::<Fu>().len());
    let differ = |a: &[f64], b: &[f64]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    let (d1, d2) = (
        differ(&f64_jac::<Fun<4>>(&sh, &z), &f64_jac::<Fun<1>>(&sh, &z)),
        differ(&f64_jac::<Dtn<4>>(&sh, &z), &f64_jac::<Dtn<1>>(&sh, &z)),
    );
    println!("four tangents: {n4} evaluations against {n1}; entries differing {d1}, {d2}");
    assert_eq!((d1, d2), (0, 0));
    assert_eq!(n1, sh.jac_tasks().len());
    assert!(n4 * 4 >= n1 && n4 < n1 / 3, "{n4} groups for {n1} tasks");
}

/// The control of `four_tangents_equal_one`: the double-double base trajectory (`Dtn<4>`) must give columns that
/// differ from the f64 one's (`Fun<4>`).
#[test]
fn four_tangents_control_base_precision_differs() {
    let (sh, zs) = planted();
    let z = perturbed(&zs, 1e-2);
    let d = f64_jac::<Dtn<4>>(&sh, &z)
        .iter()
        .zip(&f64_jac::<Fun<4>>(&sh, &z))
        .filter(|(x, y)| x != y)
        .count();
    println!("control, Dtn<4> against Fun<4>: {d} entries differ");
    assert!(d > 0, "control did not fire");
}

/// (|J x - b| / (|J| |x| + |b|) in max norms, |x - x_dense| / |x|) for the staircase solve x of J x = b: J the dense
/// f64 Jacobian of `Fu` at z, b the residual there, x_dense the dense f64 LU's solve
fn staircase_errors(sh: &Shooting<Planted<B>, B>, z: &[Vec<f64>]) -> (f64, f64) {
    let p = parts(z);
    let b = sh.residual::<Double>(&p, true).unwrap();
    let solve = |s: Storage| {
        let lu = sh.mixed_lu::<Fu>(&p, s, true).unwrap();
        let mut x = b.clone();
        lu.solve(&mut x);
        x
    };
    let (xs, xd) = (solve(Storage::Staircase), solve(Storage::Dense));
    let (c, ld) = sh.jacobian_f64::<Fu>(&p, true).unwrap();
    let l = sh.len();
    let jx: Vec<DD> = (0..l)
        .map(|i| (0..l).fold(DD::ZERO, |a, j| a + DD::from_f64(c[j * ld + i]) * xs[j]) - b[i])
        .collect();
    let jn = (0..l)
        .map(|i| (0..l).map(|j| c[j * ld + i].abs()).sum::<f64>())
        .fold(0.0, nan_max);
    let xn = max_abs::<Double>(&xs);
    let back = max_abs::<Double>(&jx) / (jn * xn + max_abs::<Double>(&b));
    let dx: Vec<DD> = xs.iter().zip(&xd).map(|(a, b)| *a - *b).collect();
    (back, max_abs::<Double>(&dx) / xn)
}

/// The staircase LU's solve of the f64 system at z* perturbed by 1e-2, against the dense f64 Jacobian. Both LUs are
/// partial pivoting in f64, backward stable to about n eps. Pass: backward error below 1e-13, and the solution within
/// 1e-8 of the dense LU's (relative; the two pivot in different orders).
#[test]
fn staircase_solves_the_f64_system() {
    let (sh, zs) = planted();
    let (back, dx) = staircase_errors(&sh, &perturbed(&zs, 1e-2));
    println!("staircase: backward error {back:.3e}, against the dense LU {dx:.3e}");
    assert!(back < 1e-13 && dx < 1e-8, "{back:e} {dx:e}");
}

/// Newton's settings
fn opts(tol: f64, max_iter: usize) -> NewtonOpts {
    NewtonOpts {
        tol,
        max_iter,
        ..Default::default()
    }
}

/// max |z - z*| over the unknowns, in double-double
fn dist(z: &[Vec<f64>], zs: &[Vec<f64>]) -> f64 {
    (0..z[0].len())
        .map(|i| (DD::new(z[0][i], z[1][i]) - DD::new(zs[0][i], zs[1][i])).hi.abs())
        .fold(0.0, nan_max)
}

/// The mixed Newton in double-double from z* perturbed by 1e-3 with lanes T in the given storage: (report, |z - z*|)
fn mixed<T: Lane>(
    sh: &Shooting<Planted<B>, B>,
    zs: &[Vec<f64>],
    s: Storage,
    it: usize,
) -> (multishoot::NewtonReport, f64) {
    let mut z = perturbed(zs, 1e-3);
    let rep = sh.newton_mixed::<Double, T>(&mut z, &opts(1e-28, it), s).unwrap();
    (rep, dist(&z, zs))
}

/// the Newton steps the mixed Newton is allowed, one more than it takes on the planted problem
const STEPS: usize = 5;

/// a list of numbers in scientific notation
fn sci(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>().join(" ")
}

/// The mixed-precision Newton (double-double residual, f64 Jacobian) from z* perturbed by 1e-3, with `Fu` dense,
/// `Fu` in staircase storage and `Dtn<4>` in staircase storage, and on the problem with the right block's v rows
/// projected (`Fu` dense). Pass: max |F| at most 1e-28 within 5 steps and every unknown within 1e-27 of z*, as the
/// exact Newton's.
#[test]
fn mixed_newton_converges() {
    let (sh, zs) = planted();
    let (pr, _) = projected(true);
    for (name, (rep, d)) in [
        ("Fu dense", mixed::<Fu>(&sh, &zs, Storage::Dense, STEPS)),
        ("Fu staircase", mixed::<Fu>(&sh, &zs, Storage::Staircase, STEPS)),
        ("Dtn<4> staircase", mixed::<Dtn<4>>(&sh, &zs, Storage::Staircase, STEPS)),
        ("projected, Fu dense", mixed::<Fu>(&pr, &zs, Storage::Dense, STEPS)),
    ] {
        println!(
            "mixed Newton, {name}: max |F| {}; theta {}; |z - z*| {d:.2e}",
            sci(&rep.fnorm),
            sci(&rep.theta)
        );
        assert!(rep.converged && d < 1e-27, "{name}: {:?} {d:e}", rep.fnorm);
    }
}

/// The mixed Newton with a quad-double residual (and the f64 Jacobian of `Dtn<4>` in staircase storage) from the
/// double-double root. Pass: max |F| at most 1e-60 within 6 steps.
#[test]
fn mixed_newton_reaches_quad_double() {
    let (sh, zs) = planted();
    let mut z = perturbed(&zs, 1e-3);
    sh.newton::<Double>(&mut z, &opts(1e-28, 10)).unwrap();
    let mut zq = widen(&z, 4);
    let rep = sh
        .newton_mixed::<Quad, Dtn<4>>(&mut zq, &opts(1e-60, 6), Storage::Staircase)
        .unwrap();
    println!("mixed Newton in quad-double: max |F| {}", sci(&rep.fnorm));
    assert!(rep.converged, "{:?}", rep.fnorm);
}

/// The planted problem with block 2's u rows projected onto the odd harmonics k <= 3 and node 1's u pinned (in
/// block 2's input group, so the staircase storage holds it), or with the right block's v rows projected and the left
/// data pinned (`right`: outside that block's groups)
fn projected(right: bool) -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    let (mut sh, zs) = planted();
    if right {
        let k = sh.k();
        sh.project_rows(k, 1, 0, multishoot::periodic::Parity::Even, 2).unwrap();
    } else {
        let pin = sh.off_node(1);
        sh.project_rows(2, 0, pin, multishoot::periodic::Parity::Odd, 3)
            .unwrap();
    }
    (sh, zs)
}

/// The f64 Jacobian (`Dtn<4>`) of both projected layouts against their exact Jacobians: pass, every column within
/// 1e-11, relative; and the staircase solve of the layout it holds within 1e-13 backward error and 1e-8 of the dense
/// solve, the other layout refused.
#[test]
fn f64_jacobian_projects_its_rows() {
    for right in [false, true] {
        let (sh, zs) = projected(right);
        let z = perturbed(&zs, 1e-2);
        let (e, col) = f64_vs_exact::<Dtn<4>>(&sh, &z);
        println!(
            "projected ({}): f64 Jacobian against the exact one {e:.3e} at column {col}",
            if right { "right block" } else { "block 2" }
        );
        assert!(e < 1e-11, "{e:e}");
    }
    let (sh, zs) = projected(false);
    let (back, dx) = staircase_errors(&sh, &perturbed(&zs, 1e-2));
    println!("projected block 2: staircase backward error {back:.3e}, against the dense LU {dx:.3e}");
    assert!(back < 1e-13 && dx < 1e-8, "{back:e} {dx:e}");
    let (sh, zs) = projected(true);
    assert!(sh.jacobian_staircase::<Fu>(&parts(&zs), true).is_err());
}

/// The control of `f64_jacobian_projects_its_rows`: the unprojected problem's f64 Jacobian must fail against the
/// projected exact Jacobian, on both layouts.
#[test]
fn f64_projection_control_unprojected_fails() {
    for right in [false, true] {
        let (sh, zs) = projected(right);
        let (plain, _) = planted();
        let z = perturbed(&zs, 1e-2);
        let (c, ld) = sh.jacobian::<Double>(&parts(&z), true).unwrap();
        let (f, ldf) = plain.jacobian_f64::<Dtn<4>>(&parts(&z), true).unwrap();
        let e = (0..sh.len())
            .flat_map(|j| (0..sh.len()).map(move |i| (i, j)))
            .map(|(i, j)| (c[0][j * ld + i] - f[j * ldf + i]).abs())
            .fold(0.0, nan_max);
        println!("control, unprojected f64 Jacobian: largest difference {e:.3e}");
        assert!(e >= 1e-11, "control did not fire");
    }
}

/// The f64 Jacobian at z (leading dimension ld) assembled outside the engine from its column evaluations, as a caller
/// that chooses the lanes of each block would: block `dd_block`'s tasks by `jac_column` (double-double, rounded when
/// stored, and with `store` false left out of the matrix), every other task by `jac_columns_f64::<Dtn<4>>` in the
/// engine's groups, then the targets' -1, with `project` the projected rows by `project_f64_block` (given the
/// double-double columns when `dd`), and the extra row's weights.
fn assembled(
    sh: &Shooting<Planted<B>, B>,
    z: &[Vec<f64>],
    ld: usize,
    dd_block: Option<usize>,
    dd: bool,
    project: bool,
    store: bool,
) -> Vec<f64> {
    let zp = parts(z);
    let (l, nr) = (sh.len(), 2 * Shooting::<Planted<B>, B>::M);
    let g = sh.globals::<Double>(&zp);
    // the planted problem's preparation is (); a caller forms it as for any problem
    #[allow(clippy::let_unit_value)]
    let prep = sh.eq.try_prepare(&z[0], &z[1]).unwrap();
    let mut w = vec![0.0; ld * l];
    // the double-double columns of block dd_block's tasks, by unknown
    let mut dd_cols: HashMap<usize, Vec<DD>> = HashMap::new();
    for gr in sh.f64_groups::<Dtn<4>>() {
        let (b, r0) = (gr[0].0, sh.row_block(gr[0].0));
        if Some(b) == dd_block {
            for &t in &gr {
                let col = sh.jac_column::<Double>(&zp, &g, &prep, t);
                let c = sh.seed_columns(b, t.1)[0];
                for (r, v) in col.iter().enumerate().filter(|_| store) {
                    w[c * ld + r0 + r] = v.hi;
                }
                dd_cols.insert(c, col);
            }
            continue;
        }
        for (t, col) in gr.iter().zip(sh.jac_columns_f64::<Dtn<4>>(&zp, &g, &prep, &gr)) {
            let c = sh.seed_columns(b, t.1)[0];
            w[c * ld + r0..c * ld + r0 + nr].copy_from_slice(&col);
        }
    }
    for b in 0..=sh.k() {
        let (t, r0) = (sh.block_target(b), sh.row_block(b));
        for r in 0..nr {
            w[(t + r) * ld + r0 + r] -= 1.0;
        }
        if project {
            let dd_column = |j: usize| {
                if dd && Some(b) == dd_block {
                    dd_cols.get(&j).map(|c| c.as_slice())
                } else {
                    None
                }
            };
            sh.project_f64_block(b, &mut w, ld, r0, l, Some, dd_column).unwrap();
        }
    }
    let re = sh.row_extra();
    for (r, row) in sh.extra.iter().enumerate() {
        for &(i, wt) in &row.terms {
            w[i * ld + re + r] += wt.hi;
        }
    }
    w
}

/// entries of a and b (column-major, leading dimension ld, n columns) in the rows `rows` whose bits differ
fn bits_differ(a: &[f64], b: &[f64], ld: usize, n: usize, rows: std::ops::Range<usize>) -> usize {
    (0..n)
        .flat_map(|j| rows.clone().map(move |i| j * ld + i))
        .filter(|&k| a[k].to_bits() != b[k].to_bits())
        .count()
}

/// The public projection serves a caller that assembles the f64 Jacobian itself: pass, the matrix assembled from the
/// engine's column evaluations with `project_f64_block` equals `jacobian_f64`'s in every bit, on both projected
/// layouts.
#[test]
fn assembled_f64_jacobian_is_the_engines() {
    for right in [false, true] {
        let (sh, zs) = projected(right);
        let z = perturbed(&zs, 1e-2);
        let (f, ld) = sh.jacobian_f64::<Dtn<4>>(&parts(&z), true).unwrap();
        let a = assembled(&sh, &z, ld, None, false, true, true);
        let n = bits_differ(&a, &f, ld, sh.len(), 0..sh.len());
        println!("projected ({right}): {n} entries of the assembled f64 Jacobian differ from the engine's");
        assert_eq!(n, 0);
    }
}

/// The control of `assembled_f64_jacobian_is_the_engines`: without the projection the assembled matrix must differ.
#[test]
fn assembled_f64_jacobian_control_unprojected_differs() {
    for right in [false, true] {
        let (sh, zs) = projected(right);
        let z = perturbed(&zs, 1e-2);
        let (f, ld) = sh.jacobian_f64::<Dtn<4>>(&parts(&z), true).unwrap();
        let a = assembled(&sh, &z, ld, None, false, false, true);
        assert!(
            bits_differ(&a, &f, ld, sh.len(), 0..sh.len()) > 0,
            "control did not fire"
        );
    }
}

/// The planted problem with block 2's u rows projected onto the odd harmonics k <= 3 and node `pin`'s u pinned: node
/// 3, outside the block's columns (node 1, the global and node 2), or node 1, its input
fn block2_projected(pin: usize) -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    let (mut sh, zs) = planted();
    let off = sh.off_node(pin);
    sh.project_rows(2, 0, off, multishoot::periodic::Parity::Odd, 3)
        .unwrap();
    (sh, zs)
}

/// (entries of block 2's rows that differ from the double-double Jacobian's hi parts, entries of the other rows that
/// differ from `jacobian_f64`'s) for the f64 Jacobian assembled with block 2's columns in double-double, given to the
/// projection when `dd`, with node `pin`'s u pinned (`block2_projected`)
fn dd_block_rows(dd: bool, pin: usize) -> (usize, usize) {
    let (sh, zs) = block2_projected(pin);
    let z = perturbed(&zs, 1e-2);
    let (c, ld) = sh.jacobian::<Double>(&parts(&z), true).unwrap();
    let (f, ldf) = sh.jacobian_f64::<Dtn<4>>(&parts(&z), true).unwrap();
    assert_eq!(ld, ldf);
    let a = assembled(&sh, &z, ld, Some(2), dd, true, true);
    let (n, blk) = (sh.len(), sh.row_block(2)..sh.row_block(3));
    let inside = bits_differ(&a, &c[0], ld, n, blk.clone());
    let outside = bits_differ(&a, &f, ld, n, 0..blk.start) + bits_differ(&a, &f, ld, n, blk.end..n);
    (inside, outside)
}

/// Double-double columns given to the projection: pass, block 2's rows of the assembled f64 Jacobian equal the
/// double-double Jacobian's hi parts in every bit, its projected rows included, and every other row equals
/// `jacobian_f64`'s, with the pinned unknowns outside the block's columns and among them.
#[test]
fn dd_columns_project_to_the_dd_jacobians_rows() {
    for pin in [3, 1] {
        let (inside, outside) = dd_block_rows(true, pin);
        println!(
            "double-double columns, node {pin} pinned: {inside} entries of block 2's rows differ from the dd \
             Jacobian's, {outside} of the others from jacobian_f64's"
        );
        assert!(inside == 0 && outside == 0, "node {pin}: {inside} {outside}");
    }
}

/// The control of `dd_columns_project_to_the_dd_jacobians_rows`: projecting the columns' f64 roundings instead must
/// leave some entry of block 2's rows off the double-double Jacobian's.
#[test]
fn dd_columns_control_rounded_columns_differ() {
    for pin in [3, 1] {
        let (inside, _) = dd_block_rows(false, pin);
        println!("control, rounded columns, node {pin} pinned: {inside} entries of block 2's rows differ");
        assert!(inside > 0, "control did not fire");
    }
}

/// The double-double columns are what the projection reads: pass, block 2's projected rows are the same in every bit
/// when the matrix holds the columns' roundings and when it holds zeros there.
#[test]
fn dd_columns_are_read_where_the_matrix_is_zero() {
    let (sh, zs) = block2_projected(3);
    let z = perturbed(&zs, 1e-2);
    let (_, ld) = sh.jacobian_f64::<Dtn<4>>(&parts(&z), true).unwrap();
    let a = assembled(&sh, &z, ld, Some(2), true, true, true);
    let b = assembled(&sh, &z, ld, Some(2), true, true, false);
    let r0 = sh.row_block(2);
    let n = bits_differ(&a, &b, ld, sh.len(), r0..r0 + Shooting::<Planted<B>, B>::M);
    println!("block 2's projected rows: {n} entries differ with the double-double columns left out of the matrix");
    assert_eq!(n, 0);
}

/// The control of `dd_columns_are_read_where_the_matrix_is_zero`: with the columns left out of the matrix and not
/// given to the projection either, block 2's projected rows must differ.
#[test]
fn dd_columns_control_not_given_differ() {
    let (sh, zs) = block2_projected(3);
    let z = perturbed(&zs, 1e-2);
    let (_, ld) = sh.jacobian_f64::<Dtn<4>>(&parts(&z), true).unwrap();
    let a = assembled(&sh, &z, ld, Some(2), true, true, true);
    let b = assembled(&sh, &z, ld, Some(2), false, true, false);
    let r0 = sh.row_block(2);
    let n = bits_differ(&a, &b, ld, sh.len(), r0..r0 + Shooting::<Planted<B>, B>::M);
    println!("control, columns neither stored nor given: {n} entries of block 2's projected rows differ");
    assert!(n > 0, "control did not fire");
}

/// The public projection's checks. Pass: it refuses a block past the last, a matrix too short for its columns, a
/// block's rows past the leading dimension, sizes whose products overflow, a double-double column of the wrong length,
/// a pinned unknown without a column and two pinned unknowns on one column, which
/// `assembled_f64_jacobian_is_the_engines` shows it accepts when they are right.
#[test]
fn project_f64_block_refuses_bad_inputs() {
    let (sh, zs) = projected(false);
    let z = perturbed(&zs, 1e-2);
    let (f, ld) = sh.jacobian_f64::<Dtn<4>>(&parts(&z), true).unwrap();
    let l = sh.len();
    let r0 = sh.row_block(2);
    let short = sh.project_f64_block(2, &mut f.clone()[..ld * l - 1], ld, r0, l, Some, |_| None);
    let past = sh.project_f64_block(2, &mut f.clone(), ld, ld - 1, l, Some, |_| None);
    let rows_overflow = sh.project_f64_block(2, &mut f.clone(), ld, usize::MAX, l, Some, |_| None);
    let cols_overflow = sh.project_f64_block(2, &mut f.clone(), usize::MAX, r0, l, Some, |_| None);
    let col = vec![DD::ONE; 3];
    let wrong = sh.project_f64_block(2, &mut f.clone(), ld, r0, l, Some, |_| Some(col.as_slice()));
    let unpinned = sh.project_f64_block(2, &mut f.clone(), ld, r0, l, |_| None, |_| None);
    let pin = sh.off_node(1);
    let shared = sh.project_f64_block(2, &mut f.clone(), ld, r0, l, |u| Some(u.min(pin + 1)), |_| None);
    let no_block = sh.project_f64_block(sh.k() + 1, &mut f.clone(), ld, r0, l, Some, |_| None);
    let all = [
        short,
        past,
        rows_overflow,
        cols_overflow,
        wrong,
        unpinned,
        shared,
        no_block,
    ];
    println!("refusals: {all:?}");
    assert!(all.iter().all(|r| r.is_err()));
}

/// A double-double column given as zero replaces the matrix's entries: pass, block 2's projected rows of one of its
/// input columns, nonzero in the matrix, are zero after the projection is given zeros for that column, and the other
/// columns' rows are unchanged from the projection given no column. Its control: given no column, the projection must
/// leave nonzero entries in those rows.
#[test]
fn zero_dd_column_replaces_the_matrix_entries() {
    let (sh, zs) = block2_projected(3);
    let z = perturbed(&zs, 1e-2);
    let (f, ld) = sh.jacobian_f64::<Dtn<4>>(&parts(&z), true).unwrap();
    let (l, m, r0) = (sh.len(), Shooting::<Planted<B>, B>::M, sh.row_block(2));
    let jc = sh.off_node(1) + 2;
    assert!((0..m).any(|i| f[jc * ld + r0 + i] != 0.0));
    let zeros = vec![DD::ZERO; 2 * m];
    let (mut a, mut b) = (f.clone(), f.clone());
    sh.project_f64_block(2, &mut a, ld, r0, l, Some, |j| (j == jc).then_some(zeros.as_slice()))
        .unwrap();
    sh.project_f64_block(2, &mut b, ld, r0, l, Some, |_| None).unwrap();
    let nonzero = |w: &[f64]| (0..m).filter(|&i| w[jc * ld + r0 + i] != 0.0).count();
    let (left, control) = (nonzero(&a), nonzero(&b));
    let other = (0..l)
        .filter(|&j| j != jc)
        .flat_map(|j| (0..m).map(move |i| j * ld + r0 + i))
        .filter(|&k| a[k].to_bits() != b[k].to_bits())
        .count();
    println!(
        "zero column: {left} nonzero entries left in its projected rows ({control} given no column), {other} \
         entries of other columns differ"
    );
    assert!(control > 0, "control did not fire");
    assert!(left == 0 && other == 0, "{left} {other}");
}

/// The hashes of the f64 Jacobians assembled with `project_f64_block` at z* perturbed by 1e-2, with block 2's u rows
/// projected and node 1's u, its input, pinned: from f64 columns only, and with block 2's columns in double-double
/// given to the projection; `ulp` changes the last bit of unknown 37
fn assembled_hashes(ulp: bool) -> [u64; 2] {
    let (sh, zs) = block2_projected(1);
    let mut z = perturbed(&zs, 1e-2);
    if ulp {
        z[0][37] = f64::from_bits(z[0][37].to_bits() + 1);
    }
    let (_, ld) = sh.jacobian_f64::<Dtn<4>>(&parts(&z), true).unwrap();
    let h = |w: Vec<f64>| {
        let mut f = Fnv::new();
        w.into_iter().for_each(|x| f.f64(x));
        f.0
    };
    [
        h(assembled(&sh, &z, ld, None, false, true, true)),
        h(assembled(&sh, &z, ld, Some(2), true, true, true)),
    ]
}

/// the reference hashes of `assembled_hashes`, recorded on aarch64 (NEON) and on the portable backend
const ASSEMBLED_REFERENCE: [u64; 2] = [0xa9594acbd32b96ae, 0x86a3a742b8ac736a];

/// The public projection's bytes. Pass: the hashes equal the reference on every SIMD backend of the CI matrix.
#[test]
fn assembled_bytes_match_reference() {
    let h = assembled_hashes(false);
    println!(
        "assembled hashes [{}]",
        h.iter().map(|x| format!("{x:#018x}")).collect::<Vec<_>>().join(", ")
    );
    assert_eq!(
        h, ASSEMBLED_REFERENCE,
        "the assembled f64 Jacobians' bytes differ from the reference"
    );
}

/// The control of `assembled_bytes_match_reference`: one ulp of one unknown must change both hashes.
#[test]
fn assembled_bytes_control_one_ulp_differs() {
    let (a, b) = (assembled_hashes(false), assembled_hashes(true));
    assert!(a[0] != b[0] && a[1] != b[1], "control did not fire");
}

/// The layout checks of the fast path. Pass: the staircase storage refuses a second extra row, an extra row on node
/// 1 and a row_scale of the wrong length, which the dense f64 Jacobian also refuses, as it does a system that is
/// not square.
#[test]
fn staircase_refuses_other_layouts() {
    let (mut sh, zs) = planted();
    let og = sh.off_global();
    sh.extra.push(ExtraRow {
        terms: vec![(og, DD::ONE)],
        rhs: DD::ZERO,
    });
    assert!(sh.jacobian_staircase::<Fu>(&parts(&zs), true).is_err());
    let (mut sh, zs) = planted();
    let i = sh.off_node(1);
    sh.extra[0].terms.push((i, DD::ONE));
    assert!(sh.jacobian_staircase::<Fu>(&parts(&zs), true).is_err());
    // a row_scale of the wrong length, refused by both storages; the right length is accepted (the control)
    let (mut sh, zs) = scaled();
    sh.row_scale.pop();
    assert!(sh.jacobian_staircase::<Fu>(&parts(&zs), true).is_err());
    assert!(sh.jacobian_f64::<Fu>(&parts(&zs), true).is_err());
    sh.row_scale.push(1.0);
    assert!(sh.jacobian_staircase::<Fu>(&parts(&zs), true).is_ok());
    // a system that is not square, refused by the dense storage
    let (mut sh, zs) = planted();
    sh.extra.clear();
    assert!(sh.jacobian_f64::<Fu>(&parts(&zs), true).is_err());
}

/// the planted problem with block 3's rows scaled by 2^100 (`row_scale`)
fn scaled() -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    let (mut sh, zs) = planted();
    let (r0, r1) = (sh.row_block(3), sh.row_block(4));
    sh.row_scale = (0..sh.rows())
        .map(|r| if (r0..r1).contains(&r) { 2f64.powi(100) } else { 1.0 })
        .collect();
    (sh, zs)
}

/// (entries of the scaled problem's f64 Jacobian that differ from the unscaled one's times `t` on block 3's rows and
/// times 1 elsewhere, the relative difference of the scaled and unscaled staircase solves of J x = F)
fn scaling_errors(t: f64) -> (usize, f64) {
    let (plain, zs) = planted();
    let (sc, _) = scaled();
    let z = perturbed(&zs, 1e-2);
    let p = parts(&z);
    let ((a, ld), (b, _)) = (
        plain.jacobian_f64::<Fu>(&p, true).unwrap(),
        sc.jacobian_f64::<Fu>(&p, true).unwrap(),
    );
    let (r0, r1) = (sc.row_block(3), sc.row_block(4));
    let l = sc.len();
    let differ = (0..l)
        .flat_map(|j| (0..l).map(move |i| (i, j)))
        .filter(|&(i, j)| a[j * ld + i] * if (r0..r1).contains(&i) { t } else { 1.0 } != b[j * ld + i])
        .count();
    let solve = |sh: &Shooting<Planted<B>, B>| {
        let mut x = sh.residual::<Double>(&p, true).unwrap();
        sh.mixed_lu::<Fu>(&p, Storage::Staircase, true).unwrap().solve(&mut x);
        x
    };
    let (x0, x1) = (solve(&plain), solve(&sc));
    let dx: Vec<DD> = x0.iter().zip(&x1).map(|(a, b)| *a - *b).collect();
    (differ, max_abs::<Double>(&dx) / max_abs::<Double>(&x0))
}

/// Scaled rows in the f64 Jacobian and the staircase storage: pass, the f64 Jacobian with block 3 scaled by 2^100
/// equal to the unscaled one times the factors in every entry (powers of two scale exactly), and the staircase solve
/// of the scaled system within 1e-8 of the unscaled one (the scaling changes the rows, not the solution).
#[test]
fn f64_jacobian_and_staircase_scale_rows() {
    let (differ, dx) = scaling_errors(2f64.powi(100));
    println!("scaled rows: {differ} f64 entries differ, staircase solutions {dx:.3e} apart");
    assert!(differ == 0 && dx < 1e-8, "{differ} {dx:e}");
}

/// The control of `f64_jacobian_and_staircase_scale_rows`: told the factor is 2^101, the entries must differ.
#[test]
fn f64_scaling_control_wrong_factor_differs() {
    let (differ, _) = scaling_errors(2f64.powi(101));
    println!("control, wrong factor: {differ} f64 entries differ");
    assert!(differ > 0, "control did not fire");
}

/// (entries of the f64 Jacobian of the planted problem, `coupled` or pointwise, declared pointwise that differ from
/// the pointwise problem's per-column one), lanes `Fun<4>`
fn pointwise_f64_differs(coupled: bool) -> usize {
    let (plain, zs) = Planted::<B>::shooting_of(Planted::with_coupling(false), &NODES, 30, 1e-32);
    let (sh, _) = Planted::<B>::shooting_of(Planted::with_coupling(coupled), &NODES, 30, 1e-32);
    let z = perturbed(&zs, 1e-2);
    let (a, _) = plain.jacobian_f64::<Fun<4>>(&parts(&z), true).unwrap();
    let (b, _) = as_pointwise(sh).jacobian_f64::<Fun<4>>(&parts(&z), true).unwrap();
    a.iter().zip(&b).filter(|(x, y)| x != y).count()
}

/// The pointwise problem's f64 Jacobian, one tangent per input field (every point at once), against per column.
/// Pass: every entry equal.
#[test]
fn pointwise_f64_jacobian_equals_per_column() {
    let d = pointwise_f64_differs(false);
    println!("pointwise f64 Jacobian against per column: {d} entries differ");
    assert_eq!(d, 0);
}

/// The control of `pointwise_f64_jacobian_equals_per_column`: the coupled problem declared pointwise must differ.
#[test]
fn pointwise_f64_control_coupled_differs() {
    let d = pointwise_f64_differs(true);
    println!("control, coupled problem declared pointwise: {d} entries differ");
    assert!(d > 0, "control did not fire");
}

/// (whether z + dz from `newton_step_mixed` in the storage equals, in every part, the state after one full step of
/// `newton_mixed` with dense storage, whose first step the planted problem takes whole) at z* perturbed by 1e-3
fn step_matches_loop(storage: Storage) -> bool {
    let (sh, zs) = planted();
    let z = perturbed(&zs, 1e-3);
    let (dz, f, lu) = sh.newton_step_mixed::<Double, Fu>(&parts(&z), storage, true).unwrap();
    assert!(lu.pivot_range().0 > 0.0 && max_abs::<Double>(&f) > 0.0);
    let mut zl = z.clone();
    let rep = sh
        .newton_mixed::<Double, Fu>(&mut zl, &opts(1e-28, 1), Storage::Dense)
        .unwrap();
    assert_eq!(rep.scale, [1.0]);
    (0..sh.len()).all(|i| {
        let v = DD::new(z[0][i], z[1][i]) + DD::new(dz[0][i], dz[1][i]);
        v.hi == zl[0][i] && v.lo == zl[1][i]
    })
}

/// The mixed step is the loop's step: pass, z + dz of `newton_step_mixed` (dense) equal to the state after the loop's
/// first, whole step, in both parts of every unknown.
#[test]
fn newton_step_mixed_is_the_loops_step() {
    assert!(step_matches_loop(Storage::Dense));
}

/// The control of `newton_step_mixed_is_the_loops_step`: the step from staircase storage, which pivots in another
/// order, must differ from the dense loop's in some bit.
#[test]
fn newton_step_mixed_control_other_storage_differs() {
    assert!(!step_matches_loop(Storage::Staircase));
}

/// The hashes of the f64 Jacobians of `Fu` and `Dtn<4>` at z* perturbed by 1e-2, of the staircase solve there, and of
/// the mixed Newton's root (`Fu`, staircase, 8 steps); `ulp` changes the last bit of unknown 37 in both states
fn hashes(ulp: bool) -> [u64; 4] {
    let (sh, zs) = planted();
    let bump = |z: &mut Vec<Vec<f64>>| z[0][37] = f64::from_bits(z[0][37].to_bits() + 1);
    let mut z = perturbed(&zs, 1e-2);
    if ulp {
        bump(&mut z);
    }
    let h = |v: &mut dyn Iterator<Item = f64>| {
        let mut f = Fnv::new();
        v.for_each(|x| f.f64(x));
        f.0
    };
    let ju = h(&mut f64_jac::<Fu>(&sh, &z).into_iter());
    let jd = h(&mut f64_jac::<Dtn<4>>(&sh, &z).into_iter());
    let lu: MixedLu = sh.mixed_lu::<Fu>(&parts(&z), Storage::Staircase, true).unwrap();
    let mut x = sh.residual::<Double>(&parts(&z), true).unwrap();
    lu.solve(&mut x);
    let hx = h(&mut x.iter().flat_map(|v| [v.hi, v.lo]));
    let mut zn = perturbed(&zs, 1e-3);
    if ulp {
        bump(&mut zn);
    }
    sh.newton_mixed::<Double, Fu>(&mut zn, &opts(1e-28, 8), Storage::Staircase)
        .unwrap();
    let hz = h(&mut zn.iter().flatten().copied());
    [ju, jd, hx, hz]
}

/// the reference hashes, recorded on aarch64 (NEON) and on the portable backend
const REFERENCE: [u64; 4] = [
    0x4e34899a74370968,
    0x52724fc11d9ada8d,
    0x9a0a0f1dda3264f9,
    0xffa6576f016ea19f,
];

/// The fast path's bytes. Pass: the hashes equal the reference on every SIMD backend of the CI matrix and with any
/// number of threads.
#[test]
fn fast_bytes_match_reference() {
    let h = hashes(false);
    println!(
        "fast path hashes [{}]",
        h.iter().map(|x| format!("{x:#018x}")).collect::<Vec<_>>().join(", ")
    );
    assert_eq!(h, REFERENCE, "the fast path's bytes differ from the reference");
}

/// The control of `fast_bytes_match_reference`: one ulp of one unknown must change the Jacobians' and the solve's
/// hashes (the Newton root, converged, need not change).
#[test]
fn fast_bytes_control_one_ulp_differs() {
    let (a, b) = (hashes(false), hashes(true));
    assert!(a[0] != b[0] && a[1] != b[1] && a[2] != b[2], "control did not fire");
}

/// The controls (feature `mutate`), each against the check it breaks.
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::Control;

    /// a dropped column fails `f64_jacobians_match_the_exact_one` at that column
    #[test]
    fn dropped_column_fails_the_f64_jacobian() {
        let (mut sh, zs) = planted();
        let z = perturbed(&zs, 1e-2);
        let col = sh.off_node(2) + 5;
        let (c, ld) = sh.jacobian::<Double>(&parts(&z), true).unwrap();
        sh.set_control(Control::DropColumn(col));
        let (f, ldf) = sh.jacobian_f64::<Dtn<4>>(&parts(&z), true).unwrap();
        let e = (0..sh.len())
            .map(|i| (c[0][col * ld + i] - f[col * ldf + i]).abs())
            .fold(0.0, nan_max);
        println!("control, dropped column {col}: largest difference {e:.3e}");
        assert!(e >= 1e-11, "control did not fire");
    }

    /// the staircase controls fail `staircase_solves_the_f64_system`
    #[test]
    fn staircase_pivot_and_link_fail_the_solve() {
        for ctl in [Control::StaircasePivot, Control::StaircaseLink] {
            let (mut sh, zs) = planted();
            sh.set_control(ctl);
            let (back, dx) = staircase_errors(&sh, &perturbed(&zs, 1e-2));
            println!("control {ctl:?}: backward error {back:.3e}, against the dense LU {dx:.3e}");
            assert!(back >= 1e-13 || dx >= 1e-8, "control {ctl:?} did not fire");
        }
    }

    /// the state read in half its parts (the quad-double state in double-double) keeps the quad-double mixed Newton
    /// above the 1e-60 of `mixed_newton_reaches_quad_double`, in the same 6 steps
    #[test]
    fn half_parts_fail_the_quad_double_mixed_newton() {
        let (mut sh, zs) = planted();
        let mut z = perturbed(&zs, 1e-3);
        sh.newton::<Double>(&mut z, &opts(1e-28, 10)).unwrap();
        sh.set_control(Control::HalfParts);
        let mut zq = widen(&z, 4);
        let rep = sh
            .newton_mixed::<Quad, Dtn<4>>(&mut zq, &opts(1e-60, 6), Storage::Staircase)
            .unwrap();
        println!(
            "control, half parts: quad-double mixed Newton max |F| {}",
            sci(&rep.fnorm)
        );
        assert!(!rep.converged, "control did not fire");
    }

    /// a pivot off by 1e-6 slows the mixed Newton: it must not converge in the steps `mixed_newton_converges` allows
    #[test]
    fn staircase_pivot_slows_the_mixed_newton() {
        let (mut sh, zs) = planted();
        sh.set_control(Control::StaircasePivot);
        let (rep, d) = mixed::<Fu>(&sh, &zs, Storage::Staircase, STEPS);
        println!("control, pivot: max |F| {}; |z - z*| {d:.2e}", sci(&rep.fnorm));
        assert!(!rep.converged, "control did not fire");
    }
}
