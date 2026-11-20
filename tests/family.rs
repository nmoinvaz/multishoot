//! The layout as a family in its global (`Shooting::in_global`) on Bratu's problem, whose branch folds at a known
//! parameter: the family against the layout's own residual and Jacobian, multisolver's continuation through the fold
//! and the fold's location against the closed form, and the f64 factors of the corrector's bordered matrix.

use multishoot::lane::{Dtn, Lane, DV};
use multishoot::multidouble::dd::split;
use multishoot::multisolver::arclength::{continuation, locate_fold, start, ArcOpts, Bordered, FoldOpts};
use multishoot::multisolver::newton::{exact, Factored};
use multishoot::multisolver::prec::{nan_max, parts, widen};
use multishoot::multisolver::NewtonOpts;
use multishoot::{
    cs, Double, Equations, ExtraRow, Family, Fl, InGlobal, Native, Plan, Precision, Quad, Shooting, State, Work, DD,
};

/// lambda_c, the fold of Bratu's problem on [0, 1], in quad-double parts: the root theta_c of theta tanh(theta / 4) = 4
/// and theta_c^2 / (2 cosh^2(theta_c / 4)), formed to 110 digits (Python's decimal) and rounded part by part
const LAMBDA_C: [f64; 4] = [
    3.513830719125161,
    1.566482374576037e-19,
    -6.75357710800099e-36,
    6.406094134319189e-52,
];

/// Bratu's problem u'' + a lambda e^u = 0, u(0) = u(1) = 0, with a = 1 / (j + 1) on lane j (B = 1, pointwise), as the
/// system u' = p, p' = -a lambda E, E' = p E in u, p and E = e^u with E(0) = 1. lambda is the layout's global, fixed by
/// one extra row. Its branch from u = 0 at lambda = 0 is u = 2 ln(cosh(theta / 4) / cosh((x - 1/2) theta / 2)) with
/// a lambda = theta^2 / (2 cosh^2(theta / 4)), which folds where theta tanh(theta / 4) = 4: lane j at a lambda =
/// lambda_c, with u'(0) = theta tanh(theta / 4) = 4. The layout's branch folds first at lane 0's fold, lambda =
/// lambda_c, where the other lanes are regular, so the fold is simple. With `wrong`, the control of the fold checks,
/// the E recursion is scaled by 1 + 2^-30: then E = e^((1 + 2^-30) u), and the fold moves to lambda_c / (1 + 2^-30)
/// with u'(0) = 4 / (1 + 2^-30) there.
struct Bratu {
    a: DV,
    wrong: bool,
}

impl Equations<1> for Bratu {
    const NF: usize = 3; // u, p, E
    const NL: usize = 1; // p(0)
    const NR: usize = 2; // p(1), E(1)
    const NG: usize = 1; // lambda
    const POINTWISE: bool = true;
    type Prep = ();

    /// (k + 1) u_(k+1) = p_k, (k + 1) p_(k+1) = -a lambda E_k, (k + 1) E_(k+1) = (p E)_k, with 1 / (k + 1) formed in
    /// the lane's precision: as a double-double constant its rounding would move the fold by about 1e-34
    fn taylor_coeffs<T: Lane>(&self, _x0: f64, g: &[T], w: &mut Work<T, 1>) {
        for k in 0..w.kt {
            let ik = T::splat(DD::ONE) / T::splat(DD::from_f64((k + 1) as f64));
            let pe = cs(&w.c[1], &w.c[2], k);
            let pe = if self.wrong {
                pe.muls(DD::ONE + DD::from_f64(2f64.powi(-30)))
            } else {
                pe
            };
            let ae = Fl([w.c[2][k].0[0].mulc(self.a)]);
            w.c[0][k + 1] = w.c[1][k].mul_t(ik);
            w.c[1][k + 1] = -ae.mul_t(g[0]).mul_t(ik);
            w.c[2][k + 1] = pe.mul_t(ik);
        }
    }

    /// u(0) = 0, E(0) = 1; the data is p(0)
    fn left_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], _g: &[T], _prep: &()) -> State<T, 1> {
        vec![Fl::zero(), data[0], Fl::splat(DD::ONE)]
    }

    /// u(1) = 0; the data is p(1) and E(1)
    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], _g: &[T], _prep: &()) -> State<T, 1> {
        vec![Fl::zero(), data[0], data[1]]
    }
}

/// Bratu's layout on five nodes with the extra row lambda = 0, and its exact state there (u = p = 0, E = 1) in P,
/// the segments planned from it at order kt with steps of 1/16
fn bratu<P: Precision<Native>>(kt: usize) -> (Shooting<Bratu, 1>, Vec<Vec<f64>>) {
    bratu_with::<P>(kt, false)
}

/// `bratu`, with the E recursion scaled when `wrong`
fn bratu_with<P: Precision<Native>>(kt: usize, wrong: bool) -> (Shooting<Bratu, 1>, Vec<Vec<f64>>) {
    let a: Vec<DD> = (0..8).map(|j| DD::ONE / DD::from_f64((j + 1) as f64)).collect();
    let (hi, lo) = split(&a);
    let mut sh = Shooting::new(
        Bratu {
            a: Fl::<DV, 1>::load(&hi, &lo).0[0],
            wrong,
        },
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        kt,
    );
    let og = sh.off_global();
    sh.extra.push(ExtraRow {
        terms: vec![(og, DD::ONE)],
        rhs: DD::ZERO,
    });
    let mut zh = vec![0.0; sh.len()];
    for i in 0..sh.k() {
        let o = sh.off_node(i) + 16;
        zh[o..o + 8].fill(1.0);
    }
    let o = sh.off_right() + 8;
    zh[o..o + 8].fill(1.0);
    let z = vec![zh, vec![0.0; sh.len()]];
    sh.plan_segments(
        &z[0],
        &z[1],
        &Plan {
            kt,
            tol: 1e-70,
            hmax: 1.0 / 16.0,
        },
    );
    (sh, widen(&z, P::PARTS))
}

/// the state z moved off the branch: every unknown by a deterministic pattern of about 1e-3, and lambda set to 1.5
fn off_branch(sh: &Shooting<Bratu, 1>, z: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut z = z.to_vec();
    for (i, v) in z[0].iter_mut().enumerate() {
        *v += ((i * 7919) % 1000) as f64 * 1e-6;
    }
    z[0][sh.off_global()] = 1.5;
    z
}

/// The family against the layout at an off-branch state in P: whether its residual and Jacobian are the layout's,
/// re-indexed, bit for bit, and dF/dp's largest difference from the layout Jacobian's lambda column, relative to
/// its largest entry
fn view_check<P: Precision<Native>>(sh: &Shooting<Bratu, 1>, z: &[Vec<f64>]) -> (bool, bool, f64) {
    let fam = sh.in_global(0, 0).unwrap();
    let (col, row) = fam.col_row();
    let (zf, p) = fam.family_z::<P>(&parts(z));
    let n = Family::<P>::len(&fam);
    let bits = |x: P::Scalar| (0..P::PARTS).map(|q| P::part(x, q).to_bits()).collect::<Vec<_>>();
    let f = sh.residual::<P>(&parts(z), false).unwrap();
    let ff = Family::<P>::residual_at(&fam, &parts(&zf), p, false).unwrap();
    let skip = |i: usize| i + usize::from(i >= row);
    let res_ok = ff.len() == n && (0..n).all(|i| bits(ff[i]) == bits(f[skip(i)]));
    let (c, ld) = sh.jacobian::<P>(&parts(z), false).unwrap();
    let (cf, ldf) = Family::<P>::jacobian_at(&fam, &parts(&zf), p, false).unwrap();
    let jac_ok = (0..n).all(|j| {
        let jl = j + usize::from(j >= col);
        (0..n).all(|i| (0..P::PARTS).all(|q| cf[q][j * ldf + i].to_bits() == c[q][jl * ld + skip(i)].to_bits()))
    });
    let dp = Family::<P>::dp_at(&fam, &parts(&zf), p, false).unwrap();
    let at = |i: usize| P::from_raw(&(0..P::PARTS).map(|q| c[q][col * ld + skip(i)]).collect::<Vec<_>>());
    let big = (0..n).map(|i| P::to_dd(at(i)).hi.abs()).fold(0.0, nan_max);
    let diff = (0..n).map(|i| P::to_dd(dp[i] - at(i)).hi.abs()).fold(0.0, nan_max);
    (res_ok, jac_ok, diff / big)
}

/// Pass, written before the first run: at an off-branch state, in double-double and quad-double, the family's
/// residual is the layout's without the dropped row and its Jacobian the layout's without that row and lambda's column,
/// bit for bit, and dF/dp (one J v) is the layout Jacobian's lambda column without the row to 1e-30 relative in
/// double-double and 1e-62 in quad-double. The controls: the state's lambda moved by 2^-40 changes dF/dp against the
/// unmoved column by more than 1e-30; and `Control::FamilyColumn` breaks the Jacobian's bits (`controls`).
#[test]
fn family_is_the_layout_reindexed() {
    let (sh, z) = bratu::<Double>(36);
    let zo = off_branch(&sh, &z);
    let (r2, j2, d2) = view_check::<Double>(&sh, &zo);
    let (shq, zq) = bratu::<Quad>(36);
    let zoq = off_branch(&shq, &zq);
    let (r4, j4, d4) = view_check::<Quad>(&shq, &zoq);
    println!("double-double: residual {r2}, Jacobian {j2}, dF/dp {d2:.3e}; quad-double: {r4}, {j4}, {d4:.3e}");
    assert!(r2 && j2 && d2 <= 1e-30, "double-double");
    assert!(r4 && j4 && d4 <= 1e-62, "quad-double");

    // the control: dF/dp at lambda moved by 2^-40 against the column at the unmoved lambda
    let fam = sh.in_global(0, 0).unwrap();
    let (col, row) = fam.col_row();
    let (zf, p) = fam.family_z::<Double>(&parts(&zo));
    let moved = Family::<Double>::dp_at(&fam, &parts(&zf), p + DD::from_f64(2f64.powi(-40)), false).unwrap();
    let (c, ld) = sh.jacobian::<Double>(&parts(&zo), false).unwrap();
    let off = (0..moved.len())
        .map(|i| {
            let il = i + usize::from(i >= row);
            (moved[i] - DD::new(c[0][col * ld + il], c[1][col * ld + il])).hi.abs()
        })
        .fold(0.0, nan_max);
    assert!(off > 1e-30, "control did not fire: {off:e}");
}

/// The fold of the branch from lambda = 0 found by continuation in P: the corrector to max |F| <= `ntol` with the
/// bordered matrices factored by `factor`, stopped at the first point past the fold, which `locate_fold` then
/// refines to |t_p| <= `ftol`. Returns (|lambda - lambda_c|, |u'(0) - 4| on lane 0, the points, the fold's corrections).
fn fold<'a, P: Precision<Native>, F: Factored<P>>(
    fam: &'a InGlobal<'a, Bratu, 1>,
    z: &[Vec<f64>],
    ntol: f64,
    ftol: f64,
    factor: impl Fn(&Bordered<'_, P, InGlobal<'a, Bratu, 1>>, &[&[f64]]) -> Result<F, String>,
) -> (f64, f64, usize, usize) {
    let (zf, p) = fam.family_z::<P>(&parts(z));
    let opts = ArcOpts {
        ds: 0.25,
        ds_min: 1e-6,
        ds_max: 1.0,
        max_points: 200,
        newton: NewtonOpts {
            tol: ntol,
            max_iter: 30,
            ..Default::default()
        },
        ..Default::default()
    };
    let st = start::<P, _, F>(fam, &parts(&zf), p, 1.0, false, &factor).unwrap();
    let path = continuation::<P, _, F>(fam, st, &opts, &factor, |pt| P::to_dd(pt.tp()).hi < 0.0).unwrap();
    let i = *path.folds.first().expect("no fold on the path");
    let fopts = FoldOpts {
        tol: ftol,
        max_iter: 50,
    };
    let (pt, its) = locate_fold::<P, _, F>(fam, &path.points[i], path.ds[i], &fopts, &opts.newton, &factor).unwrap();
    let lam_c = P::from_raw(&LAMBDA_C[..P::PARTS]);
    let p0 = P::from_raw(&(0..P::PARTS).map(|q| pt.z[q][0]).collect::<Vec<_>>());
    let four = P::from_dd(DD::from_f64(4.0));
    (
        P::to_dd(pt.p - lam_c).hi.abs(),
        P::to_dd(p0 - four).hi.abs(),
        path.points.len(),
        its,
    )
}

/// The control of `continuation_passes_the_bratu_fold`: Bratu's problem with its E recursion scaled by 1 + 2^-30
/// (`Bratu::wrong`), whose fold lies about 3.3e-9 below lambda_c with u'(0) about 3.7e-9 below 4, fails both of its
/// checks in double-double and in quad-double, with the exact bordered LU in P.
#[test]
fn moved_fold_fails_the_fold_checks() {
    let (sh, z) = bratu_with::<Double>(36, true);
    let fam = sh.in_global(0, 0).unwrap();
    let (dl, dp, _, _) = fold::<Double, _>(&fam, &z, 1e-28, 1e-15, |b, w| exact::<Double, _>(b, w, false));
    println!("control, double-double: |lambda - lambda_c| {dl:.3e}, |u'(0) - 4| {dp:.3e}");
    assert!(dl > 1e-27 && dp > 1e-12, "control did not fire in double-double");
    let (shq, zq) = bratu_with::<Quad>(72, true);
    let famq = shq.in_global(0, 0).unwrap();
    let (dl, dp, _, _) = fold::<Quad, _>(&famq, &zq, 1e-60, 1e-30, |b, w| exact::<Quad, _>(b, w, false));
    println!("control, quad-double: |lambda - lambda_c| {dl:.3e}, |u'(0) - 4| {dp:.3e}");
    assert!(dl > 1e-57 && dp > 1e-26, "control did not fire in quad-double");
}

/// Pass, written before the first run: continuation from u = 0 at lambda = 0 with the exact bordered LU in P passes
/// the fold, and `locate_fold` puts it at lambda_c within 1e-27 with lane 0's u'(0) within 1e-12 of 4 in
/// double-double (|t_p| <= 1e-15), and within 1e-57 and 1e-26 in quad-double (|t_p| <= 1e-30, Taylor order 72). The
/// control: the moved fold of `moved_fold_fails_the_fold_checks` fails both checks in both precisions.
///
/// The first run failed in quad-double, both errors about 1e-34 with either factors: the recursion formed 1 / (k + 1)
/// as a double-double constant, whose rounding moves the discrete fold. It now forms it in the lane.
#[test]
fn continuation_passes_the_bratu_fold() {
    let (sh, z) = bratu::<Double>(36);
    let fam = sh.in_global(0, 0).unwrap();
    let (dl, dp, pts, its) = fold::<Double, _>(&fam, &z, 1e-28, 1e-15, |b, w| exact::<Double, _>(b, w, false));
    println!("double-double: |lambda - lambda_c| {dl:.3e}, |u'(0) - 4| {dp:.3e}, {pts} points, {its} corrections");
    assert!(dl <= 1e-27 && dp <= 1e-12, "double-double");

    let (shq, zq) = bratu::<Quad>(72);
    let famq = shq.in_global(0, 0).unwrap();
    let (dl, dp, pts, its) = fold::<Quad, _>(&famq, &zq, 1e-60, 1e-30, |b, w| exact::<Quad, _>(b, w, false));
    println!("quad-double: |lambda - lambda_c| {dl:.3e}, |u'(0) - 4| {dp:.3e}, {pts} points, {its} corrections");
    assert!(dl <= 1e-57 && dp <= 1e-26, "quad-double");
}

/// The largest difference, relative to the largest entry, between the solves of a bordered matrix by
/// `bordered_lu::<P, Dtn<4>>` and by the exact LU in P, at an off-branch state with a deterministic unit-norm normal
/// t and right-hand side
fn bordered_error<P: Precision<Native>>(sh: &Shooting<Bratu, 1>, z: &[Vec<f64>]) -> f64 {
    let fam = sh.in_global(0, 0).unwrap();
    let (zf, p) = fam.family_z::<P>(&parts(z));
    let n = Family::<P>::len(&fam);
    let raw: Vec<f64> = (0..=n).map(|i| 1.0 + ((i * 104729) % 97) as f64 / 97.0).collect();
    let nrm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
    let t: Vec<P::Scalar> = raw.iter().map(|x| P::from_dd(DD::from_f64(x / nrm))).collect();
    let mut w: Vec<Vec<f64>> = zf.clone();
    for (q, wq) in w.iter_mut().enumerate() {
        wq.push(P::part(p, q));
    }
    let b = Bordered {
        fam: &fam,
        t,
        pred: vec![P::ZERO; n + 1],
    };
    let r: Vec<P::Scalar> = (0..=n)
        .map(|i| P::from_dd(DD::from_f64(((i * 31) % 17) as f64 - 8.0)))
        .collect();
    let lu = fam.bordered_lu::<P, Dtn<4>>(&b, &parts(&w), false).unwrap();
    let x = Factored::<P>::solve_neg(&lu, &r);
    let y = exact::<P, _>(&b, &parts(&w), false).unwrap().solve_neg(&r);
    let big = y.iter().map(|&v| P::to_dd(v).hi.abs()).fold(0.0, nan_max);
    x.iter()
        .zip(&y)
        .map(|(&a, &c)| P::to_dd(a - c).hi.abs())
        .fold(0.0, nan_max)
        / big
}

/// Pass, written before the first run: `bordered_lu`'s f64 factors solve the bordered matrix as its exact LU in P
/// does, to 1e-10 relative, at an off-branch state in double-double; and continuation with them meets the fold
/// criteria of `continuation_passes_the_bratu_fold` in double-double and quad-double (lambda_c within 1e-27 and
/// 1e-57, u'(0) within 1e-12 and 1e-26 of 4). The controls: `Control::BorderPin`, the dropped row kept in place of
/// the tangent, fails the solve check (`controls`); and with these factors the moved fold of
/// `moved_fold_fails_the_fold_checks` fails both fold checks in both precisions.
#[test]
fn bordered_factors_pass_the_fold() {
    let (sh, z) = bratu::<Double>(36);
    let e = bordered_error::<Double>(&sh, &off_branch(&sh, &z));
    let fam = sh.in_global(0, 0).unwrap();
    let (dl, dp, pts, its) = fold::<Double, _>(&fam, &z, 1e-28, 1e-15, |b, w| {
        fam.bordered_lu::<Double, Dtn<4>>(b, w, false)
    });
    println!(
        "bordered f64 factors: solve {e:.3e}; double-double fold |lambda - lambda_c| {dl:.3e}, |u'(0) - 4| {dp:.3e}, \
         {pts} points, {its} corrections"
    );
    assert!(e <= 1e-10, "solve");
    assert!(dl <= 1e-27 && dp <= 1e-12, "double-double");

    let (shq, zq) = bratu::<Quad>(72);
    let famq = shq.in_global(0, 0).unwrap();
    let (dl, dp, pts, its) = fold::<Quad, _>(&famq, &zq, 1e-60, 1e-30, |b, w| {
        famq.bordered_lu::<Quad, Dtn<4>>(b, w, false)
    });
    println!("quad-double fold, f64 factors: |lambda - lambda_c| {dl:.3e}, |u'(0) - 4| {dp:.3e}, {pts} points, {its}");
    assert!(dl <= 1e-57 && dp <= 1e-26, "quad-double");

    // the controls: the moved fold with the same factors
    let (sh, z) = bratu_with::<Double>(36, true);
    let fam = sh.in_global(0, 0).unwrap();
    let (dl, dp, _, _) = fold::<Double, _>(&fam, &z, 1e-28, 1e-15, |b, w| {
        fam.bordered_lu::<Double, Dtn<4>>(b, w, false)
    });
    println!("control, double-double, f64 factors: |lambda - lambda_c| {dl:.3e}, |u'(0) - 4| {dp:.3e}");
    assert!(dl > 1e-27 && dp > 1e-12, "control did not fire in double-double");
    let (shq, zq) = bratu_with::<Quad>(72, true);
    let famq = shq.in_global(0, 0).unwrap();
    let (dl, dp, _, _) = fold::<Quad, _>(&famq, &zq, 1e-60, 1e-30, |b, w| {
        famq.bordered_lu::<Quad, Dtn<4>>(b, w, false)
    });
    println!("control, quad-double, f64 factors: |lambda - lambda_c| {dl:.3e}, |u'(0) - 4| {dp:.3e}");
    assert!(dl > 1e-57 && dp > 1e-26, "control did not fire in quad-double");
}

/// The controls (feature `mutate`), each against the check it breaks.
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::Control;

    /// the family's Jacobian with the column before lambda's skipped in place of lambda's is not the layout's
    #[test]
    fn family_column_breaks_the_view() {
        let (mut sh, z) = bratu::<Double>(36);
        let zo = off_branch(&sh, &z);
        sh.set_control(Control::FamilyColumn);
        let (r, j, _) = view_check::<Double>(&sh, &zo);
        println!("control, family column: residual {r}, Jacobian {j}");
        assert!(r && !j, "control did not fire");
    }

    /// the bordered factors with the dropped row kept miss the exact solve by more than 1e-10
    #[test]
    fn border_pin_fails_the_solve() {
        let (mut sh, z) = bratu::<Double>(36);
        let zo = off_branch(&sh, &z);
        sh.set_control(Control::BorderPin);
        let e = bordered_error::<Double>(&sh, &zo);
        println!("control, border pin: solve {e:.3e}");
        assert!(e > 1e-10, "control did not fire");
    }
}
