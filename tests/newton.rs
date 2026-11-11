//! The residual, the exact Jacobian and Newton's method in double-double on the planted problem `Planted`, whose
//! solution is manufactured.
mod common;

use common::{as_pointwise, perturbed, Fnv, Planted};
use multishoot::prec::{max_abs, nan_max, parts};
use multishoot::{Double, NewtonOpts, Shooting, DD};

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

fn planted() -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    Planted::<B>::shooting(&NODES, 30, 1e-32)
}

/// The largest |J_ij - (F_i(z + h e_j) - F_i(z - h e_j)) / 2h| over the rows, relative to max(1, max_i |J_ij|),
/// for every column j: (worst ratio, its column)
fn jacobian_vs_central(sh: &Shooting<Planted<B>, B>, z: &[Vec<f64>], h: f64) -> (f64, usize) {
    let l = sh.len();
    let (c, ld) = sh.jacobian::<Double>(&parts(z), true).unwrap();
    let mut worst = (0.0f64, 0usize);
    for j in 0..l {
        let mut zp = z.to_vec();
        let mut zm = z.to_vec();
        let vj = DD::new(z[0][j], z[1][j]);
        let (p, mm) = (vj + DD::from_f64(h), vj - DD::from_f64(h));
        (zp[0][j], zp[1][j]) = (p.hi, p.lo);
        (zm[0][j], zm[1][j]) = (mm.hi, mm.lo);
        let fp = sh.residual::<Double>(&parts(&zp), true).unwrap();
        let fm = sh.residual::<Double>(&parts(&zm), true).unwrap();
        let col: Vec<DD> = (0..l).map(|i| DD::new(c[0][j * ld + i], c[1][j * ld + i])).collect();
        let scale = col.iter().map(|v| v.hi.abs()).fold(1.0, nan_max);
        let err = (0..l)
            .map(|i| (col[i] - (fp[i] - fm[i]) / DD::from_f64(2.0 * h)).hi.abs())
            .fold(0.0, nan_max);
        if err / scale > worst.0 {
            worst = (err / scale, j);
        }
    }
    worst
}

/// The manufactured solution solves the discrete system: pass, max |F(z*)| below 1e-29 (its sources are rounded to
/// double-double, 2^-106 relative, and carried over about 20 steps).
#[test]
fn planted_residual_vanishes() {
    let (sh, z) = planted();
    let f = sh.residual::<Double>(&parts(&z), true).unwrap();
    let r = max_abs::<Double>(&f);
    println!(
        "L = {}, {} steps, max |F(z*)| = {r:.3e}",
        sh.len(),
        sh.segs.iter().map(|s| s.len() - 1).sum::<usize>()
    );
    assert!(r < 1e-29, "{r:e}");
}

/// The exact Jacobian against central differences of the residual, every column, at a point off the solution (z* with
/// a 1e-2 perturbation), h = 2^-30: the differences' truncation is about h^2 = 1e-18 times the third derivative and
/// their rounding 1e-32 / h = 1e-23. Pass: every column within 1e-13 of the differences, relative to max(1, the
/// column's largest entry).
#[test]
fn jacobian_matches_central_differences() {
    let (sh, z) = planted();
    let zp = perturbed(&z, 1e-2);
    let (worst, col) = jacobian_vs_central(&sh, &zp, 2f64.powi(-30));
    println!("Jacobian vs central differences: worst {worst:.3e} at column {col}");
    assert!(worst < 1e-13, "column {col}: {worst:e}");
}

/// Newton from z* perturbed by 1e-3 (relative, every unknown, omega included). Pass: max |F| <= 1e-28 within 10
/// steps, and every unknown within 1e-27 of z* (the discrete solution differs from z* by the sources' rounding times
/// the problem's conditioning).
#[test]
fn planted_newton_converges() {
    let (sh, zs) = planted();
    let mut z = perturbed(&zs, 1e-3);
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 10,
        ..Default::default()
    };
    let rep = sh.newton::<Double>(&mut z, &opts).unwrap();
    let err = (0..sh.len())
        .map(|i| (DD::new(z[0][i], z[1][i]) - DD::new(zs[0][i], zs[1][i])).hi.abs())
        .fold(0.0, nan_max);
    println!(
        "max |F|: {:?}; steps {:?}; max |z - z*| = {err:.3e}",
        rep.fnorm.iter().map(|v| format!("{v:.2e}")).collect::<Vec<_>>(),
        rep.scale
    );
    assert!(rep.converged, "max |F| {:?}", rep.fnorm);
    assert!(err < 1e-27, "{err:e}");
}

/// The Newton step solves J dz = -F: pass, max |J dz + F| below 1e-28 times max |F| (the double-double LU's backward
/// error at this conditioning), at the perturbed point.
#[test]
fn newton_step_solves_the_linear_system() {
    let (sh, z) = planted();
    let zp = perturbed(&z, 1e-3);
    let st = sh.newton_step::<Double>(&parts(&zp), true).unwrap();
    let (c, ld) = sh.jacobian::<Double>(&parts(&zp), true).unwrap();
    let l = sh.len();
    let mut worst = 0.0f64;
    for i in 0..l {
        let mut acc = st.f[i];
        for j in 0..l {
            acc = acc + DD::new(c[0][j * ld + i], c[1][j * ld + i]) * DD::new(st.dz[0][j], st.dz[1][j]);
        }
        worst = worst.max(acc.hi.abs());
    }
    let fmax = max_abs::<Double>(&st.f);
    println!("max |J dz + F| = {worst:.3e}, max |F| = {fmax:.3e}");
    assert!(worst < 1e-28 * fmax, "{worst:e}");
}

/// The pointwise problem (-omega u in place of -omega u_theta) declared pointwise, whose Jacobian takes one dual
/// evaluation per input field (all M columns at once), against the same problem column by column. Pass: every entry
/// equal in every part (as f64 values, so the zeros' signs do not count), at z* perturbed by 1e-2.
#[test]
fn pointwise_jacobian_equals_per_column() {
    let (sh, z) = Planted::<B>::shooting_of(Planted::with_coupling(false), &NODES, 30, 1e-32);
    let zp = perturbed(&z, 1e-2);
    let (c1, _) = sh.jacobian::<Double>(&parts(&zp), true).unwrap();
    let n1 = sh.jac_tasks().len();
    let pw = as_pointwise(sh);
    let (c2, _) = pw.jacobian::<Double>(&parts(&zp), true).unwrap();
    let n2 = pw.jac_tasks().len();
    let differ = c1
        .iter()
        .zip(&c2)
        .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
        .sum::<usize>();
    println!("pointwise Jacobian: {n2} evaluations instead of {n1}, {differ} entries differ");
    assert_eq!(differ, 0);
}

/// The control of `pointwise_jacobian_equals_per_column`: the coupled problem declared pointwise must give a
/// different Jacobian (some entry off by more than 1e-3), since d/dtheta carries a point's tangent to the others.
#[test]
fn pointwise_control_coupled_problem_differs() {
    let (sh, z) = planted();
    let zp = perturbed(&z, 1e-2);
    let (c1, _) = sh.jacobian::<Double>(&parts(&zp), true).unwrap();
    let (c2, _) = as_pointwise(sh).jacobian::<Double>(&parts(&zp), true).unwrap();
    let worst = c1[0].iter().zip(&c2[0]).map(|(x, y)| (x - y).abs()).fold(0.0, nan_max);
    println!("control, coupled problem declared pointwise: largest difference {worst:.3e}");
    assert!(worst > 1e-3, "control did not fire: {worst:e}");
}

/// The fingerprints of the double-double results: the residual at z* perturbed by 1e-3, the Newton solution from
/// there, and the pointwise problem's batched Jacobian at z* perturbed by 1e-2; `ulp` changes the last bit of
/// unknown 37's hi part in both starting points, and `ulp_rhs` that of the extra row's omega*
fn results_hashes(ulp: bool, ulp_rhs: bool) -> [u64; 3] {
    let bump = |z: &mut Vec<Vec<f64>>| z[0][37] = f64::from_bits(z[0][37].to_bits() + 1);
    let (mut sh, zs) = planted();
    if ulp_rhs {
        sh.extra[0].rhs.hi = f64::from_bits(sh.extra[0].rhs.hi.to_bits() + 1);
    }
    let mut z = perturbed(&zs, 1e-3);
    if ulp {
        bump(&mut z);
    }
    let f = sh.residual::<Double>(&parts(&z), true).unwrap();
    let mut hf = Fnv::new();
    f.iter().for_each(|v| {
        hf.f64(v.hi);
        hf.f64(v.lo)
    });
    sh.newton::<Double>(
        &mut z,
        &NewtonOpts {
            tol: 1e-28,
            max_iter: 10,
            ..Default::default()
        },
    )
    .unwrap();
    let mut hz = Fnv::new();
    z.iter().flatten().for_each(|&v| hz.f64(v));
    let (pw, zs) = Planted::<B>::shooting_of(Planted::with_coupling(false), &NODES, 30, 1e-32);
    let mut zp = perturbed(&zs, 1e-2);
    if ulp {
        bump(&mut zp);
    }
    let (c, _) = as_pointwise(pw).jacobian::<Double>(&parts(&zp), true).unwrap();
    let mut hj = Fnv::new();
    c.iter().flatten().for_each(|&v| hj.f64(v));
    [hf.0, hz.0, hj.0]
}

/// the reference fingerprints, recorded on aarch64 (NEON) and on the portable backend
const RESULTS_REFERENCE: [u64; 3] = [0x4930892ca98ba844, 0xb6c4584a17a30dbb, 0x48b1fe26a0d6c709];

/// The residual, the Newton solution and the batched Jacobian do not depend on multidouble's SIMD backend or on the
/// number of threads. Pass: `results_hashes` equals the reference on every build of the CI matrix.
#[test]
fn results_bytes_match_reference() {
    let h = results_hashes(false, false);
    println!(
        "results hashes [{}]",
        h.iter().map(|v| format!("{v:#018x}")).collect::<Vec<_>>().join(", ")
    );
    assert_eq!(h, RESULTS_REFERENCE, "the results' bytes differ from the reference");
}

/// The control of `results_bytes_match_reference`: one ulp of one unknown must change the residual's and the
/// Jacobian's fingerprints, and one ulp of omega* the Newton solution's.
#[test]
fn results_bytes_control_one_ulp_differs() {
    let base = results_hashes(false, false);
    let h = results_hashes(true, false);
    assert!(h[0] != base[0] && h[2] != base[2], "control did not fire");
    let h = results_hashes(false, true);
    assert_ne!(h[1], base[1], "control did not fire");
}

/// The controls (feature `mutate`): a dropped Jacobian column must fail `jacobian_matches_central_differences`, and
/// a shifted mismatch row must fail `planted_newton_converges`.
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::Control;

    #[test]
    fn dropped_column_fails_central_differences() {
        let (mut sh, z) = planted();
        let col = sh.off_node(2) + 5;
        sh.set_control(Control::DropColumn(col));
        let zp = perturbed(&z, 1e-2);
        let (worst, at) = jacobian_vs_central(&sh, &zp, 2f64.powi(-30));
        println!("control, column {col} dropped: worst {worst:.3e} at column {at}");
        assert!(worst >= 1e-13 && at == col, "control did not fire: {worst:e} at {at}");
    }

    #[test]
    fn shifted_row_fails_newton() {
        let (mut sh, zs) = planted();
        sh.set_control(Control::ShiftRow(sh.row_block(3) + 7));
        let mut z = perturbed(&zs, 1e-3);
        let opts = NewtonOpts {
            tol: 1e-28,
            max_iter: 10,
            ..Default::default()
        };
        let rep = sh.newton::<Double>(&mut z, &opts).unwrap();
        let err = (0..sh.len())
            .map(|i| (DD::new(z[0][i], z[1][i]) - DD::new(zs[0][i], zs[1][i])).hi.abs())
            .fold(0.0, nan_max);
        println!(
            "control, shifted row: converged {}, max |F| {:.3e}, max |z - z*| = {err:.3e}",
            rep.converged,
            rep.fnorm.last().unwrap()
        );
        assert!(!rep.converged || err >= 1e-27, "control did not fire");
    }
}
