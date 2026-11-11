//! A global unknown with a phase condition, and the residual's series in a parameter: the eigenproblem `Cylinder`,
//! whose lowest eigenvalue is pi^2 + mu with the mode sin(pi x) cos theta / pi.
mod common;

use common::{perturbed, wave, Cylinder, Fnv};
use multishoot::prec::{max_abs, parts, widen};
use multishoot::{Double, Native, NewtonOpts, Octo, Precision, Quad, Shooting, DD};

const B: usize = 1;
const KT: usize = 30;

type Sh = Shooting<Cylinder<B>, B>;

fn opts(tol: f64, max_iter: usize) -> NewtonOpts {
    NewtonOpts {
        tol,
        max_iter,
        ..Default::default()
    }
}

/// Newton in double-double from the mode perturbed by 1e-3, then in quad-double: (converged in dd, max |F| in qd,
/// |lambda - (pi^2 + 1)|, |nu|, max |z - z*|), the last three in quad-double
fn solve(sh: &Sh, zs: &[Vec<f64>]) -> (bool, f64, f64, f64, f64) {
    let mut z = perturbed(zs, 1e-3);
    let dd = sh.newton::<Double>(&mut z, &opts(1e-28, 12)).unwrap();
    let mut zq = widen(&z, 4);
    let qd = sh.newton::<Quad>(&mut zq, &opts(1e-60, 4)).unwrap();
    let og = sh.off_global();
    let q = |i: usize| <Quad as Precision<Native>>::new(&[zq[0][i], zq[1][i], zq[2][i], zq[3][i]]);
    let lam_err = (q(og) - <Quad as Precision<Native>>::from_dd(sh.eq.lambda()))
        .to_dd()
        .hi
        .abs();
    let nu = q(og + 1).to_dd().hi.abs();
    let zerr = (0..sh.len())
        .map(|i| (q(i).to_dd() - DD::new(zs[0][i], zs[1][i])).hi.abs())
        .fold(0.0, f64::max);
    let fq = *qd.fnorm.last().unwrap();
    println!(
        "dd max |F| {:?}, qd {:?}; |lambda - (pi^2 + 1)| {lam_err:.3e}, |nu| {nu:.3e}, max |z - z*| {zerr:.3e}",
        dd.fnorm.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>(),
        qd.fnorm.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>()
    );
    (dd.converged, fq, lam_err, nu, zerr)
}

/// whether `solve`'s results pass: dd converged, qd max |F| below 1e-60, and lambda, nu and the mode within 1e-28 of
/// their exact values (the discrete problem's constants are double-double: pi^2, D, the steps' truncation)
fn passes(r: (bool, f64, f64, f64, f64)) -> bool {
    r.0 && r.1 < 1e-60 && r.2 < 1e-28 && r.3 < 1e-28 && r.4 < 1e-28
}

/// The eigenpair with two globals (lambda, nu) and two extra rows (amplitude, phase), from the mode perturbed by
/// 1e-3: Newton in double-double, then quad-double. Pass: `passes`.
#[test]
fn eigenpair_converges_in_double_and_quad() {
    let (sh, zs) = Cylinder::<B>::shooting(Cylinder::new(), KT);
    let r = solve(&sh, &zs);
    assert!(passes(r), "{r:?}");
}

/// The control of the eigenvalue: lambda's coefficient in the equations off by 2^-50 must move it off pi^2 + 1.
#[test]
fn eigenpair_control_wrong_coefficient_fails() {
    let mut eq = Cylinder::<B>::new();
    eq.wrong_lambda = true;
    let (sh, zs) = Cylinder::<B>::shooting(eq, KT);
    assert!(!passes(solve(&sh, &zs)), "control did not fire");
}

/// The control of the phase condition: a phase row along the amplitude's direction (cos theta weights, consistent
/// with the amplitude row) leaves the shifts in theta free, and the solve must not pass.
#[test]
fn eigenpair_control_phase_row_along_amplitude_fails() {
    let (mut sh, zs) = Cylinder::<B>::shooting(Cylinder::new(), KT);
    sh.extra[1] = multishoot::ExtraRow {
        terms: sh.extra[0].terms.clone(),
        rhs: DD::ONE,
    };
    let mut z = perturbed(&zs, 1e-3);
    let res = sh.newton::<Double>(&mut z, &opts(1e-28, 12));
    let ok = match res {
        Err(e) => {
            println!("control: {e}");
            false
        }
        Ok(_) => passes(solve(&sh, &zs)),
    };
    assert!(!ok, "control did not fire");
}

/// a direction for z_1: every unknown moved by up to 0.1, from `wave` (the same on every platform)
fn direction(l: usize) -> Vec<Vec<f64>> {
    vec![(0..l).map(|i| 0.1 * wave(i + 1000)).collect(), vec![0.0; l]]
}

/// F(z + h v; mu + h) for the shooting problem
fn resid_along(sh: &mut Sh, z: &[Vec<f64>], v: &[Vec<f64>], h: f64) -> Vec<DD> {
    let mu0 = sh.eq.mu;
    sh.eq.mu = mu0 + DD::from_f64(h);
    let zh: Vec<Vec<f64>> = {
        let n = z[0].len();
        let s: Vec<DD> = (0..n)
            .map(|i| DD::new(z[0][i], z[1][i]) + DD::new(v[0][i], v[1][i]).mul_f(h))
            .collect();
        vec![s.iter().map(|x| x.hi).collect(), s.iter().map(|x| x.lo).collect()]
    };
    let f = sh.residual::<Double>(&parts(&zh), true).unwrap();
    sh.eq.mu = mu0;
    f
}

/// The jet's orders 0, 1 and 2 at (z, v) against the residual and central differences along z + h v, mu + h: the
/// largest relative deviations (order 0 absolute)
fn jet_vs_differences(sh: &mut Sh, z: &[Vec<f64>], v: &[Vec<f64>]) -> (f64, f64, f64) {
    let jet = sh.residual_jet::<Double, 3>(&[parts(z), parts(v)], true, 3).unwrap();
    let f0 = sh.residual::<Double>(&parts(z), true).unwrap();
    let d0 = jet[0]
        .iter()
        .zip(&f0)
        .map(|(a, b)| (*a - *b).hi.abs())
        .fold(0.0, f64::max);
    let h1 = 2f64.powi(-30);
    let (p, m) = (resid_along(sh, z, v, h1), resid_along(sh, z, v, -h1));
    let s1 = jet[1].iter().map(|x| x.hi.abs()).fold(1.0, f64::max);
    let d1 = (0..p.len())
        .map(|i| (jet[1][i] - (p[i] - m[i]) / DD::from_f64(2.0 * h1)).hi.abs())
        .fold(0.0, f64::max);
    let h2 = 2f64.powi(-14);
    let (p, m) = (resid_along(sh, z, v, h2), resid_along(sh, z, v, -h2));
    let s2 = jet[2].iter().map(|x| x.hi.abs()).fold(1.0, f64::max);
    let d2 = (0..p.len())
        .map(|i| {
            (jet[2][i] - (p[i] - f0[i].mul_f(2.0) + m[i]) / DD::from_f64(2.0 * h2 * h2))
                .hi
                .abs()
        })
        .fold(0.0, f64::max);
    println!(
        "jet against differences: order 0 {d0:.3e}, order 1 {:.3e}, order 2 {:.3e}",
        d1 / s1,
        d2 / s2
    );
    (d0, d1 / s1, d2 / s2)
}

/// The residual's jet in mu, with z_1 a direction v, at the mode perturbed by 1e-3. Pass: order 0 within 1e-30 of
/// the residual; order 1 within 1e-13 of the central difference along (z + h v, mu + h), h = 2^-30 (truncation about
/// h^2 = 1e-18); order 2 within 1e-7 of the second difference, h = 2^-14 (truncation h^2 = 4e-9), both relative to
/// max(1, the order's largest coefficient).
#[test]
fn jet_orders_match_central_differences() {
    let (mut sh, zs) = Cylinder::<B>::shooting(Cylinder::new(), KT);
    let z = perturbed(&zs, 1e-3);
    let v = direction(sh.len());
    let (d0, d1, d2) = jet_vs_differences(&mut sh, &z, &v);
    assert!(d0 < 1e-30 && d1 < 1e-13 && d2 < 1e-7, "{d0:e} {d1:e} {d2:e}");
}

/// The control of `jet_orders_match_central_differences`: mu entered as a constant instead of the parameter must
/// break order 1 (above 1e-13).
#[test]
fn jet_control_frozen_parameter_fails() {
    let mut eq = Cylinder::<B>::new();
    eq.freeze_mu = true;
    let (mut sh, zs) = Cylinder::<B>::shooting(eq, KT);
    let z = perturbed(&zs, 1e-3);
    let v = direction(sh.len());
    // the differences move mu in the equations, which the frozen jet ignores
    sh.eq.freeze_mu = false;
    let f_pm = (resid_along(&mut sh, &z, &v, 1e-9), resid_along(&mut sh, &z, &v, -1e-9));
    sh.eq.freeze_mu = true;
    let jet = sh.residual_jet::<Double, 2>(&[parts(&z), parts(&v)], true, 2).unwrap();
    let d1 = (0..jet[1].len())
        .map(|i| (jet[1][i] - (f_pm.0[i] - f_pm.1[i]) / DD::from_f64(2e-9)).hi.abs())
        .fold(0.0, f64::max);
    println!("control, frozen mu: order 1 off by {d1:.3e}");
    assert!(d1 > 1e-13, "control did not fire: {d1:e}");
}

/// The jet in the three precisions at a double-double (z, v): Jet against QJet to 1e-28 and QJet against OJet to
/// 1e-60, every order, relative to max(1, |coefficient|).
#[test]
fn jet_precisions_agree() {
    let (sh, zs) = Cylinder::<B>::shooting(Cylinder::new(), KT);
    let z = perturbed(&zs, 1e-3);
    let v = direction(sh.len());
    let jd = sh.residual_jet::<Double, 3>(&[parts(&z), parts(&v)], true, 3).unwrap();
    let jq = sh
        .residual_jet::<Quad, 3>(&[parts(&widen(&z, 4)), parts(&widen(&v, 4))], true, 3)
        .unwrap();
    let jo = sh
        .residual_jet::<Octo, 3>(&[parts(&widen(&z, 8)), parts(&widen(&v, 8))], true, 3)
        .unwrap();
    let (mut dq, mut qo) = (0.0f64, 0.0f64);
    for k in 0..3 {
        for i in 0..jd[k].len() {
            let s = jq[k][i].to_dd().hi.abs().max(1.0);
            dq = dq.max((jd[k][i] - jq[k][i].to_dd()).hi.abs() / s);
            let d = <Octo as Precision<Native>>::new(&jq[k][i].x) - jo[k][i];
            qo = qo.max(max_abs::<Octo>(&[d]) / s);
        }
    }
    println!("jet: |dd - qd| {dq:.3e}, |qd - od| {qo:.3e}");
    assert!(dq < 1e-28 && qo < 1e-60, "{dq:e} {qo:e}");
}

/// the fingerprints of the jet in dd, qd and od at (z, v), `ulp` changing one unknown of z by one ulp
fn jet_hashes(ulp: bool) -> [u64; 3] {
    let (sh, zs) = Cylinder::<B>::shooting(Cylinder::new(), KT);
    let mut z = perturbed(&zs, 1e-3);
    if ulp {
        z[0][11] = f64::from_bits(z[0][11].to_bits() + 1);
    }
    let v = direction(sh.len());
    let mut out = [0u64; 3];
    let mut h = Fnv::new();
    for c in sh
        .residual_jet::<Double, 3>(&[parts(&z), parts(&v)], true, 3)
        .unwrap()
        .iter()
        .flatten()
    {
        h.f64(c.hi);
        h.f64(c.lo);
    }
    out[0] = h.0;
    let mut h = Fnv::new();
    for c in sh
        .residual_jet::<Quad, 3>(&[parts(&widen(&z, 4)), parts(&widen(&v, 4))], true, 3)
        .unwrap()
        .iter()
        .flatten()
    {
        c.x.iter().for_each(|&p| h.f64(p));
    }
    out[1] = h.0;
    let mut h = Fnv::new();
    for c in sh
        .residual_jet::<Octo, 3>(&[parts(&widen(&z, 8)), parts(&widen(&v, 8))], true, 3)
        .unwrap()
        .iter()
        .flatten()
    {
        c.x.iter().for_each(|&p| h.f64(p));
    }
    out[2] = h.0;
    out
}

/// the reference fingerprints, recorded on aarch64 (NEON) and on the portable backend
const JET_REFERENCE: [u64; 3] = [0x1c1e0f05043cc0ea, 0xc773b58691b23171, 0x8b6a08899263ffbf];

/// The jet's bytes in every precision do not depend on multidouble's SIMD backend. Pass: `jet_hashes` equals the
/// reference on every build of the CI matrix.
#[test]
fn jet_bytes_match_reference() {
    let h = jet_hashes(false);
    println!(
        "jet hashes [{}]",
        h.iter().map(|v| format!("{v:#018x}")).collect::<Vec<_>>().join(", ")
    );
    assert_eq!(h, JET_REFERENCE, "the jet's bytes differ from the reference");
}

/// The control of `jet_bytes_match_reference`: one ulp of one unknown must change every fingerprint.
#[test]
fn jet_bytes_control_one_ulp_differs() {
    let (a, b) = (jet_hashes(false), jet_hashes(true));
    assert!((0..3).all(|k| a[k] != b[k]), "control did not fire");
}
