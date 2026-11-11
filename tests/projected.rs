//! Projected rows: the projector against harmonics sampled from cosines that share nothing with the engine's, and the
//! planted problem with its right block's v rows projected, solved to octo-double against the exact rows.
mod common;

use common::{perturbed, Fnv, Planted};
use multishoot::multidouble::od::{OdAcc, OD};
use multishoot::periodic::Parity;
use multishoot::prec::{max_abs, nan_max, parts, widen};
use multishoot::{Double, Native, NewtonOpts, Octo, Precision, ProjectedRows, Quad, Shooting, DD};

const B: usize = 2;
const M: usize = 8 * B;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];
/// the highest harmonic of v kept: v* has harmonics 0 and 2
const KMAX: usize = 2;

/// cos(t pi / m) for t < 2m, independently of the engine's sines and cosines and of pi: c = cos(pi / 2m) is the
/// largest root of the Chebyshev polynomial T_m, found by Newton in octo-double (T_m' = m U_(m-1)) from libm's f64
/// value, and cos(t pi / m) = T_2t(c) by the three-term recurrence T_(n+1) = 2 c T_n - T_(n-1)
fn chebyshev_cosines(m: usize) -> Vec<OD> {
    let one = OD::from_f64(1.0);
    let mut c = OD::from_f64((std::f64::consts::PI / (2.0 * m as f64)).cos());
    for _ in 0..6 {
        let (mut t, mut tp) = (c, one);
        let (mut u, mut up) = (c.mul_f(2.0), one);
        for _ in 1..m {
            (t, tp) = (c.mul_f(2.0) * t - tp, t);
        }
        for _ in 1..m - 1 {
            (u, up) = (c.mul_f(2.0) * u - up, u);
        }
        c = c - t / (u.mul_f(m as f64));
    }
    let mut tn = vec![one, c];
    for n in 1..4 * m {
        tn.push(c.mul_f(2.0) * tn[n] - tn[n - 1]);
    }
    (0..2 * m).map(|t| tn[2 * t]).collect()
}

/// the harmonics of the grid of `parity` on m points, (k, kept by the projector onto k <= kmax), and the angle step
/// of theta_j in units of pi / m (2 on the full grid, 1 on the half grids)
fn grid_harmonics(parity: Parity, m: usize, kmax: usize) -> (Vec<(usize, bool)>, usize) {
    let ks: Vec<usize> = match parity {
        Parity::Full => (0..=m / 2).collect(),
        Parity::Odd => (1..m).step_by(2).collect(),
        Parity::Even => (0..=m).step_by(2).collect(),
    };
    (
        ks.into_iter().map(|k| (k, k <= kmax)).collect(),
        if parity == Parity::Full { 2 } else { 1 },
    )
}

/// The samples of cos(k theta_j) and sin(k theta_j) on the grid (sin x = cos(x - pi / 2), pi / 2 = (m / 2) pi / m),
/// the sine left out where it vanishes on the grid (k = 0 and the Nyquist harmonic)
fn harmonic_samples(cs: &[OD], m: usize, k: usize, step: usize) -> Vec<Vec<OD>> {
    let p = 2 * m as i64;
    let cosine: Vec<OD> = (0..m).map(|j| cs[((k * j * step) as i64 % p) as usize]).collect();
    let sine: Vec<OD> = (0..m)
        .map(|j| cs[(((k * j * step) as i64 - m as i64 / 2).rem_euclid(p)) as usize])
        .collect();
    if sine.iter().all(|v| v.to_f64().abs() < 1e-100) {
        vec![cosine]
    } else {
        vec![cosine, sine]
    }
}

/// The largest error of the projector (entries read as `dd` asks) on the grid's harmonics: |Pi h - h| for kept
/// harmonics, |Pi h| for dropped ones, in octo-double
fn projector_error(parity: Parity, m: usize, kmax: usize, dd: bool) -> f64 {
    let p = ProjectedRows::new(m, 0, 0, 0, parity, kmax);
    let cs = chebyshev_cosines(m);
    let (hs, step) = grid_harmonics(parity, m, kmax);
    let mut worst = 0.0f64;
    for (k, kept) in hs {
        for h in harmonic_samples(&cs, m, k, step) {
            let ph = p.apply::<Octo>(&h, None, dd);
            for j in 0..m {
                let want = if kept { h[j] } else { OD::from_f64(0.0) };
                worst = nan_max(worst, (ph[j] - want).to_f64().abs());
            }
        }
    }
    worst
}

/// the grids and cutoffs of the projector checks
const CASES: [(Parity, usize, usize); 7] = [
    (Parity::Odd, 16, 3),
    (Parity::Odd, 8, 5),
    (Parity::Even, 16, 2),
    (Parity::Even, 16, 16),
    (Parity::Full, 16, 0),
    (Parity::Full, 16, 5),
    (Parity::Full, 8, 4),
];

/// The projector in octo-double reproduces each kept harmonic of its grid and annihilates each dropped one, against
/// samples of cos and sin from Chebyshev roots (no pi, no series), on the three grids, with the Nyquist harmonics
/// dropped and kept. Pass: every error below 1e-124.
#[test]
fn projector_is_exact_in_octo() {
    for (parity, m, kmax) in CASES {
        let e = projector_error(parity, m, kmax, false);
        println!("{parity:?} M = {m}, kmax = {kmax}: projector error {e:.3e}");
        assert!(e < 1e-124, "{parity:?} {m} {kmax}: {e:e}");
    }
}

/// The control of `projector_is_exact_in_octo`: the entries rounded to double-double (their first two parts) must
/// fail the bound on every grid whose projector has entries that are not dyadic (a projector that keeps every
/// harmonic, the identity, or only the constant, 1 / M, is exact in any precision).
#[test]
fn projector_control_double_double_fails() {
    for (parity, m, kmax) in CASES {
        let e = projector_error(parity, m, kmax, true);
        println!("control, {parity:?} M = {m}, kmax = {kmax}: projector error {e:.3e}");
        let dyadic = kmax == 0 || kmax >= if parity == Parity::Full { m / 2 } else { m - 1 };
        assert!(
            dyadic || e >= 1e-124,
            "control did not fire: {parity:?} {m} {kmax}: {e:e}"
        );
    }
}

/// The planted problem with the right block's v rows projected: Pi onto the even harmonics k <= 2 (v*'s), and the
/// left data v held to Pi' in place of the high harmonics' matching. z* still solves it (its v has no high
/// harmonics), as does its unprojected copy.
fn projected() -> (Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    let (mut sh, z) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let k = sh.k();
    sh.project_rows(k, 1, 0, Parity::Even, KMAX).unwrap();
    (sh, z)
}

fn opts(tol: f64, max_iter: usize) -> NewtonOpts {
    NewtonOpts {
        tol,
        max_iter,
        ..Default::default()
    }
}

/// The exact projected rows at z (octo-double parts): Pi d + Pi' y with d the right block's v mismatch (from the
/// unprojected problem's residual), y the left data, and Pi from the Chebyshev cosines; their largest magnitude
fn exact_rows(z: &[Vec<f64>]) -> f64 {
    let (plain, _) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let f = plain.residual::<Octo>(&parts(z), true).unwrap();
    let r0 = plain.row_block(plain.k()) + M;
    let cs = chebyshev_cosines(M);
    let (hs, _) = grid_harmonics(Parity::Even, M, KMAX);
    let pi = |i: usize, j: usize| {
        let l = i.abs_diff(j);
        let s = hs.iter().filter(|h| h.1).fold(OD::from_f64(0.0), |a, &(k, _)| {
            a + cs[(k * l) % (2 * M)].mul_f(if k == 0 || k == M { 1.0 } else { 2.0 })
        });
        s / OD::from_f64(M as f64)
    };
    let y = |j: usize| <Octo as Precision<Native>>::from_raw(&z.iter().map(|p| p[j]).collect::<Vec<_>>());
    (0..M)
        .map(|i| {
            let mut acc = OdAcc::zero();
            for j in 0..M {
                let pij = pi(i, j);
                let qij = if i == j { OD::from_f64(1.0) - pij } else { -pij };
                acc = acc.add_prod(pij, f[r0 + j]).add_prod(qij, y(j));
            }
            acc.finish().to_f64().abs()
        })
        .fold(0.0, nan_max)
}

/// Newton in double-, quad- and octo-double from z* perturbed by 1e-3: (the three max |F|, max |z_od - z*|, the exact
/// rows at z_od)
fn solve(sh: &Shooting<Planted<B>, B>, zs: &[Vec<f64>]) -> ([f64; 3], f64, f64) {
    let mut z = perturbed(zs, 1e-3);
    let dd = sh.newton::<Double>(&mut z, &opts(1e-28, 10)).unwrap();
    let mut zq = widen(&z, 4);
    let qd = sh.newton::<Quad>(&mut zq, &opts(1e-60, 4)).unwrap();
    let mut zo = widen(&zq, 8);
    let od = sh.newton::<Octo>(&mut zo, &opts(1e-110, 4)).unwrap();
    let zw = widen(zs, 8);
    let d: Vec<_> = (0..sh.len())
        .map(|i| {
            let a = |z: &[Vec<f64>]| <Octo as Precision<Native>>::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>());
            a(&zo) - a(&zw)
        })
        .collect();
    let f = [
        *dd.fnorm.last().unwrap(),
        *qd.fnorm.last().unwrap(),
        *od.fnorm.last().unwrap(),
    ];
    (f, max_abs::<Octo>(&d), exact_rows(&zo))
}

/// The projected problem solved to octo-double, the rows checked with the exact projector. Pass: max |F| below 1e-28,
/// 1e-60 and 1e-110 in double-, quad- and octo-double, the octo-double root within 1e-27 of z* (the sources'
/// rounding), and the exact projected rows there below 1e-115.
#[test]
fn projected_newton_reaches_the_exact_rows() {
    let (sh, zs) = projected();
    let (f, dz, rows) = solve(&sh, &zs);
    println!(
        "projected: max |F| dd {:.2e}, qd {:.2e}, od {:.2e}; |z - z*| {dz:.2e}; exact rows {rows:.3e}",
        f[0], f[1], f[2]
    );
    assert!(f[0] < 1e-28 && f[1] < 1e-60 && f[2] < 1e-110, "{f:?}");
    assert!(dz < 1e-27, "{dz:e}");
    assert!(rows < 1e-115, "{rows:e}");
}

/// For each column j: the largest |J_ij - (F_i(z + h e_j) - F_i(z - h e_j)) / 2h| over the projected rows i,
/// relative to max(1, max_i |J_ij|), J the Jacobian of `jac` and F the projected problem's residual: (worst, column)
fn projected_vs_central(jac: &Shooting<Planted<B>, B>, z: &[Vec<f64>], h: f64) -> (f64, usize) {
    let (sh, _) = projected();
    let (c, ld) = jac.jacobian::<Double>(&parts(z), true).unwrap();
    let r0 = sh.row_block(sh.k()) + M;
    let mut worst = (0.0f64, 0usize);
    for j in 0..sh.len() {
        let (mut zp, mut zm) = (z.to_vec(), z.to_vec());
        let vj = DD::new(z[0][j], z[1][j]);
        let (p, mm) = (vj + DD::from_f64(h), vj - DD::from_f64(h));
        (zp[0][j], zp[1][j]) = (p.hi, p.lo);
        (zm[0][j], zm[1][j]) = (mm.hi, mm.lo);
        let fp = sh.residual::<Double>(&parts(&zp), true).unwrap();
        let fm = sh.residual::<Double>(&parts(&zm), true).unwrap();
        let col = |i: usize| DD::new(c[0][j * ld + i], c[1][j * ld + i]);
        let scale = (r0..r0 + M).map(|i| col(i).hi.abs()).fold(1.0, nan_max);
        let err = (r0..r0 + M)
            .map(|i| (col(i) - (fp[i] - fm[i]) / DD::from_f64(2.0 * h)).hi.abs())
            .fold(0.0, nan_max);
        if err / scale > worst.0 {
            worst = (err / scale, j);
        }
    }
    worst
}

/// The projected rows of the Jacobian (-Pi on the target node's v, Pi' on the left data) against central differences
/// of the projected residual, h = 2^-30, at z* perturbed by 1e-2, every column. Pass: within 1e-13, relative.
#[test]
fn projected_jacobian_matches_central_differences() {
    let (sh, zs) = projected();
    let (worst, col) = projected_vs_central(&sh, &perturbed(&zs, 1e-2), 2f64.powi(-30));
    println!("projected Jacobian rows against central differences: worst {worst:.3e} at column {col}");
    assert!(worst < 1e-13, "column {col}: {worst:e}");
}

/// The control of `projected_jacobian_matches_central_differences`: the unprojected problem's Jacobian must fail it.
#[test]
fn projected_jacobian_control_unprojected_fails() {
    let (plain, zs) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let (worst, col) = projected_vs_central(&plain, &perturbed(&zs, 1e-2), 2f64.powi(-30));
    println!("control, unprojected Jacobian: worst {worst:.3e} at column {col}");
    assert!(worst >= 1e-13, "control did not fire: {worst:e}");
}

/// max_i |(J v)_i - jet_1,i| over the rows, relative to max_i sum_j |J_ij| |v_j|: J the Jacobian of `jac` at z, jet_1
/// the projected problem's residual series' order 1 on z + h v (the planted equations have no parameter, so order 1
/// is J v)
fn jet_against_jacobian(jac: &Shooting<Planted<B>, B>, z: &[Vec<f64>], v: &[Vec<f64>]) -> f64 {
    let (sh, _) = projected();
    let l = sh.len();
    let jet = sh.residual_jet::<Double, 2>(&[parts(z), parts(v)], true, 2).unwrap();
    let (c, ld) = jac.jacobian::<Double>(&parts(z), true).unwrap();
    let (mut scale, mut err) = (0.0f64, 0.0f64);
    for i in 0..l {
        let mut acc = DD::ZERO;
        let mut s = 0.0;
        for j in 0..l {
            let cij = DD::new(c[0][j * ld + i], c[1][j * ld + i]);
            acc = acc + cij * DD::new(v[0][j], v[1][j]);
            s += cij.hi.abs() * v[0][j].abs();
        }
        scale = nan_max(scale, s);
        err = nan_max(err, (acc - jet[1][i]).hi.abs());
    }
    err / scale
}

/// a direction with every unknown in it
fn direction(l: usize) -> Vec<Vec<f64>> {
    vec![(0..l).map(|i| common::wave(i + 11)).collect(), vec![0.0; l]]
}

/// The projected residual's series in a direction: order 1 against the projected Jacobian times v, at z* perturbed by
/// 1e-2. Pass: within 1e-28 of max_i sum_j |J_ij| |v_j|.
#[test]
fn projected_jet_is_the_jacobian_product() {
    let (sh, zs) = projected();
    let z = perturbed(&zs, 1e-2);
    let e = jet_against_jacobian(&sh, &z, &direction(sh.len()));
    println!("projected jet order 1 against J v: {e:.3e}");
    assert!(e < 1e-28, "{e:e}");
}

/// The control of `projected_jet_is_the_jacobian_product`: the unprojected Jacobian's product must fail it.
#[test]
fn projected_jet_control_unprojected_fails() {
    let (plain, zs) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let z = perturbed(&zs, 1e-2);
    let e = jet_against_jacobian(&plain, &z, &direction(plain.len()));
    println!("control, unprojected J v: {e:.3e}");
    assert!(e >= 1e-28, "control did not fire: {e:e}");
}

/// FNV-1a over every part of the scalars
fn fnv<P: Precision<Native>>(f: &[P::Scalar]) -> u64 {
    let mut h = Fnv::new();
    f.iter().for_each(|&x| (0..P::PARTS).for_each(|q| h.f64(P::part(x, q))));
    h.0
}

/// The octo-double parts of z_i (1 + 1e-3 / (3 + i))^2, every part in use
fn full_parts(z: &[f64]) -> Vec<Vec<f64>> {
    type O = Octo;
    let mut out: Vec<Vec<f64>> = (0..8).map(|_| vec![0.0; z.len()]).collect();
    for (i, &x) in z.iter().enumerate() {
        let r = <O as Precision<Native>>::ONE
            + <O as Precision<Native>>::new(&[1e-3]) * <O as Precision<Native>>::new(&[1.0 / (3.0 + i as f64)]);
        let v = <O as Precision<Native>>::new(&[x]) * r * r;
        for (q, o) in out.iter_mut().enumerate() {
            o[i] = <O as Precision<Native>>::part(v, q);
        }
    }
    out
}

/// The hashes of the projected residual in double-, quad- and octo-double and of its double-double Jacobian
fn hashes(z8: &[Vec<f64>]) -> [u64; 4] {
    let (sh, _) = projected();
    let z = parts(z8);
    let (c, _) = sh.jacobian::<Double>(&z[..2], true).unwrap();
    let mut hj = Fnv::new();
    c.iter().flatten().for_each(|&v| hj.f64(v));
    [
        fnv::<Double>(&sh.residual::<Double>(&z[..2], true).unwrap()),
        fnv::<Quad>(&sh.residual::<Quad>(&z[..4], true).unwrap()),
        fnv::<Octo>(&sh.residual::<Octo>(&z, true).unwrap()),
        hj.0,
    ]
}

/// the reference hashes, recorded on aarch64 (NEON) and on the portable backend
const REFERENCE: [u64; 4] = [
    0x89fa2854700d40ae,
    0xbd90acb1476d86a4,
    0x0864a5779c33b38a,
    0x4f3db3e1f35e25bf,
];

/// The projected residual's bytes in every precision and its Jacobian's, at a state with every part in use. Pass: the
/// hashes equal the reference on every SIMD backend of the CI matrix and with any number of threads.
#[test]
fn projected_bytes_match_reference() {
    let (_, zs) = projected();
    let h = hashes(&full_parts(&perturbed(&zs, 1e-3)[0]));
    println!(
        "projected hashes [{}]",
        h.iter().map(|x| format!("{x:#018x}")).collect::<Vec<_>>().join(", ")
    );
    assert_eq!(h, REFERENCE, "the projected rows' bytes differ from the reference");
}

/// The control of `projected_bytes_match_reference`: one ulp of a node unknown must change every hash.
#[test]
fn projected_bytes_control_one_ulp_differs() {
    let (sh, zs) = projected();
    let z = full_parts(&perturbed(&zs, 1e-3)[0]);
    let base = hashes(&z);
    let mut zu = z.clone();
    let i = sh.off_node(1) + M + 5;
    zu[0][i] = f64::from_bits(zu[0][i].to_bits() + 1);
    let h = hashes(&zu);
    for k in 0..4 {
        assert_ne!(h[k], base[k], "hash {k} unchanged");
    }
}

/// The projected problem with the right block's rows scaled by 2^80 (`row_scale`, applied after the projection):
/// (whether the residual equals the unscaled one times the factors bit for bit, whether every Jacobian entry does),
/// with the comparison told the factor `t`
fn scaled_projection(t: f64) -> (bool, bool) {
    let (plain, zs) = projected();
    let (mut sc, _) = projected();
    let r0 = sc.row_block(sc.k());
    sc.row_scale = (0..sc.rows())
        .map(|r| if r >= r0 && r < r0 + 2 * M { 2f64.powi(80) } else { 1.0 })
        .collect();
    let z = perturbed(&zs, 1e-3);
    let f = |r: usize| DD::from_f64(if sc.row_scale[r] == 1.0 { 1.0 } else { t });
    let (f0, f1) = (
        plain.residual::<Double>(&parts(&z), true).unwrap(),
        sc.residual::<Double>(&parts(&z), true).unwrap(),
    );
    let ((c0, ld), (c1, _)) = (
        plain.jacobian::<Double>(&parts(&z), true).unwrap(),
        sc.jacobian::<Double>(&parts(&z), true).unwrap(),
    );
    let l = sc.len();
    (
        (0..l).all(|r| f0[r] * f(r) == f1[r]),
        (0..l).all(|j| {
            (0..l).all(|r| {
                DD::new(c0[0][j * ld + r], c0[1][j * ld + r]) * f(r) == DD::new(c1[0][j * ld + r], c1[1][j * ld + r])
            })
        }),
    )
}

/// Pass: with the right block scaled by 2^80, the projected residual and Jacobian are the unscaled ones times the
/// factors, bit for bit (the projection comes before the scaling).
#[test]
fn projected_rows_scale_after_the_projection() {
    assert_eq!(scaled_projection(2f64.powi(80)), (true, true));
}

/// The control of `projected_rows_scale_after_the_projection`: told the factor is 2^81, both comparisons must fail.
#[test]
fn projected_rows_scale_control_wrong_factor_fails() {
    assert_eq!(scaled_projection(2f64.powi(81)), (false, false));
}

/// Pass: `project_rows` refuses a block past the last, a field past NF, pinned unknowns past the fields, and a field
/// projected twice.
#[test]
fn project_rows_refuses_bad_layouts() {
    let (mut sh, _) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let (k, og) = (sh.k(), sh.off_global());
    assert!(sh.project_rows(k + 1, 0, 0, Parity::Odd, 3).is_err());
    assert!(sh.project_rows(k, 2, 0, Parity::Odd, 3).is_err());
    assert!(sh.project_rows(k, 1, og - M + 1, Parity::Even, 2).is_err());
    assert!(sh.project_rows(k, 1, 0, Parity::Even, 2).is_ok());
    assert!(sh.project_rows(k, 1, 0, Parity::Even, 2).is_err());
}

/// The control (feature `mutate`): with the projectors rounded to double-double the octo-double Newton converges to
/// its own rows, and the exact rows there must stay above the bound of `projected_newton_reaches_the_exact_rows`.
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::Control;

    #[test]
    fn double_double_projector_misses_the_exact_rows() {
        let (mut sh, zs) = projected();
        sh.set_control(Control::ProjectorDD);
        let (f, dz, rows) = solve(&sh, &zs);
        println!(
            "control, projector in double-double: max |F| od {:.2e}, |z - z*| {dz:.2e}, exact rows {rows:.3e}",
            f[2]
        );
        assert!(f[2] < 1e-110, "the controlled solve did not converge: {f:?}");
        assert!(rows >= 1e-115, "control did not fire: {rows:e}");
    }
}
