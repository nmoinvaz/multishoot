//! Trigonometric interpolation in any precision (`interp::Interp`), Boyd's arctan map (`interp::ArctanMap`), the
//! mapped grid's d/dtheta (`interp::dmat_mapped`) and solutions moved between layouts in quad- and octo-double
//! (`Shooting::resample_in`), each against fields whose values are known in octo-double.
mod common;

use common::Planted;
use multidouble::field::Fl;
use multidouble::od::OD;
use multishoot::interp::{atan2_od, dmat_mapped, grid_s, interp_between, ArctanMap, Interp};
use multishoot::lane::OV;
use multishoot::periodic::{dmat_c, sin_cos_pi_od, Parity, Resampler};
use multishoot::{Double, Native, Octo, Parities, Precision, Quad, DD};

/// the harmonics of each parity's test field, all below the Nyquist harmonic of a grid of 8 points
fn harmonics(parity: Parity) -> Vec<i64> {
    match parity {
        Parity::Full => vec![0, 1, 2, 3],
        Parity::Odd => vec![1, 3, 5],
        Parity::Even => vec![0, 2, 4, 6],
    }
}

/// the test field sum_k (0.3 + 0.1 k) cos k s + (0.2 - 0.05 k) sin k s at s, in octo-double
fn field_at(parity: Parity, s: OD) -> OD {
    harmonics(parity).iter().fold(OD::ZERO, |t, &k| {
        let (sn, cs) = s.mul_f(k as f64).sin_cos();
        t + cs.mul_f(0.3 + 0.1 * k as f64) + sn.mul_f(0.2 - 0.05 * k as f64)
    })
}

/// the test field on the uniform grid of m points, from the exact reductions of the grid's angles
fn field_on_grid(parity: Parity, m: usize) -> Vec<OD> {
    let n = if parity == Parity::Full { m } else { 2 * m } as i64;
    (0..m)
        .map(|j| {
            harmonics(parity).iter().fold(OD::ZERO, |t, &k| {
                let (sn, cs) = sin_cos_pi_od(2 * k * j as i64, n);
                t + cs.mul_f(0.3 + 0.1 * k as f64) + sn.mul_f(0.2 - 0.05 * k as f64)
            })
        })
        .collect()
}

/// values in P from octo-double, and the largest difference of values in P from octo-double ones
fn in_p<P: Precision<Native>>(v: &[OD]) -> Vec<P::Scalar> {
    v.iter().map(|x| P::new(&x.x[..P::PARTS])).collect()
}
fn err_p<P: Precision<Native>>(got: &[P::Scalar], want: &[OD]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| {
            let mut p = [0.0; 8];
            for (q, v) in p.iter_mut().enumerate().take(P::PARTS) {
                *v = P::part(g, q);
            }
            (OD::new(p) - w).to_f64().abs()
        })
        .fold(0.0, multisolver::prec::nan_max)
}

/// the largest error of `Interp::uniform(with, m, m2)` on the test field of `parity` in P, over m2 in 8, 16, 24, 32
fn uniform_err<P: Precision<Native>>(parity: Parity, with: Parity, m: usize) -> f64 {
    let f = in_p::<P>(&field_on_grid(parity, m));
    [8, 16, 24, 32]
        .iter()
        .map(|&m2| err_p::<P>(&Interp::uniform(with, m, m2).apply::<P>(&f), &field_on_grid(parity, m2)))
        .fold(0.0, multisolver::prec::nan_max)
}

/// Pass, written before the first run: the test fields of the three parities move from 16 points to 8, 16, 24 and 32
/// by `Interp::uniform` to 1e-30 in double-double, 1e-60 in quad-double and 1e-115 in octo-double, and in
/// double-double it agrees with `Resampler` to 1e-30. The controls: an odd field moved with the even map misses by
/// more than 1e-3, and the quad-double values of the double-double map miss 1e-60.
#[test]
fn uniform_interp_keeps_each_precision() {
    let mut worst = [0.0f64; 3];
    for p in [Parity::Full, Parity::Odd, Parity::Even] {
        worst[0] = worst[0].max(uniform_err::<Double>(p, p, 16));
        worst[1] = worst[1].max(uniform_err::<Quad>(p, p, 16));
        worst[2] = worst[2].max(uniform_err::<Octo>(p, p, 16));
        let f: Vec<DD> = in_p::<Double>(&field_on_grid(p, 16));
        let d = Resampler::new(p, 16, 24)
            .apply(&f)
            .iter()
            .zip(Interp::uniform(p, 16, 24).apply::<Double>(&f))
            .map(|(a, b)| (*a - b).hi.abs())
            .fold(0.0, f64::max);
        assert!(d <= 1e-30, "{p:?}: Resampler and Interp differ by {d:e}");
    }
    println!("uniform: dd {:.3e}, qd {:.3e}, od {:.3e}", worst[0], worst[1], worst[2]);
    assert!(
        worst[0] <= 1e-30 && worst[1] <= 1e-60 && worst[2] <= 1e-115,
        "{worst:?}"
    );
    let c = uniform_err::<Double>(Parity::Odd, Parity::Even, 16);
    let f = in_p::<Double>(&field_on_grid(Parity::Even, 16));
    let dd_in_qd: Vec<OD> = Interp::uniform(Parity::Even, 16, 32)
        .apply::<Double>(&f)
        .iter()
        .map(|&v| OD::from_dd(v))
        .collect();
    let want = field_on_grid(Parity::Even, 32);
    let q = dd_in_qd
        .iter()
        .zip(&want)
        .map(|(a, b)| (*a - *b).to_f64().abs())
        .fold(0.0, f64::max);
    println!("controls: odd field, even map {c:.3e}; double-double map read in quad-double {q:.3e}");
    assert!(c > 1e-3 && q > 1e-60, "control did not fire: {c:e} {q:e}");
}

/// points spread over two periods, none on a grid point of 16
fn points() -> Vec<OD> {
    (0..23)
        .map(|i| OD::from_f64(-6.1 + 0.553 * i as f64) / OD::from_f64(1.0 + 1e-3))
        .collect()
}

/// the largest error of `Interp::at(with, m, points)` on the test field of `parity` in P
fn at_err<P: Precision<Native>>(parity: Parity, with: Parity, m: usize) -> f64 {
    let s = points();
    let f = in_p::<P>(&field_on_grid(parity, m));
    let want: Vec<OD> = s.iter().map(|&x| field_at(parity, x)).collect();
    err_p::<P>(&Interp::at(with, m, &s).apply::<P>(&f), &want)
}

/// Pass, written before the first run: the interpolants of the test fields on 16 points (and the full field's on 9
/// points, an odd N) evaluated at 23 points over two periods match the fields to 1e-30, 1e-60 and 1e-110 in double-,
/// quad- and octo-double, and the grid's own points (and their images a period on) give its values bit for bit,
/// with the odd field's sign flipped past pi. The control: the even kernel for an odd field misses by more than 1e-3.
#[test]
fn interp_at_evaluates_the_interpolant() {
    let mut worst = [0.0f64; 3];
    for (p, m) in [
        (Parity::Full, 16),
        (Parity::Full, 9),
        (Parity::Odd, 16),
        (Parity::Even, 16),
    ] {
        if p == Parity::Full && m == 9 {
            // harmonics 0..3 lie below the Nyquist of 9 points
            assert!(harmonics(p).iter().all(|&k| 2 * k < 9));
        }
        worst[0] = worst[0].max(at_err::<Double>(p, p, m));
        worst[1] = worst[1].max(at_err::<Quad>(p, p, m));
        worst[2] = worst[2].max(at_err::<Octo>(p, p, m));
    }
    println!(
        "at points: dd {:.3e}, qd {:.3e}, od {:.3e}",
        worst[0], worst[1], worst[2]
    );
    assert!(
        worst[0] <= 1e-30 && worst[1] <= 1e-60 && worst[2] <= 1e-110,
        "{worst:?}"
    );
    for p in [Parity::Full, Parity::Odd, Parity::Even] {
        let f = in_p::<Octo>(&field_on_grid(p, 16));
        let period = if p == Parity::Full { 16 } else { 32 };
        let s: Vec<OD> = (0..16)
            .map(|j| grid_s(p, 16, j) + grid_s(p, 16, 16).mul_f(1.0))
            .collect();
        let shifted = Interp::at(p, 16, &s).apply::<Octo>(&f);
        let sign = if p == Parity::Odd { -1.0 } else { 1.0 };
        for j in 0..16 {
            assert_eq!(shifted[j].x, (f[j].mul_f(sign)).x, "{p:?} point {j} (period {period})");
        }
    }
    let c = at_err::<Double>(Parity::Odd, Parity::Even, 16);
    println!("control, odd field with the even kernel: {c:.3e}");
    assert!(c > 1e-3, "control did not fire");
}

/// the largest |g(g^-1(theta)) - theta| and |g'(s) - L / (cos^2 u + L^2 sin^2 u)| over the points
fn map_errors(map: &ArctanMap, skip_inverse: bool) -> (f64, f64) {
    let (mut e1, mut e2) = (0.0f64, 0.0f64);
    for &t in &points() {
        let s = if skip_inverse { t } else { map.inverse(t) };
        e1 = e1.max((map.g(s).0 - t).to_f64().abs());
        let (su, cu) = (t - OD::from_f64(map.s0)).sin_cos();
        let l = OD::from_f64(map.l);
        let gp = l / (cu * cu + l * l * su * su);
        e2 = e2.max((map.g(t).1 - gp).to_f64().abs());
    }
    (e1, e2)
}

/// Pass, written before the first run: for L = 1.7, 0.6 and 4 about s0 = 0.4, g(g^-1(theta)) = theta to 1e-115 at
/// the 23 points, and g' equals its other form L / (cos^2 u + L^2 sin^2 u) to 1e-115; L = 1 is the identity bit for
/// bit, and `atan2_od` gives pi / 4 at (1, 1) to 1e-120. The control: g^-1 skipped misses theta by more than 1e-3.
#[test]
fn arctan_map_inverts() {
    for l in [1.7, 0.6, 4.0] {
        let (e1, e2) = map_errors(&ArctanMap { s0: 0.4, l }, false);
        println!("L {l}: |g(g^-1) - theta| {e1:.3e}, |g' - its other form| {e2:.3e}");
        assert!(e1 <= 1e-115 && e2 <= 1e-115, "L {l}: {e1:e} {e2:e}");
    }
    let id = ArctanMap { s0: 0.4, l: 1.0 };
    for &t in &points() {
        assert_eq!(id.g(t).0.x, t.x);
        assert_eq!(id.inverse(t).x, t.x);
    }
    let q = (atan2_od(OD::ONE, OD::ONE).mul_f(4.0) - multishoot::periodic::PI_OD)
        .to_f64()
        .abs();
    assert!(q <= 1e-120, "atan2(1, 1): {q:e}");
    let (c, _) = map_errors(&ArctanMap { s0: 0.4, l: 1.7 }, true);
    println!("control, g^-1 skipped: {c:.3e}");
    assert!(c > 1e-3, "control did not fire");
}

/// max_i |(D sin)(s_i) - cos(s_i) / g'(s_i)| in octo-double lanes, D the mapped grid's d/dtheta of `d` on 32 odd points
fn mapped_derivative_err(d: &multishoot::periodic::DMatC<4>, map: &ArctanMap) -> f64 {
    let m = 32;
    let s: Vec<OD> = (0..m).map(|j| grid_s(Parity::Odd, m, j)).collect();
    let parts = |v: &[OD]| -> Fl<OV, 4> {
        let p: Vec<Vec<f64>> = (0..8).map(|q| v.iter().map(|x| x.x[q]).collect()).collect();
        Fl::<OV, 4>::load_parts_od(std::array::from_fn(|q| p[q].as_slice()))
    };
    let f = parts(&s.iter().map(|x| x.sin()).collect::<Vec<_>>());
    let df = d.apply(&f);
    (0..m)
        .map(|j| (df.get(j) - s[j].cos() / map.g(s[j]).1).to_f64().abs())
        .fold(0.0, f64::max)
}

/// Pass, written before the first run: on the odd grid of 32 points mapped with L = 1.7, `dmat_mapped` differentiates
/// sin(s), a field of the grid's harmonics in s, to cos(s) / g'(s) within 1e-44 in octo-double lanes (the entries'
/// remainders kept); the identity map gives `dmat_c` bit for bit. The control: `dmat_c`, without the factor 1 / g',
/// misses by more than 1e-3.
#[test]
fn mapped_dmat_differentiates_in_theta() {
    let map = ArctanMap { s0: 0.4, l: 1.7 };
    let e = mapped_derivative_err(&dmat_mapped::<4>(Parity::Odd, &map), &map);
    let id = dmat_mapped::<4>(Parity::Odd, &ArctanMap { s0: 0.4, l: 1.0 });
    let plain = dmat_c::<4>(Parity::Odd);
    for j in 0..32 {
        for i in 0..32 {
            assert_eq!(id.d.cols[j].get(i), plain.d.cols[j].get(i));
            assert_eq!(id.c.cols[j].get(i), plain.c.cols[j].get(i));
        }
    }
    let c = mapped_derivative_err(&plain, &map);
    println!("mapped d/dtheta: {e:.3e}; control, unmapped matrix {c:.3e}");
    assert!(e <= 1e-44, "{e:e}");
    assert!(c > 1e-3, "control did not fire");
}

const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

fn parities() -> Parities {
    Parities {
        left: vec![Parity::Even],
        node: vec![Parity::Odd, Parity::Even],
        right: vec![Parity::Odd],
    }
}

/// z (two parts) widened to P's parts
fn widen<P: Precision<Native>>(z: &[Vec<f64>]) -> Vec<Vec<f64>> {
    multisolver::prec::widen(z, P::PARTS)
}

/// the largest |z_back - z| (in octo-double) of the planted solution moved from 16 points to 32 and back in P
fn round_trip<P: Precision<Native>>() -> f64 {
    let (sh, zs) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    let (to, _) = Planted::<4>::shooting(&NODES, 30, 1e-32);
    let z = widen::<P>(&zs);
    fn refs(v: &[Vec<f64>]) -> Vec<&[f64]> {
        v.iter().map(Vec::as_slice).collect()
    }
    let up = sh
        .resample_in::<P, _, 4>(&refs(&z), &parities(), &to, None, None)
        .unwrap();
    let back = to
        .resample_in::<P, _, 2>(&refs(&up), &parities(), &sh, None, None)
        .unwrap();
    (0..sh.len())
        .map(|i| {
            let p = |v: &[Vec<f64>]| OD::new(std::array::from_fn(|q| if q < v.len() { v[q][i] } else { 0.0 }));
            (p(&back) - p(&z)).to_f64().abs()
        })
        .fold(0.0, f64::max)
}

/// Pass, written before the first run: the planted solution moved from 16 points to 32 and back (up, then down with
/// the harmonics past 16 points' Nyquist dropped, which the trip up did not make) returns to 1e-60 in quad-double and
/// 1e-115 in octo-double. The control: the same trip by `resample`, in double-double, misses 1e-60.
#[test]
fn resample_in_keeps_quad_and_octo_double() {
    let (q, o) = (round_trip::<Quad>(), round_trip::<Octo>());
    let (sh, zs) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    let (to, _) = Planted::<4>::shooting(&NODES, 30, 1e-32);
    let (uh, ul) = sh.resample(&zs[0], &zs[1], &parities(), &to).unwrap();
    let (bh, bl) = to.resample(&uh, &ul, &parities(), &sh).unwrap();
    let dd = (0..sh.len())
        .map(|i| {
            (OD::from_dd(DD::new(bh[i], bl[i])) - OD::from_dd(DD::new(zs[0][i], zs[1][i])))
                .to_f64()
                .abs()
        })
        .fold(0.0, f64::max);
    println!("16 -> 32 -> 16: qd {q:.3e}, od {o:.3e}; control, double-double resample {dd:.3e}");
    assert!(q <= 1e-60 && o <= 1e-115, "{q:e} {o:e}");
    assert!(dd > 1e-60, "control did not fire");
}

/// the planted fields at x = NODES[1] (node 1, u* = g A + B, v* = g^2 C + E) at theta in octo-double
fn planted_at(theta: OD, x: f64) -> (OD, OD) {
    let g = OD::ONE / OD::from_f64(1.0 + x);
    let sc = |k: f64| theta.mul_f(k).sin_cos();
    let ((s1, c1), (s2, c2), (s3, c3)) = (sc(1.0), sc(2.0), sc(3.0));
    let a = s1 + c3.mul_f(0.3);
    let b = c1.mul_f(0.2) - s3.mul_f(0.1);
    let c = OD::from_f64(0.5) + c2.mul_f(0.25);
    let e = OD::ONE + s2.mul_f(0.1);
    (g * a + b, g * g * c + e)
}

/// the largest error of node 1's fields moved from the uniform grid of 16 points to the 32 points mapped by `to_map`
/// (read as uniform when `ignore`), against the planted fields at the mapped points theta_j = g(s_j)
fn mapped_err(ignore: bool) -> f64 {
    let map = ArctanMap { s0: 0.4, l: 1.7 };
    let (sh, zs) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    let (to, _) = Planted::<4>::shooting(&NODES, 30, 1e-32);
    let z = widen::<Quad>(&zs);
    let refs: Vec<&[f64]> = z.iter().map(Vec::as_slice).collect();
    let out = sh
        .resample_in::<Quad, _, 4>(&refs, &parities(), &to, None, if ignore { None } else { Some(&map) })
        .unwrap();
    let o = to.off_node(1);
    let theta = multishoot::interp::grid_theta(Parity::Odd, 32, Some(&map));
    let val = |i: usize| OD::new(std::array::from_fn(|q| if q < 4 { out[q][i] } else { 0.0 }));
    (0..32)
        .map(|j| {
            let (u, v) = planted_at(theta[j], NODES[1]);
            (val(o + j) - u)
                .to_f64()
                .abs()
                .max((val(o + 32 + j) - v).to_f64().abs())
        })
        .fold(0.0, f64::max)
}

/// Pass, written before the first run: node 1's planted fields (harmonics up to 3, given in double-double) moved
/// from 16 uniform points to 32 points mapped with L = 1.7 match the planted fields at the mapped points to 1e-30.
/// The control: the map left out of the target misses by more than 1e-3.
#[test]
fn resample_in_moves_to_a_mapped_grid() {
    let (e, c) = (mapped_err(false), mapped_err(true));
    println!("uniform 16 -> mapped 32: {e:.3e}; control, map left out {c:.3e}");
    assert!(e <= 1e-30, "{e:e}");
    assert!(c > 1e-3, "control did not fire");
}

/// the identity maps give the uniform path: `interp_between` with L = 1 on both sides equals `Interp::at` at the
/// uniform target points, entry for entry; pass, written before the first run, every entry equal
#[test]
fn identity_maps_give_the_uniform_grid() {
    let id = ArctanMap { s0: 0.4, l: 1.0 };
    for p in [Parity::Odd, Parity::Even, Parity::Full] {
        let a = interp_between(p, 16, 24, Some(&id), Some(&id));
        let s: Vec<OD> = (0..24).map(|j| grid_s(p, 24, j)).collect();
        let b = Interp::at(p, 16, &s);
        for i in 0..24 {
            for j in 0..16 {
                assert_eq!(a.entry(i, j).x, b.entry(i, j).x, "{p:?} ({i}, {j})");
            }
        }
    }
}
