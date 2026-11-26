//! d/dtheta with the remainders of its double-double entries (`periodic::dmat_c`, `DMatC`): its parts against mpmath,
//! its derivatives of harmonics in quad- and octo-double, its order-by-order product on jets, and the split f = 1 + X
//! of a solve exact for the matrix applied.
use multidouble::dd::DD;
use multidouble::field::{FieldSolve, Fl};
use multidouble::linalg::DDLu;
use multidouble::od::OD;
use multishoot::lane::Lane;
use multishoot::lane::{Jet, OJet, QJet, DV, OV, QV};
use multishoot::periodic::{dmat, dmat_c, dmat_od, sin_cos_pi_od, DMatC, Parity};

/// the reference parts of scripts/dmat_mp/dmat_mp.py: (parity, m, i, j, hi, lo, c)
fn reference() -> Vec<(Parity, usize, usize, usize, f64, f64, f64)> {
    include_str!("data/dmat_mp.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let w: Vec<&str> = l.split_whitespace().collect();
            let parity = match w[0] {
                "odd" => Parity::Odd,
                "even" => Parity::Even,
                _ => Parity::Full,
            };
            let n = |k: usize| w[k].parse::<usize>().unwrap();
            let f = |k: usize| f64::from_bits(u64::from_str_radix(w[k], 16).unwrap());
            (parity, n(1), n(2), n(3), f(4), f(5), f(6))
        })
        .collect()
}

/// the number of entries of `mat(parity)` whose hi, lo or c differ from the reference at M = 8 B, and the number of
/// entries compared
fn differing<const B: usize>(mat: impl Fn(Parity) -> DMatC<B>) -> (usize, usize) {
    let m = 8 * B;
    let (mut bad, mut n) = (0, 0);
    for parity in [Parity::Odd, Parity::Even, Parity::Full] {
        let a = mat(parity);
        for &(p, mm, i, j, hi, lo, c) in &reference() {
            if p != parity || mm != m {
                continue;
            }
            let (d, r) = (a.d.cols[j].get(i), a.c.cols[j].get(i));
            n += 1;
            bad += usize::from(d.hi != hi || d.lo != lo || r.hi != c || r.lo != 0.0);
        }
    }
    (bad, n)
}

/// The parts against mpmath: pass, every entry's hi, lo and c equal those of the 120-digit values split greedily
/// (scripts/dmat_mp/dmat_mp.py), on the three grids at M = 8 and 16 (960 entries).
#[test]
fn dmat_c_parts_match_mpmath() {
    let (b1, n1) = differing::<1>(dmat_c::<1>);
    let (b2, n2) = differing::<2>(dmat_c::<2>);
    assert_eq!((n1, n2), (192, 768), "reference entries");
    assert_eq!((b1, b2), (0, 0), "entries off mpmath's parts");
}

/// The control of `dmat_c_parts_match_mpmath`: with the remainders dropped (c = 0) the parts must differ, at every
/// entry the double-double does not hold exactly; and `dmat`'s double-double arithmetic, with c = 0, at least as
/// many.
#[test]
fn dmat_c_mpmath_control_without_remainder_fails() {
    let drop_c = |mut a: DMatC<2>| {
        a.c = DMatC::<2>::from_od(&vec![OD::ZERO; 256]).c;
        a
    };
    let (bad, _) = differing::<2>(|p| drop_c(dmat_c::<2>(p)));
    let (bad_dd, _) = differing::<2>(|p| {
        let mut a = drop_c(dmat_c::<2>(p));
        a.d = dmat::<2>(p);
        a
    });
    println!("c = 0: {bad} of 768 entries differ; dmat with c = 0: {bad_dd}");
    assert!(bad > 600 && bad_dd >= bad, "control did not fire: {bad} {bad_dd}");
}

/// the field v (8 B octo-double values) in octo-double lanes
fn od_field<const B: usize>(v: &[OD]) -> Fl<OV, B> {
    let parts: Vec<Vec<f64>> = (0..8).map(|q| v.iter().map(|x| x.x[q]).collect()).collect();
    Fl::<OV, B>::load_parts_od(std::array::from_fn(|q| parts[q].as_slice()))
}

/// the number of grid points per period
fn period(parity: Parity, m: usize) -> i64 {
    match parity {
        Parity::Full => m as i64,
        _ => 2 * m as i64,
    }
}

/// the largest error of D applied to sin k theta and cos k theta, against k cos and -k sin, in octo-double and in
/// quad-double lanes, over harmonics below the Nyquist harmonic of each grid at M = 8 B
fn harmonic_error<const B: usize>(mat: impl Fn(Parity) -> DMatC<B>) -> (f64, f64) {
    let m = 8 * B;
    let (mut eo, mut eq) = (0.0f64, 0.0f64);
    for parity in [Parity::Odd, Parity::Even, Parity::Full] {
        let a = mat(parity);
        let n = period(parity, m);
        let ks: Vec<i64> = (0..n / 2)
            .filter(|&k| match parity {
                Parity::Odd => k % 2 == 1,
                Parity::Even => k % 2 == 0,
                Parity::Full => true,
            })
            .collect();
        for k in ks {
            let sc: Vec<(OD, OD)> = (0..m).map(|j| sin_cos_pi_od(2 * k * j as i64, n)).collect();
            let s: Vec<OD> = sc.iter().map(|p| p.0).collect();
            let c: Vec<OD> = sc.iter().map(|p| p.1).collect();
            let kf = k as f64;
            let ds = a.apply(&od_field::<B>(&s));
            let dc = a.apply(&od_field::<B>(&c));
            let qs = a.apply(&Fl::<QV, B>(std::array::from_fn(|b| od_field::<B>(&s).0[b].to_qv())));
            let qc = a.apply(&Fl::<QV, B>(std::array::from_fn(|b| od_field::<B>(&c).0[b].to_qv())));
            for j in 0..m {
                eo = eo
                    .max((ds.get(j) - c[j].mul_f(kf)).to_f64().abs())
                    .max((dc.get(j) + s[j].mul_f(kf)).to_f64().abs());
                let (cq, sq) = (c[j].to_qd(), s[j].to_qd());
                eq = eq
                    .max((qs.get(j) - cq.mul_f(kf)).to_f64().abs())
                    .max((qc.get(j) + sq.mul_f(kf)).to_f64().abs());
            }
        }
    }
    (eo, eq)
}

/// Harmonics differentiated in the wide lanes: pass, D sin k theta = k cos k theta and D cos k theta = -k sin k theta
/// to 1e-44 in octo- and quad-double lanes at M = 16 and 32, every harmonic below the Nyquist harmonic of the three
/// grids (entries of size up to N / 2 pi held to about 2^-160, summed over M points).
#[test]
fn dmat_c_differentiates_harmonics_in_wide_lanes() {
    let (o2, q2) = harmonic_error::<2>(dmat_c::<2>);
    let (o4, q4) = harmonic_error::<4>(dmat_c::<4>);
    println!("M 16: od {o2:.3e} qd {q2:.3e}; M 32: od {o4:.3e} qd {q4:.3e}");
    assert!(o2.max(q2).max(o4).max(q4) < 1e-44, "{o2:e} {q2:e} {o4:e} {q4:e}");
}

/// The control of `dmat_c_differentiates_harmonics_in_wide_lanes`: without the remainders (the double-double matrix
/// alone, as `dmat` was applied) the errors must exceed 1e-40 at M = 16 and 32, in both lanes.
#[test]
fn dmat_c_harmonics_control_without_remainder_fails() {
    fn no_c<const B: usize>(p: Parity) -> DMatC<B> {
        let mut a = dmat_c::<B>(p);
        let m = 8 * B;
        a.c = DMatC::<B>::from_od(&vec![OD::ZERO; m * m]).c;
        a
    }
    let (o2, q2) = harmonic_error::<2>(no_c::<2>);
    let (o4, q4) = harmonic_error::<4>(no_c::<4>);
    println!("control, c = 0: M 16: od {o2:.3e} qd {q2:.3e}; M 32: od {o4:.3e} qd {q4:.3e}");
    assert!(
        o2.min(q2).min(o4).min(q4) > 1e-40,
        "control did not fire: {o2:e} {q2:e} {o4:e} {q4:e}"
    );
}

/// a jet field of N coefficients on the odd grid at M = 8 B: coefficient i the samples of sin (2i + 1) theta / (i + 1)
/// in octo-double, set through `set_coef_od` (rounded to the jet's coefficient precision)
fn jet_field<T: Lane, const B: usize, const N: usize>() -> Fl<T, B> {
    let m = 8 * B;
    let mut f = Fl::<T, B>::zero();
    for i in 0..N {
        let v: Vec<OD> = (0..m)
            .map(|j| sin_cos_pi_od((2 * i as i64 + 1) * j as i64, 2 * m as i64).0 / OD::from_f64((i + 1) as f64))
            .collect();
        let o = od_field::<B>(&v);
        for b in 0..B {
            f.0[b].set_coef_od(i, o.0[b]);
        }
    }
    f
}

/// the number of coefficients (over the jet's N orders and the M points) at which `apply_at`, order by order, differs
/// from coefficient j of `apply`, compared in octo-double
fn apply_at_differing<T: Lane, const B: usize, const N: usize>(a: &DMatC<B>, b: &DMatC<B>) -> usize {
    let x = jet_field::<T, B, N>();
    let full = a.apply(&x);
    let mut out = Fl::<T, B>::zero();
    for j in 0..N {
        b.apply_at(&x, j, &mut out);
    }
    let mut bad = 0;
    for j in 0..N {
        let (u, v) = (
            Fl::<OV, B>(std::array::from_fn(|l| full.0[l].coef_od(j))),
            Fl::<OV, B>(std::array::from_fn(|l| out.0[l].coef_od(j))),
        );
        bad += (0..8 * B).filter(|&i| u.get(i) != v.get(i)).count();
    }
    bad
}

/// The order-by-order product: pass, `apply_at` for each order equals that coefficient of `apply`, every part, on
/// `Jet<4>`, `QJet<4>` and `OJet<4>` lanes at M = 16, odd grid.
#[test]
fn dmat_c_apply_at_is_apply() {
    let a = dmat_c::<2>(Parity::Odd);
    let n = (
        apply_at_differing::<Jet<4>, 2, 4>(&a, &a),
        apply_at_differing::<QJet<4>, 2, 4>(&a, &a),
        apply_at_differing::<OJet<4>, 2, 4>(&a, &a),
    );
    assert_eq!(n, (0, 0, 0));
}

/// The control of `dmat_c_apply_at_is_apply`: `apply_at` with the remainders dropped must differ from `apply` on the
/// quad- and octo-double jets, and not on `Jet<4>`, which does not read them.
#[test]
fn dmat_c_apply_at_control_without_remainder_differs() {
    let a = dmat_c::<2>(Parity::Odd);
    let mut b = a.clone();
    b.c = DMatC::<2>::from_od(&vec![OD::ZERO; 256]).c;
    let n = (
        apply_at_differing::<Jet<4>, 2, 4>(&a, &b),
        apply_at_differing::<QJet<4>, 2, 4>(&a, &b),
        apply_at_differing::<OJet<4>, 2, 4>(&a, &b),
    );
    println!("control, c = 0: {n:?} of 64 coefficients differ");
    assert!(n.0 == 0 && n.1 > 0 && n.2 > 0, "control did not fire: {n:?}");
}

/// omega and kappa of the split's model solve: L f = omega D f + diag(kappa) f on the even grid at M = 16
const OMEGA: f64 = 1.25;

/// kappa_j = 3 + j / 16, exact in f64
fn kappa(j: usize) -> f64 {
    3.0 + (j as f64) / 16.0
}

/// The even grid's matrix at M = 16 with entries formed elsewhere, as a stand-in for a matrix from another evaluation
/// (mpmath, say): entry (i, j) times 1 + 2^-165 ((7 i + 3 j) mod 5 - 2), which breaks the cancellation of its rows in
/// pairs
fn perturbed() -> DMatC<2> {
    let a = dmat_od::<2>(Parity::Even);
    let e = OD::from_f64(2f64.powi(-165));
    DMatC::from_od(
        &(0..256)
            .map(|k| {
                let (i, j) = (k / 16, k % 16);
                a[k] + a[k] * e.mul_f(((7 * i + 3 * j) % 5) as f64 - 2.0)
            })
            .collect::<Vec<_>>(),
    )
}

/// (max |L B - q|, max |B - B_direct|) of the split B = 1 + X, L X = q - r, in octo-double lanes with the matrix a
/// (D in L), with r = L 1 from the row sums when `row_sums`, else r = kappa (D 1 taken as zero); q = L(1 + X0),
/// X0 = 1e-3 cos 2 theta, and B_direct the direct solve of L B = q, each solve refined 8 steps from the double-double
/// factors
fn split_errors(a: &DMatC<2>, row_sums: bool) -> (f64, f64) {
    const B: usize = 2;
    let m = 8 * B;
    let om = OV::splat(OD::from_f64(OMEGA));
    let kap = od_field::<B>(&(0..m).map(|j| OD::from_f64(kappa(j))).collect::<Vec<_>>());
    let l = |f: &Fl<OV, B>| a.apply(f).mul_t(om) + kap * *f;
    // the factors of omega (hi + lo) + diag(kappa), column-major
    let (mut hi, mut lo) = (vec![0.0; m * m], vec![0.0; m * m]);
    for i in 0..m {
        for j in 0..m {
            let v = a.d.cols[j].get(i) * DD::from_f64(OMEGA) + DD::from_f64(if i == j { kappa(i) } else { 0.0 });
            (hi[j * m + i], lo[j * m + i]) = (v.hi, v.lo);
        }
    }
    let lu = DDLu::factor(m, m, hi, lo, false).unwrap();
    let x0: Vec<OD> = (0..m)
        .map(|j| sin_cos_pi_od(4 * j as i64, 2 * m as i64).1.mul_f(1e-3))
        .collect();
    let one = Fl::<OV, B>::splat(DD::ONE);
    let q = l(&(one + od_field::<B>(&x0)));
    let direct = lu.solve_lanes_refine(&q, 8, |x| q - l(x));
    let r = if row_sums {
        a.row_sums::<OV>().mul_t(om) + kap
    } else {
        kap
    };
    let rhs = q - r;
    let bb = one + lu.solve_lanes_refine(&rhs, 8, |x| rhs - l(x));
    let res = q - l(&bb);
    let diff = bb - direct;
    (0..m).fold((0.0f64, 0.0f64), |(e, d), j| {
        (e.max(res.get(j).to_f64().abs()), d.max(diff.get(j).to_f64().abs()))
    })
}

/// the largest |row sum| of a in octo-double lanes
fn row_sum_max(a: &DMatC<2>) -> f64 {
    let r = a.row_sums::<OV>();
    (0..16).map(|j| r.get(j).to_f64().abs()).fold(0.0, f64::max)
}

/// The split exact for the matrix applied: pass, for the matrix of `perturbed`, whose rows sum to more than 1e-52, B =
/// 1 + X with the right side q - L 1, L 1 from `row_sums`, solves L B = q to 1e-60 in octo-double lanes and equals
/// the direct solve to 1e-60.
#[test]
fn dmat_c_split_exact_with_row_sums() {
    let a = perturbed();
    let s = row_sum_max(&a);
    let (res, diff) = split_errors(&a, true);
    println!("split with the row sums: rows {s:.3e}, |L B - q| {res:.3e}, |B - B_direct| {diff:.3e}");
    assert!(s > 1e-52, "the rows of the perturbed matrix sum to {s:e}");
    assert!(res < 1e-60 && diff < 1e-60, "{res:e} {diff:e}");
}

/// The control of `dmat_c_split_exact_with_row_sums`: with D 1 taken as zero (the right side q - kappa) the residual
/// and the distance to the direct solve must exceed 1e-55.
#[test]
fn dmat_c_split_control_without_row_sums_fails() {
    let (res, diff) = split_errors(&perturbed(), false);
    println!("control, D 1 = 0: |L B - q| {res:.3e}, |B - B_direct| {diff:.3e}");
    assert!(res > 1e-55 && diff > 1e-55, "control did not fire: {res:e} {diff:e}");
}

/// Rows that sum to zero: pass, `dmat_c`'s matrices on the even and full grids at M = 8, 16 and 32 have row sums
/// exactly zero in quad- and octo-double lanes, their entries at offsets d and -d negatives of each other, so the
/// split B = 1 + X needs no row-sum term with them; and its split with that term equals its split without, bitwise.
#[test]
fn dmat_c_rows_sum_to_zero() {
    fn zero<const B: usize>(p: Parity) -> bool {
        let a = dmat_c::<B>(p);
        let (o, q) = (a.row_sums::<OV>(), a.row_sums::<QV>());
        (0..8 * B).all(|j| o.get(j).to_f64() == 0.0 && q.get(j).to_f64() == 0.0)
    }
    for p in [Parity::Even, Parity::Full] {
        assert!(zero::<1>(p) && zero::<2>(p) && zero::<4>(p), "{p:?}");
    }
    let a = dmat_c::<2>(Parity::Even);
    assert_eq!(split_errors(&a, true), split_errors(&a, false));
}

/// The control of `dmat_c_rows_sum_to_zero`: with one remainder moved to the next f64 (row 3, column 5 of the even
/// grid at M = 16) row 3's sum must be nonzero in octo-double lanes.
#[test]
fn dmat_c_rows_control_one_remainder_moved_fails() {
    let mut a = dmat_c::<2>(Parity::Even);
    let mut c: Vec<f64> = (0..256).map(|k| a.c.cols[k % 16].get(k / 16).hi).collect();
    c[3 * 16 + 5] = c[3 * 16 + 5].next_up();
    a.c = multishoot::lane::DMat::<2>::from_rows(&c, &[0.0; 256]);
    let r = a.row_sums::<OV>().get(3).to_f64();
    println!("control, one remainder moved: row 3 sums to {r:.3e}");
    assert!(r != 0.0, "control did not fire");
}

/// The row sums in each lane: pass, `row_sums` on `DV` lanes is `d.apply` of 1 bitwise (no remainder there); and on
/// `OV` lanes for the matrix of `perturbed` it differs from `d.apply` of 1 (the remainders' row sums enter).
#[test]
fn dmat_c_row_sums_by_lane() {
    let a = perturbed();
    let dv = a.row_sums::<DV>();
    let dd = a.d.apply(&Fl::<DV, 2>::splat(DD::ONE));
    assert!((0..16).all(|j| dv.get(j) == dd.get(j)));
    let ov = a.row_sums::<OV>();
    let od = a.d.apply(&Fl::<OV, 2>::splat(DD::ONE));
    assert!((0..16).any(|j| ov.get(j) != od.get(j)));
}
