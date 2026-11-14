//! Moving solutions between layouts: the trigonometric resampler on fields of each parity, the planted problem moved
//! from 16 to 32 collocation points, and its solution moved to new node positions.
mod common;

use common::{in_pool, Fnv, Planted};
use multishoot::periodic::{sin_cos_grid, Parity, Resampler};
use multishoot::prec::nan_max;
use multishoot::{Parities, Plan, DD};

/// the field sum over the harmonics ks of (0.3 + 0.1 k) cos k theta + (0.2 - 0.05 k) sin k theta on the grid of m
/// points of the parity, with a nonzero lo part in every coefficient
fn field(parity: Parity, m: usize, ks: &[i64]) -> Vec<DD> {
    (0..m)
        .map(|j| {
            ks.iter().fold(DD::ZERO, |t, &k| {
                let (s, c) = sin_cos_grid(parity, m, j, k);
                let (a, b) = (0.3 + 0.1 * k as f64, 0.2 - 0.05 * k as f64);
                t + c * DD::new(a, a * 1e-17) + s * DD::new(b, -b * 1e-17)
            })
        })
        .collect()
}

fn max_diff(a: &[DD], b: &[DD]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x - *y).hi.abs()).fold(0.0, f64::max)
}

/// the largest error of moving the field of harmonics ks from 16 points to 8, 16 and 32 with the resampler of
/// `with` (its own parity for the check, another for the control)
fn resample_err(parity: Parity, with: Parity, ks: &[i64]) -> f64 {
    let f = field(parity, 16, ks);
    [8, 16, 32]
        .iter()
        .map(|&m2| max_diff(&Resampler::new(with, 16, m2).apply(&f), &field(parity, m2, ks)))
        .fold(0.0, f64::max)
}

/// Pass, written before the first run: fields whose harmonics lie below both grids' Nyquist (a full field of
/// harmonics 0..3, an odd one of 1, 3, 5 and an even one of 0, 2, 4, 6) move from 16 points to 8, 16 and 32 to
/// 1e-30. The control: an odd field moved with the even resampler misses by more than 1e-3.
#[test]
fn resampler_moves_resolved_fields_exactly() {
    let full = resample_err(Parity::Full, Parity::Full, &[0, 1, 2, 3]);
    let odd = resample_err(Parity::Odd, Parity::Odd, &[1, 3, 5]);
    let even = resample_err(Parity::Even, Parity::Even, &[0, 2, 4, 6]);
    println!("resampler: full {full:.3e}, odd {odd:.3e}, even {even:.3e}");
    assert!(full <= 1e-30 && odd <= 1e-30 && even <= 1e-30);
    let c = resample_err(Parity::Odd, Parity::Even, &[1, 3, 5]);
    println!("control, odd field with the even resampler: {c:.3e}");
    assert!(c > 1e-3, "control did not fire");
}

/// the largest error of moving the field of harmonics ks from m points to each of m2s, with the resampler of its parity
fn resample_err_m(parity: Parity, m: usize, m2s: &[usize], ks: &[i64]) -> f64 {
    let f = field(parity, m, ks);
    m2s.iter()
        .map(|&m2| max_diff(&Resampler::new(parity, m, m2).apply(&f), &field(parity, m2, ks)))
        .fold(0.0, nan_max)
}

/// Pass, written before the first run: on grids whose period is not a power of two, where 1 / N is not an f64, and on
/// odd grids, which have no Nyquist harmonic, resolved fields move to 1e-30: a full field of harmonics 0..5 from 24
/// points to 12, 24 and 48, odd (1, 3, 5) and even (0, 2, 4) fields from 12 points to 6, 12 and 24, and a full field
/// of harmonics 0..2 from 5 points to 5, 7 and 10. The control: harmonic 3 on 5 points, which that grid aliases,
/// misses by more than 1e-3.
#[test]
fn resampler_is_exact_on_grids_of_any_size() {
    let errs = [
        resample_err_m(Parity::Full, 24, &[12, 24, 48], &[0, 1, 2, 3, 4, 5]),
        resample_err_m(Parity::Odd, 12, &[6, 12, 24], &[1, 3, 5]),
        resample_err_m(Parity::Even, 12, &[6, 12, 24], &[0, 2, 4]),
        resample_err_m(Parity::Full, 5, &[5, 7, 10], &[0, 1, 2]),
    ];
    println!(
        "resampler: full 24 {:.3e}, odd 12 {:.3e}, even 12 {:.3e}, full 5 {:.3e}",
        errs[0], errs[1], errs[2], errs[3]
    );
    assert!(errs.iter().all(|&e| e <= 1e-30));
    let c = resample_err_m(Parity::Full, 5, &[10], &[0, 1, 3]);
    println!("control, harmonic 3 on 5 points: {c:.3e}");
    assert!(c > 1e-3, "control did not fire");
}

const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

/// the planted problem's solution on 16 points moved to 32 with the node parities given: max |z - z*| against the
/// manufactured solution on 32 points
fn planted_resample(node: [Parity; 2]) -> f64 {
    let (sh, zs) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    let (to, zt) = Planted::<4>::shooting(&NODES, 30, 1e-32);
    let par = Parities {
        left: vec![Parity::Even],
        node: node.to_vec(),
        right: vec![Parity::Odd],
    };
    let (h, l) = sh.resample(&zs[0], &zs[1], &par, &to).unwrap();
    assert_eq!(h.len(), to.len());
    (0..to.len())
        .map(|i| (DD::new(h[i], l[i]) - DD::new(zt[0][i], zt[1][i])).hi.abs())
        .fold(0.0, f64::max)
}

/// Pass, written before the first run: the planted solution (u odd, v even, harmonics up to 3) moved from 16 to 32
/// points equals the manufactured solution on 32 points to 1e-30. The control: u and v given each other's parity miss
/// by more than 1e-3.
#[test]
fn planted_solution_resamples_to_more_points() {
    let e = planted_resample([Parity::Odd, Parity::Even]);
    let c = planted_resample([Parity::Even, Parity::Odd]);
    println!("planted 16 -> 32 points: {e:.3e}; control, parities swapped: {c:.3e}");
    assert!(e <= 1e-30);
    assert!(c > 1e-3, "control did not fire");
}

const NEW: [f64; 6] = [0.0, 0.3, 0.8, 1.2, 1.7, 2.0];

/// the planted solution moved to the nodes NEW with the plan: max |z - z*| against the manufactured solution there
fn planted_renode(plan: &Plan) -> f64 {
    let (sh, zs) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    let (to, zt) = Planted::<2>::shooting(&NEW, 30, 1e-32);
    let (h, l) = sh.renode(&zs[0], &zs[1], &NEW, plan).unwrap();
    assert_eq!(h.len(), to.len());
    (0..to.len())
        .map(|i| (DD::new(h[i], l[i]) - DD::new(zt[0][i], zt[1][i])).hi.abs())
        .fold(0.0, f64::max)
}

/// Pass, written before the first run: the planted solution moved to new interior nodes by Taylor paths of order 30
/// (last terms below 1e-32) equals the manufactured solution there to 1e-28. The control: paths of order 6 with last
/// terms up to 1e-6 miss by more than 1e-20.
#[test]
fn planted_solution_moves_to_new_nodes() {
    let e = planted_renode(&Plan {
        kt: 30,
        tol: 1e-32,
        hmax: 0.25,
    });
    let c = planted_renode(&Plan {
        kt: 6,
        tol: 1e-6,
        hmax: 0.5,
    });
    println!("planted renode: {e:.3e}; control, order 6: {c:.3e}");
    assert!(e <= 1e-28);
    assert!(c > 1e-20, "control did not fire");
}

/// the hash of the planted solution moved to the nodes NEW (order 30, last terms below 1e-32), the last bit of
/// unknown 37 changed with `ulp`
fn renode_hash(ulp: bool) -> u64 {
    let (sh, mut zs) = Planted::<2>::shooting(&NODES, 30, 1e-32);
    if ulp {
        zs[0][37] = f64::from_bits(zs[0][37].to_bits() + 1);
    }
    let plan = Plan {
        kt: 30,
        tol: 1e-32,
        hmax: 0.25,
    };
    let (h, l) = sh.renode(&zs[0], &zs[1], &NEW, &plan).unwrap();
    let mut f = Fnv::new();
    h.iter().chain(&l).for_each(|&x| f.f64(x));
    f.0
}

/// the reference hash of `renode_hash`, recorded with the serial loop over the new nodes on aarch64 (NEON)
const RENODE_REFERENCE: u64 = 0xce991f54ebf3bc7f;

/// The moved solution does not depend on the threads that form the new nodes' states: pass, `renode_hash` in pools
/// of 1, 2 and 5 threads equals the reference of the serial loop. The control: one ulp of unknown 37 changes it.
#[test]
fn renode_does_not_depend_on_threads() {
    for t in [1, 2, 5] {
        let h = in_pool(t, || renode_hash(false));
        println!("renode hash, {t} threads: {h:#018x}");
        assert_eq!(h, RENODE_REFERENCE, "{t} threads");
    }
    assert_ne!(
        in_pool(2, || renode_hash(true)),
        RENODE_REFERENCE,
        "control did not fire"
    );
}
