//! The mode matrix: the linearization for perturbations that grow as exp(lambda t), on the planted problem and on the
//! linear problem `Wave`, whose determinant is known in closed form for every complex lambda.
mod common;

use common::{as_pointwise, in_pool, perturbed, wave, Fnv, Planted};
use multishoot::lane::{DMat, Lane};
use multishoot::periodic::{dmat, Parity};
use multishoot::taylor::taylor_path;
use multishoot::{Equations, ExtraRow, Fl, Rate, Shooting, State, Work, DD};
use multisolver::prec::parts;

/// The linear wave: two odd fields u, v on the half grid, with t = theta / omega and omega a global,
///
/// ```text
/// u_x = -v_t,   v_x = u_t
/// ```
///
/// u given (zero) at x = 0 and v given (zero) at x = X_END, past the last node: the right endpoint map carries the
/// state (data, 0) from X_END to the last node by Taylor steps at the rate of its preparation. A perturbation
/// exp(lambda t) g has g_x = -(D_t + lambda) g_v, (g_v)_x = (D_t + lambda) g_u, D_t = omega d/dtheta, whose
/// eigenvalues on the odd grid are +-i omega k, k = 1, 3, ..., M - 1. Along each eigenvector the map from x = 0 to
/// X_END is the rotation by L (lambda +- i omega k), L = X_END, so the shooting matrix is singular exactly when
/// cos(L (lambda +- i omega k)) vanishes, and det M(lambda) = +-prod_k cos(L (lambda + i omega k)) cos(L (lambda -
/// i omega k)) (the blocks between the nodes eliminate with unit determinant, and the backward map has determinant
/// one). With `explicit` = Some(mu) the equations carry mu u and mu v as values, `u_x = -(v_t + mu v)`, the problem
/// whose Jacobian is the mode matrix at the real rate mu.
pub struct Wave<const B: usize> {
    pub d: DMat<B>,
    pub explicit: Option<DD>,
}

/// the end of the right endpoint map, past the last node
const X_END: f64 = 2.5;
/// the steps of the right endpoint map, X_END to the last node
const END_STEPS: [f64; 5] = [2.5, 2.375, 2.25, 2.125, 2.0];
/// omega, the global
const OMEGA: f64 = 0.5;

impl<const B: usize> Wave<B> {
    fn dt<T: Lane>(&self, rate: Rate, x: &Fl<T, B>, om: T) -> Fl<T, B> {
        let mut d = rate.shift(x, self.d.apply(x).mul_t(om));
        if let Some(mu) = self.explicit {
            d = d + x.muls(mu);
        }
        d
    }
}

impl<const B: usize> Equations<B> for Wave<B> {
    const NF: usize = 2;
    const NL: usize = 1;
    const NR: usize = 1;
    const NG: usize = 1;
    type Prep = Rate;

    fn prepare_rate(&self, _zh: &[f64], _zl: &[f64], rate: Rate) -> Rate {
        rate
    }

    fn taylor_coeffs<T: Lane>(&self, _x0: f64, g: &[T], w: &mut Work<T, B>) {
        for k in 0..w.kt {
            let ik = DD::ONE / DD::from_f64((k + 1) as f64);
            let ut = self.dt(w.rate, &w.c[0][k], g[0]);
            let vt = self.dt(w.rate, &w.c[1][k], g[0]);
            w.c[0][k + 1] = (-vt).muls(ik);
            w.c[1][k + 1] = ut.muls(ik);
        }
    }

    fn left_state<T: Lane>(&self, _x: f64, data: &[Fl<T, B>], _g: &[T], _prep: &Rate) -> State<T, B> {
        vec![Fl::zero(), data[0]]
    }

    fn right_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &Rate) -> State<T, B> {
        assert_eq!(x, *END_STEPS.last().unwrap());
        let mut w = Work::<T, B>::new(2, 0, 30);
        w.rate = *prep;
        taylor_path(self, &[data[0], Fl::zero()], &END_STEPS, g, &mut w)
    }
}

const WB: usize = 1;

/// The wave on the nodes 0, 0.5, ..., 2 (steps of 0.125, order 30: |sigma h| <= 0.7 for the rates below, a
/// truncation below 1e-38), its extra row omega = OMEGA, and a z of arbitrary values (the problem is linear, so its
/// matrices do not depend on z) with omega in place
fn wave_problem(explicit: Option<DD>) -> (Shooting<Wave<WB>, WB>, Vec<Vec<f64>>) {
    let nodes = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    let mut sh = Shooting::new(
        Wave {
            d: dmat::<WB>(Parity::Odd),
            explicit,
        },
        nodes.clone(),
        30,
    );
    sh.segs = nodes
        .windows(2)
        .map(|w| (0..=4).map(|i| w[0] + 0.125 * i as f64).collect())
        .collect();
    let og = sh.off_global();
    sh.extra.push(ExtraRow {
        terms: vec![(og, DD::ONE)],
        rhs: DD::from_f64(OMEGA),
    });
    let mut z = vec![(0..sh.len()).map(wave).collect::<Vec<f64>>(), vec![0.0; sh.len()]];
    z[0][og] = OMEGA;
    (sh, z)
}

/// ln|cos(a + ib)| and arg cos(a + ib), cos(a + ib) = cos a cosh b - i sin a sinh b
fn ln_cos(a: f64, b: f64) -> (f64, f64) {
    let (s, c) = a.sin_cos();
    (
        0.5 * (c * c + b.sinh() * b.sinh()).ln(),
        (-s * b.sinh()).atan2(c * b.cosh()),
    )
}

/// (ln|det|, arg det) of the closed form prod_k cos(L (lambda + i omega k)) cos(L (lambda - i omega k)), k odd < M
fn planted_logdet(lr: f64, li: f64) -> (f64, f64) {
    let m = 8 * WB;
    let (mut la, mut arg) = (0.0, 0.0);
    for k in (1..m).step_by(2) {
        for s in [1.0, -1.0] {
            let (l, a) = ln_cos(X_END * lr, X_END * (li + s * OMEGA * k as f64));
            la += l;
            arg += a;
        }
    }
    (la, arg)
}

/// x reduced to (-pi, pi]
fn wrap(x: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let y = x - t * (x / t).round();
    if y <= -std::f64::consts::PI {
        y + t
    } else {
        y
    }
}

/// the rates of the determinant check: real, imaginary and complex, on both sides of the zeros
const RATES: [(f64, f64); 6] = [
    (0.3, 0.0),
    (0.0, 0.25),
    (0.3, 0.2),
    (-0.4, 0.7),
    (1.1, -0.35),
    (0.62, 0.48),
];

/// For each rate: (|ln|det M| - ln|det|_planted|, the arg difference reduced modulo pi)
fn determinant_errors(sh: &Shooting<Wave<WB>, WB>, z: &[Vec<f64>]) -> Vec<(f64, f64)> {
    RATES
        .iter()
        .map(|&(lr, li)| {
            let (la, arg) = sh
                .mode_logdet(&parts(z), Rate::new(DD::from_f64(lr), DD::from_f64(li)), true)
                .unwrap();
            let (pa, parg) = planted_logdet(lr, li);
            let d = wrap(arg - parg);
            (
                (la - pa).abs(),
                d.abs()
                    .min((d - std::f64::consts::PI).abs())
                    .min((d + std::f64::consts::PI).abs()),
            )
        })
        .collect()
}

/// The wave's det M(lambda) against its closed form at real, imaginary and complex rates, one of them (0.62 +
/// 0.48 i) 0.02 from a zero. ln|det| and arg det are sums of 96 pivots' logarithms in f64, so they agree to about
/// 1e-13. Pass: |ln|det M| - ln|det|| below 1e-10, and arg det M within 1e-10 of the closed form's arg modulo pi (the
/// sign of the layout's elimination).
#[test]
fn wave_determinant_is_planted() {
    let (sh, z) = wave_problem(None);
    let e = determinant_errors(&sh, &z);
    println!(
        "det M against the closed form (ln, arg): {}",
        e.iter()
            .map(|(a, b)| format!("({a:.1e}, {b:.1e})"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    for (r, &(a, b)) in RATES.iter().zip(&e) {
        assert!(a < 1e-10 && b < 1e-10, "rate {r:?}: {a:e} {b:e}");
    }
}

/// The winding number of det M along the circle of radius r about c, from n samples: the sum of the arg steps over
/// 2 pi, and the largest step (which must stay below pi for the count to hold)
fn winding(sh: &Shooting<Wave<WB>, WB>, z: &[Vec<f64>], c: (f64, f64), r: f64, n: usize) -> (f64, f64) {
    let args: Vec<f64> = (0..=n)
        .map(|i| {
            let (s, co) = (std::f64::consts::TAU * i as f64 / n as f64).sin_cos();
            let rate = Rate::new(DD::from_f64(c.0 + r * co), DD::from_f64(c.1 + r * s));
            sh.mode_logdet(&parts(z), rate, true).unwrap().1
        })
        .collect();
    let steps: Vec<f64> = args.windows(2).map(|w| wrap(w[1] - w[0])).collect();
    (
        steps.iter().sum::<f64>() / std::f64::consts::TAU,
        steps.iter().fold(0.0f64, |a, s| a.max(s.abs())),
    )
}

/// the circles of the winding count, (centre, radius, zeros inside): about the zero pi / 2L + i omega, about the
/// origin (no zeros; the nearest are 0.8 away), and about pi / 2L with the two zeros pi / 2L +- i omega
fn circles() -> [((f64, f64), f64, f64); 3] {
    let z0 = std::f64::consts::PI / (2.0 * X_END);
    [((z0, OMEGA), 0.3, 1.0), ((0.0, 0.0), 0.3, 0.0), ((z0, 0.0), 0.8, 2.0)]
}

/// The modes counted by the argument principle: the winding number of det M(lambda) along three circles, 64 samples
/// each. Pass: each count within 1e-6 of the planted number of zeros inside (1, 0, 2), with every arg step below 1.
#[test]
fn winding_counts_the_planted_zeros() {
    let (sh, z) = wave_problem(None);
    for (c, r, want) in circles() {
        let (w, step) = winding(&sh, &z, c, r, 64);
        println!("circle about {c:?}, radius {r}: winding {w:.9}, largest arg step {step:.3}");
        assert!(
            (w - want).abs() < 1e-6 && step < 1.0,
            "{c:?}: {w} (want {want}), step {step}"
        );
    }
}

/// The real-rate mode matrix (complex tangents with a zero imaginary part) against the Jacobian (`Du`) of the wave
/// that carries the rate as values, `explicit` = Some(mu): the same operations on the tangents. Pass: every entry of
/// the real part equal to the Jacobian's in both parts, and the imaginary part zero.
#[test]
fn real_rate_is_the_explicit_shift() {
    let (differ, nonzero) = explicit_comparison();
    println!("real rate against the explicit shift: {differ} entries differ, {nonzero} imaginary parts nonzero");
    assert_eq!((differ, nonzero), (0, 0));
}

/// (entries of the real part that differ from the explicitly shifted Jacobian, imaginary parts not zero), mu = 0.37
fn explicit_comparison() -> (usize, usize) {
    let mu = DD::from_f64(0.37);
    let (sh, z) = wave_problem(None);
    let mm = sh.mode_matrix(&parts(&z), Rate::new(mu, DD::ZERO), true).unwrap();
    let (ex, z) = wave_problem(Some(mu));
    let (c, ld) = ex.jacobian::<multishoot::Double>(&parts(&z), true).unwrap();
    block_comparison(&mm, &c, ld)
}

/// (entries of mm's real part not equal to the Jacobian's leading block in both parts, entries of its imaginary part
/// not zero)
fn block_comparison(mm: &multishoot::ModeMatrix, c: &[Vec<f64>], ld: usize) -> (usize, usize) {
    let (mut differ, mut nonzero) = (0, 0);
    for j in 0..mm.n {
        for i in 0..mm.n {
            let (re, im) = mm.at(i, j);
            if re.hi != c[0][j * ld + i] || re.lo != c[1][j * ld + i] {
                differ += 1;
            }
            if im.hi != 0.0 || im.lo != 0.0 {
                nonzero += 1;
            }
        }
    }
    (differ, nonzero)
}

const PB: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

/// The planted problem's mode matrix at rate zero against its Jacobian, at z* perturbed by 1e-2 (a background where
/// the linearization depends on the state). Pass: the real part equal to the Jacobian's block rows and columns before
/// omega in both parts of every entry, and the imaginary part zero.
#[test]
fn rate_zero_is_the_jacobian_block() {
    let (differ, nonzero) = planted_block_comparison(|_| {}, |_| {});
    println!("rate zero against the Jacobian: {differ} entries differ, {nonzero} imaginary parts nonzero");
    assert_eq!((differ, nonzero), (0, 0));
}

/// The rate-zero comparison with block 2's rows scaled by 2^60 (`row_scale`, which the Jacobian and the mode matrix
/// both carry). Pass: equal in both parts of every entry, imaginary parts zero.
#[test]
fn rate_zero_is_the_scaled_jacobian_block() {
    let (differ, nonzero) = planted_block_comparison(
        |sh| {
            let (r0, r1) = (sh.row_block(2), sh.row_block(3));
            sh.row_scale = (0..sh.rows())
                .map(|r| if (r0..r1).contains(&r) { 2f64.powi(60) } else { 1.0 })
                .collect();
        },
        |_| {},
    );
    println!("rate zero against the scaled Jacobian: {differ} entries differ, {nonzero} imaginary parts nonzero");
    assert_eq!((differ, nonzero), (0, 0));
}

/// The control of `rate_zero_is_the_scaled_jacobian_block`: the mode matrix without the scaling (the Jacobian
/// scaled) must differ on block 2's rows.
#[test]
fn rate_zero_control_unscaled_differs() {
    let (differ, _) = planted_block_comparison(
        |sh| {
            let (r0, r1) = (sh.row_block(2), sh.row_block(3));
            sh.row_scale = (0..sh.rows())
                .map(|r| if (r0..r1).contains(&r) { 2f64.powi(60) } else { 1.0 })
                .collect();
        },
        |sh| sh.row_scale.clear(),
    );
    println!("control, unscaled mode matrix: {differ} entries differ");
    assert!(differ > 0, "control did not fire");
}

/// the planted problem's rows of u in block 2 projected onto the odd harmonics k <= 3, node 1's u pinned (an input
/// of block 2, so its columns carry both the block derivative and Pi'), and block 2 scaled by 2^60
fn project_and_scale(sh: &mut Shooting<Planted<PB>, PB>) {
    let pin = sh.off_node(1);
    sh.project_rows(2, 0, pin, multishoot::periodic::Parity::Odd, 3)
        .unwrap();
    let (r0, r1) = (sh.row_block(2), sh.row_block(3));
    sh.row_scale = (0..sh.rows())
        .map(|r| if (r0..r1).contains(&r) { 2f64.powi(60) } else { 1.0 })
        .collect();
}

/// The rate-zero comparison with projected and scaled rows (`project_and_scale`), which the Jacobian and the mode
/// matrix both carry, the projection first. Pass: equal in both parts of every entry, imaginary parts zero.
#[test]
fn rate_zero_is_the_projected_jacobian_block() {
    let (differ, nonzero) = planted_block_comparison(project_and_scale, |_| {});
    println!("rate zero against the projected Jacobian: {differ} entries differ, {nonzero} imaginary parts nonzero");
    assert_eq!((differ, nonzero), (0, 0));
}

/// The controls of `rate_zero_is_the_projected_jacobian_block`: the mode matrix without the projection, and the
/// mode matrix with the factor 2^61 in place of 2^60, must each differ from the Jacobian's block.
#[test]
fn rate_zero_projected_controls_differ() {
    let (unprojected, _) = planted_block_comparison(project_and_scale, |sh| sh.projected.clear());
    let (wrong, _) = planted_block_comparison(project_and_scale, |sh| {
        sh.row_scale
            .iter_mut()
            .filter(|s| **s != 1.0)
            .for_each(|s| *s = 2f64.powi(61))
    });
    println!("controls, projected rate zero: unprojected {unprojected}, wrong factor {wrong} entries differ");
    assert!(unprojected > 0 && wrong > 0, "control did not fire");
}

/// (entries of the projected mode matrix that differ from the unprojected one's projected by hand, in either part),
/// at a complex rate: Pi on the real and imaginary parts of block 2's u rows in every column, Pi' added to the real
/// part of the pinned columns; with `real_only` the hand projection leaves the imaginary part alone (the control)
fn complex_projection_differs(real_only: bool) -> usize {
    let (plain, zs) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    let (mut sh, _) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    let pin = sh.off_node(1);
    sh.project_rows(2, 0, pin, multishoot::periodic::Parity::Odd, 3)
        .unwrap();
    let z = perturbed(&zs, 1e-2);
    let m0 = plain.mode_matrix(&parts(&z), hash_rate(), true).unwrap();
    let m1 = sh.mode_matrix(&parts(&z), hash_rate(), true).unwrap();
    let p = &sh.projected[0];
    let (m, r0) = (8 * PB, sh.row_block(2));
    let mut differ = 0;
    for j in 0..m0.n {
        let col: Vec<(DD, DD)> = (0..m).map(|i| m0.at(r0 + i, j)).collect();
        let re: Vec<DD> = col.iter().map(|c| c.0).collect();
        let im: Vec<DD> = col.iter().map(|c| c.1).collect();
        let mut pre = p.apply::<multishoot::Double>(&re, None, false);
        let pim = if real_only {
            im.clone()
        } else {
            p.apply::<multishoot::Double>(&im, None, false)
        };
        if (pin..pin + m).contains(&j) {
            for (i, v) in pre.iter_mut().enumerate() {
                *v = *v + p.entry::<multishoot::Double>(false, i, j - pin, false);
            }
        }
        for i in 0..m {
            let (a, b) = m1.at(r0 + i, j);
            if a != pre[i] || b != pim[i] {
                differ += 1;
            }
        }
    }
    differ
}

/// The projection at a complex rate reaches both parts: pass, the projected mode matrix's block 2 u rows equal the
/// unprojected matrix's projected by hand (`ProjectedRows::apply`) in every entry.
#[test]
fn complex_rate_projects_both_parts() {
    let d = complex_projection_differs(false);
    println!("complex rate, projected rows against the hand projection: {d} entries differ");
    assert_eq!(d, 0);
}

/// The control of `complex_rate_projects_both_parts`: the hand projection without the imaginary part must differ.
#[test]
fn complex_rate_projection_control_real_only_differs() {
    let d = complex_projection_differs(true);
    println!("control, real part only: {d} entries differ");
    assert!(d > 0, "control did not fire");
}

/// `block_comparison` for the planted problem at rate zero: the layout changed by `both` for the Jacobian and the
/// mode matrix, then by `set` for the mode matrix alone (a control)
fn planted_block_comparison(
    both: impl Fn(&mut Shooting<Planted<PB>, PB>),
    set: impl Fn(&mut Shooting<Planted<PB>, PB>),
) -> (usize, usize) {
    let (mut sh, zs) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    let z = perturbed(&zs, 1e-2);
    both(&mut sh);
    let (c, ld) = sh.jacobian::<multishoot::Double>(&parts(&z), true).unwrap();
    set(&mut sh);
    let mm = sh.mode_matrix(&parts(&z), Rate::ZERO, true).unwrap();
    block_comparison(&mm, &c, ld)
}

/// (entries of the mode matrix of `pw` that differ from that of `plain`, in either part; entries of plain's with a
/// nonzero imaginary part) at z* perturbed by 1e-2 and the fingerprint's rate: one evaluation per input field for
/// the pointwise declaration against one per column, the declared problem's layout changed by `set` (a control)
fn pointwise_differs(
    plain: &Shooting<Planted<PB>, PB>,
    z: &[Vec<f64>],
    coupled: bool,
    set: impl Fn(&mut Shooting<common::Pointwise<Planted<PB>>, PB>),
) -> (usize, usize) {
    let (sh, _) = Planted::<PB>::shooting_of(Planted::with_coupling(coupled), &NODES, 30, 1e-32);
    let mut pw = as_pointwise(sh);
    set(&mut pw);
    let a = plain.mode_matrix(&parts(z), hash_rate(), true).unwrap();
    let b = pw.mode_matrix(&parts(z), hash_rate(), true).unwrap();
    let all = || (0..a.n).flat_map(|j| (0..a.n).map(move |i| (i, j)));
    (
        all().filter(|&(i, j)| a.at(i, j) != b.at(i, j)).count(),
        all().filter(|&(i, j)| a.at(i, j).1 != DD::ZERO).count(),
    )
}

/// The pointwise problem's mode matrix at a complex rate (its -omega u marked as a t-derivative, so the rate enters
/// every point), batched against per column. Pass: every entry equal in both parts, with imaginary parts that are
/// not zero.
#[test]
fn pointwise_mode_matrix_equals_per_column() {
    let (plain, zs) = Planted::<PB>::shooting_of(Planted::with_coupling(false), &NODES, 30, 1e-32);
    let (d, im) = pointwise_differs(&plain, &perturbed(&zs, 1e-2), false, |_| {});
    println!("pointwise mode matrix against per column: {d} entries differ, {im} imaginary parts nonzero");
    assert!(d == 0 && im > 0, "{d} {im}");
}

/// The control of `pointwise_mode_matrix_equals_per_column`: the coupled problem declared pointwise must differ, as
/// d/dtheta carries a point's tangent to the others (a second control, the rate's real part only, is with the
/// `mutate` controls).
#[test]
fn pointwise_mode_control_coupled_differs() {
    let (plain, zs) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    let (d, _) = pointwise_differs(&plain, &perturbed(&zs, 1e-2), true, |_| {});
    println!("control, coupled problem declared pointwise: {d} entries differ");
    assert!(d > 0, "control did not fire");
}

/// (entries of the mode matrix with block 2's rows scaled by 2^60 that differ from the unscaled one times `t` there
/// and times 1 elsewhere, in either part; imaginary parts of the unscaled matrix that are not zero) at the
/// fingerprint's complex rate
fn complex_scaling_differs(t: f64) -> (usize, usize) {
    let (plain, zs) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    let (mut sc, _) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    let (r0, r1) = (sc.row_block(2), sc.row_block(3));
    sc.row_scale = (0..sc.rows())
        .map(|r| if (r0..r1).contains(&r) { 2f64.powi(60) } else { 1.0 })
        .collect();
    let z = perturbed(&zs, 1e-2);
    let a = plain.mode_matrix(&parts(&z), hash_rate(), true).unwrap();
    let b = sc.mode_matrix(&parts(&z), hash_rate(), true).unwrap();
    let all = || (0..a.n).flat_map(|j| (0..a.n).map(move |i| (i, j)));
    let f = |i: usize| DD::from_f64(if (r0..r1).contains(&i) { t } else { 1.0 });
    (
        all()
            .filter(|&(i, j)| {
                let ((ar, ai), (br, bi)) = (a.at(i, j), b.at(i, j));
                ar * f(i) != br || ai * f(i) != bi
            })
            .count(),
        all().filter(|&(i, j)| a.at(i, j).1 != DD::ZERO).count(),
    )
}

/// Row scaling at a complex rate: pass, the scaled mode matrix is the unscaled one times the factors in both parts of
/// every entry, bit for bit (a power of two scales exactly), with imaginary parts that are not zero.
#[test]
fn complex_rate_rows_scale() {
    let (d, im) = complex_scaling_differs(2f64.powi(60));
    println!("complex rate, scaled rows: {d} entries differ, {im} imaginary parts nonzero");
    assert!(d == 0 && im > 0, "{d} {im}");
}

/// The control of `complex_rate_rows_scale`: told the factor is 2^61, the comparison must fail.
#[test]
fn complex_rate_rows_scale_control_wrong_factor() {
    let (d, _) = complex_scaling_differs(2f64.powi(61));
    println!("control, wrong factor at a complex rate: {d} entries differ");
    assert!(d > 0, "control did not fire");
}

/// Pass: a `row_scale` of the wrong length is refused by the mode matrix, as by the residual.
#[test]
fn mode_matrix_refuses_a_wrong_row_scale() {
    let (mut sh, zs) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    sh.row_scale = vec![1.0; sh.rows() - 1];
    assert!(sh.mode_matrix(&parts(&zs), Rate::ZERO, true).is_err());
    sh.row_scale.push(1.0);
    assert!(sh.mode_matrix(&parts(&zs), Rate::ZERO, true).is_ok());
}

/// The FNV-1a hash of every part of the planted problem's mode matrix at z* perturbed by 1e-2 and the rate
fn mode_hash(rate: Rate) -> u64 {
    mode_hash_par(rate, true)
}

/// `mode_hash` with the columns on rayon tasks (`par`) or one after another
fn mode_hash_par(rate: Rate, par: bool) -> u64 {
    let (sh, zs) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    let z = perturbed(&zs, 1e-2);
    let mm = sh.mode_matrix(&parts(&z), rate, par).unwrap();
    let mut h = Fnv::new();
    mm.re.iter().chain(&mm.im).flatten().for_each(|&v| h.f64(v));
    h.0
}

/// the rate of the fingerprint
fn hash_rate() -> Rate {
    Rate::new(DD::from_f64(0.3), DD::from_f64(0.2))
}

/// the reference hash, recorded on aarch64 (NEON) and on the portable backend
const MODE_REFERENCE: u64 = 0x17e8dd6459d89e55;

/// The planted problem's mode matrix at a complex rate. Pass: its FNV-1a hash equals the reference on every SIMD
/// backend of the CI matrix, with any number of threads, and with the columns in parallel or in sequence.
#[test]
fn mode_bytes_match_reference() {
    let h = mode_hash(hash_rate());
    println!("mode matrix hash {h:#018x}");
    assert_eq!(h, MODE_REFERENCE, "the mode matrix's bytes differ from the reference");
    assert_eq!(
        mode_hash_par(hash_rate(), false),
        MODE_REFERENCE,
        "the sequential mode matrix's bytes differ"
    );
}

/// The control of `mode_bytes_match_reference`: one ulp of the rate's imaginary part must change the hash.
#[test]
fn mode_bytes_control_one_ulp_differs() {
    let mut r = hash_rate();
    r.im.hi = f64::from_bits(r.im.hi.to_bits() + 1);
    assert_ne!(mode_hash(r), mode_hash(hash_rate()), "control did not fire");
}

/// n rates on the circle of radius 0.4 about 0.3 + 0.2 i
fn contour(n: usize) -> Vec<Rate> {
    (0..n)
        .map(|i| {
            let (s, c) = (std::f64::consts::TAU * i as f64 / n as f64).sin_cos();
            Rate::new(DD::from_f64(0.3 + 0.4 * c), DD::from_f64(0.2 + 0.4 * s))
        })
        .collect()
}

/// the bits of (ln|det|, arg det) at each rate
fn logdet_bits(v: &[(f64, f64)]) -> Vec<(u64, u64)> {
    v.iter().map(|&(a, b)| (a.to_bits(), b.to_bits())).collect()
}

/// The contour's determinants side by side. Pass: `mode_logdets` with `par`, in pools of 1 and 6 threads, with 8
/// rates (on 6 threads 6 rates serial and 2 in parallel) and 2 (each in parallel on 6 threads), and without `par` in a
/// pool of 3, gives the bits of a loop of `mode_logdet` with and without `par`, rate by rate in order. The control:
/// the loop with two rates swapped differs.
#[test]
fn mode_logdets_are_each_rates_logdet() {
    let (sh, zs) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    let z = perturbed(&zs, 1e-2);
    let zp = parts(&z);
    for n in [8, 2] {
        let rates = contour(n);
        let one = |r: &[Rate], par: bool| -> Vec<(f64, f64)> {
            r.iter().map(|&r| sh.mode_logdet(&zp, r, par).unwrap()).collect()
        };
        let want = logdet_bits(&one(&rates, true));
        assert_eq!(
            logdet_bits(&one(&rates, false)),
            want,
            "{n} rates: mode_logdet depends on par"
        );
        for (t, par) in [(1, true), (6, true), (3, false)] {
            let got: Vec<(f64, f64)> = in_pool(t, || sh.mode_logdets(&zp, &rates, par))
                .into_iter()
                .map(|r| r.unwrap())
                .collect();
            assert_eq!(logdet_bits(&got), want, "{n} rates, {t} threads, par {par}");
        }
        let mut swapped = rates.clone();
        swapped.swap(0, 1);
        assert_ne!(logdet_bits(&one(&swapped, true)), want, "control did not fire");
    }
}

/// the bits of both words of every value of a double-double field
fn bits(f: &Fl<multishoot::lane::DV, 1>) -> Vec<u64> {
    (0..8)
        .flat_map(|j| [f.get(j).hi.to_bits(), f.get(j).lo.to_bits()])
        .collect()
}

/// The shift at rate zero, or on a lane without a tangent, is no shift: pass, a nonzero rate on a value lane returns
/// dx in every bit of both words, and so does the zero rate on a tangent lane, value and tangent, with dx holding
/// signed zeros among its values and tangents.
#[test]
fn shift_is_the_identity_without_a_tangent_or_a_rate() {
    let (x, dx) = identity_inputs();
    let r = Rate::new(DD::from_f64(0.5), DD::from_f64(0.25)).shift(&x, dx);
    assert_eq!(bits(&r), bits(&dx));
    let xu = Fl::<multishoot::lane::Du, 1>::unit_tangent(&x, 3);
    let du = Fl::<multishoot::lane::Du, 1>::from_vt(&dx, &dx);
    let ru = Rate::ZERO.shift(&xu, du);
    assert_eq!(bits(&ru.val()), bits(&du.val()));
    assert_eq!(bits(&ru.tan()), bits(&du.tan()));
}

/// x and dx of the identity checks: dx with -0 and +0 among its values
fn identity_inputs() -> (Fl<multishoot::lane::DV, 1>, Fl<multishoot::lane::DV, 1>) {
    let x = Fl::<multishoot::lane::DV, 1>::splat(DD::from_f64(0.7));
    let v: Vec<DD> = (0..8)
        .map(|j| match j {
            2 => DD { hi: -0.0, lo: 0.0 },
            5 => DD::ZERO,
            _ => DD::from_f64(-1.3 + j as f64),
        })
        .collect();
    let (h, l) = multidouble::dd::split(&v);
    (x, Fl::load(&h, &l))
}

/// The controls of the bitwise comparison in `shift_is_the_identity_without_a_tangent_or_a_rate`: dx with its -0
/// made +0 must differ in the bits (though equal as numbers), and a value-lane result moved by one ulp must differ.
#[test]
fn shift_identity_controls_signed_zero_and_value_differ() {
    let (_, dx) = identity_inputs();
    let mut v: Vec<DD> = (0..8).map(|j| dx.get(j)).collect();
    v[2] = DD::ZERO;
    let (h, l) = multidouble::dd::split(&v);
    let flipped = Fl::<multishoot::lane::DV, 1>::load(&h, &l);
    assert!(
        (0..8).all(|j| flipped.get(j) == dx.get(j)),
        "the signed zero is not a numeric change"
    );
    assert_ne!(bits(&flipped), bits(&dx));
    v[2] = dx.get(2);
    v[0].hi = f64::from_bits(v[0].hi.to_bits() + 1);
    let (h, l) = multidouble::dd::split(&v);
    assert_ne!(bits(&Fl::<multishoot::lane::DV, 1>::load(&h, &l)), bits(&dx));
}

/// The control of `shift_is_the_identity_without_a_tangent_or_a_rate`: a real rate on a tangent lane must move the
/// tangent at the point x's tangent seeds, by the rate times it, and leave the value.
#[test]
fn shift_control_real_rate_moves_the_tangent() {
    let x = Fl::<multishoot::lane::DV, 1>::splat(DD::from_f64(0.7));
    let dx = Fl::<multishoot::lane::DV, 1>::splat(DD::from_f64(-1.3));
    let xu = Fl::<multishoot::lane::Du, 1>::unit_tangent(&x, 3);
    let du = Fl::<multishoot::lane::Du, 1>::unit_tangent(&dx, 5);
    let ru = Rate::new(DD::from_f64(0.5), DD::ZERO).shift(&xu, du);
    assert_eq!(ru.tan().get(3), DD::from_f64(0.5));
    assert!((0..8).all(|j| ru.val().get(j) == du.val().get(j)));
}

/// A complex rate on real tangents cannot be carried: pass, it panics.
#[test]
#[should_panic(expected = "a complex one Dc lanes")]
fn complex_rate_on_real_tangents_panics() {
    let x = Fl::<multishoot::lane::Du, 1>::unit_tangent(&Fl::splat(DD::ONE), 0);
    Rate::new(DD::ZERO, DD::ONE).shift(&x, x);
}

/// The controls (feature `mutate`), each run once against the check it breaks.
#[cfg(feature = "mutate")]
mod controls {
    use super::*;
    use multishoot::control::Control;

    /// the rate's imaginary part dropped: the closed form fails at the complex rates, and the circle about pi / 2L +
    /// i omega winds zero times
    #[test]
    fn rate_real_fails_the_determinant_and_the_count() {
        let (mut sh, z) = wave_problem(None);
        sh.set_control(Control::RateReal);
        let worst = determinant_errors(&sh, &z)
            .iter()
            .fold(0.0f64, |a, e| a.max(e.0).max(e.1));
        let (c, r, want) = circles()[0];
        let (w, _) = winding(&sh, &z, c, r, 64);
        println!("control, real part of the rate only: worst determinant error {worst:.3e}, winding {w:.3}");
        assert!(worst >= 1e-10 && (w - want).abs() >= 1e-6, "control did not fire");
    }

    /// the Taylor steps at rate zero: the real-rate matrix differs from the explicitly shifted Jacobian
    #[test]
    fn rate_not_in_steps_fails_the_explicit_shift() {
        let mu = DD::from_f64(0.37);
        let (mut sh, z) = wave_problem(None);
        sh.set_control(Control::RateNotInSteps);
        let mm = sh.mode_matrix(&parts(&z), Rate::new(mu, DD::ZERO), true).unwrap();
        let (ex, z) = wave_problem(Some(mu));
        let (c, ld) = ex.jacobian::<multishoot::Double>(&parts(&z), true).unwrap();
        let (differ, _) = block_comparison(&mm, &c, ld);
        println!("control, rate not in the steps: {differ} entries differ");
        assert!(differ > 0, "control did not fire");
    }

    /// the preparation without the rate: the right endpoint map runs unshifted, and the closed form fails
    #[test]
    fn rate_not_prepared_fails_the_determinant() {
        let (mut sh, z) = wave_problem(None);
        sh.set_control(Control::RateNotPrepared);
        let worst = determinant_errors(&sh, &z)
            .iter()
            .fold(0.0f64, |a, e| a.max(e.0).max(e.1));
        println!("control, rate not prepared: worst determinant error {worst:.3e}");
        assert!(worst >= 1e-10, "control did not fire");
    }

    /// the batched pointwise matrix with the rate's real part only: it must differ from the per-column one
    #[test]
    fn pointwise_rate_real_differs() {
        let (plain, zs) = Planted::<PB>::shooting_of(Planted::with_coupling(false), &NODES, 30, 1e-32);
        let (d, _) = pointwise_differs(&plain, &perturbed(&zs, 1e-2), false, |pw| {
            pw.set_control(Control::RateReal)
        });
        println!("control, pointwise with the real rate only: {d} entries differ");
        assert!(d > 0, "control did not fire");
    }

    /// a dropped column: the rate-zero matrix differs from the Jacobian's block
    #[test]
    fn dropped_column_fails_the_jacobian_block() {
        let (differ, _) =
            planted_block_comparison(|_| {}, |sh| sh.set_control(Control::DropColumn(sh.off_node(2) + 5)));
        println!("control, dropped column: {differ} entries differ");
        assert!(differ > 0, "control did not fire");
    }
}

/// the rectangle (x0, x1) x (y0, y1), counter-clockwise
fn rect(x0: f64, x1: f64, y0: f64, y1: f64) -> Vec<(f64, f64)> {
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

/// Pass, written before the first run: `Shooting::mode_winding` counts the wave's modes, the zeros (pi / 2 + n pi) / L
/// -+ i omega k of its det M, in boxes: 1 in (0.3, 1) x (0.2, 0.8) about pi / 2L + i omega, 0 about the origin, 2 in
/// (0.3, 1) x (-0.8, 0.8), and 4 in (0.3, 2.2) x (-1, 1), each an integer to 1e-6 and resolved. The first box's
/// moment locates its zero to 1e-2, and `mode_zero` refines it to 1e-10 of pi / 2L + i omega. The controls: the first
/// box's count fails the count check for 0 and for 2 zeros, its moment moved by 0.05 fails the moment check, and
/// refined from the mirrored box's moment `mode_zero` reaches the conjugate zero, more than 0.5 from pi / 2L + i omega.
#[test]
fn mode_winding_counts_and_locates_the_modes() {
    use multisolver::winding::WindOpts;
    let (sh, z) = wave_problem(None);
    let opts = WindOpts::default();
    let z0 = std::f64::consts::PI / (2.0 * X_END);
    let count_ok = |w: &multisolver::winding::Winding, want: i64| (w.turns - want as f64).abs() < 1e-6 && w.resolved;
    for (poly, want) in [
        (rect(0.3, 1.0, 0.2, 0.8), 1),
        (rect(-0.3, 0.3, -0.3, 0.3), 0),
        (rect(0.3, 1.0, -0.8, 0.8), 2),
        (rect(0.3, 2.2, -1.0, 1.0), 4),
    ] {
        let w = sh.mode_winding(&parts(&z), &poly, &opts, true).unwrap();
        println!(
            "box {poly:?}: winding {:.9}, {} evaluations, resolved {}",
            w.turns, w.evals, w.resolved
        );
        assert!(count_ok(&w, want), "{poly:?}: {} (want {want})", w.turns);
    }
    let planted = (z0, OMEGA);
    let dist = |a: (f64, f64)| (a.0 - planted.0).hypot(a.1 - planted.1);
    let w = sh
        .mode_winding(&parts(&z), &rect(0.3, 1.0, 0.2, 0.8), &opts, true)
        .unwrap();
    assert!(!count_ok(&w, 0) && !count_ok(&w, 2), "count control did not fire");
    let m = w.moment();
    let moment_ok = |m: (f64, f64)| dist(m) <= 1e-2;
    assert!(!moment_ok((m.0 + 0.05, m.1)), "moment control did not fire");
    let r = sh.mode_zero(&parts(&z), m, 1e-3, 1e-14, true).unwrap();
    println!(
        "zero: moment {m:?} ({:.1e} off), refined {r:?} ({:.1e} off)",
        dist(m),
        dist(r)
    );
    assert!(moment_ok(m) && dist(r) <= 1e-10);
    let wm = sh
        .mode_winding(&parts(&z), &rect(0.3, 1.0, -0.8, -0.2), &opts, true)
        .unwrap();
    let rc = sh.mode_zero(&parts(&z), wm.moment(), 1e-3, 1e-14, true).unwrap();
    println!("control, from the mirrored box: {rc:?} ({:.2} off)", dist(rc));
    assert!(dist(rc) > 0.5, "control did not fire");
}

/// Pass: `mode_winding`, its batches of samples evaluated side by side by `mode_logdets`, in pools of 1 and 6 threads
/// and without `par`, gives the samples, turns and evaluations of `winding` with `mode_logdet` one rate at a time, bit
/// for bit, on the wave's box (0.3, 1) x (-0.8, 0.8). The control: the box with one corner moved by 1e-3 differs.
#[test]
fn mode_winding_is_the_one_by_one_count() {
    use multisolver::winding::{winding, WindOpts, Winding};
    let (sh, z) = wave_problem(None);
    let opts = WindOpts::default();
    let bits = |w: &Winding| -> (Vec<u64>, u64, usize) {
        let s = w
            .edges
            .iter()
            .flat_map(|e| {
                e.t.iter()
                    .map(|t| t.to_bits())
                    .chain(e.val.iter().flat_map(|v| [v.0.to_bits(), v.1.to_bits()]))
            })
            .collect();
        (s, w.turns.to_bits(), w.evals)
    };
    let one_by_one = |poly: &[(f64, f64)]| {
        let f = |l: (f64, f64)| sh.mode_logdet(&parts(&z), Rate::new(DD::from_f64(l.0), DD::from_f64(l.1)), true);
        bits(&winding(f, poly, &opts).unwrap())
    };
    let poly = rect(0.3, 1.0, -0.8, 0.8);
    let want = one_by_one(&poly);
    for (t, par) in [(1, true), (6, true), (3, false)] {
        let w = in_pool(t, || sh.mode_winding(&parts(&z), &poly, &opts, par).unwrap());
        assert!(bits(&w) == want, "{t} threads, par {par}");
    }
    let mut moved = poly.clone();
    moved[2].0 += 1e-3;
    let w = in_pool(6, || sh.mode_winding(&parts(&z), &moved, &opts, true).unwrap());
    assert!(bits(&w) != want, "control did not fire");
}

// ---------------------------------------------------------------------------------------------------------------
// the mode matrix at a real rate in any precision (`mode_matrix_real`)
// ---------------------------------------------------------------------------------------------------------------

/// the rates of the real-rate checks: zero, and real rates of both signs
const REAL_RATES: [f64; 4] = [0.0, 0.3, -0.45, 1.1];

/// (entries of `real.mode_matrix_real::<Double>` at lr_real that differ in the bits of either part from the real part
/// of `mode.mode_matrix` at lr, entries of the latter's imaginary part not zero)
fn double_differs<E: Equations<B>, const B: usize>(
    mode: &Shooting<E, B>,
    real: &Shooting<E, B>,
    z: &[Vec<f64>],
    lr: f64,
    lr_real: f64,
) -> (usize, usize) {
    let mm = mode
        .mode_matrix(&parts(z), Rate::new(DD::from_f64(lr), DD::ZERO), true)
        .unwrap();
    let (c, ld) = real
        .mode_matrix_real::<multishoot::Double>(&parts(z), DD::from_f64(lr_real), true)
        .unwrap();
    assert_eq!((c.len(), ld), (2, mm.ld));
    let (mut differ, mut nonzero) = (0, 0);
    for j in 0..mm.n {
        for i in 0..mm.n {
            let k = j * ld + i;
            if (0..2).any(|q| mm.re[q][k].to_bits() != c[q][k].to_bits()) {
                differ += 1;
            }
            if mm.im[0][k] != 0.0 || mm.im[1][k] != 0.0 {
                nonzero += 1;
            }
        }
    }
    (differ, nonzero)
}

/// The double-double comparison (`double_differs`) on three layouts: the planted problem plain and with projected
/// and scaled rows (`project_and_scale`), at z* perturbed by 1e-2, and the wave, whose preparation carries the rate.
/// `planted` and `wave` change the real-rate layout of each pair (a control).
fn double_layouts_differ(
    lr: f64,
    lr_real: f64,
    planted: impl Fn(&mut Shooting<Planted<PB>, PB>),
    wave: impl Fn(&mut Shooting<Wave<WB>, WB>),
) -> Vec<(&'static str, (usize, usize))> {
    let (plain, zs) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    let z = perturbed(&zs, 1e-2);
    let (mut plain2, _) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    planted(&mut plain2);
    let (mut proj, _) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    project_and_scale(&mut proj);
    let (mut proj2, _) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    project_and_scale(&mut proj2);
    planted(&mut proj2);
    let (w, wz) = wave_problem(None);
    let (mut w2, _) = wave_problem(None);
    wave(&mut w2);
    vec![
        ("planted", double_differs(&plain, &plain2, &z, lr, lr_real)),
        (
            "planted, projected and scaled",
            double_differs(&proj, &proj2, &z, lr, lr_real),
        ),
        ("wave", double_differs(&w, &w2, &wz, lr, lr_real)),
    ]
}

/// The real-rate mode matrix in double-double (`mode_matrix_real::<Double>`, `Du` lanes) against the real part of
/// `mode_matrix` (`Dc` lanes) at the same rate, on the three layouts of `double_layouts_differ` and the rates
/// `REAL_RATES`. Pass: every entry equal in the bits of both parts, and `mode_matrix`'s imaginary part zero.
#[test]
fn real_rate_matrix_is_mode_matrix_at_double() {
    for lr in REAL_RATES {
        for (name, (differ, nonzero)) in double_layouts_differ(lr, lr, |_| {}, |_| {}) {
            println!("rate {lr}, {name}: {differ} entries differ, {nonzero} imaginary parts nonzero");
            assert_eq!((differ, nonzero), (0, 0), "rate {lr}, {name}");
        }
    }
}

/// The control of `real_rate_matrix_is_mode_matrix_at_double`: the real-rate matrix at 0.3 moved by one ulp must
/// differ from `mode_matrix`'s at 0.3 on every layout.
#[test]
fn real_rate_matrix_control_one_ulp_differs() {
    let lr = 0.3f64;
    let up = f64::from_bits(lr.to_bits() + 1);
    for (name, (differ, _)) in double_layouts_differ(lr, up, |_| {}, |_| {}) {
        println!("control, rate one ulp up, {name}: {differ} entries differ");
        assert!(differ > 0, "{name}: control did not fire");
    }
}

/// max |a - b| / max |b| over the n x n entries of two matrices of column-major part arrays (leading dimension ld),
/// each entry read in P from its first PARTS parts at most (`P::new`), the difference formed in P
fn rel_gap<P: multishoot::Precision<multishoot::Native>>(a: &[Vec<f64>], b: &[Vec<f64>], n: usize, ld: usize) -> f64 {
    let at = |c: &[Vec<f64>], k: usize| P::new(&c.iter().take(P::PARTS).map(|q| q[k]).collect::<Vec<_>>());
    let ks: Vec<usize> = (0..n).flat_map(|j| (0..n).map(move |i| j * ld + i)).collect();
    let d: Vec<P::Scalar> = ks.iter().map(|&k| at(a, k) - at(b, k)).collect();
    let s: Vec<P::Scalar> = ks.iter().map(|&k| at(b, k)).collect();
    multisolver::prec::max_abs::<P>(&d) / multisolver::prec::max_abs::<P>(&s)
}

/// the planted problem at z* perturbed by 1e-2, its layout changed by `set`, and that z in `parts` parts
fn planted_real(
    parts_n: usize,
    set: impl Fn(&mut Shooting<Planted<PB>, PB>),
) -> (Shooting<Planted<PB>, PB>, Vec<Vec<f64>>) {
    let (mut sh, zs) = Planted::<PB>::shooting(&NODES, 30, 1e-32);
    set(&mut sh);
    (sh, multisolver::prec::widen(&perturbed(&zs, 1e-2), parts_n))
}

/// rel_gap of the planted problem's real-rate mode matrix in P (rate lr + i 0, the double-double lr = (lr, lo)) against
/// the one in Q at lr, the gap formed in the lower precision G of the two; P's layout changed by `set`
fn planted_gap<P, Q, G>(lr: f64, lo: f64, set: impl Fn(&mut Shooting<Planted<PB>, PB>)) -> f64
where
    P: multishoot::Precision<multishoot::Native>,
    Q: multishoot::Precision<multishoot::Native>,
    G: multishoot::Precision<multishoot::Native>,
{
    let (sp, zp) = planted_real(P::PARTS, set);
    let (sq, zq) = planted_real(Q::PARTS, |_| {});
    let (a, ld) = sp.mode_matrix_real::<P>(&parts(&zp), DD { hi: lr, lo }, true).unwrap();
    let (b, _) = sq.mode_matrix_real::<Q>(&parts(&zq), DD::from_f64(lr), true).unwrap();
    rel_gap::<G>(&a, &b, sp.off_global(), ld)
}

/// The planted problem's real-rate mode matrix in quad-double against double-double, a well-conditioned case: the
/// Taylor steps, the truncation and the inputs are the same, so the two differ by double-double's rounding of the
/// entries. Pass: max |M_qd - M_dd| / max |M_dd| at most 1e-28 at the rates `REAL_RATES`. The control: the
/// quad-double matrix at the rate 0.3 + 1e-24 (a double-double rate) must exceed it.
#[test]
fn real_rate_matrix_quad_agrees_with_double() {
    use multishoot::{Double, Quad};
    for lr in REAL_RATES {
        let g = planted_gap::<Quad, Double, Quad>(lr, 0.0, |_| {});
        println!("rate {lr}: quad-double against double-double, relative gap {g:.2e}");
        assert!(g <= 1e-28, "rate {lr}: {g:e}");
    }
    let g = planted_gap::<Quad, Double, Quad>(0.3, 1e-24, |_| {});
    println!("control, quad-double at 0.3 + 1e-24: relative gap {g:.2e}");
    assert!(g > 1e-28, "control did not fire");
}

/// The planted problem's real-rate mode matrix in octo-double against quad-double: the two differ by quad-double's
/// rounding, so the quad-double entries carry no double-double rounding. Pass: max |M_od - M_qd| / max |M_qd| at most
/// 1e-58 at the rates `REAL_RATES`. The control: the quad-double matrix at the rate 0.3 + 1e-50 must exceed it.
#[test]
fn real_rate_matrix_octo_agrees_with_quad() {
    use multishoot::{Octo, Quad};
    for lr in REAL_RATES {
        let g = planted_gap::<Octo, Quad, Quad>(lr, 0.0, |_| {});
        println!("rate {lr}: octo-double against quad-double, relative gap {g:.2e}");
        assert!(g <= 1e-58, "rate {lr}: {g:e}");
    }
    let g = planted_gap::<Quad, Octo, Quad>(0.3, 1e-50, |_| {});
    println!("control, quad-double at 0.3 + 1e-50: relative gap {g:.2e}");
    assert!(g > 1e-58, "control did not fire");
}

/// (entries of the wave's real-rate mode matrix in P at the rate mu that differ in any part from the leading block
/// of the Jacobian in P of the wave carrying mu_explicit as values)
fn explicit_differs<P: multishoot::Precision<multishoot::Native>>(mu: f64, mu_explicit: f64) -> usize {
    let (sh, z) = wave_problem(None);
    let zp = multisolver::prec::widen(&z, P::PARTS);
    let (mm, ld) = sh.mode_matrix_real::<P>(&parts(&zp), DD::from_f64(mu), true).unwrap();
    let (ex, z) = wave_problem(Some(DD::from_f64(mu_explicit)));
    let zp = multisolver::prec::widen(&z, P::PARTS);
    let (c, ldj) = ex.jacobian::<P>(&parts(&zp), true).unwrap();
    let n = sh.off_global();
    (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .filter(|&(i, j)| (0..P::PARTS).any(|q| mm[q][j * ld + i].to_bits() != c[q][j * ldj + i].to_bits()))
        .count()
}

/// The real-rate mode matrix in quad- and octo-double against the Jacobian in the same precision of the wave that
/// carries the rate as values (`explicit` = Some(mu)), as `real_rate_is_the_explicit_shift` checks `mode_matrix`:
/// `Rate::shift` on `Qu` and `Ou` lanes against the explicit mu u, the same operations on the tangents. Pass: every
/// entry equal in the bits of every part at mu = 0.37. The control: the Jacobian with mu = 0.38 must differ.
#[test]
fn real_rate_is_the_explicit_shift_in_quad_and_octo() {
    use multishoot::{Octo, Quad};
    let (q, o) = (
        explicit_differs::<Quad>(0.37, 0.37),
        explicit_differs::<Octo>(0.37, 0.37),
    );
    let (qc, oc) = (
        explicit_differs::<Quad>(0.37, 0.38),
        explicit_differs::<Octo>(0.37, 0.38),
    );
    println!("real rate against the explicit shift: quad {q}, octo {o} entries differ; controls (mu 0.38) {qc}, {oc}");
    assert_eq!((q, o), (0, 0));
    assert!(qc > 0 && oc > 0, "control did not fire");
}

/// For each real rate of `REAL_RATES`: |ln|det M| - ln|det|_planted| of the wave's real-rate mode matrix in
/// quad-double, factored by multidouble's `QDLu`
fn quad_determinant_errors(sh: &Shooting<Wave<WB>, WB>, z: &[Vec<f64>]) -> Vec<f64> {
    let zq = multisolver::prec::widen(z, 4);
    REAL_RATES
        .iter()
        .map(|&lr| {
            let (c, ld) = sh
                .mode_matrix_real::<multishoot::Quad>(&parts(&zq), DD::from_f64(lr), true)
                .unwrap();
            let n = sh.off_global();
            let lu = multishoot::multidouble::linalg::QDLu::factor(n, ld, c.try_into().unwrap(), true).unwrap();
            (lu.logdet().1 - planted_logdet(lr, 0.0).0).abs()
        })
        .collect()
}

/// The wave's det M(lambda) in quad-double against its closed form at the real rates `REAL_RATES`. Pass:
/// |ln|det M| - ln|det|| at most 1e-10 (the f64 sums of the pivots' and the closed form's logarithms).
#[test]
fn wave_quad_determinant_is_planted() {
    let (sh, z) = wave_problem(None);
    let e = quad_determinant_errors(&sh, &z);
    let shown: Vec<String> = e.iter().map(|x| format!("{x:.1e}")).collect();
    println!("quad-double det M against the closed form (ln): {}", shown.join(" "));
    assert!(e.iter().all(|&x| x <= 1e-10), "{e:?}");
}

/// a quad-double and an octo-double dual field of 8 values
type QField = Fl<multishoot::lane::Qu, 1>;
type OField = Fl<multishoot::lane::Ou, 1>;

/// x with the tangent e_3 and dx with the tangent e_5, in quad-double and in octo-double, for the shift checks
fn shift_inputs() -> ((QField, QField), (OField, OField)) {
    use multishoot::lane::{Ou, Qu, OV, QV};
    let q = |v: f64, j: usize| Fl::<Qu, 1>::from_vt_qd(&Fl::<QV, 1>::splat(DD::from_f64(v)), &Fl::<QV, 1>::unit(j));
    let o = |v: f64, j: usize| Fl::<Ou, 1>::from_vt_od(&Fl::<OV, 1>::splat(DD::from_f64(v)), &Fl::<OV, 1>::unit(j));
    ((q(0.7, 3), q(-1.3, 5)), (o(0.7, 3), o(-1.3, 5)))
}

/// (tangent entries of s other than 0.5 at point 3, 1 at point 5 and 0 elsewhere, entries whose value parts differ
/// from dx's), for s the shift by 0.5 of `shift_inputs`' dx, in quad-double and in octo-double
fn shift_errors(sq: &QField, so: &OField) -> [(usize, usize); 2] {
    let ((_, dq), (_, do_)) = shift_inputs();
    let want = |j: usize| [(3, 0.5), (5, 1.0)].iter().find(|w| w.0 == j).map_or(0.0, |w| w.1);
    let qd = |x: f64| multishoot::multidouble::qd::QD::from_dd(DD::from_f64(x));
    let od = |x: f64| multishoot::multidouble::od::OD::from_dd(DD::from_f64(x));
    let count = |f: &dyn Fn(usize) -> bool| (0..8).filter(|&j| f(j)).count();
    [
        (
            count(&|j| sq.coef_qd(1).get(j) != qd(want(j))),
            count(&|j| sq.coef_qd(0).get(j).x != dq.coef_qd(0).get(j).x),
        ),
        (
            count(&|j| so.coef_od(1).get(j) != od(want(j))),
            count(&|j| so.coef_od(0).get(j).x != do_.coef_od(0).get(j).x),
        ),
    ]
}

/// A real rate on quad- and octo-double tangents: pass, the tangent at the point that x's tangent seeds moves by the
/// rate times it, the tangent dx seeds keeps its value, and every value keeps its parts (`shift_errors` all zero).
#[test]
fn shift_real_rate_moves_quad_and_octo_tangents() {
    let r = Rate::new(DD::from_f64(0.5), DD::ZERO);
    let ((xq, dq), (xo, do_)) = shift_inputs();
    let e = shift_errors(&r.shift(&xq, dq), &r.shift(&xo, do_));
    println!("shift by 0.5 on quad and octo tangents (tangent, value errors): {e:?}");
    assert_eq!(e, [(0, 0); 2]);
}

/// The controls of `shift_real_rate_moves_quad_and_octo_tangents`: the unshifted dx must fail the tangent check, and
/// the shifted result with one value's third part moved by one ulp must fail the value check, in both precisions.
#[test]
fn shift_quad_and_octo_controls_differ() {
    let r = Rate::new(DD::from_f64(0.5), DD::ZERO);
    let ((xq, dq), (xo, do_)) = shift_inputs();
    let unshifted = shift_errors(&dq, &do_);
    let (mut sq, mut so) = (r.shift(&xq, dq), r.shift(&xo, do_));
    // the third part of every point's value, its lowest bit up (a zero part becomes the least normal number)
    let bump = |v: multishoot::multidouble::native::V8| {
        multishoot::multidouble::native::V8::from_array(v.to_array().map(|x| {
            if x == 0.0 {
                f64::MIN_POSITIVE
            } else {
                f64::from_bits(x.to_bits() + 1)
            }
        }))
    };
    sq.0[0].v.c[2] = bump(sq.0[0].v.c[2]);
    so.0[0].v.c[2] = bump(so.0[0].v.c[2]);
    let moved = shift_errors(&sq, &so);
    println!("controls (tangent, value errors): unshifted {unshifted:?}, a value part moved {moved:?}");
    assert!(unshifted.iter().all(|e| e.0 > 0), "tangent control did not fire");
    assert!(moved.iter().all(|e| e.1 > 0), "value control did not fire");
}

/// A complex rate on quad-double tangents cannot be carried: pass, it panics.
#[test]
#[should_panic(expected = "a complex one Dc lanes")]
fn complex_rate_on_quad_tangents_panics() {
    use multishoot::lane::{Qu, QV};
    let x = Fl::<Qu, 1>::from_vt_qd(&Fl::<QV, 1>::splat(DD::ONE), &Fl::<QV, 1>::unit(0));
    Rate::new(DD::ZERO, DD::ONE).shift(&x, x);
}

/// The controls of the real-rate checks (feature `mutate`), each run once against the check it breaks.
#[cfg(feature = "mutate")]
mod real_controls {
    use super::*;
    use multishoot::control::Control;

    /// the real-rate matrix with its Taylor steps at rate zero: it differs from `mode_matrix`'s real part on the
    /// planted layouts at the rate 0.3
    #[test]
    fn rate_not_in_steps_fails_the_double_identity() {
        let r = double_layouts_differ(0.3, 0.3, |sh| sh.set_control(Control::RateNotInSteps), |_| {});
        for (name, (differ, _)) in &r[..2] {
            println!("control, real-rate matrix with its steps at rate zero, {name}: {differ} entries differ");
            assert!(*differ > 0, "{name}: control did not fire");
        }
    }

    /// the real-rate matrix prepared without the rate: on the wave, whose right endpoint map steps at the
    /// preparation's rate, it differs from `mode_matrix`'s real part at 0.3
    #[test]
    fn rate_not_prepared_fails_the_double_identity() {
        let r = double_layouts_differ(0.3, 0.3, |_| {}, |sh| sh.set_control(Control::RateNotPrepared));
        let (name, (differ, _)) = r[2];
        println!("control, real-rate matrix not prepared, {name}: {differ} entries differ");
        assert!(differ > 0, "control did not fire");
    }

    /// the quad-double matrix with its Taylor steps at rate zero: its gap to double-double at 0.3 exceeds 1e-28
    #[test]
    fn rate_not_in_steps_fails_the_quad_agreement() {
        use multishoot::{Double, Quad};
        let g = planted_gap::<Quad, Double, Quad>(0.3, 0.0, |sh| sh.set_control(Control::RateNotInSteps));
        println!("control, quad-double steps at rate zero: relative gap {g:.2e}");
        assert!(g > 1e-28, "control did not fire");
    }

    /// the wave's quad-double matrix prepared without the rate: the closed form fails at the nonzero rates
    #[test]
    fn rate_not_prepared_fails_the_quad_determinant() {
        let (mut sh, z) = wave_problem(None);
        sh.set_control(Control::RateNotPrepared);
        let worst = quad_determinant_errors(&sh, &z).iter().fold(0.0f64, |a, &e| a.max(e));
        println!("control, quad-double not prepared: worst determinant error {worst:.3e}");
        assert!(worst > 1e-10, "control did not fire");
    }

    /// the quad-double matrix with the rate's shift dropped from the Taylor steps: the explicit-shift identity fails
    #[test]
    fn rate_not_in_steps_fails_the_quad_explicit_shift() {
        use multishoot::Quad;
        let (mut sh, z) = wave_problem(None);
        sh.set_control(Control::RateNotInSteps);
        let zq = multisolver::prec::widen(&z, 4);
        let (mm, ld) = sh
            .mode_matrix_real::<Quad>(&parts(&zq), DD::from_f64(0.37), true)
            .unwrap();
        let (ex, z) = wave_problem(Some(DD::from_f64(0.37)));
        let zq = multisolver::prec::widen(&z, 4);
        let (c, ldj) = ex.jacobian::<Quad>(&parts(&zq), true).unwrap();
        let n = sh.off_global();
        let differ = (0..n)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .filter(|&(i, j)| (0..4).any(|q| mm[q][j * ld + i].to_bits() != c[q][j * ldj + i].to_bits()))
            .count();
        println!("control, quad-double steps at rate zero against the explicit shift: {differ} entries differ");
        assert!(differ > 0, "control did not fire");
    }
}
