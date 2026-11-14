//! The weak subspace, the sensitivities and Newton with a subspace removed, on a linear problem with a planted singular
//! value decomposition: A = R diag(sigma) S^T with R and S products of Householder reflections I - v v^T / 2 (v of
//! four ones, so every entry is dyadic and A is exact in double-double) and sigma = 2^-40, 2^-30, 1, 1.5, ...: its
//! weakest right singular directions are the first two columns of S, with singular values 2^-40 and 2^-30.
use multishoot::multidouble::linalg::ld_for;
use multishoot::newton::NewtonOpts;
use multishoot::prec::{nan_max, parts};
use multishoot::subspace::{newton_deflated, weak_subspace, Factors};
use multishoot::{Double, Problem, DD};

const N: usize = 8;

/// I - v v^T / 2 for v with ones at the four indices given
fn householder(ones: [usize; 4]) -> Vec<Vec<DD>> {
    (0..N)
        .map(|i| {
            (0..N)
                .map(|j| {
                    let id = if i == j { 1.0 } else { 0.0 };
                    let vv = if ones.contains(&i) && ones.contains(&j) {
                        0.5
                    } else {
                        0.0
                    };
                    DD::from_f64(id - vv)
                })
                .collect()
        })
        .collect()
}

fn mul(a: &[Vec<DD>], b: &[Vec<DD>]) -> Vec<Vec<DD>> {
    (0..N)
        .map(|i| {
            (0..N)
                .map(|j| (0..N).fold(DD::ZERO, |t, k| t + a[i][k] * b[k][j]))
                .collect()
        })
        .collect()
}

fn sigma() -> Vec<f64> {
    (0..N)
        .map(|i| match i {
            0 => 2f64.powi(-40),
            1 => 2f64.powi(-30),
            _ => 1.0 + 0.5 * (i - 2) as f64,
        })
        .collect()
}

/// (A, S): A = R diag(sigma) S^T by rows, and S whose columns are A's right singular directions
fn planted() -> (Vec<Vec<DD>>, Vec<Vec<DD>>) {
    let r = mul(&householder([0, 1, 2, 3]), &householder([2, 3, 4, 5]));
    let s = mul(&householder([1, 2, 5, 6]), &householder([0, 3, 4, 7]));
    let sg = sigma();
    let a = (0..N)
        .map(|i| {
            (0..N)
                .map(|j| (0..N).fold(DD::ZERO, |t, k| t + r[i][k] * DD::from_f64(sg[k]) * s[j][k]))
                .collect()
        })
        .collect();
    (a, s)
}

/// F(z) = A z - b
struct Lin {
    a: Vec<Vec<DD>>,
    b: Vec<DD>,
}

impl Problem<Double> for Lin {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        Ok((0..N)
            .map(|i| (0..N).fold(DD::ZERO, |t, j| t + self.a[i][j] * DD::new(z[0][j], z[1][j])) - self.b[i])
            .collect())
    }
    fn jacobian(&self, _z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; 2];
        for (i, row) in self.a.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                (c[0][j * ld + i], c[1][j * ld + i]) = (v.hi, v.lo);
            }
        }
        Ok((c, ld))
    }
}

/// the planted problem with root z*_i = 1 + i / 8, and z*'s parts
fn lin() -> (Lin, Vec<Vec<f64>>) {
    let (a, _) = planted();
    let zs: Vec<DD> = (0..N).map(|i| DD::from_f64(1.0 + i as f64 / 8.0)).collect();
    let b = (0..N)
        .map(|i| (0..N).fold(DD::ZERO, |t, j| t + a[i][j] * zs[j]))
        .collect();
    (
        Lin { a, b },
        vec![zs.iter().map(|v| v.hi).collect(), zs.iter().map(|v| v.lo).collect()],
    )
}

/// the distance of the unit vector x from the span of S's first two columns
fn off_span(x: &[f64], s: &[Vec<DD>]) -> f64 {
    let cols: Vec<Vec<f64>> = (0..2).map(|k| (0..N).map(|i| s[i][k].hi).collect()).collect();
    let mut r = x.to_vec();
    for col in &cols {
        let c: f64 = col.iter().zip(x).map(|(a, b)| a * b).sum();
        r.iter_mut().zip(col).for_each(|(ri, ci)| *ri -= c * ci);
    }
    r.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// weak_subspace of width 2 after `iters` products: (its basis, its singular values)
fn weak(iters: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (pr, zs) = lin();
    let f = Factors::<Double>::new(&pr, &parts(&zs), false).unwrap();
    weak_subspace::<Double>(&f, 2, iters, 7)
}

/// (largest distance of the basis from the planted span, the relative errors of the two singular values)
fn weak_errors(basis: &[Vec<f64>], sg: &[f64]) -> (f64, f64, f64) {
    let (_, s) = planted();
    let d = basis.iter().map(|b| off_span(b, &s)).fold(0.0, nan_max);
    let want = sigma();
    (d, (sg[0] / want[0] - 1.0).abs(), (sg[1] / want[1] - 1.0).abs())
}

/// the checks of a weak subspace: [its vectors in the planted span to 1e-12, each singular value to 1e-10 relative]
fn weak_checks((d, e0, e1): (f64, f64, f64)) -> [bool; 3] {
    [d <= 1e-12, e0 <= 1e-10, e1 <= 1e-10]
}

/// Pass, written before the first run: after 4 inverse iterations the subspace's vectors lie in the planted span to
/// 1e-12 and its singular values match 2^-40 and 2^-30 to 1e-10 relative. The controls: without iterations the
/// random start lies far from the span (more than 1e-2), and the converged singular values off by 1e-8 relative fail
/// both of their checks.
#[test]
fn weak_subspace_finds_the_planted_directions() {
    let (basis, sg) = weak(4);
    let (d, e0, e1) = weak_errors(&basis, &sg);
    println!("weak subspace: off the span {d:.3e}, sigma errors {e0:.3e} {e1:.3e}");
    assert_eq!(weak_checks((d, e0, e1)), [true; 3]);
    let (b0, sg0) = weak(0);
    let (d, _, _) = weak_errors(&b0, &sg0);
    println!("control, no iterations: off the span {d:.3e}");
    assert!(
        d > 1e-2 && !weak_checks(weak_errors(&b0, &sg0))[0],
        "control did not fire"
    );
    let off: Vec<f64> = sg.iter().map(|v| v * (1.0 + 1e-8)).collect();
    let e = weak_errors(&basis, &off);
    println!(
        "control, singular values off by 1e-8: sigma errors {:.3e} {:.3e}",
        e.1, e.2
    );
    assert_eq!(weak_checks(e), [true, false, false], "control did not fire");
}

/// the diagonal of [`Scaled`]'s Jacobian over s: the weak pair 1, 2 well apart from the rest, so that 4 inverse
/// iterations resolve it
const SCALED: [f64; N] = [1.0, 2.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0];

/// F(z) = s diag(SCALED) (z - 1): a uniformly scaled Jacobian whose weakest directions are e_0 and e_1, with
/// singular values s and 2 s
struct Scaled(f64);

impl Problem<Double> for Scaled {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        Ok((0..N)
            .map(|i| DD::from_f64(self.0 * SCALED[i]) * (DD::new(z[0][i], z[1][i]) - DD::ONE))
            .collect())
    }
    fn jacobian(&self, _z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; 2];
        for i in 0..N {
            c[0][i * ld + i] = self.0 * SCALED[i];
        }
        Ok((c, ld))
    }
}

/// Pass: at s = 1e-100 and 1e100, where (J^T J)^-1 reaches 1e200 and 1e-200 and the
/// squares of the subspace's entries overflow and underflow, weak_subspace of width 2 after 4 iterations gives
/// vectors within 1e-12 of span(e_0, e_1) and singular values s and 2 s to 1e-10 relative. The control: without
/// iterations the random start lies far from that span (more than 1e-2).
#[test]
fn weak_subspace_at_extreme_scales() {
    let zero = vec![vec![0.0; N]; 2];
    for scale in [1e-100, 1e100] {
        let f = Factors::<Double>::new(&Scaled(scale), &parts(&zero), false).unwrap();
        let run = |iters: usize| {
            let (basis, sg) = weak_subspace::<Double>(&f, 2, iters, 7);
            let d = basis
                .iter()
                .map(|b| b[2..].iter().map(|v| v * v).sum::<f64>().sqrt())
                .fold(0.0, nan_max);
            (d, (sg[0] / scale - 1.0).abs(), (sg[1] / (2.0 * scale) - 1.0).abs())
        };
        let (d, e0, e1) = run(4);
        println!("s {scale:e}: off the span {d:.3e}, sigma errors {e0:.3e} {e1:.3e}");
        assert_eq!(weak_checks((d, e0, e1)), [true; 3]);
        let (d, _, _) = run(0);
        println!("control, no iterations: off the span {d:.3e}");
        assert!(d > 1e-2, "control did not fire");
    }
}

/// |A^T y - e_k| / (|A| |y|) for y the sensitivity of unknown k, or (control) J^-1 e_k in its place
fn sensitivity_residual(k: usize, transpose: bool) -> f64 {
    let (pr, zs) = lin();
    let f = Factors::<Double>::new(&pr, &parts(&zs), false).unwrap();
    let y = if transpose {
        f.sensitivity(k)
    } else {
        let mut e = vec![DD::ZERO; N];
        e[k] = DD::ONE;
        f.solve(&e)
    };
    let amax = pr.a.iter().flatten().map(|v| v.hi.abs()).fold(0.0, f64::max);
    let ymax = y.iter().map(|v| v.hi.abs()).fold(0.0, f64::max);
    (0..N)
        .map(|j| {
            let aty = (0..N).fold(DD::ZERO, |t, i| t + pr.a[i][j] * y[i]);
            (aty - if j == k { DD::ONE } else { DD::ZERO }).hi.abs()
        })
        .fold(0.0, f64::max)
        / (amax * ymax)
}

/// Pass, written before the first run: the sensitivity y = J^-T e_k solves A^T y = e_k to 1e-30 relative to |A| |y|,
/// for every k. The control: J^-1 e_k, the solve without the transpose, misses for this unsymmetric A by more than
/// 1e-3.
#[test]
fn sensitivity_solves_the_transpose() {
    let e = (0..N).map(|k| sensitivity_residual(k, true)).fold(0.0, f64::max);
    let c = sensitivity_residual(3, false);
    println!("sensitivity: residual {e:.3e}; control, J^-1 e_k {c:.3e}");
    assert!(e <= 1e-30);
    assert!(c > 1e-3, "control did not fire");
}

/// Pass, written before the first run: Newton from z* + delta with the planted weak span removed from its steps leaves
/// delta's component along the span unchanged (to 1e-25) and removes the rest (to 1e-25). The control: plain Newton
/// moves along the span, by more than 1e-6.
#[test]
fn deflated_newton_never_moves_along_the_subspace() {
    let (pr, zs) = lin();
    let (_, s) = planted();
    let f = Factors::<Double>::new(&pr, &parts(&zs), false).unwrap();
    let (basis, _) = weak_subspace::<Double>(&f, 2, 6, 7);
    let start: Vec<Vec<f64>> = vec![
        zs[0]
            .iter()
            .enumerate()
            .map(|(i, v)| v + 1e-3 * (1.0 + i as f64 / 3.0))
            .collect(),
        zs[1].clone(),
    ];
    // (|component of z - start along the span|, |component of z - z* off the span|)
    let measure = |z: &[Vec<f64>]| {
        let dz: Vec<f64> = (0..N)
            .map(|i| (DD::new(z[0][i], z[1][i]) - DD::new(start[0][i], start[1][i])).hi)
            .collect();
        let de: Vec<f64> = (0..N)
            .map(|i| (DD::new(z[0][i], z[1][i]) - DD::new(zs[0][i], zs[1][i])).hi)
            .collect();
        let along: f64 = (0..2)
            .map(|k| (0..N).map(|i| s[i][k].hi * dz[i]).sum::<f64>().abs())
            .fold(0.0, nan_max);
        (along, off_span(&de, &s))
    };
    // [unmoved along the span to 1e-25, converged off it to 1e-25]
    let checks = |(along, off): (f64, f64)| [along <= 1e-25, off <= 1e-25];
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 3,
        ..Default::default()
    };
    let mut z = start.clone();
    newton_deflated::<Double, _>(&pr, &mut z, &opts, &basis).unwrap();
    let (along, off) = measure(&z);
    println!("deflated Newton: moved along the span {along:.3e}, left off the span {off:.3e}");
    assert_eq!(checks((along, off)), [true; 2]);
    let mut z = start.clone();
    multishoot::newton::newton::<Double, _>(&pr, &mut z, &opts).unwrap();
    let (along, off) = measure(&z);
    println!("control, plain Newton: moved along the span {along:.3e}, left off it {off:.3e}");
    assert!(
        along > 1e-6 && checks((along, off)) == [false, true],
        "control did not fire"
    );
    let (along, off) = measure(&start);
    println!("control, the start: moved along the span {along:.3e}, left off it {off:.3e}");
    assert_eq!(checks((along, off)), [true, false], "control did not fire");
}
