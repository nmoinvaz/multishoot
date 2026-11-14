//! The asymptotic numerical method on two planted branches: a cyclic system whose root is a quadratic polynomial in
//! the parameter, and the eigenproblem `Cylinder`, whose eigenvalue pi^2 + mu is linear in its parameter mu while its
//! mode does not change.
mod common;

use common::{perturbed, Cylinder};
use multishoot::anm::{series, Branch};
use multishoot::multidouble::linalg::ld_for;
use multishoot::prec::{nan_max, parts};
use multishoot::{Double, Jvp, Native, NewtonOpts, Precision, Problem, Quad, ResidualJet, DD};

const N: usize = 10;
const KAPPA: f64 = 0.3;
const LAM0: f64 = 0.7;

/// F_i(z, lam) = z_i^3 - w_i(lam)^3 + kappa (z_(i+1) - w_(i+1)(lam)), the index cyclic, with the quadratic
/// w_i(lam) = a_i + b_i lam + c_i lam^2, so the branch through lam0 is z(lam0 + h) = w(lam0 + h): z_1 = b + 2 c lam0,
/// z_2 = c, z_k = 0 for k >= 3. `freeze`: the series with lam a constant (the control). `nan_jvp`: the products J v
/// all NaN (a broken refinement, the control of the recorded residuals).
struct Poly<P: Precision<Native>> {
    a: Vec<P::Scalar>,
    b: Vec<P::Scalar>,
    c: Vec<P::Scalar>,
    freeze: bool,
    nan_jvp: bool,
}

fn s<P: Precision<Native>>(x: f64) -> P::Scalar {
    P::from_dd(DD::from_f64(x))
}

fn scalars<P: Precision<Native>>(z: &[&[f64]]) -> Vec<P::Scalar> {
    (0..z[0].len())
        .map(|i| P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>()))
        .collect()
}

fn to_parts<P: Precision<Native>>(v: &[P::Scalar]) -> Vec<Vec<f64>> {
    (0..P::PARTS)
        .map(|q| v.iter().map(|&x| P::part(x, q)).collect())
        .collect()
}

/// the product of two series truncated to n terms
fn conv<P: Precision<Native>>(x: &[P::Scalar], y: &[P::Scalar], n: usize) -> Vec<P::Scalar> {
    (0..n)
        .map(|k| (0..=k).fold(P::ZERO, |t, j| t + x[j] * y[k - j]))
        .collect()
}

impl<P: Precision<Native>> Poly<P> {
    fn new(freeze: bool) -> Self {
        Poly {
            a: (0..N).map(|i| s::<P>(1.0 + 0.1 * (i as f64).sin())).collect(),
            b: (0..N).map(|i| s::<P>(0.2 * (i as f64).cos())).collect(),
            c: (0..N).map(|i| s::<P>(0.05 * (2.0 * i as f64).sin())).collect(),
            freeze,
            nan_jvp: false,
        }
    }
    /// w_i(lam0 + h) as a series of n terms
    fn w(&self, i: usize, n: usize) -> Vec<P::Scalar> {
        let l = s::<P>(LAM0);
        let two = s::<P>(2.0);
        let mut w = vec![P::ZERO; n.max(3)];
        w[0] = self.a[i] + self.b[i] * l + self.c[i] * l * l;
        if !self.freeze {
            w[1] = self.b[i] + two * self.c[i] * l;
            w[2] = self.c[i];
        }
        w.truncate(n);
        w
    }
    /// the root at lam0
    fn root(&self) -> Vec<Vec<f64>> {
        to_parts::<P>(&(0..N).map(|i| self.w(i, 1)[0]).collect::<Vec<_>>())
    }
}

impl<P: Precision<Native>> Problem<P> for Poly<P> {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        Ok(self.residual_jet(&[z.to_vec()], 1, par)?.remove(0))
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; P::PARTS];
        for i in 0..N {
            let d = s::<P>(3.0) * z[i] * z[i];
            for (q, cq) in c.iter_mut().enumerate() {
                cq[i * ld + i] = P::part(d, q);
                cq[((i + 1) % N) * ld + i] = P::part(s::<P>(KAPPA), q);
            }
        }
        Ok((c, ld))
    }
}

impl<P: Precision<Native>> Jvp<P> for Poly<P> {
    fn jvp(&self, z: &[&[f64]], v: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let (z, v) = (scalars::<P>(z), scalars::<P>(v));
        if self.nan_jvp {
            return Ok(vec![s::<P>(f64::NAN); N]);
        }
        Ok((0..N)
            .map(|i| s::<P>(3.0) * z[i] * z[i] * v[i] + s::<P>(KAPPA) * v[(i + 1) % N])
            .collect())
    }
}

impl<P: Precision<Native>> ResidualJet<P> for Poly<P> {
    fn residual_jet(&self, zs: &[Vec<&[f64]>], nout: usize, _par: bool) -> Result<Vec<Vec<P::Scalar>>, String> {
        let zk: Vec<Vec<P::Scalar>> = zs.iter().map(|z| scalars::<P>(z)).collect();
        // the series of unknown i, n terms
        let ser = |i: usize| -> Vec<P::Scalar> {
            (0..nout)
                .map(|k| if k < zk.len() { zk[k][i] } else { P::ZERO })
                .collect()
        };
        let cube = |x: &[P::Scalar]| conv::<P>(&conv::<P>(x, x, nout), x, nout);
        let rows: Vec<Vec<P::Scalar>> = (0..N)
            .map(|i| {
                let j = (i + 1) % N;
                let (zi, zj, wi, wj) = (ser(i), ser(j), self.w(i, nout), self.w(j, nout));
                let (z3, w3) = (cube(&zi), cube(&wi));
                (0..nout)
                    .map(|k| z3[k] - w3[k] + s::<P>(KAPPA) * (zj[k] - wj[k]))
                    .collect()
            })
            .collect();
        Ok((0..nout).map(|k| rows.iter().map(|r| r[k]).collect()).collect())
    }
}

/// The series of order 8 at the root of the polynomial branch in P, refined once
fn poly_branch<P: Precision<Native>>(freeze: bool, nan_jvp: bool) -> Branch<P> {
    let mut pr = Poly::<P>::new(freeze);
    pr.nan_jvp = nan_jvp;
    let z0 = pr.root();
    series::<P, _>(&pr, &parts(&z0), 8, 1, false).unwrap()
}

/// a series of order 8 against the true branch: [|z_1 - w'| / |w'|, |z_2 - c| / |c|, max over k >= 3 of |z_k|,
/// |z(0.3) - w(lam0 + 0.3)| / |w(lam0 + 0.3)|], max norms, NaN where any entry is
fn poly_errors<P: Precision<Native>>(br: &Branch<P>) -> [f64; 4] {
    let truth = Poly::<P>::new(false);
    let rel = |x: &[P::Scalar], y: &[P::Scalar]| {
        let d = x
            .iter()
            .zip(y)
            .map(|(&a, &b)| P::to_dd(a - b).hi.abs())
            .fold(0.0, nan_max);
        d / y.iter().map(|&b| P::to_dd(b).hi.abs()).fold(0.0, nan_max)
    };
    let w1: Vec<P::Scalar> = (0..N).map(|i| truth.w(i, 2)[1]).collect();
    let high = (3..=8)
        .flat_map(|k| br.z[k].iter().map(|&v| P::to_dd(v).hi.abs()))
        .fold(0.0, nan_max);
    let wa: Vec<P::Scalar> = (0..N)
        .map(|i| {
            let w = truth.w(i, 3);
            let a = s::<P>(0.3);
            w[0] + (w[1] + w[2] * a) * a
        })
        .collect();
    [
        rel(&br.z[1], &w1),
        rel(&br.z[2], &truth.c),
        high,
        rel(&br.eval(0.3), &wa),
    ]
}

/// the accuracy checks of the polynomial branch: [z_1, z_2, z(0.3)] within tol, [z_3..z_8] within tol_high
fn poly_checks(e: [f64; 4], tol: f64, tol_high: f64) -> [bool; 4] {
    [e[0] <= tol, e[1] <= tol, e[2] <= tol_high, e[3] <= tol]
}

/// Pass, written before the first run: at the root of the polynomial branch the series of order 8 gives z_1 and z_2
/// to 1e-29 relative, every higher order below 1e-28, and z(0.3) to 1e-29, in double-double; in quad-double the same
/// to 1e-60 and 1e-59. The controls, each a wrong series in double-double: with the parameter frozen in the
/// residual's series, z_1, z_2 and z(0.3) miss (its higher orders are zero); z_2 off by 1e-20 fails z_2 and z(0.3);
/// z_3 = 1e-20 in one entry, and a NaN in z_5, fail the higher orders and z(0.3).
#[test]
fn anm_reproduces_a_polynomial_branch() {
    let d = poly_errors(&poly_branch::<Double>(false, false));
    println!(
        "double-double: z_1 {:.3e}, z_2 {:.3e}, z_3..z_8 {:.3e}, z(0.3) {:.3e}",
        d[0], d[1], d[2], d[3]
    );
    assert_eq!(poly_checks(d, 1e-29, 1e-28), [true; 4]);
    let q = poly_errors(&poly_branch::<Quad>(false, false));
    println!(
        "quad-double: z_1 {:.3e}, z_2 {:.3e}, z_3..z_8 {:.3e}, z(0.3) {:.3e}",
        q[0], q[1], q[2], q[3]
    );
    assert_eq!(poly_checks(q, 1e-60, 1e-59), [true; 4]);
    let control = |name: &str, br: &Branch<Double>, want: [bool; 4]| {
        let e = poly_errors(br);
        println!(
            "control, {name}: z_1 {:.3e}, z_2 {:.3e}, z_3..z_8 {:.3e}, z(0.3) {:.3e}",
            e[0], e[1], e[2], e[3]
        );
        assert_eq!(poly_checks(e, 1e-29, 1e-28), want, "control {name} did not fire");
    };
    control(
        "parameter frozen",
        &poly_branch::<Double>(true, false),
        [false, false, true, false],
    );
    let good = poly_branch::<Double>(false, false);
    let mut br = good.clone();
    br.z[2][0] = br.z[2][0] + DD::from_f64(1e-20);
    control("z_2 off by 1e-20", &br, [true, false, true, false]);
    let mut br = good.clone();
    br.z[3][0] = DD::from_f64(1e-20);
    control("z_3 = 1e-20", &br, [true, true, false, false]);
    let mut br = good;
    br.z[5][0] = DD::from_f64(f64::NAN);
    control("NaN in z_5", &br, [true, true, false, false]);
}

/// Pass, written before the first run: the solves of the polynomial branch's eight orders have relative residuals of
/// at most 1e-30 after one refinement, and its frozen branch, constant in the parameter, has an infinite radius. The
/// control: with the products J v all NaN every recorded residual is NaN, and the check rejects them.
#[test]
fn anm_records_its_solve_residuals() {
    let check = |br: &Branch<Double>| br.residual.iter().copied().fold(0.0, nan_max) <= 1e-30;
    let br = poly_branch::<Double>(false, false);
    println!("residuals: {:?}", br.residual);
    assert!(check(&br));
    let frozen = poly_branch::<Double>(true, false);
    println!("frozen branch: radius {}", frozen.radius(1e-8));
    assert_eq!(frozen.radius(1e-8), f64::INFINITY);
    let bad = poly_branch::<Double>(false, true);
    println!("control, products NaN: residuals {:?}", bad.residual);
    assert!(
        bad.residual.iter().all(|r| r.is_nan()) && !check(&bad),
        "control did not fire"
    );
}

const B: usize = 1;

/// The series of order `order` of `Cylinder`'s branch in mu at its double-double root, refined once, and the index
/// of the eigenvalue among the unknowns
fn cylinder(freeze: bool, order: usize) -> (Branch<Double>, usize) {
    let (mut sh, zs) = Cylinder::<B>::shooting(Cylinder::new(), 30);
    let mut z = perturbed(&zs, 1e-3);
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 12,
        ..Default::default()
    };
    assert!(sh.newton::<Double>(&mut z, &opts).unwrap().converged);
    sh.eq.freeze_mu = freeze;
    (
        series::<Double, _>(&sh, &parts(&z), order, 1, true).unwrap(),
        sh.off_global(),
    )
}

/// [|z_1 - e_lambda|, max over k >= 2 of |z_k|], NaN where any entry is
fn cylinder_errors(br: &Branch<Double>, og: usize) -> [f64; 2] {
    let e1 = br.z[1]
        .iter()
        .enumerate()
        .map(|(i, &v)| (v - if i == og { DD::ONE } else { DD::ZERO }).hi.abs())
        .fold(0.0, nan_max);
    let high = (2..=br.order())
        .flat_map(|k| br.z[k].iter().map(|v| v.hi.abs()))
        .fold(0.0, nan_max);
    [e1, high]
}

/// the accuracy checks of Cylinder's branch: [z_1 within 1e-28, the higher orders within 1e-26]
fn cylinder_checks(e: [f64; 2]) -> [bool; 2] {
    [e[0] <= 1e-28, e[1] <= 1e-26]
}

/// Pass, written before the first run: through the multiple-shooting problem, the series of order 10 (jets of 8 and
/// 16 coefficients) gives z_1 = e_lambda, one in the eigenvalue and zero elsewhere, to 1e-28, and every higher order
/// below 1e-26. The controls: with mu frozen in the residual's series, z_1 misses e_lambda by all of it (its higher
/// orders are zero); z_2 = 1e-20 in one entry, and a NaN in z_4, fail the higher orders.
#[test]
fn anm_follows_the_eigenvalue_through_shooting() {
    let (br, og) = cylinder(false, 10);
    let e = cylinder_errors(&br, og);
    let res = br.residual.iter().copied().fold(0.0, nan_max);
    println!(
        "cylinder: |z_1 - e_lambda| {:.3e}, z_2..z_10 {:.3e}, solve residuals up to {res:.3e}",
        e[0], e[1]
    );
    assert_eq!(cylinder_checks(e), [true; 2]);
    let control = |name: &str, br: &Branch<Double>, want: [bool; 2]| {
        let e = cylinder_errors(br, og);
        println!(
            "control, {name}: |z_1 - e_lambda| {:.3e}, higher orders {:.3e}",
            e[0], e[1]
        );
        assert_eq!(cylinder_checks(e), want, "control {name} did not fire");
    };
    control("mu frozen", &cylinder(true, 3).0, [false, true]);
    let mut bad = br.clone();
    bad.z[2][0] = DD::from_f64(1e-20);
    control("z_2 = 1e-20", &bad, [true, false]);
    let mut bad = br;
    bad.z[4][0] = DD::from_f64(f64::NAN);
    control("NaN in z_4", &bad, [true, false]);
}
