//! The preconditioned Krylov solver `gcr` on a planted linear system, and the Newton method it refines
//! (`newton_refined`) on the planted multiple-shooting problem with the mixed step's f64 factors as preconditioner.
mod common;

use common::{perturbed, Planted};
use multishoot::gcr::{gcr, newton_refined, Coef, GcrOpts};
use multishoot::lane::Fu;
use multishoot::multidouble::linalg::ld_for;
use multishoot::newton::{newton_with, reuse};
use multishoot::prec::nan_max;
use multishoot::{Double, Native, NewtonOpts, Precision, Quad, Shooting, Storage, DD};

const N: usize = 20;

/// A x* = b with A = diag(4 + i / 10) plus off-diagonal entries of size 0.1, every entry a double-double whose lo part
/// is nonzero, x*_i = 1 + sin(i) / 3, and b = A x* formed in P. `perturb` scales entry (3, 5) of the A that `apply`
/// uses by 1 + perturb. Returns (A, b, x*), A by rows.
#[allow(clippy::type_complexity)]
fn system<P: Precision<Native>>() -> (Vec<Vec<P::Scalar>>, Vec<P::Scalar>, Vec<P::Scalar>) {
    let a: Vec<Vec<P::Scalar>> = (0..N)
        .map(|i| {
            (0..N)
                .map(|j| {
                    let v = if i == j {
                        4.0 + i as f64 / 10.0
                    } else {
                        0.1 * ((7 * i + 3 * j) as f64).sin()
                    };
                    P::from_dd(DD::new(v, v * 1e-17 * (1.0 + (i * N + j) as f64 / 400.0)))
                })
                .collect()
        })
        .collect();
    let xs: Vec<P::Scalar> = (0..N)
        .map(|i| P::from_dd(DD::from_f64(1.0 + (i as f64).sin() / 3.0)))
        .collect();
    let b = a
        .iter()
        .map(|row| P::dot(row.iter().copied().zip(xs.iter().copied())))
        .collect();
    (a, b, xs)
}

/// gcr on the planted system in P, preconditioned by A's f64 rounding (its lo parts dropped) factored in P, with the
/// products by A with entry (3, 5) scaled by 1 + perturb: (relative error of x, products, converged)
fn solve<P: Precision<Native>>(coef: Coef, perturb: f64, step_tol: f64) -> (f64, usize, bool) {
    let (a, b, xs) = system::<P>();
    let ld = ld_for(N);
    let mut c = vec![vec![0.0; ld * N]; P::PARTS];
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            c[0][j * ld + i] = P::part(v, 0);
        }
    }
    let m = P::lu_factor(N, ld, c, false).unwrap();
    let mut ap = a.clone();
    ap[3][5] = ap[3][5] * P::from_dd(DD::new(1.0, perturb));
    let apply = |u: &[P::Scalar]| -> Result<Vec<P::Scalar>, String> {
        Ok(ap
            .iter()
            .map(|row| P::dot(row.iter().copied().zip(u.iter().copied())))
            .collect())
    };
    let precond = |r: &[P::Scalar]| -> Result<Vec<P::Scalar>, String> {
        let mut x = r.to_vec();
        P::lu_solve(&m, &mut x);
        Ok(x)
    };
    let opts = GcrOpts {
        coef,
        step_tol,
        ..Default::default()
    };
    let (x, info) = gcr::<P>(apply, precond, &b, &opts).unwrap();
    let err = x
        .iter()
        .zip(&xs)
        .map(|(&a, &b)| P::to_dd(a - b).hi.abs())
        .fold(0.0, nan_max);
    let scale = xs.iter().map(|&v| P::to_dd(v).hi.abs()).fold(0.0, nan_max);
    (err / scale, info.products, info.converged)
}

/// Pass, written before the first run: preconditioned by A's f64 rounding, gcr with its coefficients in P reaches
/// x* to 1e-29 in double-double within 6 products (step_tol 1e-28) and to 1e-60 in quad-double within 10 (step_tol
/// 1e-62); with f64 coefficients it still gives x* to 1e-12, a Newton step's worth.
#[test]
fn gcr_reaches_the_planted_solution() {
    let (e, p, ok) = solve::<Double>(Coef::Full, 0.0, 1e-28);
    println!("double-double, coefficients in P: error {e:.3e}, {p} products, converged {ok}");
    assert!(ok && e <= 1e-29 && p <= 6);
    let (e, p, ok) = solve::<Quad>(Coef::Full, 0.0, 1e-62);
    println!("quad-double, coefficients in P: error {e:.3e}, {p} products, converged {ok}");
    assert!(ok && e <= 1e-60 && p <= 10);
    let (e, p, _) = solve::<Double>(Coef::F64, 0.0, 1e-28);
    println!("double-double, f64 coefficients: error {e:.3e}, {p} products");
    assert!(e <= 1e-12);
}

/// The control: products by an A with one entry off by 1e-20 relative make gcr solve another system, whose
/// solution misses x* by far more than 1e-29.
#[test]
fn gcr_control_inexact_products_miss() {
    let (e, p, _) = solve::<Double>(Coef::Full, 1e-20, 1e-28);
    println!("control, entry (3, 5) off by 1e-20: error {e:.3e} after {p} products");
    assert!(e > 1e-25, "control did not fire");
}

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

fn opts(tol: f64, max_iter: usize) -> NewtonOpts {
    NewtonOpts {
        tol,
        max_iter,
        ..Default::default()
    }
}

/// Pass: on the planted multiple-shooting problem from z* perturbed by 1e-3, with the f64 factors of `Fu` lanes
/// formed at the first step and reused for every step (`reuse`), Newton converges only linearly, while the same
/// factors as the preconditioner of gcr on the exact products give steps that converge to max |F| <= 1e-28 in at
/// most one step more than the exact Newton, and in fewer steps than the reused factors alone. (The first version of
/// this test compared against the mixed Newton with fresh factors at every step; on this problem the f64 Jacobian
/// contracts by about 1e-16 per step, so the mixed Newton already takes the exact Newton's steps and no comparison
/// with it can show the refinement.)
#[test]
fn refined_newton_converges_like_exact_newton() {
    let (sh, zs): (Shooting<Planted<B>, B>, _) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let start = perturbed(&zs, 1e-3);
    let mut z = start.clone();
    let exact = sh.newton::<Double>(&mut z, &opts(1e-28, 10)).unwrap();
    let f64_lu = |z: &[&[f64]]| sh.mixed_lu::<Fu>(z, Storage::Dense, true);
    let mut z = start.clone();
    let chord = newton_with::<Double, _, _>(&sh, &mut z, &opts(1e-28, 30), reuse(usize::MAX, f64_lu)).unwrap();
    let mut z = start.clone();
    let g = GcrOpts {
        step_tol: 1e-10,
        ..Default::default()
    };
    let (refined, products) =
        newton_refined::<Double, _, _>(&sh, &mut z, &opts(1e-28, 10), &g, reuse(usize::MAX, f64_lu)).unwrap();
    let steps = |r: &multishoot::NewtonReport| r.scale.len();
    println!(
        "steps: exact {}, reused f64 factors {} (converged {}), refined on them {} ({products} products); refined max \
         |F| {:.3e}",
        steps(&exact),
        steps(&chord),
        chord.converged,
        steps(&refined),
        refined.fnorm.last().unwrap()
    );
    assert!(exact.converged && refined.converged);
    assert!(steps(&refined) <= steps(&exact) + 1 && (!chord.converged || steps(&refined) < steps(&chord)));
}
