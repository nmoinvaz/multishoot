//! The Newton method that multisolver's Krylov solver refines (`newton_refined`) on the planted multiple-shooting
//! problem, with the mixed step's f64 factors as preconditioner. The solver's model-problem tests are multisolver's.
mod common;

use common::{perturbed, Planted};
use multishoot::lane::Fu;
use multishoot::{Double, Shooting, Storage};
use multisolver::gcr::{newton_refined, GcrOpts};
use multisolver::newton::{newton_with, reuse};
use multisolver::NewtonOpts;

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
    let steps = |r: &multisolver::NewtonReport| r.scale.len();
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
