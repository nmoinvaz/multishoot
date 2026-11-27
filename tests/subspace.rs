//! The weak subspace and the factors (multisolver's) on the planted multiple-shooting problem, bit for bit in every
//! thread count. Their model-problem tests are multisolver's.
mod common;

use common::{in_pool, perturbed, Fnv, Planted};
use multishoot::{Double, DD};
use multisolver::prec::parts;
use multisolver::subspace::{weak_subspace, Factors};

/// The hash of the planted shooting problem's factors at z* perturbed by 1e-2 (with `par`), through the weak
/// subspace of width 3 after 3 products, sensitivity 5 and a solve with each factor; `seed` is the subspace's start
fn factors_hash(par: bool, seed: u64) -> u64 {
    let (sh, zs) = Planted::<1>::shooting(&[0.0, 0.5, 1.0, 1.5, 2.0], 30, 1e-32);
    let z = perturbed(&zs, 1e-2);
    let f = Factors::<Double>::new(&sh, &parts(&z), par).unwrap();
    let (basis, sg) = weak_subspace::<Double>(&f, 3, 3, seed);
    let b: Vec<DD> = (0..f.n).map(|i| DD::from_f64(common::wave(i))).collect();
    let mut h = Fnv::new();
    basis.iter().flatten().chain(&sg).for_each(|&x| h.f64(x));
    for v in [f.sensitivity(5), f.solve(&b), f.solve_t(&b)] {
        v.iter().for_each(|x| {
            h.f64(x.hi);
            h.f64(x.lo);
        });
    }
    h.0
}

/// the reference hash of `factors_hash`, recorded with the factorizations and the subspace's products in sequence on
/// aarch64 (NEON)
const FACTORS_REFERENCE: u64 = 0x29d7bfb26e1a6575;

/// The factors and the weak subspace do not depend on the threads: pass, `factors_hash` with `par` in pools of 1, 3
/// and 6 threads, and without it, equals the reference of the serial code. The control: another start of the subspace
/// changes it.
#[test]
fn factors_and_subspace_do_not_depend_on_threads() {
    for (t, par) in [(1, true), (3, true), (6, true), (3, false)] {
        let h = in_pool(t, || factors_hash(par, 7));
        println!("factors hash, {t} threads, par {par}: {h:#018x}");
        assert_eq!(h, FACTORS_REFERENCE, "{t} threads, par {par}");
    }
    assert_ne!(
        in_pool(3, || factors_hash(true, 8)),
        FACTORS_REFERENCE,
        "control did not fire"
    );
}
