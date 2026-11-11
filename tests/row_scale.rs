//! Scaled rows: the residual, the Jacobian and the jet with `row_scale` are the unscaled ones times the factors.
mod common;

use common::{perturbed, Planted};
use multishoot::prec::parts;
use multishoot::{Double, Shooting, DD};

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

/// the planted problem with block 3's rows scaled by s (power of two), and z* perturbed by 1e-3
fn scaled(s: f64) -> (Shooting<Planted<B>, B>, Shooting<Planted<B>, B>, Vec<Vec<f64>>) {
    let (plain, zs) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let (mut sc, _) = Planted::<B>::shooting(&NODES, 30, 1e-32);
    let (r0, r1) = (sc.row_block(3), sc.row_block(4));
    sc.row_scale = (0..sc.rows())
        .map(|r| if (r0..r1).contains(&r) { s } else { 1.0 })
        .collect();
    (plain, sc, perturbed(&zs, 1e-3))
}

/// Pass: with block 3 scaled by 2^100 the residual, every Jacobian entry and both orders of the jet equal the
/// unscaled ones times the factor, bit for bit (a power of two scales exactly), and the other rows are unchanged.
#[test]
fn row_scale_scales_residual_jacobian_and_jet() {
    let s = 2f64.powi(100);
    let (plain, sc, z) = scaled(s);
    let p = parts(&z);
    let f = |r: usize| {
        if sc.row_scale[r] == 1.0 {
            DD::ONE
        } else {
            DD::from_f64(s)
        }
    };
    let (f0, f1) = (
        plain.residual::<Double>(&p, true).unwrap(),
        sc.residual::<Double>(&p, true).unwrap(),
    );
    assert!((0..f0.len()).all(|r| f0[r] * f(r) == f1[r]));
    let ((c0, ld), (c1, _)) = (
        plain.jacobian::<Double>(&p, true).unwrap(),
        sc.jacobian::<Double>(&p, true).unwrap(),
    );
    let l = plain.len();
    for j in 0..l {
        for r in 0..l {
            let (a, b) = (
                DD::new(c0[0][j * ld + r], c0[1][j * ld + r]),
                DD::new(c1[0][j * ld + r], c1[1][j * ld + r]),
            );
            assert_eq!(a * f(r), b, "entry ({r}, {j})");
        }
    }
    let v = perturbed(&z, 1e-1);
    let (j0, j1) = (
        plain
            .residual_jet::<Double, 2>(&[p.clone(), parts(&v)], true, 2)
            .unwrap(),
        sc.residual_jet::<Double, 2>(&[p.clone(), parts(&v)], true, 2).unwrap(),
    );
    assert!((0..2).all(|k| (0..l).all(|r| j0[k][r] * f(r) == j1[k][r])));
}

/// The control: a factor list of the wrong length must be refused, not applied.
#[test]
fn row_scale_control_wrong_length_fails() {
    let (_, mut sc, z) = scaled(2.0);
    sc.row_scale.pop();
    assert!(sc.residual::<Double>(&parts(&z), true).is_err() && sc.jacobian::<Double>(&parts(&z), true).is_err());
}
