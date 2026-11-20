//! `Shooting::split_segments` on the manufactured solution `Planted`: the split layout's residual at the returned z
//! against the old layout's, row for row, in every precision, outward and inward, with the extra row, row factors
//! and a projected field carried along.
mod common;
use common::{in_pool, perturbed, Planted};
use multishoot::multidouble::precision::Precision;
use multishoot::{Double, Native, Octo, Quad, Shooting};

/// Planted on nodes 0, 1, 2 (two segments) at order 16, steps planned to 1e-30 from its solution, then z moved off
/// it by 1e-3 so that every residual row is far from zero
fn layout(inward: bool) -> (Shooting<Planted<1>, 1>, Vec<Vec<f64>>) {
    let (mut sh, z) = Planted::<1>::shooting(&[0.0, 1.0, 2.0], 16, 1e-30);
    if inward {
        sh.match_node = 0;
        sh.plan_segments(
            &z[0],
            &z[1],
            &multishoot::Plan {
                kt: 16,
                tol: 1e-30,
                hmax: 0.5,
            },
        );
    }
    (sh, perturbed(&z, 1e-3))
}

/// z widened to n parts, the parts past two zero
fn widen(z: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut out = z.to_vec();
    out.resize(n, vec![0.0; z[0].len()]);
    out
}

fn refs(z: &[Vec<f64>]) -> Vec<&[f64]> {
    z.iter().map(|p| &p[..]).collect()
}

/// the new row of each old row, from node positions alone: an endpoint block's rows stay first or last, a segment's
/// go to the new segment that ends at its target node from its side (the node itself outward, one past it inward),
/// the extra rows past the new blocks. `first_piece` maps a segment to its first piece instead, the control.
fn row_map(old: &Shooting<Planted<1>, 1>, new: &Shooting<Planted<1>, 1>, first_piece: bool) -> Vec<usize> {
    let nfm = 2 * 8;
    let at = |x: f64| {
        new.nodes
            .iter()
            .position(|&y| y == x)
            .expect("an old node is a new node")
    };
    let (k, k2) = (old.k(), new.k());
    let mut map = Vec::with_capacity(old.rows());
    for b in 0..=k {
        let nb = if b == 0 {
            0
        } else if b == k {
            k2
        } else if old.seg_inward(b) {
            if first_piece {
                at(old.nodes[b])
            } else {
                at(old.nodes[b - 1]) + 1
            }
        } else if first_piece {
            at(old.nodes[b - 1]) + 1
        } else {
            at(old.nodes[b])
        };
        map.extend((0..nfm).map(|r| nb * nfm + r));
    }
    map.extend((old.row_extra()..old.rows()).map(|r| r + new.row_extra() - old.row_extra()));
    map
}

/// rows of the split layout's residual at zn that break the rule: an old row's bits differ at its new place, or a row
/// no old row maps to is not +0 in every part
fn broken<P: Precision<Native>>(
    old: &Shooting<Planted<1>, 1>,
    z: &[Vec<f64>],
    new: &Shooting<Planted<1>, 1>,
    zn: &[Vec<f64>],
    map: &[usize],
) -> usize {
    let f0 = old.residual::<P>(&refs(z), false).unwrap();
    let f1 = new.residual::<P>(&refs(zn), false).unwrap();
    let bits = |x: P::Scalar| -> Vec<u64> { (0..P::PARTS).map(|q| P::part(x, q).to_bits()).collect() };
    let mut mapped = vec![false; f1.len()];
    let mut bad = 0;
    for (r, &n) in map.iter().enumerate() {
        mapped[n] = true;
        bad += usize::from(bits(f0[r]) != bits(f1[n]));
    }
    bad + (0..f1.len())
        .filter(|&n| !mapped[n] && bits(f1[n]).iter().any(|&b| b != 0))
        .count()
}

/// split at `max_steps` in P: (old layout, its z, new layout, new z)
#[allow(clippy::type_complexity)]
fn split<P: Precision<Native>>(
    inward: bool,
    max_steps: usize,
    extras: bool,
    par: bool,
) -> (
    Shooting<Planted<1>, 1>,
    Vec<Vec<f64>>,
    Shooting<Planted<1>, 1>,
    Vec<Vec<f64>>,
) {
    let (mut old, z) = layout(inward);
    if extras {
        // powers of two per row, and field 0 of the right block projected with the left data pinned
        old.row_scale = (0..old.rows()).map(|r| [1.0, 2.0, 0.25, 8.0][r % 4]).collect();
        old.project_rows(old.k(), 0, 0, multishoot::periodic::Parity::Odd, 3)
            .unwrap();
    }
    let z = widen(&z, P::PARTS);
    let (mut new, _) = layout(inward);
    new.row_scale = old.row_scale.clone();
    new.projected = old.projected.clone();
    let zn = new.split_segments::<P>(&refs(&z), max_steps, par).unwrap();
    (old, z, new, zn)
}

/// Pass: with segments of about 60 steps split at 20 (4 pieces each), in Double, Quad and Octo, outward and inward,
/// with and without row factors and a projection, every old residual row has its bits at its new place and every
/// other row is +0 (0 rows broken); every new segment has at most 20 steps, its points are old points, and the new
/// layout has more nodes.
#[test]
fn split_keeps_the_residual_bitwise() {
    fn one<P: Precision<Native>>(inward: bool, extras: bool) {
        let (old, z, new, zn) = split::<P>(inward, 20, extras, false);
        assert!(new.k() > old.k(), "no segment was split");
        let pts: Vec<f64> = old.segs.iter().flatten().copied().collect();
        for s in &new.segs {
            assert!(s.len() - 1 <= 20, "a piece of {} steps", s.len() - 1);
            assert!(
                s.iter().all(|x| pts.contains(x)),
                "a piece's point is not an old step point"
            );
        }
        let map = row_map(&old, &new, false);
        let bad = broken::<P>(&old, &z, &new, &zn, &map);
        assert_eq!(
            bad,
            0,
            "{} parts, inward {inward}, extras {extras}: {bad} rows broken",
            P::PARTS
        );
    }
    for inward in [false, true] {
        for extras in [false, true] {
            one::<Double>(inward, extras);
            one::<Quad>(inward, extras);
            one::<Octo>(inward, extras);
        }
    }
}

/// The controls of `split_keeps_the_residual_bitwise`: one ulp more in the leading part of the first interior
/// node's first value, and the segments' rows looked for at their first piece, must each break rows.
#[test]
fn split_control_ulp_and_wrong_piece_break_rows() {
    let (old, z, new, mut zn) = split::<Double>(false, 20, false, false);
    let o = new.off_node(1);
    zn[0][o] = zn[0][o].next_up();
    let ulp = broken::<Double>(&old, &z, &new, &zn, &row_map(&old, &new, false));
    let (old, z, new, zn) = split::<Double>(false, 20, false, false);
    let wrong = broken::<Double>(&old, &z, &new, &zn, &row_map(&old, &new, true));
    assert!(ulp > 0 && wrong > 0, "the controls broke {ulp} and {wrong} rows");
}

/// Pass: the split's z and layout are the same with rayon tasks (a pool of 4) as without, every part's bits, in Octo.
#[test]
fn split_is_the_same_on_tasks() {
    let (_, _, a, za) = split::<Octo>(false, 20, true, false);
    let (_, _, b, zb) = in_pool(4, || split::<Octo>(false, 20, true, true));
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.segs, b.segs);
    let bits = |z: &[Vec<f64>]| -> Vec<u64> { z.iter().flatten().map(|x| x.to_bits()).collect() };
    assert_eq!(bits(&za), bits(&zb));
}

/// the largest |difference| (hi plus lo parts) between the old layout's z0 and the new layout's z1 at the old nodes
fn root_gap(old: &Shooting<Planted<1>, 1>, z0: &[Vec<f64>], new: &Shooting<Planted<1>, 1>, z1: &[Vec<f64>]) -> f64 {
    let mut worst = 0.0f64;
    for (i, &x) in old.nodes.iter().enumerate() {
        let j = new.nodes.iter().position(|&y| y == x).unwrap();
        for r in 0..16 {
            let (a, b) = (old.off_node(i) + r, new.off_node(j) + r);
            let d = (z0[0][a] - z1[0][b]) + (z0[1][a] - z1[1][b]);
            worst = worst.max(d.abs());
        }
    }
    worst
}

/// Pass: Newton's method in Double from the perturbed z converges on both layouts, and the roots' values at the old
/// nodes agree within 1e-28 (the same discrete problem, solved with different unknowns). The control: the new root
/// with one old node's value moved by 1e-20 must fail the same bound.
#[test]
fn split_has_the_same_root() {
    let (old, z, new, zn) = split::<Double>(false, 20, false, false);
    let (mut z0, mut z1) = (z.clone(), zn.clone());
    assert!(old.newton::<Double>(&mut z0, &Default::default()).unwrap().converged);
    assert!(new.newton::<Double>(&mut z1, &Default::default()).unwrap().converged);
    let gap = root_gap(&old, &z0, &new, &z1);
    assert!(gap <= 1e-28, "roots differ by {gap:e} at the old nodes");
    let j = new.nodes.iter().position(|&y| y == old.nodes[1]).unwrap();
    z1[1][new.off_node(j) + 3] += 1e-20;
    let moved = root_gap(&old, &z0, &new, &z1);
    assert!(moved > 1e-28, "the control's gap {moved:e} passed the bound");
}

/// Pass: a projection whose pinned unknowns cross from node 0 into node 1, which the split moves apart, is an error,
/// and the layout keeps its nodes, segments and projection. The control: the same projection pinned to node 0's first
/// field, inside one node, splits.
#[test]
fn split_refuses_a_pin_across_moved_nodes() {
    let (mut sh, z) = layout(false);
    let zr = refs(&z);
    let across = sh.off_node(1) - 4;
    sh.project_rows(sh.k(), 0, across, multishoot::periodic::Parity::Odd, 3)
        .unwrap();
    let (nodes, segs) = (sh.nodes.clone(), sh.segs.clone());
    let e = sh.split_segments::<Double>(&zr, 20, false).unwrap_err();
    assert!(e.contains("pinned unknowns"), "{e}");
    assert_eq!((&sh.nodes, &sh.segs, sh.projected[0].pin), (&nodes, &segs, across));
    let (mut inside, _) = layout(false);
    let pin = inside.off_node(0);
    inside
        .project_rows(inside.k(), 0, pin, multishoot::periodic::Parity::Odd, 3)
        .unwrap();
    assert!(inside.split_segments::<Double>(&zr, 20, false).is_ok());
}

/// Pass: `max_steps` 0 is an error, and one at least the longest segment leaves the layout as it was.
#[test]
fn split_bounds() {
    let (mut sh, z) = layout(false);
    let zr = refs(&z);
    assert!(sh.split_segments::<Double>(&zr, 0, false).is_err());
    let (nodes, segs) = (sh.nodes.clone(), sh.segs.clone());
    let zn = sh.split_segments::<Double>(&zr, 10_000, false).unwrap();
    assert_eq!((sh.nodes.clone(), sh.segs.clone()), (nodes, segs));
    assert_eq!(zn, z);
}

/// FNV-1a of the split's z (Octo, outward, with row factors and a projection), its nodes and its points; with `ulp`,
/// the leading part of the first interior node's first value one ulp larger (the control)
fn split_hash(ulp: bool) -> u64 {
    let (_, _, new, mut zn) = split::<Octo>(false, 20, true, false);
    if ulp {
        let o = new.off_node(1);
        zn[0][o] = zn[0][o].next_up();
    }
    let mut h = common::Fnv::new();
    for x in new
        .nodes
        .iter()
        .chain(new.segs.iter().flatten())
        .chain(zn.iter().flatten())
    {
        h.f64(*x);
    }
    h.0
}

/// the reference hash of `split_hash`, recorded on aarch64 (NEON)
const SPLIT_REFERENCE: u64 = 0xc13acc9a1bef30ae;

/// Pass: `split_hash` equals the reference on every SIMD backend. The control: one ulp of an interior node changes it.
#[test]
fn split_does_not_depend_on_the_backend() {
    let h = split_hash(false);
    println!("split hash {h:#018x}");
    assert_eq!(h, SPLIT_REFERENCE);
    assert_ne!(split_hash(true), SPLIT_REFERENCE, "the control kept the reference");
}
