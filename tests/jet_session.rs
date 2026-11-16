//! The residual's series one order at a time (`Shooting::jet_session`) against `residual_jet`: on the eigenproblem
//! `Cylinder` (a parameter in its equations, two globals, two extra rows, planned segments of several steps) and on
//! the planted problem with projected rows, every order of the session is `residual_jet`'s, bit for bit. The session
//! calls the equations' order-k methods in the order their contract states, and keeps an endpoint map's series between
//! orders.
mod common;

use common::{wave, Cylinder, Planted};
use multishoot::periodic::Parity;
use multishoot::{Double, Equations, Fl, Native, Octo, Precision, Quad, Shooting, State, Work};
use multisolver::prec::{parts, widen};
use std::collections::HashMap;
use std::sync::Mutex;

/// the coefficient vectors z_1, z_2, ...: waves of size 0.1^k on every unknown, in double-double parts
fn coef_vector(l: usize, k: usize) -> Vec<Vec<f64>> {
    let s = 0.1f64.powi(k as i32);
    vec![(0..l).map(|i| s * wave(i + 31 * k)).collect(), vec![0.0; l]]
}

/// the number of rows whose parts differ in any bit; every row of the longer when the lengths differ
fn bits_differ<P: Precision<Native>>(a: &[P::Scalar], b: &[P::Scalar]) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len());
    }
    a.iter()
        .zip(b)
        .filter(|(&u, &v)| (0..P::PARTS).any(|q| P::part(u, q).to_bits() != P::part(v, q).to_bits()))
        .count()
}

/// Gives z_0..z_(n-1) to a session on `P::J<8>` lanes and, after each, compares the order it returns and every order
/// it has formed with `residual_jet` on the same vectors: (rows compared, rows that differ in the returned orders,
/// rows that differ in the formed orders)
fn compare<E: Equations<B>, P: Precision<Native>, const B: usize>(
    sh: &Shooting<E, B>,
    z0: &[Vec<f64>],
    n: usize,
    par: bool,
) -> (usize, usize, usize) {
    let l = sh.len();
    let zs: Vec<Vec<Vec<f64>>> = (0..n)
        .map(|k| widen(&if k == 0 { z0.to_vec() } else { coef_vector(l, k) }, P::PARTS))
        .collect();
    let mut s = sh.jet_session::<P, 8>();
    let (mut rows, mut ret, mut formed) = (0, 0, 0);
    for k in 0..n {
        let next = s.next(&parts(&zs[k]), par).unwrap();
        let refs: Vec<Vec<&[f64]>> = zs[..=k].iter().map(|z| parts(z)).collect();
        let full = sh.residual_jet::<P, 8>(&refs, par, k + 2).unwrap();
        ret += bits_differ::<P>(&next, &full[k + 1]);
        for (j, fj) in full.iter().enumerate() {
            formed += bits_differ::<P>(&s.order(j).unwrap(), fj);
            rows += fj.len();
        }
    }
    (rows, ret, formed)
}

fn cylinder() -> (Shooting<Cylinder<1>, 1>, Vec<Vec<f64>>) {
    Cylinder::<1>::shooting(Cylinder::new(), 30)
}

fn projected() -> (Shooting<Planted<2>, 2>, Vec<Vec<f64>>) {
    let (mut sh, z) = Planted::<2>::shooting(&[0.0, 0.5, 1.0, 1.5, 2.0], 30, 1e-32);
    let k = sh.k();
    sh.project_rows(k, 1, 0, Parity::Even, 2).unwrap();
    (sh, z)
}

/// Pass, written before the first run: on `Cylinder` in double-, quad- and octo-double and on the projected planted
/// problem in double-double, with and without threads, the six orders a session returns from z_0..z_5 and every order
/// it has formed after each vector are `residual_jet`'s on the same vectors, bit for bit (0 rows differ).
#[test]
fn session_orders_are_residual_jet_bitwise() {
    let (cy, zc) = cylinder();
    let (pr, zp) = projected();
    assert!(
        cy.segs.iter().any(|s| s.len() > 2),
        "the cylinder's segments should have several steps"
    );
    for par in [false, true] {
        let runs = [
            ("cylinder, double-double", compare::<_, Double, 1>(&cy, &zc, 6, par)),
            ("cylinder, quad-double", compare::<_, Quad, 1>(&cy, &zc, 6, par)),
            ("cylinder, octo-double", compare::<_, Octo, 1>(&cy, &zc, 6, par)),
            (
                "projected planted, double-double",
                compare::<_, Double, 2>(&pr, &zp, 6, par),
            ),
        ];
        for (name, (rows, ret, formed)) in runs {
            println!(
                "{name}, par {par}: {rows} rows compared, {ret} differ in the returned orders, {formed} in the formed"
            );
            assert_eq!(
                (ret, formed),
                (0, 0),
                "{name}, par {par}: the session is not residual_jet's"
            );
        }
    }
}

/// `Cylinder` with its order-k calls recorded: (method, step start or block, order), and each endpoint map's kept
/// series grown by one entry per call
struct Recorder {
    inner: Cylinder<1>,
    calls: Mutex<Vec<(char, u64, usize)>>,
}

impl Equations<1> for Recorder {
    const NF: usize = 2;
    const NL: usize = 1;
    const NR: usize = 1;
    const NG: usize = 2;
    type Prep = ();

    fn taylor_coeffs<T: multishoot::lane::Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, 1>) {
        self.inner.taylor_coeffs(x0, g, w)
    }
    fn left_state<T: multishoot::lane::Lane>(&self, x: f64, d: &[Fl<T, 1>], g: &[T], p: &()) -> State<T, 1> {
        self.inner.left_state(x, d, g, p)
    }
    fn right_state<T: multishoot::lane::Lane>(&self, x: f64, d: &[Fl<T, 1>], g: &[T], p: &()) -> State<T, 1> {
        self.inner.right_state(x, d, g, p)
    }
    fn taylor_coeffs_at<T: multishoot::lane::Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, 1>, k: usize) {
        self.calls.lock().unwrap().push(('T', x0.to_bits(), k));
        self.inner.taylor_coeffs_at(x0, g, w, k)
    }
    fn left_state_at<T: multishoot::lane::Lane>(
        &self,
        x: f64,
        d: &[Fl<T, 1>],
        g: &[T],
        p: &(),
        keep: &mut Vec<Vec<Fl<T, 1>>>,
        k: usize,
    ) -> State<T, 1> {
        self.calls.lock().unwrap().push(('L', keep.len() as u64, k));
        keep.push(Vec::new());
        self.inner.left_state_at(x, d, g, p, keep, k)
    }
    fn right_state_at<T: multishoot::lane::Lane>(
        &self,
        x: f64,
        d: &[Fl<T, 1>],
        g: &[T],
        p: &(),
        keep: &mut Vec<Vec<Fl<T, 1>>>,
        k: usize,
    ) -> State<T, 1> {
        self.calls.lock().unwrap().push(('R', keep.len() as u64, k));
        keep.push(Vec::new());
        self.inner.right_state_at(x, d, g, p, keep, k)
    }
}

/// The failures of a recorded call list against the contract: every one of `steps` Taylor steps and both endpoint
/// maps called for the orders `want` in that sequence, and each endpoint map's kept series holding one entry per call
/// before it
fn call_failures(calls: &[(char, u64, usize)], steps: usize, want: &[usize]) -> Vec<String> {
    let mut by: HashMap<(char, u64), Vec<usize>> = HashMap::new();
    let mut keeps: HashMap<char, Vec<u64>> = HashMap::new();
    for &(m, key, k) in calls {
        if m == 'T' {
            by.entry((m, key)).or_default().push(k);
        } else {
            by.entry((m, 0)).or_default().push(k);
            keeps.entry(m).or_default().push(key);
        }
    }
    let mut fails = Vec::new();
    let taylor = by.keys().filter(|(m, _)| *m == 'T').count();
    if taylor != steps {
        fails.push(format!("{taylor} of {steps} Taylor steps called"));
    }
    for m in ['L', 'R'] {
        if !by.contains_key(&(m, 0)) {
            fails.push(format!("no {m} endpoint calls"));
        }
    }
    for (key, seq) in &by {
        if seq != want {
            fails.push(format!("{key:?}: orders {seq:?}"));
        }
    }
    let lens: Vec<u64> = (0..want.len() as u64).collect();
    for (m, l) in &keeps {
        if l != &lens {
            fails.push(format!("{m}: kept series lengths {l:?}"));
        }
    }
    fails
}

/// Pass, written before the first run: after z_0..z_3, every Taylor step and both endpoint maps were called for the
/// orders 0, 1, 1, 2, 2, 3, 3, 4 in that sequence (order k formed with z_k zero, then again once it is given), each
/// endpoint map saw its kept series hold one entry per call before it, and the session's orders are still
/// `residual_jet`'s, bit for bit. The controls: the same calls without the right endpoint's, and without one Taylor
/// step's, fail the contract.
#[test]
fn session_calls_the_order_methods_in_sequence() {
    let (cy, z) = cylinder();
    let mut sh = Shooting::new(
        Recorder {
            inner: Cylinder::new(),
            calls: Mutex::new(Vec::new()),
        },
        cy.nodes.clone(),
        cy.kt,
    );
    sh.segs = cy.segs.clone();
    sh.extra = cy.extra.clone();
    let (_, ret, formed) = compare::<_, Double, 1>(&sh, &z, 4, true);
    assert_eq!((ret, formed), (0, 0), "the recorded session is not residual_jet's");
    let calls = sh.eq.calls.lock().unwrap().clone();
    let want = [0, 1, 1, 2, 2, 3, 3, 4];
    let steps: usize = cy.segs.iter().map(|s| s.len() - 1).sum();
    let fails = call_failures(&calls, steps, &want);
    println!(
        "{} calls on {steps} Taylor steps and two endpoint maps: {fails:?}",
        calls.len()
    );
    assert!(fails.is_empty(), "{fails:?}");
    let no_right: Vec<_> = calls.iter().copied().filter(|c| c.0 != 'R').collect();
    let first_step = calls.iter().find(|c| c.0 == 'T').unwrap().1;
    let no_step: Vec<_> = calls
        .iter()
        .copied()
        .filter(|c| !(c.0 == 'T' && c.1 == first_step))
        .collect();
    for (name, cut) in [("no right endpoint", no_right), ("one Taylor step missing", no_step)] {
        let f = call_failures(&cut, steps, &want);
        println!("control, {name}: {f:?}");
        assert!(!f.is_empty(), "control {name} did not fire");
    }
}

/// Pass, written before the first run: the row comparison counts every row of a residual that is missing rows, so a
/// session order of the wrong length cannot pass as bitwise equal.
#[test]
fn bits_differ_counts_missing_rows() {
    let one = <Double as Precision<Native>>::from_dd(multishoot::DD::ONE);
    assert_eq!(bits_differ::<Double>(&[], &[one, one]), 2);
    assert_eq!(bits_differ::<Double>(&[one, one], &[one]), 2);
    assert_eq!(bits_differ::<Double>(&[one], &[one]), 0);
}

/// The control of `session_orders_are_residual_jet_bitwise` (feature `mutate`): without the second forming of order
/// k once z_k is given (`Control::JetStale`), the formed orders differ from `residual_jet`'s.
#[cfg(feature = "mutate")]
#[test]
fn session_control_stale_orders_differ() {
    let (mut cy, z) = cylinder();
    cy.set_control(multishoot::control::Control::JetStale);
    let (rows, ret, formed) = compare::<_, Double, 1>(&cy, &z, 6, true);
    println!("control, stale orders: {formed} of {rows} rows differ in the formed orders, {ret} in the returned");
    assert!(formed > 0, "control did not fire");
}
