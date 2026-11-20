//! The choice of the Taylor order (`Shooting::plan_segments_auto`) on the model problems `Rotor` and `Planted`, whose
//! solutions are known in closed form: the order kept is the cost minimum among the candidates, the steps planned at
//! it hold the planner's bound, and its bits are the same on every backend and in every thread pool.
mod common;

use common::{in_pool, max_diff, planted_z, Fnv, Planted, Rotor};
use multishoot::lane::DV;
use multishoot::taylor::{order_guess, step_cost, taylor_step, OrderChoice, Plan, STEP_COST_W};
use multishoot::{Equations, ExtraRow, Shooting, State, Work, DD};
use multisolver::prec::nan_max;

const B: usize = 2;
const NODES: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

/// A model problem with its layout and the z of its exact solution
trait Model: Equations<B> + Sized {
    /// the problem on NODES, one step per segment, at a placeholder order
    fn shooting() -> Shooting<Self, B>;
    /// z of the exact solution (hi, lo)
    fn z(sh: &Shooting<Self, B>) -> Vec<Vec<f64>>;
    /// the exact state at x
    fn exact_at(&self, x: f64) -> State<DV, B>;
    /// the globals of the exact solution, in planner lanes
    fn globals(&self) -> Vec<DV>;
}

impl Model for Rotor<B> {
    fn shooting() -> Shooting<Self, B> {
        Shooting::new(Rotor::<B>::new(), NODES.to_vec(), 2)
    }
    /// the exact node states (the endpoint data and globals are not read by the planner)
    fn z(sh: &Shooting<Self, B>) -> Vec<Vec<f64>> {
        let m = 8 * B;
        let (mut zh, mut zl) = (vec![0.0; sh.len()], vec![0.0; sh.len()]);
        for (i, &x) in sh.nodes.iter().enumerate() {
            let o = sh.off_node(i);
            for (f, fl) in sh.eq.exact(x).iter().enumerate() {
                fl.store(&mut zh[o + f * m..o + (f + 1) * m], &mut zl[o + f * m..o + (f + 1) * m]);
            }
        }
        vec![zh, zl]
    }
    fn exact_at(&self, x: f64) -> State<DV, B> {
        self.exact(x)
    }
    fn globals(&self) -> Vec<DV> {
        Vec::new()
    }
}

impl Model for Planted<B> {
    fn shooting() -> Shooting<Self, B> {
        let mut sh = Shooting::new(Planted::<B>::new(), NODES.to_vec(), 2);
        let og = sh.off_global();
        sh.extra.push(ExtraRow {
            terms: vec![(og, DD::ONE)],
            rhs: DD::from_f64(sh.eq.omega),
        });
        sh
    }
    fn z(sh: &Shooting<Self, B>) -> Vec<Vec<f64>> {
        planted_z(sh)
    }
    fn exact_at(&self, x: f64) -> State<DV, B> {
        self.exact(x)
    }
    fn globals(&self) -> Vec<DV> {
        vec![DV::splat(DD::from_f64(self.omega))]
    }
}

/// The order choice on the model's exact solution, with the problem it leaves planned
fn choose<M: Model>(
    tol: f64,
    hmax: f64,
    orders: &[usize],
    cost: impl Fn(usize) -> f64,
) -> (Shooting<M, B>, OrderChoice) {
    let mut sh = M::shooting();
    let z = M::z(&sh);
    let choice = sh.plan_segments_auto(&z[0], &z[1], tol, hmax, orders, cost);
    (sh, choice)
}

/// One candidate planned by `plan_segments` at its fixed order
struct Planned {
    kt: usize,
    /// the steps of each segment
    steps: Vec<usize>,
    segs: Vec<Vec<f64>>,
}

/// Every candidate planned by `plan_segments` at its fixed order, by increasing order
fn brute<M: Model>(tol: f64, hmax: f64, orders: &[usize]) -> Vec<Planned> {
    let mut ks = orders.to_vec();
    ks.sort_unstable();
    ks.dedup();
    ks.iter()
        .map(|&kt| {
            let mut sh = M::shooting();
            let z = M::z(&sh);
            sh.plan_segments(&z[0], &z[1], &Plan { kt, tol, hmax });
            let steps = sh.segs.iter().map(|s| s.len() - 1).collect();
            Planned {
                kt,
                steps,
                segs: sh.segs,
            }
        })
        .collect()
}

/// The check of a choice against the brute force: the candidates are the distinct orders asked for, increasing; each
/// candidate kept reports the brute force's steps per segment and the cost `sum(steps) cost(kt)` exactly, and each
/// dropped one costs more than the order chosen; the order chosen is the least cost of the brute force, the lowest
/// order on a tie; and the problem is left at that order with the brute force's points, bit for bit.
fn check_choice<M: Model>(
    sh: &Shooting<M, B>,
    choice: &OrderChoice,
    brute: &[Planned],
    cost: impl Fn(usize) -> f64,
) -> Result<(), String> {
    let costs: Vec<f64> = brute
        .iter()
        .map(|b| b.steps.iter().sum::<usize>() as f64 * cost(b.kt))
        .collect();
    let kts: Vec<usize> = choice.candidates.iter().map(|c| c.kt).collect();
    if kts != brute.iter().map(|b| b.kt).collect::<Vec<_>>() {
        return Err(format!("candidates {kts:?}"));
    }
    let (imin, &cmin) = costs
        .iter()
        .enumerate()
        .fold((0, &f64::INFINITY), |a, (i, c)| if *c < *a.1 { (i, c) } else { a });
    for ((c, b), &bc) in choice.candidates.iter().zip(brute).zip(&costs) {
        match (&c.steps, c.cost) {
            (Some(s), Some(e)) if *s == b.steps && e == bc => {}
            (None, None) if bc > cmin => {}
            _ => {
                return Err(format!(
                    "order {}: {:?} {:?} against {:?} {bc}",
                    c.kt, c.steps, c.cost, b.steps
                ))
            }
        }
    }
    let want = brute[imin].kt;
    if choice.kt != want || sh.kt != want || sh.segs != brute[imin].segs {
        return Err(format!(
            "chose {} (problem at {}), the least cost is order {want}",
            choice.kt, sh.kt
        ));
    }
    Ok(())
}

/// the cases of the brute-force check: tolerance, cap, and whether the cost is the default model (else every step
/// costs 1, so the orders that reach the cap tie)
const CASES: [(f64, f64, bool); 4] = [
    (1e-20, 0.5, true),
    (1e-32, 0.5, true),
    (1e-60, 0.5, true),
    (1e-20, 0.125, false),
];

fn unit_cost(_: usize) -> f64 {
    1.0
}

/// The kept candidate's cost
fn kept_cost(c: &OrderChoice) -> f64 {
    c.candidates.iter().find(|o| o.kt == c.kt).and_then(|o| o.cost).unwrap()
}

/// The check of the second pass's candidates: consecutive orders holding the kept one, strictly between its
/// neighbors among the first pass's candidates where it has them
fn check_refined(first: &OrderChoice, refined: &[usize]) -> Result<(), String> {
    let ks: Vec<usize> = first.candidates.iter().map(|c| c.kt).collect();
    let i = ks.iter().position(|&k| k == first.kt).unwrap();
    let consecutive = refined.windows(2).all(|p| p[1] == p[0] + 1) && refined.contains(&first.kt);
    let lo_ok = i == 0 || refined[0] == ks[i - 1] + 1;
    let hi_ok = i + 1 == ks.len() || *refined.last().unwrap() == ks[i + 1] - 1;
    if consecutive && lo_ok && hi_ok && refined[0] >= 2 {
        Ok(())
    } else {
        Err(format!("refined {refined:?} around {} of {ks:?}", first.kt))
    }
}

/// The choice and the check of every case on the model, with `Plan::candidate_orders` and then a second pass over
/// `OrderChoice::refined_orders`, and once with the candidates 40, 12, 20, 40 at 1e-32 (the low orders dropped early)
fn check_all<M: Model>(name: &str) -> Result<(), String> {
    let mut cases: Vec<(f64, f64, bool, Vec<usize>)> = CASES
        .iter()
        .map(|&(t, h, d)| (t, h, d, Plan::candidate_orders(t)))
        .collect();
    cases.push((1e-32, 0.5, true, vec![40, 12, 20, 40]));
    for (tol, hmax, default, orders) in cases {
        let cost: fn(usize) -> f64 = if default { step_cost } else { unit_cost };
        let mut first: Option<OrderChoice> = None;
        for pass in 1..=2 {
            let orders = match &first {
                None => orders.clone(),
                Some(c) => c.refined_orders(),
            };
            let (sh, choice) = choose::<M>(tol, hmax, &orders, cost);
            let b = brute::<M>(tol, hmax, &orders);
            let show: Vec<String> = choice
                .candidates
                .iter()
                .map(|c| match &c.steps {
                    Some(s) => format!("{}:{}", c.kt, s.iter().sum::<usize>()),
                    None => format!("{}:x", c.kt),
                })
                .collect();
            println!(
                "{name} tol {tol:.0e} hmax {hmax} pass {pass}: kt {} of [{}]",
                choice.kt,
                show.join(" ")
            );
            let at = |e: String| format!("{name} tol {tol:e} hmax {hmax} pass {pass}: {e}");
            check_choice(&sh, &choice, &b, cost).map_err(at)?;
            if let Some(f) = &first {
                check_refined(f, &orders).map_err(at)?;
                if kept_cost(&choice) > kept_cost(f) {
                    return Err(at(format!(
                        "cost {} above the first pass's {}",
                        kept_cost(&choice),
                        kept_cost(f)
                    )));
                }
            }
            first = Some(choice);
        }
    }
    Ok(())
}

/// Pass, written before the first run: on Rotor and Planted at tol 1e-20, 1e-32 and 1e-60 (cap 0.5, default cost,
/// `Plan::candidate_orders`), at 1e-20 with the cap 0.125 and a unit step cost (ties between the orders that reach the
/// cap), and at 1e-32 with the candidates 40, 12, 20, 40, `check_choice` holds: the order kept is the least cost of the
/// candidates planned one by one, the lowest on a tie, with their points bit for bit, and every candidate's report
/// is the brute force's. The second pass over `OrderChoice::refined_orders` passes the same check, its candidates
/// are the consecutive orders strictly between the first choice's neighbors (`check_refined`), and its least cost is
/// at most the first pass's.
#[test]
fn auto_order_is_the_least_cost() {
    check_all::<Rotor<B>>("rotor").unwrap();
    check_all::<Planted<B>>("planted").unwrap();
}

/// The control of `check_refined`: the second pass's candidates without their lowest order, or with one order past
/// the upper neighbor, must fail it (Rotor at 1e-32).
#[test]
fn refined_orders_control_wrong_range_fails() {
    let (_, first) = choose::<Rotor<B>>(1e-32, 0.5, &Plan::candidate_orders(1e-32), step_cost);
    let r = first.refined_orders();
    assert!(check_refined(&first, &r).is_ok());
    let mut longer = r.clone();
    longer.push(r.last().unwrap() + 1);
    assert!(check_refined(&first, &r[1..]).is_err(), "control did not fire");
    assert!(check_refined(&first, &longer).is_err(), "control did not fire");
}

/// One step of each segment planned at the chosen order from the exact state at its start, against the exact state at
/// its end, the step scaled by `stretch`: the largest error relative to tol, and the order.
fn bound_ratio<M: Model>(tol: f64, stretch: f64) -> (f64, usize) {
    let (sh, choice) = choose::<M>(tol, 0.5, &Plan::candidate_orders(tol), step_cost);
    let g = sh.eq.globals();
    let mut w = Work::<DV, B>::new(M::NF, M::NAUX, choice.kt);
    let mut worst = 0.0f64;
    for p in sh.segs.iter().flat_map(|s| s.windows(2)) {
        let x1 = p[0] + stretch * (p[1] - p[0]);
        let s = taylor_step(&sh.eq, &sh.eq.exact_at(p[0]), p[0], x1, &g, &mut w);
        worst = nan_max(worst, max_diff(&s, &sh.eq.exact_at(x1)) / tol);
    }
    (worst, choice.kt)
}

/// Pass, written before the first run: at tol 1e-20 (far above double-double rounding, so the error measured is the
/// truncation) and the order chosen from `Plan::candidate_orders`, every planned step of Rotor and of Planted from the
/// exact start has an error of at most tol (the planner's bound holds at the chosen order), and the largest is at
/// least 1e-3 tol (the bound is not loose by more than that). The control: the planned steps doubled break the bound
/// (some error above tol) on both.
#[test]
fn auto_order_holds_the_planner_bound() {
    for (name, (worst, kt), (doubled, _)) in [
        (
            "rotor",
            bound_ratio::<Rotor<B>>(1e-20, 1.0),
            bound_ratio::<Rotor<B>>(1e-20, 2.0),
        ),
        (
            "planted",
            bound_ratio::<Planted<B>>(1e-20, 1.0),
            bound_ratio::<Planted<B>>(1e-20, 2.0),
        ),
    ] {
        println!("{name}: order {kt}, largest error / tol {worst:.3e}, doubled steps {doubled:.3e}");
        assert!((1e-3..=1.0).contains(&worst), "{name}: largest error {worst:e} tol");
        assert!(doubled > 1.0, "{name}: control did not fire, {doubled:e} tol");
    }
}

/// The fingerprint of one model's two-pass choice at tol (cap 0.5, default cost, the candidates `orders` and then the
/// first pass's `refined_orders`): the order kept, every candidate's order, steps and cost, and the planned points
fn put_choice<M: Model>(h: &mut Fnv, tol: f64, orders: &[usize]) {
    let first = choose::<M>(tol, 0.5, orders, step_cost);
    let second = choose::<M>(tol, 0.5, &first.1.refined_orders(), step_cost);
    for (sh, c) in [first, second] {
        h.f64(c.kt as f64);
        for o in &c.candidates {
            h.f64(o.kt as f64);
            o.steps.iter().flatten().for_each(|&n| h.f64(n as f64));
            h.f64(o.cost.unwrap_or(-1.0));
        }
        sh.segs.iter().flatten().for_each(|&x| h.f64(x));
    }
}

/// The fingerprint of the order choice: `Plan::optimal_kt` at tol 1e-20, 1e-32 and 1e-60, and the two-pass choice of
/// Rotor and Planted at those tolerances times `tol_scale` (`Plan::candidate_orders` of the unscaled tol)
fn choice_hash(tol_scale: f64) -> u64 {
    let mut h = Fnv::new();
    for tol in [1e-20, 1e-32, 1e-60] {
        h.f64(Plan::optimal_kt(tol) as f64);
        let orders = Plan::candidate_orders(tol);
        put_choice::<Rotor<B>>(&mut h, tol * tol_scale, &orders);
        put_choice::<Planted<B>>(&mut h, tol * tol_scale, &orders);
    }
    h.0
}

/// the reference fingerprint of the order choice, recorded on aarch64 (NEON) and on the portable backend
const CHOICE_REFERENCE: u64 = 0x3696801289049ab8;

/// Pass, written before the first run: `choice_hash` equals the reference on every build of the CI matrix (SSE2,
/// AVX2, NEON, portable, MSVC), and in rayon pools of 1, 2, 3 and 7 threads. The control: the tolerance scaled by
/// 1 + 2^-30 changes the fingerprint, in a pool of 3.
#[test]
fn auto_order_bytes_match_reference() {
    let h = choice_hash(1.0);
    println!("choice hash {h:#018x}");
    assert_eq!(
        h, CHOICE_REFERENCE,
        "the order choice's bytes differ from the reference"
    );
    for t in [1, 2, 3, 7] {
        assert_eq!(in_pool(t, || choice_hash(1.0)), CHOICE_REFERENCE, "{t} threads");
    }
    let c = in_pool(3, || choice_hash(1.0 + 2f64.powi(-30)));
    assert_ne!(c, CHOICE_REFERENCE, "control did not fire");
}

/// The check of the a-priori order at tol and w against the root k* of `2k^2 + (w - L) k - L w = 0`, L = ln(1 / tol)
/// by libm (the stationary point of ln(k^2 + w k) + L / k): the guess is within one of max(2, ceil(k*)).
fn guess_close(guess: usize, tol: f64, w: usize) -> bool {
    let l = -tol.ln();
    let w = w as f64;
    let k = ((l - w) + ((l - w).powi(2) + 8.0 * l * w).sqrt()) / 4.0;
    let want = k.ceil().max(2.0) as i64;
    (guess as i64 - want).abs() <= 1
}

/// Pass, written before the first run: for tol = 10^-j (j = 1..=300) and 3^-j (j = 1..=600), `order_guess` with w = 0
/// (Jorba and Zou's L / 2) and with w = `STEP_COST_W` is within one of the closed-form optimum, and
/// `Plan::candidate_orders` is increasing, distinct, at least 2 and holds `Plan::optimal_kt`. The control: the guess
/// plus two fails `guess_close` at every one of those tolerances.
#[test]
fn order_guess_is_the_cost_model_optimum() {
    let tols: Vec<f64> = (1..=300)
        .map(|j| 10f64.powi(-j))
        .chain((1..=600).map(|j| 3f64.powi(-j)))
        .collect();
    for &tol in &tols {
        for w in [0, STEP_COST_W] {
            let g = order_guess(tol, w);
            assert!(guess_close(g, tol, w), "tol {tol:e} w {w}: guess {g}");
            assert!(!guess_close(g + 2, tol, w), "control did not fire at tol {tol:e} w {w}");
        }
        let c = Plan::candidate_orders(tol);
        assert!(c.windows(2).all(|p| p[0] < p[1]) && c[0] >= 2 && c.contains(&Plan::optimal_kt(tol)));
    }
    println!(
        "optimal_kt at 1e-30, 1e-60, 1e-100: {} {} {}",
        Plan::optimal_kt(1e-30),
        Plan::optimal_kt(1e-60),
        Plan::optimal_kt(1e-100)
    );
}

/// The control of `auto_order_is_the_least_cost` behind the feature `mutate`: with `Control::FirstOrder` the choice
/// keeps the lowest candidate, which `check_choice` must reject on both models at 1e-32.
#[cfg(feature = "mutate")]
#[test]
fn auto_order_control_first_order_fails() {
    fn fires<M: Model>() -> bool {
        let orders = Plan::candidate_orders(1e-32);
        let mut sh = M::shooting();
        sh.set_control(multishoot::control::Control::FirstOrder);
        let z = M::z(&sh);
        let choice = sh.plan_segments_auto(&z[0], &z[1], 1e-32, 0.5, &orders, step_cost);
        let r = check_choice(&sh, &choice, &brute::<M>(1e-32, 0.5, &orders), step_cost);
        println!("control: {r:?}");
        r.is_err()
    }
    assert!(fires::<Rotor<B>>() && fires::<Planted<B>>(), "control did not fire");
}
