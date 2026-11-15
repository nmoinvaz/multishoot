//! The argument-principle count on a synthetic g(lambda) = prod (lambda - z_i) exp(c lambda + d lambda^2) with zeros
//! at known places, including the two that a count on the arg steps alone misses: a close pair of zeros between two
//! samples, and a row of zeros along an edge that turns the arg by whole turns between samples. Each rule of the
//! sampling has a control that switches it off and must miscount.
use multishoot::winding::{winding, winding_batch, WindOpts, Winding, C64};

/// (ln|g|, arg g) of g = prod (lambda - z_i) exp(c lambda + d lambda^2)
fn planted(zeros: &[C64], c: C64, d: C64) -> impl Fn(C64) -> Result<(f64, f64), String> + '_ {
    move |l: C64| {
        let (mut ln, mut arg) = (0.0, 0.0);
        for z in zeros {
            let (dr, di) = (l.0 - z.0, l.1 - z.1);
            ln += dr.hypot(di).ln();
            arg += di.atan2(dr);
        }
        // c lambda + d lambda^2
        let l2 = (l.0 * l.0 - l.1 * l.1, 2.0 * l.0 * l.1);
        ln += c.0 * l.0 - c.1 * l.1 + d.0 * l2.0 - d.1 * l2.1;
        arg += c.0 * l.1 + c.1 * l.0 + d.0 * l2.1 + d.1 * l2.0;
        Ok((ln, arg))
    }
}

/// the rectangle (x0, x1) x (y0, y1), counter-clockwise
fn rect(x0: f64, x1: f64, y0: f64, y1: f64) -> Vec<C64> {
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

/// (count, |turns - count|, resolved, largest step)
fn count(zeros: &[C64], c: C64, d: C64, poly: &[C64], opts: &WindOpts) -> (i64, f64, bool, f64) {
    let w = winding(planted(zeros, c, d), poly, opts).unwrap();
    (
        w.count(),
        (w.turns - w.count() as f64).abs(),
        w.resolved,
        w.largest_step,
    )
}

/// whether a count is right: the planted number, an integer to 1e-9, resolved, every step at most pi / 4
fn right((n, frac, ok, step): (i64, f64, bool, f64), want: i64) -> bool {
    n == want && frac <= 1e-9 && ok && step <= std::f64::consts::FRAC_PI_4 + 1e-12
}

/// Pass, written before the first run: with zeros 1 + 0.3i, 2.5 - 0.4i, 3.2 + 0.6i inside (0, 4) x (-1, 1) and 5,
/// 1 + 2i, -1 - 0.5i outside, and the growth exp((3 + 0.7i) lambda + (0.2 + 0.1i) lambda^2), the counts in (0, 4) x
/// (-1, 1), (0, 2) x (-1, 1) and (2, 4) x (0, 1) are 3, 1 and 1, each an integer to 1e-9, resolved, with every step at
/// most pi / 4. The control: with the zero 3.2 + 0.6i moved to 3.2 - 0.6i the last box counts 0, which fails.
#[test]
fn counts_planted_zeros_in_boxes() {
    let (c, d) = ((3.0, 0.7), (0.2, 0.1));
    let zeros = [
        (1.0, 0.3),
        (2.5, -0.4),
        (3.2, 0.6),
        (5.0, 0.0),
        (1.0, 2.0),
        (-1.0, -0.5),
    ];
    let opts = WindOpts::default();
    for (poly, want) in [
        (rect(0.0, 4.0, -1.0, 1.0), 3),
        (rect(0.0, 2.0, -1.0, 1.0), 1),
        (rect(2.0, 4.0, 0.0, 1.0), 1),
    ] {
        let r = count(&zeros, c, d, &poly, &opts);
        println!("box {poly:?}: {r:?}");
        assert!(right(r, want), "{poly:?}: {r:?}, want {want}");
    }
    let mut moved = zeros;
    moved[2] = (3.2, -0.6);
    let r = count(&moved, c, d, &rect(2.0, 4.0, 0.0, 1.0), &opts);
    assert!(!right(r, 1), "control did not fire: {r:?}");
}

/// the settings of the pair and row cases: an initial grid of exactly 8 intervals of 0.5 on the edges of length 4
fn half() -> WindOpts {
    WindOpts {
        h0: 0.5,
        ..WindOpts::default()
    }
}

/// Pass: two zeros 0.008 apart, 0.01 inside the top edge of (0, 4) x (-1, 1) in the middle of an interval of the
/// initial grid (Re 1.25 and 1.258), and no growth, so that the arg turns by about 2 pi between the interval's two
/// samples and moves slowly elsewhere: the count is 2, resolved, with every step at most pi / 4. The controls: without
/// ln|g| in the alias rule's scale, and without the alias rule, the pair is missed. A development case: its first
/// version, with a fast growth exp(14 i lambda), resolved the pair through the arg rate alone, so the ln|g| control
/// did not fire there.
#[test]
fn close_pair_between_samples() {
    let zeros = [(1.25, 0.99), (1.258, 0.99)];
    let (c, d) = ((0.0, 0.0), (0.0, 0.0));
    let poly = rect(0.0, 4.0, -1.0, 1.0);
    let r = count(&zeros, c, d, &poly, &half());
    println!("close pair: {r:?}");
    assert!(right(r, 2), "{r:?}");
    for (name, opts) in [
        (
            "no ln|g| in the scale",
            WindOpts {
                use_ln: false,
                ..half()
            },
        ),
        (
            "no alias rule",
            WindOpts {
                alias: f64::INFINITY,
                ..half()
            },
        ),
    ] {
        let r = count(&zeros, c, d, &poly, &opts);
        println!("control, {name}: {r:?}");
        assert!(!right(r, 2), "control {name} did not fire: {r:?}");
    }
}

/// (ln|g|, arg g) of g = sin(pi (lambda - z0) / p): the infinite row of zeros z0 + j p, along which ln|g| and the arg's
/// steps over two periods repeat exactly
fn row(z0: C64, p: f64) -> impl Fn(C64) -> Result<(f64, f64), String> {
    move |l: C64| {
        let (a, b) = (
            std::f64::consts::PI * (l.0 - z0.0) / p,
            std::f64::consts::PI * (l.1 - z0.1) / p,
        );
        // sin(a + ib) = sin a cosh b + i cos a sinh b
        let (sa, ca) = a.sin_cos();
        Ok((
            0.5 * (sa * sa + b.sinh() * b.sinh()).ln(),
            (ca * b.sinh()).atan2(sa * b.cosh()),
        ))
    }
}

/// Pass: the infinite row of zeros 0.125 + 0.25 j + 0.9 i, 0.1 below the top edge of (0, 4) x (-1, 1) (16 of them
/// inside), so that the arg turns by 2 pi in every interval of the initial grid along the top edge while its samples,
/// midway between zeros, see the same ln|g| and a flat arg: the count is 16, resolved, with every step at most pi / 4.
/// Only the perpendicular edges, which pass between zeros near the top corners, see the row. The controls: without the
/// alias rule, and without the corners' rates, the row is missed. A development case: its first version, a finite row,
/// let ln|g| fall along the top edge and resolved the row without the corners, so the corner control did not fire.
#[test]
fn long_row_along_an_edge() {
    let g = row((0.125, 0.9), 0.25);
    let poly = rect(0.0, 4.0, -1.0, 1.0);
    let run = |opts: &WindOpts| {
        let w = winding(&g, &poly, opts).unwrap();
        (
            w.count(),
            (w.turns - w.count() as f64).abs(),
            w.resolved,
            w.largest_step,
        )
    };
    let r = run(&half());
    println!("long row: {r:?}");
    assert!(right(r, 16), "{r:?}");
    for (name, opts) in [
        (
            "no alias rule",
            WindOpts {
                alias: f64::INFINITY,
                ..half()
            },
        ),
        (
            "no corners",
            WindOpts {
                corners: false,
                ..half()
            },
        ),
    ] {
        let r = run(&opts);
        println!("control, {name}: {r:?}");
        assert!(!right(r, 16), "control {name} did not fire: {r:?}");
    }
}

fn cm(a: C64, b: C64) -> C64 {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Pass, written before the first run: an edge whose initial grid alone exceeds the evaluations allowed (h0 1e-12,
/// max_evals 16) is refused with an error before the grid is allocated. The control: h0 0.5 with max_evals 1000
/// counts.
#[test]
fn initial_grid_beyond_the_budget_is_refused() {
    let g = planted(&[], (0.0, 0.0), (0.0, 0.0));
    let poly = rect(0.0, 1.0, 0.0, 1.0);
    let tiny = WindOpts {
        h0: 1e-12,
        max_evals: 16,
        ..WindOpts::default()
    };
    let r = winding(&g, &poly, &tiny);
    println!(
        "h0 1e-12, max_evals 16: {:?}",
        r.as_ref().map(|w| w.count()).map_err(|e| e.clone())
    );
    assert!(r.is_err());
    let fine = WindOpts {
        h0: 0.5,
        max_evals: 1000,
        ..WindOpts::default()
    };
    assert!(winding(&g, &poly, &fine).is_ok(), "control did not fire");
}

/// Pass, written before the first run: on the unit square with g = exp(0.6 i z), h0 1 and hmin 0.6, two steps exceed
/// tol / 2 and their verification would need intervals of 0.5, below hmin: the count is not resolved. The control:
/// with hmin 1e-4 it is.
#[test]
fn verification_below_hmin_is_unresolved() {
    let g = planted(&[], (0.0, 0.6), (0.0, 0.0));
    let poly = rect(0.0, 1.0, 0.0, 1.0);
    let w = winding(
        &g,
        &poly,
        &WindOpts {
            h0: 1.0,
            hmin: 0.6,
            ..WindOpts::default()
        },
    )
    .unwrap();
    println!("hmin 0.6: resolved {}", w.resolved);
    assert!(!w.resolved);
    let w = winding(
        &g,
        &poly,
        &WindOpts {
            h0: 1.0,
            hmin: 1e-4,
            ..WindOpts::default()
        },
    )
    .unwrap();
    assert!(w.resolved, "control did not fire");
}

/// Pass: g = exp(q(z)), q(z) = 0.6 z - 320 z (z - 1)(z - 1 - i)(z - i)(z - 1 - i / 2), has no zeros, so on the unit
/// square from h0 1 its count is 0 and its moment 0 to 1e-2, the accuracy it needs as Muller's start, resolved: the
/// samples that verification inserts, where q reaches 100 + 0.3 i, are checked against the alias rule again (without
/// that, the count came out resolved with a moment of 7.96). The first criterion, 1e-6, failed at 1.7e-4, the error of
/// the moment's midpoint rule where ln g changes by up to pi / 2 per interval, not of the sampling.
#[test]
fn verification_samples_are_rechecked() {
    let g = |z: C64| -> Result<(f64, f64), String> {
        let mut p = cm(z, (z.0 - 1.0, z.1));
        p = cm(p, (z.0 - 1.0, z.1 - 1.0));
        p = cm(p, (z.0, z.1 - 1.0));
        p = cm(p, (z.0 - 1.0, z.1 - 0.5));
        Ok((0.6 * z.0 - 320.0 * p.0, 0.6 * z.1 - 320.0 * p.1))
    };
    let w = winding(
        g,
        &rect(0.0, 1.0, 0.0, 1.0),
        &WindOpts {
            h0: 1.0,
            ..WindOpts::default()
        },
    )
    .unwrap();
    let m = w.moment();
    println!(
        "exp(q): count {}, moment {m:?}, resolved {}, {} evaluations",
        w.count(),
        w.resolved,
        w.evals
    );
    assert!(w.count() == 0 && m.0.hypot(m.1) <= 1e-2 && w.resolved);
}

/// Pass, written before the first run: Muller's method for g(z) = z started at its zero returns that zero, where the
/// normalization by g there would divide by zero.
#[test]
fn muller_started_at_a_zero() {
    let g = |z: C64| -> Result<(f64, f64), String> { Ok((z.0.hypot(z.1).ln(), z.1.atan2(z.0))) };
    let r = multishoot::winding::muller(g, (0.0, 0.0), 1.0, 1e-14, 50);
    println!("muller from the zero: {r:?}");
    assert_eq!(r, Ok((0.0, 0.0)));
}

/// the bits of a winding's samples (every edge's t and values), turns, evaluations, resolution and mismatches
fn winding_bits(w: &Winding) -> (Vec<u64>, u64, usize, bool, Vec<usize>) {
    let samples = w
        .edges
        .iter()
        .flat_map(|e| {
            e.t.iter()
                .map(|t| t.to_bits())
                .chain(e.val.iter().flat_map(|v| [v.0.to_bits(), v.1.to_bits()]))
        })
        .collect();
    let mm = w.edges.iter().map(|e| e.mismatches).collect();
    (samples, w.turns.to_bits(), w.evals, w.resolved, mm)
}

/// Pass, written before the first run: `winding_batch`, its f evaluating each batch at once, gives `winding`'s samples,
/// turns, evaluations, resolution and mismatches bit for bit on the planted zeros of `counts_planted_zeros_in_boxes`
/// in the box (0, 4) x (-1, 1) and with the close pair of `close_pair_between_samples`; and the same error where the
/// budget runs out (max_evals 40) and where f fails on the half plane re lambda > 3.1 (its message names the point).
/// The batches hold more than one point. The control: g with one zero moved by 1e-9 changes the samples.
#[test]
fn batched_sampling_is_the_one_by_one() {
    let (c, d) = ((3.0, 0.7), (0.2, 0.1));
    let zeros = [(1.0, 0.3), (2.5, -0.4), (3.2, 0.6), (5.0, 0.0)];
    let g = planted(&zeros, c, d);
    let pair = [(0.5, 0.5), (0.5 + 1e-3, 0.5)];
    let gp = planted(&pair, (0.0, 0.0), (0.0, 0.0));
    let poly = rect(0.0, 4.0, -1.0, 1.0);
    let opts = WindOpts::default();
    let largest = std::cell::Cell::new(0);
    let batched = |f: &dyn Fn(C64) -> Result<(f64, f64), String>, poly: &[C64], opts: &WindOpts| {
        winding_batch(
            |xs: &[C64]| {
                largest.set(largest.get().max(xs.len()));
                xs.iter().map(|&x| f(x)).collect()
            },
            poly,
            opts,
        )
    };
    for (f, poly) in [
        (&g as &dyn Fn(C64) -> _, &poly[..]),
        (&gp, &rect(0.0, 1.0, 0.0, 1.0)[..]),
    ] {
        let one = winding(f, poly, &opts).unwrap();
        let all = batched(f, poly, &opts).unwrap();
        assert!(winding_bits(&all) == winding_bits(&one), "{poly:?}");
    }
    let small = WindOpts { max_evals: 40, ..opts };
    let e1 = winding(&g, &poly, &small).unwrap_err();
    assert_eq!(batched(&g, &poly, &small).unwrap_err(), e1);
    let failing = |x: C64| {
        if x.0 > 3.1 {
            Err(format!("no value at {x:?}"))
        } else {
            g(x)
        }
    };
    let e2 = winding(failing, &poly, &opts).unwrap_err();
    assert_eq!(batched(&failing, &poly, &opts).unwrap_err(), e2);
    println!("errors: {e1}; {e2}; largest batch {}", largest.get());
    assert!(largest.get() > 1, "no batch held more than one point");
    let moved = [(1.0, 0.3), (2.5, -0.4), (3.2 + 1e-9, 0.6), (5.0, 0.0)];
    let gm = planted(&moved, c, d);
    let one = winding(&g, &poly, &opts).unwrap();
    assert!(
        winding_bits(&batched(&gm, &poly, &opts).unwrap()) != winding_bits(&one),
        "control did not fire"
    );
}

/// Pass, written before the first run: where the budget runs out (max_evals 40 on the box (0, 4) x (-1, 1)), the error
/// is "winding: more than 40 evaluations" and f was given 40 points in all; where f fails at re lambda >= 4, the error
/// names (4, -1), the corner that ends the first edge, the first such point in the edges' order. The controls feed the
/// same checks wrong errors and must be rejected: the run with max_evals 41, and an f whose every failure names the
/// last failing point of its batch.
#[test]
fn errors_name_the_first_point_in_order() {
    let zeros = [(1.0, 0.3), (2.5, -0.4), (3.2, 0.6), (5.0, 0.0)];
    let g = planted(&zeros, (3.0, 0.7), (0.2, 0.1));
    let poly = rect(0.0, 4.0, -1.0, 1.0);
    let opts = WindOpts::default();
    // the budget's error and the number of points f was given
    let budget = |max_evals: usize| {
        let given = std::cell::Cell::new(0);
        let f = |xs: &[C64]| {
            given.set(given.get() + xs.len());
            xs.iter().map(|&x| g(x)).collect()
        };
        let e = winding_batch(f, &poly, &WindOpts { max_evals, ..opts }).unwrap_err();
        (e, given.get())
    };
    // f failing at re lambda >= 4, its failures naming their own point or, with `last`, the batch's last failing one
    let failing = |last: bool| {
        let f = |xs: &[C64]| {
            let end = xs.iter().rev().find(|x| x.0 >= 4.0).copied();
            xs.iter()
                .map(|&x| match (x.0 >= 4.0, last) {
                    (false, _) => g(x),
                    (true, false) => Err(format!("no value at {x:?}")),
                    (true, true) => Err(format!("no value at {:?}", end.unwrap())),
                })
                .collect()
        };
        winding_batch(f, &poly, &opts).unwrap_err()
    };
    let budget_ok = |(e, given): &(String, usize)| e == "winding: more than 40 evaluations" && *given == 40;
    let first_ok = |e: &String| *e == format!("no value at {:?}", (4.0, -1.0));
    let (b, f) = (budget(40), failing(false));
    let (b_wrong, f_wrong) = (budget(41), failing(true));
    println!("budget {b:?}, control {b_wrong:?}; failure {f}, control {f_wrong}");
    assert!(budget_ok(&b) && first_ok(&f));
    assert!(!budget_ok(&b_wrong) && !first_ok(&f_wrong), "control did not fire");
}

/// Pass, written before the first run: on the box (0, 100) x (-1, 1) at h0 0.01 with max_evals 15000, each edge's grid
/// fits the budget but the grids together run out of it on the third edge's, the top. The error is then "winding: more
/// than 15000 evaluations" with 15000 points given to f, also when f fails only past the budget (on the top edge at re
/// lambda < 50); and when f fails on the top edge at re lambda > 60, before the budget runs out, the error is f's at
/// (100, 1), the corner that starts it. Each check, fed the other case's error, must reject it.
#[test]
fn budget_in_the_first_grids_comes_after_errors_of_f() {
    let poly = rect(0.0, 100.0, -1.0, 1.0);
    let opts = WindOpts {
        h0: 0.01,
        max_evals: 15000,
        ..WindOpts::default()
    };
    // the error where f fails at the points `bad` picks, and the number of points f was given
    let run = |bad: &dyn Fn(C64) -> bool| {
        let given = std::cell::Cell::new(0);
        let f = |xs: &[C64]| {
            given.set(given.get() + xs.len());
            xs.iter()
                .map(|&x| {
                    if bad(x) {
                        Err(format!("no value at {x:?}"))
                    } else {
                        Ok((0.0, 0.0))
                    }
                })
                .collect()
        };
        (winding_batch(f, &poly, &opts).unwrap_err(), given.get())
    };
    let budget = |(e, given): &(String, usize)| e == "winding: more than 15000 evaluations" && *given == 15000;
    let corner = |(e, _): &(String, usize)| *e == format!("no value at {:?}", (100.0, 1.0));
    let none = run(&|_| false);
    let past = run(&|x| x.1 == 1.0 && x.0 < 50.0);
    let before = run(&|x| x.1 == 1.0 && x.0 > 60.0);
    println!("{none:?}; {past:?}; {before:?}");
    assert!(budget(&none) && budget(&past) && corner(&before));
    assert!(!budget(&before) && !corner(&past), "control did not fire");
}
