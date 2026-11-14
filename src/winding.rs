//! Counting and locating the zeros of an analytic function g(lambda) inside a closed polygon by the argument principle,
//! from its values as (ln|g|, arg g): the mode count of the mode matrix's determinant ([`crate::mode`]), whose zeros
//! are the growth rates of a background's modes, or of any such function.
//!
//! The winding number of g along the polygon is the sum of the wrapped steps of arg g between neighbouring samples,
//! over 2 pi: the number of zeros inside minus the number of poles. It holds when no step hides a whole turn, so the
//! sampling is adaptive. Each edge starts on a grid of spacing `h0`, and an interval is bisected while its arg step
//! exceeds `tol`, or while its length times the local rate of ln g exceeds `alias`: the rate |Delta ln g| / h, with
//! Delta ln g = Delta ln|g| + i Delta arg g, on the interval and up to `neighbours` intervals each side, and at an
//! edge's first and last intervals also on the other edge's intervals at that corner (by Cauchy-Riemann the arg rate
//! along an edge is the normal slope of ln|g|, which the perpendicular edge sees unaliased). ln|g| does not wrap, so
//! a close pair of zeros between two samples shows in it. Then every interval whose step exceeds tol / 2 is bisected
//! once more and its halves are checked to add up to its step: a mismatch is a hidden turn. The edges are swept until
//! a sweep bisects nothing and finds no mismatch. An edge that would need an interval below `hmin` is not resolved,
//! and neither is its count.
//!
//! The zeros inside are located by the contour's first moment (1 / 2 pi i) contour integral of lambda d ln g, the sum
//! of the zeros inside minus that of the poles, on the samples, and a zero is refined by Muller's method (Muller, Math.
//! Tables Aids Comput. 10 (1956) 208) on g normalized by its value at the latest point, so that it neither overflows
//! nor underflows.

/// a complex number (re, im) in f64
pub type C64 = (f64, f64);

fn cadd(a: C64, b: C64) -> C64 {
    (a.0 + b.0, a.1 + b.1)
}

fn csub(a: C64, b: C64) -> C64 {
    (a.0 - b.0, a.1 - b.1)
}

fn cmul(a: C64, b: C64) -> C64 {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn cdiv(a: C64, b: C64) -> C64 {
    // Smith's division, which neither overflows nor underflows for scaled operands
    if b.0.abs() >= b.1.abs() {
        let r = b.1 / b.0;
        let d = b.0 + b.1 * r;
        ((a.0 + a.1 * r) / d, (a.1 - a.0 * r) / d)
    } else {
        let r = b.0 / b.1;
        let d = b.0 * r + b.1;
        ((a.0 * r + a.1) / d, (a.1 * r - a.0) / d)
    }
}

fn cabs(a: C64) -> f64 {
    a.0.hypot(a.1)
}

fn csqrt(a: C64) -> C64 {
    let r = cabs(a);
    let re = ((r + a.0) / 2.0).max(0.0).sqrt();
    let im = ((r - a.0) / 2.0).max(0.0).sqrt();
    (re, if a.1 < 0.0 { -im } else { im })
}

/// x reduced to (-pi, pi]
pub fn wrap(x: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let y = x - t * (x / t).round();
    if y <= -std::f64::consts::PI {
        y + t
    } else {
        y
    }
}

/// The settings of [`winding`]
#[derive(Clone, Copy, Debug)]
pub struct WindOpts {
    /// the starting spacing of every edge
    pub h0: f64,
    /// the largest arg step accepted
    pub tol: f64,
    /// the largest interval length times the local rate of ln g (infinite: no alias rule)
    pub alias: f64,
    /// the shortest interval; an edge that needs a shorter one is not resolved
    pub hmin: f64,
    /// the intervals each side whose rates enter an interval's scale
    pub neighbours: usize,
    /// whether the scale includes the change of ln|g| (else the arg step alone)
    pub use_ln: bool,
    /// whether an edge's end intervals include the rates of the other edge at that corner
    pub corners: bool,
    /// the evaluations at most
    pub max_evals: usize,
}

impl Default for WindOpts {
    fn default() -> Self {
        WindOpts {
            h0: 0.4,
            tol: std::f64::consts::FRAC_PI_4,
            alias: std::f64::consts::FRAC_PI_2,
            hmin: 1e-4,
            neighbours: 3,
            use_ln: true,
            corners: true,
            max_evals: 200_000,
        }
    }
}

/// One edge of the polygon: its ends, the samples' parameters t in [0, 1] (lambda = from + t (to - from)) and values
/// (ln|g|, arg g), whether it is resolved, and the mismatched halves its verifications found
#[derive(Clone, Debug)]
pub struct Edge {
    pub from: C64,
    pub to: C64,
    pub t: Vec<f64>,
    pub val: Vec<(f64, f64)>,
    pub resolved: bool,
    pub mismatches: usize,
}

impl Edge {
    fn at(&self, t: f64) -> C64 {
        if t == 1.0 {
            self.to
        } else {
            cadd(self.from, cmul(csub(self.to, self.from), (t, 0.0)))
        }
    }
    fn len(&self) -> f64 {
        cabs(csub(self.to, self.from))
    }
    /// the arg steps of the intervals
    fn steps(&self) -> Vec<f64> {
        self.val.windows(2).map(|w| wrap(w[1].1 - w[0].1)).collect()
    }
    /// the rates |Delta ln g| / h (or |Delta arg| / h) of the intervals
    fn rates(&self, use_ln: bool) -> Vec<f64> {
        let l = self.len();
        (0..self.t.len() - 1)
            .map(|j| {
                let h = l * (self.t[j + 1] - self.t[j]);
                let s = wrap(self.val[j + 1].1 - self.val[j].1);
                let d = if use_ln {
                    (self.val[j + 1].0 - self.val[j].0).hypot(s)
                } else {
                    s.abs()
                };
                d / h
            })
            .collect()
    }
}

/// What [`winding`] found: the edges, the winding number (the sum of the arg steps over 2 pi), whether every edge is
/// resolved, the evaluations of g, and the largest arg step
#[derive(Clone, Debug)]
pub struct Winding {
    pub edges: Vec<Edge>,
    pub turns: f64,
    pub resolved: bool,
    pub evals: usize,
    pub largest_step: f64,
}

impl Winding {
    /// the winding number rounded: zeros minus poles inside
    pub fn count(&self) -> i64 {
        self.turns.round() as i64
    }

    /// (1 / 2 pi i) contour integral of lambda d ln g on the samples (the midpoint of each interval times its step of
    /// ln g): the sum of the zeros inside minus that of the poles
    pub fn moment(&self) -> C64 {
        let mut s = (0.0, 0.0);
        for e in &self.edges {
            for j in 0..e.t.len() - 1 {
                let mid = cmul(cadd(e.at(e.t[j]), e.at(e.t[j + 1])), (0.5, 0.0));
                let dl = (e.val[j + 1].0 - e.val[j].0, wrap(e.val[j + 1].1 - e.val[j].1));
                s = cadd(s, cmul(mid, dl));
            }
        }
        let tau = std::f64::consts::TAU;
        (s.1 / tau, -s.0 / tau)
    }
}

/// a cache of g's evaluations, keyed by lambda's bits, so that the polygon's corners are evaluated once
struct Cache<F> {
    f: F,
    seen: std::collections::HashMap<(u64, u64), (f64, f64)>,
    max: usize,
}

impl<F: FnMut(C64) -> Result<(f64, f64), String>> Cache<F> {
    fn get(&mut self, x: C64) -> Result<(f64, f64), String> {
        let key = (x.0.to_bits(), x.1.to_bits());
        if let Some(&v) = self.seen.get(&key) {
            return Ok(v);
        }
        if self.seen.len() >= self.max {
            return Err(format!("winding: more than {} evaluations", self.max));
        }
        let v = (self.f)(x)?;
        self.seen.insert(key, v);
        Ok(v)
    }
}

/// the scale of interval j of edge e: the largest rate on it and up to `n` intervals each side, and with `corners` on
/// the other edge's first or last `n` intervals at a corner within `n` intervals of it
fn scales(edges: &[Edge], e: usize, opts: &WindOpts) -> Vec<f64> {
    let k = edges.len();
    let r = edges[e].rates(opts.use_ln);
    let n = opts.neighbours;
    let prev = edges[(e + k - 1) % k].rates(opts.use_ln);
    let next = edges[(e + 1) % k].rates(opts.use_ln);
    let corner_in = prev[prev.len().saturating_sub(n)..]
        .iter()
        .fold(0.0f64, |a, &b| a.max(b));
    let corner_out = next[..n.min(next.len())].iter().fold(0.0f64, |a, &b| a.max(b));
    (0..r.len())
        .map(|j| {
            let lo = j.saturating_sub(n);
            let hi = (j + n + 1).min(r.len());
            let mut s = r[lo..hi].iter().fold(0.0f64, |a, &b| a.max(b));
            if opts.corners && j < n {
                s = s.max(corner_in);
            }
            if opts.corners && j + n >= r.len() {
                s = s.max(corner_out);
            }
            s
        })
        .collect()
}

/// insert the midpoint of interval j of edge e
fn bisect<F: FnMut(C64) -> Result<(f64, f64), String>>(
    edge: &mut Edge,
    j: usize,
    cache: &mut Cache<F>,
) -> Result<(), String> {
    let tm = (edge.t[j] + edge.t[j + 1]) / 2.0;
    let v = cache.get(edge.at(tm))?;
    edge.t.insert(j + 1, tm);
    edge.val.insert(j + 1, v);
    Ok(())
}

/// The winding number of g along the closed polygon through `vertices` (in order; the last joins the first), from
/// `f(lambda)` = (ln|g(lambda)|, arg g(lambda)), sampled as the module's documentation describes.
pub fn winding<F: FnMut(C64) -> Result<(f64, f64), String>>(
    f: F,
    vertices: &[C64],
    opts: &WindOpts,
) -> Result<Winding, String> {
    if vertices.len() < 3 {
        return Err("winding: a polygon needs three vertices or more".into());
    }
    let mut cache = Cache {
        f,
        seen: std::collections::HashMap::new(),
        max: opts.max_evals,
    };
    let k = vertices.len();
    let mut edges = Vec::with_capacity(k);
    for i in 0..k {
        let (from, to) = (vertices[i], vertices[(i + 1) % k]);
        let mut e = Edge {
            from,
            to,
            t: Vec::new(),
            val: Vec::new(),
            resolved: true,
            mismatches: 0,
        };
        let n = (e.len() / opts.h0).ceil().max(1.0) as usize;
        e.t = (0..=n)
            .map(|j| if j == n { 1.0 } else { j as f64 / n as f64 })
            .collect();
        e.val = e.t.iter().map(|&t| cache.get(e.at(t))).collect::<Result<_, _>>()?;
        edges.push(e);
    }
    loop {
        let mut changed = false;
        for e in 0..k {
            let l = edges[e].len();
            // bisection: steps above tol, intervals that the local rate says may alias
            loop {
                let sc = scales(&edges, e, opts);
                let steps = edges[e].steps();
                let mut split = Vec::new();
                for j in 0..steps.len() {
                    let h = l * (edges[e].t[j + 1] - edges[e].t[j]);
                    if steps[j].abs() > opts.tol || h * sc[j] > opts.alias {
                        if h / 2.0 < opts.hmin {
                            edges[e].resolved = false;
                        } else {
                            split.push(j);
                        }
                    }
                }
                if split.is_empty() {
                    break;
                }
                changed = true;
                for &j in split.iter().rev() {
                    bisect(&mut edges[e], j, &mut cache)?;
                }
            }
            // verification: the halves of every step above tol / 2 must add up to it
            let steps = edges[e].steps();
            let check: Vec<usize> = (0..steps.len())
                .filter(|&j| {
                    steps[j].abs() > opts.tol / 2.0 && l * (edges[e].t[j + 1] - edges[e].t[j]) / 2.0 >= opts.hmin
                })
                .collect();
            for &j in check.iter().rev() {
                let whole = steps[j];
                bisect(&mut edges[e], j, &mut cache)?;
                let v = &edges[e].val;
                let (h1, h2) = (wrap(v[j + 1].1 - v[j].1), wrap(v[j + 2].1 - v[j + 1].1));
                if wrap(h1 + h2 - whole).abs() > 1e-9 {
                    edges[e].mismatches += 1;
                    changed = true;
                }
                if h1.abs() > opts.tol || h2.abs() > opts.tol {
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let steps: Vec<f64> = edges.iter().flat_map(|e| e.steps()).collect();
    let turns = steps.iter().sum::<f64>() / std::f64::consts::TAU;
    let largest_step = steps.iter().fold(0.0f64, |a, s| a.max(s.abs()));
    let resolved = edges.iter().all(|e| e.resolved);
    let evals = cache.seen.len();
    Ok(Winding {
        edges,
        turns,
        resolved,
        evals,
        largest_step,
    })
}

/// A zero of g near `x0` by Muller's method from x0 - h, x0 + h and x0, with `f(lambda)` = (ln|g|, arg g): each
/// step's three values are g / g(latest point), so that they neither overflow nor underflow. It stops when a step is
/// at most `tol` max(1, |lambda|), and fails after `max_iter` steps or at a non-finite value.
pub fn muller<F: FnMut(C64) -> Result<(f64, f64), String>>(
    mut f: F,
    x0: C64,
    h: f64,
    tol: f64,
    max_iter: usize,
) -> Result<C64, String> {
    let mut xs = [csub(x0, (h, 0.0)), cadd(x0, (h, 0.0)), x0];
    let mut ls = [f(xs[0])?, f(xs[1])?, f(xs[2])?];
    for _ in 0..max_iter {
        // g_i / g_2
        let g = |i: usize| -> C64 {
            let (dl, da) = (ls[i].0 - ls[2].0, ls[i].1 - ls[2].1);
            let m = dl.exp();
            (m * da.cos(), m * da.sin())
        };
        let (f0, f1, f2) = (g(0), g(1), (1.0, 0.0));
        let q = cdiv(csub(xs[2], xs[1]), csub(xs[1], xs[0]));
        let q1 = cadd(q, (1.0, 0.0));
        let a = cadd(csub(cmul(q, f2), cmul(cmul(q, q1), f1)), cmul(cmul(q, q), f0));
        let b = cadd(
            csub(cmul(cadd(cmul(q, (2.0, 0.0)), (1.0, 0.0)), f2), cmul(cmul(q1, q1), f1)),
            cmul(cmul(q, q), f0),
        );
        let c = cmul(q1, f2);
        let disc = csqrt(csub(cmul(b, b), cmul(cmul(a, c), (4.0, 0.0))));
        let (dp, dm) = (cadd(b, disc), csub(b, disc));
        let den = if cabs(dp) >= cabs(dm) { dp } else { dm };
        if cabs(den) == 0.0 || !cabs(den).is_finite() {
            return Err("muller: a vanishing or non-finite denominator".into());
        }
        let step = cdiv(cmul(csub(xs[2], xs[1]), cmul(c, (2.0, 0.0))), den);
        let x3 = csub(xs[2], step);
        if !cabs(x3).is_finite() {
            return Err("muller: a non-finite iterate".into());
        }
        if cabs(step) <= tol * cabs(x3).max(1.0) {
            return Ok(x3);
        }
        let l3 = f(x3)?;
        if !(l3.0.is_finite() || l3.0 == f64::NEG_INFINITY) {
            return Err("muller: a non-finite value".into());
        }
        if l3.0 == f64::NEG_INFINITY {
            return Ok(x3);
        }
        xs = [xs[1], xs[2], x3];
        ls = [ls[1], ls[2], l3];
    }
    Err(format!("muller: no convergence in {max_iter} steps"))
}
