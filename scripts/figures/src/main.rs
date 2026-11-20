//! The data of the README's figures: each function runs multishoot on a model problem and writes what its figure draws
//! as JSON into the directory given (docs/figures/data). The models are the integration tests' (tests/common) and four
//! copied from the README and the tests: the README's example, Bratu's problem with lambda a global and with lambda
//! the equations' parameter, and the linear wave of the mode-matrix tests.
#[path = "../../../tests/common/mod.rs"]
mod common;

use common::{perturbed, wave, Fnv, Planted, Rotor};
use multishoot::lane::{DMat, Dtn, Fu, Lane, Param, DV};
use multishoot::multidouble::dd::split;
use multishoot::multidouble::simd::{Native, Simd};
use multishoot::multisolver::anm::series;
use multishoot::multisolver::arclength::{continuation, locate_fold, start, ArcOpts, Bordered, FoldOpts};
use multishoot::multisolver::newton::{exact, Factored};
use multishoot::multisolver::prec::{parts, widen};
use multishoot::multisolver::quasi::QuasiOpts;
use multishoot::multisolver::winding::WindOpts;
use multishoot::multisolver::NewtonOpts;
use multishoot::periodic::{dmat, sin_cos_grid, theta, Parity, Resampler};
use multishoot::taylor::{plan_path, step_cost, taylor_path, taylor_step};
use multishoot::{cs, Double, Equations, ExtraRow, Fl, InGlobal, Plan, Quad, Rate, Shooting, State, Storage, Work, DD};
use std::time::Instant;

// ---------------------------------------------------------------------------------------------------------------
// JSON

fn num(x: f64) -> String {
    if x.is_finite() {
        format!("{x:e}")
    } else {
        "null".into()
    }
}

fn arr(v: impl IntoIterator<Item = f64>) -> String {
    format!("[{}]", v.into_iter().map(num).collect::<Vec<_>>().join(","))
}

fn arr2(v: &[Vec<f64>]) -> String {
    format!(
        "[{}]",
        v.iter().map(|r| arr(r.iter().copied())).collect::<Vec<_>>().join(",")
    )
}

fn ints(v: impl IntoIterator<Item = usize>) -> String {
    format!(
        "[{}]",
        v.into_iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
    )
}

/// `{"k": v, ...}` from pairs whose values are JSON already
fn obj(fields: &[(&str, String)]) -> String {
    let body: Vec<String> = fields.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
    format!("{{{}}}", body.join(","))
}

fn string(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// the output of a command, trimmed; None when it fails or prints nothing
fn output(cmd: &str, args: &[&str]) -> Option<String> {
    let o = std::process::Command::new(cmd).args(args).output().ok()?;
    let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
    (o.status.success() && !s.is_empty()).then_some(s)
}

/// the CPU, memory and OS of this machine, as data_graphs.host_specs gives them
fn host() -> String {
    if cfg!(target_os = "macos") {
        let cpu = output("sysctl", &["-n", "machdep.cpu.brand_string"]);
        let mem = output("sysctl", &["-n", "hw.memsize"]).and_then(|m| m.parse::<u64>().ok());
        if let (Some(c), Some(m), Some(v)) = (cpu, mem, output("sw_vers", &["-productVersion"])) {
            return format!("{c}, {} GB, macOS {v}", m >> 30);
        }
    } else if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(l) = info.lines().find(|l| l.starts_with("model name")) {
            return l.split(':').nth(1).unwrap_or_default().trim().to_string();
        }
    }
    format!("{} {}", std::env::consts::ARCH, std::env::consts::OS)
}

/// Where a figure's data came from, saved with it so that a redraw elsewhere keeps it: the engine's version, the last
/// commit of its sources, the tests' models and the lock (marked dirty when they differ from it), and the machine
fn source() -> &'static str {
    static SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SOURCE.get_or_init(|| {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
        let version = std::fs::read_to_string(format!("{root}/Cargo.toml"))
            .ok()
            .and_then(|t| {
                t.lines()
                    .find(|l| l.starts_with("version"))
                    .map(|l| l.split('"').nth(1).unwrap_or_default().to_string())
            })
            .unwrap_or_default();
        let git = |args: &[&str]| output("git", &[&["-C", root], args].concat()).unwrap_or_default();
        let engine = ["--", "src", "tests/common", "Cargo.lock"];
        let mut commit = git(&[&["log", "-1", "--format=%h"][..], &engine].concat());
        if !git(&[&["status", "--porcelain"][..], &engine].concat()).is_empty() {
            commit.push_str("+dirty");
        }
        obj(&[
            ("version", string(&version)),
            ("commit", string(&commit)),
            ("machine", string(&host())),
        ])
    })
}

/// the figure's data with its source, as `<dir>/<name>.json`
fn save(dir: &str, name: &str, body: String) {
    std::fs::create_dir_all(dir).unwrap();
    let body = format!("{{\"source\":{},{}", source(), &body[1..]);
    std::fs::write(format!("{dir}/{name}.json"), body + "\n").unwrap();
    println!("wrote {dir}/{name}.json");
}

/// the median wall time of `f` over `n` runs, in milliseconds
fn median_ms(n: usize, mut f: impl FnMut()) -> f64 {
    let mut t: Vec<f64> = (0..n)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[n / 2]
}

fn opts(tol: f64, max_iter: usize) -> NewtonOpts {
    NewtonOpts {
        tol,
        max_iter,
        par: false,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The README's example: y'' = 3/2 y^2 on [0, 1], y = 4 / (a + x)^2 with a = 1 + j / 8 on lane j

struct Usage {
    y0: Fl<DV, 1>,
    y1: Fl<DV, 1>,
}

impl Equations<1> for Usage {
    const NF: usize = 2;
    const NL: usize = 1;
    const NR: usize = 1;
    const POINTWISE: bool = true;
    type Prep = ();

    fn taylor_coeffs<T: Lane>(&self, _x0: f64, _g: &[T], w: &mut Work<T, 1>) {
        for k in 0..w.kt {
            let ik = DD::ONE / DD::from_f64((k + 1) as f64);
            let yy = cs(&w.c[0], &w.c[0], k);
            w.c[0][k + 1] = w.c[1][k].muls(ik);
            w.c[1][k + 1] = yy.muls(DD::from_f64(1.5) * ik);
        }
    }

    fn left_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], _g: &[T], _prep: &()) -> State<T, 1> {
        vec![Fl::cst(&self.y0), data[0]]
    }

    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], _g: &[T], _prep: &()) -> State<T, 1> {
        vec![Fl::cst(&self.y1), data[0]]
    }
}

fn usage_exact(x: f64) -> Fl<DV, 1> {
    let v: Vec<DD> = (0..8)
        .map(|j| {
            let s = DD::from_f64(1.0 + j as f64 / 8.0 + x);
            DD::from_f64(4.0) / (s * s)
        })
        .collect();
    let (hi, lo) = split(&v);
    Fl::load(&hi, &lo)
}

/// the README's layout, planned from its start (the line between the end values), and that start
fn usage() -> (Shooting<Usage, 1>, Vec<Vec<f64>>) {
    let model = Usage {
        y0: usage_exact(0.0),
        y1: usage_exact(1.0),
    };
    let mut sh = Shooting::new(model, vec![0.0, 0.25, 0.5, 0.75, 1.0], 24);
    let mut z = vec![vec![0.0; sh.len()]; 2];
    for j in 0..8 {
        let (y0, y1) = (sh.eq.y0.get(j).hi, sh.eq.y1.get(j).hi);
        z[0][j] = y1 - y0;
        for (i, &x) in sh.nodes.iter().enumerate() {
            z[0][sh.off_node(i) + j] = y0 + (y1 - y0) * x;
            z[0][sh.off_node(i) + 8 + j] = y1 - y0;
        }
        z[0][sh.off_right() + j] = y1 - y0;
    }
    let plan = Plan {
        kt: 24,
        tol: 1e-32,
        hmax: 0.25,
    };
    sh.plan_segments(&z[0], &z[1], &plan);
    (sh, z)
}

/// The multiple-shooting iterates of the README's example: at each Newton iterate, the node values and each
/// segment's Taylor path from its source node, sampled within its steps, on every lane; and max |F| per iterate.
fn layout(dir: &str) {
    let (sh, z0) = usage();
    let mut full = z0.clone();
    let rep = sh.newton::<Double>(&mut full, &NewtonOpts::default()).unwrap();
    let iterates = rep.fnorm.len();
    let mut w = Work::<DV, 1>::new(2, 0, sh.kt);
    let mut frames = Vec::new();
    for it in 0..iterates {
        let mut z = z0.clone();
        if it > 0 {
            sh.newton::<Double>(
                &mut z,
                &NewtonOpts {
                    max_iter: it,
                    ..Default::default()
                },
            )
            .unwrap();
        }
        let state = |i: usize| -> State<DV, 1> {
            let o = sh.off_node(i);
            vec![
                Fl::load(&z[0][o..o + 8], &z[1][o..o + 8]),
                Fl::load(&z[0][o + 8..o + 16], &z[1][o + 8..o + 16]),
            ]
        };
        let nodes: Vec<Vec<f64>> = (0..sh.k())
            .map(|i| (0..8).map(|j| state(i)[0].get(j).hi).collect())
            .collect();
        let mut paths = Vec::new();
        for b in 1..sh.k() {
            let xs = &sh.segs[b - 1];
            let mut s = state(sh.seg_source(b));
            let (mut px, mut py) = (vec![xs[0]], vec![(0..8).map(|j| s[0].get(j).hi).collect::<Vec<f64>>()]);
            for q in 0..xs.len() - 1 {
                for t in 1..=8 {
                    let x = if t == 8 {
                        xs[q + 1]
                    } else {
                        xs[q] + (xs[q + 1] - xs[q]) * t as f64 / 8.0
                    };
                    let st = taylor_step(&sh.eq, &s, xs[q], x, &[], &mut w);
                    px.push(x);
                    py.push((0..8).map(|j| st[0].get(j).hi).collect());
                }
                s = taylor_step(&sh.eq, &s, xs[q], xs[q + 1], &[], &mut w);
            }
            // by lane: py[lane][sample]
            let by_lane: Vec<Vec<f64>> = (0..8).map(|j| py.iter().map(|v| v[j]).collect()).collect();
            paths.push(obj(&[("x", arr(px)), ("y", arr2(&by_lane))]));
        }
        frames.push(obj(&[
            ("nodes", arr2(&nodes)),
            ("paths", format!("[{}]", paths.join(","))),
        ]));
    }
    let mut zq = widen(&full, 4);
    let repq = sh
        .newton::<Quad>(
            &mut zq,
            &NewtonOpts {
                tol: 1e-60,
                ..Default::default()
            },
        )
        .unwrap();
    let exact: Vec<Vec<f64>> = (0..=40)
        .map(|i| {
            let x = i as f64 / 40.0;
            (0..8).map(|j| 4.0 / (1.0 + j as f64 / 8.0 + x).powi(2)).collect()
        })
        .collect();
    save(
        dir,
        "layout",
        obj(&[
            ("node_x", arr(sh.nodes.iter().copied())),
            ("segs", arr2(&sh.segs)),
            ("fnorm", arr(rep.fnorm.iter().copied())),
            ("fnorm_qd", num(*repq.fnorm.last().unwrap())),
            ("frames", format!("[{}]", frames.join(","))),
            ("exact", arr2(&exact)),
            ("len", sh.len().to_string()),
            ("rows", sh.rows().to_string()),
        ]),
    );
    println!(
        "layout: max |F| {:?}, quad-double {:.1e}",
        rep.fnorm,
        repq.fnorm.last().unwrap()
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Taylor steps

fn rotor_z(sh: &Shooting<Rotor<1>, 1>) -> Vec<Vec<f64>> {
    let (mut zh, mut zl) = (vec![0.0; sh.len()], vec![0.0; sh.len()]);
    for (i, &x) in sh.nodes.iter().enumerate() {
        let o = sh.off_node(i);
        for (f, fl) in sh.eq.exact(x).iter().enumerate() {
            fl.store(&mut zh[o + f * 8..o + (f + 1) * 8], &mut zl[o + f * 8..o + (f + 1) * 8]);
        }
    }
    vec![zh, zl]
}

/// The planner on `Rotor`, whose series about x0 has radius 1 + x0: the step points of one path, and the cost of the
/// layout's segments at every order against the a-priori order and the order choice.
fn taylor(dir: &str) {
    let eq = Rotor::<1>::new();
    let plan = Plan {
        kt: 20,
        tol: 1e-32,
        hmax: 100.0,
    };
    let (xa, xb) = (-0.9, 3.0);
    let xs = plan_path(&eq, &eq.exact(xa), xa, xb, &[], &plan);

    let (tol, hmax) = (1e-32, 2.0);
    let mut sh = Shooting::new(Rotor::<1>::new(), vec![0.0, 0.5, 1.0, 1.5, 2.0], 2);
    let z = rotor_z(&sh);
    let (mut ks, mut steps, mut cost) = (Vec::new(), Vec::new(), Vec::new());
    // below order 8 a step at 1e-32 is under 1e-4 long, and the segments take tens of thousands of steps
    for k in 8..=64 {
        sh.plan_segments(&z[0], &z[1], &Plan { kt: k, tol, hmax });
        let n: usize = sh.segs.iter().map(|s| s.len() - 1).sum();
        ks.push(k);
        steps.push(n);
        cost.push(n as f64 * step_cost(k));
    }
    let first = sh.plan_segments_auto(&z[0], &z[1], tol, hmax, &Plan::candidate_orders(tol), step_cost);
    let second = sh.plan_segments_auto(&z[0], &z[1], tol, hmax, &first.refined_orders(), step_cost);
    let cands: Vec<usize> = first.candidates.iter().map(|c| c.kt).collect();
    save(
        dir,
        "taylor",
        obj(&[
            ("path_x", arr(xs.iter().copied())),
            ("path_kt", plan.kt.to_string()),
            ("path_tol", num(plan.tol)),
            ("orders", ints(ks)),
            ("steps", ints(steps)),
            ("cost", arr(cost)),
            ("tol", num(tol)),
            ("a_priori", Plan::optimal_kt(tol).to_string()),
            ("candidates", ints(cands)),
            ("first", first.kt.to_string()),
            ("second", second.kt.to_string()),
        ]),
    );
    println!(
        "taylor: {} steps on [{xa}, {xb}]; a priori {}, first pass {}, second pass {}",
        xs.len() - 1,
        Plan::optimal_kt(tol),
        first.kt,
        second.kt
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Residual and derivatives

/// The planted problem's Jacobian at z* perturbed by 1e-2: every nonzero entry's log10 |J|, each row's block, and per
/// column the block evaluations that form it.
fn jacobian(dir: &str) {
    let (sh, zs) = Planted::<1>::shooting(&[0.0, 0.5, 1.0, 1.5, 2.0], 30, 1e-32);
    let z = perturbed(&zs, 1e-2);
    let (c, ld) = sh.jacobian::<Double>(&parts(&z), false).unwrap();
    let (n, rows) = (sh.len(), sh.rows());
    let mut ent = Vec::new();
    for j in 0..n {
        for i in 0..rows {
            let v = c[0][j * ld + i];
            if v != 0.0 {
                ent.push(format!("[{i},{j},{}]", num(v.abs().log10())));
            }
        }
    }
    let row_block: Vec<usize> = (0..rows)
        .map(|r| {
            if r >= sh.row_extra() {
                sh.k() + 1
            } else {
                (0..=sh.k()).rev().find(|&b| sh.row_block(b) <= r).unwrap()
            }
        })
        .collect();
    let m = 8;
    let mut evals = vec![0usize; n];
    for b in 0..=sh.k() {
        for o in sh.block_inputs(b) {
            for e in &mut evals[o..o + m] {
                *e += 1;
            }
        }
        for e in &mut evals[sh.off_global()..n] {
            *e += 1;
        }
    }
    let col_group: Vec<usize> = (0..n)
        .map(|j| {
            if j < sh.off_node(0) {
                0
            } else if j < sh.off_right() {
                1 + (j - sh.off_node(0)) / (2 * m)
            } else if j < sh.off_global() {
                sh.k() + 1
            } else {
                sh.k() + 2
            }
        })
        .collect();
    save(
        dir,
        "jacobian",
        obj(&[
            ("n", n.to_string()),
            ("rows", rows.to_string()),
            ("k", sh.k().to_string()),
            ("entries", format!("[{}]", ent.join(","))),
            ("row_block", ints(row_block)),
            ("col_group", ints(col_group)),
            ("evals", ints(evals.iter().copied())),
            ("jvp_evals", (sh.k() + 1).to_string()),
        ]),
    );
    println!(
        "jacobian: {n} x {rows}, {} nonzeros, {} block evaluations, J v {}",
        ent.len(),
        evals.iter().sum::<usize>(),
        sh.k() + 1
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Bratu's problem u'' + a lambda e^u = 0, u(0) = u(1) = 0, a = 1 / (j + 1) on lane j (tests/family.rs)

fn bratu_a() -> DV {
    let a: Vec<DD> = (0..8).map(|j| DD::ONE / DD::from_f64((j + 1) as f64)).collect();
    let (hi, lo) = split(&a);
    Fl::<DV, 1>::load(&hi, &lo).0[0]
}

/// u = p = 0, E = 1 on a layout of Bratu's problem
fn bratu_zero<E: Equations<1>>(sh: &Shooting<E, 1>) -> Vec<Vec<f64>> {
    let mut zh = vec![0.0; sh.len()];
    for i in 0..sh.k() {
        let o = sh.off_node(i) + 16;
        zh[o..o + 8].fill(1.0);
    }
    let o = sh.off_right() + 8;
    zh[o..o + 8].fill(1.0);
    vec![zh, vec![0.0; sh.len()]]
}

/// lambda a global, fixed by an extra row
struct Bratu {
    a: DV,
}

impl Equations<1> for Bratu {
    const NF: usize = 3;
    const NL: usize = 1;
    const NR: usize = 2;
    const NG: usize = 1;
    const POINTWISE: bool = true;
    type Prep = ();

    fn taylor_coeffs<T: Lane>(&self, _x0: f64, g: &[T], w: &mut Work<T, 1>) {
        for k in 0..w.kt {
            let ik = T::splat(DD::ONE) / T::splat(DD::from_f64((k + 1) as f64));
            let pe = cs(&w.c[1], &w.c[2], k);
            let ae = Fl([w.c[2][k].0[0].mulc(self.a)]);
            w.c[0][k + 1] = w.c[1][k].mul_t(ik);
            w.c[1][k + 1] = -ae.mul_t(g[0]).mul_t(ik);
            w.c[2][k + 1] = pe.mul_t(ik);
        }
    }

    fn left_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], _g: &[T], _prep: &()) -> State<T, 1> {
        vec![Fl::zero(), data[0], Fl::splat(DD::ONE)]
    }

    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], _g: &[T], _prep: &()) -> State<T, 1> {
        vec![Fl::zero(), data[0], data[1]]
    }
}

/// lambda the equations' parameter, `lane::Param::var(lam)`: c on value and dual lanes, c + h on a jet
struct BratuParam {
    a: DV,
    lam: DD,
}

impl Equations<1> for BratuParam {
    const NF: usize = 3;
    const NL: usize = 1;
    const NR: usize = 2;
    const POINTWISE: bool = true;
    type Prep = ();

    fn taylor_coeffs<T: Lane>(&self, _x0: f64, _g: &[T], w: &mut Work<T, 1>) {
        let lam = <T::P as Param>::var(self.lam);
        for k in 0..w.kt {
            let ik = T::splat(DD::ONE) / T::splat(DD::from_f64((k + 1) as f64));
            let pe = cs(&w.c[1], &w.c[2], k);
            let ae = Fl([w.c[2][k].0[0].mulc(self.a)]).mulp(lam);
            w.c[0][k + 1] = w.c[1][k].mul_t(ik);
            w.c[1][k + 1] = -ae.mul_t(ik);
            w.c[2][k + 1] = pe.mul_t(ik);
        }
    }

    fn left_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], _g: &[T], _prep: &()) -> State<T, 1> {
        vec![Fl::zero(), data[0], Fl::splat(DD::ONE)]
    }

    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], _g: &[T], _prep: &()) -> State<T, 1> {
        vec![Fl::zero(), data[0], data[1]]
    }
}

const BRATU_NODES: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

fn bratu_plan(kt: usize) -> Plan {
    Plan {
        kt,
        tol: 1e-34,
        hmax: 1.0 / 16.0,
    }
}

/// The branch of Bratu's problem in its parameter: the solution at lambda = 2 and the series of its branch in
/// h = lambda - 2 to order 30 (multisolver's asymptotic numerical method on `residual_jet`), lane 0's u'(0).
fn series_fig(dir: &str) {
    let lam0 = 2.0;
    let mut sh = Shooting::new(
        BratuParam {
            a: bratu_a(),
            lam: DD::from_f64(lam0),
        },
        BRATU_NODES.to_vec(),
        36,
    );
    let mut z = bratu_zero(&sh);
    sh.plan_segments(&z[0].clone(), &z[1].clone(), &bratu_plan(36));
    let rep = sh.newton::<Double>(&mut z, &opts(1e-28, 20)).unwrap();
    assert!(rep.converged, "{:?}", rep.fnorm);
    let order = 30;
    let br = series::<Double, _>(&sh, &parts(&z), order, 1, false).unwrap();
    let coef: Vec<f64> = std::iter::once(z[0][0])
        .chain((1..=order).map(|k| br.z[k][0].hi))
        .collect();
    save(
        dir,
        "series",
        obj(&[
            ("lam0", num(lam0)),
            ("coef", arr(coef.iter().copied())),
            ("residual", arr(br.residual.iter().copied())),
        ]),
    );
    println!("series: u'(0) = {:.6}, coefficients {:?}", coef[0], &coef[1..5]);
}

/// Pseudo-arclength continuation of Bratu's branch from u = 0 at lambda = 0 through its fold, with the exact
/// bordered LU (multisolver's `exact`) on the family `in_global(0, 0)`, and the fold located.
fn continuation_fig(dir: &str) {
    let mut sh = Shooting::new(Bratu { a: bratu_a() }, BRATU_NODES.to_vec(), 36);
    let og = sh.off_global();
    sh.extra.push(ExtraRow {
        terms: vec![(og, DD::ONE)],
        rhs: DD::ZERO,
    });
    let z = bratu_zero(&sh);
    sh.plan_segments(&z[0], &z[1], &bratu_plan(36));
    let fam = sh.in_global(0, 0).unwrap();
    let (pts, fold, its) = follow(&fam, &z, |b, w| exact::<Double, _>(b, w, false));
    save(
        dir,
        "continuation",
        obj(&[
            ("lam", arr(pts.iter().map(|p| p.0))),
            ("dudx", arr(pts.iter().map(|p| p.1))),
            ("tp", arr(pts.iter().map(|p| p.2))),
            ("newton", ints(pts.iter().map(|p| p.3))),
            ("fold", arr([fold.0, fold.1])),
            ("fold_err", num(fold.2)),
            ("fold_its", its.to_string()),
        ]),
    );
    println!(
        "continuation: {} points, fold at lambda {:.15} ({its} corrections)",
        pts.len(),
        fold.0
    );
}

/// lambda_c, the fold of Bratu's problem on [0, 1], to double-double (tests/family.rs)
const LAMBDA_C: [f64; 2] = [3.513830719125161, 1.566482374576037e-19];

type Pt = (f64, f64, f64, usize);

fn follow<'a, F: Factored<Double>>(
    fam: &'a InGlobal<'a, Bratu, 1>,
    z: &[Vec<f64>],
    factor: impl Fn(&Bordered<'_, Double, InGlobal<'a, Bratu, 1>>, &[&[f64]]) -> Result<F, String>,
) -> (Vec<Pt>, (f64, f64, f64), usize) {
    let (zf, p) = fam.family_z::<Double>(&parts(z));
    let aopts = ArcOpts {
        ds: 0.25,
        ds_min: 1e-6,
        ds_max: 1.0,
        max_points: 200,
        newton: opts(1e-28, 30),
        ..Default::default()
    };
    let st = start::<Double, _, F>(fam, &parts(&zf), p, 1.0, false, &factor).unwrap();
    let path = continuation::<Double, _, F>(fam, st, &aopts, &factor, |pt| pt.z[0][0] > 12.0).unwrap();
    let pts: Vec<Pt> = path
        .points
        .iter()
        .enumerate()
        .map(|(i, pt)| {
            let its = if i == 0 { 0 } else { path.reports[i - 1].fnorm.len() - 1 };
            (pt.p.hi, pt.z[0][0], pt.tp().hi, its)
        })
        .collect();
    let i = *path.folds.first().expect("no fold");
    let fopts = FoldOpts {
        tol: 1e-15,
        max_iter: 50,
    };
    let (pt, its) =
        locate_fold::<Double, _, F>(fam, &path.points[i], path.ds[i], &fopts, &aopts.newton, &factor).unwrap();
    let err = (pt.p - DD::new(LAMBDA_C[0], LAMBDA_C[1])).hi.abs();
    (pts, (pt.p.hi, pt.z[0][0], err), its)
}

// ---------------------------------------------------------------------------------------------------------------
// Mode matrix: the linear wave of tests/mode.rs, det M(lambda) = +-prod_k cos(L (lambda + i omega k)) cos(L (lambda -
// i omega k)), k odd < M, L = 2.5, omega = 0.5

struct Wave {
    d: DMat<1>,
}

const X_END: f64 = 2.5;
const END_STEPS: [f64; 5] = [2.5, 2.375, 2.25, 2.125, 2.0];
const OMEGA: f64 = 0.5;

impl Equations<1> for Wave {
    const NF: usize = 2;
    const NL: usize = 1;
    const NR: usize = 1;
    const NG: usize = 1;
    type Prep = Rate;

    fn prepare_rate(&self, _zh: &[f64], _zl: &[f64], rate: Rate) -> Rate {
        rate
    }

    fn taylor_coeffs<T: Lane>(&self, _x0: f64, g: &[T], w: &mut Work<T, 1>) {
        for k in 0..w.kt {
            let ik = DD::ONE / DD::from_f64((k + 1) as f64);
            let ut = w.rate.shift(&w.c[0][k], self.d.apply(&w.c[0][k]).mul_t(g[0]));
            let vt = w.rate.shift(&w.c[1][k], self.d.apply(&w.c[1][k]).mul_t(g[0]));
            w.c[0][k + 1] = (-vt).muls(ik);
            w.c[1][k + 1] = ut.muls(ik);
        }
    }

    fn left_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], _g: &[T], _prep: &Rate) -> State<T, 1> {
        vec![Fl::zero(), data[0]]
    }

    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, 1>], g: &[T], prep: &Rate) -> State<T, 1> {
        let mut w = Work::<T, 1>::new(2, 0, 30);
        w.rate = *prep;
        taylor_path(self, &[data[0], Fl::zero()], &END_STEPS, g, &mut w)
    }
}

/// ln|det M| of the wave along a line of complex rates through three modes, the winding number along a rectangle about two of its modes
/// with every adaptive sample, and one mode by Muller's method.
fn mode(dir: &str) {
    let nodes = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    let mut sh = Shooting::new(
        Wave {
            d: dmat::<1>(Parity::Odd),
        },
        nodes.clone(),
        30,
    );
    sh.segs = nodes
        .windows(2)
        .map(|w| (0..=4).map(|i| w[0] + 0.125 * i as f64).collect())
        .collect();
    let og = sh.off_global();
    sh.extra.push(ExtraRow {
        terms: vec![(og, DD::ONE)],
        rhs: DD::from_f64(OMEGA),
    });
    let mut z = vec![(0..sh.len()).map(wave).collect::<Vec<f64>>(), vec![0.0; sh.len()]];
    z[0][og] = OMEGA;

    // the line Im lambda = 0.5 through three modes, pi / 2L (2n + 1) + 0.5 i
    let (re0, re1, im, nr) = (-1.5, 2.5, 0.5, 801);
    let rates: Vec<Rate> = (0..nr)
        .map(|a| {
            Rate::new(
                DD::from_f64(re0 + (re1 - re0) * a as f64 / (nr - 1) as f64),
                DD::from_f64(im),
            )
        })
        .collect();
    let ld: Vec<f64> = sh
        .mode_logdets(&parts(&z), &rates, true)
        .into_iter()
        .map(|r| r.unwrap().0)
        .collect();
    let rect = [(0.2, -0.9), (1.1, -0.9), (1.1, 0.9), (0.2, 0.9)];
    let wnd = sh.mode_winding(&parts(&z), &rect, &WindOpts::default(), true).unwrap();
    let mut samples = Vec::new();
    for e in &wnd.edges {
        for (q, &t) in e.t.iter().enumerate() {
            if q + 1 == e.t.len() {
                continue; // the next edge's first sample
            }
            let re = e.from.0 + t * (e.to.0 - e.from.0);
            let im = e.from.1 + t * (e.to.1 - e.from.1);
            samples.push(format!("[{},{},{}]", num(re), num(im), num(e.val[q].1)));
        }
    }
    let zero = sh.mode_zero(&parts(&z), (0.7, 0.42), 0.05, 1e-13, true).unwrap();
    save(
        dir,
        "mode",
        obj(&[
            ("re", arr([re0, re1])),
            ("im", num(im)),
            ("nr", nr.to_string()),
            ("logdet", arr(ld)),
            (
                "rect",
                format!(
                    "[{}]",
                    rect.iter().map(|c| arr([c.0, c.1])).collect::<Vec<_>>().join(",")
                ),
            ),
            ("samples", format!("[{}]", samples.join(","))),
            ("turns", num(wnd.turns)),
            ("count", wnd.count().to_string()),
            ("evals", wnd.evals.to_string()),
            ("zero", arr([zero.0, zero.1])),
            ("n", (sh.len() - 1).to_string()),
            ("L", num(X_END)),
            ("omega", num(OMEGA)),
        ]),
    );
    println!(
        "mode: {} rates, winding {:.9} ({} evaluations), zero {zero:?}",
        nr, wnd.turns, wnd.evals
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Periodic direction and relayout

/// An odd field on the half grid of 8 points moved to 64 by the trigonometric resampler, and the planted problem's
/// segments of about 60 steps split at 20.
fn periodic(dir: &str) {
    let (m, m2) = (8, 64);
    let f = |m: usize, j: usize| -> DD {
        let (s1, _) = sin_cos_grid(Parity::Odd, m, j, 1);
        let (_, c3) = sin_cos_grid(Parity::Odd, m, j, 3);
        let (s5, _) = sin_cos_grid(Parity::Odd, m, j, 5);
        s1 + c3.mul_f(0.3) - s5.mul_f(0.2)
    };
    let f8: Vec<DD> = (0..m).map(|j| f(m, j)).collect();
    let f64s = Resampler::new(Parity::Odd, m, m2).apply(&f8);
    let err = (0..m2).map(|j| (f64s[j] - f(m2, j)).hi.abs()).fold(0.0, f64::max);
    let th = |m: usize| arr((0..m).map(|j| theta(Parity::Odd, m, j).hi));

    let (old, zs) = Planted::<1>::shooting(&[0.0, 1.0, 2.0], 16, 1e-30);
    let z = perturbed(&zs, 1e-3);
    let mut new = Planted::<1>::shooting(&[0.0, 1.0, 2.0], 16, 1e-30).0;
    let zn = new.split_segments::<Double>(&parts(&z), 20, false).unwrap();
    let f_old = old.residual::<Double>(&parts(&z), false).unwrap();
    let f_new = new.residual::<Double>(&parts(&zn), false).unwrap();
    let steps = |sh: &Shooting<Planted<1>, 1>| ints(sh.segs.iter().map(|s| s.len() - 1));
    let ends = |sh: &Shooting<Planted<1>, 1>| {
        arr2(
            &sh.segs
                .iter()
                .map(|s| vec![s[0], *s.last().unwrap()])
                .collect::<Vec<_>>(),
        )
    };
    save(
        dir,
        "periodic",
        obj(&[
            ("theta8", th(m)),
            ("f8", arr(f8.iter().map(|v| v.hi))),
            ("theta64", th(m2)),
            ("f64", arr(f64s.iter().map(|v| v.hi))),
            ("resample_err", num(err)),
            ("old_steps", steps(&old)),
            ("new_steps", steps(&new)),
            ("old_ends", ends(&old)),
            ("new_ends", ends(&new)),
            ("old_rows", f_old.len().to_string()),
            ("new_rows", f_new.len().to_string()),
        ]),
    );
    println!(
        "periodic: resample error {err:.1e}; segments {:?} -> {:?}",
        old.segs.iter().map(|s| s.len() - 1).collect::<Vec<_>>(),
        new.segs.iter().map(|s| s.len() - 1).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Fast path

/// a solve from z: (max |F| per step, Jacobians factored)
type Solve<'a> = dyn Fn(&mut Vec<Vec<f64>>) -> (Vec<f64>, usize) + 'a;

/// Newton's method on the planted problem (B = 2) from z* perturbed by 1e-3 in double-double: the exact Jacobian,
/// the f64 Jacobian dense and in staircase storage, and Broyden's updates between f64 factorizations; max |F| per
/// step and the median time of a solve.
fn fast(dir: &str) {
    let (sh, zs) = Planted::<2>::shooting(&[0.0, 0.5, 1.0, 1.5, 2.0], 30, 1e-32);
    let z0 = perturbed(&zs, 1e-3);
    let tol = 1e-28;
    let mut runs = Vec::new();
    let mut run = |name: &str, f: &Solve<'_>| {
        let mut z = z0.clone();
        let (fnorm, facts) = f(&mut z);
        let ms = median_ms(5, || {
            let mut z = z0.clone();
            f(&mut z);
        });
        println!(
            "fast: {name}: {} steps, {facts} factorizations, {ms:.1} ms",
            fnorm.len() - 1
        );
        runs.push(obj(&[
            ("name", format!("\"{name}\"")),
            ("fnorm", arr(fnorm.iter().copied())),
            ("factorizations", facts.to_string()),
            ("ms", num(ms)),
        ]));
    };
    run("newton", &|z| {
        let r = sh.newton::<Double>(z, &opts(tol, 20)).unwrap();
        (r.fnorm.clone(), r.fnorm.len() - 1)
    });
    run("newton_mixed, dense", &|z| {
        let r = sh
            .newton_mixed::<Double, Fu>(z, &opts(tol, 20), Storage::Dense)
            .unwrap();
        (r.fnorm.clone(), r.fnorm.len() - 1)
    });
    run("newton_mixed, staircase", &|z| {
        let r = sh
            .newton_mixed::<Double, Dtn<4>>(z, &opts(tol, 20), Storage::Staircase)
            .unwrap();
        (r.fnorm.clone(), r.fnorm.len() - 1)
    });
    run("newton_quasi_mixed", &|z| {
        let q = QuasiOpts {
            max_updates: 20,
            ..Default::default()
        };
        let (r, info) = sh
            .newton_quasi_mixed::<Double, Fu>(z, &opts(tol, 20), &q, Storage::Dense)
            .unwrap();
        (r.fnorm.clone(), info.factorizations)
    });
    save(
        dir,
        "fast",
        obj(&[
            ("n", sh.len().to_string()),
            ("tol", num(tol)),
            ("runs", format!("[{}]", runs.join(","))),
        ]),
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Backends

/// FNV-1a of every part of the README example's double- and quad-double roots, on the backend this build selects
fn backend(dir: &str) {
    let (sh, mut z) = usage();
    sh.newton::<Double>(&mut z, &NewtonOpts::default()).unwrap();
    let mut zq = widen(&z, 4);
    sh.newton::<Quad>(
        &mut zq,
        &NewtonOpts {
            tol: 1e-60,
            ..Default::default()
        },
    )
    .unwrap();
    let mut h = Fnv::new();
    for v in z.iter().chain(&zq).flatten() {
        h.f64(*v);
    }
    // the backend multidouble selected for this build, by its own name
    let name = match Native::NAME {
        "neon" => "NEON",
        "sse2" => "SSE2",
        "avx2" => "AVX2",
        "avx512" => "AVX-512",
        other => other,
    };
    let arch = std::env::consts::ARCH;
    save(
        dir,
        &format!("backend-{}", Native::NAME),
        obj(&[
            ("backend", format!("\"{name}\"")),
            ("arch", format!("\"{arch}\"")),
            ("hash", format!("\"{:016x}\"", h.0)),
        ]),
    );
    println!("backend {name} ({arch}): {:016x}", h.0);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "docs/figures/data".into());
    let only: Vec<String> = args.collect();
    let want = |n: &str| only.is_empty() || only.iter().any(|o| o == n);
    type Figure = (&'static str, fn(&str));
    let figs: [Figure; 9] = [
        ("backend", backend),
        ("layout", layout),
        ("taylor", taylor),
        ("jacobian", jacobian),
        ("series", series_fig),
        ("continuation", continuation_fig),
        ("mode", mode),
        ("periodic", periodic),
        ("fast", fast),
    ];
    for (name, f) in figs {
        if want(name) {
            f(&dir);
        }
    }
}
