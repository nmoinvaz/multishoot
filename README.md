# multishoot

[![Build](https://github.com/nmoinvaz/multishoot/actions/workflows/build.yml/badge.svg)](https://github.com/nmoinvaz/multishoot/actions/workflows/build.yml)
[![Lint](https://github.com/nmoinvaz/multishoot/actions/workflows/lint.yml/badge.svg)](https://github.com/nmoinvaz/multishoot/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/nmoinvaz/multishoot/graph/badge.svg)](https://codecov.io/gh/nmoinvaz/multishoot)
[![crates.io](https://img.shields.io/crates/v/multishoot.svg)](https://crates.io/crates/multishoot)
[![docs.rs](https://docs.rs/multishoot/badge.svg)](https://docs.rs/multishoot)

Multiple shooting for boundary-value problems in double-double, quad-double and octo-double precision, with exact
Jacobians and the same results, bit for bit, on every SIMD backend.

A boundary-value problem asks for the solution of differential equations that meets conditions at two or more places.
Multiple shooting splits the interval at nodes, carries the solution across each piece by Taylor series, and lets
Newton's method adjust the values at the nodes, and any unknown parameters, until every piece ends where the next one
begins.

This crate is the engine of that method, generic over the equations. The equations are written once over a lane type
of [multidouble](https://github.com/nmoinvaz/multidouble), and the engine runs that one code with values (the
residual), dual numbers (the exact Jacobian and its product with a vector), truncated power series (the residual's
series in a parameter) and complex tangents (the mode matrix). Newton's method and the other solvers of F(z) = 0 are
the crate [multisolver](https://github.com/nmoinvaz/multisolver).

## API

The crate root re-exports what most code needs: `Equations`, `State`, `Work`, `Shooting`, `ExtraRow`, `Plan`, `Rate`,
`ModeMatrix`, `ProjectedRows`, `Parities`, `MixedLu` and `Storage`; multidouble's `DD`, `Fl`, `cs`, `cs_range`, `cs_w`,
`horner`, `Precision`, `Double`, `Quad`, `Octo` and `Native`; multisolver's `Problem`, `Jvp` and `ResidualJet`; and both
crates whole, as `multishoot::multidouble` and `multishoot::multisolver`.

A state z in the precision P is held as `P::PARTS` arrays of f64, 2 for `Double`, 4 for `Quad` and 8 for `Octo`, with
part q of unknown i at `z[q][i]`. multisolver's `prec::parts` borrows a state's slices, `prec::widen` carries it to more
parts, and `prec::max_abs` gives max |F| of a residual.

### The problem

| Item | What it does |
|---|---|
| `Equations<B>` | the problem: its sizes, the Taylor recursion of its right-hand side, and the maps from the endpoint data to the states at the end nodes |
| `Equations::taylor_coeffs` | Taylor coefficients 1..=kt of every field about x0, from order 0 |
| `Equations::left_state`, `right_state` | the state at the first or last node from that end's data: boundary conditions at a regular end, a series summed to the node at a singular one |
| `Equations::prepare`, `try_prepare` | what the endpoint maps need once per evaluation, such as a series' coefficient table; the engine forms it through `try_prepare` and returns its error |
| `Equations::prepare_rate`, `try_prepare_rate` | the same for a mode-matrix evaluation at a growth rate |
| `Equations::taylor_coeffs_at`, `left_state_at`, `right_state_at` | the same maps for one order k in the parameter, from the orders kept; by default they form every order |
| `State<T, B>` | the `NF` fields at a point, `Vec<Fl<T, B>>` |
| `Work<T, B>` | one step's Taylor coefficients `c[f][k]`, the recursion's scratch series `aux`, the order `kt` and the growth rate of a mode-matrix evaluation |
| `lane::Lane` | the lane types the equations are generic over: multidouble's `DV`, `QV`, `OV` (values), `Du`, `Qu`, `Ou` (one tangent), `Dc` (a complex tangent), `Jet<N>`, `QJet<N>`, `OJet<N>` (series) and `Fu`, `Fun<N>`, `Dtn<N>` (f64 tangents) |

A problem has `NF` fields, each held on M = 8 B collocation points of a periodic variable θ (`Fl<T, B>`, B lane vectors
of 8). `NL` fields of data start the solution at the left end and `NR` at the right. `NG` global unknowns, such as an
eigenvalue or a period, enter every block, and `NAUX` scratch series serve the recursion.

`POINTWISE` says that every method acts on each point alone. A problem without a periodic direction sets it and holds
8 independent problems in its lanes, and one dual evaluation then gives the Jacobian columns of all M points of a field.
`ORDER_BY_ORDER` says that the order-k methods (`taylor_coeffs_at`, `left_state_at`, `right_state_at`) form one order
from the orders kept, so that the residual's series of order N costs about N orders instead of N evaluations of
growing length.

Constants enter in double-double (`Fl::cst`, `Fl::muls`). A parameter that the residual is expanded in enters through
the lane's parameter arithmetic (`lane::Param::var`): the constant c on value and dual lanes, the series c + h on a jet.

### Taylor steps

| Item | What it does |
|---|---|
| `Plan` | the planner's settings: Taylor order `kt`, tolerance `tol` on the last two terms, largest step `hmax` |
| `taylor::taylor_step`, `taylor_path` | one Taylor step of a state summed by Horner's rule, and a path of steps through given points |
| `taylor::plan_path` | step points between two ends, each step the largest whose terms of orders kt - 1 and kt stay below `tol` |
| `taylor::step_size`, `kth_root` | that step from one step's coefficients, by a k-th root from IEEE operations alone, the same bits on every platform |

A step from x0 to x1 is exact in double-double, as both points are f64. A step that would leave less than 40% of the
rest becomes half the rest, so a segment ends in two near-equal steps. `plan_path` panics on a coefficient that is not
finite and on a step that does not move x.

### The layout

| Item | What it does |
|---|---|
| `Shooting<E, B>` | the equations, the nodes, each segment's step points, the extra rows, the row factors and the projected rows |
| `Shooting::new`, `plan_segments` | a layout on increasing nodes, and every segment's steps planned from a state z, on rayon tasks |
| `ExtraRow` | a linear condition `sum_i w_i z_i - rhs = 0`: a phase condition, a normalization, a pinned unknown |
| `Shooting::off_node`, `off_right`, `off_global`, `row_block`, `row_extra` | the offsets of the unknowns and of the residual's rows |
| `Shooting::block_inputs`, `block_target`, `block_state` | what block b reads, the node it is matched to, and the state it produces |
| `Shooting::match_node`, `row_scale` | the last segment that runs outward (the later ones run inward), and a factor per residual row |

With K nodes the unknowns are

```text
z = [ left data (NL M) | node 0 (NF M) | ... | node K-1 (NF M) | right data (NR M) | globals (NG) ]
```

and the residual has K + 1 blocks of NF M rows, then the extra rows. Block 0 is the left end's state at node 0 minus
node 0, block b is segment b's Taylor path from its source node minus its target node, and block K is the right end's
state at node K-1 minus node K-1. The system is square when NL + NR = NF and there is one extra row per global.

### Residual and derivatives

| Item | What it does |
|---|---|
| `Shooting::residual::<P>` | F(z) in the precision P, one scalar per row, with each block on its own rayon task under `par` |
| `Shooting::jacobian::<P>` | the exact Jacobian dF/dz by forward-mode duals over the residual's code, as column-major part arrays |
| `Shooting::jacobian_tangents::<P, T>` | `jacobian`, bit for bit, from lanes with several tangents (multidouble's `Oun<N>`): one evaluation of a block for up to N of its column tasks |
| `Shooting::jvp::<P>` | J v and F together, one dual pass per block, without the Jacobian |
| `Shooting::taylor_check::<P>` | the Taylor paths' truncation error per block, estimated from the residual with every step halved |
| `Shooting::newton::<P>`, `newton_step::<P>` | multisolver's damped Newton method and its step on this layout |

The Jacobian takes one evaluation of a block per column, one per input field for a pointwise problem. A product J v
costs K + 1 evaluations, about as many as K + 1 columns, so a Krylov solver needs no matrix.

### Series in a parameter

| Item | What it does |
|---|---|
| `Shooting::residual_jet::<P, N>` | the residual's coefficients in the equations' parameter on the jet state `sum_k z_k h^k`, on jets of N coefficients |
| `Shooting::jet_session::<P, N>`, `jet::JetSession` | the same series one order at a time: `next` takes z_k and returns order k + 1 with z_(k+1) zero, and `order` reads an order formed |
| `JetSession::tangent_from` | from a given order on, each order moved by its tangent once z_k is given instead of formed again, equal to `residual_jet`'s to rounding |

Order 0 is the residual, and with z_1 given order 1 is J z_1 + dF/dparameter. A session keeps each block's state, its
endpoint map's series and each Taylor step's work between orders. With the order-k methods an order costs about one
evaluation at one coefficient; with their defaults a session gives the bits of `residual_jet` at its cost.

### Mode matrix

| Item | What it does |
|---|---|
| `Rate` | the complex growth rate λ of a mode-matrix evaluation, zero otherwise; `Rate::shift` marks a t-derivative in the equations |
| `ModeMatrix` | M(λ), the linearization for perturbations growing as exp(λt), in complex double-double; `logdet` gives ln\|det M\| and arg det M |
| `Shooting::mode_matrix`, `mode_logdet` | M(λ) at a background z from complex tangents (`Dc`), the globals and extra rows held fixed, and its log-determinant |
| `Shooting::mode_logdets` | `mode_logdet` at many rates, whole rates side by side on rayon tasks, each value that rate's bits |
| `Shooting::mode_winding`, `mode_zero` | the winding number of det M(λ) along a polygon by multisolver's argument principle (the modes inside, less any poles), and one mode's rate by Muller's method |

At λ = 0, M is the Jacobian's leading block. det M(λ) vanishes where a mode exp(λt) g solves the linearized problem,
so its winding number along a closed contour counts the modes inside, when det M has no poles there.

### Periodic direction and relayout

| Item | What it does |
|---|---|
| `periodic::Parity` | a field's symmetry under θ → θ + π: odd or even harmonics on the half grid, or all of them on the full grid |
| `periodic::dmat` | d/dθ on the M points of a grid, exact for harmonics below its Nyquist |
| `periodic::theta`, `sin_cos`, `sin_cos_grid`, `sin_cos_pi`, `sin_cos_pi_od` | the grid points, and sines and cosines in double-double, those of rational multiples of π also in octo-double |
| `periodic::spectrum`, `tail`, `Resampler` | a field's Fourier coefficients, its resolution, and its trigonometric interpolant on another number of points |
| `ProjectedRows`, `Shooting::project_rows` | one field's rows of a block replaced by Π (state - target) + Π' y: its high harmonics left unmatched and those of the pinned unknowns y held at zero |
| `Parities` | each field's parity at the left end, at the nodes and at the right end |
| `Shooting::renode`, `resample` | a solution moved to new node positions, or to another number of collocation points, as the start of a Newton solve there |

The projector Π onto the low harmonics is formed in octo-double from cosines of exact multiples of π, and each
precision reads its leading parts, so the projected rows are exact to the precision of the solve.

### Fast path

| Item | What it does |
|---|---|
| `Shooting::jacobian_f64::<T>` | the Jacobian in f64 from lanes with f64 tangents (`Fu`, `Fun<N>`, `Dtn<N>`), several columns per evaluation |
| `Shooting::jacobian_staircase::<T>` | the same Jacobian built block by block and factored in staircase storage (multidouble's `StaircaseLu`), never dense |
| `Storage`, `MixedLu` | dense or staircase storage, and the f64 factors, which solve in double-double |
| `Shooting::newton_mixed::<P, T>`, `newton_step_mixed`, `mixed_lu` | Newton's method with the residual in P and the f64 Jacobian as its matrix |

The mixed Newton method converges to the root of `newton`, linearly, each step contracting by about the f64 Jacobian's
relative error times the conditioning. Staircase storage needs one global and one extra row, whose terms lie on the
left data, the global and node 0.

### Solvers

`Shooting` implements three traits of multisolver, and its solvers run on any problem that does:

| Trait | What it gives | Used by |
|---|---|---|
| `Problem<P>` | `len`, `residual` and `jacobian` | Newton, chord, Levenberg-Marquardt, Newton with weak directions removed |
| `Jvp<P>` | `jvp`, J v without the Jacobian | the Krylov-refined Newton method |
| `ResidualJet<P>` | `residual_jet` on jets of 8, 16 or 32 coefficients, and `jet_orders`: a `JetSession` when `ORDER_BY_ORDER` is set, else multisolver's `Recompute` | the asymptotic numerical method, the exact tangent predictor |

multisolver also holds the series singularities and the argument-principle zero count; its documentation describes
them.

## Installation

Add the crate from [crates.io](https://crates.io/crates/multishoot):

```sh
cargo add multishoot
```

or as a dependency in `Cargo.toml`:

```toml
[dependencies]
multishoot = "3.3"
```

It needs Rust 1.89 or later, as multidouble does. It re-exports multidouble and multisolver, so it needs no other
dependency. The feature `portable` selects multidouble's plain-array backend on every target, and `mutate` adds the
deliberate errors of the tests' controls.

## Usage

The example solves y'' = 3/2 y² on [0, 1] with y(0) = 4 / a² and y(1) = 4 / (1 + a)², whose solution is
y = 4 / (a + x)², for 8 values of a at once. Each lane holds one value of a, so the problem is pointwise. Newton's
method starts in double-double from the line between the end values, and a quad-double solve then refines the root.

```rust
use multishoot::lane::{Lane, DV};
use multishoot::multidouble::dd::split;
use multishoot::multisolver::{prec::widen, NewtonOpts};
use multishoot::{cs, Double, Equations, Fl, Plan, Quad, Shooting, State, Work, DD};

/// y'' = 3/2 y^2 as y' = p, p' = 3/2 y^2, with y given at both ends
struct Model {
    y0: Fl<DV, 1>,
    y1: Fl<DV, 1>,
}

impl Equations<1> for Model {
    const NF: usize = 2; // y, p
    const NL: usize = 1; // p(0)
    const NR: usize = 1; // p(1)
    const POINTWISE: bool = true;
    type Prep = ();

    /// (k + 1) y_(k+1) = p_k, (k + 1) p_(k+1) = 3/2 (y y)_k
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

/// y = 4 / (a + x)^2 at x, a = 1 + j / 8 on lane j
fn exact(x: f64) -> Fl<DV, 1> {
    let v: Vec<DD> = (0..8)
        .map(|j| {
            let s = DD::from_f64(1.0 + j as f64 / 8.0 + x);
            DD::from_f64(4.0) / (s * s)
        })
        .collect();
    let (hi, lo) = split(&v);
    Fl::load(&hi, &lo)
}

fn main() {
    let model = Model {
        y0: exact(0.0),
        y1: exact(1.0),
    };
    let mut sh = Shooting::new(model, vec![0.0, 0.25, 0.5, 0.75, 1.0], 24);

    // z = [p(0) | y, p at each node | p(1)], 8 lanes each, as hi and lo parts; the start is the line between the ends
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

    let rep = sh.newton::<Double>(&mut z, &NewtonOpts::default()).unwrap();
    assert!(rep.converged);
    let err = (0..sh.k())
        .map(|i| {
            let o = sh.off_node(i);
            (Fl::<DV, 1>::load(&z[0][o..o + 8], &z[1][o..o + 8]) - exact(sh.nodes[i])).amax()
        })
        .fold(0.0, f64::max);
    let fnorm: Vec<String> = rep.fnorm.iter().map(|f| format!("{f:.1e}")).collect();
    println!("double-double: max |F| {}; max |y - y*| {err:.1e}", fnorm.join(", "));

    // the same problem in quad-double, from the double-double root
    let mut zq = widen(&z, 4);
    let opts = NewtonOpts {
        tol: 1e-60,
        ..Default::default()
    };
    let rep = sh.newton::<Quad>(&mut zq, &opts).unwrap();
    assert!(rep.converged);
    println!("quad-double: max |F| {:.1e}", rep.fnorm.last().unwrap());
}
```

It prints

```text
double-double: max |F| 5.6e0, 2.8e-1, 2.0e-3, 8.8e-8, 2.0e-16, 1.7e-31; max |y - y*| 3.7e-32
quad-double: max |F| 5.3e-64
```

The steps are planned once, from the start. The quad-double solve finds the root of the same discrete system to
quad-double rounding. Its end values are double-double constants (`Fl::cst`), so that root stays about 1e-32 from y*.

## Accuracy

Three rules govern the tests.

- Every check can fail. Each accuracy check comes with a control that feeds it a wrong result, either a test that
  asserts the failure or a deliberate error in the engine behind the feature `mutate` (`control::Control`, set by
  `Shooting::set_control`), such as a Jacobian column without its block derivative, a residual row matched to the
  wrong unknown, a state read in half its parts, or a mode matrix without its rate. Each control is run once to see
  it fire.
- Planted truth first. The engine is tested on problems with a known solution, an analytic boundary-value problem or
  a manufactured one, through the same code path real problems use. Each test states its pass criterion in its doc
  comment, written before its first run.
- Results do not depend on multidouble's SIMD backend. Fingerprints of the bits of the results (FNV-1a), of the Taylor
  steps and plans, the residual, J v, the jets, the mode matrix and the f64 Jacobians among them, must equal fixed
  references on every backend, so a difference of one bit fails.

Non-finite values never pass a check: max |F| and the planner's coefficient norms propagate NaN, and Newton's method
never reports a NaN residual as converged.

The engine knows nothing of any one problem. The tests use model problems:

| Problem | Known solution | What it checks |
|---|---|---|
| `Rotor` | two odd fields shifted in θ, rotated into each other and decaying, in closed form | Taylor steps, the planner's bound against the true error |
| `Planted` | a manufactured solution: sources that make chosen fields solve the collocation system to rounding | the residual, the Jacobian, Newton in every precision, J v, the fast path, projected rows, relayout |
| `Cylinder` | the lowest mode sin(πx) cos θ / π of an eigenproblem, with eigenvalue π² + μ | two globals with amplitude and phase rows, the series in μ, the asymptotic numerical method |
| `Bessel` | 2^ν ν! J_ν(x) for ν = 0..7, one order per lane | a singular endpoint started from its Frobenius series, the pointwise Jacobian |
| `Wave` | det M(λ) as a product of cosines, for every complex λ | the mode matrix, the winding count of its zeros |
| `Branch` | a solution branch whose series in the parameter is known | the order-by-order series and its tangent update |

Some of the pass criteria:

| Check | Problem | Pass |
|---|---|---|
| Taylor paths against the closed form | `Rotor` | every node within 1e-28 |
| the planner's bound | `Rotor` | every step's error at most `tol`, the largest at least 1e-3 `tol` |
| Jacobian against central differences | `Planted` | every column within 1e-13, relative |
| Newton in double-double | `Planted` | max \|F\| at most 1e-28 in 10 steps, every unknown within 1e-27 |
| Newton in quad-double, octo-double | `Planted` | max \|F\| below 1e-60 and 1e-110 |
| jets across precisions | `Cylinder` | `Jet` against `QJet` within 1e-28, `QJet` against `OJet` within 1e-60 |
| det M(λ) against the closed form | `Wave` | ln\|det\| within 1e-10, arg det within 1e-10 modulo π |

## Building and testing

```sh
cargo test --release                                            # SSE2 on x86-64, NEON on aarch64
cargo test --release --features portable                        # the plain-array backend
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --release   # AVX2
cargo test --release --features mutate                          # the mutation controls
cargo fmt --check
cargo clippy --all-targets --features mutate -- -D warnings
cargo clippy --all-targets --features portable,mutate -- -D warnings
cargo llvm-cov --release --features mutate                      # coverage (cargo-llvm-cov)
```

The tests run in release mode, as the planted problems run Newton's method in quad-double and octo-double. A default
build on aarch64 is the NEON backend, so that run covers it. CI runs the suite on:

- the SSE2, AVX2 and portable backends on x86-64 Linux, and the mutation controls;
- the NEON backend on ARM64 Linux and ARM64 macOS;
- Windows MSVC on x64.

It also runs coverage, rustfmt, clippy on the SSE2, portable and NEON backends, the documentation, and a library
check at the minimum Rust version.

## References

1. R. E. Wengert, "A simple automatic derivative evaluation program", Commun. ACM 7(8), 463–464, 1964,
   [doi:10.1145/355586.364791](https://doi.org/10.1145/355586.364791). Forward-mode duals, the exact Jacobian and J v.
2. C. B. Moler, "Iterative refinement in floating point", J. ACM 14(2), 316–321, 1967,
   [doi:10.1145/321386.321394](https://doi.org/10.1145/321386.321394). The mixed-precision Newton step and the
   refined periodic solves of endpoint maps.
3. L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, Philadelphia, 2000, ch. 3,
   [doi:10.1137/1.9780898719598](https://doi.org/10.1137/1.9780898719598). The derivative matrix of the periodic grid.

## Related crates

- [multidouble](https://github.com/nmoinvaz/multidouble) is the double-, quad- and octo-double arithmetic this crate
  is written over, with the dual numbers that give its exact Jacobians.
- [multisolver](https://github.com/nmoinvaz/multisolver) holds the nonlinear solvers that this crate's layout is
  solved with, from Newton's method to the argument-principle zero count.

## License

BSD 3-Clause, see [LICENSE](LICENSE).
