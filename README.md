# multishoot

[![Build](https://github.com/nmoinvaz/multishoot/actions/workflows/build.yml/badge.svg)](https://github.com/nmoinvaz/multishoot/actions/workflows/build.yml)
[![Lint](https://github.com/nmoinvaz/multishoot/actions/workflows/lint.yml/badge.svg)](https://github.com/nmoinvaz/multishoot/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/nmoinvaz/multishoot/graph/badge.svg)](https://codecov.io/gh/nmoinvaz/multishoot)
[![crates.io](https://img.shields.io/crates/v/multishoot.svg)](https://crates.io/crates/multishoot)
[![docs.rs](https://docs.rs/multishoot/badge.svg)](https://docs.rs/multishoot)

Multiple shooting for boundary-value problems, in double-, quad- and octo-double precision.

A boundary-value problem asks for the solution of differential equations that meets conditions at two or more places.
Multiple shooting splits the interval at nodes, integrates the equations across each piece, and lets Newton's method
adjust the values at the nodes, and any unknown parameters, until every piece ends where the next one begins.

This crate is the engine of that method, generic over the equations, on the arithmetic of
[multidouble](https://github.com/nmoinvaz/multidouble):

| Part | What it does |
|---|---|
| equations trait | the problem enters as the Taylor coefficients of its right-hand side, and the series that start the solution at singular endpoints |
| Taylor stepping | the solution carried between nodes by Taylor series, each step planned so that its truncation term stays below a tolerance |
| periodic direction | fields that depend on a periodic variable, on collocation points, with exact derivatives in it |
| layout | node blocks, mismatch rows, global parameters (an eigenvalue), a phase condition, pinned rows, projected rows (a field's high harmonics unmatched and held at zero elsewhere, exact in every precision) |
| exact Jacobian | forward-mode dual numbers over the code that forms the residual |
| Jacobian-vector product | J v by one dual pass per block, without forming the Jacobian, in every precision |
| mode matrix | the linearization for perturbations growing as exp(λt) at a complex rate λ, and its determinant, whose winding number counts the modes |
| Newton | in double-double, quad-double or octo-double (multidouble's `Precision` trait), with the residual's Taylor series in a parameter of the equations; written over a `Problem` trait (the residual and its exact Jacobian), which the layout implements and any other system can |
| fast path | the Jacobian in f64 from lanes with f64 tangents, several columns per evaluation, dense or in staircase storage, and the mixed-precision Newton on it |

**Status:** early. The API may change between minor versions until 1.0. It depends on multidouble from crates.io,
whose `Precision` trait (double-, quad- and octo-double) it is written over.

## Building

```
cargo test --release
cargo test --release --features portable
cargo clippy --all-targets --features mutate -- -D warnings
```

## License

BSD 3-Clause, see [LICENSE](LICENSE).
