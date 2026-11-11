# multishoot

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
| layout | node blocks, mismatch rows, global parameters (an eigenvalue), a phase condition, pinned rows |
| exact Jacobian | forward-mode dual numbers over the code that forms the residual |
| Newton | in double-double, quad-double or octo-double (multidouble's `Precision` trait), with the residual's Taylor series in a parameter of the equations |

It comes out of the solver in [choptuik-large-d](https://github.com/nmoinvaz/choptuik-large-d)'s `hp/` crate, which
computes Choptuik's critical solution in D dimensions and keeps its equations there.

**Status:** under construction. Not published; it depends on multidouble's `Precision` trait, which is not yet
released.

## Building

```
cargo test --release
cargo test --release --features portable
cargo clippy --all-targets --features mutate -- -D warnings
```

## License

BSD 3-Clause, see [LICENSE](LICENSE).
