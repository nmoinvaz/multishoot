- Every check must be able to fail: each new accuracy check comes with a control that feeds it a wrong result
  (behind the `mutate` feature, or a test that asserts the failure), and it is run once to see it fire.
- Planted truth first: the engine is tested on problems with a known solution (an analytic boundary-value problem, a
  manufactured solution) through the same code path the real problems use, with the pass criteria written in the
  test's doc comment before the first run.
- Results must not depend on multidouble's SIMD backend: run `cargo test --release` on the default build, with
  `--features portable`, with `-C target-feature=+avx2,+fma`, and on aarch64 (NEON). Comparisons across backends are
  bitwise.
- The engine knows nothing of any one problem. Equations, endpoint series and constants of a physical model belong to
  the crate that uses the engine; tests here use model problems.
- `cargo fmt` (max width 120) and `cargo clippy --all-targets --features mutate -- -D warnings` (and once with
  `--features portable,mutate`) before committing.
- Commit messages: imperative, a lowercase module prefix where one applies (`taylor: ...`, `newton: ...`), 1-3
  sentences of body saying what changed and why, no attribution trailers.
