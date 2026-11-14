//! Solvers of F(z) = 0 written over a [`Problem`], not over multiple shooting: a problem gives its residual and the
//! exact Jacobian in one of multidouble's precisions, and the solvers here need nothing else of it.
//! [`crate::Shooting`] is one such problem.
//!
//! * [`newton`]: the Newton step and the damped Newton loop.
use multidouble::precision::Precision;
use multidouble::simd::Native;

pub mod newton;

/// A system F(z) = 0 of `len()` equations in as many unknowns, evaluated in the precision P. The state z is held as
/// P::PARTS part arrays of f64 (part q of unknown i at `z[q][i]`).
pub trait Problem<P: Precision<Native>> {
    /// the number of unknowns, and of residual rows
    fn len(&self) -> usize;
    /// whether the problem has no unknowns
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// F(z)
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String>;
    /// The exact Jacobian dF/dz at z, as `P::lu_factor` takes it: part q of entry (i, j) at `c[q][j * ld + i]`, with
    /// the leading dimension ld a multiple of 8 and at least `len()` (`multidouble::linalg::ld_for`). Returns (c, ld).
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String>;
}
