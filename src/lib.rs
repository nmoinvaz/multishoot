//! Multiple shooting for boundary-value problems, in double-, quad- and octo-double precision.
//!
//! A boundary-value problem asks for the solution of differential equations that meets conditions at two or more
//! places. Multiple shooting splits the interval at nodes, integrates the equations across each piece from the
//! values held at its start, and asks Newton's method to make every piece end where the next one begins. The unknowns
//! are the values at the nodes, the data of series that start the solution at singular endpoints, and global
//! parameters such as an eigenvalue; the residual is the list of mismatches; it is zero exactly when the pieces join
//! into one solution.
//!
//! The crate is the engine of that method, generic over the equations, built on
//! [`multidouble`](https://github.com/nmoinvaz/multidouble)'s arithmetic:
//!
//! * the equations enter through one trait: the Taylor coefficients of their right-hand side at a point, and the
//!   series that start the solution at the endpoints;
//! * between the nodes the solution is carried by Taylor series, with steps planned so that each truncation term
//!   stays below a tolerance;
//! * the fields may depend on a periodic variable, represented on collocation points, with exact derivatives in it;
//! * the Jacobian is exact, by forward-mode dual numbers over the same code that forms the residual, and so is its
//!   product with a vector, from one pass per block;
//! * Newton's method runs in double-double, quad-double or octo-double (multidouble's `Precision`), as the
//!   conditioning requires, and the residual can be expanded as a Taylor series in a parameter of the equations.
//!
//! Status: under construction (0.1.0, not published).
//!
//! Modules:
//!
//! * [`equations`]: the [`Equations`] trait, the state and the Taylor work arrays;
//! * [`taylor`]: Taylor steps, paths and the step planner;
//! * [`shooting`]: the layout ([`Shooting`]): nodes, segments, the unknown vector and the residual's blocks;
//! * [`residual`]: the residual F(z) in any precision;
//! * [`jacobian`]: its exact Jacobian by forward-mode duals;
//! * [`jvp`]: the Jacobian-vector product J v by one forward-mode pass per block, without the Jacobian;
//! * [`solve`]: solvers of F(z) = 0 over any [`Problem`], which [`Shooting`] is one of: [`newton`], the Newton step
//!   and the damped Newton loop;
//! * [`fast`]: the f64 Jacobian from f64-tangent lanes, dense or in staircase storage, and the mixed-precision Newton;
//! * [`jet`]: the residual's Taylor series in a parameter of the equations;
//! * [`mode`]: the mode matrix, the linearization for perturbations growing at a complex rate, and its determinant;
//! * [`prec`]: helpers for states held as part arrays in any of multidouble's precisions ([`Precision`]:
//!   double-double [`Double`], quad-double [`Quad`], octo-double [`Octo`]);
//! * [`periodic`]: collocation grids in the periodic variable and their derivative matrices;
//! * [`projected`]: one field's rows of a block replaced by a projection onto its low harmonics;
//! * [`lane`]: the lane types the engine is generic over;
//! * `control` (feature `mutate`): deliberate errors, the controls of the accuracy checks.
#[cfg(feature = "mutate")]
pub mod control;
pub mod equations;
pub mod fast;
pub mod jacobian;
pub mod jet;
pub mod jvp;
pub mod lane;
pub mod mode;
pub mod periodic;
pub mod prec;
pub mod projected;
pub mod residual;
pub mod shooting;
pub mod solve;
pub mod taylor;

pub use equations::{Equations, State, Work};
pub use fast::{MixedLu, Storage};
pub use mode::{ModeMatrix, Rate};
pub use multidouble;
pub use multidouble::dd::DD;
pub use multidouble::field::{cs, cs_range, cs_w, horner, Fl};
pub use multidouble::precision::{Double, Octo, Precision, Quad};
pub use multidouble::simd::Native;
pub use projected::ProjectedRows;
pub use shooting::{ExtraRow, Shooting};
pub use solve::newton::{NewtonOpts, NewtonReport};
pub use solve::{anm, gcr, lm, newton, predict};
pub use solve::{Jvp, Problem, ResidualJet};
pub use taylor::Plan;
