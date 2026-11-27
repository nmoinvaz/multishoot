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
//! Newton's method and the other solvers of F(z) = 0 are the crate [`multisolver`], written over its [`Problem`]
//! trait, which [`Shooting`] implements.
//!
//! Modules:
//!
//! * [`equations`]: the [`Equations`] trait, the state and the Taylor work arrays;
//! * [`taylor`]: Taylor steps, paths, the step planner and the choice of the Taylor order;
//! * [`shooting`]: the layout ([`Shooting`]): nodes, segments, the unknown vector and the residual's blocks;
//! * [`residual`]: the residual F(z) in any precision;
//! * [`jacobian`]: its exact Jacobian by forward-mode duals, with one tangent per evaluation or several (as a
//!   [`Problem`], [`WithTangents`]);
//! * [`sub`]: a block of the Jacobian, `J[rows, cols]`, and of the residual, `F[rows]`, from the evaluations that reach
//!   them;
//! * [`jvp`]: the Jacobian-vector product J v by one forward-mode pass per block, without the Jacobian;
//! * [`fast`]: the f64 Jacobian from f64-tangent lanes, dense or in staircase storage, and the mixed-precision Newton;
//! * [`family`]: the layout as a family in one of its globals ([`InGlobal`], a [`Family`]), for multisolver's
//!   continuation through folds, and the f64 factors of its corrector's bordered matrix;
//! * [`jet`]: the residual's Taylor series in a parameter of the equations;
//! * [`mode`]: the mode matrix, the linearization for perturbations growing at a complex rate, and its determinant,
//!   whose zeros in the rate are counted by multisolver's argument principle;
//! * [`periodic`]: collocation grids in the periodic variable and their derivative matrices;
//! * [`implicit`]: Taylor steps of an implicit second-order ODE, a polynomial in y'', with its root branch carried by
//!   continuity and the places where dN/dy'' vanishes reported;
//! * [`interp`]: trigonometric interpolation in any precision, between uniform grids or at any points, and grids
//!   mapped by Boyd's periodic arctan map with their derivative matrices;
//! * [`series`]: the periodic solves of a series at a singular end, one per order, in every lane type;
//! * [`mobius`]: the Moebius variable of a series about a singular end, which keeps a far singularity far;
//! * [`projected`]: one field's rows of a block replaced by a projection onto its low harmonics;
//! * [`relayout`]: a solution moved to new node positions or another number of collocation points;
//! * [`lane`]: the lane types the engine is generic over, and constants in each lane's own precision;
//! * `control` (feature `mutate`): deliberate errors, the controls of the accuracy checks.
#[cfg(feature = "mutate")]
pub mod control;
pub mod equations;
pub mod family;
pub mod fast;
pub mod implicit;
pub mod interp;
pub mod jacobian;
pub mod jet;
pub mod jvp;
pub mod lane;
pub mod mobius;
pub mod mode;
pub mod periodic;
pub mod projected;
pub mod relayout;
pub mod residual;
pub mod series;
pub mod shooting;
pub mod sub;
pub mod taylor;

pub use equations::{Equations, State, Work};
pub use family::{BorderedLu, InGlobal};
pub use fast::{BlockLanes, MixedLu, Storage};
pub use jacobian::WithTangents;
pub use lane::LaneConst;
pub use mobius::Mobius;
pub use mode::{ModeMatrix, Rate};
pub use multidouble;
pub use multidouble::dd::DD;
pub use multidouble::field::{cs, cs_range, cs_w, horner, Fl};
pub use multidouble::precision::{Double, Octo, Precision, Quad};
pub use multidouble::simd::Native;
pub use multisolver;
pub use multisolver::{Family, Jvp, Problem, ResidualJet};
pub use projected::ProjectedRows;
pub use relayout::Parities;
pub use series::{series_lus, PeriodicLu, Refine};
pub use shooting::{ExtraRow, Shooting};
pub use taylor::{OrderChoice, OrderCost, Plan};
