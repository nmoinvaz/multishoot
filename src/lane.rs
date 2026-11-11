//! The lane types the engine is generic over: multidouble's `dual::Lane` (values, tangents, jets) with its
//! `field::MatLane` (the product with a constant matrix such as d/dtheta), on the compile-time backend `Native`.
//!
//! Every lane type of multidouble's is one: `DV` (double-double values), `Du` (value and one tangent), `Dc` (value
//! and a complex tangent), `QV`, `Qu`, `OV`, `Ou` (quad- and octo-double), `Jet<N>`, `QJet<N>`, `OJet<N>` (truncated
//! power series in a parameter). The equations are written once over [`Lane`], and the engine runs them with values,
//! exact derivatives or series.
pub use multidouble::dual::{Lane as BaseLane, Param};
pub use multidouble::field::MatLane;
pub use multidouble::native::{DMat, Dc, Du, Jet, OJet, Ou, QJet, Qu, DV, OV, QV};
use multidouble::simd::Native;

/// multidouble's lanes on the compile-time backend, with the product by a constant double-double matrix
pub trait Lane: MatLane<S = Native> {}

impl<T: MatLane<S = Native>> Lane for T {}
