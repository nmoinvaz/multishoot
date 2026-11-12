//! The problem's side of the engine: the [`Equations`] trait, the state at a point and the work arrays of one Taylor
//! step.
//!
//! A problem has `NF` fields, each a function of x and of a periodic variable theta held on `M = 8 B` collocation
//! points (`Fl<T, B>`). Between the nodes the engine carries the state by Taylor series in x; the equations supply
//! the series' coefficients by the recursion of their right-hand side. At the two ends the solution starts from
//! unknown data (`NL` fields at the left end, `NR` at the right): the equations map the data to the state at the
//! adjacent node, by a series at a singular point or by boundary conditions at a regular one. `NG` global unknowns
//! (an eigenvalue, a period) enter every block.
use crate::lane::Lane;
use crate::mode::Rate;
use multidouble::field::Fl;

/// The state at a point: the `NF` fields
pub type State<T, const B: usize> = Vec<Fl<T, B>>;

/// The Taylor coefficients of one step about its start x0 and the equations' scratch series.
///
/// `c[f][k]` is coefficient k of field f, k = 0..=kt. The engine sets order 0 (the state at x0) before
/// [`Equations::taylor_coeffs`] and reads orders 0..=kt after it. `aux[i]` are `NAUX` series the recursion may keep
/// (products, right-hand sides), each of the same length; their contents are the equations'.
pub struct Work<T: Lane, const B: usize> {
    /// the Taylor order: coefficients 0..=kt
    pub kt: usize,
    pub c: Vec<Vec<Fl<T, B>>>,
    pub aux: Vec<Vec<Fl<T, B>>>,
    /// the growth rate of a mode-matrix evaluation, zero otherwise: the recursion passes each t-derivative of a
    /// coefficient through `rate.shift` ([`crate::mode`])
    pub rate: Rate,
}

impl<T: Lane, const B: usize> Work<T, B> {
    /// work for `nf` fields, `naux` scratch series and order kt; every series has kt + 2 entries, zero
    pub fn new(nf: usize, naux: usize, kt: usize) -> Self {
        let v = || vec![Fl::<T, B>::zero(); kt + 2];
        Work {
            kt,
            c: (0..nf).map(|_| v()).collect(),
            aux: (0..naux).map(|_| v()).collect(),
            rate: Rate::ZERO,
        }
    }
}

/// A boundary-value problem for the engine: its sizes, the Taylor recursion of its right-hand side, and the maps
/// from the endpoint data to the states at the end nodes.
///
/// Every method is generic over the lane type, so that one code gives the residual (values), the exact Jacobian
/// (duals), the Newton step in any precision and the residual's series in a parameter (jets). Constants enter as
/// double-double (`Fl::muls`, `Fl::cst`); a parameter of the equations that the residual is expanded in enters through
/// the lane's parameter arithmetic (`Lane::P`, `Param::var`).
pub trait Equations<const B: usize>: Sync {
    /// fields per point
    const NF: usize;
    /// fields of the left endpoint's data
    const NL: usize;
    /// fields of the right endpoint's data
    const NR: usize;
    /// global unknowns
    const NG: usize = 0;
    /// scratch series of the Taylor recursion (`Work::aux`)
    const NAUX: usize = 0;
    /// Whether every method acts on each collocation point alone, with no d/dtheta or other product across points:
    /// the M points (or lanes) are then independent problems side by side, a block's Jacobian is diagonal across
    /// them, and one dual evaluation with a unit tangent on every point gives the Jacobian columns of all M points of
    /// an input field at once. A problem without a periodic direction fills its lanes this way (8 parameter values,
    /// 8 initial data) and sets it; a problem that couples the points must leave it false.
    const POINTWISE: bool = false;

    /// What the endpoint maps need once per evaluation (of the residual, the Jacobian or the jet) rather than once
    /// per block or column: a series' coefficient table, the factors of a periodic solve. `()` when nothing. A
    /// preparation that can fail is formed by `try_prepare`.
    type Prep: Sync + Default;

    /// The preparation for one evaluation at the state z, given by its double-double parts (the first two of any
    /// precision). It holds values only: what the lanes' tangents and series need is formed in the lane type by the
    /// endpoint maps, which may use the preparation as a preconditioner, such as a double-double LU refined to the
    /// lane's precision by iterative refinement (Moler, J. ACM 14, 316 (1967)).
    fn prepare(&self, _zh: &[f64], _zl: &[f64]) -> Self::Prep {
        Self::Prep::default()
    }

    /// The preparation for a mode-matrix evaluation at the growth rate ([`crate::mode`]): `prepare`'s by default,
    /// which serves endpoint maps without a t-derivative. Endpoint maps with one keep the rate in their preparation,
    /// with what it changes (the factors of a shifted periodic solve), and pass their t-derivatives through
    /// `rate.shift`.
    fn prepare_rate(&self, zh: &[f64], zl: &[f64], _rate: Rate) -> Self::Prep {
        self.prepare(zh, zl)
    }

    /// The preparation of `prepare`, or the reason it could not be formed (a factorization with a zero pivot, a
    /// value that is not finite). The engine forms every evaluation's preparation through this method, and its
    /// entry points return the error as theirs. By default `prepare`'s, which cannot fail.
    ///
    /// Equations that override it and are evaluated by the mode matrix override `try_prepare_rate` as well: its
    /// default is `prepare_rate`'s, which does not call this method.
    fn try_prepare(&self, zh: &[f64], zl: &[f64]) -> Result<Self::Prep, String> {
        Ok(self.prepare(zh, zl))
    }

    /// The preparation of `prepare_rate`, or the reason it could not be formed, as `try_prepare`: the mode matrix
    /// forms its preparation through this method. By default `prepare_rate`'s, which cannot fail.
    fn try_prepare_rate(&self, zh: &[f64], zl: &[f64], rate: Rate) -> Result<Self::Prep, String> {
        Ok(self.prepare_rate(zh, zl, rate))
    }

    /// Taylor coefficients 1..=w.kt of every field about x0, from order 0 in `w.c[f][0]`; g holds the globals.
    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>);

    /// The state at the first node x from the left endpoint's data (`NL` fields): boundary conditions at a regular
    /// end, or a series from a singular one (a Frobenius series at a regular singular point) summed to x.
    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &Self::Prep) -> State<T, B>;

    /// The state at the last node x from the right endpoint's data (`NR` fields), as `left_state`.
    fn right_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], g: &[T], prep: &Self::Prep) -> State<T, B>;
}
