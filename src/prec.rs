//! The arithmetic the residual, the Jacobian and Newton's method run in, behind one trait: [`Prec`], with
//! double-double ([`Double`]: `DD`, `DV`, `Du`, `Jet<N>`, `DDLu`).
//!
//! It names the same kinds of types and operations as multidouble's `precision::Precision`: a scalar of PARTS f64
//! parts, its 8-lane vector, forward-mode dual and power series, a dense LU, and the loads and reads between them. A
//! state z is held as PARTS arrays of f64, part q of unknown i at `z[q][i]`.
use crate::lane::{Du, Jet, Lane, DV};
use multidouble::dd::DD;
use multidouble::field::Fl;
use multidouble::linalg::DDLu;
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

/// One multi-double precision: its types, and the operations whose names differ between precisions.
pub trait Prec: Copy + Send + Sync + 'static {
    /// the number of f64 parts of a scalar
    const PARTS: usize;
    /// the scalar
    type Scalar: Copy
        + Send
        + Sync
        + Debug
        + PartialEq
        + Add<Output = Self::Scalar>
        + Sub<Output = Self::Scalar>
        + Mul<Output = Self::Scalar>
        + Neg<Output = Self::Scalar>;
    /// 8 scalars in lanes
    type V: Lane;
    /// forward-mode dual: value and one tangent
    type U: Lane;
    /// truncated power series with N coefficients
    type J<const N: usize>: Lane;
    /// dense LU with partial pivoting
    type Lu: Send + Sync;

    const ZERO: Self::Scalar;
    const ONE: Self::Scalar;

    /// a scalar from up to PARTS parts, the missing ones zero, renormalized
    fn new(p: &[f64]) -> Self::Scalar;
    /// a scalar from exactly PARTS parts as they are
    fn from_raw(p: &[f64]) -> Self::Scalar;
    /// part q of a scalar
    fn part(x: Self::Scalar, q: usize) -> f64;
    /// from double-double, exactly
    fn from_dd(d: DD) -> Self::Scalar;
    /// rounded to double-double
    fn to_dd(x: Self::Scalar) -> DD;
    /// the sum of the products of the pairs
    fn dot(pairs: impl Iterator<Item = (Self::Scalar, Self::Scalar)>) -> Self::Scalar;

    /// the scalar in every lane
    fn splat(x: Self::Scalar) -> Self::V;
    /// a dual from its value and tangent lanes
    fn dual(v: Self::V, d: Self::V) -> Self::U;
    /// coefficient k of a lane in this precision
    fn coef<L: Lane>(l: L, k: usize) -> Self::V;
    /// set coefficient k of a lane from this precision
    fn set_coef<L: Lane>(l: &mut L, k: usize, c: Self::V);

    /// a field from PARTS part arrays of at least 8 B values each
    fn load_field<const B: usize>(parts: &[&[f64]]) -> Fl<Self::V, B>;
    /// value j of a field
    fn field_get<const B: usize>(f: &Fl<Self::V, B>, j: usize) -> Self::Scalar;
    /// coefficient k of a field in this precision
    fn field_coef<L: Lane, const B: usize>(f: &Fl<L, B>, k: usize) -> Fl<Self::V, B>;
    /// set coefficient k of a field from this precision
    fn field_set_coef<L: Lane, const B: usize>(f: &mut Fl<L, B>, k: usize, c: &Fl<Self::V, B>);
    /// a dual field from its values and tangents, exactly
    fn field_dual<const B: usize>(v: &Fl<Self::V, B>, t: &Fl<Self::V, B>) -> Fl<Self::U, B>;

    /// factor the n x n matrix of PARTS column-major part arrays with leading dimension ld
    fn lu_factor(n: usize, ld: usize, c: Vec<Vec<f64>>, par: bool) -> Result<Self::Lu, String>;
    /// solve in place with the factored matrix
    fn lu_solve(lu: &Self::Lu, b: &mut [Self::Scalar]);
}

/// Double-double: `DD`, `DV`, `Du`, `Jet<N>`, `DDLu`; the dot product is the plain sum of double-double products in
/// order
#[derive(Clone, Copy, Debug)]
pub struct Double;

impl Prec for Double {
    const PARTS: usize = 2;
    type Scalar = DD;
    type V = DV;
    type U = Du;
    type J<const N: usize> = Jet<N>;
    type Lu = DDLu;
    const ZERO: DD = DD::ZERO;
    const ONE: DD = DD::ONE;

    #[inline]
    fn new(p: &[f64]) -> DD {
        assert!(p.len() <= 2, "Double::new: at most 2 parts, not {}", p.len());
        DD::new(p.first().copied().unwrap_or(0.0), p.get(1).copied().unwrap_or(0.0))
    }
    #[inline]
    fn from_raw(p: &[f64]) -> DD {
        assert_eq!(p.len(), 2, "Double::from_raw: 2 parts");
        DD { hi: p[0], lo: p[1] }
    }
    #[inline]
    fn part(x: DD, q: usize) -> f64 {
        [x.hi, x.lo][q]
    }
    #[inline]
    fn from_dd(d: DD) -> DD {
        d
    }
    #[inline]
    fn to_dd(x: DD) -> DD {
        x
    }
    fn dot(pairs: impl Iterator<Item = (DD, DD)>) -> DD {
        let mut acc = DD::ZERO;
        for (a, b) in pairs {
            acc = acc + a * b;
        }
        acc
    }
    #[inline]
    fn splat(x: DD) -> DV {
        DV::splat(x)
    }
    #[inline]
    fn dual(v: DV, d: DV) -> Du {
        Du { v, d }
    }
    #[inline]
    fn coef<L: Lane>(l: L, k: usize) -> DV {
        l.coef(k)
    }
    #[inline]
    fn set_coef<L: Lane>(l: &mut L, k: usize, c: DV) {
        l.set_coef(k, c)
    }
    fn load_field<const B: usize>(parts: &[&[f64]]) -> Fl<DV, B> {
        assert_eq!(parts.len(), 2, "Double::load_field: 2 parts");
        Fl::load(parts[0], parts[1])
    }
    #[inline]
    fn field_get<const B: usize>(f: &Fl<DV, B>, j: usize) -> DD {
        f.get(j)
    }
    #[inline]
    fn field_coef<L: Lane, const B: usize>(f: &Fl<L, B>, k: usize) -> Fl<DV, B> {
        f.coef(k)
    }
    #[inline]
    fn field_set_coef<L: Lane, const B: usize>(f: &mut Fl<L, B>, k: usize, c: &Fl<DV, B>) {
        f.set_coef(k, c)
    }
    #[inline]
    fn field_dual<const B: usize>(v: &Fl<DV, B>, t: &Fl<DV, B>) -> Fl<Du, B> {
        Fl::<Du, B>::from_vt(v, t)
    }
    fn lu_factor(n: usize, ld: usize, c: Vec<Vec<f64>>, par: bool) -> Result<DDLu, String> {
        let [hi, lo]: [Vec<f64>; 2] = c
            .try_into()
            .map_err(|c: Vec<Vec<f64>>| format!("Double::lu_factor: 2 parts, not {}", c.len()))?;
        DDLu::factor(n, ld, hi, lo, par)
    }
    #[inline]
    fn lu_solve(lu: &DDLu, b: &mut [DD]) {
        lu.solve(b)
    }
}

/// The PARTS slices of a state held as owned part arrays
pub fn parts(z: &[Vec<f64>]) -> Vec<&[f64]> {
    z.iter().map(|p| p.as_slice()).collect()
}

/// max |x| over the scalars (hi part of the double-double rounding)
pub fn max_abs<P: Prec>(x: &[P::Scalar]) -> f64 {
    x.iter().map(|&v| P::to_dd(v).hi.abs()).fold(0.0, f64::max)
}
