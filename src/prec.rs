//! Helpers for states held as PARTS arrays of f64 (part q of unknown i at `z[q][i]`), in any of multidouble's
//! precisions (`precision::Precision`: `Double`, `Quad`, `Octo`), which the residual, the Jacobian and Newton's
//! method are written over.
use multidouble::precision::Precision;
use multidouble::simd::Native;

/// The PARTS slices of a state held as owned part arrays
pub fn parts(z: &[Vec<f64>]) -> Vec<&[f64]> {
    z.iter().map(|p| p.as_slice()).collect()
}

/// max |x| over the scalars (hi part of the double-double rounding)
pub fn max_abs<P: Precision<Native>>(x: &[P::Scalar]) -> f64 {
    x.iter().map(|&v| P::to_dd(v).hi.abs()).fold(0.0, f64::max)
}

/// A state in more parts: the given part arrays, then zero arrays up to `parts` (a double-double state as the start
/// of a quad-double Newton, a quad-double one for octo-double)
pub fn widen(z: &[Vec<f64>], parts: usize) -> Vec<Vec<f64>> {
    assert!(parts >= z.len(), "widen: {} parts into {parts}", z.len());
    let l = z.first().map_or(0, |p| p.len());
    z.iter()
        .cloned()
        .chain((z.len()..parts).map(|_| vec![0.0; l]))
        .collect()
}
