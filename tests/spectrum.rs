//! The Fourier coefficients of a field on its periodic grid, and its resolution, on planted fields with exact
//! coefficients.
use multishoot::periodic::{sin_cos_grid, spectrum, tail, Parity};
use multishoot::prec::nan_max;
use multishoot::DD;

/// exact unit pairs (cos, sin) for the phases: c^2 + s^2 = 1 in rationals
const PHASES: [(f64, f64); 4] = [
    (0.6, 0.8),
    (5.0 / 13.0, 12.0 / 13.0),
    (8.0 / 17.0, 15.0 / 17.0),
    (1.0, 0.0),
];

/// the planted coefficients (k, a_k, b_k): amplitude 10^-k (exact as given) and an exact phase, with b_k = 0 at k = 0
/// and the Nyquist harmonic (the grid cannot hold a sine there)
fn planted(parity: Parity, m: usize) -> Vec<(i64, DD, DD)> {
    let n = if parity == Parity::Full { m as i64 } else { 2 * m as i64 };
    (0..=n / 2)
        .filter(|&k| match parity {
            Parity::Full => true,
            Parity::Odd => k % 2 == 1,
            Parity::Even => k % 2 == 0,
        })
        .map(|k| {
            let amp = DD::from_f64(10f64.powi(-(k as i32)));
            let (c, s) = if k == 0 || 2 * k == n {
                (1.0, 0.0)
            } else {
                PHASES[k as usize % 4]
            };
            (k, amp * DD::from_f64(c), amp * DD::from_f64(s))
        })
        .collect()
}

/// the field sum_k a_k cos k theta_j + b_k sin k theta_j on the grid
fn field(parity: Parity, m: usize, coef: &[(i64, DD, DD)]) -> Vec<DD> {
    (0..m)
        .map(|j| {
            coef.iter().fold(DD::ZERO, |t, &(k, a, b)| {
                let (s, c) = sin_cos_grid(parity, m, j, k);
                t + a * c + b * s
            })
        })
        .collect()
}

/// the largest difference of spectrum's coefficients from the wanted ones, in double-double
fn coef_err(got: &[(i64, DD, DD)], want: &[(i64, DD, DD)]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            assert_eq!(g.0, w.0);
            (g.1 - w.1).hi.abs().max((g.2 - w.2).hi.abs())
        })
        .fold(0.0, nan_max)
}

/// Pass, written before the first run: on the full grid of 16 points and the odd and even half grids of 8, a field of
/// amplitudes 10^-k with exact phases gives back its coefficients to 1e-30, and its tail over the top two harmonics
/// is the planted ratio to 1e-12 relative. The control: the field of a planted set whose a_3 is changed by 1e-20
/// misses the original coefficients by more than 1e-25.
#[test]
fn spectrum_recovers_planted_coefficients() {
    for (parity, m) in [(Parity::Full, 16), (Parity::Odd, 8), (Parity::Even, 8)] {
        let want = planted(parity, m);
        let f = field(parity, m, &want);
        let e = coef_err(&spectrum(parity, &f), &want);
        let amp: Vec<f64> = want.iter().map(|&(_, a, b)| a.hi.hypot(b.hi)).collect();
        let ratio = amp[amp.len() - 2..].iter().copied().fold(0.0, nan_max) / amp.iter().copied().fold(0.0, nan_max);
        let t = tail(parity, &f, 2);
        println!("{parity:?} on {m}: coefficients {e:.3e}, tail {t:.6e} (planted {ratio:.6e})");
        assert!(e <= 1e-30 && (t / ratio - 1.0).abs() <= 1e-12, "{parity:?}");
    }
    let want = planted(Parity::Full, 16);
    let mut off = want.clone();
    off[3].1 = off[3].1 + DD::from_f64(1e-20);
    let e = coef_err(&spectrum(Parity::Full, &field(Parity::Full, 16, &off)), &want);
    println!("control, a_3 changed by 1e-20: {e:.3e}");
    assert!(e > 1e-25, "control did not fire");
}
