//! The Moebius variable of a series about a singular point, which keeps a far singularity far from the evaluation
//! point.
//!
//! A series about the point w = 1, summed at a node w_l, converges in the affine variable 1 - w only inside the disc
//! that reaches the nearest singularity, here put at w = 0: |1 - w_l| < 1. Its coefficients there fall only like that
//! disc's radius, and the recurrence of a solution regular at w = 0 (a minimal solution) loses digits in forward
//! recursion: rounding excites the solutions singular at w = 0, which decay more slowly (Gautschi, SIAM Rev. 9, 24
//! (1967)). The Moebius variable
//!
//! ```text
//! t = (1 - w) / (1 + a w),    tau = t / t_l,    t_l = (1 - w_l) / (1 + a w_l),
//! ```
//!
//! keeps w = 1 at t = 0, puts the node at tau = 1, w = 0 at tau = 1 / t_l and w = infinity at tau = -1 / (a t_l). A
//! larger a moves w = 0 farther from the node in tau at the cost of bringing w = infinity nearer (a = 0 is the affine
//! variable). The series in tau converges at the node when every singularity of the solution, mapped by `tau`, lies
//! outside the unit disc |tau| <= 1, and that is the caller's to check: a > 0 does not ensure it, nor does
//! |a t_l| < 1, which places w = infinity alone. A pole at w = 2 with a = 1 and w_l = 1/4 sits at tau = -5/9, inside,
//! where the affine series, of radius 1 about w = 1, converges at the node. Where the singularities lie outside it, the
//! series converges where the affine one need not, and the coefficients of a solution regular at w = 0 stay of order
//! one.
//!
//! In tau, w = (1 - t_l tau) / (1 + a t_l tau), and the derivatives are polynomials of degree two times d/dtau:
//!
//! ```text
//! w^j d/dw = -(P_j(t_l tau) / t_l) d/dtau,    P_j(t) = (1 - t)^j (1 + a t)^(2 - j) / (1 + a),    j = 0, 1, 2,
//! ```
//!
//! and a product w g of a series g follows from (1 + a t_l tau) (w g) = (1 - t_l tau) g, two terms per order, exact
//! and stable for |a t_l| < 1, where a Cauchy sum with w's coefficients, which fall only like (a t_l)^j, costs O(k)
//! per order and is cut short if banded. Every constant is formed in octo-double and applied in the lane's own
//! precision ([`LaneConst`]), since t_l and the coefficients are not exact in double-double.
//!
//! The variable w is the problem's: an affine image of x, or x^2 for fields of one parity in x, with the singular
//! point at w = 1 and the singularity to keep far at w = 0.
#[cfg(feature = "mutate")]
use crate::control::Control;
use crate::lane::{Lane, LaneConst};
use multidouble::field::Fl;
use multidouble::od::OD;

/// The Moebius variable tau = t / t_l, t = (1 - w) / (1 + a w), of a series about w = 1 summed at w_l (tau = 1). Its
/// constants follow from a and w_l when it is made, so another a or w_l takes a new variable.
#[derive(Clone, Debug)]
pub struct Mobius {
    /// the parameter a >= 0
    a: f64,
    /// the node w_l, in octo-double
    wl: OD,
    t_l: LaneConst,
    /// a t_l, the products' recurrence coefficient
    at_l: LaneConst,
    /// w's coefficients in tau: w_0 = 1, w_1, and the ratio of w_(j+1) to w_j for j >= 1
    w1: LaneConst,
    wr: LaneConst,
    /// the coefficients of -P_j(t_l tau) / t_l in tau, j = 0, 1, 2
    q: [[LaneConst; 3]; 3],
}

impl Mobius {
    /// The variable with parameter a about w = 1, summed at w_l. Panics unless 1 + a w_l and 1 - w_l are nonzero
    /// and |a t_l| < 1, where w = infinity lies outside the unit disc in tau and the products' recurrence is stable.
    /// The solution's other singularities, mapped by [`Mobius::tau`], must lie outside it for the series to converge
    /// at the node.
    pub fn new(a: f64, wl: OD) -> Self {
        Self::build(a, wl, false, false, false)
    }

    /// `new` with a deliberate error ([`Control::MobiusSign`], [`Control::MobiusRecurrence`],
    /// [`Control::MobiusConstDD`]), for a control
    #[cfg(feature = "mutate")]
    pub fn with_control(a: f64, wl: OD, c: Control) -> Self {
        Self::build(
            a,
            wl,
            c == Control::MobiusSign,
            c == Control::MobiusRecurrence,
            c == Control::MobiusConstDD,
        )
    }

    fn build(a: f64, wl: OD, sign: bool, rec: bool, dd: bool) -> Self {
        let one = OD::ONE;
        let ao = OD::from_f64(a);
        let den = one + ao * wl;
        assert!(
            a >= 0.0 && den.to_f64() != 0.0 && (one - wl).to_f64() != 0.0,
            "Mobius: a = {a}, w_l = {}",
            wl.to_f64()
        );
        let tl = (one - wl) / den;
        let atl = ao * tl;
        assert!(
            atl.to_f64().abs() < 1.0,
            "Mobius: |a t_l| = {} is not below 1",
            atl.to_f64().abs()
        );
        let c = |v: OD| {
            if dd {
                LaneConst::new(OD::from_dd(v.to_dd()))
            } else {
                LaneConst::new(v)
            }
        };
        // the map's a, flipped by MobiusSign in w(tau) and the products only
        let am = if sign { -ao } else { ao };
        let atm = am * tl;
        let atr = if rec {
            atm * (one + OD::from_f64(2f64.powi(-60)))
        } else {
            atm
        };
        let ia1 = one / (one + ao);
        let itl = one / tl;
        let two = OD::from_f64(2.0);
        // P_j's coefficients in t, then -p_i t_l^(i - 1) in tau
        let p = [
            [ia1, two * ao * ia1, ao * ao * ia1],
            [ia1, (ao - one) * ia1, -(ao * ia1)],
            [ia1, -(two * ia1), ia1],
        ];
        let q = p.map(|pj| [c(-(pj[0] * itl)), c(-pj[1]), c(-(pj[2] * tl))]);
        Mobius {
            a,
            wl,
            t_l: c(tl),
            at_l: c(atr),
            w1: c(-((one + am) * tl)),
            wr: c(-atm),
            q,
        }
    }

    /// the parameter a
    pub fn a(&self) -> f64 {
        self.a
    }

    /// the node w_l
    pub fn wl(&self) -> OD {
        self.wl
    }

    /// t_l = (1 - w_l) / (1 + a w_l), the unit of tau
    pub fn t_l(&self) -> LaneConst {
        self.t_l
    }

    /// tau at the point w: (1 - w) / ((1 + a w) t_l)
    pub fn tau(&self, w: OD) -> OD {
        (OD::ONE - w) / ((OD::ONE + OD::from_f64(self.a) * w) * self.t_l.od())
    }

    /// coefficient j of w(tau) = (1 - t_l tau) / (1 + a t_l tau): 1, -(1 + a) t_l, then times -a t_l per order
    pub fn w_coef(&self, j: usize) -> LaneConst {
        match j {
            0 => LaneConst::from_f64(1.0),
            _ => {
                let mut v = self.w1.od();
                for _ in 1..j {
                    v = v * self.wr.od();
                }
                LaneConst::new(v)
            }
        }
    }

    /// Coefficient k of the product w g from g_k, g_(k-1) and the product's coefficient k - 1, as
    /// (1 + a t_l tau) (w g) = (1 - t_l tau) g gives it: (g_k - t_l g_(k-1)) - a t_l (w g)_(k-1); g_0 at k = 0.
    #[inline]
    pub fn mul_w<T: Lane, const B: usize>(&self, g: &[Fl<T, B>], wg: &[Fl<T, B>], k: usize) -> Fl<T, B> {
        if k == 0 {
            g[0]
        } else {
            (g[k] - self.t_l.mul(g[k - 1])) - self.at_l.mul(wg[k - 1])
        }
    }

    /// The coefficients q_i, i = 0, 1, 2, of w^j d/dw = (q_0 + q_1 tau + q_2 tau^2) d/dtau, j = 0, 1, 2:
    /// q_i = -p_i t_l^(i - 1), p_i those of P_j(t) = (1 - t)^j (1 + a t)^(2 - j) / (1 + a)
    pub fn deriv_poly(&self, j: usize) -> [LaneConst; 3] {
        self.q[j]
    }

    /// The part of coefficient k of w^j dg/dw that the orders k and k - 1 of g give, sum_{i=1,2} q_i (k + 1 - i)
    /// g_(k+1-i): with q_0 (k + 1) g_(k+1) ([`Mobius::deriv_poly`]) the whole coefficient, so that an equation's
    /// order k is solved for g_(k+1)
    #[inline]
    pub fn deriv_lower<T: Lane, const B: usize>(&self, j: usize, g: &[Fl<T, B>], k: usize) -> Fl<T, B> {
        let q = &self.q[j];
        let mut s = Fl::<T, B>::zero();
        for (i, qi) in q.iter().enumerate().take(k.min(2) + 1).skip(1) {
            s = s + qi.mul(g[k + 1 - i]).mulf((k + 1 - i) as f64);
        }
        s
    }

    /// coefficient k of w^j dg/dw from g's coefficients 0..=k + 1: q_0 (k + 1) g_(k+1) + `deriv_lower`
    #[inline]
    pub fn deriv<T: Lane, const B: usize>(&self, j: usize, g: &[Fl<T, B>], k: usize) -> Fl<T, B> {
        self.q[j][0].mul(g[k + 1]).mulf((k + 1) as f64) + self.deriv_lower(j, g, k)
    }

    /// the series summed at the node, tau = 1: its coefficients added from the last to the first
    pub fn sum<T: Lane, const B: usize>(g: &[Fl<T, B>]) -> Fl<T, B> {
        g.iter().rev().fold(Fl::<T, B>::zero(), |s, c| s + *c)
    }
}
