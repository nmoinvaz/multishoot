//! Lane types made from and read as parts (value, tangents, jet coefficients) in octo-double, for the planted checks
//! that run one code over every lane type the engine evaluates.
use multidouble::native::{Oun, F8};
use multidouble::od::OD;
use multishoot::lane::{Dc, Dtn, Du, Fu, Fun, Jet, Lane, OJet, Ou, QJet, Qu, DV, OV, QV};
use multishoot::Fl;
use multisolver::prec::nan_max;

/// The precision of a lane type's values or of its derivative parts
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prec {
    F64,
    Dd,
    Qd,
    Od,
}

impl Prec {
    /// the relative error of a well-conditioned solve in this precision: about 2^-52, 2^-104, 2^-208 and 2^-416 times
    /// a few hundred
    pub fn bound(self) -> f64 {
        match self {
            Prec::F64 => 1e-11,
            Prec::Dd => 1e-27,
            Prec::Qd => 1e-57,
            Prec::Od => 1e-115,
        }
    }
}

/// A lane type of the planted checks
pub trait Probe: Lane {
    const NAME: &'static str;
    /// the precision of the values and of the derivative parts
    const VAL: Prec;
    const DER: Prec;
    /// the derivative parts p = 1, 2, ...: (n, sigma, tau), coefficient n of a series in h along the direction
    /// sigma of a field's seed and tau of a scalar's (a tangent is n = 1, a jet's coefficient p is n = p along (1, 1))
    fn parts() -> Vec<(usize, f64, f64)>;
    /// the field of value v and derivative parts t (part p is t[p - 1]: a tangent, or a jet's coefficient p)
    fn make<const B: usize>(v: &Fl<DV, B>, t: &[Fl<DV, B>]) -> Fl<Self, B>;
    /// part p of f (0 the value) at each point, in octo-double
    fn get<const B: usize>(f: &Fl<Self, B>, p: usize) -> Vec<OD>;
}

/// part k of a field read by `coef_od` (the value, a tangent of a dual in its precision, a jet's coefficient)
fn by_coef<T: Lane, const B: usize>(f: &Fl<T, B>, k: usize) -> Vec<OD> {
    let c = f.coef_od(k);
    (0..8 * B).map(|i| c.get(i)).collect()
}

/// the double-double field f at each point in octo-double
fn dd_parts<const B: usize>(f: &Fl<DV, B>) -> Vec<OD> {
    (0..8 * B).map(|i| OD::from_dd(f.get(i))).collect()
}

/// a value lane: no derivative parts
macro_rules! values {
    ($t:ty, $name:expr, $p:expr) => {
        impl Probe for $t {
            const NAME: &'static str = $name;
            const VAL: Prec = $p;
            const DER: Prec = $p;
            fn parts() -> Vec<(usize, f64, f64)> {
                vec![]
            }
            fn make<const B: usize>(v: &Fl<DV, B>, _t: &[Fl<DV, B>]) -> Fl<Self, B> {
                Fl::from_vt(v, &Fl::zero())
            }
            fn get<const B: usize>(f: &Fl<Self, B>, p: usize) -> Vec<OD> {
                by_coef(f, p)
            }
        }
    };
}
values!(DV, "DV", Prec::Dd);
values!(QV, "QV", Prec::Qd);
values!(OV, "OV", Prec::Od);

/// a dual with tangents in its value's precision
macro_rules! duals {
    ($t:ty, $name:expr, $p:expr, $parts:expr) => {
        impl Probe for $t {
            const NAME: &'static str = $name;
            const VAL: Prec = $p;
            const DER: Prec = $p;
            fn parts() -> Vec<(usize, f64, f64)> {
                $parts
            }
            fn make<const B: usize>(v: &Fl<DV, B>, t: &[Fl<DV, B>]) -> Fl<Self, B> {
                Fl::from_vt_n(v, t)
            }
            fn get<const B: usize>(f: &Fl<Self, B>, p: usize) -> Vec<OD> {
                by_coef(f, p)
            }
        }
    };
}
duals!(Du, "Du", Prec::Dd, vec![(1, 1.0, 1.0)]);
duals!(Qu, "Qu", Prec::Qd, vec![(1, 1.0, 1.0)]);
duals!(Ou, "Ou", Prec::Od, vec![(1, 1.0, 1.0)]);
duals!(Oun<2>, "Oun<2>", Prec::Od, vec![(1, 1.0, 1.0), (1, 0.0, 1.0)]);

/// a jet of four coefficients
macro_rules! jets {
    ($t:ty, $name:expr, $p:expr) => {
        impl Probe for $t {
            const NAME: &'static str = $name;
            const VAL: Prec = $p;
            const DER: Prec = $p;
            fn parts() -> Vec<(usize, f64, f64)> {
                vec![(1, 1.0, 1.0), (2, 1.0, 1.0), (3, 1.0, 1.0)]
            }
            fn make<const B: usize>(v: &Fl<DV, B>, t: &[Fl<DV, B>]) -> Fl<Self, B> {
                let mut f = Fl::<Self, B>::zero();
                f.set_coef(0, v);
                for (k, c) in t.iter().enumerate() {
                    f.set_coef(k + 1, c);
                }
                f
            }
            fn get<const B: usize>(f: &Fl<Self, B>, p: usize) -> Vec<OD> {
                by_coef(f, p)
            }
        }
    };
}
jets!(Jet<4>, "Jet<4>", Prec::Dd);
jets!(QJet<4>, "QJet<4>", Prec::Qd);
jets!(OJet<4>, "OJet<4>", Prec::Od);

impl Probe for Dc {
    const NAME: &'static str = "Dc";
    const VAL: Prec = Prec::Dd;
    const DER: Prec = Prec::Dd;
    fn parts() -> Vec<(usize, f64, f64)> {
        vec![(1, 1.0, 1.0), (1, 0.0, 0.5)]
    }
    fn make<const B: usize>(v: &Fl<DV, B>, t: &[Fl<DV, B>]) -> Fl<Self, B> {
        Fl::<Dc, B>::from_vri(v, &t[0], &t[1])
    }
    fn get<const B: usize>(f: &Fl<Self, B>, p: usize) -> Vec<OD> {
        match p {
            0 => dd_parts(&f.val()),
            1 => dd_parts(&f.tan_re()),
            _ => dd_parts(&f.tan_im()),
        }
    }
}

/// a lane with f64 tangents (`Fu`, `Fun`, `Dtn`)
macro_rules! f64_tangents {
    ($t:ty, $name:expr, $v:expr, $parts:expr) => {
        impl Probe for $t {
            const NAME: &'static str = $name;
            const VAL: Prec = $v;
            const DER: Prec = Prec::F64;
            fn parts() -> Vec<(usize, f64, f64)> {
                $parts
            }
            fn make<const B: usize>(v: &Fl<DV, B>, t: &[Fl<DV, B>]) -> Fl<Self, B> {
                let z = Fl::<DV, B>::zero();
                let mut f = Fl::<Self, B>::from_vt(v, &z);
                for (k, c) in t.iter().enumerate() {
                    f.set_tan64(k, &Fl::<F8, B>::from_vt(c, &z));
                }
                f
            }
            fn get<const B: usize>(f: &Fl<Self, B>, p: usize) -> Vec<OD> {
                if p == 0 {
                    return by_coef(f, 0);
                }
                dd_parts(&f.tan64(p - 1).val())
            }
        }
    };
}
f64_tangents!(Fu, "Fu", Prec::F64, vec![(1, 1.0, 1.0)]);
f64_tangents!(Fun<2>, "Fun<2>", Prec::F64, vec![(1, 1.0, 1.0), (1, 0.0, 1.0)]);
f64_tangents!(Dtn<2>, "Dtn<2>", Prec::Dd, vec![(1, 1.0, 1.0), (1, 0.0, 1.0)]);

/// max_i |a_i - b_i| / max_i |b_i|, in f64 (zero when b is zero)
pub fn rel_err(a: &[OD], b: &[OD]) -> f64 {
    let big = b.iter().map(|x| x.to_f64().abs()).fold(0.0, nan_max);
    let d = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).to_f64().abs())
        .fold(0.0, nan_max);
    if big == 0.0 {
        d
    } else {
        d / big
    }
}

/// every part of f at every point into the hash: the eight f64 components of each part's octo-double
pub fn hash_parts<T: Probe, const B: usize>(h: &mut super::Fnv, f: &Fl<T, B>) {
    for p in 0..=T::parts().len() {
        for v in T::get(f, p) {
            v.x.iter().for_each(|&x| h.f64(x));
        }
    }
}
