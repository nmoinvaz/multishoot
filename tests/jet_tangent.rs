//! The jet session's tangent update (`JetSession::tangent_from`): each order formed once, with z_k zero, and moved
//! by its change once z_k is given, from a two-coefficient session at z_0. On `Branch`, whose order-k methods form one
//! order from the series they keep (in `Work::aux` and an endpoint map's `keep`) and whose parameter enters through a
//! constant that double-double rounds, and on `Cylinder` and the projected planted problem with the default order-k
//! methods, the moved orders agree with `residual_jet`'s to rounding, the orders below the first moved one bit for
//! bit; the series of `Branch`'s planted branch in its parameter comes out as planted.
mod common;

use common::{field, wave, Cylinder, Planted};
use multidouble::field::cs_at;
use multishoot::lane::{DMat, Lane, Param, DV};
use multishoot::periodic::{dmat, sin_cos_grid, Parity};
use multishoot::{cs, Double, Equations, ExtraRow, Fl, Native, Octo, Plan, Precision, Quad, Shooting, State, Work, DD};
use multisolver::newton::NewtonOpts;
use multisolver::prec::{parts, widen};
use std::collections::BTreeMap;

/// the parameter's value
const MU0: f64 = 2.0;
/// the frequency of the planted solution, the global's value
const OMEGA: f64 = 1.3;
/// the highest powers of g = 1 / (1 + x) and of the parameter in the sources
const NMAX: usize = 4;
const MMAX: usize = 4;

/// A polynomial in g = 1 / (1 + x) and the parameter mu with field coefficients: (power of g, power of mu) -> field
type Poly<const B: usize> = BTreeMap<(usize, usize), Fl<DV, B>>;

fn padd<const B: usize>(a: &Poly<B>, b: &Poly<B>, s: f64) -> Poly<B> {
    let mut out = a.clone();
    for (key, f) in b {
        let t = f.mulf(s);
        out.entry(*key).and_modify(|e| *e = *e + t).or_insert(t);
    }
    out
}

fn pmul<const B: usize>(a: &Poly<B>, b: &Poly<B>) -> Poly<B> {
    let mut out = Poly::<B>::new();
    for ((n, m), f) in a {
        for ((n2, m2), f2) in b {
            let t = *f * *f2;
            out.entry((n + n2, m + m2)).and_modify(|e| *e = *e + t).or_insert(t);
        }
    }
    out
}

/// d/dx, with dg/dx = -g^2
fn pdx<const B: usize>(a: &Poly<B>) -> Poly<B> {
    a.iter()
        .filter(|((n, _), _)| *n > 0)
        .map(|((n, m), f)| ((n + 1, *m), f.mulf(-(*n as f64))))
        .collect()
}

/// Taylor coefficients 0..=kt about x0 of g^n, n = 0..=NMAX: `(g^n)_k = (-1)^k C(n + k - 1, k) (1 + x0)^-(n + k)`
fn gpow(x0: f64, kt: usize) -> Vec<Vec<DD>> {
    let sig = DD::ONE / (DD::ONE + DD::from_f64(x0));
    (0..=NMAX)
        .map(|n| {
            let mut gk = vec![DD::ZERO; kt + 1];
            let mut c = DD::ONE;
            for _ in 0..n {
                c = c * sig;
            }
            gk[0] = c;
            if n > 0 {
                for (k, gkk) in gk.iter_mut().enumerate().skip(1) {
                    c = (c * sig).mul_f(-((n + k - 1) as f64)) / DD::from_f64(k as f64);
                    *gkk = c;
                }
            }
            gk
        })
        .collect()
}

/// The planted branch in a parameter: an odd field u and an even field v on the half grid, the frequency omega a
/// global with the extra row omega = OMEGA, mu the equations' parameter at MU0,
///
/// ```text
/// u_x = -omega u_theta + mu u v + s_u(x, mu)
/// v_x = u^2 - v + s_v(x, mu)
/// ```
///
/// u at the left end is u*(x_0, mu) and v is `d mu / (mu + 1)` from the left data d; at the right end the data is u
/// and v is v*(x_K, mu). The sources are made, as `Planted`'s, from the manufactured solution `u* = g A + B + mu g P
/// + mu^2 R`, `v* = g^2 C + E + mu g Q` (g = 1 / (1 + x)), so its samples solve the collocation system to rounding
/// for every mu, and the branch's series in h = mu - MU0 is known: u* and v* are polynomials in h, and the left data
/// is `v*(x_0, mu) (1 + 1 / mu)`, whose coefficients in h are `1 + 1 / MU0` and `(1 / MU0) (-1 / MU0)^l`.
///
/// Its order-k methods form one order from the series they keep: D u_k and (u v)_k in `Work::aux`, which enter
/// products with omega and with mu, and d mu in the left map's `keep`, which enters a product with 1 / (mu + 1). That
/// constant is 1/3 at MU0, which double-double rounds, so a tangent whose parameter is double-double (a dual lane's)
/// misses the jets' derivative at its rounding. `wrong_pair_keep` keeps one more series on two-coefficient jets, the
/// control of the session's shape check.
pub struct Branch<const B: usize> {
    d: DMat<B>,
    /// the manufactured solution and the sources as polynomials in g and mu
    u: Poly<B>,
    v: Poly<B>,
    su: Poly<B>,
    sv: Poly<B>,
    wrong_pair_keep: bool,
}

impl<const B: usize> Branch<B> {
    fn new() -> Self {
        let m = 8 * B;
        let sc = |j: usize, k: i64| sin_cos_grid(Parity::Odd, m, j, k);
        let d = dmat::<B>(Parity::Odd);
        let a = field::<B>(|j| sc(j, 1).0 + sc(j, 3).1.mul_f(0.3));
        let b = field::<B>(|j| sc(j, 1).1.mul_f(0.2) - sc(j, 3).0.mul_f(0.1));
        let p = field::<B>(|j| sc(j, 2).0.mul_f(0.2) + sc(j, 1).1.mul_f(0.1));
        let r = field::<B>(|j| sc(j, 1).0.mul_f(0.05));
        let c = field::<B>(|j| DD::from_f64(0.5) + sc(j, 2).1.mul_f(0.25));
        let e = field::<B>(|j| DD::ONE + sc(j, 2).0.mul_f(0.1));
        let q = field::<B>(|j| sc(j, 1).1.mul_f(0.15));
        let u: Poly<B> = [((1, 0), a), ((0, 0), b), ((1, 1), p), ((0, 2), r)]
            .into_iter()
            .collect();
        let v: Poly<B> = [((2, 0), c), ((0, 0), e), ((1, 1), q)].into_iter().collect();
        let om = DD::from_f64(OMEGA);
        let du: Poly<B> = u.iter().map(|(key, f)| (*key, d.apply(f).muls(om))).collect();
        let muuv: Poly<B> = pmul(&u, &v).into_iter().map(|((n, m), f)| ((n, m + 1), f)).collect();
        let su = padd(&padd(&pdx(&u), &du, 1.0), &muuv, -1.0);
        let sv = padd(&padd(&pdx(&v), &pmul(&u, &u), -1.0), &v, 1.0);
        assert!(su.keys().chain(sv.keys()).all(|&(n, m)| n <= NMAX && m <= MMAX));
        Branch {
            d,
            u,
            v,
            su,
            sv,
            wrong_pair_keep: false,
        }
    }

    /// the parameter's powers 0..=MMAX in the lane's parameter arithmetic
    fn mu_powers<T: Lane>() -> Vec<T::P> {
        let mu = T::P::var(DD::from_f64(MU0));
        let mut mp = vec![T::P::one()];
        for m in 1..=MMAX {
            mp.push(mp[m - 1] * mu);
        }
        mp
    }

    /// the double-double fields of each power of mu of `sum_(n, m) s_nm (g^n)_k mu^m`
    fn by_power(s: &Poly<B>, gn: &[Vec<DD>], k: usize) -> Vec<Option<Fl<DV, B>>> {
        (0..=MMAX)
            .map(|m| {
                let mut f: Option<Fl<DV, B>> = None;
                for (n, g) in gn.iter().enumerate() {
                    if let Some(snm) = s.get(&(n, m)) {
                        let t = snm.muls(g[k]);
                        f = Some(f.map_or(t, |x| x + t));
                    }
                }
                f
            })
            .collect()
    }

    /// `sum_(n, m) s_nm (g^n)_k mu^m` on a lane
    fn sum<T: Lane>(s: &Poly<B>, gn: &[Vec<DD>], k: usize, mp: &[T::P]) -> Fl<T, B> {
        let mut out = Fl::<T, B>::zero();
        for (m, f) in Self::by_power(s, gn, k).into_iter().enumerate() {
            if let Some(f) = f {
                out = out + Fl::<T, B>::cst(&f).mulp(mp[m]);
            }
        }
        out
    }

    /// `sum`'s coefficient j alone
    fn sum_at<T: Lane>(s: &Poly<B>, gn: &[Vec<DD>], k: usize, mp: &[T::P], j: usize) -> Fl<T, B> {
        let mut out = Fl::<T, B>::zero();
        for (m, f) in Self::by_power(s, gn, k).into_iter().enumerate() {
            if let Some(f) = f {
                let mut t = Fl::<T, B>::zero();
                Fl::<T, B>::cst(&f).mulp_at(&mp[m], j, &mut t);
                out = out + t;
            }
        }
        out
    }

    /// the branch's field (u* or v*) at x, coefficient k in h = mu - MU0: `sum_(n, m) p_nm g^n C(m, k) MU0^(m - k)`
    fn branch_coef(p: &Poly<B>, x: f64, k: usize) -> Fl<DV, B> {
        let gn = gpow(x, 0);
        let mut out = Fl::<DV, B>::zero();
        for ((n, m), f) in p {
            if *m < k {
                continue;
            }
            let mut c = DD::ONE;
            for i in 0..k {
                c = c.mul_f((m - i) as f64) / DD::from_f64((i + 1) as f64);
            }
            for _ in k..*m {
                c = c.mul_f(MU0);
            }
            out = out + f.muls(gn[*n][0] * c);
        }
        out
    }
}

impl<const B: usize> Equations<B> for Branch<B> {
    const NF: usize = 2;
    const NL: usize = 1;
    const NR: usize = 1;
    const NG: usize = 1;
    const NAUX: usize = 2;
    const ORDER_BY_ORDER: bool = true;
    type Prep = ();

    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        let kt = w.kt;
        let mp = Self::mu_powers::<T>();
        let gn = gpow(x0, kt);
        let [u, v] = &mut w.c[..] else {
            unreachable!("two fields")
        };
        let [du, uv] = &mut w.aux[..] else {
            unreachable!("two series")
        };
        for k in 0..kt {
            let su = Self::sum::<T>(&self.su, &gn, k, &mp);
            let sv = Self::sum::<T>(&self.sv, &gn, k, &mp);
            du[k] = self.d.apply(&u[k]);
            uv[k] = cs(u, v, k);
            let ru = (uv[k].mulp(mp[1]) - du[k].mul_t(g[0])) + su;
            let rv = (cs(u, u, k) - v[k]) + sv;
            let ik = DD::ONE / DD::from_f64((k + 1) as f64);
            u[k + 1] = ru.muls(ik);
            v[k + 1] = rv.muls(ik);
        }
    }

    fn taylor_coeffs_at<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>, j: usize) {
        let kt = w.kt;
        let mp = Self::mu_powers::<T>();
        let gn = gpow(x0, kt);
        let [u, v] = &mut w.c[..] else {
            unreachable!("two fields")
        };
        let [du, uv] = &mut w.aux[..] else {
            unreachable!("two series")
        };
        let (mut s, mut t, mut q) = (Fl::<T, B>::zero(), Fl::<T, B>::zero(), Fl::<T, B>::zero());
        for k in 0..kt {
            let su = Self::sum_at::<T>(&self.su, &gn, k, &mp, j);
            let sv = Self::sum_at::<T>(&self.sv, &gn, k, &mp, j);
            du[k].copy_at(&self.d.apply(&u[k]), j);
            cs_at(&mut uv[k], u, v, k, j);
            uv[k].mulp_at(&mp[1], j, &mut s);
            du[k].mul_t_at(&g[0], j, &mut t);
            cs_at(&mut q, u, u, k, j);
            let ik = DD::ONE / DD::from_f64((k + 1) as f64);
            let (nu, nv) = (((s - t) + su).muls(ik), ((q - v[k]) + sv).muls(ik));
            u[k + 1].copy_at(&nu, j);
            v[k + 1].copy_at(&nv, j);
        }
    }

    fn left_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], _g: &[T], _prep: &()) -> State<T, B> {
        let mp = Self::mu_powers::<T>();
        let y = data[0].mulp(mp[1]);
        let v = y.mulp(T::P::one() / (mp[1] + T::P::one()));
        vec![Self::sum::<T>(&self.u, &gpow(x, 0), 0, &mp), v]
    }

    fn left_state_at<T: Lane>(
        &self,
        x: f64,
        data: &[Fl<T, B>],
        _g: &[T],
        _prep: &(),
        keep: &mut Vec<Vec<Fl<T, B>>>,
        j: usize,
    ) -> State<T, B> {
        if keep.is_empty() {
            *keep = vec![vec![Fl::zero()]];
            if self.wrong_pair_keep && T::NJ == 2 {
                keep.push(vec![Fl::zero()]);
            }
        }
        let mp = Self::mu_powers::<T>();
        data[0].mulp_at(&mp[1], j, &mut keep[0][0]);
        let mut v = Fl::<T, B>::zero();
        keep[0][0].mulp_at(&(T::P::one() / (mp[1] + T::P::one())), j, &mut v);
        vec![Self::sum_at::<T>(&self.u, &gpow(x, 0), 0, &mp, j), v]
    }

    fn right_state<T: Lane>(&self, x: f64, data: &[Fl<T, B>], _g: &[T], _prep: &()) -> State<T, B> {
        let mp = Self::mu_powers::<T>();
        vec![data[0], Self::sum::<T>(&self.v, &gpow(x, 0), 0, &mp)]
    }

    fn right_state_at<T: Lane>(
        &self,
        x: f64,
        data: &[Fl<T, B>],
        _g: &[T],
        _prep: &(),
        _keep: &mut Vec<Vec<Fl<T, B>>>,
        j: usize,
    ) -> State<T, B> {
        let mp = Self::mu_powers::<T>();
        vec![data[0], Self::sum_at::<T>(&self.v, &gpow(x, 0), 0, &mp, j)]
    }
}

/// `Branch`'s layout on four nodes with its extra row omega = OMEGA, planned from the planted solution (order 30, last
/// terms below 1e-32, steps at most 0.25), and the planted series' coefficient vectors 0..=n in h = mu - MU0, each as
/// double-double values
fn branch(n: usize) -> (Shooting<Branch<1>, 1>, Vec<Vec<DD>>) {
    let mut sh = Shooting::new(Branch::<1>::new(), vec![0.0, 0.5, 1.0, 1.5], 30);
    let og = sh.off_global();
    sh.extra.push(ExtraRow {
        terms: vec![(og, DD::ONE)],
        rhs: DD::from_f64(OMEGA),
    });
    let zs = planted_series(&sh, n);
    let (zh, zl): (Vec<f64>, Vec<f64>) = zs[0].iter().map(|v| (v.hi, v.lo)).unzip();
    sh.plan_segments(
        &zh,
        &zl,
        &Plan {
            kt: 30,
            tol: 1e-32,
            hmax: 0.25,
        },
    );
    (sh, zs)
}

/// The planted branch's coefficient vectors 0..=n: the node states u*, v*, the right data u*(x_K) and omega (at order
/// 0) by `Branch::branch_coef`; the left data `d_k = sum_i v*_i(x_0) c_(k - i)`, c_l the coefficients of 1 + 1 / mu:
/// `c_0 = 1 + 1 / MU0`, `c_l = (1 / MU0) (-1 / MU0)^l`
fn planted_series<const B: usize>(sh: &Shooting<Branch<B>, B>, n: usize) -> Vec<Vec<DD>> {
    let m = 8 * B;
    let k = sh.k();
    (0..=n)
        .map(|order| {
            let mut z = vec![DD::ZERO; sh.len()];
            let mut put = |o: usize, f: &Fl<DV, B>| {
                for j in 0..m {
                    z[o + j] = f.get(j);
                }
            };
            let mut d = Fl::<DV, B>::zero();
            for i in 0..=order {
                let l = order - i;
                let mut c = DD::from_f64(1.0 / MU0);
                for _ in 0..l {
                    c = c.mul_f(-1.0 / MU0);
                }
                if l == 0 {
                    c = c + DD::ONE;
                }
                d = d + Branch::<B>::branch_coef(&sh.eq.v, sh.nodes[0], i).muls(c);
            }
            put(0, &d);
            for i in 0..k {
                put(sh.off_node(i), &Branch::<B>::branch_coef(&sh.eq.u, sh.nodes[i], order));
                put(
                    sh.off_node(i) + m,
                    &Branch::<B>::branch_coef(&sh.eq.v, sh.nodes[i], order),
                );
            }
            put(
                sh.off_right(),
                &Branch::<B>::branch_coef(&sh.eq.u, sh.nodes[k - 1], order),
            );
            if order == 0 {
                z[sh.off_global()] = DD::from_f64(OMEGA);
            }
            z
        })
        .collect()
}

/// `Cylinder`'s layout and its mode (the default order-k methods)
fn cylinder() -> (Shooting<Cylinder<1>, 1>, Vec<Vec<f64>>) {
    Cylinder::<1>::shooting(Cylinder::new(), 30)
}

/// the planted problem with projected rows (the default order-k methods), and its solution
fn projected() -> (Shooting<Planted<2>, 2>, Vec<Vec<f64>>) {
    let (mut sh, z) = Planted::<2>::shooting(&[0.0, 0.5, 1.0, 1.5, 2.0], 30, 1e-32);
    let k = sh.k();
    sh.project_rows(k, 1, 0, Parity::Even, 2).unwrap();
    (sh, z)
}

/// `Branch`'s layout and its planted solution in double-double parts
fn branch_z0() -> (Shooting<Branch<1>, 1>, Vec<Vec<f64>>) {
    let (sh, zs) = branch(0);
    let (zh, zl): (Vec<f64>, Vec<f64>) = zs[0].iter().map(|v| (v.hi, v.lo)).unzip();
    (sh, vec![zh, zl])
}

/// the coefficient vectors z_1, z_2, ...: waves of size 0.1^k on every unknown, in double-double parts
fn coef_vector(l: usize, k: usize) -> Vec<Vec<f64>> {
    let s = 0.1f64.powi(k as i32);
    vec![(0..l).map(|i| s * wave(i + 31 * k)).collect(), vec![0.0; l]]
}

/// the number of rows whose parts differ in any bit, every row of the longer when the lengths differ
fn bits_differ<P: Precision<Native>>(a: &[P::Scalar], b: &[P::Scalar]) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len());
    }
    a.iter()
        .zip(b)
        .filter(|(&u, &v)| (0..P::PARTS).any(|q| P::part(u, q).to_bits() != P::part(v, q).to_bits()))
        .count()
}

/// max |a - b| / max |b| over the rows, in P's arithmetic rounded to f64; infinite when the lengths differ, NaN when
/// any difference is
fn rel_diff<P: Precision<Native>>(a: &[P::Scalar], b: &[P::Scalar]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mag = |x: P::Scalar| P::to_dd(x).hi.abs();
    let d = a
        .iter()
        .zip(b)
        .map(|(&u, &v)| mag(u - v))
        .fold(0.0, multisolver::prec::nan_max);
    let s = b.iter().map(|&v| mag(v)).fold(0.0, f64::max);
    if s > 0.0 {
        d / s
    } else {
        d
    }
}

/// the bound on the moved orders' normwise relative difference from `residual_jet`'s: 2^-(53 PARTS - 20), 2^20 times
/// the precision's unit roundoff
fn bound<P: Precision<Native>>() -> f64 {
    2f64.powi(-(53 * P::PARTS as i32 - 20))
}

/// Gives z_0..z_(n-1) to a session on `P::J<8>` lanes that moves orders from `from` on by their tangents and, after
/// each, compares the order it returns and every order it has formed with `residual_jet`'s on the same vectors: (rows
/// compared, rows of the orders below `from` that differ in any bit, the largest normwise relative difference of an
/// order from `from` on)
fn compare<E: Equations<B>, P: Precision<Native>, const B: usize>(
    sh: &Shooting<E, B>,
    z0: &[Vec<f64>],
    n: usize,
    from: Option<usize>,
    par: bool,
) -> Result<(usize, usize, f64), String> {
    compare_below::<E, P, B>(sh, z0, n, from, from, par)
}

/// `compare` with the orders compared bit for bit those below `bitwise` rather than below `from`, for the control
/// of the bitwise comparison
fn compare_below<E: Equations<B>, P: Precision<Native>, const B: usize>(
    sh: &Shooting<E, B>,
    z0: &[Vec<f64>],
    n: usize,
    from: Option<usize>,
    bitwise: Option<usize>,
    par: bool,
) -> Result<(usize, usize, f64), String> {
    let l = sh.len();
    let zs: Vec<Vec<Vec<f64>>> = (0..n)
        .map(|k| widen(&if k == 0 { z0.to_vec() } else { coef_vector(l, k) }, P::PARTS))
        .collect();
    let mut s = sh.jet_session::<P, 8>();
    s.tangent_from(from);
    let first = bitwise.map_or(usize::MAX, |f| f.max(1));
    let (mut rows, mut below, mut worst) = (0, 0, 0.0f64);
    for k in 0..n {
        let next = s.next(&parts(&zs[k]), par)?;
        let refs: Vec<Vec<&[f64]>> = zs[..=k].iter().map(|z| parts(z)).collect();
        let full = sh.residual_jet::<P, 8>(&refs, par, k + 2)?;
        // the returned order k + 1 is formed from the orders before it, so it is exact while they are
        let mut tally = |j: usize, a: &[P::Scalar], b: &[P::Scalar]| {
            if j < first {
                below += bits_differ::<P>(a, b);
            } else {
                worst = multisolver::prec::nan_max(worst, rel_diff::<P>(a, b));
            }
        };
        tally(k + 1, &next, &full[k + 1]);
        for (j, fj) in full.iter().enumerate() {
            tally(j, &s.order(j)?, fj);
            rows += fj.len();
        }
    }
    Ok((rows, below, worst))
}

/// Pass, written before the first run: with the session's default (no tangent), `Branch`'s order-k methods give
/// `residual_jet`'s orders bit for bit (0 rows differ) in double-, quad- and octo-double, with and without threads,
/// so that they are a fair test of the tangent update.
#[test]
fn branch_session_is_residual_jet_bitwise() {
    let (sh, z) = branch_z0();
    for par in [false, true] {
        let runs = [
            ("double-double", compare::<_, Double, 1>(&sh, &z, 6, None, par).unwrap()),
            ("quad-double", compare::<_, Quad, 1>(&sh, &z, 6, None, par).unwrap()),
            ("octo-double", compare::<_, Octo, 1>(&sh, &z, 6, None, par).unwrap()),
        ];
        for (name, (rows, below, _)) in runs {
            println!("branch, {name}, par {par}: {rows} rows compared, {below} differ");
            assert_eq!(below, 0, "{name}, par {par}: the session is not residual_jet's");
        }
    }
}

/// Pass, written before the first run: on `Branch` (its own order-k methods) and on `Cylinder` and the projected
/// planted problem (the defaults), in double-, quad- and octo-double, with and without threads, a session that moves
/// orders from 1 on, and one that moves them from 3 on, gives orders whose normwise relative difference from
/// `residual_jet`'s is at most 2^-(53 PARTS - 20) (`bound`), and with the tangent from 3 the orders 0..2 are
/// `residual_jet`'s bit for bit (0 rows differ).
#[test]
fn tangent_orders_agree_with_residual_jet() {
    let (br, zb) = branch_z0();
    let (cy, zc) = cylinder();
    let (pr, zp) = projected();
    for par in [false, true] {
        for from in [1, 3] {
            let f = Some(from);
            let runs = [
                (
                    "branch, dd",
                    compare::<_, Double, 1>(&br, &zb, 6, f, par),
                    bound::<Double>(),
                ),
                (
                    "branch, qd",
                    compare::<_, Quad, 1>(&br, &zb, 6, f, par),
                    bound::<Quad>(),
                ),
                (
                    "branch, od",
                    compare::<_, Octo, 1>(&br, &zb, 6, f, par),
                    bound::<Octo>(),
                ),
                (
                    "cylinder, dd",
                    compare::<_, Double, 1>(&cy, &zc, 6, f, par),
                    bound::<Double>(),
                ),
                (
                    "cylinder, od",
                    compare::<_, Octo, 1>(&cy, &zc, 6, f, par),
                    bound::<Octo>(),
                ),
                (
                    "projected, dd",
                    compare::<_, Double, 2>(&pr, &zp, 6, f, par),
                    bound::<Double>(),
                ),
            ];
            for (name, r, b) in runs {
                let (rows, below, worst) = r.unwrap();
                println!(
                    "{name}, from {from}, par {par}: {rows} rows, {below} differ below order {from}, moved orders \
                     within {worst:.3e} of residual_jet's (bound {b:.3e})"
                );
                assert_eq!(
                    below, 0,
                    "{name}, from {from}, par {par}: an order below {from} changed"
                );
                assert!(worst <= b, "{name}, from {from}, par {par}: {worst:.3e} past {b:.3e}");
            }
        }
    }
}

/// Pass, written before the first run: a session whose two-coefficient pair keeps a series the session's jets do not
/// (`Branch::wrong_pair_keep`) stops with an error at the first moved order rather than misplacing the changes, and the
/// same session without it runs.
#[test]
fn tangent_checks_the_pair_layout() {
    let (mut br, z) = branch_z0();
    br.eq.wrong_pair_keep = true;
    let bad = compare::<_, Double, 1>(&br, &z, 3, Some(1), true);
    println!("pair keeping another shape: {:?}", bad.as_ref().map(|r| r.2));
    assert!(bad.is_err(), "the shape check did not fire");
    br.eq.wrong_pair_keep = false;
    assert!(compare::<_, Double, 1>(&br, &z, 3, Some(1), true).is_ok());
}

/// The series z_1..z_n of `Branch`'s branch in h = mu - MU0 at the solution z0 (P::PARTS parts) by a session on
/// `P::J<16>` lanes: each order's right-hand side solved with the Jacobian's LU at z0
fn series<P: Precision<Native>>(
    sh: &Shooting<Branch<1>, 1>,
    z0: &[Vec<f64>],
    n: usize,
    from: Option<usize>,
) -> Vec<Vec<P::Scalar>> {
    let lu = sh.newton_step::<P>(&parts(z0), true).unwrap().lu;
    let mut s = sh.jet_session::<P, 16>();
    s.tangent_from(from);
    let mut z = z0.to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let f = s.next(&parts(&z), true).unwrap();
        let mut x: Vec<P::Scalar> = f.iter().map(|&v| -v).collect();
        P::lu_solve(&lu, &mut x);
        z = (0..P::PARTS)
            .map(|q| x.iter().map(|&v| P::part(v, q)).collect())
            .collect();
        out.push(x);
    }
    out
}

/// max over the orders 1..=n and the unknowns of |z_k - z*_k| (P's difference rounded to f64)
fn planted_error<P: Precision<Native>>(got: &[Vec<P::Scalar>], want: &[Vec<DD>]) -> f64 {
    got.iter()
        .zip(&want[1..])
        .flat_map(|(g, w)| g.iter().zip(w).map(|(&a, &b)| P::to_dd(a - P::from_dd(b)).hi.abs()))
        .fold(0.0, multisolver::prec::nan_max)
}

/// `Branch`'s solution at MU0 in P (Newton from the planted samples), its planted series to order n, and the layout
fn branch_solved<P: Precision<Native>>(n: usize) -> (Shooting<Branch<1>, 1>, Vec<Vec<f64>>, Vec<Vec<DD>>) {
    let (sh, zs) = branch(n);
    let (zh, zl): (Vec<f64>, Vec<f64>) = zs[0].iter().map(|v| (v.hi, v.lo)).unzip();
    let mut z = widen(&[zh, zl], P::PARTS);
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 12,
        ..Default::default()
    };
    assert!(sh.newton::<P>(&mut z, &opts).unwrap().converged);
    (sh, z, zs)
}

/// the bound on the planted series' error: absolute, the coefficients being at most about 3
const PLANTED: f64 = 1e-24;

/// Pass, written before the first run: the series of order 12 of `Branch`'s planted branch, by a session that moves
/// orders from 1 on by their tangents, in double- and octo-double, is the planted one within 1e-24 at every order
/// (z_1, z_2 of the nodes nonzero, z_3.. of the nodes zero, the left data's every order nonzero), as is the series by
/// the session that forms every order again.
#[test]
fn tangent_series_is_the_planted_branch() {
    let n = 12;
    let (sh, z, want) = branch_solved::<Double>(n);
    let exact = planted_error::<Double>(&series::<Double>(&sh, &z, n, None), &want);
    let moved = planted_error::<Double>(&series::<Double>(&sh, &z, n, Some(1)), &want);
    let (sho, zo, _) = branch_solved::<Octo>(n);
    let moved_o = planted_error::<Octo>(&series::<Octo>(&sho, &zo, n, Some(1)), &want);
    let size = want[1..].iter().flatten().map(|v| v.hi.abs()).fold(0.0, f64::max);
    println!(
        "planted branch to order {n} (coefficients up to {size:.3}): formed again {exact:.3e}, moved {moved:.3e}, \
         moved in octo-double {moved_o:.3e} (bound {PLANTED:.0e})"
    );
    assert!(size > 0.1, "the planted series is trivial");
    assert!(exact <= PLANTED && moved <= PLANTED && moved_o <= PLANTED);
}

/// Pass, written before the first run: the bitwise comparison of the orders below the first moved one rejects a moved
/// order. With the tangent from 1, taking orders 1 and 2 as bitwise counts rows that differ, while the same session
/// with orders below 1 bitwise counts none; and `bits_differ` counts one row for a one-ulp change of one part.
#[test]
fn tangent_bitwise_check_rejects_a_moved_order() {
    let (br, z) = branch_z0();
    let (_, moved, _) = compare_below::<_, Double, 1>(&br, &z, 4, Some(1), Some(3), true).unwrap();
    let (_, unmoved, _) = compare_below::<_, Double, 1>(&br, &z, 4, Some(1), Some(1), true).unwrap();
    let one = <Double as Precision<Native>>::from_dd(DD::ONE);
    let next = <Double as Precision<Native>>::from_raw(&[1.0, f64::from_bits(1e-17f64.to_bits() + 1)]);
    let base = <Double as Precision<Native>>::from_raw(&[1.0, 1e-17]);
    let ulp = bits_differ::<Double>(&[one, base], &[one, next]);
    println!("orders 1, 2 compared bit for bit: {moved} rows differ; none: {unmoved}; a one-ulp change: {ulp} row");
    assert!(moved > 0, "the bitwise comparison did not see the moved orders");
    assert_eq!((unmoved, ulp), (0, 1));
}

/// rows that differ in any bit, over every order a session on `Double` jets moving orders from 1 on has formed after
/// each of zs, from `residual_jet`'s on the same vectors
fn rows_moved(sh: &Shooting<Branch<1>, 1>, zs: &[Vec<Vec<f64>>]) -> usize {
    let mut s = sh.jet_session::<Double, 8>();
    s.tangent_from(Some(1));
    let mut differ = 0;
    for k in 0..zs.len() {
        s.next(&parts(&zs[k]), true).unwrap();
        let refs: Vec<Vec<&[f64]>> = zs[..=k].iter().map(|z| parts(z)).collect();
        let full = sh.residual_jet::<Double, 8>(&refs, true, k + 2).unwrap();
        for (j, fj) in full.iter().enumerate() {
            differ += bits_differ::<Double>(&s.order(j).unwrap(), fj);
        }
    }
    differ
}

/// Pass, written before the first run: when z_2 is about 2^1000 times z_1, its seed scaled toward z_1's size would
/// lose its lower parts to underflow, so the session forms order 2 again and every order is `residual_jet`'s bit for
/// bit. The control: z_1 of z_2's size, whose seed scales back, moves order 2, and its bits differ.
#[test]
fn tangent_forms_again_when_the_seed_underflows() {
    let (br, z0) = branch_z0();
    let l = br.len();
    let vector = |s: f64, k: usize| -> Vec<Vec<f64>> {
        let hi: Vec<f64> = (0..l).map(|i| s * (0.5 + 0.5 * wave(i + 31 * k).abs())).collect();
        let lo = hi.iter().map(|&h| h * 1e-30).collect();
        vec![hi, lo]
    };
    let underflow = rows_moved(&br, &[z0.clone(), vector(1e-301, 1), vector(1.0, 2)]);
    let scaled = rows_moved(&br, &[z0, vector(0.1, 1), vector(1.0, 2)]);
    println!("z_2 2^1000 times z_1: {underflow} rows differ from residual_jet; z_1 of its size: {scaled}");
    assert_eq!(underflow, 0, "the seed that underflows was not formed again");
    assert!(scaled > 0, "control: a moved order did not change its bits");
}

/// The controls of `tangent_orders_agree_with_residual_jet` and `tangent_series_is_the_planted_branch` (feature
/// `mutate`): the tangent update without the series the equations keep (`Control::JetTangentKeep`), and without any
/// update (`Control::JetStale`), take `Branch`'s moved orders past the bound and its series off the planted one.
#[cfg(feature = "mutate")]
#[test]
fn tangent_controls_fire() {
    use multishoot::control::Control;
    let n = 12;
    for c in [Control::JetTangentKeep, Control::JetStale] {
        let (mut sh, z, want) = branch_solved::<Double>(n);
        sh.set_control(c);
        let (_, _, worst) = compare::<_, Double, 1>(&sh, &z, 6, Some(1), true).unwrap();
        let planted = planted_error::<Double>(&series::<Double>(&sh, &z, n, Some(1)), &want);
        println!("control {c:?}: moved orders within {worst:.3e} of residual_jet's, planted error {planted:.3e}");
        assert!(worst > bound::<Double>(), "control {c:?} did not move the orders");
        assert!(planted > PLANTED, "control {c:?} did not move the series");
    }
}

/// The series of order n of `Branch`'s planted branch by multisolver's ANM (`anm::series`, which takes its orders
/// from `jet_orders`), unrefined, with `Shooting::jet_tangent_from` set to `from`
fn anm_series(sh: &mut Shooting<Branch<1>, 1>, z: &[Vec<f64>], n: usize, from: Option<usize>) -> Vec<Vec<DD>> {
    sh.jet_tangent_from = from;
    multisolver::anm::series::<Double, _>(sh, &parts(z), n, 0, true)
        .unwrap()
        .z
}

/// the number of coefficients whose bits differ between two series
fn coefs_differ(a: &[Vec<DD>], b: &[Vec<DD>]) -> usize {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .filter(|(x, y)| x.hi.to_bits() != y.hi.to_bits() || x.lo.to_bits() != y.lo.to_bits())
        .count()
}

/// Pass, written before the first run: through `jet_orders`, the ANM's series of order 12 with `jet_tangent_from` 1
/// is the planted branch within 1e-24, as is the default's, and its bits differ from the default's in some
/// coefficient, so the setting reaches the session; with `jet_tangent_from` None it is the default's bit for bit.
#[test]
fn anm_takes_the_tangent_update_through_jet_orders() {
    let n = 12;
    let (mut sh, z, want) = branch_solved::<Double>(n);
    let plain = anm_series(&mut sh, &z, n, None);
    let moved = anm_series(&mut sh, &z, n, Some(1));
    let again = anm_series(&mut sh, &z, n, None);
    let (ep, em) = (
        planted_error::<Double>(&plain[1..], &want),
        planted_error::<Double>(&moved[1..], &want),
    );
    let (differ, same) = (coefs_differ(&plain, &moved), coefs_differ(&plain, &again));
    println!(
        "ANM through jet_orders: planted error {ep:.3e} formed again, {em:.3e} moved; {differ} coefficients differ"
    );
    assert!(ep <= PLANTED && em <= PLANTED, "{ep:e} {em:e}");
    assert!(differ > 0, "the tangent update did not reach the ANM's session");
    assert_eq!(same, 0);
}

/// The control of `anm_takes_the_tangent_update_through_jet_orders` (feature `mutate`): the tangent update without
/// the series the equations keep (`Control::JetTangentKeep`) must take the ANM's series off the planted one with
/// `jet_tangent_from` 1, and must leave it as it was without (no update is made then).
#[cfg(feature = "mutate")]
#[test]
fn anm_tangent_control_fires() {
    use multishoot::control::Control;
    let n = 12;
    let (mut sh, z, want) = branch_solved::<Double>(n);
    let plain = anm_series(&mut sh, &z, n, None);
    sh.set_control(Control::JetTangentKeep);
    let moved = planted_error::<Double>(&anm_series(&mut sh, &z, n, Some(1))[1..], &want);
    let unmoved = coefs_differ(&plain, &anm_series(&mut sh, &z, n, None));
    println!("control JetTangentKeep through the ANM: planted error {moved:.3e}; without the update {unmoved} differ");
    assert!(moved > PLANTED, "control did not fire: {moved:e}");
    assert_eq!(unmoved, 0);
}
