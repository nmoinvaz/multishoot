//! The residual's Taylor series in a parameter: the residual on a jet state `z(h) = sum_k z_k h^k`, in the lanes
//! `P::J<N>` (multidouble's `Jet`, `QJet`, `OJet`), with the equations' parameter entered as the jet variable.
//!
//! The equations form a parameter's constants through the lane's parameter arithmetic (`Lane::P`): `Param::var(c)`
//! is c on every other lane and the series c + h on a jet, so every block's coefficients carry their own Taylor
//! coefficients in it. With z_k for k >= 1 absent (zero) the coefficient of order k is the right-hand side of order k
//! of the Taylor continuation in the parameter; with z_1 given, order 1 is J z_1 + dF/dparameter.
//!
//! A series of order N built by `residual_jet` costs N evaluations of growing length, each forming every lower order
//! again. [`JetSession`] forms it one order at a time instead: it keeps each block's state, its endpoint map's
//! series and each Taylor step's work between orders, and the equations' order-k methods
//! ([`Equations::taylor_coeffs_at`], `left_state_at`, `right_state_at`) form order k from the lower orders kept. Their
//! defaults form every order again, so a session over equations without them gives `residual_jet`'s bits at the cost
//! of `residual_jet`.
//!
//! A session forms order k twice: with z_k zero, for the right-hand side of z_k's solve, and again once z_k is given.
//! Between the two only the inputs' coefficient k changes, and coefficient k of every series is affine in it, its
//! linear part the derivative of order 0. With [`JetSession::tangent_from`] the second forming is replaced by that
//! derivative, taken from a second session on two-coefficient jets `P::J<2>` at z_0: its order 1 formed with the
//! inputs' coefficient 1 set to the change, less its order 1 with them zero (the parameter's own part), is the change of
//! every series. The pair's jets carry the parameter in the same arithmetic as the session's (a dual lane would carry it
//! in double-double), and its order 1 costs two products per term where forming order k again costs k + 1.
use crate::equations::{Equations, State, Work};
use crate::lane::{BaseLane, Lane};
use crate::shooting::Shooting;
use multidouble::dd::DD;
use multidouble::field::{horner, Fl};
use multidouble::precision::Precision;
use multidouble::simd::Native;
use rayon::prelude::*;

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// The residual's coefficients of orders 0..nout - 1 on the jet state with coefficient vectors zs (each PARTS
    /// part arrays of length `len()`, at most N of them, z_k zero past them), nout <= N, one scalar per row. The
    /// extra rows are linear: order 0 is `sum w_i z_0,i - rhs`, order k >= 1 `sum w_i z_k,i`.
    pub fn residual_jet<P: Precision<Native>, const N: usize>(
        &self,
        zs: &[Vec<&[f64]>],
        par: bool,
        nout: usize,
    ) -> Result<Vec<Vec<P::Scalar>>, String> {
        if zs.is_empty() || zs.len() > N || nout == 0 || nout > N {
            return Err(format!(
                "residual_jet: 1..={N} coefficient vectors and 1..={N} outputs, not {} and {nout}",
                zs.len()
            ));
        }
        for z in zs {
            self.check_z::<P>(z)?;
        }
        let m = Self::M;
        let fld_k = |k: usize, o: usize| -> Fl<P::V, B> {
            if k < zs.len() {
                self.load::<P>(&zs[k], o)
            } else {
                Fl::zero()
            }
        };
        let fld_jet = |o: usize| -> Fl<P::J<N>, B> {
            let mut f = Fl::<P::J<N>, B>::zero();
            for k in 0..zs.len() {
                P::field_set_coef(&mut f, k, &fld_k(k, o));
            }
            f
        };
        let og = self.off_global();
        let g: Vec<P::J<N>> = (0..E::NG)
            .map(|i| {
                let mut gi = <P::J<N> as BaseLane>::zero();
                for (k, z) in zs.iter().enumerate() {
                    P::set_coef(&mut gi, k, P::splat(self.scalar::<P>(z, og + i)));
                }
                gi
            })
            .collect();
        let prep = self.prep(&zs[0])?;
        let one = |b: usize| -> State<P::J<N>, B> {
            let inp: Vec<Fl<P::J<N>, B>> = self.block_inputs(b).iter().map(|&o| fld_jet(o)).collect();
            self.block_state::<P::J<N>>(b, &inp, &g, &prep)
        };
        let nb = self.k() + 1;
        let states: Vec<State<P::J<N>, B>> = if par {
            (0..nb).into_par_iter().map(one).collect()
        } else {
            (0..nb).map(one).collect()
        };
        let mut out = vec![vec![P::ZERO; self.rows()]; nout];
        for (b, s) in states.iter().enumerate() {
            let t = self.block_target(b);
            let r0 = self.row_block(b);
            for q in 0..E::NF {
                for (k, ok) in out.iter_mut().enumerate() {
                    let d = P::field_coef(&s[q], k) - fld_k(k, t + q * m);
                    for i in 0..m {
                        ok[r0 + q * m + i] = P::field_get(&d, i);
                    }
                }
            }
        }
        for (k, ok) in out.iter_mut().enumerate() {
            self.project_residual::<P>(ok, |p| {
                (0..m)
                    .map(|j| {
                        if k < zs.len() {
                            self.scalar::<P>(&zs[k], p.pin + j)
                        } else {
                            P::ZERO
                        }
                    })
                    .collect()
            });
        }
        let re = self.row_extra();
        for (k, ok) in out.iter_mut().enumerate() {
            for (r, row) in self.extra.iter().enumerate() {
                ok[re + r] = if k == 0 {
                    self.extra_row::<P>(&zs[0], r)
                } else if k < zs.len() {
                    P::dot(
                        row.terms
                            .iter()
                            .map(|&(i, w)| (P::from_dd(w), self.scalar::<P>(&zs[k], i))),
                    )
                } else {
                    P::ZERO
                };
            }
            self.scale_rows::<P>(ok)?;
        }
        Ok(out)
    }
}

/// What a session keeps of a block between orders: an endpoint map's series, or a segment's work and end state at
/// each Taylor step
enum Kept<T: Lane, const B: usize> {
    End(Vec<Vec<Fl<T, B>>>),
    Seg(Vec<Work<T, B>>, Vec<State<T, B>>),
}

/// a block's index, its kept series and its state, as `JetSession::form` visits them
type BlockMut<'b, T, const B: usize> = (usize, (&'b mut Kept<T, B>, &'b mut State<T, B>));

/// What a series of the session is to the tangent update
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// set from its source as forming sets it (a Taylor step's start, a segment's state), not moved
    Copied,
    /// kept by the equations (`Work::aux`, an endpoint map's `keep`)
    Own,
    /// formed by the equations or the session (Taylor coefficients, end states, an endpoint state)
    Formed,
}

/// Every series of a session's blocks in a fixed order, with its role: an endpoint map's kept series and its state;
/// or each Taylor step's start, coefficients, scratch series and end state, then the segment's state
fn series_of<'b, T: Lane, const B: usize>(
    kept: &'b mut [Kept<T, B>],
    states: &'b mut [State<T, B>],
) -> Vec<(Role, &'b mut [Fl<T, B>])> {
    let mut out = Vec::new();
    for (kp, st) in kept.iter_mut().zip(states.iter_mut()) {
        let state_role = match kp {
            Kept::End(keep) => {
                out.extend(keep.iter_mut().map(|x| (Role::Own, &mut x[..])));
                Role::Formed
            }
            Kept::Seg(works, ends) => {
                for (w, e) in works.iter_mut().zip(ends.iter_mut()) {
                    for c in w.c.iter_mut() {
                        let n = c.len().min(1);
                        let (start, rest) = c.split_at_mut(n);
                        out.push((Role::Copied, start));
                        out.push((Role::Formed, rest));
                    }
                    out.extend(w.aux.iter_mut().map(|x| (Role::Own, &mut x[..])));
                    out.push((Role::Formed, &mut e[..]));
                }
                Role::Copied
            }
        };
        out.push((state_role, &mut st[..]));
    }
    out
}

/// The two-coefficient session of the tangent update at z_0, and coefficient 1 of each of its series (`series_of`'s
/// order) as formed with z_1 zero
struct Pair<'a, E: Equations<B>, P: Precision<Native>, const B: usize> {
    s: JetSession<'a, E, P, 2, B>,
    base: Vec<Fl<P::V, B>>,
}

/// The residual's series in the equations' parameter formed one order at a time on `P::J<N>` lanes
/// ([`Shooting::jet_session`]): each [`JetSession::next`] takes the next coefficient vector z_k and returns order
/// k + 1 with z_(k+1) zero, the right-hand side of the next order's solve. Order k was formed with z_k zero by the call
/// before, so it is formed again with z_k first; each call forms two orders. With the equations' order-k methods each
/// order costs about one evaluation at one coefficient; with their defaults, one at N. Order j of every block's state
/// and of the residual is bit for bit `residual_jet`'s with the same coefficient vectors, unless
/// [`JetSession::tangent_from`] moves the orders by their tangents instead of forming them again.
pub struct JetSession<'a, E: Equations<B>, P: Precision<Native>, const N: usize, const B: usize> {
    sh: &'a Shooting<E, B>,
    prep: Option<E::Prep>,
    /// the first order moved by its tangent once its coefficient vector is given; None: every order formed again
    tangent: Option<usize>,
    /// the tangent update's two-coefficient session, made at its first update
    pair: Option<Box<Pair<'a, E, P, B>>>,
    /// the coefficient vectors given, z_0..z_(k-1)
    zs: Vec<Vec<Vec<f64>>>,
    /// each block's inputs and the globals, with the coefficients given
    inputs: Vec<Vec<Fl<P::J<N>, B>>>,
    g: Vec<P::J<N>>,
    kept: Vec<Kept<P::J<N>, B>>,
    /// each block's state, its orders formed
    states: Vec<State<P::J<N>, B>>,
}

impl<E: Equations<B>, const B: usize> Shooting<E, B> {
    /// A [`JetSession`] on `P::J<N>` lanes, for orders up to N - 1, with no coefficient vector given yet
    pub fn jet_session<P: Precision<Native>, const N: usize>(&self) -> JetSession<'_, E, P, N, B> {
        let nb = self.k() + 1;
        let zero = Fl::<P::J<N>, B>::zero();
        let kept = (0..nb)
            .map(|b| {
                if b == 0 || b == nb - 1 {
                    Kept::End(Vec::new())
                } else {
                    let steps = self.segs[b - 1].len().saturating_sub(1);
                    Kept::Seg(
                        (0..steps).map(|_| Work::new(E::NF, E::NAUX, self.kt)).collect(),
                        vec![vec![zero; E::NF]; steps],
                    )
                }
            })
            .collect();
        JetSession {
            sh: self,
            prep: None,
            tangent: None,
            pair: None,
            zs: Vec::new(),
            inputs: (0..nb).map(|b| vec![zero; self.block_inputs(b).len()]).collect(),
            g: vec![<P::J<N> as BaseLane>::zero(); E::NG],
            kept,
            states: vec![vec![zero; E::NF]; nb],
        }
    }
}

impl<'a, E: Equations<B>, P: Precision<Native>, const N: usize, const B: usize> JetSession<'a, E, P, N, B> {
    /// the number of coefficient vectors given
    pub fn given(&self) -> usize {
        self.zs.len()
    }

    /// From order `from` on (at least 1), order k is formed once, with z_k zero, and moved by its tangent once z_k is
    /// given, in place of being formed again (the module's documentation): each series' coefficient k plus the change
    /// of order 1 of a session on `P::J<2>` at z_0, seeded with z_k scaled by a power of two toward z_1's size. The
    /// orders then agree with `residual_jet`'s to rounding, not bit for bit; those below `from` are formed again, and
    /// None (the default) forms every order again. The update costs about two order-units against k + 1 for forming
    /// order k again, and the pair session's first orders about three once.
    pub fn tangent_from(&mut self, from: Option<usize>) {
        self.tangent = from.map(|f| f.max(1));
    }

    /// Takes z_k (k = `given()`, P::PARTS part arrays of length `len()`) and returns order k + 1 of the residual's
    /// series with z_(k+1) zero, one scalar per row. The preparation is formed from z_0 at the first call.
    pub fn next(&mut self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        let k = self.zs.len();
        if k + 2 > N {
            return Err(format!(
                "JetSession: order {} past the {N} coefficients of its jets",
                k + 1
            ));
        }
        let sh = self.sh;
        sh.check_z::<P>(z)?;
        if k == 0 {
            self.prep = Some(sh.prep(z)?);
        }
        for (b, inp) in self.inputs.iter_mut().enumerate() {
            for (f, &o) in sh.block_inputs(b).iter().enumerate() {
                P::field_set_coef(&mut inp[f], k, &sh.load::<P>(z, o));
            }
        }
        let og = sh.off_global();
        for (i, gi) in self.g.iter_mut().enumerate() {
            P::set_coef(gi, k, P::splat(sh.scalar::<P>(z, og + i)));
        }
        self.zs.push(z.iter().map(|p| p.to_vec()).collect());
        #[cfg(feature = "mutate")]
        let stale = sh.control == crate::control::Control::JetStale && k > 0;
        #[cfg(not(feature = "mutate"))]
        let stale = false;
        if !stale {
            match self.tangent {
                Some(from) if k >= from => self.update(k, par)?,
                _ => self.form(k, par),
            }
        }
        self.form(k + 1, par);
        self.order(k + 1)
    }

    /// Moves order j >= 1 of every series by its tangent: the pair session's order 1, formed again with the inputs' and
    /// the globals' coefficient 1 set to z_j times 2^e, less its order 1 as formed with them zero, times 2^-e. The
    /// power of two brings z_j toward z_1's size, so that the parameter's part, which the difference removes, is not
    /// much larger than what it leaves. A step's start and a segment's state are then copied from their sources.
    fn update(&mut self, j: usize, par: bool) -> Result<(), String> {
        let sh = self.sh;
        // the largest part of a vector, NaN if any is
        let lead = |z: &[Vec<f64>]| {
            z.iter().flatten().fold(0.0f64, |a, &v| {
                if a.is_nan() || v.is_nan() {
                    f64::NAN
                } else {
                    a.max(v.abs())
                }
            })
        };
        let (m1, mj) = (lead(&self.zs[1]), lead(&self.zs[j]));
        if mj == 0.0 {
            // z_j zero: order j is as it was formed
            return Ok(());
        }
        let ex = |x: f64| x.log2().floor() as i32;
        let e = if !(mj.is_finite() && m1.is_finite()) {
            0
        } else if m1 > 0.0 {
            (ex(m1) - ex(mj)).clamp(-960, 960)
        } else {
            (-ex(mj)).clamp(-960, 960)
        };
        let (up, down) = (2f64.powi(e), 2f64.powi(-e));
        // a seed that overflows or loses a part to underflow cannot be scaled back: form the order again
        if !self.zs[j].iter().flatten().all(|&p| {
            let q = p * up;
            q.is_finite() && q * down == p
        }) {
            self.form(j, par);
            return Ok(());
        }
        if self.pair.is_none() {
            let mut s = sh.jet_session::<P, 2>();
            let z0: Vec<&[f64]> = self.zs[0].iter().map(|p| p.as_slice()).collect();
            s.next(&z0, par)?;
            let base = series_of(&mut s.kept, &mut s.states)
                .into_iter()
                .flat_map(|(_, x)| x.iter().map(|f| P::field_coef(f, 1)).collect::<Vec<_>>())
                .collect();
            self.pair = Some(Box::new(Pair { s, base }));
        }
        let Pair { s: t, base } = &mut **self.pair.as_mut().expect("made above");
        let zj: Vec<&[f64]> = self.zs[j].iter().map(|p| p.as_slice()).collect();
        for (b, inp) in t.inputs.iter_mut().enumerate() {
            for (f, &o) in sh.block_inputs(b).iter().enumerate() {
                P::field_set_coef(&mut inp[f], 1, &sh.load::<P>(&zj, o).mul_pwr2(up));
            }
        }
        let og = sh.off_global();
        for (i, gi) in t.g.iter_mut().enumerate() {
            P::set_coef(gi, 1, P::splat(sh.scalar::<P>(&zj, og + i)).mul_pwr2(up));
        }
        t.form(1, par);
        #[cfg(feature = "mutate")]
        let own_too = sh.control != crate::control::Control::JetTangentKeep;
        #[cfg(not(feature = "mutate"))]
        let own_too = true;
        let mine = series_of(&mut self.kept, &mut self.states);
        let pair = series_of(&mut t.kept, &mut t.states);
        if mine.len() != pair.len()
            || mine
                .iter()
                .zip(&pair)
                .any(|(a, b)| a.0 != b.0 || a.1.len() != b.1.len())
        {
            return Err("JetSession: the tangent's two-coefficient session keeps series of other shapes".to_string());
        }
        if base.len() != pair.iter().map(|(_, d)| d.len()).sum::<usize>() {
            return Err("JetSession: the tangent's two-coefficient session changed its series since z_0".to_string());
        }
        let mut base = base.iter();
        for ((role, a), (_, d)) in mine.into_iter().zip(pair) {
            for (x, y) in a.iter_mut().zip(d.iter()) {
                let b0 = base.next().expect("one per series entry");
                if role == Role::Formed || (role == Role::Own && own_too) {
                    let c = P::field_coef(x, j) + (P::field_coef(y, 1) - *b0).mul_pwr2(down);
                    P::field_set_coef(x, j, &c);
                }
            }
        }
        for (b, (kp, st)) in self.kept.iter_mut().zip(self.states.iter_mut()).enumerate() {
            if let Kept::Seg(works, ends) = kp {
                let inp = &self.inputs[b];
                for (i, w) in works.iter_mut().enumerate() {
                    let src: &[Fl<P::J<N>, B>] = if i == 0 { inp } else { &ends[i - 1] };
                    for (f, sf) in src.iter().enumerate() {
                        P::field_set_coef(&mut w.c[f][0], j, &P::field_coef(sf, j));
                    }
                }
                let last = ends.last().map_or(inp.as_slice(), |e| e.as_slice());
                for (sf, df) in last.iter().zip(st.iter_mut()) {
                    P::field_set_coef(df, j, &P::field_coef(sf, j));
                }
            }
        }
        Ok(())
    }

    /// Forms order j of every block's state from the inputs' and the globals' coefficients 0..=j and the lower
    /// orders kept
    fn form(&mut self, j: usize, par: bool) {
        let JetSession {
            sh,
            prep,
            inputs,
            g,
            kept,
            states,
            ..
        } = self;
        let (sh, prep, inputs, g) = (*sh, prep.as_ref().expect("formed after z_0"), &*inputs, &*g);
        let k = sh.k();
        let one = |(b, (kp, st)): BlockMut<'_, P::J<N>, B>| {
            let inp = &inputs[b];
            let mut put = |s: &[Fl<P::J<N>, B>]| {
                for (sf, df) in s.iter().zip(st.iter_mut()) {
                    P::field_set_coef(df, j, &P::field_coef(sf, j));
                }
            };
            match kp {
                Kept::End(keep) => {
                    let s = if b == 0 {
                        sh.eq.left_state_at(sh.nodes[0], inp, g, prep, keep, j)
                    } else {
                        sh.eq.right_state_at(sh.nodes[k - 1], inp, g, prep, keep, j)
                    };
                    put(&s);
                }
                Kept::Seg(works, ends) => {
                    let xs = &sh.segs[b - 1];
                    for i in 0..works.len() {
                        let (done, rest) = ends.split_at_mut(i);
                        let src: &[Fl<P::J<N>, B>] = if i == 0 { inp } else { &done[i - 1] };
                        let w = &mut works[i];
                        for (f, sf) in src.iter().enumerate() {
                            P::field_set_coef(&mut w.c[f][0], j, &P::field_coef(sf, j));
                        }
                        sh.eq.taylor_coeffs_at(xs[i], g, w, j);
                        let h = DD::from_f64(xs[i + 1]) - DD::from_f64(xs[i]);
                        // the step's sum at coefficient j alone: a jet's Horner step multiplies and adds coefficient
                        // by coefficient, in the value lanes' arithmetic, so this is the full sum's coefficient j
                        let mut cj = Vec::with_capacity(w.kt + 1);
                        for (f, ef) in rest[0].iter_mut().enumerate() {
                            cj.clear();
                            cj.extend(w.c[f][..=w.kt].iter().map(|c| P::field_coef(c, j)));
                            P::field_set_coef(ef, j, &horner(&cj, h));
                        }
                    }
                    put(ends.last().map_or(inp.as_slice(), |e| e.as_slice()));
                }
            }
        };
        if par {
            kept.par_iter_mut().zip(states.par_iter_mut()).enumerate().for_each(one);
        } else {
            kept.iter_mut().zip(states.iter_mut()).enumerate().for_each(one);
        }
    }

    /// Order j <= `given()` of the residual's series as formed, one scalar per row: with z_j when it is given, and
    /// with it zero for j = `given()`. Bit for bit order j of `residual_jet` on the vectors given.
    pub fn order(&self, j: usize) -> Result<Vec<P::Scalar>, String> {
        let sh = self.sh;
        if self.zs.is_empty() || j > self.zs.len() {
            return Err(format!(
                "JetSession: order {j} not formed ({} vectors given)",
                self.zs.len()
            ));
        }
        let m = Shooting::<E, B>::M;
        let zj: Option<Vec<&[f64]>> = self.zs.get(j).map(|z| z.iter().map(|p| p.as_slice()).collect());
        let fld = |o: usize| -> Fl<P::V, B> {
            match &zj {
                Some(z) => sh.load::<P>(z, o),
                None => Fl::zero(),
            }
        };
        let mut ok = vec![P::ZERO; sh.rows()];
        for (b, s) in self.states.iter().enumerate() {
            let t = sh.block_target(b);
            let r0 = sh.row_block(b);
            for q in 0..E::NF {
                let d = P::field_coef(&s[q], j) - fld(t + q * m);
                for i in 0..m {
                    ok[r0 + q * m + i] = P::field_get(&d, i);
                }
            }
        }
        sh.project_residual::<P>(&mut ok, |p| {
            (0..m)
                .map(|i| match &zj {
                    Some(z) => sh.scalar::<P>(z, p.pin + i),
                    None => P::ZERO,
                })
                .collect()
        });
        let re = sh.row_extra();
        let z0: Vec<&[f64]> = self.zs[0].iter().map(|p| p.as_slice()).collect();
        for (r, row) in sh.extra.iter().enumerate() {
            ok[re + r] = if j == 0 {
                sh.extra_row::<P>(&z0, r)
            } else if let Some(z) = &zj {
                P::dot(row.terms.iter().map(|&(i, w)| (P::from_dd(w), sh.scalar::<P>(z, i))))
            } else {
                P::ZERO
            };
        }
        sh.scale_rows::<P>(&mut ok)?;
        Ok(ok)
    }
}

/// The session as multisolver's series of a branch takes its orders ([`multisolver::JetOrders`])
impl<E: Equations<B>, P: Precision<Native>, const N: usize, const B: usize> multisolver::JetOrders<P>
    for JetSession<'_, E, P, N, B>
{
    fn next(&mut self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        JetSession::next(self, z, par)
    }
}
