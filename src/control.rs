//! Deliberate breakage of the engine, for the controls of its accuracy checks (feature `mutate`): each check is run
//! once with one of these and must fail. Set on a problem with `Shooting::set_control`; without the feature the
//! engine has no hooks.

/// A deliberate error in the residual or the Jacobian
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    /// none
    None,
    /// the Jacobian's column c without its block derivative (only the -1 of its target and the extra rows' weights
    /// remain), in the Jacobian and in the Jacobian-vector product (v's entry c not seeded into its block)
    DropColumn(usize),
    /// residual row r matched to the unknown after its target instead of its target
    ShiftRow(usize),
    /// the state read in the first half of its parts only (a quad-double state in double-double, an octo-double one
    /// in quad-double), by the residual and the Jacobian
    HalfParts,
    /// the unknowns outside the fields (extra rows, globals, pinned unknowns) renormalized by `P::new` when read,
    /// where the fields read their parts as they are
    RenormScalar,
    /// the mode matrix with the growth rate's real part only, in the Taylor steps and the preparation
    RateReal,
    /// the mode matrix with the Taylor steps at rate zero (`Work::rate` not set)
    RateNotInSteps,
    /// the mode matrix with the preparation of `Equations::try_prepare`, without the rate
    RateNotPrepared,
    /// every evaluation prepared by the infallible `Equations::prepare` and `prepare_rate` in place of `try_prepare`
    /// and `try_prepare_rate`: a failed preparation goes unseen, and the entry points use `prepare`'s values
    InfalliblePrepare,
    /// the projected rows' projectors read in double-double (their first two parts) in every precision
    ProjectorDD,
    /// the staircase LU with one pivot of its middle step scaled by 1 + 1e-6 (multidouble's `StaircaseControl::Pivot`)
    StaircasePivot,
    /// the staircase LU with the copies of the global untied from it (`StaircaseControl::Link`)
    StaircaseLink,
    /// the jet session ([`crate::jet::JetSession`]) without its second forming of order k once z_k is given: the
    /// order keeps what it was with z_k zero
    JetStale,
    /// the jet session's tangent update ([`crate::jet::JetSession::tangent_from`]) without the series the equations
    /// keep (`Work::aux`, an endpoint map's `keep`): the later orders read their coefficient k as it was with z_k
    /// zero
    JetTangentKeep,
    /// the multi-tangent Jacobian (`Shooting::jacobian_tangents`) with each evaluation's tangents read in reverse
    /// order, so that its columns land in one another's places
    TangentOrder,
    /// the order choice (`Shooting::plan_segments_auto`) keeping its lowest candidate order whatever the estimates,
    /// planned in full; the candidates' estimates are reported as without it
    FirstOrder,
    /// the multi-tangent Jacobian with each evaluation's columns read one tangent early: the first from the value,
    /// column k from tangent k - 1 (with one tangent too, unlike `TangentOrder`)
    TangentShift,
    /// the sub-Jacobian (`Shooting::jacobian_sub`) without the first column task it needs, and the sub-residual
    /// (`Shooting::residual_sub`) without the first block it needs: their entries read as if no evaluation wrote them
    SubSkip,
    /// the family view (`Shooting::in_global`) with its Jacobian's columns read past the column before the
    /// parameter's instead of past the parameter's, so that dF/dp stands among dF/dz
    FamilyColumn,
    /// the corrector's bordered factors (`InGlobal::bordered_lu`) with the layout's dropped row kept in place of the
    /// tangent
    BorderPin,
    /// the periodic solve ([`crate::series::PeriodicLu::solve`]) with the derivative of its matrix dropped from the
    /// right-hand side of a tangent and of a jet's orders past 0: A_0 f_d = r_d in place of A_0 f_d = r_d - A_d f_v
    PeriodicCoupling,
    /// the periodic solve at a nonzero rate with its tangents solved by the factors at rate zero
    RateUnshifted,
    /// the periodic solve at a nonzero rate with the refinement residual of its quad- and octo-double tangents formed
    /// without the rate's shift, so that the refinement converges to the tangent of the system at rate zero
    RateResidual,
    /// the Moebius variable ([`crate::mobius::Mobius`]) with the sign of a flipped in w(tau) and in the products with
    /// w, its derivative polynomials unchanged
    MobiusSign,
    /// the products with w ([`crate::mobius::Mobius::mul_w`]) with the recurrence's coefficient a t_l times 1 + 2^-60
    MobiusRecurrence,
    /// the Moebius variable's constants rounded to double-double, so that a quad- or octo-double lane reads them to
    /// 2^-106 only
    MobiusConstDD,
}

/// A deliberate error of the plane stepper ([`crate::plane::Plane::set_control`])
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneControl {
    /// none
    None,
    /// the Taylor sum of every step cut at order m while the coefficients are formed to the plan's order
    LowSum(usize),
    /// the reciprocals 1 / m of the recursion formed in double-double and widened to the lane
    RecipDD,
    /// no orthogonalization between steps: each column only scaled to unit length after a step, its `ln` of the
    /// length added to the log-volume, and one orthonormalization at the end of `advance` with its `ln R_ii`
    ScaleOnly,
    /// the volume of the starting basis left out of `ln_vol`
    NoGram,
    /// the last column's `ln R_kk` left out of every step's term of `ln_vol`
    LogDrop,
    /// no orthonormalization after the steps i, i + 1, ..., i + m - 1 (counted from 0; never after the last step of
    /// the advance), and so no `ln R_ii` of them: in exact arithmetic the next orthonormalization carries what these
    /// left, and `SkipQr(i, 1)` skips one
    SkipQr(usize, usize),
    /// the projections of the orthonormalization use the product without the conjugate (and the norm the sum of z z, not
    /// |z|^2): on a complex plane the phase of the projection coefficients and of the norms is lost, and the plane's
    /// columns come back in a gauge without the rotation the system gave them (a real plane is unchanged)
    NoConj,
}

/// A deliberate error of the Taylor table ([`crate::table`]): `Wide`, `FractionOne`, `LooseTail`, `IgnoreValidTo` and
/// `Unscaled` act when the table is built (`Opts::control`), `NoShift` and `ShiftF64` in its lookup
/// (`Table::set_control`)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableControl {
    /// none
    None,
    /// every step runs to the end of the source's piece, of the path or to `hmax`, the width ignored
    Wide,
    /// `fraction` 1: a step is the whole width, with no room for a consumer's step
    FractionOne,
    /// `tail` 1e-5 in place of the setting
    LooseTail,
    /// the end of the source's piece not looked at, so that a step can run across it
    IgnoreValidTo,
    /// the tail test formed in plain `f64` products, which underflow where a coefficient is large and `h^k` small
    Unscaled,
    /// the lookup does not shift: the series about the step's start is read as the series about `x0`
    NoShift,
    /// the shift's `delta` formed as the difference of two `f64` numbers in `f64` and widened, where it is exact in the
    /// lane's own type
    ShiftF64,
}

/// A deliberate error in the implicit-ODE recursion or walk ([`crate::implicit`]), the controls of its planted checks
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImplicitControl {
    /// none
    None,
    /// `y''_k` left zero at order k, so that order k of N is not imposed and N drifts
    SkipOrder(usize),
    /// 1/k and the reciprocal of A0 rounded to double-double
    RecipDD,
    /// A0 = 0 not tested before the division
    NoPivotCheck,
    /// the sign change of A0 between two step starts not looked for
    NoCrossing,
    /// the zero of A0 estimated with the exponent 1/2 fixed
    ExponentHalf,
    /// `pivot_abs` ignored
    NoPivotStop,
    /// `l^j` rounded to `f64` before it scales a jet's coefficient, as it underflows or overflows where the product
    /// does not
    PlainScale,
    /// `Walker::advance` sums the series the last `form` left, whether or not it was formed for the state
    StaleSeries,
    /// the crossing of the step before appended at every `form`, not once
    RepeatCrossing,
    /// the end of the last accepted step of a `run` not looked at for a crossing
    NoEndCrossing,
    /// the crossing at the end of a `run`'s last step recorded in the walk, not only in the result
    RecordEnd,
    /// equal checkpoints of a `run` not merged
    NoDedup,
    /// the estimate of the zero returned with every `run` after a pivot stop, not only the one that stopped there
    KeepZero,
    /// `taylor_shift` without its return on an empty polynomial
    EmptyShift,
    /// the cached reciprocal of A0 used where it overflows (a pivot below 2^-1023 in `f64`)
    PlainReciprocal,
    /// the exponent of the zero estimate formed as a0 a2 / a1^2 of the coefficients of A0 as they are: not scaled by a
    /// power of two, with no shortcut for a linear A0 and no quotient form where the squares underflow
    PlainRho,
    /// the quadratic coefficient of the zero estimate read from the scaled coefficients only, so that one that the
    /// scaling rounds to zero makes the pivot look linear
    ScaledQuad,
    /// a zero estimate whose coordinate is not finite returned
    NoFiniteZero,
    /// 2 (a0 / a1) formed before it is multiplied by a2 / a1 in the quotient form of the exponent, which overflows
    DoubleFirst,
    /// beta a0 formed before it is divided by a1 in the zero estimate, which overflows or underflows
    ProductFirst,
    /// the coordinate of the zero estimate formed as x + s l with the product s l whole, which overflows
    ProductX,
    /// the bracket of a crossing bisected 80 times and called located whether or not it has closed
    OpenBracket,
    /// a midpoint of the bracket of a crossing that evaluates to exactly zero put on the far side of the bracket
    ZeroMid,
    /// the midpoint of the bracket of a crossing formed as half the sum of its ends, which overflows
    SumMidpoint,
    /// the refinement of a crossing's position kept in its bracket tested as `lo <= next <= hi`, which no value
    /// satisfies on a backward step
    BackwardBracket,
    /// the sign of the last `form` of a state stored for the next comparison, the first one's compared
    LastSign,
    /// a pivot that is not finite at the end of a `run` read for its sign
    NoFiniteEnd,
}
