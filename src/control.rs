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
    /// the mode matrix with the preparation of `Equations::prepare`, without the rate
    RateNotPrepared,
    /// the projected rows' projectors read in double-double (their first two parts) in every precision
    ProjectorDD,
    /// the staircase LU with one pivot of its middle step scaled by 1 + 1e-6 (multidouble's `StaircaseControl::Pivot`)
    StaircasePivot,
    /// the staircase LU with the copies of the global untied from it (`StaircaseControl::Link`)
    StaircaseLink,
}
