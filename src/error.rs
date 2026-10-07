//! Error types: one `LsodaError` variant per negative `istate` of DLSODA.

use std::fmt;

/// Reasons for an illegal-input return (`ISTATE = -3`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum InputError {
    /// `NEQ < 1`.
    NeqNotPositive,
    /// A per-component tolerance vector has the wrong length.
    ToleranceLength { expected: usize, got: usize },
    /// `RTOL(i) < 0` (or NaN).
    NegativeRtol { index: usize, value: f64 },
    /// `ATOL(i) < 0` (or NaN).
    NegativeAtol { index: usize, value: f64 },
    /// Banded Jacobian with `ML >= NEQ`.
    IllegalMl { ml: usize, neq: usize },
    /// Banded Jacobian with `MU >= NEQ`.
    IllegalMu { mu: usize, neq: usize },
    /// A user-supplied Jacobian kind was selected but the system provides none.
    MissingJacobian,
    /// `HMAX < 0` (or NaN).
    NegativeHmax(f64),
    /// `HMIN < 0` (or NaN).
    NegativeHmin(f64),
    /// `H0` is NaN.
    InvalidH0(f64),
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NeqNotPositive => write!(f, "NEQ < 1"),
            Self::ToleranceLength { expected, got } => {
                write!(f, "tolerance vector has length {got}, expected {expected}")
            }
            Self::NegativeRtol { index, value } => write!(f, "RTOL({index}) = {value} < 0.0"),
            Self::NegativeAtol { index, value } => write!(f, "ATOL({index}) = {value} < 0.0"),
            Self::IllegalMl { ml, neq } => write!(f, "ML = {ml} illegal: must be < NEQ = {neq}"),
            Self::IllegalMu { mu, neq } => write!(f, "MU = {mu} illegal: must be < NEQ = {neq}"),
            Self::MissingJacobian => {
                write!(f, "user-supplied Jacobian requested but none provided")
            }
            Self::NegativeHmax(v) => write!(f, "HMAX = {v} < 0"),
            Self::NegativeHmin(v) => write!(f, "HMIN = {v} < 0.0"),
            Self::InvalidH0(v) => write!(f, "H0 = {v} is not a number"),
        }
    }
}

/// Failure of an LSODA call. Variants map to the negative `ISTATE` values.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LsodaError {
    /// `ISTATE = -1`: `mxstep` steps taken before reaching `tout`, at time `t`.
    MaxStepsExceeded { t: f64, mxstep: usize },
    /// `ISTATE = -2`: too much accuracy requested; `tolsf` is the suggested
    /// tolerance scale factor (`RWORK(14)`).
    TooMuchAccuracy { t: f64, tolsf: f64 },
    /// `ISTATE = -3`: illegal input.
    IllegalInput(InputError),
    /// `ISTATE = -4`: repeated error test failures or `|h| = hmin`.
    /// `imxer` is the 0-based component with the largest weighted error.
    ErrorTestFailed { t: f64, h: f64, imxer: usize },
    /// `ISTATE = -5`: repeated convergence failures or `|h| = hmin`.
    ConvergenceFailed { t: f64, h: f64, imxer: usize },
    /// `ISTATE = -6`: error weight became zero for the 0-based `component`.
    ErrorWeightZero { t: f64, component: usize },
    /// `ISTATE = -7`: work arrays too small. Kept for completeness; Rust
    /// allocates its own workspace so this is not normally produced.
    WorkspaceTooSmall { t: f64 },
    /// A user callback (`rhs` or `jac`) reported failure with this code
    /// (the `ierr` argument of the Fortran fork's callbacks).
    Callback { ierr: i32 },
}

impl fmt::Display for LsodaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxStepsExceeded { t, mxstep } => write!(
                f,
                "at t = {t}, {mxstep} steps taken on this call before reaching tout"
            ),
            Self::TooMuchAccuracy { t, tolsf } => write!(
                f,
                "at t = {t}, too much accuracy requested for machine precision (tolsf = {tolsf})"
            ),
            Self::IllegalInput(e) => write!(f, "illegal input: {e}"),
            Self::ErrorTestFailed { t, h, imxer } => write!(
                f,
                "at t = {t}, h = {h}: error test failed repeatedly or with |h| = hmin (component {imxer})"
            ),
            Self::ConvergenceFailed { t, h, imxer } => write!(
                f,
                "at t = {t}, h = {h}: convergence failed repeatedly or with |h| = hmin (component {imxer})"
            ),
            Self::ErrorWeightZero { t, component } => {
                write!(f, "at t = {t}, error weight for component {component} became zero")
            }
            Self::WorkspaceTooSmall { t } => write!(f, "at t = {t}, work array too small"),
            Self::Callback { ierr } => write!(f, "user callback failed with ierr = {ierr}"),
        }
    }
}

impl std::error::Error for LsodaError {}

impl From<InputError> for LsodaError {
    fn from(e: InputError) -> Self {
        Self::IllegalInput(e)
    }
}
