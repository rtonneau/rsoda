//! Solver options, tolerances, Jacobian kinds and the validating builder.

use crate::consts::{MORD_NONSTIFF, MORD_STIFF, MXHNL0, MXSTP0};
use crate::error::{InputError, LsodaError};
use crate::system::OdeSystem;

/// A scalar or per-component tolerance value.
#[derive(Debug, Clone, PartialEq)]
pub enum Tol {
    Scalar(f64),
    PerComponent(Vec<f64>),
}

/// Relative and absolute error tolerances (Fortran `ITOL` 1-4).
#[derive(Debug, Clone, PartialEq)]
pub struct Tolerance {
    pub rtol: Tol,
    pub atol: Tol,
}

impl Tolerance {
    /// Scalar rtol and atol.
    pub fn scalar(rtol: f64, atol: f64) -> Self {
        Self { rtol: Tol::Scalar(rtol), atol: Tol::Scalar(atol) }
    }

    /// Per-component rtol and atol.
    pub fn per_component(rtol: Vec<f64>, atol: Vec<f64>) -> Self {
        Self { rtol: Tol::PerComponent(rtol), atol: Tol::PerComponent(atol) }
    }

    /// `rtol` for component `i`.
    pub fn rtol_at(&self, i: usize) -> f64 {
        match &self.rtol {
            Tol::Scalar(v) => *v,
            Tol::PerComponent(v) => v[i],
        }
    }

    /// `atol` for component `i`.
    pub fn atol_at(&self, i: usize) -> f64 {
        match &self.atol {
            Tol::Scalar(v) => *v,
            Tol::PerComponent(v) => v[i],
        }
    }

    fn validate(&self, neq: usize) -> Result<(), InputError> {
        for (tol, is_rtol) in [(&self.rtol, true), (&self.atol, false)] {
            let vals: &[f64] = match tol {
                Tol::Scalar(v) => std::slice::from_ref(v),
                Tol::PerComponent(v) => {
                    if v.len() != neq {
                        return Err(InputError::ToleranceLength { expected: neq, got: v.len() });
                    }
                    v
                }
            };
            for (index, &value) in vals.iter().enumerate() {
                if !(value >= 0.0) {
                    return Err(if is_rtol {
                        InputError::NegativeRtol { index, value }
                    } else {
                        InputError::NegativeAtol { index, value }
                    });
                }
            }
        }
        Ok(())
    }
}

/// How the Jacobian is obtained and stored (Fortran `JT` 1, 2, 4, 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianKind {
    /// `JT = 1`: user-supplied dense.
    DenseUser,
    /// `JT = 2`: finite-difference dense (default).
    DenseFd,
    /// `JT = 4`: user-supplied banded.
    BandedUser { ml: usize, mu: usize },
    /// `JT = 5`: finite-difference banded.
    BandedFd { ml: usize, mu: usize },
}

impl JacobianKind {
    /// Whether the user must supply the Jacobian.
    pub fn is_user(&self) -> bool {
        matches!(self, Self::DenseUser | Self::BandedUser { .. })
    }
}

/// Validated solver options.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub neq: usize,
    pub tolerance: Tolerance,
    pub jacobian: JacobianKind,
    /// Initial step size; `0.0` lets the solver choose.
    pub h0: f64,
    /// Maximum |h|; `f64::MAX` means effectively unbounded.
    pub hmax: f64,
    /// Minimum |h|.
    pub hmin: f64,
    /// Report data at each method switch (`IXPR`).
    pub ixpr: bool,
    pub mxstep: usize,
    pub mxhnil: usize,
    /// Max Adams order (already capped to 12).
    pub mxordn: usize,
    /// Max BDF order (already capped to 5).
    pub mxords: usize,
}

/// Builder applying the Fortran defaults and DLSODA input checks.
#[derive(Debug, Clone)]
pub struct OptionsBuilder {
    neq: usize,
    tolerance: Tolerance,
    jacobian: JacobianKind,
    h0: f64,
    hmax: f64,
    hmin: f64,
    ixpr: bool,
    mxstep: usize,
    mxhnil: usize,
    mxordn: usize,
    mxords: usize,
}

impl OptionsBuilder {
    /// Start a builder for `neq` equations with the Fortran defaults.
    /// Fortran requires tolerances from the caller; scalar 1e-6 is used
    /// here until [`OptionsBuilder::tolerance`] is called.
    pub fn new(neq: usize) -> Self {
        Self {
            neq,
            tolerance: Tolerance::scalar(1e-6, 1e-6),
            jacobian: JacobianKind::DenseFd,
            h0: 0.0,
            hmax: f64::MAX,
            hmin: 0.0,
            ixpr: false,
            mxstep: MXSTP0,
            mxhnil: MXHNL0,
            mxordn: MORD_NONSTIFF,
            mxords: MORD_STIFF,
        }
    }

    pub fn tolerance(mut self, tolerance: Tolerance) -> Self {
        self.tolerance = tolerance;
        self
    }
    pub fn jacobian(mut self, jacobian: JacobianKind) -> Self {
        self.jacobian = jacobian;
        self
    }
    pub fn h0(mut self, h0: f64) -> Self {
        self.h0 = h0;
        self
    }
    pub fn hmax(mut self, hmax: f64) -> Self {
        self.hmax = hmax;
        self
    }
    pub fn hmin(mut self, hmin: f64) -> Self {
        self.hmin = hmin;
        self
    }
    pub fn ixpr(mut self, ixpr: bool) -> Self {
        self.ixpr = ixpr;
        self
    }
    /// Maximum steps per call; `0` selects the default (10000), as in Fortran.
    pub fn mxstep(mut self, mxstep: usize) -> Self {
        self.mxstep = if mxstep == 0 { MXSTP0 } else { mxstep };
        self
    }
    /// Maximum `T + H = T` warnings; `0` selects the default (10).
    pub fn mxhnil(mut self, mxhnil: usize) -> Self {
        self.mxhnil = if mxhnil == 0 { MXHNL0 } else { mxhnil };
        self
    }
    /// Max Adams order; `0` means the cap (12), larger values are reduced to 12.
    pub fn mxordn(mut self, mxordn: usize) -> Self {
        self.mxordn = if mxordn == 0 { MORD_NONSTIFF } else { mxordn.min(MORD_NONSTIFF) };
        self
    }
    /// Max BDF order; `0` means the cap (5), larger values are reduced to 5.
    pub fn mxords(mut self, mxords: usize) -> Self {
        self.mxords = if mxords == 0 { MORD_STIFF } else { mxords.min(MORD_STIFF) };
        self
    }

    /// Validate and build. `system` is consulted only to check that a
    /// user-supplied Jacobian kind has a Jacobian available.
    pub fn build<S: OdeSystem + ?Sized>(self, system: &S) -> Result<Options, LsodaError> {
        let neq = self.neq;
        if neq == 0 {
            return Err(InputError::NeqNotPositive.into());
        }
        self.tolerance.validate(neq)?;
        if let JacobianKind::BandedUser { ml, mu } | JacobianKind::BandedFd { ml, mu } =
            self.jacobian
        {
            if ml >= neq {
                return Err(InputError::IllegalMl { ml, neq }.into());
            }
            if mu >= neq {
                return Err(InputError::IllegalMu { mu, neq }.into());
            }
        }
        if self.jacobian.is_user() && !system.has_jac() {
            return Err(InputError::MissingJacobian.into());
        }
        if self.h0.is_nan() {
            return Err(InputError::InvalidH0(self.h0).into());
        }
        if !(self.hmax >= 0.0) {
            return Err(InputError::NegativeHmax(self.hmax).into());
        }
        if !(self.hmin >= 0.0) {
            return Err(InputError::NegativeHmin(self.hmin).into());
        }
        Ok(Options {
            neq,
            tolerance: self.tolerance,
            jacobian: self.jacobian,
            h0: self.h0,
            hmax: self.hmax,
            hmin: self.hmin,
            ixpr: self.ixpr,
            mxstep: self.mxstep,
            mxhnil: self.mxhnil,
            mxordn: self.mxordn,
            mxords: self.mxords,
        })
    }
}

#[cfg(test)]
mod options_tests {
    use super::*;
    use crate::system::ClosureSystem;

    fn sys() -> ClosureSystem<'static> {
        ClosureSystem::new(|_, _, _| Ok(()))
    }
    fn illegal(r: Result<Options, LsodaError>) -> InputError {
        match r {
            Err(LsodaError::IllegalInput(e)) => e,
            other => panic!("expected IllegalInput, got {other:?}"),
        }
    }

    #[test]
    fn defaults() {
        let o = OptionsBuilder::new(3).build(&sys()).unwrap();
        assert_eq!(o.jacobian, JacobianKind::DenseFd);
        assert_eq!((o.h0, o.hmin, o.hmax), (0.0, 0.0, f64::MAX));
        assert_eq!((o.mxstep, o.mxhnil, o.mxordn, o.mxords), (10000, 10, 12, 5));
        assert!(!o.ixpr);
    }

    #[test]
    fn zero_means_default_and_orders_capped() {
        let o = OptionsBuilder::new(2)
            .mxstep(0)
            .mxhnil(0)
            .mxordn(100)
            .mxords(0)
            .build(&sys())
            .unwrap();
        assert_eq!((o.mxstep, o.mxhnil, o.mxordn, o.mxords), (10000, 10, 12, 5));
        let o = OptionsBuilder::new(2).mxordn(4).mxords(3).build(&sys()).unwrap();
        assert_eq!((o.mxordn, o.mxords), (4, 3));
    }

    #[test]
    fn rejects_zero_neq() {
        assert_eq!(illegal(OptionsBuilder::new(0).build(&sys())), InputError::NeqNotPositive);
    }

    #[test]
    fn rejects_negative_tolerances() {
        let r = OptionsBuilder::new(2).tolerance(Tolerance::scalar(-1.0, 0.0)).build(&sys());
        assert_eq!(illegal(r), InputError::NegativeRtol { index: 0, value: -1.0 });
        let t = Tolerance::per_component(vec![1e-6, 1e-6], vec![1e-8, -2.0]);
        let r = OptionsBuilder::new(2).tolerance(t).build(&sys());
        assert_eq!(illegal(r), InputError::NegativeAtol { index: 1, value: -2.0 });
    }

    #[test]
    fn rejects_tolerance_length_mismatch() {
        let t = Tolerance::per_component(vec![1e-6], vec![1e-8, 1e-8]);
        let r = OptionsBuilder::new(2).tolerance(t).build(&sys());
        assert_eq!(illegal(r), InputError::ToleranceLength { expected: 2, got: 1 });
    }

    #[test]
    fn accepts_per_component_tolerance() {
        let t = Tolerance::per_component(vec![1e-6, 1e-7], vec![1e-8, 1e-9]);
        let o = OptionsBuilder::new(2).tolerance(t).build(&sys()).unwrap();
        assert_eq!(o.tolerance.rtol_at(1), 1e-7);
        assert_eq!(o.tolerance.atol_at(0), 1e-8);
    }

    #[test]
    fn rejects_bad_bandwidths() {
        let r = OptionsBuilder::new(3)
            .jacobian(JacobianKind::BandedFd { ml: 3, mu: 0 })
            .build(&sys());
        assert_eq!(illegal(r), InputError::IllegalMl { ml: 3, neq: 3 });
        let r = OptionsBuilder::new(3)
            .jacobian(JacobianKind::BandedFd { ml: 1, mu: 5 })
            .build(&sys());
        assert_eq!(illegal(r), InputError::IllegalMu { mu: 5, neq: 3 });
        assert!(OptionsBuilder::new(3)
            .jacobian(JacobianKind::BandedFd { ml: 2, mu: 2 })
            .build(&sys())
            .is_ok());
    }

    #[test]
    fn user_jacobian_requires_jac() {
        for k in [JacobianKind::DenseUser, JacobianKind::BandedUser { ml: 1, mu: 1 }] {
            let r = OptionsBuilder::new(3).jacobian(k).build(&sys());
            assert_eq!(illegal(r), InputError::MissingJacobian);
            let with = sys().with_jac(|_, _, _| Ok(()));
            assert!(OptionsBuilder::new(3).jacobian(k).build(&with).is_ok());
        }
    }

    #[test]
    fn rejects_negative_step_bounds() {
        let r = OptionsBuilder::new(2).hmax(-1.0).build(&sys());
        assert_eq!(illegal(r), InputError::NegativeHmax(-1.0));
        let r = OptionsBuilder::new(2).hmin(-0.5).build(&sys());
        assert_eq!(illegal(r), InputError::NegativeHmin(-0.5));
    }
}
