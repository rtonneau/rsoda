//! The user-facing ODE system trait and a closure adapter.

use crate::error::{InputError, LsodaError};
use crate::sparse::SparsityPattern;

/// An ODE system `y' = f(t, y)` with an optional analytic Jacobian.
pub trait OdeSystem {
    /// Evaluate the right-hand side: write `f(t, y)` into `ydot`.
    fn rhs(&mut self, t: f64, y: &[f64], ydot: &mut [f64]) -> Result<(), LsodaError>;

    /// Evaluate the Jacobian `df/dy` into `pd`, column-major with leading
    /// dimension `nrowpd`: dense `nrowpd = neq`, `pd[i + j*neq] = df_i/dy_j`;
    /// banded `nrowpd = ml + mu + 1`, `pd[(i - j + mu) + j*nrowpd] = df_i/dy_j`.
    /// Only called for user-supplied Jacobian kinds.
    fn jac(&mut self, _t: f64, _y: &[f64], _pd: &mut [f64]) -> Result<(), LsodaError> {
        Err(InputError::MissingJacobian.into())
    }

    /// Whether [`OdeSystem::jac`] is implemented. Used to validate the
    /// requested Jacobian kind.
    fn has_jac(&self) -> bool {
        false
    }

    /// The sparsity pattern of the Jacobian, required by the sparse Jacobian
    /// kinds ([`crate::JacobianKind::SparseUser`],
    /// [`crate::JacobianKind::SparseFd`]). It must not change during the
    /// integration.
    fn sparsity_pattern(&self) -> Option<&SparsityPattern> {
        None
    }

    /// Evaluate the Jacobian `df/dy` at the entries of
    /// [`OdeSystem::sparsity_pattern`]: `values[k]` is `df_i/dy_j` for the
    /// `k`-th stored entry `(i, j)` (CSC order, see
    /// [`SparsityPattern::find`]). `values` is zeroed before the call. Only
    /// called for [`crate::JacobianKind::SparseUser`].
    fn jac_sparse(&mut self, _t: f64, _y: &[f64], _values: &mut [f64]) -> Result<(), LsodaError> {
        Err(InputError::MissingJacobian.into())
    }

    /// Whether [`OdeSystem::jac_sparse`] is implemented.
    fn has_jac_sparse(&self) -> bool {
        false
    }
}

type Callback<'a> = Box<dyn FnMut(f64, &[f64], &mut [f64]) -> Result<(), LsodaError> + 'a>;

/// Adapter implementing [`OdeSystem`] from closures.
pub struct ClosureSystem<'a> {
    rhs: Callback<'a>,
    jac: Option<Callback<'a>>,
    pattern: Option<SparsityPattern>,
    jac_sparse: Option<Callback<'a>>,
}

impl<'a> ClosureSystem<'a> {
    /// System with only a right-hand side (Jacobian by finite differences).
    pub fn new<F>(rhs: F) -> Self
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), LsodaError> + 'a,
    {
        Self { rhs: Box::new(rhs), jac: None, pattern: None, jac_sparse: None }
    }

    /// Add a user-supplied Jacobian closure (layout as in [`OdeSystem::jac`]).
    pub fn with_jac<J>(mut self, jac: J) -> Self
    where
        J: FnMut(f64, &[f64], &mut [f64]) -> Result<(), LsodaError> + 'a,
    {
        self.jac = Some(Box::new(jac));
        self
    }

    /// Declare the Jacobian's sparsity pattern (for
    /// [`crate::JacobianKind::SparseFd`]).
    pub fn with_sparsity(mut self, pattern: SparsityPattern) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Declare the sparsity pattern and add a sparse Jacobian closure (layout
    /// as in [`OdeSystem::jac_sparse`]), for
    /// [`crate::JacobianKind::SparseUser`].
    pub fn with_sparse_jac<J>(mut self, pattern: SparsityPattern, jac: J) -> Self
    where
        J: FnMut(f64, &[f64], &mut [f64]) -> Result<(), LsodaError> + 'a,
    {
        self.pattern = Some(pattern);
        self.jac_sparse = Some(Box::new(jac));
        self
    }
}

impl OdeSystem for ClosureSystem<'_> {
    fn rhs(&mut self, t: f64, y: &[f64], ydot: &mut [f64]) -> Result<(), LsodaError> {
        (self.rhs)(t, y, ydot)
    }

    fn jac(&mut self, t: f64, y: &[f64], pd: &mut [f64]) -> Result<(), LsodaError> {
        match self.jac.as_mut() {
            Some(j) => j(t, y, pd),
            None => Err(InputError::MissingJacobian.into()),
        }
    }

    fn has_jac(&self) -> bool {
        self.jac.is_some()
    }

    fn sparsity_pattern(&self) -> Option<&SparsityPattern> {
        self.pattern.as_ref()
    }

    fn jac_sparse(&mut self, t: f64, y: &[f64], values: &mut [f64]) -> Result<(), LsodaError> {
        match self.jac_sparse.as_mut() {
            Some(j) => j(t, y, values),
            None => Err(InputError::MissingJacobian.into()),
        }
    }

    fn has_jac_sparse(&self) -> bool {
        self.jac_sparse.is_some()
    }
}
