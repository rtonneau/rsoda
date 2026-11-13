//! The linear solver of the stiff corrector: a pluggable owner of the
//! iteration matrix `P = I - h*el0*J`.
//!
//! In the Fortran code, DPRJA forms and factors `P` into the `WM`/`IWM` work
//! arrays and DSOLSY solves with it. Here both steps go through the
//! [`LinearSolver`] trait, so the storage of `P` is not fixed:
//!
//! * [`MatrixSolver`] is the built-in implementation for the dense and band
//!   Jacobian kinds (`MITER = 1, 2, 4, 5`). It runs [`prja`] and
//!   [`IterationMatrix::solve`] unchanged, so its results are those of DPRJA
//!   and DSOLSY.
//! * Users can implement [`LinearSolver`] for their own storage or method
//!   (sparse, block-diagonal, iterative) and select it with
//!   [`JacobianKind::Custom`] through [`crate::Lsoda::with_linear_solver`].

use std::fmt;

use crate::error::LsodaError;
use crate::options::JacobianKind;
use crate::prja::{PrjaInput, PrjaOutput, prja};
use crate::solsy::{IterationMatrix, MatrixStorage, SingularMatrix};
use crate::sparse::SparsityPattern;
use crate::system::OdeSystem;

/// Forms, factors and solves the iteration matrix `P = I - h*el0*J` of the
/// stiff (BDF) corrector.
///
/// The solver is used only while the BDF method is active. Each Jacobian
/// update calls [`LinearSolver::prepare`] once, followed by any number of
/// [`LinearSolver::solve`] calls with the same factors.
///
/// Solvers are `Send` so that the integrator stays `Send`.
pub trait LinearSolver: Send {
    /// Evaluates the Jacobian `J` at the predicted solution, forms
    /// `P = I - h*el0*J` and factors it (`DPRJA`).
    ///
    /// * `kind` is the Jacobian kind in use (`MITER`); it is
    ///   [`JacobianKind::Custom`] for a user-supplied solver.
    /// * `system` gives the right-hand side (for finite differences) and the
    ///   user Jacobian.
    /// * `input` holds the step data; `y` the predicted solution (equal to
    ///   `input.yh1`), which may be perturbed temporarily but must be restored
    ///   before returning; `ftem` (length `>= n`) is scratch space.
    /// * `nje` must be increased by one and `nfe` by the number of
    ///   right-hand-side evaluations.
    ///
    /// Returns the weighted norm of `J` (`PDNORM`, used by the stiffness
    /// test; see [`crate::norms::fnorm`] for the dense definition) and
    /// whether `P` was found singular (then [`LinearSolver::solve`] is not
    /// called and the step is retried, as for a convergence failure).
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &mut self,
        kind: &JacobianKind,
        system: &mut dyn OdeSystem,
        input: &PrjaInput<'_>,
        y: &mut [f64],
        ftem: &mut [f64],
        nfe: &mut usize,
        nje: &mut usize,
    ) -> Result<PrjaOutput, LsodaError>;

    /// Solves `P x = b` in place (`DSOLSY`) with the factors of the last
    /// [`LinearSolver::prepare`]: `x` holds `b` on entry and the solution on
    /// return. `h` and `el0` are the current values (a solver may rescale
    /// its matrix when `h*el0` changed, as DSOLSY's diagonal case does).
    fn solve(&mut self, h: f64, el0: f64, x: &mut [f64]) -> Result<(), SingularMatrix>;
}

impl fmt::Debug for dyn LinearSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("dyn LinearSolver")
    }
}

/// The built-in [`LinearSolver`] for the dense and band Jacobian kinds:
/// [`prja`] and [`IterationMatrix::solve`], with LINPACK LU factors.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixSolver {
    /// The factored iteration matrix.
    pub wm: IterationMatrix,
}

impl MatrixSolver {
    /// Zeroed storage for an `n x n` iteration matrix of Jacobian kind
    /// `kind`, or `None` if `kind` does not use dense or band storage.
    pub fn new(n: usize, kind: &JacobianKind) -> Option<Self> {
        MatrixStorage::for_jacobian(kind).map(|s| Self { wm: IterationMatrix::new(n, s) })
    }
}

impl LinearSolver for MatrixSolver {
    fn prepare(
        &mut self,
        kind: &JacobianKind,
        system: &mut dyn OdeSystem,
        input: &PrjaInput<'_>,
        y: &mut [f64],
        ftem: &mut [f64],
        nfe: &mut usize,
        nje: &mut usize,
    ) -> Result<PrjaOutput, LsodaError> {
        prja(system, *kind, input, y, ftem, &mut self.wm, nfe, nje)
    }

    fn solve(&mut self, h: f64, el0: f64, x: &mut [f64]) -> Result<(), SingularMatrix> {
        self.wm.solve(h, el0, x)
    }
}

/// Whether a solver allocated for Jacobian kind `a` can serve kind `b`
/// (DLSODA reuses `WM` unless the storage changes): the same dense or band
/// storage, both sparse (the pattern is fixed by the system), or both
/// custom.
pub fn same_storage(a: &JacobianKind, b: &JacobianKind) -> bool {
    use JacobianKind::{Custom, SparseFd, SparseUser};
    match (a, b) {
        (Custom, Custom) => true,
        (SparseUser | SparseFd, SparseUser | SparseFd) => true,
        _ => {
            let sa = MatrixStorage::for_jacobian(a);
            sa.is_some() && sa == MatrixStorage::for_jacobian(b)
        }
    }
}

/// The built-in solver for `kind`, or `None` for [`JacobianKind::Custom`]
/// (and for the sparse kinds without the `sparse` feature or without a
/// `pattern`). The sparse kinds use `SparseLu` (feature `sparse`) on `pattern`.
pub fn builtin_solver(
    n: usize,
    kind: &JacobianKind,
    pattern: Option<&SparsityPattern>,
) -> Option<Box<dyn LinearSolver>> {
    match kind {
        JacobianKind::SparseUser | JacobianKind::SparseFd => sparse_solver(pattern),
        _ => MatrixSolver::new(n, kind).map(|s| Box::new(s) as Box<dyn LinearSolver>),
    }
}

#[cfg(feature = "sparse")]
fn sparse_solver(pattern: Option<&SparsityPattern>) -> Option<Box<dyn LinearSolver>> {
    pattern.map(|p| Box::new(crate::sparse_lu::SparseLu::new(p.clone())) as Box<dyn LinearSolver>)
}

#[cfg(not(feature = "sparse"))]
fn sparse_solver(_pattern: Option<&SparsityPattern>) -> Option<Box<dyn LinearSolver>> {
    None
}

/// Adapter turning any (possibly unsized) system into a sized one, so it can
/// be passed to [`LinearSolver::prepare`] as `&mut dyn OdeSystem`.
pub(crate) struct DynSystem<'s, S: ?Sized>(pub &'s mut S);

impl<S: OdeSystem + ?Sized> OdeSystem for DynSystem<'_, S> {
    fn rhs(&mut self, t: f64, y: &[f64], ydot: &mut [f64]) -> Result<(), LsodaError> {
        self.0.rhs(t, y, ydot)
    }

    fn jac(&mut self, t: f64, y: &[f64], pd: &mut [f64]) -> Result<(), LsodaError> {
        self.0.jac(t, y, pd)
    }

    fn has_jac(&self) -> bool {
        self.0.has_jac()
    }

    fn sparsity_pattern(&self) -> Option<&SparsityPattern> {
        self.0.sparsity_pattern()
    }

    fn jac_sparse(&mut self, t: f64, y: &[f64], values: &mut [f64]) -> Result<(), LsodaError> {
        self.0.jac_sparse(t, y, values)
    }

    fn has_jac_sparse(&self) -> bool {
        self.0.has_jac_sparse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_storage_follows_matrix_storage() {
        use JacobianKind::*;
        assert!(same_storage(&DenseUser, &DenseFd));
        assert!(same_storage(&BandedUser { ml: 1, mu: 2 }, &BandedFd { ml: 1, mu: 2 }));
        assert!(!same_storage(&BandedFd { ml: 1, mu: 2 }, &BandedFd { ml: 2, mu: 2 }));
        assert!(!same_storage(&DenseFd, &BandedFd { ml: 0, mu: 0 }));
        assert!(same_storage(&Custom, &Custom));
        assert!(!same_storage(&Custom, &DenseFd));
        assert!(same_storage(&SparseUser, &SparseFd));
        assert!(!same_storage(&SparseFd, &DenseFd));
        assert!(builtin_solver(3, &Custom, None).is_none());
        assert!(builtin_solver(3, &DenseFd, None).is_some());
        assert!(builtin_solver(3, &SparseFd, None).is_none());
    }
}
