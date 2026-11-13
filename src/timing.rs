//! Linear-solver call counts and wall times (cargo feature `timing`).
//!
//! With the feature on, the stiff corrector times every
//! [`crate::LinearSolver::prepare`] and [`crate::LinearSolver::solve`] call it
//! makes, whatever the solver (dense, band, sparse or custom). The totals are
//! read with [`crate::Lsoda::timing`]. Without the feature this module does
//! not exist and no timer is taken: the results are the same either way.

use std::time::Duration;

/// Call counts and accumulated wall times of the linear solver, since the
/// integration started (or was restarted by a re-initialization).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TimingStats {
    /// Number of [`crate::LinearSolver::prepare`] calls (one per Jacobian
    /// evaluation, so it equals [`crate::Lsoda::nje`]).
    pub n_prepare: usize,
    /// Total time in `prepare`: Jacobian evaluation (user or finite
    /// differences, right-hand-side calls included), assembly of
    /// `P = I - h*el0*J` and its factorization.
    pub t_prepare: Duration,
    /// Number of [`crate::LinearSolver::solve`] calls.
    pub n_solve: usize,
    /// Total time in `solve`.
    pub t_solve: Duration,
}
