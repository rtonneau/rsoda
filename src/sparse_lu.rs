//! [`SparseLu`]: the built-in [`LinearSolver`] of the sparse Jacobian kinds,
//! a sparse LU factorization with partial pivoting from `faer` (cargo
//! feature `sparse`).
//!
//! The pattern of `P = I - h*el0*J` is fixed, so the fill-reducing column
//! ordering (COLAMD) is computed once, when the solver is created. Each
//! Jacobian update assembles `P` in the pattern's CSC order (see
//! [`crate::sparse`]) and refactors it numerically with faer's simplicial
//! left-looking LU, which picks the largest remaining entry of each column
//! as the pivot. The factor storage and the work buffer are reused between
//! updates.

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::perm::PermRef;
use faer::sparse::linalg::LuError;
use faer::sparse::linalg::colamd;
use faer::sparse::linalg::lu::simplicial::{
    SimplicialLu, factorize_simplicial_numeric_lu, factorize_simplicial_numeric_lu_scratch, solve_in_place_scratch,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par};

use crate::error::LsodaError;
use crate::linsol::LinearSolver;
use crate::options::JacobianKind;
use crate::prja::{PrjaInput, PrjaOutput};
use crate::solsy::SingularMatrix;
use crate::sparse::{ColumnColoring, SparsityPattern, fd_jacobian, finish_iteration_matrix, scale_jacobian};
use crate::system::OdeSystem;

/// Sparse LU [`LinearSolver`] for [`JacobianKind::SparseUser`] and
/// [`JacobianKind::SparseFd`].
///
/// Used through a [`JacobianKind::Custom`] solver
/// ([`crate::Lsoda::with_linear_solver`]), it obtains `J` as set by
/// [`SparseLu::with_user_jacobian`] (finite differences by default).
pub struct SparseLu {
    pattern: SparsityPattern,
    coloring: ColumnColoring,
    /// Whether a [`JacobianKind::Custom`] caller wants the user Jacobian.
    user: bool,
    /// Values of `P` (pattern order).
    values: Vec<f64>,
    rowsum: Vec<f64>,
    col_perm_fwd: Vec<usize>,
    col_perm_inv: Vec<usize>,
    row_perm_fwd: Vec<usize>,
    row_perm_inv: Vec<usize>,
    lu: SimplicialLu<usize, f64>,
    mem: MemBuffer,
}

impl std::fmt::Debug for SparseLu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseLu")
            .field("n", &self.pattern.n())
            .field("nnz", &self.pattern.nnz())
            .field("ncolors", &self.coloring.ncolors())
            .finish_non_exhaustive()
    }
}

impl SparseLu {
    /// A solver for iteration matrices on `pattern`: computes the column
    /// coloring and the COLAMD column ordering, and allocates the work
    /// buffers.
    ///
    /// # Panics
    ///
    /// If faer cannot allocate its work space.
    pub fn new(pattern: SparsityPattern) -> Self {
        let n = pattern.n();
        let nnz = pattern.nnz();
        let mut col_perm_fwd = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        {
            let mut mem = MemBuffer::new(colamd::order_scratch::<usize>(n, n, nnz));
            let a = SymbolicSparseColMatRef::new_checked(n, n, pattern.col_ptr(), None, pattern.row_idx());
            colamd::order(&mut col_perm_fwd, &mut col_perm_inv, a, colamd::Control::default(), MemStack::new(&mut mem))
                .expect("COLAMD ordering of the sparsity pattern");
        }
        let req = StackReq::any_of(&[
            factorize_simplicial_numeric_lu_scratch::<usize, f64>(n, n),
            solve_in_place_scratch::<usize, f64>(n, 1, Par::Seq),
        ]);
        Self {
            coloring: pattern.coloring(),
            user: false,
            values: vec![0.0; nnz],
            rowsum: vec![0.0; n],
            col_perm_fwd,
            col_perm_inv,
            row_perm_fwd: vec![0; n],
            row_perm_inv: vec![0; n],
            lu: SimplicialLu::new(),
            mem: MemBuffer::new(req),
            pattern,
        }
    }

    /// For use as a [`JacobianKind::Custom`] solver: take `J` from
    /// [`OdeSystem::jac_sparse`] (`true`) or finite differences (`false`).
    pub fn with_user_jacobian(mut self, user: bool) -> Self {
        self.user = user;
        self
    }

    /// The sparsity pattern of `J` and `P`.
    pub fn pattern(&self) -> &SparsityPattern {
        &self.pattern
    }

    /// The column coloring used for finite differences.
    pub fn coloring(&self) -> &ColumnColoring {
        &self.coloring
    }

    /// Refactors `P` from [`Self::values`]; `true` if `P` is singular (a
    /// structurally missing or numerically zero pivot, as `dgefa`'s `INFO`).
    fn factor(&mut self) -> bool {
        let n = self.pattern.n();
        let sym = SymbolicSparseColMatRef::new_checked(n, n, self.pattern.col_ptr(), None, self.pattern.row_idx());
        let a = SparseColMatRef::new(sym, &self.values);
        let col_perm = PermRef::new_checked(&self.col_perm_fwd, &self.col_perm_inv, n);
        match factorize_simplicial_numeric_lu(
            &mut self.row_perm_fwd,
            &mut self.row_perm_inv,
            &mut self.lu,
            a,
            col_perm,
            MemStack::new(&mut self.mem),
        ) {
            Ok(()) => {}
            Err(LuError::SymbolicSingular { .. }) => return true,
            Err(LuError::Generic(e)) => panic!("sparse LU factorization failed: {e:?}"),
        }
        // The pivot of column j of U is its entry in row j.
        let u = self.lu.u_factor_unsorted();
        (0..n).any(|j| {
            let rows = u.row_idx_of_col_raw(j);
            let vals = u.val_of_col(j);
            !rows.iter().zip(vals).any(|(&i, &v)| i == j && v != 0.0)
        })
    }
}

impl LinearSolver for SparseLu {
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
        let n = self.pattern.n();
        assert!(y.len() >= n && ftem.len() >= n);
        let user = match kind {
            JacobianKind::SparseUser => true,
            JacobianKind::SparseFd => false,
            JacobianKind::Custom => self.user,
            other => panic!("SparseLu cannot serve Jacobian kind {other:?}"),
        };
        *nje += 1;
        let hl0 = input.h * input.el0;
        if user {
            self.values.fill(0.0);
            system.jac_sparse(input.tn, &y[..n], &mut self.values)?;
            scale_jacobian(&mut self.values, hl0);
        } else {
            fd_jacobian(system, &self.pattern, &self.coloring, input, y, ftem, &mut self.values, nfe)?;
        }
        let pdnorm = finish_iteration_matrix(&self.pattern, &mut self.values, input.ewt, hl0, &mut self.rowsum);
        let singular = self.factor();
        Ok(PrjaOutput { pdnorm, singular })
    }

    fn solve(&mut self, _h: f64, _el0: f64, x: &mut [f64]) -> Result<(), SingularMatrix> {
        let n = self.pattern.n();
        let row_perm = PermRef::new_checked(&self.row_perm_fwd, &self.row_perm_inv, n);
        let col_perm = PermRef::new_checked(&self.col_perm_fwd, &self.col_perm_inv, n);
        let rhs = MatMut::from_column_major_slice_mut(&mut x[..n], n, 1);
        self.lu.solve_in_place_with_conj(row_perm, col_perm, Conj::No, rhs, Par::Seq, MemStack::new(&mut self.mem));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::ClosureSystem;

    /// A linear system `f = J y` with a fixed 5x5 sparse `J`.
    fn jentries() -> Vec<(usize, usize, f64)> {
        vec![
            (0, 0, -4.0),
            (1, 0, 1.0),
            (1, 1, -3.0),
            (3, 1, 2.0),
            (0, 2, 0.5),
            (2, 2, -5.0),
            (2, 3, 1.5),
            (3, 3, -2.0),
            (4, 3, 0.25),
            (1, 4, -1.0),
            (4, 4, -6.0),
        ]
    }

    fn system() -> (ClosureSystem<'static>, SparsityPattern) {
        let e = jentries();
        let p = SparsityPattern::from_entries(5, e.iter().map(|&(i, j, _)| (i, j))).unwrap();
        let (e1, e2, p2) = (e.clone(), e, p.clone());
        let sys = ClosureSystem::new(move |_t, y: &[f64], f: &mut [f64]| {
            f.fill(0.0);
            for &(i, j, v) in &e1 {
                f[i] += v * y[j];
            }
            Ok(())
        })
        .with_sparse_jac(p.clone(), move |_t, _y: &[f64], values: &mut [f64]| {
            for &(i, j, v) in &e2 {
                values[p2.find(i, j).unwrap()] = v;
            }
            Ok(())
        });
        (sys, p)
    }

    fn run(kind: JacobianKind) -> (Vec<f64>, PrjaOutput, usize, usize) {
        let (mut sys, p) = system();
        let n = 5;
        let y0: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
        let ewt: Vec<f64> = (0..n).map(|i| 1.0 / (1e-6 * y0[i].abs() + 1e-8)).collect();
        let mut savf = vec![0.0; n];
        sys.rhs(0.0, &y0, &mut savf).unwrap();
        let (h, el0) = (0.05, 0.8);
        let input = PrjaInput { tn: 0.0, h, el0, yh1: &y0, ewt: &ewt, savf: &savf };
        let mut lu = SparseLu::new(p);
        let (mut y, mut ftem, mut nfe, mut nje) = (y0.clone(), vec![0.0; n], 0, 0);
        let out = lu.prepare(&kind, &mut sys, &input, &mut y, &mut ftem, &mut nfe, &mut nje).unwrap();
        assert_eq!(y, y0);
        // Solve P x = b for a known x.
        let hl0 = h * el0;
        let x_true: Vec<f64> = (0..n).map(|i| 1.0 - 0.3 * i as f64).collect();
        let mut b: Vec<f64> = x_true.clone();
        for &(i, j, v) in &jentries() {
            b[i] -= hl0 * v * x_true[j];
        }
        lu.solve(h, el0, &mut b).unwrap();
        for i in 0..n {
            assert!((b[i] - x_true[i]).abs() <= 1e-6, "{kind:?}: {b:?} vs {x_true:?}");
        }
        (b, out, nfe, nje)
    }

    #[test]
    fn sparse_lu_user_and_fd_solve_p() {
        let (_, out_u, nfe_u, nje_u) = run(JacobianKind::SparseUser);
        let (_, out_f, nfe_f, nje_f) = run(JacobianKind::SparseFd);
        assert!(!out_u.singular && !out_f.singular);
        assert_eq!((nje_u, nje_f), (1, 1));
        assert_eq!(nfe_u, 0);
        let (_, p) = system();
        assert_eq!(nfe_f, p.coloring().ncolors());
        assert!(p.coloring().ncolors() < 5);
        // PDNORM = weighted norm of J (unit-free check: same for both).
        assert!((out_u.pdnorm - out_f.pdnorm).abs() <= 1e-6 * out_u.pdnorm);
    }

    #[test]
    fn sparse_lu_reports_singular_p() {
        // J with P = I - hl0*J having a zero column: J(0,0) = 1/hl0.
        let hl0 = 0.25;
        let p = SparsityPattern::from_entries(2, [(1, 0)]).unwrap();
        let p2 = p.clone();
        let mut sys = ClosureSystem::new(|_t, _y: &[f64], f: &mut [f64]| {
            f.fill(0.0);
            Ok(())
        })
        .with_sparse_jac(p.clone(), move |_t, _y: &[f64], v: &mut [f64]| {
            v[p2.find(0, 0).unwrap()] = 1.0 / hl0;
            Ok(())
        });
        let (y0, ewt, savf) = (vec![1.0, 1.0], vec![1.0, 1.0], vec![0.0, 0.0]);
        let input = PrjaInput { tn: 0.0, h: hl0, el0: 1.0, yh1: &y0, ewt: &ewt, savf: &savf };
        let mut lu = SparseLu::new(p);
        let (mut y, mut ftem, mut nfe, mut nje) = (y0.clone(), vec![0.0; 2], 0, 0);
        let out =
            lu.prepare(&JacobianKind::SparseUser, &mut sys, &input, &mut y, &mut ftem, &mut nfe, &mut nje).unwrap();
        assert!(out.singular);
    }
}
