//! Sparse Jacobians: the sparsity pattern, its column coloring, and the
//! assembly of the iteration matrix `P = I - h*el0*J` in compressed sparse
//! column (CSC) form.
//!
//! These pieces are pure and free of dependencies. The sparse LU solver
//! behind the `sparse` feature builds on them, and they are public so that
//! a user [`crate::LinearSolver`] can reuse them.
//!
//! * [`SparsityPattern`]: the fixed positions where `J` may be nonzero,
//!   always including the diagonal (needed by `P`).
//! * [`ColumnColoring`]: a greedy Curtis-Powell-Reid grouping of columns
//!   that share no row, so one evaluation of `f` per group estimates all of
//!   its columns by finite differences.
//! * [`fd_jacobian`] and [`scale_jacobian`] fill `-h*el0*J` in pattern
//!   order, and [`finish_iteration_matrix`] returns `PDNORM` and adds the
//!   identity, as DPRJA does for the dense and band kinds.

use crate::consts::UROUND;
use crate::error::{InputError, LsodaError};
use crate::norms::{csc_fnorm, vmnorm};
use crate::prja::PrjaInput;
use crate::system::OdeSystem;

/// The positions where an `n x n` Jacobian may be nonzero, in compressed
/// sparse column form: the rows of column `j` are
/// `row_idx[col_ptr[j]..col_ptr[j + 1]]`, strictly increasing.
///
/// The diagonal is always part of the pattern (it is added if missing), so
/// that `P = I - h*el0*J` fits the same pattern. Jacobian values are stored
/// in the order of `row_idx`; use [`SparsityPattern::find`] or the column
/// ranges to locate an entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparsityPattern {
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    /// Position of `(j, j)` in `row_idx`, for each column `j`.
    diag: Vec<usize>,
}

impl SparsityPattern {
    /// A pattern from CSC arrays, validated: `col_ptr` has `n + 1`
    /// nondecreasing entries from `0` to `row_idx.len()`, and the rows of each
    /// column are strictly increasing and below `n`. Missing diagonal entries
    /// are inserted.
    ///
    /// # Errors
    ///
    /// [`InputError::InvalidSparsityPattern`] if a check fails.
    pub fn new(n: usize, col_ptr: Vec<usize>, row_idx: Vec<usize>) -> Result<Self, InputError> {
        let bad = |reason: &'static str| InputError::InvalidSparsityPattern { reason };
        if n == 0 {
            return Err(bad("n must be positive"));
        }
        if col_ptr.len() != n + 1 {
            return Err(bad("col_ptr must have n + 1 entries"));
        }
        if col_ptr[0] != 0 || col_ptr[n] != row_idx.len() {
            return Err(bad("col_ptr must run from 0 to row_idx.len()"));
        }
        if col_ptr.windows(2).any(|w| w[0] > w[1]) {
            return Err(bad("col_ptr must be nondecreasing"));
        }
        for j in 0..n {
            let rows = &row_idx[col_ptr[j]..col_ptr[j + 1]];
            if rows.iter().any(|&i| i >= n) {
                return Err(bad("row index out of range"));
            }
            if rows.windows(2).any(|w| w[0] >= w[1]) {
                return Err(bad("rows of a column must be strictly increasing"));
            }
        }
        // Insert the missing diagonal entries.
        let mut cp = Vec::with_capacity(n + 1);
        let mut ri = Vec::with_capacity(row_idx.len() + n);
        let mut diag = Vec::with_capacity(n);
        cp.push(0);
        for j in 0..n {
            let rows = &row_idx[col_ptr[j]..col_ptr[j + 1]];
            let at = rows.partition_point(|&i| i < j);
            ri.extend_from_slice(&rows[..at]);
            diag.push(ri.len());
            ri.push(j);
            let rest = if rows.get(at) == Some(&j) { at + 1 } else { at };
            ri.extend_from_slice(&rows[rest..]);
            cp.push(ri.len());
        }
        Ok(Self { n, col_ptr: cp, row_idx: ri, diag })
    }

    /// A pattern from `(row, col)` entries in any order; duplicates are
    /// merged and the diagonal is added.
    ///
    /// # Errors
    ///
    /// [`InputError::InvalidSparsityPattern`] for `n = 0` or an index not
    /// below `n`.
    pub fn from_entries(n: usize, entries: impl IntoIterator<Item = (usize, usize)>) -> Result<Self, InputError> {
        if n == 0 {
            return Err(InputError::InvalidSparsityPattern { reason: "n must be positive" });
        }
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, j) in entries {
            if i >= n || j >= n {
                return Err(InputError::InvalidSparsityPattern { reason: "row or column index out of range" });
            }
            cols[j].push(i);
        }
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
        for mut c in cols {
            c.sort_unstable();
            c.dedup();
            row_idx.extend(c);
            col_ptr.push(row_idx.len());
        }
        Self::new(n, col_ptr, row_idx)
    }

    /// The dense `n x n` pattern.
    pub fn dense(n: usize) -> Result<Self, InputError> {
        Self::from_entries(n, (0..n).flat_map(|j| (0..n).map(move |i| (i, j))))
    }

    /// The band pattern with `ml` sub- and `mu` superdiagonals.
    pub fn banded(n: usize, ml: usize, mu: usize) -> Result<Self, InputError> {
        Self::from_entries(
            n,
            (0..n).flat_map(move |j| (j.saturating_sub(mu)..=(j + ml).min(n.max(1) - 1)).map(move |i| (i, j))),
        )
    }

    /// Matrix size `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }

    /// Column pointers (`n + 1` entries).
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    /// Row indices of the stored entries, column by column.
    pub fn row_idx(&self) -> &[usize] {
        &self.row_idx
    }

    /// Value positions of the diagonal entries, by column.
    pub fn diag(&self) -> &[usize] {
        &self.diag
    }

    /// Value position of entry `(row, col)`, if it is in the pattern.
    pub fn find(&self, row: usize, col: usize) -> Option<usize> {
        let start = self.col_ptr[col];
        self.row_idx[start..self.col_ptr[col + 1]].binary_search(&row).ok().map(|k| start + k)
    }

    /// The greedy Curtis-Powell-Reid coloring of the columns.
    pub fn coloring(&self) -> ColumnColoring {
        ColumnColoring::new(self)
    }
}

/// A partition of the columns into groups whose columns share no nonzero
/// row (Curtis, Powell and Reid). Perturbing all columns of a group at once
/// estimates each of them from one evaluation of `f`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnColoring {
    /// Columns of group `g`: `cols[group_ptr[g]..group_ptr[g + 1]]`, increasing.
    group_ptr: Vec<usize>,
    cols: Vec<usize>,
}

impl ColumnColoring {
    /// Greedy coloring in column order: each column takes the smallest color
    /// not used by an earlier column sharing a row with it.
    pub fn new(pattern: &SparsityPattern) -> Self {
        let n = pattern.n;
        let (cp, ri) = (&pattern.col_ptr, &pattern.row_idx);
        // Row-wise transpose: the columns of each row.
        let mut rp = vec![0usize; n + 1];
        for &i in ri {
            rp[i + 1] += 1;
        }
        for i in 0..n {
            rp[i + 1] += rp[i];
        }
        let mut next = rp.clone();
        let mut rc = vec![0usize; ri.len()];
        for j in 0..n {
            for &i in &ri[cp[j]..cp[j + 1]] {
                rc[next[i]] = j;
                next[i] += 1;
            }
        }
        let mut color = vec![usize::MAX; n];
        // forbidden[c] == j + 1 marks color c as taken for column j.
        let mut forbidden: Vec<usize> = Vec::new();
        let mut ncolors = 0;
        for j in 0..n {
            for &i in &ri[cp[j]..cp[j + 1]] {
                for &k in &rc[rp[i]..rp[i + 1]] {
                    if k < j {
                        forbidden[color[k]] = j + 1;
                    }
                }
            }
            let c = (0..ncolors).find(|&c| forbidden[c] != j + 1).unwrap_or(ncolors);
            if c == ncolors {
                ncolors += 1;
                forbidden.push(0);
            }
            color[j] = c;
        }
        let mut group_ptr = vec![0usize; ncolors + 1];
        for &c in &color {
            group_ptr[c + 1] += 1;
        }
        for c in 0..ncolors {
            group_ptr[c + 1] += group_ptr[c];
        }
        let mut fill = group_ptr.clone();
        let mut cols = vec![0usize; n];
        for (j, &c) in color.iter().enumerate() {
            cols[fill[c]] = j;
            fill[c] += 1;
        }
        Self { group_ptr, cols }
    }

    /// Number of groups (colors): the number of `f` evaluations per
    /// finite-difference Jacobian.
    pub fn ncolors(&self) -> usize {
        self.group_ptr.len() - 1
    }

    /// The columns of group `g`, in increasing order.
    pub fn group(&self, g: usize) -> &[usize] {
        &self.cols[self.group_ptr[g]..self.group_ptr[g + 1]]
    }
}

/// Fills `values` (pattern order) with `-h*el0*J`, `J` estimated by finite
/// differences of `f` over the groups of `coloring`, with DPRJA's increments
/// `r_j = max(sqrt(UROUND)*|y_j|, r0/ewt_j)` and
/// `r0 = 1000*|h|*UROUND*n*||savf||` (`r0 = 1` if that is zero).
///
/// `y` holds the predicted solution (equal to `input.yh1`); perturbed
/// components are restored from `input.yh1`, also when `f` fails. `ftem`
/// (length `>= n`) is scratch space. `nfe` is increased by the number of
/// groups. On a single-column-per-group coloring this is DPRJA's dense
/// finite-difference Jacobian, entry for entry.
///
/// # Errors
///
/// An `Err` of [`OdeSystem::rhs`], unchanged (`nfe` not increased).
pub fn fd_jacobian(
    system: &mut dyn OdeSystem,
    pattern: &SparsityPattern,
    coloring: &ColumnColoring,
    input: &PrjaInput<'_>,
    y: &mut [f64],
    ftem: &mut [f64],
    values: &mut [f64],
    nfe: &mut usize,
) -> Result<(), LsodaError> {
    let n = pattern.n;
    let PrjaInput { tn, h, el0, yh1, ewt, savf } = *input;
    assert!(y.len() >= n && ftem.len() >= n && values.len() >= pattern.nnz());
    let hl0 = h * el0;
    let srur = UROUND.sqrt();
    let fac = vmnorm(&savf[..n], ewt);
    let mut r0 = 1000.0 * h.abs() * UROUND * n as f64 * fac;
    if r0 == 0.0 {
        r0 = 1.0;
    }
    for g in 0..coloring.ncolors() {
        let group = coloring.group(g);
        for &j in group {
            let r = (srur * y[j].abs()).max(r0 / ewt[j]);
            y[j] += r;
        }
        let res = system.rhs(tn, &y[..n], &mut ftem[..n]);
        for &j in group {
            y[j] = yh1[j];
        }
        res?;
        for &j in group {
            let r = (srur * y[j].abs()).max(r0 / ewt[j]);
            let fac = -hl0 / r;
            for k in pattern.col_ptr[j]..pattern.col_ptr[j + 1] {
                let i = pattern.row_idx[k];
                values[k] = (ftem[i] - savf[i]) * fac;
            }
        }
    }
    *nfe += coloring.ncolors();
    Ok(())
}

/// Scales user Jacobian values (pattern order) by `-hl0`, as DPRJA does for
/// the user-supplied kinds.
pub fn scale_jacobian(values: &mut [f64], hl0: f64) {
    let con = -hl0;
    for v in values {
        *v *= con;
    }
}

/// Completes `P = I - h*el0*J` from `values = -hl0*J` (pattern order):
/// returns `PDNORM = ||J||` (the [`crate::norms::fnorm`] norm of `values`
/// divided by `|hl0|`), then adds the identity on the diagonal. `rowsum`
/// (length `>= n`) is scratch space.
pub fn finish_iteration_matrix(
    pattern: &SparsityPattern,
    values: &mut [f64],
    ewt: &[f64],
    hl0: f64,
    rowsum: &mut [f64],
) -> f64 {
    let n = pattern.n;
    let pdnorm = csc_fnorm(n, &pattern.col_ptr, &pattern.row_idx, values, ewt, rowsum) / hl0.abs();
    for &k in &pattern.diag {
        values[k] += 1.0;
    }
    pdnorm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::dgefa;
    use crate::options::JacobianKind;
    use crate::prja::prja;
    use crate::solsy::{IterationMatrix, MatrixStorage};
    use crate::system::ClosureSystem;

    fn invalid(r: Result<SparsityPattern, InputError>) -> &'static str {
        match r {
            Err(InputError::InvalidSparsityPattern { reason }) => reason,
            other => panic!("expected InvalidSparsityPattern, got {other:?}"),
        }
    }

    #[test]
    fn pattern_inserts_diagonal_and_validates() {
        // 3x3, column 0: rows {1}, column 1: rows {0, 1}, column 2: {}.
        let p = SparsityPattern::new(3, vec![0, 1, 3, 3], vec![1, 0, 1]).unwrap();
        assert_eq!(p.col_ptr(), &[0, 2, 4, 5]);
        assert_eq!(p.row_idx(), &[0, 1, 0, 1, 2]);
        assert_eq!(p.diag(), &[0, 3, 4]);
        assert_eq!((p.find(1, 0), p.find(2, 0), p.find(2, 2)), (Some(1), None, Some(4)));
        assert_eq!(p.nnz(), 5);

        assert_eq!(invalid(SparsityPattern::new(0, vec![0], vec![])), "n must be positive");
        assert_eq!(invalid(SparsityPattern::new(2, vec![0, 1], vec![0])), "col_ptr must have n + 1 entries");
        assert_eq!(
            invalid(SparsityPattern::new(2, vec![0, 1, 1], vec![0, 1])),
            "col_ptr must run from 0 to row_idx.len()"
        );
        assert_eq!(invalid(SparsityPattern::new(2, vec![0, 2, 1], vec![0])), "col_ptr must be nondecreasing");
        assert_eq!(invalid(SparsityPattern::new(2, vec![0, 1, 2], vec![0, 2])), "row index out of range");
        assert_eq!(
            invalid(SparsityPattern::new(2, vec![0, 2, 2], vec![1, 0])),
            "rows of a column must be strictly increasing"
        );
        assert_eq!(invalid(SparsityPattern::from_entries(2, [(0, 2)])), "row or column index out of range");
    }

    #[test]
    fn pattern_from_entries_merges_duplicates() {
        let p = SparsityPattern::from_entries(3, [(2, 0), (0, 0), (2, 0), (1, 2)]).unwrap();
        assert_eq!(p.col_ptr(), &[0, 2, 3, 5]);
        assert_eq!(p.row_idx(), &[0, 2, 1, 1, 2]);
        assert_eq!(SparsityPattern::dense(3).unwrap().nnz(), 9);
        let b = SparsityPattern::banded(5, 1, 2).unwrap();
        assert_eq!(b.nnz(), 5 + 4 + 4 + 3);
        assert!(b.find(3, 2).is_some() && b.find(0, 2).is_some() && b.find(4, 2).is_none());
    }

    /// Small deterministic pseudo-random generator (no dependencies).
    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state >> 33
    }

    fn assert_valid_coloring(p: &SparsityPattern, c: &ColumnColoring) {
        let n = p.n();
        let mut seen = vec![false; n];
        for g in 0..c.ncolors() {
            let mut rows_used = vec![false; n];
            for &j in c.group(g) {
                assert!(!seen[j], "column {j} in two groups");
                seen[j] = true;
                for k in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                    let i = p.row_idx()[k];
                    assert!(!rows_used[i], "group {g} has two columns in row {i}");
                    rows_used[i] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "every column is colored");
    }

    #[test]
    fn coloring_is_valid_on_random_patterns() {
        let mut s = 12345u64;
        for trial in 0..50 {
            let n = 1 + (lcg(&mut s) % 40) as usize;
            let m = (lcg(&mut s) % (3 * n as u64 + 1)) as usize;
            let entries: Vec<_> =
                (0..m).map(|_| ((lcg(&mut s) % n as u64) as usize, (lcg(&mut s) % n as u64) as usize)).collect();
            let p = SparsityPattern::from_entries(n, entries).unwrap();
            let c = p.coloring();
            assert_valid_coloring(&p, &c);
            assert!(c.ncolors() <= n, "trial {trial}");
        }
    }

    #[test]
    fn coloring_of_structured_patterns() {
        assert_eq!(SparsityPattern::banded(10, 1, 1).unwrap().coloring().ncolors(), 3);
        assert_eq!(SparsityPattern::banded(10, 2, 1).unwrap().coloring().ncolors(), 4);
        assert_eq!(SparsityPattern::from_entries(6, []).unwrap().coloring().ncolors(), 1);
        let dense = SparsityPattern::dense(5).unwrap().coloring();
        assert_eq!(dense.ncolors(), 5);
        assert_eq!((0..5).map(|g| dense.group(g).to_vec()).collect::<Vec<_>>(), [[0], [1], [2], [3], [4]]);
        let tri = SparsityPattern::banded(7, 1, 1).unwrap().coloring();
        assert_eq!(tri.group(0), &[0, 3, 6]);
    }

    /// A nonlinear test system with a tridiagonal Jacobian.
    fn tridiag_rhs(n: usize) -> impl FnMut(f64, &[f64], &mut [f64]) -> Result<(), LsodaError> {
        move |_t, y, f| {
            for i in 0..n {
                let left = if i > 0 { y[i - 1] } else { 0.0 };
                let right = if i + 1 < n { y[i + 1] } else { 0.0 };
                f[i] = -2.0 * y[i] * y[i] + left - 0.5 * right * y[i] + 0.1 * i as f64;
            }
            Ok(())
        }
    }

    fn setup(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let y: Vec<f64> = (0..n).map(|i| 0.5 + 0.3 * i as f64).collect();
        let ewt: Vec<f64> = (0..n).map(|i| 1.0 / (1e-6 * y[i].abs() + 1e-8)).collect();
        let mut savf = vec![0.0; n];
        tridiag_rhs(n)(0.0, &y, &mut savf).unwrap();
        (y, ewt, savf)
    }

    #[test]
    fn fd_on_dense_pattern_reproduces_dense_prja_bit_for_bit() {
        let n = 6;
        let (y0, ewt, savf) = setup(n);
        let (h, el0) = (0.05, 0.8);
        let input = PrjaInput { tn: 0.0, h, el0, yh1: &y0, ewt: &ewt, savf: &savf };

        // Dense DPRJA, MITER = 2, before factorization: reproduce by hand.
        let mut sys = ClosureSystem::new(tridiag_rhs(n));
        let mut wm = IterationMatrix::new(n, MatrixStorage::Dense);
        let (mut y, mut ftem) = (y0.clone(), vec![0.0; n]);
        let (mut nfe_d, mut nje) = (0, 0);
        let out =
            prja(&mut sys, JacobianKind::DenseFd, &input, &mut y, &mut ftem, &mut wm, &mut nfe_d, &mut nje).unwrap();

        let p = SparsityPattern::dense(n).unwrap();
        let c = p.coloring();
        let mut values = vec![0.0; p.nnz()];
        let mut nfe = 0;
        let mut y = y0.clone();
        fd_jacobian(&mut sys, &p, &c, &input, &mut y, &mut ftem, &mut values, &mut nfe).unwrap();
        assert_eq!(y, y0, "y restored");
        assert_eq!(nfe, n);
        let mut rowsum = vec![0.0; n];
        let pdnorm = finish_iteration_matrix(&p, &mut values, &ewt, h * el0, &mut rowsum);
        assert_eq!(pdnorm, out.pdnorm);
        // Dense P before factoring, then factor it the same way.
        let mut a = vec![0.0; n * n];
        for j in 0..n {
            for k in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                a[p.row_idx()[k] + j * n] = values[k];
            }
        }
        let mut ipvt = vec![0; n];
        dgefa(&mut a, n, n, &mut ipvt);
        assert_eq!(a, wm.a);
        assert_eq!(ipvt, wm.ipvt);
    }

    #[test]
    fn colored_fd_matches_dense_fd_on_tridiagonal_system() {
        let n = 9;
        let (y0, ewt, savf) = setup(n);
        let input = PrjaInput { tn: 0.0, h: 0.05, el0: 0.8, yh1: &y0, ewt: &ewt, savf: &savf };
        let mut sys = ClosureSystem::new(tridiag_rhs(n));
        let mut ftem = vec![0.0; n];

        let dense = SparsityPattern::dense(n).unwrap();
        let mut dv = vec![0.0; dense.nnz()];
        let (mut y, mut nfe) = (y0.clone(), 0);
        fd_jacobian(&mut sys, &dense, &dense.coloring(), &input, &mut y, &mut ftem, &mut dv, &mut nfe).unwrap();

        let tri = SparsityPattern::banded(n, 1, 1).unwrap();
        let c = tri.coloring();
        let mut tv = vec![0.0; tri.nnz()];
        let mut nfe = 0;
        fd_jacobian(&mut sys, &tri, &c, &input, &mut y, &mut ftem, &mut tv, &mut nfe).unwrap();
        assert_eq!(nfe, 3);
        assert_eq!(y, y0);
        for j in 0..n {
            for i in 0..n {
                let d = dv[dense.find(i, j).unwrap()];
                match tri.find(i, j) {
                    // Same increments: the colored estimate of an entry only
                    // differs from the dense one by roundoff of f.
                    Some(k) => assert!((tv[k] - d).abs() <= 1e-9 * d.abs().max(1.0), "({i},{j}): {} vs {d}", tv[k]),
                    None => assert_eq!(d, 0.0, "({i},{j}) outside the pattern"),
                }
            }
        }
    }

    #[test]
    fn fd_restores_y_when_f_fails() {
        let n = 4;
        let (y0, ewt, savf) = setup(n);
        let input = PrjaInput { tn: 0.0, h: 0.05, el0: 0.8, yh1: &y0, ewt: &ewt, savf: &savf };
        let mut calls = 0;
        let mut sys = ClosureSystem::new(move |_t, _y: &[f64], _f: &mut [f64]| {
            calls += 1;
            if calls == 2 { Err(LsodaError::Callback { ierr: -1 }) } else { Ok(()) }
        });
        let p = SparsityPattern::banded(n, 1, 1).unwrap();
        let mut values = vec![0.0; p.nnz()];
        let (mut y, mut ftem, mut nfe) = (y0.clone(), vec![0.0; n], 0);
        let r = fd_jacobian(&mut sys, &p, &p.coloring(), &input, &mut y, &mut ftem, &mut values, &mut nfe);
        assert!(r.is_err());
        assert_eq!((y, nfe), (y0, 0));
    }

    #[test]
    fn scale_and_finish_form_p() {
        let p = SparsityPattern::from_entries(2, [(1, 0), (0, 1)]).unwrap();
        // J = [[1, 2], [3, 4]] in pattern order (0,0), (1,0), (0,1), (1,1).
        let mut v = vec![1.0, 3.0, 2.0, 4.0];
        scale_jacobian(&mut v, 0.5);
        assert_eq!(v, [-0.5, -1.5, -1.0, -2.0]);
        let pdnorm = finish_iteration_matrix(&p, &mut v, &[1.0, 1.0], 0.5, &mut [0.0; 2]);
        assert_eq!(pdnorm, 7.0);
        assert_eq!(v, [0.5, -1.5, -1.0, -1.0]);
    }
}
