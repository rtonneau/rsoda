//! LINPACK dense and banded LU factorization and solve (`DGEFA`/`DGESL`,
//! `DGBFA`/`DGBSL`).
//!
//! These are direct ports of the LINPACK routines (not LAPACK), keeping the
//! original operation order so results match the Fortran code.
//!
//! Conventions:
//! * Matrices are stored **column-major** with a leading dimension `lda`, as in
//!   Fortran: element `(i, j)` (1-based) is at `a[(i - 1) + (j - 1) * lda]`.
//! * `ipvt` holds **0-based** pivot row indices (`ipvt[k]` is the row that was
//!   interchanged with row `k` at step `k`).
//! * The factorization routines return the LINPACK `info` indicator unchanged:
//!   `0` for success, otherwise the **1-based** index `k` of the (last) zero
//!   pivot `u(k, k)`. A non-zero `info` means `U` is singular; the factors must
//!   not be used with the solve routines.
//!
//! Internally the code uses 1-based loop indices to stay close to the Fortran
//! source; `ix` converts to a flat offset.

/// Flat offset of 1-based element `(i, j)` with leading dimension `lda`.
#[inline(always)]
fn ix(i: usize, j: usize, lda: usize) -> usize {
    (i - 1) + (j - 1) * lda
}

/// `IDAMAX` on `a[start .. start + len]` (contiguous). Returns the 0-based
/// offset of the first element of maximum absolute value.
fn idamax(a: &[f64], start: usize, len: usize) -> usize {
    let mut imax = 0;
    let mut dmax = a[start].abs();
    for i in 1..len {
        let v = a[start + i].abs();
        if v > dmax {
            imax = i;
            dmax = v;
        }
    }
    imax
}

/// LU factorization with partial pivoting of the `n x n` matrix `a`
/// (`DGEFA`).
///
/// `a` is column-major with leading dimension `lda >= n`; on return it holds
/// `U` in the upper triangle and the negated multipliers of `L` below the
/// diagonal. `ipvt` (length >= `n`) receives the 0-based pivot rows.
///
/// Returns `0` on success, or the 1-based index `k` of the last zero pivot
/// `u(k, k)` (the factorization is still completed, but cannot be used to
/// solve).
pub fn dgefa(a: &mut [f64], lda: usize, n: usize, ipvt: &mut [usize]) -> usize {
    assert!(lda >= n.max(1));
    assert!(ipvt.len() >= n);
    if n == 0 {
        return 0;
    }
    assert!(a.len() >= lda * (n - 1) + n);

    let mut info = 0;
    for k in 1..n {
        // Find the pivot row l.
        let l = idamax(a, ix(k, k, lda), n - k + 1) + k;
        ipvt[k - 1] = l - 1;

        if a[ix(l, k, lda)] == 0.0 {
            info = k;
            continue;
        }

        // Interchange if necessary.
        if l != k {
            a.swap(ix(l, k, lda), ix(k, k, lda));
        }

        // Compute multipliers.
        let t = -1.0 / a[ix(k, k, lda)];
        for i in (k + 1)..=n {
            a[ix(i, k, lda)] *= t;
        }

        // Row elimination with column indexing.
        for j in (k + 1)..=n {
            let t = a[ix(l, j, lda)];
            if l != k {
                a[ix(l, j, lda)] = a[ix(k, j, lda)];
                a[ix(k, j, lda)] = t;
            }
            for i in (k + 1)..=n {
                a[ix(i, j, lda)] += t * a[ix(i, k, lda)];
            }
        }
    }
    ipvt[n - 1] = n - 1;
    if a[ix(n, n, lda)] == 0.0 {
        info = n;
    }
    info
}

/// Solves `A x = b` (`transpose == false`) or `A^T x = b`
/// (`transpose == true`) using the factors from [`dgefa`] (`DGESL`).
///
/// `b` (length >= `n`) holds the right-hand side on entry and the solution on
/// return. Only call this when [`dgefa`] returned `0`; a zero pivot gives
/// division by zero.
pub fn dgesl(a: &[f64], lda: usize, n: usize, ipvt: &[usize], b: &mut [f64], transpose: bool) {
    assert!(lda >= n.max(1));
    assert!(ipvt.len() >= n);
    assert!(b.len() >= n);
    if n == 0 {
        return;
    }
    assert!(a.len() >= lda * (n - 1) + n);

    if !transpose {
        // Solve L * y = b.
        for k in 1..n {
            let l = ipvt[k - 1] + 1;
            let t = b[l - 1];
            if l != k {
                b[l - 1] = b[k - 1];
                b[k - 1] = t;
            }
            for i in (k + 1)..=n {
                b[i - 1] += t * a[ix(i, k, lda)];
            }
        }
        // Solve U * x = y.
        for k in (1..=n).rev() {
            b[k - 1] /= a[ix(k, k, lda)];
            let t = -b[k - 1];
            for i in 1..k {
                b[i - 1] += t * a[ix(i, k, lda)];
            }
        }
    } else {
        // Solve U^T * y = b.
        for k in 1..=n {
            let mut t = 0.0;
            for i in 1..k {
                t += a[ix(i, k, lda)] * b[i - 1];
            }
            b[k - 1] = (b[k - 1] - t) / a[ix(k, k, lda)];
        }
        // Solve L^T * x = y.
        for k in (1..n).rev() {
            let mut t = 0.0;
            for i in (k + 1)..=n {
                t += a[ix(i, k, lda)] * b[i - 1];
            }
            b[k - 1] += t;
            let l = ipvt[k - 1] + 1;
            if l != k {
                b.swap(l - 1, k - 1);
            }
        }
    }
}

/// LU factorization with partial pivoting of an `n x n` band matrix with `ml`
/// sub-diagonals and `mu` super-diagonals (`DGBFA`).
///
/// `abd` is column-major with leading dimension `lda >= 2*ml + mu + 1`. On
/// entry, with `m = ml + mu + 1`, element `A(i, j)` must be stored at the
/// 1-based position `abd(i - j + m, j)` (rows `1..=ml` are left free for
/// fill-in; they are zeroed here as needed). On return `abd` holds the factors
/// as in LINPACK. `ipvt` (length >= `n`) receives the 0-based pivot rows.
///
/// Returns `0` on success, or the 1-based index `k` of the last zero pivot
/// (the factorization is still completed, but cannot be used to solve).
pub fn dgbfa(
    abd: &mut [f64],
    lda: usize,
    n: usize,
    ml: usize,
    mu: usize,
    ipvt: &mut [usize],
) -> usize {
    let m = ml + mu + 1;
    assert!(lda >= 2 * ml + mu + 1);
    assert!(ipvt.len() >= n);
    if n == 0 {
        return 0;
    }
    assert!(abd.len() >= lda * (n - 1) + 2 * ml + mu + 1);

    let mut info = 0;

    // Zero initial fill-in columns.
    let j0 = mu + 2;
    let j1 = n.min(m) - 1;
    for jz in j0..=j1 {
        let i0 = m + 1 - jz;
        for i in i0..=ml {
            abd[ix(i, jz, lda)] = 0.0;
        }
    }
    let mut jz = j1;
    let mut ju = 0usize;

    // Gaussian elimination with partial pivoting.
    for k in 1..n {
        let kp1 = k + 1;

        // Zero next fill-in column.
        jz += 1;
        if jz <= n {
            for i in 1..=ml {
                abd[ix(i, jz, lda)] = 0.0;
            }
        }

        // Find l = pivot index.
        let lm = ml.min(n - k);
        let mut l = idamax(abd, ix(m, k, lda), lm + 1) + m;
        let pivot_row = l + k - m; // 1-based row of A
        ipvt[k - 1] = pivot_row - 1;

        // Zero pivot implies this column already triangularized.
        if abd[ix(l, k, lda)] == 0.0 {
            info = k;
            continue;
        }

        // Interchange if necessary.
        if l != m {
            abd.swap(ix(l, k, lda), ix(m, k, lda));
        }

        // Compute multipliers.
        let t = -1.0 / abd[ix(m, k, lda)];
        for i in 1..=lm {
            abd[ix(m + i, k, lda)] *= t;
        }

        // Row elimination with column indexing.
        ju = ju.max(mu + pivot_row).min(n);
        let mut mm = m;
        for j in kp1..=ju {
            l -= 1;
            mm -= 1;
            let t = abd[ix(l, j, lda)];
            if l != mm {
                abd[ix(l, j, lda)] = abd[ix(mm, j, lda)];
                abd[ix(mm, j, lda)] = t;
            }
            for i in 1..=lm {
                abd[ix(mm + i, j, lda)] += t * abd[ix(m + i, k, lda)];
            }
        }
    }
    ipvt[n - 1] = n - 1;
    if abd[ix(m, n, lda)] == 0.0 {
        info = n;
    }
    info
}

/// Solves `A x = b` (`transpose == false`) or `A^T x = b`
/// (`transpose == true`) using the band factors from [`dgbfa`] (`DGBSL`).
///
/// `b` (length >= `n`) holds the right-hand side on entry and the solution on
/// return. Only call this when [`dgbfa`] returned `0`.
pub fn dgbsl(
    abd: &[f64],
    lda: usize,
    n: usize,
    ml: usize,
    mu: usize,
    ipvt: &[usize],
    b: &mut [f64],
    transpose: bool,
) {
    let m = mu + ml + 1;
    assert!(lda >= 2 * ml + mu + 1);
    assert!(ipvt.len() >= n);
    assert!(b.len() >= n);
    if n == 0 {
        return;
    }
    assert!(abd.len() >= lda * (n - 1) + 2 * ml + mu + 1);

    if !transpose {
        // Solve L * y = b.
        if ml != 0 {
            for k in 1..n {
                let lm = ml.min(n - k);
                let l = ipvt[k - 1] + 1;
                let t = b[l - 1];
                if l != k {
                    b[l - 1] = b[k - 1];
                    b[k - 1] = t;
                }
                for i in 1..=lm {
                    b[k + i - 1] += t * abd[ix(m + i, k, lda)];
                }
            }
        }
        // Solve U * x = y.
        for k in (1..=n).rev() {
            b[k - 1] /= abd[ix(m, k, lda)];
            let lm = k.min(m) - 1;
            let la = m - lm;
            let lb = k - lm;
            let t = -b[k - 1];
            for i in 0..lm {
                b[lb + i - 1] += t * abd[ix(la + i, k, lda)];
            }
        }
    } else {
        // Solve U^T * y = b.
        for k in 1..=n {
            let lm = k.min(m) - 1;
            let la = m - lm;
            let lb = k - lm;
            let mut t = 0.0;
            for i in 0..lm {
                t += abd[ix(la + i, k, lda)] * b[lb + i - 1];
            }
            b[k - 1] = (b[k - 1] - t) / abd[ix(m, k, lda)];
        }
        // Solve L^T * x = y.
        if ml != 0 {
            for k in (1..n).rev() {
                let lm = ml.min(n - k);
                let mut t = 0.0;
                for i in 1..=lm {
                    t += abd[ix(m + i, k, lda)] * b[k + i - 1];
                }
                b[k - 1] += t;
                let l = ipvt[k - 1] + 1;
                if l != k {
                    b.swap(l - 1, k - 1);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row-major nested rows -> column-major dense matrix with leading dim `lda`.
    fn dense_from_rows(rows: &[&[f64]], lda: usize) -> Vec<f64> {
        let n = rows.len();
        let mut a = vec![0.0; lda * n];
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                a[i + j * lda] = v;
            }
        }
        a
    }

    /// `y = A x` for a dense row-major reference matrix.
    fn matvec(rows: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        rows.iter()
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    fn matvec_t(rows: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        let n = rows.len();
        (0..n)
            .map(|j| (0..n).map(|i| rows[i][j] * x[i]).sum())
            .collect()
    }

    /// Pack a full matrix (only entries inside the band) into LINPACK band storage.
    fn pack_band(full: &[Vec<f64>], ml: usize, mu: usize, lda: usize) -> Vec<f64> {
        let n = full.len();
        let m = ml + mu + 1;
        let mut abd = vec![0.0; lda * n];
        for j in 0..n {
            for i in 0..n {
                if i + mu >= j && i <= j + ml {
                    // 1-based abd(i - j + m, j)
                    let row = (i + 1) + m - (j + 1);
                    abd[(row - 1) + j * lda] = full[i][j];
                }
            }
        }
        abd
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{x} vs {y}");
        }
    }

    #[test]
    fn linalg_dense_known_system() {
        // 2x + y - z = 8; -3x - y + 2z = -11; -2x + y + 2z = -3 -> (2, 3, -1)
        let lda = 4; // lda > n exercises the leading dimension
        let mut a = dense_from_rows(&[&[2.0, 1.0, -1.0], &[-3.0, -1.0, 2.0], &[-2.0, 1.0, 2.0]], lda);
        let mut ipvt = [0usize; 3];
        assert_eq!(dgefa(&mut a, lda, 3, &mut ipvt), 0);
        let mut b = [8.0, -11.0, -3.0];
        dgesl(&a, lda, 3, &ipvt, &mut b, false);
        assert_close(&b, &[2.0, 3.0, -1.0], 1e-13);
    }

    #[test]
    fn linalg_dense_requires_pivoting() {
        let mut a = dense_from_rows(&[&[0.0, 1.0], &[1.0, 0.0]], 2);
        let mut ipvt = [0usize; 2];
        assert_eq!(dgefa(&mut a, 2, 2, &mut ipvt), 0);
        assert_eq!(ipvt[0], 1);
        let mut b = [3.0, 5.0];
        dgesl(&a, 2, 2, &ipvt, &mut b, false);
        assert_close(&b, &[5.0, 3.0], 1e-15);
    }

    #[test]
    fn linalg_dense_transpose_solve() {
        let rows = vec![
            vec![4.0, -2.0, 1.0, 0.5],
            vec![3.0, 6.0, -1.0, 2.0],
            vec![2.0, 1.0, 8.0, -3.0],
            vec![-1.0, 0.0, 2.0, 5.0],
        ];
        let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let mut a = dense_from_rows(&refs, 4);
        let mut ipvt = [0usize; 4];
        assert_eq!(dgefa(&mut a, 4, 4, &mut ipvt), 0);
        let x = [1.0, -2.0, 3.0, 0.25];

        let mut b = matvec(&rows, &x);
        dgesl(&a, 4, 4, &ipvt, &mut b, false);
        assert_close(&b, &x, 1e-13);

        let mut bt = matvec_t(&rows, &x);
        dgesl(&a, 4, 4, &ipvt, &mut bt, true);
        assert_close(&bt, &x, 1e-13);
    }

    #[test]
    fn linalg_dense_singular_detected() {
        // Rows 1 and 2 proportional: the zero pivot appears at the last step.
        let mut a = dense_from_rows(&[&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], &[1.0, 0.0, 1.0]], 3);
        let mut ipvt = [0usize; 3];
        assert_eq!(dgefa(&mut a, 3, 3, &mut ipvt), 3);

        // Zero first column: info = 1 (zero pivot found at k = 1).
        let mut a = dense_from_rows(&[&[0.0, 1.0], &[0.0, 2.0]], 2);
        let mut ipvt = [0usize; 2];
        assert_eq!(dgefa(&mut a, 2, 2, &mut ipvt), 1);

        // 1x1 zero matrix.
        let mut a = [0.0];
        let mut ipvt = [0usize; 1];
        assert_eq!(dgefa(&mut a, 1, 1, &mut ipvt), 1);
    }

    #[test]
    fn linalg_dense_one_by_one() {
        let mut a = [4.0];
        let mut ipvt = [0usize; 1];
        assert_eq!(dgefa(&mut a, 1, 1, &mut ipvt), 0);
        let mut b = [2.0];
        dgesl(&a, 1, 1, &ipvt, &mut b, false);
        assert_close(&b, &[0.5], 1e-15);
    }

    /// Deterministic pseudo-random band matrix, diagonally weak enough to force pivoting.
    fn band_matrix(n: usize, ml: usize, mu: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / ((1u64 << 31) as f64) * 2.0 - 1.0
        };
        let mut full = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i + mu >= j && i <= j + ml {
                    full[i][j] = next();
                }
            }
        }
        full
    }

    #[test]
    fn linalg_banded_matches_dense() {
        for &(n, ml, mu) in &[
            (1, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (5, 1, 1),
            (6, 2, 1),
            (7, 1, 3),
            (8, 3, 3),
            (10, 2, 2),
            (4, 3, 3),
        ] {
            let full = band_matrix(n, ml, mu, (n * 100 + ml * 10 + mu) as u64);
            let lda = 2 * ml + mu + 1;
            let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64 - (i % 3) as f64).collect();

            let mut abd = pack_band(&full, ml, mu, lda);
            let mut ipvt = vec![0usize; n];
            assert_eq!(dgbfa(&mut abd, lda, n, ml, mu, &mut ipvt), 0, "n={n} ml={ml} mu={mu}");

            let mut b = matvec(&full, &x);
            dgbsl(&abd, lda, n, ml, mu, &ipvt, &mut b, false);
            assert_close(&b, &x, 1e-10);

            let mut bt = matvec_t(&full, &x);
            dgbsl(&abd, lda, n, ml, mu, &ipvt, &mut bt, true);
            assert_close(&bt, &x, 1e-10);

            // Dense LINPACK on the same matrix must agree.
            let refs: Vec<&[f64]> = full.iter().map(|r| r.as_slice()).collect();
            let mut a = dense_from_rows(&refs, n);
            let mut dip = vec![0usize; n];
            assert_eq!(dgefa(&mut a, n, n, &mut dip), 0);
            let mut bd = matvec(&full, &x);
            dgesl(&a, n, n, &dip, &mut bd, false);
            assert_close(&bd, &x, 1e-10);
        }
    }

    #[test]
    fn linalg_banded_tridiagonal_known() {
        // Classic 1D Laplacian [-1 2 -1], n = 4, rhs = [1,0,0,1] -> x = [1,1,1,1].
        let n = 4;
        let (ml, mu) = (1, 1);
        let lda = 2 * ml + mu + 1;
        let mut full = vec![vec![0.0; n]; n];
        for i in 0..n {
            full[i][i] = 2.0;
            if i + 1 < n {
                full[i][i + 1] = -1.0;
                full[i + 1][i] = -1.0;
            }
        }
        let mut abd = pack_band(&full, ml, mu, lda);
        let mut ipvt = vec![0usize; n];
        assert_eq!(dgbfa(&mut abd, lda, n, ml, mu, &mut ipvt), 0);
        let mut b = [1.0, 0.0, 0.0, 1.0];
        dgbsl(&abd, lda, n, ml, mu, &ipvt, &mut b, false);
        assert_close(&b, &[1.0, 1.0, 1.0, 1.0], 1e-14);
    }

    #[test]
    fn linalg_banded_singular_detected() {
        let (ml, mu) = (1, 1);
        let lda = 2 * ml + mu + 1;

        // Zero last row/column: singular at k = n.
        let n = 3;
        let mut full = vec![vec![0.0; n]; n];
        full[0][0] = 2.0;
        full[0][1] = 1.0;
        full[1][0] = 1.0;
        full[1][1] = 2.0;
        let mut abd = pack_band(&full, ml, mu, lda);
        let mut ipvt = vec![0usize; n];
        assert_eq!(dgbfa(&mut abd, lda, n, ml, mu, &mut ipvt), 3);

        // Zero first column: info = 1.
        let n = 3;
        let mut full = vec![vec![0.0; n]; n];
        full[0][1] = 1.0;
        full[1][1] = 1.0;
        full[1][2] = 1.0;
        full[2][1] = 1.0;
        full[2][2] = 3.0;
        let mut abd = pack_band(&full, ml, mu, lda);
        let mut ipvt = vec![0usize; n];
        assert_eq!(dgbfa(&mut abd, lda, n, ml, mu, &mut ipvt), 1);

        // 1x1 zero band matrix.
        let mut abd = [0.0];
        let mut ipvt = [0usize; 1];
        assert_eq!(dgbfa(&mut abd, 1, 1, 0, 0, &mut ipvt), 1);
    }
}
