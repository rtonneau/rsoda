//! Linear solve for the chord (corrector) iteration (`DSOLSY`).
//!
//! The iteration matrix `P = I - h*el0*J` is held by an [`IterationMatrix`]
//! (the Rust counterpart of the `WM`/`IWM` work arrays): the Jacobian
//! preparation fills and factors it, and [`IterationMatrix::solve`] applies
//! it to the corrector right-hand side.

use std::fmt;

use crate::linalg::{dgbsl, dgesl};
use crate::options::JacobianKind;

/// How the iteration matrix is stored (`MITER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixStorage {
    /// `MITER = 1, 2`: full `n x n` LU factors (leading dimension `n`).
    Dense,
    /// `MITER = 3`: diagonal approximation, stored as the inverse diagonal.
    /// Not used by LSODA itself; kept for completeness of the DSOLSY port.
    Diagonal,
    /// `MITER = 4, 5`: band LU factors with leading dimension
    /// `2*ml + mu + 1`.
    Banded { ml: usize, mu: usize },
}

impl MatrixStorage {
    /// Storage used for a Jacobian kind.
    pub fn for_jacobian(kind: &JacobianKind) -> Self {
        match *kind {
            JacobianKind::DenseUser | JacobianKind::DenseFd => Self::Dense,
            JacobianKind::BandedUser { ml, mu } | JacobianKind::BandedFd { ml, mu } => {
                Self::Banded { ml, mu }
            }
        }
    }
}

/// `IERSL = 1`: a zero diagonal element arose while updating the diagonal
/// iteration matrix (`MITER = 3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingularMatrix {
    /// 0-based index of the zero diagonal element.
    pub index: usize,
}

impl SingularMatrix {
    /// The Fortran `IERSL` value.
    pub fn iersl(&self) -> i32 {
        1
    }
}

impl fmt::Display for SingularMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "singular diagonal iteration matrix at component {}", self.index)
    }
}

impl std::error::Error for SingularMatrix {}

/// The factored iteration matrix and its pivots (`WM(3:)`, `IWM(21:)`), plus
/// `HL0` (`WM(2)`), the value of `h*el0` it was formed with.
#[derive(Debug, Clone, PartialEq)]
pub struct IterationMatrix {
    /// Storage scheme.
    pub storage: MatrixStorage,
    /// System size `n`.
    pub n: usize,
    /// Matrix entries, column-major with leading dimension [`Self::lda`]:
    /// LU factors (dense or band, as produced by `dgefa`/`dgbfa`) or the
    /// inverse diagonal.
    pub a: Vec<f64>,
    /// 0-based pivot rows from the LU factorization (unused for diagonal).
    pub ipvt: Vec<usize>,
    /// `h*el0` used when the matrix was last formed or updated (`WM(2)`).
    pub hl0: f64,
}

impl IterationMatrix {
    /// Allocates zeroed storage for an `n x n` iteration matrix.
    pub fn new(n: usize, storage: MatrixStorage) -> Self {
        let mut m = Self { storage, n, a: Vec::new(), ipvt: Vec::new(), hl0: 0.0 };
        m.a = vec![0.0; m.lda() * n];
        if storage != MatrixStorage::Diagonal {
            m.ipvt = vec![0; n];
        }
        m
    }

    /// Leading dimension of [`Self::a`]: `n` (dense), `1` (diagonal) or
    /// `MEBAND = 2*ml + mu + 1` (banded).
    pub fn lda(&self) -> usize {
        match self.storage {
            MatrixStorage::Dense => self.n,
            MatrixStorage::Diagonal => 1,
            MatrixStorage::Banded { ml, mu } => 2 * ml + mu + 1,
        }
    }

    /// Solves `P x = b` in place (`DSOLSY`): `x` holds `b` on entry and the
    /// solution on return.
    ///
    /// For dense and band storage this back-substitutes with the stored LU
    /// factors (`h` and `el0` are not used). For diagonal storage it first
    /// rescales the inverse diagonal from the stored `hl0` to `h*el0`, as
    /// DSOLSY does, and fails with [`SingularMatrix`] (`IERSL = 1`) if a
    /// diagonal element becomes zero; `x` is then left unchanged.
    pub fn solve(&mut self, h: f64, el0: f64, x: &mut [f64]) -> Result<(), SingularMatrix> {
        let n = self.n;
        assert!(x.len() >= n);
        match self.storage {
            MatrixStorage::Dense => {
                dgesl(&self.a, n, n, &self.ipvt, x, false);
            }
            MatrixStorage::Banded { ml, mu } => {
                let meband = 2 * ml + mu + 1;
                dgbsl(&self.a, meband, n, ml, mu, &self.ipvt, x, false);
            }
            MatrixStorage::Diagonal => {
                let phl0 = self.hl0;
                let hl0 = h * el0;
                self.hl0 = hl0;
                if hl0 != phl0 {
                    let r = hl0 / phl0;
                    for i in 0..n {
                        let di = 1.0 - r * (1.0 - 1.0 / self.a[i]);
                        if di.abs() == 0.0 {
                            return Err(SingularMatrix { index: i });
                        }
                        self.a[i] = 1.0 / di;
                    }
                }
                for (xi, ai) in x[..n].iter_mut().zip(&self.a) {
                    *xi *= ai;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{dgbfa, dgefa};

    fn matvec(n: usize, get: impl Fn(usize, usize) -> f64, x: &[f64]) -> Vec<f64> {
        (0..n).map(|i| (0..n).map(|j| get(i, j) * x[j]).sum()).collect()
    }

    #[test]
    fn solsy_dense_solves_factored_system() {
        let n = 3;
        // P = I - h*el0*J for a small J, column-major.
        let p = [4.0, 1.0, -2.0, 2.0, 5.0, 1.0, 0.0, -1.0, 3.0];
        let mut m = IterationMatrix::new(n, MatrixStorage::Dense);
        assert_eq!(m.lda(), 3);
        m.a.copy_from_slice(&p);
        assert_eq!(dgefa(&mut m.a, n, n, &mut m.ipvt), 0);
        let x_true = [1.0, -2.0, 0.5];
        let mut x = matvec(n, |i, j| p[i + j * n], &x_true);
        m.solve(0.1, 1.0, &mut x).unwrap();
        for i in 0..n {
            assert!((x[i] - x_true[i]).abs() < 1e-14, "{x:?}");
        }
    }

    #[test]
    fn solsy_banded_solves_factored_system() {
        let (n, ml, mu) = (5usize, 1usize, 2usize);
        let mut m = IterationMatrix::new(n, MatrixStorage::Banded { ml, mu });
        let lda = m.lda();
        assert_eq!(lda, 2 * ml + mu + 1);
        let elem = |i: usize, j: usize| -> f64 {
            // 0-based; diagonally dominant band matrix.
            if i == j {
                6.0 + i as f64
            } else if i + mu >= j && j + ml >= i {
                1.0 + 0.5 * (i + 2 * j) as f64 / 10.0
            } else {
                0.0
            }
        };
        let mrow = ml + mu; // 0-based row of the diagonal in band storage
        for j in 0..n {
            for i in 0..n {
                if i + mu >= j && j + ml >= i {
                    m.a[(i + mrow - j) + j * lda] = elem(i, j);
                }
            }
        }
        assert_eq!(dgbfa(&mut m.a, lda, n, ml, mu, &mut m.ipvt), 0);
        let x_true = [1.0, 2.0, -1.0, 0.25, 3.0];
        let mut x = matvec(n, elem, &x_true);
        m.solve(0.1, 1.0, &mut x).unwrap();
        for i in 0..n {
            assert!((x[i] - x_true[i]).abs() < 1e-13, "{x:?}");
        }
    }

    #[test]
    fn solsy_storage_from_jacobian_kind() {
        assert_eq!(MatrixStorage::for_jacobian(&JacobianKind::DenseFd), MatrixStorage::Dense);
        assert_eq!(MatrixStorage::for_jacobian(&JacobianKind::DenseUser), MatrixStorage::Dense);
        assert_eq!(
            MatrixStorage::for_jacobian(&JacobianKind::BandedFd { ml: 2, mu: 1 }),
            MatrixStorage::Banded { ml: 2, mu: 1 }
        );
        assert_eq!(
            MatrixStorage::for_jacobian(&JacobianKind::BandedUser { ml: 0, mu: 3 }),
            MatrixStorage::Banded { ml: 0, mu: 3 }
        );
    }

    #[test]
    fn solsy_diagonal_without_rescale() {
        let mut m = IterationMatrix::new(2, MatrixStorage::Diagonal);
        // Inverse diagonal of P = diag(1 - hl0*j) formed with hl0 = 0.5.
        m.hl0 = 0.5;
        m.a.copy_from_slice(&[0.5, 0.25]);
        let mut x = [2.0, 8.0];
        m.solve(0.25, 2.0, &mut x).unwrap();
        assert_eq!(x, [1.0, 2.0]);
        assert_eq!(m.hl0, 0.5);
    }

    #[test]
    fn solsy_diagonal_rescales_to_new_hl0() {
        // J = diag(-2, -6), formed with hl0 = 0.5: P = diag(2, 4).
        let mut m = IterationMatrix::new(2, MatrixStorage::Diagonal);
        m.hl0 = 0.5;
        m.a.copy_from_slice(&[1.0 / 2.0, 1.0 / 4.0]);
        // New hl0 = 0.25: P = diag(1.5, 2.5).
        let mut x = [3.0, 5.0];
        m.solve(0.125, 2.0, &mut x).unwrap();
        assert_eq!(m.hl0, 0.25);
        assert!((x[0] - 2.0).abs() < 1e-15 && (x[1] - 2.0).abs() < 1e-15, "{x:?}");
    }

    #[test]
    fn solsy_diagonal_singular_reports_iersl() {
        // J = diag(4), formed with hl0 = 0.125: P = 0.5. With hl0 = 0.25,
        // P = 1 - 0.25*4 = 0 -> IERSL = 1.
        let mut m = IterationMatrix::new(1, MatrixStorage::Diagonal);
        m.hl0 = 0.125;
        m.a[0] = 2.0;
        let mut x = [7.0];
        let err = m.solve(0.25, 1.0, &mut x).unwrap_err();
        assert_eq!(err, SingularMatrix { index: 0 });
        assert_eq!(err.iersl(), 1);
        assert_eq!(x, [7.0]);
    }
}
