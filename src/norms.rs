//! Weighted norms and the error-weight vector (`DMNORM`, `DFNORM`, `DBNORM`,
//! `DEWSET`).
//!
//! LSODA measures every error and correction in the weighted max-norm
//! `max_i |v(i)| * w(i)`, where `w(i) = 1 / EWT(i)` is the **inverted** error
//! weight. [`ewset`] loads `EWT` and [`invert_weights`] inverts it in place,
//! exactly as DLSODA does ("Load and invert the EWT array").
//!
//! Matrices follow the column-major Fortran layout of [`crate::linalg`].

use crate::options::Tolerance;

/// Weighted max-norm `max_i |v(i)| * w(i)` (`DMNORM`).
///
/// `w` holds the inverted error weights. Only the first `v.len()` entries of
/// `w` are used. Returns `0.0` for an empty vector.
pub fn vmnorm(v: &[f64], w: &[f64]) -> f64 {
    assert!(w.len() >= v.len());
    let mut vm = 0.0_f64;
    for (vi, wi) in v.iter().zip(w) {
        vm = fortran_max(vm, vi.abs() * wi);
    }
    vm
}

/// Norm of a full `n x n` matrix consistent with [`vmnorm`] (`DFNORM`):
/// `max_i ( w(i) * sum_j |a(i,j)| / w(j) )`.
///
/// `a` is column-major with leading dimension `n` (as `WM(3)` in DPRJA).
pub fn fnorm(n: usize, a: &[f64], w: &[f64]) -> f64 {
    assert!(a.len() >= n * n);
    assert!(w.len() >= n);
    let mut an = 0.0_f64;
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i + j * n].abs() / w[j];
        }
        an = fortran_max(an, sum * w[i]);
    }
    an
}

/// Norm of an `n x n` band matrix consistent with [`vmnorm`] (`DBNORM`):
/// `max_i ( w(i) * sum_j |a(i,j)| / w(j) )`.
///
/// `a` is column-major with leading dimension `nra >= ml + mu + 1`, and the
/// matrix element `a(i, j)` (1-based) is stored at the 1-based position
/// `(i - j + mu + 1, j)`. DPRJA passes `WM(ML+3)`, i.e. the factor storage
/// offset by `ml` rows, so the caller slices accordingly.
pub fn bnorm(n: usize, a: &[f64], nra: usize, ml: usize, mu: usize, w: &[f64]) -> f64 {
    assert!(nra >= ml + mu + 1);
    assert!(w.len() >= n);
    if n == 0 {
        return 0.0;
    }
    assert!(a.len() >= nra * (n - 1) + ml + mu + 1);
    let mut an = 0.0_f64;
    // 1-based indices, as in the Fortran.
    for i in 1..=n {
        let mut sum = 0.0;
        let i1 = i + mu + 1;
        let jlo = if i > ml { i - ml } else { 1 };
        let jhi = (i + mu).min(n);
        for j in jlo..=jhi {
            sum += a[(i1 - j - 1) + (j - 1) * nra].abs() / w[j - 1];
        }
        an = fortran_max(an, sum * w[i - 1]);
    }
    an
}

/// Sets the error weights `ewt(i) = rtol(i) * |ycur(i)| + atol(i)`
/// (`DEWSET`), with scalar tolerances broadcast as for `ITOL = 1..4`.
///
/// Only the first `ycur.len()` entries of `ewt` are written.
pub fn ewset(tol: &Tolerance, ycur: &[f64], ewt: &mut [f64]) {
    assert!(ewt.len() >= ycur.len());
    for (i, (e, y)) in ewt.iter_mut().zip(ycur).enumerate() {
        *e = tol.rtol_at(i) * y.abs() + tol.atol_at(i);
    }
}

/// Inverts the error weights in place (`EWT(i) = 1 / EWT(i)`), as DLSODA
/// does right after each [`ewset`] call.
///
/// Stops at the first weight `<= 0.0` and returns its 0-based index as
/// `Err` (DLSODA reports `ISTATE = -3` at the start of the problem and
/// `ISTATE = -6` afterwards); the weights before it have already been
/// inverted, as in the Fortran loop. A NaN weight is not caught, matching the
/// Fortran `.LE.` test.
pub fn invert_weights(ewt: &mut [f64]) -> Result<(), usize> {
    for (i, e) in ewt.iter_mut().enumerate() {
        if *e <= 0.0 {
            return Err(i);
        }
        *e = 1.0 / *e;
    }
    Ok(())
}

/// Running maximum for the norm loops (`AN = MAX(AN, x)`): `b` replaces `a`
/// only when `b > a`.
#[inline(always)]
fn fortran_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vmnorm_is_weighted_max() {
        let v = [1.0, -4.0, 2.0];
        let w = [1.0, 0.5, 3.0];
        assert_eq!(vmnorm(&v, &w), 6.0);
        assert_eq!(vmnorm(&[], &[]), 0.0);
        assert_eq!(vmnorm(&[-0.0], &[1.0]), 0.0);
    }

    /// Reference: `max_i w(i) * sum_j |a(i,j)| / w(j)` on a dense `n x n`.
    fn dense_reference(n: usize, get: impl Fn(usize, usize) -> f64, w: &[f64]) -> f64 {
        (0..n)
            .map(|i| w[i] * (0..n).map(|j| get(i, j).abs() / w[j]).sum::<f64>())
            .fold(0.0, f64::max)
    }

    #[test]
    fn fnorm_matches_definition() {
        let n = 3;
        // Column-major: a(i,j) = a[i + j*n].
        let a = [1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 0.5, 0.0, 8.0];
        let w = [1.0, 2.0, 0.25];
        let expected = dense_reference(n, |i, j| a[i + j * n], &w);
        assert_eq!(fnorm(n, &a, &w), expected);
        // Unit weights reduce to the infinity norm (max row sum: 3 + 6 + 8).
        let ones = [1.0; 3];
        assert_eq!(fnorm(n, &a, &ones), 17.0);
    }

    #[test]
    fn bnorm_matches_dense_norm() {
        // 5x5 band matrix with ml = 1, mu = 2, nra = ml + mu + 1 = 4.
        let (n, ml, mu) = (5usize, 1usize, 2usize);
        let nra = ml + mu + 1;
        let elem = |i: usize, j: usize| -> f64 {
            // 1-based; zero outside the band.
            if i + mu >= j && j + ml >= i {
                (i * 10 + j) as f64 * if (i + j).is_multiple_of(2) { 1.0 } else { -1.0 }
            } else {
                0.0
            }
        };
        let mut band = vec![0.0; nra * n];
        for j in 1..=n {
            for i in 1..=n {
                if i + mu >= j && j + ml >= i {
                    band[(i + mu - j) + (j - 1) * nra] = elem(i, j);
                }
            }
        }
        let w = [1.0, 0.5, 2.0, 4.0, 0.125];
        let expected = dense_reference(n, |i, j| elem(i + 1, j + 1), &w);
        let got = bnorm(n, &band, nra, ml, mu, &w);
        assert!((got - expected).abs() <= 1e-14 * expected, "{got} vs {expected}");
    }

    #[test]
    fn bnorm_accepts_dprja_offset_storage() {
        // DPRJA stores the band in rows ml+1.. of a 2*ml+mu+1 leading
        // dimension array and passes WM(ML+3), i.e. the slice from row ml.
        let (n, ml, mu) = (4usize, 1usize, 1usize);
        let meband = 2 * ml + mu + 1;
        let mut wm = vec![0.0; meband * n];
        // Tridiagonal: diag 2, off-diagonals -1.
        for j in 1..=n {
            for i in j.saturating_sub(mu).max(1)..=(j + ml).min(n) {
                let v = if i == j { 2.0 } else { -1.0 };
                wm[(i + ml + mu - j) + (j - 1) * meband] = v;
            }
        }
        let w = [1.0; 4];
        assert_eq!(bnorm(n, &wm[ml..], meband, ml, mu, &w), 4.0);
    }

    #[test]
    fn ewset_all_itol_forms() {
        let y = [1.0, -2.0, 0.0];
        let mut ewt = [0.0; 3];

        ewset(&Tolerance::scalar(0.1, 1e-3), &y, &mut ewt); // ITOL = 1
        assert_eq!(ewt, [0.1 * 1.0 + 1e-3, 0.1 * 2.0 + 1e-3, 1e-3]);

        let tol = Tolerance { rtol: crate::Tol::Scalar(0.1), atol: crate::Tol::PerComponent(vec![1.0, 2.0, 3.0]) };
        ewset(&tol, &y, &mut ewt); // ITOL = 2
        assert_eq!(ewt, [0.1 + 1.0, 0.2 + 2.0, 3.0]);

        let tol = Tolerance { rtol: crate::Tol::PerComponent(vec![1.0, 2.0, 3.0]), atol: crate::Tol::Scalar(0.5) };
        ewset(&tol, &y, &mut ewt); // ITOL = 3
        assert_eq!(ewt, [1.5, 4.5, 0.5]);

        ewset(&Tolerance::per_component(vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0]), &y, &mut ewt); // ITOL = 4
        assert_eq!(ewt, [1.0, 5.0, 2.0]);
    }

    #[test]
    fn invert_weights_reports_first_nonpositive() {
        let mut ewt = [2.0, 4.0];
        assert_eq!(invert_weights(&mut ewt), Ok(()));
        assert_eq!(ewt, [0.5, 0.25]);

        let mut ewt = [2.0, 0.0, -1.0];
        assert_eq!(invert_weights(&mut ewt), Err(1));
        assert_eq!(ewt[0], 0.5);
    }
}
