//! Interpolation of the solution and its derivatives from the Nordsieck
//! history array (`DINTDY`).
//!
//! The history array `YH(NYH, L)` is column-major: column `j` (1-based) holds
//! `h^(j-1) y^(j-1)(tn) / (j-1)!` and its element `(i, j)` is at
//! `yh[(i - 1) + (j - 1) * nyh]`.

use std::fmt;

use crate::consts::UROUND;

/// The step data DINTDY reads from the solver state (`COMMON /DLS001/`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepData {
    /// Current order `NQ` (`NQCUR`); the history has `L = nq + 1` columns.
    pub nq: usize,
    /// Current time `TN` (`TCUR`).
    pub tn: f64,
    /// Step size `H` to be attempted next (`HCUR`), the scale of `yh`.
    pub h: f64,
    /// Step size `HU` last used successfully.
    pub hu: f64,
}

/// Illegal arguments to [`intdy`] (`IFLAG = -1` or `-2`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntdyError {
    /// `IFLAG = -1`: the derivative order `k` exceeds the current order.
    IllegalK { k: usize, nq: usize },
    /// `IFLAG = -2`: `t` is outside `[tcur - hu, tcur]` (with roundoff
    /// allowance `tp` as the lower end).
    IllegalT { t: f64, tp: f64, tn: f64 },
}

impl IntdyError {
    /// The Fortran `IFLAG` value.
    pub fn iflag(&self) -> i32 {
        match self {
            Self::IllegalK { .. } => -1,
            Self::IllegalT { .. } => -2,
        }
    }
}

impl fmt::Display for IntdyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IllegalK { k, nq } => write!(f, "DINTDY- K (={k}) illegal (current order {nq})"),
            Self::IllegalT { t, tp, tn } => write!(
                f,
                "DINTDY- T (={t}) illegal: T not in interval TCUR - HU (= {tp}) to TCUR (={tn})"
            ),
        }
    }
}

impl std::error::Error for IntdyError {}

/// Computes the `k`-th derivative of the interpolating polynomial at `t`
/// into `dky[..n]` (`DINTDY`):
///
/// `dky(i) = sum_{j=k}^{nq} c(j,k) (t - tn)^(j-k) h^(-j) yh(i, j+1)`,
/// with `c(j,k) = j (j-1) ... (j-k+1)`, summed in reverse order.
///
/// `t` must lie in `[tn - hu, tn]` (up to a `100 * UROUND` relative
/// allowance) and `k <= nq`. On error `dky` is left untouched.
pub fn intdy(
    t: f64,
    k: usize,
    yh: &[f64],
    nyh: usize,
    n: usize,
    step: &StepData,
    dky: &mut [f64],
) -> Result<(), IntdyError> {
    let StepData { nq, tn, h, hu } = *step;
    if k > nq {
        return Err(IntdyError::IllegalK { k, nq });
    }
    let tp = tn - hu - 100.0 * UROUND * (tn.abs() + hu.abs()).copysign(hu);
    if (t - tp) * (t - tn) > 0.0 {
        return Err(IntdyError::IllegalT { t, tp, tn });
    }

    let l = nq + 1;
    assert!(nyh >= n);
    assert!(dky.len() >= n);
    if n > 0 {
        assert!(yh.len() >= (l - 1) * nyh + n);
    }
    let col = |j: usize| &yh[(j - 1) * nyh..(j - 1) * nyh + n];

    let s = (t - tn) / h;
    let mut ic: i64 = 1;
    if k != 0 {
        for jj in (l - k)..=nq {
            ic *= jj as i64;
        }
    }
    let c = ic as f64;
    for (d, y) in dky[..n].iter_mut().zip(col(l)) {
        *d = c * y;
    }
    if k != nq {
        let jb2 = nq - k;
        for jb in 1..=jb2 {
            let j = nq - jb;
            let jp1 = j + 1;
            let mut ic: i64 = 1;
            if k != 0 {
                for jj in (jp1 - k)..=j {
                    ic *= jj as i64;
                }
            }
            let c = ic as f64;
            for (d, y) in dky[..n].iter_mut().zip(col(jp1)) {
                *d = c * y + s * *d;
            }
        }
        if k == 0 {
            return Ok(());
        }
    }
    let r = h.powi(-(k as i32));
    for d in &mut dky[..n] {
        *d *= r;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds the Nordsieck array of a vector polynomial
    /// `y_i(t) = sum_m coef[i][m] (t - tn)^m` at `tn` with scale `h`:
    /// `yh(i, j+1) = h^j y_i^(j)(tn) / j! = h^j coef[i][j]`.
    fn nordsieck(coef: &[Vec<f64>], nq: usize, h: f64) -> Vec<f64> {
        let n = coef.len();
        let mut yh = vec![0.0; n * (nq + 1)];
        for (i, c) in coef.iter().enumerate() {
            for j in 0..=nq {
                yh[i + j * n] = h.powi(j as i32) * c.get(j).copied().unwrap_or(0.0);
            }
        }
        yh
    }

    /// `k`-th derivative of `sum_m coef[m] x^m` at `x`.
    fn poly_deriv(coef: &[f64], k: usize, x: f64) -> f64 {
        let mut acc = 0.0;
        for (m, &c) in coef.iter().enumerate().skip(k) {
            let falling: f64 = ((m - k + 1)..=m).map(|v| v as f64).product();
            acc += c * falling * x.powi((m - k) as i32);
        }
        acc
    }

    #[test]
    fn interpolate_reproduces_polynomials_and_derivatives() {
        let coef = vec![
            vec![1.0, -2.0, 0.5, 3.0, -0.25],
            vec![0.0, 1.0, 0.0, 0.0, 0.0],
            vec![-3.0, 0.75, 2.0, -1.0, 4.0],
        ];
        let (nq, tn, h, hu) = (4usize, 2.0, 0.3, 0.25);
        let yh = nordsieck(&coef, nq, h);
        let step = StepData { nq, tn, h, hu };
        let mut dky = [0.0; 3];
        for &t in &[tn, tn - 0.1, tn - hu, tn - 0.5 * hu] {
            for k in 0..=nq {
                intdy(t, k, &yh, 3, 3, &step, &mut dky).unwrap();
                for i in 0..3 {
                    let want = poly_deriv(&coef[i], k, t - tn);
                    let tol = 1e-12 * (1.0 + want.abs());
                    assert!((dky[i] - want).abs() <= tol, "k={k} t={t} i={i}: {} vs {want}", dky[i]);
                }
            }
        }
    }

    #[test]
    fn interpolate_negative_step_and_padded_rows() {
        // Integrating backwards (h, hu < 0), with nyh > n.
        let coef = vec![vec![2.0, 1.0, -1.0, 0.5]];
        let (nq, tn, h, hu) = (3usize, -1.0, -0.2, -0.4);
        let nyh = 2;
        let tight = nordsieck(&coef, nq, h);
        let mut yh = vec![f64::NAN; nyh * (nq + 1)];
        for j in 0..=nq {
            yh[j * nyh] = tight[j];
        }
        let step = StepData { nq, tn, h, hu };
        let mut dky = [0.0; 1];
        for &t in &[tn, tn + 0.3, tn - hu] {
            for k in 0..=nq {
                intdy(t, k, &yh, nyh, 1, &step, &mut dky).unwrap();
                let want = poly_deriv(&coef[0], k, t - tn);
                assert!((dky[0] - want).abs() <= 1e-12 * (1.0 + want.abs()), "k={k} t={t}");
            }
        }
    }

    #[test]
    fn interpolate_at_tn_returns_first_column_exactly() {
        let yh = [1.25, -7.5, 3.0, 4.0];
        let step = StepData { nq: 1, tn: 5.0, h: 0.5, hu: 0.5 };
        let mut dky = [0.0; 2];
        intdy(5.0, 0, &yh, 2, 2, &step, &mut dky).unwrap();
        assert_eq!(dky, [1.25, -7.5]);
        intdy(5.0, 1, &yh, 2, 2, &step, &mut dky).unwrap();
        assert_eq!(dky, [6.0, 8.0]);
    }

    #[test]
    fn interpolate_rejects_illegal_k() {
        let yh = [1.0, 2.0];
        let step = StepData { nq: 1, tn: 0.0, h: 1.0, hu: 1.0 };
        let mut dky = [9.0];
        let err = intdy(0.0, 2, &yh, 1, 1, &step, &mut dky).unwrap_err();
        assert_eq!(err, IntdyError::IllegalK { k: 2, nq: 1 });
        assert_eq!(err.iflag(), -1);
        assert_eq!(dky, [9.0]);
    }

    #[test]
    fn interpolate_rejects_t_outside_last_step() {
        let yh = [1.0, 2.0];
        let step = StepData { nq: 1, tn: 1.0, h: 0.5, hu: 0.5 };
        let mut dky = [9.0];
        for t in [1.1, 0.4] {
            let err = intdy(t, 0, &yh, 1, 1, &step, &mut dky).unwrap_err();
            assert_eq!(err.iflag(), -2);
            assert!(matches!(err, IntdyError::IllegalT { tn: 1.0, .. }));
        }
        assert_eq!(dky, [9.0]);
        // The roundoff allowance admits t slightly below tn - hu.
        let tp = 0.5 - 100.0 * UROUND * 1.5;
        assert!(intdy(0.5 - 50.0 * UROUND * 1.5, 0, &yh, 1, 1, &step, &mut dky).is_ok());
        assert!(intdy(tp - 1e-12, 0, &yh, 1, 1, &step, &mut dky).is_err());
    }
}
