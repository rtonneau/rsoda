//! Jacobian preparation for the stiff corrector (`DPRJA`).
//!
//! [`prja`] forms the Newton iteration matrix `P = I - h*el0*J` in an
//! [`IterationMatrix`] and LU-factors it, where `J` is the Jacobian `df/dy`
//! at the predicted solution. `J` comes from [`OdeSystem::jac`] for the
//! user-supplied Jacobian kinds (`MITER = 1, 4`) or from finite differences
//! of [`OdeSystem::rhs`] (`MITER = 2, 5`), with the increments of DPRJA.
//!
//! As in the Fortran code, `J` is first stored scaled by `-h*el0`, its norm
//! (consistent with the weighted max-norm of [`crate::norms::vmnorm`]) is
//! returned as `PDNORM`, then the identity is added and `P` is factored with
//! [`dgefa`] (dense) or [`dgbfa`] (banded).

use crate::consts::UROUND;
use crate::error::LsodaError;
use crate::linalg::{dgbfa, dgefa};
use crate::norms::{bnorm, fnorm, vmnorm};
use crate::options::JacobianKind;
use crate::solsy::{IterationMatrix, MatrixStorage};
use crate::system::OdeSystem;

/// The step data and vectors DPRJA reads (`COMMON /DLS001/` and the DSTODA
/// arguments). All slices have at least `n` entries.
#[derive(Debug, Clone, Copy)]
pub struct PrjaInput<'a> {
    /// Current time `TN`.
    pub tn: f64,
    /// Step size `H`.
    pub h: f64,
    /// `EL0 = EL(1)` of the current method and order.
    pub el0: f64,
    /// First column of the Nordsieck array, `YH(:, 1)`: the predicted `y`.
    /// The banded finite-difference path restores perturbed components of
    /// `y` from it, as DPRJA does.
    pub yh1: &'a [f64],
    /// **Inverted** error weights (`EWT` after DLSODA inverts it).
    pub ewt: &'a [f64],
    /// `f` evaluated at the predicted `y` (`SAVF`).
    pub savf: &'a [f64],
}

/// Result of a successful [`prja`] call. The Jacobian is then current
/// (`JCUR = 1`) and `NJE` has been incremented.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrjaOutput {
    /// Weighted norm of the Jacobian `J` (`PDNORM`), used by DSTODA's
    /// stiffness test.
    pub pdnorm: f64,
    /// `true` if the LU factorization found `P` singular (`IERPJ = 1`). The
    /// factors must then not be used to solve; DSTODA treats this as a
    /// convergence failure.
    pub singular: bool,
}

impl PrjaOutput {
    /// The Fortran `IERPJ` flag: `0` (no trouble) or `1` (`P` singular).
    pub fn ierpj(&self) -> i32 {
        i32::from(self.singular)
    }
}

/// Computes and LU-factors `P = I - h*el0*J` into `wm` (`DPRJA`).
///
/// * `kind` selects how `J` is obtained and must match `wm.storage`.
/// * `y` holds the predicted solution (`Y`, equal to `input.yh1`); the
///   finite-difference paths perturb it temporarily and restore it before
///   returning.
/// * `ftem` (length >= `n`) is scratch space for `f` evaluations (`ACOR` in
///   DSTODA).
/// * `nfe` is increased by the number of `f` evaluations (`n` dense, or
///   `min(ml + mu + 1, n)` banded) and `nje` by one, as in DPRJA. `wm.hl0`
///   is set to `h*el0`.
///
/// A callback `Err` is returned unchanged. As in the Fortran, `nje` has then
/// already been incremented while `nfe` has not; `y` is restored (banded
/// finite differences restore from `input.yh1`) and `wm` holds a partial
/// matrix. A singular `P` is not an error: it is reported through
/// [`PrjaOutput::singular`] (`IERPJ = 1`).
#[allow(clippy::too_many_arguments)]
pub fn prja<S: OdeSystem + ?Sized>(
    system: &mut S,
    kind: JacobianKind,
    input: &PrjaInput<'_>,
    y: &mut [f64],
    ftem: &mut [f64],
    wm: &mut IterationMatrix,
    nfe: &mut usize,
    nje: &mut usize,
) -> Result<PrjaOutput, LsodaError> {
    let n = wm.n;
    assert_eq!(MatrixStorage::for_jacobian(&kind), wm.storage, "Jacobian kind and storage differ");
    assert!(y.len() >= n && ftem.len() >= n);
    assert!(input.yh1.len() >= n && input.ewt.len() >= n && input.savf.len() >= n);

    *nje += 1;
    let PrjaInput { tn, h, el0, yh1, ewt, savf } = *input;
    let hl0 = h * el0;
    wm.hl0 = hl0;
    let srur = UROUND.sqrt(); // WM(1)

    match kind {
        JacobianKind::DenseUser | JacobianKind::DenseFd => {
            let lenp = n * n;
            let a = &mut wm.a[..lenp];
            if kind == JacobianKind::DenseUser {
                // MITER = 1: call JAC and multiply by -hl0.
                a.fill(0.0);
                system.jac(tn, &y[..n], a)?;
                let con = -hl0;
                for v in a.iter_mut() {
                    *v *= con;
                }
            } else {
                // MITER = 2: make n calls to f to approximate J.
                let fac = vmnorm(&savf[..n], ewt);
                let mut r0 = 1000.0 * h.abs() * UROUND * n as f64 * fac;
                if r0 == 0.0 {
                    r0 = 1.0;
                }
                for j in 0..n {
                    let yj = y[j];
                    let r = (srur * yj.abs()).max(r0 / ewt[j]);
                    y[j] += r;
                    let fac = -hl0 / r;
                    let res = system.rhs(tn, &y[..n], &mut ftem[..n]);
                    y[j] = yj;
                    res?;
                    let col = &mut a[j * n..(j + 1) * n];
                    for i in 0..n {
                        col[i] = (ftem[i] - savf[i]) * fac;
                    }
                }
                *nfe += n;
            }
            // Norm of J, then add the identity.
            let pdnorm = fnorm(n, a, ewt) / hl0.abs();
            for i in 0..n {
                a[i * (n + 1)] += 1.0;
            }
            let info = dgefa(&mut wm.a, n, n, &mut wm.ipvt);
            Ok(PrjaOutput { pdnorm, singular: info != 0 })
        }
        JacobianKind::BandedUser { ml, mu } | JacobianKind::BandedFd { ml, mu } => {
            let mband = ml + mu + 1;
            let meband = mband + ml;
            let lenp = meband * n;
            let a = &mut wm.a[..lenp];
            if let JacobianKind::BandedUser { .. } = kind {
                // MITER = 4: call JAC and multiply by -hl0. DPRJA hands JAC
                // WM(ML+3) with leading dimension MEBAND; the OdeSystem
                // contract uses leading dimension MBAND, so the band is
                // copied into place (no arithmetic, same values).
                a.fill(0.0);
                let mut pd = vec![0.0; mband * n];
                system.jac(tn, &y[..n], &mut pd)?;
                for j in 0..n {
                    let src = &pd[j * mband..(j + 1) * mband];
                    a[ml + j * meband..ml + j * meband + mband].copy_from_slice(src);
                }
                let con = -hl0;
                for v in a.iter_mut() {
                    *v *= con;
                }
            } else {
                // MITER = 5: make min(mband, n) calls to f, perturbing each
                // group of columns j, j + mband, j + 2*mband, ... together.
                let mba = mband.min(n);
                let fac = vmnorm(&savf[..n], ewt);
                let mut r0 = 1000.0 * h.abs() * UROUND * n as f64 * fac;
                if r0 == 0.0 {
                    r0 = 1.0;
                }
                for j in 0..mba {
                    for i in (j..n).step_by(mband) {
                        let yi = y[i];
                        let r = (srur * yi.abs()).max(r0 / ewt[i]);
                        y[i] += r;
                    }
                    if let Err(e) = system.rhs(tn, &y[..n], &mut ftem[..n]) {
                        for jj in (j..n).step_by(mband) {
                            y[jj] = yh1[jj];
                        }
                        return Err(e);
                    }
                    for jj in (j..n).step_by(mband) {
                        y[jj] = yh1[jj];
                        let yjj = y[jj];
                        let r = (srur * yjj.abs()).max(r0 / ewt[jj]);
                        let fac = -hl0 / r;
                        let i1 = jj.saturating_sub(mu);
                        let i2 = (jj + ml).min(n - 1);
                        // Element (i, jj) of J at band row i - jj + ml + mu
                        // (0-based) of column jj.
                        for i in i1..=i2 {
                            a[i + ml + mu - jj + jj * meband] = (ftem[i] - savf[i]) * fac;
                        }
                    }
                }
                *nfe += mba;
            }
            // Norm of J (band stored from row ml), then add the identity.
            let pdnorm = bnorm(n, &a[ml..], meband, ml, mu, ewt) / hl0.abs();
            for i in 0..n {
                a[(mband - 1) + i * meband] += 1.0;
            }
            let info = dgbfa(&mut wm.a, meband, n, ml, mu, &mut wm.ipvt);
            Ok(PrjaOutput { pdnorm, singular: info != 0 })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::ClosureSystem;

    /// Pentadiagonal-ish test Jacobian (ml = 1, mu = 2), 0-based, zero
    /// outside the band.
    const ML: usize = 1;
    const MU: usize = 2;
    fn jac_elem(i: usize, j: usize) -> f64 {
        if i + MU >= j && j + ML >= i {
            if i == j { -3.0 - i as f64 } else { 0.5 + 0.25 * (i + 2 * j) as f64 }
        } else {
            0.0
        }
    }

    fn linear_rhs(n: usize) -> impl FnMut(f64, &[f64], &mut [f64]) -> Result<(), LsodaError> {
        move |_t, y, f| {
            for i in 0..n {
                f[i] = (0..n).map(|j| jac_elem(i, j) * y[j]).sum();
            }
            Ok(())
        }
    }

    fn dense_jac(n: usize) -> impl FnMut(f64, &[f64], &mut [f64]) -> Result<(), LsodaError> {
        move |_t, _y, pd| {
            for j in 0..n {
                for i in 0..n {
                    pd[i + j * n] = jac_elem(i, j);
                }
            }
            Ok(())
        }
    }

    fn band_jac(n: usize) -> impl FnMut(f64, &[f64], &mut [f64]) -> Result<(), LsodaError> {
        move |_t, _y, pd| {
            let nrowpd = ML + MU + 1;
            for j in 0..n {
                for i in j.saturating_sub(MU)..=(j + ML).min(n - 1) {
                    pd[(i + MU - j) + j * nrowpd] = jac_elem(i, j);
                }
            }
            Ok(())
        }
    }

    struct Setup {
        y: Vec<f64>,
        ewt: Vec<f64>,
        savf: Vec<f64>,
    }

    fn setup(n: usize) -> Setup {
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 0.3 * i as f64).collect();
        let ewt: Vec<f64> = (0..n).map(|i| 1.0 / (1e-6 * y[i].abs() + 1e-8)).collect();
        let mut savf = vec![0.0; n];
        linear_rhs(n)(0.0, &y, &mut savf).unwrap();
        Setup { y, ewt, savf }
    }

    /// Runs prja and checks the factored matrix against `P = I - hl0*J`.
    fn check_kind(kind: JacobianKind, n: usize, tol: f64) {
        let h = 0.05;
        let el0 = 0.8;
        let hl0 = h * el0;
        let s = setup(n);
        let mut sys = ClosureSystem::new(linear_rhs(n));
        sys = match kind {
            JacobianKind::DenseUser => sys.with_jac(dense_jac(n)),
            JacobianKind::BandedUser { .. } => sys.with_jac(band_jac(n)),
            _ => sys,
        };
        let mut wm = IterationMatrix::new(n, MatrixStorage::for_jacobian(&kind));
        let mut y = s.y.clone();
        let mut ftem = vec![0.0; n];
        let (mut nfe, mut nje) = (3usize, 7usize);
        let input = PrjaInput { tn: 0.0, h, el0, yh1: &s.y, ewt: &s.ewt, savf: &s.savf };
        let out = prja(&mut sys, kind, &input, &mut y, &mut ftem, &mut wm, &mut nfe, &mut nje).unwrap();

        assert!(!out.singular);
        assert_eq!(out.ierpj(), 0);
        assert_eq!(nje, 8);
        let expected_nfe = match kind {
            JacobianKind::DenseUser | JacobianKind::BandedUser { .. } => 3,
            JacobianKind::DenseFd => 3 + n,
            JacobianKind::BandedFd { ml, mu } => 3 + (ml + mu + 1).min(n),
        };
        assert_eq!(nfe, expected_nfe, "{kind:?}");
        assert_eq!(y, s.y, "y must be restored");
        assert_eq!(wm.hl0, hl0);

        // PDNORM = weighted norm of J.
        let jn = (0..n)
            .map(|i| s.ewt[i] * (0..n).map(|j| jac_elem(i, j).abs() / s.ewt[j]).sum::<f64>())
            .fold(0.0, f64::max);
        assert!((out.pdnorm - jn).abs() <= tol * jn, "{kind:?}: pdnorm {} vs {jn}", out.pdnorm);

        // The factors solve P x = b with P = I - hl0*J.
        let p = |i: usize, j: usize| (if i == j { 1.0 } else { 0.0 }) - hl0 * jac_elem(i, j);
        let x_true: Vec<f64> = (0..n).map(|i| 1.0 - 0.5 * i as f64).collect();
        let mut x: Vec<f64> = (0..n).map(|i| (0..n).map(|j| p(i, j) * x_true[j]).sum()).collect();
        wm.solve(h, el0, &mut x).unwrap();
        for i in 0..n {
            assert!((x[i] - x_true[i]).abs() <= tol, "{kind:?}: {x:?} vs {x_true:?}");
        }
    }

    #[test]
    fn prja_dense_user_factors_p() {
        check_kind(JacobianKind::DenseUser, 5, 1e-13);
    }

    #[test]
    fn prja_dense_fd_factors_p() {
        check_kind(JacobianKind::DenseFd, 5, 1e-6);
    }

    #[test]
    fn prja_banded_user_factors_p() {
        check_kind(JacobianKind::BandedUser { ml: ML, mu: MU }, 6, 1e-13);
        // n smaller than the full bandwidth ml + mu + 1.
        check_kind(JacobianKind::BandedUser { ml: ML, mu: MU }, 3, 1e-13);
    }

    #[test]
    fn prja_banded_fd_factors_p() {
        check_kind(JacobianKind::BandedFd { ml: ML, mu: MU }, 7, 1e-6);
        // mba = min(mband, n) = n.
        check_kind(JacobianKind::BandedFd { ml: ML, mu: MU }, 3, 1e-6);
    }

    #[test]
    fn prja_user_dense_matches_unfactored_formula_exactly() {
        // 1x1: P = 1 - hl0*j, factored in place is P itself.
        let mut sys = ClosureSystem::new(|_, _, f: &mut [f64]| {
            f[0] = 0.0;
            Ok(())
        })
        .with_jac(|_, _, pd: &mut [f64]| {
            pd[0] = -4.0;
            Ok(())
        });
        let mut wm = IterationMatrix::new(1, MatrixStorage::Dense);
        let (mut y, mut ftem) = ([2.0], [0.0]);
        let input = PrjaInput { tn: 0.0, h: 0.5, el0: 0.5, yh1: &[2.0], ewt: &[0.5], savf: &[0.0] };
        let (mut nfe, mut nje) = (0, 0);
        let out = prja(&mut sys, JacobianKind::DenseUser, &input, &mut y, &mut ftem, &mut wm, &mut nfe, &mut nje)
            .unwrap();
        assert_eq!(wm.a[0], 1.0 + 0.25 * 4.0);
        // pdnorm = |(-hl0)*j| * w / w / |hl0| = 4.
        assert_eq!(out.pdnorm, 4.0);
    }

    /// Records the perturbation `y - y0` seen at each rhs call.
    fn recording_system<'a>(
        y0: Vec<f64>,
        log: &'a std::cell::RefCell<Vec<Vec<f64>>>,
    ) -> ClosureSystem<'a> {
        ClosureSystem::new(move |_t, y: &[f64], f: &mut [f64]| {
            log.borrow_mut().push(y.iter().zip(&y0).map(|(a, b)| a - b).collect());
            for i in 0..y.len() {
                f[i] = 2.0 * y[i];
            }
            Ok(())
        })
    }

    #[test]
    fn prja_dense_fd_uses_fortran_increments() {
        let n = 3;
        let y0 = vec![1e3, -2.0, 1e-12];
        let ewt = vec![1e-2, 4.0, 1e6]; // inverted weights
        let savf = vec![3.0, -1.0, 0.5];
        let (h, el0) = (1e-3, 1.0);
        let log = std::cell::RefCell::new(Vec::new());
        let mut sys = recording_system(y0.clone(), &log);
        let mut wm = IterationMatrix::new(n, MatrixStorage::Dense);
        let mut y = y0.clone();
        let mut ftem = vec![0.0; n];
        let input = PrjaInput { tn: 0.0, h, el0, yh1: &y0, ewt: &ewt, savf: &savf };
        let (mut nfe, mut nje) = (0, 0);
        prja(&mut sys, JacobianKind::DenseFd, &input, &mut y, &mut ftem, &mut wm, &mut nfe, &mut nje).unwrap();

        let srur = UROUND.sqrt();
        let fac = vmnorm(&savf, &ewt);
        let r0 = 1000.0 * h.abs() * UROUND * n as f64 * fac;
        let log = log.borrow();
        assert_eq!(log.len(), n);
        for j in 0..n {
            let r = (srur * y0[j].abs()).max(r0 / ewt[j]);
            for i in 0..n {
                let expect = if i == j { (y0[j] + r) - y0[j] } else { 0.0 };
                assert_eq!(log[j][i], expect, "call {j}, component {i}");
            }
        }
        assert_eq!((nfe, nje), (n, 1));
        assert_eq!(y, y0);
    }

    #[test]
    fn prja_fd_r0_defaults_to_one_when_savf_zero() {
        let n = 2;
        let y0 = vec![0.0, 0.0];
        let ewt = vec![1.0, 0.5];
        let savf = vec![0.0, 0.0];
        let log = std::cell::RefCell::new(Vec::new());
        let mut sys = recording_system(y0.clone(), &log);
        let mut wm = IterationMatrix::new(n, MatrixStorage::Dense);
        let mut y = y0.clone();
        let mut ftem = vec![0.0; n];
        let input = PrjaInput { tn: 0.0, h: 0.1, el0: 1.0, yh1: &y0, ewt: &ewt, savf: &savf };
        let (mut nfe, mut nje) = (0, 0);
        prja(&mut sys, JacobianKind::DenseFd, &input, &mut y, &mut ftem, &mut wm, &mut nfe, &mut nje).unwrap();
        // r = max(0, 1 / ewt(j)).
        let log = log.borrow();
        assert_eq!(log[0], vec![1.0, 0.0]);
        assert_eq!(log[1], vec![0.0, 2.0]);
    }

    #[test]
    fn prja_banded_fd_groups_columns() {
        // n = 7, ml = 1, mu = 0: mband = 2 -> 2 calls, columns {0,2,4,6}, {1,3,5}.
        let n = 7;
        let (ml, mu) = (1, 0);
        let y0: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
        let ewt = vec![1.0; n];
        let savf: Vec<f64> = y0.iter().map(|v| 2.0 * v).collect();
        let (h, el0) = (0.01, 1.0);
        let log = std::cell::RefCell::new(Vec::new());
        let mut sys = recording_system(y0.clone(), &log);
        let kind = JacobianKind::BandedFd { ml, mu };
        let mut wm = IterationMatrix::new(n, MatrixStorage::for_jacobian(&kind));
        let mut y = y0.clone();
        let mut ftem = vec![0.0; n];
        let input = PrjaInput { tn: 0.0, h, el0, yh1: &y0, ewt: &ewt, savf: &savf };
        let (mut nfe, mut nje) = (0, 0);
        prja(&mut sys, kind, &input, &mut y, &mut ftem, &mut wm, &mut nfe, &mut nje).unwrap();

        let srur = UROUND.sqrt();
        let r0 = 1000.0 * h * UROUND * n as f64 * vmnorm(&savf, &ewt);
        let log = log.borrow();
        assert_eq!(log.len(), 2);
        assert_eq!((nfe, nje), (2, 1));
        for (call, perturbed) in log.iter().enumerate() {
            for i in 0..n {
                let expect = if i % 2 == call {
                    let r = (srur * y0[i].abs()).max(r0 / ewt[i]);
                    (y0[i] + r) - y0[i]
                } else {
                    0.0
                };
                assert_eq!(perturbed[i], expect, "call {call}, component {i}");
            }
        }
        assert_eq!(y, y0);
    }

    #[test]
    fn prja_callback_errors_propagate() {
        let err = LsodaError::Callback { ierr: -3 };
        let n = 3;
        let s = setup(n);
        for kind in [
            JacobianKind::DenseUser,
            JacobianKind::DenseFd,
            JacobianKind::BandedUser { ml: 1, mu: 1 },
            JacobianKind::BandedFd { ml: 1, mu: 1 },
        ] {
            let e1 = err.clone();
            let e2 = err.clone();
            let mut sys = ClosureSystem::new(move |_, _, _| Err(e1.clone()))
                .with_jac(move |_, _, _| Err(e2.clone()));
            let mut wm = IterationMatrix::new(n, MatrixStorage::for_jacobian(&kind));
            let mut y = s.y.clone();
            let mut ftem = vec![0.0; n];
            let input = PrjaInput { tn: 0.0, h: 0.1, el0: 1.0, yh1: &s.y, ewt: &s.ewt, savf: &s.savf };
            let (mut nfe, mut nje) = (5, 2);
            let r = prja(&mut sys, kind, &input, &mut y, &mut ftem, &mut wm, &mut nfe, &mut nje);
            assert_eq!(r, Err(err.clone()), "{kind:?}");
            // As in DPRJA: NJE already incremented, NFE not.
            assert_eq!((nfe, nje), (5, 3), "{kind:?}");
            assert_eq!(y, s.y, "{kind:?}");
        }
    }

    #[test]
    fn prja_singular_matrix_sets_ierpj() {
        // J = diag(1, 2, 4) with hl0 = 0.5: P = diag(0.5, 0, -1) is singular.
        let n = 3;
        let diag = [1.0, 2.0, 4.0];
        let y0 = vec![1.0; n];
        let ewt = vec![1.0; n];
        let savf: Vec<f64> = diag.to_vec();
        for kind in [
            JacobianKind::DenseUser,
            JacobianKind::DenseFd,
            JacobianKind::BandedUser { ml: 0, mu: 1 },
            JacobianKind::BandedFd { ml: 0, mu: 1 },
        ] {
            let mut sys = ClosureSystem::new(move |_, y: &[f64], f: &mut [f64]| {
                for i in 0..y.len() {
                    f[i] = diag[i] * y[i];
                }
                Ok(())
            })
            .with_jac(move |_, _, pd: &mut [f64]| {
                // Dense (pd[j + 3j]) and band (ml = 0, mu = 1, nrowpd = 2:
                // diagonal at row mu, pd[1 + 2j]) layouts.
                let dense = pd.len() == 9;
                for j in 0..3 {
                    let k = if dense { j + 3 * j } else { 1 + 2 * j };
                    pd[k] = diag[j];
                }
                Ok(())
            });
            let mut wm = IterationMatrix::new(n, MatrixStorage::for_jacobian(&kind));
            let mut y = y0.clone();
            let mut ftem = vec![0.0; n];
            // Powers of two keep the finite differences exact.
            let input = PrjaInput { tn: 0.0, h: 0.25, el0: 2.0, yh1: &y0, ewt: &ewt, savf: &savf };
            let (mut nfe, mut nje) = (0, 0);
            let out = prja(&mut sys, kind, &input, &mut y, &mut ftem, &mut wm, &mut nfe, &mut nje).unwrap();
            assert!(out.singular, "{kind:?}");
            assert_eq!(out.ierpj(), 1);
            assert_eq!(out.pdnorm, 4.0, "{kind:?}");
            assert_eq!(nje, 1);
        }
    }
}
