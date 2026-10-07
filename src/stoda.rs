//! The core integrator: one step of LSODA (`DSTODA`).
//!
//! [`stoda`] performs one step of the integration with automatic switching
//! between the nonstiff Adams methods (functional iteration) and the stiff
//! BDF methods (chord iteration with the Jacobian kind `jtyp`). It predicts
//! by the Pascal triangle, iterates the corrector, makes the local error
//! test, selects the next step size and order, and rescales the Nordsieck
//! array. Once 20 steps have passed since the start or the last method
//! switch (`ICOUNT`), it compares after every successful step the step sizes
//! the two methods could have used, and switches when the other method wins
//! by the required margin.
//!
//! The port follows DSTODA statement by statement: the Fortran `GO TO`
//! structure is replaced by a loop over a private `Label` enum (one variant
//! per jump target, named after the Fortran statement label), and every
//! constant and the order of floating-point operations are kept. Fortran
//! `MAX`/`MIN` map to [`f64::max`]/[`f64::min`] and `x**y` with a real
//! exponent to [`f64::powf`].
//!
//! The caller (DLSODA) owns the protocol around a step: it refreshes the
//! inverted error weights `ewt` from `YH(:,1)` before each call, and after a
//! successful step that switched methods it calls
//! [`State::record_method_switch`] so the next call (with `JSTART = -1`)
//! completes the switch.

use crate::coefficients::{Method, cfode};
use crate::consts::UROUND;
use crate::error::LsodaError;
use crate::norms::vmnorm;
use crate::prja::{PrjaInput, prja};
use crate::state::State;
use crate::system::OdeSystem;

/// `SM1(12)`: step-size factors bounding `h * |J|` inside the stability
/// region of the Adams method of each order (`DATA SM1`).
pub const SM1: [f64; 12] = [
    0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.20, 0.15, 0.10, 0.075, 0.050, 0.025,
];

/// Outcome of a [`stoda`] call (the Fortran `KFLAG`, also left in
/// [`State::kflag`]).
///
/// On a failure, `tn` and `yh` are as of the beginning of the last step and
/// `h` is the last step size attempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// `KFLAG = 0`: the step was successful.
    Success,
    /// `KFLAG = -1`: the requested error could not be achieved, either with
    /// `|h| = hmin` or after 10 error test failures.
    ErrorTestFailed,
    /// `KFLAG = -2`: corrector convergence could not be achieved, either
    /// with `|h| = hmin` or after `mxncf` convergence failures.
    ConvergenceFailed,
    /// `KFLAG = -3`: fatal error in the Jacobian preparation or the linear
    /// solve. Kept for completeness: neither reports such errors in LSODA.
    SolverFailed,
}

impl StepOutcome {
    /// The outcome for a `KFLAG` value.
    pub fn from_kflag(kflag: i32) -> Self {
        match kflag {
            0 => Self::Success,
            -1 => Self::ErrorTestFailed,
            -2 => Self::ConvergenceFailed,
            _ => Self::SolverFailed,
        }
    }

    /// The Fortran `KFLAG` value.
    pub fn kflag(self) -> i32 {
        match self {
            Self::Success => 0,
            Self::ErrorTestFailed => -1,
            Self::ConvergenceFailed => -2,
            Self::SolverFailed => -3,
        }
    }
}

/// Jump targets of DSTODA, named after the Fortran statement labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    /// First call (`JSTART = 0`): initialisation before statement 150.
    Start,
    /// Parameter changes (`JSTART = -1`).
    L100,
    /// Reset `EL` and related constants for the current method and order.
    L150,
    /// New `h` requested by the caller (`JSTART = -2`).
    L160,
    /// Change `h` by the ratio `RH`, bounded below by `HMIN`.
    L170,
    /// Change `h` by `RH`, bounded by `RMAX`, `HMXI` and (Adams) stability.
    L175,
    /// Predict.
    L200,
    /// Start (or restart) the corrector iteration.
    L220,
    /// Clear `ACOR` before the first corrector iteration.
    L250,
    /// One corrector iteration (functional, or chord from 350).
    L270,
    /// Corrector convergence test.
    L400,
    /// Another corrector iteration, if allowed.
    L405,
    /// Corrector failed: retry with a new Jacobian if it is out of date.
    L410,
    /// Corrector failed: retract `yh` and reduce `h`.
    L430,
    /// Corrector converged: local error test.
    L450,
    /// Adams method in use: consider switching to BDF (statements after
    /// `IF (METH .EQ. 2) GO TO 480` up to 478).
    L460,
    /// BDF method in use: consider switching to Adams.
    L480,
    /// No switch: usual step-size and order selection countdown.
    L488,
    /// Error test failed.
    L500,
    /// Compute `RHUP` for an order increase.
    L520,
    /// Compute `RHSM`, `RHDN` and choose the new order and step size
    /// (statements 540 to 630).
    L540,
    /// Three or more error test failures: restart at order 1.
    L640,
    /// Successful step with a change of `h`: `RMAX = 10`.
    L690,
    /// Successful step: scale `ACOR` to the local error estimate.
    L700,
}

/// The order chosen at statement 560: same order (570), one lower (580) or
/// one higher (590).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pick {
    L570,
    L580,
    L590,
}

/// The `GO TO (160, 170, 200), IRET` dispatch at the end of block 150.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Iret {
    To160,
    To170,
    To200,
}

/// Performs one step of the integration (`DSTODA`).
///
/// All inputs and outputs go through `st` (see [`State`]): `st.jstart`
/// selects first step (0), continuation (> 0), continuation after changes to
/// the method or other parameters (-1), or continuation with the new step
/// size `st.h` (-2). On return `st.jstart = 1`, `st.hold = st.h` and
/// `st.kflag` holds the completion code, also returned as a
/// [`StepOutcome`]. After a successful step `st.acor` holds the estimated
/// local error and `st.mused` the method that was used; if `st.meth` now
/// differs from it, a method switch has been made.
///
/// `st.ewt` must hold the inverted error weights for `YH(:,1)`.
///
/// # Errors
///
/// A callback error from `system` is returned at once, as the fork's DSTODA
/// returns on `ierr < 0`: the counter of the failed evaluation is not
/// incremented and `st` is left mid-step (`tn` advanced, `yh` predicted), so
/// the integration cannot be continued from it.
pub fn stoda<S: OdeSystem + ?Sized>(st: &mut State, system: &mut S) -> Result<StepOutcome, LsodaError> {
    use Label::*;

    let n = st.n;
    let nyh = st.nyh;
    // Indices of YH(1:N, j) in the flat history array.
    let col = move |j: usize| (j - 1) * nyh..(j - 1) * nyh + n;

    // Locals shared between jump targets. Every path assigns them before use;
    // the initial values only satisfy the compiler.
    let mut iredo: i32 = 0;
    let mut iret = Iret::To200;
    let mut m: usize = 0;
    let mut del = 0.0_f64;
    let mut dsm = 0.0_f64;
    let mut pnorm = 0.0_f64;
    let mut rate = 0.0_f64;
    let mut rh = 0.0_f64;
    let mut rhup = 0.0_f64;

    st.kflag = 0;
    let told = st.tn;
    let mut ncf: usize = 0;
    st.ierpj = 0;
    st.iersl = 0;
    st.jcur = false;
    st.icf = 0;
    let mut delp = 0.0_f64;

    let mut next = if st.jstart > 0 {
        L200
    } else if st.jstart == -1 {
        L100
    } else if st.jstart == -2 {
        L160
    } else {
        Start
    };

    loop {
        next = match next {
            // -----------------------------------------------------------------
            // On the first call, the order is set to 1, and other variables are
            // initialized. RMAX is the maximum ratio by which H can be increased
            // in a single step. It is initially 1.E4 to compensate for the small
            // initial H, but then is normally equal to 10. If a failure occurs
            // (in corrector convergence or error test), RMAX is set at 2 for the
            // next increase. DCFODE is called to get the needed coefficients for
            // both methods.
            // -----------------------------------------------------------------
            Start => {
                st.lmax = st.maxord + 1;
                st.nq = 1;
                st.l = 2;
                st.ialth = 2;
                st.rmax = 10000.0;
                st.rc = 0.0;
                st.el0 = 1.0;
                st.crate_ = 0.7;
                st.hold = st.h;
                st.nslp = 0;
                st.ipup = st.miter.is_some();
                iret = Iret::To200;
                // Initialize switching parameters. METH = 1 is assumed initially.
                st.icount = 20;
                st.irflag = false;
                st.pdest = 0.0;
                st.pdlast = 0.0;
                st.ratio = 5.0;
                cfode(Method::Bdf, &mut st.elco, &mut st.tesco);
                for i in 1..=5 {
                    st.cm2[i - 1] = st.tesco[i - 1][1] * st.elco[i - 1][i];
                }
                cfode(Method::Adams, &mut st.elco, &mut st.tesco);
                for i in 1..=12 {
                    st.cm1[i - 1] = st.tesco[i - 1][1] * st.elco[i - 1][i];
                }
                L150
            }

            // -----------------------------------------------------------------
            // Preliminaries needed when JSTART = -1. IPUP is set to MITER to
            // force a matrix update. If an order increase is about to be
            // considered (IALTH = 1), IALTH is reset to 2 to postpone
            // consideration one more step. If the caller has changed METH,
            // DCFODE is called to reset the coefficients of the method. If H is
            // to be changed, YH must be rescaled. If H or METH is being changed,
            // IALTH is reset to L = NQ + 1 to prevent further changes in H for
            // that many steps.
            // -----------------------------------------------------------------
            L100 => {
                st.ipup = st.miter.is_some();
                st.lmax = st.maxord + 1;
                if st.ialth == 1 {
                    st.ialth = 2;
                }
                if st.mused == Some(st.meth) {
                    L160
                } else {
                    cfode(st.meth, &mut st.elco, &mut st.tesco);
                    st.ialth = st.l as i32;
                    iret = Iret::To160;
                    L150
                }
            }

            // The el vector and related constants are reset whenever the order
            // NQ is changed, or at the start of the problem.
            L150 => {
                for i in 1..=st.l {
                    st.el[i - 1] = st.elco[st.nq - 1][i - 1];
                }
                st.nqnyh = st.nq * nyh;
                st.rc = st.rc * st.el[0] / st.el0;
                st.el0 = st.el[0];
                st.conit = 0.5 / (st.nq + 2) as f64;
                match iret {
                    Iret::To160 => L160,
                    Iret::To170 => L170,
                    Iret::To200 => L200,
                }
            }

            // -----------------------------------------------------------------
            // If H is being changed, the H ratio RH is checked against RMAX,
            // HMIN, and HMXI, and the YH array rescaled. IALTH is set to
            // L = NQ + 1 to prevent a change of H for that many steps, unless
            // forced by a convergence or error test failure.
            // -----------------------------------------------------------------
            L160 => {
                if st.h == st.hold {
                    L200
                } else {
                    rh = st.h / st.hold;
                    st.h = st.hold;
                    iredo = 3;
                    L175
                }
            }
            L170 => {
                rh = rh.max(st.hmin / st.h.abs());
                L175
            }
            L175 => {
                rh = rh.min(st.rmax);
                rh /= 1.0_f64.max(st.h.abs() * st.hmxi * rh);
                // If METH = 1, also restrict the new step size by the
                // stability region. If this reduces H, set IRFLAG to 1 so that
                // if there are roundoff problems later, we can assume that is
                // the cause of the trouble.
                if st.meth == Method::Adams {
                    st.irflag = false;
                    let pdh = (st.h.abs() * st.pdlast).max(0.000001);
                    if rh * pdh * 1.00001 >= SM1[st.nq - 1] {
                        rh = SM1[st.nq - 1] / pdh;
                        st.irflag = true;
                    }
                }
                // 178
                let mut r = 1.0_f64;
                for j in 2..=st.l {
                    r *= rh;
                    for i in col(j) {
                        st.yh[i] *= r;
                    }
                }
                st.h *= rh;
                st.rc *= rh;
                st.ialth = st.l as i32;
                if iredo == 0 { L690 } else { L200 }
            }

            // -----------------------------------------------------------------
            // This section computes the predicted values by effectively
            // multiplying the YH array by the Pascal triangle matrix. RC is the
            // ratio of new to old values of the coefficient H*EL(1). When RC
            // differs from 1 by more than CCMAX, IPUP is set to MITER to force
            // PJAC to be called, if a Jacobian is involved. In any case, PJAC is
            // called at least every MSBP steps.
            // -----------------------------------------------------------------
            L200 => {
                if (st.rc - 1.0).abs() > st.ccmax {
                    st.ipup = st.miter.is_some();
                }
                if st.nst >= st.nslp + st.msbp {
                    st.ipup = st.miter.is_some();
                }
                st.tn += st.h;
                predict(&mut st.yh, st.nqnyh, nyh, st.nq);
                pnorm = vmnorm(&st.yh[..n], &st.ewt);
                L220
            }

            // -----------------------------------------------------------------
            // Up to MAXCOR corrector iterations are taken. A convergence test is
            // made on the max-norm of each correction, weighted by the error
            // weight vector EWT. The sum of the corrections is accumulated in
            // the vector ACOR(i). The YH array is not altered in the corrector
            // loop.
            // -----------------------------------------------------------------
            L220 => {
                m = 0;
                rate = 0.0;
                del = 0.0;
                st.y[..n].copy_from_slice(&st.yh[..n]);
                system.rhs(st.tn, &st.y[..n], &mut st.savf[..n])?;
                st.nfe += 1;
                // IPUP > 0 implies MITER > 0 (IPUP is only ever set to MITER).
                match (st.ipup, st.miter) {
                    (true, Some(kind)) => {
                        // If indicated, the matrix P = I - H*EL(1)*J is
                        // reevaluated and preprocessed before starting the
                        // corrector iteration. IPUP is set to 0 as an indicator
                        // that this has been done.
                        let input = PrjaInput {
                            tn: st.tn,
                            h: st.h,
                            el0: st.el0,
                            yh1: &st.yh[..n],
                            ewt: &st.ewt,
                            savf: &st.savf,
                        };
                        let out = prja(
                            system,
                            kind,
                            &input,
                            &mut st.y,
                            &mut st.acor,
                            &mut st.wm,
                            &mut st.nfe,
                            &mut st.nje,
                        )?;
                        // DPRJA's outputs IERPJ, JCUR = 1 and PDNORM.
                        st.ierpj = out.ierpj();
                        st.jcur = true;
                        st.pdnorm = out.pdnorm;
                        st.ipup = false;
                        st.rc = 1.0;
                        st.nslp = st.nst;
                        st.crate_ = 0.7;
                        if st.ierpj != 0 { L430 } else { L250 }
                    }
                    _ => L250,
                }
            }
            L250 => {
                st.acor[..n].fill(0.0);
                L270
            }
            L270 => {
                if st.miter.is_none() {
                    // In the case of functional iteration, update Y directly
                    // from the result of the last function evaluation.
                    for i in 0..n {
                        st.savf[i] = st.h * st.savf[i] - st.yh[nyh + i];
                        st.y[i] = st.savf[i] - st.acor[i];
                    }
                    del = vmnorm(&st.y[..n], &st.ewt);
                    for i in 0..n {
                        st.y[i] = st.yh[i] + st.el[0] * st.savf[i];
                        st.acor[i] = st.savf[i];
                    }
                    L400
                } else {
                    // 350: In the case of the chord method, compute the
                    // corrector error, and solve the linear system with that as
                    // right-hand side and P as coefficient matrix.
                    for i in 0..n {
                        st.y[i] = st.h * st.savf[i] - (st.yh[nyh + i] + st.acor[i]);
                    }
                    st.iersl = match st.wm.solve(st.h, st.el0, &mut st.y[..n]) {
                        Ok(()) => 0,
                        Err(e) => e.iersl(),
                    };
                    if st.iersl < 0 {
                        L430
                    } else if st.iersl > 0 {
                        L410
                    } else {
                        del = vmnorm(&st.y[..n], &st.ewt);
                        for i in 0..n {
                            st.acor[i] += st.y[i];
                            st.y[i] = st.yh[i] + st.el[0] * st.acor[i];
                        }
                        L400
                    }
                }
            }

            // -----------------------------------------------------------------
            // Test for convergence. If M .gt. 0, an estimate of the convergence
            // rate constant is stored in CRATE, and this is used in the test.
            //
            // We first check for a change of iterates that is the size of
            // roundoff error. If this occurs, the iteration has converged, and a
            // new rate estimate is not formed. In all other cases, force at
            // least two iterations to estimate a local Lipschitz constant
            // estimate for Adams methods. On convergence, form PDEST = local
            // maximum Lipschitz constant estimate. PDLAST is the most recent
            // nonzero estimate.
            // -----------------------------------------------------------------
            L400 => {
                if del <= 100.0 * pnorm * UROUND {
                    L450
                } else if m == 0 && st.meth == Method::Adams {
                    L405
                } else {
                    if m != 0 {
                        let mut rm = 1024.0;
                        if del <= 1024.0 * delp {
                            rm = del / delp;
                        }
                        rate = rate.max(rm);
                        st.crate_ = (0.2 * st.crate_).max(rm);
                    }
                    // 402
                    let dcon = del * 1.0_f64.min(1.5 * st.crate_) / (st.tesco[st.nq - 1][1] * st.conit);
                    if dcon > 1.0 {
                        L405
                    } else {
                        st.pdest = st.pdest.max(rate / (st.h * st.el[0]).abs());
                        if st.pdest != 0.0 {
                            st.pdlast = st.pdest;
                        }
                        L450
                    }
                }
            }
            L405 => {
                m += 1;
                if m == st.maxcor || (m >= 2 && del > 2.0 * delp) {
                    L410
                } else {
                    delp = del;
                    system.rhs(st.tn, &st.y[..n], &mut st.savf[..n])?;
                    st.nfe += 1;
                    L270
                }
            }

            // -----------------------------------------------------------------
            // The corrector iteration failed to converge. If MITER .ne. 0 and
            // the Jacobian is out of date, PJAC is called for the next try.
            // Otherwise the YH array is retracted to its values before
            // prediction, and H is reduced, if possible. If H cannot be reduced
            // or MXNCF failures have occurred, exit with KFLAG = -2.
            // -----------------------------------------------------------------
            L410 => {
                if st.miter.is_none() || st.jcur {
                    L430
                } else {
                    st.icf = 1;
                    st.ipup = st.miter.is_some();
                    L220
                }
            }
            L430 => {
                st.icf = 2;
                ncf += 1;
                st.rmax = 2.0;
                st.tn = told;
                retract(&mut st.yh, st.nqnyh, nyh, st.nq);
                if st.ierpj < 0 || st.iersl < 0 {
                    // 680
                    st.kflag = -3;
                    break;
                }
                if st.h.abs() <= st.hmin * 1.00001 || ncf == st.mxncf {
                    // 670
                    st.kflag = -2;
                    break;
                }
                rh = 0.25;
                st.ipup = st.miter.is_some();
                iredo = 1;
                L170
            }

            // -----------------------------------------------------------------
            // The corrector has converged. JCUR is set to 0 to signal that the
            // Jacobian involved may need updating later. The local error test
            // is made and control passes to statement 500 if it fails.
            // -----------------------------------------------------------------
            L450 => {
                st.jcur = false;
                if m == 0 {
                    dsm = del / st.tesco[st.nq - 1][1];
                }
                if m > 0 {
                    dsm = vmnorm(&st.acor[..n], &st.ewt) / st.tesco[st.nq - 1][1];
                }
                if dsm > 1.0 {
                    L500
                } else {
                    // ---------------------------------------------------------
                    // After a successful step, update the YH array. Decrease
                    // ICOUNT by 1, and if it is -1, consider switching methods.
                    // If a method switch is made, reset various parameters,
                    // rescale the YH array, and exit. If there is no switch,
                    // consider changing H if IALTH = 1. Otherwise decrease
                    // IALTH by 1. If IALTH is then 1 and NQ .lt. MAXORD, then
                    // ACOR is saved for use in a possible order increase on the
                    // next step. If a change in H is considered, an increase or
                    // decrease in order by one is considered also. A change in
                    // H is made only if it is by a factor of at least 1.1. If
                    // not, IALTH is set to 3 to prevent testing for that many
                    // steps.
                    // ---------------------------------------------------------
                    st.kflag = 0;
                    iredo = 0;
                    st.nst += 1;
                    st.hu = st.h;
                    st.nqu = st.nq;
                    st.mused = Some(st.meth);
                    for j in 1..=st.l {
                        let elj = st.el[j - 1];
                        let base = (j - 1) * nyh;
                        for i in 0..n {
                            st.yh[base + i] += elj * st.acor[i];
                        }
                    }
                    st.icount -= 1;
                    if st.icount >= 0 {
                        L488
                    } else if st.meth == Method::Bdf {
                        L480
                    } else {
                        L460
                    }
                }
            }

            // -----------------------------------------------------------------
            // We are currently using an Adams method. Consider switching to
            // BDF. If the current order is greater than 5, assume the problem is
            // not stiff, and skip this section. If the Lipschitz constant and
            // error estimate are not polluted by roundoff, go to 470 and
            // perform the usual test. Otherwise, switch to the BDF methods if
            // the last step was restricted to insure stability (irflag = 1),
            // and stay with Adams method if not. When switching to BDF with
            // polluted error estimates, in the absence of other information,
            // double the step size.
            //
            // When the estimates are OK, we make the usual test by computing
            // the step size we could have (ideally) used on this step, with the
            // current (Adams) method, and also that for the BDF. If
            // NQ .gt. MXORDS, we consider changing to order MXORDS on
            // switching. Compare the two step sizes to decide whether to
            // switch. The step size advantage must be at least RATIO = 5 to
            // switch.
            // -----------------------------------------------------------------
            L460 => {
                // Some((RH2, NQM2)) when the switch test passed (statement 478).
                let switch: Option<(f64, usize)> = if st.nq > 5 {
                    None
                } else if dsm > 100.0 * pnorm * UROUND && st.pdest != 0.0 {
                    // 470
                    let exsm = 1.0 / st.l as f64;
                    let mut rh1 = 1.0 / (1.2 * dsm.powf(exsm) + 0.0000012);
                    let mut rh1it = 2.0 * rh1;
                    let pdh = st.pdlast * st.h.abs();
                    if pdh * rh1 > 0.00001 {
                        rh1it = SM1[st.nq - 1] / pdh;
                    }
                    rh1 = rh1.min(rh1it);
                    let (rh2, nqm2);
                    if st.nq <= st.mxords {
                        // 474
                        let dm2 = dsm * (st.cm1[st.nq - 1] / st.cm2[st.nq - 1]);
                        rh2 = 1.0 / (1.2 * dm2.powf(exsm) + 0.0000012);
                        nqm2 = st.nq;
                    } else {
                        nqm2 = st.mxords;
                        let lm2 = st.mxords + 1;
                        let exm2 = 1.0 / lm2 as f64;
                        let lm2p1 = lm2 + 1;
                        let dm2 = vmnorm(&st.yh[col(lm2p1)], &st.ewt) / st.cm2[st.mxords - 1];
                        rh2 = 1.0 / (1.2 * dm2.powf(exm2) + 0.0000012);
                    }
                    // 476
                    if rh2 < st.ratio * rh1 { None } else { Some((rh2, nqm2)) }
                } else if !st.irflag {
                    None
                } else {
                    Some((2.0, st.nq.min(st.mxords)))
                };
                match switch {
                    None => L488,
                    Some((rh2, nqm2)) => {
                        // 478: The switch test passed. Reset relevant
                        // quantities for BDF.
                        rh = rh2;
                        st.icount = 20;
                        st.meth = Method::Bdf;
                        st.miter = Some(st.jtyp);
                        st.pdlast = 0.0;
                        st.nq = nqm2;
                        st.l = st.nq + 1;
                        L170
                    }
                }
            }

            // -----------------------------------------------------------------
            // We are currently using a BDF method. Consider switching to Adams.
            // Compute the step size we could have (ideally) used on this step,
            // with the current (BDF) method, and also that for the Adams. If
            // NQ .gt. MXORDN, we consider changing to order MXORDN on
            // switching. Compare the two step sizes to decide whether to
            // switch. The step size advantage must be at least 5/RATIO = 1 to
            // switch. If the step size for Adams would be so small as to cause
            // roundoff pollution, we stay with BDF.
            // -----------------------------------------------------------------
            L480 => {
                let exsm = 1.0 / st.l as f64;
                let (nqm1, exm1, mut dm1, mut rh1);
                if st.mxordn >= st.nq {
                    // 484
                    dm1 = dsm * (st.cm2[st.nq - 1] / st.cm1[st.nq - 1]);
                    rh1 = 1.0 / (1.2 * dm1.powf(exsm) + 0.0000012);
                    nqm1 = st.nq;
                    exm1 = exsm;
                } else {
                    nqm1 = st.mxordn;
                    let lm1 = st.mxordn + 1;
                    exm1 = 1.0 / lm1 as f64;
                    let lm1p1 = lm1 + 1;
                    dm1 = vmnorm(&st.yh[col(lm1p1)], &st.ewt) / st.cm1[st.mxordn - 1];
                    rh1 = 1.0 / (1.2 * dm1.powf(exm1) + 0.0000012);
                }
                // 486
                let mut rh1it = 2.0 * rh1;
                let pdh = st.pdnorm * st.h.abs();
                if pdh * rh1 > 0.00001 {
                    rh1it = SM1[nqm1 - 1] / pdh;
                }
                rh1 = rh1.min(rh1it);
                let rh2 = 1.0 / (1.2 * dsm.powf(exsm) + 0.0000012);
                if rh1 * st.ratio < 5.0 * rh2 {
                    L488
                } else {
                    let alpha = 0.001_f64.max(rh1);
                    dm1 *= alpha.powf(exm1); // DM1 = (ALPHA**EXM1)*DM1
                    if dm1 <= 1000.0 * UROUND * pnorm {
                        L488
                    } else {
                        // The switch test passed. Reset relevant quantities for
                        // Adams.
                        rh = rh1;
                        st.icount = 20;
                        st.meth = Method::Adams;
                        st.miter = None;
                        st.pdlast = 0.0;
                        st.nq = nqm1;
                        st.l = st.nq + 1;
                        L170
                    }
                }
            }

            // No method switch is being made. Do the usual step/order selection.
            L488 => {
                st.ialth -= 1;
                if st.ialth == 0 {
                    L520
                } else if st.ialth > 1 || st.l == st.lmax {
                    L700
                } else {
                    let c = col(st.lmax);
                    st.yh[c].copy_from_slice(&st.acor[..n]);
                    L700
                }
            }

            // -----------------------------------------------------------------
            // The error test failed. KFLAG keeps track of multiple failures.
            // Restore TN and the YH array to their previous values, and prepare
            // to try the step again. Compute the optimum step size for this or
            // one lower order. After 2 or more failures, H is forced to
            // decrease by a factor of 0.2 or less.
            // -----------------------------------------------------------------
            L500 => {
                st.kflag -= 1;
                st.tn = told;
                retract(&mut st.yh, st.nqnyh, nyh, st.nq);
                st.rmax = 2.0;
                if st.h.abs() <= st.hmin * 1.00001 {
                    // 660
                    st.kflag = -1;
                    break;
                }
                if st.kflag <= -3 {
                    L640
                } else {
                    iredo = 2;
                    rhup = 0.0;
                    L540
                }
            }

            // -----------------------------------------------------------------
            // Regardless of the success or failure of the step, factors RHDN,
            // RHSM, and RHUP are computed, by which H could be multiplied at
            // order NQ - 1, order NQ, or order NQ + 1, respectively. In the case
            // of failure, RHUP = 0.0 to avoid an order increase. The largest of
            // these is determined and the new order chosen accordingly. If the
            // order is to be increased, we compute one additional scaled
            // derivative.
            // -----------------------------------------------------------------
            L520 => {
                rhup = 0.0;
                if st.l != st.lmax {
                    let base = (st.lmax - 1) * nyh;
                    for i in 0..n {
                        st.savf[i] = st.acor[i] - st.yh[base + i];
                    }
                    let dup = vmnorm(&st.savf[..n], &st.ewt) / st.tesco[st.nq - 1][2];
                    let exup = 1.0 / (st.l + 1) as f64;
                    rhup = 1.0 / (1.4 * dup.powf(exup) + 0.0000014);
                }
                L540
            }
            L540 => {
                let exsm = 1.0 / st.l as f64;
                let mut rhsm = 1.0 / (1.2 * dsm.powf(exsm) + 0.0000012);
                let mut rhdn = 0.0_f64;
                if st.nq != 1 {
                    let ddn = vmnorm(&st.yh[col(st.l)], &st.ewt) / st.tesco[st.nq - 1][0];
                    let exdn = 1.0 / st.nq as f64;
                    rhdn = 1.0 / (1.3 * ddn.powf(exdn) + 0.0000013);
                }
                // 550: If METH = 1, limit RH according to the stability
                // region also. PDH is only read below when METH = 1.
                let mut pdh = 0.0_f64;
                if st.meth == Method::Adams {
                    pdh = (st.h.abs() * st.pdlast).max(0.000001);
                    if st.l < st.lmax {
                        rhup = rhup.min(SM1[st.l - 1] / pdh);
                    }
                    rhsm = rhsm.min(SM1[st.nq - 1] / pdh);
                    if st.nq > 1 {
                        rhdn = rhdn.min(SM1[st.nq - 2] / pdh);
                    }
                    st.pdest = 0.0;
                }
                // 560
                let pick = if rhsm >= rhup {
                    if rhsm < rhdn { Pick::L580 } else { Pick::L570 }
                } else if rhup > rhdn {
                    Pick::L590
                } else {
                    Pick::L580
                };
                if pick == Pick::L590 {
                    let newq = st.l;
                    rh = rhup;
                    if rh < 1.1 {
                        // 610
                        st.ialth = 3;
                        L700
                    } else {
                        let r = st.el[st.l - 1] / st.l as f64;
                        let base = newq * nyh; // column NEWQ + 1
                        for i in 0..n {
                            st.yh[base + i] = st.acor[i] * r;
                        }
                        // 630
                        st.nq = newq;
                        st.l = st.nq + 1;
                        iret = Iret::To170;
                        L150
                    }
                } else {
                    let newq;
                    if pick == Pick::L570 {
                        newq = st.nq;
                        rh = rhsm;
                    } else {
                        // 580
                        newq = st.nq - 1;
                        rh = rhdn;
                        if st.kflag < 0 && rh > 1.0 {
                            rh = 1.0;
                        }
                    }
                    // 620: If METH = 1 and H is restricted by stability, bypass
                    // 10 percent test.
                    let bypass = st.meth == Method::Adams && rh * pdh * 1.00001 >= SM1[newq - 1];
                    if !bypass && st.kflag == 0 && rh < 1.1 {
                        // 622 -> 610
                        st.ialth = 3;
                        L700
                    } else {
                        // 625
                        if st.kflag <= -2 {
                            rh = rh.min(0.2);
                        }
                        // If there is a change of order, reset NQ, L, and the
                        // coefficients. In any case H is reset according to RH
                        // and the YH array is rescaled. Then exit from 690 if
                        // the step was OK, or redo the step otherwise.
                        if newq == st.nq {
                            L170
                        } else {
                            // 630
                            st.nq = newq;
                            st.l = st.nq + 1;
                            iret = Iret::To170;
                            L150
                        }
                    }
                }
            }

            // -----------------------------------------------------------------
            // Control reaches this section if 3 or more failures have occured.
            // If 10 failures have occurred, exit with KFLAG = -1. It is assumed
            // that the derivatives that have accumulated in the YH array have
            // errors of the wrong order. Hence the first derivative is
            // recomputed, and the order is set to 1. Then H is reduced by a
            // factor of 10, and the step is retried, until it succeeds or H
            // reaches HMIN.
            // -----------------------------------------------------------------
            L640 => {
                if st.kflag == -10 {
                    // 660
                    st.kflag = -1;
                    break;
                }
                rh = 0.1;
                rh = (st.hmin / st.h.abs()).max(rh);
                st.h *= rh;
                st.y[..n].copy_from_slice(&st.yh[..n]);
                system.rhs(st.tn, &st.y[..n], &mut st.savf[..n])?;
                st.nfe += 1;
                for i in 0..n {
                    st.yh[nyh + i] = st.h * st.savf[i];
                }
                st.ipup = st.miter.is_some();
                st.ialth = 5;
                if st.nq == 1 {
                    L200
                } else {
                    st.nq = 1;
                    st.l = 2;
                    iret = Iret::To200;
                    L150
                }
            }

            // All returns are made through 720. H is saved in HOLD to allow the
            // caller to change H on the next step.
            L690 => {
                st.rmax = 10.0;
                L700
            }
            L700 => {
                let r = 1.0 / st.tesco[st.nqu - 1][1];
                for a in &mut st.acor[..n] {
                    *a *= r;
                }
                break;
            }
        };
    }

    // 720
    st.hold = st.h;
    st.jstart = 1;
    Ok(StepOutcome::from_kflag(st.kflag))
}

/// Predicts by multiplying the history array by the Pascal triangle matrix
/// (loops 210/215 of DSTODA, `YH1(I) = YH1(I) + YH1(I+NYH)`).
fn predict(yh: &mut [f64], nqnyh: usize, nyh: usize, nq: usize) {
    let mut i1 = nqnyh;
    for _ in 0..nq {
        i1 -= nyh;
        for i in i1..nqnyh {
            yh[i] += yh[i + nyh];
        }
    }
}

/// Retracts the history array to its values before prediction (loops
/// 440/445 and 510/515 of DSTODA, `YH1(I) = YH1(I) - YH1(I+NYH)`).
fn retract(yh: &mut [f64], nqnyh: usize, nyh: usize, nq: usize) {
    let mut i1 = nqnyh;
    for _ in 0..nq {
        i1 -= nyh;
        for i in i1..nqnyh {
            yh[i] -= yh[i + nyh];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::{StepData, intdy};
    use crate::norms::{ewset, invert_weights};
    use crate::options::{JacobianKind, Tolerance};
    use crate::system::ClosureSystem;

    /// A minimal DLSODA-style driver around the stepper: block C (initial
    /// `f` call, error weights, initial step size), then for each step the
    /// EWT refresh from `YH(:,1)`, one DSTODA call and the block F method
    /// switch bookkeeping.
    struct Driver<'a> {
        st: State,
        sys: ClosureSystem<'a>,
        tol: Tolerance,
        /// `(nst, tn, new method)` at each method switch.
        switches: Vec<(usize, f64, Method)>,
    }

    impl<'a> Driver<'a> {
        /// Block C of DLSODA. `h0 = 0.0` computes the initial step from
        /// `tout` as DLSODA does.
        fn new(
            mut sys: ClosureSystem<'a>,
            jt: JacobianKind,
            y0: &[f64],
            t0: f64,
            tout: f64,
            tol: Tolerance,
            h0: f64,
        ) -> Self {
            let n = y0.len();
            let mut st = State::new(n, jt, 12, 5);
            st.tn = t0;
            st.tsw = t0;
            sys.rhs(t0, y0, &mut st.yh[n..2 * n]).unwrap();
            st.nfe = 1;
            st.yh[..n].copy_from_slice(y0);
            st.nq = 1;
            st.h = 1.0;
            ewset(&tol, &st.yh[..n], &mut st.ewt);
            invert_weights(&mut st.ewt).unwrap();
            let mut h0 = h0;
            if h0 == 0.0 {
                let tdist = (tout - t0).abs();
                let w0 = t0.abs().max(tout.abs());
                let mut tl = tol.rtol_at(0);
                for i in 0..n {
                    tl = tl.max(tol.rtol_at(i));
                }
                if tl <= 0.0 {
                    for (i, &yi) in y0.iter().enumerate() {
                        if yi != 0.0 {
                            tl = tl.max(tol.atol_at(i) / yi.abs());
                        }
                    }
                }
                tl = tl.max(100.0 * UROUND);
                tl = tl.min(0.001);
                let sum = vmnorm(&st.yh[n..2 * n], &st.ewt);
                let sum = 1.0 / (tl * w0 * w0) + tl * sum * sum;
                h0 = 1.0 / sum.sqrt();
                h0 = h0.min(tdist);
                h0 = h0.abs().copysign(tout - t0);
            }
            st.h = h0;
            for v in &mut st.yh[n..2 * n] {
                *v *= h0;
            }
            Driver { st, sys, tol, switches: Vec::new() }
        }

        fn step(&mut self) -> StepOutcome {
            let n = self.st.n;
            ewset(&self.tol, &self.st.yh[..n], &mut self.st.ewt);
            invert_weights(&mut self.st.ewt).unwrap();
            let out = stoda(&mut self.st, &mut self.sys).unwrap();
            if let Some(m) = self.st.record_method_switch() {
                self.switches.push((self.st.nst, self.st.tn, m));
            }
            out
        }

        /// Steps until `tn` reaches `tout` (DLSODA ITASK = 1) and returns
        /// `y(tout)` interpolated from the history array.
        fn advance_to(&mut self, tout: f64) -> Vec<f64> {
            while (self.st.tn - tout) * self.st.h < 0.0 {
                assert_eq!(self.step(), StepOutcome::Success, "at tn = {}", self.st.tn);
            }
            let st = &self.st;
            let mut y = vec![0.0; st.n];
            let sd = StepData { nq: st.nq, tn: st.tn, h: st.h, hu: st.hu };
            intdy(tout, 0, &st.yh, st.nyh, st.n, &sd, &mut y).unwrap();
            y
        }
    }

    fn rober_f(y: &[f64], f: &mut [f64]) {
        // Same evaluation order as FEX in the DLSODA prologue.
        f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
        f[2] = 3.0e7 * y[1] * y[1];
        f[1] = -f[0] - f[2];
    }

    fn rober_jac(y: &[f64]) -> [[f64; 3]; 3] {
        [
            [-0.04, 1.0e4 * y[2], 1.0e4 * y[1]],
            [0.04, -1.0e4 * y[2] - 6.0e7 * y[1], -1.0e4 * y[1]],
            [0.0, 6.0e7 * y[1], 0.0],
        ]
    }

    /// The Robertson problem of the DLSODA prologue, with a Jacobian for
    /// the dense (`pd` of length 9) and the full banded `ml = mu = 2`
    /// (`pd` of length 15) layouts.
    fn robertson() -> ClosureSystem<'static> {
        ClosureSystem::new(|_t, y: &[f64], f: &mut [f64]| {
            rober_f(y, f);
            Ok(())
        })
        .with_jac(|_t, y: &[f64], pd: &mut [f64]| {
            let j = rober_jac(y);
            let banded = pd.len() == 15;
            for (r, row) in j.iter().enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    if banded {
                        pd[(r + 2 - c) + c * 5] = v;
                    } else {
                        pd[r + c * 3] = v;
                    }
                }
            }
            Ok(())
        })
    }

    fn rober_tol() -> Tolerance {
        Tolerance { rtol: crate::Tol::Scalar(1e-4), atol: crate::Tol::PerComponent(vec![1e-6, 1e-10, 1e-6]) }
    }

    const ALL_KINDS: [JacobianKind; 4] = [
        JacobianKind::DenseUser,
        JacobianKind::DenseFd,
        JacobianKind::BandedUser { ml: 2, mu: 2 },
        JacobianKind::BandedFd { ml: 2, mu: 2 },
    ];

    /// `y` printed by the DLSODA prologue example (Robertson problem,
    /// `JT = 2`) at `t = 0.4 * 10^k`, `k = 0..=11`.
    const PROLOGUE_Y: [[f64; 3]; 12] = [
        [9.851712e-01, 3.386380e-05, 1.479493e-02],
        [9.055333e-01, 2.240655e-05, 9.444430e-02],
        [7.158403e-01, 9.186334e-06, 2.841505e-01],
        [4.505250e-01, 3.222964e-06, 5.494717e-01],
        [1.831975e-01, 8.941774e-07, 8.168016e-01],
        [3.898730e-02, 1.621940e-07, 9.610125e-01],
        [4.936363e-03, 1.984221e-08, 9.950636e-01],
        [5.161831e-04, 2.065786e-09, 9.994838e-01],
        [5.179817e-05, 2.072032e-10, 9.999482e-01],
        [5.283401e-06, 2.113371e-11, 9.999947e-01],
        [4.659031e-07, 1.863613e-12, 9.999995e-01],
        [1.404280e-08, 5.617126e-14, 1.000000e+00],
    ];

    fn closure(f: impl FnMut(f64, &[f64], &mut [f64]) + 'static) -> ClosureSystem<'static> {
        let mut f = f;
        ClosureSystem::new(move |t, y: &[f64], ydot: &mut [f64]| {
            f(t, y, ydot);
            Ok(())
        })
    }

    fn decay() -> ClosureSystem<'static> {
        closure(|_t, y, f| f[0] = -y[0])
    }

    #[test]
    fn stoda_first_step_loads_coefficients_of_both_methods() {
        let mut d = Driver::new(decay(), JacobianKind::DenseFd, &[1.0], 0.0, 1.0, Tolerance::scalar(1e-6, 1e-6), 1e-4);
        assert_eq!(d.step(), StepOutcome::Success);
        let (mut elco, mut tesco) = ([[0.0; 13]; 12], [[0.0; 3]; 12]);
        cfode(Method::Bdf, &mut elco, &mut tesco);
        for q in 1..=5 {
            assert_eq!(d.st.cm2[q - 1], tesco[q - 1][1] * elco[q - 1][q], "CM2({q})");
        }
        cfode(Method::Adams, &mut elco, &mut tesco);
        for q in 1..=12 {
            assert_eq!(d.st.cm1[q - 1], tesco[q - 1][1] * elco[q - 1][q], "CM1({q})");
        }
        // The Adams coefficients are left loaded, as by the last DCFODE call.
        assert_eq!((d.st.elco, d.st.tesco), (elco, tesco));
        assert_eq!((d.st.icount, d.st.ratio, d.st.lmax), (19, 5.0, 13));
        assert_eq!((d.st.jstart, d.st.kflag, d.st.mused), (1, 0, Some(Method::Adams)));
    }

    #[test]
    fn stoda_constant_solution_follows_hand_traced_step_and_order_selection() {
        // y' = 0: every corrector converges at once (one f per step), the
        // error estimates are zero and the steps are bounded only by RMAX and
        // the Adams stability factors SM1 with PDH = 1e-6. Expected values
        // traced by hand through DSTODA: the first change of h uses
        // RMAX = 1e4 and raises the order to 2 (RHUP = SM1(2)/PDH beats
        // RHSM = SM1(1)/PDH); later RHSM = SM1(2)/PDH >= RHUP = SM1(3)/PDH
        // keeps order 2 and bypasses the 10% test (statement 620), so h grows
        // by RMAX = 10 every L = 3 steps.
        let sys = closure(|_t, _y, f| f[0] = 0.0);
        let mut d = Driver::new(sys, JacobianKind::DenseFd, &[1.0], 0.0, 1.0, Tolerance::scalar(1e-6, 1e-6), 1e-3);
        // (nq, h, hu, rmax, ialth) after each step.
        let expected: [(usize, f64, f64, f64, i32); 8] = [
            (1, 1e-3, 1e-3, 1e4, 1),
            (2, 10.0, 1e-3, 10.0, 3),
            (2, 10.0, 10.0, 10.0, 2),
            (2, 10.0, 10.0, 10.0, 1),
            (2, 100.0, 10.0, 10.0, 3),
            (2, 100.0, 100.0, 10.0, 2),
            (2, 100.0, 100.0, 10.0, 1),
            (2, 1000.0, 100.0, 10.0, 3),
        ];
        let mut nqu_prev = 1;
        for (k, &(nq, h, hu, rmax, ialth)) in expected.iter().enumerate() {
            assert_eq!(d.step(), StepOutcome::Success);
            let st = &d.st;
            assert_eq!((st.nst, st.nfe), (k + 1, k + 2), "step {}", k + 1);
            assert_eq!((st.nq, st.h, st.hu, st.rmax, st.ialth), (nq, h, hu, rmax, ialth), "step {}", k + 1);
            assert_eq!(st.nqu, nqu_prev, "step {}", k + 1);
            nqu_prev = st.nq;
        }
        assert_eq!(d.st.el[..3], [0.5, 1.0, 0.5]);
        assert_eq!(d.st.conit, 0.125);
        // No stiffness: ICOUNT keeps counting down past -1 without a switch.
        for _ in 0..20 {
            assert_eq!(d.step(), StepOutcome::Success);
        }
        assert_eq!(d.st.icount, 20 - 28);
        assert!(d.switches.is_empty());
        assert_eq!((d.st.meth, d.st.nje), (Method::Adams, 0));
    }

    #[test]
    fn stoda_error_test_failure_at_hmin_returns_kflag_minus_1() {
        // y' = -y, h = 0.01, weights 5e5: the corrector converges after two
        // f evaluations (rate 0.01) but DSM = 24.75 > 1.
        let mut d = Driver::new(decay(), JacobianKind::DenseFd, &[1.0], 0.0, 1.0, Tolerance::scalar(1e-6, 1e-6), 0.01);
        d.st.hmin = 0.01;
        assert_eq!(d.step(), StepOutcome::ErrorTestFailed);
        let st = &d.st;
        assert_eq!((st.kflag, st.nst, st.nfe, st.jstart), (-1, 0, 3, 1));
        assert_eq!((st.tn, st.hold, st.rmax), (0.0, 0.01, 2.0));
        assert!((st.yh[0] - 1.0).abs() <= 1e-15, "YH retracted: {}", st.yh[0]);
        assert_eq!(st.mused, None);
        // PDEST = RATE/|H*EL(1)| with RATE = 0.01.
        assert!((st.pdlast - 1.0).abs() < 1e-9, "{}", st.pdlast);
    }

    #[test]
    fn stoda_error_test_failure_retries_with_rhsm() {
        let mut d = Driver::new(decay(), JacobianKind::DenseFd, &[1.0], 0.0, 1.0, Tolerance::scalar(1e-6, 1e-6), 0.01);
        assert_eq!(d.step(), StepOutcome::Success);
        let st = &d.st;
        // Retried at order 1 with RH = RHSM = 1/(1.2*DSM**(1/2) + 1.2e-6).
        let rh = 1.0 / (1.2 * 24.75_f64.sqrt() + 0.0000012);
        assert!((st.hu / (0.01 * rh) - 1.0).abs() < 1e-6, "hu = {}", st.hu);
        assert_eq!((st.nst, st.nfe, st.nqu, st.tn), (1, 5, 1, st.hu));
        // RMAX = 2 after a failure; H unchanged on the successful retry.
        assert_eq!((st.rmax, st.h, st.icf), (2.0, st.hu, 0));
    }

    #[test]
    fn stoda_convergence_failure_at_hmin_returns_kflag_minus_2() {
        // y' = -y with h = 10: functional iteration diverges (rate 10).
        let mut d = Driver::new(decay(), JacobianKind::DenseFd, &[1.0], 0.0, 100.0, Tolerance::scalar(1e-6, 1e-6), 10.0);
        d.st.hmin = 10.0;
        assert_eq!(d.step(), StepOutcome::ConvergenceFailed);
        let st = &d.st;
        assert_eq!((st.kflag, st.icf, st.nst, st.nfe), (-2, 2, 0, 3));
        assert_eq!((st.tn, st.yh[0], st.yh[1]), (0.0, 1.0, -10.0));
        assert_eq!((st.rmax, st.hold, st.jstart), (2.0, 10.0, 1));
    }

    #[test]
    fn stoda_convergence_failures_stop_at_mxncf() {
        let mut d = Driver::new(decay(), JacobianKind::DenseFd, &[1.0], 0.0, 100.0, Tolerance::scalar(1e-6, 1e-6), 10.0);
        d.st.mxncf = 1;
        assert_eq!(d.step(), StepOutcome::ConvergenceFailed);
        assert_eq!((d.st.nfe, d.st.tn, d.st.h), (3, 0.0, 10.0));
    }

    #[test]
    fn stoda_convergence_failure_retries_with_quarter_step() {
        let mut d = Driver::new(decay(), JacobianKind::DenseFd, &[1.0], 0.0, 100.0, Tolerance::scalar(1e-6, 1e-6), 10.0);
        assert_eq!(d.step(), StepOutcome::Success);
        let st = &d.st;
        assert_eq!((st.icf, st.nst, st.rmax), (2, 1, 2.0));
        assert!(st.hu <= 2.5 && st.hu > 0.0, "hu = {}", st.hu);
        assert!((st.yh[0] - (-st.hu).exp()).abs() < 1e-4);
    }

    #[test]
    fn stoda_repeated_error_test_failures_restart_at_order_one() {
        // y' jumps from 0 to 1e3 at t = 1, just after t0: the first two
        // error test failures shrink h by RHSM (the second also by <= 0.2),
        // the third and fourth go through statement 640 (H*0.1, YH(:,2)
        // reloaded from f at TN, order 1, IALTH = 5) until the step stops
        // crossing t = 1.
        let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let log2 = log.clone();
        let sys = closure(move |t, _y, f| {
            log2.borrow_mut().push(t);
            f[0] = if t < 1.0 { 0.0 } else { 1.0e3 };
        });
        let t0 = 1.0 - 1e-9;
        let mut d = Driver::new(sys, JacobianKind::DenseFd, &[0.0], t0, 10.0, Tolerance::scalar(0.0, 1e-6), 1e-3);
        assert_eq!(d.step(), StepOutcome::Success);
        // Calls at TN = t0 itself come from statement 640.
        let restarts = log.borrow()[1..].iter().filter(|&&t| t == t0).count();
        assert_eq!(restarts, 2);
        // 4 failed attempts x 2 f + 2 restarts + 1 for the successful
        // attempt, plus the initial call.
        assert_eq!(d.st.nfe, 12);
        assert_eq!((d.st.nq, d.st.ialth, d.st.kflag), (1, 4, 0));
        let w = 1.0 / 1e-6;
        let rh1 = 1.0 / (1.2 * (1e-3 * 1e3 * w / 2.0_f64).sqrt() + 0.0000012);
        let h1 = 1e-3 * rh1;
        let rh2 = (1.0 / (1.2 * (h1 * 1e3 * w / 2.0).sqrt() + 0.0000012)).min(0.2);
        let h = h1 * rh2 * 0.1 * 0.1;
        assert!((d.st.hu / h - 1.0).abs() < 1e-9, "hu = {} vs {h}", d.st.hu);
        assert!(d.st.tn < 1.0);
    }

    #[test]
    fn stoda_jstart_minus_2_rescales_history_to_new_h() {
        // y' = 1: the corrector is exact, so the step taken with the new h is
        // easy to check.
        let sys = closure(|_t, _y, f| f[0] = 1.0);
        let mut d = Driver::new(sys, JacobianKind::DenseFd, &[0.0], 0.0, 10.0, Tolerance::scalar(1e-6, 1e-6), 0.5);
        assert_eq!(d.step(), StepOutcome::Success);
        assert_eq!((d.st.tn, d.st.h, d.st.yh_col(2)), (0.5, 0.5, &[0.5][..]));
        d.st.h = 0.25;
        d.st.jstart = -2;
        assert_eq!(d.step(), StepOutcome::Success);
        let st = &d.st;
        assert_eq!((st.hu, st.h, st.hold, st.tn), (0.25, 0.25, 0.25, 0.75));
        assert_eq!((st.yh_col(1), st.yh_col(2)), (&[0.75][..], &[0.25][..]));
        // IALTH was reset to L = 2 by the rescale; RMAX untouched (no 690).
        assert_eq!((st.ialth, st.rmax, st.jstart), (1, 1e4, 1));
    }

    #[test]
    fn stoda_callback_error_propagates_without_counting() {
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let c = calls.clone();
        let sys = ClosureSystem::new(move |_t, y: &[f64], f: &mut [f64]| {
            c.set(c.get() + 1);
            if c.get() == 3 {
                return Err(LsodaError::Callback { ierr: -7 });
            }
            f[0] = -y[0];
            Ok(())
        });
        let mut d = Driver::new(sys, JacobianKind::DenseFd, &[1.0], 0.0, 1.0, Tolerance::scalar(1e-6, 1e-6), 0.01);
        let n = d.st.n;
        ewset(&d.tol, &d.st.yh[..n], &mut d.st.ewt);
        invert_weights(&mut d.st.ewt).unwrap();
        let r = stoda(&mut d.st, &mut d.sys);
        assert_eq!(r, Err(LsodaError::Callback { ierr: -7 }));
        // The second corrector evaluation failed: not counted, no return
        // bookkeeping (JSTART stays 0).
        assert_eq!((calls.get(), d.st.nfe, d.st.nst, d.st.jstart), (3, 2, 0, 0));
    }

    #[test]
    fn stoda_nonstiff_problems_stay_adams() {
        // Harmonic oscillator.
        let sys = closure(|_t, y, f| {
            f[0] = y[1];
            f[1] = -y[0];
        });
        let mut d = Driver::new(sys, JacobianKind::DenseFd, &[1.0, 0.0], 0.0, 20.0, Tolerance::scalar(1e-8, 1e-8), 0.0);
        let mut max_order = 0;
        while d.st.tn < 20.0 {
            let nst = d.st.nst;
            assert_eq!(d.step(), StepOutcome::Success);
            assert_eq!(d.st.nst, nst + 1);
            assert_eq!((d.st.meth, d.st.mused, d.st.miter), (Method::Adams, Some(Method::Adams), None));
            max_order = max_order.max(d.st.nqu);
        }
        let y = d.advance_to(20.0);
        assert!((y[0] - 20f64.cos()).abs() < 1e-5 && (y[1] + 20f64.sin()).abs() < 1e-5, "{y:?}");
        assert!(d.switches.is_empty());
        assert_eq!(d.st.nje, 0);
        assert!(max_order > 5, "Adams order reached {max_order}");

        // Lotka-Volterra, as in the fork's `test_simple`.
        let sys = closure(|_t, y, f| {
            f[0] = y[0] - y[0] * y[1];
            f[1] = y[0] * y[1] - y[1];
        });
        let mut d = Driver::new(sys, JacobianKind::DenseFd, &[5.0, 0.8], 0.0, 10.0, Tolerance::scalar(1e-8, 1e-8), 0.0);
        let y = d.advance_to(10.0);
        let want = [0.38583246250193476, 4.602012234037773];
        for i in 0..2 {
            assert!((y[i] - want[i]).abs() <= 1e-5 * want[i], "{y:?} vs {want:?}");
        }
        assert!(d.switches.is_empty());
        assert_eq!((d.st.meth, d.st.nje), (Method::Adams, 0));
    }

    #[test]
    fn stoda_stiff_robertson_switches_to_bdf_step_by_step() {
        for kind in ALL_KINDS {
            let mut d = Driver::new(robertson(), kind, &[1.0, 0.0, 0.0], 0.0, 0.4, rober_tol(), 0.0);
            d.st.nslast = 7; // driver-owned: DSTODA never changes it
            // Adams steps until the switch.
            loop {
                let (nst, nfe, nq) = (d.st.nst, d.st.nfe, d.st.nq);
                assert_eq!(d.step(), StepOutcome::Success, "{kind:?}");
                assert_eq!(d.st.nst, nst + 1);
                assert!(d.st.nfe > nfe);
                assert!(d.st.nqu <= nq);
                assert_eq!(d.st.mused, Some(Method::Adams), "{kind:?}");
                assert_eq!(d.st.nje, 0, "{kind:?}: no Jacobian while Adams");
                if !d.switches.is_empty() {
                    break;
                }
                assert_eq!((d.st.meth, d.st.miter), (Method::Adams, None));
            }
            // The switch step: DSTODA set METH = 2 and MITER = JTYP after a
            // successful Adams step (MUSED = 1); block F recorded it.
            let st = &d.st;
            assert_eq!(d.switches, vec![(38, st.tn, Method::Bdf)], "{kind:?}");
            assert!((st.tn - 6.0092e-3).abs() < 5e-8, "{kind:?}: switch at {}", st.tn);
            assert_eq!((st.meth, st.miter, st.mused), (Method::Bdf, Some(kind), Some(Method::Adams)));
            assert_eq!((st.tsw, st.maxord, st.jstart, st.icount, st.pdlast), (st.tn, 5, -1, 20, 0.0));
            assert!(st.nq <= 5);

            // First BDF step: JSTART = -1 loads the BDF coefficients and
            // forces a Jacobian evaluation.
            assert_eq!(d.step(), StepOutcome::Success);
            let (mut elco, mut tesco) = (d.st.elco, d.st.tesco);
            cfode(Method::Bdf, &mut elco, &mut tesco);
            assert_eq!((d.st.elco, d.st.tesco), (elco, tesco));
            assert_eq!((d.st.mused, d.st.nje, d.st.lmax, d.st.jstart), (Some(Method::Bdf), 1, 6, 1));

            // BDF until t = 0.4: no switch back, Jacobian at least every
            // MSBP = 20 steps.
            while d.st.tn < 0.4 {
                let (nst, nje) = (d.st.nst, d.st.nje);
                assert_eq!(d.step(), StepOutcome::Success);
                assert_eq!(d.st.nst, nst + 1);
                assert!(d.st.nje >= nje);
                assert!(d.st.nst - d.st.nslp <= d.st.msbp);
                assert!(d.st.nqu <= 5 && !d.st.jcur);
                assert_eq!((d.st.mused, d.st.meth), (Some(Method::Bdf), Method::Bdf));
            }
            assert_eq!(d.switches.len(), 1);
            assert_eq!(d.st.nslast, 7);
            let y = d.advance_to(0.4);
            for i in 0..3 {
                assert!((y[i] - PROLOGUE_Y[0][i]).abs() <= 1e-6 * PROLOGUE_Y[0][i], "{kind:?}: {y:?}");
            }
        }
    }

    #[test]
    fn stoda_robertson_reproduces_dlsoda_prologue_run() {
        // The DLSODA prologue prints, for this problem with JT = 2:
        //   No. steps = 361  No. f-s = 693  No. J-s =  64
        //   Method last used = 2   Last switch was at t =  6.0092e-03
        // The step and switch sequence is reproduced exactly with every
        // Jacobian kind; the finite-difference kinds spend n = 3 extra f
        // evaluations per Jacobian (ml + mu + 1 >= n for the band).
        let atol = [1e-6, 1e-10, 1e-6];
        for kind in ALL_KINDS {
            let mut d = Driver::new(robertson(), kind, &[1.0, 0.0, 0.0], 0.0, 0.4, rober_tol(), 0.0);
            let mut tout = 0.4;
            for want in PROLOGUE_Y {
                let y = d.advance_to(tout);
                for i in 0..3 {
                    let tol = 1e-5 * want[i].abs() + 0.1 * atol[i];
                    assert!((y[i] - want[i]).abs() <= tol, "{kind:?} t = {tout:e}: {y:?} vs {want:?}");
                }
                tout *= 10.0;
            }
            let st = &d.st;
            let nfe = if kind.is_user() { 693 - 3 * 64 } else { 693 };
            assert_eq!((st.nst, st.nfe, st.nje), (361, nfe, 64), "{kind:?}");
            assert_eq!(st.mused, Some(Method::Bdf));
            assert_eq!(d.switches.len(), 1);
            assert!((st.tsw - 6.0092e-3).abs() < 5e-8, "{kind:?}: tsw = {}", st.tsw);
        }
    }

    #[test]
    fn stoda_switches_back_to_adams_when_stiffness_fades() {
        // y' = -lambda(t) (y - cos t) - sin t with y = cos t: stiff while
        // lambda = 1e3 exp(-2t) is large, nonstiff later.
        let sys = closure(|t, y, f| {
            let lambda = 1.0e3 * (-2.0 * t).exp();
            f[0] = -lambda * (y[0] - t.cos()) - t.sin();
        });
        let mut d = Driver::new(sys, JacobianKind::DenseFd, &[1.0], 0.0, 10.0, Tolerance::scalar(1e-6, 1e-8), 0.0);
        while d.st.tn < 10.0 {
            let (meth, switches) = (d.st.meth, d.switches.len());
            assert_eq!(d.step(), StepOutcome::Success);
            assert_eq!(d.st.mused, Some(meth), "MUSED is the method of the step just taken");
            if d.switches.len() > switches {
                let st = &d.st;
                assert_ne!(st.meth, meth);
                assert_eq!((st.icount, st.pdlast, st.jstart), (20, 0.0, -1));
                match st.meth {
                    Method::Bdf => assert_eq!((st.miter, st.maxord), (Some(JacobianKind::DenseFd), 5)),
                    Method::Adams => assert_eq!((st.miter, st.maxord), (None, 12)),
                }
            }
        }
        let methods: Vec<Method> = d.switches.iter().map(|s| s.2).collect();
        assert!(methods.len() >= 2, "{:?}", d.switches);
        assert_eq!(methods[..2], [Method::Bdf, Method::Adams]);
        assert!(d.switches[0].1 < 1.0 && d.switches[1].1 > 2.0, "{:?}", d.switches);
        let y = d.advance_to(10.0);
        assert!((y[0] - 10f64.cos()).abs() < 1e-4, "{y:?}");
    }
}
