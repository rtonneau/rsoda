//! The LSODA driver (`DLSODA`): the stateful [`Lsoda`] solver.
//!
//! [`Lsoda`] replaces the fork's `lsoda_class` and the common data it carries.
//! It owns the [`OdeSystem`], the validated [`Options`] and the solver
//! [`State`], and reproduces DLSODA's call protocol:
//!
//! | DLSODA | rsoda |
//! |---|---|
//! | `ITASK = 1` | [`Lsoda::integrate`] |
//! | `ITASK = 2` | [`Lsoda::step`] |
//! | `ITASK = 3` | [`Lsoda::advance_past`] |
//! | `ITASK = 4` | [`Lsoda::integrate_tcrit`] |
//! | `ITASK = 5` | [`Lsoda::step_tcrit`] |
//! | `ISTATE = 1` | the first call after [`Lsoda::new`], and every call until one has completed a step |
//! | `ISTATE = 2` | any later call |
//! | `ISTATE = 3` | the first call after a setter ([`Lsoda::set_tolerance`], ...) |
//! | `ISTATE < 0` | `Err(`[`LsodaError`]`)` |
//! | `IXPR`, `T + H = T` messages | [`Lsoda::on_warning`], [`Lsoda::on_method_switch`] |
//!
//! The port follows DLSODA block by block (A to H, named in the comments) and
//! keeps its order of floating-point operations. Nothing is ever printed:
//! the messages DLSODA writes through `XERRWD` go to opt-in hooks or into the
//! returned error.

use std::fmt;

use crate::coefficients::Method;
use crate::consts::{MXHNL0, MXSTP0, UROUND};
use crate::error::{InputError, LsodaError};
use crate::interpolate::{StepData, intdy};
use crate::norms::{ewset, invert_weights, vmnorm};
use crate::options::{JacobianKind, Options, OptionsBuilder, Tol, Tolerance};
use crate::solsy::{IterationMatrix, MatrixStorage};
use crate::state::State;
use crate::stoda::{StepOutcome, stoda};
use crate::system::OdeSystem;

/// A method switch, with the data DLSODA prints for it when `IXPR = 1`
/// (messages 105 to 107).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MethodSwitch {
    /// The method switched to (`METH`), used from the next step on.
    pub method: Method,
    /// `TN` at the switch: the end of the step after which it was decided
    /// (also [`Lsoda::tsw`] afterwards).
    pub t: f64,
    /// Tentative size of the next step (`H`).
    pub h: f64,
    /// Number of steps taken so far (`NST`).
    pub nst: usize,
}

/// A non-fatal message DLSODA would print through `XERRWD`. Delivered to the
/// hook set with [`Lsoda::on_warning`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Warning {
    /// `T + H = T` in machine arithmetic on the next step (messages 101 and
    /// 102); the solver continues anyway. Issued at most `mxhnil` times per
    /// problem: `count` (`NHNIL`) numbers the warnings, and the one with
    /// `count == mxhnil` is the last.
    NegligibleStep { t: f64, h: f64, count: usize, mxhnil: usize },
    /// A method switch, reported here only when [`Options::ixpr`] is set
    /// (DLSODA's `IXPR = 1`). [`Lsoda::on_method_switch`] receives every
    /// switch regardless of `ixpr`.
    MethodSwitch(MethodSwitch),
}

/// The `ITASK` of a call, with `TCRIT` for tasks 4 and 5.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Task {
    /// `ITASK = 1`: output at `tout` by overshooting and interpolating.
    Integrate,
    /// `ITASK = 2`: one step.
    Step,
    /// `ITASK = 3`: stop at the first mesh point at or beyond `tout`.
    AdvancePast,
    /// `ITASK = 4`: as 1 without passing `TCRIT`.
    IntegrateTcrit(f64),
    /// `ITASK = 5`: as 2 without passing `TCRIT`.
    StepTcrit(f64),
}

impl Task {
    fn tcrit(self) -> Option<f64> {
        match self {
            Self::IntegrateTcrit(tcrit) | Self::StepTcrit(tcrit) => Some(tcrit),
            _ => None,
        }
    }
}

type WarningHook<'a> = Box<dyn FnMut(&Warning) + Send + 'a>;
type SwitchHook<'a> = Box<dyn FnMut(&MethodSwitch) + Send + 'a>;

/// The LSODA solver: integrates `y' = f(t, y)` from `t0`, switching
/// automatically between the nonstiff Adams methods and the stiff BDF
/// methods (`DLSODA`).
///
/// Build the [`Options`] with [`OptionsBuilder`], create the solver with
/// [`Lsoda::new`], then call the stepping methods, which return the `t` the
/// solution [`Lsoda::y`] now corresponds to:
///
/// * [`integrate`](Lsoda::integrate) (`ITASK = 1`), [`step`](Lsoda::step)
///   (2), [`advance_past`](Lsoda::advance_past) (3),
///   [`integrate_tcrit`](Lsoda::integrate_tcrit) (4) and
///   [`step_tcrit`](Lsoda::step_tcrit) (5).
/// * The first call starts the problem (`ISTATE = 1`): it evaluates `f` at
///   `t0`, chooses the initial step size from `tout` unless `h0` was given,
///   and checks the inputs DLSODA checks then. A first call with
///   `tout == t0` does nothing. Until a call has completed a step, the next
///   call starts the problem again from [`Lsoda::t`] and [`Lsoda::y`].
/// * Later calls continue the integration (`ISTATE = 2`); the first call
///   after a setter applies the changed options (`ISTATE = 3`).
///
/// On [`MaxStepsExceeded`](LsodaError::MaxStepsExceeded),
/// [`TooMuchAccuracy`](LsodaError::TooMuchAccuracy),
/// [`ErrorTestFailed`](LsodaError::ErrorTestFailed),
/// [`ConvergenceFailed`](LsodaError::ConvergenceFailed) and
/// [`ErrorWeightZero`](LsodaError::ErrorWeightZero), [`Lsoda::t`] and
/// [`Lsoda::y`] hold the farthest mesh point reached, and the integration can
/// be continued; after `TooMuchAccuracy` and `ErrorWeightZero` the
/// tolerances must be changed first. On
/// [`IllegalInput`](LsodaError::IllegalInput), [`Lsoda::t`] and [`Lsoda::y`]
/// are unchanged (options changed by setters are taken over all the same,
/// as by DLSODA's `ISTATE = 3`). A callback error is returned as is; after
/// one in the middle of a step the solver is halted and every later call
/// fails with [`InputError::Halted`].
///
/// # Example
///
/// Integrate `y' = -y`, `y(0) = 1`, and read the solution at `t = 1`:
///
/// ```
/// use rsoda::{ClosureSystem, Lsoda, OptionsBuilder, Tolerance};
///
/// let system = ClosureSystem::new(|_t, y, f| {
///     f[0] = -y[0];
///     Ok(())
/// });
/// let options = OptionsBuilder::new(1)
///     .tolerance(Tolerance::scalar(1e-10, 1e-12))
///     .build(&system)?;
/// let mut solver = Lsoda::new(system, options, 0.0, &[1.0])?;
///
/// let t = solver.integrate(1.0)?;
/// assert_eq!(t, 1.0);
/// assert!((solver.y()[0] - (-1.0f64).exp()).abs() < 1e-8);
/// # Ok::<(), rsoda::LsodaError>(())
/// ```
pub struct Lsoda<'a, S> {
    system: S,
    opts: Options,
    st: State,
    /// `T`: where [`Lsoda::y`] was last returned.
    t: f64,
    /// `Y`: the solution at `t`.
    y: Vec<f64>,
    /// `INIT`: a step has been completed since the start of the problem.
    init: bool,
    /// A setter changed the options since the last call (next call has
    /// `ISTATE = 3`).
    changed: bool,
    /// A callback failed in the middle of a step.
    halted: bool,
    /// `NHNIL`: number of `T + H = T` warnings so far.
    nhnil: usize,
    on_warning: Option<WarningHook<'a>>,
    on_switch: Option<SwitchHook<'a>>,
}

impl<S> fmt::Debug for Lsoda<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lsoda")
            .field("t", &self.t)
            .field("y", &self.y)
            .field("tcur", &self.st.tn)
            .field("nst", &self.st.nst)
            .field("options", &self.opts)
            .finish_non_exhaustive()
    }
}

impl<'a, S: OdeSystem> Lsoda<'a, S> {
    /// Creates a solver for `system` starting at `t0` with initial values
    /// `y0`. Nothing is evaluated until the first stepping call.
    ///
    /// `options` are checked again (they may have been built by hand) and
    /// `0` values of `mxstep`, `mxhnil`, `mxordn` and `mxords` replaced by
    /// the defaults, as [`OptionsBuilder::build`] does.
    ///
    /// # Errors
    ///
    /// [`LsodaError::IllegalInput`] for invalid options or a `y0` whose
    /// length is not `options.neq`.
    pub fn new(system: S, options: Options, t0: f64, y0: &[f64]) -> Result<Self, LsodaError> {
        let opts = checked(options, &system)?;
        if y0.len() != opts.neq {
            return Err(InputError::YLength { expected: opts.neq, got: y0.len() }.into());
        }
        let mut st = State::new(opts.neq, opts.jacobian, opts.mxordn, opts.mxords);
        st.tn = t0;
        st.tsw = t0;
        Ok(Self {
            system,
            opts,
            st,
            t: t0,
            y: y0.to_vec(),
            init: false,
            changed: false,
            halted: false,
            nhnil: 0,
            on_warning: None,
            on_switch: None,
        })
    }

    // ----- Stepping (ITASK 1 to 5) ------------------------------------------

    /// Computes `y(tout)` (`ITASK = 1`): steps until `tout` is reached or
    /// passed, then interpolates. `tout` may also lie inside the last step
    /// taken (between [`Lsoda::tcur`] `-` [`Lsoda::hu`] and
    /// [`Lsoda::tcur`]); then no step is taken. Returns `tout`.
    pub fn integrate(&mut self, tout: f64) -> Result<f64, LsodaError> {
        self.solve(tout, Task::Integrate)
    }

    /// Takes one step (`ITASK = 2`) and returns the new mesh point
    /// [`Lsoda::tcur`].
    ///
    /// As in DLSODA, `tout` is only used by the call that starts the
    /// problem, to fix the direction of integration and the scale of the
    /// initial step; later calls ignore it.
    pub fn step(&mut self, tout: f64) -> Result<f64, LsodaError> {
        self.solve(tout, Task::Step)
    }

    /// Steps until the first mesh point at or beyond `tout` (`ITASK = 3`)
    /// and returns it, without interpolating. If [`Lsoda::tcur`] is already
    /// at or beyond `tout`, no step is taken.
    pub fn advance_past(&mut self, tout: f64) -> Result<f64, LsodaError> {
        self.solve(tout, Task::AdvancePast)
    }

    /// As [`Lsoda::integrate`] without ever stepping past `tcrit`
    /// (`ITASK = 4`), for problems with a singularity at or beyond `tcrit`.
    /// `tcrit` may equal or lie beyond `tout`, not behind it. If `tcrit` is
    /// reached (up to roundoff) before `tout`, returns `tcrit` exactly.
    ///
    /// The last step before `tcrit` is shortened to end just short of it;
    /// `hmin` is not enforced on that step.
    pub fn integrate_tcrit(&mut self, tout: f64, tcrit: f64) -> Result<f64, LsodaError> {
        self.solve(tout, Task::IntegrateTcrit(tcrit))
    }

    /// As [`Lsoda::step`] without stepping past `tcrit` (`ITASK = 5`).
    /// Returns `tcrit` exactly once it has been reached up to roundoff, and
    /// takes no step if it already was.
    ///
    /// `tout` is only used by the call that starts the problem, as in
    /// [`Lsoda::step`]; it must not lie beyond `tcrit`.
    pub fn step_tcrit(&mut self, tout: f64, tcrit: f64) -> Result<f64, LsodaError> {
        self.solve(tout, Task::StepTcrit(tcrit))
    }

    // ----- Re-entry with changed inputs (ISTATE = 3) ------------------------

    /// Changes the error tolerances. The next call has `ISTATE = 3`.
    ///
    /// # Errors
    ///
    /// [`LsodaError::IllegalInput`] (and no change) if a tolerance is
    /// negative or a tolerance vector has the wrong length.
    pub fn set_tolerance(&mut self, tolerance: Tolerance) -> Result<(), LsodaError> {
        self.update(|o| o.tolerance = tolerance)
    }

    /// Changes the Jacobian kind (`JT`, `ML`, `MU`). The next call has
    /// `ISTATE = 3`.
    ///
    /// As in DLSODA, the new kind is used from the next switch to BDF on. If
    /// BDF is in use, it keeps its current kind, except that a banded kind
    /// takes the new band widths if the new kind is banded too (DPRJA reads
    /// `ML` and `MU` from `IWORK` on every call).
    ///
    /// # Errors
    ///
    /// [`LsodaError::IllegalInput`] (and no change) for band widths not below
    /// `neq`, or a user-supplied kind for a system without a Jacobian.
    pub fn set_jacobian(&mut self, kind: JacobianKind) -> Result<(), LsodaError> {
        self.update(|o| o.jacobian = kind)
    }

    /// Changes the maximum `|h|`; `0.0` or `f64::MAX` removes the bound. The
    /// next call has `ISTATE = 3`. As in DLSODA, the step size in use is only
    /// bounded at its next change.
    ///
    /// # Errors
    ///
    /// [`LsodaError::IllegalInput`] (and no change) if `hmax < 0` or NaN.
    pub fn set_hmax(&mut self, hmax: f64) -> Result<(), LsodaError> {
        self.update(|o| o.hmax = hmax)
    }

    /// Changes the minimum `|h|`. The next call has `ISTATE = 3`.
    ///
    /// # Errors
    ///
    /// [`LsodaError::IllegalInput`] (and no change) if `hmin < 0` or NaN.
    pub fn set_hmin(&mut self, hmin: f64) -> Result<(), LsodaError> {
        self.update(|o| o.hmin = hmin)
    }

    /// Changes the maximum number of steps per call; `0` selects the default
    /// (10000). The next call has `ISTATE = 3`.
    pub fn set_mxstep(&mut self, mxstep: usize) {
        self.opts.mxstep = if mxstep == 0 { MXSTP0 } else { mxstep };
        self.mark_changed();
    }

    /// Changes the maximum number of `T + H = T` warnings; `0` selects the
    /// default (10). The next call has `ISTATE = 3`.
    pub fn set_mxhnil(&mut self, mxhnil: usize) {
        self.opts.mxhnil = if mxhnil == 0 { MXHNL0 } else { mxhnil };
        self.mark_changed();
    }

    /// Changes whether method switches are also reported as
    /// [`Warning::MethodSwitch`] (`IXPR`). The next call has `ISTATE = 3`.
    pub fn set_ixpr(&mut self, ixpr: bool) {
        self.opts.ixpr = ixpr;
        self.mark_changed();
    }

    // ----- Hooks ------------------------------------------------------------

    /// Sets the hook receiving the [`Warning`]s DLSODA would print. Without
    /// one, warnings are dropped.
    pub fn on_warning(&mut self, hook: impl FnMut(&Warning) + Send + 'a) -> &mut Self {
        self.on_warning = Some(Box::new(hook));
        self
    }

    /// Sets the hook called after every method switch, regardless of `ixpr`.
    pub fn on_method_switch(&mut self, hook: impl FnMut(&MethodSwitch) + Send + 'a) -> &mut Self {
        self.on_switch = Some(Box::new(hook));
        self
    }

    // ----- Outputs ----------------------------------------------------------

    /// `T`: the value of `t` that [`Lsoda::y`] corresponds to (the last
    /// value returned, or `t0`).
    pub fn t(&self) -> f64 {
        self.t
    }

    /// `Y`: the solution at [`Lsoda::t`].
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// `HU` (`RWORK(11)`): the step size last used successfully.
    pub fn hu(&self) -> f64 {
        self.st.hu
    }

    /// `HCUR` (`RWORK(12)`): the step size to be attempted next.
    pub fn hcur(&self) -> f64 {
        self.st.h
    }

    /// `TCUR` (`RWORK(13)`): the mesh point the solver has reached; at
    /// least as far as [`Lsoda::t`].
    pub fn tcur(&self) -> f64 {
        self.st.tn
    }

    /// `TSW` (`RWORK(15)`): `t` at the last method switch (`t0` if none).
    pub fn tsw(&self) -> f64 {
        self.st.tsw
    }

    /// `NST` (`IWORK(11)`): number of steps taken.
    pub fn nst(&self) -> usize {
        self.st.nst
    }

    /// `NFE` (`IWORK(12)`): number of `f` evaluations.
    pub fn nfe(&self) -> usize {
        self.st.nfe
    }

    /// `NJE` (`IWORK(13)`): number of Jacobian evaluations (and LU
    /// factorizations).
    pub fn nje(&self) -> usize {
        self.st.nje
    }

    /// `NQU` (`IWORK(14)`): the order last used successfully (0 before the
    /// first step).
    pub fn nqu(&self) -> usize {
        self.st.nqu
    }

    /// `NQCUR` (`IWORK(15)`): the order to be attempted next.
    pub fn nqcur(&self) -> usize {
        self.st.nq
    }

    /// `MUSED` (`IWORK(19)`): the method of the last successful step.
    pub fn mused(&self) -> Option<Method> {
        self.st.mused
    }

    /// `MCUR` (`IWORK(20)`): the method to be attempted next. It differs
    /// from [`Lsoda::mused`] only right after a method switch.
    pub fn mcur(&self) -> Method {
        self.st.meth
    }

    /// The options in effect.
    pub fn options(&self) -> &Options {
        &self.opts
    }

    /// The ODE system.
    pub fn system(&self) -> &S {
        &self.system
    }

    /// The ODE system, mutably (e.g. to change parameters between calls).
    pub fn system_mut(&mut self) -> &mut S {
        &mut self.system
    }

    /// Consumes the solver and returns the ODE system.
    pub fn into_system(self) -> S {
        self.system
    }

    // ----- DLSODA -----------------------------------------------------------

    /// One call to DLSODA with `TOUT = tout` and the `ITASK` of `task`.
    fn solve(&mut self, tout: f64, task: Task) -> Result<f64, LsodaError> {
        let istate = if !self.init {
            // Block A, ISTATE = 1: a call with TOUT = T does nothing.
            if tout == self.t {
                return Ok(self.t);
            }
            // Block B: with optional inputs (IOPT = 1), TOUT must lie ahead
            // of T in the direction of H0.
            if (tout - self.t) * self.opts.h0 < 0.0 {
                let e = InputError::H0Direction { t: self.t, tout, h0: self.opts.h0 };
                return Err(e.into());
            }
            self.start(tout, task)?;
            1
        } else {
            if self.halted {
                return Err(InputError::Halted.into());
            }
            let istate = if self.changed {
                self.apply_changes();
                3
            } else {
                2
            };
            if let Some(t) = self.check_stop(tout, task, istate)? {
                return Ok(t);
            }
            istate
        };
        self.run(tout, task, istate == 1)
    }

    /// Block C: the initializations of the first call, the initial call to
    /// `f` and the initial step size. The state is rebuilt from scratch.
    fn start(&mut self, tout: f64, task: Task) -> Result<(), LsodaError> {
        let o = &self.opts;
        let n = o.neq;
        let t = self.t;
        // JSTART = 0, NST = NJE = NSLAST = 0, HU = 0, NQU = 0, MUSED = 0,
        // METH = 1, MITER = 0, MAXORD = MXORDN and the CCMAX, MAXCOR, MSBP,
        // MXNCF constants are set by State::new.
        let mut st = State::new(n, o.jacobian, o.mxordn, o.mxords);
        st.hmxi = hmxi(o.hmax);
        st.hmin = o.hmin;
        st.tn = t;
        st.tsw = t;
        let mut h0 = o.h0;
        if let Some(tcrit) = task.tcrit() {
            if (tcrit - tout) * (tout - t) < 0.0 {
                return Err(InputError::TcritBehindTout { tcrit, tout }.into());
            }
            if h0 != 0.0 && (t + h0 - tcrit) * h0 > 0.0 {
                h0 = tcrit - t;
            }
        }
        self.nhnil = 0;
        // Initial call to F into YH(:,2).
        let f0 = st.nyh..st.nyh + n;
        self.system.rhs(t, &self.y, &mut st.yh[f0.clone()])?;
        st.nfe = 1;
        // Load the initial value vector in YH.
        st.yh[..n].copy_from_slice(&self.y);
        // Load and invert the EWT array (H is temporarily set to 1.0).
        st.nq = 1;
        st.h = 1.0;
        ewset(&o.tolerance, &st.yh[..n], &mut st.ewt);
        if let Err(index) = invert_weights(&mut st.ewt[..n]) {
            let e = InputError::ErrorWeightNonPositive { index, value: st.ewt[index] };
            return Err(e.into());
        }
        // Unless given, H0 follows from
        //   H0**(-2) = 1/(TOL * W0**2) + TOL * norm(F)**2,
        // with W0 = max(|T|, |TOUT|), F = f(t0, y0) and TOL = max RTOL(i) if
        // positive, else max ATOL(i)/|Y(i)|, kept in [100*UROUND, 1e-3]. The
        // sign of H0 is that of TOUT - T, and |H0| <= |TOUT - T|.
        if h0 == 0.0 {
            let tdist = (tout - t).abs();
            let w0 = t.abs().max(tout.abs());
            if tdist < 2.0 * UROUND * w0 {
                return Err(InputError::TooutTooClose { t, tout }.into());
            }
            let mut tol = o.tolerance.rtol_at(0);
            if let Tol::PerComponent(_) = o.tolerance.rtol {
                for i in 0..n {
                    tol = tol.max(o.tolerance.rtol_at(i));
                }
            }
            if tol <= 0.0 {
                for (i, yi) in self.y.iter().enumerate() {
                    let ayi = yi.abs();
                    if ayi != 0.0 {
                        tol = tol.max(o.tolerance.atol_at(i) / ayi);
                    }
                }
            }
            tol = tol.max(100.0 * UROUND);
            tol = tol.min(0.001);
            let sum = vmnorm(&st.yh[f0.clone()], &st.ewt);
            let sum = 1.0 / (tol * w0 * w0) + tol * (sum * sum);
            h0 = 1.0 / sum.sqrt();
            h0 = h0.min(tdist);
            h0 = h0.abs().copysign(tout - t);
        }
        // Adjust H0 if necessary to meet the HMAX bound.
        let rh = h0.abs() * st.hmxi;
        if rh > 1.0 {
            h0 /= rh;
        }
        // Load H with H0 and scale YH(:,2) by H0.
        st.h = h0;
        for v in &mut st.yh[f0] {
            *v *= h0;
        }
        self.st = st;
        Ok(())
    }

    /// Block B on a continuation call with `ISTATE = 3`: takes over the
    /// options changed by the setters (already checked) and sets
    /// `JSTART = -1` to signal parameter changes to DSTODA.
    fn apply_changes(&mut self) {
        let st = &mut self.st;
        st.hmxi = hmxi(self.opts.hmax);
        st.hmin = self.opts.hmin;
        let jt = self.opts.jacobian;
        st.jtyp = jt;
        st.miter = st.miter.map(|m| with_band_of(m, jt));
        st.jstart = -1;
        self.changed = false;
    }

    /// Block D: the stop conditions of a continuation call, checked before
    /// any step. Returns the output `t` if the call is complete without a
    /// step.
    fn check_stop(&mut self, tout: f64, task: Task, istate: u8) -> Result<Option<f64>, LsodaError> {
        let st = &mut self.st;
        st.nslast = st.nst;
        let (tn, h) = (st.tn, st.h);
        match task {
            // 210
            Task::Integrate => {
                if (tn - tout) * h < 0.0 {
                    return Ok(None);
                }
                self.interpolate(tout).map(Some)
            }
            Task::Step => Ok(None),
            // 220
            Task::AdvancePast => {
                let tp = tn - st.hu * (1.0 + 100.0 * UROUND);
                if (tp - tout) * h > 0.0 {
                    return Err(InputError::TooutBehindLastStep { tout, tp }.into());
                }
                if (tn - tout) * h < 0.0 {
                    return Ok(None);
                }
                Ok(Some(self.mesh_output(tn)))
            }
            // 230
            Task::IntegrateTcrit(tcrit) => {
                if (tn - tcrit) * h > 0.0 {
                    return Err(InputError::TcritBehindTcur { tcrit, tcur: tn }.into());
                }
                if (tcrit - tout) * h < 0.0 {
                    return Err(InputError::TcritBehindTout { tcrit, tout }.into());
                }
                if (tn - tout) * h < 0.0 {
                    return Ok(self.near_tcrit(tcrit, istate == 2));
                }
                self.interpolate(tout).map(Some)
            }
            // 240
            Task::StepTcrit(tcrit) => {
                if (tn - tcrit) * h > 0.0 {
                    return Err(InputError::TcritBehindTcur { tcrit, tcur: tn }.into());
                }
                Ok(self.near_tcrit(tcrit, istate == 2))
            }
        }
    }

    /// Statement 245: returns `TCRIT` as the output if it has been reached;
    /// otherwise limits the next step to end before it (with `JSTART = -2`
    /// only on an `ISTATE = 2` call: `ISTATE = 3` has set `JSTART = -1`).
    fn near_tcrit(&mut self, tcrit: f64, rescale: bool) -> Option<f64> {
        if self.approach_tcrit(tcrit, rescale) { Some(self.mesh_output(tcrit)) } else { None }
    }

    /// Blocks E and F: the step loop. Each pass checks `mxstep`, refreshes
    /// the error weights, checks for too much accuracy and `T + H = T`, takes
    /// a step and then tests the stop condition of `task`. The first pass of
    /// the call that starts the problem enters at statement 270, as the
    /// weights are fresh from block C.
    fn run(&mut self, tout: f64, task: Task, mut at_start: bool) -> Result<f64, LsodaError> {
        let n = self.st.n;
        loop {
            if !at_start {
                // 250
                if self.st.nst - self.st.nslast >= self.opts.mxstep {
                    let e = LsodaError::MaxStepsExceeded { t: self.st.tn, mxstep: self.opts.mxstep };
                    return Err(self.interrupt(e));
                }
                ewset(&self.opts.tolerance, &self.st.yh[..n], &mut self.st.ewt);
                if let Err(component) = invert_weights(&mut self.st.ewt[..n]) {
                    let e = LsodaError::ErrorWeightZero { t: self.st.tn, component };
                    return Err(self.interrupt(e));
                }
            }
            at_start = false;
            // 270
            let tolsf = UROUND * vmnorm(&self.st.yh[..n], &self.st.ewt);
            if tolsf > 1.0 {
                let tolsf = tolsf * 2.0;
                if self.st.nst == 0 {
                    return Err(InputError::TooMuchAccuracyAtStart { tolsf }.into());
                }
                let e = LsodaError::TooMuchAccuracy { t: self.st.tn, tolsf };
                return Err(self.interrupt(e));
            }
            // 280
            if self.st.tn + self.st.h == self.st.tn {
                self.warn_negligible_step();
            }
            // 290
            self.match_matrix_storage();
            let outcome = match stoda(&mut self.st, &mut self.system) {
                Ok(outcome) => outcome,
                Err(e) => {
                    // DSTODA stopped mid-step. Before the first completed
                    // step, the next call starts the problem afresh.
                    self.halted = self.init;
                    return Err(e);
                }
            };
            match outcome {
                StepOutcome::Success => {}
                // 530
                StepOutcome::ErrorTestFailed => {
                    let e = LsodaError::ErrorTestFailed { t: self.st.tn, h: self.st.h, imxer: self.imxer() };
                    return Err(self.interrupt(e));
                }
                // 540. KFLAG = -3 cannot occur in LSODA (neither DPRJA nor
                // DSOLSY reports a fatal error); it comes from the same
                // corrector failure path.
                StepOutcome::ConvergenceFailed | StepOutcome::SolverFailed => {
                    let e = LsodaError::ConvergenceFailed { t: self.st.tn, h: self.st.h, imxer: self.imxer() };
                    return Err(self.interrupt(e));
                }
            }
            // 300
            self.init = true;
            if let Some(method) = self.st.record_method_switch() {
                self.report_switch(method);
            }
            // 310
            let (tn, h) = (self.st.tn, self.st.h);
            match task {
                // 320
                Task::Integrate => {
                    if (tn - tout) * h >= 0.0 {
                        return self.interpolate(tout);
                    }
                }
                Task::Step => return Ok(self.mesh_output(tn)),
                // 330
                Task::AdvancePast => {
                    if (tn - tout) * h >= 0.0 {
                        return Ok(self.mesh_output(tn));
                    }
                }
                // 340, 345
                Task::IntegrateTcrit(tcrit) => {
                    if (tn - tout) * h >= 0.0 {
                        return self.interpolate(tout);
                    }
                    if self.approach_tcrit(tcrit, true) {
                        return Ok(self.mesh_output(tcrit));
                    }
                }
                // 350
                Task::StepTcrit(tcrit) => {
                    let t = if reaches(&self.st, tcrit) { tcrit } else { tn };
                    return Ok(self.mesh_output(t));
                }
            }
        }
    }

    /// Statements 245 and 345: `IHIT` (`TN` equals `TCRIT` up to roundoff).
    /// If not, and the next step would pass `TCRIT`, shortens `H` to end just
    /// short of it and, if `rescale`, sets `JSTART = -2` (unless a pending
    /// `JSTART = -1` covers it) so that DSTODA rescales `YH`.
    fn approach_tcrit(&mut self, tcrit: f64, rescale: bool) -> bool {
        let st = &mut self.st;
        if reaches(st, tcrit) {
            return true;
        }
        let tnext = st.tn + st.h * (1.0 + 4.0 * UROUND);
        if (tnext - tcrit) * st.h <= 0.0 {
            return false;
        }
        st.h = (tcrit - st.tn) * (1.0 - 4.0 * UROUND);
        if rescale && st.jstart >= 0 {
            st.jstart = -2;
        }
        false
    }

    /// Statements 280 to 290: counts a `T + H = T` step and reports it while
    /// the count does not exceed `mxhnil`.
    fn warn_negligible_step(&mut self) {
        self.nhnil += 1;
        if self.nhnil > self.opts.mxhnil {
            return;
        }
        let w = Warning::NegligibleStep {
            t: self.st.tn,
            h: self.st.h,
            count: self.nhnil,
            mxhnil: self.opts.mxhnil,
        };
        if let Some(hook) = self.on_warning.as_mut() {
            hook(&w);
        }
    }

    /// Block F: reports a method switch to the hooks (to the warning hook
    /// only with `IXPR`).
    fn report_switch(&mut self, method: Method) {
        let event = MethodSwitch { method, t: self.st.tn, h: self.st.h, nst: self.st.nst };
        if let Some(hook) = self.on_switch.as_mut() {
            hook(&event);
        }
        if self.opts.ixpr
            && let Some(hook) = self.on_warning.as_mut()
        {
            hook(&Warning::MethodSwitch(event));
        }
    }

    /// Reallocates the iteration matrix when the Jacobian kind of the BDF
    /// method (`MITER`) needs another storage. This only happens after
    /// [`Lsoda::set_jacobian`], and the matrix is then formed anew before it
    /// is used (`JSTART = -1` forces `IPUP`).
    fn match_matrix_storage(&mut self) {
        let st = &mut self.st;
        if let Some(kind) = st.miter {
            let storage = MatrixStorage::for_jacobian(&kind);
            if st.wm.storage != storage {
                st.wm = IterationMatrix::new(st.n, storage);
            }
        }
    }

    /// Statements 560 to 570: `IMXER`, the 0-based component of largest
    /// weighted local error `|ACOR(i) * EWT(i)|` (first one on ties).
    fn imxer(&self) -> usize {
        let st = &self.st;
        let mut big = 0.0_f64;
        let mut imxer = 0;
        for i in 0..st.n {
            let size = (st.acor[i] * st.ewt[i]).abs();
            if big >= size {
                continue;
            }
            big = size;
            imxer = i;
        }
        imxer
    }

    /// Interpolates `y(tout)` from the history array into the output (DINTDY
    /// with `K = 0`).
    fn interpolate(&mut self, tout: f64) -> Result<f64, LsodaError> {
        let st = &self.st;
        let step = StepData { nq: st.nq, tn: st.tn, h: st.h, hu: st.hu };
        intdy(tout, 0, &st.yh, st.nyh, st.n, &step, &mut self.y).map_err(InputError::Interpolation)?;
        self.t = tout;
        Ok(tout)
    }

    /// Block G (statement 400): outputs `YH(:,1)` with `T = t`.
    fn mesh_output(&mut self, t: f64) -> f64 {
        let n = self.st.n;
        self.y.copy_from_slice(&self.st.yh[..n]);
        self.t = t;
        t
    }

    /// Block H (statement 580): an interrupted call outputs the last mesh
    /// point reached.
    fn interrupt(&mut self, e: LsodaError) -> LsodaError {
        self.mesh_output(self.st.tn);
        e
    }

    /// Applies a checked change of the options.
    fn update(&mut self, change: impl FnOnce(&mut Options)) -> Result<(), LsodaError> {
        let mut o = self.opts.clone();
        change(&mut o);
        self.opts = checked(o, &self.system)?;
        self.mark_changed();
        Ok(())
    }

    /// Once the problem has started, the next call must take over changed
    /// options (`ISTATE = 3`); before, the starting call reads them all.
    fn mark_changed(&mut self) {
        if self.init {
            self.changed = true;
        }
    }
}

/// `IHIT`: `TN` equals `TCRIT` up to roundoff (statements 245, 345, 350).
fn reaches(st: &State, tcrit: f64) -> bool {
    let hmx = st.tn.abs() + st.h.abs();
    (st.tn - tcrit).abs() <= 100.0 * UROUND * hmx
}

/// `HMXI = 1/HMAX`, with `0` meaning no bound for `HMAX = 0` (DLSODA) and
/// for the builder's "unbounded" default `f64::MAX` (where the fork's
/// `1/HUGE` never binds either).
fn hmxi(hmax: f64) -> f64 {
    if hmax > 0.0 && hmax < f64::MAX { 1.0 / hmax } else { 0.0 }
}

/// `MITER` after `JT` changed to `jt` on an `ISTATE = 3` call: DLSODA keeps
/// `MITER` until the next switch to BDF, but DPRJA reads `ML` and `MU` from
/// `IWORK` on every call, so a banded `MITER` takes the band of a banded
/// `jt`.
fn with_band_of(miter: JacobianKind, jt: JacobianKind) -> JacobianKind {
    let (JacobianKind::BandedUser { ml, mu } | JacobianKind::BandedFd { ml, mu }) = jt else {
        return miter;
    };
    match miter {
        JacobianKind::BandedUser { .. } => JacobianKind::BandedUser { ml, mu },
        JacobianKind::BandedFd { .. } => JacobianKind::BandedFd { ml, mu },
        dense => dense,
    }
}

/// Checks (and normalizes) options that may not have come from
/// [`OptionsBuilder::build`], with the builder's own checks.
fn checked<S: OdeSystem + ?Sized>(o: Options, system: &S) -> Result<Options, LsodaError> {
    OptionsBuilder::new(o.neq)
        .tolerance(o.tolerance)
        .jacobian(o.jacobian)
        .h0(o.h0)
        .hmax(o.hmax)
        .hmin(o.hmin)
        .ixpr(o.ixpr)
        .mxstep(o.mxstep)
        .mxhnil(o.mxhnil)
        .mxordn(o.mxordn)
        .mxords(o.mxords)
        .build(system)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::IntdyError;
    use crate::system::ClosureSystem;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Mutex;

    /// A solver for `sys` from `(t0, y0)`, with options set by `b`.
    fn lsoda<'h, 'a>(
        sys: ClosureSystem<'a>,
        t0: f64,
        y0: &[f64],
        b: impl FnOnce(OptionsBuilder) -> OptionsBuilder,
    ) -> Lsoda<'h, ClosureSystem<'a>> {
        let opts = b(OptionsBuilder::new(y0.len())).build(&sys).unwrap();
        Lsoda::new(sys, opts, t0, y0).unwrap()
    }

    fn illegal<T: fmt::Debug>(r: Result<T, LsodaError>) -> InputError {
        match r {
            Err(LsodaError::IllegalInput(e)) => e,
            other => panic!("expected IllegalInput, got {other:?}"),
        }
    }

    fn closure(f: impl FnMut(f64, &[f64], &mut [f64]) + 'static) -> ClosureSystem<'static> {
        let mut f = f;
        ClosureSystem::new(move |t, y: &[f64], ydot: &mut [f64]| {
            f(t, y, ydot);
            Ok(())
        })
    }

    /// `y' = -y`, with its Jacobian (`pd[0]` is the only entry for both the
    /// dense and the `ml = mu = 0` banded layout).
    fn decay() -> ClosureSystem<'static> {
        closure(|_t, y, f| f[0] = -y[0]).with_jac(|_t, _y: &[f64], pd: &mut [f64]| {
            pd[0] = -1.0;
            Ok(())
        })
    }

    const DECAY_KINDS: [JacobianKind; 4] = [
        JacobianKind::DenseUser,
        JacobianKind::DenseFd,
        JacobianKind::BandedUser { ml: 0, mu: 0 },
        JacobianKind::BandedFd { ml: 0, mu: 0 },
    ];

    fn tight() -> Tolerance {
        Tolerance::scalar(1e-6, 1e-6)
    }

    /// The Robertson problem of the DLSODA prologue, with a Jacobian for the
    /// dense (`pd` of length 9) and the full banded `ml = mu = 2` (`pd` of
    /// length 15) layouts.
    fn robertson() -> ClosureSystem<'static> {
        closure(|_t, y, f| {
            // Same evaluation order as FEX in the DLSODA prologue.
            f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
            f[2] = 3.0e7 * y[1] * y[1];
            f[1] = -f[0] - f[2];
        })
        .with_jac(|_t, y: &[f64], pd: &mut [f64]| {
            let j = [
                [-0.04, 1.0e4 * y[2], 1.0e4 * y[1]],
                [0.04, -1.0e4 * y[2] - 6.0e7 * y[1], -1.0e4 * y[1]],
                [0.0, 6.0e7 * y[1], 0.0],
            ];
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

    const ROBER_ATOL: [f64; 3] = [1e-6, 1e-10, 1e-6];

    fn rober_tol() -> Tolerance {
        Tolerance { rtol: Tol::Scalar(1e-4), atol: Tol::PerComponent(ROBER_ATOL.to_vec()) }
    }

    fn rober(kind: JacobianKind) -> Lsoda<'static, ClosureSystem<'static>> {
        lsoda(robertson(), 0.0, &[1.0, 0.0, 0.0], |b| b.jacobian(kind).tolerance(rober_tol()))
    }

    const ROBER_KINDS: [JacobianKind; 4] = [
        JacobianKind::DenseUser,
        JacobianKind::DenseFd,
        JacobianKind::BandedUser { ml: 2, mu: 2 },
        JacobianKind::BandedFd { ml: 2, mu: 2 },
    ];

    /// `y` printed by the DLSODA prologue example at `t = 0.4 * 10^k`.
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

    /// `y` matches the prologue output to the printed digits (the run takes
    /// the prologue's steps).
    fn assert_prologue_y(y: &[f64], k: usize, what: &str) {
        let want = PROLOGUE_Y[k];
        for i in 0..3 {
            let tol = 1e-5 * want[i].abs() + 0.1 * ROBER_ATOL[i];
            assert!((y[i] - want[i]).abs() <= tol, "{what} k = {k}: {y:?} vs {want:?}");
        }
    }

    /// `y(tout)` of the Robertson problem is within the requested tolerances
    /// of a solution computed with tight ones. For runs that leave the
    /// prologue's step sequence (e.g. after an `ISTATE = 3` call), whose
    /// global error differs from the prologue's.
    fn assert_rober_accurate(y: &[f64], tout: f64, what: &str) {
        let mut r = lsoda(robertson(), 0.0, &[1.0, 0.0, 0.0], |b| {
            b.jacobian(JacobianKind::DenseUser).tolerance(Tolerance::scalar(1e-11, 1e-16))
        });
        r.integrate(tout).unwrap();
        let want = r.y();
        for i in 0..3 {
            let tol = 1e-4 * want[i].abs() + ROBER_ATOL[i];
            assert!((y[i] - want[i]).abs() <= tol, "{what} t = {tout}: {y:?} vs {want:?}");
        }
    }

    // ----- Smoke test and DLSODA prologue ------------------------------------

    #[test]
    fn lsoda_smoke_decay_over_unit_interval_all_jacobian_kinds() {
        let exact = (-1.0_f64).exp();
        let mut runs = Vec::new();
        for kind in DECAY_KINDS {
            let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.jacobian(kind).tolerance(Tolerance::scalar(1e-10, 1e-12)));
            assert_eq!(s.integrate(1.0), Ok(1.0), "{kind:?}");
            assert_eq!(s.t(), 1.0);
            assert!((s.y()[0] - exact).abs() < 1e-8, "{kind:?}: {} vs {exact}", s.y()[0]);
            assert!(s.tcur() >= 1.0 && s.tcur() - s.hu() < 1.0, "{kind:?}: tcur = {}", s.tcur());
            assert!(s.nst() > 0 && s.nfe() > s.nst() && s.nqu() > 0);
            // Not stiff: Adams all the way, so no Jacobian is ever formed.
            assert_eq!((s.mused(), s.mcur(), s.nje(), s.tsw()), (Some(Method::Adams), Method::Adams, 0, 0.0));
            runs.push((s.nst(), s.nfe(), s.y()[0]));
        }
        assert!(runs.iter().all(|r| *r == runs[0]), "{runs:?}");
    }

    #[test]
    fn lsoda_robertson_reproduces_dlsoda_prologue_run() {
        // The DLSODA prologue run (ITASK = 1, TOUT = 0.4 * 10^k) prints
        //   No. steps = 361  No. f-s = 693  No. J-s =  64
        //   Method last used = 2   Last switch was at t =  6.0092e-03
        // The user-supplied kinds save the n = 3 f evaluations per Jacobian.
        for kind in ROBER_KINDS {
            let switches = Mutex::new(Vec::new());
            let mut s = rober(kind);
            s.on_method_switch(|e| switches.lock().unwrap().push(*e));
            let mut tout = 0.4;
            for k in 0..12 {
                assert_eq!(s.integrate(tout), Ok(tout), "{kind:?}");
                assert_prologue_y(s.y(), k, &format!("{kind:?}"));
                tout *= 10.0;
            }
            let nfe = if kind.is_user() { 693 - 3 * 64 } else { 693 };
            assert_eq!((s.nst(), s.nfe(), s.nje()), (361, nfe, 64), "{kind:?}");
            assert_eq!((s.mused(), s.mcur()), (Some(Method::Bdf), Method::Bdf));
            assert!((s.tsw() - 6.0092e-3).abs() < 5e-8, "{kind:?}: tsw = {}", s.tsw());
            let tsw = s.tsw();
            drop(s);
            let switches = switches.into_inner().unwrap();
            assert_eq!(switches.len(), 1, "{kind:?}");
            let sw = switches[0];
            assert_eq!((sw.method, sw.nst, sw.t), (Method::Bdf, 38, tsw));
            assert!(sw.h > 0.0, "{sw:?}");
        }
    }

    // ----- ITASK = 1 to 5 ----------------------------------------------------

    #[test]
    fn lsoda_first_call_with_tout_equal_t_does_nothing() {
        let mut s = lsoda(decay(), 0.5, &[1.0], |b| b);
        assert_eq!(s.integrate(0.5), Ok(0.5));
        assert_eq!((s.nfe(), s.nst(), s.y()), (0, 0, &[1.0][..]));
        // The next call is still the first call of the problem.
        assert_eq!(s.integrate(1.0), Ok(1.0));
        assert!((s.y()[0] - (-0.5_f64).exp()).abs() < 1e-5);
    }

    #[test]
    fn lsoda_integrate_continuation_matches_one_long_call() {
        // ISTATE = 2: TOUT = 0.5 then 1.0 takes exactly the steps of one call
        // to 1.0. (H0 is given: otherwise it depends on the first TOUT.)
        let mut a = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()).h0(1e-4));
        assert_eq!(a.integrate(0.5), Ok(0.5));
        assert!((a.y()[0] - (-0.5_f64).exp()).abs() < 1e-5);
        assert!(a.tcur() >= 0.5);
        assert_eq!(a.integrate(1.0), Ok(1.0));
        let mut b = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()).h0(1e-4));
        assert_eq!(b.integrate(1.0), Ok(1.0));
        assert_eq!((a.nst(), a.nfe(), a.tcur(), a.y()), (b.nst(), b.nfe(), b.tcur(), b.y()));

        // A TOUT inside the last step interpolates without stepping; one
        // behind it is "trouble in DINTDY".
        let (nst, t_back) = (b.nst(), b.tcur() - 0.5 * b.hu());
        assert_eq!(b.integrate(t_back), Ok(t_back));
        assert_eq!(b.nst(), nst);
        assert!((b.y()[0] - (-t_back).exp()).abs() < 1e-5);
        let e = illegal(b.integrate(0.0));
        assert!(matches!(e, InputError::Interpolation(IntdyError::IllegalT { t: 0.0, .. })), "{e:?}");
        assert_eq!((b.t(), b.nst()), (t_back, nst));
    }

    #[test]
    fn lsoda_integrates_backwards() {
        let mut s = lsoda(decay(), 1.0, &[1.0], |b| b.tolerance(tight()));
        assert_eq!(s.integrate(0.0), Ok(0.0));
        assert!(s.hu() < 0.0 && s.tcur() <= 0.0);
        let exact = 1.0_f64.exp();
        assert!((s.y()[0] - exact).abs() < 1e-5 * exact, "{}", s.y()[0]);
    }

    #[test]
    fn lsoda_step_takes_one_step_per_call() {
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()));
        let mut t_prev = 0.0;
        while s.t() < 1.0 {
            let nst = s.nst();
            // TOUT only matters on the first call.
            let t = s.step(if nst == 0 { 1.0 } else { -5.0 }).unwrap();
            assert_eq!(s.nst(), nst + 1);
            assert_eq!((t, s.t()), (s.tcur(), s.tcur()));
            assert!((t - t_prev - s.hu()).abs() <= 1e-15 * t.abs());
            assert_eq!(s.y(), &s.st.yh[..1]);
            t_prev = t;
        }
        assert!((s.y()[0] - (-s.t()).exp()).abs() < 1e-5);
    }

    #[test]
    fn lsoda_advance_past_stops_at_first_mesh_point_beyond_tout() {
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()));
        let t = s.advance_past(0.5).unwrap();
        assert!(t >= 0.5 && t - s.hu() < 0.5, "t = {t}, hu = {}", s.hu());
        assert_eq!((s.t(), s.tcur(), s.y()), (t, t, &s.st.yh[..1]));
        // Already beyond TOUT: no step.
        let nst = s.nst();
        assert_eq!(s.advance_past(0.5), Ok(t));
        assert_eq!(s.nst(), nst);
        // TOUT behind TCUR - HU.
        let e = illegal(s.advance_past(0.0));
        assert!(matches!(e, InputError::TooutBehindLastStep { tout: 0.0, .. }), "{e:?}");
        let t2 = s.advance_past(1.0).unwrap();
        assert!(t2 >= 1.0 && s.nst() > nst);
    }

    #[test]
    fn lsoda_integrate_tcrit_never_steps_past_tcrit() {
        for kind in DECAY_KINDS {
            let tcrit = 1.0;
            let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.jacobian(kind).tolerance(tight()));
            assert_eq!(s.integrate_tcrit(0.5, tcrit), Ok(0.5));
            assert!(s.tcur() >= 0.5 && s.tcur() <= tcrit);
            assert!((s.y()[0] - (-0.5_f64).exp()).abs() < 1e-5);
            assert_eq!(s.integrate_tcrit(1.0, tcrit), Ok(1.0));
            assert!(s.tcur() <= tcrit * (1.0 + 1e-14), "{kind:?}: tcur = {}", s.tcur());
            assert!((s.y()[0] - (-1.0_f64).exp()).abs() < 1e-5);
            // TCRIT behind TCUR, and TCRIT behind TOUT.
            let e = illegal(s.integrate_tcrit(2.0, 0.5));
            assert_eq!(e, InputError::TcritBehindTcur { tcrit: 0.5, tcur: s.tcur() });
            let e = illegal(s.integrate_tcrit(3.0, 2.0));
            assert_eq!(e, InputError::TcritBehindTout { tcrit: 2.0, tout: 3.0 });
        }
        // On the first call, TCRIT behind TOUT.
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b);
        let e = illegal(s.integrate_tcrit(1.0, 0.5));
        assert_eq!(e, InputError::TcritBehindTout { tcrit: 0.5, tout: 1.0 });
        assert_eq!(s.nfe(), 0);
    }

    #[test]
    fn lsoda_integrate_tcrit_returns_tcrit_reached_before_tout() {
        // TOUT beyond TCRIT is illegal, so TCRIT = TOUT: whether the last
        // step lands within roundoff of TCRIT or just past TOUT, T = TOUT.
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()));
        assert_eq!(s.integrate_tcrit(0.7, 0.7), Ok(0.7));
        assert_eq!(s.t(), 0.7);
        assert!((s.tcur() - 0.7).abs() <= 1e-14, "tcur = {}", s.tcur());
        assert!((s.y()[0] - (-0.7_f64).exp()).abs() < 1e-5);
    }

    #[test]
    fn lsoda_step_tcrit_ends_exactly_at_tcrit() {
        let tcrit = 1.0;
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()));
        let mut calls = 0;
        loop {
            let nst = s.nst();
            let t = s.step_tcrit(0.5, tcrit).unwrap();
            calls += 1;
            assert_eq!(s.nst(), nst + 1);
            assert!(s.tcur() <= tcrit * (1.0 + 1e-14), "tcur = {}", s.tcur());
            assert_eq!(s.y(), &s.st.yh[..1]);
            if t == tcrit {
                break;
            }
            assert_eq!(t, s.tcur());
            assert!(calls < 1000);
        }
        assert!((s.y()[0] - (-1.0_f64).exp()).abs() < 1e-5);
        // At TCRIT already: no further step.
        let nst = s.nst();
        assert_eq!(s.step_tcrit(0.5, tcrit), Ok(tcrit));
        assert_eq!(s.nst(), nst);
        let e = illegal(s.step_tcrit(0.5, 0.5));
        assert!(matches!(e, InputError::TcritBehindTcur { tcrit: 0.5, .. }), "{e:?}");
    }

    #[test]
    fn lsoda_step_tcrit_first_step_clipped_to_tcrit() {
        // H0 = 0.5 would pass TCRIT = 0.25: block C sets H0 = TCRIT - T, and
        // y' = 1 is integrated exactly, so the first step lands on TCRIT.
        let mut s = lsoda(closure(|_t, _y, f| f[0] = 1.0), 0.0, &[0.0], |b| b.h0(0.5));
        assert_eq!(s.step_tcrit(0.25, 0.25), Ok(0.25));
        assert_eq!((s.nst(), s.hu(), s.tcur(), s.y()), (1, 0.25, 0.25, &[0.25][..]));
    }

    // ----- Initial step size, hmin, hmax ------------------------------------

    /// H0 by the DLSODA formula for a scalar problem with scalar tolerances
    /// and `rtol > 0`.
    fn dlsoda_h0(t: f64, tout: f64, y0: f64, f0: f64, rtol: f64, atol: f64) -> f64 {
        let tol = rtol.max(100.0 * UROUND);
        let tol = tol.min(0.001);
        let w0 = t.abs().max(tout.abs());
        let norm = f0.abs() * (1.0 / (rtol * y0.abs() + atol));
        let sum = 1.0 / (tol * w0 * w0) + tol * (norm * norm);
        (1.0 / sum.sqrt()).min((tout - t).abs()).copysign(tout - t)
    }

    #[test]
    fn lsoda_initial_step_size_follows_dlsoda_formula() {
        let tol = Tolerance::scalar(1e-6, 1e-8);
        let h0 = dlsoda_h0(1.0, 3.0, 2.0, -2.0, 1e-6, 1e-8);
        let mut s = lsoda(decay(), 1.0, &[2.0], |b| b.tolerance(tol.clone()));
        s.step(3.0).unwrap();
        assert_eq!(s.hu(), h0);

        // Backwards: the sign follows TOUT - T.
        let h0 = dlsoda_h0(1.0, -3.0, 2.0, -2.0, 1e-6, 1e-8);
        let mut s = lsoda(decay(), 1.0, &[2.0], |b| b.tolerance(tol.clone()));
        s.step(-3.0).unwrap();
        assert!(h0 < 0.0);
        assert_eq!(s.hu(), h0);

        // HMAX bounds H0: H0 / (|H0| * HMXI).
        let h0 = dlsoda_h0(1.0, 3.0, 2.0, -2.0, 1e-6, 1e-8);
        let mut s = lsoda(decay(), 1.0, &[2.0], |b| b.tolerance(tol.clone()).hmax(1e-4));
        s.step(3.0).unwrap();
        assert_eq!(s.hu(), h0 / (h0.abs() * (1.0 / 1e-4)));

        // |H0| <= |TOUT - T|: with f = 0 the formula gives sqrt(TOL) * W0 =
        // 1e-3, more than TOUT - T.
        let mut s = lsoda(closure(|_t, _y, f| f[0] = 0.0), 1.0, &[1.0], |b| b);
        let tout = 1.0 + 1e-6;
        s.step(tout).unwrap();
        assert_eq!(s.hu(), tout - 1.0);

        // Pure absolute control (RTOL = 0): TOL = max ATOL(i)/|Y(i)|.
        let mut s = lsoda(decay(), 0.0, &[4.0], |b| b.tolerance(Tolerance::scalar(0.0, 1e-6)));
        s.step(1.0).unwrap();
        let tol = 1e-6_f64 / 4.0;
        let norm = 4.0 * (1.0 / 1e-6);
        let h0 = 1.0 / (1.0 / (tol * 1.0 * 1.0) + tol * (norm * norm)).sqrt();
        assert_eq!(s.hu(), h0);
    }

    #[test]
    fn lsoda_given_h0_is_used_and_checked_against_direction() {
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.h0(1e-3));
        s.step(1.0).unwrap();
        assert_eq!(s.hu(), 1e-3);
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.h0(-1e-3));
        let e = illegal(s.integrate(1.0));
        assert_eq!(e, InputError::H0Direction { t: 0.0, tout: 1.0, h0: -1e-3 });
        assert_eq!(s.nfe(), 0);
    }

    #[test]
    fn lsoda_hmax_maps_to_hmxi_like_dlsoda() {
        for (hmax, hmxi) in [(f64::MAX, 0.0), (0.0, 0.0), (f64::INFINITY, 0.0), (0.5, 2.0)] {
            let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.hmax(hmax));
            s.integrate(1.0).unwrap();
            assert_eq!(s.st.hmxi, hmxi, "hmax = {hmax}");
            assert!(hmxi == 0.0 || s.hu() <= hmax);
        }
    }

    #[test]
    fn lsoda_hmin_is_passed_to_the_stepper() {
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.hmin(1e-3));
        s.integrate(1.0).unwrap();
        assert_eq!(s.st.hmin, 1e-3);
    }

    // ----- Error returns -----------------------------------------------------

    #[test]
    fn lsoda_new_checks_y0_and_hand_built_options() {
        let opts = OptionsBuilder::new(2).build(&decay()).unwrap();
        let e = illegal(Lsoda::new(decay(), opts.clone(), 0.0, &[1.0]));
        assert_eq!(e, InputError::YLength { expected: 2, got: 1 });
        // Hand-built options are checked and normalized like the builder's.
        let mut o = opts;
        (o.neq, o.mxordn, o.mxords, o.mxstep, o.mxhnil) = (1, 0, 9, 0, 0);
        let s = Lsoda::new(decay(), o.clone(), 0.0, &[1.0]).unwrap();
        let n = s.options();
        assert_eq!((n.mxordn, n.mxords, n.mxstep, n.mxhnil), (12, 5, 10000, 10));
        o.hmin = -1.0;
        assert_eq!(illegal(Lsoda::new(decay(), o, 0.0, &[1.0])), InputError::NegativeHmin(-1.0));
    }

    #[test]
    fn lsoda_max_steps_exceeded_then_continues() {
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()).mxstep(10));
        let mut interrupts = 0;
        loop {
            let nst = s.nst();
            match s.integrate(1.0) {
                Ok(t) => {
                    assert_eq!(t, 1.0);
                    break;
                }
                Err(e) => {
                    // Each call may take MXSTEP steps; T and Y are at TCUR.
                    assert_eq!(e, LsodaError::MaxStepsExceeded { t: s.tcur(), mxstep: 10 });
                    assert_eq!(s.nst(), nst + 10);
                    assert_eq!((s.t(), s.y()), (s.tcur(), &s.st.yh[..1]));
                    interrupts += 1;
                }
            }
        }
        assert!(interrupts >= 1);
        // The interruptions do not change the steps taken.
        let mut b = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()));
        b.integrate(1.0).unwrap();
        assert_eq!((s.nst(), s.nfe(), s.y()), (b.nst(), b.nfe(), b.y()));
    }

    #[test]
    fn lsoda_too_much_accuracy_at_start_is_illegal_input() {
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(Tolerance::scalar(0.0, 1e-20)));
        let e = illegal(s.integrate(1.0));
        assert_eq!(e, InputError::TooMuchAccuracyAtStart { tolsf: UROUND * (1.0 / 1e-20) * 2.0 });
        assert_eq!((s.nst(), s.t(), s.y()), (0, 0.0, &[1.0][..]));
        // Still not started: a call with usable tolerances starts afresh.
        s.set_tolerance(tight()).unwrap();
        assert!(!s.changed);
        assert_eq!(s.integrate(1.0), Ok(1.0));
        assert_eq!(s.nhnil, 0);
    }

    #[test]
    fn lsoda_too_much_accuracy_during_the_run_then_resumes() {
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()));
        s.integrate(0.5).unwrap();
        let (tcur, nst) = (s.tcur(), s.nst());
        s.set_tolerance(Tolerance::scalar(0.0, 1e-20)).unwrap();
        let Err(LsodaError::TooMuchAccuracy { t, tolsf }) = s.integrate(1.0) else { panic!() };
        assert_eq!((t, s.t(), s.tcur(), s.nst()), (tcur, tcur, tcur, nst));
        assert!(tolsf > 1.0);
        assert_eq!(s.y(), &s.st.yh[..1]);
        // ISTATE = 3 with tolerances scaled back up.
        s.set_tolerance(tight()).unwrap();
        assert_eq!(s.integrate(1.0), Ok(1.0));
        assert!((s.y()[0] - (-1.0_f64).exp()).abs() < 1e-5);
    }

    #[test]
    fn lsoda_error_weight_zero_during_the_run() {
        // y' = -4 from y = 1 with H0 = 0.25: the first step lands exactly on
        // y = 0, where pure relative control gives EWT = 0.
        let mut s = lsoda(closure(|_t, _y, f| f[0] = -4.0), 0.0, &[1.0], |b| {
            b.h0(0.25).tolerance(Tolerance::scalar(1e-3, 0.0))
        });
        assert_eq!(s.integrate(1.0), Err(LsodaError::ErrorWeightZero { t: 0.25, component: 0 }));
        assert_eq!((s.t(), s.y(), s.nst()), (0.25, &[0.0][..], 1));
        // At the start of the problem it is an illegal input.
        let sys = closure(|_t, y, f| {
            f[0] = -y[0];
            f[1] = 0.0;
        });
        let mut s = lsoda(sys, 0.0, &[1.0, 0.0], |b| b.tolerance(Tolerance::scalar(1e-3, 0.0)));
        let e = illegal(s.integrate(1.0));
        assert_eq!(e, InputError::ErrorWeightNonPositive { index: 1, value: 0.0 });
    }

    #[test]
    fn lsoda_error_test_failure_at_hmin() {
        // As in the stepper tests: with H = HMIN = 0.01 the error test fails.
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()).h0(0.01).hmin(0.01));
        let e = s.integrate(1.0).unwrap_err();
        assert_eq!(e, LsodaError::ErrorTestFailed { t: 0.0, h: 0.01, imxer: 0 });
        assert_eq!((s.t(), s.nst()), (0.0, 0));
        assert!((s.y()[0] - 1.0).abs() <= 1e-15);
    }

    #[test]
    fn lsoda_convergence_failure_at_hmin_and_imxer() {
        // Functional iteration diverges with H = HMIN = 10.
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()).h0(10.0).hmin(10.0));
        let e = s.integrate(100.0).unwrap_err();
        assert_eq!(e, LsodaError::ConvergenceFailed { t: 0.0, h: 10.0, imxer: 0 });
        assert_eq!((s.t(), s.nst()), (0.0, 0));
        // IMXER: the component with the largest |ACOR(i) * EWT(i)|.
        let sys = closure(|_t, y, f| {
            f[0] = -y[0];
            f[1] = -3.0 * y[1];
            f[2] = -2.0 * y[2];
        });
        let mut s = lsoda(sys, 0.0, &[1.0; 3], |b| b.tolerance(tight()).h0(10.0).hmin(10.0));
        let e = s.integrate(100.0).unwrap_err();
        assert!(matches!(e, LsodaError::ConvergenceFailed { imxer: 1, .. }), "{e:?}");
    }

    #[test]
    fn lsoda_tout_too_close_to_t() {
        let mut s = lsoda(decay(), 1.0, &[1.0], |b| b);
        let tout = 1.0 + f64::EPSILON;
        assert_eq!(illegal(s.integrate(tout)), InputError::TooutTooClose { t: 1.0, tout });
    }

    #[test]
    fn lsoda_callback_error_before_first_step_allows_a_fresh_start() {
        // f fails once, on its `fail_at`-th call: in block C (1) or in the
        // first step (2). No step was completed, so the next call starts the
        // problem again.
        for fail_at in [1, 2] {
            let calls = Rc::new(Cell::new(0));
            let c = calls.clone();
            let sys = ClosureSystem::new(move |_t, y: &[f64], f: &mut [f64]| {
                c.set(c.get() + 1);
                if c.get() == fail_at {
                    return Err(LsodaError::Callback { ierr: -3 });
                }
                f[0] = -y[0];
                Ok(())
            });
            let mut s = lsoda(sys, 0.0, &[1.0], |b| b.tolerance(tight()));
            assert_eq!(s.integrate(1.0), Err(LsodaError::Callback { ierr: -3 }));
            assert_eq!((s.t(), s.y()), (0.0, &[1.0][..]));
            assert_eq!(s.integrate(1.0), Ok(1.0));
            assert!((s.y()[0] - (-1.0_f64).exp()).abs() < 1e-5);
        }
    }

    #[test]
    fn lsoda_callback_error_mid_step_halts() {
        let calls = Rc::new(Cell::new(0));
        let c = calls.clone();
        let sys = ClosureSystem::new(move |_t, y: &[f64], f: &mut [f64]| {
            c.set(c.get() + 1);
            if c.get() == 12 {
                return Err(LsodaError::Callback { ierr: -1 });
            }
            f[0] = -y[0];
            Ok(())
        });
        let mut s = lsoda(sys, 0.0, &[1.0], |b| b.tolerance(tight()));
        s.step(1.0).unwrap();
        let (t, y) = (s.t(), s.y().to_vec());
        assert_eq!(s.integrate(1.0), Err(LsodaError::Callback { ierr: -1 }));
        // T and Y keep the last output; the solver refuses to continue.
        assert_eq!((s.t(), s.y()), (t, &y[..]));
        let calls_before = calls.get();
        assert_eq!(illegal(s.integrate(1.0)), InputError::Halted);
        assert_eq!(illegal(s.step(1.0)), InputError::Halted);
        assert_eq!(calls.get(), calls_before);
    }

    // ----- ISTATE = 3 --------------------------------------------------------

    #[test]
    fn lsoda_setters_apply_on_the_next_call_as_istate_3() {
        let mut s = rober(JacobianKind::DenseFd);
        // Before the start, setters only change the options.
        s.set_mxstep(0);
        assert!(!s.changed);
        assert_eq!(s.options().mxstep, MXSTP0);
        s.integrate(0.4).unwrap();
        assert_eq!((s.mcur(), s.st.jstart), (Method::Bdf, 1));
        s.set_tolerance(rober_tol()).unwrap();
        assert!(s.changed);
        // A call that needs no step still takes over the changes, with
        // JSTART = -1 for DSTODA.
        let (nst, nje) = (s.nst(), s.nje());
        assert_eq!(s.integrate(0.4), Ok(0.4));
        assert_eq!((s.changed, s.st.jstart, s.nst()), (false, -1, nst));
        // JSTART = -1 forces a new Jacobian on the next step.
        s.step(1.0).unwrap();
        assert_eq!((s.nje(), s.st.jstart), (nje + 1, 1));
        s.integrate(4.0).unwrap();
        assert_rober_accurate(s.y(), 4.0, "after ISTATE = 3");

        // Rejected changes leave everything as it was.
        let e = illegal(s.set_tolerance(Tolerance::scalar(-1.0, 1e-6)));
        assert_eq!(e, InputError::NegativeRtol { index: 0, value: -1.0 });
        let e = illegal(s.set_tolerance(Tolerance::per_component(vec![1e-4; 2], vec![1e-6; 3])));
        assert_eq!(e, InputError::ToleranceLength { expected: 3, got: 2 });
        assert_eq!(illegal(s.set_hmax(-1.0)), InputError::NegativeHmax(-1.0));
        assert!(matches!(illegal(s.set_hmin(f64::NAN)), InputError::NegativeHmin(v) if v.is_nan()));
        let e = illegal(s.set_jacobian(JacobianKind::BandedFd { ml: 3, mu: 0 }));
        assert_eq!(e, InputError::IllegalMl { ml: 3, neq: 3 });
        let mut no_jac = lsoda(closure(|_t, y, f| f[0] = -y[0]), 0.0, &[1.0], |b| b);
        assert_eq!(illegal(no_jac.set_jacobian(JacobianKind::DenseUser)), InputError::MissingJacobian);
        assert_eq!(s.options().tolerance, rober_tol());
        assert_eq!(s.options().jacobian, JacobianKind::DenseFd);
        assert!(!s.changed);
    }

    #[test]
    fn lsoda_set_hmax_bounds_steps_from_the_next_change_of_h() {
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()));
        s.integrate(1.0).unwrap();
        assert!(s.hcur() > 0.1, "h = {}", s.hcur());
        s.set_hmax(0.05).unwrap();
        s.set_hmin(1e-9).unwrap();
        for k in 0..60 {
            s.step(10.0).unwrap();
            // H is only rescaled at its next change, within NQ + 1 steps.
            if k >= 14 {
                assert!(s.hu() <= 0.05 * (1.0 + 1e-12), "step {k}: hu = {}", s.hu());
            }
        }
        assert_eq!((s.st.hmxi, s.st.hmin), (1.0 / 0.05, 1e-9));
        s.set_hmax(f64::MAX).unwrap();
        s.step(10.0).unwrap();
        assert_eq!(s.st.hmxi, 0.0);
    }

    #[test]
    fn lsoda_set_mxstep_takes_effect_on_the_next_call() {
        let mut s = lsoda(decay(), 0.0, &[1.0], |b| b.tolerance(tight()));
        s.integrate(0.1).unwrap();
        s.set_mxstep(3);
        let nst = s.nst();
        assert!(matches!(s.integrate(10.0), Err(LsodaError::MaxStepsExceeded { mxstep: 3, .. })));
        assert_eq!(s.nst(), nst + 3);
    }

    #[test]
    fn lsoda_set_jacobian_follows_dlsoda_miter_rules() {
        // BDF with a dense MITER keeps it after a change to a banded JT ...
        let mut s = rober(JacobianKind::DenseFd);
        s.integrate(0.4).unwrap();
        s.set_jacobian(JacobianKind::BandedFd { ml: 2, mu: 2 }).unwrap();
        s.integrate(4.0).unwrap();
        assert_eq!(s.st.jtyp, JacobianKind::BandedFd { ml: 2, mu: 2 });
        assert_eq!((s.st.miter, s.st.wm.storage), (Some(JacobianKind::DenseFd), MatrixStorage::Dense));
        assert_rober_accurate(s.y(), 4.0, "dense MITER kept");

        // ... while a banded MITER keeps its type but takes the new band
        // widths at once (ml = 1 is exact here: df3/dy1 = 0).
        let mut s = rober(JacobianKind::BandedFd { ml: 2, mu: 2 });
        s.integrate(0.4).unwrap();
        s.set_jacobian(JacobianKind::BandedUser { ml: 1, mu: 2 }).unwrap();
        s.integrate(4.0).unwrap();
        assert_eq!(s.st.miter, Some(JacobianKind::BandedFd { ml: 1, mu: 2 }));
        assert_eq!(s.st.wm.storage, MatrixStorage::Banded { ml: 1, mu: 2 });
        assert_rober_accurate(s.y(), 4.0, "banded MITER with new band");

        // A change made while Adams is in use applies at the switch to BDF.
        let mut s = rober(JacobianKind::DenseFd);
        s.integrate(1e-3).unwrap();
        assert_eq!(s.mcur(), Method::Adams);
        s.set_jacobian(JacobianKind::BandedUser { ml: 2, mu: 2 }).unwrap();
        s.integrate(0.4).unwrap();
        assert_eq!(s.st.miter, Some(JacobianKind::BandedUser { ml: 2, mu: 2 }));
        assert_eq!(s.st.wm.storage, MatrixStorage::Banded { ml: 2, mu: 2 });
        assert_rober_accurate(s.y(), 0.4, "JT changed before the switch");
    }

    // ----- Hooks -------------------------------------------------------------

    #[test]
    fn lsoda_warns_through_hook_when_t_plus_h_equals_t() {
        // At t = 1e20 every step with |h| < 8192 leaves T unchanged. With
        // y' = 0 and H0 = 1e-3 the steps are 1e-3, 1e-3, 10, 10, 10, 100.
        let warnings = Mutex::new(Vec::new());
        let mut s = lsoda(closure(|_t, _y, f| f[0] = 0.0), 1e20, &[1.0], |b| b.h0(1e-3).mxhnil(3).mxstep(6));
        s.on_warning(|w| warnings.lock().unwrap().push(*w));
        assert_eq!(s.integrate(2e20), Err(LsodaError::MaxStepsExceeded { t: 1e20, mxstep: 6 }));
        assert_eq!(s.nhnil, 6);
        drop(s);
        let warnings = warnings.into_inner().unwrap();
        let want: Vec<Warning> = [(1e-3, 1), (1e-3, 2), (10.0, 3)]
            .into_iter()
            .map(|(h, count)| Warning::NegligibleStep { t: 1e20, h, count, mxhnil: 3 })
            .collect();
        assert_eq!(warnings, want);

        // Without a hook the warnings are counted but go nowhere.
        let mut s = lsoda(closure(|_t, _y, f| f[0] = 0.0), 1e20, &[1.0], |b| b.h0(1e-3).mxstep(6));
        assert!(s.integrate(2e20).is_err());
        assert_eq!(s.nhnil, 6);
    }

    #[test]
    fn lsoda_method_switch_reaches_warning_hook_only_with_ixpr() {
        for ixpr in [false, true] {
            let warnings = Mutex::new(Vec::new());
            let switches = Mutex::new(Vec::new());
            let mut s = lsoda(robertson(), 0.0, &[1.0, 0.0, 0.0], |b| b.tolerance(rober_tol()).ixpr(ixpr));
            s.on_warning(|w| warnings.lock().unwrap().push(*w))
                .on_method_switch(|e| switches.lock().unwrap().push(*e));
            s.integrate(0.4).unwrap();
            let tsw = s.tsw();
            drop(s);
            let switches = switches.into_inner().unwrap();
            assert_eq!(switches.len(), 1);
            assert_eq!((switches[0].method, switches[0].t), (Method::Bdf, tsw));
            let want: Vec<Warning> = if ixpr { vec![Warning::MethodSwitch(switches[0])] } else { vec![] };
            assert_eq!(warnings.into_inner().unwrap(), want);
        }
        // set_ixpr turns the reports on for a running problem.
        let warnings = Mutex::new(Vec::new());
        let mut s = rober(JacobianKind::DenseFd);
        s.on_warning(|w| warnings.lock().unwrap().push(*w));
        s.integrate(1e-3).unwrap();
        s.set_ixpr(true);
        s.integrate(0.4).unwrap();
        drop(s);
        let warnings = warnings.into_inner().unwrap();
        assert!(matches!(warnings[..], [Warning::MethodSwitch(MethodSwitch { method: Method::Bdf, .. })]), "{warnings:?}");
    }

    #[test]
    fn lsoda_is_send_when_the_system_is() {
        struct Decay;
        impl OdeSystem for Decay {
            fn rhs(&mut self, _t: f64, y: &[f64], ydot: &mut [f64]) -> Result<(), LsodaError> {
                ydot[0] = -y[0];
                Ok(())
            }
        }
        fn assert_send<T: Send>(_: &T) {}
        let opts = OptionsBuilder::new(1).build(&Decay).unwrap();
        let mut s = Lsoda::new(Decay, opts, 0.0, &[1.0]).unwrap();
        s.on_warning(|_| {});
        assert_send(&s);
        let y = std::thread::spawn(move || {
            s.integrate(1.0).unwrap();
            s.y()[0]
        })
        .join()
        .unwrap();
        assert!((y - (-1.0_f64).exp()).abs() < 1e-5);
    }
}
