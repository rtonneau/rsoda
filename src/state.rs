//! Solver state shared by the stepper and the driver.
//!
//! [`State`] is the Rust analogue of the fork's `odepack_common_data` as
//! LSODA uses it: the scalars of `COMMON /DLS001/` and `COMMON /DLSA01/`, plus
//! the work arrays DLSODA carves out of `RWORK` and `IWORK` (`YH`, `EWT`,
//! `SAVF`, `ACOR`, `WM`/`IWM`) and the `Y` array DSTODA uses as scratch.
//!
//! Field names follow the Fortran names in lower case. Indices stored in the
//! state (`nq`, `l`, `lmax`, ...) keep their Fortran (1-based) meaning; only
//! array subscripts are shifted to 0-based. The pointer variables (`LYH`,
//! `LEWT`, ...), the `IOWNS`/`ROWNS` placeholders, `UROUND` (see
//! [`crate::consts::UROUND`]) and the work-space flags `INSUFR`/`INSUFI` have
//! no counterpart: Rust allocates every array at its full size up front.

use crate::coefficients::{Elco, Method, Tesco};
use crate::consts::{MORD_NONSTIFF, MORD_STIFF};
use crate::options::JacobianKind;
use crate::solsy::{IterationMatrix, MatrixStorage};

/// The LSODA common data and work arrays (`/DLS001/`, `/DLSA01/`, `RWORK`,
/// `IWORK`).
///
/// [`crate::stoda::stoda`] reads and updates the stepper fields; the fields
/// marked "driver" are maintained by DLSODA and never changed by DSTODA.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    // ----- /DLS001/: reals -------------------------------------------------
    /// `CONIT = 0.5 / (NQ + 2)`, used in the corrector convergence test.
    pub conit: f64,
    /// `CRATE`: estimated corrector convergence rate.
    pub crate_: f64,
    /// `EL(13)`: coefficients of the current method and order
    /// (`el[i - 1] = EL(i)`).
    pub el: [f64; 13],
    /// `ELCO(13,12)`, filled by [`crate::coefficients::cfode`].
    pub elco: Elco,
    /// `HOLD`: step size in use when the last step ended (lets the caller
    /// change `h` between steps).
    pub hold: f64,
    /// `RMAX`: largest step-size ratio allowed on the next change of `h`.
    pub rmax: f64,
    /// `TESCO(3,12)`, filled by [`crate::coefficients::cfode`].
    pub tesco: Tesco,
    /// `CCMAX`: largest relative change in `h*el0` before the iteration
    /// matrix is re-formed (0.3).
    pub ccmax: f64,
    /// `EL0 = EL(1)` when the iteration matrix was last formed.
    pub el0: f64,
    /// `H`: step size to be attempted on the next step.
    pub h: f64,
    /// `HMIN`: smallest `|h|` allowed.
    pub hmin: f64,
    /// `HMXI = 1 / HMAX`, with `0.0` meaning no maximum.
    pub hmxi: f64,
    /// `HU`: step size of the last successful step.
    pub hu: f64,
    /// `RC`: ratio of the current `h*el(1)` to the value used in the
    /// iteration matrix.
    pub rc: f64,
    /// `TN`: the independent variable, advanced on each step.
    pub tn: f64,

    // ----- /DLS001/: integers ----------------------------------------------
    /// `NSLAST` (driver): `nst` at the start of the current driver call,
    /// used for the `mxstep` test.
    pub nslast: usize,
    /// `NYH`: leading dimension of `yh` (the problem size at the start).
    pub nyh: usize,
    /// `IALTH`: steps left before a change of step size or order is
    /// considered.
    pub ialth: i32,
    /// `IPUP`: `true` when the iteration matrix must be re-formed before the
    /// next corrector iteration (Fortran `IPUP = MITER`, tested `IPUP > 0`).
    pub ipup: bool,
    /// `LMAX = MAXORD + 1`.
    pub lmax: usize,
    /// `NQNYH = NQ * NYH`.
    pub nqnyh: usize,
    /// `NSLP`: value of `nst` when the iteration matrix was last formed.
    pub nslp: usize,
    /// `ICF`: corrector failure flag of the last call (0 none, 1 recovered
    /// by a Jacobian update, 2 convergence failure).
    pub icf: i32,
    /// `IERPJ`: Jacobian preparation error flag (1 = singular matrix).
    pub ierpj: i32,
    /// `IERSL`: linear solve error flag.
    pub iersl: i32,
    /// `JCUR`: the Jacobian was evaluated at the current step.
    pub jcur: bool,
    /// `JSTART`: 0 first step, > 0 continue, -1 continue after changes to
    /// the method or parameters, -2 continue with a new `h` only. DSTODA
    /// sets it to 1 on return.
    pub jstart: i32,
    /// `KFLAG`: completion code of the last step (0, -1, -2 or -3).
    pub kflag: i32,
    /// `L = NQ + 1`: number of columns of `yh` in use.
    pub l: usize,
    /// `METH`: the current method.
    pub meth: Method,
    /// `MITER`: `None` for functional iteration (Adams), otherwise the
    /// chord iteration of Jacobian kind `jtyp` (BDF).
    pub miter: Option<JacobianKind>,
    /// `MAXORD`: largest order allowed for the current method.
    pub maxord: usize,
    /// `MAXCOR`: maximum corrector iterations per attempt (3).
    pub maxcor: usize,
    /// `MSBP`: maximum steps between iteration-matrix updates (20).
    pub msbp: usize,
    /// `MXNCF`: maximum convergence failures per step (10).
    pub mxncf: usize,
    /// `N`: number of equations.
    pub n: usize,
    /// `NQ`: order to be attempted on the next step.
    pub nq: usize,
    /// `NST`: number of steps taken.
    pub nst: usize,
    /// `NFE`: number of `f` evaluations.
    pub nfe: usize,
    /// `NJE`: number of Jacobian evaluations (and matrix factorizations).
    pub nje: usize,
    /// `NQU`: order of the last successful step.
    pub nqu: usize,

    // ----- /DLSA01/ ---------------------------------------------------------
    /// `TSW` (driver): `tn` at the last method switch.
    pub tsw: f64,
    /// `CM1(12)`: `TESCO(2,q) * ELCO(q+1,q)` for the Adams methods.
    pub cm1: [f64; 12],
    /// `CM2(5)`: `TESCO(2,q) * ELCO(q+1,q)` for the BDF methods.
    pub cm2: [f64; 5],
    /// `PDEST`: estimated Lipschitz constant over the current step.
    pub pdest: f64,
    /// `PDLAST`: most recent nonzero `PDEST`.
    pub pdlast: f64,
    /// `RATIO`: step-size advantage required to switch to BDF (5).
    pub ratio: f64,
    /// `PDNORM`: norm of the last Jacobian, from the Jacobian preparation.
    pub pdnorm: f64,
    /// `ICOUNT`: steps left before a method switch is considered.
    pub icount: i32,
    /// `IRFLAG`: the last step-size change was restricted by the Adams
    /// stability region.
    pub irflag: bool,
    /// `JTYP`: the Jacobian kind used whenever the BDF method is active.
    pub jtyp: JacobianKind,
    /// `MUSED`: method of the last successful step (`None` before the first
    /// one, Fortran `MUSED = 0`).
    pub mused: Option<Method>,
    /// `MXORDN`: maximum Adams order.
    pub mxordn: usize,
    /// `MXORDS`: maximum BDF order.
    pub mxords: usize,

    // ----- Work arrays ------------------------------------------------------
    /// `Y`: scratch vector passed to every `f` and Jacobian call (length `n`).
    pub y: Vec<f64>,
    /// `YH(NYH, LMAX)`: the Nordsieck history array, column-major;
    /// `YH(i, j)` is `yh[(i - 1) + (j - 1) * nyh]`. It holds
    /// `max(mxordn, mxords) + 1` columns.
    pub yh: Vec<f64>,
    /// `EWT`: the **inverted** error weights (length `n`).
    pub ewt: Vec<f64>,
    /// `SAVF`: work vector (length `n`).
    pub savf: Vec<f64>,
    /// `ACOR`: accumulated corrections; after a successful step, the
    /// estimated local error (length `n`).
    pub acor: Vec<f64>,
    /// `WM`/`IWM`: the iteration matrix `P = I - h*el0*J` and its pivots,
    /// stored for `jtyp`.
    pub wm: IterationMatrix,
}

impl State {
    /// Allocates the state for `n` equations, BDF Jacobian kind `jtyp` and
    /// maximum orders `mxordn` (Adams, `1..=12`) and `mxords` (BDF,
    /// `1..=5`).
    ///
    /// The constants DLSODA sets for the initial call (block C) are applied:
    /// `METH = 1`, `MITER = 0`, `MAXORD = MXORDN`, `CCMAX = 0.3`,
    /// `MAXCOR = 3`, `MSBP = 20`, `MXNCF = 10`, `JSTART = 0`, zero counters,
    /// `HU = 0`, `NQU = 0`, `MUSED = 0`, `NQ = 1`. `HMIN` and `HMXI` are 0
    /// (no bounds). The driver still has to load `tn`, `h`, the first two
    /// columns of `yh` and `ewt` before the first step.
    pub fn new(n: usize, jtyp: JacobianKind, mxordn: usize, mxords: usize) -> Self {
        assert!(n > 0, "n must be positive");
        assert!((1..=MORD_NONSTIFF).contains(&mxordn), "mxordn must be in 1..=12");
        assert!((1..=MORD_STIFF).contains(&mxords), "mxords must be in 1..=5");
        let nyh = n;
        let ncol = mxordn.max(mxords) + 1;
        Self {
            conit: 0.0,
            crate_: 0.0,
            el: [0.0; 13],
            elco: [[0.0; 13]; 12],
            hold: 0.0,
            rmax: 0.0,
            tesco: [[0.0; 3]; 12],
            ccmax: 0.3,
            el0: 0.0,
            h: 0.0,
            hmin: 0.0,
            hmxi: 0.0,
            hu: 0.0,
            rc: 0.0,
            tn: 0.0,
            nslast: 0,
            nyh,
            ialth: 0,
            ipup: false,
            lmax: mxordn + 1,
            nqnyh: 0,
            nslp: 0,
            icf: 0,
            ierpj: 0,
            iersl: 0,
            jcur: false,
            jstart: 0,
            kflag: 0,
            l: 2,
            meth: Method::Adams,
            miter: None,
            maxord: mxordn,
            maxcor: 3,
            msbp: 20,
            mxncf: 10,
            n,
            nq: 1,
            nst: 0,
            nfe: 0,
            nje: 0,
            nqu: 0,
            tsw: 0.0,
            cm1: [0.0; 12],
            cm2: [0.0; 5],
            pdest: 0.0,
            pdlast: 0.0,
            ratio: 0.0,
            pdnorm: 0.0,
            icount: 0,
            irflag: false,
            jtyp,
            mused: None,
            mxordn,
            mxords,
            y: vec![0.0; n],
            yh: vec![0.0; nyh * ncol],
            ewt: vec![0.0; n],
            savf: vec![0.0; n],
            acor: vec![0.0; n],
            wm: IterationMatrix::new(n, MatrixStorage::for_jacobian(&jtyp)),
        }
    }

    /// Column `j` (1-based) of the history array, `YH(1:N, j)`.
    pub fn yh_col(&self, j: usize) -> &[f64] {
        let start = (j - 1) * self.nyh;
        &self.yh[start..start + self.n]
    }

    /// The method-switch bookkeeping DLSODA does after a successful step
    /// (block F, `KFLAG = 0`).
    ///
    /// If DSTODA has just switched methods (`METH != MUSED`), this records
    /// the switch time `TSW = TN`, resets `MAXORD` for the new method and
    /// sets `JSTART = -1` so that the next step completes the switch (new
    /// coefficients and a forced Jacobian update). Returns the new method if
    /// a switch occurred, `None` otherwise (and after a failed step).
    ///
    /// The Fortran's `RWORK(LWM) = SQRT(UROUND)` has no counterpart: the
    /// Jacobian preparation computes `sqrt(UROUND)` itself.
    pub fn record_method_switch(&mut self) -> Option<Method> {
        if self.kflag != 0 || self.mused == Some(self.meth) {
            return None;
        }
        self.tsw = self.tn;
        self.maxord = match self.meth {
            Method::Adams => self.mxordn,
            Method::Bdf => self.mxords,
        };
        self.jstart = -1;
        Some(self.meth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stoda_state_new_applies_dlsoda_initial_constants() {
        let st = State::new(3, JacobianKind::BandedFd { ml: 1, mu: 0 }, 12, 5);
        assert_eq!((st.ccmax, st.maxcor, st.msbp, st.mxncf), (0.3, 3, 20, 10));
        assert_eq!((st.meth, st.miter, st.maxord, st.jstart), (Method::Adams, None, 12, 0));
        assert_eq!((st.nst, st.nfe, st.nje, st.nqu, st.mused), (0, 0, 0, 0, None));
        assert_eq!(st.yh.len(), 3 * 13);
        assert_eq!(st.wm.storage, MatrixStorage::Banded { ml: 1, mu: 0 });
        // History storage covers the larger of the two maximum orders.
        let st = State::new(2, JacobianKind::DenseFd, 3, 5);
        assert_eq!(st.yh.len(), 2 * 6);
    }

    #[test]
    fn stoda_state_records_method_switch_like_block_f() {
        let mut st = State::new(2, JacobianKind::DenseFd, 10, 4);
        st.tn = 1.5;
        st.mused = Some(Method::Adams);
        assert_eq!(st.record_method_switch(), None);
        st.meth = Method::Bdf;
        st.jstart = 1;
        assert_eq!(st.record_method_switch(), Some(Method::Bdf));
        assert_eq!((st.tsw, st.maxord, st.jstart), (1.5, 4, -1));
        // Back to Adams.
        st.mused = Some(Method::Bdf);
        st.meth = Method::Adams;
        st.tn = 3.0;
        assert_eq!(st.record_method_switch(), Some(Method::Adams));
        assert_eq!((st.tsw, st.maxord), (3.0, 10));
        // Never after a failed step.
        st.mused = None;
        st.kflag = -1;
        assert_eq!(st.record_method_switch(), None);
    }
}
