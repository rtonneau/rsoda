//! `Lsoda::restart` against a solver created afresh by `Lsoda::new` with the
//! same options: the integrations must agree bitwise (solution, step and
//! evaluation counts), whatever the solver did before the restart.
//!
//! `Lsoda::restart_warm`: on BDF it starts on BDF at order 1 and stays
//! within tolerance of a fresh solve; on Adams it is `restart`.

use rsoda::{
    ClosureSystem, InputError, JacobianKind, Lsoda, LsodaError, Method, Options, OptionsBuilder, Tol, Tolerance,
};

/// The Robertson problem of the DLSODA prologue, with its Jacobian in the
/// dense (`pd` of length 9) and the full banded `ml = mu = 2` (`pd` of
/// length 15) layouts.
fn robertson() -> ClosureSystem<'static> {
    ClosureSystem::new(|_t, y: &[f64], f: &mut [f64]| {
        f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
        f[2] = 3.0e7 * y[1] * y[1];
        f[1] = -f[0] - f[2];
        Ok(())
    })
    .with_jac(|_t, y: &[f64], pd: &mut [f64]| {
        let j = jacobian(y);
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

fn jacobian(y: &[f64]) -> [[f64; 3]; 3] {
    [[-0.04, 1.0e4 * y[2], 1.0e4 * y[1]], [0.04, -1.0e4 * y[2] - 6.0e7 * y[1], -1.0e4 * y[1]], [0.0, 6.0e7 * y[1], 0.0]]
}

/// Robertson with the sparse Jacobian (feature `sparse`).
#[cfg(feature = "sparse")]
fn robertson_sparse(user: bool) -> ClosureSystem<'static> {
    use rsoda::SparsityPattern;
    let entries = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)];
    let p = SparsityPattern::from_entries(3, entries).unwrap();
    let sys = ClosureSystem::new(|_t, y: &[f64], f: &mut [f64]| {
        f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
        f[2] = 3.0e7 * y[1] * y[1];
        f[1] = -f[0] - f[2];
        Ok(())
    });
    if !user {
        return sys.with_sparsity(p);
    }
    let pj = p.clone();
    sys.with_sparse_jac(p, move |_t, y: &[f64], values: &mut [f64]| {
        let j = jacobian(y);
        for (r, c) in entries {
            values[pj.find(r, c).unwrap()] = j[r][c];
        }
        Ok(())
    })
}

fn options(sys: &ClosureSystem<'_>, kind: JacobianKind, h0: f64) -> Options {
    let tolerance = Tolerance { rtol: Tol::Scalar(1e-4), atol: Tol::PerComponent(vec![1e-6, 1e-10, 1e-6]) };
    OptionsBuilder::new(3).jacobian(kind).tolerance(tolerance).h0(h0).build(sys).unwrap()
}

/// The starting states `(t0, y0, h0)`: the prologue's, later points of the
/// Robertson solution with a given or a computed first step.
const STATES: [(f64, [f64; 3], f64); 4] = [
    (0.0, [1.0, 0.0, 0.0], 0.0),
    (0.4, [9.851712e-01, 3.386380e-05, 1.479493e-02], 1e-6),
    (40.0, [7.158403e-01, 9.186334e-06, 2.841505e-01], 0.0),
    (4.0e3, [1.831975e-01, 8.941774e-07, 8.168016e-01], 1e-3),
];

/// What an integration produced, as bits: the outputs and the final
/// counters and step data.
#[derive(Debug, PartialEq)]
struct Trace {
    outputs: Vec<(u64, [u64; 3])>,
    counts: [usize; 5],
    hu: u64,
    tcur: u64,
    tsw: u64,
    mused: Option<Method>,
}

/// Integrates from the solver's current start through ten decades of `t`.
fn trace<S: rsoda::OdeSystem>(s: &mut Lsoda<'_, S>) -> Trace {
    let t0 = s.t();
    let mut outputs = Vec::new();
    let mut tout = 0.4;
    for _ in 0..10 {
        let t = s.integrate(t0 + tout).unwrap();
        let y = s.y();
        outputs.push((t.to_bits(), [y[0].to_bits(), y[1].to_bits(), y[2].to_bits()]));
        tout *= 10.0;
    }
    Trace {
        outputs,
        counts: [s.nst(), s.nfe(), s.nje(), s.nqu(), s.nqcur()],
        hu: s.hu().to_bits(),
        tcur: s.tcur().to_bits(),
        tsw: s.tsw().to_bits(),
        mused: s.mused(),
    }
}

/// Runs every state of [`STATES`] on one solver, restarted between them,
/// and on a fresh solver each, and checks they agree.
fn check_restart(kind: JacobianKind, system: impl Fn() -> ClosureSystem<'static>) {
    let (t0, y0, h0) = STATES[0];
    let sys = system();
    let opts = options(&sys, kind, h0);
    let mut reused = Lsoda::new(sys, opts, t0, &y0).unwrap();
    let first = trace(&mut reused);
    assert_eq!(first.mused, Some(Method::Bdf), "{kind:?}: Robertson should end on BDF");
    assert!(first.counts[2] > 0, "{kind:?}: no Jacobian evaluated");
    for (k, &(t0, y0, h0)) in STATES.iter().enumerate() {
        reused.restart(t0, &y0, h0).unwrap();
        assert_eq!(reused.options().h0, h0);
        // Before any step, the restarted solver reads as a new one.
        assert_eq!((reused.t(), reused.y(), reused.tcur(), reused.tsw()), (t0, &y0[..], t0, t0));
        assert_eq!((reused.nst(), reused.nfe(), reused.nje(), reused.nqu(), reused.mused()), (0, 0, 0, 0, None));
        assert_eq!((reused.nqcur(), reused.mcur()), (1, Method::Adams));
        let got = trace(&mut reused);
        let sys = system();
        let opts = options(&sys, kind, h0);
        let want = trace(&mut Lsoda::new(sys, opts, t0, &y0).unwrap());
        assert_eq!(got, want, "{kind:?}: state {k}");
    }
}

const MATRIX_KINDS: [JacobianKind; 4] = [
    JacobianKind::DenseUser,
    JacobianKind::DenseFd,
    JacobianKind::BandedUser { ml: 2, mu: 2 },
    JacobianKind::BandedFd { ml: 2, mu: 2 },
];

#[test]
fn restart_matches_new_dense_and_banded() {
    for kind in MATRIX_KINDS {
        check_restart(kind, robertson);
    }
}

#[cfg(feature = "sparse")]
#[test]
fn restart_matches_new_sparse() {
    check_restart(JacobianKind::SparseUser, || robertson_sparse(true));
    check_restart(JacobianKind::SparseFd, || robertson_sparse(false));
}

#[cfg(feature = "sparse")]
#[test]
fn restart_keeps_a_custom_linear_solver() {
    use rsoda::{OdeSystem, SparseLu};
    let solver = |sys: &ClosureSystem<'_>| Box::new(SparseLu::new(sys.sparsity_pattern().unwrap().clone()));
    let (t0, y0, h0) = STATES[0];
    let sys = robertson_sparse(false);
    let (opts, wm) = (options(&sys, JacobianKind::DenseFd, h0), solver(&sys));
    let mut reused = Lsoda::with_linear_solver(sys, opts, t0, &y0, wm).unwrap();
    trace(&mut reused);
    for &(t0, y0, h0) in &STATES[1..] {
        reused.restart(t0, &y0, h0).unwrap();
        let got = trace(&mut reused);
        let sys = robertson_sparse(false);
        let (opts, wm) = (options(&sys, JacobianKind::DenseFd, h0), solver(&sys));
        let want = trace(&mut Lsoda::with_linear_solver(sys, opts, t0, &y0, wm).unwrap());
        assert_eq!(got, want, "state at t0 = {t0}");
    }
}

/// A restart after a failed call, and after a change of the Jacobian kind
/// to another storage, gives the run of a new solver with the options in
/// effect.
#[test]
fn restart_after_an_error_and_a_jacobian_change() {
    let (t0, y0, h0) = STATES[0];
    let sys = robertson();
    let opts = options(&sys, JacobianKind::DenseUser, h0);
    let mut reused = Lsoda::new(sys, opts, t0, &y0).unwrap();
    reused.integrate(40.0).unwrap();
    reused.set_mxstep(5);
    assert!(matches!(reused.integrate(4.0e5), Err(LsodaError::MaxStepsExceeded { .. })));
    reused.set_mxstep(0);
    reused.set_jacobian(JacobianKind::BandedFd { ml: 2, mu: 2 }).unwrap();
    for &(t0, y0, h0) in &STATES {
        reused.restart(t0, &y0, h0).unwrap();
        let got = trace(&mut reused);
        let sys = robertson();
        let want = trace(&mut Lsoda::new(sys, reused.options().clone(), t0, &y0).unwrap());
        assert_eq!(got, want, "state at t0 = {t0}");
    }
}

#[test]
fn restart_rejects_bad_input_without_change() {
    let (t0, y0, h0) = STATES[0];
    let sys = robertson();
    let opts = options(&sys, JacobianKind::DenseUser, h0);
    let mut s = Lsoda::new(sys, opts, t0, &y0).unwrap();
    s.integrate(0.4).unwrap();
    let (t, y, nst) = (s.t(), s.y().to_vec(), s.nst());
    let illegal = |r: Result<(), LsodaError>| match r {
        Err(LsodaError::IllegalInput(e)) => e,
        other => panic!("expected IllegalInput, got {other:?}"),
    };
    assert_eq!(illegal(s.restart(1.0, &[1.0, 0.0], 0.0)), InputError::YLength { expected: 3, got: 2 });
    assert!(matches!(illegal(s.restart(1.0, &y0, f64::NAN)), InputError::InvalidH0(_)));
    assert_eq!((s.t(), s.y(), s.nst(), s.options().h0), (t, &y[..], nst, h0));
    // The solver carries on as if nothing happened.
    s.integrate(4.0).unwrap();
}

// ----- Warm start (Lsoda::restart_warm) ---------------------------------------

/// The largest weighted difference `|a - b| / (rtol |b| + atol)` between two
/// solutions, with the tolerances of [`options`].
fn weighted_diff(a: &[f64], b: &[f64]) -> f64 {
    let atol = [1e-6, 1e-10, 1e-6];
    a.iter().zip(b).zip(atol).map(|((a, b), atol)| (a - b).abs() / (1e-4 * b.abs() + atol)).fold(0.0, f64::max)
}

/// The global error allowed against a tight reference, in units of the
/// tolerance (`weighted_diff`): LSODA controls the local error, and a fresh
/// solve of these problems reaches about 2.1 (a warm one about 2.7).
const GLOBAL_ERROR: f64 = 5.0;

/// A warm restart of a solver that ended on BDF starts the new problem on
/// BDF at order 1, and its solution stays within tolerance of a fresh
/// solve: at every output time of [`trace`], both are within
/// [`GLOBAL_ERROR`] of a tight reference.
fn check_warm_start(kind: JacobianKind, system: impl Fn() -> ClosureSystem<'static>) {
    let (t0, y0, h0) = STATES[0];
    let sys = system();
    let opts = options(&sys, kind, h0);
    let mut warm = Lsoda::new(sys, opts, t0, &y0).unwrap();
    let switches = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = switches.clone();
    warm.on_method_switch(move |e| log.lock().unwrap().push(*e));
    warm.integrate(4.0e5).unwrap();
    for (k, &(t0, y0, h0)) in STATES.iter().enumerate() {
        assert_eq!(warm.mcur(), Method::Bdf, "{kind:?}: state {k}: the previous solve ends on BDF");
        assert!(warm.nqcur() > 1, "{kind:?}: state {k}: the previous solve ends above order 1");
        warm.restart_warm(t0, &y0, h0).unwrap();
        switches.lock().unwrap().clear();
        assert_eq!((warm.t(), warm.y(), warm.tcur(), warm.tsw()), (t0, &y0[..], t0, t0));
        assert_eq!((warm.nst(), warm.nfe(), warm.nje(), warm.nqu(), warm.mused()), (0, 0, 0, 0, None));
        assert_eq!((warm.nqcur(), warm.mcur()), (1, Method::Bdf), "{kind:?}: state {k}");
        // The first step is a BDF step of order 1 with a new iteration
        // matrix, and no method switch.
        warm.step(t0 + 4.0e5).unwrap();
        assert_eq!((warm.nst(), warm.nqu(), warm.mused()), (1, 1, Some(Method::Bdf)), "{kind:?}: state {k}");
        assert!(warm.nje() >= 1, "{kind:?}: state {k}");
        assert!(switches.lock().unwrap().is_empty(), "{kind:?}: state {k}");

        let sys = system();
        let mut fresh = Lsoda::new(sys, options(&system(), kind, h0), t0, &y0).unwrap();
        let tight = Tolerance { rtol: Tol::Scalar(1e-10), atol: Tol::PerComponent(vec![1e-12, 1e-16, 1e-12]) };
        let sys = system();
        let ropts = OptionsBuilder::new(3).jacobian(kind).tolerance(tight).mxstep(100_000).build(&sys).unwrap();
        let mut reference = Lsoda::new(sys, ropts, t0, &y0).unwrap();
        let mut tout = 0.4;
        for _ in 0..10 {
            for s in [&mut warm as &mut Lsoda<'_, _>, &mut fresh, &mut reference] {
                s.integrate(t0 + tout).unwrap();
            }
            let (w, f, r) = (warm.y(), fresh.y(), reference.y());
            let at = format!("{kind:?}: state {k}, t = {:e}", t0 + tout);
            assert!(weighted_diff(f, r) <= GLOBAL_ERROR, "{at}: fresh {f:?}, reference {r:?}");
            assert!(weighted_diff(w, r) <= GLOBAL_ERROR, "{at}: warm {w:?}, reference {r:?}");
            assert!(weighted_diff(w, f) <= 2.0 * GLOBAL_ERROR, "{at}: warm {w:?}, fresh {f:?}");
            tout *= 10.0;
        }
    }
}

#[test]
fn warm_start_stays_within_tolerance_dense_and_banded() {
    for kind in MATRIX_KINDS {
        check_warm_start(kind, robertson);
    }
}

#[cfg(feature = "sparse")]
#[test]
fn warm_start_stays_within_tolerance_sparse() {
    check_warm_start(JacobianKind::SparseUser, || robertson_sparse(true));
    check_warm_start(JacobianKind::SparseFd, || robertson_sparse(false));
}

/// On Adams, the warm start is the cold one, bit for bit; after a warm start
/// on BDF, `restart` starts on Adams again, as a new solver does.
#[test]
fn warm_start_on_adams_and_cold_restart_after_a_warm_one() {
    let (t0, y0, h0) = STATES[0];
    let sys = robertson();
    let opts = options(&sys, JacobianKind::DenseUser, h0);
    let mut reused = Lsoda::new(sys, opts, t0, &y0).unwrap();
    // Robertson switches to BDF at t = 6.0e-3.
    reused.integrate(1.0e-3).unwrap();
    assert_eq!(reused.mcur(), Method::Adams);
    for &(t0, y0, h0) in &STATES {
        let sys = robertson();
        let opts = options(&sys, JacobianKind::DenseUser, h0);
        let want = trace(&mut Lsoda::new(sys, opts, t0, &y0).unwrap());
        reused.restart(t0, &y0, h0).unwrap();
        reused.integrate(t0 + 1.0e-3).unwrap();
        assert_eq!(reused.mcur(), Method::Adams);
        reused.restart_warm(t0, &y0, h0).unwrap();
        assert_eq!(reused.mcur(), Method::Adams);
        let got = trace(&mut reused);
        assert_eq!(got.mused, Some(Method::Bdf));
        // Now on BDF: warm, then cold.
        reused.restart_warm(t0, &y0, h0).unwrap();
        assert_eq!(reused.mcur(), Method::Bdf);
        reused.integrate(t0 + 4.0).unwrap();
        reused.restart(t0, &y0, h0).unwrap();
        assert_eq!((reused.mcur(), reused.nqcur()), (Method::Adams, 1));
        let cold = trace(&mut reused);
        assert_eq!(cold, want, "cold restart after a warm one, t0 = {t0}");
        assert_eq!(got, want, "warm start on Adams, t0 = {t0}");
    }
}

#[test]
fn warm_start_rejects_bad_input_without_change() {
    let (t0, y0, h0) = STATES[0];
    let sys = robertson();
    let opts = options(&sys, JacobianKind::DenseUser, h0);
    let mut s = Lsoda::new(sys, opts, t0, &y0).unwrap();
    s.integrate(40.0).unwrap();
    let (t, y, nst, meth) = (s.t(), s.y().to_vec(), s.nst(), s.mcur());
    assert!(matches!(s.restart_warm(1.0, &[1.0, 0.0], 0.0), Err(LsodaError::IllegalInput(InputError::YLength { .. }))));
    assert!(matches!(s.restart_warm(1.0, &y0, f64::NAN), Err(LsodaError::IllegalInput(InputError::InvalidH0(_)))));
    assert_eq!((s.t(), s.y(), s.nst(), s.mcur(), s.options().h0), (t, &y[..], nst, meth, h0));
    s.integrate(400.0).unwrap();
}
