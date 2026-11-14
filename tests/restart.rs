//! `Lsoda::restart` against a solver created afresh by `Lsoda::new` with the
//! same options: the integrations must agree bitwise (solution, step and
//! evaluation counts), whatever the solver did before the restart.

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
