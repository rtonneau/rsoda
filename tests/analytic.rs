//! Integration tests against analytic or high-accuracy reference solutions,
//! using only the public API.

use rsoda::{ClosureSystem, JacobianKind, Lsoda, LsodaError, Method, OptionsBuilder, Tolerance};

type R = Result<(), LsodaError>;

fn decay<'a>() -> ClosureSystem<'a> {
    ClosureSystem::new(|_t, y, f| {
        f[0] = -y[0];
        Ok(())
    })
    .with_jac(|_t, _y, pd| {
        pd[0] = -1.0;
        Ok(())
    })
}

fn oscillator<'a>() -> ClosureSystem<'a> {
    ClosureSystem::new(|_t, y, f| {
        f[0] = y[1];
        f[1] = -y[0];
        Ok(())
    })
    .with_jac(|_t, _y, pd| {
        // pd[i + j*2] = df_i/dy_j
        pd[0] = 0.0;
        pd[1] = -1.0;
        pd[2] = 1.0;
        pd[3] = 0.0;
        Ok(())
    })
}

fn solver<'a>(
    sys: ClosureSystem<'a>,
    neq: usize,
    tol: Tolerance,
    jac: JacobianKind,
    h0: f64,
    y0: &[f64],
) -> Lsoda<'a, ClosureSystem<'a>> {
    let opts = OptionsBuilder::new(neq).tolerance(tol).jacobian(jac).h0(h0).build(&sys).unwrap();
    Lsoda::new(sys, opts, 0.0, y0).unwrap()
}

// ----- Exponential decay, itask 1 and dense Jacobian kinds -------------------

#[test]
fn decay_integrate_fd_and_user_jacobian() -> R {
    for jac in [JacobianKind::DenseFd, JacobianKind::DenseUser] {
        let mut s = solver(decay(), 1, Tolerance::scalar(1e-10, 1e-12), jac, 0.0, &[1.0]);
        for k in 1..=10 {
            let t = k as f64;
            assert_eq!(s.integrate(t)?, t);
            let exact = (-t).exp();
            assert!((s.y()[0] - exact).abs() < 1e-8 * exact.max(1e-3), "{jac:?} t={t}");
        }
    }
    Ok(())
}

// ----- Harmonic oscillator, itask 1 to 5 ------------------------------------

fn check_osc(s: &Lsoda<'_, ClosureSystem<'_>>, t: f64, tol: f64) {
    assert!((s.y()[0] - t.cos()).abs() < tol, "y0 at t={t}: {}", s.y()[0]);
    assert!((s.y()[1] + t.sin()).abs() < tol, "y1 at t={t}: {}", s.y()[1]);
}

fn osc(jac: JacobianKind) -> Lsoda<'static, ClosureSystem<'static>> {
    solver(oscillator(), 2, Tolerance::scalar(1e-10, 1e-12), jac, 1e-3, &[1.0, 0.0])
}

#[test]
fn oscillator_itask1_integrate() -> R {
    let mut s = osc(JacobianKind::DenseUser);
    for k in 1..=8 {
        let t = k as f64 * 0.75;
        s.integrate(t)?;
        check_osc(&s, t, 1e-7);
    }
    Ok(())
}

#[test]
fn oscillator_itask2_step() -> R {
    let mut s = osc(JacobianKind::DenseFd);
    let mut last = 0.0;
    for _ in 0..50 {
        let t = s.step(10.0)?;
        assert!(t > last);
        last = t;
        // y() holds the solution at the returned mesh point.
        check_osc(&s, t, 1e-6);
    }
    Ok(())
}

#[test]
fn oscillator_itask3_advance_past() -> R {
    let mut s = osc(JacobianKind::DenseUser);
    let t = s.advance_past(5.0)?;
    assert!(t >= 5.0);
    check_osc(&s, t, 1e-6);
    // Already at or beyond tout: no further step.
    let nst = s.nst();
    assert_eq!(s.advance_past(t)?, t);
    assert_eq!(s.nst(), nst);
    Ok(())
}

#[test]
fn oscillator_itask4_integrate_tcrit() -> R {
    let mut s = osc(JacobianKind::DenseUser);
    let tcrit = 6.0;
    let t = s.integrate_tcrit(3.0, tcrit)?;
    assert_eq!(t, 3.0);
    check_osc(&s, 3.0, 1e-7);
    let t = s.integrate_tcrit(tcrit, tcrit)?;
    assert_eq!(t, tcrit);
    check_osc(&s, tcrit, 1e-7);
    assert!(s.tcur() <= tcrit * (1.0 + 1e-12));
    Ok(())
}

#[test]
fn oscillator_itask5_step_tcrit() -> R {
    let mut s = osc(JacobianKind::DenseFd);
    let tcrit = 4.0;
    let mut t = 0.0;
    let mut count = 0;
    while t < tcrit {
        let next = s.step_tcrit(tcrit, tcrit)?;
        assert!(next > t && next <= tcrit);
        t = next;
        count += 1;
        assert!(count < 10_000);
    }
    assert_eq!(t, tcrit);
    check_osc(&s, tcrit, 1e-6);
    // Already at tcrit: no step.
    let nst = s.nst();
    assert_eq!(s.step_tcrit(tcrit, tcrit)?, tcrit);
    assert_eq!(s.nst(), nst);
    Ok(())
}

// ----- Banded Jacobians -----------------------------------------------------

const N: usize = 6;

fn chain_rhs(y: &[f64], f: &mut [f64]) {
    for i in 0..N {
        let l = if i > 0 { y[i - 1] } else { 0.0 };
        let r = if i + 1 < N { y[i + 1] } else { 0.0 };
        f[i] = l - 2.0 * y[i] + r;
    }
}

fn chain<'a>(with_jac: bool) -> ClosureSystem<'a> {
    let sys = ClosureSystem::new(|_t, y, f| {
        chain_rhs(y, f);
        Ok(())
    });
    if !with_jac {
        return sys;
    }
    sys.with_jac(|_t, _y, pd| {
        // ml = mu = 1, nrowpd = 3: pd[(i - j + mu) + j*nrowpd] = df_i/dy_j
        pd.fill(0.0);
        for j in 0..N {
            pd[1 + j * 3] = -2.0;
            if j > 0 {
                pd[j * 3] = 1.0; // i = j - 1
            }
            if j + 1 < N {
                pd[2 + j * 3] = 1.0; // i = j + 1
            }
        }
        Ok(())
    })
}

#[test]
fn banded_kinds_match_dense() -> R {
    let y0: Vec<f64> = (0..N).map(|i| 1.0 + i as f64).collect();
    let tol = Tolerance::scalar(1e-10, 1e-12);
    let mut reference = solver(chain(false), N, tol.clone(), JacobianKind::DenseFd, 1e-3, &y0);
    reference.integrate(2.0)?;
    for (jac, with_jac) in [
        (JacobianKind::BandedFd { ml: 1, mu: 1 }, false),
        (JacobianKind::BandedUser { ml: 1, mu: 1 }, true),
    ] {
        let mut s = solver(chain(with_jac), N, tol.clone(), jac, 1e-3, &y0);
        s.integrate(2.0)?;
        for i in 0..N {
            assert!((s.y()[i] - reference.y()[i]).abs() < 1e-7, "{jac:?} component {i}");
        }
    }
    Ok(())
}

// ----- Van der Pol, stiff ---------------------------------------------------

fn van_der_pol<'a>(mu: f64) -> ClosureSystem<'a> {
    ClosureSystem::new(move |_t, y, f| {
        f[0] = y[1];
        f[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
        Ok(())
    })
    .with_jac(move |_t, y, pd| {
        pd[0] = 0.0;
        pd[1] = -2.0 * mu * y[0] * y[1] - 1.0;
        pd[2] = 1.0;
        pd[3] = mu * (1.0 - y[0] * y[0]);
        Ok(())
    })
}

#[test]
fn van_der_pol_stiff_switches_to_bdf_and_is_consistent() -> R {
    let mu = 1000.0;
    let y0 = [2.0, 0.0];
    let t8 = Tolerance::scalar(1e-8, 1e-8);
    let mut fd = solver(van_der_pol(mu), 2, t8.clone(), JacobianKind::DenseFd, 1e-6, &y0);
    let mut user = solver(van_der_pol(mu), 2, t8, JacobianKind::DenseUser, 1e-6, &y0);
    let mut tight = solver(
        van_der_pol(mu),
        2,
        Tolerance::scalar(1e-11, 1e-11),
        JacobianKind::DenseUser,
        1e-6,
        &y0,
    );
    for t in [100.0, 500.0, 1000.0, 2000.0] {
        fd.integrate(t)?;
        user.integrate(t)?;
        tight.integrate(t)?;
        assert!((user.y()[0] - tight.y()[0]).abs() < 1e-3, "t={t}: {:?} {:?}", user.y(), tight.y());
        assert!((fd.y()[0] - tight.y()[0]).abs() < 1e-3, "t={t}");
        assert!(user.y()[0].abs() < 2.1);
    }
    assert_eq!(user.mused(), Some(Method::Bdf));
    // A stiff solver must need far fewer steps than the problem scale mu.
    assert!(user.nst() < 5000);
    Ok(())
}

// ----- Robertson ------------------------------------------------------------

fn robertson<'a>() -> ClosureSystem<'a> {
    ClosureSystem::new(|_t, y, f| {
        f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
        f[2] = 3.0e7 * y[1] * y[1];
        f[1] = -f[0] - f[2];
        Ok(())
    })
    .with_jac(|_t, y, pd| {
        pd[0] = -0.04;
        pd[1] = 0.04;
        pd[2] = 0.0;
        pd[3] = 1.0e4 * y[2];
        pd[4] = -1.0e4 * y[2] - 6.0e7 * y[1];
        pd[5] = 6.0e7 * y[1];
        pd[6] = 1.0e4 * y[1];
        pd[7] = -1.0e4 * y[1];
        pd[8] = 0.0;
        Ok(())
    })
}

#[test]
fn robertson_reference_values() -> R {
    // Reference (Hairer/Wanner, high accuracy):
    // t = 0.4: 0.98517, 3.3864e-5, 0.014796
    // t = 40 : 0.7158270687, 9.185534764e-6, 0.2841637457
    let tol = Tolerance::per_component(vec![1e-8; 3], vec![1e-12, 1e-14, 1e-12]);
    for jac in [JacobianKind::DenseFd, JacobianKind::DenseUser] {
        let mut s = solver(robertson(), 3, tol.clone(), jac, 1e-8, &[1.0, 0.0, 0.0]);
        s.integrate(0.4)?;
        assert!((s.y()[0] - 0.98517).abs() < 1e-4, "{jac:?}");
        assert!((s.y()[1] - 3.3864e-5).abs() < 1e-7, "{jac:?}");
        s.integrate(40.0)?;
        let y = s.y();
        assert!((y[0] - 0.7158270687).abs() < 1e-5, "{jac:?}: {y:?}");
        assert!((y[1] - 9.185534764e-6).abs() < 1e-8, "{jac:?}: {y:?}");
        assert!((y[2] - 0.2841637457).abs() < 1e-5, "{jac:?}: {y:?}");
        assert!((y[0] + y[1] + y[2] - 1.0).abs() < 1e-6);
        s.integrate(4.0e5)?;
        assert!((s.y().iter().sum::<f64>() - 1.0).abs() < 1e-5);
        assert_eq!(s.mused(), Some(Method::Bdf));
    }
    Ok(())
}

// ----- Thread independence --------------------------------------------------

fn run_decay(rate: f64) -> Result<Vec<f64>, LsodaError> {
    let sys = ClosureSystem::new(move |_t, y, f| {
        f[0] = -rate * y[0];
        Ok(())
    });
    let opts = OptionsBuilder::new(1)
        .tolerance(Tolerance::scalar(1e-10, 1e-12))
        .h0(1e-4)
        .build(&sys)?;
    let mut s = Lsoda::new(sys, opts, 0.0, &[1.0])?;
    let mut out = Vec::new();
    for k in 1..=20 {
        s.integrate(k as f64 * 0.1)?;
        out.push(s.y()[0]);
    }
    Ok(out)
}

#[test]
fn two_solvers_in_two_threads_are_independent() -> R {
    let a = std::thread::spawn(|| run_decay(1.0));
    let b = std::thread::spawn(|| run_decay(5.0));
    let ya = a.join().unwrap()?;
    let yb = b.join().unwrap()?;
    for k in 0..20 {
        let t = (k + 1) as f64 * 0.1;
        assert!((ya[k] - (-t).exp()).abs() < 1e-8);
        assert!((yb[k] - (-5.0 * t).exp()).abs() < 1e-8);
    }
    // Results equal those of a sequential run.
    assert_eq!(ya, run_decay(1.0)?);
    assert_eq!(yb, run_decay(5.0)?);
    Ok(())
}
