//! Linear-solver call counts and times (cargo feature `timing`), using only
//! the public API.
#![cfg(feature = "timing")]

use rsoda::{ClosureSystem, JacobianKind, Lsoda, OptionsBuilder, Tolerance};

/// The Robertson problem: stiff, so LSODA switches to BDF and uses the
/// linear solver.
fn robertson<'a>() -> ClosureSystem<'a> {
    ClosureSystem::new(|_t, y, f| {
        f[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        f[2] = 3e7 * y[1] * y[1];
        f[1] = -f[0] - f[2];
        Ok(())
    })
}

#[test]
fn counts_match_the_solver_counters() {
    for kind in [JacobianKind::DenseFd, JacobianKind::BandedFd { ml: 2, mu: 2 }] {
        let sys = robertson();
        let tol = Tolerance::per_component(vec![1e-4; 3], vec![1e-8, 1e-14, 1e-6]);
        let opts = OptionsBuilder::new(3).tolerance(tol).jacobian(kind).build(&sys).unwrap();
        let mut s = Lsoda::new(sys, opts, 0.0, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(s.timing(), rsoda::TimingStats::default());
        s.integrate(4e5).unwrap();
        let t = s.timing();
        assert!(s.nje() > 0, "{kind:?}: the BDF method was never used");
        assert_eq!(t.n_prepare, s.nje(), "{kind:?}");
        assert!(t.n_solve >= t.n_prepare, "{kind:?}: {t:?}");
        assert!(!t.t_prepare.is_zero() || !t.t_solve.is_zero(), "{kind:?}: {t:?}");
    }
}
