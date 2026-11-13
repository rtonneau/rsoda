//! The sparse Jacobian kinds (cargo feature `sparse`) against the banded and
//! dense kinds, using only the public API.
#![cfg(feature = "sparse")]

use std::sync::{Arc, Mutex};

use rsoda::{
    ClosureSystem, InputError, JacobianKind, LinearSolver, Lsoda, LsodaError, Method, OptionsBuilder, SparseLu,
    SparsityPattern, Tolerance,
};

/// The 2D Brusselator on an `m x m` grid of the unit square, with
/// reflecting (Neumann) boundaries, unknowns interleaved as
/// `(u, v)` per cell: `y[2*(i*m + j)] = u(x_i, y_j)`, `+1` for `v`.
///
/// `u' = 1 + u^2 v - 4.4 u + a*lap(u)`, `v' = 3.4 u - u^2 v + a*lap(v)`,
/// `a = 0.1 / dx^2` (diffusion makes it stiff). The Jacobian has bandwidth `2m` but only 6 nonzeros
/// per row.
struct Brusselator {
    m: usize,
    a: f64,
}

impl Brusselator {
    fn new(m: usize) -> Self {
        let dx = 1.0 / (m - 1) as f64;
        Self { m, a: 0.1 / (dx * dx) }
    }

    fn n(&self) -> usize {
        2 * self.m * self.m
    }

    /// The four neighbor cells of `(i, j)`, mirrored at the boundary.
    fn neighbors(&self, i: usize, j: usize) -> [usize; 4] {
        let m = self.m;
        let up = if i == 0 { 1 } else { i - 1 };
        let down = if i == m - 1 { m - 2 } else { i + 1 };
        let left = if j == 0 { 1 } else { j - 1 };
        let right = if j == m - 1 { m - 2 } else { j + 1 };
        [up * m + j, down * m + j, i * m + left, i * m + right]
    }

    fn rhs(&self, y: &[f64], f: &mut [f64]) {
        let m = self.m;
        for i in 0..m {
            for j in 0..m {
                let c = i * m + j;
                let (u, v) = (y[2 * c], y[2 * c + 1]);
                let (mut lu, mut lv) = (-4.0 * u, -4.0 * v);
                for nb in self.neighbors(i, j) {
                    lu += y[2 * nb];
                    lv += y[2 * nb + 1];
                }
                f[2 * c] = 1.0 + u * u * v - 4.4 * u + self.a * lu;
                f[2 * c + 1] = 3.4 * u - u * u * v + self.a * lv;
            }
        }
    }

    /// Calls `add(row, col, value)` for every Jacobian contribution
    /// (entries may repeat at mirrored boundaries; they add up).
    fn jac(&self, y: &[f64], mut add: impl FnMut(usize, usize, f64)) {
        let m = self.m;
        for i in 0..m {
            for j in 0..m {
                let c = i * m + j;
                let (u, v) = (y[2 * c], y[2 * c + 1]);
                let (ru, rv) = (2 * c, 2 * c + 1);
                add(ru, ru, 2.0 * u * v - 4.4 - 4.0 * self.a);
                add(ru, rv, u * u);
                add(rv, ru, 3.4 - 2.0 * u * v);
                add(rv, rv, -u * u - 4.0 * self.a);
                for nb in self.neighbors(i, j) {
                    add(ru, 2 * nb, self.a);
                    add(rv, 2 * nb + 1, self.a);
                }
            }
        }
    }

    fn pattern(&self) -> SparsityPattern {
        let y = vec![1.0; self.n()];
        let mut entries = Vec::new();
        self.jac(&y, |i, j, _| entries.push((i, j)));
        SparsityPattern::from_entries(self.n(), entries).unwrap()
    }

    fn y0(&self) -> Vec<f64> {
        let m = self.m;
        let dx = 1.0 / (m - 1) as f64;
        let mut y = vec![0.0; self.n()];
        for i in 0..m {
            for j in 0..m {
                let c = i * m + j;
                y[2 * c] = 0.5 + j as f64 * dx;
                y[2 * c + 1] = 1.0 + 5.0 * i as f64 * dx;
            }
        }
        y
    }
}

fn brusselator_system(m: usize, kind: JacobianKind) -> ClosureSystem<'static> {
    let b = Arc::new(Brusselator::new(m));
    let br = b.clone();
    let sys = ClosureSystem::new(move |_t, y: &[f64], f: &mut [f64]| {
        br.rhs(y, f);
        Ok(())
    });
    match kind {
        JacobianKind::BandedUser { ml, mu } => {
            let bj = b.clone();
            sys.with_jac(move |_t, y: &[f64], pd: &mut [f64]| {
                let nrowpd = ml + mu + 1;
                bj.jac(y, |i, j, v| pd[(i + mu - j) + j * nrowpd] += v);
                Ok(())
            })
        }
        JacobianKind::SparseUser => {
            let p = b.pattern();
            let (bj, pj) = (b.clone(), p.clone());
            sys.with_sparse_jac(p, move |_t, y: &[f64], values: &mut [f64]| {
                bj.jac(y, |i, j, v| values[pj.find(i, j).unwrap()] += v);
                Ok(())
            })
        }
        JacobianKind::SparseFd => sys.with_sparsity(b.pattern()),
        _ => sys,
    }
}

struct Run {
    y: Vec<f64>,
    nst: usize,
    nfe: usize,
    nje: usize,
    bdf: bool,
}

fn run_brusselator(m: usize, kind: JacobianKind, tout: f64) -> Run {
    let sys = brusselator_system(m, kind);
    let n = 2 * m * m;
    let opts = OptionsBuilder::new(n).jacobian(kind).tolerance(Tolerance::scalar(1e-8, 1e-10)).build(&sys).unwrap();
    let y0 = Brusselator::new(m).y0();
    let mut s = Lsoda::new(sys, opts, 0.0, &y0).unwrap();
    s.integrate(tout).unwrap();
    Run { y: s.y().to_vec(), nst: s.nst(), nfe: s.nfe(), nje: s.nje(), bdf: s.mused() == Some(Method::Bdf) }
}

fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
    a.iter().zip(b).fold(0.0_f64, |d, (x, y)| d.max((x - y).abs() / scale))
}

#[test]
fn brusselator_sparse_matches_banded() {
    let m = 16;
    let band = JacobianKind::BandedUser { ml: 2 * m, mu: 2 * m };
    let reference = run_brusselator(m, band, 10.0);
    assert!(reference.bdf, "the Brusselator should be integrated with BDF");
    for kind in [JacobianKind::SparseUser, JacobianKind::SparseFd] {
        let r = run_brusselator(m, kind, 10.0);
        let d = max_rel_diff(&r.y, &reference.y);
        eprintln!(
            "{kind:?}: max relative difference {d:e}, nst {} nje {} (banded nst {} nje {})",
            r.nst, r.nje, reference.nst, reference.nje
        );
        assert!(d <= 1e-6, "{kind:?}: max relative difference {d:e} (nst {} vs {})", r.nst, reference.nst);
        assert!(r.bdf && r.nje > 0, "{kind:?}");
    }
}

#[test]
fn brusselator_sparse_fd_uses_one_rhs_per_color() {
    let m = 8;
    let b = Brusselator::new(m);
    let ncolors = b.pattern().coloring().ncolors();
    assert!(ncolors < b.n() / 4, "{ncolors} colors for n = {}", b.n());

    // Count f evaluations made inside the linear solver.
    struct Counting {
        inner: SparseLu,
        nfe_in_prepare: Arc<Mutex<Vec<usize>>>,
    }
    impl LinearSolver for Counting {
        fn prepare(
            &mut self,
            kind: &JacobianKind,
            system: &mut dyn rsoda::OdeSystem,
            input: &rsoda::PrjaInput<'_>,
            y: &mut [f64],
            ftem: &mut [f64],
            nfe: &mut usize,
            nje: &mut usize,
        ) -> Result<rsoda::PrjaOutput, LsodaError> {
            assert_eq!(*kind, JacobianKind::Custom);
            let before = *nfe;
            let out = self.inner.prepare(kind, system, input, y, ftem, nfe, nje)?;
            self.nfe_in_prepare.lock().unwrap().push(*nfe - before);
            Ok(out)
        }
        fn solve(&mut self, h: f64, el0: f64, x: &mut [f64]) -> Result<(), rsoda::SingularMatrix> {
            self.inner.solve(h, el0, x)
        }
    }
    let log = Arc::new(Mutex::new(Vec::new()));
    let solver = Counting { inner: SparseLu::new(b.pattern()).with_user_jacobian(false), nfe_in_prepare: log.clone() };
    let sys = brusselator_system(m, JacobianKind::SparseFd);
    let opts = OptionsBuilder::new(b.n()).tolerance(Tolerance::scalar(1e-8, 1e-10)).build(&sys).unwrap();
    let mut s = Lsoda::with_linear_solver(sys, opts, 0.0, &b.y0(), Box::new(solver)).unwrap();
    s.integrate(10.0).unwrap();
    let log = log.lock().unwrap();
    assert_eq!(log.len(), s.nje());
    assert!(!log.is_empty() && log.iter().all(|&k| k == ncolors), "{log:?} vs {ncolors}");

    // The same run through JacobianKind::SparseFd is identical.
    let r = run_brusselator(m, JacobianKind::SparseFd, 10.0);
    assert_eq!((r.y.as_slice(), r.nst, r.nfe, r.nje), (s.y(), s.nst(), s.nfe(), s.nje()));
}

fn robertson<'a>(kind: JacobianKind) -> ClosureSystem<'a> {
    let sys = ClosureSystem::new(|_t, y: &[f64], f: &mut [f64]| {
        f[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        f[2] = 3e7 * y[1] * y[1];
        f[1] = -f[0] - f[2];
        Ok(())
    });
    let jac = |y: &[f64]| {
        [[-0.04, 1e4 * y[2], 1e4 * y[1]], [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]], [0.0, 6e7 * y[1], 0.0]]
    };
    let p = SparsityPattern::from_entries(3, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)]).unwrap();
    match kind {
        JacobianKind::DenseUser => sys.with_jac(move |_t, y: &[f64], pd: &mut [f64]| {
            let j = jac(y);
            for c in 0..3 {
                for r in 0..3 {
                    pd[r + 3 * c] = j[r][c];
                }
            }
            Ok(())
        }),
        JacobianKind::SparseUser => {
            let pj = p.clone();
            sys.with_sparse_jac(p, move |_t, y: &[f64], values: &mut [f64]| {
                let j = jac(y);
                for c in 0..3 {
                    for k in pj.col_ptr()[c]..pj.col_ptr()[c + 1] {
                        values[k] = j[pj.row_idx()[k]][c];
                    }
                }
                Ok(())
            })
        }
        _ => sys.with_sparsity(p),
    }
}

#[test]
fn robertson_sparse_switches_methods_and_matches_dense() {
    let tol = || Tolerance::scalar(1e-8, 1e-12);
    for kind in [JacobianKind::SparseUser, JacobianKind::SparseFd] {
        let mut dense = {
            let sys = robertson(JacobianKind::DenseUser);
            let o = OptionsBuilder::new(3).jacobian(JacobianKind::DenseUser).tolerance(tol()).build(&sys).unwrap();
            Lsoda::new(sys, o, 0.0, &[1.0, 0.0, 0.0]).unwrap()
        };
        let sys = robertson(kind);
        let o = OptionsBuilder::new(3).jacobian(kind).tolerance(tol()).build(&sys).unwrap();
        let mut s = Lsoda::new(sys, o, 0.0, &[1.0, 0.0, 0.0]).unwrap();
        let switches = Arc::new(Mutex::new(Vec::new()));
        let sw = switches.clone();
        s.on_method_switch(move |e| sw.lock().unwrap().push(e.method));
        for k in 0..6 {
            let tout = 0.4 * 10f64.powi(k);
            s.integrate(tout).unwrap();
            dense.integrate(tout).unwrap();
            let d = max_rel_diff(s.y(), dense.y());
            assert!(d <= 1e-6, "{kind:?} t = {tout}: {d:e}");
        }
        assert_eq!(switches.lock().unwrap().first(), Some(&Method::Bdf), "{kind:?}: Adams to BDF");
        assert_eq!(s.mcur(), Method::Bdf);
    }
}

#[test]
fn sparse_kinds_validate_the_system() {
    let illegal = |sys: &ClosureSystem<'_>, kind: JacobianKind| match OptionsBuilder::new(3).jacobian(kind).build(sys) {
        Err(LsodaError::IllegalInput(e)) => e,
        other => panic!("expected IllegalInput, got {other:?}"),
    };
    let plain = ClosureSystem::new(|_t, _y: &[f64], _f: &mut [f64]| Ok(()));
    assert_eq!(illegal(&plain, JacobianKind::SparseFd), InputError::MissingSparsityPattern);
    let fd_only = robertson(JacobianKind::SparseFd);
    assert_eq!(illegal(&fd_only, JacobianKind::SparseUser), InputError::MissingJacobian);
    let wrong =
        ClosureSystem::new(|_t, _y: &[f64], _f: &mut [f64]| Ok(())).with_sparsity(SparsityPattern::dense(4).unwrap());
    assert_eq!(illegal(&wrong, JacobianKind::SparseFd), InputError::SparsityPatternSize { expected: 3, got: 4 });
    assert!(OptionsBuilder::new(3).jacobian(JacobianKind::SparseFd).build(&fd_only).is_ok());
}
