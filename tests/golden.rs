//! Golden-file comparison against the Reference implementation (Fortran
//! ODEPACK `DLSODA`).
//!
//! The Golden files live in `tests/golden/<case>.txt` and are written by the
//! Fortran driver in `fortran/` (see `fortran/generate.ps1` and the README).
//! None is committed: when no Golden file exists, this test prints a message
//! and returns, so it passes on a machine without them.
//!
//! File format (plain text, `#` starts a comment line). One row per output
//! point, whitespace separated:
//!
//! ```text
//! tout nst nfe nje nq meth y(1) ... y(neq)
//! ```
//!
//! `nst`, `nfe`, `nje`, `nq` (order to be attempted next) and `meth`
//! (1 = Adams, 2 = BDF, method to be attempted next) are compared exactly;
//! the sequence of `meth` over the rows is the Method-switch sequence. The
//! state `y` is compared within 1e-12 relative. The rows come from one
//! `ITASK = 1` call per `tout`.
//!
//! The problems, tolerances (`rtol = 1e-8`, `atol = 1e-10`, scalar) and
//! Jacobian kinds below must match `fortran/golden_driver.f90`.

use rsoda::{ClosureSystem, JacobianKind, Lsoda, Method, OptionsBuilder, Tolerance};
use std::path::PathBuf;

const RTOL: f64 = 1e-8;
const ATOL: f64 = 1e-10;
const REL_TOL: f64 = 1e-12;

fn decay<'a>(user: bool) -> ClosureSystem<'a> {
    let sys = ClosureSystem::new(|_t, y, f| {
        f[0] = -y[0];
        Ok(())
    });
    if !user {
        return sys;
    }
    sys.with_jac(|_t, _y, pd| {
        pd[0] = -1.0;
        Ok(())
    })
}

fn oscillator<'a>(user: bool) -> ClosureSystem<'a> {
    let sys = ClosureSystem::new(|_t, y, f| {
        f[0] = y[1];
        f[1] = -y[0];
        Ok(())
    });
    if !user {
        return sys;
    }
    sys.with_jac(|_t, _y, pd| {
        pd[0] = 0.0;
        pd[1] = -1.0;
        pd[2] = 1.0;
        pd[3] = 0.0;
        Ok(())
    })
}

fn van_der_pol<'a>(user: bool) -> ClosureSystem<'a> {
    let mu = 1000.0;
    let sys = ClosureSystem::new(move |_t, y, f| {
        f[0] = y[1];
        f[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
        Ok(())
    });
    if !user {
        return sys;
    }
    sys.with_jac(move |_t, y, pd| {
        pd[0] = 0.0;
        pd[1] = -2.0 * mu * y[0] * y[1] - 1.0;
        pd[2] = 1.0;
        pd[3] = mu * (1.0 - y[0] * y[0]);
        Ok(())
    })
}

fn robertson<'a>(user: bool) -> ClosureSystem<'a> {
    let sys = ClosureSystem::new(|_t, y, f| {
        f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
        f[2] = 3.0e7 * y[1] * y[1];
        f[1] = -f[0] - f[2];
        Ok(())
    });
    if !user {
        return sys;
    }
    sys.with_jac(|_t, y, pd| {
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

struct Case {
    name: String,
    neq: usize,
    y0: &'static [f64],
    user: bool,
    system: fn(bool) -> ClosureSystem<'static>,
}

fn cases() -> Vec<Case> {
    type Def = (&'static str, usize, &'static [f64], fn(bool) -> ClosureSystem<'static>);
    let defs: [Def; 4] = [
        ("decay", 1, &[1.0], decay),
        ("osc", 2, &[1.0, 0.0], oscillator),
        ("vdp", 2, &[2.0, 0.0], van_der_pol),
        ("rober", 3, &[1.0, 0.0, 0.0], robertson),
    ];
    let mut v = Vec::new();
    for (base, neq, y0, system) in defs {
        // jt = 1 (user dense Jacobian) and jt = 2 (finite-difference dense).
        for (jt, user) in [(1, true), (2, false)] {
            v.push(Case { name: format!("{base}_jt{jt}"), neq, y0, user, system });
        }
    }
    v
}

struct Row {
    tout: f64,
    nst: usize,
    nfe: usize,
    nje: usize,
    nq: usize,
    meth: usize,
    y: Vec<f64>,
}

fn parse(text: &str, neq: usize) -> Vec<Row> {
    let mut rows = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tok: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(tok.len(), 6 + neq, "line {}: expected {} fields", n + 1, 6 + neq);
        let int = |s: &str| s.parse::<usize>().unwrap_or_else(|e| panic!("line {}: {e}", n + 1));
        let real = |s: &str| s.parse::<f64>().unwrap_or_else(|e| panic!("line {}: {e}", n + 1));
        rows.push(Row {
            tout: real(tok[0]),
            nst: int(tok[1]),
            nfe: int(tok[2]),
            nje: int(tok[3]),
            nq: int(tok[4]),
            meth: int(tok[5]),
            y: tok[6..].iter().map(|s| real(s)).collect(),
        });
    }
    rows
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= REL_TOL * a.abs().max(b.abs())
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden").join(format!("{name}.txt"))
}

#[test]
fn matches_reference_golden_files() {
    let all = cases();
    let present: Vec<&Case> = all.iter().filter(|c| golden_path(&c.name).is_file()).collect();
    if present.is_empty() {
        println!(
            "SKIPPED: no Golden files in tests/golden/. Generate them with \
             fortran/generate.ps1 (needs gfortran and the ODEPACK source, see README)."
        );
        return;
    }
    for case in present {
        let text = std::fs::read_to_string(golden_path(&case.name)).unwrap();
        let rows = parse(&text, case.neq);
        assert!(!rows.is_empty(), "{}: Golden file has no rows", case.name);

        let sys = (case.system)(case.user);
        let jac = if case.user { JacobianKind::DenseUser } else { JacobianKind::DenseFd };
        let opts = OptionsBuilder::new(case.neq)
            .tolerance(Tolerance::scalar(RTOL, ATOL))
            .jacobian(jac)
            .build(&sys)
            .unwrap();
        let mut s = Lsoda::new(sys, opts, 0.0, case.y0).unwrap();

        for (k, row) in rows.iter().enumerate() {
            let ctx = format!("{} row {} (tout = {:e})", case.name, k + 1, row.tout);
            s.integrate(row.tout).unwrap_or_else(|e| panic!("{ctx}: {e:?}"));
            assert_eq!(s.nst(), row.nst, "{ctx}: nst");
            assert_eq!(s.nfe(), row.nfe, "{ctx}: nfe");
            assert_eq!(s.nje(), row.nje, "{ctx}: nje");
            assert_eq!(s.nqcur(), row.nq, "{ctx}: nq");
            let meth = match s.mcur() {
                Method::Adams => 1,
                Method::Bdf => 2,
            };
            assert_eq!(meth, row.meth, "{ctx}: method");
            for (i, (&a, &b)) in s.y().iter().zip(&row.y).enumerate() {
                assert!(close(a, b), "{ctx}: y[{i}] rust {a:e} vs golden {b:e}");
            }
        }
    }
}
