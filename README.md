# rsoda

A Rust port of LSODA, the ODE solver of ODEPACK that switches automatically
between non-stiff (Adams) and stiff (BDF) methods. By default the crate has
no dependencies (the optional `sparse` feature adds `faer`). It follows the thread-safe Fortran fork of ODEPACK (see
[NOTICE](NOTICE)); each `Lsoda` solver owns its state, so several can run in
parallel threads.

```rust
use rsoda::{ClosureSystem, Lsoda, OptionsBuilder, Tolerance};

let system = ClosureSystem::new(|_t, y, f| {
    f[0] = -y[0];
    Ok(())
});
let options = OptionsBuilder::new(1)
    .tolerance(Tolerance::scalar(1e-10, 1e-12))
    .build(&system)
    .unwrap();
let mut solver = Lsoda::new(system, options, 0.0, &[1.0]).unwrap();
solver.integrate(1.0).unwrap();
println!("y(1) = {}", solver.y()[0]);
```

DLSODA's `ITASK` values 1 to 5 map to `integrate`, `step`, `advance_past`,
`integrate_tcrit` and `step_tcrit`. Jacobians can be dense or banded, user
supplied or finite-difference (`JacobianKind`).
`dky` is DINTDY: it interpolates the solution or its derivatives inside the
last step without stepping. `nordsieck` gives read-only access to the history
array and its step data, for callers that build their own dense output. See
the rustdoc of `Lsoda`.

## Linear solvers and sparse Jacobians

The stiff corrector solves with the iteration matrix `P = I - h*el0*J`. That
matrix is owned by a `LinearSolver` (DPRJA's `prepare`, DSOLSY's `solve`):

* `MatrixSolver` is the built-in dense and banded LU (LINPACK), used by the
  dense and banded `JacobianKind`s. Its results are those of DLSODA.
* With the `sparse` cargo feature, `JacobianKind::SparseUser` and
  `JacobianKind::SparseFd` use `SparseLu`, a sparse LU with partial pivoting
  from [faer](https://crates.io/crates/faer). The system declares its fixed
  `SparsityPattern` (`OdeSystem::sparsity_pattern`); the fill-reducing column
  ordering is computed once and `P` is refactored numerically at each
  Jacobian update. `SparseUser` takes the values from `OdeSystem::jac_sparse`
  (in pattern order); `SparseFd` estimates them by finite differences, with
  one `f` evaluation per group of a Curtis-Powell-Reid `ColumnColoring`.
  Without the feature, the sparse kinds are rejected as illegal input.
* `Lsoda::with_linear_solver` plugs in any user `LinearSolver`
  (`JacobianKind::Custom`), e.g. an iterative or block-diagonal solver. The
  helpers of `rsoda::sparse` (pattern, coloring, finite-difference assembly,
  norm) are public for such solvers.

```toml
rsoda = { git = "https://github.com/rtonneau/rsoda", features = ["sparse"] }
```

```rust,ignore
use rsoda::{ClosureSystem, JacobianKind, Lsoda, OptionsBuilder, SparsityPattern};

// y0' = -y0 + y1, y1' = -1000 y1: J has entries (0,0), (0,1), (1,1).
let pattern = SparsityPattern::from_entries(2, [(0, 0), (0, 1), (1, 1)]).unwrap();
let p = pattern.clone();
let system = ClosureSystem::new(|_t, y, f| {
    f[0] = -y[0] + y[1];
    f[1] = -1000.0 * y[1];
    Ok(())
})
.with_sparse_jac(pattern, move |_t, _y, values| {
    values[p.find(0, 0).unwrap()] = -1.0;
    values[p.find(0, 1).unwrap()] = 1.0;
    values[p.find(1, 1).unwrap()] = -1000.0;
    Ok(())
});
let options = OptionsBuilder::new(2).jacobian(JacobianKind::SparseUser).build(&system).unwrap();
let mut solver = Lsoda::new(system, options, 0.0, &[1.0, 1.0]).unwrap();
solver.integrate(1.0).unwrap();
```

## Profiling

The `timing` cargo feature counts and times the linear solver's calls:
`Lsoda::timing()` returns a `TimingStats` with the number and total wall time
of `LinearSolver::prepare` calls (Jacobian evaluation, assembly of
`P = I - h*el0*J` and factorization) and of `LinearSolver::solve` calls. Without
the feature no timer is taken, and the results are the same either way.

## Tests

```
cargo test
cargo test --features sparse
cargo test --features timing
```

* `tests/analytic.rs` checks the solver against analytic or high-accuracy
  solutions.
* `tests/golden.rs` compares counters (steps, `f` evaluations, Jacobian
  evaluations, order), the Adams/BDF method sequence (exactly) and the state
  (within 1e-12 relative) with the Reference implementation, the Fortran fork
  of ODEPACK. It reads Golden files from `tests/golden/`. **No Golden files
  are committed**: without them the test prints a "SKIPPED" message and
  passes.
* `tests/sparse.rs` (feature `sparse`) checks the sparse kinds against the
  banded and dense ones on a 2D Brusselator grid and the Robertson problem.
* `tests/timing.rs` (feature `timing`) checks the call counts against the
  solver's counters.

## Regenerating the Golden files

Golden files come from the Fortran driver in `fortran/`, which needs
**gfortran to be installed** (it is not needed for anything else), plus a
checkout of the Fortran fork of ODEPACK
(<https://github.com/Nicholaswogan/odepack>):

```powershell
$env:ODEPACK_SRC = "<checkout>/src"   # or pass -OdepackSrc
./fortran/generate.ps1                # writes tests/golden/*.txt
cargo test --test golden
```

The driver and script have not been built or run yet (they were written on a
machine without gfortran); expect to fix small problems on the first run.
The file format is described at the top of `tests/golden.rs`.

## License

See [NOTICE](NOTICE): ODEPACK is in the public domain.
