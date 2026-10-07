# rsoda

A Rust port of LSODA, the ODE solver of ODEPACK that switches automatically
between non-stiff (Adams) and stiff (BDF) methods. The crate has no
dependencies. It follows the thread-safe Fortran fork of ODEPACK (see
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
supplied or finite-difference (`JacobianKind`). See the rustdoc of `Lsoda`.

## Tests

```
cargo test
```

* `tests/analytic.rs` checks the solver against analytic or high-accuracy
  solutions.
* `tests/golden.rs` compares counters (steps, `f` evaluations, Jacobian
  evaluations, order), the Adams/BDF method sequence (exactly) and the state
  (within 1e-12 relative) with the Reference implementation, the Fortran fork
  of ODEPACK. It reads Golden files from `tests/golden/`. **No Golden files
  are committed**: without them the test prints a "SKIPPED" message and
  passes.

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
