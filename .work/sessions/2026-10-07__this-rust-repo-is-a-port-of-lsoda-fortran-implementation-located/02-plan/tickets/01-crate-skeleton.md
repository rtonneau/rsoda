# Ticket 01: crate-skeleton

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `src/lib.rs` declares modules for errors, options, system, and the solver, replacing the placeholder `add` function.
- [ ] `LsodaError` has one variant per negative istate of DLSODA, carrying the data the Fortran reports.
- [ ] `Tolerance` supports scalar and per-component rtol/atol; `JacobianKind` covers dense/banded x user/finite-difference.
- [ ] `OdeSystem` trait has `rhs` and `jac` methods returning `Result`, and a closure adapter implements it.
- [ ] The builder applies the Fortran defaults and rejects the same illegal inputs as `lsoda_initialize`/DLSODA (istate=-3 cases).

**Files to Touch:**
- `Cargo.toml`
- `src/lib.rs`
- `src/error.rs`
- `src/options.rs`
- `src/system.rs`

**Verification Step:**

Run:
```bash
cargo test options
```

Expected:
Builder validation tests pass (defaults, and each illegal input rejected); exit code 0.

**Notes:**

Read `odepack_mod.f90` `lsoda_initialize` and the input checks in `DLSODA` in `odepack.f`. Machine constants (unit roundoff) belong in a small `consts` module. Error variants should carry data such as `tolsf` and the offending component index.
