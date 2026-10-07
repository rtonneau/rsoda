# Ticket 08: golden-harness-docs

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] A Fortran driver under `fortran/` runs the Reference implementation on the same problems as the analytic tests and writes Golden files; a script builds and runs it.
- [ ] A Rust integration test compares counters and Method-switch sequence exactly and state values within 1e-12 relative; it skips with a clear message when Golden files are absent.
- [ ] README, NOTICE (crediting ODEPACK as public domain and the fork), and a rustdoc example on `Lsoda` are added.
- [ ] README states that gfortran must be installed to regenerate Golden files.

**Files to Touch:**
- `fortran/golden_driver.f90`
- `fortran/generate.ps1`
- `tests/golden.rs`
- `README.md`
- `NOTICE`
- `src/lsoda.rs`

**Verification Step:**

Run:
```bash
cargo test --test golden && cargo test --doc
```

Expected:
Golden test reports skipped (no Golden files) or passes if present; doc example passes; exit code 0.

**Notes:**

gfortran is not installed on this machine, so the Fortran driver cannot be built or run here and no Golden files are committed. Say so in the ticket's commit log, and leave generation to the user.
