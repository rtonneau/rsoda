# Ticket 07: analytic-tests

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] Tests for exponential decay, harmonic oscillator, Van der Pol (stiff, mu large) and Robertson verify solutions against analytic or high-accuracy reference values within tolerance.
- [ ] Each itask mode (1 to 5) and each Jacobian kind is exercised at least once.
- [ ] A test confirms that two solvers in two threads run independently.

**Files to Touch:**
- `tests/analytic.rs`

**Verification Step:**

Run:
```bash
cargo test --test analytic
```

Expected:
All analytic tests pass; exit code 0.

**Notes:**

Mirror the problems in the fork's `test/test_lsoda.f90` and `test/test_threadsafe.f90` where possible.
