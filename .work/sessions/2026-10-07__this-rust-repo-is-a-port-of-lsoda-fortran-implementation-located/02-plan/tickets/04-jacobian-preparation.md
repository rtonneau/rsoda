# Ticket 04: jacobian-preparation

**Model:** opus
**Effort:** high

**Acceptance Criteria:**
- [ ] PRJA equivalent builds and factors `I - h*el0*J` for dense and banded storage.
- [ ] User-supplied Jacobian and finite-difference Jacobian paths both work, with the Fortran increment rules.
- [ ] Callback errors (`Err`) propagate; singular matrix is reported as the Fortran does.
- [ ] Tests check the factored matrix on a small linear system for all four Jacobian kinds.

**Files to Touch:**
- `src/prja.rs`
- `src/lib.rs`

**Verification Step:**

Run:
```bash
cargo test prja
```

Expected:
Tests pass; exit code 0.

**Notes:**

Source: DPRJA in `odepack_sub1.f`. Mind the banded finite-difference grouping (`mba`, `ngp`) and the evaluation-count counters (nfe, nje).
