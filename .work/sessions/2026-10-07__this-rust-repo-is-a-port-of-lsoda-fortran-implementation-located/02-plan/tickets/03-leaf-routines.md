# Ticket 03: leaf-routines

**Model:** opus
**Effort:** high

**Acceptance Criteria:**
- [ ] Weighted max-norm routines (VMNORM, FNORM, BNORM equivalents) and the error-weight setup (EWSET) are ported.
- [ ] Method coefficients (CFODE) for Adams and BDF match the Fortran tables exactly.
- [ ] Interpolation (INTDY) and the corrector solve (SOLSY) are ported with their error returns.
- [ ] Unit tests check coefficients against values transcribed from the Fortran source and interpolation against polynomials.

**Files to Touch:**
- `src/norms.rs`
- `src/coefficients.rs`
- `src/interpolate.rs`
- `src/solsy.rs`
- `src/lib.rs`

**Verification Step:**

Run:
```bash
cargo test coefficients interpolate norms solsy
```

Expected:
Tests pass; exit code 0. (If cargo rejects multiple filters, run `cargo test` and confirm these modules' tests pass.)

**Notes:**

Source: `odepack_sub1.f` and `odepack.f` (DCFODE, DINTDY, DEWSET, DVNORM, DFNORM, DBNORM, DSOLSY). Keep the solver state in one struct (the Rust analogue of the fork's common-data derived type).
