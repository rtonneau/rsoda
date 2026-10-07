# Ticket 02: linear-algebra

**Model:** sonnet
**Effort:** high

**Acceptance Criteria:**
- [ ] Dense LU with partial pivoting (factor and solve) matches the semantics of DGEFA/DGESL, including the singularity indicator.
- [ ] Banded LU (factor and solve) matches DGBFA/DGBSL for given `ml`, `mu`.
- [ ] Unit tests solve known systems and verify singular detection for both storage layouts.

**Files to Touch:**
- `src/linalg.rs`
- `src/lib.rs`

**Verification Step:**

Run:
```bash
cargo test linalg
```

Expected:
All linalg tests pass; exit code 0.

**Notes:**

Use column-major storage as in Fortran so later tickets index the same way. Port the LINPACK algorithm directly (not LAPACK) for predictable operation order.
