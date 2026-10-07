# Ticket 06: driver

**Model:** opus
**Effort:** xhigh

**Acceptance Criteria:**
- [ ] `Lsoda::integrate(tout)`, `step()`, and tcrit-bounded variants reproduce DLSODA itask 1-5, with re-entry covering istate 2 and 3 behavior via setters.
- [ ] First-step size selection (when h0 is not given) and all mxstep/hmin/hmax rules match DLSODA.
- [ ] Every negative istate is returned as the matching `LsodaError`; `mxhnil`-style warnings and Method-switch events go through opt-in hooks; nothing is printed.
- [ ] A smoke test integrates y' = -y over [0, 1] with each Jacobian kind.

**Files to Touch:**
- `src/lsoda.rs`
- `src/lib.rs`

**Verification Step:**

Run:
```bash
cargo test lsoda
```

Expected:
Smoke tests pass for all four Jacobian kinds; exit code 0.

**Notes:**

Source: DLSODA in `odepack.f` and `odepack_mod.f90` `lsoda_integrate`. The Rust `Lsoda` object replaces the fork's `lsoda_class` and its common-data derived type.
