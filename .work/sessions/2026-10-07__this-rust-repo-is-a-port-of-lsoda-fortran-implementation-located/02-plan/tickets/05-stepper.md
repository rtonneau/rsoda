# Ticket 05: stepper

**Model:** opus
**Effort:** xhigh

**Acceptance Criteria:**
- [ ] STODA equivalent performs one step with corrector iteration, error test, step-size and order selection, and Nordsieck array rescaling.
- [ ] Stiffness detection and Method switching between Adams and BDF follow DSTODA exactly, including the counters (nst, nfe, nje, nqu, nslast) and the method-used record.
- [ ] Tests integrate a nonstiff and a stiff problem step by step and observe a switch to BDF.

**Files to Touch:**
- `src/stoda.rs`
- `src/state.rs`
- `src/lib.rs`

**Verification Step:**

Run:
```bash
cargo test stoda
```

Expected:
Tests pass; exit code 0, including the stiff problem switching Method.

**Notes:**

Source: DSTODA in `odepack_sub1.f` (DSTODA and DRCHECK are the only parts specific to LSODA vs LSODE). Port constants and control flow verbatim; replace the GOTO structure with explicit loops/state enums without changing the order of operations. This is the highest-risk ticket.
