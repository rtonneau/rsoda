# Ticket 03: leaf-routines

**Status:** ✅ Done

## Local Test Result

`cargo test coefficients interpolate norms solsy` is rejected by cargo (one filter allowed). Ran the ticket's fallback, `cargo test`: 40 passed, 0 failed, no warnings. The subagent also ran each filter alone (coefficients 6, interpolate 5, norms 6, solsy 6).

## Review Notes

Implemented by an opus subagent (high effort); reviewed in this session by re-running the tests and confirming in `odepack.f` that LSODA uses DMNORM (no DVNORM). Not re-read line by line against the Fortran. Open points carried forward: (1) DSOLSY uses the ported LINPACK `dgesl`/`dgbsl`, whereas the Fortran fork calls LAPACK `dgetrs`/`dgbtrs`, so rounding may differ (within the agreed ~1e-12 criterion); (2) INTDY scaling uses `powi`, unchecked against Fortran output until Golden files exist; (3) the solver state struct is deferred to ticket 05, so these routines take arguments plus `StepData`. No changes after review.

## Blockers / Challenges

The verification command in the ticket used several cargo filters, which cargo rejects. CFODE computes its tables, so tests compare against hand-written values and an exact-fraction computation (4 ulps).

## Commits

- 0619603 feat: norms, method coefficients, interpolation and corrector solve (ticket 03)

## Time Spent

6m (ticket-start.js to ticket-complete.js)
