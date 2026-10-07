# Ticket 04: jacobian-preparation

**Status:** ✅ Done

## Local Test Result

`cargo test prja`: 11 passed. Full `cargo test`: 50 passed, 0 failed, no warnings.

## Review Notes

Implemented by an opus subagent (high effort); reviewed in this session by re-running the suite and checking the finite-difference increment formula against `odepack_sub1.f` (R0 = 1000*|H|*UROUND*N*DMNORM(SAVF,EWT)). The rest of DPRJA was not re-read line by line; tests cover all four Jacobian kinds, banded grouping, error propagation and singular detection. Choices to carry forward: the banded user Jacobian is filled in a temporary buffer (ticket 01's `jac` contract uses leading dimension ml+mu+1, DPRJA passes 2*ml+mu+1), which allocates per call; on a callback error `y` is restored, unlike the Fortran; LINPACK ports replace the fork's LAPACK dgetrf/dgbtrf. No changes after review.

## Blockers / Challenges

None. Clippy shows only needless_range_loop lints in test code.

## Commits

- f53c14b feat: Jacobian preparation for dense and banded kinds (ticket 04)

## Time Spent

4m (ticket-start.js to ticket-complete.js)
