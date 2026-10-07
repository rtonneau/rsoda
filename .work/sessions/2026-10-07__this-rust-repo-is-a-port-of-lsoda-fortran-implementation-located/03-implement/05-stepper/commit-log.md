# Ticket 05: stepper

**Status:** ✅ Done

## Local Test Result

`cargo test stoda`: 16 passed. Full `cargo test`: 66 passed, 0 failed, no warnings.

## Review Notes

Implemented by an opus subagent (xhigh effort); reviewed in this session by re-running the suite and checking the Robertson figures against the DLSODA prologue in `odepack.f` (lines 213-214: 361 steps, 693 f-evals, 64 Jacobians, method 2, switch at t=6.0092e-03). The Rust run reproduces all of them, for all four Jacobian kinds on the step and Jacobian counts. The 5-figure values match too. DSTODA was not re-read line by line; this prologue match is the main fidelity evidence until Golden files exist (ticket 08). Notes for ticket 06: the driver maintains `nslast` and `tsw` and calls `State::record_method_switch()`; on a callback error `stoda` returns mid-step and the integration cannot continue; `hmax = f64::MAX` must become `hmxi = 0`. No changes after review.

## Blockers / Challenges

None. The GO TO structure became a loop over a private `Label` enum, one variant per Fortran label. Existing files are not formatted with default rustfmt, so the style was matched, not reformatted.

## Commits

- c4a4d9d feat: core stepper with order selection and Adams/BDF switching (ticket 05)

## Time Spent

22m (ticket-start.js to ticket-complete.js)
