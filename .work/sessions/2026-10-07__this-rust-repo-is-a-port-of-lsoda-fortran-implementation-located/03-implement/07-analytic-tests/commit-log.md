# Ticket 07: analytic-tests

**Status:** ✅ Done

## Local Test Result

`cargo test --test analytic`: 10 passed, 0 failed, no warnings.

## Review Notes

Implemented by a sonnet subagent (medium effort); reviewed in this session by re-running the tests and checking coverage in `tests/analytic.rs`: itask 1 to 5 each have a test, all four Jacobian kinds (DenseFd, DenseUser, BandedFd, BandedUser) are used, Van der Pol and Robertson cover the stiff case, and a two-thread test checks independence. Reference values for Robertson/Van der Pol were not independently recomputed. The subagent's report was a single line; I did not read the test bodies in full. No changes after review.

## Blockers / Challenges

None.

## Commits

- 02ded70 test: analytic-solution suite across itask modes and Jacobian kinds (ticket 07)

## Time Spent

2m (ticket-start.js to ticket-complete.js)
